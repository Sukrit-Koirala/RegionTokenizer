#!/usr/bin/env python3
"""
train_region_pairwise_reranker_v2.py — Region Pairwise Reranker V2

Changes from V1:
  - Gated delta: actual_delta = delta_scale * sigmoid(gate) * tanh(raw_delta)
  - No alpha parameter; delta bounded by gate * delta_scale
  - Mixed batches: target (confuser) rows + noharm (base-correct) rows
  - No-harm losses: noharm CE, margin preservation, gate sparsity
  - Checkpoint metric: noharm_adjusted_score = ctg - 2*caw + 0.25*pwr

V1 reference (same_region_pair_v1):
  pairwise_win_rate_refined = 0.1379
  changed_to_gold_rate      = 0.1018
  full_vocab_gain           = -0.0050
  top1_acc_drop             = -0.0101
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_RANK_EMB_DIM = 8

# Checkpoint metric: higher is better
_NOHARM_SCORE_KEY = "noharm_adjusted_score"

_SUBSET_ORDER_V2 = [
    "all",
    "covered",
    "base_correct_covered",
    "base_wrong_covered",
    "target_confuser",
    "same_region_confuser",
    "same_superregion_confuser",
    "base_miss",
]


# ─────────────────────────────────────────────────────────────────────────────
# Backbone loading (identical to V1)
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(ckpt_path: str, device: torch.device):
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        ckpt_path, device)
    tok_w = backbone.token_emb.weight.detach().float()
    del backbone
    return tok_w, d_model, vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# Map loading (identical to V1)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_map(raw):
    if isinstance(raw, list):
        return {i: v for i, v in enumerate(raw) if v is not None}
    elif isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    raise ValueError(f"Expected list or dict, got {type(raw)}")


def load_maps(t2r_path: str, super_path: Optional[str]):
    with open(t2r_path) as f:
        t2r = _parse_map(json.load(f))
    max_region = max(t2r.values()) if t2r else 0
    unk_region = int(max_region) + 1

    if super_path and os.path.isfile(super_path):
        with open(super_path) as f:
            r2s = _parse_map(json.load(f))
        max_super = max(r2s.values()) if r2s else 0
        unk_super = int(max_super) + 1
        sr_enabled = True
    else:
        r2s = {}
        unk_super = 0
        sr_enabled = False

    print(f"  maps: n_regions={unk_region}  unk_region={unk_region}  "
          f"n_supers={unk_super if sr_enabled else 0}  sr_enabled={sr_enabled}")
    return t2r, r2s, unk_region, unk_super, sr_enabled


def build_tok_arr(t2r, unk_region: int, vocab_size: int) -> np.ndarray:
    arr = np.full(vocab_size, unk_region, dtype=np.int32)
    for tok, reg in t2r.items():
        if 0 <= tok < vocab_size:
            arr[tok] = int(reg)
    return arr


def build_reg_arr(r2s, unk_super: int, unk_region: int) -> np.ndarray:
    if not r2s:
        return np.full(unk_region + 1, unk_super, dtype=np.int32)
    arr = np.full(unk_region + 1, unk_super, dtype=np.int32)
    for reg, sup in r2s.items():
        if 0 <= reg <= unk_region:
            arr[reg] = int(sup)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Shard loading (identical to V1)
# ─────────────────────────────────────────────────────────────────────────────

def _get_field(shard, *names, required=True):
    for n in names:
        if n in shard:
            return shard[n]
    if required:
        raise KeyError(f"Shard missing all aliases {names}. Have: {list(shard.keys())}")
    return None


def load_shards(shard_dir: str, top_k: int, split_name: str,
                max_rows: Optional[int] = None, load_ids: bool = False):
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")

    all_hp, all_topk, all_lgt, all_gold, all_ids, all_rid = [], [], [], [], [], []
    total = 0

    for sp in shards:
        if max_rows is not None and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)

        hp   = _get_field(sh, "h_prime", "h_ctx").float()
        topk = _get_field(sh, "base_topk_ids", "base_topk").long()
        lgt  = _get_field(sh, "base_topk_lgt", "base_topk_logits", required=True).float()
        gold = _get_field(sh, "gold_token", "gold").long()
        rid  = _get_field(sh, "row_id", required=False)
        ids  = _get_field(sh, "input_ids", required=False) if load_ids else None

        B, K = topk.shape
        if K < top_k:
            topk = torch.cat([topk, torch.zeros(B, top_k - K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, top_k - K), float("nan"))], 1)
        elif K > top_k:
            topk, lgt = topk[:, :top_k], lgt[:, :top_k]

        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            hp, topk, lgt, gold = hp[:keep], topk[:keep], lgt[:keep], gold[:keep]
            if rid is not None:
                rid = rid[:keep] if isinstance(rid, list) else rid[:keep].tolist()
            if ids is not None:
                ids = ids[:keep]
            B = keep

        all_hp.append(hp); all_topk.append(topk); all_lgt.append(lgt); all_gold.append(gold)
        if rid is not None:
            all_rid.extend(rid if isinstance(rid, list) else rid.tolist())
        if ids is not None:
            all_ids.append(ids)
        total += B

    data = {
        "h_prime":  torch.cat(all_hp, 0),
        "topk_ids": torch.cat(all_topk, 0),
        "topk_lgt": torch.cat(all_lgt, 0),
        "gold":     torch.cat(all_gold, 0),
        "rid":      all_rid if all_rid else None,
        "ids":      torch.cat(all_ids, 0) if all_ids else None,
    }
    N = data["h_prime"].shape[0]
    print(f"  {split_name}: {N:,} rows  d={data['h_prime'].shape[1]}  K={top_k}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Train pool construction (replaces V1 apply_filter)
# ─────────────────────────────────────────────────────────────────────────────

def build_train_pools(data, tok_arr_t, reg_arr_t,
                      unk_region: int, unk_super: int, sr_enabled: bool):
    """Return (target_idx, noharm_idx, stats_dict) — all CPU tensors."""
    gold = data["gold"]
    topk = data["topk_ids"]

    # All on CPU — keep lookup arrays on CPU for this function
    tok_cpu = tok_arr_t.cpu()
    reg_cpu = reg_arr_t.cpu()

    vs = tok_cpu.shape[0]
    covered   = (topk == gold.unsqueeze(1)).any(1)
    base_wrong = topk[:, 0] != gold

    gold_safe = gold.clamp(0, vs - 1)
    top1_safe = topk[:, 0].clamp(0, vs - 1)
    gold_reg  = tok_cpu[gold_safe]
    top1_reg  = tok_cpu[top1_safe]
    same_reg  = (gold_reg == top1_reg) & (gold_reg != unk_region) & (top1_reg != unk_region)

    if sr_enabled:
        rlen = reg_cpu.shape[0] - 1
        gold_sup = reg_cpu[gold_reg.clamp(0, rlen)]
        top1_sup = reg_cpu[top1_reg.clamp(0, rlen)]
        same_sup = (gold_sup == top1_sup) & (gold_sup != unk_super) & (top1_sup != unk_super)
    else:
        same_sup = torch.zeros(len(gold), dtype=torch.bool)

    # target: confuser rows (covered + base_wrong + same_region_or_super)
    target_mask = covered & base_wrong & (same_reg | same_sup)
    # noharm: base-correct covered rows
    noharm_mask = covered & ~base_wrong
    # other: covered + base_wrong but not same region/super
    other_mask  = covered & base_wrong & ~(same_reg | same_sup)

    target_idx = target_mask.nonzero(as_tuple=False).squeeze(1)
    noharm_idx = noharm_mask.nonzero(as_tuple=False).squeeze(1)
    other_idx  = other_mask.nonzero(as_tuple=False).squeeze(1)

    N = len(gold)
    stats = {
        "total_rows":        N,
        "target_rows":       int(len(target_idx)),
        "noharm_rows":       int(len(noharm_idx)),
        "other_rows":        int(len(other_idx)),
        "target_rate":       len(target_idx) / N if N > 0 else 0.0,
        "noharm_rate":       len(noharm_idx) / N if N > 0 else 0.0,
        "covered_rate":      float(covered.float().mean()),
        "base_wrong_rate":   float(base_wrong.float().mean()),
        "same_region_rate":  float(same_reg.float().mean()),
        "same_super_rate":   float(same_sup.float().mean()) if sr_enabled else 0.0,
        "gold_unmapped_rate":float((tok_cpu[gold_safe] == unk_region).float().mean()),
    }
    return target_idx, noharm_idx, other_idx, stats


# ─────────────────────────────────────────────────────────────────────────────
# Model V2: Gated delta
# ─────────────────────────────────────────────────────────────────────────────

class RegionPairwiseRerankerV2(nn.Module):
    def __init__(self, token_emb_weight: torch.Tensor,
                 tok_arr: np.ndarray, reg_arr: np.ndarray,
                 d_model: int, n_regions: int, n_supers: int,
                 sr_enabled: bool,
                 resolver_dim: int = 256, resolver_layers: int = 2,
                 resolver_heads: int = 4,
                 region_emb_dim: int = 64, super_emb_dim: int = 32,
                 delta_scale: float = 0.25, gate_bias_init: float = -4.0,
                 top_k: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d_model      = d_model
        self.sr_enabled   = sr_enabled
        self.delta_scale  = delta_scale
        self.top_k        = top_k
        self.resolver_dim = resolver_dim

        self.register_buffer("token_emb_weight",
                             token_emb_weight.detach().float())
        self.register_buffer("tok_arr",
                             torch.from_numpy(tok_arr.astype(np.int32)).long())
        self.register_buffer("reg_arr",
                             torch.from_numpy(reg_arr.astype(np.int32)).long())

        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim)
        if sr_enabled:
            self.super_emb = nn.Embedding(n_supers + 2, super_emb_dim)
        self.rank_emb = nn.Embedding(top_k + 1, _RANK_EMB_DIM)

        feat_dim = d_model + region_emb_dim + _RANK_EMB_DIM + 4
        if sr_enabled:
            feat_dim += super_emb_dim + 1
        self.cand_proj = nn.Linear(feat_dim, resolver_dim)

        self.ctx_proj = nn.Sequential(
            nn.Linear(d_model, resolver_dim),
            nn.GELU(),
            nn.Linear(resolver_dim, resolver_dim),
        )
        self.pos_emb = nn.Embedding(top_k + 1, resolver_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resolver_dim, nhead=resolver_heads,
            dim_feedforward=resolver_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=resolver_layers)

        # Delta head: zero-init weights and bias → tanh(0) = 0 → actual_delta = 0
        self.raw_delta_head = nn.Linear(resolver_dim, 1, bias=True)
        nn.init.zeros_(self.raw_delta_head.weight)
        nn.init.zeros_(self.raw_delta_head.bias)

        # Gate head: zero-init weights, bias = gate_bias_init
        # sigmoid(gate_bias_init=-4) ≈ 0.018 → near no-op at init
        self.gate_head = nn.Linear(resolver_dim, 1, bias=True)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, gate_bias_init)

    def forward(self, h_prime, candidate_ids, base_logits, cand_regs, cand_sups=None):
        """
        Returns (refined_logits [B,K], actual_delta [B,K], gate [B,K]).
        actual_delta = delta_scale * gate * tanh(raw_delta)
        """
        B, K = candidate_ids.shape
        vs = self.token_emb_weight.shape[0]
        tok_embs  = self.token_emb_weight[candidate_ids.clamp(0, vs - 1)]  # [B,K,d]
        reg_embs  = self.region_emb(cand_regs)
        ranks     = torch.arange(K, device=h_prime.device).unsqueeze(0).expand(B, -1)
        rank_embs = self.rank_emb(ranks)
        top1_reg  = cand_regs[:, 0:1]
        same_reg  = (cand_regs == top1_reg).float()
        logit_gap = base_logits - base_logits[:, 0:1]
        hp_dot    = torch.bmm(tok_embs, h_prime.unsqueeze(2)).squeeze(2)

        parts = [tok_embs, reg_embs, rank_embs,
                 base_logits.unsqueeze(2), logit_gap.unsqueeze(2),
                 same_reg.unsqueeze(2), hp_dot.unsqueeze(2)]
        if self.sr_enabled and cand_sups is not None:
            sup_embs  = self.super_emb(cand_sups)
            same_sup  = (cand_sups == cand_sups[:, 0:1]).float()
            parts.extend([sup_embs, same_sup.unsqueeze(2)])

        cand_emb = self.cand_proj(torch.cat(parts, 2))
        ctx      = self.ctx_proj(h_prime).unsqueeze(1)
        seq      = torch.cat([ctx, cand_emb], 1)
        pos_ids  = torch.arange(K + 1, device=h_prime.device)
        seq      = seq + self.pos_emb(pos_ids).unsqueeze(0)
        out      = self.transformer(seq)[:, 1:]                   # [B,K,dim]

        raw_delta  = self.raw_delta_head(out).squeeze(2)          # [B,K]
        gate_logit = self.gate_head(out).squeeze(2)               # [B,K]
        gate       = torch.sigmoid(gate_logit)                    # [B,K] in (0,1)
        actual_delta = self.delta_scale * gate * torch.tanh(raw_delta)

        refined_logits = base_logits + actual_delta
        return refined_logits, actual_delta, gate


# ─────────────────────────────────────────────────────────────────────────────
# Region/super lookup (same as V1)
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_regs(token_ids, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled):
    vs   = tok_arr_t.shape[0]
    safe = token_ids.clamp(0, vs - 1)
    regs = tok_arr_t[safe]
    if sr_enabled:
        rlen = reg_arr_t.shape[0] - 1
        sups = reg_arr_t[regs.clamp(0, rlen)]
    else:
        sups = None
    return regs, sups


# ─────────────────────────────────────────────────────────────────────────────
# Loss V2
# ─────────────────────────────────────────────────────────────────────────────

def _compute_losses_v2(refined_logits, actual_delta, gate, base_logits,
                        gold_idx, covered, row_types,
                        cand_regs, cand_sups, gold_regs, gold_sups,
                        unk_region, unk_super, sr_enabled, args, device):
    B, K = refined_logits.shape
    arange    = torch.arange(B, device=device)
    gold_safe = gold_idx.clamp(0, K - 1)
    target_m  = row_types == 0                              # [B] target rows
    noharm_m  = row_types == 1                              # [B] noharm rows

    # ── 1. Pairwise (target rows, covered, base_wrong) ───────────────────────
    pair_m = target_m & covered & (gold_idx != 0)
    if pair_m.any():
        z_g = refined_logits[pair_m, gold_safe[pair_m]]
        z_b = refined_logits[pair_m, 0]
        L_pair = F.softplus(-(z_g - z_b)).mean()
    else:
        L_pair = torch.tensor(0.0, device=device)

    # ── 2. Multi-confuser (target rows) ──────────────────────────────────────
    not_gold = torch.ones(B, K, dtype=torch.bool, device=device)
    not_gold.scatter_(1, gold_safe.unsqueeze(1), False)

    sr_cand = (cand_regs == gold_regs.unsqueeze(1)) & (gold_regs.unsqueeze(1) != unk_region)
    if sr_enabled and cand_sups is not None:
        ss_cand = (cand_sups == gold_sups.unsqueeze(1)) & (gold_sups.unsqueeze(1) != unk_super)
        conf_mask = not_gold & (sr_cand | ss_cand)
    else:
        conf_mask = not_gold & sr_cand

    multi_loss = torch.tensor(0.0, device=device)
    n_multi = 0
    for i in range(B):
        if not (target_m[i] and covered[i]):
            continue
        ci = conf_mask[i].nonzero(as_tuple=False).squeeze(1)[:args.num_confusers]
        if len(ci) == 0:
            continue
        z_g = refined_logits[i, gold_safe[i]]
        z_c = refined_logits[i, ci]
        multi_loss = multi_loss + F.softplus(-(z_g - z_c)).mean()
        n_multi += 1
    if n_multi > 0:
        multi_loss = multi_loss / n_multi

    # ── 3. Candidate CE (target rows with gold covered) ───────────────────────
    target_cov = target_m & covered
    if target_cov.any():
        log_p = F.log_softmax(refined_logits, 1)
        L_ce  = (-log_p[arange, gold_safe])[target_cov].mean()
    else:
        L_ce  = torch.tensor(0.0, device=device)

    # ── 4. KL to base (all rows) ──────────────────────────────────────────────
    p_b = F.softmax(base_logits, 1)
    p_r = F.softmax(refined_logits, 1)
    L_kl = (p_b * ((p_b + 1e-10).log() - (p_r + 1e-10).log())).sum(1).mean()

    # ── 5. Delta penalty (all rows, use actual_delta) ─────────────────────────
    L_delta = (actual_delta ** 2).mean()

    # ── 6. No-harm CE (noharm rows with gold covered) ─────────────────────────
    noharm_cov = noharm_m & covered
    if noharm_cov.any():
        log_p = F.log_softmax(refined_logits, 1)
        L_noharm_ce = (-log_p[arange, gold_safe])[noharm_cov].mean()
    else:
        L_noharm_ce = torch.tensor(0.0, device=device)

    # ── 7. No-harm margin preservation (noharm rows with gold covered) ────────
    if noharm_cov.any():
        # Mask out gold position to find max wrong candidate logit
        base_for_max = base_logits.clone()
        base_for_max[arange, gold_safe] = -1e9
        max_wrong_base = base_for_max.max(1).values                # [B]

        ref_for_max = refined_logits.clone()
        ref_for_max[arange, gold_safe] = -1e9
        max_wrong_ref = ref_for_max.max(1).values

        m_base = (base_logits[arange, gold_safe] - max_wrong_base).clamp(max=5.0)
        m_ref  = refined_logits[arange, gold_safe] - max_wrong_ref
        L_noharm_margin = F.relu(m_base - m_ref)[noharm_cov].mean()
    else:
        L_noharm_margin = torch.tensor(0.0, device=device)

    # ── 8. Gate sparsity ──────────────────────────────────────────────────────
    L_gate = gate.mean()

    total = (args.lambda_pair         * L_pair
           + args.lambda_multi        * multi_loss
           + args.lambda_ce           * L_ce
           + args.lambda_kl           * L_kl
           + args.lambda_delta        * L_delta
           + args.lambda_noharm_ce    * L_noharm_ce
           + args.lambda_noharm_margin* L_noharm_margin
           + args.lambda_gate         * L_gate)

    return total, {
        "pair":            L_pair.item(),
        "multi":           multi_loss.item(),
        "ce":              L_ce.item(),
        "kl":              L_kl.item(),
        "delta":           L_delta.item(),
        "noharm_ce":       L_noharm_ce.item(),
        "noharm_margin":   L_noharm_margin.item(),
        "gate_sparsity":   L_gate.item(),
        "total":           total.item(),
        "n_multi_rows":    n_multi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Identity check
# ─────────────────────────────────────────────────────────────────────────────

def run_identity_check(model, val_data, tok_arr_t, reg_arr_t,
                       unk_region, unk_super, sr_enabled, args, device, output_dir):
    model.eval()
    n   = min(64, val_data["h_prime"].shape[0])
    hp  = val_data["h_prime"][:n].to(device)
    topk= val_data["topk_ids"][:n].to(device)
    lgt = val_data["topk_lgt"][:n].to(device)
    gold= val_data["gold"][:n].to(device)

    cr, cs = _lookup_regs(topk, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled)
    with torch.no_grad():
        ref, actual_delta, gate = model(hp, topk, lgt, cr, cs)

    delta_max  = float(actual_delta.abs().max())
    gate_mean  = float(gate.mean())
    gate_max   = float(gate.max())
    ce_diff    = float((ref - lgt).abs().max())
    top1_base  = topk[:, 0]
    top1_ref   = topk[torch.arange(n), ref.argmax(1)]
    acc_base   = (top1_base == gold).float().mean().item()
    acc_ref    = (top1_ref  == gold).float().mean().item()

    passed = (delta_max < 1e-6 and ce_diff < 1e-5 and abs(acc_base - acc_ref) < 1e-6)
    result = {
        "raw_delta_head_norm": float(model.raw_delta_head.weight.norm()),
        "gate_mean":           gate_mean,
        "gate_max":            gate_max,
        "actual_delta_abs_max":delta_max,
        "candidate_ce_diff_max": ce_diff,
        "top1_acc_base":       acc_base,
        "top1_acc_refined":    acc_ref,
        "passed":              passed,
    }

    with open(os.path.join(output_dir, "identity_check.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[identity] delta_max={delta_max:.2e}  gate_mean={gate_mean:.4f}  "
          f"gate_max={gate_max:.4f}  ce_diff={ce_diff:.2e}  passed={passed}")

    if not passed:
        raise RuntimeError(
            "Identity check FAILED. actual_delta not zero at init. Aborting.")

    model.train()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(arr):
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) > 0 else float("nan")


def _safe_median(arr):
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) > 0 else float("nan")


def _safe_rate(arr_bool, total):
    if total == 0:
        return float("nan")
    return float(np.asarray(arr_bool, dtype=float).sum()) / total


def _subset_metrics_v2(mask, row):
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}

    def M(key):
        return np.asarray(row[key])[mask]

    cov  = M("covered")
    bw   = M("base_wrong")
    ctg  = M("changed_to_gold")
    caw  = M("changed_away")
    rw   = M("refined_wrong")
    gri  = M("gold_rank_improved")
    grw  = M("gold_rank_worsened")
    grb  = M("gold_rank_base_1b")
    grr  = M("gold_rank_ref_1b")
    pmb  = M("pair_margin_base")
    pmr  = M("pair_margin_ref")
    ceb  = M("ce_base")
    cer  = M("ce_ref")
    pwb  = M("pairwise_win_base")
    pwr  = M("pairwise_win_ref")
    dab  = M("actual_delta_abs")
    gm   = M("mean_gate")
    gmax = M("max_gate")
    bgm  = M("base_gold_margin")
    rgm  = M("ref_gold_margin")

    bwc  = bw & cov
    n_bwc = int(bwc.sum())

    return {
        "n":                          n,
        "covered_rate":               _safe_rate(cov, n),
        "base_wrong_rate":            _safe_rate(bw, n),
        "candidate_ce_base":          _safe_mean(ceb[cov]),
        "candidate_ce_refined":       _safe_mean(cer[cov]),
        "candidate_ce_gain":          _safe_mean((ceb - cer)[cov]),
        "top1_acc_base":              _safe_rate(~bw, n),
        "top1_acc_refined":           _safe_rate(~rw, n),
        "top1_acc_gain":              _safe_rate(~rw, n) - _safe_rate(~bw, n),
        "changed_to_gold_rate":       _safe_rate(ctg[bwc], n_bwc) if n_bwc > 0 else float("nan"),
        "changed_away_rate":          _safe_rate(caw[~bw], int((~bw).sum())) if (~bw).any() else float("nan"),
        "gold_rank_improved_rate":    _safe_rate(gri[cov], int(cov.sum())),
        "gold_rank_worsened_rate":    _safe_rate(grw[cov], int(cov.sum())),
        "mean_gold_rank_base":        _safe_mean(grb[cov]),
        "mean_gold_rank_refined":     _safe_mean(grr[cov]),
        "median_gold_rank_base":      _safe_median(grb[cov]),
        "median_gold_rank_refined":   _safe_median(grr[cov]),
        "pairwise_win_rate_base":     _safe_rate(pwb[bwc], n_bwc) if n_bwc > 0 else float("nan"),
        "pairwise_win_rate_refined":  _safe_rate(pwr[bwc], n_bwc) if n_bwc > 0 else float("nan"),
        "mean_pair_margin_base":      _safe_mean(pmb[bwc]),
        "mean_pair_margin_refined":   _safe_mean(pmr[bwc]),
        "mean_pair_margin_gain":      _safe_mean((pmr - pmb)[bwc]),
        "mean_actual_delta_abs":      _safe_mean(dab),
        "max_actual_delta_abs":       float(np.nanmax(dab)) if n > 0 else float("nan"),
        "mean_gate":                  _safe_mean(gm),
        "max_gate":                   float(np.nanmax(gmax)) if n > 0 else float("nan"),
        "mean_base_gold_margin":      _safe_mean(bgm[cov]),
        "mean_ref_gold_margin":       _safe_mean(rgm[cov]),
        "mean_margin_change":         _safe_mean((rgm - bgm)[cov]),
        "margin_eroded_rate":         _safe_rate((rgm < bgm)[cov], int(cov.sum())),
    }


_SUBSET_DEFS_V2 = {
    "all":                   lambda d: np.ones(d["N"], dtype=bool),
    "covered":               lambda d: d["covered"],
    "base_correct_covered":  lambda d: d["covered"] & ~d["base_wrong"],
    "base_wrong_covered":    lambda d: d["covered"] & d["base_wrong"],
    "target_confuser":       lambda d: d["covered"] & d["base_wrong"] & (d["same_reg"] | d["same_sup"]),
    "same_region_confuser":  lambda d: d["covered"] & d["base_wrong"] & d["same_reg"],
    "same_superregion_confuser": lambda d: d["covered"] & d["base_wrong"] & d["same_sup"],
    "base_miss":             lambda d: ~d["covered"],
}


def evaluate(model, val_data, tok_arr_t, reg_arr_t,
             unk_region, unk_super, sr_enabled,
             args, device, step, output_dir):
    model.eval()
    N, K = val_data["h_prime"].shape[0], val_data["topk_ids"].shape[1]
    BSZ  = args.eval_batch_size

    row = {k: np.zeros(N, dtype=np.float32) for k in [
        "actual_delta_abs", "mean_gate", "max_gate",
        "gold_rank_base_1b", "gold_rank_ref_1b",
        "pair_margin_base", "pair_margin_ref",
        "ce_base", "ce_ref",
        "base_gold_margin", "ref_gold_margin",
    ]}
    for k in ["covered", "base_wrong", "same_reg", "same_sup",
              "changed_to_gold", "changed_away", "refined_wrong",
              "gold_rank_improved", "gold_rank_worsened",
              "pairwise_win_base", "pairwise_win_ref"]:
        row[k] = np.zeros(N, dtype=bool)
    for k in ["gold_rank_base_1b", "gold_rank_ref_1b",
              "pair_margin_base", "pair_margin_ref",
              "ce_base", "ce_ref", "base_gold_margin", "ref_gold_margin"]:
        row[k] = np.full(N, float("nan"), dtype=np.float32)

    # For example reports
    row["gold_tok"]        = np.zeros(N, dtype=np.int64)
    row["base_top1_tok"]   = np.zeros(N, dtype=np.int64)
    row["ref_top1_tok"]    = np.zeros(N, dtype=np.int64)
    row["gold_idx"]        = np.zeros(N, dtype=np.int32)
    row["delta_top10"]     = np.zeros((N, min(10, K)), dtype=np.float32)
    row["gate_top10"]      = np.zeros((N, min(10, K)), dtype=np.float32)

    def np_(t):
        return t.cpu().numpy()

    with torch.no_grad():
        for s in range(0, N, BSZ):
            e    = min(s + BSZ, N)
            b    = e - s
            ar_b = torch.arange(b, device=device)

            hp   = val_data["h_prime"][s:e].to(device)
            topk = val_data["topk_ids"][s:e].to(device)
            lgt  = val_data["topk_lgt"][s:e].to(device)
            gold = val_data["gold"][s:e].to(device)

            cr, cs = _lookup_regs(topk, tok_arr_t, reg_arr_t,
                                   unk_region, unk_super, sr_enabled)
            ref, act_delta, gate_b = model(hp, topk, lgt, cr, cs)

            # Gold idx
            gold_in = (topk == gold.unsqueeze(1))
            cov_b   = gold_in.any(1)
            gidx_b  = gold_in.long().argmax(1)
            gsafe   = gidx_b.clamp(0, K - 1)

            base_t1 = topk[:, 0]
            ref_t1i = ref.argmax(1)
            ref_t1  = topk[ar_b, ref_t1i]

            # Gold rank in refined
            n_higher = (ref > ref[ar_b, gsafe].unsqueeze(1)).sum(1)
            grb_b    = torch.where(cov_b, gidx_b.long(), torch.tensor(-1, device=device))
            grr_b    = torch.where(cov_b, n_higher.long(), torch.tensor(-1, device=device))

            # Margins
            bm_for_max = lgt.clone()
            bm_for_max[ar_b, gsafe] = -1e9
            max_wb = bm_for_max.max(1).values
            rm_for_max = ref.clone()
            rm_for_max[ar_b, gsafe] = -1e9
            max_wr = rm_for_max.max(1).values

            base_gm = lgt[ar_b, gsafe] - max_wb
            ref_gm  = ref[ar_b, gsafe]  - max_wr

            pmb_b = lgt[ar_b, gsafe] - lgt[:, 0]
            pmr_b = ref[ar_b, gsafe] - ref[:, 0]

            log_pb = F.log_softmax(lgt, 1)
            log_pr = F.log_softmax(ref, 1)
            ceb_b  = -log_pb[ar_b, gsafe]
            cer_b  = -log_pr[ar_b, gsafe]

            bw_b  = base_t1 != gold
            ctg_b = bw_b & cov_b & (ref_t1 == gold)
            caw_b = ~bw_b & (ref_t1 != gold)
            rw_b  = ref_t1 != gold
            gri_b = cov_b & (grr_b < grb_b) & (grr_b >= 0)
            grw_b = cov_b & (grr_b > grb_b)
            pwb_b = bw_b & cov_b & (pmb_b > 0)
            pwr_b = bw_b & cov_b & (pmr_b > 0)

            vs = tok_arr_t.shape[0]
            gold_reg = tok_arr_t[gold.clamp(0, vs-1)]
            t1_reg   = cr[:, 0]
            sr_b     = (gold_reg == t1_reg) & (gold_reg != unk_region)
            if sr_enabled and cs is not None:
                rlen = reg_arr_t.shape[0] - 1
                gold_sup = reg_arr_t[gold_reg.clamp(0, rlen)]
                t1_sup   = cs[:, 0]
                ss_b     = (gold_sup == t1_sup) & (gold_sup != unk_super)
            else:
                ss_b = torch.zeros(b, dtype=torch.bool, device=device)

            sl = slice(s, e)
            row["covered"][sl]          = np_(cov_b)
            row["base_wrong"][sl]       = np_(bw_b)
            row["same_reg"][sl]         = np_(sr_b)
            row["same_sup"][sl]         = np_(ss_b)
            row["changed_to_gold"][sl]  = np_(ctg_b)
            row["changed_away"][sl]     = np_(caw_b)
            row["refined_wrong"][sl]    = np_(rw_b)
            row["gold_rank_improved"][sl] = np_(gri_b)
            row["gold_rank_worsened"][sl] = np_(grw_b)
            row["pairwise_win_base"][sl]= np_(pwb_b)
            row["pairwise_win_ref"][sl] = np_(pwr_b)
            row["actual_delta_abs"][sl] = np_(act_delta.abs().mean(1))
            row["mean_gate"][sl]        = np_(gate_b.mean(1))
            row["max_gate"][sl]         = np_(gate_b.max(1).values)
            row["gold_tok"][sl]         = np_(gold)
            row["base_top1_tok"][sl]    = np_(base_t1)
            row["ref_top1_tok"][sl]     = np_(ref_t1)
            row["gold_idx"][sl]         = np_(gidx_b)
            kk = min(10, K)
            row["delta_top10"][sl]      = np_(act_delta[:, :kk])
            row["gate_top10"][sl]       = np_(gate_b[:, :kk])

            for rk, arr, valid in [
                ("gold_rank_base_1b",  grb_b + 1, cov_b),
                ("gold_rank_ref_1b",   grr_b + 1, cov_b),
                ("pair_margin_base",   pmb_b, bw_b & cov_b),
                ("pair_margin_ref",    pmr_b, bw_b & cov_b),
                ("ce_base",  ceb_b, cov_b),
                ("ce_ref",   cer_b, cov_b),
                ("base_gold_margin", base_gm, cov_b),
                ("ref_gold_margin",  ref_gm,  cov_b),
            ]:
                vals = np_(arr.float())
                vals[~np_(valid)] = float("nan")
                row[rk][sl] = vals

    row["N"] = N
    subsets = {}
    for name, fn in _SUBSET_DEFS_V2.items():
        if name == "same_superregion_confuser" and not sr_enabled:
            continue
        m = fn(row)
        sm = _subset_metrics_v2(m, row)
        subsets[name] = sm

    # Compute noharm_adjusted_score
    ctg = subsets.get("target_confuser", {}).get("changed_to_gold_rate", float("nan"))
    caw = subsets.get("base_correct_covered", {}).get("changed_away_rate", float("nan"))
    pwr = subsets.get("target_confuser", {}).get("pairwise_win_rate_refined", float("nan"))
    if not any(math.isnan(v) for v in [ctg, caw, pwr]):
        score = ctg - 2.0 * caw + 0.25 * pwr
    else:
        score = float("nan")
    subsets["_noharm_adjusted_score"] = score

    model.train()
    return subsets, row


# ─────────────────────────────────────────────────────────────────────────────
# Full-vocab eval
# ─────────────────────────────────────────────────────────────────────────────

def eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                    unk_region, unk_super, sr_enabled, args, device):
    try:
        model.eval()
        N  = val_data["h_prime"].shape[0]
        K  = val_data["topk_ids"].shape[1]
        VS = model.token_emb_weight.shape[0]

        tot_nll_b, tot_nll_r = 0.0, 0.0
        tot_acc_b, tot_acc_r = 0, 0

        with torch.no_grad():
            for s in range(0, N, args.eval_batch_size):
                e    = min(s + args.eval_batch_size, N)
                hp   = val_data["h_prime"][s:e].to(device)
                topk = val_data["topk_ids"][s:e].to(device)
                lgt  = val_data["topk_lgt"][s:e].to(device)
                gold = val_data["gold"][s:e].to(device)

                cr, cs = _lookup_regs(topk, tok_arr_t, reg_arr_t,
                                       unk_region, unk_super, sr_enabled)
                _, act_delta, _ = model(hp, topk, lgt, cr, cs)

                fv_base = hp @ model.token_emb_weight.T           # [b, VS]
                fv_ref  = fv_base.clone()
                fv_ref.scatter_add_(1, topk.clamp(0, VS-1), act_delta)

                gs = gold.clamp(0, VS - 1)
                tot_nll_b += F.cross_entropy(fv_base, gs, reduction="sum").item()
                tot_nll_r += F.cross_entropy(fv_ref,  gs, reduction="sum").item()
                tot_acc_b += (fv_base.argmax(1) == gs).sum().item()
                tot_acc_r += (fv_ref.argmax(1)  == gs).sum().item()

        model.train()
        return {
            "full_vocab_base_nll":         tot_nll_b / N,
            "full_vocab_refined_nll":      tot_nll_r / N,
            "full_vocab_gain":             (tot_nll_b - tot_nll_r) / N,
            "full_vocab_top1_acc_base":    tot_acc_b / N,
            "full_vocab_top1_acc_refined": tot_acc_r / N,
        }
    except Exception as e:
        model.train()
        print(f"[warn] full_vocab eval failed: {e}")
        return {"full_vocab_eval_failed": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Example reports
# ─────────────────────────────────────────────────────────────────────────────

def _decode(ids, tok):
    try:
        return tok.decode(ids.tolist() if hasattr(ids, "tolist") else list(ids),
                          skip_special_tokens=False)
    except Exception:
        return str(ids)


def _fmt_example_v2(ex_i, row_i, val_data, row, tok_arr_t, reg_arr_t,
                    unk_region, unk_super, tokenizer, args):
    gt = int(row["gold_tok"][row_i])
    b1 = int(row["base_top1_tok"][row_i])
    r1 = int(row["ref_top1_tok"][row_i])
    grb = row["gold_rank_base_1b"][row_i]
    grr = row["gold_rank_ref_1b"][row_i]
    pmb = row["pair_margin_base"][row_i]
    pmr = row["pair_margin_ref"][row_i]
    mg  = row["mean_gate"][row_i]
    rid = str(val_data["rid"][row_i]) if val_data.get("rid") else str(row_i)

    vs = tok_arr_t.shape[0]
    gr_gt = int(tok_arr_t[min(gt, vs-1)].item())
    gr_b1 = int(tok_arr_t[min(b1, vs-1)].item())
    gr_r1 = int(tok_arr_t[min(r1, vs-1)].item())

    def d(tid):
        try: return f"`{tokenizer.decode([tid])}`"
        except: return str(tid)

    lines = [f"### Example {ex_i} (row_id={rid})\n"]
    if val_data.get("ids") is not None:
        lines.append(f"**Context:** `{_decode(val_data['ids'][row_i][-64:], tokenizer)}`\n")
    lines.append(f"**Gold:** {d(gt)} id={gt} region={gr_gt}")
    lines.append(f"**Base top-1:** {d(b1)} id={b1} region={gr_b1}")
    lines.append(f"**Refined top-1:** {d(r1)} id={r1} region={gr_r1}")
    lines.append(f"**Gold rank:** base={grb:.0f}  refined={grr:.0f}")
    lines.append(f"**Pair margin:** base={pmb:.3f}  refined={pmr:.3f}")
    lines.append(f"**Mean gate:** {mg:.4f}\n")

    K  = val_data["topk_ids"].shape[1]
    topk_row = val_data["topk_ids"][row_i]
    lgt_row  = val_data["topk_lgt"][row_i]
    dt_arr   = row["delta_top10"][row_i]
    ga_arr   = row["gate_top10"][row_i]

    lines.append("| Rank | Token | ID | Region | Gate | Delta | Base logit |")
    lines.append("|------|-------|----|--------|------|-------|------------|")
    for r in range(min(10, K)):
        tid = int(topk_row[r])
        try: ts = tokenizer.decode([tid])
        except: ts = str(tid)
        reg = int(tok_arr_t[min(tid, vs-1)].item())
        bl  = float(lgt_row[r])
        dt  = float(dt_arr[r]) if r < len(dt_arr) else 0.0
        ga  = float(ga_arr[r]) if r < len(ga_arr) else 0.0
        mk  = "✓" if tid == gt else ""
        lines.append(f"| {r+1} | `{ts}` | {tid} | {reg} | {ga:.4f} | {dt:.4f} | {bl:.3f} | {mk}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_example_reports(model, val_data, tok_arr_t, reg_arr_t,
                          unk_region, unk_super, sr_enabled,
                          args, device, tokenizer, subsets, row, output_dir):
    n_max = args.num_examples
    rng   = np.random.default_rng(42)

    cov   = row["covered"]
    bw    = row["base_wrong"]
    ctg   = row["changed_to_gold"]
    caw   = row["changed_away"]
    sr    = row["same_reg"]
    ss    = row["same_sup"]
    mg    = row["mean_gate"]

    buckets = {
        "target_flipped_to_gold": np.where(ctg & bw & cov)[0],
        "target_failed":          np.where(bw & cov & (sr | ss) & ~ctg)[0],
        "noharm_changed_away":    np.where(caw)[0],
        "noharm_preserved":       np.where(~bw & cov & ~caw)[0],
        "gate_high":              np.argsort(-mg)[:n_max * 4],
    }

    kw = dict(val_data=val_data, row=row,
              tok_arr_t=tok_arr_t.cpu(), reg_arr_t=reg_arr_t.cpu(),
              unk_region=unk_region, unk_super=unk_super,
              tokenizer=tokenizer, args=args)

    titles = {
        "target_flipped_to_gold": "Target: Flipped to Gold",
        "target_failed":          "Target: Failed Same-Region Confuser",
        "noharm_changed_away":    "No-harm: Changed Away from Gold",
        "noharm_preserved":       "No-harm: Preserved Gold Top-1",
        "gate_high":              "High Gate Values",
    }

    for bname, indices in buckets.items():
        if len(indices) > n_max:
            indices = rng.choice(indices, n_max, replace=False)
        path = os.path.join(output_dir, f"examples_{bname}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {titles[bname]}\n\n_{len(indices)} examples_\n\n---\n\n")
            for ei, ri in enumerate(indices):
                f.write(_fmt_example_v2(ei + 1, int(ri), **kw))
                f.write("\n---\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv_row(path, row_dict, write_header=False):
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(run_dir, args, pool_stats, best_metrics,
                 final_subsets, best_step, fv_final):
    path = os.path.join(run_dir, "report.md")
    m_all = final_subsets.get("all", {})
    m_tgt = final_subsets.get("target_confuser", {})
    m_nh  = final_subsets.get("base_correct_covered", {})

    def f(v, fmt=".4f"):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "nan"
        if isinstance(v, float):
            return format(v, fmt)
        return str(v)

    lines = []
    lines.append("# Region Pairwise Reranker V2 — Report\n")
    lines.append(f"**Run:** `{args.run_name}`  |  **Steps:** {args.steps}  "
                 f"|  **Best step:** {best_step}\n")

    lines.append("## V1 Reference (same_region_pair_v1)\n")
    lines.append("| Metric | V1 Value |")
    lines.append("|--------|----------|")
    lines.append("| pairwise_win_rate_refined | 0.1379 |")
    lines.append("| changed_to_gold_rate | 0.1018 |")
    lines.append("| full_vocab_gain | -0.0050 |")
    lines.append("| top1_acc drop | -0.0101 |\n")

    lines.append("## Training Pool Stats\n")
    for k, v in pool_stats.items():
        lines.append(f"  {k:30s} = {f(v)}")
    lines.append("")

    lines.append("## Final Val Metrics\n")
    lines.append("### All rows\n")
    for k in ["n", "top1_acc_base", "top1_acc_refined", "top1_acc_gain",
               "candidate_ce_base", "candidate_ce_refined", "candidate_ce_gain",
               "mean_actual_delta_abs", "max_actual_delta_abs",
               "mean_gate", "max_gate"]:
        lines.append(f"  {k:40s} = {f(m_all.get(k, float('nan')))}")
    lines.append("")

    lines.append("### target_confuser\n")
    for k in ["n", "changed_to_gold_rate", "changed_away_rate",
               "pairwise_win_rate_base", "pairwise_win_rate_refined",
               "mean_pair_margin_base", "mean_pair_margin_refined",
               "gold_rank_improved_rate", "gold_rank_worsened_rate",
               "mean_gate", "mean_actual_delta_abs"]:
        lines.append(f"  {k:40s} = {f(m_tgt.get(k, float('nan')))}")
    lines.append("")

    lines.append("### base_correct_covered (no-harm subset)\n")
    for k in ["n", "changed_away_rate", "margin_eroded_rate",
               "mean_margin_change", "mean_gate", "mean_actual_delta_abs"]:
        lines.append(f"  {k:40s} = {f(m_nh.get(k, float('nan')))}")
    lines.append("")

    if fv_final:
        lines.append("### Full-vocab eval\n")
        for k in ["full_vocab_base_nll", "full_vocab_refined_nll",
                   "full_vocab_gain", "full_vocab_top1_acc_base",
                   "full_vocab_top1_acc_refined"]:
            if k in fv_final:
                lines.append(f"  {k:40s} = {f(fv_final[k])}")
        lines.append("")

    score = final_subsets.get("_noharm_adjusted_score", float("nan"))
    lines.append(f"**noharm_adjusted_score:** {f(score)}\n")

    lines.append("## Interpretation\n")

    ctg = m_tgt.get("changed_to_gold_rate", float("nan"))
    caw = m_nh.get("changed_away_rate",     float("nan"))
    pwr = m_tgt.get("pairwise_win_rate_refined", float("nan"))
    fvg = fv_final.get("full_vocab_gain",   float("nan")) if fv_final else float("nan")
    mg_t = m_tgt.get("mean_gate",           float("nan"))
    mg_n = m_nh.get("mean_gate",            float("nan"))

    def chk(cond, yes, no):
        return yes if (not math.isnan(float(cond)) and cond) else no

    lines.append(f"**V2 vs V1 target corrections?**  "
                 f"ctg={f(ctg)}  vs  V1=0.1018  "
                 + chk(not math.isnan(ctg) and ctg >= 0.07,
                        "✅ Comparable", "⚠️  Lower than V1"))

    lines.append(f"**V2 reduced changed_away?**  "
                 f"caw={f(caw)}  "
                 + chk(not math.isnan(caw) and caw < 0.01,
                        "✅ Yes (< 0.01)", "⚠️  Still too high"))

    lines.append(f"**Full-vocab NLL improved vs V1?**  "
                 f"fv_gain={f(fvg)}  (V1 gain=-0.0050)  "
                 + chk(not math.isnan(fvg) and fvg > -0.001,
                        "✅ Neutral or positive", "❌ Still hurts global NLL"))

    if not math.isnan(mg_t) and not math.isnan(mg_n):
        lines.append(f"**Gate selective?**  "
                     f"mean_gate_target={f(mg_t)}  mean_gate_noharm={f(mg_n)}  "
                     + chk(mg_t > mg_n * 1.5,
                            "✅ Yes (target gate > noharm gate)", "⚠️  Gate not well-differentiated"))

    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# AMP context
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _amp_ctx(use_amp):
    if use_amp:
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


# ─────────────────────────────────────────────────────────────────────────────
# JSON helper
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    run_dir = os.path.join(args.output_root, args.run_name)
    if os.path.exists(run_dir):
        raise RuntimeError(
            f"Run dir already exists: {run_dir}\n"
            "Use --run_name to avoid overwriting old runs.")
    os.makedirs(run_dir, exist_ok=False)

    print("\n[backbone] Loading token embeddings...")
    tok_w, d_model, vocab_size = load_backbone(args.small_ckpt, device)
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    print("\n[maps] Loading region/super maps...")
    t2r, r2s, unk_region, unk_super, sr_enabled = load_maps(
        args.token_to_region, args.super_map)
    if not sr_enabled:
        print("  [INFO] superregion_enabled = false")

    tok_arr_np = build_tok_arr(t2r, unk_region, vocab_size)
    reg_arr_np = build_reg_arr(r2s, unk_super, unk_region)
    tok_arr_t  = torch.from_numpy(tok_arr_np).long().to(device)
    reg_arr_t  = torch.from_numpy(reg_arr_np).long().to(device)

    print("\n[data] Loading shards...")
    train_data = load_shards(args.train_dir, args.top_k, "train",
                             max_rows=args.max_train_rows, load_ids=False)
    val_data   = load_shards(args.val_dir,   args.top_k, "val",
                             max_rows=args.max_val_rows,   load_ids=True)

    print("\n[pools] Building train pools...")
    target_idx, noharm_idx, other_idx, pool_stats = build_train_pools(
        train_data, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled)
    for k, v in pool_stats.items():
        print(f"  {k:30s} = {v:.4f}" if isinstance(v, float) else f"  {k:30s} = {v}")

    if len(target_idx) == 0:
        raise RuntimeError("No target training rows found.")
    if len(noharm_idx) == 0:
        raise RuntimeError("No noharm training rows found.")

    # Build model
    n_regions = unk_region
    n_supers  = unk_super if sr_enabled else 1

    print("\n[model] Building RegionPairwiseRerankerV2...")
    model = RegionPairwiseRerankerV2(
        token_emb_weight=tok_w.to(device),
        tok_arr=tok_arr_np, reg_arr=reg_arr_np,
        d_model=d_model, n_regions=n_regions, n_supers=n_supers,
        sr_enabled=sr_enabled,
        resolver_dim=args.resolver_dim, resolver_layers=args.resolver_layers,
        resolver_heads=args.resolver_heads,
        region_emb_dim=args.region_emb_dim, super_emb_dim=args.super_emb_dim,
        delta_scale=args.delta_scale, gate_bias_init=args.gate_bias_init,
        top_k=args.top_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    config = vars(args).copy()
    config.update({"d_model": d_model, "vocab_size": vocab_size,
                   "n_regions": n_regions, "n_supers": n_supers,
                   "sr_enabled": sr_enabled, "n_params": n_params,
                   "pool_stats": pool_stats})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\n[identity] Running identity check...")
    run_identity_check(model, val_data, tok_arr_t, reg_arr_t,
                       unk_region, unk_super, sr_enabled, args, device, run_dir)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None

    # Batch size split
    target_bsz = max(1, round(args.batch_size * args.target_fraction))
    noharm_bsz = max(0, args.batch_size - target_bsz)

    N_t = len(target_idx)
    N_n = len(noharm_idx)
    t_perm = torch.randperm(N_t)
    n_perm = torch.randperm(N_n) if N_n > 0 else None
    t_pos  = 0
    n_pos  = 0

    train_log  = os.path.join(run_dir, "train_log.csv")
    eval_log   = os.path.join(run_dir, "eval_log.csv")
    subset_log = os.path.join(run_dir, "subset_eval_log.csv")
    hdr_train = hdr_eval = hdr_sub = False

    best_score     = -float("inf")
    best_step      = 0
    best_fv        = {}
    step           = 0
    accum_count    = 0
    t0             = time.time()
    optimizer.zero_grad()

    print(f"\n[train] Steps={args.steps}  batch={args.batch_size} "
          f"(target={target_bsz} noharm={noharm_bsz})  "
          f"accum={args.grad_accum_steps}  lr={args.lr}\n")

    while step < args.steps:
        # Sample target rows
        if t_pos + target_bsz > N_t:
            t_perm = torch.randperm(N_t); t_pos = 0
        t_global = target_idx[t_perm[t_pos:t_pos + target_bsz]]
        t_pos += target_bsz

        # Sample noharm rows
        if noharm_bsz > 0 and N_n > 0:
            if n_pos + noharm_bsz > N_n:
                n_perm = torch.randperm(N_n); n_pos = 0
            n_global = noharm_idx[n_perm[n_pos:n_pos + noharm_bsz]]
            n_pos += noharm_bsz
        else:
            n_global = torch.empty(0, dtype=torch.long)

        batch_global = torch.cat([t_global, n_global])
        row_types_cpu = torch.cat([
            torch.zeros(len(t_global), dtype=torch.long),
            torch.ones(len(n_global),  dtype=torch.long),
        ])
        perm_b = torch.randperm(len(batch_global))
        batch_global  = batch_global[perm_b]
        row_types_cpu = row_types_cpu[perm_b]

        hp    = train_data["h_prime"][batch_global].to(device)
        topk  = train_data["topk_ids"][batch_global].to(device)
        lgt   = train_data["topk_lgt"][batch_global].to(device)
        gold  = train_data["gold"][batch_global].to(device)
        rtype = row_types_cpu.to(device)
        B     = hp.shape[0]

        gold_in = (topk == gold.unsqueeze(1))
        covered = gold_in.any(1)
        gold_idx_t = gold_in.long().argmax(1)

        cr, cs = _lookup_regs(topk, tok_arr_t, reg_arr_t,
                               unk_region, unk_super, sr_enabled)
        vs = tok_arr_t.shape[0]
        gold_safe_t = gold.clamp(0, vs - 1)
        gold_regs_b = tok_arr_t[gold_safe_t]
        if sr_enabled and cs is not None:
            rlen = reg_arr_t.shape[0] - 1
            gold_sups_b = reg_arr_t[gold_regs_b.clamp(0, rlen)]
        else:
            gold_sups_b = torch.zeros(B, dtype=torch.long, device=device)

        with _amp_ctx(args.amp):
            ref, act_delta, gate_b = model(hp, topk, lgt, cr, cs)
            total_loss, ld = _compute_losses_v2(
                ref, act_delta, gate_b, lgt, gold_idx_t, covered, rtype,
                cr, cs, gold_regs_b, gold_sups_b,
                unk_region, unk_super, sr_enabled, args, device)
            loss_sc = total_loss / args.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss_sc).backward()
        else:
            loss_sc.backward()

        accum_count += 1
        if accum_count < args.grad_accum_steps:
            continue

        accum_count = 0
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 50 == 0 or step == 1:
            print(f"  step={step:5d}  loss={ld['total']:.4f}  "
                  f"pair={ld['pair']:.4f}  nh_ce={ld['noharm_ce']:.4f}  "
                  f"nh_mg={ld['noharm_margin']:.4f}  gate={ld['gate_sparsity']:.4f}  "
                  f"t={time.time()-t0:.0f}s")

        tr = {"step": step, **ld}
        _write_csv_row(train_log, tr, write_header=not hdr_train); hdr_train = True

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n[eval] step={step}")
            subsets, row_res = evaluate(
                model, val_data, tok_arr_t, reg_arr_t,
                unk_region, unk_super, sr_enabled, args, device, step, run_dir)

            fv = {}
            if args.eval_full_vocab:
                fv = eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                                     unk_region, unk_super, sr_enabled, args, device)

            m_tgt = subsets.get("target_confuser", {})
            m_nh  = subsets.get("base_correct_covered", {})
            m_all = subsets.get("all", {})
            score = subsets.get("_noharm_adjusted_score", float("nan"))

            print(f"  [all]  top1_base={m_all.get('top1_acc_base', float('nan')):.4f}  "
                  f"top1_ref={m_all.get('top1_acc_refined', float('nan')):.4f}  "
                  f"gate={m_all.get('mean_gate', float('nan')):.4f}")
            print(f"  [tgt]  n={m_tgt.get('n',0)}  "
                  f"ctg={m_tgt.get('changed_to_gold_rate', float('nan')):.4f}  "
                  f"pwr={m_tgt.get('pairwise_win_rate_refined', float('nan')):.4f}")
            print(f"  [nh]   n={m_nh.get('n',0)}  "
                  f"caw={m_nh.get('changed_away_rate', float('nan')):.4f}  "
                  f"eroded={m_nh.get('margin_eroded_rate', float('nan')):.4f}")
            print(f"  noharm_score={score:.4f}" if not math.isnan(score) else "  noharm_score=nan")
            if fv:
                print(f"  fv_gain={fv.get('full_vocab_gain', float('nan')):.4f}")

            torch.save({"step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "subsets": subsets, "fv": fv},
                       os.path.join(run_dir, "latest_reranker.pt"))

            if not math.isnan(score) and score > best_score:
                best_score = score
                best_step  = step
                best_fv    = fv
                torch.save({"step": step, "model": model.state_dict(),
                            "subsets": subsets, "fv": fv},
                           os.path.join(run_dir, "best_reranker.pt"))
                with open(os.path.join(run_dir, "best_metrics.json"), "w") as fp:
                    json.dump(_json_safe({"step": step, "score": score,
                                         "subsets": subsets, "fv": fv}), fp, indent=2)
                print(f"  [best] step={step}  noharm_score={score:.4f}")

            ev_row = {"step": step, **fv}
            for sn, sm in subsets.items():
                if sn.startswith("_"): continue
                for k, v in sm.items(): ev_row[f"{sn}_{k}"] = v
            _write_csv_row(eval_log, ev_row, write_header=not hdr_eval); hdr_eval = True

            for sn, sm in subsets.items():
                if sn.startswith("_"): continue
                sr = {"step": step, "subset": sn, **sm}
                _write_csv_row(subset_log, sr, write_header=not hdr_sub); hdr_sub = True

            print()

    # Final eval
    print("[final eval]")
    final_subsets, final_row = evaluate(
        model, val_data, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled, args, device, args.steps, run_dir)

    fv_final = {}
    if args.eval_full_vocab:
        fv_final = eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                                   unk_region, unk_super, sr_enabled, args, device)

    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump(_json_safe({"step": args.steps,
                              "subsets": final_subsets, "fv": fv_final}), f, indent=2)

    try:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            tokenizer = type("FT", (), {"decode": lambda s, ids, **kw: str(ids)})()
        write_example_reports(model, val_data, tok_arr_t, reg_arr_t,
                              unk_region, unk_super, sr_enabled,
                              args, device, tokenizer,
                              final_subsets, final_row, run_dir)
    except Exception as e:
        print(f"[warn] Example reports failed: {e}")

    write_report(run_dir, args, pool_stats, {}, final_subsets, best_step, fv_final)

    print(f"\n{'='*60}")
    print(f" Region Pairwise Reranker V2 complete.")
    print(f" Run dir: {run_dir}")
    print(f" Best step: {best_step}  noharm_score={best_score:.4f}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Region Pairwise Reranker V2")
    p.add_argument("--small_ckpt",       required=True)
    p.add_argument("--train_dir",        required=True)
    p.add_argument("--val_dir",          required=True)
    p.add_argument("--token_to_region",  required=True)
    p.add_argument("--super_map",        default=None)
    p.add_argument("--output_root",      default="runs/region_pairwise_reranker_v2")
    p.add_argument("--run_name",         default="gated_same_region_pair_v1")

    p.add_argument("--top_k",            type=int,   default=256)
    p.add_argument("--max_train_rows",   type=int,   default=None)
    p.add_argument("--max_val_rows",     type=int,   default=None)
    p.add_argument("--num_confusers",    type=int,   default=8)
    p.add_argument("--num_examples",     type=int,   default=50)
    p.add_argument("--target_fraction",  type=float, default=0.5)

    p.add_argument("--resolver_dim",     type=int,   default=256)
    p.add_argument("--resolver_layers",  type=int,   default=2)
    p.add_argument("--resolver_heads",   type=int,   default=4)
    p.add_argument("--region_emb_dim",   type=int,   default=64)
    p.add_argument("--super_emb_dim",    type=int,   default=32)
    p.add_argument("--delta_scale",      type=float, default=0.25)
    p.add_argument("--gate_bias_init",   type=float, default=-4.0)

    p.add_argument("--lr",               type=float, default=1e-5)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--grad_accum_steps", type=int,   default=2)
    p.add_argument("--steps",            type=int,   default=3000)
    p.add_argument("--eval_every",       type=int,   default=500)
    p.add_argument("--eval_batch_size",  type=int,   default=64)
    p.add_argument("--grad_clip",        type=float, default=0.5)
    p.add_argument("--amp",              action="store_true")
    p.add_argument("--eval_full_vocab",  action="store_true")
    p.add_argument("--seed",             type=int,   default=42)

    p.add_argument("--lambda_pair",          type=float, default=2.0)
    p.add_argument("--lambda_multi",         type=float, default=0.25)
    p.add_argument("--lambda_ce",            type=float, default=0.25)
    p.add_argument("--lambda_kl",            type=float, default=2.0)
    p.add_argument("--lambda_delta",         type=float, default=5e-3)
    p.add_argument("--lambda_noharm_ce",     type=float, default=1.0)
    p.add_argument("--lambda_noharm_margin", type=float, default=1.0)
    p.add_argument("--lambda_gate",          type=float, default=1e-3)

    return p.parse_args()


if __name__ == "__main__":
    main()
