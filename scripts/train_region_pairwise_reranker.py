#!/usr/bin/env python3
"""
train_region_pairwise_reranker.py — Region Pairwise Reranker V1

Tests whether pairwise loss trains a small region-aware reranker to flip
gold above the wrong base-top-1 candidate, using only:
  base topK logits, token embeddings (frozen backbone), h_prime, interaction-derived regions.

No retrieval. No manual token features. No hand-coded categories.
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
from typing import Dict, List, Optional, Tuple

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

_TRAIN_FILTERS = [
    "covered_base_wrong",
    "same_region_confuser",
    "same_superregion_confuser",
    "same_region_or_superregion_confuser",
    "all_covered",
]

_RANK_EMB_DIM = 8

_BEST_METRIC_KEY = "same_region_or_superregion_confuser_pairwise_win_rate_refined"
_BEST_METRIC_SECONDARY = "same_region_or_superregion_confuser_changed_to_gold_rate"


# ─────────────────────────────────────────────────────────────────────────────
# Backbone loading
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(ckpt_path: str, device: torch.device):
    """Returns (token_emb_weight, d_model, vocab_size) — backbone is discarded."""
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        ckpt_path, device)
    tok_w = backbone.token_emb.weight.detach().float()
    del backbone
    return tok_w, d_model, vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# Map loading
# ─────────────────────────────────────────────────────────────────────────────

def _parse_map(raw):
    if isinstance(raw, list):
        return {i: v for i, v in enumerate(raw) if v is not None}
    elif isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    raise ValueError(f"Expected list or dict, got {type(raw)}")


def load_maps(t2r_path: str, super_path: Optional[str]):
    """Load token→region and (optionally) region→superregion.

    Returns (t2r, r2s, unk_region, unk_super, sr_enabled).
    unk_region = max_region + 1; unk_super = max_super + 1 (or 0 if no super).
    """
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

    n_reg = unk_region
    n_sup = unk_super if sr_enabled else 0
    print(f"  maps: n_regions={n_reg}  unk_region={unk_region}  "
          f"n_supers={n_sup}  sr_enabled={sr_enabled}")
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
    max_reg = unk_region  # unk_region = max_mapped + 1
    arr = np.full(max_reg + 1, unk_super, dtype=np.int32)
    for reg, sup in r2s.items():
        if 0 <= reg <= max_reg:
            arr[reg] = int(sup)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Shard loading
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

        hp   = _get_field(sh, "h_prime", "h_ctx").float()        # [B, d]
        topk = _get_field(sh, "base_topk_ids", "base_topk").long()  # [B, K]
        lgt  = _get_field(sh, "base_topk_lgt", "base_topk_logits", required=True).float()
        gold = _get_field(sh, "gold_token", "gold").long()        # [B]
        rid  = _get_field(sh, "row_id", required=False)
        ids  = _get_field(sh, "input_ids", required=False) if load_ids else None

        B, K = topk.shape

        if K < top_k:
            pad_t = torch.zeros(B, top_k - K, dtype=torch.long)
            pad_l = torch.full((B, top_k - K), float("nan"))
            topk = torch.cat([topk, pad_t], 1)
            lgt  = torch.cat([lgt,  pad_l], 1)
        elif K > top_k:
            topk = topk[:, :top_k]
            lgt  = lgt[:,  :top_k]

        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            hp, topk, lgt, gold = hp[:keep], topk[:keep], lgt[:keep], gold[:keep]
            if rid is not None:
                rid = rid[:keep] if isinstance(rid, list) else rid[:keep].tolist()
            if ids is not None:
                ids = ids[:keep]
            B = keep

        all_hp.append(hp)
        all_topk.append(topk)
        all_lgt.append(lgt)
        all_gold.append(gold)
        if rid is not None:
            all_rid.extend(rid if isinstance(rid, list) else rid.tolist())
        if ids is not None:
            all_ids.append(ids)
        total += B

    data = {
        "h_prime":  torch.cat(all_hp,   0),
        "topk_ids": torch.cat(all_topk, 0),
        "topk_lgt": torch.cat(all_lgt,  0),
        "gold":     torch.cat(all_gold, 0),
        "rid":      all_rid if all_rid else None,
        "ids":      torch.cat(all_ids, 0) if all_ids else None,
    }
    N = data["h_prime"].shape[0]
    print(f"  {split_name}: {N:,} rows from {len(shards)} shards  "
          f"(d={data['h_prime'].shape[1]}, K={data['topk_ids'].shape[1]})")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Filter
# ─────────────────────────────────────────────────────────────────────────────

def apply_filter(data, filter_name: str, tok_arr_t, reg_arr_t,
                 unk_region: int, unk_super: int, sr_enabled: bool):
    """Return (indices_tensor, stats_dict) for the requested train filter."""
    N = data["gold"].shape[0]
    K = data["topk_ids"].shape[1]
    gold    = data["gold"]          # [N]
    topk    = data["topk_ids"]      # [N, K]

    # Covered: gold appears anywhere in topK
    covered = (topk == gold.unsqueeze(1)).any(dim=1)          # [N]

    # Base wrong
    base_wrong = (topk[:, 0] != gold)                         # [N]

    # Region lookups — data is on CPU; force lookup arrays to CPU here
    tok_arr_cpu = tok_arr_t.cpu()
    reg_arr_cpu = reg_arr_t.cpu()

    vs = tok_arr_cpu.shape[0]
    gold_safe = gold.clamp(0, vs - 1)
    top1_safe = topk[:, 0].clamp(0, vs - 1)

    gold_reg = tok_arr_cpu[gold_safe]    # [N] CPU
    top1_reg = tok_arr_cpu[top1_safe]    # [N] CPU

    same_region = (gold_reg == top1_reg) & (gold_reg != unk_region) & (top1_reg != unk_region)

    if sr_enabled:
        rlen = reg_arr_cpu.shape[0] - 1
        gold_reg_safe = gold_reg.clamp(0, rlen)
        top1_reg_safe = top1_reg.clamp(0, rlen)
        gold_sup = reg_arr_cpu[gold_reg_safe]
        top1_sup = reg_arr_cpu[top1_reg_safe]
        same_super = (gold_sup == top1_sup) & (gold_sup != unk_super) & (top1_sup != unk_super)
    else:
        same_super = torch.zeros(N, dtype=torch.bool)

    covered_bw = covered & base_wrong

    if filter_name == "all_covered":
        mask = covered
    elif filter_name == "covered_base_wrong":
        mask = covered_bw
    elif filter_name == "same_region_confuser":
        mask = covered_bw & same_region
    elif filter_name == "same_superregion_confuser":
        if not sr_enabled:
            print("[WARN] same_superregion_confuser requested but sr_enabled=False; "
                  "falling back to same_region_confuser")
            mask = covered_bw & same_region
        else:
            mask = covered_bw & same_super
    elif filter_name == "same_region_or_superregion_confuser":
        mask = covered_bw & (same_region | same_super)
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    indices = mask.nonzero(as_tuple=False).squeeze(1)

    n_keep      = int(indices.shape[0])
    n_cov       = int(covered.sum())
    n_bw        = int(base_wrong.sum())
    n_sr        = int(same_region.sum())
    n_ss        = int(same_super.sum()) if sr_enabled else 0
    n_unmapped  = int((tok_arr_cpu[gold_safe] == unk_region).sum())

    stats = {
        "total_rows":         N,
        "rows_kept":          n_keep,
        "kept_rate":          n_keep / N if N > 0 else 0.0,
        "covered_rate":       n_cov / N if N > 0 else 0.0,
        "base_wrong_rate":    n_bw / N if N > 0 else 0.0,
        "same_region_rate":   n_sr / N if N > 0 else 0.0,
        "same_super_rate":    n_ss / N if N > 0 else 0.0,
        "gold_unmapped_rate": n_unmapped / N if N > 0 else 0.0,
    }
    return indices, stats


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class RegionPairwiseReranker(nn.Module):
    def __init__(self, token_emb_weight: torch.Tensor,
                 tok_arr: np.ndarray, reg_arr: np.ndarray,
                 d_model: int, n_regions: int, n_supers: int,
                 sr_enabled: bool,
                 resolver_dim: int = 256, resolver_layers: int = 2,
                 resolver_heads: int = 4,
                 region_emb_dim: int = 64, super_emb_dim: int = 32,
                 delta_scale: float = 0.25, top_k: int = 256,
                 dropout: float = 0.0):
        super().__init__()

        self.d_model       = d_model
        self.sr_enabled    = sr_enabled
        self.delta_scale   = delta_scale
        self.top_k         = top_k
        self.resolver_dim  = resolver_dim

        # Frozen buffers
        self.register_buffer("token_emb_weight",
                             token_emb_weight.detach().float())
        self.register_buffer("tok_arr",
                             torch.from_numpy(tok_arr.astype(np.int32)).long())
        self.register_buffer("reg_arr",
                             torch.from_numpy(reg_arr.astype(np.int32)).long())

        # Region / super embeddings (learnable inside reranker)
        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim)
        if sr_enabled:
            self.super_emb = nn.Embedding(n_supers + 2, super_emb_dim)

        # Rank embeddings
        self.rank_emb = nn.Embedding(top_k + 1, _RANK_EMB_DIM)

        # Candidate feature dim
        # tok_emb(d_model) + reg_emb(region_emb_dim) + rank_emb(_RANK_EMB_DIM)
        # + base_logit(1) + logit_gap(1) + same_reg_top1(1) + hp_dot(1)
        feat_dim = d_model + region_emb_dim + _RANK_EMB_DIM + 4
        if sr_enabled:
            feat_dim += super_emb_dim + 1   # super_emb + same_sup_top1

        self.cand_proj = nn.Linear(feat_dim, resolver_dim)

        # Context projection (MLP)
        self.ctx_proj = nn.Sequential(
            nn.Linear(d_model, resolver_dim),
            nn.GELU(),
            nn.Linear(resolver_dim, resolver_dim),
        )

        # Positional embedding [CTX + K candidates]
        self.pos_emb = nn.Embedding(top_k + 1, resolver_dim)

        # Transformer encoder (pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resolver_dim, nhead=resolver_heads,
            dim_feedforward=resolver_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=resolver_layers)

        # Delta head — zero initialized → identity at step 0
        self.delta_head = nn.Linear(resolver_dim, 1, bias=False)
        nn.init.zeros_(self.delta_head.weight)

        # Learnable mixing scalar
        self.register_buffer("alpha", torch.ones(1))

    def _region_super_lookup(self, token_ids: torch.Tensor):
        """token_ids: [B, K] → (cand_regs, cand_sups). cand_sups may be None."""
        vs = self.tok_arr.shape[0]
        safe = token_ids.clamp(0, vs - 1)
        cand_regs = self.tok_arr[safe]                                 # [B, K]
        if self.sr_enabled:
            rlen = self.reg_arr.shape[0] - 1
            cand_sups = self.reg_arr[cand_regs.clamp(0, rlen)]        # [B, K]
        else:
            cand_sups = None
        return cand_regs, cand_sups

    def forward(self, h_prime: torch.Tensor,
                candidate_ids: torch.Tensor,
                base_logits: torch.Tensor,
                cand_regs: torch.Tensor,
                cand_sups: Optional[torch.Tensor] = None):
        """
        h_prime:       [B, d_model]
        candidate_ids: [B, K]
        base_logits:   [B, K]
        cand_regs:     [B, K]  int — region id per candidate
        cand_sups:     [B, K]  int — super id per candidate (if sr_enabled)
        Returns (refined_logits [B, K], delta [B, K])
        """
        B, K = candidate_ids.shape

        # Frozen token embeddings [B, K, d_model]
        vs = self.token_emb_weight.shape[0]
        safe = candidate_ids.clamp(0, vs - 1)
        tok_embs = self.token_emb_weight[safe]

        # Region embeddings [B, K, region_emb_dim]
        reg_embs = self.region_emb(cand_regs)

        # Rank embeddings [B, K, _RANK_EMB_DIM]
        ranks = torch.arange(K, device=h_prime.device).unsqueeze(0).expand(B, -1)
        rank_embs = self.rank_emb(ranks)

        # Top-1 region for same_region feature
        top1_reg = cand_regs[:, 0:1]                                  # [B, 1]
        same_reg = (cand_regs == top1_reg).float()                    # [B, K]

        # Logit gap vs top-1 [B, K]
        logit_gap = base_logits - base_logits[:, 0:1]

        # h_prime · tok_emb [B, K]
        hp_dot = torch.bmm(tok_embs, h_prime.unsqueeze(2)).squeeze(2)

        parts = [
            tok_embs,                          # [B, K, d_model]
            reg_embs,                          # [B, K, region_emb_dim]
            rank_embs,                         # [B, K, _RANK_EMB_DIM]
            base_logits.unsqueeze(2),          # [B, K, 1]
            logit_gap.unsqueeze(2),            # [B, K, 1]
            same_reg.unsqueeze(2),             # [B, K, 1]
            hp_dot.unsqueeze(2),               # [B, K, 1]
        ]
        if self.sr_enabled and cand_sups is not None:
            sup_embs = self.super_emb(cand_sups)
            top1_sup = cand_sups[:, 0:1]
            same_sup = (cand_sups == top1_sup).float()
            parts.extend([sup_embs, same_sup.unsqueeze(2)])

        cand_feats = torch.cat(parts, dim=2)                          # [B, K, feat_dim]
        cand_emb   = self.cand_proj(cand_feats)                       # [B, K, resolver_dim]

        # CTX embedding [B, 1, resolver_dim]
        ctx = self.ctx_proj(h_prime).unsqueeze(1)

        # Sequence [B, K+1, resolver_dim] + positional
        seq = torch.cat([ctx, cand_emb], dim=1)
        pos_ids = torch.arange(K + 1, device=h_prime.device)
        seq = seq + self.pos_emb(pos_ids).unsqueeze(0)

        # Transformer
        seq_out = self.transformer(seq)                               # [B, K+1, resolver_dim]

        # Delta [B, K]
        raw_delta = self.delta_head(seq_out[:, 1:]).squeeze(2)
        delta = self.delta_scale * torch.tanh(raw_delta)

        refined_logits = base_logits + self.alpha * delta
        return refined_logits, delta


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _compute_losses(refined_logits, delta, base_logits,
                    gold_idx, covered, cand_regs, cand_sups,
                    gold_regs, gold_sups, unk_region, unk_super,
                    sr_enabled, args, device):
    B, K = refined_logits.shape
    arange = torch.arange(B, device=device)
    gold_idx_safe = gold_idx.clamp(0, K - 1)

    # ── Pairwise: gold vs base top-1 ──────────────────────────────────────────
    pair_mask = covered & (torch.arange(B, device=device).ge(0))
    # base_wrong implicitly: we only care where top1 != gold_idx position 0
    base_wrong = covered & (gold_idx != 0)
    pair_mask = base_wrong

    if pair_mask.any():
        z_g = refined_logits[pair_mask, gold_idx_safe[pair_mask]]
        z_b = refined_logits[pair_mask, 0]
        L_pair = F.softplus(-(z_g - z_b)).mean()
    else:
        L_pair = torch.tensor(0.0, device=device)

    # ── Multi-confuser ────────────────────────────────────────────────────────
    not_gold_mask = torch.ones(B, K, dtype=torch.bool, device=device)
    for i in range(B):
        not_gold_mask[i, gold_idx_safe[i]] = False

    same_reg_cand = (cand_regs == gold_regs.unsqueeze(1)) & \
                    (gold_regs.unsqueeze(1) != unk_region)
    if sr_enabled and cand_sups is not None:
        same_sup_cand = (cand_sups == gold_sups.unsqueeze(1)) & \
                        (gold_sups.unsqueeze(1) != unk_super)
        confuser_mask = not_gold_mask & (same_reg_cand | same_sup_cand)
    else:
        confuser_mask = not_gold_mask & same_reg_cand

    multi_loss = torch.tensor(0.0, device=device)
    n_multi = 0
    for i in range(B):
        if not covered[i]:
            continue
        conf_idx = confuser_mask[i].nonzero(as_tuple=False).squeeze(1)
        if len(conf_idx) == 0:
            continue
        conf_idx = conf_idx[:args.num_confusers]
        gi = gold_idx_safe[i]
        z_g = refined_logits[i, gi]
        z_c = refined_logits[i, conf_idx]
        multi_loss = multi_loss + F.softplus(-(z_g - z_c)).mean()
        n_multi += 1
    if n_multi > 0:
        multi_loss = multi_loss / n_multi

    # ── Candidate CE ──────────────────────────────────────────────────────────
    if covered.any():
        log_p_ref = F.log_softmax(refined_logits, dim=1)
        ce_ref = -log_p_ref[arange, gold_idx_safe]
        L_ce = ce_ref[covered].mean()
    else:
        L_ce = torch.tensor(0.0, device=device)

    # ── KL to base ────────────────────────────────────────────────────────────
    p_base = F.softmax(base_logits, dim=1)
    p_ref  = F.softmax(refined_logits, dim=1)
    # KL(p_base || p_ref) = sum(p_base * log(p_base / p_ref))
    log_ratio = (p_base + 1e-10).log() - (p_ref + 1e-10).log()
    L_kl = (p_base * log_ratio).sum(dim=1).mean()

    # ── Delta penalty ─────────────────────────────────────────────────────────
    L_delta = (delta ** 2).mean()

    total = (args.lambda_pair  * L_pair
           + args.lambda_multi * multi_loss
           + args.lambda_ce    * L_ce
           + args.lambda_kl    * L_kl
           + args.lambda_delta * L_delta)

    return total, {
        "pair":   L_pair.item(),
        "multi":  multi_loss.item(),
        "ce":     L_ce.item(),
        "kl":     L_kl.item(),
        "delta":  L_delta.item(),
        "total":  total.item(),
        "n_multi_rows": n_multi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Region/super lookup helpers (outside model, for training loop)
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_regs(token_ids, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled):
    vs = tok_arr_t.shape[0]
    safe = token_ids.clamp(0, vs - 1)
    regs = tok_arr_t[safe]
    if sr_enabled:
        rlen = reg_arr_t.shape[0] - 1
        sups = reg_arr_t[regs.clamp(0, rlen)]
    else:
        sups = None
    return regs, sups


# ─────────────────────────────────────────────────────────────────────────────
# Identity check
# ─────────────────────────────────────────────────────────────────────────────

def run_identity_check(model, val_data, tok_arr_t, reg_arr_t,
                       unk_region, unk_super, sr_enabled,
                       args, device, output_dir):
    model.eval()
    n = min(64, val_data["h_prime"].shape[0])
    idx = torch.arange(n)

    hp    = val_data["h_prime"][idx].to(device)
    topk  = val_data["topk_ids"][idx].to(device)
    lgt   = val_data["topk_lgt"][idx].to(device)
    gold  = val_data["gold"][idx].to(device)

    cand_regs, cand_sups = _lookup_regs(
        topk, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled)

    with torch.no_grad():
        ref_lgt, delta = model(hp, topk, lgt, cand_regs, cand_sups)

    delta_abs_max  = float(delta.abs().max())
    alpha          = float(model.alpha.item())
    diff_lgt       = (ref_lgt - lgt).abs()
    ce_diff        = float(diff_lgt.max())
    top1_base      = topk[:, 0]
    top1_ref_idx   = ref_lgt.argmax(dim=1)
    top1_ref_tok   = topk[torch.arange(n), top1_ref_idx]
    acc_base       = (top1_base == gold).float().mean().item()
    acc_ref        = (top1_ref_tok == gold).float().mean().item()

    pair_margin_diff_max = float((ref_lgt - lgt).abs().max())

    pass_delta = delta_abs_max < 1e-6
    pass_ce    = ce_diff < 1e-5
    pass_acc   = abs(acc_base - acc_ref) < 1e-6
    passed     = pass_delta and pass_ce and pass_acc

    result = {
        "delta_abs_max":            delta_abs_max,
        "alpha":                    alpha,
        "ce_diff_max":              ce_diff,
        "top1_acc_base":            acc_base,
        "top1_acc_refined":         acc_ref,
        "pair_margin_diff_max":     pair_margin_diff_max,
        "refined_logits_equal_base": passed,
        "passed":                   passed,
    }

    with open(os.path.join(output_dir, "identity_check.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[identity] delta_abs_max={delta_abs_max:.2e}  "
          f"ce_diff={ce_diff:.2e}  "
          f"acc_base={acc_base:.4f}  acc_ref={acc_ref:.4f}  "
          f"passed={passed}")

    if not passed:
        raise RuntimeError(
            "Identity check FAILED — delta_head is not zero at init. "
            "Aborting training. Check model initialization.")

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


def _subset_metrics(mask, row_res):
    """Compute metrics for a boolean subset mask."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}

    def M(key):
        return np.asarray(row_res[key])[mask]

    cov   = M("covered")
    bw    = M("base_wrong")
    ctg   = M("changed_to_gold")
    caw   = M("changed_away")
    gri   = M("gold_rank_improved")
    grw   = M("gold_rank_worsened")
    grb   = M("gold_rank_base_1b")
    grr   = M("gold_rank_ref_1b")
    pmb   = M("pair_margin_base")
    pmr   = M("pair_margin_ref")
    ceb   = M("ce_base")
    cer   = M("ce_ref")
    pw_b  = M("pairwise_win_base")
    pw_r  = M("pairwise_win_ref")
    dab   = M("delta_abs")

    def rate(arr):
        arr_f = arr.astype(float)
        return float(arr_f.mean()) if len(arr_f) > 0 else float("nan")

    def bw_mask():
        return bw & cov

    bwc = bw_mask()
    n_bwc = int(bwc.sum())

    return {
        "n":                        n,
        "covered_rate":             rate(cov),
        "base_wrong_rate":          rate(bw),
        "candidate_ce_base":        _safe_mean(ceb[cov]),
        "candidate_ce_refined":     _safe_mean(cer[cov]),
        "candidate_ce_gain":        _safe_mean(ceb[cov] - cer[cov]),
        "top1_acc_base":            rate(~bw),
        "top1_acc_refined":         rate(~M("refined_wrong")),
        "top1_acc_gain":            rate(~M("refined_wrong")) - rate(~bw),
        "changed_to_gold_rate":     rate(ctg[bw & cov]) if (bw & cov).any() else float("nan"),
        "changed_away_rate":        rate(caw[~bw]) if (~bw).any() else float("nan"),
        "gold_rank_improved_rate":  rate(gri[cov]),
        "gold_rank_worsened_rate":  rate(grw[cov]),
        "mean_gold_rank_base":      _safe_mean(grb[cov]),
        "mean_gold_rank_refined":   _safe_mean(grr[cov]),
        "median_gold_rank_base":    _safe_median(grb[cov]),
        "median_gold_rank_refined": _safe_median(grr[cov]),
        "pairwise_win_rate_base":   rate(pw_b[bwc]) if n_bwc > 0 else float("nan"),
        "pairwise_win_rate_refined":rate(pw_r[bwc]) if n_bwc > 0 else float("nan"),
        "mean_pair_margin_base":    _safe_mean(pmb[bwc]),
        "mean_pair_margin_refined": _safe_mean(pmr[bwc]),
        "mean_pair_margin_gain":    _safe_mean((pmr - pmb)[bwc]),
        "mean_delta_abs":           _safe_mean(dab),
        "max_delta_abs":            float(np.nanmax(dab)) if len(dab) > 0 else float("nan"),
        "alpha":                    float("nan"),  # filled outside
    }


_SUBSET_DEFS = {
    "all":                                    lambda d: np.ones(d["N"], dtype=bool),
    "covered":                                lambda d: d["covered"],
    "base_wrong_covered":                     lambda d: d["covered"] & d["base_wrong"],
    "same_region_confuser":                   lambda d: d["covered"] & d["base_wrong"] & d["same_reg"],
    "same_superregion_confuser":              lambda d: d["covered"] & d["base_wrong"] & d["same_sup"],
    "same_region_or_superregion_confuser":    lambda d: d["covered"] & d["base_wrong"] & (d["same_reg"] | d["same_sup"]),
    "base_miss":                              lambda d: ~d["covered"],
}


def evaluate(model, val_data, tok_arr_t, reg_arr_t,
             unk_region, unk_super, sr_enabled,
             args, device, tokenizer, step, output_dir,
             write_examples=False):
    model.eval()
    N   = val_data["h_prime"].shape[0]
    K   = val_data["topk_ids"].shape[1]
    BSZ = args.eval_batch_size

    # Per-row collectors
    row = {
        "covered":           np.zeros(N, dtype=bool),
        "base_wrong":        np.zeros(N, dtype=bool),
        "same_reg":          np.zeros(N, dtype=bool),
        "same_sup":          np.zeros(N, dtype=bool),
        "changed_to_gold":   np.zeros(N, dtype=bool),
        "changed_away":      np.zeros(N, dtype=bool),
        "refined_wrong":     np.zeros(N, dtype=bool),
        "gold_rank_improved":np.zeros(N, dtype=bool),
        "gold_rank_worsened":np.zeros(N, dtype=bool),
        "gold_rank_base_1b": np.full(N, float("nan"), dtype=np.float32),
        "gold_rank_ref_1b":  np.full(N, float("nan"), dtype=np.float32),
        "pair_margin_base":  np.full(N, float("nan"), dtype=np.float32),
        "pair_margin_ref":   np.full(N, float("nan"), dtype=np.float32),
        "ce_base":           np.full(N, float("nan"), dtype=np.float32),
        "ce_ref":            np.full(N, float("nan"), dtype=np.float32),
        "pairwise_win_base": np.zeros(N, dtype=bool),
        "pairwise_win_ref":  np.zeros(N, dtype=bool),
        "delta_abs":         np.zeros(N, dtype=np.float32),
        # For examples
        "refined_top1_tok":  np.zeros(N, dtype=np.int64),
        "gold_tok":          np.zeros(N, dtype=np.int64),
        "base_top1_tok":     np.zeros(N, dtype=np.int64),
        "gold_idx":          np.zeros(N, dtype=np.int32),
        "delta_all":         np.zeros((N, min(10, K)), dtype=np.float32),
    }

    with torch.no_grad():
        for start in range(0, N, BSZ):
            end   = min(start + BSZ, N)
            idx   = slice(start, end)
            b     = end - start

            hp   = val_data["h_prime"][idx].to(device)
            topk = val_data["topk_ids"][idx].to(device)
            lgt  = val_data["topk_lgt"][idx].to(device)
            gold = val_data["gold"][idx].to(device)
            arange_b = torch.arange(b, device=device)

            cand_regs, cand_sups = _lookup_regs(
                topk, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled)

            ref_lgt, delta_b = model(hp, topk, lgt, cand_regs, cand_sups)

            # Gold idx in topK
            gold_in_topk = (topk == gold.unsqueeze(1))             # [b, K]
            cov_b  = gold_in_topk.any(1)                           # [b]
            gidx_b = gold_in_topk.long().argmax(1)                 # [b]
            gidx_safe = gidx_b.clamp(0, K - 1)

            # Top-1 tokens
            base_t1_tok = topk[:, 0]
            ref_t1_idx  = ref_lgt.argmax(1)
            ref_t1_tok  = topk[arange_b, ref_t1_idx]

            # Gold rank in base (= gold_idx since sorted) and refined
            n_higher_ref = (ref_lgt > ref_lgt[arange_b, gidx_safe].unsqueeze(1)).sum(1)
            grb_b = torch.where(cov_b, gidx_b.long(), torch.tensor(-1, device=device))
            grr_b = torch.where(cov_b, n_higher_ref.long(), torch.tensor(-1, device=device))

            # Pairwise margins
            pmb_b = lgt[arange_b, gidx_safe] - lgt[:, 0]          # [b], neg for bw
            pmr_b = ref_lgt[arange_b, gidx_safe] - ref_lgt[:, 0]

            # CE
            log_p_b   = F.log_softmax(lgt, 1)
            log_p_ref = F.log_softmax(ref_lgt, 1)
            ce_b_b = -log_p_b[arange_b, gidx_safe]
            ce_r_b = -log_p_ref[arange_b, gidx_safe]

            # Derived booleans
            bw_b    = (base_t1_tok != gold)                        # [b]
            ctg_b   = bw_b & cov_b & (ref_t1_tok == gold)
            caw_b   = ~bw_b & (ref_t1_tok != gold)
            rw_b    = (ref_t1_tok != gold)
            gri_b   = cov_b & (grr_b < grb_b) & (grr_b >= 0)
            grw_b   = cov_b & (grr_b > grb_b)
            pw_b_b  = bw_b & cov_b & (pmb_b > 0)
            pw_r_b  = bw_b & cov_b & (pmr_b > 0)

            # Region/super same-as-gold for subsets
            vs = tok_arr_t.shape[0]
            gold_safe_b = gold.clamp(0, vs - 1)
            gold_reg_b  = tok_arr_t[gold_safe_b]
            t1_reg_b    = cand_regs[:, 0]
            sr_b = (gold_reg_b == t1_reg_b) & (gold_reg_b != unk_region)
            if sr_enabled and cand_sups is not None:
                rlen = reg_arr_t.shape[0] - 1
                gold_sup_b = reg_arr_t[gold_reg_b.clamp(0, rlen)]
                t1_sup_b   = cand_sups[:, 0]
                ss_b = (gold_sup_b == t1_sup_b) & (gold_sup_b != unk_super)
            else:
                ss_b = torch.zeros(b, dtype=torch.bool, device=device)

            def np_(t): return t.cpu().numpy()

            sl = slice(start, end)
            row["covered"][sl]            = np_(cov_b)
            row["base_wrong"][sl]         = np_(bw_b)
            row["same_reg"][sl]           = np_(sr_b)
            row["same_sup"][sl]           = np_(ss_b)
            row["changed_to_gold"][sl]    = np_(ctg_b)
            row["changed_away"][sl]       = np_(caw_b)
            row["refined_wrong"][sl]      = np_(rw_b)
            row["gold_rank_improved"][sl] = np_(gri_b)
            row["gold_rank_worsened"][sl] = np_(grw_b)
            row["pairwise_win_base"][sl]  = np_(pw_b_b)
            row["pairwise_win_ref"][sl]   = np_(pw_r_b)
            row["delta_abs"][sl]          = np_(delta_b.abs().mean(1))

            for r, arr, valid in [
                ("gold_rank_base_1b", grb_b + 1, cov_b),
                ("gold_rank_ref_1b",  grr_b + 1, cov_b),
                ("pair_margin_base",  pmb_b, bw_b & cov_b),
                ("pair_margin_ref",   pmr_b, bw_b & cov_b),
                ("ce_base",   ce_b_b, cov_b),
                ("ce_ref",    ce_r_b, cov_b),
            ]:
                vals = np_(arr.float())
                vals[~np_(valid)] = float("nan")
                row[r][sl] = vals

            row["gold_tok"][sl]          = np_(gold)
            row["base_top1_tok"][sl]     = np_(base_t1_tok)
            row["refined_top1_tok"][sl]  = np_(ref_t1_tok)
            row["gold_idx"][sl]          = np_(gidx_b)
            kk = min(10, K)
            row["delta_all"][sl]         = np_(delta_b[:, :kk])

    row["N"] = N
    alpha_val = float(model.alpha.item())

    subsets = {}
    for name, mask_fn in _SUBSET_DEFS.items():
        if name == "same_superregion_confuser" and not sr_enabled:
            continue
        mask = mask_fn(row)
        m = _subset_metrics(mask, row)
        m["alpha"] = alpha_val
        subsets[name] = m

    model.train()
    return subsets, row


# ─────────────────────────────────────────────────────────────────────────────
# Full-vocab eval
# ─────────────────────────────────────────────────────────────────────────────

def eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                    unk_region, unk_super, sr_enabled,
                    args, device):
    try:
        model.eval()
        N   = val_data["h_prime"].shape[0]
        K   = val_data["topk_ids"].shape[1]
        BSZ = args.eval_batch_size
        vocab_size = model.token_emb_weight.shape[0]

        tot_nll_base, tot_nll_ref = 0.0, 0.0
        tot_acc_base, tot_acc_ref = 0, 0

        with torch.no_grad():
            for start in range(0, N, BSZ):
                end  = min(start + BSZ, N)
                b    = end - start

                hp   = val_data["h_prime"][start:end].to(device)
                topk = val_data["topk_ids"][start:end].to(device)
                lgt  = val_data["topk_lgt"][start:end].to(device)
                gold = val_data["gold"][start:end].to(device)

                cand_regs, cand_sups = _lookup_regs(
                    topk, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled)

                _, delta_b = model(hp, topk, lgt, cand_regs, cand_sups)

                # Full-vocab base logits
                fv_base = hp @ model.token_emb_weight.T            # [b, vocab]

                # Scatter delta into full-vocab logits
                fv_ref = fv_base.clone()
                safe_ids = topk.clamp(0, vocab_size - 1)
                fv_ref.scatter_add_(1, safe_ids, delta_b * model.alpha)

                gold_safe = gold.clamp(0, vocab_size - 1)
                nll_b = F.cross_entropy(fv_base, gold_safe, reduction="sum")
                nll_r = F.cross_entropy(fv_ref,  gold_safe, reduction="sum")
                acc_b = (fv_base.argmax(1) == gold_safe).sum()
                acc_r = (fv_ref.argmax(1)  == gold_safe).sum()

                tot_nll_base += nll_b.item()
                tot_nll_ref  += nll_r.item()
                tot_acc_base += acc_b.item()
                tot_acc_ref  += acc_r.item()

        model.train()
        return {
            "full_vocab_base_nll":          tot_nll_base / N,
            "full_vocab_refined_nll":       tot_nll_ref  / N,
            "full_vocab_gain":              (tot_nll_base - tot_nll_ref) / N,
            "full_vocab_top1_acc_base":     tot_acc_base / N,
            "full_vocab_top1_acc_refined":  tot_acc_ref  / N,
        }
    except Exception as e:
        model.train()
        print(f"[warn] full_vocab eval failed: {e}")
        return {"full_vocab_eval_failed": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Example report writing
# ─────────────────────────────────────────────────────────────────────────────

def _decode(ids, tokenizer):
    try:
        return tokenizer.decode(
            ids.tolist() if hasattr(ids, "tolist") else list(ids),
            skip_special_tokens=False)
    except Exception:
        return str(ids)


def _fmt_example(ex_idx, row_i, val_data, row_results, model,
                 tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
                 tokenizer, args, device):
    gold_tok   = int(row_results["gold_tok"][row_i])
    b1_tok     = int(row_results["base_top1_tok"][row_i])
    r1_tok     = int(row_results["refined_top1_tok"][row_i])
    grb        = row_results["gold_rank_base_1b"][row_i]
    grr        = row_results["gold_rank_ref_1b"][row_i]
    pmb        = row_results["pair_margin_base"][row_i]
    pmr        = row_results["pair_margin_ref"][row_i]

    rid_s = str(val_data["rid"][row_i]) if val_data.get("rid") else str(row_i)

    vs = tok_arr_t.shape[0]
    gr_gold  = int(tok_arr_t[min(gold_tok, vs-1)])
    gr_b1    = int(tok_arr_t[min(b1_tok,   vs-1)])
    gr_r1    = int(tok_arr_t[min(r1_tok,   vs-1)])

    try: gold_s = f"`{tokenizer.decode([gold_tok])}`"
    except: gold_s = str(gold_tok)
    try: b1_s = f"`{tokenizer.decode([b1_tok])}`"
    except: b1_s = str(b1_tok)
    try: r1_s = f"`{tokenizer.decode([r1_tok])}`"
    except: r1_s = str(r1_tok)

    lines = [f"### Example {ex_idx} (row_id={rid_s})\n"]

    if val_data.get("ids") is not None:
        ctx = _decode(val_data["ids"][row_i][-64:], tokenizer)
        lines.append(f"**Context:** `{ctx}`\n")

    lines.append(f"**Gold:** {gold_s} (id={gold_tok}, region={gr_gold})")
    lines.append(f"**Base top-1:** {b1_s} (id={b1_tok}, region={gr_b1})")
    lines.append(f"**Refined top-1:** {r1_s} (id={r1_tok}, region={gr_r1})")
    lines.append(f"**Gold rank:** base={grb:.0f}  refined={grr:.0f}")
    lines.append(f"**Pair margin:** base={pmb:.3f}  refined={pmr:.3f}\n")

    # Top-10 table from stored delta
    K  = val_data["topk_ids"].shape[1]
    topk_row = val_data["topk_ids"][row_i]
    lgt_row  = val_data["topk_lgt"][row_i]
    dt_arr   = row_results["delta_all"][row_i]

    lines.append("| Rank | Token | ID | Region | Delta | Base logit |")
    lines.append("|------|-------|----|--------|-------|------------|")
    for r in range(min(10, K)):
        tid = int(topk_row[r])
        try: ts = tokenizer.decode([tid])
        except: ts = str(tid)
        reg = int(tok_arr_t[min(tid, vs-1)])
        bl  = float(lgt_row[r])
        dt  = float(dt_arr[r]) if r < len(dt_arr) else 0.0
        same = "✓" if tid == gold_tok else ""
        lines.append(f"| {r+1} | `{ts}` | {tid} | {reg} | {dt:.4f} | {bl:.3f} | {same}")

    lines.append("")
    return "\n".join(lines) + "\n"


def write_example_reports(model, val_data, tok_arr_t, reg_arr_t,
                          unk_region, unk_super, sr_enabled,
                          args, device, tokenizer, subsets, row_results, output_dir):
    n_max = args.num_examples

    covered  = row_results["covered"]
    bw       = row_results["base_wrong"]
    ctg      = row_results["changed_to_gold"]
    caw      = row_results["changed_away"]
    sr       = row_results["same_reg"]
    ss       = row_results["same_sup"]
    gri      = row_results["gold_rank_improved"]
    pmr      = row_results["pair_margin_ref"]
    pmb      = row_results["pair_margin_base"]

    buckets = {
        "flipped_to_gold": np.where(ctg)[0],
        "failed_same_region": np.where(
            bw & covered & (sr | ss) & ~ctg)[0],
        "changed_away": np.where(caw)[0],
        "pair_margin_improved": np.where(
            bw & covered & np.isfinite(pmr) & np.isfinite(pmb) & (pmr > pmb))[0],
    }

    common_kw = dict(
        val_data=val_data, row_results=row_results, model=model,
        tok_arr_t=tok_arr_t, reg_arr_t=reg_arr_t,
        unk_region=unk_region, unk_super=unk_super, sr_enabled=sr_enabled,
        tokenizer=tokenizer, args=args, device=device,
    )

    titles = {
        "flipped_to_gold": "Flipped to Gold",
        "failed_same_region": "Failed Same-Region Confuser",
        "changed_away": "Changed Away from Gold",
        "pair_margin_improved": "Pair Margin Improved (not yet gold)",
    }

    for bname, indices in buckets.items():
        rng = np.random.default_rng(42)
        if len(indices) > n_max:
            indices = rng.choice(indices, n_max, replace=False)
        path = os.path.join(output_dir, f"examples_{bname}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {titles[bname]}\n\n")
            f.write(f"_{len(indices)} examples_\n\n---\n\n")
            for ex_i, row_i in enumerate(indices):
                f.write(_fmt_example(
                    ex_i + 1, int(row_i), **common_kw))
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

def write_report(run_dir, args, train_filter_stats, best_metrics,
                 final_metrics, best_step):
    path = os.path.join(run_dir, "report.md")
    m_all  = final_metrics.get("all", {})
    m_src  = final_metrics.get("same_region_or_superregion_confuser", {})
    m_bw   = final_metrics.get("base_wrong_covered", {})

    def f(v, fmt=".4f"):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "nan"
        if isinstance(v, float):
            return format(v, fmt)
        return str(v)

    lines = []
    lines.append("# Region Pairwise Reranker V1 — Report\n")
    lines.append(f"**Run:** `{args.run_name}`  |  **Steps:** {args.steps}  "
                 f"|  **Filter:** `{args.train_filter}`\n")

    lines.append("## Motivation\n")
    lines.append("Region Gold Rank Audit on the val set showed:\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append("| covered_rate | 0.8807 |")
    lines.append("| median_gold_rank_1b | 2.0 |")
    lines.append("| top1_same_region_rate | 0.5801 |")
    lines.append("| base_correct_rate | 0.3638 |\n")
    lines.append("Gold is frequently near the top and often in the same region as base top-1."
                 " This motivates pairwise reranking without retrieval.\n")

    lines.append("## Training Filter Stats\n")
    for k, v in train_filter_stats.items():
        lines.append(f"  {k:35s} = {f(v)}")
    lines.append("")

    lines.append("## Best Checkpoint\n")
    lines.append(f"Step: {best_step}  |  "
                 f"Metric: `{_BEST_METRIC_KEY}` = {f(best_metrics.get(_BEST_METRIC_KEY, float('nan')))}\n")

    lines.append("## Final Val Metrics\n")
    lines.append("### All rows\n")
    for k in ["n", "top1_acc_base", "top1_acc_refined", "top1_acc_gain",
               "candidate_ce_base", "candidate_ce_refined", "candidate_ce_gain",
               "mean_gold_rank_base", "median_gold_rank_base",
               "mean_gold_rank_refined", "median_gold_rank_refined",
               "mean_delta_abs", "max_delta_abs", "alpha"]:
        lines.append(f"  {k:40s} = {f(m_all.get(k, float('nan')))}")
    lines.append("")

    lines.append("### same_region_or_superregion_confuser\n")
    for k in ["n", "changed_to_gold_rate", "changed_away_rate",
               "pairwise_win_rate_base", "pairwise_win_rate_refined",
               "mean_pair_margin_base", "mean_pair_margin_refined",
               "mean_pair_margin_gain",
               "gold_rank_improved_rate", "gold_rank_worsened_rate"]:
        lines.append(f"  {k:40s} = {f(m_src.get(k, float('nan')))}")
    lines.append("")

    lines.append("## Interpretation\n")
    ctg  = m_src.get("changed_to_gold_rate", float("nan"))
    caw  = m_src.get("changed_away_rate", float("nan"))
    pwr  = m_src.get("pairwise_win_rate_refined", float("nan"))
    pwb  = m_src.get("pairwise_win_rate_base", float("nan"))
    gri  = m_src.get("gold_rank_improved_rate", float("nan"))
    grw  = m_src.get("gold_rank_worsened_rate", float("nan"))
    ce_g = m_all.get("candidate_ce_gain", float("nan"))

    def chk(cond, yes, no):
        return yes if (not math.isnan(cond) and cond) else no

    lines.append(f"**Pairwise win rate improved?**  "
                 f"base={f(pwb)}  refined={f(pwr)}  "
                 + chk(not math.isnan(pwr) and not math.isnan(pwb) and pwr > pwb + 0.01,
                        "✅ Yes", "⚠️  Marginal or no"))
    lines.append(f"**changed_to_gold > changed_away?**  "
                 f"ctg={f(ctg)}  caw={f(caw)}  "
                 + chk(not math.isnan(ctg) and not math.isnan(caw) and ctg > caw,
                        "✅ Yes (reranker corrects more than it breaks)",
                        "❌ No"))
    lines.append(f"**Gold rank improved rate > worsened?**  "
                 f"improved={f(gri)}  worsened={f(grw)}  "
                 + chk(not math.isnan(gri) and not math.isnan(grw) and gri > grw,
                        "✅ Yes", "⚠️  No"))
    lines.append(f"**Candidate CE gain?**  {f(ce_g)}  "
                 + chk(not math.isnan(ce_g) and ce_g > 0,
                        "✅ Positive (reranker improves candidate distribution)",
                        "❌ Negative or zero"))
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
            "Use a different --run_name to avoid overwriting old runs.")
    os.makedirs(run_dir, exist_ok=False)

    print("\n[backbone] Loading token embeddings from checkpoint...")
    tok_w, d_model, vocab_size = load_backbone(args.small_ckpt, device)
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    print("\n[maps] Loading region/super maps...")
    t2r, r2s, unk_region, unk_super, sr_enabled = load_maps(
        args.token_to_region, args.super_map)

    if not sr_enabled:
        print("  [INFO] superregion_enabled = false — using region-only mode.")

    tok_arr_np = build_tok_arr(t2r, unk_region, vocab_size)
    reg_arr_np = build_reg_arr(r2s, unk_super, unk_region)
    tok_arr_t  = torch.from_numpy(tok_arr_np).long().to(device)
    reg_arr_t  = torch.from_numpy(reg_arr_np).long().to(device)

    print("\n[data] Loading shards...")
    train_data = load_shards(args.train_dir, args.top_k, "train",
                             max_rows=args.max_train_rows, load_ids=False)
    val_data   = load_shards(args.val_dir,   args.top_k, "val",
                             max_rows=args.max_val_rows,   load_ids=True)

    print(f"\n[filter] Applying filter: {args.train_filter}")
    train_idx, filter_stats = apply_filter(
        train_data, args.train_filter, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled)
    print(f"  Kept {filter_stats['rows_kept']:,} / {filter_stats['total_rows']:,} "
          f"({filter_stats['kept_rate']:.3f})")
    for k, v in filter_stats.items():
        print(f"    {k:35s} = {v:.4f}" if isinstance(v, float) else
              f"    {k:35s} = {v}")

    if len(train_idx) == 0:
        raise RuntimeError("No training rows after filter. Check filter config.")

    # Build model
    n_regions = unk_region          # = max_region + 1; Embedding needs +2 for UNK+pad
    n_supers  = unk_super if sr_enabled else 1

    print("\n[model] Building RegionPairwiseReranker...")
    model = RegionPairwiseReranker(
        token_emb_weight=tok_w.to(device),
        tok_arr=tok_arr_np, reg_arr=reg_arr_np,
        d_model=d_model, n_regions=n_regions, n_supers=n_supers,
        sr_enabled=sr_enabled,
        resolver_dim=args.resolver_dim, resolver_layers=args.resolver_layers,
        resolver_heads=args.resolver_heads,
        region_emb_dim=args.region_emb_dim, super_emb_dim=args.super_emb_dim,
        delta_scale=args.delta_scale, top_k=args.top_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Save config
    config = vars(args).copy()
    config.update({
        "d_model": d_model, "vocab_size": vocab_size,
        "n_regions": n_regions, "n_supers": n_supers,
        "sr_enabled": sr_enabled, "n_params": n_params,
        "superregion_enabled": sr_enabled,
        "train_filter_stats": filter_stats,
    })
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Identity check
    print("\n[identity] Running identity check...")
    run_identity_check(model, val_data, tok_arr_t, reg_arr_t,
                       unk_region, unk_super, sr_enabled, args, device, run_dir)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    train_log_path  = os.path.join(run_dir, "train_log.csv")
    eval_log_path   = os.path.join(run_dir, "eval_log.csv")
    subset_log_path = os.path.join(run_dir, "subset_eval_log.csv")

    best_metric_val = -float("inf")
    best_step       = 0
    eval_history    = []
    step            = 0
    header_written_train = False
    header_written_eval  = False
    header_written_sub   = False

    print(f"\n[train] Starting training: {args.steps} steps, "
          f"batch={args.batch_size}, accum={args.grad_accum_steps}, "
          f"lr={args.lr}, amp={args.amp}")
    print(f"  Primary metric: {_BEST_METRIC_KEY}\n")

    N_train = len(train_idx)
    perm    = torch.randperm(N_train)
    perm_pos = 0

    optimizer.zero_grad()
    accum_count = 0
    t0 = time.time()

    while step < args.steps:
        if perm_pos + args.batch_size > N_train:
            perm = torch.randperm(N_train)
            perm_pos = 0

        batch_local_idx = perm[perm_pos: perm_pos + args.batch_size]
        perm_pos += args.batch_size
        batch_global_idx = train_idx[batch_local_idx]

        hp   = train_data["h_prime"][batch_global_idx].to(device)
        topk = train_data["topk_ids"][batch_global_idx].to(device)
        lgt  = train_data["topk_lgt"][batch_global_idx].to(device)
        gold = train_data["gold"][batch_global_idx].to(device)
        B    = hp.shape[0]

        # Derived fields
        gold_in_topk = (topk == gold.unsqueeze(1))
        covered = gold_in_topk.any(1)
        gold_idx = gold_in_topk.long().argmax(1)

        cand_regs, cand_sups = _lookup_regs(
            topk, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled)

        arange_b = torch.arange(B, device=device)
        vs = tok_arr_t.shape[0]
        gold_safe_b = gold.clamp(0, vs - 1)
        gold_regs_b = tok_arr_t[gold_safe_b]
        if sr_enabled and cand_sups is not None:
            rlen = reg_arr_t.shape[0] - 1
            gold_sups_b = reg_arr_t[gold_regs_b.clamp(0, rlen)]
        else:
            gold_sups_b = torch.zeros(B, dtype=torch.long, device=device)

        with _amp_ctx(args.amp):
            ref_lgt, delta = model(hp, topk, lgt, cand_regs, cand_sups)
            total_loss, loss_dict = _compute_losses(
                ref_lgt, delta, lgt, gold_idx, covered,
                cand_regs, cand_sups, gold_regs_b, gold_sups_b,
                unk_region, unk_super, sr_enabled, args, device)
            loss_scaled = total_loss / args.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        accum_count += 1
        if accum_count < args.grad_accum_steps:
            continue

        accum_count = 0
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        step += 1

        # Log
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  step={step:5d}  loss={loss_dict['total']:.4f}  "
                  f"pair={loss_dict['pair']:.4f}  "
                  f"multi={loss_dict['multi']:.4f}  "
                  f"ce={loss_dict['ce']:.4f}  "
                  f"kl={loss_dict['kl']:.4f}  "
                  f"alpha={float(model.alpha.item()):.4f}  "
                  f"t={elapsed:.0f}s")

        train_row = {"step": step, **loss_dict,
                     "alpha": float(model.alpha.item()),
                     "n_train_rows": len(train_idx)}
        _write_csv_row(train_log_path, train_row, write_header=(step == 1))

        # Eval
        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n[eval] step={step}")
            subsets, row_results = evaluate(
                model, val_data, tok_arr_t, reg_arr_t,
                unk_region, unk_super, sr_enabled,
                args, device, None, step, run_dir)

            fv = {}
            if args.eval_full_vocab:
                fv = eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                                     unk_region, unk_super, sr_enabled, args, device)

            # Print key metrics
            m_src = subsets.get("same_region_or_superregion_confuser", {})
            m_all = subsets.get("all", {})
            print(f"  [all]    ce_base={m_all.get('candidate_ce_base', float('nan')):.4f}  "
                  f"top1_acc_base={m_all.get('top1_acc_base', float('nan')):.4f}  "
                  f"top1_acc_ref={m_all.get('top1_acc_refined', float('nan')):.4f}")
            print(f"  [src]  n={m_src.get('n', 0)}  "
                  f"pwr_base={m_src.get('pairwise_win_rate_base', float('nan')):.4f}  "
                  f"pwr_ref={m_src.get('pairwise_win_rate_refined', float('nan')):.4f}  "
                  f"ctg={m_src.get('changed_to_gold_rate', float('nan')):.4f}  "
                  f"caw={m_src.get('changed_away_rate', float('nan')):.4f}")
            if fv:
                print(f"  [fv] base_nll={fv.get('full_vocab_base_nll', float('nan')):.4f}  "
                      f"ref_nll={fv.get('full_vocab_refined_nll', float('nan')):.4f}  "
                      f"gain={fv.get('full_vocab_gain', float('nan')):.4f}")

            primary_val = m_src.get(_BEST_METRIC_KEY.split("_confuser_")[1]
                                     if "_confuser_" in _BEST_METRIC_KEY
                                     else _BEST_METRIC_KEY, float("nan"))
            primary_val = m_src.get("pairwise_win_rate_refined", float("nan"))

            # Save latest
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "subsets": subsets,
                "fv": fv,
            }, os.path.join(run_dir, "latest_reranker.pt"))

            # Save best
            if not math.isnan(primary_val) and primary_val > best_metric_val:
                best_metric_val = primary_val
                best_step = step
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "subsets": subsets,
                    "fv": fv,
                }, os.path.join(run_dir, "best_reranker.pt"))
                print(f"  [best] new best at step {step}: {_BEST_METRIC_KEY}={primary_val:.4f}")
                with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
                    json.dump({"step": step, "metric": primary_val,
                               "subsets": _json_safe(subsets), "fv": _json_safe(fv)},
                              f, indent=2)

            # CSV
            eval_flat = {"step": step, **fv}
            for sname, sm in subsets.items():
                for k, v in sm.items():
                    eval_flat[f"{sname}_{k}"] = v
            _write_csv_row(eval_log_path, eval_flat,
                           write_header=not header_written_eval)
            header_written_eval = True

            for sname, sm in subsets.items():
                sub_row = {"step": step, "subset": sname, **sm}
                _write_csv_row(subset_log_path, sub_row,
                               write_header=not header_written_sub)
                header_written_sub = True

            eval_history.append({"step": step, "subsets": subsets, "fv": fv})
            print()

    # Final eval
    print("[final eval]")
    final_subsets, final_row = evaluate(
        model, val_data, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled,
        args, device, None, args.steps, run_dir,
        write_examples=True)

    fv_final = {}
    if args.eval_full_vocab:
        fv_final = eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                                   unk_region, unk_super, sr_enabled, args, device)

    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump(_json_safe({"step": args.steps,
                              "subsets": final_subsets, "fv": fv_final}), f, indent=2)

    # Example reports
    try:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            tokenizer = type("FakeTok", (), {"decode": lambda self, ids, **kw: str(ids)})()

        write_example_reports(model, val_data, tok_arr_t, reg_arr_t,
                              unk_region, unk_super, sr_enabled,
                              args, device, tokenizer,
                              final_subsets, final_row, run_dir)
    except Exception as e:
        print(f"[warn] Example report writing failed: {e}")

    # Load best model metrics for report
    bm_path = os.path.join(run_dir, "best_metrics.json")
    best_metrics_for_report = {}
    if os.path.isfile(bm_path):
        with open(bm_path) as f:
            bd = json.load(f)
        best_metrics_for_report = {
            k: v for s in bd.get("subsets", {}).values()
            for k, v in s.items()
        }
        best_metrics_for_report[_BEST_METRIC_KEY] = bd.get("metric", float("nan"))

    write_report(run_dir, args, filter_stats,
                 best_metrics_for_report, final_subsets, best_step)

    print(f"\n{'='*60}")
    print(f" Region Pairwise Reranker V1 complete.")
    print(f" Run dir: {run_dir}")
    print(f" Best step: {best_step}  "
          f"{_BEST_METRIC_KEY}={best_metric_val:.4f}")
    print(f"{'='*60}\n")


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


def _parse():
    p = argparse.ArgumentParser(description="Region Pairwise Reranker V1")
    p.add_argument("--small_ckpt",       required=True)
    p.add_argument("--train_dir",        required=True)
    p.add_argument("--val_dir",          required=True)
    p.add_argument("--token_to_region",  required=True)
    p.add_argument("--super_map",        default=None)
    p.add_argument("--output_root",      default="runs/region_pairwise_reranker_v1")
    p.add_argument("--run_name",         default="same_region_pair_v1")

    p.add_argument("--top_k",            type=int, default=256)
    p.add_argument("--max_train_rows",   type=int, default=None)
    p.add_argument("--max_val_rows",     type=int, default=None)
    p.add_argument("--train_filter",     default="same_region_or_superregion_confuser",
                   choices=_TRAIN_FILTERS)
    p.add_argument("--num_confusers",    type=int, default=8)
    p.add_argument("--num_examples",     type=int, default=50)

    p.add_argument("--resolver_dim",     type=int,   default=256)
    p.add_argument("--resolver_layers",  type=int,   default=2)
    p.add_argument("--resolver_heads",   type=int,   default=4)
    p.add_argument("--region_emb_dim",   type=int,   default=64)
    p.add_argument("--super_emb_dim",    type=int,   default=32)
    p.add_argument("--delta_scale",      type=float, default=0.25)

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

    p.add_argument("--lambda_pair",      type=float, default=2.0)
    p.add_argument("--lambda_multi",     type=float, default=0.5)
    p.add_argument("--lambda_ce",        type=float, default=0.25)
    p.add_argument("--lambda_kl",        type=float, default=1.0)
    p.add_argument("--lambda_delta",     type=float, default=1e-3)

    return p.parse_args()


if __name__ == "__main__":
    main()
