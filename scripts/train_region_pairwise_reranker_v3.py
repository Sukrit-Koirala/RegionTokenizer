#!/usr/bin/env python3
"""
train_region_pairwise_reranker_v3.py — Surgical Pairwise Correction (Oracle Diagnostic)

V3 hypothesis: broad top-256 logit editing caused V1/V2 collateral damage.
Test: surgically edit ONLY gold vs base_top1 logits on known confuser rows.

  actual_delta applied: +0.5*md to gold, -0.5*md to base_top1 (all others unchanged)

ORACLE DIAGNOSTIC NOTE:
  apply_policy=target_only uses the gold label to identify which rows to correct.
  This is NOT deployable at inference time.
  Purpose: test whether pairwise features can learn surgical correction without collateral.
  If yes → problem is candidate selection, not the pairwise signal itself.

V1/V2 reference:
  V1: changed_to_gold=0.1018  pairwise_win_ref=0.1379  fv_gain=-0.0050
  V2: changed_to_gold=0.0193  pairwise_win_ref=0.0299  fv_gain=-0.0105  (gate saturated)
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

_APPLY_POLICIES = ["target_only", "all_base_wrong_covered", "all_covered"]

_SUBSET_ORDER_V3 = [
    "all", "covered", "base_correct_covered",
    "base_wrong_covered", "target_confuser",
    "same_region_confuser", "same_superregion_confuser", "base_miss",
]


# ─────────────────────────────────────────────────────────────────────────────
# Backbone / maps / shards  (same as V2)
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(ckpt_path: str, device: torch.device):
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        ckpt_path, device)
    tok_w = backbone.token_emb.weight.detach().float()
    del backbone
    return tok_w, d_model, vocab_size


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
        r2s = {}; unk_super = 0; sr_enabled = False
    print(f"  maps: n_regions={unk_region}  unk_region={unk_region}  "
          f"sr_enabled={sr_enabled}  unk_super={unk_super}")
    return t2r, r2s, unk_region, unk_super, sr_enabled


def build_tok_arr(t2r, unk_region: int, vocab_size: int) -> np.ndarray:
    arr = np.full(vocab_size, unk_region, dtype=np.int32)
    for tok, reg in t2r.items():
        if 0 <= tok < vocab_size:
            arr[tok] = int(reg)
    return arr


def build_reg_arr(r2s, unk_super: int, unk_region: int) -> np.ndarray:
    arr = np.full(unk_region + 1, unk_super, dtype=np.int32)
    for reg, sup in r2s.items():
        if 0 <= reg <= unk_region:
            arr[reg] = int(sup)
    return arr


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


def build_train_pools(data, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled):
    gold = data["gold"]; topk = data["topk_ids"]
    tok_cpu = tok_arr_t.cpu(); reg_cpu = reg_arr_t.cpu()
    vs = tok_cpu.shape[0]
    covered    = (topk == gold.unsqueeze(1)).any(1)
    base_wrong = topk[:, 0] != gold
    gold_reg   = tok_cpu[gold.clamp(0, vs-1)]
    top1_reg   = tok_cpu[topk[:, 0].clamp(0, vs-1)]
    same_reg   = (gold_reg == top1_reg) & (gold_reg != unk_region) & (top1_reg != unk_region)
    if sr_enabled:
        rlen = reg_cpu.shape[0] - 1
        gold_sup = reg_cpu[gold_reg.clamp(0, rlen)]
        top1_sup = reg_cpu[top1_reg.clamp(0, rlen)]
        same_sup = (gold_sup == top1_sup) & (gold_sup != unk_super) & (top1_sup != unk_super)
    else:
        same_sup = torch.zeros(len(gold), dtype=torch.bool)
    target_idx  = (covered & base_wrong & (same_reg | same_sup)).nonzero(as_tuple=False).squeeze(1)
    noharm_idx  = (covered & ~base_wrong).nonzero(as_tuple=False).squeeze(1)
    other_idx   = (covered & base_wrong & ~(same_reg | same_sup)).nonzero(as_tuple=False).squeeze(1)
    N = len(gold)
    stats = {
        "total_rows":        N,
        "target_rows":       len(target_idx),
        "noharm_rows":       len(noharm_idx),
        "other_wrong_rows":  len(other_idx),
        "covered_rate":      float(covered.float().mean()),
        "target_rate":       len(target_idx) / N,
        "noharm_rate":       len(noharm_idx) / N,
        "same_region_rate":  float(same_reg.float().mean()),
        "same_super_rate":   float(same_sup.float().mean()) if sr_enabled else 0.0,
        "gold_unmapped_rate":float((tok_cpu[gold.clamp(0,vs-1)] == unk_region).float().mean()),
    }
    return target_idx, noharm_idx, other_idx, stats


def _lookup_regs(token_ids, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled):
    vs   = tok_arr_t.shape[0]
    safe = token_ids.clamp(0, vs - 1)
    regs = tok_arr_t[safe]
    sups = reg_arr_t[regs.clamp(0, reg_arr_t.shape[0]-1)] if sr_enabled else None
    return regs, sups


# ─────────────────────────────────────────────────────────────────────────────
# Model V3: Surgical Pair MLP
# ─────────────────────────────────────────────────────────────────────────────

class SurgicalPairwiseReranker(nn.Module):
    """Produces a scalar margin_delta per row from gold-vs-base_top1 features.

    Correction applied surgically:
      refined[gold_idx] += +0.5 * margin_delta
      refined[0]        += -0.5 * margin_delta
      all others unchanged
    """

    def __init__(self, token_emb_weight: torch.Tensor,
                 tok_arr: np.ndarray, reg_arr: np.ndarray,
                 d_model: int, n_regions: int, n_supers: int,
                 sr_enabled: bool, unk_region: int, unk_super: int,
                 region_emb_dim: int = 64, super_emb_dim: int = 32,
                 hidden_dim: int = 256, n_hidden: int = 3,
                 margin_delta_scale: float = 1.0,
                 top_k: int = 256):
        super().__init__()
        self.d_model           = d_model
        self.sr_enabled        = sr_enabled
        self.unk_region        = unk_region
        self.unk_super         = unk_super
        self.margin_delta_scale = margin_delta_scale
        self.top_k             = top_k

        self.register_buffer("token_emb_weight",
                             token_emb_weight.detach().float())
        self.register_buffer("tok_arr",
                             torch.from_numpy(tok_arr.astype(np.int32)).long())
        self.register_buffer("reg_arr",
                             torch.from_numpy(reg_arr.astype(np.int32)).long())

        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim)
        if sr_enabled:
            self.super_emb  = nn.Embedding(n_supers + 2, super_emb_dim)

        # Feature dim:
        # 3*d_model (tok_g, tok_b, tok_g-tok_b)
        # 3*region_emb_dim (reg_g, reg_b, reg_g-reg_b)
        # d_model (h_prime)
        # scalars: lgt_g, lgt_b, base_margin, gold_rank_norm,
        #          same_reg, hp_dot_g, hp_dot_b, hp_dot_margin  → 8
        feat_dim = 4 * d_model + 3 * region_emb_dim + 8
        if sr_enabled:
            feat_dim += 2 * super_emb_dim + 1   # sup_g, sup_b, same_sup

        layers = []
        in_dim = feat_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        self.pair_mlp = nn.Sequential(*layers)

        # Output: zero-init → margin_delta = 0 at step 0
        self.delta_head = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, h_prime, candidate_ids, base_logits, gold_idx, cand_regs, cand_sups=None):
        """
        Returns margin_delta [B] — positive correction pushes gold above base_top1.
        Caller applies surgically.
        """
        B, K    = candidate_ids.shape
        arange  = torch.arange(B, device=h_prime.device)
        gold_s  = gold_idx.clamp(0, K - 1)
        vs      = self.token_emb_weight.shape[0]

        # Token IDs
        gold_ids = candidate_ids[arange, gold_s].clamp(0, vs - 1)
        base_ids = candidate_ids[:, 0].clamp(0, vs - 1)

        tok_g = self.token_emb_weight[gold_ids]   # [B, d]
        tok_b = self.token_emb_weight[base_ids]   # [B, d]

        # Region embeddings
        reg_g = cand_regs[arange, gold_s]         # [B]
        reg_b = cand_regs[:, 0]                   # [B]
        remb_g = self.region_emb(reg_g)
        remb_b = self.region_emb(reg_b)

        # Scalar features
        lgt_g = base_logits[arange, gold_s]       # [B]
        lgt_b = base_logits[:, 0]                 # [B]
        base_margin = lgt_g - lgt_b
        gold_rank_n = gold_s.float() / K

        same_reg_s = ((reg_g == reg_b) & (reg_g != self.unk_region)).float()

        hp_dot_g  = (h_prime * tok_g).sum(-1)
        hp_dot_b  = (h_prime * tok_b).sum(-1)
        hp_dot_mg = hp_dot_g - hp_dot_b

        parts = [
            tok_g, tok_b, tok_g - tok_b,
            remb_g, remb_b, remb_g - remb_b,
            h_prime,
            lgt_g.unsqueeze(1), lgt_b.unsqueeze(1),
            base_margin.unsqueeze(1), gold_rank_n.unsqueeze(1),
            same_reg_s.unsqueeze(1),
            hp_dot_g.unsqueeze(1), hp_dot_b.unsqueeze(1),
            hp_dot_mg.unsqueeze(1),
        ]
        if self.sr_enabled and cand_sups is not None:
            sup_g  = cand_sups[arange, gold_s]
            sup_b  = cand_sups[:, 0]
            semb_g = self.super_emb(sup_g)
            semb_b = self.super_emb(sup_b)
            same_sup_s = ((sup_g == sup_b) & (sup_g != self.unk_super)).float()
            parts.extend([semb_g, semb_b, same_sup_s.unsqueeze(1)])

        feat    = torch.cat(parts, dim=1)
        hidden  = self.pair_mlp(feat)
        raw     = self.delta_head(hidden).squeeze(1)
        return self.margin_delta_scale * torch.tanh(raw)           # [B]


def apply_correction(base_logits, margin_delta, gold_idx, apply_mask):
    """Surgically edit gold and base_top1 for rows in apply_mask."""
    B, K    = base_logits.shape
    refined = base_logits.clone()
    if not apply_mask.any():
        return refined
    ar = torch.arange(B, device=base_logits.device)
    gs = gold_idx.clamp(0, K - 1)
    # Safety: skip rows where gold IS base_top1 (shouldn't happen for target rows)
    valid = apply_mask & (gs != 0)
    if valid.any():
        md = margin_delta[valid]
        refined[ar[valid], gs[valid]] += 0.5 * md
        refined[ar[valid], 0]         -= 0.5 * md
    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Loss V3 (simplified: no KL, no all-token delta)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_losses_v3(refined_logits, margin_delta, gold_idx, covered,
                        base_logits, args, device):
    B, K = refined_logits.shape
    arange = torch.arange(B, device=device)
    gs = gold_idx.clamp(0, K - 1)

    # ── 1. Pairwise on all target rows in batch ───────────────────────────────
    pair_mask = covered & (gs != 0)      # all target rows should have covered & base_wrong
    if pair_mask.any():
        margin_ref = refined_logits[pair_mask, gs[pair_mask]] - refined_logits[pair_mask, 0]
        L_pair = F.softplus(args.target_margin - margin_ref).mean()
    else:
        L_pair = torch.tensor(0.0, device=device)

    # ── 2. Positive correction penalty (penalize negative margin_delta) ───────
    if not args.allow_signed_delta:
        L_positive = F.relu(-margin_delta).mean()
    else:
        L_positive = torch.tensor(0.0, device=device)

    # ── 3. Delta magnitude penalty ────────────────────────────────────────────
    L_delta = (margin_delta ** 2).mean()

    total = (args.lambda_pair         * L_pair
           + args.lambda_positive     * L_positive
           + args.lambda_margin_delta * L_delta)

    return total, {
        "pair":         L_pair.item(),
        "positive":     L_positive.item(),
        "delta":        L_delta.item(),
        "total":        total.item(),
        "n_pair_rows":  int(pair_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Identity check
# ─────────────────────────────────────────────────────────────────────────────

def run_identity_check(model, val_data, tok_arr_t, reg_arr_t,
                       unk_region, unk_super, sr_enabled,
                       args, device, output_dir):
    model.eval()
    n    = min(64, val_data["h_prime"].shape[0])
    K    = val_data["topk_ids"].shape[1]
    hp   = val_data["h_prime"][:n].to(device)
    topk = val_data["topk_ids"][:n].to(device)
    lgt  = val_data["topk_lgt"][:n].to(device)
    gold = val_data["gold"][:n].to(device)

    cr, cs = _lookup_regs(topk, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled)
    gold_in = (topk == gold.unsqueeze(1))
    covered = gold_in.any(1)
    gold_idx_b = gold_in.long().argmax(1)

    with torch.no_grad():
        md = model(hp, topk, lgt, gold_idx_b, cr, cs)

    md_max  = float(md.abs().max())
    # No correction applied yet — just check raw md is near zero
    passed  = md_max < 1e-6
    result  = {
        "margin_delta_abs_max":     md_max,
        "margin_delta_mean":        float(md.mean()),
        "passed":                   passed,
    }
    with open(os.path.join(output_dir, "identity_check.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[identity] margin_delta_abs_max={md_max:.2e}  passed={passed}")
    if not passed:
        raise RuntimeError("Identity check FAILED. Aborting.")
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


def _rate(arr_bool, denom):
    if denom == 0:
        return float("nan")
    return float(np.asarray(arr_bool, dtype=float).sum()) / denom


def _subset_v3(mask, row):
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}

    def M(k):
        return np.asarray(row[k])[mask]

    cov  = M("covered")
    bw   = M("base_wrong")
    app  = M("applied")
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
    md   = M("margin_delta")

    bwc  = bw & cov
    n_bwc = int(bwc.sum())
    n_cov = int(cov.sum())
    n_app = int(app.sum())

    return {
        "n":                          n,
        "applied_rate":               _rate(app, n),
        "covered_rate":               _rate(cov, n),
        "base_wrong_rate":            _rate(bw, n),
        "mean_margin_delta":          _safe_mean(md[app]) if n_app > 0 else float("nan"),
        "mean_abs_margin_delta":      _safe_mean(np.abs(md[app])) if n_app > 0 else float("nan"),
        "candidate_ce_base":          _safe_mean(ceb[cov]),
        "candidate_ce_refined":       _safe_mean(cer[cov]),
        "candidate_ce_gain":          _safe_mean((ceb - cer)[cov]),
        "top1_acc_base":              _rate(~bw, n),
        "top1_acc_refined":           _rate(~rw, n),
        "top1_acc_gain":              _rate(~rw, n) - _rate(~bw, n),
        "changed_to_gold_rate":       _rate(ctg[bwc], n_bwc) if n_bwc > 0 else float("nan"),
        "changed_away_rate":          _rate(caw[~bw], int((~bw).sum())) if (~bw).any() else float("nan"),
        "gold_rank_improved_rate":    _rate(gri[cov], n_cov),
        "gold_rank_worsened_rate":    _rate(grw[cov], n_cov),
        "mean_gold_rank_base":        _safe_mean(grb[cov]),
        "mean_gold_rank_refined":     _safe_mean(grr[cov]),
        "median_gold_rank_base":      _safe_median(grb[cov]),
        "median_gold_rank_refined":   _safe_median(grr[cov]),
        "pairwise_win_rate_base":     _rate(pwb[bwc], n_bwc) if n_bwc > 0 else float("nan"),
        "pairwise_win_rate_refined":  _rate(pwr[bwc], n_bwc) if n_bwc > 0 else float("nan"),
        "mean_pair_margin_base":      _safe_mean(pmb[bwc]),
        "mean_pair_margin_refined":   _safe_mean(pmr[bwc]),
        "mean_pair_margin_gain":      _safe_mean((pmr - pmb)[bwc]),
    }


_SUBSET_DEFS_V3 = {
    "all":                        lambda d: np.ones(d["N"], dtype=bool),
    "covered":                    lambda d: d["covered"],
    "base_correct_covered":       lambda d: d["covered"] & ~d["base_wrong"],
    "base_wrong_covered":         lambda d: d["covered"] & d["base_wrong"],
    "target_confuser":            lambda d: d["covered"] & d["base_wrong"] & (d["same_reg"] | d["same_sup"]),
    "same_region_confuser":       lambda d: d["covered"] & d["base_wrong"] & d["same_reg"],
    "same_superregion_confuser":  lambda d: d["covered"] & d["base_wrong"] & d["same_sup"],
    "base_miss":                  lambda d: ~d["covered"],
}


def evaluate(model, val_data, tok_arr_t, reg_arr_t,
             unk_region, unk_super, sr_enabled, args, device):
    model.eval()
    N, K = val_data["h_prime"].shape[0], val_data["topk_ids"].shape[1]
    BSZ  = args.eval_batch_size

    float_nan = float("nan")
    row = {k: np.full(N, float_nan, dtype=np.float32) for k in [
        "gold_rank_base_1b", "gold_rank_ref_1b",
        "pair_margin_base", "pair_margin_ref",
        "ce_base", "ce_ref", "margin_delta",
    ]}
    for k in ["covered", "base_wrong", "same_reg", "same_sup", "applied",
              "changed_to_gold", "changed_away", "refined_wrong",
              "gold_rank_improved", "gold_rank_worsened",
              "pairwise_win_base", "pairwise_win_ref"]:
        row[k] = np.zeros(N, dtype=bool)

    # For examples
    row["gold_tok"]       = np.zeros(N, dtype=np.int64)
    row["base_top1_tok"]  = np.zeros(N, dtype=np.int64)
    row["ref_top1_tok"]   = np.zeros(N, dtype=np.int64)
    row["gold_idx"]       = np.zeros(N, dtype=np.int32)
    row["base_lgt_top10"] = np.zeros((N, min(10,K)), dtype=np.float32)
    row["ref_lgt_top10"]  = np.zeros((N, min(10,K)), dtype=np.float32)

    def np_(t): return t.cpu().numpy()

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
            gold_in = (topk == gold.unsqueeze(1))
            cov_b   = gold_in.any(1)
            gidx_b  = gold_in.long().argmax(1)
            gsafe   = gidx_b.clamp(0, K-1)

            # Region/super for subset masks
            vs = tok_arr_t.shape[0]
            gold_reg = tok_arr_t[gold.clamp(0, vs-1)]
            top1_reg = cr[:, 0]
            sr_b = (gold_reg == top1_reg) & (gold_reg != unk_region)
            if sr_enabled and cs is not None:
                rlen = reg_arr_t.shape[0] - 1
                gold_sup = reg_arr_t[gold_reg.clamp(0, rlen)]
                top1_sup = cs[:, 0]
                ss_b = (gold_sup == top1_sup) & (gold_sup != unk_super)
            else:
                ss_b = torch.zeros(b, dtype=torch.bool, device=device)

            # apply_mask
            if args.apply_policy == "target_only":
                apply_b = cov_b & (topk[:, 0] != gold) & (sr_b | ss_b)
            elif args.apply_policy == "all_base_wrong_covered":
                apply_b = cov_b & (topk[:, 0] != gold)
            else:
                apply_b = cov_b

            # Model forward
            md_b = model(hp, topk, lgt, gidx_b, cr, cs)   # [b]

            # Apply correction
            ref = apply_correction(lgt, md_b, gsafe, apply_b)

            # Metrics
            bw_b     = topk[:, 0] != gold
            base_t1  = topk[:, 0]
            ref_t1i  = ref.argmax(1)
            ref_t1   = topk[ar_b, ref_t1i]

            n_higher_ref = (ref > ref[ar_b, gsafe].unsqueeze(1)).sum(1)
            grb_b   = torch.where(cov_b, gidx_b.long(), torch.tensor(-1, device=device))
            grr_b   = torch.where(cov_b, n_higher_ref.long(), torch.tensor(-1, device=device))

            pmb_b   = lgt[ar_b, gsafe] - lgt[:, 0]
            pmr_b   = ref[ar_b, gsafe] - ref[:, 0]
            ceb_b   = -F.log_softmax(lgt, 1)[ar_b, gsafe]
            cer_b   = -F.log_softmax(ref, 1)[ar_b, gsafe]

            ctg_b   = bw_b & cov_b & (ref_t1 == gold)
            caw_b   = ~bw_b & (ref_t1 != gold)
            rw_b    = ref_t1 != gold
            gri_b   = cov_b & (grr_b < grb_b) & (grr_b >= 0)
            grw_b   = cov_b & (grr_b > grb_b)
            pwb_b   = bw_b & cov_b & (pmb_b > 0)
            pwr_b   = bw_b & cov_b & (pmr_b > 0)

            sl = slice(s, e)
            row["covered"][sl]          = np_(cov_b)
            row["base_wrong"][sl]       = np_(bw_b)
            row["same_reg"][sl]         = np_(sr_b)
            row["same_sup"][sl]         = np_(ss_b)
            row["applied"][sl]          = np_(apply_b)
            row["changed_to_gold"][sl]  = np_(ctg_b)
            row["changed_away"][sl]     = np_(caw_b)
            row["refined_wrong"][sl]    = np_(rw_b)
            row["gold_rank_improved"][sl] = np_(gri_b)
            row["gold_rank_worsened"][sl] = np_(grw_b)
            row["pairwise_win_base"][sl]= np_(pwb_b)
            row["pairwise_win_ref"][sl] = np_(pwr_b)
            row["margin_delta"][sl]     = np_(md_b)
            row["gold_tok"][sl]         = np_(gold)
            row["base_top1_tok"][sl]    = np_(base_t1)
            row["ref_top1_tok"][sl]     = np_(ref_t1)
            row["gold_idx"][sl]         = np_(gidx_b)
            kk = min(10, K)
            row["base_lgt_top10"][sl]   = np_(lgt[:, :kk])
            row["ref_lgt_top10"][sl]    = np_(ref[:, :kk])

            for rk, arr, valid in [
                ("gold_rank_base_1b", grb_b + 1, cov_b),
                ("gold_rank_ref_1b",  grr_b + 1, cov_b),
                ("pair_margin_base",  pmb_b, bw_b & cov_b),
                ("pair_margin_ref",   pmr_b, bw_b & cov_b),
                ("ce_base",  ceb_b, cov_b),
                ("ce_ref",   cer_b, cov_b),
            ]:
                vals = np_(arr.float())
                vals[~np_(valid)] = float_nan
                row[rk][sl] = vals

    row["N"] = N
    subsets = {}
    for name, fn in _SUBSET_DEFS_V3.items():
        if name == "same_superregion_confuser" and not sr_enabled:
            continue
        m = fn(row)
        subsets[name] = _subset_v3(m, row)

    # Checkpoint score
    ctg  = subsets.get("target_confuser", {}).get("changed_to_gold_rate", float("nan"))
    pwr  = subsets.get("target_confuser", {}).get("pairwise_win_rate_refined", float("nan"))
    caw  = subsets.get("base_correct_covered", {}).get("changed_away_rate", float("nan"))
    score_parts = [ctg, pwr, caw]
    if not any(math.isnan(v) for v in score_parts):
        score = ctg + 0.25 * pwr - 2.0 * caw
    else:
        score = float("nan")
    subsets["_checkpoint_score"] = score

    model.train()
    return subsets, row


# ─────────────────────────────────────────────────────────────────────────────
# Full-vocab eval (surgical: only two logits change)
# ─────────────────────────────────────────────────────────────────────────────

def eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                    unk_region, unk_super, sr_enabled, args, device):
    try:
        model.eval()
        N, K = val_data["h_prime"].shape[0], val_data["topk_ids"].shape[1]
        VS   = model.token_emb_weight.shape[0]
        BSZ  = args.eval_batch_size
        tot_nll_b = tot_nll_r = tot_acc_b = tot_acc_r = 0.0

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
                gold_in = (topk == gold.unsqueeze(1))
                cov_b   = gold_in.any(1)
                gidx_b  = gold_in.long().argmax(1)
                gsafe   = gidx_b.clamp(0, K-1)

                sr_b = ((tok_arr_t[gold.clamp(0, tok_arr_t.shape[0]-1)] ==
                         cr[:, 0]) & (cr[:, 0] != unk_region))
                if sr_enabled and cs is not None:
                    rlen = reg_arr_t.shape[0] - 1
                    gs_reg = tok_arr_t[gold.clamp(0, tok_arr_t.shape[0]-1)]
                    gs_sup = reg_arr_t[gs_reg.clamp(0, rlen)]
                    ss_b   = (gs_sup == cs[:, 0]) & (gs_sup != unk_super)
                else:
                    ss_b = torch.zeros(b, dtype=torch.bool, device=device)

                if args.apply_policy == "target_only":
                    apply_b = cov_b & (topk[:, 0] != gold) & (sr_b | ss_b)
                elif args.apply_policy == "all_base_wrong_covered":
                    apply_b = cov_b & (topk[:, 0] != gold)
                else:
                    apply_b = cov_b

                md_b = model(hp, topk, lgt, gidx_b, cr, cs)

                fv_base = hp @ model.token_emb_weight.T            # [b, VS]
                fv_ref  = fv_base.clone()

                if apply_b.any():
                    gold_tok_ids  = topk[ar_b, gsafe].clamp(0, VS-1)
                    base_tok_ids  = topk[:, 0].clamp(0, VS-1)
                    md_app        = md_b * apply_b.float()         # zero for non-applied
                    fv_ref[ar_b, gold_tok_ids]  += 0.5 * md_app
                    fv_ref[ar_b, base_tok_ids]  -= 0.5 * md_app

                gs_v = gold.clamp(0, VS-1)
                tot_nll_b += F.cross_entropy(fv_base, gs_v, reduction="sum").item()
                tot_nll_r += F.cross_entropy(fv_ref,  gs_v, reduction="sum").item()
                tot_acc_b += (fv_base.argmax(1) == gs_v).sum().item()
                tot_acc_r += (fv_ref.argmax(1)  == gs_v).sum().item()

        model.train()
        return {
            "oracle_policy_full_vocab_base_nll":    tot_nll_b / N,
            "oracle_policy_full_vocab_refined_nll": tot_nll_r / N,
            "oracle_policy_full_vocab_gain":        (tot_nll_b - tot_nll_r) / N,
            "oracle_policy_full_vocab_acc_base":    tot_acc_b / N,
            "oracle_policy_full_vocab_acc_refined": tot_acc_r / N,
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


def _fmt_example_v3(ex_i, ri, val_data, row, tok_arr_t, unk_region, tokenizer):
    gt  = int(row["gold_tok"][ri])
    b1  = int(row["base_top1_tok"][ri])
    r1  = int(row["ref_top1_tok"][ri])
    grb = row["gold_rank_base_1b"][ri]
    grr = row["gold_rank_ref_1b"][ri]
    pmb = row["pair_margin_base"][ri]
    pmr = row["pair_margin_ref"][ri]
    md  = row["margin_delta"][ri]
    rid = str(val_data["rid"][ri]) if val_data.get("rid") else str(ri)

    vs = tok_arr_t.shape[0]
    def regstr(tid): return str(int(tok_arr_t[min(tid, vs-1)].item()))
    def d(tid):
        try: return f"`{tokenizer.decode([tid])}`"
        except: return str(tid)

    lines = [f"### Example {ex_i} (row_id={rid})\n"]
    if val_data.get("ids") is not None:
        lines.append(f"**Context:** `{_decode(val_data['ids'][ri][-64:], tokenizer)}`\n")
    lines.append(f"**Gold:** {d(gt)} id={gt} region={regstr(gt)}")
    lines.append(f"**Base top-1:** {d(b1)} id={b1} region={regstr(b1)}")
    lines.append(f"**Refined top-1:** {d(r1)} id={r1} region={regstr(r1)}")
    lines.append(f"**Gold rank:** base={grb:.0f}  refined={grr:.0f}")
    lines.append(f"**Pair margin:** base={pmb:.3f}  refined={pmr:.3f}")
    lines.append(f"**margin_delta:** {md:.4f}  (surgical: only gold+base_top1 edited)\n")

    K   = val_data["topk_ids"].shape[1]
    topk_row = val_data["topk_ids"][ri]
    blg  = row["base_lgt_top10"][ri]
    rlg  = row["ref_lgt_top10"][ri]
    gidx = int(row["gold_idx"][ri])

    lines.append("| Rank | Token | ID | Region | Base logit | Ref logit | Edited |")
    lines.append("|------|-------|----|--------|------------|-----------|--------|")
    for r in range(min(10, K)):
        tid = int(topk_row[r])
        try: ts = tokenizer.decode([tid])
        except: ts = str(tid)
        bl  = float(blg[r])
        rl  = float(rlg[r])
        edited = "gold" if r == gidx else ("base_top1" if r == 0 else "")
        mk     = "✓" if tid == gt else ""
        lines.append(f"| {r+1} | `{ts}` | {tid} | {regstr(tid)} | {bl:.3f} | {rl:.3f} | {edited} | {mk}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_example_reports(model, val_data, tok_arr_t, reg_arr_t,
                          unk_region, unk_super, sr_enabled,
                          args, device, tokenizer, subsets, row, output_dir):
    n_max = args.num_examples
    rng   = np.random.default_rng(42)
    cov   = row["covered"]; bw = row["base_wrong"]
    ctg   = row["changed_to_gold"]; sr = row["same_reg"]; ss = row["same_sup"]
    md    = row["margin_delta"]
    app   = row["applied"]

    buckets = {
        "oracle_flipped_to_gold":   np.where(ctg & bw & cov)[0],
        "oracle_failed":            np.where(bw & cov & (sr|ss) & ~ctg)[0],
        "large_positive_delta":     np.where(app & (md > np.nanpercentile(md[app], 90)))[0] if app.any() else np.array([]),
        "negative_delta":           np.where(app & (md < 0))[0],
    }
    kw = dict(val_data=val_data, row=row,
              tok_arr_t=tok_arr_t.cpu(), unk_region=unk_region,
              tokenizer=tokenizer)
    titles = {
        "oracle_flipped_to_gold": "Oracle: Flipped to Gold",
        "oracle_failed":          "Oracle: Failed (Same-Region Confuser Not Fixed)",
        "large_positive_delta":   "Large Positive margin_delta",
        "negative_delta":         "Negative margin_delta (penalized by L_positive)",
    }
    for bname, indices in buckets.items():
        if len(indices) > n_max:
            indices = rng.choice(indices, n_max, replace=False)
        path = os.path.join(output_dir, f"examples_{bname}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {titles[bname]}\n\n_{len(indices)} examples_\n\n---\n\n")
            for ei, ri in enumerate(indices):
                f.write(_fmt_example_v3(ei+1, int(ri), **kw))
                f.write("\n---\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# CSV / report
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv_row(path, d, header=False):
    mode = "w" if header else "a"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d.keys()))
        if header:
            w.writeheader()
        w.writerow(d)


def _json_safe(obj):
    if isinstance(obj, dict):  return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj); return None if math.isnan(v) else v
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj


def write_report(run_dir, args, pool_stats, final_subsets, best_step, fv_final):
    def f(v, fmt=".4f"):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return "nan"
        if isinstance(v, float): return format(v, fmt)
        return str(v)

    m_tgt = final_subsets.get("target_confuser", {})
    m_nh  = final_subsets.get("base_correct_covered", {})
    m_all = final_subsets.get("all", {})

    lines = ["# Region Pairwise Reranker V3 — Surgical Oracle Diagnostic\n"]
    lines.append(f"**Run:** `{args.run_name}`  |  **Steps:** {args.steps}  "
                 f"|  **Apply policy:** `{args.apply_policy}`\n")

    lines.append("## ⚠️  Oracle Diagnostic Notice\n")
    lines.append("V3 uses `gold_idx` at eval time to identify which candidate pair to correct. "
                 "This is NOT deployable at inference time (gold unknown).\n")
    lines.append("**Purpose:** Test whether region-derived pairwise features can learn "
                 "a useful correction *when the pair is known*.\n")
    lines.append("If V3 succeeds → pairwise signal is real; next step is a non-oracle pair selector.\n"
                 "If V3 fails → pairwise features are insufficient even in the oracle case.\n")

    lines.append("## V1/V2 Reference\n")
    lines.append("| Version | changed_to_gold | pairwise_win_ref | fv_gain |")
    lines.append("|---------|----------------|-----------------|---------|")
    lines.append("| V1 | 0.1018 | 0.1379 | -0.0050 |")
    lines.append("| V2 | 0.0193 | 0.0299 | -0.0105 (gate saturated) |\n")

    lines.append("## Pool Stats\n")
    for k, v in pool_stats.items():
        lines.append(f"  {k:30s} = {f(v)}")
    lines.append("")

    lines.append("## Final Metrics\n")
    lines.append("### target_confuser (oracle-applied)\n")
    for k in ["n", "applied_rate", "changed_to_gold_rate", "changed_away_rate",
               "pairwise_win_rate_base", "pairwise_win_rate_refined",
               "mean_pair_margin_base", "mean_pair_margin_refined",
               "mean_margin_delta", "mean_abs_margin_delta"]:
        lines.append(f"  {k:40s} = {f(m_tgt.get(k, float('nan')))}")
    lines.append("")
    lines.append("### base_correct_covered (no-harm: expect changed_away=0 under target_only)\n")
    for k in ["n", "changed_away_rate", "applied_rate"]:
        lines.append(f"  {k:40s} = {f(m_nh.get(k, float('nan')))}")
    lines.append("")

    if fv_final:
        lines.append("### Full-vocab (oracle policy)\n")
        for k, v in fv_final.items():
            lines.append(f"  {k:40s} = {f(v)}")
        lines.append("")

    lines.append("## Interpretation\n")
    ctg = m_tgt.get("changed_to_gold_rate", float("nan"))
    pwr = m_tgt.get("pairwise_win_rate_refined", float("nan"))
    caw = m_nh.get("changed_away_rate", float("nan"))
    fvg = fv_final.get("oracle_policy_full_vocab_gain", float("nan")) if fv_final else float("nan")

    def chk(c, yes, no): return yes if (not math.isnan(float(c)) and c) else no
    lines.append(f"**Target correction > V1?**  ctg={f(ctg)} vs V1=0.1018  "
                 + chk(not math.isnan(ctg) and ctg > 0.1018, "✅ Yes", "⚠️  No"))
    lines.append(f"**Collateral damage?**  caw={f(caw)}  "
                 + chk(not math.isnan(caw) and caw < 0.001, "✅ None (expected under target_only)", "❌ Unexpected damage"))
    lines.append(f"**Oracle full-vocab gain?**  fv_gain={f(fvg)}  "
                 + chk(not math.isnan(fvg) and fvg >= 0, "✅ Non-negative", "⚠️  Negative"))
    lines.append("")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# AMP
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _amp_ctx(use_amp):
    if use_amp:
        with torch.cuda.amp.autocast(): yield
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
    print(f"device: {device}  apply_policy: {args.apply_policy}")

    run_dir = os.path.join(args.output_root, args.run_name)
    if os.path.exists(run_dir):
        raise RuntimeError(f"Run dir exists: {run_dir}. Use --run_name.")
    os.makedirs(run_dir)

    print("\n[backbone] Loading token embeddings...")
    tok_w, d_model, vocab_size = load_backbone(args.small_ckpt, device)
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    print("\n[maps] Loading region/super maps...")
    t2r, r2s, unk_region, unk_super, sr_enabled = load_maps(
        args.token_to_region, args.super_map)

    tok_arr_np = build_tok_arr(t2r, unk_region, vocab_size)
    reg_arr_np = build_reg_arr(r2s, unk_super, unk_region)
    tok_arr_t  = torch.from_numpy(tok_arr_np).long().to(device)
    reg_arr_t  = torch.from_numpy(reg_arr_np).long().to(device)

    print("\n[data] Loading shards...")
    train_data = load_shards(args.train_dir, args.top_k, "train",
                             args.max_train_rows, load_ids=False)
    val_data   = load_shards(args.val_dir,   args.top_k, "val",
                             args.max_val_rows,   load_ids=True)

    print("\n[pools] Building train pools...")
    target_idx, noharm_idx, other_idx, pool_stats = build_train_pools(
        train_data, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled)
    for k, v in pool_stats.items():
        print(f"  {k:30s} = {v:.4f}" if isinstance(v, float) else f"  {k:30s} = {v}")
    if len(target_idx) == 0:
        raise RuntimeError("No target training rows found.")

    n_regions = unk_region; n_supers = unk_super if sr_enabled else 1
    print("\n[model] Building SurgicalPairwiseReranker...")
    model = SurgicalPairwiseReranker(
        token_emb_weight=tok_w.to(device),
        tok_arr=tok_arr_np, reg_arr=reg_arr_np,
        d_model=d_model, n_regions=n_regions, n_supers=n_supers,
        sr_enabled=sr_enabled, unk_region=unk_region, unk_super=unk_super,
        region_emb_dim=args.region_emb_dim, super_emb_dim=args.super_emb_dim,
        hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
        margin_delta_scale=args.margin_delta_scale, top_k=args.top_k,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    config = vars(args).copy()
    config.update({"d_model": d_model, "vocab_size": vocab_size,
                   "sr_enabled": sr_enabled, "n_params": n_params,
                   "pool_stats": pool_stats, "oracle_pair_eval": True})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\n[identity] Running identity check...")
    run_identity_check(model, val_data, tok_arr_t, reg_arr_t,
                       unk_region, unk_super, sr_enabled, args, device, run_dir)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None

    target_bsz = max(1, round(args.batch_size * args.target_fraction))
    noharm_bsz = args.batch_size - target_bsz

    N_t = len(target_idx)
    N_n = len(noharm_idx)
    t_perm = torch.randperm(N_t)
    n_perm = torch.randperm(N_n) if N_n > 0 else None
    t_pos = n_pos = 0

    train_log  = os.path.join(run_dir, "train_log.csv")
    eval_log   = os.path.join(run_dir, "eval_log.csv")
    sub_log    = os.path.join(run_dir, "subset_eval_log.csv")
    hdr_t = hdr_e = hdr_s = False

    best_score = -float("inf"); best_step = 0
    step = accum = 0; t0 = time.time()
    optimizer.zero_grad()

    print(f"\n[train] steps={args.steps}  batch={args.batch_size} "
          f"(target={target_bsz} noharm={noharm_bsz})  lr={args.lr}\n")

    while step < args.steps:
        # Sample target rows
        if t_pos + target_bsz > N_t:
            t_perm = torch.randperm(N_t); t_pos = 0
        t_global = target_idx[t_perm[t_pos:t_pos + target_bsz]]; t_pos += target_bsz

        # Sample noharm rows (included in batch but loss is zero for them)
        if noharm_bsz > 0 and N_n > 0:
            if n_pos + noharm_bsz > N_n:
                n_perm = torch.randperm(N_n); n_pos = 0
            n_global = noharm_idx[n_perm[n_pos:n_pos + noharm_bsz]]; n_pos += noharm_bsz
            batch_global = torch.cat([t_global, n_global])
        else:
            batch_global = t_global

        hp   = train_data["h_prime"][batch_global].to(device)
        topk = train_data["topk_ids"][batch_global].to(device)
        lgt  = train_data["topk_lgt"][batch_global].to(device)
        gold = train_data["gold"][batch_global].to(device)
        B    = hp.shape[0]

        gold_in = (topk == gold.unsqueeze(1))
        covered = gold_in.any(1)
        gold_idx_b = gold_in.long().argmax(1)
        gsafe    = gold_idx_b.clamp(0, args.top_k - 1)

        cr, cs = _lookup_regs(topk, tok_arr_t, reg_arr_t,
                               unk_region, unk_super, sr_enabled)

        # Only apply correction to target rows in training batch (first target_bsz)
        apply_mask_train = torch.zeros(B, dtype=torch.bool, device=device)
        apply_mask_train[:target_bsz] = True
        apply_mask_train &= covered & (gsafe != 0)

        with _amp_ctx(args.amp):
            md_b = model(hp, topk, lgt, gold_idx_b, cr, cs)
            ref  = apply_correction(lgt, md_b, gsafe, apply_mask_train)
            # Loss only on target rows (first target_bsz or all covered+base_wrong)
            tgt_covered  = apply_mask_train
            total_loss, ld = _compute_losses_v3(
                ref, md_b, gsafe, tgt_covered, lgt, args, device)
            loss_sc = total_loss / args.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss_sc).backward()
        else:
            loss_sc.backward()

        accum += 1
        if accum < args.grad_accum_steps:
            continue

        accum = 0
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
                  f"pair={ld['pair']:.4f}  pos={ld['positive']:.4f}  "
                  f"delta={ld['delta']:.4f}  t={time.time()-t0:.0f}s")

        _write_csv_row(train_log, {"step": step, **ld}, header=not hdr_t); hdr_t = True

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n[eval] step={step}")
            subsets, row_res = evaluate(
                model, val_data, tok_arr_t, reg_arr_t,
                unk_region, unk_super, sr_enabled, args, device)

            fv = {}
            if args.eval_full_vocab:
                fv = eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                                     unk_region, unk_super, sr_enabled, args, device)

            m_tgt = subsets.get("target_confuser", {})
            m_nh  = subsets.get("base_correct_covered", {})
            score = subsets.get("_checkpoint_score", float("nan"))
            print(f"  [tgt]  n={m_tgt.get('n',0)}  "
                  f"ctg={m_tgt.get('changed_to_gold_rate', float('nan')):.4f}  "
                  f"pwr={m_tgt.get('pairwise_win_rate_refined', float('nan')):.4f}  "
                  f"md={m_tgt.get('mean_margin_delta', float('nan')):.4f}")
            print(f"  [nh]   caw={m_nh.get('changed_away_rate', float('nan')):.4f}  "
                  f"(expect 0 under target_only)")
            if fv:
                print(f"  fv_gain={fv.get('oracle_policy_full_vocab_gain', float('nan')):.4f}")
            print(f"  score={score:.4f}" if not math.isnan(score) else "  score=nan")

            torch.save({"step": step, "model": model.state_dict(),
                        "subsets": subsets, "fv": fv},
                       os.path.join(run_dir, "latest_reranker.pt"))

            if not math.isnan(score) and score > best_score:
                best_score = score; best_step = step
                torch.save({"step": step, "model": model.state_dict(),
                            "subsets": subsets, "fv": fv},
                           os.path.join(run_dir, "best_reranker.pt"))
                with open(os.path.join(run_dir, "best_metrics.json"), "w") as fp:
                    json.dump(_json_safe({"step": step, "score": score,
                                         "subsets": subsets, "fv": fv}), fp, indent=2)
                print(f"  [best] step={step}  score={score:.4f}")

            ev = {"step": step, **fv}
            for sn, sm in subsets.items():
                if sn.startswith("_"): continue
                for k, v in sm.items(): ev[f"{sn}_{k}"] = v
            _write_csv_row(eval_log, ev, header=not hdr_e); hdr_e = True
            for sn, sm in subsets.items():
                if sn.startswith("_"): continue
                _write_csv_row(sub_log, {"step": step, "subset": sn, **sm},
                               header=not hdr_s); hdr_s = True
            print()

    print("[final eval]")
    final_subsets, final_row = evaluate(
        model, val_data, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled, args, device)
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

    write_report(run_dir, args, pool_stats, final_subsets, best_step, fv_final)

    print(f"\n{'='*60}")
    print(f" V3 Surgical Pairwise complete.")
    print(f" Run dir: {run_dir}")
    print(f" Best step: {best_step}  score={best_score:.4f}")
    print(f"{'='*60}\n")


def _parse():
    p = argparse.ArgumentParser(description="Region Pairwise Reranker V3 — Surgical Oracle")
    p.add_argument("--small_ckpt",       required=True)
    p.add_argument("--train_dir",        required=True)
    p.add_argument("--val_dir",          required=True)
    p.add_argument("--token_to_region",  required=True)
    p.add_argument("--super_map",        default=None)
    p.add_argument("--output_root",      default="runs/region_pairwise_reranker_v3")
    p.add_argument("--run_name",         default="surgical_pair_v1")
    p.add_argument("--top_k",            type=int,   default=256)
    p.add_argument("--max_train_rows",   type=int,   default=None)
    p.add_argument("--max_val_rows",     type=int,   default=None)
    p.add_argument("--apply_policy",     default="target_only", choices=_APPLY_POLICIES)
    p.add_argument("--num_examples",     type=int,   default=50)
    p.add_argument("--target_fraction",  type=float, default=0.75)
    p.add_argument("--region_emb_dim",   type=int,   default=64)
    p.add_argument("--super_emb_dim",    type=int,   default=32)
    p.add_argument("--hidden_dim",       type=int,   default=256)
    p.add_argument("--n_hidden",         type=int,   default=3)
    p.add_argument("--margin_delta_scale",type=float,default=1.0)
    p.add_argument("--allow_signed_delta",action="store_true")
    p.add_argument("--target_margin",    type=float, default=0.0)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--grad_accum_steps", type=int,   default=1)
    p.add_argument("--steps",            type=int,   default=2000)
    p.add_argument("--eval_every",       type=int,   default=500)
    p.add_argument("--eval_batch_size",  type=int,   default=128)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    p.add_argument("--amp",              action="store_true")
    p.add_argument("--eval_full_vocab",  action="store_true")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--lambda_pair",          type=float, default=2.0)
    p.add_argument("--lambda_positive",      type=float, default=0.1)
    p.add_argument("--lambda_margin_delta",  type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    main()
