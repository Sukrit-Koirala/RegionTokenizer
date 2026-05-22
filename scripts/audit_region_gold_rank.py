"""
audit_region_gold_rank.py — Pre-reranker Region Gold Rank Audit

Reads patched val (and optionally train) shards produced by Stage01B.
No model is loaded. No retrieval is performed. No manual token classes used.

Only reads from each shard row:
  gold_token / gold
  base_topk_ids / base_topk
  base_topk_lgt / base_topk_logits
  input_ids
  row_id (optional)

Maps each token through token_to_region → region_to_superregion, handling
unmapped tokens as UNK explicitly (never dropped silently).

All ranks reported are rank-in-topK (0-based internally, 1-based in output).
Full-vocab rank is never claimed.

Outputs (to output_root/ for val, output_root/train/ for train):
  summary.md
  subset_metrics.csv
  gold_rank_histograms.csv
  topk_region_curves.csv
  examples_gold_rank_1.md
  examples_covered_wrong_base.md
  examples_covered_high_region_rank.md
  examples_not_covered.md
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_RANK_BINS = [
    ("rank_1",       0,   0),
    ("rank_2",       1,   1),
    ("rank_3_5",     2,   4),
    ("rank_6_10",    5,   9),
    ("rank_11_25",  10,  24),
    ("rank_26_50",  25,  49),
    ("rank_51_100", 50,  99),
    ("rank_101_256",100, 255),
]

_LOCAL_RANK_BINS = [
    ("local_rank_1",    0, 0),
    ("local_rank_2",    1, 1),
    ("local_rank_3_5",  2, 4),
    ("local_rank_6_10", 5, 9),
    ("local_rank_gt10",10, 99999),
]

_SUBSET_ORDER = [
    "all",
    "base_correct",
    "base_wrong",
    "covered",
    "base_wrong_covered",
    "base_wrong_not_covered",
    "base_wrong_top1_same_region",
    "base_wrong_top1_same_superregion",
    "base_wrong_top1_not_same_region",
    "base_wrong_top1_not_same_superregion",
    "covered_top1_same_region",
    "covered_top1_same_superregion",
]

# ─────────────────────────────────────────────────────────────────────────────
# Map loading
# ─────────────────────────────────────────────────────────────────────────────

def load_maps(token_to_region_path, super_map_path):
    """Load token→region and region→superregion maps.

    token_to_region may be a list (index = token_id) or dict (str key).
    Returns (t2r_dict, r2s_dict, unk_region, unk_super).
    """
    with open(token_to_region_path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        t2r = {i: v for i, v in enumerate(raw) if v is not None}
    elif isinstance(raw, dict):
        t2r = {int(k): v for k, v in raw.items()}
    else:
        raise ValueError(f"token_to_region must be list or dict, got {type(raw)}")

    max_region = max(t2r.values()) if t2r else 0
    unk_region = int(max_region) + 1

    if super_map_path and os.path.isfile(super_map_path):
        with open(super_map_path) as f:
            raw_s = json.load(f)
        if isinstance(raw_s, list):
            r2s = {i: v for i, v in enumerate(raw_s) if v is not None}
        elif isinstance(raw_s, dict):
            r2s = {int(k): v for k, v in raw_s.items()}
        else:
            raise ValueError(f"super_map must be list or dict, got {type(raw_s)}")
        max_super = max(r2s.values()) if r2s else 0
        unk_super = int(max_super) + 1
    else:
        r2s = {}
        unk_super = 0

    return t2r, r2s, unk_region, unk_super


# ─────────────────────────────────────────────────────────────────────────────
# Shard loading
# ─────────────────────────────────────────────────────────────────────────────

def _get_field(shard, *names, required=True):
    for n in names:
        if n in shard:
            return shard[n]
    if required:
        raise KeyError(
            f"Shard missing all aliases for field. Tried: {names}. "
            f"Available: {list(shard.keys())}"
        )
    return None


def load_shards(shard_dir, top_k, split_name="val"):
    """Load all shard_*.pt in shard_dir; return concatenated arrays.

    Returns:
        gold_t   (N,)         int64
        topk_t   (N, top_k)   int64  — padded to top_k width
        lgt_t    (N, top_k)   float32 or None
        ids_t    (N, L)       int64 or None
        rid_t    (N,)         str array or None
        total    int
    """
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt found in {shard_dir}")

    all_gold, all_topk, all_lgt, all_ids, all_rid = [], [], [], [], []
    total = 0

    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)

        gold = _get_field(shard, "gold_token", "gold").long()          # (B,)
        topk = _get_field(shard, "base_topk_ids", "base_topk").long()  # (B, K)
        lgt  = _get_field(shard, "base_topk_lgt", "base_topk_logits",
                          required=False)
        ids  = _get_field(shard, "input_ids", required=False)
        rid  = _get_field(shard, "row_id", required=False)

        B = gold.shape[0]
        K = topk.shape[1]

        if K < top_k:
            pad = torch.zeros(B, top_k - K, dtype=torch.long)
            topk = torch.cat([topk, pad], dim=1)
        elif K > top_k:
            topk = topk[:, :top_k]

        if lgt is not None:
            lgt = lgt.float()
            if lgt.shape[1] < top_k:
                pad = torch.full((B, top_k - lgt.shape[1]), float("nan"))
                lgt = torch.cat([lgt, pad], dim=1)
            elif lgt.shape[1] > top_k:
                lgt = lgt[:, :top_k]

        all_gold.append(gold)
        all_topk.append(topk)
        if lgt is not None:
            all_lgt.append(lgt)
        if ids is not None:
            all_ids.append(ids)
        if rid is not None:
            all_rid.append(rid if isinstance(rid, (list, tuple)) else rid.tolist())
        total += B

    gold_t = torch.cat(all_gold, 0)
    topk_t = torch.cat(all_topk, 0)
    lgt_t  = torch.cat(all_lgt,  0) if all_lgt else None
    ids_t  = torch.cat(all_ids,  0) if all_ids else None
    rid_t  = sum(all_rid, [])        if all_rid else None

    print(f"  Loaded {split_name}: {total} rows from {len(shards)} shards "
          f"(top_k={top_k})")
    return gold_t, topk_t, lgt_t, ids_t, rid_t, total


# ─────────────────────────────────────────────────────────────────────────────
# Integer-indexed lookup arrays
# ─────────────────────────────────────────────────────────────────────────────

def build_lookups(t2r, r2s, unk_region, unk_super, max_tok):
    """Build integer-indexed numpy arrays for O(1) vectorized lookups.

    tok_arr[token_id] = region_id  (unk_region if unmapped)
    reg_arr[region_id] = super_id  (unk_super if unmapped)
    """
    tok_arr = np.full(max_tok + 1, unk_region, dtype=np.int32)
    for tok, reg in t2r.items():
        if tok <= max_tok:
            tok_arr[tok] = int(reg)

    max_reg = int(unk_region)   # unk_region = max_region + 1
    reg_arr = np.full(max_reg + 1, unk_super, dtype=np.int32)
    for reg, sup in r2s.items():
        if reg < len(reg_arr):
            reg_arr[reg] = int(sup)

    return tok_arr, reg_arr, max_reg


# ─────────────────────────────────────────────────────────────────────────────
# Derived field computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(gold_t, topk_t, lgt_t, tok_arr, reg_arr,
                unk_region, unk_super, ks, top_k):
    """Compute all derived fields, returning a dict of numpy arrays.

    All ranks are 0-based internally; caller converts to 1-based for display.
    -1 means "not applicable / not found."
    """
    N = gold_t.shape[0]

    gold_np  = gold_t.numpy().astype(np.int64)    # (N,)
    topk_np  = topk_t.numpy().astype(np.int64)    # (N, top_k)
    lgt_np   = lgt_t.numpy().astype(np.float32) if lgt_t is not None else None

    # Clamp to valid tok_arr range
    max_tok = len(tok_arr) - 1
    gold_safe = np.clip(gold_np, 0, max_tok)
    topk_safe = np.clip(topk_np, 0, max_tok)

    # Region + super for all candidates and gold
    cand_regions = tok_arr[topk_safe]                    # (N, top_k)
    gold_regions = tok_arr[gold_safe]                    # (N,)

    max_reg = len(reg_arr) - 1
    cand_regions_safe = np.clip(cand_regions, 0, max_reg)
    gold_regions_safe = np.clip(gold_regions, 0, max_reg)

    cand_supers = reg_arr[cand_regions_safe]             # (N, top_k)
    gold_supers = reg_arr[gold_regions_safe]             # (N,)

    top1_tokens  = topk_np[:, 0]                         # (N,)
    top1_regions = cand_regions[:, 0]                    # (N,)
    top1_supers  = cand_supers[:, 0]                     # (N,)

    base_correct      = (gold_np == top1_tokens)                   # (N,)
    top1_same_reg     = (gold_regions == top1_regions)             # (N,)
    top1_same_sup     = (gold_supers == top1_supers)               # (N,)
    gold_unmapped     = (gold_regions == unk_region)               # (N,)
    top1_unmapped     = (top1_regions == unk_region)               # (N,)

    # Coverage: gold token appears somewhere in topk candidates
    gold_in_topk = (topk_np == gold_np[:, None])                   # (N, top_k)
    covered      = gold_in_topk.any(axis=1)                        # (N,)

    # Gold rank in topK (0-based; -1 = not covered)
    # argmax on boolean array gives first True index
    gold_rank_0b = np.where(covered,
                            gold_in_topk.argmax(axis=1).astype(np.int32),
                            np.int32(-1))                          # (N,)

    # First candidate index with same region / super as gold (0-based; top_k = not found)
    same_reg_mask = (cand_regions == gold_regions[:, None])        # (N, top_k)
    same_sup_mask = (cand_supers  == gold_supers[:, None])         # (N, top_k)

    # Exclude UNK from "same region" matching
    unk_gold_reg = gold_regions == unk_region
    unk_gold_sup = gold_supers  == unk_super

    same_reg_mask[unk_gold_reg] = False
    same_sup_mask[unk_gold_sup] = False

    has_same_reg = same_reg_mask.any(axis=1)
    has_same_sup = same_sup_mask.any(axis=1)

    first_reg_idx = np.where(has_same_reg,
                             same_reg_mask.argmax(axis=1).astype(np.int32),
                             np.int32(top_k))
    first_sup_idx = np.where(has_same_sup,
                             same_sup_mask.argmax(axis=1).astype(np.int32),
                             np.int32(top_k))

    num_same_reg = same_reg_mask.sum(axis=1).astype(np.int32)     # (N,)
    num_same_sup = same_sup_mask.sum(axis=1).astype(np.int32)      # (N,)

    # Logit gap: lgt[gold_rank] - lgt[0]  (nan if not covered or no logits)
    if lgt_np is not None:
        logit_gap = np.where(
            covered & (gold_rank_0b >= 0),
            lgt_np[np.arange(N), np.maximum(gold_rank_0b, 0)] - lgt_np[:, 0],
            np.nan
        ).astype(np.float32)
    else:
        logit_gap = np.full(N, np.nan, dtype=np.float32)

    # Region-local rank: rank of gold within same-region candidates (0-based; -1 = N/A)
    # Super-local rank: rank of gold within same-super candidates
    # These require a Python loop (per-row variable-length sets)
    region_local_rank = np.full(N, -1, dtype=np.int32)
    super_local_rank  = np.full(N, -1, dtype=np.int32)

    for i in range(N):
        if not covered[i]:
            continue
        gr = gold_regions[i]
        gs = gold_supers[i]
        if gr != unk_region:
            reg_cands = np.where(cand_regions[i] == gr)[0]
            gt_pos = np.searchsorted(reg_cands, gold_rank_0b[i])
            region_local_rank[i] = int(gt_pos)
        if gs != unk_super:
            sup_cands = np.where(cand_supers[i] == gs)[0]
            gt_pos = np.searchsorted(sup_cands, gold_rank_0b[i])
            super_local_rank[i] = int(gt_pos)

    return {
        "gold_np":           gold_np,
        "topk_np":           topk_np,
        "lgt_np":            lgt_np,
        "cand_regions":      cand_regions,
        "gold_regions":      gold_regions,
        "cand_supers":       cand_supers,
        "gold_supers":       gold_supers,
        "top1_tokens":       top1_tokens,
        "top1_regions":      top1_regions,
        "top1_supers":       top1_supers,
        "base_correct":      base_correct,
        "top1_same_reg":     top1_same_reg,
        "top1_same_sup":     top1_same_sup,
        "gold_unmapped":     gold_unmapped,
        "top1_unmapped":     top1_unmapped,
        "covered":           covered,
        "gold_rank_0b":      gold_rank_0b,
        "first_reg_idx":     first_reg_idx,
        "first_sup_idx":     first_sup_idx,
        "num_same_reg":      num_same_reg,
        "num_same_sup":      num_same_sup,
        "logit_gap":         logit_gap,
        "region_local_rank": region_local_rank,
        "super_local_rank":  super_local_rank,
        "same_reg_mask":     same_reg_mask,
        "same_sup_mask":     same_sup_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Subset masks
# ─────────────────────────────────────────────────────────────────────────────

def build_masks(d, unk_region, unk_super):
    bc  = d["base_correct"]
    bw  = ~bc
    cov = d["covered"]
    t1r = d["top1_same_reg"]
    t1s = d["top1_same_sup"]
    tu  = d["top1_unmapped"]
    N   = bc.shape[0]

    return {
        "all":                               np.ones(N, dtype=bool),
        "base_correct":                      bc,
        "base_wrong":                        bw,
        "covered":                           cov,
        "base_wrong_covered":                bw & cov,
        "base_wrong_not_covered":            bw & ~cov,
        "base_wrong_top1_same_region":       bw & t1r & ~tu,
        "base_wrong_top1_same_superregion":  bw & t1s & ~tu,
        "base_wrong_top1_not_same_region":   bw & ~t1r & ~tu,
        "base_wrong_top1_not_same_superregion": bw & ~t1s & ~tu,
        "covered_top1_same_region":          cov & t1r & ~tu,
        "covered_top1_same_superregion":     cov & t1s & ~tu,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics per subset
# ─────────────────────────────────────────────────────────────────────────────

def _safe_rate(num, den):
    return float(num) / float(den) if den > 0 else float("nan")


def _global_rank_hist(ranks_0b, n_sub, top_k):
    """ranks_0b: 1-D int array, -1 = not found. Returns bin → [count, rate]."""
    out = {}
    not_covered = int((ranks_0b == -1).sum())
    out["not_in_top" + str(top_k)] = [not_covered, _safe_rate(not_covered, n_sub)]
    valid = ranks_0b[ranks_0b >= 0]
    for name, lo, hi in _GLOBAL_RANK_BINS:
        c = int(((valid >= lo) & (valid <= hi)).sum())
        out[name] = [c, _safe_rate(c, n_sub)]
    return out


def _local_rank_hist(ranks_0b, n_sub):
    """ranks_0b: 1-D int array, -1 = N/A. Returns bin → [count, rate]."""
    out = {}
    na = int((ranks_0b == -1).sum())
    out["not_covered_or_unk"] = [na, _safe_rate(na, n_sub)]
    valid = ranks_0b[ranks_0b >= 0]
    for name, lo, hi in _LOCAL_RANK_BINS:
        c = int(((valid >= lo) & (valid <= hi)).sum())
        out[name] = [c, _safe_rate(c, n_sub)]
    return out


def _mean1b(arr_0b):
    valid = arr_0b[arr_0b >= 0]
    return float(valid.mean()) + 1.0 if len(valid) > 0 else float("nan")


def _median1b(arr_0b):
    valid = arr_0b[arr_0b >= 0]
    return float(np.median(valid)) + 1.0 if len(valid) > 0 else float("nan")


def _topk_region_curve(d, mask, ks, top_k):
    """For each k in ks, fraction of masked rows where gold region appears in top-k."""
    sub_same_reg = d["same_reg_mask"][mask]   # (M, top_k)
    sub_same_sup = d["same_sup_mask"][mask]
    sub_cov      = d["covered"][mask]
    M = mask.sum()
    curve_reg = {}
    curve_sup = {}
    for k in ks:
        k = min(k, top_k)
        in_reg = sub_same_reg[:, :k].any(axis=1)
        in_sup = sub_same_sup[:, :k].any(axis=1)
        curve_reg[f"top{k}_has_same_region"] = [int(in_reg.sum()), _safe_rate(int(in_reg.sum()), M)]
        curve_sup[f"top{k}_has_same_super"]  = [int(in_sup.sum()), _safe_rate(int(in_sup.sum()), M)]
    return curve_reg, curve_sup


def compute_subset_metrics(mask, d, unk_region, unk_super, ks, top_k, n_all):
    n_sub = int(mask.sum())
    if n_sub == 0:
        return {"n": 0, "rate_of_all": 0.0}

    rate_of_all = _safe_rate(n_sub, n_all)

    gr_sub   = d["gold_rank_0b"][mask]
    rlr_sub  = d["region_local_rank"][mask]
    slr_sub  = d["super_local_rank"][mask]
    lg_sub   = d["logit_gap"][mask]
    frid_sub = d["first_reg_idx"][mask]
    fsid_sub = d["first_sup_idx"][mask]
    nsr_sub  = d["num_same_reg"][mask]
    nss_sub  = d["num_same_sup"][mask]
    cov_sub  = d["covered"][mask]
    bc_sub   = d["base_correct"][mask]
    t1r_sub  = d["top1_same_reg"][mask]
    t1s_sub  = d["top1_same_sup"][mask]
    gu_sub   = d["gold_unmapped"][mask]
    tu_sub   = d["top1_unmapped"][mask]

    valid_lg = lg_sub[~np.isnan(lg_sub)]
    fr_found = frid_sub[frid_sub < top_k]
    fs_found = fsid_sub[fsid_sub < top_k]

    curve_reg, curve_sup = _topk_region_curve(d, mask, ks, top_k)

    return {
        "n":                       n_sub,
        "rate_of_all":             rate_of_all,
        "base_correct_rate":       _safe_rate(int(bc_sub.sum()), n_sub),
        "covered_rate":            _safe_rate(int(cov_sub.sum()), n_sub),
        "top1_same_region_rate":   _safe_rate(int(t1r_sub.sum()), n_sub),
        "top1_same_super_rate":    _safe_rate(int(t1s_sub.sum()), n_sub),
        "gold_unmapped_rate":      _safe_rate(int(gu_sub.sum()), n_sub),
        "top1_unmapped_rate":      _safe_rate(int(tu_sub.sum()), n_sub),
        "mean_gold_rank_1b":       _mean1b(gr_sub),
        "median_gold_rank_1b":     _median1b(gr_sub),
        "mean_region_local_rank_1b":  _mean1b(rlr_sub),
        "median_region_local_rank_1b": _median1b(rlr_sub),
        "mean_super_local_rank_1b":   _mean1b(slr_sub),
        "median_super_local_rank_1b": _median1b(slr_sub),
        "mean_logit_gap":          float(valid_lg.mean())   if len(valid_lg) > 0 else float("nan"),
        "mean_first_same_reg_rank_1b": _mean1b(fr_found)   if len(fr_found) > 0 else float("nan"),
        "mean_first_same_sup_rank_1b": _mean1b(fs_found)   if len(fs_found) > 0 else float("nan"),
        "mean_num_same_reg_in_topK":   float(nsr_sub.mean()),
        "mean_num_same_sup_in_topK":   float(nss_sub.mean()),
        "gold_rank_hist":          _global_rank_hist(gr_sub, n_sub, top_k),
        "region_local_rank_hist":  _local_rank_hist(rlr_sub, n_sub),
        "super_local_rank_hist":   _local_rank_hist(slr_sub, n_sub),
        "topk_region_curve":       curve_reg,
        "topk_super_curve":        curve_sup,
    }


def compute_all_subsets(d, masks, unk_region, unk_super, ks, top_k):
    n_all = int(masks["all"].sum())
    return {
        name: compute_subset_metrics(masks[name], d, unk_region, unk_super,
                                     ks, top_k, n_all)
        for name in _SUBSET_ORDER
        if name in masks
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────

def write_subset_csv(subsets, output_dir):
    path = os.path.join(output_dir, "subset_metrics.csv")
    scalar_keys = [
        "n", "rate_of_all", "base_correct_rate", "covered_rate",
        "top1_same_region_rate", "top1_same_super_rate",
        "gold_unmapped_rate", "top1_unmapped_rate",
        "mean_gold_rank_1b", "median_gold_rank_1b",
        "mean_region_local_rank_1b", "median_region_local_rank_1b",
        "mean_super_local_rank_1b", "median_super_local_rank_1b",
        "mean_logit_gap",
        "mean_first_same_reg_rank_1b", "mean_first_same_sup_rank_1b",
        "mean_num_same_reg_in_topK", "mean_num_same_sup_in_topK",
    ]
    header = ["subset"] + scalar_keys
    rows = []
    for name in _SUBSET_ORDER:
        m = subsets.get(name, {})
        rows.append([name] + [m.get(k, "") for k in scalar_keys])
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return path


def write_hist_csv(subsets, output_dir, top_k):
    path = os.path.join(output_dir, "gold_rank_histograms.csv")
    bin_names = [f"not_in_top{top_k}"] + [b[0] for b in _GLOBAL_RANK_BINS]
    header = ["subset", "hist_type"] + [b + "_count" for b in bin_names] + \
             [b + "_rate" for b in bin_names]
    rows = []
    for name in _SUBSET_ORDER:
        m = subsets.get(name, {})
        for hist_key, nice in [("gold_rank_hist", "gold_global"),
                                ("region_local_rank_hist", "region_local"),
                                ("super_local_rank_hist", "super_local")]:
            h = m.get(hist_key, {})
            if not h:
                continue
            counts = [h.get(b, [0, 0.0])[0] for b in bin_names]
            rates  = [h.get(b, [0, 0.0])[1] for b in bin_names]
            rows.append([name, hist_key] + counts + rates)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return path


def write_topk_csv(subsets, ks, output_dir, top_k):
    path = os.path.join(output_dir, "topk_region_curves.csv")
    ks_capped = [min(k, top_k) for k in ks]
    col_reg = [f"top{k}_has_same_region_rate" for k in ks_capped]
    col_sup = [f"top{k}_has_same_super_rate"  for k in ks_capped]
    header = ["subset"] + col_reg + col_sup
    rows = []
    for name in _SUBSET_ORDER:
        m = subsets.get(name, {})
        cr = m.get("topk_region_curve", {})
        cs = m.get("topk_super_curve", {})
        reg_vals = [cr.get(f"top{k}_has_same_region", [0, float("nan")])[1] for k in ks_capped]
        sup_vals = [cs.get(f"top{k}_has_same_super",  [0, float("nan")])[1] for k in ks_capped]
        rows.append([name] + reg_vals + sup_vals)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Example formatting
# ─────────────────────────────────────────────────────────────────────────────

def _decode(ids, tokenizer):
    try:
        return tokenizer.decode(ids.tolist() if hasattr(ids, "tolist") else list(ids),
                                skip_special_tokens=False)
    except Exception:
        return str(list(ids))


def format_example_md(idx, i, gold_t, topk_t, lgt_t, ids_t, rid_t, d,
                      tokenizer, unk_region, unk_super, top_k):
    gold_tok  = int(d["gold_np"][i])
    top1_tok  = int(d["top1_tokens"][i])
    gr        = int(d["gold_regions"][i])
    gs        = int(d["gold_supers"][i])
    t1r       = int(d["top1_regions"][i])
    t1s       = int(d["top1_supers"][i])
    rank_0b   = int(d["gold_rank_0b"][i])
    rlr       = int(d["region_local_rank"][i])
    slr       = int(d["super_local_rank"][i])
    lg        = float(d["logit_gap"][i])
    nsr       = int(d["num_same_reg"][i])
    nss       = int(d["num_same_sup"][i])

    gold_str  = f"`{tokenizer.decode([gold_tok])}` ({gold_tok})"
    top1_str  = f"`{tokenizer.decode([top1_tok])}` ({top1_tok})"
    gr_str    = "UNK" if gr == unk_region else str(gr)
    gs_str    = "UNK" if gs == unk_super  else str(gs)
    t1r_str   = "UNK" if t1r == unk_region else str(t1r)
    t1s_str   = "UNK" if t1s == unk_super  else str(t1s)
    rank_str  = f"not_in_top{top_k}" if rank_0b == -1 else str(rank_0b + 1)
    rlr_str   = "N/A" if rlr == -1 else str(rlr + 1)
    slr_str   = "N/A" if slr == -1 else str(slr + 1)
    lg_str    = "nan" if math.isnan(lg) else f"{lg:.3f}"

    rid_str = str(rid_t[i]) if rid_t is not None else str(i)

    lines = [f"### Example {idx} (row_id={rid_str}, shard_idx={i})\n"]
    if ids_t is not None:
        ctx = _decode(ids_t[i][-64:], tokenizer)
        lines.append(f"**Context tail:** `{ctx}`\n")
    lines.append(f"**Gold:** {gold_str}  |  gold_region={gr_str}  |  gold_super={gs_str}")
    lines.append(f"**Base top-1:** {top1_str}  |  top1_region={t1r_str}  |  top1_super={t1s_str}")
    lines.append(f"**Gold rank (in top{top_k}):** {rank_str}")
    lines.append(f"**Region-local rank:** {rlr_str}  |  super-local rank: {slr_str}")
    lines.append(f"**Logit gap (gold − top1):** {lg_str}")
    lines.append(f"**Same-region cands in top{top_k}:** {nsr}  |  same-super: {nss}")
    lines.append("")

    lgt_np = d["lgt_np"]
    topk_np = d["topk_np"]
    lines.append(f"| Rank | Token | ID | Region | Super | Same gold reg | Logit |")
    lines.append(f"|------|-------|----|--------|-------|---------------|-------|")
    for r in range(min(10, top_k)):
        tok = int(topk_np[i, r])
        reg = int(d["cand_regions"][i, r])
        sup = int(d["cand_supers"][i, r])
        reg_s = "UNK" if reg == unk_region else str(reg)
        sup_s = "UNK" if sup == unk_super  else str(sup)
        same  = "✓" if tok == gold_tok else ("~R" if reg == gr and gr != unk_region
                                              else ("~S" if sup == gs and gs != unk_super else ""))
        lg_r  = f"{float(lgt_np[i, r]):.2f}" if lgt_np is not None else "—"
        tok_s = tokenizer.decode([tok])
        lines.append(f"| {r+1} | `{tok_s}` | {tok} | {reg_s} | {sup_s} | {same} | {lg_r} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_examples(indices, path, title, note, gold_t, topk_t, lgt_t,
                   ids_t, rid_t, d, tokenizer, unk_region, unk_super,
                   top_k, n_max=50):
    selected = indices[:n_max]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"_{note}_\n\n")
        f.write(f"Showing {len(selected)} of {len(indices)} matching rows.\n\n")
        f.write("---\n\n")
        for ex_idx, row_i in enumerate(selected):
            f.write(format_example_md(ex_idx + 1, row_i, gold_t, topk_t,
                                      lgt_t, ids_t, rid_t, d,
                                      tokenizer, unk_region, unk_super, top_k))
            f.write("\n---\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# Interpretation
# ─────────────────────────────────────────────────────────────────────────────

def _interpret(subsets, ks, top_k):
    """Heuristic verdict: supports/moderate/weak contrastive reranking."""
    m_all  = subsets.get("all", {})
    m_bw   = subsets.get("base_wrong", {})
    m_cov  = subsets.get("covered", {})
    m_bwcov = subsets.get("base_wrong_covered", {})

    sr_rate   = m_all.get("top1_same_region_rate", float("nan"))
    ss_rate   = m_all.get("top1_same_super_rate",  float("nan"))
    cov_rate  = m_all.get("covered_rate",           float("nan"))
    med_rank  = m_all.get("median_gold_rank_1b",    float("nan"))
    not_cov   = 1.0 - cov_rate if not math.isnan(cov_rate) else float("nan")

    near_rank_rate = float("nan")
    if not math.isnan(med_rank):
        near_rank_rate = 1.0 if med_rank <= 10 else 0.0

    hist = m_all.get("gold_rank_hist", {})
    top10_count = sum(
        hist.get(b, [0])[0] for b in ["rank_1", "rank_2", "rank_3_5", "rank_6_10"]
    )
    top10_total = m_all.get("n", 1)
    top10_rate = _safe_rate(top10_count, top10_total)

    lines = []
    lines.append("## Interpretation\n")

    q1_sym = "✅" if sr_rate >= 0.35 else ("⚠️" if sr_rate >= 0.20 else "❌")
    lines.append(f"**Q1 — Does top-1 tend to share region with gold?**  "
                 f"{q1_sym}  top1_same_region={sr_rate:.3f}  "
                 f"top1_same_super={ss_rate:.3f}")
    lines.append(f"> {'Region agreement is strong — contrastive reranking signal likely meaningful.' if sr_rate >= 0.35 else ('Region agreement is moderate — reranking may help in a subset of cases.' if sr_rate >= 0.20 else 'Region agreement is weak — contrastive reranking may not be well-motivated.')}\n")

    q2_sym = "✅" if cov_rate >= 0.70 else ("⚠️" if cov_rate >= 0.50 else "❌")
    lines.append(f"**Q2 — Is gold token frequently in the top-{top_k}?**  "
                 f"{q2_sym}  covered_rate={cov_rate:.3f}")
    lines.append(f"> {'Gold is frequently reachable — reranker has the correct answer to surface.' if cov_rate >= 0.70 else ('Gold is present roughly half the time — reranker can only help in that subset.' if cov_rate >= 0.50 else 'Gold is rarely in top candidates — reranking alone cannot fix coverage.')}\n")

    q3_sym = "✅" if top10_rate >= 0.40 else ("⚠️" if top10_rate >= 0.20 else "❌")
    lines.append(f"**Q3 — Is gold near the top (rank ≤ 10)?**  "
                 f"{q3_sym}  top10_rate={top10_rate:.3f}  "
                 f"median_gold_rank={med_rank:.1f}")
    lines.append(f"> {'Gold is often near the top — logit-gap reranking may suffice.' if top10_rate >= 0.40 else 'Gold is often buried deeper — strong reranker signal needed.'}\n")

    bw_sr = m_bwcov.get("top1_same_region_rate", float("nan"))
    q4_sym = "✅" if not math.isnan(bw_sr) and bw_sr >= 0.30 else "⚠️"
    lines.append(f"**Q4 — On base-wrong covered rows, does top-1 share region?**  "
                 f"{q4_sym}  top1_same_region={bw_sr:.3f}")
    lines.append(f"> {'Errors tend to stay within region — region-aware reranking may correct them.' if not math.isnan(bw_sr) and bw_sr >= 0.30 else 'Errors cross region boundaries frequently — region signal alone may be insufficient.'}\n")

    nc_sym = "✅" if not_cov <= 0.30 else ("⚠️" if not_cov <= 0.50 else "❌")
    lines.append(f"**Q5 — Not-covered rate (gold outside top-{top_k})?**  "
                 f"{nc_sym}  not_covered={not_cov:.3f}")
    lines.append(f"> {'Most gold tokens are reachable — reranker focus is justified.' if not_cov <= 0.30 else ('Coverage is partial — consider increasing top_k or fixing base model.' if not_cov <= 0.50 else 'Gold is frequently not in topK — fix base model first.')}\n")

    # Overall verdict
    strong = (sr_rate >= 0.35 and cov_rate >= 0.70 and top10_rate >= 0.40)
    moderate = (sr_rate >= 0.20 and cov_rate >= 0.50)
    if strong:
        verdict = "✅ STRONG SUPPORT for pairwise region contrastive reranking."
    elif moderate:
        verdict = "⚠️  MODERATE SUPPORT — reranking viable but gains may be limited."
    else:
        verdict = "❌ WEAK SUPPORT — revisit base model or retrieval before building reranker."

    lines.append(f"### Overall Verdict\n\n**{verdict}**\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Summary markdown
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, fmt=".4f"):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def write_summary_md(subsets, ks, top_k, output_dir, split_label, args,
                     unk_region, unk_super):
    path = os.path.join(output_dir, "summary.md")
    m_all = subsets.get("all", {})
    N     = m_all.get("n", 0)

    lines = []
    lines.append(f"# Region Gold Rank Audit — {split_label}\n")
    lines.append(f"**Split:** {split_label}  |  **N:** {N:,}  |  "
                 f"**top_k:** {top_k}  |  "
                 f"**unk_region sentinel:** {unk_region}  |  "
                 f"**unk_super sentinel:** {unk_super}\n")
    lines.append(f"**Shard dir:** `{args.val_dir}`\n")
    if getattr(args, "train_dir", None):
        lines.append(f"**Train dir:** `{args.train_dir}`\n")
    lines.append("")

    # Key scalars table
    lines.append("## Key Scalars\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key in [
        "base_correct_rate", "covered_rate",
        "top1_same_region_rate", "top1_same_super_rate",
        "gold_unmapped_rate", "top1_unmapped_rate",
        "mean_gold_rank_1b", "median_gold_rank_1b",
        "mean_region_local_rank_1b", "median_region_local_rank_1b",
        "mean_super_local_rank_1b", "median_super_local_rank_1b",
        "mean_logit_gap",
        "mean_first_same_reg_rank_1b", "mean_first_same_sup_rank_1b",
        "mean_num_same_reg_in_topK", "mean_num_same_sup_in_topK",
    ]:
        v = m_all.get(key, "—")
        lines.append(f"| `{key}` | {_fmt(v)} |")
    lines.append("")

    # Subset overview table
    lines.append("## Subset Overview\n")
    cols = ["n", "covered_rate", "top1_same_region_rate",
            "top1_same_super_rate", "median_gold_rank_1b"]
    lines.append("| Subset | " + " | ".join(cols) + " |")
    lines.append("|--------|" + "|".join(["---"] * len(cols)) + "|")
    for name in _SUBSET_ORDER:
        m = subsets.get(name, {})
        vals = [_fmt(m.get(c, "—")) for c in cols]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines.append("")

    # Gold rank histogram (all subset)
    lines.append("## Gold Rank Histogram (all rows)\n")
    hist = m_all.get("gold_rank_hist", {})
    lines.append("| Bin | Count | Rate |")
    lines.append("|-----|-------|------|")
    for b in [f"not_in_top{top_k}"] + [b[0] for b in _GLOBAL_RANK_BINS]:
        c, r = hist.get(b, [0, 0.0])
        lines.append(f"| {b} | {c:,} | {r:.4f} |")
    lines.append("")

    # Region-local rank histogram (covered rows)
    lines.append("## Region-Local Rank Histogram (covered rows)\n")
    rlh = subsets.get("covered", {}).get("region_local_rank_hist", {})
    lines.append("| Bin | Count | Rate |")
    lines.append("|-----|-------|------|")
    for b in ["not_covered_or_unk"] + [b[0] for b in _LOCAL_RANK_BINS]:
        c, r = rlh.get(b, [0, 0.0])
        lines.append(f"| {b} | {c:,} | {r:.4f} |")
    lines.append("")

    # TopK region curves
    lines.append("## TopK Region Curves (all rows)\n")
    cr = m_all.get("topk_region_curve", {})
    cs = m_all.get("topk_super_curve",  {})
    ks_capped = [min(k, top_k) for k in ks]
    lines.append("| k | has_same_region | has_same_super |")
    lines.append("|---|-----------------|----------------|")
    for k in ks_capped:
        rr = cr.get(f"top{k}_has_same_region", [0, float("nan")])[1]
        ss = cs.get(f"top{k}_has_same_super",  [0, float("nan")])[1]
        lines.append(f"| {k} | {_fmt(rr)} | {_fmt(ss)} |")
    lines.append("")

    # Interpretation section
    lines.append(_interpret(subsets, ks, top_k))

    # File inventory
    lines.append("## Output Files\n")
    for fname in [
        "subset_metrics.csv", "gold_rank_histograms.csv",
        "topk_region_curves.csv",
        "examples_gold_rank_1.md", "examples_covered_wrong_base.md",
        "examples_covered_high_region_rank.md", "examples_not_covered.md",
    ]:
        p = os.path.join(output_dir, fname)
        exists = "✓" if os.path.isfile(p) else "—"
        lines.append(f"- {exists} `{fname}`")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Run audit for one split
# ─────────────────────────────────────────────────────────────────────────────

def run_audit(shard_dir, split_name, t2r, r2s, unk_region, unk_super,
              ks, top_k, num_examples, tokenizer, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[{split_name}] Loading shards from {shard_dir}")

    gold_t, topk_t, lgt_t, ids_t, rid_t, N = load_shards(
        shard_dir, top_k, split_name)

    max_tok = max(int(gold_t.max()), int(topk_t.max()))
    tok_arr, reg_arr, max_reg = build_lookups(
        t2r, r2s, unk_region, unk_super, max_tok)

    print(f"[{split_name}] Computing derived fields for {N:,} rows ...")
    d = compute_all(gold_t, topk_t, lgt_t, tok_arr, reg_arr,
                    unk_region, unk_super, ks, top_k)

    print(f"[{split_name}] Building subset masks ...")
    masks = build_masks(d, unk_region, unk_super)

    print(f"[{split_name}] Computing subset metrics ...")
    subsets = compute_all_subsets(d, masks, unk_region, unk_super, ks, top_k)

    # Quick console summary
    m = subsets.get("all", {})
    print(f"\n[{split_name}] N={N:,}  "
          f"base_correct={m.get('base_correct_rate', float('nan')):.4f}  "
          f"covered={m.get('covered_rate', float('nan')):.4f}  "
          f"top1_same_reg={m.get('top1_same_region_rate', float('nan')):.4f}  "
          f"med_gold_rank={m.get('median_gold_rank_1b', float('nan')):.1f}  "
          f"gold_unmapped={m.get('gold_unmapped_rate', float('nan')):.4f}")

    # CSV outputs
    p1 = write_subset_csv(subsets, output_dir)
    p2 = write_hist_csv(subsets, output_dir, top_k)
    p3 = write_topk_csv(subsets, ks, output_dir, top_k)
    print(f"[{split_name}] Wrote {p1}")
    print(f"[{split_name}] Wrote {p2}")
    print(f"[{split_name}] Wrote {p3}")

    # Example buckets — select indices
    gold_rank_0b = d["gold_rank_0b"]
    covered      = d["covered"]
    base_correct = d["base_correct"]
    rlr          = d["region_local_rank"]

    # Bucket 1: gold rank = 1 (0-based index 0)
    b1 = np.where(gold_rank_0b == 0)[0]
    np.random.shuffle(b1)

    # Bucket 2: covered, base wrong
    b2 = np.where(covered & ~base_correct)[0]
    np.random.shuffle(b2)

    # Bucket 3: covered, base wrong, region_local_rank ≥ 5 (0-based ≥ 4)
    b3 = np.where(covered & ~base_correct & (rlr >= 4))[0]
    np.random.shuffle(b3)

    # Bucket 4: not covered, base wrong
    b4 = np.where(~covered & ~base_correct)[0]
    np.random.shuffle(b4)

    common_kw = dict(
        gold_t=gold_t, topk_t=topk_t, lgt_t=lgt_t,
        ids_t=ids_t, rid_t=rid_t, d=d,
        tokenizer=tokenizer, unk_region=unk_region,
        unk_super=unk_super, top_k=top_k, n_max=num_examples,
    )

    ex1 = os.path.join(output_dir, "examples_gold_rank_1.md")
    write_examples(b1, ex1,
        "Gold Rank = 1 (gold is top-1)",
        "Base top-1 already matches gold. Region context shown for reference.",
        **common_kw)

    ex2 = os.path.join(output_dir, "examples_covered_wrong_base.md")
    write_examples(b2, ex2,
        "Covered + Base Wrong",
        "Gold is in topK but base ranked another token first. "
        "Reranker could correct these.",
        **common_kw)

    ex3 = os.path.join(output_dir, "examples_covered_high_region_rank.md")
    write_examples(b3, ex3,
        "Covered + Base Wrong + Region-Local Rank ≥ 5",
        "Gold is in topK but buried within its own region. "
        "Hardest cases for region-contrastive reranking.",
        **common_kw)

    ex4 = os.path.join(output_dir, "examples_not_covered.md")
    write_examples(b4, ex4,
        "Not Covered (gold outside top-K)",
        f"Gold token was not in top-{top_k}. Reranking cannot fix these.",
        **common_kw)

    print(f"[{split_name}] Wrote example reports to {output_dir}/examples_*.md")

    # Summary markdown
    sp = write_summary_md(subsets, ks, top_k, output_dir, split_name,
                          args, unk_region, unk_super)
    print(f"[{split_name}] Wrote summary: {sp}")

    return subsets


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialization helper
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
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Pre-reranker Region Gold Rank Audit")
    p.add_argument("--val_dir",          required=True)
    p.add_argument("--train_dir",        default=None)
    p.add_argument("--token_to_region",  required=True)
    p.add_argument("--super_map",        default=None,
                   help="Path to region_to_superregion JSON (optional)")
    p.add_argument("--output_root",      default="runs/region_gold_rank_audit")
    p.add_argument("--top_k",            type=int, default=256)
    p.add_argument("--ks",               default="1,2,4,8,16,32,64,128,256",
                   help="Comma-separated list of k values for topK curves")
    p.add_argument("--num_examples",     type=int, default=50)
    p.add_argument("--tokenizer",        default="gpt2")
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


def main():
    args = _parse()
    np.random.seed(args.seed)

    ks = [int(x) for x in args.ks.split(",")]

    print("=" * 64)
    print(" Region Gold Rank Audit")
    print("=" * 64)
    print(f"  val_dir:          {args.val_dir}")
    print(f"  token_to_region:  {args.token_to_region}")
    print(f"  super_map:        {args.super_map}")
    print(f"  output_root:      {args.output_root}")
    print(f"  top_k:            {args.top_k}")
    print(f"  ks:               {ks}")
    print()

    print("[maps] Loading token→region and region→super maps ...")
    t2r, r2s, unk_region, unk_super = load_maps(
        args.token_to_region, args.super_map)
    print(f"  t2r size: {len(t2r):,}  |  r2s size: {len(r2s):,}  |  "
          f"unk_region={unk_region}  unk_super={unk_super}")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        print(f"[warn] Could not load tokenizer '{args.tokenizer}': {e}")
        tokenizer = None

    class _FakeTokenizer:
        def decode(self, ids, **kw):
            return str(ids)

    if tokenizer is None:
        tokenizer = _FakeTokenizer()

    os.makedirs(args.output_root, exist_ok=True)

    # Val split
    val_subsets = run_audit(
        shard_dir=args.val_dir,
        split_name="val",
        t2r=t2r, r2s=r2s,
        unk_region=unk_region, unk_super=unk_super,
        ks=ks, top_k=args.top_k,
        num_examples=args.num_examples,
        tokenizer=tokenizer,
        output_dir=args.output_root,
        args=args,
    )

    # Train split (optional)
    if args.train_dir:
        train_out = os.path.join(args.output_root, "train")
        run_audit(
            shard_dir=args.train_dir,
            split_name="train",
            t2r=t2r, r2s=r2s,
            unk_region=unk_region, unk_super=unk_super,
            ks=ks, top_k=args.top_k,
            num_examples=args.num_examples,
            tokenizer=tokenizer,
            output_dir=train_out,
            args=args,
        )

    # Write combined JSON
    combined = {
        "val": _json_safe(
            {k: {kk: vv for kk, vv in v.items()
                 if not isinstance(vv, dict) or kk not in (
                     "gold_rank_hist", "region_local_rank_hist",
                     "super_local_rank_hist", "topk_region_curve",
                     "topk_super_curve"
                 )}
             for k, v in val_subsets.items()}
        ),
    }
    jpath = os.path.join(args.output_root, "audit_results.json")
    with open(jpath, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[done] Wrote combined JSON: {jpath}")

    print("\n" + "=" * 64)
    print(" Region Gold Rank Audit complete.")
    print("=" * 64)
    print(f"  Outputs: {args.output_root}/")
    print(f"    summary.md")
    print(f"    subset_metrics.csv")
    print(f"    gold_rank_histograms.csv")
    print(f"    topk_region_curves.csv")
    print(f"    examples_*.md")
    print()


if __name__ == "__main__":
    main()
