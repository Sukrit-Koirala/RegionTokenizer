#!/usr/bin/env python3
"""
analyze_pointer_help_slices_v1.py — Focused pointer-help slice diagnostic.

Phase 2: measures whether pointer support helps exactly where intended.
NO training. NO model changes. Gold used only after scoring, for metrics only.

Questions answered:
  Q1. What fraction of Bucket A is copy-supported?
  Q2. When gold is in context, how often is it pointer top1/3/5?
  Q3. Does pointer improve sip in gold_exact_present?
  Q4. Does pointer hurt when gold is not present?
  Q5. Does pointer reduce changed_away or increase changed_to_gold?
  Q6. Is there a usable confidence/margin threshold?
  Q7. Oracle gain on pointer_should_help_strong?
  Q8. Should pointer become a memory expert in the final mixer?

Usage:
  python scripts/analyze_pointer_help_slices_v1.py \\
    --val_dir runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val \\
    --phase1_dir runs/pointer_sentinel_v1/pointer_support_baseline \\
    --token_to_region runs/region_maps_128/token_to_region.json \\
    --output_dir runs/pointer_sentinel_v1/pointer_help_slices_v1 \\
    --top_k 256 --candidate_pool_size 32 --memory_len 128 --recency_tau 32 \\
    --lambda_grid 0.0,0.25,0.5,1.0,2.0,4.0 --seed 42
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

_enc = None
try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
except Exception:
    pass

def _decode_tok(tid):
    if _enc is not None:
        try:
            return repr(_enc.decode([int(tid)]))[1:-1]
        except Exception:
            pass
    return f"<{tid}>"

def _decode_ids(ids):
    if _enc is not None:
        try:
            return _enc.decode([int(i) for i in ids if 0 <= int(i) < 50257])
        except Exception:
            pass
    return " ".join(_decode_tok(i) for i in ids)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

_TOPK_ALIASES = ["base_topk_ids",   "base_topk",    "topk_ids"]
_LGT_ALIASES  = ["base_topk_logits","base_topk_lgt","topk_lgt","topk_logits"]
_GOLD_ALIASES = ["gold_token",      "gold",         "labels"]
_IDS_ALIASES  = ["input_ids"]

def _get(shard, aliases, required=True):
    for a in aliases:
        if a in shard:
            return shard[a]
    if required:
        raise KeyError(f"Shard missing field. Tried {aliases}. Have: {list(shard.keys())}")
    return None

def load_shards(shard_dir, top_k, max_rows=None):
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")
    print(f"[shards] {len(paths)} shard(s) in {shard_dir}")
    bufs = defaultdict(list)
    total = 0
    first = True
    for sp in paths:
        if max_rows is not None and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print("[preflight] shard keys:")
            for k, v in sh.items():
                print(f"  {k}: shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',type(v).__name__)}")
            first = False
        topk = _get(sh, _TOPK_ALIASES).long()
        lgt  = _get(sh, _LGT_ALIASES).float()
        gold = _get(sh, _GOLD_ALIASES).long()
        ids  = _get(sh, _IDS_ALIASES).long()
        B, K = topk.shape
        if K < top_k:
            topk = torch.cat([topk, torch.zeros(B, top_k-K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, top_k-K), float("nan"))], 1)
        elif K > top_k:
            topk, lgt = topk[:, :top_k], lgt[:, :top_k]
        if max_rows is not None and total+B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, ids = topk[:keep], lgt[:keep], gold[:keep], ids[:keep]
            B = keep
        bufs["topk"].append(topk.numpy())
        bufs["lgt"].append(lgt.numpy())
        bufs["gold"].append(gold.numpy())
        bufs["ids"].append(ids.numpy())
        total += B
    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    print(f"[shards] {total:,} rows  K={top_k}  seq={data['ids'].shape[1]}")
    return data

def load_maps(t2r_path, super_path=None):
    with open(t2r_path) as f: raw = json.load(f)
    if isinstance(raw, list):
        t2r = {i: v for i, v in enumerate(raw) if v is not None}
    else:
        t2r = {int(k): v for k, v in raw.items()}
    unk_r = int(max(t2r.values())) + 1 if t2r else 1
    r2s = {}; unk_s = 1; sr = False
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f: raw2 = json.load(f)
        r2s = ({int(k): v for k, v in raw2.items()} if not isinstance(raw2, list)
               else {i: v for i, v in enumerate(raw2) if v is not None})
        unk_s = int(max(r2s.values())) + 1 if r2s else 1
        sr = True
    V = max(t2r.keys(), default=0) + 2
    tok_arr = np.full(V, unk_r, dtype=np.int32)
    for t, r in t2r.items():
        if 0 <= t < V: tok_arr[t] = int(r)
    R = unk_r + 2
    reg_arr = np.full(R, unk_s, dtype=np.int32)
    for r, s in r2s.items():
        if 0 <= r < R: reg_arr[int(r)] = int(s)
    return tok_arr, reg_arr, unk_r, unk_s, sr

# ─────────────────────────────────────────────────────────────────────────────
# Pointer features (vectorised per batch)
# ─────────────────────────────────────────────────────────────────────────────

def pointer_features(input_ids_np, cand_ids_np, memory_len, tau):
    """Returns exact_count, recency, nearest_dist, exact_present per [N,P]."""
    N, _ = input_ids_np.shape
    P    = cand_ids_np.shape[1]
    T    = memory_len
    ctx  = input_ids_np[:, -T:].astype(np.int32)           # [N, T]
    dists      = np.arange(T-1, -1, -1, dtype=np.float32)  # dist=0 = most recent
    wts        = np.exp(-dists / max(tau, 1e-6))
    _INF       = float(T + 9999)

    exact_count  = np.zeros((N, P), dtype=np.int32)
    recency      = np.zeros((N, P), dtype=np.float32)
    nearest_dist = np.full((N, P), _INF, dtype=np.float32)

    for j in range(P):
        c = cand_ids_np[:, j]                              # [N]
        m = (ctx == c[:, None]).astype(np.float32)         # [N, T]
        exact_count[:, j]  = m.sum(axis=1).astype(np.int32)
        recency[:, j]      = (m * wts).sum(axis=1)
        nd_m               = dists * m + _INF * (1.0 - m)
        nearest_dist[:, j] = nd_m.min(axis=1)

    exact_present  = exact_count > 0
    last_occ_score = (1.0 / np.sqrt(1.0 + nearest_dist)).astype(np.float32)
    return exact_count, recency, nearest_dist, exact_present, last_occ_score

def pointer_row_stats(ptr_scores, exact_present, gold_ids, base_logit_top_pool_idx, cand_ids):
    """
    Returns per-row pointer stats (all [N]):
      ptr_top_idx, ptr_top_tok, ptr_top_score,
      ptr_entropy, ptr_margin, ptr_confidence,
      gold_ptr_score, gold_ptr_rank, base_ptr_score,
      ptr_top_is_gold, ptr_top_agrees_logit
    """
    N, P = ptr_scores.shape
    ar   = np.arange(N)
    _EPS = 1e-9

    ptr_top_idx  = ptr_scores.argmax(axis=1)
    ptr_top_tok  = cand_ids[ar, ptr_top_idx]
    ptr_top_score = ptr_scores[ar, ptr_top_idx]

    # Confidence = top_score / sum (eps-safe)
    ptr_sum      = ptr_scores.sum(axis=1).clip(_EPS)
    ptr_conf     = ptr_top_score / ptr_sum

    # Margin = top1 - top2 (eps-safe for P==1)
    if P >= 2:
        sorted_s  = np.sort(ptr_scores, axis=1)[:, ::-1]
        ptr_margin = sorted_s[:, 0] - sorted_s[:, 1]
    else:
        ptr_margin = np.zeros(N, dtype=np.float32)

    # Entropy (after softmax)
    shifted   = ptr_scores - ptr_scores.max(axis=1, keepdims=True)
    e_s       = np.exp(shifted)
    soft      = e_s / e_s.sum(axis=1, keepdims=True).clip(_EPS)
    soft      = np.clip(soft, _EPS, None)
    ptr_ent   = -(soft * np.log(soft)).sum(axis=1)

    # Gold pointer rank and score
    gold_ptr_rank  = np.full(N, P, dtype=np.int32)   # P = "not in pool"
    gold_ptr_score = np.zeros(N, dtype=np.float32)
    for i in range(N):
        gold_id = gold_ids[i]
        match   = np.where(cand_ids[i] == gold_id)[0]
        if len(match) > 0:
            j = match[0]
            gold_ptr_score[i] = ptr_scores[i, j]
            # rank = number of pool candidates with higher ptr_score
            gold_ptr_rank[i] = int((ptr_scores[i] > ptr_scores[i, j]).sum())

    # Base logit top-pool pointer score
    base_ptr_score = ptr_scores[ar, base_logit_top_pool_idx]

    return {
        "ptr_top_idx":    ptr_top_idx,
        "ptr_top_tok":    ptr_top_tok,
        "ptr_top_score":  ptr_top_score,
        "ptr_confidence": ptr_conf,
        "ptr_margin":     ptr_margin,
        "ptr_entropy":    ptr_ent,
        "gold_ptr_score": gold_ptr_score,
        "gold_ptr_rank":  gold_ptr_rank,
        "base_ptr_score": base_ptr_score,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Surgical edit (same convention as training: boost sel by md_half, base -md_half)
# ─────────────────────────────────────────────────────────────────────────────

def surgical_edit(topk_ids, topk_lgt, sel_toks, apply_mask, md_half=0.5):
    N, K = topk_ids.shape
    ar   = np.arange(N)
    ref  = np.where(np.isfinite(topk_lgt), topk_lgt, -1e9)
    ref[apply_mask, 0] -= md_half
    edit_rows = np.where(apply_mask)[0]
    if len(edit_rows):
        st  = sel_toks[edit_rows]
        pos = (topk_ids[edit_rows] == st[:, None]).argmax(1)
        ok  = topk_ids[edit_rows, pos] == st
        ref[edit_rows[ok], pos[ok]] += md_half
    return topk_ids[ar, ref.argmax(1)]

# ─────────────────────────────────────────────────────────────────────────────
# Slice definitions
# ─────────────────────────────────────────────────────────────────────────────

SLICE_NAMES = [
    "all",
    "bucketA_confuser",
    "gold_exact_present",
    "gold_recent",
    "gold_repeated",
    "gold_high_recency",
    "base_wrong_gold_present",
    "base_correct_selected_present",
    "candidate_pointer_advantage",
    "pointer_disagrees_with_base",
    "pointer_agrees_with_gold",
    "pointer_agrees_with_base",
    "pointer_misleading",
    "pointer_should_help_strong",
    "pointer_should_help_weak",
    "pointer_should_not_help",
    "pointer_danger",
    "same_region_confuser",
    "same_superregion_confuser",
]

def build_slices(gold, base_top1, cand_ids, gold_in_pool, ctx, ptr_scores,
                 exact_count, nearest_dist, exact_present,
                 logit_top_pool_idx, pstats, tok_arr, reg_arr, unk_r, unk_s, sr,
                 memory_len, recency_tau, high_recency_thresh=0.05,
                 ptr_conf_thresh_danger=0.5):
    N  = len(gold)
    V  = tok_arr.shape[0]
    R  = reg_arr.shape[0]
    ar = np.arange(N)
    T  = ctx.shape[1]
    _INF = float(T + 9999)

    base_wrong   = base_top1 != gold
    base_correct = ~base_wrong

    # Gold in context (last memory_len tokens)
    gold_in_ctx   = exact_count[:, 0] > 0  # will be computed below per gold token
    # Recompute for gold token (not a candidate — need separate check)
    gold_in_ctx_arr = np.zeros(N, dtype=bool)
    gold_recent_arr = np.zeros(N, dtype=bool)
    gold_repeated_arr = np.zeros(N, dtype=bool)
    gold_nearest_arr = np.full(N, _INF, dtype=np.float32)
    gold_recency_arr = np.zeros(N, dtype=np.float32)

    dists = np.arange(T-1, -1, -1, dtype=np.float32)
    wts   = np.exp(-dists / max(recency_tau, 1e-6))

    # Vectorised gold lookup in ctx
    for i in range(N):
        g = gold[i]
        m = ctx[i] == g
        gold_in_ctx_arr[i]   = m.any()
        gold_repeated_arr[i] = m.sum() >= 2
        gold_recency_arr[i]  = float((m.astype(np.float32) * wts).sum())
        # recent = within last 32
        gold_recent_arr[i]   = m[-min(32, T):].any()
        if m.any():
            gold_nearest_arr[i] = float(dists[m].min())

    # Gold pointer rank (already in pstats)
    gold_ptr_rank   = pstats["gold_ptr_rank"]       # [N]
    ptr_top_tok     = pstats["ptr_top_tok"]
    ptr_top_score   = pstats["ptr_top_score"]
    ptr_conf        = pstats["ptr_confidence"]
    ptr_margin      = pstats["ptr_margin"]

    # Pointer agrees/disagrees with base (within pool)
    logit_top_tok = cand_ids[ar, logit_top_pool_idx]
    ptr_top_idx   = pstats["ptr_top_idx"]
    ptr_top_pool_tok = cand_ids[ar, ptr_top_idx]

    # base_correct_selected_present: base==gold AND ptr_top is in context
    # (risk: pointer may edit away from gold)
    ptr_top_exact = exact_present[ar, ptr_top_idx]

    # candidate_pointer_advantage: gold ptr_score > all non-gold pool candidates
    gold_ptr_score = pstats["gold_ptr_score"]
    # max ptr score excluding gold
    non_gold_max_ptr = np.zeros(N, dtype=np.float32)
    for j in range(ptr_scores.shape[1]):
        mask_non_gold = cand_ids[:, j] != gold
        non_gold_max_ptr = np.where(mask_non_gold,
                                    np.maximum(non_gold_max_ptr, ptr_scores[:, j]),
                                    non_gold_max_ptr)
    cand_ptr_adv = (gold_in_pool & (gold_ptr_score > non_gold_max_ptr + 1e-9))

    # Region info
    base_reg = tok_arr[base_top1.clip(0, V-1)]
    gold_reg = tok_arr[gold.clip(0, V-1)]
    same_reg = (base_reg == gold_reg) & (base_reg != unk_r)
    same_sup = np.zeros(N, dtype=bool)
    if sr:
        base_sup = reg_arr[base_reg.clip(0, R-1)]
        gold_sup = reg_arr[gold_reg.clip(0, R-1)]
        same_sup = (base_sup == gold_sup) & (base_sup != unk_s)

    # Broad buckets
    psh_strong = (base_wrong & gold_in_pool & gold_in_ctx_arr &
                  ((ptr_top_pool_tok == gold) | (gold_ptr_rank <= 2)))
    psh_weak   = (base_wrong & gold_in_pool & gold_in_ctx_arr &
                  (gold_ptr_rank <= 9))
    psnh       = (base_wrong & gold_in_pool & ~gold_in_ctx_arr)
    pdanger    = (base_correct & (ptr_top_pool_tok != gold) &
                  (ptr_conf >= ptr_conf_thresh_danger) & (ptr_top_score > 0))

    return {
        "all":                          np.ones(N, dtype=bool),
        "bucketA_confuser":             base_wrong & gold_in_pool,
        "gold_exact_present":           gold_in_ctx_arr,
        "gold_recent":                  gold_recent_arr,
        "gold_repeated":                gold_repeated_arr,
        "gold_high_recency":            gold_recency_arr > high_recency_thresh,
        "base_wrong_gold_present":      base_wrong & gold_in_ctx_arr,
        "base_correct_selected_present": base_correct & ptr_top_exact,
        "candidate_pointer_advantage":  cand_ptr_adv,
        "pointer_disagrees_with_base":  ptr_top_idx != logit_top_pool_idx,
        "pointer_agrees_with_gold":     ptr_top_pool_tok == gold,
        "pointer_agrees_with_base":     ptr_top_idx == logit_top_pool_idx,
        "pointer_misleading":           (ptr_top_pool_tok != gold) & (base_correct | gold_in_ctx_arr),
        "pointer_should_help_strong":   psh_strong,
        "pointer_should_help_weak":     psh_weak,
        "pointer_should_not_help":      psnh,
        "pointer_danger":               pdanger,
        "same_region_confuser":         base_wrong & gold_in_pool & same_reg,
        "same_superregion_confuser":    base_wrong & gold_in_pool & same_sup,
    }, gold_in_ctx_arr, gold_ptr_rank, gold_recency_arr, gold_nearest_arr

# ─────────────────────────────────────────────────────────────────────────────
# Metric accumulator
# ─────────────────────────────────────────────────────────────────────────────

class MetricAcc:
    __slots__ = ["n", "apply", "ctg", "caw", "base_ok", "final_ok",
                 "sip_n", "sip_match", "gip_n"]
    def __init__(self):
        self.n = self.apply = self.ctg = self.caw = 0
        self.base_ok = self.final_ok = 0
        self.sip_n = self.sip_match = self.gip_n = 0

    def update(self, gold, base_top1, final_top1, apply_mask, gold_in_pool, sel_tok, subset):
        g  = gold[subset]; b = base_top1[subset]; ft = final_top1[subset]
        am = apply_mask[subset]; gip = gold_in_pool[subset]; st = sel_tok[subset]
        self.n        += len(g)
        self.apply    += am.sum()
        self.ctg      += (am & (ft == g)).sum()
        self.caw      += (am & (b == g) & (ft != g)).sum()
        self.base_ok  += (b == g).sum()
        self.final_ok += (ft == g).sum()
        self.gip_n    += gip.sum()
        self.sip_n    += gip.sum()
        self.sip_match += (gip & (st == g)).sum()

    def metrics(self):
        _E = 1e-9
        n = max(self.n, 1)
        return {
            "n":                    self.n,
            "apply_rate":           self.apply / n,
            "changed_to_gold":      self.ctg / n,
            "changed_away":         self.caw / n,
            "net_correction":       (self.ctg - self.caw) / n,
            "bdr":                  self.ctg / max(self.caw, _E),
            "applied_prec_ctg":     self.ctg / max(self.apply, 1),
            "base_acc":             self.base_ok / n,
            "policy_acc":           self.final_ok / n,
            "acc_gain":             (self.final_ok - self.base_ok) / n,
            "sip":                  self.sip_match / max(self.sip_n, 1),
            "gold_in_pool_rate":    self.gip_n / n,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Rank histogram accumulator
# ─────────────────────────────────────────────────────────────────────────────

class RankAcc:
    def __init__(self, P):
        self.P = P
        self.hist = np.zeros(P + 1, dtype=np.int64)  # +1 for "not in pool"
        self.n_in_pool = 0
        self.sum_conf  = 0.0
        self.sum_marg  = 0.0
        self.n_total   = 0
        self.ptr_top_is_gold_n = 0

    def update(self, gold_ptr_rank, gold_in_pool, ptr_conf, ptr_margin, ptr_top_is_gold, subset):
        gpr = gold_ptr_rank[subset]
        gip = gold_in_pool[subset]
        for r in gpr:
            self.hist[min(r, self.P)] += 1
        self.n_in_pool           += gip.sum()
        self.sum_conf            += ptr_conf[subset].sum()
        self.sum_marg            += ptr_margin[subset].sum()
        self.n_total             += subset.sum()
        self.ptr_top_is_gold_n   += ptr_top_is_gold[subset].sum()

    def stats(self):
        n = max(self.n_total, 1)
        return {
            "n":                   int(self.n_total),
            "ptr_top_is_gold":     self.ptr_top_is_gold_n / n,
            "rank_le_0":           int(self.hist[0]),  # rank 0 = best
            "rank_le_1":           int(self.hist[:2].sum()),
            "rank_le_3":           int(self.hist[:4].sum()),
            "rank_le_5":           int(self.hist[:6].sum()),
            "rank_le_10":          int(self.hist[:11].sum()),
            "mean_ptr_confidence": self.sum_conf / n,
            "mean_ptr_margin":     self.sum_marg / n,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Policy list
# ─────────────────────────────────────────────────────────────────────────────

def make_policies(lambda_grid, conf_thresholds, margin_thresholds):
    """
    Returns list of (policy_name, policy_fn).
    policy_fn(cand_ids, cand_lgts, ptr_scores, exact_present, ptr_top_idx, ptr_conf, ptr_margin,
              gold_in_pool, should_help_strong) -> (sel_tok [N], apply_mask [N])
    """
    policies = []

    def base_candidate(cand_ids, cand_lgts, ptr_scores, exact_present,
                        ptr_top_idx, ptr_conf, ptr_margin,
                        gold_in_pool, should_help_strong):
        sel_idx = cand_lgts.argmax(axis=1)
        return cand_ids[np.arange(len(sel_idx)), sel_idx], np.ones(len(sel_idx), bool)

    policies.append(("base_candidate", base_candidate))

    def pointer_only(cand_ids, cand_lgts, ptr_scores, exact_present,
                      ptr_top_idx, ptr_conf, ptr_margin,
                      gold_in_pool, should_help_strong):
        N = len(ptr_top_idx)
        ar = np.arange(N)
        sel_tok = cand_ids[ar, ptr_top_idx]
        all_zero = (ptr_scores.max(axis=1) <= 0)
        apply    = ~all_zero
        return sel_tok, apply

    policies.append(("pointer_only", pointer_only))

    for lam in lambda_grid:
        lam_ = lam
        def make_lam(l):
            def policy(cand_ids, cand_lgts, ptr_scores, exact_present,
                       ptr_top_idx, ptr_conf, ptr_margin,
                       gold_in_pool, should_help_strong):
                scores  = cand_lgts + l * ptr_scores
                sel_idx = scores.argmax(axis=1)
                return cand_ids[np.arange(len(sel_idx)), sel_idx], np.ones(len(sel_idx), bool)
            return policy
        policies.append((f"lambda_{lam_:.2f}", make_lam(lam_)))

    for conf_t in conf_thresholds:
        for marg_t in margin_thresholds:
            ct, mt = conf_t, marg_t
            def make_gated(c, m):
                def policy(cand_ids, cand_lgts, ptr_scores, exact_present,
                            ptr_top_idx, ptr_conf, ptr_margin,
                            gold_in_pool, should_help_strong):
                    N  = len(ptr_top_idx)
                    ar = np.arange(N)
                    gate = (ptr_conf >= c) & (ptr_margin >= m) & (ptr_scores[ar, ptr_top_idx] > 0)
                    sel_tok_ptr  = cand_ids[ar, ptr_top_idx]
                    sel_tok_base = cand_ids[cand_lgts.argmax(axis=1)]  # fallback
                    sel_tok_base2 = cand_ids[ar, cand_lgts.argmax(axis=1)]
                    sel_tok = np.where(gate, sel_tok_ptr, sel_tok_base2)
                    return sel_tok, gate
                return policy
            policies.append((f"gated_c{ct}_m{mt}", make_gated(ct, mt)))

    # Oracle: use pointer only on pointer_should_help_strong
    def oracle_slice(cand_ids, cand_lgts, ptr_scores, exact_present,
                      ptr_top_idx, ptr_conf, ptr_margin,
                      gold_in_pool, should_help_strong):
        N  = len(ptr_top_idx)
        ar = np.arange(N)
        sel_tok_ptr  = cand_ids[ar, ptr_top_idx]
        sel_tok_base = cand_ids[ar, cand_lgts.argmax(axis=1)]
        sel_tok  = np.where(should_help_strong, sel_tok_ptr, sel_tok_base)
        apply    = np.ones(N, dtype=bool)
        return sel_tok, apply

    policies.append(("oracle_DIAGNOSTIC", oracle_slice))

    return policies

# ─────────────────────────────────────────────────────────────────────────────
# Example collectors
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_KINDS = ["pointer_helps", "pointer_hurts",
                 "should_help_fails", "pointer_danger"]

def _example_row(i, row_idx, gold, base_top1, topk_lgt, cand_ids, cand_lgts, ptr_scores,
                 exact_count, nearest_dist, exact_present, ctx, pstats, best_lam, gold_in_ctx,
                 gold_ptr_rank, gold_nearest):
    P   = cand_ids.shape[0]
    scores_disp = cand_lgts + best_lam * ptr_scores
    sel_idx_d   = scores_disp.argmax()

    cands = []
    for j in range(min(P, 16)):
        nd = nearest_dist[j]
        cands.append({
            "tok_id":       int(cand_ids[j]),
            "base_logit":   float(cand_lgts[j]),
            "ptr_score":    float(ptr_scores[j]),
            "exact_count":  int(exact_count[j]),
            "nearest_dist": float(nd) if nd < 9000 else float("inf"),
            "final_score":  float(scores_disp[j]),
            "is_gold":      int(cand_ids[j]) == int(gold),
            "is_base_top1": False,   # base_top1 not in pool by design
            "is_ptr_top":   j == int(pstats["ptr_top_idx"]),
            "is_selected":  j == int(sel_idx_d),
        })

    return {
        "row_idx":         int(row_idx),
        "ctx_ids":         ctx[-128:].tolist(),
        "gold_id":         int(gold),
        "base_top1_id":    int(base_top1),
        "base_top1_lgt":   float(topk_lgt[0]),
        "ptr_top_tok":     int(pstats["ptr_top_tok"]),
        "ptr_top_score":   float(pstats["ptr_top_score"]),
        "sel_lam_tok":     int(cand_ids[sel_idx_d]),
        "gold_in_ctx":     bool(gold_in_ctx),
        "gold_nearest_d":  float(gold_nearest) if gold_nearest < 9000 else float("inf"),
        "gold_ptr_rank":   int(gold_ptr_rank),
        "ptr_confidence":  float(pstats["ptr_confidence"]),
        "ptr_margin":      float(pstats["ptr_margin"]),
        "candidates":      cands,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(data, tok_arr, reg_arr, unk_r, unk_s, sr, args):
    topk_ids = data["topk"]    # [N, K]
    topk_lgt = data["lgt"]     # [N, K]
    gold     = data["gold"]    # [N]
    inp_ids  = data["ids"]     # [N, seq]
    N        = len(gold)
    P        = args.candidate_pool_size
    T        = args.memory_len
    md_half  = 0.5
    ar       = np.arange(N)

    lam_grid   = [float(x) for x in args.lambda_grid.split(",")]
    conf_ts    = [float(x) for x in args.pointer_conf_thresholds.split(",")]
    margin_ts  = [float(x) for x in args.pointer_margin_thresholds.split(",")]

    # ── Pool ─────────────────────────────────────────────────────────────────
    cand_ids  = topk_ids[:, 1:P+1].copy()
    cand_lgts = np.where(np.isfinite(topk_lgt[:, 1:P+1]),
                         topk_lgt[:, 1:P+1], -1e9)
    base_top1 = topk_ids[:, 0]
    gold_in_pool = (cand_ids == gold[:, None]).any(axis=1)

    # ── Pointer features ─────────────────────────────────────────────────────
    print("[eval] Computing pointer features...")
    exact_count, recency, nearest_dist, exact_present, _ = pointer_features(
        inp_ids, cand_ids, T, args.recency_tau)
    ptr_scores = recency   # [N, P]

    # Logit top pool idx (base_candidate selector)
    logit_top_pool_idx = cand_lgts.argmax(axis=1)

    # Per-row pointer stats
    pstats = pointer_row_stats(ptr_scores, exact_present, gold, logit_top_pool_idx, cand_ids)
    ptr_top_is_gold = pstats["ptr_top_tok"] == gold

    # ── Context tail ─────────────────────────────────────────────────────────
    ctx = inp_ids[:, -T:]    # [N, T]

    # ── Slices ───────────────────────────────────────────────────────────────
    print("[eval] Building slices...")
    slices, gold_in_ctx, gold_ptr_rank, gold_recency, gold_nearest = build_slices(
        gold, base_top1, cand_ids, gold_in_pool, ctx, ptr_scores,
        exact_count, nearest_dist, exact_present,
        logit_top_pool_idx, pstats, tok_arr, reg_arr, unk_r, unk_s, sr,
        T, args.recency_tau)

    # ── Policies ─────────────────────────────────────────────────────────────
    policies = make_policies(lam_grid, conf_ts, margin_ts)
    print(f"[eval] {len(policies)} policies × {len(SLICE_NAMES)} slices")

    # Precompute final_top1 and sel_tok for each policy (vectorised)
    policy_results = {}   # policy_name -> (sel_tok [N], apply [N], final_top1 [N])
    psh_strong = slices["pointer_should_help_strong"]

    for pname, pfn in policies:
        sel_tok, apply_mask = pfn(
            cand_ids, cand_lgts, ptr_scores, exact_present,
            pstats["ptr_top_idx"], pstats["ptr_confidence"], pstats["ptr_margin"],
            gold_in_pool, psh_strong)
        ft = surgical_edit(topk_ids.copy(), topk_lgt.copy(), sel_tok, apply_mask, md_half)
        policy_results[pname] = (sel_tok, apply_mask, ft)

    # ── Metric accumulators ───────────────────────────────────────────────────
    metrics = {sn: {pn: MetricAcc() for pn, _ in policies} for sn in SLICE_NAMES}
    rank_stats = {sn: RankAcc(P) for sn in SLICE_NAMES}

    print("[eval] Accumulating metrics...")
    for sn in SLICE_NAMES:
        subset = slices[sn]
        if not subset.any():
            continue
        rank_stats[sn].update(gold_ptr_rank, gold_in_pool, pstats["ptr_confidence"],
                               pstats["ptr_margin"], ptr_top_is_gold, subset)
        for pname, _ in policies:
            sel_tok, apply_mask, ft = policy_results[pname]
            metrics[sn][pname].update(gold, base_top1, ft, apply_mask, gold_in_pool, sel_tok, subset)

    # ── Collect examples ─────────────────────────────────────────────────────
    print("[eval] Collecting examples...")
    best_lam = max(lam_grid, key=lambda l: (
        metrics["all"][f"lambda_{l:.2f}"].metrics().get("net_correction", -1e9)))

    ex_buckets = {k: [] for k in EXAMPLE_KINDS}
    max_per    = args.max_examples // len(EXAMPLE_KINDS)
    rng        = random.Random(args.seed)

    def candidate_idx(i):
        return _example_row(
            None, i, gold[i], base_top1[i], topk_lgt[i],
            cand_ids[i], cand_lgts[i], ptr_scores[i],
            exact_count[i], nearest_dist[i], exact_present[i],
            inp_ids[i], {k: v[i] for k, v in pstats.items()},
            best_lam, gold_in_ctx[i], gold_ptr_rank[i], gold_nearest[i])

    idxs = list(range(N))
    rng.shuffle(idxs)

    for i in idxs:
        bw  = base_top1[i] != gold[i]
        gip = gold_in_pool[i]
        gic = gold_in_ctx[i]
        _, _, ft_best = policy_results[f"lambda_{best_lam:.2f}"]

        # pointer_helps: base wrong, gold in pool & ctx, ptr correctly selected gold
        if (len(ex_buckets["pointer_helps"]) < max_per and
                bw and gip and gic and ptr_top_is_gold[i] and ft_best[i] == gold[i]):
            ex_buckets["pointer_helps"].append(candidate_idx(i))

        # pointer_hurts: base correct, ptr selects non-gold present candidate
        if (len(ex_buckets["pointer_hurts"]) < max_per and
                not bw and exact_present[i, pstats["ptr_top_idx"][i]] and
                pstats["ptr_top_tok"][i] != gold[i]):
            ex_buckets["pointer_hurts"].append(candidate_idx(i))

        # should_help_fails: pointer_should_help_strong but ptr_top != gold
        if (len(ex_buckets["should_help_fails"]) < max_per and
                psh_strong[i] and not ptr_top_is_gold[i]):
            ex_buckets["should_help_fails"].append(candidate_idx(i))

        # pointer_danger: base==gold, high ptr confidence on non-gold
        if (len(ex_buckets["pointer_danger"]) < max_per and
                slices["pointer_danger"][i]):
            ex_buckets["pointer_danger"].append(candidate_idx(i))

        if all(len(v) >= max_per for v in ex_buckets.values()):
            break

    return metrics, rank_stats, slices, policy_results, ex_buckets, policies, lam_grid, pstats, gold_in_ctx, ptr_top_is_gold

# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_KEYS = ["n", "apply_rate", "changed_to_gold", "changed_away",
                "net_correction", "bdr", "applied_prec_ctg",
                "base_acc", "policy_acc", "acc_gain", "sip", "gold_in_pool_rate"]

def write_slice_summary_csv(metrics, policies, out_path):
    fields = ["policy", "slice"] + _METRIC_KEYS
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sn in SLICE_NAMES:
            for pname, _ in policies:
                m = metrics[sn][pname].metrics()
                row = {"policy": pname, "slice": sn}
                for k in _METRIC_KEYS:
                    v = m.get(k, "")
                    row[k] = f"{v:.6f}" if isinstance(v, float) else v
                w.writerow(row)
    print(f"[csv] {out_path}")


def write_policy_grid_csv(metrics, policies, slices_of_interest, out_path):
    """Focus grid: key slices × all policies."""
    fields = ["policy", "slice"] + _METRIC_KEYS
    focus  = ["all", "bucketA_confuser", "gold_exact_present",
              "pointer_should_help_strong", "pointer_should_not_help", "pointer_danger"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sn in focus:
            if sn not in metrics:
                continue
            for pname, _ in policies:
                m = metrics[sn][pname].metrics()
                row = {"policy": pname, "slice": sn}
                for k in _METRIC_KEYS:
                    v = m.get(k, "")
                    row[k] = f"{v:.6f}" if isinstance(v, float) else v
                w.writerow(row)
    print(f"[csv] {out_path}")


def write_rank_stats_csv(rank_stats, out_path):
    fields = ["slice", "n", "ptr_top_is_gold", "rank_le_0", "rank_le_1",
              "rank_le_3", "rank_le_5", "rank_le_10",
              "mean_ptr_confidence", "mean_ptr_margin"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sn in SLICE_NAMES:
            s  = rank_stats[sn].stats()
            n  = max(s["n"], 1)
            row = {"slice": sn}
            row["n"] = s["n"]
            row["ptr_top_is_gold"] = f"{s['ptr_top_is_gold']:.4f}"
            for k in ["rank_le_0","rank_le_1","rank_le_3","rank_le_5","rank_le_10"]:
                row[k] = f"{s[k]/n:.4f}"
            row["mean_ptr_confidence"] = f"{s['mean_ptr_confidence']:.4f}"
            row["mean_ptr_margin"]     = f"{s['mean_ptr_margin']:.4f}"
            w.writerow(row)
    print(f"[csv] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Markdown examples
# ─────────────────────────────────────────────────────────────────────────────

def _cand_table(cands):
    hdr = ("| rank | token | base_logit | ptr_score | exact_n "
           "| nearest_d | final_score | gold? | base? | ptr_top? | sel? |")
    sep = ("|------|-------|------------|-----------|---------|"
           "-----------|-------------|-------|-------|----------|------|")
    rows = [hdr, sep]
    for j, c in enumerate(cands):
        nd = c["nearest_dist"]
        nd_s = f"{nd:.1f}" if nd < 9000 else "∞"
        rows.append(
            f"| {j} | `{_decode_tok(c['tok_id']):<10}` "
            f"| {c['base_logit']:8.3f} | {c['ptr_score']:.4f} "
            f"| {c['exact_count']} | {nd_s} | {c['final_score']:8.3f} "
            f"| {'✓' if c['is_gold'] else ''} "
            f"| {'✓' if c['is_base_top1'] else ''} "
            f"| {'✓' if c['is_ptr_top'] else ''} "
            f"| {'✓' if c['is_selected'] else ''} |")
    return "\n".join(rows)


def write_examples_md(ex_buckets, kind_label_map, out_path):
    lines = [f"# Pointer Help Slice Analysis — Examples\n", "---\n"]
    for kind, label in kind_label_map.items():
        exs = ex_buckets.get(kind, [])
        lines.append(f"## {label}  ({len(exs)} examples)\n")
        for idx, ex in enumerate(exs):
            ctx_str = _decode_ids(ex["ctx_ids"])
            gnd = _decode_tok(ex["gold_id"])
            b   = _decode_tok(ex["base_top1_id"])
            pt  = _decode_tok(ex["ptr_top_tok"])
            sl  = _decode_tok(ex["sel_lam_tok"])
            lines.append(f"### Example {idx+1}  (row {ex['row_idx']})\n")
            lines.append(
                f"**Gold:** `{gnd}`  **Base:** `{b}` ({ex['base_top1_lgt']:.3f})  "
                f"**PtrTop:** `{pt}` (score={ex['ptr_top_score']:.4f})  "
                f"**Selected(λ):** `{sl}`\n")
            lines.append(
                f"gold_in_ctx={ex['gold_in_ctx']}  "
                f"gold_nearest_d={ex['gold_nearest_d']:.1f}  "
                f"gold_ptr_rank={ex['gold_ptr_rank']}  "
                f"ptr_conf={ex['ptr_confidence']:.3f}  "
                f"ptr_margin={ex['ptr_margin']:.4f}\n")
            lines.append("\n**Context tail:**\n```\n" + ctx_str[-400:] + "\n```\n")
            lines.append("\n**Candidates:**\n" + _cand_table(ex["candidates"]) + "\n\n---\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[md] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _f(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)

def _m(metrics, sn, pname, key):
    return metrics.get(sn, {}).get(pname, MetricAcc()).metrics().get(key, float("nan"))

def _rs(rank_stats, sn, key):
    return rank_stats.get(sn, RankAcc(1)).stats().get(key, float("nan"))

def write_report(metrics, rank_stats, slices_dict, policies, lam_grid, out_path, args, N_total):
    pnames = [p for p, _ in policies]
    lam_pnames = [f"lambda_{l:.2f}" for l in lam_grid]

    # Best lambda by net_correction on 'all'
    best_lam_name = max(lam_pnames,
                        key=lambda p: _m(metrics, "all", p, "net_correction"))
    best_lam_val  = float(best_lam_name.split("_")[1])

    lines = [
        "# Pointer Help Slice Analysis V1 — Report\n",
        f"**val_dir:** `{args.val_dir}`  "
        f"**memory_len:** {args.memory_len}  **tau:** {args.recency_tau}\n",
        f"**N total:** {N_total:,}  **pool_size:** {args.candidate_pool_size}\n",
        "",
    ]

    # ── Slice sizes ──────────────────────────────────────────────────────────
    lines.append("## 0. Slice Sizes\n")
    lines.append("| slice | n | fraction |")
    lines.append("|-------|---|----------|")
    for sn in SLICE_NAMES:
        n = _m(metrics, sn, pnames[0], "n") if pnames else 0
        lines.append(f"| {sn} | {int(n) if isinstance(n,float) else n} "
                     f"| {int(n)/max(N_total,1):.4f} |")
    lines.append("")

    # ── Lambda sweep on key slices ────────────────────────────────────────────
    lines.append("## 1. Lambda Sweep on Key Slices\n")
    for sn in ["all", "bucketA_confuser", "gold_exact_present",
               "pointer_should_help_strong", "pointer_should_not_help"]:
        lines.append(f"### {sn}\n")
        lines.append("| λ | ctg | caw | net | sip | acc_gain |")
        lines.append("|---|-----|-----|-----|-----|----------|")
        for p in lam_pnames:
            lv = p.split("_")[1]
            lines.append(
                f"| {lv} "
                f"| {_f(_m(metrics,sn,p,'changed_to_gold'))} "
                f"| {_f(_m(metrics,sn,p,'changed_away'))} "
                f"| {_f(_m(metrics,sn,p,'net_correction'))} "
                f"| {_f(_m(metrics,sn,p,'sip'))} "
                f"| {_f(_m(metrics,sn,p,'acc_gain'))} |")
        lines.append("")

    # ── Rank stats table ─────────────────────────────────────────────────────
    lines.append("## 2. Gold Pointer Rank per Slice\n")
    lines.append("| slice | ptr_is_gold | rank≤0 | rank≤1 | rank≤3 | rank≤5 | rank≤10 | conf | margin |")
    lines.append("|-------|-------------|--------|--------|--------|--------|---------|------|--------|")
    for sn in ["all","bucketA_confuser","gold_exact_present","gold_recent",
               "gold_repeated","pointer_should_help_strong","pointer_should_help_weak"]:
        rs = rank_stats[sn].stats()
        n  = max(rs["n"], 1)
        lines.append(
            f"| {sn} "
            f"| {rs['ptr_top_is_gold']/1:.4f} "
            f"| {rs['rank_le_0']/n:.4f} "
            f"| {rs['rank_le_1']/n:.4f} "
            f"| {rs['rank_le_3']/n:.4f} "
            f"| {rs['rank_le_5']/n:.4f} "
            f"| {rs['rank_le_10']/n:.4f} "
            f"| {rs['mean_ptr_confidence']:.4f} "
            f"| {rs['mean_ptr_margin']:.4f} |")
    lines.append("")

    # ── Diagnostic Q&A ────────────────────────────────────────────────────────
    lines.append("## 3. Diagnostic Questions\n")

    n_all    = int(_m(metrics, "all", "base_candidate", "n") or 1)
    n_ba     = int(_m(metrics, "bucketA_confuser", "base_candidate", "n") or 0)
    n_gep    = int(_m(metrics, "gold_exact_present", "base_candidate", "n") or 0)
    n_psh    = int(_m(metrics, "pointer_should_help_strong", "base_candidate", "n") or 0)
    n_psnh   = int(_m(metrics, "pointer_should_not_help", "base_candidate", "n") or 0)

    lines.append("### Q1. What fraction of Bucket A is copy-supported?")
    frac_ba_gep = n_gep / max(n_ba, 1)  # rough: both measured on all
    lines.append(f"- total N: {n_all:,}  bucketA n: {n_ba:,}  gold_exact_present n: {n_gep:,}")
    lines.append(f"- gold_exact_present fraction of all: {n_gep/n_all:.4f}")
    lines.append(f"- pointer_should_help_strong n: {n_psh:,}  "
                 f"(base_wrong + gold_in_pool + gold_in_ctx + gold_ptr_rank≤2)")
    ans1 = "**substantial**" if n_gep / max(n_all, 1) > 0.15 else "**limited** (<15%)"
    lines.append(f"- Copy-support coverage is {ans1}\n")

    lines.append("### Q2. When gold is in context, how often is it pointer top1/3/5?")
    gep_rs = rank_stats["gold_exact_present"].stats()
    n_gep_ = max(gep_rs["n"], 1)
    lines.append(f"- gold_exact_present n={gep_rs['n']}")
    lines.append(f"  ptr_top_is_gold:  {gep_rs['ptr_top_is_gold']/1:.4f}")
    lines.append(f"  rank≤1: {gep_rs['rank_le_1']/n_gep_:.4f}  "
                 f"rank≤3: {gep_rs['rank_le_3']/n_gep_:.4f}  "
                 f"rank≤5: {gep_rs['rank_le_5']/n_gep_:.4f}")
    ptr_ranks_gold = gep_rs["ptr_top_is_gold"]
    ans2 = ("**YES — frequent**" if ptr_ranks_gold > 0.5
            else ("**sometimes**" if ptr_ranks_gold > 0.2 else "**rarely**"))
    lines.append(f"- {ans2}\n")

    lines.append("### Q3. Does pointer improve sip in gold_exact_present?")
    sip_base = _m(metrics, "gold_exact_present", "base_candidate", "sip")
    sip_best = _m(metrics, "gold_exact_present", best_lam_name, "sip")
    lines.append(f"- sip base_candidate: {_f(sip_base)}  sip {best_lam_name}: {_f(sip_best)}")
    if not (math.isnan(sip_base) or math.isnan(sip_best)):
        d = sip_best - sip_base
        ans3 = "**YES**" if d > 0.01 else ("**marginal**" if d > 0.002 else "**NO**")
        lines.append(f"- Δsip = {d:+.4f}: {ans3}")
    lines.append("")

    lines.append("### Q4. Does pointer hurt when gold is not present?")
    net_psnh = _m(metrics, "pointer_should_not_help", best_lam_name, "net_correction")
    caw_psnh = _m(metrics, "pointer_should_not_help", best_lam_name, "changed_away")
    caw_psnh_base = _m(metrics, "pointer_should_not_help", "base_candidate", "changed_away")
    lines.append(f"- pointer_should_not_help  net={_f(net_psnh)}  "
                 f"caw={_f(caw_psnh)} (base: {_f(caw_psnh_base)})")
    if not math.isnan(net_psnh):
        ans4 = ("**YES — net negative**" if net_psnh < -0.001
                else ("**neutral**" if abs(net_psnh) < 0.001 else "**surprisingly positive**"))
        lines.append(f"- {ans4}\n")

    lines.append("### Q5. Does pointer mostly reduce changed_away or increase changed_to_gold?")
    caw_all_base = _m(metrics, "all", "base_candidate", "changed_away")
    caw_all_best = _m(metrics, "all", best_lam_name, "changed_away")
    ctg_all_base = _m(metrics, "all", "base_candidate", "changed_to_gold")
    ctg_all_best = _m(metrics, "all", best_lam_name, "changed_to_gold")
    lines.append(f"- ctg: {_f(ctg_all_base)} → {_f(ctg_all_best)} "
                 f"(Δ={ctg_all_best-ctg_all_base:+.4f})")
    lines.append(f"- caw: {_f(caw_all_base)} → {_f(caw_all_best)} "
                 f"(Δ={caw_all_best-caw_all_base:+.4f})")
    if not (math.isnan(ctg_all_best) or math.isnan(caw_all_best)):
        dctg = ctg_all_best - ctg_all_base
        dcaw = caw_all_best - caw_all_base
        if abs(dctg) > abs(dcaw):
            lines.append("- Primary effect: **changing more predictions to gold**")
        else:
            lines.append("- Primary effect: **reducing changed_away (safer edits)**")
    lines.append("")

    lines.append("### Q6. Is there a usable confidence/margin threshold for pointer?")
    # Find best gated policy by net_correction on 'all'
    gated_policies  = [(p, pf) for p, pf in [("base_candidate", None)] if "gated_c" in p]
    gated_names     = [p for p, _ in policies if "gated_c" in p]
    if gated_names:
        best_gated = max(gated_names, key=lambda p: _m(metrics, "all", p, "net_correction"))
        bg_net = _m(metrics, "all", best_gated, "net_correction")
        bg_caw = _m(metrics, "all", best_gated, "changed_away")
        bg_ctg = _m(metrics, "all", best_gated, "changed_to_gold")
        bg_ar  = _m(metrics, "all", best_gated, "apply_rate")
        lines.append(f"- Best gated policy: `{best_gated}`")
        lines.append(f"  net={_f(bg_net)}  ctg={_f(bg_ctg)}  caw={_f(bg_caw)}  apply_rate={_f(bg_ar)}")
        best_net = _m(metrics, "all", best_lam_name, "net_correction")
        if not (math.isnan(bg_net) or math.isnan(best_net)):
            ans6 = ("**YES — gating helps**" if bg_net > best_net + 0.0005
                    else "**NO — gating does not significantly improve over best lambda**")
            lines.append(f"- {ans6}")
    else:
        lines.append("- No gated policies evaluated.")
    lines.append("")

    lines.append("### Q7. Oracle gain if pointer used only on pointer_should_help_strong?")
    oracle_net = _m(metrics, "all", "oracle_DIAGNOSTIC", "net_correction")
    oracle_ctg = _m(metrics, "all", "oracle_DIAGNOSTIC", "changed_to_gold")
    oracle_caw = _m(metrics, "all", "oracle_DIAGNOSTIC", "changed_away")
    best_real_net = max((_m(metrics,"all",p,"net_correction") for p in lam_pnames),
                        default=float("nan"))
    lines.append(f"- Oracle (pointer on psh_strong only): "
                 f"net={_f(oracle_net)}  ctg={_f(oracle_ctg)}  caw={_f(oracle_caw)}")
    lines.append(f"- Best real policy net: {_f(best_real_net)}")
    if not (math.isnan(oracle_net) or math.isnan(best_real_net)):
        oracle_gap = oracle_net - best_real_net
        lines.append(f"- Oracle gap above best real: {oracle_gap:+.4f}")
        if oracle_gap > 0.003:
            lines.append("- ⚠️  **ORACLE_DIAGNOSTIC** only — signal exists but routing/gating is the hard part.")
        else:
            lines.append("- Oracle gain is small — pointer signal may be weak even with perfect routing.")
    lines.append("")

    lines.append("### Q8. Should pointer become a memory expert in the final mixer?\n")
    # Synthesis
    psh_ptr_gold = _rs(rank_stats, "pointer_should_help_strong", "ptr_top_is_gold")
    n_psh_rs     = rank_stats["pointer_should_help_strong"].stats()["n"]
    frac_psh     = n_psh_rs / max(n_all, 1)
    psh_net_best = max((_m(metrics,"pointer_should_help_strong",p,"net_correction")
                        for p in lam_pnames), default=float("nan"))

    reasons_for  = []
    reasons_against = []

    if not math.isnan(psh_ptr_gold) and psh_ptr_gold > 0.4:
        reasons_for.append(f"pointer top-1 is gold in {psh_ptr_gold:.1%} of "
                           f"pointer_should_help_strong rows")
    if not math.isnan(psh_net_best) and psh_net_best > 0.005:
        reasons_for.append(f"net_correction={psh_net_best:.4f} on pointer_should_help_strong "
                           f"at best lambda")
    if frac_psh > 0.05:
        reasons_for.append(f"pointer_should_help_strong covers {frac_psh:.1%} of all rows "
                           f"— non-trivial scope")
    if not math.isnan(oracle_net) and oracle_net > 0.01:
        reasons_for.append(f"oracle net_correction={oracle_net:.4f} — ceiling is meaningful")

    if n_gep / max(n_all, 1) < 0.05:
        reasons_against.append(f"only {n_gep/n_all:.1%} of all rows have gold in context — "
                                "sparse copy signal")
    if not math.isnan(best_real_net) and best_real_net < 0.001:
        reasons_against.append("best real-policy net_correction is near zero — "
                                "cannot convert oracle signal to real gain yet")
    psnh_net = _m(metrics, "pointer_should_not_help", best_lam_name, "net_correction")
    if not math.isnan(psnh_net) and psnh_net < -0.002:
        reasons_against.append("pointer hurts on non-copy rows — requires a reliable gate "
                                "before deployment")

    if reasons_for:
        lines.append("**Evidence FOR pointer as a memory expert:**")
        for r in reasons_for: lines.append(f"- {r}")
    if reasons_against:
        lines.append("\n**Evidence AGAINST / cautions:**")
        for r in reasons_against: lines.append(f"- {r}")

    # Final verdict
    strong_for = len(reasons_for) >= 2 and (not math.isnan(psh_net_best)) and psh_net_best > 0.005
    if strong_for and not math.isnan(psnh_net) and psnh_net < -0.002:
        verdict = ("**Conditional YES** — pointer is a useful specialized evidence path for "
                   "copy-supported cases, but requires a confident gate before applying. "
                   "Implement as a gated memory expert: activate only when pointer_confidence "
                   "is high AND exact_present is True for the candidate.")
    elif strong_for:
        verdict = ("**YES** — pointer is a useful specialized evidence path. "
                   "Implement as one memory expert in the final mixer. "
                   "Treat pointer as a conditional path, not a general Bucket A solver.")
    elif len(reasons_for) > 0:
        verdict = ("**Marginal** — pointer shows some promise on copy-supported cases "
                   "but signal is weak or noisy. Use pointer_score as one input feature "
                   "to the gate/selector rather than a dedicated expert.")
    else:
        verdict = ("**NO** — exact pointer does not expose sufficient signal even where "
                   "intended. Consider: (a) soft/embedding-based copy scorer; "
                   "(b) n-gram overlap; "
                   "(c) backbone hidden-state memory instead.")
    lines.append(f"\n**Verdict:** {verdict}\n")

    # ── Framing note ─────────────────────────────────────────────────────────
    lines.append("## 4. Scope of Pointer Memory\n")
    lines.append("Pointer memory is expected to help on:\n")
    lines.append("- repeated tokens / entities / names / numbers / format patterns\n")
    lines.append("- local copy scenarios (copy of recent context)\n")
    lines.append("\nPointer memory is NOT expected to solve:\n")
    lines.append("- unseen entity continuation\n"
                 "- factual value conversion\n"
                 "- nonlocal exact passage recall\n"
                 "- syntax-only punctuation choices\n"
                 "- multi-token path coherence\n")
    lines.append("\n*Pointer memory is a **specialized evidence path for copy-supported cases**, "
                 "not a general Bucket A solver.*\n")
    lines.append("---\n")
    lines.append("*Generated by analyze_pointer_help_slices_v1.py — training-free diagnostic.*\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Pointer help slice diagnostic")
    p.add_argument("--val_dir",           required=True)
    p.add_argument("--phase1_dir",        default=None,
                   help="Phase 1 output dir (read config.json if present)")
    p.add_argument("--token_to_region",   required=True)
    p.add_argument("--super_map",         default=None)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--top_k",             type=int,   default=256)
    p.add_argument("--candidate_pool_size", type=int, default=32)
    p.add_argument("--memory_len",        type=int,   default=128)
    p.add_argument("--recency_tau",       type=float, default=32.0)
    p.add_argument("--lambda_grid",       default="0.0,0.25,0.5,1.0,2.0,4.0")
    p.add_argument("--pointer_conf_thresholds",   default="0.1,0.2,0.3,0.4,0.5,0.7")
    p.add_argument("--pointer_margin_thresholds", default="0.0,0.05,0.1,0.2,0.5")
    p.add_argument("--max_examples",      type=int,   default=50)
    p.add_argument("--seed",              type=int,   default=42)
    return p, p.parse_args()


def main():
    p, args = _parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Try to read phase1 config
    if args.phase1_dir:
        cfg_p = os.path.join(args.phase1_dir, "config.json")
        if os.path.isfile(cfg_p):
            with open(cfg_p) as f:
                p1cfg = json.load(f)
            for k in ("memory_len", "recency_tau", "top_k", "candidate_pool_size"):
                if k in p1cfg and getattr(args, k) == p.get_default(k):
                    setattr(args, k, p1cfg[k])
            print(f"[config] Loaded phase1 config from {cfg_p}")

    cfg = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[data] Loading val shards...")
    data = load_shards(args.val_dir, args.top_k)
    N_total = len(data["gold"])

    print(f"\n[maps] Loading region maps...")
    tok_arr, reg_arr, unk_r, unk_s, sr = load_maps(args.token_to_region, args.super_map)
    print(f"  sr_enabled={sr}")

    (metrics, rank_stats, slices_dict, policy_results,
     ex_buckets, policies, lam_grid, pstats, gold_in_ctx,
     ptr_top_is_gold) = run_eval(data, tok_arr, reg_arr, unk_r, unk_s, sr, args)

    print("\n[write] Writing outputs...")

    write_slice_summary_csv(metrics, policies,
        os.path.join(args.output_dir, "pointer_help_slice_summary.csv"))
    write_policy_grid_csv(metrics, policies, SLICE_NAMES,
        os.path.join(args.output_dir, "pointer_policy_grid.csv"))
    write_rank_stats_csv(rank_stats,
        os.path.join(args.output_dir, "pointer_rank_stats.csv"))

    kind_labels = {
        "pointer_helps":     "Pointer HELPS (base wrong, gold in ctx+pool, ptr selects gold)",
        "pointer_hurts":     "Pointer HURTS (base correct, ptr selects non-gold from context)",
        "should_help_fails": "Should Help But Fails (pointer_should_help_strong, ptr wrong)",
        "pointer_danger":    "Pointer DANGER (base==gold, high-confidence wrong ptr selection)",
    }
    # Write separate md files for each kind
    for kind in EXAMPLE_KINDS:
        exs = ex_buckets.get(kind, [])
        if exs:
            write_examples_md({kind: exs}, {kind: kind_labels[kind]},
                os.path.join(args.output_dir, f"examples_{kind}.md"))
    # Also combined
    write_examples_md(ex_buckets, kind_labels,
        os.path.join(args.output_dir, "pointer_help_examples_all.md"))

    write_report(metrics, rank_stats, slices_dict, policies, lam_grid,
        os.path.join(args.output_dir, "pointer_help_report.md"),
        args, N_total)

    print(f"\n[done] Outputs in: {args.output_dir}")

    # Console summary
    lam_pnames = [f"lambda_{l:.2f}" for l in lam_grid]
    print("\n=== slice × best-lambda summary ===")
    focus = ["all", "bucketA_confuser", "gold_exact_present",
             "pointer_should_help_strong", "pointer_should_not_help"]
    print(f"{'slice':<32} {'λ':>4}  {'ctg':>7}  {'caw':>7}  {'net':>7}  {'sip':>7}")
    print("-" * 70)
    for sn in focus:
        best_p = max(lam_pnames,
                     key=lambda p: metrics[sn][p].metrics().get("net_correction", -1e9))
        lv = best_p.split("_")[1]
        m  = metrics[sn][best_p].metrics()
        print(f"{sn:<32} {lv:>4}  "
              f"{m.get('changed_to_gold',float('nan')):>7.4f}  "
              f"{m.get('changed_away',float('nan')):>7.4f}  "
              f"{m.get('net_correction',float('nan')):>7.4f}  "
              f"{m.get('sip',float('nan')):>7.4f}")
    print()


if __name__ == "__main__":
    main()
