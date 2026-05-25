#!/usr/bin/env python3
"""
eval_ngram_phrase_recall_v1.py — Phase 1B: Evaluate n-gram phrase recall.

For each val row, looks up the context suffix in the train datastore and
evaluates whether phrase memory can improve candidate selection.

NO training. NO model changes. Gold used only after lookup, for metrics only.

Usage:
  python scripts/eval_ngram_phrase_recall_v1.py \
    --val_dir runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val \
    --datastore_dir runs/phrase_memory_v1/ngram_phrase_recall/datastore \
    --token_to_region runs/region_maps_128/token_to_region.json \
    --super_map runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \
    --output_dir runs/phrase_memory_v1/ngram_phrase_recall/eval \
    --candidate_pool_size 32 \
    --ngram_lengths 4,8,16,32 \
    --lambda_grid 0.0,0.25,0.5,1.0,2.0,4.0 \
    --min_count_grid 1,2,4,8 \
    --prob_thresholds 0.1,0.2,0.3,0.5,0.7 \
    --margin_thresholds 0.0,0.05,0.1,0.2 \
    --max_examples 50 \
    --seed 42
"""

import argparse
import csv
import glob
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np
import torch

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer (optional, graceful fallback)
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
# Shard loading
# ─────────────────────────────────────────────────────────────────────────────

_TOPK_ALIASES  = ["base_topk_ids",    "base_topk",     "topk_ids"]
_LGT_ALIASES   = ["base_topk_logits", "base_topk_lgt", "topk_lgt", "topk_logits"]
_GOLD_ALIASES  = ["gold_token",       "gold",          "labels"]
_IDS_ALIASES   = ["input_ids"]
_ROWID_ALIASES = ["row_id",           "row_ids"]

def _get(shard, aliases, required=True):
    for a in aliases:
        if a in shard:
            return shard[a]
    if required:
        raise KeyError(f"Missing field. Tried: {aliases}. Have: {list(shard.keys())}")
    return None

def load_val_shards(val_dir, top_k, max_rows=None):
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {val_dir}")
    print(f"[shards] {len(paths)} val shard(s)")
    bufs = defaultdict(list)
    total = 0
    first = True
    for sp in paths:
        if max_rows is not None and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print("[preflight] first val shard keys:")
            for k, v in sh.items():
                print(f"  {k}: shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',type(v).__name__)}")
            first = False
        topk = _get(sh, _TOPK_ALIASES).long()
        lgt  = _get(sh, _LGT_ALIASES).float()
        gold = _get(sh, _GOLD_ALIASES).long()
        ids  = _get(sh, _IDS_ALIASES).long()
        row_ids = _get(sh, _ROWID_ALIASES, required=False)
        B, K = topk.shape
        if row_ids is None:
            row_ids = torch.arange(total, total + B)
        if K < top_k:
            topk = torch.cat([topk, torch.zeros(B, top_k-K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, top_k-K), float("nan"))], 1)
        elif K > top_k:
            topk, lgt = topk[:, :top_k], lgt[:, :top_k]
        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, ids, row_ids = (x[:keep] for x in (topk, lgt, gold, ids, row_ids))
            B = keep
        bufs["topk"].append(topk.numpy())
        bufs["lgt"].append(lgt.numpy())
        bufs["gold"].append(gold.numpy())
        bufs["ids"].append(ids.numpy())
        bufs["row_ids"].append(row_ids.numpy())
        total += B
    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    print(f"[shards] {total:,} val rows  K={top_k}  seq={data['ids'].shape[1]}")
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Map loading
# ─────────────────────────────────────────────────────────────────────────────

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
# Datastore loading
# ─────────────────────────────────────────────────────────────────────────────

def load_datastores(datastore_dir, ngram_lengths):
    stores = {}
    for n in ngram_lengths:
        p = os.path.join(datastore_dir, f"ngram_{n}.pkl")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Datastore not found: {p}")
        with open(p, "rb") as f:
            stores[n] = pickle.load(f)
        nkeys = len(stores[n]["counts"])
        print(f"  [OK] ngram_{n}.pkl  — {nkeys:,} keys")
    return stores

# ─────────────────────────────────────────────────────────────────────────────
# Per-row phrase lookup
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-9

def phrase_lookup(ctx_np, cand_ids_np, stores, ngram_lengths):
    """
    ctx_np:      [seq] int array (full context for one row)
    cand_ids_np: [P]   int array (candidate pool)

    Returns dict keyed by n with fields:
      key_hit, total_count, phrase_counts[P], phrase_prob[P],
      phrase_top_idx, phrase_top_tok, phrase_top_prob,
      phrase_entropy, phrase_margin
    """
    seq = len(ctx_np)
    P   = len(cand_ids_np)
    res = {}
    for n in ngram_lengths:
        if seq < n:
            res[n] = _empty_phrase(P)
            continue
        key = tuple(ctx_np[-n:].tolist())
        counts_map = stores[n]["counts"]
        if key not in counts_map:
            res[n] = _empty_phrase(P)
            continue
        cdict = counts_map[key]
        total = sum(cdict.values())
        # Per-candidate counts
        pc = np.array([cdict.get(int(c), 0) for c in cand_ids_np], dtype=np.float32)
        pp = pc / max(total, 1)
        # Top
        top_idx  = int(pc.argmax())
        top_tok  = int(cand_ids_np[top_idx])
        top_prob = float(pp[top_idx])
        # Entropy over candidates that have counts
        pp_clip  = np.clip(pp, _EPS, None)
        pp_norm  = pp_clip / pp_clip.sum()
        entropy  = float(-(pp_norm * np.log(pp_norm)).sum())
        # Margin top1 - top2
        if P >= 2:
            sorted_pp = np.sort(pp)[::-1]
            margin = float(sorted_pp[0] - sorted_pp[1])
        else:
            margin = 0.0
        res[n] = {
            "key_hit":        True,
            "total_count":    int(total),
            "phrase_counts":  pc,
            "phrase_prob":    pp,
            "phrase_top_idx": top_idx,
            "phrase_top_tok": top_tok,
            "phrase_top_prob":top_prob,
            "phrase_entropy": entropy,
            "phrase_margin":  margin,
        }
    return res

def _empty_phrase(P):
    return {
        "key_hit":        False,
        "total_count":    0,
        "phrase_counts":  np.zeros(P, dtype=np.float32),
        "phrase_prob":    np.zeros(P, dtype=np.float32),
        "phrase_top_idx": 0,
        "phrase_top_tok": -1,
        "phrase_top_prob":0.0,
        "phrase_entropy": 0.0,
        "phrase_margin":  0.0,
    }

def gold_phrase_stats(gold_id, cand_ids, phrase_res, ngram_lengths):
    """Returns per-n gold rank/prob/count in candidate pool."""
    out = {}
    for n in ngram_lengths:
        pr  = phrase_res[n]
        pp  = pr["phrase_prob"]      # [P]
        pc  = pr["phrase_counts"]    # [P]
        P   = len(cand_ids)
        # Find gold in pool
        match = np.where(cand_ids == gold_id)[0]
        if len(match) == 0:
            out[n] = {"in_pool": False, "gold_phrase_count": 0,
                      "gold_phrase_prob": 0.0, "gold_phrase_rank": P}
            continue
        j = match[0]
        gold_pc   = float(pc[j])
        gold_pp   = float(pp[j])
        # rank = number of candidates with higher phrase_prob
        gold_rank = int((pp > pp[j]).sum())
        out[n] = {"in_pool": True, "gold_phrase_count": gold_pc,
                  "gold_phrase_prob": gold_pp, "gold_phrase_rank": gold_rank}
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Surgical edit (consistent with training: boost sel +md_half, penalise base_top1 -md_half)
# ─────────────────────────────────────────────────────────────────────────────

def surgical_edit(topk_ids_row, topk_lgt_row, sel_tok, md_half=0.5):
    """
    topk_ids_row: [K], topk_lgt_row: [K] for one example.
    Returns final_top1_id (int).
    """
    ref = np.where(np.isfinite(topk_lgt_row), topk_lgt_row, -1e9).copy()
    ref[0] -= md_half   # penalise base_top1 at index 0
    pos = np.where(topk_ids_row == sel_tok)[0]
    if len(pos):
        ref[pos[0]] += md_half
    return int(topk_ids_row[ref.argmax()])

# ─────────────────────────────────────────────────────────────────────────────
# Metric accumulator
# ─────────────────────────────────────────────────────────────────────────────

class MetricAcc:
    __slots__ = ["n", "apply", "noop", "ctg", "caw", "base_ok", "final_ok",
                 "sip_n", "sip_match", "gip_n",
                 "key_hit_n", "total_count_sum"]
    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, 0)

    def update(self, gold, base_top1, final_top1, apply_mask, sel_tok,
               gold_in_pool, key_hit, total_count, subset):
        g   = gold[subset]; b = base_top1[subset]; ft = final_top1[subset]
        am  = apply_mask[subset]; gip = gold_in_pool[subset]
        st  = sel_tok[subset]; kh = key_hit[subset]; tc = total_count[subset]
        self.n           += len(g)
        self.apply       += am.sum()
        self.noop        += (~am).sum()
        self.ctg         += (am & (ft == g)).sum()
        self.caw         += (am & (b == g) & (ft != g)).sum()
        self.base_ok     += (b == g).sum()
        self.final_ok    += (ft == g).sum()
        self.gip_n       += gip.sum()
        self.sip_n       += gip.sum()
        self.sip_match   += (gip & (st == g)).sum()
        self.key_hit_n   += kh.sum()
        self.total_count_sum += tc.sum()

    def metrics(self):
        _E = 1e-9
        n = max(self.n, 1)
        return {
            "n":                    self.n,
            "fraction_of_val":      float("nan"),   # set outside
            "key_hit_rate":         self.key_hit_n / n,
            "gold_in_pool_rate":    self.gip_n / n,
            "apply_rate":           self.apply / n,
            "noop_rate":            self.noop / n,
            "base_acc":             self.base_ok / n,
            "policy_acc":           self.final_ok / n,
            "acc_gain":             (self.final_ok - self.base_ok) / n,
            "selected_gold_given_in_pool": self.sip_match / max(self.sip_n, 1),
            "changed_to_gold":      self.ctg / n,
            "changed_away":         self.caw / n,
            "net_correction":       (self.ctg - self.caw) / n,
            "benefit_damage_ratio": self.ctg / max(self.caw, _E),
            "applied_precision_ctg": self.ctg / max(self.apply, 1),
            "mean_total_count":     self.total_count_sum / n,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Rank accumulator
# ─────────────────────────────────────────────────────────────────────────────

class RankAcc:
    def __init__(self, P):
        self.P = P
        self.hist = np.zeros(P + 1, dtype=np.int64)
        self.sum_prob  = 0.0
        self.sum_top_prob = 0.0
        self.sum_entropy  = 0.0
        self.sum_count    = 0.0
        self.n = 0; self.n_in_pool = 0; self.n_hit = 0

    def update(self, gold_phrase_rank, gold_in_pool, gold_phrase_prob,
               phrase_top_prob, phrase_entropy, total_count, key_hit, subset):
        gpr = gold_phrase_rank[subset]
        gip = gold_in_pool[subset]
        gpp = gold_phrase_prob[subset]
        kh  = key_hit[subset]
        m   = subset.sum()
        for r in gpr: self.hist[min(r, self.P)] += 1
        self.n_in_pool     += int(gip.sum())
        self.n_hit         += int(kh.sum())
        self.sum_prob      += float(gpp.sum())
        self.sum_top_prob  += float(phrase_top_prob[subset].sum())
        self.sum_entropy   += float(phrase_entropy[subset].sum())
        self.sum_count     += float(total_count[subset].sum())
        self.n             += m

    def stats(self):
        n = max(self.n, 1)
        return {
            "n": self.n,
            "key_hit_rate":          self.n_hit / n,
            "mean_gold_phrase_prob": self.sum_prob / n,
            "mean_phrase_top_prob":  self.sum_top_prob / n,
            "mean_phrase_entropy":   self.sum_entropy / n,
            "total_count_mean":      self.sum_count / n,
            "gold_phrase_rank_le_1": int(self.hist[:2].sum()),
            "gold_phrase_rank_le_3": int(self.hist[:4].sum()),
            "gold_phrase_rank_le_5": int(self.hist[:6].sum()),
            "gold_phrase_rank_le_10":int(self.hist[:11].sum()),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Policy builder
# ─────────────────────────────────────────────────────────────────────────────

def make_policies(ngram_lengths, lambda_grid, min_count_grid, prob_thresholds, margin_thresholds):
    """
    Returns list of (policy_name, fn).
    fn(cand_ids [P], cand_lgts [P], phrase_res, gold_in_pool: bool, n_primary: int)
      -> (sel_tok: int, apply: bool)
    """
    policies = []

    # 1. base_candidate
    def base_candidate(cand_ids, cand_lgts, phrase_res, gold_in_pool, n_primary):
        return int(cand_ids[cand_lgts.argmax()]), True
    policies.append(("base_candidate", base_candidate))

    # 2. phrase_only per n
    for n in ngram_lengths:
        def make_phrase_only(n_):
            def phrase_only(cand_ids, cand_lgts, phrase_res, gold_in_pool, n_primary):
                pr = phrase_res[n_]
                if not pr["key_hit"] or pr["phrase_top_tok"] < 0:
                    return int(cand_ids[cand_lgts.argmax()]), False
                return int(pr["phrase_top_tok"]), True
            return phrase_only
        policies.append((f"phrase_only_n{n}", make_phrase_only(n)))

    # 3. base_plus_phrase lambda per n and lambda
    for n in ngram_lengths:
        for lam in lambda_grid:
            def make_lam(n_, l_):
                def policy(cand_ids, cand_lgts, phrase_res, gold_in_pool, n_primary):
                    pr = phrase_res[n_]
                    pp = pr["phrase_prob"]      # [P]
                    scores = cand_lgts + l_ * np.log(pp + _EPS)
                    sel_idx = int(scores.argmax())
                    return int(cand_ids[sel_idx]), True
                return policy
            policies.append((f"phrase_n{n}_lam{lam:.2f}", make_lam(n, lam)))

    # 4. confidence_gated per n
    for n in ngram_lengths:
        for mc in min_count_grid:
            for pt in prob_thresholds:
                for mt in margin_thresholds:
                    def make_gated(n_, mc_, pt_, mt_):
                        def policy(cand_ids, cand_lgts, phrase_res, gold_in_pool, n_primary):
                            pr = phrase_res[n_]
                            if (pr["key_hit"] and
                                    pr["total_count"] >= mc_ and
                                    pr["phrase_top_prob"] >= pt_ and
                                    pr["phrase_margin"] >= mt_):
                                return int(pr["phrase_top_tok"]), True
                            return int(cand_ids[cand_lgts.argmax()]), False
                        return policy
                    policies.append((f"gated_n{n}_mc{mc}_p{pt}_m{mt}", make_gated(n, mc, pt, mt)))

    # 5. oracle_DIAGNOSTIC per n
    for n in ngram_lengths:
        def make_oracle(n_):
            def policy(cand_ids, cand_lgts, phrase_res, gold_in_pool, n_primary):
                # Use phrase only if phrase_top_tok == gold (oracle: gold is known after scoring)
                # We cannot pass gold in here — caller sets oracle flag separately
                # This function always applies phrase top; the caller post-filters by is_oracle_correct
                pr = phrase_res[n_]
                if not pr["key_hit"]:
                    return int(cand_ids[cand_lgts.argmax()]), False
                return int(pr["phrase_top_tok"]), True
            return policy
        policies.append((f"oracle_n{n}_DIAGNOSTIC", make_oracle(n)))

    return policies

# ─────────────────────────────────────────────────────────────────────────────
# Slice definitions
# ─────────────────────────────────────────────────────────────────────────────

SLICE_NAMES_BASE = [
    "all",
    "bucketA_confuser",
    "high_base_margin",
    "low_base_margin",
    "same_region_confuser",
    "same_superregion_confuser",
]

def build_fixed_slices(gold, base_top1, cand_ids, gold_in_pool, cand_lgts,
                       tok_arr, reg_arr, unk_r, unk_s, sr):
    N  = len(gold)
    V  = tok_arr.shape[0]
    R  = reg_arr.shape[0]

    base_wrong   = base_top1 != gold
    # base margin: top1 logit - top2 logit within pool
    base_lgts_top1 = cand_lgts[:, 0]
    if cand_lgts.shape[1] >= 2:
        base_lgts_top2 = np.sort(cand_lgts, axis=1)[:, -2]
        base_margin    = cand_lgts.max(axis=1) - base_lgts_top2
    else:
        base_margin = np.zeros(N, dtype=np.float32)

    base_reg = tok_arr[base_top1.clip(0, V-1)]
    gold_reg = tok_arr[gold.clip(0, V-1)]
    same_reg = (base_reg == gold_reg) & (base_reg != unk_r)
    same_sup = np.zeros(N, dtype=bool)
    if sr:
        base_sup = reg_arr[base_reg.clip(0, R-1)]
        gold_sup = reg_arr[gold_reg.clip(0, R-1)]
        same_sup = (base_sup == gold_sup) & (base_sup != unk_s)

    med_margin = np.median(base_margin)
    return {
        "all":                    np.ones(N, dtype=bool),
        "bucketA_confuser":       base_wrong & gold_in_pool,
        "high_base_margin":       base_margin > med_margin,
        "low_base_margin":        base_margin <= med_margin,
        "same_region_confuser":   base_wrong & gold_in_pool & same_reg,
        "same_superregion_confuser": base_wrong & gold_in_pool & same_sup,
    }

def build_ngram_slices(phrase_results_all, ngram_lengths, N):
    """Returns per-n slices as flat dict with names like phrase_key_hit_4."""
    sl = {}
    for n in ngram_lengths:
        kh = np.array([phrase_results_all[i][n]["key_hit"] for i in range(N)], dtype=bool)
        tc = np.array([phrase_results_all[i][n]["total_count"] for i in range(N)], dtype=np.float32)
        tp = np.array([phrase_results_all[i][n]["phrase_top_prob"] for i in range(N)], dtype=np.float32)
        gpr_arr = np.array([gold_phrase_ranks[i][n]["gold_phrase_rank"] if gold_phrase_ranks[i][n]["in_pool"]
                             else 999 for i in range(N)], dtype=np.int32)
        gip_arr = np.array([gold_phrase_ranks[i][n]["in_pool"] for i in range(N)], dtype=bool)
        sl[f"phrase_key_hit_{n}"]         = kh
        sl[f"phrase_gold_supported_{n}"]  = kh & (np.array(
            [phrase_results_all[i][n]["phrase_counts"][
                np.where(cand_ids_batch[i] == gold_batch[i])[0][0]]
             if gip_arr[i] and len(np.where(cand_ids_batch[i] == gold_batch[i])[0]) > 0
             else 0
             for i in range(N)], dtype=np.float32) > 0)
        sl[f"phrase_gold_top1_{n}"]       = kh & (gpr_arr == 0) & gip_arr
        sl[f"phrase_gold_top3_{n}"]       = kh & (gpr_arr <= 2) & gip_arr
        sl[f"phrase_confident_{n}"]       = kh & (tc >= 4) & (tp >= 0.3)
    return sl

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(data, stores, tok_arr, reg_arr, unk_r, unk_s, sr, args):
    topk_ids = data["topk"]     # [N, K]
    topk_lgt = data["lgt"]      # [N, K]
    gold     = data["gold"]     # [N]
    inp_ids  = data["ids"]      # [N, seq]
    row_ids  = data["row_ids"]  # [N]
    N        = len(gold)
    P        = args.candidate_pool_size
    md_half  = 0.5

    ngram_lengths   = [int(x) for x in args.ngram_lengths.split(",")]
    lambda_grid     = [float(x) for x in args.lambda_grid.split(",")]
    min_count_grid  = [int(x)   for x in args.min_count_grid.split(",")]
    prob_ts         = [float(x) for x in args.prob_thresholds.split(",")]
    margin_ts       = [float(x) for x in args.margin_thresholds.split(",")]

    cand_ids  = topk_ids[:, 1:P+1].copy()   # [N, P] exclude base_top1 at 0
    cand_lgts = np.where(np.isfinite(topk_lgt[:, 1:P+1]),
                         topk_lgt[:, 1:P+1], -1e9)
    base_top1 = topk_ids[:, 0]
    gold_in_pool = (cand_ids == gold[:, None]).any(axis=1)

    # ── Per-row phrase lookup (slow loop — unavoidable) ───────────────────────
    print(f"[eval] Phrase lookup for {N:,} rows × {len(ngram_lengths)} n-gram lengths...")
    phrase_results_all = []   # [N] -> dict of n -> phrase_res
    gpstats_all        = []   # [N] -> dict of n -> gold stats

    # Build batch arrays needed by slice builder
    global cand_ids_batch, gold_batch, gold_phrase_ranks
    cand_ids_batch = cand_ids
    gold_batch     = gold

    for i in range(N):
        pr = phrase_lookup(inp_ids[i], cand_ids[i], stores, ngram_lengths)
        phrase_results_all.append(pr)
        gpstats_all.append(gold_phrase_stats(gold[i], cand_ids[i], pr, ngram_lengths))
        if i % 10000 == 0:
            print(f"  {i:,}/{N:,}...")

    gold_phrase_ranks = gpstats_all

    # Build per-n arrays for slices and rank accumulators
    key_hit_arr    = {}   # n -> [N] bool
    total_count_arr= {}   # n -> [N] float
    phrase_tp_arr  = {}   # n -> [N] float (phrase top prob)
    phrase_ent_arr = {}   # n -> [N] float
    gold_gpr_arr   = {}   # n -> [N] int (gold phrase rank)
    gold_gpp_arr   = {}   # n -> [N] float (gold phrase prob)
    for n in ngram_lengths:
        key_hit_arr[n]     = np.array([phrase_results_all[i][n]["key_hit"]         for i in range(N)], dtype=bool)
        total_count_arr[n] = np.array([phrase_results_all[i][n]["total_count"]      for i in range(N)], dtype=np.float32)
        phrase_tp_arr[n]   = np.array([phrase_results_all[i][n]["phrase_top_prob"]  for i in range(N)], dtype=np.float32)
        phrase_ent_arr[n]  = np.array([phrase_results_all[i][n]["phrase_entropy"]   for i in range(N)], dtype=np.float32)
        gold_gpr_arr[n]    = np.array([gpstats_all[i][n]["gold_phrase_rank"]        for i in range(N)], dtype=np.int32)
        gold_gpp_arr[n]    = np.array([gpstats_all[i][n]["gold_phrase_prob"]        for i in range(N)], dtype=np.float32)

    # ── Slices ────────────────────────────────────────────────────────────────
    print("[eval] Building slices...")
    fixed_slices = build_fixed_slices(gold, base_top1, cand_ids, gold_in_pool, cand_lgts,
                                      tok_arr, reg_arr, unk_r, unk_s, sr)

    # Build n-gram slices manually (avoid global state issues)
    ngram_slices = {}
    for n in ngram_lengths:
        kh  = key_hit_arr[n]
        tc  = total_count_arr[n]
        tp  = phrase_tp_arr[n]
        gpr = gold_gpr_arr[n]
        gip = gold_in_pool
        # gold_supported: gold has phrase count > 0 when key hit
        gold_supp = np.array(
            [kh[i] and gip[i] and
             float(phrase_results_all[i][n]["phrase_counts"][
                 np.where(cand_ids[i] == gold[i])[0][0]
             ]) > 0
             if gip[i] and len(np.where(cand_ids[i] == gold[i])[0]) > 0
             else False
             for i in range(N)], dtype=bool)
        ngram_slices[f"phrase_key_hit_{n}"]        = kh
        ngram_slices[f"phrase_gold_supported_{n}"] = gold_supp
        ngram_slices[f"phrase_gold_top1_{n}"]      = kh & (gpr == 0) & gip
        ngram_slices[f"phrase_gold_top3_{n}"]      = kh & (gpr <= 2) & gip
        ngram_slices[f"phrase_confident_{n}"]      = kh & (tc >= 4) & (tp >= 0.3)

    all_slices = {**fixed_slices, **ngram_slices}
    all_slice_names = list(all_slices.keys())

    # ── Policies ─────────────────────────────────────────────────────────────
    policies = make_policies(ngram_lengths, lambda_grid, min_count_grid, prob_ts, margin_ts)
    print(f"[eval] {len(policies)} policies × {len(all_slice_names)} slices")

    # ── Compute final_top1 for every policy (row-by-row) ─────────────────────
    print("[eval] Applying policies...")
    policy_results = {}   # pname -> (sel_tok [N], apply [N], final_top1 [N])
    for pname, pfn in policies:
        sel_toks   = np.zeros(N, dtype=np.int64)
        apply_mask = np.zeros(N, dtype=bool)
        for i in range(N):
            st, ap = pfn(cand_ids[i], cand_lgts[i], phrase_results_all[i],
                         bool(gold_in_pool[i]), ngram_lengths[0])
            sel_toks[i]  = st
            apply_mask[i]= ap
        # Compute final_top1 via surgical_edit
        final_top1 = np.array([
            surgical_edit(topk_ids[i], topk_lgt[i], int(sel_toks[i]), md_half)
            if apply_mask[i] else int(base_top1[i])
            for i in range(N)], dtype=np.int64)
        policy_results[pname] = (sel_toks, apply_mask, final_top1)
        if pname.endswith("DIAGNOSTIC"):
            # Oracle: override final_top1 — only apply when phrase_top == gold
            n_ora = int(pname.split("_n")[1].split("_")[0])
            oracle_mask = apply_mask & (sel_toks == gold)
            ft_oracle = np.where(oracle_mask,
                                 np.array([surgical_edit(topk_ids[i], topk_lgt[i], gold[i], md_half)
                                           for i in range(N)], dtype=np.int64),
                                 base_top1.copy())
            policy_results[pname] = (sel_toks, oracle_mask, ft_oracle)

    # ── Accumulate metrics ───────────────────────────────────────────────────
    print("[eval] Accumulating metrics...")
    # Use n of first ngram_length as representative for key_hit / total_count in base slices
    n_primary = ngram_lengths[0]

    metrics    = {sn: {pn: MetricAcc() for pn, _ in policies} for sn in all_slice_names}
    rank_stats = {sn: {n: RankAcc(P)   for n in ngram_lengths} for sn in all_slice_names}

    for sn in all_slice_names:
        subset = all_slices[sn]
        if not subset.any():
            continue
        for n in ngram_lengths:
            kh = key_hit_arr[n]
            tc = total_count_arr[n]
            tp = phrase_tp_arr[n]
            et = phrase_ent_arr[n]
            rank_stats[sn][n].update(
                gold_gpr_arr[n], gold_in_pool, gold_gpp_arr[n],
                tp, et, tc, kh, subset)
        for pname, _ in policies:
            st, am, ft = policy_results[pname]
            # Use key_hit / total_count from n embedded in policy name if possible
            # For base_candidate, use n_primary
            try:
                p_n = int(pname.split("_n")[1].split("_")[0])
                kh = key_hit_arr.get(p_n, key_hit_arr[n_primary])
                tc = total_count_arr.get(p_n, total_count_arr[n_primary])
            except Exception:
                kh = key_hit_arr[n_primary]
                tc = total_count_arr[n_primary]
            metrics[sn][pname].update(gold, base_top1, ft, am, st, gold_in_pool, kh, tc, subset)

    # ── Examples ─────────────────────────────────────────────────────────────
    print("[eval] Collecting examples...")
    # Best lambda policy on 'all' slice
    lam_pnames = [f"phrase_n{ngram_lengths[0]}_lam{l:.2f}" for l in lambda_grid]
    best_lam_name = max(lam_pnames,
                        key=lambda p: (metrics["all"].get(p, MetricAcc()).metrics()
                                       .get("net_correction", -1e9)))

    EXAMPLE_KINDS = ["phrase_helps", "phrase_hurts",
                     "phrase_should_help_but_fails", "phrase_key_hits", "phrase_no_hit"]
    ex_buckets = {k: [] for k in EXAMPLE_KINDS}
    max_per    = max(args.max_examples // len(EXAMPLE_KINDS), 1)
    rng        = random.Random(args.seed)

    def make_example(i, n_best):
        pr  = phrase_results_all[i]
        gps = gpstats_all[i]
        pr_n = pr[n_best]
        cands = []
        for j in range(min(P, 16)):
            cands.append({
                "tok_id":       int(cand_ids[i][j]),
                "base_logit":   float(cand_lgts[i][j]),
                "phrase_count": float(pr_n["phrase_counts"][j]),
                "phrase_prob":  float(pr_n["phrase_prob"][j]),
                "final_score":  float(cand_lgts[i][j] + 1.0 * math.log(float(pr_n["phrase_prob"][j]) + _EPS)),
                "is_gold":      int(cand_ids[i][j]) == int(gold[i]),
                "is_base_top1": False,
                "is_phrase_top": j == pr_n["phrase_top_idx"],
            })
        return {
            "row_id":           int(row_ids[i]),
            "ctx_ids":          inp_ids[i, -128:].tolist(),
            "gold_id":          int(gold[i]),
            "base_top1_id":     int(base_top1[i]),
            "base_top1_lgt":    float(topk_lgt[i, 0]),
            "phrase_top_tok":   int(pr_n["phrase_top_tok"]),
            "phrase_top_prob":  float(pr_n["phrase_top_prob"]),
            "ngram_n":          n_best,
            "key_hit":          bool(pr_n["key_hit"]),
            "total_count":      int(pr_n["total_count"]),
            "gold_phrase_rank": int(gps[n_best]["gold_phrase_rank"]),
            "gold_phrase_prob": float(gps[n_best]["gold_phrase_prob"]),
            "gold_phrase_count":float(gps[n_best]["gold_phrase_count"]),
            "selected_tok":     int(policy_results[best_lam_name][0][i]),
            "candidates":       cands,
        }

    idxs = list(range(N)); rng.shuffle(idxs)
    n_best_ex = ngram_lengths[0]  # use shortest for examples (most hits)
    _, _, ft_best = policy_results[best_lam_name]

    for i in idxs:
        bw  = base_top1[i] != gold[i]
        gip = gold_in_pool[i]
        kh4 = phrase_results_all[i][n_best_ex]["key_hit"]
        pt  = phrase_results_all[i][n_best_ex]["phrase_top_tok"]
        gpr = gpstats_all[i][n_best_ex]["gold_phrase_rank"]

        # phrase_helps: base wrong, gold in pool, phrase top == gold, ft improved
        if (len(ex_buckets["phrase_helps"]) < max_per and
                bw and gip and pt == gold[i] and ft_best[i] == gold[i]):
            ex_buckets["phrase_helps"].append(make_example(i, n_best_ex))

        # phrase_hurts: base correct, phrase top != gold, ft changed away
        if (len(ex_buckets["phrase_hurts"]) < max_per and
                not bw and kh4 and pt != gold[i] and ft_best[i] != gold[i]):
            ex_buckets["phrase_hurts"].append(make_example(i, n_best_ex))

        # phrase_should_help_but_fails: base wrong, gold in pool, key hits but phrase top != gold
        if (len(ex_buckets["phrase_should_help_but_fails"]) < max_per and
                bw and gip and kh4 and pt != gold[i]):
            ex_buckets["phrase_should_help_but_fails"].append(make_example(i, n_best_ex))

        # phrase_key_hits: any key hit
        if len(ex_buckets["phrase_key_hits"]) < max_per and kh4:
            ex_buckets["phrase_key_hits"].append(make_example(i, n_best_ex))

        # phrase_no_hit: no key hit for any n
        no_hit = not any(phrase_results_all[i][n_]["key_hit"] for n_ in ngram_lengths)
        if len(ex_buckets["phrase_no_hit"]) < max_per and no_hit:
            ex_buckets["phrase_no_hit"].append(make_example(i, n_best_ex))

        if all(len(v) >= max_per for v in ex_buckets.values()):
            break

    return (metrics, rank_stats, all_slices, all_slice_names, policy_results,
            ex_buckets, policies, ngram_lengths, lambda_grid,
            key_hit_arr, total_count_arr, gold_gpr_arr, gold_gpp_arr,
            phrase_tp_arr, phrase_ent_arr, gold_in_pool, base_top1, gold, N)

# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_KEYS = ["n", "key_hit_rate", "gold_in_pool_rate", "apply_rate", "noop_rate",
                "base_acc", "policy_acc", "acc_gain",
                "selected_gold_given_in_pool",
                "changed_to_gold", "changed_away", "net_correction",
                "benefit_damage_ratio", "applied_precision_ctg", "mean_total_count"]


def write_slice_summary_csv(metrics, policies, all_slice_names, N_total, out_path):
    fields = ["policy", "slice", "fraction_of_val"] + _METRIC_KEYS
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sn in all_slice_names:
            for pname, _ in policies:
                m = metrics[sn][pname].metrics()
                m["fraction_of_val"] = m["n"] / max(N_total, 1)
                row = {"policy": pname, "slice": sn,
                       "fraction_of_val": f"{m['fraction_of_val']:.6f}"}
                for k in _METRIC_KEYS:
                    v = m.get(k, "")
                    row[k] = f"{v:.6f}" if isinstance(v, float) else v
                w.writerow(row)
    print(f"[csv] {out_path}")


def write_policy_grid_csv(metrics, policies, all_slice_names, N_total, out_path):
    focus = ["all", "bucketA_confuser"] + \
            [s for s in all_slice_names if "phrase_gold_top1" in s or "phrase_confident" in s]
    fields = ["policy", "slice", "fraction_of_val"] + _METRIC_KEYS
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sn in focus:
            if sn not in metrics: continue
            for pname, _ in policies:
                m = metrics[sn][pname].metrics()
                m["fraction_of_val"] = m["n"] / max(N_total, 1)
                row = {"policy": pname, "slice": sn,
                       "fraction_of_val": f"{m['fraction_of_val']:.6f}"}
                for k in _METRIC_KEYS:
                    v = m.get(k, "")
                    row[k] = f"{v:.6f}" if isinstance(v, float) else v
                w.writerow(row)
    print(f"[csv] {out_path}")


def write_rank_stats_csv(rank_stats, all_slice_names, ngram_lengths, out_path):
    fields = ["slice", "n", "n_rows", "key_hit_rate",
              "gold_phrase_rank_le_1", "gold_phrase_rank_le_3",
              "gold_phrase_rank_le_5", "gold_phrase_rank_le_10",
              "mean_gold_phrase_prob", "mean_phrase_top_prob",
              "mean_phrase_entropy", "total_count_mean"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sn in all_slice_names:
            for n in ngram_lengths:
                s  = rank_stats[sn][n].stats()
                nn = max(s["n"], 1)
                row = {"slice": sn, "n": n, "n_rows": s["n"],
                       "key_hit_rate":          f"{s['key_hit_rate']:.4f}",
                       "gold_phrase_rank_le_1":  f"{s['gold_phrase_rank_le_1']/nn:.4f}",
                       "gold_phrase_rank_le_3":  f"{s['gold_phrase_rank_le_3']/nn:.4f}",
                       "gold_phrase_rank_le_5":  f"{s['gold_phrase_rank_le_5']/nn:.4f}",
                       "gold_phrase_rank_le_10": f"{s['gold_phrase_rank_le_10']/nn:.4f}",
                       "mean_gold_phrase_prob":  f"{s['mean_gold_phrase_prob']:.4f}",
                       "mean_phrase_top_prob":   f"{s['mean_phrase_top_prob']:.4f}",
                       "mean_phrase_entropy":    f"{s['mean_phrase_entropy']:.4f}",
                       "total_count_mean":       f"{s['total_count_mean']:.2f}"}
                w.writerow(row)
    print(f"[csv] {out_path}")


def _cand_table(cands):
    hdr = ("| rank | token | base_logit | phrase_count | phrase_prob "
           "| final_score | gold? | base? | phrase_top? |")
    sep = ("|------|-------|------------|--------------|-------------|"
           "-------------|-------|-------|-------------|")
    rows = [hdr, sep]
    for j, c in enumerate(cands):
        rows.append(
            f"| {j} | `{_decode_tok(c['tok_id']):<10}` "
            f"| {c['base_logit']:8.3f} "
            f"| {c['phrase_count']:.0f} "
            f"| {c['phrase_prob']:.4f} "
            f"| {c['final_score']:8.3f} "
            f"| {'✓' if c['is_gold'] else ''} "
            f"| {'✓' if c['is_base_top1'] else ''} "
            f"| {'✓' if c['is_phrase_top'] else ''} |")
    return "\n".join(rows)


def write_examples_md(ex_buckets, kind_labels, out_dir):
    # Write combined
    all_lines = [f"# N-gram Phrase Recall — Examples\n", "---\n"]
    for kind, label in kind_labels.items():
        exs = ex_buckets.get(kind, [])
        all_lines.append(f"## {label}  ({len(exs)} examples)\n")
        for idx, ex in enumerate(exs):
            ctx_str = _decode_ids(ex["ctx_ids"])
            gnd = _decode_tok(ex["gold_id"])
            b   = _decode_tok(ex["base_top1_id"])
            pt  = _decode_tok(ex["phrase_top_tok"])
            sl  = _decode_tok(ex["selected_tok"])
            all_lines.append(f"### Example {idx+1}  (row {ex['row_id']})\n")
            all_lines.append(
                f"**Gold:** `{gnd}`  **Base:** `{b}` ({ex['base_top1_lgt']:.3f})  "
                f"**PhraseTop:** `{pt}` (prob={ex['phrase_top_prob']:.4f})  "
                f"**Selected:** `{sl}`\n")
            all_lines.append(
                f"n={ex['ngram_n']}  key_hit={ex['key_hit']}  "
                f"total_count={ex['total_count']}  "
                f"gold_phrase_rank={ex['gold_phrase_rank']}  "
                f"gold_phrase_prob={ex['gold_phrase_prob']:.4f}\n")
            all_lines.append("\n**Context tail:**\n```\n" + ctx_str[-400:] + "\n```\n")
            all_lines.append("\n**Candidates:**\n" + _cand_table(ex["candidates"]) + "\n\n---\n")
        # Per-kind file
        kind_path = os.path.join(out_dir, f"examples_{kind}.md")
        with open(kind_path, "w", encoding="utf-8") as f:
            f.write(f"# {label}\n\n" + "\n".join(
                all_lines[all_lines.index(f"## {label}  ({len(exs)} examples)\n"):]))
        print(f"[md] {kind_path}")
    combined = os.path.join(out_dir, "phrase_examples_all.md")
    with open(combined, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"[md] {combined}")


def _f(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "N/A"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)

def _m(metrics, sn, pname, key):
    return metrics.get(sn, {}).get(pname, MetricAcc()).metrics().get(key, float("nan"))


def write_report(metrics, rank_stats, policies, ngram_lengths, lambda_grid,
                 all_slice_names, out_path, args, N_total):
    pnames    = [p for p, _ in policies]
    lam_pnames_by_n = {n: [f"phrase_n{n}_lam{l:.2f}" for l in lambda_grid]
                        for n in ngram_lengths}
    n_primary = ngram_lengths[0]

    lines = [
        "# N-gram Phrase Recall Baseline V1 — Report\n",
        f"**val_dir:** `{args.val_dir}`  "
        f"**datastore:** `{args.datastore_dir}`\n",
        f"**N total:** {N_total:,}  **pool_size:** {args.candidate_pool_size}\n",
        "",
    ]

    # Slice sizes
    lines.append("## 0. Slice Sizes\n")
    lines.append("| slice | n | fraction |")
    lines.append("|-------|---|----------|")
    for sn in all_slice_names:
        n = _m(metrics, sn, pnames[0], "n") if pnames else 0
        lines.append(f"| {sn} | {int(n) if isinstance(n,float) else n} "
                     f"| {int(n)/max(N_total,1):.4f} |")
    lines.append("")

    # Q1: Key hit rates per n
    lines.append("## Q1. How often do suffixes have exact train matches?\n")
    lines.append("| n | all_key_hit | bucketA_key_hit | phrase_confident_n |")
    lines.append("|---|-------------|-----------------|-------------------|")
    for n in ngram_lengths:
        sn_kh  = f"phrase_key_hit_{n}"
        sn_c   = f"phrase_confident_{n}"
        kh_all  = _m(metrics, "all",             f"phrase_only_n{n}", "key_hit_rate")
        kh_ba   = _m(metrics, "bucketA_confuser", f"phrase_only_n{n}", "key_hit_rate")
        kh_conf = _m(metrics, sn_c,              f"phrase_only_n{n}", "key_hit_rate") if sn_c in metrics else float("nan")
        lines.append(f"| {n} | {_f(kh_all)} | {_f(kh_ba)} | {_f(kh_conf)} |")
    lines.append("")

    # Q2: When key hits, how often does top recalled token == gold
    lines.append("## Q2. When suffix key hits, how often does top recalled token equal gold?\n")
    for n in ngram_lengths:
        sn_kh = f"phrase_key_hit_{n}"
        rs = rank_stats.get(sn_kh, {}).get(n, RankAcc(1)).stats()
        nn = max(rs["n"], 1)
        lines.append(f"**n={n}** (n_rows={rs['n']:,} key-hit rows)")
        lines.append(f"  ptr_top_is_gold (rank_le_1): {rs['gold_phrase_rank_le_1']/nn:.4f}  "
                     f"rank_le_3: {rs['gold_phrase_rank_le_3']/nn:.4f}  "
                     f"mean_gold_prob: {rs['mean_gold_phrase_prob']:.4f}\n")
    lines.append("")

    # Q3: Does phrase recall improve Bucket A
    lines.append("## Q3. Does phrase recall improve Bucket A / confuser cases?\n")
    lines.append("| n | λ | ctg | caw | net | sip | acc_gain |")
    lines.append("|---|---|-----|-----|-----|-----|----------|")
    for n in ngram_lengths:
        best_lp = max(lam_pnames_by_n[n],
                      key=lambda p: _m(metrics, "bucketA_confuser", p, "net_correction"))
        lv = best_lp.split("lam")[1]
        lines.append(
            f"| {n} | {lv} "
            f"| {_f(_m(metrics,'bucketA_confuser',best_lp,'changed_to_gold'))} "
            f"| {_f(_m(metrics,'bucketA_confuser',best_lp,'changed_away'))} "
            f"| {_f(_m(metrics,'bucketA_confuser',best_lp,'net_correction'))} "
            f"| {_f(_m(metrics,'bucketA_confuser',best_lp,'selected_gold_given_in_pool'))} "
            f"| {_f(_m(metrics,'bucketA_confuser',best_lp,'acc_gain'))} |")
    lines.append("")

    # Q4: Does phrase help where pointer cannot
    lines.append("## Q4. Does phrase recall help when gold is not repeated in context?\n")
    lines.append("(Slice `phrase_key_hit_n` ∩ `phrase_gold_supported_n` with n=primary)\n")
    n = n_primary
    sn_gs = f"phrase_gold_supported_{n}"
    if sn_gs in metrics:
        best_lp = max(lam_pnames_by_n[n],
                      key=lambda p: _m(metrics, sn_gs, p, "net_correction"))
        lines.append(f"- n={n}  best_policy={best_lp}")
        lines.append(f"  net={_f(_m(metrics,sn_gs,best_lp,'net_correction'))}  "
                     f"ctg={_f(_m(metrics,sn_gs,best_lp,'changed_to_gold'))}  "
                     f"caw={_f(_m(metrics,sn_gs,best_lp,'changed_away'))}\n")

    # Q5: Which ngram length is best
    lines.append("## Q5. Which ngram length is best?\n")
    lines.append("| n | best_net_all | best_net_bucketA | key_hit_rate_all |")
    lines.append("|---|-------------|------------------|------------------|")
    for n in ngram_lengths:
        best_net_all = max((_m(metrics,"all",p,"net_correction") for p in lam_pnames_by_n[n]),
                           default=float("nan"))
        best_net_ba  = max((_m(metrics,"bucketA_confuser",p,"net_correction") for p in lam_pnames_by_n[n]),
                           default=float("nan"))
        kh = _m(metrics, "all", f"phrase_only_n{n}", "key_hit_rate")
        lines.append(f"| {n} | {_f(best_net_all)} | {_f(best_net_ba)} | {_f(kh)} |")
    lines.append("")

    # Q6: Help vs overfit
    lines.append("## Q6. Does phrase recall mostly help or mostly overfit/repeat wrong tokens?\n")
    for n in [n_primary]:
        ctg_all = max((_m(metrics,"all",p,"changed_to_gold") for p in lam_pnames_by_n[n]), default=float("nan"))
        caw_all = max((_m(metrics,"all",p,"changed_away") for p in lam_pnames_by_n[n]), default=float("nan"))
        if not (math.isnan(ctg_all) or math.isnan(caw_all)):
            if ctg_all > caw_all + 0.001:
                ans6 = f"**Mostly helps** (max ctg={_f(ctg_all)} > max caw={_f(caw_all)})"
            elif caw_all > ctg_all + 0.001:
                ans6 = f"**Mostly hurts** (max caw={_f(caw_all)} > max ctg={_f(ctg_all)})"
            else:
                ans6 = f"**Neutral** (ctg≈caw, max ctg={_f(ctg_all)}, max caw={_f(caw_all)})"
            lines.append(f"- n={n}: {ans6}")
    lines.append("")

    # Q7: Usable confidence gate
    lines.append("## Q7. Is there a usable confidence gate?\n")
    for n in [n_primary]:
        gated_pnames = [p for p, _ in policies if f"gated_n{n}_" in p]
        if gated_pnames:
            best_g = max(gated_pnames, key=lambda p: _m(metrics,"all",p,"net_correction"))
            bg_net = _m(metrics,"all",best_g,"net_correction")
            bg_ar  = _m(metrics,"all",best_g,"apply_rate")
            best_lam_net = max((_m(metrics,"all",p,"net_correction") for p in lam_pnames_by_n[n]),
                               default=float("nan"))
            lines.append(f"- n={n}  best_gated={best_g}")
            lines.append(f"  net={_f(bg_net)}  apply_rate={_f(bg_ar)}  (vs best_lam_net={_f(best_lam_net)})")
            if not (math.isnan(bg_net) or math.isnan(best_lam_net)):
                ans7 = "**YES — gate improves**" if bg_net > best_lam_net + 0.0003 else \
                       "**NO — gating does not improve over best lambda**"
                lines.append(f"  {ans7}")
    lines.append("")

    # Q8: Verdict
    lines.append("## Q8. Should phrase recall become memory expert #2?\n")
    reasons_for = []; reasons_against = []
    n = n_primary
    sn_gt1 = f"phrase_gold_top1_{n}"
    rs_gt1 = rank_stats.get(sn_gt1, {}).get(n, RankAcc(1)).stats()
    nn_gt1 = max(rs_gt1["n"], 1)
    frac_gt1 = rs_gt1["gold_phrase_rank_le_1"] / nn_gt1
    if frac_gt1 > 0.3:
        reasons_for.append(f"phrase_gold_top1_{n}: {frac_gt1:.1%} of key-hit rows gold is phrase top1")
    best_net_all = max((_m(metrics,"all",p,"net_correction") for p in lam_pnames_by_n[n]),
                       default=float("nan"))
    best_net_ba  = max((_m(metrics,"bucketA_confuser",p,"net_correction") for p in lam_pnames_by_n[n]),
                       default=float("nan"))
    if not math.isnan(best_net_all) and best_net_all > 0.003:
        reasons_for.append(f"net_correction on 'all'={best_net_all:.4f} at best lambda")
    if not math.isnan(best_net_ba) and best_net_ba > 0.003:
        reasons_for.append(f"net_correction on bucketA={best_net_ba:.4f}")
    kh_all = _m(metrics,"all",f"phrase_only_n{n}","key_hit_rate")
    if not math.isnan(kh_all) and kh_all < 0.05:
        reasons_against.append(f"key_hit_rate only {kh_all:.1%} on n={n} — sparse coverage")
    if not math.isnan(best_net_all) and best_net_all < 0.001:
        reasons_against.append("net_correction near zero — phrase signal too weak or noisy")
    caw_all = max((_m(metrics,"all",p,"changed_away") for p in lam_pnames_by_n[n]), default=float("nan"))
    if not math.isnan(caw_all) and not math.isnan(best_net_all) and caw_all > best_net_all + 0.001:
        reasons_against.append("changed_away exceeds net benefit — safe only with tight gating")

    if reasons_for:
        lines.append("**Evidence FOR:**")
        for r in reasons_for: lines.append(f"- {r}")
    if reasons_against:
        lines.append("\n**Evidence AGAINST / cautions:**")
        for r in reasons_against: lines.append(f"- {r}")

    strong = len(reasons_for) >= 2 and not math.isnan(best_net_all) and best_net_all > 0.003
    if strong and reasons_against:
        verdict = ("**Conditional YES** — phrase memory shows signal for copy-supported / "
                   "repeated-pattern cases but requires a confidence gate before deployment.")
    elif strong:
        verdict = ("**YES** — phrase memory is a useful specialized evidence path. "
                   "Implement as a gated memory expert alongside pointer memory.")
    elif reasons_for:
        verdict = ("**Marginal** — phrase signal exists but is weak. Consider as a soft "
                   "feature input to gate/selector rather than a standalone expert.")
    else:
        verdict = ("**NO** — exact n-gram recall is insufficient. Consider: (a) fuzzy phrase "
                   "retrieval; (b) BM25/TF-IDF over context; (c) embedding-based retrieval.")
    lines.append(f"\n**Verdict:** {verdict}\n")

    lines.append("---\n")
    lines.append("*Generated by eval_ngram_phrase_recall_v1.py — training-free diagnostic.*\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Evaluate n-gram phrase recall")
    p.add_argument("--val_dir",          required=True)
    p.add_argument("--datastore_dir",    required=True)
    p.add_argument("--token_to_region",  default=None)
    p.add_argument("--super_map",        default=None)
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--candidate_pool_size", type=int, default=32)
    p.add_argument("--top_k",            type=int, default=256)
    p.add_argument("--ngram_lengths",    default="4,8,16,32")
    p.add_argument("--lambda_grid",      default="0.0,0.25,0.5,1.0,2.0,4.0")
    p.add_argument("--min_count_grid",   default="1,2,4,8")
    p.add_argument("--prob_thresholds",  default="0.1,0.2,0.3,0.5,0.7")
    p.add_argument("--margin_thresholds",default="0.0,0.05,0.1,0.2")
    p.add_argument("--max_examples",     type=int, default=50)
    p.add_argument("--seed",             type=int, default=42)
    return p, p.parse_args()


def main():
    p, args = _parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ngram_lengths = [int(x) for x in args.ngram_lengths.split(",")]

    print("=" * 60)
    print(" N-gram Phrase Recall Eval V1")
    print(f" val_dir:       {args.val_dir}")
    print(f" datastore_dir: {args.datastore_dir}")
    print(f" output_dir:    {args.output_dir}")
    print(f" ngram_lengths: {ngram_lengths}")
    print("=" * 60)

    # Preflight
    if not os.path.isdir(args.val_dir):
        print(f"ERROR: val_dir not found: {args.val_dir}"); sys.exit(1)
    if not os.path.isdir(args.datastore_dir):
        print(f"ERROR: datastore_dir not found: {args.datastore_dir}"); sys.exit(1)

    print("\n[data] Loading val shards...")
    data = load_val_shards(args.val_dir, args.top_k)
    N_total = len(data["gold"])

    print("\n[maps] Loading region maps...")
    if args.token_to_region and os.path.isfile(args.token_to_region):
        tok_arr, reg_arr, unk_r, unk_s, sr = load_maps(args.token_to_region, args.super_map)
        print(f"  sr_enabled={sr}")
    else:
        print("  [WARN] token_to_region not found — region slices disabled")
        V = int(data["topk"].max()) + 2
        tok_arr  = np.zeros(V, dtype=np.int32)
        reg_arr  = np.zeros(2, dtype=np.int32)
        unk_r = 0; unk_s = 0; sr = False

    print("\n[datastore] Loading datastores...")
    stores = load_datastores(args.datastore_dir, ngram_lengths)

    (metrics, rank_stats, all_slices, all_slice_names, policy_results,
     ex_buckets, policies, ngram_lengths, lambda_grid,
     key_hit_arr, total_count_arr, gold_gpr_arr, gold_gpp_arr,
     phrase_tp_arr, phrase_ent_arr, gold_in_pool, base_top1, gold, N) = run_eval(
         data, stores, tok_arr, reg_arr, unk_r, unk_s, sr, args)

    print("\n[write] Writing outputs...")

    write_slice_summary_csv(metrics, policies, all_slice_names, N_total,
        os.path.join(args.output_dir, "phrase_slice_metrics.csv"))
    write_policy_grid_csv(metrics, policies, all_slice_names, N_total,
        os.path.join(args.output_dir, "phrase_policy_grid.csv"))
    write_rank_stats_csv(rank_stats, all_slice_names, ngram_lengths,
        os.path.join(args.output_dir, "phrase_rank_stats.csv"))

    kind_labels = {
        "phrase_helps":               "Phrase HELPS (base wrong, gold top1 in phrase, ft=gold)",
        "phrase_hurts":               "Phrase HURTS (base correct, phrase selects wrong, ft≠gold)",
        "phrase_should_help_but_fails":"Should Help But Fails (bucketA + key_hit, phrase top ≠ gold)",
        "phrase_key_hits":            "Key Hit Examples (any ngram hit)",
        "phrase_no_hit":              "No Hit Examples (no ngram hit)",
    }
    write_examples_md(ex_buckets, kind_labels, args.output_dir)

    write_report(metrics, rank_stats, policies, ngram_lengths, lambda_grid,
        all_slice_names,
        os.path.join(args.output_dir, "phrase_recall_report.md"),
        args, N_total)

    cfg = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Console summary
    n_p = ngram_lengths[0]
    lp_all = [f"phrase_n{n_p}_lam{l:.2f}" for l in lambda_grid]
    print("\n=== n-gram eval summary (n={}, all slice) ===".format(n_p))
    print(f"{'policy':<28} {'net':>7}  {'ctg':>7}  {'caw':>7}  {'sip':>7}  {'khr':>7}")
    print("-" * 65)
    for pname in lp_all[:6]:
        m = metrics["all"].get(pname, MetricAcc()).metrics()
        print(f"{pname:<28} "
              f"{m.get('net_correction',float('nan')):>7.4f}  "
              f"{m.get('changed_to_gold',float('nan')):>7.4f}  "
              f"{m.get('changed_away',float('nan')):>7.4f}  "
              f"{m.get('selected_gold_given_in_pool',float('nan')):>7.4f}  "
              f"{m.get('key_hit_rate',float('nan')):>7.4f}")
    print(f"\n[done] Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
