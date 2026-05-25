#!/usr/bin/env python3
"""
eval_pointer_support_baseline_v1.py — Training-free pointer/copy support baseline.

Phase 1 diagnostic: tests whether explicit pointer/copy evidence exists in the
current validation data for candidate reranking.

NO neural model. NO training. NO gold leakage into candidate selection or scoring.
Gold is used ONLY after scores are computed, for metrics only.

Algorithm per row:
  ctx = input_ids[-memory_len:]
  For each pool candidate c:
    exact_count            = #{pos in ctx | ctx[pos] == c}
    recency_weighted_count = sum exp(-dist/tau) for matching positions
    nearest_distance       = min distance to matching position (inf if absent)
    exact_present          = 1 if exact_count > 0 else 0
    last_occurrence_score  = 1 / sqrt(1 + nearest_distance)
    pointer_score          = recency_weighted_count  [default]

  score_lambda(c) = base_logit(c) + lambda * pointer_score(c)
  selected = argmax score_lambda over pool
  If action_policy == "pure_pointer": EDIT only if exact_present(selected) else NO_OP
  Else: always EDIT to selected (for lambda sweep diagnostics)

Outputs:
  pointer_support_summary.csv
  pointer_support_bucket_stats.csv
  pointer_support_examples.md
  pointer_support_report.md
  config.json

Usage:
  python scripts/eval_pointer_support_baseline_v1.py \\
    --val_dir  runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val \\
    --token_to_region runs/region_maps_128/token_to_region.json \\
    --super_map runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
    --output_dir runs/pointer_sentinel_v1/pointer_support_baseline \\
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

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Project root
# ─────────────────────────────────────────────────────────────────────────────

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# GPT-2 tokenizer (optional)
# ─────────────────────────────────────────────────────────────────────────────

_enc = None
try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
except Exception:
    pass

def _decode_tok(tid: int) -> str:
    if _enc is not None:
        try:
            return repr(_enc.decode([int(tid)]))[1:-1]
        except Exception:
            pass
    return f"<{tid}>"

def _decode_ids(ids) -> str:
    if _enc is not None:
        try:
            return _enc.decode([int(i) for i in ids if 0 <= int(i) < 50257])
        except Exception:
            pass
    return " ".join(_decode_tok(i) for i in ids)

# ─────────────────────────────────────────────────────────────────────────────
# Shard loading — robust key aliases
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = {
    "topk_ids":  ["base_topk_ids",   "base_topk",    "topk_ids"],
    "topk_lgt":  ["base_topk_logits","base_topk_lgt","topk_lgt","topk_logits"],
    "gold":      ["gold_token",      "gold",         "labels"],
    "input_ids": ["input_ids"],
}
_OPTIONAL_FIELDS = {
    "h_prime":   ["h_prime", "h_ctx"],
}


def _get_field(shard: dict, aliases: list, required: bool = True):
    for name in aliases:
        if name in shard:
            return shard[name]
    if required:
        raise KeyError(
            f"Shard missing required field. "
            f"Tried: {aliases}. "
            f"Available keys: {list(shard.keys())}")
    return None


def load_shards(shard_dir: str, top_k: int, max_rows=None):
    """
    Load all val shards.  Returns a dict of numpy arrays.
    Prints shard keys on first shard (preflight).
    """
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt files found in: {shard_dir}")

    print(f"[shards] Found {len(shards)} shard(s) in {shard_dir}")

    all_topk, all_lgt, all_gold, all_ids = [], [], [], []
    total = 0
    first_shard = True

    for sp in shards:
        if max_rows is not None and total >= max_rows:
            break

        sh = torch.load(sp, map_location="cpu", weights_only=False)

        if first_shard:
            print("[preflight] First shard keys:")
            for k, v in sh.items():
                shape = getattr(v, "shape", None)
                dtype = getattr(v, "dtype", type(v).__name__)
                print(f"  {k}: shape={shape}  dtype={dtype}")
            first_shard = False

        topk = _get_field(sh, _REQUIRED_FIELDS["topk_ids"]).long()
        lgt  = _get_field(sh, _REQUIRED_FIELDS["topk_lgt"]).float()
        gold = _get_field(sh, _REQUIRED_FIELDS["gold"]).long()
        ids  = _get_field(sh, _REQUIRED_FIELDS["input_ids"])
        if ids is None:
            raise RuntimeError(
                f"input_ids required for pointer baseline. "
                f"Shard {sp} does not contain input_ids. "
                f"Keys: {list(sh.keys())}")
        ids = ids.long()

        B, K = topk.shape
        if K < top_k:
            topk = torch.cat([topk, torch.zeros(B, top_k - K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, top_k - K), float("nan"))],     1)
        elif K > top_k:
            topk, lgt = topk[:, :top_k], lgt[:, :top_k]

        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, ids = topk[:keep], lgt[:keep], gold[:keep], ids[:keep]
            B = keep

        all_topk.append(topk.numpy())
        all_lgt.append(lgt.numpy())
        all_gold.append(gold.numpy())
        all_ids.append(ids.numpy())
        total += B
        if total % 50000 == 0 and total > 0:
            print(f"  loaded {total:,} rows...")

    data = {
        "topk_ids": np.concatenate(all_topk, 0),   # [N, K]
        "topk_lgt": np.concatenate(all_lgt,  0),   # [N, K]
        "gold":     np.concatenate(all_gold, 0),   # [N]
        "ids":      np.concatenate(all_ids,  0),   # [N, seq_len]
    }
    N = data["topk_ids"].shape[0]
    print(f"[shards] Loaded {N:,} rows  "
          f"K={top_k}  seq_len={data['ids'].shape[1]}")
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Region maps
# ─────────────────────────────────────────────────────────────────────────────

def load_maps(t2r_path: str, super_path=None):
    with open(t2r_path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        t2r = {i: v for i, v in enumerate(raw) if v is not None}
    else:
        t2r = {int(k): v for k, v in raw.items()}

    unk_region = int(max(t2r.values())) + 1 if t2r else 1
    r2s = {}; unk_super = 1; sr_enabled = False

    if super_path and os.path.isfile(super_path):
        with open(super_path) as f:
            raw2 = json.load(f)
        if isinstance(raw2, list):
            r2s = {i: v for i, v in enumerate(raw2) if v is not None}
        else:
            r2s = {int(k): v for k, v in raw2.items()}
        unk_super = int(max(r2s.values())) + 1 if r2s else 1
        sr_enabled = True

    vocab_size = max(t2r.keys(), default=0) + 2
    tok_arr = np.full(vocab_size, unk_region, dtype=np.int32)
    for tok, reg in t2r.items():
        if 0 <= tok < vocab_size:
            tok_arr[tok] = int(reg)

    reg_arr = np.full(unk_region + 2, unk_super, dtype=np.int32)
    for reg, sup in r2s.items():
        if 0 <= reg <= unk_region:
            reg_arr[int(reg)] = int(sup)

    return tok_arr, reg_arr, unk_region, unk_super, sr_enabled

# ─────────────────────────────────────────────────────────────────────────────
# Candidate pool  (top_rank: topk[1 : pool_size+1], no gold forced in)
# ─────────────────────────────────────────────────────────────────────────────

def build_pool(topk_ids: np.ndarray, topk_lgt: np.ndarray, pool_size: int):
    """
    Returns cand_ids [N, P], cand_lgts [N, P].
    Pool = topk[1:pool_size+1] — excludes base_top1 at index 0.
    NaN logits → -1e9 to avoid argmax issues.
    """
    cand_ids  = topk_ids[:, 1:pool_size + 1].copy()   # [N, P]
    cand_lgts = topk_lgt[:, 1:pool_size + 1].copy()   # [N, P]
    cand_lgts = np.where(np.isfinite(cand_lgts), cand_lgts, -1e9)
    return cand_ids, cand_lgts

# ─────────────────────────────────────────────────────────────────────────────
# Pointer feature computation — vectorised over the batch
# ─────────────────────────────────────────────────────────────────────────────

def compute_pointer_features(
    input_ids_np: np.ndarray,   # [N, seq_len]
    cand_ids_np:  np.ndarray,   # [N, P]
    memory_len:   int,
    tau:          float,
):
    """
    Returns:
      exact_count   [N, P]  int32
      recency       [N, P]  float32   (pointer_score default)
      nearest_dist  [N, P]  float32   (inf if absent)
      exact_present [N, P]  bool
      last_occ_score [N, P] float32   = 1 / sqrt(1 + nearest_dist)
    """
    N, _ = input_ids_np.shape
    P    = cand_ids_np.shape[1]
    T    = memory_len

    ctx = input_ids_np[:, -T:].astype(np.int32)  # [N, T]

    # Distances from current position: pos=T-1 (most recent) → dist=0; pos=0 → dist=T-1
    distances    = np.arange(T - 1, -1, -1, dtype=np.float32)  # [T]
    dist_weights = np.exp(-distances / max(tau, 1e-6))           # [T]
    _INF         = float(T + 9999)

    exact_count    = np.zeros((N, P), dtype=np.int32)
    recency        = np.zeros((N, P), dtype=np.float32)
    nearest_dist   = np.full((N, P), _INF, dtype=np.float32)
    exact_present  = np.zeros((N, P), dtype=bool)
    last_occ_score = np.zeros((N, P), dtype=np.float32)

    for j in range(P):
        c = cand_ids_np[:, j]                           # [N]
        matches = (ctx == c[:, None]).astype(np.float32) # [N, T]

        ec  = matches.sum(axis=1).astype(np.int32)      # [N]
        rec = (matches * dist_weights[None, :]).sum(axis=1)  # [N]

        # Nearest distance: mask non-matches with _INF, then take min over T
        nd  = (distances[None, :] * matches
               + _INF * (1.0 - matches)).min(axis=1)    # [N]

        exact_count[:, j]    = ec
        recency[:, j]        = rec.astype(np.float32)
        nearest_dist[:, j]   = nd.astype(np.float32)
        exact_present[:, j]  = ec > 0
        last_occ_score[:, j] = (1.0 / np.sqrt(1.0 + nd)).astype(np.float32)

    return exact_count, recency, nearest_dist, exact_present, last_occ_score

# ─────────────────────────────────────────────────────────────────────────────
# Surgical edit simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_surgical_edit(
    topk_ids:  np.ndarray,   # [N, K]
    topk_lgt:  np.ndarray,   # [N, K]
    sel_toks:  np.ndarray,   # [N]
    apply_mask: np.ndarray,  # [N] bool — True → EDIT
    md_half:   float = 0.5,
) -> np.ndarray:
    """
    Returns final_top1_ids [N].
    For EDIT rows: boost sel_tok by md_half, penalise base_top1 by md_half.
    For NO_OP rows: leave logits unchanged.
    """
    N, K = topk_ids.shape
    ref_lgts = np.where(np.isfinite(topk_lgt), topk_lgt, -1e9)

    ar = np.arange(N)
    # Penalise base_top1 for edit rows
    ref_lgts[apply_mask, 0] -= md_half

    # Find position of sel_tok in topk for edit rows
    edit_idx = np.where(apply_mask)[0]
    if len(edit_idx) > 0:
        st = sel_toks[edit_idx]                               # [n_edit]
        pos = (topk_ids[edit_idx] == st[:, None]).argmax(1)  # [n_edit]
        found = topk_ids[edit_idx, pos] == st
        ref_lgts[edit_idx[found], pos[found]] += md_half

    final_top1 = topk_ids[ar, ref_lgts.argmax(1)]
    return final_top1

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-9

def compute_metrics(
    gold:        np.ndarray,  # [N]
    base_top1:   np.ndarray,  # [N]
    final_top1:  np.ndarray,  # [N]
    apply_mask:  np.ndarray,  # [N] bool
    gold_in_pool: np.ndarray, # [N] bool
    selected_tok: np.ndarray, # [N]
    cand_ids:    np.ndarray,  # [N, P] — for gold_in_pool rank check
    subset:      np.ndarray,  # [N] bool — rows to include
) -> dict:
    N_sub = subset.sum()
    if N_sub == 0:
        return {"n": 0}

    g   = gold[subset]
    b   = base_top1[subset]
    ft  = final_top1[subset]
    am  = apply_mask[subset]
    gip = gold_in_pool[subset]
    st  = selected_tok[subset]

    base_wrong    = b != g
    base_correct  = ~base_wrong
    n_bw          = base_wrong.sum()

    changed_to_gold = am & (ft == g)
    changed_away    = am & (b == g) & (ft != g)

    apply_rate     = am.mean()
    ctg_rate       = changed_to_gold.mean()
    caw_rate       = changed_away.mean()
    net_corr       = ctg_rate - caw_rate
    bdr            = ctg_rate / max(caw_rate, _EPS)
    applied_apply  = am.sum()
    applied_prec   = changed_to_gold.sum() / max(applied_apply, 1)

    # Accuracy
    base_acc  = (b == g).mean()
    ref_acc   = (ft == g).mean()
    acc_gain  = ref_acc - base_acc

    # selected_gold_given_in_pool
    sip = (gip & (st == g)).sum() / max(gip.sum(), 1)

    # selected_gold on base_wrong_covered rows
    bwcov = base_wrong & gip
    sgbwcov = (bwcov & (st == g)).sum() / max(bwcov.sum(), 1)

    return {
        "n":                        int(N_sub),
        "apply_rate":               float(apply_rate),
        "changed_to_gold_rate":     float(ctg_rate),
        "changed_away_rate":        float(caw_rate),
        "net_correction":           float(net_corr),
        "benefit_damage_ratio":     float(bdr),
        "applied_precision_ctg":    float(applied_prec),
        "base_top1_acc":            float(base_acc),
        "refined_top1_acc":         float(ref_acc),
        "candidate_pool_acc_gain":  float(acc_gain),
        "selected_gold_given_pool": float(sip),
        "selected_gold_bwcov":      float(sgbwcov),
        "base_wrong_rate":          float(base_wrong.mean()),
        "gold_in_pool_rate":        float(gip.mean()),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Subset masks
# ─────────────────────────────────────────────────────────────────────────────

def build_subsets(
    gold:          np.ndarray,   # [N]
    base_top1:     np.ndarray,   # [N]
    gold_in_pool:  np.ndarray,   # [N]
    ctx:           np.ndarray,   # [N, T]
    tok_arr:       np.ndarray,   # [V]
    reg_arr:       np.ndarray,   # [R]
    unk_region:    int,
    unk_super:     int,
    sr_enabled:    bool,
) -> dict:
    N  = len(gold)
    V  = tok_arr.shape[0]
    R  = reg_arr.shape[0]

    base_wrong = base_top1 != gold

    # copy_supported_gold: gold token appears in context
    gold_in_ctx = np.zeros(N, dtype=bool)
    for i in range(N):
        gold_in_ctx[i] = np.any(ctx[i] == gold[i])

    # Region info
    def _reg(tids): return tok_arr[np.clip(tids, 0, V - 1)]
    def _sup(regs): return reg_arr[np.clip(regs, 0, R - 1)]

    base_reg = _reg(base_top1)
    gold_reg = _reg(gold)
    same_reg = (base_reg == gold_reg) & (base_reg != unk_region)

    same_sup = np.zeros(N, dtype=bool)
    if sr_enabled:
        base_sup = _sup(base_reg)
        gold_sup = _sup(gold_reg)
        same_sup = (base_sup == gold_sup) & (base_sup != unk_super)

    return {
        "all":                  np.ones(N, dtype=bool),
        "bucketA_confuser":     base_wrong & gold_in_pool,
        "copy_supported_gold":  gold_in_ctx,
        "no_copy_gold":         ~gold_in_ctx,
        "same_region_confuser": base_wrong & gold_in_pool & same_reg,
    }, gold_in_ctx  # return gold_in_ctx for later per-row use

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(data: dict, tok_arr, reg_arr, unk_region, unk_super, sr_enabled, args):
    topk_ids  = data["topk_ids"]   # [N, K]
    topk_lgt  = data["topk_lgt"]   # [N, K]
    gold      = data["gold"]       # [N]
    input_ids = data["ids"]        # [N, seq_len]

    N  = topk_ids.shape[0]
    T  = args.memory_len
    P  = args.candidate_pool_size
    md_half = 0.5   # surgical edit magnitude (training default)

    print(f"\n[eval] N={N:,}  T={T}  P={P}  tau={args.recency_tau}")

    # ── Candidate pool ────────────────────────────────────────────────────────
    cand_ids, cand_lgts = build_pool(topk_ids, topk_lgt, P)   # [N, P]
    base_top1 = topk_ids[:, 0]

    # Gold in pool
    gold_in_pool = (cand_ids == gold[:, None]).any(axis=1)   # [N]

    # Context tail
    ctx = input_ids[:, -T:]   # [N, T]

    # ── Pointer features ──────────────────────────────────────────────────────
    print("[eval] Computing pointer features...")
    exact_count, recency, nearest_dist, exact_present, last_occ_score = \
        compute_pointer_features(input_ids, cand_ids, T, args.recency_tau)
    # pointer_score (default) = recency_weighted_count
    ptr_score = recency   # [N, P]

    # ── Subsets ───────────────────────────────────────────────────────────────
    subsets, gold_in_ctx = build_subsets(
        gold, base_top1, gold_in_pool, ctx,
        tok_arr, reg_arr, unk_region, unk_super, sr_enabled)

    # ── Lambda sweep ─────────────────────────────────────────────────────────
    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    results = []   # list of dicts

    print(f"[eval] Lambda sweep: {lambda_grid}")
    for lam in lambda_grid:
        scores = cand_lgts + lam * ptr_score   # [N, P]
        sel_idx = scores.argmax(axis=1)         # [N]
        sel_tok = cand_ids[np.arange(N), sel_idx]   # [N]
        sel_ptr_score = ptr_score[np.arange(N), sel_idx]

        # Policy: always apply edit (reranking diagnostic)
        apply_mask = np.ones(N, dtype=bool)

        final_top1 = simulate_surgical_edit(
            topk_ids.copy(), topk_lgt.copy(), sel_tok, apply_mask, md_half)

        # copy_supported_selected: selected token appears in context
        sel_in_ctx = exact_present[np.arange(N), sel_idx]

        for subset_name, subset_mask in subsets.items():
            m = compute_metrics(gold, base_top1, final_top1, apply_mask,
                                gold_in_pool, sel_tok, cand_ids, subset_mask)
            m["lambda"]  = lam
            m["subset"]  = subset_name
            m["policy"]  = "always_edit"
            results.append(m)

        # copy_supported_selected subset (dynamic per lambda)
        m = compute_metrics(gold, base_top1, final_top1, apply_mask,
                            gold_in_pool, sel_tok, cand_ids,
                            sel_in_ctx.astype(bool))
        m["lambda"]  = lam
        m["subset"]  = "copy_supported_selected"
        m["policy"]  = "always_edit"
        results.append(m)

    # ── Pure pointer policy ───────────────────────────────────────────────────
    print("[eval] Pure pointer policy...")
    # Score: recency only; select only if exact_present
    ptr_only_scores = ptr_score.copy()
    ptr_only_scores[~exact_present] = -1e9  # suppress absent candidates
    pp_sel_idx = ptr_only_scores.argmax(axis=1)
    pp_sel_tok = cand_ids[np.arange(N), pp_sel_idx]
    pp_apply   = exact_present[np.arange(N), pp_sel_idx]  # only if selected candidate is present

    pp_final = simulate_surgical_edit(
        topk_ids.copy(), topk_lgt.copy(), pp_sel_tok, pp_apply, md_half)
    pp_sel_in_ctx = exact_present[np.arange(N), pp_sel_idx]

    for subset_name, subset_mask in subsets.items():
        m = compute_metrics(gold, base_top1, pp_final, pp_apply,
                            gold_in_pool, pp_sel_tok, cand_ids, subset_mask)
        m["lambda"]  = -1.0   # sentinel: pure pointer
        m["subset"]  = subset_name
        m["policy"]  = "pure_pointer"
        results.append(m)

    m = compute_metrics(gold, base_top1, pp_final, pp_apply,
                        gold_in_pool, pp_sel_tok, cand_ids,
                        pp_sel_in_ctx.astype(bool))
    m["lambda"]  = -1.0
    m["subset"]  = "copy_supported_selected"
    m["policy"]  = "pure_pointer"
    results.append(m)

    # ── Collect example rows ──────────────────────────────────────────────────
    print("[eval] Collecting examples...")
    examples = _collect_examples(
        gold, base_top1, topk_ids, topk_lgt, cand_ids, cand_lgts,
        exact_count, recency, nearest_dist, exact_present, ptr_score,
        gold_in_pool, gold_in_ctx, input_ids, T, args.max_examples, args.seed)

    return results, examples, subsets

# ─────────────────────────────────────────────────────────────────────────────
# Example collection
# ─────────────────────────────────────────────────────────────────────────────

def _collect_examples(gold, base_top1, topk_ids, topk_lgt, cand_ids, cand_lgts,
                      exact_count, recency, nearest_dist, exact_present, ptr_score,
                      gold_in_pool, gold_in_ctx, input_ids, T, max_examples, seed):
    rng    = random.Random(seed)
    N      = len(gold)
    P      = cand_ids.shape[1]

    # "pointer helps": gold in pool, gold in ctx, base_wrong
    helps_idx = np.where(
        (base_top1 != gold) & gold_in_pool & gold_in_ctx)[0].tolist()

    # "pointer hurts": base==gold, gold in ctx, at least one cand in ctx
    hurts_idx = np.where(
        (base_top1 == gold) & gold_in_ctx &
        exact_present.any(axis=1))[0].tolist()

    rng.shuffle(helps_idx)
    rng.shuffle(hurts_idx)
    keep_helps = helps_idx[:max_examples // 2]
    keep_hurts = hurts_idx[:max_examples // 2]

    examples = []
    for kind, indices in [("pointer_helps", keep_helps), ("pointer_hurts", keep_hurts)]:
        for i in indices:
            # Best lambda=1.0 selection for display
            scores_disp = cand_lgts[i] + 1.0 * ptr_score[i]
            sel_idx_d   = scores_disp.argmax()
            sel_tok_d   = int(cand_ids[i, sel_idx_d])

            cands = []
            for j in range(min(P, 16)):
                cands.append({
                    "tok_id":        int(cand_ids[i, j]),
                    "base_logit":    float(cand_lgts[i, j]),
                    "pointer_score": float(ptr_score[i, j]),
                    "exact_count":   int(exact_count[i, j]),
                    "nearest_dist":  float(nearest_dist[i, j]),
                    "final_score":   float(scores_disp[j]),
                    "is_gold":       int(cand_ids[i, j]) == int(gold[i]),
                    "is_selected":   j == int(sel_idx_d),
                })

            examples.append({
                "kind":          kind,
                "row_idx":       int(i),
                "ctx_ids":       input_ids[i, -T:].tolist(),
                "gold_id":       int(gold[i]),
                "base_top1_id":  int(base_top1[i]),
                "base_top1_lgt": float(topk_lgt[i, 0]),
                "sel_tok_id":    sel_tok_d,
                "gold_in_pool":  bool(gold_in_pool[i]),
                "gold_in_ctx":   bool(gold_in_ctx[i]),
                "candidates":    cands,
            })

    return examples

# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_FIELDS = [
    "policy", "lambda", "subset", "n",
    "apply_rate", "changed_to_gold_rate", "changed_away_rate",
    "net_correction", "benefit_damage_ratio", "applied_precision_ctg",
    "base_top1_acc", "refined_top1_acc", "candidate_pool_acc_gain",
    "selected_gold_given_pool", "selected_gold_bwcov",
    "base_wrong_rate", "gold_in_pool_rate",
]

def write_summary_csv(results: list, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {}
            for k in _SUMMARY_FIELDS:
                v = r.get(k, "")
                if isinstance(v, float):
                    row[k] = "" if math.isnan(v) else f"{v:.6f}"
                else:
                    row[k] = v
            w.writerow(row)
    print(f"[csv] {out_path}")


def write_bucket_stats_csv(results: list, out_path: str):
    """One row per (policy, subset), showing best-lambda result."""
    # Group by (policy, subset), take best net_correction over lambdas
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["policy"], r["subset"])].append(r)

    fields = ["policy", "subset", "best_lambda",
              "n", "net_correction", "changed_to_gold_rate", "changed_away_rate",
              "benefit_damage_ratio", "applied_precision_ctg",
              "selected_gold_given_pool", "candidate_pool_acc_gain",
              "gold_in_pool_rate"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (policy, subset), rows in sorted(groups.items()):
            best = max(rows, key=lambda r: r.get("net_correction", -1e9)
                       if r.get("n", 0) > 0 else -1e9)
            row = {
                "policy":    policy,
                "subset":    subset,
                "best_lambda": best.get("lambda", ""),
            }
            for k in fields[3:]:
                v = best.get(k, "")
                if isinstance(v, float):
                    row[k] = "" if math.isnan(v) else f"{v:.6f}"
                else:
                    row[k] = v
            w.writerow(row)
    print(f"[csv] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Markdown examples
# ─────────────────────────────────────────────────────────────────────────────

def _cand_table_md(cands: list) -> str:
    lines = [
        "| rank | token | base_logit | ptr_score | exact_n | nearest_d | final_score | gold? | sel? |",
        "|------|-------|------------|-----------|---------|-----------|-------------|-------|------|",
    ]
    for j, c in enumerate(cands):
        nd = c['nearest_dist']
        nd_str = f"{nd:.1f}" if nd < 9000 else "∞"
        lines.append(
            f"| {j} | `{_decode_tok(c['tok_id']):<10}` "
            f"| {c['base_logit']:8.3f} "
            f"| {c['pointer_score']:.4f} "
            f"| {c['exact_count']} "
            f"| {nd_str} "
            f"| {c['final_score']:8.3f} "
            f"| {'✓' if c['is_gold'] else ''} "
            f"| {'✓' if c['is_selected'] else ''} |"
        )
    return "\n".join(lines)


def write_examples_md(examples: list, out_path: str):
    helps = [e for e in examples if e["kind"] == "pointer_helps"]
    hurts = [e for e in examples if e["kind"] == "pointer_hurts"]

    lines = [
        "# Pointer Support Baseline — Examples\n",
        f"Showing up to {len(helps)} helpful and {len(hurts)} harmful pointer cases "
        f"(at λ=1.0).\n",
        "---\n",
    ]

    for section_label, exs in [("Pointer HELPS (base wrong, gold in context & pool)", helps),
                                ("Pointer HURTS (base correct, wrong candidate has context match)", hurts)]:
        lines.append(f"## {section_label}\n")
        for idx, ex in enumerate(exs):
            ctx_str    = _decode_ids(ex["ctx_ids"])
            gold_str   = _decode_tok(ex["gold_id"])
            base_str   = _decode_tok(ex["base_top1_id"])
            sel_str    = _decode_tok(ex["sel_tok_id"])
            lines.append(f"### Example {idx+1}  (row {ex['row_idx']})\n")
            lines.append(f"**Gold:** `{gold_str}`  "
                         f"**Base top-1:** `{base_str}` (logit={ex['base_top1_lgt']:.3f})  "
                         f"**Selected (λ=1.0):** `{sel_str}`  "
                         f"gold_in_pool={ex['gold_in_pool']}  "
                         f"gold_in_ctx={ex['gold_in_ctx']}\n")
            lines.append("\n**Context tail:**\n")
            lines.append("```\n" + ctx_str[-400:] + "\n```\n")
            lines.append("\n**Candidate table:**\n")
            lines.append(_cand_table_md(ex["candidates"]) + "\n")
            lines.append("\n---\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[md] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Interpretation report
# ─────────────────────────────────────────────────────────────────────────────

def _find(results, policy, subset, lam=None):
    for r in results:
        if r["policy"] == policy and r["subset"] == subset:
            if lam is None or r.get("lambda") == lam:
                return r
    return {}

def _best_lam(results, policy, subset):
    rows = [r for r in results if r["policy"] == policy and r["subset"] == subset
            and r.get("n", 0) > 0]
    if not rows:
        return None, {}
    best = max(rows, key=lambda r: r.get("net_correction", -1e9))
    return best.get("lambda"), best

def _f(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def write_report(results: list, out_path: str, args):
    lines = [
        "# Pointer Support Baseline V1 — Diagnostic Report\n",
        f"**val_dir:** `{args.val_dir}`\n",
        f"**memory_len:** {args.memory_len}  "
        f"**recency_tau:** {args.recency_tau}  "
        f"**pool_size:** {args.candidate_pool_size}\n",
        f"**lambda grid:** {args.lambda_grid}\n",
        "",
    ]

    # ── Lambda sweep table (all subset) ──────────────────────────────────────
    lines.append("## 1. Lambda Sweep — `all` subset (always_edit policy)\n")
    lines.append("| λ | apply_r | ctg | caw | net | bdr | prec | sip | acc_gain |")
    lines.append("|---|---------|-----|-----|-----|-----|------|-----|----------|")
    lam_grid = [float(x) for x in args.lambda_grid.split(",")]
    for lam in lam_grid:
        r = _find(results, "always_edit", "all", lam)
        if not r:
            continue
        lines.append(
            f"| {lam} "
            f"| {_f(r.get('apply_rate'))} "
            f"| {_f(r.get('changed_to_gold_rate'))} "
            f"| {_f(r.get('changed_away_rate'))} "
            f"| {_f(r.get('net_correction'))} "
            f"| {_f(r.get('benefit_damage_ratio'))} "
            f"| {_f(r.get('applied_precision_ctg'))} "
            f"| {_f(r.get('selected_gold_given_pool'))} "
            f"| {_f(r.get('candidate_pool_acc_gain'))} |"
        )
    lines.append("")

    # ── Best lambda per subset ────────────────────────────────────────────────
    lines.append("## 2. Best Lambda per Subset\n")
    lines.append("| subset | best_λ | ctg | caw | net | sip | acc_gain |")
    lines.append("|--------|--------|-----|-----|-----|-----|----------|")
    subset_names = ["all", "bucketA_confuser", "copy_supported_gold",
                    "copy_supported_selected", "no_copy_gold", "same_region_confuser"]
    for sn in subset_names:
        bl, br = _best_lam(results, "always_edit", sn)
        if br:
            lines.append(
                f"| {sn} | {bl} "
                f"| {_f(br.get('changed_to_gold_rate'))} "
                f"| {_f(br.get('changed_away_rate'))} "
                f"| {_f(br.get('net_correction'))} "
                f"| {_f(br.get('selected_gold_given_pool'))} "
                f"| {_f(br.get('candidate_pool_acc_gain'))} |")
    lines.append("")

    # ── Pure pointer policy ───────────────────────────────────────────────────
    lines.append("## 3. Pure Pointer Policy\n")
    lines.append("| subset | apply_r | ctg | caw | net | sip |")
    lines.append("|--------|---------|-----|-----|-----|-----|")
    for sn in subset_names:
        r = _find(results, "pure_pointer", sn)
        if r:
            lines.append(
                f"| {sn} "
                f"| {_f(r.get('apply_rate'))} "
                f"| {_f(r.get('changed_to_gold_rate'))} "
                f"| {_f(r.get('changed_away_rate'))} "
                f"| {_f(r.get('net_correction'))} "
                f"| {_f(r.get('selected_gold_given_pool'))} |")
    lines.append("")

    # ── Diagnostic Q&A ────────────────────────────────────────────────────────
    lines.append("## 4. Diagnostic Questions\n")

    # Q1: gold usually in context for Bucket A?
    r_ba = _find(results, "always_edit", "bucketA_confuser", 0.0)
    r_csg = _find(results, "always_edit", "copy_supported_gold", 0.0)
    r_all = _find(results, "always_edit", "all", 0.0)
    n_all = r_all.get("n", 1)
    n_ba  = r_ba.get("n",  0)
    n_csg = r_csg.get("n", 0)

    gip_ba = r_ba.get("gold_in_pool_rate", float("nan"))
    lines.append("### Q1. Does gold usually appear in context for Bucket A?")
    lines.append(f"- `copy_supported_gold` n={n_csg} / total n={n_all}")
    lines.append(f"- Among bucketA_confuser rows: n={n_ba}  gold_in_pool_rate={_f(gip_ba)}")
    present_frac = n_csg / max(n_all, 1)
    lines.append(f"- Fraction of all rows where gold in context: {_f(present_frac)}")
    ans1 = "**YES**" if present_frac > 0.15 else "**NO** (rare copy signal)"
    lines.append(f"- **Answer:** {ans1}\n")

    # Q2: does pointer rank gold higher?
    r_csg_lam0 = _find(results, "always_edit", "copy_supported_gold", 0.0)
    r_csg_bestl, r_csg_best = _best_lam(results, "always_edit", "copy_supported_gold")
    sip_0    = r_csg_lam0.get("selected_gold_given_pool", float("nan"))
    sip_best = r_csg_best.get("selected_gold_given_pool", float("nan")) if r_csg_best else float("nan")
    lines.append("### Q2. When gold appears in context, does pointer_score rank it higher?")
    lines.append(f"- sip at λ=0 (pure logit): {_f(sip_0)}")
    lines.append(f"- sip at best λ={r_csg_bestl}: {_f(sip_best)}")
    if not (math.isnan(sip_0) or math.isnan(sip_best)):
        delta_sip = sip_best - sip_0
        ans2 = "**YES**" if delta_sip > 0.01 else "**NO** (pointer does not improve gold rank)"
        lines.append(f"- Δsip = {delta_sip:+.4f}: {ans2}")
    lines.append("")

    # Q3: does pointer improve sip overall?
    r_all_0  = _find(results, "always_edit", "all", 0.0)
    bl_all, r_all_best = _best_lam(results, "always_edit", "all")
    sip_all_0    = r_all_0.get("selected_gold_given_pool", float("nan"))
    sip_all_best = r_all_best.get("selected_gold_given_pool", float("nan")) if r_all_best else float("nan")
    lines.append("### Q3. Does pointer reranking improve selected_gold_given_pool?")
    lines.append(f"- sip λ=0: {_f(sip_all_0)}  →  best λ={bl_all}: {_f(sip_all_best)}")
    if not (math.isnan(sip_all_0) or math.isnan(sip_all_best)):
        d = sip_all_best - sip_all_0
        lines.append(f"- Δsip = {d:+.4f}: {'**YES**' if d > 0.005 else '**MARGINAL/NO**'}")
    lines.append("")

    # Q4: pointer reduce or increase caw?
    caw_0    = r_all_0.get("changed_away_rate", float("nan"))
    caw_best = r_all_best.get("changed_away_rate", float("nan")) if r_all_best else float("nan")
    lines.append("### Q4. Does pointer reranking reduce or increase changed_away?")
    lines.append(f"- caw λ=0: {_f(caw_0)}  →  best λ={bl_all}: {_f(caw_best)}")
    if not (math.isnan(caw_0) or math.isnan(caw_best)):
        dcaw = caw_best - caw_0
        if dcaw < -0.001:
            lines.append(f"- Δcaw = {dcaw:+.4f}: **REDUCES** changed_away (safer edits)")
        elif dcaw > 0.002:
            lines.append(f"- Δcaw = {dcaw:+.4f}: **INCREASES** changed_away (risky — need gate)")
        else:
            lines.append(f"- Δcaw = {dcaw:+.4f}: **NEUTRAL**")
    lines.append("")

    # Q5: best lambda
    lines.append("### Q5. Which lambda gives best net correction?")
    lines.append(f"- Best λ for `all` subset: {bl_all}")
    if r_all_best:
        lines.append(f"  net={_f(r_all_best.get('net_correction'))}  "
                     f"ctg={_f(r_all_best.get('changed_to_gold_rate'))}  "
                     f"caw={_f(r_all_best.get('changed_away_rate'))}")
    # Bucket A best
    bl_ba, r_ba_best = _best_lam(results, "always_edit", "bucketA_confuser")
    if r_ba_best:
        lines.append(f"- Best λ for `bucketA_confuser`: {bl_ba}  "
                     f"net={_f(r_ba_best.get('net_correction'))}")
    lines.append("")

    # Q6: worth turning into a learned module?
    lines.append("### Q6. Is pointer evidence worth turning into a learned module?\n")
    net_best = r_all_best.get("net_correction", float("nan")) if r_all_best else float("nan")
    net_ba   = r_ba_best.get("net_correction", float("nan")) if r_ba_best else float("nan")
    n_csg_csupported = _find(results, "always_edit", "copy_supported_gold", 0.0).get("n", 0)
    frac_csg = n_csg_csupported / max(n_all, 1)

    reasons = []
    if not math.isnan(net_best) and net_best > 0.002:
        reasons.append(f"net_correction={_f(net_best)} > 0 across all rows")
    if not math.isnan(net_ba) and net_ba > 0.005:
        reasons.append(f"strong net_correction={_f(net_ba)} in bucketA_confuser")
    if not math.isnan(sip_best) and sip_best > sip_0 + 0.01:
        reasons.append("pointer improves gold selection in copy_supported_gold examples")
    if frac_csg > 0.10:
        reasons.append(f"{frac_csg*100:.1f}% of rows have gold in context — substantial copy signal")

    risks = []
    if not math.isnan(caw_best) and caw_best > caw_0 + 0.002:
        risks.append("pointer increases changed_away — a learned gate is essential before deployment")
    if frac_csg < 0.05:
        risks.append(f"only {frac_csg*100:.1f}% of rows have copy signal — limited applicability")

    if reasons:
        lines.append("**Evidence FOR building a learned pointer module:**")
        for r in reasons:
            lines.append(f"- {r}")
    if risks:
        lines.append("\n**Risks / cautions:**")
        for r in risks:
            lines.append(f"- {r}")
    if not reasons:
        lines.append("**Evidence is weak.** Pure pointer does not show clear improvement. "
                     "Consider: (a) exact-match + recency are too sparse; "
                     "(b) soft/embedding-based match may be needed.")
    lines.append("")

    # ── Decision summary ─────────────────────────────────────────────────────
    lines.append("## 5. Decision Summary\n")

    net_b   = net_best if not math.isnan(net_best) else 0.0
    dcaw_v  = (caw_best - caw_0) if not (math.isnan(caw_best) or math.isnan(caw_0)) else 0.0
    dsip_v  = (sip_best - sip_0) if not (math.isnan(sip_best) or math.isnan(sip_0)) else 0.0

    if frac_csg > 0.10 and dsip_v > 0.01 and net_b > 0:
        decision = (
            "**Pointer path is useful for copy-supported cases.** "
            "Build a learned pointer encoder as a feature alongside the existing "
            "selector. Gate the edit on pointer confidence. "
            "Avoid applying pointer to non-copy rows.")
    elif dsip_v > 0.01 and dcaw_v > 0.003:
        decision = (
            "**Pointer improves selection but increases harm.** "
            "Do NOT apply pointer edits without a trained gate. "
            "Use pointer_score as an input feature to the gate only.")
    elif dsip_v < 0.005 and net_b < 0.001:
        decision = (
            "**Pure exact pointer does not expose enough signal.** "
            "Exact token match is too sparse or the wrong granularity. "
            "Alternatives: (a) embedding-similarity-based soft pointer; "
            "(b) n-gram match; "
            "(c) use h_prime attention at context positions instead.")
    else:
        decision = (
            "**Marginal signal.** Pointer reranking is mildly helpful but not decisive. "
            "Combine pointer_score as one of several features in the selector/gate, "
            "but do not build a standalone pointer model.")

    lines.append(decision + "\n")
    lines.append("---\n")
    lines.append("*Generated by eval_pointer_support_baseline_v1.py — training-free diagnostic.*\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Training-free pointer/copy support baseline")
    p.add_argument("--val_dir",           required=True)
    p.add_argument("--token_to_region",   required=True)
    p.add_argument("--super_map",         default=None)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--top_k",             type=int,   default=256)
    p.add_argument("--candidate_pool_size", type=int, default=32)
    p.add_argument("--memory_len",        type=int,   default=128)
    p.add_argument("--recency_tau",       type=float, default=32.0)
    p.add_argument("--lambda_grid",       default="0.0,0.25,0.5,1.0,2.0,4.0")
    p.add_argument("--max_examples",      type=int,   default=200)
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


def main():
    args = _parse()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[data] Loading val shards from {args.val_dir} ...")
    data = load_shards(args.val_dir, args.top_k)

    # ── Load maps ─────────────────────────────────────────────────────────────
    print(f"\n[maps] Loading region maps...")
    tok_arr, reg_arr, unk_region, unk_super, sr_enabled = load_maps(
        args.token_to_region, args.super_map)
    print(f"  unk_region={unk_region}  sr_enabled={sr_enabled}")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results, examples, subsets = run_eval(
        data, tok_arr, reg_arr, unk_region, unk_super, sr_enabled, args)

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\n[write] Writing outputs...")
    write_summary_csv(results,
                      os.path.join(args.output_dir, "pointer_support_summary.csv"))
    write_bucket_stats_csv(results,
                           os.path.join(args.output_dir, "pointer_support_bucket_stats.csv"))
    write_examples_md(examples,
                      os.path.join(args.output_dir, "pointer_support_examples.md"))
    write_report(results,
                 os.path.join(args.output_dir, "pointer_support_report.md"), args)

    print(f"\n[done] Outputs in: {args.output_dir}")
    print(f"  Key file: {args.output_dir}/pointer_support_report.md")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== Quick summary (all, always_edit) ===")
    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    print(f"{'λ':>6}  {'ctg':>8}  {'caw':>8}  {'net':>8}  {'sip':>8}  {'acc_gain':>9}")
    print("-" * 60)
    for lam in lambda_grid:
        r = _find(results, "always_edit", "all", lam)
        if r.get("n", 0) == 0:
            continue
        print(f"{lam:>6.2f}  "
              f"{r.get('changed_to_gold_rate', float('nan')):>8.4f}  "
              f"{r.get('changed_away_rate', float('nan')):>8.4f}  "
              f"{r.get('net_correction', float('nan')):>8.4f}  "
              f"{r.get('selected_gold_given_pool', float('nan')):>8.4f}  "
              f"{r.get('candidate_pool_acc_gain', float('nan')):>9.4f}")
    r_pp = _find(results, "pure_pointer", "all")
    if r_pp.get("n", 0) > 0:
        print(f"{'ptr':>6}  "
              f"{r_pp.get('changed_to_gold_rate', float('nan')):>8.4f}  "
              f"{r_pp.get('changed_away_rate', float('nan')):>8.4f}  "
              f"{r_pp.get('net_correction', float('nan')):>8.4f}  "
              f"{r_pp.get('selected_gold_given_pool', float('nan')):>8.4f}  "
              f"  (pure_pointer)")
    print()


if __name__ == "__main__":
    main()
