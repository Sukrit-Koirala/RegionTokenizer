#!/usr/bin/env python3
"""
evaluate_compute_aware_region_maps.py — Phase 1B evaluation

Evaluates region maps for candidate compression quality WITHOUT training a router.
Uses train/val gold distribution and frequency-prior region ranking.

Key metrics:
  - region size distribution
  - oracle gold coverage (what k needed to cover gold for each row)
  - frequency-prior top-k coverage curves
  - head coverage for V2C
  - recall@k at fixed candidate fractions
  - recall-per-cost score

Usage:
  python scripts/evaluate_compute_aware_region_maps.py \\
    --maps_root runs/cheap_ai/phase1B_compute_aware_region_maps \\
    --train_dir runs/.../train \\
    --val_dir   runs/.../val \\
    --old_map   runs/region_maps_128/token_to_region.json \\
    --output_dir runs/cheap_ai/phase1B_compute_aware_region_maps/eval \\
    --ks 1,2,4,8,16,32,64 --seed 42
"""

import argparse
import csv
import glob
import json
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)

_EPS = 1e-9

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    if isinstance(v, float): return f"{v:.5f}" if v == v else "nan"
    return str(v)

def _wcsv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[save] {path}")

def _write(path, text):
    with open(path, "w", encoding="utf-8") as f: f.write(text)
    print(f"[save] {path}")

def _safediv(a, b):
    return a / b if b > 0 else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Data / map loading
# ══════════════════════════════════════════════════════════════════════════════

_GOLD_AL = ["gold_token", "gold", "labels"]
_TOPK_AL = ["base_topk_ids", "base_topk", "topk_ids"]

def _get_key(sh, aliases, required=True):
    for a in aliases:
        if a in sh: return sh[a]
    if required: raise KeyError(f"Need one of {aliases}")
    return None


def load_gold_tokens(shard_dir: str, max_rows: Optional[int], label: str) -> np.ndarray:
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not paths: raise FileNotFoundError(f"No shards in {shard_dir}")
    golds = []
    total = 0
    for sp in paths:
        if max_rows and total >= max_rows: break
        sh   = torch.load(sp, map_location="cpu", weights_only=False)
        gold = _get_key(sh, _GOLD_AL).numpy().astype(np.int32)
        if max_rows and total + len(gold) > max_rows:
            gold = gold[:max_rows - total]
        golds.append(gold); total += len(gold)
    arr = np.concatenate(golds)
    print(f"[{label}] gold tokens: {len(arr):,}")
    return arr


def load_base_topk(shard_dir: str, max_rows: Optional[int]) -> Optional[np.ndarray]:
    """Load base_topk_ids from val shards (for additional coverage metrics)."""
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not paths: return None
    sh = torch.load(paths[0], map_location="cpu", weights_only=False)
    topk = _get_key(sh, _TOPK_AL, required=False)
    if topk is None: return None
    # Load just a sample for evaluation
    bufs = []
    total = 0
    for sp in paths:
        if max_rows and total >= max_rows: break
        sh   = torch.load(sp, map_location="cpu", weights_only=False)
        tk   = _get_key(sh, _TOPK_AL, required=False)
        if tk is None: continue
        tk   = tk.numpy().astype(np.int32)
        if max_rows and total + len(tk) > max_rows: tk = tk[:max_rows - total]
        bufs.append(tk); total += len(tk)
    return np.concatenate(bufs) if bufs else None


def load_map(map_dir: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Load a region map from map_dir.
    Returns (map_info_dict, policy_type).
    """
    t2r_path   = os.path.join(map_dir, "token_to_region.json")
    head_path  = os.path.join(map_dir, "head_token_ids.npy")
    policy_path= os.path.join(map_dir, "candidate_policy.json")
    cfg_path   = os.path.join(map_dir, "map_config.json")

    if not os.path.isfile(t2r_path):
        return None, None

    with open(t2r_path) as f: t2r = json.load(f)
    t2r_int = {int(k): int(v) for k, v in t2r.items()}
    K = max(t2r_int.values()) + 1 if t2r_int else 1
    V = max(t2r_int.keys(), default=0) + 2
    tok_arr = np.full(V, K, dtype=np.int32)
    for t, r in t2r_int.items(): tok_arr[t] = r

    head_ids = None
    if os.path.isfile(head_path):
        head_ids = np.load(head_path)

    cfg = {}
    if os.path.isfile(cfg_path):
        cfg = json.load(open(cfg_path))

    policy = "standard"
    if os.path.isfile(policy_path):
        pc = json.load(open(policy_path))
        policy = pc.get("policy", "standard")

    # Region sizes
    region_sz = np.bincount(tok_arr[tok_arr < K], minlength=K).astype(np.int64)
    vocab_known = int((tok_arr < K).sum())

    return {
        "map_name":     os.path.basename(map_dir),
        "map_dir":      map_dir,
        "tok_arr":      tok_arr,
        "K":            K,
        "V":            V,
        "head_ids":     head_ids,
        "policy":       policy,
        "region_sz":    region_sz,
        "vocab_known":  vocab_known,
        "cfg":          cfg,
    }, policy


def load_old_map_info(path: str) -> Optional[dict]:
    if not path or not os.path.isfile(path): return None
    with open(path) as f: raw = json.load(f)
    if isinstance(raw, list):
        t2r = {i: v for i, v in enumerate(raw) if v is not None}
    else:
        t2r = {int(k): v for k, v in raw.items()}
    K  = int(max(t2r.values())) + 1 if t2r else 1
    V  = max(t2r.keys(), default=0) + 2
    ta = np.full(V, K, dtype=np.int32)
    for t, r in t2r.items(): ta[int(t)] = int(r)
    sz = np.bincount(ta[ta < K], minlength=K).astype(np.int64)
    return {"map_name": "OLD_K128", "map_dir": os.path.dirname(path),
            "tok_arr": ta, "K": K, "V": V, "head_ids": None,
            "policy": "standard", "region_sz": sz,
            "vocab_known": int((ta < K).sum()), "cfg": {}}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation logic
# ══════════════════════════════════════════════════════════════════════════════

def compute_gold_distribution(gold: np.ndarray, tok_arr: np.ndarray, K: int,
                               head_ids: Optional[np.ndarray]) -> Tuple[np.ndarray, dict]:
    """Returns gold_region [N] array and distribution stats."""
    Vt = len(tok_arr)
    gold_clipped = np.clip(gold.astype(np.int64), 0, Vt - 1)
    gold_region  = tok_arr[gold_clipped].astype(np.int64)
    known_mask   = (gold_region < K)

    head_set = set(int(h) for h in head_ids) if head_ids is not None else set()
    gold_in_head = np.array([int(g) in head_set for g in gold], dtype=bool)

    region_gold_freq = np.bincount(gold_region[known_mask], minlength=K).astype(np.int64)
    stats = {
        "n_total":           len(gold),
        "n_known_region":    int(known_mask.sum()),
        "n_unknown_region":  int((~known_mask).sum()),
        "n_gold_in_head":    int(gold_in_head.sum()),
        "head_gold_rate":    _safediv(int(gold_in_head.sum()), len(gold)),
        "tail_gold_rate":    _safediv(int(known_mask.sum()), len(gold)),
    }
    return gold_region, region_gold_freq, known_mask, gold_in_head, stats


def frequency_prior_recall_curve(gold_region: np.ndarray,
                                  region_gold_train: np.ndarray,
                                  region_sz: np.ndarray,
                                  known_mask: np.ndarray,
                                  gold_in_head: np.ndarray,
                                  head_ids: Optional[np.ndarray],
                                  vocab_known: int,
                                  K: int,
                                  ks: List[int]) -> List[dict]:
    """Frequency-prior recall curve: rank regions by train gold frequency."""
    freq_order = np.argsort(-region_gold_train)  # descending
    N_known = int(known_mask.sum())
    n_head  = int(gold_in_head.sum()) if gold_in_head is not None else 0

    rows = []
    for k in ks:
        k_ = min(k, K)
        top_r   = set(freq_order[:k_].tolist())
        # Effective coverage includes head coverage for V2C
        hits_tail = sum(1 for i in np.where(known_mask)[0] if int(gold_region[i]) in top_r)
        hits_head = n_head  # gold_in_head is always covered for V2C
        if head_ids is not None:
            eff_recall = (hits_tail + hits_head) / max(len(gold_region), 1)
        else:
            eff_recall = hits_tail / max(N_known, 1)
        tail_recall = hits_tail / max(N_known, 1)

        # Candidate size
        cand_sz  = int(region_sz[freq_order[:k_]].sum())
        head_sz  = len(head_ids) if head_ids is not None else 0
        total_cand = cand_sz + head_sz
        frac     = _safediv(total_cand, vocab_known)
        score    = _safediv(eff_recall, frac)

        rows.append({
            "k":                  k,
            "freq_prior_recall":  round(eff_recall, 5),
            "freq_prior_tail_recall": round(tail_recall, 5),
            "avg_candidate_size": cand_sz + head_sz,
            "candidate_fraction": round(frac, 5),
            "score_recall_per_frac": round(score, 4),
        })
    return rows


def oracle_coverage_distribution(gold_region: np.ndarray,
                                  known_mask: np.ndarray,
                                  gold_in_head: np.ndarray,
                                  head_ids: Optional[np.ndarray],
                                  region_sz: np.ndarray,
                                  vocab_known: int,
                                  K: int) -> Tuple[dict, dict]:
    """
    For each val row, compute minimum k (number of regions) needed to cover gold.
    Returns (percentile_stats, coverage_by_frac).
    """
    # For V2C: if gold is in head, oracle k=0 (auto-covered)
    has_head = head_ids is not None
    min_k_arr = np.full(len(gold_region), K + 1, dtype=np.int32)

    for i in np.where(known_mask)[0]:
        gr = int(gold_region[i])
        if has_head and gold_in_head[i]:
            min_k_arr[i] = 0
        elif gr < K:
            # Oracle k = rank of gold_region in frequency order (0-indexed) + 1
            # We compute this as: among known regions, what rank would gold have?
            # For oracle, it's always k=1 if gold is included. We want "min k to include gold"
            # = 1 for all known rows (oracle routing picks the right region)
            min_k_arr[i] = 1  # oracle always covers with 1 region (trivially)

    # More useful: oracle distribution of REGION SIZE needed to include gold token
    oracle_cand_sz = np.zeros(len(gold_region), dtype=np.int64)
    for i in np.where(known_mask)[0]:
        gr = int(gold_region[i])
        if has_head and gold_in_head[i]:
            oracle_cand_sz[i] = len(head_ids)
        elif gr < K:
            oracle_cand_sz[i] = int(region_sz[gr]) + (len(head_ids) if has_head else 0)

    head_sz = len(head_ids) if has_head else 0
    km = known_mask
    stats = {
        "oracle_avg_cand_size": round(float(oracle_cand_sz[km].mean()), 1) if km.any() else float("nan"),
        "oracle_p50_cand_size": round(float(np.median(oracle_cand_sz[km])), 1) if km.any() else float("nan"),
        "oracle_p90_cand_size": round(float(np.percentile(oracle_cand_sz[km], 90)), 1) if km.any() else float("nan"),
        "oracle_p99_cand_size": round(float(np.percentile(oracle_cand_sz[km], 99)), 1) if km.any() else float("nan"),
        "oracle_avg_cand_frac": round(_safediv(float(oracle_cand_sz[km].mean()) if km.any() else 0.0, vocab_known), 5),
        "oracle_worst_region_size": int(region_sz.max()),
        "oracle_coverage_100pct_requires_frac": round(_safediv(int(region_sz.max()) + head_sz, vocab_known), 5),
    }

    # Coverage at fixed fractions [0.10, 0.20, 0.30, 0.40, 0.50]
    cov_by_frac = {}
    for frac_target in [0.10, 0.20, 0.30, 0.40, 0.50]:
        cand_limit = int(frac_target * vocab_known)
        # How many rows can be covered if we allocate cand_limit tokens?
        # With oracle routing, each row needs oracle_cand_sz[i] tokens.
        covered = int((oracle_cand_sz[km] <= cand_limit).sum())
        cov_by_frac[frac_target] = round(_safediv(covered, int(km.sum())), 5)

    return stats, cov_by_frac


def recall_at_fixed_fractions(freq_curve: List[dict]) -> dict:
    """Interpolate recall at fixed candidate fractions from frequency-prior curve."""
    fracs = [0.10, 0.20, 0.30, 0.40, 0.50]
    out   = {}
    for frac_target in fracs:
        # Find first curve point >= frac_target
        found = None
        for row in freq_curve:
            if row["candidate_fraction"] >= frac_target - 1e-4:
                found = row
                break
        out[f"recall_at_frac_{int(frac_target*100):02d}pct"] = found["freq_prior_recall"] if found else float("nan")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Map eval summary
# ══════════════════════════════════════════════════════════════════════════════

def eval_one_map(map_info: dict,
                 train_gold: np.ndarray,
                 val_gold: np.ndarray,
                 ks: List[int]) -> dict:
    tok_arr   = map_info["tok_arr"]
    K         = map_info["K"]
    region_sz = map_info["region_sz"]
    head_ids  = map_info["head_ids"]
    vocab_known = map_info["vocab_known"]
    map_name  = map_info["map_name"]

    # Train gold distribution → frequency prior ordering
    t_gold_r, t_freq, t_known, t_head, t_stats = compute_gold_distribution(
        train_gold, tok_arr, K, head_ids)
    # Val gold distribution → for coverage eval
    v_gold_r, v_freq, v_known, v_head, v_stats = compute_gold_distribution(
        val_gold, tok_arr, K, head_ids)

    # Frequency prior curve (using train frequency, evaluated on val)
    freq_curve = frequency_prior_recall_curve(
        v_gold_r, t_freq, region_sz, v_known, v_head, head_ids, vocab_known, K, ks)

    # Oracle coverage stats
    oracle_stats, oracle_by_frac = oracle_coverage_distribution(
        v_gold_r, v_known, v_head, head_ids, region_sz, vocab_known, K)

    # Recall at fixed fractions
    frac_recall = recall_at_fixed_fractions(freq_curve)

    # Size diagnostics
    sizes = region_sz.astype(np.float64)
    sz_diag = {
        "n_regions":            K,
        "vocab_known":          vocab_known,
        "min_size":             int(sizes.min()),
        "max_size":             int(sizes.max()),
        "mean_size":            round(float(sizes.mean()), 1),
        "p90_size":             round(float(np.percentile(sizes, 90)), 1),
        "p99_size":             round(float(np.percentile(sizes, 99)), 1),
        "num_over_2x_mean":     int((sizes > 2 * sizes.mean()).sum()),
        "head_size":            len(head_ids) if head_ids is not None else 0,
        "head_gold_rate_val":   round(v_stats["head_gold_rate"], 5),
    }

    summary = {
        "map_name": map_name,
        **sz_diag,
        **v_stats,
        **oracle_stats,
        **{f"oracle_at_frac_{int(k*100):02d}pct": v for k, v in oracle_by_frac.items()},
        **frac_recall,
    }

    # Add freq prior at key k values
    for row in freq_curve:
        k = row["k"]
        summary[f"freq_prior_recall@{k}"]   = row["freq_prior_recall"]
        summary[f"candidate_fraction@{k}"]  = row["candidate_fraction"]
        summary[f"recall_per_frac_score@{k}"] = row["score_recall_per_frac"]

    return summary, freq_curve


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def write_eval_report(summaries: List[dict], freq_curves: Dict[str, List[dict]],
                       ks: List[int], out_dir: str, args) -> str:

    def _v(d, k): return d.get(k, float("nan"))

    # Compare recall at fixed fractions: find best map at frac=0.10, 0.20, 0.30
    best_at = {}
    for frac in [10, 20, 30]:
        key = f"recall_at_frac_{frac:02d}pct"
        best = max(summaries, key=lambda d: _v(d, key) if _v(d, key) == _v(d, key) else -1)
        best_at[frac] = (best.get("map_name", "?"), round(_v(best, key), 5))

    lines = [
        "# Phase 1B: Compute-Aware Region Maps — Evaluation Report",
        "",
        f"**val_dir:** {args.val_dir}  |  **train_dir:** {args.train_dir}",
        "", "---", "",
        "## Map Summary Table", "",
        "| map | K | max_sz | over_2x | head_sz | head_gold% | "
        "recall@8 | frac@8 | score@8 | recall@16 | frac@16 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in summaries:
        lines.append(
            f"| {d.get('map_name','?')} | {d.get('n_regions','?')} "
            f"| {d.get('max_size','?')} | {d.get('num_over_2x_mean','?')} "
            f"| {d.get('head_size',0)} "
            f"| {_fmt(d.get('head_gold_rate_val',float('nan')))} "
            f"| {_fmt(_v(d,'freq_prior_recall@8'))} "
            f"| {_fmt(_v(d,'candidate_fraction@8'))} "
            f"| {_fmt(_v(d,'recall_per_frac_score@8'))} "
            f"| {_fmt(_v(d,'freq_prior_recall@16'))} "
            f"| {_fmt(_v(d,'candidate_fraction@16'))} |")

    lines += ["", "## Recall at Fixed Candidate Fractions", ""]
    frac_cols = ["recall_at_frac_10pct", "recall_at_frac_20pct",
                 "recall_at_frac_30pct", "recall_at_frac_40pct", "recall_at_frac_50pct"]
    lines.append("| map | @10% | @20% | @30% | @40% | @50% |")
    lines.append("|---|---|---|---|---|---|")
    for d in summaries:
        lines.append("| " + d.get("map_name","?") + " | " +
                     " | ".join(_fmt(_v(d, fc)) for fc in frac_cols) + " |")

    lines += ["", "## Oracle Coverage Stats", ""]
    lines.append("| map | oracle_avg_cand_sz | oracle_p90_cand_sz | oracle_avg_frac | worst_region |")
    lines.append("|---|---|---|---|---|")
    for d in summaries:
        lines.append(f"| {d.get('map_name','?')} "
                     f"| {_fmt(_v(d,'oracle_avg_cand_size'))} "
                     f"| {_fmt(_v(d,'oracle_p90_cand_size'))} "
                     f"| {_fmt(_v(d,'oracle_avg_cand_frac'))} "
                     f"| {_v(d,'oracle_worst_region_size')} |")

    lines += ["", "## Best Map at Fixed Fractions", ""]
    for frac, (mn, rv) in best_at.items():
        lines.append(f"- **@{frac}% candidate fraction:** {mn}  freq_prior_recall={rv}")

    lines += ["", "## Q&A", ""]
    def _q(n, q, ans, detail=""):
        lines.append(f"### Q{n}: {q}"); lines.append(f"**{ans}**")
        if detail: lines.append(f"\n{detail}"); lines.append("")

    _q(1, "Which map has lowest max region size?",
       str(min(summaries, key=lambda d: _v(d,"max_size") or 1e9).get("map_name","?")))
    _q(2, "Which map has best recall@8 at fraction <= 0.20?",
       str(best_at.get(20, ("?","?"))[0]))
    _q(3, "Does V2C head improve effective coverage?",
       "V2C adds head coverage (gold_in_head rate) on top of routed coverage. "
       "See head_gold_rate_val column.")
    _q(4, "Which map to use for router training?",
       str(best_at.get(20, ("?","?"))[0]) + " (best recall@20% fraction)",
       "Train router and re-evaluate with audit_static_region_router_outputs.py")

    # Recommendation
    best_map_20 = best_at.get(20, ("?", 0.0))
    rec_candidates = {d.get("map_name","?"): _v(d,"recall_at_frac_20pct") for d in summaries}
    best_rec_name = max(rec_candidates, key=lambda k: rec_candidates[k] if rec_candidates[k] == rec_candidates[k] else -1)

    lines += ["", "---", f"## RECOMMENDED MAP FOR ROUTER TRAINING: {best_rec_name}", ""]

    rpt_path = os.path.join(out_dir, "map_eval_report.md")
    _write(rpt_path, "\n".join(lines))
    return best_rec_name


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 1B: Evaluate Compute-Aware Region Maps")
    p.add_argument("--maps_root",    required=True)
    p.add_argument("--train_dir",    required=True)
    p.add_argument("--val_dir",      required=True)
    p.add_argument("--old_map",      default=None)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--ks",           type=str, default="1,2,4,8,16,32,64")
    p.add_argument("--max_val_rows", type=int, default=None)
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ks = [int(k) for k in args.ks.split(",")]
    t0 = time.time()

    # ── Load gold tokens ──────────────────────────────────────────────────────
    print("[step 1] Loading gold tokens...")
    train_gold = load_gold_tokens(args.train_dir, args.max_train_rows, "train")
    val_gold   = load_gold_tokens(args.val_dir,   args.max_val_rows,   "val")

    # ── Discover maps ─────────────────────────────────────────────────────────
    print("[step 2] Discovering maps...")
    map_dirs = sorted([
        d for d in glob.glob(os.path.join(args.maps_root, "V2*"))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "token_to_region.json"))
    ])
    if not map_dirs:
        print(f"[WARN] No V2* map directories found in {args.maps_root}")
    print(f"  Found {len(map_dirs)} maps: {[os.path.basename(d) for d in map_dirs]}")

    # ── Load and evaluate each map ────────────────────────────────────────────
    summaries: List[dict] = []
    freq_curves: Dict[str, List[dict]] = {}
    all_freq_rows: List[dict] = []

    print("[step 3] Evaluating maps...")
    for md in map_dirs:
        minfo, _ = load_map(md)
        if minfo is None:
            print(f"  [SKIP] {md}"); continue
        mname = minfo["map_name"]
        print(f"  [{mname}]  K={minfo['K']}  vocab_known={minfo['vocab_known']:,}")
        s, fc = eval_one_map(minfo, train_gold, val_gold, ks)
        summaries.append(s); freq_curves[mname] = fc
        for row in fc:
            all_freq_rows.append({"map_name": mname, **row})
        r8  = s.get("freq_prior_recall@8",  float("nan"))
        r16 = s.get("freq_prior_recall@16", float("nan"))
        f8  = s.get("candidate_fraction@8", float("nan"))
        sc8 = s.get("recall_per_frac_score@8", float("nan"))
        print(f"    recall@8={_fmt(r8)}  recall@16={_fmt(r16)}  "
              f"frac@8={_fmt(f8)}  score@8={_fmt(sc8)}")

    # ── Add old map ───────────────────────────────────────────────────────────
    if args.old_map:
        old_info = load_old_map_info(args.old_map)
        if old_info:
            print(f"  [OLD_K128]  K={old_info['K']}  vocab_known={old_info['vocab_known']:,}")
            s, fc = eval_one_map(old_info, train_gold, val_gold, ks)
            summaries.append(s); freq_curves["OLD_K128"] = fc
            for row in fc:
                all_freq_rows.append({"map_name": "OLD_K128", **row})
            print(f"    recall@8={_fmt(s.get('freq_prior_recall@8'))}  "
                  f"frac@8={_fmt(s.get('candidate_fraction@8'))}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    _wcsv(os.path.join(args.output_dir, "map_eval_summary.csv"), summaries)
    _wcsv(os.path.join(args.output_dir, "freq_prior_coverage_curves.csv"), all_freq_rows)

    # ── Write report ──────────────────────────────────────────────────────────
    best_map = write_eval_report(summaries, freq_curves, ks, args.output_dir, args)

    # ── Console summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f" PHASE 1B MAP EVALUATION  ({elapsed/60:.1f} min)")
    print(f"{'='*65}")
    print(f"  {'map':<32}  {'recall@8':>9}  {'frac@8':>7}  {'score@8':>8}  "
          f"{'recall@16':>10}  {'frac@16':>7}")
    print(f"  {'─'*65}")
    for d in summaries:
        print(
            f"  {d.get('map_name','?'):<32}"
            f"  {_fmt(d.get('freq_prior_recall@8', float('nan'))):>9}"
            f"  {_fmt(d.get('candidate_fraction@8', float('nan'))):>7}"
            f"  {_fmt(d.get('recall_per_frac_score@8', float('nan'))):>8}"
            f"  {_fmt(d.get('freq_prior_recall@16', float('nan'))):>10}"
            f"  {_fmt(d.get('candidate_fraction@16', float('nan'))):>7}")
    print(f"\n  Recommended for router training: {best_map}")
    print(f"  Report: {args.output_dir}/map_eval_report.md")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
