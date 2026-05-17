#!/usr/bin/env python3
"""
Audit a pre-built path-refiner shard dataset produced by
build_clean_path_refiner_dataset.py.

Checks:
  1. Coverage / fallback rate (overall and per split/type)
  2. Candidate count distribution (mean, median, p90, p95, p99, max)
  3. gold_cand_idx consistency:
       covered[i] <=> gold_cand_idx[i] >= 0
  4. Gold token at declared index:
       covered[i] => cand_tok[i, gold_cand_idx[i]] == gold_token[i]
  5. (Optional) comparison to masked-softmax eval CSV — PASS/FAIL on
     coverage delta > --tol

Usage:
    python scripts/audit_path_refiner_dataset.py \
        --shard_dir runs/path_refiner_clean/data/val_hgrid_K24

    # With comparison to masked-softmax baseline:
    python scripts/audit_path_refiner_dataset.py \
        --shard_dir runs/path_refiner_clean/data/val_hgrid_K24 \
        --masked_eval_csv runs/path_refiner_clean/eval/masked_softmax_eval.csv \
        --policy hgrid_K24_srccombined_nts1_bkr12_fb24_cm0.15_en1.25_corec60m30 \
        --tol 0.01
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

SPLIT_NAMES = {0: "core", 1: "medium", 2: "boundary", 3: "tight"}
TYPE_NAMES  = {0: "other", 1: "A", 2: "B", 3: "C"}


def load_shards(shard_dir: str):
    paths = sorted(
        p for p in (
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        )
    )
    if not paths:
        raise RuntimeError(f"No shard_*.pt files found in {shard_dir}")
    return paths


def audit(args):
    shard_dir = args.shard_dir
    paths = load_shards(shard_dir)
    print(f"[audit] {len(paths)} shards in {shard_dir}")

    # Accumulators
    total_n        = 0
    total_cov      = 0
    cand_counts    = []    # list of arrays: #valid candidates per position

    # Per-split and per-type coverage
    split_cov  = {k: [0, 0] for k in SPLIT_NAMES}   # [covered, total]
    type_cov   = {k: [0, 0] for k in TYPE_NAMES}

    # Error tracking
    idx_flag_mismatches  = 0   # covered flag disagrees with gold_cand_idx sign
    gold_tok_mismatches  = 0   # gold token at declared index doesn't match
    gold_tok_checked     = 0

    for pi, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu", weights_only=True)

        cov       = shard["covered"].bool()             # (N,)
        g_idx     = shard["gold_cand_idx"].long()       # (N,)
        g_tok     = shard["gold_token"].long()          # (N,)
        cand_tok  = shard["cand_tok"].long()            # (N, C)
        split_arr = shard["split"].long()               # (N,)
        has_type  = "type_arr" in shard
        type_arr  = shard["type_arr"].long() if has_type else torch.zeros(len(cov), dtype=torch.long)

        N = len(cov)
        total_n   += N
        total_cov += int(cov.sum())

        # Candidate counts: number of non-pad (>= 0) entries per row
        n_cands = (cand_tok >= 0).sum(dim=1).numpy()   # (N,)
        cand_counts.append(n_cands)

        # gold_cand_idx consistency: covered <=> g_idx >= 0
        flag_from_idx = (g_idx >= 0)
        mismatch = (cov != flag_from_idx)
        idx_flag_mismatches += int(mismatch.sum())

        # Gold token at declared index
        cov_pos = torch.where(cov)[0]
        for i in cov_pos:
            ci   = int(g_idx[i])
            if ci < 0 or ci >= cand_tok.shape[1]:
                gold_tok_mismatches += 1
            else:
                gold_tok_checked += 1
                if int(cand_tok[i, ci]) != int(g_tok[i]):
                    gold_tok_mismatches += 1

        # Per-split and per-type coverage
        for s_id in SPLIT_NAMES:
            mask = (split_arr == s_id)
            split_cov[s_id][0] += int((cov & mask).sum())
            split_cov[s_id][1] += int(mask.sum())
        for t_id in TYPE_NAMES:
            mask = (type_arr == t_id)
            type_cov[t_id][0] += int((cov & mask).sum())
            type_cov[t_id][1] += int(mask.sum())

        if (pi + 1) % 10 == 0 or (pi + 1) == len(paths):
            running_cov = total_cov / max(total_n, 1)
            print(f"  shard {pi+1:4d}/{len(paths)}  n={total_n:>9,}  cov={running_cov:.4f}")

    # ── Aggregate stats ───────────────────────────────────────────────────────

    all_counts = np.concatenate(cand_counts)
    coverage   = total_cov / max(total_n, 1)

    print()
    print("=" * 60)
    print(f"AUDIT RESULTS: {shard_dir}")
    print("=" * 60)
    print()

    print(f"Positions total  : {total_n:,}")
    print(f"Covered          : {total_cov:,}  ({coverage:.4f})")
    print(f"Fallback (uncov) : {total_n - total_cov:,}  ({1-coverage:.4f})")
    print()

    print("Candidate count distribution (over all positions):")
    print(f"  mean   = {all_counts.mean():.1f}")
    print(f"  median = {np.median(all_counts):.1f}")
    print(f"  p90    = {np.percentile(all_counts, 90):.1f}")
    print(f"  p95    = {np.percentile(all_counts, 95):.1f}")
    print(f"  p99    = {np.percentile(all_counts, 99):.1f}")
    print(f"  max    = {all_counts.max()}")
    print(f"  zero   = {int((all_counts == 0).sum())}  (positions with no candidates — should be 0)")
    print()

    print("Per-split coverage:")
    for s_id, s_name in SPLIT_NAMES.items():
        c, t = split_cov[s_id]
        print(f"  {s_name:>10}: {c:>8,}/{t:>8,}  cov={c/max(t,1):.4f}")
    print()

    has_types = any(type_cov[t][1] > 0 for t in (1, 2, 3))
    if has_types:
        print("Per-type coverage:")
        for t_id, t_name in TYPE_NAMES.items():
            c, t = type_cov[t_id]
            if t > 0:
                print(f"  type {t_name}: {c:>8,}/{t:>8,}  cov={c/max(t,1):.4f}")
        print()

    print("Consistency checks:")
    if idx_flag_mismatches == 0:
        print(f"  gold_cand_idx vs covered flag  : PASS  (0 mismatches)")
    else:
        print(f"  gold_cand_idx vs covered flag  : FAIL  ({idx_flag_mismatches} mismatches)")
    if gold_tok_mismatches == 0:
        print(f"  gold token at gold_cand_idx    : PASS  (checked {gold_tok_checked:,} positions)")
    else:
        print(f"  gold token at gold_cand_idx    : FAIL  ({gold_tok_mismatches} bad out of {gold_tok_checked:,} checked)")
    print()

    # ── Optional comparison to masked-softmax CSV ─────────────────────────────

    if args.masked_eval_csv:
        if not os.path.isfile(args.masked_eval_csv):
            print(f"WARNING: masked_eval_csv not found: {args.masked_eval_csv}")
        else:
            ref_row = None
            with open(args.masked_eval_csv) as f:
                for row in csv.DictReader(f):
                    if row.get("policy", "").strip() == args.policy.strip():
                        ref_row = row
                        break
            if ref_row is None:
                print(f"WARNING: policy {args.policy!r} not found in {args.masked_eval_csv}")
                print("         Available policies:")
                with open(args.masked_eval_csv) as f:
                    for row in csv.DictReader(f):
                        print(f"           {row.get('policy','?')!r}")
            else:
                ref_cov = float(ref_row["coverage"])
                delta   = abs(coverage - ref_cov)
                tol     = args.tol
                result  = "PASS" if delta <= tol else "FAIL"
                print(f"Comparison to masked-softmax eval ({args.policy}):")
                print(f"  Dataset coverage   : {coverage:.4f}")
                print(f"  Masked-softmax cov : {ref_cov:.4f}")
                print(f"  Delta              : {delta:.4f}  (tol={tol})")
                print(f"  Coverage check     : {result}")
                if "covered_nll" in ref_row:
                    print(f"  Masked-softmax covered_nll : {float(ref_row['covered_nll']):.4f}")
                if "mixed_nll" in ref_row:
                    print(f"  Masked-softmax mixed_nll   : {float(ref_row['mixed_nll']):.4f}")
                print()
                if result == "FAIL":
                    print("  ACTION: coverage mismatch exceeds tolerance.")
                    print("  Check --candidate_cap (mean_cands above should be < cap).")
                    print("  If mean_cands >= cap, rebuild with a larger --candidate_cap.")

    # ── Dataset config ────────────────────────────────────────────────────────

    cfg_path = os.path.join(shard_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
        print("dataset_config.json:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        cap = cfg.get("candidate_cap", None)
        if cap is not None:
            mean_c = all_counts.mean()
            if mean_c >= cap * 0.9:
                print(f"\n  WARNING: mean_cands ({mean_c:.0f}) is >= 90% of candidate_cap ({cap}).")
                print(f"  Many positions are likely being truncated. Consider raising cap.")
            else:
                print(f"\n  candidate_cap={cap} looks OK (mean_cands={mean_c:.0f})")
    else:
        print("(no dataset_config.json found)")

    return {
        "total_n":              total_n,
        "total_cov":            total_cov,
        "coverage":             coverage,
        "idx_flag_mismatches":  idx_flag_mismatches,
        "gold_tok_mismatches":  gold_tok_mismatches,
        "mean_cands":           float(all_counts.mean()),
        "max_cands":            int(all_counts.max()),
    }


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir",       required=True,
                   help="Directory containing shard_*.pt files to audit.")
    p.add_argument("--masked_eval_csv", default=None,
                   help="Path to masked_softmax_eval.csv for coverage comparison.")
    p.add_argument("--policy",          default=None,
                   help="Policy name to look up in masked_eval_csv.")
    p.add_argument("--tol",             type=float, default=0.01,
                   help="Maximum allowed coverage delta vs masked-softmax eval (default 0.01).")
    return p.parse_args()


if __name__ == "__main__":
    audit(_parse())
