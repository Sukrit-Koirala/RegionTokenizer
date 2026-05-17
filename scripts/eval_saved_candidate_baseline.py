#!/usr/bin/env python3
"""
Architecture-independent baseline evaluator for the saved path-refiner candidate dataset.

Computes force-zero NLL directly from shard tensors:
    scores[b, c] = h_prime[b] · tok_emb[cand_tok[b, c]]
    covered_nll  = cross_entropy(scores[covered], gold_cand_idx[covered])

Does NOT instantiate any refiner model, router, or region transformer.
Loads only the token embedding matrix from the backbone checkpoint.

This is the OFFICIAL reference baseline.  All refiner variants must reproduce
this number exactly (within 1e-4) when evaluated with force_zero_delta=True
and the same dataset fingerprint.

Outputs:
    <output_dir>/saved_candidate_baseline.json    ← official numbers + fingerprint
    <output_dir>/saved_candidate_baseline.md
    <output_dir>/saved_candidate_baseline_by_split.csv

Usage:
    python scripts/eval_saved_candidate_baseline.py \\
        --val_dir   runs/path_refiner_clean/data/val_hgrid_K24 \\
        --small_ckpt runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --output_dir runs/path_refiner_clean/baselines \\
        --device cuda
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SPLIT_NAMES = {0: "core", 1: "medium", 2: "boundary", 3: "tight"}
TYPE_NAMES  = {0: "other", 1: "A", 2: "B", 3: "C"}


# ── Token embedding loader ────────────────────────────────────────────────────

def load_token_embedding(ckpt_path: str, device) -> torch.Tensor:
    """
    Extract tok_emb weight from a backbone checkpoint without instantiating the model.
    Returns float32 tensor of shape (vocab_size, d_model).
    """
    raw   = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = raw.get("model_state_dict", raw.get("model", raw))

    candidate_keys = [
        "token_emb.weight",
        "transformer.wte.weight",
        "backbone.token_emb.weight",
        "backbone.transformer.wte.weight",
        "wte.weight",
    ]
    for key in candidate_keys:
        if key in state:
            print(f"[baseline] found tok_emb at key '{key}'  shape={tuple(state[key].shape)}")
            return state[key].float().to(device)

    # Fallback: search by suffix
    for key, val in state.items():
        if key.endswith("wte.weight") or key.endswith("token_emb.weight"):
            print(f"[baseline] found tok_emb at key '{key}'  shape={tuple(val.shape)}")
            return val.float().to(device)

    raise RuntimeError(
        f"Cannot find token embedding in checkpoint {ckpt_path}.\n"
        f"Available keys (first 20): {list(state.keys())[:20]}"
    )


# ── Core evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(val_dir: str, tok_emb_w: torch.Tensor,
             eval_batch_size: int, device) -> Tuple[Dict, str]:
    """
    Stream through all shards and compute force-zero metrics + fingerprint.
    Evaluates ALL examples — no max_batches limit.
    """
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt files in {val_dir}")
    print(f"[baseline] {len(paths)} shards  eval_batch_size={eval_batch_size}")

    emb_w = tok_emb_w.float().to(device)

    # Accumulators: {(split_id, type_id): [ce_sum, n_cov, n_total]}
    stats: Dict = defaultdict(lambda: [0.0, 0, 0])

    # Top-k accuracy
    acc1_sum = 0
    acc5_sum = 0

    # Fingerprint
    total_n          = 0
    total_cov        = 0
    sum_cand_counts  = 0
    sum_gold_idx_cov = 0
    sum_gold_tok     = 0

    t0 = time.time()
    for pi, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu", weights_only=True)
        N         = len(shard["covered"])
        has_type  = "type_arr" in shard
        split_arr = shard["split"].long()
        type_arr  = shard["type_arr"].long() if has_type else torch.zeros(N, dtype=torch.long)
        gold_tok  = shard["gold_token"].long() if "gold_token" in shard else None

        for start in range(0, N, eval_batch_size):
            end     = min(start + eval_batch_size, N)
            h       = shard["h_prime"][start:end].float().to(device)      # (B, d_model)
            ct      = shard["cand_tok"][start:end].long().to(device)      # (B, C)
            g_idx   = shard["gold_cand_idx"][start:end].long().to(device) # (B,)
            covered = shard["covered"][start:end].bool().to(device)       # (B,)
            sp      = split_arr[start:end]
            tp      = type_arr[start:end]
            B, C    = ct.shape

            cmask  = (ct >= 0)                                            # (B, C)
            ct_c   = ct.clamp(min=0)
            tok_e  = F.embedding(ct_c, emb_w)                            # (B, C, d_model)
            scores = (h.unsqueeze(1) * tok_e).sum(-1)                    # (B, C)
            scores = scores.masked_fill(~cmask, float("-inf"))

            cov = covered
            n_cov_b = int(cov.sum())

            if n_cov_b > 0:
                sc_cov = scores[cov]                                      # (n_cov, C)
                gi_cov = g_idx[cov]                                       # (n_cov,)

                lp      = F.log_softmax(sc_cov, dim=-1)
                ce_vals = -lp[torch.arange(n_cov_b, device=device), gi_cov].cpu()

                # acc@k
                topk5 = torch.topk(sc_cov, min(5, C), dim=-1).indices    # (n_cov, k)
                for k in range(n_cov_b):
                    gold = int(gi_cov[k])
                    top5 = topk5[k].tolist()
                    acc1_sum += int(top5[0] == gold)
                    acc5_sum += int(gold in top5)

                cov_idx = torch.where(cov)[0].cpu()
                for j, i in enumerate(cov_idx.tolist()):
                    key = (int(sp[i].item()), int(tp[i].item()))
                    stats[key][0] += float(ce_vals[j])
                    stats[key][1] += 1

            for i in range(B):
                key = (int(sp[i].item()), int(tp[i].item()))
                stats[key][2] += 1

            # Fingerprint
            total_n          += B
            total_cov        += n_cov_b
            sum_cand_counts  += int(cmask.sum())
            if n_cov_b > 0:
                sum_gold_idx_cov += int(g_idx[cov].sum())
            if gold_tok is not None:
                sum_gold_tok += int(gold_tok[start:end].sum())

        if (pi + 1) % 10 == 0 or (pi + 1) == len(paths):
            cov_so_far = total_cov / max(total_n, 1)
            print(f"  shard {pi+1:4d}/{len(paths)}  n={total_n:>9,}  cov={cov_so_far:.4f}"
                  f"  t={time.time()-t0:.0f}s")

    # ── Aggregate ─────────────────────────────────────────────────────────────

    total_ce  = sum(s[0] for s in stats.values())
    total_cov_agg = sum(s[1] for s in stats.values())
    total_n_agg   = sum(s[2] for s in stats.values())

    covered_nll = total_ce  / max(total_cov_agg, 1)
    coverage    = total_cov_agg / max(total_n_agg, 1)

    results: Dict = {
        "num_examples":        total_n,
        "num_covered":         total_cov,
        "coverage":            coverage,
        "fallback_rate":       1.0 - coverage,
        "mean_cand_count":     sum_cand_counts / max(total_n, 1),
        "covered_nll":         covered_nll,
        "covered_ppl":         math.exp(min(covered_nll, 30.0)),
        "acc_at_1":            acc1_sum / max(total_cov, 1),
        "acc_at_5":            acc5_sum / max(total_cov, 1),
    }

    by_split: Dict = {}
    for sid, sname in SPLIT_NAMES.items():
        sub = [s for k, s in stats.items() if k[0] == sid]
        cn  = sum(s[1] for s in sub)
        tn  = sum(s[2] for s in sub)
        cce = sum(s[0] for s in sub)
        by_split[sname] = {
            "coverage":    cn / max(tn, 1),
            "covered_nll": cce / max(cn, 1),
            "num_covered": cn,
            "num_total":   tn,
        }
        results[f"cov_{sname}"]      = cn / max(tn, 1)
        results[f"cnll_{sname}"]     = cce / max(cn, 1)
        results[f"fallback_{sname}"] = 1.0 - results[f"cov_{sname}"]

    by_type: Dict = {}
    for tid, tname in TYPE_NAMES.items():
        sub = [s for k, s in stats.items() if k[1] == tid]
        cn  = sum(s[1] for s in sub)
        tn  = sum(s[2] for s in sub)
        cce = sum(s[0] for s in sub)
        if tn > 0:
            by_type[tname] = {
                "coverage":    cn / max(tn, 1),
                "covered_nll": cce / max(cn, 1),
                "num_covered": cn,
                "num_total":   tn,
            }

    # ── Fingerprint ───────────────────────────────────────────────────────────

    fp_data = {
        "num_shards":        len(paths),
        "total_n":           total_n,
        "total_cov":         total_cov,
        "sum_cand_counts":   sum_cand_counts,
        "sum_gold_idx_cov":  sum_gold_idx_cov,
        "sum_gold_tok":      sum_gold_tok,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fp_data, sort_keys=True).encode()
    ).hexdigest()[:16]
    results["dataset_fingerprint"] = fingerprint

    return results, fingerprint, by_split, by_type


# ── Writers ───────────────────────────────────────────────────────────────────

def write_json(results: Dict, fingerprint: str, out_dir: str):
    payload = dict(results)
    payload["dataset_fingerprint"] = fingerprint
    path = os.path.join(out_dir, "saved_candidate_baseline.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[baseline] JSON → {path}")
    return path


def write_md(results: Dict, fingerprint: str, by_split: Dict, by_type: Dict,
             val_dir: str, out_dir: str):
    lines = [
        "# Official Saved-Candidate Baseline",
        "",
        f"Dataset: `{val_dir}`",
        f"Fingerprint: `{fingerprint}`",
        "",
        "## Global metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| num_examples   | {results['num_examples']:,} |",
        f"| num_covered    | {results['num_covered']:,} |",
        f"| coverage       | {results['coverage']:.6f} |",
        f"| fallback_rate  | {results['fallback_rate']:.6f} |",
        f"| mean_cand_count| {results['mean_cand_count']:.1f} |",
        f"| covered_nll    | {results['covered_nll']:.6f} |",
        f"| covered_ppl    | {results['covered_ppl']:.4f} |",
        f"| acc@1          | {results['acc_at_1']:.4f} |",
        f"| acc@5          | {results['acc_at_5']:.4f} |",
        "",
        "## Per-split",
        "",
        "| split | coverage | covered_nll | num_covered | num_total |",
        "|-------|----------|-------------|-------------|-----------|",
    ]
    for sname, m in by_split.items():
        lines.append(
            f"| {sname} | {m['coverage']:.4f} | {m['covered_nll']:.4f} "
            f"| {m['num_covered']:,} | {m['num_total']:,} |"
        )
    if by_type:
        lines += [
            "",
            "## Per-type",
            "",
            "| type | coverage | covered_nll | num_covered | num_total |",
            "|------|----------|-------------|-------------|-----------|",
        ]
        for tname, m in by_type.items():
            lines.append(
                f"| {tname} | {m['coverage']:.4f} | {m['covered_nll']:.4f} "
                f"| {m['num_covered']:,} | {m['num_total']:,} |"
            )
    lines += [
        "",
        "## Consistency check target",
        "",
        "Every refiner variant evaluated with `force_zero_delta=True` on this",
        "dataset must reproduce:",
        "",
        f"```",
        f"covered_nll         = {results['covered_nll']:.6f}  (tol 1e-4)",
        f"coverage            = {results['coverage']:.6f}  (tol 1e-6)",
        f"dataset_fingerprint = {fingerprint}",
        f"```",
    ]
    path = os.path.join(out_dir, "saved_candidate_baseline.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[baseline] MD  → {path}")


def write_csv(by_split: Dict, by_type: Dict, out_dir: str):
    path = os.path.join(out_dir, "saved_candidate_baseline_by_split.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group", "name", "coverage", "covered_nll", "num_covered", "num_total"
        ])
        writer.writeheader()
        for sname, m in by_split.items():
            writer.writerow({"group": "split", "name": sname, **m})
        for tname, m in by_type.items():
            writer.writerow({"group": "type", "name": tname, **m})
    print(f"[baseline] CSV → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[baseline] loading tok_emb from: {args.small_ckpt}")
    tok_emb_w = load_token_embedding(args.small_ckpt, device)
    print(f"[baseline] tok_emb shape: {tuple(tok_emb_w.shape)}")

    print(f"[baseline] evaluating: {args.val_dir}")
    results, fingerprint, by_split, by_type = evaluate(
        args.val_dir, tok_emb_w, args.eval_batch_size, device
    )

    print()
    print("=" * 56)
    print("OFFICIAL SAVED-CANDIDATE BASELINE")
    print("=" * 56)
    print(f"  fingerprint  : {fingerprint}")
    print(f"  num_examples : {results['num_examples']:,}")
    print(f"  num_covered  : {results['num_covered']:,}")
    print(f"  coverage     : {results['coverage']:.6f}")
    print(f"  covered_nll  : {results['covered_nll']:.6f}")
    print(f"  covered_ppl  : {results['covered_ppl']:.4f}")
    print(f"  acc@1        : {results['acc_at_1']:.4f}")
    print(f"  acc@5        : {results['acc_at_5']:.4f}")
    print()

    write_json(results, fingerprint, args.output_dir)
    write_md(results, fingerprint, by_split, by_type, args.val_dir, args.output_dir)
    write_csv(by_split, by_type, args.output_dir)

    print(f"\n[baseline] done.  outputs → {args.output_dir}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir",       required=True,
                   help="Directory containing shard_*.pt files (val split).")
    p.add_argument("--small_ckpt",    required=True,
                   help="Backbone checkpoint (used only to extract token embedding).")
    p.add_argument("--output_dir",    required=True,
                   help="Where to write baseline JSON/MD/CSV.")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
