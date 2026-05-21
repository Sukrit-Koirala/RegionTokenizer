#!/usr/bin/env python3
"""
Stage 04 — Audit Context + Retrieval Data.

Verifies the new live-context + retrieval data is clean and useful.
Runs frozen backbone on val input_ids to compute live base_lgt, then checks:
  gold_in_base_top{K}       fraction of positions where gold is in base top-K
  gold_in_neighbors         fraction where gold is among retrieved neighbors
  retrieval_added_gold      fraction where neighbors add gold not in top-K
  all metrics repeated for boundary positions only

Hard pass conditions:
  gold_in_base_top256 > 0.40        (sanity: model is reasonable)
  neighbor_score_finite = True      (no NaN/inf)
  neighbor_tokens_valid = True      (no out-of-range tokens)
  audit_pass = True
"""

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe, get_hs_small
from scripts.train_hard_position_refiner import compute_filter_mask


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[stage04] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().float()  # (V, d)
    else:
        raise RuntimeError("Cannot locate token_emb in backbone")

    K = args.top_k
    N_ret = args.num_neighbors
    filter_kwargs = {"margin_thresh": 0.1, "entropy_thresh": 2.0}

    # ── Collect val shards ────────────────────────────────────────────────────
    ctx_paths  = sorted([os.path.join(args.val_ctx_dir,       f) for f in
                         os.listdir(args.val_ctx_dir)
                         if f.startswith("shard_") and f.endswith(".pt")])
    cand_paths = sorted([os.path.join(args.val_cand_dir,      f) for f in
                         os.listdir(args.val_cand_dir)
                         if f.startswith("shard_") and f.endswith(".pt")])
    ret_paths  = sorted([os.path.join(args.val_retrieval_dir, f) for f in
                         os.listdir(args.val_retrieval_dir)
                         if f.startswith("shard_") and f.endswith(".pt")])

    if not (len(ctx_paths) == len(cand_paths) == len(ret_paths)):
        raise RuntimeError(
            f"Shard count mismatch: ctx={len(ctx_paths)} "
            f"cand={len(cand_paths)} ret={len(ret_paths)}"
        )

    # ── Accumulators ─────────────────────────────────────────────────────────
    acc: Dict[str, float] = {
        "gold_in_topK_all": 0.0,      "n_all": 0.0,
        "gold_in_nbr_all":  0.0,
        "ret_added_all":    0.0,
        "gold_in_topK_bnd": 0.0,      "n_bnd": 0.0,
        "gold_in_nbr_bnd":  0.0,
        "ret_added_bnd":    0.0,
        "nbr_score_finite": 1.0,
        "nbr_tok_valid":    1.0,
    }

    t0 = time.time()

    for si, (ctx_p, cand_p, ret_p) in enumerate(zip(ctx_paths, cand_paths, ret_paths)):
        ctx_d  = torch.load(ctx_p,  map_location="cpu", weights_only=True)
        cand_d = torch.load(cand_p, map_location="cpu", weights_only=True)
        ret_d  = torch.load(ret_p,  map_location="cpu", weights_only=True)

        # Alignment check
        if not torch.equal(ctx_d["gold_token"].int(), cand_d["gold_token"].int()):
            raise RuntimeError(f"ctx/cand gold_token mismatch shard {si:05d}")
        if not torch.equal(ctx_d["gold_token"].int(), ret_d["gold_token"].int()):
            raise RuntimeError(f"ctx/ret gold_token mismatch shard {si:05d}")

        N = ctx_d["gold_token"].shape[0]
        boundary_mask = compute_filter_mask(cand_d, "boundary", **filter_kwargs)  # (N,) bool

        # ── Neighbor sanity ───────────────────────────────────────────────────
        nbr_scores = ret_d["neighbor_scores"].float()   # (N, N_ret)
        nbr_toks   = ret_d["neighbor_gold_tokens"].long()  # (N, N_ret)
        nbr_valid  = (nbr_toks >= 0)                    # (N, N_ret)

        if not torch.isfinite(nbr_scores[nbr_valid]).all():
            acc["nbr_score_finite"] = 0.0
            print(f"  WARNING: shard {si:05d}: non-finite neighbor scores")

        if nbr_valid.any() and (nbr_toks[nbr_valid] >= vocab_size).any():
            acc["nbr_tok_valid"] = 0.0
            print(f"  WARNING: shard {si:05d}: out-of-range neighbor tokens")

        # ── Live base logits ──────────────────────────────────────────────────
        for start in range(0, N, args.batch_size):
            end  = min(start + args.batch_size, N)
            sl   = slice(start, end)
            B    = end - start

            ids  = ctx_d["input_ids"][sl].long().to(device)         # (B, ctx_len)
            gt   = cand_d["gold_token"][sl].long().to(device)       # (B,)
            nbr_t_b = nbr_toks[sl].to(device)                       # (B, N_ret)
            bnd_b = boundary_mask[sl].to(device)                    # (B,) bool

            with torch.no_grad():
                h_all  = get_hs_small(backbone, ids, device)        # (B, T, d)
            h_ctx      = h_all[:, -1, :].float()                    # (B, d)
            base_lgt   = h_ctx @ tok_emb_w.to(device).T             # (B, V)

            topk_ids   = base_lgt.topk(K, dim=1).indices            # (B, K)

            # gold_in_topK
            gt_col = gt.unsqueeze(1)                                 # (B, 1)
            in_topK = (topk_ids == gt_col).any(1)                   # (B,)

            # gold_in_neighbors
            in_nbr  = (nbr_t_b == gt_col).any(1)                   # (B,) (checks all nbrs)

            # retrieval_added_gold: gold in neighbors but NOT in topK
            ret_added = in_nbr & ~in_topK

            # Accumulate
            acc["gold_in_topK_all"] += float(in_topK.float().sum())
            acc["gold_in_nbr_all"]  += float(in_nbr.float().sum())
            acc["ret_added_all"]    += float(ret_added.float().sum())
            acc["n_all"]            += B

            if bnd_b.any():
                acc["gold_in_topK_bnd"] += float(in_topK[bnd_b].float().sum())
                acc["gold_in_nbr_bnd"]  += float(in_nbr[bnd_b].float().sum())
                acc["ret_added_bnd"]    += float(ret_added[bnd_b].float().sum())
                acc["n_bnd"]            += float(bnd_b.float().sum())

        if si % 5 == 0 or si == len(ctx_paths) - 1:
            print(f"  shard {si:05d}/{len(ctx_paths)-1}  "
                  f"t={time.time()-t0:.0f}s")

    # ── Compute rates ─────────────────────────────────────────────────────────
    def _rate(num, den):
        return float(num) / max(float(den), 1)

    n_all = acc["n_all"]
    n_bnd = acc["n_bnd"]

    results = {
        "n_all":                   n_all,
        "n_boundary":              n_bnd,
        "gold_in_base_topK_all":   _rate(acc["gold_in_topK_all"], n_all),
        "gold_in_neighbors_all":   _rate(acc["gold_in_nbr_all"],  n_all),
        "retrieval_added_gold_all":_rate(acc["ret_added_all"],    n_all),
        "gold_in_base_topK_bnd":   _rate(acc["gold_in_topK_bnd"], n_bnd),
        "gold_in_neighbors_bnd":   _rate(acc["gold_in_nbr_bnd"],  n_bnd),
        "retrieval_added_gold_bnd":_rate(acc["ret_added_bnd"],    n_bnd),
        "top_k":                   K,
        "num_neighbors":           N_ret,
        "neighbor_scores_finite":  bool(acc["nbr_score_finite"]),
        "neighbor_tokens_valid":   bool(acc["nbr_tok_valid"]),
    }

    # ── Pass/fail ─────────────────────────────────────────────────────────────
    audit_pass = (
        results["gold_in_base_topK_all"]    > 0.40 and
        results["neighbor_scores_finite"]          and
        results["neighbor_tokens_valid"]
    )
    results["audit_pass"] = audit_pass

    os.makedirs(args.output_dir, exist_ok=True)
    rpath = os.path.join(args.output_dir, "retrieval_audit.json")
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[stage04] Results:")
    print(f"  gold_in_base_topK_all     = {results['gold_in_base_topK_all']:.4f}")
    print(f"  gold_in_neighbors_all     = {results['gold_in_neighbors_all']:.4f}")
    print(f"  retrieval_added_gold_all  = {results['retrieval_added_gold_all']:.4f}")
    print(f"  gold_in_base_topK_bnd     = {results['gold_in_base_topK_bnd']:.4f}")
    print(f"  gold_in_neighbors_bnd     = {results['gold_in_neighbors_bnd']:.4f}")
    print(f"  retrieval_added_gold_bnd  = {results['retrieval_added_gold_bnd']:.4f}")
    print(f"  audit_pass                = {audit_pass}")
    print(f"[stage04] Report → {rpath}")

    if not audit_pass:
        raise RuntimeError("Stage 04 audit FAILED. See report.")
    print("[stage04] PASS")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",         required=True)
    p.add_argument("--val_ctx_dir",        required=True)
    p.add_argument("--val_cand_dir",       required=True)
    p.add_argument("--val_retrieval_dir",  required=True)
    p.add_argument("--output_dir",         required=True)
    p.add_argument("--top_k",              type=int, default=256)
    p.add_argument("--num_neighbors",      type=int, default=32)
    p.add_argument("--batch_size",         type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
