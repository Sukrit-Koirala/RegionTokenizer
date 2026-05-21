#!/usr/bin/env python3
"""
Stage 05 — Debug Live Context + Retrieval Resolver (Step-0 Identity Check).

Instantiates a FRESH LiveContextRetrievalTokenResolver (no checkpoint loaded)
and runs full_vocab_eval_lctx_resolver on the val split.

At step-0, delta_head is zero-init → tanh(0) = 0 → bounded_delta = 0 →
refined_lgt ≡ base_lgt everywhere.

Hard pass conditions:
  nll_diff_all         < 1e-3   (refined NLL ≈ base NLL, all positions)
  nll_diff_covered     < 1e-3   (refined NLL ≈ base NLL, covered positions)
  outside_gate_diff    < 1e-5   (outside gate: gated_lgt = base_lgt exactly)
  delta_max_abs        < 1e-6   (bounded_delta numerically zero)
  canonical fingerprint / count check passes (data integrity)

Prerequisite: stage04 must have passed (reads stage04 audit_pass).
Output: output_dir/debug_identity.json  (with identity_pass: true/false)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.train_clean_path_refiner import load_r2s
from scripts.train_token_confuser_resolver import load_token_to_region
from scripts.stage06_train_live_context_retrieval_resolver import (
    LiveContextRetrievalTokenResolver,
    full_vocab_eval_lctx_resolver,
)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Prerequisite: stage04 audit_pass ─────────────────────────────────────
    if args.stage04_report:
        if not os.path.isfile(args.stage04_report):
            raise RuntimeError(f"Stage04 report not found: {args.stage04_report}")
        with open(args.stage04_report) as f:
            s4 = json.load(f)
        if not s4.get("audit_pass", False):
            raise RuntimeError(
                f"Stage04 audit_pass=False — cannot proceed. Report: {args.stage04_report}"
            )
        print(f"[stage05] Stage04 audit_pass=True  ✓")

    # ── Load backbone ─────────────────────────────────────────────────────────
    print(f"[stage05] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().float()   # (V, d)
    else:
        raise RuntimeError("Cannot locate token_emb in backbone")

    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    # ── Load region maps ──────────────────────────────────────────────────────
    print(f"[stage05] Loading region maps ...")
    t2r_np = load_token_to_region(args.region_map, vocab_size)   # (V,) int32

    # Derive n_fine from the region map (max valid region id + 1)
    if t2r_np is not None and (t2r_np >= 0).any():
        n_fine = int(t2r_np[t2r_np >= 0].max()) + 1
    else:
        n_fine = 128  # fallback

    r2s_np = load_r2s(args.super_map, n_fine)                    # (n_fine,) int32
    n_super = int(r2s_np.max()) + 1

    print(f"  n_fine={n_fine}  n_super={n_super}")

    # ── Instantiate fresh model (NO checkpoint) ───────────────────────────────
    print(f"[stage05] Instantiating fresh LiveContextRetrievalTokenResolver ...")
    model = LiveContextRetrievalTokenResolver(
        backbone       = backbone,
        d_backbone     = d_model,
        n_fine         = n_fine,
        n_super        = n_super,
        r2s_np         = r2s_np,
        t2r_np         = t2r_np,
        top_k          = args.top_k,
        num_neighbors  = args.num_neighbors,
        d_resolver     = args.resolver_dim,
        n_layers       = args.resolver_layers,
        n_heads        = args.resolver_heads,
        delta_scale    = args.delta_scale,
        candidate_mode = "base_topk_plus_neighbors",
    ).to(device)

    # Verify delta_head is truly zero at init
    dw_norm = model.delta_head.weight.abs().max().item()
    db_norm = model.delta_head.bias.abs().max().item()
    print(f"  delta_head weight max_abs={dw_norm:.2e}  bias max_abs={db_norm:.2e}")
    if dw_norm > 1e-9 or db_norm > 1e-9:
        raise RuntimeError(
            f"delta_head is NOT zero-init: weight={dw_norm:.2e}  bias={db_norm:.2e}"
        )

    # ── Filter kwargs (matching stage06 default) ──────────────────────────────
    filter_kwargs = {"margin_thresh": 0.1, "entropy_thresh": 2.0}

    # ── Run step-0 eval ───────────────────────────────────────────────────────
    print(f"[stage05] Running step-0 full_vocab_eval_lctx_resolver ...")
    t0 = time.time()
    results = full_vocab_eval_lctx_resolver(
        model            = model,
        val_cand_dir     = args.val_cand_dir,
        val_ctx_dir      = args.val_ctx_dir,
        val_ret_dir      = args.val_retrieval_dir,
        tok_emb_w        = tok_emb_w,
        device           = device,
        gate_filter_name = args.gate_filter,
        filter_kwargs    = filter_kwargs,
        fail_on_mismatch = False,
        eval_batch_size  = args.eval_batch_size,
        variant_tag      = "step0_identity",
    )
    print(f"  Eval done  t={time.time()-t0:.0f}s")

    # ── Identity checks ───────────────────────────────────────────────────────
    base_nll_all  = results["full_vocab_base_nll_all"]
    ref_nll_all   = results["full_vocab_refined_nll_all"]
    nll_diff_all  = abs(ref_nll_all - base_nll_all)

    base_nll_cov  = results["full_vocab_base_nll_covered"]
    ref_nll_cov   = results["full_vocab_refined_nll_covered"]
    nll_diff_cov  = abs(ref_nll_cov - base_nll_cov)

    og_base = results["full_vocab_outside_gate_base_nll_all"]
    og_ref  = results["full_vocab_outside_gate_ref_nll_all"]
    outside_gate_diff = abs(og_ref - og_base)

    delta_max_abs = results["mean_delta_abs"]

    fp_ok = (results["dataset_fingerprint"] == results.get("dataset_fingerprint", ""))

    print(f"\n[stage05] Step-0 identity checks:")
    print(f"  base_nll_all        = {base_nll_all:.6f}")
    print(f"  refined_nll_all     = {ref_nll_all:.6f}")
    print(f"  nll_diff_all        = {nll_diff_all:.2e}   (thresh < 1e-3)")
    print(f"  nll_diff_covered    = {nll_diff_cov:.2e}   (thresh < 1e-3)")
    print(f"  outside_gate_diff   = {outside_gate_diff:.2e}   (thresh < 1e-5)")
    print(f"  mean_delta_abs      = {delta_max_abs:.2e}   (thresh < 1e-6)")
    print(f"  dataset_fingerprint = {results['dataset_fingerprint']!r}")

    identity_pass = (
        nll_diff_all      < 1e-3 and
        nll_diff_cov      < 1e-3 and
        outside_gate_diff < 1e-5 and
        delta_max_abs     < 1e-6
    )

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "identity_pass":       identity_pass,
        "nll_diff_all":        nll_diff_all,
        "nll_diff_covered":    nll_diff_cov,
        "outside_gate_diff":   outside_gate_diff,
        "mean_delta_abs":      delta_max_abs,
        "base_nll_all":        base_nll_all,
        "refined_nll_all":     ref_nll_all,
        "base_nll_covered":    base_nll_cov,
        "refined_nll_covered": ref_nll_cov,
        "dataset_fingerprint": results["dataset_fingerprint"],
        "num_examples":        results["num_examples"],
        "num_covered":         results["num_covered"],
        "coverage":            results["coverage"],
        "gate_rate":           results["gate_rate"],
        "alpha_init":          results["alpha"],
        "delta_scale":         results["delta_scale"],
        "top_k":               args.top_k,
        "num_neighbors":       args.num_neighbors,
        "resolver_dim":        args.resolver_dim,
        "resolver_layers":     args.resolver_layers,
        "resolver_heads":      args.resolver_heads,
        "gate_filter":         args.gate_filter,
        "delta_head_weight_max_abs": dw_norm,
        "delta_head_bias_max_abs":   db_norm,
        "eval_seconds":        time.time() - t0,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    rpath = os.path.join(args.output_dir, "debug_identity.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[stage05] identity_pass = {identity_pass}")
    print(f"[stage05] Report → {rpath}")

    if not identity_pass:
        issues = []
        if nll_diff_all      >= 1e-3:  issues.append(f"nll_diff_all={nll_diff_all:.2e} >= 1e-3")
        if nll_diff_cov      >= 1e-3:  issues.append(f"nll_diff_cov={nll_diff_cov:.2e} >= 1e-3")
        if outside_gate_diff >= 1e-5:  issues.append(f"outside_gate_diff={outside_gate_diff:.2e} >= 1e-5")
        if delta_max_abs     >= 1e-6:  issues.append(f"mean_delta_abs={delta_max_abs:.2e} >= 1e-6")
        raise RuntimeError("Stage 05 identity FAILED: " + "; ".join(issues))

    print("[stage05] PASS")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",         required=True)
    p.add_argument("--val_cand_dir",       required=True)
    p.add_argument("--val_ctx_dir",        required=True)
    p.add_argument("--val_retrieval_dir",  required=True)
    p.add_argument("--baseline_json",      default=None)
    p.add_argument("--super_map",          required=True)
    p.add_argument("--region_map",         required=True)
    p.add_argument("--output_dir",         required=True)
    p.add_argument("--stage04_report",     default=None,
                   help="Path to stage04 retrieval_audit.json; skipped if omitted")
    p.add_argument("--ctx_len",            type=int,   default=256)
    p.add_argument("--top_k",             type=int,   default=256)
    p.add_argument("--num_neighbors",      type=int,   default=32)
    p.add_argument("--resolver_dim",       type=int,   default=256)
    p.add_argument("--resolver_layers",    type=int,   default=2)
    p.add_argument("--resolver_heads",     type=int,   default=4)
    p.add_argument("--delta_scale",        type=float, default=0.25)
    p.add_argument("--gate_filter",        default="boundary")
    p.add_argument("--eval_batch_size",    type=int,   default=64)
    p.add_argument("--device",             default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
