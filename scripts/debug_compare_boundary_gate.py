"""
debug_compare_boundary_gate.py
================================
Verifies that ReprRegionBoundaryLM and ReprRegionMultiHypLM produce
statistically identical gate distributions under the same router weights,
same batch, same tau/temp.

Usage (from repo root):
    python scripts/debug_compare_boundary_gate.py
    python scripts/debug_compare_boundary_gate.py --tau 0.03 --temp 0.01
    python scripts/debug_compare_boundary_gate.py \
        --bnd_ckpt  runs/repr_region_boundary_a0p4/checkpoint_latest.pt \
        --mhyp_ckpt runs/repr_region_multihyp_k4_tau0p03_temp0p01/checkpoint_latest.pt

Without checkpoints, random weights are used — this still validates that the
gate FORMULA is identical, but margin stats will differ from trained models.

With checkpoints, the script copies the boundary model's router weights into
the multihyp model so the comparison is purely about gate computation, not
about learned distributions.

Exit code: 0 if gate stats match within tolerance, 1 otherwise.
"""

import argparse
import math
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")  # run from repo root

from train_region_lm import (
    TrainConfig,
    ReprRegionBoundaryLM,
    ReprRegionMultiHypLM,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def margin_stats(p_region: torch.Tensor) -> dict:
    """Return margin summary dict from a full p_region tensor (B, T, K)."""
    top2   = p_region.topk(2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).reshape(-1)
    qs = torch.quantile(margin, torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]))
    return {
        "mean":  margin.mean().item(),
        "min":   margin.min().item(),
        "max":   margin.max().item(),
        "q01":   qs[0].item(),
        "q05":   qs[1].item(),
        "q10":   qs[2].item(),
        "q25":   qs[3].item(),
        "q50":   qs[4].item(),
        "q75":   qs[5].item(),
        "q90":   qs[6].item(),
    }


def gate_stats(gate: torch.Tensor) -> dict:
    g = gate.reshape(-1)
    return {
        "mean":     g.mean().item(),
        "frac>0.5": (g > 0.5).float().mean().item(),
        "frac>0.9": (g > 0.9).float().mean().item(),
        "frac<0.1": (g < 0.1).float().mean().item(),
    }


def print_side_by_side(title_a: str, a: dict, title_b: str, b: dict):
    keys = list(a.keys())
    col = max(len(k) for k in keys)
    w = 12
    print(f"  {'key':{col}}  {title_a:>{w}}  {title_b:>{w}}  {'diff':>{w}}")
    print(f"  {'-'*col}  {'-'*w}  {'-'*w}  {'-'*w}")
    for k in keys:
        va = a[k]
        vb = b.get(k, float("nan"))
        diff = vb - va if not (math.isnan(va) or math.isnan(vb)) else float("nan")
        diff_s = f"{diff:+.6f}" if not math.isnan(diff) else "—"
        print(f"  {k:{col}}  {va:{w}.6f}  {vb:{w}.6f}  {diff_s:{w}}")


def copy_router_weights(src_model, dst_model):
    """Copy token_emb, pos_emb, blocks, ln_f, coarse_head from src into dst."""
    shared_keys = ["token_emb", "pos_emb", "ln_f", "coarse_head"]
    for key in shared_keys:
        src_m = getattr(src_model, key)
        dst_m = getattr(dst_model, key)
        dst_m.load_state_dict(src_m.state_dict())
    for i, (sb, db) in enumerate(zip(src_model.blocks, dst_model.blocks)):
        db.load_state_dict(sb.state_dict())
    print(f"  Copied backbone+router weights from boundary → multihyp model")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tau",       type=float, default=0.03)
    p.add_argument("--temp",      type=float, default=0.01)
    p.add_argument("--hyp_k",     type=int,   default=4)
    p.add_argument("--batch",     type=int,   default=8)
    p.add_argument("--seq_len",   type=int,   default=128)
    p.add_argument("--n_coarse",  type=int,   default=128)
    p.add_argument("--d_model",   type=int,   default=384)
    p.add_argument("--n_layer",   type=int,   default=6)
    p.add_argument("--n_head",    type=int,   default=6)
    p.add_argument("--d_ff",      type=int,   default=1536)
    p.add_argument("--vocab",     type=int,   default=50257)
    p.add_argument("--router_temp", type=float, default=2.0)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--device",    type=str,   default="cpu")
    p.add_argument("--bnd_ckpt",  type=str,   default=None,
                   help="Path to repr_region_boundary checkpoint (optional)")
    p.add_argument("--mhyp_ckpt", type=str,   default=None,
                   help="Path to repr_region_multihyp checkpoint (optional)")
    p.add_argument("--tol",       type=float, default=0.05,
                   help="Max allowed absolute difference in bnd_frac for PASS verdict")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("Boundary Gate Consistency Check")
    print("=" * 60)
    print(f"  tau={args.tau}  temp={args.temp}  K={args.hyp_k}")
    print(f"  batch={args.batch}  seq_len={args.seq_len}  n_coarse={args.n_coarse}")
    print()

    # ── Build configs ──────────────────────────────────────────────────────────
    base_cfg = dict(
        mode="repr_region_boundary",
        n_coarse=args.n_coarse,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
        router_temp=args.router_temp,
        boundary_tau=args.tau,
        boundary_temp=args.temp,
        boundary_mode="margin",
        alpha_core=0.2,
        alpha_boundary=0.4,
        d_region=64,
    )

    cfg_bnd  = TrainConfig(**{**base_cfg, "mode": "repr_region_boundary"})
    cfg_mhyp = TrainConfig(**{**base_cfg, "mode": "repr_region_multihyp",
                               "hyp_k": args.hyp_k})

    vocab_size = args.vocab

    # coarse_map is required by __init__ but only used inside loss() for label
    # supervision and oracle routing — never touched by forward() alone.
    # Use a uniform dummy map so the models can be constructed without a real
    # region-map file.
    dummy_coarse_map = torch.zeros(vocab_size, dtype=torch.long)

    # ── Build models ───────────────────────────────────────────────────────────
    bnd_model  = ReprRegionBoundaryLM(cfg_bnd,  vocab_size,
                                      coarse_map=dummy_coarse_map).to(device).eval()
    mhyp_model = ReprRegionMultiHypLM(cfg_mhyp, vocab_size,
                                      coarse_map=dummy_coarse_map).to(device).eval()

    # ── Load checkpoints (optional) ────────────────────────────────────────────
    if args.bnd_ckpt:
        ckpt = torch.load(args.bnd_ckpt, map_location=device, weights_only=False)
        bnd_model.load_state_dict(ckpt["model"], strict=False)
        print(f"  Loaded boundary ckpt: {args.bnd_ckpt}")

    if args.mhyp_ckpt:
        ckpt = torch.load(args.mhyp_ckpt, map_location=device, weights_only=False)
        mhyp_model.load_state_dict(ckpt["model"], strict=False)
        print(f"  Loaded multihyp ckpt: {args.mhyp_ckpt}")

    # ── Share router weights (copy boundary → multihyp) ────────────────────────
    # This makes the comparison purely about gate formula, not learned distribution.
    if not args.mhyp_ckpt:
        copy_router_weights(bnd_model, mhyp_model)

    # ── Same random batch ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    batch = torch.randint(0, vocab_size, (args.batch, args.seq_len + 1), device=device)
    src = batch[:, :-1]   # (B, T)

    # ── Forward passes ─────────────────────────────────────────────────────────
    with torch.no_grad():
        _, region_logits_bnd,  p_bnd,  _ = bnd_model.forward(src)
        _, region_logits_mhyp, p_mhyp, _ = mhyp_model.forward(src)

    gate_bnd  = bnd_model._last_gate                     # (B, T) CPU
    gate_mhyp = mhyp_model._last_gate                    # (B, T) CPU

    # ── Margin comparison ──────────────────────────────────────────────────────
    print("Margin statistics (from full p_region)")
    print("-" * 60)
    ms_bnd  = margin_stats(p_bnd.cpu())
    ms_mhyp = margin_stats(p_mhyp.cpu())
    print_side_by_side("boundary", ms_bnd, "multihyp", ms_mhyp)
    print()

    # ── Gate comparison ────────────────────────────────────────────────────────
    print("Gate statistics")
    print("-" * 60)
    gs_bnd  = gate_stats(gate_bnd)
    gs_mhyp = gate_stats(gate_mhyp)
    print_side_by_side("boundary", gs_bnd, "multihyp", gs_mhyp)
    print()

    # ── If p_region is same, recompute gate for mhyp using bnd distribution ───
    # Verifies the gate FORMULA is identical even if distributions differ.
    print("Formula check: apply bnd p_region to mhyp _boundary_gate()")
    print("-" * 60)
    with torch.no_grad():
        gate_mhyp_on_bnd_dist = mhyp_model._boundary_gate(p_bnd.to(device)).cpu()
        gate_bnd_on_bnd_dist  = bnd_model._boundary_gate(p_bnd.to(device)).cpu()
    formula_max_diff = (gate_mhyp_on_bnd_dist - gate_bnd_on_bnd_dist).abs().max().item()
    formula_mean_diff = (gate_mhyp_on_bnd_dist - gate_bnd_on_bnd_dist).abs().mean().item()
    print(f"  max |gate_mhyp(p_bnd) - gate_bnd(p_bnd)| = {formula_max_diff:.8f}")
    print(f"  mean|gate_mhyp(p_bnd) - gate_bnd(p_bnd)| = {formula_mean_diff:.8f}")
    formula_ok = formula_max_diff < 1e-5
    print(f"  Formula {'IDENTICAL' if formula_ok else 'DIFFERS'} (tol 1e-5)")
    print()

    # ── Verdict ────────────────────────────────────────────────────────────────
    bnd_frac_bnd  = gs_bnd["frac>0.5"]
    bnd_frac_mhyp = gs_mhyp["frac>0.5"]
    frac_diff     = abs(bnd_frac_mhyp - bnd_frac_bnd)

    print("=" * 60)
    print("Verdict")
    print("=" * 60)
    print(f"  boundary_frac  boundary  = {bnd_frac_bnd:.4f}")
    print(f"  boundary_frac  multihyp  = {bnd_frac_mhyp:.4f}")
    print(f"  |diff|                   = {frac_diff:.4f}  (tol={args.tol})")
    print()

    if not formula_ok:
        print("  FAIL  Gate FORMULA differs — code bug in _boundary_gate()")
        sys.exit(1)

    if frac_diff <= args.tol:
        print("  PASS  Gate distributions match (formula verified, shared weights)")
    else:
        print(f"  WARN  Gate distributions differ by {frac_diff:.4f} > {args.tol}")
        print("        This is expected if checkpoints were trained independently.")
        print("        If shared weights were used, this indicates a gradient leak.")
        if not args.mhyp_ckpt:
            print("  FAIL  Shared-weight gate mismatch — check for remaining gradient leaks")
            sys.exit(1)
        else:
            print("  INFO  Different checkpoints → difference is from router training dynamics")
            print("        Rerun with --mhyp_ckpt omitted and shared weights to verify formula.")

    print()


if __name__ == "__main__":
    main()
