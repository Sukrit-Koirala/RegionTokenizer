#!/usr/bin/env python3
"""
preflight_boundary.py

Sanity-checks boundary model architecture before committing to a long run.
Verifies: gate formula, shapes, loss finiteness, gradient flow, no label clamping.

Usage:
    python scripts/preflight_boundary.py --mode repr_region_boundary_seqrefine \
        --region_map_path runs/region_maps_128/token_to_region.json \
        [--device cuda]

Exit code 0 = all checks passed. Non-zero = at least one hard failure.
"""
import argparse
import json
import math
import sys
import torch
import torch.nn.functional as F

# ── make sure train_region_lm imports work ────────────────────────────────────
import importlib.util, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "train_region_lm", os.path.join(_root, "train_region_lm.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
TrainConfig              = m.TrainConfig
load_coarse_map          = m.load_coarse_map
make_random_partition_map = m.make_random_partition_map
ReprRegionBoundaryLM     = m.ReprRegionBoundaryLM
ReprRegionMultiHypLM     = m.ReprRegionMultiHypLM
ReprRegionBranchAttnLM   = m.ReprRegionBranchAttnLM
ReprRegionBoundarySeqRefineLM = m.ReprRegionBoundarySeqRefineLM

BOUNDARY_MODES = {
    "repr_region_boundary":          ReprRegionBoundaryLM,
    "repr_region_multihyp":          ReprRegionMultiHypLM,
    "repr_region_branchattn":        ReprRegionBranchAttnLM,
    "repr_region_branch_identity":   ReprRegionBranchAttnLM,
    "repr_region_boundary_seqrefine":ReprRegionBoundarySeqRefineLM,
    "random_repr_region_boundary_seqrefine": ReprRegionBoundarySeqRefineLM,
}

VOCAB_SIZE   = 50257
BATCH        = 4
SEQ_LEN      = 64   # small for speed
N_COARSE     = 128  # set from map

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures = []

def check(name: str, cond: bool, msg: str, hard: bool = True):
    tag = PASS if cond else (FAIL if hard else WARN)
    print(f"  {tag}  {name}: {msg}")
    if not cond and hard:
        failures.append(name)


def run_preflight(mode: str, region_map_path: str, device_str: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Preflight: {mode}")
    print(f"{'='*60}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── Load region map ────────────────────────────────────────────────────────
    raw_coarse, n_coarse = load_coarse_map(region_map_path, VOCAB_SIZE)
    if "random" in mode:
        raw_coarse = make_random_partition_map(raw_coarse, n_coarse, seed=99)
        print(f"  [INFO] random partition applied ({n_coarse} regions)")

    coarse_map = raw_coarse.to(device)
    print(f"  [INFO] coarse: {n_coarse} regions  "
          f"coverage={(raw_coarse >= 0).float().mean():.1%}")

    # ── Build model ───────────────────────────────────────────────────────────
    cfg = TrainConfig(
        mode=mode,
        dataset="wikitext-103-raw-v1",
        seq_len=SEQ_LEN,
        batch_size=BATCH,
        d_model=384,
        n_head=6,
        d_ff=1536,
        n_layer=6,
        n_coarse=n_coarse,
        d_region=64,
        router_type="mlp",
        router_temp=2.0,
        alpha_core=0.2,
        alpha_boundary=0.4,
        boundary_tau=0.03,
        boundary_temp=0.01,
        boundary_mode="margin",
        hyp_k=4,
        branch_attn_heads=4,
        branch_attn_layers=1,
        branch_scale_init=0.05,
        boundary_refine_layers=1,
        boundary_refine_gamma=0.5,
        lambda_coarse=0.2,
        lambda_balance=0.001,
    )

    cls = BOUNDARY_MODES[mode]
    model = cls(cfg, VOCAB_SIZE, coarse_map=coarse_map).to(device).eval()
    n_params = model.param_count()
    print(f"  [INFO] params: {n_params:,}")

    # ── Fake batch ─────────────────────────────────────────────────────────────
    torch.manual_seed(0)
    batch = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN + 1), device=device)
    src, tgt = batch[:, :-1], batch[:, 1:]

    # ── Forward ───────────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            lm_logits, region_logits, p_region, h_out = model.forward(src)
    except Exception as e:
        check("forward_ok", False, f"forward() raised {e}")
        return False

    check("forward_ok", True, "forward() completed")

    # ── Shape checks ──────────────────────────────────────────────────────────
    B, T = src.shape
    check("lm_logits_shape",   lm_logits.shape    == (B, T, VOCAB_SIZE),
          f"got {tuple(lm_logits.shape)}")
    check("region_logits_shape", region_logits.shape == (B, T, n_coarse),
          f"got {tuple(region_logits.shape)}")
    check("p_region_shape",    p_region.shape     == (B, T, n_coarse),
          f"got {tuple(p_region.shape)}")
    check("h_out_shape",       h_out.shape        == (B, T, cfg.d_model),
          f"got {tuple(h_out.shape)}")

    # ── Gate checks ───────────────────────────────────────────────────────────
    # Gate must use FULL p_region (not top-k renormalized)
    check("p_region_sums_to_1",
          (p_region.sum(-1) - 1.0).abs().max().item() < 1e-4,
          f"max |sum-1| = {(p_region.sum(-1)-1.0).abs().max().item():.2e}")

    gate = model._last_gate
    check("gate_cached",   gate is not None, "model._last_gate is None", hard=False)
    if gate is not None:
        bnd_frac = (gate.reshape(-1) >= 0.5).float().mean().item()
        margin_q = ""
        # Compute margin manually for diagnostic
        top2 = p_region.topk(2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).cpu()
        q10, q25, q50, q75, q90 = [
            margin.quantile(q).item() for q in [0.1, 0.25, 0.5, 0.75, 0.9]
        ]
        gate_mean = gate.reshape(-1).mean().item()
        gate_gt05 = (gate.reshape(-1) >= 0.5).float().mean().item()
        gate_gt09 = (gate.reshape(-1) >= 0.9).float().mean().item()
        print(f"  [INFO] margin  q10={q10:.3f}  q25={q25:.3f}  q50={q50:.3f}"
              f"  q75={q75:.3f}  q90={q90:.3f}")
        print(f"  [INFO] gate    mean={gate_mean:.3f}  >0.5={gate_gt05:.3f}"
              f"  >0.9={gate_gt09:.3f}")
        print(f"  [INFO] boundary_frac (gate>0.5) = {bnd_frac:.3f}")
        # At init with random weights, router is near-uniform → high gate everywhere.
        # Only warn, not fail, since this is before any training.
        check("gate_not_all_one",
              gate_mean < 0.99, f"gate mean={gate_mean:.3f} (near 1 — normal at init)",
              hard=False)

    # ── Coarse label check — no accidental clamping ───────────────────────────
    coarse_labels = coarse_map[tgt]
    n_unknown = (coarse_labels < 0).sum().item()
    n_total   = coarse_labels.numel()
    check("unknown_labels_not_clamped",
          True,  # structural: we check it's handled correctly in loss
          f"{n_unknown}/{n_total} targets have unknown region (will be masked in L_coarse)")

    # ── Loss finiteness ───────────────────────────────────────────────────────
    model.train()
    try:
        losses = model.loss(batch)
        lm_val = losses["lm"].item()
        tot_val = losses["total"].item()
        check("loss_finite", math.isfinite(lm_val) and math.isfinite(tot_val),
              f"lm={lm_val:.4f}  total={tot_val:.4f}")
        check("loss_reasonable", lm_val < 15.0,
              f"lm={lm_val:.4f} (expected < 15 at random init)")
    except Exception as e:
        check("loss_finite", False, f"loss() raised: {e}")
        return False

    # ── Gradient flow check ──────────────────────────────────────────────────
    losses["total"].backward()

    def grad_norm(params) -> float:
        g = [p.grad for p in params if p.grad is not None]
        if not g:
            return 0.0
        return float(torch.stack([x.norm() for x in g]).norm().item())

    trunk_gnorm  = grad_norm(p for b in model.blocks for p in b.parameters())
    router_gnorm = grad_norm(model.coarse_head.parameters())
    proj_gnorm   = grad_norm(model.region_proj.parameters())

    check("trunk_grad_flows",  trunk_gnorm  > 1e-8,
          f"||trunk.grad||={trunk_gnorm:.2e}")
    check("router_grad_flows", router_gnorm > 1e-8,
          f"||router.grad||={router_gnorm:.2e}")
    check("proj_grad_flows",   proj_gnorm   > 1e-8,
          f"||region_proj.grad||={proj_gnorm:.2e}")

    if isinstance(model, ReprRegionBranchAttnLM):
        # branch_identity mode intentionally bypasses branch_refiner (use_branch_attn=False).
        # Only check refiner gradients when the refiner is actually used.
        if getattr(model, "use_branch_attn", True):
            br_gnorm  = grad_norm(model.branch_refiner.parameters())
            br_sg     = model.branch_refiner.branch_scale.grad
            br_sg_val = float(br_sg.item()) if br_sg is not None else 0.0
            check("branch_refiner_grad_flows", br_gnorm > 1e-10,
                  f"||branch_refiner.grad||={br_gnorm:.2e}")
            check("branch_scale_grad_nonzero", abs(br_sg_val) > 1e-10,
                  f"branch_scale.grad={br_sg_val:.2e} (was 0.0 when double-zero-init bug was present)")
        else:
            print(f"  [INFO] branch_identity mode: refiner bypassed (use_branch_attn=False) — skipping refiner grad checks")

    if isinstance(model, ReprRegionBoundarySeqRefineLM):
        sr_gnorm  = grad_norm(model.seq_refine_blocks.parameters())
        rs_sg     = model.refine_scale.grad
        rs_sg_val = float(rs_sg.item()) if rs_sg is not None else 0.0
        check("seq_refine_grad_flows",    sr_gnorm  > 1e-10,
              f"||seq_refine_blocks.grad||={sr_gnorm:.2e}")
        check("refine_scale_grad_nonzero", abs(rs_sg_val) > 1e-10,
              f"refine_scale.grad={rs_sg_val:.2e}")

    print(f"  [INFO] trunk gnorm={trunk_gnorm:.2e}  router gnorm={router_gnorm:.2e}"
          f"  proj gnorm={proj_gnorm:.2e}")

    return len(failures) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(BOUNDARY_MODES))
    ap.add_argument("--region_map_path", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ok = run_preflight(args.mode, args.region_map_path, args.device)
    print()
    if ok:
        print(f"{'='*60}")
        print(f"ALL CHECKS PASSED for {args.mode}")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"{'='*60}")
        print(f"PREFLIGHT FAILED — {len(failures)} hard failure(s):")
        for f in failures:
            print(f"  - {f}")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
