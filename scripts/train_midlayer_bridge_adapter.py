#!/usr/bin/env python3
"""
Mid-Layer Bridge Adapter — conditions the bridge on a mid-backbone layer state.

Key difference from BridgeResidualAdapter (V1):
  The bridge sequence begins with an INSERT_STATE token built from
  h_layers[:, insert_layer_idx, :] — a mid-backbone hidden state.
  This gives the PATH token access to what the backbone "knew" at layer L
  before routing and final-layer processing.

Architecture:
  Bridge sequence:
    [INSERT_STATE]            — layer_proj(h_layers[insert_idx]) + insert_marker_emb
    [layer_0, ..., layer_n-1] — all n_ctx_layers layer tokens (same as V1)
    [ROUTER]
    [MEMORY]
    [fine_0, ..., fine_{F-1}]
    [super_0, ..., super_{S-1}]
    [PATH × num_path_tokens]  — pooled output → delta_h

Write target: h_refined = h_prime + alpha * out_proj(path_out)  (same as V1)
Decode:       logits = h_refined @ token_emb.T  (full-vocab, V=50,257)

Identity invariant: out_proj zero-init → delta_h=0 at step 0.

NOTE: True mid-layer injection requires full sequence context (backbone.blocks use
fixed-size causal attention masks). This design tests "mid-layer conditioning" —
the bridge sees h_insert and uses it to compute the residual update for h_prime.

Usage:
    python scripts/train_midlayer_bridge_adapter.py \\
        --small_ckpt  runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --train_cand_dir runs/path_refiner_clean/data/train_hgrid_K24 \\
        --val_cand_dir   runs/path_refiner_clean/data/val_hgrid_K24 \\
        --train_feat_dir runs/path_refiner_residual_interface/features/train_multilayer \\
        --val_feat_dir   runs/path_refiner_residual_interface/features/val_multilayer \\
        --baseline_json  runs/path_refiner_clean/baselines/saved_candidate_baseline.json \\
        --super_map runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --output_dir runs/path_refiner_midlayer_bridge/boundary_insert4_v1 \\
        --insert_after_block 4 \\
        --train_filter boundary --gate_filter boundary \\
        --bridge_dim 256 --num_bridge_layers 2 --num_heads 4 \\
        --steps 10000 --eval_every 1000 --batch_size 32 --grad_accum_steps 2 \\
        --lr 5e-5 --lambda_kl 0.2 --lambda_delta 3e-4 --kl_topk 512 \\
        --use_filtered_train_loader --eval_before_train --fail_on_baseline_mismatch
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.train_clean_path_refiner import load_r2s
from scripts.train_bridge_residual_adapter import (
    # Canonical constants
    CANONICAL_FINGERPRINT, CANONICAL_NUM_EXAMPLES, CANONICAL_NUM_COVERED,
    CANONICAL_COVERAGE, MASKED_CAND_BASELINE_NLL,
    # Dataset / collate
    BridgeShardDataset, FilteredBridgeShardDataset, collate_bridge,
    # Base model class (inherited by MidLayerBridgeAdapter)
    BridgeResidualAdapter,
    # Loss and eval (accept any model with same forward signature)
    compute_bridge_loss, full_vocab_eval_bridge, local_subset_eval_bridge,
)


# ── Layer-index resolver ──────────────────────────────────────────────────────

def _resolve_insert_layer_idx(insert_after_block, layer_ids: List[int]) -> int:
    """
    Map --insert_after_block (int or 'final') to index in h_layers tensor.

    Example: layer_ids=[0, 2, 4, 5, -1], insert_after_block=4 → returns 2.
    """
    if insert_after_block in ("final", -1, "-1"):
        try:
            return layer_ids.index(-1)
        except ValueError:
            return len(layer_ids) - 1
    iab = int(insert_after_block)
    if iab in layer_ids:
        return layer_ids.index(iab)
    # Find closest non-final layer
    candidates = [(i, abs(lid - iab)) for i, lid in enumerate(layer_ids) if lid != -1]
    if not candidates:
        return len(layer_ids) - 1
    best_i = min(candidates, key=lambda x: x[1])[0]
    print(
        f"  WARNING: insert_after_block={iab} not in layer_ids={layer_ids}. "
        f"Using closest: layer_ids[{best_i}]={layer_ids[best_i]}"
    )
    return best_i


# ── Model ─────────────────────────────────────────────────────────────────────

class MidLayerBridgeAdapter(BridgeResidualAdapter):
    """
    Extends BridgeResidualAdapter by prepending an INSERT_STATE token.

    The INSERT_STATE is built from h_layers[:, insert_layer_idx, :] — the
    backbone hidden state after a specific block — combined with a learned
    insert_marker_emb to distinguish it from regular layer tokens.

    Bridge sequence:
      [INSERT_STATE | LAYER_0..n-1 | ROUTER | MEMORY | FINE×F | SUPER×S | PATH×P]

    Path output is pooled over P tokens before passing through out_proj.
    out_proj is zero-initialized → identity at step 0.
    """

    def __init__(
        self,
        d_model:          int,
        n_ctx_layers:     int,
        bridge_dim:       int,
        num_heads:        int,
        num_layers:       int,
        ff_mult:          int,
        n_fine:           int,
        n_super:          int,
        top_fine_k:       int,
        top_super_k:      int,
        insert_layer_idx: int,
        num_path_tokens:  int   = 1,
        dropout:          float = 0.0,
        r2s_np:           Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            d_model=d_model, n_ctx_layers=n_ctx_layers,
            bridge_dim=bridge_dim, num_heads=num_heads, num_layers=num_layers,
            ff_mult=ff_mult, n_fine=n_fine, n_super=n_super,
            top_fine_k=top_fine_k, top_super_k=top_super_k,
            dropout=dropout, r2s_np=r2s_np,
        )
        self.insert_layer_idx = insert_layer_idx
        self.num_path_tokens  = num_path_tokens

        # Learned embedding marks INSERT_STATE as distinct from layer tokens
        self.insert_marker_emb = nn.Embedding(1, bridge_dim)

        # Replace parent's path_token to support P > 1
        self.path_token = nn.Parameter(torch.zeros(1, num_path_tokens, bridge_dim))
        nn.init.normal_(self.path_token, std=0.02)

        # out_proj is already zero-initialized by super().__init__() — identity at step 0

    def forward(
        self,
        h_prime:    torch.Tensor,   # (B, d_model)
        h_layers:   torch.Tensor,   # (B, n_layers, d_model)
        cand_tok:   torch.Tensor,   # (B, C)
        cand_fine:  torch.Tensor,   # (B, C)
        cand_mask:  torch.Tensor,   # (B, C) bool
        tok_emb_w:  torch.Tensor,   # (V, d_model)
        r_topk_reg: torch.Tensor,   # (B, K)
        r_topk_prb: torch.Tensor,   # (B, K)
        m_topk_reg: torch.Tensor,   # (B, K)
        m_topk_prb: torch.Tensor,   # (B, K)
        r_margin:   torch.Tensor,   # (B,)
        m_margin:   torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B      = h_prime.shape[0]
        device = h_prime.device

        # ── INSERT_STATE token ────────────────────────────────────────────────
        h_insert   = h_layers[:, self.insert_layer_idx, :].float()        # (B, d_model)
        ins_proj   = self.layer_proj(h_insert)                            # (B, D)
        ins_marker = self.insert_marker_emb(
            torch.zeros(B, dtype=torch.long, device=device))              # (B, D)
        insert_t   = (ins_proj + ins_marker).unsqueeze(1)                 # (B, 1, D)

        # ── Layer tokens (all n_ctx_layers, same as V1) ───────────────────────
        layer_t = self.layer_proj(h_layers.float())                       # (B, n, D)
        ids     = torch.arange(self.n_ctx_layers, device=device)
        layer_t = layer_t + self.layer_id_emb(ids).unsqueeze(0)

        # ── Router / memory / fine / super tokens (inherited from V1) ────────
        rt = self._router_token(r_topk_reg, r_topk_prb, r_margin)
        mt = self._memory_token(m_topk_reg, m_topk_prb, m_margin)
        ft = self._fine_region_tokens(
            r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
            cand_tok, cand_fine, cand_mask, tok_emb_w, h_prime,
        )
        st = self._super_tokens(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb)

        # ── PATH tokens ───────────────────────────────────────────────────────
        path_tok = self.path_token.expand(B, -1, -1)                      # (B, P, D)

        # ── Full sequence: INSERT_STATE leads ─────────────────────────────────
        seq = torch.cat(
            [insert_t, layer_t, rt.unsqueeze(1), mt.unsqueeze(1), ft, st, path_tok], dim=1
        )
        seq = self.transformer(seq)

        # ── PATH output: pool over P tokens → delta_h ────────────────────────
        path_out = seq[:, -self.num_path_tokens:, :].mean(dim=1)          # (B, D)
        delta_h  = self.out_proj(path_out)                                # (B, d_model)

        h_refined = h_prime.float() + self.alpha * delta_h
        return h_refined, delta_h


# ── Report ────────────────────────────────────────────────────────────────────

def _generate_report(
    output_dir:          str,
    args,
    full_vocab_base_nll: float,
    final_metrics:       Dict,
    insert_layer_idx:    int,
    layer_ids:           List[int],
) -> None:
    """Write report.md to output_dir, optionally comparing with V1 results."""
    v1_bm_path = os.path.join(
        "runs", "path_refiner_bridge_adapter",
        "bridge_boundary_v1_hardsampler", "best_metrics.json"
    )
    v1_bm: Optional[Dict] = None
    if os.path.isfile(v1_bm_path):
        with open(v1_bm_path) as f:
            v1_bm = json.load(f)

    no_ckpt = final_metrics.get("no_improving_checkpoint", True)
    best_gated = final_metrics.get("best_full_vocab_gated_nll_all")
    gain = final_metrics.get("full_vocab_gain_all")
    best_step = final_metrics.get("best_step", -1)
    eff_bs = args.batch_size * getattr(args, "grad_accum_steps", 1)

    lines = [
        "# Mid-Layer Bridge Adapter — Run Report",
        "",
        f"**Variant:** `{os.path.basename(output_dir)}`",
        f"**Date:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| insert_after_block | {args.insert_after_block} |",
        f"| insert_layer_idx | {insert_layer_idx} (layer_ids={layer_ids}) |",
        f"| num_path_tokens | {getattr(args, 'num_path_tokens', 1)} |",
        f"| bridge_dim | {args.bridge_dim} |",
        f"| num_bridge_layers | {args.num_bridge_layers} |",
        f"| num_heads | {args.num_heads} |",
        f"| train_filter | {args.train_filter} |",
        f"| gate_filter | {args.gate_filter} |",
        f"| batch_size | {args.batch_size} |",
        f"| grad_accum_steps | {getattr(args, 'grad_accum_steps', 1)} |",
        f"| effective_hard_batch_size | {eff_bs} |",
        f"| lr | {args.lr} |",
        f"| lambda_kl | {args.lambda_kl} |",
        f"| lambda_delta | {args.lambda_delta} |",
        f"| steps | {args.steps} |",
        f"| use_filtered_train_loader | {getattr(args, 'use_filtered_train_loader', False)} |",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| full_vocab_base_nll_all (step 0) | {full_vocab_base_nll:.6f} |",
    ]

    if no_ckpt:
        lines += [
            f"| best_full_vocab_gated_nll_all | N/A — no improving checkpoint |",
            f"| full_vocab_gain_all | N/A |",
            f"| best_step | N/A |",
            "",
            "**RESULT: No improving checkpoint.** The mid-layer bridge did not improve "
            f"over `full_vocab_base_nll_all={full_vocab_base_nll:.6f}` for this configuration.",
        ]
    else:
        lines += [
            f"| best_full_vocab_gated_nll_all | {best_gated:.6f} |",
            f"| full_vocab_gain_all | {gain:+.6f} |",
            f"| best_step | {best_step} |",
            "",
            f"**RESULT: Checkpoint saved at step {best_step}** "
            f"(gain={gain:+.6f} over base).",
        ]

    # Comparison to V1
    if v1_bm is not None:
        lines += [
            "",
            "## Comparison to V1 Bridge Adapter (boundary hardsampler)",
            "",
            "| Metric | V1 (late bridge) | MidLayer (block {}) | Delta |".format(
                args.insert_after_block),
            "|--------|-----------------|----------------------|-------|",
        ]

        def _fmt(d: Optional[Dict], key: str, default="N/A") -> str:
            if d is None:
                return default
            v = d.get(key)
            if v is None:
                return default
            return f"{v:.6f}"

        def _delta(d1: Optional[Dict], d2: Optional[Dict], key: str) -> str:
            if d1 is None or d2 is None:
                return "N/A"
            v1_v = d1.get(key)
            v2_v = d2.get(key)
            if v1_v is None or v2_v is None:
                return "N/A"
            delta = v2_v - v1_v
            return f"{delta:+.6f}"

        v1_no_ckpt = v1_bm.get("no_improving_checkpoint", True)
        ml_bm_path = os.path.join(output_dir, "best_metrics.json")
        ml_bm: Optional[Dict] = None
        if os.path.isfile(ml_bm_path) and not no_ckpt:
            with open(ml_bm_path) as f:
                ml_bm = json.load(f)

        compare_keys = [
            ("full_vocab_gain_all", "full_vocab_gain_all"),
            ("full_vocab_inside_gate_gain_all", "inside_gate_gain_all"),
            ("full_vocab_gain_covered", "full_vocab_gain_covered"),
            ("masked_cand_gain", "masked_cand_gain"),
        ]
        for key, label in compare_keys:
            v1_val  = _fmt(v1_bm if not v1_no_ckpt else None, key)
            ml_val  = _fmt(ml_bm, key)
            dlt_val = _delta(v1_bm if not v1_no_ckpt else None, ml_bm, key)
            lines.append(f"| {label} | {v1_val} | {ml_val} | {dlt_val} |")

        if v1_no_ckpt:
            lines.append("")
            lines.append("*V1 had no improving checkpoint.*")
    else:
        lines += [
            "",
            "## Comparison to V1",
            "",
            f"V1 results not found at `{v1_bm_path}`. Run V1 training first for comparison.",
        ]

    lines += [
        "",
        "## Interpretation",
        "",
        "- `full_vocab_gain_all > 0`: mid-layer conditioning helps overall.",
        "- `inside_gate_gain_all > 0`: within the gate, the residual write improves predictions.",
        f"- Baseline (masked-cand ref): `{MASKED_CAND_BASELINE_NLL:.6f}` (prior refiners used this).",
        "- Compare `masked_cand_gain` to V1 MLP: +0.001284 (global), +0.000530 (hard-boundary).",
        "",
        "## Architecture Note",
        "",
        "The INSERT_STATE token carries the backbone hidden state after block "
        f"{args.insert_after_block} (h_layers index {insert_layer_idx}).",
        "The transformer attends jointly over INSERT_STATE + all layer tokens + region tokens.",
        "This tests whether earlier-layer information not captured in h_prime helps the bridge.",
        "The write target is still h_prime — no mid-layer backbone re-execution.",
    ]

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [report] written to {report_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_midlayer(
    args,
    d_model:          int,
    n_ctx_layers:     int,
    n_fine:           int,
    n_super:          int,
    r2s_np:           np.ndarray,
    tok_emb_w:        torch.Tensor,
    device,
    insert_layer_idx: int,
    layer_ids:        List[int],
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[train] variant               = midlayer_bridge_adapter")
    print(f"[train] insert_after_block    = {args.insert_after_block}"
          f"  → insert_layer_idx={insert_layer_idx}  (layer_ids={layer_ids})")
    print(f"[train] num_path_tokens       = {args.num_path_tokens}")
    print(f"[train] decode_mode           = full_vocab  (primary: _all, secondary: _covered)")
    print(f"[train] selection_mode        = no_candidate_selection")
    print(f"[train] gold_force_included   = never (structural guarantee)")
    print(f"[train] train_objective       = "
          f"{'filter & covered' if args.train_covered_only else 'filter only'}")
    print(f"[train] n_ctx_layers          = {n_ctx_layers}")
    print(f"[train] n_fine={n_fine}  n_super={n_super}")

    model = MidLayerBridgeAdapter(
        d_model          = d_model,
        n_ctx_layers     = n_ctx_layers,
        bridge_dim       = args.bridge_dim,
        num_heads        = args.num_heads,
        num_layers       = args.num_bridge_layers,
        ff_mult          = args.ff_mult,
        n_fine           = n_fine,
        n_super          = n_super,
        top_fine_k       = args.top_fine_regions,
        top_super_k      = args.top_superregions,
        insert_layer_idx = insert_layer_idx,
        num_path_tokens  = args.num_path_tokens,
        dropout          = args.dropout,
        r2s_np           = r2s_np,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] bridge_dim={args.bridge_dim}  heads={args.num_heads}  "
          f"layers={args.num_bridge_layers}  path_tokens={args.num_path_tokens}  "
          f"params={n_params:,}")

    cfg = {**vars(args), "insert_layer_idx": insert_layer_idx, "layer_ids": layer_ids}
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    fail_hard      = args.fail_on_baseline_mismatch
    filter_kwargs  = {"margin_thresh": args.margin_thresh,
                      "entropy_thresh": args.entropy_thresh}
    variant_tag    = (f"midlayer-blk{args.insert_after_block}"
                      f"-M{args.top_fine_regions}+S{args.top_superregions}")

    if args.baseline_json and os.path.isfile(args.baseline_json):
        with open(args.baseline_json) as f:
            masked_bl = json.load(f)
        print(f"\n[train] Masked-candidate baseline (reference only):")
        print(f"  covered_nll = {masked_bl['covered_nll']:.6f}")
        print(f"  NOTE: full-vocab NLL != masked-candidate NLL.")

    tok_dev = tok_emb_w.float().to(device)

    # ── Step-0 identity check ─────────────────────────────────────────────────
    full_vocab_base_nll: float = float("nan")

    if args.eval_before_train:
        print(f"\n[train] === step-0 identity check ===")
        g0 = full_vocab_eval_bridge(
            model, args.val_cand_dir, args.val_feat_dir,
            tok_dev, r2s_np, device,
            gate_filter_name = args.gate_filter,
            filter_kwargs    = filter_kwargs,
            fail_on_mismatch = fail_hard,
            eval_batch_size  = args.eval_batch_size,
            variant_tag      = variant_tag,
        )
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]

        nll_diff_all  = abs(g0["full_vocab_gated_nll_all"]     - g0["full_vocab_base_nll_all"])
        nll_diff_cov  = abs(g0["full_vocab_gated_nll_covered"] - g0["full_vocab_base_nll_covered"])
        mc_diff       = abs(g0["masked_cand_gated_nll"]        - g0["masked_cand_base_nll"])
        og_gate_diff  = abs(g0["full_vocab_outside_gate_gated_nll_all"]
                            - g0["full_vocab_outside_gate_base_nll_all"])
        d_max         = g0["delta_norm_max"]

        print(f"  insert_after_block={args.insert_after_block}  "
              f"insert_layer_idx={insert_layer_idx}  "
              f"layer_ids={layer_ids}")
        print(f"  --- PRIMARY (full-vocab, all positions) ---")
        print(f"  full_vocab_base_nll_all      = {g0['full_vocab_base_nll_all']:.6f}")
        print(f"  full_vocab_gated_nll_all     = {g0['full_vocab_gated_nll_all']:.6f}")
        print(f"  diff_all (must be < 1e-3)    = {nll_diff_all:.2e}")
        print(f"  --- SECONDARY (full-vocab, covered only) ---")
        print(f"  full_vocab_base_nll_covered  = {g0['full_vocab_base_nll_covered']:.6f}")
        print(f"  full_vocab_gated_nll_covered = {g0['full_vocab_gated_nll_covered']:.6f}")
        print(f"  diff_covered (must < 1e-3)   = {nll_diff_cov:.2e}")
        print(f"  --- SECONDARY (masked-candidate) ---")
        print(f"  masked_cand_base_nll         = {g0['masked_cand_base_nll']:.6f}  "
              f"(canonical ref = {MASKED_CAND_BASELINE_NLL:.6f})")
        print(f"  masked_cand_diff             = {mc_diff:.2e}  (must be < 1e-3)")
        print(f"  --- Outside-gate invariant ---")
        print(f"  outside_gate_base_nll_all    = {g0['full_vocab_outside_gate_base_nll_all']:.6f}")
        print(f"  outside_gate_gated_nll_all   = {g0['full_vocab_outside_gate_gated_nll_all']:.6f}")
        print(f"  og_gate_diff (must < 1e-5)   = {og_gate_diff:.2e}")
        print(f"  --- Model state ---")
        print(f"  delta_norm_max (must be 0)   = {d_max:.2e}")
        print(f"  alpha                        = {g0['alpha']:.4f}")

        masked_ref_diff = abs(g0["masked_cand_base_nll"] - MASKED_CAND_BASELINE_NLL)
        if masked_ref_diff > 1e-3:
            raise RuntimeError(
                f"Step-0: masked_cand_base_nll={g0['masked_cand_base_nll']:.6f} "
                f"!= canonical {MASKED_CAND_BASELINE_NLL:.6f} "
                f"(diff={masked_ref_diff:.2e} > 1e-3). Dataset alignment issue.")
        if nll_diff_all >= 1e-3:
            raise RuntimeError(
                f"Step-0 identity FAIL (_all): diff={nll_diff_all:.2e}. "
                "out_proj must be zero-init.")
        if nll_diff_cov >= 1e-3:
            raise RuntimeError(
                f"Step-0 identity FAIL (_covered): diff={nll_diff_cov:.2e}.")
        if mc_diff >= 1e-3:
            raise RuntimeError(
                f"Step-0 identity FAIL (masked_cand): diff={mc_diff:.2e}.")
        if og_gate_diff >= 1e-5:
            raise RuntimeError(
                f"Step-0 outside-gate FAIL: diff={og_gate_diff:.2e} > 1e-5.")
        if d_max > 1e-6:
            raise RuntimeError(
                f"Step-0 identity FAIL: delta_norm_max={d_max:.2e} > 0. "
                "out_proj not zero-init.")

        print(f"  [step-0] IDENTITY PASS — insert_after_block={args.insert_after_block}  "
              f"all families confirmed  gold_force_included_rate=0.0000")

        with open(os.path.join(args.output_dir, "full_vocab_baseline.json"), "w") as f:
            json.dump({
                "full_vocab_base_nll_all":     full_vocab_base_nll,
                "full_vocab_base_nll_covered": g0["full_vocab_base_nll_covered"],
                "masked_cand_base_nll":        g0["masked_cand_base_nll"],
                "masked_cand_baseline_ref":    MASKED_CAND_BASELINE_NLL,
                "coverage":                    g0["coverage"],
                "dataset_fingerprint":         g0["dataset_fingerprint"],
                "num_examples":                g0["num_examples"],
                "num_covered":                 g0["num_covered"],
                "insert_after_block":          str(args.insert_after_block),
                "insert_layer_idx":            insert_layer_idx,
                "note": (
                    "full_vocab_base_nll_all is the PRIMARY threshold for best checkpoint. "
                    "Best checkpoint saved only if full_vocab_gated_nll_all < full_vocab_base_nll_all."
                ),
            }, f, indent=2)

        with open(os.path.join(args.output_dir, "debug_identity.json"), "w") as f:
            json.dump({
                "insert_after_block":    str(args.insert_after_block),
                "insert_layer_idx":      insert_layer_idx,
                "layer_ids":             layer_ids,
                "nll_diff_all":          nll_diff_all,
                "nll_diff_covered":      nll_diff_cov,
                "mc_diff":               mc_diff,
                "og_gate_diff":          og_gate_diff,
                "delta_norm_max":        d_max,
                "identity_pass":         True,
                **{k: g0[k] for k in (
                    "full_vocab_base_nll_all", "full_vocab_gated_nll_all",
                    "full_vocab_base_nll_covered", "full_vocab_gated_nll_covered",
                    "masked_cand_base_nll", "masked_cand_gated_nll",
                    "dataset_fingerprint", "alpha",
                )},
            }, f, indent=2)
    else:
        print("[train] Computing full-vocab baseline (no --eval_before_train) ...")
        g0 = full_vocab_eval_bridge(
            model, args.val_cand_dir, args.val_feat_dir,
            tok_dev, r2s_np, device,
            gate_filter_name = args.gate_filter,
            filter_kwargs    = filter_kwargs,
            fail_on_mismatch = fail_hard,
            eval_batch_size  = args.eval_batch_size,
            variant_tag      = variant_tag,
        )
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]
        print(f"  full_vocab_base_nll_all = {full_vocab_base_nll:.6f}")

    if full_vocab_base_nll != full_vocab_base_nll:  # isnan
        raise RuntimeError("full_vocab_base_nll is NaN — eval failed.")

    print(f"\n[train] Best checkpoint threshold: full_vocab_gated_nll_all < {full_vocab_base_nll:.6f}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    use_filtered = args.use_filtered_train_loader
    grad_accum   = max(1, args.grad_accum_steps)
    eff_bs       = args.batch_size * grad_accum

    print(f"[train] use_filtered_train_loader = {use_filtered}")
    print(f"[train] batch_size                = {args.batch_size}")
    print(f"[train] grad_accum_steps          = {grad_accum}")
    print(f"[train] effective_hard_batch_size = {eff_bs}")
    print()

    if use_filtered:
        train_ds = FilteredBridgeShardDataset(
            args.train_cand_dir, args.train_feat_dir,
            r2s_np, args.train_filter,
            filter_kwargs      = filter_kwargs,
            train_covered_only = args.train_covered_only,
            shuffle            = True,
        )
    else:
        train_ds = BridgeShardDataset(
            args.train_cand_dir, args.train_feat_dir,
            r2s_np, args.train_filter,
            filter_kwargs = filter_kwargs, shuffle=True,
        )

    def _infinite():
        while True:
            for batch in DataLoader(train_ds, batch_size=args.batch_size,
                                    collate_fn=collate_bridge, num_workers=0,
                                    drop_last=False):
                yield batch
    train_inf = _infinite()

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler("cuda") if args.amp else None
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)

    # ── Log files ─────────────────────────────────────────────────────────────
    train_log_path = os.path.join(args.output_dir, "train_log.csv")
    eval_log_path  = os.path.join(args.output_dir, "eval_log.csv")
    subset_path    = os.path.join(args.output_dir, "local_subset_eval.csv")

    train_fields = ["step", "ce", "kl", "delta_norm", "n_train", "alpha", "lr"]
    eval_fields  = [
        "step",
        "full_vocab_base_nll_all", "full_vocab_gated_nll_all", "full_vocab_gain_all",
        "full_vocab_inside_gate_base_nll_all", "full_vocab_inside_gate_ref_nll_all",
        "full_vocab_inside_gate_gain_all",
        "full_vocab_outside_gate_base_nll_all", "full_vocab_outside_gate_gated_nll_all",
        "full_vocab_base_acc1_all", "full_vocab_gated_acc1_all",
        "full_vocab_base_nll_covered", "full_vocab_gated_nll_covered", "full_vocab_gain_covered",
        "full_vocab_inside_gate_base_nll_covered", "full_vocab_inside_gate_ref_nll_covered",
        "full_vocab_inside_gate_gain_covered",
        "full_vocab_outside_gate_base_nll_covered", "full_vocab_outside_gate_gated_nll_covered",
        "masked_cand_base_nll", "masked_cand_gated_nll", "masked_cand_gain",
        "gate_rate", "coverage", "inside_gate_n_all", "inside_gate_n_cov",
        "alpha", "delta_norm_mean", "delta_norm_max",
        "gold_force_included_rate", "dataset_fingerprint",
    ]
    subset_fields = ["step", "subset", "n", "full_vocab_base_nll",
                     "full_vocab_refined_nll", "full_vocab_gain",
                     "base_acc1", "refined_acc1"]

    train_logf  = open(train_log_path, "w", newline="")
    eval_logf   = open(eval_log_path,  "w", newline="")
    subset_logf = open(subset_path,    "w", newline="")
    train_csv   = csv.DictWriter(train_logf,  fieldnames=train_fields,  extrasaction="ignore")
    eval_csv    = csv.DictWriter(eval_logf,   fieldnames=eval_fields,   extrasaction="ignore")
    subset_csv  = csv.DictWriter(subset_logf, fieldnames=subset_fields, extrasaction="ignore")
    train_csv.writeheader()
    eval_csv.writeheader()
    subset_csv.writeheader()

    best_path = os.path.join(args.output_dir, "best_refiner.pt")
    best_nll  = full_vocab_base_nll     # checkpoint only saved if model beats this
    best_step = -1
    ema_ce    = None
    t0        = time.time()
    model.train()
    opt.zero_grad()

    for step in range(1, args.steps + 1):
        accum_infos: List[Dict] = []
        valid_micro = 0

        for _micro in range(grad_accum):
            batch = next(train_inf)
            if use_filtered:
                train_mask = None
            else:
                train_mask = (
                    (batch["filter_mask"] & batch["covered"])
                    if args.train_covered_only
                    else batch["filter_mask"]
                )

            if args.amp:
                with autocast("cuda"):
                    loss, info = compute_bridge_loss(
                        model, batch, device, tok_dev, train_mask,
                        args.lambda_kl, args.lambda_delta, args.kl_topk, amp_enabled=True)
            else:
                loss, info = compute_bridge_loss(
                    model, batch, device, tok_dev, train_mask,
                    args.lambda_kl, args.lambda_delta, args.kl_topk)

            if loss is None:
                continue

            scaled = loss / grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            accum_infos.append(info)
            valid_micro += 1

        if valid_micro == 0:
            continue

        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        sched.step()
        opt.zero_grad()

        avg_ce  = float(np.mean([d["ce"]         for d in accum_infos]))
        avg_kl  = float(np.mean([d["kl"]         for d in accum_infos]))
        avg_dn  = float(np.mean([d["delta_norm"] for d in accum_infos]))
        tot_n   = sum(d["n_train"] for d in accum_infos)

        ema_ce  = avg_ce if ema_ce is None else 0.95 * ema_ce + 0.05 * avg_ce
        alpha_v = float(model.alpha.item())
        lr_now  = sched.get_last_lr()[0]

        train_csv.writerow({"step": step, "ce": avg_ce, "kl": avg_kl,
                            "delta_norm": avg_dn, "n_train": tot_n,
                            "alpha": alpha_v, "lr": lr_now})
        if step % 100 == 0:
            train_logf.flush()
            print(f"  step={step:5d}  ema_ce={ema_ce:.4f}  "
                  f"ce={avg_ce:.4f}  kl={avg_kl:.4f}  "
                  f"d_norm={avg_dn:.4f}  alpha={alpha_v:.4f}  "
                  f"n_train={tot_n}  eff_bs={eff_bs}  t={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] step={step} ...")
            g_m = full_vocab_eval_bridge(
                model, args.val_cand_dir, args.val_feat_dir,
                tok_dev, r2s_np, device,
                gate_filter_name = args.gate_filter,
                filter_kwargs    = filter_kwargs,
                fail_on_mismatch = fail_hard,
                eval_batch_size  = args.eval_batch_size,
                variant_tag      = variant_tag,
            )
            gated_nll = g_m["full_vocab_gated_nll_all"]
            fv_gain   = g_m["full_vocab_gain_all"]

            eval_csv.writerow({"step": step, **g_m})
            eval_logf.flush()

            og_diff = abs(g_m["full_vocab_outside_gate_gated_nll_all"]
                          - g_m["full_vocab_outside_gate_base_nll_all"])
            print(f"  [eval] step={step}  insert_block={args.insert_after_block}")
            print(f"    PRIMARY (full-vocab, all positions):")
            print(f"    full_vocab_base_nll_all        = {g_m['full_vocab_base_nll_all']:.6f}")
            print(f"    full_vocab_gated_nll_all       = {gated_nll:.6f}  "
                  f"(threshold = {full_vocab_base_nll:.6f})")
            print(f"    full_vocab_gain_all            = {fv_gain:+.6f}")
            print(f"    inside_gate_base_nll_all       = {g_m['full_vocab_inside_gate_base_nll_all']:.6f}")
            print(f"    inside_gate_ref_nll_all        = {g_m['full_vocab_inside_gate_ref_nll_all']:.6f}")
            print(f"    inside_gate_gain_all           = {g_m['full_vocab_inside_gate_gain_all']:+.6f}")
            print(f"    outside_gate_diff_all          = {og_diff:.2e}  (must be ~0)")
            print(f"    SECONDARY (covered):")
            print(f"    full_vocab_gated_nll_covered   = {g_m['full_vocab_gated_nll_covered']:.6f}  "
                  f"gain={g_m['full_vocab_gain_covered']:+.6f}")
            print(f"    SECONDARY (masked-cand):")
            print(f"    masked_cand_gain               = {g_m['masked_cand_gain']:+.6f}  "
                  f"(MLP ref: +0.001284 global, +0.000530 hard-boundary)")
            print(f"    alpha={g_m['alpha']:.4f}  "
                  f"delta_mean={g_m['delta_norm_mean']:.4f}  "
                  f"delta_max={g_m['delta_norm_max']:.4f}")
            print(f"    gold_force_included_rate = {g_m['gold_force_included_rate']:.4f}  "
                  f"(must be 0.0000)")

            sub_rows = local_subset_eval_bridge(
                model, args.val_cand_dir, args.val_feat_dir,
                tok_dev, r2s_np, device,
                filter_kwargs   = filter_kwargs,
                eval_batch_size = args.eval_batch_size,
            )
            for r in sub_rows:
                subset_csv.writerow({"step": step, **r})
            subset_logf.flush()
            for r in sub_rows:
                if r["subset"] == args.gate_filter:
                    print(f"    [{args.gate_filter}]  n={r['n']}  "
                          f"base_nll={r['full_vocab_base_nll']:.6f}  "
                          f"ref_nll={r['full_vocab_refined_nll']:.6f}  "
                          f"gain={r['full_vocab_gain']:+.6f}")

            if gated_nll < best_nll:
                best_nll  = gated_nll
                best_step = step
                best_metrics = {
                    "variant":                               "midlayer_bridge_adapter",
                    "insert_after_block":                    str(args.insert_after_block),
                    "insert_layer_idx":                      insert_layer_idx,
                    "num_path_tokens":                       args.num_path_tokens,
                    "selection_mode":                        "no_candidate_selection",
                    "eval_force_include_gold":               False,
                    "gold_force_included_rate":              0.0,
                    "step":                                  step,
                    "train_filter":                          args.train_filter,
                    "gate_filter":                           args.gate_filter,
                    # PRIMARY
                    "full_vocab_base_nll_all":               full_vocab_base_nll,
                    "full_vocab_gated_nll_all":              gated_nll,
                    "full_vocab_gain_all":                   fv_gain,
                    "full_vocab_inside_gate_base_nll_all":   g_m["full_vocab_inside_gate_base_nll_all"],
                    "full_vocab_inside_gate_ref_nll_all":    g_m["full_vocab_inside_gate_ref_nll_all"],
                    "full_vocab_inside_gate_gain_all":       g_m["full_vocab_inside_gate_gain_all"],
                    "full_vocab_outside_gate_base_nll_all":  g_m["full_vocab_outside_gate_base_nll_all"],
                    "full_vocab_outside_gate_gated_nll_all": g_m["full_vocab_outside_gate_gated_nll_all"],
                    # SECONDARY covered
                    "full_vocab_base_nll_covered":           g_m["full_vocab_base_nll_covered"],
                    "full_vocab_gated_nll_covered":          g_m["full_vocab_gated_nll_covered"],
                    "full_vocab_gain_covered":               g_m["full_vocab_gain_covered"],
                    "full_vocab_inside_gate_base_nll_covered": g_m["full_vocab_inside_gate_base_nll_covered"],
                    "full_vocab_inside_gate_ref_nll_covered":  g_m["full_vocab_inside_gate_ref_nll_covered"],
                    "full_vocab_inside_gate_gain_covered":     g_m["full_vocab_inside_gate_gain_covered"],
                    # SECONDARY masked
                    "masked_cand_base_nll":                  g_m["masked_cand_base_nll"],
                    "masked_cand_gated_nll":                 g_m["masked_cand_gated_nll"],
                    "masked_cand_gain":                      g_m["masked_cand_gain"],
                    "masked_cand_baseline_ref":              MASKED_CAND_BASELINE_NLL,
                    # META
                    "gate_rate":                             g_m["gate_rate"],
                    "coverage":                              g_m["coverage"],
                    "num_examples":                          g_m["num_examples"],
                    "num_covered":                           g_m["num_covered"],
                    "alpha":                                 g_m["alpha"],
                    "delta_norm_mean":                       g_m["delta_norm_mean"],
                    "dataset_fingerprint":                   g_m["dataset_fingerprint"],
                    "no_improving_checkpoint":               False,
                }
                torch.save({"step": step, "model": model.state_dict(),
                            "metrics": best_metrics, "args": vars(args)}, best_path)
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"  *** NEW BEST  full_vocab_gated_nll_all={best_nll:.6f}  "
                      f"gain={fv_gain:+.6f}  → {best_path}")
            else:
                print(f"  [no improvement]  gated_nll_all={gated_nll:.6f} >= "
                      f"best={best_nll:.6f}  (baseline={full_vocab_base_nll:.6f})")

    # ── End of training ───────────────────────────────────────────────────────
    torch.save({"step": args.steps, "model": model.state_dict(), "args": vars(args)},
               os.path.join(args.output_dir, "last_refiner.pt"))
    train_logf.close()
    eval_logf.close()
    subset_logf.close()

    final_metrics = {
        "no_improving_checkpoint":       (best_step < 0),
        "best_step":                     best_step,
        "best_full_vocab_gated_nll_all": best_nll if best_step >= 0 else None,
        "full_vocab_base_nll_all":       full_vocab_base_nll,
        "full_vocab_gain_all":           full_vocab_base_nll - best_nll if best_step >= 0 else None,
        "masked_cand_baseline_ref":      MASKED_CAND_BASELINE_NLL,
        "insert_after_block":            str(args.insert_after_block),
        "insert_layer_idx":              insert_layer_idx,
    }
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    if best_step < 0:
        print(f"\n[train] RESULT: no improving checkpoint — bridge never beat "
              f"full_vocab_base_nll_all={full_vocab_base_nll:.6f}.")
        with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
            json.dump({"no_improving_checkpoint": True,
                       "full_vocab_base_nll_all": full_vocab_base_nll,
                       "insert_after_block": str(args.insert_after_block),
                       "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL}, f, indent=2)
    else:
        print(f"\n[train] done  best_full_vocab_gated_nll_all={best_nll:.6f}  "
              f"gain={full_vocab_base_nll-best_nll:+.6f}  step={best_step}")

    _generate_report(args.output_dir, args, full_vocab_base_nll,
                     final_metrics, insert_layer_idx, layer_ids)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device(args.device)

    print(f"[main] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, _ = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    print(f"  d_model={d_model}  n_blocks={len(backbone.blocks)}")

    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().cpu()
    elif hasattr(backbone, "transformer"):
        tok_emb_w = backbone.transformer.wte.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot locate token embedding in backbone")
    print(f"  tok_emb_w shape = {tuple(tok_emb_w.shape)}")

    n_fine = 128
    n_super = 24
    cfg_path = os.path.join(args.val_cand_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            ds_cfg = json.load(f)
        n_fine  = ds_cfg.get("n_fine",  128)
        n_super = ds_cfg.get("n_super",  24)

    r2s_np = np.zeros(n_fine, dtype=np.int32)
    if args.super_map and os.path.isfile(args.super_map):
        r2s_np  = load_r2s(args.super_map, n_fine)
        n_super = int(r2s_np.max()) + 1
    print(f"  n_fine={n_fine}  n_super={n_super}")

    feat_cfg_path = os.path.join(args.val_feat_dir, "config.json")
    if not os.path.isfile(feat_cfg_path):
        raise RuntimeError(
            f"Feature config not found: {feat_cfg_path}. "
            "Run slurm_build_multilayer_residual_features.sh first.")
    with open(feat_cfg_path) as f:
        feat_cfg = json.load(f)
    n_ctx_layers = feat_cfg["n_layers_saved"]
    layer_ids    = feat_cfg["layer_ids"]
    print(f"  n_ctx_layers={n_ctx_layers}  layer_ids={layer_ids}")

    insert_layer_idx = _resolve_insert_layer_idx(args.insert_after_block, layer_ids)
    resolved_block   = layer_ids[insert_layer_idx]
    print(f"  insert_after_block={args.insert_after_block} → "
          f"insert_layer_idx={insert_layer_idx}  (h_layers[{insert_layer_idx}] = block {resolved_block})")

    if args.gate_filter is None:
        args.gate_filter = args.train_filter

    train_midlayer(args, d_model, n_ctx_layers, n_fine, n_super, r2s_np, tok_emb_w, device,
                   insert_layer_idx, layer_ids)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--small_ckpt",     required=True)
    p.add_argument("--train_cand_dir", required=True)
    p.add_argument("--val_cand_dir",   required=True)
    p.add_argument("--train_feat_dir", required=True)
    p.add_argument("--val_feat_dir",   required=True)
    p.add_argument("--baseline_json",  default=None)
    p.add_argument("--super_map",      default=None)
    p.add_argument("--output_dir",     required=True)
    # Architecture
    p.add_argument("--bridge_dim",         type=int,   default=256)
    p.add_argument("--num_bridge_layers",  type=int,   default=2)
    p.add_argument("--num_heads",          type=int,   default=4)
    p.add_argument("--ff_mult",            type=int,   default=4)
    p.add_argument("--dropout",            type=float, default=0.0)
    p.add_argument("--top_fine_regions",   type=int,   default=24)
    p.add_argument("--top_superregions",   type=int,   default=8)
    p.add_argument("--insert_after_block", default="4",
                   help="Backbone block whose output is used as INSERT_STATE. "
                        "Must be in layer_ids. Use 'final' for backbone.ln_f output. "
                        "Default: 4 (maps to layer_ids[2] for layer_ids=[0,2,4,5,-1]).")
    p.add_argument("--num_path_tokens",    type=int,   default=1,
                   help="Number of PATH tokens at the end of the bridge sequence. "
                        "Their outputs are mean-pooled before out_proj. Default: 1.")
    # Filter
    p.add_argument("--train_filter",   default="boundary")
    p.add_argument("--gate_filter",    default=None,
                   help="Gate filter for eval. Defaults to train_filter.")
    p.add_argument("--margin_thresh",  type=float, default=0.1)
    p.add_argument("--entropy_thresh", type=float, default=2.0)
    # Training
    p.add_argument("--steps",           type=int,   default=10000)
    p.add_argument("--eval_every",      type=int,   default=1000)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--eval_batch_size", type=int,   default=64)
    p.add_argument("--lr",              type=float, default=5e-5)
    p.add_argument("--lambda_kl",       type=float, default=0.2)
    p.add_argument("--lambda_delta",    type=float, default=3e-4)
    p.add_argument("--kl_topk",         type=int,   default=512)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--grad_accum_steps",type=int,   default=2,
                   help="Gradient accumulation. eff_bs = batch_size * grad_accum_steps.")
    # Flags
    p.add_argument("--amp",                      action="store_true")
    p.add_argument("--eval_before_train",        action="store_true")
    p.add_argument("--fail_on_baseline_mismatch",action="store_true")
    p.add_argument("--use_filtered_train_loader",action="store_true")
    p.add_argument("--train_covered_only",       action="store_true")
    p.add_argument("--device",                   default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
