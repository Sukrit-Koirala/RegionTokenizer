#!/usr/bin/env python3
"""
CTF Gold Leakage Diagnostic.

Evaluates a trained CTF checkpoint under 8 candidate-selection modes to
determine whether the large training gain (+0.255 at step 1000) is caused
by oracle gold force-inclusion in the selected candidate set.

CRITICAL TEST:
  MODE 1  (oracle_force_gold)  — reproduces current eval (gold inserted if missing)
  MODE 2  (no_force_gold)      — true inference: no oracle, top-M only

If MODE 1 >> MODE 2: LEAKAGE CONFIRMED
If MODE 1 ≈ MODE 2: the CTF is genuinely learning

Canonical invariants enforced on every full-val pass:
  fingerprint  = f57cabcdc46d69ce
  num_examples = 239,362
  num_covered  = 227,017
  coverage     = 0.948425

Usage:
    python scripts/debug_ctf_gold_leakage.py \\
        --small_ckpt  runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --val_dir     runs/path_refiner_clean/data/val_hgrid_K24 \\
        --baseline_json runs/path_refiner_clean/baselines/saved_candidate_baseline.json \\
        --ctf_ckpt    runs/path_refiner_candidate_transformer/variant_CTF_boundary_M256/best_refiner.pt \\
        --output_dir  runs/path_refiner_candidate_transformer/gold_leakage_debug \\
        --selected_M  256 --gate_filter boundary --batch_size 32
"""

import argparse
import contextlib
import csv
import glob
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.train_clean_path_refiner import load_r2s
from scripts.train_hard_position_refiner import EVAL_SUBSETS, compute_filter_mask
from scripts.train_candidate_transformer_refiner import CandidateSetTransformerRefiner


# ── Constants ──────────────────────────────────────────────────────────────────

EXPECTED_FP        = "f57cabcdc46d69ce"
EXPECTED_COVERAGE  = 0.948425
EXPECTED_N         = 239_362
EXPECTED_N_COV     = 227_017
BASELINE_NLL       = 3.378606   # force-zero covered NLL
HARD_MLP_GAIN      = 0.000530   # hard boundary MLP gated gain
GLOBAL_MLP_GAIN    = 0.001284   # global MLP best gated gain


# ── Canonical fingerprint (must match train_clean_path_refiner.py exactly) ────

def _compute_fingerprint(
    num_shards: int, total_n: int, total_cov: int,
    sum_cand_counts: int, sum_gold_idx_cov: int, sum_gold_tok: int,
) -> str:
    fp_data = {
        "num_shards":       num_shards,
        "total_n":          total_n,
        "total_cov":        total_cov,
        "sum_cand_counts":  sum_cand_counts,
        "sum_gold_idx_cov": sum_gold_idx_cov,
        "sum_gold_tok":     sum_gold_tok,
    }
    return hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()[:16]


def _assert_canonical(fp, coverage, num_examples, num_covered, mode_name, fail):
    issues = []
    if fp       != EXPECTED_FP:                            issues.append(f"fp {fp!r}")
    if abs(coverage - EXPECTED_COVERAGE) >= 1e-5:          issues.append(f"coverage {coverage:.6f}")
    if num_examples != EXPECTED_N:                         issues.append(f"num_examples {num_examples}")
    if num_covered  != EXPECTED_N_COV:                     issues.append(f"num_covered {num_covered}")
    if issues:
        msg = f"[{mode_name}] Canonical invariant FAIL: " + "; ".join(issues)
        if fail:
            raise RuntimeError(msg)
        print(f"  WARNING: {msg}")
        return False
    print(f"  [{mode_name}] canonical invariants OK  fp={fp}  cov={coverage:.6f}  "
          f"n={num_examples:,}  n_cov={num_covered:,}")
    return True


# ── Random-slot selection context manager ─────────────────────────────────────

@contextlib.contextmanager
def random_slot_selection(model, seed: int):
    """
    Temporarily patches model.select_candidates to insert forced-gold at a
    uniformly random slot (0..M-1) instead of always the last slot.

    If the model learned that the last slot = oracle gold, performance will
    degrade compared to oracle (last-slot) mode.
    """
    import types
    M   = model.selected_M
    rng = np.random.RandomState(seed)

    def _rand_select(self_m, base, cand_mask, gold_idx, covered):
        B, C   = base.shape
        m_act  = min(M, C)
        device = base.device

        _, sel_raw = base.topk(m_act, dim=-1, largest=True, sorted=True)
        msk_raw    = cand_mask.gather(1, sel_raw)

        if m_act < M:
            pad     = torch.zeros(B, M - m_act, dtype=torch.long,  device=device)
            pad_msk = torch.zeros(B, M - m_act, dtype=torch.bool,  device=device)
            sel_idx  = torch.cat([sel_raw, pad],     dim=1)
            sel_mask = torch.cat([msk_raw, pad_msk], dim=1)
        else:
            sel_idx  = sel_raw.clone()
            sel_mask = msk_raw.clone()

        gold_in_sel = (sel_idx == gold_idx.unsqueeze(1)).any(dim=1)
        gold_forced = covered & ~gold_in_sel

        if gold_forced.any():
            fb         = gold_forced.nonzero(as_tuple=True)[0].tolist()
            rand_slots = rng.randint(0, m_act, size=len(fb))
            for i, b in enumerate(fb):
                sel_idx[b, rand_slots[i]] = gold_idx[b]
                sel_mask[b, rand_slots[i]] = True

        return sel_idx, sel_mask, gold_forced

    orig                      = model.select_candidates
    model.select_candidates   = types.MethodType(_rand_select, model)
    try:
        yield model
    finally:
        model.select_candidates = orig


# ── Unified full-val evaluation ────────────────────────────────────────────────

@torch.no_grad()
def eval_mode_full_val(
    model,                          # CandidateSetTransformerRefiner or None
    val_dir:         str,
    tok_emb_w:       torch.Tensor,
    r2s_np:          np.ndarray,
    device,
    mode:            str,           # see MODE_* constants below
    selected_M:      int,
    gate_filter_name: str,
    eval_batch_size: int  = 64,
    fail_on_mismatch: bool = False,
    filter_kwargs:   dict = None,
) -> Dict:
    """
    Single unified evaluation loop for all leakage diagnostic modes.

    Mode strings:
      "force_zero"         — no model, base logits only
      "oracle_force_gold"  — CTF with gold force-inclusion (reproduces training eval)
      "no_force_gold"      — CTF top-M only, no oracle gold insertion
      "random_force_slot"  — CTF with random insertion slot (use random_slot_selection ctx mgr first)
      "forced_delta_zero"  — oracle selection, but delta zeroed for force-inserted gold slot
      "natural_only"       — no-force CTF, NLL only on positions where gold is in natural top-M
      "forced_only"        — no-force CTF, NLL only on positions where gold is NOT in top-M
      "forced_only_oracle" — oracle CTF, NLL only on positions where gold was force-inserted
      "shuffled_gold"      — oracle with shuffled gold_idx for insertion; true gold for CE

    Returns a dict with:
      nll, base_nll, n_eval, global_covered_nll, delta_vs_baseline,
      gated_covered_nll, inside_gate_model_nll, inside_gate_base_nll,
      outside_gate_model_nll, outside_gate_base_nll,
      gate_rate, coverage, fingerprint, num_examples, num_covered,
      natural_gold_in_topM_rate, gold_force_included_rate, forced_gold_count,
      mean_abs_delta, max_abs_delta, canonical_ok,
      subset_results: {subset: {n_total, n_covered, model_nll, base_nll, delta_nll,
                                model_acc1, model_acc5, base_acc1, base_acc5,
                                natural_gold_topM_rate, forced_gold_rate}}
    """
    if filter_kwargs is None:
        filter_kwargs = {}

    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt in {val_dir}")

    emb_w = tok_emb_w.float().to(device)
    if model is not None:
        model.eval()

    # Fingerprint counters (must exactly match canonical_eval_refiner.py)
    total_n = total_cov = 0
    sum_cand_counts = sum_gold_idx_cov = sum_gold_tok = 0

    # Global covered NLL (all covered positions — for canonical comparison)
    g_ce_m = g_ce_b = 0.0
    g_nc = 0

    # Mode-specific NLL (subset for natural_only / forced_only modes)
    m_ce_m = m_ce_b = 0.0
    m_nc = 0

    # Gated stats (model inside gate, base outside gate)
    ig_ce_m = ig_ce_b = og_ce_m = og_ce_b = 0.0
    ig_nc = ig_nt = og_nc = og_nt = 0

    # Gold tracking (over covered)
    nat_cnt = frc_cnt = 0

    # Delta magnitude stats
    d_sum = d_max = 0.0
    d_n = 0

    # Per-subset accumulators
    sub_acc = {s: {"n_total": 0, "n_cov": 0,
                   "m_ce": 0.0, "b_ce": 0.0,
                   "m_a1": 0, "m_a5": 0, "b_a1": 0, "b_a5": 0,
                   "nat": 0, "frc": 0}
               for s in EVAL_SUBSETS}

    for path in paths:
        shard    = torch.load(path, map_location="cpu", weights_only=True)
        N        = len(shard["covered"])
        has_type = "type_arr" in shard
        has_gtok = "gold_token" in shard

        sp_all = shard["split"].long()
        tp_all = (shard["type_arr"].long() if has_type
                  else torch.zeros(N, dtype=torch.long))

        # cand_super: map fine → super via r2s_np
        cf_np    = shard["cand_fine"].numpy().astype(np.int64).clip(min=0)
        cs_np    = r2s_np[cf_np].astype(np.int64)
        cs_np[shard["cand_fine"].numpy() < 0] = 0
        cs_shard = torch.from_numpy(cs_np)

        # Per-shard fingerprint contributions (covering ALL positions)
        sum_cand_counts += int((shard["cand_tok"] >= 0).sum())
        cov_shard = shard["covered"].bool()
        sum_gold_idx_cov += int(shard["gold_cand_idx"][cov_shard].sum())
        if has_gtok:
            sum_gold_tok += int(shard["gold_token"].long().sum())  # ALL, not just covered

        # Pre-compute subset and gate masks for this shard
        smasks = {}
        for sub in EVAL_SUBSETS:
            try:
                smasks[sub] = compute_filter_mask(shard, sub, **filter_kwargs)
            except (KeyError, ValueError):
                smasks[sub] = torch.zeros(N, dtype=torch.bool)
        gate_shard = compute_filter_mask(shard, gate_filter_name, **filter_kwargs)

        for start in range(0, N, eval_batch_size):
            end = min(start + eval_batch_size, N)

            h        = shard["h_prime"][start:end].float().to(device)
            ct       = shard["cand_tok"][start:end].long().to(device)
            cf       = shard["cand_fine"][start:end].long().to(device)
            cs       = cs_shard[start:end].long().to(device)
            g_idx    = shard["gold_cand_idx"][start:end].long().to(device)
            covered  = shard["covered"][start:end].bool().to(device)
            gate     = gate_shard[start:end].to(device)
            B, C     = ct.shape

            cmask    = (ct >= 0)
            tok_e    = F.embedding(ct.clamp(min=0), emb_w)
            base_raw = (h.unsqueeze(1) * tok_e).sum(-1)
            base_sc  = base_raw.masked_fill(~cmask, float("-inf"))

            r_reg    = shard["router_topk_reg"][start:end].long().to(device)
            r_prb    = shard["router_topk_prb"][start:end].float().to(device)
            m_reg    = shard["mem_topk_reg"][start:end].long().to(device)
            m_prb    = shard["mem_topk_prb"][start:end].float().to(device)
            r_margin = shard["router_margin"][start:end].float().to(device)
            m_margin = shard["mem_margin"][start:end].float().to(device)

            total_n   += B
            total_cov += int(covered.sum())

            # Per-batch gold selection flags
            gold_forced_b   = torch.zeros(B, dtype=torch.bool, device=device)
            natural_in_topM = torch.zeros(B, dtype=torch.bool, device=device)

            # ── Mode dispatch ────────────────────────────────────────────────
            if mode == "force_zero":
                mdl_sc = base_sc

            elif mode == "oracle_force_gold":
                mdl_sc, _ = model(
                    h, ct, cf, cs, cmask, emb_w,
                    r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
                    gold_idx=g_idx, covered=covered,
                )
                gf = model._last_sel_stats.get("gold_forced")
                if gf is not None:
                    gold_forced_b = gf
                natural_in_topM = covered & ~gold_forced_b

            elif mode in ("no_force_gold", "natural_only", "forced_only"):
                mdl_sc, _ = model(
                    h, ct, cf, cs, cmask, emb_w,
                    r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
                    gold_idx=None, covered=None,   # ← disables force-inclusion
                )
                m_act = min(selected_M, C)
                _, sel_top = base_sc.topk(m_act, dim=-1, largest=True, sorted=True)
                gold_in_top = (sel_top == g_idx.unsqueeze(1)).any(dim=1)
                natural_in_topM = covered & gold_in_top
                gold_forced_b   = covered & ~gold_in_top  # "would have been forced" flag

            elif mode == "random_force_slot":
                # model.select_candidates has already been patched by the caller
                mdl_sc, _ = model(
                    h, ct, cf, cs, cmask, emb_w,
                    r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
                    gold_idx=g_idx, covered=covered,
                )
                gf = model._last_sel_stats.get("gold_forced")
                if gf is not None:
                    gold_forced_b = gf
                natural_in_topM = covered & ~gold_forced_b

            elif mode == "forced_delta_zero":
                mdl_sc, _ = model(
                    h, ct, cf, cs, cmask, emb_w,
                    r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
                    gold_idx=g_idx, covered=covered,
                )
                gf = model._last_sel_stats.get("gold_forced")
                if gf is not None:
                    gold_forced_b = gf
                natural_in_topM = covered & ~gold_forced_b
                # Zero the delta on forced-gold slots (gold is always at cand_idx = g_idx[b])
                if gold_forced_b.any():
                    fb      = gold_forced_b.nonzero(as_tuple=True)[0]
                    mdl_sc  = mdl_sc.clone()
                    mdl_sc[fb, g_idx[fb]] = base_sc[fb, g_idx[fb]]

            elif mode == "forced_only_oracle":
                mdl_sc, _ = model(
                    h, ct, cf, cs, cmask, emb_w,
                    r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
                    gold_idx=g_idx, covered=covered,
                )
                gf = model._last_sel_stats.get("gold_forced")
                if gf is not None:
                    gold_forced_b = gf
                natural_in_topM = covered & ~gold_forced_b

            elif mode == "shuffled_gold":
                # Shuffle gold indices within the batch for selection only
                perm = torch.randperm(B, device=device)
                shuffled_g = g_idx[perm]
                mdl_sc, _ = model(
                    h, ct, cf, cs, cmask, emb_w,
                    r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
                    gold_idx=shuffled_g, covered=covered,
                )
                gf = model._last_sel_stats.get("gold_forced")
                if gf is not None:
                    gold_forced_b = gf          # forced using wrong (shuffled) gold
                # "natural" = covered positions where true gold happens to be in top-M
                m_act = min(selected_M, C)
                _, sel_top = base_sc.topk(m_act, dim=-1, largest=True, sorted=True)
                gold_in_top = (sel_top == g_idx.unsqueeze(1)).any(dim=1)
                natural_in_topM = covered & gold_in_top
                # NB: CE below uses true g_idx, not shuffled_g

            else:
                raise ValueError(f"Unknown mode: {mode!r}")

            # ── Determine covered mask for this mode's primary NLL ───────────
            if mode == "natural_only":
                eval_cov = natural_in_topM
            elif mode in ("forced_only", "forced_only_oracle"):
                eval_cov = gold_forced_b
            else:
                eval_cov = covered

            n_cov_b  = int(covered.sum())
            n_eval_b = int(eval_cov.sum())

            # Global NLL (always over ALL covered — for canonical / main comparison)
            if n_cov_b > 0:
                ar  = torch.arange(n_cov_b, device=device)
                gi  = g_idx[covered]
                lpm = F.log_softmax(mdl_sc[covered], dim=-1)
                lpb = F.log_softmax(base_sc[covered], dim=-1)
                g_ce_m += float(-lpm[ar, gi].sum())
                g_ce_b += float(-lpb[ar, gi].sum())
                g_nc   += n_cov_b

            # Mode-specific NLL (subset modes only)
            if mode in ("natural_only", "forced_only", "forced_only_oracle") and n_eval_b > 0:
                ar  = torch.arange(n_eval_b, device=device)
                gi  = g_idx[eval_cov]
                lpm = F.log_softmax(mdl_sc[eval_cov], dim=-1)
                lpb = F.log_softmax(base_sc[eval_cov], dim=-1)
                m_ce_m += float(-lpm[ar, gi].sum())
                m_ce_b += float(-lpb[ar, gi].sum())
                m_nc   += n_eval_b

            # Gated stats (inside gate uses model, outside gate uses base)
            if n_cov_b > 0:
                gate_cov  = gate[covered]          # (n_cov_b,) bool
                n_ig      = int(gate_cov.sum())
                n_og      = n_cov_b - n_ig
                ig_nt    += int(gate.sum())
                ig_nc    += n_ig
                og_nt    += B - int(gate.sum())
                og_nc    += n_og

                mdl_c = mdl_sc[covered]
                bas_c = base_sc[covered]
                gi_c  = g_idx[covered]

                if n_ig > 0:
                    ar_ig    = torch.arange(n_ig, device=device)
                    ig_ce_m += float(-F.log_softmax(mdl_c[gate_cov],  dim=-1)[ar_ig, gi_c[gate_cov]].sum())
                    ig_ce_b += float(-F.log_softmax(bas_c[gate_cov],  dim=-1)[ar_ig, gi_c[gate_cov]].sum())
                if n_og > 0:
                    ar_og    = torch.arange(n_og, device=device)
                    og_ce_m += float(-F.log_softmax(mdl_c[~gate_cov], dim=-1)[ar_og, gi_c[~gate_cov]].sum())
                    og_ce_b += float(-F.log_softmax(bas_c[~gate_cov], dim=-1)[ar_og, gi_c[~gate_cov]].sum())

            # Gold tracking (over covered)
            nat_cnt += int(natural_in_topM.sum())
            frc_cnt += int(gold_forced_b.sum())

            # Delta magnitude
            if mode != "force_zero" and model is not None and cmask.any():
                dv    = (mdl_sc - base_sc)[cmask]
                d_sum += float(dv.abs().sum())
                d_max  = max(d_max, float(dv.abs().max()))
                d_n   += int(cmask.sum())

            # Per-subset accumulation
            k5 = min(5, C)
            for sub in EVAL_SUBSETS:
                smask_b = smasks[sub][start:end].to(device)
                sub_acc[sub]["n_total"] += int(smask_b.sum())

                cov_s = covered & smask_b
                nc_s  = int(cov_s.sum())
                sub_acc[sub]["n_cov"] += nc_s
                if nc_s == 0:
                    continue

                ar_s = torch.arange(nc_s, device=device)
                gi_s = g_idx[cov_s]
                sub_acc[sub]["m_ce"] += float(-F.log_softmax(mdl_sc[cov_s],  dim=-1)[ar_s, gi_s].sum())
                sub_acc[sub]["b_ce"] += float(-F.log_softmax(base_sc[cov_s], dim=-1)[ar_s, gi_s].sum())

                m5m = mdl_sc[cov_s].topk(k5, dim=-1).indices
                m5b = base_sc[cov_s].topk(k5, dim=-1).indices
                gi_e = gi_s.unsqueeze(1)
                sub_acc[sub]["m_a1"] += int((m5m[:, :1] == gi_e).any(1).sum())
                sub_acc[sub]["m_a5"] += int((m5m         == gi_e).any(1).sum())
                sub_acc[sub]["b_a1"] += int((m5b[:, :1] == gi_e).any(1).sum())
                sub_acc[sub]["b_a5"] += int((m5b         == gi_e).any(1).sum())
                sub_acc[sub]["nat"]  += int((natural_in_topM & cov_s).sum())
                sub_acc[sub]["frc"]  += int((gold_forced_b   & cov_s).sum())

    if model is not None:
        model.train()

    # ── Canonical check ───────────────────────────────────────────────────────
    fp = _compute_fingerprint(
        len(paths), total_n, total_cov, sum_cand_counts, sum_gold_idx_cov, sum_gold_tok)
    coverage    = total_cov / max(total_n, 1)
    canonical_ok = _assert_canonical(fp, coverage, total_n, total_cov, mode, fail_on_mismatch)

    # ── Primary NLL ───────────────────────────────────────────────────────────
    # For subset modes, primary NLL = subset NLL; for all others = global covered NLL
    is_subset = mode in ("natural_only", "forced_only", "forced_only_oracle")
    if is_subset and m_nc > 0:
        prim_nll_m = m_ce_m / m_nc
        prim_nll_b = m_ce_b / m_nc
        prim_nc    = m_nc
    else:
        prim_nll_m = g_ce_m / max(g_nc, 1)
        prim_nll_b = g_ce_b / max(g_nc, 1)
        prim_nc    = g_nc

    global_nll_m = g_ce_m / max(g_nc, 1)
    global_nll_b = g_ce_b / max(g_nc, 1)

    # ── Gated covered NLL: model inside gate + base outside gate ─────────────
    gated_nll_m = (ig_ce_m + og_ce_b) / max(ig_nc + og_nc, 1)

    # ── Subset results ────────────────────────────────────────────────────────
    subset_results = {}
    for sub, s in sub_acc.items():
        nc = s["n_cov"]
        subset_results[sub] = {
            "n_total":              s["n_total"],
            "n_covered":            nc,
            "model_nll":            s["m_ce"] / max(nc, 1),
            "base_nll":             s["b_ce"] / max(nc, 1),
            "delta_nll":            (s["b_ce"] - s["m_ce"]) / max(nc, 1),
            "model_acc1":           s["m_a1"] / max(nc, 1),
            "model_acc5":           s["m_a5"] / max(nc, 1),
            "base_acc1":            s["b_a1"] / max(nc, 1),
            "base_acc5":            s["b_a5"] / max(nc, 1),
            "natural_gold_topM_rate": s["nat"] / max(nc, 1),
            "forced_gold_rate":     s["frc"] / max(nc, 1),
        }

    return {
        "mode":                     mode,
        "nll":                      prim_nll_m,
        "base_nll":                 prim_nll_b,
        "n_eval":                   prim_nc,
        "global_covered_nll":       global_nll_m,
        "delta_vs_baseline":        BASELINE_NLL - prim_nll_m,
        "gated_covered_nll":        gated_nll_m,
        "delta_gated_vs_baseline":  BASELINE_NLL - gated_nll_m,
        "coverage":                 coverage,
        "fingerprint":              fp,
        "num_examples":             total_n,
        "num_covered":              total_cov,
        "gate_rate":                ig_nt / max(total_n, 1),
        "inside_gate_model_nll":    ig_ce_m / max(ig_nc, 1),
        "inside_gate_base_nll":     ig_ce_b / max(ig_nc, 1),
        "outside_gate_model_nll":   og_ce_m / max(og_nc, 1),
        "outside_gate_base_nll":    og_ce_b / max(og_nc, 1),
        "natural_gold_in_topM_rate": nat_cnt / max(total_cov, 1),
        "gold_force_included_rate": frc_cnt / max(total_cov, 1),
        "forced_gold_count":        frc_cnt,
        "mean_abs_delta":           d_sum / max(d_n, 1) if d_n > 0 else 0.0,
        "max_abs_delta":            d_max,
        "canonical_ok":             canonical_ok,
        "subset_results":           subset_results,
    }


# ── Multi-seed random-slot runner (Mode 3) ────────────────────────────────────

def run_random_slot_modes(model, val_dir, tok_emb_w, r2s_np, device,
                          selected_M, gate_filter_name, eval_batch_size,
                          fail_on_mismatch, filter_kwargs,
                          seeds=(1, 2, 3, 4, 5)) -> Dict:
    """Run Mode 3 with multiple seeds; report mean ± std NLL."""
    per_seed = {}
    for seed in seeds:
        print(f"    random_force_slot seed={seed} ...")
        with random_slot_selection(model, seed):
            r = eval_mode_full_val(
                model, val_dir, tok_emb_w, r2s_np, device,
                mode="random_force_slot",
                selected_M=selected_M, gate_filter_name=gate_filter_name,
                eval_batch_size=eval_batch_size,
                fail_on_mismatch=fail_on_mismatch,
                filter_kwargs=filter_kwargs,
            )
        per_seed[seed] = r
        print(f"      nll={r['nll']:.6f}  gated={r['gated_covered_nll']:.6f}  "
              f"delta={r['delta_vs_baseline']:+.6f}")

    nlls   = [per_seed[s]["nll"]              for s in seeds]
    gateds = [per_seed[s]["gated_covered_nll"] for s in seeds]

    combined = per_seed[seeds[0]].copy()
    combined["mode"]               = "random_force_slot"
    combined["nll"]                = float(np.mean(nlls))
    combined["nll_std"]            = float(np.std(nlls))
    combined["gated_covered_nll"]  = float(np.mean(gateds))
    combined["gated_std"]          = float(np.std(gateds))
    combined["delta_vs_baseline"]  = BASELINE_NLL - float(np.mean(nlls))
    combined["delta_gated_vs_baseline"] = BASELINE_NLL - float(np.mean(gateds))
    combined["nll_per_seed"]       = {f"seed_{s}": nlls[i] for i, s in enumerate(seeds)}
    combined["gated_per_seed"]     = {f"seed_{s}": gateds[i] for i, s in enumerate(seeds)}
    return combined


# ── Verdict ────────────────────────────────────────────────────────────────────

def determine_verdict(oracle_r: Dict, no_force_r: Dict,
                      forced_oracle_r: Optional[Dict],
                      forced_noforce_r: Optional[Dict]) -> str:
    """
    Rules (operating on gated_covered_nll gain):
      LEAKAGE CONFIRMED  if oracle_gain - no_force_gain >= 0.05
                         OR (forced-only oracle_gain large & forced-only no_force_gain ≈ 0)
      LEAKAGE LIKELY     if oracle_gain - no_force_gain >= 0.01
      NO LEAKAGE EVIDENT if gap < 0.01 AND no_force still beats global MLP
      INCONCLUSIVE       otherwise
    """
    oracle_gain   = BASELINE_NLL - oracle_r["gated_covered_nll"]
    no_force_gain = BASELINE_NLL - no_force_r["gated_covered_nll"]
    gap = oracle_gain - no_force_gain

    forced_leakage = False
    if forced_oracle_r is not None and forced_noforce_r is not None:
        fo_gain  = forced_oracle_r["delta_vs_baseline"]
        fn_gain  = forced_noforce_r["delta_vs_baseline"]
        forced_leakage = fo_gain > 0.1 and fn_gain < fo_gain * 0.1

    if gap >= 0.05 or forced_leakage:
        return "LEAKAGE CONFIRMED"
    if gap >= 0.01:
        return "LEAKAGE LIKELY"
    if gap < 0.01 and no_force_gain > GLOBAL_MLP_GAIN:
        return "NO LEAKAGE EVIDENT"
    return "INCONCLUSIVE"


# ── Report generation ──────────────────────────────────────────────────────────

def write_report(results: Dict[str, Dict], output_dir: str, args) -> None:
    """Write gold_leakage_metrics.csv, gold_leakage_subset_metrics.csv, gold_leakage_report.md."""

    oracle_r   = results.get("oracle_force_gold")
    no_force_r = results.get("no_force_gold")

    # ── Main metrics CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "gold_leakage_metrics.csv")
    fields = ["mode", "nll", "delta_vs_baseline", "gated_covered_nll",
              "delta_gated_vs_baseline", "global_covered_nll",
              "n_eval", "natural_gold_in_topM_rate", "gold_force_included_rate",
              "forced_gold_count", "inside_gate_model_nll", "inside_gate_base_nll",
              "outside_gate_model_nll", "outside_gate_base_nll",
              "gate_rate", "mean_abs_delta", "max_abs_delta",
              "fingerprint", "coverage", "num_examples", "num_covered", "canonical_ok"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for mode_name, r in results.items():
            row = {k: r.get(k, "") for k in fields}
            row["mode"] = mode_name
            # Flatten nll_std / gated_std for random slot
            if "nll_std" in r:
                row["nll"] = f"{r['nll']:.6f}±{r['nll_std']:.6f}"
            w.writerow(row)
    print(f"  wrote {csv_path}")

    # ── Subset metrics CSV ────────────────────────────────────────────────────
    sub_csv = os.path.join(output_dir, "gold_leakage_subset_metrics.csv")
    sub_fields = ["mode", "subset", "n_covered", "base_nll", "model_nll", "delta_nll",
                  "base_acc1", "model_acc1", "base_acc5", "model_acc5",
                  "natural_gold_topM_rate", "forced_gold_rate"]
    with open(sub_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sub_fields, extrasaction="ignore")
        w.writeheader()
        for mode_name, r in results.items():
            for sub, sr in r.get("subset_results", {}).items():
                row = {"mode": mode_name, "subset": sub}
                row.update(sr)
                w.writerow(row)
    print(f"  wrote {sub_csv}")

    # ── Markdown report ───────────────────────────────────────────────────────
    verdict = "UNKNOWN"
    if oracle_r and no_force_r:
        verdict = determine_verdict(
            oracle_r, no_force_r,
            results.get("forced_only_oracle"),
            results.get("forced_only"),
        )

    md_path = os.path.join(output_dir, "gold_leakage_report.md")
    lines = []
    lines += [
        "# CTF Gold Leakage Diagnostic Report",
        "",
        "## Setup",
        f"- CTF checkpoint : `{args.ctf_ckpt}`",
        f"- Gate filter    : `{args.gate_filter}`",
        f"- Selected M     : `{args.selected_M}`",
        f"- Val dir        : `{args.val_dir}`",
        "",
        "## Baseline (force-zero)",
        f"- `covered_nll        = {BASELINE_NLL:.6f}`",
        f"- `hard_MLP_boundary_gain = +{HARD_MLP_GAIN:.6f}`",
        f"- `global_MLP_gain        = +{GLOBAL_MLP_GAIN:.6f}`",
        "",
        "## Main comparison (primary metric: gated_covered_nll)",
        "",
        "| mode | gated_nll | gain | natural_topM | force_rate | note |",
        "|------|----------:|-----:|-------------:|-----------:|------|",
    ]

    oracle_labels = {"oracle_force_gold", "random_force_slot", "forced_delta_zero",
                     "forced_only_oracle", "shuffled_gold"}
    mode_order = [
        "force_zero", "oracle_force_gold", "no_force_gold",
        "random_force_slot", "forced_delta_zero",
        "natural_only", "forced_only", "forced_only_oracle", "shuffled_gold",
    ]
    for mn in mode_order:
        if mn not in results:
            continue
        r    = results[mn]
        gain = r.get("delta_gated_vs_baseline", BASELINE_NLL - r.get("gated_covered_nll", BASELINE_NLL))
        note = "ORACLE / NOT INFERENCE VALID" if mn in oracle_labels else ""
        if mn == "force_zero":
            note = "anchor"
        elif mn == "no_force_gold":
            note = "**TRUE INFERENCE METRIC**"
        nll_s  = f"{r.get('gated_covered_nll', r.get('nll', '?')):.6f}"
        nat_s  = f"{r.get('natural_gold_in_topM_rate', 0):.4f}"
        frc_s  = f"{r.get('gold_force_included_rate', 0):.4f}"
        lines.append(f"| {mn} | {nll_s} | {gain:+.6f} | {nat_s} | {frc_s} | {note} |")

    if oracle_r and no_force_r:
        o_gain = BASELINE_NLL - oracle_r["gated_covered_nll"]
        n_gain = BASELINE_NLL - no_force_r["gated_covered_nll"]
        gap    = o_gain - n_gain
        lines += [
            "",
            "## Leakage analysis",
            "",
            f"| metric | value |",
            f"|--------|------:|",
            f"| oracle_gain  | {o_gain:+.6f} |",
            f"| no_force_gain | {n_gain:+.6f} |",
            f"| gap = oracle - no_force | {gap:+.6f} |",
            f"| hard_MLP_boundary_gain | +{HARD_MLP_GAIN:.6f} |",
            f"| global_MLP_gain | +{GLOBAL_MLP_GAIN:.6f} |",
            "",
        ]

    if results.get("forced_only_oracle") and results.get("forced_only"):
        fo = results["forced_only_oracle"]
        fn = results["forced_only"]
        lines += [
            "## Forced-positions analysis",
            "",
            "Positions where gold was NOT naturally in top-M:",
            "",
            f"| mode | nll | gain | n_eval |",
            f"|------|----:|-----:|-------:|",
            f"| forced_only_oracle | {fo['nll']:.6f} | {fo['delta_vs_baseline']:+.6f} | {fo['n_eval']:,} |",
            f"| forced_only (no_force selection) | {fn['nll']:.6f} | {fn['delta_vs_baseline']:+.6f} | {fn['n_eval']:,} |",
            "",
            "If oracle_gain >> no_force_gain on forced positions, the model directly exploits gold.",
            "",
        ]

    lines += [
        "## Subset comparison (boundary)",
        "",
        "| mode | n_cov | base_nll | model_nll | delta | acc@1 | base_acc@1 |",
        "|------|------:|---------:|----------:|------:|------:|-----------:|",
    ]
    for mn in ["force_zero", "oracle_force_gold", "no_force_gold"]:
        if mn not in results:
            continue
        sr = results[mn].get("subset_results", {}).get("boundary", {})
        if not sr:
            continue
        lines.append(
            f"| {mn} | {sr['n_covered']:,} | {sr['base_nll']:.4f} | "
            f"{sr['model_nll']:.4f} | {sr['delta_nll']:+.4f} | "
            f"{sr['model_acc1']:.3f} | {sr['base_acc1']:.3f} |"
        )

    lines += [
        "",
        f"## Verdict",
        "",
        f"**{verdict}**",
        "",
    ]

    if verdict == "LEAKAGE CONFIRMED":
        lines += [
            "The large +0.255 gain is caused by oracle gold insertion.",
            "The model learns to score the last (forced-gold) slot highly.",
            "",
            "**Action required**: disable `gold_idx`/`covered` in all validation evals.",
            "Re-evaluate using `no_force_gold` mode as the canonical metric.",
            "Retrain if needed.",
        ]
    elif verdict == "LEAKAGE LIKELY":
        lines += [
            "Moderate leakage signal. Oracle significantly outperforms no-force eval.",
            "**Recommended**: switch default eval to `no_force_gold` and retrain.",
        ]
    elif verdict == "NO LEAKAGE EVIDENT":
        lines += [
            "no_force_gold gain is close to oracle gain and beats MLP baselines.",
            "The CTF is genuinely learning to rerank candidates.",
            "Continue training with `no_force_gold` as the canonical metric.",
        ]
    else:
        lines += [
            "Results are ambiguous. Check per-seed random_force_slot variance.",
            "Consider running more training steps before re-diagnosis.",
        ]

    lines += ["", "---", ""]
    lines += [
        "## Notes",
        "",
        "From this diagnostic forward, `eval_force_include_gold = false` is the default.",
        "Oracle/force-gold evals must be labeled **ORACLE / NOT INFERENCE VALID**.",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {md_path}")


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_ctf_checkpoint(ctf_ckpt_arg: str, output_dir_or_var_dir: str,
                        d_model: int, n_fine: int, n_super: int,
                        device) -> "CandidateSetTransformerRefiner":
    """
    Load CTF checkpoint. Supports:
      --ctf_ckpt path/to/best_refiner.pt
      --ctf_ckpt latest   → searches for best_refiner.pt then last_refiner.pt
    """
    if ctf_ckpt_arg == "latest":
        for name in ("best_refiner.pt", "last_refiner.pt"):
            p = os.path.join(output_dir_or_var_dir, name)
            if os.path.isfile(p):
                ctf_ckpt_arg = p
                break
        if ctf_ckpt_arg == "latest":
            raise FileNotFoundError(
                f"No best_refiner.pt or last_refiner.pt in {output_dir_or_var_dir}")

    if not os.path.isfile(ctf_ckpt_arg):
        raise FileNotFoundError(f"CTF checkpoint not found: {ctf_ckpt_arg}")

    print(f"[load] CTF checkpoint: {ctf_ckpt_arg}")
    ckpt = torch.load(ctf_ckpt_arg, map_location="cpu", weights_only=True)

    saved_args = ckpt.get("args", {})
    step       = ckpt.get("step", "?")
    metrics    = ckpt.get("metrics", {})
    print(f"  step={step}  saved_args keys={list(saved_args.keys())[:8]}")
    if metrics:
        gnll = metrics.get("gated_covered_nll", metrics.get("covered_nll", "?"))
        print(f"  checkpoint metrics: gated_nll={gnll}")

    model = CandidateSetTransformerRefiner(
        d_model=d_model,
        n_fine=n_fine,
        n_super=n_super,
        refiner_dim=saved_args.get("refiner_dim", 256),
        num_layers=saved_args.get("num_layers",  2),
        num_heads=saved_args.get("num_heads",    4),
        ff_mult=saved_args.get("ff_mult",        4),
        dropout=saved_args.get("dropout",        0.0),
        selected_M=saved_args.get("selected_M",  256),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  WARNING: missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys in checkpoint: {unexpected}")

    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def run_diagnostic(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ── Load backbone + token embeddings ──────────────────────────────────────
    print(f"[main] loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, _ = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().cpu()
    elif hasattr(backbone, "transformer"):
        tok_emb_w = backbone.transformer.wte.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot locate token embedding in backbone")

    # ── Dataset config ────────────────────────────────────────────────────────
    cfg_path = os.path.join(args.val_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            ds_cfg = json.load(f)
        n_fine  = ds_cfg.get("n_fine",  128)
        n_super = ds_cfg.get("n_super",  24)
    else:
        n_fine = n_super = 128

    super_map = getattr(args, "super_map", None)
    r2s_np = np.zeros(n_fine, dtype=np.int32)
    if super_map and os.path.isfile(super_map):
        r2s_np  = load_r2s(super_map, n_fine)
        n_super = int(r2s_np.max()) + 1
    print(f"[main] n_fine={n_fine}  n_super={n_super}")

    # ── Load CTF model ────────────────────────────────────────────────────────
    # Resolve "latest" relative to the directory containing best_refiner.pt
    ckpt_dir = os.path.dirname(args.ctf_ckpt) if args.ctf_ckpt != "latest" else os.path.join(
        os.path.dirname(args.output_dir), "variant_CTF_boundary_M256")
    model = load_ctf_checkpoint(
        args.ctf_ckpt, ckpt_dir, d_model, n_fine, n_super, device)
    model.register_buffer("_tok_emb_w", tok_emb_w.float().to(device))

    # ── Load official baseline ────────────────────────────────────────────────
    baseline: Optional[Dict] = None
    if os.path.isfile(args.baseline_json):
        with open(args.baseline_json) as f:
            baseline = json.load(f)
        print(f"[main] baseline: nll={baseline['covered_nll']:.6f}  "
              f"fp={baseline['dataset_fingerprint']}")

    filter_kwargs = {}
    eval_batch    = args.batch_size
    selected_M    = args.selected_M
    gate_filter   = args.gate_filter
    tok_dev       = tok_emb_w.to(device)
    fail          = args.fail_on_baseline_mismatch

    results: Dict[str, Dict] = {}

    def _run(mode_name: str, label: str = "") -> Dict:
        tag = label or mode_name
        print(f"\n{'='*60}")
        print(f"  MODE: {tag}")
        print(f"{'='*60}")
        t0 = time.time()
        r  = eval_mode_full_val(
            model if mode_name != "force_zero" else None,
            args.val_dir, tok_dev, r2s_np, device,
            mode=mode_name, selected_M=selected_M,
            gate_filter_name=gate_filter,
            eval_batch_size=eval_batch,
            fail_on_mismatch=fail,
            filter_kwargs=filter_kwargs,
        )
        print(f"  nll={r['nll']:.6f}  gated={r['gated_covered_nll']:.6f}  "
              f"delta={r['delta_vs_baseline']:+.6f}  "
              f"nat_topM={r['natural_gold_in_topM_rate']:.4f}  "
              f"force_rate={r['gold_force_included_rate']:.4f}  "
              f"t={time.time()-t0:.0f}s")
        for sub in ["boundary", "hard_union", "router_top8_miss", "type_A"]:
            sr = r["subset_results"].get(sub, {})
            if sr.get("n_covered", 0) > 0:
                print(f"    {sub:30s}  model={sr['model_nll']:.4f}  "
                      f"base={sr['base_nll']:.4f}  delta={sr['delta_nll']:+.4f}  "
                      f"acc@1={sr['model_acc1']:.3f}  (base {sr['base_acc1']:.3f})")
        return r

    # ── Mode 0: force_zero ────────────────────────────────────────────────────
    results["force_zero"] = _run("force_zero")

    # ── Mode 1: oracle_force_gold ─────────────────────────────────────────────
    print("\n  *** ORACLE / NOT INFERENCE VALID ***")
    results["oracle_force_gold"] = _run("oracle_force_gold")

    # ── Mode 2: no_force_gold (TRUE INFERENCE METRIC) ─────────────────────────
    print("\n  *** TRUE INFERENCE METRIC ***")
    results["no_force_gold"] = _run("no_force_gold")

    # ── Mode 3: random_force_slot (5 seeds) ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MODE: random_force_slot (5 seeds) — ORACLE")
    print(f"{'='*60}")
    t0 = time.time()
    results["random_force_slot"] = run_random_slot_modes(
        model, args.val_dir, tok_dev, r2s_np, device,
        selected_M, gate_filter, eval_batch, fail, filter_kwargs,
    )
    r = results["random_force_slot"]
    print(f"  mean_nll={r['nll']:.6f}±{r.get('nll_std',0):.6f}  "
          f"mean_gated={r['gated_covered_nll']:.6f}±{r.get('gated_std',0):.6f}  "
          f"t={time.time()-t0:.0f}s")

    # ── Mode 4a: forced_delta_zero ────────────────────────────────────────────
    results["forced_delta_zero"] = _run("forced_delta_zero")

    # ── Mode 5: naturally_selected_gold_only ──────────────────────────────────
    results["natural_only"] = _run("natural_only", "natural_only (gold in top-M)")

    # ── Mode 6a: forced_only_oracle ───────────────────────────────────────────
    results["forced_only_oracle"] = _run("forced_only_oracle",
                                         "forced_only (oracle scores, forced-position subset)")

    # ── Mode 6b: forced_only (no-force scores, same subset) ──────────────────
    results["forced_only"] = _run("forced_only",
                                  "forced_only (no-force scores, forced-position subset)")

    # ── Mode 7: shuffled_gold_control ─────────────────────────────────────────
    results["shuffled_gold"] = _run("shuffled_gold",
                                    "shuffled_gold (wrong candidate inserted, true gold CE)")

    # ── Print verdict ─────────────────────────────────────────────────────────
    oracle_r   = results["oracle_force_gold"]
    no_force_r = results["no_force_gold"]
    o_gain = BASELINE_NLL - oracle_r["gated_covered_nll"]
    n_gain = BASELINE_NLL - no_force_r["gated_covered_nll"]
    gap    = o_gain - n_gain
    verdict = determine_verdict(
        oracle_r, no_force_r,
        results.get("forced_only_oracle"),
        results.get("forced_only"),
    )

    print(f"\n{'='*60}")
    print("  GOLD LEAKAGE DIAGNOSTIC — SUMMARY")
    print(f"{'='*60}")
    print(f"  baseline NLL            = {BASELINE_NLL:.6f}")
    print(f"  oracle_force_gold  gain = {o_gain:+.6f}  "
          f"(gated_nll={oracle_r['gated_covered_nll']:.6f})")
    print(f"  no_force_gold gain      = {n_gain:+.6f}  "
          f"(gated_nll={no_force_r['gated_covered_nll']:.6f})")
    print(f"  gap (oracle - no_force) = {gap:+.6f}")
    print(f"  hard_MLP_boundary_gain  = +{HARD_MLP_GAIN:.6f}")
    print(f"  global_MLP_gain         = +{GLOBAL_MLP_GAIN:.6f}")
    print()
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print()
    write_report(results, args.output_dir, args)

    all_modes_json = os.path.join(args.output_dir, "all_modes.json")
    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk != "subset_results"}
        for k, v in results.items()
    }
    with open(all_modes_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  wrote {all_modes_json}")
    print(f"\nDiagnostic complete. Outputs in: {args.output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="CTF gold leakage diagnostic")
    p.add_argument("--small_ckpt",   required=True)
    p.add_argument("--val_dir",      required=True)
    p.add_argument("--baseline_json",required=True)
    p.add_argument("--ctf_ckpt",     required=True,
                   help="Path to best_refiner.pt, or 'latest' to auto-detect")
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--super_map",    default=None)
    p.add_argument("--selected_M",   type=int, default=256)
    p.add_argument("--gate_filter",  default="boundary")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--fail_on_baseline_mismatch", action="store_true")
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run_diagnostic(_parse())
