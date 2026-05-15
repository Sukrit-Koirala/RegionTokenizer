#!/usr/bin/env python3
"""
Hierarchical candidate policy tuning — Tasks 1-6.

Loads per_position.npz + pre-built region_to_superregion_K{K}.json files from
the hard_memory_predictive_hierarchy run, evaluates a comprehensive grid of
candidate-selection policies, and writes the Pareto frontier + tuned report.

Fast: skips Part-1 (controller grid) and Part-2 (clustering). Only runs the
policy evaluation arithmetic (all numpy, no GPU needed).

Outputs (all in --output_dir):
    tuned_policy_results.csv        all evaluated policies
    tuned_pareto_frontier.csv       Pareto-optimal subset
    tuned_final_report.md           best policies per vocab budget

Usage:
    python scripts/tune_hier_policies.py \\
        --knn_run_dir  runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20 \\
        --hier_dir     runs/hard_memory_predictive_hierarchy \\
        --region_map   runs/region_maps_128/token_to_region.json \\
        --output_dir   runs/hard_memory_predictive_hierarchy
"""

import argparse
import csv
import datetime
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np

N_FINE       = 128
FULL_VOCAB   = 50257
SUPER_K_LIST = [8, 16, 24, 32, 48, 64]

# ── I/O ───────────────────────────────────────────────────────────────────────

def _csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    seen: set = set()
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

def _flt(d: dict, k: str, default: float = float("nan")) -> float:
    try:
        return float(d[k])
    except Exception:
        return default

# ── Array helpers ─────────────────────────────────────────────────────────────

def scatter_indicator(top_regions: np.ndarray, K: int, n_fine: int = N_FINE) -> np.ndarray:
    N, K_max = top_regions.shape
    K_eff = min(K, K_max)
    I = np.zeros((N, n_fine), dtype=bool)
    rows = np.repeat(np.arange(N), K_eff)
    cols = top_regions[:, :K_eff].ravel().astype(np.int32)
    I[rows, cols] = True
    return I

def tok_vec(top_regions: np.ndarray, K: int, tpr: np.ndarray) -> np.ndarray:
    K_eff = min(K, top_regions.shape[1])
    return tpr[top_regions[:, :K_eff].astype(np.int32)].sum(axis=1)

def overlap_any(a: np.ndarray, Ka: int, b: np.ndarray, Kb: int) -> np.ndarray:
    return np.any(
        a[:, :Ka, np.newaxis].astype(np.int32) == b[:, np.newaxis, :Kb].astype(np.int32),
        axis=(1, 2),
    )

# ── Data loading ──────────────────────────────────────────────────────────────

def load_per_position(knn_run_dir: str) -> Dict[str, np.ndarray]:
    for name in ("per_position.npz", "per_position_topk.npz"):
        p = os.path.join(knn_run_dir, name)
        if os.path.exists(p):
            d = dict(np.load(p))
            print(f"[load] {p}  N={len(d['gold_region']):,}")
            return d
    raise FileNotFoundError(f"per_position.npz not found in {knn_run_dir}")

def load_tpr(region_map_path: str, n_fine: int = N_FINE) -> np.ndarray:
    with open(region_map_path) as f:
        rm = json.load(f)
    tpr = np.zeros(n_fine, dtype=np.float32)
    for _, rid in rm.items():
        r = int(rid)
        if 0 <= r < n_fine:
            tpr[r] += 1.0
    return tpr

def load_r2s(hier_dir: str, K: int) -> np.ndarray:
    p = os.path.join(hier_dir, f"region_to_superregion_K{K}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Run analyze_hard_memory_and_predictive_hierarchy.py first."
        )
    with open(p) as f:
        raw = json.load(f)
    arr = np.zeros(N_FINE, dtype=np.int32)
    for k, v in raw.items():
        arr[int(k)] = int(v)
    return arr

# ── Superregion helpers ───────────────────────────────────────────────────────

def aggregate_to_superregions(
    top_regions: np.ndarray, top_probs: np.ndarray,
    r2s: np.ndarray, n_super: int,
) -> np.ndarray:
    N, K = top_regions.shape
    super_regs = r2s[top_regions.astype(np.int32)]
    sp = np.zeros((N, n_super), dtype=np.float64)
    rows_idx = np.repeat(np.arange(N), K)
    np.add.at(sp, (rows_idx, super_regs.ravel()), top_probs.ravel().astype(np.float64))
    ss = sp.sum(axis=1, keepdims=True)
    ss[ss == 0] = 1.0
    return (sp / ss).astype(np.float32)

def super_margin_entropy(sp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gold-free: returns (margin, entropy) for (N, n_super) probs."""
    sorted_p = np.sort(sp, axis=1)[:, ::-1]
    margin  = (sorted_p[:, 0] - sorted_p[:, 1]).astype(np.float32)
    entropy = (-(sp * np.log(sp + 1e-10)).sum(axis=1)).astype(np.float32)
    return margin, entropy

def build_children_indicator(r2s: np.ndarray, n_super: int, n_fine: int = N_FINE) -> np.ndarray:
    """(n_super, n_fine) bool: C[s, j] = True iff fine region j belongs to super s."""
    C = np.zeros((n_super, n_fine), dtype=bool)
    for j, s in enumerate(r2s):
        C[int(s), j] = True
    return C

def dense_probs(top_regions: np.ndarray, top_probs: np.ndarray, n_fine: int = N_FINE) -> np.ndarray:
    """(N, n_fine) float32: scatter top_probs into dense region-probability vector."""
    N, K = top_regions.shape
    P = np.zeros((N, n_fine), dtype=np.float32)
    rows_idx = np.repeat(np.arange(N), K)
    np.add.at(P, (rows_idx, top_regions.ravel().astype(np.int32)), top_probs.ravel().astype(np.float32))
    return P

# ── Policy row builder ────────────────────────────────────────────────────────

def policy_row(
    policy: str,
    covered: np.ndarray,
    tok_per_pos: np.ndarray,
    reg_per_pos: np.ndarray,
    groups: Dict[str, np.ndarray],
    tpr: np.ndarray,
    **extra,
) -> Dict:
    total_mapped = float(tpr.sum())
    avg_tok = float(tok_per_pos.mean())
    row: Dict = {
        "policy":               policy,
        "gold_region_coverage": round(float(covered.mean()), 4),
        "avg_regions":          round(float(reg_per_pos.mean()), 2),
        "avg_tokens":           round(avg_tok, 1),
        "vocab_percent":        round(avg_tok / FULL_VOCAB * 100, 2),
        "mapped_vocab_percent": round(avg_tok / total_mapped * 100, 2) if total_mapped > 0 else float("nan"),
    }
    for gname, gmask in groups.items():
        n = int(gmask.sum())
        row[f"coverage_{gname}"] = round(float(covered[gmask].mean()), 4) if n > 0 else float("nan")
    row.update(extra)
    return row

# ── Gated-base precompute (shared) ────────────────────────────────────────────

def _build_gated_base(
    r_mg: np.ndarray, r_gr: np.ndarray, m_gr: np.ndarray,
    r_top: np.ndarray, m_top: np.ndarray, tpr: np.ndarray, N: int,
) -> Dict:
    high = r_mg >= 0.30
    med  = (r_mg >= 0.10) & (r_mg < 0.30)
    bnd  = r_mg < 0.10
    T_r4  = tok_vec(r_top, 4,  tpr)
    T_r8  = tok_vec(r_top, 8,  tpr)
    T_r16 = tok_vec(r_top, 16, tpr)
    cov_r4  = r_gr < 4
    cov_r8  = r_gr < 8
    cov_r16 = r_gr < 16
    T_r8m4 = T_r8.copy()
    R_r8m4 = np.full(N, 8.0, dtype=np.float32)
    if bnd.any():
        I_bnd = scatter_indicator(r_top[bnd], 8) | scatter_indicator(m_top[bnd], 4)
        T_r8m4[bnd] = (I_bnd.astype(np.float32) @ tpr)
        R_r8m4[bnd] = I_bnd.sum(axis=1).astype(np.float32)
    ovlp = np.zeros(N, dtype=bool)
    if bnd.any():
        ovlp[bnd] = overlap_any(r_top[bnd], 8, m_top[bnd], 4)
    return dict(
        high=high, med=med, bnd=bnd,
        T_r4=T_r4, T_r8=T_r8, T_r16=T_r16, T_r8m4=T_r8m4,
        R_r8m4=R_r8m4,
        cov_r4=cov_r4, cov_r8=cov_r8, cov_r16=cov_r16,
        cov_r8m4=(r_gr < 8) | (m_gr < 4),
        ovlp=ovlp,
    )

# ── Precompute per-K arrays ────────────────────────────────────────────────────

def precompute_per_K(
    r_top: np.ndarray, m_top: np.ndarray,
    r_prob: np.ndarray, m_prob: np.ndarray,
    r2s: np.ndarray, n_super: int, tpr: np.ndarray, N: int,
) -> Dict:
    sp_mem    = aggregate_to_superregions(m_top, m_prob, r2s, n_super)
    sp_router = aggregate_to_superregions(r_top, r_prob, r2s, n_super)
    sp_comb   = 0.5 * sp_mem + 0.5 * sp_router
    sm_mg, sm_en = super_margin_entropy(sp_mem)
    sr_mg, sr_en = super_margin_entropy(sp_router)
    sc_mg, sc_en = super_margin_entropy(sp_comb)
    C = build_children_indicator(r2s, n_super)
    P_r = dense_probs(r_top, r_prob)
    P_m = dense_probs(m_top, m_prob)
    return dict(
        sp_mem=sp_mem, sp_router=sp_router, sp_comb=sp_comb,
        sm_mg=sm_mg, sm_en=sm_en,
        sr_mg=sr_mg, sr_en=sr_en,
        sc_mg=sc_mg, sc_en=sc_en,
        C=C, P_r=P_r, P_m=P_m,
        n_super=n_super,
    )

# ── Core-tier helpers ─────────────────────────────────────────────────────────

def _tok_K(r_top: np.ndarray, K: int, tpr: np.ndarray) -> np.ndarray:
    return tpr[r_top[:, :min(K, r_top.shape[1])].astype(np.int32)].sum(axis=1)

def _apply_core_tiers(
    r_mg: np.ndarray, r_gr: np.ndarray, r_top: np.ndarray, tpr: np.ndarray,
    base: Dict,
    core_high: float, core_mid: float, mid_k: int,
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (T, R, cov, high_mask, med_mask) for the non-boundary portion.
    core_high: threshold above which use top-4
    core_mid:  threshold above which use top-mid_k, below core_high
    mid_k:     router top-K for mid tier
    Positions with r_mg < 0.10 are boundary and handled separately.
    """
    T   = np.zeros(N, dtype=np.float32)
    R   = np.zeros(N, dtype=np.float32)
    cov = np.zeros(N, dtype=bool)

    very_high = r_mg >= core_high                              # top-4
    mid_high  = (r_mg >= core_mid) & (r_mg < core_high)       # top-mid_k
    low_high  = (r_mg >= 0.30) & (r_mg < core_mid) if core_mid >= 0.30 else np.zeros(N, dtype=bool)
    med_low   = (r_mg >= 0.10) & (r_mg < min(core_mid, 0.30)) # top-8

    T_mh = _tok_K(r_top, mid_k, tpr)
    cov_mh = r_gr < mid_k

    T[very_high]  = base["T_r4"][very_high];   R[very_high]  = 4.0;    cov[very_high]  = base["cov_r4"][very_high]
    T[mid_high]   = T_mh[mid_high];            R[mid_high]   = mid_k;  cov[mid_high]   = cov_mh[mid_high]
    T[low_high]   = base["T_r8"][low_high];    R[low_high]   = 8.0;    cov[low_high]   = base["cov_r8"][low_high]
    T[med_low]    = base["T_r8"][med_low];     R[med_low]    = 8.0;    cov[med_low]    = base["cov_r8"][med_low]

    non_bnd = very_high | mid_high | low_high | med_low
    return T, R, cov, non_bnd

# ── TASK 1+4+5: Full hierarchy grid ──────────────────────────────────────────

def eval_hier_full_grid(
    r_mg: np.ndarray, m_mg: np.ndarray, m_en: np.ndarray,
    r_gr: np.ndarray, m_gr: np.ndarray,
    r_top: np.ndarray, m_top: np.ndarray,
    gold: np.ndarray,
    pkK: Dict,
    base: Dict,
    r2s: np.ndarray,
    K_label: int,
    groups: Dict, tpr: np.ndarray, N: int,
) -> List[Dict]:
    """
    Task 1: full (n_top_super × base_kr × fallback_k × sup_src) grid.
    Task 4: core margin tiers.
    Task 5: boundary-only hierarchy (core/medium fixed, only boundary uses hier).
    """
    rows: List[Dict] = []
    bnd = base["bnd"]
    C   = pkK["C"]           # (n_super, n_fine)
    n_super = pkK["n_super"]

    gold_super = r2s[gold.astype(np.int32)]  # (N,) gold fine → coarse

    # Source options for superregion prob reference (argsort key for top-super selection)
    SUP_SRCS = {
        "combined": pkK["sp_comb"],
        "mem":      pkK["sp_mem"],
    }

    # Precompute typeA_like (fixed thresholds match original)
    typeA_like = bnd & (m_mg > 0.20) & (m_en < 1.50) & base["ovlp"]

    # Precompute fallback token vectors
    FB_KS = [8, 12, 16, 24, 32]
    T_fb  = {k: _tok_K(r_top, k, tpr) for k in FB_KS}
    cov_fb = {k: r_gr < k            for k in FB_KS}

    # Precompute router base indicators/tokens (for hierarchy expansion)
    BASE_KRS = [4, 6, 8, 10, 12, 16]
    T_bkr   = {k: _tok_K(r_top, k, tpr) for k in BASE_KRS}
    cov_bkr = {k: r_gr < k              for k in BASE_KRS}

    # Precompute tpr[r_top[bnd, :max_base_kr]] for cheap overlap subtraction
    bnd_idx = np.where(bnd)[0]
    N_bnd   = int(bnd.sum())
    r_top_bnd = r_top[bnd]      # (N_bnd, K_avail)
    gold_bnd  = gold[bnd]

    # Core tier configs (Task 4) — also include default (0.30, 0.20, 8)
    CORE_TIERS = [
        (0.30, 0.10, 8,  "std"),   # original: high=top4, med=top8
        (0.40, 0.20, 6,  "c40m20"),
        (0.50, 0.25, 6,  "c50m25"),
        (0.60, 0.30, 8,  "c60m30"),
    ]

    # Coarse-conf threshold pairs
    COARSE_PAIRS = [
        (0.15, 1.25),
        (0.20, 1.50),
        (0.25, 1.75),
    ]

    N_TOP_SUPER_LIST = [1, 2, 3, 4]

    total_configs = (len(SUP_SRCS) * len(COARSE_PAIRS) * len(N_TOP_SUPER_LIST)
                     * len(BASE_KRS) * len(FB_KS) * len(CORE_TIERS))
    print(f"  [hier_grid K={K_label}] {total_configs} configs ...")

    for src_name, sp_ref in SUP_SRCS.items():
        # Pre-sort all boundary positions by this source (descending)
        sp_ref_bnd = sp_ref[bnd]    # (N_bnd, n_super)
        top_supers_bnd_all = np.argsort(-sp_ref_bnd, axis=1)  # (N_bnd, n_super)

        for cm_thr, en_thr in COARSE_PAIRS:
            # Gold-free coarse-conf detection: OR of mem and router superregion signals
            coarse_conf_bnd = ~typeA_like[bnd] & (
                ((pkK["sm_mg"][bnd] > cm_thr) & (pkK["sm_en"][bnd] < en_thr)) |
                ((pkK["sr_mg"][bnd] > cm_thr) & (pkK["sr_en"][bnd] < en_thr))
            )
            # Indices within bnd arrays
            cc_bnd_idx = np.where(coarse_conf_bnd)[0]   # positions in bnd that are coarse_conf
            N_cc = len(cc_bnd_idx)

            # Global boolean mask for coarse_conf positions (stable across n_ts/base_kr/fallback_k)
            cc_global_base = np.zeros(N, dtype=bool)
            if N_cc > 0:
                cc_global_base[bnd_idx[cc_bnd_idx]] = True

            for n_ts in N_TOP_SUPER_LIST:
                if N_cc == 0:
                    # No coarse_conf positions — hierarchy doesn't trigger
                    top_supers_cc = np.empty((0, n_ts), dtype=np.int32)
                    child_I_cc    = np.empty((0, N_FINE), dtype=bool)
                    T_child_cc    = np.empty(0, dtype=np.float32)
                    gold_in_child_cc = np.empty(0, dtype=bool)
                    n_children_cc    = np.empty(0, dtype=np.float32)
                else:
                    top_supers_cc = top_supers_bnd_all[cc_bnd_idx, :n_ts]  # (N_cc, n_ts)
                    # Union of children of selected superregions: (N_cc, n_fine)
                    child_I_cc    = C[top_supers_cc].any(axis=1)
                    T_child_cc    = (child_I_cc.astype(np.float32) @ tpr)   # (N_cc,)
                    gold_in_child_cc = child_I_cc[np.arange(N_cc), gold_bnd[cc_bnd_idx]]
                    n_children_cc    = child_I_cc.sum(axis=1).astype(np.float32)

                for base_kr in BASE_KRS:
                    # For coarse_conf positions: overlap between router_top_base and children
                    # T_h = T_router_base + T_child - T_overlap
                    if N_cc > 0:
                        r_top_cc     = r_top_bnd[cc_bnd_idx]  # (N_cc, K_avail)
                        K_eff_cc     = min(base_kr, r_top_cc.shape[1])
                        tpr_rtop_cc  = tpr[r_top_cc[:, :K_eff_cc].astype(np.int32)]  # (N_cc, K_eff)
                        ovlp_mask_cc = child_I_cc[np.arange(N_cc)[:, None],
                                                  r_top_cc[:, :K_eff_cc]]  # (N_cc, K_eff)
                        T_both_cc    = (ovlp_mask_cc.astype(np.float32) * tpr_rtop_cc).sum(axis=1)
                        T_hier_cc    = T_bkr[base_kr][bnd][cc_bnd_idx] + T_child_cc - T_both_cc
                        R_hier_cc    = base_kr + n_children_cc - ovlp_mask_cc.sum(axis=1).astype(np.float32)
                        cov_hier_cc  = (r_gr[bnd][cc_bnd_idx] < base_kr) | gold_in_child_cc
                    else:
                        T_hier_cc = np.empty(0, dtype=np.float32)
                        R_hier_cc = np.empty(0, dtype=np.float32)
                        cov_hier_cc = np.empty(0, dtype=bool)

                    # Boundary fallback mask (stable across fallback_k and core tiers)
                    bnd_fb_global = bnd & ~typeA_like & ~cc_global_base

                    for core_high, core_mid, mid_k, core_tag in CORE_TIERS:
                        # Core-tier masks and tokens (stable across fallback_k)
                        very_high = r_mg >= core_high
                        mid_high  = (r_mg >= core_mid) & (r_mg < core_high)
                        low_high  = ((r_mg >= 0.30) & (r_mg < core_mid)
                                     if core_mid >= 0.30 else np.zeros(N, dtype=bool))
                        med_part  = (r_mg >= 0.10) & (r_mg < min(core_mid, 0.30))
                        T_mh   = _tok_K(r_top, mid_k, tpr)
                        cov_mh = r_gr < mid_k

                        for fallback_k in FB_KS:
                            # ── Build full N-length arrays ──────────────────
                            T   = np.zeros(N, dtype=np.float32)
                            R   = np.zeros(N, dtype=np.float32)
                            cov = np.zeros(N, dtype=bool)

                            T[very_high]  = base["T_r4"][very_high];    R[very_high]  = 4.0
                            cov[very_high] = base["cov_r4"][very_high]
                            T[mid_high]   = T_mh[mid_high];             R[mid_high]   = float(mid_k)
                            cov[mid_high]  = cov_mh[mid_high]
                            T[low_high]   = base["T_r8"][low_high];     R[low_high]   = 8.0
                            cov[low_high]  = base["cov_r8"][low_high]
                            T[med_part]   = base["T_r8"][med_part];     R[med_part]   = 8.0
                            cov[med_part]  = base["cov_r8"][med_part]

                            # typeA_like boundary
                            T[typeA_like]   = base["T_r8m4"][typeA_like]
                            R[typeA_like]   = base["R_r8m4"][typeA_like]
                            cov[typeA_like] = base["cov_r8m4"][typeA_like]

                            T[bnd_fb_global]   = T_fb[fallback_k][bnd_fb_global]
                            R[bnd_fb_global]   = float(fallback_k)
                            cov[bnd_fb_global] = cov_fb[fallback_k][bnd_fb_global]

                            # Coarse-conf (hierarchy) positions
                            if N_cc > 0:
                                T[bnd_idx[cc_bnd_idx]]   = T_hier_cc
                                R[bnd_idx[cc_bnd_idx]]   = R_hier_cc
                                cov[bnd_idx[cc_bnd_idx]] = cov_hier_cc

                            tag = (f"hgrid_K{K_label}_src{src_name}_nts{n_ts}"
                                   f"_bkr{base_kr}_fb{fallback_k}"
                                   f"_cm{cm_thr:.2f}_en{en_thr:.2f}"
                                   f"_core{core_tag}")
                            rows.append(policy_row(
                                tag, cov, T, R, groups, tpr,
                                K_super=K_label, n_top_super=n_ts,
                                base_kr=base_kr, fallback_k=fallback_k,
                                src=src_name, coarse_margin_thr=cm_thr,
                                coarse_entropy_thr=en_thr,
                                core_high=core_high, core_mid=core_mid, mid_k=mid_k,
                                n_coarse_conf=N_cc, task="hier_grid",
                            ))

    print(f"  [hier_grid K={K_label}] {len(rows)} rows generated")
    return rows


# ── TASK 2: Adaptive top-super selection ─────────────────────────────────────

def eval_adaptive_super(
    r_mg: np.ndarray, m_mg: np.ndarray, m_en: np.ndarray,
    r_gr: np.ndarray, m_gr: np.ndarray,
    r_top: np.ndarray, m_top: np.ndarray,
    gold: np.ndarray,
    pkK: Dict,
    base: Dict,
    r2s: np.ndarray,
    K_label: int,
    groups: Dict, tpr: np.ndarray, N: int,
) -> List[Dict]:
    """
    Adaptive n_top_super: use n_top_super=1 when coarse signal is confident,
    else expand to n_top_super=2.
    Grid: margin-based and entropy-based thresholds.
    """
    rows: List[Dict] = []
    bnd = base["bnd"]
    bnd_idx = np.where(bnd)[0]
    C       = pkK["C"]
    n_super = pkK["n_super"]
    gold_super = r2s[gold.astype(np.int32)]
    sp_comb  = pkK["sp_comb"]
    sc_mg    = pkK["sc_mg"]
    sc_en    = pkK["sc_en"]

    typeA_like = bnd & (m_mg > 0.20) & (m_en < 1.50) & base["ovlp"]

    # Fixed coarse-conf gate (use default thresholds)
    cm_thr_def, en_thr_def = 0.20, 1.50
    coarse_conf_bnd = ~typeA_like[bnd] & (
        ((pkK["sm_mg"][bnd] > cm_thr_def) & (pkK["sm_en"][bnd] < en_thr_def)) |
        ((pkK["sr_mg"][bnd] > cm_thr_def) & (pkK["sr_en"][bnd] < en_thr_def))
    )
    cc_bnd_idx = np.where(coarse_conf_bnd)[0]
    N_cc = len(cc_bnd_idx)

    sp_ref_bnd   = sp_comb[bnd]
    sc_mg_bnd    = sc_mg[bnd]
    sc_en_bnd    = sc_en[bnd]
    r_top_bnd    = r_top[bnd]
    gold_bnd     = gold[bnd]
    r_gr_bnd     = r_gr[bnd]

    top2_bnd_all = np.argsort(-sp_ref_bnd, axis=1)[:, :2]  # (N_bnd, 2)

    MARGIN_THRS  = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    ENTROPY_THRS = [0.75, 1.00, 1.25, 1.50, 1.75]
    BASE_KR, FALLBACK_K = 8, 16

    for adapt_type, thresholds in [("margin", MARGIN_THRS), ("entropy", ENTROPY_THRS)]:
        for thr in thresholds:
            if N_cc > 0:
                cc_mg = sc_mg_bnd[cc_bnd_idx]
                cc_en = sc_en_bnd[cc_bnd_idx]

                if adapt_type == "margin":
                    use_top1 = cc_mg > thr
                    thr_fmt  = f"m{thr:.2f}"
                else:
                    use_top1 = cc_en < thr
                    thr_fmt  = f"e{thr:.2f}"

                # Effective n_ts per position: 1 if confident, else 2
                # Build per-position child_I
                top1_cc  = top2_bnd_all[cc_bnd_idx, :1]   # (N_cc, 1)
                top2_cc  = top2_bnd_all[cc_bnd_idx, :]     # (N_cc, 2)
                child1   = C[top1_cc[:, 0]]                 # (N_cc, n_fine)
                child2   = C[top2_cc].any(axis=1)           # (N_cc, n_fine)
                child_I_cc = np.where(use_top1[:, np.newaxis], child1, child2)

                T_child_cc = (child_I_cc.astype(np.float32) @ tpr)
                gold_in_child_cc = child_I_cc[np.arange(N_cc), gold_bnd[cc_bnd_idx]]
                n_children_cc    = child_I_cc.sum(axis=1).astype(np.float32)

                r_top_cc    = r_top_bnd[cc_bnd_idx]
                K_eff_cc    = min(BASE_KR, r_top_cc.shape[1])
                tpr_rtop_cc = tpr[r_top_cc[:, :K_eff_cc].astype(np.int32)]
                ovlp_cc     = child_I_cc[np.arange(N_cc)[:, None], r_top_cc[:, :K_eff_cc]]
                T_both_cc   = (ovlp_cc.astype(np.float32) * tpr_rtop_cc).sum(axis=1)
                T_hier_cc   = _tok_K(r_top_bnd[cc_bnd_idx], BASE_KR, tpr) + T_child_cc - T_both_cc
                R_hier_cc   = BASE_KR + n_children_cc - ovlp_cc.sum(axis=1).astype(np.float32)
                cov_hier_cc = (r_gr_bnd[cc_bnd_idx] < BASE_KR) | gold_in_child_cc
            else:
                T_hier_cc = cov_hier_cc = R_hier_cc = np.empty(0)
                thr_fmt   = f"m{thr:.2f}" if adapt_type == "margin" else f"e{thr:.2f}"

            T   = np.zeros(N, dtype=np.float32)
            R   = np.zeros(N, dtype=np.float32)
            cov = np.zeros(N, dtype=bool)

            very_high = r_mg >= 0.30
            med_part  = (r_mg >= 0.10) & (r_mg < 0.30)
            T[very_high] = base["T_r4"][very_high]; R[very_high] = 4.0;  cov[very_high] = base["cov_r4"][very_high]
            T[med_part]  = base["T_r8"][med_part];  R[med_part]  = 8.0;  cov[med_part]  = base["cov_r8"][med_part]
            T[typeA_like] = base["T_r8m4"][typeA_like]; R[typeA_like] = base["R_r8m4"][typeA_like]
            cov[typeA_like] = base["cov_r8m4"][typeA_like]

            cc_global = np.zeros(N, dtype=bool)
            if N_cc > 0:
                cc_global[bnd_idx[cc_bnd_idx]] = True
            bnd_fb = bnd & ~typeA_like & ~cc_global
            T[bnd_fb]   = _tok_K(r_top, FALLBACK_K, tpr)[bnd_fb]
            R[bnd_fb]   = float(FALLBACK_K)
            cov[bnd_fb] = (r_gr < FALLBACK_K)[bnd_fb]

            if N_cc > 0:
                T[bnd_idx[cc_bnd_idx]]   = T_hier_cc
                R[bnd_idx[cc_bnd_idx]]   = R_hier_cc
                cov[bnd_idx[cc_bnd_idx]] = cov_hier_cc

            tag = f"hier_adaptive_super_K{K_label}_{adapt_type}{thr_fmt}"
            rows.append(policy_row(
                tag, cov, T, R, groups, tpr,
                K_super=K_label, adapt_type=adapt_type, adapt_thr=thr,
                base_kr=BASE_KR, fallback_k=FALLBACK_K, task="adaptive_super",
            ))

    print(f"  [adaptive_super K={K_label}] {len(rows)} rows")
    return rows


# ── TASK 3: Child-region capped hierarchy ─────────────────────────────────────

def eval_child_capped(
    r_mg: np.ndarray, m_mg: np.ndarray, m_en: np.ndarray,
    r_gr: np.ndarray, m_gr: np.ndarray,
    r_top: np.ndarray, m_top: np.ndarray,
    gold: np.ndarray,
    pkK: Dict,
    base: Dict,
    r2s: np.ndarray,
    K_label: int,
    groups: Dict, tpr: np.ndarray, N: int,
) -> List[Dict]:
    """
    Score children of selected superregions by alpha*router + (1-alpha)*mem,
    keep top-M, union with router_top_base.
    """
    rows: List[Dict] = []
    bnd = base["bnd"]
    bnd_idx = np.where(bnd)[0]
    C       = pkK["C"]
    P_r     = pkK["P_r"]
    P_m     = pkK["P_m"]
    sp_comb = pkK["sp_comb"]

    typeA_like = bnd & (m_mg > 0.20) & (m_en < 1.50) & base["ovlp"]

    # Fixed: n_ts=1, coarse_conf uses default threshold
    N_TOP_SUPER = 1
    cm_thr, en_thr = 0.20, 1.50
    coarse_conf_bnd = ~typeA_like[bnd] & (
        ((pkK["sm_mg"][bnd] > cm_thr) & (pkK["sm_en"][bnd] < en_thr)) |
        ((pkK["sr_mg"][bnd] > cm_thr) & (pkK["sr_en"][bnd] < en_thr))
    )
    cc_bnd_idx = np.where(coarse_conf_bnd)[0]
    N_cc = len(cc_bnd_idx)

    sp_ref_bnd = sp_comb[bnd]
    top1_bnd   = np.argsort(-sp_ref_bnd, axis=1)[:, :1]  # (N_bnd, 1)

    CHILD_TOP_M   = [4, 6, 8, 10, 12, 16, 24]
    ALPHAS        = [0.0, 0.25, 0.50, 0.75, 1.0]
    ROUTER_BASE_KS = [4, 6, 8, 12]
    FALLBACK_K     = 16

    if N_cc > 0:
        top1_cc  = top1_bnd[cc_bnd_idx, 0]  # (N_cc,) — selected superregion per position
        # Candidate mask: children of selected super
        child_mask_cc = C[top1_cc]           # (N_cc, n_fine)
        P_r_cc = P_r[bnd_idx[cc_bnd_idx]]   # (N_cc, n_fine)
        P_m_cc = P_m[bnd_idx[cc_bnd_idx]]   # (N_cc, n_fine)
        gold_cc = gold[bnd_idx[cc_bnd_idx]] # (N_cc,)
        r_gr_cc = r_gr[bnd_idx[cc_bnd_idx]]

    for alpha in ALPHAS:
        if N_cc > 0:
            score_cc = alpha * P_r_cc + (1.0 - alpha) * P_m_cc  # (N_cc, n_fine)
            score_cc[~child_mask_cc] = -1.0  # exclude non-children

        for child_top_m in CHILD_TOP_M:
            if N_cc > 0:
                m_eff = min(child_top_m, N_FINE)
                # Top-M children per position
                topM_idx = np.argpartition(-score_cc, m_eff - 1, axis=1)[:, :m_eff]
                I_child = np.zeros((N_cc, N_FINE), dtype=bool)
                ri = np.repeat(np.arange(N_cc), m_eff)
                ci = topM_idx.ravel()
                valid = score_cc[ri, ci] >= 0.0  # was a real child
                I_child[ri[valid], ci[valid]] = True
                T_child_cc    = (I_child.astype(np.float32) @ tpr)
                gold_in_child_cc = I_child[np.arange(N_cc), gold_cc]

            for router_base_k in ROUTER_BASE_KS:
                if N_cc > 0:
                    r_top_cc    = r_top[bnd_idx[cc_bnd_idx]]
                    K_eff_cc    = min(router_base_k, r_top_cc.shape[1])
                    I_base      = scatter_indicator(r_top_cc, router_base_k)
                    I_h         = I_base | I_child
                    T_hier_cc   = (I_h.astype(np.float32) @ tpr)
                    R_hier_cc   = I_h.sum(axis=1).astype(np.float32)
                    cov_hier_cc = I_h[np.arange(N_cc), gold_cc]
                else:
                    T_hier_cc = R_hier_cc = cov_hier_cc = np.empty(0)

                T   = np.zeros(N, dtype=np.float32)
                R   = np.zeros(N, dtype=np.float32)
                cov = np.zeros(N, dtype=bool)

                non_bnd = ~bnd
                T[non_bnd & (r_mg >= 0.30)] = base["T_r4"][non_bnd & (r_mg >= 0.30)]
                R[non_bnd & (r_mg >= 0.30)] = 4.0
                cov[non_bnd & (r_mg >= 0.30)] = base["cov_r4"][non_bnd & (r_mg >= 0.30)]
                med = (r_mg >= 0.10) & (r_mg < 0.30)
                T[med] = base["T_r8"][med]; R[med] = 8.0; cov[med] = base["cov_r8"][med]
                T[typeA_like] = base["T_r8m4"][typeA_like]
                R[typeA_like] = base["R_r8m4"][typeA_like]
                cov[typeA_like] = base["cov_r8m4"][typeA_like]

                cc_global = np.zeros(N, dtype=bool)
                if N_cc > 0:
                    cc_global[bnd_idx[cc_bnd_idx]] = True
                bnd_fb = bnd & ~typeA_like & ~cc_global
                T[bnd_fb]   = _tok_K(r_top, FALLBACK_K, tpr)[bnd_fb]
                R[bnd_fb]   = float(FALLBACK_K)
                cov[bnd_fb] = (r_gr < FALLBACK_K)[bnd_fb]

                if N_cc > 0:
                    T[bnd_idx[cc_bnd_idx]]   = T_hier_cc
                    R[bnd_idx[cc_bnd_idx]]   = R_hier_cc
                    cov[bnd_idx[cc_bnd_idx]] = cov_hier_cc

                a_fmt = f"{alpha:.2f}".replace(".", "p")
                tag = (f"hier_childcap_K{K_label}_m{child_top_m}"
                       f"_a{a_fmt}_rb{router_base_k}")
                rows.append(policy_row(
                    tag, cov, T, R, groups, tpr,
                    K_super=K_label, child_top_m=child_top_m,
                    alpha=alpha, router_base_k=router_base_k,
                    fallback_k=FALLBACK_K, task="child_capped",
                ))

    print(f"  [child_capped K={K_label}] {len(rows)} rows")
    return rows


# ── TASK 6: Pareto frontier + tuned report ────────────────────────────────────

def compute_pareto(rows: List[Dict]) -> List[Dict]:
    """
    Pareto-dominant subset over (gold_region_coverage MAX, vocab_percent MIN).
    A row dominates another if it has >= coverage AND <= vocab_percent,
    with strict inequality on at least one.
    """
    if not rows:
        return []
    cov_arr = np.array([_flt(r, "gold_region_coverage") for r in rows])
    voc_arr = np.array([_flt(r, "vocab_percent")        for r in rows])
    n = len(rows)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        # Any j that strictly dominates i?
        j_dom = ((cov_arr >= cov_arr[i]) & (voc_arr <= voc_arr[i]) &
                 ((cov_arr > cov_arr[i]) | (voc_arr < voc_arr[i])))
        j_dom[i] = False
        if j_dom.any():
            dominated[i] = True
    return [rows[i] for i in range(n) if not dominated[i]]


def write_tuned_report(
    out_dir: str,
    all_rows: List[Dict],
    pareto_rows: List[Dict],
    baseline_rows: Dict,
    n_total: int,
) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("# Hierarchical Policy Tuning Report\n")
    lines.append(f"*Generated: {now}*  \n")
    lines.append(f"*Total policies evaluated: {len(all_rows):,}*  \n")
    lines.append(f"*Pareto-optimal: {len(pareto_rows):,}*  \n\n")

    # Reference baselines
    lines.append("## Reference Baselines\n")
    lines.append("| policy | coverage | vocab% | boundary_cov |\n")
    lines.append("|--------|----------|--------|--------------|\n")
    for name, row in baseline_rows.items():
        cov  = _flt(row, "gold_region_coverage")
        voc  = _flt(row, "vocab_percent")
        bcov = _flt(row, "coverage_boundary")
        lines.append(f"| {name} | {cov:.4f} | {voc:.2f} | {bcov:.4f} |\n")

    # Best per vocab budget
    BUDGETS = [
        ("vocab <= 6%",  6.0),
        ("vocab <= 7%",  7.0),
        ("vocab <= 8%",  8.0),
        ("vocab <= 10%", 10.0),
        ("vocab <= 15%", 15.0),
    ]
    lines.append("\n## Best Policy per Vocab Budget\n")
    for budget_name, voc_limit in BUDGETS:
        candidates = [r for r in all_rows if _flt(r, "vocab_percent") <= voc_limit]
        if not candidates:
            lines.append(f"\n### {budget_name}\nNo policies in this budget.\n")
            continue
        best = max(candidates, key=lambda r: _flt(r, "gold_region_coverage"))
        cov  = _flt(best, "gold_region_coverage")
        voc  = _flt(best, "vocab_percent")
        bcov = _flt(best, "coverage_boundary")
        tAcov = _flt(best, "coverage_type_A")
        tBcov = _flt(best, "coverage_type_B")
        tCcov = _flt(best, "coverage_type_C")
        mu   = _flt(best, "memory_used_rate", 0.0)
        fb   = _flt(best, "fallback_rate", 0.0)

        ref_cov = _flt(baseline_rows.get("router_top8", {}), "gold_region_coverage", 0.9009)
        ref_voc = _flt(baseline_rows.get("router_top8", {}), "vocab_percent", 7.92)
        beats   = "YES" if cov >= ref_cov and voc <= ref_voc else "NO"

        lines.append(f"\n### {budget_name}\n")
        lines.append(f"- **policy**: `{best['policy']}`\n")
        lines.append(f"- coverage: {cov:.4f}  ({'**beats router_top8**' if beats == 'YES' else 'below router_top8'})\n")
        lines.append(f"- vocab%: {voc:.2f}\n")
        lines.append(f"- avg_tokens: {_flt(best,'avg_tokens'):.1f}\n")
        lines.append(f"- coverage_boundary: {bcov:.4f}\n")
        lines.append(f"- coverage_type_A: {tAcov:.4f}\n")
        lines.append(f"- coverage_type_B: {tBcov:.4f}\n")
        lines.append(f"- coverage_type_C: {tCcov:.4f}\n")
        lines.append(f"- memory_used_rate: {mu:.4f}  fallback_rate: {fb:.4f}\n")
        lines.append(f"- K_super: {best.get('K_super','')}  "
                     f"n_top_super: {best.get('n_top_super','')}  "
                     f"base_kr: {best.get('base_kr','')}  "
                     f"fallback_k: {best.get('fallback_k','')}  "
                     f"task: {best.get('task','')}\n")

    # Success criteria check
    lines.append("\n## Success Criteria Check\n")
    primary = [r for r in all_rows
               if _flt(r, "gold_region_coverage") >= 0.900
               and _flt(r, "vocab_percent") <= 8.0]
    ref_bcov = _flt(baseline_rows.get("router_top8", {}), "coverage_boundary", 0.8105)
    primary_better_bnd = [r for r in primary
                          if _flt(r, "coverage_boundary") > ref_bcov]
    strong = [r for r in all_rows
              if _flt(r, "gold_region_coverage") > 0.9009
              and _flt(r, "vocab_percent") <= 7.92]
    high_cov = [r for r in all_rows
                if _flt(r, "gold_region_coverage") >= 0.95
                and _flt(r, "vocab_percent") <= 15.0]

    def _bool(x): return "**YES**" if x else "NO"
    lines.append(f"- Primary (cov>=0.900, vocab<=8%, bnd>router): {_bool(bool(primary_better_bnd))}  "
                 f"({len(primary_better_bnd)} policies)\n")
    lines.append(f"- Strong (cov>0.9009, vocab<=7.92%): {_bool(bool(strong))}  "
                 f"({len(strong)} policies)\n")
    lines.append(f"- High-coverage (cov>=0.95, vocab<=15%): {_bool(bool(high_cov))}  "
                 f"({len(high_cov)} policies)\n")

    if strong:
        best_s = max(strong, key=lambda r: _flt(r, "gold_region_coverage"))
        lines.append(f"\nStrongest: `{best_s['policy']}`  "
                     f"cov={_flt(best_s,'gold_region_coverage'):.4f}  "
                     f"vocab={_flt(best_s,'vocab_percent'):.2f}%\n")
    if high_cov:
        best_h = max(high_cov, key=lambda r: _flt(r, "gold_region_coverage"))
        lines.append(f"High-cov best: `{best_h['policy']}`  "
                     f"cov={_flt(best_h,'gold_region_coverage'):.4f}  "
                     f"vocab={_flt(best_h,'vocab_percent'):.2f}%\n")

    path = os.path.join(out_dir, "tuned_final_report.md")
    with open(path, "w") as f:
        f.write("".join(lines))
    print(f"[report] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--knn_run_dir",  required=True, help="Directory with per_position.npz")
    p.add_argument("--hier_dir",     required=True,
                   help="Directory with region_to_superregion_K*.json files")
    p.add_argument("--region_map",   required=True, help="token_to_region.json path")
    p.add_argument("--output_dir",   required=True, help="Where to write tuned outputs")
    p.add_argument("--n_fine",       type=int, default=N_FINE)
    p.add_argument("--topk",         type=int, default=32)
    p.add_argument("--K_list",       nargs="+", type=int, default=SUPER_K_LIST,
                   help="Which K_super values to evaluate")
    p.add_argument("--skip_grid",    action="store_true", help="Skip Task-1 full grid (fast check)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    pp   = load_per_position(args.knn_run_dir)
    tpr  = load_tpr(args.region_map, args.n_fine)

    N = len(pp["gold_region"])
    if "router_topk_regions" not in pp:
        raise KeyError("router_topk_regions missing — run with probe active (--probe_temp 1.0)")

    gold   = pp["gold_region"].astype(np.int32)
    split  = pp["split"].astype(np.int32)
    type_a = pp["type"].astype(np.int32)
    m_top  = pp["mem_topk_regions"].astype(np.int32)
    m_prob = pp["mem_topk_probs"].astype(np.float32)
    m_mg   = pp["mem_margin"].astype(np.float32)
    m_en   = pp["mem_entropy"].astype(np.float32)
    m_gr   = pp["mem_gold_rank"].astype(np.int32)
    r_top  = pp["router_topk_regions"].astype(np.int32)
    r_prob = pp["router_topk_probs"].astype(np.float32)
    r_mg   = pp["router_margin"].astype(np.float32)
    r_en   = pp["router_entropy"].astype(np.float32)
    r_gr   = pp["router_gold_rank"].astype(np.int32)

    K = min(args.topk, m_top.shape[1])
    print(f"[main] N={N:,}  K={K}  n_fine={args.n_fine}")

    # ── Groups ────────────────────────────────────────────────────────────────
    bnd_global = r_mg < 0.10
    groups: Dict[str, np.ndarray] = {
        "all":            np.ones(N, dtype=bool),
        "core":           split == 0,
        "medium":         split == 1,
        "boundary":       split >= 2,
        "tight_boundary": split == 3,
        "type_A":         type_a == 1,
        "type_B":         type_a == 2,
        "type_C":         type_a == 3,
    }

    # ── Shared gated base (position-level precompute) ─────────────────────────
    print("[main] precomputing gated base ...")
    base = _build_gated_base(r_mg, r_gr, m_gr, r_top, m_top, tpr, N)

    # ── Reference baselines (router_top8, adaptive_v3, union_r16_m4) ─────────
    baseline_rows: Dict = {}
    for bk in [8, 16]:
        cov  = r_gr < bk
        tv   = _tok_K(r_top, bk, tpr)
        rv   = np.full(N, float(bk), dtype=np.float32)
        row  = policy_row(f"router_top{bk}", cov, tv, rv, groups, tpr)
        baseline_rows[f"router_top{bk}"] = row

    # union_r16_m4
    cov_u = (r_gr < 16) | (m_gr < 4)
    I_u   = scatter_indicator(r_top, 16) | scatter_indicator(m_top, 4)
    tv_u  = (I_u.astype(np.float32) @ tpr)
    rv_u  = I_u.sum(axis=1).astype(np.float32)
    baseline_rows["union_r16_m4"] = policy_row("union_r16_m4", cov_u, tv_u, rv_u, groups, tpr)

    print("[main] baselines:")
    for name, row in baseline_rows.items():
        print(f"  {name:20s}: cov={_flt(row,'gold_region_coverage'):.4f}  "
              f"vocab={_flt(row,'vocab_percent'):.2f}%  "
              f"bnd={_flt(row,'coverage_boundary'):.4f}")

    # ── Per-K evaluation loop ─────────────────────────────────────────────────
    all_rows: List[Dict] = []

    for K_sup in args.K_list:
        print(f"\n[K={K_sup}] loading region mapping ...")
        r2s = load_r2s(args.hier_dir, K_sup)
        n_super = int(r2s.max()) + 1
        print(f"  n_super={n_super}")

        print(f"  precomputing superregion arrays ...")
        pkK = precompute_per_K(r_top, m_top, r_prob, m_prob, r2s, n_super, tpr, N)

        if not args.skip_grid:
            rows_grid = eval_hier_full_grid(
                r_mg, m_mg, m_en, r_gr, m_gr, r_top, m_top,
                gold, pkK, base, r2s, K_sup, groups, tpr, N,
            )
            all_rows.extend(rows_grid)

        rows_adapt = eval_adaptive_super(
            r_mg, m_mg, m_en, r_gr, m_gr, r_top, m_top,
            gold, pkK, base, r2s, K_sup, groups, tpr, N,
        )
        all_rows.extend(rows_adapt)

        rows_cap = eval_child_capped(
            r_mg, m_mg, m_en, r_gr, m_gr, r_top, m_top,
            gold, pkK, base, r2s, K_sup, groups, tpr, N,
        )
        all_rows.extend(rows_cap)

        elapsed = time.time() - t0
        print(f"  [K={K_sup}] done ({elapsed:.1f}s total)  all_rows={len(all_rows):,}")

    # Append baselines for reference
    all_rows_with_baselines = list(baseline_rows.values()) + all_rows

    # ── Write tuned_policy_results.csv ────────────────────────────────────────
    print(f"\n[main] writing {len(all_rows_with_baselines):,} rows to tuned_policy_results.csv ...")
    _csv(all_rows_with_baselines,
         os.path.join(args.output_dir, "tuned_policy_results.csv"))

    # ── Pareto frontier ───────────────────────────────────────────────────────
    print("[main] computing Pareto frontier ...")
    pareto = compute_pareto(all_rows_with_baselines)
    pareto_sorted = sorted(pareto, key=lambda r: _flt(r, "vocab_percent"))
    _csv(pareto_sorted, os.path.join(args.output_dir, "tuned_pareto_frontier.csv"))
    print(f"  {len(pareto_sorted)} Pareto-optimal policies")

    # ── Final report ──────────────────────────────────────────────────────────
    write_tuned_report(args.output_dir, all_rows_with_baselines, pareto_sorted, baseline_rows, N)

    print(f"\n[done] {time.time()-t0:.1f}s  outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
