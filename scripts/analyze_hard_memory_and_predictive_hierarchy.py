#!/usr/bin/env python3
"""
Hard-Position Memory Controller + Type-B Predictive Hierarchy Analysis.

Part 1  Hard-position memory controller: can we detect memory-useful positions
        without gold? Tests router baselines, fixed union, gated memory grid,
        Type-A-like detector, and learned logistic controller.

Part 2  Type-B predictive hierarchy: co-occurrence graph → predictive
        super-region clustering → coarse B-to-A conversion rates + B1/B2/B3
        subtype breakdown.

Part 3  Combined hierarchical candidate selection policies (hier_v1-v4).

Claims tested
─────────────
  C1. Memory helps only on Type-A (same-predictive-region ambiguity).
  C2. Type-B uncertainty may be sharp at a coarser predictive resolution.
  C3. Hierarchical router+memory+superregion candidate set can meet the
      coverage/cost target without memory replacing the router globally.

Inputs
──────
  per_position.npz  from  offline_region_knn.py  (must exist in knn_run_dir).
  token_to_region.json  (region map).

Outputs
───────
  runs/hard_memory_predictive_hierarchy/
    hard_position_controller_results.csv
    learned_controller_results.csv
    typeB_mem_region_graph.csv
    typeB_router_region_graph.csv
    typeB_union_region_graph.csv
    region_to_superregion_K{K}.json  (K in 8,16,24,32,48,64)
    typeB_coarse_conversion.csv
    typeB_subtype_breakdown.csv
    hierarchical_candidate_policies.csv
    final_report.md
    plots/

Usage
─────
    python scripts/analyze_hard_memory_and_predictive_hierarchy.py \\
        --knn_run_dir runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20 \\
        --region_map_path runs/region_maps_128/token_to_region.json \\
        --output_dir runs/hard_memory_predictive_hierarchy \\
        --topk 32
"""

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.cluster import SpectralClustering, AgglomerativeClustering
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

# ── Constants ─────────────────────────────────────────────────────────────────

N_FINE       = 128
FULL_VOCAB   = 50257
SUPER_K_LIST = [8, 16, 24, 32, 48, 64]

SPLIT_NAMES = {0: "core", 1: "medium", 2: "boundary", 3: "tight_boundary"}
TYPE_NAMES  = {0: "other", 1: "type_A", 2: "type_B", 3: "type_C"}

# ── I/O helpers ───────────────────────────────────────────────────────────────

def _csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    # Union all keys across all rows (preserving first-appearance order)
    seen: set = set()
    all_keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def _j(x) -> str:
    return "nan" if (isinstance(x, float) and math.isnan(x)) else str(x)


def _flt(d: dict, k: str, default=float("nan")) -> float:
    try:
        return float(d[k])
    except Exception:
        return default

# ── Data loading ──────────────────────────────────────────────────────────────

def load_per_position(knn_run_dir: str) -> Dict[str, np.ndarray]:
    for name in ("per_position.npz", "per_position_topk.npz"):
        path = os.path.join(knn_run_dir, name)
        if os.path.exists(path):
            d = dict(np.load(path))
            print(f"[load] {path}  N={len(d['gold_region']):,}")
            return d
    raise FileNotFoundError(
        f"\n[ERROR] per_position.npz not found in: {knn_run_dir}\n"
        f"        Re-run offline_region_knn.py with FORCE_RUN=1:\n"
        f"          FORCE_RUN=1 python scripts/offline_region_knn.py \\\n"
        f"              --output_dir {knn_run_dir}  [same args as original run]\n"
        f"        Memory will load from cache (fast)."
    )


def load_tpr(region_map_path: str, n_fine: int = N_FINE) -> np.ndarray:
    with open(region_map_path) as f:
        rm = json.load(f)
    tpr = np.zeros(n_fine, dtype=np.float32)
    for _tok, rid in rm.items():
        r = int(rid)
        if 0 <= r < n_fine:
            tpr[r] += 1.0
    return tpr


def load_knn_config(knn_run_dir: str) -> dict:
    p = os.path.join(knn_run_dir, "knn_config.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ── Array helpers ─────────────────────────────────────────────────────────────

def scatter_indicator(top_regions: np.ndarray, K: int, n_fine: int = N_FINE) -> np.ndarray:
    """Return (N, n_fine) bool indicator for top-K regions."""
    N, K_max = top_regions.shape
    K_eff = min(K, K_max)
    I = np.zeros((N, n_fine), dtype=bool)
    rows = np.repeat(np.arange(N), K_eff)
    cols = top_regions[:, :K_eff].ravel().astype(np.int32)
    I[rows, cols] = True
    return I


def avg_tokens_from_I(I: np.ndarray, tpr: np.ndarray) -> float:
    return float((I.astype(np.float32) @ tpr).mean())


def avg_regions_from_I(I: np.ndarray) -> float:
    return float(I.sum(axis=1).mean())


def tok_vec(top_regions: np.ndarray, K: int, tpr: np.ndarray) -> np.ndarray:
    """Per-position token count for top-K (no duplicates guaranteed by topk)."""
    K_eff = min(K, top_regions.shape[1])
    return tpr[top_regions[:, :K_eff].astype(np.int32)].sum(axis=1)


def overlap_any(a: np.ndarray, Ka: int, b: np.ndarray, Kb: int) -> np.ndarray:
    """(N,) bool: any match between a_top_Ka and b_top_Kb."""
    return np.any(
        a[:, :Ka, np.newaxis].astype(np.int32) == b[:, np.newaxis, :Kb].astype(np.int32),
        axis=(1, 2)
    )


def overlap_count(a: np.ndarray, Ka: int, b: np.ndarray, Kb: int) -> np.ndarray:
    """(N,) int: count of matching regions."""
    match = (a[:, :Ka, np.newaxis].astype(np.int32) == b[:, np.newaxis, :Kb].astype(np.int32))
    return match.any(axis=2).sum(axis=1).astype(np.int32)


# ── Hard groups ───────────────────────────────────────────────────────────────

def build_hard_groups(
    split: np.ndarray,
    type_arr: np.ndarray,
    r_mg: np.ndarray,
    m_mg: np.ndarray,
    m_en: np.ndarray,
    r_en: np.ndarray,
    r_gr: np.ndarray,
    m_gr: np.ndarray,
    r_top: np.ndarray,
    m_top: np.ndarray,
) -> Dict[str, np.ndarray]:
    N = len(split)
    hi_ent_thr = np.percentile(r_en, 75)
    groups: Dict[str, np.ndarray] = {
        "all":                np.ones(N, dtype=bool),
        "core":               split == 0,
        "medium":             split == 1,
        "boundary":           split >= 2,
        "tight_boundary":     split == 3,
        "high_entropy":       r_en > hi_ent_thr,
        "memory_confident":   (m_mg > 0.20) & (m_en < 1.50),
        "router_mem_agree":   r_top[:, 0].astype(np.int32) == m_top[:, 0].astype(np.int32),
        "router_mem_overlap": overlap_any(r_top, 8, m_top, 4),
        # gold-dependent (analysis-only)
        "router_top4_miss":   r_gr >= 4,
        "router_top8_miss":   r_gr >= 8,
        "type_A":             type_arr == 1,
        "type_B":             type_arr == 2,
        "type_C":             type_arr == 3,
    }
    # Reconstruct approximate types if not present
    bnd = split >= 2
    typeA_like = bnd & (m_mg > 0.20) & (m_en < 1.50) & (m_gr < 4)
    typeB_approx = bnd & ~typeA_like & (m_en >= 1.50)
    typeC_approx = bnd & (m_mg > 0.20) & (m_en < 1.50) & (m_gr >= 8)
    if type_arr.max() == 0:
        groups["type_A"] = typeA_like
        groups["type_B"] = typeB_approx
        groups["type_C"] = typeC_approx
    # Non-gold TypeA-like for deployable controllers
    groups["typeA_like_weak"] = bnd & (m_mg > 0.20) & (m_en < 1.50) & overlap_any(r_top, 8, m_top, 4)
    return groups


# ── Coverage/token row builder ────────────────────────────────────────────────

def policy_row(
    policy: str,
    covered: np.ndarray,        # (N,) bool
    tok_per_pos: np.ndarray,    # (N,) float32
    reg_per_pos: np.ndarray,    # (N,) float32
    groups: Dict[str, np.ndarray],
    tpr: np.ndarray,
    memory_used: Optional[np.ndarray] = None,
    fallback: Optional[np.ndarray] = None,
    **extra,
) -> Dict:
    total_mapped = float(tpr.sum())
    avg_tok = float(tok_per_pos.mean())
    avg_reg = float(reg_per_pos.mean())
    row: Dict = {
        "policy": policy,
        "gold_region_coverage": round(float(covered.mean()), 4),
        "avg_regions":          round(avg_reg, 2),
        "avg_tokens":           round(avg_tok, 1),
        "vocab_percent":        round(avg_tok / FULL_VOCAB * 100, 2),
        "mapped_vocab_percent": round(avg_tok / total_mapped * 100, 2) if total_mapped > 0 else float("nan"),
    }
    for gname, gmask in groups.items():
        if gmask.sum() == 0:
            row[f"coverage_{gname}"] = float("nan")
        else:
            row[f"coverage_{gname}"] = round(float(covered[gmask].mean()), 4)
    row["memory_used_rate"] = round(float(memory_used.mean()), 4) if memory_used is not None else 0.0
    row["fallback_rate"]    = round(float(fallback.mean()), 4) if fallback is not None else 0.0
    row.update(extra)
    return row


# ── Part 1: Controller evaluation ─────────────────────────────────────────────

def eval_router_baselines(
    r_gr: np.ndarray, r_top: np.ndarray,
    groups: Dict, tpr: np.ndarray, N: int,
) -> List[Dict]:
    rows = []
    for K in [4, 8, 12, 16, 24, 32]:
        cov = r_gr < K
        tv  = tok_vec(r_top, K, tpr)
        rv  = np.full(N, float(K), dtype=np.float32)
        rows.append(policy_row(f"router_top{K}", cov, tv, rv, groups, tpr))
    return rows


def eval_fixed_union(
    r_gr: np.ndarray, m_gr: np.ndarray,
    r_top: np.ndarray, m_top: np.ndarray,
    groups: Dict, tpr: np.ndarray, N: int,
) -> List[Dict]:
    configs = [
        (4, 2), (4, 4), (8, 2), (8, 4), (8, 8),
        (12, 4), (12, 8), (16, 4), (16, 8), (16, 16),
    ]
    rows = []
    for Kr, Km in configs:
        cov = (r_gr < Kr) | (m_gr < Km)
        I   = scatter_indicator(r_top, Kr) | scatter_indicator(m_top, Km)
        tv  = (I.astype(np.float32) @ tpr)
        rv  = I.sum(axis=1).astype(np.float32)
        mem = np.ones(N, dtype=bool)
        rows.append(policy_row(
            f"union_r{Kr}_m{Km}", cov, tv, rv, groups, tpr,
            memory_used=mem,
        ))
    return rows


def _build_gated_base(
    r_mg: np.ndarray, r_gr: np.ndarray, m_gr: np.ndarray,
    r_top: np.ndarray, m_top: np.ndarray, tpr: np.ndarray, N: int,
):
    """Precompute per-position arrays needed for all gated configs."""
    high = r_mg >= 0.30
    med  = (r_mg >= 0.10) & (r_mg < 0.30)
    bnd  = r_mg < 0.10

    # Coverage atoms
    cov_r4  = r_gr < 4
    cov_r8  = r_gr < 8
    cov_r16 = r_gr < 16
    cov_r8m4 = (r_gr < 8) | (m_gr < 4)

    # Token atoms
    T_r4  = tok_vec(r_top, 4,  tpr)
    T_r8  = tok_vec(r_top, 8,  tpr)
    T_r16 = tok_vec(r_top, 16, tpr)

    # Union tokens for boundary positions (precomputed once)
    T_r8m4 = np.zeros(N, dtype=np.float32)
    if bnd.any():
        I_bnd = scatter_indicator(r_top[bnd], 8) | scatter_indicator(m_top[bnd], 4)
        T_r8m4[bnd] = (I_bnd.astype(np.float32) @ tpr)
        T_r8m4[~bnd] = T_r8[~bnd]  # unused but tidy
    else:
        T_r8m4 = T_r8.copy()

    # Region-count atoms
    R_r4  = np.full(N, 4.0, dtype=np.float32)
    R_r8  = np.full(N, 8.0, dtype=np.float32)
    R_r16 = np.full(N, 16.0, dtype=np.float32)
    R_r8m4 = np.zeros(N, dtype=np.float32)
    if bnd.any():
        I_bnd_obj = scatter_indicator(r_top[bnd], 8) | scatter_indicator(m_top[bnd], 4)
        R_r8m4[bnd] = I_bnd_obj.sum(axis=1).astype(np.float32)
    R_r8m4[~bnd] = R_r8[~bnd]

    # Overlap (for overlap_required=True)
    ovlp = np.zeros(N, dtype=bool)
    if bnd.any():
        ovlp[bnd] = overlap_any(r_top[bnd], 8, m_top[bnd], 4)

    return dict(
        high=high, med=med, bnd=bnd,
        cov_r4=cov_r4, cov_r8=cov_r8, cov_r16=cov_r16, cov_r8m4=cov_r8m4,
        T_r4=T_r4, T_r8=T_r8, T_r16=T_r16, T_r8m4=T_r8m4,
        R_r4=R_r4, R_r8=R_r8, R_r16=R_r16, R_r8m4=R_r8m4,
        ovlp=ovlp,
    )


def eval_gated_memory_grid(
    r_mg: np.ndarray, m_mg: np.ndarray, m_en: np.ndarray,
    r_gr: np.ndarray, m_gr: np.ndarray,
    r_top: np.ndarray, m_top: np.ndarray,
    groups: Dict, tpr: np.ndarray, N: int,
) -> List[Dict]:
    base = _build_gated_base(r_mg, r_gr, m_gr, r_top, m_top, tpr, N)
    high, med, bnd = base["high"], base["med"], base["bnd"]

    RMTHR = [0.05, 0.10, 0.15, 0.20]
    MMTHR = [0.10, 0.15, 0.20, 0.25, 0.30]
    ETHR  = [1.00, 1.25, 1.50, 1.75, 2.00]
    OVLP  = [False, True]

    rows = []
    for rmthr in RMTHR:
        for mmthr in MMTHR:
            for ethr in ETHR:
                for ovlp_req in OVLP:
                    gate = bnd & (r_mg < rmthr) & (m_mg > mmthr) & (m_en < ethr)
                    if ovlp_req:
                        gate = gate & base["ovlp"]

                    bg  = bnd & gate
                    bug = bnd & ~gate

                    cov = (
                        (high & base["cov_r4"]) |
                        (med  & base["cov_r8"]) |
                        (bg   & base["cov_r8m4"]) |
                        (bug  & base["cov_r16"])
                    )
                    tv = (
                        high.astype(np.float32) * base["T_r4"] +
                        med.astype(np.float32)  * base["T_r8"] +
                        bg.astype(np.float32)   * base["T_r8m4"] +
                        bug.astype(np.float32)  * base["T_r16"]
                    )
                    rv = (
                        high.astype(np.float32) * base["R_r4"] +
                        med.astype(np.float32)  * base["R_r8"] +
                        bg.astype(np.float32)   * base["R_r8m4"] +
                        bug.astype(np.float32)  * base["R_r16"]
                    )
                    tag = f"gated_rm{rmthr:.2f}_mm{mmthr:.2f}_e{ethr:.2f}_ovlp{int(ovlp_req)}"
                    rows.append(policy_row(
                        tag, cov, tv, rv, groups, tpr,
                        memory_used=bg,
                        fallback=bug,
                        rmthr=rmthr, mmthr=mmthr, ethr=ethr, overlap_req=int(ovlp_req),
                    ))
    return rows


def eval_typeA_like_policies(
    r_mg: np.ndarray, m_mg: np.ndarray, m_en: np.ndarray,
    r_gr: np.ndarray, m_gr: np.ndarray,
    r_top: np.ndarray, m_top: np.ndarray,
    groups: Dict, tpr: np.ndarray, N: int,
) -> List[Dict]:
    base = _build_gated_base(r_mg, r_gr, m_gr, r_top, m_top, tpr, N)
    high, med, bnd = base["high"], base["med"], base["bnd"]

    MEM_THR = [0.15, 0.20, 0.25, 0.30]
    ENT_THR = [1.25, 1.50, 1.75]
    OVLP    = [False, True]
    FALLBACK_K = [8, 12, 16, 24]

    # Precompute fallback indicators/tokens for boundary ungated
    _T_fb = {k: tok_vec(r_top, k, tpr) for k in FALLBACK_K}
    _R_fb = {k: np.full(N, float(k), dtype=np.float32) for k in FALLBACK_K}
    _cov_fb = {k: r_gr < k for k in FALLBACK_K}

    rows = []
    for mem_thr in MEM_THR:
        for ent_thr in ENT_THR:
            for ovlp_req in OVLP:
                typeA_like = (
                    bnd & (m_mg > mem_thr) & (m_en < ent_thr)
                )
                if ovlp_req:
                    typeA_like = typeA_like & base["ovlp"]
                bnd_other = bnd & ~typeA_like

                for fb_k in FALLBACK_K:
                    cov = (
                        (high & base["cov_r4"]) |
                        (med  & base["cov_r8"]) |
                        (typeA_like & base["cov_r8m4"]) |
                        (bnd_other  & _cov_fb[fb_k])
                    )
                    tv = (
                        high.astype(np.float32) * base["T_r4"] +
                        med.astype(np.float32)  * base["T_r8"] +
                        typeA_like.astype(np.float32) * base["T_r8m4"] +
                        bnd_other.astype(np.float32)  * _T_fb[fb_k]
                    )
                    rv = (
                        high.astype(np.float32) * base["R_r4"] +
                        med.astype(np.float32)  * base["R_r8"] +
                        typeA_like.astype(np.float32) * base["R_r8m4"] +
                        bnd_other.astype(np.float32)  * _R_fb[fb_k]
                    )
                    tag = (f"typeAlike_mm{mem_thr:.2f}_e{ent_thr:.2f}"
                           f"_ovlp{int(ovlp_req)}_fb{fb_k}")
                    rows.append(policy_row(
                        tag, cov, tv, rv, groups, tpr,
                        memory_used=typeA_like,
                        fallback=bnd_other,
                        mem_thr=mem_thr, ent_thr=ent_thr,
                        overlap_req=int(ovlp_req), fallback_k=fb_k,
                    ))
    return rows


def _jsd_batch(
    r_top: np.ndarray, r_prob: np.ndarray,
    m_top: np.ndarray, m_prob: np.ndarray,
    n_fine: int = N_FINE, batch: int = 4096,
) -> np.ndarray:
    N = len(r_top)
    K = r_top.shape[1]
    jsd = np.zeros(N, dtype=np.float32)
    for s in range(0, N, batch):
        e = min(s + batch, N)
        Nb = e - s
        P = np.zeros((Nb, n_fine), dtype=np.float64)
        Q = np.zeros((Nb, n_fine), dtype=np.float64)
        rows_b = np.repeat(np.arange(Nb), K)
        np.add.at(P, (rows_b, r_top[s:e].ravel().astype(np.int32)),
                  r_prob[s:e].ravel().astype(np.float64))
        np.add.at(Q, (rows_b, m_top[s:e].ravel().astype(np.int32)),
                  m_prob[s:e].ravel().astype(np.float64))
        Ps = P.sum(1, keepdims=True) + 1e-10
        Qs = Q.sum(1, keepdims=True) + 1e-10
        P /= Ps; Q /= Qs
        M = 0.5 * (P + Q)
        kl_p = (P * np.log((P + 1e-10) / (M + 1e-10))).sum(1)
        kl_q = (Q * np.log((Q + 1e-10) / (M + 1e-10))).sum(1)
        jsd[s:e] = (0.5 * (kl_p + kl_q)).astype(np.float32)
    return jsd


def eval_learned_controller(
    r_mg: np.ndarray, r_en: np.ndarray,
    m_mg: np.ndarray, m_en: np.ndarray,
    r_top: np.ndarray, r_prob: np.ndarray,
    m_top: np.ndarray, m_prob: np.ndarray,
    r_gr: np.ndarray, m_gr: np.ndarray,
    groups: Dict, tpr: np.ndarray, N: int,
    out_dir: str, seed: int = 42,
) -> List[Dict]:
    if not _HAS_SKLEARN:
        print("[learned_ctrl] scikit-learn not available — skipping Policy 4")
        return []

    print("[learned_ctrl] building features ...")
    feats = np.column_stack([
        r_mg.astype(np.float32),
        r_en.astype(np.float32),
        m_mg.astype(np.float32),
        m_en.astype(np.float32),
        r_prob[:, 0].astype(np.float32),
        m_prob[:, 0].astype(np.float32),
        (r_top[:, 0].astype(np.int32) == m_top[:, 0].astype(np.int32)).astype(np.float32),
        overlap_count(r_top, 4, m_top, 4).astype(np.float32),
        overlap_count(r_top, 8, m_top, 4).astype(np.float32),
        overlap_count(r_top, 8, m_top, 8).astype(np.float32),
        _jsd_batch(r_top, r_prob, m_top, m_prob),
        (m_prob.astype(np.float32) > 0.05).sum(axis=1).astype(np.float32),
    ])

    # Binary label: memory adds gold that router_top8 misses
    y_useful = ((r_gr >= 8) & (m_gr < 4)).astype(np.int32)
    print(f"[learned_ctrl] memory_useful rate: {y_useful.mean():.3f} "
          f"({y_useful.sum():,} / {N:,})")

    # Train/val split (first 70% / last 30%)
    n_train = int(N * 0.70)
    X_tr, X_va = feats[:n_train], feats[n_train:]
    y_tr, y_va = y_useful[:n_train], y_useful[n_train:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=seed, solver="lbfgs", C=1.0,
    )
    clf.fit(X_tr_s, y_tr)

    y_pred_va = clf.predict(X_va_s)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_va, y_pred_va, average="binary", zero_division=0
    )
    print(f"[learned_ctrl] val  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

    # Save model info
    coef_path = os.path.join(out_dir, "learned_controller_coef.json")
    feat_names = [
        "router_margin", "router_entropy", "mem_margin", "mem_entropy",
        "router_top1_prob", "mem_top1_prob", "agreement_top1",
        "overlap_r4m4", "overlap_r8m4", "overlap_r8m8",
        "jsd_approx", "n_mem_high_conf_regions",
    ]
    with open(coef_path, "w") as f:
        json.dump({
            "intercept": float(clf.intercept_[0]),
            "coef": {fn: float(c) for fn, c in zip(feat_names, clf.coef_[0])},
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
        }, f, indent=2)

    # Apply policy: use memory (union r8m4) where clf predicts 1, else use default gated
    y_all_s = scaler.transform(feats)
    mem_pred = clf.predict(y_all_s).astype(bool)

    # Build base masks
    high = r_mg >= 0.30
    med  = (r_mg >= 0.10) & (r_mg < 0.30)
    bnd  = r_mg < 0.10
    use_mem = bnd & mem_pred
    use_fb  = bnd & ~mem_pred

    base = _build_gated_base(r_mg, r_gr, m_gr, r_top, m_top, tpr, N)
    cov = (
        (high & base["cov_r4"]) | (med & base["cov_r8"]) |
        (use_mem & base["cov_r8m4"]) | (use_fb & base["cov_r16"])
    )
    tv = (
        high.astype(np.float32) * base["T_r4"] +
        med.astype(np.float32)  * base["T_r8"] +
        use_mem.astype(np.float32) * base["T_r8m4"] +
        use_fb.astype(np.float32)  * base["T_r16"]
    )
    rv = (
        high.astype(np.float32) * base["R_r4"] +
        med.astype(np.float32)  * base["R_r8"] +
        use_mem.astype(np.float32) * base["R_r8m4"] +
        use_fb.astype(np.float32)  * base["R_r16"]
    )

    rows = []
    rows.append(policy_row(
        "learned_ctrl_binary", cov, tv, rv, groups, tpr,
        memory_used=use_mem, fallback=use_fb,
        precision=round(float(prec), 4), recall=round(float(rec), 4), f1=round(float(f1), 4),
    ))

    # Save learned controller results separately
    lc_rows = [{
        "policy": "learned_ctrl_binary",
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "memory_used_rate": round(float(use_mem.mean()), 4),
        "coverage": round(float(cov.mean()), 4),
        "avg_tokens": round(float(tv.mean()), 1),
    }]
    _csv(lc_rows, os.path.join(out_dir, "learned_controller_results.csv"))
    return rows


# ── Part 2a: Co-occurrence graphs ─────────────────────────────────────────────

def build_cooccurrence_graph(
    top_regions: np.ndarray, top_probs: Optional[np.ndarray],
    K: int, n_fine: int = N_FINE, weighted: bool = True,
) -> np.ndarray:
    K_eff = min(K, top_regions.shape[1])
    regs  = top_regions[:, :K_eff].astype(np.int32)
    A = np.zeros((n_fine, n_fine), dtype=np.float64)
    for ai, bi in combinations(range(K_eff), 2):
        ra = regs[:, ai]
        rb = regs[:, bi]
        if weighted and top_probs is not None:
            p = top_probs.astype(np.float32)
            w = p[:, ai].astype(np.float64) * p[:, bi].astype(np.float64)
        else:
            w = np.ones(len(ra), dtype=np.float64)
        np.add.at(A, (ra, rb), w)
        np.add.at(A, (rb, ra), w)
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    A = A / row_sum
    A = (A + A.T) / 2  # symmetrize (row-norm breaks symmetry; SpectralClustering requires it)
    return A.astype(np.float32)


def save_graph_edges(A: np.ndarray, path: str, top_n: int = 500) -> None:
    n = A.shape[0]
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                rows.append({"region_i": i, "region_j": j, "weight": float(A[i, j])})
    rows.sort(key=lambda x: -x["weight"])
    _csv(rows[:top_n], path)


# ── Part 2b: Superregion clustering ──────────────────────────────────────────

def cluster_superregions(
    A: np.ndarray, n_super: int, method: str = "auto", seed: int = 42,
) -> np.ndarray:
    n_fine = A.shape[0]
    if method == "auto":
        if _HAS_SKLEARN:
            method = "spectral"
        elif _HAS_NX:
            method = "greedy_modularity"
        else:
            method = "agglomerative_dist"

    if method == "spectral" and _HAS_SKLEARN:
        # Use A as affinity matrix (already normalized, non-negative)
        sc = SpectralClustering(
            n_clusters=n_super, affinity="precomputed",
            random_state=seed, assign_labels="kmeans", n_init=10,
        )
        return sc.fit_predict(A).astype(np.int32)

    if (method == "agglomerative" or method == "agglomerative_dist") and _HAS_SKLEARN:
        dist = 1.0 - A  # similarity → distance (A is row-normalized similarity)
        dist = np.clip(dist, 0, None)
        ac = AgglomerativeClustering(
            n_clusters=n_super, metric="precomputed", linkage="average",
        )
        return ac.fit_predict(dist).astype(np.int32)

    if method == "greedy_modularity" and _HAS_NX:
        G = nx.from_numpy_array(A)
        comms = list(nx.community.greedy_modularity_communities(G))
        labels = np.zeros(n_fine, dtype=np.int32)
        for ci, comm in enumerate(comms):
            for node in comm:
                labels[node] = ci
        # Merge small clusters if too many
        return _merge_to_k(labels, n_super, A)

    # Fallback: KMeans on row vectors
    if _HAS_SKLEARN:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_super, random_state=seed, n_init=5)
        return km.fit_predict(A).astype(np.int32)

    # Last resort: evenly divide
    print(f"[WARNING] no clustering library available; using even partition for K={n_super}")
    labels = np.arange(n_fine, dtype=np.int32) % n_super
    return labels


def _merge_to_k(labels: np.ndarray, k: int, A: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    if len(unique) <= k:
        # Remap to 0..len(unique)-1
        mapping = {v: i for i, v in enumerate(unique)}
        return np.array([mapping[l] for l in labels], dtype=np.int32)
    # Merge smallest clusters iteratively
    while len(np.unique(labels)) > k:
        counts = np.bincount(labels, minlength=labels.max() + 1)
        smallest = np.argmin(counts)
        # Find nearest (most similar) cluster
        sims = np.zeros(len(counts))
        idx_s = np.where(labels == smallest)[0]
        for ci in np.unique(labels):
            if ci == smallest:
                continue
            idx_c = np.where(labels == ci)[0]
            sims[ci] = A[np.ix_(idx_s, idx_c)].mean()
        nearest = np.argmax(sims)
        labels[labels == smallest] = nearest
    # Remap
    unique = np.unique(labels)
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[l] for l in labels], dtype=np.int32)


# ── Part 2c: Superregion aggregation ─────────────────────────────────────────

def aggregate_to_superregions(
    top_regions: np.ndarray, top_probs: np.ndarray,
    region_to_super: np.ndarray, n_super: int,
) -> np.ndarray:
    """Return (N, n_super) float32 superregion probability distribution."""
    N, K = top_regions.shape
    super_regs = region_to_super[top_regions.astype(np.int32)]  # (N, K)
    super_probs = np.zeros((N, n_super), dtype=np.float64)
    rows = np.repeat(np.arange(N), K)
    np.add.at(super_probs, (rows, super_regs.ravel()), top_probs.ravel().astype(np.float64))
    # Normalize
    ss = super_probs.sum(axis=1, keepdims=True)
    ss[ss == 0] = 1.0
    return (super_probs / ss).astype(np.float32)


def superregion_metrics(
    super_probs: np.ndarray, gold_region: np.ndarray,
    region_to_super: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (margin, entropy, gold_super_rank)."""
    sorted_p = np.sort(super_probs, axis=1)[:, ::-1]
    margin = (sorted_p[:, 0] - sorted_p[:, 1]).astype(np.float32)
    entropy = (-(super_probs * np.log(super_probs + 1e-10)).sum(axis=1)).astype(np.float32)
    gold_super = region_to_super[gold_region.astype(np.int32)]
    gold_prob  = super_probs[np.arange(len(gold_region)), gold_super]
    gold_rank  = (super_probs > gold_prob[:, None]).sum(axis=1).astype(np.int32)
    return margin, entropy, gold_rank


# ── Part 2d: B-to-A conversion + subtypes ────────────────────────────────────

def compute_B_to_A_conversion(
    super_probs_mem: np.ndarray,
    super_probs_router: np.ndarray,
    gold_region: np.ndarray,
    region_to_super: np.ndarray,
    n_super: int,
    K_label: int,
    margin_thr_list: List[float],
    entropy_thr_list: List[float],
) -> List[Dict]:
    N = len(gold_region)
    if N == 0:
        return []

    sm_mg, sm_en, sm_gr_mem    = superregion_metrics(super_probs_mem, gold_region, region_to_super)
    sr_mg, sr_en, sm_gr_router = superregion_metrics(super_probs_router, gold_region, region_to_super)

    gold_super = region_to_super[gold_region.astype(np.int32)]
    # Coarse accuracy
    m_acc1 = (sm_gr_mem == 0).mean()
    m_acc2 = (sm_gr_mem < 2).mean()
    m_acc4 = (sm_gr_mem < 4).mean()
    r_acc1 = (sm_gr_router == 0).mean()
    r_acc2 = (sm_gr_router < 2).mean()
    r_acc4 = (sm_gr_router < 4).mean()
    # Union coarse acc
    union_I = (
        scatter_indicator(np.argsort(-super_probs_mem,  axis=1), 4, n_super) |
        scatter_indicator(np.argsort(-super_probs_router, axis=1), 4, n_super)
    )
    union_rows = np.arange(N)
    union_cov = union_I[union_rows, gold_super].astype(float).mean()

    rows = []
    for margin_thr in margin_thr_list:
        for entropy_thr in entropy_thr_list:
            B2A_mem = (sm_mg > margin_thr) & (sm_en < entropy_thr) & (sm_gr_mem == 0)
            B2A_rtr = (sr_mg > margin_thr) & (sr_en < entropy_thr) & (sm_gr_router == 0)
            rows.append({
                "K_super":               K_label,
                "margin_thr":            margin_thr,
                "entropy_thr":           entropy_thr,
                "n_typeB":               N,
                "coarse_mem_acc1":       round(float(m_acc1), 4),
                "coarse_mem_acc2":       round(float(m_acc2), 4),
                "coarse_mem_acc4":       round(float(m_acc4), 4),
                "coarse_router_acc1":    round(float(r_acc1), 4),
                "coarse_router_acc2":    round(float(r_acc2), 4),
                "coarse_router_acc4":    round(float(r_acc4), 4),
                "coarse_union_cov4":     round(float(union_cov), 4),
                "B2A_mem_rate":          round(float(B2A_mem.mean()), 4),
                "B2A_router_rate":       round(float(B2A_rtr.mean()), 4),
            })
    return rows


def classify_B_subtypes(
    super_probs_mem: np.ndarray,
    super_probs_router: np.ndarray,
    gold_region: np.ndarray,
    region_to_super: np.ndarray,
    n_super: int,
    K_label: int,
    margin_thr: float = 0.20,
    entropy_thr: float = 1.50,
) -> Dict:
    N = len(gold_region)
    if N == 0:
        return {}

    sm_mg, sm_en, sm_gr_mem = superregion_metrics(super_probs_mem, gold_region, region_to_super)
    sr_mg, sr_en, sm_gr_router = superregion_metrics(super_probs_router, gold_region, region_to_super)
    gold_super = region_to_super[gold_region.astype(np.int32)]

    # B1: coarse is sharp and gold inside top-2
    B1 = (sm_mg > margin_thr) & (sm_en < entropy_thr) & (sm_gr_mem < 2)
    # B2: gold not in union coarse top-4
    union_top4_I = (
        scatter_indicator(np.argsort(-super_probs_mem,    axis=1), 4, n_super) |
        scatter_indicator(np.argsort(-super_probs_router, axis=1), 4, n_super)
    )
    gold_in_union4 = union_top4_I[np.arange(N), gold_super]
    B2 = ~gold_in_union4
    # B3: coarse still spread but gold inside top-4 union
    B3 = (sm_en >= entropy_thr) & gold_in_union4
    # Remainder
    rest = ~B1 & ~B2 & ~B3

    return {
        "K_super": K_label,
        "n_typeB": N,
        "n_B1": int(B1.sum()), "pct_B1": round(float(B1.mean()) * 100, 2),
        "n_B2": int(B2.sum()), "pct_B2": round(float(B2.mean()) * 100, 2),
        "n_B3": int(B3.sum()), "pct_B3": round(float(B3.mean()) * 100, 2),
        "n_rest": int(rest.sum()), "pct_rest": round(float(rest.mean()) * 100, 2),
        "B1_coarse_mem_acc1": round(float(sm_gr_mem[B1].mean()) if B1.any() else float("nan"), 4),
    }


# ── Part 3: Hierarchical candidate policies ───────────────────────────────────

def build_super_children(
    region_to_super: np.ndarray, n_super: int, n_fine: int = N_FINE,
) -> Dict[int, np.ndarray]:
    children = {}
    for s in range(n_super):
        children[s] = np.where(region_to_super == s)[0].astype(np.int32)
    return children


def eval_hierarchical_policies(
    r_mg: np.ndarray, m_mg: np.ndarray, m_en: np.ndarray,
    r_gr: np.ndarray, m_gr: np.ndarray,
    r_top: np.ndarray, m_top: np.ndarray,
    super_probs_mem: np.ndarray,
    super_probs_router: np.ndarray,
    region_to_super: np.ndarray,
    children: Dict[int, np.ndarray],
    n_super: int, K_label: int,
    groups: Dict, tpr: np.ndarray, N: int,
    gold_region: np.ndarray,
    coarse_margin_thr: float = 0.20,
    coarse_entropy_thr: float = 1.50,
) -> List[Dict]:
    base = _build_gated_base(r_mg, r_gr, m_gr, r_top, m_top, tpr, N)
    high, med, bnd = base["high"], base["med"], base["bnd"]

    # Gold fine → coarse mapping (used for correct coverage)
    gold_super = region_to_super[gold_region.astype(np.int32)]  # (N,)

    # Gold-free Type-A-like detector
    typeA_like = bnd & (m_mg > 0.20) & (m_en < 1.50) & base["ovlp"]

    # Coarse confident: non-gold signal — use margin/entropy only
    # Pass dummy gold=0 since we only need margin and entropy, not gold rank.
    sm_mg = superregion_metrics(super_probs_mem,    np.zeros(N, dtype=np.int16), region_to_super)[0]
    sm_en = superregion_metrics(super_probs_mem,    np.zeros(N, dtype=np.int16), region_to_super)[1]
    sr_mg = superregion_metrics(super_probs_router, np.zeros(N, dtype=np.int16), region_to_super)[0]
    sr_en = superregion_metrics(super_probs_router, np.zeros(N, dtype=np.int16), region_to_super)[1]
    coarse_conf = bnd & ~typeA_like & (
        ((sm_mg > coarse_margin_thr) & (sm_en < coarse_entropy_thr)) |
        ((sr_mg > coarse_margin_thr) & (sr_en < coarse_entropy_thr))
    )
    bnd_fallback = bnd & ~typeA_like & ~coarse_conf

    combined_super = 0.5 * super_probs_mem + 0.5 * super_probs_router

    variants = [
        # (name, n_top_super, base_Kr, fallback_K, use_mem_sup, use_router_sup, use_combined)
        ("hier_v1", 1, 8,  16, False, False, True),
        ("hier_v2", 2, 8,  16, False, False, True),
        ("hier_v3", 1, 12, 24, False, False, True),
        ("hier_v4", 2, 8,  16, True,  False, False),
    ]

    rows = []
    for vname, n_ts, base_kr, fb_k, use_m, use_r, use_c in variants:
        sup_ref = (super_probs_mem    if use_m else
                   super_probs_router if use_r else
                   combined_super)

        # Token + region count
        T = np.zeros(N, dtype=np.float32)
        R = np.zeros(N, dtype=np.float32)
        T[high] = base["T_r4"][high];  R[high] = 4.0
        T[med]  = base["T_r8"][med];   R[med]  = 8.0
        T[typeA_like] = base["T_r8m4"][typeA_like]
        R[typeA_like] = base["R_r8m4"][typeA_like]
        T[bnd_fallback] = tok_vec(r_top, fb_k, tpr)[bnd_fallback]
        R[bnd_fallback] = float(fb_k)

        # Coverage
        cov = np.zeros(N, dtype=bool)
        cov[high] = base["cov_r4"][high]
        cov[med]  = base["cov_r8"][med]
        cov[typeA_like] = base["cov_r8m4"][typeA_like]
        cov[bnd_fallback] = r_gr[bnd_fallback] < fb_k

        cm = coarse_conf
        if cm.any():
            top_supers = np.argsort(-sup_ref[cm], axis=1)[:, :n_ts]  # (N_cm, n_ts)
            # Token expansion: start with router base, add children of top superregions
            I_h = scatter_indicator(r_top[cm], base_kr)
            for k_idx in range(n_ts):
                for si in np.unique(top_supers[:, k_idx]):
                    pos = top_supers[:, k_idx] == si
                    if children[si].size:
                        I_h[np.ix_(np.where(pos)[0], children[si])] = True
            T[cm] = (I_h.astype(np.float32) @ tpr)
            R[cm] = I_h.sum(axis=1).astype(np.float32)
            # Coverage: gold fine region covered by router base OR gold superregion
            # is among the selected superregions (hierarchy coverage)
            gold_super_cm = gold_super[cm]
            gold_in_hier  = np.any(top_supers == gold_super_cm[:, np.newaxis], axis=1)
            cov[cm] = (r_gr[cm] < base_kr) | gold_in_hier

        rows.append(policy_row(
            f"{vname}_K{K_label}", cov, T, R, groups, tpr,
            memory_used=typeA_like | cm,
            fallback=bnd_fallback,
            K_super=K_label, n_top_super=n_ts, base_kr=base_kr, fallback_k=fb_k,
        ))

    return rows


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(
    out_dir: str,
    ctrl_rows: List[Dict],
    type_stats: Dict,
    A_typeB_mem: np.ndarray,
    super_results: Dict,
    hier_rows: List[Dict],
    groups: Dict,
    r_mg: np.ndarray, m_mg: np.ndarray,
    r_en: np.ndarray, m_en: np.ndarray,
    type_arr: np.ndarray,
) -> None:
    if not _HAS_PLT:
        print("[plots] matplotlib not available — skipping")
        return
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Plot 01: Type A/B/C memory effect ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    type_names = ["type_A", "type_B", "type_C"]
    colors = ["#2196F3", "#FF9800", "#F44336"]
    for ax, tname, col in zip(axes, type_names, colors):
        stats = type_stats.get(tname, {})
        vals = [stats.get("router_nll", float("nan")),
                stats.get("mem_nll", float("nan")),
                stats.get("mix_nll", float("nan"))]
        valid = [v for v in vals if not math.isnan(v)]
        if not valid:
            ax.set_title(f"{tname} (no data)")
            continue
        for xi, (label, val, al) in enumerate(zip(
                ["router", "memory", "mix(0.5)"], vals, [0.5, 0.7, 1.0])):
            ax.bar([xi], [val], color=col, alpha=al)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["router", "memory", "mix(0.5)"])
        ax.set_title(f"{tname}\n(n={stats.get('n', 0):,})")
        ax.set_ylabel("NLL")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_typeA_typeB_typeC_memory_effect.png"), dpi=120)
    plt.close()

    # ── Plot 02: Controller coverage vs tokens ───────────────────────────────
    if ctrl_rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        policy_types = {
            "router": ("#2196F3", "o"),
            "union":  ("#4CAF50", "s"),
            "gated":  ("#FF9800", "^"),
            "typeA":  ("#E91E63", "D"),
            "learn":  ("#9C27B0", "*"),
        }
        for row in ctrl_rows:
            pol = row["policy"]
            cov = _flt(row, "gold_region_coverage")
            tok = _flt(row, "avg_tokens")
            if math.isnan(cov) or math.isnan(tok):
                continue
            if pol.startswith("router"):   style = "router"
            elif pol.startswith("union"):  style = "union"
            elif pol.startswith("gated"):  style = "gated"
            elif pol.startswith("typeA"):  style = "typeA"
            else:                          style = "learn"
            c, m = policy_types[style]
            ax.scatter(tok / FULL_VOCAB * 100, cov, c=c, marker=m, s=40, alpha=0.6)
        # Add reference lines
        ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.8, label="90% coverage")
        ax.axhline(0.95, color="gray", linestyle=":",  linewidth=0.8, label="95% coverage")
        ax.axvline(8.0,  color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(15.0, color="gray", linestyle=":",  linewidth=0.8)
        patches = [mpatches.Patch(color=c, label=k) for k, (c, _) in policy_types.items()]
        ax.legend(handles=patches, fontsize=8)
        ax.set_xlabel("Avg vocab % used"); ax.set_ylabel("Gold region coverage")
        ax.set_title("Controller Pareto: coverage vs vocab %")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "02_controller_coverage_vs_tokens.png"), dpi=120)
        plt.close()

    # ── Plot 03: Type-B region co-occurrence heatmap ─────────────────────────
    if A_typeB_mem is not None and A_typeB_mem.size > 0:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(A_typeB_mem, cmap="hot_r", aspect="auto")
        ax.set_title("Type-B memory top-8 co-occurrence (row-normalized)")
        ax.set_xlabel("Region index"); ax.set_ylabel("Region index")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "03_typeB_region_cooccurrence_heatmap.png"), dpi=120)
        plt.close()

    # ── Plot 04: Superregion B-to-A conversion rate by K ────────────────────
    conv_data = super_results.get("conversion_rows", [])
    if conv_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        from collections import defaultdict
        by_K: Dict[int, List] = defaultdict(list)
        for row in conv_data:
            if _flt(row, "margin_thr") == 0.20 and _flt(row, "entropy_thr") == 1.50:
                by_K[int(row["K_super"])].append(_flt(row, "B2A_mem_rate"))
        Ks = sorted(by_K.keys())
        rates = [float(np.mean(by_K[k])) if by_K[k] else float("nan") for k in Ks]
        ax.plot(Ks, rates, "o-", color="#2196F3")
        ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="25% target")
        ax.axhline(0.40, color="gray", linestyle=":",  linewidth=0.8, label="40% strong target")
        ax.set_xlabel("Number of predictive super-regions K")
        ax.set_ylabel("Type-B → B1 coarse-resolvable rate")
        ax.set_title("Type-B Coarse Conversion Rate vs Super-Region Resolution")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "04_superregion_conversion_rate_by_K.png"), dpi=120)
        plt.close()

    # ── Plot 05: B1/B2/B3 breakdown by K ────────────────────────────────────
    subtype_data = super_results.get("subtype_rows", [])
    if subtype_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        Ks   = [row["K_super"] for row in subtype_data]
        B1   = [row["pct_B1"]   for row in subtype_data]
        B2   = [row["pct_B2"]   for row in subtype_data]
        B3   = [row["pct_B3"]   for row in subtype_data]
        rest = [row["pct_rest"] for row in subtype_data]
        ax.stackplot(Ks, B1, B2, B3, rest,
                     labels=["B1 coarse-resolvable", "B2 key-failure",
                             "B3 true-multimodal", "rest"],
                     colors=["#4CAF50", "#F44336", "#FF9800", "#9E9E9E"], alpha=0.8)
        ax.set_xlabel("K super-regions"); ax.set_ylabel("% of Type-B positions")
        ax.set_title("Type-B Subtype Breakdown by Predictive Super-Region Resolution")
        ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "05_B1_B2_B3_breakdown_by_K.png"), dpi=120)
        plt.close()

    # ── Plot 06: Hierarchical policy coverage vs vocab ───────────────────────
    if hier_rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        for row in hier_rows:
            pol = row["policy"]
            cov = _flt(row, "gold_region_coverage")
            voc = _flt(row, "vocab_percent")
            if math.isnan(cov) or math.isnan(voc):
                continue
            color = {"hier_v1": "#2196F3", "hier_v2": "#4CAF50",
                     "hier_v3": "#FF9800",  "hier_v4": "#E91E63"}.get(pol.split("_K")[0], "gray")
            ax.scatter(voc, cov, c=color, s=50, alpha=0.7)
            ax.annotate(pol.split("_K")[0] + f"(K={row.get('K_super','')})",
                        (voc, cov), fontsize=7, alpha=0.7)
        ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(8.0,  color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(15.0, color="gray", linestyle=":",  linewidth=0.8)
        ax.set_xlabel("Vocab %"); ax.set_ylabel("Coverage")
        ax.set_title("Hierarchical Policy Coverage vs Vocab %")
        patches = [mpatches.Patch(color=c, label=k) for k, c in
                   [("hier_v1","#2196F3"),("hier_v2","#4CAF50"),
                    ("hier_v3","#FF9800"),("hier_v4","#E91E63")]]
        ax.legend(handles=patches, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "06_hierarchical_policy_coverage_vs_vocab.png"), dpi=120)
        plt.close()

    # ── Plot 07: Router vs memory superregion entropy (Type B) ───────────────
    if super_results.get("best_K_data"):
        d = super_results["best_K_data"]
        sm_en_B = d.get("sm_en_B"); sr_en_B = d.get("sr_en_B")
        if sm_en_B is not None and sr_en_B is not None and len(sm_en_B) > 0:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(sr_en_B[:5000], sm_en_B[:5000], s=5, alpha=0.3, c="#FF9800")
            lim = max(sr_en_B.max(), sm_en_B.max())
            ax.plot([0, lim], [0, lim], "k--", linewidth=0.8)
            ax.set_xlabel("Router super-entropy (Type B)")
            ax.set_ylabel("Memory super-entropy (Type B)")
            ax.set_title(f"Super-Region Entropy: Router vs Memory (Type B, K={d.get('K')})")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "07_router_vs_memory_superregion_entropy.png"), dpi=120)
            plt.close()

    print(f"[plots] saved to {plots_dir}/")


# ── Final report ──────────────────────────────────────────────────────────────

def write_final_report(
    out_dir: str,
    ctrl_rows: List[Dict],
    super_results: Dict,
    hier_rows: List[Dict],
    learned_rows: List[Dict],
    groups: Dict,
    knn_config: dict,
    n_eval: int,
) -> None:
    path = os.path.join(out_dir, "final_report.md")
    lines = []
    def h(s): lines.append(f"\n## {s}\n")
    def p(s): lines.append(s)
    def hr(): lines.append("\n---\n")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append("# Hard-Position Memory + Predictive Hierarchy Analysis\n")
    lines.append(f"*Generated: {now}*  \n")
    lines.append(f"*Eval positions: N={n_eval:,}*\n")
    if knn_config:
        lines.append(f"*kNN config: k={knn_config.get('knn_k','?')}  "
                     f"temp={knn_config.get('knn_temp','?')}  "
                     f"mem={knn_config.get('max_memory_positions','?')}  "
                     f"key_src={knn_config.get('knn_key_source','?')}*\n")

    # Helpers
    def best_policy(rows, filter_fn=None, metric="gold_region_coverage"):
        cands = [r for r in rows if filter_fn is None or filter_fn(r)]
        if not cands:
            return None
        return max(cands, key=lambda r: _flt(r, metric))

    def policy_summary(row):
        if row is None:
            return "N/A"
        cov = _flt(row, "coverage_all", _flt(row, "gold_region_coverage"))
        tok = _flt(row, "avg_tokens")
        voc = _flt(row, "vocab_percent")
        typeA_cov = _flt(row, "coverage_type_A")
        return (f"policy={row['policy']}  cov={cov:.3f}  "
                f"avg_tok={tok:.0f}  vocab={voc:.1f}%  "
                f"typeA_cov={typeA_cov:.3f}")

    hr()
    h("Q1. Can we detect Type-A-like positions without gold?")
    best_gated = best_policy(ctrl_rows, lambda r: r["policy"].startswith("gated"),
                             "coverage_type_A")
    best_tAl   = best_policy(ctrl_rows, lambda r: r["policy"].startswith("typeAlike"),
                             "coverage_type_A")
    if best_gated:
        p(f"Best gold-free gated policy for Type-A coverage:\n  {policy_summary(best_gated)}")
    if best_tAl:
        p(f"\nBest Type-A-like policy:\n  {policy_summary(best_tAl)}")
    # Compare to router_top8
    rt8 = next((r for r in ctrl_rows if r["policy"] == "router_top8"), None)
    if rt8 and best_tAl:
        tA_delta = _flt(best_tAl, "coverage_type_A") - _flt(rt8, "coverage_type_A")
        tok_ratio = _flt(best_tAl, "avg_tokens") / max(_flt(rt8, "avg_tokens"), 1)
        p(f"\nType-A coverage delta vs router_top8: {tA_delta:+.4f}  "
          f"token ratio: {tok_ratio:.2f}x")
        if tA_delta > 0.01:
            p("**YES** — gold-free Type-A-like detector improves Type-A coverage.")
        else:
            p("**MARGINAL** — Type-A-like detector does not clearly improve Type-A coverage.")

    h("Q2. Does memory help when gated to Type-A-like positions?")
    # Type-A coverage for union_r8_m4 vs router_top8
    union84 = next((r for r in ctrl_rows if r["policy"] == "union_r8_m4"), None)
    if rt8 and union84:
        tA_gain = _flt(union84, "coverage_type_A") - _flt(rt8, "coverage_type_A")
        tB_gain = _flt(union84, "coverage_type_B") - _flt(rt8, "coverage_type_B")
        tC_gain = _flt(union84, "coverage_type_C") - _flt(rt8, "coverage_type_C")
        p(f"union_r8_m4 vs router_top8:")
        p(f"  Type-A coverage change: {tA_gain:+.4f}")
        p(f"  Type-B coverage change: {tB_gain:+.4f}")
        p(f"  Type-C coverage change: {tC_gain:+.4f}")
        tok_cost = _flt(union84, "avg_tokens") - _flt(rt8, "avg_tokens")
        p(f"  Extra avg tokens: +{tok_cost:.0f}")
        if tA_gain > 0 and tB_gain >= -0.01:
            p("\n**YES** — memory gated to boundary positions helps Type A without hurting Type B.")
        else:
            p("\n**PARTIAL** — memory helps some types but the gating needs refinement.")

    h("Q3. Does Type B contain stable predictive co-occurrence groups?")
    conv_rows = super_results.get("conversion_rows", [])
    if conv_rows:
        # Find best B2A rate
        best_conv = max(conv_rows, key=lambda r: _flt(r, "B2A_mem_rate"))
        p(f"Best B-to-A conversion:  K={best_conv['K_super']}  "
          f"margin_thr={best_conv['margin_thr']}  entropy_thr={best_conv['entropy_thr']}\n"
          f"  B2A_mem_rate={_flt(best_conv,'B2A_mem_rate'):.3f}  "
          f"coarse_mem_acc1={_flt(best_conv,'coarse_mem_acc1'):.3f}")
        n_B = int(best_conv.get("n_typeB", 0))
        p(f"  Type B count: {n_B:,}")
        best_rate = _flt(best_conv, "B2A_mem_rate")
        if best_rate >= 0.40:
            p("\n**STRONG YES** — ≥40% of Type B becomes coarse-resolvable.")
        elif best_rate >= 0.25:
            p("\n**YES** — ≥25% of Type B becomes coarse-resolvable.")
        else:
            p(f"\n**NO** — only {best_rate*100:.1f}% of Type B becomes coarse-resolvable.")
    else:
        p("Superregion conversion data not available.")

    h("Q4. What superregion resolution works best?")
    if conv_rows:
        # Group by K, pick best B2A_mem_rate
        from collections import defaultdict
        by_K_conv: Dict = defaultdict(list)
        for row in conv_rows:
            by_K_conv[int(row["K_super"])].append(_flt(row, "B2A_mem_rate"))
        p("| K super-regions | Best B2A_mem_rate | Coarse mem acc@1 |")
        p("|-----------------|------------------|-----------------|")
        for K in SUPER_K_LIST:
            if K not in by_K_conv:
                continue
            rates = by_K_conv[K]
            best_K_rate = max(rates)
            best_K_row = max(
                [r for r in conv_rows if int(r["K_super"]) == K],
                key=lambda r: _flt(r, "B2A_mem_rate")
            )
            p(f"| {K:3d} | {best_K_rate:.3f} | {_flt(best_K_row,'coarse_mem_acc1'):.3f} |")

    h("Q5. What fraction of Type B becomes coarse-resolvable?")
    subtype_rows = super_results.get("subtype_rows", [])
    if subtype_rows:
        p("B1 = coarse-resolvable  B2 = key-failure  B3 = true-multimodal\n")
        p("| K | n_B1 | %B1 | n_B2 | %B2 | n_B3 | %B3 |")
        p("|---|------|-----|------|-----|------|-----|")
        for row in subtype_rows:
            p(f"| {row['K_super']} | {row['n_B1']} | {row['pct_B1']:.1f}% | "
              f"{row['n_B2']} | {row['pct_B2']:.1f}% | "
              f"{row['n_B3']} | {row['pct_B3']:.1f}% |")
    else:
        p("Subtype data not available.")

    h("Q6. How much Type B is B1/B2/B3?")
    if subtype_rows:
        best_sub = max(subtype_rows, key=lambda r: r.get("pct_B1", 0))
        p(f"At K={best_sub['K_super']}: "
          f"B1={best_sub['pct_B1']:.1f}%  B2={best_sub['pct_B2']:.1f}%  "
          f"B3={best_sub['pct_B3']:.1f}%  rest={best_sub['pct_rest']:.1f}%")
        p("\nInterpretation:")
        p("- B1: these positions should be resolvable by expanding candidate set to "
          "child regions of the top predictive super-region.")
        p("- B2: retrieval key or coarse map failure; even hierarchy doesn't help.")
        p("- B3: genuinely multimodal — the model must pick from many plausible continuations.")

    h("Q7. Does hierarchical candidate selection beat router-only / adaptive_v3?")
    # Reference baselines from background
    REF = [
        ("router_top8",  0.901, 7.9),
        ("adaptive_v3",  0.907, 7.4),
        ("union_r16_m4", 0.951, 14.4),
    ]
    p("Reference baselines (from extensive sweep):\n")
    for name, cov, voc in REF:
        p(f"  {name}: coverage={cov:.3f}  vocab={voc:.1f}%")
    if hier_rows:
        best_hier = best_policy(hier_rows, metric="gold_region_coverage")
        p(f"\nBest hierarchical policy: {policy_summary(best_hier)}")
        best_cov = _flt(best_hier, "gold_region_coverage", _flt(best_hier, "coverage_all"))
        best_voc = _flt(best_hier, "vocab_percent")
        beats_ref = (best_cov >= 0.90 and best_voc <= 8.0) or (best_cov >= 0.95 and best_voc <= 15.0)
        if beats_ref:
            p("\n**YES** — hierarchical policy meets coverage/cost target.")
        else:
            p(f"\n**NOT YET** — best hier policy: cov={best_cov:.3f} vocab={best_voc:.1f}%")
    else:
        p("\nHierarchical policy results not available.")

    h("Q8. Should we proceed to a path-conditioned refiner?")
    # Evaluate all success criteria
    ctrl_ok = False
    if best_tAl:
        tA_gain = _flt(best_tAl, "coverage_type_A") - _flt(rt8, "coverage_type_A") if rt8 else 0
        ctrl_ok = tA_gain > 0.005

    hier_ok = False
    if hier_rows:
        best_hier = best_policy(hier_rows, metric="gold_region_coverage")
        bc = _flt(best_hier, "gold_region_coverage", _flt(best_hier, "coverage_all"))
        bv = _flt(best_hier, "vocab_percent")
        hier_ok = (bc >= 0.90 and bv <= 8.0) or (bc >= 0.95 and bv <= 15.0)

    hierarchy_useful = False
    if conv_rows:
        best_B2A = max(_flt(r, "B2A_mem_rate") for r in conv_rows)
        hierarchy_useful = best_B2A >= 0.25

    p(f"Hard-position controller works: **{'YES' if ctrl_ok else 'NO'}**")
    p(f"Hierarchical policy meets coverage/cost target: **{'YES' if hier_ok else 'NO'}**")
    p(f"Type-B hierarchy is useful (>=25% resolvable): **{'YES' if hierarchy_useful else 'NO'}**")

    if ctrl_ok and (hier_ok or hierarchy_useful):
        p("\n**RECOMMENDATION: PROCEED to path-conditioned refiner.**")
        p("\nNext architecture:")
        p("```")
        p("h → router(fine regions)")
        p("  → retrieval_proj(h) → Region-kNN memory")
        p("  → predictive super-region aggregation")
        p("  → controller selects candidate fine regions")
        p("  → path-conditioned refiner scores candidates")
        p("```")
    else:
        p("\n**RECOMMENDATION: CONTINUE improving hierarchy/controller.**")
        if not ctrl_ok:
            p("- Improve Type-A-like detector (try stronger mem_margin thresholds or richer features).")
        if not hierarchy_useful:
            p("- Type-B hierarchy not yet useful — check if B2 (key-failure) dominates.")
        if not hier_ok:
            p("- Hierarchical candidate policy not meeting coverage/cost target — "
              "tune coarse_conf threshold or increase top_super expansion.")

    hr()
    p("*Output files:*\n")
    for fn in [
        "hard_position_controller_results.csv",
        "learned_controller_results.csv",
        "typeB_mem_region_graph.csv",
        "typeB_coarse_conversion.csv",
        "typeB_subtype_breakdown.csv",
        "hierarchical_candidate_policies.csv",
        "plots/",
    ]:
        p(f"- `{fn}`")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {path}")


# ── NLL helpers for type stats ────────────────────────────────────────────────

def _type_nll_stats(
    r_prob: np.ndarray, m_prob: np.ndarray,
    r_gr: np.ndarray, m_gr: np.ndarray,
    mask: np.ndarray,
    clip: float = 1e-7,
) -> Dict:
    if mask.sum() == 0:
        return {}
    K = r_prob.shape[1]
    def _nll(prob, gr, m):
        nll = np.full(m.sum(), -math.log(clip), dtype=np.float32)
        in_k = gr[m] < K
        if in_k.any():
            idx = np.where(in_k)[0]
            p = prob[m][idx, gr[m][idx].astype(np.int32)].astype(np.float32)
            nll[idx] = -np.log(np.maximum(p, clip))
        return float(nll.mean())
    r_nll = _nll(r_prob, r_gr, mask)
    m_nll = _nll(m_prob, m_gr, mask)
    mix_p = 0.5 * r_prob[mask] + 0.5 * m_prob[mask]
    mix_gr = np.minimum(r_gr[mask], m_gr[mask])  # approx mix gold rank
    mix_nll = _nll(mix_p, mix_gr, np.ones(mask.sum(), dtype=bool))
    return {"n": int(mask.sum()), "router_nll": r_nll, "mem_nll": m_nll, "mix_nll": mix_nll}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--knn_run_dir",    required=True)
    p.add_argument("--region_map_path", required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--topk",           type=int, default=32)
    p.add_argument("--device",         default="cpu",
                   help="Accepted but unused (all computation is numpy-based).")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--n_fine",         type=int, default=N_FINE)
    p.add_argument("--coarse_margin_thr",  type=float, default=0.20)
    p.add_argument("--coarse_entropy_thr", type=float, default=1.50)
    p.add_argument("--skip_plots",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    pp  = load_per_position(args.knn_run_dir)
    tpr = load_tpr(args.region_map_path, args.n_fine)
    cfg = load_knn_config(args.knn_run_dir)

    total_mapped = float(tpr.sum())
    N = len(pp["gold_region"])
    K = min(args.topk, pp["mem_topk_regions"].shape[1])
    print(f"[main] N={N:,}  K={K}  n_fine={args.n_fine}  total_mapped={total_mapped:.0f}")

    # Required router fields
    if "router_topk_regions" not in pp:
        raise KeyError(
            "[ERROR] router_topk_regions not in per_position.npz. "
            "The probe must be active during the kNN run (pass --probe_temp 1.0)."
        )

    # Core arrays
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

    # ── Hard groups ───────────────────────────────────────────────────────────
    print("[main] building hard groups ...")
    groups = build_hard_groups(split, type_a, r_mg, m_mg, m_en, r_en,
                                r_gr, m_gr, r_top, m_top)

    # Print group sizes
    for gname, gmask in groups.items():
        print(f"  {gname:30s}: {gmask.sum():8,} ({gmask.mean()*100:.1f}%)")

    # Type NLL stats for report
    type_stats: Dict[str, Dict] = {}
    for tname, tidx in [("type_A", 1), ("type_B", 2), ("type_C", 3)]:
        type_stats[tname] = _type_nll_stats(r_prob, m_prob, r_gr, m_gr, type_a == tidx)

    # ── Part 1: Controller evaluation ─────────────────────────────────────────
    print("\n[Part 1] controller evaluation ...")
    ctrl_rows: List[Dict] = []

    ctrl_rows += eval_router_baselines(r_gr, r_top, groups, tpr, N)
    ctrl_rows += eval_fixed_union(r_gr, m_gr, r_top, m_top, groups, tpr, N)
    print(f"  baselines+union: {len(ctrl_rows)} configs done")

    gated_rows = eval_gated_memory_grid(r_mg, m_mg, m_en, r_gr, m_gr,
                                         r_top, m_top, groups, tpr, N)
    ctrl_rows += gated_rows
    print(f"  gated grid: {len(gated_rows)} configs done")

    tAl_rows = eval_typeA_like_policies(r_mg, m_mg, m_en, r_gr, m_gr,
                                         r_top, m_top, groups, tpr, N)
    ctrl_rows += tAl_rows
    print(f"  typeA-like: {len(tAl_rows)} configs done")

    learned_rows = eval_learned_controller(
        r_mg, r_en, m_mg, m_en, r_top, r_prob, m_top, m_prob,
        r_gr, m_gr, groups, tpr, N, args.output_dir, args.seed,
    )
    ctrl_rows += learned_rows
    print(f"  learned: {len(learned_rows)} configs done")

    _csv(ctrl_rows, os.path.join(args.output_dir, "hard_position_controller_results.csv"))
    print(f"[Part 1] saved {len(ctrl_rows)} rows  ({time.time()-t0:.1f}s)")

    # ── Part 2: Type B predictive hierarchy ───────────────────────────────────
    print("\n[Part 2] Type-B predictive hierarchy ...")
    bnd_B = type_a == 2
    n_B   = int(bnd_B.sum())
    print(f"  Type B: {n_B:,} positions")

    A_typeB_mem    = None
    A_typeB_router = None
    A_typeB_union  = None

    if n_B > 0:
        m_top_B  = m_top[bnd_B]
        m_prob_B = m_prob[bnd_B]
        r_top_B  = r_top[bnd_B]
        r_prob_B = r_prob[bnd_B]

        print("  building co-occurrence graphs ...")
        A_typeB_mem    = build_cooccurrence_graph(m_top_B, m_prob_B, K=8)
        A_typeB_router = build_cooccurrence_graph(r_top_B, r_prob_B, K=8)
        # Union top-8: combine indices
        union_top = np.concatenate([m_top_B[:, :4], r_top_B[:, :4]], axis=1)
        union_prob = np.concatenate([m_prob_B[:, :4], r_prob_B[:, :4]], axis=1)
        A_typeB_union  = build_cooccurrence_graph(union_top, union_prob, K=8)

        save_graph_edges(A_typeB_mem,    os.path.join(args.output_dir, "typeB_mem_region_graph.csv"))
        save_graph_edges(A_typeB_router, os.path.join(args.output_dir, "typeB_router_region_graph.csv"))
        save_graph_edges(A_typeB_union,  os.path.join(args.output_dir, "typeB_union_region_graph.csv"))
        print("  co-occurrence graphs saved")

    # Cluster into superregions and compute conversion
    super_results: Dict = {"conversion_rows": [], "subtype_rows": [], "best_K_data": None}
    all_hier_rows: List[Dict] = []
    region_to_super_all: Dict[int, np.ndarray] = {}

    for K_sup in SUPER_K_LIST:
        print(f"  clustering K={K_sup} ...")
        A_cluster = A_typeB_mem if A_typeB_mem is not None else np.eye(args.n_fine)
        r2s = cluster_superregions(A_cluster, K_sup, seed=args.seed)
        region_to_super_all[K_sup] = r2s

        # Save mapping
        r2s_path = os.path.join(args.output_dir, f"region_to_superregion_K{K_sup}.json")
        with open(r2s_path, "w") as f:
            json.dump({str(r): int(s) for r, s in enumerate(r2s)}, f, indent=2)

        if n_B == 0:
            continue

        # Aggregate Type B to superregions
        sp_mem_B    = aggregate_to_superregions(m_top_B, m_prob_B, r2s, K_sup)
        sp_router_B = aggregate_to_superregions(r_top_B, r_prob_B, r2s, K_sup)
        gold_B = gold[bnd_B]

        # Conversion metrics
        conv_rows = compute_B_to_A_conversion(
            sp_mem_B, sp_router_B, gold_B, r2s, K_sup, K_sup,
            margin_thr_list=[0.10, 0.15, 0.20, 0.25, 0.30],
            entropy_thr_list=[1.00, 1.25, 1.50, 1.75, 2.00],
        )
        super_results["conversion_rows"].extend(conv_rows)

        # Subtype breakdown
        sub = classify_B_subtypes(sp_mem_B, sp_router_B, gold_B, r2s, K_sup, K_sup,
                                   margin_thr=args.coarse_margin_thr,
                                   entropy_thr=args.coarse_entropy_thr)
        if sub:
            super_results["subtype_rows"].append(sub)

        # Store entropy for best-K plot (use K=32 as representative)
        if K_sup == 32:
            sm_mg_B, sm_en_B, _ = superregion_metrics(sp_mem_B, gold_B, r2s)
            sr_mg_B, sr_en_B, _ = superregion_metrics(sp_router_B, gold_B, r2s)
            super_results["best_K_data"] = {"K": K_sup, "sm_en_B": sm_en_B, "sr_en_B": sr_en_B}

        # Part 3: Hierarchical policies
        sp_mem_all    = aggregate_to_superregions(m_top, m_prob, r2s, K_sup)
        sp_router_all = aggregate_to_superregions(r_top, r_prob, r2s, K_sup)
        children = build_super_children(r2s, K_sup, args.n_fine)

        hier_rows_K = eval_hierarchical_policies(
            r_mg, m_mg, m_en, r_gr, m_gr, r_top, m_top,
            sp_mem_all, sp_router_all, r2s, children, K_sup, K_sup,
            groups, tpr, N, gold,
            coarse_margin_thr=args.coarse_margin_thr,
            coarse_entropy_thr=args.coarse_entropy_thr,
        )
        all_hier_rows.extend(hier_rows_K)

    _csv(super_results["conversion_rows"],
         os.path.join(args.output_dir, "typeB_coarse_conversion.csv"))
    _csv(super_results["subtype_rows"],
         os.path.join(args.output_dir, "typeB_subtype_breakdown.csv"))
    _csv(all_hier_rows,
         os.path.join(args.output_dir, "hierarchical_candidate_policies.csv"))
    print(f"[Part 2+3] done  ({time.time()-t0:.1f}s)")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not args.skip_plots:
        make_plots(
            args.output_dir, ctrl_rows, type_stats,
            A_typeB_mem, super_results, all_hier_rows, groups,
            r_mg, m_mg, r_en, m_en, type_a,
        )

    # ── Final report ───────────────────────────────────────────────────────────
    write_final_report(
        args.output_dir, ctrl_rows, super_results, all_hier_rows, learned_rows,
        groups, cfg, N,
    )

    print(f"\n[done] total time: {time.time()-t0:.1f}s")
    print(f"[done] outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
