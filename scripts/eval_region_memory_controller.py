#!/usr/bin/env python3
"""
Offline Region-Memory Controller Evaluation.

Loads per_position.npz from a Region-kNN run and tests policies 0–4:

  Policy 0: router top-K baseline (K ∈ {4,8,12,16,24,32})
  Policy 1: fixed union(router_Kr, mem_Km) grid
  Policy 2: confidence-gated memory expansion (threshold grid)
  Policy 3: Type-A detector policy (threshold grid)
  Policy 4: learned lightweight controller (logistic regression)

Coverage uses gold_rank arrays for O(1) per-position region coverage;
token counts use the region map if provided.

Outputs:
  controller_results.csv          — all policy configs
  pareto_frontier.csv             — coverage vs avg_regions Pareto
  best_controller.json            — best policy config
  controller_plots/               — diagnostic plots

Usage:
    python scripts/eval_region_memory_controller.py \
        --knn_run_dir runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20 \
        --region_map_path runs/region_maps_128/token_to_region.json \
        --output_dir runs/region_memory_controller_proxy010
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── Data loading ──────────────────────────────────────────────────────────────

def load_per_position(knn_run_dir: str) -> Dict[str, np.ndarray]:
    path = os.path.join(knn_run_dir, "per_position.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"per_position.npz not found in {knn_run_dir}. "
            "Re-run offline_region_knn.py (it now saves per_position.npz automatically)."
        )
    d = np.load(path)
    return dict(d)


def load_tpr(region_map_path: str, n_coarse: int, vocab_size: int = 50257) -> np.ndarray:
    """tokens-per-region array, shape (n_coarse,)."""
    import json as _json
    with open(region_map_path) as f:
        rm = _json.load(f)
    tpr = np.zeros(n_coarse, dtype=np.float32)
    for tok_str, rid in rm.items():
        r = int(rid)
        if r < n_coarse:
            tpr[r] += 1
    return tpr


# ── Coverage helpers ──────────────────────────────────────────────────────────

def router_topk_coverage(router_gold_rank: np.ndarray, K: int) -> float:
    return float((router_gold_rank < K).mean())


def mem_topk_coverage(mem_gold_rank: np.ndarray, K: int) -> float:
    return float((mem_gold_rank < K).mean())


def union_coverage(router_gold_rank: np.ndarray, mem_gold_rank: np.ndarray,
                   Kr: int, Km: int) -> float:
    return float(((router_gold_rank < Kr) | (mem_gold_rank < Km)).mean())


def union_avg_tokens(router_topk_reg: np.ndarray, mem_topk_reg: np.ndarray,
                     tpr: np.ndarray, Kr: int, Km: int) -> float:
    """Vectorised average token count for union(router_top_Kr, mem_top_Km)."""
    r_regs = router_topk_reg[:, :Kr]   # (N, Kr)
    m_regs = mem_topk_reg[:, :Km]      # (N, Km)
    # new mem regions = mem regions not already in router set
    new_mask = ~np.any(
        m_regs[:, :, np.newaxis] == r_regs[:, np.newaxis, :], axis=-1
    )  # (N, Km) bool
    router_tok = tpr[r_regs].sum(axis=-1)
    new_mem_tok = (tpr[m_regs] * new_mask).sum(axis=-1)
    return float((router_tok + new_mem_tok).mean())


def union_avg_regions(router_topk_reg: np.ndarray, mem_topk_reg: np.ndarray,
                      Kr: int, Km: int) -> float:
    r_regs = router_topk_reg[:, :Kr]
    m_regs = mem_topk_reg[:, :Km]
    new_mask = ~np.any(
        m_regs[:, :, np.newaxis] == r_regs[:, np.newaxis, :], axis=-1
    )
    return float(Kr + new_mask.sum(axis=-1).mean())


def policy2_coverage(
    router_gold_rank: np.ndarray,
    mem_gold_rank: np.ndarray,
    router_margin: np.ndarray,
    mem_margin: np.ndarray,
    mem_entropy: np.ndarray,
    mem_thr: float,
    ent_thr: float,
    Kr_bnd: int = 8,
    Km_bnd: int = 4,
    Kr_fallback: int = 16,
) -> float:
    """Confidence-gated policy: router 4/8/[union8+4|16]."""
    use_mem = (router_margin < 0.10) & (mem_margin > mem_thr) & (mem_entropy < ent_thr)
    c1 = (router_margin >= 0.30) & (router_gold_rank < 4)
    c2 = (router_margin >= 0.10) & (router_margin < 0.30) & (router_gold_rank < 8)
    c3 = (router_margin < 0.10) & use_mem & (
        (router_gold_rank < Kr_bnd) | (mem_gold_rank < Km_bnd)
    )
    c4 = (router_margin < 0.10) & ~use_mem & (router_gold_rank < Kr_fallback)
    return float((c1 | c2 | c3 | c4).mean())


def policy2_avg_regions(
    router_topk_reg: np.ndarray,
    mem_topk_reg: np.ndarray,
    router_margin: np.ndarray,
    mem_margin: np.ndarray,
    mem_entropy: np.ndarray,
    mem_thr: float, ent_thr: float,
    Kr_bnd: int = 8, Km_bnd: int = 4, Kr_fallback: int = 16,
) -> float:
    use_mem = (router_margin < 0.10) & (mem_margin > mem_thr) & (mem_entropy < ent_thr)
    case_sizes = np.where(
        router_margin >= 0.30, 4,
        np.where(
            router_margin >= 0.10, 8,
            np.where(use_mem, Kr_bnd, Kr_fallback),
        ),
    ).astype(np.float32)
    # Add new mem regions for union case
    bnd_union = (router_margin < 0.10) & use_mem
    if bnd_union.any():
        r_regs = router_topk_reg[bnd_union, :Kr_bnd]
        m_regs = mem_topk_reg[bnd_union, :Km_bnd]
        new_mask = ~np.any(
            m_regs[:, :, np.newaxis] == r_regs[:, np.newaxis, :], axis=-1
        )
        case_sizes[bnd_union] += new_mask.sum(axis=-1).astype(np.float32)
    return float(case_sizes.mean())


# ── NLL helpers ───────────────────────────────────────────────────────────────

def compute_nll_from_ranks(topk_probs: np.ndarray, gold_rank: np.ndarray,
                            clip_min: float = 1e-7) -> np.ndarray:
    """
    Returns NLL per position using saved top-K distributions.
    Positions where gold is outside top-K get -log(clip_min) (pessimistic).
    """
    K = topk_probs.shape[1]
    nll = np.full(len(gold_rank), -np.log(clip_min), dtype=np.float32)
    in_topk = gold_rank < K
    idx = np.where(in_topk)[0]
    if len(idx):
        probs = topk_probs[idx, gold_rank[idx]]
        nll[idx] = -np.log(np.maximum(probs.astype(np.float32), clip_min))
    return nll


# ── Policy 4: lightweight logistic controller ─────────────────────────────────

def build_features(pp: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Build gold-free feature matrix for controller training."""
    required = ("router_margin", "router_entropy", "mem_margin", "mem_entropy",
                "router_topk_probs", "mem_topk_probs",
                "router_topk_regions", "mem_topk_regions")
    if any(k not in pp for k in required):
        return None

    r_mg = pp["router_margin"].astype(np.float32)
    r_en = pp["router_entropy"].astype(np.float32)
    m_mg = pp["mem_margin"].astype(np.float32)
    m_en = pp["mem_entropy"].astype(np.float32)
    r_top1_prob = pp["router_topk_probs"][:, 0].astype(np.float32)
    m_top1_prob = pp["mem_topk_probs"][:, 0].astype(np.float32)
    agreement = (pp["router_topk_regions"][:, 0] == pp["mem_topk_regions"][:, 0]
                 ).astype(np.float32)
    # fraction of top-8 router regions appearing in top-8 mem regions
    r8 = pp["router_topk_regions"][:, :8]
    m8 = pp["mem_topk_regions"][:, :8]
    overlap8 = np.any(
        r8[:, :, np.newaxis] == m8[:, np.newaxis, :], axis=-1
    ).sum(axis=-1).astype(np.float32) / 8.0

    return np.column_stack([
        r_mg, r_en, m_mg, m_en,
        r_top1_prob, m_top1_prob,
        agreement, overlap8,
    ])


def policy4_labels(pp: Dict[str, np.ndarray], Kr: int = 8, Km: int = 4) -> np.ndarray:
    """
    Label = 1 if memory adds gold when router misses:
        router_gold_rank >= Kr  AND  mem_gold_rank < Km
    """
    return (
        (pp["router_gold_rank"] >= Kr) & (pp["mem_gold_rank"] < Km)
    ).astype(np.int32)


def train_controller(X_train: np.ndarray, y_train: np.ndarray
                     ) -> Tuple[object, object]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf.fit(X_scaled, y_train)
    return clf, scaler


def controller_policy_coverage(
    clf, scaler,
    X: np.ndarray,
    pp: Dict[str, np.ndarray],
    prob_thresh: float = 0.5,
    Kr_core: int = 4,
    Kr_mid: int = 8,
    Kr_bnd: int = 8, Km_bnd: int = 4,
    Kr_fallback: int = 16,
) -> Tuple[float, float]:
    """Apply trained controller to select candidate regions. Returns (coverage, avg_regions)."""
    r_mg = pp["router_margin"].astype(np.float32)
    r_gr = pp["router_gold_rank"]
    m_gr = pp["mem_gold_rank"]
    r_topk = pp["router_topk_regions"]
    m_topk = pp["mem_topk_regions"]

    X_scaled = scaler.transform(X)
    mem_useful_prob = clf.predict_proba(X_scaled)[:, 1]
    use_mem = (r_mg < 0.10) & (mem_useful_prob >= prob_thresh)

    c1 = (r_mg >= 0.30) & (r_gr < Kr_core)
    c2 = (r_mg >= 0.10) & (r_mg < 0.30) & (r_gr < Kr_mid)
    c3 = (r_mg < 0.10) & use_mem & ((r_gr < Kr_bnd) | (m_gr < Km_bnd))
    c4 = (r_mg < 0.10) & ~use_mem & (r_gr < Kr_fallback)
    coverage = float((c1 | c2 | c3 | c4).mean())

    # avg regions
    case_sz = np.where(
        r_mg >= 0.30, float(Kr_core),
        np.where(r_mg >= 0.10, float(Kr_mid),
                 np.where(use_mem, float(Kr_bnd), float(Kr_fallback)))
    ).astype(np.float32)
    bnd_u = (r_mg < 0.10) & use_mem
    if bnd_u.any():
        r8 = r_topk[bnd_u, :Kr_bnd]
        m4 = m_topk[bnd_u, :Km_bnd]
        new_m = ~np.any(m4[:, :, np.newaxis] == r8[:, np.newaxis, :], axis=-1)
        case_sz[bnd_u] += new_m.sum(axis=-1)
    return coverage, float(case_sz.mean())


# ── Pareto filtering ──────────────────────────────────────────────────────────

def pareto_frontier(rows: List[Dict], x_key: str, y_key: str,
                    maximize_y: bool = True) -> List[Dict]:
    """Return Pareto-optimal rows (minimize x, maximize/minimize y)."""
    valid = [r for r in rows if not math.isnan(float(r.get(x_key, "nan") or "nan"))
             and not math.isnan(float(r.get(y_key, "nan") or "nan"))]
    valid.sort(key=lambda r: float(r[x_key]))
    frontier = []
    best_y = -math.inf if maximize_y else math.inf
    for r in valid:
        y = float(r[y_key])
        if (maximize_y and y > best_y) or (not maximize_y and y < best_y):
            frontier.append(r)
            best_y = y
    return frontier


# ── Output helpers ────────────────────────────────────────────────────────────

def _write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    keys: Dict[str, None] = {}
    for r in rows:
        for k in r:
            keys[k] = None
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keys.keys()), restval="")
        w.writeheader()
        w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Region-Memory Controller Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--knn_run_dir", required=True,
                        help="Region-kNN run dir containing per_position.npz")
    parser.add_argument("--region_map_path", default=None,
                        help="token_to_region.json (for token counts)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_frac", type=float, default=0.7,
                        help="Fraction of positions used to train Policy 4 controller")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # ── Load per-position data ────────────────────────────────────────────────
    print(f"[ctrl] loading per_position.npz from {args.knn_run_dir}")
    pp = load_per_position(args.knn_run_dir)
    N = len(pp["gold_region"])
    print(f"[ctrl] {N:,} positions loaded")

    # Load knn config for metadata
    cfg_path = os.path.join(args.knn_run_dir, "knn_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    n_coarse = cfg.get("n_coarse", int(pp["mem_topk_regions"].max()) + 1)
    print(f"[ctrl] n_coarse={n_coarse}  K_saved={pp['mem_topk_regions'].shape[1]}")

    # Region map → tokens per region
    tpr = None
    total_mapped = 0
    region_map_path = args.region_map_path or cfg.get("region_map_path")
    if region_map_path and os.path.exists(region_map_path):
        tpr = load_tpr(region_map_path, n_coarse)
        total_mapped = int(tpr.sum())
        print(f"[ctrl] loaded tpr  total_mapped={total_mapped:,}")

    has_router = "router_gold_rank" in pp
    r_gr  = pp["router_gold_rank"].astype(np.int32) if has_router else None
    m_gr  = pp["mem_gold_rank"].astype(np.int32)
    r_mg  = pp["router_margin"].astype(np.float32) if has_router else None
    m_mg  = pp["mem_margin"].astype(np.float32)
    m_en  = pp["mem_entropy"].astype(np.float32)
    r_en  = pp["router_entropy"].astype(np.float32) if has_router else None
    r_topk_reg  = pp["router_topk_regions"]  if has_router else None
    m_topk_reg  = pp["mem_topk_regions"]
    r_topk_prob = pp["router_topk_probs"]    if has_router else None
    m_topk_prob = pp["mem_topk_probs"]

    if not has_router:
        print("[ctrl] WARNING: no router data in per_position.npz — policies needing "
              "router will be skipped")

    results: List[Dict] = []

    def _row(name, cov, avg_reg, avg_tok=None, extra=None):
        r = {"policy": name, "coverage": round(cov, 4),
             "avg_regions": round(avg_reg, 2)}
        if avg_tok is not None and total_mapped > 0:
            r["avg_tokens"]         = round(float(avg_tok), 0)
            r["mapped_vocab_pct"]   = round(float(avg_tok) / total_mapped * 100, 2)
        if extra:
            r.update(extra)
        return r

    def _topk_tok(topk_reg, K):
        if tpr is None:
            return None
        return float(tpr[topk_reg[:, :K]].sum(axis=-1).mean())

    def _union_tok(r_reg, m_reg, Kr, Km):
        if tpr is None:
            return None
        return union_avg_tokens(r_reg, m_reg, tpr, Kr, Km)

    # ── Policy 0: router top-K ────────────────────────────────────────────────
    print("[ctrl] Policy 0: router top-K baseline")
    if has_router:
        for K in [4, 8, 12, 16, 24, 32]:
            cov  = router_topk_coverage(r_gr, K)
            regs = float(K)
            tok  = _topk_tok(r_topk_reg, K)
            results.append(_row(f"policy0_router_top{K}", cov, regs, tok,
                                {"policy_class": "P0"}))
    else:
        print("[ctrl] skipping Policy 0 (no router data)")

    # ── Policy 1: fixed union ─────────────────────────────────────────────────
    print("[ctrl] Policy 1: fixed union")
    if has_router:
        for Kr, Km in [(4,2),(4,4),(8,2),(8,4),(8,8),(12,4),(12,8),(16,4),(16,8),(16,16)]:
            cov  = union_coverage(r_gr, m_gr, Kr, Km)
            regs = union_avg_regions(r_topk_reg, m_topk_reg, Kr, Km)
            tok  = _union_tok(r_topk_reg, m_topk_reg, Kr, Km)
            results.append(_row(f"policy1_union_r{Kr}_m{Km}", cov, regs, tok,
                                {"policy_class": "P1", "Kr": Kr, "Km": Km}))

    # ── Policy 2: confidence-gated grid ──────────────────────────────────────
    print("[ctrl] Policy 2: confidence-gated grid search")
    if has_router:
        thr_r_list  = [0.05, 0.10, 0.15, 0.20]
        thr_m_list  = [0.10, 0.15, 0.20, 0.25, 0.30]
        thr_en_list = [1.0, 1.25, 1.5, 1.75, 2.0]
        for thr_r in thr_r_list:
            for thr_m in thr_m_list:
                for thr_en in thr_en_list:
                    cov  = policy2_coverage(r_gr, m_gr, r_mg, m_mg, m_en,
                                            thr_m, thr_en)
                    regs = policy2_avg_regions(r_topk_reg, m_topk_reg, r_mg, m_mg, m_en,
                                               thr_m, thr_en)
                    tok  = None  # skip detailed token count for grid
                    results.append(_row(
                        f"policy2_tm{thr_m:.2f}_te{thr_en:.2f}", cov, regs, tok,
                        {"policy_class": "P2",
                         "mem_margin_thr": thr_m, "entropy_thr": thr_en},
                    ))
        print(f"  {len(thr_r_list)*len(thr_m_list)*len(thr_en_list)} P2 configs evaluated")

    # ── Policy 3: Type-A detector ─────────────────────────────────────────────
    print("[ctrl] Policy 3: Type-A detector policy")
    if has_router:
        for thr_m in [0.15, 0.20, 0.25, 0.30]:
            for thr_en in [1.25, 1.50, 1.75, 2.0]:
                for Kr_bnd in [8, 12]:
                    is_typeA = (r_mg < 0.10) & (m_mg > thr_m) & (m_en < thr_en)
                    c1 = (r_mg >= 0.30) & (r_gr < 4)
                    c2 = (r_mg >= 0.10) & (r_mg < 0.30) & (r_gr < 8)
                    c3 = (r_mg < 0.10) & is_typeA & ((r_gr < Kr_bnd) | (m_gr < 4))
                    c4 = (r_mg < 0.10) & ~is_typeA & (r_gr < 16)
                    cov  = float((c1 | c2 | c3 | c4).mean())
                    # avg regions
                    sz = np.where(r_mg >= 0.30, 4,
                                  np.where(r_mg >= 0.10, 8,
                                           np.where(is_typeA, float(Kr_bnd), 16.0))
                                  ).astype(np.float32)
                    bnd_u = (r_mg < 0.10) & is_typeA
                    if bnd_u.any():
                        rr = r_topk_reg[bnd_u, :Kr_bnd]
                        mm = m_topk_reg[bnd_u, :4]
                        new = ~np.any(mm[:, :, np.newaxis] == rr[:, np.newaxis, :], axis=-1)
                        sz[bnd_u] += new.sum(axis=-1)
                    regs = float(sz.mean())
                    results.append(_row(
                        f"policy3_m{thr_m:.2f}_e{thr_en:.2f}_k{Kr_bnd}",
                        cov, regs, None,
                        {"policy_class": "P3", "mem_margin_thr": thr_m,
                         "entropy_thr": thr_en, "Kr_bnd": Kr_bnd},
                    ))

    # ── Policy 4: learned controller ─────────────────────────────────────────
    print("[ctrl] Policy 4: learned logistic controller")
    clf4, scaler4, X_all = None, None, None
    if has_router and _HAS_SKLEARN:
        X_all = build_features(pp)
        if X_all is not None:
            y_all  = policy4_labels(pp)
            n_train = int(N * args.train_frac)
            idx_perm = np.random.permutation(N)
            tr_idx, val_idx = idx_perm[:n_train], idx_perm[n_train:]
            X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
            X_val       = X_all[val_idx]
            pp_val      = {k: v[val_idx] for k, v in pp.items()}

            print(f"  train={len(tr_idx):,}  val={len(val_idx):,}  "
                  f"pos_rate={y_tr.mean():.3f}")
            clf4, scaler4 = train_controller(X_tr, y_tr)
            print(f"  controller coef_norm={np.linalg.norm(clf4.coef_):.3f}")

            for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                cov, regs = controller_policy_coverage(
                    clf4, scaler4, X_val, pp_val, prob_thresh=thresh,
                )
                results.append(_row(
                    f"policy4_thresh{thresh:.1f}", cov, regs, None,
                    {"policy_class": "P4", "prob_thresh": thresh,
                     "train_frac": args.train_frac},
                ))
            # Best threshold by coverage on validation
            p4_rows  = [r for r in results if r.get("policy_class") == "P4"]
            best_p4  = max(p4_rows, key=lambda r: r["coverage"]) if p4_rows else {}
            print(f"  best P4: coverage={best_p4.get('coverage','?')}  "
                  f"avg_regions={best_p4.get('avg_regions','?')}")
        else:
            print("  skipping — per_position.npz missing router features")
    elif not _HAS_SKLEARN:
        print("  skipping — scikit-learn not installed")

    # ── Pareto & best config ──────────────────────────────────────────────────
    pareto = pareto_frontier(results, "avg_regions", "coverage", maximize_y=True)
    print(f"\n[ctrl] Pareto frontier ({len(pareto)} points, increasing coverage):")
    for r in pareto:
        print(f"  {r['policy']:<45}  cov={r['coverage']:.3f}  "
              f"reg={r['avg_regions']:.1f}  "
              f"mapped%={r.get('mapped_vocab_pct','?')}")

    # ── Success criteria check ────────────────────────────────────────────────
    print("\n[ctrl] Success criteria:")
    for label, cov_thr, reg_thr in [("strong", 0.95, 15.0), ("aggressive", 0.90, 8.0)]:
        cand = [r for r in results
                if r["coverage"] >= cov_thr
                and r.get("mapped_vocab_pct", 100) <= reg_thr]
        if not cand and tpr is None:
            # Fall back to avg_regions as proxy
            cand = [r for r in results
                    if r["coverage"] >= cov_thr and r["avg_regions"] <= reg_thr]
        if cand:
            best = min(cand, key=lambda r: r["avg_regions"])
            print(f"  {label}: PASS  best={best['policy']}  "
                  f"cov={best['coverage']:.3f}  reg={best['avg_regions']:.1f}")
        else:
            # find best coverage
            best_cov = max(results, key=lambda r: r["coverage"]) if results else {}
            print(f"  {label}: FAIL  best_cov={best_cov.get('coverage','?'):.3f}")

    # Best policy by Pareto (closest to strong threshold)
    strong_cand = sorted(
        [r for r in pareto if r["coverage"] >= 0.90],
        key=lambda r: r["avg_regions"],
    )
    best_policy = strong_cand[0] if strong_cand else (
        max(results, key=lambda r: r["coverage"]) if results else {}
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_csv(results, os.path.join(args.output_dir, "controller_results.csv"))
    _write_csv(pareto,  os.path.join(args.output_dir, "pareto_frontier.csv"))

    # best_controller.json — used by masked softmax script
    best_cfg: Dict = {
        "policy_name":    best_policy.get("policy", "policy0_router_top8"),
        "policy_class":   best_policy.get("policy_class", "P0"),
        "coverage":       best_policy.get("coverage", 0.0),
        "avg_regions":    best_policy.get("avg_regions", 8.0),
        "mapped_vocab_pct": best_policy.get("mapped_vocab_pct", None),
        # Decode policy parameters
        "mem_margin_thr": best_policy.get("mem_margin_thr", 0.20),
        "entropy_thr":    best_policy.get("entropy_thr",    1.50),
        "Kr_bnd":         best_policy.get("Kr_bnd",          8),
        "Km_bnd":         best_policy.get("Km",              4),
        "Kr_fallback":    best_policy.get("Kr_fallback",    16),
        "prob_thresh":    best_policy.get("prob_thresh",    0.50),
        "knn_run_dir":    args.knn_run_dir,
    }
    with open(os.path.join(args.output_dir, "best_controller.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"\n[ctrl] best_controller.json: {best_cfg['policy_name']}  "
          f"coverage={best_cfg['coverage']:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if _HAS_PLT:
        plot_dir = os.path.join(args.output_dir, "controller_plots")
        os.makedirs(plot_dir, exist_ok=True)
        colors = {"P0": "steelblue", "P1": "orange", "P2": "green",
                  "P3": "red", "P4": "purple"}

        fig, ax = plt.subplots(figsize=(9, 6))
        for pc, col in colors.items():
            pts = [r for r in results if r.get("policy_class") == pc]
            if pts:
                ax.scatter([r["avg_regions"] for r in pts],
                           [r["coverage"] for r in pts],
                           s=20, alpha=0.5, color=col, label=pc)
        if pareto:
            ax.plot([r["avg_regions"] for r in pareto],
                    [r["coverage"] for r in pareto],
                    "k--", linewidth=1.5, label="Pareto")
        ax.axhline(0.90, color="grey", linestyle=":", linewidth=0.8)
        ax.axhline(0.95, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Avg regions kept")
        ax.set_ylabel("Gold region coverage")
        ax.set_title("Coverage vs cost — all policies")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "01_pareto_coverage_vs_regions.png"), dpi=120)
        plt.close(fig)
        print(f"[ctrl] plot saved → {plot_dir}/")

    print(f"\n[ctrl] DONE → {args.output_dir}/")


if __name__ == "__main__":
    main()
