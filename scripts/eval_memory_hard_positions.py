#!/usr/bin/env python3
"""
Hard-Position Memory Evaluation.

Loads per_position.npz and evaluates region coverage / NLL for groups where
memory is supposed to be most (or least) useful.

Hard groups:
  boundary          router_margin < 0.10
  tight_boundary    router_margin < 0.03
  high_entropy      router_entropy > 75th percentile
  router_top4_miss  gold not in router top-4
  router_top8_miss  gold not in router top-8
  type_A            same-region ambiguity (mem confident + correct)
  type_B            multi-region ambiguity (mem spread)
  type_C            misleading memory (mem confident + wrong)
  mem_confident     mem_margin > 0.20 and mem_entropy < 1.5

For each group reports:
  count, router NLL, mem NLL, mix NLL (beta=0.5),
  router top-k coverage (k=1,4,8,16),
  union top-k coverage ((r8,m4), (r16,m4)),
  controller coverage from best_controller.json if available.

Usage:
    python scripts/eval_memory_hard_positions.py \
        --knn_run_dir runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20 \
        --output_dir runs/region_memory_hard_positions_proxy010 \
        [--best_controller runs/region_memory_controller_proxy010/best_controller.json]
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pp(knn_run_dir: str) -> Dict[str, np.ndarray]:
    path = os.path.join(knn_run_dir, "per_position.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"per_position.npz not found in {knn_run_dir}. "
            "Re-run offline_region_knn.py to generate it."
        )
    return dict(np.load(path))


def compute_nll(topk_probs: np.ndarray, gold_rank: np.ndarray,
                clip: float = 1e-7) -> np.ndarray:
    K = topk_probs.shape[1]
    nll = np.full(len(gold_rank), -math.log(clip), dtype=np.float32)
    in_k = gold_rank < K
    idx  = np.where(in_k)[0]
    if len(idx):
        probs = topk_probs[idx, gold_rank[idx].astype(np.int32)].astype(np.float32)
        nll[idx] = -np.log(np.maximum(probs, clip))
    return nll


def mix_nll(r_topk_probs, r_gold_rank, m_topk_probs, m_gold_rank,
            beta: float = 0.5, clip: float = 1e-7) -> np.ndarray:
    """Approximate mix NLL using saved top-K probabilities."""
    K = r_topk_probs.shape[1]
    N = len(r_gold_rank)

    r_gold_prob = np.full(N, clip, dtype=np.float32)
    in_k = r_gold_rank < K
    if in_k.any():
        idx = np.where(in_k)[0]
        r_gold_prob[idx] = r_topk_probs[idx, r_gold_rank[idx].astype(np.int32)].astype(np.float32)

    m_gold_prob = np.full(N, clip, dtype=np.float32)
    in_k = m_gold_rank < K
    if in_k.any():
        idx = np.where(in_k)[0]
        m_gold_prob[idx] = m_topk_probs[idx, m_gold_rank[idx].astype(np.int32)].astype(np.float32)

    mixed = (1 - beta) * r_gold_prob + beta * m_gold_prob
    return -np.log(np.maximum(mixed, clip))


def union_coverage(r_gr: np.ndarray, m_gr: np.ndarray, Kr: int, Km: int) -> float:
    return float(((r_gr < Kr) | (m_gr < Km)).mean())


def group_stats(pp: Dict, mask: np.ndarray,
                r_gr: np.ndarray, m_gr: np.ndarray,
                has_router: bool) -> Dict:
    """Compute coverage and NLL metrics for a position subset."""
    n = int(mask.sum())
    if n == 0:
        return {"count": 0}

    r_gr_g  = r_gr[mask]
    m_gr_g  = m_gr[mask]
    r_tp    = pp["router_topk_probs"][mask]  if has_router else None
    m_tp    = pp["mem_topk_probs"][mask]

    stats: Dict = {"count": n}

    # NLL
    if has_router and r_tp is not None:
        stats["router_nll"] = round(float(compute_nll(r_tp, r_gr_g).mean()), 4)
    stats["mem_nll"] = round(float(compute_nll(m_tp, m_gr_g).mean()), 4)

    if has_router and r_tp is not None:
        stats["mix50_nll"] = round(float(
            mix_nll(r_tp, r_gr_g, m_tp, m_gr_g, beta=0.5).mean()
        ), 4)

    # Coverage
    for K in [1, 4, 8, 16]:
        if has_router:
            stats[f"router_top{K}_cov"] = round(float((r_gr_g < K).mean()), 4)
        stats[f"mem_top{K}_cov"] = round(float((m_gr_g < K).mean()), 4)

    if has_router:
        for Kr, Km in [(8, 4), (16, 4)]:
            stats[f"union_r{Kr}_m{Km}_cov"] = round(union_coverage(r_gr_g, m_gr_g, Kr, Km), 4)

    return stats


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
        description="Hard-Position Memory Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--knn_run_dir",     required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--best_controller", default=None,
                        help="best_controller.json from eval_region_memory_controller.py")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[hard] loading per_position.npz from {args.knn_run_dir}")
    pp = load_pp(args.knn_run_dir)
    N  = len(pp["gold_region"])
    print(f"[hard] {N:,} positions")

    has_router = "router_gold_rank" in pp
    r_gr  = pp["router_gold_rank"].astype(np.int32)    if has_router else np.full(N, 999)
    m_gr  = pp["mem_gold_rank"].astype(np.int32)
    r_mg  = pp["router_margin"].astype(np.float32)     if has_router else np.zeros(N)
    m_mg  = pp["mem_margin"].astype(np.float32)
    m_en  = pp["mem_entropy"].astype(np.float32)
    r_en  = pp["router_entropy"].astype(np.float32)    if has_router else np.zeros(N)
    type_ = pp["type"]    # 0=other, 1=A, 2=B, 3=C
    split = pp["split"]   # 0=core, 1=medium, 2=boundary, 3=tight_boundary

    # ── Define hard groups ────────────────────────────────────────────────────
    ent_p75 = float(np.percentile(r_en, 75)) if has_router else 0.0
    groups = {
        "all":               np.ones(N, dtype=bool),
        "core":              split == 0,
        "medium":            split == 1,
        "boundary":          split >= 2,
        "tight_boundary":    split == 3,
        "high_entropy":      r_en > ent_p75,
        "router_top4_miss":  r_gr >= 4,
        "router_top8_miss":  r_gr >= 8,
        "type_A":            type_ == 1,
        "type_B":            type_ == 2,
        "type_C":            type_ == 3,
        "mem_confident":     (m_mg > 0.20) & (m_en < 1.5),
        "mem_conf_correct":  (m_mg > 0.20) & (m_en < 1.5) & (m_gr < 1),
        "mem_conf_wrong":    (m_mg > 0.20) & (m_en < 1.5) & (m_gr >= 1),
    }

    rows: List[Dict] = []
    print("\n[hard] Group              count   r_nll   m_nll  mix50  r_top8  union_r8m4")
    print("-" * 75)
    for gname, mask in groups.items():
        s = group_stats(pp, mask, r_gr, m_gr, has_router)
        s["group"] = gname
        rows.append(s)
        cnt = s["count"]
        if cnt == 0:
            print(f"  {gname:<20}  {cnt:>6}  (empty)")
            continue
        r_nll = s.get("router_nll", float("nan"))
        m_nll = s.get("mem_nll", float("nan"))
        x_nll = s.get("mix50_nll", float("nan"))
        r_t8  = s.get("router_top8_cov", float("nan"))
        u_r8m4= s.get("union_r8_m4_cov", float("nan"))
        print(f"  {gname:<20}  {cnt:>6}  "
              f"{r_nll:6.3f}  {m_nll:6.3f}  {x_nll:6.3f}  "
              f"{r_t8:6.3f}  {u_r8m4:6.3f}")

    # ── Critical tests ────────────────────────────────────────────────────────
    print("\n[hard] Critical tests:")
    a_row = next((r for r in rows if r["group"] == "type_A"), {})
    b_row = next((r for r in rows if r["group"] == "type_B"), {})
    c_row = next((r for r in rows if r["group"] == "type_C"), {})

    if a_row.get("count", 0) > 0:
        a_improvement = a_row.get("router_nll", 0) - a_row.get("mix50_nll", 0)
        print(f"  Type A mix improvement over router: {a_improvement:+.3f}  "
              + ("PASS" if a_improvement > 0 else "FAIL"))

    if b_row.get("count", 0) > 0:
        b_hurt = b_row.get("mix50_nll", 0) - b_row.get("router_nll", 0)
        print(f"  Type B mix hurt vs router:          {b_hurt:+.3f}  "
              + ("GATED OK" if b_hurt < 0.05 else "HURTS"))

    if c_row.get("count", 0) > 0:
        c_hurt = c_row.get("mix50_nll", 0) - c_row.get("router_nll", 0)
        print(f"  Type C mix hurt vs router:          {c_hurt:+.3f}  "
              + ("GATED OK" if c_hurt < 0.05 else "DANGEROUS"))

    # router_top8_miss: can union recover?
    r8m = next((r for r in rows if r["group"] == "router_top8_miss"), {})
    if r8m.get("count", 0) > 0:
        r8_miss_recovery = r8m.get("union_r8_m4_cov", 0) - r8m.get("router_top8_cov", 0)
        print(f"  router_top8_miss: union_r8m4 recovery: {r8_miss_recovery:+.3f}")

    # ── Apply best controller if available ────────────────────────────────────
    if args.best_controller and os.path.exists(args.best_controller):
        print(f"\n[hard] Applying best_controller.json: {args.best_controller}")
        ctrl = json.load(open(args.best_controller))
        # Build controller coverage for each group
        mem_thr = float(ctrl.get("mem_margin_thr", 0.20))
        ent_thr = float(ctrl.get("entropy_thr",    1.50))
        Kr_bnd  = int(ctrl.get("Kr_bnd", 8))

        use_mem = (r_mg < 0.10) & (m_mg > mem_thr) & (m_en < ent_thr)
        c1 = (r_mg >= 0.30) & (r_gr < 4)
        c2 = (r_mg >= 0.10) & (r_mg < 0.30) & (r_gr < 8)
        c3 = (r_mg < 0.10) & use_mem & ((r_gr < Kr_bnd) | (m_gr < 4))
        c4 = (r_mg < 0.10) & ~use_mem & (r_gr < 16)
        ctrl_covered = c1 | c2 | c3 | c4

        print("\n  Controller coverage per group:")
        for gname, mask in groups.items():
            n_g = int(mask.sum())
            if n_g == 0:
                continue
            cov = float(ctrl_covered[mask].mean())
            # update row
            for r in rows:
                if r["group"] == gname:
                    r["controller_coverage"] = round(cov, 4)
                    break
            print(f"    {gname:<20}  {cov:.3f}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_csv = os.path.join(args.output_dir, "hard_positions_results.csv")
    _write_csv(rows, out_csv)
    print(f"\n[hard] wrote {out_csv}")

    # ── Summary markdown ──────────────────────────────────────────────────────
    md_lines = ["# Hard-Position Memory Evaluation", "",
                f"Run: `{args.knn_run_dir}`", "",
                "| Group | Count | Router NLL | Mem NLL | Mix50 NLL | Router Top8 | Union R8M4 |",
                "|-------|------:|-----------:|--------:|----------:|------------:|-----------:|"]
    for r in rows:
        if r["count"] == 0:
            continue
        md_lines.append(
            f"| {r['group']} "
            f"| {r['count']:,} "
            f"| {r.get('router_nll', float('nan')):.3f} "
            f"| {r.get('mem_nll', float('nan')):.3f} "
            f"| {r.get('mix50_nll', float('nan')):.3f} "
            f"| {r.get('router_top8_cov', float('nan')):.3f} "
            f"| {r.get('union_r8_m4_cov', float('nan')):.3f} |"
        )
    md_path = os.path.join(args.output_dir, "hard_positions_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if _HAS_PLT:
        plot_groups = ["boundary", "tight_boundary", "type_A", "type_B", "type_C",
                       "mem_confident", "router_top8_miss"]
        plot_rows = [r for r in rows if r["group"] in plot_groups and r["count"] > 0]
        if plot_rows:
            labels = [r["group"] for r in plot_rows]
            r_nlls = [r.get("router_nll", float("nan")) for r in plot_rows]
            m_nlls = [r.get("mem_nll", float("nan")) for r in plot_rows]
            x_nlls = [r.get("mix50_nll", float("nan")) for r in plot_rows]
            x = np.arange(len(labels))
            w = 0.25
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - w, r_nlls, w, label="router", color="steelblue")
            ax.bar(x,     m_nlls, w, label="mem",    color="orange")
            ax.bar(x + w, x_nlls, w, label="mix50",  color="green")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Region NLL")
            ax.set_title("Router vs Memory vs Mix — Hard Position Groups")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, "hard_groups_nll.png"), dpi=120)
            plt.close(fig)

    print(f"\n[hard] DONE → {args.output_dir}/")


if __name__ == "__main__":
    main()
