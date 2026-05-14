#!/usr/bin/env python3
"""
Aggregate results from the extensive Region-kNN sweep.

Reads all per-run CSVs from runs/region_knn_extensive_sweep/ and produces:
  aggregate_results.csv         — one row per run, key metrics
  aggregate_by_split.csv        — one row per run × split
  aggregate_candidate_policies.csv — one row per run × policy
  aggregate_type_analysis.csv   — one row per run × type A/B/C
  final_report.md               — human-readable Q1-Q12 answers + recommendation
  plots/01-10.png               — diagnostic plots

Usage:
    python scripts/aggregate_region_knn_sweep.py \
        --sweep_dir  runs/region_knn_extensive_sweep \
        --output_dir runs/region_knn_extensive_sweep
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _flt(d: Dict, key: str, default: float = float("nan")) -> float:
    v = d.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    seen: Dict[str, None] = {}
    for row in rows:
        for k in row:
            seen[k] = None
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seen.keys()), restval="")
        w.writeheader()
        w.writerows(rows)


# ── Run metadata parsing ──────────────────────────────────────────────────────

def _parse_run_dir(d: str) -> Dict[str, Any]:
    """Extract sweep axes from directory name."""
    name = os.path.basename(d)
    meta: Dict[str, Any] = {"run_dir": d, "run_name": name}

    # checkpoint label
    if name.startswith("reference"):
        meta["checkpoint"] = "reference"
        meta["lambda"] = 0.0
    elif "proxy005" in name:
        meta["checkpoint"] = "proxy0.05"
        meta["lambda"] = 0.05
    elif "proxy010" in name:
        meta["checkpoint"] = "proxy0.10"
        meta["lambda"] = 0.10
    else:
        meta["checkpoint"] = "unknown"
        meta["lambda"] = float("nan")

    # key source
    if "retrproj" in name:
        meta["key_source"] = "retrieval_proj"
    elif "raw_h" in name:
        meta["key_source"] = "raw_h"
    else:
        meta["key_source"] = "unknown"

    # memory size
    m = re.search(r"mem(\d+[kKmM])", name)
    if m:
        tag = m.group(1).lower()
        mult = 1_000_000 if "m" in tag else 1_000
        meta["mem_positions"] = int(re.sub(r"[km]", "", tag)) * mult
    else:
        meta["mem_positions"] = -1

    # knn_k
    m = re.search(r"_k(\d+)_", name)
    meta["knn_k"] = int(m.group(1)) if m else -1

    # knn_temp
    m = re.search(r"_t([\dp]+)$", name)
    if m:
        meta["knn_temp"] = float(m.group(1).replace("p", "."))
    else:
        meta["knn_temp"] = float("nan")

    return meta


def _load_run(d: str) -> Optional[Dict]:
    """Load all CSV/JSON from a single run directory."""
    if not os.path.isdir(d):
        return None
    metrics_all  = _read_csv(os.path.join(d, "metrics_all.csv"))
    metrics_spl  = _read_csv(os.path.join(d, "metrics_by_split.csv"))
    cov_rows     = _read_csv(os.path.join(d, "candidate_coverage.csv"))
    type_rows    = _read_csv(os.path.join(d, "type_analysis.csv"))
    cfg_path     = os.path.join(d, "knn_config.json")
    cfg          = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cal_path     = os.path.join(d, "mem_conf_calibration.json")
    cal          = json.load(open(cal_path)) if os.path.exists(cal_path) else {}

    if not metrics_all:
        return None

    # Index metrics_all by method
    m_all = {r["method"]: r for r in metrics_all}

    return {
        "m_all":      m_all,
        "m_spl":      metrics_spl,
        "cov_rows":   cov_rows,
        "type_rows":  type_rows,
        "cfg":        cfg,
        "cal":        cal,
    }


# ── Aggregate builders ────────────────────────────────────────────────────────

def build_aggregate_results(run_metas: List[Dict], run_data: List[Dict]
                             ) -> List[Dict]:
    rows = []
    for meta, data in zip(run_metas, run_data):
        if data is None:
            continue
        m_all = data["m_all"]
        row = dict(meta)

        # Key per-method metrics
        for method in ("router", "mem_weighted", "mem_unweighted",
                       "mix_0.10", "mix_0.25", "mix_0.50", "mix_0.75",
                       "mix_margin", "mix_mem_conf"):
            m = m_all.get(method, {})
            pfx = method.replace(".", "p").replace("_", "")
            for k in ("acc1", "acc4", "acc8", "acc16", "region_nll",
                      "gold_rank_mean", "margin_mean", "entropy_mean"):
                row[f"{pfx}_{k}"] = _flt(m, k)

        # Calibration
        cal = data["cal"]
        row["mem_conf_top1_acc"] = cal.get("mem_top1_acc_when_high_conf", float("nan"))
        row["mem_conf_n_high"]   = cal.get("n_high_conf", 0)

        rows.append(row)
    return rows


def build_aggregate_by_split(run_metas: List[Dict], run_data: List[Dict]
                               ) -> List[Dict]:
    rows = []
    for meta, data in zip(run_metas, run_data):
        if data is None:
            continue
        for srow in data["m_spl"]:
            row = dict(meta)
            row.update({k: v for k, v in srow.items()})
            rows.append(row)
    return rows


def build_aggregate_policies(run_metas: List[Dict], run_data: List[Dict]
                               ) -> List[Dict]:
    rows = []
    for meta, data in zip(run_metas, run_data):
        if data is None:
            continue
        for crow in data["cov_rows"]:
            row = dict(meta)
            row.update({k: v for k, v in crow.items()})
            rows.append(row)
    return rows


def build_aggregate_types(run_metas: List[Dict], run_data: List[Dict]
                           ) -> List[Dict]:
    rows = []
    for meta, data in zip(run_metas, run_data):
        if data is None:
            continue
        for tr in data["type_rows"]:
            row = dict(meta)
            row.update({k: v for k, v in tr.items()})
            rows.append(row)
    return rows


# ── Final report ─────────────────────────────────────────────────────────────

def _best_run(agg: List[Dict], key: str, among=None, minimize=True) -> Optional[Dict]:
    rows = [r for r in agg if among is None or r.get("checkpoint") in among]
    rows = [r for r in rows if not math.isnan(float(r.get(key, "nan") or "nan"))]
    if not rows:
        return None
    return min(rows, key=lambda r: float(r.get(key, "nan"))) if minimize \
        else max(rows, key=lambda r: float(r.get(key, "nan")))


def write_final_report(
    agg_res: List[Dict],
    agg_spl: List[Dict],
    agg_pol: List[Dict],
    agg_typ: List[Dict],
    out_path: str,
) -> None:
    lines = [
        "# Region-kNN Extensive Sweep — Final Report",
        "",
        f"Generated from {len(agg_res)} completed runs.",
        "",
    ]

    # ── Q1: Does memory scale improve? ───────────────────────────────────────
    lines += ["## Q1: Does Region-kNN improve with memory scale?", ""]
    # Show mem_nll vs mem for proxy010 retrieval_proj
    p10_rp = sorted(
        [r for r in agg_res
         if r.get("checkpoint") == "proxy0.10" and r.get("key_source") == "retrieval_proj"
         and r.get("knn_k") == 64],
        key=lambda r: r.get("mem_positions", 0),
    )
    if p10_rp:
        lines.append("| Memory | mem NLL | router NLL | mem acc@1 |")
        lines.append("|-------:|--------:|-----------:|----------:|")
        for r in p10_rp:
            lines.append(
                f"| {r.get('mem_positions',0):,} "
                f"| {_flt(r,'memweighted_region_nll'):.3f} "
                f"| {_flt(r,'router_region_nll'):.3f} "
                f"| {_flt(r,'memweighted_acc1'):.3f} |"
            )
        nlls = [_flt(r, "memweighted_region_nll") for r in p10_rp
                if not math.isnan(_flt(r, "memweighted_region_nll"))]
        if len(nlls) >= 2:
            verdict = "**SCALES**" if nlls[-1] < nlls[0] - 0.01 else "**PLATEAUS**"
            lines.append(f"\n{verdict}: NLL went from {nlls[0]:.3f} → {nlls[-1]:.3f}")

    # ── Q2: proxy 0.05 vs 0.10 ───────────────────────────────────────────────
    lines += ["", "## Q2: Is proxy λ=0.05 or λ=0.10 better?", ""]
    for ckpt in ("proxy0.05", "proxy0.10"):
        best = _best_run(
            [r for r in agg_res if r.get("checkpoint") == ckpt
             and r.get("key_source") == "retrieval_proj"],
            "memweighted_region_nll", minimize=True,
        )
        if best:
            lines.append(
                f"- **{ckpt}** best mem NLL={_flt(best,'memweighted_region_nll'):.3f} "
                f"acc@1={_flt(best,'memweighted_acc1'):.3f} "
                f"(mem={best.get('mem_positions',0):,} k={best.get('knn_k')} "
                f"temp={best.get('knn_temp')})"
            )

    # ── Q3: Best knn_k and knn_temp ──────────────────────────────────────────
    lines += ["", "## Q3: What knn_k / knn_temp works best?", ""]
    best_kt = _best_run(
        [r for r in agg_res if r.get("key_source") == "retrieval_proj"],
        "memweighted_region_nll", minimize=True,
    )
    if best_kt:
        lines.append(
            f"Best: k={best_kt.get('knn_k')} temp={best_kt.get('knn_temp')} "
            f"checkpoint={best_kt.get('checkpoint')} "
            f"mem NLL={_flt(best_kt,'memweighted_region_nll'):.3f}"
        )
    # k sweep table (fixed temp=0.20)
    k_sweep = sorted(
        [r for r in agg_res
         if r.get("knn_temp") == 0.20
         and r.get("checkpoint") == "proxy0.10"
         and r.get("key_source") == "retrieval_proj"],
        key=lambda r: r.get("knn_k", 0),
    )
    if k_sweep:
        lines += ["", "| k | mem NLL | mem acc@1 |", "|--:|--------:|----------:|"]
        for r in k_sweep:
            lines.append(
                f"| {r.get('knn_k')} "
                f"| {_flt(r,'memweighted_region_nll'):.3f} "
                f"| {_flt(r,'memweighted_acc1'):.3f} |"
            )

    # ── Q4: Boundary token help ───────────────────────────────────────────────
    lines += ["", "## Q4: Does Region-kNN help boundary tokens?", ""]
    best_overall = _best_run(
        [r for r in agg_res if r.get("key_source") == "retrieval_proj"],
        "memweighted_region_nll",
    )
    if best_overall:
        bname = best_overall.get("run_name", "")
        bnd_rows = [r for r in agg_spl
                    if r.get("run_name") == bname and r.get("split") == "boundary"]
        if bnd_rows:
            lines.append("| Method | Boundary acc@1 | Boundary NLL |")
            lines.append("|--------|---------------:|-------------:|")
            for meth in ("router", "mem_weighted", "mix_0.50", "mix_margin"):
                mr = next((r for r in bnd_rows if r.get("method") == meth), None)
                if mr:
                    lines.append(
                        f"| {meth} "
                        f"| {_flt(mr,'acc1'):.3f} "
                        f"| {_flt(mr,'region_nll'):.3f} |"
                    )

    # ── Q5: Type A vs Type B ──────────────────────────────────────────────────
    lines += ["", "## Q5: Does Region-kNN help Type A more than Type B?", ""]
    if agg_typ and best_overall:
        bname = best_overall.get("run_name", "")
        typ_rows = [r for r in agg_typ if r.get("run_name") == bname]
        if typ_rows:
            lines.append("| Type | Count | % boundary | Router NLL | Mem NLL | Mix NLL | Improvement |")
            lines.append("|------|------:|-----------:|-----------:|--------:|--------:|------------:|")
            for tp in ("A", "B", "C"):
                tr = next((r for r in typ_rows if r.get("type") == tp), None)
                if tr:
                    lines.append(
                        f"| {tp} "
                        f"| {tr.get('count','?')} "
                        f"| {_flt(tr,'pct_of_boundary'):.1%} "
                        f"| {_flt(tr,'router_nll'):.3f} "
                        f"| {_flt(tr,'mem_nll'):.3f} "
                        f"| {_flt(tr,'mix50_nll'):.3f} "
                        f"| {_flt(tr,'improvement_vs_router'):+.3f} |"
                    )
        imp_A = next((_flt(r, "improvement_vs_router") for r in typ_rows if r.get("type") == "A"), 0.0)
        imp_B = next((_flt(r, "improvement_vs_router") for r in typ_rows if r.get("type") == "B"), 0.0)
        verdict = "**PASS**" if imp_A > imp_B + 0.02 else "**INCONCLUSIVE**"
        lines.append(f"\n{verdict}: Type A improvement={imp_A:.3f}  Type B improvement={imp_B:.3f}")

    # ── Q6: Candidate coverage ────────────────────────────────────────────────
    lines += ["", "## Q6: Does router + retrieval-kNN improve candidate coverage?", ""]
    if agg_pol and best_overall:
        bname = best_overall.get("run_name", "")
        pol_rows = [r for r in agg_pol if r.get("run_name") == bname]
        key_pols = [
            "router_top1", "router_top4", "router_top8",
            "mem_top4",
            "union_r4_m2", "union_r8_m4", "union_r16_m4",
            "adaptive_v1", "adaptive_v2", "adaptive_v3", "adaptive_v4", "adaptive_v5",
        ]
        pols_to_show = [r for r in pol_rows if r.get("policy") in key_pols]
        if pols_to_show:
            lines.append("| Policy | Coverage | Avg Regions | Avg Tokens | % Vocab | Mapped% |")
            lines.append("|--------|----------:|------------:|-----------:|--------:|--------:|")
            for pr in sorted(pols_to_show, key=lambda r: -_flt(r, "gold_region_coverage")):
                lines.append(
                    f"| {pr.get('policy')} "
                    f"| {_flt(pr,'gold_region_coverage'):.3f} "
                    f"| {_flt(pr,'avg_regions_kept'):.1f} "
                    f"| {_flt(pr,'avg_tokens_kept'):.0f} "
                    f"| {_flt(pr,'pct_vocab_kept'):.1f}% "
                    f"| {_flt(pr,'mapped_vocab_percent'):.1f}% |"
                )

    # ── Q7: Mix vs union vs diagnostic ────────────────────────────────────────
    lines += ["", "## Q7: Mix, candidate union, or diagnostic only?", ""]
    if best_overall and agg_spl:
        bname = best_overall.get("run_name", "")
        all_rows = [r for r in agg_spl if r.get("run_name") == bname
                    and r.get("split") == "all"]
        r_nll = next((_flt(r, "region_nll") for r in all_rows if r.get("method") == "router"),
                     float("nan"))
        m_nll = next((_flt(r, "region_nll") for r in all_rows if r.get("method") == "mem_weighted"),
                     float("nan"))
        x50_nll = next((_flt(r, "region_nll") for r in all_rows if r.get("method") == "mix_0.50"),
                       float("nan"))
        lines.append(f"- Router NLL: {r_nll:.3f}")
        lines.append(f"- Memory NLL: {m_nll:.3f}")
        lines.append(f"- Mix 0.50 NLL: {x50_nll:.3f}")
        if not math.isnan(x50_nll) and x50_nll < r_nll - 0.01:
            lines.append("→ **Mix improves over router: use as probability mixture**")
        else:
            lines.append("→ Mix does not beat router; use memory for **candidate expansion only**")

    # ── Q8: Memory misleading rate ────────────────────────────────────────────
    lines += ["", "## Q8: Does misleading rate drop with scale?", ""]
    p10_cal = sorted(
        [r for r in agg_res
         if r.get("checkpoint") == "proxy0.10"
         and r.get("key_source") == "retrieval_proj"
         and r.get("knn_k") == 64],
        key=lambda r: r.get("mem_positions", 0),
    )
    if p10_cal:
        lines.append("| Memory | Mem-conf top1 acc |")
        lines.append("|-------:|------------------:|")
        for r in p10_cal:
            acc = _flt(r, "mem_conf_top1_acc")
            lines.append(f"| {r.get('mem_positions',0):,} | {acc:.3f} |")

    # ── Q9: retrieval_proj vs raw_h ───────────────────────────────────────────
    lines += ["", "## Q9: Does retrieval_proj consistently beat raw_h?", ""]
    for ckpt in ("proxy0.05", "proxy0.10"):
        rh = next((r for r in agg_res
                   if r.get("checkpoint") == ckpt
                   and r.get("key_source") == "raw_h"
                   and r.get("mem_positions") == 500000
                   and r.get("knn_k") == 64), None)
        rp = next((r for r in agg_res
                   if r.get("checkpoint") == ckpt
                   and r.get("key_source") == "retrieval_proj"
                   and r.get("mem_positions") == 500000
                   and r.get("knn_k") == 64), None)
        if rh and rp:
            delta_nll  = _flt(rh, "memweighted_region_nll") - _flt(rp, "memweighted_region_nll")
            delta_acc1 = _flt(rp, "memweighted_acc1") - _flt(rh, "memweighted_acc1")
            verdict = "**PROJ BETTER**" if delta_nll > 0.01 else "**SIMILAR**"
            lines.append(
                f"- {ckpt}: raw_h NLL={_flt(rh,'memweighted_region_nll'):.3f}  "
                f"proj NLL={_flt(rp,'memweighted_region_nll'):.3f}  "
                f"ΔNLL={delta_nll:+.3f}  Δacc1={delta_acc1:+.3f}  {verdict}"
            )

    # ── Best overall config ───────────────────────────────────────────────────
    lines += ["", "## Best Overall Region-kNN Configuration", ""]
    if best_overall:
        lines.append(
            f"- **Checkpoint:** {best_overall.get('checkpoint')} "
            f"(λ={best_overall.get('lambda',0)})"
        )
        lines.append(f"- **Key source:** {best_overall.get('key_source')}")
        lines.append(f"- **Memory:** {best_overall.get('mem_positions',0):,}")
        lines.append(f"- **k:** {best_overall.get('knn_k')}")
        lines.append(f"- **temp:** {best_overall.get('knn_temp')}")
        lines.append(
            f"- **mem NLL:** {_flt(best_overall,'memweighted_region_nll'):.3f}  "
            f"router NLL: {_flt(best_overall,'router_region_nll'):.3f}"
        )

    # ── Best candidate policies by vocab budget ───────────────────────────────
    lines += ["", "## Best Candidate Policies by Mapped-Vocab Budget", ""]
    if agg_pol and best_overall:
        bname = best_overall.get("run_name", "")
        pol_rows = sorted(
            [r for r in agg_pol if r.get("run_name") == bname],
            key=lambda r: -_flt(r, "gold_region_coverage"),
        )
        for budget, budget_label in [(5, "≤5%"), (10, "≤10%"), (15, "≤15%")]:
            cand = next(
                (r for r in pol_rows
                 if not math.isnan(_flt(r, "mapped_vocab_percent"))
                 and _flt(r, "mapped_vocab_percent") <= budget),
                None,
            )
            if cand:
                lines.append(
                    f"- **{budget_label} mapped vocab**: `{cand.get('policy')}`  "
                    f"coverage={_flt(cand,'gold_region_coverage'):.3f}  "
                    f"mapped%={_flt(cand,'mapped_vocab_percent'):.1f}%"
                )
            else:
                lines.append(f"- **{budget_label} mapped vocab**: no policy found within budget")

    # ── Recommendation ────────────────────────────────────────────────────────
    lines += ["", "## Recommendation", ""]

    # Gather signals
    r_nll_all  = float("nan")
    m_nll_all  = float("nan")
    x50_nll_all = float("nan")
    if best_overall and agg_spl:
        bname = best_overall.get("run_name", "")
        all_s = [r for r in agg_spl if r.get("run_name") == bname and r.get("split") == "all"]
        r_nll_all   = next((_flt(r, "region_nll") for r in all_s if r.get("method") == "router"), float("nan"))
        m_nll_all   = next((_flt(r, "region_nll") for r in all_s if r.get("method") == "mem_weighted"), float("nan"))
        x50_nll_all = next((_flt(r, "region_nll") for r in all_s if r.get("method") == "mix_0.50"), float("nan"))

    mix_beats_router = (not math.isnan(x50_nll_all)) and (x50_nll_all < r_nll_all - 0.01)
    mem_scales       = len(nlls := [_flt(r, "memweighted_region_nll")
                                    for r in p10_rp]) >= 2 and nlls[-1] < nlls[0] - 0.01 \
                       if p10_rp else False
    best_adp_cov     = 0.0
    if agg_pol and best_overall:
        bname = best_overall.get("run_name", "")
        adp_rows = [r for r in agg_pol
                    if r.get("run_name") == bname
                    and str(r.get("policy", "")).startswith("adaptive")]
        if adp_rows:
            best_adp_cov = max(_flt(r, "gold_region_coverage") for r in adp_rows)

    cov_useful = best_adp_cov >= 0.90

    if mix_beats_router and mem_scales and cov_useful:
        conclusion = "A"
        rec = (
            "**Conclusion A — Keep as memory module.**\n\n"
            "Memory scale improves metrics, mixture beats router, and adaptive "
            "policies reach ≥90% gold-region coverage. "
            "Use `retrieval_proj(h)` as the kNN key with proxy retrieval training."
        )
    elif cov_useful and not mix_beats_router:
        conclusion = "B"
        rec = (
            "**Conclusion B — Use only for candidate expansion.**\n\n"
            "Memory probabilities do not beat the router in mixture, but union "
            "policies expand candidate coverage usefully. "
            "Do not mix memory into probabilities; use it to expand top-k candidate sets."
        )
    else:
        conclusion = "C"
        rec = (
            "**Conclusion C — Drop for now.**\n\n"
            "Memory does not scale, does not improve boundary coverage, and "
            "adaptive policies do not reach ≥90% gold-region coverage. "
            "Use router-only adaptive candidate pruning instead."
        )

    lines.append(rec)
    lines += [
        "",
        "---",
        f"*Conclusion: **{conclusion}***  "
        f"mix_beats_router={mix_beats_router}  "
        f"mem_scales={mem_scales}  "
        f"best_adaptive_coverage={best_adp_cov:.3f}",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {out_path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(agg_res: List[Dict], agg_spl: List[Dict],
               agg_pol: List[Dict], agg_typ: List[Dict],
               plots_dir: str) -> None:
    if not _HAS_PLT:
        print("[plots] matplotlib not available — skipping")
        return
    os.makedirs(plots_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ── 01: Memory scale — mem NLL (proxy010 retrieval_proj, k=64) ────────────
    rows = sorted(
        [r for r in agg_res
         if r.get("checkpoint") == "proxy0.10"
         and r.get("key_source") == "retrieval_proj"
         and r.get("knn_k") == 64],
        key=lambda r: r.get("mem_positions", 0),
    )
    if rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = [r["mem_positions"] for r in rows]
        ax.plot(xs, [_flt(r, "memweighted_region_nll") for r in rows],
                marker="o", label="mem NLL")
        ax.plot(xs, [_flt(r, "router_region_nll") for r in rows],
                linestyle="--", label="router NLL (constant)")
        ax.set_xscale("log")
        ax.set_xlabel("Memory positions")
        ax.set_ylabel("Region NLL")
        ax.set_title("Memory scale vs NLL (proxy0.10 retrieval_proj k=64 t=0.20)")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "01_memory_scale_mem_nll.png"), dpi=120)
        plt.close(fig)

    # ── 02: Memory scale — mem acc@1 ─────────────────────────────────────────
    if rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, [_flt(r, "memweighted_acc1") for r in rows],
                marker="o", label="mem acc@1")
        ax.plot(xs, [_flt(r, "router_acc1") for r in rows],
                linestyle="--", label="router acc@1")
        ax.set_xscale("log")
        ax.set_xlabel("Memory positions")
        ax.set_ylabel("Acc@1")
        ax.set_title("Memory scale vs acc@1")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "02_memory_scale_mem_acc1.png"), dpi=120)
        plt.close(fig)

    # ── 03: Memory scale — boundary NLL ──────────────────────────────────────
    bnd_nlls = []
    for r in rows:
        bname = r.get("run_name", "")
        bnd = next((s for s in agg_spl
                    if s.get("run_name") == bname
                    and s.get("method") == "mem_weighted"
                    and s.get("split") == "boundary"), None)
        bnd_nlls.append(_flt(bnd, "region_nll") if bnd else float("nan"))
    if rows and any(not math.isnan(v) for v in bnd_nlls):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r["mem_positions"] for r in rows], bnd_nlls, marker="o", label="boundary mem NLL")
        ax.set_xscale("log")
        ax.set_xlabel("Memory positions")
        ax.set_ylabel("Region NLL (boundary)")
        ax.set_title("Memory scale vs boundary NLL")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "03_memory_scale_boundary_nll.png"), dpi=120)
        plt.close(fig)

    # ── 04: k/temp heatmap — mem NLL ─────────────────────────────────────────
    kt_rows = [r for r in agg_res
               if r.get("checkpoint") == "proxy0.10"
               and r.get("key_source") == "retrieval_proj"
               and r.get("mem_positions") == 500000]
    ks    = sorted(set(r["knn_k"]    for r in kt_rows if r.get("knn_k", -1) > 0))
    temps = sorted(set(r["knn_temp"] for r in kt_rows
                       if not math.isnan(r.get("knn_temp", float("nan")))))
    if ks and temps:
        mat = np.full((len(ks), len(temps)), float("nan"))
        for r in kt_rows:
            ki = ks.index(r["knn_k"])  if r.get("knn_k")   in ks    else -1
            ti = temps.index(r["knn_temp"]) if r.get("knn_temp") in temps else -1
            if ki >= 0 and ti >= 0:
                mat[ki, ti] = _flt(r, "memweighted_region_nll")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(temps))); ax.set_xticklabels([str(t) for t in temps])
        ax.set_yticks(range(len(ks)));    ax.set_yticklabels([str(k) for k in ks])
        ax.set_xlabel("knn_temp"); ax.set_ylabel("knn_k")
        ax.set_title("mem NLL heatmap (proxy0.10 retrieval_proj mem=500k)")
        plt.colorbar(im, ax=ax, label="mem NLL")
        for i in range(len(ks)):
            for j in range(len(temps)):
                if not math.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                            fontsize=7, color="white")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "04_knn_k_temp_heatmap_mem_nll.png"), dpi=120)
        plt.close(fig)

    # ── 05: k/temp heatmap — boundary NLL ────────────────────────────────────
    if ks and temps:
        mat_bnd = np.full((len(ks), len(temps)), float("nan"))
        for r in kt_rows:
            bname = r.get("run_name", "")
            bnd = next((s for s in agg_spl
                        if s.get("run_name") == bname
                        and s.get("method") == "mem_weighted"
                        and s.get("split") == "boundary"), None)
            if bnd:
                ki = ks.index(r["knn_k"])       if r.get("knn_k")    in ks    else -1
                ti = temps.index(r["knn_temp"])  if r.get("knn_temp") in temps else -1
                if ki >= 0 and ti >= 0:
                    mat_bnd[ki, ti] = _flt(bnd, "region_nll")
        if not np.all(np.isnan(mat_bnd)):
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(mat_bnd, aspect="auto", cmap="viridis_r")
            ax.set_xticks(range(len(temps))); ax.set_xticklabels([str(t) for t in temps])
            ax.set_yticks(range(len(ks)));    ax.set_yticklabels([str(k) for k in ks])
            ax.set_xlabel("knn_temp"); ax.set_ylabel("knn_k")
            ax.set_title("boundary NLL heatmap (proxy0.10 retrieval_proj mem=500k)")
            plt.colorbar(im, ax=ax, label="boundary NLL")
            for i in range(len(ks)):
                for j in range(len(temps)):
                    if not math.isnan(mat_bnd[i, j]):
                        ax.text(j, i, f"{mat_bnd[i,j]:.3f}", ha="center", va="center",
                                fontsize=7, color="white")
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "05_knn_k_temp_heatmap_boundary_nll.png"), dpi=120)
            plt.close(fig)

    # ── 06: Coverage vs tokens kept ──────────────────────────────────────────
    best_overall_name = ""
    best_nll = float("inf")
    for r in agg_res:
        if r.get("key_source") == "retrieval_proj":
            v = _flt(r, "memweighted_region_nll")
            if v < best_nll:
                best_nll = v
                best_overall_name = r.get("run_name", "")

    pol_rows = [r for r in agg_pol if r.get("run_name") == best_overall_name]
    if pol_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        for ri, pr in enumerate(pol_rows):
            ax.scatter(_flt(pr, "avg_tokens_kept"), _flt(pr, "gold_region_coverage"),
                       s=60, color=colors[ri % len(colors)], zorder=3)
            ax.annotate(str(pr.get("policy", "")),
                        (_flt(pr, "avg_tokens_kept"), _flt(pr, "gold_region_coverage")),
                        textcoords="offset points", xytext=(4, 2), fontsize=6)
        ax.set_xlabel("Avg tokens kept")
        ax.set_ylabel("Gold region coverage")
        ax.set_title("Coverage vs tokens kept (best run)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "06_candidate_coverage_vs_tokens.png"), dpi=120)
        plt.close(fig)

    # ── 07: Coverage vs regions kept ─────────────────────────────────────────
    if pol_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        for ri, pr in enumerate(pol_rows):
            ax.scatter(_flt(pr, "avg_regions_kept"), _flt(pr, "gold_region_coverage"),
                       s=60, color=colors[ri % len(colors)], zorder=3)
            ax.annotate(str(pr.get("policy", "")),
                        (_flt(pr, "avg_regions_kept"), _flt(pr, "gold_region_coverage")),
                        textcoords="offset points", xytext=(4, 2), fontsize=6)
        ax.set_xlabel("Avg regions kept")
        ax.set_ylabel("Gold region coverage")
        ax.set_title("Coverage vs regions kept (best run)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "07_candidate_coverage_vs_regions.png"), dpi=120)
        plt.close(fig)

    # ── 08: Type A vs B improvement ──────────────────────────────────────────
    best_typ = [r for r in agg_typ if r.get("run_name") == best_overall_name]
    if best_typ:
        types = [r.get("type") for r in best_typ]
        imps  = [_flt(r, "improvement_vs_router") for r in best_typ]
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_t = {"A": "green", "B": "orange", "C": "red"}
        bars = ax.bar(types, imps, color=[colors_t.get(t, "grey") for t in types])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("NLL improvement over router (mix 0.50)")
        ax.set_title("NLL improvement by type A/B/C — boundary positions")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "08_type_A_vs_type_B_improvement.png"), dpi=120)
        plt.close(fig)

    # ── 09: Memory margin calibration across memory sizes ────────────────────
    cal_data = sorted(
        [(r.get("mem_positions", 0), _flt(r, "mem_conf_top1_acc"))
         for r in agg_res
         if r.get("checkpoint") == "proxy0.10"
         and r.get("key_source") == "retrieval_proj"
         and r.get("knn_k") == 64
         and not math.isnan(_flt(r, "mem_conf_top1_acc"))],
    )
    if cal_data:
        xs_c, ys_c = zip(*cal_data)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs_c, ys_c, marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("Memory positions")
        ax.set_ylabel("Memory top-1 acc when margin > 0.25")
        ax.set_title("Memory confidence calibration vs scale")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "09_memory_margin_calibration.png"), dpi=120)
        plt.close(fig)

    # ── 10: Router vs memory error overlap (raw_h vs retrieval_proj) ─────────
    comp_rows = [r for r in agg_res
                 if r.get("mem_positions") == 500000 and r.get("knn_k") == 64]
    if comp_rows:
        labels = [f"{r.get('checkpoint')}\n{r.get('key_source')}" for r in comp_rows]
        r_nlls = [_flt(r, "router_region_nll") for r in comp_rows]
        m_nlls = [_flt(r, "memweighted_region_nll") for r in comp_rows]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
        ax.bar(x - w/2, r_nlls, w, label="router NLL", color=colors[0])
        ax.bar(x + w/2, m_nlls, w, label="mem NLL",    color=colors[1])
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Region NLL")
        ax.set_title("Router vs memory NLL (mem=500k k=64)")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "10_router_vs_mem_error_overlap.png"), dpi=120)
        plt.close(fig)

    print(f"[plots] wrote up to 10 plots → {plots_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Region-kNN extensive sweep results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep_dir",  required=True,
                        help="Root directory containing per-run subdirs")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write aggregated files (defaults to sweep_dir)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.sweep_dir
    os.makedirs(out_dir, exist_ok=True)

    # Discover run directories
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(args.sweep_dir, "*"))
        if os.path.isdir(d)
        and os.path.exists(os.path.join(d, "metrics_all.csv"))
    )
    print(f"[aggregate] found {len(run_dirs)} completed run dirs in {args.sweep_dir}")

    run_metas = [_parse_run_dir(d) for d in run_dirs]
    run_data  = [_load_run(d) for d in run_dirs]

    # Build aggregates
    agg_res = build_aggregate_results(run_metas, run_data)
    agg_spl = build_aggregate_by_split(run_metas, run_data)
    agg_pol = build_aggregate_policies(run_metas, run_data)
    agg_typ = build_aggregate_types(run_metas, run_data)

    # Write CSVs
    _write_csv(agg_res, os.path.join(out_dir, "aggregate_results.csv"))
    _write_csv(agg_spl, os.path.join(out_dir, "aggregate_by_split.csv"))
    _write_csv(agg_pol, os.path.join(out_dir, "aggregate_candidate_policies.csv"))
    _write_csv(agg_typ, os.path.join(out_dir, "aggregate_type_analysis.csv"))
    print(f"[aggregate] wrote 4 CSVs to {out_dir}/")

    # Final report
    write_final_report(
        agg_res, agg_spl, agg_pol, agg_typ,
        os.path.join(out_dir, "final_report.md"),
    )

    # Plots
    make_plots(
        agg_res, agg_spl, agg_pol, agg_typ,
        os.path.join(out_dir, "plots"),
    )

    print(f"[aggregate] DONE — {out_dir}/")


if __name__ == "__main__":
    main()
