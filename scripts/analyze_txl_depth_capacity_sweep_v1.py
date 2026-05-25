#!/usr/bin/env python3
"""
analyze_txl_depth_capacity_sweep_v1.py — TXL Depth/Capacity Sweep V1 Analyzer

Reads config.json + final_metrics.json from each run under --sweep_root,
extracts key metrics, computes deltas vs Stage 2C reference, then writes:
  - sweep_root/sweep_summary.csv   (sorted by fv_gain desc then sip desc)
  - sweep_root/sweep_report.md     (7-section human-readable report)

Usage:
    python scripts/analyze_txl_depth_capacity_sweep_v1.py \
        --sweep_root runs/txl_depth_capacity_sweep_v1 \
        --stage2c_ctg 0.0084 --stage2c_caw 0.0089 \
        --stage2c_fv_gain -0.0017 --stage2c_sip 0.2457 \
        --stage2c_applied_precision 0.1047
"""

import argparse
import csv
import json
import math
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(v, digits=4):
    """Format float or return 'nan'/'N/A'."""
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _safe(d, *keys, default=float("nan")):
    """Nested dict get with default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _delta(v, ref, label=""):
    """Return signed delta string."""
    if math.isnan(v) or math.isnan(ref):
        return "nan"
    d = v - ref
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Load one run
# ─────────────────────────────────────────────────────────────────────────────

_PRIMARY_METRICS = [
    "selected_gold_given_in_pool",   # sip
    "selector_acc_val",
    "applied_precision_ctg",
    "benefit_damage_ratio",
    "changed_to_gold_rate",          # ctg
    "changed_away_rate",             # caw
    "net_correction",
    "apply_rate",
    "noop_acc_on_base_correct",
    "top1_acc_gain",
    "selector_ce_val",
]

_FULL_VOCAB_METRICS = [
    "full_vocab_gain",
    "full_vocab_base_nll",
    "full_vocab_refined_nll",
    "full_vocab_top1_acc_base",
    "full_vocab_top1_acc_refined",
]


def load_run(run_dir):
    """
    Load config.json + final_metrics.json from run_dir.
    Returns a flat dict of all fields needed for reporting, or None if missing.
    """
    cfg_path = os.path.join(run_dir, "config.json")
    fm_path  = os.path.join(run_dir, "final_metrics.json")

    if not os.path.isfile(cfg_path) or not os.path.isfile(fm_path):
        return None

    with open(cfg_path)  as f: cfg = json.load(f)
    with open(fm_path)   as f: fm  = json.load(f)

    run_name = os.path.basename(run_dir)

    # Architecture identity
    backend      = cfg.get("memory_backend", "token")
    txl_layers   = cfg.get("txl_layers", 0)
    txl_heads    = cfg.get("txl_heads",  0)
    mem_dim      = cfg.get("mem_dim",    cfg.get("resolver_dim", 256))
    resolver_dim = cfg.get("resolver_dim", 256)
    hidden_dim   = cfg.get("hidden_dim", 512)
    n_slots      = cfg.get("num_memory_slots", 0)
    pos_enc      = cfg.get("pos_encoding", "N/A")
    n_params     = cfg.get("n_params", float("nan"))
    n_txl_params = cfg.get("n_txl_params", float("nan"))

    # Argmax metrics (primary operating point)
    am = fm.get("argmax_metrics", {})
    bc = fm.get("best_conservative", {})
    bn = fm.get("best_net_correction", {})
    fv = fm.get("full_vocab", {})

    row = {
        "run_name":    run_name,
        "backend":     backend,
        "txl_layers":  txl_layers,
        "txl_heads":   txl_heads,
        "mem_dim":     mem_dim,
        "resolver_dim": resolver_dim,
        "hidden_dim":  hidden_dim,
        "n_slots":     n_slots,
        "pos_enc":     pos_enc,
        "n_params":    n_params,
        "n_txl_params": n_txl_params,
        "n_other_params": (n_params - n_txl_params
                           if not (math.isnan(n_params) or math.isnan(n_txl_params))
                           else float("nan")),
    }

    # Argmax metrics
    for k in _PRIMARY_METRICS:
        row[f"am_{k}"] = _safe(am, k)

    # Full-vocab metrics
    for k in _FULL_VOCAB_METRICS:
        row[f"fv_{k}"] = _safe(fv, k)

    # Best conservative
    row["bc_gate_thr"]   = bc.get("gate_threshold", float("nan"))
    row["bc_smt"]        = bc.get("selector_margin_threshold", float("nan"))
    row["bc_ctg"]        = _safe(bc, "changed_to_gold_rate")
    row["bc_caw"]        = _safe(bc, "changed_away_rate")
    row["bc_apply_rate"] = _safe(bc, "apply_rate")
    row["bc_precision"]  = _safe(bc, "applied_precision_ctg")
    row["bc_bdr"]        = _safe(bc, "benefit_damage_ratio")

    # Best net-correction
    row["bn_gate_thr"]   = bn.get("gate_threshold", float("nan"))
    row["bn_smt"]        = bn.get("selector_margin_threshold", float("nan"))
    row["bn_ctg"]        = _safe(bn, "changed_to_gold_rate")
    row["bn_caw"]        = _safe(bn, "changed_away_rate")
    row["bn_net"]        = _safe(bn, "net_correction")

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Discover runs
# ─────────────────────────────────────────────────────────────────────────────

_EXPECTED_RUNS = [
    "sweep_A0_txl_L2_D256",
    "sweep_A1_txl_L4_D256",
    "sweep_A2_txl_L6_D256",
    "sweep_A3_txl_L4_D384",
    "sweep_A4_txl_L6_D384",
    "sweep_B0_token_h1024",
    "sweep_B1_token_h1408",
    "sweep_B2_token_r384_h1536",
]


def discover_runs(sweep_root):
    """
    Yield (run_name, loaded_dict) for every run we can read.
    Checks _EXPECTED_RUNS first, then any other sub-dir with final_metrics.json.
    """
    found = set()
    runs  = []

    # Expected runs first (deterministic order)
    for rn in _EXPECTED_RUNS:
        rd = os.path.join(sweep_root, rn)
        r  = load_run(rd)
        if r is not None:
            found.add(rn)
            runs.append(r)

    # Any additional runs (alphabetical)
    try:
        for entry in sorted(os.listdir(sweep_root)):
            if entry in found:
                continue
            rd = os.path.join(sweep_root, entry)
            if not os.path.isdir(rd):
                continue
            r = load_run(rd)
            if r is not None:
                found.add(entry)
                runs.append(r)
    except OSError:
        pass

    return runs


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "run_name", "backend", "txl_layers", "txl_heads", "mem_dim",
    "resolver_dim", "hidden_dim", "n_slots", "pos_enc",
    "n_params", "n_txl_params", "n_other_params",
    # argmax primary
    "am_selected_gold_given_in_pool", "am_selector_acc_val",
    "am_applied_precision_ctg", "am_benefit_damage_ratio",
    "am_changed_to_gold_rate", "am_changed_away_rate",
    "am_net_correction", "am_apply_rate",
    "am_noop_acc_on_base_correct", "am_top1_acc_gain", "am_selector_ce_val",
    # full vocab
    "fv_full_vocab_gain", "fv_full_vocab_base_nll", "fv_full_vocab_refined_nll",
    "fv_full_vocab_top1_acc_base", "fv_full_vocab_top1_acc_refined",
    # best conservative
    "bc_gate_thr", "bc_smt", "bc_ctg", "bc_caw", "bc_apply_rate",
    "bc_precision", "bc_bdr",
    # best net-correction
    "bn_gate_thr", "bn_smt", "bn_ctg", "bn_caw", "bn_net",
    # deltas vs stage2c
    "delta_ctg", "delta_caw", "delta_fv_gain",
    "delta_sip", "delta_applied_precision",
]


def add_deltas(row, ref):
    """Compute delta_* fields vs Stage 2C reference."""
    row["delta_ctg"]               = row["am_changed_to_gold_rate"]    - ref["ctg"]
    row["delta_caw"]               = row["am_changed_away_rate"]       - ref["caw"]
    row["delta_fv_gain"]           = row["fv_full_vocab_gain"]         - ref["fv_gain"]
    row["delta_sip"]               = row["am_selected_gold_given_in_pool"] - ref["sip"]
    row["delta_applied_precision"] = row["am_applied_precision_ctg"]   - ref["applied_precision"]
    # Make NaN explicit
    for k in ["delta_ctg", "delta_caw", "delta_fv_gain", "delta_sip", "delta_applied_precision"]:
        v = row[k]
        if not isinstance(v, float) or (not math.isfinite(v) and not math.isnan(v)):
            row[k] = float("nan")


def write_csv(runs, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            # Format floats
            row = {}
            for k in _CSV_FIELDS:
                v = r.get(k, "")
                if isinstance(v, float):
                    row[k] = "" if math.isnan(v) else f"{v:.6f}"
                else:
                    row[k] = v
            w.writerow(row)
    print(f"[CSV] Written: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _group(runs, backend):
    return [r for r in runs if r.get("backend") == backend]


def _pct(v):
    if isinstance(v, float) and not math.isnan(v):
        return f"{v*100:.2f}%"
    return "N/A"


def _bdr_str(v):
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:.3f}"
    return "N/A"


def write_report(runs, out_path, ref, sweep_root):
    txl_runs   = sorted(_group(runs, "txl"),   key=lambda r: r.get("txl_layers", 0))
    token_runs = _group(runs, "token")

    def tbl_row(r, indent=""):
        sip  = _f(r["am_selected_gold_given_in_pool"])
        ctg  = _f(r["am_changed_to_gold_rate"])
        caw  = _f(r["am_changed_away_rate"])
        fvg  = _f(r["fv_full_vocab_gain"])
        prec = _f(r["am_applied_precision_ctg"])
        bdr  = _f(r["am_benefit_damage_ratio"])
        ar   = _f(r["am_apply_rate"])
        np_  = r.get("n_params", float("nan"))
        np_s = f"{int(np_):,}" if isinstance(np_, float) and not math.isnan(np_) else "N/A"
        return (f"{indent}| {r['run_name']:<32} | {r['backend']:<6} | "
                f"L={r['txl_layers']} D={r['mem_dim']} | "
                f"{sip:<8} | {ctg:<8} | {caw:<8} | {fvg:<8} | "
                f"{prec:<8} | {bdr:<6} | {ar:<8} | {np_s:<10} |")

    def tbl_header():
        return (
            "| run_name                         | back   | arch      "
            "| sip      | ctg      | caw      | fv_gain  "
            "| precision| bdr    | apply_r  | n_params   |\n"
            "|----------------------------------|--------|-----------|"
            "----------|----------|----------|----------|"
            "----------|--------|----------|------------|"
        )

    lines = []
    lines.append("# TXL Depth/Capacity Sweep V1 — Analysis Report\n")
    lines.append(f"**Sweep root:** `{sweep_root}`\n")
    lines.append(f"**Runs loaded:** {len(runs)}"
                 f"  ({len(txl_runs)} TXL, {len(token_runs)} token control)\n")
    lines.append("")

    # ── Section 1: Stage 2C reference ─────────────────────────────────────────
    lines.append("## 1. Stage 2C Reference (Baseline)\n")
    lines.append(f"- apply_rate                   = {ref['apply_rate']}")
    lines.append(f"- changed_to_gold_rate (ctg)   = {ref['ctg']}")
    lines.append(f"- changed_away_rate (caw)       = {ref['caw']}")
    lines.append(f"- full_vocab_gain               = {ref['fv_gain']}")
    lines.append(f"- selected_gold_in_pool (sip)  = {ref['sip']}")
    lines.append(f"- applied_precision_ctg        = {ref['applied_precision']}")
    lines.append(f"- V3 oracle target_ctg          = 0.2407  fv_gain = +0.1584")
    lines.append("")

    # ── Section 2: All runs table ──────────────────────────────────────────────
    lines.append("## 2. All Runs — Argmax Metrics\n")
    if not runs:
        lines.append("*No runs found.*\n")
    else:
        lines.append(tbl_header())
        for r in runs:
            lines.append(tbl_row(r))
    lines.append("")

    # ── Section 3: TXL depth trend ────────────────────────────────────────────
    lines.append("## 3. TXL Depth Trend (Group A)\n")
    if not txl_runs:
        lines.append("*No TXL runs available.*\n")
    else:
        lines.append("Deeper = more layers at fixed mem_dim; "
                     "wider = larger mem_dim + resolver_dim.\n")
        lines.append("| run               | L | mem_dim | sip    | ctg    | fv_gain | Δsip   | Δctg   | Δfv    |")
        lines.append("|-------------------|---|---------|--------|--------|---------|--------|--------|--------|")
        for r in txl_runs:
            dsip = _delta(r["am_selected_gold_given_in_pool"], ref["sip"])
            dctg = _delta(r["am_changed_to_gold_rate"],        ref["ctg"])
            dfvg = _delta(r["fv_full_vocab_gain"],             ref["fv_gain"])
            lines.append(
                f"| {r['run_name']:<17} | {r['txl_layers']} | {r['mem_dim']:<7} | "
                f"{_f(r['am_selected_gold_given_in_pool']):<6} | "
                f"{_f(r['am_changed_to_gold_rate']):<6} | "
                f"{_f(r['fv_full_vocab_gain']):<7} | "
                f"{dsip:<6} | {dctg:<6} | {dfvg:<6} |"
            )
    lines.append("")

    # ── Section 4: Parameter-matched token controls ────────────────────────────
    lines.append("## 4. Parameter-Matched Token-Embedding Controls (Group B)\n")
    if not token_runs:
        lines.append("*No token-backend runs available.*\n")
    else:
        lines.append("| run                      | hidden | resolver | sip    | ctg    | fv_gain | Δsip   | Δctg   | Δfv    |")
        lines.append("|--------------------------|--------|----------|--------|--------|---------|--------|--------|--------|")
        for r in token_runs:
            dsip = _delta(r["am_selected_gold_given_in_pool"], ref["sip"])
            dctg = _delta(r["am_changed_to_gold_rate"],        ref["ctg"])
            dfvg = _delta(r["fv_full_vocab_gain"],             ref["fv_gain"])
            lines.append(
                f"| {r['run_name']:<24} | {r['hidden_dim']:<6} | {r['resolver_dim']:<8} | "
                f"{_f(r['am_selected_gold_given_in_pool']):<6} | "
                f"{_f(r['am_changed_to_gold_rate']):<6} | "
                f"{_f(r['fv_full_vocab_gain']):<7} | "
                f"{dsip:<6} | {dctg:<6} | {dfvg:<6} |"
            )
    lines.append("")

    # ── Section 5: TXL vs token controls comparison ───────────────────────────
    lines.append("## 5. TXL vs Token-Embedding Controls\n")
    if txl_runs and token_runs:
        # Best TXL by fv_gain
        def sort_key(r):
            fvg = r["fv_full_vocab_gain"]
            sip = r["am_selected_gold_given_in_pool"]
            fvg = -1e9 if math.isnan(fvg) else fvg
            sip = -1e9 if math.isnan(sip) else sip
            return (fvg, sip)

        best_txl   = max(txl_runs,   key=sort_key)
        best_token = max(token_runs, key=sort_key)

        lines.append(f"**Best TXL run:**   `{best_txl['run_name']}`")
        lines.append(f"  - sip={_f(best_txl['am_selected_gold_given_in_pool'])}  "
                     f"ctg={_f(best_txl['am_changed_to_gold_rate'])}  "
                     f"fv_gain={_f(best_txl['fv_full_vocab_gain'])}  "
                     f"n_params={int(best_txl['n_params']):,}" if not math.isnan(best_txl['n_params'])
                     else f"  - sip={_f(best_txl['am_selected_gold_given_in_pool'])}  "
                          f"ctg={_f(best_txl['am_changed_to_gold_rate'])}  "
                          f"fv_gain={_f(best_txl['fv_full_vocab_gain'])}")
        lines.append("")
        lines.append(f"**Best token run:** `{best_token['run_name']}`")
        lines.append(f"  - sip={_f(best_token['am_selected_gold_given_in_pool'])}  "
                     f"ctg={_f(best_token['am_changed_to_gold_rate'])}  "
                     f"fv_gain={_f(best_token['fv_full_vocab_gain'])}  "
                     f"n_params={int(best_token['n_params']):,}" if not math.isnan(best_token['n_params'])
                     else f"  - sip={_f(best_token['am_selected_gold_given_in_pool'])}  "
                          f"ctg={_f(best_token['am_changed_to_gold_rate'])}  "
                          f"fv_gain={_f(best_token['fv_full_vocab_gain'])}")
        lines.append("")

        bt_fvg = best_txl["fv_full_vocab_gain"]
        bk_fvg = best_token["fv_full_vocab_gain"]
        bt_sip = best_txl["am_selected_gold_given_in_pool"]
        bk_sip = best_token["am_selected_gold_given_in_pool"]

        if not (math.isnan(bt_fvg) or math.isnan(bk_fvg)):
            if bt_fvg > bk_fvg + 0.001:
                lines.append("**Conclusion:** TXL contextual memory **outperforms** token-embedding "
                              f"controls on fv_gain (Δ={bt_fvg - bk_fvg:+.4f}).")
            elif bk_fvg > bt_fvg + 0.001:
                lines.append("**Conclusion:** Token-embedding controls **outperform** TXL on fv_gain "
                              f"(Δ={bk_fvg - bt_fvg:+.4f}). TXL contextual encoding not helping.")
            else:
                lines.append("**Conclusion:** TXL and token-embedding controls perform **similarly** "
                              f"on fv_gain (Δ={bt_fvg - bk_fvg:+.4f}). "
                              "Contextual encoding adds overhead without benefit.")
        else:
            lines.append("*Comparison not available — one or both runs missing fv_gain.*")
        lines.append("")

        if not (math.isnan(bt_sip) or math.isnan(bk_sip)):
            if bt_sip > bk_sip + 0.005:
                lines.append(f"  SIP: TXL retrieves gold more reliably "
                              f"(Δ={bt_sip - bk_sip:+.4f}), suggesting contextual "
                              "encoding helps narrow the candidate pool.")
            elif bk_sip > bt_sip + 0.005:
                lines.append(f"  SIP: Token controls retrieve gold more reliably "
                              f"(Δ={bk_sip - bt_sip:+.4f}).")
            else:
                lines.append(f"  SIP: Similar gold-in-pool retrieval (Δ={bt_sip - bk_sip:+.4f}).")
    else:
        lines.append("*Not enough runs from both groups to compare.*\n")
    lines.append("")

    # ── Section 6: Sorted ranking ──────────────────────────────────────────────
    lines.append("## 6. Run Ranking (sorted by fv_gain ↓, then sip ↓)\n")

    def sort_key_full(r):
        fvg = r["fv_full_vocab_gain"]
        sip = r["am_selected_gold_given_in_pool"]
        return (-(1e9 if math.isnan(fvg) else -fvg),
                -(1e9 if math.isnan(sip) else -sip))

    sorted_runs = sorted(runs, key=lambda r: (
        -1e9 if math.isnan(r["fv_full_vocab_gain"]) else r["fv_full_vocab_gain"],
        -1e9 if math.isnan(r["am_selected_gold_given_in_pool"]) else r["am_selected_gold_given_in_pool"],
    ), reverse=True)

    lines.append("| rank | run_name                         | backend | fv_gain  | sip      | ctg      | caw      |")
    lines.append("|------|----------------------------------|---------|----------|----------|----------|----------|")
    for i, r in enumerate(sorted_runs, 1):
        lines.append(
            f"| {i:<4} | {r['run_name']:<32} | {r['backend']:<7} | "
            f"{_f(r['fv_full_vocab_gain']):<8} | "
            f"{_f(r['am_selected_gold_given_in_pool']):<8} | "
            f"{_f(r['am_changed_to_gold_rate']):<8} | "
            f"{_f(r['am_changed_away_rate']):<8} |"
        )
    lines.append("")

    # ── Section 7: Missing runs ────────────────────────────────────────────────
    lines.append("## 7. Run Status\n")
    loaded_names = {r["run_name"] for r in runs}
    all_ok = True
    for rn in _EXPECTED_RUNS:
        status = "✅ OK" if rn in loaded_names else "❌ MISSING"
        lines.append(f"- {status}: `{rn}`")
        if rn not in loaded_names:
            all_ok = False
    lines.append("")
    if all_ok:
        lines.append("All expected runs are present.\n")
    else:
        lines.append("Some runs are missing — check logs for failures.\n")

    # ── Section 8: Interpretation guide ───────────────────────────────────────
    lines.append("## 8. Interpretation Guide\n")
    lines.append("- **sip** (selected_gold_given_in_pool): gold is rank-1 among candidates → "
                 "selector quality; higher is better.")
    lines.append("- **ctg** (changed_to_gold_rate): fraction of all tokens where model applies "
                 "surgical edit AND gold becomes top1 → primary win metric.")
    lines.append("- **caw** (changed_away_rate): fraction where edit *hurts* (gold was already "
                 "top1 but edit knocked it out) → primary harm metric.")
    lines.append("- **fv_gain**: change in full-vocab top1 accuracy after applying all edits "
                 "(positive = net improvement across all tokens).")
    lines.append("- **net_correction** = ctg − caw; positive means benefit exceeds harm.")
    lines.append("- **applied_precision_ctg**: of tokens where we edited, fraction that were "
                 "genuine corrections (ctg rows) → precision of editing.")
    lines.append("- **benefit_damage_ratio** = ctg / caw; > 1 means more corrections than harms.")
    lines.append("- **TXL wins** if: best-TXL fv_gain > best-token fv_gain + 0.001 AND "
                 "sip gain is non-trivial (> 0.005). Both must hold — sip alone is not enough.")
    lines.append("- **If token controls win**: TXL overhead not justified; consider "
                 "architectural alternatives (e.g., RNN memory, hierarchical slots).")
    lines.append("- **If similar**: TXL and token MLP are equivalent; use simpler token baseline.")
    lines.append("")

    report = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"[report] Written: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="TXL Depth/Capacity Sweep V1 Analyzer")
    p.add_argument("--sweep_root",              required=True,
                   help="Root directory containing all sweep run sub-dirs.")
    p.add_argument("--stage2c_ctg",             type=float, default=0.0084)
    p.add_argument("--stage2c_caw",             type=float, default=0.0089)
    p.add_argument("--stage2c_fv_gain",         type=float, default=-0.0017)
    p.add_argument("--stage2c_sip",             type=float, default=0.2457)
    p.add_argument("--stage2c_applied_precision", type=float, default=0.1047)
    p.add_argument("--stage2c_apply_rate",      type=float, default=0.0798)
    return p.parse_args()


def main():
    args = _parse()

    if not os.path.isdir(args.sweep_root):
        print(f"ERROR: sweep_root not found: {args.sweep_root}", file=sys.stderr)
        sys.exit(1)

    ref = {
        "ctg":               args.stage2c_ctg,
        "caw":               args.stage2c_caw,
        "fv_gain":           args.stage2c_fv_gain,
        "sip":               args.stage2c_sip,
        "applied_precision": args.stage2c_applied_precision,
        "apply_rate":        args.stage2c_apply_rate,
    }

    print(f"[analyze] Scanning {args.sweep_root} ...")
    runs = discover_runs(args.sweep_root)
    print(f"[analyze] Found {len(runs)} completed run(s).")

    if not runs:
        print("WARNING: No completed runs found. Nothing to analyze.")
        # Still write empty outputs so the SLURM script doesn't fail the check.
        csv_path    = os.path.join(args.sweep_root, "sweep_summary.csv")
        report_path = os.path.join(args.sweep_root, "sweep_report.md")
        with open(csv_path,    "w") as f: f.write(",".join(_CSV_FIELDS) + "\n")
        with open(report_path, "w") as f: f.write("# TXL Depth/Capacity Sweep V1\n\nNo runs completed.\n")
        print(f"[CSV]    Written (empty): {csv_path}")
        print(f"[report] Written (empty): {report_path}")
        return

    # Compute deltas
    for r in runs:
        add_deltas(r, ref)

    # Sort by fv_gain desc, then sip desc
    def sort_key(r):
        fvg = r["fv_full_vocab_gain"]
        sip = r["am_selected_gold_given_in_pool"]
        return (
            -1e9 if math.isnan(fvg) else fvg,
            -1e9 if math.isnan(sip) else sip,
        )
    runs_sorted = sorted(runs, key=sort_key, reverse=True)

    csv_path    = os.path.join(args.sweep_root, "sweep_summary.csv")
    report_path = os.path.join(args.sweep_root, "sweep_report.md")

    write_csv(runs_sorted, csv_path)
    write_report(runs_sorted, report_path, ref, args.sweep_root)

    # Console summary
    print("")
    print("=== Sweep Summary (sorted by fv_gain ↓) ===")
    print(f"{'run':<34} {'back':<6} {'fv_gain':>8} {'sip':>8} {'ctg':>8} {'caw':>8}")
    print("-" * 76)
    for r in runs_sorted:
        print(f"{r['run_name']:<34} {r['backend']:<6} "
              f"{_f(r['fv_full_vocab_gain']):>8} "
              f"{_f(r['am_selected_gold_given_in_pool']):>8} "
              f"{_f(r['am_changed_to_gold_rate']):>8} "
              f"{_f(r['am_changed_away_rate']):>8}")
    print("")
    print(f"[reference Stage 2C]  fv_gain={ref['fv_gain']:+.4f}  "
          f"sip={ref['sip']:.4f}  ctg={ref['ctg']:.4f}  caw={ref['caw']:.4f}")
    print("")


if __name__ == "__main__":
    main()
