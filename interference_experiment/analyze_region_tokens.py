#!/usr/bin/env python3
"""
analyze_region_tokens.py

Maps token sub-indices in full_vocab_region_eval artifacts back to actual
token strings and produces human-readable region reports.

Works with outputs from full_vocab_region_eval.py (leaf_region_to_tokens.json,
frequent_token_ids.npy, token_freqs.npy, region_tree.json).

Usage:
    python interference_experiment/analyze_region_tokens.py \
        --output_dir interference_experiment/full_vocab_region_eval \
        --model_name gpt2-xl \
        --top_n 30

Outputs (all inside --output_dir/region_token_analysis/):
    region_token_report.txt   — human-readable per-region token listing
    region_tokens.json        — same format as cluster_tokens_coactivation_final.json
    region_stats.csv          — per-region cohesion metrics
    coarse_token_report.txt   — same but for coarse regions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",
                   default="interference_experiment/full_vocab_region_eval")
    p.add_argument("--model_name", default="gpt2-xl")
    p.add_argument("--top_n", type=int, default=30,
                   help="Max tokens to show per region in the report")
    p.add_argument("--min_size", type=int, default=2,
                   help="Skip regions smaller than this")
    p.add_argument("--sort_by", default="size",
                   choices=["size", "cohesion", "id"],
                   help="How to order regions in the report")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Token helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def tok_str(tokenizer, token_id: int) -> str:
    """Readable token string: replaces Ġ with a visible space marker."""
    raw = tokenizer.convert_ids_to_tokens(int(token_id))
    if raw is None:
        return f"<id={token_id}>"
    # GPT-2 uses Ġ (U+0120) for word-initial space, Ċ for newline
    return raw.replace("Ġ", " ").replace("Ċ", "\\n").replace("ĉ", "\\t")


def classify_token(s: str) -> str:
    """Coarse type label for a token string."""
    core = s.lstrip()
    if not core:
        return "whitespace"
    if re.fullmatch(r"[.,!?;:'\"\-\(\)\[\]\{\}\/\\@#%&*+=<>~`|^_]+", core):
        return "punctuation"
    if re.fullmatch(r"\d+[\d,.]*", core):
        return "numeric"
    if re.fullmatch(r"[A-Za-z]+", core):
        return "alpha"
    if re.fullmatch(r"[A-Za-z\d]+", core):
        return "alphanum"
    return "mixed"


# ──────────────────────────────────────────────────────────────────────────────
# Cohesion metrics
# ──────────────────────────────────────────────────────────────────────────────

def cohesion_metrics(token_strings: list[str], freqs: list[float]) -> dict:
    """
    Returns a dict of cohesion/homogeneity metrics for a set of token strings.
    """
    n = len(token_strings)
    if n == 0:
        return {}

    cores  = [s.lstrip() for s in token_strings]
    types  = [classify_token(s) for s in token_strings]
    type_counts = Counter(types)

    space_frac  = sum(1 for s in token_strings if s.startswith(" ")) / n
    upper_frac  = sum(1 for c in cores if c and c[0].isupper()) / n
    digit_frac  = type_counts["numeric"] / n
    alpha_frac  = (type_counts["alpha"] + type_counts["alphanum"]) / n
    punct_frac  = type_counts["punctuation"] / n

    # Most common 2-char prefix coverage
    prefixes = [c[:2].lower() for c in cores if len(c) >= 2]
    if prefixes:
        top_prefix, top_count = Counter(prefixes).most_common(1)[0]
        prefix2_coverage = top_count / n
    else:
        top_prefix, prefix2_coverage = "", 0.0

    # Most common 3-char prefix
    prefixes3 = [c[:3].lower() for c in cores if len(c) >= 3]
    if prefixes3:
        top_prefix3, top_count3 = Counter(prefixes3).most_common(1)[0]
        prefix3_coverage = top_count3 / n
    else:
        top_prefix3, prefix3_coverage = "", 0.0

    # Dominant type fraction
    dominant_type = type_counts.most_common(1)[0][0]
    dominant_type_frac = type_counts.most_common(1)[0][1] / n

    # Frequency Gini (higher = one token dominates)
    f = np.array(freqs, dtype=np.float64)
    if f.sum() > 0 and n > 1:
        f = f / f.sum()
        f_sorted = np.sort(f)
        n_f = len(f_sorted)
        gini = (2 * np.sum(np.arange(1, n_f + 1) * f_sorted) - (n_f + 1)) / n_f
    else:
        gini = 0.0

    # Subword fraction: tokens that do NOT start with a space and are NOT punctuation
    subword_frac = sum(
        1 for s, t in zip(token_strings, types)
        if not s.startswith(" ") and t in ("alpha", "alphanum", "mixed")
    ) / n

    return {
        "space_frac":        round(space_frac,        4),
        "upper_frac":        round(upper_frac,         4),
        "digit_frac":        round(digit_frac,         4),
        "alpha_frac":        round(alpha_frac,         4),
        "punct_frac":        round(punct_frac,         4),
        "subword_frac":      round(subword_frac,       4),
        "dominant_type":     dominant_type,
        "dominant_type_frac":round(dominant_type_frac, 4),
        "top_prefix2":       top_prefix,
        "prefix2_coverage":  round(prefix2_coverage,  4),
        "top_prefix3":       top_prefix3,
        "prefix3_coverage":  round(prefix3_coverage,  4),
        "freq_gini":         round(gini,               4),
    }


def region_label(metrics: dict) -> str:
    """Heuristic label describing what kind of tokens dominate the region."""
    sf  = metrics["space_frac"]
    df  = metrics["digit_frac"]
    pf  = metrics["punct_frac"]
    uf  = metrics["upper_frac"]
    swf = metrics["subword_frac"]
    af  = metrics["alpha_frac"]
    p2  = metrics["prefix2_coverage"]

    if df >= 0.5:
        return "NUMERIC"
    if pf >= 0.5:
        return "PUNCTUATION"
    if swf >= 0.6:
        if p2 >= 0.4:
            return f"SUBWORD-PREFIX({metrics['top_prefix2']})"
        return "SUBWORD"
    if sf >= 0.7:
        if uf >= 0.5:
            return "WORD-CAPITALIZED"
        if af >= 0.8:
            return "WORD-LOWER"
        return "WORD-MIXED"
    if sf >= 0.3:
        return "MIXED-POSITION"
    return "OTHER"


# ──────────────────────────────────────────────────────────────────────────────
# Load artifacts
# ──────────────────────────────────────────────────────────────────────────────

def load_artifacts(out_dir: str) -> dict:
    def req(fname: str) -> str:
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            sys.exit(f"Required file not found: {p}\n"
                     f"Make sure the cluster stage has completed.")
        return p

    top_ids    = np.load(req("frequent_token_ids.npy"))   # (V,) actual vocab IDs
    tok_freqs  = np.load(req("token_freqs.npy"))          # (V,) corpus counts

    with open(req("leaf_region_to_tokens.json")) as f:
        leaf_r2t_str: dict[str, list[int]] = json.load(f)
    leaf_region_to_tokens = {int(k): v for k, v in leaf_r2t_str.items()}

    with open(req("region_tree.json")) as f:
        region_tree: dict = json.load(f)

    return {
        "top_ids":               top_ids,
        "token_freqs":           tok_freqs,
        "leaf_region_to_tokens": leaf_region_to_tokens,
        "region_tree":           region_tree,
    }


def coarse_regions_from_tree(region_tree: dict) -> dict[int, list[int]]:
    """Extract coarse region → sub-indices from tree top-level children."""
    result: dict[int, list[int]] = {}
    for coarse_id, child in enumerate(region_tree.get("children", [])):
        result[coarse_id] = child.get("vocab_ids", [])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Build per-region token records
# ──────────────────────────────────────────────────────────────────────────────

def build_region_records(
    region_to_sub_indices: dict[int, list[int]],
    top_ids: np.ndarray,
    token_freqs: np.ndarray,
    tokenizer,
    min_size: int,
) -> list[dict]:
    """
    Returns list of region dicts, each containing:
        id, size, tokens (sorted by freq desc), cohesion metrics, label
    """
    records = []
    for region_id, sub_indices in region_to_sub_indices.items():
        if len(sub_indices) < min_size:
            continue

        # Sub-index → actual vocab token ID → string
        vocab_ids = top_ids[sub_indices]
        freqs     = token_freqs[sub_indices]

        # Sort by frequency descending
        order     = np.argsort(freqs)[::-1]
        vocab_ids = vocab_ids[order]
        freqs     = freqs[order]
        sub_sorted = np.array(sub_indices)[order]

        tokens = []
        for vid, freq, sub in zip(vocab_ids, freqs, sub_sorted):
            tokens.append({
                "token":    tok_str(tokenizer, int(vid)),
                "token_id": int(vid),
                "sub_idx":  int(sub),
                "freq":     int(freq),
            })

        tok_strings = [t["token"] for t in tokens]
        freq_vals   = [t["freq"]  for t in tokens]
        metrics     = cohesion_metrics(tok_strings, freq_vals)
        label       = region_label(metrics)

        records.append({
            "id":        region_id,
            "size":      len(tokens),
            "total_freq": int(sum(freq_vals)),
            "label":     label,
            "metrics":   metrics,
            "tokens":    tokens,
        })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Sort + report
# ──────────────────────────────────────────────────────────────────────────────

def sort_records(records: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "size":
        return sorted(records, key=lambda r: -r["size"])
    if sort_by == "cohesion":
        return sorted(records, key=lambda r: -r["metrics"].get("dominant_type_frac", 0))
    return sorted(records, key=lambda r: r["id"])


def write_text_report(
    records: list[dict],
    path: str,
    top_n: int,
    title: str = "Region Token Report",
) -> None:
    sizes = [r["size"] for r in records]
    lines: list[str] = [
        "=" * 70,
        title,
        "=" * 70,
        f"Total regions:  {len(records)}",
        f"Total tokens:   {sum(sizes):,}",
        f"Size  min:      {min(sizes)}",
        f"Size  median:   {int(np.median(sizes))}",
        f"Size  mean:     {np.mean(sizes):.1f}",
        f"Size  max:      {max(sizes)}",
        "",
        "Label distribution:",
    ]
    label_counts = Counter(r["label"] for r in records)
    for label, count in label_counts.most_common():
        lines.append(f"  {label:35s}  {count:5d}  ({100*count/len(records):.1f}%)")
    lines.append("")

    for r in records:
        m = r["metrics"]
        lines += [
            "─" * 70,
            f"Region {r['id']:5d}  |  size={r['size']:5d}  |  label={r['label']}",
            f"  total_freq={r['total_freq']:,}  "
            f"space={m['space_frac']:.2f}  "
            f"upper={m['upper_frac']:.2f}  "
            f"digit={m['digit_frac']:.2f}  "
            f"subword={m['subword_frac']:.2f}  "
            f"gini={m['freq_gini']:.2f}",
            f"  prefix2='{m['top_prefix2']}' ({m['prefix2_coverage']:.2f})  "
            f"prefix3='{m['top_prefix3']}' ({m['prefix3_coverage']:.2f})",
            f"  Top {min(top_n, r['size'])} tokens (freq):",
        ]
        for t in r["tokens"][:top_n]:
            lines.append(f"    {t['token']!r:25s}  id={t['token_id']:6d}  freq={t['freq']:,}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {path}")


def write_json(records: list[dict], path: str) -> None:
    """Write in same format as cluster_tokens_coactivation_final.json."""
    out: dict[str, list[dict]] = {}
    for r in records:
        out[str(r["id"])] = [
            {"token": t["token"], "token_id": t["token_id"], "freq": t["freq"]}
            for t in r["tokens"]
        ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Written: {path}")


def write_stats_csv(records: list[dict], path: str) -> None:
    if not records:
        return
    fieldnames = ["id", "size", "total_freq", "label"] + list(records[0]["metrics"].keys())
    rows = []
    for r in records:
        row = {"id": r["id"], "size": r["size"],
               "total_freq": r["total_freq"], "label": r["label"]}
        row.update(r["metrics"])
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(records: list[dict], title: str, top_n_regions: int = 20) -> None:
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    label_counts = Counter(r["label"] for r in records)
    print(f"Regions: {len(records)}   Tokens covered: {sum(r['size'] for r in records):,}")
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        bar = "█" * int(30 * count / len(records))
        print(f"  {label:35s} {count:4d}  {bar}")

    print(f"\nTop {top_n_regions} regions by size:")
    print(f"  {'id':>6}  {'size':>6}  {'label':35s}  top-5 tokens")
    print(f"  {'─'*6}  {'─'*6}  {'─'*35}  {'─'*40}")
    for r in sorted(records, key=lambda r: -r["size"])[:top_n_regions]:
        top5 = "  ".join(repr(t["token"]) for t in r["tokens"][:5])
        print(f"  {r['id']:>6}  {r['size']:>6}  {r['label']:35s}  {top5}")

    # Highlight interesting tight clusters (small, high cohesion)
    tight = [
        r for r in records
        if r["size"] <= 50 and r["metrics"]["dominant_type_frac"] >= 0.8
    ]
    if tight:
        print(f"\nTight cohesive regions (size≤50, dominant_type≥0.8): {len(tight)}")
        for r in sorted(tight, key=lambda r: -r["metrics"]["dominant_type_frac"])[:10]:
            top5 = "  ".join(repr(t["token"]) for t in r["tokens"][:5])
            print(f"  id={r['id']:5d}  size={r['size']:3d}  "
                  f"label={r['label']:30s}  {top5}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    ana_dir = os.path.join(out_dir, "region_token_analysis")
    Path(ana_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading artifacts from {out_dir} …")
    arts = load_artifacts(out_dir)

    print(f"Loading tokenizer {args.model_name} …")
    tokenizer = load_tokenizer(args.model_name)

    top_ids    = arts["top_ids"]
    tok_freqs  = arts["token_freqs"]
    V = len(top_ids)
    print(f"Vocab subset: {V:,} tokens")

    # ── Leaf regions ────────────────────────────────────────────────────────
    print(f"Building leaf region records …")
    leaf_records = build_region_records(
        arts["leaf_region_to_tokens"],
        top_ids, tok_freqs, tokenizer, args.min_size,
    )
    leaf_records = sort_records(leaf_records, args.sort_by)

    write_text_report(
        leaf_records,
        os.path.join(ana_dir, "region_token_report.txt"),
        args.top_n,
        title="Leaf Region Token Report",
    )
    write_json(
        leaf_records,
        os.path.join(ana_dir, "region_tokens.json"),
    )
    write_stats_csv(
        leaf_records,
        os.path.join(ana_dir, "region_stats.csv"),
    )

    # ── Coarse regions ───────────────────────────────────────────────────────
    coarse_r2t = coarse_regions_from_tree(arts["region_tree"])
    if coarse_r2t:
        print(f"Building coarse region records ({len(coarse_r2t)} coarse regions) …")
        coarse_records = build_region_records(
            coarse_r2t, top_ids, tok_freqs, tokenizer, args.min_size,
        )
        coarse_records = sort_records(coarse_records, args.sort_by)

        write_text_report(
            coarse_records,
            os.path.join(ana_dir, "coarse_token_report.txt"),
            args.top_n,
            title="Coarse Region Token Report",
        )
        write_stats_csv(
            coarse_records,
            os.path.join(ana_dir, "coarse_region_stats.csv"),
        )
        print_summary(coarse_records, "COARSE REGIONS", top_n_regions=10)

    # ── Console summary ──────────────────────────────────────────────────────
    print_summary(leaf_records, "LEAF REGIONS", top_n_regions=20)

    # ── Quick vocab coverage check ───────────────────────────────────────────
    covered = sum(r["size"] for r in leaf_records)
    print(f"\nVocab coverage: {covered:,}/{V:,} ({100*covered/V:.1f}%)")

    print(f"\nAll output written to {ana_dir}/")


if __name__ == "__main__":
    main()
