#!/usr/bin/env python3
"""
evaluate_regions.py

Five evaluation dimensions for token competition regions:

A. Graph quality      — modularity, intra/inter, conductance
B. Router probes      — linear probe: h_layer → region_id, per layer
C. Logit locality     — top-K predicted tokens concentrated in same region?
D. Gold recall        — given top-r predicted regions, does candidate set contain gold?
E. Routed softmax     — NLL/PPL with region-masked logits vs full softmax

Outputs (output_dir/eval/):
    probe_metrics.csv
    logit_locality.csv
    gold_recall.csv
    routed_softmax.csv
    summary.md
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
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_logging, set_seed, ensure_dirs,
    load_json, save_json,
    fit_probe, probe_predict_topk,
)

log = setup_logging("evaluate_regions")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_probe_cache(cache_dir: str):
    meta = load_json(os.path.join(cache_dir, "meta.json"))
    layers = meta["layers"]
    hidden = {l: np.load(os.path.join(cache_dir, f"hidden_L{l:02d}.npy")) for l in layers}
    probe_token_ids = np.load(os.path.join(cache_dir, "probe_token_ids.npy"))
    topk_ids        = np.load(os.path.join(cache_dir, "topk_ids.npy"))
    return hidden, probe_token_ids, topk_ids, layers, meta


def build_probe_labels(probe_token_ids: np.ndarray, token_to_region: Dict[str, int]) -> np.ndarray:
    """Map gold next-token vocab IDs to compact region labels [0..K-1]."""
    labels = np.array(
        [token_to_region.get(str(int(t)), -1) for t in probe_token_ids],
        dtype=np.int32,
    )
    return labels


def compact_labels(labels: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Remap arbitrary label IDs to [0..K-1] and return:
      (compact_labels, n_classes, valid_mask)
    """
    valid = labels >= 0
    present = np.unique(labels[valid])
    remap = np.full(int(labels.max()) + 2, -1, dtype=np.int32)
    for new_id, old_id in enumerate(present):
        remap[old_id] = new_id
    out = np.where(valid, remap[np.clip(labels, 0, len(remap) - 1)], -1)
    return out, len(present), valid


def build_region_token_sets(region_to_tokens: Dict[str, List[int]]) -> Dict[int, List[int]]:
    return {int(k): v for k, v in region_to_tokens.items()}


# ── A. Graph quality ──────────────────────────────────────────────────────────

def eval_graph_quality(output_dir: str) -> dict:
    metrics_path = os.path.join(output_dir, "best_partition_metrics.json")
    if os.path.exists(metrics_path):
        return load_json(metrics_path)
    return {}


# ── B. Router probes ──────────────────────────────────────────────────────────

def eval_router_probes(
    hidden: Dict[int, np.ndarray],
    probe_token_ids: np.ndarray,
    token_to_region: Dict[str, int],
    layers: List[int],
    device: torch.device,
    rng: np.random.Generator,
    epochs: int = 200,
) -> List[dict]:
    log.info("=== B. Router probes ===")
    raw_labels = build_probe_labels(probe_token_ids, token_to_region)
    labels, n_classes, valid = compact_labels(raw_labels)

    if n_classes < 2:
        log.warning("Fewer than 2 classes — skipping probe training")
        return []

    # 70/30 split over valid positions
    valid_idx = np.where(valid)[0]
    perm = rng.permutation(len(valid_idx))
    n_tr = int(0.7 * len(perm))
    tr_idx = valid_idx[perm[:n_tr]]
    te_idx = valid_idx[perm[n_tr:]]
    y_tr = labels[tr_idx]
    y_te = labels[te_idx]

    results = []
    for l in layers:
        h = hidden[l].astype(np.float32)
        X_tr = h[tr_idx]; X_te = h[te_idx]
        log.info("  Layer %2d  n_tr=%d  n_te=%d  n_classes=%d", l, len(X_tr), len(X_te), n_classes)
        m, probe = fit_probe(X_tr, y_tr, X_te, y_te, n_classes, device, epochs=epochs)
        m["layer"] = l
        log.info("    top1=%.3f  top2=%.3f  top4=%.3f  CE=%.3f bits  (random=%.3f)",
                 m["top1"], m["top2"], m["top4"], m["ce_bits"], m["random_top1"])
        results.append(m)

    return results


# ── C. Logit locality ─────────────────────────────────────────────────────────

def eval_logit_locality(
    topk_ids: np.ndarray,
    probe_token_ids: np.ndarray,
    token_to_region: Dict[str, int],
    region_to_tokens: Dict[str, List[int]],
    ks: List[int] = (10, 50, 100, 500),
) -> List[dict]:
    """
    For each position, look at the top-K predicted tokens.
    Measure what fraction land in the same region as the top-1 prediction.
    Also measure region entropy of the top-K distribution.
    """
    log.info("=== C. Logit locality ===")
    T = len(topk_ids)
    n_regions = max(int(v) for v in token_to_region.values()) + 1

    # Precompute region for each vocab ID in the subset we care about
    def region_of(vocab_id: int) -> int:
        return token_to_region.get(str(vocab_id), -1)

    results = []
    for K in ks:
        K = min(K, topk_ids.shape[1])
        same_top1_fracs  = []
        same_gold_fracs  = []
        n_unique_regions = []
        region_entropies = []

        for i in range(T):
            top1_vid   = int(topk_ids[i, 0])
            gold_vid   = int(probe_token_ids[i])
            top1_region = region_of(top1_vid)
            gold_region = region_of(gold_vid)
            topk_vids  = topk_ids[i, :K]
            topk_regions = np.array([region_of(int(v)) for v in topk_vids], dtype=np.int32)
            valid = topk_regions >= 0

            if not valid.any():
                continue

            valid_r = topk_regions[valid]
            counts  = np.bincount(valid_r, minlength=n_regions)
            pk      = counts / counts.sum()
            nz      = pk[pk > 0]
            H       = float(-np.sum(nz * np.log(nz + 1e-12)) / np.log(2))   # bits

            same_top1_fracs.append(float((valid_r == top1_region).mean()) if top1_region >= 0 else float("nan"))
            same_gold_fracs.append(float((valid_r == gold_region).mean()) if gold_region >= 0 else float("nan"))
            n_unique_regions.append(len(np.unique(valid_r)))
            region_entropies.append(H)

        results.append({
            "K":                  K,
            "same_top1_frac":     float(np.nanmean(same_top1_fracs)),
            "same_gold_frac":     float(np.nanmean(same_gold_fracs)),
            "avg_unique_regions": float(np.mean(n_unique_regions)),
            "avg_region_entropy": float(np.mean(region_entropies)),
        })
        log.info("  K=%4d  same_top1=%.3f  same_gold=%.3f  unique_regions=%.1f  H=%.3f bits",
                 K, results[-1]["same_top1_frac"], results[-1]["same_gold_frac"],
                 results[-1]["avg_unique_regions"], results[-1]["avg_region_entropy"])

    return results


# ── D. Gold recall under region routing ───────────────────────────────────────

def eval_gold_recall(
    hidden: Dict[int, np.ndarray],
    probe_token_ids: np.ndarray,
    token_to_region: Dict[str, int],
    region_token_sets: Dict[int, List[int]],
    layers: List[int],
    device: torch.device,
    rng: np.random.Generator,
    r_values: List[int] = (1, 2, 4, 8),
    epochs: int = 200,
) -> List[dict]:
    """
    For each layer, train a router probe then measure:
    - recall@r: gold token in top-r predicted regions
    - avg candidate set size
    - oracle recall (using true gold region)
    - random-region baseline
    """
    log.info("=== D. Gold recall ===")
    raw_labels = build_probe_labels(probe_token_ids, token_to_region)
    labels, n_classes, valid = compact_labels(raw_labels)
    if n_classes < 2:
        return []

    n_regions = n_classes
    # Build compact_region → [vocab_ids] mapping
    label_to_region_orig: Dict[int, int] = {}
    present = np.unique(raw_labels[raw_labels >= 0])
    for new_id, old_id in enumerate(present):
        label_to_region_orig[new_id] = int(old_id)

    valid_idx = np.where(valid)[0]
    perm = rng.permutation(len(valid_idx))
    n_tr = int(0.7 * len(perm))
    tr_idx = valid_idx[perm[:n_tr]]
    te_idx = valid_idx[perm[n_tr:]]

    # Oracle: recall when we always select the true gold region
    gold_te_compact = labels[te_idx]  # compact region IDs
    oracle_recall_per_r = {}
    for r in r_values:
        # Oracle gives gold region; recall = 1.0 if gold is in region (always true)
        oracle_recall_per_r[r] = 1.0   # by definition

    # Random baseline: sample r random regions uniformly
    all_results = []
    use_layer = layers[-1]   # eval gold recall at final layer only (most predictive)
    log.info("  Gold recall evaluated at final layer: %d", use_layer)

    h = hidden[use_layer].astype(np.float32)
    X_tr = h[tr_idx]; X_te = h[te_idx]
    y_tr = labels[tr_idx]; y_te = labels[te_idx]

    _, probe = fit_probe(X_tr, y_tr, X_te, y_te, n_classes, device, epochs=epochs)
    pred_topk = probe_predict_topk(probe, X_te, max(r_values), device)  # (N_te, max_r)

    # Build compact_region_id → set of vocab IDs
    cr_to_vocab: Dict[int, set] = {}
    for new_id, old_id in label_to_region_orig.items():
        vids = set(region_token_sets.get(old_id, []))
        cr_to_vocab[new_id] = vids

    gold_vocab_te = probe_token_ids[te_idx]

    for r in r_values:
        recalls, cand_sizes, random_recalls = [], [], []
        for i in range(len(te_idx)):
            gold_vid    = int(gold_vocab_te[i])
            pred_regions = pred_topk[i, :r].tolist()
            cand = set()
            for cr in pred_regions:
                cand |= cr_to_vocab.get(int(cr), set())
            recalls.append(int(gold_vid in cand))
            cand_sizes.append(len(cand))

            # Random baseline
            rand_regions = rng.choice(n_regions, size=min(r, n_regions), replace=False)
            rand_cand = set()
            for cr in rand_regions:
                rand_cand |= cr_to_vocab.get(int(cr), set())
            random_recalls.append(int(gold_vid in rand_cand))

        total_vocab = sum(len(v) for v in cr_to_vocab.values())
        all_results.append({
            "r_regions":            r,
            "gold_recall":          float(np.mean(recalls)),
            "avg_cand_size":        float(np.mean(cand_sizes)),
            "cand_frac_vocab":      float(np.mean(cand_sizes)) / max(total_vocab, 1),
            "oracle_recall":        oracle_recall_per_r[r],
            "random_recall":        float(np.mean(random_recalls)),
        })
        log.info("  r=%d  recall=%.3f  cand_size=%.0f (%.2f%% vocab)  random=%.3f",
                 r, all_results[-1]["gold_recall"],
                 all_results[-1]["avg_cand_size"],
                 100 * all_results[-1]["cand_frac_vocab"],
                 all_results[-1]["random_recall"])

    return all_results


# ── E. Routed softmax simulation ──────────────────────────────────────────────

def eval_routed_softmax(
    hidden_final: np.ndarray,
    probe_token_ids: np.ndarray,
    token_to_region: Dict[str, int],
    region_token_sets: Dict[int, List[int]],
    freq_ids: np.ndarray,
    model_name: str,
    device: torch.device,
    rng: np.random.Generator,
    n_eval: int = 5_000,
    r_values: List[int] = (1, 2, 4),
    epochs: int = 200,
) -> List[dict]:
    """
    Apply LM head to cached final-layer hidden states to get full logits.
    No second forward pass needed — weight-tied head: logits = ln_f(h) @ wte.T.
    """
    log.info("=== E. Routed softmax simulation ===")

    # Load LM head (ln_f + wte)
    log.info("  Loading LM head from %s ...", model_name)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    lm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    ln_f     = lm.transformer.ln_f.eval().float().to(device)
    lm_weight = lm.transformer.wte.weight.detach().float().to(device)  # (vocab, D)
    del lm
    torch.cuda.empty_cache()

    # Probe labels
    raw_labels = build_probe_labels(probe_token_ids, token_to_region)
    labels_full, n_classes, valid_full = compact_labels(raw_labels)
    if n_classes < 2:
        return []

    valid_idx = np.where(valid_full)[0]
    present = np.unique(raw_labels[raw_labels >= 0])

    # 70/30 split; use test set for routed softmax eval
    perm   = rng.permutation(len(valid_idx))
    n_tr   = int(0.7 * len(perm))
    tr_idx = valid_idx[perm[:n_tr]]
    te_idx = valid_idx[perm[n_tr:]]

    # Subsample eval set
    n_eval = min(n_eval, len(te_idx))
    ev_idx = te_idx[rng.choice(len(te_idx), n_eval, replace=False)]

    # Train probe on train set
    H = hidden_final.astype(np.float32)
    y_tr = labels_full[tr_idx]; y_te = labels_full[ev_idx]
    log.info("  Training probe for routed softmax eval (n_tr=%d) ...", len(tr_idx))
    _, probe = fit_probe(H[tr_idx], y_tr, H[ev_idx], y_te, n_classes, device, epochs=epochs)
    pred_topk = probe_predict_topk(probe, H[ev_idx], max(r_values), device)  # (N_ev, max_r)

    # Build compact_region → vocab index set (within freq_ids)
    label_to_region_orig: Dict[int, int] = {}
    for new_id, old_id in enumerate(present):
        label_to_region_orig[new_id] = int(old_id)
    freq_set = set(int(f) for f in freq_ids)

    cr_to_vocab_idx: Dict[int, List[int]] = {}   # maps to indices in lm_weight
    for cr, orig_r in label_to_region_orig.items():
        vids = [v for v in region_token_sets.get(int(orig_r), []) if v < lm_weight.shape[0]]
        cr_to_vocab_idx[cr] = vids

    # Apply LM head in batches
    log.info("  Applying LM head to %d eval positions ...", n_eval)
    batch = 256
    all_logits: List[np.ndarray] = []
    for s in range(0, n_eval, batch):
        h_b = torch.from_numpy(H[ev_idx[s:s+batch]]).to(device)
        with torch.no_grad():
            h_ln = ln_f(h_b.float())
            logits_b = (h_ln @ lm_weight.T).cpu().float().numpy()   # (B, vocab)
        all_logits.append(logits_b)
    logits_all = np.concatenate(all_logits, axis=0)   # (N_ev, vocab)

    gold_vids = probe_token_ids[ev_idx]

    all_results = []
    for r in r_values:
        full_nlls, routed_nlls, oracle_nlls, rand_nlls = [], [], [], []
        failures = 0

        for i in range(n_eval):
            gold_vid = int(gold_vids[i])
            logit_row = logits_all[i]

            # Full softmax NLL
            full_log_probs = torch.from_numpy(logit_row).log_softmax(dim=0).numpy()
            if gold_vid < len(full_log_probs):
                full_nlls.append(-float(full_log_probs[gold_vid]))
            else:
                full_nlls.append(float("nan"))

            def masked_nll(candidate_vids):
                if not candidate_vids:
                    return float("inf")
                cand = np.array(candidate_vids, dtype=np.int64)
                cand = cand[cand < len(logit_row)]
                if len(cand) == 0:
                    return float("inf")
                masked = np.full(len(logit_row), -1e9, dtype=np.float32)
                masked[cand] = logit_row[cand]
                lp = torch.from_numpy(masked).log_softmax(dim=0).numpy()
                if gold_vid < len(lp) and masked[gold_vid] > -1e8:
                    return -float(lp[gold_vid])
                return float("inf")

            # Predicted regions
            pred_cand = []
            for cr in pred_topk[i, :r].tolist():
                pred_cand.extend(cr_to_vocab_idx.get(int(cr), []))
            r_nll = masked_nll(pred_cand)
            if r_nll == float("inf"):
                failures += 1
            routed_nlls.append(r_nll)

            # Oracle: use true gold region
            gold_region_orig = token_to_region.get(str(gold_vid))
            if gold_region_orig is not None:
                oracle_cand = region_token_sets.get(gold_region_orig, [])
            else:
                oracle_cand = [gold_vid]
            oracle_nlls.append(masked_nll(oracle_cand))

            # Random regions
            rand_rs = rng.choice(n_classes, size=min(r, n_classes), replace=False)
            rand_cand = []
            for cr in rand_rs:
                rand_cand.extend(cr_to_vocab_idx.get(int(cr), []))
            rand_nlls.append(masked_nll(rand_cand))

        def safe_mean(v):
            finite = [x for x in v if x != float("inf") and not np.isnan(x)]
            return float(np.mean(finite)) if finite else float("nan")

        full_nll = safe_mean(full_nlls)
        row = {
            "r_regions":       r,
            "full_nll":        full_nll,
            "full_ppl":        float(np.exp(full_nll)) if not np.isnan(full_nll) else float("nan"),
            "routed_nll":      safe_mean(routed_nlls),
            "routed_ppl":      float(np.exp(safe_mean(routed_nlls))),
            "oracle_nll":      safe_mean(oracle_nlls),
            "oracle_ppl":      float(np.exp(safe_mean(oracle_nlls))),
            "random_nll":      safe_mean(rand_nlls),
            "nll_delta":       safe_mean(routed_nlls) - full_nll,
            "failure_rate":    failures / max(n_eval, 1),
        }
        log.info("  r=%d  full_ppl=%.1f  routed_ppl=%.1f  nll_delta=%.3f  failures=%.1f%%",
                 r, row["full_ppl"], row["routed_ppl"], row["nll_delta"], 100 * row["failure_rate"])
        all_results.append(row)

    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_plots(
    probe_results: List[dict],
    locality_results: List[dict],
    recall_results: List[dict],
    routed_results: List[dict],
    eval_dir: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return

    # Router recall vs layer
    if probe_results:
        layers = [r["layer"] for r in probe_results]
        xlabels = [str(l) for l in layers]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xlabels, [r["top1"] for r in probe_results], "o-", label="top-1", lw=2)
        ax.plot(xlabels, [r["top2"] for r in probe_results], "s--", label="top-2", lw=2)
        ax.plot(xlabels, [r["top4"] for r in probe_results], "^:", label="top-4", lw=2)
        ax.axhline(probe_results[0]["random_top1"], ls="--", color="gray", lw=1, label="random")
        ax.set_xlabel("Layer"); ax.set_ylabel("Accuracy")
        ax.set_title("Router probe accuracy vs layer")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, "router_recall_vs_layer.png"), dpi=150)
        plt.close(fig)

    # Logit locality
    if locality_results:
        ks     = [r["K"] for r in locality_results]
        same1  = [r["same_top1_frac"] for r in locality_results]
        same_g = [r["same_gold_frac"] for r in locality_results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, same1,  "o-", label="same region as top-1", lw=2)
        ax.plot(ks, same_g, "s--", label="same region as gold", lw=2)
        ax.set_xscale("log"); ax.set_xlabel("K (top-K predictions)")
        ax.set_ylabel("Fraction in same region")
        ax.set_title("Logit locality vs K")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, "logit_locality_vs_K.png"), dpi=150)
        plt.close(fig)

    # Gold recall vs candidate size
    if recall_results:
        cand_sizes = [r["avg_cand_size"] for r in recall_results]
        recalls    = [r["gold_recall"]   for r in recall_results]
        rand_r     = [r["random_recall"] for r in recall_results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(cand_sizes, recalls, "o-", label="predicted regions", lw=2)
        ax.plot(cand_sizes, rand_r,  "s--", color="gray", label="random regions", lw=1.5)
        ax.set_xlabel("Avg candidate set size")
        ax.set_ylabel("Gold recall")
        ax.set_title("Gold recall vs candidate set size")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, "gold_recall_vs_cand_size.png"), dpi=150)
        plt.close(fig)

    # Routed softmax PPL delta
    if routed_results:
        r_vals   = [r["r_regions"]  for r in routed_results]
        deltas   = [r["nll_delta"]  for r in routed_results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([str(r) for r in r_vals], deltas, color="tab:orange")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("r (top-r regions in candidate set)")
        ax.set_ylabel("NLL delta (routed - full)")
        ax.set_title("Routed softmax NLL delta")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, "routed_softmax_nll_delta.png"), dpi=150)
        plt.close(fig)

    log.info("Plots saved to %s/", eval_dir)


# ── Summary markdown ──────────────────────────────────────────────────────────

def write_summary(
    graph_metrics: dict,
    probe_results: List[dict],
    locality_results: List[dict],
    recall_results: List[dict],
    routed_results: List[dict],
    eval_dir: str,
) -> None:
    lines = ["# Interference Region Pipeline — Evaluation Summary\n"]

    def q(key, d, fmt=".3f"):
        v = d.get(key, "N/A")
        return f"{v:{fmt}}" if isinstance(v, float) else str(v)

    # 1
    Q = graph_metrics.get("modularity", "N/A")
    ratio = graph_metrics.get("intra_inter_ratio", "N/A")
    n_cl = graph_metrics.get("n_clusters", "N/A")
    lines.append(f"## 1. Do predictive interference regions exist?")
    lines.append(f"Modularity Q = {Q}  |  intra/inter = {ratio}  |  n_regions = {n_cl}")
    lines.append("Strong support if Q >> random baseline and intra/inter > 1.\n")

    # 3 + 4
    if probe_results:
        last = probe_results[-1]
        lines.append(f"## 3. Are regions predictable from hidden states?")
        lines.append(f"Final-layer router probe: top-1={last['top1']:.3f}  top-4={last['top4']:.3f}  "
                     f"(random={last['random_top1']:.3f})")
        lines.append("Strong support if top-4 > 0.95.\n")

    # 5
    if recall_results:
        lines.append(f"## 5. How small can candidate sets get while preserving gold recall?")
        for r in recall_results:
            lines.append(f"  r={r['r_regions']}  recall={r['gold_recall']:.3f}  "
                         f"cand={r['avg_cand_size']:.0f}  ({100*r['cand_frac_vocab']:.1f}% vocab subset)")
        lines.append("")

    # 6
    if locality_results:
        lines.append(f"## 6. Do regions structure final logits?")
        for r in locality_results:
            lines.append(f"  K={r['K']}  same_top1={r['same_top1_frac']:.3f}  "
                         f"same_gold={r['same_gold_frac']:.3f}")
        lines.append("")

    # 7
    if routed_results:
        lines.append(f"## 7. Does routed softmax preserve perplexity?")
        for r in routed_results:
            lines.append(f"  r={r['r_regions']}  full_ppl={r['full_ppl']:.1f}  "
                         f"routed_ppl={r['routed_ppl']:.1f}  "
                         f"delta_nll={r['nll_delta']:+.3f}  "
                         f"failures={100*r['failure_rate']:.1f}%")
        lines.append("")

    # Go / no-go
    lines.append("## Go / No-Go\n")
    go_signals = []
    if isinstance(Q, float) and Q > 0.1:
        go_signals.append(f"✓ Modularity Q={Q:.3f} > 0.10")
    if probe_results and probe_results[-1]["top4"] > 0.90:
        go_signals.append(f"✓ Router top-4 recall = {probe_results[-1]['top4']:.3f} > 0.90")
    if recall_results:
        best_recall = max(r["gold_recall"] for r in recall_results)
        if best_recall > 0.80:
            go_signals.append(f"✓ Gold recall = {best_recall:.3f} achievable")
    if routed_results and routed_results[-1]["nll_delta"] < 0.5:
        go_signals.append(f"✓ Routed NLL delta = {routed_results[-1]['nll_delta']:.3f} < 0.5")

    if len(go_signals) >= 3:
        lines.append("**STRONG GO — cluster routing appears viable.**")
    elif len(go_signals) >= 2:
        lines.append("**WEAK GO — some evidence; tune granularity.**")
    else:
        lines.append("**NO-GO — regions do not form meaningful routing boundaries.**")
    lines.extend(go_signals)

    path = os.path.join(eval_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Summary → %s", path)

    # Print to console
    print("\n" + "\n".join(lines) + "\n")


# ── CSV saving ────────────────────────────────────────────────────────────────

def save_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args) -> None:
    set_seed(args.seed)
    eval_dir = os.path.join(args.output_dir, "eval")
    ensure_dirs(eval_dir)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load shared data
    cache_dir = os.path.join(args.output_dir, "probe_cache")
    hidden, probe_token_ids, topk_ids, layers, meta = load_probe_cache(cache_dir)
    token_to_region  = load_json(os.path.join(args.output_dir, "token_to_region.json"))
    region_to_tokens = load_json(os.path.join(args.output_dir, "region_to_tokens.json"))
    freq_ids         = np.load(os.path.join(args.graphs_dir, "frequent_token_ids.npy"))
    region_token_sets = build_region_token_sets(region_to_tokens)

    log.info("N_probe=%d  layers=%s  n_regions=%d",
             len(probe_token_ids), layers, len(region_to_tokens))

    # A. Graph quality
    log.info("=== A. Graph quality ===")
    graph_metrics = eval_graph_quality(args.output_dir)
    for k, v in graph_metrics.items():
        log.info("  %s = %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    # B. Router probes
    probe_results = eval_router_probes(
        hidden, probe_token_ids, token_to_region, layers,
        device, rng, epochs=args.probe_epochs,
    )
    save_csv(probe_results, os.path.join(eval_dir, "probe_metrics.csv"))

    # C. Logit locality
    locality_results = eval_logit_locality(
        topk_ids, probe_token_ids, token_to_region, region_to_tokens,
        ks=args.logit_ks,
    )
    save_csv(locality_results, os.path.join(eval_dir, "logit_locality.csv"))

    # D. Gold recall
    recall_results = eval_gold_recall(
        hidden, probe_token_ids, token_to_region, region_token_sets,
        layers, device, rng,
        r_values=args.r_values, epochs=args.probe_epochs,
    )
    save_csv(recall_results, os.path.join(eval_dir, "gold_recall.csv"))

    # E. Routed softmax
    routed_results = eval_routed_softmax(
        hidden[layers[-1]], probe_token_ids, token_to_region, region_token_sets,
        freq_ids, args.model_name, device, rng,
        n_eval=args.n_routed_eval, r_values=args.r_values, epochs=args.probe_epochs,
    )
    save_csv(routed_results, os.path.join(eval_dir, "routed_softmax.csv"))

    # Plots + summary
    save_plots(probe_results, locality_results, recall_results, routed_results, eval_dir)
    write_summary(graph_metrics, probe_results, locality_results, recall_results, routed_results, eval_dir)
    log.info("Evaluation complete → %s/", eval_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate token competition regions",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output_dir",       default="interference_region_pipeline/results")
    p.add_argument("--graphs_dir",       default="interference_region_pipeline/graphs")
    p.add_argument("--model_name",       default="gpt2-xl")
    p.add_argument("--probe_epochs",     type=int,   default=200)
    p.add_argument("--logit_ks",         type=int,   nargs="+", default=[10, 50, 100, 500])
    p.add_argument("--r_values",         type=int,   nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n_routed_eval",    type=int,   default=5_000)
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
