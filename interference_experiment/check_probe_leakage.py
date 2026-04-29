#!/usr/bin/env python3
"""
check_probe_leakage.py

Diagnoses whether the layerwise probe results are explained by trivial baselines:

  1. Marginal baseline   — always predict the top-K most common clusters
  2. Bigram baseline     — predict next cluster from current cluster transition stats
  3. Current-token baseline — linear probe on layer-0 hidden states predicting
                             cluster(current token) rather than cluster(next token)

If the marginal or bigram baseline already hits ~98% top-4, the probe results
are trivially explained by corpus statistics, not transformer computation.

Usage (no GPU or model needed — reads from the probe cache):
    python check_probe_leakage.py \\
        --cache_dir  interference_experiment/results/probe_cache \\
        --clusters   interference_experiment/results/cluster_tokens_coactivation_final.json \\
        --labels     interference_experiment/graphs/layer_final_coactivation_labels_louvain.npy \\
        --freq_ids   interference_experiment/graphs/layer_final_frequent_token_ids.npy
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def topk_recall_from_probs(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def topk_recall_from_fixed_set(predicted_set: np.ndarray, labels: np.ndarray) -> float:
    """Recall when we always predict the same fixed set of clusters."""
    return float(np.isin(labels, predicted_set).mean())


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Baseline 1: Marginal (always predict top-K most common clusters)
# ---------------------------------------------------------------------------

def marginal_baseline(clusters: np.ndarray, n_classes: int) -> dict:
    """
    If we always predict the K most frequent clusters (regardless of context),
    what is our top-K recall?

    This is the absolute floor: a probe that ignores hidden states entirely.
    """
    counts = np.bincount(clusters.astype(np.int64), minlength=n_classes)
    freq_order = np.argsort(counts)[::-1]  # most frequent first
    dist = counts / counts.sum()

    results = {}
    for k in [1, 2, 4]:
        top_k_clusters = freq_order[:k]
        recall = topk_recall_from_fixed_set(top_k_clusters, clusters)
        results[f"top{k}"] = recall

    print_section("BASELINE 1: Marginal (always predict most-common clusters)")
    print(f"  Cluster frequency distribution:")
    for c in freq_order:
        print(f"    Cluster {c:2d}: {counts[c]:7d} ({100*dist[c]:.1f}%)")
    print()
    print(f"  Always predicting top-1 cluster:  top-1 recall = {results['top1']:.3f}")
    print(f"  Always predicting top-2 clusters: top-2 recall = {results['top2']:.3f}")
    print(f"  Always predicting top-4 clusters: top-4 recall = {results['top4']:.3f}")
    print()
    if results["top4"] > 0.90:
        print("  ⚠  TOP-4 MARGINAL BASELINE > 90% — the label distribution is so")
        print("     skewed that top-4 recall is nearly trivial regardless of the probe.")
    else:
        print(f"  Top-4 marginal baseline = {results['top4']:.3f}. "
              f"Probe must beat this to be informative.")
    return results


# ---------------------------------------------------------------------------
# Baseline 2: Bigram (predict next cluster from current cluster)
# ---------------------------------------------------------------------------

def bigram_baseline(clusters: np.ndarray, n_classes: int) -> dict:
    """
    Use consecutive entries in `clusters` to approximate the bigram transition
    distribution: P(next_cluster | current_cluster).

    clusters[j]   = cluster of the NEXT token at position j
    clusters[j-1] ≈ cluster of the CURRENT token at position j  (approximate;
                    breaks at window boundaries and skipped positions)

    Even with ~15% broken pairs this gives a good approximation.
    """
    # Build transition matrix C[i, j] = count(current=i, next=j)
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    curr = clusters[:-1].astype(np.int64)   # approximate current cluster
    nxt  = clusters[1:].astype(np.int64)    # next cluster
    valid = (curr >= 0) & (nxt >= 0)
    np.add.at(C, (curr[valid], nxt[valid]), 1)

    # Row-normalise to get probabilities
    row_sums = C.sum(axis=1, keepdims=True).clip(1, None)
    P = C / row_sums  # (n_classes, n_classes)

    # For each current cluster, rank next-cluster predictions by probability
    # Evaluate on the (approximate) pairs
    curr_eval = clusters[:-1][valid]
    nxt_eval  = clusters[1:][valid]

    results = {}
    for k in [1, 2, 4]:
        probs = P[curr_eval]          # (N, n_classes) — look up transition row
        r = topk_recall_from_probs(probs, nxt_eval, k)
        results[f"top{k}"] = r

    print_section("BASELINE 2: Bigram (predict next cluster from current cluster)")
    print(f"  Transition matrix P(next | current):")
    header = "          " + " ".join(f"  C{j}" for j in range(n_classes))
    print(header)
    for i in range(n_classes):
        row = " ".join(f"{P[i,j]:5.2f}" for j in range(n_classes))
        print(f"    C{i} → [ {row} ]")
    print()
    print(f"  N valid bigram pairs (approx): {valid.sum():,}")
    print(f"  Bigram top-1 recall: {results['top1']:.3f}")
    print(f"  Bigram top-2 recall: {results['top2']:.3f}")
    print(f"  Bigram top-4 recall: {results['top4']:.3f}")
    print()
    if results["top4"] > 0.90:
        print("  ⚠  BIGRAM TOP-4 > 90% — a probe that only sees the current")
        print("     cluster (9 bits of information) already explains the result.")
        print("     The 1600-dim probe may simply be learning bigram statistics.")
    else:
        print(f"  Bigram top-4 = {results['top4']:.3f}. "
              f"Any probe significantly above this is adding real value.")
    return results


# ---------------------------------------------------------------------------
# Baseline 3: Current-token probe (h_0 → cluster(current token))
# ---------------------------------------------------------------------------

def current_token_probe_baseline(
    h_layer0: np.ndarray,
    clusters_next: np.ndarray,
    clusters_curr_approx: np.ndarray,
    probe_device_str: str = "cpu",
) -> dict:
    """
    Trains two probes on h_j^(0):
      A) h_j^(0) → cluster(tokens[j])   (current token — should be ~100%)
      B) h_j^(0) → cluster(tokens[j+1]) (next token — what we're measuring)

    If A >> B at layer 0, hidden states mostly encode current-token identity.
    If A ≈ B, hidden states already look ahead.

    clusters_curr_approx = clusters[:-1] (approximate current cluster).
    clusters_next        = clusters[1:]  (next cluster, same positions).
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    # Align: use pairs where both are valid
    N = min(len(h_layer0) - 1, len(clusters_curr_approx))
    X   = h_layer0[:N].astype(np.float32)
    y_curr = clusters_curr_approx[:N].astype(np.int64)
    y_next = clusters_next[:N].astype(np.int64)

    valid = (y_curr >= 0) & (y_next >= 0)
    X      = X[valid]
    y_curr = y_curr[valid]
    y_next = y_next[valid]

    n_classes = int(max(y_curr.max(), y_next.max())) + 1
    n_train = int(0.7 * len(X))

    device = torch.device(probe_device_str if torch.cuda.is_available() else "cpu")

    def _train_and_eval(y_labels, label: str):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[:n_train])
        X_te = scaler.transform(X[n_train:])
        y_tr = y_labels[:n_train]
        y_te = y_labels[n_train:]

        lin = nn.Linear(X_tr.shape[1], n_classes).to(device)
        X_t = torch.from_numpy(X_tr).to(device)
        y_t = torch.from_numpy(y_tr).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(150):
            perm = torch.randperm(len(X_t), device=device)
            for i in range(0, len(X_t), 2048):
                idx = perm[i:i+2048]
                opt.zero_grad()
                loss_fn(lin(X_t[idx]), y_t[idx]).backward()
                opt.step()

        lin.eval()
        with torch.no_grad():
            probs_t = torch.softmax(
                lin(torch.from_numpy(X_te).to(device)), dim=-1
            ).cpu().numpy()

        t1 = float((np.argmax(probs_t, 1) == y_te).mean())
        t4 = float(np.mean([y_te[i] in np.argsort(probs_t[i])[-4:] for i in range(len(y_te))]))
        print(f"  h_0 → cluster({label}):  top-1={t1:.3f}  top-4={t4:.3f}")
        return {"top1": t1, "top4": t4}

    print_section("BASELINE 3: What does h_j^(0) actually encode?")
    print(f"  N valid pairs: {valid.sum():,}")
    print()
    r_curr = _train_and_eval(y_curr, "current token")
    r_next = _train_and_eval(y_next, "next token   ")
    print()
    gap = r_next["top4"] - r_curr["top4"]
    if r_curr["top1"] > 0.95:
        print("  ✓  h_0 encodes current token cluster with high fidelity (>95% top-1).")
        print("     This confirms layer-0 is essentially the input embedding.")
    if abs(gap) < 0.05:
        print("  ⚠  top-4 recall for CURRENT and NEXT cluster are nearly equal.")
        print("     The probe is likely learning bigram statistics from current-token identity,")
        print("     NOT genuine look-ahead computation by the transformer.")
    else:
        print(f"  Next-token probe top-4 = {r_next['top4']:.3f}, "
              f"current-token probe top-4 = {r_curr['top4']:.3f}  (gap={gap:+.3f})")
    return {"current": r_curr, "next": r_next}


# ---------------------------------------------------------------------------
# Summary verdict
# ---------------------------------------------------------------------------

def verdict(
    marginal: dict,
    bigram: dict,
    probe_top4_layer0: float,
    n_classes: int,
) -> None:
    print_section("VERDICT")
    random_top4 = min(4 / n_classes, 1.0)
    print(f"  Random top-4 baseline:   {random_top4:.3f}  (4/{n_classes} classes)")
    print(f"  Marginal top-4 baseline: {marginal['top4']:.3f}  (always predict most-common clusters)")
    print(f"  Bigram top-4 baseline:   {bigram['top4']:.3f}  (current-cluster → next-cluster stats)")
    print(f"  Layer-0 probe top-4:     {probe_top4_layer0:.3f}  (from main experiment)")
    print()

    leakage_gap = probe_top4_layer0 - bigram["top4"]
    print(f"  Layer-0 probe excess over bigram baseline: {leakage_gap:+.3f}")

    if marginal["top4"] > 0.90:
        print()
        print("  *** CRITICAL: Marginal baseline already > 90% top-4. ***")
        print("      The label distribution is so skewed that top-4 recall is")
        print("      nearly trivial. The probe results do NOT demonstrate that")
        print("      the transformer is doing useful computation.")
        print("      FIX: Use top-1 accuracy or balanced evaluation instead.")

    elif bigram["top4"] > 0.90:
        print()
        print("  *** WARNING: Bigram baseline already > 90% top-4. ***")
        print("      Knowing only the CURRENT token's cluster (9 bits) is enough")
        print("      to predict the next cluster with >90% top-4 recall.")
        print("      The layer-0 result is largely trivial.")
        print()
        print("  CORRECT INTERPRETATION: Compare each layer's probe to the bigram")
        print("  baseline. Ask: 'At which layer does the probe SIGNIFICANTLY EXCEED")
        print("  the bigram baseline?' That is the real emergence signal.")

    elif leakage_gap < 0.05:
        print()
        print("  WARNING: Layer-0 probe barely exceeds bigram baseline (+{:.3f}).".format(leakage_gap))
        print("  Most of layer-0's performance is explained by bigram statistics.")

    else:
        print()
        print(f"  Layer-0 probe exceeds bigram baseline by {leakage_gap:.3f}.")
        print("  The probe is learning something beyond simple bigram statistics.")

    print()
    print("  RECOMMENDED: Re-interpret probe results as 'excess over bigram baseline'.")
    print("  Plot: (layer_top4 - bigram_top4) vs. layer. The layer where this")
    print("  first exceeds 0 significantly is where the transformer adds real value.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cache_dir", required=True,
                        help="probe_cache directory from layerwise_region_probe.py")
    parser.add_argument("--clusters", required=True,
                        help="cluster_tokens JSON")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--freq_ids", default=None)
    parser.add_argument("--layer0_top4", type=float, default=None,
                        help="Top-4 recall at layer 0 from the main experiment "
                             "(read from metrics.json if not provided)")
    parser.add_argument("--metrics_json", default=None,
                        help="metrics.json from the main experiment")
    parser.add_argument("--skip_current_token_probe", action="store_true",
                        help="Skip the GPU-based current-token probe (baseline 3)")
    args = parser.parse_args()

    # Load cache
    meta_path = os.path.join(args.cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: cache not found at {args.cache_dir}", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    layers     = meta["layers"]
    clusters   = np.load(os.path.join(args.cache_dir, "clusters.npy"))
    n_classes  = int(clusters.max()) + 1

    print(f"Loaded cache: {len(clusters):,} samples  {n_classes} clusters  layers={layers}")

    # Get layer-0 probe top-4 from metrics.json
    probe_top4_layer0 = args.layer0_top4
    if probe_top4_layer0 is None and args.metrics_json:
        with open(args.metrics_json) as f:
            m = json.load(f)
        probe_top4_layer0 = m.get("linear_probe", {}).get("0", {}).get("top4", None)
    if probe_top4_layer0 is None:
        # Try to find metrics.json next to cache
        candidate = os.path.join(os.path.dirname(args.cache_dir), "metrics.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                m = json.load(f)
            probe_top4_layer0 = m.get("linear_probe", {}).get("0", {}).get("top4", None)
    if probe_top4_layer0 is None:
        probe_top4_layer0 = 0.984   # from the log output the user pasted
        print(f"Using hardcoded layer-0 top4={probe_top4_layer0} (pass --metrics_json to auto-read)")

    # Baseline 1: Marginal
    marg = marginal_baseline(clusters, n_classes)

    # Baseline 2: Bigram
    bigr = bigram_baseline(clusters, n_classes)

    # Baseline 3: Current-token probe (optional, needs GPU)
    if not args.skip_current_token_probe:
        h0_path = os.path.join(args.cache_dir, "hidden_L00.npy")
        if os.path.exists(h0_path):
            print(f"\nLoading layer-0 hidden states for current-token probe...")
            h0 = np.load(h0_path)
            curr_approx = clusters[:-1]   # approximate cluster(current token)
            nxt         = clusters[1:]    # cluster(next token)
            current_token_probe_baseline(h0, nxt, curr_approx)
        else:
            print(f"\nLayer-0 hidden states not found at {h0_path} — skipping baseline 3.")

    # Verdict
    verdict(marg, bigr, probe_top4_layer0, n_classes)


if __name__ == "__main__":
    main()
