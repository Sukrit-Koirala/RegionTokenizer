#!/usr/bin/env python3
"""
context_vs_cluster.py

Tests whether context causes *local movement within clusters* rather than
*global movement across clusters* in GPT-2 XL hidden representations.

Expected signal if hypothesis holds:
    inter_cluster >> intra_cluster >= contextual_drift    (normalized)
    cluster_flip_rate is low
    subcluster_flip_rate > cluster_flip_rate

All metrics are computed per layer from cached hidden states (no model needed).

Definitions
-----------
contextual_drift  : For token t, mean L2 distance of each occurrence from the
                    token's mean hidden state across contexts. Averaged over tokens.
                    Measures how much context moves a token within representation space.

intra_cluster     : Mean pairwise L2 distance between per-token mean vectors,
                    computed within each cluster, then averaged across clusters.
                    Measures the spread of distinct token identities within a cluster.

inter_cluster     : Mean pairwise L2 distance between cluster centroids
                    (centroid = mean of per-token means in that cluster).
                    Measures separation between clusters.

cluster_flip_rate : For each token, fraction of its occurrences where
                    nearest-centroid classification predicts a different cluster
                    than the token's true cluster. Averaged over tokens.

All distances normalized by the layer's mean hidden-state L2 norm.

Usage
-----
python interference_experiment/context_vs_cluster.py \\
    --cache_dir  interference_experiment/results/probe_cache \\
    --subcluster_map interference_experiment/results/cluster7_subcluster_mapping.json \\
    --output_dir interference_experiment/results/context_vs_cluster
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("ctx_cluster")


log = logging.getLogger("ctx_cluster")


# ─────────────────────────────────────────────────────────────────────────────
# Distance utilities
# ─────────────────────────────────────────────────────────────────────────────

def mean_pairwise_l2(vecs: torch.Tensor) -> float:
    """
    Mean L2 distance over all unique pairs in vecs of shape (m, D).
    Uses the identity ||a-b||² = ||a||²+||b||²-2(a·b) to avoid the O(m²D)
    broadcast; only builds an (m, m) matrix which is cheap for m << D.
    """
    m = len(vecs)
    if m < 2:
        return 0.0
    sq    = (vecs * vecs).sum(dim=1, keepdim=True)    # (m, 1)
    cross = vecs @ vecs.T                              # (m, m)
    sq_dist = (sq + sq.T - 2.0 * cross).clamp(min=0)  # (m, m)
    dist = sq_dist.sqrt()
    rows, cols = torch.triu_indices(m, m, offset=1, device=vecs.device)
    return float(dist[rows, cols].mean()) if rows.numel() > 0 else 0.0


def nearest_centroid_assign(h: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Assign each row of h (N, D) to its nearest centroid in (K, D).
    Returns (N,) int64 indices into the centroids tensor.
    Fully vectorized: builds (N, K) distance matrix, never (N, K, D).
    """
    h_sq  = (h * h).sum(dim=1, keepdim=True)           # (N, 1)
    c_sq  = (centroids * centroids).sum(dim=1)          # (K,)
    cross = h @ centroids.T                             # (N, K)
    sq_dists = h_sq + c_sq.unsqueeze(0) - 2.0 * cross  # (N, K)
    return sq_dists.clamp(min=0).argmin(dim=1)          # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(
    cache_dir: str,
) -> Tuple[Dict[int, str], np.ndarray, np.ndarray, List[int]]:
    """
    Returns hidden state file paths (loaded lazily), cluster labels,
    token IDs, and layer list.

    clusters[i] = coarse cluster of gold next-token at position i  (-1 = unknown)
    tokens[i]   = vocab ID of gold next-token at position i
    """
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        log.error("meta.json not found in %s", cache_dir)
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    layers: List[int] = meta["layers"]
    hidden_paths: Dict[int, str] = {}
    for l in layers:
        p = os.path.join(cache_dir, f"hidden_L{l:02d}.npy")
        if not os.path.exists(p):
            log.error("Missing hidden state file: %s", p)
            sys.exit(1)
        hidden_paths[l] = p
    clusters = np.load(os.path.join(cache_dir, "clusters.npy")).astype(np.int32)
    tokens   = np.load(os.path.join(cache_dir, "tokens.npy")).astype(np.int32)
    log.info("Cache: N=%d  layers=%s", len(clusters), layers)
    return hidden_paths, clusters, tokens, layers


def load_subcluster_map(
    path: Optional[str],
    vocab_size: int = 50257,
) -> Optional[np.ndarray]:
    """
    Returns int32 array (vocab_size,) where arr[vocab_id] = subcluster_id,
    -1 if the token has no subcluster label.

    Accepts the vocab_to_subcluster format from recursive_cluster7.py,
    or a flat {token_id_str: subcluster_id} dict.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    arr = np.full(vocab_size, -1, dtype=np.int32)
    mapping = data.get("vocab_to_subcluster", data)
    for vid_str, sub_id in mapping.items():
        try:
            vid = int(vid_str)
            if 0 <= vid < vocab_size:
                arr[vid] = int(sub_id)
        except (ValueError, TypeError):
            pass
    n_mapped = int((arr >= 0).sum())
    n_sub = int(arr.max()) + 1 if n_mapped > 0 else 0
    log.info("Subcluster map: %d tokens mapped to %d subclusters", n_mapped, n_sub)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Token grouping  (computed once, reused across all layers)
# ─────────────────────────────────────────────────────────────────────────────

class TokenGroups:
    """Pre-computed grouping structures, built once from tokens/clusters arrays."""

    def __init__(
        self,
        tokens: np.ndarray,
        clusters: np.ndarray,
        subcluster_arr: Optional[np.ndarray],
        min_occurrences: int,
    ) -> None:
        N = len(tokens)
        # ── Count occurrences and record cluster per token ─────────────────
        tok_counts: Dict[int, int] = defaultdict(int)
        tok_cluster: Dict[int, int] = {}
        for i in range(N):
            tok = int(tokens[i])
            clu = int(clusters[i])
            if clu < 0:
                continue
            tok_counts[tok] += 1
            tok_cluster[tok] = clu   # constant per token; last write wins (same value)

        # ── Filter to minimum occurrences ──────────────────────────────────
        qualified = frozenset(
            t for t, cnt in tok_counts.items()
            if cnt >= min_occurrences and t in tok_cluster
        )
        log.info(
            "Tokens with ≥%d occurrences: %d / %d unique tokens in cache",
            min_occurrences, len(qualified), len(tok_counts),
        )

        # ── Build index arrays  token_id → sample indices ─────────────────
        tmp: Dict[int, List[int]] = defaultdict(list)
        for i in range(N):
            tok = int(tokens[i])
            if tok in qualified and int(clusters[i]) >= 0:
                tmp[tok].append(i)
        self.token_to_idx: Dict[int, np.ndarray] = {
            t: np.array(idxs, dtype=np.int64) for t, idxs in tmp.items()
        }

        # ── Token → cluster / cluster → tokens ────────────────────────────
        self.token_to_cluster: Dict[int, int] = {
            t: tok_cluster[t] for t in qualified
        }
        c2t: Dict[int, List[int]] = defaultdict(list)
        for t, c in self.token_to_cluster.items():
            c2t[c].append(t)
        self.cluster_to_tokens: Dict[int, List[int]] = dict(c2t)
        self.cluster_ids: List[int] = sorted(self.cluster_to_tokens.keys())

        log.info(
            "Clusters represented: %d  tokens per cluster (avg): %.1f",
            len(self.cluster_ids),
            np.mean([len(v) for v in self.cluster_to_tokens.values()]),
        )

        # ── Subcluster mapping (optional) ──────────────────────────────────
        self.token_to_sub: Optional[Dict[int, int]] = None
        self.subcluster_to_tokens: Optional[Dict[int, List[int]]] = None
        self.sub_ids: Optional[List[int]] = None

        if subcluster_arr is not None:
            tok_sub: Dict[int, int] = {}
            s2t: Dict[int, List[int]] = defaultdict(list)
            for t in qualified:
                vid = int(t)
                if 0 <= vid < len(subcluster_arr):
                    sub = int(subcluster_arr[vid])
                    if sub >= 0:
                        tok_sub[t] = sub
                        s2t[sub].append(t)
            if tok_sub:
                self.token_to_sub = tok_sub
                self.subcluster_to_tokens = dict(s2t)
                self.sub_ids = sorted(s2t.keys())
                log.info(
                    "Tokens with subcluster label: %d / %d  subclusters: %d",
                    len(tok_sub), len(qualified), len(self.sub_ids),
                )


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_layer(
    h_np: np.ndarray,          # (N, D) float16 from cache
    groups: TokenGroups,
    device: torch.device,
) -> dict:
    """
    Compute all five metrics for one layer.
    Loads h_np into GPU float32; everything else is torch.
    """
    h = torch.from_numpy(h_np.astype(np.float32)).to(device)  # (N, D)

    avg_norm = float(h.norm(dim=1).mean())
    norm_denom = max(avg_norm, 1e-8)

    # ── 1. Per-token means ───────────────────────────────────────────────────
    token_means: Dict[int, torch.Tensor] = {}
    for tok, idx in groups.token_to_idx.items():
        idx_t = torch.from_numpy(idx).to(device)
        token_means[tok] = h[idx_t].mean(dim=0)   # (D,)

    # ── 2. Contextual drift ──────────────────────────────────────────────────
    drifts: List[float] = []
    for tok, idx in groups.token_to_idx.items():
        idx_t = torch.from_numpy(idx).to(device)
        h_tok = h[idx_t]                                      # (n_t, D)
        mu    = token_means[tok].unsqueeze(0)                 # (1, D)
        drifts.append(float((h_tok - mu).norm(dim=1).mean()))
    contextual_drift_raw = float(np.mean(drifts)) if drifts else 0.0

    # ── 3. Intra-cluster distance (mean pairwise between token means) ────────
    intra_raw_list: List[float] = []
    for cid in groups.cluster_ids:
        vecs = [token_means[t] for t in groups.cluster_to_tokens[cid]
                if t in token_means]
        if len(vecs) < 2:
            continue
        intra_raw_list.append(mean_pairwise_l2(torch.stack(vecs)))
    intra_cluster_raw = float(np.mean(intra_raw_list)) if intra_raw_list else 0.0

    # ── 4. Inter-cluster distance (mean pairwise between cluster centroids) ──
    centroid_list: List[torch.Tensor] = []
    centroid_ids:  List[int] = []
    for cid in groups.cluster_ids:
        vecs = [token_means[t] for t in groups.cluster_to_tokens[cid]
                if t in token_means]
        if not vecs:
            continue
        centroid_list.append(torch.stack(vecs).mean(dim=0))
        centroid_ids.append(cid)
    centroids = torch.stack(centroid_list)             # (K, D)
    inter_cluster_raw = mean_pairwise_l2(centroids)

    # ── 5. Normalize by average hidden-state norm ────────────────────────────
    contextual_drift = contextual_drift_raw / norm_denom
    intra_cluster    = intra_cluster_raw    / norm_denom
    inter_cluster    = inter_cluster_raw    / norm_denom

    # ── 6. Cluster flip rate (nearest centroid over all N samples at once) ───
    pred_centroid_idx = nearest_centroid_assign(h, centroids)  # (N,) indices
    pred_cid_for_sample = torch.tensor(
        [centroid_ids[i] for i in pred_centroid_idx.cpu().tolist()],
        dtype=torch.long,
    )   # (N,) cluster IDs

    flip_rates: List[float] = []
    for tok, idx in groups.token_to_idx.items():
        true_c = groups.token_to_cluster[tok]
        pred_c = pred_cid_for_sample[torch.from_numpy(idx)]
        flip_rates.append(float((pred_c != true_c).float().mean()))
    cluster_flip_rate = float(np.mean(flip_rates)) if flip_rates else 0.0

    # ── 7. Subcluster flip rate ──────────────────────────────────────────────
    subcluster_flip_rate: Optional[float] = None
    if (groups.token_to_sub is not None
            and groups.subcluster_to_tokens is not None
            and groups.sub_ids is not None
            and len(groups.sub_ids) >= 2):

        # Build subcluster centroids
        sub_centroid_list: List[torch.Tensor] = []
        sub_centroid_ids:  List[int] = []
        for sid in groups.sub_ids:
            vecs = [token_means[t] for t in groups.subcluster_to_tokens[sid]
                    if t in token_means]
            if not vecs:
                continue
            sub_centroid_list.append(torch.stack(vecs).mean(dim=0))
            sub_centroid_ids.append(sid)

        if len(sub_centroid_list) >= 2:
            sub_centroids = torch.stack(sub_centroid_list)   # (M, D)

            # Tokens that have both subcluster labels and sufficient occurrences
            sub_qualified = sorted(
                t for t in groups.token_to_sub if t in groups.token_to_idx
            )
            if sub_qualified:
                # Gather all their sample indices in a consistent order
                n_per_tok = [len(groups.token_to_idx[t]) for t in sub_qualified]
                all_idx = np.concatenate(
                    [groups.token_to_idx[t] for t in sub_qualified]
                )
                h_sub = h[torch.from_numpy(all_idx).to(device)]   # (N_sub, D)

                pred_sub_cidx = nearest_centroid_assign(h_sub, sub_centroids)
                pred_sub_ids = torch.tensor(
                    [sub_centroid_ids[i] for i in pred_sub_cidx.cpu().tolist()],
                    dtype=torch.long,
                )   # (N_sub,)

                sub_flips: List[float] = []
                offset = 0
                for tok, n_t in zip(sub_qualified, n_per_tok):
                    true_sub = groups.token_to_sub[tok]
                    pred_tok = pred_sub_ids[offset : offset + n_t]
                    offset  += n_t
                    sub_flips.append(float((pred_tok != true_sub).float().mean()))
                subcluster_flip_rate = float(np.mean(sub_flips))

    return {
        "avg_norm":             round(avg_norm, 4),
        "contextual_drift":     round(contextual_drift, 6),
        "intra_cluster":        round(intra_cluster, 6),
        "inter_cluster":        round(inter_cluster, 6),
        "cluster_flip_rate":    round(cluster_flip_rate, 6),
        "subcluster_flip_rate": round(subcluster_flip_rate, 6) if subcluster_flip_rate is not None else None,
        # Raw distances (before normalization) for reference
        "contextual_drift_raw": round(contextual_drift_raw, 4),
        "intra_cluster_raw":    round(intra_cluster_raw, 4),
        "inter_cluster_raw":    round(inter_cluster_raw, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics(all_metrics: Dict[int, dict], output_dir: str) -> None:
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump({str(l): m for l, m in all_metrics.items()}, f, indent=2)
    log.info("Metrics → %s", path)


def _plot_setup():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return None


def plot_distances(all_metrics: Dict[int, dict], layers: List[int], output_dir: str) -> None:
    plt = _plot_setup()
    if plt is None:
        return

    xlabels = [str(l) for l in layers]
    drift = [all_metrics[l]["contextual_drift"] for l in layers]
    intra = [all_metrics[l]["intra_cluster"]    for l in layers]
    inter = [all_metrics[l]["inter_cluster"]    for l in layers]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xlabels, inter, "o-",  color="tab:red",    lw=2.5, label="inter-cluster (centroid separation)")
    ax.plot(xlabels, intra, "s--", color="tab:blue",   lw=2,   label="intra-cluster (token-mean spread)")
    ax.plot(xlabels, drift, "^:",  color="tab:green",  lw=1.5, label="contextual drift (same-token variance)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 distance / avg hidden norm")
    ax.set_title(
        "Distances vs layer  (normalized)\n"
        "Expect: inter-cluster  ≫  intra-cluster  ≥  contextual drift"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "distances_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Plot → %s", path)


def plot_flip_rates(all_metrics: Dict[int, dict], layers: List[int], output_dir: str) -> None:
    plt = _plot_setup()
    if plt is None:
        return

    xlabels = [str(l) for l in layers]
    c_flip = [all_metrics[l]["cluster_flip_rate"] for l in layers]
    s_flip = [all_metrics[l]["subcluster_flip_rate"] for l in layers]
    has_sub = any(v is not None for v in s_flip)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xlabels, c_flip, "o-", color="tab:blue",   lw=2, label="cluster flip rate")
    if has_sub:
        s_vals = [v if v is not None else float("nan") for v in s_flip]
        ax.plot(xlabels, s_vals, "s--", color="tab:orange", lw=2, label="subcluster flip rate")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Flip rate  (fraction of occurrences misclassified by nearest centroid)")
    ax.set_title(
        "Cluster flip rate vs layer\n"
        "Expect: cluster flip rate LOW, subcluster flip rate HIGHER"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "flip_rates_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Plot → %s", path)


def print_summary(all_metrics: Dict[int, dict], layers: List[int]) -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print("  CONTEXT vs CLUSTER — RESULTS")
    print(sep)
    print(f"  {'Layer':>6}  {'drift':>8}  {'intra':>8}  {'inter':>8}  "
          f"{'c_flip':>8}  {'sc_flip':>9}  {'inter/intra':>12}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*9}  {'-'*12}")
    for l in layers:
        m = all_metrics[l]
        ratio = (m["inter_cluster"] / m["intra_cluster"]
                 if m["intra_cluster"] > 0 else float("inf"))
        sc = f"{m['subcluster_flip_rate']:.4f}" if m["subcluster_flip_rate"] is not None else "  N/A   "
        print(f"  {l:>6}  {m['contextual_drift']:8.4f}  {m['intra_cluster']:8.4f}  "
              f"{m['inter_cluster']:8.4f}  {m['cluster_flip_rate']:8.4f}  "
              f"{sc:>9}  {ratio:12.2f}x")
    print()

    # Interpretation
    last = all_metrics[layers[-1]]
    first_inter_gt_intra = next(
        (l for l in layers if all_metrics[l]["inter_cluster"] > all_metrics[l]["intra_cluster"]),
        None,
    )
    hierarchy_holds = (
        last["inter_cluster"] > last["intra_cluster"] >= last["contextual_drift"]
    )
    flip_is_low = last["cluster_flip_rate"] < 0.15

    print("  INTERPRETATION (final layer):")
    print(f"    inter > intra >= drift hierarchy:  {'YES' if hierarchy_holds else 'NO'}")
    if first_inter_gt_intra is not None:
        print(f"    First layer where inter > intra:   Layer {first_inter_gt_intra}")
    print(f"    Cluster flip rate (final):         {last['cluster_flip_rate']:.3f}  "
          f"({'LOW — clusters are stable' if flip_is_low else 'HIGH — context crosses cluster boundaries'})")
    if last["subcluster_flip_rate"] is not None:
        sc_higher = last["subcluster_flip_rate"] > last["cluster_flip_rate"]
        print(f"    Subcluster flip rate (final):      {last['subcluster_flip_rate']:.3f}  "
              f"({'higher than cluster — expected' if sc_higher else 'not higher — unexpected'})")

    print()
    if hierarchy_holds and flip_is_low:
        print("  CONCLUSION: Context moves representations LOCALLY within clusters.")
        print("  Supports the hypothesis that clusters are geometrically coherent")
        print("  and context-robust — motivating cluster-based routing.")
    elif hierarchy_holds:
        print("  CONCLUSION: Cluster geometry is well-separated, but context sometimes")
        print("  crosses cluster boundaries (flip rate > 0.15).")
    else:
        print("  CONCLUSION: Hierarchy does NOT hold clearly at the final layer.")
        print("  Context drift may be comparable to or exceed intra-cluster spread.")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context vs cluster geometry analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache_dir",      required=True,
                        help="probe_cache dir from layerwise_region_probe.py")
    parser.add_argument("--subcluster_map", default=None,
                        help="cluster7_subcluster_mapping.json from recursive_cluster7.py")
    parser.add_argument("--output_dir",
                        default="interference_experiment/results/context_vs_cluster")
    parser.add_argument("--min_occurrences", type=int, default=20,
                        help="Minimum occurrences per token to include in analysis")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    _setup_logging()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load cache ──────────────────────────────────────────────────────────
    log.info("=== Loading cache ===")
    hidden_paths, clusters, tokens, layers = load_cache(args.cache_dir)

    # ── Load subcluster map ─────────────────────────────────────────────────
    log.info("=== Loading subcluster map ===")
    subcluster_arr = load_subcluster_map(args.subcluster_map)

    # ── Build token groups (once — reused across all layers) ────────────────
    log.info("=== Building token groups (min_occurrences=%d) ===", args.min_occurrences)
    groups = TokenGroups(tokens, clusters, subcluster_arr, args.min_occurrences)

    # ── Per-layer analysis ──────────────────────────────────────────────────
    log.info("=== Analyzing layers ===")
    all_metrics: Dict[int, dict] = {}

    for l in layers:
        t0 = time.time()
        log.info("--- Layer %d ---", l)
        h_np = np.load(hidden_paths[l])   # (N, D) float16

        metrics = analyze_layer(h_np, groups, device)
        del h_np   # free immediately

        elapsed = time.time() - t0
        sub_str = (f"{metrics['subcluster_flip_rate']:.4f}"
                   if metrics["subcluster_flip_rate"] is not None else "N/A")
        log.info(
            "  drift=%.4f  intra=%.4f  inter=%.4f  "
            "c_flip=%.4f  sc_flip=%s  ratio=%.1fx  (%.1fs)",
            metrics["contextual_drift"],
            metrics["intra_cluster"],
            metrics["inter_cluster"],
            metrics["cluster_flip_rate"],
            sub_str,
            metrics["inter_cluster"] / max(metrics["intra_cluster"], 1e-8),
            elapsed,
        )
        all_metrics[l] = metrics

    # ── Save and plot ───────────────────────────────────────────────────────
    log.info("=== Saving outputs to %s/ ===", args.output_dir)
    save_metrics(all_metrics, args.output_dir)
    plot_distances(all_metrics, layers, args.output_dir)
    plot_flip_rates(all_metrics, layers, args.output_dir)
    print_summary(all_metrics, layers)

    log.info("Done.")


if __name__ == "__main__":
    main()
