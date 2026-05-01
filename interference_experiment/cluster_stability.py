#!/usr/bin/env python3
"""
cluster_stability.py

Measures how consistently each token's representation stays within the same
cluster across different contexts, as a function of transformer layer depth.

Reference centroids are built from the FINAL layer only:
    centroid[c] = mean(h_final[cluster_ids == c]),  unit-normalized.

These centroids are fixed and reused for every layer.  The flip rate at
layer L therefore asks: "does h_L land in the correct final-layer cluster
region?" — not a trivially low same-layer self-consistency check.

Per-token metrics (majority probability, entropy, flip rate) are computed
from predicted cluster assignments; A/B/C pair distances use gold labels.

Expected signal if clusters are stable containers:
    - avg_p_majority increases with depth
    - avg_entropy and avg_flip_rate decrease
    - A (same token) < B (same cluster, diff token) < C (diff cluster)

Outputs (--output_dir/):
    metrics.csv       per-layer aggregate stats
    p_majority.png    avg_p_majority vs layer
    entropy.png       avg_entropy vs layer
    flip_rate.png     avg_flip_rate vs layer
    distances.png     A/B/C pair distances vs layer
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cluster_stability")


# ── Cache loading ─────────────────────────────────────────────────────────────

def load_cache(cache_dir: str):
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        log.error("meta.json not found in %s", cache_dir)
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    layers = meta["layers"]
    hidden_paths = {}
    for l in layers:
        p = os.path.join(cache_dir, f"hidden_L{l:02d}.npy")
        if not os.path.exists(p):
            log.error("Missing hidden state file: %s", p)
            sys.exit(1)
        hidden_paths[l] = p
    cluster_ids = np.load(os.path.join(cache_dir, "clusters.npy")).astype(np.int32)
    token_ids   = np.load(os.path.join(cache_dir, "tokens.npy")).astype(np.int32)
    log.info("Cache: N=%d  layers=%s", len(cluster_ids), layers)
    return hidden_paths, cluster_ids, token_ids, layers


# ── Reference centroids from the final layer ──────────────────────────────────

def build_centroids(
    h_np: np.ndarray,
    cluster_ids_np: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """
    centroid[c] = mean(h[cluster_ids == c]), unit-normalized.
    Returns (centroids (K, D), ordered list of cluster IDs).
    """
    h = torch.from_numpy(h_np.astype(np.float32)).to(device)
    unique_clusters = sorted(int(c) for c in np.unique(cluster_ids_np) if c >= 0)
    centroid_list = []
    for c in unique_clusters:
        mask = torch.from_numpy(cluster_ids_np == c).to(device)
        centroid_list.append(h[mask].mean(dim=0))
    centroids = torch.stack(centroid_list)                               # (K, D)
    centroids = centroids / centroids.norm(dim=1, keepdim=True).clamp(min=1e-8)
    log.info("Built %d unit-normalized centroids from final layer", len(unique_clusters))
    return centroids, unique_clusters


# ── Nearest-centroid assignment ───────────────────────────────────────────────

def nearest_centroid(h: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Assign each row of h (N, D) to its nearest centroid (K, D).
    Builds (N, K) distance matrix — never (N, K, D).
    Returns (N,) int64 indices into centroids.
    """
    h_sq  = (h * h).sum(dim=1, keepdim=True)           # (N, 1)
    c_sq  = (centroids * centroids).sum(dim=1)          # (K,)
    cross = h @ centroids.T                             # (N, K)
    sq_d  = h_sq + c_sq.unsqueeze(0) - 2.0 * cross     # (N, K)
    return sq_d.clamp(min=0).argmin(dim=1)


# ── Token index (built once, reused every layer) ──────────────────────────────

def build_token_index(
    token_ids_np: np.ndarray,
    cluster_ids_np: np.ndarray,
    min_occ: int,
) -> Dict[int, np.ndarray]:
    """
    token_id -> int64 array of sample indices (only positions with valid cluster).
    Filtered to tokens with >= min_occ occurrences.
    """
    tmp: Dict[int, list] = defaultdict(list)
    for i in range(len(token_ids_np)):
        if cluster_ids_np[i] >= 0:
            tmp[int(token_ids_np[i])].append(i)
    result = {
        t: np.array(idx, dtype=np.int64)
        for t, idx in tmp.items()
        if len(idx) >= min_occ
    }
    log.info("Qualified tokens (>= %d occ): %d", min_occ, len(result))
    return result


# ── Per-token cluster consistency ─────────────────────────────────────────────

def token_consistency(
    pred_cluster_np: np.ndarray,
    token_to_idx: Dict[int, np.ndarray],
    n_flip_pairs: int,
    rng: np.random.Generator,
) -> Dict[int, dict]:
    """
    For each qualified token compute:
      p_majority  — fraction of occurrences in the modal cluster
      entropy     — H = -sum(p_k * log(p_k + 1e-9))  [nats]
      flip_rate   — fraction of randomly sampled pairs that land in diff clusters
    """
    results = {}
    for tok, idx in token_to_idx.items():
        c = pred_cluster_np[idx]
        total = len(c)

        _, counts = np.unique(c, return_counts=True)
        p_majority = float(counts.max()) / total
        pk = counts / total
        H = float(-np.sum(pk * np.log(pk + 1e-9)))

        k = n_flip_pairs * 4
        p1 = rng.integers(0, total, size=k)
        p2 = rng.integers(0, total, size=k)
        valid = p1 != p2
        p1, p2 = p1[valid][:n_flip_pairs], p2[valid][:n_flip_pairs]
        flip_rate = float((c[p1] != c[p2]).mean()) if len(p1) > 0 else 0.0

        results[tok] = {"p_majority": p_majority, "entropy": H, "flip_rate": flip_rate}
    return results


# ── Per-layer aggregation ─────────────────────────────────────────────────────

def aggregate(tok_stats: Dict[int, dict]) -> dict:
    p_maj = [v["p_majority"] for v in tok_stats.values()]
    entr  = [v["entropy"]    for v in tok_stats.values()]
    flips = [v["flip_rate"]  for v in tok_stats.values()]
    p_arr = np.array(p_maj)
    return {
        "avg_p_majority":   float(np.mean(p_maj)),
        "avg_entropy":      float(np.mean(entr)),
        "avg_flip_rate":    float(np.mean(flips)),
        "pct_majority_0.8": float(np.mean(p_arr > 0.8)),
        "pct_majority_0.9": float(np.mean(p_arr > 0.9)),
    }


# ── A / B / C pair distances ──────────────────────────────────────────────────

def pair_distances(
    h: torch.Tensor,
    token_ids_np: np.ndarray,
    cluster_ids_np: np.ndarray,
    token_to_idx: Dict[int, np.ndarray],
    n_pairs: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    A — same token, different occurrence        (contextual variation)
    B — same gold cluster, different token      (intra-cluster spread)
    C — different gold cluster                  (inter-cluster distance)

    h is already unit-normalized.  Returns mean L2 per condition.
    No O(N²): random pair sampling throughout.
    """
    N = h.shape[0]

    # A: per-token sampling, mean over tokens
    k_per_tok = max(n_pairs // max(len(token_to_idx), 1), 10)
    a_vals = []
    for idx in token_to_idx.values():
        n_t = len(idx)
        if n_t < 2:
            continue
        p1 = rng.integers(0, n_t, size=k_per_tok * 4)
        p2 = rng.integers(0, n_t, size=k_per_tok * 4)
        keep = p1 != p2
        p1, p2 = p1[keep][:k_per_tok], p2[keep][:k_per_tok]
        if len(p1) == 0:
            continue
        h1 = h[torch.from_numpy(idx[p1]).to(device)]
        h2 = h[torch.from_numpy(idx[p2]).to(device)]
        a_vals.append(float((h1 - h2).norm(dim=1).mean()))
    dist_A = float(np.nanmean(a_vals)) if a_vals else float("nan")

    # B: same cluster, different token — global oversample
    p1 = rng.integers(0, N, size=n_pairs * 12).astype(np.int64)
    p2 = rng.integers(0, N, size=n_pairs * 12).astype(np.int64)
    keep = (
        (cluster_ids_np[p1] == cluster_ids_np[p2])
        & (token_ids_np[p1]  != token_ids_np[p2])
        & (cluster_ids_np[p1] >= 0)
    )
    p1b, p2b = p1[keep][:n_pairs], p2[keep][:n_pairs]
    if len(p1b) > 0:
        h1 = h[torch.from_numpy(p1b).to(device)]
        h2 = h[torch.from_numpy(p2b).to(device)]
        dist_B = float((h1 - h2).norm(dim=1).mean())
    else:
        dist_B = float("nan")

    # C: different cluster — global oversample
    p1 = rng.integers(0, N, size=n_pairs * 4).astype(np.int64)
    p2 = rng.integers(0, N, size=n_pairs * 4).astype(np.int64)
    keep = (
        (cluster_ids_np[p1] != cluster_ids_np[p2])
        & (cluster_ids_np[p1] >= 0)
        & (cluster_ids_np[p2] >= 0)
    )
    p1c, p2c = p1[keep][:n_pairs], p2[keep][:n_pairs]
    if len(p1c) > 0:
        h1 = h[torch.from_numpy(p1c).to(device)]
        h2 = h[torch.from_numpy(p2c).to(device)]
        dist_C = float((h1 - h2).norm(dim=1).mean())
    else:
        dist_C = float("nan")

    return dist_A, dist_B, dist_C


# ── Plotting ──────────────────────────────────────────────────────────────────

def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return None


def save_plots(all_metrics: List[dict], layers: List[int], output_dir: str) -> None:
    plt = _get_plt()
    if plt is None:
        return
    xs = [str(l) for l in layers]

    def curve(key):
        return [m[key] for m in all_metrics]

    # Single-metric plots
    for fname, key, ylabel, color, ylim, hlines in [
        ("p_majority.png", "avg_p_majority", "avg p_majority",    "tab:blue",   (0, 1.05), [0.8, 0.9]),
        ("entropy.png",    "avg_entropy",    "avg entropy (nats)", "tab:orange", None,      []),
        ("flip_rate.png",  "avg_flip_rate",  "avg flip rate",     "tab:red",    (0, 1.05), []),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, curve(key), "o-", lw=2, color=color)
        for hv in hlines:
            ax.axhline(hv, ls="--", color="gray", lw=1, alpha=0.6)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs layer")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)

    # A/B/C distances
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, curve("dist_C"), "o-",  lw=2.5, color="tab:red",   label="C — diff cluster")
    ax.plot(xs, curve("dist_B"), "s--", lw=2,   color="tab:blue",  label="B — same cluster, diff token")
    ax.plot(xs, curve("dist_A"), "^:",  lw=1.5, color="tab:green", label="A — same token (context)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("mean L2  (unit sphere)")
    ax.set_title("A/B/C pair distances vs layer\nExpect: A < B < C")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "distances.png"), dpi=150)
    plt.close(fig)

    log.info("Plots saved to %s/", output_dir)


# ── CSV ───────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "layer",
    "avg_p_majority", "avg_entropy", "avg_flip_rate",
    "pct_majority_0.8", "pct_majority_0.9",
    "dist_A", "dist_B", "dist_C",
]


def save_csv(all_metrics: List[dict], layers: List[int], output_dir: str) -> None:
    path = os.path.join(output_dir, "metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for l, m in zip(layers, all_metrics):
            row: dict = {"layer": l}
            for k in _CSV_FIELDS[1:]:
                v = m.get(k, float("nan"))
                row[k] = f"{v:.6f}" if isinstance(v, float) else v
            writer.writerow(row)
    log.info("Metrics → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context vs cluster stability analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache_dir",       required=True,
                        help="probe_cache dir (hidden_L*.npy + clusters.npy + tokens.npy + meta.json)")
    parser.add_argument("--output_dir",
                        default="interference_experiment/results/cluster_stability")
    parser.add_argument("--min_occurrences", type=int, default=20,
                        help="Min occurrences per token to include in analysis")
    parser.add_argument("--n_pairs",         type=int, default=20000,
                        help="Pairs to sample per A/B/C condition per layer")
    parser.add_argument("--n_flip_pairs",    type=int, default=20,
                        help="Pairs to sample per token for flip rate")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load ──────────────────────────────────────────────────────────────────
    hidden_paths, cluster_ids_np, token_ids_np, layers = load_cache(args.cache_dir)
    token_to_idx = build_token_index(token_ids_np, cluster_ids_np, args.min_occurrences)

    # ── Final-layer centroids (fixed reference for all layers) ────────────────
    final_layer = layers[-1]
    log.info("Building centroids from layer %d ...", final_layer)
    h_final_np = np.load(hidden_paths[final_layer])
    centroids, centroid_cluster_ids = build_centroids(h_final_np, cluster_ids_np, device)
    del h_final_np

    # ── Per-layer loop ────────────────────────────────────────────────────────
    all_metrics: List[dict] = []

    for l in layers:
        t0 = time.time()
        log.info("--- Layer %d ---", l)

        h_np = np.load(hidden_paths[l])
        h = torch.from_numpy(h_np.astype(np.float32)).to(device)
        del h_np
        h = h / h.norm(dim=1, keepdim=True).clamp(min=1e-8)   # unit sphere

        # Predicted cluster for all N positions using final-layer centroids
        pred_kidx = nearest_centroid(h, centroids)               # (N,) indices into centroids
        pred_cluster_np = np.array(
            [centroid_cluster_ids[k] for k in pred_kidx.cpu().tolist()],
            dtype=np.int32,
        )

        # Per-token consistency stats
        tok_stats = token_consistency(pred_cluster_np, token_to_idx, args.n_flip_pairs, rng)
        agg = aggregate(tok_stats)

        # A/B/C distances on unit-normalized h (gold cluster_ids for B/C grouping)
        dist_A, dist_B, dist_C = pair_distances(
            h, token_ids_np, cluster_ids_np, token_to_idx,
            args.n_pairs, rng, device,
        )

        metrics = {**agg, "dist_A": dist_A, "dist_B": dist_B, "dist_C": dist_C}
        all_metrics.append(metrics)
        del h

        log.info(
            "  p_maj=%.3f  H=%.3f  flip=%.3f  pct>0.9=%.2f  "
            "A=%.4f  B=%.4f  C=%.4f  (%.1fs)",
            metrics["avg_p_majority"], metrics["avg_entropy"],
            metrics["avg_flip_rate"], metrics["pct_majority_0.9"],
            dist_A, dist_B, dist_C,
            time.time() - t0,
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_csv(all_metrics, layers, args.output_dir)
    save_plots(all_metrics, layers, args.output_dir)

    # ── Console summary ───────────────────────────────────────────────────────
    sep = "=" * 84
    print(f"\n{sep}")
    print("  CLUSTER STABILITY RESULTS")
    print(sep)
    print(
        f"  {'Layer':>6}  {'p_maj':>7}  {'H':>7}  {'flip':>7}  "
        f"{'%>0.8':>7}  {'%>0.9':>7}  {'A':>8}  {'B':>8}  {'C':>8}"
    )
    print(
        f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  "
        f"{'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}"
    )
    for l, m in zip(layers, all_metrics):
        print(
            f"  {l:>6}  {m['avg_p_majority']:7.3f}  {m['avg_entropy']:7.4f}  "
            f"{m['avg_flip_rate']:7.4f}  {m['pct_majority_0.8']:7.3f}  "
            f"{m['pct_majority_0.9']:7.3f}  "
            f"{m['dist_A']:8.4f}  {m['dist_B']:8.4f}  {m['dist_C']:8.4f}"
        )
    print(sep)

    last = all_metrics[-1]
    ok = last["dist_A"] < last["dist_B"] < last["dist_C"]
    print(f"\n  A < B < C at final layer:  {'YES' if ok else 'NO'}")
    print(f"  avg p_majority (final):    {last['avg_p_majority']:.3f}")
    print(f"  tokens with p_maj > 0.9:   {last['pct_majority_0.9']:.1%}\n")


if __name__ == "__main__":
    main()
