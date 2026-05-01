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

def _pairwise_l2_sampled(
    h: torch.Tensor,        # (N, D)
    idx1: np.ndarray,       # (P,) int64 — first index of each pair
    idx2: np.ndarray,       # (P,) int64 — second index of each pair
    device: torch.device,
) -> float:
    """Mean L2 distance for a pre-sampled set of pairs. O(P·D) — no N² anywhere."""
    if len(idx1) == 0:
        return float("nan")
    h1 = h[torch.from_numpy(idx1).to(device)]   # (P, D)
    h2 = h[torch.from_numpy(idx2).to(device)]   # (P, D)
    return float((h1 - h2).norm(dim=1).mean())


def sample_pair_distances(
    h: torch.Tensor,                    # (N, D) — already on device, optionally normalized
    groups: "TokenGroups",
    clusters_np: np.ndarray,            # (N,) int32 — cluster label for all N positions
    n_pairs: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Sample pairs for three conditions and return mean L2 distances.

    A — same token, different occurrence:
        token_ids[i] == token_ids[j],  i != j

    B — same cluster, different token:
        cluster_ids[i] == cluster_ids[j],  token_ids[i] != token_ids[j]

    C — different cluster:
        cluster_ids[i] != cluster_ids[j]

    Each condition samples up to n_pairs pairs total (oversampling then slicing).
    No O(N²) anywhere — all pair indices are materialized as int64 arrays.
    """
    N = h.shape[0]

    # ── A: same token, different occurrence ──────────────────────────────────
    # Per-token: sample k pairs from that token's occurrences; average across tokens.
    n_tokens = max(len(groups.token_to_idx), 1)
    k_per_tok = max(n_pairs // n_tokens, 10)
    a_vals: List[float] = []
    for idx in groups.token_to_idx.values():
        n_t = len(idx)
        if n_t < 2:
            continue
        p1 = rng.integers(0, n_t, size=k_per_tok * 3)
        p2 = rng.integers(0, n_t, size=k_per_tok * 3)
        keep = p1 != p2
        p1, p2 = p1[keep][:k_per_tok], p2[keep][:k_per_tok]
        if len(p1) == 0:
            continue
        a_vals.append(_pairwise_l2_sampled(h, idx[p1], idx[p2], device))
    dist_A = float(np.nanmean(a_vals)) if a_vals else float("nan")

    # ── B: same cluster, different token ─────────────────────────────────────
    # Per cluster: pool all qualified-token occurrences, sample pairs with
    # different token_id labels; average across clusters.
    n_clusters = max(len(groups.cluster_ids), 1)
    k_per_cl = max(n_pairs // n_clusters, 50)
    b_vals: List[float] = []
    for cid in groups.cluster_ids:
        tok_list = [t for t in groups.cluster_to_tokens[cid]
                    if t in groups.token_to_idx]
        if len(tok_list) < 2:
            continue
        all_idx_c = np.concatenate([groups.token_to_idx[t] for t in tok_list])
        all_tok_c = np.concatenate(
            [np.full(len(groups.token_to_idx[t]), t, dtype=np.int64) for t in tok_list]
        )
        N_c = len(all_idx_c)
        p1 = rng.integers(0, N_c, size=k_per_cl * 4)
        p2 = rng.integers(0, N_c, size=k_per_cl * 4)
        keep = all_tok_c[p1] != all_tok_c[p2]
        p1, p2 = p1[keep][:k_per_cl], p2[keep][:k_per_cl]
        if len(p1) == 0:
            continue
        b_vals.append(_pairwise_l2_sampled(h, all_idx_c[p1], all_idx_c[p2], device))
    dist_B = float(np.nanmean(b_vals)) if b_vals else float("nan")

    # ── C: different cluster ──────────────────────────────────────────────────
    # Sample from all N positions; ~88% of random pairs cross cluster boundaries.
    p1 = rng.integers(0, N, size=n_pairs * 2).astype(np.int64)
    p2 = rng.integers(0, N, size=n_pairs * 2).astype(np.int64)
    keep = clusters_np[p1] != clusters_np[p2]
    p1, p2 = p1[keep][:n_pairs], p2[keep][:n_pairs]
    dist_C = _pairwise_l2_sampled(h, p1, p2, device)

    return dist_A, dist_B, dist_C


def build_final_layer_centroids(
    h_np: np.ndarray,       # (N, D) float16 — final layer hidden states
    groups: "TokenGroups",
    device: torch.device,
    normalize: bool,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute cluster centroids from the FINAL layer's hidden states.
    Centroid of cluster k = mean of per-token means for all qualified tokens in k.

    These centroids are fixed and reused across all layers for the flip-rate metric,
    so that the flip rate measures "does h at layer l map to the correct region
    as defined by the final representation?" rather than "does h map to its own
    layer's centroid?" (which is trivially low for any well-trained model).
    """
    h = torch.from_numpy(h_np.astype(np.float32)).to(device)
    if normalize:
        h = h / h.norm(dim=1, keepdim=True).clamp(min=1e-8)

    centroid_list: List[torch.Tensor] = []
    centroid_ids:  List[int] = []
    for cid in groups.cluster_ids:
        tok_list = [t for t in groups.cluster_to_tokens[cid]
                    if t in groups.token_to_idx]
        if not tok_list:
            continue
        mu_list = []
        for t in tok_list:
            idx_t = torch.from_numpy(groups.token_to_idx[t]).to(device)
            mu_list.append(h[idx_t].mean(dim=0))
        centroid_list.append(torch.stack(mu_list).mean(dim=0))
        centroid_ids.append(cid)

    return torch.stack(centroid_list), centroid_ids   # (K, D),  [int, ...]


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
    h_np: np.ndarray,
    groups: TokenGroups,
    clusters_np: np.ndarray,
    final_centroids: torch.Tensor,
    final_centroid_ids: List[int],
    final_sub_centroids: Optional[torch.Tensor],
    final_sub_centroid_ids: Optional[List[int]],
    device: torch.device,
    n_pairs: int,
    normalize: bool,
    rng: np.random.Generator,
) -> dict:
    """
    Compute all metrics for one layer using sampled pair distances.
    A = same token (contextual drift), B = same cluster diff token, C = diff cluster.
    Flip rate uses precomputed final-layer centroids so it measures whether
    representations at layer l map to the correct final-layer cluster region.
    """
    h = torch.from_numpy(h_np.astype(np.float32)).to(device)

    avg_norm = float(h.norm(dim=1).mean())

    if normalize:
        h = h / h.norm(dim=1, keepdim=True).clamp(min=1e-8)
        norm_denom = 1.0
    else:
        norm_denom = max(avg_norm, 1e-8)

    # ── A/B/C sampled pair distances ─────────────────────────────────────────
    dist_A_raw, dist_B_raw, dist_C_raw = sample_pair_distances(
        h, groups, clusters_np, n_pairs, rng, device
    )
    dist_A = dist_A_raw / norm_denom
    dist_B = dist_B_raw / norm_denom
    dist_C = dist_C_raw / norm_denom

    # ── Cluster flip rate (final-layer centroids) ─────────────────────────────
    fc = final_centroids.to(device)
    pred_centroid_idx = nearest_centroid_assign(h, fc)  # (N,)
    pred_cid_list = [final_centroid_ids[i] for i in pred_centroid_idx.cpu().tolist()]
    pred_cid_t = torch.tensor(pred_cid_list, dtype=torch.long)

    flip_rates: List[float] = []
    for tok, idx in groups.token_to_idx.items():
        true_c = groups.token_to_cluster[tok]
        pred_c = pred_cid_t[torch.from_numpy(idx)]
        flip_rates.append(float((pred_c != true_c).float().mean()))
    cluster_flip_rate = float(np.mean(flip_rates)) if flip_rates else 0.0

    # ── Cluster entropy over predicted assignments ────────────────────────────
    counts = torch.bincount(
        pred_centroid_idx.long(), minlength=len(final_centroid_ids)
    ).float()
    probs = counts / counts.sum().clamp(min=1e-12)
    nz = probs[probs > 0]
    cluster_entropy = float(-(nz * nz.log()).sum() / np.log(2))

    # ── Subcluster flip rate (final-layer sub-centroids) ──────────────────────
    subcluster_flip_rate: Optional[float] = None
    if (final_sub_centroids is not None
            and final_sub_centroid_ids is not None
            and groups.token_to_sub is not None
            and len(final_sub_centroid_ids) >= 2):

        fsc = final_sub_centroids.to(device)
        sub_qualified = sorted(t for t in groups.token_to_sub if t in groups.token_to_idx)
        if sub_qualified:
            n_per_tok = [len(groups.token_to_idx[t]) for t in sub_qualified]
            all_idx = np.concatenate([groups.token_to_idx[t] for t in sub_qualified])
            h_sub = h[torch.from_numpy(all_idx).to(device)]

            pred_sub_cidx = nearest_centroid_assign(h_sub, fsc)
            pred_sub_ids = torch.tensor(
                [final_sub_centroid_ids[i] for i in pred_sub_cidx.cpu().tolist()],
                dtype=torch.long,
            )

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
        "dist_A":               round(dist_A, 6),
        "dist_B":               round(dist_B, 6),
        "dist_C":               round(dist_C, 6),
        "cluster_flip_rate":    round(cluster_flip_rate, 6),
        "cluster_entropy":      round(cluster_entropy, 4),
        "subcluster_flip_rate": (round(subcluster_flip_rate, 6)
                                 if subcluster_flip_rate is not None else None),
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
    dist_A = [all_metrics[l]["dist_A"] for l in layers]
    dist_B = [all_metrics[l]["dist_B"] for l in layers]
    dist_C = [all_metrics[l]["dist_C"] for l in layers]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xlabels, dist_C, "o-",  color="tab:red",   lw=2.5, label="C — diff cluster")
    ax.plot(xlabels, dist_B, "s--", color="tab:blue",  lw=2,   label="B — same cluster, diff token")
    ax.plot(xlabels, dist_A, "^:",  color="tab:green", lw=1.5, label="A — same token (contextual drift)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 distance / avg hidden norm")
    ax.set_title(
        "A/B/C pair distances vs layer  (normalized)\n"
        "Expect: C  ≫  B  ≥  A"
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
    c_flip  = [all_metrics[l]["cluster_flip_rate"]  for l in layers]
    s_flip  = [all_metrics[l]["subcluster_flip_rate"] for l in layers]
    entropy = [all_metrics[l]["cluster_entropy"]    for l in layers]
    has_sub = any(v is not None for v in s_flip)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(xlabels, c_flip, "o-", color="tab:blue",   lw=2, label="cluster flip rate")
    if has_sub:
        s_vals = [v if v is not None else float("nan") for v in s_flip]
        ax1.plot(xlabels, s_vals, "s--", color="tab:orange", lw=2, label="subcluster flip rate")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Flip rate")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xlabels, entropy, "D:", color="tab:gray", lw=1.5, label="cluster entropy (bits)")
    ax2.set_ylabel("Cluster entropy (bits)", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(
        "Cluster flip rate vs layer  (final-layer centroids)\n"
        "Expect: cluster flip rate LOW, subcluster flip rate HIGHER"
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "flip_rates_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Plot → %s", path)


def print_summary(all_metrics: Dict[int, dict], layers: List[int]) -> None:
    sep = "=" * 82
    print(f"\n{sep}")
    print("  CONTEXT vs CLUSTER — RESULTS  (A=same token / B=same cluster / C=diff cluster)")
    print(sep)
    print(f"  {'Layer':>6}  {'A (drift)':>10}  {'B (intra)':>10}  {'C (inter)':>10}  "
          f"{'C/B':>8}  {'c_flip':>8}  {'sc_flip':>9}  {'entropy':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}")
    for l in layers:
        m = all_metrics[l]
        ratio = m["dist_C"] / m["dist_B"] if m["dist_B"] > 0 else float("inf")
        sc = f"{m['subcluster_flip_rate']:.4f}" if m["subcluster_flip_rate"] is not None else "  N/A   "
        print(f"  {l:>6}  {m['dist_A']:10.4f}  {m['dist_B']:10.4f}  {m['dist_C']:10.4f}  "
              f"{ratio:7.2f}x  {m['cluster_flip_rate']:8.4f}  "
              f"{sc:>9}  {m['cluster_entropy']:8.3f}")
    print()

    last = all_metrics[layers[-1]]
    first_c_gt_b = next(
        (l for l in layers if all_metrics[l]["dist_C"] > all_metrics[l]["dist_B"]),
        None,
    )
    hierarchy_holds = last["dist_C"] > last["dist_B"] >= last["dist_A"]
    flip_is_low = last["cluster_flip_rate"] < 0.15

    print("  INTERPRETATION (final layer):")
    print(f"    C > B >= A hierarchy:              {'YES' if hierarchy_holds else 'NO'}")
    if first_c_gt_b is not None:
        print(f"    First layer where C > B:           Layer {first_c_gt_b}")
    print(f"    Cluster flip rate (final):         {last['cluster_flip_rate']:.3f}  "
          f"({'LOW — clusters are stable' if flip_is_low else 'HIGH — context crosses cluster boundaries'})")
    print(f"    Cluster entropy (final):           {last['cluster_entropy']:.3f} bits")
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
    parser.add_argument("--n_pairs",         type=int, default=10000,
                        help="Pairs to sample per A/B/C condition per layer")
    parser.add_argument("--normalize",       action="store_true",
                        help="L2-normalize hidden states before computing distances")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    _setup_logging()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
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

    # ── Build final-layer centroids (fixed reference for flip rate) ──────────
    final_layer = layers[-1]
    log.info("=== Building final-layer centroids from layer %d ===", final_layer)
    h_final_np = np.load(hidden_paths[final_layer])
    final_centroids, final_centroid_ids = build_final_layer_centroids(
        h_final_np, groups, device, normalize=args.normalize
    )

    # ── Build final-layer sub-centroids (if subcluster map available) ────────
    final_sub_centroids: Optional[torch.Tensor] = None
    final_sub_centroid_ids: Optional[List[int]] = None
    if (groups.token_to_sub is not None
            and groups.subcluster_to_tokens is not None
            and groups.sub_ids is not None):
        log.info("Building final-layer sub-centroids ...")
        h_final = torch.from_numpy(h_final_np.astype(np.float32)).to(device)
        if args.normalize:
            h_final = h_final / h_final.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sub_centroid_list: List[torch.Tensor] = []
        sub_centroid_ids_: List[int] = []
        for sid in groups.sub_ids:
            tok_list = [t for t in groups.subcluster_to_tokens[sid]
                        if t in groups.token_to_idx]
            if not tok_list:
                continue
            mu_list = [h_final[torch.from_numpy(groups.token_to_idx[t]).to(device)].mean(dim=0)
                       for t in tok_list]
            sub_centroid_list.append(torch.stack(mu_list).mean(dim=0))
            sub_centroid_ids_.append(sid)
        if len(sub_centroid_list) >= 2:
            final_sub_centroids    = torch.stack(sub_centroid_list)
            final_sub_centroid_ids = sub_centroid_ids_
        del h_final

    del h_final_np

    # ── Per-layer analysis ──────────────────────────────────────────────────
    log.info(
        "=== Analyzing layers (n_pairs=%d, normalize=%s) ===",
        args.n_pairs, args.normalize,
    )
    all_metrics: Dict[int, dict] = {}

    for l in layers:
        t0 = time.time()
        log.info("--- Layer %d ---", l)
        h_np = np.load(hidden_paths[l])

        metrics = analyze_layer(
            h_np, groups, clusters, final_centroids, final_centroid_ids,
            final_sub_centroids, final_sub_centroid_ids,
            device, args.n_pairs, args.normalize, rng,
        )
        del h_np

        elapsed = time.time() - t0
        sub_str = (f"{metrics['subcluster_flip_rate']:.4f}"
                   if metrics["subcluster_flip_rate"] is not None else "N/A")
        log.info(
            "  A=%.4f  B=%.4f  C=%.4f  C/B=%.1fx  "
            "c_flip=%.4f  sc_flip=%s  entropy=%.3f  (%.1fs)",
            metrics["dist_A"], metrics["dist_B"], metrics["dist_C"],
            metrics["dist_C"] / max(metrics["dist_B"], 1e-8),
            metrics["cluster_flip_rate"],
            sub_str,
            metrics["cluster_entropy"],
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
