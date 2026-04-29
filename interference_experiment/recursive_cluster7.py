#!/usr/bin/env python3
"""
recursive_cluster7.py

Recursive subclustering of a large target cluster from the GPT-2 XL
interference graph experiment.

Usage:
  python recursive_cluster7.py \\
      --graph        graphs/layer_final_W_norm.npy \\
      --clusters     results/cluster_tokens_coactivation_final.json \\
      --target_cluster 7 \\
      --output_dir   results/ \\
      --recursive_depth 2

The script auto-detects sibling files (Louvain labels, frequent_token_ids)
from the --graph path so you rarely need to pass them explicitly.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("cluster7")


log = logging.getLogger("cluster7")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_graph(graph_path: str) -> np.ndarray:
    """Load the full N×N adjacency matrix from .npy or .npz."""
    p = Path(graph_path)
    if not p.exists():
        log.error("Graph file not found: %s", graph_path)
        sys.exit(1)

    if p.suffix == ".npz":
        from scipy.sparse import load_npz
        W = load_npz(graph_path).toarray().astype(np.float32)
    else:
        W = np.load(graph_path).astype(np.float32)

    np.fill_diagonal(W, 0.0)
    W = (W + W.T) * 0.5          # enforce symmetry (numerical noise)
    log.info("Loaded graph: shape=%s  max=%.4f  sparsity=%.3f",
             W.shape, W.max(), (W == 0).mean())
    return W


def load_cluster_json(path: str) -> Dict[str, List[dict]]:
    """Load cluster_tokens JSON → {cluster_id_str: [{"token", "token_id", "freq"}, ...]}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded cluster JSON from %s (%d clusters)", path, len(data))
    return data


def _try_load_npy(candidates: List[str], label: str) -> Optional[np.ndarray]:
    """Load the first existing candidate path; warn if none found."""
    for c in candidates:
        if os.path.exists(c):
            arr = np.load(c)
            log.info("Loaded %s from %s  (shape=%s)", label, c, arr.shape)
            return arr
    log.warning(
        "%s not found (tried: %s). Some features will be limited.",
        label, ", ".join(candidates),
    )
    return None


def load_labels(labels_arg: Optional[str], graph_path: str) -> Optional[np.ndarray]:
    """
    Load Louvain cluster label array (shape = N_sub).
    Auto-detects from graph path: tries *_labels_louvain.npy and *_labels_spectral.npy.
    """
    graph_dir = str(Path(graph_path).parent)
    stem_map = {
        # e.g. layer_final_W_norm.npy → layer_final_coactivation_labels_louvain.npy
        "layer_final_coactivation_labels_louvain.npy",
        "layer_final_labels_louvain.npy",
        "labels_louvain.npy",
        "layer_final_coactivation_labels_spectral.npy",
    }
    candidates: List[str] = []
    if labels_arg:
        candidates.append(labels_arg)
    for name in stem_map:
        candidates.append(os.path.join(graph_dir, name))

    arr = _try_load_npy(candidates, "Louvain labels")
    return arr.astype(np.int32) if arr is not None else None


def load_freq_ids(freq_ids_arg: Optional[str], graph_path: str) -> Optional[np.ndarray]:
    """
    Load frequent_token_ids array (shape = N_sub).
    Maps local index → original tokenizer vocab ID.
    """
    graph_dir = str(Path(graph_path).parent)
    candidates: List[str] = []
    if freq_ids_arg:
        candidates.append(freq_ids_arg)
    candidates += [
        os.path.join(graph_dir, "layer_final_frequent_token_ids.npy"),
        os.path.join(graph_dir, "frequent_token_ids.npy"),
        os.path.join(graph_dir, "freq_ids.npy"),
    ]
    arr = _try_load_npy(candidates, "frequent_token_ids")
    return arr.astype(np.int32) if arr is not None else None


def decode_tokens_with_tokenizer(token_ids: List[int], model_name: str) -> List[str]:
    """Decode a list of token IDs to strings using the HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        return [tok.decode([tid]) for tid in token_ids]
    except Exception as e:
        log.warning("Tokenizer unavailable (%s) — using raw token IDs as strings.", e)
        return [str(tid) for tid in token_ids]


# ---------------------------------------------------------------------------
# Cluster membership extraction
# ---------------------------------------------------------------------------

def extract_cluster_members(
    target_cluster: int,
    cluster_json: Dict[str, List[dict]],
    labels: Optional[np.ndarray],
    freq_ids: Optional[np.ndarray],
    model_name: str,
    W_size: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return (local_indices, token_strings) for all members of target_cluster.

    Priority:
      1. labels.npy + freq_ids → full membership, proper string decoding
      2. labels.npy alone      → full membership, raw token-ID strings
      3. cluster JSON          → top-20 only (warns loudly), may be wrong indices
    """
    key = str(target_cluster)

    # ---- Route 1: full labels + freq_ids ----
    if labels is not None:
        local_indices = np.where(labels == target_cluster)[0].astype(np.int32)
        log.info(
            "Cluster %d membership from labels.npy: %d tokens (out of %d)",
            target_cluster, len(local_indices), len(labels),
        )
        if freq_ids is not None:
            token_ids = [int(freq_ids[i]) for i in local_indices]
            token_strings = decode_tokens_with_tokenizer(token_ids, model_name)
        else:
            # Decode from JSON for tokens we have, fall back to index string
            json_map: Dict[int, str] = {}
            if key in cluster_json:
                for entry in cluster_json[key]:
                    json_map[int(entry["token_id"])] = entry.get("token", "")
            # Without freq_ids we can't know which local_idx → token_id.
            # Use placeholder strings keyed by local index.
            log.warning(
                "freq_ids unavailable — token strings will be local index labels."
            )
            token_strings = [f"tok_{i}" for i in local_indices]
        return local_indices, token_strings

    # ---- Route 2 / 3: fall back to JSON (top-20 only) ----
    log.warning(
        "labels.npy not found — falling back to JSON (top-20 tokens per cluster). "
        "Subclustering will run on a severely reduced token set."
    )
    if key not in cluster_json:
        log.error("Cluster %d not in JSON either.", target_cluster)
        sys.exit(1)

    entries = cluster_json[key]
    # If freq_ids available, map token_id → local_index; else assume token_id IS local_index
    if freq_ids is not None:
        id_to_local = {int(freq_ids[i]): i for i in range(len(freq_ids))}
        pairs = [(id_to_local[int(e["token_id"])], e.get("token", "?"))
                 for e in entries if int(e["token_id"]) in id_to_local]
    else:
        log.warning(
            "Neither labels.npy nor freq_ids found. Assuming token_id == local W index. "
            "This is INCORRECT if token_ids are vocab IDs, not subset indices."
        )
        pairs = [(int(e["token_id"]), e.get("token", "?")) for e in entries
                 if int(e["token_id"]) < W_size]

    if not pairs:
        log.error("No valid token mappings found. Check your inputs.")
        sys.exit(1)

    local_indices = np.array([p[0] for p in pairs], dtype=np.int32)
    token_strings = [p[1] for p in pairs]
    return local_indices, token_strings


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def induce_subgraph(W: np.ndarray, local_indices: np.ndarray) -> np.ndarray:
    """Extract and symmetrize W[indices, :][:, indices]."""
    W7 = W[np.ix_(local_indices, local_indices)].astype(np.float32)
    np.fill_diagonal(W7, 0.0)
    W7 = (W7 + W7.T) * 0.5
    log.info(
        "Induced subgraph: shape=%s  max=%.4f  sparsity=%.3f  total_weight=%.1f",
        W7.shape, W7.max(), (W7 == 0).mean(), W7.sum(),
    )
    return W7


def prep_affinity(W: np.ndarray) -> np.ndarray:
    """Clip negatives, zero diagonal, symmetrize."""
    A = np.clip(W, 0.0, None).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return (A + A.T) * 0.5


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------

def run_louvain(W: np.ndarray, resolution: float, seed: int) -> np.ndarray:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    A = prep_affinity(W)
    G = nx.from_numpy_array(A)
    communities = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    labels = np.zeros(len(A), dtype=np.int32)
    for c_idx, comm in enumerate(communities):
        for node in comm:
            labels[node] = c_idx
    log.info("  Louvain(res=%.1f): %d communities", resolution, len(communities))
    return labels


def run_spectral(W: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    A = prep_affinity(W)
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        n_init=5,
        random_state=seed,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(A).astype(np.int32)
    log.info("  Spectral(k=%d): done", k)
    return labels


def run_agglomerative(W: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    A = prep_affinity(W)
    A_max = A.max()
    # Convert affinity → distance for agglomerative
    D = (A_max - A) / (A_max + 1e-9)
    np.fill_diagonal(D, 0.0)
    agg = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    )
    labels = agg.fit_predict(D).astype(np.int32)
    log.info("  Agglomerative(k=%d): done", k)
    return labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_modularity(W: np.ndarray, labels: np.ndarray) -> float:
    """Weighted modularity Q for an undirected graph."""
    m2 = W.sum()
    if m2 == 0:
        return 0.0
    k = W.sum(axis=1)
    Q = 0.0
    for c in np.unique(labels):
        mask = labels == c
        A_c = W[np.ix_(mask, mask)]
        k_c = k[mask]
        Q += A_c.sum() - (k_c.sum() ** 2) / m2
    return float(Q / m2)


def cluster_entropy(labels: np.ndarray) -> float:
    """Shannon entropy (nats) of the cluster-size distribution."""
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def intra_inter(W: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """Returns (mean_intra, mean_inter, ratio)."""
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    diff = ~same
    np.fill_diagonal(diff, False)
    intra = float(W[same].mean()) if same.any() else 0.0
    inter = float(W[diff].mean()) if diff.any() else 0.0
    return intra, inter, intra / max(inter, 1e-9)


def partition_score(W: np.ndarray, labels: np.ndarray) -> float:
    """
    Composite quality score for selecting the best partition.

    Combines modularity, entropy balance, and penalties for degenerate
    partitions (singleton flood or single giant cluster).
    """
    n = len(labels)
    unique = np.unique(labels)
    k = len(unique)
    if k <= 1 or k >= n:
        return -1.0

    Q = compute_modularity(W, labels)
    H = cluster_entropy(labels)
    H_max = np.log(float(k) + 1e-9)
    H_norm = H / H_max                       # 1.0 = perfectly uniform sizes

    sizes = np.bincount(labels)
    singleton_frac = float((sizes == 1).sum()) / k
    giant_frac = float(sizes.max()) / n

    # Penalise too many singletons or one overwhelming cluster
    balance_penalty = singleton_frac * 0.5 + max(0.0, giant_frac - 0.6) * 0.5

    return Q * (0.5 + 0.5 * H_norm) * max(0.0, 1.0 - balance_penalty)


def compute_metrics(W: np.ndarray, labels: np.ndarray, method_name: str) -> dict:
    Q = compute_modularity(W, labels)
    H = cluster_entropy(labels)
    intra, inter, ratio = intra_inter(W, labels)
    score = partition_score(W, labels)
    sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    return {
        "method": method_name,
        "modularity": round(Q, 4),
        "entropy": round(H, 4),
        "n_clusters": int(len(np.unique(labels))),
        "intra_mean": round(intra, 6),
        "inter_mean": round(inter, 6),
        "intra_inter_ratio": round(ratio, 3),
        "composite_score": round(score, 4),
        "cluster_sizes": sizes,
        "max_size": max(sizes),
        "min_size": min(sizes),
    }


def log_metrics(m: dict) -> None:
    log.info(
        "    %-26s  Q=%.4f  n=%3d  score=%.4f  ratio=%.2f",
        m["method"], m["modularity"], m["n_clusters"],
        m["composite_score"], m["intra_inter_ratio"],
    )


# ---------------------------------------------------------------------------
# Eigenvalue analysis
# ---------------------------------------------------------------------------

def eigenvalue_analysis(W: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Compute eigenvalue decomposition of the symmetrically normalised subgraph.
    Returns (eigenvalues_desc, info_dict).
    """
    A = prep_affinity(W)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]

    evals = np.linalg.eigvalsh(A_norm)[::-1]   # descending
    lam = np.abs(evals)
    total = lam.sum()
    cumvar = np.cumsum(lam) / max(total, 1e-9)

    dims_50 = int(np.searchsorted(cumvar, 0.50)) + 1
    dims_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    pr = float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-9))

    top20 = evals[:min(20, len(evals))]
    gaps = np.diff(np.abs(top20))
    gap_idx = int(np.argmax(np.abs(gaps))) + 1 if len(gaps) > 0 else 1

    info = {
        "participation_ratio": round(pr, 2),
        "dims_for_50pct": dims_50,
        "dims_for_90pct": dims_90,
        "largest_eigenvalue": float(evals[0]),
        "spectral_gap_location": gap_idx,
        "spectral_gap_size": float(abs(gaps[gap_idx - 1])) if len(gaps) >= gap_idx else 0.0,
    }
    log.info(
        "Eigenvalues: PR=%.1f  dims@50%%=%d  dims@90%%=%d  gap@rank=%d",
        pr, dims_50, dims_90, gap_idx,
    )
    return evals, info


# ---------------------------------------------------------------------------
# Run all clustering methods
# ---------------------------------------------------------------------------

# Default hyperparameter grids
_LOUVAIN_RES = (0.5, 1.0, 1.5, 2.0)
_SPECTRAL_KS = (5, 10, 20)
_AGG_KS = (5, 10, 20)

# Named tuple type for clarity
PartitionResult = Tuple[str, np.ndarray, dict]   # (method_name, labels, metrics)


def run_all_methods(
    W: np.ndarray,
    seed: int = 42,
    louvain_resolutions: Tuple[float, ...] = _LOUVAIN_RES,
    spectral_ks: Tuple[int, ...] = _SPECTRAL_KS,
    run_agg: bool = True,
    agg_ks: Tuple[int, ...] = _AGG_KS,
) -> List[PartitionResult]:
    """
    Run all three clustering families on subgraph W.
    Returns list of (method_name, labels, metrics) sorted by composite_score desc.
    """
    results: List[PartitionResult] = []
    N = len(W)

    log.info("--- Louvain subclustering (N=%d) ---", N)
    for res in louvain_resolutions:
        try:
            labels = run_louvain(W, resolution=res, seed=seed)
            m = compute_metrics(W, labels, f"louvain_res{res:.1f}")
            log_metrics(m)
            results.append((m["method"], labels, m))
        except Exception as exc:
            log.warning("Louvain res=%.1f failed: %s", res, exc)

    log.info("--- Spectral subclustering ---")
    for k in spectral_ks:
        if k >= N:
            log.info("  Skipping spectral k=%d (N=%d)", k, N)
            continue
        try:
            labels = run_spectral(W, k=k, seed=seed)
            m = compute_metrics(W, labels, f"spectral_k{k}")
            log_metrics(m)
            results.append((m["method"], labels, m))
        except Exception as exc:
            log.warning("Spectral k=%d failed: %s", k, exc)

    if run_agg:
        log.info("--- Agglomerative subclustering ---")
        for k in agg_ks:
            if k >= N:
                log.info("  Skipping agg k=%d (N=%d)", k, N)
                continue
            try:
                labels = run_agglomerative(W, k=k)
                m = compute_metrics(W, labels, f"agg_k{k}")
                log_metrics(m)
                results.append((m["method"], labels, m))
            except Exception as exc:
                log.warning("Agglomerative k=%d failed: %s", k, exc)

    results.sort(key=lambda x: x[2]["composite_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Token centrality helpers
# ---------------------------------------------------------------------------

def top_tokens_by_centrality(
    mask: np.ndarray,
    token_strings: List[str],
    W: np.ndarray,
    top_n: int = 20,
) -> List[str]:
    """
    Pick the top_n most central tokens within a subcluster.
    Centrality = within-subcluster weighted degree (row-sum over subcluster).
    """
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []
    W_sub = W[np.ix_(indices, indices)]
    centrality = W_sub.sum(axis=1)
    order = np.argsort(centrality)[::-1][:top_n]
    return [token_strings[indices[i]] for i in order]


# ---------------------------------------------------------------------------
# Recursive subclustering
# ---------------------------------------------------------------------------

RecursiveInfo = Dict[int, Tuple[np.ndarray, List[str], np.ndarray]]
# {subcluster_id: (sub_labels, sub_token_strings, sub_W)}


def recursive_subcluster(
    W: np.ndarray,
    token_strings: List[str],
    depth: int,
    max_depth: int,
    min_size_to_split: int,
    seed: int,
    run_agg: bool,
) -> Tuple[np.ndarray, List[PartitionResult], RecursiveInfo]:
    """
    Run clustering at the current depth, then recurse into large subclusters.

    Returns:
        best_labels:   (N,) int32 — best partition at this level
        all_results:   sorted list of (name, labels, metrics)
        recursive_info: mapping from subcluster_id → deeper results
    """
    log.info("Depth=%d | N=%d tokens", depth, len(W))

    all_results = run_all_methods(W, seed=seed, run_agg=run_agg)

    if not all_results:
        log.warning("All clustering methods failed at depth=%d.", depth)
        return np.zeros(len(W), dtype=np.int32), [], {}

    best_name, best_labels, best_m = all_results[0]
    log.info(
        "Best @ depth=%d: %s  Q=%.4f  n_sc=%d  score=%.4f",
        depth, best_name, best_m["modularity"],
        best_m["n_clusters"], best_m["composite_score"],
    )

    recursive_info: RecursiveInfo = {}

    if depth < max_depth:
        for sc_id in np.unique(best_labels):
            mask = best_labels == sc_id
            sub_size = int(mask.sum())
            if sub_size < min_size_to_split:
                continue
            log.info(
                "  Recursing into subcluster %d (size=%d) @ depth=%d",
                sc_id, sub_size, depth + 1,
            )
            sub_idx = np.where(mask)[0]
            sub_W = W[np.ix_(sub_idx, sub_idx)]
            sub_tokens = [token_strings[i] for i in sub_idx]

            sub_results = run_all_methods(sub_W, seed=seed, run_agg=run_agg)
            if sub_results:
                _, sub_labels, sub_m = sub_results[0]
                log.info(
                    "    Depth-%d sub-%d: %s  Q=%.4f  n_sub=%d",
                    depth + 1, sc_id, sub_m["method"],
                    sub_m["modularity"], sub_m["n_clusters"],
                )
                recursive_info[int(sc_id)] = (sub_labels, sub_tokens, sub_W)

    return best_labels, all_results, recursive_info


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _palette(n: int):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    return [cmap(i / max(n, 1)) for i in range(n)]


def save_heatmap(
    W: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    """Two-panel heatmap: reordered adjacency + cluster-level aggregated weights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(W)
    order = np.argsort(labels, kind="stable")
    W_ord = W[np.ix_(order, order)]

    max_display = 600
    if N > max_display:
        idx = np.linspace(0, N - 1, max_display, dtype=int)
        W_disp = W_ord[np.ix_(idx, idx)]
    else:
        W_disp = W_ord

    n_sc = int(labels.max()) + 1
    W_agg = np.zeros((n_sc, n_sc), dtype=np.float32)
    for ci in range(n_sc):
        for cj in range(n_sc):
            mi, mj = labels == ci, labels == cj
            sub = W[np.ix_(mi, mj)]
            W_agg[ci, cj] = float(sub.mean()) if sub.size > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    pos_vals = W_disp[W_disp > 0]
    vmax = float(np.percentile(pos_vals, 99)) if len(pos_vals) > 0 else 1.0
    im0 = axes[0].imshow(W_disp, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[0].set_title(f"Reordered adjacency (N={N}, shown={len(W_disp)})")
    axes[0].set_xlabel("Token (by subcluster)")
    axes[0].set_ylabel("Token (by subcluster)")

    # Draw subcluster boundary lines on the (non-subsampled) heatmap
    if N <= max_display:
        sizes = np.bincount(labels[order])
        cumsum = np.cumsum(sizes)[:-1]
        for b in cumsum:
            axes[0].axhline(b - 0.5, color="white", lw=0.5, alpha=0.7)
            axes[0].axvline(b - 0.5, color="white", lw=0.5, alpha=0.7)

    pos_agg = W_agg[W_agg > 0]
    vmax_agg = float(np.percentile(pos_agg, 99)) if len(pos_agg) > 0 else 1.0
    im1 = axes[1].imshow(W_agg, aspect="auto", cmap="hot", vmin=0, vmax=vmax_agg)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    axes[1].set_title(f"Cluster-level mean weights ({n_sc} subclusters)")
    axes[1].set_xlabel("Subcluster ID")
    axes[1].set_ylabel("Subcluster ID")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap → %s", out_path)


def save_cluster_sizes(
    all_results: List[PartitionResult],
    out_path: str,
    target_cluster: int,
) -> None:
    """Grid of bar charts showing cluster-size distributions for every method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_results)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (name, labels, m) in zip(axes_flat, all_results):
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        colours = _palette(len(sizes))
        ax.bar(range(len(sizes)), sizes, color=colours, edgecolor="none", width=1.0)
        ax.set_title(
            f"{name}\nQ={m['modularity']:.3f}  score={m['composite_score']:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("Subcluster rank")
        ax.set_ylabel("Size")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Subcluster size distributions — Cluster {target_cluster}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cluster sizes → %s", out_path)


def save_eigen_spectrum(
    evals: np.ndarray,
    eig_info: dict,
    out_path: str,
    target_cluster: int,
) -> None:
    """Two-panel eigen-spectrum: sorted eigenvalue decay + cumulative variance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = np.abs(evals)
    cumvar = np.cumsum(lam) / max(lam.sum(), 1e-9)
    x = np.arange(1, len(lam) + 1)
    show = min(len(lam), 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Eigenvalue spectrum — Cluster {target_cluster} subgraph  "
        f"(PR={eig_info['participation_ratio']:.1f})",
        fontsize=12, fontweight="bold",
    )

    axes[0].semilogy(x[:show], lam[:show] + 1e-12, lw=1.5, color="steelblue")
    axes[0].axvline(
        eig_info["spectral_gap_location"], color="crimson", ls="--", lw=1.2,
        label=f"Gap @ rank {eig_info['spectral_gap_location']}",
    )
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Eigenvalue magnitude (log)")
    axes[0].set_title("Eigenvalue decay")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x[:show], cumvar[:show], lw=1.5, color="darkorange")
    axes[1].axhline(0.50, color="gray", ls=":", lw=1)
    axes[1].axhline(0.90, color="crimson", ls="--", lw=1, label="90% threshold")
    axes[1].axvline(eig_info["dims_for_90pct"], color="crimson", ls="--", lw=1)
    axes[1].set_xlabel("Number of eigenvalues")
    axes[1].set_ylabel("Cumulative spectral weight")
    axes[1].set_title(
        f"Cumulative variance  (dims@90%={eig_info['dims_for_90pct']})"
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved eigenvalue spectrum → %s", out_path)


def save_cluster_graph(
    W: np.ndarray,
    labels: np.ndarray,
    token_strings: List[str],
    out_path: str,
    target_cluster: int,
    max_nodes: int = 400,
    seed: int = 42,
) -> None:
    """Force-directed NetworkX layout coloured by subcluster assignment."""
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(W)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(N, size=max_nodes, replace=False) if N > max_nodes
           else np.arange(N))
    idx.sort()

    W_sub = W[np.ix_(idx, idx)]
    labels_sub = labels[idx]

    pos_vals = W_sub[W_sub > 0]
    threshold = float(np.percentile(pos_vals, 95)) if len(pos_vals) > 0 else 0.0

    G = nx.Graph()
    G.add_nodes_from(range(len(idx)))
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if W_sub[i, j] > threshold:
                G.add_edge(i, j, weight=float(W_sub[i, j]))

    n_sc = int(labels_sub.max()) + 1
    palette = _palette(n_sc)
    node_colors = [palette[int(labels_sub[n]) % n_sc] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 11))
    pos = nx.spring_layout(G, weight="weight", seed=seed, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=25, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.12, width=0.4,
                           edge_color="grey", ax=ax)

    # Label 2 representative nodes per subcluster
    label_dict: Dict[int, str] = {}
    shown: set = set()
    for sc_id in range(n_sc):
        sc_nodes = [n for n in G.nodes() if labels_sub[n] == sc_id and n not in shown]
        for node in sc_nodes[:2]:
            raw = token_strings[idx[node]].replace("\n", "\\n").strip()
            label_dict[node] = raw or "<SP>"
            shown.add(node)
    nx.draw_networkx_labels(G, pos, label_dict, font_size=6, ax=ax)

    ax.set_title(
        f"Token subgraph — Cluster {target_cluster}  "
        f"({len(idx)} nodes, top-5% edges, {n_sc} subclusters)",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cluster graph → %s", out_path)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def save_subcluster_mapping(
    best_labels: np.ndarray,
    local_indices: np.ndarray,
    freq_ids: Optional[np.ndarray],
    token_strings: List[str],
    target_cluster: int,
    best_name: str,
    out_path: str,
) -> None:
    """
    Save a JSON file that maps vocab_id → remapped subcluster_id.

    The remap is identical to the one used in save_subcluster_report
    (sorted by descending cluster size), so IDs agree across both files.
    """
    unique_labels = sorted(np.unique(best_labels).tolist())
    sizes = [(int((best_labels == lbl).sum()), lbl) for lbl in unique_labels]
    sizes.sort(reverse=True)
    remap = {old_lbl: new_id for new_id, (_, old_lbl) in enumerate(sizes)}

    vocab_to_subcluster: Dict[str, int] = {}
    member_vocab_ids: List[int] = []
    subcluster_labels_out: List[int] = []

    for i, local_idx in enumerate(local_indices):
        vocab_id = int(freq_ids[local_idx]) if freq_ids is not None else int(local_idx)
        sub_id   = remap[int(best_labels[i])]
        vocab_to_subcluster[str(vocab_id)] = sub_id
        member_vocab_ids.append(vocab_id)
        subcluster_labels_out.append(sub_id)

    mapping = {
        "target_cluster":     target_cluster,
        "n_subclusters":      len(unique_labels),
        "method":             best_name,
        "n_members":          len(member_vocab_ids),
        "vocab_to_subcluster": vocab_to_subcluster,
        "member_vocab_ids":   member_vocab_ids,
        "subcluster_labels":  subcluster_labels_out,
        "token_strings":      token_strings,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    log.info(
        "Saved subcluster mapping → %s  (%d members, %d subclusters)",
        out_path, len(member_vocab_ids), len(unique_labels),
    )


def save_subcluster_report(
    W: np.ndarray,
    best_labels: np.ndarray,
    token_strings: List[str],
    out_path: str,
    target_cluster: int,
    method_name: str,
    recursive_info: RecursiveInfo,
) -> None:
    """Write human-readable cluster7_subclusters.txt."""
    unique_labels = sorted(np.unique(best_labels).tolist())
    n_sc = len(unique_labels)

    # Stable remap 0..n_sc-1 sorted by descending cluster size
    sizes = [(int((best_labels == lbl).sum()), lbl) for lbl in unique_labels]
    sizes.sort(reverse=True)
    remap = {old_lbl: new_id for new_id, (_, old_lbl) in enumerate(sizes)}
    remapped = np.array([remap[l] for l in best_labels], dtype=np.int32)

    lines: List[str] = [
        f"Cluster {target_cluster} — Recursive Subclustering Report",
        f"Method: {method_name}",
        f"Found {n_sc} subclusters",
        "=" * 70,
        "",
    ]

    for new_id in range(n_sc):
        original_id = sizes[new_id][1]
        mask = remapped == new_id
        size = int(mask.sum())
        top_toks = top_tokens_by_centrality(mask, token_strings, W, top_n=20)
        clean = [t.replace("\n", "\\n").strip() or "<SP>" for t in top_toks]
        lines.append(f"Subcluster {new_id} (size={size})")
        lines.append("  " + " | ".join(clean))
        lines.append("")

        # Depth-2 subclusters
        if original_id in recursive_info:
            sub_labels, sub_tokens, sub_W = recursive_info[original_id]
            sub_unique = sorted(np.unique(sub_labels).tolist())
            sub_sizes = [(int((sub_labels == sl).sum()), sl) for sl in sub_unique]
            sub_sizes.sort(reverse=True)
            lines.append(
                f"  └─ Depth-2 ({len(sub_unique)} sub-subclusters):"
            )
            for sub_size, sub_id in sub_sizes:
                smask = sub_labels == sub_id
                stoks = top_tokens_by_centrality(smask, sub_tokens, sub_W, top_n=10)
                clean_s = [t.replace("\n", "\\n").strip() or "<SP>" for t in stoks]
                lines.append(
                    f"       Sub-{sub_id:2d} (size={sub_size:4d}): "
                    f"{' | '.join(clean_s)}"
                )
            lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Saved subcluster report → %s", out_path)


# ---------------------------------------------------------------------------
# Final summary printer
# ---------------------------------------------------------------------------

def print_final_report(
    all_results: List[PartitionResult],
    best_labels: np.ndarray,
    token_strings: List[str],
    W: np.ndarray,
    eig_info: dict,
    recursive_info: RecursiveInfo,
    target_cluster: int,
) -> None:
    sep = "=" * 66
    best_name, _, best_m = all_results[0]

    print(f"\n{sep}")
    print(f"  CLUSTER {target_cluster} RECURSIVE SUBCLUSTERING — FINAL REPORT")
    print(sep)

    print(f"\n1. Best partition method : {best_name}")
    print(f"2. Discovered subclusters: {best_m['n_clusters']}")
    print(f"3. Best modularity (Q)   : {best_m['modularity']:.4f}")
    print(f"   Intra/Inter ratio     : {best_m['intra_inter_ratio']:.2f}x")
    print(f"   Composite score       : {best_m['composite_score']:.4f}")

    print(f"\n4. All methods ranked by composite score:")
    for i, (name, _, m) in enumerate(all_results):
        mark = " ← BEST" if i == 0 else ""
        print(
            f"   [{i+1:2d}] {name:<26s}  Q={m['modularity']:.4f}  "
            f"n_sc={m['n_clusters']:3d}  score={m['composite_score']:.4f}{mark}"
        )

    print(f"\n5. Largest subclusters and token examples:")
    sizes = [(int((best_labels == l).sum()), l) for l in np.unique(best_labels)]
    sizes.sort(reverse=True)
    for rank, (size, sc_id) in enumerate(sizes[:12]):
        mask = best_labels == sc_id
        top = top_tokens_by_centrality(mask, token_strings, W, top_n=8)
        clean = [t.replace("\n", "\\n").strip() or "<SP>" for t in top]
        print(f"   SC-{sc_id:3d} (size={size:4d}): {' | '.join(clean)}")

    print(f"\n6. Eigenvalue analysis:")
    print(f"   Participation ratio  : {eig_info['participation_ratio']:.1f}")
    print(f"   Dims for 50% variance: {eig_info['dims_for_50pct']}")
    print(f"   Dims for 90% variance: {eig_info['dims_for_90pct']}")
    print(f"   Largest spectral gap : rank {eig_info['spectral_gap_location']}  "
          f"(size={eig_info['spectral_gap_size']:.4f})")

    print(f"\n7. Recursive depth-2 results:")
    if recursive_info:
        for sc_id, (sub_labels, _, _sub_W) in recursive_info.items():
            n_sub = len(np.unique(sub_labels))
            print(f"   Subcluster {sc_id} → {n_sub} deeper sub-subclusters")
    else:
        print("   None (depth=1 or no subcluster large enough to split)")

    # ---- Hierarchical structure verdict ----
    Q = best_m["modularity"]
    ratio = best_m["intra_inter_ratio"]
    is_modular = Q > 0.05 and ratio > 1.5
    is_hierarchical = bool(recursive_info) and is_modular

    print(f"\n8. Is Cluster {target_cluster} hierarchically structured?")
    if is_hierarchical:
        verdict = "YES"
        detail = (
            "Strong modularity at depth-1 with recursive sub-structure at depth-2.\n"
            f"   Cluster {target_cluster} encodes hierarchically organised semantic categories."
        )
    elif is_modular:
        verdict = "PARTIALLY"
        detail = (
            "Modular structure found at depth-1, but no subcluster was large\n"
            "   enough for meaningful depth-2 recursion."
        )
    else:
        verdict = "UNCERTAIN"
        detail = (
            f"Low modularity (Q={Q:.4f}) suggests Cluster {target_cluster} may be\n"
            "   a residual catch-all cluster rather than a coherent semantic region."
        )
    print(f"   {verdict} — {detail}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recursive subclustering of a target cluster from the interference graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph", required=True,
                        help="Adjacency matrix (.npy or .npz)")
    parser.add_argument("--clusters", required=True,
                        help="cluster_tokens JSON (cluster_id → [{token, token_id, freq}])")
    parser.add_argument("--target_cluster", type=int, default=7,
                        help="Cluster ID to recursively subcluster")
    parser.add_argument("--labels", default=None,
                        help="Louvain labels .npy (auto-detected from --graph if omitted)")
    parser.add_argument("--freq_ids", default=None,
                        help="frequent_token_ids .npy (auto-detected from --graph if omitted)")
    parser.add_argument("--model", default="gpt2-xl",
                        help="HuggingFace model name — used only for tokenizer string decoding")
    parser.add_argument("--output_dir", default="results",
                        help="Directory for output files")
    parser.add_argument("--recursive_depth", type=int, default=2,
                        help="Maximum recursion depth for splitting large subclusters")
    parser.add_argument("--min_size_to_split", type=int, default=200,
                        help="Minimum subcluster size to recurse into at depth+1")
    parser.add_argument("--no_agg", action="store_true",
                        help="Skip agglomerative clustering")
    parser.add_argument("--no_graph_plot", action="store_true",
                        help="Skip force-directed graph plot (can be slow for large N)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    C = args.target_cluster
    log.info("=" * 60)
    log.info("Recursive Subclustering — Cluster %d", C)
    log.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load inputs                                                       #
    # ------------------------------------------------------------------ #
    log.info("=== Loading inputs ===")
    W          = load_graph(args.graph)
    cluster_json = load_cluster_json(args.clusters)
    labels     = load_labels(args.labels, args.graph)
    freq_ids   = load_freq_ids(args.freq_ids, args.graph)

    if str(C) not in cluster_json:
        log.error(
            "Target cluster %d not in JSON. Available keys: %s",
            C, sorted(cluster_json.keys()),
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Extract cluster membership + token strings                        #
    # ------------------------------------------------------------------ #
    log.info("=== Extracting Cluster %d membership ===", C)
    local_indices, token_strings = extract_cluster_members(
        target_cluster=C,
        cluster_json=cluster_json,
        labels=labels,
        freq_ids=freq_ids,
        model_name=args.model,
        W_size=len(W),
    )
    log.info("Cluster %d: %d members", C, len(local_indices))

    if len(local_indices) < 5:
        log.error(
            "Only %d tokens found for cluster %d. "
            "Check that --labels / --freq_ids point to the correct files.",
            len(local_indices), C,
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3. Induce subgraph W7                                               #
    # ------------------------------------------------------------------ #
    log.info("=== Inducing subgraph ===")
    W7 = induce_subgraph(W, local_indices)

    # ------------------------------------------------------------------ #
    # 4. Eigenvalue analysis                                               #
    # ------------------------------------------------------------------ #
    log.info("=== Eigenvalue analysis ===")
    evals, eig_info = eigenvalue_analysis(W7)

    # ------------------------------------------------------------------ #
    # 5. Run recursive subclustering                                       #
    # ------------------------------------------------------------------ #
    log.info("=== Running subclustering (depth=%d) ===", args.recursive_depth)
    best_labels, all_results, recursive_info = recursive_subcluster(
        W=W7,
        token_strings=token_strings,
        depth=1,
        max_depth=args.recursive_depth,
        min_size_to_split=args.min_size_to_split,
        seed=args.seed,
        run_agg=not args.no_agg,
    )

    if not all_results:
        log.error("All clustering methods failed. Exiting.")
        sys.exit(1)

    best_name, _, best_m = all_results[0]

    # ------------------------------------------------------------------ #
    # 6. Save all outputs                                                  #
    # ------------------------------------------------------------------ #
    log.info("=== Saving outputs to %s/ ===", args.output_dir)

    report_path = os.path.join(args.output_dir, f"cluster{C}_subclusters.txt")
    save_subcluster_report(
        W=W7,
        best_labels=best_labels,
        token_strings=token_strings,
        out_path=report_path,
        target_cluster=C,
        method_name=best_name,
        recursive_info=recursive_info,
    )

    save_heatmap(
        W=W7,
        labels=best_labels,
        out_path=os.path.join(args.output_dir, f"heatmap_cluster{C}.png"),
        title=(
            f"Cluster {C} induced subgraph — {best_name}  "
            f"Q={best_m['modularity']:.4f}  n_sc={best_m['n_clusters']}"
        ),
    )

    save_cluster_sizes(
        all_results=all_results,
        out_path=os.path.join(args.output_dir, f"cluster_sizes_cluster{C}.png"),
        target_cluster=C,
    )

    save_eigen_spectrum(
        evals=evals,
        eig_info=eig_info,
        out_path=os.path.join(args.output_dir, f"eigen_spectrum_cluster{C}.png"),
        target_cluster=C,
    )

    if not args.no_graph_plot:
        save_cluster_graph(
            W=W7,
            labels=best_labels,
            token_strings=token_strings,
            out_path=os.path.join(args.output_dir, f"graph_cluster{C}.png"),
            target_cluster=C,
            seed=args.seed,
        )

    save_subcluster_mapping(
        best_labels=best_labels,
        local_indices=local_indices,
        freq_ids=freq_ids,
        token_strings=token_strings,
        target_cluster=C,
        best_name=best_name,
        out_path=os.path.join(args.output_dir, f"cluster{C}_subcluster_mapping.json"),
    )

    # ------------------------------------------------------------------ #
    # 7. Print final report                                                #
    # ------------------------------------------------------------------ #
    print_final_report(
        all_results=all_results,
        best_labels=best_labels,
        token_strings=token_strings,
        W=W7,
        eig_info=eig_info,
        recursive_info=recursive_info,
        target_cluster=C,
    )

    log.info("Done. Outputs in %s/", args.output_dir)


if __name__ == "__main__":
    main()
