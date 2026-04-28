"""
Visualisation suite for the interference graph experiment.

Generates the following plots (all saved to cfg.results_dir):

  1. heatmap_{coactivation,gradient}.png
     Adjacency matrix reordered by cluster assignment.
     Two variants: full matrix (downsampled) and cluster-aggregated view.

  2. eigenvalue_spectrum_{coactivation,gradient}.png
     Sorted eigenvalues of the normalised adjacency + participation ratio.

  3. cluster_size_distribution_{coactivation,gradient}.png
     Cluster size histogram for each method.

  4. cluster_graph_{coactivation,gradient}.png  (optional)
     Force-directed NetworkX layout coloured by cluster.

  5. layer_comparison.png  (optional)
     Louvain modularity Q vs layer depth.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from utils import ExperimentConfig, load_json, setup_logging


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def _cluster_palette(n: int):
    """Return n visually distinct colours."""
    cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    return [cmap(i / n) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Adjacency heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    W: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    max_size: int = 500,
    vmax_percentile: float = 99.0,
) -> None:
    """
    Plot the weight matrix W reordered by cluster label.

    If N > max_size, the matrix is uniformly subsampled to max_size × max_size
    for display (preserving cluster order).

    Two panels:
      Left: full reordered matrix (subsampled)
      Right: cluster-level aggregate (C × C mean weights)
    """
    log = logging.getLogger("interference")
    N = len(W)
    order = np.argsort(labels)
    W_ord = W[np.ix_(order, order)]

    # Subsample if needed
    if N > max_size:
        idx = np.linspace(0, N - 1, max_size, dtype=int)
        W_ord = W_ord[np.ix_(idx, idx)]

    n_clusters = int(labels.max()) + 1
    # Cluster-level aggregate: mean weight between clusters
    W_agg = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    for ci in range(n_clusters):
        for cj in range(n_clusters):
            mask_i = labels == ci
            mask_j = labels == cj
            sub = W[np.ix_(mask_i, mask_j)]
            W_agg[ci, cj] = float(sub.mean()) if sub.size > 0 else 0.0

    vmax_full = float(np.percentile(np.abs(W_ord), vmax_percentile))
    vmax_agg = float(np.percentile(np.abs(W_agg), vmax_percentile))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: full reordered matrix
    axes[0].set_title(f"Reordered adjacency (N={N}, shown={len(W_ord)})")
    im0 = axes[0].imshow(W_ord, aspect="auto", cmap="viridis", vmin=0, vmax=vmax_full)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[0].set_xlabel("Token index (by cluster)")
    axes[0].set_ylabel("Token index (by cluster)")

    # Draw cluster boundaries
    sizes = np.bincount(labels[order]) if N <= max_size else None
    if sizes is not None:
        cumsum = np.cumsum(sizes)[:-1]
        for b in cumsum:
            axes[0].axhline(b - 0.5, color="white", lw=0.4, alpha=0.6)
            axes[0].axvline(b - 0.5, color="white", lw=0.4, alpha=0.6)

    # Panel 2: cluster aggregate
    axes[1].set_title(f"Cluster-level mean weights ({n_clusters} clusters)")
    im1 = axes[1].imshow(W_agg, aspect="auto", cmap="hot", vmin=0, vmax=vmax_agg)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("Cluster ID")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap → %s", out_path)


# ---------------------------------------------------------------------------
# 2. Eigenvalue spectrum
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    title: str,
    out_path: str,
    participation_ratio: Optional[float] = None,
    dims_90: Optional[int] = None,
) -> None:
    """
    Plot sorted eigenvalue magnitudes (log scale).
    Marks the 90%-variance knee and the participation ratio.
    """
    log = logging.getLogger("interference")
    lam = np.abs(eigenvalues)
    lam_sorted = np.sort(lam)[::-1]
    cumvar = np.cumsum(lam_sorted) / max(lam_sorted.sum(), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: raw sorted eigenvalues (log scale)
    x = np.arange(1, len(lam_sorted) + 1)
    axes[0].semilogy(x, lam_sorted + 1e-12, lw=1.5, color="steelblue")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Eigenvalue magnitude (log)")
    axes[0].set_title("Eigenvalue decay")
    axes[0].grid(True, alpha=0.3)

    if dims_90 is not None and dims_90 < len(lam_sorted):
        axes[0].axvline(dims_90, color="crimson", ls="--", lw=1.2,
                        label=f"90% knee (k={dims_90})")
        axes[0].legend(fontsize=9)

    if participation_ratio is not None:
        axes[0].set_title(
            f"Eigenvalue decay  (PR={participation_ratio:.1f})"
        )

    # Right: cumulative variance
    axes[1].plot(x, cumvar, lw=1.5, color="darkorange")
    axes[1].axhline(0.50, color="gray", ls=":", lw=1)
    axes[1].axhline(0.90, color="crimson", ls="--", lw=1, label="90% threshold")
    if dims_90 is not None:
        axes[1].axvline(dims_90, color="crimson", ls="--", lw=1)
    axes[1].set_xlim(1, min(len(lam_sorted), 200))
    axes[1].set_xlabel("Number of eigenvalues")
    axes[1].set_ylabel("Cumulative fraction of spectral weight")
    axes[1].set_title("Cumulative variance")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved eigenvalue spectrum → %s", out_path)


# ---------------------------------------------------------------------------
# 3. Cluster size distribution
# ---------------------------------------------------------------------------

def plot_cluster_sizes(
    labels_dict: Dict[str, np.ndarray],
    title: str,
    out_path: str,
) -> None:
    """
    Bar chart of cluster sizes for each method.  Methods are shown as
    separate subplots sorted by cluster size (largest first).
    """
    log = logging.getLogger("interference")
    n_methods = len(labels_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (name, labels) in zip(axes, labels_dict.items()):
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        colours = _cluster_palette(len(sizes))
        ax.bar(range(len(sizes)), sizes, color=colours, edgecolor="none", width=1.0)
        ax.set_title(f"{name}  ({len(sizes)} clusters)")
        ax.set_xlabel("Cluster rank (by size)")
        ax.set_ylabel("Size")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cluster size distribution → %s", out_path)


# ---------------------------------------------------------------------------
# 4. Cluster graph (NetworkX force-directed)
# ---------------------------------------------------------------------------

def plot_cluster_graph(
    W: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    max_nodes: int = 300,
    edge_threshold_percentile: float = 90.0,
) -> None:
    """
    Force-directed graph of a token subset coloured by cluster.
    Edges above the 90th-percentile weight are shown.
    Uses a random subset if N > max_nodes.
    """
    import networkx as nx

    log = logging.getLogger("interference")
    N = len(W)
    if N > max_nodes:
        idx = np.random.default_rng(42).choice(N, size=max_nodes, replace=False)
        idx.sort()
    else:
        idx = np.arange(N)

    W_sub = W[np.ix_(idx, idx)]
    labels_sub = labels[idx]

    threshold = np.percentile(W_sub[W_sub > 0], edge_threshold_percentile) if W_sub.max() > 0 else 0.0
    G = nx.Graph()
    G.add_nodes_from(range(len(idx)))
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if W_sub[i, j] > threshold:
                G.add_edge(i, j, weight=float(W_sub[i, j]))

    n_clusters = int(labels_sub.max()) + 1
    palette = _cluster_palette(n_clusters)
    node_colors = [palette[labels_sub[n]] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=20, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, edge_color="grey", ax=ax)
    ax.set_title(f"{title}  ({len(idx)} nodes, top-{100-edge_threshold_percentile:.0f}% edges)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cluster graph → %s", out_path)


# ---------------------------------------------------------------------------
# 5. Layer comparison
# ---------------------------------------------------------------------------

def plot_layer_comparison(
    layer_tags: List[str],
    modularity_values: List[float],
    out_path: str,
) -> None:
    """
    Line plot of Louvain modularity across layer depths.
    """
    log = logging.getLogger("interference")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layer_tags, modularity_values, marker="o", lw=2, color="steelblue",
            markersize=8, label="Louvain Q")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Modularity Q")
    ax.set_title("Interference graph modularity vs transformer layer depth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved layer comparison → %s", out_path)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def visualise_graph(
    W: np.ndarray,
    graph_type: str,      # "coactivation" or "gradient"
    layer_tag: str,
    cfg: ExperimentConfig,
    clustering_results: Optional[dict] = None,
    draw_cluster_graph: bool = True,
) -> None:
    """
    Run the full visualisation suite for one graph matrix.
    Loads clustering labels from disk if clustering_results not provided.
    """
    log = logging.getLogger("interference")
    res_dir = cfg.results_dir
    prefix = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}_{graph_type}")

    # Load clustering results from disk if not passed in
    if clustering_results is None:
        json_path = f"{prefix}_clustering_results.json"
        if os.path.exists(json_path):
            clustering_results = load_json(json_path)
        else:
            log.warning("No clustering results found at %s", json_path)
            return

    # Determine which label file to use for the primary heatmap (prefer Louvain)
    labels_path = f"{prefix}_labels_louvain.npy"
    if not os.path.exists(labels_path):
        labels_path = f"{prefix}_labels_spectral.npy"
    labels_primary = np.load(labels_path)

    # ---- 1. Heatmap ----
    heatmap_title = f"Interference graph heatmap — {graph_type} | layer={layer_tag}"
    plot_heatmap(
        W, labels_primary,
        title=heatmap_title,
        out_path=os.path.join(res_dir, f"heatmap_{graph_type}_{layer_tag}.png"),
    )

    # ---- 2. Eigenvalue spectrum ----
    eig_data = clustering_results.get("eigenvalues", {})
    eigenvalues = np.array(eig_data.get("eigenvalues_top50", []))
    pr = eig_data.get("participation_ratio")
    dims_90 = eig_data.get("dims_for_90pct")
    plot_eigenvalue_spectrum(
        eigenvalues,
        title=f"Eigenvalue spectrum — {graph_type} | layer={layer_tag}",
        out_path=os.path.join(res_dir, f"eigenvalue_spectrum_{graph_type}_{layer_tag}.png"),
        participation_ratio=pr,
        dims_90=dims_90,
    )

    # ---- 3. Cluster size distribution ----
    labels_dict = {}
    for method in ("louvain", "spectral", "kmeans"):
        p = f"{prefix}_labels_{method}.npy"
        if os.path.exists(p):
            labels_dict[method] = np.load(p)
    if labels_dict:
        plot_cluster_sizes(
            labels_dict,
            title=f"Cluster sizes — {graph_type} | layer={layer_tag}",
            out_path=os.path.join(res_dir, f"cluster_sizes_{graph_type}_{layer_tag}.png"),
        )

    # ---- 4. Cluster graph ----
    if draw_cluster_graph:
        plot_cluster_graph(
            W, labels_primary,
            title=f"Token cluster graph — {graph_type} | layer={layer_tag}",
            out_path=os.path.join(res_dir, f"cluster_graph_{graph_type}_{layer_tag}.png"),
        )


def visualise_layer_comparison(
    layer_tags: List[str],
    cfg: ExperimentConfig,
    graph_type: str = "coactivation",
) -> None:
    """Collect Louvain Q for each layer and draw the comparison plot."""
    q_values: List[float] = []
    for tag in layer_tags:
        path = os.path.join(
            cfg.graphs_dir, f"layer_{tag}_{graph_type}_clustering_results.json"
        )
        if os.path.exists(path):
            data = load_json(path)
            q_values.append(data.get("louvain", {}).get("modularity", float("nan")))
        else:
            q_values.append(float("nan"))

    plot_layer_comparison(
        layer_tags,
        q_values,
        out_path=os.path.join(cfg.results_dir, f"layer_comparison_{graph_type}.png"),
    )
