"""
Community detection on the interference graph.

Three methods are run:
  (a) Louvain   — via networkx.algorithms.community.louvain_communities
                  (built into networkx >= 2.7, no extra package needed)
  (b) Spectral  — sklearn SpectralClustering on the precomputed affinity
  (c) K-Means   — on the top eigenvectors of the adjacency matrix

Metrics computed for each clustering:
  - modularity Q
  - number of clusters actually found
  - Shannon entropy of cluster-size distribution
  - mean intra-cluster weight
  - mean inter-cluster weight
  - intra/inter ratio
  - NMI between Louvain and Spectral (cluster consistency check)
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from utils import (
    ExperimentConfig,
    cluster_entropy,
    compute_modularity_numpy,
    intra_inter_weights,
    random_graph_modularity,
    save_json,
    setup_dirs,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Graph pre-processing
# ---------------------------------------------------------------------------

def prep_affinity(W: np.ndarray, clip_negative: bool = True) -> np.ndarray:
    """
    Prepare a weight matrix for clustering algorithms that expect a
    non-negative affinity matrix.

    For the gradient graph I the off-diagonal range is [−1, 1].
    We shift/clip to [0, 1] so that negative similarity is treated as
    zero affinity rather than anti-affinity.
    """
    A = W.copy().astype(np.float32)
    np.fill_diagonal(A, 0.0)   # remove self-loops for clustering
    if clip_negative:
        A = np.clip(A, 0.0, None)
    # Ensure symmetry (numerical noise can break it)
    A = (A + A.T) * 0.5
    return A


# ---------------------------------------------------------------------------
# Method (a) — Louvain
# ---------------------------------------------------------------------------

def louvain_clustering(
    W: np.ndarray,
    resolution: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Run Louvain community detection via NetworkX.
    Returns integer label array of shape (N,).
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    log = logging.getLogger("interference")
    A = prep_affinity(W)
    N = len(A)

    log.info("Building NetworkX graph (N=%d nodes) …", N)
    G = nx.from_numpy_array(A)

    log.info("Running Louvain (resolution=%.2f) …", resolution)
    communities = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)

    labels = np.zeros(N, dtype=np.int32)
    for c_idx, comm in enumerate(communities):
        for node in comm:
            labels[node] = c_idx

    log.info("Louvain found %d communities.", len(communities))
    return labels


# ---------------------------------------------------------------------------
# Method (b) — Spectral clustering
# ---------------------------------------------------------------------------

def spectral_clustering(
    W: np.ndarray,
    n_clusters: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    sklearn SpectralClustering with precomputed affinity matrix.
    """
    from sklearn.cluster import SpectralClustering

    log = logging.getLogger("interference")
    A = prep_affinity(W)

    log.info("Running Spectral clustering (n_clusters=%d) …", n_clusters)
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        n_init=5,
        random_state=seed,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(A).astype(np.int32)
    log.info("Spectral clustering done.")
    return labels


# ---------------------------------------------------------------------------
# Method (c) — K-Means on leading eigenvectors
# ---------------------------------------------------------------------------

def kmeans_spectral_embedding(
    W: np.ndarray,
    n_clusters: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute the top n_clusters eigenvectors of W and run K-Means.
    This is the classical spectral embedding + k-means pipeline, which is
    more interpretable than sklearn's internal variant.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    log = logging.getLogger("interference")
    A = prep_affinity(W)

    log.info("Computing top-%d eigenvectors of adjacency …", n_clusters)
    # Use eigh (symmetric) for efficiency; eigenvalues are real.
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    # eigh returns ascending order — take the last n_clusters (largest)
    V = eigenvectors[:, -n_clusters:]   # (N, n_clusters) — leading eigenvectors

    # L2-normalise rows before K-Means (standard for spectral clustering)
    V_norm = normalize(V, norm="l2", axis=1)

    log.info("Running K-Means (n_clusters=%d) on spectral embedding …", n_clusters)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(V_norm).astype(np.int32)

    log.info("K-Means done.  Inertia=%.2f", km.inertia_)
    return labels, eigenvalues


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(W: np.ndarray, labels: np.ndarray, method_name: str) -> dict:
    """Compute a full set of clustering metrics for one label assignment."""
    log = logging.getLogger("interference")
    n_clusters = int(labels.max()) + 1
    q = compute_modularity_numpy(W, labels)
    h = cluster_entropy(labels)
    intra, inter = intra_inter_weights(W, labels)
    ratio = intra / max(inter, 1e-9)

    sizes = sorted(np.bincount(labels).tolist(), reverse=True)

    metrics = {
        "method": method_name,
        "n_clusters": n_clusters,
        "modularity": round(q, 4),
        "cluster_entropy": round(h, 4),
        "intra_cluster_weight": round(intra, 6),
        "inter_cluster_weight": round(inter, 6),
        "intra_inter_ratio": round(ratio, 3),
        "cluster_sizes": sizes,
        "max_cluster_size": max(sizes),
        "min_cluster_size": min(sizes),
    }

    log.info(
        "[%-10s] Q=%.4f  n_clusters=%d  entropy=%.3f  intra/inter=%.2f",
        method_name,
        q, n_clusters, h, ratio,
    )
    return metrics


def nmi_score(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Normalised Mutual Information between two label arrays."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(labels_a, labels_b))


# ---------------------------------------------------------------------------
# Eigenvalue analysis
# ---------------------------------------------------------------------------

def eigenvalue_analysis(W: np.ndarray) -> dict:
    """
    Compute eigenvalues of the (symmetrically normalised) adjacency matrix
    and derive spectral metrics.
    """
    from utils import participation_ratio

    log = logging.getLogger("interference")
    A = prep_affinity(W)

    # Symmetric normalisation for clean spectral analysis
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]

    log.info("Computing eigenvalues of normalised adjacency (N=%d) …", len(A_norm))
    eigenvalues = np.linalg.eigvalsh(A_norm)   # ascending order, real
    eigenvalues_sorted = eigenvalues[::-1]     # descending — largest first

    # Variance explained by top-k
    total_abs = np.abs(eigenvalues_sorted).sum()
    cumvar = np.cumsum(np.abs(eigenvalues_sorted)) / max(total_abs, 1e-9)

    pr = participation_ratio(eigenvalues_sorted)

    # How many eigenvalues needed to explain 90% of total spectral weight
    dims_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    dims_50 = int(np.searchsorted(cumvar, 0.50)) + 1

    # Eigenvalue gap (difference between consecutive eigenvalues in top-20)
    top20 = eigenvalues_sorted[:20]
    gaps = np.diff(np.abs(top20))
    largest_gap_idx = int(np.argmax(np.abs(gaps))) + 1  # 1-indexed

    result = {
        "eigenvalues_top50": eigenvalues_sorted[:50].tolist(),
        "participation_ratio": round(pr, 2),
        "dims_for_50pct": dims_50,
        "dims_for_90pct": dims_90,
        "largest_eigenvalue": float(eigenvalues_sorted[0]),
        "spectral_gap_location": largest_gap_idx,
        "spectral_gap_size": float(abs(gaps[largest_gap_idx - 1])),
    }

    log.info(
        "Eigenvalue analysis: PR=%.1f  dims@50%%=%d  dims@90%%=%d  "
        "gap@idx=%d (size=%.4f)",
        pr, dims_50, dims_90, largest_gap_idx, abs(gaps[largest_gap_idx - 1]),
    )
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def cluster_and_analyse(
    W: np.ndarray,
    cfg: ExperimentConfig,
    layer_tag: str = "final",
    graph_type: str = "coactivation",
) -> dict:
    """
    Run all three clustering methods on matrix W, compute metrics,
    save results, and return a summary dict.
    """
    log = logging.getLogger("interference")
    log.info("=== Clustering: layer=%s  graph=%s  N=%d ===", layer_tag, graph_type, len(W))

    out_prefix = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}_{graph_type}")

    results: dict = {"layer_tag": layer_tag, "graph_type": graph_type}

    # ---- (a) Louvain ----
    labels_louvain = louvain_clustering(W, resolution=cfg.louvain_resolution, seed=cfg.seed)
    np.save(f"{out_prefix}_labels_louvain.npy", labels_louvain)
    results["louvain"] = compute_metrics(W, labels_louvain, "louvain")

    # ---- (b) Spectral ----
    labels_spectral = spectral_clustering(W, n_clusters=cfg.n_clusters, seed=cfg.seed)
    np.save(f"{out_prefix}_labels_spectral.npy", labels_spectral)
    results["spectral"] = compute_metrics(W, labels_spectral, "spectral")

    # ---- (c) K-Means on spectral embedding ----
    labels_kmeans, eigenvalues = kmeans_spectral_embedding(
        W, n_clusters=cfg.n_clusters, seed=cfg.seed
    )
    np.save(f"{out_prefix}_labels_kmeans.npy", labels_kmeans)
    results["kmeans"] = compute_metrics(W, labels_kmeans, "kmeans")

    # ---- Cross-method consistency ----
    nmi_lv_sp = nmi_score(labels_louvain, labels_spectral)
    nmi_lv_km = nmi_score(labels_louvain, labels_kmeans)
    nmi_sp_km = nmi_score(labels_spectral, labels_kmeans)
    results["nmi"] = {
        "louvain_spectral": round(nmi_lv_sp, 4),
        "louvain_kmeans": round(nmi_lv_km, 4),
        "spectral_kmeans": round(nmi_sp_km, 4),
    }
    log.info(
        "NMI — L/S=%.3f  L/K=%.3f  S/K=%.3f",
        nmi_lv_sp, nmi_lv_km, nmi_sp_km,
    )

    # ---- Eigenvalue analysis ----
    results["eigenvalues"] = eigenvalue_analysis(W)
    np.save(f"{out_prefix}_eigenvalues.npy",
            np.array(results["eigenvalues"]["eigenvalues_top50"]))

    # ---- Random baseline modularity ----
    q_random = random_graph_modularity(len(W), cfg.n_clusters)
    results["random_baseline_modularity"] = round(q_random, 4)
    results["louvain_beats_random"] = (
        results["louvain"]["modularity"] > q_random * 1.5
    )

    save_json(f"{out_prefix}_clustering_results.json", results)
    log.info("Clustering results saved to %s_clustering_results.json", out_prefix)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cluster interference graph.")
    parser.add_argument("--layer_tag", default="final")
    parser.add_argument("--graph", default="coactivation",
                        choices=["coactivation", "gradient"],
                        help="Which graph matrix to cluster.")
    args = parser.parse_args()

    setup_logging()
    cfg = ExperimentConfig()
    setup_dirs(cfg)

    prefix = os.path.join(cfg.graphs_dir, f"layer_{args.layer_tag}")

    if args.graph == "coactivation":
        W = np.load(f"{prefix}_W_norm.npy")
    else:
        W = np.load(f"{prefix}_I_gradient.npy")

    cluster_and_analyse(W, cfg, layer_tag=args.layer_tag, graph_type=args.graph)


if __name__ == "__main__":
    main()
