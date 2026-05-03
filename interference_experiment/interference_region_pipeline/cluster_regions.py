#!/usr/bin/env python3
"""
cluster_regions.py

Run community detection on the coactivation graph to find predictive
interference communities (token competition regions).

Tries Leiden (best, supports resolution) → python-louvain → spectral fallback.
Sweeps resolutions [0.5, 1.0, 1.5, 2.0]; selects best by modularity.
Also runs spectral and random-partition baselines.

Outputs (output_dir/):
    region_labels.npy           (V,) int32
    token_to_region.json        {vocab_id_str: region_id}
    region_to_tokens.json       {region_id_str: [vocab_ids]}
    cluster_tokens.txt          human-readable, top-20 tokens per region
    granularity_sweep.csv       metrics per resolution / method
    best_partition_metrics.json graph quality of selected partition
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from utils import setup_logging, set_seed, ensure_dirs, save_json, load_json

log = setup_logging("cluster_regions")


# ── Graph loading ─────────────────────────────────────────────────────────────

def load_graph(graphs_dir: str):
    W = sparse.load_npz(os.path.join(graphs_dir, "coactivation_W_norm.npz"))
    freq_ids = np.load(os.path.join(graphs_dir, "frequent_token_ids.npy"))
    return W, freq_ids


# ── igraph conversion (fast for 5k nodes) ────────────────────────────────────

def to_igraph(W: sparse.spmatrix):
    try:
        import igraph as ig
    except ImportError:
        return None
    W_coo = W.tocoo()
    mask = W_coo.row < W_coo.col   # upper triangle only (undirected)
    src  = W_coo.row[mask].tolist()
    dst  = W_coo.col[mask].tolist()
    wts  = W_coo.data[mask].tolist()
    G = ig.Graph(n=W.shape[0], edges=list(zip(src, dst)), directed=False)
    G.es["weight"] = wts
    return G


# ── Clustering methods ────────────────────────────────────────────────────────

def cluster_leiden(G, resolution: float, seed: int) -> Tuple[np.ndarray, float, str]:
    import leidenalg
    part = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = np.array(part.membership, dtype=np.int32)
    Q = part.modularity
    return labels, Q, f"leiden_res{resolution:.1f}"


def cluster_louvain_nx(W: sparse.spmatrix, resolution: float, seed: int) -> Tuple[np.ndarray, float, str]:
    import community as cl
    import networkx as nx
    G_nx = nx.from_scipy_sparse_array(W, edge_attribute="weight")
    partition = cl.best_partition(G_nx, resolution=resolution, weight="weight", random_state=seed)
    labels = np.array([partition[i] for i in range(W.shape[0])], dtype=np.int32)
    Q = cl.modularity(partition, G_nx, weight="weight")
    return labels, Q, f"louvain_res{resolution:.1f}"


def cluster_spectral(W: sparse.spmatrix, k: int, seed: int) -> Tuple[np.ndarray, float, str]:
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(
        n_clusters=k, affinity="precomputed",
        assign_labels="kmeans", random_state=seed, n_init=5,
    )
    labels = sc.fit_predict(W.toarray()).astype(np.int32)
    return labels, _modularity_nx(W, labels), f"spectral_k{k}"


def _modularity_nx(W: sparse.spmatrix, labels: np.ndarray) -> float:
    try:
        import networkx as nx
        G_nx = nx.from_scipy_sparse_array(W, edge_attribute="weight")
        n_cl = int(labels.max()) + 1
        comms = [set(np.where(labels == c)[0].tolist()) for c in range(n_cl) if (labels == c).any()]
        return float(nx.algorithms.community.modularity(G_nx, comms, weight="weight"))
    except Exception:
        return float("nan")


# ── Partition metrics ─────────────────────────────────────────────────────────

def partition_metrics(W: sparse.spmatrix, labels: np.ndarray, Q: float) -> dict:
    n_cl = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=n_cl)
    nonempty = sizes[sizes > 0]

    W_coo = W.tocoo()
    same = labels[W_coo.row] == labels[W_coo.col]
    intra_w = float(W_coo.data[same].mean())  if same.any()  else 0.0
    inter_w = float(W_coo.data[~same].mean()) if (~same).any() else 0.0

    p = nonempty / nonempty.sum()
    size_entropy = float(-np.sum(p * np.log(p + 1e-12)))

    return {
        "n_clusters":        n_cl,
        "modularity":        Q if not np.isnan(Q) else _modularity_nx(W, labels),
        "intra_w":           intra_w,
        "inter_w":           inter_w,
        "intra_inter_ratio": intra_w / (inter_w + 1e-12),
        "size_entropy":      size_entropy,
        "size_min":          int(nonempty.min()),
        "size_max":          int(nonempty.max()),
        "size_mean":         float(nonempty.mean()),
        "size_std":          float(nonempty.std()),
    }


def _score(metrics: dict) -> float:
    Q = metrics["modularity"]
    if np.isnan(Q) or Q < 0:
        return -1.0
    # Penalise near-collapse (one cluster holds > 50% of nodes)
    frac_max = metrics["size_max"] / (metrics["size_mean"] * metrics["n_clusters"] + 1e-8)
    if frac_max > 0.5:
        Q *= 0.5
    return float(Q)


# ── Partition saving ──────────────────────────────────────────────────────────

def save_partition(labels: np.ndarray, freq_ids: np.ndarray,
                   output_dir: str, tokenizer=None) -> None:
    V = len(freq_ids)
    n_cl = int(labels.max()) + 1

    token_to_region: Dict[str, int] = {str(int(freq_ids[i])): int(labels[i]) for i in range(V)}
    region_to_tokens: Dict[str, List[int]] = {}
    for i in range(V):
        c = str(int(labels[i]))
        region_to_tokens.setdefault(c, []).append(int(freq_ids[i]))

    np.save(os.path.join(output_dir, "region_labels.npy"), labels)
    save_json(token_to_region,  os.path.join(output_dir, "token_to_region.json"))
    save_json(region_to_tokens, os.path.join(output_dir, "region_to_tokens.json"))

    with open(os.path.join(output_dir, "cluster_tokens.txt"), "w") as f:
        for c_str, tids in sorted(region_to_tokens.items(), key=lambda x: int(x[0])):
            f.write(f"Region {c_str}  ({len(tids)} tokens)\n")
            shown = tids[:20]
            if tokenizer:
                decoded = [repr(tokenizer.decode([t])) for t in shown]
                f.write("  " + "  ".join(decoded) + "\n")
            else:
                f.write("  " + " ".join(str(t) for t in shown) + "\n")
            f.write("\n")

    log.info("Partition saved: %d regions over %d tokens", n_cl, V)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args) -> None:
    set_seed(args.seed)
    ensure_dirs(args.output_dir)
    rng = np.random.default_rng(args.seed)

    log.info("Loading graph from %s", args.graphs_dir)
    W, freq_ids = load_graph(args.graphs_dir)
    V = W.shape[0]
    log.info("Graph: %d nodes, %d nonzeros", V, W.nnz)

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        pass

    G_ig = to_igraph(W)
    has_leiden = False
    if G_ig is not None:
        try:
            import leidenalg  # noqa
            has_leiden = True
            log.info("Using Leiden algorithm")
        except ImportError:
            log.info("leidenalg not found; will try python-louvain")

    has_louvain_nx = False
    try:
        import community  # noqa
        has_louvain_nx = True
        if not has_leiden:
            log.info("Using python-louvain (community package)")
    except ImportError:
        pass

    if not has_leiden and not has_louvain_nx:
        log.warning("Neither leidenalg nor python-louvain found; only spectral will run")

    # ── Resolution sweep ──────────────────────────────────────────────────
    all_results: List[dict] = []
    best_score  = -1.0
    best_labels: Optional[np.ndarray] = None
    best_method = "none"

    for res in args.resolutions:
        log.info("Resolution %.1f ...", res)
        try:
            if has_leiden and G_ig is not None:
                labels, Q, method = cluster_leiden(G_ig, res, args.seed)
            elif has_louvain_nx:
                labels, Q, method = cluster_louvain_nx(W, res, args.seed)
            else:
                continue
        except Exception as e:
            log.warning("Clustering at res=%.1f failed: %s", res, e)
            continue

        m = partition_metrics(W, labels, Q)
        m.update({"resolution": res, "method": method})
        all_results.append(m)
        log.info("  n_cl=%d  Q=%.4f  intra/inter=%.2fx  sizes(min=%d, max=%d)",
                 m["n_clusters"], m["modularity"], m["intra_inter_ratio"],
                 m["size_min"], m["size_max"])

        sc = _score(m)
        if sc > best_score:
            best_score  = sc
            best_labels = labels.copy()
            best_method = method

    # ── Spectral baseline ─────────────────────────────────────────────────
    n_cl_ref = int(np.median([r["n_clusters"] for r in all_results])) if all_results else 16
    try:
        log.info("Spectral baseline (k=%d) ...", n_cl_ref)
        sp_labels, sp_Q, sp_method = cluster_spectral(W, n_cl_ref, args.seed)
        sp_m = partition_metrics(W, sp_labels, sp_Q)
        sp_m.update({"resolution": -1, "method": sp_method})
        all_results.append(sp_m)
        log.info("  spectral Q=%.4f", sp_m["modularity"])
    except Exception as e:
        log.warning("Spectral failed: %s", e)

    # ── Random baseline ───────────────────────────────────────────────────
    if best_labels is not None:
        n_rand = int(best_labels.max()) + 1
        rand_labels = rng.integers(0, n_rand, size=V, dtype=np.int32)
        rand_m = partition_metrics(W, rand_labels, float("nan"))
        rand_m.update({"resolution": -2, "method": "random"})
        all_results.append(rand_m)
        log.info("Random baseline Q=%.4f  (n_cl=%d)", rand_m["modularity"], rand_m["n_clusters"])

    # ── Save sweep CSV ────────────────────────────────────────────────────
    if all_results:
        fields = ["method", "resolution", "n_clusters", "modularity",
                  "intra_w", "inter_w", "intra_inter_ratio",
                  "size_entropy", "size_min", "size_max", "size_mean", "size_std"]
        csv_path = os.path.join(args.output_dir, "granularity_sweep.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in all_results:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
        log.info("Granularity sweep → %s", csv_path)

    if best_labels is None:
        log.error("No clustering succeeded.")
        sys.exit(1)

    log.info("Best partition: %s  (score=%.4f)", best_method, best_score)
    save_partition(best_labels, freq_ids, args.output_dir, tok)
    best_m = partition_metrics(W, best_labels, best_score)
    best_m["method"] = best_method
    save_json(best_m, os.path.join(args.output_dir, "best_partition_metrics.json"))


def parse_args():
    p = argparse.ArgumentParser(description="Cluster interference graph into token competition regions",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--graphs_dir",  default="interference_region_pipeline/graphs")
    p.add_argument("--output_dir",  default="interference_region_pipeline/results")
    p.add_argument("--model_name",  default="gpt2-xl")
    p.add_argument("--resolutions", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
