#!/usr/bin/env python3
"""
recursive_subcluster.py

For each large region (size > recursive_min_size), extract its subgraph
and recursively cluster to depth recursive_depth.

Stops recursing when:
  - cluster size < recursive_min_size
  - modularity gain < 0.05
  - fewer than 3 nodes remain

Outputs (output_dir/):
    region_tree.json          full hierarchy as nested dict
    region_tree_report.txt    human-readable, top-20 tokens per node
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from utils import setup_logging, set_seed, ensure_dirs, save_json, load_json

log = setup_logging("recursive_subcluster")


# ── Subgraph extraction ───────────────────────────────────────────────────────

def extract_subgraph(W: sparse.spmatrix, node_indices: np.ndarray) -> sparse.spmatrix:
    """Extract the induced subgraph on node_indices (subset of [0..V-1])."""
    return W[np.ix_(node_indices, node_indices)]


# ── Single-level clustering ───────────────────────────────────────────────────

def cluster_subgraph(W_sub: sparse.spmatrix, seed: int) -> Optional[np.ndarray]:
    """
    Attempt Leiden → python-louvain → spectral on a subgraph.
    Returns labels array or None on failure.
    """
    n = W_sub.shape[0]
    if n < 4:
        return None

    # Build igraph
    try:
        import igraph as ig
        W_coo = W_sub.tocoo()
        mask = W_coo.row < W_coo.col
        src  = W_coo.row[mask].tolist()
        dst  = W_coo.col[mask].tolist()
        wts  = W_coo.data[mask].tolist()
        if not src:
            return None
        G = ig.Graph(n=n, edges=list(zip(src, dst)), directed=False)
        G.es["weight"] = wts

        try:
            import leidenalg
            part = leidenalg.find_partition(
                G, leidenalg.ModularityVertexPartition,
                weights="weight", seed=seed,
            )
        except ImportError:
            part = G.community_multilevel(weights="weight")

        return np.array(part.membership, dtype=np.int32)

    except ImportError:
        pass

    # Fallback: python-louvain
    try:
        import community as cl
        import networkx as nx
        G_nx = nx.from_scipy_sparse_array(W_sub, edge_attribute="weight")
        partition = cl.best_partition(G_nx, weight="weight", random_state=seed)
        return np.array([partition[i] for i in range(n)], dtype=np.int32)
    except Exception:
        return None


def modularity_subgraph(W_sub: sparse.spmatrix, labels: np.ndarray) -> float:
    try:
        import networkx as nx
        G_nx = nx.from_scipy_sparse_array(W_sub, edge_attribute="weight")
        n_cl = int(labels.max()) + 1
        comms = [set(np.where(labels == c)[0].tolist()) for c in range(n_cl) if (labels == c).any()]
        return float(nx.algorithms.community.modularity(G_nx, comms, weight="weight"))
    except Exception:
        return 0.0


# ── Recursive tree building ───────────────────────────────────────────────────

def build_subtree(
    W: sparse.spmatrix,
    node_indices: np.ndarray,       # global indices into W
    freq_ids: np.ndarray,           # freq_ids[node_idx] = vocab_id
    depth: int,
    max_depth: int,
    min_size: int,
    min_modularity_gain: float,
    seed: int,
    tokenizer=None,
    region_id_prefix: str = "",
) -> dict:
    """
    Recursively cluster and return a tree node:
    {
      "vocab_ids": [...],
      "n_tokens": N,
      "modularity": Q,
      "top_tokens": ["token", ...],
      "children": {
        "region_0": {...},
        "region_1": {...},
        ...
      }
    }
    """
    vocab_ids = freq_ids[node_indices].tolist()
    top_tokens: List[str] = []
    if tokenizer:
        for vid in vocab_ids[:20]:
            try:
                top_tokens.append(repr(tokenizer.decode([int(vid)])))
            except Exception:
                top_tokens.append(str(vid))

    node = {
        "vocab_ids": [int(v) for v in vocab_ids],
        "n_tokens": len(vocab_ids),
        "top_tokens": top_tokens,
        "modularity": None,
        "children": {},
    }

    if depth >= max_depth or len(node_indices) < min_size:
        return node

    W_sub = extract_subgraph(W, node_indices)
    labels = cluster_subgraph(W_sub, seed)

    if labels is None:
        return node

    n_sub = int(labels.max()) + 1
    if n_sub <= 1:
        return node

    Q = modularity_subgraph(W_sub, labels)
    node["modularity"] = Q

    if Q < min_modularity_gain:
        log.debug("  Stopping at prefix=%s: Q=%.4f < threshold", region_id_prefix, Q)
        return node

    log.info("  %s  n=%d  n_sub=%d  Q=%.4f",
             region_id_prefix or "root", len(node_indices), n_sub, Q)

    for sub_id in range(n_sub):
        sub_local = np.where(labels == sub_id)[0]
        sub_global = node_indices[sub_local]
        if len(sub_global) < 2:
            continue
        child_prefix = f"{region_id_prefix}.{sub_id}" if region_id_prefix else str(sub_id)
        child = build_subtree(
            W, sub_global, freq_ids,
            depth + 1, max_depth, min_size,
            min_modularity_gain, seed, tokenizer,
            region_id_prefix=child_prefix,
        )
        node["children"][str(sub_id)] = child

    return node


def save_report(tree: dict, path: str, indent: int = 0) -> None:
    with open(path, "w", encoding="utf-8") as f:
        _write_node(f, tree["children"], indent=0)


def _write_node(f, children: dict, indent: int) -> None:
    prefix = "  " * indent
    for cid, node in sorted(children.items(), key=lambda x: int(x[0])):
        top = ", ".join(node.get("top_tokens", [])[:10])
        Q_str = f"  Q={node['modularity']:.4f}" if node.get("modularity") else ""
        f.write(f"{prefix}Region {cid}  ({node['n_tokens']} tokens){Q_str}\n")
        if top:
            f.write(f"{prefix}  {top}\n")
        if node.get("children"):
            _write_node(f, node["children"], indent + 1)
        f.write("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args) -> None:
    set_seed(args.seed)
    ensure_dirs(args.output_dir)

    log.info("Loading graph from %s", args.graphs_dir)
    W = sparse.load_npz(os.path.join(args.graphs_dir, "coactivation_W_norm.npz"))
    freq_ids = np.load(os.path.join(args.graphs_dir, "frequent_token_ids.npy"))
    labels = np.load(os.path.join(args.output_dir, "region_labels.npy"))
    V = len(freq_ids)

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        pass

    n_regions = int(labels.max()) + 1
    log.info("Building recursive hierarchy: %d top-level regions  (min_size=%d, max_depth=%d)",
             n_regions, args.recursive_min_size, args.recursive_depth)

    root = {"vocab_ids": freq_ids.tolist(), "n_tokens": V, "children": {}}
    for r in range(n_regions):
        region_nodes = np.where(labels == r)[0]
        if len(region_nodes) < args.recursive_min_size:
            # Small region: leaf node, no recursion
            node = build_subtree(
                W, region_nodes, freq_ids,
                depth=args.recursive_depth,   # depth >= max_depth → no recursion
                max_depth=args.recursive_depth,
                min_size=args.recursive_min_size,
                min_modularity_gain=args.min_modularity_gain,
                seed=args.seed,
                tokenizer=tok,
                region_id_prefix=str(r),
            )
        else:
            log.info("Subclustering region %d (n=%d) ...", r, len(region_nodes))
            node = build_subtree(
                W, region_nodes, freq_ids,
                depth=0,
                max_depth=args.recursive_depth,
                min_size=args.recursive_min_size,
                min_modularity_gain=args.min_modularity_gain,
                seed=args.seed,
                tokenizer=tok,
                region_id_prefix=str(r),
            )
        root["children"][str(r)] = node

    tree = {"root": root}
    save_json(tree, os.path.join(args.output_dir, "region_tree.json"))
    save_report(tree["root"], os.path.join(args.output_dir, "region_tree_report.txt"))
    log.info("Region tree saved → %s", args.output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Recursive subcluster large regions",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--graphs_dir",            default="interference_region_pipeline/graphs")
    p.add_argument("--output_dir",            default="interference_region_pipeline/results")
    p.add_argument("--model_name",            default="gpt2-xl")
    p.add_argument("--recursive_depth",       type=int,   default=2)
    p.add_argument("--recursive_min_size",    type=int,   default=200)
    p.add_argument("--min_modularity_gain",   type=float, default=0.05)
    p.add_argument("--seed",                  type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
