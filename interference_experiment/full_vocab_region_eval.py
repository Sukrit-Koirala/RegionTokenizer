#!/usr/bin/env python3
"""
full_vocab_region_eval.py

Scalable robustness test for predictive interference regions at 20k–50k vocab scale.
Uses sparse/approximate methods throughout; never builds a dense V×V matrix.

Stages (--stage):
  build_graph   — LM inference → sparse coactivation graph
  cluster       — KNN sparsify → community detection → recursive leaves
  cache_hidden  — collect hidden states + leaf labels
  eval          — probes, gold recall, logit locality, routed softmax, summary
  all           — run all stages in order

Usage:
  python full_vocab_region_eval.py --stage all \\
      --model_name gpt2-xl \\
      --vocab_subset_size 20000 \\
      --output_dir interference_experiment/full_vocab_region_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Optional[str] = None) -> logging.Logger:
    fmt = "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)
    return logging.getLogger("fvr")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-vocab interference region evaluation")

    # Model / data
    p.add_argument("--model_name", default="gpt2-xl")
    p.add_argument("--dataset_name", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--max_tokens", type=int, default=1_000_000)
    p.add_argument("--vocab_subset_size", type=int, default=20_000)
    p.add_argument("--top_k_logits", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--layers", default="0,4,8,12,16,20,24,47")
    p.add_argument("--output_dir",
                   default="interference_experiment/full_vocab_region_eval")
    p.add_argument("--seed", type=int, default=42)

    # Graph sparsification
    p.add_argument("--knn_edges", type=int, default=128,
                   help="Top-m neighbours to keep per node before clustering")

    # Clustering
    p.add_argument("--recursive_depth", type=int, default=2)
    p.add_argument("--recursive_min_size", type=int, default=300)
    p.add_argument("--min_modularity_gain", type=float, default=0.03)

    # Hidden-state cache
    p.add_argument("--n_probe_max", type=int, default=200_000,
                   help="Max prediction positions to cache hidden states for")

    # Probe training
    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_batch_size", type=int, default=2048)

    # Soft routing
    p.add_argument("--soft_routing_layer", type=int, default=47,
                   help="Layer to use for soft routing eval (default: 47)")

    # Stage control
    p.add_argument("--stage", default="all",
                   choices=["build_graph", "cluster", "cache_hidden", "eval",
                            "soft_routing_eval", "all"])

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_model_and_tokenizer(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log = logging.getLogger("fvr")
    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    log.info(f"Loading model: {model_name} fp16")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer


def load_corpus_tokens(
    dataset_name: str,
    dataset_config: str,
    tokenizer,
    max_tokens: int,
    seq_len: int,
    seed: int,
) -> np.ndarray:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    log = logging.getLogger("fvr")
    log.info(f"Loading corpus {dataset_name}/{dataset_config} …")
    ds = load_dataset(dataset_name, dataset_config, split="train")
    texts = [ex["text"] for ex in ds if ex.get("text", "").strip()]
    rng = random.Random(seed)
    rng.shuffle(texts)

    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        if len(all_ids) >= max_tokens + seq_len:
            break

    arr = np.array(all_ids[: max_tokens + seq_len], dtype=np.int32)
    log.info(f"Corpus: {len(arr):,} tokens")
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BUILD LARGE SPARSE COACTIVATION GRAPH
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(args: argparse.Namespace, device: torch.device, out_dir: str) -> None:
    log = logging.getLogger("fvr")
    log.info("=" * 60)
    log.info("STAGE: build_graph")
    log.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(args.model_name, device)
    corpus = load_corpus_tokens(
        args.dataset_name, args.dataset_config,
        tokenizer, args.max_tokens, args.seq_len, args.seed,
    )

    full_vocab_size = tokenizer.vocab_size
    V = args.vocab_subset_size
    K = args.top_k_logits

    # ── select vocab subset: top-V tokens by corpus unigram frequency ──
    log.info("Computing unigram frequencies …")
    freq_counts = np.bincount(
        corpus.clip(0, full_vocab_size - 1).astype(np.int64),
        minlength=full_vocab_size,
    )
    top_ids = np.argsort(freq_counts)[::-1][:V].astype(np.int32)
    top_ids_sorted = np.sort(top_ids)          # sorted for readability; order preserved
    token_freqs = freq_counts[top_ids_sorted].astype(np.float32)

    # vocab_id → sub-index (0..V-1), -1 if not in subset
    vocab_to_sub = np.full(full_vocab_size, -1, dtype=np.int32)
    for sub_i, tok_id in enumerate(top_ids_sorted):
        vocab_to_sub[tok_id] = sub_i

    log.info(
        f"Vocab subset: {V:,} tokens  "
        f"freq range [{token_freqs.min():.0f}, {token_freqs.max():.0f}]"
    )

    # ── accumulate coactivation edges ──
    # For each prediction position we take the top-K logit tokens, filter to the
    # vocab subset, then add a co-occurrence count for every (i,j) pair with i<j.
    # We buffer pairs as numpy arrays and flush to a running CSR every FLUSH steps.

    FLUSH_EVERY = 25_000   # prediction positions between flushes

    running: sp.csr_matrix = sp.csr_matrix((V, V), dtype=np.float32)
    buf_rows: list[np.ndarray] = []
    buf_cols: list[np.ndarray] = []
    n_positions = 0

    def flush_buffer() -> None:
        nonlocal running, buf_rows, buf_cols
        if not buf_rows:
            return
        r = np.concatenate(buf_rows)
        c = np.concatenate(buf_cols)
        ones = np.ones(len(r), dtype=np.float32)
        batch_sp = sp.coo_matrix((ones, (r, c)), shape=(V, V)).tocsr()
        running = running + batch_sp
        buf_rows.clear()
        buf_cols.clear()

    seq_len = args.seq_len
    batch_size = args.batch_size
    n_seqs = (args.max_tokens - seq_len) // seq_len
    all_starts = list(range(0, n_seqs * seq_len, seq_len))

    t0 = time.time()
    for batch_start in range(0, len(all_starts), batch_size):
        batch_starts_b = all_starts[batch_start: batch_start + batch_size]
        # seqs shape: (B, seq_len+1)
        seqs = np.stack([corpus[s: s + seq_len + 1] for s in batch_starts_b])
        input_ids = torch.from_numpy(seqs[:, :-1].astype(np.int64)).to(device)

        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits.float()           # (B, L, vocab_size)

        # top-K predicted token ids for each position
        topk_ids = torch.topk(logits, K, dim=-1).indices.cpu().numpy()   # (B, L, K)
        B, L, _ = topk_ids.shape

        for b in range(B):
            for pos in range(L - 1):   # pos+1 is the true next token
                raw_ids = topk_ids[b, pos]                    # (K,)
                sub_ids = vocab_to_sub[raw_ids]
                sub_ids = sub_ids[sub_ids >= 0]               # filter to subset
                m = len(sub_ids)
                if m < 2:
                    continue

                # All upper-triangle pairs via triu_indices
                ii, jj = np.triu_indices(m, k=1)
                buf_rows.append(sub_ids[ii])
                buf_cols.append(sub_ids[jj])
                n_positions += 1

        if n_positions % FLUSH_EVERY < batch_size * (L - 1):
            flush_buffer()

        if (batch_start // batch_size) % 200 == 0:
            pct = batch_start / max(len(all_starts), 1) * 100
            log.info(
                f"  {pct:5.1f}%  positions={n_positions:,}  "
                f"nnz={running.nnz:,}  elapsed={time.time()-t0:.0f}s"
            )

    flush_buffer()
    log.info(f"Graph built: {n_positions:,} positions  {running.nnz:,} raw edges")

    # ── symmetrize ──
    W = running + running.T
    W = W.tocsr()

    # ── PMI-style normalisation: W_norm[i,j] = W[i,j] / sqrt(freq_i*freq_j + eps) ──
    eps = 1.0
    freq_sqrt_inv = 1.0 / np.sqrt(token_freqs + eps)
    D_inv = sp.diags(freq_sqrt_inv.astype(np.float32))
    W_norm = (D_inv @ W @ D_inv).tocsr().astype(np.float32)

    # ── save ──
    np.save(os.path.join(out_dir, "frequent_token_ids.npy"), top_ids_sorted)
    np.save(os.path.join(out_dir, "token_freqs.npy"), token_freqs)
    sp.save_npz(os.path.join(out_dir, "coactivation_sparse.npz"), W.astype(np.float32))
    sp.save_npz(os.path.join(out_dir, "normalized_sparse.npz"), W_norm)

    log.info(f"Saved graph files to {out_dir}/")
    log.info(f"  frequent_token_ids.npy  {top_ids_sorted.shape}")
    log.info(f"  coactivation_sparse.npz nnz={W.nnz:,}")
    log.info(f"  normalized_sparse.npz   nnz={W_norm.nnz:,}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — KNN SPARSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def knn_sparsify(W: sp.csr_matrix, top_m: int) -> sp.csr_matrix:
    """Keep top_m neighbours per node by weight then take element-wise max to symmetrize."""
    log = logging.getLogger("fvr")
    V = W.shape[0]
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    data_out: list[np.ndarray] = []

    for i in range(V):
        row = W.getrow(i)
        if row.nnz == 0:
            continue
        idx = row.indices
        vals = row.data
        if len(vals) > top_m:
            keep = np.argpartition(vals, -top_m)[-top_m:]
        else:
            keep = np.arange(len(vals))
        rows_out.append(np.full(len(keep), i, dtype=np.int32))
        cols_out.append(idx[keep].astype(np.int32))
        data_out.append(vals[keep].astype(np.float32))

    G = sp.coo_matrix(
        (np.concatenate(data_out),
         (np.concatenate(rows_out), np.concatenate(cols_out))),
        shape=(V, V),
    ).tocsr()

    # Symmetrize via element-wise max so no edge is lost
    G_sym = G.maximum(G.T).tocsr().astype(np.float32)
    log.info(f"KNN graph (top_m={top_m}): {G_sym.nnz:,} edges after symmetrization")
    return G_sym


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — COMMUNITY DETECTION + RECURSIVE SUBCLUSTERING
# ──────────────────────────────────────────────────────────────────────────────

def _sparse_to_igraph(W: sp.csr_matrix):
    import igraph as ig
    W_coo = W.tocoo()
    mask = W_coo.row < W_coo.col          # upper triangle only (undirected)
    edges = list(zip(W_coo.row[mask].tolist(), W_coo.col[mask].tolist()))
    weights = W_coo.data[mask].tolist()
    g = ig.Graph(n=W.shape[0], edges=edges, directed=False)
    g.es["weight"] = weights
    return g


def _leiden(g, resolution: float, seed: int = 42):
    try:
        import leidenalg
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
        )
        return np.array(part.membership, dtype=np.int32), float(part.modularity)
    except Exception:
        return None, None


def _louvain_fallback(g):
    try:
        result = g.community_multilevel(weights="weight")
        return np.array(result.membership, dtype=np.int32), float(result.modularity)
    except Exception:
        return None, None


def _partition_stats(labels: np.ndarray, W: sp.csr_matrix) -> dict:
    n_clusters = int(labels.max()) + 1
    total = float(W.sum())
    if total == 0:
        return {"intra_frac": 0.0, "intra_inter_ratio": 0.0,
                "max_cluster_frac": 1.0, "n_clusters": n_clusters}

    W_coo = W.tocoo()
    same_mask = labels[W_coo.row] == labels[W_coo.col]
    intra = float(W_coo.data[same_mask].sum())
    intra_frac = intra / total
    ratio = intra_frac / (1.0 - intra_frac + 1e-9)
    sizes = np.bincount(labels, minlength=n_clusters)
    max_frac = float(sizes.max() / max(len(labels), 1))
    return {
        "intra_frac": intra_frac,
        "intra_inter_ratio": ratio,
        "max_cluster_frac": max_frac,
        "n_clusters": n_clusters,
    }


def _best_partition(
    candidates: list[tuple[np.ndarray, dict]]
) -> tuple[np.ndarray, dict]:
    scored = []
    for labels, meta in candidates:
        if labels is None:
            continue
        score = meta["intra_frac"] * 0.6 + (1.0 - meta["max_cluster_frac"]) * 0.4
        scored.append((score, labels, meta))
    if not scored:
        raise RuntimeError("All partitioning attempts failed — check igraph/leidenalg install")
    scored.sort(key=lambda x: -x[0])
    _, best_labels, best_meta = scored[0]
    return best_labels, best_meta


def _recursive_subcluster(
    node_indices: np.ndarray,  # global sub-indices (0..V-1)
    W_local: sp.csr_matrix,    # sub-graph with local indexing matching node_indices
    depth: int,
    max_depth: int,
    min_size: int,
    min_mod_gain: float,
    seed: int,
) -> dict:
    """Return a tree node dict: {vocab_ids, children}."""
    tree_node: dict = {"vocab_ids": node_indices.tolist(), "children": []}

    if depth >= max_depth or len(node_indices) < min_size:
        return tree_node

    try:
        g = _sparse_to_igraph(W_local)
    except Exception:
        return tree_node

    labels, mod = _leiden(g, resolution=1.0, seed=seed)
    if labels is None:
        labels, mod = _louvain_fallback(g)
    if labels is None or mod is None or mod < min_mod_gain:
        return tree_node

    n_parts = int(labels.max()) + 1
    if n_parts <= 1:
        return tree_node

    for part_id in range(n_parts):
        local_mask = np.where(labels == part_id)[0]
        child_global = node_indices[local_mask]
        if len(child_global) == 0:
            continue
        child_W = W_local[local_mask][:, local_mask]
        child_node = _recursive_subcluster(
            child_global, child_W,
            depth + 1, max_depth, min_size, min_mod_gain, seed,
        )
        tree_node["children"].append(child_node)

    return tree_node


def _flatten_leaves(tree_node: dict) -> list[list[int]]:
    """DFS: return leaf nodes (no children) as list-of-token-lists."""
    if not tree_node["children"]:
        return [tree_node["vocab_ids"]]
    leaves: list[list[int]] = []
    for child in tree_node["children"]:
        leaves.extend(_flatten_leaves(child))
    return leaves


def cluster(args: argparse.Namespace, out_dir: str) -> None:
    log = logging.getLogger("fvr")
    log.info("=" * 60)
    log.info("STAGE: cluster")
    log.info("=" * 60)

    W_norm = sp.load_npz(os.path.join(out_dir, "normalized_sparse.npz"))
    top_ids = np.load(os.path.join(out_dir, "frequent_token_ids.npy"))
    V = W_norm.shape[0]
    log.info(f"Loaded normalised graph: {V:,} nodes  {W_norm.nnz:,} edges")

    # ── Step 2: KNN sparsify ──
    log.info(f"KNN sparsifying (top_m={args.knn_edges}) …")
    t0 = time.time()
    G_knn = knn_sparsify(W_norm, args.knn_edges)
    log.info(f"KNN done in {time.time()-t0:.1f}s")
    sp.save_npz(os.path.join(out_dir, "sparse_knn_graph.npz"), G_knn)

    # ── Step 3: Multi-resolution community detection ──
    resolutions = [0.5, 1.0, 1.5, 2.0, 3.0]
    log.info(f"Building igraph ({V:,} nodes) …")
    try:
        g = _sparse_to_igraph(G_knn)
    except ImportError:
        raise ImportError("pip install igraph leidenalg")

    candidates: list[tuple[np.ndarray, dict]] = []
    for res in resolutions:
        labels, mod = _leiden(g, resolution=res, seed=args.seed)
        if labels is None:
            labels, mod = _louvain_fallback(g)
        if labels is None:
            log.warning(f"  res={res}: clustering failed")
            continue
        meta = _partition_stats(labels, G_knn)
        meta["resolution"] = res
        meta["raw_modularity"] = float(mod) if mod is not None else 0.0
        log.info(
            f"  res={res}: n_clusters={meta['n_clusters']:,}  "
            f"intra_frac={meta['intra_frac']:.4f}  ratio={meta['intra_inter_ratio']:.2f}  "
            f"max_frac={meta['max_cluster_frac']:.3f}  Q={meta['raw_modularity']:.4f}"
        )
        candidates.append((labels, meta))

    best_labels, best_meta = _best_partition(candidates)
    n_coarse = best_meta["n_clusters"]
    log.info(
        f"Selected: res={best_meta['resolution']}  "
        f"n_clusters={n_coarse}  intra_frac={best_meta['intra_frac']:.4f}"
    )

    # ── Recursive subclustering ──
    log.info(
        f"Recursive subclustering depth={args.recursive_depth}  "
        f"min_size={args.recursive_min_size}  min_mod_gain={args.min_modularity_gain} …"
    )
    region_tree: dict = {"vocab_ids": list(range(V)), "children": []}
    for coarse_id in range(n_coarse):
        sub_local = np.where(best_labels == coarse_id)[0].astype(np.int32)
        if len(sub_local) == 0:
            continue
        W_sub = G_knn[sub_local][:, sub_local]
        child_node = _recursive_subcluster(
            sub_local, W_sub,
            depth=1,
            max_depth=args.recursive_depth,
            min_size=args.recursive_min_size,
            min_mod_gain=args.min_modularity_gain,
            seed=args.seed,
        )
        region_tree["children"].append(child_node)

    # ── Flatten to leaf regions ──
    leaf_lists = _flatten_leaves(region_tree)
    n_leaves = len(leaf_lists)
    sizes = np.array([len(tl) for tl in leaf_lists], dtype=np.int32)
    covered = int(sizes.sum())

    log.info(f"Leaf regions: {n_leaves}  covered={covered}/{V} ({100*covered/V:.1f}%)")

    # Build lookup maps (use string keys for JSON)
    leaf_token_to_region: dict[str, int] = {}
    leaf_region_to_tokens: dict[str, list[int]] = {}
    for leaf_id, tok_list in enumerate(leaf_lists):
        leaf_region_to_tokens[str(leaf_id)] = tok_list
        for tok in tok_list:
            leaf_token_to_region[str(tok)] = leaf_id

    # ── Human-readable report ──
    report_lines = [
        f"Coarse regions:   {n_coarse}",
        f"Leaf regions:     {n_leaves}",
        f"Vocab covered:    {covered}/{V} ({100*covered/V:.1f}%)",
        f"Leaf size min:    {sizes.min()}",
        f"Leaf size median: {int(np.median(sizes))}",
        f"Leaf size mean:   {sizes.mean():.1f}",
        f"Leaf size max:    {sizes.max()}",
        "",
        "Largest 20 leaves:",
    ]
    for rank, li in enumerate(np.argsort(sizes)[::-1][:20]):
        report_lines.append(f"  rank {rank+1:2d}: leaf_id={li:5d}  size={sizes[li]}")

    report_str = "\n".join(report_lines)
    log.info("\n" + report_str)

    # ── Save ──
    with open(os.path.join(out_dir, "region_tree.json"), "w") as f:
        json.dump(region_tree, f)
    with open(os.path.join(out_dir, "leaf_token_to_region.json"), "w") as f:
        json.dump(leaf_token_to_region, f)
    with open(os.path.join(out_dir, "leaf_region_to_tokens.json"), "w") as f:
        json.dump(leaf_region_to_tokens, f)
    with open(os.path.join(out_dir, "leaf_region_report.txt"), "w") as f:
        f.write(report_str + "\n")

    log.info(f"Saved cluster artefacts to {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — CACHE HIDDEN STATES + LABELS
# ──────────────────────────────────────────────────────────────────────────────

def cache_hidden(args: argparse.Namespace, device: torch.device, out_dir: str) -> None:
    log = logging.getLogger("fvr")
    log.info("=" * 60)
    log.info("STAGE: cache_hidden")
    log.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(args.model_name, device)
    corpus = load_corpus_tokens(
        args.dataset_name, args.dataset_config,
        tokenizer, args.max_tokens, args.seq_len, args.seed,
    )

    top_ids = np.load(os.path.join(out_dir, "frequent_token_ids.npy"))
    V = len(top_ids)
    full_vocab_size = tokenizer.vocab_size

    vocab_to_sub = np.full(full_vocab_size, -1, dtype=np.int32)
    for sub_i, tok_id in enumerate(top_ids):
        vocab_to_sub[tok_id] = sub_i

    with open(os.path.join(out_dir, "leaf_token_to_region.json")) as f:
        leaf_token_to_region_str: dict[str, int] = json.load(f)

    # sub-index → leaf_id, -1 if not assigned
    sub_to_leaf = np.full(V, -1, dtype=np.int32)
    for tok_str, leaf_id in leaf_token_to_region_str.items():
        si = vocab_to_sub[int(tok_str)]
        if si >= 0:
            sub_to_leaf[si] = leaf_id

    target_layers = [int(x) for x in args.layers.split(",")]
    D = model.config.hidden_size
    n_probe_max = args.n_probe_max

    log.info(f"Collecting hidden states: layers={target_layers}  D={D}  max={n_probe_max:,}")

    # ── Shared hook cache ──
    _hcache: dict[int, torch.Tensor] = {}

    def _make_hook(l_idx: int):
        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            _hcache[l_idx] = h.detach().cpu().to(torch.float16)
        return _hook

    hooks = [
        model.transformer.h[l].register_forward_hook(_make_hook(l))
        for l in target_layers
    ]

    # Pre-allocate buffers
    H_buf: dict[int, list[np.ndarray]] = {l: [] for l in target_layers}
    gold_buf: list[int] = []
    leaf_buf: list[int] = []

    seq_len = args.seq_len
    batch_size = args.batch_size
    n_seqs = (args.max_tokens - seq_len) // seq_len
    all_starts = list(range(0, n_seqs * seq_len, seq_len))
    rng = random.Random(args.seed + 1)
    rng.shuffle(all_starts)

    n_collected = 0
    t0 = time.time()

    for batch_start in range(0, len(all_starts), batch_size):
        if n_collected >= n_probe_max:
            break

        batch_starts_b = all_starts[batch_start: batch_start + batch_size]
        seqs = np.stack([corpus[s: s + seq_len + 1] for s in batch_starts_b])
        input_ids = torch.from_numpy(seqs[:, :-1].astype(np.int64)).to(device)
        next_ids_np = seqs[:, 1:]     # (B, L) — ground-truth next tokens

        with torch.no_grad():
            model(input_ids)          # populates _hcache via hooks

        B, L = next_ids_np.shape

        for b in range(B):
            for pos in range(L - 1):  # predict next_ids_np[b, pos]
                gold_raw = int(next_ids_np[b, pos])
                if gold_raw >= full_vocab_size:
                    continue
                sub_idx = int(vocab_to_sub[gold_raw])
                if sub_idx < 0:
                    continue
                leaf_id = int(sub_to_leaf[sub_idx])
                if leaf_id < 0:
                    continue

                gold_buf.append(sub_idx)
                leaf_buf.append(leaf_id)
                for l in target_layers:
                    # _hcache[l]: (B, L, D) fp16
                    vec = _hcache[l][b, pos].numpy()   # (D,) fp16
                    H_buf[l].append(vec)

                n_collected += 1
                if n_collected >= n_probe_max:
                    break
            if n_collected >= n_probe_max:
                break

        if batch_start % (batch_size * 100) == 0:
            log.info(
                f"  Collected {n_collected:,}/{n_probe_max:,}  "
                f"elapsed={time.time()-t0:.0f}s"
            )

    for h in hooks:
        h.remove()

    n_total_positions = (len(corpus) - seq_len) * (seq_len - 1)
    coverage_pct = 100.0 * n_collected / max(n_total_positions, 1)
    log.info(f"Collected {n_collected:,} samples ({coverage_pct:.2f}% of positions)")

    gold_arr = np.array(gold_buf, dtype=np.int32)
    leaf_arr = np.array(leaf_buf, dtype=np.int32)

    np.save(os.path.join(out_dir, "gold_tokens.npy"), gold_arr)
    np.save(os.path.join(out_dir, "leaf_labels.npy"), leaf_arr)

    for l in target_layers:
        arr = np.stack(H_buf[l], axis=0)     # (N, D) float16
        path = os.path.join(out_dir, f"hidden_L{l:02d}.npy")
        np.save(path, arr)
        log.info(f"  hidden_L{l:02d}.npy  {arr.shape}")

    meta = {
        "layers": target_layers,
        "N_probe": n_collected,
        "D": D,
        "model_name": args.model_name,
        "top_k_logits": args.top_k_logits,
        "vocab_subset_size": args.vocab_subset_size,
    }
    with open(os.path.join(out_dir, "hidden_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Saved hidden states to {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5 — EVAL: LEAF ROUTER PROBES
# ──────────────────────────────────────────────────────────────────────────────

class _LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _train_probe(
    H: np.ndarray,          # (N, D) float16
    labels: np.ndarray,     # (N,) int32
    n_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[_LinearProbe, dict]:
    N, D = H.shape
    perm = np.random.permutation(N)
    split = int(0.7 * N)
    tr_idx, te_idx = perm[:split], perm[split:]

    H_tr = torch.from_numpy(H[tr_idx].astype(np.float32)).to(device)
    y_tr = torch.from_numpy(labels[tr_idx].astype(np.int64)).to(device)
    H_te = torch.from_numpy(H[te_idx].astype(np.float32)).to(device)
    y_te = torch.from_numpy(labels[te_idx].astype(np.int64)).to(device)

    probe = _LinearProbe(D, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    ce_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        probe.train()
        idx_perm = torch.randperm(len(H_tr), device=device)
        for i in range(0, len(H_tr), batch_size):
            idx = idx_perm[i: i + batch_size]
            loss = ce_fn(probe(H_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        test_logits = probe(H_te)         # (N_te, n_classes)
        ce_loss = float(ce_fn(test_logits, y_te).item())

        k_vals = [1, 4, 8, 16, 32]
        max_k = min(max(k_vals), n_classes)
        _, pred_topk = test_logits.topk(max_k, dim=1)
        y_col = y_te.unsqueeze(1)
        metrics: dict = {"ce": ce_loss}
        for k in k_vals:
            k_cap = min(k, n_classes)
            hits = (pred_topk[:, :k_cap] == y_col).any(dim=1).float().mean().item()
            metrics[f"top{k}"] = float(hits)

        metrics["random_top1"] = 1.0 / n_classes

        # frequency baseline: always predict most-common training class
        mode_class = int(torch.bincount(y_tr, minlength=n_classes).argmax().item())
        metrics["freq_baseline_top1"] = float((y_te == mode_class).float().mean().item())

    return probe, metrics


def eval_router_probes(
    out_dir: str,
    target_layers: list[int],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[int, tuple[_LinearProbe, dict]]:
    log = logging.getLogger("fvr")
    log.info("─" * 40)
    log.info("Step 5 — Leaf router probes")

    leaf_labels = np.load(os.path.join(out_dir, "leaf_labels.npy"))
    n_classes = int(leaf_labels.max()) + 1
    log.info(f"  {len(leaf_labels):,} samples  {n_classes} leaf classes")

    rows: list[dict] = []
    probes: dict[int, tuple[_LinearProbe, dict]] = {}

    for l in target_layers:
        hpath = os.path.join(out_dir, f"hidden_L{l:02d}.npy")
        if not os.path.exists(hpath):
            log.warning(f"  hidden_L{l:02d}.npy not found — skipping layer")
            continue
        H = np.load(hpath)
        probe, metrics = _train_probe(
            H, leaf_labels, n_classes, device,
            args.probe_epochs, args.probe_lr, args.probe_batch_size,
        )
        probes[l] = (probe, metrics)
        row = {"layer": l, **metrics}
        rows.append(row)
        log.info(
            f"  L{l:02d}: top1={metrics['top1']:.4f}  top4={metrics['top4']:.4f}  "
            f"top8={metrics['top8']:.4f}  top32={metrics['top32']:.4f}  "
            f"random={metrics['random_top1']:.6f}  ce={metrics['ce']:.4f}"
        )

    if rows:
        csv_path = os.path.join(out_dir, "leaf_router_metrics.csv")
        _write_csv(csv_path, rows)
        log.info(f"  Saved {csv_path}")

    return probes


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 6 — GOLD RECALL VS CANDIDATE SIZE
# ──────────────────────────────────────────────────────────────────────────────

def eval_gold_recall(
    out_dir: str,
    target_layers: list[int],
    probes: dict[int, tuple[_LinearProbe, dict]],
    leaf_region_to_tokens: dict[int, list[int]],
    leaf_labels: np.ndarray,
    gold_tokens: np.ndarray,
    device: torch.device,
    r_values: list[int],
) -> None:
    log = logging.getLogger("fvr")
    log.info("─" * 40)
    log.info("Step 6 — Gold recall vs candidate size")

    V = sum(len(v) for v in leaf_region_to_tokens.values())
    n_leaves_total = len(leaf_region_to_tokens)
    N = len(leaf_labels)
    rows: list[dict] = []

    for l in target_layers:
        if l not in probes:
            continue
        probe, _ = probes[l]
        H = np.load(os.path.join(out_dir, f"hidden_L{l:02d}.npy"))

        probe.eval()
        BATCH = 4096
        probe_logits_list: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, N, BATCH):
                h = torch.from_numpy(H[i:i+BATCH].astype(np.float32)).to(device)
                probe_logits_list.append(probe(h).cpu())
        probe_logits = torch.cat(probe_logits_list, dim=0)   # (N, n_classes)

        for r in r_values:
            r_cap = min(r, probe_logits.shape[1])
            _, top_r = probe_logits.topk(r_cap, dim=1)   # (N, r)
            top_r_np = top_r.numpy()

            recalls, cand_sizes = [], []
            for i in range(N):
                gold_sub = int(gold_tokens[i])
                cand: set[int] = set()
                for lid in top_r_np[i]:
                    cand.update(leaf_region_to_tokens.get(lid, []))
                cand_sizes.append(len(cand))
                recalls.append(1 if gold_sub in cand else 0)

            recall = float(np.mean(recalls))
            avg_cand = float(np.mean(cand_sizes))
            rows.append({
                "layer": l,
                "r": r,
                "recall": recall,
                "avg_cand_size": avg_cand,
                "median_cand_size": float(np.median(cand_sizes)),
                "cand_fraction": avg_cand / V if V else 0.0,
                "failure_rate": 1.0 - recall,
            })
            log.info(
                f"  L{l:02d} r={r:3d}: recall={recall:.4f}  "
                f"cand={avg_cand:.0f} ({avg_cand/V:.3f})  fail={1-recall:.4f}"
            )

    # ── Random leaf baseline (pick r leaves uniformly at random) ──
    all_leaf_ids = list(leaf_region_to_tokens.keys())
    log.info("  Random leaf baseline …")
    for r in r_values:
        rand_recalls, rand_cands = [], []
        for i in range(N):
            gold_sub = int(gold_tokens[i])
            picked = random.sample(all_leaf_ids, min(r, n_leaves_total))
            cand: set[int] = set()
            for lid in picked:
                cand.update(leaf_region_to_tokens.get(lid, []))
            rand_cands.append(len(cand))
            rand_recalls.append(1 if gold_sub in cand else 0)
        avg_cand = float(np.mean(rand_cands))
        recall = float(np.mean(rand_recalls))
        rows.append({
            "layer": "random_leaf",
            "r": r,
            "recall": recall,
            "avg_cand_size": avg_cand,
            "median_cand_size": float(np.median(rand_cands)),
            "cand_fraction": avg_cand / V if V else 0.0,
            "failure_rate": 1.0 - recall,
        })

    if rows:
        csv_path = os.path.join(out_dir, "leaf_gold_recall.csv")
        _write_csv(csv_path, rows)
        log.info(f"  Saved {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 7 — LOGIT LOCALITY
# ──────────────────────────────────────────────────────────────────────────────

def eval_logit_locality(
    out_dir: str,
    final_layer: int,
    leaf_region_to_tokens: dict[int, list[int]],
    leaf_labels: np.ndarray,
    gold_tokens: np.ndarray,
    model,
    top_ids: np.ndarray,
    logit_ks: list[int],
    n_sample: int = 10_000,
) -> None:
    log = logging.getLogger("fvr")
    log.info("─" * 40)
    log.info("Step 7 — Logit locality")

    V = len(top_ids)

    # sub-index → leaf_id lookup array
    max_sub = max(max(tl) for tl in leaf_region_to_tokens.values() if tl) + 1
    sub_to_leaf = np.full(max_sub, -1, dtype=np.int32)
    for lid, toks in leaf_region_to_tokens.items():
        for t in toks:
            sub_to_leaf[t] = lid

    # LM head parameters (apply on CPU to avoid OOM at V=50k)
    wte_sub = model.transformer.wte.weight[top_ids].detach().cpu().float()  # (V, D)
    ln_f = model.transformer.ln_f.cpu().float()

    H = np.load(os.path.join(out_dir, f"hidden_L{final_layer:02d}.npy"))
    N = len(H)
    sample_idx = np.random.choice(N, min(n_sample, N), replace=False)
    H_sample = H[sample_idx]                # (n_sample, D) fp16
    gold_sample = gold_tokens[sample_idx]   # (n_sample,) sub-indices

    rows: list[dict] = []
    BATCH = 256

    for K in logit_ks:
        top1_same_list, gold_same_list, n_unique_list, ent_list = [], [], [], []

        for i in range(0, len(sample_idx), BATCH):
            h_b = torch.from_numpy(H_sample[i:i+BATCH].astype(np.float32))

            with torch.no_grad():
                h_ln = ln_f(h_b)                      # (B, D)
                logits_sub = h_ln @ wte_sub.T          # (B, V)

            k_cap = min(K, V)
            topK_sub_idx = logits_sub.topk(k_cap, dim=1).indices.numpy()  # (B, k)

            for b_i in range(topK_sub_idx.shape[0]):
                topK = topK_sub_idx[b_i]
                top1_sub = int(topK[0])
                gold_sub = int(gold_sample[i + b_i])

                top1_leaf = int(sub_to_leaf[top1_sub]) if top1_sub < len(sub_to_leaf) else -1
                gold_leaf = int(sub_to_leaf[gold_sub]) if gold_sub < len(sub_to_leaf) else -1

                topK_leaves = [
                    int(sub_to_leaf[t]) if t < len(sub_to_leaf) else -1
                    for t in topK
                ]
                valid = [lv for lv in topK_leaves if lv >= 0]
                n_valid = len(valid) or 1

                top1_same = sum(1 for lv in valid if lv == top1_leaf) / n_valid
                gold_same = sum(1 for lv in valid if lv == gold_leaf) / n_valid
                n_unique = len(set(valid))

                cnt = Counter(valid)
                total = sum(cnt.values())
                ent = -sum(
                    (v / total) * math.log2(v / total + 1e-12)
                    for v in cnt.values()
                ) if cnt else 0.0

                top1_same_list.append(top1_same)
                gold_same_list.append(gold_same)
                n_unique_list.append(n_unique)
                ent_list.append(ent)

        rows.append({
            "K": K,
            "frac_topK_same_leaf_as_top1": float(np.mean(top1_same_list)),
            "frac_topK_same_leaf_as_gold": float(np.mean(gold_same_list)),
            "avg_unique_leaves_in_topK": float(np.mean(n_unique_list)),
            "avg_leaf_entropy_bits": float(np.mean(ent_list)),
        })
        log.info(
            f"  K={K:4d}: same_top1={rows[-1]['frac_topK_same_leaf_as_top1']:.4f}  "
            f"same_gold={rows[-1]['frac_topK_same_leaf_as_gold']:.4f}  "
            f"unique_leaves={rows[-1]['avg_unique_leaves_in_topK']:.2f}  "
            f"entropy={rows[-1]['avg_leaf_entropy_bits']:.3f} bits"
        )

    csv_path = os.path.join(out_dir, "leaf_logit_locality.csv")
    _write_csv(csv_path, rows)
    log.info(f"  Saved {csv_path}")

    # Move ln_f back to GPU if needed by later eval steps
    ln_f.to(next(model.parameters()).device)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 8 — ROUTED SOFTMAX SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def eval_routed_softmax(
    out_dir: str,
    final_layer: int,
    probes: dict[int, tuple[_LinearProbe, dict]],
    leaf_region_to_tokens: dict[int, list[int]],
    leaf_labels: np.ndarray,
    gold_tokens: np.ndarray,
    model,
    top_ids: np.ndarray,
    device: torch.device,
    r_values: list[int],
) -> None:
    log = logging.getLogger("fvr")
    log.info("─" * 40)
    log.info("Step 8 — Routed softmax simulation")

    if final_layer not in probes:
        log.warning(f"  No probe for layer {final_layer} — skipping routed softmax")
        return

    probe, _ = probes[final_layer]
    V = len(top_ids)
    N = len(leaf_labels)

    # LM head on CPU (avoids OOM for large V)
    wte_sub = model.transformer.wte.weight[top_ids].detach().cpu().float()  # (V, D)
    ln_f = model.transformer.ln_f.cpu().float()

    # sub_idx → leaf_id
    sub_to_leaf = np.full(V, -1, dtype=np.int32)
    for lid, toks in leaf_region_to_tokens.items():
        for t in toks:
            if t < V:
                sub_to_leaf[t] = lid

    # leaf_id → sorted token array for fast mask construction
    leaf_to_arr: dict[int, np.ndarray] = {
        lid: np.array(toks, dtype=np.int32)
        for lid, toks in leaf_region_to_tokens.items()
    }

    H = np.load(os.path.join(out_dir, f"hidden_L{final_layer:02d}.npy"))

    BATCH = 128
    rows: list[dict] = []

    for r in r_values:
        full_nlls: list[float] = []
        cond_nlls: list[float] = []
        fail_flags: list[int] = []

        r_cap = min(r, len(leaf_region_to_tokens))

        for i in range(0, N, BATCH):
            h_b = torch.from_numpy(H[i:i+BATCH].astype(np.float32))

            # Full logits over vocab subset (CPU)
            with torch.no_grad():
                h_ln = ln_f(h_b)                     # (B, D)
                logits_full = h_ln @ wte_sub.T        # (B, V)

            # Probe logits (GPU then CPU)
            probe.eval()
            with torch.no_grad():
                probe_out = probe(h_b.to(device)).cpu()
            _, top_r_leaves = probe_out.topk(r_cap, dim=1)   # (B, r_cap)

            gold_subs_b = gold_tokens[i:i+BATCH]

            for b_i in range(h_b.shape[0]):
                gold_sub = int(gold_subs_b[b_i])
                if gold_sub >= V:
                    continue

                predicted_leaves = top_r_leaves[b_i].tolist()

                # Build boolean mask over V tokens
                cand_mask = np.zeros(V, dtype=bool)
                for lid in predicted_leaves:
                    arr = leaf_to_arr.get(lid)
                    if arr is not None:
                        cand_mask[arr] = True

                gold_in_cand = bool(cand_mask[gold_sub])

                lg = logits_full[b_i]   # (V,)

                # Full NLL
                full_lp = F.log_softmax(lg, dim=0)
                full_nlls.append(-float(full_lp[gold_sub].item()))

                if not gold_in_cand:
                    fail_flags.append(1)
                    continue
                fail_flags.append(0)

                # Routed NLL — conditional on gold being in candidate set
                routed_lg = lg.clone()
                routed_lg[~torch.from_numpy(cand_mask)] = -1e9
                routed_lp = F.log_softmax(routed_lg, dim=0)
                cond_nlls.append(-float(routed_lp[gold_sub].item()))

        failure_rate = float(np.mean(fail_flags)) if fail_flags else 1.0
        full_nll = float(np.nanmean(full_nlls)) if full_nlls else float("nan")
        full_ppl = math.exp(min(full_nll, 100)) if not math.isnan(full_nll) else float("nan")

        cond_nll = float(np.mean(cond_nlls)) if cond_nlls else float("nan")
        cond_ppl = math.exp(min(cond_nll, 100)) if not math.isnan(cond_nll) else float("nan")

        # Effective NLL: penalise failures with log(V) — do NOT ignore them
        penalty = math.log(V)
        eff_nll = (
            (1.0 - failure_rate) * cond_nll + failure_rate * penalty
            if cond_nlls else float("nan")
        )

        rows.append({
            "r": r,
            "full_nll": full_nll,
            "full_ppl": full_ppl,
            "routed_cond_nll": cond_nll,
            "routed_cond_ppl": cond_ppl,
            "failure_rate": failure_rate,
            "effective_nll": eff_nll,
        })
        log.info(
            f"  r={r:3d}: full_nll={full_nll:.4f}  "
            f"cond_nll={cond_nll:.4f}  fail={failure_rate:.4f}  eff_nll={eff_nll:.4f}"
        )

    # Move ln_f back
    ln_f.to(next(model.parameters()).device)

    csv_path = os.path.join(out_dir, "leaf_routed_softmax.csv")
    _write_csv(csv_path, rows)
    log.info(f"  Saved {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 9 — RANDOM BASELINE (same leaf-size distribution)
# ──────────────────────────────────────────────────────────────────────────────

def eval_random_baseline(
    out_dir: str,
    leaf_region_to_tokens: dict[int, list[int]],
    leaf_labels: np.ndarray,
    gold_tokens: np.ndarray,
    r_values: list[int],
) -> None:
    log = logging.getLogger("fvr")
    log.info("─" * 40)
    log.info("Step 9 — Random partition baseline")

    V = sum(len(v) for v in leaf_region_to_tokens.values())
    leaf_sizes = [len(v) for v in leaf_region_to_tokens.values()]

    # Shuffle sub-indices and re-partition with same size distribution
    all_sub_ids = list(range(V))
    random.shuffle(all_sub_ids)
    rand_leaf_to_tokens: dict[int, list[int]] = {}
    ptr = 0
    for lid, sz in enumerate(leaf_sizes):
        rand_leaf_to_tokens[lid] = all_sub_ids[ptr: ptr + sz]
        ptr += sz

    n_leaves = len(rand_leaf_to_tokens)
    all_leaf_ids = list(rand_leaf_to_tokens.keys())
    N = len(leaf_labels)
    rows: list[dict] = []

    for r in r_values:
        recalls, cand_sizes = [], []
        for i in range(N):
            gold_sub = int(gold_tokens[i])
            picked = random.sample(all_leaf_ids, min(r, n_leaves))
            cand: set[int] = set()
            for lid in picked:
                cand.update(rand_leaf_to_tokens.get(lid, []))
            cand_sizes.append(len(cand))
            recalls.append(1 if gold_sub in cand else 0)

        avg_cand = float(np.mean(cand_sizes))
        rows.append({
            "r": r,
            "recall": float(np.mean(recalls)),
            "avg_cand_size": avg_cand,
            "median_cand_size": float(np.median(cand_sizes)),
            "cand_fraction": avg_cand / V if V else 0.0,
            "failure_rate": 1.0 - float(np.mean(recalls)),
        })
        log.info(
            f"  r={r:3d}: recall={rows[-1]['recall']:.4f}  "
            f"cand={avg_cand:.0f} ({rows[-1]['cand_fraction']:.3f})"
        )

    csv_path = os.path.join(out_dir, "random_baseline.csv")
    _write_csv(csv_path, rows)
    log.info(f"  Saved {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 10 — SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def write_summary(
    out_dir: str,
    leaf_region_to_tokens: dict[int, list[int]],
    target_layers: list[int],
    r_values: list[int],
) -> None:
    log = logging.getLogger("fvr")
    log.info("─" * 40)
    log.info("Step 10 — Summary")

    def _load(name: str) -> list[dict]:
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return list(csv.DictReader(f))

    recall_rows = _load("leaf_gold_recall.csv")
    probe_rows = _load("leaf_router_metrics.csv")
    rand_rows = _load("random_baseline.csv")
    softmax_rows = _load("leaf_routed_softmax.csv")

    final_layer = max(target_layers)
    V = sum(len(v) for v in leaf_region_to_tokens.values())
    n_leaves = len(leaf_region_to_tokens)
    sizes = sorted(len(v) for v in leaf_region_to_tokens.values())
    median_sz = sizes[len(sizes) // 2] if sizes else 0
    mean_sz = sum(sizes) / len(sizes) if sizes else 0.0

    # ── parse CSVs ──
    final_recall: dict[int, float] = {
        int(r["r"]): float(r["recall"])
        for r in recall_rows
        if str(r["layer"]) == str(final_layer)
    }
    final_cand_frac: dict[int, float] = {
        int(r["r"]): float(r["cand_fraction"])
        for r in recall_rows
        if str(r["layer"]) == str(final_layer)
    }
    rand_recall: dict[int, float] = {
        int(r["r"]): float(r["recall"])
        for r in rand_rows
    }
    probe_final = {
        k: float(v)
        for r in probe_rows
        if int(r["layer"]) == final_layer
        for k, v in r.items()
        if k != "layer"
    }

    lines: list[str] = [
        "# Full-Vocab Interference Region Evaluation — Summary",
        "",
        f"**Model:** {os.path.basename(out_dir)}",
        f"**Vocab subset size:** {V:,}",
        f"**Leaf regions:** {n_leaves}",
        f"**Leaf size:** min={min(sizes) if sizes else 0}  "
        f"median={median_sz}  mean={mean_sz:.1f}  max={max(sizes) if sizes else 0}",
        f"**Layers evaluated:** {target_layers}",
        "",
        "---",
        "",
        "## Q1 — Do hierarchical routing regions form at large vocab scale?",
    ]
    if n_leaves > 5 and median_sz < 2000:
        lines.append(
            f"**YES** — {n_leaves} leaf regions formed with median size {median_sz}."
        )
    else:
        lines.append(
            f"**UNCLEAR** — {n_leaves} leaves, median size {median_sz}. "
            f"Check clustering quality."
        )

    lines += [
        "",
        "## Q2 — How many leaves are produced?",
        f"- {n_leaves} leaf regions from {V:,} vocab tokens.",
        "",
        "## Q3 — Are leaf sizes reasonable?",
        f"- min={min(sizes) if sizes else 0}  median={median_sz}  "
        f"mean={mean_sz:.1f}  max={max(sizes) if sizes else 0}",
        "",
        "## Q4 — Are leaves predictable from hidden states?",
    ]
    if probe_final:
        t1 = probe_final.get("top1", float("nan"))
        t8 = probe_final.get("top8", float("nan"))
        rand = probe_final.get("random_top1", float("nan"))
        lines.append(
            f"- Layer {final_layer}: top1={t1:.4f}  top8={t8:.4f}  "
            f"random={rand:.6f}  ratio={t1/rand:.1f}x above random"
        )
        if t1 > 5 * rand:
            lines.append("- **YES**: strongly above random.")
        elif t1 > 2 * rand:
            lines.append("- **MODERATE**: above random but not strongly.")
        else:
            lines.append("- **WEAK**: near random.")
    else:
        lines.append("- Probe metrics not available.")

    lines += ["", "## Q5/Q6 — Top-r leaf routing vs recall / candidate fraction"]
    lines.append(
        f"| r | recall (L{final_layer}) | cand_frac | random recall | delta |"
    )
    lines.append("|---|---|---|---|---|")
    for r in r_values:
        rec = final_recall.get(r, float("nan"))
        frac = final_cand_frac.get(r, float("nan"))
        rand_rec = rand_recall.get(r, float("nan"))
        delta = rec - rand_rec if not (math.isnan(rec) or math.isnan(rand_rec)) else float("nan")
        lines.append(
            f"| {r} | {rec:.4f} | {frac:.4f} | {rand_rec:.4f} | {delta:+.4f} |"
        )

    lines += ["", "### Recall thresholds"]
    for target_rec, label in [(0.90, "90%"), (0.95, "95%"), (0.98, "98%")]:
        met = [r for r in r_values if final_recall.get(r, 0.0) >= target_rec]
        if met:
            r_met = min(met)
            frac = final_cand_frac.get(r_met, float("nan"))
            lines.append(f"- **{label} recall** at r={r_met}, cand_frac={frac:.4f}")
        else:
            lines.append(f"- **{label} recall** not achieved within r_values={r_values}")

    lines += ["", "## Q7 — Is this better than random?"]
    for r in [16, 32]:
        if r not in r_values:
            continue
        pred = final_recall.get(r, float("nan"))
        rand = rand_recall.get(r, float("nan"))
        delta = pred - rand if not (math.isnan(pred) or math.isnan(rand)) else float("nan")
        lines.append(
            f"- r={r}: predicted={pred:.4f}  random={rand:.4f}  delta={delta:+.4f}"
        )

    lines += ["", "## Q8 — Strong enough for 2-stage architecture?", ""]

    # ── Verdict ──
    rec16 = final_recall.get(16, 0.0)
    frac16 = final_cand_frac.get(16, 1.0)
    rec32 = final_recall.get(32, 0.0)
    frac32 = final_cand_frac.get(32, 1.0)
    probe_top1 = probe_final.get("top1", 0.0)
    probe_rand = probe_final.get("random_top1", 1.0)

    locality_gone = (probe_top1 < 2 * probe_rand) if probe_rand > 0 else True

    if rec16 >= 0.95 and frac16 <= 0.20 and not locality_gone and median_sz <= 200:
        verdict = "STRONG GO"
        verdict_emoji = "🟢"
        reason = (
            f"top-16 recall={rec16:.4f} ≥ 0.95, "
            f"cand_frac={frac16:.4f} ≤ 0.20, "
            f"median leaf size={median_sz} ≤ 200"
        )
    elif rec32 >= 0.96 and frac32 <= 0.40:
        verdict = "GO"
        verdict_emoji = "🟡"
        reason = f"top-32 recall={rec32:.4f} ≥ 0.96, cand_frac={frac32:.4f} ≤ 0.40"
    elif locality_gone or (probe_top1 < 1.5 * probe_rand and probe_rand > 0):
        verdict = "NO-GO"
        verdict_emoji = "🔴"
        reason = (
            f"Leaf prediction near random "
            f"(top1={probe_top1:.4f}, random={probe_rand:.6f}). "
            f"Locality has disappeared at this vocab scale."
        )
    else:
        verdict = "WEAK"
        verdict_emoji = "🟠"
        reason = (
            f"top-16 recall={rec16:.4f} < 0.85 "
            f"or cand_frac={frac16:.4f} > 0.40. "
            f"Investigate leaf granularity and clustering resolution."
        )

    lines += [
        f"### Verdict: **{verdict}** {verdict_emoji}",
        "",
        f"**{reason}**",
        "",
        "| Threshold | Criterion | Status |",
        "|-----------|-----------|--------|",
        f"| STRONG GO | top16 recall ≥ 0.95 | {'✓' if rec16 >= 0.95 else '✗'} ({rec16:.4f}) |",
        f"| STRONG GO | cand_frac ≤ 0.20    | {'✓' if frac16 <= 0.20 else '✗'} ({frac16:.4f}) |",
        f"| STRONG GO | median leaf ≤ 200   | {'✓' if median_sz <= 200 else '✗'} ({median_sz}) |",
        f"| GO        | top32 recall ≥ 0.96 | {'✓' if rec32 >= 0.96 else '✗'} ({rec32:.4f}) |",
        f"| GO        | cand_frac ≤ 0.40    | {'✓' if frac32 <= 0.40 else '✗'} ({frac32:.4f}) |",
        "",
        "---",
        "*Generated by full_vocab_region_eval.py*",
    ]

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Saved {summary_path}")
    log.info(f"\nVERDICT: {verdict} {verdict_emoji}\n{reason}")


# ──────────────────────────────────────────────────────────────────────────────
# Eval orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(args: argparse.Namespace, device: torch.device, out_dir: str) -> None:
    log = logging.getLogger("fvr")
    log.info("=" * 60)
    log.info("STAGE: eval")
    log.info("=" * 60)

    target_layers = [int(x) for x in args.layers.split(",")]
    final_layer = max(target_layers)
    r_values = [1, 2, 4, 8, 16, 32, 64]
    logit_ks = [10, 50, 100, 500]

    leaf_labels = np.load(os.path.join(out_dir, "leaf_labels.npy"))
    gold_tokens = np.load(os.path.join(out_dir, "gold_tokens.npy"))
    top_ids = np.load(os.path.join(out_dir, "frequent_token_ids.npy"))

    with open(os.path.join(out_dir, "leaf_region_to_tokens.json")) as f:
        leaf_region_to_tokens_str = json.load(f)
    leaf_region_to_tokens: dict[int, list[int]] = {
        int(k): v for k, v in leaf_region_to_tokens_str.items()
    }

    # Load model once for logit locality + routed softmax
    model, _ = load_model_and_tokenizer(args.model_name, device)

    probes = eval_router_probes(out_dir, target_layers, device, args)

    eval_gold_recall(
        out_dir, target_layers, probes,
        leaf_region_to_tokens, leaf_labels, gold_tokens,
        device, r_values,
    )
    eval_logit_locality(
        out_dir, final_layer,
        leaf_region_to_tokens, leaf_labels, gold_tokens,
        model, top_ids, logit_ks,
    )
    eval_routed_softmax(
        out_dir, final_layer, probes,
        leaf_region_to_tokens, leaf_labels, gold_tokens,
        model, top_ids, device, r_values,
    )
    eval_random_baseline(
        out_dir, leaf_region_to_tokens, leaf_labels, gold_tokens, r_values,
    )
    write_summary(out_dir, leaf_region_to_tokens, target_layers, r_values)


# ──────────────────────────────────────────────────────────────────────────────
# CSV helper
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = args.output_dir
    ensure_dir(out_dir)

    log = setup_logging(os.path.join(out_dir, "run.log"))
    log.info("full_vocab_region_eval.py")
    log.info(f"stage={args.stage}  vocab_subset_size={args.vocab_subset_size:,}  "
             f"knn_edges={args.knn_edges}  output_dir={out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(
            f"GPU: {torch.cuda.get_device_name(0)}  "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    run_all = (args.stage == "all")

    if run_all or args.stage == "build_graph":
        build_graph(args, device, out_dir)

    if run_all or args.stage == "cluster":
        cluster(args, out_dir)

    if run_all or args.stage == "cache_hidden":
        cache_hidden(args, device, out_dir)

    if run_all or args.stage == "eval":
        run_eval(args, device, out_dir)

    if args.stage == "soft_routing_eval":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from soft_routing_eval import run_soft_routing_eval
        run_soft_routing_eval(args, device, out_dir)

    log.info("=" * 60)
    log.info("Done.")


if __name__ == "__main__":
    main()
