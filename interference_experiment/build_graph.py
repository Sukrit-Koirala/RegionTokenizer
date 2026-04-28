"""
Co-activation interference graph.

For each context position t, the top-k predicted tokens form a competing set.
Every pair (i, j) within that set increments W[i, j], capturing how often
two tokens simultaneously compete for the same prediction slot.

Only the vocab_subset_size most-frequently predicted tokens are included in
the dense output graph; the full sparse co-occurrence matrix is also saved.

Memory strategy
---------------
We avoid materialising the full vocab×vocab dense matrix (which would be
~10 GB for GPT-2's 50 K vocabulary).  Instead we:
  1. Build a fully vectorised (N_positions × n_pairs) representation.
  2. Map only the frequent-token pairs to a compact local index space.
  3. Use np.bincount on a flat index to count efficiently.
"""

import argparse
import logging
import os
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz

from utils import (
    ExperimentConfig,
    save_json,
    setup_dirs,
    setup_logging,
    symmetric_normalize,
)
from extract_hidden_states import load_extracted


# ---------------------------------------------------------------------------
# Core graph builder
# ---------------------------------------------------------------------------

def build_coactivation_graph(
    topk_indices: np.ndarray,          # (N_positions, K)  uint16 or int32
    frequent_token_ids: np.ndarray,    # (N_sub,)           int32
    vocab_size: int,
) -> np.ndarray:
    """
    Build and return the dense co-activation matrix restricted to the
    frequent-token vocabulary subset.

    Returns W_raw (N_sub × N_sub) float32, raw co-occurrence counts.
    """
    log = logging.getLogger("interference")
    N, K = topk_indices.shape
    N_sub = len(frequent_token_ids)

    # ---- Build a fast lookup: token_id → local_index (-1 if not frequent) ----
    # Using a numpy array of size vocab_size for O(1) lookup.
    lookup = np.full(vocab_size, -1, dtype=np.int32)
    for local_idx, tok_id in enumerate(frequent_token_ids):
        lookup[int(tok_id)] = local_idx

    log.info(
        "Building co-activation graph: N_pos=%d  K=%d  vocab_sub=%d",
        N, K, N_sub,
    )

    # ---- Map top-k token IDs to local indices ----
    # Cast to int32 first (uint16 can't represent -1 correctly).
    topk_local = lookup[topk_indices.astype(np.int32)]  # (N, K) int32; -1 = out-of-subset

    # ---- Generate all upper-triangle pairs within each position's top-k ----
    ii, jj = np.triu_indices(K, k=1)   # shapes (n_pairs,); n_pairs = K*(K-1)//2
    n_pairs = len(ii)

    # pair_a[t, p] = local index of token ii[p] at position t
    pair_a = topk_local[:, ii]  # (N, n_pairs) int32
    pair_b = topk_local[:, jj]  # (N, n_pairs) int32

    # ---- Keep only pairs where both tokens are in the frequent subset ----
    valid = (pair_a >= 0) & (pair_b >= 0)  # (N, n_pairs) bool
    row_idx = pair_a[valid]   # (n_valid,) int32
    col_idx = pair_b[valid]   # (n_valid,) int32

    log.info(
        "Valid pairs: %d / %d (%.1f %%)",
        len(row_idx),
        N * n_pairs,
        100.0 * len(row_idx) / max(N * n_pairs, 1),
    )

    # ---- Count via bincount on a flat index ----
    # Both directions (symmetric) — add (row, col) and (col, row).
    flat_rc = row_idx * N_sub + col_idx   # (n_valid,) upper-triangle
    flat_cr = col_idx * N_sub + row_idx   # (n_valid,) lower-triangle
    flat_all = np.concatenate([flat_rc, flat_cr]).astype(np.int64)

    counts = np.bincount(flat_all, minlength=N_sub * N_sub)
    W_raw = counts.reshape(N_sub, N_sub).astype(np.float32)

    log.info(
        "W_raw: shape=%s  total_weight=%.0f  max=%.0f  sparsity=%.3f",
        W_raw.shape,
        W_raw.sum(),
        W_raw.max(),
        (W_raw == 0).mean(),
    )
    return W_raw


def build_full_sparse_graph(
    topk_indices: np.ndarray,
    vocab_size: int,
) -> csr_matrix:
    """
    Build the full (vocab_size × vocab_size) sparse co-occurrence matrix.
    Useful for downstream analysis that doesn't restrict the vocabulary.
    """
    log = logging.getLogger("interference")
    N, K = topk_indices.shape
    ii, jj = np.triu_indices(K, k=1)

    topk_int = topk_indices.astype(np.int32)
    pair_a = topk_int[:, ii].flatten()
    pair_b = topk_int[:, jj].flatten()

    # symmetric: both directions
    rows = np.concatenate([pair_a, pair_b])
    cols = np.concatenate([pair_b, pair_a])
    data = np.ones(len(rows), dtype=np.float32)

    W_sparse = coo_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
    W_sparse = W_sparse.tocsr()
    log.info(
        "Sparse graph: shape=%s  nnz=%d", W_sparse.shape, W_sparse.nnz
    )
    return W_sparse


# ---------------------------------------------------------------------------
# Normalisation options
# ---------------------------------------------------------------------------

def normalize_graph(W: np.ndarray, method: str = "symmetric") -> np.ndarray:
    """
    Normalise the raw co-activation counts into a similarity-like matrix.

    method options
    --------------
    'symmetric' : D^{-1/2} W D^{-1/2}  (graph-Laplacian normalisation)
    'max'       : divide by global maximum
    'none'      : return float32 copy unchanged
    """
    if method == "symmetric":
        return symmetric_normalize(W)
    elif method == "max":
        m = W.max()
        return (W / m).astype(np.float32) if m > 0 else W.astype(np.float32)
    elif method == "none":
        return W.astype(np.float32)
    else:
        raise ValueError(f"Unknown normalisation method: {method!r}")


# ---------------------------------------------------------------------------
# High-level build function
# ---------------------------------------------------------------------------

def build_and_save(
    prefix: str,
    cfg: ExperimentConfig,
    norm_method: str = "symmetric",
    build_full_sparse: bool = True,
) -> np.ndarray:
    """
    Load extraction artefacts from *prefix*, build co-activation graph,
    normalise, save, and return the dense normalised matrix W_norm.
    """
    log = logging.getLogger("interference")
    data = load_extracted(prefix)

    topk_indices = data["topk_indices"]            # (N, K) uint16
    frequent_ids = data["frequent_token_ids"]      # (N_sub,) int32
    vocab_size = int(data["vocab_size"])

    # Dense restricted graph
    W_raw = build_coactivation_graph(topk_indices, frequent_ids, vocab_size)
    W_norm = normalize_graph(W_raw, method=norm_method)

    layer_tag = data.get("layer_tag", "final")
    out_prefix = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}")

    np.save(f"{out_prefix}_W_raw.npy", W_raw)
    np.save(f"{out_prefix}_W_norm.npy", W_norm)

    log.info("Saved dense graph → %s_W_{raw,norm}.npy", out_prefix)

    # Full sparse graph (optional — takes more memory but covers full vocab)
    if build_full_sparse:
        W_sparse = build_full_sparse_graph(topk_indices, vocab_size)
        save_npz(f"{out_prefix}_W_sparse.npz", W_sparse)
        log.info("Saved sparse graph → %s_W_sparse.npz", out_prefix)

    meta = {
        "layer_tag": layer_tag,
        "norm_method": norm_method,
        "vocab_subset_size": len(frequent_ids),
        "W_shape": list(W_norm.shape),
        "W_total_weight": float(W_raw.sum()),
        "W_max_weight": float(W_raw.max()),
        "W_sparsity": float((W_norm == 0).mean()),
    }
    save_json(f"{out_prefix}_W_meta.json", meta)
    return W_norm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build co-activation interference graph.")
    parser.add_argument("--layer_tag", default="final",
                        help="Tag matching the extraction run to process.")
    parser.add_argument("--norm", default="symmetric",
                        choices=["symmetric", "max", "none"])
    parser.add_argument("--no_sparse", action="store_true",
                        help="Skip building the full sparse graph.")
    args = parser.parse_args()

    setup_logging()
    cfg = ExperimentConfig()
    setup_dirs(cfg)

    prefix = os.path.join(cfg.graphs_dir, f"layer_{args.layer_tag}")
    build_and_save(prefix, cfg, norm_method=args.norm, build_full_sparse=not args.no_sparse)


if __name__ == "__main__":
    main()
