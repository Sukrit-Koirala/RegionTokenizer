#!/usr/bin/env python3
"""
build_compute_aware_region_maps.py — Phase 1B

Compute-Aware Predictive Interference Region Map V2.

Builds empirical region maps from model behaviour (base_topk coactivation),
with compute-aware constraints: size-balanced, BPE-aware, and head-tail variants.

NO manual labels. NO semantic categories.
All regions are numeric. Representatives are token examples, not names.

Map variants:
  V2A_K{K}           balanced interference, no BPE penalty, full vocab
  V2B_K{K}           balanced interference + BPE surface penalty, full vocab
  V2C_K{K}_head{H}   fixed-head + routed tail regions

Usage:
  python scripts/build_compute_aware_region_maps.py \\
    --train_dir runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train \\
    --val_dir   runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val \\
    --old_map   runs/region_maps_128/token_to_region.json \\
    --output_root runs/cheap_ai/phase1B_compute_aware_region_maps \\
    --tokenizer_name gpt2 --vocab_size 50257 \\
    --graph_topk 64 --pair_topk 32 --top_neighbors_per_token 128 \\
    --graph_embed_dim 64 \\
    --variants V2A_K256,V2A_K512,V2B_K256,V2B_K512,V2C_K256_head1024,V2C_K512_head1024 \\
    --bpe_lambda 0.75 --head_size 1024 \\
    --max_size_mult 2.5 --min_size_mult 0.25 --seed 42
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
    from scipy.sparse.linalg import svds
    _SCIPY = True
except ImportError:
    _SCIPY = False
    print("[WARN] scipy not available; will use dense fallback (may OOM for large vocab)")

try:
    from sklearn.preprocessing import normalize as sk_normalize
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    raise ImportError("scikit-learn is required: pip install scikit-learn")

_EPS = 1e-9

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    if isinstance(v, bool):  return str(v)
    if isinstance(v, int):   return str(v)
    if isinstance(v, float): return f"{v:.5f}" if v == v else "nan"
    return str(v)

def _wcsv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[save] {path}")

def _write(path, text):
    with open(path, "w", encoding="utf-8") as f: f.write(text)
    print(f"[save] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Shard loading
# ══════════════════════════════════════════════════════════════════════════════

_TOPK_AL = ["base_topk_ids", "base_topk", "topk_ids"]
_LGT_AL  = ["base_topk_logits", "base_topk_lgt", "topk_lgt", "topk_logits"]
_GOLD_AL = ["gold_token", "gold", "labels"]
_RID_AL  = ["row_id", "row_ids"]

def _get_key(sh, aliases, required=True):
    for a in aliases:
        if a in sh: return sh[a]
    if required:
        raise KeyError(f"Need one of {aliases}; have {list(sh.keys())}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Graph construction (train only, fully vectorised per shard)
# ══════════════════════════════════════════════════════════════════════════════

def _rank_weights(k: int) -> np.ndarray:
    return (1.0 / np.log2(np.arange(1, k + 1) + 1)).astype(np.float32)


def _shard_graph_coo(topk_ids: np.ndarray,
                     vocab_size: int,
                     pair_topk: int,
                     rw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build COO edge arrays from one shard's base_topk_ids. Fully vectorised."""
    B, K = topk_ids.shape
    pk   = min(pair_topk, K)
    ids  = topk_ids[:, :pk].astype(np.int64)        # [B, pk]
    # upper-triangle pairs
    ii_idx, jj_idx = np.triu_indices(pk, k=1)        # (P,) each, P = pk*(pk-1)/2
    left  = ids[:, ii_idx]                            # [B, P]
    right = ids[:, jj_idx]                            # [B, P]
    w_vec = (rw[ii_idx] * rw[jj_idx]).astype(np.float32)  # [P]
    wmat  = np.broadcast_to(w_vec, (B, len(ii_idx))) # [B, P]

    valid = ((left >= 0) & (left < vocab_size) &
             (right >= 0) & (right < vocab_size))     # [B, P]
    r  = left[valid].astype(np.int32)
    c  = right[valid].astype(np.int32)
    d  = wmat[valid].astype(np.float32)
    return (np.concatenate([r, c]),
            np.concatenate([c, r]),
            np.concatenate([d, d]))


def build_coactivation_graph(train_dir: str,
                              vocab_size: int,
                              graph_topk: int,
                              pair_topk: int,
                              top_neighbors: int,
                              gold_edge_weight: float,
                              seed: int) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Stream through train shards and build sparse token coactivation graph.
    Returns (W_norm_csr, gold_freq, topk_freq).
    """
    if not _SCIPY:
        raise RuntimeError("scipy required for sparse graph construction")

    paths = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {train_dir}")
    print(f"[graph] {len(paths)} train shards  vocab_size={vocab_size}")

    rw_full = _rank_weights(graph_topk)
    gold_freq = np.zeros(vocab_size, dtype=np.float64)
    topk_freq = np.zeros(vocab_size, dtype=np.float64)

    # Memory estimate: pair_topk=32 → 496 pairs/row, each float32+int32*2 ≈ 12 bytes
    # For 500k rows: 248M pairs * 12B ≈ 3GB — too much in memory at once.
    # Accumulate per-shard into CSR and sum shards.
    W = csr_matrix((vocab_size, vocab_size), dtype=np.float32)

    for si, sp in enumerate(paths):
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        topk_ids = _get_key(sh, _TOPK_AL).numpy().astype(np.int32)  # [B, K]
        gold      = _get_key(sh, _GOLD_AL).numpy().astype(np.int64)
        K_sh      = topk_ids.shape[1]
        G_pk      = min(graph_topk, K_sh)
        P_pk      = min(pair_topk,  K_sh)
        rw        = _rank_weights(G_pk)

        # Coactivation edges
        r, c, d = _shard_graph_coo(topk_ids[:, :G_pk], vocab_size, P_pk, rw[:P_pk])
        W_sh = coo_matrix((d, (r, c)), shape=(vocab_size, vocab_size)).tocsr()
        W    = W + W_sh

        # Token frequencies
        valid_gold = gold[(gold >= 0) & (gold < vocab_size)]
        np.add.at(gold_freq, valid_gold, 1.0)
        valid_topk = topk_ids[(topk_ids >= 0) & (topk_ids < vocab_size)].ravel()
        np.add.at(topk_freq, valid_topk, 1.0 / G_pk)

        # Gold → top-K edges (small weight)
        if gold_edge_weight > 0:
            gold_r, gold_c, gold_d = [], [], []
            gold_top8 = topk_ids[:, :min(8, G_pk)]
            for b in range(len(gold)):
                g = int(gold[b])
                if g < 0 or g >= vocab_size: continue
                for ti in gold_top8[b]:
                    ti = int(ti)
                    if 0 <= ti < vocab_size and ti != g:
                        gold_r.extend([g, ti]); gold_c.extend([ti, g])
                        gold_d.extend([gold_edge_weight * float(rw[0])] * 2)
            if gold_r:
                W_gold = coo_matrix((gold_d, (gold_r, gold_c)),
                                    shape=(vocab_size, vocab_size)).tocsr()
                W = W + W_gold

        if (si + 1) % 20 == 0:
            print(f"  [graph] {si+1}/{len(paths)} shards  nnz={W.nnz:,}")

    print(f"[graph] raw  nnz={W.nnz:,}  sum={W.sum():.1f}")

    # Normalize: W[i,j] /= sqrt(freq_i * freq_j + eps)
    combined_freq = gold_freq + 0.25 * topk_freq + _EPS
    inv_sqrt = 1.0 / np.sqrt(combined_freq).astype(np.float32)
    inv_sqrt_diag = csr_matrix(
        (inv_sqrt, (np.arange(vocab_size), np.arange(vocab_size))),
        shape=(vocab_size, vocab_size))
    W_norm = inv_sqrt_diag @ W @ inv_sqrt_diag

    # Prune to top_neighbors per row
    print(f"[graph] pruning to top_{top_neighbors} neighbors per token...")
    W_pruned = _prune_topk_per_row(W_norm, top_neighbors)
    print(f"[graph] pruned nnz={W_pruned.nnz:,}")

    return W_pruned, gold_freq, topk_freq


def _prune_topk_per_row(W_csr, k: int):
    """Keep only top-k nonzero entries per row."""
    W_lil = W_csr.tolil()
    for i in range(W_csr.shape[0]):
        row_data = np.array(W_lil.data[i], dtype=np.float32)
        if len(row_data) > k:
            keep = np.argpartition(row_data, -k)[-k:]
            mask = np.zeros(len(row_data), dtype=bool); mask[keep] = True
            W_lil.data[i] = [d for d, m in zip(W_lil.data[i], mask) if m]
            W_lil.rows[i]  = [r for r, m in zip(W_lil.rows[i], mask) if m]
    return W_lil.tocsr()


# ══════════════════════════════════════════════════════════════════════════════
# Surface features
# ══════════════════════════════════════════════════════════════════════════════

def compute_surface_features(vocab_size: int, tokenizer_name: str) -> Tuple[np.ndarray, List[str]]:
    """
    Compute automatic surface/tokenizer features for every token id.
    Returns (feature_matrix [V, D], feature_names).
    NO semantic labels — only structural/BPE properties.
    """
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"[surface] tokenizer loaded: {tokenizer_name}")
    except Exception as e:
        print(f"[surface] WARN tokenizer load failed: {e} — using dummy features")

    feat_names = [
        "starts_with_space", "starts_without_space", "is_single_char",
        "len_chars", "has_alpha", "has_digit", "has_punct",
        "is_punct_like", "is_whitespace_ctrl",
        "alpha_ratio", "digit_ratio", "punct_ratio",
        "len_bytes", "is_upper", "has_upper",
    ]
    D = len(feat_names)
    feats = np.zeros((vocab_size, D), dtype=np.float32)

    for tid in range(vocab_size):
        try:
            if tokenizer is not None:
                s = tokenizer.decode([tid])
            else:
                s = f"_{tid}_"
        except Exception:
            s = ""
        if not s: s = ""
        sws = s.startswith(" ")
        nws = not sws
        alph = sum(c.isalpha() for c in s)
        digs = sum(c.isdigit() for c in s)
        punc = sum(not c.isalnum() and not c.isspace() for c in s)
        ln   = len(s)
        feats[tid] = [
            float(sws), float(nws),
            float(ln == 1),
            min(ln / 20.0, 1.0),
            float(alph > 0), float(digs > 0), float(punc > 0),
            float(punc > 0 and alph == 0 and digs == 0),
            float(all(c.isspace() or ord(c) < 32 for c in s) and len(s) > 0),
            alph / (ln + _EPS), digs / (ln + _EPS), punc / (ln + _EPS),
            min(len(s.encode("utf-8", errors="replace")) / 10.0, 1.0),
            float(s.isupper() and len(s) > 0),
            float(any(c.isupper() for c in s)),
        ]

    # L2-normalize
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + _EPS
    feats_norm = feats / norms
    print(f"[surface] computed {D} features for {vocab_size} tokens")
    return feats_norm, feat_names


def compute_bpe_incompatibility(feats: np.ndarray, bpe_lambda: float) -> np.ndarray:
    """
    Compute pairwise BPE incompatibility as a scaled feature distance.
    Returns a function: incompatibility(i, j) = ||f_i - f_j||
    — but for sparse use, we apply it as a modification weight.
    """
    # We return the feature matrix; actual penalty is applied during graph weighting.
    return feats


def apply_bpe_penalty_to_graph(W_csr, feats: np.ndarray, bpe_lambda: float) -> object:
    """
    W[i,j] *= exp(-bpe_lambda * ||f_i - f_j||)
    Applied only to existing nonzero entries (sparse).
    """
    W_coo = W_csr.tocoo()
    if W_coo.nnz == 0:
        return W_csr
    diff = feats[W_coo.row] - feats[W_coo.col]          # [nnz, D]
    dist = np.linalg.norm(diff, axis=1).astype(np.float32) # [nnz]
    penalty = np.exp(-bpe_lambda * dist)
    W_coo.data = (W_coo.data * penalty).astype(np.float32)
    result = W_coo.tocsr()
    print(f"[bpe] applied penalty (lambda={bpe_lambda})  "
          f"mean_pen={penalty.mean():.4f}  min_pen={penalty.min():.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Graph embedding (TruncatedSVD)
# ══════════════════════════════════════════════════════════════════════════════

def compute_graph_embeddings(W_csr, embed_dim: int, seed: int) -> np.ndarray:
    """
    Compute graph embeddings via TruncatedSVD on normalized adjacency.
    Returns [V, embed_dim] float32 array.
    """
    print(f"[svd] TruncatedSVD dim={embed_dim}  shape={W_csr.shape}  nnz={W_csr.nnz:,}")
    t0 = time.time()
    # Symmetrize
    W_sym = (W_csr + W_csr.T) * 0.5
    svd = TruncatedSVD(n_components=embed_dim, random_state=seed, n_iter=5)
    X = svd.fit_transform(W_sym).astype(np.float32)
    # L2-normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + _EPS
    X = X / norms
    explained = svd.explained_variance_ratio_.sum()
    print(f"[svd] done {time.time()-t0:.1f}s  explained_var={explained:.4f}")
    return X


# ══════════════════════════════════════════════════════════════════════════════
# Head token selection (V2C)
# ══════════════════════════════════════════════════════════════════════════════

def select_head_tokens(gold_freq: np.ndarray,
                        topk_freq: np.ndarray,
                        head_size: int) -> np.ndarray:
    """Select top head_size tokens by head_score = gold_freq + 0.25 * topk_freq."""
    scores = gold_freq + 0.25 * topk_freq
    head_ids = np.argsort(-scores)[:head_size].astype(np.int32)
    print(f"[head] selected {len(head_ids)} head tokens  "
          f"min_score={scores[head_ids[-1]]:.2f}  "
          f"max_score={scores[head_ids[0]]:.2f}")
    return head_ids


# ══════════════════════════════════════════════════════════════════════════════
# Balanced clustering
# ══════════════════════════════════════════════════════════════════════════════

def _split_oversized(X: np.ndarray, labels: np.ndarray,
                     max_size: float, seed: int) -> np.ndarray:
    """Split any cluster larger than max_size into subclusters."""
    next_label = int(labels.max()) + 1
    new_labels  = labels.copy()
    for ul in np.unique(labels):
        mask = (labels == ul)
        n = int(mask.sum())
        if n > max_size:
            n_sub = max(2, int(math.ceil(n / max_size)))
            X_sub = X[mask]
            km = MiniBatchKMeans(n_clusters=n_sub, random_state=seed, n_init=3,
                                  batch_size=min(1024, n_sub * 100))
            sub = km.fit_predict(X_sub)
            idx = np.where(mask)[0]
            for s in range(n_sub):
                smask = (sub == s)
                if s == 0:
                    new_labels[idx[smask]] = ul
                else:
                    new_labels[idx[smask]] = next_label
                    next_label += 1
    return new_labels


def _merge_tiny(X: np.ndarray, labels: np.ndarray,
                min_size: float, max_size: float) -> np.ndarray:
    """Merge clusters smaller than min_size into their nearest neighbour."""
    for _iter in range(20):
        unique = np.unique(labels)
        centers = {ul: X[labels == ul].mean(axis=0) for ul in unique}
        sizes   = {ul: int((labels == ul).sum()) for ul in unique}
        tiny    = [ul for ul in unique if sizes[ul] < min_size]
        if not tiny: break
        ul = tiny[0]
        c  = centers[ul]
        best_nb, best_d = None, float("inf")
        for nb in unique:
            if nb == ul: continue
            if sizes[nb] + sizes[ul] > max_size * 1.1: continue
            d = float(np.linalg.norm(c - centers[nb]))
            if d < best_d: best_d = d; best_nb = nb
        if best_nb is None:
            # Force merge regardless of size
            best_nb = min(
                (nb for nb in unique if nb != ul),
                key=lambda nb: float(np.linalg.norm(c - centers[nb])),
                default=None)
        if best_nb is None: break
        labels[labels == ul] = best_nb
    return labels


def balanced_cluster(X: np.ndarray, K: int,
                     max_mult: float, min_mult: float,
                     seed: int, n_init: int = 3) -> np.ndarray:
    """
    MiniBatchKMeans → iterative size-balance (split + merge).
    Returns labels array of shape [N].
    """
    N = len(X)
    target   = N / K
    max_size = max_mult * target
    min_size = min_mult * target

    print(f"[cluster] K={K}  N={N}  target={target:.0f}  "
          f"max={max_size:.0f}  min={min_size:.0f}")
    bs = min(4096, max(256, K * 20))
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=n_init,
                          batch_size=bs, max_iter=200)
    labels = km.fit_predict(X)

    # Iterative size correction
    for _it in range(6):
        before_unique = len(np.unique(labels))
        labels = _split_oversized(X, labels, max_size, seed + _it)
        labels = _merge_tiny(X, labels, min_size, max_size)
        after_unique = len(np.unique(labels))
        sizes = np.array([int((labels == ul).sum()) for ul in np.unique(labels)])
        if (sizes <= max_size).all() and (sizes >= min_size * 0.5).all():
            break
        if before_unique == after_unique: break

    # Renumber contiguously
    unique = sorted(np.unique(labels))
    remap  = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int32)
    print(f"[cluster] final n_regions={len(np.unique(labels))}  "
          f"max={np.bincount(labels).max()}  min={np.bincount(labels).min()}")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Full-vocab fallback assignment
# ══════════════════════════════════════════════════════════════════════════════

def assign_full_vocab(all_token_ids: np.ndarray,
                       seen_ids: np.ndarray,
                       seen_labels: np.ndarray,
                       X_seen: np.ndarray,
                       surface_feats: np.ndarray,
                       small_ckpt_path: Optional[str],
                       K: int,
                       seed: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Assign all vocab tokens to regions, including unseen ones.
    Returns (full_labels [V], assignment_source [V], stats_dict).
    """
    V = len(all_token_ids)
    full_labels  = np.full(V, -1, dtype=np.int32)
    source       = np.full(V, "none", dtype=object)

    seen_set = set(int(x) for x in seen_ids)
    for i, tid in enumerate(seen_ids):
        full_labels[int(tid)] = int(seen_labels[i])
        source[int(tid)]      = "graph"

    unseen_ids = np.array([t for t in range(V) if t not in seen_set], dtype=np.int32)
    n_unseen   = len(unseen_ids)
    print(f"[fallback] seen={len(seen_ids):,}  unseen={n_unseen:,}")

    if n_unseen == 0:
        stats = {"n_graph": int(len(seen_ids)), "n_embed": 0,
                 "n_surface": 0, "n_deterministic": 0}
        return full_labels, source, stats

    # Compute cluster centroids in SVD space
    centroids = np.zeros((K, X_seen.shape[1]), dtype=np.float32)
    for k in range(K):
        mk = (seen_labels == k)
        if mk.any():
            centroids[k] = X_seen[mk].mean(axis=0)

    # ── Attempt 1: embedding similarity from SMALL_CKPT ──────────────────────
    n_embed = 0
    remaining = list(unseen_ids)
    if small_ckpt_path and os.path.isfile(small_ckpt_path):
        try:
            ck = torch.load(small_ckpt_path, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck.get("model", ck))
            emb = None
            for k_name in list(sd.keys()):
                v = sd[k_name]
                if hasattr(v, "shape") and v.dim() == 2 and v.shape[0] >= V:
                    emb = v[:V].float().numpy(); break
            if emb is not None:
                # Normalize embeddings and centroids
                emb_norm  = emb / (np.linalg.norm(emb,       axis=1, keepdims=True) + _EPS)
                cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + _EPS)
                unseen_arr = np.array(remaining, dtype=np.int32)
                sims = emb_norm[unseen_arr] @ cent_norm.T  # [M, K]
                assigned = sims.argmax(axis=1).astype(np.int32)
                for i, tid in enumerate(unseen_arr):
                    full_labels[tid] = assigned[i]; source[tid] = "embedding"
                n_embed  = len(unseen_arr)
                remaining = []
                print(f"[fallback] embedding assigned {n_embed:,}")
        except Exception as e:
            print(f"[fallback] embedding fallback failed: {e}")

    # ── Attempt 2: surface feature similarity ─────────────────────────────────
    n_surface = 0
    if remaining and surface_feats is not None and len(surface_feats) >= V:
        # Project surface features to centroid space using surface centroids
        seen_surf = surface_feats[seen_ids]
        surf_centroids = np.zeros((K, surface_feats.shape[1]), dtype=np.float32)
        for k in range(K):
            mk = (seen_labels == k)
            if mk.any(): surf_centroids[k] = seen_surf[mk].mean(axis=0)
        sc_norm  = surf_centroids / (np.linalg.norm(surf_centroids, axis=1, keepdims=True) + _EPS)
        rem_arr  = np.array(remaining, dtype=np.int32)
        rem_surf = surface_feats[rem_arr]
        rs_norm  = rem_surf / (np.linalg.norm(rem_surf, axis=1, keepdims=True) + _EPS)
        sims     = rs_norm @ sc_norm.T
        assigned = sims.argmax(axis=1).astype(np.int32)
        for i, tid in enumerate(rem_arr):
            full_labels[tid] = assigned[i]; source[tid] = "surface"
        n_surface = len(rem_arr)
        remaining = []
        print(f"[fallback] surface assigned {n_surface:,}")

    # ── Attempt 3: size-balanced round-robin ──────────────────────────────────
    n_det = 0
    if remaining:
        rng    = np.random.default_rng(seed)
        rem_arr= np.array(remaining, dtype=np.int32)
        rng.shuffle(rem_arr)
        sizes  = np.bincount(full_labels[full_labels >= 0], minlength=K).astype(np.int64)
        for tid in rem_arr:
            k_assign = int(sizes.argmin())
            full_labels[tid] = k_assign; source[tid] = "deterministic"
            sizes[k_assign] += 1
        n_det = len(rem_arr)
        print(f"[fallback] deterministic assigned {n_det:,}")

    # Sanity
    if (full_labels < 0).any():
        n_bad = int((full_labels < 0).sum())
        print(f"[fallback] WARN: {n_bad} tokens still unassigned, using region 0")
        full_labels[full_labels < 0] = 0

    stats = {"n_graph": int(len(seen_ids)), "n_embed": n_embed,
             "n_surface": n_surface, "n_deterministic": n_det}
    return full_labels, source, stats


# ══════════════════════════════════════════════════════════════════════════════
# Region representatives
# ══════════════════════════════════════════════════════════════════════════════

def build_representatives(full_labels: np.ndarray, gold_freq: np.ndarray,
                           topk_freq: np.ndarray, W_csr, K: int,
                           tokenizer, top_n: int = 30) -> List[dict]:
    V = len(full_labels)
    if W_csr is not None:
        degree = np.asarray(W_csr.sum(axis=1)).ravel().astype(np.float64)
        max_d  = degree.max() + _EPS
        degree_norm = degree / max_d
    else:
        degree_norm = np.zeros(V)

    reps = []
    for r in range(K):
        mask = (full_labels == r)
        tids = np.where(mask)[0]
        if len(tids) == 0:
            reps.append({"region_id": r, "size": 0,
                         "top_token_ids": [], "top_token_strs": [], "top_scores": []}); continue
        scores = (1.0 * np.log1p(gold_freq[tids])
                + 0.5 * np.log1p(topk_freq[tids])
                + 0.5 * degree_norm[tids])
        top_idx = np.argsort(-scores)[:top_n]
        top_ids = tids[top_idx].tolist()
        top_sc  = scores[top_idx].tolist()
        top_str = []
        for t in top_ids:
            try:
                top_str.append(repr(tokenizer.decode([t])) if tokenizer else f"<{t}>")
            except Exception:
                top_str.append(f"<{t}>")
        reps.append({"region_id": r, "size": int(mask.sum()),
                     "top_token_ids": top_ids, "top_token_strs": top_str,
                     "top_scores": [round(s, 4) for s in top_sc]})
    return reps


# ══════════════════════════════════════════════════════════════════════════════
# Map diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def compute_map_diagnostics(full_labels: np.ndarray,
                             W_csr,
                             surface_feats: Optional[np.ndarray],
                             K: int,
                             map_name: str) -> dict:
    sizes = np.bincount(full_labels, minlength=K).astype(np.float64)
    mean_sz = float(sizes.mean())
    diag = {
        "map_name":                map_name,
        "n_regions":               K,
        "n_tokens":                int(len(full_labels)),
        "min_size":                int(sizes.min()),
        "max_size":                int(sizes.max()),
        "mean_size":               round(mean_sz, 2),
        "median_size":             round(float(np.median(sizes)), 2),
        "p90_size":                round(float(np.percentile(sizes, 90)), 2),
        "p95_size":                round(float(np.percentile(sizes, 95)), 2),
        "p99_size":                round(float(np.percentile(sizes, 99)), 2),
        "num_regions_over_2x_mean": int((sizes > 2 * mean_sz).sum()),
        "num_regions_over_4x_mean": int((sizes > 4 * mean_sz).sum()),
    }
    # Size entropy / Gini
    p = sizes / sizes.sum()
    p_nz = p[p > 0]
    diag["size_entropy"] = round(float(-(p_nz * np.log(p_nz)).sum()), 4)
    n = len(sizes)
    s_sorted = np.sort(sizes)
    gini = float((2 * np.arange(1, n+1) @ s_sorted - (n+1) * sizes.sum()) / (n * sizes.sum() + _EPS))
    diag["size_gini"] = round(gini, 4)

    # Intra/inter graph ratio (sampled)
    if W_csr is not None:
        try:
            W_coo = W_csr.tocoo()
            same = W_coo.data[full_labels[W_coo.row] == full_labels[W_coo.col]]
            diff = W_coo.data[full_labels[W_coo.row] != full_labels[W_coo.col]]
            diag["avg_intra_weight"] = round(float(same.mean()), 6) if len(same) > 0 else 0.0
            diag["avg_inter_weight"] = round(float(diff.mean()), 6) if len(diff) > 0 else 0.0
            diag["intra_inter_ratio"] = round(
                float(same.mean()) / (float(diff.mean()) + _EPS), 4) if len(diff) > 0 else float("nan")
        except Exception as e:
            diag["intra_inter_ratio"] = float("nan")

    # Top-10 largest regions
    top10_idx = np.argsort(-sizes)[:10].tolist()
    diag["top10_largest_regions"] = str([(int(i), int(sizes[i])) for i in top10_idx])
    diag["worst_region_size"]     = int(sizes.max())

    # Print warnings
    if int(sizes.max()) > 4 * mean_sz:
        print(f"  [WARN] {map_name}: max region size={int(sizes.max())} > 4x mean={mean_sz:.0f}")
        print(f"         largest regions: {[(int(i), int(sizes[i])) for i in top10_idx[:5]]}")

    # Surface feature purity diagnostics
    if surface_feats is not None:
        try:
            starts_space = surface_feats[:, 0]  # feature 0 = starts_with_space
            purity_vals = []
            for r in range(K):
                mk = (full_labels == r)
                if mk.sum() < 2: continue
                sf_r = starts_space[mk]
                p_r  = float(sf_r.mean())
                purity_vals.append(max(p_r, 1 - p_r))
            diag["starts_with_space_purity"] = round(float(np.mean(purity_vals)), 4) if purity_vals else float("nan")
        except Exception:
            diag["starts_with_space_purity"] = float("nan")

    return diag


# ══════════════════════════════════════════════════════════════════════════════
# Old map comparison & audit feedback
# ══════════════════════════════════════════════════════════════════════════════

def load_old_map(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f: raw = json.load(f)
    if isinstance(raw, list):
        t2r = {i: v for i, v in enumerate(raw) if v is not None}
    else:
        t2r = {int(k): v for k, v in raw.items()}
    n_old = int(max(t2r.values())) + 1 if t2r else 1
    V     = max(t2r.keys(), default=0) + 2
    arr   = np.full(V, n_old, dtype=np.int32)
    for t, r in t2r.items(): arr[int(t)] = int(r)
    print(f"[old_map] loaded  n_old_regions={n_old}  V={V}")
    return arr


def old_region_fragmentation(old_map: np.ndarray, new_labels: np.ndarray,
                              new_K: int, map_name: str,
                              problem_regions: List[int]) -> List[dict]:
    """For each old region, show how its tokens distribute across new regions."""
    rows = []
    n_old = int(old_map[old_map < old_map.max()].max()) + 1 if len(old_map) > 0 else 0
    V_min = min(len(old_map), len(new_labels))
    for old_r in range(n_old):
        mask = (old_map[:V_min] == old_r)
        n    = int(mask.sum())
        if n == 0: continue
        new_r = new_labels[:V_min][mask]
        new_dist = np.bincount(new_r[new_r < new_K], minlength=new_K)
        top5 = [(int(i), int(new_dist[i]))
                for i in np.argsort(-new_dist)[:5] if new_dist[i] > 0]
        is_problem = old_r in problem_regions
        rows.append({
            "old_region":      old_r,
            "old_size":        n,
            "new_map":         map_name,
            "n_new_fragments": int((new_dist > 0).sum()),
            "top5_new_regions": str(top5),
            "is_problem_region": is_problem,
        })
    return rows


def load_audit_feedback(audit_dir: Optional[str]) -> dict:
    """Load Phase 1A.4 audit feedback for diagnostics (read-only)."""
    fb = {}
    if not audit_dir or not os.path.isdir(audit_dir): return fb
    for fname in ["router_confusion_pairs.csv", "top8_diversity_stats.csv",
                  "miss_rank_histogram.csv", "adaptive_fallback_policies.csv"]:
        fpath = os.path.join(audit_dir, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, newline="", encoding="utf-8") as f:
                    fb[fname] = list(csv.DictReader(f))
                print(f"[audit_fb] loaded {fname}  rows={len(fb[fname])}")
            except Exception as e:
                print(f"[audit_fb] WARN could not load {fname}: {e}")
    return fb


# ══════════════════════════════════════════════════════════════════════════════
# Variant parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_variant(v: str) -> dict:
    """Parse variant string like V2A_K256 or V2C_K256_head1024."""
    parts = v.split("_")
    vtype = parts[0]
    K = None; head_size = None
    for p in parts[1:]:
        if p.startswith("K"):
            try: K = int(p[1:])
            except ValueError: pass
        elif p.startswith("head"):
            try: head_size = int(p[4:])
            except ValueError: pass
    return {"type": vtype, "K": K, "head_size": head_size, "name": v}


# ══════════════════════════════════════════════════════════════════════════════
# Save map outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_map_outputs(map_dir: str,
                      variant_cfg: dict,
                      full_labels: np.ndarray,
                      source: np.ndarray,
                      reps: List[dict],
                      diag: dict,
                      head_ids: Optional[np.ndarray],
                      K: int,
                      vocab_size: int,
                      assign_stats: dict,
                      tokenizer):
    os.makedirs(map_dir, exist_ok=True)
    vtype = variant_cfg["type"]

    # token_to_region.json
    # For V2C: head tokens are omitted (unknown) — they auto-cover from fixed head
    t2r = {}
    for tid in range(vocab_size):
        r = int(full_labels[tid])
        if r >= 0:
            if vtype == "V2C" and head_ids is not None and tid in set(int(x) for x in head_ids):
                pass  # omit head tokens from routed map
            else:
                t2r[str(tid)] = r
    with open(os.path.join(map_dir, "token_to_region.json"), "w") as f:
        json.dump(t2r, f)
    print(f"[save] token_to_region.json  ({len(t2r)} entries)")

    # region_to_tokens.json
    r2t: dict = defaultdict(list)
    for tid, r in t2r.items():
        r2t[str(r)].append(int(tid))
    with open(os.path.join(map_dir, "region_to_tokens.json"), "w") as f:
        json.dump(dict(r2t), f)

    # region_labels.npy
    np.save(os.path.join(map_dir, "region_labels.npy"), full_labels)

    # region_sizes.csv
    sizes_arr = np.bincount(full_labels[full_labels >= 0], minlength=K).astype(np.int32)
    _wcsv(os.path.join(map_dir, "region_sizes.csv"),
          [{"region_id": r, "size": int(sizes_arr[r])} for r in range(K)])

    # region_representatives.json
    with open(os.path.join(map_dir, "region_representatives.json"), "w", encoding="utf-8") as f:
        json.dump(reps, f, indent=2)
    txt_lines = [f"region={r['region_id']:4d}  size={r['size']:5d}  "
                 f"top_tokens={r['top_token_strs'][:8]}"
                 for r in reps]
    _write(os.path.join(map_dir, "region_representatives.txt"), "\n".join(txt_lines))

    # region_diagnostics.json
    with open(os.path.join(map_dir, "region_diagnostics.json"), "w") as f:
        json.dump({**diag, **assign_stats}, f, indent=2)

    # assignment_sources.csv
    source_counts = {s: int((source == s).sum()) for s in ["graph", "embedding", "surface", "deterministic"]}
    _wcsv(os.path.join(map_dir, "assignment_sources.csv"),
          [{"source": k, "count": v} for k, v in source_counts.items()])

    # V2C-specific outputs
    if vtype == "V2C" and head_ids is not None:
        np.save(os.path.join(map_dir, "head_token_ids.npy"), head_ids)
        tail_mask   = np.ones(vocab_size, dtype=bool)
        tail_mask[head_ids] = False
        tail_t2r = {k: v for k, v in t2r.items()}  # already excludes head
        with open(os.path.join(map_dir, "tail_token_to_region.json"), "w") as f:
            json.dump(tail_t2r, f)
        routed_r2t = dict(r2t)
        with open(os.path.join(map_dir, "routed_region_to_tokens.json"), "w") as f:
            json.dump(routed_r2t, f)
        policy = {"policy": "head_tail", "K_tail": K,
                  "head_size": len(head_ids),
                  "head_token_ids_file": "head_token_ids.npy"}
        with open(os.path.join(map_dir, "candidate_policy.json"), "w") as f:
            json.dump(policy, f, indent=2)
        # Head tokens text
        head_strs = []
        for t in head_ids[:200]:
            try: s = repr(tokenizer.decode([int(t)])) if tokenizer else f"<{int(t)}>"
            except Exception: s = f"<{int(t)}>"
            head_strs.append(f"{int(t):6d}  {s}")
        _write(os.path.join(map_dir, "head_tokens.txt"),
               "\n".join(head_strs))

    print(f"[save] {map_dir}/  (map complete)")


# ══════════════════════════════════════════════════════════════════════════════
# Final report
# ══════════════════════════════════════════════════════════════════════════════

def write_phase1b_report(all_diags: List[dict],
                          all_frag: List[dict],
                          audit_fb: dict,
                          old_map_diag: Optional[dict],
                          args,
                          out_dir: str) -> str:
    def _v(d, k): return d.get(k, float("nan")) if d else float("nan")

    # Rank maps by: smallest max_size / smallest num_over_2x / best intra_inter
    def _score(d):
        max_sz = _v(d, "max_size") or 1e6
        n_over = _v(d, "num_regions_over_2x_mean") or 100
        ratio  = _v(d, "intra_inter_ratio") or 0.0
        return -max_sz - 1000 * n_over + 10000 * ratio

    sorted_diags = sorted(all_diags, key=_score, reverse=True)
    best = sorted_diags[0] if sorted_diags else {}
    best_name = _v(best, "map_name")

    # Recommendation
    if not all_diags:
        rec = "NEED_MORE_MAP_VARIANTS"
    elif best_name.startswith("V2C") and "K512" in best_name:
        rec = "USE_V2C_K512_HEAD1024"
    elif best_name.startswith("V2C") and "K256" in best_name:
        rec = "USE_V2C_K256_HEAD1024"
    elif best_name.startswith("V2B") and "K512" in best_name:
        rec = "USE_V2B_K512"
    elif best_name.startswith("V2B") and "K256" in best_name:
        rec = "USE_V2B_K256"
    elif best_name.startswith("V2A") and "K512" in best_name:
        rec = "USE_V2A_K512"
    elif best_name.startswith("V2A") and "K256" in best_name:
        rec = "USE_V2A_K256"
    else:
        rec = "TRAIN_ROUTER_ON_TOP_2_MAPS"

    # Old map diagnostics
    old_max = _v(old_map_diag, "max_size") if old_map_diag else float("nan")
    old_n2x = _v(old_map_diag, "num_regions_over_2x_mean") if old_map_diag else float("nan")

    lines = [
        "# Phase 1B: Compute-Aware Predictive Interference Region Maps",
        "",
        f"**seed:** {args.seed}  |  **vocab_size:** {args.vocab_size}  |  "
        f"**bpe_lambda:** {args.bpe_lambda}",
        "", "---", "",
        "## Map Comparison Table", "",
        "| map | K | n_tokens | max_size | over_2x | intra_inter | size_gini | src_purity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in all_diags:
        lines.append(
            f"| {d.get('map_name','?')} | {d.get('n_regions','?')} "
            f"| {d.get('n_tokens','?'):,} | {d.get('max_size','?')} "
            f"| {d.get('num_regions_over_2x_mean','?')} "
            f"| {_fmt(_v(d,'intra_inter_ratio'))} "
            f"| {_fmt(_v(d,'size_gini'))} "
            f"| {_fmt(_v(d,'starts_with_space_purity'))} |")
    if old_map_diag:
        lines.append(
            f"| OLD_MAP_K128 | 128 | {_v(old_map_diag,'n_tokens')} "
            f"| {_v(old_map_diag,'max_size')} | {_v(old_map_diag,'num_regions_over_2x_mean')} "
            f"| {_fmt(_v(old_map_diag,'intra_inter_ratio'))} "
            f"| {_fmt(_v(old_map_diag,'size_gini'))} | — |")

    lines += ["", "---", "", "## Q&A", ""]

    def _q(n, q, ans, detail=""):
        lines.append(f"### Q{n}: {q}")
        lines.append(f"**{ans}**")
        if detail: lines.append(f"\n{detail}")
        lines.append("")

    best_max = _v(best, "max_size")
    _q(1, "Did balancing remove huge regions?",
       f"Old max={old_max}  Best new max={best_max}",
       f"Regions over 2x mean: old={old_n2x}  best_new={_v(best,'num_regions_over_2x_mean')}")
    _q(2, "Did BPE-aware compatibility reduce subword mixing?",
       "V2B/V2C maps include BPE surface penalty — see starts_with_space_purity column",
       f"V2B purity vs V2A: see table above")
    _q(3, "Did head-tail routing reduce top-k waste on generic tokens?",
       "V2C maps fix frequent head tokens — router focuses on informative tail tokens",
       f"Head size: {args.head_size}")
    _q(4, "Which map has the best size distribution?",
       f"{best_name}  (lowest max_size and over_2x score)")
    _q(5, "Which map has the best expected candidate compression?",
       f"See evaluate_compute_aware_region_maps.py output for oracle coverage curves")
    _q(6, "Which map should be used for router retraining?",
       f"Recommended: {rec}  (best size distribution by heuristic)")
    _q(7, "Does evidence support K=256 or K=512?",
       "K=512 gives finer granularity at cost of more routing classes. "
       "K=256 more robust for small routers. Test both.")
    _q(8, "Is full-vocab coverage fixed?",
       f"Full vocab ({args.vocab_size}) assigned via graph→embedding→surface→deterministic fallback")
    _q(9, "Did any map plausibly move toward 99% coverage?",
       "Run evaluate_compute_aware_region_maps.py to measure oracle coverage curves vs old map")
    _q(10, "Should Phase 1 proceed to routed candidate softmax?",
       "Train router on recommended map and re-audit. If recall@16 >= 0.90 at frac <= 0.25 → proceed.")

    lines += ["", "---", "",
              "## Problem Region Fragmentation", "",
              "*(Old regions 52, 127, 125, 121, 126, 124 tracking)*", ""]
    problem = [52, 127, 125, 121, 126, 124]
    prob_frags = [r for r in all_frag if r.get("is_problem_region") and r.get("old_region") in problem]
    if prob_frags:
        lines.append("| old_region | old_size | map | n_fragments | top5_new |")
        lines.append("|---|---|---|---|---|")
        for r in prob_frags:
            lines.append(f"| {r['old_region']} | {r['old_size']} | {r['new_map']} "
                         f"| {r['n_new_fragments']} | {r['top5_new_regions']} |")
    else:
        lines.append("*(No old-map fragmentation data available — run with --old_map)*")

    lines += ["", "---", "",
              "## Audit Feedback Integration", ""]
    if "miss_rank_histogram.csv" in audit_fb:
        lines.append("**Phase 1A.4 miss histogram available.** Key insight: "
                     "regions that missed rank 9-16 are candidates for splitting in V2+ maps.")
    else:
        lines.append("*(No audit feedback available — run Phase 1A.4 first for richer diagnostics)*")

    lines += ["", "---", "",
              f"## FINAL RECOMMENDATION: {rec}", "",
              f"*Train router on {rec.replace('USE_','').lower()} map and re-evaluate.*", ""]

    rpt_path = os.path.join(out_dir, "phase1B_region_map_report.md")
    _write(rpt_path, "\n".join(lines))
    return rec


# ══════════════════════════════════════════════════════════════════════════════
# Recall-per-cost curve (from gold distribution only, no router)
# ══════════════════════════════════════════════════════════════════════════════

def recall_per_cost_score(gold_labels: np.ndarray,
                           region_sizes: np.ndarray,
                           freq_order: np.ndarray,
                           K: int,
                           vocab_known_count: int,
                           ks: List[int]) -> dict:
    """
    Compute oracle recall at fixed candidate fractions using frequency-prior region ranking.
    Returns dict: k → {recall, cand_frac, score@k}.
    """
    out = {}
    N = len(gold_labels)
    for k in ks:
        k_ = min(k, K)
        top_regions = freq_order[:k_]
        top_set     = set(top_regions.tolist())
        hits        = sum(1 for g in gold_labels if int(g) in top_set and int(g) < K)
        cand_size   = int(region_sizes[top_regions].sum())
        recall      = hits / N if N > 0 else 0.0
        frac        = cand_size / max(vocab_known_count, 1)
        out[k] = {"recall": round(recall, 5), "cand_frac": round(frac, 5),
                  "score_recall_per_frac": round(recall / (frac + 1e-6), 4)}
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 1B: Build Compute-Aware Region Maps")
    p.add_argument("--train_dir",             required=True)
    p.add_argument("--val_dir",               default=None)
    p.add_argument("--old_map",               default=None)
    p.add_argument("--audit_dir",             default=None)
    p.add_argument("--small_ckpt",            default=None)
    p.add_argument("--output_root",           required=True)
    p.add_argument("--tokenizer_name",        default="gpt2")
    p.add_argument("--vocab_size",            type=int,   default=50257)
    p.add_argument("--graph_topk",            type=int,   default=64)
    p.add_argument("--pair_topk",             type=int,   default=32)
    p.add_argument("--top_neighbors_per_token", type=int, default=128)
    p.add_argument("--graph_embed_dim",       type=int,   default=64)
    p.add_argument("--variants",              type=str,
                   default="V2A_K256,V2A_K512,V2B_K256,V2B_K512,V2C_K256_head1024,V2C_K512_head1024")
    p.add_argument("--bpe_lambda",            type=float, default=0.75)
    p.add_argument("--head_size",             type=int,   default=1024)
    p.add_argument("--gold_edge_weight",      type=float, default=0.25)
    p.add_argument("--max_size_mult",         type=float, default=2.5)
    p.add_argument("--min_size_mult",         type=float, default=0.25)
    p.add_argument("--n_init_kmeans",         type=int,   default=3)
    p.add_argument("--seed",                  type=int,   default=42)
    p.add_argument("--no_tokenizer",          action="store_true")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # Memory sanity: warn if pair_topk * pair_topk * train_shards could OOM
    n_train_shards = len(glob.glob(os.path.join(args.train_dir, "shard_*.pt")))
    est_pairs_per_shard = 1000 * args.pair_topk * (args.pair_topk - 1) // 2
    est_gb = est_pairs_per_shard * n_train_shards * 12 / 1e9
    if est_gb > 8.0:
        print(f"[WARN] Estimated graph data ~{est_gb:.1f}GB — using per-shard CSR summation")

    os.makedirs(args.output_root, exist_ok=True)
    t0 = time.time()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        except Exception as e:
            print(f"[WARN] tokenizer failed: {e}")

    # ── Surface features ──────────────────────────────────────────────────────
    print("\n[step 1] Computing surface features...")
    surface_feats, feat_names = compute_surface_features(args.vocab_size, args.tokenizer_name)
    _wcsv(os.path.join(args.output_root, "token_surface_features.csv"),
          [{"token_id": i, **{fn: float(surface_feats[i, fi]) for fi, fn in enumerate(feat_names)}}
           for i in range(min(args.vocab_size, 500))])   # save first 500 as sample
    with open(os.path.join(args.output_root, "compatibility_config.json"), "w") as f:
        json.dump({"bpe_lambda": args.bpe_lambda, "feat_names": feat_names,
                   "tokenizer_name": args.tokenizer_name}, f, indent=2)

    # ── Build coactivation graph ───────────────────────────────────────────────
    print("\n[step 2] Building coactivation graph from train data...")
    W_raw, gold_freq, topk_freq = build_coactivation_graph(
        args.train_dir, args.vocab_size, args.graph_topk, args.pair_topk,
        args.top_neighbors_per_token, args.gold_edge_weight, args.seed)

    # Save graph stats
    token_stats = [{"token_id": i, "gold_freq": int(gold_freq[i]),
                    "topk_freq": round(float(topk_freq[i]), 2),
                    "graph_degree": int(W_raw.getrow(i).nnz)}
                   for i in range(args.vocab_size)]
    _wcsv(os.path.join(args.output_root, "token_stats.csv"), token_stats[:2000])  # sample
    try:
        from scipy.sparse import save_npz
        save_npz(os.path.join(args.output_root, "sparse_graph_norm.npz"), W_raw)
        print(f"[save] sparse_graph_norm.npz")
    except Exception as e:
        print(f"[WARN] Could not save graph npz: {e}")

    # ── Load old map & audit feedback ─────────────────────────────────────────
    old_map_arr = load_old_map(args.old_map)
    audit_fb    = load_audit_feedback(args.audit_dir)

    # ── Graph embeddings (shared) ─────────────────────────────────────────────
    print("\n[step 3] Computing graph embeddings (SVD)...")
    X_svd = compute_graph_embeddings(W_raw, args.graph_embed_dim, args.seed)

    # Tokens that appear in graph (degree > 0)
    degrees     = np.asarray(W_raw.sum(axis=1)).ravel()
    seen_ids    = np.where(degrees > 0)[0].astype(np.int32)
    X_seen      = X_svd[seen_ids]
    all_tok_ids = np.arange(args.vocab_size, dtype=np.int32)

    # ── BPE-penalised graph ───────────────────────────────────────────────────
    print("\n[step 4] Applying BPE penalty (for V2B/V2C)...")
    W_bpe = apply_bpe_penalty_to_graph(W_raw, surface_feats, args.bpe_lambda)
    X_bpe_svd = compute_graph_embeddings(W_bpe, args.graph_embed_dim, args.seed)
    X_bpe_seen = X_bpe_svd[seen_ids]

    # ── Head tokens (for V2C) ─────────────────────────────────────────────────
    head_ids_default = select_head_tokens(gold_freq, topk_freq, args.head_size)

    # ── Parse and run variants ────────────────────────────────────────────────
    variants  = [parse_variant(v.strip()) for v in args.variants.split(",") if v.strip()]
    print(f"\n[step 5] Building {len(variants)} map variants...")

    all_diags: List[dict] = []
    all_frag:  List[dict] = []
    problem_regions = [52, 127, 125, 121, 126, 124]

    for vcfg in variants:
        vname = vcfg["name"]
        vtype = vcfg["type"]
        K     = vcfg["K"]
        h_sz  = vcfg.get("head_size") or args.head_size

        if K is None:
            print(f"[WARN] Could not parse K from variant '{vname}', skipping"); continue
        if vtype not in ("V2A", "V2B", "V2C"):
            print(f"[WARN] Unknown variant type '{vtype}', skipping"); continue

        print(f"\n{'─'*60}")
        print(f"[variant] {vname}  K={K}  type={vtype}"
              + (f"  head_size={h_sz}" if vtype == "V2C" else ""))
        map_dir = os.path.join(args.output_root, vname)
        os.makedirs(map_dir, exist_ok=True)

        np.random.seed(args.seed); random.seed(args.seed)

        # Select embedding space
        if vtype == "V2A":
            X_for_cluster = X_seen        # raw SVD, no BPE
        else:
            X_for_cluster = X_bpe_seen    # BPE-penalised SVD

        # V2C: separate head and cluster tail
        head_ids = None
        if vtype == "V2C":
            head_ids = head_ids_default[:h_sz] if h_sz != args.head_size else head_ids_default
            head_set_seen = set(int(x) for x in head_ids).intersection(set(int(x) for x in seen_ids))
            # Cluster only tail tokens (seen tokens not in head)
            tail_seen_mask = np.array([int(x) not in set(int(h) for h in head_ids) for x in seen_ids])
            seen_ids_tail  = seen_ids[tail_seen_mask]
            X_tail         = X_for_cluster[tail_seen_mask]
            print(f"  tail seen tokens: {len(seen_ids_tail):,}  "
                  f"(head seen: {int(tail_seen_mask.size - tail_seen_mask.sum())})")
            seen_ids_use   = seen_ids_tail
            X_use          = X_tail
        else:
            seen_ids_use = seen_ids
            X_use        = X_for_cluster

        # Cluster
        t_cl = time.time()
        labels_seen = balanced_cluster(
            X_use, K, args.max_size_mult, args.min_size_mult,
            args.seed, args.n_init_kmeans)
        print(f"  clustering done {time.time()-t_cl:.1f}s")

        # Full vocab assignment
        t_fa = time.time()
        full_labels, source, assign_stats = assign_full_vocab(
            all_tok_ids, seen_ids_use, labels_seen, X_use,
            surface_feats, args.small_ckpt, K, args.seed)
        print(f"  fallback done {time.time()-t_fa:.1f}s  "
              f"graph={assign_stats['n_graph']}  surf={assign_stats['n_surface']}  "
              f"det={assign_stats['n_deterministic']}")

        # If V2C, set head tokens to K (unknown/excluded from routed map)
        if vtype == "V2C" and head_ids is not None:
            for h in head_ids: full_labels[int(h)] = K  # sentinel = out of range

        # Diagnostics
        labels_for_diag = full_labels[full_labels < K]  # exclude head tokens
        diag = compute_map_diagnostics(
            np.where(full_labels < K, full_labels, 0), W_raw, surface_feats, K, vname)
        all_diags.append(diag)

        # Old map fragmentation
        if old_map_arr is not None:
            frag_rows = old_region_fragmentation(old_map_arr, full_labels, K, vname, problem_regions)
            all_frag.extend(frag_rows)

        # Region representatives
        reps = build_representatives(full_labels, gold_freq, topk_freq, W_raw, K, tokenizer)

        # Save
        save_map_outputs(map_dir, vcfg, full_labels, source, reps, diag,
                         head_ids, K, args.vocab_size, assign_stats, tokenizer)

        # Save per-variant config
        with open(os.path.join(map_dir, "map_config.json"), "w") as f:
            json.dump({"variant": vname, "type": vtype, "K": K,
                       "head_size": h_sz if vtype == "V2C" else None,
                       "bpe_lambda": args.bpe_lambda if vtype in ("V2B","V2C") else 0.0,
                       "seed": args.seed, **assign_stats}, f, indent=2)

        sizes_arr = np.bincount(full_labels[full_labels < K], minlength=K).astype(np.int32)
        print(f"  max_size={sizes_arr.max()}  min={sizes_arr.min()}  "
              f"mean={sizes_arr.mean():.0f}  over_2x={int((sizes_arr > 2*sizes_arr.mean()).sum())}")

    # ── Old map diagnostics ───────────────────────────────────────────────────
    old_map_diag = None
    if old_map_arr is not None:
        old_K = int(old_map_arr.max()) if old_map_arr.max() < args.vocab_size else 128
        old_labels_clean = old_map_arr[:args.vocab_size].copy()
        old_labels_clean[old_labels_clean >= old_K] = 0
        old_map_diag = compute_map_diagnostics(old_labels_clean, W_raw, None, old_K, "OLD_K128")
        print(f"\n[old_map] diag: max={old_map_diag['max_size']}  "
              f"over_2x={old_map_diag['num_regions_over_2x_mean']}  "
              f"gini={old_map_diag['size_gini']}")

    # ── Save diagnostics CSV ──────────────────────────────────────────────────
    diag_all = all_diags + ([old_map_diag] if old_map_diag else [])
    _wcsv(os.path.join(args.output_root, "map_diagnostics.csv"), diag_all)
    if all_frag:
        _wcsv(os.path.join(args.output_root, "old_region_fragmentation.csv"),
              all_frag[:10000])

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n[step 6] Writing Phase 1B report...")
    recommendation = write_phase1b_report(all_diags, all_frag, audit_fb,
                                           old_map_diag, args, args.output_root)

    # ── Console summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f" PHASE 1B COMPUTE-AWARE REGION MAPS  ({elapsed/60:.1f} min)")
    print(f"{'='*65}")
    print(f"  {'map':<30}  K    max_sz  over_2x  intra_inter  gini")
    print(f"  {'─'*65}")
    for d in all_diags:
        print(f"  {d.get('map_name','?'):<30}  {d.get('n_regions','?'):4d}"
              f"  {d.get('max_size','?'):6}  {d.get('num_regions_over_2x_mean','?'):7}"
              f"  {_fmt(d.get('intra_inter_ratio',float('nan'))):11}"
              f"  {_fmt(d.get('size_gini',float('nan')))}")
    if old_map_diag:
        print(f"  {'OLD_K128':<30}  {old_map_diag.get('n_regions','?'):4d}"
              f"  {old_map_diag.get('max_size','?'):6}  {old_map_diag.get('num_regions_over_2x_mean','?'):7}"
              f"  {_fmt(old_map_diag.get('intra_inter_ratio',float('nan'))):11}"
              f"  {_fmt(old_map_diag.get('size_gini',float('nan')))}")
    print(f"\n  recommendation: {recommendation}")
    print(f"{'='*65}")
    print(f"\n  Next step: run evaluate_compute_aware_region_maps.py")
    print(f"  Then:      train router on recommended map via slurm_train_router_on_region_v2_maps.sh")


if __name__ == "__main__":
    main()
