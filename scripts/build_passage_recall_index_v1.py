#!/usr/bin/env python3
"""
build_passage_recall_index_v1.py — Passage Recall Dedup Diagnostic V1: Build index.

Indexes train passages for retrieval-based passage recall evaluation.
Train-only: validation data is NEVER ingested.

For each train row stores:
  {mode}_keys.mmap   : L2-normalised float16 query key  [N, D]
  gold_token.npy     : next token to predict             [N] int32
  continuations.npy  : next cont_len tokens at gold pos  [N, cont_len] int32
                       (col 0 = gold_token; -1 if corpus unavailable)
  train_prefixes.mmap: last query_len tokens of input_ids [N, query_len] int32
  row_id.npy         : [N] int32
  token_offset.npy   : [N] int64  (-1 if unavailable)
  config.json        : build hyperparameters
  build_stats.json   : runtime stats + modes_built

Modes:
  h_ctx        : L2-normalised h_ctx hidden   [N, h_dim]
  h_raw        : L2-normalised h_raw hidden   [N, h_dim]
  recency_bow  : recency-weighted BOW hash    [N, feature_dim]
  shingle5gram : 5-gram shingle hash BOW      [N, feature_dim]

Gold leakage rule: train rows only.  Val data never ingested.

Usage:
  python scripts/build_passage_recall_index_v1.py \\
    --train_dir runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train \\
    --output_dir runs/passage_memory_v1/passage_recall_dedup_v1/index \\
    --modes h_raw,h_ctx,recency_bow,shingle5gram \\
    --query_len 128 --continuation_len 8 --feature_dim 8192 \\
    --max_train_rows -1 --seed 42
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

_EPS = 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# Shard field aliases
# ─────────────────────────────────────────────────────────────────────────────

_IDS_ALIASES   = ["input_ids"]
_GOLD_ALIASES  = ["gold_token", "gold", "labels"]
_HCTX_ALIASES  = ["h_ctx", "h_context", "hidden_ctx"]
_HRAW_ALIASES  = ["h_raw", "h_backbone", "hidden_raw"]
_ROWID_ALIASES = ["row_id", "row_ids"]
_OFF_ALIASES   = ["token_offset", "offsets", "offset"]


def _get(shard, aliases, required=True):
    for a in aliases:
        if a in shard:
            return shard[a]
    if required:
        raise KeyError(f"Missing field. Tried: {aliases}. Have: {list(shard.keys())}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# L2 normalise helper
# ─────────────────────────────────────────────────────────────────────────────

def l2_norm(mat):
    n = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(n, _EPS)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised BOW / shingle feature builders (batch)
# ─────────────────────────────────────────────────────────────────────────────

def build_recency_bow_batch(ids_batch, query_len, feature_dim):
    """
    ids_batch: [B, seq_len] int array.
    Returns [B, feature_dim] float32, L2-normalised.
    Recency-weighted: most-recent token gets highest weight.
    """
    B, seq_len = ids_batch.shape
    T    = min(query_len, seq_len)
    tail = ids_batch[:, -T:].astype(np.int64)          # [B, T]
    dists   = np.arange(T - 1, -1, -1, dtype=np.float32)  # [T]
    weights = np.exp(-dists / max(T / 4.0, 1.0))           # [T]
    bins    = tail % feature_dim                            # [B, T]

    vecs = np.zeros((B, feature_dim), dtype=np.float32)
    # Scatter-add: for each position t, add weights[t] to vecs[b, bins[b,t]] for all b
    row_idx = np.repeat(np.arange(B, dtype=np.int64), T)   # [B*T]
    col_idx = bins.ravel()                                  # [B*T]
    w_flat  = np.tile(weights, B)                           # [B*T]
    np.add.at(vecs, (row_idx, col_idx), w_flat)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, _EPS)
    return vecs.astype(np.float32)


def build_shingle5gram_batch(ids_batch, query_len, feature_dim):
    """
    ids_batch: [B, seq_len] int array.
    Returns [B, feature_dim] float32, L2-normalised.
    5-gram shingle polynomial hash into feature_dim bins.
    """
    B, seq_len = ids_batch.shape
    T    = min(query_len, seq_len)
    tail = ids_batch[:, -T:].astype(np.int64)          # [B, T]
    n_sh = T - 4
    vecs = np.zeros((B, feature_dim), dtype=np.float32)
    if n_sh <= 0:
        return vecs

    # Build 5-gram indices: [n_sh, 5]
    offsets = np.arange(5, dtype=np.int64)
    starts  = np.arange(n_sh, dtype=np.int64)
    idx_mat = starts[:, None] + offsets[None, :]         # [n_sh, 5]

    # tail[:, idx_mat] → [B, n_sh, 5]
    shingles = tail[:, idx_mat]                           # [B, n_sh, 5]

    # Polynomial hash: p = [1, 31, 961, 29791, 923521]
    p      = np.array([1, 31, 961, 29791, 923521], dtype=np.int64)
    hashes = (shingles * p[None, None, :]).sum(axis=2) % feature_dim  # [B, n_sh]
    hashes = np.abs(hashes).astype(np.int64)

    row_idx = np.repeat(np.arange(B, dtype=np.int64), n_sh)   # [B*n_sh]
    col_idx = hashes.ravel()                                   # [B*n_sh]
    np.add.at(vecs, (row_idx, col_idx), 1.0)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, _EPS)
    return vecs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus loading (3-tier fallback)
# ─────────────────────────────────────────────────────────────────────────────

def try_load_corpus(corpus_path_arg, train_dir):
    """
    Tier 1: --corpus_path arg (explicit).
    Tier 2: pipeline cache at .../00_raw_token_source/train_tokens.npy.
    Tier 3: None — continuations will be gold_token only.
    Returns np.ndarray int32 or None.
    """
    if corpus_path_arg:
        if os.path.isfile(corpus_path_arg):
            print(f"[corpus] loading from --corpus_path: {corpus_path_arg}")
            c = np.load(corpus_path_arg)
            print(f"[corpus] loaded: shape={c.shape}  dtype={c.dtype}")
            return c.astype(np.int32)
        print(f"[corpus] --corpus_path not found: {corpus_path_arg}")

    # Pipeline cache: train_dir is .../01_live_dataset_patched/train
    # corpus is at  .../00_raw_token_source/train_tokens.npy
    cand = os.path.normpath(
        os.path.join(train_dir, "..", "..", "00_raw_token_source", "train_tokens.npy"))
    if os.path.isfile(cand):
        print(f"[corpus] loading from pipeline cache: {cand}")
        c = np.load(cand)
        print(f"[corpus] loaded: shape={c.shape}  dtype={c.dtype}")
        return c.astype(np.int32)

    print("[corpus] WARNING: no train corpus found — continuations will be gold_token only")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def build_index(train_dir, output_dir, modes, query_len, continuation_len,
                feature_dim, max_train_rows, seed, corpus_path_arg=None):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    t_start = time.time()

    paths = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {train_dir}")
    print(f"[shards] {len(paths)} train shards in {train_dir}")

    # ── Probe first shard ──────────────────────────────────────────────────
    print("[preflight] first shard keys:")
    first_sh = torch.load(paths[0], map_location="cpu", weights_only=False)
    shard_schema = {}
    for k, v in first_sh.items():
        shp = getattr(v, "shape", None)
        dt  = getattr(v, "dtype", type(v).__name__)
        print(f"  {k}: shape={shp}  dtype={dt}")
        shard_schema[k] = {"shape": str(shp), "dtype": str(dt)}

    h_dim = None
    for aliases in [_HCTX_ALIASES, _HRAW_ALIASES]:
        h = _get(first_sh, aliases, required=False)
        if h is not None:
            h_dim = h.shape[-1] if h.ndim >= 2 else int(h.shape[0])
            break

    has_hctx = _get(first_sh, _HCTX_ALIASES, required=False) is not None
    has_hraw = _get(first_sh, _HRAW_ALIASES, required=False) is not None
    has_off  = _get(first_sh, _OFF_ALIASES, required=False) is not None
    print(f"[preflight] h_ctx={has_hctx}  h_raw={has_hraw}  "
          f"token_offset={has_off}  h_dim={h_dim}")

    # Filter modes by availability
    active_modes = []
    for m in modes:
        if m == "h_ctx" and not has_hctx:
            print(f"[WARN] mode h_ctx requested but h_ctx absent in shards — skip"); continue
        if m == "h_raw" and not has_hraw:
            print(f"[WARN] mode h_raw requested but h_raw absent in shards — skip"); continue
        if m in ("h_ctx", "h_raw") and h_dim is None:
            print(f"[WARN] mode {m}: h_dim unknown — skip"); continue
        active_modes.append(m)
    if not active_modes:
        raise RuntimeError("No valid modes remain after availability check.")
    print(f"[modes] active: {active_modes}")

    # ── Count rows (fast first pass) ───────────────────────────────────────
    print("[count] counting train rows...")
    N = 0
    for sp in paths:
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        N += _get(sh, _IDS_ALIASES).shape[0]
        if max_train_rows > 0 and N >= max_train_rows:
            N = min(N, max_train_rows)
            break
    print(f"[count] N = {N:,}")

    # ── Load corpus ────────────────────────────────────────────────────────
    corpus = try_load_corpus(corpus_path_arg, train_dir)
    has_corpus  = corpus is not None
    corpus_len  = len(corpus) if has_corpus else 0

    # ── Pre-allocate memmaps ───────────────────────────────────────────────
    print("[alloc] pre-allocating memmaps...")
    key_mmaps  = {}
    key_dims   = {}
    for m in list(active_modes):
        D = h_dim if m in ("h_ctx", "h_raw") else feature_dim
        if D is None:
            print(f"[WARN] unknown dim for mode {m} — skip"); active_modes.remove(m); continue
        p = os.path.join(output_dir, f"{m}_keys.mmap")
        key_mmaps[m]  = np.memmap(p, dtype=np.float16, mode="w+", shape=(N, D))
        key_dims[m]   = D
        print(f"  {m}_keys.mmap  [{N}, {D}]  float16  "
              f"({N * D * 2 / 1e6:.1f} MB)")

    pref_path = os.path.join(output_dir, "train_prefixes.mmap")
    pref_mmap = np.memmap(pref_path, dtype=np.int32, mode="w+", shape=(N, query_len))
    print(f"  train_prefixes.mmap  [{N}, {query_len}]  int32  "
          f"({N * query_len * 4 / 1e6:.1f} MB)")

    gold_arr  = np.full(N, -1, dtype=np.int32)
    cont_arr  = np.full((N, continuation_len), -1, dtype=np.int32)
    rowid_arr = np.zeros(N, dtype=np.int32)
    off_arr   = np.full(N, -1, dtype=np.int64)

    # ── Second pass: fill arrays ───────────────────────────────────────────
    print("[build] filling index...")
    ptr = 0
    t_last = time.time()
    for sp in paths:
        if ptr >= N:
            break
        sh   = torch.load(sp, map_location="cpu", weights_only=False)
        ids  = _get(sh, _IDS_ALIASES).long().numpy()     # [B, seq_len]
        gold = _get(sh, _GOLD_ALIASES).long().numpy()    # [B]
        rids = _get(sh, _ROWID_ALIASES, required=False)
        off  = _get(sh, _OFF_ALIASES,   required=False)

        B = ids.shape[0]
        if ptr + B > N:
            B    = N - ptr
            ids  = ids[:B]
            gold = gold[:B]
            if rids is not None: rids = rids[:B]
            if off  is not None: off  = off[:B]

        # Convert helpers
        if rids is None:
            rids_np = np.arange(ptr, ptr + B, dtype=np.int32)
        else:
            rids_np = (rids.numpy() if hasattr(rids, "numpy") else np.asarray(rids))[:B].astype(np.int32)
        if off is None:
            off_np = np.full(B, -1, dtype=np.int64)
        else:
            off_np = (off.numpy() if hasattr(off, "numpy") else np.asarray(off))[:B].astype(np.int64)

        # ── Scalar arrays ────────────────────────────────────────────────
        gold_arr[ptr:ptr+B]  = gold.astype(np.int32)
        rowid_arr[ptr:ptr+B] = rids_np
        off_arr[ptr:ptr+B]   = off_np

        # ── Continuations ────────────────────────────────────────────────
        cont_arr[ptr:ptr+B, 0] = gold.astype(np.int32)   # col 0 always = gold
        if has_corpus:
            valid = off_np >= 0                           # [B] bool
            for k in range(continuation_len):
                pos      = off_np + k                    # [B]
                in_range = valid & (pos >= 0) & (pos < corpus_len)
                if in_range.any():
                    idxs = np.where(in_range)[0]
                    cont_arr[ptr + idxs, k] = corpus[pos[idxs]].astype(np.int32)

        # ── Train prefixes ────────────────────────────────────────────────
        seq_len = ids.shape[1]
        T       = min(seq_len, query_len)
        tail    = ids[:B, -T:].astype(np.int32)          # [B, T]
        if T < query_len:
            padded = np.zeros((B, query_len), dtype=np.int32)
            padded[:, query_len - T:] = tail
            pref_mmap[ptr:ptr+B] = padded
        else:
            pref_mmap[ptr:ptr+B] = tail

        # ── Key vectors ───────────────────────────────────────────────────
        for m in active_modes:
            if m == "h_ctx":
                hv = _get(sh, _HCTX_ALIASES, required=False)
                if hv is None: continue
                hv_np = hv.float().numpy()[:B]
                key_mmaps[m][ptr:ptr+B] = l2_norm(hv_np).astype(np.float16)
            elif m == "h_raw":
                hv = _get(sh, _HRAW_ALIASES, required=False)
                if hv is None: continue
                hv_np = hv.float().numpy()[:B]
                key_mmaps[m][ptr:ptr+B] = l2_norm(hv_np).astype(np.float16)
            elif m == "recency_bow":
                buf = build_recency_bow_batch(ids[:B], query_len, feature_dim)
                key_mmaps[m][ptr:ptr+B] = buf.astype(np.float16)
            elif m == "shingle5gram":
                buf = build_shingle5gram_batch(ids[:B], query_len, feature_dim)
                key_mmaps[m][ptr:ptr+B] = buf.astype(np.float16)

        ptr += B
        if ptr % 50000 < B or ptr == N:
            elapsed = time.time() - t_last
            t_last  = time.time()
            print(f"  [{ptr:,}/{N:,}]  +{elapsed:.1f}s")

    # ── Flush / validate / save ────────────────────────────────────────────
    for mm in key_mmaps.values():
        mm.flush()
    pref_mmap.flush()

    print("[validate] checking first 1000 rows for NaN/Inf...")
    for m, mm in key_mmaps.items():
        chunk = mm[:min(1000, ptr)].astype(np.float32)
        if np.isnan(chunk).any() or np.isinf(chunk).any():
            raise RuntimeError(f"NaN/Inf in key matrix: {m}")
    print("  [OK]")

    np.save(os.path.join(output_dir, "gold_token.npy"),    gold_arr[:ptr])
    np.save(os.path.join(output_dir, "continuations.npy"), cont_arr[:ptr])
    np.save(os.path.join(output_dir, "row_id.npy"),        rowid_arr[:ptr])
    np.save(os.path.join(output_dir, "token_offset.npy"),  off_arr[:ptr])
    print(f"[save] arrays saved  (N={ptr:,})")

    # ── Config & stats ─────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    cfg = {
        "train_dir":        train_dir,
        "query_len":        query_len,
        "continuation_len": continuation_len,
        "feature_dim":      feature_dim,
        "modes_requested":  modes,
        "modes_built":      active_modes,
        "num_train_rows":   ptr,
        "h_dim":            int(h_dim) if h_dim else None,
        "max_train_rows":   max_train_rows,
        "seed":             seed,
        "has_corpus":       has_corpus,
        "corpus_len":       int(corpus_len),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    stats = {
        "num_train_rows":   ptr,
        "modes_built":      active_modes,
        "h_dim":            int(h_dim) if h_dim else None,
        "feature_dim":      feature_dim,
        "has_corpus":       has_corpus,
        "corpus_len":       int(corpus_len),
        "elapsed_s":        round(elapsed_total, 1),
    }
    with open(os.path.join(output_dir, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[done] N={ptr:,}  modes={active_modes}  elapsed={elapsed_total:.1f}s")
    print(f"[done] output_dir: {output_dir}")
    for m, mm in key_mmaps.items():
        p = os.path.join(output_dir, f"{m}_keys.mmap")
        print(f"  {m}_keys.mmap  {os.path.getsize(p) / 1e6:.1f} MB")
    print(f"  train_prefixes.mmap  {os.path.getsize(pref_path) / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Build passage recall index V1")
    p.add_argument("--train_dir",        required=True,
                   help="Path to train shards directory (shard_*.pt)")
    p.add_argument("--output_dir",       required=True,
                   help="Where to write the index")
    p.add_argument("--modes",            default="h_raw,h_ctx,recency_bow,shingle5gram",
                   help="Comma-separated modes: h_raw,h_ctx,recency_bow,shingle5gram")
    p.add_argument("--query_len",        type=int, default=128,
                   help="Number of context tokens to use as query key")
    p.add_argument("--continuation_len", type=int, default=8,
                   help="Number of continuation tokens to store (incl. gold at pos 0)")
    p.add_argument("--feature_dim",      type=int, default=8192,
                   help="BOW/shingle feature dimension")
    p.add_argument("--max_train_rows",   type=int, default=-1,
                   help="-1 for all rows")
    p.add_argument("--corpus_path",      default=None,
                   help="Explicit path to train_tokens.npy (optional)")
    p.add_argument("--seed",             type=int, default=42)
    return p, p.parse_args()


def main():
    _, args = _parse()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print("=" * 64)
    print(" Passage Recall Index Builder V1")
    print(f" train_dir:        {args.train_dir}")
    print(f" output_dir:       {args.output_dir}")
    print(f" modes:            {modes}")
    print(f" query_len:        {args.query_len}")
    print(f" continuation_len: {args.continuation_len}")
    print(f" feature_dim:      {args.feature_dim}")
    print(f" max_train_rows:   {args.max_train_rows}")
    print("=" * 64)

    if not os.path.isdir(args.train_dir):
        print(f"ERROR: train_dir not found: {args.train_dir}"); sys.exit(1)
    nshards = len(glob.glob(os.path.join(args.train_dir, "shard_*.pt")))
    if nshards == 0:
        print(f"ERROR: no shard_*.pt in {args.train_dir}"); sys.exit(1)
    print(f"[preflight] {nshards} train shards found")

    build_index(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        modes=modes,
        query_len=args.query_len,
        continuation_len=args.continuation_len,
        feature_dim=args.feature_dim,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
        corpus_path_arg=args.corpus_path,
    )


if __name__ == "__main__":
    main()
