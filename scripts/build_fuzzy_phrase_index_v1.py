#!/usr/bin/env python3
"""
build_fuzzy_phrase_index_v1.py — Phase 2A: Build fuzzy phrase retrieval index.

For each train row builds query keys in four modes:
  h_ctx       : normalized h_ctx hidden vector  [N, D]
  h_raw       : normalized h_raw hidden vector  [N, D]
  bow         : hashed bag-of-tokens (last phrase_len tokens)  [N, F]
  recency_bow : same with recency weighting                    [N, F]

Saved as float16 memmap arrays. Companion arrays: gold_token, row_id,
token_offset saved as int32/int64 .npy.

Gold leakage rule: train rows only. Val data never ingested.

Usage:
  python scripts/build_fuzzy_phrase_index_v1.py \
    --train_dir runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train \
    --output_dir runs/phrase_memory_v1/fuzzy_phrase_recall/index \
    --modes h_ctx,h_raw,recency_bow,bow \
    --phrase_lens 16,32,64 \
    --feature_dim 8192 \
    --max_train_rows -1 \
    --seed 42
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
# First-pass: count rows to pre-allocate memmaps
# ─────────────────────────────────────────────────────────────────────────────

def count_train_rows(train_dir, max_rows):
    paths = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    total = 0
    for sp in paths:
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        ids = _get(sh, _IDS_ALIASES)
        total += ids.shape[0]
        if max_rows > 0 and total >= max_rows:
            return min(total, max_rows), paths
    return total, paths

# ─────────────────────────────────────────────────────────────────────────────
# BOW feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_bow_vector(ctx_np, phrase_len, feature_dim, recency=False):
    """
    ctx_np: [seq] int array
    Returns float32 L2-normalised dense vector [feature_dim].
    Uses hash trick: token_id % feature_dim → bin.
    """
    seq = len(ctx_np)
    T   = min(phrase_len, seq)
    tail = ctx_np[-T:].astype(np.int64)
    vec  = np.zeros(feature_dim, dtype=np.float32)
    if recency:
        dists = np.arange(T-1, -1, -1, dtype=np.float32)   # 0 = most recent
        weights = np.exp(-dists / max(T / 4.0, 1.0))
    else:
        weights = np.ones(T, dtype=np.float32)
    bins = tail % feature_dim
    np.add.at(vec, bins, weights)
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    return vec

# ─────────────────────────────────────────────────────────────────────────────
# L2 normalise helper (safe)
# ─────────────────────────────────────────────────────────────────────────────

def l2_norm(vec):
    n = np.linalg.norm(vec, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-9)
    return vec / n

# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def build_index(train_dir, output_dir, modes, phrase_lens, feature_dim,
                max_train_rows, seed):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    paths = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {train_dir}")
    print(f"[shards] {len(paths)} train shards")

    # First pass: count rows and probe shard schema
    first_sh = torch.load(paths[0], map_location="cpu", weights_only=False)
    print("[preflight] first shard keys:")
    shard_keys = {}
    for k, v in first_sh.items():
        shp = getattr(v, "shape", None)
        dt  = getattr(v, "dtype", type(v).__name__)
        print(f"  {k}: shape={shp} dtype={dt}")
        shard_keys[k] = {"shape": str(shp), "dtype": str(dt)}

    # Detect hidden dim
    h_dim = None
    for aliases in [_HCTX_ALIASES, _HRAW_ALIASES]:
        h = _get(first_sh, aliases, required=False)
        if h is not None:
            h_dim = h.shape[-1] if h.ndim >= 2 else int(h.shape[0])
            break

    has_hctx = _get(first_sh, _HCTX_ALIASES, required=False) is not None
    has_hraw = _get(first_sh, _HRAW_ALIASES, required=False) is not None
    print(f"[preflight] h_ctx_available={has_hctx}  h_raw_available={has_hraw}  h_dim={h_dim}")

    if "h_ctx" in modes and not has_hctx:
        print("[WARN] h_ctx mode requested but h_ctx not in shards — skipping mode")
        modes = [m for m in modes if m != "h_ctx"]
    if "h_raw" in modes and not has_hraw:
        print("[WARN] h_raw mode requested but h_raw not in shards — skipping mode")
        modes = [m for m in modes if m != "h_raw"]
    if not modes:
        raise RuntimeError("No valid modes remain after availability check.")

    # Count rows
    print("[count] Counting train rows (fast pass)...")
    total_rows = 0
    for sp in paths:
        sh  = torch.load(sp, map_location="cpu", weights_only=False)
        total_rows += _get(sh, _IDS_ALIASES).shape[0]
        if max_train_rows > 0 and total_rows >= max_train_rows:
            total_rows = min(total_rows, max_train_rows)
            break
    N = total_rows
    print(f"[count] N = {N:,} train rows")

    # Pre-allocate companion arrays
    gold_arr   = np.zeros(N, dtype=np.int32)
    rowid_arr  = np.zeros(N, dtype=np.int64)
    offset_arr = np.zeros(N, dtype=np.int64)

    # Pre-allocate key memmaps
    key_mmap = {}
    key_paths = {}
    if "h_ctx" in modes and h_dim:
        p = os.path.join(output_dir, "h_ctx_keys.mmap")
        key_mmap["h_ctx"] = np.memmap(p, dtype=np.float16, mode="w+", shape=(N, h_dim))
        key_paths["h_ctx"] = p
    if "h_raw" in modes and h_dim:
        p = os.path.join(output_dir, "h_raw_keys.mmap")
        key_mmap["h_raw"] = np.memmap(p, dtype=np.float16, mode="w+", shape=(N, h_dim))
        key_paths["h_raw"] = p
    for plen in phrase_lens:
        for mode in ["bow", "recency_bow"]:
            if mode in modes:
                name = f"{mode}_len{plen}"
                p = os.path.join(output_dir, f"{name}_keys.mmap")
                key_mmap[name] = np.memmap(p, dtype=np.float16, mode="w+", shape=(N, feature_dim))
                key_paths[name] = p

    # Second pass: fill
    print("[build] Filling index arrays...")
    t0 = time.time()
    ptr = 0
    for sp in paths:
        if ptr >= N:
            break
        sh   = torch.load(sp, map_location="cpu", weights_only=False)
        ids  = _get(sh, _IDS_ALIASES).long()
        gold = _get(sh, _GOLD_ALIASES).long()
        B    = ids.shape[0]
        row_ids = _get(sh, _ROWID_ALIASES, required=False)
        offsets = _get(sh, _OFF_ALIASES,   required=False)
        if row_ids is None: row_ids = torch.arange(ptr, ptr + B)
        if offsets is None: offsets = torch.zeros(B, dtype=torch.long)

        if ptr + B > N:
            B = N - ptr
            ids, gold, row_ids, offsets = ids[:B], gold[:B], row_ids[:B], offsets[:B]

        gold_arr[ptr:ptr+B]   = gold.numpy().astype(np.int32)
        rowid_arr[ptr:ptr+B]  = row_ids.numpy().astype(np.int64)
        offset_arr[ptr:ptr+B] = offsets.numpy().astype(np.int64)

        # h_ctx
        if "h_ctx" in key_mmap:
            hc = _get(sh, _HCTX_ALIASES).float().numpy()[:B]
            key_mmap["h_ctx"][ptr:ptr+B] = l2_norm(hc).astype(np.float16)

        # h_raw
        if "h_raw" in key_mmap:
            hr = _get(sh, _HRAW_ALIASES).float().numpy()[:B]
            key_mmap["h_raw"][ptr:ptr+B] = l2_norm(hr).astype(np.float16)

        # BOW modes
        ids_np = ids.numpy()
        for plen in phrase_lens:
            for mode in ["bow", "recency_bow"]:
                name = f"{mode}_len{plen}"
                if name not in key_mmap:
                    continue
                use_recency = (mode == "recency_bow")
                chunk = np.stack(
                    [build_bow_vector(ids_np[i], plen, feature_dim, use_recency)
                     for i in range(B)], axis=0)
                key_mmap[name][ptr:ptr+B] = chunk.astype(np.float16)

        ptr += B
        if ptr % 50000 == 0 or ptr == N:
            elapsed = time.time() - t0
            print(f"  {ptr:,}/{N:,} rows  ({elapsed:.1f}s)")

    # Flush memmaps
    for mm in key_mmap.values():
        mm.flush()

    # Save companion arrays
    np.save(os.path.join(output_dir, "values_gold_token.npy"), gold_arr)
    np.save(os.path.join(output_dir, "row_id.npy"),            rowid_arr)
    np.save(os.path.join(output_dir, "token_offset.npy"),      offset_arr)
    print(f"[save] Companion arrays saved")

    # NaN check
    print("[validate] Checking for NaNs in key matrices...")
    for name, mm in key_mmap.items():
        chunk = mm[:min(1000, N)].astype(np.float32)
        if np.isnan(chunk).any() or np.isinf(chunk).any():
            raise RuntimeError(f"NaN/Inf detected in key matrix: {name}")
    print("  [OK] No NaNs detected")

    # Stats
    stats = {
        "num_train_rows": int(N),
        "h_dim":          int(h_dim) if h_dim else None,
        "feature_dim":    int(feature_dim),
        "modes_built":    list(key_mmap.keys()),
        "key_paths":      key_paths,
    }
    with open(os.path.join(output_dir, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    cfg = {
        "train_dir":     train_dir,
        "modes":         modes,
        "phrase_lens":   phrase_lens,
        "feature_dim":   feature_dim,
        "max_train_rows":max_train_rows,
        "seed":          seed,
        "num_train_rows":int(N),
        "h_dim":         int(h_dim) if h_dim else None,
        # also store modes_built here so eval can find it in one file
        "modes_built":   list(key_mmap.keys()),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[done] {N:,} rows indexed in {elapsed:.1f}s")
    print(f"  output: {output_dir}")
    for name in key_mmap:
        path = key_paths[name]
        sz   = os.path.getsize(path) / 1e6
        print(f"  {name}: {sz:.1f} MB")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Build fuzzy phrase retrieval index")
    p.add_argument("--train_dir",     required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--modes",         default="h_ctx,h_raw,recency_bow,bow")
    p.add_argument("--phrase_lens",   default="16,32,64")
    p.add_argument("--feature_dim",   type=int, default=8192)
    p.add_argument("--max_train_rows",type=int, default=-1,
                   help="Use -1 for all rows")
    p.add_argument("--seed",          type=int, default=42)
    return p, p.parse_args()


def main():
    p, args = _parse()
    modes       = [m.strip() for m in args.modes.split(",") if m.strip()]
    phrase_lens = [int(x)    for x in args.phrase_lens.split(",")]
    max_rows    = args.max_train_rows   # -1 = all

    print("=" * 60)
    print(" Fuzzy Phrase Index Builder V1")
    print(f" train_dir:  {args.train_dir}")
    print(f" output_dir: {args.output_dir}")
    print(f" modes:      {modes}")
    print(f" phrase_lens:{phrase_lens}")
    print(f" feature_dim:{args.feature_dim}")
    print(f" max_rows:   {max_rows}")
    print("=" * 60)

    if not os.path.isdir(args.train_dir):
        print(f"ERROR: train_dir not found: {args.train_dir}"); sys.exit(1)
    nshards = len(glob.glob(os.path.join(args.train_dir, "shard_*.pt")))
    if nshards == 0:
        print(f"ERROR: no shard_*.pt in {args.train_dir}"); sys.exit(1)
    print(f"[preflight] {nshards} train shards found")

    build_index(args.train_dir, args.output_dir, modes, phrase_lens,
                args.feature_dim, max_rows, args.seed)


if __name__ == "__main__":
    main()
