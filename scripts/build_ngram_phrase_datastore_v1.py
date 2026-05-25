#!/usr/bin/env python3
"""
build_ngram_phrase_datastore_v1.py — Phase 1A: Build N-gram phrase datastore.

For each train row, records what token followed each suffix of length n
for n in ngram_lengths. Output is one file per ngram length containing:
  {suffix_tuple: {"counts": {tok_id: count}, "examples": [...]}}

No neural model. No validation data. Gold used only as the next-token target.

Usage:
  python scripts/build_ngram_phrase_datastore_v1.py \
    --train_dir runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train \
    --output_dir runs/phrase_memory_v1/ngram_phrase_recall/datastore \
    --ngram_lengths 4,8,16,32 \
    --max_train_rows -1 \
    --seed 42
"""

import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Shard loading
# ─────────────────────────────────────────────────────────────────────────────

_TOPK_ALIASES  = ["base_topk_ids",    "base_topk",     "topk_ids"]
_LGT_ALIASES   = ["base_topk_logits", "base_topk_lgt", "topk_lgt", "topk_logits"]
_GOLD_ALIASES  = ["gold_token",       "gold",          "labels"]
_IDS_ALIASES   = ["input_ids"]
_ROWID_ALIASES = ["row_id",           "row_ids"]
_OFF_ALIASES   = ["token_offset",     "offsets",       "offset"]

def _get(shard, aliases, required=True):
    for a in aliases:
        if a in shard:
            return shard[a]
    if required:
        raise KeyError(f"Missing field. Tried: {aliases}. Have: {list(shard.keys())}")
    return None

def iter_train_shards(train_dir, max_rows=None):
    """Yield (input_ids_np[N,seq], gold_np[N], row_ids_np[N], offsets_np[N]) per shard."""
    paths = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt found in {train_dir}")
    print(f"[shards] {len(paths)} train shard(s)")
    total = 0
    first = True
    for sp in paths:
        if max_rows is not None and max_rows >= 0 and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print("[preflight] first shard keys:")
            for k, v in sh.items():
                shp = getattr(v, "shape", None)
                dt  = getattr(v, "dtype", type(v).__name__)
                print(f"  {k}: shape={shp} dtype={dt}")
            first = False
        ids  = _get(sh, _IDS_ALIASES).long()
        gold = _get(sh, _GOLD_ALIASES).long()
        B    = ids.shape[0]
        row_ids = _get(sh, _ROWID_ALIASES, required=False)
        offsets = _get(sh, _OFF_ALIASES,   required=False)
        if row_ids is None:
            row_ids = torch.arange(total, total + B)
        if offsets is None:
            offsets = torch.zeros(B, dtype=torch.long)
        if max_rows is not None and max_rows >= 0 and total + B > max_rows:
            keep = max_rows - total
            ids, gold, row_ids, offsets = ids[:keep], gold[:keep], row_ids[:keep], offsets[:keep]
            B = keep
        yield ids.numpy(), gold.numpy(), row_ids.numpy(), offsets.numpy()
        total += B
        print(f"  shard {sp} — {B} rows  (total so far: {total:,})")
    print(f"[shards] done — {total:,} train rows ingested")

# ─────────────────────────────────────────────────────────────────────────────
# Datastore builder
# ─────────────────────────────────────────────────────────────────────────────

_MAX_EXAMPLES_PER_KEY = 4   # limit stored examples per key to bound memory

def build_datastores(train_dir, ngram_lengths, max_train_rows, seed):
    """
    Returns dict:  n -> {"counts": {key_tuple: {tok_id: int}},
                          "examples": {key_tuple: [example_dict]}}
    """
    rng = random.Random(seed)
    # One dict per ngram length
    stores = {}
    for n in ngram_lengths:
        stores[n] = {"counts": defaultdict(lambda: defaultdict(int)),
                     "examples": defaultdict(list)}

    num_rows = 0
    t0 = time.time()

    for ids_np, gold_np, row_ids_np, offsets_np in iter_train_shards(train_dir, max_train_rows):
        B, seq = ids_np.shape
        for i in range(B):
            ctx  = ids_np[i]          # [seq]
            gtok = int(gold_np[i])
            rid  = int(row_ids_np[i])
            off  = int(offsets_np[i])
            for n in ngram_lengths:
                if seq < n:
                    continue
                key = tuple(ctx[-n:].tolist())
                stores[n]["counts"][key][gtok] += 1
                ex_list = stores[n]["examples"][key]
                if len(ex_list) < _MAX_EXAMPLES_PER_KEY:
                    ex_list.append({"row_id": rid, "offset": off,
                                    "gold_token": gtok, "count_at_time": stores[n]["counts"][key][gtok]})
        num_rows += B

    elapsed = time.time() - t0
    print(f"[build] {num_rows:,} rows processed in {elapsed:.1f}s")
    return stores, num_rows

# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(stores, num_rows, ngram_lengths):
    stats = {"num_train_rows": num_rows, "per_n": {}}
    for n in ngram_lengths:
        counts_map = stores[n]["counts"]
        nkeys = len(counts_map)
        if nkeys == 0:
            stats["per_n"][n] = {"unique_keys": 0}
            continue
        vals_per_key = [sum(v.values()) for v in counts_map.values()]
        ntok_per_key = [len(v) for v in counts_map.values()]
        total_counts = sum(vals_per_key)
        stats["per_n"][n] = {
            "unique_keys":           nkeys,
            "total_next_token_count": total_counts,
            "mean_count_per_key":    float(np.mean(vals_per_key)),
            "max_count_per_key":     int(np.max(vals_per_key)),
            "mean_types_per_key":    float(np.mean(ntok_per_key)),
            "max_types_per_key":     int(np.max(ntok_per_key)),
            "singleton_keys":        int(sum(1 for v in vals_per_key if v == 1)),
            "keys_ge4":              int(sum(1 for v in vals_per_key if v >= 4)),
            "keys_ge16":             int(sum(1 for v in vals_per_key if v >= 16)),
        }
    return stats

# ─────────────────────────────────────────────────────────────────────────────
# Serialise
# ─────────────────────────────────────────────────────────────────────────────

def save_store(store_n, out_path):
    """Convert defaultdicts to plain dicts and save as pickle."""
    plain = {
        "counts":   {k: dict(v) for k, v in store_n["counts"].items()},
        "examples": {k: list(v) for k, v in store_n["examples"].items()},
    }
    with open(out_path, "wb") as f:
        pickle.dump(plain, f, protocol=4)
    sz = os.path.getsize(out_path) / 1e6
    print(f"  [save] {out_path}  ({sz:.1f} MB)")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Build n-gram phrase datastore")
    p.add_argument("--train_dir",     required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--ngram_lengths", default="4,8,16,32")
    p.add_argument("--max_train_rows", type=int, default=-1,
                   help="Use -1 for all rows")
    p.add_argument("--seed",          type=int, default=42)
    return p, p.parse_args()


def main():
    p, args = _parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ngram_lengths = [int(x) for x in args.ngram_lengths.split(",")]
    max_rows = args.max_train_rows if args.max_train_rows >= 0 else None

    print("=" * 60)
    print(" N-gram Phrase Datastore Builder V1")
    print(f" train_dir:     {args.train_dir}")
    print(f" output_dir:    {args.output_dir}")
    print(f" ngram_lengths: {ngram_lengths}")
    print(f" max_train_rows: {max_rows}")
    print("=" * 60)

    # Preflight
    if not os.path.isdir(args.train_dir):
        print(f"ERROR: train_dir not found: {args.train_dir}"); sys.exit(1)
    nshards = len(glob.glob(os.path.join(args.train_dir, "shard_*.pt")))
    if nshards == 0:
        print(f"ERROR: no shard_*.pt in {args.train_dir}"); sys.exit(1)
    print(f"[preflight] {nshards} train shards found")

    # Build
    stores, num_rows = build_datastores(args.train_dir, ngram_lengths, max_rows, args.seed)

    # Save
    print("\n[save] Writing datastore files...")
    for n in ngram_lengths:
        out = os.path.join(args.output_dir, f"ngram_{n}.pkl")
        save_store(stores[n], out)

    # Stats
    stats = compute_stats(stores, num_rows, ngram_lengths)
    stats_path = os.path.join(args.output_dir, "build_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats] {stats_path}")

    # Config
    cfg = {
        "train_dir":     args.train_dir,
        "ngram_lengths": ngram_lengths,
        "max_train_rows": args.max_train_rows,
        "seed":          args.seed,
        "num_train_rows": num_rows,
    }
    with open(os.path.join(args.output_dir, "build_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Console summary
    print("\n=== Datastore Summary ===")
    for n in ngram_lengths:
        s = stats["per_n"].get(n, {})
        print(f"  n={n:2d}  unique_keys={s.get('unique_keys',0):>10,}"
              f"  mean_count={s.get('mean_count_per_key',0):>7.2f}"
              f"  max_count={s.get('max_count_per_key',0):>8,}"
              f"  keys_ge4={s.get('keys_ge4',0):>7,}")

    print(f"\n[done] Datastore in: {args.output_dir}")


if __name__ == "__main__":
    main()
