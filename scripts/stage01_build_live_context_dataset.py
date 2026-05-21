#!/usr/bin/env python3
"""
Stage 01 — Build Live Context Dataset.

Reconstructs full ctx_len context windows from WikiText-103, aligned row-for-row
with the existing candidate shards (train + val).

Context convention: input_ids[i] = tokens[pos-ctx_len+1 : pos+1] (left-pad if short)
                    gold_token[i] = tokens[pos+1]

Hard pass conditions:
  gold_token alignment with candidate shards (every row verified)
  row counts match exactly
  token IDs in valid range
  no future leakage (context ends strictly before gold_token)
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch
from transformers import GPT2TokenizerFast

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_wikitext


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_coarse_map(region_map_path: str, vocab_size: int = 50257) -> np.ndarray:
    """Returns int32 (vocab_size,): token → fine_region_id, -1 = unmapped."""
    with open(region_map_path) as f:
        raw = json.load(f)
    arr = np.full(vocab_size, -1, dtype=np.int32)
    for k, v in raw.items():
        tok = int(k)
        if 0 <= tok < vocab_size:
            arr[tok] = int(v)
    return arr


def collect_shard_gold_tokens(cand_dir: str) -> List[np.ndarray]:
    paths = sorted([
        os.path.join(cand_dir, f)
        for f in os.listdir(cand_dir)
        if f.startswith("shard_") and f.endswith(".pt")
    ])
    if not paths:
        raise RuntimeError(f"No shard_*.pt in {cand_dir}")
    result = []
    for p in paths:
        cs = torch.load(p, map_location="cpu", weights_only=True)
        result.append(cs["gold_token"].numpy().astype(np.int32))
    return result  # list[shard_idx] → (N_shard,) int32


def scan_corpus(
    corpus: np.ndarray,
    coarse_map: np.ndarray,
    need: int,
) -> np.ndarray:
    """
    Returns positions (need,) int64 — corpus indices of the last context token.
    gold_token = corpus[pos+1].

    Uses numpy vectorization for speed.
    """
    # next token at each position
    next_toks = corpus[1:].astype(np.int64).clip(0, len(coarse_map) - 1)
    valid = coarse_map[next_toks] >= 0          # (len(corpus)-1,) bool
    positions = np.where(valid)[0].astype(np.int64)  # corpus index of context-end token

    if len(positions) < need:
        raise RuntimeError(
            f"Only {len(positions):,} valid positions in corpus, need {need:,}. "
            "Check region_map and corpus split."
        )
    return positions[:need]


def build_ctx_shard(
    positions: np.ndarray,          # (N,) int64 — corpus index of context-end
    corpus: np.ndarray,             # full corpus tokens
    shard_golds: np.ndarray,        # (N,) int32 expected gold tokens (from cand shard)
    ctx_len: int,
    row_id_start: int,
    fail_on_err: bool,
) -> dict:
    N = len(positions)
    input_ids      = np.zeros((N, ctx_len), dtype=np.int32)
    attention_mask = np.zeros((N, ctx_len), dtype=np.uint8)
    gold_out       = np.empty(N, dtype=np.int32)
    row_ids        = np.arange(row_id_start, row_id_start + N, dtype=np.int64)

    for i in range(N):
        pos = int(positions[i])
        gold = int(corpus[pos + 1])

        # Alignment check
        if gold != int(shard_golds[i]):
            msg = (f"Alignment FAIL at row {i} (global {row_id_start + i}): "
                   f"corpus_gold={gold} shard_gold={int(shard_golds[i])}")
            if fail_on_err:
                raise RuntimeError(msg)
            print(f"  WARNING: {msg}")

        gold_out[i] = gold

        # Left-pad context to ctx_len
        start = max(0, pos + 1 - ctx_len)
        end   = pos + 1
        seq   = corpus[start:end].astype(np.int32)
        L     = len(seq)
        input_ids[i, ctx_len - L:]      = seq
        attention_mask[i, ctx_len - L:] = 1

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "gold_token":     gold_out,
        "row_id":         row_ids,
    }


def process_split(
    split: str,
    cand_dir: str,
    corpus: np.ndarray,
    coarse_map: np.ndarray,
    ctx_len: int,
    out_dir: str,
    fail_on_err: bool,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    print(f"  [{split}] Loading shard gold tokens from {cand_dir} ...")
    shard_golds_list = collect_shard_gold_tokens(cand_dir)
    n_shards   = len(shard_golds_list)
    total_rows = sum(len(g) for g in shard_golds_list)
    print(f"  [{split}] {n_shards} shards, {total_rows:,} total rows")

    # Scan corpus once to get ALL valid positions (vectorized)
    t0 = time.time()
    positions = scan_corpus(corpus, coarse_map, total_rows)
    print(f"  [{split}] Corpus scan done  t={time.time()-t0:.1f}s")

    alignment_ok = True
    row_offset   = 0

    for si, shard_golds in enumerate(shard_golds_list):
        N = len(shard_golds)
        pos_shard = positions[row_offset: row_offset + N]
        corpus_golds = corpus[pos_shard + 1].astype(np.int32)

        mismatches = int((corpus_golds != shard_golds).sum())
        if mismatches > 0:
            alignment_ok = False
            msg = (f"[{split}] shard {si:05d}: {mismatches} gold mismatches out of {N}")
            if fail_on_err:
                raise RuntimeError(msg)
            print(f"  WARNING: {msg}")

        ctx_data = build_ctx_shard(
            pos_shard, corpus, shard_golds, ctx_len, row_offset, fail_on_err
        )
        out_path = os.path.join(out_dir, f"shard_{si:05d}.pt")
        torch.save({k: torch.from_numpy(v) for k, v in ctx_data.items()}, out_path)

        if si % 5 == 0 or si == n_shards - 1:
            cov = attention_mask_coverage(ctx_data["attention_mask"])
            print(f"  [{split}] shard {si:05d}/{n_shards-1}  N={N}  "
                  f"ctx_fill={cov:.3f}")

        row_offset += N

    return {
        "split":            split,
        "n_shards":         n_shards,
        "total_rows":       total_rows,
        "ctx_len":          ctx_len,
        "gold_alignment_ok": alignment_ok,
    }


def attention_mask_coverage(mask: np.ndarray) -> float:
    """Mean fraction of non-padded positions."""
    return float(mask.mean())


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args):
    print("[stage01] Loading tokenizer ...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = int(1e30)

    print("[stage01] Loading region map ...")
    coarse_map = load_coarse_map(args.region_map)
    n_mapped   = int((coarse_map >= 0).sum())
    print(f"  {n_mapped}/50257 tokens mapped to regions")

    # Optional backbone ctx check
    if args.small_ckpt and os.path.isfile(args.small_ckpt):
        try:
            raw     = torch.load(args.small_ckpt, map_location="cpu", weights_only=False)
            max_ctx = raw.get("cfg", {}).get("max_ctx", 1024)
            if args.ctx_len > max_ctx:
                raise ValueError(
                    f"--ctx_len {args.ctx_len} > backbone max_ctx {max_ctx}"
                )
            print(f"  backbone max_ctx={max_ctx}  ctx_len={args.ctx_len}  OK")
        except Exception as exc:
            print(f"  WARNING: backbone check failed: {exc}")

    os.makedirs(args.output_root, exist_ok=True)
    report = {
        "ctx_len": args.ctx_len,
        "region_map": args.region_map,
        "alignment_pass": False,
    }

    # ── Train split ───────────────────────────────────────────────────────────
    print("\n[stage01] Processing train split ...")
    train_tokens = load_wikitext("wikitext-103-raw-v1", tok, "train")
    print(f"  train corpus tokens: {len(train_tokens):,}")
    train_stats = process_split(
        "train", args.train_cand_dir, train_tokens, coarse_map,
        args.ctx_len, os.path.join(args.output_root, "train_ctx"),
        args.fail_on_alignment_error,
    )
    del train_tokens  # free memory before loading val

    # ── Val split ─────────────────────────────────────────────────────────────
    print("\n[stage01] Processing val split ...")
    val_tokens = load_wikitext("wikitext-103-raw-v1", tok, "validation")
    print(f"  val corpus tokens: {len(val_tokens):,}")
    val_stats = process_split(
        "val", args.val_cand_dir, val_tokens, coarse_map,
        args.ctx_len, os.path.join(args.output_root, "val_ctx"),
        args.fail_on_alignment_error,
    )
    del val_tokens

    # ── Report ────────────────────────────────────────────────────────────────
    ok = train_stats["gold_alignment_ok"] and val_stats["gold_alignment_ok"]
    report.update({
        "alignment_pass": ok,
        "train": train_stats,
        "val":   val_stats,
    })
    rpath = os.path.join(args.output_root, "alignment_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[stage01] alignment_pass = {ok}")
    print(f"[stage01] Report → {rpath}")

    if not ok:
        raise RuntimeError("Stage 01 alignment FAILED. See report for details.")
    print("[stage01] PASS")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",             default=None)
    p.add_argument("--train_cand_dir",         required=True)
    p.add_argument("--val_cand_dir",           required=True)
    p.add_argument("--region_map",             required=True)
    p.add_argument("--output_root",            required=True)
    p.add_argument("--ctx_len",                type=int,  default=256)
    p.add_argument("--fail_on_alignment_error",action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
