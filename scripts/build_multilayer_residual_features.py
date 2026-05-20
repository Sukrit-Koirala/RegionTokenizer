#!/usr/bin/env python3
"""
Extracts intermediate backbone hidden states aligned with existing candidate shards.

For each candidate shard_XXXXX.pt, produces a companion
features/shard_XXXXX.pt with the same row count, containing hidden
states captured after each requested transformer block and after ln_f.

Alignment guarantee: gold_token at each row is compared against the
candidate shard's gold_token.  Any mismatch aborts the script.

Shard format (features/shard_XXXXX.pt):
  h_layers   float16  (N, n_layers_saved, d_model)
                       dim 1 maps to layer_ids below
  layer_ids  int64    (n_layers_saved,)
                       block indices 0-indexed; -1 = h_final (after ln_f)
  gold_token int32    (N,)  alignment verification field

Usage:
    # Val (must run build_clean_path_refiner_dataset.py first):
    python scripts/build_multilayer_residual_features.py \\
        --small_ckpt runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --candidate_dir runs/path_refiner_clean/data/val_hgrid_K24 \\
        --output_dir runs/path_refiner_residual_interface/features/val_multilayer \\
        --region_map runs/region_maps_128/token_to_region.json \\
        --split val \\
        --layers auto \\
        --device cuda

    # Train:
    python scripts/build_multilayer_residual_features.py \\
        ... \\
        --candidate_dir runs/path_refiner_clean/data/train_hgrid_K24 \\
        --output_dir runs/path_refiner_residual_interface/features/train_multilayer \\
        --split train
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import (
    TokenChunkDataset,
    load_region_map,
    load_small_backbone_and_probe,
    load_wikitext,
)


# ── Layer selection ───────────────────────────────────────────────────────────

def choose_layers(n_layer: int, mode: str) -> List[int]:
    """
    Returns list of block indices to capture.  -1 = h_final (after ln_f).
    mode "auto" → all blocks + h_final.
    mode "auto_even" → every-other block + h_final.
    mode "N,M,..." → literal block indices, -1 allowed.
    """
    if mode == "auto":
        return list(range(n_layer)) + [-1]
    if mode == "auto_even":
        evens = list(range(0, n_layer, 2))
        if (n_layer - 1) not in evens:
            evens.append(n_layer - 1)
        return evens + [-1]
    ids = [int(x.strip()) for x in mode.split(",")]
    for lid in ids:
        if lid != -1 and not (0 <= lid < n_layer):
            raise ValueError(f"layer index {lid} out of range for {n_layer}-layer model")
    return ids


# ── Multi-layer hidden state extraction ──────────────────────────────────────

@torch.no_grad()
def get_multilayer_hs(
    backbone,
    src: torch.Tensor,      # (B, T) int64 input ids
    layer_ids: List[int],   # block indices; -1 = h_final
    device,
) -> Dict[int, torch.Tensor]:
    """
    Returns dict: layer_id -> (B, T, d_model) float32.
    Runs backbone block by block; captures after each requested block.
    """
    T   = src.shape[1]
    pos = torch.arange(T, device=device).unsqueeze(0)
    x   = backbone.drop(backbone.token_emb(src) + backbone.pos_emb(pos))

    cap: Dict[int, torch.Tensor] = {}
    for i, blk in enumerate(backbone.blocks):
        x = blk(x)
        if i in layer_ids:
            cap[i] = x.float().detach()

    h_final = backbone.ln_f(x).float().detach()
    if -1 in layer_ids:
        cap[-1] = h_final

    return cap


# ── Shard reader helpers ──────────────────────────────────────────────────────

def read_shard_summary(shard_dir: str) -> Tuple[List[str], List[int], List[np.ndarray]]:
    """
    Returns (paths, row_counts, gold_token_arrays) for all shards in order.
    """
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt files in {shard_dir}")
    sizes: List[int] = []
    golds: List[np.ndarray] = []
    for p in paths:
        s = torch.load(p, map_location="cpu", weights_only=True)
        sizes.append(int(s["gold_token"].shape[0]))
        golds.append(s["gold_token"].numpy().astype(np.int32))
    return paths, sizes, golds


# ── Core builder ─────────────────────────────────────────────────────────────

def build_features(
    args,
    backbone,
    coarse_map: np.ndarray,
    layer_ids: List[int],
    d_model: int,
    device,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[build] Reading candidate shards from {args.candidate_dir} ...")
    shard_paths, shard_sizes, shard_golds = read_shard_summary(args.candidate_dir)
    total_target = sum(shard_sizes)
    print(f"  {len(shard_paths)} shards  total rows = {total_target:,}")

    # Load source WikiText-103
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    print(f"[build] Loading WikiText-103 {args.split} ...")
    split_name = "validation" if args.split == "val" else "train"
    tokens  = load_wikitext(args.dataset, tokenizer, split_name)
    seq_len = min(args.seq_len, backbone.pos_emb.num_embeddings)
    ds      = TokenChunkDataset(tokens, seq_len)
    loader  = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, drop_last=False)
    print(f"  seq_len={seq_len}  batch_size={args.batch_size}  chunks={len(ds)}")

    # Save config
    cfg = {
        "layer_ids":    layer_ids,
        "n_layers_saved": len(layer_ids),
        "d_model":      d_model,
        "split":        args.split,
        "seq_len":      seq_len,
        "batch_size":   args.batch_size,
        "candidate_dir": args.candidate_dir,
        "total_rows":   total_target,
        "n_shards":     len(shard_paths),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    n_layers  = len(layer_ids)
    layer_arr = np.array(layer_ids, dtype=np.int64)

    # Accumulation buffers for current shard
    feat_buf:  List[np.ndarray] = []
    gold_buf:  List[np.ndarray] = []
    buf_n     = 0

    shard_idx = 0
    total_pos = 0
    t0        = time.time()

    for bi, batch in enumerate(loader):
        if total_pos >= total_target:
            break

        batch  = batch.to(device)
        src    = batch[:, :-1]
        tgt    = batch[:, 1:]
        if src.size(1) > seq_len:
            src = src[:, :seq_len]
            tgt = tgt[:, :seq_len]

        gold_regions = torch.from_numpy(
            coarse_map[tgt.cpu().reshape(-1).numpy()].astype(np.int64)
        )
        valid_mask = (gold_regions >= 0).numpy()
        n_valid    = int(valid_mask.sum())
        if n_valid == 0:
            continue

        # Trim if we'd overshoot total_target
        remaining = total_target - total_pos
        if n_valid > remaining:
            cumsum = np.cumsum(valid_mask)
            cutoff = int(np.searchsorted(cumsum, remaining + 1))
            valid_mask[cutoff:] = False
            n_valid = int(valid_mask.sum())

        # Extract multi-layer hidden states
        with torch.no_grad():
            caps = get_multilayer_hs(backbone, src, layer_ids, device)

        # Flatten (B, T, D) → (B*T, D) and filter to valid positions
        BT = src.shape[0] * src.shape[1]
        gold_tok_flat = tgt.cpu().reshape(-1).numpy()[valid_mask].astype(np.int32)

        # Stack layer captures: (n_valid, n_layers, d_model)
        layer_feats = []
        for lid in layer_ids:
            h_flat = caps[lid].reshape(BT, d_model)  # (B*T, D)
            h_sel  = h_flat[valid_mask]               # (n_valid, D)
            layer_feats.append(h_sel.cpu().numpy().astype(np.float16))
        feat_chunk = np.stack(layer_feats, axis=1)   # (n_valid, n_layers, D)

        feat_buf.append(feat_chunk)
        gold_buf.append(gold_tok_flat)
        buf_n     += n_valid
        total_pos += n_valid

        # Flush complete shards
        while shard_idx < len(shard_sizes) and buf_n >= shard_sizes[shard_idx]:
            target_n = shard_sizes[shard_idx]

            # Materialise exactly target_n rows from buffer
            all_feat = np.concatenate(feat_buf, axis=0)   # (buf_n, n_layers, D)
            all_gold = np.concatenate(gold_buf, axis=0)   # (buf_n,)

            out_feat = all_feat[:target_n]
            out_gold = all_gold[:target_n]

            # Alignment verification
            ref_gold = shard_golds[shard_idx]
            if not np.array_equal(out_gold, ref_gold):
                mismatch = np.where(out_gold != ref_gold)[0]
                raise RuntimeError(
                    f"Alignment FAIL at shard {shard_idx:05d}: "
                    f"gold_token mismatch at {len(mismatch)} rows "
                    f"(first: row {mismatch[0]}, got {out_gold[mismatch[0]]}, "
                    f"expected {ref_gold[mismatch[0]]}). "
                    "Check --seq_len, --batch_size, and --region_map."
                )

            out_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
            torch.save({
                "h_layers":  torch.from_numpy(out_feat),
                "layer_ids": torch.from_numpy(layer_arr),
                "gold_token": torch.from_numpy(out_gold),
            }, out_path)

            elapsed = time.time() - t0
            print(f"  shard {shard_idx:05d}  n={target_n}  "
                  f"total={total_pos:,}/{total_target:,}  {elapsed:.0f}s  "
                  f"[alignment OK]")

            # Roll buffer
            feat_buf = [all_feat[target_n:]] if all_feat.shape[0] > target_n else []
            gold_buf = [all_gold[target_n:]] if all_gold.shape[0] > target_n else []
            buf_n   -= target_n
            shard_idx += 1

        if bi % 50 == 0:
            print(f"  batch {bi}  total_pos={total_pos:,}  "
                  f"shard_idx={shard_idx}/{len(shard_sizes)}  {time.time()-t0:.0f}s")

    # Flush any remaining partial shard (should only happen if shards were smaller than expected)
    if buf_n > 0 and shard_idx < len(shard_sizes):
        target_n = shard_sizes[shard_idx]
        all_feat  = np.concatenate(feat_buf, axis=0)
        all_gold  = np.concatenate(gold_buf, axis=0)
        if len(all_gold) != target_n:
            raise RuntimeError(
                f"Remaining {len(all_gold)} rows != expected {target_n} "
                f"for shard {shard_idx:05d}."
            )
        ref_gold = shard_golds[shard_idx]
        if not np.array_equal(all_gold, ref_gold):
            raise RuntimeError(f"Alignment FAIL at final shard {shard_idx:05d}.")
        out_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
        torch.save({
            "h_layers":  torch.from_numpy(all_feat.astype(np.float16)),
            "layer_ids": torch.from_numpy(layer_arr),
            "gold_token": torch.from_numpy(all_gold),
        }, out_path)
        print(f"  shard {shard_idx:05d}  n={target_n}  [final]  [alignment OK]")
        shard_idx += 1

    n_saved = len(glob.glob(os.path.join(args.output_dir, "shard_*.pt")))
    print(f"\n[build] Done.  {n_saved} feature shards saved to {args.output_dir}")
    if n_saved != len(shard_paths):
        print(f"  WARNING: saved {n_saved} but expected {len(shard_paths)}. "
              "Check if the source dataset has enough positions.")
    else:
        print(f"  All {n_saved} shards written and alignment-verified.")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device(args.device)
    print(f"[main] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, _ = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    n_layer = len(backbone.blocks)
    print(f"  d_model={d_model}  n_layer={n_layer}")

    layer_ids = choose_layers(n_layer, args.layers)
    print(f"  layer_ids={layer_ids}  ({len(layer_ids)} states per position)")

    # Load coarse region map
    vocab_size = backbone.token_emb.num_embeddings
    print(f"[main] Loading region map: {args.region_map}")
    region_map_tensor, _ = load_region_map(args.region_map, vocab_size)
    coarse_map = region_map_tensor.numpy().astype(np.int32)
    n_covered = int((coarse_map >= 0).sum())
    print(f"  {n_covered:,}/{vocab_size:,} tokens have a region assignment")

    build_features(args, backbone, coarse_map, layer_ids, d_model, device)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",    required=True)
    p.add_argument("--candidate_dir", required=True,
                   help="Directory with candidate shards to align against.")
    p.add_argument("--output_dir",    required=True,
                   help="Directory to write feature shards.")
    p.add_argument("--region_map",    required=True,
                   help="token_to_region.json (same used for candidate shards).")
    p.add_argument("--split",         default="val", choices=["val", "train"],
                   help="WikiText-103 split to process.")
    p.add_argument("--dataset",       default="wikitext-103-raw-v1",
                   help="HuggingFace dataset name (default: wikitext-103-raw-v1).")
    p.add_argument("--layers",        default="auto",
                   help="Layer selection: auto | auto_even | comma-separated indices "
                        "(e.g. '0,2,4,-1'). -1 = h_final (after ln_f).")
    p.add_argument("--seq_len",       type=int, default=128,
                   help="Must match seq_len used in original shard build (default 128).")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
