#!/usr/bin/env python3
"""
Phase 2 — Offline path-refiner dataset builder.

Iterates the validation set, runs backbone.forward to extract h_prime
(post-region hidden state), applies a candidate selection policy, and
writes padded shard files for training the path-conditioned refiner.

Shard format (NPZ per shard, shard_size positions):
  h_prime    float16  (N, d_model)    — post-region hidden state (frozen backbone)
  cand_tok   int32    (N, C_max)      — candidate token IDs (-1=pad)
  cand_fine  int16    (N, C_max)      — fine region ID per candidate (-1=pad)
  cand_super int16    (N, C_max)      — super-region ID per candidate (-1=pad)
  cand_mask  bool     (N, C_max)      — True where valid
  gold_idx   int32    (N,)            — index into cand_tok of gold token (-1=uncovered)
  covered    bool     (N,)            — True if gold token in candidate set
  split      uint8    (N,)            — 0=core,1=medium,2=boundary,3=tight_boundary
  type_arr   uint8    (N,)            — 0=other,1=A,2=B,3=C
  r_margin   float16  (N,)
  m_margin   float16  (N,)

C_max is set conservatively (default 4096) to cover the largest policy.

Policies:
  router_topK  — router top-K regions
  union_rKm    — router top-Kr ∪ mem top-Km   (e.g. union_r12m4)

Usage:
    python scripts/build_path_refiner_dataset.py \\
        --small_ckpt   runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --knn_run_dir  runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20 \\
        --region_map   runs/region_maps_128/token_to_region.json \\
        --output_dir   runs/path_refiner/dataset \\
        --policy       router_top16 \\
        --device       cuda

    # With super-region mapping for variants B/C:
    python scripts/build_path_refiner_dataset.py ... \\
        --super_map  runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --n_super    24
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import (
    TokenChunkDataset, load_region_map, build_inverse_map,
    load_wikitext, load_small_backbone_and_probe, get_hs_small,
)

C_MAX_DEFAULT = 4096


# ── Policy → selected regions ─────────────────────────────────────────────────

def _parse_policy(policy: str) -> Dict:
    """Parse policy string into config dict."""
    p = policy.strip()
    if p.startswith("router_top"):
        K = int(p[len("router_top"):])
        return {"type": "router", "K": K}
    if p.startswith("union_r"):
        rest = p[len("union_r"):]
        parts = rest.split("m")
        Kr, Km = int(parts[0]), int(parts[1])
        return {"type": "union", "Kr": Kr, "Km": Km}
    raise ValueError(
        f"Unknown policy {p!r}. Use 'router_topK' or 'union_rKmM'."
    )


def select_regions(pp: Dict[str, np.ndarray], start: int, n: int,
                   cfg: Dict) -> np.ndarray:
    """Return (n, K_sel) int16 of fine-region IDs."""
    if cfg["type"] == "router":
        return pp["router_topk_regions"][start:start + n, :cfg["K"]]
    r = pp["router_topk_regions"][start:start + n, :cfg["Kr"]]
    m = pp["mem_topk_regions"][start:start + n, :cfg["Km"]]
    return np.concatenate([r, m], axis=1)


# ── Candidate expansion ───────────────────────────────────────────────────────

def expand_candidates(
    sel_regs:   np.ndarray,           # (N, K_sel) int16, -1=pad
    inv_map:    Dict[int, List[int]], # fine_region → list of token IDs
    r2s:        Optional[np.ndarray], # (n_fine,) int — fine→super (None if not provided)
    gold_toks:  np.ndarray,           # (N,) int32 — gold token IDs
    C_max:      int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      cand_tok   (N, C_max) int32
      cand_fine  (N, C_max) int16
      cand_super (N, C_max) int16
      cand_mask  (N, C_max) bool
      gold_idx   (N,) int32   — index into cand_tok; -1 if uncovered
      covered    (N,) bool
    """
    N          = sel_regs.shape[0]
    cand_tok   = np.full((N, C_max), -1, dtype=np.int32)
    cand_fine  = np.full((N, C_max), -1, dtype=np.int16)
    cand_super = np.full((N, C_max), -1, dtype=np.int16)
    cand_mask  = np.zeros((N, C_max), dtype=bool)
    gold_idx   = np.full(N, -1, dtype=np.int32)
    covered    = np.zeros(N, dtype=bool)

    for i in range(N):
        toks:  List[int] = []
        fines: List[int] = []
        sups:  List[int] = []
        seen:  set       = set()
        for k in range(sel_regs.shape[1]):
            r = int(sel_regs[i, k])
            if r < 0 or r not in inv_map:
                continue
            sup = int(r2s[r]) if r2s is not None else -1
            for t in inv_map[r]:
                if t not in seen:
                    toks.append(t)
                    fines.append(r)
                    sups.append(sup)
                    seen.add(t)

        n_c = min(len(toks), C_max)
        cand_tok[i,  :n_c] = toks[:n_c]
        cand_fine[i, :n_c] = fines[:n_c]
        cand_super[i,:n_c] = sups[:n_c]
        cand_mask[i, :n_c] = True

        gt = int(gold_toks[i])
        for ci in range(n_c):
            if cand_tok[i, ci] == gt:
                gold_idx[i] = ci
                covered[i]  = True
                break

    return cand_tok, cand_fine, cand_super, cand_mask, gold_idx, covered


# ── Shard writer ──────────────────────────────────────────────────────────────

def write_shard(path: str, shard: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **shard)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_build(args):
    device   = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    pol_cfg = _parse_policy(args.policy)
    C_max   = args.c_max

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[build] loading checkpoint: {args.small_ckpt}")
    backbone, probe, d_model, cfg_dict, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # ── Load region map ───────────────────────────────────────────────────────
    print(f"[build] loading region map: {args.region_map}")
    coarse_map, n_coarse = load_region_map(args.region_map, vocab_size)
    inv_map = build_inverse_map(coarse_map, n_coarse)

    # ── Load super-region map (optional) ──────────────────────────────────────
    r2s: Optional[np.ndarray] = None
    n_super = 0
    if args.super_map:
        print(f"[build] loading super map: {args.super_map}")
        with open(args.super_map) as f:
            r2s_dict = json.load(f)
        # keys are fine-region IDs (str), values are super IDs
        r2s = np.full(n_coarse, -1, dtype=np.int32)
        for k, v in r2s_dict.items():
            fid = int(k)
            if 0 <= fid < n_coarse:
                r2s[fid] = int(v)
        n_super = args.n_super
        print(f"[build] n_super={n_super}")

    # ── Load per_position.npz ─────────────────────────────────────────────────
    for fname in ("per_position.npz", "per_position_topk.npz"):
        pp_path = os.path.join(args.knn_run_dir, fname)
        if os.path.isfile(pp_path):
            break
    print(f"[build] loading per_position: {pp_path}")
    pp   = dict(np.load(pp_path))
    N_pp = len(pp["gold_region"])
    print(f"[build] per_position N={N_pp:,}")

    has_type = "type" in pp

    # ── Load val data ─────────────────────────────────────────────────────────
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    val_tokens = load_wikitext(args.dataset, tokenizer, "validation")
    safe_seq   = min(args.seq_len, backbone.pos_emb.num_embeddings)
    val_ds     = TokenChunkDataset(val_tokens, safe_seq)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, drop_last=False)
    max_batches = math.ceil(args.max_positions / (args.batch_size * safe_seq))
    print(f"[build] val batches: {len(val_loader)} (capped at {max_batches})")

    # ── Accumulate and write shards ───────────────────────────────────────────
    pp_offset    = 0
    shard_idx    = 0
    total_pos    = 0
    total_cov    = 0

    # Buffers for current shard
    hp_buf:    List[np.ndarray] = []
    ctok_buf:  List[np.ndarray] = []
    cfine_buf: List[np.ndarray] = []
    csup_buf:  List[np.ndarray] = []
    cmask_buf: List[np.ndarray] = []
    gidx_buf:  List[np.ndarray] = []
    cov_buf:   List[np.ndarray] = []
    split_buf: List[np.ndarray] = []
    type_buf:  List[np.ndarray] = []
    rmg_buf:   List[np.ndarray] = []
    mmg_buf:   List[np.ndarray] = []
    shard_n    = 0

    def _flush_shard():
        nonlocal shard_idx
        shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.npz")
        shard = {
            "h_prime":    np.concatenate(hp_buf,    axis=0),
            "cand_tok":   np.concatenate(ctok_buf,  axis=0),
            "cand_fine":  np.concatenate(cfine_buf, axis=0),
            "cand_super": np.concatenate(csup_buf,  axis=0),
            "cand_mask":  np.concatenate(cmask_buf, axis=0),
            "gold_idx":   np.concatenate(gidx_buf,  axis=0),
            "covered":    np.concatenate(cov_buf,   axis=0),
            "split":      np.concatenate(split_buf, axis=0),
            "type_arr":   np.concatenate(type_buf,  axis=0),
            "r_margin":   np.concatenate(rmg_buf,   axis=0),
            "m_margin":   np.concatenate(mmg_buf,   axis=0),
        }
        write_shard(shard_path, shard)
        n_written = shard["covered"].shape[0]
        cov_frac  = shard["covered"].mean()
        print(f"[build] shard {shard_idx:05d}  n={n_written}  cov={cov_frac:.4f}  → {shard_path}")
        shard_idx += 1
        hp_buf.clear(); ctok_buf.clear(); cfine_buf.clear()
        csup_buf.clear(); cmask_buf.clear(); gidx_buf.clear()
        cov_buf.clear(); split_buf.clear(); type_buf.clear()
        rmg_buf.clear(); mmg_buf.clear()

    t0 = time.time()
    for bi, batch in enumerate(val_loader):
        if bi >= max_batches or pp_offset >= N_pp:
            break
        batch = batch.to(device)
        src   = batch[:, :-1]
        tgt   = batch[:, 1:]
        if src.size(1) > safe_seq:
            src = src[:, :safe_seq]; tgt = tgt[:, :safe_seq]

        gold_regions = coarse_map[tgt.cpu()].reshape(-1)
        valid_np     = (gold_regions >= 0).numpy()
        n_valid      = int(valid_np.sum())
        if n_valid == 0:
            continue

        if pp_offset + n_valid > N_pp:
            need   = N_pp - pp_offset
            cumsum = np.cumsum(valid_np)
            cutoff = np.searchsorted(cumsum, need + 1)
            valid_np[cutoff:] = False
            n_valid = int(valid_np.sum())

        valid_t = torch.from_numpy(valid_np).to(device)

        # Backbone forward to get h_prime
        with torch.no_grad():
            if hasattr(backbone, "retrieval_proj"):
                _, _, _, h_prime_bt = backbone(src)         # (B, T, d_model)
            else:
                h_prime_bt = get_hs_small(backbone, src, device)  # (B, T, d_model)
            h_prime_flat = h_prime_bt.reshape(-1, d_model)  # (B*T, d_model)
            h_valid      = h_prime_flat[valid_t]             # (n_valid, d_model)

        gold_tok_flat = tgt.cpu().reshape(-1).numpy()[valid_np].astype(np.int32)  # (n_valid,)

        # Policy → candidate tokens
        sel_regs = select_regions(pp, pp_offset, n_valid, pol_cfg)  # (n_valid, K_sel)
        cand_tok, cand_fine, cand_super, cand_mask, gold_idx, covered = expand_candidates(
            sel_regs, inv_map, r2s, gold_tok_flat, C_max
        )

        # Metadata from per_position.npz
        split_chunk  = pp["split"][pp_offset:pp_offset + n_valid]
        type_chunk   = pp["type"][pp_offset:pp_offset + n_valid] if has_type else \
                       np.zeros(n_valid, dtype=np.uint8)
        rmg_chunk    = pp["router_margin"][pp_offset:pp_offset + n_valid].astype(np.float16) \
                       if "router_margin" in pp else np.zeros(n_valid, dtype=np.float16)
        mmg_chunk    = pp["mem_margin"][pp_offset:pp_offset + n_valid].astype(np.float16) \
                       if "mem_margin" in pp else np.zeros(n_valid, dtype=np.float16)

        hp_buf.append(h_valid.cpu().float().numpy().astype(np.float16))
        ctok_buf.append(cand_tok)
        cfine_buf.append(cand_fine)
        csup_buf.append(cand_super)
        cmask_buf.append(cand_mask)
        gidx_buf.append(gold_idx)
        cov_buf.append(covered)
        split_buf.append(split_chunk)
        type_buf.append(type_chunk)
        rmg_buf.append(rmg_chunk)
        mmg_buf.append(mmg_chunk)
        shard_n   += n_valid
        total_pos += n_valid
        total_cov += int(covered.sum())
        pp_offset += n_valid

        if shard_n >= args.shard_size:
            _flush_shard()
            shard_n = 0

        if bi % 20 == 0:
            elapsed = time.time() - t0
            print(f"  batch {bi}/{min(max_batches, len(val_loader))}  "
                  f"total_pos={total_pos:,}  "
                  f"cov={total_cov/max(total_pos,1):.4f}  "
                  f"t={elapsed:.0f}s")

    # Flush remaining
    if shard_n > 0:
        _flush_shard()

    elapsed = time.time() - t0
    print(f"\n[build] done.  total_pos={total_pos:,}  "
          f"coverage={total_cov/max(total_pos,1):.4f}  "
          f"shards={shard_idx}  time={elapsed:.0f}s")
    print(f"[build] dataset dir: {args.output_dir}")

    # Write metadata
    meta = {
        "policy":     args.policy,
        "n_total":    total_pos,
        "n_covered":  total_cov,
        "coverage":   total_cov / max(total_pos, 1),
        "c_max":      C_max,
        "n_fine":     n_coarse,
        "n_super":    n_super,
        "d_model":    d_model,
        "shards":     shard_idx,
    }
    import json as _json
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w") as f:
        _json.dump(meta, f, indent=2)
    print(f"[build] wrote dataset_meta.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",    required=True)
    p.add_argument("--knn_run_dir",   required=True)
    p.add_argument("--region_map",    required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--policy",        default="router_top16",
                   help="e.g. 'router_top16' or 'union_r12m4'")
    p.add_argument("--super_map",     default=None,
                   help="region_to_superregion_K{K}.json (optional)")
    p.add_argument("--n_super",       type=int, default=0)
    p.add_argument("--dataset",       default="wikitext-103-raw-v1")
    p.add_argument("--seq_len",       type=int, default=128)
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--max_positions", type=int, default=500_000)
    p.add_argument("--shard_size",    type=int, default=10_000)
    p.add_argument("--c_max",         type=int, default=C_MAX_DEFAULT)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run_build(_parse())
