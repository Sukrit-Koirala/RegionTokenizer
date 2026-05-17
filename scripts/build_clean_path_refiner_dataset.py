#!/usr/bin/env python3
"""
Phase 1 — Clean path-refiner dataset builder.

Builds SEPARATE train and val shard datasets so there is no train/val leakage.

  train: WikiText-103 train split  — runs kNN retrieval on-the-fly
  val:   WikiText-103 val split    — uses per_position.npz (already computed)

Both use the same candidate policy so shards are directly comparable.

Shard format (.pt, torch.save):
  h_prime          float16 (N, d_model)
  gold_token       int32   (N,)
  gold_region      int16   (N,)
  cand_tok         int32   (N, C_max)    candidate token ids, -1=pad
  cand_fine        int16   (N, C_max)    fine region per candidate, -1=pad
  gold_cand_idx    int32   (N,)          index in cand_tok of gold token, -1=uncovered
  covered          bool    (N,)
  router_topk_reg  int16   (N, K)        top-K router regions
  router_topk_prb  float16 (N, K)
  mem_topk_reg     int16   (N, K)        top-K mem regions
  mem_topk_prb     float16 (N, K)
  router_margin    float16 (N,)
  router_entropy   float16 (N,)
  mem_margin       float16 (N,)
  mem_entropy      float16 (N,)
  split            uint8   (N,)          0=core,1=medium,2=boundary,3=tight
  type_arr         uint8   (N,)          0=other,1=A,2=B,3=C

Policies supported:
  router_top8, router_top12, router_top16, router_top24
  union_r16m4, union_r12m4
  hgrid_K24_srccombined_nts1_bkr12_fb24_cm0.15_en1.25_corec60m30

Usage:
    # Val split (fast — reuses per_position.npz for region data):
    python scripts/build_clean_path_refiner_dataset.py \\
        --split       val \\
        --small_ckpt  runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --knn_run_dir runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20 \\
        --region_map  runs/region_maps_128/token_to_region.json \\
        --super_map   runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --output_dir  runs/path_refiner_clean/data/val_hgrid_K24 \\
        --policy      hgrid_K24 \\
        --device      cuda

    # Train split (kNN on-the-fly):
    python scripts/build_clean_path_refiner_dataset.py \\
        --split         train \\
        --max_positions 1000000 \\
        ...same args... \\
        --output_dir  runs/path_refiner_clean/data/train_hgrid_K24
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
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
    MemoryIndex,
)

N_FINE     = 128
FULL_VOCAB = 50257

# ── Superregion helpers ───────────────────────────────────────────────────────

def load_r2s(super_map_path: str, n_fine: int) -> np.ndarray:
    """Returns int32 array (n_fine,): fine_region → super_region_id."""
    with open(super_map_path) as f:
        d = json.load(f)
    r2s = np.full(n_fine, -1, dtype=np.int32)
    for k, v in d.items():
        fid = int(k)
        if 0 <= fid < n_fine:
            r2s[fid] = int(v)
    return r2s


def build_super_children(r2s: np.ndarray, n_super: int) -> Dict[int, List[int]]:
    """Inverse of r2s: super_id → list of fine region ids."""
    inv: Dict[int, List[int]] = defaultdict(list)
    for fine, sup in enumerate(r2s):
        if sup >= 0:
            inv[int(sup)].append(fine)
    return dict(inv)


def aggregate_to_super(topk_reg: np.ndarray, topk_prb: np.ndarray,
                        r2s: np.ndarray, n_super: int) -> np.ndarray:
    """(N, K) int16, (N, K) float16 → (N, n_super) float32."""
    N, K   = topk_reg.shape
    sp     = np.zeros((N, n_super), dtype=np.float32)
    valid  = topk_reg >= 0
    fine   = topk_reg.astype(np.int32).clip(min=0)
    sups   = r2s[fine]  # (N, K)
    sups[~valid] = -1
    valid2 = sups >= 0
    rows   = np.repeat(np.arange(N), K)[valid2.ravel()]
    cols   = sups.ravel()[valid2.ravel()]
    vals   = topk_prb.astype(np.float32).ravel()[valid2.ravel()]
    np.add.at(sp, (rows, cols), vals)
    return sp


# ── Candidate policies ────────────────────────────────────────────────────────

HGRID_MID_K_LOOKUP = {(60, 30): 8, (50, 25): 6, (40, 20): 6, (30, 10): 8}


def parse_policy(policy: str) -> Dict:
    """Parse policy name → config dict."""
    p = policy.strip()
    if p.startswith("router_top"):
        return {"type": "router", "K": int(p[len("router_top"):])}
    if p.startswith("union_r"):
        rest = p[len("union_r"):]
        Kr, Km = int(rest.split("m")[0]), int(rest.split("m")[1])
        return {"type": "union", "Kr": Kr, "Km": Km}
    if p.startswith("hgrid_K"):
        # e.g. hgrid_K24_srccombined_nts1_bkr12_fb24_cm0.15_en1.25_corec60m30
        import re
        cfg: Dict = {"type": "hgrid"}
        m = re.search(r"hgrid_K(\d+)", p);        cfg["K_super"]  = int(m.group(1)) if m else 24
        m = re.search(r"src(\w+?)_",   p);        cfg["src"]      = m.group(1) if m else "combined"
        m = re.search(r"nts(\d+)",     p);        cfg["n_top_super"] = int(m.group(1)) if m else 1
        m = re.search(r"bkr(\d+)",     p);        cfg["base_kr"]  = int(m.group(1)) if m else 12
        m = re.search(r"_fb(\d+)",     p);        cfg["fallback_k"] = int(m.group(1)) if m else 24
        m = re.search(r"cm([\d.]+)",   p);        cfg["cm_thr"]   = float(m.group(1)) if m else 0.15
        m = re.search(r"en([\d.]+)",   p);        cfg["en_thr"]   = float(m.group(1)) if m else 1.25
        m = re.search(r"corec(\d+)m(\d+)", p)
        cfg["core_high"] = int(m.group(1)) / 100 if m else 0.60
        cfg["core_mid"]  = int(m.group(2)) / 100 if m else 0.30
        key = (int(cfg["core_high"] * 100), int(cfg["core_mid"] * 100))
        cfg["mid_k"] = HGRID_MID_K_LOOKUP.get(key, 8)
        return cfg
    raise ValueError(f"Unknown policy: {policy!r}")


def select_regions_hgrid(
    router_topk_reg: np.ndarray,   # (N, K) int16
    router_topk_prb: np.ndarray,   # (N, K) float16
    mem_topk_reg:    np.ndarray,   # (N, K) float16
    mem_topk_prb:    np.ndarray,
    router_margin:   np.ndarray,   # (N,) float16
    mem_entropy:     np.ndarray,   # (N,) float16
    r2s:             np.ndarray,   # (n_fine,) int32
    super_children:  Dict[int, List[int]],
    cfg:             Dict,
    K_stored:        int,
) -> np.ndarray:
    """Returns (N, K_out) int16, -1=padding. K_out is generous upper bound."""
    N         = len(router_margin)
    base_kr   = cfg["base_kr"]
    mid_k     = cfg["mid_k"]
    fb_k      = cfg["fallback_k"]
    cm_thr    = cfg["cm_thr"]
    en_thr    = cfg["en_thr"]
    n_ts      = cfg["n_top_super"]
    core_high = cfg["core_high"]
    core_mid  = cfg["core_mid"]
    src       = cfg["src"]
    n_super   = cfg["K_super"]

    sp_router = aggregate_to_super(router_topk_reg, router_topk_prb, r2s, n_super)
    sp_mem    = aggregate_to_super(mem_topk_reg,    mem_topk_prb,    r2s, n_super)
    sp_comb   = 0.5 * sp_router + 0.5 * sp_mem
    sp_ref    = sp_comb if src == "combined" else sp_mem

    # Coarse margin / entropy
    top2      = np.partition(-sp_ref, kth=min(1, n_super - 1), axis=1)[:, :2] * -1
    cm_mg     = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    cm_en     = -(sp_ref * np.log(sp_ref + 1e-10)).sum(1)
    top_sups  = np.argsort(-sp_ref, axis=1)[:, :n_ts]  # (N, n_ts)

    rmg = router_margin.astype(np.float32)

    K_out = max(base_kr, fb_k) + N_FINE  # generous
    sel   = np.full((N, K_out), -1, dtype=np.int16)

    for i in range(N):
        chosen: set = set()
        if rmg[i] >= core_high:
            for k in range(min(base_kr, K_stored)):
                r = int(router_topk_reg[i, k])
                if r >= 0: chosen.add(r)
        elif rmg[i] >= core_mid:
            for k in range(min(mid_k, K_stored)):
                r = int(router_topk_reg[i, k])
                if r >= 0: chosen.add(r)
        else:
            if cm_mg[i] >= cm_thr and cm_en[i] <= en_thr:
                for k in range(min(base_kr, K_stored)):
                    r = int(router_topk_reg[i, k])
                    if r >= 0: chosen.add(r)
                for s_idx in range(n_ts):
                    sup = int(top_sups[i, s_idx])
                    for fine in super_children.get(sup, []):
                        chosen.add(fine)
            else:
                for k in range(min(fb_k, K_stored)):
                    r = int(router_topk_reg[i, k])
                    if r >= 0: chosen.add(r)

        ch = list(chosen)[:K_out]
        sel[i, :len(ch)] = ch

    return sel


def select_regions(r_topk_reg: np.ndarray, r_topk_prb: np.ndarray,
                   m_topk_reg: np.ndarray, m_topk_prb: np.ndarray,
                   r_margin: np.ndarray, m_entropy: np.ndarray,
                   cfg: Dict, r2s: Optional[np.ndarray],
                   super_children: Optional[Dict]) -> np.ndarray:
    K_stored = r_topk_reg.shape[1]
    if cfg["type"] == "router":
        K = min(cfg["K"], K_stored)
        return r_topk_reg[:, :K]
    if cfg["type"] == "union":
        Kr = min(cfg["Kr"], K_stored)
        Km = min(cfg["Km"], K_stored)
        return np.concatenate([r_topk_reg[:, :Kr], m_topk_reg[:, :Km]], axis=1)
    if cfg["type"] == "hgrid":
        return select_regions_hgrid(
            r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
            r_margin, m_entropy, r2s, super_children, cfg, K_stored,
        )
    raise ValueError(f"Unknown policy type: {cfg['type']}")


# ── Candidate token expansion ─────────────────────────────────────────────────

def expand_to_candidates(
    sel_fine:   np.ndarray,           # (N, K_sel) int16
    inv_map:    Dict[int, List[int]],
    gold_toks:  np.ndarray,           # (N,) int32
    C_max:      int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      cand_tok    (N, C_max) int32
      cand_fine   (N, C_max) int16
      gold_idx    (N,) int32
      covered     (N,) bool
    """
    N           = sel_fine.shape[0]
    cand_tok    = np.full((N, C_max), -1, dtype=np.int32)
    cand_fine_a = np.full((N, C_max), -1, dtype=np.int16)
    gold_idx    = np.full(N, -1, dtype=np.int32)
    covered     = np.zeros(N, dtype=bool)

    for i in range(N):
        toks:  List[int] = []
        fines: List[int] = []
        seen:  set       = set()
        for k in range(sel_fine.shape[1]):
            r = int(sel_fine[i, k])
            if r < 0 or r not in inv_map:
                continue
            for t in inv_map[r]:
                if t not in seen:
                    toks.append(t)
                    fines.append(r)
                    seen.add(t)

        n_c = min(len(toks), C_max)
        cand_tok[i,    :n_c] = toks[:n_c]
        cand_fine_a[i, :n_c] = fines[:n_c]

        gt = int(gold_toks[i])
        for ci in range(n_c):
            if cand_tok[i, ci] == gt:
                gold_idx[i] = ci
                covered[i]  = True
                break

    return cand_tok, cand_fine_a, gold_idx, covered


# ── Shard I/O ─────────────────────────────────────────────────────────────────

def save_shard(path: str, arrays: Dict[str, np.ndarray]) -> None:
    torch.save({k: torch.from_numpy(v) for k, v in arrays.items()}, path)


# ── Val split builder ─────────────────────────────────────────────────────────

def build_val_split(args, backbone, inv_map, coarse_map, d_model, vocab_size,
                    pp: Dict, pol_cfg: Dict,
                    r2s: Optional[np.ndarray], super_children: Optional[Dict]):
    """Build val shards using per_position.npz for region metadata."""
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    val_tokens = load_wikitext(args.dataset, tokenizer, "validation")
    safe_seq   = min(args.seq_len, backbone.pos_emb.num_embeddings)
    val_ds     = TokenChunkDataset(val_tokens, safe_seq)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, drop_last=False)

    device    = next(backbone.parameters()).device
    N_pp      = len(pp["gold_region"])
    pp_offset = 0

    hp_buf, ctok_buf, cfine_buf = [], [], []
    gtok_buf, greg_buf = [], []
    gidx_buf, cov_buf  = [], []
    rtr_buf, rtp_buf   = [], []
    mtr_buf, mtp_buf   = [], []
    rmg_buf, ren_buf   = [], []
    mmg_buf, men_buf   = [], []
    spl_buf, typ_buf   = [], []
    shard_n, shard_idx = 0, 0

    has_type = "type" in pp

    def _flush():
        nonlocal shard_n, shard_idx
        n = len(cov_buf)
        K_stored = rtr_buf[0].shape[1] if rtr_buf else 1
        shard = {
            "h_prime":         np.concatenate(hp_buf),
            "gold_token":      np.array(gtok_buf, dtype=np.int32),
            "gold_region":     np.array(greg_buf, dtype=np.int16),
            "cand_tok":        np.concatenate(ctok_buf),
            "cand_fine":       np.concatenate(cfine_buf),
            "gold_cand_idx":   np.array(gidx_buf, dtype=np.int32),
            "covered":         np.array(cov_buf, dtype=bool),
            "router_topk_reg": np.concatenate(rtr_buf),
            "router_topk_prb": np.concatenate(rtp_buf),
            "mem_topk_reg":    np.concatenate(mtr_buf),
            "mem_topk_prb":    np.concatenate(mtp_buf),
            "router_margin":   np.array(rmg_buf, dtype=np.float16),
            "router_entropy":  np.array(ren_buf, dtype=np.float16),
            "mem_margin":      np.array(mmg_buf, dtype=np.float16),
            "mem_entropy":     np.array(men_buf, dtype=np.float16),
            "split":           np.array(spl_buf, dtype=np.uint8),
            "type_arr":        np.array(typ_buf, dtype=np.uint8),
        }
        path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
        save_shard(path, shard)
        cov_rate = np.array(cov_buf).mean()
        print(f"  [val] shard {shard_idx:05d}  n={n}  cov={cov_rate:.4f}")
        shard_idx += 1
        for buf in [hp_buf, ctok_buf, cfine_buf, gtok_buf, greg_buf,
                    gidx_buf, cov_buf, rtr_buf, rtp_buf, mtr_buf, mtp_buf,
                    rmg_buf, ren_buf, mmg_buf, men_buf, spl_buf, typ_buf]:
            buf.clear()

    t0 = time.time()
    for bi, batch in enumerate(val_loader):
        if pp_offset >= N_pp:
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
            cutoff = int(np.searchsorted(cumsum, need + 1))
            valid_np[cutoff:] = False
            n_valid = int(valid_np.sum())

        valid_t = torch.from_numpy(valid_np).to(device)

        with torch.no_grad():
            if hasattr(backbone, "retrieval_proj"):
                _, _, _, h_prime_bt = backbone(src)
            else:
                h_prime_bt = get_hs_small(backbone, src, device)
            h_prime_flat = h_prime_bt.reshape(-1, d_model)
            h_valid      = h_prime_flat[valid_t]

        gold_tok_flat = tgt.cpu().reshape(-1).numpy()[valid_np].astype(np.int32)

        # Pull metadata from per_position.npz
        sl = slice(pp_offset, pp_offset + n_valid)
        r_topk_reg = pp["router_topk_regions"][sl]   # (n_valid, K)
        r_topk_prb = pp["router_topk_probs"][sl]
        m_topk_reg = pp["mem_topk_regions"][sl]
        m_topk_prb = pp["mem_topk_probs"][sl]
        r_margin   = pp["router_margin"][sl]
        r_entropy  = pp["router_entropy"][sl] if "router_entropy" in pp else \
                     np.zeros(n_valid, dtype=np.float16)
        m_margin   = pp["mem_margin"][sl]
        m_entropy  = pp["mem_entropy"][sl] if "mem_entropy" in pp else \
                     np.zeros(n_valid, dtype=np.float16)
        split_c    = pp["split"][sl]
        type_c     = pp["type"][sl] if has_type else np.zeros(n_valid, dtype=np.uint8)
        gold_reg_c = pp["gold_region"][sl]

        # Candidate selection
        sel = select_regions(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
                             r_margin, m_entropy, pol_cfg, r2s, super_children)
        cand_tok, cand_fine, gold_cand_idx, cov = expand_to_candidates(
            sel, inv_map, gold_tok_flat, args.candidate_cap
        )

        hp_buf.append(h_valid.cpu().float().numpy().astype(np.float16))
        gtok_buf.extend(gold_tok_flat.tolist())
        greg_buf.extend(gold_reg_c.tolist())
        ctok_buf.append(cand_tok)
        cfine_buf.append(cand_fine)
        gidx_buf.extend(gold_cand_idx.tolist())
        cov_buf.extend(cov.tolist())
        rtr_buf.append(r_topk_reg)
        rtp_buf.append(r_topk_prb)
        mtr_buf.append(m_topk_reg)
        mtp_buf.append(m_topk_prb)
        rmg_buf.extend(r_margin.tolist())
        ren_buf.extend(r_entropy.tolist())
        mmg_buf.extend(m_margin.tolist())
        men_buf.extend(m_entropy.tolist())
        spl_buf.extend(split_c.tolist())
        typ_buf.extend(type_c.tolist())
        shard_n   += n_valid
        pp_offset += n_valid

        if shard_n >= args.shard_size:
            _flush(); shard_n = 0
        if bi % 20 == 0:
            print(f"  [val] batch {bi}  pp_offset={pp_offset:,}/{N_pp:,}  "
                  f"cov={np.array(cov_buf).mean() if cov_buf else 0:.4f}"
                  f"  t={time.time()-t0:.0f}s")

    if shard_n > 0:
        _flush()
    return pp_offset


# ── Train split builder ───────────────────────────────────────────────────────

def build_train_split(args, backbone, probe, inv_map, coarse_map, d_model, vocab_size,
                      pol_cfg: Dict, r2s: Optional[np.ndarray],
                      super_children: Optional[Dict], n_coarse: int):
    """Build train shards from WikiText-103 train split with on-the-fly kNN."""
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    train_tokens = load_wikitext(args.dataset, tokenizer, "train")
    safe_seq     = min(args.seq_len, backbone.pos_emb.num_embeddings)
    train_ds     = TokenChunkDataset(train_tokens, safe_seq)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=0, drop_last=False)

    device = next(backbone.parameters()).device

    # Load memory index
    mem_keys_path = os.path.join(args.knn_run_dir, "memory_keys.npy")
    mem_regs_path = os.path.join(args.knn_run_dir, "memory_regions.npy")
    if not os.path.isfile(mem_keys_path):
        raise RuntimeError(f"memory_keys.npy not found in {args.knn_run_dir}\n"
                           "Run offline_region_knn.py first.")
    print("[train] loading memory index ...")
    mem_keys    = np.load(mem_keys_path)          # (M, d) float16
    mem_regions = np.load(mem_regs_path)          # (M,) int32
    mem_index   = MemoryIndex(mem_keys.shape[1], device, normalize=True)
    mem_index.build(mem_keys.astype(np.float32))

    probe_temp = args.probe_temp
    knn_temp   = args.knn_temp
    topk       = args.topk

    hp_buf, ctok_buf, cfine_buf = [], [], []
    gtok_buf, greg_buf = [], []
    gidx_buf, cov_buf  = [], []
    rtr_buf, rtp_buf   = [], []
    mtr_buf, mtp_buf   = [], []
    rmg_buf, ren_buf   = [], []
    mmg_buf, men_buf   = [], []
    spl_buf, typ_buf   = [], []
    shard_n, shard_idx, total_pos = 0, 0, 0

    def _flush():
        nonlocal shard_n, shard_idx
        n = len(cov_buf)
        shard = {
            "h_prime":         np.concatenate(hp_buf),
            "gold_token":      np.array(gtok_buf, dtype=np.int32),
            "gold_region":     np.array(greg_buf, dtype=np.int16),
            "cand_tok":        np.concatenate(ctok_buf),
            "cand_fine":       np.concatenate(cfine_buf),
            "gold_cand_idx":   np.array(gidx_buf, dtype=np.int32),
            "covered":         np.array(cov_buf, dtype=bool),
            "router_topk_reg": np.concatenate(rtr_buf),
            "router_topk_prb": np.concatenate(rtp_buf),
            "mem_topk_reg":    np.concatenate(mtr_buf),
            "mem_topk_prb":    np.concatenate(mtp_buf),
            "router_margin":   np.array(rmg_buf, dtype=np.float16),
            "router_entropy":  np.array(ren_buf, dtype=np.float16),
            "mem_margin":      np.array(mmg_buf, dtype=np.float16),
            "mem_entropy":     np.array(men_buf, dtype=np.float16),
            "split":           np.array(spl_buf, dtype=np.uint8),
            "type_arr":        np.array(typ_buf, dtype=np.uint8),
        }
        path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
        save_shard(path, shard)
        cov_rate = np.array(cov_buf).mean()
        print(f"  [train] shard {shard_idx:05d}  n={n}  cov={cov_rate:.4f}")
        shard_idx += 1
        for buf in [hp_buf, ctok_buf, cfine_buf, gtok_buf, greg_buf,
                    gidx_buf, cov_buf, rtr_buf, rtp_buf, mtr_buf, mtp_buf,
                    rmg_buf, ren_buf, mmg_buf, men_buf, spl_buf, typ_buf]:
            buf.clear()

    t0 = time.time()
    for bi, batch in enumerate(train_loader):
        if total_pos >= args.max_positions:
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

        valid_t = torch.from_numpy(valid_np).to(device)
        gold_tok_np = tgt.cpu().reshape(-1).numpy()[valid_np].astype(np.int32)

        with torch.no_grad():
            if hasattr(backbone, "retrieval_proj"):
                lm_logits_bt, _, _, h_prime_bt = backbone(src)
                h_raw = backbone._last_h           # (B, T, d_model)
            else:
                h_prime_bt = get_hs_small(backbone, src, device)
                h_raw      = h_prime_bt
                lm_logits_bt = backbone.lm_head(h_prime_bt)

            h_prime_flat = h_prime_bt.reshape(-1, d_model)
            h_raw_flat   = h_raw.reshape(-1, d_model)
            h_valid      = h_prime_flat[valid_t]
            h_raw_valid  = h_raw_flat[valid_t]

            # Router probs
            if probe is not None:
                r_logits = probe(h_raw_valid)
                p_router = F.softmax(r_logits.float() / probe_temp, dim=-1).cpu()  # (n_valid, n_coarse)
            else:
                p_router = torch.zeros(n_valid, n_coarse)

            # kNN retrieval key
            if hasattr(backbone, "retrieval_proj"):
                z = backbone.retrieval_proj(h_raw_valid.float())
                z = F.normalize(z, dim=-1)
                key_np = z.cpu().numpy().astype(np.float32)
            else:
                key_np = h_raw_valid.cpu().numpy().astype(np.float32)

        sims, nn_idx = mem_index.search(key_np, topk)          # (n_valid, topk)
        nn_regs      = mem_regions[nn_idx].astype(np.int64)    # (n_valid, topk)
        w            = F.softmax(torch.tensor(sims) / knn_temp, dim=-1)  # (n_valid, topk)
        p_mem        = torch.zeros(n_valid, n_coarse)
        p_mem.scatter_add_(1, torch.from_numpy(nn_regs), w)
        p_mem        = (p_mem / p_mem.sum(-1, keepdim=True).clamp(min=1e-10)).cpu()  # (n_valid, n_coarse)

        # Router topk
        r_topk      = torch.topk(p_router, min(topk, n_coarse), dim=-1)
        r_topk_reg  = r_topk.indices.numpy().astype(np.int16)
        r_topk_prb  = r_topk.values.numpy().astype(np.float16)

        # Mem topk
        m_topk      = torch.topk(p_mem, min(topk, n_coarse), dim=-1)
        m_topk_reg  = m_topk.indices.numpy().astype(np.int16)
        m_topk_prb  = m_topk.values.numpy().astype(np.float16)

        # Margins / entropy
        r_top2      = torch.topk(p_router, min(2, n_coarse), dim=-1).values
        r_margin    = (r_top2[:, 0] - r_top2[:, 1]).numpy().astype(np.float16)
        r_entropy   = (-(p_router * (p_router + 1e-10).log()).sum(-1)).numpy().astype(np.float16)
        m_top2      = torch.topk(p_mem, min(2, n_coarse), dim=-1).values
        m_margin    = (m_top2[:, 0] - m_top2[:, 1]).numpy().astype(np.float16)
        m_entropy   = (-(p_mem * (p_mem + 1e-10).log()).sum(-1)).numpy().astype(np.float16)

        # Split labels (same as per_position.npz convention)
        split_arr = np.zeros(n_valid, dtype=np.uint8)
        rmg_f     = r_margin.astype(np.float32)
        split_arr[rmg_f < 0.30] = 1
        split_arr[rmg_f < 0.10] = 2
        split_arr[rmg_f < 0.03] = 3

        # Candidate selection
        sel = select_regions(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
                             r_margin, m_entropy, pol_cfg, r2s, super_children)
        cand_tok, cand_fine, gold_cand_idx, cov = expand_to_candidates(
            sel, inv_map, gold_tok_np, args.candidate_cap
        )

        gold_reg_np = gold_regions.numpy()[valid_np].astype(np.int16)

        hp_buf.append(h_valid.cpu().float().numpy().astype(np.float16))
        gtok_buf.extend(gold_tok_np.tolist())
        greg_buf.extend(gold_reg_np.tolist())
        ctok_buf.append(cand_tok)
        cfine_buf.append(cand_fine)
        gidx_buf.extend(gold_cand_idx.tolist())
        cov_buf.extend(cov.tolist())
        rtr_buf.append(r_topk_reg)
        rtp_buf.append(r_topk_prb)
        mtr_buf.append(m_topk_reg)
        mtp_buf.append(m_topk_prb)
        rmg_buf.extend(r_margin.tolist())
        ren_buf.extend(r_entropy.tolist())
        mmg_buf.extend(m_margin.tolist())
        men_buf.extend(m_entropy.tolist())
        spl_buf.extend(split_arr.tolist())
        typ_buf.extend([0] * n_valid)   # type A/B/C not available for train split
        shard_n   += n_valid
        total_pos += n_valid

        if shard_n >= args.shard_size:
            _flush(); shard_n = 0
        if bi % 50 == 0:
            pct = total_pos / args.max_positions * 100
            print(f"  [train] batch {bi}  pos={total_pos:,}/{args.max_positions:,} ({pct:.1f}%)"
                  f"  cov={np.array(cov_buf).mean() if cov_buf else 0:.4f}"
                  f"  t={time.time()-t0:.0f}s")

    if shard_n > 0:
        _flush()
    return total_pos


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    pol_cfg = parse_policy(args.policy)
    print(f"[build] policy={args.policy}  cfg={pol_cfg}")

    # Model
    print(f"[build] loading checkpoint: {args.small_ckpt}")
    backbone, probe, d_model, cfg_dict, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Region map
    coarse_map, n_coarse = load_region_map(args.region_map, vocab_size)
    inv_map = build_inverse_map(coarse_map, n_coarse)

    # Super map (optional)
    r2s: Optional[np.ndarray]       = None
    super_children: Optional[Dict]  = None
    if args.super_map and os.path.isfile(args.super_map):
        r2s           = load_r2s(args.super_map, N_FINE)
        n_super       = int(r2s.max()) + 1
        super_children = build_super_children(r2s, n_super)
        print(f"[build] loaded super map  n_super={n_super}")
    elif pol_cfg["type"] == "hgrid":
        raise RuntimeError("HGrid policy requires --super_map.")

    if args.split == "val":
        # Validate per_position.npz exists
        for fname in ("per_position.npz", "per_position_topk.npz"):
            pp_path = os.path.join(args.knn_run_dir, fname)
            if os.path.isfile(pp_path):
                break
        else:
            raise RuntimeError(f"per_position.npz not found in {args.knn_run_dir}")
        print(f"[build] val split  loading {pp_path}")
        pp = dict(np.load(pp_path))
        # Rename keys for uniform access
        for old, new in [("router_topk_regions", "router_topk_regions"),
                         ("router_topk_probs",   "router_topk_probs"),
                         ("mem_topk_regions",    "mem_topk_regions"),
                         ("mem_topk_probs",      "mem_topk_probs")]:
            pass  # already correct
        n = build_val_split(args, backbone, inv_map, coarse_map, d_model, vocab_size,
                            pp, pol_cfg, r2s, super_children)
    else:
        n = build_train_split(args, backbone, probe, inv_map, coarse_map, d_model,
                              vocab_size, pol_cfg, r2s, super_children, n_coarse)

    # Config file
    n_super_out = int(r2s.max()) + 1 if r2s is not None else 0
    meta = {
        "split":          args.split,
        "policy":         args.policy,
        "n_positions":    n,
        "candidate_cap":  args.candidate_cap,
        "n_fine":         n_coarse,
        "n_super":        n_super_out,
        "d_model":        d_model,
        "topk_stored":    args.topk,
    }
    with open(os.path.join(args.output_dir, "dataset_config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[build] done.  split={args.split}  positions={n:,}  dir={args.output_dir}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--split",          required=True, choices=["train", "val"])
    p.add_argument("--small_ckpt",     required=True)
    p.add_argument("--knn_run_dir",    required=True)
    p.add_argument("--region_map",     required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--policy",         default="hgrid_K24_srccombined_nts1_bkr12_fb24_cm0.15_en1.25_corec60m30")
    p.add_argument("--super_map",      default=None)
    p.add_argument("--dataset",        default="wikitext-103-raw-v1")
    p.add_argument("--seq_len",        type=int,   default=128)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--max_positions",  type=int,   default=1_000_000,
                   help="Max covered train positions (train split only).")
    p.add_argument("--shard_size",     type=int,   default=10_000)
    p.add_argument("--candidate_cap",  type=int,   default=4096)
    p.add_argument("--topk",           type=int,   default=32)
    p.add_argument("--knn_temp",       type=float, default=0.20)
    p.add_argument("--probe_temp",     type=float, default=1.0)
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
