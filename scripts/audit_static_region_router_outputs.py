#!/usr/bin/env python3
"""
audit_static_region_router_outputs.py — Phase 1A.4

Router Output / Error Anatomy Audit.

Evaluation only. No training. No weight updates.
Model input: input_ids only.
Gold used only after prediction for metrics and categorization.

Answers:
  Q1.  Gold-region rank distribution
  Q2.  Top8 miss breakdown (rank 9-16 / 17-32 / 33+)
  Q3.  Gold superregion present in top8/top16 when region missed
  Q4.  Are top8 predictions redundant or diverse?
  Q5.  Is the router uncertain when it misses?
  Q6.  Can margin/entropy fallback recover misses?
  Q7.  Candidate fraction needed for 90/95/98/99% recall
  Q8.  Failure taxonomy
  Q9.  Best next policy for Phase 1B
  Q10. Proceed or improve?

Usage:
  python scripts/audit_static_region_router_outputs.py \\
    --val_dir   runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val \\
    --train_dir runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train \\
    --token_to_region runs/region_maps_128/token_to_region.json \\
    --super_map  runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
    --router_ckpt runs/cheap_ai/phase1A_static_region_router/best_real_region_router.pt \\
    --router_run_dir runs/cheap_ai/phase1A_static_region_router \\
    --output_dir runs/cheap_ai/phase1A_static_region_router_audit \\
    --context_len 128 --d_model 256 --n_layers 2 --n_heads 4 \\
    --batch_size 512 --num_examples 30 --tokenizer_name gpt2 --seed 42
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
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice",       category=RuntimeWarning)

_EPS = 1e-9

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# Import model classes from training script (guarded by __main__ check there)
from train_static_region_router import (
    StaticRegionRouter,
    LastTokenMLPBaseline,
    MeanEmbeddingMLPBaseline,
)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _nanmean(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0: return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(a))

def _nanmedian(arr):
    a = np.asarray(arr, dtype=np.float64)
    v = a[~np.isnan(a)]
    return float(np.median(v)) if v.size > 0 else float("nan")

def _nanpercentile(arr, pct):
    a = np.asarray(arr, dtype=np.float64)
    v = a[~np.isnan(a)]
    return float(np.percentile(v, pct)) if v.size > 0 else float("nan")

def _fmt(v):
    if isinstance(v, bool):  return str(v)
    if isinstance(v, int):   return str(v)
    if isinstance(v, float): return f"{v:.5f}" if v == v else "nan"
    return str(v)

def _safediv(a, b, default=float("nan")):
    if b == 0 or b != b: return default
    return a / b

def _wcsv(path, rows, header=None):
    if not rows: return
    keys = header if header else list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"[save] {path}")

def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[save] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

_IDS_AL   = ["input_ids"]
_GOLD_AL  = ["gold_token", "gold", "labels"]
_TOPK_AL  = ["base_topk_ids", "base_topk", "topk_ids"]
_RID_AL   = ["row_id", "row_ids"]
_TOFF_AL  = ["token_offset"]


def _get_key(sh, aliases, required=True):
    for a in aliases:
        if a in sh: return sh[a]
    if required:
        raise KeyError(f"Need one of {aliases}; shard has {list(sh.keys())}")
    return None


def load_shards(shard_dir: str, max_rows: Optional[int], label: str) -> dict:
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")
    print(f"[shards] {label}: {len(paths)} shards in {shard_dir}")
    bufs: dict = defaultdict(list)
    total = 0; first = True
    for sp in paths:
        if max_rows is not None and total >= max_rows: break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print(f"[shards] keys: {list(sh.keys())}"); first = False
        ids  = _get_key(sh, _IDS_AL).long()
        gold = _get_key(sh, _GOLD_AL).long()
        topk = _get_key(sh, _TOPK_AL, required=False)
        rids = _get_key(sh, _RID_AL,  required=False)
        toff = _get_key(sh, _TOFF_AL, required=False)
        B, T = ids.shape
        if rids is None: rids = torch.arange(total, total + B)
        if toff is None: toff = torch.zeros(B, dtype=torch.long)
        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            ids, gold, rids, toff = ids[:keep], gold[:keep], rids[:keep], toff[:keep]
            if topk is not None: topk = topk[:keep]
            B = keep
        bufs["input_ids"].append(ids.numpy().astype(np.int32))
        bufs["gold"].append(gold.numpy().astype(np.int32))
        bufs["row_ids"].append(rids.numpy().astype(np.int32))
        bufs["token_offset"].append(toff.numpy().astype(np.int32))
        if topk is not None:
            bufs["base_topk_ids"].append(topk.numpy().astype(np.int32))
        total += B
    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    data["has_base_topk"] = "base_topk_ids" in data
    data["seq_len"] = int(data["input_ids"].shape[1])
    data["n_rows"]  = total
    print(f"[shards] {label}: {total:,} rows  T={data['seq_len']}  "
          f"has_base_topk={data['has_base_topk']}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Region maps
# ══════════════════════════════════════════════════════════════════════════════

def load_region_maps(t2r_path: str, super_path: Optional[str]):
    if not t2r_path or not os.path.isfile(t2r_path):
        return np.zeros(50257, np.int32), np.zeros(2, np.int32), 1, 1
    with open(t2r_path) as f: raw = json.load(f)
    t2r = ({i: v for i, v in enumerate(raw) if v is not None}
           if isinstance(raw, list)
           else {int(k): v for k, v in raw.items()})
    n_regions = int(max(t2r.values())) + 1 if t2r else 1
    V = max(t2r.keys(), default=0) + 2
    tok_arr = np.full(V, n_regions, dtype=np.int32)
    for t, r in t2r.items(): tok_arr[int(t)] = int(r)

    r2s = {}; n_super = 1
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f: raw2 = json.load(f)
        r2s = ({int(k): v for k, v in raw2.items()} if not isinstance(raw2, list)
               else {i: v for i, v in enumerate(raw2) if v is not None})
        n_super = int(max(r2s.values())) + 1 if r2s else 1
    R_arr = max(r2s.keys(), default=0) + 2
    reg_arr = np.full(R_arr, n_super, dtype=np.int32)
    for r, s in r2s.items(): reg_arr[int(r)] = int(s)
    print(f"[maps] n_regions={n_regions}  n_super={n_super}  V={V}")
    return tok_arr, reg_arr, n_regions, n_super


def build_superregion_to_regions(reg_arr: np.ndarray, n_regions: int, n_super: int) -> List[List[int]]:
    s2r: List[List[int]] = [[] for _ in range(n_super)]
    for r in range(n_regions):
        if r < len(reg_arr):
            s = int(reg_arr[r])
            if s < n_super: s2r[s].append(r)
    return s2r


def build_region_to_tokens(tok_arr: np.ndarray, n_regions: int) -> List[List[int]]:
    r2t: List[List[int]] = [[] for _ in range(n_regions)]
    for t, r in enumerate(tok_arr):
        if r < n_regions: r2t[r].append(t)
    return r2t


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_router(ckpt_path: str,
                run_dir: Optional[str],
                args,
                vocab_size: int,
                n_regions: int,
                n_super: int,
                device: torch.device):
    """Load checkpoint and instantiate the router model."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = saved.get("variant", args.variant)
    sd      = saved.get("state_dict", saved)
    print(f"[ckpt] loaded: {ckpt_path}  variant={variant}")

    # Try to load config from run_dir
    cfg = {}
    if run_dir and os.path.isfile(os.path.join(run_dir, "config.json")):
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        print(f"[ckpt] config from {run_dir}/config.json")

    d_model     = cfg.get("d_model",     args.d_model)
    n_layers    = cfg.get("n_layers",    args.n_layers)
    n_heads     = cfg.get("n_heads",     args.n_heads)
    dropout     = cfg.get("dropout",     args.dropout)
    max_seq_len = cfg.get("max_seq_len", args.context_len)
    use_super   = cfg.get("use_super_aux", n_super > 1)
    d_ff        = 4 * d_model

    print(f"[model] d_model={d_model}  n_layers={n_layers}  n_heads={n_heads}"
          f"  max_seq_len={max_seq_len}  n_regions={n_regions}  n_super={n_super}")

    # Determine model class from variant name
    if "last_token" in variant or "last_token_mlp" in variant:
        model = LastTokenMLPBaseline(vocab_size, n_regions, d_model, dropout)
    elif "mean_emb" in variant or "mean_embedding" in variant:
        model = MeanEmbeddingMLPBaseline(vocab_size, n_regions, d_model, dropout)
    else:
        model = StaticRegionRouter(
            vocab_size=vocab_size, n_regions=n_regions, n_super=n_super,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, max_seq_len=max_seq_len,
            use_super_aux=use_super,
        )

    model.load_state_dict(sd, strict=False)
    model.to(device); model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}  loaded successfully")
    return model, {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                   "max_seq_len": max_seq_len, "n_params": n_params, "variant": variant}


# ══════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        print(f"[tokenizer] loaded: {name}")
        return tok
    except Exception as e:
        print(f"[tokenizer] WARN: failed to load '{name}': {e} — continuing with IDs only")
        return None


def decode_token(tok_id: int, tokenizer) -> str:
    if tokenizer is None: return f"<{tok_id}>"
    try:
        return repr(tokenizer.decode([tok_id]))
    except Exception:
        return f"<{tok_id}>"


def decode_context(ids: np.ndarray, tokenizer) -> str:
    if tokenizer is None:
        return " ".join(str(i) for i in ids[-20:])
    try:
        return tokenizer.decode(ids.tolist(), skip_special_tokens=False)[-200:]
    except Exception:
        return " ".join(str(i) for i in ids[-20:])


# ══════════════════════════════════════════════════════════════════════════════
# Region representatives from train data
# ══════════════════════════════════════════════════════════════════════════════

def build_region_representatives(train_data: dict,
                                  tok_arr: np.ndarray,
                                  reg_arr: np.ndarray,
                                  n_regions: int,
                                  n_super: int,
                                  tokenizer,
                                  top_n: int = 20) -> List[dict]:
    gold   = train_data["gold"].astype(np.int64)
    Vt     = len(tok_arr)
    region_token_counts: List[Dict[int, int]] = [defaultdict(int) for _ in range(n_regions)]

    for g in gold:
        g = int(g)
        if 0 <= g < Vt:
            r = int(tok_arr[g])
            if r < n_regions:
                region_token_counts[r][g] += 1

    reps = []
    for r in range(n_regions):
        counts = region_token_counts[r]
        top = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
        top_ids    = [t for t, _ in top]
        top_counts = [c for _, c in top]
        top_strs   = [decode_token(t, tokenizer) for t in top_ids]
        sr = int(reg_arr[r]) if r < len(reg_arr) else n_super
        reps.append({
            "region_id":   r,
            "superregion_id": sr,
            "region_size": int((tok_arr == r).sum()),
            "top_token_ids":    top_ids,
            "top_token_strs":   top_strs,
            "top_token_counts": top_counts,
        })
    return reps


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, data: dict, context_len: int,
                  device: torch.device, batch_size: int) -> np.ndarray:
    N   = data["n_rows"]
    ids = data["input_ids"]
    avail_T = ids.shape[1]
    if context_len > avail_T:
        raise ValueError(f"context_len={context_len} > shard T={avail_T}. Cannot pad.")
    ids_ctx = ids[:, -context_len:]

    # Infer n_regions from model
    if hasattr(model, "region_head"):
        n_regions = model.region_head.out_features
    elif hasattr(model, "mlp"):
        n_regions = model.mlp[-1].out_features
    else:
        raise RuntimeError("Cannot infer n_regions from model")

    logits_all = np.zeros((N, n_regions), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e   = min(s + batch_size, N)
            ids_t = torch.from_numpy(ids_ctx[s:e].astype(np.int64)).to(device)
            reg_l, _ = model(ids_t)
            logits_all[s:e] = reg_l.cpu().float().numpy()

    if np.isnan(logits_all).any():
        raise RuntimeError("NaN in region logits during inference!")

    return logits_all


# ══════════════════════════════════════════════════════════════════════════════
# Per-row statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_per_row_stats(region_logits: np.ndarray,
                           gold: np.ndarray,
                           tok_arr: np.ndarray,
                           reg_arr: np.ndarray,
                           n_regions: int,
                           n_super: int) -> dict:
    N, R = region_logits.shape
    Vt   = len(tok_arr); Rr = len(reg_arr)

    # Gold region / super
    gold_clipped = np.clip(gold.astype(np.int64), 0, Vt - 1)
    gold_region  = tok_arr[gold_clipped].astype(np.int64)
    known_mask   = (gold_region < n_regions)
    gold_sr_arr  = reg_arr[np.clip(gold_region, 0, Rr - 1)].astype(np.int64)

    # Softmax probs
    lmax  = region_logits.max(axis=1, keepdims=True)
    exps  = np.exp((region_logits - lmax).astype(np.float64))
    probs = (exps / (exps.sum(axis=1, keepdims=True) + _EPS)).astype(np.float32)

    # Ranked
    ranked = np.argsort(-region_logits, axis=1)  # [N, R]

    # Per-row: rank of gold
    gold_rank = np.full(N, -1, dtype=np.int64)
    gold_prob  = np.zeros(N, dtype=np.float32)
    for i in np.where(known_mask)[0]:
        gr = int(gold_region[i])
        if gr < R:
            gold_rank[i] = int(np.searchsorted(ranked[i, ::-1], gr))
            # rank via argsort
            gold_rank[i] = int(np.where(ranked[i] == gr)[0][0])
            gold_prob[i]  = float(probs[i, gr])

    # Confidence stats
    top1_prob  = probs[:, ranked[:, 0].reshape(-1)].diagonal() if N > 0 else probs[:, 0]
    # Vectorised top1/top2
    top1_prob  = np.array([probs[i, ranked[i, 0]] for i in range(N)], dtype=np.float32)
    top2_prob  = np.array([probs[i, ranked[i, 1]] if R >= 2 else 0.0 for i in range(N)], dtype=np.float32)
    margin     = top1_prob - top2_prob
    entropy    = -(probs * np.log(probs + _EPS)).sum(axis=1).astype(np.float32)

    # Top-8 superregion diversity
    k8 = min(8, R)
    top8_regions = ranked[:, :k8]
    top8_srs = np.array([
        [int(reg_arr[int(top8_regions[i, j])]) if int(top8_regions[i, j]) < Rr else n_super
         for j in range(k8)]
        for i in range(N)
    ], dtype=np.int32)
    unique_sr_top8 = np.array([len(set(top8_srs[i].tolist())) for i in range(N)], dtype=np.int32)

    return {
        "gold_region":    gold_region,
        "gold_superregion": gold_sr_arr,
        "known_mask":     known_mask,
        "probs":          probs,
        "ranked":         ranked,
        "gold_rank":      gold_rank,
        "gold_prob":      gold_prob,
        "top1_prob":      top1_prob,
        "top2_prob":      top2_prob,
        "margin":         margin,
        "entropy":        entropy,
        "top8_regions":   top8_regions,
        "top8_srs":       top8_srs,
        "unique_sr_top8": unique_sr_top8,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Coverage metrics
# ══════════════════════════════════════════════════════════════════════════════

_AUDIT_K = [1, 2, 4, 8, 16, 32, 64]


def compute_coverage(stats: dict,
                     base_topk: Optional[np.ndarray],
                     tok_arr: np.ndarray,
                     reg_arr: np.ndarray,
                     n_regions: int,
                     vocab_known_count: int) -> dict:
    N, R = stats["probs"].shape
    ranked = stats["ranked"]
    known_mask = stats["known_mask"]
    gold_region = stats["gold_region"]
    Vt = len(tok_arr); Rr = len(reg_arr)
    has_base = base_topk is not None

    # Precompute region sizes
    region_sz = np.bincount(tok_arr[tok_arr < n_regions], minlength=n_regions).astype(np.int64)

    out: dict = {}
    for k in _AUDIT_K:
        k_ = min(k, R)
        topk_r = ranked[:, :k_]   # [N, k_]

        # Gold region in top-k
        gold_in_topk = np.array([
            (known_mask[i] and int(gold_region[i]) < R and int(gold_region[i]) in topk_r[i])
            for i in range(N)
        ])
        km = known_mask
        recall = float(gold_in_topk[km].mean()) if km.any() else float("nan")
        out[f"gold_region_recall@{k}"]  = recall
        out[f"gold_token_coverage@{k}"] = recall

        # Candidate set size
        cand_sizes = np.array([region_sz[topk_r[i]].sum() for i in range(N)], dtype=np.float64)
        out[f"avg_candidate_set_size@{k}"]    = float(cand_sizes[km].mean()) if km.any() else float("nan")
        out[f"median_candidate_set_size@{k}"] = float(np.median(cand_sizes[km])) if km.any() else float("nan")
        out[f"p90_candidate_set_size@{k}"]    = float(np.percentile(cand_sizes[km], 90)) if km.any() else float("nan")
        out[f"candidate_fraction@{k}"]        = _safediv(float(cand_sizes[km].mean()) if km.any() else 0.0, vocab_known_count)

        if has_base:
            Kb = base_topk.shape[1]
            b1  = np.clip(base_topk[:, 0], 0, Vt - 1)
            b5  = np.clip(base_topk[:, :min(5,  Kb)], 0, Vt - 1)
            b10 = np.clip(base_topk[:, :min(10, Kb)], 0, Vt - 1)

            def _region_in_topk(tok_id, i):
                r = int(tok_arr[tok_id]) if tok_id < Vt else n_regions
                return r < R and r in topk_r[i]

            b1_in  = np.array([_region_in_topk(b1[i], i) for i in range(N)])
            b5_any = np.array([any(_region_in_topk(b5[i, j], i) for j in range(b5.shape[1])) for i in range(N)])
            b5_all = np.array([all(_region_in_topk(b5[i, j], i) for j in range(b5.shape[1])) for i in range(N)])
            b10_any= np.array([any(_region_in_topk(b10[i, j],i) for j in range(b10.shape[1])) for i in range(N)])

            out[f"base_top1_in_C@{k}"]       = float(b1_in[km].mean()) if km.any() else float("nan")
            out[f"base_top5_any_in_C@{k}"]   = float(b5_any[km].mean()) if km.any() else float("nan")
            out[f"base_top5_all_in_C@{k}"]   = float(b5_all[km].mean()) if km.any() else float("nan")
            out[f"base_top10_any_in_C@{k}"]  = float(b10_any[km].mean()) if km.any() else float("nan")
        else:
            for sfx in ["base_top1_in_C", "base_top5_any_in_C", "base_top5_all_in_C", "base_top10_any_in_C"]:
                out[f"{sfx}@{k}"] = float("nan")

    return out, region_sz


# ══════════════════════════════════════════════════════════════════════════════
# Miss rank histogram
# ══════════════════════════════════════════════════════════════════════════════

def compute_miss_rank_histogram(stats: dict, n_regions: int) -> List[dict]:
    gold_rank  = stats["gold_rank"]
    known_mask = stats["known_mask"]

    buckets = [
        ("rank_1",       0,   0),
        ("rank_2_4",     1,   3),
        ("rank_5_8",     4,   7),
        ("rank_9_16",    8,  15),
        ("rank_17_32",  16,  31),
        ("rank_33_64",  32,  63),
        ("rank_65_128", 64, 127),
    ]
    rows = []
    N_known = int(known_mask.sum())
    if N_known == 0: return rows

    cum = 0
    for label, lo, hi in buckets:
        mask = known_mask & (gold_rank >= lo) & (gold_rank <= hi)
        n    = int(mask.sum())
        frac = n / N_known
        cum += n
        rows.append({
            "bucket":                     label,
            "n":                          n,
            "frac":                       round(frac, 5),
            "cumulative_recall_upper_bound": round(cum / N_known, 5),
        })

    unk_n = int((~known_mask).sum())
    rows.append({
        "bucket": "unknown_region",
        "n": unk_n,
        "frac": round(unk_n / (N_known + unk_n), 5),
        "cumulative_recall_upper_bound": float("nan"),
    })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Superregion near-miss analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_superregion_near_miss(stats: dict,
                                   n_regions: int,
                                   n_super: int,
                                   reg_arr: np.ndarray,
                                   s2r: List[List[int]]) -> Tuple[List[dict], dict]:
    N   = len(stats["gold_rank"])
    R   = stats["probs"].shape[1]
    Rr  = len(reg_arr)
    known_mask  = stats["known_mask"]
    gold_rank   = stats["gold_rank"]
    gold_region = stats["gold_region"]
    gold_sr_arr = stats["gold_superregion"]
    ranked      = stats["ranked"]

    # Misses = known rows where gold not in top-8
    miss_mask = known_mask & (gold_rank >= 8) & (gold_rank >= 0)
    miss_idx  = np.where(miss_mask)[0]

    rows = []
    for i in miss_idx:
        gr  = int(gold_region[i])
        gsr = int(gold_sr_arr[i])
        top8_reg  = ranked[i, :min(8, R)].tolist()
        top16_reg = ranked[i, :min(16, R)].tolist()
        top8_srs  = [int(reg_arr[r]) if r < Rr else n_super for r in top8_reg]
        top16_srs = [int(reg_arr[r]) if r < Rr else n_super for r in top16_reg]
        gsr_in8   = gsr in top8_srs
        gsr_in16  = gsr in top16_srs

        # Nearest same-superregion region rank
        same_sr_rank = next(
            (rk for rk, rr in enumerate(ranked[i]) if rr < Rr and int(reg_arr[rr]) == gsr),
            -1)

        rows.append({
            "row_idx":                    int(i),
            "gold_rank":                  int(gold_rank[i]),
            "gold_region":                int(gr),
            "gold_superregion":           int(gsr),
            "top8_superregions":          str(sorted(set(top8_srs))),
            "gold_superregion_in_top8":   gsr_in8,
            "gold_superregion_in_top16":  gsr_in16,
            "nearest_same_sr_rank":       same_sr_rank,
            "unique_superregions_top8":   len(set(top8_srs)),
            "unique_superregions_top16":  len(set(top16_srs)),
        })

    n_miss = len(miss_idx)
    n_known = int(known_mask.sum())
    summary = {
        "n_top8_miss":                 n_miss,
        "pct_of_known":                _safediv(n_miss, n_known),
        "pct_gold_rank_9_16":          _safediv(int(known_mask & (gold_rank >= 8)  & (gold_rank <= 15) & (gold_rank >= 0)).sum() if False
                                                else int(np.sum(known_mask & (gold_rank >= 8) & (gold_rank <= 15) & (gold_rank >= 0))),
                                                n_miss),
        "pct_gold_rank_17_32":         _safediv(int(np.sum(known_mask & (gold_rank >= 16) & (gold_rank <= 31) & (gold_rank >= 0))), n_miss),
        "pct_gold_rank_33_plus":       _safediv(int(np.sum(known_mask & (gold_rank >= 32) & (gold_rank >= 0))), n_miss),
        "pct_gold_superregion_in_top8":  _safediv(sum(r["gold_superregion_in_top8"]  for r in rows), n_miss),
        "pct_gold_superregion_in_top16": _safediv(sum(r["gold_superregion_in_top16"] for r in rows), n_miss),
    }
    return rows, summary


# ══════════════════════════════════════════════════════════════════════════════
# Top-8 diversity
# ══════════════════════════════════════════════════════════════════════════════

def compute_top8_diversity(stats: dict,
                            region_sz: np.ndarray,
                            n_regions: int,
                            vocab_known_count: int) -> Tuple[List[dict], dict]:
    N   = len(stats["gold_rank"])
    R   = stats["probs"].shape[1]
    k8  = min(8, R)
    top8  = stats["top8_regions"]       # [N, k8]
    srs8  = stats["top8_srs"]           # [N, k8]
    u_srs = stats["unique_sr_top8"]     # [N]
    probs = stats["probs"]

    cand_sizes = np.array([region_sz[top8[i]].sum() for i in range(N)], dtype=np.float64)
    cand_fracs = cand_sizes / max(vocab_known_count, 1)
    p90_cs     = float(np.percentile(cand_sizes, 90))

    entropy_top8 = np.array([
        float(-(probs[i, top8[i]] * np.log(probs[i, top8[i]] + _EPS)).sum())
        for i in range(N)
    ])

    redundant = u_srs <= 2
    diverse   = u_srs >= 6
    bloated   = cand_fracs >= np.percentile(cand_fracs, 90)
    focused   = (u_srs <= 3) & (entropy_top8 < np.percentile(entropy_top8, 33))

    km = stats["known_mask"]
    summary = {
        "pct_redundant_top8":  float(redundant[km].mean()) if km.any() else float("nan"),
        "pct_diverse_top8":    float(diverse[km].mean())   if km.any() else float("nan"),
        "pct_bloated_top8":    float(bloated[km].mean())   if km.any() else float("nan"),
        "pct_focused_top8":    float(focused[km].mean())   if km.any() else float("nan"),
        "avg_unique_sr_top8":  float(u_srs[km].mean())     if km.any() else float("nan"),
        "avg_cand_size_top8":  float(cand_sizes[km].mean()) if km.any() else float("nan"),
        "p90_cand_size_top8":  p90_cs,
        "avg_cand_frac_top8":  float(cand_fracs[km].mean()) if km.any() else float("nan"),
        "avg_entropy_top8":    float(entropy_top8[km].mean()) if km.any() else float("nan"),
    }

    stat_rows = []
    for i in range(N):
        stat_rows.append({
            "row_idx":              i,
            "unique_sr_top8":       int(u_srs[i]),
            "cand_size_top8":       int(cand_sizes[i]),
            "cand_frac_top8":       round(float(cand_fracs[i]), 5),
            "entropy_top8":         round(float(entropy_top8[i]), 5),
            "is_redundant":         bool(redundant[i]),
            "is_diverse":           bool(diverse[i]),
            "is_bloated":           bool(bloated[i]),
            "is_focused":           bool(focused[i]),
        })
    return stat_rows, summary


# ══════════════════════════════════════════════════════════════════════════════
# Confidence bins
# ══════════════════════════════════════════════════════════════════════════════

def _bin_stats(mask: np.ndarray, stats: dict, n_regions: int,
               km: np.ndarray, vocab_known_count: int) -> dict:
    m = mask & km
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "recall@1": float("nan"), "recall@4": float("nan"),
                "recall@8": float("nan"), "recall@16": float("nan"),
                "avg_cand_frac@8": float("nan"), "avg_gold_rank": float("nan"),
                "pct_top8_miss": float("nan"),
                "pct_rank9_16_of_miss": float("nan"),
                "pct_gold_sr_in_top8_of_miss": float("nan")}
    ranked     = stats["ranked"]
    gold_rank  = stats["gold_rank"]
    gold_region= stats["gold_region"]
    gold_sr    = stats["gold_superregion"]
    srs8       = stats["top8_srs"]
    probs      = stats["probs"]
    R          = probs.shape[1]
    region_sz  = np.bincount(np.zeros(1, dtype=np.int32), minlength=n_regions)  # placeholder

    midx = np.where(m)[0]
    r8   = sum(int(gold_region[i]) in ranked[i, :min(8,R)].tolist() for i in midx) / n
    r1   = sum(int(gold_region[i]) == ranked[i, 0] for i in midx) / n
    r4   = sum(int(gold_region[i]) in ranked[i, :min(4,R)].tolist() for i in midx) / n
    r16  = sum(int(gold_region[i]) in ranked[i, :min(16,R)].tolist() for i in midx) / n
    miss = [i for i in midx if int(gold_region[i]) not in ranked[i, :min(8,R)].tolist()]
    r9_16 = sum(1 for i in miss if 8 <= gold_rank[i] <= 15) / len(miss) if miss else float("nan")
    gsr_in_top8 = sum(1 for i in miss
                      if int(gold_sr[i]) in [int(s) for s in srs8[i]]) / len(miss) if miss else float("nan")
    avg_rank = float(gold_rank[list(midx)].mean())

    return {
        "n": n,
        "recall@1":  round(r1, 5),
        "recall@4":  round(r4, 5),
        "recall@8":  round(r8, 5),
        "recall@16": round(r16, 5),
        "avg_cand_frac@8": float("nan"),  # skipped for speed
        "avg_gold_rank": round(avg_rank, 2),
        "pct_top8_miss": round(1 - r8, 5),
        "pct_rank9_16_of_miss": _fmt(r9_16),
        "pct_gold_sr_in_top8_of_miss": _fmt(gsr_in_top8),
    }


def compute_confidence_bins(stats: dict, n_regions: int,
                             vocab_known_count: int) -> Tuple[List[dict], List[dict], List[dict]]:
    km  = stats["known_mask"]
    t1p = stats["top1_prob"]
    mrg = stats["margin"]
    ent = stats["entropy"]

    def _bins_for(vals, n_bins, label, rows_out):
        # uniform quantile bins
        km_vals = vals[km]
        if len(km_vals) == 0: return
        edges = [np.percentile(km_vals, p) for p in np.linspace(0, 100, n_bins + 1)]
        for b in range(n_bins):
            lo = edges[b]; hi = edges[b + 1]
            mask = km & (vals >= lo) & (vals <= hi)
            bs = _bin_stats(mask, stats, n_regions, km, vocab_known_count)
            rows_out.append({label: f"[{lo:.3f},{hi:.3f}]", **bs})

    rows_t1p, rows_mrg, rows_ent = [], [], []
    _bins_for(t1p, 10, "top1_prob_bin",    rows_t1p)
    _bins_for(mrg, 10, "margin_bin",       rows_mrg)
    _bins_for(ent, 10, "entropy_bin",      rows_ent)
    return rows_t1p, rows_mrg, rows_ent


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive fallback simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_fallback_policies(stats: dict,
                                base_topk: Optional[np.ndarray],
                                tok_arr: np.ndarray,
                                reg_arr: np.ndarray,
                                n_regions: int,
                                n_super: int,
                                s2r: List[List[int]],
                                vocab_known_count: int) -> List[dict]:
    N, R = stats["probs"].shape
    km       = stats["known_mask"]
    ranked   = stats["ranked"]
    gold_r   = stats["gold_region"]
    gold_sr  = stats["gold_superregion"]
    margin   = stats["margin"]
    entropy  = stats["entropy"]
    Rr       = len(reg_arr)
    Vt       = len(tok_arr)
    region_sz= np.bincount(tok_arr[tok_arr < n_regions], minlength=n_regions).astype(np.int64)
    has_base = base_topk is not None

    def _eval_policy(effective_k_per_row: np.ndarray, extra_regions_per_row: Optional[List[set]] = None,
                     label: str = "?", fallback_mask: Optional[np.ndarray] = None) -> dict:
        gold_in = np.zeros(N, dtype=bool)
        cand_sz = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if not km[i]: continue
            k_i  = int(effective_k_per_row[i])
            k_i  = min(k_i, R)
            chosen = set(ranked[i, :k_i].tolist())
            if extra_regions_per_row:
                chosen |= extra_regions_per_row[i]
            chosen = {r for r in chosen if r < n_regions}
            if int(gold_r[i]) in chosen:
                gold_in[i] = True
            cand_sz[i] = sum(region_sz[r] for r in chosen)

        recall  = float(gold_in[km].mean()) if km.any() else float("nan")
        avg_cs  = float(cand_sz[km].mean())  if km.any() else float("nan")
        med_cs  = float(np.median(cand_sz[km])) if km.any() else float("nan")
        p90_cs  = float(np.percentile(cand_sz[km], 90)) if km.any() else float("nan")
        frac    = _safediv(avg_cs, vocab_known_count)
        flop_r  = 1.0 - _safediv(avg_cs * 384, vocab_known_count * 384)
        fall_rt = float(fallback_mask.mean()) if fallback_mask is not None else float("nan")

        b5any = float("nan")
        if has_base:
            b5 = np.clip(base_topk[:, :min(5, base_topk.shape[1])], 0, Vt - 1)
            b5_any = np.zeros(N, dtype=bool)
            for i in range(N):
                if not km[i]: continue
                k_i = int(effective_k_per_row[i])
                chosen = set(ranked[i, :min(k_i, R)].tolist())
                if extra_regions_per_row: chosen |= extra_regions_per_row[i]
                chosen = {r for r in chosen if r < n_regions}
                b5_any[i] = any(int(tok_arr[b5[i, j]]) in chosen
                                for j in range(b5.shape[1]) if b5[i,j] < Vt)
            b5any = float(b5_any[km].mean()) if km.any() else float("nan")

        return {
            "policy": label,
            "fallback_rate": round(fall_rt, 5),
            "effective_region_recall": round(recall, 5),
            "gold_token_coverage": round(recall, 5),
            "avg_candidate_set_size": round(avg_cs, 1),
            "median_candidate_set_size": round(med_cs, 1),
            "p90_candidate_set_size": round(p90_cs, 1),
            "candidate_fraction": round(frac, 5),
            "base_top5_any_in_C": round(b5any, 5),
            "estimated_output_flop_reduction": round(flop_r, 5),
        }

    rows = []

    # Fixed-k baselines
    for k in [8, 16, 32]:
        eff_k = np.full(N, k, dtype=np.int32)
        rows.append(_eval_policy(eff_k, label=f"top{k}_always"))

    # Margin fallback
    ent_q = {q: float(np.percentile(entropy[km], q)) for q in [50, 60, 70, 80, 90]} if km.any() else {}
    for thr in [0.01, 0.02, 0.05, 0.10, 0.20]:
        fall = margin < thr
        eff_k = np.where(fall, 16, 8).astype(np.int32)
        rows.append(_eval_policy(eff_k, label=f"margin_fallback_t{thr:.2f}",
                                 fallback_mask=fall))
    for thr in [0.01, 0.02, 0.05, 0.10, 0.20]:
        fall = margin < thr
        eff_k = np.where(fall, 32, 8).astype(np.int32)
        rows.append(_eval_policy(eff_k, label=f"margin_fallback32_t{thr:.2f}",
                                 fallback_mask=fall))

    # Entropy fallback
    for qname, qval in ent_q.items():
        fall = entropy > qval
        eff_k = np.where(fall, 16, 8).astype(np.int32)
        rows.append(_eval_policy(eff_k, label=f"entropy_fallback16_q{qname}",
                                 fallback_mask=fall))
    for qname, qval in ent_q.items():
        fall = entropy > qval
        eff_k = np.where(fall, 32, 8).astype(np.int32)
        rows.append(_eval_policy(eff_k, label=f"entropy_fallback32_q{qname}",
                                 fallback_mask=fall))

    # Superregion expansion: top8 + all regions in same superregions
    eff_k8 = np.full(N, 8, dtype=np.int32)
    extra_sr_expand = []
    for i in range(N):
        top8_srs_i = set(stats["top8_srs"][i].tolist())
        extra = set()
        for s in top8_srs_i:
            if s < n_super:
                extra |= set(s2r[s])
        extra_sr_expand.append(extra)
    rows.append(_eval_policy(eff_k8, extra_regions_per_row=extra_sr_expand,
                             label="top8_plus_superregion_expansion"))

    # Top8 + N most frequent regions
    region_freq = np.bincount(
        tok_arr[np.clip(stats["gold_region"][km], 0, len(tok_arr)-1)] if km.any() else np.array([], dtype=np.int64),
        minlength=n_regions)
    freq_order = np.argsort(-region_freq)
    for n_freq in [2, 4, 8]:
        extra_freq = [{int(r) for r in freq_order[:n_freq]}] * N
        rows.append(_eval_policy(eff_k8, extra_regions_per_row=extra_freq,
                                 label=f"top8_plus_freq{n_freq}"))

    # Oracle: top8 + gold superregion regions (DIAGNOSTIC ONLY)
    extra_oracle = []
    for i in range(N):
        gs = int(stats["gold_superregion"][i])
        extra_oracle.append(set(s2r[gs]) if gs < n_super else set())
    rows.append(_eval_policy(eff_k8, extra_regions_per_row=extra_oracle,
                             label="top8_plus_gold_sr_ORACLE_INVALID"))

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def compute_confusion(stats: dict, n_regions: int, reps: List[dict],
                      top_n: int = 200) -> List[dict]:
    N   = len(stats["gold_rank"])
    km  = stats["known_mask"]
    gold_r  = stats["gold_region"]
    ranked  = stats["ranked"]
    gold_sr = stats["gold_superregion"]
    R = stats["probs"].shape[1]

    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for i in np.where(km)[0]:
        gr  = int(gold_r[i])
        pr  = int(ranked[i, 0])
        if gr < n_regions and pr < n_regions:
            counts[(gr, pr)] += 1

    # Build superregion map
    def _sr(r, reps):
        if 0 <= r < len(reps): return reps[r]["superregion_id"]
        return -1

    pairs = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    rows  = []
    for (gr, pr), cnt in pairs:
        g_reps = reps[gr]["top_token_strs"][:5] if gr < len(reps) else []
        p_reps = reps[pr]["top_token_strs"][:5] if pr < len(reps) else []
        rows.append({
            "gold_region":       gr,
            "predicted_region":  pr,
            "count":             cnt,
            "gold_superregion":  _sr(gr, reps),
            "predicted_superregion": _sr(pr, reps),
            "same_superregion":  _sr(gr, reps) == _sr(pr, reps),
            "gold_region_reps":  str(g_reps),
            "predicted_region_reps": str(p_reps),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Example generation
# ══════════════════════════════════════════════════════════════════════════════

def build_example(i: int, data: dict, stats: dict, reps: List[dict],
                  context_len: int, n_regions: int, n_super: int,
                  tok_arr: np.ndarray, reg_arr: np.ndarray,
                  region_sz: np.ndarray, tokenizer,
                  vocab_known_count: int) -> str:
    N_disp  = min(8, stats["probs"].shape[1])
    probs   = stats["probs"][i]
    ranked  = stats["ranked"][i]
    gold_r  = int(stats["gold_region"][i])
    gold_sr = int(stats["gold_superregion"][i])
    gr_rank = int(stats["gold_rank"][i])
    gold_id = int(data["gold"][i])
    t1p     = float(stats["top1_prob"][i])
    t2p     = float(stats["top2_prob"][i])
    ent     = float(stats["entropy"][i])
    marg    = float(stats["margin"][i])
    gold_p  = float(stats["gold_prob"][i])

    ids_ctx = data["input_ids"][i, -context_len:]
    ctx_str = decode_context(ids_ctx, tokenizer)
    gold_str= decode_token(gold_id, tokenizer)

    gold_reps = (reps[gold_r]["top_token_strs"][:5]
                 if 0 <= gold_r < len(reps) else [])

    # Coverage
    R = stats["probs"].shape[1]
    top8_set  = set(ranked[:min(8, R)].tolist())
    top16_set = set(ranked[:min(16, R)].tolist())
    cov8  = gold_r in top8_set
    cov16 = gold_r in top16_set
    cs8   = int(sum(region_sz[r] for r in top8_set if r < len(region_sz)))
    frac8 = _safediv(cs8, vocab_known_count)

    top8_srs  = set(stats["top8_srs"][i].tolist())
    u_srs8    = len(top8_srs)
    base_top10 = data["base_topk_ids"][i, :10].tolist() if data.get("has_base_topk") else None

    lines = [
        f"## Example (row_idx={i}  row_id={int(data['row_ids'][i])}  "
        f"token_offset={int(data['token_offset'][i])})",
        "",
        "**Context (last 200 chars):**",
        f"```",
        f"{ctx_str}",
        f"```",
        "",
        f"**Gold:** token={gold_id} {gold_str}  region={gold_r}  "
        f"superregion={gold_sr}  reps={gold_reps}",
        "",
        f"**Router stats:**  gold_rank={gr_rank}  gold_prob={gold_p:.5f}  "
        f"top1_prob={t1p:.5f}  top2_prob={t2p:.5f}  "
        f"margin={marg:.5f}  entropy={ent:.4f}",
        "",
        "**Top predicted regions:**",
    ]
    for rk in range(N_disp):
        r   = int(ranked[rk])
        p   = float(probs[r])
        sr  = int(reg_arr[r]) if r < len(reg_arr) else n_super
        sz  = int(region_sz[r]) if r < len(region_sz) else 0
        trps= reps[r]["top_token_strs"][:4] if r < len(reps) else []
        marker = " <-- GOLD" if r == gold_r else ""
        lines.append(f"  {rk+1:2d}. region={r:4d}  p={p:.5f}  super={sr}  "
                     f"size={sz}  reps={trps}{marker}")
    lines += [
        "",
        f"**top8_unique_superregions:** {u_srs8}  "
        f"cand_set_size@8={cs8}  candidate_fraction@8={frac8:.4f}",
        f"**gold_covered@8:** {cov8}  **gold_covered@16:** {cov16}",
    ]
    if base_top10:
        b_strs = [decode_token(t, tokenizer) for t in base_top10]
        lines.append(f"**base_top10:** {list(zip(base_top10, b_strs))}")
    lines.append("")
    return "\n".join(lines)


def write_examples(data: dict, stats: dict, reps: List[dict],
                   context_len: int, n_regions: int, n_super: int,
                   tok_arr: np.ndarray, reg_arr: np.ndarray,
                   region_sz: np.ndarray, s2r: List[List[int]],
                   tokenizer, vocab_known_count: int,
                   out_dir: str, num_examples: int):

    R   = stats["probs"].shape[1]
    km  = stats["known_mask"]
    gr  = stats["gold_rank"]
    gsr = stats["gold_superregion"]
    top8_srs = stats["top8_srs"]
    u_srs    = stats["unique_sr_top8"]
    probs    = stats["probs"]
    ranked   = stats["ranked"]
    k8 = min(8, R)
    gold_r   = stats["gold_region"]

    def _cand_sz(i):
        return int(sum(region_sz[r] for r in ranked[i, :k8].tolist() if r < len(region_sz)))

    cand_fracs  = np.array([_safediv(_cand_sz(i), vocab_known_count) for i in range(len(km))])
    p90_cf = float(np.percentile(cand_fracs[km], 90)) if km.any() else 1.0

    categories = {
        "examples_top1_success":
            np.where(km & (gr == 0))[0],
        "examples_top8_success":
            np.where(km & (gr > 0) & (gr < 8) & (gr >= 0))[0],
        "examples_top8_near_miss_rank9_16":
            np.where(km & (gr >= 8) & (gr <= 15) & (gr >= 0))[0],
        "examples_top8_medium_miss_rank17_32":
            np.where(km & (gr >= 16) & (gr <= 31) & (gr >= 0))[0],
        "examples_top8_hard_miss_rank33plus":
            np.where(km & (gr >= 32) & (gr >= 0))[0],
        "examples_high_conf_wrong":
            np.where(km & (gr >= 8) & (gr >= 0) & (stats["top1_prob"] >= 0.3))[0],
        "examples_low_conf_wrong":
            np.where(km & (gr >= 8) & (gr >= 0) & (stats["entropy"] > np.percentile(stats["entropy"][km], 80) if km.any() else stats["entropy"] > 0))[0],
        "examples_gold_superregion_present_but_region_missed":
            np.where(km & (gr >= 8) & (gr >= 0) &
                     np.array([int(gsr[i]) in set(top8_srs[i].tolist()) for i in range(len(km))]))[0],
        "examples_gold_superregion_absent":
            np.where(km & (gr >= 8) & (gr >= 0) &
                     np.array([int(gsr[i]) not in set(top8_srs[i].tolist()) for i in range(len(km))]))[0],
        "examples_redundant_top8":
            np.where(km & (u_srs <= 2))[0],
        "examples_diverse_top8":
            np.where(km & (u_srs >= 6))[0],
        "examples_bloated_candidate_set":
            np.where(km & (cand_fracs >= p90_cf))[0],
    }

    rng = np.random.default_rng(42)
    for fname, idxs in categories.items():
        chosen = idxs[:num_examples] if len(idxs) <= num_examples else rng.choice(idxs, num_examples, replace=False)
        parts  = [f"# {fname.replace('_',' ').title()}\n\n"
                  f"n_total_in_category={len(idxs)}  showing={len(chosen)}\n\n"]
        for i in sorted(chosen):
            parts.append(build_example(
                int(i), data, stats, reps, context_len,
                n_regions, n_super, tok_arr, reg_arr, region_sz,
                tokenizer, vocab_known_count))
        _write(os.path.join(out_dir, f"{fname}.md"), "".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# Coverage target: find k needed for X% recall
# ══════════════════════════════════════════════════════════════════════════════

def find_coverage_targets(stats: dict, region_sz: np.ndarray,
                           vocab_known_count: int, n_regions: int) -> dict:
    km      = stats["known_mask"]
    gold_r  = stats["gold_region"]
    ranked  = stats["ranked"]
    N, R    = stats["probs"].shape
    targets = {0.90: None, 0.95: None, 0.98: None, 0.99: None}

    for k in range(1, R + 1):
        hits = sum(1 for i in np.where(km)[0]
                   if int(gold_r[i]) in ranked[i, :k].tolist())
        recall = hits / int(km.sum()) if km.any() else 0.0
        for tgt in list(targets.keys()):
            if recall >= tgt and targets[tgt] is None:
                cand_sizes = np.array([region_sz[ranked[i, :k]].sum() for i in np.where(km)[0]])
                targets[tgt] = {
                    "k": k,
                    "recall": round(recall, 5),
                    "avg_candidate_set_size": round(float(cand_sizes.mean()), 1),
                    "candidate_fraction": round(_safediv(float(cand_sizes.mean()), vocab_known_count), 5),
                }
        if all(v is not None for v in targets.values()):
            break
    return targets


# ══════════════════════════════════════════════════════════════════════════════
# Main audit report
# ══════════════════════════════════════════════════════════════════════════════

def write_audit_report(cov: dict, miss_hist: List[dict], near_miss_sum: dict,
                        div_sum: dict, fallback_rows: List[dict],
                        coverage_targets: dict,
                        model_cfg: dict, out_dir: str, args) -> str:

    def _v(k, default=float("nan")): return cov.get(k, default)
    def _fb(label):
        for r in fallback_rows:
            if r["policy"] == label: return r
        return {}

    recall8  = _v("gold_region_recall@8")
    recall16 = _v("gold_region_recall@16")
    recall32 = _v("gold_region_recall@32")
    frac8    = _v("candidate_fraction@8")

    best_policy = max((r for r in fallback_rows if "ORACLE" not in r["policy"]),
                      key=lambda r: r.get("effective_region_recall", 0),
                      default={})

    # Recommendation logic
    if recall8 == recall8 and recall16 == recall16:
        if recall8 >= 0.90 and recall16 >= 0.95:
            rec = "PROCEED_TO_ROUTED_CANDIDATE_SOFTMAX"
        elif best_policy.get("effective_region_recall", 0) >= 0.95 and \
                best_policy.get("candidate_fraction", 1.0) <= 0.25:
            rec = "BUILD_ADAPTIVE_FALLBACK_POLICY"
        elif near_miss_sum.get("pct_gold_superregion_in_top8", 0) < 0.5:
            rec = "NEED_BETTER_REGION_MAP"
        elif recall8 < 0.70:
            rec = "DO_NOT_PROCEED"
        else:
            rec = "IMPROVE_LOCAL_ROUTER"
    else:
        rec = "NEED_FULL_50K_REGION_COVERAGE"

    lines = [
        "# Phase 1A.4: Router Output / Error Anatomy Audit",
        "",
        f"**model:** {model_cfg.get('variant','?')}  |  "
        f"**d_model:** {model_cfg.get('d_model','?')}  |  "
        f"**n_layers:** {model_cfg.get('n_layers','?')}  |  "
        f"**params:** {model_cfg.get('n_params',0):,}",
        "", "---", "",
        "## Global Coverage", "",
        "| k | recall@k | avg_cand_size | cand_fraction | base_top5_any |",
        "|---|---|---|---|---|",
    ]
    for k in _AUDIT_K:
        lines.append(f"| {k} | {_fmt(_v(f'gold_region_recall@{k}'))} "
                     f"| {_fmt(_v(f'avg_candidate_set_size@{k}'))} "
                     f"| {_fmt(_v(f'candidate_fraction@{k}'))} "
                     f"| {_fmt(_v(f'base_top5_any_in_C@{k}'))} |")

    lines += ["", "## Coverage Targets", ""]
    for tgt, info in sorted(coverage_targets.items()):
        if info:
            lines.append(f"- **{tgt*100:.0f}%** recall: k={info['k']}  "
                         f"avg_cand={info['avg_candidate_set_size']}  "
                         f"fraction={info['candidate_fraction']}")
        else:
            lines.append(f"- **{tgt*100:.0f}%** recall: NOT REACHED in {stats_placeholder} regions")

    lines += ["", "## Miss Rank Histogram", "",
              "| bucket | n | frac | cumulative_upper_bound |",
              "|---|---|---|---|"]
    for r in miss_hist:
        lines.append(f"| {r['bucket']} | {r['n']} | {r['frac']} | {r['cumulative_recall_upper_bound']} |")

    lines += ["", "## Near-Miss Superregion Analysis", ""]
    for k, v in near_miss_sum.items():
        lines.append(f"- **{k}:** {_fmt(v)}")

    lines += ["", "## Top-8 Diversity", ""]
    for k, v in div_sum.items():
        lines.append(f"- **{k}:** {_fmt(v)}")

    lines += ["", "## Best Fallback Policies (top 10 by recall)", "",
              "| policy | fallback_rate | recall | cand_frac | avg_cands | est_flop_red |",
              "|---|---|---|---|---|---|"]
    non_oracle = [r for r in fallback_rows if "ORACLE" not in r["policy"]]
    for r in sorted(non_oracle, key=lambda x: -x.get("effective_region_recall", 0))[:10]:
        lines.append(f"| {r['policy']} | {r['fallback_rate']} "
                     f"| {r['effective_region_recall']} | {r['candidate_fraction']} "
                     f"| {r['avg_candidate_set_size']} "
                     f"| {r['estimated_output_flop_reduction']} |")

    def _q(n, q, ans, detail=""):
        lines.append(f"\n### Q{n}: {q}")
        lines.append(f"**{ans}**")
        if detail: lines.append(f"\n{detail}")

    lines += ["", "---", "", "## Q&A", ""]

    rank_hist_str = "  ".join(f"{r['bucket']}={r['frac']:.3f}" for r in miss_hist if r["bucket"] != "unknown_region")
    _q(1, "What is the gold-region rank distribution?", rank_hist_str)
    n9_16 = next((r["n"] for r in miss_hist if r["bucket"] == "rank_9_16"), 0)
    n_tot_miss = sum(r["n"] for r in miss_hist if r["bucket"] not in ("rank_1","unknown_region"))
    _q(2, "Of top8 misses, how many are rank 9-16?",
       f"{n9_16} ({_fmt(_safediv(n9_16, n_tot_miss))} of misses)",
       f"These are recoverable with top16 fallback.")
    _q(3, "Of top8 misses, how many have gold superregion present in top8/top16?",
       f"in_top8={_fmt(near_miss_sum.get('pct_gold_superregion_in_top8'))}  "
       f"in_top16={_fmt(near_miss_sum.get('pct_gold_superregion_in_top16'))}")
    _q(4, "Are top8 predictions redundant/similar or diverse/spread out?",
       f"avg_unique_sr={_fmt(div_sum.get('avg_unique_sr_top8'))}  "
       f"pct_redundant={_fmt(div_sum.get('pct_redundant_top8'))}  "
       f"pct_diverse={_fmt(div_sum.get('pct_diverse_top8'))}")
    _q(5, "Is the router uncertain when it misses?",
       "See confidence_bins_*.csv for entropy/margin vs recall breakdown.")
    best_fb = best_policy
    _q(6, "Can margin/entropy fallback recover misses?",
       f"Best non-oracle policy: {best_fb.get('policy','?')}  "
       f"recall={best_fb.get('effective_region_recall','?')}  "
       f"fraction={best_fb.get('candidate_fraction','?')}  "
       f"fallback_rate={best_fb.get('fallback_rate','?')}")
    ct_lines = []
    for tgt, info in sorted(coverage_targets.items()):
        if info:
            ct_lines.append(f"  {tgt*100:.0f}%: k={info['k']} fraction={info['candidate_fraction']}")
        else:
            ct_lines.append(f"  {tgt*100:.0f}%: not reached")
    _q(7, "What candidate fraction is needed to reach 90/95/98/99% recall?",
       "\n".join(ct_lines) or "see coverage_targets in report")
    _q(8, "What do failures look like?",
       "See examples_*.md and confusion analysis.",
       f"pct_gold_sr_in_top8={_fmt(near_miss_sum.get('pct_gold_superregion_in_top8'))}"
       f"  pct_redundant={_fmt(div_sum.get('pct_redundant_top8'))}")
    _q(9, "What is the best next policy for Phase 1B?",
       f"{best_fb.get('policy','?')} — achieve {best_fb.get('effective_region_recall','?')} recall "
       f"with fraction {best_fb.get('candidate_fraction','?')}")
    _q(10, "Should we improve router, change region map, or proceed?",
       rec,
       f"recall@8={_fmt(recall8)}  recall@16={_fmt(recall16)}  recall@32={_fmt(recall32)}")

    lines += ["", "---", "",
              f"## FINAL RECOMMENDATION: {rec}", ""]

    path = os.path.join(out_dir, "audit_report.md")
    _write(path, "\n".join(lines))
    return rec


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 1A.4: Router Output/Error Anatomy Audit")
    p.add_argument("--val_dir",         required=True)
    p.add_argument("--train_dir",       required=True)
    p.add_argument("--token_to_region", required=True)
    p.add_argument("--super_map",       default=None)
    p.add_argument("--router_ckpt",     required=True)
    p.add_argument("--router_run_dir",  default=None)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--context_len",     type=int,   default=128)
    p.add_argument("--d_model",         type=int,   default=256)
    p.add_argument("--n_layers",        type=int,   default=2)
    p.add_argument("--n_heads",         type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--variant",         type=str,   default="real_region_router")
    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--num_examples",    type=int,   default=30)
    p.add_argument("--max_val_rows",    type=int,   default=None)
    p.add_argument("--max_train_rows",  type=int,   default=None)
    p.add_argument("--tokenizer_name",  type=str,   default="gpt2")
    p.add_argument("--no_tokenizer",    action="store_true")
    p.add_argument("--seed",            type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    t0 = time.time()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = None
    if not args.no_tokenizer:
        tokenizer = load_tokenizer(args.tokenizer_name)

    # ── Region maps ───────────────────────────────────────────────────────────
    print("\n[step 1] Loading region maps...")
    tok_arr, reg_arr, n_regions, n_super = load_region_maps(
        args.token_to_region, args.super_map)
    vocab_known_count = int((tok_arr < n_regions).sum())
    s2r = build_superregion_to_regions(reg_arr, n_regions, n_super)
    r2t = build_region_to_tokens(tok_arr, n_regions)
    region_sz = np.array([len(r2t[r]) for r in range(n_regions)], dtype=np.int64)
    print(f"[maps] vocab_known_count={vocab_known_count:,}")

    # ── Infer vocab size from data ────────────────────────────────────────────
    print("\n[step 2] Loading val data (for vocab size probe)...")
    val_data   = load_shards(args.val_dir,   args.max_val_rows,   "val")
    vocab_size = max(int(val_data["input_ids"].max()) + 1, 50257)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[step 3] Loading router checkpoint...")
    model, model_cfg = load_router(
        args.router_ckpt, args.router_run_dir, args,
        vocab_size, n_regions, n_super, device)

    # ── Train data for region representatives ─────────────────────────────────
    print("\n[step 4] Loading train data for region representatives...")
    train_data = load_shards(args.train_dir, args.max_train_rows, "train")
    print("[step 4] Building region representatives...")
    reps = build_region_representatives(train_data, tok_arr, reg_arr, n_regions, n_super, tokenizer)

    # Save region representatives
    with open(os.path.join(args.output_dir, "region_representatives.json"), "w", encoding="utf-8") as f:
        json.dump(reps, f, indent=2)
    print(f"[save] {args.output_dir}/region_representatives.json")

    txt_lines = []
    for r in reps:
        txt_lines.append(f"region={r['region_id']:3d}  super={r['superregion_id']:2d}  "
                         f"size={r['region_size']:5d}  "
                         f"top_tokens={r['top_token_strs'][:8]}")
    _write(os.path.join(args.output_dir, "region_representatives.txt"), "\n".join(txt_lines))

    sz_rows = [{"region_id": r["region_id"], "superregion_id": r["superregion_id"],
                "region_size": r["region_size"],
                "top_token_ids": str(r["top_token_ids"][:5]),
                "top_token_strs": str(r["top_token_strs"][:5])}
               for r in reps]
    _wcsv(os.path.join(args.output_dir, "region_size_table.csv"), sz_rows)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\n[step 5] Running inference on {val_data['n_rows']:,} val rows...")
    t_inf = time.time()
    region_logits = run_inference(model, val_data, args.context_len, device, args.batch_size)
    print(f"[step 5] inference done in {time.time()-t_inf:.1f}s")

    # ── Per-row statistics ────────────────────────────────────────────────────
    print("\n[step 6] Computing per-row statistics...")
    stats = compute_per_row_stats(
        region_logits, val_data["gold"].astype(np.int64),
        tok_arr, reg_arr, n_regions, n_super)
    # Make stats globally visible for write_audit_report placeholder
    global stats_placeholder
    stats_placeholder = n_regions

    # ── Coverage ──────────────────────────────────────────────────────────────
    print("[step 6] Computing coverage metrics...")
    base_topk = val_data.get("base_topk_ids") if val_data.get("has_base_topk") else None
    cov_metrics, _rszcheck = compute_coverage(
        stats, base_topk, tok_arr, reg_arr, n_regions, vocab_known_count)
    cov_rows = [{"metric": k, "value": v} for k, v in cov_metrics.items()]
    _wcsv(os.path.join(args.output_dir, "coverage_by_k.csv"), cov_rows)

    # Summary CSV
    summary_rows = [{"metric": k, "value": v} for k, v in {
        "n_total":          val_data["n_rows"],
        "n_known_region":   int(stats["known_mask"].sum()),
        "n_unknown_region": int((~stats["known_mask"]).sum()),
        **{k: v for k, v in cov_metrics.items() if "@8" in k or "@16" in k or "@32" in k},
    }.items()]
    _wcsv(os.path.join(args.output_dir, "router_error_summary.csv"), summary_rows)

    # ── Miss rank histogram ───────────────────────────────────────────────────
    print("[step 7] Computing miss rank histogram...")
    miss_hist = compute_miss_rank_histogram(stats, n_regions)
    _wcsv(os.path.join(args.output_dir, "miss_rank_histogram.csv"), miss_hist)

    # ── Superregion near-miss ─────────────────────────────────────────────────
    print("[step 8] Computing superregion near-miss analysis...")
    near_miss_rows, near_miss_sum = compute_superregion_near_miss(
        stats, n_regions, n_super, reg_arr, s2r)
    _wcsv(os.path.join(args.output_dir, "superregion_near_miss.csv"), near_miss_rows[:5000])

    # ── Top-8 diversity ───────────────────────────────────────────────────────
    print("[step 9] Computing top-8 diversity...")
    div_stat_rows, div_sum = compute_top8_diversity(stats, region_sz, n_regions, vocab_known_count)
    _wcsv(os.path.join(args.output_dir, "top8_diversity_stats.csv"), div_stat_rows[:5000])
    _wcsv(os.path.join(args.output_dir, "top8_diversity_summary.csv"), [div_sum])

    # ── Confidence bins ───────────────────────────────────────────────────────
    print("[step 10] Computing confidence bins...")
    rows_t1p, rows_mrg, rows_ent = compute_confidence_bins(stats, n_regions, vocab_known_count)
    _wcsv(os.path.join(args.output_dir, "confidence_bins_top1prob.csv"), rows_t1p)
    _wcsv(os.path.join(args.output_dir, "confidence_bins_margin.csv"),   rows_mrg)
    _wcsv(os.path.join(args.output_dir, "confidence_bins_entropy.csv"),  rows_ent)

    # ── Adaptive fallback simulation ──────────────────────────────────────────
    print("[step 11] Simulating adaptive fallback policies...")
    fallback_rows = simulate_fallback_policies(
        stats, base_topk, tok_arr, reg_arr, n_regions, n_super,
        s2r, vocab_known_count)
    _wcsv(os.path.join(args.output_dir, "adaptive_fallback_policies.csv"), fallback_rows)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print("[step 12] Computing confusion matrix...")
    conf_rows = compute_confusion(stats, n_regions, reps)
    _wcsv(os.path.join(args.output_dir, "router_confusion_pairs.csv"), conf_rows)
    conf_md = ["# Top Region Confusions\n",
               "| gold_region | predicted_region | count | same_superregion | "
               "gold_reps | predicted_reps |",
               "|---|---|---|---|---|---|"]
    for r in conf_rows[:100]:
        conf_md.append(f"| {r['gold_region']} | {r['predicted_region']} "
                       f"| {r['count']} | {r['same_superregion']} "
                       f"| {r['gold_region_reps'][:60]} "
                       f"| {r['predicted_region_reps'][:60]} |")
    _write(os.path.join(args.output_dir, "top_region_confusions.md"), "\n".join(conf_md))

    # ── Coverage targets ──────────────────────────────────────────────────────
    print("[step 13] Computing coverage targets...")
    coverage_targets = find_coverage_targets(stats, region_sz, vocab_known_count, n_regions)
    print(f"  coverage targets: {coverage_targets}")

    # ── Examples ──────────────────────────────────────────────────────────────
    print("[step 14] Writing examples...")
    write_examples(val_data, stats, reps, args.context_len,
                   n_regions, n_super, tok_arr, reg_arr,
                   region_sz, s2r, tokenizer, vocab_known_count,
                   args.output_dir, args.num_examples)

    # ── Audit report ──────────────────────────────────────────────────────────
    print("[step 15] Writing audit report...")
    recommendation = write_audit_report(
        cov_metrics, miss_hist, near_miss_sum, div_sum,
        fallback_rows, coverage_targets, model_cfg, args.output_dir, args)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg_out = {
        "router_ckpt": args.router_ckpt,
        "val_dir": args.val_dir,
        "train_dir": args.train_dir,
        "context_len": args.context_len,
        "n_val_rows": val_data["n_rows"],
        "n_known_region": int(stats["known_mask"].sum()),
        "vocab_known_count": vocab_known_count,
        "n_regions": n_regions,
        "n_super": n_super,
        **model_cfg,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    km = stats["known_mask"]
    gr = stats["gold_rank"]
    n_known = int(km.sum())
    n9_16   = int(np.sum(km & (gr >= 8)  & (gr <= 15) & (gr >= 0)))
    n17_32  = int(np.sum(km & (gr >= 16) & (gr <= 31) & (gr >= 0)))
    n33plus = int(np.sum(km & (gr >= 32) & (gr >= 0)))

    best_fb = max((r for r in fallback_rows if "ORACLE" not in r["policy"]),
                  key=lambda r: r.get("effective_region_recall", 0), default={})

    print(f"\n{'='*60}")
    print(f" PHASE 1A.4 ROUTER OUTPUT AUDIT  ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  recall@8:   {_fmt(cov_metrics.get('gold_region_recall@8'))}")
    print(f"  recall@16:  {_fmt(cov_metrics.get('gold_region_recall@16'))}")
    print(f"  recall@32:  {_fmt(cov_metrics.get('gold_region_recall@32'))}")
    print()
    print(f"  top8 miss breakdown (of {n_known} known rows):")
    print(f"    rank 9-16:    {n9_16} ({_fmt(_safediv(n9_16, n_known))})")
    print(f"    rank 17-32:   {n17_32} ({_fmt(_safediv(n17_32, n_known))})")
    print(f"    rank 33+:     {n33plus} ({_fmt(_safediv(n33plus, n_known))})")
    print()
    print(f"  gold superregion present:")
    print(f"    in top8:  {_fmt(near_miss_sum.get('pct_gold_superregion_in_top8'))}")
    print(f"    in top16: {_fmt(near_miss_sum.get('pct_gold_superregion_in_top16'))}")
    print()
    print(f"  confidence fallback best policy:")
    print(f"    policy:    {best_fb.get('policy','?')}")
    print(f"    recall:    {best_fb.get('effective_region_recall','?')}")
    print(f"    fraction:  {best_fb.get('candidate_fraction','?')}")
    print(f"    fallback%: {best_fb.get('fallback_rate','?')}")
    print()
    print(f"  recommendation: {recommendation}")
    print(f"{'='*60}")
    print(f"\nOutputs: {args.output_dir}/")


if __name__ == "__main__":
    main()
