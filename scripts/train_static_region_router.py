#!/usr/bin/env python3
"""
train_static_region_router.py — Phase 1A: Static Region Router

CORE QUESTION
─────────────
Can a cheap causal attention router predict the gold-containing static region?

Given input_ids, predict which of R=128 static regions contains the gold token.
Success = gold region in top-k predicted regions.

Model input:  input_ids only.
Model target: token_to_region[gold_token].
No h_ctx, no h_raw, no base_topk_ids/logits as input.
No gold force-inclusion. No token-level reranking.

Variants trained:
  real_region_router        — real token_to_region map
  shuffled_region_router    — shuffled map (same region sizes)
  random_region_router      — random map (same region sizes)

Baselines:
  frequency_prior_baseline  — always rank regions by train frequency
  last_token_mlp_baseline   — embed last token, MLP, region logits
  mean_embedding_mlp_baseline — mean-pool token embeddings, MLP, region logits

Usage:
  python scripts/train_static_region_router.py \\
    --train_dir  runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train \\
    --val_dir    runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val \\
    --token_to_region runs/region_maps_128/token_to_region.json \\
    --super_map  runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
    --output_dir runs/cheap_ai/phase1A_static_region_router \\
    --d_model 256 --n_layers 2 --n_heads 4 --dropout 0.1 \\
    --batch_size 256 --steps 10000 --eval_every 500 \\
    --lr 3e-4 --weight_decay 0.01 --grad_clip 1.0 \\
    --lambda_super 0.2 --seed 42 --amp
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
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice",       category=RuntimeWarning)

_EPS = 1e-9

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _nanmean(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(a))


def _nanmedian(arr):
    a = np.asarray(arr, dtype=np.float64)
    valid = a[~np.isnan(a)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def _nanpercentile(arr, pct):
    a = np.asarray(arr, dtype=np.float64)
    valid = a[~np.isnan(a)]
    if valid.size == 0:
        return float("nan")
    return float(np.percentile(valid, pct))


def _wcsv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {path}")


def _fmt(v):
    if isinstance(v, bool):  return str(v)
    if isinstance(v, int):   return str(v)
    if isinstance(v, float): return f"{v:.5f}" if v == v else "nan"
    return str(v)


def _safediv(a, b, default=float("nan")):
    if b == 0 or b != b:
        return default
    return a / b


# ══════════════════════════════════════════════════════════════════════════════
# Shard loading (input_ids-centric)
# ══════════════════════════════════════════════════════════════════════════════

_IDS_ALIASES  = ["input_ids"]
_GOLD_ALIASES = ["gold_token", "gold", "labels"]
_TOPK_ALIASES = ["base_topk_ids", "base_topk", "topk_ids"]
_RID_ALIASES  = ["row_id", "row_ids"]
_TOFF_ALIASES = ["token_offset"]


def _get_key(sh, aliases, required=True):
    for a in aliases:
        if a in sh:
            return sh[a]
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
        if max_rows is not None and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print(f"[shards] keys: {list(sh.keys())}")
            first = False

        ids  = _get_key(sh, _IDS_ALIASES).long()          # [B, T]
        gold = _get_key(sh, _GOLD_ALIASES).long()          # [B]
        topk = _get_key(sh, _TOPK_ALIASES, required=False) # [B, K] optional
        rids = _get_key(sh, _RID_ALIASES,  required=False) # [B]
        toff = _get_key(sh, _TOFF_ALIASES, required=False) # [B]

        B, T = ids.shape
        if rids is None:
            rids = torch.arange(total, total + B)
        if toff is None:
            toff = torch.zeros(B, dtype=torch.long)

        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            ids  = ids[:keep]; gold = gold[:keep]; rids = rids[:keep]; toff = toff[:keep]
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
    """Returns tok_arr, reg_arr, n_regions, n_super."""
    if not t2r_path or not os.path.isfile(t2r_path):
        return np.zeros(50257, np.int32), np.zeros(2, np.int32), 1, 1
    with open(t2r_path) as f:
        raw = json.load(f)
    t2r = ({i: v for i, v in enumerate(raw) if v is not None}
           if isinstance(raw, list)
           else {int(k): v for k, v in raw.items()})
    n_regions = int(max(t2r.values())) + 1 if t2r else 1
    V = max(t2r.keys(), default=0) + 2
    tok_arr = np.full(V, n_regions, dtype=np.int32)  # unknown = n_regions
    for t, r in t2r.items():
        tok_arr[int(t)] = int(r)

    r2s = {}; n_super = 1
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f:
            raw2 = json.load(f)
        r2s = ({int(k): v for k, v in raw2.items()} if not isinstance(raw2, list)
               else {i: v for i, v in enumerate(raw2) if v is not None})
        n_super = int(max(r2s.values())) + 1 if r2s else 1
    R_arr = max(r2s.keys(), default=0) + 2
    reg_arr = np.full(R_arr, n_super, dtype=np.int32)
    for r, s in r2s.items():
        reg_arr[int(r)] = int(s)
    print(f"[maps] n_regions={n_regions}  n_super={n_super}  V={V}")
    return tok_arr, reg_arr, n_regions, n_super


def make_shuffled_tok_arr(tok_arr: np.ndarray, n_regions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    valid = np.where(tok_arr < n_regions)[0]
    vals = tok_arr[valid].copy()
    rng.shuffle(vals)
    out = tok_arr.copy()
    out[valid] = vals
    return out


def make_random_tok_arr(tok_arr: np.ndarray, n_regions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    valid = np.where(tok_arr < n_regions)[0]
    vals  = tok_arr[valid]
    sizes = np.bincount(vals, minlength=n_regions)
    new   = np.repeat(np.arange(n_regions, dtype=np.int32), sizes)
    rng.shuffle(new)
    out = tok_arr.copy()
    out[valid] = new
    return out


def region_sizes(tok_arr: np.ndarray, n_regions: int) -> List[int]:
    unk = n_regions
    valid = tok_arr[tok_arr < unk]
    return np.bincount(valid, minlength=n_regions).tolist()


def build_region_to_tokens(tok_arr: np.ndarray, n_regions: int) -> List[List[int]]:
    """Returns list of length n_regions; each entry is list of token ids in that region."""
    r2t: List[List[int]] = [[] for _ in range(n_regions)]
    for t, r in enumerate(tok_arr):
        if r < n_regions:
            r2t[r].append(t)
    return r2t


def build_region_cumsize(r2t: List[List[int]]) -> np.ndarray:
    """Precompute region sizes array for fast candidate_set_size computation."""
    return np.array([len(r2t[r]) for r in range(len(r2t))], dtype=np.int64)


def save_tok_arr_as_json(tok_arr: np.ndarray, n_regions: int, path: str):
    out = {}
    for t, r in enumerate(tok_arr):
        if r < n_regions:
            out[str(t)] = int(r)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[save] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Model: StaticRegionRouter
# ══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(
            ~self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class StaticRegionRouter(nn.Module):
    """
    Causal transformer router: input_ids → region logits over R regions.
    Uses only input_ids; no h_ctx, no base logits.
    """

    def __init__(self,
                 vocab_size:  int,
                 n_regions:   int,
                 n_super:     int,
                 d_model:     int = 256,
                 n_layers:    int = 2,
                 n_heads:     int = 4,
                 d_ff:        int = 1024,
                 dropout:     float = 0.1,
                 max_seq_len: int = 128,
                 use_super_aux: bool = True):
        super().__init__()
        self.n_regions  = n_regions
        self.n_super    = n_super
        self.d_model    = d_model
        self.max_seq_len = max_seq_len
        self.use_super_aux = use_super_aux and (n_super > 1)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.region_head = nn.Linear(d_model, n_regions)
        self.super_head  = (nn.Linear(d_model, n_super)
                            if self.use_super_aux else None)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        T    = min(T, self.max_seq_len)
        ids  = input_ids[:, :T]
        pos  = torch.arange(T, device=ids.device).unsqueeze(0)
        x    = self.emb_drop(self.tok_emb(ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        h_last = x[:, -1, :]
        region_logits = self.region_head(h_last)
        super_logits  = self.super_head(h_last) if self.use_super_aux else None
        return region_logits, super_logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_forward_flops(self, seq_len: int) -> int:
        T = min(seq_len, self.max_seq_len)
        d = self.d_model
        n = len(self.blocks)
        # per layer: attn QKV + proj + FF
        attn_flops = 2 * T * (3 * d * d) + T * T * d + 2 * T * d * d
        ff_flops   = 2 * T * d * (4 * d)
        total = n * (attn_flops + ff_flops) + T * d * self.n_regions
        return total


# ══════════════════════════════════════════════════════════════════════════════
# Baselines
# ══════════════════════════════════════════════════════════════════════════════

class LastTokenMLPBaseline(nn.Module):
    """Embed last input token → MLP → region logits."""

    def __init__(self, vocab_size: int, n_regions: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_regions),
        )

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        last = input_ids[:, -1].clamp(0, self.tok_emb.num_embeddings - 1)
        h = self.tok_emb(last)
        return self.mlp(h), None

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MeanEmbeddingMLPBaseline(nn.Module):
    """Mean-pool token embeddings → MLP → region logits."""

    def __init__(self, vocab_size: int, n_regions: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_regions),
        )

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        h = self.tok_emb(input_ids.clamp(0, self.tok_emb.num_embeddings - 1)).mean(dim=1)
        return self.mlp(h), None

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# Gold region computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_gold_regions(gold: np.ndarray, tok_arr: np.ndarray,
                         n_regions: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (gold_region, known_mask).
    gold_region[i] = n_regions if unknown.
    known_mask[i]  = True if gold[i] has a known region.
    """
    Vt = len(tok_arr)
    gold_clipped = np.clip(gold.astype(np.int64), 0, Vt - 1)
    gold_region  = tok_arr[gold_clipped].astype(np.int64)
    known_mask   = (gold_region < n_regions)
    return gold_region, known_mask


def compute_gold_superregions(gold_region: np.ndarray, reg_arr: np.ndarray,
                              n_super: int) -> np.ndarray:
    Rr = len(reg_arr)
    gr_clipped = np.clip(gold_region, 0, Rr - 1)
    gold_super = reg_arr[gr_clipped].astype(np.int64)
    return gold_super


# ══════════════════════════════════════════════════════════════════════════════
# Frequency prior baseline
# ══════════════════════════════════════════════════════════════════════════════

def compute_frequency_prior(gold: np.ndarray, tok_arr: np.ndarray,
                            n_regions: int) -> np.ndarray:
    """Compute region frequency as gold in train. Returns sorted region indices (desc)."""
    gold_region, known_mask = compute_gold_regions(gold, tok_arr, n_regions)
    counts = np.bincount(gold_region[known_mask], minlength=n_regions).astype(np.float64)
    freq_order = np.argsort(-counts)  # descending frequency
    return counts, freq_order


# ══════════════════════════════════════════════════════════════════════════════
# Coverage metrics computation
# ══════════════════════════════════════════════════════════════════════════════

_TOPK_LIST = [1, 2, 4, 8, 16, 32]


def coverage_metrics(region_logits: np.ndarray,
                     gold_region: np.ndarray,
                     gold_tokens: np.ndarray,
                     base_topk_ids: Optional[np.ndarray],
                     region_sizes_arr: np.ndarray,
                     known_mask: np.ndarray,
                     n_regions: int,
                     vocab_known_count: int) -> Dict[str, float]:
    """
    region_logits: [N, R]  (float, may be nan for unknown rows)
    gold_region:   [N]     (int, n_regions = unknown)
    gold_tokens:   [N]     (int)
    base_topk_ids: [N, K]  or None
    region_sizes_arr: [R]  cumulative per-region token counts
    known_mask: [N]
    """
    N, R = region_logits.shape
    out  = {}

    ranked = np.argsort(-region_logits, axis=1)  # [N, R]

    for k in _TOPK_LIST:
        k_ = min(k, R)
        topk_regions = ranked[:, :k_]             # [N, k_]

        # gold_region_recall@k
        gold_in_topk = np.zeros(N, dtype=bool)
        for i in range(N):
            if known_mask[i]:
                gold_in_topk[i] = gold_region[i] in topk_regions[i]
        recall_all = float(gold_in_topk[known_mask].mean()) if known_mask.any() else float("nan")
        out[f"gold_region_recall@{k}"] = recall_all

        # gold_token_coverage@k (same as recall since regions are disjoint)
        out[f"gold_token_coverage@{k}"] = recall_all

        # candidate_set_size@k
        cand_sizes = np.array([
            region_sizes_arr[topk_regions[i]].sum() for i in range(N)
        ], dtype=np.float64)
        out[f"avg_candidate_set_size@{k}"]    = float(cand_sizes[known_mask].mean()) if known_mask.any() else float("nan")
        out[f"median_candidate_set_size@{k}"] = float(np.median(cand_sizes[known_mask])) if known_mask.any() else float("nan")
        out[f"p90_candidate_set_size@{k}"]    = float(np.percentile(cand_sizes[known_mask], 90)) if known_mask.any() else float("nan")
        out[f"candidate_fraction@{k}"]        = _safediv(
            float(cand_sizes[known_mask].mean()) if known_mask.any() else 0.0,
            vocab_known_count)

        # base coverage metrics
        if base_topk_ids is not None:
            base1  = base_topk_ids[:, 0]         # top-1 base token
            base5  = base_topk_ids[:, :min(5, base_topk_ids.shape[1])]
            base10 = base_topk_ids[:, :min(10, base_topk_ids.shape[1])]

            # For each example, build set of candidate tokens = union of tokens in top-k regions
            # For large N this is expensive; use vectorised region-membership check instead:
            # token t is in C_k iff tok_arr[t] is in top-k regions for that example.
            # We have base_topk_ids and region_sizes_arr; we need per-example region membership.
            # Use a smaller approach: check if the region of each base token is in top-k regions.
            # (We don't have tok_arr here; we use gold_region for gold — same logic applies for base tokens)
            # We pass tok_arr in separately, done via _tok_arr_for_coverage set at call site
            pass
        out[f"base_top1_in_C@{k}"]      = float("nan")
        out[f"base_top5_any_in_C@{k}"]  = float("nan")
        out[f"base_top5_all_in_C@{k}"]  = float("nan")
        out[f"base_top10_any_in_C@{k}"] = float("nan")

    return out


def coverage_metrics_with_tok_arr(region_logits: np.ndarray,
                                   gold_region: np.ndarray,
                                   base_topk_ids: Optional[np.ndarray],
                                   tok_arr: np.ndarray,
                                   region_sizes_arr: np.ndarray,
                                   known_mask: np.ndarray,
                                   n_regions: int,
                                   vocab_known_count: int) -> Dict[str, float]:
    """Full coverage metrics including base-token coverage."""
    N, R = region_logits.shape
    out  = {}

    ranked = np.argsort(-region_logits, axis=1)  # [N, R]
    Vt = len(tok_arr)

    has_base = base_topk_ids is not None

    for k in _TOPK_LIST:
        k_ = min(k, R)
        topk_regions = ranked[:, :k_]  # [N, k_]

        # Build a boolean mask over regions for each example: [N, R]
        in_topk = np.zeros((N, R), dtype=bool)
        for ki in range(k_):
            r_col = topk_regions[:, ki]             # [N]
            valid = r_col < R
            in_topk[np.where(valid)[0], r_col[valid]] = True

        # gold_region_recall@k
        known_idx = np.where(known_mask)[0]
        recall_vals = np.zeros(len(known_idx), dtype=bool)
        for ii, i in enumerate(known_idx):
            gr = gold_region[i]
            if gr < R:
                recall_vals[ii] = in_topk[i, gr]
        recall = float(recall_vals.mean()) if len(recall_vals) > 0 else float("nan")

        out[f"gold_region_recall@{k}"]   = recall
        out[f"gold_token_coverage@{k}"]  = recall

        # candidate_set_size@k — vectorised using region_sizes_arr
        cand_sizes = (in_topk.astype(np.int64) * region_sizes_arr[np.newaxis, :]).sum(axis=1).astype(np.float64)

        km = known_mask
        out[f"avg_candidate_set_size@{k}"]    = float(cand_sizes[km].mean()) if km.any() else float("nan")
        out[f"median_candidate_set_size@{k}"] = float(np.median(cand_sizes[km])) if km.any() else float("nan")
        out[f"p90_candidate_set_size@{k}"]    = float(np.percentile(cand_sizes[km], 90)) if km.any() else float("nan")
        out[f"candidate_fraction@{k}"]        = _safediv(float(cand_sizes[km].mean()) if km.any() else 0.0, vocab_known_count)

        # base token coverage
        if has_base:
            K_base = base_topk_ids.shape[1]
            b1  = np.clip(base_topk_ids[:, 0], 0, Vt - 1)
            b5  = np.clip(base_topk_ids[:, :min(5,  K_base)], 0, Vt - 1)
            b10 = np.clip(base_topk_ids[:, :min(10, K_base)], 0, Vt - 1)

            b1_reg  = tok_arr[b1]                          # [N]
            b5_reg  = tok_arr[b5.reshape(-1)].reshape(-1, min(5, K_base))  # [N, 5]
            b10_reg = tok_arr[b10.reshape(-1)].reshape(-1, min(10, K_base))# [N, 10]

            b1_in   = np.array([in_topk[i, b1_reg[i]] if b1_reg[i] < R else False for i in range(N)])
            b5_any  = np.array([any(in_topk[i, r] if r < R else False for r in b5_reg[i]) for i in range(N)])
            b5_all  = np.array([all(in_topk[i, r] if r < R else False for r in b5_reg[i]) for i in range(N)])
            b10_any = np.array([any(in_topk[i, r] if r < R else False for r in b10_reg[i]) for i in range(N)])

            out[f"base_top1_in_C@{k}"]      = float(b1_in[km].mean())  if km.any() else float("nan")
            out[f"base_top5_any_in_C@{k}"]  = float(b5_any[km].mean()) if km.any() else float("nan")
            out[f"base_top5_all_in_C@{k}"]  = float(b5_all[km].mean()) if km.any() else float("nan")
            out[f"base_top10_any_in_C@{k}"] = float(b10_any[km].mean()) if km.any() else float("nan")
        else:
            for key in [f"base_top1_in_C@{k}", f"base_top5_any_in_C@{k}",
                        f"base_top5_all_in_C@{k}", f"base_top10_any_in_C@{k}"]:
                out[key] = float("nan")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Model evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_model(variant_name: str,
               model,                    # nn.Module or None (frequency prior)
               freq_order: Optional[np.ndarray],
               data: dict,
               tok_arr: np.ndarray,
               reg_arr: np.ndarray,
               n_regions: int,
               n_super: int,
               region_sizes_arr: np.ndarray,
               vocab_known_count: int,
               device: torch.device,
               batch_size: int,
               args) -> Tuple[dict, List[dict], List[dict]]:
    """
    Evaluates a model or baseline on data.
    Returns (metrics_dict, slice_rows, example_rows).
    """
    N  = data["n_rows"]
    gold   = data["gold"].astype(np.int64)
    inp    = data["input_ids"]
    base_topk = data.get("base_topk_ids")  # may be None

    gold_region, known_mask = compute_gold_regions(gold, tok_arr, n_regions)
    gold_super              = compute_gold_superregions(gold_region, reg_arr, n_super)

    # ── Collect region logits ──────────────────────────────────────────────────
    region_logits_all = np.zeros((N, n_regions), dtype=np.float32)
    super_logits_all  = None
    if n_super > 1 and (model is None or (hasattr(model, "use_super_aux") and model.use_super_aux)):
        super_logits_all = np.zeros((N, n_super), dtype=np.float32)

    if model is None:
        # Frequency prior: fixed logit = log(freq + 1) for every example
        assert freq_order is not None
        freq_logits = np.zeros(n_regions, dtype=np.float32)
        for rank, r in enumerate(freq_order):
            freq_logits[r] = n_regions - rank
        region_logits_all[:] = freq_logits[np.newaxis, :]
    else:
        model.eval()
        with torch.no_grad():
            for s in range(0, N, batch_size):
                e = min(s + batch_size, N)
                ids_t = torch.from_numpy(inp[s:e].astype(np.int64)).to(device)
                reg_l, sup_l = model(ids_t)
                region_logits_all[s:e] = reg_l.cpu().float().numpy()
                if super_logits_all is not None and sup_l is not None:
                    super_logits_all[s:e] = sup_l.cpu().float().numpy()

    # NaN check
    if np.isnan(region_logits_all).any():
        raise RuntimeError(f"[{variant_name}] NaN in region logits during eval!")

    # ── Ranked regions ────────────────────────────────────────────────────────
    ranked = np.argsort(-region_logits_all, axis=1)  # [N, R]

    # ── Region metrics ────────────────────────────────────────────────────────
    probs    = torch.from_numpy(region_logits_all).float()
    probs_sm = F.softmax(probs, dim=-1).numpy()

    region_ce_arr   = np.full(N, np.nan, np.float64)
    region_rank_arr = np.full(N, np.nan, np.float64)
    region_ent_arr  = np.full(N, np.nan, np.float64)
    margin_arr      = np.full(N, np.nan, np.float64)

    known_idx = np.where(known_mask)[0]
    for i in known_idx:
        gr  = int(gold_region[i])
        if gr >= n_regions:
            continue
        # CE
        lsm = region_logits_all[i] - np.log(np.exp(region_logits_all[i].astype(np.float64)).sum() + _EPS)
        region_ce_arr[i] = -lsm[gr].astype(np.float64)
        # Rank (0-indexed)
        region_rank_arr[i] = float(np.sum(region_logits_all[i] > region_logits_all[i, gr]))
        # Entropy
        p = probs_sm[i]
        region_ent_arr[i] = float(-(p * np.log(p + _EPS)).sum())
        # Margin top1 - top2
        top2 = np.partition(-region_logits_all[i], 1)[:2]
        margin_arr[i] = float(-top2[1] - (-top2[0]))

    # recall@k
    recall_at = {}
    for k in _TOPK_LIST:
        k_ = min(k, n_regions)
        hits = np.array([
            (gold_region[i] in ranked[i, :k_])
            for i in known_idx
        ], dtype=bool) if len(known_idx) > 0 else np.array([], dtype=bool)
        recall_at[k] = float(hits.mean()) if hits.size > 0 else float("nan")

    # ── Superregion metrics ───────────────────────────────────────────────────
    super_metrics = {}
    if super_logits_all is not None:
        s_ranked = np.argsort(-super_logits_all, axis=1)
        s_known  = known_mask & (gold_super < n_super)
        sk_idx   = np.where(s_known)[0]
        super_ce_arr = np.full(N, np.nan)
        for i in sk_idx:
            gs = int(gold_super[i])
            lsm = super_logits_all[i] - np.log(np.exp(super_logits_all[i].astype(np.float64)).sum() + _EPS)
            super_ce_arr[i] = float(-lsm[gs])
        super_metrics["superregion_ce"]      = _nanmean(super_ce_arr)
        super_metrics["superregion_acc@1"]   = float(
            (s_ranked[sk_idx, 0] == gold_super[sk_idx]).mean()) if len(sk_idx) > 0 else float("nan")
        for k in [2, 4, 8]:
            k_ = min(k, n_super)
            hits = np.array([gold_super[i] in s_ranked[i, :k_] for i in sk_idx], dtype=bool)
            super_metrics[f"superregion_recall@{k}"] = float(hits.mean()) if hits.size > 0 else float("nan")

    # ── Coverage metrics ──────────────────────────────────────────────────────
    cov_metrics = coverage_metrics_with_tok_arr(
        region_logits_all, gold_region, base_topk, tok_arr,
        region_sizes_arr, known_mask, n_regions, vocab_known_count)

    # ── Efficiency estimates ──────────────────────────────────────────────────
    base_d_model = getattr(args, "base_d_model", 384)
    full_lm_flops = int(getattr(args, "vocab_size_full", 50257)) * base_d_model
    params = model.count_params() if model is not None and hasattr(model, "count_params") else 0
    seq_len = data["seq_len"]
    fwd_flops = model.estimate_forward_flops(seq_len) if (model is not None and hasattr(model, "estimate_forward_flops")) else 0

    efficiency = {"router_params": params, "router_forward_flops_estimate": fwd_flops,
                  "full_lm_head_flops_estimate": full_lm_flops}
    for k in _TOPK_LIST:
        avg_cs = cov_metrics.get(f"avg_candidate_set_size@{k}", float("nan"))
        sel_fl = avg_cs * base_d_model if avg_cs == avg_cs else float("nan")
        red    = 1.0 - _safediv(sel_fl, full_lm_flops) if (sel_fl == sel_fl and full_lm_flops > 0) else float("nan")
        efficiency[f"selected_lm_head_flops_estimate@{k}"]  = sel_fl
        efficiency[f"theoretical_output_flop_reduction@{k}"] = red

    # ── Aggregate metrics dict ────────────────────────────────────────────────
    n_known  = int(known_mask.sum())
    n_unk    = N - n_known
    acc_at1 = float((ranked[known_idx, 0] == gold_region[known_idx]).mean()) if n_known > 0 else float("nan")

    metrics = {
        "variant":               variant_name,
        "n_total":               N,
        "n_known_region":        n_known,
        "n_unknown_region":      n_unk,
        "region_ce":             _nanmean(region_ce_arr),
        "region_acc@1":          acc_at1,
        "mean_rank_gold_region": _nanmean(region_rank_arr),
        "median_rank_gold_region": _nanmedian(region_rank_arr),
        "region_entropy":        _nanmean(region_ent_arr),
        "region_margin_top1_top2": _nanmean(margin_arr),
    }
    for k in _TOPK_LIST:
        metrics[f"region_recall@{k}"] = recall_at[k]
    metrics.update(super_metrics)
    metrics.update(cov_metrics)
    metrics.update(efficiency)

    # ── Slice metrics ─────────────────────────────────────────────────────────
    # Compute helper arrays for slices
    has_base = base_topk is not None
    base_correct_mask = np.zeros(N, dtype=bool)
    gold_in_base64_mask = np.zeros(N, dtype=bool)
    if has_base:
        base_top1 = base_topk[:, 0]
        base_correct_mask = (base_top1 == gold)
        K_base = base_topk.shape[1]
        b64 = base_topk[:, :min(64, K_base)]
        gold_in_base64_mask = (b64 == gold[:, np.newaxis]).any(axis=1)

    region_freqs = np.bincount(gold_region[known_mask], minlength=n_regions).astype(np.float64) if n_known > 0 else np.zeros(n_regions)
    freq_threshold_high = np.percentile(region_freqs[region_freqs > 0], 75) if (region_freqs > 0).any() else 0.0
    freq_threshold_low  = np.percentile(region_freqs[region_freqs > 0], 25) if (region_freqs > 0).any() else 0.0
    high_freq_regions = set(np.where(region_freqs >= freq_threshold_high)[0].tolist())
    low_freq_regions  = set(np.where(region_freqs <= freq_threshold_low)[0].tolist())

    short_ctx = (data["input_ids"].shape[1] < 32) if data["input_ids"].shape[1] < 32 else None
    seq_len_per_ex = np.array([data["seq_len"]] * N)  # uniform in this dataset

    def _slice_mask(name: str) -> np.ndarray:
        if name == "all":
            return known_mask
        if name == "high_frequency_gold_regions":
            return known_mask & np.array([gold_region[i] in high_freq_regions for i in range(N)])
        if name == "low_frequency_gold_regions":
            return known_mask & np.array([gold_region[i] in low_freq_regions for i in range(N)])
        if name == "base_correct":
            return known_mask & base_correct_mask
        if name == "base_wrong":
            return known_mask & ~base_correct_mask
        if name == "gold_in_base_top64":
            return known_mask & gold_in_base64_mask
        if name == "gold_not_in_base_top64":
            return known_mask & ~gold_in_base64_mask
        return np.zeros(N, dtype=bool)

    slice_names = ["all", "high_frequency_gold_regions", "low_frequency_gold_regions",
                   "base_correct", "base_wrong", "gold_in_base_top64", "gold_not_in_base_top64"]
    if not has_base:
        slice_names = ["all", "high_frequency_gold_regions", "low_frequency_gold_regions"]

    slice_rows = []
    for sname in slice_names:
        smask = _slice_mask(sname)
        sn    = int(smask.sum())
        if sn == 0:
            row = {"variant": variant_name, "slice": sname, "n": 0}
            for k_ in [1, 4, 8, 16]:
                row[f"region_acc@{k_}"]   = float("nan")
            for k_ in [8]:
                row[f"avg_candidate_set_size@{k_}"] = float("nan")
                row[f"candidate_fraction@{k_}"]      = float("nan")
                row[f"gold_token_coverage@{k_}"]     = float("nan")
            slice_rows.append(row)
            continue
        sidx = np.where(smask)[0]
        s_ranked = ranked[sidx]
        s_gold_r = gold_region[sidx]
        srow = {"variant": variant_name, "slice": sname, "n": sn}
        srow["region_acc@1"] = float((s_ranked[:, 0] == s_gold_r).mean())
        for k_ in [4, 8, 16]:
            k__ = min(k_, n_regions)
            hits = np.array([s_gold_r[ii] in s_ranked[ii, :k__] for ii in range(len(sidx))], dtype=bool)
            srow[f"region_recall@{k_}"] = float(hits.mean())
        # candidate set size and coverage for k=8
        k8 = min(8, n_regions)
        cand_sizes_s = (
            np.array([region_sizes_arr[s_ranked[ii, :k8]].sum() for ii in range(len(sidx))], dtype=np.float64)
        )
        srow["avg_candidate_set_size@8"]  = float(cand_sizes_s.mean())
        srow["candidate_fraction@8"]       = _safediv(float(cand_sizes_s.mean()), vocab_known_count)
        srow["gold_token_coverage@8"]      = srow[f"region_recall@8"] if "region_recall@8" in srow else float("nan")
        slice_rows.append(srow)

    # ── Example rows ──────────────────────────────────────────────────────────
    example_rows = []
    max_ex = 20
    k_ex   = min(8, n_regions)
    for i in known_idx[:max_ex * 5]:
        if len(example_rows) >= max_ex:
            break
        gr     = int(gold_region[i])
        top8   = ranked[i, :k_ex].tolist()
        covered = gr in top8
        prob8  = [float(probs_sm[i, r]) for r in top8]
        cs8    = int(region_sizes_arr[top8].sum())
        ex = {
            "row_id":         int(data["row_ids"][i]),
            "token_offset":   int(data["token_offset"][i]),
            "gold_token_id":  int(gold[i]),
            "gold_region":    gr,
            "top8_regions":   top8,
            "top8_probs":     [round(p, 5) for p in prob8],
            "gold_covered@8": covered,
            "cand_set_size@8": cs8,
            "base_top10":     base_topk[i, :10].tolist() if has_base else None,
        }
        example_rows.append(ex)

    return metrics, slice_rows, example_rows


# ══════════════════════════════════════════════════════════════════════════════
# Training loop (one variant)
# ══════════════════════════════════════════════════════════════════════════════

def train_variant(variant_name: str,
                  model: nn.Module,
                  train_data: dict,
                  val_data: dict,
                  tok_arr: np.ndarray,
                  reg_arr: np.ndarray,
                  n_regions: int,
                  n_super: int,
                  region_sizes_arr: np.ndarray,
                  vocab_known_count: int,
                  args,
                  device: torch.device,
                  out_dir: str) -> dict:

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    N_train = train_data["n_rows"]
    gold_train = train_data["gold"].astype(np.int64)
    inp_train  = train_data["input_ids"]
    gold_region_train, known_mask_train = compute_gold_regions(gold_train, tok_arr, n_regions)
    gold_super_train = compute_gold_superregions(gold_region_train, reg_arr, n_super)

    known_idx_train = np.where(known_mask_train)[0]
    if len(known_idx_train) == 0:
        print(f"[{variant_name}] WARNING: no known-region training rows!")

    rng = np.random.default_rng(args.seed)
    train_log: List[dict] = []
    eval_log:  List[dict] = []
    best_score  = -float("inf")
    best_ckpt   = None
    best_step   = None
    best_vm     = None
    t0 = time.time()

    model.train()
    step = 0

    while step < args.steps:
        # Sample batch from known rows only
        idx_pool = rng.choice(known_idx_train,
                               size=min(args.batch_size, len(known_idx_train)),
                               replace=len(known_idx_train) < args.batch_size)
        ids_t  = torch.from_numpy(inp_train[idx_pool].astype(np.int64)).to(device)
        gold_r = torch.from_numpy(gold_region_train[idx_pool]).to(device)
        gold_s = torch.from_numpy(gold_super_train[idx_pool]).to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast("cuda"):
                reg_logits, sup_logits = model(ids_t)
                loss_r = F.cross_entropy(reg_logits, gold_r.long())
                loss = loss_r
                if sup_logits is not None and args.lambda_super > 0:
                    sup_known = (gold_s < n_super)
                    if sup_known.any():
                        loss_s = F.cross_entropy(sup_logits[sup_known], gold_s[sup_known].long())
                        loss = loss + args.lambda_super * loss_s
                    else:
                        loss_s = torch.zeros(1, device=device).squeeze()
                else:
                    loss_s = torch.zeros(1, device=device).squeeze()
        else:
            reg_logits, sup_logits = model(ids_t)
            loss_r = F.cross_entropy(reg_logits, gold_r.long())
            loss = loss_r
            if sup_logits is not None and args.lambda_super > 0:
                sup_known = (gold_s < n_super)
                if sup_known.any():
                    loss_s = F.cross_entropy(sup_logits[sup_known], gold_s[sup_known].long())
                    loss = loss + args.lambda_super * loss_s
                else:
                    loss_s = torch.zeros(1, device=device).squeeze()
            else:
                loss_s = torch.zeros(1, device=device).squeeze()

        if torch.isnan(loss):
            raise RuntimeError(f"[{variant_name}] NaN loss at step {step}!")

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        train_log.append({
            "step": step + 1,
            "variant": variant_name,
            "loss_region": float(loss_r.item()),
            "loss_super":  float(loss_s.item()) if isinstance(loss_s, torch.Tensor) else 0.0,
            "loss_total":  float(loss.item()),
        })

        step += 1

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            vm, _, _ = eval_model(
                variant_name, model, None,
                val_data, tok_arr, reg_arr, n_regions, n_super,
                region_sizes_arr, vocab_known_count, device,
                args.batch_size * 4, args)
            model.train()

            val_score = vm.get("region_recall@8", float("nan"))
            tie_break = -vm.get("region_ce", float("inf"))
            score_key = (val_score, tie_break)

            elapsed = time.time() - t0
            print(f"[{variant_name}] step={step:5d}  "
                  f"loss={train_log[-1]['loss_total']:.4f}  "
                  f"recall@1={vm.get('region_acc@1',float('nan')):.4f}  "
                  f"recall@4={vm.get('region_recall@4',float('nan')):.4f}  "
                  f"recall@8={val_score:.4f}  "
                  f"recall@16={vm.get('region_recall@16',float('nan')):.4f}  "
                  f"cand_frac@8={vm.get('candidate_fraction@8',float('nan')):.4f}  "
                  f"{elapsed:.0f}s")

            eval_log.append({"step": step, "variant": variant_name, **vm})

            if val_score == val_score and val_score > best_score:
                best_score = val_score
                best_ckpt  = deepcopy(model.state_dict())
                best_vm    = deepcopy(vm)
                best_step  = step
                print(f"  → new best  recall@8={val_score:.5f}")

    # Save best checkpoint
    ckpt_path = None
    no_best   = (best_ckpt is None)
    if not no_best:
        ckpt_path = os.path.join(out_dir, f"best_{variant_name}.pt")
        torch.save({"state_dict": best_ckpt, "variant": variant_name,
                    "val_recall@8": best_score, "best_step": best_step}, ckpt_path)
        print(f"[{variant_name}] saved best: {ckpt_path}  recall@8={best_score:.5f}")
    else:
        print(f"[{variant_name}] no improving checkpoint found")

    return {
        "best_metrics":    best_vm or {},
        "best_step":       best_step,
        "no_improving_checkpoint_found": no_best,
        "checkpoint_path": ckpt_path,
        "train_log":       train_log,
        "eval_log":        eval_log,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════════

def write_report(all_best: dict, out_dir: str, args) -> str:
    def _v(name, k): return all_best.get(name, {}).get(k, float("nan"))
    def _ok(a, b):   return (a == a) and (b == b) and a > b + 0.02

    real_r8     = _v("real_region_router", "region_recall@8")
    real_r8_freq = _v("frequency_prior_baseline", "region_recall@8")
    real_r8_shuf = _v("shuffled_region_router", "region_recall@8")
    real_r8_rand = _v("random_region_router", "region_recall@8")
    real_frac8  = _v("real_region_router", "candidate_fraction@8")
    real_cov8   = _v("real_region_router", "gold_token_coverage@8")
    real_flop_r = _v("real_region_router", "theoretical_output_flop_reduction@8")
    real_r16    = _v("real_region_router", "region_recall@16")

    beats_freq  = _ok(real_r8, real_r8_freq)
    beats_shuf  = _ok(real_r8, real_r8_shuf)
    beats_rand  = _ok(real_r8, real_r8_rand)

    # Success criteria
    weak_success       = beats_freq and beats_shuf and (real_frac8 != real_frac8 or real_frac8 < 0.25)
    meaningful_success = (real_r8 == real_r8 and real_r8 >= 0.85
                          and (real_frac8 != real_frac8 or real_frac8 <= 0.15)
                          and beats_shuf and beats_rand)
    strong_success     = (real_r8 == real_r8 and real_r8 >= 0.88
                          and (real_frac8 != real_frac8 or real_frac8 <= 0.15)
                          and beats_freq and beats_shuf and beats_rand)

    if strong_success:
        recommendation = "PROCEED_TO_ROUTED_CANDIDATE_SOFTMAX"
    elif meaningful_success or weak_success:
        recommendation = "PARTIAL_GO_IMPROVE_ROUTER"
    else:
        recommendation = "DO_NOT_PROCEED"

    lines = [
        "# Phase 1A: Static Region Router — Report",
        "",
        f"**steps:** {args.steps}  |  **d_model:** {args.d_model}  |  "
        f"**n_layers:** {args.n_layers}  |  **n_heads:** {args.n_heads}  |  **seed:** {args.seed}",
        "", "---", "",
        "## Comparison Table",
        "",
        "Primary metric: `region_recall@8` and `candidate_fraction@8`.",
        "",
    ]

    variants_order = [
        "frequency_prior_baseline",
        "last_token_mlp_baseline",
        "mean_embedding_mlp_baseline",
        "real_region_router",
        "shuffled_region_router",
        "random_region_router",
    ]
    cols = ["variant", "region_acc@1", "region_recall@4", "region_recall@8", "region_recall@16",
            "candidate_fraction@8", "avg_candidate_set_size@8",
            "gold_token_coverage@8", "base_top5_any_in_C@8",
            "theoretical_output_flop_reduction@8"]

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for vn in variants_order:
        vm = all_best.get(vn, {})
        row_vals = [vn] + [_fmt(vm.get(c, float("nan"))) for c in cols[1:]]
        lines.append("| " + " | ".join(row_vals) + " |")

    lines += ["", "---", "", "## Coverage by K", ""]
    lines.append("| k | recall@k | avg_cand_size | cand_fraction | base_top5_any |")
    lines.append("|---|---|---|---|---|")
    real_vm = all_best.get("real_region_router", {})
    for k in _TOPK_LIST:
        lines.append(
            f"| {k} "
            f"| {_fmt(real_vm.get(f'region_recall@{k}', float('nan')))} "
            f"| {_fmt(real_vm.get(f'avg_candidate_set_size@{k}', float('nan')))} "
            f"| {_fmt(real_vm.get(f'candidate_fraction@{k}', float('nan')))} "
            f"| {_fmt(real_vm.get(f'base_top5_any_in_C@{k}', float('nan')))} |"
        )

    lines += ["", "---", "", "## Q&A", ""]

    def _q(n, q, ans, detail=""):
        lines.append(f"### Q{n}: {q}")
        lines.append(f"**{ans}**")
        if detail: lines.append(f"\n{detail}")
        lines.append("")

    _q(1, "Can a cheap attention router predict the gold-containing static region?",
       f"real recall@8={_fmt(real_r8)}",
       "YES — router exceeds random chance" if (real_r8 == real_r8 and real_r8 > 0.2) else "UNCLEAR")
    _q(2, "How good is top-k region recall?",
       f"@1={_fmt(_v('real_region_router','region_acc@1'))}  "
       f"@4={_fmt(_v('real_region_router','region_recall@4'))}  "
       f"@8={_fmt(real_r8)}  "
       f"@16={_fmt(real_r16)}")
    _q(3, "How many candidate tokens are selected by top-k regions?",
       f"@8: avg={_fmt(_v('real_region_router','avg_candidate_set_size@8'))}  "
       f"fraction={_fmt(real_frac8)}")
    _q(4, "Does real region routing beat frequency prior?",
       "YES" if beats_freq else "NO",
       f"real={_fmt(real_r8)}  freq_prior={_fmt(real_r8_freq)}")
    _q(5, "Does real region routing beat shuffled/random controls?",
       "YES" if (beats_shuf and beats_rand) else "PARTIAL" if (beats_shuf or beats_rand) else "NO",
       f"real={_fmt(real_r8)}  shuffled={_fmt(real_r8_shuf)}  random={_fmt(real_r8_rand)}")
    _q(6, "How much theoretical output-layer compute can be saved?",
       f"@8: {_fmt(real_flop_r)} flop reduction",
       f"full_lm_flops={50257*384:,}  selected@8≈{_fmt(_v('real_region_router','selected_lm_head_flops_estimate@8'))}")
    _q(7, "Is recall@8 high enough to proceed to routed candidate softmax?",
       "YES" if (real_r8 == real_r8 and real_r8 >= 0.85) else "BORDERLINE" if (real_r8 == real_r8 and real_r8 >= 0.75) else "NO",
       f"recall@8={_fmt(real_r8)}")
    _q(8, "What are the major failure slices?",
       "See slice_metrics.csv for per-slice breakdown.")

    lines += [
        "---", "",
        "## PHASE 1A STATIC REGION ROUTER VERDICT", "",
        "```",
    ]
    hdr = (f"  {'variant':<36}  acc@1   rec@4   rec@8   rec@16  "
           f"cfrac@8  cand@8     cov@8   b5any@8  flop_red@8")
    lines.append(hdr)
    lines.append("  " + "─" * len(hdr))
    for vn in variants_order:
        vm = all_best.get(vn, {})
        lines.append(
            f"  {vn:<36}"
            f"  {_fmt(vm.get('region_acc@1', float('nan'))):6}"
            f"  {_fmt(vm.get('region_recall@4', float('nan'))):6}"
            f"  {_fmt(vm.get('region_recall@8', float('nan'))):6}"
            f"  {_fmt(vm.get('region_recall@16', float('nan'))):6}"
            f"  {_fmt(vm.get('candidate_fraction@8', float('nan'))):7}"
            f"  {_fmt(vm.get('avg_candidate_set_size@8', float('nan'))):10}"
            f"  {_fmt(vm.get('gold_token_coverage@8', float('nan'))):6}"
            f"  {_fmt(vm.get('base_top5_any_in_C@8', float('nan'))):7}"
            f"  {_fmt(vm.get('theoretical_output_flop_reduction@8', float('nan')))}"
        )
    lines += ["", f"recommendation: {recommendation}", "```", ""]

    path = os.path.join(out_dir, "phase1A_static_region_router_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[save] {path}")
    return recommendation


def write_example_files(real_examples: List[dict], real_metrics: dict,
                        freq_metrics: dict, out_dir: str):
    def _md_example(ex: dict) -> str:
        lines = [
            f"- row_id={ex['row_id']}  token_offset={ex['token_offset']}",
            f"  gold_token_id={ex['gold_token_id']}  gold_region={ex['gold_region']}",
            f"  top8_regions={ex['top8_regions']}",
            f"  top8_probs={ex['top8_probs']}",
            f"  gold_covered@8={ex['gold_covered@8']}  cand_set_size@8={ex['cand_set_size@8']}",
            f"  base_top10={ex['base_top10']}",
            "",
        ]
        return "\n".join(lines)

    success_top1  = [e for e in real_examples if e["gold_covered@8"] and e["gold_region"] == e["top8_regions"][0]][:10]
    success_top8  = [e for e in real_examples if e["gold_covered@8"] and e["gold_region"] != e["top8_regions"][0]][:10]
    fail_miss     = [e for e in real_examples if not e["gold_covered@8"]][:10]

    for fname, title, exs in [
        ("examples_success_top1.md",    "Success: gold region predicted rank 1",    success_top1),
        ("examples_success_top8.md",    "Success: gold region in top-8 (not top-1)", success_top8),
        ("examples_fail_missed_region.md", "Failure: gold region not in top-8",     fail_miss),
        ("examples_real_beats_frequency.md", "Real router vs frequency prior",      real_examples[:10]),
    ]:
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            if exs:
                for e in exs:
                    f.write(_md_example(e))
            else:
                f.write("No examples in this category.\n")
        print(f"[save] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Capacity + Context-Length Ablation
# ══════════════════════════════════════════════════════════════════════════════

_ABLATION_VARIANT_CONFIGS: Dict[str, dict] = {
    "last_token_mlp":     {"model_type": "last_token_mlp",  "n_layers": 0},
    "mean_embedding_mlp": {"model_type": "mean_emb_mlp",    "n_layers": 0},
    "real_router_L1":     {"model_type": "transformer",     "n_layers": 1},
    "real_router_L2":     {"model_type": "transformer",     "n_layers": 2},
    "real_router_L4":     {"model_type": "transformer",     "n_layers": 4},
}

_ABLATION_VARIANT_ORDER = [
    "last_token_mlp",
    "mean_embedding_mlp",
    "real_router_L1",
    "real_router_L2",
    "real_router_L4",
]

_QUICK_VARIANTS      = ["last_token_mlp", "real_router_L1", "real_router_L2"]
_QUICK_CONTEXT_LENS  = [32, 128]


def _slice_data_context(data: dict, context_len: int) -> dict:
    avail_T = data["input_ids"].shape[1]
    if context_len > avail_T:
        raise ValueError(
            f"context_len={context_len} exceeds available sequence length T={avail_T}. "
            f"Use shorter context_len or shards with longer sequences. "
            f"Do NOT silently pad.")
    sliced = dict(data)
    sliced["input_ids"] = data["input_ids"][:, -context_len:]
    sliced["seq_len"]   = context_len
    return sliced


def _make_ablation_model(variant_name: str,
                          vocab_size: int,
                          n_regions: int,
                          n_super: int,
                          d_model: int,
                          n_heads: int,
                          d_ff: int,
                          dropout: float,
                          context_len: int,
                          use_super_aux: bool) -> nn.Module:
    cfg = _ABLATION_VARIANT_CONFIGS[variant_name]
    mtype = cfg["model_type"]
    nlayers = cfg["n_layers"]

    if mtype == "last_token_mlp":
        return LastTokenMLPBaseline(vocab_size, n_regions, d_model, dropout)
    elif mtype == "mean_emb_mlp":
        return MeanEmbeddingMLPBaseline(vocab_size, n_regions, d_model, dropout)
    elif mtype == "transformer":
        return StaticRegionRouter(
            vocab_size=vocab_size, n_regions=n_regions, n_super=n_super,
            d_model=d_model, n_layers=nlayers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, max_seq_len=context_len,
            use_super_aux=use_super_aux,
        )
    else:
        raise ValueError(f"Unknown model_type: {mtype}")


def _estimate_ablation_flops(model: nn.Module, context_len: int) -> int:
    if hasattr(model, "estimate_forward_flops"):
        return model.estimate_forward_flops(context_len)
    # MLP baselines: embedding lookup + two Linear layers
    d = model.tok_emb.embedding_dim
    n_r = model.mlp[-1].out_features
    return 2 * (d * d + d * n_r)


def run_capacity_context_ablation(args,
                                   train_data: dict,
                                   val_data: dict,
                                   tok_arr: np.ndarray,
                                   reg_arr: np.ndarray,
                                   n_regions: int,
                                   n_super: int,
                                   rsizes: np.ndarray,
                                   vocab_known_count: int,
                                   vocab_size: int,
                                   use_super_aux: bool,
                                   device: torch.device) -> List[dict]:
    """
    Run all (variant, context_len) combinations.
    Returns list of result dicts for aggregation.
    """
    avail_T = train_data["seq_len"]
    d_ff    = 4 * args.d_model
    base_d  = args.base_d_model
    full_lm_flops = vocab_known_count * base_d   # per spec

    # Parse run matrix
    context_lens: List[int] = sorted(set(
        int(c) for c in args.context_lens.split(",") if c.strip()))
    variants: List[str] = [v.strip() for v in args.variants.split(",") if v.strip()]

    # Validate context lengths
    for cl in context_lens:
        if cl > avail_T:
            raise ValueError(
                f"context_len={cl} > shard sequence length {avail_T}. "
                f"Shard does not have enough context. Aborting.")
    unknown_v = [v for v in variants if v not in _ABLATION_VARIANT_CONFIGS]
    if unknown_v:
        raise ValueError(f"Unknown variants: {unknown_v}. "
                         f"Valid: {list(_ABLATION_VARIANT_CONFIGS.keys())}")

    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)

    all_results: List[dict] = []
    total_runs = len(context_lens) * len(variants)
    run_idx    = 0

    for context_len in context_lens:
        train_sl = _slice_data_context(train_data, context_len)
        val_sl   = _slice_data_context(val_data,   context_len)

        for variant_name in variants:
            run_idx += 1
            run_label = f"{variant_name}_ctx{context_len}"
            run_dir   = os.path.join(out_root, run_label)
            os.makedirs(run_dir, exist_ok=True)

            cfg     = _ABLATION_VARIANT_CONFIGS[variant_name]
            n_layers= cfg["n_layers"]

            print(f"\n{'─'*60}")
            print(f"[ablation] run {run_idx}/{total_runs}: {run_label}")
            print(f"  variant={variant_name}  context_len={context_len}  n_layers={n_layers}"
                  f"  d_model={args.d_model}  n_heads={args.n_heads}")

            np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

            model = _make_ablation_model(
                variant_name, vocab_size, n_regions, n_super,
                args.d_model, args.n_heads, d_ff, args.dropout,
                context_len, use_super_aux).to(device)
            n_params       = model.count_params()
            router_flops   = _estimate_ablation_flops(model, context_len)
            print(f"  params={n_params:,}  router_flops_est={router_flops:,}")

            # Save per-run config
            run_cfg = {
                "variant": variant_name, "context_len": context_len,
                "n_layers": n_layers, "d_model": args.d_model,
                "n_heads": args.n_heads, "dropout": args.dropout,
                "params": n_params, "router_flops_est": router_flops,
                "steps": args.steps, "batch_size": args.batch_size,
                "lr": args.lr, "seed": args.seed,
            }
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(run_cfg, f, indent=2)

            # Train
            t_train_start = time.time()
            result = train_variant(
                variant_name, model, train_sl, val_sl,
                tok_arr, reg_arr, n_regions, n_super,
                rsizes, vocab_known_count, args, device, run_dir)
            train_seconds = time.time() - t_train_start

            # Load best checkpoint for final eval
            ckpt_path = result["checkpoint_path"]
            if ckpt_path and os.path.isfile(ckpt_path):
                ck = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ck["state_dict"])

            # Final eval
            t_eval_start = time.time()
            model.eval()
            vm, slices, _ = eval_model(
                variant_name, model, None,
                val_sl, tok_arr, reg_arr, n_regions, n_super,
                rsizes, vocab_known_count, device, args.batch_size * 4, args)
            eval_seconds = time.time() - t_eval_start

            # Save per-run CSVs
            _wcsv(os.path.join(run_dir, "train_log.csv"), result["train_log"])
            _wcsv(os.path.join(run_dir, "eval_log.csv"),  result["eval_log"])
            _wcsv(os.path.join(run_dir, "slice_metrics.csv"), slices)
            with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
                bm_out = {k: (v if not isinstance(v, float) or v == v else None)
                          for k, v in vm.items()}
                json.dump(bm_out, f, indent=2)

            # Compute ablation-specific efficiency metrics
            avg_cs8   = vm.get("avg_candidate_set_size@8", float("nan"))
            sel_fl8   = avg_cs8 * base_d if avg_cs8 == avg_cs8 else float("nan")
            out_red8  = 1.0 - _safediv(sel_fl8, full_lm_flops) if sel_fl8 == sel_fl8 else float("nan")
            total_c8  = (router_flops + sel_fl8) if sel_fl8 == sel_fl8 else float("nan")
            speedup8  = _safediv(full_lm_flops, total_c8) if total_c8 == total_c8 else float("nan")

            row = {
                "variant":       variant_name,
                "context_len":   context_len,
                "n_layers":      n_layers,
                "d_model":       args.d_model,
                "n_heads":       args.n_heads,
                "params":        n_params,
                "router_flops_est":          router_flops,
                "train_seconds":             round(train_seconds, 1),
                "eval_seconds":              round(eval_seconds, 1),
                "best_step":                 result.get("best_step"),
                "no_improving_checkpoint":   result.get("no_improving_checkpoint_found", False),
                # region metrics
                "region_ce":                 vm.get("region_ce",             float("nan")),
                "region_acc@1":              vm.get("region_acc@1",          float("nan")),
                "region_recall@2":           vm.get("region_recall@2",       float("nan")),
                "region_recall@4":           vm.get("region_recall@4",       float("nan")),
                "region_recall@8":           vm.get("region_recall@8",       float("nan")),
                "region_recall@16":          vm.get("region_recall@16",      float("nan")),
                "region_recall@32":          vm.get("region_recall@32",      float("nan")),
                "mean_rank_gold_region":     vm.get("mean_rank_gold_region", float("nan")),
                "median_rank_gold_region":   vm.get("median_rank_gold_region", float("nan")),
                "region_entropy":            vm.get("region_entropy",        float("nan")),
                "region_margin_top1_top2":   vm.get("region_margin_top1_top2", float("nan")),
                # candidate coverage
                "candidate_fraction@4":      vm.get("candidate_fraction@4",  float("nan")),
                "candidate_fraction@8":      vm.get("candidate_fraction@8",  float("nan")),
                "candidate_fraction@16":     vm.get("candidate_fraction@16", float("nan")),
                "avg_candidate_set_size@4":  vm.get("avg_candidate_set_size@4", float("nan")),
                "avg_candidate_set_size@8":  avg_cs8,
                "avg_candidate_set_size@16": vm.get("avg_candidate_set_size@16", float("nan")),
                "gold_token_coverage@4":     vm.get("gold_token_coverage@4", float("nan")),
                "gold_token_coverage@8":     vm.get("gold_token_coverage@8", float("nan")),
                "gold_token_coverage@16":    vm.get("gold_token_coverage@16", float("nan")),
                "base_top1_in_C@8":          vm.get("base_top1_in_C@8",     float("nan")),
                "base_top5_any_in_C@8":      vm.get("base_top5_any_in_C@8", float("nan")),
                "base_top10_any_in_C@8":     vm.get("base_top10_any_in_C@8", float("nan")),
                # efficiency (clearly labeled as estimates)
                "full_lm_head_flops_estimate":          full_lm_flops,
                "selected_lm_head_flops_estimate@8":    sel_fl8,
                "output_flop_reduction@8":              out_red8,
                "total_estimated_cost@8":               total_c8,
                "estimated_speedup_vs_full_lm_head@8":  speedup8,
            }
            all_results.append(row)

            r8  = row["region_recall@8"]
            r16 = row["region_recall@16"]
            sp8 = row["estimated_speedup_vs_full_lm_head@8"]
            print(f"  [done] recall@8={_fmt(r8)}  recall@16={_fmt(r16)}"
                  f"  cand_frac@8={_fmt(row['candidate_fraction@8'])}"
                  f"  est_speedup@8={_fmt(sp8)}"
                  f"  train={train_seconds:.0f}s")

    return all_results


def write_ablation_report(results: List[dict],
                           out_dir: str,
                           args) -> str:
    """Write comparison CSVs, markdown table, and analysis report."""
    if not results:
        return "DO_NOT_PROCEED"

    # ── Sort order for tables ────────────────────────────────────────────────
    def _sort_key(r):
        ctx  = r["context_len"]
        vidx = _ABLATION_VARIANT_ORDER.index(r["variant"]) if r["variant"] in _ABLATION_VARIANT_ORDER else 99
        return (ctx, vidx)
    results_sorted = sorted(results, key=_sort_key)

    # ── Main comparison CSV ──────────────────────────────────────────────────
    comp_cols = [
        "variant", "context_len", "params", "router_flops_est",
        "region_acc@1", "region_recall@4", "region_recall@8", "region_recall@16", "region_recall@32",
        "candidate_fraction@8", "avg_candidate_set_size@8",
        "base_top5_any_in_C@8",
        "output_flop_reduction@8", "total_estimated_cost@8",
        "estimated_speedup_vs_full_lm_head@8",
        "train_seconds", "best_step",
    ]
    _wcsv(os.path.join(out_dir, "capacity_context_ablation.csv"),
          [{c: r.get(c, float("nan")) for c in comp_cols} for r in results_sorted])

    # ── Best by context length ───────────────────────────────────────────────
    ctx_best: Dict[int, dict] = {}
    for r in results:
        cl  = r["context_len"]
        r8  = r.get("region_recall@8", float("nan"))
        if cl not in ctx_best or (r8 == r8 and r8 > ctx_best[cl].get("region_recall@8", float("nan"))):
            ctx_best[cl] = r
    _wcsv(os.path.join(out_dir, "best_by_context.csv"),
          [{c: ctx_best[cl].get(c, float("nan")) for c in comp_cols} for cl in sorted(ctx_best)])

    # ── Best by compute-recall tradeoff ──────────────────────────────────────
    def _tradeoff_score(r):
        sp = r.get("estimated_speedup_vs_full_lm_head@8", float("nan"))
        r8 = r.get("region_recall@8", float("nan"))
        if sp != sp or r8 != r8:
            return float("-inf")
        return sp * r8
    results_by_tradeoff = sorted(results, key=_tradeoff_score, reverse=True)
    _wcsv(os.path.join(out_dir, "best_by_compute_tradeoff.csv"),
          [{c: r.get(c, float("nan")) for c in comp_cols} for r in results_by_tradeoff[:10]])

    # ── Derive key values for Q&A ────────────────────────────────────────────
    def _get(variant, ctx, key, default=float("nan")):
        for r in results:
            if r["variant"] == variant and r["context_len"] == ctx:
                return r.get(key, default)
        return default

    context_lens = sorted(set(r["context_len"] for r in results))
    variants_used = [v for v in _ABLATION_VARIANT_ORDER if any(r["variant"] == v for r in results)]

    # Q1/Q2: Context length effect on recall@8 for best transformer
    # Use real_router_L2 as reference (or L1/L4 if L2 not present)
    ref_v = "real_router_L2" if "real_router_L2" in variants_used else (
            "real_router_L1" if "real_router_L1" in variants_used else variants_used[-1])
    ctx_recall = {cl: _get(ref_v, cl, "region_recall@8") for cl in context_lens}
    ctx_recall16 = {cl: _get(ref_v, cl, "region_recall@16") for cl in context_lens}

    # Context improvement
    ctx_vals = [ctx_recall[cl] for cl in context_lens if ctx_recall[cl] == ctx_recall[cl]]
    max_ctx_gain = max(ctx_vals) - min(ctx_vals) if len(ctx_vals) >= 2 else float("nan")

    # Q3-Q5: Layer effect
    best_ctx = context_lens[-1]  # largest context as reference
    r8_lt  = _get("last_token_mlp",  best_ctx, "region_recall@8")
    r8_L1  = _get("real_router_L1",  best_ctx, "region_recall@8")
    r8_L2  = _get("real_router_L2",  best_ctx, "region_recall@8")
    r8_L4  = _get("real_router_L4",  best_ctx, "region_recall@8")
    gain_L1_vs_lt = r8_L1 - r8_lt if (r8_L1 == r8_L1 and r8_lt == r8_lt) else float("nan")
    gain_L2_vs_L1 = r8_L2 - r8_L1 if (r8_L2 == r8_L2 and r8_L1 == r8_L1) else float("nan")
    gain_L4_vs_L2 = r8_L4 - r8_L2 if (r8_L4 == r8_L4 and r8_L2 == r8_L2) else float("nan")

    # Q6: Best model overall by recall@8
    best_r8_row = max(results, key=lambda r: r.get("region_recall@8", float("-inf")))
    best_variant_by_recall = best_r8_row["variant"]
    best_recall8 = best_r8_row["region_recall@8"]
    best_recall16 = best_r8_row.get("region_recall@16", float("nan"))

    # Q7: Best compute-recall tradeoff
    best_tradeoff_row = results_by_tradeoff[0] if results_by_tradeoff else {}
    best_variant_by_tradeoff = best_tradeoff_row.get("variant", "?")
    best_tradeoff_ctx = best_tradeoff_row.get("context_len", "?")

    # Q8: Is longer context worth the compute?
    if len(context_lens) >= 2:
        short_ctx  = context_lens[0]
        long_ctx   = context_lens[-1]
        r8_short   = ctx_recall.get(short_ctx, float("nan"))
        r8_long    = ctx_recall.get(long_ctx,  float("nan"))
        ctx_marginal_gain = (r8_long - r8_short) if (r8_long == r8_long and r8_short == r8_short) else float("nan")
        sp8_short  = _get(ref_v, short_ctx, "estimated_speedup_vs_full_lm_head@8")
        sp8_long   = _get(ref_v, long_ctx,  "estimated_speedup_vs_full_lm_head@8")
        ctx_worth_it = (ctx_marginal_gain == ctx_marginal_gain and
                        ctx_marginal_gain > 0.02 and
                        (sp8_long == sp8_long and sp8_long > 1.0))
    else:
        ctx_marginal_gain = float("nan"); ctx_worth_it = False
        sp8_long = float("nan"); sp8_short = float("nan")

    # ── Recommendation logic ─────────────────────────────────────────────────
    def _nn(v): return v == v  # not nan

    # 1. If best recall@8 >= 0.85 and recall@16 >= 0.92 → proceed
    if _nn(best_recall8) and best_recall8 >= 0.85 and _nn(best_recall16) and best_recall16 >= 0.92:
        recommendation = "PROCEED_TO_ROUTED_CANDIDATE_SOFTMAX"
    elif _nn(best_recall8) and best_recall8 < 0.50:
        recommendation = "DO_NOT_PROCEED"
    elif (_nn(r8_L4) and _nn(r8_L2) and r8_L4 - r8_L2 >= 0.02
          and _nn(sp8_long) and sp8_long > 1.0):
        recommendation = "USE_L4_ROUTER"
    elif (_nn(r8_L2) and _nn(r8_L1) and
          not (_nn(r8_L1) and r8_L1 >= 0.95 * r8_L2)):
        recommendation = "USE_L2_ROUTER"
    elif _nn(r8_L1) and _nn(r8_L2) and r8_L1 >= 0.95 * r8_L2:
        recommendation = "USE_L1_ROUTER"
    elif (_nn(ctx_marginal_gain) and ctx_marginal_gain > 0.03
          and long_ctx == max(context_lens)):
        recommendation = "NEED_LONGER_CONTEXT_SHARDS"
    elif _nn(r8_L1) and r8_L1 < 0.70:
        # L1 router barely beats baseline
        if _nn(r8_lt) and _nn(r8_L1) and r8_L1 - r8_lt < 0.02:
            recommendation = "USE_LAST_TOKEN_BASELINE"
        else:
            recommendation = "PARTIAL_GO_IMPROVE_ROUTER"
    else:
        recommendation = "PARTIAL_GO_IMPROVE_ROUTER"

    # ── Markdown comparison table ────────────────────────────────────────────
    def _col(r, key, w=8):
        v = r.get(key, float("nan"))
        return _fmt(v)[:w] if _fmt(v) != "nan" else "nan"

    md_rows = [
        "| variant | ctx | params | r@4 | r@8 | r@16 | cand_frac@8 | avg_cands@8 | est_speedup@8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results_sorted:
        md_rows.append(
            f"| {r['variant']} "
            f"| {r['context_len']} "
            f"| {r.get('params',0):,} "
            f"| {_col(r,'region_recall@4')} "
            f"| {_col(r,'region_recall@8')} "
            f"| {_col(r,'region_recall@16')} "
            f"| {_col(r,'candidate_fraction@8')} "
            f"| {_col(r,'avg_candidate_set_size@8',10)} "
            f"| {_col(r,'estimated_speedup_vs_full_lm_head@8')} |"
        )

    # ── ASCII sparkline for recall@8 by context_len ──────────────────────────
    def _sparkline(vals, width=40):
        vv = [v for v in vals if v == v]
        if not vv: return ""
        lo, hi = min(vv), max(vv)
        span = hi - lo if hi > lo else 1.0
        bars = " _.-^*"
        out_parts = []
        for v in vals:
            if v != v:
                out_parts.append(" ")
            else:
                idx = int((v - lo) / span * (len(bars) - 1))
                out_parts.append(bars[min(idx, len(bars)-1)])
        return "".join(out_parts)

    ctx_lines = []
    for vn in variants_used:
        recalls = [_get(vn, cl, "region_recall@8") for cl in context_lens]
        spark   = _sparkline(recalls)
        ctx_lines.append(f"  {vn:<28} ctx={context_lens} : {spark} "
                         f"  values={[_fmt(v) for v in recalls]}")

    # ── Report text ──────────────────────────────────────────────────────────
    lines = [
        "# Phase 1A Router Capacity + Context-Length Ablation Report",
        "",
        f"**steps:** {args.steps}  |  **d_model:** {args.d_model}  |  "
        f"**n_heads:** {args.n_heads}  |  **seed:** {args.seed}",
        f"**context_lens tested:** {context_lens}",
        f"**variants tested:** {variants_used}",
        "", "---", "",
        "## Comparison Table",
        "",
        "*(All efficiency numbers are theoretical estimates, not measured end-to-end speedups.)*",
        "",
        *md_rows,
        "", "---", "",
        "## Recall@8 by Context Length",
        "",
        "*(ASCII sparkline: left=short ctx, right=long ctx, higher char = higher recall)*",
        "",
        *ctx_lines,
        "", "---", "",
        "## Q&A", "",
    ]

    def _q(n, q, ans, detail=""):
        lines.append(f"### Q{n}: {q}")
        lines.append(f"**{ans}**")
        if detail: lines.append(f"\n{detail}")
        lines.append("")

    _q(1, "How much does context length improve recall@8?",
       f"Max gain = {_fmt(max_ctx_gain)} (across ctx={context_lens}, model={ref_v})",
       f"recall@8 by context: " + "  ".join(f"ctx{cl}={_fmt(ctx_recall.get(cl))}" for cl in context_lens))
    _q(2, "Does recall saturate at 32, 64, or 128 tokens?",
       f"See values above. Saturation if gain from ctx64→ctx128 < 0.005.",
       f"ctx_marginal_gain ({context_lens[0]}→{context_lens[-1]}) = {_fmt(ctx_marginal_gain)}")
    _q(3, "How much does L1 beat last-token MLP?",
       f"L1 recall@8={_fmt(r8_L1)}  last_token={_fmt(r8_lt)}  gain={_fmt(gain_L1_vs_lt)}",
       f"(at context_len={best_ctx})")
    _q(4, "How much does L2 beat L1?",
       f"L2 recall@8={_fmt(r8_L2)}  L1={_fmt(r8_L1)}  gain={_fmt(gain_L2_vs_L1)}",
       f"(at context_len={best_ctx})")
    _q(5, "How much does L4 beat L2?",
       f"L4 recall@8={_fmt(r8_L4)}  L2={_fmt(r8_L2)}  gain={_fmt(gain_L4_vs_L2)}",
       f"(at context_len={best_ctx})")
    _q(6, "Which model has the best recall@8?",
       f"{best_variant_by_recall} at ctx={best_r8_row.get('context_len','?')} "
       f"(recall@8={_fmt(best_recall8)}  recall@16={_fmt(best_recall16)})")
    _q(7, "Which model has the best recall-per-compute tradeoff?",
       f"{best_variant_by_tradeoff} at ctx={best_tradeoff_ctx} "
       f"(speedup*recall score={_fmt(_tradeoff_score(best_tradeoff_row))})",
       f"speedup@8={_fmt(best_tradeoff_row.get('estimated_speedup_vs_full_lm_head@8'))}"
       f"  recall@8={_fmt(best_tradeoff_row.get('region_recall@8'))}")
    _q(8, "Is longer context worth the extra attention compute?",
       f"{'YES' if ctx_worth_it else 'MARGINAL' if (_nn(ctx_marginal_gain) and ctx_marginal_gain > 0.01) else 'NO'}",
       f"recall gain ({context_lens[0]}→{context_lens[-1]}) = {_fmt(ctx_marginal_gain)}  "
       f"speedup@8 short={_fmt(sp8_short)} long={_fmt(sp8_long)}")
    _q(9, "Should Phase 1 continue with last-token, L1, L2, or L4 router?",
       f"Based on tradeoff: {best_variant_by_tradeoff} at ctx={best_tradeoff_ctx}",
       f"L1 vs L2 gap={_fmt(gain_L2_vs_L1)}  L2 vs L4 gap={_fmt(gain_L4_vs_L2)}")
    _q(10, "Should the next experiment be routed candidate softmax or router improvement?",
       "PROCEED" if recommendation == "PROCEED_TO_ROUTED_CANDIDATE_SOFTMAX" else "IMPROVE_ROUTER",
       f"Best recall@8={_fmt(best_recall8)}  best recall@16={_fmt(best_recall16)}")

    # Verdict block
    lines += [
        "---", "",
        "## PHASE 1A ROUTER CAPACITY + CONTEXT-LENGTH ABLATION VERDICT", "",
        "```",
        f"  {'variant':<28}  ctx   params    r@4     r@8     r@16    cand_f@8  speedup@8",
        "  " + "─" * 90,
    ]
    for r in results_sorted:
        lines.append(
            f"  {r['variant']:<28}  {r['context_len']:3d}"
            f"  {r.get('params',0):8,}"
            f"  {_fmt(r.get('region_recall@4',   float('nan'))):6}"
            f"  {_fmt(r.get('region_recall@8',   float('nan'))):6}"
            f"  {_fmt(r.get('region_recall@16',  float('nan'))):6}"
            f"  {_fmt(r.get('candidate_fraction@8', float('nan'))):8}"
            f"  {_fmt(r.get('estimated_speedup_vs_full_lm_head@8', float('nan')))}"
        )
    lines += [
        "",
        f"Best recall@8:           {best_variant_by_recall}  ({_fmt(best_recall8)})",
        f"Best recall@16:          {best_r8_row.get('variant','?')}  ({_fmt(best_recall16)})",
        f"Best recall-per-compute: {best_variant_by_tradeoff}  ctx={best_tradeoff_ctx}",
        f"Best context length:     ctx{context_lens[-1] if ctx_recall[context_lens[-1]] == ctx_recall[context_lens[-1]] else '?'}",
        f"Recommended router:      {recommendation}",
        f"Final recommendation:    {recommendation}",
        "```", "",
    ]

    # Save report
    rpt_path = os.path.join(out_dir, "capacity_context_ablation_report.md")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[save] {rpt_path}")

    # Save markdown table separately
    md_path = os.path.join(out_dir, "capacity_context_ablation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 1A Router Capacity + Context-Length Ablation\n\n")
        f.write("*(Efficiency numbers are theoretical estimates only.)*\n\n")
        f.write("\n".join(md_rows))
        f.write("\n")
    print(f"[save] {md_path}")

    return recommendation


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 1A: Static Region Router")
    p.add_argument("--train_dir",       required=True)
    p.add_argument("--val_dir",         required=True)
    p.add_argument("--token_to_region", required=True)
    p.add_argument("--super_map",       default=None)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--d_model",         type=int,   default=256)
    p.add_argument("--n_layers",        type=int,   default=2)
    p.add_argument("--n_heads",         type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--max_seq_len",     type=int,   default=128)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--steps",           type=int,   default=10000)
    p.add_argument("--eval_every",      type=int,   default=500)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--lambda_super",    type=float, default=0.2)
    p.add_argument("--no_super_aux",    action="store_true")
    p.add_argument("--max_train_rows",  type=int,   default=None)
    p.add_argument("--max_val_rows",    type=int,   default=None)
    p.add_argument("--base_d_model",    type=int,   default=384,
                   help="Base LM hidden size for FLOP estimates")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--amp",             action="store_true")
    # ── Ablation mode ─────────────────────────────────────────────────────────
    p.add_argument("--run_capacity_context_ablation", action="store_true",
                   help="Run the capacity + context-length ablation matrix")
    p.add_argument("--context_lens",    type=str,   default="16,32,64,128",
                   help="Comma-separated context lengths for ablation")
    p.add_argument("--variants",        type=str,
                   default="last_token_mlp,mean_embedding_mlp,real_router_L1,real_router_L2,real_router_L4",
                   help="Comma-separated variant names for ablation")
    p.add_argument("--quick",           action="store_true",
                   help="Run quick ablation: context_lens=32,128 and L1/L2 only")
    args = p.parse_args()

    if args.quick and args.run_capacity_context_ablation:
        args.context_lens = ",".join(str(c) for c in _QUICK_CONTEXT_LENS)
        args.variants      = ",".join(_QUICK_VARIANTS)

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    t0_global = time.time()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[step 1] Loading shards...")
    train_data = load_shards(args.train_dir, args.max_train_rows, "train")
    val_data   = load_shards(args.val_dir,   args.max_val_rows,   "val")

    seq_len    = train_data["seq_len"]
    args.max_seq_len = max(args.max_seq_len, seq_len)
    vocab_size = int(train_data["input_ids"].max()) + 1
    vocab_size = max(vocab_size, 50257)
    args.__dict__["vocab_size_full"] = vocab_size
    print(f"[data] vocab_size={vocab_size}  seq_len={seq_len}  "
          f"train={train_data['n_rows']:,}  val={val_data['n_rows']:,}")

    # ── Region maps ────────────────────────────────────────────────────────────
    print("\n[step 2] Loading region maps...")
    tok_arr_real, reg_arr, n_regions, n_super = load_region_maps(
        args.token_to_region, args.super_map)
    use_super = not args.no_super_aux and n_super > 1

    # ── Ablation branch ───────────────────────────────────────────────────────
    if args.run_capacity_context_ablation:
        r2t_real  = build_region_to_tokens(tok_arr_real, n_regions)
        rsizes_real = build_region_cumsize(r2t_real)
        vocab_known_count = int((tok_arr_real < n_regions).sum())
        print(f"[ablation] vocab_known_count={vocab_known_count:,}")
        print(f"[ablation] context_lens={args.context_lens}  variants={args.variants}")

        ablation_results = run_capacity_context_ablation(
            args, train_data, val_data,
            tok_arr_real, reg_arr, n_regions, n_super,
            rsizes_real, vocab_known_count, vocab_size, use_super, device)

        recommendation = write_ablation_report(ablation_results, args.output_dir, args)

        # ── Final console output ──────────────────────────────────────────────
        elapsed = time.time() - t0_global
        print(f"\n{'='*70}")
        print(f" PHASE 1A ROUTER CAPACITY + CONTEXT LENGTH ABLATION  ({elapsed/60:.1f} min)")
        print(f"{'='*70}")
        print(f"\n| {'variant':<28} | {'ctx':>4} | {'params':>9} | {'r@4':>7} | "
              f"{'r@8':>7} | {'r@16':>7} | {'cand_frac@8':>11} | {'avg_cands@8':>11} | {'est_speedup@8':>13} |")
        print(f"|{'-'*30}|{'-'*6}|{'-'*11}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*13}|{'-'*13}|{'-'*15}|")
        order_fn = lambda r: (r["context_len"],
                              _ABLATION_VARIANT_ORDER.index(r["variant"])
                              if r["variant"] in _ABLATION_VARIANT_ORDER else 99)
        for r in sorted(ablation_results, key=order_fn):
            print(f"| {r['variant']:<28} | {r['context_len']:>4} | {r.get('params',0):>9,} "
                  f"| {_fmt(r.get('region_recall@4',  float('nan'))):>7} "
                  f"| {_fmt(r.get('region_recall@8',  float('nan'))):>7} "
                  f"| {_fmt(r.get('region_recall@16', float('nan'))):>7} "
                  f"| {_fmt(r.get('candidate_fraction@8', float('nan'))):>11} "
                  f"| {_fmt(r.get('avg_candidate_set_size@8', float('nan'))):>11} "
                  f"| {_fmt(r.get('estimated_speedup_vs_full_lm_head@8', float('nan'))):>13} |")
        best_r8_row  = max(ablation_results, key=lambda r: r.get("region_recall@8", float("-inf")))
        best_r16_row = max(ablation_results, key=lambda r: r.get("region_recall@16", float("-inf")))
        def _tradeoff(r):
            sp = r.get("estimated_speedup_vs_full_lm_head@8", float("nan"))
            r8 = r.get("region_recall@8", float("nan"))
            return sp * r8 if (sp == sp and r8 == r8) else float("-inf")
        best_td_row = max(ablation_results, key=_tradeoff)
        print(f"\nBest recall@8:           {best_r8_row['variant']}  ctx={best_r8_row['context_len']}"
              f"  ({_fmt(best_r8_row.get('region_recall@8'))})")
        print(f"Best recall@16:          {best_r16_row['variant']}  ctx={best_r16_row['context_len']}"
              f"  ({_fmt(best_r16_row.get('region_recall@16'))})")
        print(f"Best recall-per-compute: {best_td_row['variant']}  ctx={best_td_row['context_len']}")
        print(f"Best context length:     ctx{best_r8_row['context_len']}")
        print(f"Recommended router:      {recommendation}")
        print(f"Final recommendation:    {recommendation}")
        print(f"{'='*70}")
        print(f"\nKey outputs:")
        print(f"  {args.output_dir}/capacity_context_ablation.csv")
        print(f"  {args.output_dir}/capacity_context_ablation.md")
        print(f"  {args.output_dir}/capacity_context_ablation_report.md")
        return

    # ── Control maps ───────────────────────────────────────────────────────────
    print("\n[step 3] Building control maps...")
    tok_arr_shuf = make_shuffled_tok_arr(tok_arr_real, n_regions, args.seed)
    tok_arr_rand = make_random_tok_arr(tok_arr_real,  n_regions, args.seed)

    # Save control maps and size stats
    sizes_real = region_sizes(tok_arr_real, n_regions)
    sizes_shuf = region_sizes(tok_arr_shuf, n_regions)
    sizes_rand = region_sizes(tok_arr_rand, n_regions)

    def _assert_sizes_match(a, b, name):
        if sorted(a) != sorted(b):
            raise AssertionError(f"Region size histogram mismatch for {name}!")
        print(f"  [OK] {name} region size histogram matches real")

    _assert_sizes_match(sizes_real, sizes_shuf, "shuffled")
    _assert_sizes_match(sizes_real, sizes_rand, "random")

    for fname, data_j in [
        ("real_region_sizes.json",     {"sizes": sizes_real, "n_regions": n_regions}),
        ("shuffled_region_sizes.json", {"sizes": sizes_shuf}),
        ("random_region_sizes.json",   {"sizes": sizes_rand}),
    ]:
        with open(os.path.join(args.output_dir, fname), "w") as f:
            json.dump(data_j, f)
        print(f"[save] {os.path.join(args.output_dir, fname)}")

    save_tok_arr_as_json(tok_arr_shuf, n_regions,
                         os.path.join(args.output_dir, "shuffled_token_to_region.json"))
    save_tok_arr_as_json(tok_arr_rand, n_regions,
                         os.path.join(args.output_dir, "random_token_to_region.json"))

    # Region frequency stats
    freq_region_stat_rows = []
    for r in range(n_regions):
        freq_region_stat_rows.append({"region": r, "size": sizes_real[r]})
    _wcsv(os.path.join(args.output_dir, "region_frequency_stats.csv"), freq_region_stat_rows)

    # Region → tokens lookup
    r2t_real = build_region_to_tokens(tok_arr_real, n_regions)
    r2t_shuf = build_region_to_tokens(tok_arr_shuf, n_regions)
    r2t_rand = build_region_to_tokens(tok_arr_rand, n_regions)

    rsizes_real = build_region_cumsize(r2t_real)
    rsizes_shuf = build_region_cumsize(r2t_shuf)
    rsizes_rand = build_region_cumsize(r2t_rand)

    vocab_known_count = int((tok_arr_real < n_regions).sum())
    print(f"[maps] vocab_known_count={vocab_known_count:,}")

    # ── Frequency prior ────────────────────────────────────────────────────────
    print("\n[step 4] Computing frequency prior...")
    freq_counts, freq_order = compute_frequency_prior(
        train_data["gold"].astype(np.int64), tok_arr_real, n_regions)

    # ── Save config ────────────────────────────────────────────────────────────
    config = {
        "d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
        "dropout": args.dropout, "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size, "steps": args.steps, "eval_every": args.eval_every,
        "lr": args.lr, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "lambda_super": args.lambda_super, "use_super_aux": use_super,
        "seed": args.seed, "amp": args.amp,
        "vocab_size": vocab_size, "seq_len": seq_len,
        "n_regions": n_regions, "n_super": n_super,
        "vocab_known_count": vocab_known_count,
        "base_d_model": args.base_d_model,
        "token_to_region": args.token_to_region,
        "super_map": args.super_map,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[save] {os.path.join(args.output_dir, 'config.json')}")

    d_ff = 4 * args.d_model

    # ══════════════════════════════════════════════════════════════════════════
    # Define all variants to train
    # ══════════════════════════════════════════════════════════════════════════

    router_variants = [
        ("real_region_router",     tok_arr_real, reg_arr, rsizes_real, n_regions, n_super),
        ("shuffled_region_router", tok_arr_shuf, reg_arr, rsizes_shuf, n_regions, n_super),
        ("random_region_router",   tok_arr_rand, reg_arr, rsizes_rand, n_regions, n_super),
    ]

    all_train_log: List[dict] = []
    all_eval_log:  List[dict] = []
    all_slice_rows: List[dict] = []
    all_best: Dict[str, dict] = {}

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Frequency prior baseline
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[step 5] Evaluating frequency prior baseline...")
    freq_vm, freq_slices, freq_examples = eval_model(
        "frequency_prior_baseline", None, freq_order,
        val_data, tok_arr_real, reg_arr, n_regions, n_super,
        rsizes_real, vocab_known_count, device, args.batch_size * 4, args)
    all_best["frequency_prior_baseline"] = freq_vm
    all_slice_rows.extend(freq_slices)
    print(f"  recall@8={freq_vm.get('region_recall@8', float('nan')):.4f}  "
          f"cand_frac@8={freq_vm.get('candidate_fraction@8', float('nan')):.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6: Small MLP baselines
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[step 6] Training LastToken MLP baseline...")
    last_tok_model = LastTokenMLPBaseline(vocab_size, n_regions, args.d_model, args.dropout).to(device)
    lt_result = train_variant(
        "last_token_mlp_baseline", last_tok_model,
        train_data, val_data, tok_arr_real, reg_arr, n_regions, n_super,
        rsizes_real, vocab_known_count, args, device, args.output_dir)
    if lt_result["best_metrics"]:
        all_best["last_token_mlp_baseline"] = lt_result["best_metrics"]
    else:
        last_tok_model.eval()
        vm_lt, sl_lt, _ = eval_model(
            "last_token_mlp_baseline", last_tok_model, None,
            val_data, tok_arr_real, reg_arr, n_regions, n_super,
            rsizes_real, vocab_known_count, device, args.batch_size * 4, args)
        all_best["last_token_mlp_baseline"] = vm_lt
    all_train_log.extend(lt_result["train_log"])
    all_eval_log.extend(lt_result["eval_log"])
    all_slice_rows.extend([s for s in lt_result["eval_log"][-1:]])  # placeholder

    print("\n[step 6b] Training MeanEmbedding MLP baseline...")
    mean_emb_model = MeanEmbeddingMLPBaseline(vocab_size, n_regions, args.d_model, args.dropout).to(device)
    me_result = train_variant(
        "mean_embedding_mlp_baseline", mean_emb_model,
        train_data, val_data, tok_arr_real, reg_arr, n_regions, n_super,
        rsizes_real, vocab_known_count, args, device, args.output_dir)
    if me_result["best_metrics"]:
        all_best["mean_embedding_mlp_baseline"] = me_result["best_metrics"]
    else:
        mean_emb_model.eval()
        vm_me, sl_me, _ = eval_model(
            "mean_embedding_mlp_baseline", mean_emb_model, None,
            val_data, tok_arr_real, reg_arr, n_regions, n_super,
            rsizes_real, vocab_known_count, device, args.batch_size * 4, args)
        all_best["mean_embedding_mlp_baseline"] = vm_me
    all_train_log.extend(me_result["train_log"])
    all_eval_log.extend(me_result["eval_log"])

    # ══════════════════════════════════════════════════════════════════════════
    # Step 7: Transformer router variants
    # ══════════════════════════════════════════════════════════════════════════
    for vi, (vname, tok_arr_v, reg_arr_v, rsizes_v, n_reg_v, n_sup_v) in enumerate(router_variants):
        print(f"\n[step 7.{vi+1}] Training {vname}...")
        model = StaticRegionRouter(
            vocab_size   = vocab_size,
            n_regions    = n_reg_v,
            n_super      = n_sup_v,
            d_model      = args.d_model,
            n_layers     = args.n_layers,
            n_heads      = args.n_heads,
            d_ff         = d_ff,
            dropout      = args.dropout,
            max_seq_len  = args.max_seq_len,
            use_super_aux = use_super,
        ).to(device)
        print(f"  params={model.count_params():,}")

        result = train_variant(
            vname, model,
            train_data, val_data, tok_arr_v, reg_arr_v, n_reg_v, n_sup_v,
            rsizes_v, vocab_known_count, args, device, args.output_dir)

        all_train_log.extend(result["train_log"])
        all_eval_log.extend(result["eval_log"])

        # Final slice eval using best checkpoint
        ckpt_path = result["checkpoint_path"]
        if ckpt_path and os.path.isfile(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["state_dict"])
        model.eval()
        vm_final, slices_final, examples_final = eval_model(
            vname, model, None,
            val_data, tok_arr_v, reg_arr_v, n_reg_v, n_sup_v,
            rsizes_v, vocab_known_count, device, args.batch_size * 4, args)

        best_m = result["best_metrics"] if result["best_metrics"] else vm_final
        best_m["no_improving_checkpoint_found"] = result["no_improving_checkpoint_found"]
        best_m["best_step"]                      = result["best_step"]
        all_best[vname] = best_m
        all_slice_rows.extend(slices_final)

        if vname == "real_region_router":
            real_examples = examples_final

    # ══════════════════════════════════════════════════════════════════════════
    # Step 8: Slice eval for frequency prior and baselines
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[step 8] Final slice evaluation for all variants...")
    for bname, bmodel, btok in [
        ("frequency_prior_baseline",      None,            tok_arr_real),
        ("last_token_mlp_baseline",       last_tok_model,  tok_arr_real),
        ("mean_embedding_mlp_baseline",   mean_emb_model,  tok_arr_real),
    ]:
        bm_ckpt = os.path.join(args.output_dir, f"best_{bname}.pt")
        if bmodel is not None and os.path.isfile(bm_ckpt):
            ck = torch.load(bm_ckpt, map_location=device, weights_only=False)
            bmodel.load_state_dict(ck["state_dict"])
        _, slices_b, _ = eval_model(
            bname, bmodel if bmodel is not None else None,
            freq_order if bname == "frequency_prior_baseline" else None,
            val_data, btok, reg_arr, n_regions, n_super,
            rsizes_real, vocab_known_count, device, args.batch_size * 4, args)
        # Replace earlier stub slice rows
        all_slice_rows = [r for r in all_slice_rows if r.get("variant") != bname]
        all_slice_rows.extend(slices_b)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 9: Save CSV logs and reports
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[step 9] Writing outputs...")

    _wcsv(os.path.join(args.output_dir, "train_log.csv"), all_train_log)
    _wcsv(os.path.join(args.output_dir, "eval_log.csv"),  all_eval_log)
    _wcsv(os.path.join(args.output_dir, "slice_metrics.csv"), all_slice_rows)

    # Comparison CSV
    comp_cols = [
        "variant", "region_acc@1", "region_recall@2", "region_recall@4",
        "region_recall@8", "region_recall@16", "region_recall@32",
        "mean_rank_gold_region", "median_rank_gold_region",
        "region_ce", "region_entropy", "region_margin_top1_top2",
        "candidate_fraction@8", "avg_candidate_set_size@8", "gold_token_coverage@8",
        "base_top5_any_in_C@8", "theoretical_output_flop_reduction@8",
        "router_params", "no_improving_checkpoint_found", "best_step",
    ]
    comp_rows = []
    for vn in ["frequency_prior_baseline", "last_token_mlp_baseline",
               "mean_embedding_mlp_baseline",
               "real_region_router", "shuffled_region_router", "random_region_router"]:
        vm = all_best.get(vn, {})
        row = {"variant": vn}
        for c in comp_cols[1:]:
            row[c] = vm.get(c, float("nan"))
        comp_rows.append(row)
    _wcsv(os.path.join(args.output_dir, "phase1A_region_router_comparison.csv"), comp_rows)

    # Coverage by k CSV
    cov_rows = []
    for k in _TOPK_LIST:
        vm_r = all_best.get("real_region_router", {})
        cov_rows.append({
            "k": k,
            "variant": "real_region_router",
            "gold_region_recall":      vm_r.get(f"region_recall@{k}", float("nan")),
            "gold_token_coverage":     vm_r.get(f"gold_token_coverage@{k}", float("nan")),
            "avg_candidate_set_size":  vm_r.get(f"avg_candidate_set_size@{k}", float("nan")),
            "median_candidate_set_size": vm_r.get(f"median_candidate_set_size@{k}", float("nan")),
            "p90_candidate_set_size":  vm_r.get(f"p90_candidate_set_size@{k}", float("nan")),
            "candidate_fraction":      vm_r.get(f"candidate_fraction@{k}", float("nan")),
            "base_top1_in_C":          vm_r.get(f"base_top1_in_C@{k}", float("nan")),
            "base_top5_any_in_C":      vm_r.get(f"base_top5_any_in_C@{k}", float("nan")),
            "base_top5_all_in_C":      vm_r.get(f"base_top5_all_in_C@{k}", float("nan")),
            "base_top10_any_in_C":     vm_r.get(f"base_top10_any_in_C@{k}", float("nan")),
            "selected_lm_flops":       vm_r.get(f"selected_lm_head_flops_estimate@{k}", float("nan")),
            "flop_reduction":          vm_r.get(f"theoretical_output_flop_reduction@{k}", float("nan")),
        })
    _wcsv(os.path.join(args.output_dir, "coverage_by_k.csv"), cov_rows)

    # Best metrics JSON
    with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
        json.dump({vn: {k: (v if not isinstance(v, float) or v == v else None)
                        for k, v in vm.items()}
                   for vn, vm in all_best.items()}, f, indent=2)
    print(f"[save] {os.path.join(args.output_dir, 'best_metrics.json')}")

    # ── Markdown report ───────────────────────────────────────────────────────
    recommendation = write_report(all_best, args.output_dir, args)

    # ── Example files ─────────────────────────────────────────────────────────
    real_examples_list = locals().get("real_examples", [])
    write_example_files(real_examples_list, all_best.get("real_region_router", {}),
                        all_best.get("frequency_prior_baseline", {}), args.output_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Final console output
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0_global
    print(f"\n{'='*70}")
    print(f" PHASE 1A STATIC REGION ROUTER VERDICT  ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    hdr = (f"  {'variant':<36}  acc@1   rec@4   rec@8   rec@16  "
           f"cfrac@8  cand@8     cov@8   b5any@8  flop_red@8")
    print(hdr)
    print("  " + "─" * len(hdr))
    for vn in ["frequency_prior_baseline", "last_token_mlp_baseline",
               "mean_embedding_mlp_baseline", "real_region_router",
               "shuffled_region_router", "random_region_router"]:
        vm = all_best.get(vn, {})
        print(
            f"  {vn:<36}"
            f"  {_fmt(vm.get('region_acc@1',           float('nan'))):6}"
            f"  {_fmt(vm.get('region_recall@4',         float('nan'))):6}"
            f"  {_fmt(vm.get('region_recall@8',         float('nan'))):6}"
            f"  {_fmt(vm.get('region_recall@16',        float('nan'))):6}"
            f"  {_fmt(vm.get('candidate_fraction@8',    float('nan'))):7}"
            f"  {_fmt(vm.get('avg_candidate_set_size@8',float('nan'))):10}"
            f"  {_fmt(vm.get('gold_token_coverage@8',  float('nan'))):6}"
            f"  {_fmt(vm.get('base_top5_any_in_C@8',   float('nan'))):7}"
            f"  {_fmt(vm.get('theoretical_output_flop_reduction@8', float('nan')))}"
        )
    print()
    print(f"recommendation: {recommendation}")
    print(f"{'='*70}")
    print(f"\nKey outputs:")
    print(f"  {args.output_dir}/phase1A_region_router_comparison.csv")
    print(f"  {args.output_dir}/coverage_by_k.csv")
    print(f"  {args.output_dir}/phase1A_static_region_router_report.md")
    print(f"\nrecommendation: {recommendation}")


if __name__ == "__main__":
    main()
