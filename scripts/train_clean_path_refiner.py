#!/usr/bin/env python3
"""
Clean path-conditioned refiner training.

Trains variants over pre-built .pt shard datasets (from build_clean_path_refiner_dataset.py).
Supports:
  C     — Richer MLP scorer with router/mem probs + confidence features
  D3    — RegionTransformerRefiner: attends over n_fine=128 region slots, then
          scores tokens conditioned on the refined region representation.

Key constraints:
  - Base model frozen; only refiner parameters are updated.
  - CE loss only on covered positions (gold in candidate set).
  - Base candidate logit always included; refiner learns residual near zero.
  - KL + delta regularisation to prevent collapse.
  - Step-based training (--steps), eval every --eval_every steps.

Shard format (.pt files from build_clean_path_refiner_dataset.py):
  h_prime, gold_token, gold_region, cand_tok, cand_fine, gold_cand_idx,
  covered, router_topk_reg, router_topk_prb, mem_topk_reg, mem_topk_prb,
  router_margin, mem_margin, split, type_arr

Usage:
    python scripts/train_clean_path_refiner.py \
        --train_dir  runs/path_refiner_clean/data/train_hgrid_K24 \
        --val_dir    runs/path_refiner_clean/data/val_hgrid_K24 \
        --small_ckpt runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \
        --super_map  runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \
        --variant    D3 --d3_size small \
        --output_dir runs/path_refiner_clean/d3_small \
        --steps 20000 --device cuda

    # Variant C (richer MLP):
    python scripts/train_clean_path_refiner.py ... --variant C --d_hidden 256
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe

SPLIT_NAMES = {0: "core", 1: "medium", 2: "boundary", 3: "tight"}
TYPE_NAMES  = {0: "other", 1: "A", 2: "B", 3: "C"}


# ── Super-map helper ──────────────────────────────────────────────────────────

def load_r2s(path: str, n_fine: int) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    r2s = np.full(n_fine, 0, dtype=np.int32)
    for k, v in d.items():
        fid = int(k)
        if 0 <= fid < n_fine:
            r2s[fid] = int(v)
    return r2s


# ── Model variants ────────────────────────────────────────────────────────────

class RicherMLPRefiner(nn.Module):
    """
    Variant C (richer MLP).
    Features per candidate: h_proj, tok_proj, fine_emb, super_emb,
    p_router_fine, p_mem_fine, router_margin, mem_margin, is_router, is_mem
    """
    def __init__(self, d_model: int, n_fine: int, n_super: int,
                 d_region: int = 32, d_hidden: int = 256) -> None:
        super().__init__()
        self.d_model   = d_model
        self.n_fine    = n_fine
        self.n_super   = n_super
        self.fine_emb  = nn.Embedding(n_fine  + 1, d_region, padding_idx=n_fine)
        self.super_emb = nn.Embedding(n_super + 1, d_region, padding_idx=n_super)
        self.h_proj    = nn.Linear(d_model, d_region, bias=False)
        self.tok_proj  = nn.Linear(d_model, d_region, bias=False)
        # extra scalars: p_router_fine, p_mem_fine, router_margin, mem_margin, is_router, is_mem
        feat_dim = d_region * 4 + 6
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.h_proj.weight, std=0.02)
        nn.init.normal_(self.tok_proj.weight, std=0.02)
        nn.init.zeros_(self.fine_emb.weight)
        nn.init.zeros_(self.super_emb.weight)
        nn.init.normal_(self.mlp[0].weight, std=0.01)
        nn.init.zeros_(self.mlp[0].bias)

    def _build_fine_probs(self, topk_reg, topk_prb):
        """(B, K) → (B, n_fine) float32."""
        B = topk_reg.size(0)
        p     = torch.zeros(B, self.n_fine, device=topk_reg.device)
        valid = (topk_reg >= 0).float()
        reg_c = topk_reg.clamp(min=0).long()
        p.scatter_add_(1, reg_c, topk_prb.float() * valid)
        return p

    def _build_fine_indicator(self, topk_reg):
        """(B, K) → (B, n_fine) bool."""
        B, K  = topk_reg.shape
        ind   = torch.zeros(B, self.n_fine, dtype=torch.bool, device=topk_reg.device)
        valid = topk_reg >= 0                              # (B, K)
        flat_b = torch.arange(B, device=topk_reg.device).unsqueeze(1).expand(-1, K)
        flat_b = flat_b[valid]
        flat_r = topk_reg[valid].long()
        ind[flat_b, flat_r] = True
        return ind

    def forward(self, h_prime, cand_tok, cand_fine, cand_super, cand_mask,
                token_emb_w, r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
                r_margin, m_margin):
        B, C = cand_tok.shape
        cf   = cand_fine.clamp(min=0).long()
        cs   = cand_super.clamp(min=0).long()
        ct   = cand_tok.clamp(min=0).long()

        tok_e  = F.embedding(ct, token_emb_w)   # (B, C, d_model)
        fine_e = self.fine_emb(cf)               # (B, C, d_r)
        sup_e  = self.super_emb(cs)              # (B, C, d_r)
        h_e    = self.h_proj(h_prime.float())    # (B, d_r)
        t_e    = self.tok_proj(tok_e.float())    # (B, C, d_r)
        h_exp  = h_e.unsqueeze(1).expand(-1, C, -1)

        # Per-candidate: router/mem prob for its fine region
        p_r_fine = self._build_fine_probs(r_topk_reg, r_topk_prb)   # (B, n_fine)
        p_m_fine = self._build_fine_probs(m_topk_reg, m_topk_prb)   # (B, n_fine)
        in_r     = self._build_fine_indicator(r_topk_reg)             # (B, n_fine) bool
        in_m     = self._build_fine_indicator(m_topk_reg)             # (B, n_fine) bool

        cand_pr  = p_r_fine.gather(1, cf).unsqueeze(-1)              # (B, C, 1)
        cand_pm  = p_m_fine.gather(1, cf).unsqueeze(-1)              # (B, C, 1)
        cand_ir  = in_r.gather(1, cf).float().unsqueeze(-1)          # (B, C, 1)
        cand_im  = in_m.gather(1, cf).float().unsqueeze(-1)          # (B, C, 1)
        rmg_exp  = r_margin.float().unsqueeze(1).unsqueeze(2).expand(-1, C, 1)  # (B, C, 1)
        mmg_exp  = m_margin.float().unsqueeze(1).unsqueeze(2).expand(-1, C, 1)

        feat     = torch.cat([h_exp, t_e, fine_e, sup_e,
                              cand_pr, cand_pm, rmg_exp, mmg_exp,
                              cand_ir, cand_im], dim=-1)               # (B, C, feat_dim)
        # Base: h_prime · tok_emb[cand] approximates backbone logit for each candidate
        base_raw = (h_prime.float().unsqueeze(1) * tok_e.float()).sum(-1)   # (B, C)
        delta    = self.mlp(feat).squeeze(-1) * self.residual_scale           # (B, C)
        base     = base_raw.masked_fill(~cand_mask, -1e9)
        scores   = (base_raw + delta).masked_fill(~cand_mask, -1e9)
        return scores, base


class RegionTransformerRefiner(nn.Module):
    """
    D3: attends over all n_fine=128 region slots (full attention, no masking needed
    at n_fine=128), then scores candidate tokens conditioned on the refined region
    representation.
    """
    def __init__(self, d_model: int, n_fine: int, n_super: int,
                 r2s_tensor: torch.Tensor,
                 d_region: int = 128, n_layers: int = 1,
                 n_heads: int = 4, d_ffn: int = 256,
                 d_tok: int = 64) -> None:
        super().__init__()
        self.d_model   = d_model
        self.n_fine    = n_fine
        self.n_super   = n_super
        self.d_region  = d_region
        self.register_buffer("r2s", r2s_tensor.long())

        self.fine_emb  = nn.Embedding(n_fine,  d_region)
        self.super_emb = nn.Embedding(n_super, d_region)
        self.h_proj    = nn.Linear(d_model, d_region)
        # input: fine_emb(d_r) + super_emb(d_r) + h_ctx(d_r) + p_router(1) + p_mem(1)
        self.inp_proj  = nn.Linear(d_region * 3 + 2, d_region)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_region, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.tok_proj  = nn.Linear(d_model, d_tok, bias=False)
        self.reg_proj  = nn.Linear(d_region, d_tok, bias=False)
        self.scorer    = nn.Linear(d_tok * 2, 1)

        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fine_emb.weight,  std=0.02)
        nn.init.normal_(self.super_emb.weight, std=0.02)
        nn.init.normal_(self.h_proj.weight,    std=0.02)
        nn.init.zeros_(self.h_proj.bias)

    def _build_fine_probs(self, topk_reg, topk_prb):
        B     = topk_reg.size(0)
        p     = torch.zeros(B, self.n_fine, device=topk_reg.device)
        valid = (topk_reg >= 0).float()
        reg_c = topk_reg.clamp(min=0).long()
        p.scatter_add_(1, reg_c, topk_prb.float() * valid)
        return p

    def forward(self, h_prime, cand_tok, cand_fine, cand_super, cand_mask,
                token_emb_w, r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
                r_margin, m_margin):
        B, C   = cand_tok.shape
        device = h_prime.device

        # Full (B, n_fine) prob distributions
        p_r = self._build_fine_probs(r_topk_reg, r_topk_prb)  # (B, n_fine)
        p_m = self._build_fine_probs(m_topk_reg, m_topk_prb)  # (B, n_fine)

        # Region node features
        all_fine = torch.arange(self.n_fine, device=device)          # (n_fine,)
        all_sup  = self.r2s[all_fine]                                 # (n_fine,)
        fine_e   = self.fine_emb(all_fine)                            # (n_fine, d_r)
        sup_e    = self.super_emb(all_sup)                            # (n_fine, d_r)
        h_ctx    = self.h_proj(h_prime.float())                       # (B, d_r)

        # Expand to (B, n_fine, ...)
        fine_e   = fine_e.unsqueeze(0).expand(B, -1, -1)             # (B, n_fine, d_r)
        sup_e    = sup_e.unsqueeze(0).expand(B, -1, -1)              # (B, n_fine, d_r)
        h_exp    = h_ctx.unsqueeze(1).expand(-1, self.n_fine, -1)    # (B, n_fine, d_r)
        pr_exp   = p_r.unsqueeze(-1)                                  # (B, n_fine, 1)
        pm_exp   = p_m.unsqueeze(-1)                                  # (B, n_fine, 1)

        region_inp  = torch.cat([fine_e, sup_e, h_exp, pr_exp, pm_exp], dim=-1)
        region_nodes = self.inp_proj(region_inp)                      # (B, n_fine, d_r)

        region_out = self.transformer(region_nodes)                   # (B, n_fine, d_r)

        # Token scoring
        cf  = cand_fine.clamp(min=0).long()    # (B, C)
        ct  = cand_tok.clamp(min=0).long()     # (B, C)
        tok_e_raw = F.embedding(ct, token_emb_w)        # (B, C, d_model)
        # Base: h_prime · tok_emb[cand] approximates backbone logit for each candidate
        base_raw  = (h_prime.float().unsqueeze(1) * tok_e_raw.float()).sum(-1)  # (B, C)
        tok_e     = self.tok_proj(tok_e_raw.float())    # (B, C, d_tok)

        # Gather region representation for each candidate's fine region
        b_idx    = torch.arange(B, device=device).unsqueeze(1).expand(-1, C)
        reg_e    = region_out[b_idx, cf]        # (B, C, d_r)
        reg_e    = self.reg_proj(reg_e)         # (B, C, d_tok)

        delta    = self.scorer(torch.cat([tok_e, reg_e], dim=-1)).squeeze(-1)
        delta    = delta * self.residual_scale

        base   = base_raw.masked_fill(~cand_mask, -1e9)
        scores = (base_raw + delta).masked_fill(~cand_mask, -1e9)
        return scores, base


def build_refiner(args, d_model: int, n_fine: int, n_super: int,
                  r2s_tensor: Optional[torch.Tensor]) -> nn.Module:
    v = args.variant.upper()
    if v == "C":
        return RicherMLPRefiner(d_model=d_model, n_fine=n_fine, n_super=n_super,
                                d_region=args.d_region, d_hidden=args.d_hidden)
    if v == "D3":
        assert r2s_tensor is not None, "--super_map required for D3 variant"
        size = args.d3_size.lower()
        if size == "small":
            return RegionTransformerRefiner(d_model=d_model, n_fine=n_fine, n_super=n_super,
                                            r2s_tensor=r2s_tensor,
                                            d_region=128, n_layers=1, n_heads=4, d_ffn=256, d_tok=64)
        if size == "base":
            return RegionTransformerRefiner(d_model=d_model, n_fine=n_fine, n_super=n_super,
                                            r2s_tensor=r2s_tensor,
                                            d_region=256, n_layers=2, n_heads=4, d_ffn=512, d_tok=128)
        raise ValueError(f"Unknown d3_size {args.d3_size!r}; choose small or base")
    raise ValueError(f"Unknown variant {args.variant!r}; choose C or D3")


# ── Shard streaming dataset ───────────────────────────────────────────────────

class ShardStreamDataset(IterableDataset):
    """
    Streams through .pt shard files one at a time; reshuffles shard order
    each epoch. Computes cand_super and cand_mask from cand_fine + r2s.
    """

    def __init__(self, shard_dir: str, r2s_np: np.ndarray,
                 shuffle: bool = True) -> None:
        paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
        if not paths:
            raise RuntimeError(f"No shard_*.pt files in {shard_dir}")
        self.paths  = paths
        self.r2s    = r2s_np
        self.shuffle = shuffle
        # Count total positions without loading data
        self.total = None
        print(f"[dataset] {len(paths)} shards in {shard_dir}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        paths = list(self.paths)
        if self.shuffle:
            random.shuffle(paths)
        for path in paths:
            shard = torch.load(path, map_location="cpu", weights_only=True)
            N     = len(shard["covered"])
            idxs  = list(range(N))
            if self.shuffle:
                random.shuffle(idxs)
            cf    = shard["cand_fine"]   # (N, C) int16
            ct    = shard["cand_tok"]    # (N, C) int32

            # Compute cand_super and cand_mask once per shard
            cf_np   = cf.numpy().astype(np.int64).clip(min=0)
            cs_np   = self.r2s[cf_np].astype(np.int64)
            cs_np[cf.numpy() < 0] = 0
            cand_super = torch.from_numpy(cs_np)               # (N, C)
            cand_mask  = (ct >= 0)                             # (N, C) bool

            for i in idxs:
                has_t = "type_arr" in shard
                yield {
                    "h_prime":      shard["h_prime"][i].float(),
                    "cand_tok":     shard["cand_tok"][i].long(),
                    "cand_fine":    shard["cand_fine"][i].long(),
                    "cand_super":   cand_super[i],
                    "cand_mask":    cand_mask[i],
                    "gold_idx":     shard["gold_cand_idx"][i].long(),
                    "covered":      shard["covered"][i].bool(),
                    "r_topk_reg":   shard["router_topk_reg"][i].long(),
                    "r_topk_prb":   shard["router_topk_prb"][i].float(),
                    "m_topk_reg":   shard["mem_topk_reg"][i].long(),
                    "m_topk_prb":   shard["mem_topk_prb"][i].float(),
                    "r_margin":     shard["router_margin"][i].float(),
                    "m_margin":     shard["mem_margin"][i].float(),
                    "split":        shard["split"][i].long(),
                    "type_arr":     shard["type_arr"][i].long() if has_t
                                    else torch.zeros(1, dtype=torch.long).squeeze(),
                }


def make_infinite(dataset: IterableDataset, batch_size: int):
    """Infinite iterator over batches; restarts dataset each epoch."""
    while True:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, drop_last=False)
        for batch in loader:
            yield batch


# ── Eval helpers ─────────────────────────────────────────────────────────────

def _aggregate_eval_stats(stats: Dict) -> Dict:
    results: Dict = {}
    total_ce  = sum(s[0] for s in stats.values())
    total_cov = sum(s[1] for s in stats.values())
    total_n   = sum(s[2] for s in stats.values())
    results["covered_nll"]   = total_ce  / max(total_cov, 1)
    results["coverage"]      = total_cov / max(total_n, 1)
    results["fallback_rate"] = 1.0 - results["coverage"]
    for sid, sname in SPLIT_NAMES.items():
        sub = [s for k, s in stats.items() if k[0] == sid]
        cn  = sum(s[1] for s in sub)
        tn  = sum(s[2] for s in sub)
        cce = sum(s[0] for s in sub)
        results[f"cnll_{sname}"]     = cce / max(cn, 1)
        results[f"cov_{sname}"]      = cn  / max(tn, 1)
        results[f"fallback_{sname}"] = 1.0 - results[f"cov_{sname}"]
    return results


@torch.no_grad()
def eval_force_zero(val_dir: str, tok_emb_w: torch.Tensor,
                    eval_batch_size: int, device) -> Tuple[Dict, str]:
    """
    Architecture-independent force-zero baseline eval.

    Loads shards directly (no model, no ShardStreamDataset, no r2s).
    Computes:  scores[b,c] = h_prime[b] · tok_emb[cand_tok[b,c]]
    Evaluates ALL examples — no max_batches cap.
    Returns (metrics_dict, dataset_fingerprint).
    """
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt in {val_dir}")

    emb_w = tok_emb_w.float().to(device)
    stats: Dict = defaultdict(lambda: [0.0, 0, 0])

    total_n          = 0
    total_cov        = 0
    sum_cand_counts  = 0
    sum_gold_idx_cov = 0
    sum_gold_tok     = 0

    for path in paths:
        shard    = torch.load(path, map_location="cpu", weights_only=True)
        N        = len(shard["covered"])
        has_type = "type_arr" in shard
        sp_all   = shard["split"].long()
        tp_all   = shard["type_arr"].long() if has_type else torch.zeros(N, dtype=torch.long)
        has_gtok = "gold_token" in shard

        for start in range(0, N, eval_batch_size):
            end     = min(start + eval_batch_size, N)
            h       = shard["h_prime"][start:end].float().to(device)       # (B, d_model)
            ct      = shard["cand_tok"][start:end].long().to(device)       # (B, C)
            g_idx   = shard["gold_cand_idx"][start:end].long().to(device)  # (B,)
            covered = shard["covered"][start:end].bool().to(device)        # (B,)
            sp      = sp_all[start:end]
            tp      = tp_all[start:end]
            B, C    = ct.shape

            cmask  = (ct >= 0)
            tok_e  = F.embedding(ct.clamp(min=0), emb_w)                  # (B, C, d_model)
            scores = (h.unsqueeze(1) * tok_e).sum(-1)                      # (B, C)
            scores = scores.masked_fill(~cmask, float("-inf"))

            cov      = covered
            n_cov_b  = int(cov.sum())

            if n_cov_b > 0:
                lp      = F.log_softmax(scores[cov], dim=-1)
                gi      = g_idx[cov]
                ce_vals = -lp[torch.arange(n_cov_b, device=device), gi].cpu()
                cov_idx = torch.where(cov)[0].cpu()
                for j, i in enumerate(cov_idx.tolist()):
                    key = (int(sp[i].item()), int(tp[i].item()))
                    stats[key][0] += float(ce_vals[j])
                    stats[key][1] += 1
                sum_gold_idx_cov += int(g_idx[cov].sum())

            for i in range(B):
                key = (int(sp[i].item()), int(tp[i].item()))
                stats[key][2] += 1

            total_n         += B
            total_cov       += n_cov_b
            sum_cand_counts += int(cmask.sum())
            if has_gtok:
                sum_gold_tok += int(shard["gold_token"][start:end].long().sum())

    results = _aggregate_eval_stats(stats)
    results["num_examples"]  = total_n
    results["num_covered"]   = total_cov
    results["mean_cand_count"] = sum_cand_counts / max(total_n, 1)

    fp_data = {
        "num_shards":       len(paths),
        "total_n":          total_n,
        "total_cov":        total_cov,
        "sum_cand_counts":  sum_cand_counts,
        "sum_gold_idx_cov": sum_gold_idx_cov,
        "sum_gold_tok":     sum_gold_tok,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fp_data, sort_keys=True).encode()
    ).hexdigest()[:16]
    results["dataset_fingerprint"] = fingerprint

    return results, fingerprint


def check_baseline_match(results: Dict, fingerprint: str,
                         baseline_path: str, fail: bool,
                         context: str = "") -> bool:
    """
    Compare force-zero eval results against the official saved-candidate baseline.
    Returns True if PASS, raises RuntimeError if fail=True and check fails.
    """
    with open(baseline_path) as f:
        ref = json.load(f)

    nll_diff  = abs(results["covered_nll"] - ref["covered_nll"])
    cov_diff  = abs(results["coverage"]    - ref["coverage"])
    fp_match  = fingerprint == ref.get("dataset_fingerprint", "")
    ok        = nll_diff < 1e-4 and cov_diff < 1e-6 and fp_match
    tag       = "PASS" if ok else "FAIL"
    pfx       = f"[{context}] " if context else ""

    print(f"  {pfx}baseline check: {tag}")
    print(f"    covered_nll : {results['covered_nll']:.6f}  "
          f"(ref={ref['covered_nll']:.6f}  diff={nll_diff:.2e})")
    print(f"    coverage    : {results['coverage']:.6f}  "
          f"(ref={ref['coverage']:.6f}  diff={cov_diff:.2e})")
    print(f"    fingerprint : {fingerprint}  "
          f"(ref={ref.get('dataset_fingerprint','?')}  match={fp_match})")

    if not ok and fail:
        raise RuntimeError(
            f"{pfx}Force-zero baseline MISMATCH. "
            f"nll_diff={nll_diff:.2e}  cov_diff={cov_diff:.2e}  fp_match={fp_match}. "
            f"Fix before training."
        )
    return ok


@torch.no_grad()
def check_init_identity(model, val_dir: str, tok_emb_w: torch.Tensor,
                        r2s_np: np.ndarray, device,
                        print_delta_stats: bool = False) -> None:
    """
    Verify that the initialized model is an exact identity transform:
      model scores == h_prime @ tok_emb[cand_tok]  (force-zero baseline)
    on the first batch (up to 64 examples) of the first val shard.

    Asserts NLL diff < 1e-4 and max |delta| < 1e-5.
    Raises RuntimeError on failure — training will not proceed.
    """
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt in {val_dir}")

    shard    = torch.load(paths[0], map_location="cpu", weights_only=True)
    N        = len(shard["covered"])
    B        = min(64, N)

    h        = shard["h_prime"][:B].float().to(device)
    ct       = shard["cand_tok"][:B].long().to(device)
    cf_cpu   = shard["cand_fine"][:B]
    covered  = shard["covered"][:B].bool().to(device)
    g_idx    = shard["gold_cand_idx"][:B].long().to(device)
    r_reg    = shard["router_topk_reg"][:B].long().to(device)
    r_prb    = shard["router_topk_prb"][:B].float().to(device)
    m_reg    = shard["mem_topk_reg"][:B].long().to(device)
    m_prb    = shard["mem_topk_prb"][:B].float().to(device)
    r_margin = shard["router_margin"][:B].float().to(device)
    m_margin = shard["mem_margin"][:B].float().to(device)

    # Build cand_super on CPU (same logic as ShardStreamDataset)
    cf_np  = cf_cpu.numpy().astype(np.int64).clip(min=0)
    cs_np  = r2s_np[cf_np].astype(np.int64)
    cs_np[cf_cpu.numpy() < 0] = 0
    cs     = torch.from_numpy(cs_np).long().to(device)
    cf     = cf_cpu.long().to(device)

    cmask  = (ct >= 0)
    emb_w  = tok_emb_w.float().to(device)

    # Force-zero scores (architecture-independent)
    tok_e_fz  = F.embedding(ct.clamp(min=0), emb_w)
    fz_scores = (h.unsqueeze(1) * tok_e_fz).sum(-1).masked_fill(~cmask, float("-inf"))

    # Model scores (no autocast — need exact float32 arithmetic)
    model.eval()
    scores, _ = model(h, ct, cf, cs, cmask, emb_w,
                      r_reg, r_prb, m_reg, m_prb, r_margin, m_margin)
    model.train()

    # Delta over valid (non-padded) candidates — must be 0.0 with residual_scale=0
    delta_all     = (scores - fz_scores)[cmask]
    max_abs_delta = float(delta_all.abs().max()) if cmask.any() else 0.0
    mean_abs_delta = float(delta_all.abs().mean()) if cmask.any() else 0.0

    # NLL comparison on covered examples
    cov = covered
    if cov.sum() > 0:
        n_cov       = int(cov.sum())
        arange      = torch.arange(n_cov, device=device)
        fz_nll      = -F.log_softmax(fz_scores[cov], dim=-1)[arange, g_idx[cov]]
        md_nll      = -F.log_softmax(scores[cov],    dim=-1)[arange, g_idx[cov]]
        fz_mean_nll = float(fz_nll.mean())
        md_mean_nll = float(md_nll.mean())
        nll_diff    = abs(md_mean_nll - fz_mean_nll)
    else:
        fz_mean_nll = float("nan")
        md_mean_nll = float("nan")
        nll_diff    = 0.0

    ok  = nll_diff < 1e-4 and max_abs_delta < 1e-5
    tag = "PASS" if ok else "FAIL"

    print(f"  [init identity check] {tag}")
    print(f"    force_zero NLL : {fz_mean_nll:.6f}")
    print(f"    model NLL      : {md_mean_nll:.6f}  (diff={nll_diff:.2e}  threshold=1e-4)")
    print(f"    max |delta|    : {max_abs_delta:.2e}  (threshold=1e-5)")
    if print_delta_stats:
        print(f"    mean |delta|   : {mean_abs_delta:.2e}")
        print(f"    n_valid_cands  : {int(cmask.sum())}")
        print(f"    n_covered      : {int(cov.sum())}")

    if not ok:
        raise RuntimeError(
            f"Init identity FAIL: nll_diff={nll_diff:.2e}  max_abs_delta={max_abs_delta:.2e}. "
            f"The refiner is not an identity at initialization. "
            f"Ensure residual_scale is initialized to 0.0 in both RicherMLPRefiner and "
            f"RegionTransformerRefiner, and that the forward uses '* self.residual_scale'."
        )


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_loss(model, batch, device, lambda_kl: float, lambda_delta: float):
    h         = batch["h_prime"].to(device)
    ct        = batch["cand_tok"].to(device)
    cf        = batch["cand_fine"].to(device)
    cs        = batch["cand_super"].to(device)
    cmask     = batch["cand_mask"].to(device)
    gold_idx  = batch["gold_idx"].to(device)
    covered   = batch["covered"].to(device)
    r_reg     = batch["r_topk_reg"].to(device)
    r_prb     = batch["r_topk_prb"].to(device)
    m_reg     = batch["m_topk_reg"].to(device)
    m_prb     = batch["m_topk_prb"].to(device)
    r_margin  = batch["r_margin"].to(device)
    m_margin  = batch["m_margin"].to(device)

    token_emb_w = model._tok_emb_w

    scores, base_scores = model(h, ct, cf, cs, cmask, token_emb_w,
                                r_reg, r_prb, m_reg, m_prb, r_margin, m_margin)

    cov_mask = covered.bool()
    n_cov    = int(cov_mask.sum())
    if n_cov == 0:
        return None, {}

    sc  = scores[cov_mask]       # (n_cov, C) — final scores (base + delta), masked
    bc  = base_scores[cov_mask]  # (n_cov, C) — base scores only, masked
    gi  = gold_idx[cov_mask]     # (n_cov,)

    loss_ce = F.cross_entropy(sc, gi)

    # KL toward base (h_prime · tok_emb approximation of backbone logit)
    p_pred  = F.softmax(sc, dim=-1)
    p_base  = F.softmax(bc.detach(), dim=-1)
    loss_kl = (p_pred * (torch.log(p_pred + 1e-9) - torch.log(p_base + 1e-9))).sum(-1).mean()

    # Delta regularisation: L2 on residual over all valid (non-pad) candidates
    delta   = scores[cmask] - base_scores[cmask].detach()
    loss_dl = delta.pow(2).mean()

    total = loss_ce + lambda_kl * loss_kl + lambda_delta * loss_dl
    info = {"ce": loss_ce.item(), "kl": loss_kl.item(), "delta": loss_dl.item()}
    return total, info


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_dataset(model, shard_dir: str, r2s_np: np.ndarray,
                 batch_size: int, device, max_batches: int = 500,
                 eval_batch_size: int = 0) -> Dict:
    """
    Evaluate model (with delta) on a shard dataset.
    eval_batch_size: if > 0, overrides batch_size for the eval dataloader.
    max_batches: cap on number of batches (for fast mid-training checks).
    """
    model.eval()
    bs         = eval_batch_size if eval_batch_size > 0 else batch_size
    val_ds     = ShardStreamDataset(shard_dir, r2s_np, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=0, drop_last=False)

    # Per split/type accumulators: {key: [ce_sum, n_cov, n_total]}
    stats: Dict = defaultdict(lambda: [0.0, 0, 0])

    for bi, batch in enumerate(val_loader):
        if bi >= max_batches:
            break
        h        = batch["h_prime"].to(device)
        ct       = batch["cand_tok"].to(device)
        cf       = batch["cand_fine"].to(device)
        cs       = batch["cand_super"].to(device)
        cmask    = batch["cand_mask"].to(device)
        gold_idx = batch["gold_idx"].to(device)
        covered  = batch["covered"].to(device)
        r_reg    = batch["r_topk_reg"].to(device)
        r_prb    = batch["r_topk_prb"].to(device)
        m_reg    = batch["m_topk_reg"].to(device)
        m_prb    = batch["m_topk_prb"].to(device)
        r_margin = batch["r_margin"].to(device)
        m_margin = batch["m_margin"].to(device)
        split    = batch["split"]
        type_arr = batch["type_arr"]

        token_emb_w = model._tok_emb_w
        with autocast("cuda"):
            scores, _ = model(h, ct, cf, cs, cmask, token_emb_w,
                              r_reg, r_prb, m_reg, m_prb, r_margin, m_margin)

        B = h.size(0)
        cov_mask = covered.bool()
        if cov_mask.sum() > 0:
            lp = F.log_softmax(scores[cov_mask], dim=-1)
            gi = gold_idx[cov_mask]
            ce_vals = -lp[torch.arange(int(cov_mask.sum()), device=device), gi].cpu()
            cov_idx = torch.where(cov_mask)[0].cpu()
            for j, i in enumerate(cov_idx.tolist()):
                key = (int(split[i].item()), int(type_arr[i].item()))
                stats[key][0] += float(ce_vals[j])
                stats[key][1] += 1
        for i in range(B):
            key = (int(split[i].item()), int(type_arr[i].item()))
            stats[key][2] += 1

    results = _aggregate_eval_stats(stats)
    model.train()
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train_variant(args, backbone, d_model: int, n_fine: int, n_super: int,
                  r2s_np: np.ndarray, token_emb_w: torch.Tensor,
                  device: torch.device) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    r2s_tensor = torch.from_numpy(r2s_np).long().to(device)
    model = build_refiner(args, d_model, n_fine, n_super, r2s_tensor).to(device)
    # Attach frozen token embedding as a non-parameter buffer (on device)
    model.register_buffer("_tok_emb_w", token_emb_w.float().to(device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] variant={args.variant}  params={n_params:,}")

    train_ds  = ShardStreamDataset(args.train_dir, r2s_np, shuffle=True)
    train_inf = make_infinite(train_ds, args.batch_size)

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler("cuda")
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

    log_path  = os.path.join(args.output_dir, "train_log.csv")
    best_path = os.path.join(args.output_dir, "best_refiner.pt")
    last_path = os.path.join(args.output_dir, "last_refiner.pt")

    log_fields = ["step", "ce", "kl", "delta", "covered_nll", "coverage",
                  "fallback_rate"] + \
                 [f"cnll_{s}" for s in SPLIT_NAMES.values()] + \
                 [f"cov_{s}"  for s in SPLIT_NAMES.values()]

    log_file = open(log_path, "w", newline="")
    log_csv  = csv.DictWriter(log_file, fieldnames=log_fields, extrasaction="ignore")
    log_csv.writeheader()

    best_nll   = float("inf")
    ema_ce     = None
    t0         = time.time()
    model.train()

    if getattr(args, "eval_before_train", False):
        print("[train] === step-0 eval (eval_before_train) ===")

        # Architecture-independent force-zero: h_prime @ tok_emb[cand_tok], full dataset
        tok_emb_dev = model._tok_emb_w
        m0_zero, fp = eval_force_zero(args.val_dir, tok_emb_dev,
                                       args.eval_batch_size, device)
        print(f"  [step 0 / force_zero] covered_nll={m0_zero['covered_nll']:.6f}  "
              f"cov={m0_zero['coverage']:.6f}  fingerprint={fp}")
        print(f"    num_examples={m0_zero['num_examples']:,}  "
              f"num_covered={m0_zero['num_covered']:,}  "
              f"mean_cands={m0_zero['mean_cand_count']:.1f}")

        # Baseline consistency check
        baseline_path = getattr(args, "official_baseline", None)
        fail_hard     = getattr(args, "fail_on_baseline_mismatch", False)
        if baseline_path and os.path.isfile(baseline_path):
            ok = check_baseline_match(m0_zero, fp, baseline_path, fail_hard,
                                      context=f"variant={args.variant}")
            if not ok:
                print("  WARNING: force-zero baseline MISMATCH — results will not be "
                      "comparable across variants until this is fixed.")
        else:
            print("  WARNING: no --official_baseline provided; cannot verify "
                  "cross-variant consistency. Run eval_saved_candidate_baseline.py first.")

        # Init identity check: residual_scale=0 → delta=0 → scores == force_zero
        print()
        print("  [init identity] verifying residual_scale=0 → delta=0 ...")
        check_init_identity(model, args.val_dir, tok_emb_dev, r2s_np, device,
                            print_delta_stats=getattr(args, "print_delta_stats", False))

        # Log step-0 using force_zero metrics (full dataset, authoritative)
        row0 = {"step": 0, "ce": float("nan"), "kl": float("nan"), "delta": float("nan"),
                **m0_zero}
        log_csv.writerow(row0)
        log_file.flush()
        model.train()

    for step in range(1, args.steps + 1):
        batch = next(train_inf)

        with autocast("cuda"):
            loss, info = compute_loss(model, batch, device,
                                      args.lambda_kl, args.lambda_delta)
        if loss is None:
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()

        ema_ce = info["ce"] if ema_ce is None else 0.98 * ema_ce + 0.02 * info["ce"]

        if step % 100 == 0:
            print(f"  step {step:6d}/{args.steps}  ce={ema_ce:.4f}  "
                  f"kl={info['kl']:.4f}  t={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"  [eval] step {step} ...")
            metrics = eval_dataset(model, args.val_dir, r2s_np,
                                   args.batch_size, device,
                                   max_batches=args.eval_max_batches)
            row = {"step": step, **info, **metrics}
            log_csv.writerow(row)
            log_file.flush()

            nll = metrics["covered_nll"]
            print(f"  [eval] step={step}  covered_nll={nll:.4f}  "
                  f"cov={metrics['coverage']:.4f}  "
                  f"fb={metrics['fallback_rate']:.4f}")

            if nll < best_nll:
                best_nll = nll
                torch.save({"step": step, "model": model.state_dict(),
                            "metrics": metrics, "args": vars(args)}, best_path)
                print(f"  [eval] new best  covered_nll={best_nll:.4f}  saved {best_path}")

    torch.save({"step": args.steps, "model": model.state_dict(),
                "metrics": {}, "args": vars(args)}, last_path)
    log_file.close()
    print(f"\n[train] done  best_covered_nll={best_nll:.4f}  ckpt={best_path}")


# ── Eval-only mode ────────────────────────────────────────────────────────────

def run_eval_only(args, token_emb_w: torch.Tensor, device,
                  model=None, r2s_np: Optional[np.ndarray] = None) -> None:
    """
    Run eval-only mode (no training).
    - --force_zero_delta: architecture-independent force-zero eval (no model needed)
    - --check_init_identity: verify model is exact identity at init (model required)
    Both flags may be combined.
    """
    force_zero = getattr(args, "force_zero_delta",     False)
    check_id   = getattr(args, "check_init_identity",  False)

    if not force_zero and not check_id:
        raise RuntimeError(
            "--eval_only requires at least one of --force_zero_delta or --check_init_identity."
        )

    variant_tag = args.variant.upper()
    if variant_tag == "D3":
        variant_tag += f"-{args.d3_size}"
    print(f"\n[eval_only] variant={variant_tag}")
    print(f"[eval_only] val_dir={args.val_dir}")

    if force_zero:
        results, fp = eval_force_zero(args.val_dir, token_emb_w, args.eval_batch_size, device)

        print()
        print(f"  dataset_fingerprint : {fp}")
        print(f"  num_examples        : {results['num_examples']:,}")
        print(f"  num_covered         : {results['num_covered']:,}")
        print(f"  coverage            : {results['coverage']:.6f}")
        print(f"  covered_nll         : {results['covered_nll']:.6f}")
        print(f"  mean_cand_count     : {results['mean_cand_count']:.1f}")

        baseline_path = getattr(args, "official_baseline", None)
        fail_hard     = getattr(args, "fail_on_baseline_mismatch", False)

        if baseline_path and os.path.isfile(baseline_path):
            ok = check_baseline_match(results, fp, baseline_path, fail_hard,
                                      context=variant_tag)
            print(f"\n  Force-zero result: {'PASS' if ok else 'FAIL'}")
        elif fail_hard:
            raise RuntimeError(
                "--fail_on_baseline_mismatch requires a valid --official_baseline path."
            )
        else:
            print("\n  WARNING: no --official_baseline; cannot verify cross-variant consistency.")

    if check_id:
        if model is None or r2s_np is None:
            raise RuntimeError(
                "--check_init_identity requires a built model and r2s_np mapping. "
                "Pass --super_map and ensure variant args are set correctly."
            )
        print()
        print(f"[eval_only] check_init_identity ...")
        check_init_identity(model, args.val_dir, token_emb_w, r2s_np, device,
                            print_delta_stats=getattr(args, "print_delta_stats", False))


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)

    # Load backbone (frozen — used only for token embeddings + d_model)
    print(f"[main] loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Get token embedding weight
    if hasattr(backbone, "token_emb"):
        token_emb_w = backbone.token_emb.weight.detach().cpu()
    elif hasattr(backbone, "transformer"):
        token_emb_w = backbone.transformer.wte.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot find token embedding in backbone")

    # Early exit: eval-only mode
    if getattr(args, "eval_only", False):
        check_id = getattr(args, "check_init_identity", False)
        if check_id:
            # Need config, r2s, and a freshly built model to test init identity
            cfg_path = os.path.join(args.val_dir, "dataset_config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    ds_cfg = json.load(f)
                n_fine  = ds_cfg.get("n_fine",  128)
                n_super = ds_cfg.get("n_super", 24)
            else:
                n_fine  = args.n_fine
                n_super = args.n_super
            r2s_np_oi = np.zeros(n_fine, dtype=np.int32)
            if args.super_map and os.path.isfile(args.super_map):
                r2s_np_oi = load_r2s(args.super_map, n_fine)
                n_super   = int(r2s_np_oi.max()) + 1
            elif args.variant.upper() == "D3":
                raise RuntimeError("--super_map is required for D3 --check_init_identity")
            r2s_tensor_oi = torch.from_numpy(r2s_np_oi).long().to(device)
            model_oi = build_refiner(args, d_model, n_fine, n_super, r2s_tensor_oi).to(device)
            model_oi.register_buffer("_tok_emb_w", token_emb_w.float().to(device))
            run_eval_only(args, token_emb_w, device, model=model_oi, r2s_np=r2s_np_oi)
        else:
            run_eval_only(args, token_emb_w, device)
        return

    # Training-mode guards
    if not args.train_dir:
        raise RuntimeError("--train_dir is required for training")
    if not args.output_dir:
        raise RuntimeError("--output_dir is required for training")

    # Region config from dataset_config.json
    cfg_path = os.path.join(args.train_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            ds_cfg = json.load(f)
        n_fine  = ds_cfg.get("n_fine",  128)
        n_super = ds_cfg.get("n_super", 24)
        print(f"[main] n_fine={n_fine}  n_super={n_super} (from dataset_config)")
    else:
        n_fine  = args.n_fine
        n_super = args.n_super

    # r2s mapping
    r2s_np = np.zeros(n_fine, dtype=np.int32)
    if args.super_map and os.path.isfile(args.super_map):
        r2s_np = load_r2s(args.super_map, n_fine)
        n_super = int(r2s_np.max()) + 1
        print(f"[main] loaded super_map  n_super={n_super}")
    elif args.variant.upper() == "D3":
        raise RuntimeError("--super_map is required for D3 variant")

    train_variant(args, backbone, d_model, n_fine, n_super, r2s_np, token_emb_w, device)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",    default="",
                   help="Train shard dir. Required for training; not needed for --eval_only.")
    p.add_argument("--val_dir",      required=True)
    p.add_argument("--small_ckpt",   required=True)
    p.add_argument("--output_dir",   default="",
                   help="Checkpoint output dir. Required for training; not needed for --eval_only.")
    p.add_argument("--super_map",    default=None)
    p.add_argument("--variant",      default="D3", choices=["C", "D3"])
    p.add_argument("--d3_size",      default="small", choices=["small", "base"])
    p.add_argument("--d_region",     type=int,   default=32,    help="C only")
    p.add_argument("--d_hidden",     type=int,   default=256,   help="C only")
    p.add_argument("--n_fine",       type=int,   default=128)
    p.add_argument("--n_super",      type=int,   default=24)
    p.add_argument("--steps",        type=int,   default=20_000)
    p.add_argument("--eval_every",   type=int,   default=1_000)
    p.add_argument("--eval_max_batches", type=int, default=500)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--lambda_kl",               type=float, default=0.01)
    p.add_argument("--lambda_delta",            type=float, default=1e-4)
    p.add_argument("--eval_batch_size",         type=int,   default=64,
                   help="Batch size used for all eval passes (force_zero and training eval).")
    p.add_argument("--eval_before_train",       action="store_true",
                   help="Run step-0 force-zero + with-delta eval before training starts.")
    p.add_argument("--eval_only",               action="store_true",
                   help="Run eval only (no training). Requires --force_zero_delta.")
    p.add_argument("--force_zero_delta",        action="store_true",
                   help="In eval_only mode, compute force-zero baseline.")
    p.add_argument("--official_baseline",       default=None,
                   help="Path to saved_candidate_baseline.json for consistency checking.")
    p.add_argument("--fail_on_baseline_mismatch", action="store_true",
                   help="Exit non-zero if force-zero eval does not match official baseline.")
    p.add_argument("--check_init_identity",      action="store_true",
                   help="In eval_only mode, verify model is exact identity at init "
                        "(residual_scale=0 → delta=0 → scores==force_zero).")
    p.add_argument("--print_delta_stats",        action="store_true",
                   help="Print additional delta diagnostics during init identity check.")
    p.add_argument("--device",                  default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
