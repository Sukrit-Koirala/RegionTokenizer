#!/usr/bin/env python3
"""
Bridge-Residual Adapter — tests whether a residual write mechanism helps.

Central hypothesis:
  A small specialist should not patch logits after the decision.
  It should build a path-conditioned residual update before final token decoding.

Architecture:
  Frozen backbone (ReprRegionRetrievalLM).
  BridgeResidualAdapter reads:
    h_layers  — multi-layer backbone hidden states [0, 2, 4, 5, -1] (from pre-extracted features)
    h_prime   — backbone h_prime = h + alpha*region_feat (from candidate shards)
    router/memory region summaries (router_topk_reg/prb, mem_topk_reg/prb, margins)
    candidate metadata for per-region features (cand_tok, cand_fine, cand_mask)
  Builds bridge token sequence:
    [5 layer tokens][router token][memory token][24 fine-region tokens][8 super tokens][PATH token]
  Attends (Pre-LN TransformerEncoder), extracts PATH token output.
  Produces delta_h = out_proj(path_out)   — zero at init (zero-init out_proj)
  Applies:  h_refined = h_prime + alpha * delta_h
  Decodes:  logits_refined = h_refined @ token_embedding.T    (full-vocab, no candidate set)

Primary metric:
  full_vocab_gated_nll_all — CE(logits_refined|gate, logits_base|~gate, gold_token)
    over ALL positions (not covered-only). Avoids the top-M/gold-force trap.

Secondary metrics:
  full_vocab_gated_nll_covered — same but over covered positions only.
  masked_cand_gated_nll        — for direct comparison with prior CTF/MLP refiners.

Key invariants:
  1. No gold leakage.  gold_token, covered, gold_cand_idx, gold_region are NEVER
     passed to the model forward pass.  They are only used to compute loss/metrics
     after logits are produced.
  2. No gold force-inclusion.  There is no candidate selection step.
     gold_force_included_rate = 0.0 always (structurally impossible to be nonzero).
  3. Identity init.  delta_h = 0 at step 0 because out_proj is zero-initialised.
     Step-0 full_vocab_gated_nll_all must equal full_vocab_base_nll_all within tolerance.
     RuntimeError raised if not.
  4. Best checkpoint only saved if full_vocab_gated_nll_all < full_vocab_base_nll_all.
     full_vocab_base_nll_all is computed at step 0 (differs from masked baseline 3.378606).
     Do NOT compare full-vocab NLL to masked-candidate NLL as if they are the same.
  5. Outside-gate positions use base logits exactly.

Canonical eval (fingerprint, coverage, n_examples, n_covered):
  Must match the known-good values across all runs.

Usage:
    python scripts/train_bridge_residual_adapter.py \\
        --small_ckpt  runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --train_cand_dir runs/path_refiner_clean/data/train_hgrid_K24 \\
        --val_cand_dir   runs/path_refiner_clean/data/val_hgrid_K24 \\
        --train_feat_dir runs/path_refiner_residual_interface/features/train_multilayer \\
        --val_feat_dir   runs/path_refiner_residual_interface/features/val_multilayer \\
        --baseline_json  runs/path_refiner_clean/baselines/saved_candidate_baseline.json \\
        --super_map runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --output_dir runs/path_refiner_bridge_adapter/bridge_boundary_v1 \\
        --train_filter boundary --gate_filter boundary \\
        --bridge_dim 256 --num_bridge_layers 2 --num_heads 4 \\
        --top_superregions 8 --top_fine_regions 24 \\
        --steps 5000 --eval_every 1000 --batch_size 32 \\
        --lr 1e-4 --lambda_kl 0.1 --lambda_delta 1e-4 --kl_topk 512 \\
        --amp --eval_before_train --fail_on_baseline_mismatch
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.train_clean_path_refiner import load_r2s, SPLIT_NAMES, TYPE_NAMES
from scripts.train_hard_position_refiner import compute_filter_mask, EVAL_SUBSETS

# ── Canonical invariants ──────────────────────────────────────────────────────

CANONICAL_FINGERPRINT  = "09b0a71955cc9c43"
CANONICAL_NUM_EXAMPLES = 239_362
CANONICAL_NUM_COVERED  = 227_017
CANONICAL_COVERAGE     = 0.948425
MASKED_CAND_BASELINE_NLL = 3.378606   # saved-candidate masked baseline; different from full-vocab


# ── Dataset ───────────────────────────────────────────────────────────────────

class BridgeShardDataset(IterableDataset):
    """
    Streams (candidate_shard, feature_shard) pairs row by row.

    Alignment verified: gold_token in feature shard must equal gold_token in
    candidate shard (RuntimeError on mismatch).

    Yields dicts — no gold in forward-pass fields:
      h_prime       (d_model,)          float32  — h + alpha*region_feat from backbone
      h_layers      (n_layers, d_model) float32  — multi-layer states from feature shard
      cand_tok      (C,)                int64    — candidate token IDs (-1 = padding)
      cand_fine     (C,)                int64    — candidate fine-region IDs
      cand_super    (C,)                int64    — candidate superregion IDs
      cand_mask     (C,)                bool     — True = valid candidate slot
      r_topk_reg    (K,)                int64    — router top-K region IDs
      r_topk_prb    (K,)                float32
      m_topk_reg    (K,)                int64    — memory top-K region IDs
      m_topk_prb    (K,)                float32
      r_margin      ()                  float32
      m_margin      ()                  float32
      gold_token    ()                  int64    — USE ONLY AS CE TARGET AFTER LOGITS
      gold_cand_idx ()                  int64    — USE ONLY FOR MASKED SECONDARY EVAL
      covered       ()                  bool
      filter_mask   ()                  bool     — matches dataset's filter_name
      split         ()                  int64
    """

    def __init__(self, cand_dir: str, feat_dir: str, r2s_np: np.ndarray,
                 filter_name: str, filter_kwargs: Optional[Dict] = None,
                 shuffle: bool = False) -> None:
        self.cand_dir     = cand_dir
        self.feat_dir     = feat_dir
        self.r2s_np       = r2s_np
        self.filter_name  = filter_name
        self.filter_kwargs = filter_kwargs or {}
        self.shuffle      = shuffle
        paths = sorted(glob.glob(os.path.join(cand_dir, "shard_*.pt")))
        if not paths:
            raise RuntimeError(f"No shard_*.pt in {cand_dir}")
        self._paths = paths

    def __iter__(self):
        paths = list(self._paths)
        if self.shuffle:
            import random
            random.shuffle(paths)

        for cand_path in paths:
            shard_idx = int(os.path.basename(cand_path)
                            .replace("shard_", "").replace(".pt", ""))
            feat_path = os.path.join(self.feat_dir, f"shard_{shard_idx:05d}.pt")
            if not os.path.exists(feat_path):
                raise RuntimeError(f"Missing feature shard: {feat_path}")

            cs = torch.load(cand_path, map_location="cpu", weights_only=True)
            fs = torch.load(feat_path,  map_location="cpu", weights_only=True)

            # Alignment guard
            if not torch.equal(cs["gold_token"].int(), fs["gold_token"].int()):
                raise RuntimeError(
                    f"Alignment FAIL: gold_token mismatch in shard {shard_idx:05d}")

            N     = cs["gold_token"].shape[0]
            fmask = compute_filter_mask(cs, self.filter_name, **self.filter_kwargs)

            cf_np     = cs["cand_fine"].numpy().clip(min=0)
            csuper_np = self.r2s_np[cf_np].astype(np.int64)
            csuper_np[cs["cand_fine"].numpy() < 0] = 0
            cand_super_t = torch.from_numpy(csuper_np)

            has_mem = "mem_topk_reg" in cs
            m_reg_t = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
            m_prb_t = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
            m_mar_t = cs["mem_margin"]   if has_mem else torch.zeros(N)

            split_t = cs["split"]  if "split"  in cs else torch.zeros(N, dtype=torch.long)

            for i in range(N):
                yield {
                    "h_prime":      cs["h_prime"][i].float(),
                    "h_layers":     fs["h_layers"][i].float(),
                    "cand_tok":     cs["cand_tok"][i].long(),
                    "cand_fine":    cs["cand_fine"][i].long(),
                    "cand_super":   cand_super_t[i].long(),
                    "cand_mask":    (cs["cand_tok"][i] >= 0),
                    "r_topk_reg":   cs["router_topk_reg"][i].long(),
                    "r_topk_prb":   cs["router_topk_prb"][i].float(),
                    "m_topk_reg":   m_reg_t[i].long(),
                    "m_topk_prb":   m_prb_t[i].float(),
                    "r_margin":     cs["router_margin"][i].float(),
                    "m_margin":     m_mar_t[i].float(),
                    "gold_token":   cs["gold_token"][i].long(),
                    "gold_cand_idx":cs["gold_cand_idx"][i].long(),
                    "covered":      cs["covered"][i].bool(),
                    "filter_mask":  fmask[i].bool(),
                    "split":        split_t[i].long(),
                }


def collate_bridge(batch: List[Dict]) -> Dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class FilteredBridgeShardDataset(IterableDataset):
    """
    Like BridgeShardDataset but yields ONLY rows where filter_mask == True.

    Because every yielded row already satisfies the filter, the training loop
    can set train_mask=None (all rows), giving a full dense batch of hard
    examples.  For batch_size=64 with boundary filter (~17%), the old approach
    gave n_train≈3–9; this approach gives n_train=64.

    Row selection:
        keep = filter_mask
        if train_covered_only:
            keep = keep & covered

    Full-vocab training does NOT require covered=True (gold_token is always in
    the full vocabulary).  Only masked-candidate eval needs covered, and that
    is handled by the eval function using the yielded `covered` field.

    Prints stats on construction (one-time scan of candidate shards only):
        [FilteredBridgeShardDataset]
          filter=boundary  train_covered_only=False
          shards=97  total_rows=1,000,209
          kept_rows=177,353  kept_rate=17.7%
          covered_within_kept=166,729 / 177,353 = 94.0%
    """

    def __init__(self, cand_dir: str, feat_dir: str, r2s_np: np.ndarray,
                 filter_name: str, filter_kwargs: Optional[Dict] = None,
                 train_covered_only: bool = False,
                 shuffle: bool = False,
                 print_stats: bool = True) -> None:
        self.cand_dir           = cand_dir
        self.feat_dir           = feat_dir
        self.r2s_np             = r2s_np
        self.filter_name        = filter_name
        self.filter_kwargs      = filter_kwargs or {}
        self.train_covered_only = train_covered_only
        self.shuffle            = shuffle

        paths = sorted(glob.glob(os.path.join(cand_dir, "shard_*.pt")))
        if not paths:
            raise RuntimeError(f"No shard_*.pt in {cand_dir}")
        self._paths = paths

        if print_stats:
            self._print_stats()

    def _print_stats(self) -> None:
        total_rows = kept_rows = cov_within_kept = 0
        for cand_path in self._paths:
            cs  = torch.load(cand_path, map_location="cpu", weights_only=True)
            N   = cs["gold_token"].shape[0]
            fm  = compute_filter_mask(cs, self.filter_name, **self.filter_kwargs)
            cov = cs["covered"].bool()
            keep = fm & cov if self.train_covered_only else fm
            total_rows      += N
            kept_rows       += int(keep.sum())
            cov_within_kept += int((cov & keep).sum())

        kept_rate = kept_rows / max(total_rows, 1)
        cov_rate  = cov_within_kept / max(kept_rows, 1)
        print(f"[FilteredBridgeShardDataset]")
        print(f"  filter              = {self.filter_name}")
        print(f"  train_covered_only  = {self.train_covered_only}")
        print(f"  shards              = {len(self._paths)}")
        print(f"  total_rows          = {total_rows:,}")
        print(f"  kept_rows           = {kept_rows:,}")
        print(f"  kept_rate           = {kept_rate:.1%}")
        print(f"  covered_within_kept = {cov_within_kept:,} / {kept_rows:,} = {cov_rate:.1%}")

    def __iter__(self):
        paths = list(self._paths)
        if self.shuffle:
            import random
            random.shuffle(paths)

        for cand_path in paths:
            shard_idx = int(os.path.basename(cand_path)
                            .replace("shard_", "").replace(".pt", ""))
            feat_path = os.path.join(self.feat_dir, f"shard_{shard_idx:05d}.pt")
            if not os.path.exists(feat_path):
                raise RuntimeError(f"Missing feature shard: {feat_path}")

            cs = torch.load(cand_path, map_location="cpu", weights_only=True)
            fs = torch.load(feat_path,  map_location="cpu", weights_only=True)

            if not torch.equal(cs["gold_token"].int(), fs["gold_token"].int()):
                raise RuntimeError(
                    f"Alignment FAIL: gold_token mismatch in shard {shard_idx:05d}")

            N    = cs["gold_token"].shape[0]
            fmask = compute_filter_mask(cs, self.filter_name, **self.filter_kwargs)
            cov   = cs["covered"].bool()
            keep  = fmask & cov if self.train_covered_only else fmask

            keep_idx = keep.nonzero(as_tuple=False).squeeze(1)
            if keep_idx.numel() == 0:
                continue
            if self.shuffle:
                keep_idx = keep_idx[torch.randperm(keep_idx.numel())]

            cf_np     = cs["cand_fine"].numpy().clip(min=0)
            csuper_np = self.r2s_np[cf_np].astype(np.int64)
            csuper_np[cs["cand_fine"].numpy() < 0] = 0
            cand_super_t = torch.from_numpy(csuper_np)

            has_mem = "mem_topk_reg" in cs
            m_reg_t = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
            m_prb_t = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
            m_mar_t = cs["mem_margin"]   if has_mem else torch.zeros(N)

            split_t = cs["split"] if "split" in cs else torch.zeros(N, dtype=torch.long)

            for i in keep_idx.tolist():
                yield {
                    "h_prime":      cs["h_prime"][i].float(),
                    "h_layers":     fs["h_layers"][i].float(),
                    "cand_tok":     cs["cand_tok"][i].long(),
                    "cand_fine":    cs["cand_fine"][i].long(),
                    "cand_super":   cand_super_t[i].long(),
                    "cand_mask":    (cs["cand_tok"][i] >= 0),
                    "r_topk_reg":   cs["router_topk_reg"][i].long(),
                    "r_topk_prb":   cs["router_topk_prb"][i].float(),
                    "m_topk_reg":   m_reg_t[i].long(),
                    "m_topk_prb":   m_prb_t[i].float(),
                    "r_margin":     cs["router_margin"][i].float(),
                    "m_margin":     m_mar_t[i].float(),
                    "gold_token":   cs["gold_token"][i].long(),
                    "gold_cand_idx":cs["gold_cand_idx"][i].long(),
                    "covered":      cov[i],
                    "filter_mask":  fmask[i].bool(),   # always True by construction
                    "split":        split_t[i].long(),
                }


# ── Model ─────────────────────────────────────────────────────────────────────

class BridgeResidualAdapter(nn.Module):
    """
    Reads multi-layer backbone states + path evidence, produces delta_h,
    writes delta_h into the residual stream before full-vocab decoding.

    Bridge token sequence (length = n_ctx_layers + 2 + top_fine_k + top_super_k + 1):
      [layer_0, ..., layer_{n-1}]   — projected backbone layer states
      [ROUTER]                      — weighted-pooled router region embedding + scalars
      [MEMORY]                      — weighted-pooled memory region embedding + scalars
      [fine_0, ..., fine_{F-1}]     — per-fine-region tokens (router+memory top-K, NO gold)
      [super_0, ..., super_{S-1}]   — per-superregion tokens (from router top-K, NO gold)
      [PATH]                        — learned query token; its output → delta_h

    Identity invariant: out_proj is zero-initialised → delta_h = 0 at step 0.
    No gold in any input: gold_token/covered/gold_region NEVER passed here.
    """

    def __init__(
        self,
        d_model:       int,
        n_ctx_layers:  int,
        bridge_dim:    int,
        num_heads:     int,
        num_layers:    int,
        ff_mult:       int,
        n_fine:        int,
        n_super:       int,
        top_fine_k:    int,
        top_super_k:   int,
        dropout:       float = 0.0,
        r2s_np:        Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.d_model      = d_model
        self.n_ctx_layers = n_ctx_layers
        self.bridge_dim   = bridge_dim
        self.n_fine       = n_fine
        self.n_super      = n_super
        self.top_fine_k   = top_fine_k
        self.top_super_k  = top_super_k

        # ── Layer-state tokens ────────────────────────────────────────────────
        self.layer_proj   = nn.Linear(d_model, bridge_dim)
        self.layer_id_emb = nn.Embedding(n_ctx_layers, bridge_dim)

        # ── Region embeddings (fine + super, used for region tokens) ──────────
        self.fine_emb  = nn.Embedding(n_fine  + 1, bridge_dim, padding_idx=n_fine)
        self.super_emb = nn.Embedding(n_super + 1, bridge_dim, padding_idx=n_super)

        # ── Router state token: pool(p_k * region_emb_k) + scalar proj ───────
        # Scalars: [entropy, margin, top1_prob]  (3)
        self.router_scalar_proj = nn.Linear(3, bridge_dim)

        # ── Memory state token ────────────────────────────────────────────────
        self.mem_scalar_proj    = nn.Linear(3, bridge_dim)

        # ── Fine-region token scalar features ─────────────────────────────────
        # Per region: [r_prob, m_prob, in_r, in_m, n_cands_frac, max_base_logit, mean_base_logit] (7)
        self.reg_scalar_proj    = nn.Linear(7, bridge_dim)

        # ── Superregion token scalar features ────────────────────────────────
        # Per super: [sum_r_prob, sum_m_prob, n_fine_in_super_frac] (3)
        self.super_scalar_proj  = nn.Linear(3, bridge_dim)

        # ── Learned PATH query token ──────────────────────────────────────────
        self.path_token = nn.Parameter(torch.zeros(1, 1, bridge_dim))
        nn.init.normal_(self.path_token, std=0.02)

        # ── Pre-LN transformer encoder ────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=bridge_dim,
            nhead=num_heads,
            dim_feedforward=bridge_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ── Output: zero-init → delta_h = 0 at init ──────────────────────────
        self.out_proj = nn.Linear(bridge_dim, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Learnable residual scale (starts at 1.0)
        self.alpha = nn.Parameter(torch.ones(1))

        # r2s lookup buffer (fine_region → superregion, needed for super tokens)
        if r2s_np is not None:
            self.register_buffer("r2s_buf",
                                 torch.from_numpy(r2s_np.astype(np.int64)))
        else:
            self.register_buffer("r2s_buf",
                                 torch.zeros(n_fine, dtype=torch.int64))

    # ── Internal builders (all inference-valid, no gold) ─────────────────────

    def _router_token(self, r_reg: torch.Tensor, r_prb: torch.Tensor,
                      r_margin: torch.Tensor) -> torch.Tensor:
        """(B, bridge_dim) — weighted pool of router region embeddings + scalars."""
        r_reg_s  = r_reg.clamp(min=0, max=self.n_fine - 1)
        r_emb    = self.fine_emb(r_reg_s)                              # (B, K, D)
        p_norm   = r_prb / (r_prb.sum(1, keepdim=True) + 1e-8)
        pooled   = (r_emb * p_norm.unsqueeze(-1)).sum(1)               # (B, D)
        entropy  = -(r_prb * torch.log(r_prb + 1e-9)).sum(1)           # (B,)
        top1_prb = r_prb[:, 0]
        scalars  = torch.stack([entropy, r_margin.float(), top1_prb], 1)  # (B, 3)
        return pooled + self.router_scalar_proj(scalars)               # (B, D)

    def _memory_token(self, m_reg: torch.Tensor, m_prb: torch.Tensor,
                      m_margin: torch.Tensor) -> torch.Tensor:
        """(B, bridge_dim) — weighted pool of memory region embeddings + scalars."""
        m_reg_s = m_reg.clamp(min=0, max=self.n_fine - 1)
        m_emb   = self.fine_emb(m_reg_s)
        p_norm  = m_prb / (m_prb.sum(1, keepdim=True) + 1e-8)
        pooled  = (m_emb * p_norm.unsqueeze(-1)).sum(1)
        entropy = -(m_prb * torch.log(m_prb + 1e-9)).sum(1)
        top1_p  = m_prb[:, 0]
        scalars = torch.stack([entropy, m_margin.float(), top1_p], 1)
        return pooled + self.mem_scalar_proj(scalars)

    def _fine_region_tokens(
        self,
        r_reg: torch.Tensor, r_prb: torch.Tensor,
        m_reg: torch.Tensor, m_prb: torch.Tensor,
        cand_tok: torch.Tensor, cand_fine: torch.Tensor,
        cand_mask: torch.Tensor, tok_emb_w: torch.Tensor,
        h_prime: torch.Tensor,
    ) -> torch.Tensor:
        """
        (B, top_fine_k, bridge_dim) — one token per selected fine region.

        Selection (NO GOLD): take top top_fine_k//2 from router and top_fine_k//2
        from memory by their probability ordering.  These are already sorted by
        the KNN scoring, so the first entries are the most probable.
        """
        B      = r_reg.shape[0]
        K_r    = r_reg.shape[1]
        K_m    = m_reg.shape[1]
        half   = self.top_fine_k // 2
        take_r = min(K_r, half)
        take_m = min(K_m, self.top_fine_k - take_r)

        fine_ids = torch.cat(
            [r_reg[:, :take_r], m_reg[:, :take_m]], dim=1
        )  # (B, top_fine_k)
        R = fine_ids.shape[1]
        fine_s = fine_ids.clamp(min=0, max=self.n_fine - 1)

        # Region embedding
        reg_emb = self.fine_emb(fine_s)  # (B, R, bridge_dim)

        # Router prob for each selected region
        match_r = (r_reg.unsqueeze(2) == fine_s.unsqueeze(1))            # (B, K_r, R)
        r_prob  = (r_prb.unsqueeze(2) * match_r.float()).sum(1)          # (B, R)
        in_r    = match_r.any(1).float()                                 # (B, R)

        # Memory prob
        match_m = (m_reg.unsqueeze(2) == fine_s.unsqueeze(1))
        m_prob  = (m_prb.unsqueeze(2) * match_m.float()).sum(1)
        in_m    = match_m.any(1).float()

        # Per-region candidate stats using base logit = h_prime · tok_emb
        emb_w    = tok_emb_w.float()
        tok_e    = F.embedding(cand_tok.clamp(min=0), emb_w)             # (B, C, d)
        base_lgt = (h_prime.float().unsqueeze(1) * tok_e).sum(-1)        # (B, C)
        base_lgt = base_lgt.masked_fill(~cand_mask, float("-inf"))

        # match_cand[b, c, r] = cand_fine[b, c] == fine_s[b, r] and valid
        match_c  = (cand_fine.unsqueeze(2) == fine_s.unsqueeze(1)) & cand_mask.unsqueeze(2)
        n_cands  = match_c.float().sum(1)                                 # (B, R)
        total_cands = cand_mask.float().sum(1, keepdim=True).clamp(min=1)
        n_frac   = n_cands / total_cands                                  # (B, R)

        # Max logit per region (fill inf → 0 for regions with no candidates)
        lgt_exp  = base_lgt.unsqueeze(2).expand(-1, -1, R)               # (B, C, R)
        lgt_exp  = lgt_exp.masked_fill(~match_c, float("-inf"))
        max_lgt  = lgt_exp.amax(1)                                        # (B, R)
        max_lgt  = max_lgt.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # Mean logit per region
        lgt_sum  = lgt_exp.masked_fill(~match_c, 0.0).sum(1)             # (B, R)
        mean_lgt = lgt_sum / n_cands.clamp(min=1)

        scalars  = torch.stack(
            [r_prob, m_prob, in_r, in_m, n_frac, max_lgt, mean_lgt], dim=-1
        )  # (B, R, 7)
        return reg_emb + self.reg_scalar_proj(scalars)                    # (B, R, bridge_dim)

    def _super_tokens(
        self,
        r_reg: torch.Tensor,
        r_prb: torch.Tensor,
        m_reg: torch.Tensor,
        m_prb: torch.Tensor,
    ) -> torch.Tensor:
        """
        (B, top_super_k, bridge_dim) — one token per selected superregion.

        Selection (NO GOLD): map router top-K fine regions to superregions,
        then take the top_super_k most-probable superregions.
        """
        B = r_reg.shape[0]
        K = min(r_reg.shape[1], self.top_super_k * 4)  # oversample, then pick best
        r_reg_s = r_reg[:, :K].clamp(min=0, max=self.n_fine - 1)
        r2s     = self.r2s_buf                                           # (n_fine,)
        sup_ids = r2s[r_reg_s]                                           # (B, K)

        # Aggregate probability by superregion: sum over router top-K
        sup_prb = r_prb[:, :K]                                           # (B, K)

        # For each of top_super_k unique superregions, compute sum prob
        # Simple approach: select the top_super_k superregions with highest
        # cumulative router probability.  Ties broken by index order.
        # Implementation: scatter sum into super_scores[B, n_super]
        sup_scores = torch.zeros(B, self.n_super + 1, device=r_reg.device)
        sup_ids_c  = sup_ids.clamp(min=0, max=self.n_super - 1)
        sup_scores.scatter_add_(1, sup_ids_c, sup_prb)
        # Also add memory probabilities
        K_m    = min(m_reg.shape[1], self.top_super_k * 4)
        m_reg_s = m_reg[:, :K_m].clamp(min=0, max=self.n_fine - 1)
        m_sup   = r2s[m_reg_s]
        m_sup_c = m_sup.clamp(min=0, max=self.n_super - 1)
        m_prb_k = m_prb[:, :K_m]
        sup_scores.scatter_add_(1, m_sup_c, m_prb_k)

        # Select top_super_k
        top_scores, top_super = sup_scores[:, :self.n_super].topk(
            self.top_super_k, dim=-1)                                    # (B, S)
        top_super_s = top_super.clamp(min=0, max=self.n_super - 1)

        super_emb   = self.super_emb(top_super_s)                        # (B, S, bridge_dim)

        # Scalar features: sum_r_prob, sum_m_prob, n_fine_in_super_frac
        # sum_r_prob is top_scores (we aggregated both r and m above — split not clean here,
        # but use top_scores as combined evidence score)
        combined_score = top_scores                                       # (B, S)
        # Count fine regions mapping to each selected superregion in router top-K
        r2s_top        = r2s[r_reg[:, :K].clamp(0, self.n_fine - 1)]   # (B, K)
        match_sup      = (r2s_top.unsqueeze(2) == top_super.unsqueeze(1))# (B, K, S)
        n_fine_in_sup  = match_sup.float().sum(1) / (K + 1e-8)          # (B, S)
        zero_feat      = torch.zeros_like(combined_score)
        scalars        = torch.stack(
            [combined_score, zero_feat, n_fine_in_sup], dim=-1)          # (B, S, 3)
        return super_emb + self.super_scalar_proj(scalars)               # (B, S, bridge_dim)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        h_prime:    torch.Tensor,   # (B, d_model)  — base hidden state (USE as additive base)
        h_layers:   torch.Tensor,   # (B, n_layers, d_model) — multi-layer states
        cand_tok:   torch.Tensor,   # (B, C)  int64
        cand_fine:  torch.Tensor,   # (B, C)  int64
        cand_mask:  torch.Tensor,   # (B, C)  bool
        tok_emb_w:  torch.Tensor,   # (V, d_model)
        r_topk_reg: torch.Tensor,   # (B, K)  int64
        r_topk_prb: torch.Tensor,   # (B, K)  float32
        m_topk_reg: torch.Tensor,   # (B, K)  int64
        m_topk_prb: torch.Tensor,   # (B, K)  float32
        r_margin:   torch.Tensor,   # (B,)    float32
        m_margin:   torch.Tensor,   # (B,)    float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_refined: (B, d_model) — h_prime + alpha * delta_h
          delta_h:   (B, d_model) — zero at init
        """
        B      = h_prime.shape[0]
        device = h_prime.device

        # ── 1. Layer-state tokens ─────────────────────────────────────────────
        layer_t = self.layer_proj(h_layers.float())                      # (B, n_layers, D)
        ids     = torch.arange(self.n_ctx_layers, device=device)
        layer_t = layer_t + self.layer_id_emb(ids).unsqueeze(0)          # broadcast (B, n, D)

        # ── 2. Router token ───────────────────────────────────────────────────
        rt = self._router_token(r_topk_reg, r_topk_prb, r_margin)        # (B, D)

        # ── 3. Memory token ───────────────────────────────────────────────────
        mt = self._memory_token(m_topk_reg, m_topk_prb, m_margin)        # (B, D)

        # ── 4. Fine-region tokens ─────────────────────────────────────────────
        ft = self._fine_region_tokens(
            r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
            cand_tok, cand_fine, cand_mask, tok_emb_w, h_prime,
        )  # (B, top_fine_k, D)

        # ── 5. Superregion tokens ─────────────────────────────────────────────
        st = self._super_tokens(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb)
        # (B, top_super_k, D)

        # ── 6. PATH query token ───────────────────────────────────────────────
        path_tok = self.path_token.expand(B, 1, -1)                      # (B, 1, D)

        # ── 7. Concatenate and attend ─────────────────────────────────────────
        seq = torch.cat(
            [layer_t, rt.unsqueeze(1), mt.unsqueeze(1), ft, st, path_tok], dim=1
        )  # (B, seq_len, bridge_dim)
        seq = self.transformer(seq)                                       # (B, seq_len, D)

        # ── 8. PATH token output → delta_h ────────────────────────────────────
        path_out = seq[:, -1, :]                                          # (B, bridge_dim)
        delta_h  = self.out_proj(path_out)                               # (B, d_model) — zero at init

        # ── 9. Residual update ────────────────────────────────────────────────
        h_refined = h_prime.float() + self.alpha * delta_h

        return h_refined, delta_h


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_bridge_loss(
    model:        BridgeResidualAdapter,
    batch:        Dict,
    device,
    tok_emb_w:    torch.Tensor,
    train_mask:   Optional[torch.Tensor],  # None = all rows; (B,) bool otherwise
    lambda_kl:    float,
    lambda_delta: float,
    kl_topk:      int,
    amp_enabled:  bool = False,
) -> Tuple[Optional[torch.Tensor], Dict]:
    """
    Computes:
      loss_ce    — full-vocab CE on train_mask positions
      loss_kl    — KL(base_topk_probs || refined_topk_logprobs) on train_mask
      loss_delta — L2 norm of delta_h on train_mask
    Returns (total_loss, info_dict).  Returns (None, {}) if no training rows.
    train_mask=None means train on all rows in the batch (use with FilteredBridgeShardDataset).
    """
    B = batch["gold_token"].shape[0]
    if train_mask is None:
        train_mask = torch.ones(B, dtype=torch.bool)
    n_train = int(train_mask.sum())
    if n_train == 0:
        return None, {}

    h_p  = batch["h_prime"].to(device).float()
    h_l  = batch["h_layers"].to(device).float()
    ct   = batch["cand_tok"].to(device)
    cf   = batch["cand_fine"].to(device)
    csu  = batch["cand_super"].to(device)
    cm   = batch["cand_mask"].to(device)
    r_r  = batch["r_topk_reg"].to(device)
    r_p  = batch["r_topk_prb"].to(device)
    m_r  = batch["m_topk_reg"].to(device)
    m_p  = batch["m_topk_prb"].to(device)
    r_ma = batch["r_margin"].to(device)
    m_ma = batch["m_margin"].to(device)
    gt   = batch["gold_token"].to(device).long()      # CE target only, after logits

    emb_w = tok_emb_w.float().to(device)

    h_ref, delta = model(h_p, h_l, ct, cf, cm, emb_w,
                         r_r, r_p, m_r, m_p, r_ma, m_ma)

    # Full-vocab logits (only on train_mask positions to save memory)
    h_ref_sel = h_ref[train_mask]                     # (n_train, d_model)
    h_p_sel   = h_p[train_mask].detach()

    ref_logits  = h_ref_sel.float() @ emb_w.T         # (n_train, V)
    base_logits = h_p_sel.float() @ emb_w.T           # (n_train, V)

    gt_sel = gt[train_mask]
    loss_ce = F.cross_entropy(ref_logits, gt_sel)

    # Top-K KL: KL(base_probs.detach() || refined_log_probs)
    # Over only kl_topk highest-probability tokens in base
    _, topk_idx = base_logits.topk(kl_topk, dim=-1)  # (n_train, kl_topk)
    p_base   = F.softmax(base_logits.gather(1, topk_idx), dim=-1).detach()
    lp_ref   = F.log_softmax(ref_logits.gather(1, topk_idx), dim=-1)
    loss_kl  = (p_base * (torch.log(p_base + 1e-9) - lp_ref)).sum(-1).mean()

    # Delta L2 regulariser
    loss_delta = delta[train_mask].pow(2).mean()

    total = loss_ce + lambda_kl * loss_kl + lambda_delta * loss_delta

    return total, {
        "ce":         loss_ce.item(),
        "kl":         loss_kl.item(),
        "delta_norm": delta[train_mask].norm(dim=-1).mean().item(),
        "n_train":    n_train,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_vocab_eval_bridge(
    model:            BridgeResidualAdapter,
    val_cand_dir:     str,
    val_feat_dir:     str,
    tok_emb_w:        torch.Tensor,
    r2s_np:           np.ndarray,
    device,
    gate_filter_name: str,
    filter_kwargs:    Dict,
    fail_on_mismatch: bool = False,
    eval_batch_size:  int  = 64,
    variant_tag:      str  = "bridge",
) -> Dict:
    """
    Full-vocabulary evaluation over ALL val positions (_all) and covered-only (_covered).

    PRIMARY metrics (suffix _all — every position in the dataset):
      full_vocab_base_nll_all               — CE(h_prime @ emb.T, gold)
      full_vocab_gated_nll_all              — CE(gated_logits, gold)
      full_vocab_gain_all
      full_vocab_inside_gate_base_nll_all   — CE(base,    gold) inside gate
      full_vocab_inside_gate_ref_nll_all    — CE(refined, gold) inside gate
      full_vocab_outside_gate_base_nll_all  — CE(base,    gold) outside gate
      full_vocab_outside_gate_gated_nll_all — CE(gated,   gold) outside gate (must == base)

    SECONDARY metrics (suffix _covered — gold token has a region assignment):
      full_vocab_base_nll_covered, full_vocab_gated_nll_covered, ...

    SECONDARY masked-candidate (covered positions only, needs candidate set):
      masked_cand_base_nll, masked_cand_gated_nll, masked_cand_gain

    gold_force_included_rate is always 0.0 — bridge has no candidate selection step.
    Fingerprint must match CANONICAL_FINGERPRINT; RuntimeError or warning on mismatch.
    """
    model.eval()
    emb_w = tok_emb_w.float().to(device)
    V     = emb_w.shape[0]

    # ── Accumulators: _all (every position) ──────────────────────────────────
    base_ce_all = gate_ce_all = 0.0
    ig_base_all = ig_ref_all  = 0.0
    og_base_all = og_gate_all = 0.0
    nc_all = ig_nc_all = og_nc_all = 0
    base_acc1_all = ref_acc1_all = 0

    # ── Accumulators: _covered (covered positions only) ───────────────────────
    base_ce_cov = gate_ce_cov = 0.0
    ig_base_cov = ig_ref_cov  = 0.0
    og_base_cov = og_gate_cov = 0.0
    nc_cov = ig_nc_cov = og_nc_cov = 0
    base_acc1_cov = ref_acc1_cov = 0

    # ── Masked-candidate (covered only) ──────────────────────────────────────
    mc_base = mc_gated = 0.0
    mc_nc   = 0

    gate_n   = total_n = total_cov = 0
    delta_norms: List[float] = []

    # Fingerprint counters
    sum_cand = sum_gi_cov = sum_gt = 0

    cand_paths = sorted(glob.glob(os.path.join(val_cand_dir, "shard_*.pt")))
    if not cand_paths:
        raise RuntimeError(f"No shard_*.pt in {val_cand_dir}")

    for cand_path in cand_paths:
        si = int(os.path.basename(cand_path).replace("shard_", "").replace(".pt", ""))
        feat_path = os.path.join(val_feat_dir, f"shard_{si:05d}.pt")
        if not os.path.exists(feat_path):
            raise RuntimeError(f"Missing feature shard: {feat_path}")

        cs = torch.load(cand_path, map_location="cpu", weights_only=True)
        fs = torch.load(feat_path,  map_location="cpu", weights_only=True)
        if not torch.equal(cs["gold_token"].int(), fs["gold_token"].int()):
            raise RuntimeError(f"Alignment FAIL in val shard {si:05d}")

        N    = cs["gold_token"].shape[0]
        gate = compute_filter_mask(cs, gate_filter_name, **filter_kwargs)  # (N,) bool

        cf_np    = cs["cand_fine"].numpy().clip(min=0)
        csuper_n = r2s_np[cf_np].astype(np.int64)
        csuper_n[cs["cand_fine"].numpy() < 0] = 0
        cand_sup = torch.from_numpy(csuper_n)

        has_mem  = "mem_topk_reg" in cs
        m_reg_t  = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
        m_prb_t  = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
        m_mar_t  = cs["mem_margin"]   if has_mem else torch.zeros(N)

        for start in range(0, N, eval_batch_size):
            end   = min(start + eval_batch_size, N)
            sl    = slice(start, end)

            h_p   = cs["h_prime"][sl].float().to(device)
            h_l   = fs["h_layers"][sl].float().to(device)
            ct    = cs["cand_tok"][sl].long().to(device)
            cf    = cs["cand_fine"][sl].long().to(device)
            csu   = cand_sup[sl].long().to(device)
            cov   = cs["covered"][sl].bool().to(device)
            g_sl  = gate[sl].to(device)
            gidx  = cs["gold_cand_idx"][sl].long().to(device)
            gt_b  = cs["gold_token"][sl].long().to(device)
            r_reg = cs["router_topk_reg"][sl].long().to(device)
            r_prb = cs["router_topk_prb"][sl].float().to(device)
            m_reg = m_reg_t[sl].long().to(device)
            m_prb = m_prb_t[sl].float().to(device)
            r_mar = cs["router_margin"][sl].float().to(device)
            m_mar = m_mar_t[sl].float().to(device)
            cmask = (ct >= 0)

            # Base and refined full-vocab logits
            base_lgt = h_p.float() @ emb_w.T                            # (B, V)
            h_ref, delta = model(h_p, h_l, ct, cf, cmask, emb_w,
                                 r_reg, r_prb, m_reg, m_prb, r_mar, m_mar)
            ref_lgt  = h_ref.float() @ emb_w.T                          # (B, V)
            delta_norms.extend(delta.norm(dim=-1).tolist())

            # Gated logits: ref where in-gate, base where outside-gate
            gate_exp  = g_sl.unsqueeze(1).expand(-1, V)
            gated_lgt = torch.where(gate_exp, ref_lgt, base_lgt)

            B_b = end - start

            # ── PRIMARY: _all — every position ───────────────────────────────
            base_ce_all += float(F.cross_entropy(base_lgt, gt_b, reduction="sum"))
            gate_ce_all += float(F.cross_entropy(gated_lgt, gt_b, reduction="sum"))
            nc_all      += B_b
            base_acc1_all += int((base_lgt.argmax(1) == gt_b).sum())
            ref_acc1_all  += int((gated_lgt.argmax(1) == gt_b).sum())

            # Inside gate (all)
            ig_all = g_sl
            if ig_all.any():
                ig_gt      = gt_b[ig_all]
                ig_base_all += float(F.cross_entropy(base_lgt[ig_all], ig_gt, reduction="sum"))
                ig_ref_all  += float(F.cross_entropy(ref_lgt[ig_all],  ig_gt, reduction="sum"))
                ig_nc_all   += int(ig_all.sum())

            # Outside gate (all) — gated_lgt should equal base_lgt here
            og_all = ~g_sl
            if og_all.any():
                og_gt      = gt_b[og_all]
                og_base_all += float(F.cross_entropy(base_lgt[og_all], og_gt, reduction="sum"))
                og_gate_all += float(F.cross_entropy(gated_lgt[og_all], og_gt, reduction="sum"))
                og_nc_all   += int(og_all.sum())

            # ── SECONDARY: _covered — covered positions only ──────────────────
            n_cov_b = int(cov.sum())
            if n_cov_b > 0:
                cov_gt       = gt_b[cov]
                base_ce_cov += float(F.cross_entropy(base_lgt[cov], cov_gt, reduction="sum"))
                gate_ce_cov += float(F.cross_entropy(gated_lgt[cov], cov_gt, reduction="sum"))
                nc_cov      += n_cov_b
                base_acc1_cov += int((base_lgt[cov].argmax(1) == cov_gt).sum())
                ref_acc1_cov  += int((gated_lgt[cov].argmax(1) == cov_gt).sum())

                # Inside gate (covered)
                ig_cov = cov & g_sl
                if ig_cov.any():
                    ig_gt_c     = gt_b[ig_cov]
                    ig_base_cov += float(F.cross_entropy(base_lgt[ig_cov], ig_gt_c, reduction="sum"))
                    ig_ref_cov  += float(F.cross_entropy(ref_lgt[ig_cov],  ig_gt_c, reduction="sum"))
                    ig_nc_cov   += int(ig_cov.sum())

                # Outside gate (covered)
                og_cov = cov & ~g_sl
                if og_cov.any():
                    og_gt_c     = gt_b[og_cov]
                    og_base_cov += float(F.cross_entropy(base_lgt[og_cov], og_gt_c, reduction="sum"))
                    og_gate_cov += float(F.cross_entropy(gated_lgt[og_cov], og_gt_c, reduction="sum"))
                    og_nc_cov   += int(og_cov.sum())

                # ── Masked candidate (secondary, covered only) ─────────────────
                cl_base  = base_lgt[cov].gather(1, ct[cov].clamp(min=0))
                cl_base  = cl_base.masked_fill(~cmask[cov], float("-inf"))
                mc_base += float(F.cross_entropy(cl_base, gidx[cov], reduction="sum"))

                cl_gated = gated_lgt[cov].gather(1, ct[cov].clamp(min=0))
                cl_gated = cl_gated.masked_fill(~cmask[cov], float("-inf"))
                mc_gated += float(F.cross_entropy(cl_gated, gidx[cov], reduction="sum"))
                mc_nc   += n_cov_b

                # Fingerprint counters
                sum_cand   += int(cmask.sum())
                sum_gi_cov += int(gidx[cov].sum())
                sum_gt     += int(gt_b[cov].sum())

            gate_n    += int(g_sl.sum())
            total_n   += B_b
            total_cov += n_cov_b

    model.train()

    # Fingerprint
    fp_data = {
        "num_shards":       len(cand_paths),
        "total_n":          total_n,
        "total_cov":        total_cov,
        "sum_cand_counts":  sum_cand,
        "sum_gold_idx_cov": sum_gi_cov,
        "sum_gold_tok":     sum_gt,
    }
    fp = hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()[:16]

    d_mean  = float(np.mean(delta_norms)) if delta_norms else 0.0
    d_max   = float(np.max(delta_norms))  if delta_norms else 0.0
    alpha_v = float(model.alpha.item())

    def _nll(ce: float, n: int) -> float:
        return ce / max(n, 1)

    results = {
        "eval_mode": "bridge_residual_full_vocab",
        # ── PRIMARY: full-vocab _all (every position) ─────────────────────────
        "full_vocab_base_nll_all":               _nll(base_ce_all, nc_all),
        "full_vocab_gated_nll_all":              _nll(gate_ce_all, nc_all),
        "full_vocab_gain_all":                   _nll(base_ce_all, nc_all) - _nll(gate_ce_all, nc_all),
        "full_vocab_inside_gate_base_nll_all":   _nll(ig_base_all, ig_nc_all),
        "full_vocab_inside_gate_ref_nll_all":    _nll(ig_ref_all,  ig_nc_all),
        "full_vocab_inside_gate_gain_all":       _nll(ig_base_all, ig_nc_all) - _nll(ig_ref_all, ig_nc_all),
        "full_vocab_outside_gate_base_nll_all":  _nll(og_base_all, og_nc_all),
        "full_vocab_outside_gate_gated_nll_all": _nll(og_gate_all, og_nc_all),
        "full_vocab_base_acc1_all":              base_acc1_all / max(nc_all, 1),
        "full_vocab_gated_acc1_all":             ref_acc1_all  / max(nc_all, 1),
        # ── SECONDARY: full-vocab _covered (covered positions only) ───────────
        "full_vocab_base_nll_covered":               _nll(base_ce_cov, nc_cov),
        "full_vocab_gated_nll_covered":              _nll(gate_ce_cov, nc_cov),
        "full_vocab_gain_covered":                   _nll(base_ce_cov, nc_cov) - _nll(gate_ce_cov, nc_cov),
        "full_vocab_inside_gate_base_nll_covered":   _nll(ig_base_cov, ig_nc_cov),
        "full_vocab_inside_gate_ref_nll_covered":    _nll(ig_ref_cov,  ig_nc_cov),
        "full_vocab_inside_gate_gain_covered":       _nll(ig_base_cov, ig_nc_cov) - _nll(ig_ref_cov, ig_nc_cov),
        "full_vocab_outside_gate_base_nll_covered":  _nll(og_base_cov, og_nc_cov),
        "full_vocab_outside_gate_gated_nll_covered": _nll(og_gate_cov, og_nc_cov),
        "full_vocab_base_acc1_covered":              base_acc1_cov / max(nc_cov, 1),
        "full_vocab_gated_acc1_covered":             ref_acc1_cov  / max(nc_cov, 1),
        # ── SECONDARY: masked-candidate (covered only) ────────────────────────
        "masked_cand_base_nll":    _nll(mc_base,  mc_nc),
        "masked_cand_gated_nll":   _nll(mc_gated, mc_nc),
        "masked_cand_gain":        _nll(mc_base, mc_nc) - _nll(mc_gated, mc_nc),
        "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL,
        # ── Gate / coverage ───────────────────────────────────────────────────
        "gate_rate":            gate_n    / max(total_n, 1),
        "coverage":             total_cov / max(total_n, 1),
        "num_examples":         total_n,
        "num_covered":          total_cov,
        "inside_gate_n_all":    ig_nc_all,
        "inside_gate_n_cov":    ig_nc_cov,
        "dataset_fingerprint":  fp,
        # ── Delta diagnostics ─────────────────────────────────────────────────
        "alpha":            alpha_v,
        "delta_norm_mean":  d_mean,
        "delta_norm_max":   d_max,
        # ── Safety ────────────────────────────────────────────────────────────
        "gold_force_included_rate": 0.0,
        "selection_mode":           "no_candidate_selection",
        "eval_force_include_gold":  False,
    }

    # Canonical check
    fp_ok  = (fp        == CANONICAL_FINGERPRINT)
    n_ok   = (total_n   == CANONICAL_NUM_EXAMPLES)
    nc_ok  = (total_cov == CANONICAL_NUM_COVERED)
    cov_d  = abs(results["coverage"] - CANONICAL_COVERAGE)
    if not (fp_ok and n_ok and nc_ok and cov_d < 1e-6):
        issues = []
        if not fp_ok:     issues.append(f"fingerprint {fp!r} != {CANONICAL_FINGERPRINT!r}")
        if not n_ok:      issues.append(f"num_examples {total_n} != {CANONICAL_NUM_EXAMPLES}")
        if not nc_ok:     issues.append(f"num_covered {total_cov} != {CANONICAL_NUM_COVERED}")
        if cov_d >= 1e-6: issues.append(f"coverage diff {cov_d:.2e}")
        msg = f"full_vocab_eval_bridge CANONICAL MISMATCH [{variant_tag}]: " + "; ".join(issues)
        if fail_on_mismatch:
            raise RuntimeError(msg)
        print(f"  WARNING: {msg}")

    return results


@torch.no_grad()
def local_subset_eval_bridge(
    model:          BridgeResidualAdapter,
    val_cand_dir:   str,
    val_feat_dir:   str,
    tok_emb_w:      torch.Tensor,
    r2s_np:         np.ndarray,
    device,
    filter_kwargs:  Dict,
    eval_batch_size: int = 64,
) -> List[Dict]:
    """
    Per-subset breakdown: for each EVAL_SUBSETS filter,
    reports base and refined full_vocab NLL + acc@1 on covered positions inside that subset.
    """
    model.eval()
    emb_w = tok_emb_w.float().to(device)
    V     = emb_w.shape[0]

    # Accumulate per-subset stats: subset → {base_ce, ref_ce, base_acc, ref_acc, n}
    stats = defaultdict(lambda: {"base_ce": 0.0, "ref_ce": 0.0,
                                  "base_a1": 0, "ref_a1": 0, "n": 0})

    cand_paths = sorted(glob.glob(os.path.join(val_cand_dir, "shard_*.pt")))
    for cand_path in cand_paths:
        si = int(os.path.basename(cand_path).replace("shard_", "").replace(".pt", ""))
        feat_path = os.path.join(val_feat_dir, f"shard_{si:05d}.pt")
        if not os.path.exists(feat_path):
            continue

        cs = torch.load(cand_path, map_location="cpu", weights_only=True)
        fs = torch.load(feat_path,  map_location="cpu", weights_only=True)
        N  = cs["gold_token"].shape[0]

        # Precompute per-subset masks
        subset_masks = {
            name: compute_filter_mask(cs, name, **filter_kwargs)
            for name in EVAL_SUBSETS
        }

        has_mem  = "mem_topk_reg" in cs
        m_reg_t  = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
        m_prb_t  = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
        m_mar_t  = cs["mem_margin"]   if has_mem else torch.zeros(N)

        cf_np    = cs["cand_fine"].numpy().clip(min=0)
        csuper_n = r2s_np[cf_np].astype(np.int64)
        csuper_n[cs["cand_fine"].numpy() < 0] = 0
        cand_sup = torch.from_numpy(csuper_n)

        for start in range(0, N, eval_batch_size):
            end = min(start + eval_batch_size, N)
            sl  = slice(start, end)

            h_p   = cs["h_prime"][sl].float().to(device)
            h_l   = fs["h_layers"][sl].float().to(device)
            ct    = cs["cand_tok"][sl].long().to(device)
            cf    = cs["cand_fine"][sl].long().to(device)
            csu   = cand_sup[sl].long().to(device)
            cov   = cs["covered"][sl].bool()                # keep on CPU for mask indexing
            gt_b  = cs["gold_token"][sl].long().to(device)
            r_reg = cs["router_topk_reg"][sl].long().to(device)
            r_prb = cs["router_topk_prb"][sl].float().to(device)
            m_reg = m_reg_t[sl].long().to(device)
            m_prb = m_prb_t[sl].float().to(device)
            r_mar = cs["router_margin"][sl].float().to(device)
            m_mar = m_mar_t[sl].float().to(device)
            cmask = (ct >= 0)

            base_lgt = h_p.float() @ emb_w.T
            h_ref, _ = model(h_p, h_l, ct, cf, cmask, emb_w,
                             r_reg, r_prb, m_reg, m_prb, r_mar, m_mar)
            ref_lgt  = h_ref.float() @ emb_w.T

            cov_cpu = cov.numpy().astype(bool)
            for name in EVAL_SUBSETS:
                sub = torch.from_numpy(
                    subset_masks[name][start:end].numpy() & cov_cpu
                ).to(device)
                if not sub.any():
                    continue
                n_sub  = int(sub.sum())
                sub_gt = gt_b[sub]
                s      = stats[name]
                s["base_ce"] += float(F.cross_entropy(base_lgt[sub], sub_gt, reduction="sum"))
                s["ref_ce"]  += float(F.cross_entropy(ref_lgt[sub],  sub_gt, reduction="sum"))
                s["base_a1"] += int((base_lgt[sub].argmax(1) == sub_gt).sum())
                s["ref_a1"]  += int((ref_lgt[sub].argmax(1)  == sub_gt).sum())
                s["n"]       += n_sub

    model.train()

    rows = []
    for name in EVAL_SUBSETS:
        s = stats[name]
        n = max(s["n"], 1)
        rows.append({
            "subset":                name,
            "n":                     s["n"],
            "full_vocab_base_nll":   s["base_ce"] / n,
            "full_vocab_refined_nll":s["ref_ce"]  / n,
            "full_vocab_gain":       s["base_ce"]/n - s["ref_ce"]/n,
            "base_acc1":             s["base_a1"] / n,
            "refined_acc1":          s["ref_a1"]  / n,
        })
    return rows


# ── Training ──────────────────────────────────────────────────────────────────

def train_bridge(args, d_model: int, n_ctx_layers: int, n_fine: int, n_super: int,
                 r2s_np: np.ndarray, tok_emb_w: torch.Tensor, device) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[train] variant               = bridge_residual_adapter")
    print(f"[train] decode_mode           = full_vocab  (primary: _all=all positions, secondary: _covered)")
    print(f"[train] selection_mode        = no_candidate_selection")
    print(f"[train] gold_force_included   = never (structural guarantee)")
    print(f"[train] train_objective       = "
          f"{'filter & covered (--train_covered_only)' if args.train_covered_only else 'filter only — all positions'}")
    print(f"[train] n_ctx_layers          = {n_ctx_layers}")
    print(f"[train] n_fine={n_fine}  n_super={n_super}")

    model = BridgeResidualAdapter(
        d_model       = d_model,
        n_ctx_layers  = n_ctx_layers,
        bridge_dim    = args.bridge_dim,
        num_heads     = args.num_heads,
        num_layers    = args.num_bridge_layers,
        ff_mult       = args.ff_mult,
        n_fine        = n_fine,
        n_super       = n_super,
        top_fine_k    = args.top_fine_regions,
        top_super_k   = args.top_superregions,
        dropout       = args.dropout,
        r2s_np        = r2s_np,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] bridge_dim={args.bridge_dim}  heads={args.num_heads}  "
          f"layers={args.num_bridge_layers}  params={n_params:,}")

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    fail_hard = args.fail_on_baseline_mismatch
    filter_kwargs = {"margin_thresh": args.margin_thresh,
                     "entropy_thresh": args.entropy_thresh}

    # Load official masked-candidate baseline for reference
    masked_bl: Optional[Dict] = None
    if args.baseline_json and os.path.isfile(args.baseline_json):
        with open(args.baseline_json) as f:
            masked_bl = json.load(f)
        print(f"\n[train] Masked-candidate baseline (reference only):")
        print(f"  covered_nll = {masked_bl['covered_nll']:.6f}  "
              f"coverage = {masked_bl.get('coverage', '?')}")
        print(f"  NOTE: full-vocab NLL != masked-candidate NLL. "
              f"A separate full-vocab baseline will be computed at step 0.")

    tok_dev     = tok_emb_w.float().to(device)
    variant_tag = f"bridge-M{args.top_fine_regions}+S{args.top_superregions}"

    # ── Step-0 identity check ─────────────────────────────────────────────────
    full_vocab_base_nll: float = float("nan")

    if args.eval_before_train:
        print(f"\n[train] === step-0 identity check ===")
        print("  Computing full-vocab base NLL (h_prime @ emb.T on covered val positions)...")

        g0 = full_vocab_eval_bridge(
            model, args.val_cand_dir, args.val_feat_dir,
            tok_dev, r2s_np, device,
            gate_filter_name=args.gate_filter,
            filter_kwargs=filter_kwargs,
            fail_on_mismatch=fail_hard,
            eval_batch_size=args.eval_batch_size,
            variant_tag=variant_tag,
        )
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]

        # At step 0, gated == base (delta=0, h_refined=h_prime everywhere)
        nll_diff_all  = abs(g0["full_vocab_gated_nll_all"]     - g0["full_vocab_base_nll_all"])
        nll_diff_cov  = abs(g0["full_vocab_gated_nll_covered"] - g0["full_vocab_base_nll_covered"])
        mc_diff       = abs(g0["masked_cand_gated_nll"]        - g0["masked_cand_base_nll"])
        og_gate_diff  = abs(g0["full_vocab_outside_gate_gated_nll_all"]
                            - g0["full_vocab_outside_gate_base_nll_all"])
        d_max         = g0["delta_norm_max"]

        print(f"  --- PRIMARY (full-vocab, all positions) ---")
        print(f"  full_vocab_base_nll_all    = {g0['full_vocab_base_nll_all']:.6f}")
        print(f"  full_vocab_gated_nll_all   = {g0['full_vocab_gated_nll_all']:.6f}")
        print(f"  diff_all (must be < 1e-3)  = {nll_diff_all:.2e}")
        print(f"  --- SECONDARY (full-vocab, covered only) ---")
        print(f"  full_vocab_base_nll_covered  = {g0['full_vocab_base_nll_covered']:.6f}")
        print(f"  full_vocab_gated_nll_covered = {g0['full_vocab_gated_nll_covered']:.6f}")
        print(f"  diff_covered (must be < 1e-3)= {nll_diff_cov:.2e}")
        print(f"  --- SECONDARY (masked-candidate) ---")
        print(f"  masked_cand_base_nll       = {g0['masked_cand_base_nll']:.6f}  "
              f"(canonical ref = {MASKED_CAND_BASELINE_NLL:.6f})")
        print(f"  masked_cand_diff           = {mc_diff:.2e}  (must be < 1e-3)")
        print(f"  --- Outside-gate invariant ---")
        print(f"  outside_gate_base_nll_all  = {g0['full_vocab_outside_gate_base_nll_all']:.6f}")
        print(f"  outside_gate_gated_nll_all = {g0['full_vocab_outside_gate_gated_nll_all']:.6f}")
        print(f"  og_gate_diff (must be < 1e-5) = {og_gate_diff:.2e}")
        print(f"  --- Model state ---")
        print(f"  delta_norm_max (must be 0) = {d_max:.2e}")
        print(f"  alpha                      = {g0['alpha']:.4f}")

        # Hard assertion: masked_cand_base_nll must match canonical reference
        masked_ref_diff = abs(g0["masked_cand_base_nll"] - MASKED_CAND_BASELINE_NLL)
        if masked_ref_diff > 1e-3:
            raise RuntimeError(
                f"Step-0: masked_cand_base_nll={g0['masked_cand_base_nll']:.6f} "
                f"does not match canonical {MASKED_CAND_BASELINE_NLL:.6f} "
                f"(diff={masked_ref_diff:.2e} > 1e-3). Check dataset alignment.")

        if nll_diff_all >= 1e-3:
            raise RuntimeError(
                f"Step-0 identity FAIL: full_vocab_all gated vs base diff = {nll_diff_all:.2e}. "
                "out_proj must be zero-init.  Check BridgeResidualAdapter.__init__.")
        if nll_diff_cov >= 1e-3:
            raise RuntimeError(
                f"Step-0 identity FAIL: full_vocab_covered gated vs base diff = {nll_diff_cov:.2e}. "
                "out_proj must be zero-init.")
        if mc_diff >= 1e-3:
            raise RuntimeError(
                f"Step-0 identity FAIL: masked_cand gated vs base diff = {mc_diff:.2e}. "
                "out_proj must be zero-init.")
        if og_gate_diff >= 1e-5:
            raise RuntimeError(
                f"Step-0 outside-gate invariant FAIL: gated_nll vs base_nll diff = {og_gate_diff:.2e} > 1e-5. "
                "Gating logic is incorrect — outside-gate positions must use base logits exactly.")
        if d_max > 1e-6:
            raise RuntimeError(
                f"Step-0 identity FAIL: delta_norm_max = {d_max:.2e} > 0. "
                "out_proj not zero-init.")

        print("  [step-0] PASS — identity confirmed on all families, gold_force_included_rate = 0.0000")

        # Save full-vocab base NLL
        with open(os.path.join(args.output_dir, "full_vocab_baseline.json"), "w") as f:
            json.dump({
                "full_vocab_base_nll_all":     full_vocab_base_nll,
                "full_vocab_base_nll_covered": g0["full_vocab_base_nll_covered"],
                "masked_cand_base_nll":        g0["masked_cand_base_nll"],
                "masked_cand_baseline_ref":    MASKED_CAND_BASELINE_NLL,
                "coverage":                    g0["coverage"],
                "dataset_fingerprint":         g0["dataset_fingerprint"],
                "num_examples":                g0["num_examples"],
                "num_covered":                 g0["num_covered"],
                "note": (
                    "full_vocab_base_nll_all is the PRIMARY threshold for best checkpoint. "
                    "full_vocab_base_nll_covered and masked_cand_base_nll are secondary. "
                    "Do NOT compare full-vocab NLL to masked-candidate NLL. "
                    "Best checkpoint saved only if full_vocab_gated_nll_all < full_vocab_base_nll_all."
                ),
            }, f, indent=2)
    else:
        # Without eval_before_train, we still need a baseline for checkpoint gating.
        # Run a quick base-only pass (no model forward) to get full_vocab_base_nll.
        print("[train] Computing full-vocab baseline (no adapter forward) ...")
        # Re-use eval function but with a dummy check
        g0 = full_vocab_eval_bridge(
            model, args.val_cand_dir, args.val_feat_dir,
            tok_dev, r2s_np, device,
            gate_filter_name=args.gate_filter,
            filter_kwargs=filter_kwargs,
            fail_on_mismatch=fail_hard,
            eval_batch_size=args.eval_batch_size,
            variant_tag=variant_tag,
        )
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]
        print(f"  full_vocab_base_nll_all = {full_vocab_base_nll:.6f}")

    if np.isnan(full_vocab_base_nll):
        raise RuntimeError("full_vocab_base_nll is NaN — eval failed.")

    print(f"\n[train] Best checkpoint threshold: full_vocab_gated_nll_all < {full_vocab_base_nll:.6f}")
    print(f"  (If no checkpoint beats this, training is unsuccessful for this variant.)\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    use_filtered = args.use_filtered_train_loader
    grad_accum   = max(1, args.grad_accum_steps)
    eff_bs       = args.batch_size * grad_accum

    print(f"[train] use_filtered_train_loader = {use_filtered}")
    print(f"[train] batch_size                = {args.batch_size}")
    print(f"[train] grad_accum_steps          = {grad_accum}")
    print(f"[train] effective_hard_batch_size = {eff_bs}")
    print(f"[train] train_covered_only        = {args.train_covered_only}")
    print()

    if use_filtered:
        train_ds = FilteredBridgeShardDataset(
            args.train_cand_dir, args.train_feat_dir,
            r2s_np, args.train_filter,
            filter_kwargs=filter_kwargs,
            train_covered_only=args.train_covered_only,
            shuffle=True,
        )
    else:
        train_ds = BridgeShardDataset(
            args.train_cand_dir, args.train_feat_dir,
            r2s_np, args.train_filter,
            filter_kwargs=filter_kwargs, shuffle=True,
        )

    def _infinite():
        while True:
            for batch in DataLoader(train_ds, batch_size=args.batch_size,
                                    collate_fn=collate_bridge, num_workers=0,
                                    drop_last=False):
                yield batch
    train_inf = _infinite()

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler("cuda") if args.amp else None
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)

    # ── Log files ─────────────────────────────────────────────────────────────
    train_log_path = os.path.join(args.output_dir, "train_log.csv")
    eval_log_path  = os.path.join(args.output_dir, "eval_log.csv")
    subset_path    = os.path.join(args.output_dir, "local_subset_eval.csv")

    train_fields = ["step", "ce", "kl", "delta_norm", "n_train", "alpha", "lr"]
    eval_fields  = [
        "step",
        # PRIMARY: full-vocab _all
        "full_vocab_base_nll_all", "full_vocab_gated_nll_all", "full_vocab_gain_all",
        "full_vocab_inside_gate_base_nll_all", "full_vocab_inside_gate_ref_nll_all",
        "full_vocab_inside_gate_gain_all",
        "full_vocab_outside_gate_base_nll_all", "full_vocab_outside_gate_gated_nll_all",
        "full_vocab_base_acc1_all", "full_vocab_gated_acc1_all",
        # SECONDARY: full-vocab _covered
        "full_vocab_base_nll_covered", "full_vocab_gated_nll_covered", "full_vocab_gain_covered",
        "full_vocab_inside_gate_base_nll_covered", "full_vocab_inside_gate_ref_nll_covered",
        "full_vocab_inside_gate_gain_covered",
        "full_vocab_outside_gate_base_nll_covered", "full_vocab_outside_gate_gated_nll_covered",
        # SECONDARY: masked-candidate
        "masked_cand_base_nll", "masked_cand_gated_nll", "masked_cand_gain",
        # META
        "gate_rate", "coverage", "inside_gate_n_all", "inside_gate_n_cov",
        "alpha", "delta_norm_mean", "delta_norm_max",
        "gold_force_included_rate", "dataset_fingerprint",
    ]
    subset_fields = ["step", "subset", "n", "full_vocab_base_nll",
                     "full_vocab_refined_nll", "full_vocab_gain",
                     "base_acc1", "refined_acc1"]

    train_logf = open(train_log_path, "w", newline="")
    eval_logf  = open(eval_log_path,  "w", newline="")
    subset_logf = open(subset_path,   "w", newline="")
    train_csv  = csv.DictWriter(train_logf, fieldnames=train_fields, extrasaction="ignore")
    eval_csv   = csv.DictWriter(eval_logf,  fieldnames=eval_fields,  extrasaction="ignore")
    subset_csv = csv.DictWriter(subset_logf,fieldnames=subset_fields, extrasaction="ignore")
    train_csv.writeheader()
    eval_csv.writeheader()
    subset_csv.writeheader()

    best_path   = os.path.join(args.output_dir, "best_refiner.pt")
    # ⚠ Safety: best_nll starts at full_vocab_base_nll_all.
    # Checkpoint only saved if model strictly improves over base.
    best_nll  = full_vocab_base_nll
    best_step = -1
    ema_ce    = None
    t0        = time.time()
    model.train()
    opt.zero_grad()

    for step in range(1, args.steps + 1):
        accum_infos: List[Dict] = []
        valid_micro = 0

        for _micro in range(grad_accum):
            batch = next(train_inf)
            if use_filtered:
                train_mask = None  # all rows already satisfy filter
            else:
                train_mask = (
                    (batch["filter_mask"] & batch["covered"])
                    if args.train_covered_only
                    else batch["filter_mask"]
                )

            if args.amp:
                with autocast("cuda"):
                    loss, info = compute_bridge_loss(
                        model, batch, device, tok_dev, train_mask,
                        args.lambda_kl, args.lambda_delta, args.kl_topk, amp_enabled=True)
            else:
                loss, info = compute_bridge_loss(
                    model, batch, device, tok_dev, train_mask,
                    args.lambda_kl, args.lambda_delta, args.kl_topk)

            if loss is None:
                continue

            scaled = loss / grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            accum_infos.append(info)
            valid_micro += 1

        if valid_micro == 0:
            continue

        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        sched.step()
        opt.zero_grad()

        avg_ce   = float(np.mean([d["ce"]         for d in accum_infos]))
        avg_kl   = float(np.mean([d["kl"]         for d in accum_infos]))
        avg_dn   = float(np.mean([d["delta_norm"] for d in accum_infos]))
        tot_n    = sum(d["n_train"] for d in accum_infos)
        avg_info = {"ce": avg_ce, "kl": avg_kl, "delta_norm": avg_dn, "n_train": tot_n}

        ema_ce  = avg_ce if ema_ce is None else 0.95 * ema_ce + 0.05 * avg_ce
        alpha_v = float(model.alpha.item())
        lr_now  = sched.get_last_lr()[0]

        train_csv.writerow({"step": step, **avg_info, "alpha": alpha_v, "lr": lr_now})
        if step % 100 == 0:
            train_logf.flush()
            print(f"  step={step:5d}  ema_ce={ema_ce:.4f}  "
                  f"ce={avg_ce:.4f}  kl={avg_kl:.4f}  "
                  f"d_norm={avg_dn:.4f}  alpha={alpha_v:.4f}  "
                  f"n_train={tot_n}  eff_bs={eff_bs}  t={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] step={step} ...")

            g_m = full_vocab_eval_bridge(
                model, args.val_cand_dir, args.val_feat_dir,
                tok_dev, r2s_np, device,
                gate_filter_name=args.gate_filter,
                filter_kwargs=filter_kwargs,
                fail_on_mismatch=fail_hard,
                eval_batch_size=args.eval_batch_size,
                variant_tag=variant_tag,
            )

            gated_nll = g_m["full_vocab_gated_nll_all"]
            fv_gain   = g_m["full_vocab_gain_all"]

            row = {"step": step, **g_m}
            eval_csv.writerow(row)
            eval_logf.flush()

            og_diff = abs(g_m["full_vocab_outside_gate_gated_nll_all"]
                          - g_m["full_vocab_outside_gate_base_nll_all"])
            print(f"  [eval] step={step}")
            print(f"    PRIMARY (full-vocab, all positions):")
            print(f"    full_vocab_base_nll_all        = {g_m['full_vocab_base_nll_all']:.6f}  "
                  f"(baseline at step 0)")
            print(f"    full_vocab_gated_nll_all       = {gated_nll:.6f}  "
                  f"(threshold = {full_vocab_base_nll:.6f})")
            print(f"    full_vocab_gain_all            = {fv_gain:+.6f}")
            print(f"    inside_gate_base_nll_all       = {g_m['full_vocab_inside_gate_base_nll_all']:.6f}")
            print(f"    inside_gate_ref_nll_all        = {g_m['full_vocab_inside_gate_ref_nll_all']:.6f}")
            print(f"    inside_gate_gain_all           = {g_m['full_vocab_inside_gate_gain_all']:+.6f}")
            print(f"    outside_gate_base_nll_all      = {g_m['full_vocab_outside_gate_base_nll_all']:.6f}")
            print(f"    outside_gate_gated_nll_all     = {g_m['full_vocab_outside_gate_gated_nll_all']:.6f}  "
                  f"(diff={og_diff:.2e}, must be ~0)")
            print(f"    SECONDARY (full-vocab, covered only):")
            print(f"    full_vocab_gated_nll_covered   = {g_m['full_vocab_gated_nll_covered']:.6f}  "
                  f"gain={g_m['full_vocab_gain_covered']:+.6f}")
            print(f"    inside_gate_ref_nll_covered    = {g_m['full_vocab_inside_gate_ref_nll_covered']:.6f}  "
                  f"gain={g_m['full_vocab_inside_gate_gain_covered']:+.6f}")
            print(f"    SECONDARY (masked-candidate, covered only):")
            print(f"    masked_cand_base_nll           = {g_m['masked_cand_base_nll']:.6f}  "
                  f"(ref={MASKED_CAND_BASELINE_NLL:.6f})")
            print(f"    masked_cand_gated_nll          = {g_m['masked_cand_gated_nll']:.6f}  "
                  f"gain={g_m['masked_cand_gain']:+.6f}")
            print(f"    alpha={g_m['alpha']:.4f}  "
                  f"delta_mean={g_m['delta_norm_mean']:.4f}  "
                  f"delta_max={g_m['delta_norm_max']:.4f}")
            print(f"    gold_force_included_rate = {g_m['gold_force_included_rate']:.4f}  "
                  f"(must be 0.0000)")

            # Local subset eval
            sub_rows = local_subset_eval_bridge(
                model, args.val_cand_dir, args.val_feat_dir,
                tok_dev, r2s_np, device,
                filter_kwargs=filter_kwargs,
                eval_batch_size=args.eval_batch_size,
            )
            for r in sub_rows:
                subset_csv.writerow({"step": step, **r})
            subset_logf.flush()
            # Print gate subset
            for r in sub_rows:
                if r["subset"] == args.gate_filter:
                    print(f"    [{args.gate_filter}]  n={r['n']}  "
                          f"base_nll={r['full_vocab_base_nll']:.6f}  "
                          f"ref_nll={r['full_vocab_refined_nll']:.6f}  "
                          f"gain={r['full_vocab_gain']:+.6f}")

            # ⚠ Checkpoint only if strictly better than full_vocab_base_nll_all
            if gated_nll < best_nll:
                best_nll  = gated_nll
                best_step = step
                best_metrics = {
                    "variant":                              "bridge_residual_adapter",
                    "selection_mode":                       "no_candidate_selection",
                    "eval_force_include_gold":              False,
                    "gold_force_included_rate":             0.0,
                    "step":                                 step,
                    "train_filter":                         args.train_filter,
                    "gate_filter":                          args.gate_filter,
                    # PRIMARY
                    "full_vocab_base_nll_all":              full_vocab_base_nll,
                    "full_vocab_gated_nll_all":             gated_nll,
                    "full_vocab_gain_all":                  fv_gain,
                    "full_vocab_inside_gate_base_nll_all":  g_m["full_vocab_inside_gate_base_nll_all"],
                    "full_vocab_inside_gate_ref_nll_all":   g_m["full_vocab_inside_gate_ref_nll_all"],
                    "full_vocab_inside_gate_gain_all":      g_m["full_vocab_inside_gate_gain_all"],
                    "full_vocab_outside_gate_base_nll_all": g_m["full_vocab_outside_gate_base_nll_all"],
                    "full_vocab_outside_gate_gated_nll_all":g_m["full_vocab_outside_gate_gated_nll_all"],
                    # SECONDARY covered
                    "full_vocab_base_nll_covered":          g_m["full_vocab_base_nll_covered"],
                    "full_vocab_gated_nll_covered":         g_m["full_vocab_gated_nll_covered"],
                    "full_vocab_gain_covered":              g_m["full_vocab_gain_covered"],
                    "full_vocab_inside_gate_base_nll_covered": g_m["full_vocab_inside_gate_base_nll_covered"],
                    "full_vocab_inside_gate_ref_nll_covered":  g_m["full_vocab_inside_gate_ref_nll_covered"],
                    "full_vocab_inside_gate_gain_covered":     g_m["full_vocab_inside_gate_gain_covered"],
                    # SECONDARY masked
                    "masked_cand_base_nll":                 g_m["masked_cand_base_nll"],
                    "masked_cand_gated_nll":                g_m["masked_cand_gated_nll"],
                    "masked_cand_gain":                     g_m["masked_cand_gain"],
                    "masked_cand_baseline_ref":             MASKED_CAND_BASELINE_NLL,
                    # META
                    "gate_rate":                            g_m["gate_rate"],
                    "coverage":                             g_m["coverage"],
                    "num_examples":                         g_m["num_examples"],
                    "num_covered":                          g_m["num_covered"],
                    "alpha":                                g_m["alpha"],
                    "delta_norm_mean":                      g_m["delta_norm_mean"],
                    "dataset_fingerprint":                  g_m["dataset_fingerprint"],
                    "no_improving_checkpoint":              False,
                }
                torch.save({"step": step, "model": model.state_dict(),
                            "metrics": best_metrics, "args": vars(args)}, best_path)
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"  *** NEW BEST  full_vocab_gated_nll_all={best_nll:.6f}  "
                      f"gain={fv_gain:+.6f}  → {best_path}")
            else:
                print(f"  [no improvement]  gated_nll_all={gated_nll:.6f} >= "
                      f"best={best_nll:.6f}  (baseline={full_vocab_base_nll:.6f})")

    # ── End of training ───────────────────────────────────────────────────────
    torch.save({"step": args.steps, "model": model.state_dict(), "args": vars(args)},
               os.path.join(args.output_dir, "last_refiner.pt"))
    train_logf.close()
    eval_logf.close()
    subset_logf.close()

    final_metrics = {
        "no_improving_checkpoint":       (best_step < 0),
        "best_step":                     best_step,
        "best_full_vocab_gated_nll_all": best_nll if best_step >= 0 else None,
        "full_vocab_base_nll_all":       full_vocab_base_nll,
        "full_vocab_gain_all":           full_vocab_base_nll - best_nll if best_step >= 0 else None,
        "masked_cand_baseline_ref":      MASKED_CAND_BASELINE_NLL,
    }
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    if best_step < 0:
        print(f"\n[train] RESULT: no improving checkpoint found — model never beat "
              f"full_vocab_base_nll_all={full_vocab_base_nll:.6f}. "
              f"Bridge residual mechanism did not help for this variant/filter.")
        # best_refiner.pt does NOT exist
        with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
            json.dump({"no_improving_checkpoint": True,
                       "full_vocab_base_nll_all": full_vocab_base_nll,
                       "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL}, f, indent=2)
    else:
        print(f"\n[train] done  best_full_vocab_gated_nll_all={best_nll:.6f}  "
              f"gain={full_vocab_base_nll-best_nll:+.6f}  step={best_step}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device(args.device)

    print(f"[main] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, _ = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    n_layer = len(backbone.blocks)
    print(f"  d_model={d_model}  n_layer={n_layer}")

    # Token embedding weight (for full-vocab decode)
    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().cpu()
    elif hasattr(backbone, "transformer"):
        tok_emb_w = backbone.transformer.wte.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot locate token embedding in backbone")
    print(f"  tok_emb_w shape = {tuple(tok_emb_w.shape)}  (vocab={tok_emb_w.shape[0]})")

    # n_fine, n_super
    n_fine = 128
    n_super = 24
    cfg_path = os.path.join(args.val_cand_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            ds_cfg = json.load(f)
        n_fine  = ds_cfg.get("n_fine",  128)
        n_super = ds_cfg.get("n_super",  24)

    r2s_np = np.zeros(n_fine, dtype=np.int32)
    if args.super_map and os.path.isfile(args.super_map):
        r2s_np  = load_r2s(args.super_map, n_fine)
        n_super = int(r2s_np.max()) + 1
    print(f"  n_fine={n_fine}  n_super={n_super}")

    # Discover n_ctx_layers from feature config
    feat_cfg_path = os.path.join(args.val_feat_dir, "config.json")
    if not os.path.isfile(feat_cfg_path):
        raise RuntimeError(
            f"Feature config not found: {feat_cfg_path}. "
            "Run build_multilayer_residual_features.py first.")
    with open(feat_cfg_path) as f:
        feat_cfg = json.load(f)
    n_ctx_layers = feat_cfg["n_layers_saved"]
    layer_ids    = feat_cfg["layer_ids"]
    print(f"  n_ctx_layers={n_ctx_layers}  layer_ids={layer_ids}")

    if args.gate_filter is None:
        args.gate_filter = args.train_filter

    train_bridge(args, d_model, n_ctx_layers, n_fine, n_super, r2s_np, tok_emb_w, device)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--small_ckpt",    required=True)
    p.add_argument("--train_cand_dir", required=True)
    p.add_argument("--val_cand_dir",   required=True)
    p.add_argument("--train_feat_dir", required=True)
    p.add_argument("--val_feat_dir",   required=True)
    p.add_argument("--baseline_json",  default=None)
    p.add_argument("--super_map",      default=None)
    p.add_argument("--output_dir",     required=True)
    # Architecture
    p.add_argument("--bridge_dim",        type=int,   default=256)
    p.add_argument("--num_bridge_layers", type=int,   default=2)
    p.add_argument("--num_heads",         type=int,   default=4)
    p.add_argument("--ff_mult",           type=int,   default=4)
    p.add_argument("--dropout",           type=float, default=0.0)
    p.add_argument("--top_fine_regions",  type=int,   default=24)
    p.add_argument("--top_superregions",  type=int,   default=8)
    # Filter
    p.add_argument("--train_filter",   default="boundary")
    p.add_argument("--gate_filter",    default=None,
                   help="Gate filter for eval. Defaults to train_filter.")
    p.add_argument("--margin_thresh",  type=float, default=0.1)
    p.add_argument("--entropy_thresh", type=float, default=2.0)
    # Training
    p.add_argument("--steps",          type=int,   default=5000)
    p.add_argument("--eval_every",     type=int,   default=1000)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--eval_batch_size",type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--lambda_kl",      type=float, default=0.1)
    p.add_argument("--lambda_delta",   type=float, default=1e-4)
    p.add_argument("--kl_topk",        type=int,   default=512,
                   help="Approximate KL over top-K base logits (full 50k is expensive).")
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Gradient accumulation steps. effective_hard_batch_size = batch_size * grad_accum_steps")
    # Flags
    p.add_argument("--amp",                    action="store_true")
    p.add_argument("--eval_before_train",      action="store_true")
    p.add_argument("--fail_on_baseline_mismatch", action="store_true")
    p.add_argument("--use_filtered_train_loader", action="store_true",
                   help="Pre-filter dataset to only yield rows matching train_filter. "
                        "Fills every batch with hard examples (n_train=batch_size). "
                        "Recommended for boundary/hard filters. Use with --batch_size 64.")
    p.add_argument("--train_covered_only",     action="store_true",
                   help="Restrict training to covered positions only (default: train on all filter positions)")
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
