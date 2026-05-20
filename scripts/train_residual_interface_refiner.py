#!/usr/bin/env python3
"""
Residual Interface Refiner — tests whether richer backbone state access helps.

Variants:
  multilayer_crossattn  — candidates cross-attend to per-layer hidden-state tokens.
                          Requires --features_root with pre-extracted layer features.
  region_state_crossattn— candidates cross-attend to region-state tokens built from
                          candidate metadata (no extra feature extraction needed).

Both variants use:
  selection_mode = topm_no_gold  (inference-valid)
  drop_train_gold_not_selected = True  (only train where gold is in top-M)

Safety:
  best checkpoint ONLY saved if gated_covered_nll < official_baseline_nll
  gold_force_included_rate must equal 0.0 in all canonical evals (asserted)
  eval_force_include_gold = False always

Canonical eval requirements:
  fingerprint  = f57cabcdc46d69ce
  num_examples = 239,362
  num_covered  = 227,017
  coverage     = 0.948425

Usage (multilayer):
    python scripts/train_residual_interface_refiner.py \\
        --variant multilayer_crossattn \\
        --small_ckpt  runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --train_dir   runs/path_refiner_clean/data/train_hgrid_K24 \\
        --val_dir     runs/path_refiner_clean/data/val_hgrid_K24 \\
        --features_root runs/path_refiner_residual_interface/features \\
        --baseline_json runs/path_refiner_clean/baselines/saved_candidate_baseline.json \\
        --super_map   runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --output_dir  runs/path_refiner_residual_interface/multilayer_boundary_M256 \\
        --train_filter boundary --gate_filter boundary \\
        --selected_M 256 --refiner_dim 256 --num_layers 1 --num_heads 4 \\
        --steps 5000 --eval_every 1000 \\
        --batch_size 32 --lr 1e-4 --lambda_kl 0.1 \\
        --fail_on_baseline_mismatch --device cuda

Usage (region_state):
    python scripts/train_residual_interface_refiner.py \\
        --variant region_state_crossattn \\
        ... (no --features_root needed) ...
        --top_super_tokens 8 --top_fine_tokens 16
"""

import argparse
import csv
import glob
import hashlib
import json
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
from scripts.train_clean_path_refiner import (
    _aggregate_eval_stats,
    check_baseline_match,
    check_init_identity,
    load_r2s,
    SPLIT_NAMES,
    TYPE_NAMES,
    canonical_eval_refiner,
)
from scripts.train_hard_position_refiner import (
    compute_filter_mask,
    EVAL_SUBSETS,
    FilteredShardStreamDataset,
    run_filter_audit,
)
from scripts.train_candidate_transformer_refiner import (
    _aggregate_eval_stats,
    _build_fine_indicator,
    _build_fine_probs,
    select_candidates_topm,
    gated_global_eval_ctf,
    local_subset_eval_ctf,
)


# ── Shared candidate selection helpers (re-exported from CTF) ─────────────────
# select_candidates_topm is imported above


# ── Dataset — paired candidate + feature shards ───────────────────────────────

class PairedShardStreamDataset(IterableDataset):
    """
    Streams batches from aligned candidate shards + multilayer feature shards.
    Falls back to no feature shard if features_dir is None (region_state variant).
    """
    def __init__(
        self,
        cand_dir:     str,
        r2s_np:       np.ndarray,
        filter_name:  str,
        features_dir: Optional[str] = None,
        filter_kwargs: Optional[Dict] = None,
        shuffle:      bool = True,
    ) -> None:
        self.cand_dir     = cand_dir
        self.r2s_np       = r2s_np
        self.filter_name  = filter_name
        self.features_dir = features_dir
        self.filter_kwargs= filter_kwargs or {}
        self.shuffle      = shuffle

        self.cand_paths = sorted(glob.glob(os.path.join(cand_dir, "shard_*.pt")))
        if not self.cand_paths:
            raise RuntimeError(f"No shard_*.pt in {cand_dir}")

        self.feat_paths: Optional[List[str]] = None
        if features_dir is not None:
            self.feat_paths = sorted(
                glob.glob(os.path.join(features_dir, "shard_*.pt"))
            )
            if len(self.feat_paths) != len(self.cand_paths):
                raise RuntimeError(
                    f"Shard count mismatch: cand={len(self.cand_paths)} "
                    f"feat={len(self.feat_paths)}"
                )

    def __iter__(self) -> Iterator[Dict]:
        indices = list(range(len(self.cand_paths)))
        if self.shuffle:
            random.shuffle(indices)
        for i in indices:
            cand_s = torch.load(
                self.cand_paths[i], map_location="cpu", weights_only=True)
            feat_s = None
            if self.feat_paths is not None:
                feat_s = torch.load(
                    self.feat_paths[i], map_location="cpu", weights_only=True)

            N = len(cand_s["covered"])
            fmask = compute_filter_mask(cand_s, self.filter_name, **self.filter_kwargs)

            cf_np  = cand_s["cand_fine"].numpy().astype(np.int64).clip(min=0)
            cs_np  = self.r2s_np[cf_np].astype(np.int64)
            cs_np[cand_s["cand_fine"].numpy() < 0] = 0
            cs_shard = torch.from_numpy(cs_np)

            for i in range(N):
                if not fmask[i]:
                    continue
                item = {
                    "h_prime":    cand_s["h_prime"][i].float(),
                    "cand_tok":   cand_s["cand_tok"][i].long(),
                    "cand_fine":  cand_s["cand_fine"][i].long(),
                    "cand_super": cs_shard[i].long(),
                    "cand_mask":  (cand_s["cand_tok"][i] >= 0),
                    "gold_idx":   cand_s["gold_cand_idx"][i].long(),
                    "covered":    cand_s["covered"][i].bool(),
                    "r_topk_reg": cand_s["router_topk_reg"][i].long(),
                    "r_topk_prb": cand_s["router_topk_prb"][i].float(),
                    "m_topk_reg": cand_s["mem_topk_reg"][i].long(),
                    "m_topk_prb": cand_s["mem_topk_prb"][i].float(),
                    "r_margin":   cand_s["router_margin"][i].float(),
                    "m_margin":   cand_s["mem_margin"][i].float(),
                }
                if feat_s is not None:
                    item["h_layers"] = feat_s["h_layers"][i].float()  # (n_layers, D)
                yield item


def collate_paired(batch: List[Dict]) -> Dict:
    out: Dict = {}
    for key in batch[0]:
        tensors = [b[key] for b in batch]
        if tensors[0].dim() == 0:
            out[key] = torch.stack(tensors)
        else:
            try:
                out[key] = torch.stack(tensors)
            except RuntimeError:
                out[key] = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=0)
    return out


# ── Model: Multi-layer Cross-Attention Refiner ────────────────────────────────

class MultiLayerCrossAttnRefiner(nn.Module):
    """
    Candidates cross-attend to per-layer backbone hidden-state tokens.

    Context tokens: one per layer, projected from h_layers.
    Candidate tokens: base logit + region embeddings + scalar features.
    Cross-attention: Q=candidates, KV=context tokens (Pre-LN).
    Output: delta per selected candidate, scattered to full candidate dim.

    Robust init: out.weight=zeros, out.bias=zeros, residual_scale=1.0
    → delta=0 at step 0.
    """
    def __init__(
        self,
        d_model:    int,
        n_fine:     int,
        n_super:    int,
        n_ctx_layers: int,          # number of layer-state tokens
        refiner_dim: int  = 256,
        num_heads:   int  = 4,
        ff_mult:     int  = 4,
        dropout:     float= 0.0,
        selected_M:  int  = 256,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.n_fine     = n_fine
        self.n_super    = n_super
        self.refiner_dim = refiner_dim
        self.selected_M  = selected_M
        self.n_ctx_layers = n_ctx_layers

        D = refiner_dim

        # Context token projections (one per layer)
        self.ctx_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ctx_layers)])
        self.ctx_projs = nn.ModuleList([nn.Linear(d_model, D) for _ in range(n_ctx_layers)])

        # Candidate encoding
        self.token_proj = nn.Linear(d_model, D, bias=False)
        self.fine_emb   = nn.Embedding(n_fine  + 1, D, padding_idx=n_fine)
        self.super_emb  = nn.Embedding(n_super + 1, D, padding_idx=n_super)
        # 5 scalar features: [base_logit, r_prob, m_prob, is_r, is_m]
        self.score_proj = nn.Linear(5, D)

        # Pre-LN cross-attention: Q=candidates, KV=context
        self.cand_ln  = nn.LayerNorm(D)
        self.ctx_ln   = nn.LayerNorm(D)
        self.cross_attn = nn.MultiheadAttention(
            D, num_heads, dropout=dropout, batch_first=True
        )

        # Feedforward
        self.ffn_ln = nn.LayerNorm(D)
        self.ffn    = nn.Sequential(
            nn.Linear(D, D * ff_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * ff_mult, D),
        )

        # Output head — zero-init
        self.out            = nn.Linear(D, 1)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

        self._last_sel_stats: Dict = {}
        self._zero_out()

    def _zero_out(self) -> None:
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        for p in self.ctx_projs:
            nn.init.normal_(p.weight, std=0.02)
            nn.init.zeros_(p.bias)
        nn.init.normal_(self.token_proj.weight, std=0.02)
        nn.init.normal_(self.fine_emb.weight,   std=0.02)
        nn.init.normal_(self.super_emb.weight,  std=0.02)
        nn.init.normal_(self.score_proj.weight, std=0.01)
        nn.init.zeros_(self.score_proj.bias)

    def forward(
        self,
        h_prime:      torch.Tensor,  # (B, d_model)
        h_layers:     torch.Tensor,  # (B, n_ctx_layers, d_model) — per-layer states
        cand_tok:     torch.Tensor,  # (B, C)
        cand_fine:    torch.Tensor,  # (B, C)
        cand_super:   torch.Tensor,  # (B, C)
        cand_mask:    torch.Tensor,  # (B, C) bool
        token_emb_w:  torch.Tensor,  # (vocab, d_model)
        r_topk_reg:   torch.Tensor,  # (B, K)
        r_topk_prb:   torch.Tensor,  # (B, K)
        m_topk_reg:   torch.Tensor,  # (B, K)
        m_topk_prb:   torch.Tensor,  # (B, K)
        r_margin:     torch.Tensor,  # (B,)
        m_margin:     torch.Tensor,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C  = cand_tok.shape
        M     = self.selected_M
        D     = self.refiner_dim
        device = h_prime.device
        emb_w  = token_emb_w.float()

        # ── Base scores ──────────────────────────────────────────────────────
        tok_e    = F.embedding(cand_tok.clamp(min=0), emb_w)
        base_raw = (h_prime.float().unsqueeze(1) * tok_e).sum(-1)
        base     = base_raw.masked_fill(~cand_mask, float("-inf"))

        # ── Candidate selection (topm_no_gold) ───────────────────────────────
        sel_idx, sel_mask, _ = select_candidates_topm(
            base, cand_mask, M, mode="topm_no_gold"
        )
        self._last_sel_stats = {
            "gold_forced":     torch.zeros(B, dtype=torch.bool, device=device),
            "sel_valid_count": sel_mask.float().sum(1),
            "sel_idx":         sel_idx.detach(),
        }
        sel_clamped = sel_idx.clamp(min=0)

        # ── Context tokens from layer states ─────────────────────────────────
        ctx_parts = []
        for j in range(self.n_ctx_layers):
            h_j   = self.ctx_norms[j](h_layers[:, j].float())  # (B, D_model)
            ctx_j = self.ctx_projs[j](h_j).unsqueeze(1)        # (B, 1, D)
            ctx_parts.append(ctx_j)
        ctx_seq = torch.cat(ctx_parts, dim=1)                   # (B, n_ctx, D)

        # ── Candidate tokens ─────────────────────────────────────────────────
        sel_tok_id = cand_tok.clamp(min=0).gather(1, sel_clamped)
        sel_fine   = cand_fine.gather(1, sel_clamped)
        sel_sup    = cand_super.gather(1, sel_clamped)
        sel_base   = base_raw.gather(1, sel_clamped)

        p_r   = _build_fine_probs(r_topk_reg, r_topk_prb, self.n_fine)
        p_m   = _build_fine_probs(m_topk_reg, m_topk_prb, self.n_fine)
        in_r  = _build_fine_indicator(r_topk_reg, self.n_fine)
        in_m  = _build_fine_indicator(m_topk_reg, self.n_fine)

        sf_c       = sel_fine.clamp(min=0, max=self.n_fine - 1)
        r_prob_sel = p_r.gather(1, sf_c)
        m_prob_sel = p_m.gather(1, sf_c)
        is_r_sel   = in_r.gather(1, sf_c).float()
        is_m_sel   = in_m.gather(1, sf_c).float()

        score_feat = torch.stack(
            [sel_base, r_prob_sel, m_prob_sel, is_r_sel, is_m_sel], dim=-1
        )

        sel_tok_e  = F.embedding(sel_tok_id, emb_w)
        tok_part   = self.token_proj(sel_tok_e.float())
        fine_part  = self.fine_emb(sel_fine.clamp(min=0, max=self.n_fine))
        super_part = self.super_emb(sel_sup.clamp(min=0, max=self.n_super))
        score_part = self.score_proj(score_feat)

        cand_feat  = tok_part + fine_part + super_part + score_part  # (B, M, D)
        cand_feat  = cand_feat * sel_mask.float().unsqueeze(-1)      # zero padding

        # ── Pre-LN cross-attention ────────────────────────────────────────────
        q = self.cand_ln(cand_feat)
        k = v = self.ctx_ln(ctx_seq)
        attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
        cand_feat   = cand_feat + attn_out

        # ── Feedforward ──────────────────────────────────────────────────────
        cand_feat = cand_feat + self.ffn(self.ffn_ln(cand_feat))

        # ── Output + scatter ─────────────────────────────────────────────────
        delta_sel  = self.out(cand_feat).squeeze(-1) * self.residual_scale
        delta_sel  = delta_sel * sel_mask.float()

        full_delta = torch.zeros(B, C, device=device)
        full_delta.scatter_(1, sel_clamped, delta_sel)

        scores = (base_raw + full_delta).masked_fill(~cand_mask, float("-inf"))
        return scores, base


# ── Model: Region-State Cross-Attention Refiner ───────────────────────────────

class RegionStateCrossAttnRefiner(nn.Module):
    """
    Candidates cross-attend to region-state tokens built from candidate metadata.

    Region-state tokens:
      [CTX]       — h_prime projection
      [SUPER_1..k]— top superregion tokens (router prob + cand statistics)
      [FINE_1..k] — top fine region tokens (router prob + cand statistics)

    No external multi-layer features needed. Tests whether richer
    region-structure representation helps (vs flat region IDs/probs).
    """
    def __init__(
        self,
        d_model:       int,
        n_fine:        int,
        n_super:       int,
        top_super_k:   int  = 8,
        top_fine_k:    int  = 16,
        refiner_dim:   int  = 256,
        num_heads:     int  = 4,
        ff_mult:       int  = 4,
        dropout:       float= 0.0,
        selected_M:    int  = 256,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.n_fine     = n_fine
        self.n_super    = n_super
        self.top_super_k = top_super_k
        self.top_fine_k  = top_fine_k
        self.refiner_dim = refiner_dim
        self.selected_M  = selected_M

        D = refiner_dim
        n_ctx_tokens = 1 + top_super_k + top_fine_k

        # Context token projection from h_prime
        self.ctx_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, D))

        # Region-state token embeddings
        self.super_emb   = nn.Embedding(n_super + 1, D, padding_idx=n_super)
        self.fine_emb    = nn.Embedding(n_fine  + 1, D, padding_idx=n_fine)
        # Scalar feature projection for each region token:
        # [router_prob, mem_prob, is_router, is_mem, n_cands_frac, mean_base_logit]
        self.reg_feat_proj = nn.Linear(6, D)

        # Candidate encoding
        self.token_proj  = nn.Linear(d_model, D, bias=False)
        self.cfine_emb   = nn.Embedding(n_fine  + 1, D, padding_idx=n_fine)
        self.csuper_emb  = nn.Embedding(n_super + 1, D, padding_idx=n_super)
        self.score_proj  = nn.Linear(5, D)

        # Pre-LN cross-attention
        self.cand_ln    = nn.LayerNorm(D)
        self.ctx_ln     = nn.LayerNorm(D)
        self.cross_attn = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)

        # Feedforward
        self.ffn_ln = nn.LayerNorm(D)
        self.ffn    = nn.Sequential(
            nn.Linear(D, D * ff_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * ff_mult, D),
        )

        # Output — zero-init
        self.out            = nn.Linear(D, 1)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

        self._last_sel_stats: Dict = {}
        self._zero_out()

    def _zero_out(self) -> None:
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.super_emb.weight,  std=0.02)
        nn.init.normal_(self.fine_emb.weight,   std=0.02)
        nn.init.normal_(self.cfine_emb.weight,  std=0.02)
        nn.init.normal_(self.csuper_emb.weight, std=0.02)
        nn.init.normal_(self.token_proj.weight, std=0.02)
        nn.init.normal_(self.score_proj.weight, std=0.01)
        nn.init.zeros_(self.score_proj.bias)
        nn.init.normal_(self.reg_feat_proj.weight, std=0.01)
        nn.init.zeros_(self.reg_feat_proj.bias)

    def _build_region_state_tokens(
        self,
        base_raw:   torch.Tensor,   # (B, C)
        cand_mask:  torch.Tensor,   # (B, C) bool
        cand_fine:  torch.Tensor,   # (B, C)  fine region per candidate
        cand_super: torch.Tensor,   # (B, C)
        r_topk_reg: torch.Tensor,   # (B, K)
        r_topk_prb: torch.Tensor,
        m_topk_reg: torch.Tensor,
        m_topk_prb: torch.Tensor,
        h_ctx:      torch.Tensor,   # (B, D)  projected h_prime
        device,
    ) -> torch.Tensor:
        """
        Builds context token sequence:
          [CTX, SUPER_0..K-1, FINE_0..K-1]  (B, 1+super_k+fine_k, D)
        """
        B   = base_raw.shape[0]
        D   = self.refiner_dim
        kS  = self.top_super_k
        kF  = self.top_fine_k

        # CTX token
        ctx_tok = h_ctx.unsqueeze(1)  # (B, 1, D)

        # ── Super-region tokens ───────────────────────────────────────────────
        # Use router top-kS superregions (derived from top-K router regions → super)
        # For simplicity, take top-kS from r_topk_reg mapped to super IDs
        # Aggregate router probabilities to superregion level
        p_r = _build_fine_probs(r_topk_reg, r_topk_prb, self.n_fine)   # (B, n_fine)
        p_m = _build_fine_probs(m_topk_reg, m_topk_prb, self.n_fine)   # (B, n_fine)

        # Super prob = max over children (approx)
        # Build a (B, n_super) by scatter-max from fine probs
        # Use router topK super indices: first kS unique super regions from router
        # (approximate — just use top router regions mapped to super)

        # We need super-level probs. Use same logic as CTF but for super.
        # Here: top-kS from r_topk_reg mapped to their super region
        # This is approximate but sufficient for context tokens
        r_reg_super = r_topk_reg.clamp(min=0).long()  # (B, K)
        # Map fine → super (use a fixed mapping; we don't have it here, so use fine directly)
        # For now, use FINE regions as both — region_state_crossattn will be a close approximation
        # TODO: pass r2s mapping if super token grouping is needed

        # Pad top-kS fine regions as "super tokens" (simpler approximation)
        pad_super = torch.full(
            (B, max(0, kS - r_reg_super.shape[1])), self.n_super,
            dtype=torch.long, device=device
        )
        if r_reg_super.shape[1] >= kS:
            top_super_idx = r_reg_super[:, :kS]
        else:
            top_super_idx = torch.cat([r_reg_super, pad_super], dim=1)  # (B, kS)

        super_tok_emb = self.super_emb(top_super_idx.clamp(max=self.n_super))  # (B, kS, D)

        # Per-superregion features: [router_prob, mem_prob, is_r, is_m, n_cands, mean_base]
        sf = self._per_region_features(
            top_super_idx, base_raw, cand_mask, cand_fine, p_r, p_m,
            _build_fine_indicator(r_topk_reg, self.n_fine),
            _build_fine_indicator(m_topk_reg, self.n_fine),
            device, self.n_super,
        )
        super_toks = super_tok_emb + self.reg_feat_proj(sf)  # (B, kS, D)

        # ── Fine-region tokens ────────────────────────────────────────────────
        # Top-kF fine regions from router
        pad_fine = torch.full(
            (B, max(0, kF - r_topk_reg.shape[1])), self.n_fine,
            dtype=torch.long, device=device
        )
        top_fine_reg = r_topk_reg.clamp(min=0, max=self.n_fine - 1).long()
        if top_fine_reg.shape[1] >= kF:
            top_fine_idx = top_fine_reg[:, :kF]
        else:
            top_fine_idx = torch.cat([top_fine_reg, pad_fine], dim=1)  # (B, kF)

        fine_tok_emb = self.fine_emb(top_fine_idx.clamp(max=self.n_fine))  # (B, kF, D)
        ff = self._per_region_features(
            top_fine_idx, base_raw, cand_mask, cand_fine, p_r, p_m,
            _build_fine_indicator(r_topk_reg, self.n_fine),
            _build_fine_indicator(m_topk_reg, self.n_fine),
            device, self.n_fine,
        )
        fine_toks = fine_tok_emb + self.reg_feat_proj(ff)  # (B, kF, D)

        return torch.cat([ctx_tok, super_toks, fine_toks], dim=1)  # (B, 1+kS+kF, D)

    def _per_region_features(
        self,
        reg_idx:   torch.Tensor,   # (B, K) region indices
        base_raw:  torch.Tensor,   # (B, C)
        cand_mask: torch.Tensor,   # (B, C)
        cand_fine: torch.Tensor,   # (B, C)
        p_r:       torch.Tensor,   # (B, n_fine)
        p_m:       torch.Tensor,   # (B, n_fine)
        in_r:      torch.Tensor,   # (B, n_fine) bool
        in_m:      torch.Tensor,   # (B, n_fine) bool
        device,
        max_reg:   int,
    ) -> torch.Tensor:
        """Returns (B, K, 6) scalar features per region token."""
        B, K = reg_idx.shape
        idx_c = reg_idx.clamp(min=0, max=max_reg - 1)

        r_prb = p_r.gather(1, idx_c)  # (B, K)
        m_prb = p_m.gather(1, idx_c)
        is_r  = in_r.gather(1, idx_c).float()
        is_m  = in_m.gather(1, idx_c).float()

        # n_cands per region (fraction of M)
        # cand_fine matches this region
        c_fine = cand_fine.clamp(min=0, max=max_reg - 1).long()  # (B, C)
        n_cands_list = []
        mean_base_list = []
        for k in range(K):
            r = idx_c[:, k]  # (B,)
            match = (c_fine == r.unsqueeze(1)) & cand_mask  # (B, C)
            n_c   = match.float().sum(1)    / max(self.selected_M, 1)  # (B,)
            b_sum = (base_raw * match).sum(1)
            b_cnt = match.float().sum(1).clamp(min=1)
            mean_b = b_sum / b_cnt                                       # (B,)
            n_cands_list.append(n_c)
            mean_base_list.append(mean_b)

        n_cands   = torch.stack(n_cands_list,   dim=1)  # (B, K)
        mean_base = torch.stack(mean_base_list, dim=1)  # (B, K)

        return torch.stack([r_prb, m_prb, is_r, is_m, n_cands, mean_base], dim=-1)

    def forward(
        self,
        h_prime:    torch.Tensor,
        h_layers:   Optional[torch.Tensor],  # unused in this variant
        cand_tok:   torch.Tensor,
        cand_fine:  torch.Tensor,
        cand_super: torch.Tensor,
        cand_mask:  torch.Tensor,
        token_emb_w: torch.Tensor,
        r_topk_reg: torch.Tensor,
        r_topk_prb: torch.Tensor,
        m_topk_reg: torch.Tensor,
        m_topk_prb: torch.Tensor,
        r_margin:   torch.Tensor,
        m_margin:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C   = cand_tok.shape
        M      = self.selected_M
        D      = self.refiner_dim
        device = h_prime.device
        emb_w  = token_emb_w.float()

        # ── Base scores ──────────────────────────────────────────────────────
        tok_e    = F.embedding(cand_tok.clamp(min=0), emb_w)
        base_raw = (h_prime.float().unsqueeze(1) * tok_e).sum(-1)
        base     = base_raw.masked_fill(~cand_mask, float("-inf"))

        # ── Candidate selection ──────────────────────────────────────────────
        sel_idx, sel_mask, _ = select_candidates_topm(base, cand_mask, M, mode="topm_no_gold")
        self._last_sel_stats = {
            "gold_forced":     torch.zeros(B, dtype=torch.bool, device=device),
            "sel_valid_count": sel_mask.float().sum(1),
            "sel_idx":         sel_idx.detach(),
        }
        sel_clamped = sel_idx.clamp(min=0)

        # ── Context tokens (region-state) ────────────────────────────────────
        h_ctx   = self.ctx_proj(h_prime.float())  # (B, D)
        ctx_seq = self._build_region_state_tokens(
            base_raw, cand_mask, cand_fine, cand_super,
            r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
            h_ctx, device,
        )

        # ── Candidate tokens ─────────────────────────────────────────────────
        sel_tok_id = cand_tok.clamp(min=0).gather(1, sel_clamped)
        sel_fine   = cand_fine.gather(1, sel_clamped)
        sel_sup    = cand_super.gather(1, sel_clamped)
        sel_base   = base_raw.gather(1, sel_clamped)

        p_r = _build_fine_probs(r_topk_reg, r_topk_prb, self.n_fine)
        p_m = _build_fine_probs(m_topk_reg, m_topk_prb, self.n_fine)
        in_r = _build_fine_indicator(r_topk_reg, self.n_fine)
        in_m = _build_fine_indicator(m_topk_reg, self.n_fine)

        sf_c       = sel_fine.clamp(min=0, max=self.n_fine - 1)
        r_prob_sel = p_r.gather(1, sf_c)
        m_prob_sel = p_m.gather(1, sf_c)
        is_r_sel   = in_r.gather(1, sf_c).float()
        is_m_sel   = in_m.gather(1, sf_c).float()
        score_feat = torch.stack([sel_base, r_prob_sel, m_prob_sel, is_r_sel, is_m_sel], dim=-1)

        sel_tok_e  = F.embedding(sel_tok_id, emb_w)
        tok_part   = self.token_proj(sel_tok_e.float())
        fine_part  = self.cfine_emb(sel_fine.clamp(min=0, max=self.n_fine))
        super_part = self.csuper_emb(sel_sup.clamp(min=0, max=self.n_super))
        score_part = self.score_proj(score_feat)

        cand_feat  = tok_part + fine_part + super_part + score_part
        cand_feat  = cand_feat * sel_mask.float().unsqueeze(-1)

        # ── Cross-attention ──────────────────────────────────────────────────
        q = self.cand_ln(cand_feat)
        k = v = self.ctx_ln(ctx_seq)
        attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
        cand_feat   = cand_feat + attn_out
        cand_feat   = cand_feat + self.ffn(self.ffn_ln(cand_feat))

        # ── Output + scatter ─────────────────────────────────────────────────
        delta_sel  = self.out(cand_feat).squeeze(-1) * self.residual_scale
        delta_sel  = delta_sel * sel_mask.float()

        full_delta = torch.zeros(B, C, device=device)
        full_delta.scatter_(1, sel_clamped, delta_sel)

        scores = (base_raw + full_delta).masked_fill(~cand_mask, float("-inf"))
        return scores, base


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_rir_loss(
    model,
    batch: Dict,
    device,
    tok_emb_w: torch.Tensor,
    lambda_kl: float,
    lambda_delta: float,
    drop_gold_not_selected: bool = True,
) -> Tuple[Optional[torch.Tensor], Dict]:
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
    h_layers  = batch.get("h_layers")
    if h_layers is not None:
        h_layers = h_layers.to(device)

    scores, base_sc = model(
        h, h_layers, ct, cf, cs, cmask, tok_emb_w,
        r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
    )

    cov_mask = covered.bool()
    if drop_gold_not_selected:
        sel_idx = model._last_sel_stats.get("sel_idx")
        if sel_idx is not None:
            gold_in_topM = (sel_idx == gold_idx.unsqueeze(1)).any(dim=1)
            cov_mask = cov_mask & gold_in_topM

    n_cov = int(cov_mask.sum())
    if n_cov == 0:
        return None, {}

    sc  = scores[cov_mask]
    bc  = base_sc[cov_mask]
    gi  = gold_idx[cov_mask]

    loss_ce  = F.cross_entropy(sc, gi)
    p_pred   = F.softmax(sc,  dim=-1)
    p_base   = F.softmax(bc.detach(), dim=-1)
    loss_kl  = (p_pred * (torch.log(p_pred + 1e-9) - torch.log(p_base + 1e-9))).sum(-1).mean()
    delta    = (scores - base_sc)[cmask]
    loss_delta = delta.pow(2).mean()

    total = loss_ce + lambda_kl * loss_kl + lambda_delta * loss_delta
    return total, {"ce": loss_ce.item(), "kl": loss_kl.item(), "delta": loss_delta.item()}


# ── Gated eval wrapper (adapts CTF eval for RIR model) ───────────────────────

@torch.no_grad()
def gated_global_eval_rir(
    model,
    val_dir: str,
    val_features_dir: Optional[str],
    tok_emb_w: torch.Tensor,
    r2s_np: np.ndarray,
    device,
    gate_filter_name: str,
    filter_kwargs: Dict,
    official_baseline: Optional[Dict] = None,
    fail_on_mismatch: bool = False,
    variant_tag: str = "",
    eval_batch_size: int = 64,
) -> Dict:
    """
    Full-val gated eval for RIR variants.
    selection_mode=topm_no_gold  eval_force_include_gold=false
    Asserts gold_force_included_rate == 0.
    """
    cand_paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    feat_paths: Optional[List[str]] = None
    if val_features_dir:
        feat_paths = sorted(glob.glob(os.path.join(val_features_dir, "shard_*.pt")))

    emb_w = tok_emb_w.float().to(device)
    model.eval()

    total_n = total_cov = 0
    sum_cand_counts = sum_gold_idx_cov = sum_gold_tok = 0
    gated_ce = gated_nc = 0
    ig_m_ce = ig_b_ce = ig_nc = ig_nt = 0
    og_ce = og_nc = 0
    gold_forced_total = gold_forced_covered = 0

    for si, cand_path in enumerate(cand_paths):
        shard   = torch.load(cand_path, map_location="cpu", weights_only=True)
        N       = len(shard["covered"])
        has_gtk = "gold_token" in shard

        gate_shard = compute_filter_mask(shard, gate_filter_name, **filter_kwargs)

        cf_np    = shard["cand_fine"].numpy().astype(np.int64).clip(min=0)
        cs_np    = r2s_np[cf_np].astype(np.int64)
        cs_np[shard["cand_fine"].numpy() < 0] = 0
        cs_shard = torch.from_numpy(cs_np)

        feat_shard = None
        if feat_paths is not None and si < len(feat_paths):
            feat_shard = torch.load(feat_paths[si], map_location="cpu", weights_only=True)

        for start in range(0, N, eval_batch_size):
            end     = min(start + eval_batch_size, N)
            h       = shard["h_prime"][start:end].float().to(device)
            ct      = shard["cand_tok"][start:end].long().to(device)
            cf      = shard["cand_fine"][start:end].long().to(device)
            cs      = cs_shard[start:end].long().to(device)
            g_idx   = shard["gold_cand_idx"][start:end].long().to(device)
            covered = shard["covered"][start:end].bool().to(device)
            gate    = gate_shard[start:end].to(device)
            B, C    = ct.shape

            cmask   = (ct >= 0)
            tok_e   = F.embedding(ct.clamp(min=0), emb_w)
            base_r  = (h.unsqueeze(1) * tok_e).sum(-1)
            base_sc = base_r.masked_fill(~cmask, float("-inf"))

            h_layers = None
            if feat_shard is not None:
                h_layers = feat_shard["h_layers"][start:end].float().to(device)

            if gate.any():
                r_reg    = shard["router_topk_reg"][start:end].long().to(device)
                r_prb    = shard["router_topk_prb"][start:end].float().to(device)
                m_reg    = shard["mem_topk_reg"][start:end].long().to(device)
                m_prb    = shard["mem_topk_prb"][start:end].float().to(device)
                r_margin = shard["router_margin"][start:end].float().to(device)
                m_margin = shard["mem_margin"][start:end].float().to(device)
                mdl_sc, _ = model(
                    h, h_layers, ct, cf, cs, cmask, emb_w,
                    r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
                )
                gold_forced_total   += int(covered.sum())
                gold_forced_covered += int(model._last_sel_stats["gold_forced"].sum())
            else:
                mdl_sc = base_sc

            gate_3d  = gate.unsqueeze(1).expand(-1, C)
            gated_sc = torch.where(gate_3d, mdl_sc, base_sc)

            n_cov_b = int(covered.sum())
            if n_cov_b > 0:
                ar   = torch.arange(n_cov_b, device=device)
                gi_c = g_idx[covered]
                lp_g = F.log_softmax(gated_sc[covered], dim=-1)
                gated_ce += float(-lp_g[ar, gi_c].sum())
                gated_nc += n_cov_b

                ig_mask = covered & gate
                n_ig    = int(ig_mask.sum())
                if n_ig > 0:
                    ar_ig  = torch.arange(n_ig, device=device)
                    gi_ig  = g_idx[ig_mask]
                    ig_m_ce += float(-F.log_softmax(mdl_sc[ig_mask], dim=-1)[ar_ig, gi_ig].sum())
                    ig_b_ce += float(-F.log_softmax(base_sc[ig_mask], dim=-1)[ar_ig, gi_ig].sum())
                    ig_nc   += n_ig

                og_mask = covered & ~gate
                n_og    = int(og_mask.sum())
                if n_og > 0:
                    ar_og  = torch.arange(n_og, device=device)
                    og_ce += float(-F.log_softmax(base_sc[og_mask], dim=-1)[ar_og, g_idx[og_mask]].sum())
                    og_nc += n_og

                sum_gold_idx_cov += int(gi_c.sum())

            ig_nt     += int(gate.sum())
            total_n   += B
            total_cov += n_cov_b
            sum_cand_counts += int(cmask.sum())
            if has_gtk:
                sum_gold_tok += int(shard["gold_token"][start:end].long().sum())

    model.train()

    fp_data = {
        "num_shards": len(cand_paths), "total_n": total_n, "total_cov": total_cov,
        "sum_cand_counts": sum_cand_counts,
        "sum_gold_idx_cov": sum_gold_idx_cov,
        "sum_gold_tok": sum_gold_tok,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fp_data, sort_keys=True).encode()
    ).hexdigest()[:16]

    results = {
        "gated_covered_nll":        gated_ce  / max(gated_nc, 1),
        "inside_gate_model_nll":    ig_m_ce   / max(ig_nc,    1),
        "inside_gate_base_nll":     ig_b_ce   / max(ig_nc,    1),
        "outside_gate_nll":         og_ce     / max(og_nc,    1),
        "gate_rate":                ig_nt     / max(total_n,  1),
        "covered_gate_rate":        ig_nc     / max(total_cov, 1),
        "coverage":                 total_cov / max(total_n,  1),
        "dataset_fingerprint":      fingerprint,
        "num_examples":             total_n,
        "num_covered":              total_cov,
        "inside_gate_n_total":      ig_nt,
        "inside_gate_n_cov":        ig_nc,
        "gold_force_included_rate": gold_forced_covered / max(gold_forced_total, 1),
        "selection_mode":           "topm_no_gold",
        "eval_force_include_gold":  False,
    }

    gf_rate = results["gold_force_included_rate"]
    if gf_rate > 0.0:
        raise RuntimeError(
            f"gated_global_eval_rir [{variant_tag}]: gold_force_included_rate={gf_rate:.6f} > 0. "
            "Bug: selection_mode=topm_no_gold must never force gold in."
        )

    if official_baseline is not None:
        ref   = official_baseline
        fp_ok = fingerprint == ref.get("dataset_fingerprint", "")
        n_ok  = total_n    == ref.get("num_examples", -1)
        nc_ok = total_cov  == ref.get("num_covered",  -1)
        cd    = abs(results["coverage"] - ref.get("coverage", -1.0))
        ok    = fp_ok and n_ok and nc_ok and cd < 1e-6
        if not ok:
            issues = []
            if not fp_ok: issues.append(f"fingerprint {fingerprint!r}")
            if not n_ok:  issues.append(f"num_examples {total_n}")
            if not nc_ok: issues.append(f"num_covered {total_cov}")
            if cd >= 1e-6: issues.append(f"coverage diff {cd:.2e}")
            msg = (f"gated_global_eval_rir MISMATCH [{variant_tag}]: "
                   + ", ".join(issues))
            if fail_on_mismatch:
                raise RuntimeError(msg)
            print(f"  WARNING: {msg}")

    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train_rir(args, d_model, n_fine, n_super, n_ctx_layers, r2s_np, tok_emb_w, device):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Safety guards ─────────────────────────────────────────────────────────
    print(f"[train] variant               = {args.variant}")
    print(f"[train] selection_mode        = topm_no_gold")
    print(f"[train] eval_force_include_gold= false")
    print(f"[train] drop_train_gold_not_selected = True")

    # Build model
    if args.variant == "multilayer_crossattn":
        model = MultiLayerCrossAttnRefiner(
            d_model=d_model, n_fine=n_fine, n_super=n_super,
            n_ctx_layers=n_ctx_layers,
            refiner_dim=args.refiner_dim,
            num_heads=args.num_heads,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
            selected_M=args.selected_M,
        )
    elif args.variant == "region_state_crossattn":
        model = RegionStateCrossAttnRefiner(
            d_model=d_model, n_fine=n_fine, n_super=n_super,
            top_super_k=args.top_super_tokens,
            top_fine_k=args.top_fine_tokens,
            refiner_dim=args.refiner_dim,
            num_heads=args.num_heads,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
            selected_M=args.selected_M,
        )
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    model = model.to(device)
    model.register_buffer("_tok_emb_w", tok_emb_w.float().to(device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params={n_params:,}  M={args.selected_M}  D={args.refiner_dim}")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    fail_hard    = getattr(args, "fail_on_baseline_mismatch", False)
    official_bl: Optional[Dict] = None
    if args.baseline_json and os.path.isfile(args.baseline_json):
        with open(args.baseline_json) as f:
            official_bl = json.load(f)
        bl_nll = official_bl["covered_nll"]
        print(f"\n=== CANONICAL EVAL  baseline_nll={bl_nll:.6f} ===")
        print(f"  Best checkpoint only saved if gated_nll < {bl_nll:.6f}")
        if fail_hard:
            print("  --fail_on_baseline_mismatch active")
        print()
    else:
        bl_nll = float("nan")

    filter_kwargs = {"margin_thresh": args.margin_thresh,
                     "entropy_thresh": args.entropy_thresh}
    variant_tag   = f"RIR-{args.variant}-M{args.selected_M}/{args.train_filter}"

    val_features_dir  = None
    train_features_dir = None
    if args.variant == "multilayer_crossattn" and args.features_root:
        val_features_dir   = os.path.join(args.features_root, "val_multilayer")
        train_features_dir = os.path.join(args.features_root, "train_multilayer")
        for d in [val_features_dir, train_features_dir]:
            if not os.path.isdir(d):
                raise RuntimeError(f"Features directory not found: {d}. "
                                   "Run build_multilayer_residual_features.py first.")

    run_filter_audit(args.train_dir, args.val_dir, filter_kwargs, args.output_dir)

    # ── Step-0 identity check ─────────────────────────────────────────────────
    if getattr(args, "eval_before_train", False):
        tok_dev = model._tok_emb_w
        print("\n[train] === step-0 identity check ===")
        print("  [force_zero] full val ...")
        m0_zero = canonical_eval_refiner(
            None, args.val_dir, tok_dev, r2s_np, device,
            force_zero=True, eval_batch_size=args.eval_batch_size,
            official_baseline=official_bl, fail_on_mismatch=fail_hard,
            variant_tag=variant_tag,
        )
        fp0 = m0_zero["dataset_fingerprint"]
        print(f"  force_zero  nll={m0_zero['covered_nll']:.6f}  fp={fp0}")
        if official_bl:
            check_baseline_match(m0_zero, official_bl, fail_hard,
                                 context=f"force_zero/{variant_tag}", check_nll=True)

        # Identity check via step-0 gated eval
        print("  [step 0 / gated] full val (selection_mode=topm_no_gold) ...")
        g0 = gated_global_eval_rir(
            model, args.val_dir, val_features_dir, tok_dev, r2s_np, device,
            gate_filter_name=args.gate_filter,
            filter_kwargs=filter_kwargs,
            official_baseline=official_bl,
            fail_on_mismatch=fail_hard,
            variant_tag=variant_tag,
            eval_batch_size=args.eval_batch_size,
        )
        nll_diff0 = abs(g0["gated_covered_nll"] - m0_zero["covered_nll"])
        fp_ok0    = g0["dataset_fingerprint"] == fp0
        print(f"  gated_nll={g0['gated_covered_nll']:.6f}  "
              f"force_zero_nll={m0_zero['covered_nll']:.6f}  diff={nll_diff0:.2e}")
        print(f"  gold_force_included_rate = {g0['gold_force_included_rate']:.4f}"
              "  (must be 0.0000)")
        if nll_diff0 >= 1e-3 or not fp_ok0:
            raise RuntimeError(
                f"Step-0 identity FAIL: nll_diff={nll_diff0:.2e}  fp_ok={fp_ok0}"
            )
        print("  [step 0] PASS")
        model.train()

    # ── Dataset + optimiser ───────────────────────────────────────────────────
    train_ds = PairedShardStreamDataset(
        args.train_dir, r2s_np, args.train_filter,
        features_dir=train_features_dir,
        filter_kwargs=filter_kwargs, shuffle=True,
    )
    def _infinite():
        while True:
            for batch in DataLoader(train_ds, batch_size=args.batch_size,
                                    collate_fn=collate_paired, num_workers=0,
                                    drop_last=False):
                yield batch
    train_inf = _infinite()

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler("cuda")
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)

    log_path   = os.path.join(args.output_dir, "train_log.csv")
    log_fields = ["step", "ce", "kl", "delta",
                  "gated_covered_nll", "inside_gate_model_nll", "inside_gate_base_nll",
                  "outside_gate_nll", "gate_rate", "covered_gate_rate",
                  "coverage", "dataset_fingerprint", "gold_force_included_rate",
                  "residual_scale", "delta_vs_baseline"]
    log_file = open(log_path, "w", newline="")
    log_csv  = csv.DictWriter(log_file, fieldnames=log_fields, extrasaction="ignore")
    log_csv.writeheader()

    best_path = os.path.join(args.output_dir, "best_refiner.pt")
    # ⚠ Safety: best_nll starts at baseline_nll (not +inf).
    # Checkpoints only saved if model actually beats the baseline.
    best_nll  = bl_nll if not np.isnan(bl_nll) else float("inf")
    best_step = -1
    ema_ce    = None
    t0        = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        batch = next(train_inf)

        with autocast("cuda"):
            loss, info = compute_rir_loss(
                model, batch, device, model._tok_emb_w,
                args.lambda_kl, args.lambda_delta,
                drop_gold_not_selected=True,
            )
        if loss is None:
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()

        ema_ce = info["ce"] if ema_ce is None else 0.98 * ema_ce + 0.02 * info["ce"]

        if step % 100 == 0:
            rscale = float(model.residual_scale)
            print(f"  step {step:6d}/{args.steps}  ce={ema_ce:.4f}  "
                  f"kl={info['kl']:.4f}  scale={rscale:.4f}  "
                  f"t={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] step={step} ...")

            g_m = gated_global_eval_rir(
                model, args.val_dir, val_features_dir,
                model._tok_emb_w, r2s_np, device,
                gate_filter_name=args.gate_filter,
                filter_kwargs=filter_kwargs,
                official_baseline=official_bl,
                fail_on_mismatch=fail_hard,
                variant_tag=variant_tag,
                eval_batch_size=args.eval_batch_size,
            )

            rscale      = float(model.residual_scale)
            gated_nll   = g_m["gated_covered_nll"]
            delta_vs_bl = bl_nll - gated_nll

            row = {"step": step, **info, **g_m,
                   "residual_scale": rscale, "delta_vs_baseline": delta_vs_bl}
            log_csv.writerow(row)
            log_file.flush()

            print(f"  [eval/gated] step={step}")
            print(f"    gated_covered_nll     = {gated_nll:.6f}  "
                  f"(baseline={bl_nll:.6f}  delta={delta_vs_bl:+.6f})")
            print(f"    inside_gate_model_nll = {g_m['inside_gate_model_nll']:.6f}")
            print(f"    inside_gate_base_nll  = {g_m['inside_gate_base_nll']:.6f}")
            print(f"    gate_rate             = {g_m['gate_rate']:.4f}")
            print(f"    gold_force_incl_rate  = {g_m['gold_force_included_rate']:.4f}")
            print(f"    residual_scale        = {rscale:.4f}")

            # ⚠ Only save if model beats baseline (not just best seen so far)
            if gated_nll < best_nll:
                best_nll  = gated_nll
                best_step = step
                best_metrics = {
                    "variant":                 args.variant,
                    "selection_mode":          "topm_no_gold",
                    "eval_force_include_gold": False,
                    "step":                    step,
                    "train_filter":            args.train_filter,
                    "gate_filter":             args.gate_filter,
                    "selected_M":              args.selected_M,
                    "official_baseline_nll":   bl_nll,
                    "gated_covered_nll":       gated_nll,
                    "delta_vs_baseline":       delta_vs_bl,
                    "gold_force_included_rate": g_m["gold_force_included_rate"],
                    **{k: g_m[k] for k in [
                        "inside_gate_model_nll", "inside_gate_base_nll",
                        "outside_gate_nll", "gate_rate", "covered_gate_rate",
                        "coverage", "dataset_fingerprint", "num_examples", "num_covered",
                    ]},
                    "residual_scale": rscale,
                }
                torch.save({"step": step, "model": model.state_dict(),
                            "metrics": best_metrics, "args": vars(args)}, best_path)
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"  *** NEW BEST  gated_nll={best_nll:.6f}  "
                      f"delta={delta_vs_bl:+.6f}  → {best_path}")
            else:
                print(f"  [eval] gated_nll={gated_nll:.6f} >= best={best_nll:.6f}  "
                      f"(baseline={bl_nll:.6f})  — checkpoint NOT saved")

    torch.save({"step": args.steps, "model": model.state_dict(), "args": vars(args)},
               os.path.join(args.output_dir, "last_refiner.pt"))
    log_file.close()

    if best_step < 0:
        print(f"\n[train] RESULT: no improving checkpoint found — model never beat baseline "
              f"({bl_nll:.6f}). The refiner interface does not help for {args.variant}.")
    else:
        print(f"\n[train] done  best_gated_nll={best_nll:.6f}  "
              f"delta={bl_nll-best_nll:+.6f}  step={best_step}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device(args.device)

    print(f"[main] loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, _ = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().cpu()
    elif hasattr(backbone, "transformer"):
        tok_emb_w = backbone.transformer.wte.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot locate token embedding in backbone")

    n_fine = n_super = 128
    cfg_path = os.path.join(args.val_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            ds_cfg = json.load(f)
        n_fine  = ds_cfg.get("n_fine",  128)
        n_super = ds_cfg.get("n_super",  24)

    r2s_np = np.zeros(n_fine, dtype=np.int32)
    if args.super_map and os.path.isfile(args.super_map):
        r2s_np  = load_r2s(args.super_map, n_fine)
        n_super = int(r2s_np.max()) + 1

    # Discover layer count for multilayer variant
    n_layer = len(backbone.blocks)
    n_ctx_layers = 0
    if args.variant == "multilayer_crossattn" and args.features_root:
        feat_cfg_path = os.path.join(args.features_root, "val_multilayer", "config.json")
        if os.path.isfile(feat_cfg_path):
            with open(feat_cfg_path) as f:
                feat_cfg = json.load(f)
            n_ctx_layers = feat_cfg["n_layers_saved"]
            print(f"[main] n_ctx_layers={n_ctx_layers} (from feature config)")
        else:
            n_ctx_layers = n_layer + 1  # fallback: all blocks + h_final
    elif args.variant == "region_state_crossattn":
        n_ctx_layers = 1 + args.top_super_tokens + args.top_fine_tokens

    if args.gate_filter is None:
        args.gate_filter = args.train_filter

    train_rir(args, d_model, n_fine, n_super, n_ctx_layers, r2s_np, tok_emb_w, device)


def _parse():
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--small_ckpt",    required=True)
    p.add_argument("--train_dir",     required=True)
    p.add_argument("--val_dir",       required=True)
    p.add_argument("--super_map",     default=None)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--baseline_json", default=None)
    p.add_argument("--features_root", default=None,
                   help="Root of pre-extracted features. Required for multilayer_crossattn.")
    # Variant
    p.add_argument("--variant", default="multilayer_crossattn",
                   choices=["multilayer_crossattn", "region_state_crossattn"])
    # Architecture
    p.add_argument("--selected_M",       type=int,   default=256)
    p.add_argument("--refiner_dim",      type=int,   default=256)
    p.add_argument("--num_heads",        type=int,   default=4)
    p.add_argument("--ff_mult",          type=int,   default=4)
    p.add_argument("--dropout",          type=float, default=0.0)
    p.add_argument("--top_super_tokens", type=int,   default=8,
                   help="Number of superregion context tokens (region_state variant).")
    p.add_argument("--top_fine_tokens",  type=int,   default=16,
                   help="Number of fine-region context tokens (region_state variant).")
    # This field is called num_layers in the CLI for consistency but unused directly
    p.add_argument("--num_layers",       type=int,   default=1,
                   help="Transformer layers in the refiner (currently 1 cross-attn block).")
    p.add_argument("--n_fine",           type=int,   default=128)
    p.add_argument("--n_super",          type=int,   default=24)
    # Filter
    p.add_argument("--train_filter",     default="boundary")
    p.add_argument("--gate_filter",      default=None)
    p.add_argument("--margin_thresh",    type=float, default=0.1)
    p.add_argument("--entropy_thresh",   type=float, default=2.0)
    # Training
    p.add_argument("--steps",            type=int,   default=5000)
    p.add_argument("--eval_every",       type=int,   default=1000)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--eval_batch_size",  type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--lambda_kl",        type=float, default=0.1)
    p.add_argument("--lambda_delta",     type=float, default=1e-4)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    # Flags
    p.add_argument("--eval_before_train",         action="store_true")
    p.add_argument("--fail_on_baseline_mismatch", action="store_true")
    p.add_argument("--device",           default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
