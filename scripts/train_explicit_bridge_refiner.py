#!/usr/bin/env python3
"""
Explicit Bridge + Refiner — separates evidence translation from correction.

Pipeline:
  [RegionTransformer outputs]
      → Bridge (translates messy evidence → clean path-state tokens)
      → Refiner Transformer ([RESIDUAL + path tokens] → delta_h)
      → Write delta to h_prime
      → Full-vocab decoder

Role separation:
  Bridge  = translator: messy region/residual/memory evidence → path-state tokens
  Refiner = thinker:    RESIDUAL_TOKEN + path tokens → specialized computation → delta_h
  Decoder = decision:   (h_prime + alpha * delta_h) @ emb.T

Identity invariant:
  out_proj in Refiner is zero-initialized → delta_h=0 at step 0 → same as base.

No gold in Bridge or Refiner inputs.
gold_force_included_rate = 0.0 structurally (no candidate selection step).

Usage:
    python scripts/train_explicit_bridge_refiner.py \\
        --small_ckpt  runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --train_cand_dir runs/path_refiner_clean/data/train_hgrid_K24 \\
        --val_cand_dir   runs/path_refiner_clean/data/val_hgrid_K24 \\
        --train_feat_dir runs/path_refiner_residual_interface/features/train_multilayer \\
        --val_feat_dir   runs/path_refiner_residual_interface/features/val_multilayer \\
        --baseline_json  runs/path_refiner_clean/baselines/saved_candidate_baseline.json \\
        --super_map runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --output_dir runs/path_refiner_explicit_bridge_refiner/boundary_insert4_refiner2_path4_v1 \\
        --insert_after_block 4 --num_path_tokens 4 \\
        --bridge_dim 256 --bridge_layers 1 --bridge_heads 4 \\
        --refiner_dim 256 --refiner_layers 2 --refiner_heads 4 \\
        --train_filter boundary --gate_filter boundary \\
        --use_filtered_train_loader \\
        --batch_size 32 --grad_accum_steps 2 --steps 5000 --eval_every 1000 \\
        --lr 5e-5 --lambda_kl 0.2 --lambda_delta 3e-4 --lambda_path 0.0 \\
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
from torch.utils.data import DataLoader

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.train_clean_path_refiner import load_r2s
from scripts.train_hard_position_refiner import compute_filter_mask, EVAL_SUBSETS
from scripts.train_bridge_residual_adapter import (
    CANONICAL_FINGERPRINT, CANONICAL_NUM_EXAMPLES, CANONICAL_NUM_COVERED,
    CANONICAL_COVERAGE, MASKED_CAND_BASELINE_NLL,
    BridgeShardDataset, FilteredBridgeShardDataset, collate_bridge,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_insert_layer_idx(insert_after_block, layer_ids: List[int]) -> int:
    if insert_after_block in ("final", -1, "-1"):
        try:
            return layer_ids.index(-1)
        except ValueError:
            return len(layer_ids) - 1
    iab = int(insert_after_block)
    if iab in layer_ids:
        return layer_ids.index(iab)
    candidates = [(i, abs(lid - iab)) for i, lid in enumerate(layer_ids) if lid != -1]
    if not candidates:
        return len(layer_ids) - 1
    best_i = min(candidates, key=lambda x: x[1])[0]
    print(f"  WARNING: insert_after_block={iab} not in layer_ids={layer_ids}. "
          f"Using closest: layer_ids[{best_i}]={layer_ids[best_i]}")
    return best_i


# ── Bridge module ─────────────────────────────────────────────────────────────

class ExplicitBridge(nn.Module):
    """
    Translates region/residual/memory evidence into path-state tokens.

    Bridge sequence:
      [INSERT_STATE]             — layer_proj(h_insert) + insert_marker_emb
      [layer_0, ..., layer_n-1]  — all backbone layer states with positional embeddings
      [ROUTER]                   — weighted-pool router region emb + scalars
      [MEMORY]                   — weighted-pool memory region emb + scalars
      [fine_0..fine_{F-1}]       — per-fine-region scalar features
      [super_0..super_{S-1}]     — per-superregion scalar features
      [PATH_query_0..PATH_query_{P-1}]  — learned query tokens (outputs = path states)

    Returns path_tokens: (B, num_path_tokens, bridge_dim)
    Does NOT have an out_proj — path tokens are passed to ExplicitRefiner.
    No gold in any input.
    """

    def __init__(
        self,
        d_model:        int,
        n_ctx_layers:   int,
        bridge_dim:     int,
        num_heads:      int,
        num_layers:     int,
        ff_mult:        int,
        n_fine:         int,
        n_super:        int,
        top_fine_k:     int,
        top_super_k:    int,
        num_path_tokens: int  = 4,
        dropout:        float = 0.0,
        r2s_np:         Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.d_model        = d_model
        self.n_ctx_layers   = n_ctx_layers
        self.bridge_dim     = bridge_dim
        self.n_fine         = n_fine
        self.n_super        = n_super
        self.top_fine_k     = top_fine_k
        self.top_super_k    = top_super_k
        self.num_path_tokens = num_path_tokens

        self.layer_proj    = nn.Linear(d_model, bridge_dim)
        self.layer_id_emb  = nn.Embedding(n_ctx_layers, bridge_dim)

        self.insert_marker_emb = nn.Embedding(1, bridge_dim)

        self.fine_emb  = nn.Embedding(n_fine  + 1, bridge_dim, padding_idx=n_fine)
        self.super_emb = nn.Embedding(n_super + 1, bridge_dim, padding_idx=n_super)

        self.router_scalar_proj = nn.Linear(3, bridge_dim)
        self.mem_scalar_proj    = nn.Linear(3, bridge_dim)
        self.reg_scalar_proj    = nn.Linear(7, bridge_dim)
        self.super_scalar_proj  = nn.Linear(3, bridge_dim)

        self.path_token = nn.Parameter(torch.zeros(1, num_path_tokens, bridge_dim))
        nn.init.normal_(self.path_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=bridge_dim, nhead=num_heads,
            dim_feedforward=bridge_dim * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if r2s_np is not None:
            self.register_buffer("r2s_buf", torch.from_numpy(r2s_np.astype(np.int64)))
        else:
            self.register_buffer("r2s_buf", torch.zeros(n_fine, dtype=torch.int64))

    # ── Token builders (inference-valid, no gold) ─────────────────────────────

    def _router_token(self, r_reg, r_prb, r_margin):
        r_reg_s = r_reg.clamp(min=0, max=self.n_fine - 1)
        r_emb   = self.fine_emb(r_reg_s)
        p_norm  = r_prb / (r_prb.sum(1, keepdim=True) + 1e-8)
        pooled  = (r_emb * p_norm.unsqueeze(-1)).sum(1)
        entropy = -(r_prb * torch.log(r_prb + 1e-9)).sum(1)
        top1_p  = r_prb[:, 0]
        scalars = torch.stack([entropy, r_margin.float(), top1_p], 1)
        return pooled + self.router_scalar_proj(scalars)

    def _memory_token(self, m_reg, m_prb, m_margin):
        m_reg_s = m_reg.clamp(min=0, max=self.n_fine - 1)
        m_emb   = self.fine_emb(m_reg_s)
        p_norm  = m_prb / (m_prb.sum(1, keepdim=True) + 1e-8)
        pooled  = (m_emb * p_norm.unsqueeze(-1)).sum(1)
        entropy = -(m_prb * torch.log(m_prb + 1e-9)).sum(1)
        top1_p  = m_prb[:, 0]
        scalars = torch.stack([entropy, m_margin.float(), top1_p], 1)
        return pooled + self.mem_scalar_proj(scalars)

    def _fine_region_tokens(self, r_reg, r_prb, m_reg, m_prb,
                             cand_tok, cand_fine, cand_mask, tok_emb_w, h_prime):
        B      = r_reg.shape[0]
        K_r    = r_reg.shape[1]
        K_m    = m_reg.shape[1]
        half   = self.top_fine_k // 2
        take_r = min(K_r, half)
        take_m = min(K_m, self.top_fine_k - take_r)
        fine_ids = torch.cat([r_reg[:, :take_r], m_reg[:, :take_m]], dim=1)
        R      = fine_ids.shape[1]
        fine_s = fine_ids.clamp(min=0, max=self.n_fine - 1)

        reg_emb  = self.fine_emb(fine_s)

        match_r  = (r_reg.unsqueeze(2) == fine_s.unsqueeze(1))
        r_prob   = (r_prb.unsqueeze(2) * match_r.float()).sum(1)
        in_r     = match_r.any(1).float()

        match_m  = (m_reg.unsqueeze(2) == fine_s.unsqueeze(1))
        m_prob   = (m_prb.unsqueeze(2) * match_m.float()).sum(1)
        in_m     = match_m.any(1).float()

        emb_w    = tok_emb_w.float()
        tok_e    = F.embedding(cand_tok.clamp(min=0), emb_w)
        base_lgt = (h_prime.float().unsqueeze(1) * tok_e).sum(-1)
        base_lgt = base_lgt.masked_fill(~cand_mask, float("-inf"))

        match_c  = (cand_fine.unsqueeze(2) == fine_s.unsqueeze(1)) & cand_mask.unsqueeze(2)
        n_cands  = match_c.float().sum(1)
        total_c  = cand_mask.float().sum(1, keepdim=True).clamp(min=1)
        n_frac   = n_cands / total_c

        lgt_exp  = base_lgt.unsqueeze(2).expand(-1, -1, R)
        lgt_exp  = lgt_exp.masked_fill(~match_c, float("-inf"))
        max_lgt  = lgt_exp.amax(1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        lgt_sum  = lgt_exp.masked_fill(~match_c, 0.0).sum(1)
        mean_lgt = lgt_sum / n_cands.clamp(min=1)

        scalars  = torch.stack([r_prob, m_prob, in_r, in_m, n_frac, max_lgt, mean_lgt], dim=-1)
        return reg_emb + self.reg_scalar_proj(scalars)

    def _super_tokens(self, r_reg, r_prb, m_reg, m_prb):
        B      = r_reg.shape[0]
        K      = min(r_reg.shape[1], self.top_super_k * 4)
        r_reg_s = r_reg[:, :K].clamp(min=0, max=self.n_fine - 1)
        r2s     = self.r2s_buf

        sup_ids    = r2s[r_reg_s]
        sup_prb    = r_prb[:, :K]
        sup_scores = torch.zeros(B, self.n_super + 1, device=r_reg.device)
        sup_ids_c  = sup_ids.clamp(min=0, max=self.n_super - 1)
        sup_scores.scatter_add_(1, sup_ids_c, sup_prb)

        K_m     = min(m_reg.shape[1], self.top_super_k * 4)
        m_reg_s = m_reg[:, :K_m].clamp(min=0, max=self.n_fine - 1)
        m_sup   = r2s[m_reg_s]
        m_sup_c = m_sup.clamp(min=0, max=self.n_super - 1)
        sup_scores.scatter_add_(1, m_sup_c, m_prb[:, :K_m])

        top_scores, top_super = sup_scores[:, :self.n_super].topk(self.top_super_k, dim=-1)
        top_super_s = top_super.clamp(min=0, max=self.n_super - 1)
        super_emb   = self.super_emb(top_super_s)

        r2s_top       = r2s[r_reg[:, :K].clamp(0, self.n_fine - 1)]
        match_sup     = (r2s_top.unsqueeze(2) == top_super.unsqueeze(1))
        n_fine_in_sup = match_sup.float().sum(1) / (K + 1e-8)
        scalars       = torch.stack([top_scores, torch.zeros_like(top_scores), n_fine_in_sup], dim=-1)
        return super_emb + self.super_scalar_proj(scalars)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        h_prime:    torch.Tensor,   # (B, d_model) — for candidate logit stats
        h_insert:   torch.Tensor,   # (B, d_model) — INSERT_STATE source
        h_layers:   torch.Tensor,   # (B, n_layers, d_model)
        cand_tok:   torch.Tensor,
        cand_fine:  torch.Tensor,
        cand_mask:  torch.Tensor,
        tok_emb_w:  torch.Tensor,
        r_topk_reg: torch.Tensor,
        r_topk_prb: torch.Tensor,
        m_topk_reg: torch.Tensor,
        m_topk_prb: torch.Tensor,
        r_margin:   torch.Tensor,
        m_margin:   torch.Tensor,
    ) -> torch.Tensor:              # (B, num_path_tokens, bridge_dim)
        B      = h_prime.shape[0]
        device = h_prime.device

        # INSERT_STATE token
        ins_proj   = self.layer_proj(h_insert.float())
        ins_marker = self.insert_marker_emb(torch.zeros(B, dtype=torch.long, device=device))
        insert_t   = (ins_proj + ins_marker).unsqueeze(1)                 # (B, 1, D)

        # Layer tokens
        layer_t = self.layer_proj(h_layers.float())
        ids     = torch.arange(self.n_ctx_layers, device=device)
        layer_t = layer_t + self.layer_id_emb(ids).unsqueeze(0)           # (B, n, D)

        # Aggregate evidence tokens
        rt = self._router_token(r_topk_reg, r_topk_prb, r_margin)
        mt = self._memory_token(m_topk_reg, m_topk_prb, m_margin)
        ft = self._fine_region_tokens(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
                                       cand_tok, cand_fine, cand_mask, tok_emb_w, h_prime)
        st = self._super_tokens(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb)

        # PATH query tokens
        path_tok = self.path_token.expand(B, -1, -1)                      # (B, P, D)

        # Attend over full bridge sequence
        seq = torch.cat(
            [insert_t, layer_t, rt.unsqueeze(1), mt.unsqueeze(1), ft, st, path_tok], dim=1
        )
        seq = self.transformer(seq)

        # Return only PATH token outputs — the "translated" evidence
        return seq[:, -self.num_path_tokens:, :]                          # (B, P, bridge_dim)


# ── Refiner module ────────────────────────────────────────────────────────────

class ExplicitRefiner(nn.Module):
    """
    Performs specialized computation on [RESIDUAL_TOKEN + path tokens].

    Refiner sequence:
      [RESIDUAL_TOKEN]   — residual_proj(h_insert)  — the thing to be corrected
      [PATH_1..PATH_P]   — translated evidence from Bridge

    After attention, RESIDUAL_TOKEN output → out_proj → delta_h.
    out_proj is zero-initialized → delta_h = 0 at step 0 (identity invariant).
    """

    def __init__(
        self,
        d_model:      int,
        bridge_dim:   int,
        refiner_dim:  int,
        num_heads:    int,
        num_layers:   int,
        ff_mult:      int,
        dropout:      float = 0.0,
    ) -> None:
        super().__init__()
        self.refiner_dim = refiner_dim

        self.residual_proj = nn.Linear(d_model, refiner_dim)
        self.path_proj     = (nn.Linear(bridge_dim, refiner_dim)
                              if bridge_dim != refiner_dim else nn.Identity())

        enc_layer = nn.TransformerEncoderLayer(
            d_model=refiner_dim, nhead=num_heads,
            dim_feedforward=refiner_dim * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(refiner_dim, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        h_insert:    torch.Tensor,  # (B, d_model) — mid-layer state
        path_tokens: torch.Tensor,  # (B, P, bridge_dim) — from Bridge
    ) -> torch.Tensor:              # (B, d_model) — delta_h, zero at init
        res_tok  = self.residual_proj(h_insert.float()).unsqueeze(1)  # (B, 1, R)
        path_ref = self.path_proj(path_tokens.float())                # (B, P, R)
        seq      = torch.cat([res_tok, path_ref], dim=1)              # (B, 1+P, R)
        seq      = self.transformer(seq)
        refined  = seq[:, 0, :]                                       # (B, R) — RESIDUAL output
        return self.out_proj(refined)                                  # (B, d_model)


# ── Combined model ────────────────────────────────────────────────────────────

class ExplicitBridgeRefiner(nn.Module):
    """
    Full model: Bridge → Refiner → write delta to h_prime.

    Forward returns (h_prime_refined, delta_h, path_tokens).
    The 3-tuple makes diagnostics explicit without hidden state.

    Write target: h_prime_refined = h_prime + alpha * delta_h
    Decode: h_prime_refined @ tok_emb.T  (full-vocab)

    step-0 identity: Refiner.out_proj zero-init → delta_h=0 → h_prime_refined=h_prime.
    """

    def __init__(
        self,
        d_model:          int,
        n_ctx_layers:     int,
        bridge_dim:       int,
        bridge_heads:     int,
        bridge_layers:    int,
        refiner_dim:      int,
        refiner_heads:    int,
        refiner_layers:   int,
        ff_mult:          int,
        n_fine:           int,
        n_super:          int,
        top_fine_k:       int,
        top_super_k:      int,
        insert_layer_idx: int,
        num_path_tokens:  int   = 4,
        dropout:          float = 0.0,
        r2s_np:           Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.insert_layer_idx = insert_layer_idx
        self.num_path_tokens  = num_path_tokens

        self.bridge = ExplicitBridge(
            d_model=d_model, n_ctx_layers=n_ctx_layers,
            bridge_dim=bridge_dim, num_heads=bridge_heads, num_layers=bridge_layers,
            ff_mult=ff_mult, n_fine=n_fine, n_super=n_super,
            top_fine_k=top_fine_k, top_super_k=top_super_k,
            num_path_tokens=num_path_tokens, dropout=dropout, r2s_np=r2s_np,
        )
        self.refiner = ExplicitRefiner(
            d_model=d_model, bridge_dim=bridge_dim,
            refiner_dim=refiner_dim, num_heads=refiner_heads, num_layers=refiner_layers,
            ff_mult=ff_mult, dropout=dropout,
        )
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(
        self,
        h_prime:    torch.Tensor,   # (B, d_model)
        h_layers:   torch.Tensor,   # (B, n_layers, d_model)
        cand_tok:   torch.Tensor,
        cand_fine:  torch.Tensor,
        cand_mask:  torch.Tensor,
        tok_emb_w:  torch.Tensor,
        r_topk_reg: torch.Tensor,
        r_topk_prb: torch.Tensor,
        m_topk_reg: torch.Tensor,
        m_topk_prb: torch.Tensor,
        r_margin:   torch.Tensor,
        m_margin:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_prime_refined, delta_h, path_tokens)."""
        h_insert = h_layers[:, self.insert_layer_idx, :].float()  # (B, d_model)

        # Bridge: evidence → path-state tokens
        path_tokens = self.bridge(
            h_prime, h_insert, h_layers,
            cand_tok, cand_fine, cand_mask, tok_emb_w,
            r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb, r_margin, m_margin,
        )  # (B, P, bridge_dim)

        # Refiner: [RESIDUAL(h_insert) + path] → delta_h
        delta_h = self.refiner(h_insert, path_tokens)              # (B, d_model)

        # Residual write to h_prime
        h_prime_refined = h_prime.float() + self.alpha * delta_h
        return h_prime_refined, delta_h, path_tokens


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_explicit_loss(
    model:        ExplicitBridgeRefiner,
    batch:        Dict,
    device,
    tok_emb_w:    torch.Tensor,
    train_mask:   Optional[torch.Tensor],
    lambda_kl:    float,
    lambda_delta: float,
    lambda_path:  float,
    kl_topk:      int,
    amp_enabled:  bool = False,
) -> Tuple[Optional[torch.Tensor], Dict]:
    """
    Full-vocab CE + top-K KL + delta L2 + optional path L2 regularizer.
    train_mask=None trains on all rows (for FilteredBridgeShardDataset).
    Returns (total_loss, info_dict). (None, {}) if no training rows.
    """
    B = batch["gold_token"].shape[0]
    if train_mask is None:
        train_mask = torch.ones(B, dtype=torch.bool)
    n_train = int(train_mask.sum())
    if n_train == 0:
        return None, {}

    h_p   = batch["h_prime"].to(device).float()
    h_l   = batch["h_layers"].to(device).float()
    ct    = batch["cand_tok"].to(device)
    cf    = batch["cand_fine"].to(device)
    cm    = batch["cand_mask"].to(device)
    r_r   = batch["r_topk_reg"].to(device)
    r_p   = batch["r_topk_prb"].to(device)
    m_r   = batch["m_topk_reg"].to(device)
    m_p   = batch["m_topk_prb"].to(device)
    r_ma  = batch["r_margin"].to(device)
    m_ma  = batch["m_margin"].to(device)
    gt    = batch["gold_token"].to(device).long()

    emb_w = tok_emb_w.float().to(device)

    h_ref, delta, path_tok = model(h_p, h_l, ct, cf, cm, emb_w,
                                    r_r, r_p, m_r, m_p, r_ma, m_ma)

    h_ref_sel   = h_ref[train_mask]
    h_p_sel     = h_p[train_mask].detach()
    ref_logits  = h_ref_sel.float() @ emb_w.T
    base_logits = h_p_sel.float() @ emb_w.T
    gt_sel      = gt[train_mask]
    loss_ce     = F.cross_entropy(ref_logits, gt_sel)

    # Top-K KL: base_probs || refined_log_probs
    _, topk_idx = base_logits.topk(kl_topk, dim=-1)
    p_base  = F.softmax(base_logits.gather(1, topk_idx), dim=-1).detach()
    lp_ref  = F.log_softmax(ref_logits.gather(1, topk_idx), dim=-1)
    loss_kl = (p_base * (torch.log(p_base + 1e-9) - lp_ref)).sum(-1).mean()

    loss_delta = delta[train_mask].pow(2).mean()
    loss_path  = (path_tok[train_mask].pow(2).mean()
                  if lambda_path > 0 else torch.zeros(1, device=device).squeeze())

    total = loss_ce + lambda_kl * loss_kl + lambda_delta * loss_delta + lambda_path * loss_path

    with torch.no_grad():
        p_n_mean = path_tok.detach().norm(dim=-1).mean().item()
        p_n_max  = path_tok.detach().norm(dim=-1).max().item()

    return total, {
        "ce":             loss_ce.item(),
        "kl":             loss_kl.item(),
        "delta_norm":     delta[train_mask].detach().norm(dim=-1).mean().item(),
        "path_norm_mean": p_n_mean,
        "path_norm_max":  p_n_max,
        "n_train":        n_train,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_vocab_eval_explicit(
    model:            ExplicitBridgeRefiner,
    val_cand_dir:     str,
    val_feat_dir:     str,
    tok_emb_w:        torch.Tensor,
    r2s_np:           np.ndarray,
    device,
    gate_filter_name: str,
    filter_kwargs:    Dict,
    fail_on_mismatch: bool = False,
    eval_batch_size:  int  = 64,
    variant_tag:      str  = "explicit_bridge_refiner",
) -> Dict:
    """Full-vocabulary eval identical to V1 but handles 3-tuple model forward."""
    model.eval()
    emb_w = tok_emb_w.float().to(device)
    V     = emb_w.shape[0]

    # _all accumulators
    base_ce_all = gate_ce_all = 0.0
    ig_base_all = ig_ref_all  = 0.0
    og_base_all = og_gate_all = 0.0
    nc_all = ig_nc_all = og_nc_all = 0
    base_acc1_all = ref_acc1_all = 0

    # _covered accumulators
    base_ce_cov = gate_ce_cov = 0.0
    ig_base_cov = ig_ref_cov  = 0.0
    og_base_cov = og_gate_cov = 0.0
    nc_cov = ig_nc_cov = og_nc_cov = 0
    base_acc1_cov = ref_acc1_cov = 0

    mc_base = mc_gated = 0.0
    mc_nc   = 0

    gate_n = total_n = total_cov = 0
    delta_norms:      List[float] = []
    path_norms_list:  List[float] = []
    h_prime_norms:    List[float] = []

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
        gate = compute_filter_mask(cs, gate_filter_name, **filter_kwargs)

        cf_np    = cs["cand_fine"].numpy().clip(min=0)
        csuper_n = r2s_np[cf_np].astype(np.int64)
        csuper_n[cs["cand_fine"].numpy() < 0] = 0
        cand_sup = torch.from_numpy(csuper_n)

        has_mem  = "mem_topk_reg" in cs
        m_reg_t  = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
        m_prb_t  = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
        m_mar_t  = cs["mem_margin"]   if has_mem else torch.zeros(N)

        for start in range(0, N, eval_batch_size):
            end  = min(start + eval_batch_size, N)
            sl   = slice(start, end)

            h_p   = cs["h_prime"][sl].float().to(device)
            h_l   = fs["h_layers"][sl].float().to(device)
            ct    = cs["cand_tok"][sl].long().to(device)
            cf    = cs["cand_fine"][sl].long().to(device)
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

            base_lgt             = h_p.float() @ emb_w.T
            h_ref, delta, path_t = model(h_p, h_l, ct, cf, cmask, emb_w,
                                          r_reg, r_prb, m_reg, m_prb, r_mar, m_mar)
            ref_lgt  = h_ref.float() @ emb_w.T

            delta_norms.extend(delta.norm(dim=-1).tolist())
            path_norms_list.extend(path_t.norm(dim=-1).view(-1).tolist())
            h_prime_norms.extend(h_p.float().norm(dim=-1).tolist())

            gate_exp  = g_sl.unsqueeze(1).expand(-1, V)
            gated_lgt = torch.where(gate_exp, ref_lgt, base_lgt)
            B_b       = end - start

            # _all
            base_ce_all += float(F.cross_entropy(base_lgt, gt_b, reduction="sum"))
            gate_ce_all += float(F.cross_entropy(gated_lgt, gt_b, reduction="sum"))
            nc_all      += B_b
            base_acc1_all += int((base_lgt.argmax(1) == gt_b).sum())
            ref_acc1_all  += int((gated_lgt.argmax(1) == gt_b).sum())

            ig_all = g_sl
            if ig_all.any():
                ig_gt       = gt_b[ig_all]
                ig_base_all += float(F.cross_entropy(base_lgt[ig_all], ig_gt, reduction="sum"))
                ig_ref_all  += float(F.cross_entropy(ref_lgt[ig_all],  ig_gt, reduction="sum"))
                ig_nc_all   += int(ig_all.sum())

            og_all = ~g_sl
            if og_all.any():
                og_gt       = gt_b[og_all]
                og_base_all += float(F.cross_entropy(base_lgt[og_all], og_gt, reduction="sum"))
                og_gate_all += float(F.cross_entropy(gated_lgt[og_all], og_gt, reduction="sum"))
                og_nc_all   += int(og_all.sum())

            # _covered
            n_cov_b = int(cov.sum())
            if n_cov_b > 0:
                cov_gt       = gt_b[cov]
                base_ce_cov += float(F.cross_entropy(base_lgt[cov], cov_gt, reduction="sum"))
                gate_ce_cov += float(F.cross_entropy(gated_lgt[cov], cov_gt, reduction="sum"))
                nc_cov      += n_cov_b
                base_acc1_cov += int((base_lgt[cov].argmax(1) == cov_gt).sum())
                ref_acc1_cov  += int((gated_lgt[cov].argmax(1) == cov_gt).sum())

                ig_cov = cov & g_sl
                if ig_cov.any():
                    ig_gt_c     = gt_b[ig_cov]
                    ig_base_cov += float(F.cross_entropy(base_lgt[ig_cov], ig_gt_c, reduction="sum"))
                    ig_ref_cov  += float(F.cross_entropy(ref_lgt[ig_cov],  ig_gt_c, reduction="sum"))
                    ig_nc_cov   += int(ig_cov.sum())

                og_cov = cov & ~g_sl
                if og_cov.any():
                    og_gt_c     = gt_b[og_cov]
                    og_base_cov += float(F.cross_entropy(base_lgt[og_cov], og_gt_c, reduction="sum"))
                    og_gate_cov += float(F.cross_entropy(gated_lgt[og_cov], og_gt_c, reduction="sum"))
                    og_nc_cov   += int(og_cov.sum())

                cl_base  = base_lgt[cov].gather(1, ct[cov].clamp(min=0))
                cl_base  = cl_base.masked_fill(~cmask[cov], float("-inf"))
                mc_base += float(F.cross_entropy(cl_base, gidx[cov], reduction="sum"))

                cl_gated = gated_lgt[cov].gather(1, ct[cov].clamp(min=0))
                cl_gated = cl_gated.masked_fill(~cmask[cov], float("-inf"))
                mc_gated += float(F.cross_entropy(cl_gated, gidx[cov], reduction="sum"))
                mc_nc    += n_cov_b

                sum_cand   += int(cmask.sum())
                sum_gi_cov += int(gidx[cov].sum())
                sum_gt     += int(gt_b[cov].sum())

            gate_n    += int(g_sl.sum())
            total_n   += B_b
            total_cov += n_cov_b

    model.train()

    fp_data = {
        "num_shards": len(cand_paths), "total_n": total_n, "total_cov": total_cov,
        "sum_cand_counts": sum_cand, "sum_gold_idx_cov": sum_gi_cov, "sum_gold_tok": sum_gt,
    }
    fp = hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()[:16]

    d_mean  = float(np.mean(delta_norms))     if delta_norms     else 0.0
    d_max   = float(np.max(delta_norms))      if delta_norms     else 0.0
    p_mean  = float(np.mean(path_norms_list)) if path_norms_list else 0.0
    p_max   = float(np.max(path_norms_list))  if path_norms_list else 0.0
    hp_mean = float(np.mean(h_prime_norms))   if h_prime_norms   else 0.0
    d_to_h  = d_mean / (hp_mean + 1e-8)
    alpha_v = float(model.alpha.item())

    def _nll(ce: float, n: int) -> float:
        return ce / max(n, 1)

    results = {
        "eval_mode": "explicit_bridge_refiner_full_vocab",
        # PRIMARY _all
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
        # SECONDARY _covered
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
        # SECONDARY masked-cand
        "masked_cand_base_nll":     _nll(mc_base,  mc_nc),
        "masked_cand_gated_nll":    _nll(mc_gated, mc_nc),
        "masked_cand_gain":         _nll(mc_base, mc_nc) - _nll(mc_gated, mc_nc),
        "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL,
        # Gate / coverage
        "gate_rate":          gate_n    / max(total_n, 1),
        "coverage":           total_cov / max(total_n, 1),
        "num_examples":       total_n,
        "num_covered":        total_cov,
        "inside_gate_n_all":  ig_nc_all,
        "inside_gate_n_cov":  ig_nc_cov,
        "dataset_fingerprint": fp,
        # Diagnostics
        "alpha":               alpha_v,
        "delta_norm_mean":     d_mean,
        "delta_norm_max":      d_max,
        "path_norm_mean":      p_mean,
        "path_norm_max":       p_max,
        "delta_to_h_norm_ratio": d_to_h,
        # Safety
        "gold_force_included_rate": 0.0,
        "selection_mode":           "no_candidate_selection",
        "eval_force_include_gold":  False,
    }

    fp_ok = (fp == CANONICAL_FINGERPRINT)
    n_ok  = (total_n   == CANONICAL_NUM_EXAMPLES)
    nc_ok = (total_cov == CANONICAL_NUM_COVERED)
    cov_d = abs(results["coverage"] - CANONICAL_COVERAGE)
    if not (fp_ok and n_ok and nc_ok and cov_d < 1e-6):
        issues = []
        if not fp_ok:     issues.append(f"fingerprint {fp!r} != {CANONICAL_FINGERPRINT!r}")
        if not n_ok:      issues.append(f"num_examples {total_n} != {CANONICAL_NUM_EXAMPLES}")
        if not nc_ok:     issues.append(f"num_covered {total_cov} != {CANONICAL_NUM_COVERED}")
        if cov_d >= 1e-6: issues.append(f"coverage diff {cov_d:.2e}")
        msg = f"full_vocab_eval_explicit CANONICAL MISMATCH [{variant_tag}]: " + "; ".join(issues)
        if fail_on_mismatch:
            raise RuntimeError(msg)
        print(f"  WARNING: {msg}")

    return results


@torch.no_grad()
def local_subset_eval_explicit(
    model:           ExplicitBridgeRefiner,
    val_cand_dir:    str,
    val_feat_dir:    str,
    tok_emb_w:       torch.Tensor,
    r2s_np:          np.ndarray,
    device,
    filter_kwargs:   Dict,
    eval_batch_size: int = 64,
) -> List[Dict]:
    """Per-subset breakdown — adapted from local_subset_eval_bridge."""
    model.eval()
    emb_w = tok_emb_w.float().to(device)

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

        subset_masks = {name: compute_filter_mask(cs, name, **filter_kwargs)
                        for name in EVAL_SUBSETS}

        has_mem  = "mem_topk_reg" in cs
        m_reg_t  = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
        m_prb_t  = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
        m_mar_t  = cs["mem_margin"]   if has_mem else torch.zeros(N)

        cf_np    = cs["cand_fine"].numpy().clip(min=0)
        csuper_n = r2s_np[cf_np].astype(np.int64)
        csuper_n[cs["cand_fine"].numpy() < 0] = 0

        for start in range(0, N, eval_batch_size):
            end = min(start + eval_batch_size, N)
            sl  = slice(start, end)

            h_p   = cs["h_prime"][sl].float().to(device)
            h_l   = fs["h_layers"][sl].float().to(device)
            ct    = cs["cand_tok"][sl].long().to(device)
            cf    = cs["cand_fine"][sl].long().to(device)
            cov   = cs["covered"][sl].bool()
            gt_b  = cs["gold_token"][sl].long().to(device)
            r_reg = cs["router_topk_reg"][sl].long().to(device)
            r_prb = cs["router_topk_prb"][sl].float().to(device)
            m_reg = m_reg_t[sl].long().to(device)
            m_prb = m_prb_t[sl].float().to(device)
            r_mar = cs["router_margin"][sl].float().to(device)
            m_mar = m_mar_t[sl].float().to(device)
            cmask = (ct >= 0)

            base_lgt                = h_p.float() @ emb_w.T
            h_ref, _, _path         = model(h_p, h_l, ct, cf, cmask, emb_w,
                                             r_reg, r_prb, m_reg, m_prb, r_mar, m_mar)
            ref_lgt = h_ref.float() @ emb_w.T

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


# ── Report ────────────────────────────────────────────────────────────────────

def _generate_report(
    output_dir:          str,
    args,
    full_vocab_base_nll: float,
    final_metrics:       Dict,
    insert_layer_idx:    int,
    layer_ids:           List[int],
) -> None:
    v1_bm_path  = os.path.join("runs", "path_refiner_bridge_adapter",
                                "bridge_boundary_v1_hardsampler", "best_metrics.json")
    ml_bm_path  = os.path.join("runs", "path_refiner_midlayer_bridge",
                                "boundary_insert4_v1", "best_metrics.json")

    def _load(p: str) -> Optional[Dict]:
        return json.load(open(p)) if os.path.isfile(p) else None

    v1_bm = _load(v1_bm_path)
    ml_bm = _load(ml_bm_path)

    no_ckpt  = final_metrics.get("no_improving_checkpoint", True)
    gain     = final_metrics.get("full_vocab_gain_all")
    ig_gain  = final_metrics.get("full_vocab_inside_gate_gain_all")
    best_step = final_metrics.get("best_step", -1)

    ml_bm_own_path = os.path.join(output_dir, "best_metrics.json")
    own_bm = _load(ml_bm_own_path) if not no_ckpt else None

    lines = [
        "# Explicit Bridge + Refiner — Run Report",
        "",
        f"**Variant:** `{os.path.basename(output_dir)}`",
        f"**Date:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| insert_after_block | {args.insert_after_block} → layer_ids[{insert_layer_idx}]={layer_ids[insert_layer_idx]} |",
        f"| num_path_tokens | {args.num_path_tokens} |",
        f"| bridge_dim / bridge_layers / bridge_heads | {args.bridge_dim} / {args.bridge_layers} / {args.bridge_heads} |",
        f"| refiner_dim / refiner_layers / refiner_heads | {args.refiner_dim} / {args.refiner_layers} / {args.refiner_heads} |",
        f"| train_filter / gate_filter | {args.train_filter} / {args.gate_filter} |",
        f"| lr / lambda_kl / lambda_delta / lambda_path | {args.lr} / {args.lambda_kl} / {args.lambda_delta} / {args.lambda_path} |",
        f"| batch_size × grad_accum → eff_bs | {args.batch_size} × {getattr(args,'grad_accum_steps',2)} → {args.batch_size * getattr(args,'grad_accum_steps',2)} |",
        f"| steps / eval_every | {args.steps} / {args.eval_every} |",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| full_vocab_base_nll_all (step 0) | {full_vocab_base_nll:.6f} |",
    ]

    if no_ckpt:
        lines += [
            f"| best_full_vocab_gated_nll_all | N/A — no improving checkpoint |",
            "",
            "**RESULT: No improving checkpoint.**",
        ]
    else:
        gn = own_bm.get("full_vocab_gated_nll_all", "?") if own_bm else "?"
        ga = own_bm.get("full_vocab_gain_all", "?") if own_bm else "?"
        ig = own_bm.get("full_vocab_inside_gate_gain_all", "?") if own_bm else "?"
        mc = own_bm.get("masked_cand_gain", "?") if own_bm else "?"
        lines += [
            f"| best_full_vocab_gated_nll_all | {gn:.6f} |" if isinstance(gn, float) else f"| best_full_vocab_gated_nll_all | {gn} |",
            f"| full_vocab_gain_all (PRIMARY) | {ga:+.6f} |" if isinstance(ga, float) else f"| full_vocab_gain_all | {ga} |",
            f"| inside_gate_gain_all | {ig:+.6f} |" if isinstance(ig, float) else f"| inside_gate_gain_all | {ig} |",
            f"| masked_cand_gain | {mc:+.6f} |" if isinstance(mc, float) else f"| masked_cand_gain | {mc} |",
            f"| best_step | {best_step} |",
        ]

    def _fmtg(d, key):
        if d is None or d.get("no_improving_checkpoint", True):
            return "no_ckpt"
        v = d.get(key)
        return f"{v:+.6f}" if isinstance(v, float) else str(v) if v is not None else "N/A"

    def _delta_str(d1, d2, key):
        s1 = _fmtg(d1, key)
        s2 = _fmtg(d2, key)
        if "no_ckpt" in (s1, s2) or "N/A" in (s1, s2):
            return "N/A"
        try:
            return f"{float(s2) - float(s1):+.6f}"
        except ValueError:
            return "N/A"

    compare_keys = [
        ("full_vocab_gain_all",            "fv_gain_all (PRIMARY)"),
        ("full_vocab_inside_gate_gain_all", "inside_gate_gain_all"),
        ("full_vocab_gain_covered",         "fv_gain_covered"),
        ("masked_cand_gain",                "masked_cand_gain"),
    ]

    lines += [
        "",
        "## Comparison to Prior Architectures",
        "",
        f"| Metric | V1 Late Bridge | MidLayer (blk4) | **ExplicitBridgeRefiner** | ExplB vs V1 |",
        f"|--------|---------------|-----------------|--------------------------|-------------|",
    ]
    for key, label in compare_keys:
        v1_v  = _fmtg(v1_bm, key)
        ml_v  = _fmtg(ml_bm, key)
        own_v = _fmtg(own_bm, key)
        dlt   = _delta_str(v1_bm if (v1_bm and not v1_bm.get("no_improving_checkpoint")) else None,
                           own_bm, key)
        lines.append(f"| {label} | {v1_v} | {ml_v} | **{own_v}** | {dlt} |")

    lines += [
        "",
        "## Success Criteria",
        "",
        "| Criterion | Threshold | Achieved |",
        "|-----------|-----------|---------|",
        f"| Weak success | gain > 0 | {'✓' if not no_ckpt and isinstance(gain, float) and gain > 0 else '✗'} |",
        f"| Meaningful  | gain > +0.003 | {'✓' if not no_ckpt and isinstance(gain, float) and gain > 0.003 else '✗'} |",
        f"| Strong      | gain > +0.005 | {'✓' if not no_ckpt and isinstance(gain, float) and gain > 0.005 else '✗'} |",
        f"| Very strong | gain > +0.010 | {'✓' if not no_ckpt and isinstance(gain, float) and gain > 0.010 else '✗'} |",
        "",
        "## Interpretation",
        "",
        "**If ExplicitBridgeRefiner beats V1:**",
        "  Explicit Bridge→Refiner separation helps; the Refiner's specialized computation",
        "  adds value beyond direct Bridge→delta_h.",
        "",
        "**If ExplicitBridgeRefiner does NOT beat V1:**",
        "  Possible causes: insertion point too late (try block 2/3), Refiner underpowered,",
        "  missing memory-neighbor tokens, gate too broad, or need joint training / LoRA.",
    ]

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [report] written to {report_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_explicit(
    args,
    d_model:          int,
    n_ctx_layers:     int,
    n_fine:           int,
    n_super:          int,
    r2s_np:           np.ndarray,
    tok_emb_w:        torch.Tensor,
    device,
    insert_layer_idx: int,
    layer_ids:        List[int],
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[train] variant               = explicit_bridge_refiner")
    print(f"[train] insert_after_block    = {args.insert_after_block}"
          f"  → insert_layer_idx={insert_layer_idx}  (layer_ids={layer_ids})")
    print(f"[train] bridge_outputs        = path_state_tokens  (num_path_tokens={args.num_path_tokens})")
    print(f"[train] refiner_layers        = {args.refiner_layers}")
    print(f"[train] decode_mode           = full_vocab")
    print(f"[train] selection_mode        = no_candidate_selection")
    print(f"[train] gold_force_included   = never (structural guarantee)")
    print(f"[train] n_ctx_layers={n_ctx_layers}  n_fine={n_fine}  n_super={n_super}")

    model = ExplicitBridgeRefiner(
        d_model          = d_model,
        n_ctx_layers     = n_ctx_layers,
        bridge_dim       = args.bridge_dim,
        bridge_heads     = args.bridge_heads,
        bridge_layers    = args.bridge_layers,
        refiner_dim      = args.refiner_dim,
        refiner_heads    = args.refiner_heads,
        refiner_layers   = args.refiner_layers,
        ff_mult          = args.ff_mult,
        n_fine           = n_fine,
        n_super          = n_super,
        top_fine_k       = args.top_fine_regions,
        top_super_k      = args.top_superregions,
        insert_layer_idx = insert_layer_idx,
        num_path_tokens  = args.num_path_tokens,
        dropout          = args.dropout,
        r2s_np           = r2s_np,
    ).to(device)

    n_bridge  = sum(p.numel() for p in model.bridge.parameters()  if p.requires_grad)
    n_refiner = sum(p.numel() for p in model.refiner.parameters() if p.requires_grad)
    print(f"[train] bridge params={n_bridge:,}  refiner params={n_refiner:,}  "
          f"total={n_bridge+n_refiner+1:,}")

    cfg = {**vars(args), "insert_layer_idx": insert_layer_idx, "layer_ids": layer_ids}
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    fail_hard     = args.fail_on_baseline_mismatch
    filter_kwargs = {"margin_thresh": args.margin_thresh,
                     "entropy_thresh": args.entropy_thresh}
    variant_tag   = (f"ebr-blk{args.insert_after_block}"
                     f"-r{args.refiner_layers}p{args.num_path_tokens}"
                     f"-M{args.top_fine_regions}+S{args.top_superregions}")
    tok_dev = tok_emb_w.float().to(device)

    if args.baseline_json and os.path.isfile(args.baseline_json):
        with open(args.baseline_json) as f:
            masked_bl = json.load(f)
        print(f"\n[train] Masked-candidate baseline (reference only):")
        print(f"  covered_nll = {masked_bl['covered_nll']:.6f}")
        print(f"  NOTE: full-vocab NLL != masked-candidate NLL.")

    # ── Step-0 identity check ─────────────────────────────────────────────────
    full_vocab_base_nll: float = float("nan")

    if args.eval_before_train:
        print(f"\n[train] === step-0 identity check ===")
        g0 = full_vocab_eval_explicit(
            model, args.val_cand_dir, args.val_feat_dir,
            tok_dev, r2s_np, device,
            gate_filter_name = args.gate_filter,
            filter_kwargs    = filter_kwargs,
            fail_on_mismatch = fail_hard,
            eval_batch_size  = args.eval_batch_size,
            variant_tag      = variant_tag,
        )
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]

        nll_diff_all  = abs(g0["full_vocab_gated_nll_all"]     - g0["full_vocab_base_nll_all"])
        nll_diff_cov  = abs(g0["full_vocab_gated_nll_covered"] - g0["full_vocab_base_nll_covered"])
        mc_diff       = abs(g0["masked_cand_gated_nll"]        - g0["masked_cand_base_nll"])
        og_gate_diff  = abs(g0["full_vocab_outside_gate_gated_nll_all"]
                            - g0["full_vocab_outside_gate_base_nll_all"])
        d_max         = g0["delta_norm_max"]

        print(f"  variant               = explicit_bridge_refiner")
        print(f"  insert_after_block    = {args.insert_after_block}  insert_layer_idx={insert_layer_idx}")
        print(f"  num_path_tokens       = {args.num_path_tokens}")
        print(f"  bridge_layers         = {args.bridge_layers}")
        print(f"  refiner_layers        = {args.refiner_layers}")
        print(f"  --- PRIMARY ---")
        print(f"  full_vocab_base_nll_all      = {g0['full_vocab_base_nll_all']:.6f}")
        print(f"  full_vocab_gated_nll_all     = {g0['full_vocab_gated_nll_all']:.6f}")
        print(f"  diff_all (must be < 1e-3)    = {nll_diff_all:.2e}")
        print(f"  --- SECONDARY (covered) ---")
        print(f"  full_vocab_base_nll_covered  = {g0['full_vocab_base_nll_covered']:.6f}")
        print(f"  full_vocab_gated_nll_covered = {g0['full_vocab_gated_nll_covered']:.6f}")
        print(f"  diff_covered (< 1e-3)        = {nll_diff_cov:.2e}")
        print(f"  --- SECONDARY (masked-cand) ---")
        print(f"  masked_cand_base_nll         = {g0['masked_cand_base_nll']:.6f}  "
              f"(canonical ref = {MASKED_CAND_BASELINE_NLL:.6f})")
        print(f"  masked_cand_diff (< 1e-3)    = {mc_diff:.2e}")
        print(f"  --- Outside-gate invariant ---")
        print(f"  og_gate_diff (< 1e-5)        = {og_gate_diff:.2e}")
        print(f"  --- Model state ---")
        print(f"  delta_norm_max (must be 0)   = {d_max:.2e}")
        print(f"  path_norm_mean               = {g0['path_norm_mean']:.4f}")
        print(f"  alpha                        = {g0['alpha']:.4f}")
        print(f"  gold_force_included_rate     = {g0['gold_force_included_rate']:.4f}")

        masked_ref_diff = abs(g0["masked_cand_base_nll"] - MASKED_CAND_BASELINE_NLL)
        if masked_ref_diff > 1e-3:
            raise RuntimeError(
                f"Step-0: masked_cand_base_nll={g0['masked_cand_base_nll']:.6f} "
                f"!= canonical {MASKED_CAND_BASELINE_NLL:.6f} (diff={masked_ref_diff:.2e}).")
        if nll_diff_all >= 1e-3:
            raise RuntimeError(
                f"Step-0 IDENTITY FAIL (_all): diff={nll_diff_all:.2e}. "
                "Refiner out_proj must be zero-init.")
        if nll_diff_cov >= 1e-3:
            raise RuntimeError(f"Step-0 IDENTITY FAIL (_covered): diff={nll_diff_cov:.2e}.")
        if mc_diff >= 1e-3:
            raise RuntimeError(f"Step-0 IDENTITY FAIL (masked_cand): diff={mc_diff:.2e}.")
        if og_gate_diff >= 1e-5:
            raise RuntimeError(
                f"Step-0 outside-gate FAIL: diff={og_gate_diff:.2e} > 1e-5.")
        if d_max > 1e-6:
            raise RuntimeError(f"Step-0 IDENTITY FAIL: delta_norm_max={d_max:.2e} > 0.")

        print(f"  [step-0] IDENTITY PASS — explicit_bridge_refiner  "
              f"insert_after_block={args.insert_after_block}  gold_force_included_rate=0.0000")

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
                "insert_after_block":          str(args.insert_after_block),
                "insert_layer_idx":            insert_layer_idx,
                "note": (
                    "full_vocab_base_nll_all is PRIMARY threshold. "
                    "Best checkpoint saved only if full_vocab_gated_nll_all < this."
                ),
            }, f, indent=2)

        with open(os.path.join(args.output_dir, "debug_identity.json"), "w") as f:
            json.dump({
                "variant":            "explicit_bridge_refiner",
                "insert_after_block": str(args.insert_after_block),
                "insert_layer_idx":   insert_layer_idx,
                "layer_ids":          layer_ids,
                "num_path_tokens":    args.num_path_tokens,
                "bridge_layers":      args.bridge_layers,
                "refiner_layers":     args.refiner_layers,
                "nll_diff_all":       nll_diff_all,
                "nll_diff_covered":   nll_diff_cov,
                "mc_diff":            mc_diff,
                "og_gate_diff":       og_gate_diff,
                "delta_norm_max":     d_max,
                "identity_pass":      True,
                **{k: g0[k] for k in (
                    "full_vocab_base_nll_all", "full_vocab_gated_nll_all",
                    "full_vocab_base_nll_covered", "full_vocab_gated_nll_covered",
                    "masked_cand_base_nll", "masked_cand_gated_nll",
                    "path_norm_mean", "path_norm_max",
                    "dataset_fingerprint", "alpha",
                )},
            }, f, indent=2)
    else:
        print("[train] Computing full-vocab baseline (no --eval_before_train) ...")
        g0 = full_vocab_eval_explicit(
            model, args.val_cand_dir, args.val_feat_dir,
            tok_dev, r2s_np, device,
            gate_filter_name = args.gate_filter,
            filter_kwargs    = filter_kwargs,
            fail_on_mismatch = fail_hard,
            eval_batch_size  = args.eval_batch_size,
            variant_tag      = variant_tag,
        )
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]
        print(f"  full_vocab_base_nll_all = {full_vocab_base_nll:.6f}")

    if full_vocab_base_nll != full_vocab_base_nll:
        raise RuntimeError("full_vocab_base_nll is NaN.")

    print(f"\n[train] Best checkpoint threshold: full_vocab_gated_nll_all < {full_vocab_base_nll:.6f}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    use_filtered = args.use_filtered_train_loader
    grad_accum   = max(1, args.grad_accum_steps)
    eff_bs       = args.batch_size * grad_accum

    print(f"[train] use_filtered_train_loader = {use_filtered}")
    print(f"[train] batch_size                = {args.batch_size}")
    print(f"[train] grad_accum_steps          = {grad_accum}")
    print(f"[train] effective_hard_batch_size = {eff_bs}")
    print()

    if use_filtered:
        train_ds = FilteredBridgeShardDataset(
            args.train_cand_dir, args.train_feat_dir,
            r2s_np, args.train_filter,
            filter_kwargs      = filter_kwargs,
            train_covered_only = args.train_covered_only,
            shuffle            = True,
        )
    else:
        train_ds = BridgeShardDataset(
            args.train_cand_dir, args.train_feat_dir,
            r2s_np, args.train_filter,
            filter_kwargs = filter_kwargs, shuffle=True,
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

    train_fields = ["step", "ce", "kl", "delta_norm", "path_norm_mean", "path_norm_max",
                    "n_train", "alpha", "lr"]
    eval_fields  = [
        "step",
        "full_vocab_base_nll_all", "full_vocab_gated_nll_all", "full_vocab_gain_all",
        "full_vocab_inside_gate_base_nll_all", "full_vocab_inside_gate_ref_nll_all",
        "full_vocab_inside_gate_gain_all",
        "full_vocab_outside_gate_base_nll_all", "full_vocab_outside_gate_gated_nll_all",
        "full_vocab_base_acc1_all", "full_vocab_gated_acc1_all",
        "full_vocab_base_nll_covered", "full_vocab_gated_nll_covered", "full_vocab_gain_covered",
        "full_vocab_inside_gate_base_nll_covered", "full_vocab_inside_gate_ref_nll_covered",
        "full_vocab_inside_gate_gain_covered",
        "full_vocab_outside_gate_base_nll_covered", "full_vocab_outside_gate_gated_nll_covered",
        "masked_cand_base_nll", "masked_cand_gated_nll", "masked_cand_gain",
        "gate_rate", "coverage", "inside_gate_n_all", "inside_gate_n_cov",
        "alpha", "delta_norm_mean", "delta_norm_max",
        "path_norm_mean", "path_norm_max", "delta_to_h_norm_ratio",
        "gold_force_included_rate", "dataset_fingerprint",
    ]
    subset_fields = ["step", "subset", "n", "full_vocab_base_nll",
                     "full_vocab_refined_nll", "full_vocab_gain", "base_acc1", "refined_acc1"]

    train_logf  = open(train_log_path, "w", newline="")
    eval_logf   = open(eval_log_path,  "w", newline="")
    subset_logf = open(subset_path,    "w", newline="")
    train_csv   = csv.DictWriter(train_logf,  fieldnames=train_fields,  extrasaction="ignore")
    eval_csv    = csv.DictWriter(eval_logf,   fieldnames=eval_fields,   extrasaction="ignore")
    subset_csv  = csv.DictWriter(subset_logf, fieldnames=subset_fields, extrasaction="ignore")
    train_csv.writeheader()
    eval_csv.writeheader()
    subset_csv.writeheader()

    best_path = os.path.join(args.output_dir, "best_refiner.pt")
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
                train_mask = None
            else:
                train_mask = (
                    (batch["filter_mask"] & batch["covered"])
                    if args.train_covered_only
                    else batch["filter_mask"]
                )

            if args.amp:
                with autocast("cuda"):
                    loss, info = compute_explicit_loss(
                        model, batch, device, tok_dev, train_mask,
                        args.lambda_kl, args.lambda_delta, args.lambda_path,
                        args.kl_topk, amp_enabled=True)
            else:
                loss, info = compute_explicit_loss(
                    model, batch, device, tok_dev, train_mask,
                    args.lambda_kl, args.lambda_delta, args.lambda_path, args.kl_topk)

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

        avg_ce  = float(np.mean([d["ce"]             for d in accum_infos]))
        avg_kl  = float(np.mean([d["kl"]             for d in accum_infos]))
        avg_dn  = float(np.mean([d["delta_norm"]     for d in accum_infos]))
        avg_pn  = float(np.mean([d["path_norm_mean"] for d in accum_infos]))
        avg_px  = float(np.mean([d["path_norm_max"]  for d in accum_infos]))
        tot_n   = sum(d["n_train"] for d in accum_infos)

        ema_ce  = avg_ce if ema_ce is None else 0.95 * ema_ce + 0.05 * avg_ce
        alpha_v = float(model.alpha.item())
        lr_now  = sched.get_last_lr()[0]

        train_csv.writerow({"step": step, "ce": avg_ce, "kl": avg_kl,
                            "delta_norm": avg_dn, "path_norm_mean": avg_pn,
                            "path_norm_max": avg_px, "n_train": tot_n,
                            "alpha": alpha_v, "lr": lr_now})
        if step % 100 == 0:
            train_logf.flush()
            print(f"  step={step:5d}  ema_ce={ema_ce:.4f}  "
                  f"ce={avg_ce:.4f}  kl={avg_kl:.4f}  "
                  f"d_norm={avg_dn:.4f}  p_norm={avg_pn:.3f}  alpha={alpha_v:.4f}  "
                  f"n_train={tot_n}  eff_bs={eff_bs}  t={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] step={step} ...")
            g_m = full_vocab_eval_explicit(
                model, args.val_cand_dir, args.val_feat_dir,
                tok_dev, r2s_np, device,
                gate_filter_name = args.gate_filter,
                filter_kwargs    = filter_kwargs,
                fail_on_mismatch = fail_hard,
                eval_batch_size  = args.eval_batch_size,
                variant_tag      = variant_tag,
            )
            gated_nll = g_m["full_vocab_gated_nll_all"]
            fv_gain   = g_m["full_vocab_gain_all"]

            eval_csv.writerow({"step": step, **g_m})
            eval_logf.flush()

            og_diff = abs(g_m["full_vocab_outside_gate_gated_nll_all"]
                          - g_m["full_vocab_outside_gate_base_nll_all"])
            print(f"  [eval] step={step}  insert_block={args.insert_after_block}")
            print(f"    PRIMARY (full-vocab, all):")
            print(f"    full_vocab_base_nll_all        = {g_m['full_vocab_base_nll_all']:.6f}")
            print(f"    full_vocab_gated_nll_all       = {gated_nll:.6f}  "
                  f"(threshold={full_vocab_base_nll:.6f})")
            print(f"    full_vocab_gain_all            = {fv_gain:+.6f}")
            print(f"    inside_gate_base_nll_all       = {g_m['full_vocab_inside_gate_base_nll_all']:.6f}")
            print(f"    inside_gate_ref_nll_all        = {g_m['full_vocab_inside_gate_ref_nll_all']:.6f}")
            print(f"    inside_gate_gain_all           = {g_m['full_vocab_inside_gate_gain_all']:+.6f}")
            print(f"    outside_gate_diff_all          = {og_diff:.2e}  (must be ~0)")
            print(f"    SECONDARY (covered):")
            print(f"    full_vocab_gated_nll_covered   = {g_m['full_vocab_gated_nll_covered']:.6f}  "
                  f"gain={g_m['full_vocab_gain_covered']:+.6f}")
            print(f"    SECONDARY (masked-cand):")
            print(f"    masked_cand_gain               = {g_m['masked_cand_gain']:+.6f}  "
                  f"(MLP ref: +0.001284 global, +0.000530 hard-boundary)")
            print(f"    DIAGNOSTICS:")
            print(f"    alpha={g_m['alpha']:.4f}  "
                  f"delta_mean={g_m['delta_norm_mean']:.4f}  "
                  f"delta_max={g_m['delta_norm_max']:.4f}")
            print(f"    path_mean={g_m['path_norm_mean']:.4f}  "
                  f"path_max={g_m['path_norm_max']:.4f}  "
                  f"delta/h_ratio={g_m['delta_to_h_norm_ratio']:.4f}")
            print(f"    gold_force_included_rate = {g_m['gold_force_included_rate']:.4f}")

            sub_rows = local_subset_eval_explicit(
                model, args.val_cand_dir, args.val_feat_dir,
                tok_dev, r2s_np, device,
                filter_kwargs   = filter_kwargs,
                eval_batch_size = args.eval_batch_size,
            )
            for r in sub_rows:
                subset_csv.writerow({"step": step, **r})
            subset_logf.flush()
            for r in sub_rows:
                if r["subset"] == args.gate_filter:
                    print(f"    [{args.gate_filter}]  n={r['n']}  "
                          f"base_nll={r['full_vocab_base_nll']:.6f}  "
                          f"ref_nll={r['full_vocab_refined_nll']:.6f}  "
                          f"gain={r['full_vocab_gain']:+.6f}")

            if gated_nll < best_nll:
                best_nll  = gated_nll
                best_step = step
                best_metrics = {
                    "variant":                               "explicit_bridge_refiner",
                    "insert_after_block":                    str(args.insert_after_block),
                    "insert_layer_idx":                      insert_layer_idx,
                    "num_path_tokens":                       args.num_path_tokens,
                    "bridge_layers":                         args.bridge_layers,
                    "refiner_layers":                        args.refiner_layers,
                    "selection_mode":                        "no_candidate_selection",
                    "eval_force_include_gold":               False,
                    "gold_force_included_rate":              0.0,
                    "step":                                  step,
                    "train_filter":                          args.train_filter,
                    "gate_filter":                           args.gate_filter,
                    # PRIMARY
                    "full_vocab_base_nll_all":               full_vocab_base_nll,
                    "full_vocab_gated_nll_all":              gated_nll,
                    "full_vocab_gain_all":                   fv_gain,
                    "full_vocab_inside_gate_base_nll_all":   g_m["full_vocab_inside_gate_base_nll_all"],
                    "full_vocab_inside_gate_ref_nll_all":    g_m["full_vocab_inside_gate_ref_nll_all"],
                    "full_vocab_inside_gate_gain_all":       g_m["full_vocab_inside_gate_gain_all"],
                    "full_vocab_outside_gate_base_nll_all":  g_m["full_vocab_outside_gate_base_nll_all"],
                    "full_vocab_outside_gate_gated_nll_all": g_m["full_vocab_outside_gate_gated_nll_all"],
                    # SECONDARY
                    "full_vocab_base_nll_covered":           g_m["full_vocab_base_nll_covered"],
                    "full_vocab_gated_nll_covered":          g_m["full_vocab_gated_nll_covered"],
                    "full_vocab_gain_covered":               g_m["full_vocab_gain_covered"],
                    "full_vocab_inside_gate_base_nll_covered": g_m["full_vocab_inside_gate_base_nll_covered"],
                    "full_vocab_inside_gate_ref_nll_covered":  g_m["full_vocab_inside_gate_ref_nll_covered"],
                    "full_vocab_inside_gate_gain_covered":     g_m["full_vocab_inside_gate_gain_covered"],
                    "masked_cand_base_nll":                  g_m["masked_cand_base_nll"],
                    "masked_cand_gated_nll":                 g_m["masked_cand_gated_nll"],
                    "masked_cand_gain":                      g_m["masked_cand_gain"],
                    "masked_cand_baseline_ref":              MASKED_CAND_BASELINE_NLL,
                    # META
                    "gate_rate":                             g_m["gate_rate"],
                    "coverage":                              g_m["coverage"],
                    "num_examples":                          g_m["num_examples"],
                    "num_covered":                           g_m["num_covered"],
                    "alpha":                                 g_m["alpha"],
                    "delta_norm_mean":                       g_m["delta_norm_mean"],
                    "delta_norm_max":                        g_m["delta_norm_max"],
                    "path_norm_mean":                        g_m["path_norm_mean"],
                    "path_norm_max":                         g_m["path_norm_max"],
                    "delta_to_h_norm_ratio":                 g_m["delta_to_h_norm_ratio"],
                    "dataset_fingerprint":                   g_m["dataset_fingerprint"],
                    "no_improving_checkpoint":               False,
                }
                torch.save({"step": step, "model": model.state_dict(),
                            "metrics": best_metrics, "args": vars(args)}, best_path)
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"  *** NEW BEST  full_vocab_gated_nll_all={best_nll:.6f}  "
                      f"gain={fv_gain:+.6f}  step={step}  → {best_path}")
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
        "no_improving_checkpoint":             (best_step < 0),
        "best_step":                           best_step,
        "best_full_vocab_gated_nll_all":       best_nll if best_step >= 0 else None,
        "full_vocab_base_nll_all":             full_vocab_base_nll,
        "full_vocab_gain_all":                 full_vocab_base_nll - best_nll if best_step >= 0 else None,
        "full_vocab_inside_gate_gain_all":     None,  # filled from best_metrics if available
        "masked_cand_baseline_ref":            MASKED_CAND_BASELINE_NLL,
        "insert_after_block":                  str(args.insert_after_block),
        "insert_layer_idx":                    insert_layer_idx,
    }
    # Populate gate gain from best_metrics if exists
    bm_path = os.path.join(args.output_dir, "best_metrics.json")
    if os.path.isfile(bm_path) and best_step >= 0:
        with open(bm_path) as f:
            bm = json.load(f)
        final_metrics["full_vocab_inside_gate_gain_all"] = bm.get("full_vocab_inside_gate_gain_all")

    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    if best_step < 0:
        print(f"\n[train] RESULT: no improving checkpoint — model never beat "
              f"full_vocab_base_nll_all={full_vocab_base_nll:.6f}.")
        with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
            json.dump({"no_improving_checkpoint": True,
                       "full_vocab_base_nll_all": full_vocab_base_nll,
                       "insert_after_block": str(args.insert_after_block),
                       "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL}, f, indent=2)
    else:
        print(f"\n[train] done  best_full_vocab_gated_nll_all={best_nll:.6f}  "
              f"gain={full_vocab_base_nll-best_nll:+.6f}  step={best_step}")

    _generate_report(args.output_dir, args, full_vocab_base_nll,
                     final_metrics, insert_layer_idx, layer_ids)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device(args.device)

    print(f"[main] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, _ = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    print(f"  d_model={d_model}  n_blocks={len(backbone.blocks)}")

    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().cpu()
    elif hasattr(backbone, "transformer"):
        tok_emb_w = backbone.transformer.wte.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot locate token embedding in backbone")
    print(f"  tok_emb_w shape = {tuple(tok_emb_w.shape)}")

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

    feat_cfg_path = os.path.join(args.val_feat_dir, "config.json")
    if not os.path.isfile(feat_cfg_path):
        raise RuntimeError(
            f"Feature config not found: {feat_cfg_path}. "
            "Run slurm_build_multilayer_residual_features.sh first.")
    with open(feat_cfg_path) as f:
        feat_cfg = json.load(f)
    n_ctx_layers = feat_cfg["n_layers_saved"]
    layer_ids    = feat_cfg["layer_ids"]
    print(f"  n_ctx_layers={n_ctx_layers}  layer_ids={layer_ids}")

    insert_layer_idx = _resolve_insert_layer_idx(args.insert_after_block, layer_ids)
    resolved_block   = layer_ids[insert_layer_idx]
    print(f"  insert_after_block={args.insert_after_block} → "
          f"insert_layer_idx={insert_layer_idx}  (layer_ids[{insert_layer_idx}]={resolved_block})")

    if args.gate_filter is None:
        args.gate_filter = args.train_filter

    train_explicit(args, d_model, n_ctx_layers, n_fine, n_super, r2s_np, tok_emb_w, device,
                   insert_layer_idx, layer_ids)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--small_ckpt",     required=True)
    p.add_argument("--train_cand_dir", required=True)
    p.add_argument("--val_cand_dir",   required=True)
    p.add_argument("--train_feat_dir", required=True)
    p.add_argument("--val_feat_dir",   required=True)
    p.add_argument("--baseline_json",  default=None)
    p.add_argument("--super_map",      default=None)
    p.add_argument("--output_dir",     required=True)
    # Bridge architecture
    p.add_argument("--bridge_dim",      type=int,   default=256)
    p.add_argument("--bridge_layers",   type=int,   default=1,
                   help="Transformer layers in Bridge (evidence translator). Default: 1.")
    p.add_argument("--bridge_heads",    type=int,   default=4)
    # Refiner architecture
    p.add_argument("--refiner_dim",     type=int,   default=256)
    p.add_argument("--refiner_layers",  type=int,   default=2,
                   help="Transformer layers in Refiner (specialist). Default: 2.")
    p.add_argument("--refiner_heads",   type=int,   default=4)
    p.add_argument("--ff_mult",         type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.0)
    # Region tokens
    p.add_argument("--top_fine_regions", type=int,  default=24)
    p.add_argument("--top_superregions", type=int,  default=8)
    p.add_argument("--num_path_tokens",  type=int,  default=4,
                   help="Number of PATH query tokens in Bridge; all passed to Refiner. Default: 4.")
    # Insertion point
    p.add_argument("--insert_after_block", default="4",
                   help="Block whose output is INSERT_STATE. Must be in layer_ids. Default: 4.")
    # Filter
    p.add_argument("--train_filter",   default="boundary")
    p.add_argument("--gate_filter",    default=None)
    p.add_argument("--margin_thresh",  type=float, default=0.1)
    p.add_argument("--entropy_thresh", type=float, default=2.0)
    # Training
    p.add_argument("--steps",           type=int,   default=5000)
    p.add_argument("--eval_every",      type=int,   default=1000)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--eval_batch_size", type=int,   default=64)
    p.add_argument("--lr",              type=float, default=5e-5)
    p.add_argument("--lambda_kl",       type=float, default=0.2)
    p.add_argument("--lambda_delta",    type=float, default=3e-4)
    p.add_argument("--lambda_path",     type=float, default=0.0,
                   help="L2 regularizer on path token norms. Default: 0.0 (disabled).")
    p.add_argument("--kl_topk",         type=int,   default=512)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--grad_accum_steps",type=int,   default=2)
    # Flags
    p.add_argument("--amp",                      action="store_true")
    p.add_argument("--eval_before_train",        action="store_true")
    p.add_argument("--fail_on_baseline_mismatch",action="store_true")
    p.add_argument("--use_filtered_train_loader",action="store_true")
    p.add_argument("--train_covered_only",       action="store_true")
    p.add_argument("--decode_mode",              default="full_vocab",
                   help="Must be 'full_vocab'. Other values reserved.")
    p.add_argument("--device",                   default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
