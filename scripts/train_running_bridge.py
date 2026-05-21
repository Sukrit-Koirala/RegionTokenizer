#!/usr/bin/env python3
"""
Running Bridge Adapter V0 — persistent path-state across backbone layers.

Pipeline:
  [RegionTransformer outputs]
      → RunningBridgeUpdate @ layer 2  (update path_state with early evidence)
      → RunningBridgeUpdate @ layer 4  (update path_state with later evidence)
      → RunningBridgeRefiner ([RESIDUAL + path_state] → delta_h_final)
      → h_refined = h_prime + alpha * delta_h_final
      → full-vocab decoder

write_mode = final_only:
  Path state tracks trajectory across layers. Single final write to h_prime.

write_mode = multi_write:
  Also writes small delta_L to h_prime at each update layer (zero-init write_projs).
  Previous delta_L fed as additional input to next update (simulated causal chain):
    h_L_effective = h_L + delta_{L-1}  (gradient flows back through the chain)
  h_refined = h_prime + alpha*delta_final + sum(alpha_L * delta_L)

Identity invariant at step 0:
  Refiner.out_proj zero-init → delta_final = 0.
  write_projs zero-init → delta_L = 0.
  h_refined = h_prime → logits_refined = logits_base.

No gold in any bridge/refiner input.
gold_force_included_rate = 0.0 structurally.

Usage:
    python scripts/train_running_bridge.py \\
        --small_ckpt runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --train_cand_dir runs/path_refiner_clean/data/train_hgrid_K24 \\
        --val_cand_dir   runs/path_refiner_clean/data/val_hgrid_K24 \\
        --train_feat_dir runs/path_refiner_residual_interface/features/train_multilayer \\
        --val_feat_dir   runs/path_refiner_residual_interface/features/val_multilayer \\
        --baseline_json  runs/path_refiner_clean/baselines/saved_candidate_baseline.json \\
        --super_map runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --output_dir runs/path_refiner_running_bridge/boundary_running_finalonly_v0 \\
        --write_mode final_only --update_layers 2,4 \\
        --bridge_dim 256 --bridge_update_layers 1 --bridge_heads 4 \\
        --refiner_dim 256 --refiner_layers 2 --refiner_heads 4 \\
        --num_path_tokens 4 \\
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

def _resolve_update_layer_idxs(update_layers_arg: str, layer_ids: List[int]) -> List[int]:
    """Parse "2,4" or "2,4,final" → list of indices into h_layers tensor."""
    result = []
    for part in [s.strip() for s in update_layers_arg.split(",")]:
        if part in ("final", "-1"):
            try:
                result.append(layer_ids.index(-1))
            except ValueError:
                result.append(len(layer_ids) - 1)
        else:
            iab = int(part)
            if iab in layer_ids:
                result.append(layer_ids.index(iab))
            else:
                candidates = [(i, abs(lid - iab)) for i, lid in enumerate(layer_ids) if lid != -1]
                if not candidates:
                    result.append(len(layer_ids) - 1)
                else:
                    best_i = min(candidates, key=lambda x: x[1])[0]
                    print(f"  WARNING: update layer {iab} not in layer_ids={layer_ids}. "
                          f"Using closest: layer_ids[{best_i}]={layer_ids[best_i]}")
                    result.append(best_i)
    return result


# ── RunningBridgeUpdate ───────────────────────────────────────────────────────

class RunningBridgeUpdate(nn.Module):
    """
    Single step of running bridge update at one backbone layer.

    Sequence:
      [path_state (P) | h_L_token (1) | ROUTER (1) | MEMORY (1) | fine (F) | super (S)]
    Total: P + 3 + top_fine_k + top_super_k  (e.g. 4+3+24+8=39 tokens)

    After attention: new_path_state = output[:, :P, :]
    No out_proj. No gold input.
    """

    def __init__(
        self,
        d_model:         int,
        bridge_dim:      int,
        num_heads:       int,
        num_layers:      int,
        ff_mult:         int,
        n_fine:          int,
        n_super:         int,
        top_fine_k:      int,
        top_super_k:     int,
        num_path_tokens: int,
        update_idx:      int,
        n_updates:       int,
        dropout:         float = 0.0,
        r2s_np:          Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.num_path_tokens = num_path_tokens
        self.top_fine_k      = top_fine_k
        self.top_super_k     = top_super_k
        self.n_fine          = n_fine
        self.n_super         = n_super
        self.update_idx      = update_idx

        self.h_proj   = nn.Linear(d_model, bridge_dim, bias=False)
        self.h_marker = nn.Embedding(n_updates, bridge_dim)

        self.fine_emb  = nn.Embedding(n_fine  + 1, bridge_dim, padding_idx=n_fine)
        self.super_emb = nn.Embedding(n_super + 1, bridge_dim, padding_idx=n_super)

        self.router_scalar_proj = nn.Linear(3, bridge_dim)
        self.mem_scalar_proj    = nn.Linear(3, bridge_dim)
        self.reg_scalar_proj    = nn.Linear(7, bridge_dim)
        self.super_scalar_proj  = nn.Linear(3, bridge_dim)

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

    def _router_token(self, r_reg, r_prb, r_margin):
        r_reg_s = r_reg.clamp(min=0, max=self.n_fine - 1)
        pooled  = (self.fine_emb(r_reg_s) *
                   (r_prb / (r_prb.sum(1, keepdim=True) + 1e-8)).unsqueeze(-1)).sum(1)
        entropy = -(r_prb * torch.log(r_prb + 1e-9)).sum(1)
        scalars = torch.stack([entropy, r_margin.float(), r_prb[:, 0]], 1)
        return pooled + self.router_scalar_proj(scalars)

    def _memory_token(self, m_reg, m_prb, m_margin):
        m_reg_s = m_reg.clamp(min=0, max=self.n_fine - 1)
        pooled  = (self.fine_emb(m_reg_s) *
                   (m_prb / (m_prb.sum(1, keepdim=True) + 1e-8)).unsqueeze(-1)).sum(1)
        entropy = -(m_prb * torch.log(m_prb + 1e-9)).sum(1)
        scalars = torch.stack([entropy, m_margin.float(), m_prb[:, 0]], 1)
        return pooled + self.mem_scalar_proj(scalars)

    def _fine_region_tokens(self, r_reg, r_prb, m_reg, m_prb,
                             cand_tok, cand_fine, cand_mask, tok_emb_w, h_prime):
        K_r    = r_reg.shape[1]
        K_m    = m_reg.shape[1]
        half   = self.top_fine_k // 2
        take_r = min(K_r, half)
        take_m = min(K_m, self.top_fine_k - take_r)
        fine_ids = torch.cat([r_reg[:, :take_r], m_reg[:, :take_m]], dim=1)
        R        = fine_ids.shape[1]
        fine_s   = fine_ids.clamp(min=0, max=self.n_fine - 1)

        reg_emb = self.fine_emb(fine_s)

        match_r = (r_reg.unsqueeze(2) == fine_s.unsqueeze(1))
        r_prob  = (r_prb.unsqueeze(2) * match_r.float()).sum(1)
        in_r    = match_r.any(1).float()

        match_m = (m_reg.unsqueeze(2) == fine_s.unsqueeze(1))
        m_prob  = (m_prb.unsqueeze(2) * match_m.float()).sum(1)
        in_m    = match_m.any(1).float()

        emb_w    = tok_emb_w.float()
        tok_e    = F.embedding(cand_tok.clamp(min=0), emb_w)
        base_lgt = (h_prime.float().unsqueeze(1) * tok_e).sum(-1)
        base_lgt = base_lgt.masked_fill(~cand_mask, float("-inf"))

        match_c = (cand_fine.unsqueeze(2) == fine_s.unsqueeze(1)) & cand_mask.unsqueeze(2)
        n_cands = match_c.float().sum(1)
        n_frac  = n_cands / cand_mask.float().sum(1, keepdim=True).clamp(min=1)

        lgt_exp = base_lgt.unsqueeze(2).expand(-1, -1, R).masked_fill(~match_c, float("-inf"))
        max_lgt = lgt_exp.amax(1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        mean_lgt = lgt_exp.masked_fill(~match_c, 0.0).sum(1) / n_cands.clamp(min=1)

        scalars = torch.stack([r_prob, m_prob, in_r, in_m, n_frac, max_lgt, mean_lgt], dim=-1)
        return reg_emb + self.reg_scalar_proj(scalars)

    def _super_tokens(self, r_reg, r_prb, m_reg, m_prb):
        B       = r_reg.shape[0]
        K       = min(r_reg.shape[1], self.top_super_k * 4)
        r2s     = self.r2s_buf
        r_reg_s = r_reg[:, :K].clamp(0, self.n_fine - 1)

        sup_ids    = r2s[r_reg_s]
        sup_scores = torch.zeros(B, self.n_super + 1, device=r_reg.device)
        sup_scores.scatter_add_(1, sup_ids.clamp(0, self.n_super - 1), r_prb[:, :K])

        K_m     = min(m_reg.shape[1], self.top_super_k * 4)
        m_reg_s = m_reg[:, :K_m].clamp(0, self.n_fine - 1)
        m_sup   = r2s[m_reg_s]
        sup_scores.scatter_add_(1, m_sup.clamp(0, self.n_super - 1), m_prb[:, :K_m])

        top_scores, top_super = sup_scores[:, :self.n_super].topk(self.top_super_k, dim=-1)
        super_emb = self.super_emb(top_super.clamp(0, self.n_super - 1))

        r2s_top       = r2s[r_reg_s]
        n_fine_in_sup = (r2s_top.unsqueeze(2) == top_super.unsqueeze(1)).float().sum(1) / (K + 1e-8)
        scalars       = torch.stack([top_scores, torch.zeros_like(top_scores), n_fine_in_sup], dim=-1)
        return super_emb + self.super_scalar_proj(scalars)

    def forward(
        self,
        path_state:  torch.Tensor,
        h_L:         torch.Tensor,
        h_prime:     torch.Tensor,
        cand_tok:    torch.Tensor,
        cand_fine:   torch.Tensor,
        cand_mask:   torch.Tensor,
        tok_emb_w:   torch.Tensor,
        r_topk_reg:  torch.Tensor,
        r_topk_prb:  torch.Tensor,
        m_topk_reg:  torch.Tensor,
        m_topk_prb:  torch.Tensor,
        r_margin:    torch.Tensor,
        m_margin:    torch.Tensor,
    ) -> torch.Tensor:
        B      = path_state.shape[0]
        device = path_state.device

        h_tok = (self.h_proj(h_L.float()) +
                 self.h_marker(torch.full((B,), self.update_idx, dtype=torch.long,
                                          device=device))).unsqueeze(1)

        rt = self._router_token(r_topk_reg, r_topk_prb, r_margin)
        mt = self._memory_token(m_topk_reg, m_topk_prb, m_margin)
        ft = self._fine_region_tokens(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
                                       cand_tok, cand_fine, cand_mask, tok_emb_w, h_prime)
        st = self._super_tokens(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb)

        seq = torch.cat([path_state, h_tok, rt.unsqueeze(1), mt.unsqueeze(1), ft, st], dim=1)
        seq = self.transformer(seq)
        return seq[:, :self.num_path_tokens, :]


# ── RunningBridgeRefiner ──────────────────────────────────────────────────────

class RunningBridgeRefiner(nn.Module):
    """[RESIDUAL_TOKEN + path_state] → delta_h. out_proj zero-init (identity at step 0)."""

    def __init__(self, d_model, bridge_dim, refiner_dim, num_heads, num_layers,
                 ff_mult, dropout=0.0):
        super().__init__()
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

    def forward(self, h_insert: torch.Tensor,
                path_state: torch.Tensor) -> torch.Tensor:
        res_tok = self.residual_proj(h_insert.float()).unsqueeze(1)
        path_r  = self.path_proj(path_state.float())
        seq     = self.transformer(torch.cat([res_tok, path_r], dim=1))
        return self.out_proj(seq[:, 0, :])


# ── RunningBridgeAdapter ──────────────────────────────────────────────────────

class RunningBridgeAdapter(nn.Module):
    """
    Running Bridge Adapter V0.

    write_mode='final_only': path tracks trajectory, single delta write to h_prime.
    write_mode='multi_write': also writes delta_L to h_prime at each update layer;
        h_L_effective = h_L + prev_delta feeds causal signal to the next update.

    Identity: all write_projs and refiner.out_proj are zero-init → h_refined = h_prime at step 0.
    """

    def __init__(
        self,
        d_model:              int,
        bridge_dim:           int,
        bridge_heads:         int,
        bridge_update_layers: int,
        refiner_dim:          int,
        refiner_heads:        int,
        refiner_layers:       int,
        ff_mult:              int,
        n_fine:               int,
        n_super:              int,
        top_fine_k:           int,
        top_super_k:          int,
        update_layer_idxs:    List[int],
        num_path_tokens:      int,
        write_mode:           str,
        dropout:              float = 0.0,
        r2s_np:               Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.update_layer_idxs  = update_layer_idxs
        self.write_mode         = write_mode
        self.num_path_tokens    = num_path_tokens
        n_updates               = len(update_layer_idxs)

        self.learned_path_tokens = nn.Parameter(
            torch.zeros(1, num_path_tokens, bridge_dim))
        nn.init.normal_(self.learned_path_tokens, std=0.02)

        self.update_modules = nn.ModuleList([
            RunningBridgeUpdate(
                d_model=d_model, bridge_dim=bridge_dim, num_heads=bridge_heads,
                num_layers=bridge_update_layers, ff_mult=ff_mult,
                n_fine=n_fine, n_super=n_super, top_fine_k=top_fine_k,
                top_super_k=top_super_k, num_path_tokens=num_path_tokens,
                update_idx=i, n_updates=n_updates, dropout=dropout, r2s_np=r2s_np,
            )
            for i in range(n_updates)
        ])

        if write_mode == "multi_write":
            self.write_projs = nn.ModuleList([
                nn.Linear(bridge_dim, d_model) for _ in range(n_updates)
            ])
            for proj in self.write_projs:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            self.alpha_layers = nn.ParameterList([
                nn.Parameter(torch.ones(1)) for _ in range(n_updates)
            ])
        else:
            self.write_projs  = None
            self.alpha_layers = None

        self.refiner = RunningBridgeRefiner(
            d_model=d_model, bridge_dim=bridge_dim, refiner_dim=refiner_dim,
            num_heads=refiner_heads, num_layers=refiner_layers, ff_mult=ff_mult,
            dropout=dropout,
        )
        self.alpha = nn.Parameter(torch.ones(1))
        self.final_update_h_idx = update_layer_idxs[-1]

    def forward(
        self,
        h_prime:    torch.Tensor,
        h_layers:   torch.Tensor,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Returns (h_refined, delta_final, final_path_state, layer_deltas)."""
        B          = h_prime.shape[0]
        path_state = self.learned_path_tokens.expand(B, -1, -1)
        layer_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
        prev_delta: Optional[torch.Tensor] = None

        for i, (upd, layer_idx) in enumerate(
                zip(self.update_modules, self.update_layer_idxs)):
            h_L = h_layers[:, layer_idx, :].float()
            if self.write_mode == "multi_write" and prev_delta is not None:
                h_L = h_L + prev_delta

            path_state = upd(
                path_state, h_L, h_prime,
                cand_tok, cand_fine, cand_mask, tok_emb_w,
                r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb, r_margin, m_margin,
            )

            if self.write_mode == "multi_write":
                delta_L = self.write_projs[i](path_state[:, 0, :])
                layer_deltas.append((delta_L, self.alpha_layers[i]))
                prev_delta = delta_L

        h_insert    = h_layers[:, self.final_update_h_idx, :].float()
        delta_final = self.refiner(h_insert, path_state)

        h_refined = h_prime.float() + self.alpha * delta_final
        for delta_L, alpha_L in layer_deltas:
            h_refined = h_refined + alpha_L * delta_L

        return h_refined, delta_final, path_state, layer_deltas


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_running_bridge_loss(
    model:        RunningBridgeAdapter,
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
    cm   = batch["cand_mask"].to(device)
    r_r  = batch["r_topk_reg"].to(device)
    r_p  = batch["r_topk_prb"].to(device)
    m_r  = batch["m_topk_reg"].to(device)
    m_p  = batch["m_topk_prb"].to(device)
    r_ma = batch["r_margin"].to(device)
    m_ma = batch["m_margin"].to(device)
    gt   = batch["gold_token"].to(device).long()

    emb_w = tok_emb_w.float().to(device)

    h_ref, delta_final, path_state, layer_deltas = model(
        h_p, h_l, ct, cf, cm, emb_w, r_r, r_p, m_r, m_p, r_ma, m_ma)

    ref_logits  = h_ref[train_mask].float() @ emb_w.T
    base_logits = h_p[train_mask].detach().float() @ emb_w.T
    gt_sel      = gt[train_mask]
    loss_ce     = F.cross_entropy(ref_logits, gt_sel)

    _, topk_idx = base_logits.topk(kl_topk, dim=-1)
    p_base  = F.softmax(base_logits.gather(1, topk_idx), dim=-1).detach()
    lp_ref  = F.log_softmax(ref_logits.gather(1, topk_idx), dim=-1)
    loss_kl = (p_base * (torch.log(p_base + 1e-9) - lp_ref)).sum(-1).mean()

    loss_delta = delta_final[train_mask].pow(2).mean()
    for delta_L, _ in layer_deltas:
        loss_delta = loss_delta + delta_L[train_mask].pow(2).mean()

    loss_path = (path_state[train_mask].pow(2).mean()
                 if lambda_path > 0 else torch.zeros(1, device=device).squeeze())

    total = loss_ce + lambda_kl * loss_kl + lambda_delta * loss_delta + lambda_path * loss_path

    with torch.no_grad():
        d_final_n = delta_final[train_mask].norm(dim=-1).mean().item()
        layer_d_n = [dl[train_mask].norm(dim=-1).mean().item() for dl, _ in layer_deltas]
        all_d_n   = [d_final_n] + layer_d_n
        p_n_mean  = path_state.norm(dim=-1).mean().item()
        p_n_max   = path_state.norm(dim=-1).max().item()

    info: Dict = {
        "ce":               loss_ce.item(),
        "kl":               loss_kl.item(),
        "delta_norm":       float(np.mean(all_d_n)) if all_d_n else 0.0,
        "final_delta_norm": d_final_n,
        "path_norm_mean":   p_n_mean,
        "path_norm_max":    p_n_max,
        "n_train":          n_train,
    }
    for i, dn in enumerate(layer_d_n):
        info[f"layer_delta_norm_{i}"] = dn
    return total, info


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_vocab_eval_running_bridge(
    model:            RunningBridgeAdapter,
    val_cand_dir:     str,
    val_feat_dir:     str,
    tok_emb_w:        torch.Tensor,
    r2s_np:           np.ndarray,
    device,
    gate_filter_name: str,
    filter_kwargs:    Dict,
    fail_on_mismatch: bool = False,
    eval_batch_size:  int  = 64,
    variant_tag:      str  = "running_bridge_adapter",
) -> Dict:
    model.eval()
    emb_w = tok_emb_w.float().to(device)
    V     = emb_w.shape[0]

    base_ce_all = gate_ce_all = 0.0
    ig_base_all = ig_ref_all  = 0.0
    og_base_all = og_gate_all = 0.0
    nc_all = ig_nc_all = og_nc_all = 0
    base_acc1_all = ref_acc1_all = 0

    base_ce_cov = gate_ce_cov = 0.0
    ig_base_cov = ig_ref_cov  = 0.0
    og_base_cov = og_gate_cov = 0.0
    nc_cov = ig_nc_cov = og_nc_cov = 0
    base_acc1_cov = ref_acc1_cov = 0

    mc_base = mc_gated = 0.0
    mc_nc   = 0

    gate_n = total_n = total_cov = 0
    final_delta_norms: List[float] = []
    layer_delta_norms: List[List[float]] = []
    path_norms:        List[float] = []
    h_prime_norms:     List[float] = []

    sum_cand = sum_gi_cov = sum_gt = 0

    cand_paths = sorted(glob.glob(os.path.join(val_cand_dir, "shard_*.pt")))
    if not cand_paths:
        raise RuntimeError(f"No shard_*.pt in {val_cand_dir}")

    for cand_path in cand_paths:
        si        = int(os.path.basename(cand_path).replace("shard_", "").replace(".pt", ""))
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

        has_mem = "mem_topk_reg" in cs
        m_reg_t = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
        m_prb_t = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
        m_mar_t = cs["mem_margin"]   if has_mem else torch.zeros(N)

        for start in range(0, N, eval_batch_size):
            end = min(start + eval_batch_size, N)
            sl  = slice(start, end)

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

            base_lgt = h_p.float() @ emb_w.T
            h_ref, d_final, path_st, l_deltas = model(
                h_p, h_l, ct, cf, cmask, emb_w,
                r_reg, r_prb, m_reg, m_prb, r_mar, m_mar)
            ref_lgt  = h_ref.float() @ emb_w.T

            final_delta_norms.extend(d_final.norm(dim=-1).tolist())
            path_norms.extend(path_st.norm(dim=-1).view(-1).tolist())
            h_prime_norms.extend(h_p.float().norm(dim=-1).tolist())
            for i, (dl, _) in enumerate(l_deltas):
                while len(layer_delta_norms) <= i:
                    layer_delta_norms.append([])
                layer_delta_norms[i].extend(dl.norm(dim=-1).tolist())

            gate_exp  = g_sl.unsqueeze(1).expand(-1, V)
            gated_lgt = torch.where(gate_exp, ref_lgt, base_lgt)
            B_b       = end - start

            base_ce_all += float(F.cross_entropy(base_lgt, gt_b, reduction="sum"))
            gate_ce_all += float(F.cross_entropy(gated_lgt, gt_b, reduction="sum"))
            nc_all      += B_b
            base_acc1_all += int((base_lgt.argmax(1) == gt_b).sum())
            ref_acc1_all  += int((gated_lgt.argmax(1) == gt_b).sum())

            ig_all = g_sl
            if ig_all.any():
                ig_base_all += float(F.cross_entropy(base_lgt[ig_all], gt_b[ig_all], reduction="sum"))
                ig_ref_all  += float(F.cross_entropy(ref_lgt[ig_all],  gt_b[ig_all], reduction="sum"))
                ig_nc_all   += int(ig_all.sum())

            og_all = ~g_sl
            if og_all.any():
                og_base_all += float(F.cross_entropy(base_lgt[og_all], gt_b[og_all], reduction="sum"))
                og_gate_all += float(F.cross_entropy(gated_lgt[og_all], gt_b[og_all], reduction="sum"))
                og_nc_all   += int(og_all.sum())

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
                    ig_base_cov += float(F.cross_entropy(base_lgt[ig_cov], gt_b[ig_cov], reduction="sum"))
                    ig_ref_cov  += float(F.cross_entropy(ref_lgt[ig_cov],  gt_b[ig_cov], reduction="sum"))
                    ig_nc_cov   += int(ig_cov.sum())

                og_cov = cov & ~g_sl
                if og_cov.any():
                    og_base_cov += float(F.cross_entropy(base_lgt[og_cov], gt_b[og_cov], reduction="sum"))
                    og_gate_cov += float(F.cross_entropy(gated_lgt[og_cov], gt_b[og_cov], reduction="sum"))
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

    def _nll(ce, n):
        return ce / max(n, 1)

    fd_mean = float(np.mean(final_delta_norms)) if final_delta_norms else 0.0
    fd_max  = float(np.max(final_delta_norms))  if final_delta_norms else 0.0
    p_mean  = float(np.mean(path_norms))         if path_norms        else 0.0
    p_max   = float(np.max(path_norms))          if path_norms        else 0.0
    hp_mean = float(np.mean(h_prime_norms))      if h_prime_norms     else 0.0

    all_d   = list(final_delta_norms)
    for ld in layer_delta_norms:
        all_d.extend(ld)
    total_d_mean = float(np.mean(all_d)) if all_d else 0.0
    d_to_h       = fd_mean / (hp_mean + 1e-8)

    alpha_v = float(model.alpha.item())

    results = {
        "eval_mode": "running_bridge_full_vocab",
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
        "masked_cand_base_nll":     _nll(mc_base,  mc_nc),
        "masked_cand_gated_nll":    _nll(mc_gated, mc_nc),
        "masked_cand_gain":         _nll(mc_base, mc_nc) - _nll(mc_gated, mc_nc),
        "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL,
        "gate_rate":         gate_n    / max(total_n, 1),
        "coverage":          total_cov / max(total_n, 1),
        "num_examples":      total_n,
        "num_covered":       total_cov,
        "inside_gate_n_all": ig_nc_all,
        "inside_gate_n_cov": ig_nc_cov,
        "dataset_fingerprint": fp,
        "alpha":                    alpha_v,
        "final_delta_norm_mean":    fd_mean,
        "final_delta_norm_max":     fd_max,
        "total_delta_norm_mean":    total_d_mean,
        "path_norm_mean":           p_mean,
        "path_norm_max":            p_max,
        "delta_to_h_norm_ratio":    d_to_h,
        # keep delta_norm_mean/max as aliases for compat with existing eval CSV headers
        "delta_norm_mean":          fd_mean,
        "delta_norm_max":           fd_max,
        "gold_force_included_rate": 0.0,
        "selection_mode":           "no_candidate_selection",
        "eval_force_include_gold":  False,
    }
    for i, ld in enumerate(layer_delta_norms):
        results[f"layer_delta_norm_mean_{i}"] = float(np.mean(ld)) if ld else 0.0

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
        msg = f"full_vocab_eval_running_bridge CANONICAL MISMATCH [{variant_tag}]: " + "; ".join(issues)
        if fail_on_mismatch:
            raise RuntimeError(msg)
        print(f"  WARNING: {msg}")

    return results


@torch.no_grad()
def local_subset_eval_running_bridge(
    model:           RunningBridgeAdapter,
    val_cand_dir:    str,
    val_feat_dir:    str,
    tok_emb_w:       torch.Tensor,
    r2s_np:          np.ndarray,
    device,
    filter_kwargs:   Dict,
    eval_batch_size: int = 64,
) -> List[Dict]:
    model.eval()
    emb_w  = tok_emb_w.float().to(device)
    stats  = defaultdict(lambda: {"base_ce": 0.0, "ref_ce": 0.0,
                                   "base_a1": 0, "ref_a1": 0, "n": 0})

    for cand_path in sorted(glob.glob(os.path.join(val_cand_dir, "shard_*.pt"))):
        si        = int(os.path.basename(cand_path).replace("shard_", "").replace(".pt", ""))
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

            base_lgt              = h_p.float() @ emb_w.T
            h_ref, _, _, _        = model(h_p, h_l, ct, cf, cmask, emb_w,
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
            "subset":                 name,
            "n":                      s["n"],
            "full_vocab_base_nll":    s["base_ce"] / n,
            "full_vocab_refined_nll": s["ref_ce"]  / n,
            "full_vocab_gain":        s["base_ce"]/n - s["ref_ce"]/n,
            "base_acc1":              s["base_a1"] / n,
            "refined_acc1":           s["ref_a1"]  / n,
        })
    return rows


# ── Report ────────────────────────────────────────────────────────────────────

def _generate_report(output_dir, args, full_vocab_base_nll, final_metrics,
                     update_layer_idxs, layer_ids):
    def _load(p):
        return json.load(open(p)) if os.path.isfile(p) else None

    v1_bm  = _load("runs/path_refiner_bridge_adapter/bridge_boundary_v1_hardsampler/best_metrics.json")
    ml_bm  = _load("runs/path_refiner_midlayer_bridge/boundary_insert4_v1/best_metrics.json")
    ebr_bm = _load("runs/path_refiner_explicit_bridge_refiner/boundary_insert4_refiner2_path4_v1/best_metrics.json")
    fo_bm  = _load("runs/path_refiner_running_bridge/boundary_running_finalonly_v0/best_metrics.json")
    mw_bm  = _load("runs/path_refiner_running_bridge/boundary_running_multiwrite_v0/best_metrics.json")
    own_bm = _load(os.path.join(output_dir, "best_metrics.json"))

    no_ckpt   = final_metrics.get("no_improving_checkpoint", True)
    best_step = final_metrics.get("best_step", -1)

    def _fmtg(d, key):
        if d is None or d.get("no_improving_checkpoint", True):
            return "no_ckpt"
        v = d.get(key)
        return f"{v:+.6f}" if isinstance(v, float) else str(v) if v is not None else "N/A"

    compare_keys = [
        ("full_vocab_gain_all",            "fv_gain_all (PRIMARY)"),
        ("full_vocab_inside_gate_gain_all", "inside_gate_gain_all"),
        ("full_vocab_gain_covered",         "fv_gain_covered"),
        ("masked_cand_gain",                "masked_cand_gain"),
    ]

    lines = [
        "# Running Bridge Adapter — Run Report",
        "",
        f"**Variant:** `{os.path.basename(output_dir)}`",
        f"**write_mode:** `{args.write_mode}`",
        f"**Date:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| write_mode | {args.write_mode} |",
        f"| update_layers (arg) | {args.update_layers} |",
        f"| update_layer_idxs (resolved) | {update_layer_idxs} → layer_ids {[layer_ids[i] for i in update_layer_idxs]} |",
        f"| num_path_tokens | {args.num_path_tokens} |",
        f"| bridge_dim / bridge_update_layers / bridge_heads | {args.bridge_dim} / {args.bridge_update_layers} / {args.bridge_heads} |",
        f"| refiner_dim / refiner_layers / refiner_heads | {args.refiner_dim} / {args.refiner_layers} / {args.refiner_heads} |",
        f"| train_filter / gate_filter | {args.train_filter} / {args.gate_filter} |",
        f"| lr / lambda_kl / lambda_delta / lambda_path | {args.lr} / {args.lambda_kl} / {args.lambda_delta} / {args.lambda_path} |",
        f"| batch_size × grad_accum → eff_bs | {args.batch_size} × {args.grad_accum_steps} → {args.batch_size * args.grad_accum_steps} |",
        f"| steps / eval_every | {args.steps} / {args.eval_every} |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| full_vocab_base_nll_all (step 0) | {full_vocab_base_nll:.6f} |",
    ]

    if no_ckpt:
        lines += [f"| best checkpoint | N/A — no improvement |", "",
                  "**RESULT: No improving checkpoint.**"]
    else:
        gn = own_bm.get("full_vocab_gated_nll_all", "?") if own_bm else "?"
        ga = own_bm.get("full_vocab_gain_all", "?") if own_bm else "?"
        ig = own_bm.get("full_vocab_inside_gate_gain_all", "?") if own_bm else "?"
        mc = own_bm.get("masked_cand_gain", "?") if own_bm else "?"
        lines += [
            f"| full_vocab_gated_nll_all | {gn:.6f} |" if isinstance(gn, float) else f"| full_vocab_gated_nll_all | {gn} |",
            f"| full_vocab_gain_all (PRIMARY) | {ga:+.6f} |" if isinstance(ga, float) else f"| full_vocab_gain_all | {ga} |",
            f"| inside_gate_gain_all | {ig:+.6f} |" if isinstance(ig, float) else f"| inside_gate_gain_all | {ig} |",
            f"| masked_cand_gain | {mc:+.6f} |" if isinstance(mc, float) else f"| masked_cand_gain | {mc} |",
            f"| best_step | {best_step} |",
        ]

    lines += [
        "",
        "## Cross-Architecture Comparison",
        "",
        f"| Architecture | fv_gain_all | inside_gate_gain | masked_cand_gain |",
        f"|---|---|---|---|",
        f"| V1 BridgeResidualAdapter (boundary) | {_fmtg(v1_bm,'full_vocab_gain_all')} | {_fmtg(v1_bm,'full_vocab_inside_gate_gain_all')} | {_fmtg(v1_bm,'masked_cand_gain')} |",
        f"| MidLayer Bridge (insert4) | {_fmtg(ml_bm,'full_vocab_gain_all')} | {_fmtg(ml_bm,'full_vocab_inside_gate_gain_all')} | {_fmtg(ml_bm,'masked_cand_gain')} |",
        f"| ExplicitBridgeRefiner (insert4) | {_fmtg(ebr_bm,'full_vocab_gain_all')} | {_fmtg(ebr_bm,'full_vocab_inside_gate_gain_all')} | {_fmtg(ebr_bm,'masked_cand_gain')} |",
        f"| **RunningBridge final_only** | {_fmtg(fo_bm,'full_vocab_gain_all')} | {_fmtg(fo_bm,'full_vocab_inside_gate_gain_all')} | {_fmtg(fo_bm,'masked_cand_gain')} |",
        f"| **RunningBridge multi_write** | {_fmtg(mw_bm,'full_vocab_gain_all')} | {_fmtg(mw_bm,'full_vocab_inside_gate_gain_all')} | {_fmtg(mw_bm,'masked_cand_gain')} |",
        "",
        "## Success Criteria",
        "",
        "| Criterion | Threshold | Achieved |",
        "|-----------|-----------|---------|",
    ]
    gain = final_metrics.get("full_vocab_gain_all")
    for label, thresh in [("Weak", 0.0), ("Meaningful", 0.0033), ("Strong", 0.005), ("Very strong", 0.010)]:
        ok = (not no_ckpt) and isinstance(gain, float) and gain > thresh
        lines.append(f"| {label} | gain > +{thresh:.4f} | {'✓' if ok else '✗'} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "**If RunningBridge final_only beats prior models:**",
        "  Persistent path-state across layers helps even without mid-layer writes.",
        "  The trajectory of path decisions matters.",
        "",
        "**If multi_write beats final_only:**",
        "  Running causal injection (previous delta feeds next update) adds value.",
        "  The simulated causal chain is beneficial.",
        "",
        "**If neither beats prior models:**",
        "  Missing piece may be: actual memory-neighbor tokens, real backbone re-execution,",
        "  joint training/LoRA, better hard-token gate, or earlier insertion.",
    ]

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [report] written to {report_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_running(
    args,
    d_model:           int,
    n_ctx_layers:      int,
    n_fine:            int,
    n_super:           int,
    r2s_np:            np.ndarray,
    tok_emb_w:         torch.Tensor,
    device,
    update_layer_idxs: List[int],
    layer_ids:         List[int],
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    n_updates = len(update_layer_idxs)
    resolved  = [layer_ids[i] for i in update_layer_idxs]

    print(f"[train] variant               = running_bridge_adapter")
    print(f"[train] write_mode            = {args.write_mode}")
    print(f"[train] update_layers (arg)   = {args.update_layers}")
    print(f"[train] update_layer_idxs     = {update_layer_idxs}  → backbone blocks {resolved}")
    print(f"[train] num_path_tokens       = {args.num_path_tokens}")
    print(f"[train] bridge_update_layers  = {args.bridge_update_layers}  (per update step)")
    print(f"[train] refiner_layers        = {args.refiner_layers}")
    print(f"[train] decode_mode           = full_vocab")
    print(f"[train] selection_mode        = no_candidate_selection")
    print(f"[train] gold_force_included   = never (structural guarantee)")

    model = RunningBridgeAdapter(
        d_model              = d_model,
        bridge_dim           = args.bridge_dim,
        bridge_heads         = args.bridge_heads,
        bridge_update_layers = args.bridge_update_layers,
        refiner_dim          = args.refiner_dim,
        refiner_heads        = args.refiner_heads,
        refiner_layers       = args.refiner_layers,
        ff_mult              = args.ff_mult,
        n_fine               = n_fine,
        n_super              = n_super,
        top_fine_k           = args.top_fine_regions,
        top_super_k          = args.top_superregions,
        update_layer_idxs    = update_layer_idxs,
        num_path_tokens      = args.num_path_tokens,
        write_mode           = args.write_mode,
        dropout              = args.dropout,
        r2s_np               = r2s_np,
    ).to(device)

    n_upd_p   = sum(p.numel() for upd in model.update_modules for p in upd.parameters()
                    if p.requires_grad)
    n_ref_p   = sum(p.numel() for p in model.refiner.parameters() if p.requires_grad)
    n_write_p = (sum(p.numel() for proj in model.write_projs for p in proj.parameters())
                 if model.write_projs else 0)
    total_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] update_modules params = {n_upd_p:,}  (×{n_updates} independent modules)")
    print(f"[train] refiner params        = {n_ref_p:,}")
    print(f"[train] write_proj params     = {n_write_p:,}  (0 for final_only)")
    print(f"[train] total params          = {total_p:,}")

    cfg = {**vars(args), "update_layer_idxs": update_layer_idxs, "layer_ids": layer_ids,
           "resolved_update_blocks": resolved}
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    fail_hard     = args.fail_on_baseline_mismatch
    filter_kwargs = {"margin_thresh": args.margin_thresh,
                     "entropy_thresh": args.entropy_thresh}
    variant_tag   = (f"rb-{args.write_mode}-upd{'_'.join(str(b) for b in resolved)}"
                     f"-p{args.num_path_tokens}-r{args.refiner_layers}")
    tok_dev = tok_emb_w.float().to(device)

    if args.baseline_json and os.path.isfile(args.baseline_json):
        with open(args.baseline_json) as f:
            masked_bl = json.load(f)
        print(f"\n[train] Masked-candidate baseline (reference): "
              f"covered_nll = {masked_bl['covered_nll']:.6f}")
        print(f"  NOTE: full-vocab NLL != masked-candidate NLL. Do NOT compare them.\n")

    # ── Step-0 identity check ─────────────────────────────────────────────────
    full_vocab_base_nll: float = float("nan")

    if args.eval_before_train:
        print(f"\n[train] === step-0 identity check ({args.write_mode}) ===")
        g0 = full_vocab_eval_running_bridge(
            model, args.val_cand_dir, args.val_feat_dir,
            tok_dev, r2s_np, device,
            gate_filter_name = args.gate_filter,
            filter_kwargs    = filter_kwargs,
            fail_on_mismatch = fail_hard,
            eval_batch_size  = args.eval_batch_size,
            variant_tag      = variant_tag,
        )
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]

        nll_diff_all = abs(g0["full_vocab_gated_nll_all"]     - g0["full_vocab_base_nll_all"])
        nll_diff_cov = abs(g0["full_vocab_gated_nll_covered"] - g0["full_vocab_base_nll_covered"])
        mc_diff      = abs(g0["masked_cand_gated_nll"]        - g0["masked_cand_base_nll"])
        og_gate_diff = abs(g0["full_vocab_outside_gate_gated_nll_all"]
                           - g0["full_vocab_outside_gate_base_nll_all"])
        d_max        = g0["final_delta_norm_max"]

        print(f"  variant               = running_bridge_adapter")
        print(f"  write_mode            = {args.write_mode}")
        print(f"  update_layers         = {args.update_layers}  (resolved: {resolved})")
        print(f"  num_path_tokens       = {args.num_path_tokens}")
        print(f"  bridge_update_layers  = {args.bridge_update_layers}")
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
        print(f"  final_delta_norm_max (→ 0)   = {d_max:.2e}")
        print(f"  all_write_delta_norms = 0    (zero-init write_projs)" if args.write_mode == "multi_write" else "")
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
                "Refiner out_proj / write_projs must be zero-init.")
        if nll_diff_cov >= 1e-3:
            raise RuntimeError(f"Step-0 IDENTITY FAIL (_covered): diff={nll_diff_cov:.2e}.")
        if mc_diff >= 1e-3:
            raise RuntimeError(f"Step-0 IDENTITY FAIL (masked_cand): diff={mc_diff:.2e}.")
        if og_gate_diff >= 1e-5:
            raise RuntimeError(f"Step-0 outside-gate FAIL: diff={og_gate_diff:.2e} > 1e-5.")
        if d_max > 1e-6:
            raise RuntimeError(f"Step-0 IDENTITY FAIL: final_delta_norm_max={d_max:.2e} > 0.")

        print(f"  [step-0] IDENTITY PASS — running_bridge_adapter  "
              f"write_mode={args.write_mode}  gold_force_included_rate=0.0000")

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
                "write_mode":                  args.write_mode,
                "update_layers":               args.update_layers,
                "update_layer_idxs":           update_layer_idxs,
                "note": (
                    "full_vocab_base_nll_all is PRIMARY threshold. "
                    "Best checkpoint saved only if full_vocab_gated_nll_all < this."
                ),
            }, f, indent=2)

        identity_data: Dict = {
            "variant":             "running_bridge_adapter",
            "write_mode":          args.write_mode,
            "update_layers":       args.update_layers,
            "update_layer_idxs":   update_layer_idxs,
            "resolved_blocks":     resolved,
            "layer_ids":           layer_ids,
            "num_path_tokens":     args.num_path_tokens,
            "bridge_update_layers": args.bridge_update_layers,
            "refiner_layers":      args.refiner_layers,
            "nll_diff_all":        nll_diff_all,
            "nll_diff_covered":    nll_diff_cov,
            "mc_diff":             mc_diff,
            "og_gate_diff":        og_gate_diff,
            "delta_norm_max":      d_max,
            "identity_pass":       True,
            **{k: g0[k] for k in (
                "full_vocab_base_nll_all", "full_vocab_gated_nll_all",
                "full_vocab_base_nll_covered", "full_vocab_gated_nll_covered",
                "masked_cand_base_nll", "masked_cand_gated_nll",
                "path_norm_mean", "path_norm_max", "total_delta_norm_mean",
                "dataset_fingerprint", "alpha",
            )},
        }
        with open(os.path.join(args.output_dir, "debug_identity.json"), "w") as f:
            json.dump(identity_data, f, indent=2)
    else:
        print("[train] Computing full-vocab baseline (no --eval_before_train) ...")
        g0 = full_vocab_eval_running_bridge(
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

    print(f"\n[train] Best checkpoint threshold: "
          f"full_vocab_gated_nll_all < {full_vocab_base_nll:.6f}\n")

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

    train_fields = ["step", "ce", "kl", "delta_norm", "final_delta_norm",
                    "path_norm_mean", "path_norm_max", "n_train", "alpha", "lr"]
    eval_fields = [
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
        "alpha", "final_delta_norm_mean", "final_delta_norm_max",
        "total_delta_norm_mean", "path_norm_mean", "path_norm_max", "delta_to_h_norm_ratio",
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
                    loss, info = compute_running_bridge_loss(
                        model, batch, device, tok_dev, train_mask,
                        args.lambda_kl, args.lambda_delta, args.lambda_path,
                        args.kl_topk, amp_enabled=True)
            else:
                loss, info = compute_running_bridge_loss(
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

        avg_ce  = float(np.mean([d["ce"]               for d in accum_infos]))
        avg_kl  = float(np.mean([d["kl"]               for d in accum_infos]))
        avg_dn  = float(np.mean([d["delta_norm"]        for d in accum_infos]))
        avg_fdn = float(np.mean([d["final_delta_norm"]  for d in accum_infos]))
        avg_pn  = float(np.mean([d["path_norm_mean"]    for d in accum_infos]))
        avg_px  = float(np.mean([d["path_norm_max"]     for d in accum_infos]))
        tot_n   = sum(d["n_train"] for d in accum_infos)

        ema_ce  = avg_ce if ema_ce is None else 0.95 * ema_ce + 0.05 * avg_ce
        alpha_v = float(model.alpha.item())
        lr_now  = sched.get_last_lr()[0]

        train_csv.writerow({"step": step, "ce": avg_ce, "kl": avg_kl,
                            "delta_norm": avg_dn, "final_delta_norm": avg_fdn,
                            "path_norm_mean": avg_pn, "path_norm_max": avg_px,
                            "n_train": tot_n, "alpha": alpha_v, "lr": lr_now})
        if step % 100 == 0:
            train_logf.flush()
            print(f"  step={step:5d}  ema_ce={ema_ce:.4f}  "
                  f"ce={avg_ce:.4f}  kl={avg_kl:.4f}  "
                  f"fd_norm={avg_fdn:.4f}  p_norm={avg_pn:.3f}  "
                  f"alpha={alpha_v:.4f}  n_train={tot_n}  "
                  f"eff_bs={eff_bs}  t={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] step={step} ...")
            g_m = full_vocab_eval_running_bridge(
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
            print(f"  [eval] step={step}  write_mode={args.write_mode}")
            print(f"    PRIMARY (full-vocab, all):")
            print(f"    full_vocab_base_nll_all        = {g_m['full_vocab_base_nll_all']:.6f}")
            print(f"    full_vocab_gated_nll_all       = {gated_nll:.6f}  "
                  f"(threshold={full_vocab_base_nll:.6f})")
            print(f"    full_vocab_gain_all            = {fv_gain:+.6f}")
            print(f"    inside_gate_base_nll_all       = {g_m['full_vocab_inside_gate_base_nll_all']:.6f}")
            print(f"    inside_gate_ref_nll_all        = {g_m['full_vocab_inside_gate_ref_nll_all']:.6f}")
            print(f"    inside_gate_gain_all           = {g_m['full_vocab_inside_gate_gain_all']:+.6f}")
            print(f"    outside_gate_diff_all          = {og_diff:.2e}  (must be ~0)")
            print(f"    full_vocab_gated_nll_covered   = {g_m['full_vocab_gated_nll_covered']:.6f}  "
                  f"gain={g_m['full_vocab_gain_covered']:+.6f}")
            print(f"    masked_cand_gain               = {g_m['masked_cand_gain']:+.6f}  "
                  f"(MLP refs: +0.001284 global, +0.000530 hard-boundary)")
            print(f"    DIAGNOSTICS:")
            print(f"    alpha={g_m['alpha']:.4f}  "
                  f"fd_norm_mean={g_m['final_delta_norm_mean']:.4f}  "
                  f"fd_norm_max={g_m['final_delta_norm_max']:.4f}")
            print(f"    total_delta_mean={g_m['total_delta_norm_mean']:.4f}  "
                  f"path_mean={g_m['path_norm_mean']:.4f}  "
                  f"delta/h={g_m['delta_to_h_norm_ratio']:.4f}")
            print(f"    gold_force_included_rate       = {g_m['gold_force_included_rate']:.4f}")

            sub_rows = local_subset_eval_running_bridge(
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
                    "variant":                               "running_bridge_adapter",
                    "write_mode":                            args.write_mode,
                    "update_layers":                         args.update_layers,
                    "update_layer_idxs":                     update_layer_idxs,
                    "resolved_blocks":                       resolved,
                    "num_path_tokens":                       args.num_path_tokens,
                    "bridge_update_layers":                  args.bridge_update_layers,
                    "refiner_layers":                        args.refiner_layers,
                    "selection_mode":                        "no_candidate_selection",
                    "eval_force_include_gold":               False,
                    "gold_force_included_rate":              0.0,
                    "step":                                  step,
                    "train_filter":                          args.train_filter,
                    "gate_filter":                           args.gate_filter,
                    "full_vocab_base_nll_all":               full_vocab_base_nll,
                    "full_vocab_gated_nll_all":              gated_nll,
                    "full_vocab_gain_all":                   fv_gain,
                    "full_vocab_inside_gate_base_nll_all":   g_m["full_vocab_inside_gate_base_nll_all"],
                    "full_vocab_inside_gate_ref_nll_all":    g_m["full_vocab_inside_gate_ref_nll_all"],
                    "full_vocab_inside_gate_gain_all":       g_m["full_vocab_inside_gate_gain_all"],
                    "full_vocab_outside_gate_base_nll_all":  g_m["full_vocab_outside_gate_base_nll_all"],
                    "full_vocab_outside_gate_gated_nll_all": g_m["full_vocab_outside_gate_gated_nll_all"],
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
                    "gate_rate":                             g_m["gate_rate"],
                    "coverage":                              g_m["coverage"],
                    "num_examples":                          g_m["num_examples"],
                    "num_covered":                           g_m["num_covered"],
                    "alpha":                                 g_m["alpha"],
                    "final_delta_norm_mean":                 g_m["final_delta_norm_mean"],
                    "final_delta_norm_max":                  g_m["final_delta_norm_max"],
                    "total_delta_norm_mean":                 g_m["total_delta_norm_mean"],
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
                print(f"  [no improvement]  gated_nll={gated_nll:.6f} >= "
                      f"best={best_nll:.6f}  (baseline={full_vocab_base_nll:.6f})")

    torch.save({"step": args.steps, "model": model.state_dict(), "args": vars(args)},
               os.path.join(args.output_dir, "last_refiner.pt"))
    train_logf.close()
    eval_logf.close()
    subset_logf.close()

    final_metrics: Dict = {
        "no_improving_checkpoint": (best_step < 0),
        "best_step":               best_step,
        "best_full_vocab_gated_nll_all": best_nll if best_step >= 0 else None,
        "full_vocab_base_nll_all": full_vocab_base_nll,
        "full_vocab_gain_all":     full_vocab_base_nll - best_nll if best_step >= 0 else None,
        "full_vocab_inside_gate_gain_all": None,
        "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL,
        "write_mode":              args.write_mode,
        "update_layers":           args.update_layers,
        "update_layer_idxs":       update_layer_idxs,
    }
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
        with open(bm_path, "w") as f:
            json.dump({"no_improving_checkpoint": True,
                       "full_vocab_base_nll_all": full_vocab_base_nll,
                       "write_mode": args.write_mode,
                       "update_layers": args.update_layers,
                       "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL}, f, indent=2)
    else:
        print(f"\n[train] done  best_full_vocab_gated_nll_all={best_nll:.6f}  "
              f"gain={full_vocab_base_nll-best_nll:+.6f}  step={best_step}")

    _generate_report(args.output_dir, args, full_vocab_base_nll,
                     final_metrics, update_layer_idxs, layer_ids)


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

    update_layer_idxs = _resolve_update_layer_idxs(args.update_layers, layer_ids)
    print(f"  update_layers={args.update_layers} → "
          f"update_layer_idxs={update_layer_idxs}  "
          f"(backbone blocks {[layer_ids[i] for i in update_layer_idxs]})")

    if args.gate_filter is None:
        args.gate_filter = args.train_filter

    train_running(args, d_model, n_ctx_layers, n_fine, n_super, r2s_np, tok_emb_w,
                  device, update_layer_idxs, layer_ids)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",     required=True)
    p.add_argument("--train_cand_dir", required=True)
    p.add_argument("--val_cand_dir",   required=True)
    p.add_argument("--train_feat_dir", required=True)
    p.add_argument("--val_feat_dir",   required=True)
    p.add_argument("--baseline_json",  default=None)
    p.add_argument("--super_map",      default=None)
    p.add_argument("--output_dir",     required=True)

    p.add_argument("--write_mode",           default="final_only",
                   choices=["final_only", "multi_write"],
                   help="final_only: single write at end. multi_write: write at each update layer.")
    p.add_argument("--update_layers",        default="2,4",
                   help="Comma-separated backbone block indices for update steps (e.g. '2,4')")
    p.add_argument("--bridge_dim",           type=int,   default=256)
    p.add_argument("--bridge_update_layers", type=int,   default=1,
                   help="Transformer layers per RunningBridgeUpdate step")
    p.add_argument("--bridge_heads",         type=int,   default=4)
    p.add_argument("--refiner_dim",          type=int,   default=256)
    p.add_argument("--refiner_layers",       type=int,   default=2)
    p.add_argument("--refiner_heads",        type=int,   default=4)
    p.add_argument("--ff_mult",              type=int,   default=4)
    p.add_argument("--dropout",              type=float, default=0.0)
    p.add_argument("--top_fine_regions",     type=int,   default=24)
    p.add_argument("--top_superregions",     type=int,   default=8)
    p.add_argument("--num_path_tokens",      type=int,   default=4)

    p.add_argument("--train_filter",   default="boundary")
    p.add_argument("--gate_filter",    default=None)
    p.add_argument("--margin_thresh",  type=float, default=0.1)
    p.add_argument("--entropy_thresh", type=float, default=2.0)

    p.add_argument("--steps",            type=int,   default=5000)
    p.add_argument("--eval_every",       type=int,   default=1000)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--eval_batch_size",  type=int,   default=64)
    p.add_argument("--lr",               type=float, default=5e-5)
    p.add_argument("--lambda_kl",        type=float, default=0.2)
    p.add_argument("--lambda_delta",     type=float, default=3e-4)
    p.add_argument("--lambda_path",      type=float, default=0.0)
    p.add_argument("--kl_topk",          type=int,   default=512)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int,   default=2)

    p.add_argument("--amp",                      action="store_true")
    p.add_argument("--eval_before_train",        action="store_true")
    p.add_argument("--fail_on_baseline_mismatch",action="store_true")
    p.add_argument("--use_filtered_train_loader",action="store_true")
    p.add_argument("--train_covered_only",       action="store_true")
    p.add_argument("--device",                   default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
