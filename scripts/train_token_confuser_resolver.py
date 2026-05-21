#!/usr/bin/env python3
"""
Top-K Token Confuser Resolver V1.

Audit finding:
  region_right_token_wrong loss mass = 52.2 %
  → dominant failure is exact token disambiguation, not region discovery.

Core idea:
  base_logits = h_prime @ tok_emb.T                     (full-vocab)
  C = base top-K tokens                                  (no gold force)
  resolver_input = [CTX, ROUTER, MEM, TOKEN_1..TOKEN_K]
  delta = transformer → delta_head → (B, K)             (zero-init → identity)
  refined_logits = base_logits + scatter(alpha * delta)
  loss = CE(refined_logits, gold) + λ_rank * rank + λ_kl * KL + λ_δ * ||δ||²

Identity at step 0 guaranteed: delta_head zero-init → delta=0 → refined=base.
Gold token used ONLY as CE/rank target. Never in model inputs.
gold_force_included_rate = 0.0 (structural).

Usage:
    python scripts/train_token_confuser_resolver.py \\
        --small_ckpt  runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --train_cand_dir runs/path_refiner_clean/data/train_hgrid_K24 \\
        --val_cand_dir   runs/path_refiner_clean/data/val_hgrid_K24 \\
        --train_feat_dir runs/path_refiner_residual_interface/features/train_multilayer \\
        --val_feat_dir   runs/path_refiner_residual_interface/features/val_multilayer \\
        --baseline_json  runs/path_refiner_clean/baselines/saved_candidate_baseline.json \\
        --super_map  runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --region_map runs/region_maps_128/token_to_region.json \\
        --output_dir runs/token_confuser_resolver/top256_boundary_v1 \\
        --confuser_source base_topk --top_k 256 \\
        --train_filter boundary --gate_filter boundary --use_filtered_train_loader \\
        --resolver_dim 256 --resolver_layers 2 --resolver_heads 4 \\
        --batch_size 16 --grad_accum_steps 4 --steps 5000 --eval_every 1000 \\
        --lr 5e-5 --lambda_rank 0.5 --lambda_kl 0.1 --lambda_delta 1e-4 \\
        --rank_margin 0.1 --grad_clip 1.0 --amp \\
        --eval_before_train --fail_on_baseline_mismatch
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

KNOWN_BASE_NLL_ALL     = 3.754938
KNOWN_BASE_NLL_COVERED = 3.481444


# ── Token-to-region lookup ────────────────────────────────────────────────────

def load_token_to_region(region_map_path: str, vocab_size: int) -> Optional[np.ndarray]:
    if not region_map_path or not os.path.isfile(region_map_path):
        return None
    with open(region_map_path) as f:
        raw = json.load(f)
    t2r = np.full(vocab_size, -1, dtype=np.int32)
    for k, v in raw.items():
        tok = int(k)
        if 0 <= tok < vocab_size:
            t2r[tok] = int(v)
    return t2r


# ── Model ─────────────────────────────────────────────────────────────────────

class TokenDeltaResolver(nn.Module):
    """
    Small transformer over base top-K token candidates.

    Sequence: [CTX | ROUTER | MEM | TOKEN_1 .. TOKEN_K]
    CTX:   proj(h_prime ‖ mean(h_layers))
    ROUTER: prob-weighted fine_emb + scalar_proj(entropy, margin, top_prob)
    MEM:   same for memory
    TOKEN_j:
      tok_emb_proj(tok_emb[j])
      + score_proj(logit_j, rank_j/K, logprob_j)
      + fine_emb[t2r[j]]
      + super_emb[r2s[t2r[j]]]
      + support_proj(router_prob_j, mem_prob_j, in_router8_j, in_mem8_j)
      → layer_norm

    Output: delta_head (zero-init) on token positions → delta (B, K)
    Caller: refined_lgt = base_lgt + scatter(alpha * delta, topk_ids)
    """

    def __init__(
        self,
        d_backbone:  int,
        n_fine:      int,
        n_super:     int,
        r2s_np:      np.ndarray,
        t2r_np:      Optional[np.ndarray],
        top_k:       int   = 256,
        d_resolver:  int   = 256,
        n_layers:    int   = 2,
        n_heads:     int   = 4,
        ff_mult:     int   = 4,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.top_k     = top_k
        self.n_fine    = n_fine
        self.n_super   = n_super
        self.d_resolver = d_resolver

        # Context token: h_prime ‖ mean(h_layers)
        self.ctx_proj = nn.Linear(2 * d_backbone, d_resolver)

        # Shared region embeddings (used for router/mem summary AND per-token lookup)
        self.fine_emb  = nn.Embedding(n_fine  + 1, d_resolver, padding_idx=n_fine)
        self.super_emb = nn.Embedding(n_super + 1, d_resolver, padding_idx=n_super)

        # Router / memory scalar projections (3 scalars each)
        self.router_scalar = nn.Linear(3, d_resolver)
        self.mem_scalar    = nn.Linear(3, d_resolver)

        # Per-token feature projections
        self.tok_emb_proj  = nn.Linear(d_backbone, d_resolver, bias=False)
        self.score_proj    = nn.Linear(3, d_resolver)   # logit, rank/K, logprob
        self.support_proj  = nn.Linear(4, d_resolver)   # r_prob, m_prob, in_r8, in_m8
        self.tok_norm      = nn.LayerNorm(d_resolver)

        # Transformer (pre-LN, batch_first)
        enc = nn.TransformerEncoderLayer(
            d_model=d_resolver, nhead=n_heads,
            dim_feedforward=d_resolver * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)

        # Delta head — zero-init guarantees step-0 identity
        self.delta_head = nn.Linear(d_resolver, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        # Learnable global scale; identity held by delta_head zero-init, not alpha=0
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # Region lookup buffers
        V_buf = t2r_np.shape[0] if t2r_np is not None else 50257
        t2r_i = (t2r_np.astype(np.int32)
                 if t2r_np is not None
                 else np.full(V_buf, n_fine, dtype=np.int32))
        self.register_buffer("t2r", torch.from_numpy(t2r_i).long())
        self.register_buffer("r2s", torch.from_numpy(r2s_np.astype(np.int64)).long())

    # ── Special sequence tokens ───────────────────────────────────────────────

    def _router_tok(self, r_reg, r_prb, r_mar) -> torch.Tensor:
        r_s = r_reg.clamp(0, self.n_fine - 1)
        w   = r_prb / (r_prb.sum(1, keepdim=True) + 1e-8)
        pooled  = (self.fine_emb(r_s) * w.unsqueeze(-1)).sum(1)
        entropy = -(r_prb * torch.log(r_prb + 1e-9)).sum(1)
        sc = torch.stack([entropy, r_mar.float(), r_prb[:, 0]], dim=-1)
        return pooled + self.router_scalar(sc)

    def _mem_tok(self, m_reg, m_prb, m_mar) -> torch.Tensor:
        m_s = m_reg.clamp(0, self.n_fine - 1)
        w   = m_prb / (m_prb.sum(1, keepdim=True) + 1e-8)
        pooled  = (self.fine_emb(m_s) * w.unsqueeze(-1)).sum(1)
        entropy = -(m_prb * torch.log(m_prb + 1e-9)).sum(1)
        sc = torch.stack([entropy, m_mar.float(), m_prb[:, 0]], dim=-1)
        return pooled + self.mem_scalar(sc)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        h_prime:       torch.Tensor,  # (B, d_backbone)
        h_layers:      torch.Tensor,  # (B, n_layers, d_backbone)
        topk_ids:      torch.Tensor,  # (B, K)  detached
        topk_logits:   torch.Tensor,  # (B, K)  detached
        topk_logprobs: torch.Tensor,  # (B, K)  detached
        r_reg:         torch.Tensor,  # (B, K_r)
        r_prb:         torch.Tensor,  # (B, K_r)
        m_reg:         torch.Tensor,  # (B, K_m)
        m_prb:         torch.Tensor,  # (B, K_m)
        r_mar:         torch.Tensor,  # (B,)
        m_mar:         torch.Tensor,  # (B,)
        tok_emb_w:     torch.Tensor,  # (V, d_backbone) — frozen
    ) -> torch.Tensor:                # (B, K) delta; caller does patching
        B, K = topk_ids.shape
        device = h_prime.device

        # Context token
        h_mean = h_layers.float().mean(1)
        ctx = self.ctx_proj(torch.cat([h_prime.float(), h_mean], dim=-1))

        # Router / memory tokens
        r_tok = self._router_tok(r_reg, r_prb, r_mar)
        m_tok = self._mem_tok(m_reg, m_prb, m_mar)

        # Token embeddings
        V = tok_emb_w.shape[0]
        t_embs = F.embedding(topk_ids.clamp(0, V - 1), tok_emb_w.float())  # (B, K, d_bb)
        tok_repr = self.tok_emb_proj(t_embs)

        # Score features: [logit, rank/K, logprob]
        ranks_norm = (torch.arange(K, device=device, dtype=torch.float32)
                      .unsqueeze(0).expand(B, -1) / K)
        scores = torch.stack([topk_logits.float(), ranks_norm, topk_logprobs.float()], dim=-1)
        tok_repr = tok_repr + self.score_proj(scores)

        # Region features per token
        fine_ids = self.t2r[topk_ids.clamp(0, self.t2r.shape[0] - 1)]  # (B, K)
        fine_ids = fine_ids.masked_fill(fine_ids < 0, self.n_fine)      # unknown → padding_idx
        super_ids = self.r2s[fine_ids.clamp(0, self.n_fine - 1)].clamp(0, self.n_super - 1)
        super_ids = super_ids.masked_fill(fine_ids == self.n_fine, self.n_super)

        tok_repr = tok_repr + self.fine_emb(fine_ids)
        tok_repr = tok_repr + self.super_emb(super_ids)

        # Router / memory support per token
        K_r = r_reg.shape[1]
        K_m = m_reg.shape[1]
        r_s = r_reg.clamp(0, self.n_fine - 1)  # (B, K_r)
        m_s = m_reg.clamp(0, self.n_fine - 1)  # (B, K_m)

        match_r = (r_s.unsqueeze(2) == fine_ids.unsqueeze(1))        # (B, K_r, K)
        router_support = (r_prb.unsqueeze(2) * match_r.float()).sum(1)
        in_router8 = match_r[:, :min(8, K_r), :].any(1).float()

        match_m = (m_s.unsqueeze(2) == fine_ids.unsqueeze(1))        # (B, K_m, K)
        mem_support = (m_prb.unsqueeze(2) * match_m.float()).sum(1)
        in_mem8 = match_m[:, :min(8, K_m), :].any(1).float()

        support = torch.stack([router_support, mem_support, in_router8, in_mem8], dim=-1)
        tok_repr = tok_repr + self.support_proj(support)
        tok_repr = self.tok_norm(tok_repr)  # (B, K, d_resolver)

        # Transformer over [CTX | ROUTER | MEM | TOKEN_1..TOKEN_K]
        seq = torch.cat([ctx.unsqueeze(1), r_tok.unsqueeze(1),
                         m_tok.unsqueeze(1), tok_repr], dim=1)  # (B, K+3, d_resolver)
        out = self.transformer(seq)
        delta = self.delta_head(out[:, 3:, :]).squeeze(-1)  # (B, K)
        return delta


# ── Eval ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_vocab_eval_resolver(
    model:             TokenDeltaResolver,
    val_cand_dir:      str,
    val_feat_dir:      str,
    tok_emb_w:         torch.Tensor,
    device,
    gate_filter_name:  str,
    filter_kwargs:     Dict,
    fail_on_mismatch:  bool = False,
    eval_batch_size:   int  = 64,
    variant_tag:       str  = "resolver",
) -> Dict:
    model.eval()
    emb_w = tok_emb_w.float().to(device)
    V, K  = emb_w.shape[0], model.top_k

    # ── CE accumulators ──────────────────────────────────────────────────────
    base_ce_all = ref_ce_all = 0.0
    ig_base_all = ig_ref_all = 0.0
    og_base_all = og_ref_all = 0.0
    nc_all = ig_nc_all = og_nc_all = 0

    base_ce_cov = ref_ce_cov = 0.0
    ig_base_cov = ig_ref_cov = 0.0
    nc_cov = ig_nc_cov = 0

    mc_base = mc_ref = 0.0
    mc_nc   = 0
    total_n = total_cov = gate_n = 0
    base_acc1_all = ref_acc1_all = 0
    base_acc5_all = ref_acc5_all = 0

    # ── Confuser / rank accumulators ─────────────────────────────────────────
    gold_in_topK_all  = gold_in_topK_gate  = 0
    gold_rank_base    = gold_rank_ref      = 0.0
    rank_improved = rank_worsened = rank_unchanged = 0
    gold_delta_sum    = gold_delta_n       = 0.0
    delta_abs_sum = delta_n = 0.0
    top1_changed_n = changed_to_gold_n = changed_away_n = 0

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

        has_mem = "mem_topk_reg" in cs
        m_reg_t = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
        m_prb_t = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
        m_mar_t = cs["mem_margin"]   if has_mem else torch.zeros(N)

        for start in range(0, N, eval_batch_size):
            end = min(start + eval_batch_size, N)
            sl  = slice(start, end)
            Bb  = end - start

            h_p   = cs["h_prime"][sl].float().to(device)
            h_l   = fs["h_layers"][sl].float().to(device)
            ct    = cs["cand_tok"][sl].long().to(device)
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

            base_lgt = h_p.float() @ emb_w.T                          # (B, V)
            topk_logits, topk_ids = base_lgt.topk(K, dim=1)
            topk_lp    = F.log_softmax(base_lgt, dim=1).gather(1, topk_ids)

            delta = model(h_p, h_l, topk_ids, topk_logits, topk_lp,
                          r_reg, r_prb, m_reg, m_prb, r_mar, m_mar, emb_w)

            delta_src = model.alpha * delta
            delta_full = torch.scatter_add(
                torch.zeros(Bb, V, device=device), 1, topk_ids, delta_src.float())
            ref_lgt = base_lgt + delta_full                            # (B, V)

            gate_exp  = g_sl.unsqueeze(1).expand(-1, V)
            gated_lgt = torch.where(gate_exp, ref_lgt, base_lgt)

            # ── CE ─────────────────────────────────────────────────────────
            base_ce_all += float(F.cross_entropy(base_lgt,  gt_b, reduction="sum"))
            ref_ce_all  += float(F.cross_entropy(gated_lgt, gt_b, reduction="sum"))
            nc_all      += Bb

            if g_sl.any():
                ig = g_sl
                ig_base_all += float(F.cross_entropy(base_lgt[ig], gt_b[ig], reduction="sum"))
                ig_ref_all  += float(F.cross_entropy(ref_lgt[ig],  gt_b[ig], reduction="sum"))
                ig_nc_all   += int(ig.sum())

            og = ~g_sl
            if og.any():
                og_base_all += float(F.cross_entropy(base_lgt[og],   gt_b[og], reduction="sum"))
                og_ref_all  += float(F.cross_entropy(gated_lgt[og],  gt_b[og], reduction="sum"))
                og_nc_all   += int(og.sum())

            n_cov = int(cov.sum())
            if n_cov > 0:
                cov_gt      = gt_b[cov]
                base_ce_cov += float(F.cross_entropy(base_lgt[cov],  cov_gt, reduction="sum"))
                ref_ce_cov  += float(F.cross_entropy(gated_lgt[cov], cov_gt, reduction="sum"))
                nc_cov      += n_cov

                ig_cov = cov & g_sl
                if ig_cov.any():
                    ig_base_cov += float(F.cross_entropy(base_lgt[ig_cov], gt_b[ig_cov], reduction="sum"))
                    ig_ref_cov  += float(F.cross_entropy(ref_lgt[ig_cov],  gt_b[ig_cov], reduction="sum"))
                    ig_nc_cov   += int(ig_cov.sum())

                cl_b = base_lgt[cov].gather(1, ct[cov].clamp(min=0)).masked_fill(~cmask[cov], float("-inf"))
                mc_base += float(F.cross_entropy(cl_b, gidx[cov], reduction="sum"))
                cl_r = gated_lgt[cov].gather(1, ct[cov].clamp(min=0)).masked_fill(~cmask[cov], float("-inf"))
                mc_ref  += float(F.cross_entropy(cl_r, gidx[cov], reduction="sum"))
                mc_nc   += n_cov

                sum_cand   += int(cmask.sum())
                sum_gi_cov += int(gidx[cov].sum())
                sum_gt     += int(gt_b[cov].sum())

            # ── Accuracy ───────────────────────────────────────────────────
            base_top1 = base_lgt.argmax(1)
            ref_top1  = gated_lgt.argmax(1)
            base_acc1_all += int((base_top1 == gt_b).sum())
            ref_acc1_all  += int((ref_top1  == gt_b).sum())
            base_top5 = base_lgt.topk(5, dim=1)[1]
            ref_top5  = gated_lgt.topk(5, dim=1)[1]
            gt_col    = gt_b.unsqueeze(1)
            base_acc5_all += int((base_top5 == gt_col).any(1).sum())
            ref_acc5_all  += int((ref_top5  == gt_col).any(1).sum())

            # Top-1 changed / to-gold / away-from-gold
            changed = (base_top1 != ref_top1)
            top1_changed_n    += int(changed.sum())
            changed_to_gold_n += int((changed & (ref_top1 == gt_b)).sum())
            changed_away_n    += int((changed & (base_top1 == gt_b)).sum())

            # ── Confuser / rank ────────────────────────────────────────────
            in_topK = (topk_ids == gt_col).any(1)
            gold_in_topK_all  += int(in_topK.sum())
            gold_in_topK_gate += int((in_topK & g_sl).sum())

            bi = torch.arange(Bb, device=device)
            gold_base_lgt = base_lgt[bi,    gt_b].unsqueeze(1)
            gold_ref_lgt  = gated_lgt[bi,   gt_b].unsqueeze(1)
            rank_b = (1 + (base_lgt  > gold_base_lgt).sum(1)).float()
            rank_r = (1 + (gated_lgt > gold_ref_lgt ).sum(1)).float()
            gold_rank_base += float(rank_b.sum())
            gold_rank_ref  += float(rank_r.sum())
            rank_improved  += int((rank_r < rank_b).sum())
            rank_worsened  += int((rank_r > rank_b).sum())
            rank_unchanged += int((rank_r == rank_b).sum())

            # Delta at gold position
            if in_topK.any():
                gold_k = (topk_ids == gt_col).float().argmax(1)     # (B,)
                sub_delta = delta[in_topK].gather(1, gold_k[in_topK].unsqueeze(1)).squeeze(1)
                gold_delta_sum += float(sub_delta.sum())
                gold_delta_n   += int(in_topK.sum())

            delta_abs_sum += float(delta.abs().sum())
            delta_n       += Bb * K

            gate_n    += int(g_sl.sum())
            total_n   += Bb
            total_cov += n_cov

    fp_data = {"num_shards": len(cand_paths), "total_n": total_n, "total_cov": total_cov,
               "sum_cand_counts": sum_cand, "sum_gold_idx_cov": sum_gi_cov, "sum_gold_tok": sum_gt}
    fp = hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()[:16]

    def _nll(ce, n): return ce / max(n, 1)

    results = {
        "eval_mode":  "token_confuser_resolver_full_vocab",
        # All
        "full_vocab_base_nll_all":         _nll(base_ce_all, nc_all),
        "full_vocab_refined_nll_all":      _nll(ref_ce_all,  nc_all),
        "full_vocab_gain_all":             _nll(base_ce_all, nc_all) - _nll(ref_ce_all, nc_all),
        "full_vocab_inside_gate_base_nll_all":  _nll(ig_base_all, ig_nc_all),
        "full_vocab_inside_gate_ref_nll_all":   _nll(ig_ref_all,  ig_nc_all),
        "full_vocab_inside_gate_gain_all":      _nll(ig_base_all, ig_nc_all) - _nll(ig_ref_all,  ig_nc_all),
        "full_vocab_outside_gate_base_nll_all": _nll(og_base_all, og_nc_all),
        "full_vocab_outside_gate_ref_nll_all":  _nll(og_ref_all,  og_nc_all),
        "full_vocab_base_acc1_all":   base_acc1_all / max(nc_all, 1),
        "full_vocab_ref_acc1_all":    ref_acc1_all  / max(nc_all, 1),
        "full_vocab_base_acc5_all":   base_acc5_all / max(nc_all, 1),
        "full_vocab_ref_acc5_all":    ref_acc5_all  / max(nc_all, 1),
        # Covered
        "full_vocab_base_nll_covered":    _nll(base_ce_cov, nc_cov),
        "full_vocab_refined_nll_covered": _nll(ref_ce_cov,  nc_cov),
        "full_vocab_gain_covered":        _nll(base_ce_cov, nc_cov) - _nll(ref_ce_cov, nc_cov),
        "full_vocab_inside_gate_base_nll_covered": _nll(ig_base_cov, ig_nc_cov),
        "full_vocab_inside_gate_ref_nll_covered":  _nll(ig_ref_cov,  ig_nc_cov),
        "full_vocab_inside_gate_gain_covered":     _nll(ig_base_cov, ig_nc_cov) - _nll(ig_ref_cov, ig_nc_cov),
        # Masked-candidate
        "masked_cand_base_nll":     _nll(mc_base, mc_nc),
        "masked_cand_ref_nll":      _nll(mc_ref,  mc_nc),
        "masked_cand_gain":         _nll(mc_base, mc_nc) - _nll(mc_ref, mc_nc),
        "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL,
        # Dataset stats
        "coverage":      total_cov / max(total_n, 1),
        "gate_rate":     gate_n    / max(total_n, 1),
        "num_examples":  total_n,
        "num_covered":   total_cov,
        "inside_gate_n_all": ig_nc_all,
        "inside_gate_n_cov": ig_nc_cov,
        "dataset_fingerprint": fp,
        "alpha": float(model.alpha.item()),
        # Confuser / rank metrics
        "gold_in_topK_rate_all":   gold_in_topK_all  / max(total_n, 1),
        "gold_in_topK_rate_gate":  gold_in_topK_gate / max(gate_n,  1),
        "gold_rank_base_mean":     gold_rank_base     / max(total_n, 1),
        "gold_rank_ref_mean":      gold_rank_ref      / max(total_n, 1),
        "gold_rank_improved_rate": rank_improved  / max(total_n, 1),
        "gold_rank_worsened_rate": rank_worsened  / max(total_n, 1),
        "top1_acc_base":           base_acc1_all  / max(nc_all, 1),
        "top1_acc_ref":            ref_acc1_all   / max(nc_all, 1),
        "top5_acc_base":           base_acc5_all  / max(nc_all, 1),
        "top5_acc_ref":            ref_acc5_all   / max(nc_all, 1),
        "top1_changed_rate":       top1_changed_n     / max(total_n, 1),
        "changed_to_gold_rate":    changed_to_gold_n  / max(total_n, 1),
        "changed_away_from_gold_rate": changed_away_n / max(total_n, 1),
        "mean_gold_delta_if_in_topK": gold_delta_sum / max(gold_delta_n, 1),
        "mean_delta_abs":          delta_abs_sum       / max(delta_n, 1),
        "gold_force_included_rate": 0.0,
    }

    # Canonical invariant check
    fp_ok  = (fp        == CANONICAL_FINGERPRINT)
    n_ok   = (total_n   == CANONICAL_NUM_EXAMPLES)
    nc_ok  = (total_cov == CANONICAL_NUM_COVERED)
    cov_d  = abs(results["coverage"] - CANONICAL_COVERAGE)
    if not (fp_ok and n_ok and nc_ok and cov_d < 1e-6):
        issues = []
        if not fp_ok:    issues.append(f"fingerprint {fp!r} != {CANONICAL_FINGERPRINT!r}")
        if not n_ok:     issues.append(f"num_examples {total_n} != {CANONICAL_NUM_EXAMPLES}")
        if not nc_ok:    issues.append(f"num_covered {total_cov} != {CANONICAL_NUM_COVERED}")
        if cov_d >= 1e-6: issues.append(f"coverage diff {cov_d:.2e}")
        msg = f"full_vocab_eval_resolver [{variant_tag}] CANONICAL MISMATCH: " + "; ".join(issues)
        if fail_on_mismatch:
            raise RuntimeError(msg)
        print(f"  WARNING: {msg}")

    model.train()
    return results


@torch.no_grad()
def local_subset_eval_resolver(
    model:           TokenDeltaResolver,
    val_cand_dir:    str,
    val_feat_dir:    str,
    tok_emb_w:       torch.Tensor,
    device,
    filter_kwargs:   Dict,
    eval_batch_size: int = 64,
) -> List[Dict]:
    model.eval()
    emb_w = tok_emb_w.float().to(device)
    V, K  = emb_w.shape[0], model.top_k
    stats = defaultdict(lambda: {"base_ce": 0.0, "ref_ce": 0.0,
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

        has_mem = "mem_topk_reg" in cs
        m_reg_t = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
        m_prb_t = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
        m_mar_t = cs["mem_margin"]   if has_mem else torch.zeros(N)

        for start in range(0, N, eval_batch_size):
            end = min(start + eval_batch_size, N)
            sl  = slice(start, end)
            Bb  = end - start

            h_p   = cs["h_prime"][sl].float().to(device)
            h_l   = fs["h_layers"][sl].float().to(device)
            cov   = cs["covered"][sl].bool()
            gt_b  = cs["gold_token"][sl].long().to(device)
            r_reg = cs["router_topk_reg"][sl].long().to(device)
            r_prb = cs["router_topk_prb"][sl].float().to(device)
            m_reg = m_reg_t[sl].long().to(device)
            m_prb = m_prb_t[sl].float().to(device)
            r_mar = cs["router_margin"][sl].float().to(device)
            m_mar = m_mar_t[sl].float().to(device)

            base_lgt = h_p.float() @ emb_w.T
            topk_logits, topk_ids = base_lgt.topk(K, dim=1)
            topk_lp = F.log_softmax(base_lgt, dim=1).gather(1, topk_ids)

            delta    = model(h_p, h_l, topk_ids, topk_logits, topk_lp,
                             r_reg, r_prb, m_reg, m_prb, r_mar, m_mar, emb_w)
            delta_full = torch.scatter_add(
                torch.zeros(Bb, V, device=device), 1, topk_ids,
                (model.alpha * delta).float())
            ref_lgt = base_lgt + delta_full

            cov_np = cov.numpy().astype(bool)
            for name in EVAL_SUBSETS:
                sub = torch.from_numpy(
                    subset_masks[name][start:end].numpy() & cov_np).to(device)
                if not sub.any():
                    continue
                s = stats[name]
                sub_gt = gt_b[sub]
                s["base_ce"] += float(F.cross_entropy(base_lgt[sub], sub_gt, reduction="sum"))
                s["ref_ce"]  += float(F.cross_entropy(ref_lgt[sub],  sub_gt, reduction="sum"))
                s["base_a1"] += int((base_lgt[sub].argmax(1) == sub_gt).sum())
                s["ref_a1"]  += int((ref_lgt[sub].argmax(1)  == sub_gt).sum())
                s["n"]       += int(sub.sum())

    model.train()
    rows = []
    for name in EVAL_SUBSETS:
        s = stats[name]
        n = max(s["n"], 1)
        rows.append({
            "subset":               name,
            "n":                    s["n"],
            "full_vocab_base_nll":  s["base_ce"] / n,
            "full_vocab_ref_nll":   s["ref_ce"]  / n,
            "full_vocab_gain":      s["base_ce"] / n - s["ref_ce"] / n,
            "base_acc1":            s["base_a1"] / n,
            "ref_acc1":             s["ref_a1"]  / n,
        })
    return rows


# ── Loss ─────────────────────────────────────────────────────────────────────

def compute_resolver_loss(
    model:         TokenDeltaResolver,
    batch:         Dict,
    device,
    tok_emb_w:     torch.Tensor,
    train_mask:    Optional[torch.Tensor],
    lambda_rank:   float,
    lambda_kl:     float,
    lambda_delta:  float,
    rank_margin:   float,
    kl_topk:       int,
    amp_enabled:   bool = False,
) -> Tuple[Optional[torch.Tensor], Dict]:

    h_p   = batch["h_prime"].float().to(device)
    h_l   = batch["h_layers"].float().to(device)
    gt_b  = batch["gold_token"].long().to(device)
    r_reg = batch["r_topk_reg"].long().to(device)
    r_prb = batch["r_topk_prb"].float().to(device)
    m_reg = batch["m_topk_reg"].long().to(device)
    m_prb = batch["m_topk_prb"].float().to(device)
    r_mar = batch["r_margin"].float().to(device)
    m_mar = batch["m_margin"].float().to(device)
    emb_w = tok_emb_w.float().to(device)

    if train_mask is not None:
        mask = train_mask.to(device)
        if not mask.any():
            return None, {}
        h_p, h_l, gt_b = h_p[mask], h_l[mask], gt_b[mask]
        r_reg, r_prb   = r_reg[mask], r_prb[mask]
        m_reg, m_prb   = m_reg[mask], m_prb[mask]
        r_mar, m_mar   = r_mar[mask], m_mar[mask]

    B = h_p.shape[0]
    if B == 0:
        return None, {}

    V, K = emb_w.shape[0], model.top_k

    # Base logits — h_p is data (no grad), emb_w is frozen
    base_lgt_f   = h_p.float() @ emb_w.T
    topk_logits, topk_ids = base_lgt_f.topk(K, dim=1)
    topk_lp = F.log_softmax(base_lgt_f, dim=1).gather(1, topk_ids)

    # Forward through resolver (trainable)
    delta = model(h_p, h_l, topk_ids.detach(), topk_logits.detach(), topk_lp.detach(),
                  r_reg, r_prb, m_reg, m_prb, r_mar, m_mar, emb_w)

    # Build refined logits: scatter (alpha * delta) into zeros, add to base
    delta_src  = model.alpha * delta
    delta_full = torch.scatter_add(
        torch.zeros(B, V, device=device), 1, topk_ids, delta_src)
    refined_lgt = base_lgt_f + delta_full

    # ── CE loss ───────────────────────────────────────────────────────────────
    ce_loss = F.cross_entropy(refined_lgt, gt_b)

    # ── Rank loss (only where gold in confuser set) ───────────────────────────
    rank_loss = refined_lgt.new_zeros(1).squeeze()
    in_topK   = (topk_ids == gt_b.unsqueeze(1)).any(1)     # (B,)
    gold_in_topK_rate = float(in_topK.float().mean().item())

    if in_topK.any() and lambda_rank > 0:
        sub_ids = topk_ids[in_topK]        # (n, K)
        sub_gt  = gt_b[in_topK]            # (n,)
        sub_ref = refined_lgt[in_topK]     # (n, V)

        # Gold refined logit
        gold_ref_lgt = sub_ref.gather(1, sub_gt.unsqueeze(1)).squeeze(1)

        # Hardest wrong token: first non-gold position in top-K
        wrong_mask = (sub_ids != sub_gt.unsqueeze(1))   # (n, K)
        wrong_idx  = wrong_mask.float().argmax(1)        # (n,)
        neg_toks   = sub_ids.gather(1, wrong_idx.unsqueeze(1)).squeeze(1)
        neg_ref_lgt = sub_ref.gather(1, neg_toks.unsqueeze(1)).squeeze(1)

        rank_loss = F.relu(rank_margin - gold_ref_lgt + neg_ref_lgt).mean()

    # ── KL loss over top-K distribution ──────────────────────────────────────
    kl_loss = refined_lgt.new_zeros(1).squeeze()
    if lambda_kl > 0:
        kk = min(kl_topk, K)
        base_kk_lp = F.log_softmax(topk_logits[:, :kk].detach(), dim=-1)
        ref_kk_lgt = refined_lgt.gather(1, topk_ids[:, :kk])
        ref_kk_lp  = F.log_softmax(ref_kk_lgt, dim=-1)
        kl_loss = F.kl_div(ref_kk_lp, base_kk_lp.exp().detach(), reduction="batchmean")

    # ── Delta regularization ─────────────────────────────────────────────────
    delta_reg = delta.pow(2).mean() if lambda_delta > 0 else refined_lgt.new_zeros(1).squeeze()

    loss = ce_loss + lambda_rank * rank_loss + lambda_kl * kl_loss + lambda_delta * delta_reg

    info = {
        "ce":               float(ce_loss.item()),
        "rank_loss":        float(rank_loss.item()),
        "kl_loss":          float(kl_loss.item()),
        "delta_reg":        float(delta_reg.item()),
        "delta_abs_mean":   float(delta.abs().mean().item()),
        "delta_abs_max":    float(delta.abs().max().item()),
        "gold_in_topK_rate": gold_in_topK_rate,
        "alpha":            float(model.alpha.item()),
        "n_train":          B,
    }
    return loss, info


# ── Report ────────────────────────────────────────────────────────────────────

def _generate_report(output_dir: str, args, full_vocab_base_nll: float,
                     final_metrics: Dict) -> None:
    def _load(p):
        return json.load(open(p)) if p and os.path.isfile(p) else None

    v1_bm  = _load("runs/path_refiner_bridge_adapter/bridge_boundary_v1_hardsampler/best_metrics.json")
    ml_bm  = _load("runs/path_refiner_midlayer_bridge/boundary_insert4_v1/best_metrics.json")
    ebr_bm = _load("runs/path_refiner_explicit_bridge_refiner/boundary_insert4_refiner2_path4_v1/best_metrics.json")
    fo_bm  = _load("runs/path_refiner_running_bridge/boundary_running_finalonly_v0/best_metrics.json")
    own_bm = _load(os.path.join(output_dir, "best_metrics.json"))

    def _g(d, key, fmt=".6f"):
        if d is None or d.get("no_improving_checkpoint", True):
            return "  (no ckpt)"
        v = d.get(key)
        return f"{v:{fmt}}" if v is not None else "  (missing)"

    no_ckpt   = final_metrics.get("no_improving_checkpoint", True)
    best_step = final_metrics.get("best_step", -1)
    own_gain  = final_metrics.get("best_full_vocab_gain_all", float("nan"))
    own_ig    = final_metrics.get("best_inside_gate_gain", float("nan"))

    lines = [
        "# Token Confuser Resolver V1 — Report",
        "",
        "## 1. Architecture",
        f"  confuser_source = {args.confuser_source}",
        f"  top_k           = {args.top_k}",
        f"  resolver_dim    = {args.resolver_dim}",
        f"  resolver_layers = {args.resolver_layers}",
        f"  resolver_heads  = {args.resolver_heads}",
        f"  train_filter    = {args.train_filter}",
        f"  gate_filter     = {getattr(args, 'gate_filter', args.train_filter)}",
        f"  lambda_rank     = {args.lambda_rank}",
        f"  lambda_kl       = {args.lambda_kl}",
        f"  lambda_delta    = {args.lambda_delta}",
        f"  rank_margin     = {args.rank_margin}",
        "",
        "## 2. Step-0 Identity",
        f"  gold_force_included_rate = 0.0000  (structural guarantee)",
        f"  delta = 0 at init  (delta_head zero-init)",
        f"  full_vocab_base_nll_all (canonical) = {KNOWN_BASE_NLL_ALL:.6f}",
        "",
        "## 3. Confuser Coverage (gold_in_top256)",
        f"  gold_in_topK_rate_all  = {own_bm.get('gold_in_topK_rate_all', float('nan')):.4f}" if own_bm else "  (no checkpoint)",
        f"  gold_in_topK_rate_gate = {own_bm.get('gold_in_topK_rate_gate', float('nan')):.4f}" if own_bm else "",
        "",
        "## 4. Primary Full-Vocab NLL Results",
        f"  full_vocab_base_nll_all  = {full_vocab_base_nll:.6f}",
        f"  full_vocab_ref_nll_all   = {_g(own_bm, 'full_vocab_refined_nll_all')}",
        f"  full_vocab_gain_all      = {_g(own_bm, 'full_vocab_gain_all')}",
        f"  best_step = {best_step}",
        "",
        "## 5. Inside-Gate NLL",
        f"  inside_gate_base_nll_all = {_g(own_bm, 'full_vocab_inside_gate_base_nll_all')}",
        f"  inside_gate_ref_nll_all  = {_g(own_bm, 'full_vocab_inside_gate_ref_nll_all')}",
        f"  inside_gate_gain_all     = {_g(own_bm, 'full_vocab_inside_gate_gain_all')}",
        "",
        "## 6. Rank Movement",
        f"  gold_rank_base_mean      = {_g(own_bm, 'gold_rank_base_mean', '.1f')}",
        f"  gold_rank_ref_mean       = {_g(own_bm, 'gold_rank_ref_mean', '.1f')}",
        f"  gold_rank_improved_rate  = {_g(own_bm, 'gold_rank_improved_rate')}",
        f"  gold_rank_worsened_rate  = {_g(own_bm, 'gold_rank_worsened_rate')}",
        f"  top1_acc_base            = {_g(own_bm, 'top1_acc_base')}",
        f"  top1_acc_ref             = {_g(own_bm, 'top1_acc_ref')}",
        f"  top5_acc_base            = {_g(own_bm, 'top5_acc_base')}",
        f"  top5_acc_ref             = {_g(own_bm, 'top5_acc_ref')}",
        f"  changed_to_gold_rate     = {_g(own_bm, 'changed_to_gold_rate')}",
        f"  changed_away_from_gold   = {_g(own_bm, 'changed_away_from_gold_rate')}",
        "",
        "## 7. Bucket Results (from local_subset_eval)",
        "  (see bucket_eval.csv for per-subset results)",
        "",
        "## 8. Comparison to Bridge/Refiner Family",
        f"  Bridge V1          gain = {_g(v1_bm,  'full_vocab_gain_all')}",
        f"  MidLayer           gain = {_g(ml_bm,  'full_vocab_gain_all')}",
        f"  ExplicitBridgeRef  gain = {_g(ebr_bm, 'full_vocab_gain_all')}",
        f"  RunningBridge FO   gain = {_g(fo_bm,  'full_vocab_gain_all')}",
        f"  TokenConfuserRes   gain = {_g(own_bm, 'full_vocab_gain_all')}  ← this run",
        "",
        "## 9. Token Disambiguation Verdict",
    ]

    if own_bm and not own_bm.get("no_improving_checkpoint", True):
        ig = own_bm.get("full_vocab_inside_gate_gain_all", 0.0)
        rank_imp = own_bm.get("gold_rank_improved_rate", 0.0)
        rank_wor = own_bm.get("gold_rank_worsened_rate", 0.0)
        top1_b   = own_bm.get("top1_acc_base", 0.0)
        top1_r   = own_bm.get("top1_acc_ref",  0.0)
        to_gold  = own_bm.get("changed_to_gold_rate", 0.0)
        away     = own_bm.get("changed_away_from_gold_rate", 0.0)
        gain     = own_bm.get("full_vocab_gain_all", 0.0)

        success = (rank_imp > rank_wor and top1_r > top1_b and to_gold > away)
        if gain > 0.010:
            verdict = "STRONG SUCCESS: full_vocab_gain > +0.010."
        elif gain > 0.005:
            verdict = "MEANINGFUL SUCCESS: full_vocab_gain > +0.005."
        elif gain > 0.0033:
            verdict = "WEAK SUCCESS: beats bridge/refiner ceiling (+0.0033)."
        elif gain > 0.0:
            verdict = "MARGINAL: small gain, below bridge ceiling."
        else:
            verdict = "NO GAIN: resolver did not improve NLL."

        lines += [
            f"  {verdict}",
            f"  Token-disambiguation signal {'detected' if success else 'NOT detected'}.",
        ]
        if success:
            lines.append(
                "  → Recommended: build stronger token-disambiguation module, "
                "add memory-neighbor next-token evidence.")
        else:
            lines.append(
                "  → Missing signal likely not in cached h_prime/top-K features. "
                "Consider context-id dataset or retrieval-neighbor tokens.")
    else:
        lines.append("  (no improving checkpoint saved — baseline not beaten)")

    lines += [
        "",
        "## 10. Recommended Next Action",
        "  See verdict above.",
        "",
        "---",
        f"NOTE: full-vocab NLL ({KNOWN_BASE_NLL_ALL:.4f}) ≠ masked-candidate NLL "
        f"({MASKED_CAND_BASELINE_NLL:.4f}). Do not compare them.",
        "NOTE: confuser set is base top-K only. gold_force_included_rate = 0.",
    ]

    rpath = os.path.join(output_dir, "report.md")
    with open(rpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] written → {rpath}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_resolver(
    args,
    d_model:   int,
    n_layers:  int,
    n_fine:    int,
    n_super:   int,
    r2s_np:    np.ndarray,
    t2r_np:    Optional[np.ndarray],
    tok_emb_w: torch.Tensor,
    device,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[train] variant         = token_confuser_resolver")
    print(f"[train] confuser_source = {args.confuser_source}")
    print(f"[train] top_k           = {args.top_k}")
    print(f"[train] resolver_dim    = {args.resolver_dim}")
    print(f"[train] resolver_layers = {args.resolver_layers}")
    print(f"[train] resolver_heads  = {args.resolver_heads}")
    print(f"[train] gold_force_included_rate = 0.0  (structural)")

    model = TokenDeltaResolver(
        d_backbone  = d_model,
        n_fine      = n_fine,
        n_super     = n_super,
        r2s_np      = r2s_np,
        t2r_np      = t2r_np,
        top_k       = args.top_k,
        d_resolver  = args.resolver_dim,
        n_layers    = args.resolver_layers,
        n_heads     = args.resolver_heads,
        ff_mult     = args.ff_mult,
        dropout     = args.dropout,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] total params    = {total_p:,}")

    cfg = vars(args).copy()
    cfg.update({"n_fine": n_fine, "n_super": n_super, "d_model": d_model, "n_ctx_layers": n_layers})
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    fail_hard     = args.fail_on_baseline_mismatch
    filter_kwargs = {"margin_thresh": args.margin_thresh, "entropy_thresh": args.entropy_thresh}
    gate_f        = args.gate_filter or args.train_filter
    variant_tag   = f"tcr-k{args.top_k}-d{args.resolver_dim}-l{args.resolver_layers}"
    tok_dev       = tok_emb_w.float().to(device)

    if args.baseline_json and os.path.isfile(args.baseline_json):
        with open(args.baseline_json) as f:
            masked_bl = json.load(f)
        print(f"\n[train] Masked-candidate baseline: covered_nll = {masked_bl['covered_nll']:.6f}")
        print("  NOTE: full-vocab NLL != masked-candidate NLL. Do NOT compare them.\n")

    # ── Step-0 identity check ─────────────────────────────────────────────────
    full_vocab_base_nll: float = float("nan")

    if args.eval_before_train:
        print(f"\n[train] === step-0 identity check ===")
        g0 = full_vocab_eval_resolver(
            model, args.val_cand_dir, args.val_feat_dir,
            tok_dev, device,
            gate_filter_name = gate_f,
            filter_kwargs    = filter_kwargs,
            fail_on_mismatch = fail_hard,
            eval_batch_size  = args.eval_batch_size,
            variant_tag      = variant_tag,
        )
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]

        nll_diff_all  = abs(g0["full_vocab_refined_nll_all"]     - g0["full_vocab_base_nll_all"])
        nll_diff_cov  = abs(g0["full_vocab_refined_nll_covered"] - g0["full_vocab_base_nll_covered"])
        mc_diff       = abs(g0["masked_cand_ref_nll"]            - g0["masked_cand_base_nll"])
        og_diff       = abs(g0["full_vocab_outside_gate_ref_nll_all"]
                            - g0["full_vocab_outside_gate_base_nll_all"])
        d_max = g0.get("mean_delta_abs", 0.0)  # should be 0 at step 0

        print(f"  variant               = token_confuser_resolver")
        print(f"  confuser_source       = {args.confuser_source}")
        print(f"  top_k                 = {args.top_k}")
        print(f"  gold_force_included_rate = 0.0000")
        print(f"  delta_max_abs            = {d_max:.2e}")
        print(f"  full_vocab_base_nll_all      = {g0['full_vocab_base_nll_all']:.6f}")
        print(f"  full_vocab_refined_nll_all   = {g0['full_vocab_refined_nll_all']:.6f}")
        print(f"  diff_all (< 1e-3)            = {nll_diff_all:.2e}")
        print(f"  full_vocab_base_nll_covered  = {g0['full_vocab_base_nll_covered']:.6f}")
        print(f"  diff_covered (< 1e-3)        = {nll_diff_cov:.2e}")
        print(f"  masked_cand_base_nll         = {g0['masked_cand_base_nll']:.6f}  "
              f"(canonical = {MASKED_CAND_BASELINE_NLL:.6f})")
        print(f"  masked_cand_diff (< 1e-3)    = {mc_diff:.2e}")
        print(f"  outside_gate_diff (< 1e-5)   = {og_diff:.2e}")
        print(f"  gold_in_topK_rate_all        = {g0['gold_in_topK_rate_all']:.4f}")
        print(f"  gold_in_topK_rate_gate       = {g0['gold_in_topK_rate_gate']:.4f}")
        print(f"  alpha                        = {g0['alpha']:.4f}")

        mc_ref_diff = abs(g0["masked_cand_base_nll"] - MASKED_CAND_BASELINE_NLL)
        if mc_ref_diff > 1e-3:
            raise RuntimeError(
                f"Step-0: masked_cand_base_nll={g0['masked_cand_base_nll']:.6f} "
                f"!= canonical {MASKED_CAND_BASELINE_NLL:.6f} (diff={mc_ref_diff:.2e}).")
        if nll_diff_all >= 1e-3:
            raise RuntimeError(
                f"Step-0 IDENTITY FAIL (all): diff={nll_diff_all:.2e}. "
                "delta_head must be zero-init.")
        if nll_diff_cov >= 1e-3:
            raise RuntimeError(f"Step-0 IDENTITY FAIL (covered): diff={nll_diff_cov:.2e}.")
        if mc_diff >= 1e-3:
            raise RuntimeError(f"Step-0 IDENTITY FAIL (masked_cand): diff={mc_diff:.2e}.")
        if og_diff >= 1e-5:
            raise RuntimeError(f"Step-0 outside-gate FAIL: diff={og_diff:.2e} > 1e-5.")

        print(f"  [step-0] IDENTITY PASS — token_confuser_resolver  "
              f"top_k={args.top_k}  gold_force_included_rate=0.0000")

        identity_data = {
            "variant": "token_confuser_resolver",
            "confuser_source": args.confuser_source,
            "top_k": args.top_k,
            "resolver_dim": args.resolver_dim,
            "nll_diff_all": nll_diff_all,
            "nll_diff_covered": nll_diff_cov,
            "mc_diff": mc_diff,
            "og_diff": og_diff,
            "delta_max_abs": d_max,
            "identity_pass": True,
            "gold_in_topK_rate_all":  g0["gold_in_topK_rate_all"],
            "gold_in_topK_rate_gate": g0["gold_in_topK_rate_gate"],
            **{k: g0[k] for k in (
                "full_vocab_base_nll_all", "full_vocab_refined_nll_all",
                "full_vocab_base_nll_covered", "full_vocab_refined_nll_covered",
                "masked_cand_base_nll", "masked_cand_ref_nll",
                "dataset_fingerprint", "alpha",
            )},
        }
        with open(os.path.join(args.output_dir, "debug_identity.json"), "w") as f:
            json.dump(identity_data, f, indent=2)
        with open(os.path.join(args.output_dir, "full_vocab_baseline.json"), "w") as f:
            json.dump({
                "full_vocab_base_nll_all":     full_vocab_base_nll,
                "full_vocab_base_nll_covered": g0["full_vocab_base_nll_covered"],
                "masked_cand_base_nll":        g0["masked_cand_base_nll"],
                "masked_cand_baseline_ref":    MASKED_CAND_BASELINE_NLL,
                "coverage": g0["coverage"], "dataset_fingerprint": g0["dataset_fingerprint"],
                "num_examples": g0["num_examples"], "num_covered": g0["num_covered"],
                "note": ("full_vocab_base_nll_all is PRIMARY threshold. "
                         "Best checkpoint saved only if full_vocab_refined_nll_all < this."),
            }, f, indent=2)
    else:
        print("[train] Computing full-vocab baseline...")
        g0 = full_vocab_eval_resolver(
            model, args.val_cand_dir, args.val_feat_dir,
            tok_dev, device,
            gate_filter_name = gate_f,
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
          f"full_vocab_refined_nll_all < {full_vocab_base_nll:.6f}\n")

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
                                    collate_fn=collate_bridge, num_workers=0, drop_last=False):
                yield batch
    train_inf = _infinite()

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler("cuda") if args.amp else None
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

    # ── Log files ─────────────────────────────────────────────────────────────
    train_log_path  = os.path.join(args.output_dir, "train_log.csv")
    eval_log_path   = os.path.join(args.output_dir, "eval_log.csv")
    bucket_log_path = os.path.join(args.output_dir, "bucket_eval.csv")

    train_fields = ["step", "ce", "rank_loss", "kl_loss", "delta_reg",
                    "delta_abs_mean", "delta_abs_max", "gold_in_topK_rate", "alpha", "lr", "n_train"]
    eval_fields  = [
        "step",
        "full_vocab_base_nll_all", "full_vocab_refined_nll_all", "full_vocab_gain_all",
        "full_vocab_inside_gate_base_nll_all", "full_vocab_inside_gate_ref_nll_all",
        "full_vocab_inside_gate_gain_all",
        "full_vocab_outside_gate_base_nll_all", "full_vocab_outside_gate_ref_nll_all",
        "full_vocab_base_acc1_all", "full_vocab_ref_acc1_all",
        "full_vocab_base_nll_covered", "full_vocab_refined_nll_covered", "full_vocab_gain_covered",
        "masked_cand_base_nll", "masked_cand_ref_nll", "masked_cand_gain",
        "gate_rate", "coverage", "inside_gate_n_all",
        "alpha",
        "gold_in_topK_rate_all", "gold_in_topK_rate_gate",
        "gold_rank_base_mean", "gold_rank_ref_mean",
        "gold_rank_improved_rate", "gold_rank_worsened_rate",
        "top1_acc_base", "top1_acc_ref", "top5_acc_base", "top5_acc_ref",
        "top1_changed_rate", "changed_to_gold_rate", "changed_away_from_gold_rate",
        "mean_gold_delta_if_in_topK", "mean_delta_abs",
        "gold_force_included_rate", "dataset_fingerprint",
    ]
    bucket_fields = ["step", "subset", "n", "full_vocab_base_nll", "full_vocab_ref_nll",
                     "full_vocab_gain", "base_acc1", "ref_acc1"]

    train_logf  = open(train_log_path,  "w", newline="")
    eval_logf   = open(eval_log_path,   "w", newline="")
    bucket_logf = open(bucket_log_path, "w", newline="")
    train_csv   = csv.DictWriter(train_logf,  fieldnames=train_fields,  extrasaction="ignore")
    eval_csv    = csv.DictWriter(eval_logf,   fieldnames=eval_fields,   extrasaction="ignore")
    bucket_csv  = csv.DictWriter(bucket_logf, fieldnames=bucket_fields, extrasaction="ignore")
    train_csv.writeheader()
    eval_csv.writeheader()
    bucket_csv.writeheader()

    best_path = os.path.join(args.output_dir, "best_resolver.pt")
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
                    loss, info = compute_resolver_loss(
                        model, batch, device, tok_dev, train_mask,
                        args.lambda_rank, args.lambda_kl, args.lambda_delta,
                        args.rank_margin, args.kl_topk, amp_enabled=True)
            else:
                loss, info = compute_resolver_loss(
                    model, batch, device, tok_dev, train_mask,
                    args.lambda_rank, args.lambda_kl, args.lambda_delta,
                    args.rank_margin, args.kl_topk)

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

        # Aggregate micro-batch info
        agg = {k: float(np.mean([i[k] for i in accum_infos if k in i]))
               for k in ("ce", "rank_loss", "kl_loss", "delta_reg",
                         "delta_abs_mean", "delta_abs_max", "gold_in_topK_rate", "n_train")}
        agg["alpha"] = float(model.alpha.item())
        agg["lr"]    = float(sched.get_last_lr()[0])
        agg["step"]  = step

        ema_ce = agg["ce"] if ema_ce is None else 0.95 * ema_ce + 0.05 * agg["ce"]
        train_csv.writerow(agg)
        train_logf.flush()

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  ce={agg['ce']:.4f}  "
                  f"rank={agg['rank_loss']:.4f}  kl={agg['kl_loss']:.4f}  "
                  f"delta={agg['delta_abs_mean']:.4f}  "
                  f"gold_in_topK={agg['gold_in_topK_rate']:.3f}  "
                  f"alpha={agg['alpha']:.4f}  "
                  f"n_train={int(agg['n_train'])}  "
                  f"t={elapsed:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n[train] eval @ step {step} ...")
            ev = full_vocab_eval_resolver(
                model, args.val_cand_dir, args.val_feat_dir,
                tok_dev, device,
                gate_filter_name = gate_f,
                filter_kwargs    = filter_kwargs,
                fail_on_mismatch = False,
                eval_batch_size  = args.eval_batch_size,
                variant_tag      = variant_tag,
            )
            ev["step"] = step
            eval_csv.writerow(ev)
            eval_logf.flush()

            print(f"  full_vocab_gain_all        = {ev['full_vocab_gain_all']:+.6f}")
            print(f"  inside_gate_gain_all       = {ev['full_vocab_inside_gate_gain_all']:+.6f}")
            print(f"  gold_rank_improved_rate    = {ev['gold_rank_improved_rate']:.4f}")
            print(f"  gold_rank_worsened_rate    = {ev['gold_rank_worsened_rate']:.4f}")
            print(f"  top1_acc_ref - top1_acc_b  = "
                  f"{ev['top1_acc_ref'] - ev['top1_acc_base']:+.4f}")
            print(f"  changed_to_gold_rate       = {ev['changed_to_gold_rate']:.4f}")
            print(f"  changed_away_rate          = {ev['changed_away_from_gold_rate']:.4f}")
            print(f"  gold_in_topK_rate_all      = {ev['gold_in_topK_rate_all']:.4f}")
            print(f"  alpha                      = {ev['alpha']:.4f}")

            # Subset eval
            sub_rows = local_subset_eval_resolver(
                model, args.val_cand_dir, args.val_feat_dir,
                tok_dev, device,
                filter_kwargs    = filter_kwargs,
                eval_batch_size  = args.eval_batch_size,
            )
            for r in sub_rows:
                r["step"] = step
                bucket_csv.writerow(r)
            bucket_logf.flush()

            # Best checkpoint
            cur_nll = ev["full_vocab_refined_nll_all"]
            if cur_nll < best_nll:
                best_nll  = cur_nll
                best_step = step
                torch.save({
                    "model": model.state_dict(),
                    "args":  {k: v for k, v in vars(args).items()
                              if isinstance(v, (int, float, str, bool, type(None)))},
                    "step":  step,
                    "full_vocab_refined_nll_all": cur_nll,
                    "full_vocab_base_nll_all":    full_vocab_base_nll,
                    "full_vocab_gain_all":        ev["full_vocab_gain_all"],
                }, best_path)
                print(f"  *** NEW BEST  step={step}  "
                      f"nll={cur_nll:.6f}  gain={ev['full_vocab_gain_all']:+.6f} ***")

                best_metrics = {**ev, "step": step, "no_improving_checkpoint": False,
                                "best_step": step,
                                "best_full_vocab_gain_all": ev["full_vocab_gain_all"],
                                "best_inside_gate_gain":    ev["full_vocab_inside_gate_gain_all"]}
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
            else:
                print(f"  (no improvement; best={best_nll:.6f} @ step {best_step})")

            print()

    # ── Final metrics ─────────────────────────────────────────────────────────
    for fh in (train_logf, eval_logf, bucket_logf):
        fh.close()

    no_ckpt = best_step < 0
    final_metrics = {
        "no_improving_checkpoint": no_ckpt,
        "best_step":               best_step,
        "best_full_vocab_gain_all": full_vocab_base_nll - best_nll if not no_ckpt else 0.0,
        "best_inside_gate_gain":   float("nan"),
    }
    if not no_ckpt:
        bm = json.load(open(os.path.join(args.output_dir, "best_metrics.json")))
        final_metrics["best_inside_gate_gain"] = bm.get("full_vocab_inside_gate_gain_all", float("nan"))

    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    if no_ckpt:
        print(f"\n[train] done  NO IMPROVING CHECKPOINT  "
              f"baseline={full_vocab_base_nll:.6f}")
    else:
        print(f"\n[train] done  best_nll={best_nll:.6f}  "
              f"gain={full_vocab_base_nll-best_nll:+.6f}  step={best_step}")

    _generate_report(args.output_dir, args, full_vocab_base_nll, final_metrics)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device(args.device)

    print(f"[main] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().cpu()
    elif hasattr(backbone, "transformer"):
        tok_emb_w = backbone.transformer.wte.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot locate token embedding in backbone")
    print(f"  tok_emb_w shape = {tuple(tok_emb_w.shape)}")

    n_fine  = 128
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

    t2r_np: Optional[np.ndarray] = None
    if args.region_map and os.path.isfile(args.region_map):
        t2r_np = load_token_to_region(args.region_map, vocab_size)
        mapped = int((t2r_np >= 0).sum())
        print(f"  token_to_region: {mapped}/{vocab_size} tokens mapped")
    else:
        print("  WARNING: --region_map not provided; token region features will be uninformative")

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

    if args.gate_filter is None:
        args.gate_filter = args.train_filter

    train_resolver(args, d_model, n_ctx_layers, n_fine, n_super,
                   r2s_np, t2r_np, tok_emb_w, device)


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
    p.add_argument("--region_map",     default=None)
    p.add_argument("--output_dir",     required=True)
    # Confuser
    p.add_argument("--confuser_source", default="base_topk",
                   choices=["base_topk"],
                   help="How to build confuser set. base_topk: top-K from base logits only (no gold force).")
    p.add_argument("--top_k",          type=int,   default=256)
    # Architecture
    p.add_argument("--resolver_dim",   type=int,   default=256)
    p.add_argument("--resolver_layers",type=int,   default=2)
    p.add_argument("--resolver_heads", type=int,   default=4)
    p.add_argument("--ff_mult",        type=int,   default=4)
    p.add_argument("--dropout",        type=float, default=0.0)
    # Filter
    p.add_argument("--train_filter",   default="boundary")
    p.add_argument("--gate_filter",    default=None)
    p.add_argument("--margin_thresh",  type=float, default=0.1)
    p.add_argument("--entropy_thresh", type=float, default=2.0)
    # Training
    p.add_argument("--steps",            type=int,   default=5000)
    p.add_argument("--eval_every",       type=int,   default=1000)
    p.add_argument("--batch_size",       type=int,   default=16)
    p.add_argument("--eval_batch_size",  type=int,   default=64)
    p.add_argument("--grad_accum_steps", type=int,   default=4)
    p.add_argument("--lr",               type=float, default=5e-5)
    p.add_argument("--lambda_rank",      type=float, default=0.5)
    p.add_argument("--lambda_kl",        type=float, default=0.1)
    p.add_argument("--lambda_delta",     type=float, default=1e-4)
    p.add_argument("--rank_margin",      type=float, default=0.1)
    p.add_argument("--kl_topk",          type=int,   default=256,
                   help="KL computed over this many top-K logits (defaults to top_k)")
    p.add_argument("--grad_clip",        type=float, default=1.0)
    # Flags
    p.add_argument("--amp",                      action="store_true")
    p.add_argument("--eval_before_train",        action="store_true")
    p.add_argument("--fail_on_baseline_mismatch",action="store_true")
    p.add_argument("--use_filtered_train_loader",action="store_true")
    p.add_argument("--train_covered_only",       action="store_true")
    p.add_argument("--device",                   default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
