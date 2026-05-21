#!/usr/bin/env python3
"""
Stage 06 — Train Live Context + Retrieval Token Resolver.

Architecture: LiveContextRetrievalTokenResolver
  - Frozen backbone runs LIVE on input_ids → h_ctx = h_all[:, -1, :]
  - base_lgt = h_ctx @ tok_emb.T                      (full vocab, live)
  - candidate_mode base_topk_plus_neighbors:
      candidates = base top-K ∪ neighbor_gold_tokens  (K + num_neighbors)
  - Resolver sequence: [CTX | ROUTER | RET | TOKEN_1 .. TOKEN_{K+N}]
      CTX:    ctx_proj(h_ctx)
      ROUTER: prob-weighted fine_emb + scalar_proj(entropy, margin, top_prob)
      RET:    score-weighted tok_emb_proj(neighbor_gold_tokens) + ret_scalar(...)
      TOKEN_i: tok_emb_proj + score_proj + fine_emb + super_emb + support_proj
  - bounded_delta = delta_scale * tanh(raw_delta)     (delta_head zero-init)
  - refined_lgt = base_lgt + scatter(alpha * bounded_delta, candidate_ids)
  - gate = boundary filter
  - gated_lgt = where(gate, refined_lgt, base_lgt)

Identity at step-0: delta_head zero-init → tanh(0)=0 → delta=0 → identity ✓
gold_force_included_rate = 0 (structural: only base top-K + neighbor gold tokens)

Success thresholds:
  Weak:       full_vocab_gain_all > +0.0033
  Meaningful: full_vocab_gain_all > +0.005
  Strong:     full_vocab_gain_all > +0.010  inside_gate > +0.05
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

from scripts.offline_region_knn import load_small_backbone_and_probe, get_hs_small
from scripts.train_clean_path_refiner import load_r2s
from scripts.train_hard_position_refiner import compute_filter_mask, EVAL_SUBSETS
from scripts.train_bridge_residual_adapter import (
    CANONICAL_FINGERPRINT, CANONICAL_NUM_EXAMPLES, CANONICAL_NUM_COVERED,
    CANONICAL_COVERAGE, MASKED_CAND_BASELINE_NLL,
)
from scripts.train_token_confuser_resolver import load_token_to_region

KNOWN_BASE_NLL_ALL = 3.754938   # h_prime-based; live context NLL may differ


# ── Model ─────────────────────────────────────────────────────────────────────

class LiveContextRetrievalTokenResolver(nn.Module):
    """
    Token resolver with live backbone context and retrieval support.

    Sequence: [CTX | ROUTER | RET | CAND_1 .. CAND_{K+N}]
    bounded delta = delta_scale * tanh(delta_head(out))  → zero at init
    """

    def __init__(
        self,
        backbone,
        d_backbone:      int,
        n_fine:          int,
        n_super:         int,
        r2s_np:          np.ndarray,
        t2r_np:          Optional[np.ndarray],
        top_k:           int   = 256,
        num_neighbors:   int   = 32,
        d_resolver:      int   = 256,
        n_layers:        int   = 2,
        n_heads:         int   = 4,
        ff_mult:         int   = 4,
        dropout:         float = 0.0,
        delta_scale:     float = 0.25,
        retrieval_tau:   float = 0.2,
        candidate_mode:  str   = "base_topk_plus_neighbors",
    ):
        super().__init__()
        self.backbone        = backbone
        self.top_k           = top_k
        self.num_neighbors   = num_neighbors
        self.n_fine          = n_fine
        self.n_super         = n_super
        self.d_resolver      = d_resolver
        self.delta_scale     = delta_scale
        self.retrieval_tau   = retrieval_tau
        self.candidate_mode  = candidate_mode
        self.n_cand          = (top_k + num_neighbors
                                if candidate_mode == "base_topk_plus_neighbors"
                                else top_k)

        # CTX token projection
        self.ctx_proj = nn.Linear(d_backbone, d_resolver)

        # Region embeddings (shared across ROUTER/MEM/TOKEN positions)
        self.fine_emb  = nn.Embedding(n_fine  + 1, d_resolver, padding_idx=n_fine)
        self.super_emb = nn.Embedding(n_super + 1, d_resolver, padding_idx=n_super)

        # ROUTER token scalars
        self.router_scalar = nn.Linear(3, d_resolver)

        # RETRIEVAL token
        self.tok_emb_proj = nn.Linear(d_backbone, d_resolver, bias=False)
        self.ret_scalar   = nn.Linear(3, d_resolver)   # mean_score, max_score, n_valid/N

        # Per-candidate feature projections
        self.score_proj   = nn.Linear(3, d_resolver)   # logit, rank/K, logprob
        self.support_proj = nn.Linear(4, d_resolver)   # r_prob, m_prob, in_r8, in_m8
        self.tok_norm     = nn.LayerNorm(d_resolver)

        # Auxiliary region head: supervised on gold_region (lambda_region)
        self.region_head = nn.Linear(d_resolver, n_fine)

        # Transformer (pre-LN, batch_first)
        enc = nn.TransformerEncoderLayer(
            d_model=d_resolver, nhead=n_heads,
            dim_feedforward=d_resolver * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)

        # Delta head — ZERO INIT → identity at step 0
        self.delta_head = nn.Linear(d_resolver, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.alpha = nn.Parameter(torch.tensor(1.0))

        # Region lookup buffers
        V_buf = t2r_np.shape[0] if t2r_np is not None else 50257
        t2r_i = (t2r_np.astype(np.int32) if t2r_np is not None
                 else np.full(V_buf, n_fine, dtype=np.int32))
        self.register_buffer("t2r", torch.from_numpy(t2r_i).long())
        self.register_buffer("r2s", torch.from_numpy(r2s_np.astype(np.int64)).long())

        # Backbone is frozen — must not receive gradients
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    # ── Special sequence tokens ───────────────────────────────────────────────

    def _router_tok(self, r_reg, r_prb, r_mar):
        r_s     = r_reg.clamp(0, self.n_fine - 1)
        w       = r_prb / (r_prb.sum(1, keepdim=True) + 1e-8)
        pooled  = (self.fine_emb(r_s) * w.unsqueeze(-1)).sum(1)
        entropy = -(r_prb * torch.log(r_prb + 1e-9)).sum(1)
        sc      = torch.stack([entropy, r_mar.float(), r_prb[:, 0]], dim=-1)
        return pooled + self.router_scalar(sc)

    def _ret_tok(self, nbr_ids, nbr_scores, emb_w):
        # nbr_ids: (B, N_ret) int64, -1=invalid
        # nbr_scores: (B, N_ret) float32
        valid  = (nbr_ids >= 0)                                    # (B, N_ret)
        safe   = nbr_ids.clamp(0, emb_w.shape[0] - 1)
        n_embs = F.embedding(safe, emb_w.float())                  # (B, N_ret, d)
        n_proj = self.tok_emb_proj(n_embs)                         # (B, N_ret, d_res)

        masked_scores = nbr_scores.float().masked_fill(~valid, -1e9)
        weights = F.softmax(masked_scores / self.retrieval_tau, dim=-1)
        weights = weights.masked_fill(~valid, 0.0)
        w_sum   = weights.sum(-1, keepdim=True).clamp(min=1e-8)
        weights = weights / w_sum

        ret_emb    = (n_proj * weights.unsqueeze(-1)).sum(1)       # (B, d_res)

        n_valid_f  = valid.float().sum(-1)                         # (B,)
        mean_sc    = (nbr_scores.float() * valid.float()).sum(-1) / n_valid_f.clamp(min=1)
        max_sc     = nbr_scores.float().masked_fill(~valid, -1e9).max(-1).values
        max_sc     = max_sc.masked_fill(n_valid_f == 0, 0.0)
        scalars    = torch.stack([mean_sc, max_sc,
                                  n_valid_f / max(self.num_neighbors, 1)], dim=-1)
        return ret_emb + self.ret_scalar(scalars)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,   # (B, ctx_len) int64
        r_reg:          torch.Tensor,   # (B, K_r) int64
        r_prb:          torch.Tensor,   # (B, K_r) float32
        m_reg:          torch.Tensor,   # (B, K_m) int64
        m_prb:          torch.Tensor,   # (B, K_m) float32
        r_mar:          torch.Tensor,   # (B,) float32
        m_mar:          torch.Tensor,   # (B,) float32
        nbr_ids:        torch.Tensor,   # (B, num_neighbors) int64
        nbr_scores:     torch.Tensor,   # (B, num_neighbors) float32
        tok_emb_w:      torch.Tensor,   # (V, d_backbone) frozen
    ) -> Dict[str, torch.Tensor]:

        device = input_ids.device
        B      = input_ids.shape[0]
        V      = tok_emb_w.shape[0]
        emb_w  = tok_emb_w.float().to(device)

        # ── Live context ──────────────────────────────────────────────────────
        with torch.no_grad():
            h_all = get_hs_small(self.backbone, input_ids, device)  # (B, T, d)
        h_ctx     = h_all[:, -1, :].detach().float()               # (B, d_backbone)

        base_lgt  = h_ctx @ emb_w.T                                 # (B, V)

        topk_lgt, topk_ids = base_lgt.topk(self.top_k, dim=1)      # (B, K)

        # ── Build candidate set ───────────────────────────────────────────────
        if self.candidate_mode == "base_topk_plus_neighbors":
            cand_ids = torch.cat([topk_ids, nbr_ids.long()], dim=1) # (B, K+N)
        else:
            cand_ids = topk_ids                                      # (B, K)

        cand_valid = (cand_ids >= 0)                                 # (B, n_cand)
        n_cand = cand_ids.shape[1]

        # ── CTX token ─────────────────────────────────────────────────────────
        ctx_tok = self.ctx_proj(h_ctx)                               # (B, d_res)

        # ── ROUTER token ──────────────────────────────────────────────────────
        r_tok = self._router_tok(r_reg, r_prb, r_mar)               # (B, d_res)

        # ── RETRIEVAL token ───────────────────────────────────────────────────
        ret_tok = self._ret_tok(nbr_ids, nbr_scores, emb_w)         # (B, d_res)

        # ── Per-candidate tokens ──────────────────────────────────────────────
        safe_cand = cand_ids.clamp(0, V - 1)
        t_embs    = F.embedding(safe_cand, emb_w)                    # (B, n_cand, d)
        tok_repr  = self.tok_emb_proj(t_embs)                       # (B, n_cand, d_res)

        # Score features: [logit, rank_norm, logprob]
        cand_lgt  = base_lgt.gather(1, safe_cand) * cand_valid.float()  # (B, n_cand)
        cand_lp   = F.log_softmax(base_lgt, dim=1).gather(1, safe_cand) * cand_valid.float()
        rank_norm = torch.arange(n_cand, device=device, dtype=torch.float32
                                 ).unsqueeze(0).expand(B, -1) / n_cand
        scores    = torch.stack([cand_lgt, rank_norm, cand_lp], dim=-1)
        tok_repr  = tok_repr + self.score_proj(scores)

        # Region features per candidate
        fine_ids  = self.t2r[safe_cand.clamp(0, self.t2r.shape[0] - 1)]
        fine_ids  = fine_ids.masked_fill(fine_ids < 0, self.n_fine)
        super_ids = self.r2s[fine_ids.clamp(0, self.n_fine - 1)].clamp(0, self.n_super - 1)
        super_ids = super_ids.masked_fill(fine_ids == self.n_fine, self.n_super)
        tok_repr  = tok_repr + self.fine_emb(fine_ids)
        tok_repr  = tok_repr + self.super_emb(super_ids)

        # Router/mem support per candidate
        K_r = r_reg.shape[1]; K_m = m_reg.shape[1]
        r_s = r_reg.clamp(0, self.n_fine - 1)
        m_s = m_reg.clamp(0, self.n_fine - 1)
        match_r      = (r_s.unsqueeze(2) == fine_ids.unsqueeze(1))         # (B, K_r, n_cand)
        router_supp  = (r_prb.unsqueeze(2) * match_r.float()).sum(1)
        in_router8   = match_r[:, :min(8, K_r), :].any(1).float()
        match_m      = (m_s.unsqueeze(2) == fine_ids.unsqueeze(1))
        mem_supp     = (m_prb.unsqueeze(2) * match_m.float()).sum(1)
        in_mem8      = match_m[:, :min(8, K_m), :].any(1).float()
        support      = torch.stack([router_supp, mem_supp, in_router8, in_mem8], dim=-1)
        tok_repr     = tok_repr + self.support_proj(support)
        tok_repr     = self.tok_norm(tok_repr) * cand_valid.float().unsqueeze(-1)

        # ── Transformer ───────────────────────────────────────────────────────
        n_prefix = 3   # CTX, ROUTER, RET
        seq = torch.cat([
            ctx_tok.unsqueeze(1), r_tok.unsqueeze(1), ret_tok.unsqueeze(1),
            tok_repr,
        ], dim=1)                                                    # (B, 3+n_cand, d_res)

        # Padding mask: True = ignore (padded candidates)
        pad_mask = torch.zeros(B, n_prefix + n_cand, dtype=torch.bool, device=device)
        pad_mask[:, n_prefix:] = ~cand_valid

        out = self.transformer(seq, src_key_padding_mask=pad_mask)  # (B, 3+n_cand, d_res)

        # ── Bounded delta ─────────────────────────────────────────────────────
        raw_delta    = self.delta_head(out[:, n_prefix:, :]).squeeze(-1)  # (B, n_cand)
        bounded_delta = self.delta_scale * torch.tanh(raw_delta)
        bounded_delta = bounded_delta * cand_valid.float()

        delta_src  = self.alpha * bounded_delta
        safe_cand2 = cand_ids.clamp(0, V - 1)
        delta_full = torch.zeros(B, V, device=device)
        delta_full.scatter_add_(1, safe_cand2, delta_src)
        refined_lgt = base_lgt + delta_full

        # ── Auxiliary: region head on CTX output ─────────────────────────────
        ctx_out_repr   = out[:, 0, :]                                # (B, d_res)
        region_logits  = self.region_head(ctx_out_repr)              # (B, n_fine)

        return {
            "refined_lgt":  refined_lgt,   # (B, V)
            "base_lgt":     base_lgt,       # (B, V)
            "cand_ids":     cand_ids,       # (B, n_cand)
            "cand_valid":   cand_valid,     # (B, n_cand)
            "region_logits":region_logits,  # (B, n_fine)
            "bounded_delta":bounded_delta,  # (B, n_cand)
            "h_ctx":        h_ctx,          # (B, d_backbone)
        }


# ── Dataset ───────────────────────────────────────────────────────────────────

class LiveContextRetrievalDataset(IterableDataset):
    """
    Streams rows from aligned (cand, ctx, retrieval) shard triples.
    Yields dicts with all fields needed by compute_lctx_loss.
    """

    def __init__(
        self,
        cand_dir: str,
        ctx_dir: str,
        retrieval_dir: str,
        filter_name: str,
        filter_kwargs: Optional[Dict] = None,
        shuffle: bool = False,
    ):
        self.cand_dir     = cand_dir
        self.ctx_dir      = ctx_dir
        self.retrieval_dir = retrieval_dir
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
            import random; random.shuffle(paths)

        for cand_path in paths:
            si       = int(os.path.basename(cand_path).replace("shard_", "").replace(".pt", ""))
            ctx_path = os.path.join(self.ctx_dir,       f"shard_{si:05d}.pt")
            ret_path = os.path.join(self.retrieval_dir, f"shard_{si:05d}.pt")
            for p in (ctx_path, ret_path):
                if not os.path.exists(p):
                    raise RuntimeError(f"Missing shard: {p}")

            cs = torch.load(cand_path, map_location="cpu", weights_only=True)
            cx = torch.load(ctx_path,  map_location="cpu", weights_only=True)
            rs = torch.load(ret_path,  map_location="cpu", weights_only=True)

            if not torch.equal(cs["gold_token"].int(), cx["gold_token"].int()):
                raise RuntimeError(f"cand/ctx gold_token mismatch shard {si:05d}")
            if not torch.equal(cs["gold_token"].int(), rs["gold_token"].int()):
                raise RuntimeError(f"cand/ret gold_token mismatch shard {si:05d}")

            N     = cs["gold_token"].shape[0]
            fmask = compute_filter_mask(cs, self.filter_name, **self.filter_kwargs)

            has_mem = "mem_topk_reg" in cs
            m_reg_t = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
            m_prb_t = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
            m_mar_t = cs["mem_margin"]   if has_mem else torch.zeros(N)

            for i in range(N):
                yield {
                    "input_ids":   cx["input_ids"][i].long(),
                    "r_topk_reg":  cs["router_topk_reg"][i].long(),
                    "r_topk_prb":  cs["router_topk_prb"][i].float(),
                    "m_topk_reg":  m_reg_t[i].long(),
                    "m_topk_prb":  m_prb_t[i].float(),
                    "r_margin":    cs["router_margin"][i].float(),
                    "m_margin":    m_mar_t[i].float(),
                    "nbr_ids":     rs["neighbor_gold_tokens"][i].long(),
                    "nbr_scores":  rs["neighbor_scores"][i].float(),
                    "gold_token":  cs["gold_token"][i].long(),
                    "gold_region": cs["gold_region"][i].long(),
                    "covered":     cs["covered"][i].bool(),
                    "filter_mask": fmask[i].bool(),
                }


class FilteredLiveContextRetrievalDataset(IterableDataset):
    """Like LiveContextRetrievalDataset but yields only filter_mask==True rows."""

    def __init__(self, cand_dir, ctx_dir, retrieval_dir, filter_name,
                 filter_kwargs=None, shuffle=False):
        self._ds = LiveContextRetrievalDataset(
            cand_dir, ctx_dir, retrieval_dir, filter_name, filter_kwargs, shuffle)

    def __iter__(self):
        for row in self._ds:
            if row["filter_mask"]:
                yield row


def collate_lctx(batch: List[Dict]) -> Dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── Eval ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_vocab_eval_lctx_resolver(
    model:           LiveContextRetrievalTokenResolver,
    val_cand_dir:    str,
    val_ctx_dir:     str,
    val_ret_dir:     str,
    tok_emb_w:       torch.Tensor,
    device,
    gate_filter_name:str,
    filter_kwargs:   Dict,
    fail_on_mismatch:bool = False,
    eval_batch_size: int  = 64,
    variant_tag:     str  = "lctx_resolver",
) -> Dict:
    model.eval()
    emb_w  = tok_emb_w.float().to(device)
    V      = emb_w.shape[0]
    K      = model.top_k

    # Accumulators
    base_ce_all = ref_ce_all = 0.0
    ig_base_all = ig_ref_all = 0.0
    og_base_all = og_ref_all = 0.0
    nc_all = ig_nc_all = og_nc_all = 0
    base_ce_cov = ref_ce_cov = 0.0
    ig_base_cov = ig_ref_cov = 0.0
    nc_cov = ig_nc_cov = 0
    mc_base = mc_ref = 0.0
    mc_nc = total_n = total_cov = gate_n = 0
    base_acc1_all = ref_acc1_all = base_acc5_all = ref_acc5_all = 0
    gold_in_topK_all = gold_in_topK_gate = 0
    gold_rank_base = gold_rank_ref = 0.0
    rank_improved = rank_worsened = rank_unchanged = 0
    top1_changed_n = changed_to_gold_n = changed_away_n = 0
    delta_abs_sum = delta_n = 0.0
    sum_cand = sum_gi_cov = sum_gt = 0

    cand_paths = sorted(glob.glob(os.path.join(val_cand_dir, "shard_*.pt")))
    if not cand_paths:
        raise RuntimeError(f"No shard_*.pt in {val_cand_dir}")

    for cand_path in cand_paths:
        si       = int(os.path.basename(cand_path).replace("shard_", "").replace(".pt", ""))
        ctx_path = os.path.join(val_ctx_dir, f"shard_{si:05d}.pt")
        ret_path = os.path.join(val_ret_dir, f"shard_{si:05d}.pt")
        for p in (ctx_path, ret_path):
            if not os.path.exists(p):
                raise RuntimeError(f"Missing eval shard: {p}")

        cs = torch.load(cand_path, map_location="cpu", weights_only=True)
        cx = torch.load(ctx_path,  map_location="cpu", weights_only=True)
        rs = torch.load(ret_path,  map_location="cpu", weights_only=True)

        N    = cs["gold_token"].shape[0]
        gate = compute_filter_mask(cs, gate_filter_name, **filter_kwargs)

        has_mem = "mem_topk_reg" in cs
        m_reg_t = cs["mem_topk_reg"] if has_mem else torch.zeros_like(cs["router_topk_reg"])
        m_prb_t = cs["mem_topk_prb"] if has_mem else torch.zeros_like(cs["router_topk_prb"])
        m_mar_t = cs["mem_margin"]   if has_mem else torch.zeros(N)

        for start in range(0, N, eval_batch_size):
            end = min(start + eval_batch_size, N)
            sl  = slice(start, end)
            B   = end - start

            ids   = cx["input_ids"][sl].long().to(device)
            cov   = cs["covered"][sl].bool().to(device)
            g_sl  = gate[sl].to(device)
            gidx  = cs["gold_cand_idx"][sl].long().to(device)
            gt_b  = cs["gold_token"][sl].long().to(device)
            ct    = cs["cand_tok"][sl].long().to(device)
            r_reg = cs["router_topk_reg"][sl].long().to(device)
            r_prb = cs["router_topk_prb"][sl].float().to(device)
            m_reg = m_reg_t[sl].long().to(device)
            m_prb = m_prb_t[sl].float().to(device)
            r_mar = cs["router_margin"][sl].float().to(device)
            m_mar = m_mar_t[sl].float().to(device)
            nbr_i = rs["neighbor_gold_tokens"][sl].long().to(device)
            nbr_s = rs["neighbor_scores"][sl].float().to(device)
            cmask = (ct >= 0)

            out   = model(ids, r_reg, r_prb, m_reg, m_prb, r_mar, m_mar,
                          nbr_i, nbr_s, emb_w)
            ref_lgt  = out["refined_lgt"]
            base_lgt = out["base_lgt"]

            gate_exp  = g_sl.unsqueeze(1).expand(-1, V)
            gated_lgt = torch.where(gate_exp, ref_lgt, base_lgt)

            # CE
            base_ce_all += float(F.cross_entropy(base_lgt,  gt_b, reduction="sum"))
            ref_ce_all  += float(F.cross_entropy(gated_lgt, gt_b, reduction="sum"))
            nc_all += B

            if g_sl.any():
                ig = g_sl
                ig_base_all += float(F.cross_entropy(base_lgt[ig], gt_b[ig], reduction="sum"))
                ig_ref_all  += float(F.cross_entropy(ref_lgt[ig],  gt_b[ig], reduction="sum"))
                ig_nc_all   += int(ig.sum())

            og = ~g_sl
            if og.any():
                og_base_all += float(F.cross_entropy(base_lgt[og], gt_b[og], reduction="sum"))
                og_ref_all  += float(F.cross_entropy(gated_lgt[og],gt_b[og], reduction="sum"))
                og_nc_all   += int(og.sum())

            n_cov = int(cov.sum())
            if n_cov > 0:
                cov_gt      = gt_b[cov]
                base_ce_cov += float(F.cross_entropy(base_lgt[cov],  cov_gt, reduction="sum"))
                ref_ce_cov  += float(F.cross_entropy(gated_lgt[cov], cov_gt, reduction="sum"))
                nc_cov += n_cov

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

            # Accuracy
            base_top1 = base_lgt.argmax(1)
            ref_top1  = gated_lgt.argmax(1)
            base_acc1_all += int((base_top1 == gt_b).sum())
            ref_acc1_all  += int((ref_top1  == gt_b).sum())
            base_acc5_all += int((base_lgt.topk(5, dim=1)[1] == gt_b.unsqueeze(1)).any(1).sum())
            ref_acc5_all  += int((gated_lgt.topk(5, dim=1)[1] == gt_b.unsqueeze(1)).any(1).sum())

            changed = (base_top1 != ref_top1)
            top1_changed_n    += int(changed.sum())
            changed_to_gold_n += int((changed & (ref_top1 == gt_b)).sum())
            changed_away_n    += int((changed & (base_top1 == gt_b)).sum())

            # Rank
            gt_col        = gt_b.unsqueeze(1)
            in_topK       = (base_lgt.topk(K, dim=1).indices == gt_col).any(1)
            gold_in_topK_all  += int(in_topK.sum())
            gold_in_topK_gate += int((in_topK & g_sl).sum())

            bi             = torch.arange(B, device=device)
            gold_base_lgt  = base_lgt[bi,    gt_b].unsqueeze(1)
            gold_ref_lgt   = gated_lgt[bi,   gt_b].unsqueeze(1)
            rank_b = (1 + (base_lgt  > gold_base_lgt).sum(1)).float()
            rank_r = (1 + (gated_lgt > gold_ref_lgt ).sum(1)).float()
            gold_rank_base += float(rank_b.sum())
            gold_rank_ref  += float(rank_r.sum())
            rank_improved  += int((rank_r < rank_b).sum())
            rank_worsened  += int((rank_r > rank_b).sum())
            rank_unchanged += int((rank_r == rank_b).sum())

            delta_abs_sum += float(out["bounded_delta"].abs().sum())
            delta_n       += B * out["bounded_delta"].shape[1]

            gate_n    += int(g_sl.sum())
            total_n   += B
            total_cov += n_cov

    fp_data = {"num_shards": len(cand_paths), "total_n": total_n, "total_cov": total_cov,
               "sum_cand_counts": sum_cand, "sum_gold_idx_cov": sum_gi_cov, "sum_gold_tok": sum_gt}
    fp = hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()[:16]

    def _nll(ce, n):
        return ce / max(n, 1)

    results = {
        "eval_mode":  "lctx_retrieval_resolver",
        "full_vocab_base_nll_all":         _nll(base_ce_all, nc_all),
        "full_vocab_refined_nll_all":      _nll(ref_ce_all,  nc_all),
        "full_vocab_gain_all":             _nll(base_ce_all, nc_all) - _nll(ref_ce_all, nc_all),
        "full_vocab_inside_gate_base_nll_all":  _nll(ig_base_all, ig_nc_all),
        "full_vocab_inside_gate_ref_nll_all":   _nll(ig_ref_all,  ig_nc_all),
        "full_vocab_inside_gate_gain_all":      _nll(ig_base_all, ig_nc_all) - _nll(ig_ref_all, ig_nc_all),
        "full_vocab_outside_gate_base_nll_all": _nll(og_base_all, og_nc_all),
        "full_vocab_outside_gate_ref_nll_all":  _nll(og_ref_all,  og_nc_all),
        "full_vocab_base_nll_covered":    _nll(base_ce_cov, nc_cov),
        "full_vocab_refined_nll_covered": _nll(ref_ce_cov,  nc_cov),
        "full_vocab_gain_covered":        _nll(base_ce_cov, nc_cov) - _nll(ref_ce_cov, nc_cov),
        "full_vocab_inside_gate_base_nll_covered": _nll(ig_base_cov, ig_nc_cov),
        "full_vocab_inside_gate_ref_nll_covered":  _nll(ig_ref_cov,  ig_nc_cov),
        "full_vocab_inside_gate_gain_covered":     _nll(ig_base_cov, ig_nc_cov) - _nll(ig_ref_cov, ig_nc_cov),
        "masked_cand_base_nll":     _nll(mc_base, mc_nc),
        "masked_cand_ref_nll":      _nll(mc_ref,  mc_nc),
        "masked_cand_gain":         _nll(mc_base, mc_nc) - _nll(mc_ref, mc_nc),
        "masked_cand_baseline_ref": MASKED_CAND_BASELINE_NLL,
        "coverage":     total_cov / max(total_n, 1),
        "gate_rate":    gate_n    / max(total_n, 1),
        "num_examples": total_n,
        "num_covered":  total_cov,
        "inside_gate_n_all": ig_nc_all,
        "inside_gate_n_cov": ig_nc_cov,
        "dataset_fingerprint": fp,
        "alpha": float(model.alpha.item()),
        "delta_scale": model.delta_scale,
        "full_vocab_base_acc1_all":  base_acc1_all / max(nc_all, 1),
        "full_vocab_ref_acc1_all":   ref_acc1_all  / max(nc_all, 1),
        "full_vocab_base_acc5_all":  base_acc5_all / max(nc_all, 1),
        "full_vocab_ref_acc5_all":   ref_acc5_all  / max(nc_all, 1),
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
        "mean_delta_abs":          delta_abs_sum       / max(delta_n, 1),
        "gold_force_included_rate": 0.0,
    }

    fp_ok  = (fp        == CANONICAL_FINGERPRINT)
    n_ok   = (total_n   == CANONICAL_NUM_EXAMPLES)
    nc_ok  = (total_cov == CANONICAL_NUM_COVERED)
    cov_d  = abs(results["coverage"] - CANONICAL_COVERAGE)
    if not (fp_ok and n_ok and nc_ok and cov_d < 1e-6):
        issues = []
        if not fp_ok:  issues.append(f"fingerprint {fp!r} != {CANONICAL_FINGERPRINT!r}")
        if not n_ok:   issues.append(f"num_examples {total_n} != {CANONICAL_NUM_EXAMPLES}")
        if not nc_ok:  issues.append(f"num_covered {total_cov} != {CANONICAL_NUM_COVERED}")
        if cov_d >= 1e-6: issues.append(f"coverage diff {cov_d:.2e}")
        msg = f"full_vocab_eval_lctx [{variant_tag}] CANONICAL MISMATCH: " + "; ".join(issues)
        if fail_on_mismatch:
            raise RuntimeError(msg)
        print(f"  WARNING: {msg}")

    model.train()
    return results


# ── Loss ─────────────────────────────────────────────────────────────────────

def compute_lctx_loss(
    model:        LiveContextRetrievalTokenResolver,
    batch:        Dict,
    device,
    tok_emb_w:    torch.Tensor,
    train_mask:   Optional[torch.Tensor],
    lambda_region:float,
    lambda_rank:  float,
    lambda_kl:    float,
    lambda_delta: float,
    rank_margin:  float,
    kl_topk:      int,
) -> Tuple[Optional[torch.Tensor], Dict]:

    ids   = batch["input_ids"].long().to(device)
    gt_b  = batch["gold_token"].long().to(device)
    g_reg = batch["gold_region"].long().to(device)
    r_reg = batch["r_topk_reg"].long().to(device)
    r_prb = batch["r_topk_prb"].float().to(device)
    m_reg = batch["m_topk_reg"].long().to(device)
    m_prb = batch["m_topk_prb"].float().to(device)
    r_mar = batch["r_margin"].float().to(device)
    m_mar = batch["m_margin"].float().to(device)
    nbr_i = batch["nbr_ids"].long().to(device)
    nbr_s = batch["nbr_scores"].float().to(device)
    emb_w = tok_emb_w.float().to(device)

    if train_mask is not None:
        mask = train_mask.to(device)
        if not mask.any():
            return None, {}
        ids, gt_b, g_reg = ids[mask], gt_b[mask], g_reg[mask]
        r_reg, r_prb = r_reg[mask], r_prb[mask]
        m_reg, m_prb = m_reg[mask], m_prb[mask]
        r_mar, m_mar = r_mar[mask], m_mar[mask]
        nbr_i, nbr_s = nbr_i[mask], nbr_s[mask]

    B = ids.shape[0]
    if B == 0:
        return None, {}

    out         = model(ids, r_reg, r_prb, m_reg, m_prb, r_mar, m_mar,
                        nbr_i, nbr_s, emb_w)
    refined_lgt = out["refined_lgt"]   # (B, V)
    base_lgt    = out["base_lgt"]
    cand_ids    = out["cand_ids"]
    bounded_dl  = out["bounded_delta"]
    region_lgt  = out["region_logits"] # (B, n_fine)
    K           = model.top_k
    V           = emb_w.shape[0]

    # ── CE loss ───────────────────────────────────────────────────────────────
    ce_loss = F.cross_entropy(refined_lgt, gt_b)

    # ── Region auxiliary loss ─────────────────────────────────────────────────
    reg_loss = refined_lgt.new_zeros(1).squeeze()
    if lambda_region > 0:
        valid_reg = (g_reg >= 0) & (g_reg < model.n_fine)
        if valid_reg.any():
            reg_loss = F.cross_entropy(region_lgt[valid_reg], g_reg[valid_reg])

    # ── Rank loss ─────────────────────────────────────────────────────────────
    rank_loss = refined_lgt.new_zeros(1).squeeze()
    if lambda_rank > 0:
        topk_ids = base_lgt.detach().topk(K, dim=1).indices
        in_topK  = (topk_ids == gt_b.unsqueeze(1)).any(1)
        if in_topK.any():
            sub_gt   = gt_b[in_topK]
            sub_ref  = refined_lgt[in_topK]
            gold_lgt = sub_ref.gather(1, sub_gt.unsqueeze(1)).squeeze(1)
            wrong    = (topk_ids[in_topK] != sub_gt.unsqueeze(1)).float().argmax(1)
            neg_toks = topk_ids[in_topK].gather(1, wrong.unsqueeze(1)).squeeze(1)
            neg_lgt  = sub_ref.gather(1, neg_toks.unsqueeze(1)).squeeze(1)
            rank_loss = F.relu(rank_margin - gold_lgt + neg_lgt).mean()

    # ── KL loss over top-K distribution ──────────────────────────────────────
    kl_loss = refined_lgt.new_zeros(1).squeeze()
    if lambda_kl > 0:
        topk_ids   = base_lgt.detach().topk(K, dim=1).indices
        kk         = min(kl_topk, K)
        base_kk_lp = F.log_softmax(base_lgt.detach().gather(1, topk_ids[:, :kk]), dim=-1)
        ref_kk_lgt = refined_lgt.gather(1, topk_ids[:, :kk])
        ref_kk_lp  = F.log_softmax(ref_kk_lgt, dim=-1)
        kl_loss    = F.kl_div(ref_kk_lp, base_kk_lp.exp().detach(), reduction="batchmean")

    # ── Delta regularization ──────────────────────────────────────────────────
    delta_reg = bounded_dl.pow(2).mean() if lambda_delta > 0 else refined_lgt.new_zeros(1).squeeze()

    loss = ce_loss + lambda_region * reg_loss + lambda_rank * rank_loss + lambda_kl * kl_loss + lambda_delta * delta_reg

    in_topK_rate = float(
        (base_lgt.detach().topk(K, dim=1).indices == gt_b.unsqueeze(1)).any(1).float().mean()
    )

    return loss, {
        "ce":               float(ce_loss.item()),
        "reg_loss":         float(reg_loss.item()),
        "rank_loss":        float(rank_loss.item()),
        "kl_loss":          float(kl_loss.item()),
        "delta_reg":        float(delta_reg.item()),
        "delta_abs_mean":   float(bounded_dl.abs().mean().item()),
        "gold_in_topK_rate":in_topK_rate,
        "alpha":            float(model.alpha.item()),
        "n_train":          B,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_lctx_resolver(args, d_model, n_fine, n_super, r2s_np, t2r_np,
                        tok_emb_w, backbone, device):
    os.makedirs(args.output_dir, exist_ok=True)

    model = LiveContextRetrievalTokenResolver(
        backbone       = backbone,
        d_backbone     = d_model,
        n_fine         = n_fine,
        n_super        = n_super,
        r2s_np         = r2s_np,
        t2r_np         = t2r_np,
        top_k          = args.top_k,
        num_neighbors  = args.num_neighbors,
        d_resolver     = args.resolver_dim,
        n_layers       = args.resolver_layers,
        n_heads        = args.resolver_heads,
        delta_scale    = args.delta_scale,
        retrieval_tau  = args.retrieval_tau,
        candidate_mode = args.candidate_mode,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params = {total_p:,}")
    print(f"[train] delta_scale      = {args.delta_scale}")
    print(f"[train] candidate_mode   = {args.candidate_mode}")

    cfg = vars(args).copy()
    cfg.update({"n_fine": n_fine, "n_super": n_super, "d_model": d_model})
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    fail_hard     = args.fail_on_baseline_mismatch
    filter_kwargs = {"margin_thresh": 0.1, "entropy_thresh": 2.0}
    gate_f        = args.gate_filter or args.train_filter
    variant_tag   = f"lctxret-k{args.top_k}-n{args.num_neighbors}-d{args.resolver_dim}"
    tok_dev       = tok_emb_w.float().to(device)

    # ── Step-0 identity check ─────────────────────────────────────────────────
    full_vocab_base_nll: float = float("nan")

    def _run_eval():
        return full_vocab_eval_lctx_resolver(
            model, args.val_cand_dir, args.val_ctx_dir, args.val_retrieval_dir,
            tok_dev, device, gate_f, filter_kwargs,
            fail_on_mismatch=fail_hard,
            eval_batch_size=args.eval_batch_size,
            variant_tag=variant_tag,
        )

    if args.eval_before_train:
        print("\n[train] === step-0 identity check ===")
        g0  = _run_eval()
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]

        nll_diff_all = abs(g0["full_vocab_refined_nll_all"] - g0["full_vocab_base_nll_all"])
        nll_diff_cov = abs(g0["full_vocab_refined_nll_covered"] - g0["full_vocab_base_nll_covered"])
        og_diff      = abs(g0["full_vocab_outside_gate_ref_nll_all"] - g0["full_vocab_outside_gate_base_nll_all"])
        d_max        = g0.get("mean_delta_abs", 0.0)

        print(f"  full_vocab_base_nll_all     = {g0['full_vocab_base_nll_all']:.6f}")
        print(f"  full_vocab_refined_nll_all  = {g0['full_vocab_refined_nll_all']:.6f}")
        print(f"  diff_all (< 1e-3)           = {nll_diff_all:.2e}")
        print(f"  diff_covered (< 1e-3)       = {nll_diff_cov:.2e}")
        print(f"  outside_gate_diff (< 1e-5)  = {og_diff:.2e}")
        print(f"  delta_max_abs               = {d_max:.2e}")
        print(f"  gold_in_topK_rate_all       = {g0['gold_in_topK_rate_all']:.4f}")
        print(f"  alpha                       = {g0['alpha']:.4f}")
        print(f"  dataset_fingerprint         = {g0['dataset_fingerprint']}")

        if nll_diff_all >= 1e-3:
            raise RuntimeError(f"Step-0 IDENTITY FAIL (all): diff={nll_diff_all:.2e}")
        if nll_diff_cov >= 1e-3:
            raise RuntimeError(f"Step-0 IDENTITY FAIL (covered): diff={nll_diff_cov:.2e}")
        if og_diff >= 1e-5:
            raise RuntimeError(f"Step-0 outside-gate FAIL: diff={og_diff:.2e}")

        print(f"  [step-0] IDENTITY PASS  gold_force_included_rate=0.0000")

        identity_data = {
            "identity_pass": True,
            "nll_diff_all":  nll_diff_all,
            "nll_diff_covered": nll_diff_cov,
            "og_diff":       og_diff,
            "delta_max_abs": d_max,
            **{k: g0[k] for k in (
                "full_vocab_base_nll_all", "full_vocab_refined_nll_all",
                "full_vocab_base_nll_covered", "full_vocab_refined_nll_covered",
                "dataset_fingerprint", "alpha",
            )},
        }
        with open(os.path.join(args.output_dir, "debug_identity.json"), "w") as f:
            json.dump(identity_data, f, indent=2)
    else:
        g0 = _run_eval()
        full_vocab_base_nll = g0["full_vocab_base_nll_all"]
        print(f"  full_vocab_base_nll_all = {full_vocab_base_nll:.6f}")

    if full_vocab_base_nll != full_vocab_base_nll:
        raise RuntimeError("full_vocab_base_nll is NaN.")

    print(f"\n[train] best-ckpt threshold: refined_nll < {full_vocab_base_nll:.6f}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    use_filtered = args.use_filtered_train_loader
    grad_accum   = max(1, args.grad_accum_steps)

    if use_filtered:
        train_ds = FilteredLiveContextRetrievalDataset(
            args.train_cand_dir, args.train_ctx_dir, args.train_retrieval_dir,
            args.train_filter, filter_kwargs, shuffle=True,
        )
    else:
        train_ds = LiveContextRetrievalDataset(
            args.train_cand_dir, args.train_ctx_dir, args.train_retrieval_dir,
            args.train_filter, filter_kwargs, shuffle=True,
        )

    def _infinite():
        while True:
            for batch in DataLoader(train_ds, batch_size=args.batch_size,
                                    collate_fn=collate_lctx, num_workers=0, drop_last=False):
                yield batch
    train_inf = _infinite()

    opt    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-2
    )
    scaler = GradScaler("cuda") if args.amp else None
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1
    )

    # ── Log files ─────────────────────────────────────────────────────────────
    train_log = open(os.path.join(args.output_dir, "train_log.csv"),  "w", newline="")
    eval_log  = open(os.path.join(args.output_dir, "eval_log.csv"),   "w", newline="")
    bucket_log= open(os.path.join(args.output_dir, "bucket_eval.csv"),"w", newline="")

    train_fields = ["step", "ce", "reg_loss", "rank_loss", "kl_loss", "delta_reg",
                    "delta_abs_mean", "gold_in_topK_rate", "alpha", "lr", "n_train"]
    eval_fields  = [
        "step",
        "full_vocab_base_nll_all", "full_vocab_refined_nll_all", "full_vocab_gain_all",
        "full_vocab_inside_gate_base_nll_all", "full_vocab_inside_gate_ref_nll_all",
        "full_vocab_inside_gate_gain_all",
        "full_vocab_base_nll_covered", "full_vocab_refined_nll_covered", "full_vocab_gain_covered",
        "masked_cand_base_nll", "masked_cand_ref_nll", "masked_cand_gain",
        "gate_rate", "coverage", "alpha",
        "gold_in_topK_rate_all", "gold_in_topK_rate_gate",
        "gold_rank_base_mean", "gold_rank_ref_mean",
        "gold_rank_improved_rate", "gold_rank_worsened_rate",
        "top1_acc_base", "top1_acc_ref",
        "changed_to_gold_rate", "changed_away_from_gold_rate",
        "mean_delta_abs", "gold_force_included_rate", "dataset_fingerprint",
    ]
    bucket_fields = ["step", "subset", "n", "full_vocab_base_nll", "full_vocab_ref_nll",
                     "full_vocab_gain", "base_acc1", "ref_acc1"]

    train_csv  = csv.DictWriter(train_log,  fieldnames=train_fields,  extrasaction="ignore")
    eval_csv   = csv.DictWriter(eval_log,   fieldnames=eval_fields,   extrasaction="ignore")
    bucket_csv = csv.DictWriter(bucket_log, fieldnames=bucket_fields, extrasaction="ignore")
    for c in (train_csv, eval_csv, bucket_csv):
        c.writeheader()

    best_path = os.path.join(args.output_dir, "best_resolver.pt")
    best_nll  = full_vocab_base_nll
    best_step = -1
    t0        = time.time()
    model.train()
    opt.zero_grad()

    kl_topk = min(args.top_k, getattr(args, "kl_topk", args.top_k))

    for step in range(1, args.steps + 1):
        accum_infos: List[Dict] = []
        valid_micro = 0

        for _micro in range(grad_accum):
            batch = next(train_inf)
            train_mask = None if use_filtered else batch["filter_mask"]

            if args.amp:
                with autocast("cuda"):
                    loss, info = compute_lctx_loss(
                        model, batch, device, tok_dev, train_mask,
                        args.lambda_region, args.lambda_rank, args.lambda_kl,
                        args.lambda_delta, args.rank_margin, kl_topk,
                    )
            else:
                loss, info = compute_lctx_loss(
                    model, batch, device, tok_dev, train_mask,
                    args.lambda_region, args.lambda_rank, args.lambda_kl,
                    args.lambda_delta, args.rank_margin, kl_topk,
                )

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
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()

        sched.step()
        opt.zero_grad()

        agg = {k: float(np.mean([i[k] for i in accum_infos if k in i]))
               for k in ("ce", "reg_loss", "rank_loss", "kl_loss", "delta_reg",
                         "delta_abs_mean", "gold_in_topK_rate", "n_train")}
        agg.update({"alpha": float(model.alpha.item()),
                    "lr":    float(sched.get_last_lr()[0]),
                    "step":  step})

        train_csv.writerow(agg); train_log.flush()

        if step % 100 == 0:
            print(f"  step {step:5d}/{args.steps}  ce={agg['ce']:.4f}  "
                  f"kl={agg['kl_loss']:.4f}  delta={agg['delta_abs_mean']:.4f}  "
                  f"gold_in_topK={agg['gold_in_topK_rate']:.3f}  "
                  f"alpha={agg['alpha']:.4f}  n={int(agg['n_train'])}  "
                  f"t={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n[train] eval @ step {step} ...")
            ev = _run_eval()
            ev["step"] = step
            eval_csv.writerow(ev); eval_log.flush()

            print(f"  full_vocab_gain_all       = {ev['full_vocab_gain_all']:+.6f}")
            print(f"  inside_gate_gain_all      = {ev['full_vocab_inside_gate_gain_all']:+.6f}")
            print(f"  gold_rank_improved_rate   = {ev['gold_rank_improved_rate']:.4f}")
            print(f"  changed_to_gold_rate      = {ev['changed_to_gold_rate']:.4f}")
            print(f"  gold_in_topK_rate_all     = {ev['gold_in_topK_rate_all']:.4f}")
            print(f"  alpha                     = {ev['alpha']:.4f}")

            # Best checkpoint
            cur_nll = ev["full_vocab_refined_nll_all"]
            if cur_nll < best_nll:
                best_nll  = cur_nll
                best_step = step
                torch.save({
                    "model": {k: v for k, v in model.state_dict().items()
                              if not k.startswith("backbone.")},
                    "args":  {k: v for k, v in vars(args).items()
                              if isinstance(v, (int, float, str, bool, type(None)))},
                    "step":  step,
                    "full_vocab_refined_nll_all": cur_nll,
                    "full_vocab_gain_all": ev["full_vocab_gain_all"],
                }, best_path)
                best_metrics = {**ev, "step": step, "no_improving_checkpoint": False,
                                "best_step": step,
                                "best_full_vocab_gain_all": ev["full_vocab_gain_all"],
                                "best_inside_gate_gain": ev["full_vocab_inside_gate_gain_all"]}
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"  *** NEW BEST step={step} nll={cur_nll:.6f} "
                      f"gain={ev['full_vocab_gain_all']:+.6f} ***")
            else:
                print(f"  (no improvement; best={best_nll:.6f} @ step {best_step})")
            print()

    for fh in (train_log, eval_log, bucket_log):
        fh.close()

    no_ckpt = best_step < 0
    final_metrics = {
        "no_improving_checkpoint": no_ckpt,
        "best_step":               best_step,
        "best_full_vocab_gain_all": full_vocab_base_nll - best_nll if not no_ckpt else 0.0,
        "best_inside_gate_gain":   float("nan"),
    }
    if not no_ckpt and os.path.isfile(os.path.join(args.output_dir, "best_metrics.json")):
        bm = json.load(open(os.path.join(args.output_dir, "best_metrics.json")))
        final_metrics["best_inside_gate_gain"] = bm.get("full_vocab_inside_gate_gain_all", float("nan"))

    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    if no_ckpt:
        print(f"\n[train] done  NO IMPROVING CHECKPOINT  baseline={full_vocab_base_nll:.6f}")
    else:
        g = full_vocab_base_nll - best_nll
        print(f"\n[train] done  best_nll={best_nll:.6f}  gain={g:+.6f}  step={best_step}")

    # Write simple report
    rpath = os.path.join(args.output_dir, "report.md")
    with open(rpath, "w") as f:
        f.write("# LiveContextRetrieval Resolver — Report\n\n")
        f.write(f"delta_scale      = {args.delta_scale}\n")
        f.write(f"candidate_mode   = {args.candidate_mode}\n")
        f.write(f"top_k            = {args.top_k}\n")
        f.write(f"num_neighbors    = {args.num_neighbors}\n")
        f.write(f"train_filter     = {args.train_filter}\n")
        f.write(f"gate_filter      = {gate_f}\n\n")
        f.write(f"full_vocab_base_nll_all = {full_vocab_base_nll:.6f}\n")
        f.write(f"best_step               = {best_step}\n")
        g = final_metrics['best_full_vocab_gain_all']
        f.write(f"best_full_vocab_gain    = {g:+.6f}\n\n")
        thresh = ("+0.010 → STRONG" if g > 0.010 else
                  "+0.005 → MEANINGFUL" if g > 0.005 else
                  "+0.0033 → WEAK" if g > 0.0033 else
                  "no_gain")
        f.write(f"verdict = {thresh}\n")
    print(f"[train] report → {rpath}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)

    # ── Stage 01B gate ────────────────────────────────────────────────────────
    # Refuse to train unless live baseline is verified compatible.
    if args.stage01b_audit:
        if not os.path.isfile(args.stage01b_audit):
            raise RuntimeError(
                f"Stage 01B audit not found: {args.stage01b_audit}\n"
                "Run stage01b_audit_live_baseline_compat.py first."
            )
        with open(args.stage01b_audit) as _f:
            _s1b = json.load(_f)
        if not _s1b.get("live_baseline_compatible", False):
            raise RuntimeError(
                "Stage 01B live_baseline_compatible=False. "
                "Do not train Stage 06 until live forward is fixed. "
                f"See {args.stage01b_audit}"
            )
        # Override ctx_len from the best convention found by the audit
        _bc = _s1b.get("best_convention")
        if _bc and args.ctx_len_from_audit:
            audit_ctx = int(_bc["ctx_len"])
            if audit_ctx != args.ctx_len:
                print(f"[main] Stage 01B override: ctx_len {args.ctx_len} → {audit_ctx}")
                args.ctx_len = audit_ctx
        print(f"[main] Stage 01B gate: PASS  live_baseline_compatible=True  ctx_len={args.ctx_len}")

    print(f"[main] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot locate token_emb in backbone")

    n_fine = 128; n_super = 24
    cfg_path = os.path.join(args.val_cand_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        ds_cfg = json.load(open(cfg_path))
        n_fine = ds_cfg.get("n_fine", 128)
        n_super = ds_cfg.get("n_super", 24)

    r2s_np = np.zeros(n_fine, dtype=np.int32)
    if args.super_map and os.path.isfile(args.super_map):
        r2s_np  = load_r2s(args.super_map, n_fine)
        n_super = int(r2s_np.max()) + 1

    t2r_np = None
    if args.region_map and os.path.isfile(args.region_map):
        t2r_np = load_token_to_region(args.region_map, vocab_size)

    if args.gate_filter is None:
        args.gate_filter = args.train_filter

    train_lctx_resolver(args, d_model, n_fine, n_super, r2s_np, t2r_np,
                        tok_emb_w, backbone, device)


def _parse():
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--small_ckpt",          required=True)
    p.add_argument("--train_cand_dir",       required=True)
    p.add_argument("--val_cand_dir",         required=True)
    p.add_argument("--train_ctx_dir",        required=True)
    p.add_argument("--val_ctx_dir",          required=True)
    p.add_argument("--train_retrieval_dir",  required=True)
    p.add_argument("--val_retrieval_dir",    required=True)
    p.add_argument("--baseline_json",        default=None)
    p.add_argument("--super_map",            default=None)
    p.add_argument("--region_map",           default=None)
    p.add_argument("--output_dir",           required=True)
    # Architecture
    p.add_argument("--ctx_len",             type=int,   default=256)
    p.add_argument("--candidate_mode",      default="base_topk_plus_neighbors",
                   choices=["base_topk", "base_topk_plus_neighbors"])
    p.add_argument("--confuser_source",     default="base_topk")
    p.add_argument("--top_k",              type=int,   default=256)
    p.add_argument("--num_neighbors",      type=int,   default=32)
    p.add_argument("--resolver_dim",       type=int,   default=256)
    p.add_argument("--resolver_layers",    type=int,   default=2)
    p.add_argument("--resolver_heads",     type=int,   default=4)
    p.add_argument("--ff_mult",            type=int,   default=4)
    p.add_argument("--dropout",            type=float, default=0.0)
    p.add_argument("--delta_scale",        type=float, default=0.25)
    p.add_argument("--retrieval_tau",      type=float, default=0.2)
    # Filter
    p.add_argument("--train_filter",       default="boundary")
    p.add_argument("--gate_filter",        default=None)
    # Training
    p.add_argument("--steps",              type=int,   default=3000)
    p.add_argument("--eval_every",         type=int,   default=500)
    p.add_argument("--batch_size",         type=int,   default=16)
    p.add_argument("--eval_batch_size",    type=int,   default=64)
    p.add_argument("--grad_accum_steps",   type=int,   default=4)
    p.add_argument("--lr",                 type=float, default=1e-5)
    p.add_argument("--lambda_region",      type=float, default=0.1)
    p.add_argument("--lambda_rank",        type=float, default=0.0)
    p.add_argument("--lambda_kl",          type=float, default=1.0)
    p.add_argument("--lambda_delta",       type=float, default=1e-3)
    p.add_argument("--rank_margin",        type=float, default=0.05)
    p.add_argument("--kl_topk",            type=int,   default=256)
    p.add_argument("--grad_clip",          type=float, default=0.5)
    p.add_argument("--amp",                action="store_true")
    p.add_argument("--eval_before_train",  action="store_true")
    p.add_argument("--fail_on_baseline_mismatch", action="store_true")
    p.add_argument("--use_filtered_train_loader", action="store_true")
    p.add_argument("--device",             default="cuda")
    # Stage 01B compatibility gate
    p.add_argument("--stage01b_audit",     default=None,
                   help="Path to stage01b live_baseline_audit.json. "
                        "Training is blocked unless live_baseline_compatible=True.")
    p.add_argument("--ctx_len_from_audit", action="store_true",
                   help="Override --ctx_len with best_convention.ctx_len from stage01b audit.")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
