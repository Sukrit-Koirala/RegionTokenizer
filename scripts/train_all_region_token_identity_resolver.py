#!/usr/bin/env python3
"""
train_all_region_token_identity_resolver.py — Phase 2B-pre

Tests whether token identity can be learned as an all-region compatibility profile.

For each candidate c in top-M:
  q_c   = query built from token embedding + context + logit features
  Z(h)  = contextual region states from frozen Phase 2A model
  identity_ctx[c] = soft_attention(q_c, Z)   over all 128 region slots
  delta_scores[c] = f(identity_ctx[c])
  final_scores    = base_logits + gate * delta_scores

Variants:
  base_only
  logit_only_mlp
  token_context_identity
  home_region_identity_real
  all_region_identity_real
  all_region_identity_shuffled
  all_region_identity_random
  all_region_identity_real_coord_permuted

Phase 2A region meaning models are FROZEN by default.
Gold used only for loss/metrics after scores are computed.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice",       category=RuntimeWarning)

_EPS   = 1e-9
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from train_contextual_region_meaning_learner import (
    ContextualRegionMeaningLearner,
    load_shards,
    load_unembedding,
    load_region_maps,
    _nanmean,
    _wcsv,
    _fmt,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_batch(data, idx, M):
    d_in = max(data.get("d_model", 1), 1)
    b = {
        "topk_ids": data["topk_ids"][idx],
        "topk_lgt": data["topk_lgt"][idx],
        "h": data["h"][idx] if data["has_h"] else np.zeros((len(idx), d_in), np.float32),
    }
    if data["has_router"]:
        b["router_reg"] = data["router_reg"][idx]
        b["router_prb"] = data["router_prb"][idx]
    return b


def _to_device(batch, device):
    return {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
            for k, v in batch.items()}


def _logsoftmax_nll(scores, gold_idx):
    s = np.asarray(scores, np.float64); s = s - s.max()
    ls = s - np.log(np.exp(s).sum() + _EPS)
    return float(-ls[gold_idx])


def _safediv(a, b, default=float("nan")):
    if b == 0 or b != b:
        return default
    return a / b


# ══════════════════════════════════════════════════════════════════════════════
# AllRegionTokenIdentityResolver
# ══════════════════════════════════════════════════════════════════════════════

class AllRegionTokenIdentityResolver(nn.Module):
    """
    Modes
    -----
    logit_only   – only base-logit/rank features, no token emb, no context, no Z
    token_context – token emb + h_ctx, no Z
    home_region  – token emb + h_ctx + candidate's home-region state only
    all_region   – full soft attention over all region states Z
    """

    def __init__(self,
                 tok_arr_v, reg_arr, U_frozen,
                 n_regions, n_super, d_in,
                 d_model, hidden_dim, num_heads, dropout,
                 region_emb_dim, super_emb_dim,
                 region_state_dim,          # Phase 2A d_model
                 use_gate, gate_init_bias,
                 mode="all_region"):
        super().__init__()
        assert mode in ("logit_only", "token_context", "home_region", "all_region")

        self.register_buffer("tok_arr_v", torch.from_numpy(tok_arr_v).int())
        self.register_buffer("reg_arr",   torch.from_numpy(reg_arr).int())
        self.register_buffer("U_frozen",  U_frozen.float())

        self.R    = n_regions
        self.S    = n_super
        self.d_model = d_model
        self.mode = mode
        self.use_gate = use_gate
        self.home_region_only = (mode == "home_region")

        self.use_token_emb     = (mode != "logit_only")
        self.use_context       = (mode != "logit_only")
        self.use_region_states = (mode in ("home_region", "all_region"))

        token_dim = U_frozen.shape[1]

        # ── Query feature dimension ────────────────────────────────────────────
        q_dim = 4   # always: base_logit, rank, gap_to_base, base_margin
        if self.use_token_emb:
            self.U_proj = nn.Linear(token_dim, d_model)
            q_dim += d_model
        if self.use_context:
            self.h_proj = nn.Linear(max(d_in, 1), d_model)
            q_dim += d_model
        if self.use_region_states:
            self.region_emb = nn.Embedding(n_regions + 1, region_emb_dim,
                                           padding_idx=n_regions)
            self.super_emb  = nn.Embedding(n_super + 1,   super_emb_dim,
                                           padding_idx=n_super)
            q_dim += region_emb_dim + super_emb_dim + 4   # +4: same_reg, same_sreg, rtr_prob, rtr_rank

        self.query_mlp = nn.Sequential(
            nn.Linear(q_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ── Identity feature dimension ─────────────────────────────────────────
        id_dim = d_model + 4   # query output + logit features
        if self.use_region_states:
            id_dim += region_state_dim * 2    # identity_ctx + diff_from_base
            if not self.home_region_only:
                # all_region: K/V projections + coord features
                self.region_k_proj  = nn.Linear(region_state_dim, d_model)
                self.region_v_proj  = nn.Linear(region_state_dim, d_model)
                self.coord_adv_dim  = max(n_regions // 2, 32)  # 64 for R=128
                self.coord_adv_proj = nn.Linear(n_regions, self.coord_adv_dim)
                id_dim += 11 + self.coord_adv_dim

        self.id_dim = id_dim

        # ── Delta head (zero-init last layer → step-0 = base) ─────────────────
        self.delta_head = nn.Sequential(
            nn.Linear(id_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

        # ── Gate head (init to near-zero sigmoid) ──────────────────────────────
        if use_gate:
            self.gate_head = nn.Sequential(
                nn.Linear(id_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, gate_init_bias)

    # ── Router per-candidate features ─────────────────────────────────────────

    def _router_features(self, cand_reg_v, router_reg, router_prb):
        """Returns [B, M, 2]: (router_prob, router_rank_normalized)."""
        B, M = cand_reg_v.shape
        K    = router_reg.shape[1]
        device = cand_reg_v.device
        r_prob = torch.zeros(B, M, device=device)
        r_rank = torch.ones(B, M, device=device)   # default = 1.0 (not found)
        for ki in range(K):
            rr   = router_reg[:, ki:ki+1].long().clamp(0, self.R - 1)   # [B, 1]
            hit  = (cand_reg_v == rr)                                     # [B, M] bool
            prob = router_prb[:, ki:ki+1].float()
            r_prob = r_prob + hit.float() * prob
            rv = ki / max(K - 1, 1)
            not_found = (r_rank == 1.0)
            r_rank = torch.where(hit & not_found, torch.full_like(r_rank, rv), r_rank)
        return torch.stack([r_prob, r_rank], dim=-1)   # [B, M, 2]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, topk_ids, topk_lgt, h=None,
                region_states=None, router_reg=None, router_prb=None, **kwargs):
        """
        region_states: [B, R, region_state_dim] from frozen Phase 2A model.
        Returns (delta_scores, gate, final_scores, coord_logits, coord_attn, identity_ctx).
        """
        B, M   = topk_ids.shape
        device = topk_ids.device
        topk_lgt = topk_lgt.float()

        # ── Logit features [B, M, 4] ────────────────────────────────────────
        rank_f    = (torch.arange(M, device=device).float() / max(M - 1, 1)
                     ).unsqueeze(0).expand(B, -1)
        gap_f     = (topk_lgt[:, 0:1] - topk_lgt).clamp(min=0.0)
        lgt_sort  = topk_lgt.sort(dim=1, descending=True).values
        margin_f  = (lgt_sort[:, 0:1] - lgt_sort[:, 1:2]).clamp(min=0.0).expand(B, M)
        logit_f   = torch.stack([topk_lgt, rank_f, gap_f, margin_f], dim=-1)  # [B,M,4]

        # ── Token embeddings [B, M, d_model] ────────────────────────────────
        U_cand = None
        if self.use_token_emb:
            Vt = self.U_frozen.shape[0]
            ids = topk_ids.clamp(0, Vt - 1).long()
            U_cand = self.U_proj(self.U_frozen[ids.reshape(-1)].float()
                                 ).reshape(B, M, self.d_model)

        # ── Context [B, d_model] → [B, M, d_model] ──────────────────────────
        h_feat = None
        if self.use_context and h is not None:
            h_feat = self.h_proj(h.float()).unsqueeze(1).expand(B, M, -1)

        # ── Region features per candidate ────────────────────────────────────
        cand_reg_v = None
        region_feat = None
        if self.use_region_states:
            Vt = self.tok_arr_v.shape[0]
            cand_reg_v = self.tok_arr_v[topk_ids.clamp(0, Vt-1).long()
                                        ].long().clamp(0, self.R - 1)   # [B, M]
            cand_sreg = self.reg_arr[cand_reg_v.clamp(0, len(self.reg_arr)-1)
                                     ].long().clamp(0, self.S - 1)      # [B, M]
            re  = self.region_emb(cand_reg_v)    # [B, M, rem_dim]
            se  = self.super_emb(cand_sreg)      # [B, M, sem_dim]
            base_reg  = cand_reg_v[:, 0:1].expand(B, M)
            base_sreg = cand_sreg[:, 0:1].expand(B, M)
            same_r  = (cand_reg_v  == base_reg).float().unsqueeze(-1)
            same_sr = (cand_sreg   == base_sreg).float().unsqueeze(-1)
            if router_reg is not None:
                rtr = self._router_features(cand_reg_v, router_reg, router_prb)  # [B,M,2]
            else:
                rtr = torch.zeros(B, M, 2, device=device)
            region_feat = torch.cat([re, se, same_r, same_sr, rtr], dim=-1)

        # ── Candidate query feature → query [B, M, d_model] ─────────────────
        parts = [logit_f]
        if U_cand      is not None: parts.append(U_cand)
        if h_feat      is not None: parts.append(h_feat)
        if region_feat is not None: parts.append(region_feat)
        cand_feat = torch.cat(parts, dim=-1)   # [B, M, q_dim]
        query = self.query_mlp(cand_feat.reshape(B * M, -1)).reshape(B, M, self.d_model)

        # ── Identity context ─────────────────────────────────────────────────
        coord_logits = coord_attn = identity_ctx = None

        if region_states is not None and self.use_region_states:
            R_ = region_states.shape[1]
            rs  = region_states.float()

            if self.home_region_only:
                # Lookup Z[candidate's variant region]
                identity_ctx = rs[
                    torch.arange(B, device=device).unsqueeze(1),
                    cand_reg_v.clamp(0, R_ - 1)
                ]  # [B, M, d_region]

            else:
                # Full attention over all region states
                K_mat = self.region_k_proj(rs)   # [B, R, d_model]
                V_mat = self.region_v_proj(rs)   # [B, R, d_model]
                coord_logits = torch.bmm(query, K_mat.transpose(1, 2)) / math.sqrt(self.d_model)  # [B,M,R]
                coord_attn   = torch.softmax(coord_logits, dim=-1)
                identity_ctx = torch.bmm(coord_attn, V_mat)  # [B, M, d_model]

        # ── Identity feature ─────────────────────────────────────────────────
        id_parts = [query, logit_f]

        if identity_ctx is not None:
            ictx_base = identity_ctx[:, 0:1, :].expand_as(identity_ctx)
            id_parts += [identity_ctx, identity_ctx - ictx_base]

            if coord_logits is not None and cand_reg_v is not None:
                R_ = coord_attn.shape[-1]
                ca  = coord_attn   # [B, M, R]
                cl  = coord_logits

                # Entropy + top stats
                ent   = -(ca * (ca + _EPS).log()).sum(-1, keepdim=True)      # [B,M,1]
                cmax  = ca.max(-1, keepdim=True).values
                top2v = torch.topk(ca, 2, dim=-1).values
                cmarg = (top2v[:, :, 0:1] - top2v[:, :, 1:2])

                # At-region features
                reg_idx = cand_reg_v.clamp(0, R_-1).unsqueeze(-1)           # [B,M,1]
                breg_idx = cand_reg_v[:, 0:1].clamp(0, R_-1).unsqueeze(-1).expand(B, M, 1)
                coord_at_home     = ca.gather(-1, reg_idx)
                coord_at_base_h   = ca.gather(-1, breg_idx)
                raw_at_home       = cl.gather(-1, reg_idx)
                raw_at_base_h     = cl.gather(-1, breg_idx)

                if router_reg is not None:
                    rt1 = router_reg[:, 0:1].long().clamp(0, R_-1)  # [B, 1]
                    rt1_exp = rt1.unsqueeze(1).expand(B, M, 1)       # [B, M, 1]
                    coord_at_rtr = ca.gather(-1, rt1_exp)
                else:
                    coord_at_rtr = torch.zeros(B, M, 1, device=device)

                # coord_adv
                cadv = ca - ca[:, 0:1, :]                    # [B, M, R]
                cadv_proj = self.coord_adv_proj(
                    cadv.reshape(B * M, R_)).reshape(B, M, -1)
                cadv_at_home = cadv.gather(-1, reg_idx)
                cadv_max     = cadv.max(-1, keepdim=True).values
                cadv_mean    = cadv.mean(-1, keepdim=True)
                raw_cadv     = cl - cl[:, 0:1, :]
                raw_cadv_at_home = raw_cadv.gather(-1, reg_idx)
                raw_cadv_at_base = raw_cadv.gather(-1, breg_idx)

                coord_sf = torch.cat([
                    ent, cmax, cmarg,
                    coord_at_home, coord_at_base_h, coord_at_rtr,
                    cadv_at_home, cadv_max, cadv_mean,
                    raw_cadv_at_home, raw_cadv_at_base,
                ], dim=-1)  # [B, M, 11]

                id_parts += [coord_sf, cadv_proj]

        identity_feat = torch.cat(id_parts, dim=-1)  # [B, M, id_dim]

        # ── Delta + gate → final scores ──────────────────────────────────────
        delta_scores = self.delta_head(
            identity_feat.reshape(B * M, self.id_dim)).reshape(B, M)

        if self.use_gate:
            gate = torch.sigmoid(
                self.gate_head(identity_feat.reshape(B * M, self.id_dim)).reshape(B, M))
        else:
            gate = torch.ones(B, M, device=device)

        final_scores = topk_lgt + gate * delta_scores

        return delta_scores, gate, final_scores, coord_logits, coord_attn, identity_ctx


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2A model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_phase2a_model(ckpt_path, tok_arr_v, reg_arr, U, n_regions, n_super,
                       d_in, phase2a_d_model, phase2a_cfg, device):
    """Load a frozen Phase 2A region-meaning model."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Phase 2A checkpoint not found: {ckpt_path}")
    saved  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = saved.get("variant", "?")
    sd = saved["state_dict"]

    region_emb_dim    = int(phase2a_cfg.get("region_emb_dim",   64))
    super_emb_dim     = int(phase2a_cfg.get("super_emb_dim",    32))
    num_region_layers = int(phase2a_cfg.get("num_region_layers", 2))
    num_heads         = int(phase2a_cfg.get("num_heads",         4))
    ff_mult           = int(phase2a_cfg.get("ff_mult",           4))

    use_context = ("static" not in variant)
    model = ContextualRegionMeaningLearner(
        tok_arr=tok_arr_v, reg_arr=reg_arr, U_frozen=U,
        n_regions=n_regions, n_super=n_super, d_in=d_in,
        d_model=phase2a_d_model,
        region_emb_dim=region_emb_dim, super_emb_dim=super_emb_dim,
        num_region_layers=num_region_layers, num_heads=num_heads,
        ff_mult=ff_mult, dropout=0.0, use_context=use_context,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[phase2a] loaded {ckpt_path}  variant={variant}  frozen")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Bucket / slice labels  (real map only)
# ══════════════════════════════════════════════════════════════════════════════

def compute_buckets(data, tok_arr_real, reg_arr_real, n_regions_real):
    gold     = data["gold"].astype(np.int64)
    topk     = data["topk_ids"].astype(np.int64)
    N, M     = topk.shape
    Vt = len(tok_arr_real); Rr = len(reg_arr_real)
    unk_r    = n_regions_real; unk_s = int(reg_arr_real.max())

    base_top1    = topk[:, 0]
    base_correct = (base_top1 == gold)
    gip_M        = (topk == gold[:, None]).any(axis=1)
    bucketA      = ~base_correct & gip_M
    gold_idx_in  = np.where(gip_M, (topk == gold[:, None]).argmax(axis=1), -1)

    gold_reg = tok_arr_real[np.clip(gold,      0, Vt-1)].astype(np.int64)
    base_reg = tok_arr_real[np.clip(base_top1, 0, Vt-1)].astype(np.int64)
    gold_sreg= reg_arr_real[np.clip(gold_reg, 0, Rr-1)].astype(np.int64)
    base_sreg= reg_arr_real[np.clip(base_reg, 0, Rr-1)].astype(np.int64)

    gold_known = (gold_reg < unk_r)
    same_reg   = (gold_reg == base_reg) & gold_known & (base_reg < unk_r)
    same_sreg  = ((gold_sreg == base_sreg) & gold_known & (base_reg < unk_r)
                  & (gold_sreg < unk_s) & (base_sreg < unk_s))

    print(f"[buckets] N={N:,}  gip_M={gip_M.sum():,}  bucketA={bucketA.sum():,}  "
          f"same_reg_conf={(bucketA & same_reg).sum():,}  "
          f"base_correct={base_correct.sum():,}")
    return {
        "gold": gold, "base_top1": base_top1, "topk": topk,
        "base_correct": base_correct, "gip_M": gip_M, "bucketA": bucketA,
        "gold_idx_in": gold_idx_in,
        "gold_reg": gold_reg, "base_reg": base_reg,
        "gold_sreg": gold_sreg, "base_sreg": base_sreg,
        "gold_known": gold_known,
        "same_reg": same_reg, "same_sreg": same_sreg,
    }


def make_balanced_sampler(buckets, n_steps, batch_size, seed):
    rng = np.random.default_rng(seed)
    N   = len(buckets["gold"])
    ra  = np.where(buckets["bucketA"])[0]
    rc  = np.where(buckets["bucketA"] & (buckets["same_reg"] | buckets["same_sreg"]))[0]
    rb  = np.where(buckets["base_correct"])[0]
    rg  = np.where(buckets["gip_M"])[0]
    all_= np.arange(N)
    na  = max(1, int(batch_size * 0.35))
    nc  = max(1, int(batch_size * 0.25))
    nb  = max(1, int(batch_size * 0.25))
    ng  = batch_size - na - nc - nb

    def samp(arr, n):
        if len(arr) == 0: return rng.choice(all_, n, replace=True)
        return rng.choice(arr, n, replace=len(arr) < n)

    return [np.concatenate([samp(ra, na), samp(rc, nc), samp(rb, nb), samp(rg, ng)])
            for _ in range(n_steps)]


# ══════════════════════════════════════════════════════════════════════════════
# Loss computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_losses(final_scores, delta_scores, gate,
                   gold_t, topk_ids_t, lgt_t,
                   tok_arr_real, n_regions_real,
                   gip_mask, bucketA_mask, same_reg_mask, same_sreg_mask,
                   base_correct_mask,
                   args, device):
    B, M = final_scores.shape
    losses = {}

    # L1 — Candidate CE (gip rows only)
    if gip_mask.sum() > 0:
        gold_idx = (topk_ids_t[gip_mask] == gold_t[gip_mask].unsqueeze(1)).float().argmax(1)
        losses["ce"] = F.cross_entropy(final_scores[gip_mask], gold_idx)
    else:
        losses["ce"] = torch.zeros(1, device=device).squeeze()

    # L2 — Gold-vs-base pair loss (bucketA)
    if bucketA_mask.sum() > 0:
        gi = (topk_ids_t[bucketA_mask] == gold_t[bucketA_mask].unsqueeze(1)).float().argmax(1)
        n_ba = int(bucketA_mask.sum())
        s_gold = final_scores[bucketA_mask][torch.arange(n_ba, device=device), gi]
        s_base = final_scores[bucketA_mask][:, 0]
        losses["pair"] = F.relu(args.margin - (s_gold - s_base)).mean()
    else:
        losses["pair"] = torch.zeros(1, device=device).squeeze()

    # L3 — Same-region confuser pair loss
    sr_mask = bucketA_mask & same_reg_mask
    if sr_mask.sum() > 0:
        gi = (topk_ids_t[sr_mask] == gold_t[sr_mask].unsqueeze(1)).float().argmax(1)
        n_sr = int(sr_mask.sum())
        s_g  = final_scores[sr_mask][torch.arange(n_sr, device=device), gi]
        s_b  = final_scores[sr_mask][:, 0]
        losses["same_reg_pair"] = F.relu(args.margin - (s_g - s_b)).mean()
    else:
        losses["same_reg_pair"] = torch.zeros(1, device=device).squeeze()

    ss_mask = bucketA_mask & same_sreg_mask
    if ss_mask.sum() > 0:
        gi = (topk_ids_t[ss_mask] == gold_t[ss_mask].unsqueeze(1)).float().argmax(1)
        n_ss = int(ss_mask.sum())
        s_g  = final_scores[ss_mask][torch.arange(n_ss, device=device), gi]
        s_b  = final_scores[ss_mask][:, 0]
        losses["same_sreg_pair"] = F.relu(args.margin - (s_g - s_b)).mean()
    else:
        losses["same_sreg_pair"] = torch.zeros(1, device=device).squeeze()

    # L4 — Within-real-region CE
    Vt = len(tok_arr_real)
    topk_np = topk_ids_t.cpu().long()
    gold_np  = gold_t.cpu().long()
    gip_np   = gip_mask.cpu().numpy()
    cand_rr  = torch.from_numpy(
        tok_arr_real[np.clip(topk_np.numpy(), 0, Vt-1)].astype(np.int64)
    ).to(device)  # [B, M]
    gold_rr = tok_arr_real[np.clip(gold_np.numpy(), 0, Vt-1)].astype(np.int64)
    gold_rr_t = torch.from_numpy(gold_rr).to(device)

    within_ces = []
    for b in range(B):
        if not gip_np[b]: continue
        gr = int(gold_rr[b])
        if gr >= n_regions_real: continue
        in_r = (cand_rr[b] == gr)
        if in_r.sum() < 2: continue
        idxs = in_r.nonzero(as_tuple=False).squeeze(1)
        gf_matches = (topk_np[b] == gold_np[b]).nonzero(as_tuple=False)
        if len(gf_matches) == 0: continue
        gf = gf_matches[0, 0].item()
        if not in_r[gf]: continue
        gold_within = (idxs == gf).nonzero(as_tuple=False)
        if len(gold_within) == 0: continue
        gw = gold_within[0, 0].item()
        scores_r = final_scores[b, idxs]
        within_ces.append(F.cross_entropy(scores_r.unsqueeze(0),
                                           torch.tensor([gw], device=device)))
    if within_ces:
        losses["within_region"] = torch.stack(within_ces).mean()
    else:
        losses["within_region"] = torch.zeros(1, device=device).squeeze()

    # L5 — Base-correct preservation (CE with target=0)
    if base_correct_mask.sum() > 0:
        losses["preserve"] = F.cross_entropy(final_scores[base_correct_mask],
                                              torch.zeros(base_correct_mask.sum(),
                                                          dtype=torch.long, device=device))
    else:
        losses["preserve"] = torch.zeros(1, device=device).squeeze()

    # L6 — Regularization
    losses["delta_l2"] = delta_scores.pow(2).mean()
    losses["gate_mean"] = gate.mean() if isinstance(gate, torch.Tensor) else torch.zeros(1, device=device).squeeze()

    total = (args.lambda_ce              * losses["ce"]
           + args.lambda_pair            * losses["pair"]
           + args.lambda_same_region_pair    * losses["same_reg_pair"]
           + args.lambda_same_superregion_pair * losses["same_sreg_pair"]
           + args.lambda_within_region   * losses["within_region"]
           + args.lambda_preserve        * losses["preserve"]
           + args.lambda_delta           * losses["delta_l2"]
           + args.lambda_gate            * losses["gate_mean"])

    return total, {k: v.item() for k, v in losses.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_resolver(variant, resolver, phase2a_model,
                  data, buckets, tok_arr_real, reg_arr_real, n_regions_real,
                  M, device, batch_size, coord_perm=None):
    """Evaluate one variant. resolver may be None (base_only)."""
    N    = len(data["gold"])
    gold = buckets["gold"]
    topk = buckets["topk"]
    lgt  = data["topk_lgt"].astype(np.float32)
    Vt   = len(tok_arr_real); Rr = len(reg_arr_real)

    # Per-row result arrays
    final_scores_all  = np.empty((N, M), np.float32)
    delta_scores_all  = np.zeros((N, M), np.float32)
    gate_all          = np.ones((N, M), np.float32)
    # Coord diagnostics (all_region mode only)
    coord_ent_all     = np.full(N, np.nan, np.float32)
    coord_top1_home   = np.full(N, np.nan, np.float32)
    coord_adv_norm    = np.full(N, np.nan, np.float32)

    if resolver is not None:
        resolver.eval()

    with torch.no_grad():
        for s in range(0, N, batch_size):
            e   = min(s + batch_size, N)
            idx = np.arange(s, e)
            B_  = e - s

            b = _to_device(_make_batch(data, idx, M), device)

            # Phase 2A region states
            Z = None
            if phase2a_model is not None:
                Z_, _, _, _ = phase2a_model(**b)
                Z = Z_.float()
                if coord_perm is not None:
                    Z = Z[:, coord_perm, :]

            if resolver is None:
                # base_only
                final_scores_all[s:e] = b["topk_lgt"].cpu().float().numpy()
                continue

            ds, g, fs, clog, cattn, _ = resolver(
                topk_ids=b["topk_ids"].long(),
                topk_lgt=b["topk_lgt"].float(),
                h=b.get("h"),
                region_states=Z,
                router_reg=b.get("router_reg"),
                router_prb=b.get("router_prb"),
            )

            final_scores_all[s:e]  = fs.cpu().float().numpy()
            delta_scores_all[s:e]  = ds.cpu().float().numpy()
            gate_all[s:e]          = g.cpu().float().numpy() if isinstance(g, torch.Tensor) else 1.0

            # Coordinate diagnostics (for all_region mode)
            if cattn is not None:
                # cattn [B_, M, R]
                ent = -(cattn * (cattn + _EPS).log()).sum(-1)  # [B_, M]
                # Mean over candidates present in gip rows
                gip_b = buckets["gip_M"][s:e]
                if gip_b.any():
                    coord_ent_all[s:e] = np.where(gip_b, ent.mean(-1).cpu().numpy(), np.nan)

                # Top-1 attended region == candidate home region?
                if resolver.use_region_states and resolver.tok_arr_v is not None:
                    Vt_ = resolver.tok_arr_v.shape[0]
                    cand_reg = resolver.tok_arr_v[
                        b["topk_ids"].clamp(0, Vt_-1).long()].long().clamp(0, cattn.shape[-1]-1)  # [B_,M]
                    top1_attn = cattn.argmax(-1)  # [B_, M]
                    is_home = (top1_attn == cand_reg).float().mean(-1).cpu().numpy()  # [B_]
                    coord_top1_home[s:e] = np.where(gip_b, is_home, np.nan)

                    # coord_adv norm (base vs self)
                    cadv = cattn - cattn[:, 0:1, :]  # [B_, M, R]
                    adv_norm = cadv.norm(dim=-1).mean(-1).cpu().numpy()  # [B_]
                    coord_adv_norm[s:e] = np.where(gip_b, adv_norm, np.nan)

    # ── Aggregate metrics ──────────────────────────────────────────────────
    bkts = buckets
    gip  = bkts["gip_M"]
    bc   = bkts["base_correct"]
    ba   = bkts["bucketA"]
    sr   = bkts["same_reg"]
    ss   = bkts["same_sreg"]
    gold_idx_in = bkts["gold_idx_in"]

    base_pred  = lgt.argmax(axis=1)
    final_pred = final_scores_all.argmax(axis=1)
    base_pred_id  = topk[np.arange(N), base_pred]
    final_pred_id = topk[np.arange(N), final_pred]

    changed_to_gold = int(((base_pred_id != gold) & (final_pred_id == gold) & gip).sum())
    changed_away    = int(((base_pred_id == gold) & (final_pred_id != gold) & bc).sum())
    no_change       = int((final_pred_id == base_pred_id).sum())
    wrong_to_wrong  = int(((base_pred_id != gold) & (final_pred_id != gold) &
                           (final_pred_id != base_pred_id) & gip).sum())

    n_gip  = int(gip.sum())
    n_ba   = int(ba.sum())
    n_bc   = int(bc.sum())
    net_corr = changed_to_gold - changed_away

    base_dam_rate = _safediv(changed_away, n_bc)
    benef = _safediv(changed_to_gold, max(n_ba, 1))
    dmg   = _safediv(changed_away,   max(n_bc, 1))
    bdr   = _safediv(benef, dmg + 1e-6)

    mean_gate  = float(gate_all.mean())
    mean_abs_d = float(np.abs(delta_scores_all).mean())
    max_abs_d  = float(np.abs(delta_scores_all).max())

    # Candidate NLL
    def _cand_nll(scores, valid_mask, gold_idx_in):
        nlls = []
        idx  = np.where(valid_mask)[0]
        for n in idx:
            gi = gold_idx_in[n]
            if gi < 0: continue
            nlls.append(_logsoftmax_nll(scores[n], gi))
        return _nanmean(nlls)

    def _pair_acc(scores, mask):
        m   = mask & gip
        n_m = int(m.sum())
        if n_m == 0: return float("nan")
        gi  = np.clip(gold_idx_in[m], 0, M-1)
        s_g = scores[m][np.arange(n_m), gi]
        s_b = scores[m][:, 0]
        return float((s_g > s_b).mean())

    base_nll = _cand_nll(lgt,              gip, gold_idx_in)
    model_nll= _cand_nll(final_scores_all, gip, gold_idx_in)
    nll_gain = (base_nll - model_nll) if (base_nll == base_nll and model_nll == model_nll) else float("nan")

    base_acc  = float((base_pred_id[gip]  == gold[gip]).mean())  if gip.sum() > 0 else float("nan")
    model_acc = float((final_pred_id[gip] == gold[gip]).mean())  if gip.sum() > 0 else float("nan")
    acc_gain  = (model_acc - base_acc) if (model_acc == model_acc and base_acc == base_acc) else float("nan")

    # Within-real-region metrics
    wr_acc_arr    = np.full(N, np.nan)
    wr_nll_arr    = np.full(N, np.nan)
    wr_rank_arr   = np.full(N, np.nan)
    bwr_acc_arr   = np.full(N, np.nan)
    bwr_nll_arr   = np.full(N, np.nan)

    for n in range(N):
        if not gip[n] or bkts["gold_reg"][n] >= n_regions_real: continue
        gr = bkts["gold_reg"][n]
        cand_rr = tok_arr_real[np.clip(topk[n], 0, Vt-1)].astype(np.int64)
        in_r = (cand_rr == gr)
        if in_r.sum() < 2: continue
        idxs = np.where(in_r)[0]
        gf = int(gold_idx_in[n])
        if gf < 0 or not in_r[gf]: continue
        gw_arr = np.where(idxs == gf)[0]
        if len(gw_arr) == 0: continue
        gw = int(gw_arr[0])

        sc_m  = final_scores_all[n, idxs]
        sc_b  = lgt[n, idxs]
        wr_acc_arr[n]  = float(sc_m.argmax() == gw)
        wr_nll_arr[n]  = _logsoftmax_nll(sc_m, gw)
        wr_rank_arr[n] = float((sc_m > sc_m[gw]).sum())
        bwr_acc_arr[n] = float(sc_b.argmax() == gw)
        bwr_nll_arr[n] = _logsoftmax_nll(sc_b, gw)

    wr_mask = ~np.isnan(wr_acc_arr)
    wr_both = wr_mask & ~np.isnan(bwr_acc_arr)
    wr_acc  = _nanmean(wr_acc_arr[wr_mask])
    wr_nll  = _nanmean(wr_nll_arr[wr_mask])
    wr_rank = _nanmean(wr_rank_arr[wr_mask])
    wr_rank_med = float(np.nanmedian(wr_rank_arr[wr_mask])) if wr_mask.sum() > 0 else float("nan")
    bwr_acc = _nanmean(bwr_acc_arr[wr_both])
    bwr_nll = _nanmean(bwr_nll_arr[wr_both])
    wr_acc_gain = _nanmean(wr_acc_arr[wr_both] - bwr_acc_arr[wr_both]) if wr_both.sum() > 0 else float("nan")
    wr_nll_gain = _nanmean(bwr_nll_arr[wr_both] - wr_nll_arr[wr_both]) if wr_both.sum() > 0 else float("nan")

    # Validation score
    def _s(v, d=0.0): return v if (v == v) else d
    val_score = (_s(nll_gain)
                 + 0.5 * _s(wr_nll_gain)
                 + 0.001 * net_corr
                 - 0.5 * _s(base_dam_rate))

    metrics = {
        "n_val":                          N,
        "n_gip_M":                        n_gip,
        "n_bucketA":                      n_ba,
        "n_base_correct":                 n_bc,
        "natural_gold_in_topM_rate":      _safediv(n_gip, N),
        # Candidate metrics
        "base_candidate_nll_given_gip":   base_nll,
        "model_candidate_nll_given_gip":  model_nll,
        "candidate_nll_gain_vs_base":     nll_gain,
        "base_candidate_acc_given_gip":   base_acc,
        "model_candidate_acc_given_gip":  model_acc,
        "candidate_acc_gain_vs_base":     acc_gain,
        # Correction metrics
        "changed_to_gold":                changed_to_gold,
        "changed_away":                   changed_away,
        "net_correction":                 net_corr,
        "wrong_to_wrong_change":          wrong_to_wrong,
        "no_change":                      no_change,
        "benefit_damage_ratio":           bdr,
        "base_correct_damage_rate":       base_dam_rate,
        "apply_rate":                     float((final_pred_id != base_pred_id).mean()),
        "mean_gate":                      mean_gate,
        "mean_abs_delta":                 mean_abs_d,
        "max_abs_delta":                  max_abs_d,
        # Confuser pair
        "bucketA_pair_acc":               _pair_acc(final_scores_all, ba),
        "same_region_pair_acc":           _pair_acc(final_scores_all, ba & sr),
        "same_superregion_pair_acc":      _pair_acc(final_scores_all, ba & ss),
        "different_region_pair_acc":      _pair_acc(final_scores_all, ba & ~sr),
        # Within-real-region
        "within_real_region_acc":         wr_acc,
        "within_real_region_nll":         wr_nll,
        "within_real_region_acc_gain_vs_base": wr_acc_gain,
        "within_real_region_nll_gain_vs_base": wr_nll_gain,
        "mean_gold_rank_inside_real_region":   wr_rank,
        "median_gold_rank_inside_real_region": wr_rank_med,
        "base_within_real_region_acc":    bwr_acc,
        "base_within_real_region_nll":    bwr_nll,
        # Coord diagnostics
        "coord_entropy_mean":             _nanmean(coord_ent_all),
        "coord_top1_is_home_region_rate": _nanmean(coord_top1_home),
        "coord_adv_norm_mean":            _nanmean(coord_adv_norm),
        # Validation
        "validation_score":               float(val_score),
    }

    # Slice metrics
    slices = {
        "all":                  np.ones(N, bool),
        "gold_in_topM":         gip,
        "bucketA":              ba,
        "base_correct":         bc,
        "same_region_confuser": ba & sr,
        "same_superregion_confuser": ba & ss,
        "different_region_confuser": ba & ~sr & gip,
        "gold_rank_1_5":    gip & (gold_idx_in >= 0) & (gold_idx_in < 5),
        "gold_rank_6_32":   gip & (gold_idx_in >= 5) & (gold_idx_in < 32),
        "gold_rank_33_M":   gip & (gold_idx_in >= 32),
    }

    def _slice_row(sl_name, mask):
        n = int(mask.sum())
        if n == 0:
            return {"variant": variant, "slice": sl_name, "n": 0,
                    "base_acc": float("nan"), "model_acc": float("nan"),
                    "acc_gain": float("nan"), "base_nll": float("nan"),
                    "model_nll": float("nan"), "nll_gain": float("nan"),
                    "changed_to_gold": 0, "changed_away": 0,
                    "pair_acc": float("nan"), "within_region_acc": float("nan")}
        b_a = float((base_pred_id[mask] == gold[mask]).mean()) if mask.sum() > 0 else float("nan")
        m_a = float((final_pred_id[mask] == gold[mask]).mean()) if mask.sum() > 0 else float("nan")
        b_n = _cand_nll(lgt, mask & gip, gold_idx_in)
        m_n = _cand_nll(final_scores_all, mask & gip, gold_idx_in)
        return {
            "variant": variant, "slice": sl_name, "n": n,
            "base_acc": b_a, "model_acc": m_a,
            "acc_gain": (m_a - b_a) if (m_a == m_a and b_a == b_a) else float("nan"),
            "base_nll": b_n, "model_nll": m_n,
            "nll_gain": (b_n - m_n) if (b_n == b_n and m_n == m_n) else float("nan"),
            "changed_to_gold": int(((base_pred_id != gold) & (final_pred_id == gold) & mask & gip).sum()),
            "changed_away":    int(((base_pred_id == gold) & (final_pred_id != gold) & mask).sum()),
            "pair_acc": _pair_acc(final_scores_all, mask) if sl_name != "all" else float("nan"),
            "within_region_acc": _nanmean(wr_acc_arr[mask]),
        }

    slice_rows = [_slice_row(sn, mk) for sn, mk in slices.items()]
    return metrics, slice_rows


# ══════════════════════════════════════════════════════════════════════════════
# Training loop (one variant)
# ══════════════════════════════════════════════════════════════════════════════

def train_variant(variant, resolver, phase2a_model,
                  train_data, val_data,
                  train_buckets, tok_arr_real, reg_arr_real, n_regions_real,
                  M, args, device, out_dir, coord_perm=None):

    optimizer = torch.optim.AdamW(resolver.parameters(), lr=args.lr,
                                   weight_decay=1e-5)
    scaler    = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    sampler   = make_balanced_sampler(train_buckets, args.steps, args.batch_size, args.seed)
    Vt        = len(tok_arr_real)

    # Step-0 eval (should equal base)
    resolver.eval()
    init_vm, _ = eval_resolver(
        variant, resolver, phase2a_model,
        val_data, compute_buckets(val_data, tok_arr_real, reg_arr_real, n_regions_real),
        tok_arr_real, reg_arr_real, n_regions_real,
        M, device, args.batch_size * 4, coord_perm=coord_perm)
    init_score = init_vm["validation_score"]
    print(f"[{variant}] step=0  val_score={init_score:.5f}  "
          f"nll_gain={init_vm['candidate_nll_gain_vs_base']:.4f}  "
          f"max_abs_delta={init_vm['max_abs_delta']:.6f}")
    if init_vm["max_abs_delta"] > 1e-4:
        print(f"  [WARN] step-0 max_abs_delta={init_vm['max_abs_delta']:.6f} > 1e-4; check zero init")

    best_score = 0.0   # save best only if > 0
    best_ckpt  = None; best_vm = None; best_step = None
    train_log  = []; eval_log  = []

    resolver.train()
    t0 = time.time()

    for step, idx in enumerate(sampler):
        b = _to_device(_make_batch(train_data, idx, M), device)
        gold_t    = torch.from_numpy(train_data["gold"][idx].astype(np.int64)).to(device)
        topk_t    = b["topk_ids"].long()
        topk_np   = train_data["topk_ids"][idx].astype(np.int64)
        gold_np   = train_data["gold"][idx].astype(np.int64)
        base_top1 = topk_np[:, 0]
        gold_rr   = tok_arr_real[np.clip(gold_np, 0, Vt-1)].astype(np.int64)
        base_rr   = tok_arr_real[np.clip(base_top1, 0, Vt-1)].astype(np.int64)
        base_sr   = reg_arr_real[np.clip(base_rr, 0, len(reg_arr_real)-1)].astype(np.int64)
        gold_sr   = reg_arr_real[np.clip(gold_rr, 0, len(reg_arr_real)-1)].astype(np.int64)

        gip_m   = torch.from_numpy((topk_np == gold_np[:, None]).any(axis=1)).to(device)
        ba_m    = torch.from_numpy(~(base_top1 == gold_np) & (topk_np == gold_np[:, None]).any(axis=1)).to(device)
        sr_m    = torch.from_numpy((base_rr == gold_rr) & (gold_rr < n_regions_real)).to(device)
        ss_m    = torch.from_numpy((base_sr == gold_sr) & (gold_sr < int(reg_arr_real.max()))).to(device)
        bc_m    = torch.from_numpy((base_top1 == gold_np)).to(device)

        optimizer.zero_grad()

        # Phase 2A region states (frozen)
        Z = None
        if phase2a_model is not None:
            with torch.no_grad():
                Z_, _, _, _ = phase2a_model(**b)
                Z = Z_.float()
                if coord_perm is not None:
                    Z = Z[:, coord_perm, :]

        ds, g, fs, _, _, _ = resolver(
            topk_ids=b["topk_ids"].long(),
            topk_lgt=b["topk_lgt"].float(),
            h=b.get("h"),
            region_states=Z,
            router_reg=b.get("router_reg"),
            router_prb=b.get("router_prb"),
        )

        total, ld = compute_losses(
            fs, ds, g, gold_t, topk_t, b["topk_lgt"].float(),
            tok_arr_real, n_regions_real,
            gip_m, ba_m, sr_m, ss_m, bc_m, args, device)

        if torch.isnan(total):
            raise RuntimeError(f"[{variant}] NaN loss at step {step}: {ld}")

        if scaler:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(resolver.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(resolver.parameters(), 1.0)
            optimizer.step()

        train_log.append({"step": step + 1, "variant": variant,
                          **{f"loss_{k}": v for k, v in ld.items()},
                          "loss_total": total.item()})

        if (step + 1) % args.eval_every == 0 or step == 0:
            resolver.eval()
            val_bkts = compute_buckets(val_data, tok_arr_real, reg_arr_real, n_regions_real)
            vm, _ = eval_resolver(
                variant, resolver, phase2a_model,
                val_data, val_bkts,
                tok_arr_real, reg_arr_real, n_regions_real,
                M, device, args.batch_size * 4, coord_perm=coord_perm)
            resolver.train()
            vs = vm["validation_score"]
            elapsed = time.time() - t0
            print(f"[{variant}] step={step+1:5d}  "
                  f"loss={ld.get('ce',0):.4f}/{ld.get('pair',0):.4f}/{ld.get('within_region',0):.4f}  "
                  f"nll_gain={vm['candidate_nll_gain_vs_base']:.4f}  "
                  f"wr_nll_gain={vm['within_real_region_nll_gain_vs_base']:.4f}  "
                  f"sr_pair={vm['same_region_pair_acc']:.4f}  "
                  f"net_corr={vm['net_correction']}  "
                  f"val_score={vs:.4f}  {elapsed:.0f}s")
            eval_log.append({"step": step + 1, "variant": variant, **vm})

            if vs > best_score:
                best_score = vs
                best_ckpt  = deepcopy(resolver.state_dict())
                best_vm    = deepcopy(vm)
                best_step  = step + 1
                print(f"  → new best  val_score={vs:.5f}")

    # Final eval
    resolver.eval()
    val_bkts = compute_buckets(val_data, tok_arr_real, reg_arr_real, n_regions_real)
    final_vm, final_slice_rows = eval_resolver(
        variant, resolver, phase2a_model,
        val_data, val_bkts,
        tok_arr_real, reg_arr_real, n_regions_real,
        M, device, args.batch_size * 4, coord_perm=coord_perm)
    eval_log.append({"step": args.steps, "variant": variant, **final_vm})

    ckpt_path = None
    no_best   = (best_ckpt is None)
    if not no_best:
        ckpt_path = os.path.join(out_dir, f"best_{variant}.pt")
        torch.save({"state_dict": best_ckpt, "variant": variant,
                    "val_score": best_score}, ckpt_path)
        print(f"[{variant}] saved best: {ckpt_path}  score={best_score:.5f}")
    else:
        print(f"[{variant}] no checkpoint beat val_score=0  "
              f"(best seen: {best_score:.5f})")

    selected_vm = best_vm if not no_best else final_vm
    return {
        "best":    {**selected_vm, "variant": variant,
                    "no_improving_checkpoint_found": no_best,
                    "selected_for_comparison": "best" if not no_best else "final_no_best",
                    "checkpoint_path": ckpt_path, "best_step": best_step},
        "final":   {"variant": variant, "final_step": args.steps, **final_vm},
        "train_log":  train_log,
        "eval_log":   eval_log,
        "slice_rows": final_slice_rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

_COMP_COLS = [
    "variant", "selected_for_comparison",
    "candidate_nll_gain_vs_base", "candidate_acc_gain_vs_base",
    "within_real_region_nll_gain_vs_base", "within_real_region_acc_gain_vs_base",
    "same_region_pair_acc", "same_superregion_pair_acc",
    "changed_to_gold", "changed_away", "net_correction",
    "benefit_damage_ratio", "base_correct_damage_rate",
    "mean_gate", "mean_abs_delta",
    "coord_entropy_mean", "coord_top1_is_home_region_rate", "coord_adv_norm_mean",
    "validation_score",
]


def build_comparison_row(best_info):
    row = {"variant": best_info["variant"],
           "selected_for_comparison": best_info.get("selected_for_comparison", "?")}
    for k in _COMP_COLS[2:]:
        row[k] = best_info.get(k, float("nan"))
    return row


def write_examples(variant_results, out_dir):
    """Write placeholder example files for each key diagnostic."""
    for fname, title in [
        ("examples_real_identity_helps.md",    "Real Identity Helps — changed_to_gold"),
        ("examples_real_identity_hurts.md",    "Real Identity Hurts — changed_away"),
        ("examples_same_region_fixed.md",      "Same-Region Confuser Fixed"),
        ("examples_same_region_failed.md",     "Same-Region Confuser Failed"),
        ("examples_shuffled_beats_real.md",    "Shuffled Beats Real (diagnostic)"),
    ]:
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(f"# {title}\n\n")
                f.write("Examples are populated by post-processing the eval outputs.\n")
                f.write("Token IDs and logits are available in eval_log.csv and slice_metrics.csv.\n")


def write_report(comparison_rows, out_dir, args):
    def _r(name):
        return next((r for r in comparison_rows if r["variant"] == name), {})

    def _v(row, k): return row.get(k, float("nan"))

    real_r  = _r("all_region_identity_real")
    stat_r  = _r("token_context_identity")
    home_r  = _r("home_region_identity_real")
    shuf_r  = _r("all_region_identity_shuffled")
    rand_r  = _r("all_region_identity_random")
    perm_r  = _r("all_region_identity_real_coord_permuted")
    ctrl_rows = [r for r in (shuf_r, rand_r, perm_r) if r]

    def _ok(a, b): return (a == a) and (b == b) and a > b + 0.005
    def _nan(v):   return v if (v == v) else float("nan")

    real_nll_g  = _v(real_r, "candidate_nll_gain_vs_base")
    real_wr_g   = _v(real_r, "within_real_region_nll_gain_vs_base")
    real_sr_p   = _v(real_r, "same_region_pair_acc")
    real_bdr    = _v(real_r, "benefit_damage_ratio")
    real_dam    = _v(real_r, "base_correct_damage_rate")

    ctrl_nll_max = max((_nan(_v(r, "candidate_nll_gain_vs_base")) for r in ctrl_rows), default=float("nan"))
    ctrl_sr_max  = max((_nan(_v(r, "same_region_pair_acc")) for r in ctrl_rows), default=float("nan"))
    ctrl_wr_max  = max((_nan(_v(r, "within_real_region_nll_gain_vs_base")) for r in ctrl_rows), default=float("nan"))

    q1_beats_tc      = _ok(real_nll_g, _v(stat_r, "candidate_nll_gain_vs_base"))
    q2_beats_home    = _ok(real_nll_g, _v(home_r, "candidate_nll_gain_vs_base"))
    q3_beats_ctrl    = _ok(real_nll_g, ctrl_nll_max)
    q4_sr_improves   = _ok(real_sr_p,  0.5)
    q5_wr_improves   = _ok(real_wr_g,  0.0)
    q6_base_safe     = (_nan(real_dam) == real_dam) and real_dam < 0.05
    q7_net_positive  = (real_r.get("net_correction", 0) or 0) > 0
    strong_success   = (q1_beats_tc and q2_beats_home and q3_beats_ctrl
                        and q4_sr_improves and q5_wr_improves and q6_base_safe)
    partial_success  = (q1_beats_tc or q2_beats_home or q3_beats_ctrl) and q7_net_positive

    if strong_success:
        recommendation = "PROCEED_TO_REGION_GENERATED_CANDIDATES"
    elif partial_success:
        recommendation = "PARTIAL_GO_IMPROVE_IDENTITY_RESOLVER"
    else:
        recommendation = "DO_NOT_PROCEED"

    lines = [
        "# Phase 2B-pre: All-Region Token Identity Resolver — Report",
        "",
        f"**selected_M:** {args.selected_M}  |  **steps:** {args.steps}  |  "
        f"**d_model:** {args.d_model}  |  **seed:** {args.seed}",
        "", "---", "",
        "## Comparison Table", "",
        "Primary metric: `candidate_nll_gain_vs_base` and `within_real_region_nll_gain_vs_base`",
        "",
    ]
    cols = ["variant", "selected_for_comparison",
            "candidate_nll_gain_vs_base", "within_real_region_nll_gain_vs_base",
            "same_region_pair_acc", "benefit_damage_ratio",
            "base_correct_damage_rate", "validation_score"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for row in comparison_rows:
        lines.append("| " + " | ".join(
            str(row.get(c, "")) if c in ("variant", "selected_for_comparison")
            else _fmt(row.get(c, float("nan"))) for c in cols) + " |")
    lines += ["", "---", "", "## Q&A", ""]

    def _q(n, q, ans, detail=""):
        lines.append(f"### Q{n}: {q}")
        lines.append(f"**{ans}**")
        if detail: lines.append(f"\n{detail}")
        lines.append("")

    _q(1, "Does all-region identity beat token/context-only scoring?",
       "YES" if q1_beats_tc else "NO",
       f"real nll_gain={_fmt(real_nll_g)}  token_context nll_gain={_fmt(_v(stat_r,'candidate_nll_gain_vs_base'))}")
    _q(2, "Does all-region identity beat home-region-only identity?",
       "YES" if q2_beats_home else "NO",
       f"real nll_gain={_fmt(real_nll_g)}  home_region nll_gain={_fmt(_v(home_r,'candidate_nll_gain_vs_base'))}")
    _q(3, "Does real all-region identity beat shuffled/random controls?",
       "YES" if q3_beats_ctrl else "NO",
       f"real={_fmt(real_nll_g)}  ctrl_max={_fmt(ctrl_nll_max)}")
    _q(4, "Does it improve same-region/superregion confuser resolution?",
       "YES" if q4_sr_improves else "NO",
       f"real same_region_pair_acc={_fmt(real_sr_p)}")
    _q(5, "Does it improve within-real-region token ranking?",
       "YES" if q5_wr_improves else "NO",
       f"real within_region_nll_gain={_fmt(real_wr_g)}")
    _q(6, "Does it preserve base-correct rows?",
       "YES" if q6_base_safe else "NO (DAMAGE HIGH)",
       f"real base_correct_damage_rate={_fmt(real_dam)}")
    _q(7, "Does it produce positive net corrections?",
       "YES" if q7_net_positive else "NO",
       f"real net_correction={real_r.get('net_correction', 0)}  "
       f"changed_to_gold={real_r.get('changed_to_gold', 0)}  "
       f"changed_away={real_r.get('changed_away', 0)}")
    _q(8, "Should we proceed to region-generated candidate proposals?",
       "YES" if strong_success else "PARTIAL" if partial_success else "NO", "")
    _q(9, "Should we proceed to full Phase 2B candidate coordinate mixer?",
       "YES" if strong_success else "NO", "")

    lines += [
        "---", "",
        "## PHASE 2B-pre ALL-REGION TOKEN IDENTITY VERDICT", "",
        "```",
    ]
    hdr = (f"  {'variant':44s}  nll_gain  wr_nll_g  sr_pair  "
           f"c2gold  c_away  bdr     dam     val")
    lines.append(hdr)
    lines.append("  " + "-" * len(hdr))
    for row in comparison_rows:
        lines.append(
            f"  {row['variant']:44s}"
            f"  {_fmt(row.get('candidate_nll_gain_vs_base', float('nan'))):8}"
            f"  {_fmt(row.get('within_real_region_nll_gain_vs_base', float('nan'))):8}"
            f"  {_fmt(row.get('same_region_pair_acc', float('nan'))):7}"
            f"  {str(row.get('changed_to_gold', 0)):6}"
            f"  {str(row.get('changed_away', 0)):6}"
            f"  {_fmt(row.get('benefit_damage_ratio', float('nan'))):6}"
            f"  {_fmt(row.get('base_correct_damage_rate', float('nan'))):6}"
            f"  {_fmt(row.get('validation_score', float('nan')))}"
        )
    lines += [f"\nrecommendation: {recommendation}", "```", ""]

    path = os.path.join(out_dir, "phase2B_pre_all_region_token_identity_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[save] {path}")
    return recommendation


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 2B-pre: All-Region Token Identity Resolver")
    p.add_argument("--train_dir",         required=True)
    p.add_argument("--val_dir",           required=True)
    p.add_argument("--small_ckpt",        required=True)
    p.add_argument("--phase2a_dir",       required=True)
    p.add_argument("--token_to_region",   required=True)
    p.add_argument("--super_map",         default=None)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--selected_M",        type=int,   default=64)
    p.add_argument("--d_model",           type=int,   default=256)
    p.add_argument("--hidden_dim",        type=int,   default=256)
    p.add_argument("--num_heads",         type=int,   default=4)
    p.add_argument("--dropout",           type=float, default=0.1)
    p.add_argument("--region_emb_dim",    type=int,   default=64)
    p.add_argument("--super_emb_dim",     type=int,   default=32)
    p.add_argument("--steps",             type=int,   default=5000)
    p.add_argument("--eval_every",        type=int,   default=500)
    p.add_argument("--batch_size",        type=int,   default=128)
    p.add_argument("--lr",                type=float, default=5e-5)
    p.add_argument("--margin",            type=float, default=0.5)
    p.add_argument("--lambda_ce",         type=float, default=1.0)
    p.add_argument("--lambda_pair",       type=float, default=1.0)
    p.add_argument("--lambda_same_region_pair",      type=float, default=2.0)
    p.add_argument("--lambda_same_superregion_pair", type=float, default=1.0)
    p.add_argument("--lambda_within_region",         type=float, default=1.0)
    p.add_argument("--lambda_preserve",   type=float, default=1.0)
    p.add_argument("--lambda_delta",      type=float, default=1e-3)
    p.add_argument("--lambda_gate",       type=float, default=1e-2)
    p.add_argument("--use_gate",          action="store_true", default=False)
    p.add_argument("--no_gate",           action="store_true", default=False)
    p.add_argument("--gate_init_bias",    type=float, default=-4.0)
    p.add_argument("--freeze_region_meaning", action="store_true", default=True)
    p.add_argument("--finetune_region_meaning", action="store_true", default=False)
    p.add_argument("--max_train_rows",    type=int,   default=None)
    p.add_argument("--max_val_rows",      type=int,   default=None)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--amp",               action="store_true")
    p.add_argument("--skip_coord_permuted", action="store_true",
                   help="Skip all_region_identity_real_coord_permuted variant")
    p.add_argument("--start_from_variant", default=None,
                   help="Skip all variants before this one (inclusive order) and load "
                        "their results from existing output_dir JSON/CSV files.")
    args = p.parse_args()

    # gate: --use_gate takes precedence; --no_gate disables
    if args.no_gate:
        args.use_gate = False
    elif not args.use_gate:
        # default: enable gate (spec says use_gate)
        args.use_gate = True

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    t0 = time.time()

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[step 1] Loading shards...")
    M          = args.selected_M
    train_data = load_shards(args.train_dir, M, args.max_train_rows, "train")
    val_data   = load_shards(args.val_dir,   M, args.max_val_rows,   "val")
    d_in       = max(train_data.get("d_model", 1), 1)

    # ── Unembedding ───────────────────────────────────────────────────────────
    print("\n[step 2] Loading unembedding...")
    U, uinfo = load_unembedding(args.small_ckpt)
    print(f"[unembedding] shape={tuple(U.shape)}")
    with open(os.path.join(args.output_dir, "unembedding_audit.json"), "w") as f:
        json.dump(uinfo, f, indent=2)

    # ── Region maps ───────────────────────────────────────────────────────────
    print("\n[step 3] Loading region maps...")
    tok_arr_real, reg_arr_real, n_regions, n_super = load_region_maps(
        args.token_to_region, args.super_map)

    def _load_json_map(path, base):
        d = json.load(open(path))
        out = base.copy()
        for k, v in d.items():
            i = int(k)
            if 0 <= i < len(out): out[i] = int(v)
        return out

    shuf_map_path = os.path.join(args.phase2a_dir, "shuffled_token_to_region.json")
    rand_map_path = os.path.join(args.phase2a_dir, "random_token_to_region.json")
    tok_arr_shuf  = _load_json_map(shuf_map_path, tok_arr_real)
    tok_arr_rand  = _load_json_map(rand_map_path, tok_arr_real)

    # ── Phase 2A config ───────────────────────────────────────────────────────
    cfg_path = os.path.join(args.phase2a_dir, "config.json")
    phase2a_cfg = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else {}
    phase2a_d_model = int(phase2a_cfg.get("d_model", 256))
    print(f"[phase2a] d_model={phase2a_d_model}  (region state dim)")

    # ── Load Phase 2A models ──────────────────────────────────────────────────
    print("\n[step 4] Loading Phase 2A models...")

    def _ckpt(variant_name): return os.path.join(args.phase2a_dir, f"best_{variant_name}.pt")

    phase2a_models = {}
    for p2a_name, tok_v in [
        ("contextual_region_meaning_real",     tok_arr_real),
        ("contextual_region_meaning_shuffled", tok_arr_shuf),
        ("contextual_region_meaning_random",   tok_arr_rand),
    ]:
        phase2a_models[p2a_name] = load_phase2a_model(
            _ckpt(p2a_name), tok_v, reg_arr_real, U,
            n_regions, n_super, d_in, phase2a_d_model, phase2a_cfg, device)

    # ── Coord permutation (fixed) ─────────────────────────────────────────────
    coord_perm_np = np.random.default_rng(args.seed + 999).permutation(n_regions)
    coord_perm_t  = torch.from_numpy(coord_perm_np).long().to(device)
    print(f"[coord_perm] generated permutation len={len(coord_perm_np)} (seed={args.seed+999})")

    # ── Train buckets ─────────────────────────────────────────────────────────
    print("\n[step 5] Computing training bucket labels...")
    train_buckets = compute_buckets(train_data, tok_arr_real, reg_arr_real, n_regions)

    # ── Config save ───────────────────────────────────────────────────────────
    cfg = {**vars(args),
           "n_regions": n_regions, "n_super": n_super, "d_in": d_in,
           "phase2a_d_model": phase2a_d_model,
           "train_rows": len(train_data["gold"]), "val_rows": len(val_data["gold"]),
           "device": str(device)}
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # ── Model factory ─────────────────────────────────────────────────────────
    def _make_resolver(mode, tok_v):
        return AllRegionTokenIdentityResolver(
            tok_arr_v=tok_v, reg_arr=reg_arr_real,
            U_frozen=U,
            n_regions=n_regions, n_super=n_super, d_in=d_in,
            d_model=args.d_model, hidden_dim=args.hidden_dim,
            num_heads=args.num_heads, dropout=args.dropout,
            region_emb_dim=args.region_emb_dim, super_emb_dim=args.super_emb_dim,
            region_state_dim=phase2a_d_model,
            use_gate=args.use_gate, gate_init_bias=args.gate_init_bias,
            mode=mode,
        ).to(device)

    # ── Variant definitions ───────────────────────────────────────────────────
    VARIANTS = [
        # (variant_name, resolver_mode, phase2a_key, tok_arr, coord_perm)
        ("base_only",                              None,           None,                                  tok_arr_real, None),
        ("logit_only_mlp",                         "logit_only",   None,                                  tok_arr_real, None),
        ("token_context_identity",                 "token_context",None,                                  tok_arr_real, None),
        ("home_region_identity_real",              "home_region",  "contextual_region_meaning_real",      tok_arr_real, None),
        ("all_region_identity_real",               "all_region",   "contextual_region_meaning_real",      tok_arr_real, None),
        ("all_region_identity_shuffled",           "all_region",   "contextual_region_meaning_shuffled",  tok_arr_shuf, None),
        ("all_region_identity_random",             "all_region",   "contextual_region_meaning_random",    tok_arr_rand, None),
    ]
    if not args.skip_coord_permuted:
        VARIANTS.append(
            ("all_region_identity_real_coord_permuted", "all_region",
             "contextual_region_meaning_real", tok_arr_real, coord_perm_t)
        )

    # ── Determine which variants to skip (load prior results) ─────────────────
    skip_set = set()
    if args.start_from_variant:
        variant_order = [v[0] for v in VARIANTS]
        if args.start_from_variant not in variant_order:
            raise ValueError(f"--start_from_variant '{args.start_from_variant}' not in variant list: "
                             f"{variant_order}")
        start_idx = variant_order.index(args.start_from_variant)
        skip_set  = set(variant_order[:start_idx])
        print(f"\n[resume] Skipping {len(skip_set)} variant(s): {sorted(skip_set)}")
        print(f"[resume] Starting from: {args.start_from_variant}")

    def _load_prior_csv_rows(csv_path, variant_filter=None):
        """Load rows from an existing CSV, optionally filtering by variant column."""
        if not os.path.isfile(csv_path):
            return []
        try:
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if variant_filter is not None:
                rows = [r for r in rows if r.get("variant") in variant_filter]
            return rows
        except Exception as e:
            print(f"  [WARN] could not load {csv_path}: {e}")
            return []

    out = args.output_dir

    # Pre-load prior results for skipped variants
    all_best_info  = {}
    all_final_info = {}
    all_comp_rows  = []
    all_slice_rows = []
    all_train_logs = []
    all_eval_logs  = []

    if skip_set:
        bm_path = os.path.join(out, "best_metrics.json")
        fm_path = os.path.join(out, "final_metrics.json")

        # Collect missing files / missing variant entries before bailing
        problems = []
        if not os.path.isfile(bm_path):
            problems.append(
                f"  best_metrics.json not found at: {bm_path}\n"
                f"  This file is written incrementally after each variant completes.\n"
                f"  If the previous run crashed before any variant finished, there is\n"
                f"  nothing to resume from — re-run without --start_from_variant.")
        if not os.path.isfile(fm_path):
            problems.append(f"  final_metrics.json not found at: {fm_path}")
        if problems:
            raise FileNotFoundError("\n".join(["[resume] Cannot load prior results:"] + problems))

        with open(bm_path) as f:
            prior_best = json.load(f)
        with open(fm_path) as f:
            prior_final = json.load(f)

        missing_variants = [v for v in skip_set if v not in prior_best]
        if missing_variants:
            raise KeyError(
                f"[resume] best_metrics.json exists but is missing these variants: "
                f"{missing_variants}\n"
                f"  Completed variants in file: {sorted(prior_best.keys())}\n"
                f"  Adjust --start_from_variant to match what actually completed.")
            all_best_info[v]  = prior_best[v]
            all_final_info[v] = prior_final.get(v, prior_best[v])
            all_comp_rows.append(build_comparison_row(all_best_info[v]))
            print(f"[resume] Loaded prior results for {v}: "
                  f"val_score={all_best_info[v].get('validation_score', float('nan')):.4f}")

        # Load prior slice rows, train/eval logs for skipped variants
        all_slice_rows = _load_prior_csv_rows(
            os.path.join(out, "slice_metrics.csv"), variant_filter=skip_set)
        all_train_logs = _load_prior_csv_rows(
            os.path.join(out, "train_log.csv"), variant_filter=skip_set)
        all_eval_logs  = _load_prior_csv_rows(
            os.path.join(out, "eval_log.csv"), variant_filter=skip_set)

    # ── Run ───────────────────────────────────────────────────────────────────
    for (variant, resolver_mode, p2a_key, tok_v, cperm) in VARIANTS:
        if variant in skip_set:
            print(f"\n[skip] {variant}  (loaded from prior output)")
            continue

        print(f"\n{'='*60}\n[variant] {variant}\n{'='*60}")

        p2a_model = phase2a_models.get(p2a_key) if p2a_key else None

        if resolver_mode is None:
            # base_only — evaluate base logits, no training
            print(f"[{variant}] base_only — evaluating base logits")
            val_bkts = compute_buckets(val_data, tok_arr_real, reg_arr_real, n_regions)
            vm, sr = eval_resolver(
                variant, None, None, val_data, val_bkts,
                tok_arr_real, reg_arr_real, n_regions,
                M, device, args.batch_size * 4)
            result = {
                "best":    {**vm, "variant": variant,
                            "no_improving_checkpoint_found": True,
                            "selected_for_comparison": "baseline",
                            "checkpoint_path": None, "best_step": None},
                "final":   {"variant": variant, **vm},
                "train_log": [], "eval_log": [{"step": 0, "variant": variant, **vm}],
                "slice_rows": sr,
            }
        else:
            resolver = _make_resolver(resolver_mode, tok_v)
            n_params = sum(p.numel() for p in resolver.parameters())
            print(f"[{variant}] mode={resolver_mode}  params={n_params:,}  "
                  f"id_dim={resolver.id_dim}")

            result = train_variant(
                variant, resolver, p2a_model,
                train_data, val_data, train_buckets,
                tok_arr_real, reg_arr_real, n_regions,
                M, args, device, args.output_dir, coord_perm=cperm)

            del resolver
            if device.type == "cuda":
                torch.cuda.empty_cache()

        bm = result["best"]; fm = result["final"]
        all_comp_rows.append(build_comparison_row(bm))
        all_train_logs.extend(result["train_log"])
        all_eval_logs.extend(result["eval_log"])
        all_best_info[variant]  = bm
        all_final_info[variant] = fm
        all_slice_rows.extend(result["slice_rows"])

        # ── Incremental save after every variant — crash-safe resume ──────────
        with open(os.path.join(out, "best_metrics.json"), "w") as f:
            json.dump(all_best_info, f, indent=2, default=str)
        with open(os.path.join(out, "final_metrics.json"), "w") as f:
            json.dump(all_final_info, f, indent=2, default=str)

        print(f"[{variant}]  nll_gain={bm.get('candidate_nll_gain_vs_base', 0):.4f}  "
              f"wr_nll_gain={bm.get('within_real_region_nll_gain_vs_base', float('nan')):.4f}  "
              f"sr_pair={bm.get('same_region_pair_acc', float('nan')):.4f}  "
              f"val_score={bm.get('validation_score', float('nan')):.4f}")

    # ── Save outputs (merge prior + new) ──────────────────────────────────────
    # Sort all_comp_rows to match VARIANTS order
    variant_order = [v[0] for v in VARIANTS]
    all_comp_rows.sort(key=lambda r: variant_order.index(r["variant"])
                       if r["variant"] in variant_order else 999)
    all_slice_rows_by_v = defaultdict(list)
    for r in all_slice_rows:
        all_slice_rows_by_v[r.get("variant","")].append(r)
    ordered_slice_rows = []
    for v in variant_order:
        ordered_slice_rows.extend(all_slice_rows_by_v.get(v, []))

    _wcsv(os.path.join(out, "phase2B_pre_comparison.csv"),  all_comp_rows)
    _wcsv(os.path.join(out, "slice_metrics.csv"),           ordered_slice_rows)
    _wcsv(os.path.join(out, "train_log.csv"),               all_train_logs)
    _wcsv(os.path.join(out, "eval_log.csv"),                all_eval_logs)

    with open(os.path.join(out, "best_metrics.json"),  "w") as f:
        json.dump(all_best_info,  f, indent=2, default=str)
    with open(os.path.join(out, "final_metrics.json"), "w") as f:
        json.dump(all_final_info, f, indent=2, default=str)

    # Coord diagnostics CSV
    coord_rows = [
        {"variant": v,
         "coord_entropy_mean":               all_best_info.get(v, {}).get("coord_entropy_mean", float("nan")),
         "coord_top1_is_home_region_rate":   all_best_info.get(v, {}).get("coord_top1_is_home_region_rate", float("nan")),
         "coord_adv_norm_mean":              all_best_info.get(v, {}).get("coord_adv_norm_mean", float("nan"))}
        for v in variant_order if v in all_best_info
    ]
    _wcsv(os.path.join(out, "coordinate_diagnostics.csv"), coord_rows)

    write_examples({}, out)
    recommendation = write_report(all_comp_rows, out, args)

    # ── Final console output ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(" PHASE 2B-pre ALL-REGION TOKEN IDENTITY VERDICT:")
    print(f"{'='*60}")
    print(f"  {'variant':44s}  nll_g   wr_g    sr_pair c2gold c_away  bdr    dam    val")
    print(f"  {'-'*110}")
    for row in all_comp_rows:
        print(
            f"  {row['variant']:44s}"
            f"  {_fmt(row.get('candidate_nll_gain_vs_base', float('nan'))):7}"
            f"  {_fmt(row.get('within_real_region_nll_gain_vs_base', float('nan'))):7}"
            f"  {_fmt(row.get('same_region_pair_acc', float('nan'))):7}"
            f"  {str(row.get('changed_to_gold', 0)):6}"
            f"  {str(row.get('changed_away', 0)):6}"
            f"  {_fmt(row.get('benefit_damage_ratio', float('nan'))):6}"
            f"  {_fmt(row.get('base_correct_damage_rate', float('nan'))):6}"
            f"  {_fmt(row.get('validation_score', float('nan')))}"
        )
    print(f"\n  recommendation: {recommendation}")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"  report:  {os.path.join(out, 'phase2B_pre_all_region_token_identity_report.md')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
