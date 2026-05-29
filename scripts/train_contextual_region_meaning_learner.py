#!/usr/bin/env python3
"""
train_contextual_region_meaning_learner.py — Phase 2A: Contextual Region Meaning Learner

Directly learns contextual region states Z(h) = [z_0(h), ..., z_{R-1}(h)].
Region states are trained with region-grounded objectives, NOT as candidate rerankers.

Primary objectives:
  L1 — Gold region prediction (region CE)
  L2 — Candidate scoring through region states (candidate CE)
  L3 — Within-true-region candidate ranking
  L4 — Confuser margin loss (same-region/superregion pair separation)
  L5 — Optional region-mass KL distillation

Variants:
  router_baseline              (no training, router recall only)
  static_region_meaning        (no h, learned embeddings only)
  contextual_region_meaning_real
  contextual_region_meaning_shuffled
  contextual_region_meaning_random

Phase 2A success → proceed to Phase 2B (candidate coordinate mixer).

Safety rules:
  Gold used only for targets/loss/metrics — never as model input.
  gold_region computed from tok_arr[gold_token] only for labeling.
  No gold force-inclusion.
  Candidate summaries use only base top-M candidates.
  Evaluation slices always define confusers with real tok_arr.
  Shuffled/random region CE measured under their own map.
  Fail loudly on NaN.
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

_EPS = 1e-9

# ── Unembedding key search ────────────────────────────────────────────────────
_U_EXACT = [
    "lm_head.weight","token_emb.weight","tok_emb.weight",
    "transformer.wte.weight","wte.weight",
    "model.lm_head.weight","model.token_emb.weight","model.tok_emb.weight",
    "model.transformer.wte.weight","embedding.weight","model.embed_tokens.weight",
]
_U_SUFFIX  = ["lm_head.weight","token_emb.weight","tok_emb.weight","wte.weight"]
_U_PREFIXES= ["module.","_orig_mod.","model.model.","model.","_model."]

# ── Shard aliases ─────────────────────────────────────────────────────────────
_TOPK = ["base_topk_ids","base_topk","topk_ids"]
_LGT  = ["base_topk_logits","base_topk_lgt","topk_lgt","topk_logits"]
_GOLD = ["gold_token","gold","labels"]
_RID  = ["row_id","row_ids"]
_RREG = ["router_topk_reg"]
_RPRB = ["router_topk_prb"]
_HCTX = ["h_ctx"]
_HRAW = ["h_raw","h"]


def _get(sh, aliases, required=True):
    for a in aliases:
        if a in sh:
            return sh[a]
    if required:
        raise KeyError(f"Need one of {aliases}; shard has {list(sh.keys())}")
    return None


def _nanmean(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(a))


def _wcsv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {path}")


def _fmt(v):
    if isinstance(v, bool):  return str(v)
    if isinstance(v, int):   return str(v)
    if isinstance(v, float): return f"{v:.5f}" if v == v else "nan"
    return str(v)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Shard loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_shards(shard_dir, selected_M, max_rows=None, label=""):
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")
    print(f"[shards] {label}: {len(paths)} shards in {shard_dir}")
    bufs = defaultdict(list)
    total = 0; first = True; d_model = 0
    for sp in paths:
        if max_rows and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print(f"[shards] keys: {list(sh.keys())}")
            first = False
        topk = _get(sh, _TOPK).long()
        lgt  = _get(sh, _LGT).float()
        gold = _get(sh, _GOLD).long()
        rids = _get(sh, ["row_id","row_ids"], required=False)
        rreg = _get(sh, _RREG, required=False)
        rprb = _get(sh, _RPRB, required=False)
        h_ctx = _get(sh, _HCTX, required=False)
        h_raw = _get(sh, _HRAW, required=False)
        h = h_ctx if h_ctx is not None else h_raw

        B, K = topk.shape
        if rids is None: rids = torch.arange(total, total + B)
        M = min(K, selected_M)
        topk = topk[:, :M]; lgt = lgt[:, :M]

        if max_rows and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, rids = topk[:keep], lgt[:keep], gold[:keep], rids[:keep]
            if rreg is not None: rreg = rreg[:keep]
            if rprb is not None: rprb = rprb[:keep]
            if h    is not None: h    = h[:keep]
            B = keep

        lgt = torch.where(torch.isfinite(lgt), lgt, torch.full_like(lgt, -1e9))
        bufs["topk_ids"].append(topk.numpy().astype(np.int32))
        bufs["topk_lgt"].append(lgt.numpy().astype(np.float32))
        bufs["gold"].append(gold.numpy().astype(np.int32))
        bufs["row_ids"].append(rids.numpy().astype(np.int32))

        if rreg is not None:
            K2 = rreg.shape[1]
            bufs["router_reg"].append(rreg.long().numpy().astype(np.int32))
            rp = rprb.float() if rprb is not None else torch.ones(B, K2) / K2
            bufs["router_prb"].append(rp.numpy().astype(np.float32))

        if h is not None:
            hf = h.float()
            if hf.dim() == 3: hf = hf[:, -1, :]
            d_model = hf.shape[-1]
            bufs["h"].append(hf.numpy().astype(np.float32))

        total += B

    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    data["has_router"] = "router_reg" in data
    data["has_h"]      = "h" in data
    data["d_model"]    = d_model
    data["M"]          = M
    print(f"[shards] {label}: {total:,} rows  M={M}  has_router={data['has_router']}  "
          f"has_h={data['has_h']}  d_model={d_model}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Unembedding
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_pfx(k):
    changed = True
    while changed:
        changed = False
        for pfx in _U_PREFIXES:
            if k.startswith(pfx):
                k = k[len(pfx):]; changed = True
    return k


def load_unembedding(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if not isinstance(sd, dict): sd = ckpt

    print(f"[unembedding] checkpoint keys ({len(sd)}):")
    for k in list(sd.keys())[:30]:
        v = sd[k]; shape = tuple(v.shape) if hasattr(v, "shape") else "?"
        print(f"  {k}: {shape}")
    if len(sd) > 30: print(f"  ... ({len(sd)-30} more)")

    tensors = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor) and v.dim() == 2}
    for raw_k, v in tensors.items():
        if _strip_pfx(raw_k) in _U_EXACT:
            info = {"unembedding_source": raw_k, "U_shape": list(v.shape), "stage": 1}
            print(f"[unembedding] exact match: {raw_k} {tuple(v.shape)}")
            return v.float(), info
    for raw_k, v in tensors.items():
        k = _strip_pfx(raw_k)
        for sfx in _U_SUFFIX:
            if k.endswith(sfx):
                info = {"unembedding_source": raw_k, "U_shape": list(v.shape), "stage": 2}
                print(f"[unembedding] suffix match: {raw_k} {tuple(v.shape)}")
                return v.float(), info
    for raw_k, v in tensors.items():
        for sfx in _U_SUFFIX:
            if raw_k.endswith(sfx):
                info = {"unembedding_source": raw_k, "U_shape": list(v.shape), "stage": 3}
                print(f"[unembedding] orig-key suffix: {raw_k} {tuple(v.shape)}")
                return v.float(), info
    raise RuntimeError(
        f"Cannot find unembedding in {ckpt_path}.\n"
        f"2D keys: {list(tensors.keys())}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Region maps + controls
# ═══════════════════════════════════════════════════════════════════════════════

def load_region_maps(t2r_path, super_path):
    if not t2r_path or not os.path.isfile(t2r_path):
        return np.zeros(50257, np.int32), np.zeros(2, np.int32), 1, 1
    with open(t2r_path) as f: raw = json.load(f)
    t2r = ({i: v for i, v in enumerate(raw) if v is not None}
           if isinstance(raw, list) else {int(k): v for k, v in raw.items()})
    n_regions = int(max(t2r.values())) + 1 if t2r else 1
    V = max(t2r.keys(), default=0) + 2
    tok_arr = np.full(V, n_regions, dtype=np.int32)  # unk = n_regions
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


def _region_sizes(tok_arr, n_regions):
    unk = n_regions
    valid = tok_arr[tok_arr < unk]
    return np.bincount(valid, minlength=n_regions).tolist()


def _shuffle_tok_arr(tok_arr, n_regions, seed):
    rng = np.random.default_rng(seed)
    valid = np.where(tok_arr < n_regions)[0]
    vals = tok_arr[valid].copy(); rng.shuffle(vals)
    out = tok_arr.copy(); out[valid] = vals
    return out


def _random_tok_arr(tok_arr, n_regions, seed):
    rng = np.random.default_rng(seed + 1)
    valid = np.where(tok_arr < n_regions)[0]
    vals  = tok_arr[valid]
    sizes = np.bincount(vals, minlength=n_regions)
    new   = np.repeat(np.arange(n_regions, dtype=np.int32), sizes)
    rng.shuffle(new)
    out = tok_arr.copy(); out[valid] = new
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Bucket labeling  (uses real tok_arr only)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bucket_labels(data, tok_arr, reg_arr, n_regions):
    gold     = data["gold"].astype(np.int64)
    topk_ids = data["topk_ids"].astype(np.int64)
    N, M     = topk_ids.shape
    Vt = len(tok_arr); Rr = len(reg_arr); unk_r = n_regions; unk_s = int(reg_arr.max())

    base_top1    = topk_ids[:, 0]
    base_correct = (base_top1 == gold)
    gip_M        = (topk_ids == gold[:, None]).any(axis=1)
    bucketA      = ~base_correct & gip_M

    base_reg = tok_arr[np.clip(base_top1, 0, Vt-1)]
    gold_reg = tok_arr[np.clip(gold,      0, Vt-1)]
    base_sreg = reg_arr[np.clip(base_reg, 0, Rr-1)]
    gold_sreg = reg_arr[np.clip(gold_reg, 0, Rr-1)]

    same_reg  = (base_reg == gold_reg)  & (gold_reg  < unk_r)
    same_sreg = (base_sreg == gold_sreg) & (gold_sreg < unk_s)
    same_reg_or_sreg = bucketA & (same_reg | same_sreg)

    print(f"[buckets] base_correct={base_correct.sum():,}  bucketA={bucketA.sum():,}  "
          f"gip_M={gip_M.sum():,}  same_reg_or_sreg={same_reg_or_sreg.sum():,}")
    return {
        "base_correct":          base_correct,
        "bucketA":               bucketA,
        "gip_M":                 gip_M,
        "same_reg_conf":         bucketA & same_reg,
        "same_sreg_conf":        bucketA & same_sreg,
        "same_reg_or_sreg_conf": same_reg_or_sreg,
        "diff_reg_conf":         bucketA & ~same_reg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model
# ═══════════════════════════════════════════════════════════════════════════════

class _PreLNBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_mult, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        nx = self.norm1(x)
        x  = x + self.attn(nx, nx, nx, need_weights=False)[0]
        x  = x + self.ff(self.norm2(x))
        return x


class ContextualRegionMeaningLearner(nn.Module):
    """
    Builds Z(h) = [z_0(h), ..., z_{R-1}(h)] — contextual region states.
    Candidate scoring flows entirely through region states.
    """
    def __init__(self, tok_arr, reg_arr, U_frozen, n_regions, n_super, d_in,
                 d_model, region_emb_dim, super_emb_dim,
                 num_region_layers, num_heads, ff_mult, dropout, use_context=True):
        super().__init__()
        self.register_buffer("tok_arr",  torch.from_numpy(tok_arr).int())
        self.register_buffer("reg_arr",  torch.from_numpy(reg_arr).int())
        self.register_buffer("U_frozen", U_frozen.float())

        self.R          = n_regions
        self.d_model    = d_model
        self.use_context = use_context
        token_dim       = U_frozen.shape[1]

        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim, padding_idx=n_regions)
        self.sreg_emb   = nn.Embedding(n_super   + 2, super_emb_dim,  padding_idx=n_super)
        self.region_id_emb = nn.Embedding(n_regions + 1, d_model)

        if use_context:
            self.h_proj = nn.Linear(max(d_in, 1), d_model)
        else:
            self.h_proj = None

        self.U_proj = nn.Linear(token_dim, d_model)

        # region_input_dim = region_emb_dim + super_emb_dim + d_model(h) + 9 scalars + d_model(pooled_U)
        self.region_input_dim = region_emb_dim + super_emb_dim + d_model + 9 + d_model
        self.region_proj = nn.Sequential(
            nn.Linear(self.region_input_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )

        self.region_layers = nn.ModuleList([
            _PreLNBlock(d_model, num_heads, ff_mult, dropout)
            for _ in range(num_region_layers)
        ])
        self.region_norm = nn.LayerNorm(d_model)

        self.region_head = nn.Linear(d_model, 1)   # [B, R, 1] → [B, R] logits

    # ── helpers ───────────────────────────────────────────────────────────────

    def _compute_U_proj(self, topk_ids):
        """Returns [B, M, d_model] float32."""
        V = self.U_frozen.shape[0]
        tok = topk_ids.clamp(0, V-1).long()
        raw = self.U_frozen[tok.reshape(-1)]            # [B*M, token_dim]
        return self.U_proj(raw).float().reshape(*tok.shape, self.d_model)

    def _build_region_tokens(self, B, R, topk_ids, topk_lgt, h_proj_out,
                              U_cand, router_reg, router_prb, device):
        """Build [B, R, d_model] region input tokens (all float32)."""
        Vt = self.tok_arr.shape[0]
        r_idx = torch.arange(R, device=device)

        # Static embeddings
        r_emb  = self.region_emb(r_idx)   # [R, region_emb_dim]
        sr_idx = self.reg_arr[:R].long().clamp(0, self.sreg_emb.num_embeddings - 1)
        sr_emb = self.sreg_emb(sr_idx)    # [R, super_emb_dim]
        r_emb_b  = r_emb.unsqueeze(0).expand(B, R, -1)
        sr_emb_b = sr_emb.unsqueeze(0).expand(B, R, -1)
        h_exp    = h_proj_out.unsqueeze(1).expand(B, R, -1)  # [B, R, d_model]

        # Router features [B, R]
        rp  = torch.zeros(B, R, device=device)
        rrk = torch.ones(B, R, device=device)
        rt1 = torch.zeros(B, R, device=device)
        rtk = torch.zeros(B, R, device=device)
        if router_reg is not None:
            K  = router_reg.shape[1]
            vr = router_reg.clamp(0, R-1).long()
            rprb_f = router_prb.float() if router_prb is not None else torch.ones(B, K, device=device) / K
            rp.scatter_add_(1, vr, rprb_f)
            rv = (torch.arange(K, device=device, dtype=torch.float32) / max(K-1, 1)).unsqueeze(0).expand(B, K)
            rrk.scatter_(1, vr, rv)
            rtk.scatter_(1, vr, torch.ones(B, K, device=device))
            rt1.scatter_(1, router_reg[:, 0:1].clamp(0, R-1).long(), torch.ones(B, 1, device=device))

        # Candidate summaries [B, R]
        M = topk_ids.shape[1]
        cand_reg = self.tok_arr[topk_ids.clamp(0, Vt-1).long()].long().clamp(0, R-1)  # [B, M]

        count_r = torch.zeros(B, R, device=device)
        count_r.scatter_add_(1, cand_reg, torch.ones(B, M, device=device))
        present_r = (count_r > 0).float()

        sum_lgt = torch.zeros(B, R, device=device)
        sum_lgt.scatter_add_(1, cand_reg, topk_lgt)
        mean_lgt = (sum_lgt / (count_r + _EPS)) * present_r

        # Max logit via masked one-hot
        cand_oh = (cand_reg.unsqueeze(2) == r_idx.view(1, 1, R))          # [B, M, R] bool
        masked  = torch.where(cand_oh, topk_lgt.unsqueeze(2),
                              torch.full_like(topk_lgt.unsqueeze(2), -1e9))
        max_lgt = masked.max(dim=1).values * present_r                     # [B, R]

        sm_lgt = torch.softmax(topk_lgt, dim=1)
        mass_r = torch.zeros(B, R, device=device)
        mass_r.scatter_add_(1, cand_reg, sm_lgt)

        # Pooled candidate token embedding (weighted by softmax mass)
        sm_U = sm_lgt.unsqueeze(-1) * U_cand                              # [B, M, d_model]
        idx3d = cand_reg.unsqueeze(-1).expand(B, M, self.d_model)
        pooled_U = torch.zeros(B, R, self.d_model, device=device)
        pooled_U.scatter_add_(1, idx3d, sm_U)
        pooled_U = (pooled_U / (mass_r.unsqueeze(-1) + _EPS)) * present_r.unsqueeze(-1)

        # Scalar stack [B, R, 9]
        scalars = torch.stack([
            count_r / max(M, 1), present_r,
            max_lgt / 10.0, mean_lgt / 10.0, mass_r,
            rp, rrk, rt1, rtk,
        ], dim=-1)

        region_input = torch.cat([r_emb_b, sr_emb_b, h_exp, scalars, pooled_U], dim=-1)
        # [B, R, region_input_dim]  all float32

        B_, R_, D_ = region_input.shape
        return self.region_proj(region_input.reshape(B_ * R_, D_)).reshape(B_, R_, self.d_model)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, topk_ids, topk_lgt, h=None, router_reg=None, router_prb=None, **kwargs):
        """
        Returns:
          region_states [B, R, d_model]
          region_logits [B, R]
          cand_scores   [B, M]
          cand_reg      [B, M]  region index of each candidate (model's own tok_arr)
        """
        B, M   = topk_ids.shape
        R      = self.R
        device = topk_ids.device
        topk_lgt = topk_lgt.float()

        # Context projection (float32)
        if self.use_context and self.h_proj is not None and h is not None:
            h_proj_out = self.h_proj(h.float()).float()
        else:
            h_proj_out = torch.zeros(B, self.d_model, device=device)

        # Candidate token embeddings (shared for region tokens + candidate scoring)
        U_cand = self._compute_U_proj(topk_ids)  # [B, M, d_model] float32

        # Build + encode region tokens
        region_tokens = self._build_region_tokens(
            B, R, topk_ids, topk_lgt, h_proj_out, U_cand, router_reg, router_prb, device)

        # Add region ID embeddings
        r_idx = torch.arange(R, device=device)
        region_tokens = region_tokens + self.region_id_emb(r_idx).unsqueeze(0)

        # Transformer over regions (set attention, no mask)
        Z = region_tokens.float()
        for layer in self.region_layers:
            Z = layer(Z)
        Z = self.region_norm(Z).float()          # [B, R, d_model]

        # Region logits
        region_logits = self.region_head(Z).squeeze(-1)  # [B, R]

        # Candidate region lookup (model's own tok_arr)
        Vt = self.tok_arr.shape[0]
        cand_reg = self.tok_arr[topk_ids.clamp(0, Vt-1).long()].long().clamp(0, R-1)  # [B, M]

        # Candidate scoring: dot(U_proj(U[c]), z_r_c) / sqrt(d_model)
        cand_z = Z[torch.arange(B, device=device).unsqueeze(1), cand_reg]  # [B, M, d_model]
        cand_scores = (U_cand * cand_z).sum(-1) / math.sqrt(self.d_model)  # [B, M]

        return Z, region_logits, cand_scores, cand_reg


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Loss
# ═══════════════════════════════════════════════════════════════════════════════

def compute_losses(region_logits, cand_scores, cand_reg_t,
                   gold_t, gold_region_t, topk_ids_t,
                   gip_mask, bucketA_mask, same_reg_mask, same_sreg_mask,
                   args, device):
    """
    All inputs are torch tensors on device.
    gold_region_t: [B] int64, -1 if unknown.
    """
    B = gold_t.shape[0]
    losses = {}

    # L1: Gold region CE
    valid_r = (gold_region_t >= 0)
    if valid_r.sum() > 0:
        losses["region_ce"] = F.cross_entropy(region_logits[valid_r], gold_region_t[valid_r])
    else:
        losses["region_ce"] = torch.zeros(1, device=device).squeeze()

    # L2: Candidate CE through region states
    gip = gip_mask
    if gip.sum() > 0:
        gold_idx = (topk_ids_t == gold_t.unsqueeze(1)).float().argmax(1)
        losses["candidate_ce"] = F.cross_entropy(cand_scores[gip], gold_idx[gip])
    else:
        losses["candidate_ce"] = torch.zeros(1, device=device).squeeze()

    # L3: Within-true-region CE (Python loop over batch)
    within_ces = []
    if gip.sum() > 0 and valid_r.sum() > 0:
        topk_np = topk_ids_t.cpu().long()
        gold_np = gold_t.cpu().long()
        greg_np = gold_region_t.cpu().long()
        creg_np = cand_reg_t.cpu().long()
        cs_cpu  = cand_scores.detach().cpu()
        for b in range(B):
            if not gip[b] or greg_np[b] < 0: continue
            gr = greg_np[b].item()
            in_r = (creg_np[b] == gr)           # [M] bool
            if in_r.sum() < 2: continue
            idxs = in_r.nonzero(as_tuple=False).squeeze(1)   # indices of cands in region
            # find gold among them
            gold_full = (topk_np[b] == gold_np[b]).nonzero(as_tuple=False)
            if len(gold_full) == 0: continue
            gf = gold_full[0, 0].item()
            if not in_r[gf]: continue
            gold_within = (idxs == gf).nonzero(as_tuple=False)
            if len(gold_within) == 0: continue
            gw = gold_within[0, 0].item()
            scores_r = cand_scores[b, idxs]
            target   = torch.tensor([gw], dtype=torch.long, device=device)
            within_ces.append(F.cross_entropy(scores_r.unsqueeze(0), target))
    if within_ces:
        losses["within_region"] = torch.stack(within_ces).mean()
    else:
        losses["within_region"] = torch.zeros(1, device=device).squeeze()

    # L4: Confuser margin loss
    confuser = bucketA_mask & (same_reg_mask | same_sreg_mask)
    if confuser.sum() > 0:
        gold_idx_c = (topk_ids_t[confuser] == gold_t[confuser].unsqueeze(1)).float().argmax(1)
        n_c = int(confuser.sum())
        s_gold = cand_scores[confuser][torch.arange(n_c, device=device), gold_idx_c]
        s_base = cand_scores[confuser][:, 0]
        losses["margin"] = F.relu(args.margin - (s_gold - s_base)).mean()
    else:
        losses["margin"] = torch.zeros(1, device=device).squeeze()

    # L5: Optional region-mass KL
    losses["region_kl"] = torch.zeros(1, device=device).squeeze()
    if args.lambda_region_kl > 0:
        sm_lgt = torch.softmax(
            torch.zeros(B, region_logits.shape[1], device=device), dim=1)  # placeholder
        pred_prob = torch.softmax(region_logits, dim=1)
        losses["region_kl"] = F.kl_div(
            pred_prob.log(), sm_lgt, reduction="batchmean")

    # L6: L2 regularization handled outside

    total = (args.lambda_region_ce    * losses["region_ce"]
           + args.lambda_candidate_ce * losses["candidate_ce"]
           + args.lambda_within_region * losses["within_region"]
           + args.lambda_margin        * losses["margin"]
           + args.lambda_region_kl     * losses["region_kl"])

    return total, {k: v.item() for k, v in losses.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Batch helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_batch(data, idx, M):
    d_in = max(data.get("d_model", 1), 1)
    b = {
        "topk_ids": data["topk_ids"][idx],
        "topk_lgt": data["topk_lgt"][idx],
        "h": (data["h"][idx] if data["has_h"] else np.zeros((len(idx), d_in), np.float32)),
    }
    if data["has_router"]:
        b["router_reg"] = data["router_reg"][idx]
        b["router_prb"] = data["router_prb"][idx]
    return b


def _to_device(batch, device):
    return {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
            for k, v in batch.items()}


def _gold_regions_batch(data, idx, tok_arr, n_regions):
    """Compute gold_region for a batch (under given tok_arr), -1 if unknown."""
    gold = data["gold"][idx].astype(np.int64)
    Vt = len(tok_arr)
    gr = tok_arr[np.clip(gold, 0, Vt-1)].astype(np.int64)
    gr[gr >= n_regions] = -1   # unknown
    return gr


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Balanced sampler (uses real buckets)
# ═══════════════════════════════════════════════════════════════════════════════

def make_balanced_sampler(buckets, n_steps, batch_size, seed):
    rng = np.random.default_rng(seed)
    ra  = np.where(buckets["bucketA"])[0]
    rb  = np.where(buckets["base_correct"])[0]
    rc  = np.where(buckets["same_reg_or_sreg_conf"])[0]
    N   = len(buckets["base_correct"])
    all_= np.arange(N)
    na  = max(1, int(batch_size * 0.40))
    nb  = max(1, int(batch_size * 0.30))
    nc  = max(1, int(batch_size * 0.20))
    nz  = batch_size - na - nb - nc

    def samp(arr, n):
        if len(arr) == 0: return rng.choice(all_, n, replace=True)
        return rng.choice(arr, n, replace=len(arr) < n)

    return [np.concatenate([samp(ra,na), samp(rb,nb), samp(rc,nc), samp(all_,nz)])
            for _ in range(n_steps)]


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Router baseline
# ═══════════════════════════════════════════════════════════════════════════════

def eval_router_baseline(data, tok_arr, n_regions, batch_size=2048):
    if not data["has_router"]:
        return {"router_available": False,
                "gold_region_recall@1": float("nan"),
                "gold_region_recall@4": float("nan"),
                "gold_region_recall@8": float("nan"),
                "gold_region_recall@16": float("nan")}

    gold = data["gold"].astype(np.int64)
    Vt = len(tok_arr)
    gold_reg = tok_arr[np.clip(gold, 0, Vt-1)].astype(np.int64)
    valid = gold_reg < n_regions

    router_reg = data["router_reg"].astype(np.int64)   # [N, K]
    N = len(gold)

    recall = {1: [], 4: [], 8: [], 16: []}
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        gr = gold_reg[s:e]; vm = valid[s:e]
        rr = router_reg[s:e]
        if vm.sum() == 0: continue
        gr = gr[vm]; rr = rr[vm]
        for k in recall:
            recall[k].append(((rr[:, :k] == gr[:, None]).any(axis=1)).mean())

    return {
        "router_available": True,
        "gold_region_recall@1":  float(np.mean(recall[1])) if recall[1] else float("nan"),
        "gold_region_recall@4":  float(np.mean(recall[4])) if recall[4] else float("nan"),
        "gold_region_recall@8":  float(np.mean(recall[8])) if recall[8] else float("nan"),
        "gold_region_recall@16": float(np.mean(recall[16])) if recall[16] else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def eval_model(model, data, tok_arr_model, reg_arr_model, n_regions_model,
               tok_arr_real, reg_arr_real, n_regions_real,
               M, args, device, batch_size=512):
    """
    tok_arr_model: the model's own map (shuffled for shuffled variant).
    tok_arr_real:  always the real map, used for confuser / candidate-vs-base comparisons.
    Region CE and within-region metrics computed under model's map.
    Confuser and candidate-vs-real-base metrics computed under real map.
    """
    N    = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk_ids"].astype(np.int64)
    lgt  = data["topk_lgt"].astype(np.float32)

    Vt_m = len(tok_arr_model); Vt_r = len(tok_arr_real)
    Rr_m = len(reg_arr_model); Rr_r = len(reg_arr_real)
    unk_m = n_regions_model;   unk_r = n_regions_real

    # Pre-compute gold regions under both maps
    gold_reg_m = tok_arr_model[np.clip(gold, 0, Vt_m-1)].astype(np.int64)
    gold_reg_m[gold_reg_m >= unk_m] = -1
    gold_reg_r = tok_arr_real[np.clip(gold, 0, Vt_r-1)].astype(np.int64)
    gold_reg_r[gold_reg_r >= unk_r] = -1

    # Pre-compute confuser masks (real map)
    base_top1  = topk[:, 0]
    base_reg_r = tok_arr_real[np.clip(base_top1, 0, Vt_r-1)].astype(np.int64)
    base_sreg_r= reg_arr_real[np.clip(base_reg_r, 0, Rr_r-1)].astype(np.int64)
    gold_sreg_r= reg_arr_real[np.clip(np.clip(gold_reg_r, 0, Rr_r-1), 0, Rr_r-1)].astype(np.int64)

    base_correct= (base_top1 == gold)
    gip         = (topk == gold[:, None]).any(axis=1)
    gold_rank_a = np.where(gip, (topk == gold[:, None]).argmax(axis=1), -1)
    same_reg_r  = (base_reg_r == gold_reg_r) & (gold_reg_r >= 0)
    same_sreg_r = (base_sreg_r == gold_sreg_r) & (gold_sreg_r >= 0) & (base_sreg_r < int(reg_arr_real.max()))
    bucketA     = ~base_correct & gip

    # Accumulation buffers
    r_logits_all = np.empty((N, n_regions_model), dtype=np.float32)
    c_scores_all = np.empty((N, M), dtype=np.float32)
    c_reg_all    = np.empty((N, M), dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e   = min(s + batch_size, N)
            idx = np.arange(s, e)
            b   = _to_device(_make_batch(data, idx, M), device)
            Z, rl, cs, cr = model(**b)
            r_logits_all[s:e] = rl.cpu().float().numpy()
            c_scores_all[s:e] = cs.cpu().float().numpy()
            c_reg_all[s:e]    = cr.cpu().numpy().astype(np.int32)

    ar = np.arange(N)

    # ── Region prediction metrics ─────────────────────────────────────────────
    pred_region = r_logits_all.argmax(axis=1)
    valid_r = (gold_reg_m >= 0)
    def _recall(k):
        if valid_r.sum() == 0: return float("nan")
        top_k = np.argsort(r_logits_all[valid_r], axis=1)[:, -k:]
        return float((top_k == gold_reg_m[valid_r, None]).any(axis=1).mean())

    region_acc1 = float((pred_region[valid_r] == gold_reg_m[valid_r]).mean()) if valid_r.sum() > 0 else float("nan")
    region_r4   = _recall(4)
    region_r8   = _recall(8)
    region_r16  = _recall(16)

    # Region CE
    if valid_r.sum() > 0:
        lsm = r_logits_all[valid_r].astype(np.float64)
        lsm -= lsm.max(axis=1, keepdims=True)
        logsm = lsm - np.log(np.exp(lsm).sum(axis=1, keepdims=True) + _EPS)
        region_ce = float(-logsm[np.arange(valid_r.sum()), gold_reg_m[valid_r]].mean())
    else:
        region_ce = float("nan")

    # Region entropy / margin
    r_sm = np.exp(r_logits_all - r_logits_all.max(axis=1, keepdims=True))
    r_sm = r_sm / (r_sm.sum(axis=1, keepdims=True) + _EPS)
    region_ent = float(_nanmean(-(r_sm * np.log(r_sm + _EPS)).sum(axis=1)))
    top2_r = np.sort(r_logits_all, axis=1)[:, -2:]
    region_margin = float(_nanmean(top2_r[:, 1] - top2_r[:, 0]))

    # ── Candidate-via-region metrics ──────────────────────────────────────────
    pred_cand_rank = c_scores_all.argmax(axis=1)
    pred_cand_id   = topk[ar, pred_cand_rank]
    gold_idx_full  = np.where(gip, (topk == gold[:, None]).argmax(axis=1), 0)

    # NLL given gold in topM
    cs_d = c_scores_all.astype(np.float64)
    cs_d -= cs_d.max(axis=1, keepdims=True)
    cs_logsm = cs_d - np.log(np.exp(cs_d).sum(axis=1, keepdims=True) + _EPS)
    nll_arr = np.full(N, float("nan"))
    if gip.sum() > 0:
        nll_arr[gip] = -cs_logsm[gip, np.clip(gold_idx_full[gip], 0, M-1)]
    cand_nll = _nanmean(nll_arr[gip])

    cand_acc = float((pred_cand_id[gip] == gold[gip]).mean()) if gip.sum() > 0 else float("nan")

    # Base NLL for comparison
    bd = lgt.astype(np.float64)
    bd -= bd.max(axis=1, keepdims=True)
    b_logsm = bd - np.log(np.exp(bd).sum(axis=1, keepdims=True) + _EPS)
    base_nll_arr = np.full(N, float("nan"))
    if gip.sum() > 0:
        base_nll_arr[gip] = -b_logsm[gip, np.clip(gold_idx_full[gip], 0, M-1)]
    base_nll = _nanmean(base_nll_arr[gip])
    base_acc = float(((topk[:, 0])[gip] == gold[gip]).mean()) if gip.sum() > 0 else float("nan")

    # ── Within-region metrics ─────────────────────────────────────────────────
    wr_accs = []; wr_nlls = []; wr_ranks = []
    for n in range(N):
        if not gip[n] or gold_reg_m[n] < 0: continue
        gr = gold_reg_m[n]
        in_r = (c_reg_all[n] == gr)
        if in_r.sum() < 2: continue
        idxs = np.where(in_r)[0]
        # gold's position in full topk
        gold_full_matches = np.where(topk[n] == gold[n])[0]
        if len(gold_full_matches) == 0: continue
        gf = gold_full_matches[0]
        in_r_mask = np.isin(idxs, [gf])
        if not in_r_mask.any(): continue
        gold_within_pos = np.where(idxs == gf)[0]
        if len(gold_within_pos) == 0: continue
        gw = gold_within_pos[0]
        scores_r = c_scores_all[n, idxs]
        sc = scores_r.astype(np.float64)
        sc -= sc.max()
        ls  = sc - np.log(np.exp(sc).sum() + _EPS)
        wr_nlls.append(-ls[gw])
        wr_accs.append(float(scores_r.argmax() == gw))
        wr_ranks.append(int((scores_r > scores_r[gw]).sum()))

    within_nll  = _nanmean(wr_nlls)
    within_acc  = _nanmean(wr_accs)
    within_rank = _nanmean(wr_ranks)

    # ── Confuser pair accuracy ────────────────────────────────────────────────
    def _pair_acc(mask):
        m = mask & gip
        if m.sum() == 0: return float("nan")
        gi = gold_idx_full[m]
        n_m = m.sum()
        s_gold = c_scores_all[m][np.arange(n_m), np.clip(gi, 0, M-1)]
        s_base = c_scores_all[m][:, 0]
        return float((s_gold > s_base).mean())

    same_reg_c  = bucketA & same_reg_r
    same_sreg_c = bucketA & same_sreg_r
    diff_reg_c  = bucketA & ~same_reg_r

    bucketA_pair_acc      = _pair_acc(bucketA)
    same_reg_pair_acc     = _pair_acc(same_reg_c)
    same_sreg_pair_acc    = _pair_acc(same_sreg_c)
    diff_reg_pair_acc     = _pair_acc(diff_reg_c)

    # ── Validation score ──────────────────────────────────────────────────────
    def _safe(v): return v if v == v else 0.0
    validation_score = (_safe(region_r8)
                        + 0.5 * _safe(same_reg_pair_acc)
                        + 0.5 * _safe(within_acc))

    return {
        "region_ce":               region_ce,
        "gold_region_acc@1":       region_acc1,
        "gold_region_recall@4":    region_r4,
        "gold_region_recall@8":    region_r8,
        "gold_region_recall@16":   region_r16,
        "region_entropy":          region_ent,
        "region_margin":           region_margin,
        "candidate_via_region_nll_given_gold_in_topM": float(cand_nll),
        "candidate_via_region_acc_given_gold_in_topM": float(cand_acc),
        "natural_gold_in_topM_rate": float(gip.mean()),
        "base_candidate_nll_given_gold_in_topM": float(base_nll),
        "base_candidate_acc_given_gold_in_topM": float(base_acc),
        "candidate_nll_gain_vs_base": float(base_nll - cand_nll) if (base_nll == base_nll and cand_nll == cand_nll) else float("nan"),
        "within_true_region_nll":  float(within_nll),
        "within_true_region_acc":  float(within_acc),
        "mean_gold_rank_inside_true_region": float(within_rank),
        "bucketA_pair_acc":           bucketA_pair_acc,
        "same_region_confuser_pair_acc": same_reg_pair_acc,
        "same_superregion_confuser_pair_acc": same_sreg_pair_acc,
        "different_region_confuser_pair_acc": diff_reg_pair_acc,
        "validation_score":        float(validation_score),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_variant(variant, model,
                  train_data, val_data,
                  tok_arr_model, reg_arr_model, n_regions_model,
                  tok_arr_real,  reg_arr_real,  n_regions_real,
                  buckets, M, args, device, out_dir):

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    scaler    = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    sampler   = make_balanced_sampler(buckets, args.steps, args.batch_size, args.seed)

    Vt_m = len(tok_arr_model); Vt_r = len(tok_arr_real)

    # Initial validation score (step 0)
    model.eval()
    init_vm = eval_model(model, val_data, tok_arr_model, reg_arr_model, n_regions_model,
                         tok_arr_real, reg_arr_real, n_regions_real, M, args, device)
    init_score = init_vm["validation_score"]
    print(f"[{variant}] initial validation_score={init_score:.5f}  "
          f"recall@8={init_vm['gold_region_recall@8']:.4f}")

    best_score = init_score
    best_ckpt  = None; best_vm = None; best_step = None
    train_log  = []; eval_log = []

    model.train()
    t0 = time.time()

    for step, idx in enumerate(sampler):
        batch    = _to_device(_make_batch(train_data, idx, M), device)
        gold_t   = torch.from_numpy(train_data["gold"][idx].astype(np.int64)).to(device)
        gold_reg = _gold_regions_batch(train_data, idx, tok_arr_model, n_regions_model)
        gold_reg_t = torch.from_numpy(gold_reg).to(device)

        # Confuser masks (real map) for margin loss
        topk_np  = train_data["topk_ids"][idx].astype(np.int64)
        base_top1= topk_np[:, 0]
        base_r   = tok_arr_real[np.clip(base_top1, 0, Vt_r-1)].astype(np.int64)
        gold_np  = train_data["gold"][idx].astype(np.int64)
        gold_r   = tok_arr_real[np.clip(gold_np, 0, Vt_r-1)].astype(np.int64)
        base_sreg= reg_arr_real[np.clip(base_r, 0, len(reg_arr_real)-1)].astype(np.int64)
        gold_sreg= reg_arr_real[np.clip(gold_r, 0, len(reg_arr_real)-1)].astype(np.int64)
        same_reg_m  = torch.from_numpy((base_r == gold_r) & (gold_r < n_regions_real)).to(device)
        same_sreg_m = torch.from_numpy((base_sreg == gold_sreg) & (gold_sreg < int(reg_arr_real.max()))).to(device)
        gip_m    = torch.from_numpy((topk_np == gold_np[:, None]).any(axis=1)).to(device)
        bc_mask  = torch.from_numpy(buckets["base_correct"][idx]).to(device)
        bucketA_m= torch.from_numpy(buckets["bucketA"][idx]).to(device)
        topk_t   = batch["topk_ids"].long()

        optimizer.zero_grad()

        Z, rl, cs, cr = model(**batch)

        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        total, ld = compute_losses(
            rl, cs, cr, gold_t, gold_reg_t, topk_t,
            gip_m, bucketA_m, same_reg_m, same_sreg_m, args, device)
        total = total + args.lambda_l2 * l2_reg

        if torch.isnan(total):
            raise RuntimeError(f"[{variant}] NaN loss at step {step}: {ld}")

        if scaler:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_log.append({"step": step+1, "variant": variant,
                          **{f"loss_{k}": v for k, v in ld.items()},
                          "loss_total": total.item()})

        if (step + 1) % args.eval_every == 0 or step == 0:
            model.eval()
            vm = eval_model(model, val_data, tok_arr_model, reg_arr_model, n_regions_model,
                            tok_arr_real, reg_arr_real, n_regions_real, M, args, device)
            model.train()
            vs = vm["validation_score"]
            elapsed = time.time() - t0
            print(f"[{variant}] step={step+1:5d}  loss={ld['region_ce']:.4f}/"
                  f"{ld['candidate_ce']:.4f}/{ld['within_region']:.4f}  "
                  f"recall@8={vm['gold_region_recall@8']:.4f}  "
                  f"conf_pair={vm['same_region_confuser_pair_acc']:.4f}  "
                  f"val_score={vs:.4f}  {elapsed:.0f}s")
            eval_log.append({"step": step+1, "variant": variant, **vm})
            if vs > best_score:
                best_score = vs; best_ckpt = deepcopy(model.state_dict())
                best_vm = deepcopy(vm); best_step = step + 1
                print(f"  → new best: val_score={vs:.5f}")

    model.eval()
    final_vm = eval_model(model, val_data, tok_arr_model, reg_arr_model, n_regions_model,
                          tok_arr_real, reg_arr_real, n_regions_real, M, args, device)
    eval_log.append({"step": args.steps, "variant": variant, **final_vm})

    ckpt_path = None
    if best_ckpt is not None:
        ckpt_path = os.path.join(out_dir, f"best_{variant}.pt")
        torch.save({"state_dict": best_ckpt, "variant": variant, "val_score": best_score}, ckpt_path)
        print(f"[{variant}] saved best: {ckpt_path}  score={best_score:.5f}")
    else:
        print(f"[{variant}] no checkpoint beat initial score={init_score:.5f}")

    best_info = {
        "variant": variant,
        "no_improving_checkpoint_found": best_ckpt is None,
        "selected_for_comparison": "best" if best_ckpt is not None else "final_no_best",
        "checkpoint_path": ckpt_path,
        "best_step": best_step,
        **(best_vm if best_vm is not None else final_vm),
    }
    final_info = {"variant": variant, "final_step": args.steps, **final_vm}
    return {"best": best_info, "final": final_info, "train_log": train_log, "eval_log": eval_log}


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Comparison table + report
# ═══════════════════════════════════════════════════════════════════════════════

def build_comparison_row(variant, vm, selected_for_comparison="baseline", no_best=False):
    def _v(k): return vm.get(k, float("nan"))
    return {
        "variant":                             variant,
        "selected_for_comparison":             selected_for_comparison,
        "no_improving_checkpoint_found":       no_best,
        "gold_region_recall@1":                _v("gold_region_acc@1"),
        "gold_region_recall@4":                _v("gold_region_recall@4"),
        "gold_region_recall@8":                _v("gold_region_recall@8"),
        "gold_region_recall@16":               _v("gold_region_recall@16"),
        "region_ce":                           _v("region_ce"),
        "candidate_via_region_nll":            _v("candidate_via_region_nll_given_gold_in_topM"),
        "candidate_via_region_acc":            _v("candidate_via_region_acc_given_gold_in_topM"),
        "candidate_nll_gain_vs_base":          _v("candidate_nll_gain_vs_base"),
        "within_true_region_acc":              _v("within_true_region_acc"),
        "within_true_region_nll":              _v("within_true_region_nll"),
        "bucketA_pair_acc":                    _v("bucketA_pair_acc"),
        "same_region_confuser_pair_acc":       _v("same_region_confuser_pair_acc"),
        "same_superregion_confuser_pair_acc":  _v("same_superregion_confuser_pair_acc"),
        "validation_score":                    _v("validation_score"),
    }


def write_report(args, comparison, out_dir):
    def _v(r, k): return r.get(k, float("nan"))

    ctx_real   = next((r for r in comparison if r["variant"] == "contextual_region_meaning_real"), None)
    static_row = next((r for r in comparison if r["variant"] == "static_region_meaning"), None)
    router_row = next((r for r in comparison if r["variant"] == "router_baseline"), None)
    shuf_rows  = [r for r in comparison if "shuffled" in r["variant"] or "random" in r["variant"]]

    def _nll(r): return _v(r, "candidate_via_region_nll") if r else float("nan")
    def _r8(r):  return _v(r, "gold_region_recall@8") if r else float("nan")
    def _conf(r): return _v(r, "same_region_confuser_pair_acc") if r else float("nan")
    def _wr(r):  return _v(r, "within_true_region_acc") if r else float("nan")

    ctx_r8   = _r8(ctx_real)
    static_r8 = _r8(static_row)
    router_r8 = _r8(router_row) if router_row else float("nan")
    shuf_r8   = max((_r8(r) for r in shuf_rows), default=float("nan"))
    ctx_conf  = _conf(ctx_real)
    shuf_conf = max((_conf(r) for r in shuf_rows), default=float("nan"))
    ctx_wr    = _wr(ctx_real)
    shuf_wr   = max((_wr(r) for r in shuf_rows), default=float("nan"))

    def _ok(a, b): return a == a and b == b and a > b + 0.001

    beats_static_r8   = _ok(ctx_r8, static_r8)
    beats_shuf_conf   = _ok(ctx_conf, shuf_conf)
    beats_shuf_wr     = _ok(ctx_wr,  shuf_wr)
    ctx_shuf_nll_gain = _nll(ctx_real) < min((_nll(r) for r in shuf_rows), default=float("inf")) - 0.001 if shuf_rows else True
    within_nontrivial = ctx_wr == ctx_wr and ctx_wr > 0.1

    proceed = beats_static_r8 and beats_shuf_conf and within_nontrivial
    verdict = "PROCEED_TO_PHASE_2B" if proceed else "DO_NOT_PROCEED_TO_PHASE_2B"

    rpt = os.path.join(out_dir, "phase2A_region_meaning_report.md")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("# Phase 2A: Contextual Region Meaning Learner — Report\n\n")
        f.write(f"**selected_M:** {args.selected_M}  |  **steps:** {args.steps}  |  "
                f"**seed:** {args.seed}  |  **d_model:** {args.d_model}\n\n---\n\n")

        f.write("## Comparison Table\n\n")
        cols = ["variant","selected_for_comparison","gold_region_recall@8","region_ce",
                "candidate_via_region_nll","candidate_nll_gain_vs_base",
                "within_true_region_acc","same_region_confuser_pair_acc","validation_score"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for row in comparison:
            f.write("| " + " | ".join(
                _fmt(_v(row, c)) if c not in ("variant","selected_for_comparison")
                else str(row.get(c,"")) for c in cols) + " |\n")

        f.write("\n---\n\n## Q&A\n\n")

        def _q(n, q, ans, detail=""):
            f.write(f"### Q{n}: {q}\n\n**{ans}**\n\n{detail}\n\n")

        _q(1, "Can contextual region states predict the gold region?",
           "YES" if ctx_r8 > 0.5 else "PARTIALLY" if ctx_r8 > 0.2 else "NO",
           f"ctx recall@8={_fmt(ctx_r8)}  router recall@8={_fmt(router_r8)}")
        _q(2, "Do contextual region states beat static region states?",
           "YES" if beats_static_r8 else "NO",
           f"ctx recall@8={_fmt(ctx_r8)}  static recall@8={_fmt(static_r8)}")
        _q(3, "Do real region meanings beat shuffled/random?",
           "YES" if beats_shuf_conf else "MIXED",
           f"real conf_pair={_fmt(ctx_conf)}  shuf conf_pair={_fmt(shuf_conf)}")
        _q(4, "Can z_gold_region rank the gold token inside its own region?",
           "YES" if within_nontrivial else "NO",
           f"within_true_region_acc={_fmt(ctx_wr)}")
        _q(5, "Can region meanings separate same-region/superregion confusers?",
           "YES" if beats_shuf_conf else "NO",
           f"real same_reg_pair_acc={_fmt(ctx_conf)}  shuf={_fmt(shuf_conf)}")
        _q(6, "Are region meanings useful beyond router probabilities?",
           "YES" if (ctx_r8 > router_r8 + 0.01 or ctx_conf > 0.6) else "UNCLEAR",
           f"ctx recall@8={_fmt(ctx_r8)}  router recall@8={_fmt(router_r8)}")
        _q(7, "Should we proceed to Phase 2B candidate coordinate mixer?",
           verdict,
           "" if proceed else "Failure reasons: " + ", ".join(
               (["ctx_does_not_beat_static_on_recall8"] if not beats_static_r8 else []) +
               (["ctx_does_not_beat_shuffled_on_confuser"] if not beats_shuf_conf else []) +
               (["within_region_acc_not_nontrivial"] if not within_nontrivial else [])))

        f.write("---\n\n## PHASE 2A VERDICT\n\n```\n")
        for row in comparison:
            f.write(f"{row['variant']:40s} "
                    f"recall@8={_fmt(_v(row,'gold_region_recall@8'))}  "
                    f"conf={_fmt(_v(row,'same_region_confuser_pair_acc'))}  "
                    f"wr_acc={_fmt(_v(row,'within_true_region_acc'))}  "
                    f"val={_fmt(_v(row,'validation_score'))}\n")
        f.write(f"\nrecommendation: {verdict}\n```\n")

    print(f"[save] {rpt}")
    return rpt


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 2A: Contextual Region Meaning Learner")
    p.add_argument("--train_dir",       required=True)
    p.add_argument("--val_dir",         required=True)
    p.add_argument("--small_ckpt",      required=True)
    p.add_argument("--token_to_region", required=True)
    p.add_argument("--super_map",       default=None)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--selected_M",      type=int,   default=64)
    p.add_argument("--d_model",         type=int,   default=256)
    p.add_argument("--region_emb_dim",  type=int,   default=64)
    p.add_argument("--super_emb_dim",   type=int,   default=32)
    p.add_argument("--num_region_layers",type=int,  default=2)
    p.add_argument("--num_heads",       type=int,   default=4)
    p.add_argument("--ff_mult",         type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--steps",           type=int,   default=5000)
    p.add_argument("--eval_every",      type=int,   default=500)
    p.add_argument("--batch_size",      type=int,   default=128)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--lambda_region_ce",     type=float, default=1.0)
    p.add_argument("--lambda_candidate_ce",  type=float, default=0.5)
    p.add_argument("--lambda_within_region", type=float, default=1.0)
    p.add_argument("--lambda_margin",        type=float, default=0.5)
    p.add_argument("--lambda_region_kl",     type=float, default=0.0)
    p.add_argument("--lambda_l2",            type=float, default=1e-5)
    p.add_argument("--margin",          type=float, default=0.5)
    p.add_argument("--max_train_rows",  type=int,   default=None)
    p.add_argument("--max_val_rows",    type=int,   default=None)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--skip_random_control", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    t0 = time.time()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[step 1] loading shards...")
    M          = args.selected_M
    train_data = load_shards(args.train_dir, M, args.max_train_rows, "train")
    val_data   = load_shards(args.val_dir,   M, args.max_val_rows,   "val")
    d_in       = max(train_data.get("d_model", 1), 1)
    has_router = train_data["has_router"]

    # ── Unembedding ───────────────────────────────────────────────────────────
    print("\n[step 2] loading unembedding...")
    U, uinfo = load_unembedding(args.small_ckpt)
    print(f"[unembedding] shape={tuple(U.shape)}")
    with open(os.path.join(args.output_dir, "unembedding_audit.json"), "w") as f:
        json.dump(uinfo, f, indent=2)

    # ── Region maps ───────────────────────────────────────────────────────────
    print("\n[step 3] region maps...")
    tok_arr, reg_arr, n_regions, n_super = load_region_maps(
        args.token_to_region, args.super_map)
    shuf_tok = _shuffle_tok_arr(tok_arr, n_regions, args.seed)
    rand_tok = _random_tok_arr(tok_arr,  n_regions, args.seed)

    real_sizes  = _region_sizes(tok_arr,  n_regions)
    shuf_sizes  = _region_sizes(shuf_tok, n_regions)
    rand_sizes  = _region_sizes(rand_tok, n_regions)
    assert real_sizes == shuf_sizes, "Shuffled map changed region sizes!"
    assert real_sizes == rand_sizes, "Random map changed region sizes!"

    out = args.output_dir
    for name, arr in [("shuffled_token_to_region.json", shuf_tok),
                      ("random_token_to_region.json",   rand_tok)]:
        json.dump({str(i): int(v) for i, v in enumerate(arr) if int(v) < n_regions},
                  open(os.path.join(out, name), "w"), indent=2)
        print(f"[save] {name}")
    for name, sizes in [("real_region_sizes.json", real_sizes),
                        ("shuffled_region_sizes.json", shuf_sizes),
                        ("random_region_sizes.json",   rand_sizes)]:
        json.dump(sizes, open(os.path.join(out, name), "w"), indent=2)

    # ── Buckets (real map) ────────────────────────────────────────────────────
    print("\n[step 4] bucket labels...")
    train_buckets = compute_bucket_labels(train_data, tok_arr, reg_arr, n_regions)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = {**vars(args),
           "n_regions": n_regions, "n_super": n_super, "d_in": d_in,
           "train_rows": len(train_data["gold"]), "val_rows": len(val_data["gold"]),
           "has_router": bool(has_router), "has_h": bool(train_data["has_h"]),
           "U_shape": list(U.shape), "device": str(device)}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # ── Router baseline ───────────────────────────────────────────────────────
    print("\n[step 5] router baseline...")
    router_vm = eval_router_baseline(val_data, tok_arr, n_regions)
    print(f"  router recall@8={router_vm['gold_region_recall@8']:.4f}" if router_vm["router_available"] else "  router: unavailable")
    router_vm.update({"variant": "router_baseline",
                      "selected_for_comparison": "baseline", "no_improving_checkpoint_found": False})

    all_comp_rows  = [build_comparison_row("router_baseline", router_vm, "baseline", False)]
    all_train_logs = []
    all_eval_logs  = []
    all_best_info  = {}
    all_final_info = {}

    def _make_model(tok_arr_v, use_context):
        return ContextualRegionMeaningLearner(
            tok_arr=tok_arr_v, reg_arr=reg_arr, U_frozen=U,
            n_regions=n_regions, n_super=n_super, d_in=d_in,
            d_model=args.d_model,
            region_emb_dim=args.region_emb_dim, super_emb_dim=args.super_emb_dim,
            num_region_layers=args.num_region_layers, num_heads=args.num_heads,
            ff_mult=args.ff_mult, dropout=args.dropout, use_context=use_context,
        ).to(device)

    def _run(variant, tok_arr_v, use_context):
        print(f"\n{'='*60}\n[variant] {variant}\n{'='*60}")
        model = _make_model(tok_arr_v, use_context)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{variant}] params: {n_params:,}")

        result = train_variant(
            variant, model,
            train_data, val_data,
            tok_arr_v, reg_arr, n_regions,   # model map
            tok_arr,   reg_arr, n_regions,    # real map for confuser metrics
            train_buckets, M, args, device, out)

        bm = result["best"]; fm = result["final"]
        vm = bm if not bm["no_improving_checkpoint_found"] else fm

        all_comp_rows.append(build_comparison_row(
            variant, vm,
            bm["selected_for_comparison"], bm["no_improving_checkpoint_found"]))
        all_train_logs.extend(result["train_log"])
        all_eval_logs.extend(result["eval_log"])
        all_best_info[variant]  = bm
        all_final_info[variant] = fm
        print(f"[{variant}] recall@8={vm['gold_region_recall@8']:.4f}  "
              f"conf_pair={vm['same_region_confuser_pair_acc']:.4f}  "
              f"wr_acc={vm['within_true_region_acc']:.4f}  "
              f"val_score={vm['validation_score']:.4f}")

    # ── Run variants ──────────────────────────────────────────────────────────
    _run("static_region_meaning",            tok_arr,  use_context=False)
    _run("contextual_region_meaning_real",   tok_arr,  use_context=True)
    _run("contextual_region_meaning_shuffled", shuf_tok, use_context=True)
    if not args.skip_random_control:
        _run("contextual_region_meaning_random", rand_tok, use_context=True)

    # ── Save outputs ──────────────────────────────────────────────────────────
    _wcsv(os.path.join(out, "phase2A_comparison.csv"), all_comp_rows)
    _wcsv(os.path.join(out, "train_log.csv"),          all_train_logs)
    _wcsv(os.path.join(out, "eval_log.csv"),           all_eval_logs)
    with open(os.path.join(out, "best_metrics.json"),  "w") as f:
        json.dump(all_best_info,  f, indent=2, default=str)
    with open(os.path.join(out, "final_metrics.json"), "w") as f:
        json.dump(all_final_info, f, indent=2, default=str)

    # ── Report ────────────────────────────────────────────────────────────────
    rpt = write_report(args, all_comp_rows, out)

    # ── Verdict ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(" PHASE 2A VERDICT:")
    print(f"{'='*60}")
    for row in all_comp_rows:
        print(f"  {row['variant']:40s}  "
              f"recall@8={_fmt(row.get('gold_region_recall@8',float('nan')))}  "
              f"conf={_fmt(row.get('same_region_confuser_pair_acc',float('nan')))}  "
              f"wr={_fmt(row.get('within_true_region_acc',float('nan')))}  "
              f"val={_fmt(row.get('validation_score',float('nan')))}")

    ctx_real  = next((r for r in all_comp_rows if r["variant"] == "contextual_region_meaning_real"), None)
    static_r  = next((r for r in all_comp_rows if r["variant"] == "static_region_meaning"), None)
    shuf_rows = [r for r in all_comp_rows if "shuffled" in r["variant"] or "random" in r["variant"]]

    def _r8(r): return r.get("gold_region_recall@8", float("nan")) if r else float("nan")
    def _c(r):  return r.get("same_region_confuser_pair_acc", float("nan")) if r else float("nan")
    def _wr(r): return r.get("within_true_region_acc", float("nan")) if r else float("nan")

    beats_static = ctx_real and static_r and _r8(ctx_real) > _r8(static_r) + 0.001
    shuf_conf    = max((_c(r)  for r in shuf_rows), default=float("nan"))
    beats_shuf   = ctx_real and _c(ctx_real) > shuf_conf + 0.001 if shuf_rows else True
    wr_ok        = ctx_real and _wr(ctx_real) == _wr(ctx_real) and _wr(ctx_real) > 0.1
    proceed      = bool(beats_static and beats_shuf and wr_ok)
    verdict      = "PROCEED_TO_PHASE_2B" if proceed else "DO_NOT_PROCEED_TO_PHASE_2B"

    if not proceed:
        reasons = []
        if not beats_static: reasons.append("ctx_does_not_beat_static_on_recall8")
        if not beats_shuf:   reasons.append("ctx_does_not_beat_shuffled_on_confuser")
        if not wr_ok:        reasons.append("within_region_acc_not_nontrivial")
        print(f"\n  failure reasons: {', '.join(reasons)}")

    print(f"\n  recommendation: {verdict}")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"  report:  {rpt}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
