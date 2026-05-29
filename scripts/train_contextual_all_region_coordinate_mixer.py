#!/usr/bin/env python3
"""
train_contextual_all_region_coordinate_mixer.py — Phase 2: Contextual All-Region Coordinate Mixer

Phase 1 result: static region IDs/embeddings did NOT beat base or logit-only MLP.
Phase 2 hypothesis: a contextual all-region state table (built per-row from hidden h +
candidate pool summaries) exposes local token identity that static labels cannot.

Variants compared:
  base_only
  logit_only_mlp
  static_region_embedding
  contextual_all_region_real
  contextual_all_region_shuffled
  contextual_all_region_random

Safety rules:
  Gold used only for CE loss and metrics — never as model input.
  Step-0 identity check for every trainable variant.
  No best checkpoint unless validation candidate NLL beats base.
  Shuffled/random controls preserve region size distribution.
  Evaluation slices always use real region map.
  Fail loudly on NaNs.
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

# ── Dims for static baseline (match Phase 1) ─────────────────────────────────
EMB_R_STATIC   = 32
EMB_S_STATIC   = 16
LOGIT_DIM      = 6
STATIC_GATE_IN = 4

# ── Dims for contextual model ─────────────────────────────────────────────────
COORD_SUMMARY_DIM = 12
MIXER_SCALAR_DIM  = 4
CTX_GATE_FEAT_DIM = 6

# ── Robust unembedding search ─────────────────────────────────────────────────
_U_EXACT   = [
    "lm_head.weight","token_emb.weight","tok_emb.weight",
    "transformer.wte.weight","wte.weight",
    "model.lm_head.weight","model.token_emb.weight","model.tok_emb.weight",
    "model.transformer.wte.weight","embedding.weight","model.embed_tokens.weight",
]
_U_SUFFIX  = ["lm_head.weight","token_emb.weight","tok_emb.weight","wte.weight"]
_U_PREFIXES= ["module.","_orig_mod.","model.model.","model.","_model."]

# ── Shard field aliases ───────────────────────────────────────────────────────
_TOPK = ["base_topk_ids","base_topk","topk_ids"]
_LGT  = ["base_topk_logits","base_topk_lgt","topk_lgt","topk_logits"]
_GOLD = ["gold_token","gold","labels"]
_RID  = ["row_id","row_ids"]
_OFF  = ["token_offset","offsets","offset"]
_RREG = ["router_topk_reg"]
_RPRB = ["router_topk_prb"]
_RMRG = ["router_margin"]
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


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Shard loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_shards(shard_dir, selected_M, max_rows=None, label=""):
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")
    print(f"[shards] {label}: {len(paths)} shards in {shard_dir}")
    bufs = defaultdict(list)
    total = 0
    first = True
    d_model_detected = 0
    for sp in paths:
        if max_rows and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print(f"[shards] shard keys: {list(sh.keys())}")
            first = False
        topk = _get(sh, _TOPK).long()
        lgt  = _get(sh, _LGT).float()
        gold = _get(sh, _GOLD).long()
        rids = _get(sh, _RID,  required=False)
        off  = _get(sh, _OFF,  required=False)
        rreg = _get(sh, _RREG, required=False)
        rprb = _get(sh, _RPRB, required=False)
        rmrg = _get(sh, _RMRG, required=False)
        h_ctx = _get(sh, _HCTX, required=False)
        h_raw = _get(sh, _HRAW, required=False)
        h = h_ctx if h_ctx is not None else h_raw

        B, K = topk.shape
        if rids is None: rids = torch.arange(total, total + B)
        if off  is None: off  = torch.full((B,), -1, dtype=torch.long)

        M = min(K, selected_M)
        topk = topk[:, :M]
        lgt  = lgt[:, :M]

        if max_rows and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, rids, off = (x[:keep] for x in (topk, lgt, gold, rids, off))
            if rreg is not None: rreg = rreg[:keep]
            if rprb is not None: rprb = rprb[:keep]
            if rmrg is not None: rmrg = rmrg[:keep]
            if h    is not None: h    = h[:keep]
            B = keep

        lgt = torch.where(torch.isfinite(lgt), lgt, torch.full_like(lgt, -1e9))

        bufs["topk_ids"].append(topk.numpy().astype(np.int32))
        bufs["topk_lgt"].append(lgt.numpy().astype(np.float32))
        bufs["gold"].append(gold.numpy().astype(np.int32))
        bufs["row_ids"].append(rids.numpy().astype(np.int32))
        bufs["token_offset"].append(off.numpy().astype(np.int32))

        if rreg is not None:
            K2 = rreg.shape[1]
            bufs["router_reg"].append(rreg.long().numpy().astype(np.int32))
            rp = rprb.float() if rprb is not None else torch.ones(B, K2) / K2
            bufs["router_prb"].append(rp.numpy().astype(np.float32))
        if rmrg is not None:
            bufs["router_margin"].append(rmrg.float().reshape(B).numpy().astype(np.float32))
        if h is not None:
            hf = h.float()
            if hf.dim() == 3:
                hf = hf[:, -1, :]
            d_model_detected = hf.shape[-1]
            bufs["h"].append(hf.numpy().astype(np.float32))

        total += B

    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    data["has_router"] = "router_reg" in data
    data["has_h"]      = "h" in data
    data["d_model"]    = d_model_detected
    data["M"]          = M
    print(f"[shards] {label}: {total:,} rows  M={M}  has_router={data['has_router']}  "
          f"has_h={data['has_h']}  d_model={data['d_model']}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Unembedding loading
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_prefixes(k):
    changed = True
    while changed:
        changed = False
        for pfx in _U_PREFIXES:
            if k.startswith(pfx):
                k = k[len(pfx):]
                changed = True
    return k


def load_unembedding(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if not isinstance(sd, dict):
        sd = ckpt

    print(f"[unembedding] checkpoint tensor keys ({len(sd)}):")
    for k in list(sd.keys())[:30]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else "?"
        print(f"  {k}: {shape}")
    if len(sd) > 30:
        print(f"  ... ({len(sd) - 30} more)")

    tensor_keys = {k: v for k, v in sd.items()
                   if isinstance(v, torch.Tensor) and v.dim() == 2}

    # Stage 1: exact match on stripped keys
    for raw_k, v in tensor_keys.items():
        k = _strip_prefixes(raw_k)
        if k in _U_EXACT:
            info = {"unembedding_source": raw_k, "U_shape": list(v.shape),
                    "dtype": str(v.dtype), "stage": 1}
            print(f"[unembedding] found via exact match: {raw_k} {tuple(v.shape)}")
            return v.float(), info

    # Stage 2: suffix match on stripped keys
    for raw_k, v in tensor_keys.items():
        k = _strip_prefixes(raw_k)
        for sfx in _U_SUFFIX:
            if k.endswith(sfx):
                info = {"unembedding_source": raw_k, "U_shape": list(v.shape),
                        "dtype": str(v.dtype), "stage": 2}
                print(f"[unembedding] found via suffix on stripped key: {raw_k} {tuple(v.shape)}")
                return v.float(), info

    # Stage 3: suffix match on original keys
    for raw_k, v in tensor_keys.items():
        for sfx in _U_SUFFIX:
            if raw_k.endswith(sfx):
                info = {"unembedding_source": raw_k, "U_shape": list(v.shape),
                        "dtype": str(v.dtype), "stage": 3}
                print(f"[unembedding] found via suffix on original key: {raw_k} {tuple(v.shape)}")
                return v.float(), info

    raise RuntimeError(
        f"Could not load unembedding from {ckpt_path}.\n"
        f"Tried exact aliases: {_U_EXACT}\n"
        f"Tried suffix aliases: {_U_SUFFIX}\n"
        f"All 2D tensor keys: {list(tensor_keys.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Region maps + controls
# ═══════════════════════════════════════════════════════════════════════════════

def load_region_maps(t2r_path, super_path):
    if not t2r_path or not os.path.isfile(t2r_path):
        return np.zeros(50257, np.int32), np.zeros(2, np.int32), 1, 1
    with open(t2r_path) as f:
        raw = json.load(f)
    t2r = ({i: v for i, v in enumerate(raw) if v is not None}
           if isinstance(raw, list) else {int(k): v for k, v in raw.items()})
    n_regions = int(max(t2r.values())) + 1 if t2r else 1
    unk_r = n_regions
    V = max(t2r.keys(), default=0) + 2
    tok_arr = np.full(V, unk_r, dtype=np.int32)
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


def _shuffle_tok_arr(tok_arr, n_regions, seed):
    rng = np.random.default_rng(seed)
    unk_r = n_regions
    valid = np.where(tok_arr < unk_r)[0]
    vals  = tok_arr[valid].copy()
    rng.shuffle(vals)
    out = tok_arr.copy()
    out[valid] = vals
    return out


def _random_tok_arr(tok_arr, n_regions, seed):
    rng = np.random.default_rng(seed + 1)
    unk_r = n_regions
    valid  = np.where(tok_arr < unk_r)[0]
    vals   = tok_arr[valid]
    sizes  = np.bincount(vals, minlength=n_regions)
    new    = np.repeat(np.arange(n_regions, dtype=np.int32), sizes)
    rng.shuffle(new)
    out = tok_arr.copy()
    out[valid] = new
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Bucket labeling (always uses real tok_arr)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bucket_labels(data, tok_arr, reg_arr):
    gold     = data["gold"].astype(np.int64)
    topk_ids = data["topk_ids"].astype(np.int64)
    N, M     = topk_ids.shape
    Vt = len(tok_arr); Rr = len(reg_arr)

    base_top1    = topk_ids[:, 0]
    base_correct = (base_top1 == gold)
    gip_M        = (topk_ids == gold[:, None]).any(axis=1)
    bucketA      = ~base_correct & gip_M

    base_reg = tok_arr[np.clip(base_top1, 0, Vt-1)]
    gold_reg = tok_arr[np.clip(gold,      0, Vt-1)]
    base_sreg = reg_arr[np.clip(base_reg, 0, Rr-1)]
    gold_sreg = reg_arr[np.clip(gold_reg, 0, Rr-1)]
    unk_r = int(tok_arr.max()); unk_s = int(reg_arr.max())

    same_reg  = (base_reg == gold_reg)  & (gold_reg  != unk_r)
    same_sreg = (base_sreg == gold_sreg) & (gold_sreg != unk_s)
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
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Models
# ═══════════════════════════════════════════════════════════════════════════════

class _RowGateMLP(nn.Module):
    """Row-level gate: [B, STATIC_GATE_IN] → [B, 1] sigmoid."""
    def __init__(self, hidden=32, init_bias=0.0, active=True):
        super().__init__()
        self.active = active
        if active:
            self.net = nn.Sequential(
                nn.Linear(STATIC_GATE_IN, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, row_feats):
        if not self.active:
            return torch.ones(row_feats.shape[0], 1, device=row_feats.device)
        return torch.sigmoid(self.net(row_feats))


class _CandMLP(nn.Module):
    """Per-candidate MLP: [B, M, in_dim] → [B, M]; zero-init final layer."""
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        B, M, _ = x.shape
        return self.net(x.reshape(B * M, -1)).reshape(B, M)


def _row_feats_static(topk_lgt):
    M = topk_lgt.shape[1]
    t1 = topk_lgt[:, 0]
    t2 = topk_lgt[:, min(1, M-1)]
    return torch.stack([t1, t1 - t2, topk_lgt.mean(1), topk_lgt.std(1)], dim=1)


# ── 5a. Logit-only MLP ────────────────────────────────────────────────────────

class LogitOnlyMixer(nn.Module):
    def __init__(self, hidden, dropout, use_gate=True, gate_init_bias=0.0):
        super().__init__()
        self.cand_mlp = _CandMLP(LOGIT_DIM, hidden, dropout)
        self.gate_mlp = _RowGateMLP(hidden=32, init_bias=gate_init_bias, active=use_gate)

    def _feats(self, topk_ids, topk_lgt):
        B, M = topk_lgt.shape
        rank_frac   = (torch.arange(M, device=topk_lgt.device, dtype=torch.float32)
                       / max(M-1, 1)).view(1, M, 1).expand(B, M, 1)
        gap         = (topk_lgt[:, 0:1] - topk_lgt).unsqueeze(-1)
        base_margin = (topk_lgt[:, 0] - topk_lgt[:, min(1, M-1)]).view(B, 1, 1).expand(B, M, 1)
        row_mean    = topk_lgt.mean(1, keepdim=True).unsqueeze(-1).expand(B, M, 1)
        row_std     = topk_lgt.std(1, keepdim=True).unsqueeze(-1).expand(B, M, 1)
        lgt_3       = topk_lgt.unsqueeze(-1)
        return torch.cat([rank_frac, lgt_3, gap, base_margin, row_mean, row_std], dim=-1)

    def forward(self, topk_ids, topk_lgt, **kwargs):
        feat  = self._feats(topk_ids, topk_lgt)
        delta = self.cand_mlp(feat)
        gate  = self.gate_mlp(_row_feats_static(topk_lgt)).expand(topk_lgt.shape[0], topk_lgt.shape[1])
        final = topk_lgt + gate * delta
        return final, delta, gate


# ── 5b. Static Region Embedding Mixer (Phase 1 parity) ───────────────────────

class StaticRegionEmbMixer(nn.Module):
    def __init__(self, tok_arr, reg_arr, n_regions, n_super,
                 hidden, dropout, has_router, use_gate=True, gate_init_bias=0.0):
        super().__init__()
        self.register_buffer("tok_arr", torch.from_numpy(tok_arr).int())
        self.register_buffer("reg_arr", torch.from_numpy(reg_arr).int())
        self.has_router = has_router

        self.region_emb = nn.Embedding(n_regions + 2, EMB_R_STATIC, padding_idx=n_regions)
        self.sreg_emb   = nn.Embedding(n_super   + 2, EMB_S_STATIC, padding_idx=n_super)

        scalar    = 6
        emb_base  = 2 * EMB_R_STATIC + 2 * EMB_S_STATIC
        router_ex = (5 + 2 * EMB_R_STATIC) if has_router else 0
        self.feat_dim = scalar + emb_base + router_ex

        self.cand_mlp = _CandMLP(self.feat_dim, hidden, dropout)
        self.gate_mlp = _RowGateMLP(hidden=32, init_bias=gate_init_bias, active=use_gate)

    def _build_features(self, topk_ids, topk_lgt, router_reg, router_prb, router_margin):
        B, M = topk_ids.shape
        Vt = self.tok_arr.shape[0]; Rr = self.reg_arr.shape[0]

        cand_reg  = self.tok_arr[topk_ids.clamp(0, Vt-1).long()].long()
        cand_sreg = self.reg_arr[cand_reg.clamp(0, Rr-1)].long()
        base_reg  = cand_reg[:, 0]; base_sreg = cand_sreg[:, 0]

        cand_reg_emb  = self.region_emb(cand_reg.clamp(0, self.region_emb.num_embeddings-1))
        cand_sreg_emb = self.sreg_emb(cand_sreg.clamp(0, self.sreg_emb.num_embeddings-1))
        base_reg_emb  = self.region_emb(base_reg.clamp(0, self.region_emb.num_embeddings-1))
        base_sreg_emb = self.sreg_emb(base_sreg.clamp(0, self.sreg_emb.num_embeddings-1))

        diff_reg  = cand_reg_emb  - base_reg_emb.unsqueeze(1)
        diff_sreg = cand_sreg_emb - base_sreg_emb.unsqueeze(1)

        rank_frac   = (torch.arange(M, device=topk_ids.device, dtype=torch.float32)
                       / max(M-1, 1)).view(1, M, 1).expand(B, M, 1)
        gap         = (topk_lgt[:, 0:1] - topk_lgt).unsqueeze(-1)
        same_reg    = (cand_reg == base_reg.unsqueeze(1)).float().unsqueeze(-1)
        same_sreg   = (cand_sreg == base_sreg.unsqueeze(1)).float().unsqueeze(-1)
        base_margin = (topk_lgt[:, 0] - topk_lgt[:, min(1, M-1)]).view(B, 1, 1).expand(B, M, 1)
        lgt_3       = topk_lgt.unsqueeze(-1)

        parts = [rank_frac, lgt_3, gap, same_reg, same_sreg, base_margin,
                 cand_reg_emb, cand_sreg_emb, diff_reg, diff_sreg]

        if router_reg is not None and self.has_router:
            K = router_reg.shape[1]
            rt1_reg  = router_reg[:, 0].long()
            rt1_sreg = self.reg_arr[rt1_reg.clamp(0, Rr-1)].long()
            rt1_emb  = self.region_emb(rt1_reg.clamp(0, self.region_emb.num_embeddings-1))
            diff_rt  = cand_reg_emb - rt1_emb.unsqueeze(1)
            same_rt  = (cand_reg == rt1_reg.unsqueeze(1)).float().unsqueeze(-1)
            same_srt = (cand_sreg == rt1_sreg.unsqueeze(1)).float().unsqueeze(-1)
            match    = (router_reg.unsqueeze(2) == cand_reg.unsqueeze(1))
            rprb_f   = router_prb.float() if router_prb is not None else torch.ones(B, K, device=topk_ids.device) / K
            rp_cand  = (rprb_f.unsqueeze(2) * match.float()).sum(1).unsqueeze(-1)
            k_idx    = torch.arange(K, device=topk_ids.device, dtype=torch.float32)
            rr_cand  = torch.where(match, k_idx.view(1,K,1)/max(K-1,1),
                                   torch.ones(1,1,1,device=topk_ids.device)).min(1).values.unsqueeze(-1)
            rm       = (router_margin.view(B,1,1).expand(B,M,1) if router_margin is not None
                        else torch.zeros(B, M, 1, device=topk_ids.device))
            parts.extend([same_rt, same_srt, rp_cand, rr_cand,
                           rt1_emb.unsqueeze(1).expand(B, M, EMB_R_STATIC), diff_rt, rm])
        elif self.has_router:
            z1 = torch.zeros(B, M, 1, device=topk_ids.device)
            ze = torch.zeros(B, M, EMB_R_STATIC, device=topk_ids.device)
            parts.extend([z1, z1, z1, z1, ze, ze, z1])

        return torch.cat(parts, dim=-1)

    def forward(self, topk_ids, topk_lgt, **kwargs):
        router_reg    = kwargs.get("router_reg")
        router_prb    = kwargs.get("router_prb")
        router_margin = kwargs.get("router_margin")
        feat  = self._build_features(topk_ids, topk_lgt, router_reg, router_prb, router_margin)
        delta = self.cand_mlp(feat)
        gate  = self.gate_mlp(_row_feats_static(topk_lgt)).expand(topk_lgt.shape[0], topk_lgt.shape[1])
        final = topk_lgt + gate * delta
        return final, delta, gate


# ── 5c. Contextual All-Region Coordinate Mixer ───────────────────────────────

class ContextualAllRegionCoordinateMixer(nn.Module):
    """
    Builds a contextual state Z for ALL R regions per row, then computes
    candidate-region compatibility via coord_logits = Q W_Q (Z W_K)^T / sqrt(d).
    """
    def __init__(self, tok_arr, reg_arr, U_frozen, n_regions, n_super, d_model,
                 has_router, model_dim, region_emb_dim, super_emb_dim,
                 coord_proj_dim, hidden_dim, dropout, no_gate, gate_init_bias):
        super().__init__()
        self.register_buffer("tok_arr",   torch.from_numpy(tok_arr).int())
        self.register_buffer("reg_arr",   torch.from_numpy(reg_arr).int())
        self.register_buffer("U_frozen",  U_frozen.float())  # [V, token_dim]

        self.R              = n_regions
        self.model_dim      = model_dim
        self.region_emb_dim = region_emb_dim
        self.super_emb_dim  = super_emb_dim
        self.coord_proj_dim = coord_proj_dim
        self.has_router     = has_router
        self.no_gate        = no_gate
        token_dim           = U_frozen.shape[1]

        # Embeddings
        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim, padding_idx=n_regions)
        self.sreg_emb   = nn.Embedding(n_super   + 2, super_emb_dim,  padding_idx=n_super)

        # Projections
        self.h_proj  = nn.Linear(max(d_model, 1), model_dim)
        self.U_proj  = nn.Linear(token_dim, model_dim)

        # Region state MLP
        # Input: region_emb + sreg_emb + h_proj + router(4) + cand_summary(5) + pooled_U
        self.region_input_dim = (region_emb_dim + super_emb_dim + model_dim
                                 + 9 + model_dim)
        self.region_mlp = nn.Sequential(
            nn.Linear(self.region_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

        # Candidate encoder MLP
        cand_scalar_dim = 6 + (4 if has_router else 0)
        self.cand_input_dim = model_dim + region_emb_dim + super_emb_dim + cand_scalar_dim
        self.cand_encoder = nn.Sequential(
            nn.Linear(self.cand_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

        # Coordinate projections (bilinear)
        self.coord_W_Q = nn.Linear(model_dim, model_dim, bias=False)
        self.coord_W_K = nn.Linear(model_dim, model_dim, bias=False)

        # Coord feature projector: concat(coord_logits, coord_adv) → coord_proj_dim
        self.coord_proj = nn.Linear(2 * n_regions, coord_proj_dim)

        # Mixer MLP: concat(Q, region_ctx, coord_feat, summaries, scalars) → delta
        self.mixer_input_dim = (model_dim + model_dim + coord_proj_dim
                                + COORD_SUMMARY_DIM + MIXER_SCALAR_DIM)
        self.mixer_mlp = nn.Sequential(
            nn.Linear(self.mixer_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Zero-init final layer for identity at step 0
        nn.init.zeros_(self.mixer_mlp[-1].weight)
        nn.init.zeros_(self.mixer_mlp[-1].bias)

        # Gate MLP (per-candidate, 6 scalars → 1 sigmoid)
        if not no_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(CTX_GATE_FEAT_DIM, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )
            nn.init.constant_(self.gate_mlp[-1].bias, gate_init_bias)
        else:
            self.gate_mlp = None

    def _compute_cand_emb(self, topk_ids):
        """Project frozen token embeddings for M candidates: [B, M, model_dim]."""
        V = self.U_frozen.shape[0]
        tok = topk_ids.clamp(0, V-1).long()
        U_cand = self.U_frozen[tok.reshape(-1)]         # [B*M, token_dim]
        return self.U_proj(U_cand).reshape(*tok.shape, self.model_dim)

    def _build_region_states(self, B, M, R, topk_ids, topk_lgt, h_proj_out,
                              cand_emb, cand_reg_c, router_reg, router_prb, device):
        """Returns [B, R, model_dim] region state and intermediate tensors."""
        # Ensure float32 — callers should already cast, but guard defensively
        h_proj_out = h_proj_out.float()
        cand_emb   = cand_emb.float()
        topk_lgt   = topk_lgt.float()
        # Static region embeddings for all R regions
        r_idx    = torch.arange(R, device=device)
        r_emb    = self.region_emb(r_idx)                               # [R, region_emb_dim]
        sr_idx   = self.reg_arr[:R].long().clamp(0, self.sreg_emb.num_embeddings-1)
        sr_emb   = self.sreg_emb(sr_idx)                                # [R, super_emb_dim]
        r_emb_b  = r_emb.unsqueeze(0).expand(B, R, -1)
        sr_emb_b = sr_emb.unsqueeze(0).expand(B, R, -1)
        h_expand = h_proj_out.unsqueeze(1).expand(B, R, -1)

        # Router features per region
        router_prob_r  = torch.zeros(B, R, device=device)
        router_rank_r  = torch.ones(B, R, device=device)
        is_top1_r      = torch.zeros(B, R, device=device)
        is_topk_r      = torch.zeros(B, R, device=device)

        if self.has_router and router_reg is not None:
            K = router_reg.shape[1]
            vr = router_reg.clamp(0, R-1).long()
            rprb_f = router_prb.float() if router_prb is not None else torch.ones(B, K, device=device) / K
            router_prob_r.scatter_add_(1, vr, rprb_f)
            rank_v = (torch.arange(K, device=device, dtype=torch.float32)
                      / max(K-1, 1)).unsqueeze(0).expand(B, K)
            router_rank_r.scatter_(1, vr, rank_v)
            is_topk_r.scatter_(1, vr, torch.ones(B, K, device=device))
            is_top1_r.scatter_(1, router_reg[:, 0:1].clamp(0, R-1).long(), torch.ones(B, 1, device=device))

        # Candidate summary per region
        count_r = torch.zeros(B, R, device=device)
        count_r.scatter_add_(1, cand_reg_c, torch.ones(B, M, device=device))
        present_r = (count_r > 0).float()

        sum_lgt = torch.zeros(B, R, device=device)
        sum_lgt.scatter_add_(1, cand_reg_c, topk_lgt)
        mean_lgt = (sum_lgt / (count_r + _EPS)) * present_r

        # Max logit per region via masked approach
        cand_oh = (cand_reg_c.unsqueeze(2) == r_idx.view(1, 1, R))     # [B, M, R] bool
        masked  = torch.where(cand_oh, topk_lgt.unsqueeze(2),
                              torch.full_like(topk_lgt.unsqueeze(2), -1e9))
        max_lgt = masked.max(dim=1).values * present_r                  # [B, R]

        softmax_s = torch.softmax(topk_lgt, dim=1)
        mass_r = torch.zeros(B, R, device=device)
        mass_r.scatter_add_(1, cand_reg_c, softmax_s)

        # Pooled candidate embedding per region
        idx3d    = cand_reg_c.unsqueeze(-1).expand(B, M, self.model_dim)
        pooled_U = torch.zeros(B, R, self.model_dim, device=device)
        pooled_U.scatter_add_(1, idx3d, cand_emb)
        pooled_U = (pooled_U / (count_r.unsqueeze(-1) + _EPS)) * present_r.unsqueeze(-1)

        # Normalize scalars
        scalars = torch.stack([
            count_r / max(M, 1),          # count_norm
            present_r,                    # present
            max_lgt / 10.0,               # max_lgt (rough normalize)
            mean_lgt / 10.0,              # mean_lgt
            mass_r,                       # softmax mass
            router_prob_r,                # router prob
            router_rank_r,                # router rank
            is_top1_r,                    # is router top1
            is_topk_r,                    # is router topk
        ], dim=-1)                                                       # [B, R, 9]

        region_input = torch.cat([r_emb_b, sr_emb_b, h_expand, scalars, pooled_U], dim=-1)
        # [B, R, region_input_dim]

        B_, R_, D_ = region_input.shape
        region_states = self.region_mlp(region_input.reshape(B_ * R_, D_)).reshape(B_, R_, self.model_dim)
        return region_states, count_r, present_r

    def _build_candidate_features(self, B, M, topk_ids, topk_lgt, cand_emb,
                                   cand_reg_c, router_reg, router_prb, router_margin, device):
        """Returns [B, M, cand_input_dim] candidate features."""
        # Ensure float32 so cat with embedding outputs (float32) doesn't raise
        cand_emb   = cand_emb.float()
        topk_lgt   = topk_lgt.float()
        Rr = self.reg_arr.shape[0]
        cand_sreg = self.reg_arr[cand_reg_c.clamp(0, Rr-1)].long()
        base_reg  = cand_reg_c[:, 0]; base_sreg = cand_sreg[:, 0]

        c_reg_emb  = self.region_emb(cand_reg_c.clamp(0, self.region_emb.num_embeddings-1))
        c_sreg_emb = self.sreg_emb(cand_sreg.clamp(0, self.sreg_emb.num_embeddings-1))

        rank_frac   = (torch.arange(M, device=device, dtype=torch.float32)
                       / max(M-1, 1)).view(1, M, 1).expand(B, M, 1)
        gap         = (topk_lgt[:, 0:1] - topk_lgt).unsqueeze(-1)
        base_margin = (topk_lgt[:, 0] - topk_lgt[:, min(1, M-1)]).view(B, 1, 1).expand(B, M, 1)
        lgt_3       = topk_lgt.unsqueeze(-1)
        same_reg    = (cand_reg_c == base_reg.unsqueeze(1)).float().unsqueeze(-1)
        same_sreg   = (cand_sreg  == base_sreg.unsqueeze(1)).float().unsqueeze(-1)

        parts = [cand_emb, c_reg_emb, c_sreg_emb,
                 lgt_3, rank_frac, gap, same_reg, same_sreg, base_margin]

        if self.has_router and router_reg is not None:
            K = router_reg.shape[1]
            rt1_reg  = router_reg[:, 0].long()
            rt1_sreg = self.reg_arr[rt1_reg.clamp(0, Rr-1)].long()
            same_rt  = (cand_reg_c == rt1_reg.unsqueeze(1)).float().unsqueeze(-1)
            same_srt = (cand_sreg  == rt1_sreg.unsqueeze(1)).float().unsqueeze(-1)
            match    = (router_reg.long().unsqueeze(2) == cand_reg_c.unsqueeze(1))  # [B,K,M]
            rprb_f   = router_prb.float() if router_prb is not None else torch.ones(B, K, device=device) / K
            rp_cand  = (rprb_f.unsqueeze(2) * match.float()).sum(1).unsqueeze(-1)
            k_idx    = torch.arange(K, device=device, dtype=torch.float32)
            rr_cand  = torch.where(match, k_idx.view(1,K,1)/max(K-1,1),
                                   torch.ones(1,1,1,device=device)).min(1).values.unsqueeze(-1)
            parts.extend([same_rt, same_srt, rp_cand, rr_cand])
        elif self.has_router:
            parts.extend([torch.zeros(B,M,1,device=device)] * 4)

        return torch.cat(parts, dim=-1)

    def _compute_coord_summaries(self, coord_logits, coord_adv, cand_reg_c, router_reg, device):
        """Compute [B, M, COORD_SUMMARY_DIM] summary features."""
        B, M, R = coord_logits.shape
        coord_sm    = torch.softmax(coord_logits, dim=-1)
        coord_max   = coord_logits.max(-1).values
        coord_mean  = coord_logits.mean(-1)
        coord_std   = coord_logits.std(-1)
        coord_ent   = -(coord_sm * (coord_sm + _EPS).log()).sum(-1)
        top2        = coord_logits.topk(min(2, R), dim=-1).values
        coord_margin = top2[:, :, 0] - top2[:, :, -1]

        coord_at_cand   = coord_logits.gather(-1, cand_reg_c.clamp(0, R-1).unsqueeze(-1)).squeeze(-1)
        base_reg_bc     = cand_reg_c[:, 0:1].expand(B, M).clamp(0, R-1)
        coord_at_base   = coord_logits.gather(-1, base_reg_bc.unsqueeze(-1)).squeeze(-1)

        if self.has_router and router_reg is not None:
            rt1_bc = router_reg[:, 0:1].long().expand(B, M).clamp(0, R-1)
            coord_at_rt = coord_logits.gather(-1, rt1_bc.unsqueeze(-1)).squeeze(-1)
        else:
            coord_at_rt = torch.zeros(B, M, device=device)

        coord_adv_at_cand = coord_adv.gather(-1, cand_reg_c.clamp(0, R-1).unsqueeze(-1)).squeeze(-1)
        adv_top2          = coord_adv.topk(min(2, R), dim=-1).values
        coord_adv_max     = coord_adv.max(-1).values
        coord_adv_mean    = coord_adv.mean(-1)
        coord_adv_margin  = adv_top2[:, :, 0] - adv_top2[:, :, -1]

        return torch.stack([
            coord_max, coord_mean, coord_std, coord_ent, coord_margin,
            coord_at_cand, coord_at_base, coord_at_rt,
            coord_adv_at_cand, coord_adv_max, coord_adv_mean, coord_adv_margin,
        ], dim=-1)                                                       # [B, M, 12]

    def forward(self, topk_ids, topk_lgt, h=None,
                router_reg=None, router_prb=None, router_margin=None, **kwargs):
        B, M   = topk_ids.shape
        R      = self.R
        device = topk_ids.device
        Vt     = self.tok_arr.shape[0]

        # Normalise topk_lgt to float32 (batch tensors are always float32;
        # this also ensures all downstream feature vectors stay float32 so
        # that scatter_add_ and torch.cat don't see mixed dtypes when AMP is on).
        topk_lgt = topk_lgt.float()

        # Context projection — cast to float32 so AMP float16 output doesn't
        # propagate into the scatter / cat operations below.
        if h is None:
            h = torch.zeros(B, 1, device=device)
        h_proj_out = self.h_proj(h.float()).float()                      # [B, model_dim]

        # Candidate token embeddings (U_proj is a Linear → float16 in AMP → cast back)
        cand_emb = self._compute_cand_emb(topk_ids).float()             # [B, M, model_dim]
        cand_reg  = self.tok_arr[topk_ids.clamp(0, Vt-1).long()].long()
        cand_reg_c = cand_reg.clamp(0, R-1)                             # [B, M]

        # Region states (region_mlp output may be float16 in AMP → cast back)
        Z, count_r, present_r = self._build_region_states(
            B, M, R, topk_ids, topk_lgt, h_proj_out,
            cand_emb, cand_reg_c, router_reg, router_prb, device)
        Z = Z.float()                                                    # [B, R, model_dim]

        # Candidate features + encoder (cand_encoder is a Linear stack → cast back)
        cand_feats = self._build_candidate_features(
            B, M, topk_ids, topk_lgt, cand_emb, cand_reg_c,
            router_reg, router_prb, router_margin, device)               # [B, M, cand_input_dim]
        B_, M_, D_ = cand_feats.shape
        Q = self.cand_encoder(cand_feats.reshape(B_ * M_, D_)).reshape(B_, M_, self.model_dim).float()

        # Coordinate computation (all ops cast to float32 after AMP)
        Q_proj       = self.coord_W_Q(Q).float()                        # [B, M, model_dim]
        Z_proj       = self.coord_W_K(Z).float()                        # [B, R, model_dim]
        scale        = math.sqrt(self.model_dim)
        coord_logits = torch.bmm(Q_proj, Z_proj.transpose(-1, -2)).float() / scale  # [B, M, R]
        coord_adv    = coord_logits - coord_logits[:, 0:1, :]            # [B, M, R]

        coord_feat     = self.coord_proj(torch.cat([coord_logits, coord_adv], dim=-1)).float()
        attn           = torch.softmax(coord_logits, dim=-1)            # [B, M, R] float32
        region_context = torch.bmm(attn, Z).float()                     # [B, M, model_dim]

        coord_summ = self._compute_coord_summaries(
            coord_logits, coord_adv, cand_reg_c, router_reg, device)    # [B, M, 12] float32

        # Mixer scalars (all float32 since topk_lgt is float32)
        gap_3 = (topk_lgt[:, 0:1] - topk_lgt).unsqueeze(-1)
        bm_3  = (topk_lgt[:, 0] - topk_lgt[:, min(1, M-1)]).view(B, 1, 1).expand(B, M, 1)
        rf_3  = (torch.arange(M, device=device, dtype=torch.float32) / max(M-1, 1)).view(1, M, 1).expand(B, M, 1)
        lgt_3 = topk_lgt.unsqueeze(-1)

        mixer_input = torch.cat([Q, region_context, coord_feat, coord_summ,
                                 lgt_3, gap_3, bm_3, rf_3], dim=-1)     # [B, M, mixer_input_dim]
        B_, M_, D_ = mixer_input.shape
        delta = self.mixer_mlp(mixer_input.reshape(B_ * M_, D_)).reshape(B_, M_).float()

        # Per-candidate gate
        if self.no_gate or self.gate_mlp is None:
            gate = torch.ones(B, M, device=device)
        else:
            gate_input = torch.stack([
                topk_lgt,
                gap_3.squeeze(-1),
                coord_summ[:, :, 4],   # coord_margin
                coord_summ[:, :, 10],  # coord_adv_mean
                rf_3.squeeze(-1),
                coord_summ[:, :, 3],   # coord_entropy
            ], dim=-1)                                                   # [B, M, 6]
            gate = torch.sigmoid(
                self.gate_mlp(gate_input.reshape(B * M, CTX_GATE_FEAT_DIM)).reshape(B, M).float())

        final = topk_lgt + gate * delta
        return final, delta, gate


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Identity check
# ═══════════════════════════════════════════════════════════════════════════════

def check_identity(model, data, M, device, variant, atol=1e-5):
    model.eval()
    n_check = min(256, len(data["gold"]))
    idx     = np.arange(n_check)
    batch   = _to_device(_make_batch(data, idx, M), device)
    gold_np = data["gold"][idx].astype(np.int64)
    topk_np = data["topk_ids"][idx].astype(np.int64)
    lgt_np  = data["topk_lgt"][idx].astype(np.float64)

    with torch.no_grad():
        final, delta, gate = model(**batch)

    max_diff = (final - batch["topk_lgt"]).abs().max().item()
    if max_diff > atol:
        raise RuntimeError(f"[{variant}] Identity FAILED: max_diff={max_diff:.2e}")

    final_np  = final.cpu().float().numpy()
    pred_top1 = topk_np[np.arange(n_check), final_np.argmax(axis=1)]
    base_top1 = topk_np[:, 0]
    base_corr = (base_top1 == gold_np)
    ctg = int((~base_corr & (pred_top1 == gold_np)).sum())
    caw = int((base_corr  & (pred_top1 != gold_np)).sum())
    if ctg != 0 or caw != 0:
        raise RuntimeError(f"[{variant}] Identity FAILED: ctg={ctg} caw={caw}")

    delta_max = float(np.abs(delta.cpu().float().numpy()).max())
    if delta_max > atol:
        raise RuntimeError(f"[{variant}] Identity FAILED: delta_abs_max={delta_max:.2e}")

    gip  = (topk_np == gold_np[:, None]).any(axis=1)
    gr   = np.where(gip, (topk_np == gold_np[:, None]).argmax(axis=1), 0)
    if gip.any():
        fs  = final_np.astype(np.float64)
        fs -= fs.max(axis=1, keepdims=True)
        m_nll = float(-( fs - np.log(np.exp(fs).sum(1, keepdims=True) + _EPS)
                        )[gip, gr[gip]].mean())
        lb = lgt_np.copy()
        lb -= lb.max(axis=1, keepdims=True)
        b_nll = float(-( lb - np.log(np.exp(lb).sum(1, keepdims=True) + _EPS)
                        )[gip, gr[gip]].mean())
        if abs(m_nll - b_nll) > 1e-4:
            raise RuntimeError(f"[{variant}] Identity FAILED: nll_diff={abs(m_nll-b_nll):.2e}")

    print(f"[{variant}] identity check PASSED: max_diff={max_diff:.2e} ctg=0 caw=0 delta_max={delta_max:.2e}")
    model.train()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Batch helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_batch(data, idx, M):
    d_model = data.get("d_model", 1) or 1
    b = {
        "topk_ids": data["topk_ids"][idx],
        "topk_lgt": data["topk_lgt"][idx],
        "h": (data["h"][idx] if data["has_h"]
              else np.zeros((len(idx), d_model), dtype=np.float32)),
    }
    if data["has_router"]:
        b["router_reg"] = data["router_reg"][idx]
        b["router_prb"] = data["router_prb"][idx]
        if "router_margin" in data:
            b["router_margin"] = data["router_margin"][idx]
    return b


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        out[k] = t.to(device)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Loss
# ═══════════════════════════════════════════════════════════════════════════════

def compute_loss(final_scores, topk_ids_t, gold_t, base_correct_mask, delta, gate, args):
    B, M = final_scores.shape
    gip      = (topk_ids_t == gold_t.unsqueeze(1)).any(1)
    gold_idx = (topk_ids_t == gold_t.unsqueeze(1)).float().argmax(1)

    loss_ce = torch.zeros(1, device=final_scores.device).squeeze()
    if gip.sum() > 0:
        loss_ce = F.cross_entropy(final_scores[gip], gold_idx[gip])

    loss_delta    = delta.pow(2).mean()
    loss_gate     = gate.mean() if args.use_gate else torch.zeros(1, device=final_scores.device).squeeze()
    loss_preserve = torch.zeros(1, device=final_scores.device).squeeze()
    bc = base_correct_mask
    if bc.sum() > 0:
        base_idx = torch.zeros(int(bc.sum()), dtype=torch.long, device=final_scores.device)
        loss_preserve = F.cross_entropy(final_scores[bc], base_idx)

    total = (loss_ce
             + args.lambda_delta    * loss_delta
             + args.lambda_gate     * loss_gate
             + args.lambda_preserve * loss_preserve)
    return total, {
        "ce": loss_ce.item(), "delta_l2": loss_delta.item(),
        "gate": loss_gate.item(), "preserve": loss_preserve.item(),
        "total": total.item(), "n_gip": int(gip.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Balanced sampler
# ═══════════════════════════════════════════════════════════════════════════════

def make_balanced_sampler(buckets, n_steps, batch_size, seed):
    rng  = np.random.default_rng(seed)
    ra   = np.where(buckets["bucketA"])[0]
    rb   = np.where(buckets["base_correct"])[0]
    rc   = np.where(buckets["same_reg_or_sreg_conf"])[0]
    all_ = np.concatenate([ra, rb, rc]) if len(ra)+len(rb)+len(rc) > 0 else np.arange(len(ra)+1)
    na   = max(1, int(batch_size * 0.40))
    nb   = max(1, int(batch_size * 0.40))
    nc   = batch_size - na - nb

    def samp(arr, n):
        if len(arr) == 0: return rng.choice(all_, n, replace=True)
        return rng.choice(arr, n, replace=len(arr) < n)

    return [np.concatenate([samp(ra, na), samp(rb, nb), samp(rc, nc)]) for _ in range(n_steps)]


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_variant(variant, model, train_data, val_data, buckets, M, args, device, out_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    sampler   = make_balanced_sampler(buckets, args.steps, args.batch_size, args.seed)

    base_nll_threshold = _eval_model(None, val_data, M, device, args, is_base=True)[
        "candidate_nll_given_gold_in_topM"]
    print(f"[{variant}] base_nll={base_nll_threshold:.5f} (must beat to save checkpoint)")

    best_nll  = float("inf")
    best_ckpt = None; best_vm = None; best_step = None
    train_log = []; eval_log = []

    model.train()
    t0 = time.time()
    for step, idx in enumerate(sampler):
        batch  = _to_device(_make_batch(train_data, idx, M), device)
        gold_t = torch.from_numpy(train_data["gold"][idx].astype(np.int64)).to(device)
        bc_mask= torch.from_numpy(buckets["base_correct"][idx]).to(device)

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast("cuda"):
                final, delta, gate = model(**batch)
                loss, ld = compute_loss(final, batch["topk_ids"].long(), gold_t, bc_mask, delta, gate, args)
            if torch.isnan(loss): raise RuntimeError(f"[{variant}] NaN at step {step}: {ld}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            final, delta, gate = model(**batch)
            loss, ld = compute_loss(final, batch["topk_ids"].long(), gold_t, bc_mask, delta, gate, args)
            if torch.isnan(loss): raise RuntimeError(f"[{variant}] NaN at step {step}: {ld}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_log.append({"step": step+1, "variant": variant,
                           **{f"loss_{k}": v for k, v in ld.items()}})

        if (step + 1) % args.eval_every == 0 or step == 0:
            model.eval()
            vm = _eval_model(model, val_data, M, device, args)
            model.train()
            nll = vm["candidate_nll_given_gold_in_topM"]
            elapsed = time.time() - t0
            print(f"[{variant}] step={step+1:5d}  loss={ld['total']:.4f}  "
                  f"val_nll={nll:.5f}  all_acc={vm['all_row_model_acc']:.4f}  "
                  f"ctg={vm['changed_to_gold']:,}  caw={vm['changed_away']:,}  "
                  f"gate={vm['gate_mean']:.3f}  {elapsed:.0f}s")
            eval_log.append({"step": step+1, "variant": variant, **vm})
            if nll < best_nll and nll < base_nll_threshold:
                best_nll = nll; best_ckpt = deepcopy(model.state_dict())
                best_vm = deepcopy(vm); best_step = step + 1
                print(f"  → new best: nll={nll:.5f}")

    model.eval()
    final_vm = _eval_model(model, val_data, M, device, args)
    eval_log.append({"step": args.steps, "variant": variant, **final_vm})

    ckpt_path = None
    if best_ckpt is not None:
        ckpt_path = os.path.join(out_dir, f"best_{variant}.pt")
        torch.save({"state_dict": best_ckpt, "variant": variant, "val_nll": best_nll}, ckpt_path)
        print(f"[{variant}] saved best: {ckpt_path}  nll={best_nll:.5f}")
    else:
        print(f"[{variant}] no checkpoint beat base_nll={base_nll_threshold:.5f}")

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
# 11. Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_model(model, data, M, device, args, batch_size=512, is_base=False):
    N    = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk_ids"].astype(np.int64)
    lgt  = data["topk_lgt"].astype(np.float32)
    ar   = np.arange(N)

    if is_base or model is None:
        pred_scores = lgt.copy()
        gate_arr    = np.ones((N, M), dtype=np.float32)
        delta_arr   = np.zeros((N, M), dtype=np.float32)
    else:
        pred_scores = np.empty((N, M), dtype=np.float32)
        gate_arr    = np.empty((N, M), dtype=np.float32)
        delta_arr   = np.empty((N, M), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, N, batch_size):
                e   = min(s + batch_size, N)
                idx = np.arange(s, e)
                b   = _to_device(_make_batch(data, idx, M), device)
                fin, delt, gat = model(**b)
                pred_scores[s:e] = fin.cpu().float().numpy()
                # gate may be [B,1] or [B,M]
                g_np = gat.cpu().float().numpy()
                if g_np.ndim == 1: g_np = g_np[:, None].repeat(M, axis=1)
                if g_np.shape[-1] == 1: g_np = np.repeat(g_np, M, axis=-1)
                gate_arr[s:e]    = g_np
                delta_arr[s:e]   = delt.cpu().float().numpy()

    # Predictions
    pred_top1_rank = pred_scores.argmax(axis=1)
    pred_top1_id   = topk[ar, pred_top1_rank]
    base_top1_id   = topk[:, 0]

    gip       = (topk == gold[:, None]).any(axis=1)
    gold_rank = np.where(gip, (topk == gold[:, None]).argmax(axis=1), -1)

    # NLL given gold in topM
    ps_d = pred_scores.astype(np.float64)
    ps_d -= ps_d.max(1, keepdims=True)
    m_logsm = ps_d - np.log(np.exp(ps_d).sum(1, keepdims=True) + _EPS)
    nll_arr = np.full(N, float("nan"))
    if gip.any():
        nll_arr[gip] = -m_logsm[gip, np.clip(gold_rank[gip], 0, M-1)]
    cand_nll = _nanmean(nll_arr[gip])

    # Base NLL
    bd = lgt.astype(np.float64)
    bd -= bd.max(1, keepdims=True)
    b_logsm = bd - np.log(np.exp(bd).sum(1, keepdims=True) + _EPS)
    bnll_arr = np.full(N, float("nan"))
    if gip.any():
        bnll_arr[gip] = -b_logsm[gip, np.clip(gold_rank[gip], 0, M-1)]
    base_cand_nll = _nanmean(bnll_arr[gip])

    cand_acc      = float((pred_top1_id[gip] == gold[gip]).mean()) if gip.any() else float("nan")
    base_cand_acc = float((base_top1_id[gip] == gold[gip]).mean()) if gip.any() else float("nan")

    # All-row metrics
    base_correct  = (base_top1_id == gold)
    all_row_base  = float(base_correct.mean())
    all_row_model = float((pred_top1_id == gold).mean())

    # Correction
    ctg  = int((~base_correct & (pred_top1_id == gold)).sum())
    caw  = int((base_correct  & (pred_top1_id != gold)).sum())
    nc   = int((pred_top1_id == base_top1_id).sum())
    w2w  = int((~base_correct & (pred_top1_id != gold) & (pred_top1_id != base_top1_id)).sum())
    net  = ctg - caw
    bdr  = ctg / max(caw, 1)
    appr = float((pred_top1_id != base_top1_id).mean())
    bcd  = caw / max(int(base_correct.sum()), 1)

    # Gate / delta stats
    gate_mean      = float(_nanmean(gate_arr))
    gate_max       = float(np.nanmax(gate_arr)) if gate_arr.size > 0 else float("nan")
    delta_abs_mean = float(_nanmean(np.abs(delta_arr)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        delta_abs_max = float(np.nanmax(np.abs(delta_arr))) if delta_arr.size > 0 else float("nan")

    return {
        "candidate_nll_given_gold_in_topM":   float(cand_nll),
        "candidate_acc_given_gold_in_topM":   float(cand_acc),
        "base_candidate_nll_given_gold_in_topM": float(base_cand_nll),
        "base_candidate_acc_given_gold_in_topM": float(base_cand_acc),
        "nll_gain_vs_base": float(base_cand_nll - cand_nll)
                             if (base_cand_nll == base_cand_nll and cand_nll == cand_nll)
                             else float("nan"),
        "acc_gain_vs_base_given_gold_in_topM": float(cand_acc - base_cand_acc)
                                                if (cand_acc == cand_acc and base_cand_acc == base_cand_acc)
                                                else float("nan"),
        "all_row_base_acc":          all_row_base,
        "all_row_model_acc":         all_row_model,
        "all_row_acc_gain":          all_row_model - all_row_base,
        "natural_gold_in_topM_rate": float(gip.mean()),
        "not_in_pool_rate":          float(1.0 - gip.mean()),
        "changed_to_gold":           ctg,
        "changed_away":              caw,
        "wrong_to_wrong_change":     w2w,
        "no_change":                 nc,
        "net_correction":            net,
        "benefit_damage_ratio":      float(bdr),
        "apply_rate":                float(appr),
        "base_correct_damage_rate":  float(bcd),
        "gate_mean":                 gate_mean,
        "gate_max":                  gate_max,
        "delta_abs_mean":            delta_abs_mean,
        "delta_abs_max":             delta_abs_max,
        "coordinate_entropy_mean":   float("nan"),
        "coordinate_margin_mean":    float("nan"),
        "region_attention_entropy_mean": float("nan"),
    }


def eval_slices(model_or_none, data, tok_arr, reg_arr, M, device, args, batch_size=512):
    """Always uses real tok_arr for slice definitions."""
    N    = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk_ids"].astype(np.int64)
    lgt  = data["topk_lgt"].astype(np.float32)
    ar   = np.arange(N)
    Vt = len(tok_arr); Rr = len(reg_arr)

    base_top1    = topk[:, 0]
    base_correct = (base_top1 == gold)
    gip          = (topk == gold[:, None]).any(axis=1)
    gold_rank_a  = np.where(gip, (topk == gold[:, None]).argmax(axis=1), -1)

    if model_or_none is not None:
        pred_scores = np.empty((N, M), dtype=np.float32)
        model_or_none.eval()
        with torch.no_grad():
            for s in range(0, N, batch_size):
                e   = min(s + batch_size, N)
                idx = np.arange(s, e)
                b   = _to_device(_make_batch(data, idx, M), device)
                f, _, _ = model_or_none(**b)
                pred_scores[s:e] = f.cpu().float().numpy()
        pred_top1 = topk[ar, pred_scores.argmax(axis=1)]
    else:
        pred_scores = lgt.copy().astype(np.float64)
        pred_top1   = base_top1.copy()

    # Real region labels (always)
    base_reg  = tok_arr[np.clip(base_top1, 0, Vt-1)]
    gold_reg  = tok_arr[np.clip(gold,      0, Vt-1)]
    base_sreg = reg_arr[np.clip(base_reg,  0, Rr-1)]
    gold_sreg = reg_arr[np.clip(gold_reg,  0, Rr-1)]
    unk_r = int(tok_arr.max()); unk_s = int(reg_arr.max())
    same_reg  = (base_reg == gold_reg)  & (gold_reg  != unk_r)
    same_sreg = (base_sreg == gold_sreg) & (gold_sreg != unk_s)

    slices = {
        "all":                    np.ones(N, bool),
        "gold_in_topM":           gip,
        "base_correct":           base_correct,
        "bucketA":                ~base_correct & gip,
        "same_region_confuser":   ~base_correct & gip & same_reg,
        "same_superregion_conf":  ~base_correct & gip & same_sreg,
        "different_region_conf":  ~base_correct & gip & ~same_reg,
        "gold_rank_1_5":          gip & (gold_rank_a >= 0) & (gold_rank_a < 5),
        "gold_rank_6_32":         gip & (gold_rank_a >= 5) & (gold_rank_a < 32),
        "gold_rank_33_M":         gip & (gold_rank_a >= 32),
    }

    def _sm(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "base_acc": float("nan"), "model_acc": float("nan"),
                    "acc_gain": float("nan"), "base_nll": float("nan"),
                    "model_nll": float("nan"), "nll_gain": float("nan"),
                    "changed_to_gold": 0, "changed_away": 0, "net_correction": 0}
        b_acc = float(base_correct[mask].mean())
        m_acc = float((pred_top1[mask] == gold[mask]).mean())
        ctg   = int((~base_correct[mask] & (pred_top1[mask] == gold[mask])).sum())
        caw   = int((base_correct[mask]  & (pred_top1[mask] != gold[mask])).sum())
        gsl = gip & mask; ng = int(gsl.sum())
        if ng > 0:
            ps = pred_scores[gsl].astype(np.float64)
            ps -= ps.max(1, keepdims=True)
            ls  = ps - np.log(np.exp(ps).sum(1, keepdims=True) + _EPS)
            gr  = gold_rank_a[gsl]
            m_nll = float(-ls[np.arange(ng), np.clip(gr, 0, M-1)].mean())
            lb = lgt[gsl].astype(np.float64)
            lb -= lb.max(1, keepdims=True)
            bl  = lb - np.log(np.exp(lb).sum(1, keepdims=True) + _EPS)
            b_nll = float(-bl[np.arange(ng), np.clip(gr, 0, M-1)].mean())
        else:
            m_nll = b_nll = float("nan")
        return {"n": n, "base_acc": b_acc, "model_acc": m_acc, "acc_gain": m_acc - b_acc,
                "base_nll": b_nll, "model_nll": m_nll,
                "nll_gain": (b_nll - m_nll) if (b_nll == b_nll and m_nll == m_nll) else float("nan"),
                "changed_to_gold": ctg, "changed_away": caw, "net_correction": ctg - caw}

    return {sn: _sm(mk) for sn, mk in slices.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Comparison table + report
# ═══════════════════════════════════════════════════════════════════════════════

def build_comparison_row(variant, vm, slice_m, base_vm, selected_for_comparison="base",
                         no_improving_ckpt=False):
    def _v(k): return vm.get(k, float("nan"))
    def _bv(k): return base_vm.get(k, float("nan"))
    def _gain(a, b): return (a - b) if (a == a and b == b) else float("nan")
    return {
        "variant":                           variant,
        "selected_for_comparison":           selected_for_comparison,
        "no_improving_checkpoint_found":     no_improving_ckpt,
        "candidate_nll_given_gold_in_topM":  _v("candidate_nll_given_gold_in_topM"),
        "candidate_acc_given_gold_in_topM":  _v("candidate_acc_given_gold_in_topM"),
        "nll_gain_vs_base":                  _gain(_bv("candidate_nll_given_gold_in_topM"),
                                                   _v("candidate_nll_given_gold_in_topM")),
        "acc_gain_vs_base_given_gold_in_topM": _gain(_v("candidate_acc_given_gold_in_topM"),
                                                      _bv("candidate_acc_given_gold_in_topM")),
        "all_row_base_acc":                  _v("all_row_base_acc"),
        "all_row_model_acc":                 _v("all_row_model_acc"),
        "all_row_acc_gain":                  _v("all_row_acc_gain"),
        "natural_gold_in_topM_rate":         _v("natural_gold_in_topM_rate"),
        "not_in_pool_rate":                  _v("not_in_pool_rate"),
        "changed_to_gold":                   _v("changed_to_gold"),
        "changed_away":                      _v("changed_away"),
        "net_correction":                    _v("net_correction"),
        "benefit_damage_ratio":              _v("benefit_damage_ratio"),
        "apply_rate":                        _v("apply_rate"),
        "base_correct_damage_rate":          _v("base_correct_damage_rate"),
        "gate_mean":                         _v("gate_mean"),
        "delta_abs_mean":                    _v("delta_abs_mean"),
        "delta_abs_max":                     _v("delta_abs_max"),
        "bucketA_acc_gain":                  slice_m.get("bucketA", {}).get("acc_gain", float("nan")),
        "same_region_conf_acc_gain":         slice_m.get("same_region_confuser", {}).get("acc_gain", float("nan")),
        "same_sreg_conf_acc_gain":           slice_m.get("same_superregion_conf", {}).get("acc_gain", float("nan")),
    }


def write_report(args, comparison, all_slice_metrics, out_dir):
    def _v(r, k): return r.get(k, float("nan"))
    def _fmt(v):
        if isinstance(v, bool): return str(v)
        if isinstance(v, int):  return str(v)
        if isinstance(v, float): return f"{v:.4f}" if v == v else "nan"
        return str(v)

    ctx_real  = next((r for r in comparison if r["variant"] == "contextual_all_region_real"), None)
    logit_row = next((r for r in comparison if r["variant"] == "logit_only_mlp"), None)
    static_row= next((r for r in comparison if r["variant"] == "static_region_embedding"), None)
    base_row  = next((r for r in comparison if r["variant"] == "base_only"), None)
    shuf_rows = [r for r in comparison if "shuffled" in r["variant"] or "random" in r["variant"]]

    def _nll(r): return _v(r, "candidate_nll_given_gold_in_topM") if r else float("nan")
    ctx_nll  = _nll(ctx_real)
    base_nll = _nll(base_row)
    logit_nll= _nll(logit_row)
    static_nll = _nll(static_row)
    shuf_nll = max((_nll(r) for r in shuf_rows), default=float("nan"))

    def _ok(a, b): return a == a and b == b and a < b - 0.0001

    beats_base   = _ok(ctx_nll, base_nll)
    beats_logit  = _ok(ctx_nll, logit_nll)
    beats_static = _ok(ctx_nll, static_nll)
    beats_shuf   = _ok(ctx_nll, shuf_nll) if shuf_rows else True
    ctg          = int(_v(ctx_real, "changed_to_gold")) if ctx_real else 0
    caw          = int(_v(ctx_real, "changed_away"))    if ctx_real else 0
    ctg_gt_caw   = ctg > caw
    bc_dmg       = _v(ctx_real, "base_correct_damage_rate") if ctx_real else 1.0
    bc_ok        = bc_dmg < args.max_base_correct_damage_rate

    proceed = beats_base and beats_logit and beats_static and beats_shuf and ctg_gt_caw and bc_ok
    verdict = "PROCEED_TO_PHASE_3" if proceed else "DO_NOT_PROCEED_TO_PHASE_3"

    rpt = os.path.join(out_dir, "phase2_contextual_all_region_report.md")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("# Phase 2: Contextual All-Region Coordinate Mixer — Report\n\n")
        f.write(f"**selected_M:** {args.selected_M}  |  **steps:** {args.steps}  |  "
                f"**model_dim:** {args.model_dim}  |  **seed:** {args.seed}\n\n---\n\n")

        f.write("## Comparison Table\n\n")
        cols = ["variant", "selected_for_comparison",
                "candidate_nll_given_gold_in_topM", "nll_gain_vs_base",
                "all_row_model_acc", "all_row_acc_gain",
                "changed_to_gold", "changed_away", "benefit_damage_ratio",
                "base_correct_damage_rate", "gate_mean", "bucketA_acc_gain"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for row in comparison:
            f.write("| " + " | ".join(_fmt(_v(row, c)) if c not in ("variant","selected_for_comparison")
                                      else str(row.get(c, "")) for c in cols) + " |\n")

        f.write("\n---\n\n## Answers to Phase 2 Questions\n\n")

        def _q(n, q, ans, detail=""):
            f.write(f"### Q{n}: {q}\n\n**{ans}**\n")
            if detail: f.write(f"\n{detail}\n")
            f.write("\n")

        _q(1, "Does contextual all-region real beat base_only?",
           "YES" if beats_base else "NO",
           f"ctx_nll={_fmt(ctx_nll)}  base_nll={_fmt(base_nll)}  gain={_fmt(base_nll - ctx_nll)}")
        _q(2, "Does contextual all-region real beat logit_only_mlp?",
           "YES" if beats_logit else "NO",
           f"ctx_nll={_fmt(ctx_nll)}  logit_nll={_fmt(logit_nll)}  gain={_fmt(logit_nll - ctx_nll)}")
        _q(3, "Does contextual all-region real beat static_region_embedding?",
           "YES" if beats_static else "NO",
           f"ctx_nll={_fmt(ctx_nll)}  static_nll={_fmt(static_nll)}  gain={_fmt(static_nll - ctx_nll)}")
        _q(4, "Does real contextual all-region beat shuffled/random contextual controls?",
           "YES" if beats_shuf else "NO",
           "\n".join(f"- {r['variant']}: nll={_fmt(_nll(r))}" for r in shuf_rows))
        _q(5, "Do same-region/superregion confusers improve?",
           "MIXED",
           "\n".join(f"- {v}: same_reg_gain={_fmt(all_slice_metrics.get(v,{}).get('same_region_confuser',{}).get('acc_gain',float('nan')))}  "
                     f"same_sreg_gain={_fmt(all_slice_metrics.get(v,{}).get('same_superregion_conf',{}).get('acc_gain',float('nan')))}"
                     for v in ["contextual_all_region_real","static_region_embedding"] if v in all_slice_metrics))
        _q(6, "Is changed_to_gold > changed_away?",
           "YES" if ctg_gt_caw else "NO",
           f"changed_to_gold={ctg}  changed_away={caw}  net={ctg - caw}")
        _q(7, "Is base-correct damage controlled?",
           "YES" if bc_ok else "NO",
           f"bc_damage={_fmt(bc_dmg)}  threshold={args.max_base_correct_damage_rate}")
        _q(8, "Should we proceed to Phase 3 ablations?",
           verdict,
           "" if proceed else "Failure reasons: " + ", ".join(
               (["contextual_does_not_beat_base"] if not beats_base else []) +
               (["contextual_does_not_beat_logit_only"] if not beats_logit else []) +
               (["contextual_does_not_beat_static"] if not beats_static else []) +
               (["contextual_does_not_beat_shuffled"] if not beats_shuf else []) +
               (["changed_away_ge_changed_to_gold"] if not ctg_gt_caw else []) +
               (["base_correct_damage_too_high"] if not bc_ok else [])))

        f.write("---\n\n## PHASE 2 VERDICT\n\n```\n")
        for row in comparison:
            f.write(f"{row['variant']:35s} "
                    f"nll_gain={_fmt(_v(row,'nll_gain_vs_base'))}  "
                    f"all_acc={_fmt(_v(row,'all_row_model_acc'))}  "
                    f"ctg={row.get('changed_to_gold',0):6d}  caw={row.get('changed_away',0):6d}  "
                    f"BDR={_fmt(_v(row,'benefit_damage_ratio'))}  "
                    f"bc_dmg={_fmt(_v(row,'base_correct_damage_rate'))}\n")
        f.write(f"\nrecommendation: {verdict}\n```\n")

    print(f"[save] {rpt}")
    return rpt


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 2: Contextual All-Region Coordinate Mixer")
    p.add_argument("--train_dir",       required=True)
    p.add_argument("--val_dir",         required=True)
    p.add_argument("--small_ckpt",      required=True)
    p.add_argument("--token_to_region", required=True)
    p.add_argument("--super_map",       default=None)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--selected_M",      type=int,   default=64)
    p.add_argument("--model_dim",       type=int,   default=256)
    p.add_argument("--region_emb_dim",  type=int,   default=64)
    p.add_argument("--super_emb_dim",   type=int,   default=32)
    p.add_argument("--coord_proj_dim",  type=int,   default=64)
    p.add_argument("--hidden_dim",      type=int,   default=256)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--steps",           type=int,   default=5000)
    p.add_argument("--eval_every",      type=int,   default=500)
    p.add_argument("--batch_size",      type=int,   default=128)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--lambda_delta",    type=float, default=1e-4)
    p.add_argument("--lambda_gate",     type=float, default=1e-3)
    p.add_argument("--lambda_preserve", type=float, default=0.5)
    p.add_argument("--max_base_correct_damage_rate", type=float, default=0.05)
    p.add_argument("--max_train_rows",  type=int,   default=None)
    p.add_argument("--max_val_rows",    type=int,   default=None)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--use_gate",        action="store_true", default=True)
    p.add_argument("--no_gate",         action="store_true", default=False)
    p.add_argument("--gate_init_bias",  type=float, default=0.0)
    p.add_argument("--use_router_prior_bias", action="store_true", default=False)
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--skip_random_control", action="store_true")
    args = p.parse_args()

    # Resolve gate
    args.use_gate = args.use_gate and not args.no_gate

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    t0 = time.time()

    # ── 1. Load shards ────────────────────────────────────────────────────────
    print("\n[step 1] loading shards...")
    M          = args.selected_M
    train_data = load_shards(args.train_dir, M, args.max_train_rows, "train")
    val_data   = load_shards(args.val_dir,   M, args.max_val_rows,   "val")
    has_router = train_data["has_router"]
    d_model    = train_data.get("d_model", 1) or 1
    if not train_data["has_h"]:
        print("[WARN] No hidden state in shards — h will be zeros; contextual model loses main signal.")

    # ── 2. Unembedding ────────────────────────────────────────────────────────
    print("\n[step 2] loading unembedding...")
    U, uinfo = load_unembedding(args.small_ckpt)
    print(f"[unembedding] shape={tuple(U.shape)}  source={uinfo['unembedding_source']}")
    with open(os.path.join(args.output_dir, "unembedding_audit.json"), "w") as f:
        json.dump(uinfo, f, indent=2)

    # ── 3. Region maps ────────────────────────────────────────────────────────
    print("\n[step 3] loading region maps...")
    tok_arr, reg_arr, n_regions, n_super = load_region_maps(
        args.token_to_region, args.super_map)
    shuf_tok = _shuffle_tok_arr(tok_arr, n_regions, args.seed)
    rand_tok = _random_tok_arr(tok_arr,  n_regions, args.seed)

    def _save_map(arr, name):
        path = os.path.join(args.output_dir, name)
        json.dump({str(i): int(v) for i, v in enumerate(arr) if int(v) < n_regions},
                  open(path, "w"), indent=2)
        print(f"[save] {path}")
    _save_map(shuf_tok, "shuffled_token_to_region.json")
    _save_map(rand_tok, "random_token_to_region.json")

    # ── 4. Buckets ────────────────────────────────────────────────────────────
    print("\n[step 4] computing bucket labels...")
    train_buckets = compute_bucket_labels(train_data, tok_arr, reg_arr)

    # ── 5. Config ─────────────────────────────────────────────────────────────
    cfg = {**vars(args),
           "n_regions": n_regions, "n_super": n_super,
           "train_rows": len(train_data["gold"]), "val_rows": len(val_data["gold"]),
           "has_router": bool(has_router), "has_h": bool(train_data["has_h"]),
           "d_model": d_model, "U_shape": list(U.shape), "device": str(device)}
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # ── 6. Base-only ─────────────────────────────────────────────────────────
    print("\n[step 5] evaluating base_only...")
    base_vm     = _eval_model(None, val_data, M, device, args, is_base=True)
    base_slices = eval_slices(None, val_data, tok_arr, reg_arr, M, device, args)
    print(f"  base_only: nll={base_vm['candidate_nll_given_gold_in_topM']:.5f}  "
          f"all_acc={base_vm['all_row_base_acc']:.4f}  "
          f"gip={base_vm['natural_gold_in_topM_rate']:.4f}")

    all_comp_rows  = []
    all_slice_m    = {"base_only": base_slices}
    all_train_logs = []
    all_eval_logs  = []
    all_best_info  = {}
    all_final_info = {}

    all_comp_rows.append(build_comparison_row(
        "base_only", base_vm, base_slices, base_vm, "base", False))

    out_dir = args.output_dir

    def _make_model(variant, tok_arr_v):
        if variant == "logit_only_mlp":
            return LogitOnlyMixer(args.hidden_dim, args.dropout,
                                  args.use_gate, args.gate_init_bias).to(device)
        elif variant == "static_region_embedding":
            return StaticRegionEmbMixer(
                tok_arr_v, reg_arr, n_regions, n_super,
                args.hidden_dim, args.dropout, has_router,
                args.use_gate, args.gate_init_bias).to(device)
        else:
            return ContextualAllRegionCoordinateMixer(
                tok_arr=tok_arr_v, reg_arr=reg_arr, U_frozen=U,
                n_regions=n_regions, n_super=n_super, d_model=d_model,
                has_router=has_router,
                model_dim=args.model_dim,
                region_emb_dim=args.region_emb_dim,
                super_emb_dim=args.super_emb_dim,
                coord_proj_dim=args.coord_proj_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                no_gate=args.no_gate,
                gate_init_bias=args.gate_init_bias,
            ).to(device)

    def _run(variant, tok_arr_v):
        print(f"\n{'='*60}\n[variant] {variant}\n{'='*60}")
        model = _make_model(variant, tok_arr_v)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{variant}] parameters: {n_params:,}")
        check_identity(model, val_data, M, device, variant)

        result = train_variant(variant, model, train_data, val_data,
                               train_buckets, M, args, device, out_dir)

        bm = result["best"]
        fm = result["final"]
        vm_for_comp = bm if not bm["no_improving_checkpoint_found"] else fm

        # Slices always use real tok_arr
        model.eval()
        slices = eval_slices(model, val_data, tok_arr, reg_arr, M, device, args)

        comp_row = build_comparison_row(
            variant, vm_for_comp, slices, base_vm,
            bm["selected_for_comparison"],
            bm["no_improving_checkpoint_found"])
        all_comp_rows.append(comp_row)
        all_slice_m[variant]   = slices
        all_train_logs.extend(result["train_log"])
        all_eval_logs.extend(result["eval_log"])
        all_best_info[variant] = bm
        all_final_info[variant]= fm

        print(f"[{variant}] done: nll={vm_for_comp['candidate_nll_given_gold_in_topM']:.5f}  "
              f"all_acc={vm_for_comp['all_row_model_acc']:.4f}  "
              f"ctg={vm_for_comp['changed_to_gold']:,}  caw={vm_for_comp['changed_away']:,}")

    # ── 7. Train variants ─────────────────────────────────────────────────────
    for v in ["logit_only_mlp", "static_region_embedding",
              "contextual_all_region_real"]:
        _run(v, tok_arr)
    _run("contextual_all_region_shuffled", shuf_tok)
    if not args.skip_random_control:
        _run("contextual_all_region_random", rand_tok)

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    _wcsv(os.path.join(out_dir, "phase2_comparison.csv"), all_comp_rows)
    _wcsv(os.path.join(out_dir, "train_log.csv"),         all_train_logs)
    _wcsv(os.path.join(out_dir, "eval_log.csv"),          all_eval_logs)

    slice_rows = [{"variant": vn, "slice": sn, **vals}
                  for vn, sm in all_slice_m.items()
                  for sn, vals in sm.items()]
    _wcsv(os.path.join(out_dir, "slice_metrics.csv"), slice_rows)

    with open(os.path.join(out_dir, "best_metrics.json"), "w") as f:
        json.dump(all_best_info, f, indent=2, default=str)
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump({"base_only": base_vm, **all_final_info}, f, indent=2, default=str)

    # ── 9. Report ─────────────────────────────────────────────────────────────
    rpt = write_report(args, all_comp_rows, all_slice_m, out_dir)

    # ── 10. Verdict ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(" PHASE 2 VERDICT:")
    print(f"{'='*60}")
    for row in all_comp_rows:
        ng = row.get("nll_gain_vs_base", float("nan"))
        aa = row.get("all_row_model_acc", float("nan"))
        ctg= row.get("changed_to_gold", 0)
        caw= row.get("changed_away", 0)
        print(f"  {row['variant']:35s}  nll_gain={ng:.4f}  all_acc={aa:.4f}  ctg={ctg:,}  caw={caw:,}")

    ctx_real  = next((r for r in all_comp_rows if r["variant"] == "contextual_all_region_real"), None)
    logit_row = next((r for r in all_comp_rows if r["variant"] == "logit_only_mlp"), None)
    static_row= next((r for r in all_comp_rows if r["variant"] == "static_region_embedding"), None)
    base_row  = next((r for r in all_comp_rows if r["variant"] == "base_only"), None)
    shuf_rows = [r for r in all_comp_rows if "shuffled" in r["variant"] or "random" in r["variant"]]

    def _nll(r): return r.get("candidate_nll_given_gold_in_topM", float("nan")) if r else float("nan")
    def _ok(a, b): return a == a and b == b and a < b - 0.0001

    beats_base   = _ok(_nll(ctx_real), _nll(base_row))
    beats_logit  = _ok(_nll(ctx_real), _nll(logit_row))
    beats_static = _ok(_nll(ctx_real), _nll(static_row))
    shuf_nll     = max((_nll(r) for r in shuf_rows), default=float("nan"))
    beats_shuf   = _ok(_nll(ctx_real), shuf_nll) if shuf_rows else True
    ctg          = int(ctx_real.get("changed_to_gold", 0)) if ctx_real else 0
    caw          = int(ctx_real.get("changed_away",    0)) if ctx_real else 0
    bc_dmg       = float(ctx_real.get("base_correct_damage_rate", 1.0)) if ctx_real else 1.0
    proceed      = (beats_base and beats_logit and beats_static and beats_shuf
                    and ctg > caw and bc_dmg < args.max_base_correct_damage_rate)
    verdict      = "PROCEED_TO_PHASE_3" if proceed else "DO_NOT_PROCEED_TO_PHASE_3"

    if not proceed:
        reasons = []
        if not beats_base:   reasons.append("contextual_does_not_beat_base")
        if not beats_logit:  reasons.append("contextual_does_not_beat_logit_only")
        if not beats_static: reasons.append("contextual_does_not_beat_static")
        if not beats_shuf:   reasons.append("contextual_does_not_beat_shuffled")
        if ctg <= caw:       reasons.append("changed_away_ge_changed_to_gold")
        if bc_dmg >= args.max_base_correct_damage_rate:
            reasons.append("base_correct_damage_too_high")
        print(f"\n  failure reasons: {', '.join(reasons)}")

    print(f"\n  recommendation: {verdict}")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"  report:  {rpt}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
