#!/usr/bin/env python3
"""
train_static_region_feature_mixer.py — Phase 1: Static Region Feature Mixer

Purpose:
  Test whether static region/superregion/router-region features provide real
  predictive signal beyond:
    1) base logits
    2) a logit-only MLP recalibration control
    3) shuffled/random region controls

Variants:
  base_only
  logit_only_mlp
  hard_region_features
  static_region_embedding
  shuffled_region_control
  random_region_control

Safety:
  - Gold is used only for loss/metrics.
  - No gold force-inclusion.
  - No gold_region input.
  - Step-0 identity must pass for every trainable variant.
  - Evaluation slices always use the REAL region map.
  - Best checkpoint saved only if validation candidate NLL beats base.
  - Best and final metrics are kept separate.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)

_EPS = 1e-9
VS = 50257
EMB_R = 32
EMB_S = 16
GATE_IN = 4

_TOPK = ["base_topk_ids", "base_topk", "topk_ids"]
_LGT = ["base_topk_logits", "base_topk_lgt", "topk_lgt", "topk_logits"]
_GOLD = ["gold_token", "gold", "labels"]
_RID = ["row_id", "row_ids"]
_OFF = ["token_offset", "offsets", "offset"]
_RREG = ["router_topk_reg"]
_RPRB = ["router_topk_prb"]
_RMRG = ["router_margin"]


def _get(sh: Dict[str, Any], aliases: List[str], required: bool = True):
    for a in aliases:
        if a in sh:
            return sh[a]
    if required:
        raise KeyError(f"Need one of {aliases}; shard has {list(sh.keys())}")
    return None


def _nanmean(arr) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(a))


def _wcsv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # union fieldnames, preserving first-row order then appending new keys
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for r in rows[1:]:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {path}")


def _json_dump(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[save] {path}")


# =============================================================================
# 1. Shard loading
# =============================================================================


def load_shards(shard_dir: str, selected_M: int, max_rows: Optional[int] = None, label: str = "") -> Dict[str, Any]:
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")
    print(f"[shards] {label}: {len(paths)} shards in {shard_dir}")
    bufs = defaultdict(list)
    total = 0
    first = True
    final_M = None

    for sp in paths:
        if max_rows and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print(f"[shards] first shard keys: {list(sh.keys())}")
            first = False

        topk = _get(sh, _TOPK).long()
        lgt = _get(sh, _LGT).float()
        gold = _get(sh, _GOLD).long()
        rids = _get(sh, _RID, required=False)
        off = _get(sh, _OFF, required=False)
        rreg = _get(sh, _RREG, required=False)
        rprb = _get(sh, _RPRB, required=False)
        rmrg = _get(sh, _RMRG, required=False)

        B, K = topk.shape
        M = min(K, selected_M)
        if final_M is None:
            final_M = M
        elif M != final_M:
            raise RuntimeError(f"Inconsistent M across shards: got {M}, expected {final_M}")

        topk = topk[:, :M]
        lgt = lgt[:, :M]
        if rids is None:
            rids = torch.arange(total, total + B)
        if off is None:
            off = torch.full((B,), -1, dtype=torch.long)

        if max_rows and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, rids, off = (x[:keep] for x in (topk, lgt, gold, rids, off))
            if rreg is not None:
                rreg = rreg[:keep]
            if rprb is not None:
                rprb = rprb[:keep]
            if rmrg is not None:
                rmrg = rmrg[:keep]
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
            if rprb is not None:
                bufs["router_prb"].append(rprb.float().numpy().astype(np.float32))
            else:
                bufs["router_prb"].append(np.zeros((B, K2), np.float32))
        if rmrg is not None:
            bufs["router_margin"].append(rmrg.float().numpy().astype(np.float32).reshape(B))

        total += B

    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    data["has_router"] = "router_reg" in data
    data["M"] = int(final_M or selected_M)
    print(f"[shards] {label}: {total:,} rows  M={data['M']}  has_router={data['has_router']}")
    return data


# =============================================================================
# 2. Region maps + controls
# =============================================================================


def load_region_maps(t2r_path: str, super_path: Optional[str]):
    if not t2r_path or not os.path.isfile(t2r_path):
        V = VS
        return np.zeros(V, np.int32), np.zeros(2, np.int32), 1, 1

    with open(t2r_path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        t2r = {i: int(v) for i, v in enumerate(raw) if v is not None and int(v) >= 0}
    else:
        t2r = {int(k): int(v) for k, v in raw.items() if v is not None and int(v) >= 0}

    n_regions = int(max(t2r.values())) + 1 if t2r else 1
    unk_r = n_regions
    V = max(max(t2r.keys(), default=0) + 1, VS)
    tok_arr = np.full(V, unk_r, dtype=np.int32)
    for t, r in t2r.items():
        if 0 <= t < V:
            tok_arr[t] = int(r)

    r2s = {}
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f:
            raw2 = json.load(f)
        if isinstance(raw2, list):
            r2s = {i: int(v) for i, v in enumerate(raw2) if v is not None and int(v) >= 0}
        else:
            r2s = {int(k): int(v) for k, v in raw2.items() if v is not None and int(v) >= 0}
    n_super = int(max(r2s.values())) + 1 if r2s else 1
    unk_s = n_super
    R = max(max(r2s.keys(), default=0) + 1, n_regions + 1)
    reg_arr = np.full(R, unk_s, dtype=np.int32)
    for r, s in r2s.items():
        if 0 <= r < R:
            reg_arr[r] = int(s)

    print(f"[maps] n_regions={n_regions}  n_super={n_super}  V={V}  R={R}")
    return tok_arr, reg_arr, n_regions, n_super


def _shuffle_tok_arr(tok_arr: np.ndarray, n_regions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    valid_tokens = np.where(tok_arr < n_regions)[0]
    valid_regions = tok_arr[valid_tokens].copy()
    rng.shuffle(valid_regions)
    out = tok_arr.copy()
    out[valid_tokens] = valid_regions
    return out


def _random_tok_arr(tok_arr: np.ndarray, n_regions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1729)
    valid_tokens = np.where(tok_arr < n_regions)[0]
    valid_regions = tok_arr[valid_tokens]
    region_sizes = np.bincount(valid_regions, minlength=n_regions)
    new_assignments = np.repeat(np.arange(n_regions, dtype=np.int32), region_sizes)
    rng.shuffle(new_assignments)
    out = tok_arr.copy()
    out[valid_tokens] = new_assignments
    return out


def save_token_region_map(arr: np.ndarray, n_regions: int, path: str):
    payload = {str(i): int(v) for i, v in enumerate(arr) if int(v) < n_regions}
    _json_dump(payload, path)


# =============================================================================
# 3. Buckets
# =============================================================================


def compute_bucket_labels(data: Dict[str, Any], tok_arr: np.ndarray, reg_arr: np.ndarray) -> Dict[str, np.ndarray]:
    gold = data["gold"].astype(np.int64)
    topk_ids = data["topk_ids"].astype(np.int64)
    Vt, Rr = len(tok_arr), len(reg_arr)

    base_top1 = topk_ids[:, 0]
    base_correct = base_top1 == gold
    gip_M = (topk_ids == gold[:, None]).any(axis=1)
    bucketA = (~base_correct) & gip_M

    base_reg = tok_arr[np.clip(base_top1, 0, Vt - 1)]
    gold_reg = tok_arr[np.clip(gold, 0, Vt - 1)]
    base_sreg = reg_arr[np.clip(base_reg, 0, Rr - 1)]
    gold_sreg = reg_arr[np.clip(gold_reg, 0, Rr - 1)]

    unk_r = int(tok_arr.max())
    unk_s = int(reg_arr.max())
    same_reg = (base_reg == gold_reg) & (gold_reg != unk_r)
    same_sreg = (base_sreg == gold_sreg) & (gold_sreg != unk_s)

    same_reg_conf = bucketA & same_reg
    same_sreg_conf = bucketA & same_sreg
    same_reg_or_sreg_conf = bucketA & (same_reg | same_sreg)

    print(
        f"[buckets] base_correct={base_correct.sum():,}  "
        f"bucketA={bucketA.sum():,}  gip_M={gip_M.sum():,}  "
        f"same_reg_conf={same_reg_conf.sum():,}  same_sreg_conf={same_sreg_conf.sum():,}"
    )
    return {
        "base_correct": base_correct,
        "bucketA": bucketA,
        "gip_M": gip_M,
        "same_reg_conf": same_reg_conf,
        "same_sreg_conf": same_sreg_conf,
        "same_reg_or_sreg_conf": same_reg_or_sreg_conf,
    }


# =============================================================================
# 4. Models
# =============================================================================


class _GateMLP(nn.Module):
    def __init__(self, hidden: int = 32, init_bias: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GATE_IN, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # deterministic gate start: sigmoid(init_bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(init_bias))

    def forward(self, row_feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(row_feats))


class _CandMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, _ = x.shape
        return self.net(x.reshape(B * M, -1)).reshape(B, M)


def _row_feats(topk_lgt: torch.Tensor) -> torch.Tensor:
    M = topk_lgt.shape[1]
    top1 = topk_lgt[:, 0]
    top2 = topk_lgt[:, min(1, M - 1)]
    return torch.stack([top1, top1 - top2, topk_lgt.mean(dim=1), topk_lgt.std(dim=1)], dim=1)


class _MixerBase(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, use_gate: bool, gate_init_bias: float):
        super().__init__()
        self.cand_mlp = _CandMLP(in_dim, hidden, dropout)
        self.use_gate = bool(use_gate)
        self.gate_mlp = _GateMLP(hidden=32, init_bias=gate_init_bias) if self.use_gate else None

    def _gate(self, topk_lgt: torch.Tensor) -> torch.Tensor:
        if not self.use_gate:
            return torch.ones(topk_lgt.shape[0], 1, device=topk_lgt.device, dtype=topk_lgt.dtype)
        return self.gate_mlp(_row_feats(topk_lgt))

    def _finalize(self, topk_lgt: torch.Tensor, feat: torch.Tensor):
        delta = self.cand_mlp(feat)
        gate = self._gate(topk_lgt)
        final = topk_lgt + gate * delta
        return final, delta, gate


class LogitOnlyMixer(_MixerBase):
    """Control: MLP sees only rank/logit/gap/base-margin/row mean/std."""
    IN_DIM = 6

    def __init__(self, hidden: int, dropout: float, use_gate: bool, gate_init_bias: float):
        super().__init__(self.IN_DIM, hidden, dropout, use_gate, gate_init_bias)

    def _build_features(self, topk_ids, topk_lgt, router_reg=None, router_prb=None, router_margin=None):
        B, M = topk_lgt.shape
        rank_frac = (torch.arange(M, device=topk_lgt.device, dtype=torch.float32) / max(M - 1, 1)).view(1, M).expand(B, M)
        gap = topk_lgt[:, 0:1] - topk_lgt
        base_margin = (topk_lgt[:, 0] - topk_lgt[:, min(1, M - 1)]).view(B, 1).expand(B, M)
        row_mean = topk_lgt.mean(dim=1, keepdim=True).expand(B, M)
        row_std = topk_lgt.std(dim=1, keepdim=True).expand(B, M)
        return torch.stack([rank_frac, topk_lgt, gap, base_margin, row_mean, row_std], dim=2)

    def forward(self, topk_ids, topk_lgt, router_reg=None, router_prb=None, router_margin=None):
        return self._finalize(topk_lgt, self._build_features(topk_ids, topk_lgt, router_reg, router_prb, router_margin))


class HardRegionMixer(_MixerBase):
    IN_DIM = 12

    def __init__(self, tok_arr, reg_arr, hidden, dropout, use_gate: bool, gate_init_bias: float):
        super().__init__(self.IN_DIM, hidden, dropout, use_gate, gate_init_bias)
        self.register_buffer("tok_arr", torch.from_numpy(tok_arr).int())
        self.register_buffer("reg_arr", torch.from_numpy(reg_arr).int())

    def _build_features(self, topk_ids, topk_lgt, router_reg=None, router_prb=None, router_margin=None):
        B, M = topk_ids.shape
        Vt = self.tok_arr.shape[0]
        Rr = self.reg_arr.shape[0]

        cand_reg = self.tok_arr[topk_ids.clamp(0, Vt - 1).long()]
        cand_sreg = self.reg_arr[cand_reg.clamp(0, Rr - 1).long()]
        base_reg = cand_reg[:, 0]
        base_sreg = cand_sreg[:, 0]

        rank_frac = (torch.arange(M, device=topk_ids.device, dtype=torch.float32) / max(M - 1, 1)).unsqueeze(0).expand(B, M)
        gap = topk_lgt[:, 0:1] - topk_lgt
        same_reg = (cand_reg == base_reg.unsqueeze(1)).float()
        same_sreg = (cand_sreg == base_sreg.unsqueeze(1)).float()
        base_margin = (topk_lgt[:, 0] - topk_lgt[:, min(1, M - 1)]).unsqueeze(1).expand(B, M)
        n_same_reg = (same_reg.sum(dim=1, keepdim=True) / M).expand(B, M)

        if router_reg is not None:
            K = router_reg.shape[1]
            router_prb = router_prb if router_prb is not None else torch.zeros(B, K, device=topk_ids.device, dtype=topk_lgt.dtype)
            rt1_reg = router_reg[:, 0].long()
            rt1_sreg = self.reg_arr[rt1_reg.clamp(0, Rr - 1).long()]
            same_reg_rt = (cand_reg == rt1_reg.unsqueeze(1)).float()
            same_sreg_rt = (cand_sreg == rt1_sreg.unsqueeze(1)).float()
            match_bkm = router_reg.long().unsqueeze(2) == cand_reg.long().unsqueeze(1)
            router_prob_cand = (router_prb.unsqueeze(2) * match_bkm.float()).sum(dim=1)
            k_idx = torch.arange(K, device=topk_ids.device, dtype=torch.float32)
            miss_rank = torch.ones(1, 1, 1, device=topk_ids.device) * float(K) / max(K - 1, 1)
            router_rank_cand = torch.where(match_bkm, k_idx.view(1, K, 1) / max(K - 1, 1), miss_rank).min(dim=1).values
            rm = router_margin.unsqueeze(1).expand(B, M) if router_margin is not None else torch.zeros(B, M, device=topk_ids.device)
        else:
            same_reg_rt = torch.zeros(B, M, device=topk_ids.device)
            same_sreg_rt = torch.zeros(B, M, device=topk_ids.device)
            router_prob_cand = torch.zeros(B, M, device=topk_ids.device)
            router_rank_cand = torch.zeros(B, M, device=topk_ids.device)
            rm = torch.zeros(B, M, device=topk_ids.device)

        return torch.stack([
            rank_frac, topk_lgt, gap, same_reg, same_sreg,
            same_reg_rt, same_sreg_rt, router_prob_cand,
            router_rank_cand, rm, n_same_reg, base_margin,
        ], dim=2)

    def forward(self, topk_ids, topk_lgt, router_reg=None, router_prb=None, router_margin=None):
        return self._finalize(topk_lgt, self._build_features(topk_ids, topk_lgt, router_reg, router_prb, router_margin))


class StaticRegionEmbMixer(_MixerBase):
    def __init__(self, tok_arr, reg_arr, n_regions, n_super, hidden, dropout, has_router, use_gate: bool, gate_init_bias: float):
        self.has_router = bool(has_router)
        scalar = 6
        emb_base = 2 * EMB_R + 2 * EMB_S  # cand reg/sreg + cand-base diffs
        router_extra = 5 + 2 * EMB_R if self.has_router else 0
        # router_extra: same_reg_rt, same_sreg_rt, prob, rank, margin, rt1_emb, cand-rt1 diff
        feat_dim = scalar + emb_base + router_extra
        super().__init__(feat_dim, hidden, dropout, use_gate, gate_init_bias)
        self.register_buffer("tok_arr", torch.from_numpy(tok_arr).int())
        self.register_buffer("reg_arr", torch.from_numpy(reg_arr).int())
        self.region_emb = nn.Embedding(n_regions + 2, EMB_R, padding_idx=n_regions)
        self.sreg_emb = nn.Embedding(n_super + 2, EMB_S, padding_idx=n_super)
        self.n_regions = n_regions
        self.n_super = n_super
        self.feat_dim = feat_dim

    def _build_features(self, topk_ids, topk_lgt, router_reg=None, router_prb=None, router_margin=None):
        B, M = topk_ids.shape
        Vt = self.tok_arr.shape[0]
        Rr = self.reg_arr.shape[0]

        cand_reg = self.tok_arr[topk_ids.clamp(0, Vt - 1).long()].long().clamp(0, self.n_regions + 1)
        cand_sreg = self.reg_arr[cand_reg.clamp(0, Rr - 1)].long().clamp(0, self.n_super + 1)
        base_reg = cand_reg[:, 0]
        base_sreg = cand_sreg[:, 0]

        cand_reg_emb = self.region_emb(cand_reg)
        cand_sreg_emb = self.sreg_emb(cand_sreg)
        base_reg_emb = self.region_emb(base_reg)
        base_sreg_emb = self.sreg_emb(base_sreg)

        diff_reg = cand_reg_emb - base_reg_emb.unsqueeze(1)
        diff_sreg = cand_sreg_emb - base_sreg_emb.unsqueeze(1)

        rank_frac = (torch.arange(M, device=topk_ids.device, dtype=torch.float32) / max(M - 1, 1)).view(1, M, 1).expand(B, M, 1)
        gap = (topk_lgt[:, 0:1] - topk_lgt).unsqueeze(-1)
        same_reg = (cand_reg == base_reg.unsqueeze(1)).float().unsqueeze(-1)
        same_sreg = (cand_sreg == base_sreg.unsqueeze(1)).float().unsqueeze(-1)
        base_margin = (topk_lgt[:, 0] - topk_lgt[:, min(1, M - 1)]).view(B, 1, 1).expand(B, M, 1)
        lgt_3 = topk_lgt.unsqueeze(-1)

        parts = [rank_frac, lgt_3, gap, same_reg, same_sreg, base_margin, cand_reg_emb, cand_sreg_emb, diff_reg, diff_sreg]

        if router_reg is not None and self.has_router:
            K = router_reg.shape[1]
            router_prb = router_prb if router_prb is not None else torch.zeros(B, K, device=topk_ids.device, dtype=topk_lgt.dtype)
            rt1_reg = router_reg[:, 0].long().clamp(0, self.n_regions + 1)
            rt1_sreg = self.reg_arr[rt1_reg.clamp(0, self.reg_arr.shape[0] - 1)].long().clamp(0, self.n_super + 1)
            rt1_emb = self.region_emb(rt1_reg)
            diff_rt = cand_reg_emb - rt1_emb.unsqueeze(1)
            same_reg_rt = (cand_reg == rt1_reg.unsqueeze(1)).float().unsqueeze(-1)
            same_sreg_rt = (cand_sreg == rt1_sreg.unsqueeze(1)).float().unsqueeze(-1)
            match_bkm = router_reg.long().unsqueeze(2) == cand_reg.long().unsqueeze(1)
            router_prob_cand = (router_prb.unsqueeze(2) * match_bkm.float()).sum(dim=1).unsqueeze(-1)
            k_idx = torch.arange(K, device=topk_ids.device, dtype=torch.float32)
            miss_rank = torch.ones(1, 1, 1, device=topk_ids.device) * float(K) / max(K - 1, 1)
            router_rank_cand = torch.where(match_bkm, k_idx.view(1, K, 1) / max(K - 1, 1), miss_rank).min(dim=1).values.unsqueeze(-1)
            rm = router_margin.view(B, 1, 1).expand(B, M, 1) if router_margin is not None else torch.zeros(B, M, 1, device=topk_ids.device)
            parts.extend([
                same_reg_rt,
                same_sreg_rt,
                router_prob_cand,
                router_rank_cand,
                rm,
                rt1_emb.unsqueeze(1).expand(B, M, EMB_R),
                diff_rt,
            ])

        feat = torch.cat(parts, dim=2)
        if feat.shape[-1] != self.feat_dim:
            raise RuntimeError(f"StaticRegionEmbMixer feat_dim mismatch: got {feat.shape[-1]}, expected {self.feat_dim}")
        return feat

    def forward(self, topk_ids, topk_lgt, router_reg=None, router_prb=None, router_margin=None):
        return self._finalize(topk_lgt, self._build_features(topk_ids, topk_lgt, router_reg, router_prb, router_margin))


# =============================================================================
# 5. Batch helpers / loss
# =============================================================================


def _make_batch(data: Dict[str, Any], idx: np.ndarray, M: int) -> Dict[str, Any]:
    b = {"topk_ids": data["topk_ids"][idx], "topk_lgt": data["topk_lgt"][idx]}
    if data["has_router"]:
        b["router_reg"] = data["router_reg"][idx]
        b["router_prb"] = data["router_prb"][idx]
        if "router_margin" in data:
            b["router_margin"] = data["router_margin"][idx]
    return b


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        out[k] = t.to(device)
    return out


def compute_loss(final_scores, topk_ids_t, gold_t, base_correct_mask, delta, gate, args):
    gip = (topk_ids_t == gold_t.unsqueeze(1)).any(dim=1)
    gold_idx = (topk_ids_t == gold_t.unsqueeze(1)).float().argmax(dim=1)

    loss_ce = torch.tensor(0.0, device=final_scores.device)
    if gip.sum() > 0:
        loss_ce = F.cross_entropy(final_scores[gip], gold_idx[gip])

    loss_delta = delta.pow(2).mean()
    loss_gate = gate.mean()

    loss_preserve = torch.tensor(0.0, device=final_scores.device)
    bc = base_correct_mask.bool()
    if bc.sum() > 0:
        base_idx = torch.zeros(int(bc.sum()), dtype=torch.long, device=final_scores.device)
        loss_preserve = F.cross_entropy(final_scores[bc], base_idx)

    total = loss_ce + args.lambda_delta * loss_delta + args.lambda_gate * loss_gate + args.lambda_preserve * loss_preserve
    return total, {
        "ce": float(loss_ce.item()),
        "delta_l2": float(loss_delta.item()),
        "gate": float(loss_gate.item()),
        "preserve": float(loss_preserve.item()),
        "total": float(total.item()),
        "n_gip": int(gip.sum().item()),
    }


# =============================================================================
# 6. Evaluation
# =============================================================================


def _softmax_nll(scores: np.ndarray, gold_rank: np.ndarray, gip: np.ndarray, M: int) -> float:
    if not gip.any():
        return float("nan")
    x = scores.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    log_sm = x - np.log(np.exp(x).sum(axis=1, keepdims=True) + _EPS)
    gr = np.clip(gold_rank[gip], 0, M - 1)
    return float((-log_sm[gip, gr]).mean())


def _metric_from_scores(pred_scores: np.ndarray, base_scores: np.ndarray, topk: np.ndarray, gold: np.ndarray, delta: Optional[np.ndarray] = None, gate: Optional[np.ndarray] = None) -> Dict[str, Any]:
    N, M = topk.shape
    ar = np.arange(N)
    base_top1_id = topk[:, 0]
    pred_rank = pred_scores.argmax(axis=1)
    pred_top1_id = topk[ar, pred_rank]
    gip = (topk == gold[:, None]).any(axis=1)
    gold_rank = np.where(gip, (topk == gold[:, None]).argmax(axis=1), -1)

    candidate_nll = _softmax_nll(pred_scores, gold_rank, gip, M)
    base_candidate_nll = _softmax_nll(base_scores, gold_rank, gip, M)

    base_correct = base_top1_id == gold
    pred_correct = pred_top1_id == gold

    candidate_acc = float(pred_correct[gip].mean()) if gip.any() else float("nan")
    base_candidate_acc = float(base_correct[gip].mean()) if gip.any() else float("nan")
    all_row_base_acc = float(base_correct.mean())
    all_row_model_acc = float(pred_correct.mean())

    changed_to_gold = int(((~base_correct) & pred_correct).sum())
    changed_away = int((base_correct & (~pred_correct)).sum())
    no_change = int((pred_top1_id == base_top1_id).sum())
    wrong_to_wrong = int(((~base_correct) & (~pred_correct) & (pred_top1_id != base_top1_id)).sum())
    apply_rate = float((pred_top1_id != base_top1_id).mean())
    bc_damage = changed_away / max(int(base_correct.sum()), 1)

    out = {
        "candidate_nll_given_gold_in_topM": float(candidate_nll),
        "candidate_acc_given_gold_in_topM": float(candidate_acc),
        "base_candidate_nll_given_gold_in_topM": float(base_candidate_nll),
        "base_candidate_acc_given_gold_in_topM": float(base_candidate_acc),
        "all_row_base_acc": all_row_base_acc,
        "all_row_model_acc": all_row_model_acc,
        "all_row_acc_gain": all_row_model_acc - all_row_base_acc,
        "natural_gold_in_topM_rate": float(gip.mean()),
        "not_in_pool_rate": float(1.0 - gip.mean()),
        "changed_to_gold": changed_to_gold,
        "changed_away": changed_away,
        "no_change": no_change,
        "wrong_to_wrong": wrong_to_wrong,
        "net_correction": changed_to_gold - changed_away,
        "benefit_damage_ratio": float(changed_to_gold / max(changed_away, 1)),
        "apply_rate": apply_rate,
        "base_correct_damage_rate": float(bc_damage),
    }
    if delta is not None:
        out["delta_abs_mean"] = float(np.mean(np.abs(delta)))
        out["delta_abs_max"] = float(np.max(np.abs(delta)))
    else:
        out["delta_abs_mean"] = 0.0
        out["delta_abs_max"] = 0.0
    if gate is not None:
        out["gate_mean"] = float(np.mean(gate))
    else:
        out["gate_mean"] = 0.0
    return out


def evaluate_base_only(data: Dict[str, Any], M: int) -> Dict[str, Any]:
    topk = data["topk_ids"].astype(np.int64)
    lgt = data["topk_lgt"].astype(np.float32)
    gold = data["gold"].astype(np.int64)
    return _metric_from_scores(lgt, lgt, topk, gold, delta=np.zeros_like(lgt), gate=np.zeros((len(gold), 1), np.float32))


def _eval_model(model, data: Dict[str, Any], M: int, device: torch.device, args, batch_size: int = 1024) -> Dict[str, Any]:
    N = len(data["gold"])
    topk = data["topk_ids"].astype(np.int64)
    lgt = data["topk_lgt"].astype(np.float32)
    gold = data["gold"].astype(np.int64)
    pred_scores = np.empty((N, M), dtype=np.float32)
    delta_arr = np.empty((N, M), dtype=np.float32)
    gate_arr = np.empty((N, 1), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            idx = np.arange(s, e)
            b = _to_device(_make_batch(data, idx, M), device)
            final, delta, gate = model(**b)
            pred_scores[s:e] = final.detach().float().cpu().numpy()
            delta_arr[s:e] = delta.detach().float().cpu().numpy()
            gate_arr[s:e] = gate.detach().float().cpu().numpy()

    return _metric_from_scores(pred_scores, lgt, topk, gold, delta_arr, gate_arr)


def check_identity(model, data: Dict[str, Any], M: int, device: torch.device, variant: str, atol: float = 1e-5):
    model.eval()
    idx = np.arange(min(512, len(data["gold"])))
    batch = _to_device(_make_batch(data, idx, M), device)
    with torch.no_grad():
        final, delta, gate = model(**batch)
    base_lgt = batch["topk_lgt"]
    max_diff = float((final - base_lgt).abs().max().item())
    if max_diff > atol:
        raise RuntimeError(f"[{variant}] identity FAILED: max_abs(final-base)={max_diff:.3e} > {atol}")

    # Mini metric identity check.
    small = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(data["gold"]) else v) for k, v in data.items()}
    base_vm = evaluate_base_only(small, M)
    pred_scores = final.detach().cpu().float().numpy()
    mini_vm = _metric_from_scores(pred_scores, small["topk_lgt"], small["topk_ids"], small["gold"], delta.detach().cpu().numpy(), gate.detach().cpu().numpy())
    nll_diff = abs(mini_vm["candidate_nll_given_gold_in_topM"] - base_vm["candidate_nll_given_gold_in_topM"])
    if nll_diff > atol:
        raise RuntimeError(f"[{variant}] identity NLL FAILED: diff={nll_diff:.3e} > {atol}")
    if mini_vm["changed_to_gold"] != 0 or mini_vm["changed_away"] != 0:
        raise RuntimeError(f"[{variant}] identity correction FAILED: ctg={mini_vm['changed_to_gold']} caw={mini_vm['changed_away']}")
    print(f"[{variant}] identity PASS: max_diff={max_diff:.2e} nll_diff={nll_diff:.2e}")
    model.train()


def eval_slices(model_or_none, data: Dict[str, Any], real_tok_arr: np.ndarray, reg_arr: np.ndarray, M: int, device: torch.device, args):
    N = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk_ids"].astype(np.int64)
    lgt = data["topk_lgt"].astype(np.float32)
    ar = np.arange(N)

    if model_or_none is None:
        pred_scores = lgt.copy()
    else:
        pred_scores = np.empty((N, M), dtype=np.float32)
        model_or_none.eval()
        with torch.no_grad():
            for s in range(0, N, 1024):
                e = min(s + 1024, N)
                idx = np.arange(s, e)
                b = _to_device(_make_batch(data, idx, M), device)
                f, _, _ = model_or_none(**b)
                pred_scores[s:e] = f.detach().cpu().float().numpy()

    pred_top1_id = topk[ar, pred_scores.argmax(axis=1)]
    base_top1 = topk[:, 0]
    base_correct = base_top1 == gold
    pred_correct = pred_top1_id == gold
    gip = (topk == gold[:, None]).any(axis=1)
    gold_rank_arr = np.where(gip, (topk == gold[:, None]).argmax(axis=1), -1)

    Vt, Rr = len(real_tok_arr), len(reg_arr)
    base_reg = real_tok_arr[np.clip(base_top1, 0, Vt - 1)]
    gold_reg = real_tok_arr[np.clip(gold, 0, Vt - 1)]
    base_sreg = reg_arr[np.clip(base_reg, 0, Rr - 1)]
    gold_sreg = reg_arr[np.clip(gold_reg, 0, Rr - 1)]
    unk_r, unk_s = int(real_tok_arr.max()), int(reg_arr.max())
    same_reg = (base_reg == gold_reg) & (gold_reg != unk_r)
    same_sreg = (base_sreg == gold_sreg) & (gold_sreg != unk_s)

    slices = {
        "all": np.ones(N, bool),
        "gold_in_topM": gip,
        "base_correct": base_correct,
        "bucketA": (~base_correct) & gip,
        "same_reg_conf": (~base_correct) & gip & same_reg,
        "same_sreg_conf": (~base_correct) & gip & same_sreg,
        "diff_reg_conf": (~base_correct) & gip & (~same_reg),
        "gold_rank_1_5": gip & (gold_rank_arr < 5),
        "gold_rank_6_32": gip & (gold_rank_arr >= 5) & (gold_rank_arr < 32),
        "gold_rank_33_M": gip & (gold_rank_arr >= 32),
    }

    def _slice_metrics(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "base_acc": float("nan"), "model_acc": float("nan"), "acc_gain": float("nan"), "base_nll": float("nan"), "model_nll": float("nan"), "nll_gain": float("nan"), "changed_to_gold": 0, "changed_away": 0, "net_correction": 0}
        ctg = int(((~base_correct[mask]) & pred_correct[mask]).sum())
        caw = int((base_correct[mask] & (~pred_correct[mask])).sum())
        gip_sl = gip & mask
        if gip_sl.any():
            # direct computation for gip rows inside this slice
            gr = gold_rank_arr[gip_sl]
            x = lgt[gip_sl].astype(np.float64); x -= x.max(axis=1, keepdims=True)
            ls = x - np.log(np.exp(x).sum(axis=1, keepdims=True) + _EPS)
            base_nll = float((-ls[np.arange(len(gr)), np.clip(gr, 0, M - 1)]).mean())
            y = pred_scores[gip_sl].astype(np.float64); y -= y.max(axis=1, keepdims=True)
            ps = y - np.log(np.exp(y).sum(axis=1, keepdims=True) + _EPS)
            model_nll = float((-ps[np.arange(len(gr)), np.clip(gr, 0, M - 1)]).mean())
        else:
            base_nll = model_nll = float("nan")
        return {
            "n": n,
            "base_acc": float(base_correct[mask].mean()),
            "model_acc": float(pred_correct[mask].mean()),
            "acc_gain": float(pred_correct[mask].mean() - base_correct[mask].mean()),
            "base_nll": base_nll,
            "model_nll": model_nll,
            "nll_gain": base_nll - model_nll if base_nll == base_nll and model_nll == model_nll else float("nan"),
            "changed_to_gold": ctg,
            "changed_away": caw,
            "net_correction": ctg - caw,
        }

    return {name: _slice_metrics(mask) for name, mask in slices.items()}


# =============================================================================
# 7. Training
# =============================================================================


def make_balanced_sampler(buckets: Dict[str, np.ndarray], n_steps: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    ra = np.where(buckets["bucketA"])[0]
    rb = np.where(buckets["base_correct"])[0]
    rc = np.where(buckets["same_reg_or_sreg_conf"])[0]
    fallback = np.arange(len(buckets["base_correct"]))
    all_idx = np.concatenate([x for x in [ra, rb, rc] if len(x) > 0]) if (len(ra) + len(rb) + len(rc)) else fallback
    na = max(1, int(batch_size * 0.40))
    nb = max(1, int(batch_size * 0.40))
    nc = batch_size - na - nb

    def samp(arr, n):
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        if len(arr) == 0:
            return rng.choice(all_idx, n, replace=True)
        return rng.choice(arr, n, replace=len(arr) < n)

    out = []
    for _ in range(n_steps):
        idx = np.concatenate([samp(ra, na), samp(rb, nb), samp(rc, nc)])
        rng.shuffle(idx)
        out.append(idx)
    return out


def train_variant(variant: str, model, train_data, val_data, buckets, M: int, args, device, out_dir: str):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    sampler = make_balanced_sampler(buckets, args.steps, args.batch_size, args.seed)
    base_nll = evaluate_base_only(val_data, M)["candidate_nll_given_gold_in_topM"]
    print(f"[{variant}] base val candidate_nll={base_nll:.6f} (must beat to save best)")

    best_state = None
    best_metrics = None
    best_step = None
    train_rows = []
    eval_rows = []
    t0 = time.time()

    model.train()
    for step, idx in enumerate(sampler, start=1):
        batch = _to_device(_make_batch(train_data, idx, M), device)
        gold_t = torch.from_numpy(train_data["gold"][idx].astype(np.int64)).to(device)
        bc_mask = torch.from_numpy(buckets["base_correct"][idx]).to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
            final, delta, gate = model(**batch)
            loss, ld = compute_loss(final, batch["topk_ids"].long(), gold_t, bc_mask, delta, gate, args)
        if not torch.isfinite(loss):
            raise RuntimeError(f"[{variant}] non-finite loss at step={step}: {ld}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.eval_every == 0:
            model.eval()
            vm = _eval_model(model, val_data, M, device, args)
            model.train()
            elapsed = time.time() - t0
            print(
                f"[{variant}] step={step:5d} loss={ld['total']:.4f} "
                f"val_nll={vm['candidate_nll_given_gold_in_topM']:.6f} "
                f"all_acc={vm['all_row_model_acc']:.4f} "
                f"ctg={vm['changed_to_gold']:,} caw={vm['changed_away']:,} "
                f"gate={vm['gate_mean']:.3f} dmean={vm['delta_abs_mean']:.4f} {elapsed:.0f}s"
            )
            row = {"step": step, "variant": variant, **{f"loss_{k}": v for k, v in ld.items()}, **vm}
            train_rows.append(row)
            eval_rows.append({"step": step, "variant": variant, **vm})
            nll = vm["candidate_nll_given_gold_in_topM"]
            if nll < base_nll and (best_metrics is None or nll < best_metrics["candidate_nll_given_gold_in_topM"]):
                best_step = step
                best_state = deepcopy(model.state_dict())
                best_metrics = deepcopy(vm)
                print(f"  → new best beats base: step={step} nll={nll:.6f}")

    model.eval()
    final_metrics = _eval_model(model, val_data, M, device, args)
    final_metrics = {"variant": variant, "final_step": args.steps, **final_metrics}

    last_path = os.path.join(out_dir, f"last_{variant}.pt")
    torch.save({"state_dict": model.state_dict(), "variant": variant, "step": args.steps, "metrics": final_metrics}, last_path)

    checkpoint_path = None
    if best_state is not None:
        checkpoint_path = os.path.join(out_dir, f"best_{variant}.pt")
        torch.save({"state_dict": best_state, "variant": variant, "step": best_step, "metrics": best_metrics}, checkpoint_path)
        print(f"[{variant}] saved best: {checkpoint_path}")
    else:
        print(f"[{variant}] no improving checkpoint found")

    best_out = {
        "variant": variant,
        "no_improving_checkpoint_found": best_state is None,
        "best_step": best_step,
        "checkpoint_path": checkpoint_path,
        "base_val_candidate_nll_given_gold_in_topM": base_nll,
    }
    if best_metrics is not None:
        best_out.update({f"best_{k}": v for k, v in best_metrics.items()})
    else:
        best_out["no_improving_checkpoint_found"] = True

    return best_out, final_metrics, train_rows, eval_rows


# =============================================================================
# 8. Comparison/report
# =============================================================================


def select_metrics_for_comparison(variant: str, best: Optional[Dict[str, Any]], final: Dict[str, Any]) -> Dict[str, Any]:
    if variant == "base_only":
        out = {"variant": variant, "selected_for_comparison": "base", "no_improving_checkpoint_found": False}
        out.update(final)
        return out
    if best and not best.get("no_improving_checkpoint_found", True):
        out = {"variant": variant, "selected_for_comparison": "best", "no_improving_checkpoint_found": False}
        prefix = "best_"
        for k, v in best.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        out["best_step"] = best.get("best_step")
        out["checkpoint_path"] = best.get("checkpoint_path")
        return out
    out = {"variant": variant, "selected_for_comparison": "final_no_best", "no_improving_checkpoint_found": True}
    out.update(final)
    return out


def build_comparison_row(variant, selected_vm, slice_m, base_vm):
    base_nll = base_vm["candidate_nll_given_gold_in_topM"]
    base_acc = base_vm["candidate_acc_given_gold_in_topM"]
    def gain(a, b):
        return (a - b) if (a == a and b == b) else float("nan")
    return {
        "variant": variant,
        "selected_for_comparison": selected_vm.get("selected_for_comparison", "unknown"),
        "no_improving_checkpoint_found": selected_vm.get("no_improving_checkpoint_found", False),
        "candidate_nll_given_gold_in_topM": selected_vm.get("candidate_nll_given_gold_in_topM", float("nan")),
        "candidate_acc_given_gold_in_topM": selected_vm.get("candidate_acc_given_gold_in_topM", float("nan")),
        "nll_gain_vs_base": gain(base_nll, selected_vm.get("candidate_nll_given_gold_in_topM", float("nan"))),
        "acc_gain_vs_base_given_gold_in_topM": gain(selected_vm.get("candidate_acc_given_gold_in_topM", float("nan")), base_acc),
        "all_row_base_acc": selected_vm.get("all_row_base_acc", base_vm.get("all_row_base_acc", float("nan"))),
        "all_row_model_acc": selected_vm.get("all_row_model_acc", float("nan")),
        "all_row_acc_gain": selected_vm.get("all_row_acc_gain", float("nan")),
        "natural_gold_in_topM_rate": selected_vm.get("natural_gold_in_topM_rate", float("nan")),
        "not_in_pool_rate": selected_vm.get("not_in_pool_rate", float("nan")),
        "changed_to_gold": selected_vm.get("changed_to_gold", 0),
        "changed_away": selected_vm.get("changed_away", 0),
        "net_correction": selected_vm.get("net_correction", 0),
        "benefit_damage_ratio": selected_vm.get("benefit_damage_ratio", float("nan")),
        "apply_rate": selected_vm.get("apply_rate", float("nan")),
        "base_correct_damage_rate": selected_vm.get("base_correct_damage_rate", float("nan")),
        "gate_mean": selected_vm.get("gate_mean", float("nan")),
        "delta_abs_mean": selected_vm.get("delta_abs_mean", float("nan")),
        "delta_abs_max": selected_vm.get("delta_abs_max", float("nan")),
        "bucketA_acc_gain": slice_m.get("bucketA", {}).get("acc_gain", float("nan")),
        "same_reg_conf_acc_gain": slice_m.get("same_reg_conf", {}).get("acc_gain", float("nan")),
        "same_sreg_conf_acc_gain": slice_m.get("same_sreg_conf", {}).get("acc_gain", float("nan")),
    }


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float) and v == v:
        return f"{v:.6f}"
    return str(v) if v == v else "nan"


def phase1_verdict(comparison: List[Dict[str, Any]], max_base_correct_damage_rate: float):
    rows = {r["variant"]: r for r in comparison}
    real_names = ["hard_region_features", "static_region_embedding"]
    real_rows = [rows[n] for n in real_names if n in rows]
    shuf_rows = [r for r in comparison if "shuffled" in r["variant"] or "random" in r["variant"]]
    logit = rows.get("logit_only_mlp")

    def best_gain(rs):
        vals = [r.get("nll_gain_vs_base", float("nan")) for r in rs]
        vals = [v for v in vals if v == v]
        return max(vals) if vals else float("-inf")

    best_real = max(real_rows, key=lambda r: r.get("nll_gain_vs_base", float("-inf")), default=None)
    best_real_gain = best_gain(real_rows)
    best_shuf_gain = best_gain(shuf_rows)
    logit_gain = logit.get("nll_gain_vs_base", float("-inf")) if logit else float("-inf")

    reasons = []
    if best_real is None or best_real_gain <= 0.0001:
        reasons.append("real_does_not_beat_base")
    if logit is not None and best_real_gain <= logit_gain + 0.0001:
        reasons.append("real_does_not_beat_logit_only")
    if shuf_rows and best_real_gain <= best_shuf_gain + 0.0001:
        reasons.append("real_does_not_beat_shuffled")
    if best_real is not None and best_real.get("changed_to_gold", 0) <= best_real.get("changed_away", 0):
        reasons.append("changed_away_ge_changed_to_gold")
    if best_real is not None and best_real.get("base_correct_damage_rate", 1.0) >= max_base_correct_damage_rate:
        reasons.append("base_correct_damage_too_high")

    proceed = len(reasons) == 0
    return ("PROCEED_TO_PHASE_2" if proceed else "DO_NOT_PROCEED_TO_PHASE_2"), reasons, best_real


def write_report(args, comparison, all_slice_metrics, out_dir):
    verdict, reasons, best_real = phase1_verdict(comparison, args.max_base_correct_damage_rate)
    rows = {r["variant"]: r for r in comparison}
    rpt = os.path.join(out_dir, "phase1_static_region_report.md")
    cols = [
        "variant", "selected_for_comparison", "candidate_nll_given_gold_in_topM",
        "nll_gain_vs_base", "all_row_model_acc", "all_row_acc_gain",
        "changed_to_gold", "changed_away", "benefit_damage_ratio",
        "base_correct_damage_rate", "bucketA_acc_gain",
    ]
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("# Phase 1: Static Region Feature Mixer — Report\n\n")
        f.write(f"**selected_M:** {args.selected_M} | **steps:** {args.steps} | **seed:** {args.seed}\n\n")
        f.write("## Comparison Table\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for r in comparison:
            f.write("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |\n")

        f.write("\n## Answers\n\n")
        logit = rows.get("logit_only_mlp")
        base = rows.get("base_only")
        hard = rows.get("hard_region_features")
        emb = rows.get("static_region_embedding")
        shuf = rows.get("shuffled_region_control")
        rand = rows.get("random_region_control")

        f.write("### Q1. Does logit_only_mlp improve over base?\n\n")
        f.write(f"**{'YES' if logit and logit.get('nll_gain_vs_base', 0) > 0 else 'NO'}** — nll_gain={_fmt(logit.get('nll_gain_vs_base', float('nan')) if logit else float('nan'))}\n\n")

        f.write("### Q2. Do real region features improve over logit_only_mlp?\n\n")
        if best_real and logit:
            f.write(f"**{'YES' if best_real.get('nll_gain_vs_base', -9) > logit.get('nll_gain_vs_base', -9) + 0.0001 else 'NO'}** — best_real={best_real['variant']} gain={_fmt(best_real.get('nll_gain_vs_base'))}, logit_gain={_fmt(logit.get('nll_gain_vs_base'))}\n\n")
        else:
            f.write("**UNKNOWN**\n\n")

        f.write("### Q3. Do learned region embeddings beat hard region IDs?\n\n")
        if hard and emb:
            f.write(f"**{'YES' if emb.get('nll_gain_vs_base', -9) > hard.get('nll_gain_vs_base', -9) else 'NO'}** — emb={_fmt(emb.get('nll_gain_vs_base'))}, hard={_fmt(hard.get('nll_gain_vs_base'))}\n\n")

        f.write("### Q4. Do real regions beat shuffled/random controls?\n\n")
        if best_real:
            ctrl_best = max([x.get('nll_gain_vs_base', float('-inf')) for x in [shuf, rand] if x], default=float('-inf'))
            f.write(f"**{'YES' if best_real.get('nll_gain_vs_base', -9) > ctrl_best + 0.0001 else 'NO'}** — best_real={_fmt(best_real.get('nll_gain_vs_base'))}, best_control={_fmt(ctrl_best)}\n\n")

        f.write("### Q5. Do same-region/superregion confusers improve?\n\n")
        for name in ["hard_region_features", "static_region_embedding"]:
            sm = all_slice_metrics.get(name, {})
            f.write(f"- {name}: same_reg={_fmt(sm.get('same_reg_conf', {}).get('acc_gain', float('nan')))}, same_sreg={_fmt(sm.get('same_sreg_conf', {}).get('acc_gain', float('nan')))}\n")
        f.write("\n")

        f.write("### Q6. Is changed_to_gold > changed_away?\n\n")
        if best_real:
            f.write(f"**{'YES' if best_real.get('changed_to_gold', 0) > best_real.get('changed_away', 0) else 'NO'}** — ctg={best_real.get('changed_to_gold')}, caw={best_real.get('changed_away')}\n\n")

        f.write("### Q7. Should we proceed to Phase 2?\n\n")
        f.write(f"**{verdict}**\n\n")
        if reasons:
            f.write("Failure reasons:\n")
            for r in reasons:
                f.write(f"- {r}\n")
            f.write("\n")

        f.write("## PHASE 1 VERDICT\n\n```text\n")
        for r in comparison:
            f.write(f"{r['variant']:30s} nll_gain={_fmt(r.get('nll_gain_vs_base'))} all_acc_gain={_fmt(r.get('all_row_acc_gain'))} BDR={_fmt(r.get('benefit_damage_ratio'))} bc_dmg={_fmt(r.get('base_correct_damage_rate'))}\n")
        f.write(f"\nrecommendation: {verdict}\n```\n")
    print(f"[save] {rpt}")
    return rpt


# =============================================================================
# 9. Main
# =============================================================================


def main():
    p = argparse.ArgumentParser(description="Phase 1: Static Region Feature Mixer")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--token_to_region", required=True)
    p.add_argument("--super_map", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--selected_M", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_delta", type=float, default=1e-4)
    p.add_argument("--lambda_gate", type=float, default=1e-3)
    p.add_argument("--lambda_preserve", type=float, default=0.5)
    p.add_argument("--max_base_correct_damage_rate", type=float, default=0.05)
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--max_val_rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--skip_random_control", action="store_true")
    p.add_argument("--gate_init_bias", type=float, default=0.0)
    p.add_argument("--disable_gate_for_logit_only", action="store_true")
    p.add_argument("--use_gate", dest="use_gate", action="store_true")
    p.add_argument("--no_gate", dest="use_gate", action="store_false")
    p.set_defaults(use_gate=True)
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    t0 = time.time()

    print("\n[step 1] loading shards")
    M = args.selected_M
    train_data = load_shards(args.train_dir, M, args.max_train_rows, "train")
    val_data = load_shards(args.val_dir, M, args.max_val_rows, "val")
    M = train_data["M"]

    print("\n[step 2] loading region maps")
    tok_arr, reg_arr, n_regions, n_super = load_region_maps(args.token_to_region, args.super_map)
    shuf_tok = _shuffle_tok_arr(tok_arr, n_regions, args.seed)
    rand_tok = _random_tok_arr(tok_arr, n_regions, args.seed)
    save_token_region_map(shuf_tok, n_regions, os.path.join(args.output_dir, "shuffled_token_to_region.json"))
    save_token_region_map(rand_tok, n_regions, os.path.join(args.output_dir, "random_token_to_region.json"))

    print("\n[step 3] computing train buckets")
    train_buckets = compute_bucket_labels(train_data, tok_arr, reg_arr)

    cfg = vars(args).copy()
    cfg.update({"n_regions": n_regions, "n_super": n_super, "train_rows": len(train_data["gold"]), "val_rows": len(val_data["gold"]), "has_router": bool(train_data["has_router"]), "device": str(device), "M": M})
    _json_dump(cfg, os.path.join(args.output_dir, "config.json"))

    print("\n[step 4] base_only")
    base_vm = evaluate_base_only(val_data, M)
    base_vm = {"variant": "base_only", **base_vm}
    base_slices = eval_slices(None, val_data, tok_arr, reg_arr, M, device, args)
    print(f"base_only: cand_nll={base_vm['candidate_nll_given_gold_in_topM']:.6f} cand_acc={base_vm['candidate_acc_given_gold_in_topM']:.4f} all_acc={base_vm['all_row_model_acc']:.4f} gip={base_vm['natural_gold_in_topM_rate']:.4f}")

    all_best: Dict[str, Dict[str, Any]] = {}
    all_final: Dict[str, Dict[str, Any]] = {"base_only": base_vm}
    all_slice_m: Dict[str, Dict[str, Any]] = {"base_only": base_slices}
    all_train_logs: List[Dict[str, Any]] = []
    all_eval_logs: List[Dict[str, Any]] = []

    has_router = bool(train_data["has_router"])

    def make_model(variant: str, tok_arr_v: np.ndarray):
        if variant == "logit_only_mlp":
            use_gate = args.use_gate and not args.disable_gate_for_logit_only
            return LogitOnlyMixer(args.hidden_dim, args.dropout, use_gate, args.gate_init_bias).to(device)
        if variant == "hard_region_features":
            return HardRegionMixer(tok_arr_v, reg_arr, args.hidden_dim, args.dropout, args.use_gate, args.gate_init_bias).to(device)
        return StaticRegionEmbMixer(tok_arr_v, reg_arr, n_regions, n_super, args.hidden_dim, args.dropout, has_router, args.use_gate, args.gate_init_bias).to(device)

    def run_variant(variant: str, tok_arr_v: np.ndarray):
        print("\n" + "=" * 70)
        print(f"[variant] {variant}")
        print("=" * 70)
        model = make_model(variant, tok_arr_v)
        check_identity(model, train_data, M, device, variant)
        best, final, tlog, elog = train_variant(variant, model, train_data, val_data, train_buckets, M, args, device, args.output_dir)
        # IMPORTANT: slices always use real tok_arr, not shuffled/random controls.
        slices = eval_slices(model, val_data, tok_arr, reg_arr, M, device, args)
        all_best[variant] = best
        all_final[variant] = final
        all_slice_m[variant] = slices
        all_train_logs.extend(tlog)
        all_eval_logs.extend(elog)
        print(f"[{variant}] final: cand_nll={final['candidate_nll_given_gold_in_topM']:.6f} all_acc={final['all_row_model_acc']:.4f} ctg={final['changed_to_gold']:,} caw={final['changed_away']:,}")

    variants = [
        ("logit_only_mlp", tok_arr),
        ("hard_region_features", tok_arr),
        ("static_region_embedding", tok_arr),
        ("shuffled_region_control", shuf_tok),
    ]
    if not args.skip_random_control:
        variants.append(("random_region_control", rand_tok))
    for v, ta in variants:
        run_variant(v, ta)

    print("\n[step 5] comparison")
    selected = {"base_only": select_metrics_for_comparison("base_only", None, base_vm)}
    for v, _ in variants:
        selected[v] = select_metrics_for_comparison(v, all_best.get(v), all_final[v])

    comp_rows = []
    for v in ["base_only"] + [x[0] for x in variants]:
        comp_rows.append(build_comparison_row(v, selected[v], all_slice_m[v], base_vm))

    _wcsv(os.path.join(args.output_dir, "phase1_comparison.csv"), comp_rows)

    slice_rows = []
    for vname, sm in all_slice_m.items():
        for sname, vals in sm.items():
            slice_rows.append({"variant": vname, "slice": sname, **vals})
    _wcsv(os.path.join(args.output_dir, "slice_metrics.csv"), slice_rows)
    _wcsv(os.path.join(args.output_dir, "train_log.csv"), all_train_logs)
    _wcsv(os.path.join(args.output_dir, "eval_log.csv"), all_eval_logs)
    _json_dump(all_best, os.path.join(args.output_dir, "best_metrics.json"))
    _json_dump(all_final, os.path.join(args.output_dir, "final_metrics.json"))

    rpt = write_report(args, comp_rows, all_slice_m, args.output_dir)
    verdict, reasons, best_real = phase1_verdict(comp_rows, args.max_base_correct_damage_rate)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("PHASE 1 VERDICT")
    print("=" * 70)
    for row in comp_rows:
        print(f"{row['variant']:30s} selected={row['selected_for_comparison']:14s} nll_gain={row['nll_gain_vs_base']:.6f} all_acc_gain={row['all_row_acc_gain']:.6f} BDR={row['benefit_damage_ratio']:.3f} bc_dmg={row['base_correct_damage_rate']:.4f}")
    print(f"\nrecommendation: {verdict}")
    if reasons:
        print("reasons:")
        for r in reasons:
            print(f"  - {r}")
    print(f"elapsed: {elapsed:.0f}s")
    print(f"report: {rpt}")
    print("=" * 70)


if __name__ == "__main__":
    main()
