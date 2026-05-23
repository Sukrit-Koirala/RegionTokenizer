#!/usr/bin/env python3
"""run_knn_pair_selector_v1.py — KNN Pair Selector (non-oracle)

V3 oracle surgical correction worked (ctg=0.2407, fv_gain=+0.1584) but used
gold at eval to pick the pair. This script replaces the oracle with a
non-parametric kNN selector:

  For each val row:
    b = base_top1
    candidates c = filtered topK[1:] (by region, no gold)
    memory_score(c) = kNN support from train pair positives/negatives
    c* = argmax memory_score
    if memory_score(c*) >= threshold: apply surgical correction
    else: no-op

Gold is never used to select c* or set threshold. Gold only used for metrics.
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
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_STORE_FILTERS = [
    "covered_base_wrong",
    "same_region_confuser",
    "same_superregion_confuser",
    "same_region_or_superregion_confuser",
]
_CAND_FILTERS = [
    "top_rank",
    "same_region_or_superregion",
    "same_region_only",
    "same_superregion_only",
]

_KEY_ALIASES = {
    "h_prime":          ["h_prime", "h_ctx"],
    "base_topk_ids":    ["base_topk_ids", "base_topk"],
    "base_topk_logits": ["base_topk_logits", "base_topk_lgt"],
    "gold_token":       ["gold_token", "gold"],
    "input_ids":        ["input_ids"],
    "row_id":           ["row_id"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Backbone / maps
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(ckpt_path: str, device: torch.device):
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(ckpt_path, device)
    tok_w = backbone.token_emb.weight.detach().float().cpu()
    del backbone
    return tok_w, d_model, vocab_size


def _parse_map(raw):
    if isinstance(raw, list):
        return {i: v for i, v in enumerate(raw) if v is not None}
    elif isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    raise ValueError(f"Expected list or dict, got {type(raw)}")


def load_maps(t2r_path: str, super_path: Optional[str]):
    with open(t2r_path) as f:
        t2r = _parse_map(json.load(f))
    max_region = max(t2r.values()) if t2r else 0
    unk_region = int(max_region) + 1
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f:
            r2s = _parse_map(json.load(f))
        max_super = max(r2s.values()) if r2s else 0
        unk_super = int(max_super) + 1
        sr_enabled = True
    else:
        r2s = {}; unk_super = 0; sr_enabled = False
    print(f"  n_regions={unk_region}  sr_enabled={sr_enabled}  unk_super={unk_super}")
    return t2r, r2s, unk_region, unk_super, sr_enabled


def build_lookup_arrays(t2r, r2s, unk_region, unk_super, vocab_size):
    tok_to_reg = np.full(vocab_size, unk_region, dtype=np.int32)
    for tok, reg in t2r.items():
        if 0 <= int(tok) < vocab_size:
            tok_to_reg[int(tok)] = int(reg)
    reg_to_sup = np.full(unk_region + 1, unk_super, dtype=np.int32)
    for reg, sup in r2s.items():
        ir = int(reg)
        if 0 <= ir <= unk_region:
            reg_to_sup[ir] = int(sup)
    return tok_to_reg, reg_to_sup


# ─────────────────────────────────────────────────────────────────────────────
# Shards
# ─────────────────────────────────────────────────────────────────────────────

def _pick(d, aliases, required=True):
    for k in aliases:
        if k in d:
            return d[k]
    if required:
        raise KeyError(f"None of {aliases} found in {list(d.keys())}")
    return None


def load_shards(shard_dir, top_k, split, max_rows=None, load_ids=False):
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")
    parts = {"h_prime": [], "topk_ids": [], "topk_lgt": [], "gold": []}
    if load_ids:
        parts["input_ids"] = []
        parts["row_id"] = []
    total = 0
    for p in paths:
        shard = torch.load(p, map_location="cpu")
        hp  = _pick(shard, _KEY_ALIASES["h_prime"]).float()
        ids = _pick(shard, _KEY_ALIASES["base_topk_ids"]).long()
        lgt = _pick(shard, _KEY_ALIASES["base_topk_logits"]).float()
        gld = _pick(shard, _KEY_ALIASES["gold_token"]).long()
        if hp.dim() == 1:  hp  = hp.unsqueeze(0)
        if ids.dim() == 1: ids = ids.unsqueeze(0)
        if lgt.dim() == 1: lgt = lgt.unsqueeze(0)
        if gld.dim() == 0: gld = gld.unsqueeze(0)
        K = ids.shape[1]
        if K > top_k:
            ids = ids[:, :top_k]; lgt = lgt[:, :top_k]
        elif K < top_k:
            pad = top_k - K
            ids = F.pad(ids, (0, pad), value=0)
            lgt = F.pad(lgt, (0, pad), value=-1e9)
        n = hp.shape[0]
        parts["h_prime"].append(hp); parts["topk_ids"].append(ids)
        parts["topk_lgt"].append(lgt); parts["gold"].append(gld)
        if load_ids:
            ii = _pick(shard, _KEY_ALIASES["input_ids"], required=False)
            ri = _pick(shard, _KEY_ALIASES["row_id"], required=False)
            parts["input_ids"].append(ii if ii is not None else torch.zeros(n, 1, dtype=torch.long))
            parts["row_id"].append(ri if ri is not None else torch.arange(total, total + n))
        total += n
        if max_rows and total >= max_rows:
            break
    out = {k: torch.cat(v, 0) for k, v in parts.items() if v}
    if max_rows:
        out = {k: v[:max_rows] for k, v in out.items()}
    print(f"  [{split}] {out['h_prime'].shape[0]} rows from {len(paths)} shards")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pair feature builder
# ─────────────────────────────────────────────────────────────────────────────

class PairFeatureBuilder:
    """Fixed random-projected pair feature vectors, no learned parameters."""

    N_SCALARS = 5  # logit_gap, rank_norm, same_region, same_superregion, hp_dot_gap

    def __init__(self, d_model: int, key_dim: int, seed: int = 42):
        self.d_model  = d_model
        self.key_dim  = key_dim
        self.raw_dim  = 4 * d_model + self.N_SCALARS
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((self.raw_dim, key_dim)).astype(np.float32)
        R /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-8
        self.R = torch.from_numpy(R)
        self.scalar_means = np.zeros(self.N_SCALARS, dtype=np.float32)
        self.scalar_stds  = np.ones(self.N_SCALARS,  dtype=np.float32)

    def fit_scalars(self, scalars: np.ndarray):
        self.scalar_means = scalars.mean(0).astype(np.float32)
        self.scalar_stds  = (scalars.std(0) + 1e-8).astype(np.float32)

    def standardize(self, scalars: np.ndarray) -> np.ndarray:
        return ((scalars - self.scalar_means) / self.scalar_stds).astype(np.float32)

    def build_keys(self, hp: torch.Tensor, emb_c: torch.Tensor, emb_b: torch.Tensor,
                   scalars_std: torch.Tensor) -> torch.Tensor:
        """All inputs on same device. Returns L2-norm keys [B, key_dim]."""
        hp_n   = F.normalize(hp,             dim=-1)
        ec_n   = F.normalize(emb_c,          dim=-1)
        eb_n   = F.normalize(emb_b,          dim=-1)
        diff_n = F.normalize(emb_c - emb_b,  dim=-1)
        raw    = torch.cat([hp_n, ec_n, eb_n, diff_n, scalars_std], dim=-1)
        proj   = raw @ self.R.to(raw.device)
        return F.normalize(proj, dim=-1)

    def state_dict(self):
        return {"R": self.R, "scalar_means": self.scalar_means,
                "scalar_stds": self.scalar_stds, "d_model": self.d_model,
                "key_dim": self.key_dim}

    @classmethod
    def from_state(cls, sd):
        obj = cls.__new__(cls)
        obj.d_model = sd["d_model"]; obj.key_dim = sd["key_dim"]
        obj.raw_dim = 4 * obj.d_model + cls.N_SCALARS
        obj.R = sd["R"]; obj.scalar_means = sd["scalar_means"]
        obj.scalar_stds = sd["scalar_stds"]
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Region lookups
# ─────────────────────────────────────────────────────────────────────────────

def get_regs_sups(tok_ids_np: np.ndarray, tok_to_reg, reg_to_sup):
    """tok_ids_np: any shape int array. Returns (reg, sup) same shape."""
    flat  = np.clip(tok_ids_np.ravel(), 0, len(tok_to_reg) - 1)
    reg   = tok_to_reg[flat]
    sup   = reg_to_sup[np.clip(reg, 0, len(reg_to_sup) - 1)]
    return reg.reshape(tok_ids_np.shape), sup.reshape(tok_ids_np.shape)


def is_same_reg_sup(a_reg, a_sup, b_reg, b_sup, unk_region, unk_super):
    sr = (a_reg == b_reg) & (a_reg != unk_region)
    ss = (a_sup == b_sup) & (a_sup != unk_super)
    return sr, ss


# ─────────────────────────────────────────────────────────────────────────────
# Pair scalar computation (vectorized over pairs)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pair_scalars(hp_np, emb_c_np, emb_b_np, lgt_c, lgt_b, rank_c,
                         top_k, c_reg, c_sup, b_reg, b_sup,
                         unk_region, unk_super) -> np.ndarray:
    """All inputs [N] or [N, d]. Returns [N, 5]."""
    logit_gap = lgt_c - lgt_b
    rank_norm = rank_c.astype(np.float32) / top_k
    sr, ss = is_same_reg_sup(c_reg, c_sup, b_reg, b_sup, unk_region, unk_super)
    hp_n  = hp_np  / (np.linalg.norm(hp_np,  axis=1, keepdims=True) + 1e-8)
    ec_n  = emb_c_np / (np.linalg.norm(emb_c_np, axis=1, keepdims=True) + 1e-8)
    eb_n  = emb_b_np / (np.linalg.norm(emb_b_np, axis=1, keepdims=True) + 1e-8)
    hp_dot_c = (hp_n * ec_n).sum(1)
    hp_dot_b = (hp_n * eb_n).sum(1)
    hp_dot_gap = hp_dot_c - hp_dot_b
    return np.stack([logit_gap, rank_norm,
                     sr.astype(np.float32), ss.astype(np.float32),
                     hp_dot_gap], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Build pair datastore
# ─────────────────────────────────────────────────────────────────────────────

def build_datastore(train_data, tok_w_np: np.ndarray, tok_to_reg, reg_to_sup,
                    unk_region, unk_super, sr_enabled,
                    builder: PairFeatureBuilder, args):
    print("\n[datastore] Building pair datastore...")
    N    = train_data["h_prime"].shape[0]
    top_k  = args.top_k
    n_neg  = args.negatives_per_positive
    sf     = args.store_filter
    max_rows = min(N, args.max_store_rows)

    # Collect pair indices first (vectorized), then build keys in batches
    rec_row_idx    = []  # index into train_data
    rec_chal_toks  = []  # challenger token id
    rec_base_toks  = []  # base_top1 token id
    rec_gold_toks  = []
    rec_labels     = []  # 1=positive, 0=negative
    rec_rank_c     = []  # rank of challenger in topK

    hp_all   = train_data["h_prime"][:max_rows].float().numpy()
    topk_all = train_data["topk_ids"][:max_rows].numpy()
    lgt_all  = train_data["topk_lgt"][:max_rows].numpy()
    gold_all = train_data["gold"][:max_rows].numpy()

    base_toks  = topk_all[:, 0]
    gold_toks  = gold_all

    # Covered and base_wrong
    gold_in_topk = (topk_all == gold_toks[:, None])          # [N, K]
    covered      = gold_in_topk.any(1)                        # [N]
    base_wrong   = (base_toks != gold_toks)                   # [N]
    gold_idx     = gold_in_topk.argmax(1).astype(np.int32)   # [N] rank of gold in topK

    # Region/super of gold and base_top1
    g_reg, g_sup = get_regs_sups(gold_toks,  tok_to_reg, reg_to_sup)
    b_reg, b_sup = get_regs_sups(base_toks,  tok_to_reg, reg_to_sup)
    sr_gb, ss_gb = is_same_reg_sup(g_reg, g_sup, b_reg, b_sup, unk_region, unk_super)

    eligible = covered & base_wrong
    if sf == "same_region_confuser":
        eligible &= sr_gb
    elif sf == "same_superregion_confuser":
        eligible &= ss_gb
    elif sf == "same_region_or_superregion_confuser":
        eligible &= (sr_gb | ss_gb)

    elig_rows = np.where(eligible)[0]
    n_eligible = len(elig_rows)
    print(f"  Eligible rows: {n_eligible} / {max_rows}  ({100*n_eligible/max(1,max_rows):.1f}%)")

    rng = np.random.default_rng(args.seed)

    for ri in elig_rows:
        # Positive: challenger = gold
        rec_row_idx.append(ri)
        rec_chal_toks.append(int(gold_toks[ri]))
        rec_base_toks.append(int(base_toks[ri]))
        rec_gold_toks.append(int(gold_toks[ri]))
        rec_labels.append(1)
        rec_rank_c.append(int(gold_idx[ri]))

        # Negatives: wrong candidates != base_top1, != gold
        cand_mask = np.ones(top_k, bool)
        cand_mask[0] = False                              # skip position 0 (base_top1)
        for ki in range(top_k):
            tid = int(topk_all[ri, ki])
            if tid == int(base_toks[ri]) or tid == int(gold_toks[ri]):
                cand_mask[ki] = False

        cand_ks = np.where(cand_mask)[0]
        if len(cand_ks) == 0:
            continue

        c_toks_neg = topk_all[ri, cand_ks]
        c_reg_neg, c_sup_neg = get_regs_sups(c_toks_neg, tok_to_reg, reg_to_sup)
        same_br = (c_reg_neg == b_reg[ri]) & (c_reg_neg != unk_region)
        same_bs = (c_sup_neg == b_sup[ri]) & (c_sup_neg != unk_super)
        pref = (same_br | same_bs).astype(np.float32) * 1000 - cand_ks.astype(np.float32)
        order = np.argsort(-pref)[:n_neg]
        for j in order:
            rec_row_idx.append(ri)
            rec_chal_toks.append(int(c_toks_neg[j]))
            rec_base_toks.append(int(base_toks[ri]))
            rec_gold_toks.append(int(gold_toks[ri]))
            rec_labels.append(0)
            rec_rank_c.append(int(cand_ks[j]))

    n_total = len(rec_labels)
    n_pos   = int(sum(rec_labels))
    n_neg_r = n_total - n_pos
    print(f"  Pairs: {n_total}  pos={n_pos}  neg={n_neg_r}")

    # Limit to max_pair_records
    if n_total > args.max_pair_records:
        pos_idx = [i for i, l in enumerate(rec_labels) if l == 1]
        neg_idx = [i for i, l in enumerate(rec_labels) if l == 0]
        keep_pos = min(len(pos_idx), args.max_pair_records // 2)
        keep_neg = min(len(neg_idx), args.max_pair_records - keep_pos)
        kept_pos = rng.choice(pos_idx, keep_pos, replace=False).tolist()
        kept_neg = rng.choice(neg_idx, keep_neg, replace=False).tolist()
        keep = sorted(kept_pos + kept_neg)
        rec_row_idx   = [rec_row_idx[i]   for i in keep]
        rec_chal_toks = [rec_chal_toks[i] for i in keep]
        rec_base_toks = [rec_base_toks[i] for i in keep]
        rec_gold_toks = [rec_gold_toks[i] for i in keep]
        rec_labels    = [rec_labels[i]    for i in keep]
        rec_rank_c    = [rec_rank_c[i]    for i in keep]
        n_total = len(rec_labels)
        n_pos   = sum(rec_labels)
        print(f"  Subsampled to {n_total}  pos={n_pos}  neg={n_total-n_pos}")

    # Build all pair keys in batches
    print(f"  Building {n_total} pair keys (key_dim={builder.key_dim})...")
    row_idx_np  = np.array(rec_row_idx,   dtype=np.int32)
    chal_np     = np.array(rec_chal_toks, dtype=np.int32)
    base_np     = np.array(rec_base_toks, dtype=np.int32)
    rank_np     = np.array(rec_rank_c,    dtype=np.int32)
    labels_np   = np.array(rec_labels,    dtype=np.uint8)

    # Fit scalar stats on a mixed sample of ALL pair records (not positives only)
    STAT_N = min(n_total, 200000)
    stat_idx = np.arange(n_total)
    if n_total > STAT_N:
        stat_idx = rng.choice(stat_idx, STAT_N, replace=False)
    ri_s = row_idx_np[stat_idx]
    ch_s = chal_np[stat_idx]
    ba_s = base_np[stat_idx]
    rk_s = rank_np[stat_idx]
    hp_s = hp_all[ri_s]
    ec_s = tok_w_np[ch_s]
    eb_s = tok_w_np[ba_s]
    lc_s = lgt_all[ri_s, rk_s]
    lb_s = lgt_all[ri_s, 0]
    cr_s, cs_s = get_regs_sups(ch_s, tok_to_reg, reg_to_sup)
    br_s, bs_s = get_regs_sups(ba_s, tok_to_reg, reg_to_sup)
    scalar_sample = compute_pair_scalars(
        hp_s, ec_s, eb_s, lc_s, lb_s, rk_s,
        args.top_k, cr_s, cs_s, br_s, bs_s, unk_region, unk_super)
    builder.fit_scalars(scalar_sample)
    pos_rate = float(labels_np[stat_idx].mean())
    print(f"  Scalar stats fit on {STAT_N} mixed pair records  pos_rate={pos_rate:.4f}")

    BATCH = 8192
    key_chunks = []
    for s in range(0, n_total, BATCH):
        e   = min(s + BATCH, n_total)
        idx = slice(s, e)
        ri_b  = row_idx_np[idx]
        ch_b  = chal_np[idx]
        ba_b  = base_np[idx]
        rk_b  = rank_np[idx]

        hp_b  = hp_all[ri_b]
        ec_b  = tok_w_np[ch_b]
        eb_b  = tok_w_np[ba_b]
        lc_b  = lgt_all[ri_b, rk_b]
        lb_b  = lgt_all[ri_b, 0]
        cr_b, cs_b = get_regs_sups(ch_b, tok_to_reg, reg_to_sup)
        br_b, bs_b = get_regs_sups(ba_b, tok_to_reg, reg_to_sup)

        sc_b = compute_pair_scalars(hp_b, ec_b, eb_b, lc_b, lb_b, rk_b,
                                    args.top_k, cr_b, cs_b, br_b, bs_b,
                                    unk_region, unk_super)
        sc_std = builder.standardize(sc_b)

        with torch.no_grad():
            keys = builder.build_keys(
                torch.from_numpy(hp_b),
                torch.from_numpy(ec_b),
                torch.from_numpy(eb_b),
                torch.from_numpy(sc_std),
            )
        key_chunks.append(keys.cpu())

    pair_keys = torch.cat(key_chunks, 0)  # [N, key_dim]

    meta = [
        {"row_id": int(rec_row_idx[i]), "challenger": int(rec_chal_toks[i]),
         "base": int(rec_base_toks[i]), "gold": int(rec_gold_toks[i]),
         "label": int(rec_labels[i])}
        for i in range(n_total)
    ]

    ds_meta = {
        "n_eligible_rows": n_eligible, "n_total": n_total,
        "n_pos": int(n_pos), "n_neg": n_total - int(n_pos),
        "max_store_rows": max_rows,
    }
    return pair_keys, labels_np, meta, ds_meta


# ─────────────────────────────────────────────────────────────────────────────
# KNN index
# ─────────────────────────────────────────────────────────────────────────────

class KNNIndex:
    def __init__(self, keys: torch.Tensor, labels: np.ndarray,
                 k: int, tau: float, backend: str = "auto", chunk_size: int = 131072):
        self.k = k; self.tau = tau
        self.labels_f = labels.astype(np.float32)
        self.chunk_size = chunk_size
        keys_f32 = keys.float().numpy()
        self.backend = backend
        self._index = None
        if backend in ("auto", "faiss"):
            try:
                import faiss
                idx = faiss.IndexFlatIP(keys.shape[1])
                idx.add(keys_f32)
                self._index = idx
                self.backend = "faiss"
                print(f"  [KNN] FAISS IndexFlatIP  n={len(labels)}")
                return
            except ImportError:
                if backend == "faiss":
                    raise
        self._keys = keys.float()
        self.backend = "torch_chunked"
        print(f"  [KNN] torch_chunked  n={len(labels)}")

    def query(self, qkeys: torch.Tensor):
        """qkeys: [Q, d]. Returns mem_scores [Q], pos_counts [Q],
           max_sims [Q], nn_inds [Q, k], nn_sims [Q, k]."""
        Q   = qkeys.shape[0]
        q32 = qkeys.float()
        if self.backend == "faiss":
            sims, inds = self._index.search(q32.numpy(), self.k)
        else:
            best_sims = torch.full((Q, self.k), -1e9)
            best_inds = torch.zeros(Q, self.k, dtype=torch.long)
            N = len(self.labels_f)
            for s in range(0, N, self.chunk_size):
                e   = min(s + self.chunk_size, N)
                sim = q32 @ self._keys[s:e].T          # [Q, chunk]
                combined = torch.cat([best_sims, sim], dim=1)
                ci_base  = torch.cat([best_inds,
                                      torch.arange(s, e).unsqueeze(0).expand(Q, -1)], dim=1)
                topk_v, topk_i = combined.topk(self.k, dim=1)
                best_sims = topk_v
                best_inds = ci_base.gather(1, topk_i)
            sims = best_sims.numpy(); inds = best_inds.numpy()
        w = np.exp(sims / self.tau)
        w = w / (w.sum(1, keepdims=True) + 1e-12)
        lbl = self.labels_f[inds]          # [Q, k]
        mem_scores = (w * lbl).sum(1)
        pos_counts = (lbl > 0.5).sum(1).astype(np.int32)
        max_sims   = sims.max(1)
        return mem_scores, pos_counts, max_sims, inds, sims


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pool
# ─────────────────────────────────────────────────────────────────────────────

def get_candidate_pool(topk_row, b_reg, b_sup, tok_to_reg, reg_to_sup,
                       unk_region, unk_super, cand_filter, pool_size):
    """Returns list of (ki, token_id) for candidates. No gold used."""
    cands = []
    for ki in range(1, len(topk_row)):
        cid = int(topk_row[ki])
        if cid == int(topk_row[0]):
            continue
        c_reg = tok_to_reg[min(cid, len(tok_to_reg) - 1)]
        c_sup = reg_to_sup[min(c_reg, len(reg_to_sup) - 1)]
        if cand_filter == "top_rank":
            cands.append(ki)
        elif cand_filter == "same_region_only":
            if c_reg == b_reg and c_reg != unk_region:
                cands.append(ki)
        elif cand_filter == "same_superregion_only":
            if c_sup == b_sup and c_sup != unk_super:
                cands.append(ki)
        elif cand_filter == "same_region_or_superregion":
            if ((c_reg == b_reg and c_reg != unk_region) or
                    (c_sup == b_sup and c_sup != unk_super)):
                cands.append(ki)
        if len(cands) >= pool_size:
            break
    if not cands:  # fallback
        cands = list(range(1, min(pool_size + 1, len(topk_row))))
    return cands[:pool_size]


# ─────────────────────────────────────────────────────────────────────────────
# Val evaluation — compute memory_score per candidate for all val rows
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_val(val_data, tok_w_np, tok_to_reg, reg_to_sup,
                 unk_region, unk_super, builder: PairFeatureBuilder,
                 knn_index: KNNIndex, args):
    print("\n[eval] Evaluating val rows...")
    N        = val_data["h_prime"].shape[0]
    top_k    = args.top_k
    cf       = args.candidate_filter
    ps       = args.candidate_pool_size
    qbatch   = args.query_batch_pairs

    hp_all   = val_data["h_prime"].float().numpy()
    topk_all = val_data["topk_ids"].numpy()
    lgt_all  = val_data["topk_lgt"].numpy()
    gold_all = val_data["gold"].numpy()

    base_toks = topk_all[:, 0]
    gold_toks = gold_all

    gold_in_topk = (topk_all == gold_toks[:, None])
    covered      = gold_in_topk.any(1)
    base_wrong   = (base_toks != gold_toks)
    gold_idx_in_k = gold_in_topk.argmax(1).astype(np.int32)

    g_reg, g_sup = get_regs_sups(gold_toks,  tok_to_reg, reg_to_sup)
    b_reg, b_sup = get_regs_sups(base_toks,  tok_to_reg, reg_to_sup)
    sr_gb, ss_gb = is_same_reg_sup(g_reg, g_sup, b_reg, b_sup, unk_region, unk_super)

    rng = np.random.default_rng(args.seed)

    # Per-row results
    row = {
        "covered":          covered,
        "base_wrong":       base_wrong,
        "same_reg_gb":      sr_gb,
        "same_sup_gb":      ss_gb,
        "gold_tok":         gold_toks,
        "base_tok":         base_toks,
        "gold_base_rank":   np.where(covered, gold_idx_in_k, -1),
        "selected_tok":     np.zeros(N, np.int64),
        "selected_ki":      np.zeros(N, np.int32),
        "best_mem_score":   np.zeros(N, np.float32),
        "gold_mem_score":   np.full(N, np.nan, np.float32),
        "gold_pool_rank":   np.full(N, -1, np.int32),
        "gold_in_pool":     np.zeros(N, bool),
        "gold_top4_mem":    np.zeros(N, bool),
        "pool_size":        np.zeros(N, np.int32),
        "rank2_tok":        np.zeros(N, np.int64),
        "rand_tok":         np.zeros(N, np.int64),
        "max_sim":          np.zeros(N, np.float32),
        "topk_lgt":         lgt_all,
        "topk_ids":         topk_all,
    }

    cand_info_all  = [None] * N   # for examples
    nn_info_all    = [None] * N

    ROWBATCH = 512
    all_query_keys = []
    all_query_rowmap = []   # (global_row_idx, local_cand_idx_in_pool)
    all_query_ki     = []   # ki in topK

    t0 = time.time()

    # Two-pass: collect all query pairs, then batch KNN
    print("  Pass 1: building pair keys for all candidates...")
    cand_pools_all = [None] * N

    for b0 in range(0, N, ROWBATCH):
        b1  = min(b0 + ROWBATCH, N)
        bsz = b1 - b0

        hp_b   = hp_all[b0:b1]
        topk_b = topk_all[b0:b1]
        lgt_b  = lgt_all[b0:b1]

        row_hp_list  = []; row_ec_list = []; row_eb_list = []
        row_sc_list  = []; row_mapinfo = []

        for ri in range(bsz):
            gi = b0 + ri
            b_reg_ri = int(b_reg[gi]); b_sup_ri = int(b_sup[gi])
            pool_ki = get_candidate_pool(
                topk_b[ri], b_reg_ri, b_sup_ri,
                tok_to_reg, reg_to_sup, unk_region, unk_super, cf, ps)
            cand_pools_all[gi] = pool_ki
            row["pool_size"][gi] = len(pool_ki)

            # Baselines
            row["rank2_tok"][gi] = int(topk_b[ri, pool_ki[0]]) if pool_ki else int(topk_b[ri, 1])
            rand_ki = int(rng.choice(pool_ki)) if pool_ki else 1
            row["rand_tok"][gi]  = int(topk_b[ri, rand_ki])

            # Gold-in-pool check
            gold_ri = int(gold_toks[gi])
            pool_toks = [int(topk_b[ri, ki]) for ki in pool_ki]
            in_pool   = (gold_ri in pool_toks) and bool(covered[gi])
            row["gold_in_pool"][gi] = in_pool

            eb_ri = tok_w_np[[int(base_toks[gi])]]  # [1, d]

            for local_j, ki in enumerate(pool_ki):
                cid   = int(topk_b[ri, ki])
                ec_ri = tok_w_np[[cid]]
                lc    = float(lgt_b[ri, ki])
                lb    = float(lgt_b[ri, 0])
                c_reg = tok_to_reg[min(cid, len(tok_to_reg) - 1)]
                c_sup = reg_to_sup[min(c_reg, len(reg_to_sup) - 1)]
                srn   = float((c_reg == b_reg_ri) and c_reg != unk_region)
                ssn   = float((c_sup == b_sup_ri) and c_sup != unk_super)
                hp_ri = hp_b[ri:ri+1]
                hp_n  = hp_ri / (np.linalg.norm(hp_ri) + 1e-8)
                ec_n  = ec_ri / (np.linalg.norm(ec_ri) + 1e-8)
                eb_n  = eb_ri / (np.linalg.norm(eb_ri) + 1e-8)
                hpc   = float((hp_n * ec_n).sum())
                hpb   = float((hp_n * eb_n).sum())
                sc_ri = np.array([[lc - lb, ki / top_k, srn, ssn, hpc - hpb]], np.float32)

                row_hp_list.append(hp_ri)
                row_ec_list.append(ec_ri)
                row_eb_list.append(eb_ri)
                row_sc_list.append(sc_ri)
                row_mapinfo.append((gi, local_j, ki, cid))

        if not row_mapinfo:
            continue

        hp_t  = torch.from_numpy(np.concatenate(row_hp_list, 0))
        ec_t  = torch.from_numpy(np.concatenate(row_ec_list, 0))
        eb_t  = torch.from_numpy(np.concatenate(row_eb_list, 0))
        sc_np = np.concatenate(row_sc_list, 0)
        sc_t  = torch.from_numpy(builder.standardize(sc_np))

        with torch.no_grad():
            keys = builder.build_keys(hp_t, ec_t, eb_t, sc_t).cpu()

        all_query_keys.append(keys)
        all_query_rowmap.extend(row_mapinfo)

        if b0 % 10000 == 0 and b0 > 0:
            print(f"  {b0}/{N} rows  pairs_so_far={sum(len(cand_pools_all[i]) for i in range(b0) if cand_pools_all[i])}  t={time.time()-t0:.0f}s")

    # Batch KNN search
    print(f"  Pass 2: KNN search on {len(all_query_rowmap)} candidate pairs...")
    query_keys_full = torch.cat(all_query_keys, 0) if all_query_keys else torch.zeros(0, builder.key_dim)
    P = query_keys_full.shape[0]

    all_ms = np.zeros(P, np.float32)
    all_pc = np.zeros(P, np.int32)
    all_mx = np.zeros(P, np.float32)
    all_nni = np.zeros((P, knn_index.k), np.int64)
    all_nns = np.zeros((P, knn_index.k), np.float32)

    for qs in range(0, P, qbatch):
        qe = min(qs + qbatch, P)
        ms, pc, mx, nni, nns = knn_index.query(query_keys_full[qs:qe])
        all_ms[qs:qe]   = ms
        all_pc[qs:qe]   = pc
        all_mx[qs:qe]   = mx
        all_nni[qs:qe]  = nni
        all_nns[qs:qe]  = nns
        if qs % 100000 == 0 and qs > 0:
            print(f"  KNN {qs}/{P}  t={time.time()-t0:.0f}s")

    # Aggregate per row
    print("  Aggregating per-row results...")
    row_scores   = {}  # gi -> list of (local_j, ki, cid, ms, nni_row, nns_row)
    for p_idx, (gi, local_j, ki, cid) in enumerate(all_query_rowmap):
        if gi not in row_scores:
            row_scores[gi] = []
        row_scores[gi].append((local_j, ki, cid,
                                float(all_ms[p_idx]), float(all_mx[p_idx]),
                                all_nni[p_idx], all_nns[p_idx]))

    for gi, entries in row_scores.items():
        scores_only = [e[3] for e in entries]
        best_local  = int(np.argmax(scores_only))
        best_entry  = entries[best_local]
        best_ki     = best_entry[1]
        best_cid    = best_entry[2]
        best_score  = best_entry[3]

        row["selected_tok"][gi]  = best_cid
        row["selected_ki"][gi]   = best_ki
        row["best_mem_score"][gi] = best_score
        row["max_sim"][gi]        = best_entry[4]

        # Gold mem score
        gold_ri = int(gold_toks[gi])
        for e in entries:
            if e[2] == gold_ri:
                row["gold_mem_score"][gi] = e[3]
                row["gold_pool_rank"][gi] = int(e[0])
                break

        # Top-4 by memory
        top4_local = np.argsort(scores_only)[-4:]
        top4_cids  = [entries[j][2] for j in top4_local]
        row["gold_top4_mem"][gi] = (gold_ri in top4_cids) and bool(covered[gi])

        # Store candidate info for examples
        pool_ki = cand_pools_all[gi]
        topk_gi = topk_all[gi]
        lgt_gi  = lgt_all[gi]
        cinfo = []
        for e in sorted(entries, key=lambda x: -x[3]):
            tid = e[2]; ki_e = e[1]
            c_reg = tok_to_reg[min(tid, len(tok_to_reg) - 1)]
            c_sup = reg_to_sup[min(c_reg, len(reg_to_sup) - 1)]
            cinfo.append({"ki": ki_e, "token_id": tid,
                          "base_logit": float(lgt_gi[ki_e]),
                          "region": int(c_reg), "superregion": int(c_sup),
                          "memory_score": e[3],
                          "is_gold": (tid == int(gold_toks[gi])),
                          "selected": (e[1] == best_ki)})
        cand_info_all[gi]  = cinfo
        nn_info_all[gi]    = (best_entry[5], best_entry[6])  # nn_inds, nn_sims

    print(f"  Done.  t={time.time()-t0:.0f}s")
    return row, cand_info_all, nn_info_all


# ─────────────────────────────────────────────────────────────────────────────
# Metrics per threshold
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(row, threshold: float, args) -> dict:
    N     = len(row["covered"])
    cov   = row["covered"]
    bw    = row["base_wrong"]
    sr    = row["same_reg_gb"]
    ss    = row["same_sup_gb"]
    gold  = row["gold_tok"]
    base  = row["base_tok"]
    sel   = row["selected_tok"]
    score = row["best_mem_score"]
    r2    = row["rank2_tok"]
    rand  = row["rand_tok"]
    inpool = row["gold_in_pool"]
    sel_ki   = row["selected_ki"]
    topk_lgt = row["topk_lgt"]
    topk_ids = row["topk_ids"]

    applied = score >= threshold
    md = 0.5 * args.margin_delta

    # Compute actual refined top1 via candidate-logit surgery (not sel==gold shortcut)
    ref_lgt = topk_lgt.copy()
    applied_idx = np.where(applied)[0]
    if len(applied_idx) > 0:
        ski = sel_ki[applied_idx]
        # guard: only move logit if selected candidate is not position-0 (base_top1)
        valid = ski != 0
        ref_lgt[applied_idx[valid], ski[valid]] += md
        ref_lgt[applied_idx, 0] -= md

    ref_top1_idx = ref_lgt.argmax(axis=1)
    ref_top1_tok = topk_ids[np.arange(N), ref_top1_idx]
    base_top1_tok = topk_ids[:, 0]

    refined_top1_acc = float((ref_top1_tok == gold).mean())
    base_top1_acc    = float((base_top1_tok == gold).mean())

    # CTG and CAW use actual post-surgery top1 token, not sel proxy
    ctg   = (base_top1_tok != gold) & (ref_top1_tok == gold)
    caw_g = (base_top1_tok == gold) & (ref_top1_tok != gold)

    target = cov & bw & (sr | ss)
    n_tgt  = int(target.sum())
    noharm = cov & ~bw
    n_nh   = int(noharm.sum())

    bwcov = cov & bw

    return {
        "threshold":               threshold,
        "n_rows":                  N,
        "apply_rate":              float(applied.mean()),
        "base_top1_acc":           base_top1_acc,
        "refined_top1_acc":        refined_top1_acc,
        "top1_acc_gain":           refined_top1_acc - base_top1_acc,
        "changed_to_gold_rate":    float(ctg.mean()),
        "changed_away_rate":       float(caw_g.mean()),
        # Selector accuracy (threshold-free: does KNN pick the right challenger?)
        "knn_sel_gold_bwcov":      float((sel[bwcov] == gold[bwcov]).mean()) if bwcov.any() else float("nan"),
        "rank2_sel_gold_bwcov":    float((r2[bwcov]  == gold[bwcov]).mean()) if bwcov.any() else float("nan"),
        "random_sel_gold_bwcov":   float((rand[bwcov] == gold[bwcov]).mean()) if bwcov.any() else float("nan"),
        # Target subset
        "n_target":                n_tgt,
        "target_gold_in_pool_rate":float(inpool[target].mean()) if target.any() else float("nan"),
        "target_apply_rate":       float(applied[target].mean()) if target.any() else float("nan"),
        "target_sel_gold_rate":    float((sel[target] == gold[target]).mean()) if target.any() else float("nan"),
        "target_ctg_rate":         float(ctg[target].mean()) if target.any() else float("nan"),
        "target_rank2_gold_rate":  float((r2[target]  == gold[target]).mean()) if target.any() else float("nan"),
        "target_rand_gold_rate":   float((rand[target] == gold[target]).mean()) if target.any() else float("nan"),
        # No-harm: uses actual ref_top1 not sel proxy
        "n_noharm":                n_nh,
        "noharm_apply_rate":       float(applied[noharm].mean()) if noharm.any() else float("nan"),
        "noharm_changed_away":     float(caw_g[noharm].mean())   if noharm.any() else float("nan"),
        # Pool
        "gold_in_pool_rate":       float(inpool[bwcov].mean()) if bwcov.any() else float("nan"),
        "gold_top4_mem_rate":      float(row["gold_top4_mem"][bwcov].mean()) if bwcov.any() else float("nan"),
        "mean_gold_mem_score":     float(np.nanmean(row["gold_mem_score"][bwcov])) if bwcov.any() else float("nan"),
        "mean_best_mem_score":     float(score[applied].mean()) if applied.any() else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full-vocab eval
# ─────────────────────────────────────────────────────────────────────────────

def eval_full_vocab(val_data, tok_w: torch.Tensor, row, threshold: float,
                    args, device: torch.device) -> dict:
    N      = val_data["h_prime"].shape[0]
    BSZ    = 64
    md     = args.margin_delta * 0.5
    tok_t  = tok_w.to(device)
    gold_t = row["gold_tok"]
    sel_t  = row["selected_tok"]
    bas_t  = row["base_tok"]
    applied = row["best_mem_score"] >= threshold

    tot_nll_b = tot_nll_r = tot_acc_b = tot_acc_r = 0.0
    with torch.no_grad():
        for s in range(0, N, BSZ):
            e   = min(s + BSZ, N)
            b   = e - s
            hp  = val_data["h_prime"][s:e].float().to(device)
            gld = torch.from_numpy(gold_t[s:e]).to(device)

            fv_b = hp @ tok_t.T        # [b, VS]
            fv_r = fv_b.clone()
            for ri in range(b):
                gi = s + ri
                if applied[gi]:
                    fv_r[ri, int(sel_t[gi])] += md
                    fv_r[ri, int(bas_t[gi])] -= md

            ar = torch.arange(b, device=device)
            tot_nll_b += (-F.log_softmax(fv_b, -1)[ar, gld]).sum().item()
            tot_nll_r += (-F.log_softmax(fv_r, -1)[ar, gld]).sum().item()
            tot_acc_b += (fv_b.argmax(-1) == gld).sum().item()
            tot_acc_r += (fv_r.argmax(-1) == gld).sum().item()

    return {
        "full_vocab_base_nll":         tot_nll_b / N,
        "full_vocab_refined_nll":      tot_nll_r / N,
        "full_vocab_gain":             (tot_nll_b - tot_nll_r) / N,
        "full_vocab_top1_acc_base":    tot_acc_b / N,
        "full_vocab_top1_acc_refined": tot_acc_r / N,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Examples
# ─────────────────────────────────────────────────────────────────────────────

def _tok_str(tid, tokenizer):
    if tokenizer is None: return str(tid)
    try: return repr(tokenizer.decode([tid]))
    except Exception: return str(tid)


def _write_example_block(f, enum, ri, val_data, row, cand_info_all,
                         nn_info_all, pair_meta, tokenizer):
    gold  = int(row["gold_tok"][ri])
    base  = int(row["base_tok"][ri])
    sel   = int(row["selected_tok"][ri])
    score = float(row["best_mem_score"][ri])
    gsc   = row["gold_mem_score"][ri]
    gbr   = int(row["gold_base_rank"][ri])

    f.write(f"## Example {enum+1} (row {ri})\n\n")
    if "input_ids" in val_data:
        iids = val_data["input_ids"][ri].tolist()
        ctx  = tokenizer.decode(iids[-64:]) if tokenizer else str(iids[-64:])
        f.write(f"**Context:** `{ctx}`\n\n")
    f.write(f"- gold:      `{_tok_str(gold, tokenizer)}` (id={gold})\n")
    f.write(f"- base_top1: `{_tok_str(base, tokenizer)}` (id={base})\n")
    f.write(f"- selected:  `{_tok_str(sel,  tokenizer)}` (id={sel})\n")
    f.write(f"- best_mem_score: {score:.4f}\n")
    if not math.isnan(gsc):
        f.write(f"- gold_mem_score: {gsc:.4f}\n")
    f.write(f"- gold_base_rank: {gbr}\n\n")

    cands = cand_info_all[ri]
    if cands:
        f.write("**Candidates (sorted by memory_score):**\n\n")
        f.write("| Token | ID | Ki | BaseLogit | R | SR | MemScore | Gold | Sel |\n")
        f.write("|-------|----|----|-----------|---|----|---------:|------|-----|\n")
        for c in cands[:10]:
            gm = "✓" if c["is_gold"] else ""; sm = "★" if c["selected"] else ""
            f.write(f"| `{_tok_str(c['token_id'], tokenizer)}` | {c['token_id']} "
                    f"| {c['ki']} | {c['base_logit']:.3f} "
                    f"| {c['region']} | {c['superregion']} "
                    f"| {c['memory_score']:.4f} | {gm} | {sm} |\n")
        f.write("\n")

    nn_info = nn_info_all[ri]
    if nn_info is not None:
        nni, nns = nn_info
        f.write("**KNN neighbors (for selected candidate):**\n\n")
        f.write("| Sim | Label | Challenger | Base | Gold | RowID |\n")
        f.write("|-----|-------|-----------|------|------|-------|\n")
        for j in range(min(8, len(nni))):
            idx = int(nni[j]); ns = float(nns[j])
            if idx < len(pair_meta):
                m = pair_meta[idx]
                f.write(f"| {ns:.4f} | {m['label']} "
                        f"| `{_tok_str(m['challenger'], tokenizer)}` "
                        f"| `{_tok_str(m['base'],       tokenizer)}` "
                        f"| `{_tok_str(m['gold'],       tokenizer)}` "
                        f"| {m['row_id']} |\n")
        f.write("\n")
    f.write("---\n\n")


def write_examples(run_dir, val_data, row, cand_info_all, nn_info_all,
                   pair_meta, args, threshold):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        tokenizer = None

    cov   = row["covered"]; bw = row["base_wrong"]
    sel   = row["selected_tok"]; gold = row["gold_tok"]
    score = row["best_mem_score"]
    applied = score >= threshold
    rng = np.random.default_rng(42)

    def _sample(mask, n):
        idx = np.where(mask)[0]
        if len(idx) > n: idx = rng.choice(idx, n, replace=False)
        return idx

    n = args.num_examples
    buckets = {
        "examples_selected_gold.md":        ("KNN Selected Gold", _sample(cov & bw & applied & (sel == gold), n)),
        "examples_selected_wrong.md":       ("KNN Selected Wrong (applied but wrong)", _sample(cov & bw & applied & (sel != gold), n)),
        "examples_noharm_false_apply.md":   ("No-harm False Apply (base correct)", _sample(cov & ~bw & applied, n)),
        "examples_high_memory_score.md":    ("High Memory Score", np.argsort(-score)[:n]),
    }

    for fname, (title, indices) in buckets.items():
        path = os.path.join(run_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n_{len(indices)} examples_\n\n---\n\n")
            for enum, ri in enumerate(indices):
                _write_example_block(f, enum, int(ri), val_data, row,
                                     cand_info_all, nn_info_all, pair_meta, tokenizer)


# ─────────────────────────────────────────────────────────────────────────────
# Summary / CSV
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj); return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj


def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def fmt(v, spec=".4f"):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return "nan"
    if isinstance(v, float): return format(v, spec)
    return str(v)


def write_summary(run_dir, all_metrics, fv_by_thr, ds_meta, args):
    best_fv_thr = None; best_fv = float("-inf")
    best_sc_thr = None; best_sc = float("-inf")
    for m in all_metrics:
        fvg = fv_by_thr.get(m["threshold"], {}).get("full_vocab_gain", float("nan"))
        sc  = m["changed_to_gold_rate"] - 2 * m["changed_away_rate"]
        if not math.isnan(fvg) and fvg > best_fv: best_fv = fvg; best_fv_thr = m["threshold"]
        if sc > best_sc: best_sc = sc; best_sc_thr = m["threshold"]

    rep_thr = best_sc_thr or all_metrics[0]["threshold"]
    rep = next(m for m in all_metrics if m["threshold"] == rep_thr)
    fv_rep = fv_by_thr.get(rep_thr, {})

    lines = ["# KNN Pair Selector V1 — Summary\n",
             f"**Run:** `{args.run_name}`  "
             f"|  **store_filter:** `{args.store_filter}`  "
             f"|  **cand_filter:** `{args.candidate_filter}`  "
             f"|  **knn_k:** {args.knn_k}  |  **tau:** {args.knn_tau}\n",
             "## Oracle V3 Reference\n",
             "| Metric | V3 Oracle |\n|--------|----------|\n"
             "| target_changed_to_gold | 0.2407 |\n"
             "| target_pairwise_win | 0.2707 |\n"
             "| oracle_full_vocab_gain | +0.1584 |\n"
             "| base_correct_changed_away | 0.0000 |\n",
             "## Datastore\n"]
    for k, v in ds_meta.items():
        lines.append(f"  {k:30s} = {v}")
    lines.append("")

    lines.append("## Threshold Sweep\n")
    lines.append("| Thr | apply | ctg | caw | acc_gain | knn_sel_gold | rank2_sel | fv_gain |")
    lines.append("|-----|-------|-----|-----|----------|-------------|-----------|---------|")
    for m in all_metrics:
        fvg = fv_by_thr.get(m["threshold"], {}).get("full_vocab_gain", float("nan"))
        lines.append(f"| {m['threshold']} | {fmt(m['apply_rate'])} "
                     f"| {fmt(m['changed_to_gold_rate'])} | {fmt(m['changed_away_rate'])} "
                     f"| {fmt(m['top1_acc_gain'])} "
                     f"| {fmt(m['knn_sel_gold_bwcov'])} | {fmt(m['rank2_sel_gold_bwcov'])} "
                     f"| {fmt(fvg)} |")
    lines.append("")

    lines += [
        "## Analysis\n",
        f"Representative threshold: **{rep_thr}** (best ctg-2*caw={fmt(best_sc)})\n",
        "### 1. Is gold in the candidate pool?\n",
        f"  gold_in_pool_rate = {fmt(rep['gold_in_pool_rate'])}\n"
        f"  gold_top4_mem_rate = {fmt(rep['gold_top4_mem_rate'])}\n",
        "### 2. Does KNN select gold better than rank2/random?\n",
        f"  knn_sel_gold   = {fmt(rep['knn_sel_gold_bwcov'])}\n"
        f"  rank2_sel_gold = {fmt(rep['rank2_sel_gold_bwcov'])}\n"
        f"  rand_sel_gold  = {fmt(rep['random_sel_gold_bwcov'])}\n",
        "### 3. Positive full-vocab gain at any threshold?\n",
    ]
    if best_fv_thr is not None and best_fv > 0:
        lines.append(f"  YES: threshold={best_fv_thr}  fv_gain={fmt(best_fv)}\n")
    else:
        lines.append(f"  Best fv_gain={fmt(best_fv)} at threshold={best_fv_thr}\n")
    lines += [
        "### 4. Is noharm changed_away low?\n",
        f"  noharm_changed_away (conditional) = {fmt(rep['noharm_changed_away'])}\n",
        "### 5. Fraction of V3 oracle target_changed_to_gold=0.2407 recovered?\n",
        f"  target_ctg = {fmt(rep['target_ctg_rate'])}  "
        f"({fmt(100 * rep['target_ctg_rate'] / 0.2407, '.1f')}% of oracle)\n",
        "## Interpretation\n",
    ]
    if rep["knn_sel_gold_bwcov"] > rep["rank2_sel_gold_bwcov"] * 1.1:
        lines.append("✅ KNN selects gold better than rank2.\n")
    else:
        lines.append("⚠️  KNN does not clearly beat rank2 for gold selection.\n")
    if best_fv > 0:
        lines.append("✅ At least one threshold gives positive full-vocab gain.\n")
    else:
        lines.append("❌ No threshold achieves positive full-vocab gain.\n")
    if rep["noharm_changed_away"] < 0.01:
        lines.append("✅ Noharm changed_away is low.\n")
    else:
        lines.append("⚠️  Noharm changed_away is non-trivial.\n")

    if fv_rep:
        lines += ["\n## Full-vocab at representative threshold\n"]
        for k, v in fv_rep.items():
            lines.append(f"  {k:40s} = {fmt(v)}")
        lines.append("")

    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(_json_safe({
            "best_fv_threshold": best_fv_thr, "best_fv_gain": best_fv,
            "best_score_threshold": best_sc_thr, "best_score": best_sc,
            "rep_threshold": rep_thr,
            "rep_metrics": rep, "rep_fv": fv_rep,
            "datastore_meta": ds_meta,
            "v3_oracle": {"target_changed_to_gold": 0.2407,
                          "target_pairwise_win": 0.2707,
                          "oracle_full_vocab_gain": 0.1584},
        }), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Arg parse
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",          required=True)
    p.add_argument("--train_dir",           required=True)
    p.add_argument("--val_dir",             required=True)
    p.add_argument("--token_to_region",     required=True)
    p.add_argument("--super_map",           default=None)
    p.add_argument("--output_root",         default="runs/knn_pair_selector_v1")
    p.add_argument("--run_name",            default="pair_memory_hprime_region_v1")
    p.add_argument("--top_k",              type=int,   default=256)
    p.add_argument("--store_filter",        default="same_region_or_superregion_confuser",
                   choices=_STORE_FILTERS)
    p.add_argument("--candidate_filter",    default="same_region_or_superregion",
                   choices=_CAND_FILTERS)
    p.add_argument("--candidate_pool_size", type=int,   default=32)
    p.add_argument("--max_store_rows",      type=int,   default=500000)
    p.add_argument("--max_pair_records",    type=int,   default=1000000)
    p.add_argument("--negatives_per_positive", type=int, default=8)
    p.add_argument("--key_dim",            type=int,   default=256)
    p.add_argument("--knn_k",             type=int,   default=32)
    p.add_argument("--knn_tau",           type=float, default=0.1)
    p.add_argument("--backend",            default="auto",
                   choices=["auto", "faiss", "torch_chunked"])
    p.add_argument("--query_batch_pairs",  type=int,   default=4096)
    p.add_argument("--index_chunk_size",   type=int,   default=131072)
    p.add_argument("--margin_delta",       type=float, default=1.0)
    p.add_argument("--thresholds",         default="0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8")
    p.add_argument("--eval_full_vocab",    action="store_true")
    p.add_argument("--num_examples",       type=int,   default=40)
    p.add_argument("--max_train_rows",     type=int,   default=None)
    p.add_argument("--max_val_rows",       type=int,   default=None)
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_dir = os.path.join(args.output_root, args.run_name)
    if os.path.exists(run_dir):
        raise RuntimeError(f"Run dir exists: {run_dir}. Use a different --run_name.")
    os.makedirs(run_dir)

    thresholds = [float(t) for t in args.thresholds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("\n[backbone] Loading token embeddings...")
    tok_w, d_model, vocab_size = load_backbone(args.small_ckpt, device)
    tok_w_np = tok_w.numpy()
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    print("\n[maps] Loading region maps...")
    t2r, r2s, unk_region, unk_super, sr_enabled = load_maps(
        args.token_to_region, args.super_map)
    tok_to_reg, reg_to_sup = build_lookup_arrays(
        t2r, r2s, unk_region, unk_super, vocab_size)

    print("\n[data] Loading train shards...")
    train_data = load_shards(args.train_dir, args.top_k, "train",
                             args.max_train_rows, load_ids=False)
    print("[data] Loading val shards...")
    val_data   = load_shards(args.val_dir,   args.top_k, "val",
                             args.max_val_rows, load_ids=True)

    builder = PairFeatureBuilder(d_model=d_model, key_dim=args.key_dim, seed=args.seed)

    pair_keys, pair_labels, pair_meta, ds_meta = build_datastore(
        train_data, tok_w_np, tok_to_reg, reg_to_sup,
        unk_region, unk_super, sr_enabled, builder, args)

    print("\n[save] Saving datastore...")
    torch.save(pair_keys.half(), os.path.join(run_dir, "pair_keys_train.pt"))
    torch.save(torch.from_numpy(pair_labels), os.path.join(run_dir, "pair_labels_train.pt"))
    torch.save(pair_meta, os.path.join(run_dir, "pair_meta_train.pt"))
    torch.save(builder.state_dict(), os.path.join(run_dir, "pair_projection.pt"))
    with open(os.path.join(run_dir, "scalar_stats.json"), "w") as f:
        json.dump({"means": builder.scalar_means.tolist(),
                   "stds":  builder.scalar_stds.tolist()}, f, indent=2)
    with open(os.path.join(run_dir, "datastore_meta.json"), "w") as f:
        json.dump(_json_safe(ds_meta), f, indent=2)

    config = vars(args).copy()
    config.update({"d_model": d_model, "vocab_size": vocab_size,
                   "sr_enabled": sr_enabled, "thresholds_list": thresholds,
                   "datastore": ds_meta})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(_json_safe(config), f, indent=2)

    print("\n[index] Building KNN index...")
    knn_index = KNNIndex(pair_keys.float(), pair_labels, k=args.knn_k,
                         tau=args.knn_tau, backend=args.backend,
                         chunk_size=args.index_chunk_size)

    row, cand_info_all, nn_info_all = evaluate_val(
        val_data, tok_w_np, tok_to_reg, reg_to_sup,
        unk_region, unk_super, builder, knn_index, args)

    print("\n[metrics] Threshold sweep...")
    all_metrics  = []
    fv_by_thr    = {}
    for thr in thresholds:
        m = compute_metrics(row, thr, args)
        all_metrics.append(m)
        print(f"  thr={thr:.2f}  apply={m['apply_rate']:.3f}  "
              f"ctg={m['changed_to_gold_rate']:.4f}  caw={m['changed_away_rate']:.4f}  "
              f"knn_sel={m['knn_sel_gold_bwcov']:.4f}  rank2={m['rank2_sel_gold_bwcov']:.4f}")
        if args.eval_full_vocab:
            fv = eval_full_vocab(val_data, tok_w, row, thr, args, device)
            fv_by_thr[thr] = fv
            print(f"    fv_gain={fv['full_vocab_gain']:.4f}")

    write_csv(os.path.join(run_dir, "threshold_sweep.csv"), all_metrics)
    write_summary(run_dir, all_metrics, fv_by_thr, ds_meta, args)

    best_sc_thr = max(all_metrics,
                      key=lambda m: m["changed_to_gold_rate"] - 2 * m["changed_away_rate"])["threshold"]
    print(f"\n[examples] Writing examples at threshold={best_sc_thr}...")
    try:
        write_examples(run_dir, val_data, row, cand_info_all, nn_info_all,
                       pair_meta, args, threshold=best_sc_thr)
    except Exception as e:
        print(f"[warn] Examples failed: {e}")

    print(f"\n{'='*60}")
    print(f" KNN Pair Selector V1 complete.")
    print(f" Run dir: {run_dir}")
    print(f" See: {run_dir}/summary.md")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
