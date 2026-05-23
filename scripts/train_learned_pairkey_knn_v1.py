#!/usr/bin/env python3
"""
train_learned_pairkey_knn_v1.py — Learned Pair-Key KNN Selector

Trains a PairKeyNet so that pair keys cluster by correction decision.
Evaluates two selectors:
  A. Parametric: score_head(key_j) over candidate pool
  B. KNN: memory_score from learned-key datastore

Reference:
  Old KNN V1: knn_sel_gold_bwcov=0.0070, rank2=0.1452, fv_gain=-0.0116
  V3 oracle:  target_ctg=0.2407, fv_gain=+0.1584
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
from contextlib import contextmanager
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(ckpt_path, device):
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(ckpt_path, device)
    tok_w = backbone.token_emb.weight.detach().float()
    del backbone
    return tok_w, d_model, vocab_size


def _parse_map(raw):
    if isinstance(raw, list):
        return {i: v for i, v in enumerate(raw) if v is not None}
    return {int(k): v for k, v in raw.items()}


def load_maps(t2r_path, super_path):
    with open(t2r_path) as f:
        t2r = _parse_map(json.load(f))
    unk_region = int(max(t2r.values())) + 1 if t2r else 1
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f:
            r2s = _parse_map(json.load(f))
        unk_super = int(max(r2s.values())) + 1 if r2s else 1
        sr_enabled = True
    else:
        r2s = {}; unk_super = 0; sr_enabled = False
    print(f"  maps: n_regions={unk_region} sr_enabled={sr_enabled} unk_super={unk_super}")
    return t2r, r2s, unk_region, unk_super, sr_enabled


def build_tok_arr(t2r, unk_region, vocab_size):
    arr = np.full(vocab_size, unk_region, dtype=np.int32)
    for tok, reg in t2r.items():
        if 0 <= tok < vocab_size:
            arr[tok] = int(reg)
    return arr


def build_reg_arr(r2s, unk_super, unk_region):
    arr = np.full(unk_region + 1, unk_super, dtype=np.int32)
    for reg, sup in r2s.items():
        if 0 <= reg <= unk_region:
            arr[reg] = int(sup)
    return arr


def _get_field(shard, *names, required=True):
    for n in names:
        if n in shard:
            return shard[n]
    if required:
        raise KeyError(f"Shard missing aliases {names}. Have: {list(shard.keys())}")
    return None


def load_shards(shard_dir, top_k, split_name, max_rows=None, load_ids=False):
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")
    all_hp, all_topk, all_lgt, all_gold, all_ids = [], [], [], [], []
    total = 0
    for sp in shards:
        if max_rows is not None and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        hp   = _get_field(sh, "h_prime", "h_ctx").float()
        topk = _get_field(sh, "base_topk_ids", "base_topk").long()
        lgt  = _get_field(sh, "base_topk_logits", "base_topk_lgt").float()
        gold = _get_field(sh, "gold_token", "gold").long()
        ids  = _get_field(sh, "input_ids", required=False) if load_ids else None
        B, K = topk.shape
        if K < top_k:
            topk = torch.cat([topk, torch.zeros(B, top_k-K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, top_k-K), float("nan"))], 1)
        elif K > top_k:
            topk, lgt = topk[:, :top_k], lgt[:, :top_k]
        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            hp, topk, lgt, gold = hp[:keep], topk[:keep], lgt[:keep], gold[:keep]
            if ids is not None: ids = ids[:keep]
            B = keep
        all_hp.append(hp); all_topk.append(topk)
        all_lgt.append(lgt); all_gold.append(gold)
        if ids is not None: all_ids.append(ids)
        total += B
    data = {
        "h_prime":  torch.cat(all_hp, 0),
        "topk_ids": torch.cat(all_topk, 0),
        "topk_lgt": torch.cat(all_lgt, 0),
        "gold":     torch.cat(all_gold, 0),
        "ids":      torch.cat(all_ids, 0) if all_ids else None,
    }
    N = data["h_prime"].shape[0]
    print(f"  {split_name}: {N:,} rows  d={data['h_prime'].shape[1]}  K={top_k}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pool
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_regs(token_ids, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled):
    vs   = tok_arr_t.shape[0]
    safe = token_ids.clamp(0, vs - 1)
    regs = tok_arr_t[safe]
    sups = reg_arr_t[regs.clamp(0, reg_arr_t.shape[0]-1)] if sr_enabled else None
    return regs, sups


def build_candidate_pool(topk_ids, topk_lgt, tok_arr_t, reg_arr_t,
                          unk_region, unk_super, sr_enabled,
                          pool_size, candidate_filter, device):
    N, K  = topk_ids.shape
    vs    = tok_arr_t.shape[0]
    rlen  = reg_arr_t.shape[0] - 1

    cand_ids  = torch.zeros(N, pool_size, dtype=torch.long,  device=device)
    cand_lgts = torch.zeros(N, pool_size, dtype=torch.float, device=device)
    cand_rnks = torch.zeros(N, pool_size, dtype=torch.long,  device=device)
    cand_regs = torch.full((N, pool_size), unk_region, dtype=torch.long, device=device)
    cand_sups = torch.full((N, pool_size), unk_super,  dtype=torch.long, device=device) if sr_enabled else None

    base_regs = tok_arr_t[topk_ids[:, 0].clamp(0, vs-1)]
    base_sups = reg_arr_t[base_regs.clamp(0, rlen)] if sr_enabled else None

    for i in range(N):
        c_toks = topk_ids[i, 1:]
        c_lgts = topk_lgt[i, 1:]
        c_rnks = torch.arange(1, K, device=device)

        if candidate_filter != "top_rank":
            cr = tok_arr_t[c_toks.clamp(0, vs-1)]
            br = base_regs[i]
            if candidate_filter == "same_region_only":
                mask = (cr == br) & (cr != unk_region)
            elif candidate_filter == "same_superregion_only":
                if sr_enabled:
                    cs = reg_arr_t[cr.clamp(0, rlen)]
                    bs = base_sups[i]
                    mask = (cs == bs) & (cs != unk_super)
                else:
                    mask = torch.zeros(len(c_toks), dtype=torch.bool, device=device)
            elif candidate_filter == "same_region_or_superregion":
                same_r = (cr == br) & (cr != unk_region)
                if sr_enabled:
                    cs = reg_arr_t[cr.clamp(0, rlen)]
                    bs = base_sups[i]
                    same_s = (cs == bs) & (cs != unk_super)
                else:
                    same_s = torch.zeros_like(same_r)
                mask = same_r | same_s
            else:
                mask = torch.ones(len(c_toks), dtype=torch.bool, device=device)
            if mask.any():
                c_toks = c_toks[mask]; c_lgts = c_lgts[mask]; c_rnks = c_rnks[mask]

        n_take = min(pool_size, len(c_toks))
        cand_ids[i, :n_take]  = c_toks[:n_take]
        cand_lgts[i, :n_take] = c_lgts[:n_take]
        cand_rnks[i, :n_take] = c_rnks[:n_take]
        cr_fill = tok_arr_t[c_toks[:n_take].clamp(0, vs-1)]
        cand_regs[i, :n_take] = cr_fill
        if sr_enabled and cand_sups is not None:
            cand_sups[i, :n_take] = reg_arr_t[cr_fill.clamp(0, rlen)]

    return cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PairKeyNet(nn.Module):
    """
    Maps (h_prime, emb_c, emb_b, logit features, region features) -> L2-normalized key.
    score_head used during training only.
    """

    def __init__(self, token_emb_weight, tok_arr, reg_arr,
                 d_model, n_regions, n_supers, sr_enabled,
                 unk_region, unk_super, top_k,
                 key_dim=128, hidden_dim=512, dropout=0.1,
                 region_emb_dim=64, super_emb_dim=32):
        super().__init__()
        self.d_model    = d_model
        self.sr_enabled = sr_enabled
        self.unk_region = unk_region
        self.unk_super  = unk_super
        self.top_k      = top_k
        self.key_dim    = key_dim

        self.register_buffer("token_emb_weight", token_emb_weight.detach().float())
        self.register_buffer("tok_arr", torch.from_numpy(tok_arr.astype(np.int32)).long())
        self.register_buffer("reg_arr", torch.from_numpy(reg_arr.astype(np.int32)).long())

        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim)
        self.super_emb  = nn.Embedding(n_supers + 2, super_emb_dim) if sr_enabled else None

        # Input features per pair:
        # h_prime(d) + emb_c(d) + emb_b(d) + (ec-eb)(d) + (ec*eb)(d)
        # + scalars: lgt_c, lgt_b, logit_gap, rank_norm, hp_dot_c, hp_dot_b, hp_dot_gap (7)
        # + reg_c(re) + reg_b(re) (same_reg: 1)
        # + sup_c(se) + sup_b(se) + same_sup(1) if sr_enabled
        feat_dim = 5 * d_model + 7 + 2 * region_emb_dim + 1
        if sr_enabled:
            feat_dim += 2 * super_emb_dim + 1

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, key_dim),
        )
        self.score_head = nn.Linear(key_dim, 1, bias=True)

    def _pair_feat(self, hp, emb_c, emb_b, lgt_c, lgt_b, rank_norm,
                   reg_c, reg_b, sup_c=None, sup_b=None):
        """Build feature vector for a single (challenger, base) pair. Batched over rows."""
        logit_gap = lgt_c - lgt_b
        hp_dot_c  = (hp * emb_c).sum(-1, keepdim=True)
        hp_dot_b  = (hp * emb_b).sum(-1, keepdim=True)
        hp_dot_gap = hp_dot_c - hp_dot_b
        same_reg  = ((reg_c == reg_b) & (reg_c != self.unk_region)).float().unsqueeze(-1)

        remb_c = self.region_emb(reg_c.clamp(0, self.tok_arr.shape[0]-1))
        remb_b = self.region_emb(reg_b.clamp(0, self.tok_arr.shape[0]-1))

        parts = [hp, emb_c, emb_b, emb_c - emb_b, emb_c * emb_b,
                 lgt_c.unsqueeze(-1), lgt_b.unsqueeze(-1),
                 logit_gap.unsqueeze(-1), rank_norm.unsqueeze(-1),
                 hp_dot_c, hp_dot_b, hp_dot_gap,
                 remb_c, remb_b, same_reg]

        if self.sr_enabled and self.super_emb is not None and sup_c is not None and sup_b is not None:
            rlen = self.reg_arr.shape[0] - 1
            semb_c = self.super_emb(sup_c.clamp(0, rlen))
            semb_b = self.super_emb(sup_b.clamp(0, rlen))
            same_sup = ((sup_c == sup_b) & (sup_c != self.unk_super)).float().unsqueeze(-1)
            parts.extend([semb_c, semb_b, same_sup])

        return torch.cat(parts, dim=-1)

    def forward_pool(self, hp, base_ids, base_lgts,
                     cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups=None):
        """
        Compute keys and scores for a full candidate pool.
        hp        [B, d]
        base_ids  [B]
        base_lgts [B, K]
        cand_ids  [B, P]
        ...
        Returns keys [B, P, key_dim], scores [B, P]
        """
        B, P  = cand_ids.shape
        K     = base_lgts.shape[1]
        vs    = self.token_emb_weight.shape[0]
        d     = self.d_model

        base_emb  = self.token_emb_weight[base_ids.clamp(0, vs-1)]           # [B, d]
        base_reg  = self.tok_arr[base_ids.clamp(0, self.tok_arr.shape[0]-1)] # [B]
        base_lgt  = base_lgts[:, 0]                                           # [B]

        if self.sr_enabled and self.super_emb is not None:
            rlen     = self.reg_arr.shape[0] - 1
            base_sup = self.reg_arr[base_reg.clamp(0, rlen)]
        else:
            base_sup = None

        cand_emb = self.token_emb_weight[
            cand_ids.reshape(-1).clamp(0, vs-1)].view(B, P, d)              # [B, P, d]

        hp_exp   = hp.unsqueeze(1).expand(B, P, d)
        be_exp   = base_emb.unsqueeze(1).expand(B, P, d)
        bl_exp   = base_lgt.unsqueeze(1).expand(B, P)
        br_exp   = base_reg.unsqueeze(1).expand(B, P)
        bs_exp   = base_sup.unsqueeze(1).expand(B, P) if base_sup is not None else None
        rn       = cand_rnks.float() / max(K, 1)

        # Flatten for batched MLP
        hp_f   = hp_exp.reshape(B*P, d)
        ec_f   = cand_emb.reshape(B*P, d)
        eb_f   = be_exp.reshape(B*P, d)
        lc_f   = cand_lgts.reshape(B*P)
        lb_f   = bl_exp.reshape(B*P)
        rn_f   = rn.reshape(B*P)
        rc_f   = cand_regs.reshape(B*P)
        rb_f   = br_exp.reshape(B*P)
        sc_f   = cand_sups.reshape(B*P) if (self.sr_enabled and cand_sups is not None) else None
        sb_f   = bs_exp.reshape(B*P) if bs_exp is not None else None

        feat   = self._pair_feat(hp_f, ec_f, eb_f, lc_f, lb_f, rn_f,
                                  rc_f, rb_f, sc_f, sb_f)                    # [B*P, feat_dim]
        proj   = self.proj(feat)                                              # [B*P, key_dim]
        keys   = F.normalize(proj, dim=-1).view(B, P, self.key_dim)          # [B, P, key_dim]
        scores = self.score_head(keys.reshape(B*P, self.key_dim)).view(B, P) # [B, P]
        return keys, scores

    def forward_single(self, hp, base_id, base_lgt_scalar, cand_id, cand_lgt, cand_rnk,
                        cand_reg, cand_sup=None):
        """Single pair key for datastore building. Batched over pairs."""
        vs = self.token_emb_weight.shape[0]
        ec = self.token_emb_weight[cand_id.clamp(0, vs-1)]
        eb = self.token_emb_weight[base_id.clamp(0, vs-1)]
        br = self.tok_arr[base_id.clamp(0, self.tok_arr.shape[0]-1)]
        if self.sr_enabled and self.super_emb is not None:
            rlen = self.reg_arr.shape[0] - 1
            base_sup_s = self.reg_arr[br.clamp(0, rlen)]
        else:
            base_sup_s = None
        rn = cand_rnk.float() / max(self.top_k, 1)
        feat = self._pair_feat(hp, ec, eb, cand_lgt, base_lgt_scalar,
                                rn, cand_reg, br, cand_sup, base_sup_s)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(scores, gold_idx, lambda_margin, target_margin, device):
    """
    scores   [B, P]   raw scores per candidate
    gold_idx [B]      index of gold candidate in pool (0-based)
    """
    valid = gold_idx >= 0
    if not valid.any():
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, {"total": 0.0, "ce": 0.0, "margin": 0.0, "n": 0}

    s_v  = scores[valid]
    g_v  = gold_idx[valid]
    L_ce = F.cross_entropy(s_v, g_v)

    L_margin = torch.tensor(0.0, device=device)
    if lambda_margin > 0 and s_v.shape[1] > 1:
        B2, P = s_v.shape
        ar = torch.arange(B2, device=device)
        s_gold = s_v[ar, g_v]
        # max wrong score
        mask = torch.ones_like(s_v, dtype=torch.bool)
        mask[ar, g_v] = False
        s_wrong = s_v.masked_fill(~mask, float("-inf")).max(1).values
        L_margin = F.softplus(target_margin - (s_gold - s_wrong)).mean()

    total = L_ce + lambda_margin * L_margin
    return total, {
        "total":  total.item(),
        "ce":     L_ce.item(),
        "margin": L_margin.item() if lambda_margin > 0 else 0.0,
        "n":      int(valid.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training rows
# ─────────────────────────────────────────────────────────────────────────────

def find_trainable_rows(data, tok_arr_t, reg_arr_t, unk_region, unk_super,
                         sr_enabled, args, device, max_rows=None):
    """
    Return indices where base_wrong & gold in candidate pool.
    Also returns gold_in_pool_idx for each such row.
    """
    N  = data["h_prime"].shape[0]
    BSZ = 4096
    valid_rows  = []
    gold_indices = []

    for s in range(0, N, BSZ):
        e    = min(s + BSZ, N)
        topk = data["topk_ids"][s:e].to(device)
        lgt  = data["topk_lgt"][s:e].to(device)
        gold = data["gold"][s:e].to(device)

        base_wrong = topk[:, 0] != gold

        cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
            topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
            args.candidate_pool_size, args.candidate_filter, device)

        gold_in_pool = (cand_ids == gold.unsqueeze(1))  # [b, P]
        pool_has_gold = gold_in_pool.any(1)
        gold_pool_idx = gold_in_pool.long().argmax(1)

        keep = (base_wrong & pool_has_gold).cpu()
        rows_local = torch.where(keep)[0]
        for ri in rows_local.tolist():
            valid_rows.append(s + ri)
            gold_indices.append(int(gold_pool_idx[ri]))
            if max_rows is not None and len(valid_rows) >= max_rows:
                break
        if max_rows is not None and len(valid_rows) >= max_rows:
            break

    print(f"  trainable rows (base_wrong & gold_in_pool): {len(valid_rows):,}")
    return np.array(valid_rows, dtype=np.int64), np.array(gold_indices, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# KNN datastore
# ─────────────────────────────────────────────────────────────────────────────

def build_datastore(model, data, tok_arr_t, reg_arr_t, unk_region, unk_super,
                     sr_enabled, train_row_idx, gold_pool_idx, args, device):
    """
    Build pair-key datastore from train rows.
    For each row: positive = gold candidate, negatives = sampled wrong candidates.
    Returns keys [M, key_dim], labels [M] (1=pos,0=neg), meta list.
    """
    model.eval()
    N_rows  = len(train_row_idx)
    P       = args.candidate_pool_size
    neg_per = args.negatives_per_positive
    BSZ     = 256
    rng     = np.random.default_rng(args.seed + 1)

    all_keys   = []
    all_labels = []
    all_meta   = []

    with torch.no_grad():
        for s in range(0, N_rows, BSZ):
            e    = min(s + BSZ, N_rows)
            ri_s = train_row_idx[s:e]
            gi_s = gold_pool_idx[s:e]

            hp   = data["h_prime"][ri_s].to(device)
            topk = data["topk_ids"][ri_s].to(device)
            lgt  = data["topk_lgt"][ri_s].to(device)

            cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
                topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
                P, args.candidate_filter, device)

            base_ids = topk[:, 0]
            keys, _ = model.forward_pool(hp, base_ids, lgt,
                                          cand_ids, cand_lgts, cand_rnks,
                                          cand_regs, cand_sups)  # [b, P, key_dim]

            b = len(ri_s)
            for bi in range(b):
                gidx = int(gi_s[bi])
                pos_key = keys[bi, gidx].cpu().numpy()
                all_keys.append(pos_key)
                all_labels.append(1)
                all_meta.append({
                    "row": int(ri_s[bi]), "label": 1,
                    "cand_tok": int(cand_ids[bi, gidx].item()),
                    "base_tok": int(base_ids[bi].item()),
                })

                # Negative candidates
                neg_pool = [j for j in range(P) if j != gidx]
                if len(neg_pool) > neg_per:
                    neg_pool = rng.choice(neg_pool, neg_per, replace=False).tolist()
                for j in neg_pool:
                    neg_key = keys[bi, j].cpu().numpy()
                    all_keys.append(neg_key)
                    all_labels.append(0)
                    all_meta.append({
                        "row": int(ri_s[bi]), "label": 0,
                        "cand_tok": int(cand_ids[bi, j].item()),
                        "base_tok": int(base_ids[bi].item()),
                    })

    keys_np   = np.array(all_keys,   dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.uint8)
    print(f"  datastore: {len(keys_np):,} records  "
          f"pos={labels_np.sum():,}  neg={(1-labels_np).sum():,}")
    model.train()
    return keys_np, labels_np, all_meta


class KNNIndex:
    def __init__(self, keys_np, labels_np, knn_tau=0.1, backend="auto"):
        self.keys     = keys_np        # [M, key_dim]
        self.labels_f = labels_np.astype(np.float32)
        self.tau      = knn_tau
        self.faiss    = None

        if backend in ("auto", "faiss"):
            try:
                import faiss as _faiss
                idx = _faiss.IndexFlatIP(keys_np.shape[1])
                idx.add(keys_np)
                self.faiss = idx
                print("  KNN backend: faiss")
            except ImportError:
                if backend == "faiss":
                    raise
                print("  KNN backend: torch_chunked (faiss unavailable)")
        else:
            print("  KNN backend: torch_chunked")

        if self.faiss is None:
            self.keys_t = torch.from_numpy(keys_np).float()

    def query(self, qkeys_np, k):
        Q = qkeys_np.shape[0]
        if self.faiss is not None:
            sims, inds = self.faiss.search(qkeys_np, k)
        else:
            qt = torch.from_numpy(qkeys_np).float()
            chunk = 4096
            all_sims, all_inds = [], []
            for s in range(0, Q, chunk):
                e  = min(s + chunk, Q)
                s_ = F.normalize(qt[s:e], dim=-1)
                si = s_ @ self.keys_t.T
                top_sims, top_inds = si.topk(min(k, self.keys_t.shape[0]), dim=-1)
                all_sims.append(top_sims.numpy())
                all_inds.append(top_inds.numpy())
            sims = np.concatenate(all_sims, 0)
            inds = np.concatenate(all_inds, 0)

        w   = np.exp(sims / self.tau)
        w   = w / (w.sum(1, keepdims=True) + 1e-12)
        lbl = self.labels_f[inds]
        mem_scores = (w * lbl).sum(1)
        return mem_scores, sims, inds


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _rate(mask, denom):
    if denom == 0: return float("nan")
    return float(np.asarray(mask, dtype=float).sum()) / denom


def _safe_mean(arr):
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) > 0 else float("nan")


def _json_safe(obj):
    if isinstance(obj, dict):  return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj); return None if math.isnan(v) else v
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj


def _write_csv_row(path, d, header=False):
    mode = "w" if header else "a"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d.keys()))
        if header: w.writeheader()
        w.writerow(d)


@contextmanager
def _amp_ctx(use_amp):
    if use_amp:
        with torch.cuda.amp.autocast(): yield
    else:
        yield


def apply_surgical(topk_lgts, topk_ids, sel_toks, apply_mask, md_half):
    """Apply +md_half to sel_toks and -md_half to position 0 for rows in apply_mask."""
    refined = topk_lgts.clone()
    if isinstance(apply_mask, np.ndarray):
        apply_mask = torch.from_numpy(apply_mask.astype(bool))
    if not apply_mask.any():
        return refined
    N = topk_lgts.shape[0]
    ar = torch.arange(N, device=topk_lgts.device)
    sel_toks_t = torch.as_tensor(sel_toks, device=topk_lgts.device)
    sel_pos = (topk_ids == sel_toks_t.unsqueeze(1)).long().argmax(1)
    sel_found = (topk_ids[ar, sel_pos] == sel_toks_t) & apply_mask.to(topk_lgts.device)
    if sel_found.any():
        refined[ar[sel_found], sel_pos[sel_found]] += md_half
        refined[ar[sel_found], 0]                  -= md_half
    return refined


def evaluate_val(model, val_data, tok_arr_t, reg_arr_t,
                  unk_region, unk_super, sr_enabled, args, device,
                  knn_index: Optional[KNNIndex] = None):
    """
    Returns per-row arrays for both parametric and KNN selectors.
    """
    model.eval()
    N, K = val_data["h_prime"].shape[0], val_data["topk_ids"].shape[1]
    P    = args.candidate_pool_size
    BSZ  = args.eval_batch_size
    md   = args.margin_delta

    # Per-row arrays
    gold_arr     = np.zeros(N, dtype=np.int64)
    base_arr     = np.zeros(N, dtype=np.int64)
    bw_arr       = np.zeros(N, dtype=bool)
    pg_arr       = np.zeros(N, dtype=bool)
    sr_arr       = np.zeros(N, dtype=bool)
    ss_arr       = np.zeros(N, dtype=bool)

    # Parametric selector output
    param_sel_tok = np.zeros(N, dtype=np.int64)
    param_max_prob = np.zeros(N, dtype=np.float32)
    param_keys_arr = np.zeros((N, P, args.key_dim), dtype=np.float32)
    param_scores_arr = np.zeros((N, P), dtype=np.float32)

    # KNN selector output (filled if knn_index is not None)
    knn_mem_scores = np.zeros((N, P), dtype=np.float32)

    # topK storage for surgical eval
    topk_ids_np  = np.zeros((N, K), dtype=np.int64)
    topk_lgts_np = np.zeros((N, K), dtype=np.float32)
    cand_ids_np  = np.zeros((N, P), dtype=np.int64)
    cand_lgts_np = np.zeros((N, P), dtype=np.float32)

    vs = tok_arr_t.shape[0]

    with torch.no_grad():
        for s in range(0, N, BSZ):
            e = min(s + BSZ, N)
            b = e - s

            hp   = val_data["h_prime"][s:e].to(device)
            topk = val_data["topk_ids"][s:e].to(device)
            lgt  = val_data["topk_lgt"][s:e].to(device)
            gold = val_data["gold"][s:e].to(device)

            cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
                topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
                P, args.candidate_filter, device)

            base_ids = topk[:, 0]
            keys, scores = model.forward_pool(
                hp, base_ids, lgt, cand_ids, cand_lgts, cand_rnks,
                cand_regs, cand_sups)                                # [b, P, key_dim], [b, P]

            # Parametric: pick argmax candidate
            probs      = scores.softmax(1)
            best_j     = probs.argmax(1)
            best_prob  = probs[torch.arange(b, device=device), best_j]
            sel_tok_p  = cand_ids[torch.arange(b, device=device), best_j]

            bw_b = (base_ids != gold).cpu().numpy()
            pg_b = (cand_ids == gold.unsqueeze(1)).any(1).cpu().numpy()

            gold_reg = tok_arr_t[gold.clamp(0, vs-1)]
            top1_reg = tok_arr_t[base_ids.clamp(0, vs-1)]
            sr_b = ((gold_reg == top1_reg) & (gold_reg != unk_region)).cpu().numpy()
            if sr_enabled:
                rlen = reg_arr_t.shape[0] - 1
                gs_sup = reg_arr_t[gold_reg.clamp(0, rlen)]
                t1_sup = reg_arr_t[top1_reg.clamp(0, rlen)]
                ss_b = ((gs_sup == t1_sup) & (gs_sup != unk_super)).cpu().numpy()
            else:
                ss_b = np.zeros(b, dtype=bool)

            sl = slice(s, e)
            gold_arr[sl]      = gold.cpu().numpy()
            base_arr[sl]      = base_ids.cpu().numpy()
            bw_arr[sl]        = bw_b
            pg_arr[sl]        = pg_b
            sr_arr[sl]        = sr_b
            ss_arr[sl]        = ss_b
            param_sel_tok[sl] = sel_tok_p.cpu().numpy()
            param_max_prob[sl]= best_prob.cpu().numpy()
            param_keys_arr[sl]= keys.cpu().numpy()
            param_scores_arr[sl] = scores.cpu().numpy()
            topk_ids_np[sl]   = topk.cpu().numpy()
            topk_lgts_np[sl]  = lgt.cpu().numpy()
            cand_ids_np[sl]   = cand_ids.cpu().numpy()
            cand_lgts_np[sl]  = cand_lgts.cpu().numpy()

            # KNN: compute memory scores for each candidate
            if knn_index is not None:
                keys_flat = keys.reshape(b * P, args.key_dim).cpu().numpy()
                mem_s, _, _ = knn_index.query(keys_flat, args.knn_k)
                knn_mem_scores[sl] = mem_s.reshape(b, P)

    model.train()

    row = {
        "gold": gold_arr, "base": base_arr, "bw": bw_arr, "pg": pg_arr,
        "sr": sr_arr, "ss": ss_arr,
        "param_sel_tok": param_sel_tok, "param_max_prob": param_max_prob,
        "param_keys": param_keys_arr, "param_scores": param_scores_arr,
        "knn_mem_scores": knn_mem_scores,
        "topk_ids": topk_ids_np, "topk_lgts": topk_lgts_np,
        "cand_ids": cand_ids_np, "cand_lgts": cand_lgts_np,
        "N": N, "P": P, "K": K,
    }
    return row


def sweep_selector(row, selector_type, thresholds, margin_delta, args):
    """
    Sweep thresholds for parametric or KNN selector.
    selector_type: 'param' or 'knn'
    Returns list of metric dicts per threshold.
    """
    N, P, K = row["N"], row["P"], row["K"]
    md      = 0.5 * margin_delta
    topk_ids  = torch.from_numpy(row["topk_ids"])
    topk_lgts = torch.from_numpy(row["topk_lgts"])
    gold_np   = row["gold"]
    base_np   = row["base"]
    bw_np     = row["bw"]
    pg_np     = row["pg"]
    sr_np     = row["sr"]
    ss_np     = row["ss"]
    cand_ids  = row["cand_ids"]

    results = []
    for thr in thresholds:
        if selector_type == "param":
            # Apply if max_prob >= thr
            # Selected = argmax score
            best_j    = row["param_scores"].argmax(1)
            max_prob  = row["param_max_prob"]
            apply_mask = max_prob >= thr
            sel_toks  = cand_ids[np.arange(N), best_j]
        else:
            # KNN: for each row pick candidate with max memory_score
            mem = row["knn_mem_scores"]                      # [N, P]
            best_j    = mem.argmax(1)
            best_mem  = mem[np.arange(N), best_j]
            apply_mask = best_mem >= thr
            sel_toks  = cand_ids[np.arange(N), best_j]

        # Surgical edit in topK
        sel_toks_t = torch.from_numpy(sel_toks.astype(np.int64))
        apply_t    = torch.from_numpy(apply_mask)
        ref_lgts   = apply_surgical(topk_lgts, topk_ids, sel_toks_t, apply_t, md)

        ref_top1_i = ref_lgts.argmax(1).numpy()
        ref_top1   = topk_ids.numpy()[np.arange(N), ref_top1_i]

        sel_is_gold = sel_toks == gold_np
        ctg = bw_np & apply_mask & (ref_top1 == gold_np)
        caw = ~bw_np & apply_mask & (ref_top1 != gold_np)

        n_bw    = int(bw_np.sum())
        n_bc    = int((~bw_np).sum())
        n_bwcov = int((bw_np & pg_np).sum())
        n_pg    = int(pg_np.sum())
        n_tgt   = int((bw_np & pg_np & (sr_np | ss_np)).sum())

        m = {
            "threshold":                thr,
            "apply_rate":               _rate(apply_mask, N),
            "selected_gold_bwcov":      _rate(sel_is_gold[bw_np & pg_np], n_bwcov),
            "selected_gold_in_pool":    _rate(sel_is_gold[pg_np], n_pg),
            "selected_gold_target":     _rate(sel_is_gold[bw_np & pg_np & (sr_np | ss_np)], n_tgt),
            "changed_to_gold_rate":     _rate(ctg, N),
            "changed_away_rate":        _rate(caw, N),
            "noharm_changed_away":      _rate(caw, n_bc),
            "top1_acc_base":            _rate(~bw_np, N),
            "top1_acc_refined":         _rate(ref_top1 == gold_np, N),
            "top1_acc_gain":            _rate(ref_top1 == gold_np, N) - _rate(~bw_np, N),
            "gold_in_pool_rate":        _rate(pg_np, N),
            "n_bwcov":                  n_bwcov,
        }
        results.append(m)

    return results


def eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                     unk_region, unk_super, sr_enabled, args, device,
                     row, param_threshold, knn_threshold=None,
                     knn_index: Optional[KNNIndex] = None):
    try:
        model.eval()
        N   = row["N"]
        P   = row["P"]
        VS  = model.token_emb_weight.shape[0]
        md  = 0.5 * args.margin_delta

        result = {}
        for selector, thr in [("param", param_threshold),
                               ("knn",  knn_threshold)]:
            if selector == "knn" and knn_index is None:
                continue
            if thr is None:
                continue

            tot_nll_b = tot_nll_r = tot_acc_b = tot_acc_r = 0.0
            BSZ = args.eval_batch_size

            with torch.no_grad():
                for s in range(0, N, BSZ):
                    e = min(s + BSZ, N)
                    b = e - s
                    ar = torch.arange(b, device=device)

                    hp   = val_data["h_prime"][s:e].to(device)
                    gold = val_data["gold"][s:e].to(device)
                    topk = val_data["topk_ids"][s:e].to(device)

                    if selector == "param":
                        best_j    = row["param_scores"][s:e].argmax(1)
                        max_prob  = row["param_max_prob"][s:e]
                        apply_mask = max_prob >= thr
                    else:
                        mem       = row["knn_mem_scores"][s:e]
                        best_j    = mem.argmax(1)
                        apply_mask = mem[np.arange(b), best_j] >= thr

                    sel_toks_np = row["cand_ids"][s:e][np.arange(b), best_j]
                    sel_toks    = torch.from_numpy(sel_toks_np).to(device)
                    base_toks   = topk[:, 0]
                    app_t       = torch.from_numpy(apply_mask.astype(np.float32)).to(device)

                    fv_base = hp @ model.token_emb_weight.T       # [b, VS]
                    fv_ref  = fv_base.clone()
                    fv_ref[ar, sel_toks.clamp(0, VS-1)]  += md * app_t
                    fv_ref[ar, base_toks.clamp(0, VS-1)] -= md * app_t

                    gs_v = gold.clamp(0, VS-1)
                    tot_nll_b += F.cross_entropy(fv_base, gs_v, reduction="sum").item()
                    tot_nll_r += F.cross_entropy(fv_ref,  gs_v, reduction="sum").item()
                    tot_acc_b += (fv_base.argmax(1) == gs_v).sum().item()
                    tot_acc_r += (fv_ref.argmax(1)  == gs_v).sum().item()

            result[selector] = {
                "threshold":               thr,
                "full_vocab_base_nll":     tot_nll_b / N,
                "full_vocab_refined_nll":  tot_nll_r / N,
                "full_vocab_gain":         (tot_nll_b - tot_nll_r) / N,
                "full_vocab_top1_acc_base":    tot_acc_b / N,
                "full_vocab_top1_acc_refined": tot_acc_r / N,
            }

        model.train()
        return result
    except Exception as exc:
        model.train()
        print(f"[warn] full_vocab eval failed: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def compute_baselines(row, margin_delta):
    N   = row["N"]
    md  = 0.5 * margin_delta
    bw  = row["bw"]; pg = row["pg"]
    sr  = row["sr"]; ss = row["ss"]
    gold_np  = row["gold"]; base_np = row["base"]
    cand_ids = row["cand_ids"]
    topk_ids  = torch.from_numpy(row["topk_ids"])
    topk_lgts = torch.from_numpy(row["topk_lgts"])

    n_bw   = int(bw.sum())
    n_bc   = int((~bw).sum())
    n_bwcov = int((bw & pg).sum())

    # Base no-op
    base_acc = _rate(~bw, N)

    # Rank2-always (select cand_ids[:, 0] always)
    r2_toks = cand_ids[:, 0]
    r2_apply = np.ones(N, dtype=bool)
    r2_ref = apply_surgical(topk_lgts, topk_ids,
                             torch.from_numpy(r2_toks.astype(np.int64)),
                             torch.from_numpy(r2_apply), md)
    r2_top1 = topk_ids.numpy()[np.arange(N), r2_ref.argmax(1).numpy()]
    r2_sg   = r2_toks == gold_np

    # Oracle-pool: select gold if in pool & base_wrong, else NO_OP
    oracle_ctg = oracle_caw = 0
    for i in range(N):
        if bw[i] and pg[i]:
            for j in range(cand_ids.shape[1]):
                if cand_ids[i, j] == gold_np[i]:
                    ref_lgt = topk_lgts[i].clone()
                    sel_pos = (topk_ids[i] == int(cand_ids[i, j])).long().argmax().item()
                    if topk_ids[i, sel_pos] == int(cand_ids[i, j]):
                        ref_lgt[sel_pos] += md
                        ref_lgt[0]       -= md
                    if topk_ids[i, ref_lgt.argmax()].item() == gold_np[i]:
                        oracle_ctg += 1
                    break

    return {
        "base_top1_acc":              base_acc,
        "gold_in_pool_rate_val":      _rate(pg, N),
        "baseline_noop_ctg":          0.0,
        "baseline_noop_caw":          0.0,
        "baseline_rank2_sg_bwcov":    _rate(r2_sg[bw & pg], n_bwcov),
        "baseline_rank2_ctg":         _rate(r2_top1[bw] == gold_np[bw], n_bw),
        "baseline_rank2_caw":         _rate(r2_top1[~bw] != gold_np[~bw], n_bc),
        "baseline_oracle_pool_ctg":   _rate(oracle_ctg, n_bwcov) if n_bwcov > 0 else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Examples
# ─────────────────────────────────────────────────────────────────────────────

def write_example_reports(val_data, row, tok_arr_t, unk_region, tokenizer,
                           args, run_dir, param_thr, knn_thr,
                           knn_index: Optional[KNNIndex] = None):
    rng = np.random.default_rng(42)
    N   = row["N"]; P = row["P"]
    bw  = row["bw"]; pg = row["pg"]
    gold_np = row["gold"]; base_np = row["base"]
    cand_ids = row["cand_ids"]
    vs  = tok_arr_t.shape[0]

    def reg(tid): return str(int(tok_arr_t[min(int(tid), vs-1)].item()))
    def d(tid):
        try: return f"`{tokenizer.decode([int(tid)])}`"
        except: return str(tid)

    # Determine per-row selections
    best_j_param = row["param_scores"].argmax(1)
    max_prob     = row["param_max_prob"]
    param_sel    = cand_ids[np.arange(N), best_j_param]
    param_apply  = max_prob >= param_thr

    if knn_index is not None and knn_thr is not None:
        mem       = row["knn_mem_scores"]
        best_j_knn = mem.argmax(1)
        best_mem   = mem[np.arange(N), best_j_knn]
        knn_sel    = cand_ids[np.arange(N), best_j_knn]
        knn_apply  = best_mem >= knn_thr
    else:
        knn_sel = knn_apply = best_j_knn = None

    param_sel_gold = param_apply & (param_sel == gold_np)
    knn_sel_gold   = (knn_apply & (knn_sel == gold_np)) if knn_sel is not None else np.zeros(N, bool)

    # topK refined
    topk_ids_t  = torch.from_numpy(row["topk_ids"])
    topk_lgts_t = torch.from_numpy(row["topk_lgts"])

    def fmt_row(ri):
        lines = [f"### Row {ri}\n"]
        if val_data.get("ids") is not None:
            try:
                ctx = tokenizer.decode(val_data["ids"][ri][-64:].tolist(),
                                       skip_special_tokens=False)
                lines.append(f"**Context:** `{ctx}`\n")
            except Exception: pass
        gt = int(gold_np[ri]); bt = int(base_np[ri])
        lines.append(f"**Gold:** {d(gt)} id={gt} region={reg(gt)}")
        lines.append(f"**Base top-1:** {d(bt)} id={bt} region={reg(bt)}\n")

        lines.append("| j | Token | ID | Region | Base rank | Base lgt | Param score | KNN mem | is_gold | sel_param | sel_knn |")
        lines.append("|---|-------|----|--------|-----------|----------|-------------|---------|---------|-----------|---------|")
        for j in range(min(P, 12)):
            cid = int(row["cand_ids"][ri, j])
            try: ts = tokenizer.decode([cid])
            except: ts = str(cid)
            rnk = int(row["topk_ids"][ri].tolist().index(cid)) + 1 if cid in row["topk_ids"][ri].tolist() else "?"
            cl  = float(row["cand_lgts"][ri, j])
            ps  = float(row["param_scores"][ri, j])
            km  = float(row["knn_mem_scores"][ri, j]) if knn_index is not None else float("nan")
            ig  = "✓" if cid == gt else ""
            sp  = "P" if j == best_j_param[ri] and param_apply[ri] else ""
            sk  = "K" if (knn_sel is not None and j == best_j_knn[ri] and knn_apply[ri]) else ""
            lines.append(f"| {j} | `{ts}` | {cid} | {reg(cid)} | {rnk} | {cl:.3f} | {ps:.3f} | {km:.3f} | {ig} | {sp} | {sk} |")

        lines.append("")
        return "\n".join(lines) + "\n"

    buckets = {
        "param_selected_gold": np.where(param_sel_gold & bw)[0],
        "knn_selected_gold":   np.where(knn_sel_gold & bw)[0] if knn_sel is not None else np.array([]),
        "selected_wrong":      np.where(param_apply & ~(param_sel == gold_np) & bw)[0],
        "changed_away":        np.where(param_apply & ~bw)[0],
    }
    for bname, indices in buckets.items():
        if len(indices) > args.num_examples:
            indices = rng.choice(indices, args.num_examples, replace=False)
        path = os.path.join(run_dir, f"examples_{bname}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {bname.replace('_',' ').title()}\n\n_{len(indices)} examples_\n\n---\n\n")
            for ri in indices:
                f.write(fmt_row(int(ri)))
                f.write("---\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(run_dir, args, baselines, best_param, best_knn, fv_results, train_stats):
    def f(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return "nan"
        if isinstance(v, float): return f"{v:.4f}"
        return str(v) if v is not None else "nan"

    bl_r2sg = baselines.get("baseline_rank2_sg_bwcov", float("nan"))
    bp_sg   = best_param.get("selected_gold_bwcov", float("nan")) if best_param else float("nan")
    bk_sg   = best_knn.get("selected_gold_bwcov",   float("nan")) if best_knn else float("nan")
    bp_fvg  = fv_results.get("param", {}).get("full_vocab_gain", float("nan"))
    bk_fvg  = fv_results.get("knn",   {}).get("full_vocab_gain", float("nan"))

    lines = ["# Learned Pair-Key KNN Selector V1\n"]
    lines.append(f"**Run:** `{args.run_name}`  |  **Steps:** {args.steps}  "
                 f"|  **candidate_filter:** `{args.candidate_filter}`\n")

    lines.append("## Reference\n")
    lines.append("| Experiment | sel_gold_bwcov | fv_gain |")
    lines.append("|-----------|----------------|---------|")
    lines.append("| Old KNN V1 (random proj) | 0.0070 | -0.0116 |")
    lines.append("| Old KNN V1 rank2 | 0.1452 | — |")
    lines.append("| V3 oracle | 0.2407 target_ctg | +0.1584 |\n")

    lines.append("## Train Stats\n")
    for k, v in train_stats.items():
        lines.append(f"  {k:40s} = {f(v)}")
    lines.append("")

    lines.append("## Baselines\n")
    for k, v in baselines.items():
        lines.append(f"  {k:40s} = {f(v)}")
    lines.append("")

    if best_param:
        lines.append("## Best Parametric Selector\n")
        for k, v in best_param.items():
            lines.append(f"  {k:40s} = {f(v)}")
        lines.append("")

    if best_knn:
        lines.append("## Best KNN Selector\n")
        for k, v in best_knn.items():
            lines.append(f"  {k:40s} = {f(v)}")
        lines.append("")

    if fv_results:
        lines.append("## Full-Vocab Results\n")
        for sel, fvd in fv_results.items():
            lines.append(f"### {sel}\n")
            for k, v in fvd.items():
                lines.append(f"  {k:40s} = {f(v)}")
            lines.append("")

    lines.append("## Training Objective Note\n")
    lines.append("> PairKeyNet V1 is trained primarily through candidate CE via score_head. "
                 "The KNN evaluation uses the learned normalized key space, but the key geometry "
                 "is not yet trained with a dedicated supervised contrastive/InfoNCE retrieval "
                 "objective. Therefore, if parametric selector improves but KNN does not, this "
                 "means the frozen features contain selection signal but the retrieval geometry "
                 "is still insufficient — not that KNN/memory is impossible.\n")

    lines.append("## Analysis\n")
    def yn(c, yes="✅", no="⚠️"): return yes if c else no

    q1 = not math.isnan(bp_sg) and not math.isnan(bl_r2sg) and bp_sg > bl_r2sg
    lines.append(f"**1. Parametric selector beats rank2 (sg_bwcov={f(bp_sg)} vs {f(bl_r2sg)})?** {yn(q1, 'Yes', 'No')}")

    q2 = not math.isnan(bk_sg) and bk_sg > 0.007 * 3
    lines.append(f"**2. Learned-key KNN beats old KNN (sg_bwcov={f(bk_sg)} vs 0.0070)?** {yn(q2, 'Yes — key geometry improved', 'No — geometry still bad')}")

    q3 = not math.isnan(bk_sg) and not math.isnan(bl_r2sg) and bk_sg > bl_r2sg
    lines.append(f"**3. Learned-key KNN beats rank2 ({f(bk_sg)} vs {f(bl_r2sg)})?** {yn(q3, 'Yes', 'No')}")

    q4 = (not math.isnan(bp_fvg) and bp_fvg >= 0) or (not math.isnan(bk_fvg) and bk_fvg >= 0)
    lines.append(f"**4. Any threshold gives fv_gain >= 0 (param={f(bp_fvg)}, knn={f(bk_fvg)})?** {yn(q4, 'Yes', 'No')}")

    lines.append(f"**5. KNN failure diagnosis:**")
    if q1 and not q3:
        lines.append("  → Parametric works but KNN does not: signal exists but key geometry is still bad for retrieval. Try larger key_dim, more training steps, or contrastive objective.")
    elif not q1 and not q3:
        lines.append("  → Neither beats rank2: frozen h_prime + token features are insufficient. Need detail memory, raw context, or recurrent architecture.")
    elif q1 and q3:
        lines.append("  → Both selectors beat rank2: learned pair-key memory is viable. Optimize threshold policy and scale.")
    else:
        lines.append("  → KNN beats rank2 but parametric does not: key space organized well but score head underfit. Check score_head capacity.")

    lines.append("\n**6. Recommended next step:**")
    if not q1 and not q3:
        lines.append("  → Features insufficient. Next: detail-preserving candidate cross-attention over recent context, or Transformer-XL-style recurrence over h_prime states.")
    elif q1 and not q3:
        lines.append("  → Better contrastive key training (e.g., InfoNCE across rows), larger datastore, or approximate-search improvements.")
    elif q4:
        lines.append("  → Learned pair-key works. Next: joint training with NO_OP policy, larger candidate pool, or scale to more rows.")
    else:
        lines.append("  → Positive signal but negative FV gain: add NO_OP gating to protect base-correct rows.")
    lines.append("")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    run_dir = os.path.join(args.output_root, args.run_name)
    if os.path.exists(run_dir):
        raise RuntimeError(f"Run dir exists: {run_dir}. Use --run_name.")
    os.makedirs(run_dir)

    print("\n[backbone] Loading token embeddings...")
    tok_w, d_model, vocab_size = load_backbone(args.small_ckpt, device)
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    print("\n[maps] Loading region/super maps...")
    t2r, r2s, unk_region, unk_super, sr_enabled = load_maps(
        args.token_to_region, args.super_map)

    tok_arr_np = build_tok_arr(t2r, unk_region, vocab_size)
    reg_arr_np = build_reg_arr(r2s, unk_super, unk_region)
    tok_arr_t  = torch.from_numpy(tok_arr_np).long().to(device)
    reg_arr_t  = torch.from_numpy(reg_arr_np).long().to(device)

    print("\n[data] Loading shards...")
    train_data = load_shards(args.train_dir, args.top_k, "train",
                             args.max_train_rows, load_ids=False)
    val_data   = load_shards(args.val_dir,   args.top_k, "val",
                             None, load_ids=True)

    print("\n[model] Building PairKeyNet...")
    n_regions = unk_region; n_supers = unk_super if sr_enabled else 1
    model = PairKeyNet(
        token_emb_weight=tok_w.to(device),
        tok_arr=tok_arr_np, reg_arr=reg_arr_np,
        d_model=d_model, n_regions=n_regions, n_supers=n_supers,
        sr_enabled=sr_enabled, unk_region=unk_region, unk_super=unk_super,
        top_k=args.top_k, key_dim=args.key_dim,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        region_emb_dim=args.region_emb_dim, super_emb_dim=args.super_emb_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    print("\n[train rows] Finding trainable rows...")
    train_row_idx, gold_pool_idx = find_trainable_rows(
        train_data, tok_arr_t, reg_arr_t, unk_region, unk_super,
        sr_enabled, args, device, max_rows=args.max_pair_train_rows)

    N_train = len(train_row_idx)
    if N_train == 0:
        raise RuntimeError("No trainable rows found.")

    train_stats = {
        "total_train_rows":      train_data["h_prime"].shape[0],
        "trainable_bwcov_rows":  N_train,
        "max_pair_train_rows":   args.max_pair_train_rows,
        "negatives_per_positive": args.negatives_per_positive,
        "candidate_pool_size":   args.candidate_pool_size,
        "candidate_filter":      args.candidate_filter,
    }

    config = vars(args).copy()
    config.update({"d_model": d_model, "vocab_size": vocab_size,
                   "sr_enabled": sr_enabled, "n_params": n_params,
                   "train_stats": train_stats})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None

    perm = np.random.permutation(N_train)
    pos  = 0
    BSZ  = args.batch_size
    P    = args.candidate_pool_size

    train_log = os.path.join(run_dir, "train_log.csv")
    eval_log  = os.path.join(run_dir, "eval_log.csv")
    hdr_t = hdr_e = False

    param_thresholds = [float(x) for x in args.param_thresholds.split(",")]
    knn_thresholds   = [float(x) for x in args.thresholds.split(",")]

    best_score = -float("inf"); best_step = 0
    step = 0; t0 = time.time()
    optimizer.zero_grad()

    print(f"\n[train] steps={args.steps}  batch={BSZ}  lr={args.lr}\n")

    while step < args.steps:
        if pos + BSZ > N_train:
            perm = np.random.permutation(N_train); pos = 0
        ri   = train_row_idx[perm[pos:pos+BSZ]]; pos += BSZ

        hp   = train_data["h_prime"][ri].to(device)
        topk = train_data["topk_ids"][ri].to(device)
        lgt  = train_data["topk_lgt"][ri].to(device)
        gold = train_data["gold"][ri].to(device)

        cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
            topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
            P, args.candidate_filter, device)

        # Recompute gold idx in fresh pool (filter may vary per batch)
        gold_in_pool = (cand_ids == gold.unsqueeze(1))
        pool_has_gold = gold_in_pool.any(1)
        gold_idx_b = gold_in_pool.long().argmax(1)
        gold_idx_b[~pool_has_gold] = -1

        base_ids = topk[:, 0]

        with _amp_ctx(args.amp):
            _, scores = model.forward_pool(hp, base_ids, lgt,
                                            cand_ids, cand_lgts, cand_rnks,
                                            cand_regs, cand_sups)
            total_loss, ld = compute_loss(scores, gold_idx_b,
                                           args.lambda_margin, args.target_margin, device)
            loss_sc = total_loss / args.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss_sc).backward()
        else:
            loss_sc.backward()

        step_accum = getattr(main, "_accum", 0) + 1
        main._accum = step_accum
        if step_accum < args.grad_accum_steps:
            continue
        main._accum = 0

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 50 == 0 or step == 1:
            print(f"  step={step:5d}  loss={ld['total']:.4f}  "
                  f"ce={ld['ce']:.4f}  margin={ld['margin']:.4f}  "
                  f"n={ld['n']}  t={time.time()-t0:.0f}s")

        _write_csv_row(train_log, {"step": step, **ld}, header=not hdr_t); hdr_t = True

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n[eval] step={step}")
            val_row = evaluate_val(model, val_data, tok_arr_t, reg_arr_t,
                                    unk_region, unk_super, sr_enabled, args, device)

            param_sweep = sweep_selector(val_row, "param", param_thresholds,
                                          args.margin_delta, args)
            # Pick best param threshold by selected_gold_bwcov
            best_pm = max(param_sweep, key=lambda x: x.get("selected_gold_bwcov", -1),
                          default=param_sweep[0] if param_sweep else {})

            sa = best_pm
            score = sa.get("selected_gold_bwcov", float("nan"))
            print(f"  [param best thr={sa.get('threshold')}]  "
                  f"sg_bwcov={sa.get('selected_gold_bwcov',float('nan')):.4f}  "
                  f"ctg={sa.get('changed_to_gold_rate',float('nan')):.4f}  "
                  f"caw={sa.get('changed_away_rate',float('nan')):.4f}")

            torch.save({"step": step, "model": model.state_dict(),
                        "args": vars(args)},
                       os.path.join(run_dir, "latest_pairkey.pt"))

            if not math.isnan(score) and score > best_score:
                best_score = score; best_step = step
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)},
                           os.path.join(run_dir, "best_pairkey.pt"))
                print(f"  [best] step={step}  score={score:.4f}")

            ev = {"step": step, **best_pm}
            _write_csv_row(eval_log, ev, header=not hdr_e); hdr_e = True
            print()

    # ── Post-training ─────────────────────────────────────────────────────────
    best_path = os.path.join(run_dir, "best_pairkey.pt")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        loaded_best_step = int(ck.get("step", -1))
        print(f"[final] Loaded best_pairkey.pt for final datastore/eval  step={loaded_best_step}")
    else:
        loaded_best_step = -1
        print("[final] WARNING: best_pairkey.pt not found; using latest model for final datastore/eval")

    print("[final] Building KNN datastore from train rows...")
    keys_np, labels_np, meta = build_datastore(
        model, train_data, tok_arr_t, reg_arr_t, unk_region, unk_super,
        sr_enabled, train_row_idx, gold_pool_idx, args, device)

    torch.save(torch.from_numpy(keys_np),   os.path.join(run_dir, "pair_keys_train.pt"))
    torch.save(torch.from_numpy(labels_np), os.path.join(run_dir, "pair_labels_train.pt"))
    with open(os.path.join(run_dir, "pair_meta_train.pt"), "wb") as f:
        torch.save(meta[:10000], f)   # save subset for inspection

    print("[final] Building KNN index...")
    knn_idx = KNNIndex(keys_np, labels_np, knn_tau=args.knn_tau, backend="auto")

    print("[final] Full val evaluation (param + KNN)...")
    val_row = evaluate_val(model, val_data, tok_arr_t, reg_arr_t,
                            unk_region, unk_super, sr_enabled, args, device,
                            knn_index=knn_idx)

    baselines = compute_baselines(val_row, args.margin_delta)
    with open(os.path.join(run_dir, "baseline_metrics.json"), "w") as f:
        json.dump(_json_safe(baselines), f, indent=2)

    param_sweep = sweep_selector(val_row, "param", param_thresholds, args.margin_delta, args)
    knn_sweep   = sweep_selector(val_row, "knn",   knn_thresholds,   args.margin_delta, args)

    # Write CSV sweeps
    param_csv = os.path.join(run_dir, "param_threshold_sweep.csv")
    knn_csv   = os.path.join(run_dir, "knn_threshold_sweep.csv")
    for sweep, path in [(param_sweep, param_csv), (knn_sweep, knn_csv)]:
        hdr = True
        for row_d in sweep:
            _write_csv_row(path, row_d, header=hdr); hdr = False

    best_param = max(param_sweep, key=lambda x: x.get("selected_gold_bwcov", -1),
                     default=None)
    best_knn   = max(knn_sweep,   key=lambda x: x.get("selected_gold_bwcov", -1),
                     default=None)

    # Full-vocab at best thresholds
    fv_results = {}
    if args.eval_full_vocab:
        print("[final] Full-vocab eval...")
        pt  = best_param.get("threshold") if best_param else None
        kt  = best_knn.get("threshold")   if best_knn else None
        fv_results = eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                                      unk_region, unk_super, sr_enabled, args, device,
                                      val_row, pt, kt, knn_idx)
        # Update best with fv
        if best_param and "param" in fv_results:
            best_param.update(fv_results["param"])
        if best_knn and "knn" in fv_results:
            best_knn.update(fv_results["knn"])

    final_metrics = {
        "step": args.steps,
        "best_step_during_training":          best_step,
        "loaded_best_step_for_final_eval":    loaded_best_step,
        "final_eval_uses_best_checkpoint":    loaded_best_step >= 0,
        "baselines": baselines,
        "best_param": best_param,
        "best_knn":   best_knn,
        "fv": fv_results,
        "train_stats": train_stats,
    }
    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump(_json_safe(final_metrics), f, indent=2)
    with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
        json.dump(_json_safe({"best_step": best_step, "best_param": best_param,
                               "best_knn": best_knn}), f, indent=2)

    # Examples
    try:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            tokenizer = type("FT", (), {"decode": lambda s, ids, **kw: str(ids)})()
        pt  = best_param.get("threshold") if best_param else 0.5
        kt  = best_knn.get("threshold")   if best_knn else 0.5
        write_example_reports(val_data, val_row, tok_arr_t.cpu(), unk_region,
                               tokenizer, args, run_dir, pt, kt, knn_idx)
    except Exception as exc:
        print(f"[warn] example reports failed: {exc}")

    train_stats["final_eval_checkpoint_step"] = loaded_best_step
    write_report(run_dir, args, baselines, best_param, best_knn, fv_results, train_stats)

    print(f"\n{'='*60}")
    print(f" Learned Pair-Key KNN Selector V1 complete.")
    print(f" Run dir: {run_dir}")
    print(f" Best step: {best_step}  score={best_score:.4f}")
    print(f"{'='*60}\n")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",           required=True)
    p.add_argument("--train_dir",            required=True)
    p.add_argument("--val_dir",              required=True)
    p.add_argument("--token_to_region",      required=True)
    p.add_argument("--super_map",            default=None)
    p.add_argument("--output_root",          default="runs/learned_pairkey_knn_v1")
    p.add_argument("--run_name",             default="learned_pairkey_hprime_v1")
    p.add_argument("--top_k",               type=int,   default=256)
    p.add_argument("--candidate_pool_size",  type=int,   default=32)
    p.add_argument("--candidate_filter",     default="top_rank",
                   choices=["top_rank","same_region_or_superregion",
                            "same_region_only","same_superregion_only"])
    p.add_argument("--max_train_rows",       type=int,   default=500000)
    p.add_argument("--max_pair_train_rows",  type=int,   default=200000)
    p.add_argument("--negatives_per_positive", type=int, default=8)
    p.add_argument("--key_dim",              type=int,   default=128)
    p.add_argument("--hidden_dim",           type=int,   default=512)
    p.add_argument("--dropout",              type=float, default=0.1)
    p.add_argument("--region_emb_dim",       type=int,   default=64)
    p.add_argument("--super_emb_dim",        type=int,   default=32)
    p.add_argument("--batch_size",           type=int,   default=256)
    p.add_argument("--grad_accum_steps",     type=int,   default=1)
    p.add_argument("--lr",                   type=float, default=1e-4)
    p.add_argument("--steps",                type=int,   default=5000)
    p.add_argument("--eval_every",           type=int,   default=500)
    p.add_argument("--eval_batch_size",      type=int,   default=256)
    p.add_argument("--grad_clip",            type=float, default=1.0)
    p.add_argument("--lambda_margin",        type=float, default=0.5)
    p.add_argument("--target_margin",        type=float, default=1.0)
    p.add_argument("--knn_k",               type=int,   default=32)
    p.add_argument("--knn_tau",              type=float, default=0.1)
    p.add_argument("--thresholds",           default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    p.add_argument("--param_thresholds",     default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    p.add_argument("--margin_delta",         type=float, default=1.0)
    p.add_argument("--eval_full_vocab",      action="store_true")
    p.add_argument("--amp",                  action="store_true")
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument("--num_examples",         type=int,   default=40)
    return p.parse_args()


if __name__ == "__main__":
    main()
