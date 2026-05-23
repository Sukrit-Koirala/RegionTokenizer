#!/usr/bin/env python3
"""
train_local_detail_selector_v1.py — Local Detail Memory Candidate Selector

Stage 2: candidates attend over recent context tokens.
Fixes Stage 1's missing NO_OP problem by including base-correct rows in training.

Stage 1 reference: sel_gold_bwcov≈0.246, ctg≈0.040, caw≈0.062
V3 oracle reference: target_ctg=0.2407, oracle_fv_gain=+0.1584
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


def load_shards(shard_dir, top_k, split_name, max_rows=None):
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
        ids  = _get_field(sh, "input_ids", required=False)
        if ids is None:
            raise RuntimeError(
                "input_ids required for local detail memory selector. "
                f"Shard {sp} does not contain input_ids.")
        B, K = topk.shape
        if K < top_k:
            topk = torch.cat([topk, torch.zeros(B, top_k-K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, top_k-K), float("nan"))], 1)
        elif K > top_k:
            topk, lgt = topk[:, :top_k], lgt[:, :top_k]
        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            hp, topk, lgt, gold, ids = hp[:keep], topk[:keep], lgt[:keep], gold[:keep], ids[:keep]
            B = keep
        all_hp.append(hp); all_topk.append(topk)
        all_lgt.append(lgt); all_gold.append(gold)
        all_ids.append(ids)
        total += B
    data = {
        "h_prime":  torch.cat(all_hp, 0),
        "topk_ids": torch.cat(all_topk, 0),
        "topk_lgt": torch.cat(all_lgt, 0),
        "gold":     torch.cat(all_gold, 0),
        "ids":      torch.cat(all_ids, 0),
    }
    N = data["h_prime"].shape[0]
    print(f"  {split_name}: {N:,} rows  d={data['h_prime'].shape[1]}  "
          f"K={top_k}  seq_len={data['ids'].shape[1]}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pool
# ─────────────────────────────────────────────────────────────────────────────

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
    cand_sups = torch.full((N, pool_size), unk_super,  dtype=torch.long, device=device) \
                if sr_enabled else None

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
                    mask = (cs == base_sups[i]) & (cs != unk_super)
                else:
                    mask = torch.zeros(len(c_toks), dtype=torch.bool, device=device)
            elif candidate_filter == "same_region_or_superregion":
                same_r = (cr == br) & (cr != unk_region)
                if sr_enabled:
                    cs = reg_arr_t[cr.clamp(0, rlen)]
                    same_s = (cs == base_sups[i]) & (cs != unk_super)
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


def assign_labels(gold, topk_ids, cand_ids, uncovered_policy):
    """Returns labels [N]: -1=ignore, 0=NO_OP, 1..P=candidate index."""
    N, P      = cand_ids.shape
    base_top1 = topk_ids[:, 0]
    correct   = base_top1 == gold
    in_pool   = (cand_ids == gold.unsqueeze(1))
    pool_has  = in_pool.any(1)
    pool_idx  = in_pool.long().argmax(1)

    labels = torch.full((N,), -1, dtype=torch.long, device=gold.device)
    labels[correct] = 0
    labels[~correct & pool_has] = pool_idx[~correct & pool_has] + 1
    if uncovered_policy == "noop":
        labels[~correct & ~pool_has] = 0
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LocalDetailSelector(nn.Module):
    """
    Candidate selector with cross-attention over recent input tokens.

    Action scores [B, P+1]: index 0 = NO_OP, 1..P = candidates.
    """

    def __init__(self, token_emb_weight, tok_arr, reg_arr,
                 d_model, n_regions, n_supers, sr_enabled,
                 unk_region, unk_super, top_k, pool_size,
                 memory_len=128, resolver_dim=256, hidden_dim=512,
                 attention_heads=4, dropout=0.1,
                 region_emb_dim=64, super_emb_dim=32):
        super().__init__()
        self.d_model    = d_model
        self.sr_enabled = sr_enabled
        self.unk_region = unk_region
        self.unk_super  = unk_super
        self.top_k      = top_k
        self.pool_size  = pool_size
        self.memory_len = memory_len
        self.resolver_dim = resolver_dim

        self.register_buffer("token_emb_weight", token_emb_weight.detach().float())
        self.register_buffer("tok_arr", torch.from_numpy(tok_arr.astype(np.int32)).long())
        self.register_buffer("reg_arr", torch.from_numpy(reg_arr.astype(np.int32)).long())

        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim)
        self.super_emb  = nn.Embedding(n_supers + 2, super_emb_dim) if sr_enabled else None

        # Pair feature dimension (same structure as Stage 1)
        pair_feat_dim = (5 * d_model + 7 + 2 * region_emb_dim + 1)
        if sr_enabled:
            pair_feat_dim += 2 * super_emb_dim + 1

        # Pair feature → query for attention
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, resolver_dim),
            nn.GELU(),
        )

        # Memory projection: d_model → resolver_dim (keys and values)
        self.mem_proj = nn.Linear(d_model, resolver_dim, bias=False)

        # Cross-attention: candidates query over memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=resolver_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(resolver_dim)

        # Candidate score head
        self.score_head = nn.Linear(resolver_dim, 1, bias=True)

        # NO_OP head: h_prime + emb(base) + lgt_b + gap01 + entropy
        noop_dim = 2 * d_model + 3
        self.noop_mlp = nn.Sequential(
            nn.Linear(noop_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, resolver_dim),
            nn.GELU(),
        )
        self.noop_head = nn.Linear(resolver_dim, 1, bias=True)

    def _pair_feat(self, hp, ec, eb, lgt_c, lgt_b, rank_norm, reg_c, reg_b,
                   sup_c=None, sup_b=None):
        logit_gap = lgt_c - lgt_b
        hp_dot_c  = (hp * ec).sum(-1, keepdim=True)
        hp_dot_b  = (hp * eb).sum(-1, keepdim=True)
        hp_dot_gap = hp_dot_c - hp_dot_b
        same_reg   = ((reg_c == reg_b) & (reg_c != self.unk_region)).float().unsqueeze(-1)
        remb_c = self.region_emb(reg_c.clamp(0, self.tok_arr.shape[0]-1))
        remb_b = self.region_emb(reg_b.clamp(0, self.tok_arr.shape[0]-1))
        parts = [hp, ec, eb, ec - eb, ec * eb,
                 lgt_c.unsqueeze(-1), lgt_b.unsqueeze(-1),
                 logit_gap.unsqueeze(-1), rank_norm.unsqueeze(-1),
                 hp_dot_c, hp_dot_b, hp_dot_gap,
                 remb_c, remb_b, same_reg]
        if self.sr_enabled and self.super_emb is not None and sup_c is not None:
            rlen = self.reg_arr.shape[0] - 1
            semb_c = self.super_emb(sup_c.clamp(0, rlen))
            semb_b = self.super_emb(sup_b.clamp(0, rlen))
            same_sup = ((sup_c == sup_b) & (sup_c != self.unk_super)).float().unsqueeze(-1)
            parts.extend([semb_c, semb_b, same_sup])
        return torch.cat(parts, dim=-1)

    def forward(self, h_prime, input_ids, base_lgts,
                cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups=None):
        """
        h_prime   [B, d]
        input_ids [B, T]
        base_lgts [B, K]
        cand_ids  [B, P]
        Returns action_scores [B, P+1]
        """
        B, P  = cand_ids.shape
        K     = base_lgts.shape[1]
        d     = self.d_model
        vs    = self.token_emb_weight.shape[0]

        base_ids  = input_ids.new_zeros(B)  # placeholder — read from topk arg below
        # base_top1 is cand at rank 0 of topK, passed separately via base_lgts[:,0]
        # We reconstruct base_top1 token id from topk_ids which is not passed directly.
        # Instead we rely on the fact the caller passes topK logits; we need the base token.
        # For clarity: base_top1_id is NOT passed here; caller passes cand_ids which starts at rank2.
        # We need emb(base_top1). Caller MUST ensure this is passed.
        # → Redesign: accept explicit base_top1_ids argument.
        raise NotImplementedError("Use forward_full instead")

    def forward_full(self, h_prime, input_ids, base_top1_ids, base_lgts,
                     cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups=None):
        """
        h_prime       [B, d]
        input_ids     [B, T]
        base_top1_ids [B]     token ids of topK[0]
        base_lgts     [B, K]
        cand_ids      [B, P]
        Returns action_scores [B, P+1]
        """
        B, P  = cand_ids.shape
        K     = base_lgts.shape[1]
        d     = self.d_model
        vs    = self.token_emb_weight.shape[0]
        rlen  = self.reg_arr.shape[0] - 1

        base_emb  = self.token_emb_weight[base_top1_ids.clamp(0, vs-1)]  # [B, d]
        base_reg  = self.tok_arr[base_top1_ids.clamp(0, self.tok_arr.shape[0]-1)]  # [B]
        base_lgt  = base_lgts[:, 0]                                        # [B]
        base_sup  = self.reg_arr[base_reg.clamp(0, rlen)] if self.sr_enabled else None

        cand_emb  = self.token_emb_weight[
            cand_ids.reshape(-1).clamp(0, vs-1)].view(B, P, d)             # [B, P, d]

        # ── Memory from recent input tokens ──────────────────────────────────
        mem_ids   = input_ids[:, -self.memory_len:]                        # [B, M]
        mem_emb   = self.token_emb_weight[mem_ids.clamp(0, vs-1)]          # [B, M, d]
        mem_kv    = self.mem_proj(mem_emb)                                  # [B, M, R]

        # ── Build pair features for all P candidates ──────────────────────────
        hp_exp    = h_prime.unsqueeze(1).expand(B, P, d)
        be_exp    = base_emb.unsqueeze(1).expand(B, P, d)
        bl_exp    = base_lgt.unsqueeze(1).expand(B, P)
        br_exp    = base_reg.unsqueeze(1).expand(B, P)
        bs_exp    = base_sup.unsqueeze(1).expand(B, P) if base_sup is not None else None
        rn        = cand_rnks.float() / max(K, 1)

        pair_feat = self._pair_feat(
            hp_exp.reshape(B*P, d),  cand_emb.reshape(B*P, d),
            be_exp.reshape(B*P, d),  cand_lgts.reshape(B*P),
            bl_exp.reshape(B*P),     rn.reshape(B*P),
            cand_regs.reshape(B*P),  br_exp.reshape(B*P),
            cand_sups.reshape(B*P) if (self.sr_enabled and cand_sups is not None) else None,
            bs_exp.reshape(B*P)    if bs_exp is not None else None,
        )                                                                   # [B*P, feat_dim]
        pair_hid  = self.pair_mlp(pair_feat).view(B, P, self.resolver_dim) # [B, P, R]

        # ── Cross-attention: candidates attend over memory ────────────────────
        # Reshape: treat B*P candidates as queries, but memory is per-row.
        # Use block-diagonal approach via reshape.
        # Query: [B, P, R] → [B*P, 1, R] (each candidate queries its row's memory)
        # Memory: [B, M, R] → repeat for each candidate → [B*P, M, R]
        q_flat   = pair_hid.reshape(B * P, 1, self.resolver_dim)
        kv_flat  = mem_kv.unsqueeze(1).expand(B, P, mem_kv.shape[1], self.resolver_dim) \
                         .reshape(B * P, mem_kv.shape[1], self.resolver_dim)

        attn_out, attn_w = self.cross_attn(q_flat, kv_flat, kv_flat,
                                            need_weights=True, average_attn_weights=True)
        # attn_out [B*P, 1, R], attn_w [B*P, 1, M]
        evidence  = attn_out.squeeze(1).view(B, P, self.resolver_dim)      # [B, P, R]
        cand_repr = self.post_attn_norm(pair_hid + evidence)               # [B, P, R]
        cand_scores = self.score_head(cand_repr).squeeze(-1)               # [B, P]

        # ── NO_OP score ───────────────────────────────────────────────────────
        lgt1      = base_lgts[:, 1].nan_to_num(0.0) if K > 1 else base_lgt
        gap01     = base_lgt - lgt1
        valid_lgt = base_lgts.nan_to_num(-1e9)
        topk_ent  = -(valid_lgt.softmax(-1) * valid_lgt.log_softmax(-1)).sum(-1)

        noop_feat = torch.cat([h_prime, base_emb,
                                base_lgt.unsqueeze(1), gap01.unsqueeze(1),
                                topk_ent.unsqueeze(1)], dim=1)             # [B, noop_dim]
        noop_hid  = self.noop_mlp(noop_feat)
        noop_score = self.noop_head(noop_hid)                              # [B, 1]

        action_scores = torch.cat([noop_score, cand_scores], dim=1)        # [B, P+1]

        # Return attention weights for examples (last batch's weights reshaped)
        attn_w_out = attn_w.squeeze(1).view(B, P, mem_kv.shape[1])        # [B, P, M]
        return action_scores, attn_w_out


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(action_scores, labels, noop_weight, cand_weight,
                 lambda_margin, target_margin, device):
    N, P1 = action_scores.shape
    valid  = labels >= 0
    if not valid.any():
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, {"total": 0.0, "ce": 0.0, "margin": 0.0, "n": 0}

    sc_v = action_scores[valid]
    lb_v = labels[valid]
    w = torch.where(lb_v == 0,
                    torch.full_like(lb_v, noop_weight, dtype=torch.float),
                    torch.full_like(lb_v, cand_weight, dtype=torch.float))
    L_ce = (F.cross_entropy(sc_v, lb_v, reduction="none") * w).mean()

    L_margin = torch.tensor(0.0, device=device)
    if lambda_margin > 0:
        cand_rows = valid & (labels > 0)
        if cand_rows.any():
            ar = torch.arange(N, device=device)
            sc_c = action_scores[cand_rows]   # [n_c, P+1]
            lb_c = labels[cand_rows]
            ar2  = torch.arange(sc_c.shape[0], device=device)
            s_gold = sc_c[ar2, lb_c]
            mask   = torch.ones_like(sc_c, dtype=torch.bool)
            mask[ar2, lb_c] = False
            s_wrong = sc_c.masked_fill(~mask, float("-inf")).max(1).values
            L_margin = F.softplus(target_margin - (s_gold - s_wrong)).mean()

    total = L_ce + lambda_margin * L_margin
    return total, {
        "total":  total.item(),
        "ce":     L_ce.item(),
        "margin": L_margin.item() if lambda_margin > 0 else 0.0,
        "n":      int(valid.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rate(mask, denom):
    if denom == 0: return float("nan")
    return float(np.asarray(mask, dtype=float).sum()) / denom


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


def _apply_surgical(topk_lgts, topk_ids, sel_toks, apply_mask, md_half):
    refined = topk_lgts.clone()
    if isinstance(apply_mask, np.ndarray):
        apply_mask = torch.from_numpy(apply_mask.astype(bool))
    if not apply_mask.any():
        return refined
    N   = topk_lgts.shape[0]
    ar  = torch.arange(N, device=topk_lgts.device)
    st  = torch.as_tensor(sel_toks, device=topk_lgts.device)
    pos = (topk_ids == st.unsqueeze(1)).long().argmax(1)
    ok  = (topk_ids[ar, pos] == st) & apply_mask.to(topk_lgts.device)
    if ok.any():
        refined[ar[ok], pos[ok]] += md_half
        refined[ar[ok], 0]       -= md_half
    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Build training index
# ─────────────────────────────────────────────────────────────────────────────

def build_train_index(data, tok_arr_t, reg_arr_t, unk_region, unk_super,
                       sr_enabled, args, device):
    N   = data["h_prime"].shape[0]
    BSZ = 4096
    noop_idx = []; cand_idx = []

    for s in range(0, N, BSZ):
        e    = min(s + BSZ, N)
        topk = data["topk_ids"][s:e].to(device)
        lgt  = data["topk_lgt"][s:e].to(device)
        gold = data["gold"][s:e].to(device)
        cand_ids, _, _, _, _ = build_candidate_pool(
            topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
            args.candidate_pool_size, args.candidate_filter, device)
        correct  = (topk[:, 0] == gold).cpu()
        pool_has = (cand_ids == gold.unsqueeze(1)).any(1).cpu()
        for i, gi in enumerate(range(s, e)):
            if correct[i]:
                noop_idx.append(gi)
            elif pool_has[i]:
                cand_idx.append(gi)

    print(f"  train index: noop={len(noop_idx):,}  cand={len(cand_idx):,}  "
          f"ignored={N - len(noop_idx) - len(cand_idx):,}")
    return np.array(noop_idx, dtype=np.int64), np.array(cand_idx, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, val_data, tok_arr_t, reg_arr_t,
              unk_region, unk_super, sr_enabled, args, device,
              thresholds):
    model.eval()
    N, K = val_data["h_prime"].shape[0], val_data["topk_ids"].shape[1]
    P    = args.candidate_pool_size
    BSZ  = args.eval_batch_size
    md   = 0.5 * args.margin_delta
    vs   = tok_arr_t.shape[0]

    gold_np   = np.zeros(N, dtype=np.int64)
    base_np   = np.zeros(N, dtype=np.int64)
    bw_np     = np.zeros(N, dtype=bool)
    pg_np     = np.zeros(N, dtype=bool)
    sr_np     = np.zeros(N, dtype=bool)
    ss_np     = np.zeros(N, dtype=bool)
    scores_np = np.zeros((N, P+1), dtype=np.float32)
    cand_np   = np.zeros((N, P), dtype=np.int64)
    cand_l_np = np.zeros((N, P), dtype=np.float32)
    topk_ids_np  = np.zeros((N, K), dtype=np.int64)
    topk_lgts_np = np.zeros((N, K), dtype=np.float32)
    attn_np   = np.zeros((N, P, args.memory_len), dtype=np.float32)
    mem_ids_np = np.zeros((N, args.memory_len), dtype=np.int64)
    total_ce_b = total_ce_r = 0.0

    with torch.no_grad():
        for s in range(0, N, BSZ):
            e = min(s + BSZ, N)
            b = e - s
            ar = torch.arange(b, device=device)

            hp   = val_data["h_prime"][s:e].to(device)
            topk = val_data["topk_ids"][s:e].to(device)
            lgt  = val_data["topk_lgt"][s:e].to(device)
            gold = val_data["gold"][s:e].to(device)
            iids = val_data["ids"][s:e].to(device)

            cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
                topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
                P, args.candidate_filter, device)

            base_top1_ids = topk[:, 0]
            act_sc, attn_w = model.forward_full(
                hp, iids, base_top1_ids, lgt,
                cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups)

            bw_b = (base_top1_ids != gold).cpu().numpy()
            pg_b = (cand_ids == gold.unsqueeze(1)).any(1).cpu().numpy()

            gold_reg = tok_arr_t[gold.clamp(0, vs-1)]
            top1_reg = tok_arr_t[base_top1_ids.clamp(0, vs-1)]
            sr_b = ((gold_reg == top1_reg) & (gold_reg != unk_region)).cpu().numpy()
            if sr_enabled:
                rlen = reg_arr_t.shape[0] - 1
                gs_sup = reg_arr_t[gold_reg.clamp(0, rlen)]
                t1_sup = reg_arr_t[top1_reg.clamp(0, rlen)]
                ss_b = ((gs_sup == t1_sup) & (gs_sup != unk_super)).cpu().numpy()
            else:
                ss_b = np.zeros(b, dtype=bool)

            sl = slice(s, e)
            gold_np[sl]  = gold.cpu().numpy()
            base_np[sl]  = base_top1_ids.cpu().numpy()
            bw_np[sl]    = bw_b
            pg_np[sl]    = pg_b
            sr_np[sl]    = sr_b
            ss_np[sl]    = ss_b
            scores_np[sl]= act_sc.cpu().numpy()
            cand_np[sl]  = cand_ids.cpu().numpy()
            cand_l_np[sl]= cand_lgts.cpu().numpy()
            topk_ids_np[sl]  = topk.cpu().numpy()
            topk_lgts_np[sl] = lgt.cpu().numpy()
            M_actual = attn_w.shape[2]
            attn_np[sl, :, :M_actual] = attn_w.cpu().numpy()
            mem_ids_np[sl] = iids[:, -args.memory_len:].cpu().numpy()

    # ── Compute action accuracy on val ────────────────────────────────────────
    lbl_np = np.full(N, -1, dtype=np.int64)
    correct_np = ~bw_np
    lbl_np[correct_np] = 0
    pg_cand = ~correct_np & pg_np
    if pg_cand.any():
        cand_ids_t  = torch.from_numpy(cand_np)
        gold_t      = torch.from_numpy(gold_np)
        in_pool = (cand_ids_t == gold_t.unsqueeze(1))
        pool_idx = in_pool.long().argmax(1).numpy()
        lbl_np[pg_cand] = pool_idx[pg_cand] + 1

    valid_lbl = lbl_np >= 0
    if valid_lbl.any():
        sc_v   = torch.from_numpy(scores_np[valid_lbl])
        lb_v   = torch.from_numpy(lbl_np[valid_lbl]).long()
        action_ce = F.cross_entropy(sc_v, lb_v).item()
        action_acc = float((sc_v.argmax(1) == lb_v).float().mean())
    else:
        action_ce = float("nan"); action_acc = float("nan")

    # ── Per-threshold metrics ─────────────────────────────────────────────────
    topk_ids_t  = torch.from_numpy(topk_ids_np)
    topk_lgts_t = torch.from_numpy(topk_lgts_np)
    sweep = []
    for thr in thresholds:
        probs    = torch.from_numpy(scores_np).softmax(1).numpy()
        action   = scores_np.argmax(1)                       # raw argmax
        if thr > 0.0:
            # Force NO_OP if best-candidate probability < threshold
            best_cand_j   = scores_np[:, 1:].argmax(1) + 1  # best cand (1-indexed)
            best_cand_prob = probs[np.arange(N), best_cand_j]
            # For rows where model picks NO_OP already: keep
            # For rows where model picks candidate but prob < thr: force NO_OP
            picks_cand   = action > 0
            force_noop   = picks_cand & (best_cand_prob < thr)
            action[force_noop] = 0

        apply_mask = action > 0
        pool_j     = (action - 1).clip(0, P-1)
        sel_toks   = cand_np[np.arange(N), pool_j]

        sel_toks_t = torch.from_numpy(sel_toks.astype(np.int64))
        apply_t    = torch.from_numpy(apply_mask)
        ref_lgts   = _apply_surgical(topk_lgts_t, topk_ids_t, sel_toks_t, apply_t, md)
        ref_top1_i = ref_lgts.argmax(1).numpy()
        ref_top1   = topk_ids_np[np.arange(N), ref_top1_i]

        sel_gold  = sel_toks == gold_np
        ctg = bw_np & apply_mask & (ref_top1 == gold_np)
        caw = ~bw_np & apply_mask & (ref_top1 != gold_np)

        n_bw    = int(bw_np.sum())
        n_bc    = int((~bw_np).sum())
        n_bwcov = int((bw_np & pg_np).sum())
        n_pg    = int(pg_np.sum())
        n_tgt   = int((bw_np & pg_np & (sr_np | ss_np)).sum())

        noop_correct = (~bw_np) & (~apply_mask)

        m = {
            "threshold":                     thr,
            "apply_rate":                    _rate(apply_mask, N),
            "noop_rate":                     _rate(~apply_mask, N),
            "noop_acc_on_base_correct":      _rate(noop_correct, n_bc),
            "false_apply_rate_on_bc":        _rate(apply_mask & ~bw_np, n_bc),
            "selected_gold_bwcov":           _rate(sel_gold[bw_np & pg_np], n_bwcov),
            "selected_gold_given_in_pool":   _rate(sel_gold[pg_np], n_pg),
            "selected_gold_target":          _rate(sel_gold[bw_np & pg_np & (sr_np|ss_np)], n_tgt),
            "changed_to_gold_rate":          _rate(ctg, N),
            "changed_away_rate":             _rate(caw, N),
            "noharm_changed_away":           _rate(caw, n_bc),
            "top1_acc_base":                 _rate(~bw_np, N),
            "top1_acc_refined":              _rate(ref_top1 == gold_np, N),
            "top1_acc_gain":                 _rate(ref_top1 == gold_np, N) - _rate(~bw_np, N),
            "gold_in_pool_rate":             _rate(pg_np, N),
            "action_ce_val":                 action_ce,
            "action_acc_val":                action_acc,
            "n_bwcov":                       n_bwcov,
        }
        sweep.append(m)

    # Score for best checkpoint
    m0 = next((m for m in sweep if m["threshold"] == 0.0), sweep[0])
    ctg_v = m0.get("changed_to_gold_rate", float("nan"))
    caw_v = m0.get("changed_away_rate", float("nan"))
    sip   = m0.get("selected_gold_given_in_pool", float("nan"))
    nac   = m0.get("noop_acc_on_base_correct", float("nan"))
    if not any(math.isnan(v) for v in [ctg_v, caw_v, sip, nac]):
        chk_score = ctg_v - 2.0 * caw_v + 0.5 * sip + 0.25 * nac
    else:
        chk_score = float("nan")

    row = {
        "gold": gold_np, "base": base_np, "bw": bw_np, "pg": pg_np,
        "sr": sr_np, "ss": ss_np, "scores": scores_np,
        "cand_ids": cand_np, "cand_lgts": cand_l_np,
        "topk_ids": topk_ids_np, "topk_lgts": topk_lgts_np,
        "attn": attn_np, "mem_ids": mem_ids_np,
        "N": N, "P": P, "K": K,
    }
    model.train()
    return sweep, chk_score, row


def eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                     unk_region, unk_super, sr_enabled, args, device, row):
    try:
        model.eval()
        N   = row["N"]; P = row["P"]
        VS  = model.token_emb_weight.shape[0]
        md  = 0.5 * args.margin_delta
        BSZ = args.eval_batch_size
        tot_nll_b = tot_nll_r = tot_acc_b = tot_acc_r = 0.0

        with torch.no_grad():
            for s in range(0, N, BSZ):
                e  = min(s + BSZ, N)
                b  = e - s
                ar = torch.arange(b, device=device)
                hp   = val_data["h_prime"][s:e].to(device)
                gold = val_data["gold"][s:e].to(device)
                topk = val_data["topk_ids"][s:e].to(device)

                sc_b  = row["scores"][s:e]
                act   = sc_b.argmax(1)
                apply_mask = act > 0
                pool_j     = (act - 1).clip(0, P-1)
                sel_toks   = row["cand_ids"][s:e][np.arange(b), pool_j]
                sel_t      = torch.from_numpy(sel_toks).to(device)
                base_t     = topk[:, 0]
                app_f      = torch.from_numpy(apply_mask.astype(np.float32)).to(device)

                fv_base = hp @ model.token_emb_weight.T    # [b, VS]
                fv_ref  = fv_base.clone()
                fv_ref[ar, sel_t.clamp(0, VS-1)]  += md * app_f
                fv_ref[ar, base_t.clamp(0, VS-1)] -= md * app_f

                gs = gold.clamp(0, VS-1)
                tot_nll_b += F.cross_entropy(fv_base, gs, reduction="sum").item()
                tot_nll_r += F.cross_entropy(fv_ref,  gs, reduction="sum").item()
                tot_acc_b += (fv_base.argmax(1) == gs).sum().item()
                tot_acc_r += (fv_ref.argmax(1)  == gs).sum().item()

        model.train()
        return {
            "full_vocab_base_nll":          tot_nll_b / N,
            "full_vocab_refined_nll":       tot_nll_r / N,
            "full_vocab_gain":              (tot_nll_b - tot_nll_r) / N,
            "full_vocab_top1_acc_base":     tot_acc_b / N,
            "full_vocab_top1_acc_refined":  tot_acc_r / N,
        }
    except Exception as exc:
        model.train()
        print(f"[warn] full_vocab eval failed: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def compute_baselines(row, margin_delta):
    N  = row["N"]; md = 0.5 * margin_delta
    bw = row["bw"]; pg = row["pg"]
    gold_np = row["gold"]; cand_np = row["cand_ids"]
    topk_ids_t  = torch.from_numpy(row["topk_ids"])
    topk_lgts_t = torch.from_numpy(row["topk_lgts"])
    n_bw = int(bw.sum()); n_bc = int((~bw).sum())
    n_bwcov = int((bw & pg).sum())

    r2_toks  = cand_np[:, 0]
    r2_apply = np.ones(N, dtype=bool)
    r2_ref   = _apply_surgical(topk_lgts_t, topk_ids_t,
                                torch.from_numpy(r2_toks.astype(np.int64)),
                                torch.from_numpy(r2_apply), md)
    r2_top1  = row["topk_ids"][np.arange(N), r2_ref.argmax(1).numpy()]
    r2_sg    = r2_toks == gold_np

    oracle_ctg = 0
    for i in range(N):
        if bw[i] and pg[i]:
            for j in range(cand_np.shape[1]):
                if cand_np[i, j] == gold_np[i]:
                    rl = topk_lgts_t[i].clone()
                    sp = (topk_ids_t[i] == int(cand_np[i, j])).long().argmax().item()
                    if topk_ids_t[i, sp] == int(cand_np[i, j]):
                        rl[sp] += md; rl[0] -= md
                    if topk_ids_t[i, rl.argmax()].item() == gold_np[i]:
                        oracle_ctg += 1
                    break

    return {
        "base_top1_acc":            _rate(~bw, N),
        "gold_in_pool_rate":        _rate(pg, N),
        "baseline_rank2_sg_bwcov":  _rate(r2_sg[bw & pg], n_bwcov),
        "baseline_rank2_ctg":       _rate(r2_top1[bw] == gold_np[bw], n_bw),
        "baseline_rank2_caw":       _rate(r2_top1[~bw] != gold_np[~bw], n_bc),
        "baseline_oracle_pool_ctg": _rate(oracle_ctg, n_bwcov) if n_bwcov > 0 else float("nan"),
        "stage1_ref_sel_gold_bwcov": 0.246,
        "stage1_ref_ctg":            0.040,
        "stage1_ref_caw":            0.062,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Example reports
# ─────────────────────────────────────────────────────────────────────────────

def write_example_reports(val_data, row, tok_arr_t, unk_region,
                           tokenizer, args, run_dir, best_thr):
    rng = np.random.default_rng(42)
    N   = row["N"]; P = row["P"]
    bw  = row["bw"]; gold_np = row["gold"]
    cand_np = row["cand_ids"]; cand_l = row["cand_lgts"]
    scores = row["scores"]
    attn   = row["attn"]; mem_ids = row["mem_ids"]
    vs     = tok_arr_t.shape[0]

    def reg(tid): return str(int(tok_arr_t[min(int(tid), vs-1)].item()))
    def d(tid):
        try: return f"`{tokenizer.decode([int(tid)])}`"
        except: return str(tid)

    probs   = torch.from_numpy(scores).softmax(1).numpy()
    action  = scores.argmax(1)
    if best_thr > 0:
        best_j_c = scores[:, 1:].argmax(1) + 1
        best_p_c = probs[np.arange(N), best_j_c]
        force_n  = (action > 0) & (best_p_c < best_thr)
        action[force_n] = 0
    apply = action > 0
    pool_j = (action - 1).clip(0, P-1)
    sel_toks = cand_np[np.arange(N), pool_j]
    sel_gold = apply & (sel_toks == gold_np)

    topk_ids_t  = torch.from_numpy(row["topk_ids"])
    topk_lgts_t = torch.from_numpy(row["topk_lgts"])
    ref_lgts = _apply_surgical(topk_lgts_t, topk_ids_t,
                                torch.from_numpy(sel_toks.astype(np.int64)),
                                torch.from_numpy(apply), 0.5 * args.margin_delta)
    ref_top1 = row["topk_ids"][np.arange(N), ref_lgts.argmax(1).numpy()]
    ctg = bw & apply & (ref_top1 == gold_np)
    caw = ~bw & apply & (ref_top1 != gold_np)

    def fmt_row(ri):
        lines = [f"### Row {ri}\n"]
        ctx_ids = mem_ids[ri]
        try:
            ctx = tokenizer.decode(ctx_ids[-64:].tolist(), skip_special_tokens=False)
            lines.append(f"**Context (last 64):** `{ctx}`\n")
        except Exception: pass
        gt = int(gold_np[ri]); bt = int(row["base"][ri])
        act_ri = int(action[ri])
        lines.append(f"**Gold:** {d(gt)} id={gt} region={reg(gt)}")
        lines.append(f"**Base top-1:** {d(bt)} id={bt} region={reg(bt)}")
        if act_ri == 0:
            lines.append("**Action:** NO_OP\n")
        else:
            st = int(sel_toks[ri])
            lines.append(f"**Action:** candidate {act_ri-1} = {d(st)} id={st}\n")

        lines.append("| j | Token | ID | Region | Base lgt | Score | Prob | is_gold |")
        lines.append("|---|-------|----|--------|----------|-------|------|---------|")
        lines.append(f"| - | (NO_OP) | - | - | - | {scores[ri,0]:.3f} | {probs[ri,0]:.3f} | - |")
        for j in range(min(P, 10)):
            cid = int(cand_np[ri, j])
            try: ts = tokenizer.decode([cid])
            except: ts = str(cid)
            ig = "✓" if cid == gt else ""
            lines.append(f"| {j+1} | `{ts}` | {cid} | {reg(cid)} | "
                         f"{cand_l[ri,j]:.3f} | {scores[ri,j+1]:.3f} | {probs[ri,j+1]:.3f} | {ig} |")
        lines.append("")

        if act_ri > 0:
            sel_j = act_ri - 1
            aw = attn[ri, sel_j]                          # [M]
            top_k_attn = min(5, len(aw))
            top_idx = np.argsort(aw)[::-1][:top_k_attn]
            lines.append("**Top attended memory tokens:**")
            lines.append("| Pos | Token | ID | Attn weight |")
            lines.append("|-----|-------|----|-------------|")
            for idx in top_idx:
                mid = int(mem_ids[ri, idx])
                try: ms = tokenizer.decode([mid])
                except: ms = str(mid)
                lines.append(f"| {idx} | `{ms}` | {mid} | {aw[idx]:.4f} |")
            lines.append("")

        return "\n".join(lines) + "\n"

    buckets = {
        "selected_gold":  np.where(sel_gold)[0],
        "selected_wrong": np.where(apply & ~(sel_toks == gold_np) & bw)[0],
        "noop_correct":   np.where((~apply) & (~bw))[0],
        "false_apply":    np.where(apply & ~bw)[0],
        "changed_to_gold": np.where(ctg)[0],
        "changed_away":    np.where(caw)[0],
        "attention_debug": np.where(apply)[0],
    }
    for bname, indices in buckets.items():
        n_max = args.num_examples if bname != "attention_debug" else min(args.num_examples, 10)
        if len(indices) > n_max:
            indices = rng.choice(indices, n_max, replace=False)
        path = os.path.join(run_dir, f"examples_{bname}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {bname.replace('_',' ').title()}\n\n_{len(indices)} examples_\n\n---\n\n")
            for ri in indices:
                f.write(fmt_row(int(ri)))
                f.write("---\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(run_dir, args, baselines, best_m0, best_fv, best_step):
    def f(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return "nan"
        if isinstance(v, float): return f"{v:.4f}"
        return str(v) if v is not None else "nan"

    sg   = best_m0.get("selected_gold_bwcov",       float("nan"))
    ctg  = best_m0.get("changed_to_gold_rate",       float("nan"))
    caw  = best_m0.get("changed_away_rate",           float("nan"))
    nca  = best_m0.get("noharm_changed_away",         float("nan"))
    nac  = best_m0.get("noop_acc_on_base_correct",    float("nan"))
    sip  = best_m0.get("selected_gold_given_in_pool", float("nan"))
    fvg  = best_fv.get("full_vocab_gain", float("nan"))

    bl_r2  = baselines.get("baseline_rank2_sg_bwcov", float("nan"))
    s1_sg  = baselines.get("stage1_ref_sel_gold_bwcov", 0.246)
    s1_ctg = baselines.get("stage1_ref_ctg", 0.040)
    s1_caw = baselines.get("stage1_ref_caw", 0.062)

    lines = ["# Local Detail Memory Selector V1\n"]
    lines.append(f"**Run:** `{args.run_name}`  |  **Steps:** {args.steps}  "
                 f"|  **memory_len:** {args.memory_len}  "
                 f"|  **candidate_filter:** `{args.candidate_filter}`\n")

    lines.append("## Reference\n")
    lines.append("| Experiment | sel_gold_bwcov | ctg | caw | fv_gain |")
    lines.append("|-----------|----------------|-----|-----|---------|")
    lines.append("| Old KNN V1 | 0.0070 | — | — | -0.0116 |")
    lines.append("| Stage 1 (frozen h_prime MLP) | ~0.246 | ~0.040 | ~0.062 | — |")
    lines.append(f"| Rank2 always | {f(bl_r2)} | — | — | — |")
    lines.append("| V3 oracle | 0.2407 target_ctg | — | 0.000 | +0.1584 |\n")

    lines.append("## Baselines\n")
    for k, v in baselines.items():
        lines.append(f"  {k:45s} = {f(v)}")
    lines.append("")

    lines.append("## Final Metrics (threshold=0.0)\n")
    for k, v in best_m0.items():
        lines.append(f"  {k:45s} = {f(v)}")
    lines.append("")

    if best_fv:
        lines.append("## Full-Vocab\n")
        for k, v in best_fv.items():
            lines.append(f"  {k:45s} = {f(v)}")
        lines.append("")

    lines.append(f"Best checkpoint: step={best_step}\n")

    lines.append("## Analysis\n")
    def yn(c): return "✅" if c else "⚠️"

    q1 = not math.isnan(sg) and sg >= s1_sg
    lines.append(f"**1. Improves over Stage 1 ({f(sg)} >= {f(s1_sg)})?** {yn(q1)}")

    q2 = not math.isnan(nca) and nca < s1_caw
    lines.append(f"**2. Lower changed_away than Stage 1 ({f(nca)} < {f(s1_caw)})?** {yn(q2)}")

    q3 = not math.isnan(ctg) and not math.isnan(caw) and ctg > caw
    lines.append(f"**3. ctg > caw ({f(ctg)} > {f(caw)})?** {yn(q3)}")

    q4 = not math.isnan(fvg) and fvg >= 0
    lines.append(f"**4. full_vocab_gain >= 0 ({f(fvg)})?** {yn(q4)}")

    lines.append(f"**5. Candidate attention useful?** Check examples_attention_debug.md.")

    lines.append(f"**6. Next step?**")
    if q1 and q2 and q3:
        lines.append("  → Local detail memory works. Proceed to Transformer-XL-style "
                     "recurrence preserving detail memory across segments.")
    elif q1 and not q2:
        lines.append("  → Selection improved but NO_OP still leaks. Improve NO_OP "
                     "calibration or increase noop_weight/candidate_fraction.")
    elif not q1:
        lines.append("  → No improvement over Stage 1. Raw token embeddings are "
                     "insufficient; need hidden-state memory or multi-token path audit.")
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
    train_data = load_shards(args.train_dir, args.top_k, "train", args.max_train_rows)
    val_data   = load_shards(args.val_dir,   args.top_k, "val",   args.max_val_rows)

    print("\n[model] Building LocalDetailSelector...")
    n_regions = unk_region; n_supers = unk_super if sr_enabled else 1
    model = LocalDetailSelector(
        token_emb_weight=tok_w.to(device),
        tok_arr=tok_arr_np, reg_arr=reg_arr_np,
        d_model=d_model, n_regions=n_regions, n_supers=n_supers,
        sr_enabled=sr_enabled, unk_region=unk_region, unk_super=unk_super,
        top_k=args.top_k, pool_size=args.candidate_pool_size,
        memory_len=args.memory_len, resolver_dim=args.resolver_dim,
        hidden_dim=args.hidden_dim, attention_heads=args.attention_heads,
        dropout=args.dropout, region_emb_dim=args.region_emb_dim,
        super_emb_dim=args.super_emb_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    print("\n[train index] Building training row indices...")
    noop_idx, cand_idx = build_train_index(
        train_data, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled, args, device)
    if len(cand_idx) == 0:
        raise RuntimeError("No candidate-label rows found in training data.")

    # Pretrain baseline on val
    print("\n[pretrain baseline]")
    thresholds = [float(x) for x in args.thresholds.split(",")]
    sweep0, _, row0 = evaluate(model, val_data, tok_arr_t, reg_arr_t,
                                 unk_region, unk_super, sr_enabled, args, device, thresholds)
    baselines = compute_baselines(row0, args.margin_delta)
    baselines["val_gold_in_pool_rate"]  = baselines.pop("gold_in_pool_rate")
    baselines["val_base_correct_rate"]  = baselines.pop("base_top1_acc")
    baselines["train_label_noop_rate"]  = _rate(noop_idx, len(noop_idx) + len(cand_idx))
    baselines["train_label_candidate_rate"] = _rate(cand_idx, len(noop_idx) + len(cand_idx))
    baselines["train_ignored_rate"]     = float("nan")   # computed separately if needed
    with open(os.path.join(run_dir, "pretrain_baseline.json"), "w") as f:
        json.dump(_json_safe(baselines), f, indent=2)
    for k, v in baselines.items():
        if v is not None:
            print(f"  {k:45s} = {v:.4f}" if isinstance(v, float) else f"  {k:45s} = {v}")

    config = vars(args).copy()
    config.update({"d_model": d_model, "vocab_size": vocab_size,
                   "sr_enabled": sr_enabled, "n_params": n_params})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None

    cand_bsz = max(1, round(args.batch_size * args.candidate_fraction))
    noop_bsz = args.batch_size - cand_bsz

    N_c = len(cand_idx); N_n = len(noop_idx)
    perm_c = np.random.permutation(N_c); pos_c = 0
    perm_n = np.random.permutation(N_n) if N_n > 0 else None; pos_n = 0

    train_log = os.path.join(run_dir, "train_log.csv")
    eval_log  = os.path.join(run_dir, "eval_log.csv")
    sub_log   = os.path.join(run_dir, "subset_eval_log.csv")
    hdr_t = hdr_e = hdr_s = False

    best_score = -float("inf"); best_step = 0
    step = 0; t0 = time.time()
    optimizer.zero_grad()

    print(f"\n[train] steps={args.steps}  batch={args.batch_size} "
          f"(cand={cand_bsz} noop={noop_bsz})  lr={args.lr}\n")

    while step < args.steps:
        # Sample candidate rows
        if pos_c + cand_bsz > N_c:
            perm_c = np.random.permutation(N_c); pos_c = 0
        c_global = cand_idx[perm_c[pos_c:pos_c + cand_bsz]]; pos_c += cand_bsz

        # Sample noop rows
        if noop_bsz > 0 and N_n > 0:
            if pos_n + noop_bsz > N_n:
                perm_n = np.random.permutation(N_n); pos_n = 0
            n_global = noop_idx[perm_n[pos_n:pos_n + noop_bsz]]; pos_n += noop_bsz
            batch_global = np.concatenate([c_global, n_global])
        else:
            batch_global = c_global

        hp   = train_data["h_prime"][batch_global].to(device)
        topk = train_data["topk_ids"][batch_global].to(device)
        lgt  = train_data["topk_lgt"][batch_global].to(device)
        gold = train_data["gold"][batch_global].to(device)
        iids = train_data["ids"][batch_global].to(device)

        cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
            topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
            args.candidate_pool_size, args.candidate_filter, device)

        lbl = assign_labels(gold, topk, cand_ids, args.uncovered_policy)

        base_top1_ids = topk[:, 0]

        with _amp_ctx(args.amp):
            act_sc, _ = model.forward_full(
                hp, iids, base_top1_ids, lgt,
                cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups)
            total_loss, ld = compute_loss(
                act_sc, lbl, args.noop_weight, args.candidate_weight,
                args.lambda_margin_loss, args.target_margin, device)
            loss_sc = total_loss / args.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss_sc).backward()
        else:
            loss_sc.backward()

        _acc = getattr(main, "_accum", 0) + 1
        main._accum = _acc
        if _acc < args.grad_accum_steps:
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
            sweep, chk_score, val_row = evaluate(
                model, val_data, tok_arr_t, reg_arr_t,
                unk_region, unk_super, sr_enabled, args, device, thresholds)

            m0 = next((m for m in sweep if m["threshold"] == 0.0), sweep[0])
            print(f"  ctg={m0.get('changed_to_gold_rate',float('nan')):.4f}  "
                  f"caw={m0.get('changed_away_rate',float('nan')):.4f}  "
                  f"sg_bwcov={m0.get('selected_gold_bwcov',float('nan')):.4f}  "
                  f"nac={m0.get('noop_acc_on_base_correct',float('nan')):.4f}  "
                  f"score={chk_score:.4f}" if not math.isnan(chk_score) else "  score=nan")

            torch.save({"step": step, "model": model.state_dict(), "args": vars(args)},
                       os.path.join(run_dir, "latest_selector.pt"))

            if not math.isnan(chk_score) and chk_score > best_score:
                best_score = chk_score; best_step = step
                torch.save({"step": step, "model": model.state_dict(), "args": vars(args)},
                           os.path.join(run_dir, "best_selector.pt"))
                with open(os.path.join(run_dir, "best_metrics.json"), "w") as fp:
                    json.dump(_json_safe({"step": step, "score": chk_score,
                                         "metrics": m0}), fp, indent=2)
                print(f"  [best] step={step}  score={chk_score:.4f}")

            ev = {"step": step, **m0}
            _write_csv_row(eval_log, ev, header=not hdr_e); hdr_e = True
            for m in sweep:
                _write_csv_row(sub_log, {"step": step, **m}, header=not hdr_s); hdr_s = True
            print()

    # ── Final eval with best checkpoint ──────────────────────────────────────
    best_path = os.path.join(run_dir, "best_selector.pt")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        loaded_step = int(ck.get("step", -1))
        print(f"[final] Loaded best_selector.pt  step={loaded_step}")
    else:
        loaded_step = -1
        print("[final] WARNING: best_selector.pt not found; using latest weights")

    print("[final] Evaluating on val...")
    final_sweep, final_score, final_row = evaluate(
        model, val_data, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled, args, device, thresholds)

    final_fv = {}
    if args.eval_full_vocab:
        print("[final] Full-vocab eval...")
        final_fv = eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                                    unk_region, unk_super, sr_enabled, args, device, final_row)

    # Write threshold sweep CSV
    thr_csv = os.path.join(run_dir, "threshold_sweep.csv")
    hdr = True
    for m in final_sweep:
        _write_csv_row(thr_csv, m, header=hdr); hdr = False

    final_m0 = next((m for m in final_sweep if m["threshold"] == 0.0), final_sweep[0])
    final_metrics = {
        "step": args.steps,
        "best_step_during_training":       best_step,
        "loaded_best_step_for_final_eval": loaded_step,
        "metrics_thr0": final_m0,
        "fv": final_fv,
        "baselines": baselines,
    }
    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump(_json_safe(final_metrics), f, indent=2)

    try:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            tokenizer = type("FT", (), {"decode": lambda s, ids, **kw: str(ids)})()
        best_thr_for_ex = 0.0
        write_example_reports(val_data, final_row, tok_arr_t.cpu(), unk_region,
                               tokenizer, args, run_dir, best_thr_for_ex)
    except Exception as exc:
        print(f"[warn] example reports failed: {exc}")

    write_report(run_dir, args, baselines, final_m0, final_fv, loaded_step)

    print(f"\n{'='*60}")
    print(f" Local Detail Selector V1 complete.")
    print(f" Run dir: {run_dir}")
    print(f" Best step: {best_step}  score={best_score:.4f}")
    print(f"{'='*60}\n")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",          required=True)
    p.add_argument("--train_dir",           required=True)
    p.add_argument("--val_dir",             required=True)
    p.add_argument("--token_to_region",     required=True)
    p.add_argument("--super_map",           default=None)
    p.add_argument("--output_root",         default="runs/local_detail_selector_v1")
    p.add_argument("--run_name",            default="cand_xattn_tokmem_v1")
    p.add_argument("--top_k",              type=int,   default=256)
    p.add_argument("--candidate_pool_size", type=int,   default=32)
    p.add_argument("--memory_len",          type=int,   default=128)
    p.add_argument("--candidate_filter",    default="top_rank",
                   choices=["top_rank","same_region_or_superregion",
                            "same_region_only","same_superregion_only"])
    p.add_argument("--uncovered_policy",    default="ignore",
                   choices=["ignore","noop"])
    p.add_argument("--resolver_dim",        type=int,   default=256)
    p.add_argument("--hidden_dim",          type=int,   default=512)
    p.add_argument("--attention_heads",     type=int,   default=4)
    p.add_argument("--dropout",             type=float, default=0.1)
    p.add_argument("--region_emb_dim",      type=int,   default=64)
    p.add_argument("--super_emb_dim",       type=int,   default=32)
    p.add_argument("--batch_size",          type=int,   default=128)
    p.add_argument("--candidate_fraction",  type=float, default=0.5)
    p.add_argument("--grad_accum_steps",    type=int,   default=1)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--steps",               type=int,   default=5000)
    p.add_argument("--eval_every",          type=int,   default=500)
    p.add_argument("--eval_batch_size",     type=int,   default=256)
    p.add_argument("--grad_clip",           type=float, default=1.0)
    p.add_argument("--noop_weight",         type=float, default=0.5)
    p.add_argument("--candidate_weight",    type=float, default=1.0)
    p.add_argument("--lambda_margin_loss",  type=float, default=0.25)
    p.add_argument("--target_margin",       type=float, default=1.0)
    p.add_argument("--margin_delta",        type=float, default=1.0)
    p.add_argument("--thresholds",          default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    p.add_argument("--eval_full_vocab",     action="store_true")
    p.add_argument("--amp",                 action="store_true")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--max_train_rows",      type=int,   default=None)
    p.add_argument("--max_val_rows",        type=int,   default=None)
    p.add_argument("--num_examples",        type=int,   default=40)
    return p.parse_args()


if __name__ == "__main__":
    main()
