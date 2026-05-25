#!/usr/bin/env python3
"""
train_txl_detail_resolver_v1.py — TXL Detail Resolver V1
Stage 3: TXL-style Detail Memory + Candidate Selector + Edit Gate.

Stage 1:  sel_gold_bwcov≈0.246, ctg≈0.040, caw≈0.062, fv_gain≈-0.0199
Stage 2:  ctg≈0.054–0.063, caw≈0.079–0.094, fv_gain≈-0.0173
Stage 2B: collapsed (apply_rate≈0.0017) due to joint NO_OP softmax pressure.

Stage 3 motivation:
  Stage 2B's joint softmax collapses into NO_OP because CE pressure from
  base-correct rows overwhelms candidate signal.  Fix: decouple
  "which candidate?" (selector) from "should we edit at all?" (gate).

Architecture:
  - forward_shared:  pair features + cross-attention → cand_repr [B,P,D]
                     + selector_scores [B,P]
  - gate_forward:    gate features (cand_repr + hp_proj + scalars) → edit_logit [B]

Training:
  - Selector CE  (correctable rows only): gold pool index
  - Gate BCE (correctable + base_correct):
      correctable → gate_label = 1 if surgery(gold) makes gold top1
      base_correct → gate_label = 0 (hard negative + no-harm margin loss)
  - Teacher forcing for gate:
      correctable  → gate_cand_idx = gold_pool_idx
      base_correct → gate_cand_idx = argmax(sel_scores.detach())

Strict constraints preserved:
  1. No retrieval.                    6. Surgical edit only (selected + base).
  2. No KNN.                          7. No hand-coded token type features.
  3. No gold at eval (argmax only).   8. Features: h_prime, embeddings, logits,
  4. No forced gold in candidates.       ranks, region/super IDs only.
  5. No edit of all top-256.          9. No candidate CE as full-vocab NLL.
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
            topk = torch.cat([topk, torch.zeros(B, top_k - K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, top_k - K), float("nan"))], 1)
        elif K > top_k:
            topk, lgt = topk[:, :top_k], lgt[:, :top_k]
        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            hp, topk, lgt, gold, ids = (hp[:keep], topk[:keep], lgt[:keep],
                                         gold[:keep], ids[:keep])
            B = keep
        all_hp.append(hp); all_topk.append(topk)
        all_lgt.append(lgt); all_gold.append(gold)
        all_ids.append(ids)
        total += B
    data = {
        "h_prime":  torch.cat(all_hp,   0),
        "topk_ids": torch.cat(all_topk, 0),
        "topk_lgt": torch.cat(all_lgt,  0),
        "gold":     torch.cat(all_gold, 0),
        "ids":      torch.cat(all_ids,  0),
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
    cand_sups = (torch.full((N, pool_size), unk_super, dtype=torch.long, device=device)
                 if sr_enabled else None)

    base_regs = tok_arr_t[topk_ids[:, 0].clamp(0, vs - 1)]
    base_sups = reg_arr_t[base_regs.clamp(0, rlen)] if sr_enabled else None

    for i in range(N):
        c_toks = topk_ids[i, 1:]
        c_lgts = topk_lgt[i, 1:]
        c_rnks = torch.arange(1, K, device=device)

        if candidate_filter != "top_rank":
            cr = tok_arr_t[c_toks.clamp(0, vs - 1)]
            br = base_regs[i]
            if candidate_filter == "same_region_only":
                mask = (cr == br) & (cr != unk_region)
            elif candidate_filter == "same_superregion_only":
                if sr_enabled:
                    cs   = reg_arr_t[cr.clamp(0, rlen)]
                    mask = (cs == base_sups[i]) & (cs != unk_super)
                else:
                    mask = torch.zeros(len(c_toks), dtype=torch.bool, device=device)
            elif candidate_filter == "same_region_or_superregion":
                same_r = (cr == br) & (cr != unk_region)
                if sr_enabled:
                    cs     = reg_arr_t[cr.clamp(0, rlen)]
                    same_s = (cs == base_sups[i]) & (cs != unk_super)
                else:
                    same_s = torch.zeros_like(same_r)
                mask = same_r | same_s
            else:
                mask = torch.ones(len(c_toks), dtype=torch.bool, device=device)
            if mask.any():
                c_toks = c_toks[mask]; c_lgts = c_lgts[mask]; c_rnks = c_rnks[mask]

        n_take = min(pool_size, len(c_toks))
        cand_ids[i,  :n_take] = c_toks[:n_take]
        cand_lgts[i, :n_take] = c_lgts[:n_take]
        cand_rnks[i, :n_take] = c_rnks[:n_take]
        cr_fill = tok_arr_t[c_toks[:n_take].clamp(0, vs - 1)]
        cand_regs[i, :n_take] = cr_fill
        if sr_enabled and cand_sups is not None:
            cand_sups[i, :n_take] = reg_arr_t[cr_fill.clamp(0, rlen)]

    return cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups


# ─────────────────────────────────────────────────────────────────────────────
# TXL Detail Memory
# ─────────────────────────────────────────────────────────────────────────────

class TXLDetailMemory(nn.Module):
    """
    Transformer-XL-style contextual detail memory encoder.

    Encodes input_ids using frozen token embeddings → contextual detail tokens
    + learned memory summary slots.

    Output: full_memory [B, T+S, mem_dim]  (T = seq, S = num_memory_slots)

    pos_encoding choices:
      "alibi"            — ALiBi relative-position bias (no position embedding)
      "learned_relative" — learned position embeddings (same impl as absolute for Patch B)
      "absolute"         — learned absolute position embeddings
    """

    def __init__(self, d_model, mem_dim, txl_layers, txl_heads, txl_ff_dim,
                 dropout, num_memory_slots, pos_encoding, max_seq_len=512):
        super().__init__()
        self.mem_dim          = mem_dim
        self.num_memory_slots = num_memory_slots
        self.pos_encoding     = pos_encoding
        self.txl_heads        = txl_heads

        self.input_proj = nn.Linear(d_model, mem_dim, bias=False)

        # Positional encoding (absolute or learned_relative use embedding;
        # alibi uses no additive embedding — bias applied in attn mask instead)
        if pos_encoding in ("absolute", "learned_relative"):
            self.pos_emb = nn.Embedding(max_seq_len, mem_dim)
        else:
            self.pos_emb = None

        # ALiBi slopes (pre-computed, registered as buffer)
        if pos_encoding == "alibi":
            slopes = self._get_alibi_slopes(txl_heads)
            self.register_buffer("alibi_slopes", slopes)
        else:
            self.register_buffer("alibi_slopes", None)

        # Transformer encoder (pre-LN for stability)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=mem_dim,
            nhead=txl_heads,
            dim_feedforward=txl_ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=txl_layers)

        # Learned memory summary slots
        self.memory_slot_init = nn.Parameter(
            torch.randn(num_memory_slots, mem_dim) * 0.02)
        slot_heads = max(1, txl_heads // 2)
        self.slot_attn = nn.MultiheadAttention(
            embed_dim=mem_dim, num_heads=slot_heads,
            dropout=dropout, batch_first=True)
        self.slot_norm = nn.LayerNorm(mem_dim)

    @staticmethod
    def _get_alibi_slopes(n_heads):
        """ALiBi slopes: 2^(-8/n * i) for i = 1..n_heads."""
        def _pow2_slopes(n):
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
            return [start * (start ** i) for i in range(n)]

        if n_heads > 0 and (n_heads & (n_heads - 1)) == 0:
            slopes = _pow2_slopes(n_heads)
        else:
            n_p2   = 2 ** math.floor(math.log2(max(n_heads, 1)))
            slopes = _pow2_slopes(n_p2)
            extra  = _pow2_slopes(2 * n_p2)[0::2][: n_heads - n_p2]
            slopes = slopes + extra
        return torch.tensor(slopes[:n_heads], dtype=torch.float32)

    def _alibi_mask(self, T, device):
        """ALiBi bias matrix [n_heads*B, T, T] — added to pre-softmax attention."""
        pos  = torch.arange(T, device=device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()   # [T,T]
        # slopes [n_heads] × dist [T,T] → [n_heads, T, T]
        bias = -self.alibi_slopes.to(device).view(-1, 1, 1) * dist.unsqueeze(0)
        return bias  # [n_heads, T, T]  — broadcast over batch in encoder

    def forward(self, input_ids, token_emb_weight):
        """
        input_ids:        [B, T]
        token_emb_weight: frozen [V, d_model]
        Returns full_memory [B, T+S, mem_dim]
        """
        B, T = input_ids.shape
        vs   = token_emb_weight.shape[0]
        dev  = input_ids.device

        tok_emb = token_emb_weight[input_ids.clamp(0, vs - 1)]  # [B, T, d_model]
        x       = self.input_proj(tok_emb)                       # [B, T, mem_dim]

        if self.pos_emb is not None:
            pos = torch.arange(T, device=dev).clamp(0, self.pos_emb.num_embeddings - 1)
            x   = x + self.pos_emb(pos)

        # Build ALiBi attn_mask for TransformerEncoder (expects [T,T] or [B*H,T,T])
        attn_mask = None
        if self.pos_encoding == "alibi" and self.alibi_slopes is not None:
            bias = self._alibi_mask(T, dev)               # [H, T, T]
            attn_mask = bias.unsqueeze(0).expand(B, -1, -1, -1) \
                           .reshape(B * self.txl_heads, T, T)

        detail_tokens = (self.encoder(x, mask=attn_mask)
                         if attn_mask is not None else self.encoder(x))   # [B,T,mem_dim]

        # Memory slots attend over detail tokens
        slots   = self.memory_slot_init.unsqueeze(0).expand(B, -1, -1)    # [B,S,mem_dim]
        s_out, _ = self.slot_attn(slots, detail_tokens, detail_tokens,
                                   need_weights=False)
        slot_out = self.slot_norm(slots + s_out)                           # [B,S,mem_dim]

        return torch.cat([detail_tokens, slot_out], dim=1)                 # [B,T+S,mem_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Model — Decoupled Selector + Gate
# ─────────────────────────────────────────────────────────────────────────────

class LocalDetailSelectorGate(nn.Module):
    """
    Decoupled selector + edit gate.

    forward_shared  → selector_scores [B,P], cand_repr [B,P,D], attn_w [B,P,M]
    gate_forward    → edit_logit [B]
    forward_eval    → selector_scores, edit_logit, attn_w, gate_cand_idx

    Gate feature dim: 2*resolver_dim + region_emb_dim + 13 (+ super_emb_dim if sr)
    13 gate scalars:
      sel_lgts, base_lgt, sel_logit_gap, base_logit_gap, sel_rank_norm,
      sel_score, sel_top1, sel_margin,
      hp_dot_sel, hp_dot_base, hp_dot_gap,
      same_region, same_super
    """

    def __init__(self, token_emb_weight, tok_arr, reg_arr,
                 d_model, n_regions, n_supers, sr_enabled,
                 unk_region, unk_super, top_k, pool_size,
                 memory_len=128, resolver_dim=256, hidden_dim=512,
                 attention_heads=4, dropout=0.1,
                 region_emb_dim=64, super_emb_dim=32,
                 # TXL params
                 memory_backend="token",
                 mem_dim=256, txl_layers=2, txl_heads=4, txl_ff_dim=1024,
                 num_memory_slots=16, pos_encoding="alibi"):
        super().__init__()
        self.d_model         = d_model
        self.sr_enabled      = sr_enabled
        self.unk_region      = unk_region
        self.unk_super       = unk_super
        self.top_k           = top_k
        self.pool_size       = pool_size
        self.memory_len      = memory_len
        self.resolver_dim    = resolver_dim
        self.memory_backend  = memory_backend

        self.register_buffer("token_emb_weight", token_emb_weight.detach().float())
        self.register_buffer("tok_arr", torch.from_numpy(tok_arr.astype(np.int32)).long())
        self.register_buffer("reg_arr", torch.from_numpy(reg_arr.astype(np.int32)).long())

        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim)
        self.super_emb  = nn.Embedding(n_supers + 2, super_emb_dim) if sr_enabled else None

        # ── Selector (same pair-feature architecture as V1C) ──
        pair_feat_dim = 5 * d_model + 7 + 2 * region_emb_dim + 1
        if sr_enabled:
            pair_feat_dim += 2 * super_emb_dim + 1

        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, resolver_dim),
            nn.GELU(),
        )

        # Memory backend — token (V1C-identical) or txl (new)
        if memory_backend == "token":
            self.mem_proj    = nn.Linear(d_model, resolver_dim, bias=False)
            self.txl_memory  = None
            self.txl_to_resolver = None
        elif memory_backend == "txl":
            self.mem_proj    = None   # not used in TXL mode
            max_seq          = memory_len + 64
            self.txl_memory  = TXLDetailMemory(
                d_model=d_model, mem_dim=mem_dim,
                txl_layers=txl_layers, txl_heads=txl_heads,
                txl_ff_dim=txl_ff_dim, dropout=dropout,
                num_memory_slots=num_memory_slots,
                pos_encoding=pos_encoding, max_seq_len=max_seq)
            # Project TXL mem_dim → resolver_dim for cross-attention keys/values
            self.txl_to_resolver = nn.Linear(mem_dim, resolver_dim, bias=False)
        else:
            raise ValueError(f"Unknown memory_backend: {memory_backend!r}. "
                             "Choose 'token' or 'txl'.")

        self.cross_attn     = nn.MultiheadAttention(
            embed_dim=resolver_dim, num_heads=attention_heads,
            dropout=dropout, batch_first=True)
        self.post_attn_norm = nn.LayerNorm(resolver_dim)
        self.score_head     = nn.Linear(resolver_dim, 1, bias=True)

        # ── Gate ──
        self.gate_hp_proj = nn.Linear(d_model, resolver_dim, bias=False)
        gate_feat_dim = 2 * resolver_dim + region_emb_dim + 13
        if sr_enabled:
            gate_feat_dim += super_emb_dim
        gate_hidden = resolver_dim // 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gate_hidden),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(gate_hidden, 1, bias=True)

    def _pair_feat(self, hp, ec, eb, lgt_c, lgt_b, rank_norm, reg_c, reg_b,
                   sup_c=None, sup_b=None):
        logit_gap  = lgt_c - lgt_b
        hp_dot_c   = (hp * ec).sum(-1, keepdim=True)
        hp_dot_b   = (hp * eb).sum(-1, keepdim=True)
        hp_dot_gap = hp_dot_c - hp_dot_b
        same_reg   = ((reg_c == reg_b) & (reg_c != self.unk_region)).float().unsqueeze(-1)
        remb_c = self.region_emb(reg_c.clamp(0, self.region_emb.num_embeddings - 1))
        remb_b = self.region_emb(reg_b.clamp(0, self.region_emb.num_embeddings - 1))
        parts = [hp, ec, eb, ec - eb, ec * eb,
                 lgt_c.unsqueeze(-1), lgt_b.unsqueeze(-1),
                 logit_gap.unsqueeze(-1), rank_norm.unsqueeze(-1),
                 hp_dot_c, hp_dot_b, hp_dot_gap,
                 remb_c, remb_b, same_reg]
        if self.sr_enabled and self.super_emb is not None and sup_c is not None:
            rlen   = self.reg_arr.shape[0] - 1
            semb_c = self.super_emb(sup_c.clamp(0, self.super_emb.num_embeddings - 1))
            semb_b = self.super_emb(sup_b.clamp(0, self.super_emb.num_embeddings - 1))
            same_sup = ((sup_c == sup_b) & (sup_c != self.unk_super)).float().unsqueeze(-1)
            parts.extend([semb_c, semb_b, same_sup])
        return torch.cat(parts, dim=-1)

    def forward_shared(self, h_prime, input_ids, base_top1_ids, base_lgts,
                       cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups=None):
        """Shared encoder: returns selector_scores [B,P], cand_repr [B,P,D], attn_w [B,P,M]."""
        B, P  = cand_ids.shape
        K     = base_lgts.shape[1]
        d     = self.d_model
        vs    = self.token_emb_weight.shape[0]
        rlen  = self.reg_arr.shape[0] - 1

        base_emb = self.token_emb_weight[base_top1_ids.clamp(0, vs - 1)]
        base_reg = self.tok_arr[base_top1_ids.clamp(0, self.tok_arr.shape[0] - 1)]
        base_lgt = base_lgts[:, 0]
        base_sup = self.reg_arr[base_reg.clamp(0, rlen)] if self.sr_enabled else None

        cand_emb = self.token_emb_weight[
            cand_ids.reshape(-1).clamp(0, vs - 1)].view(B, P, d)

        # ── Memory keys/values ───────────────────────────────────────────────
        mem_ids = input_ids[:, -self.memory_len:]
        if self.memory_backend == "token":
            mem_emb = self.token_emb_weight[mem_ids.clamp(0, vs - 1)]
            mem_kv  = self.mem_proj(mem_emb)                   # [B, M, resolver_dim]
        elif self.memory_backend == "txl":
            if self.txl_memory is None or self.txl_to_resolver is None:
                raise RuntimeError("TXL memory backend selected but not initialised.")
            full_mem = self.txl_memory(mem_ids, self.token_emb_weight)  # [B, T+S, mem_dim]
            mem_kv   = self.txl_to_resolver(full_mem)                   # [B, T+S, resolver_dim]
        else:
            raise RuntimeError(f"Unknown memory_backend: {self.memory_backend!r}")

        hp_exp = h_prime.unsqueeze(1).expand(B, P, d)
        be_exp = base_emb.unsqueeze(1).expand(B, P, d)
        bl_exp = base_lgt.unsqueeze(1).expand(B, P)
        br_exp = base_reg.unsqueeze(1).expand(B, P)
        bs_exp = base_sup.unsqueeze(1).expand(B, P) if base_sup is not None else None
        rn     = cand_rnks.float() / max(K, 1)

        pair_feat = self._pair_feat(
            hp_exp.reshape(B * P, d),  cand_emb.reshape(B * P, d),
            be_exp.reshape(B * P, d),  cand_lgts.reshape(B * P),
            bl_exp.reshape(B * P),     rn.reshape(B * P),
            cand_regs.reshape(B * P),  br_exp.reshape(B * P),
            cand_sups.reshape(B * P) if (self.sr_enabled and cand_sups is not None) else None,
            bs_exp.reshape(B * P)    if bs_exp is not None else None,
        )
        pair_hid = self.pair_mlp(pair_feat).view(B, P, self.resolver_dim)

        M       = mem_kv.shape[1]
        q_flat  = pair_hid.reshape(B * P, 1, self.resolver_dim)
        kv_flat = (mem_kv.unsqueeze(1).expand(B, P, M, self.resolver_dim)
                         .reshape(B * P, M, self.resolver_dim))

        attn_out, attn_w = self.cross_attn(q_flat, kv_flat, kv_flat,
                                            need_weights=True, average_attn_weights=True)
        evidence      = attn_out.squeeze(1).view(B, P, self.resolver_dim)
        cand_repr     = self.post_attn_norm(pair_hid + evidence)   # [B, P, resolver_dim]
        sel_scores    = self.score_head(cand_repr).squeeze(-1)     # [B, P]
        attn_w_out    = attn_w.squeeze(1).view(B, P, M)
        return sel_scores, cand_repr, attn_w_out

    def gate_forward(self, cand_repr, gate_cand_idx, h_prime, base_lgts,
                     cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups,
                     selector_scores, base_top1_ids):
        """Gate head: build features for selected candidate → edit_logit [B]."""
        B    = h_prime.shape[0]
        P    = cand_ids.shape[1]
        K    = base_lgts.shape[1]
        vs   = self.token_emb_weight.shape[0]
        rlen = self.reg_arr.shape[0] - 1
        ar   = torch.arange(B, device=h_prime.device)

        # ── Representations ─────────────────────────────────────────────────
        gate_cand_repr = cand_repr[ar, gate_cand_idx]          # [B, resolver_dim]
        hp_gate        = self.gate_hp_proj(h_prime)            # [B, resolver_dim]

        sel_reg  = cand_regs[ar, gate_cand_idx]                # [B]
        sel_remb = self.region_emb(sel_reg.clamp(0, self.region_emb.num_embeddings - 1))

        # ── 13 scalars ───────────────────────────────────────────────────────
        base_lgt_v    = base_lgts[:, 0]
        lgt1          = base_lgts[:, 1].nan_to_num(0.0) if K > 1 else base_lgt_v
        base_logit_gap = base_lgt_v - lgt1                     # confidence at top

        sel_lgt_v     = cand_lgts[ar, gate_cand_idx]
        sel_logit_gap = sel_lgt_v - base_lgt_v                 # negative (cand < base)
        sel_rank_norm = cand_rnks[ar, gate_cand_idx].float() / max(K, 1)

        sel_score_v   = selector_scores[ar, gate_cand_idx]
        sel_top1_v    = (selector_scores.argmax(1) == gate_cand_idx).float()

        # selector margin: score(gate_cand) - max(others)
        sc_copy = selector_scores.clone()
        sc_copy[ar, gate_cand_idx] = float("-inf")
        max_other_sel = sc_copy.max(1).values
        sel_margin_v  = sel_score_v - max_other_sel

        # h_prime · token_emb dot products
        sel_ids   = cand_ids[ar, gate_cand_idx]
        sel_emb   = self.token_emb_weight[sel_ids.clamp(0, vs - 1)]
        base_emb_v = self.token_emb_weight[base_top1_ids.clamp(0, vs - 1)]
        hp_dot_sel  = (h_prime * sel_emb).sum(-1)
        hp_dot_base = (h_prime * base_emb_v).sum(-1)
        hp_dot_gap  = hp_dot_sel - hp_dot_base

        base_reg_v = self.tok_arr[base_top1_ids.clamp(0, self.tok_arr.shape[0] - 1)]
        same_region = ((sel_reg == base_reg_v) & (sel_reg != self.unk_region)).float()

        if self.sr_enabled and self.super_emb is not None and cand_sups is not None:
            sel_sup  = cand_sups[ar, gate_cand_idx]
            sel_semb = self.super_emb(sel_sup.clamp(0, self.super_emb.num_embeddings - 1))
            base_sup = self.reg_arr[base_reg_v.clamp(0, rlen)]
            same_super = ((sel_sup == base_sup) & (sel_sup != self.unk_super)).float()
        else:
            sel_semb   = None
            same_super = torch.zeros(B, device=h_prime.device)

        # ── Assemble gate feature vector ─────────────────────────────────────
        def u(t):
            return t.unsqueeze(-1)

        parts = [gate_cand_repr, hp_gate, sel_remb]
        if sel_semb is not None:
            parts.append(sel_semb)
        parts += [
            u(sel_lgt_v),      u(base_lgt_v),    u(sel_logit_gap),
            u(base_logit_gap), u(sel_rank_norm),  u(sel_score_v),
            u(sel_top1_v),     u(sel_margin_v),   u(hp_dot_sel),
            u(hp_dot_base),    u(hp_dot_gap),     u(same_region),
            u(same_super),
        ]
        gate_feat  = torch.cat(parts, dim=-1)
        gate_hid   = self.gate_mlp(gate_feat)
        edit_logit = self.gate_head(gate_hid).squeeze(-1)       # [B]
        return edit_logit

    def forward_eval(self, h_prime, input_ids, base_top1_ids, base_lgts,
                     cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups=None):
        """Eval: gate_cand_idx = argmax(sel_scores). No gold used. Rule 3 preserved."""
        sel_scores, cand_repr, attn_w = self.forward_shared(
            h_prime, input_ids, base_top1_ids, base_lgts,
            cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups)
        gate_cand_idx = sel_scores.argmax(1)                    # [B]
        edit_logit = self.gate_forward(
            cand_repr, gate_cand_idx, h_prime, base_lgts,
            cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups,
            sel_scores, base_top1_ids)
        return sel_scores, edit_logit, attn_w, gate_cand_idx

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use forward_shared / gate_forward / forward_eval")


# ─────────────────────────────────────────────────────────────────────────────
# Gate labels  (vectorised surgery simulation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gate_labels(gold, topk_ids, topk_lgt, cand_ids, gate_cand_idx,
                         is_correctable, md_half, device):
    """
    Returns gate_labels [B] float.
    - is_correctable rows:  gate_label = 1 if surgery(gate_cand) makes gold top1
                            (gate_cand == gold for correctable rows by teacher forcing)
    - all other rows:       gate_label = 0  (base_correct hard negatives, ignored rows)
    """
    B = gold.shape[0]
    gate_labels = torch.zeros(B, dtype=torch.float, device=device)

    if not is_correctable.any():
        return gate_labels

    ci        = gate_cand_idx[is_correctable]          # [n_corr]
    cand_c    = cand_ids[is_correctable]               # [n_corr, P]
    n_corr    = ci.shape[0]
    ar_c      = torch.arange(n_corr, device=device)
    sel_toks  = cand_c[ar_c, ci]                       # [n_corr] ≡ gold[is_correctable]

    topk_ids_c = topk_ids[is_correctable]              # [n_corr, K]
    topk_lgt_c = topk_lgt[is_correctable].nan_to_num(-1e9)
    gold_c     = gold[is_correctable]

    # Find sel_tok position in topk (may be absent if cand_ids padded with 0)
    sel_pos    = (topk_ids_c == sel_toks.unsqueeze(1)).long().argmax(1)   # [n_corr]
    found      = topk_ids_c[ar_c, sel_pos] == sel_toks                   # [n_corr]

    # Simulate surgical edit: selected += md_half, base_top1 -= md_half
    ref_lgts = topk_lgt_c.clone()
    ref_lgts[ar_c, 0] -= md_half                          # penalise base_top1
    ref_lgts[ar_c[found], sel_pos[found]] += md_half       # boost selected (if found)

    ref_top1_pos = ref_lgts.argmax(1)
    ref_top1     = topk_ids_c[ar_c, ref_top1_pos]
    gate_labels[is_correctable] = (ref_top1 == gold_c).float()
    return gate_labels


# ─────────────────────────────────────────────────────────────────────────────
# Loss — V1C: selector CE + gate BCE + no-harm margin + selector margin
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss_v1c(sel_scores, sel_labels, edit_logit, gate_labels,
                      is_correctable, is_base_correct,
                      lambda_selector, lambda_gate, gate_pos_weight,
                      lambda_noharm_gate, gate_margin,
                      lambda_selector_margin, selector_margin_val,
                      device):
    """
    L = lambda_selector  * L_select          (CE, correctable rows)
      + lambda_gate      * L_gate            (BCE, correctable + base_correct)
      + lambda_noharm_gate * L_noharm        (softplus, base_correct rows)
      + lambda_selector_margin * L_sel_margin (softplus, correctable rows)
    """
    # ── Selector CE (correctable only) ───────────────────────────────────────
    L_select = torch.tensor(0.0, device=device)
    n_corr   = int(is_correctable.sum())
    if is_correctable.any():
        sc_c = sel_scores[is_correctable]           # [n_corr, P]
        lb_c = sel_labels[is_correctable]           # [n_corr]
        # Clamp labels to valid pool range
        P    = sc_c.shape[1]
        lb_c = lb_c.clamp(0, P - 1)
        L_select = F.cross_entropy(sc_c, lb_c)

    # ── Gate BCE (correctable + base_correct) ────────────────────────────────
    gate_mask = is_correctable | is_base_correct
    L_gate    = torch.tensor(0.0, device=device)
    if gate_mask.any():
        el_m = edit_logit[gate_mask]
        gl_m = gate_labels[gate_mask]
        pos_w = torch.tensor(gate_pos_weight, dtype=torch.float, device=device)
        L_gate = F.binary_cross_entropy_with_logits(el_m, gl_m, pos_weight=pos_w)

    # ── No-harm gate margin (base_correct rows) ──────────────────────────────
    # Pushes edit_logit negative for base_correct rows
    L_noharm = torch.tensor(0.0, device=device)
    if lambda_noharm_gate > 0 and is_base_correct.any():
        el_bc    = edit_logit[is_base_correct]
        L_noharm = F.softplus(gate_margin + el_bc).mean()

    # ── Selector margin (correctable rows) ───────────────────────────────────
    L_sel_margin = torch.tensor(0.0, device=device)
    if lambda_selector_margin > 0 and is_correctable.any():
        sc_c  = sel_scores[is_correctable]
        lb_c  = sel_labels[is_correctable].clamp(0, sc_c.shape[1] - 1)
        ar_c  = torch.arange(sc_c.shape[0], device=device)
        s_gold = sc_c[ar_c, lb_c]
        mask   = torch.ones_like(sc_c, dtype=torch.bool)
        mask[ar_c, lb_c] = False
        s_wrong = sc_c.masked_fill(~mask, float("-inf")).max(1).values
        L_sel_margin = F.softplus(selector_margin_val - (s_gold - s_wrong)).mean()

    total = (lambda_selector  * L_select
           + lambda_gate      * L_gate
           + lambda_noharm_gate * L_noharm
           + lambda_selector_margin * L_sel_margin)

    return total, {
        "total":        total.item(),
        "l_select":     L_select.item()     if n_corr > 0                           else 0.0,
        "l_gate":       L_gate.item()       if gate_mask.any()                      else 0.0,
        "l_noharm":     L_noharm.item()     if (lambda_noharm_gate > 0
                                                 and is_base_correct.any())         else 0.0,
        "l_sel_margin": L_sel_margin.item() if (lambda_selector_margin > 0
                                                 and is_correctable.any())          else 0.0,
        "n_corr":       n_corr,
        "n_bc":         int(is_base_correct.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rate(mask_or_count, denom):
    if denom == 0:
        return float("nan")
    if isinstance(mask_or_count, (int, float)):
        return float(mask_or_count) / denom
    return float(np.asarray(mask_or_count, dtype=float).sum()) / denom


def _json_safe(obj):
    if isinstance(obj, dict):        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating):
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
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def _apply_surgical(topk_lgts, topk_ids, sel_toks, apply_mask, md_half):
    refined = topk_lgts.clone()
    if isinstance(apply_mask, np.ndarray):
        apply_mask = torch.from_numpy(apply_mask.astype(bool))
    if not apply_mask.any():
        return refined
    N  = topk_lgts.shape[0]
    ar = torch.arange(N, device=topk_lgts.device)
    st = torch.as_tensor(sel_toks, device=topk_lgts.device)
    pos = (topk_ids == st.unsqueeze(1)).long().argmax(1)
    ok  = (topk_ids[ar, pos] == st) & apply_mask.to(topk_lgts.device)
    if ok.any():
        refined[ar[ok], pos[ok]] += md_half
        refined[ar[ok], 0]       -= md_half
    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation policy — gate_thr × selector_margin_thr
# ─────────────────────────────────────────────────────────────────────────────

def _apply_policy_v1c(sel_scores_np, edit_logit_np, gate_cand_idx_np,
                       gate_thr, sel_margin_thr):
    """
    Apply if:
      sigmoid(edit_logit) >= gate_thr  AND
      (sel_margin_thr <= -500  OR  sel_margin(gate_cand) >= sel_margin_thr)

    Returns action [N]: 0=NO_OP, j+1=use candidate j (j = gate_cand_idx).
    sel_margin_thr = -999 means "no margin check" (use gate alone).
    """
    N = sel_scores_np.shape[0]
    P = sel_scores_np.shape[1]
    action = np.zeros(N, dtype=np.int64)

    gate_prob = 1.0 / (1.0 + np.exp(-edit_logit_np.clip(-50.0, 50.0)))

    ar = np.arange(N)
    gc = gate_cand_idx_np.astype(np.int64).clip(0, P - 1)

    sel_sc_at_gc = sel_scores_np[ar, gc]
    sc_copy      = sel_scores_np.copy()
    sc_copy[ar, gc] = -np.inf
    max_other    = sc_copy.max(1)
    sel_margin   = sel_sc_at_gc - max_other

    gate_pass  = gate_prob >= gate_thr
    no_margin  = sel_margin_thr <= -500.0
    if no_margin:
        margin_pass = np.ones(N, dtype=bool)
    else:
        margin_pass = sel_margin >= sel_margin_thr

    apply = gate_pass & margin_pass
    action[apply] = gc[apply] + 1
    return action


def _metrics_from_action(action, gold_np, bw_np, pg_np, sr_np, ss_np,
                           cand_np, topk_ids_np, topk_lgts_np, md_half,
                           selector_ce=float("nan"), selector_acc=float("nan")):
    """Compute full metric dict from action array [N] (0=NO_OP, j+1=cand j)."""
    N  = len(action)
    P  = cand_np.shape[1]

    apply_mask = action > 0
    pool_j     = (action - 1).clip(0, P - 1)
    sel_toks   = cand_np[np.arange(N), pool_j]

    topk_ids_t  = torch.from_numpy(topk_ids_np)
    topk_lgts_t = torch.from_numpy(topk_lgts_np)
    ref_lgts  = _apply_surgical(topk_lgts_t, topk_ids_t,
                                 torch.from_numpy(sel_toks.astype(np.int64)),
                                 torch.from_numpy(apply_mask), md_half)
    ref_top1i = ref_lgts.argmax(1).numpy()
    ref_top1  = topk_ids_np[np.arange(N), ref_top1i]

    sel_gold = sel_toks == gold_np
    ctg      = bw_np  & apply_mask & (ref_top1 == gold_np)
    caw      = ~bw_np & apply_mask & (ref_top1 != gold_np)

    n_bc     = int((~bw_np).sum())
    n_bw     = int(bw_np.sum())
    n_bwcov  = int((bw_np & pg_np).sum())
    n_pg     = int(pg_np.sum())
    n_tgt    = int((bw_np & pg_np & (sr_np | ss_np)).sum())
    n_app    = int(apply_mask.sum())

    noop_correct = (~bw_np) & (~apply_mask)
    ctg_count = int(ctg.sum())
    caw_count = int(caw.sum())

    applied_prec_ctg    = (ctg_count / n_app)   if n_app > 0 else float("nan")
    applied_damage_rate = (caw_count / n_app)   if n_app > 0 else float("nan")
    bdrat               = ctg_count / max(caw_count, 1)
    net_corr            = _rate(ctg, N) - _rate(caw, N) if N > 0 else float("nan")

    return {
        "apply_rate":                  _rate(apply_mask,              N),
        "noop_rate":                   _rate(~apply_mask,             N),
        "noop_acc_on_base_correct":    _rate(noop_correct,            n_bc),
        "noharm_false_apply_rate":     _rate(apply_mask & ~bw_np,     n_bc),
        "selected_gold_bwcov":         _rate(sel_gold[bw_np & pg_np], n_bwcov),
        "selected_gold_given_in_pool": _rate(sel_gold[pg_np],         n_pg),
        "selected_gold_target":        _rate(sel_gold[bw_np & pg_np & (sr_np | ss_np)], n_tgt),
        "changed_to_gold_rate":        _rate(ctg,                     N),
        "changed_away_rate":           _rate(caw,                     N),
        "noharm_changed_away":         _rate(caw,                     n_bc),
        "top1_acc_base":               _rate(~bw_np,                  N),
        "top1_acc_refined":            _rate(ref_top1 == gold_np,     N),
        "top1_acc_gain":               _rate(ref_top1 == gold_np, N) - _rate(~bw_np, N),
        "net_correction":              net_corr,
        "applied_count":               n_app,
        "applied_precision_ctg":       applied_prec_ctg,
        "applied_damage_rate":         applied_damage_rate,
        "benefit_damage_ratio":        bdrat,
        "gold_in_pool_rate":           _rate(pg_np,                   N),
        "selector_ce_val":             selector_ce,
        "selector_acc_val":            selector_acc,
        "n_bwcov":                     n_bwcov,
    }


def _checkpoint_score(m):
    """Conservative score: heavily penalises changed_away."""
    ctg = m.get("changed_to_gold_rate",        float("nan"))
    caw = m.get("changed_away_rate",            float("nan"))
    sip = m.get("selected_gold_given_in_pool",  float("nan"))
    nac = m.get("noop_acc_on_base_correct",     float("nan"))
    if any(math.isnan(v) for v in [ctg, caw, sip, nac]):
        return float("nan")
    return ctg - 3.0 * caw + 0.25 * sip + 0.25 * nac


# ─────────────────────────────────────────────────────────────────────────────
# Training index
# ─────────────────────────────────────────────────────────────────────────────

def build_train_index_v1c(data, tok_arr_t, reg_arr_t, unk_region, unk_super,
                            sr_enabled, args, device):
    """Returns correctable_idx, base_correct_idx, n_ignored.
    correctable = bw & pg  (base wrong, gold in pool)
    base_correct = bc      (base top1 == gold)
    ignored      = bw & ~pg (base wrong, gold not in pool)
    """
    N   = data["h_prime"].shape[0]
    BSZ = 4096
    corr_idx = []
    bc_idx   = []

    for s in range(0, N, BSZ):
        e    = min(s + BSZ, N)
        topk = data["topk_ids"][s:e].to(device)
        lgt  = data["topk_lgt"][s:e].to(device)
        gold = data["gold"][s:e].to(device)
        cand_ids, _, _, _, _ = build_candidate_pool(
            topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
            args.candidate_pool_size, args.candidate_filter, device)
        correct  = (topk[:, 0] == gold).cpu().numpy()
        pool_has = (cand_ids == gold.unsqueeze(1)).any(1).cpu().numpy()
        for i, gi in enumerate(range(s, e)):
            if correct[i]:
                bc_idx.append(gi)
            elif pool_has[i]:
                corr_idx.append(gi)
            # else: ignored (bw & ~pg)

    n_ignored = N - len(corr_idx) - len(bc_idx)
    print(f"  train index: correctable={len(corr_idx):,}  "
          f"base_correct={len(bc_idx):,}  ignored={n_ignored:,}")
    return (np.array(corr_idx, dtype=np.int64),
            np.array(bc_idx,   dtype=np.int64),
            n_ignored)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — 2D grid (gate_thr × selector_margin_thr)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_v1c(model, val_data, tok_arr_t, reg_arr_t,
                  unk_region, unk_super, sr_enabled, args, device,
                  gate_thresholds, selector_margin_thresholds):
    """Forward pass + 2D grid. Returns (grid, chk_score, raw)."""
    model.eval()
    N, K = val_data["h_prime"].shape[0], val_data["topk_ids"].shape[1]
    P    = args.candidate_pool_size
    BSZ  = args.eval_batch_size
    md   = 0.5 * args.margin_delta
    vs   = tok_arr_t.shape[0]

    gold_np          = np.zeros(N, dtype=np.int64)
    base_np          = np.zeros(N, dtype=np.int64)
    bw_np            = np.zeros(N, dtype=bool)
    pg_np            = np.zeros(N, dtype=bool)
    sr_np            = np.zeros(N, dtype=bool)
    ss_np            = np.zeros(N, dtype=bool)
    sel_scores_np    = np.zeros((N, P),             dtype=np.float32)
    edit_logit_np    = np.zeros(N,                  dtype=np.float32)
    gate_cand_idx_np = np.zeros(N,                  dtype=np.int64)
    cand_np          = np.zeros((N, P),             dtype=np.int64)
    cand_l_np        = np.zeros((N, P),             dtype=np.float32)
    topk_ids_np      = np.zeros((N, K),             dtype=np.int64)
    topk_lgts_np     = np.zeros((N, K),             dtype=np.float32)
    # TXL memory length = seq_len + num_memory_slots; token mode = memory_len
    _attn_mem_width  = (args.memory_len + args.num_memory_slots
                        if args.memory_backend == "txl" else args.memory_len)
    attn_np          = np.zeros((N, P, _attn_mem_width), dtype=np.float32)
    mem_ids_np       = np.zeros((N, args.memory_len),    dtype=np.int64)

    with torch.no_grad():
        for s in range(0, N, BSZ):
            e = min(s + BSZ, N)
            b = e - s

            hp   = val_data["h_prime"][s:e].to(device)
            topk = val_data["topk_ids"][s:e].to(device)
            lgt  = val_data["topk_lgt"][s:e].to(device)
            gold = val_data["gold"][s:e].to(device)
            iids = val_data["ids"][s:e].to(device)

            cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
                topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
                P, args.candidate_filter, device)

            base_top1_ids = topk[:, 0]
            sel_sc, edit_lg, attn_w, gc_idx = model.forward_eval(
                hp, iids, base_top1_ids, lgt,
                cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups)

            bw_b = (base_top1_ids != gold).cpu().numpy()
            pg_b = (cand_ids == gold.unsqueeze(1)).any(1).cpu().numpy()

            gold_reg = tok_arr_t[gold.clamp(0, vs - 1)]
            top1_reg = tok_arr_t[base_top1_ids.clamp(0, vs - 1)]
            sr_b = ((gold_reg == top1_reg) & (gold_reg != unk_region)).cpu().numpy()
            if sr_enabled:
                rlen   = reg_arr_t.shape[0] - 1
                gs_sup = reg_arr_t[gold_reg.clamp(0, rlen)]
                t1_sup = reg_arr_t[top1_reg.clamp(0, rlen)]
                ss_b   = ((gs_sup == t1_sup) & (gs_sup != unk_super)).cpu().numpy()
            else:
                ss_b = np.zeros(b, dtype=bool)

            sl = slice(s, e)
            gold_np[sl]          = gold.cpu().numpy()
            base_np[sl]          = base_top1_ids.cpu().numpy()
            bw_np[sl]            = bw_b
            pg_np[sl]            = pg_b
            sr_np[sl]            = sr_b
            ss_np[sl]            = ss_b
            sel_scores_np[sl]    = sel_sc.cpu().numpy()
            edit_logit_np[sl]    = edit_lg.cpu().numpy()
            gate_cand_idx_np[sl] = gc_idx.cpu().numpy()
            cand_np[sl]          = cand_ids.cpu().numpy()
            cand_l_np[sl]        = cand_lgts.cpu().numpy()
            topk_ids_np[sl]      = topk.cpu().numpy()
            topk_lgts_np[sl]     = lgt.cpu().numpy()
            M_actual             = attn_w.shape[2]
            attn_np[sl, :, :M_actual] = attn_w.cpu().numpy()
            mem_ids_np[sl]       = iids[:, -args.memory_len:].cpu().numpy()

    # Selector CE / acc on correctable rows
    correct_np = ~bw_np
    pg_cand    = ~correct_np & pg_np
    if pg_cand.any():
        cand_ids_t = torch.from_numpy(cand_np)
        gold_t     = torch.from_numpy(gold_np)
        in_pool    = (cand_ids_t == gold_t.unsqueeze(1))
        pool_idx_v = in_pool.long().argmax(1).numpy()
        sc_v       = torch.from_numpy(sel_scores_np[pg_cand])
        lb_v       = torch.from_numpy(pool_idx_v[pg_cand]).long()
        selector_ce  = F.cross_entropy(sc_v, lb_v).item()
        selector_acc = float((sc_v.argmax(1) == lb_v).float().mean())
    else:
        selector_ce = float("nan"); selector_acc = float("nan")

    # 2D grid
    grid = []
    for gt in gate_thresholds:
        for smt in selector_margin_thresholds:
            action = _apply_policy_v1c(
                sel_scores_np, edit_logit_np, gate_cand_idx_np, gt, smt)
            m = _metrics_from_action(
                    action, gold_np, bw_np, pg_np, sr_np, ss_np,
                    cand_np, topk_ids_np, topk_lgts_np, md,
                    selector_ce, selector_acc)
            m["gate_threshold"]              = gt
            m["selector_margin_threshold"]   = smt
            grid.append(m)

    # Checkpoint score at (min_gate_thr, no-margin corner)
    min_gt = min(gate_thresholds)
    argmax_row = next(
        (m for m in grid
         if m["gate_threshold"] == min_gt and m["selector_margin_threshold"] <= -500.0),
        next((m for m in grid if m["gate_threshold"] == min_gt), grid[0]))
    chk_score = _checkpoint_score(argmax_row)

    raw = {
        "gold": gold_np, "base": base_np,
        "bw":   bw_np,   "pg":   pg_np,
        "sr":   sr_np,   "ss":   ss_np,
        "sel_scores":     sel_scores_np,
        "edit_logit":     edit_logit_np,
        "gate_cand_idx":  gate_cand_idx_np,
        "cand_ids":       cand_np,
        "cand_lgts":      cand_l_np,
        "topk_ids":       topk_ids_np,
        "topk_lgts":      topk_lgts_np,
        "attn":           attn_np,
        "mem_ids":        mem_ids_np,
        "N": N, "P": P, "K": K,
        "selector_ce":    selector_ce,
        "selector_acc":   selector_acc,
    }
    model.train()
    return grid, chk_score, raw


def eval_full_vocab_from_action(model, val_data, action_arr, cand_np_,
                                  topk_ids_np_, args, device):
    """Full-vocab NLL / top-1 for a specific action array."""
    try:
        model.eval()
        N   = val_data["h_prime"].shape[0]
        VS  = model.token_emb_weight.shape[0]
        md  = 0.5 * args.margin_delta
        P   = cand_np_.shape[1]
        BSZ = args.eval_batch_size
        tot_nll_b = tot_nll_r = tot_acc_b = tot_acc_r = 0.0

        with torch.no_grad():
            for s in range(0, N, BSZ):
                e  = min(s + BSZ, N)
                b  = e - s
                ar = torch.arange(b, device=device)

                hp   = val_data["h_prime"][s:e].to(device)
                gold = val_data["gold"][s:e].to(device)
                topk = torch.from_numpy(topk_ids_np_[s:e]).to(device)

                act        = action_arr[s:e]
                apply_mask = act > 0
                pool_j     = (act - 1).clip(0, P - 1)
                sel_toks   = cand_np_[s:e][np.arange(b), pool_j]
                sel_t      = torch.from_numpy(sel_toks).to(device)
                base_t     = topk[:, 0]
                app_f      = torch.from_numpy(apply_mask.astype(np.float32)).to(device)

                fv_base = hp @ model.token_emb_weight.T         # [b, VS]
                fv_ref  = fv_base.clone()
                fv_ref[ar, sel_t.clamp(0, VS - 1)]  += md * app_f
                fv_ref[ar, base_t.clamp(0, VS - 1)] -= md * app_f

                gs = gold.clamp(0, VS - 1)
                tot_nll_b += F.cross_entropy(fv_base, gs, reduction="sum").item()
                tot_nll_r += F.cross_entropy(fv_ref,  gs, reduction="sum").item()
                tot_acc_b += (fv_base.argmax(1) == gs).sum().item()
                tot_acc_r += (fv_ref.argmax(1)  == gs).sum().item()

        model.train()
        return {
            "full_vocab_base_nll":         tot_nll_b / N,
            "full_vocab_refined_nll":      tot_nll_r / N,
            "full_vocab_gain":             (tot_nll_b - tot_nll_r) / N,
            "full_vocab_top1_acc_base":    tot_acc_b / N,
            "full_vocab_top1_acc_refined": tot_acc_r / N,
        }
    except Exception as exc:
        model.train()
        print(f"[warn] full_vocab eval failed: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def compute_baselines(raw, margin_delta):
    N   = raw["N"]; md = 0.5 * margin_delta
    bw  = raw["bw"]; pg = raw["pg"]
    gold_np  = raw["gold"]; cand_np_ = raw["cand_ids"]
    topk_ids_t  = torch.from_numpy(raw["topk_ids"])
    topk_lgts_t = torch.from_numpy(raw["topk_lgts"])
    n_bw  = int(bw.sum()); n_bc = int((~bw).sum())
    n_bwcov = int((bw & pg).sum())

    r2_toks  = cand_np_[:, 0]
    r2_apply = np.ones(N, dtype=bool)
    r2_ref   = _apply_surgical(topk_lgts_t, topk_ids_t,
                                torch.from_numpy(r2_toks.astype(np.int64)),
                                torch.from_numpy(r2_apply), md)
    r2_top1  = raw["topk_ids"][np.arange(N), r2_ref.argmax(1).numpy()]
    r2_sg    = r2_toks == gold_np

    oracle_ctg = 0
    for i in range(N):
        if bw[i] and pg[i]:
            for j in range(cand_np_.shape[1]):
                if cand_np_[i, j] == gold_np[i]:
                    rl  = topk_lgts_t[i].clone()
                    sp  = (topk_ids_t[i] == int(cand_np_[i, j])).long().argmax().item()
                    if topk_ids_t[i, sp] == int(cand_np_[i, j]):
                        rl[sp] += md; rl[0] -= md
                    if topk_ids_t[i, rl.argmax()].item() == gold_np[i]:
                        oracle_ctg += 1
                    break

    return {
        "base_top1_acc":             _rate(~bw,  N),
        "gold_in_pool_rate":         _rate(pg,   N),
        "baseline_rank2_sg_bwcov":   _rate(r2_sg[bw & pg], n_bwcov),
        "baseline_rank2_ctg":        _rate(r2_top1[bw]  == gold_np[bw],   n_bw),
        "baseline_rank2_caw":        _rate(r2_top1[~bw] != gold_np[~bw],  n_bc),
        "baseline_oracle_pool_ctg":  (_rate(oracle_ctg, n_bwcov) if n_bwcov > 0
                                       else float("nan")),
        "stage1_ref_sel_gold_bwcov": 0.246,
        "stage1_ref_ctg":            0.040,
        "stage1_ref_caw":            0.062,
        "stage2_ref_ctg_lo":         0.054,
        "stage2_ref_ctg_hi":         0.063,
        "stage2_ref_caw_lo":         0.079,
        "stage2_ref_caw_hi":         0.094,
        "stage2_ref_fv_gain":        -0.0173,
        "stage2b_ref_fv_gain":       None,      # fill after Stage 2B completes
    }


# ─────────────────────────────────────────────────────────────────────────────
# Example reports
# ─────────────────────────────────────────────────────────────────────────────

def write_example_reports_v1c(val_data, raw, tok_arr_t, unk_region,
                               tokenizer, args, run_dir, best_gt, best_smt):
    rng      = np.random.default_rng(42)
    N, P     = raw["N"], raw["P"]
    bw       = raw["bw"];  gold_np = raw["gold"]
    cand_np_ = raw["cand_ids"];  cand_l  = raw["cand_lgts"]
    sel_sc   = raw["sel_scores"]; edit_lg = raw["edit_logit"]
    gc_idx   = raw["gate_cand_idx"]
    attn     = raw["attn"];  mem_ids = raw["mem_ids"]
    vs       = tok_arr_t.shape[0]

    def reg(tid):
        return str(int(tok_arr_t[min(int(tid), vs - 1)].item()))

    def d(tid):
        try:   return f"`{tokenizer.decode([int(tid)])}`"
        except: return str(tid)

    gate_prob = 1.0 / (1.0 + np.exp(-edit_lg.clip(-50, 50)))
    action    = _apply_policy_v1c(sel_sc, edit_lg, gc_idx, best_gt, best_smt)
    apply     = action > 0
    pool_j    = (action - 1).clip(0, P - 1)
    sel_toks  = cand_np_[np.arange(N), pool_j]
    sel_gold  = apply & (sel_toks == gold_np)

    topk_ids_t  = torch.from_numpy(raw["topk_ids"])
    topk_lgts_t = torch.from_numpy(raw["topk_lgts"])
    ref_lgts = _apply_surgical(topk_lgts_t, topk_ids_t,
                                torch.from_numpy(sel_toks.astype(np.int64)),
                                torch.from_numpy(apply), 0.5 * args.margin_delta)
    ref_top1 = raw["topk_ids"][np.arange(N), ref_lgts.argmax(1).numpy()]
    ctg = bw & apply & (ref_top1 == gold_np)
    caw = ~bw & apply & (ref_top1 != gold_np)

    def fmt_row(ri):
        lines = [f"### Row {ri}\n"]
        ctx_ids = mem_ids[ri]
        try:
            ctx = tokenizer.decode(ctx_ids[-64:].tolist(), skip_special_tokens=False)
            lines.append(f"**Context (last 64):** `{ctx}`\n")
        except Exception:
            pass
        gt_tok = int(gold_np[ri]); bt = int(raw["base"][ri])
        act_ri = int(action[ri])
        gp     = gate_prob[ri]
        gc_j   = int(gc_idx[ri])
        lines.append(f"**Gold:** {d(gt_tok)} id={gt_tok} region={reg(gt_tok)}")
        lines.append(f"**Base top-1:** {d(bt)} id={bt} region={reg(bt)}")
        lines.append(f"**Gate prob:** {gp:.3f}  gate_thr={best_gt}  "
                     f"gate_cand_idx={gc_j}")
        if act_ri == 0:
            lines.append(f"**Action:** NO_OP\n")
        else:
            st  = int(sel_toks[ri])
            smi = sel_sc[ri, gc_j]
            sc_copy = sel_sc[ri].copy(); sc_copy[gc_j] = -np.inf
            smt_v = smi - sc_copy.max() if sc_copy.max() > -np.inf else float("nan")
            lines.append(f"**Action:** gate_cand={gc_j} = {d(st)} id={st}  "
                         f"gate_prob={gp:.3f}  sel_score={smi:.3f}  "
                         f"sel_margin={smt_v:.3f}\n")

        lines.append("| j | Token | ID | Region | Base lgt | Sel score | is_gate_cand | is_gold |")
        lines.append("|---|-------|----|--------|----------|-----------|--------------|---------|")
        for j in range(min(P, 10)):
            cid = int(cand_np_[ri, j])
            try:    ts = tokenizer.decode([cid])
            except: ts = str(cid)
            ig   = "✓" if cid == gt_tok else ""
            igc  = "●" if j == gc_j else ""
            lines.append(f"| {j} | `{ts}` | {cid} | {reg(cid)} | "
                         f"{cand_l[ri,j]:.3f} | {sel_sc[ri,j]:.3f} | {igc} | {ig} |")
        lines.append("")

        if act_ri > 0:
            aw  = attn[ri, gc_j]
            tk  = min(5, len(aw))
            top_i = np.argsort(aw)[::-1][:tk]
            lines.append("**Top attended memory tokens (gate candidate):**")
            lines.append("| Pos | Token | ID | Attn weight |")
            lines.append("|-----|-------|----|-------------|")
            mem_width = mem_ids.shape[1]
            for idx in top_i:
                if idx < mem_width:
                    mid = int(mem_ids[ri, idx])
                    try:    ms = tokenizer.decode([mid])
                    except: ms = str(mid)
                    lines.append(f"| {idx} | `{ms}` | {mid} | {aw[idx]:.4f} |")
                else:
                    slot_id = idx - mem_width
                    lines.append(f"| {idx} | `<MEM_SLOT_{slot_id}>` | -1 | {aw[idx]:.4f} |")
            lines.append("")
        return "\n".join(lines) + "\n"

    buckets = {
        "selected_gold":     np.where(sel_gold)[0],
        "selected_wrong":    np.where(apply & ~(sel_toks == gold_np) & bw)[0],
        "noop_correct":      np.where((~apply) & (~bw))[0],
        "false_apply":       np.where(apply & ~bw)[0],
        "changed_to_gold":   np.where(ctg)[0],
        "changed_away":      np.where(caw)[0],
        "attention_debug":   np.where(apply)[0],
        "high_conf_wrong":   np.where(apply & ~(sel_toks == gold_np) & (gate_prob > 0.7))[0],
    }
    for bname, indices in buckets.items():
        n_max = min(args.num_examples, 10) if bname == "attention_debug" else args.num_examples
        if len(indices) > n_max:
            indices = rng.choice(indices, n_max, replace=False)
        path = os.path.join(run_dir, f"examples_{bname}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# {bname.replace('_',' ').title()}\n\n"
                     f"_{len(indices)} examples_  "
                     f"policy=(gt={best_gt},smt={best_smt})\n\n---\n\n")
            for ri in indices:
                fh.write(fmt_row(int(ri)))
                fh.write("---\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# Report — 8 questions
# ─────────────────────────────────────────────────────────────────────────────

def write_report_v1c(run_dir, args, baselines, best_m_conservative,
                      best_m_net, argmax_m, best_fv, best_step):
    def f(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return "nan"
        if isinstance(v, float): return f"{v:.4f}"
        return str(v) if v is not None else "nan"

    m   = best_m_conservative
    ctg = m.get("changed_to_gold_rate",        float("nan"))
    caw = m.get("changed_away_rate",            float("nan"))
    nca = m.get("noharm_changed_away",          float("nan"))
    nac = m.get("noop_acc_on_base_correct",     float("nan"))
    sip = m.get("selected_gold_given_in_pool",  float("nan"))
    net = m.get("net_correction",               float("nan"))
    apt = m.get("applied_precision_ctg",        float("nan"))
    bdr = m.get("benefit_damage_ratio",         float("nan"))
    fvg = best_fv.get("full_vocab_gain", float("nan")) if best_fv else float("nan")
    ar_ctg = argmax_m.get("changed_to_gold_rate", float("nan"))
    ar_caw = argmax_m.get("changed_away_rate",     float("nan"))
    ar_app = argmax_m.get("apply_rate",            float("nan"))
    ar_nac = argmax_m.get("noop_acc_on_base_correct", float("nan"))
    ar_sip = argmax_m.get("selected_gold_given_in_pool", float("nan"))

    s2b_caw = baselines.get("stage2_ref_caw_hi", 0.094)
    s2_fvg  = baselines.get("stage2_ref_fv_gain", -0.0173)
    net_max = best_m_net.get("net_correction", float("nan"))
    gip_v   = argmax_m.get("gold_in_pool_rate", float("nan"))

    lines = ["# TXL Detail Resolver V1 — Transformer-XL-Style Detail Memory\n"]
    lines.append(f"**Run:** `{args.run_name}`  |  **Steps:** {args.steps}  "
                 f"|  **memory_len:** {args.memory_len}  "
                 f"|  **correctable_fraction:** {args.correctable_fraction}  "
                 f"|  **gate_pos_weight:** {args.gate_pos_weight}\n")
    lines.append(f"**Memory:** backend=`{args.memory_backend}`  "
                 f"mem_dim={args.mem_dim}  txl_layers={args.txl_layers}  "
                 f"txl_heads={args.txl_heads}  txl_ff_dim={args.txl_ff_dim}  "
                 f"num_memory_slots={args.num_memory_slots}  "
                 f"pos_encoding=`{args.pos_encoding}`  "
                 f"enable_shard_recurrence={args.enable_shard_recurrence}  "
                 f"lambda_memory_contrastive={args.lambda_memory_contrastive}\n")

    lines.append("## Reference\n")
    lines.append("| Experiment | ctg | caw | fv_gain |")
    lines.append("|-----------|-----|-----|---------|")
    lines.append("| Stage 1 (frozen h_prime) | ~0.040 | ~0.062 | ~-0.0199 |")
    lines.append("| Stage 2 (local token memory) | 0.054–0.063 | 0.079–0.094 | ~-0.0173 |")
    lines.append("| Stage 2B (noharm calibration) | — | — | pending |")
    lines.append("| V3 oracle | 0.2407 | 0.000 | +0.1584 |\n")

    lines.append("## Baselines\n")
    for k, v in baselines.items():
        if v is not None:
            lines.append(f"  {k:48s} = {f(v)}")
    lines.append("")

    lines.append("## Argmax Metrics (gt=min, smt=no-margin)\n")
    for k, v in argmax_m.items():
        lines.append(f"  {k:48s} = {f(v)}")
    lines.append("")

    lines.append("## Best-Conservative Metrics "
                 f"(gt={m.get('gate_threshold',0)}, "
                 f"smt={m.get('selector_margin_threshold',0)})\n")
    for k, v in best_m_conservative.items():
        lines.append(f"  {k:48s} = {f(v)}")
    lines.append("")

    lines.append("## Best-Net-Correction Metrics "
                 f"(gt={best_m_net.get('gate_threshold',0)}, "
                 f"smt={best_m_net.get('selector_margin_threshold',0)})\n")
    for k, v in best_m_net.items():
        lines.append(f"  {k:48s} = {f(v)}")
    lines.append("")

    if best_fv:
        lines.append("## Full-Vocab\n")
        for k, v in best_fv.items():
            lines.append(f"  {k:48s} = {f(v)}")
        lines.append("")

    lines.append(f"Best checkpoint: step={best_step}\n")
    lines.append("## Analysis (8 Questions)\n")

    def yn(c): return "✅" if c else "⚠️"

    # Q1: Did decoupling fix NO_OP collapse?
    q1 = not math.isnan(ar_app) and ar_app > 0.05
    lines.append(f"**1. Did decoupling fix the NO_OP collapse?**  {yn(q1)}")
    lines.append(f"   argmax apply_rate={f(ar_app)}  "
                 f"(Stage 2B collapsed to ~0.0017; target ≥ 0.05)\n")

    # Q2: Did gate suppress base-correct edits?
    q2 = not math.isnan(ar_nac) and ar_nac > 0.80
    lines.append(f"**2. Did the gate learn to suppress edits on base-correct rows?**  {yn(q2)}")
    lines.append(f"   noop_acc_on_base_correct={f(ar_nac)}  "
                 f"(target ≥ 0.80; Stage 2 was low)\n")

    # Q3: Did selector learn gold?
    q3 = not math.isnan(ar_sip) and ar_sip > 0.30
    lines.append(f"**3. Did the selector learn to identify gold when in pool?**  {yn(q3)}")
    lines.append(f"   selected_gold_given_in_pool={f(ar_sip)}  (target ≥ 0.30)\n")

    # Q4: ctg > caw at any policy?
    q4 = not math.isnan(net_max) and net_max > 0
    lines.append(f"**4. Does ctg > caw at any policy (net_correction > 0)?**  {yn(q4)}")
    lines.append(f"   best net_correction={f(net_max)}  "
                 f"(gt={best_m_net.get('gate_threshold','?')}, "
                 f"smt={best_m_net.get('selector_margin_threshold','?')})\n")

    # Q5: fv_gain improved vs Stage 2?
    q5a = not math.isnan(fvg) and fvg >= 0
    q5b = not math.isnan(fvg) and not math.isnan(s2_fvg) and fvg > s2_fvg
    lines.append(f"**5. Did full_vocab_gain improve vs Stage 2?**  "
                 f"{yn(q5a)} (≥0)  {yn(q5b)} (>Stage2 {f(s2_fvg)})")
    lines.append(f"   fv_gain={f(fvg)}\n")

    # Q6: Best tradeoff
    lines.append(f"**6. Best tradeoff policy:**  "
                 f"gt={m.get('gate_threshold',0)}  smt={m.get('selector_margin_threshold',0)}")
    lines.append(f"   ctg={f(ctg)}  caw={f(caw)}  apply_rate={f(m.get('apply_rate',float('nan')))}  "
                 f"precision={f(apt)}  benefit_damage_ratio={f(bdr)}\n")

    # Q7: If still negative, bottleneck
    lines.append(f"**7. Primary bottleneck (if net_correction ≤ 0):**")
    if q4:
        lines.append("   → None: Stage 2C succeeded. Consider hidden-state memory for further gains.")
    else:
        lines.append(f"   gold_in_pool_rate          = {f(gip_v)}  (pool coverage; target ~0.25+)")
        lines.append(f"   selected_gold_given_in_pool = {f(ar_sip)}  (selector quality; target ~0.40+)")
        lines.append(f"   noop_acc_on_base_correct    = {f(ar_nac)}  (gate calibration; target ~0.90+)")
        lines.append(f"   changed_to_gold_rate        = {f(ar_ctg)}  (actual corrections)")
        if not math.isnan(gip_v) and gip_v < 0.15:
            lines.append("   → PRIMARY: candidate pool coverage too low. Gold rarely in top-32.")
        elif not math.isnan(ar_sip) and ar_sip < 0.30:
            lines.append("   → PRIMARY: selector weak. Gold in pool but not picked. "
                         "Consider hidden-state memory.")
        elif not math.isnan(ar_nac) and ar_nac < 0.80:
            lines.append("   → PRIMARY: gate not calibrated. Increase gate_pos_weight or "
                         "lambda_noharm_gate.")
        else:
            lines.append("   → UNCLEAR: all metrics borderline. May need stronger memory signal.")
    lines.append("")

    # Q8: Stage comparison
    lines.append("**8. Stage comparison:**\n")
    lines.append("| Metric | Stage 1 | Stage 2 | Stage 2B | Stage 3 TXL (this) |")
    lines.append("|--------|---------|---------|----------|-----------------|")
    lines.append(f"| ctg   | 0.040   | 0.054–0.063 | ~0.000 | {f(ar_ctg)} |")
    lines.append(f"| caw   | 0.062   | 0.079–0.094 | ~0.000 | {f(ar_caw)} |")
    lines.append(f"| fv_gain | -0.0199 | -0.0173 | ~0.000 | {f(fvg)} |")
    lines.append(f"| apply_rate | — | high | ~0.0017 | {f(ar_app)} |")
    lines.append(f"| noop_acc_bc | — | — | — | {f(ar_nac)} |")
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

    # Compatibility / safety guards for Stage 3 TXL runs.
    if not (0.0 <= args.correctable_fraction <= 1.0):
        raise ValueError(f"correctable_fraction must be in [0,1], got {args.correctable_fraction}")
    if args.memory_backend == "txl" and args.mem_dim != args.resolver_dim:
        print(f"[info] mem_dim ({args.mem_dim}) != resolver_dim ({args.resolver_dim}); "
              "using txl_to_resolver projection.")
    print(f"[config] memory_backend={args.memory_backend}  "
          f"txl_layers={args.txl_layers}  txl_heads={args.txl_heads}  "
          f"mem_dim={args.mem_dim}  num_memory_slots={args.num_memory_slots}")

    run_dir = os.path.join(args.output_root, args.run_name)
    if os.path.exists(run_dir):
        raise RuntimeError(
            f"Run dir exists: {run_dir}. Use --run_name to avoid overwriting.")
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

    print("\n[model] Building LocalDetailSelectorGate...")
    n_regions = unk_region; n_supers = unk_super if sr_enabled else 1
    print(f"  memory_backend={args.memory_backend}  "
          f"mem_dim={args.mem_dim}  txl_layers={args.txl_layers}  "
          f"pos_encoding={args.pos_encoding}  "
          f"num_memory_slots={args.num_memory_slots}")
    model = LocalDetailSelectorGate(
        token_emb_weight=tok_w.to(device),
        tok_arr=tok_arr_np, reg_arr=reg_arr_np,
        d_model=d_model, n_regions=n_regions, n_supers=n_supers,
        sr_enabled=sr_enabled, unk_region=unk_region, unk_super=unk_super,
        top_k=args.top_k, pool_size=args.candidate_pool_size,
        memory_len=args.memory_len, resolver_dim=args.resolver_dim,
        hidden_dim=args.hidden_dim, attention_heads=args.attention_heads,
        dropout=args.dropout, region_emb_dim=args.region_emb_dim,
        super_emb_dim=args.super_emb_dim,
        memory_backend=args.memory_backend,
        mem_dim=args.mem_dim, txl_layers=args.txl_layers,
        txl_heads=args.txl_heads, txl_ff_dim=args.txl_ff_dim,
        num_memory_slots=args.num_memory_slots,
        pos_encoding=args.pos_encoding,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_txl_params = (sum(p.numel() for p in model.txl_memory.parameters()
                        if p.requires_grad)
                    if model.txl_memory is not None else 0)
    print(f"  Trainable params: {n_params:,}  "
          f"(txl_memory: {n_txl_params:,}  other: {n_params - n_txl_params:,}"
          f"  backend={args.memory_backend})")

    print("\n[train index] Building training row indices...")
    correctable_idx, base_correct_idx, n_ignored = build_train_index_v1c(
        train_data, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled, args, device)
    if len(correctable_idx) == 0:
        raise RuntimeError("No correctable rows found in training data.")

    gate_thresholds            = [float(x) for x in args.gate_thresholds.split(",")]
    selector_margin_thresholds = [float(x) for x in args.selector_margin_thresholds.split(",")]
    min_gt = min(gate_thresholds)

    # Pretrain baseline
    print("\n[pretrain baseline]")
    _grid0, _, row0 = evaluate_v1c(
        model, val_data, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled, args, device,
        [min_gt], selector_margin_thresholds[:1])
    baselines = compute_baselines(row0, args.margin_delta)
    N_train   = train_data["h_prime"].shape[0]
    total_lbl = len(correctable_idx) + len(base_correct_idx) + n_ignored
    baselines["train_correctable_rate"]  = len(correctable_idx)  / max(total_lbl, 1)
    baselines["train_base_correct_rate"] = len(base_correct_idx) / max(total_lbl, 1)
    baselines["train_ignored_rate"]      = n_ignored              / max(total_lbl, 1)
    baselines["val_gold_in_pool_rate"]   = baselines.pop("gold_in_pool_rate", float("nan"))
    baselines["val_base_correct_rate"]   = baselines.pop("base_top1_acc",     float("nan"))
    with open(os.path.join(run_dir, "pretrain_baseline.json"), "w") as fp:
        json.dump(_json_safe(baselines), fp, indent=2)
    for k, v in baselines.items():
        if v is not None:
            print(f"  {k:48s} = "
                  f"{v:.4f}" if isinstance(v, float) and not math.isnan(v)
                  else f"  {k:48s} = {v}")

    config = vars(args).copy()
    config.update({"d_model": d_model, "vocab_size": vocab_size,
                   "sr_enabled": sr_enabled, "n_params": n_params,
                   "n_txl_params": n_txl_params})
    with open(os.path.join(run_dir, "config.json"), "w") as fp:
        json.dump(config, fp, indent=2)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scaler = (torch.amp.GradScaler("cuda")
              if (args.amp and torch.cuda.is_available()) else None)

    corr_bsz = max(1, round(args.batch_size * args.correctable_fraction))
    bc_bsz   = args.batch_size - corr_bsz

    N_corr = len(correctable_idx); N_bc = len(base_correct_idx)
    perm_corr = np.random.permutation(N_corr); pos_corr = 0
    perm_bc   = np.random.permutation(N_bc) if N_bc > 0 else None; pos_bc = 0

    train_log = os.path.join(run_dir, "train_log.csv")
    eval_log  = os.path.join(run_dir, "eval_log.csv")
    grid_log  = os.path.join(run_dir, "grid_eval_log.csv")
    hdr_t = hdr_e = hdr_g = False

    best_score = -float("inf"); best_step = 0
    step = 0; t0 = time.time()
    optimizer.zero_grad()

    print(f"\n[train] steps={args.steps}  batch={args.batch_size} "
          f"(corr={corr_bsz} bc={bc_bsz})  lr={args.lr}  "
          f"gate_pos_weight={args.gate_pos_weight}  "
          f"lambda_noharm_gate={args.lambda_noharm_gate}\n")

    _accum = 0
    while step < args.steps:
        # ── Sample correctable rows ──────────────────────────────────────────
        if pos_corr + corr_bsz > N_corr:
            perm_corr = np.random.permutation(N_corr); pos_corr = 0
        c_global = correctable_idx[perm_corr[pos_corr:pos_corr + corr_bsz]]
        pos_corr += corr_bsz

        # ── Sample base_correct rows ─────────────────────────────────────────
        if bc_bsz > 0 and N_bc > 0:
            if pos_bc + bc_bsz > N_bc:
                perm_bc = np.random.permutation(N_bc); pos_bc = 0
            bc_global = base_correct_idx[perm_bc[pos_bc:pos_bc + bc_bsz]]
            pos_bc += bc_bsz
            batch_global = np.concatenate([c_global, bc_global])
        else:
            batch_global = c_global
        B = len(batch_global)

        hp   = train_data["h_prime"][batch_global].to(device)
        topk = train_data["topk_ids"][batch_global].to(device)
        lgt  = train_data["topk_lgt"][batch_global].to(device)
        gold = train_data["gold"][batch_global].to(device)
        iids = train_data["ids"][batch_global].to(device)

        cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
            topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
            args.candidate_pool_size, args.candidate_filter, device)

        base_top1     = topk[:, 0]
        is_base_correct = (base_top1 == gold)
        in_pool       = (cand_ids == gold.unsqueeze(1))
        pool_has      = in_pool.any(1)
        is_correctable = ~is_base_correct & pool_has
        gold_pool_idx = in_pool.long().argmax(1)          # [B] valid when pool_has

        with _amp_ctx(args.amp):
            sel_scores, cand_repr, _ = model.forward_shared(
                hp, iids, base_top1, lgt,
                cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups)

            # ── Teacher forcing gate_cand_idx ────────────────────────────────
            gate_cand_idx = torch.zeros(B, dtype=torch.long, device=device)
            if is_correctable.any():
                gate_cand_idx[is_correctable] = gold_pool_idx[is_correctable]
            if is_base_correct.any():
                # Detach: prevent gate's hard-negative from pushing selector
                gate_cand_idx[is_base_correct] = (
                    sel_scores.detach()[is_base_correct].argmax(1))

            # ── Gate labels (surgery simulation for correctable rows) ─────────
            gate_labels = compute_gate_labels(
                gold, topk, lgt, cand_ids, gate_cand_idx,
                is_correctable, 0.5 * args.margin_delta, device)

            # ── Gate forward ─────────────────────────────────────────────────
            edit_logit = model.gate_forward(
                cand_repr, gate_cand_idx, hp, lgt,
                cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups,
                sel_scores, base_top1)

            total_loss, ld = compute_loss_v1c(
                sel_scores, gold_pool_idx, edit_logit, gate_labels,
                is_correctable, is_base_correct,
                args.lambda_selector, args.lambda_gate, args.gate_pos_weight,
                args.lambda_noharm_gate, args.gate_margin,
                args.lambda_selector_margin, args.selector_margin,
                device)

            loss_sc = total_loss / args.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss_sc).backward()
        else:
            loss_sc.backward()

        _accum += 1
        if _accum < args.grad_accum_steps:
            continue
        _accum = 0

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
            print(f"  step={step:5d}  total={ld['total']:.4f}  "
                  f"l_sel={ld['l_select']:.4f}  l_gate={ld['l_gate']:.4f}  "
                  f"l_noharm={ld['l_noharm']:.4f}  l_smgn={ld['l_sel_margin']:.4f}  "
                  f"n_corr={ld['n_corr']}  n_bc={ld['n_bc']}  "
                  f"t={time.time()-t0:.0f}s")
        _write_csv_row(train_log, {"step": step, **ld}, header=not hdr_t); hdr_t = True

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n[eval] step={step}")
            grid, chk_score, val_row = evaluate_v1c(
                model, val_data, tok_arr_t, reg_arr_t,
                unk_region, unk_super, sr_enabled, args, device,
                gate_thresholds, selector_margin_thresholds)

            argmax_m = next(
                (m for m in grid
                 if m["gate_threshold"] == min_gt
                 and m["selector_margin_threshold"] <= -500.0),
                next((m for m in grid if m["gate_threshold"] == min_gt), grid[0]))

            ctg_v = argmax_m.get("changed_to_gold_rate", float("nan"))
            caw_v = argmax_m.get("changed_away_rate",     float("nan"))
            nac_v = argmax_m.get("noop_acc_on_base_correct", float("nan"))
            net_v = argmax_m.get("net_correction",         float("nan"))
            print(f"  [argmax] ctg={ctg_v:.4f}  caw={caw_v:.4f}  "
                  f"nac={nac_v:.4f}  net={net_v:.4f}  score="
                  f"{'nan' if math.isnan(chk_score) else f'{chk_score:.4f}'}")

            torch.save({"step": step, "model": model.state_dict(), "args": vars(args)},
                       os.path.join(run_dir, "latest_txl_detail_resolver.pt"))

            if not math.isnan(chk_score) and chk_score > best_score:
                best_score = chk_score; best_step = step
                torch.save({"step": step, "model": model.state_dict(), "args": vars(args)},
                           os.path.join(run_dir, "best_txl_detail_resolver.pt"))
                with open(os.path.join(run_dir, "best_metrics.json"), "w") as fp:
                    json.dump(_json_safe({"step": step, "score": chk_score,
                                         "argmax_metrics": argmax_m}), fp, indent=2)
                print(f"  [best] step={step}  score={chk_score:.4f}")

            _write_csv_row(eval_log, {"step": step, **argmax_m}, header=not hdr_e)
            hdr_e = True
            for gm in grid:
                _write_csv_row(grid_log, {"step": step, **gm}, header=not hdr_g)
                hdr_g = True
            print()

    # ── Final eval with best checkpoint ──────────────────────────────────────
    best_path = os.path.join(run_dir, "best_txl_detail_resolver.pt")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        loaded_step = int(ck.get("step", -1))
        print(f"[final] Loaded best_txl_detail_resolver.pt  step={loaded_step}")
    else:
        loaded_step = -1
        print("[final] WARNING: best_txl_detail_resolver.pt not found; using latest weights")

    print("[final] Full grid evaluation on val...")
    final_grid, final_score, final_row = evaluate_v1c(
        model, val_data, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled, args, device,
        gate_thresholds, selector_margin_thresholds)

    # Write threshold_grid.csv
    grid_csv = os.path.join(run_dir, "threshold_grid.csv")
    hdr = True
    for gm in final_grid:
        _write_csv_row(grid_csv, gm, header=hdr); hdr = False

    # Key policies
    argmax_final = next(
        (m for m in final_grid
         if m["gate_threshold"] == min_gt
         and m["selector_margin_threshold"] <= -500.0),
        next((m for m in final_grid if m["gate_threshold"] == min_gt), final_grid[0]))

    def _cscore(m):
        s = _checkpoint_score(m)
        return s if not math.isnan(s) else -1e9

    best_conservative = max(final_grid, key=_cscore)
    best_net = max(final_grid,
                   key=lambda m: (m.get("net_correction", float("-inf"))
                                  if not math.isnan(m.get("net_correction", float("nan")))
                                  else float("-inf")))

    # Full-vocab eval for key policies
    best_fv = {}
    if args.eval_full_vocab:
        print("[final] Full-vocab eval for key policies...")
        seen = set()
        for pm in [argmax_final, best_conservative, best_net]:
            key = (pm["gate_threshold"], pm["selector_margin_threshold"])
            if key in seen: continue
            seen.add(key)
            label = f"gt{pm['gate_threshold']}_smt{pm['selector_margin_threshold']}"
            action_arr = _apply_policy_v1c(
                final_row["sel_scores"], final_row["edit_logit"],
                final_row["gate_cand_idx"],
                pm["gate_threshold"], pm["selector_margin_threshold"])
            fv = eval_full_vocab_from_action(
                model, val_data, action_arr,
                final_row["cand_ids"], final_row["topk_ids"], args, device)
            fv["gate_threshold"]            = pm["gate_threshold"]
            fv["selector_margin_threshold"] = pm["selector_margin_threshold"]
            print(f"  [{label}] fv_gain={fv.get('full_vocab_gain', float('nan')):.4f}")
            if key == (argmax_final["gate_threshold"], argmax_final["selector_margin_threshold"]):
                best_fv = fv
            for gm in final_grid:
                if (gm["gate_threshold"] == pm["gate_threshold"]
                        and gm["selector_margin_threshold"] == pm["selector_margin_threshold"]):
                    gm["full_vocab_gain"] = fv.get("full_vocab_gain", float("nan"))

    final_metrics = {
        "step":                            args.steps,
        "best_step_during_training":       best_step,
        "loaded_best_step_for_final_eval": loaded_step,
        "final_eval_uses_best_checkpoint": loaded_step >= 0,
        "argmax_metrics":      _json_safe(argmax_final),
        "best_conservative":   _json_safe(best_conservative),
        "best_net_correction": _json_safe(best_net),
        "full_vocab":          _json_safe(best_fv),
        "baselines":           _json_safe(baselines),
    }
    with open(os.path.join(run_dir, "final_metrics.json"), "w") as fp:
        json.dump(final_metrics, fp, indent=2)

    # Example reports
    try:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            tokenizer = type("FT", (), {"decode": lambda s, ids, **kw: str(ids)})()
        write_example_reports_v1c(
            val_data, final_row, tok_arr_t.cpu(), unk_region,
            tokenizer, args, run_dir,
            best_conservative["gate_threshold"],
            best_conservative["selector_margin_threshold"])
    except Exception as exc:
        print(f"[warn] example reports failed: {exc}")

    write_report_v1c(run_dir, args, baselines,
                      best_conservative, best_net, argmax_final,
                      best_fv, loaded_step)

    print(f"\n{'='*60}")
    print(f" TXL Detail Resolver V1 complete.")
    print(f" Run dir:   {run_dir}")
    print(f" Best step: {best_step}  training_score={best_score:.4f}")
    ctg_f = argmax_final.get("changed_to_gold_rate", float("nan"))
    caw_f = argmax_final.get("changed_away_rate",     float("nan"))
    net_f = argmax_final.get("net_correction",        float("nan"))
    print(f" Argmax:    ctg={ctg_f:.4f}  caw={caw_f:.4f}  net={net_f:.4f}")
    if best_fv:
        print(f" FV gain:   {best_fv.get('full_vocab_gain', float('nan')):.4f}")
    print(f"{'='*60}\n")


def _parse():
    p = argparse.ArgumentParser(
        description="TXL Detail Resolver V1 — Decoupled Selector + Gate")
    p.add_argument("--small_ckpt",          required=True)
    p.add_argument("--train_dir",           required=True)
    p.add_argument("--val_dir",             required=True)
    p.add_argument("--token_to_region",     required=True)
    p.add_argument("--super_map",           default=None)
    p.add_argument("--output_root",         default="runs/txl_detail_resolver_v1")
    p.add_argument("--run_name",            default="txlmem_selector_gate_v1")
    p.add_argument("--top_k",               type=int,   default=256)
    p.add_argument("--candidate_pool_size", type=int,   default=32)
    p.add_argument("--memory_len",          type=int,   default=128)
    p.add_argument("--candidate_filter",    default="top_rank",
                   choices=["top_rank", "same_region_or_superregion",
                            "same_region_only", "same_superregion_only"])
    p.add_argument("--uncovered_policy",    default="ignore",
                   choices=["ignore", "noop"])
    p.add_argument("--resolver_dim",        type=int,   default=256)
    p.add_argument("--hidden_dim",          type=int,   default=512)
    p.add_argument("--attention_heads",     type=int,   default=4)
    p.add_argument("--dropout",             type=float, default=0.1)
    p.add_argument("--region_emb_dim",      type=int,   default=64)
    p.add_argument("--super_emb_dim",       type=int,   default=32)
    p.add_argument("--batch_size",          type=int,   default=128)
    p.add_argument("--correctable_fraction", "--candidate_fraction", dest="correctable_fraction",
                   type=float, default=0.5,
                   help="Fraction of each batch drawn from correctable rows; --candidate_fraction is accepted as an alias.")
    p.add_argument("--grad_accum_steps",    type=int,   default=1)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--steps",               type=int,   default=5000)
    p.add_argument("--eval_every",          type=int,   default=500)
    p.add_argument("--eval_batch_size",     type=int,   default=256)
    p.add_argument("--grad_clip",           type=float, default=1.0)
    # V1C loss weights
    p.add_argument("--lambda_selector", "--lambda_select", dest="lambda_selector",
                   type=float, default=1.0)
    p.add_argument("--lambda_gate",             type=float, default=1.0)
    p.add_argument("--gate_pos_weight",         type=float, default=3.0)
    p.add_argument("--lambda_noharm_gate", "--lambda_noharm_gate_margin",
                   dest="lambda_noharm_gate", type=float, default=0.5)
    p.add_argument("--gate_margin",             type=float, default=1.0)
    p.add_argument("--lambda_selector_margin",  type=float, default=0.25)
    p.add_argument("--selector_margin",         type=float, default=1.0)
    p.add_argument("--margin_delta",            type=float, default=1.0)
    # V1C eval grid
    p.add_argument("--gate_thresholds",
                   default="0.3,0.4,0.5,0.6,0.7,0.8")
    p.add_argument("--selector_margin_thresholds",
                   default="-999,0.0,0.5,1.0,1.5,2.0")
    p.add_argument("--eval_full_vocab",     action="store_true")
    p.add_argument("--amp",                 action="store_true")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--max_train_rows",      type=int,   default=None)
    p.add_argument("--max_val_rows",        type=int,   default=None)
    p.add_argument("--num_examples",        type=int,   default=40)
    p.add_argument("--seq_len",             type=int, default=128,
                   help="Accepted for Slurm compatibility; input_ids determine actual seq length.")
    # ── TXL / memory backend args ────────────────────────────────────────────
    p.add_argument("--memory_backend",      default="txl",
                   choices=["token", "txl"],
                   help="'token'=V1C-identical raw emb; 'txl'=TXL contextual memory")
    p.add_argument("--mem_dim",             type=int,   default=256)
    p.add_argument("--txl_layers",          type=int,   default=2)
    p.add_argument("--txl_heads",           type=int,   default=4)
    p.add_argument("--txl_ff_dim",          type=int,   default=1024)
    p.add_argument("--txl_mem_len",         type=int,   default=64)
    p.add_argument("--num_memory_slots",    type=int,   default=16)
    p.add_argument("--pos_encoding",        default="alibi",
                   choices=["alibi", "learned_relative", "absolute"])
    p.add_argument("--enable_shard_recurrence", action="store_true",
                   help="Reserved for cross-row recurrence; NOT active in Patch A.")
    p.add_argument("--lambda_memory_contrastive", type=float, default=0.0,
                   help="Disabled by default; reserved for contrastive memory aux loss.")
    return p.parse_args()


if __name__ == "__main__":
    main()
