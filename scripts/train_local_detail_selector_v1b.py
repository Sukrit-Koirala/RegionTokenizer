#!/usr/bin/env python3
"""
train_local_detail_selector_v1b.py — Local Detail Selector V1B
Stage 2B: Conservative NO_OP / No-Harm Calibration.

Stage 1:  sel_gold_bwcov≈0.246, ctg≈0.040, caw≈0.062, fv_gain≈-0.0199
Stage 2:  sel_gold_bwcov≈0.247–0.250, ctg≈0.054–0.063, caw≈0.079–0.094, fv_gain≈-0.0173

Stage 2B changes vs Stage 2:
  1. candidate_fraction=0.35 (more NO_OP rows in each batch)
  2. noop_weight=1.5 (stronger NO_OP CE weight)
  3. L_noharm: softplus(noop_margin-(s_noop-max_cand)) on base-correct rows
  4. L_candidate_margin: softplus(cand_margin-(s_gold-max_other)) on candidate rows
  5. Conservative 2D eval grid: (prob_threshold × margin_threshold)
  6. New metrics: net_correction, applied_precision_ctg, benefit_damage_ratio
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


def build_train_index(data, tok_arr_t, reg_arr_t, unk_region, unk_super,
                       sr_enabled, args, device):
    """Returns noop_idx, cand_idx, n_ignored.

    Label rates: noop + cand + ignored = N (all training rows).
    """
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

    n_ignored = N - len(noop_idx) - len(cand_idx)
    print(f"  train index: noop={len(noop_idx):,}  cand={len(cand_idx):,}  "
          f"ignored={n_ignored:,}")
    return np.array(noop_idx, dtype=np.int64), np.array(cand_idx, dtype=np.int64), n_ignored


# ─────────────────────────────────────────────────────────────────────────────
# Model (same architecture as Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

class LocalDetailSelector(nn.Module):
    """
    Same architecture as Stage 2.
    Conservative behaviour comes from the V1B loss and eval policy.

    Action scores [B, P+1]: index 0 = NO_OP, 1..P = candidates.
    """

    def __init__(self, token_emb_weight, tok_arr, reg_arr,
                 d_model, n_regions, n_supers, sr_enabled,
                 unk_region, unk_super, top_k, pool_size,
                 memory_len=128, resolver_dim=256, hidden_dim=512,
                 attention_heads=4, dropout=0.1,
                 region_emb_dim=64, super_emb_dim=32):
        super().__init__()
        self.d_model      = d_model
        self.sr_enabled   = sr_enabled
        self.unk_region   = unk_region
        self.unk_super    = unk_super
        self.top_k        = top_k
        self.pool_size    = pool_size
        self.memory_len   = memory_len
        self.resolver_dim = resolver_dim

        self.register_buffer("token_emb_weight", token_emb_weight.detach().float())
        self.register_buffer("tok_arr", torch.from_numpy(tok_arr.astype(np.int32)).long())
        self.register_buffer("reg_arr", torch.from_numpy(reg_arr.astype(np.int32)).long())

        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim)
        self.super_emb  = nn.Embedding(n_supers + 2, super_emb_dim) if sr_enabled else None

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
        self.mem_proj = nn.Linear(d_model, resolver_dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=resolver_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(resolver_dim)
        self.score_head     = nn.Linear(resolver_dim, 1, bias=True)

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
        logit_gap  = lgt_c - lgt_b
        hp_dot_c   = (hp * ec).sum(-1, keepdim=True)
        hp_dot_b   = (hp * eb).sum(-1, keepdim=True)
        hp_dot_gap = hp_dot_c - hp_dot_b
        same_reg   = ((reg_c == reg_b) & (reg_c != self.unk_region)).float().unsqueeze(-1)
        remb_c = self.region_emb(reg_c.clamp(0, self.tok_arr.shape[0] - 1))
        remb_b = self.region_emb(reg_b.clamp(0, self.tok_arr.shape[0] - 1))
        parts = [hp, ec, eb, ec - eb, ec * eb,
                 lgt_c.unsqueeze(-1), lgt_b.unsqueeze(-1),
                 logit_gap.unsqueeze(-1), rank_norm.unsqueeze(-1),
                 hp_dot_c, hp_dot_b, hp_dot_gap,
                 remb_c, remb_b, same_reg]
        if self.sr_enabled and self.super_emb is not None and sup_c is not None:
            rlen   = self.reg_arr.shape[0] - 1
            semb_c = self.super_emb(sup_c.clamp(0, rlen))
            semb_b = self.super_emb(sup_b.clamp(0, rlen))
            same_sup = ((sup_c == sup_b) & (sup_c != self.unk_super)).float().unsqueeze(-1)
            parts.extend([semb_c, semb_b, same_sup])
        return torch.cat(parts, dim=-1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use forward_full instead")

    def forward_full(self, h_prime, input_ids, base_top1_ids, base_lgts,
                     cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups=None):
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

        mem_ids = input_ids[:, -self.memory_len:]
        mem_emb = self.token_emb_weight[mem_ids.clamp(0, vs - 1)]
        mem_kv  = self.mem_proj(mem_emb)

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
        kv_flat = mem_kv.unsqueeze(1).expand(B, P, M, self.resolver_dim) \
                        .reshape(B * P, M, self.resolver_dim)

        attn_out, attn_w = self.cross_attn(q_flat, kv_flat, kv_flat,
                                            need_weights=True, average_attn_weights=True)
        evidence    = attn_out.squeeze(1).view(B, P, self.resolver_dim)
        cand_repr   = self.post_attn_norm(pair_hid + evidence)
        cand_scores = self.score_head(cand_repr).squeeze(-1)

        lgt1     = base_lgts[:, 1].nan_to_num(0.0) if K > 1 else base_lgt
        gap01    = base_lgt - lgt1
        valid_l  = base_lgts.nan_to_num(-1e9)
        topk_ent = -(valid_l.softmax(-1) * valid_l.log_softmax(-1)).sum(-1)

        noop_feat  = torch.cat([h_prime, base_emb,
                                 base_lgt.unsqueeze(1), gap01.unsqueeze(1),
                                 topk_ent.unsqueeze(1)], dim=1)
        noop_hid   = self.noop_mlp(noop_feat)
        noop_score = self.noop_head(noop_hid)

        action_scores = torch.cat([noop_score, cand_scores], dim=1)
        attn_w_out    = attn_w.squeeze(1).view(B, P, M)
        return action_scores, attn_w_out


# ─────────────────────────────────────────────────────────────────────────────
# Loss — V1B: weighted CE + NO_OP margin + candidate margin
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss_v1b(action_scores, labels, noop_weight, cand_weight,
                      lambda_noharm, noop_margin,
                      lambda_cand_margin, cand_margin, device):
    """
    L = weighted_CE
        + lambda_noharm   * L_noharm        (NO_OP-labeled rows)
        + lambda_cand_margin * L_cand_margin (candidate-labeled rows)

    L_noharm     = softplus(noop_margin  - (s_noop - max_cand_score))
    L_cand_margin = softplus(cand_margin - (s_gold - max_other_score))
    """
    valid = labels >= 0
    if not valid.any():
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, {"total": 0.0, "ce": 0.0, "noharm": 0.0, "cand_margin": 0.0, "n": 0}

    # Weighted CE
    sc_v = action_scores[valid]
    lb_v = labels[valid]
    w    = torch.where(lb_v == 0,
                       torch.full_like(lb_v, noop_weight,  dtype=torch.float),
                       torch.full_like(lb_v, cand_weight,  dtype=torch.float))
    L_ce = (F.cross_entropy(sc_v, lb_v, reduction="none") * w).mean()

    # NO_OP margin: s_noop should exceed max candidate by noop_margin
    L_noharm_t = torch.tensor(0.0, device=device)
    if lambda_noharm > 0:
        noop_rows = valid & (labels == 0)
        if noop_rows.any():
            sc_n        = action_scores[noop_rows]           # [n, P+1]
            s_noop      = sc_n[:, 0]
            s_best_cand = sc_n[:, 1:].max(1).values
            L_noharm_t  = F.softplus(noop_margin - (s_noop - s_best_cand)).mean()

    # Candidate margin: gold action should beat all others by cand_margin
    L_cand_t = torch.tensor(0.0, device=device)
    if lambda_cand_margin > 0:
        cand_rows = valid & (labels > 0)
        if cand_rows.any():
            sc_c  = action_scores[cand_rows]                 # [n_c, P+1]
            lb_c  = labels[cand_rows]
            ar    = torch.arange(sc_c.shape[0], device=device)
            s_gold = sc_c[ar, lb_c]
            mask   = torch.ones_like(sc_c, dtype=torch.bool)
            mask[ar, lb_c] = False
            s_wrong = sc_c.masked_fill(~mask, float("-inf")).max(1).values
            L_cand_t = F.softplus(cand_margin - (s_gold - s_wrong)).mean()

    total = L_ce + lambda_noharm * L_noharm_t + lambda_cand_margin * L_cand_t
    return total, {
        "total":       total.item(),
        "ce":          L_ce.item(),
        "noharm":      L_noharm_t.item()  if lambda_noharm      > 0 else 0.0,
        "cand_margin": L_cand_t.item()    if lambda_cand_margin > 0 else 0.0,
        "n":           int(valid.sum()),
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
    if isinstance(obj, dict):          return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):          return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):    return int(obj)
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
# Evaluation — 2D grid (prob_threshold × margin_threshold)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_policy(scores_np, prob_thr, margin_thr):
    """
    Combined conservative policy:
      If model picks candidate j:
        keep only if prob(j) >= prob_thr AND score(j) - score(NO_OP) >= margin_thr
        else force NO_OP.
    """
    N      = scores_np.shape[0]
    probs  = torch.from_numpy(scores_np).softmax(1).numpy()
    action = scores_np.argmax(1).copy()
    picks  = action > 0
    if picks.any():
        ar          = np.arange(N)
        chosen_prob  = probs[ar, action]
        chosen_score = scores_np[ar, action]
        noop_score   = scores_np[:, 0]
        margin_arr   = chosen_score - noop_score
        force_noop   = picks & ~((chosen_prob >= prob_thr) & (margin_arr >= margin_thr))
        action[force_noop] = 0
    return action


def _metrics_from_action(action, gold_np, bw_np, pg_np, sr_np, ss_np,
                           cand_np, topk_ids_np, topk_lgts_np, md_half,
                           action_ce=float("nan"), action_acc=float("nan")):
    """Compute full metric dict given an action array [N] and pre-collected arrays."""
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

    applied_prec_ctg    = (ctg_count / n_app)        if n_app > 0 else float("nan")
    applied_damage_rate = (caw_count / n_app)        if n_app > 0 else float("nan")
    bdrat               = ctg_count / max(caw_count, 1)
    net_corr            = (_rate(ctg, N) - _rate(caw, N)) if N > 0 else float("nan")

    return {
        "apply_rate":                  _rate(apply_mask,              N),
        "noop_rate":                   _rate(~apply_mask,             N),
        "noop_acc_on_base_correct":    _rate(noop_correct,            n_bc),
        "noharm_false_apply_rate":     _rate(apply_mask & ~bw_np,     n_bc),
        "false_apply_rate_on_bc":      _rate(apply_mask & ~bw_np,     n_bc),
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
        "action_ce_val":               action_ce,
        "action_acc_val":              action_acc,
        "n_bwcov":                     n_bwcov,
    }


def _checkpoint_score(m):
    """Conservative checkpoint score: heavily penalises changed_away."""
    ctg = m.get("changed_to_gold_rate",        float("nan"))
    caw = m.get("changed_away_rate",            float("nan"))
    sip = m.get("selected_gold_given_in_pool",  float("nan"))
    nac = m.get("noop_acc_on_base_correct",     float("nan"))
    if any(math.isnan(v) for v in [ctg, caw, sip, nac]):
        return float("nan")
    return ctg - 3.0 * caw + 0.25 * sip + 0.25 * nac


def evaluate_v1b(model, val_data, tok_arr_t, reg_arr_t,
                  unk_region, unk_super, sr_enabled, args, device,
                  prob_thresholds, margin_thresholds):
    """Forward pass + 2D grid evaluation.  Returns (grid, chk_score, raw_dict)."""
    model.eval()
    N, K = val_data["h_prime"].shape[0], val_data["topk_ids"].shape[1]
    P    = args.candidate_pool_size
    BSZ  = args.eval_batch_size
    md   = 0.5 * args.margin_delta
    vs   = tok_arr_t.shape[0]

    gold_np      = np.zeros(N, dtype=np.int64)
    base_np      = np.zeros(N, dtype=np.int64)
    bw_np        = np.zeros(N, dtype=bool)
    pg_np        = np.zeros(N, dtype=bool)
    sr_np        = np.zeros(N, dtype=bool)
    ss_np        = np.zeros(N, dtype=bool)
    scores_np    = np.zeros((N, P + 1), dtype=np.float32)
    cand_np      = np.zeros((N, P),     dtype=np.int64)
    cand_l_np    = np.zeros((N, P),     dtype=np.float32)
    topk_ids_np  = np.zeros((N, K),     dtype=np.int64)
    topk_lgts_np = np.zeros((N, K),     dtype=np.float32)
    attn_np      = np.zeros((N, P, args.memory_len), dtype=np.float32)
    mem_ids_np   = np.zeros((N, args.memory_len),    dtype=np.int64)

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
            act_sc, attn_w = model.forward_full(
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
            gold_np[sl]      = gold.cpu().numpy()
            base_np[sl]      = base_top1_ids.cpu().numpy()
            bw_np[sl]        = bw_b
            pg_np[sl]        = pg_b
            sr_np[sl]        = sr_b
            ss_np[sl]        = ss_b
            scores_np[sl]    = act_sc.cpu().numpy()
            cand_np[sl]      = cand_ids.cpu().numpy()
            cand_l_np[sl]    = cand_lgts.cpu().numpy()
            topk_ids_np[sl]  = topk.cpu().numpy()
            topk_lgts_np[sl] = lgt.cpu().numpy()
            M_actual         = attn_w.shape[2]
            attn_np[sl, :, :M_actual] = attn_w.cpu().numpy()
            mem_ids_np[sl]   = iids[:, -args.memory_len:].cpu().numpy()

    # Action CE / acc on labeled subset
    lbl_np     = np.full(N, -1, dtype=np.int64)
    correct_np = ~bw_np
    lbl_np[correct_np] = 0
    pg_cand = ~correct_np & pg_np
    if pg_cand.any():
        cand_ids_t = torch.from_numpy(cand_np)
        gold_t     = torch.from_numpy(gold_np)
        in_pool    = (cand_ids_t == gold_t.unsqueeze(1))
        pool_idx   = in_pool.long().argmax(1).numpy()
        lbl_np[pg_cand] = pool_idx[pg_cand] + 1

    valid_lbl = lbl_np >= 0
    if valid_lbl.any():
        sc_v       = torch.from_numpy(scores_np[valid_lbl])
        lb_v       = torch.from_numpy(lbl_np[valid_lbl]).long()
        action_ce  = F.cross_entropy(sc_v, lb_v).item()
        action_acc = float((sc_v.argmax(1) == lb_v).float().mean())
    else:
        action_ce = float("nan"); action_acc = float("nan")

    # Build 2D grid
    grid = []
    for pt in prob_thresholds:
        for mt in margin_thresholds:
            action = _apply_policy(scores_np, pt, mt)
            m = _metrics_from_action(
                    action, gold_np, bw_np, pg_np, sr_np, ss_np,
                    cand_np, topk_ids_np, topk_lgts_np, md,
                    action_ce, action_acc)
            m["prob_threshold"]   = pt
            m["margin_threshold"] = mt
            grid.append(m)

    # Checkpoint score at most-permissive corner (≈ argmax)
    min_mt     = min(margin_thresholds)
    argmax_row = next((m for m in grid
                       if m["prob_threshold"] == 0.0
                       and m["margin_threshold"] == min_mt), grid[0])
    chk_score  = _checkpoint_score(argmax_row)

    raw = {
        "gold": gold_np, "base": base_np, "bw": bw_np,  "pg": pg_np,
        "sr":   sr_np,   "ss":   ss_np,   "scores": scores_np,
        "cand_ids": cand_np, "cand_lgts": cand_l_np,
        "topk_ids": topk_ids_np, "topk_lgts": topk_lgts_np,
        "attn": attn_np, "mem_ids": mem_ids_np,
        "lbl":  lbl_np,
        "N": N, "P": P, "K": K,
    }
    model.train()
    return grid, chk_score, raw


def eval_full_vocab_from_action(model, val_data, action_arr, cand_np_,
                                  topk_ids_np_, args, device):
    """Full-vocab NLL / top-1 for one specific action array."""
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

                fv_base = hp @ model.token_emb_weight.T       # [b, VS]
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

def compute_baselines(raw, margin_delta):
    N   = raw["N"]; md = 0.5 * margin_delta
    bw  = raw["bw"]; pg = raw["pg"]
    gold_np  = raw["gold"]; cand_np_ = raw["cand_ids"]
    topk_ids_t  = torch.from_numpy(raw["topk_ids"])
    topk_lgts_t = torch.from_numpy(raw["topk_lgts"])
    n_bw    = int(bw.sum());  n_bc = int((~bw).sum())
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
                    rl = topk_lgts_t[i].clone()
                    sp = (topk_ids_t[i] == int(cand_np_[i, j])).long().argmax().item()
                    if topk_ids_t[i, sp] == int(cand_np_[i, j]):
                        rl[sp] += md; rl[0] -= md
                    if topk_ids_t[i, rl.argmax()].item() == gold_np[i]:
                        oracle_ctg += 1
                    break

    return {
        "base_top1_acc":             _rate(~bw,  N),
        "gold_in_pool_rate":         _rate(pg,   N),
        "baseline_rank2_sg_bwcov":   _rate(r2_sg[bw & pg], n_bwcov),
        "baseline_rank2_ctg":        _rate(r2_top1[bw] == gold_np[bw], n_bw),
        "baseline_rank2_caw":        _rate(r2_top1[~bw] != gold_np[~bw], n_bc),
        "baseline_oracle_pool_ctg":  _rate(oracle_ctg, n_bwcov) if n_bwcov > 0 else float("nan"),
        "stage1_ref_sel_gold_bwcov": 0.246,
        "stage1_ref_ctg":            0.040,
        "stage1_ref_caw":            0.062,
        "stage2_ref_ctg_lo":         0.054,
        "stage2_ref_ctg_hi":         0.063,
        "stage2_ref_caw_lo":         0.079,
        "stage2_ref_caw_hi":         0.094,
        "stage2_ref_fv_gain":        -0.0173,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Example reports
# ─────────────────────────────────────────────────────────────────────────────

def write_example_reports_v1b(val_data, raw, tok_arr_t, unk_region,
                               tokenizer, args, run_dir, best_pt, best_mt):
    rng    = np.random.default_rng(42)
    N, P   = raw["N"], raw["P"]
    bw     = raw["bw"];  gold_np = raw["gold"]
    cand_np_ = raw["cand_ids"];  cand_l = raw["cand_lgts"]
    scores  = raw["scores"]
    attn    = raw["attn"];  mem_ids = raw["mem_ids"]
    vs      = tok_arr_t.shape[0]

    def reg(tid):
        return str(int(tok_arr_t[min(int(tid), vs - 1)].item()))

    def d(tid):
        try:   return f"`{tokenizer.decode([int(tid)])}`"
        except: return str(tid)

    probs  = torch.from_numpy(scores).softmax(1).numpy()
    action = _apply_policy(scores, best_pt, best_mt)
    apply  = action > 0
    pool_j = (action - 1).clip(0, P - 1)
    sel_toks = cand_np_[np.arange(N), pool_j]
    sel_gold = apply & (sel_toks == gold_np)

    topk_ids_t  = torch.from_numpy(raw["topk_ids"])
    topk_lgts_t = torch.from_numpy(raw["topk_lgts"])
    ref_lgts = _apply_surgical(topk_lgts_t, topk_ids_t,
                                torch.from_numpy(sel_toks.astype(np.int64)),
                                torch.from_numpy(apply), 0.5 * args.margin_delta)
    ref_top1 = raw["topk_ids"][np.arange(N), ref_lgts.argmax(1).numpy()]
    ctg = bw & apply & (ref_top1 == gold_np)
    caw = ~bw & apply & (ref_top1 != gold_np)

    # High-confidence: p(chosen action) for applied rows
    chosen_p = probs[np.arange(N), action]

    def fmt_row(ri):
        lines = [f"### Row {ri}\n"]
        ctx_ids = mem_ids[ri]
        try:
            ctx = tokenizer.decode(ctx_ids[-64:].tolist(), skip_special_tokens=False)
            lines.append(f"**Context (last 64):** `{ctx}`\n")
        except Exception:
            pass
        gt     = int(gold_np[ri]); bt = int(raw["base"][ri])
        act_ri = int(action[ri])
        pn     = probs[ri, 0]
        lines.append(f"**Gold:** {d(gt)} id={gt} region={reg(gt)}")
        lines.append(f"**Base top-1:** {d(bt)} id={bt} region={reg(bt)}")
        if act_ri == 0:
            lines.append(f"**Action:** NO_OP  p_noop={pn:.3f}\n")
        else:
            st   = int(sel_toks[ri])
            pc   = probs[ri, act_ri]
            mgn  = scores[ri, act_ri] - scores[ri, 0]
            lines.append(f"**Action:** candidate {act_ri-1} = {d(st)} id={st}  "
                         f"p_noop={pn:.3f}  p_sel={pc:.3f}  "
                         f"score_sel-score_noop={mgn:.3f}\n")

        lines.append("| j | Token | ID | Region | Base lgt | Score | Prob | is_gold |")
        lines.append("|---|-------|----|--------|----------|-------|------|---------|")
        lines.append(f"| - | (NO_OP) | - | - | - | {scores[ri,0]:.3f} | {probs[ri,0]:.3f} | - |")
        for j in range(min(P, 10)):
            cid = int(cand_np_[ri, j])
            try:    ts = tokenizer.decode([cid])
            except: ts = str(cid)
            ig = "✓" if cid == gt else ""
            lines.append(f"| {j+1} | `{ts}` | {cid} | {reg(cid)} | "
                         f"{cand_l[ri,j]:.3f} | {scores[ri,j+1]:.3f} | "
                         f"{probs[ri,j+1]:.3f} | {ig} |")
        lines.append("")

        if act_ri > 0:
            sel_j = act_ri - 1
            aw    = attn[ri, sel_j]
            tk    = min(5, len(aw))
            top_i = np.argsort(aw)[::-1][:tk]
            lines.append("**Top attended memory tokens:**")
            lines.append("| Pos | Token | ID | Attn weight |")
            lines.append("|-----|-------|----|-------------|")
            for idx in top_i:
                mid = int(mem_ids[ri, idx])
                try:    ms = tokenizer.decode([mid])
                except: ms = str(mid)
                lines.append(f"| {idx} | `{ms}` | {mid} | {aw[idx]:.4f} |")
            lines.append("")
        return "\n".join(lines) + "\n"

    HIGH_CONF = 0.6
    buckets = {
        "selected_gold":    np.where(sel_gold)[0],
        "selected_wrong":   np.where(apply & ~(sel_toks == gold_np) & bw)[0],
        "noop_correct":     np.where((~apply) & (~bw))[0],
        "false_apply":      np.where(apply & ~bw)[0],
        "changed_to_gold":  np.where(ctg)[0],
        "changed_away":     np.where(caw)[0],
        "attention_debug":  np.where(apply)[0],
        "high_conf_correct": np.where(apply & sel_gold & (chosen_p > HIGH_CONF))[0],
        "high_conf_wrong":   np.where(apply & ~(sel_toks == gold_np) & (chosen_p > HIGH_CONF))[0],
    }
    for bname, indices in buckets.items():
        n_max = min(args.num_examples, 10) if bname == "attention_debug" else args.num_examples
        if len(indices) > n_max:
            indices = rng.choice(indices, n_max, replace=False)
        path = os.path.join(run_dir, f"examples_{bname}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# {bname.replace('_',' ').title()}\n\n"
                     f"_{len(indices)} examples_  "
                     f"policy=(pt={best_pt},mt={best_mt})\n\n---\n\n")
            for ri in indices:
                fh.write(fmt_row(int(ri)))
                fh.write("---\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report_v1b(run_dir, args, baselines, best_m_conservative,
                      best_m_net, argmax_m, best_fv, best_step):
    def f(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return "nan"
        if isinstance(v, float): return f"{v:.4f}"
        return str(v) if v is not None else "nan"

    m   = best_m_conservative  # primary metrics for questions
    ctg = m.get("changed_to_gold_rate",        float("nan"))
    caw = m.get("changed_away_rate",            float("nan"))
    nca = m.get("noharm_changed_away",          float("nan"))
    nac = m.get("noop_acc_on_base_correct",     float("nan"))
    sip = m.get("selected_gold_given_in_pool",  float("nan"))
    net = m.get("net_correction",               float("nan"))
    apt = m.get("applied_precision_ctg",        float("nan"))
    bdr = m.get("benefit_damage_ratio",         float("nan"))
    fvg = best_fv.get("full_vocab_gain",        float("nan")) if best_fv else float("nan")

    am_ctg = argmax_m.get("changed_to_gold_rate", float("nan"))
    am_caw = argmax_m.get("changed_away_rate",     float("nan"))

    s1_caw  = baselines.get("stage1_ref_caw",  0.062)
    s2_caw_hi = baselines.get("stage2_ref_caw_hi", 0.094)
    s2_ctg_lo = baselines.get("stage2_ref_ctg_lo", 0.054)
    s2_fvg    = baselines.get("stage2_ref_fv_gain", -0.0173)
    bl_r2     = baselines.get("baseline_rank2_sg_bwcov", float("nan"))

    lines = ["# Local Detail Memory Selector V1B — Conservative NO_OP Calibration\n"]
    lines.append(f"**Run:** `{args.run_name}`  |  **Steps:** {args.steps}  "
                 f"|  **memory_len:** {args.memory_len}  "
                 f"|  **noop_weight:** {args.noop_weight}  "
                 f"|  **candidate_fraction:** {args.candidate_fraction}\n")

    lines.append("## Reference\n")
    lines.append("| Experiment | ctg | caw | fv_gain |")
    lines.append("|-----------|-----|-----|---------|")
    lines.append("| Stage 1 (frozen h_prime) | ~0.040 | ~0.062 | ~-0.0199 |")
    lines.append("| Stage 2 (local token memory) | 0.054–0.063 | 0.079–0.094 | ~-0.0173 |")
    lines.append("| KNN V1 | — | — | -0.0116 |")
    lines.append("| V3 oracle | 0.2407 | 0.000 | +0.1584 |\n")

    lines.append("## Baselines\n")
    for k, v in baselines.items():
        lines.append(f"  {k:48s} = {f(v)}")
    lines.append("")

    lines.append("## Argmax Metrics (pt=0.0, mt=most permissive)\n")
    for k, v in argmax_m.items():
        lines.append(f"  {k:48s} = {f(v)}")
    lines.append("")

    lines.append("## Best-Conservative Metrics "
                 f"(pt={m.get('prob_threshold',0)}, mt={m.get('margin_threshold',0)})\n")
    for k, v in best_m_conservative.items():
        lines.append(f"  {k:48s} = {f(v)}")
    lines.append("")

    lines.append("## Best-Net-Correction Metrics "
                 f"(pt={best_m_net.get('prob_threshold',0)}, "
                 f"mt={best_m_net.get('margin_threshold',0)})\n")
    for k, v in best_m_net.items():
        lines.append(f"  {k:48s} = {f(v)}")
    lines.append("")

    if best_fv:
        lines.append("## Full-Vocab\n")
        for k, v in best_fv.items():
            lines.append(f"  {k:48s} = {f(v)}")
        lines.append("")

    lines.append(f"Best checkpoint: step={best_step}\n")
    lines.append("## Analysis\n")

    def yn(c): return "✅" if c else "⚠️"

    # Q1: Did caw reduce vs Stage 2?
    q1 = not math.isnan(am_caw) and am_caw < s2_caw_hi
    lines.append(f"**1. Did Stage 2B reduce changed_away vs Stage 2?**  {yn(q1)}")
    lines.append(f"   argmax caw={f(am_caw)}  vs Stage 2 hi={f(s2_caw_hi)}\n")

    # Q2: Did noharm / false apply reduce?
    q2 = not math.isnan(nca) and nca < s1_caw
    lines.append(f"**2. Did Stage 2B reduce noharm_changed_away / false-apply?**  {yn(q2)}")
    lines.append(f"   noharm_changed_away={f(nca)}  Stage 1 caw={f(s1_caw)}\n")

    # Q3: ctg > caw at any policy?
    net_max = best_m_net.get("net_correction", float("nan"))
    q3 = not math.isnan(net_max) and net_max > 0
    lines.append(f"**3. Does changed_to_gold exceed changed_away at any policy?**  {yn(q3)}")
    lines.append(f"   best net_correction={f(net_max)}  "
                 f"(pt={best_m_net.get('prob_threshold',0)}, "
                 f"mt={best_m_net.get('margin_threshold',0)})\n")

    # Q4: full_vocab_gain less negative?
    q4a = not math.isnan(fvg) and fvg >= 0
    q4b = not math.isnan(fvg) and not math.isnan(s2_fvg) and fvg > s2_fvg
    lines.append(f"**4. Does full_vocab_gain become positive or less negative?**  "
                 f"{yn(q4a)} (>=0)  {yn(q4b)} (>Stage2)")
    lines.append(f"   fv_gain={f(fvg)}  Stage 2 ref={f(s2_fvg)}\n")

    # Q5: Best tradeoff policy
    bm = best_m_conservative
    lines.append(f"**5. Best tradeoff policy:**  "
                 f"pt={bm.get('prob_threshold',0)}  mt={bm.get('margin_threshold',0)}")
    lines.append(f"   ctg={f(bm.get('changed_to_gold_rate',float('nan')))}  "
                 f"caw={f(bm.get('changed_away_rate',float('nan')))}  "
                 f"apply_rate={f(bm.get('apply_rate',float('nan')))}  "
                 f"applied_precision={f(bm.get('applied_precision_ctg',float('nan')))}  "
                 f"benefit_damage_ratio={f(bm.get('benefit_damage_ratio',float('nan')))}\n")

    # Q6: Is local memory useful when calibrated conservatively?
    q6 = q2 or q3 or q4b
    lines.append(f"**6. Is local detail memory useful once calibrated conservatively?**  {yn(q6)}")
    if q3:
        lines.append("   → YES: ctg > caw at best policy. "
                     "Local token memory provides useful evidence.")
    elif q2:
        lines.append("   → PARTIAL: conservatism reduced harm but ctg still ≤ caw. "
                     "Memory helps with NO_OP calibration but candidate selection is weak.")
    else:
        lines.append("   → UNCLEAR: neither harm reduced nor net positive. "
                     "May need hidden-state memory.")
    lines.append("")

    # Q7: Bottleneck diagnosis
    lines.append("**7. If still negative, primary bottleneck:**")
    if not q3:
        ctg_v = argmax_m.get("changed_to_gold_rate",       float("nan"))
        sip_v = argmax_m.get("selected_gold_given_in_pool", float("nan"))
        nac_v = argmax_m.get("noop_acc_on_base_correct",    float("nan"))
        gip_v = argmax_m.get("gold_in_pool_rate",           float("nan"))
        lines.append(f"   gold_in_pool_rate            = {f(gip_v)}  "
                     f"(pool coverage; target ~0.25+)")
        lines.append(f"   selected_gold_given_in_pool  = {f(sip_v)}  "
                     f"(candidate selection quality; target ~0.40+)")
        lines.append(f"   noop_acc_on_base_correct     = {f(nac_v)}  "
                     f"(NO_OP calibration; target ~0.90+)")
        lines.append(f"   changed_to_gold_rate         = {f(ctg_v)}  (actual corrections)")

        if not math.isnan(gip_v) and gip_v < 0.15:
            lines.append("   → PRIMARY: candidate pool coverage too low. "
                         "Gold rarely appears in top-32.")
        elif not math.isnan(sip_v) and sip_v < 0.30:
            lines.append("   → PRIMARY: weak candidate selection. "
                         "Gold in pool but model does not pick it. "
                         "Consider hidden-state memory or path-audit features.")
        elif not math.isnan(nac_v) and nac_v < 0.85:
            lines.append("   → PRIMARY: weak NO_OP calibration. "
                         "Model still applies on base-correct rows. "
                         "Increase noop_weight or noop_margin.")
        else:
            lines.append("   → Unclear; likely insufficient local token memory signal. "
                         "Next: hidden-state memory or Transformer-XL-style recurrent detail.")
    else:
        lines.append("   → No primary bottleneck: Stage 2B succeeded. "
                     "Consider Transformer-XL-style recurrent detail memory.")
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
        raise RuntimeError(f"Run dir exists: {run_dir}. Use --run_name to avoid overwriting.")
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
    noop_idx, cand_idx, n_ignored = build_train_index(
        train_data, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled, args, device)
    if len(cand_idx) == 0:
        raise RuntimeError("No candidate-label rows found in training data.")

    prob_thresholds   = [float(x) for x in args.prob_thresholds.split(",")]
    margin_thresholds = [float(x) for x in args.margin_thresholds.split(",")]
    min_mt = min(margin_thresholds)

    # Pretrain baseline
    print("\n[pretrain baseline]")
    _grid0, _, row0 = evaluate_v1b(
        model, val_data, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled, args, device,
        [0.0], [min_mt])
    baselines = compute_baselines(row0, args.margin_delta)
    N_train = train_data["h_prime"].shape[0]
    total_lbl = len(noop_idx) + len(cand_idx) + n_ignored
    baselines["train_label_noop_rate"]      = len(noop_idx)  / max(total_lbl, 1)
    baselines["train_label_candidate_rate"] = len(cand_idx)  / max(total_lbl, 1)
    baselines["train_ignored_rate"]         = n_ignored       / max(total_lbl, 1)
    baselines["val_gold_in_pool_rate"]      = baselines.pop("gold_in_pool_rate", float("nan"))
    baselines["val_base_correct_rate"]      = baselines.pop("base_top1_acc",     float("nan"))
    with open(os.path.join(run_dir, "pretrain_baseline.json"), "w") as fp:
        json.dump(_json_safe(baselines), fp, indent=2)
    for k, v in baselines.items():
        print(f"  {k:48s} = "
              f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"  {k:48s} = {v}")

    config = vars(args).copy()
    config.update({"d_model": d_model, "vocab_size": vocab_size,
                   "sr_enabled": sr_enabled, "n_params": n_params})
    with open(os.path.join(run_dir, "config.json"), "w") as fp:
        json.dump(config, fp, indent=2)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and torch.cuda.is_available()) else None

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
          f"(cand={cand_bsz} noop={noop_bsz})  lr={args.lr}  "
          f"noop_weight={args.noop_weight}  lambda_noharm={args.lambda_noharm_margin}\n")

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

        lbl           = assign_labels(gold, topk, cand_ids, args.uncovered_policy)
        base_top1_ids = topk[:, 0]

        with _amp_ctx(args.amp):
            act_sc, _ = model.forward_full(
                hp, iids, base_top1_ids, lgt,
                cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups)
            total_loss, ld = compute_loss_v1b(
                act_sc, lbl,
                args.noop_weight, args.candidate_weight,
                args.lambda_noharm_margin, args.noop_margin,
                args.lambda_candidate_margin, args.candidate_margin,
                device)
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
            print(f"  step={step:5d}  total={ld['total']:.4f}  ce={ld['ce']:.4f}  "
                  f"noharm={ld['noharm']:.4f}  cand_mgn={ld['cand_margin']:.4f}  "
                  f"n={ld['n']}  t={time.time()-t0:.0f}s")
        _write_csv_row(train_log, {"step": step, **ld}, header=not hdr_t); hdr_t = True

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n[eval] step={step}")
            grid, chk_score, val_row = evaluate_v1b(
                model, val_data, tok_arr_t, reg_arr_t,
                unk_region, unk_super, sr_enabled, args, device,
                prob_thresholds, margin_thresholds)

            argmax_m = next((m for m in grid
                             if m["prob_threshold"] == 0.0
                             and m["margin_threshold"] == min_mt), grid[0])
            ctg_v = argmax_m.get("changed_to_gold_rate", float("nan"))
            caw_v = argmax_m.get("changed_away_rate",     float("nan"))
            nac_v = argmax_m.get("noop_acc_on_base_correct", float("nan"))
            net_v = argmax_m.get("net_correction",         float("nan"))
            print(f"  [argmax] ctg={ctg_v:.4f}  caw={caw_v:.4f}  "
                  f"nac={nac_v:.4f}  net={net_v:.4f}  score={chk_score:.4f}"
                  if not math.isnan(chk_score) else
                  f"  [argmax] ctg={ctg_v:.4f}  caw={caw_v:.4f}  score=nan")

            torch.save({"step": step, "model": model.state_dict(), "args": vars(args)},
                       os.path.join(run_dir, "latest_selector.pt"))

            if not math.isnan(chk_score) and chk_score > best_score:
                best_score = chk_score; best_step = step
                torch.save({"step": step, "model": model.state_dict(), "args": vars(args)},
                           os.path.join(run_dir, "best_selector.pt"))
                with open(os.path.join(run_dir, "best_metrics.json"), "w") as fp:
                    json.dump(_json_safe({"step": step, "score": chk_score,
                                         "argmax_metrics": argmax_m}), fp, indent=2)
                print(f"  [best] step={step}  score={chk_score:.4f}")

            _write_csv_row(eval_log, {"step": step, **argmax_m}, header=not hdr_e)
            hdr_e = True
            for m in grid:
                _write_csv_row(sub_log, {"step": step, **m}, header=not hdr_s)
                hdr_s = True
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

    print("[final] Full grid evaluation on val...")
    final_grid, final_score, final_row = evaluate_v1b(
        model, val_data, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled, args, device,
        prob_thresholds, margin_thresholds)

    # Write threshold_grid.csv
    grid_csv = os.path.join(run_dir, "threshold_grid.csv")
    hdr = True
    for m in final_grid:
        _write_csv_row(grid_csv, m, header=hdr); hdr = False

    # Identify key policies
    argmax_final = next((m for m in final_grid
                         if m["prob_threshold"] == 0.0
                         and m["margin_threshold"] == min_mt), final_grid[0])

    def _cscore(m):
        return _checkpoint_score(m) if not math.isnan(_checkpoint_score(m)) else -1e9

    best_conservative = max(final_grid, key=_cscore)
    best_net = max(final_grid,
                   key=lambda m: m.get("net_correction", float("-inf"))
                                 if not math.isnan(m.get("net_correction", float("nan")))
                                 else float("-inf"))

    # Full-vocab eval for best policies (deduplicated by action hash)
    best_fv = {}
    if args.eval_full_vocab:
        print("[final] Full-vocab eval for key policies...")
        seen_policies = set()
        fv_candidates = [argmax_final, best_conservative, best_net]
        for pm in fv_candidates:
            key = (pm["prob_threshold"], pm["margin_threshold"])
            if key in seen_policies:
                continue
            seen_policies.add(key)
            label = f"pt{pm['prob_threshold']}_mt{pm['margin_threshold']}"
            action_arr = _apply_policy(final_row["scores"],
                                        pm["prob_threshold"],
                                        pm["margin_threshold"])
            fv = eval_full_vocab_from_action(
                model, val_data, action_arr,
                final_row["cand_ids"], final_row["topk_ids"], args, device)
            fv["prob_threshold"]   = pm["prob_threshold"]
            fv["margin_threshold"] = pm["margin_threshold"]
            print(f"  [{label}] fv_gain={fv.get('full_vocab_gain', float('nan')):.4f}")
            if label == f"pt{argmax_final['prob_threshold']}_mt{argmax_final['margin_threshold']}":
                best_fv = fv
            # attach fv to matching grid rows
            for gm in final_grid:
                if (gm["prob_threshold"] == pm["prob_threshold"]
                        and gm["margin_threshold"] == pm["margin_threshold"]):
                    gm["full_vocab_gain"] = fv.get("full_vocab_gain", float("nan"))

    final_metrics = {
        "step":                           args.steps,
        "best_step_during_training":      best_step,
        "loaded_best_step_for_final_eval": loaded_step,
        "final_eval_uses_best_checkpoint": loaded_step >= 0,
        "argmax_metrics":       _json_safe(argmax_final),
        "best_conservative":    _json_safe(best_conservative),
        "best_net_correction":  _json_safe(best_net),
        "full_vocab":           _json_safe(best_fv),
        "baselines":            _json_safe(baselines),
    }
    with open(os.path.join(run_dir, "final_metrics.json"), "w") as fp:
        json.dump(final_metrics, fp, indent=2)

    # Examples
    try:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            tokenizer = type("FT", (), {"decode": lambda s, ids, **kw: str(ids)})()
        write_example_reports_v1b(
            val_data, final_row, tok_arr_t.cpu(), unk_region,
            tokenizer, args, run_dir,
            best_conservative["prob_threshold"],
            best_conservative["margin_threshold"])
    except Exception as exc:
        print(f"[warn] example reports failed: {exc}")

    write_report_v1b(run_dir, args, baselines,
                      best_conservative, best_net, argmax_final,
                      best_fv, loaded_step)

    print(f"\n{'='*60}")
    print(f" Local Detail Selector V1B complete.")
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
        description="Local Detail Selector V1B — Conservative NO_OP Calibration")
    p.add_argument("--small_ckpt",          required=True)
    p.add_argument("--train_dir",           required=True)
    p.add_argument("--val_dir",             required=True)
    p.add_argument("--token_to_region",     required=True)
    p.add_argument("--super_map",           default=None)
    p.add_argument("--output_root",         default="runs/local_detail_selector_v1b")
    p.add_argument("--run_name",            default="cand_xattn_tokmem_noharm_v1b")
    p.add_argument("--top_k",              type=int,   default=256)
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
    p.add_argument("--candidate_fraction",  type=float, default=0.35)
    p.add_argument("--grad_accum_steps",    type=int,   default=1)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--steps",               type=int,   default=5000)
    p.add_argument("--eval_every",          type=int,   default=500)
    p.add_argument("--eval_batch_size",     type=int,   default=256)
    p.add_argument("--grad_clip",           type=float, default=1.0)
    # V1B loss weights
    p.add_argument("--noop_weight",             type=float, default=1.5)
    p.add_argument("--candidate_weight",        type=float, default=1.0)
    p.add_argument("--lambda_noharm_margin",    type=float, default=1.0)
    p.add_argument("--noop_margin",             type=float, default=1.0)
    p.add_argument("--lambda_candidate_margin", type=float, default=0.25)
    p.add_argument("--candidate_margin",        type=float, default=1.0)
    p.add_argument("--margin_delta",            type=float, default=1.0)
    # V1B eval grid
    p.add_argument("--prob_thresholds",
                   default="0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--margin_thresholds",
                   default="-1.0,-0.5,0.0,0.5,1.0,1.5,2.0")
    p.add_argument("--primary_policy",          default="combined",
                   choices=["argmax", "prob", "margin", "combined"])
    p.add_argument("--eval_full_vocab",     action="store_true")
    p.add_argument("--amp",                 action="store_true")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--max_train_rows",      type=int,   default=None)
    p.add_argument("--max_val_rows",        type=int,   default=None)
    p.add_argument("--num_examples",        type=int,   default=40)
    return p.parse_args()


if __name__ == "__main__":
    main()
