#!/usr/bin/env python3
"""
train_multicandidate_selector_v1.py — Learned Multi-Candidate Selector (Non-Oracle)

At eval: model chooses NO_OP or one challenger candidate from candidate pool.
No gold is used at eval. Surgical correction only on selected pair.

V3 oracle ref: target_ctg=0.2407, oracle_fv_gain=+0.1584
KNN V1 ref:    knn_sel_gold_bwcov=0.0070, rank2_sel=0.1452, fv_gain=-0.0116
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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(ckpt_path, device):
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        ckpt_path, device)
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
        raise KeyError(f"Shard missing all aliases {names}. Have: {list(shard.keys())}")
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
            topk = torch.cat([topk, torch.zeros(B, top_k - K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, top_k - K), float("nan"))], 1)
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
# Candidate pool construction (no gold)
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_regs(token_ids, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled):
    vs   = tok_arr_t.shape[0]
    safe = token_ids.clamp(0, vs - 1)
    regs = tok_arr_t[safe]
    sups = reg_arr_t[regs.clamp(0, reg_arr_t.shape[0] - 1)] if sr_enabled else None
    return regs, sups


def build_candidate_pool(topk_ids, topk_lgt, tok_arr_t, reg_arr_t,
                          unk_region, unk_super, sr_enabled,
                          pool_size, candidate_filter, device):
    """
    Returns candidate_ids [N, pool_size], candidate_lgts [N, pool_size],
    candidate_ranks [N, pool_size] (0-based rank within topK).
    All candidates are drawn from topK[1:] — no gold used.
    """
    N, K = topk_ids.shape
    base_regs = tok_arr_t[topk_ids[:, 0].clamp(0, tok_arr_t.shape[0]-1)]   # [N]
    if sr_enabled:
        base_sups = reg_arr_t[base_regs.clamp(0, reg_arr_t.shape[0]-1)]    # [N]
    else:
        base_sups = None

    cand_ids  = torch.zeros(N, pool_size, dtype=torch.long,  device=device)
    cand_lgts = torch.zeros(N, pool_size, dtype=torch.float, device=device)
    cand_rnks = torch.zeros(N, pool_size, dtype=torch.long,  device=device)
    cand_regs_out = torch.full((N, pool_size), unk_region, dtype=torch.long, device=device)
    cand_sups_out = torch.full((N, pool_size), unk_super,  dtype=torch.long, device=device) if sr_enabled else None

    for i in range(N):
        # Candidates are topK[1:]
        cand_toks = topk_ids[i, 1:]     # [K-1]
        cand_l    = topk_lgt[i, 1:]     # [K-1]
        cand_r    = torch.arange(1, K, device=device)  # original rank in topK

        if candidate_filter != "top_rank":
            cr_i = tok_arr_t[cand_toks.clamp(0, tok_arr_t.shape[0]-1)]
            if sr_enabled:
                cs_i = reg_arr_t[cr_i.clamp(0, reg_arr_t.shape[0]-1)]
            br = base_regs[i]; bs = base_sups[i] if sr_enabled else None

            if candidate_filter == "same_region_only":
                mask = (cr_i == br) & (cr_i != unk_region)
            elif candidate_filter == "same_superregion_only":
                if sr_enabled:
                    mask = (cs_i == bs) & (cs_i != unk_super)
                else:
                    mask = torch.zeros(len(cand_toks), dtype=torch.bool, device=device)
            elif candidate_filter == "same_region_or_superregion":
                same_r = (cr_i == br) & (cr_i != unk_region)
                if sr_enabled:
                    same_s = (cs_i == bs) & (cs_i != unk_super)
                else:
                    same_s = torch.zeros_like(same_r)
                mask = same_r | same_s
            else:
                mask = torch.ones(len(cand_toks), dtype=torch.bool, device=device)

            filt_toks = cand_toks[mask]
            if len(filt_toks) == 0:
                # fallback to top_rank
                pass
            else:
                cand_toks = filt_toks
                cand_l    = cand_l[mask]
                cand_r    = cand_r[mask]

        n_take = min(pool_size, len(cand_toks))
        cand_ids[i, :n_take]  = cand_toks[:n_take]
        cand_lgts[i, :n_take] = cand_l[:n_take]
        cand_rnks[i, :n_take] = cand_r[:n_take]
        cr_fill = tok_arr_t[cand_toks[:n_take].clamp(0, tok_arr_t.shape[0]-1)]
        cand_regs_out[i, :n_take] = cr_fill
        if sr_enabled and cand_sups_out is not None:
            cand_sups_out[i, :n_take] = reg_arr_t[cr_fill.clamp(0, reg_arr_t.shape[0]-1)]

    return cand_ids, cand_lgts, cand_rnks, cand_regs_out, cand_sups_out


def assign_labels(gold, topk_ids, cand_ids, uncovered_policy):
    """
    Returns labels [N] (int):
      -1  = ignore
       0  = NO_OP
       j  = candidate index j (1-based, i.e. cand_ids[:, j-1])
    """
    N, P = cand_ids.shape
    base_top1 = topk_ids[:, 0]
    base_correct = (base_top1 == gold)

    # Find gold in candidate pool
    gold_in_pool = (cand_ids == gold.unsqueeze(1))  # [N, P]
    pool_has_gold = gold_in_pool.any(1)
    gold_pool_idx = gold_in_pool.long().argmax(1)    # [N], valid only where pool_has_gold

    labels = torch.full((N,), -1, dtype=torch.long)  # default ignore

    # Base-correct -> NO_OP (label 0)
    labels[base_correct] = 0

    # Base-wrong, gold in pool -> label = pool_idx + 1 (1-indexed)
    covered_wrong = ~base_correct & pool_has_gold
    labels[covered_wrong] = gold_pool_idx[covered_wrong] + 1

    # Base-wrong, gold not in pool -> noop or ignore
    uncovered_wrong = ~base_correct & ~pool_has_gold
    if uncovered_policy == "noop":
        labels[uncovered_wrong] = 0
    # else: leave as -1 (ignore)

    return labels, base_correct, pool_has_gold


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MultiCandidateSelector(nn.Module):
    """
    MLP multi-candidate selector.

    Output: action scores [B, P+1]
      index 0 = NO_OP
      index j = select candidate j-1 (0-indexed in pool)

    At eval: action = argmax(scores), no gold used.
    """

    def __init__(self, token_emb_weight, tok_arr, reg_arr,
                 d_model, n_regions, n_supers, sr_enabled,
                 unk_region, unk_super, top_k,
                 pool_size=32, region_emb_dim=64, super_emb_dim=32,
                 hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_model    = d_model
        self.sr_enabled = sr_enabled
        self.unk_region = unk_region
        self.unk_super  = unk_super
        self.top_k      = top_k
        self.pool_size  = pool_size

        self.register_buffer("token_emb_weight", token_emb_weight.detach().float())
        self.register_buffer("tok_arr", torch.from_numpy(tok_arr.astype(np.int32)).long())
        self.register_buffer("reg_arr", torch.from_numpy(reg_arr.astype(np.int32)).long())

        self.region_emb = nn.Embedding(n_regions + 2, region_emb_dim)
        self.super_emb  = nn.Embedding(n_supers + 2, super_emb_dim) if sr_enabled else None

        # Per-candidate pair feature:
        # [norm(h_prime), tok_c, tok_b, tok_c-tok_b,
        #  reg_c, reg_b, reg_c-reg_b, (sup_c, sup_b if sr),
        #  logit_c, logit_b, logit_gap, rank_norm,
        #  same_region, same_super, hp_dot_c, hp_dot_b, hp_dot_gap]
        pair_dim = (d_model                    # h_prime
                  + 3 * d_model                # tok_c, tok_b, tok_c-tok_b
                  + 3 * region_emb_dim         # reg_c, reg_b, diff
                  + 7)                         # scalars: lgt_c, lgt_b, gap, rank_norm,
                                               #          same_reg, hp_dot_c, hp_dot_b
        if sr_enabled:
            pair_dim += 2 * super_emb_dim + 2  # sup_c, sup_b, same_sup, hp_dot_gap

        pair_layers: List[nn.Module] = []
        in_d = pair_dim
        for _ in range(n_layers):
            pair_layers += [nn.Linear(in_d, hidden_dim), nn.GELU(),
                            nn.Dropout(dropout)]
            in_d = hidden_dim
        self.pair_mlp = nn.Sequential(*pair_layers)
        self.cand_score_head = nn.Linear(hidden_dim, 1, bias=True)

        # NO_OP feature:
        # h_prime, tok_b, logit_b, logit_gap_rank1_rank2, topk_entropy
        noop_dim = d_model + d_model + 3
        noop_layers: List[nn.Module] = []
        in_d2 = noop_dim
        for _ in range(n_layers):
            noop_layers += [nn.Linear(in_d2, hidden_dim), nn.GELU(),
                            nn.Dropout(dropout)]
            in_d2 = hidden_dim
        self.noop_mlp  = nn.Sequential(*noop_layers)
        self.noop_head = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, h_prime, base_ids, base_lgts,
                cand_ids, cand_lgts, cand_rnks,
                cand_regs, cand_sups=None):
        """
        h_prime   [B, d]
        base_ids  [B]       topK[0]
        base_lgts [B, K]    full topK logits
        cand_ids  [B, P]
        cand_lgts [B, P]
        cand_rnks [B, P]    0-based rank in topK
        cand_regs [B, P]
        cand_sups [B, P] or None

        Returns action_scores [B, P+1]  (index 0 = NO_OP)
        """
        B, P  = cand_ids.shape
        K     = base_lgts.shape[1]
        vs    = self.token_emb_weight.shape[0]
        d     = self.d_model

        base_tok = self.token_emb_weight[base_ids.clamp(0, vs-1)]  # [B, d]
        base_reg = self.tok_arr[base_ids.clamp(0, self.tok_arr.shape[0]-1)]  # [B]
        base_remb = self.region_emb(base_reg)                       # [B, re]

        lgt0  = base_lgts[:, 0]                                     # [B]
        lgt1  = base_lgts[:, 1].nan_to_num(0.0) if K > 1 else lgt0 # [B]
        gap01 = lgt0 - lgt1                                         # [B]

        # topK entropy (softmax over valid logits)
        valid_lgt = base_lgts.nan_to_num(-1e9)
        topk_ent  = -(valid_lgt.softmax(-1) * valid_lgt.log_softmax(-1)).sum(-1)  # [B]

        # ── NO_OP score ──────────────────────────────────────────────────────
        noop_feat = torch.cat([
            h_prime, base_tok,
            lgt0.unsqueeze(1), gap01.unsqueeze(1), topk_ent.unsqueeze(1),
        ], dim=1)                                                    # [B, noop_dim]
        noop_hid   = self.noop_mlp(noop_feat)
        noop_score = self.noop_head(noop_hid).squeeze(1)            # [B]

        # ── Candidate scores ─────────────────────────────────────────────────
        # Flatten over P for batched MLP
        cand_tok = self.token_emb_weight[
            cand_ids.reshape(-1).clamp(0, vs-1)].view(B, P, d)     # [B, P, d]
        cand_remb = self.region_emb(
            cand_regs.clamp(0, self.tok_arr.shape[0]-1))            # [B, P, re]

        hp_exp    = h_prime.unsqueeze(1).expand(B, P, d)            # [B, P, d]
        bt_exp    = base_tok.unsqueeze(1).expand(B, P, d)
        br_exp    = base_remb.unsqueeze(1).expand_as(cand_remb)

        logit_c   = cand_lgts                                        # [B, P]
        logit_b   = lgt0.unsqueeze(1).expand(B, P)
        logit_gap = logit_c - logit_b
        rank_norm = cand_rnks.float() / max(K, 1)
        same_reg  = ((cand_regs == base_reg.unsqueeze(1)) &
                     (cand_regs != self.unk_region)).float()
        hp_dot_c  = (hp_exp * cand_tok).sum(-1)
        hp_dot_b  = (hp_exp * bt_exp).sum(-1)

        parts = [
            hp_exp, cand_tok, bt_exp, cand_tok - bt_exp,
            cand_remb, br_exp, cand_remb - br_exp,
            logit_c.unsqueeze(-1), logit_b.unsqueeze(-1),
            logit_gap.unsqueeze(-1), rank_norm.unsqueeze(-1),
            same_reg.unsqueeze(-1),
            hp_dot_c.unsqueeze(-1), hp_dot_b.unsqueeze(-1),
        ]

        if self.sr_enabled and cand_sups is not None and self.super_emb is not None:
            rlen = self.reg_arr.shape[0] - 1
            base_sup = self.reg_arr[base_reg.clamp(0, rlen)]
            base_semb = self.super_emb(base_sup)                    # [B, se]
            cand_semb = self.super_emb(
                cand_sups.clamp(0, rlen))                           # [B, P, se]
            bs_exp = base_semb.unsqueeze(1).expand_as(cand_semb)
            same_sup = ((cand_sups == base_sup.unsqueeze(1)) &
                        (cand_sups != self.unk_super)).float()
            hp_dot_gap = (hp_dot_c - hp_dot_b).unsqueeze(-1)
            parts.extend([cand_semb, bs_exp, same_sup.unsqueeze(-1), hp_dot_gap])

        pair_feat = torch.cat(parts, dim=-1)                        # [B, P, pair_dim]
        pair_flat = pair_feat.view(B * P, -1)
        cand_hid  = self.pair_mlp(pair_flat).view(B, P, -1)
        cand_scores = self.cand_score_head(cand_hid).squeeze(-1)    # [B, P]

        action_scores = torch.cat([noop_score.unsqueeze(1), cand_scores], dim=1)  # [B, P+1]
        return action_scores


# ─────────────────────────────────────────────────────────────────────────────
# Surgical correction
# ─────────────────────────────────────────────────────────────────────────────

def apply_selection(topk_lgts, topk_ids, action, cand_ids, margin_delta, vocab_size):
    """
    action [N]: 0=NO_OP, j=select candidate j-1 (pool index)
    Returns refined_lgts [N, K] (only two entries changed per row where action>0).
    """
    refined = topk_lgts.clone()
    N, K    = topk_lgts.shape
    P       = cand_ids.shape[1]
    md      = 0.5 * margin_delta

    apply_mask = action > 0
    if not apply_mask.any():
        return refined

    ar = torch.arange(N, device=topk_lgts.device)
    pool_idx  = (action - 1).clamp(0, P - 1)                       # 0-indexed in pool
    sel_tok   = cand_ids[ar, pool_idx]                              # [N] selected token

    # Find position of selected token and base_top1 in topk_ids
    sel_pos   = (topk_ids == sel_tok.unsqueeze(1)).long().argmax(1) # [N]
    # sel_pos valid only if token actually appears; guard: valid = found & apply
    sel_found = (topk_ids[ar, sel_pos] == sel_tok) & apply_mask
    # base_top1 is always position 0

    # Surgical edit
    if sel_found.any():
        refined[ar[sel_found], sel_pos[sel_found]] += md
        refined[ar[sel_found], 0]                  -= md

    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(action_scores, labels, topk_lgts, topk_ids, cand_ids,
                 gold, margin_delta, noop_weight, cand_weight, lambda_margin,
                 target_margin, device):
    N, P1 = action_scores.shape   # P1 = P+1
    P     = P1 - 1

    valid = labels >= 0
    if not valid.any():
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"total": 0.0, "ce": 0.0, "margin": 0.0, "n_valid": 0}

    sc_v = action_scores[valid]
    lb_v = labels[valid]

    # Per-example weights
    w = torch.where(lb_v == 0,
                    torch.full_like(lb_v, noop_weight, dtype=torch.float),
                    torch.full_like(lb_v, cand_weight, dtype=torch.float))
    L_ce = (F.cross_entropy(sc_v, lb_v, reduction="none") * w).mean()

    L_margin = torch.tensor(0.0, device=device)
    if lambda_margin > 0:
        # Only rows where label is a candidate (not NO_OP) and gold is in pool
        cand_label_rows = valid & (labels > 0)
        if cand_label_rows.any():
            ar = torch.arange(N, device=device)
            pool_idx  = (labels - 1).clamp(0, P - 1)
            sel_tok   = cand_ids[ar, pool_idx]
            # Find sel_tok position in topk
            sel_pos   = (topk_ids == sel_tok.unsqueeze(1)).long().argmax(1)
            sel_found = (topk_ids[ar, sel_pos] == sel_tok) & cand_label_rows
            if sel_found.any():
                # Compute refined margin assuming oracle correction
                ref_lgt_sel  = topk_lgts[sel_found]
                ref_pos      = sel_pos[sel_found]
                ar2 = torch.arange(sel_found.sum(), device=device)
                ref_gold_lgt = ref_lgt_sel[ar2, ref_pos] + 0.5 * margin_delta
                ref_base_lgt = ref_lgt_sel[:, 0] - 0.5 * margin_delta
                margin_ref   = ref_gold_lgt - ref_base_lgt
                L_margin = F.softplus(target_margin - margin_ref).mean()

    total = L_ce + lambda_margin * L_margin
    return total, {
        "total":    total.item(),
        "ce":       L_ce.item(),
        "margin":   L_margin.item() if lambda_margin > 0 else 0.0,
        "n_valid":  int(valid.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(arr):
    a = np.asarray(arr, dtype=np.float32)
    return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else float("nan")


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


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines(data, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
                  args, device):
    """
    Compute baseline metrics on val:
      base_noop, rank2_always, oracle_pool.
    """
    N, K = data["h_prime"].shape[0], data["topk_ids"].shape[1]
    topk = data["topk_ids"]
    lgt  = data["topk_lgt"]
    gold = data["gold"]

    base_top1 = topk[:, 0]
    base_correct = (base_top1 == gold)
    md = 0.5 * args.margin_delta

    # Gold in topK
    gold_in_topk = (topk == gold.unsqueeze(1)).any(1)

    # Region masks
    vs = tok_arr_t.shape[0]
    gold_reg = tok_arr_t[gold.clamp(0, vs-1)]
    top1_reg = tok_arr_t[base_top1.clamp(0, vs-1)]
    same_reg = (gold_reg == top1_reg) & (gold_reg != unk_region)
    if sr_enabled:
        rlen = reg_arr_t.shape[0] - 1
        gold_sup = reg_arr_t[gold_reg.clamp(0, rlen)]
        top1_sup = reg_arr_t[top1_reg.clamp(0, rlen)]
        same_sup = (gold_sup == top1_sup) & (gold_sup != unk_super)
    else:
        same_sup = torch.zeros(N, dtype=torch.bool)
    target_mask = ~base_correct & gold_in_topk & (same_reg | same_sup)

    # Build candidate pool once (top_rank)
    cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
        topk.to(device), lgt.to(device), tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled,
        args.candidate_pool_size, args.candidate_filter, device)

    labels, base_correct_t, pool_has_gold_t = assign_labels(
        gold.to(device), topk.to(device), cand_ids, args.uncovered_policy)

    pool_has_gold_np = pool_has_gold_t.cpu().numpy()
    labels_np        = labels.cpu().numpy()
    base_correct_np  = base_correct.numpy()
    gold_np          = gold.numpy()
    topk_np          = topk.numpy()
    lgt_np           = lgt.numpy()
    cand_ids_np      = cand_ids.cpu().numpy()
    cand_lgts_np     = cand_lgts.cpu().numpy()

    n_base_wrong_cov = int((~base_correct_np & pool_has_gold_np).sum())
    n_target         = int(target_mask.numpy().sum())

    def pool_gold_idx(i):
        for j in range(cand_ids_np.shape[1]):
            if cand_ids_np[i, j] == gold_np[i]:
                return j
        return -1

    # Oracle-pool baseline: select gold if in pool and base wrong, else NO_OP
    oracle_ctg = oracle_caw = oracle_apply = 0
    for i in range(N):
        if not base_correct_np[i] and pool_has_gold_np[i]:
            gj = pool_gold_idx(i)
            if gj >= 0:
                oracle_apply += 1
                # Surgical edit in topK
                ref_lgt = lgt_np[i].copy()
                sel_tok = cand_ids_np[i, gj]
                sel_pos = np.where(topk_np[i] == sel_tok)[0]
                if len(sel_pos) > 0:
                    ref_lgt[sel_pos[0]] += md
                    ref_lgt[0]          -= md
                ref_top1 = topk_np[i, ref_lgt.argmax()]
                if ref_top1 == gold_np[i]:
                    oracle_ctg += 1
        elif base_correct_np[i]:
            # NO_OP on base-correct — check no damage (trivially none here)
            pass

    # Rank2-always: always select pool[0]
    rank2_ctg = rank2_caw = rank2_n_changed = 0
    for i in range(N):
        ref_lgt = lgt_np[i].copy()
        sel_tok = cand_ids_np[i, 0]
        sel_pos = np.where(topk_np[i] == sel_tok)[0]
        if len(sel_pos) > 0:
            ref_lgt[sel_pos[0]] += md
            ref_lgt[0]          -= md
        ref_top1 = topk_np[i, ref_lgt.argmax()]
        rank2_n_changed += 1
        if not base_correct_np[i] and ref_top1 == gold_np[i]:
            rank2_ctg += 1
        if base_correct_np[i] and ref_top1 != gold_np[i]:
            rank2_caw += 1

    n_bc  = int(base_correct_np.sum())
    n_bw  = int((~base_correct_np).sum())

    stats = {
        "N": N,
        "base_top1_acc":              _rate(base_correct_np, N),
        "gold_in_topk_rate":          float(gold_in_topk.float().mean()),
        "train_gold_in_pool_rate":    None,  # filled below
        "val_gold_in_pool_rate":      _rate(pool_has_gold_np, N),
        "label_noop_rate":            _rate(labels_np == 0, N),
        "label_candidate_rate":       _rate(labels_np > 0, N),
        "ignored_rate":               _rate(labels_np == -1, N),
        "n_base_wrong_covered":       n_base_wrong_cov,
        "n_target":                   n_target,
        # Baselines
        "baseline_noop_ctg":          0.0,
        "baseline_noop_caw":          0.0,
        "baseline_rank2_ctg":         _rate(rank2_ctg, n_bw) if n_bw > 0 else float("nan"),
        "baseline_rank2_caw":         _rate(rank2_caw, n_bc) if n_bc > 0 else float("nan"),
        "baseline_oracle_pool_ctg":   _rate(oracle_ctg, n_base_wrong_cov) if n_base_wrong_cov > 0 else float("nan"),
        "baseline_oracle_apply_rate": _rate(oracle_apply, N),
    }
    return stats, cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups, labels


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, val_data, tok_arr_t, reg_arr_t,
             unk_region, unk_super, sr_enabled, args, device):
    model.eval()
    N, K = val_data["h_prime"].shape[0], val_data["topk_ids"].shape[1]
    P    = args.candidate_pool_size
    BSZ  = args.eval_batch_size
    vs   = tok_arr_t.shape[0]
    md   = args.margin_delta

    # Per-row result arrays
    action_arr    = np.zeros(N, dtype=np.int32)
    sel_tok_arr   = np.zeros(N, dtype=np.int64)
    gold_arr      = np.zeros(N, dtype=np.int64)
    base_arr      = np.zeros(N, dtype=np.int64)
    ref_top1_arr  = np.zeros(N, dtype=np.int64)
    pool_gold_arr = np.zeros(N, dtype=bool)
    ctg_arr       = np.zeros(N, dtype=bool)
    caw_arr       = np.zeros(N, dtype=bool)
    bw_arr        = np.zeros(N, dtype=bool)
    sr_arr        = np.zeros(N, dtype=bool)
    ss_arr        = np.zeros(N, dtype=bool)
    sel_gold_arr  = np.zeros(N, dtype=bool)  # selected == gold
    base_lgts_top10 = np.zeros((N, min(10, K)), dtype=np.float32)
    ref_lgts_top10  = np.zeros((N, min(10, K)), dtype=np.float32)
    topk_ids_store  = np.zeros((N, min(10, K)), dtype=np.int64)
    cand_ids_store  = np.zeros((N, P), dtype=np.int64)
    cand_lgts_store = np.zeros((N, P), dtype=np.float32)
    cand_scores_store = np.zeros((N, P+1), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, N, BSZ):
            e   = min(s + BSZ, N)
            b   = e - s

            hp   = val_data["h_prime"][s:e].to(device)
            topk = val_data["topk_ids"][s:e].to(device)
            lgt  = val_data["topk_lgt"][s:e].to(device)
            gold = val_data["gold"][s:e].to(device)

            cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
                topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
                P, args.candidate_filter, device)

            base_ids = topk[:, 0]
            scores   = model(hp, base_ids, lgt, cand_ids, cand_lgts, cand_rnks,
                             cand_regs, cand_sups)          # [b, P+1]
            action   = scores.argmax(1)                     # [b]

            ref_lgt  = apply_selection(lgt, topk, action, cand_ids, md,
                                       model.token_emb_weight.shape[0])  # [b, K]

            ar_b = torch.arange(b, device=device)
            ref_top1_i  = ref_lgt.argmax(1)
            ref_top1    = topk[ar_b, ref_top1_i]
            base_top1   = topk[:, 0]
            bw_b        = base_top1 != gold

            gold_in_pool_b = (cand_ids == gold.unsqueeze(1)).any(1)
            pool_idx_sel   = (action - 1).clamp(0, P - 1)
            sel_tok_b      = cand_ids[ar_b, pool_idx_sel]  # only valid if action>0
            sel_is_gold_b  = (action > 0) & (sel_tok_b == gold)

            ctg_b = bw_b & (ref_top1 == gold)
            caw_b = ~bw_b & (ref_top1 != gold)

            gold_reg = tok_arr_t[gold.clamp(0, vs-1)]
            top1_reg = tok_arr_t[base_top1.clamp(0, vs-1)]
            sr_b = (gold_reg == top1_reg) & (gold_reg != unk_region)
            if sr_enabled:
                rlen = reg_arr_t.shape[0] - 1
                gs_sup = reg_arr_t[gold_reg.clamp(0, rlen)]
                t1_sup = reg_arr_t[top1_reg.clamp(0, rlen)]
                ss_b = (gs_sup == t1_sup) & (gs_sup != unk_super)
            else:
                ss_b = torch.zeros(b, dtype=torch.bool, device=device)

            sl = slice(s, e)
            action_arr[sl]    = action.cpu().numpy()
            sel_tok_arr[sl]   = sel_tok_b.cpu().numpy()
            gold_arr[sl]      = gold.cpu().numpy()
            base_arr[sl]      = base_top1.cpu().numpy()
            ref_top1_arr[sl]  = ref_top1.cpu().numpy()
            pool_gold_arr[sl] = gold_in_pool_b.cpu().numpy()
            ctg_arr[sl]       = ctg_b.cpu().numpy()
            caw_arr[sl]       = caw_b.cpu().numpy()
            bw_arr[sl]        = bw_b.cpu().numpy()
            sr_arr[sl]        = sr_b.cpu().numpy()
            ss_arr[sl]        = ss_b.cpu().numpy()
            sel_gold_arr[sl]  = sel_is_gold_b.cpu().numpy()
            kk = min(10, K)
            base_lgts_top10[sl] = lgt[:, :kk].cpu().numpy()
            ref_lgts_top10[sl]  = ref_lgt[:, :kk].cpu().numpy()
            topk_ids_store[sl]  = topk[:, :kk].cpu().numpy()
            cand_ids_store[sl]  = cand_ids.cpu().numpy()
            cand_lgts_store[sl] = cand_lgts.cpu().numpy()
            cand_scores_store[sl] = scores.cpu().numpy()

    row = {
        "N": N, "action": action_arr, "sel_tok": sel_tok_arr,
        "gold": gold_arr, "base": base_arr, "ref_top1": ref_top1_arr,
        "pool_gold": pool_gold_arr, "ctg": ctg_arr, "caw": caw_arr,
        "bw": bw_arr, "sr": sr_arr, "ss": ss_arr, "sel_gold": sel_gold_arr,
        "base_lgts_top10": base_lgts_top10, "ref_lgts_top10": ref_lgts_top10,
        "topk_ids_store": topk_ids_store,
        "cand_ids_store": cand_ids_store, "cand_lgts_store": cand_lgts_store,
        "cand_scores_store": cand_scores_store,
    }

    # Subset metrics
    subsets = {}
    defs = {
        "all":                   np.ones(N, bool),
        "base_correct":          ~bw_arr,
        "base_wrong_covered":    bw_arr & pool_gold_arr,
        "gold_in_pool":          pool_gold_arr,
        "gold_not_in_pool":      ~pool_gold_arr,
        "target_confuser":       bw_arr & pool_gold_arr & (sr_arr | ss_arr),
    }
    for sname, mask in defs.items():
        n = int(mask.sum())
        if n == 0:
            subsets[sname] = {"n": 0}
            continue
        m = lambda k: row[k][mask]  # noqa
        n_bw = int(m("bw").sum())
        n_bc = int((~m("bw")).sum())
        n_pg = int(m("pool_gold").sum())
        n_app= int((m("action") > 0).sum())
        subsets[sname] = {
            "n": n,
            "apply_rate":                    _rate(m("action") > 0, n),
            "noop_rate":                     _rate(m("action") == 0, n),
            "base_top1_acc":                 _rate(~m("bw"), n),
            "refined_top1_acc":              _rate(m("ref_top1") == m("gold"), n),
            "top1_acc_gain":                 _rate(m("ref_top1") == m("gold"), n) - _rate(~m("bw"), n),
            "changed_to_gold_rate":          _rate(m("ctg"), n),
            "changed_away_rate":             _rate(m("caw"), n),
            "selected_gold_rate_on_bwcov":   _rate(m("sel_gold")[m("bw") & m("pool_gold")],
                                                   int((m("bw") & m("pool_gold")).sum()))
                                             if (m("bw") & m("pool_gold")).any() else float("nan"),
            "selected_gold_rate_in_pool":    _rate(m("sel_gold")[m("pool_gold")], n_pg) if n_pg > 0 else float("nan"),
            "false_apply_on_base_correct":   _rate((m("action") > 0)[~m("bw")], n_bc) if n_bc > 0 else float("nan"),
            "changed_away_on_base_correct":  _rate(m("caw")[~m("bw")], n_bc) if n_bc > 0 else float("nan"),
        }

    # Checkpoint score
    sm  = subsets.get("all", {})
    ctg = sm.get("changed_to_gold_rate", float("nan"))
    caw = sm.get("changed_away_rate",    float("nan"))
    sip = subsets.get("gold_in_pool", {}).get("selected_gold_rate_in_pool", float("nan"))
    if not any(math.isnan(v) for v in [ctg, caw, sip]):
        chk_score = ctg - 2.0 * caw + 0.5 * sip
    else:
        chk_score = float("nan")
    subsets["_checkpoint_score"] = chk_score

    model.train()
    return subsets, row


# ─────────────────────────────────────────────────────────────────────────────
# Full-vocab eval
# ─────────────────────────────────────────────────────────────────────────────

def eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                    unk_region, unk_super, sr_enabled, args, device):
    try:
        model.eval()
        N, K = val_data["h_prime"].shape[0], val_data["topk_ids"].shape[1]
        VS   = model.token_emb_weight.shape[0]
        BSZ  = args.eval_batch_size
        P    = args.candidate_pool_size
        md   = args.margin_delta
        tot_nll_b = tot_nll_r = tot_acc_b = tot_acc_r = 0.0

        with torch.no_grad():
            for s in range(0, N, BSZ):
                e    = min(s + BSZ, N)
                b    = e - s
                ar_b = torch.arange(b, device=device)

                hp   = val_data["h_prime"][s:e].to(device)
                topk = val_data["topk_ids"][s:e].to(device)
                lgt  = val_data["topk_lgt"][s:e].to(device)
                gold = val_data["gold"][s:e].to(device)

                cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
                    topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
                    P, args.candidate_filter, device)

                base_ids = topk[:, 0]
                scores   = model(hp, base_ids, lgt, cand_ids, cand_lgts, cand_rnks,
                                 cand_regs, cand_sups)
                action   = scores.argmax(1)
                apply_mask = action > 0

                fv_base = hp @ model.token_emb_weight.T         # [b, VS]
                fv_ref  = fv_base.clone()

                if apply_mask.any():
                    pool_idx = (action - 1).clamp(0, P - 1)
                    sel_tok  = cand_ids[ar_b, pool_idx]
                    base_tok = topk[:, 0]
                    sel_tok_clamp  = sel_tok.clamp(0, VS - 1)
                    base_tok_clamp = base_tok.clamp(0, VS - 1)
                    app_f = apply_mask.float() * 0.5 * md
                    fv_ref[ar_b, sel_tok_clamp]  += app_f
                    fv_ref[ar_b, base_tok_clamp] -= app_f

                gs_v = gold.clamp(0, VS - 1)
                tot_nll_b += F.cross_entropy(fv_base, gs_v, reduction="sum").item()
                tot_nll_r += F.cross_entropy(fv_ref,  gs_v, reduction="sum").item()
                tot_acc_b += (fv_base.argmax(1) == gs_v).sum().item()
                tot_acc_r += (fv_ref.argmax(1)  == gs_v).sum().item()

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
        return {"full_vocab_eval_failed": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Example reports
# ─────────────────────────────────────────────────────────────────────────────

def _decode_tok(tid, tokenizer):
    try: return f"`{tokenizer.decode([int(tid)])}`"
    except: return str(tid)


def _fmt_example(ex_i, ri, val_data, row, tok_arr_t, unk_region, tokenizer, args):
    gold   = int(row["gold"][ri])
    base   = int(row["base"][ri])
    ref_t1 = int(row["ref_top1"][ri])
    act    = int(row["action"][ri])
    sel_t  = int(row["sel_tok"][ri])
    vs     = tok_arr_t.shape[0]

    def reg(tid): return str(int(tok_arr_t[min(int(tid), vs-1)].item()))
    def d(tid): return _decode_tok(tid, tokenizer)

    lines = [f"### Example {ex_i}  (row={ri})\n"]
    if val_data.get("ids") is not None:
        try:
            ctx = tokenizer.decode(val_data["ids"][ri][-64:].tolist(), skip_special_tokens=False)
            lines.append(f"**Context:** `{ctx}`\n")
        except Exception:
            pass
    lines.append(f"**Gold:** {d(gold)} id={gold} region={reg(gold)}")
    lines.append(f"**Base top-1:** {d(base)} id={base} region={reg(base)}")
    lines.append(f"**Refined top-1:** {d(ref_t1)} id={ref_t1}")
    lines.append(f"**Action:** {'NO_OP' if act == 0 else f'candidate {act-1} = {d(sel_t)} id={sel_t}'}\n")

    P = args.candidate_pool_size
    cands  = row["cand_ids_store"][ri]
    clgts  = row["cand_lgts_store"][ri]
    cscrs  = row["cand_scores_store"][ri]   # [P+1]
    base_lgts = row["base_lgts_top10"][ri]
    ref_lgts  = row["ref_lgts_top10"][ri]
    topk_ids  = row["topk_ids_store"][ri]

    lines.append("| j | Token | ID | Region | Base logit | Pool score | is_gold |")
    lines.append("|---|-------|----|--------|------------|------------|---------|")
    lines.append(f"| - | (NO_OP) | - | - | - | {cscrs[0]:.3f} | - |")
    for j in range(min(P, 8)):
        tid = int(cands[j])
        try: ts = tokenizer.decode([tid])
        except: ts = str(tid)
        mark = "✓" if tid == gold else ""
        lines.append(f"| {j+1} | `{ts}` | {tid} | {reg(tid)} | {clgts[j]:.3f} | {cscrs[j+1]:.3f} | {mark} |")
    lines.append("")
    lines.append("**Base/Ref top-10:**")
    lines.append("| Rank | Token | ID | Base lgt | Ref lgt |")
    lines.append("|------|-------|----|----------|---------|")
    for r in range(min(10, len(topk_ids))):
        tid = int(topk_ids[r])
        try: ts = tokenizer.decode([tid])
        except: ts = str(tid)
        mark = "✓" if tid == gold else ""
        lines.append(f"| {r+1} | `{ts}` | {tid} | {base_lgts[r]:.3f} | {ref_lgts[r]:.3f} | {mark}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_example_reports(val_data, row, tok_arr_t, unk_region, tokenizer, args, run_dir):
    rng  = np.random.default_rng(42)
    n    = args.num_examples
    ctg  = row["ctg"]; caw = row["caw"]
    bw   = row["bw"];  act = row["action"]; sg = row["sel_gold"]
    pg   = row["pool_gold"]

    buckets = {
        "changed_to_gold":      np.where(ctg)[0],
        "changed_away":         np.where(caw)[0],
        "selected_gold":        np.where((act > 0) & sg)[0],
        "selected_wrong":       np.where((act > 0) & ~sg & bw)[0],
        "noop_correct":         np.where((act == 0) & ~bw)[0],
        "false_apply":          np.where((act > 0) & ~bw)[0],
    }
    kw = dict(val_data=val_data, row=row, tok_arr_t=tok_arr_t.cpu(),
              unk_region=unk_region, tokenizer=tokenizer, args=args)
    for bname, indices in buckets.items():
        if len(indices) > n:
            indices = rng.choice(indices, n, replace=False)
        path = os.path.join(run_dir, f"examples_{bname}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {bname.replace('_', ' ').title()}\n\n_{len(indices)} examples_\n\n---\n\n")
            for ei, ri in enumerate(indices):
                f.write(_fmt_example(ei+1, int(ri), **kw))
                f.write("---\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(run_dir, args, pretrain_bl, final_subsets, best_step, fv_final):
    def f(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return "nan"
        if isinstance(v, float): return f"{v:.4f}"
        return str(v) if v is not None else "nan"

    sa = final_subsets.get("all", {})
    sb = final_subsets.get("base_correct", {})
    sw = final_subsets.get("base_wrong_covered", {})
    st = final_subsets.get("target_confuser", {})
    si = final_subsets.get("gold_in_pool", {})

    ctg = sa.get("changed_to_gold_rate", float("nan"))
    caw = sa.get("changed_away_rate",    float("nan"))
    sip = si.get("selected_gold_rate_in_pool", float("nan"))
    fvg = fv_final.get("full_vocab_gain", float("nan")) if fv_final else float("nan")

    bl_ctg  = pretrain_bl.get("baseline_oracle_pool_ctg", float("nan"))
    bl_r2   = pretrain_bl.get("baseline_rank2_ctg",       float("nan"))
    bl_r2c  = pretrain_bl.get("baseline_rank2_caw",       float("nan"))

    lines = ["# Multi-Candidate Selector V1\n"]
    lines.append(f"**Run:** `{args.run_name}`  |  **Steps:** {args.steps}  "
                 f"|  **candidate_filter:** `{args.candidate_filter}`  "
                 f"|  **pool_size:** {args.candidate_pool_size}\n")

    lines.append("## Reference Points\n")
    lines.append("| Experiment | sel_gold_bwcov / ctg | fv_gain |")
    lines.append("|-----------|---------------------|---------|")
    lines.append("| KNN V1 (knn_sel_gold_bwcov) | 0.0070 | -0.0116 |")
    lines.append("| KNN V1 rank2 baseline | 0.1452 | — |")
    lines.append("| V3 oracle pair corr (target_ctg) | 0.2407 | +0.1584 |\n")

    lines.append("## Pre-training Baselines\n")
    for k, v in pretrain_bl.items():
        if v is not None:
            lines.append(f"  {k:45s} = {f(v)}")
    lines.append("")

    lines.append("## Final Metrics\n")
    lines.append("### all\n")
    for k in ["n","apply_rate","noop_rate","base_top1_acc","refined_top1_acc",
              "top1_acc_gain","changed_to_gold_rate","changed_away_rate"]:
        lines.append(f"  {k:45s} = {f(sa.get(k, float('nan')))}")
    lines.append("")
    lines.append("### gold_in_pool\n")
    for k in ["n","selected_gold_rate_in_pool","apply_rate"]:
        lines.append(f"  {k:45s} = {f(si.get(k, float('nan')))}")
    lines.append("")
    lines.append("### base_correct (no-harm)\n")
    for k in ["n","false_apply_on_base_correct","changed_away_on_base_correct"]:
        lines.append(f"  {k:45s} = {f(sb.get(k, float('nan')))}")
    lines.append("")
    lines.append("### target_confuser\n")
    for k in ["n","changed_to_gold_rate","changed_away_rate","selected_gold_rate_in_pool"]:
        lines.append(f"  {k:45s} = {f(st.get(k, float('nan')))}")
    lines.append("")

    if fv_final:
        lines.append("## Full-Vocab\n")
        for k, v in fv_final.items():
            lines.append(f"  {k:45s} = {f(v)}")
        lines.append("")

    lines.append("## Analysis\n")
    def yn(cond, yes="✅", no="⚠️"): return yes if cond else no

    beats_r2 = (not math.isnan(ctg) and not math.isnan(bl_r2) and ctg > bl_r2)
    lines.append(f"**1. Beats rank2 (ctg={f(ctg)} vs rank2={f(bl_r2)})?**  {yn(beats_r2, 'Yes', 'No')}")

    no_harm = (not math.isnan(float(sb.get('changed_away_on_base_correct', float('nan'))))
               and float(sb.get('changed_away_on_base_correct', 1.0)) < 0.005)
    lines.append(f"**2. NO_OP protects base-correct (caw_bc={f(sb.get('changed_away_on_base_correct', float('nan')))} < 0.005)?**  {yn(no_harm, 'Yes', 'No')}")

    better = (not math.isnan(ctg) and not math.isnan(caw) and ctg > caw)
    lines.append(f"**3. ctg > caw ({f(ctg)} > {f(caw)})?**  {yn(better, 'Yes', 'No')}")

    pos_fv = (not math.isnan(fvg) and fvg >= 0)
    lines.append(f"**4. Positive fv_gain ({f(fvg)})?**  {yn(pos_fv, 'Yes', 'No')}")

    oracle_rec = float("nan")
    if not math.isnan(bl_ctg) and bl_ctg > 0 and not math.isnan(ctg):
        oracle_rec = ctg / bl_ctg
    lines.append(f"**5. Oracle recovery ({f(ctg)} / oracle={f(bl_ctg)} = {f(oracle_rec)})?**  "
                 + yn(not math.isnan(oracle_rec) and oracle_rec > 0.3, "Partial+", "Low"))

    lines.append(f"**6. Better than KNN (ctg={f(ctg)} vs 0.0070)?**  "
                 + yn(not math.isnan(ctg) and ctg > 0.007, "Yes", "No"))
    lines.append("")
    lines.append(f"Best checkpoint: step={best_step}")
    lines.append("")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Training data helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_train_labels(data, tok_arr_t, reg_arr_t, unk_region, unk_super,
                       sr_enabled, args, device):
    N = data["h_prime"].shape[0]
    BSZ = 4096
    all_labels   = []
    all_pool_has = []
    n_pos_lbl = n_noop_lbl = n_ign_lbl = 0
    for s in range(0, N, BSZ):
        e    = min(s + BSZ, N)
        topk = data["topk_ids"][s:e].to(device)
        lgt  = data["topk_lgt"][s:e].to(device)
        gold = data["gold"][s:e].to(device)

        cand_ids, _, _, _, _ = build_candidate_pool(
            topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
            args.candidate_pool_size, args.candidate_filter, device)

        labels, _, pool_has_gold = assign_labels(
            gold, topk, cand_ids, args.uncovered_policy)

        all_labels.append(labels.cpu())
        all_pool_has.append(pool_has_gold.cpu())
        n_pos_lbl  += int((labels > 0).sum())
        n_noop_lbl += int((labels == 0).sum())
        n_ign_lbl  += int((labels == -1).sum())

    labels_all = torch.cat(all_labels, 0)
    pool_has_gold_all = torch.cat(all_pool_has, 0)
    print(f"  train labels: noop={n_noop_lbl}  candidate={n_pos_lbl}  ignore={n_ign_lbl}")
    return labels_all, pool_has_gold_all


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
        raise RuntimeError(f"Run dir exists: {run_dir}. Choose a different --run_name.")
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
                             args.max_val_rows, load_ids=True)

    print("\n[model] Building MultiCandidateSelector...")
    n_regions = unk_region; n_supers = unk_super if sr_enabled else 1
    model = MultiCandidateSelector(
        token_emb_weight=tok_w.to(device),
        tok_arr=tok_arr_np, reg_arr=reg_arr_np,
        d_model=d_model, n_regions=n_regions, n_supers=n_supers,
        sr_enabled=sr_enabled, unk_region=unk_region, unk_super=unk_super,
        top_k=args.top_k, pool_size=args.candidate_pool_size,
        region_emb_dim=args.region_emb_dim, super_emb_dim=args.super_emb_dim,
        hidden_dim=args.hidden_dim, n_layers=args.layers, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    print("\n[pretrain baseline] Evaluating baselines on val...")
    pretrain_bl, _, _, _, _, _, _ = run_baselines(
        val_data, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
        args, device)
    for k, v in pretrain_bl.items():
        if v is not None:
            vf = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k:45s} = {vf}")

    print("\n[train labels] Building train labels...")
    train_labels, train_pool_has_gold = build_train_labels(
        train_data, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled, args, device)
    pretrain_bl["train_gold_in_pool_rate"] = float(train_pool_has_gold.float().mean())

    with open(os.path.join(run_dir, "pretrain_baseline.json"), "w") as f:
        json.dump(_json_safe(pretrain_bl), f, indent=2)

    config = vars(args).copy()
    config.update({"d_model": d_model, "vocab_size": vocab_size,
                   "sr_enabled": sr_enabled, "n_params": n_params})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None

    # Build sampling index: only rows with valid labels (>= 0)
    valid_mask = train_labels >= 0
    valid_idx  = valid_mask.nonzero(as_tuple=False).squeeze(1)
    N_valid    = len(valid_idx)
    print(f"\n  valid training rows: {N_valid:,}")
    if N_valid == 0:
        raise RuntimeError("No valid training rows found.")

    perm  = torch.randperm(N_valid)
    pos   = 0

    train_log = os.path.join(run_dir, "train_log.csv")
    eval_log  = os.path.join(run_dir, "eval_log.csv")
    sub_log   = os.path.join(run_dir, "subset_eval_log.csv")
    hdr_t = hdr_e = hdr_s = False

    best_score = -float("inf"); best_step = 0
    step = 0; t0 = time.time()
    optimizer.zero_grad()

    print(f"\n[train] steps={args.steps}  batch={args.batch_size}  lr={args.lr}\n")

    while step < args.steps:
        if pos + args.batch_size > N_valid:
            perm = torch.randperm(N_valid); pos = 0
        batch_local = perm[pos:pos + args.batch_size]; pos += args.batch_size
        batch_global = valid_idx[batch_local]

        hp   = train_data["h_prime"][batch_global].to(device)
        topk = train_data["topk_ids"][batch_global].to(device)
        lgt  = train_data["topk_lgt"][batch_global].to(device)
        gold = train_data["gold"][batch_global].to(device)
        lbl  = train_labels[batch_global].to(device)

        cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
            topk, lgt, tok_arr_t, reg_arr_t, unk_region, unk_super, sr_enabled,
            args.candidate_pool_size, args.candidate_filter, device)

        # Re-compute labels on fresh candidate pool (in case filter produces diff pool)
        lbl, _, _ = assign_labels(gold, topk, cand_ids, args.uncovered_policy)

        base_ids = topk[:, 0]

        with _amp_ctx(args.amp):
            scores = model(hp, base_ids, lgt, cand_ids, cand_lgts, cand_rnks,
                           cand_regs, cand_sups)
            total_loss, ld = compute_loss(
                scores, lbl, lgt, topk, cand_ids, gold,
                args.margin_delta, args.noop_weight, args.candidate_weight,
                args.lambda_margin_loss, args.target_margin, device)
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
                  f"n_valid={ld['n_valid']}  t={time.time()-t0:.0f}s")

        _write_csv_row(train_log, {"step": step, **ld}, header=not hdr_t); hdr_t = True

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n[eval] step={step}")
            subsets, row_res = evaluate(
                model, val_data, tok_arr_t, reg_arr_t,
                unk_region, unk_super, sr_enabled, args, device)

            fv = {}
            if args.eval_full_vocab:
                fv = eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                                     unk_region, unk_super, sr_enabled, args, device)

            sa = subsets.get("all", {}); si = subsets.get("gold_in_pool", {})
            score = subsets.get("_checkpoint_score", float("nan"))
            print(f"  [all]  ctg={sa.get('changed_to_gold_rate',float('nan')):.4f}  "
                  f"caw={sa.get('changed_away_rate',float('nan')):.4f}  "
                  f"apply={sa.get('apply_rate',float('nan')):.4f}")
            print(f"  [pool] sel_gold={si.get('selected_gold_rate_in_pool',float('nan')):.4f}")
            if fv:
                print(f"  fv_gain={fv.get('full_vocab_gain',float('nan')):.4f}")
            print(f"  score={score:.4f}" if not math.isnan(score) else "  score=nan")

            torch.save({"step": step, "model": model.state_dict(),
                        "args": vars(args)},
                       os.path.join(run_dir, "latest_selector.pt"))

            if not math.isnan(score) and score > best_score:
                best_score = score; best_step = step
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)},
                           os.path.join(run_dir, "best_selector.pt"))
                with open(os.path.join(run_dir, "best_metrics.json"), "w") as fp:
                    json.dump(_json_safe({"step": step, "score": score,
                                         "subsets": subsets, "fv": fv}), fp, indent=2)
                print(f"  [best] step={step}  score={score:.4f}")

            ev = {"step": step, **fv}
            for sn, sm in subsets.items():
                if sn.startswith("_"): continue
                for k, v in sm.items(): ev[f"{sn}_{k}"] = v
            _write_csv_row(eval_log, ev, header=not hdr_e); hdr_e = True
            for sn, sm in subsets.items():
                if sn.startswith("_"): continue
                _write_csv_row(sub_log, {"step": step, "subset": sn, **sm},
                               header=not hdr_s); hdr_s = True
            print()

    print("[final eval]")
    final_subsets, final_row = evaluate(
        model, val_data, tok_arr_t, reg_arr_t,
        unk_region, unk_super, sr_enabled, args, device)
    fv_final = {}
    if args.eval_full_vocab:
        fv_final = eval_full_vocab(model, val_data, tok_arr_t, reg_arr_t,
                                   unk_region, unk_super, sr_enabled, args, device)

    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump(_json_safe({"step": args.steps,
                              "subsets": final_subsets, "fv": fv_final}), f, indent=2)

    try:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            tokenizer = type("FT", (), {"decode": lambda s, ids, **kw: str(ids)})()
        write_example_reports(val_data, final_row, tok_arr_t.cpu(), unk_region,
                              tokenizer, args, run_dir)
    except Exception as exc:
        print(f"[warn] example reports failed: {exc}")

    write_report(run_dir, args, pretrain_bl, final_subsets, best_step, fv_final)

    print(f"\n{'='*60}")
    print(f" Multi-Candidate Selector V1 complete.")
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
    p.add_argument("--output_root",         default="runs/multicandidate_selector_v1")
    p.add_argument("--run_name",            default="noop_candidate_selector_v1")
    p.add_argument("--top_k",              type=int,   default=256)
    p.add_argument("--candidate_pool_size", type=int,   default=32)
    p.add_argument("--candidate_filter",    default="top_rank",
                   choices=["top_rank","same_region_or_superregion",
                            "same_region_only","same_superregion_only"])
    p.add_argument("--uncovered_policy",    default="ignore",
                   choices=["ignore", "noop"])
    p.add_argument("--architecture",        default="mlp", choices=["mlp", "transformer"])
    p.add_argument("--hidden_dim",          type=int,   default=256)
    p.add_argument("--layers",              type=int,   default=3)
    p.add_argument("--dropout",             type=float, default=0.1)
    p.add_argument("--region_emb_dim",      type=int,   default=64)
    p.add_argument("--super_emb_dim",       type=int,   default=32)
    p.add_argument("--batch_size",          type=int,   default=128)
    p.add_argument("--grad_accum_steps",    type=int,   default=1)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--steps",               type=int,   default=5000)
    p.add_argument("--eval_every",          type=int,   default=500)
    p.add_argument("--eval_batch_size",     type=int,   default=256)
    p.add_argument("--grad_clip",           type=float, default=1.0)
    p.add_argument("--noop_weight",         type=float, default=0.5)
    p.add_argument("--candidate_weight",    type=float, default=1.0)
    p.add_argument("--lambda_margin_loss",  type=float, default=0.5)
    p.add_argument("--target_margin",       type=float, default=0.0)
    p.add_argument("--margin_delta",        type=float, default=1.0)
    p.add_argument("--eval_full_vocab",     action="store_true")
    p.add_argument("--amp",                 action="store_true")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--max_train_rows",      type=int,   default=None)
    p.add_argument("--max_val_rows",        type=int,   default=None)
    p.add_argument("--num_examples",        type=int,   default=40)
    return p.parse_args()


if __name__ == "__main__":
    main()
