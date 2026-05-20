#!/usr/bin/env python3
"""
Candidate-set transformer refiner (CTF, variant=CandidateTransformer).

Instead of independently scoring each candidate with an MLP, builds a short
sequence over a selected subset of top-M candidates plus context/router/memory
summary tokens, then runs a small Pre-LN transformer to produce score deltas.

Candidates outside the selected top-M receive delta=0; the scatter operation
returns the full candidate score tensor unchanged for those positions.

Sequence layout (length = 3 + M):
  [CTX] [ROUTER] [MEM] [CAND_0] ... [CAND_M-1]

Candidate selection: topm_no_gold (inference-valid) by default. Gold is NEVER
consulted for candidate selection in training or eval. select_candidates_topm()
offers oracle modes for diagnostics only; oracle results must never be used for
best-checkpoint selection or reported as canonical metrics.

Robust init: out.weight=zeros, out.bias=zeros, residual_scale=1.0 → delta=0
at step 0. Gradients flow through inner transformer weights from step 1.

Canonical eval requirements (enforced at every full-val eval):
  fingerprint  = f57cabcdc46d69ce
  num_examples = 239,362
  num_covered  = 227,017
  coverage     = 0.948425

Usage:
    python scripts/train_candidate_transformer_refiner.py \\
        --small_ckpt runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --train_dir  runs/path_refiner_clean/data/train_hgrid_K24 \\
        --val_dir    runs/path_refiner_clean/data/val_hgrid_K24 \\
        --super_map  runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --official_baseline runs/path_refiner_clean/baselines/saved_candidate_baseline.json \\
        --output_dir runs/path_refiner_candidate_transformer/variant_CTF_boundary_M256 \\
        --train_filter boundary --gate_filter boundary \\
        --selected_M 256 --refiner_dim 256 --num_layers 2 --num_heads 4 \\
        --steps 10000 --eval_every 1000 --device cuda
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.train_clean_path_refiner import (
    _aggregate_eval_stats,
    check_baseline_match,
    check_init_identity,
    load_r2s,
    make_infinite,
    SPLIT_NAMES,
    TYPE_NAMES,
    canonical_eval_refiner,
)
from scripts.train_hard_position_refiner import (
    compute_filter_mask,
    EVAL_SUBSETS,
    FilteredShardStreamDataset,
    run_filter_audit,
)


# ── Shared fine-prob helpers (standalone so evals can call without a model) ───

def _build_fine_probs(topk_reg: torch.Tensor, topk_prb: torch.Tensor,
                      n_fine: int) -> torch.Tensor:
    B     = topk_reg.size(0)
    p     = torch.zeros(B, n_fine, device=topk_reg.device)
    valid = (topk_reg >= 0).float()
    reg_c = topk_reg.clamp(min=0).long()
    p.scatter_add_(1, reg_c, topk_prb.float() * valid)
    return p


def _build_fine_indicator(topk_reg: torch.Tensor, n_fine: int) -> torch.Tensor:
    B, K  = topk_reg.shape
    ind   = torch.zeros(B, n_fine, dtype=torch.bool, device=topk_reg.device)
    valid = topk_reg >= 0
    flat_b = torch.arange(B, device=topk_reg.device).unsqueeze(1).expand(-1, K)
    ind[flat_b[valid], topk_reg[valid].long()] = True
    return ind


# ── Standalone candidate selector ────────────────────────────────────────────

def select_candidates_topm(
    base: torch.Tensor,       # (B, C)  -inf at invalid positions
    cand_mask: torch.Tensor,  # (B, C)  bool
    selected_M: int,
    mode: str = "topm_no_gold",
    gold_idx: Optional[torch.Tensor] = None,   # (B,)  required for oracle modes
    covered: Optional[torch.Tensor] = None,    # (B,)  required for oracle modes
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Canonical candidate selector.

    mode="topm_no_gold"            — top-M by base logit only. Gold never consulted.
                                     Inference-valid. DEFAULT for all real evals.
    mode="oracle_force_gold_last"  — covered gold not in top-M → forced into slot[-1].
                                     DIAGNOSTIC ONLY. NOT INFERENCE VALID.
    mode="oracle_force_gold_random"— like oracle_force_gold_last but random slot.
                                     DIAGNOSTIC ONLY. NOT INFERENCE VALID.

    Returns:
        sel_idx:     (B, M)  indices into the C-dim candidate space
        sel_mask:    (B, M)  True = valid candidate slot
        gold_forced: (B,)    True = gold was force-inserted (always False for topm_no_gold)
    """
    B, C   = base.shape
    M      = selected_M
    m_act  = min(M, C)
    device = base.device

    _, sel_raw = base.topk(m_act, dim=-1, largest=True, sorted=True)  # (B, m_act)
    msk_raw    = cand_mask.gather(1, sel_raw)                          # (B, m_act)

    if m_act < M:
        pad      = torch.zeros(B, M - m_act, dtype=torch.long, device=device)
        pad_msk  = torch.zeros(B, M - m_act, dtype=torch.bool, device=device)
        sel_idx  = torch.cat([sel_raw, pad],     dim=1)
        sel_mask = torch.cat([msk_raw, pad_msk], dim=1)
    else:
        sel_idx  = sel_raw
        sel_mask = msk_raw

    gold_forced = torch.zeros(B, dtype=torch.bool, device=device)

    if mode == "topm_no_gold":
        pass  # inference-valid — gold is never consulted

    elif mode in ("oracle_force_gold_last", "oracle_force_gold_random"):
        if gold_idx is None or covered is None:
            raise ValueError(
                f"select_candidates_topm: mode={mode!r} requires gold_idx and covered"
            )
        gold_in_sel = (sel_idx == gold_idx.unsqueeze(1)).any(dim=1)
        gold_forced = covered & ~gold_in_sel
        if gold_forced.any():
            fb = gold_forced.nonzero(as_tuple=True)[0]
            if mode == "oracle_force_gold_last":
                sel_idx[fb, -1] = gold_idx[fb]
                sel_mask[fb, -1] = True
            else:
                _rng = rng or random
                for b in fb.tolist():
                    slot = _rng.randint(0, M - 1)
                    sel_idx[b, slot] = int(gold_idx[b])
                    sel_mask[b, slot] = True

    else:
        raise ValueError(
            f"select_candidates_topm: unknown mode={mode!r}. "
            "Valid: topm_no_gold | oracle_force_gold_last | oracle_force_gold_random"
        )

    return sel_idx, sel_mask, gold_forced


# ── Model ──────────────────────────────────────────────────────────────────────

class CandidateSetTransformerRefiner(nn.Module):
    """
    Small transformer over a selected top-M candidate subset.

    Sequence: [CTX] [ROUTER] [MEM] [CAND_0 .. CAND_{M-1}]
    Output: delta per selected candidate, scattered into full candidate space.
    delta = 0 for candidates outside selected M.

    Init: out.weight=0, out.bias=0, residual_scale=1.0 → all deltas are 0 at
    step 0 regardless of transformer output.
    """

    def __init__(self, d_model: int, n_fine: int, n_super: int,
                 refiner_dim: int = 256, num_layers: int = 2,
                 num_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.0, selected_M: int = 256) -> None:
        super().__init__()
        self.d_model    = d_model
        self.n_fine     = n_fine
        self.n_super    = n_super
        self.refiner_dim = refiner_dim
        self.selected_M  = selected_M

        # Context token: compress backbone hidden state
        self.ctx_proj    = nn.Linear(d_model, refiner_dim)

        # Router / memory summary tokens: [margin, top-4 probs] → D
        self.router_proj = nn.Sequential(nn.Linear(5, refiner_dim), nn.GELU())
        self.mem_proj    = nn.Sequential(nn.Linear(5, refiner_dim), nn.GELU())

        # Candidate token components
        self.token_proj  = nn.Linear(d_model, refiner_dim, bias=False)
        self.fine_emb    = nn.Embedding(n_fine  + 1, refiner_dim, padding_idx=n_fine)
        self.super_emb   = nn.Embedding(n_super + 1, refiner_dim, padding_idx=n_super)
        # Scalar features: [base_logit, rank_norm, r_prob, m_prob, is_r, is_m]
        self.score_proj  = nn.Linear(6, refiner_dim)

        # Positional / type embeddings
        # types: 0=CTX  1=ROUTER  2=MEM  3=CAND
        self.type_emb    = nn.Embedding(4, refiner_dim)
        self.rank_emb    = nn.Embedding(selected_M, refiner_dim)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=refiner_dim, nhead=num_heads,
            dim_feedforward=refiner_dim * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm_out    = nn.LayerNorm(refiner_dim)

        # Output head — zero-initialized so delta=0 at step 0
        self.out             = nn.Linear(refiner_dim, 1)
        self.residual_scale  = nn.Parameter(torch.tensor(1.0))

        self._last_sel_stats: Dict = {}
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.ctx_proj.weight,   std=0.02)
        nn.init.zeros_(self.ctx_proj.bias)
        nn.init.normal_(self.router_proj[0].weight, std=0.02)
        nn.init.zeros_(self.router_proj[0].bias)
        nn.init.normal_(self.mem_proj[0].weight,    std=0.02)
        nn.init.zeros_(self.mem_proj[0].bias)
        nn.init.normal_(self.token_proj.weight, std=0.02)
        nn.init.normal_(self.fine_emb.weight,   std=0.02)
        nn.init.normal_(self.super_emb.weight,  std=0.02)
        nn.init.normal_(self.score_proj.weight, std=0.01)
        nn.init.zeros_(self.score_proj.bias)
        nn.init.normal_(self.type_emb.weight,   std=0.02)
        nn.init.normal_(self.rank_emb.weight,   std=0.02)
        # Zero-init output head: delta=0 at step 0
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ── Candidate selection ──────────────────────────────────────────────────

    def select_candidates(
        self,
        base: torch.Tensor,       # (B, C)  -inf at invalid positions
        cand_mask: torch.Tensor,  # (B, C)  bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-M by base logit only — inference valid. Gold is never consulted.
        Returns sel_idx (B, M) and sel_mask (B, M).
        For oracle selection use select_candidates_topm() directly.
        """
        sel_idx, sel_mask, _ = select_candidates_topm(
            base, cand_mask, self.selected_M, mode="topm_no_gold"
        )
        return sel_idx, sel_mask

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        h_prime:    torch.Tensor,   # (B, d_model)
        cand_tok:   torch.Tensor,   # (B, C)
        cand_fine:  torch.Tensor,   # (B, C)
        cand_super: torch.Tensor,   # (B, C)
        cand_mask:  torch.Tensor,   # (B, C) bool
        token_emb_w: torch.Tensor,  # (vocab, d_model)
        r_topk_reg: torch.Tensor,   # (B, K)
        r_topk_prb: torch.Tensor,   # (B, K)
        m_topk_reg: torch.Tensor,   # (B, K)
        m_topk_prb: torch.Tensor,   # (B, K)
        r_margin:   torch.Tensor,   # (B,)
        m_margin:   torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C   = cand_tok.shape
        M      = self.selected_M
        D      = self.refiner_dim
        device = h_prime.device

        emb_w  = token_emb_w.float()

        # ── Base scores ──────────────────────────────────────────────────────
        tok_e    = F.embedding(cand_tok.clamp(min=0), emb_w)          # (B, C, d_model)
        base_raw = (h_prime.float().unsqueeze(1) * tok_e).sum(-1)     # (B, C)
        base     = base_raw.masked_fill(~cand_mask, float("-inf"))    # (B, C)

        # ── Candidate selection (topm_no_gold — inference valid) ─────────────
        sel_idx, sel_mask = self.select_candidates(base, cand_mask)
        self._last_sel_stats = {
            "gold_forced":     torch.zeros(B, dtype=torch.bool, device=device),
            "sel_valid_count": sel_mask.float().sum(1),
            "sel_idx":         sel_idx.detach(),
        }

        # ── Context token ────────────────────────────────────────────────────
        ctx_tok = self.ctx_proj(h_prime.float()).unsqueeze(1)                     # (B,1,D)
        ctx_tok = ctx_tok + self.type_emb.weight[0].view(1, 1, D)

        # ── Router summary token ─────────────────────────────────────────────
        r_prb_4 = r_topk_prb[:, :4].float()
        if r_prb_4.size(1) < 4:
            r_prb_4 = F.pad(r_prb_4, (0, 4 - r_prb_4.size(1)))
        r_feat  = torch.cat([r_margin.float().unsqueeze(-1), r_prb_4], dim=-1)   # (B, 5)
        r_tok   = self.router_proj(r_feat).unsqueeze(1)                           # (B,1,D)
        r_tok   = r_tok + self.type_emb.weight[1].view(1, 1, D)

        # ── Memory summary token ─────────────────────────────────────────────
        m_prb_4 = m_topk_prb[:, :4].float()
        if m_prb_4.size(1) < 4:
            m_prb_4 = F.pad(m_prb_4, (0, 4 - m_prb_4.size(1)))
        m_feat  = torch.cat([m_margin.float().unsqueeze(-1), m_prb_4], dim=-1)   # (B, 5)
        m_tok   = self.mem_proj(m_feat).unsqueeze(1)                              # (B,1,D)
        m_tok   = m_tok + self.type_emb.weight[2].view(1, 1, D)

        # ── Candidate tokens ─────────────────────────────────────────────────
        sel_clamped = sel_idx.clamp(min=0)  # safety clamp (sel_idx is always in [0,C))

        # Gather selected token IDs / fine / super
        sel_tok_id = cand_tok.clamp(min=0).gather(1, sel_clamped)   # (B, M)
        sel_fine   = cand_fine.gather(1,  sel_clamped)              # (B, M)
        sel_sup    = cand_super.gather(1, sel_clamped)              # (B, M)
        sel_base   = base_raw.gather(1,   sel_clamped)              # (B, M)

        # Token / region projections
        sel_tok_e   = F.embedding(sel_tok_id, emb_w)                # (B, M, d_model)
        tok_part    = self.token_proj(sel_tok_e.float())             # (B, M, D)
        fine_part   = self.fine_emb( sel_fine.clamp(min=0, max=self.n_fine))   # (B,M,D)
        super_part  = self.super_emb(sel_sup.clamp( min=0, max=self.n_super))  # (B,M,D)

        # Per-candidate router / memory probs and indicators
        p_r = _build_fine_probs(r_topk_reg, r_topk_prb, self.n_fine)   # (B,n_fine)
        p_m = _build_fine_probs(m_topk_reg, m_topk_prb, self.n_fine)   # (B,n_fine)
        in_r = _build_fine_indicator(r_topk_reg, self.n_fine)           # (B,n_fine)
        in_m = _build_fine_indicator(m_topk_reg, self.n_fine)           # (B,n_fine)

        sf_c = sel_fine.clamp(min=0, max=self.n_fine - 1)
        r_prob_sel = p_r.gather(1, sf_c)                               # (B, M)
        m_prob_sel = p_m.gather(1, sf_c)                               # (B, M)
        is_r_sel   = in_r.gather(1, sf_c).float()                      # (B, M)
        is_m_sel   = in_m.gather(1, sf_c).float()                      # (B, M)

        # Rank in [0, 1] (0 = highest base logit)
        rank_idx  = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)  # (B,M)
        rank_norm = rank_idx.float() / max(M - 1, 1)

        score_feat = torch.stack(
            [sel_base, rank_norm, r_prob_sel, m_prob_sel, is_r_sel, is_m_sel], dim=-1
        )                                                              # (B, M, 6)
        score_part = self.score_proj(score_feat)                       # (B, M, D)

        cand_feat = tok_part + fine_part + super_part + score_part     # (B, M, D)
        cand_feat = cand_feat + self.type_emb.weight[3].view(1, 1, D)
        cand_feat = cand_feat + self.rank_emb(rank_idx)                # (B, M, D)
        # Zero out embeddings for padding slots so they don't add noise
        cand_feat = cand_feat * sel_mask.float().unsqueeze(-1)

        # ── Sequence + key-padding mask ──────────────────────────────────────
        seq = torch.cat([ctx_tok, r_tok, m_tok, cand_feat], dim=1)    # (B, 3+M, D)
        # src_key_padding_mask: True = ignore this key position
        kpm = torch.cat([
            torch.zeros(B, 3, dtype=torch.bool, device=device),
            ~sel_mask,                                                  # (B, M)
        ], dim=1)                                                       # (B, 3+M)

        # ── Transformer ──────────────────────────────────────────────────────
        seq_out   = self.transformer(seq, src_key_padding_mask=kpm)   # (B, 3+M, D)
        cand_out  = self.norm_out(seq_out[:, 3:, :])                  # (B, M, D)

        # ── Output + scatter ─────────────────────────────────────────────────
        delta_sel = self.out(cand_out).squeeze(-1)                     # (B, M)
        delta_sel = delta_sel * self.residual_scale
        delta_sel = delta_sel * sel_mask.float()                       # zero invalid

        full_delta = torch.zeros(B, C, device=device)
        full_delta.scatter_(1, sel_clamped, delta_sel)

        scores = (base_raw + full_delta).masked_fill(~cand_mask, float("-inf"))
        return scores, base


# ── CTF-specific canonical eval (selection_mode=topm_no_gold, inference-valid) ─

@torch.no_grad()
def canonical_eval_ctf(
    model: CandidateSetTransformerRefiner,
    val_dir: str,
    tok_emb_w: torch.Tensor,
    r2s_np: np.ndarray,
    device,
    eval_batch_size: int = 64,
    official_baseline: Optional[Dict] = None,
    fail_on_mismatch: bool = False,
    variant_tag: str = "",
) -> Dict:
    """
    Full-val eval for CTF.
    selection_mode=topm_no_gold  eval_force_include_gold=false  [CANONICAL]
    Gold is never passed to the model. Asserts gold_force_included_rate == 0.
    Fingerprint/counts/coverage must match baseline.
    """
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt in {val_dir}")

    emb_w = tok_emb_w.float().to(device)
    model.eval()

    stats: Dict = defaultdict(lambda: [0.0, 0, 0])
    total_n = total_cov = 0
    sum_cand_counts = sum_gold_idx_cov = sum_gold_tok = 0
    dabs_sum = dabs_max = 0.0
    dabs_n = 0
    gold_forced_total = gold_forced_covered = 0

    for path in paths:
        shard    = torch.load(path, map_location="cpu", weights_only=True)
        N        = len(shard["covered"])
        has_type = "type_arr" in shard
        sp_all   = shard["split"].long()
        tp_all   = shard["type_arr"].long() if has_type else torch.zeros(N, dtype=torch.long)
        has_gtok = "gold_token" in shard

        cf_np    = shard["cand_fine"].numpy().astype(np.int64).clip(min=0)
        cs_np    = r2s_np[cf_np].astype(np.int64)
        cs_np[shard["cand_fine"].numpy() < 0] = 0
        cs_shard = torch.from_numpy(cs_np)

        for start in range(0, N, eval_batch_size):
            end     = min(start + eval_batch_size, N)
            h       = shard["h_prime"][start:end].float().to(device)
            ct      = shard["cand_tok"][start:end].long().to(device)
            cf      = shard["cand_fine"][start:end].long().to(device)
            cs      = cs_shard[start:end].long().to(device)
            g_idx   = shard["gold_cand_idx"][start:end].long().to(device)
            covered = shard["covered"][start:end].bool().to(device)
            sp      = sp_all[start:end]
            tp      = tp_all[start:end]
            B, C    = ct.shape

            cmask   = (ct >= 0)
            r_reg   = shard["router_topk_reg"][start:end].long().to(device)
            r_prb   = shard["router_topk_prb"][start:end].float().to(device)
            m_reg   = shard["mem_topk_reg"][start:end].long().to(device)
            m_prb   = shard["mem_topk_prb"][start:end].float().to(device)
            r_margin = shard["router_margin"][start:end].float().to(device)
            m_margin = shard["mem_margin"][start:end].float().to(device)

            scores, base_sc = model(
                h, ct, cf, cs, cmask, emb_w,
                r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
            )

            # selection_mode=topm_no_gold: gold_forced must always be 0
            gold_forced_total   += int(covered.sum())
            gold_forced_covered += int(model._last_sel_stats["gold_forced"].sum())

            # Delta diagnostics
            if cmask.any():
                dv = (scores - base_sc)[cmask]
                dabs_sum += float(dv.abs().sum())
                dabs_max  = max(dabs_max, float(dv.abs().max()))
                dabs_n   += int(cmask.sum())

            n_cov_b = int(covered.sum())
            if n_cov_b > 0:
                ar    = torch.arange(n_cov_b, device=device)
                gi_c  = g_idx[covered]
                lp    = F.log_softmax(scores[covered], dim=-1)
                ce_v  = -lp[ar, gi_c].cpu()
                ci    = torch.where(covered)[0].cpu()
                for j, i in enumerate(ci.tolist()):
                    key = (int(sp[i].item()), int(tp[i].item()))
                    stats[key][0] += float(ce_v[j])
                    stats[key][1] += 1
                sum_gold_idx_cov += int(gi_c.sum())

            for i in range(B):
                key = (int(sp[i].item()), int(tp[i].item()))
                stats[key][2] += 1

            total_n         += B
            total_cov       += n_cov_b
            sum_cand_counts += int(cmask.sum())
            if has_gtok:
                sum_gold_tok += int(shard["gold_token"][start:end].long().sum())

    model.train()

    results = _aggregate_eval_stats(stats)
    results["num_examples"]    = total_n
    results["num_covered"]     = total_cov
    results["mean_cand_count"] = sum_cand_counts / max(total_n, 1)
    results["delta_abs_mean"]  = dabs_sum / max(dabs_n, 1)
    results["delta_abs_max"]   = dabs_max

    fp_data = {
        "num_shards": len(paths), "total_n": total_n, "total_cov": total_cov,
        "sum_cand_counts": sum_cand_counts,
        "sum_gold_idx_cov": sum_gold_idx_cov,
        "sum_gold_tok": sum_gold_tok,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fp_data, sort_keys=True).encode()
    ).hexdigest()[:16]
    results["dataset_fingerprint"]    = fingerprint
    results["eval_mode"]              = "canonical_full_val"
    results["selection_mode"]         = "topm_no_gold"
    results["eval_force_include_gold"] = False

    gf_rate = gold_forced_covered / max(gold_forced_total, 1)
    results["gold_force_included_rate"] = gf_rate
    if gf_rate > 0.0:
        raise RuntimeError(
            f"canonical_eval_ctf [{variant_tag}]: gold_force_included_rate={gf_rate:.6f} > 0. "
            "This is a bug — selection_mode=topm_no_gold must never force gold in."
        )

    if official_baseline is not None:
        ref      = official_baseline
        fp_ok    = fingerprint == ref.get("dataset_fingerprint", "")
        n_ok     = total_n    == ref.get("num_examples", -1)
        nc_ok    = total_cov  == ref.get("num_covered",  -1)
        cov_diff = abs(results["coverage"] - ref.get("coverage", -1.0))
        cov_ok   = cov_diff < 1e-6
        ok       = fp_ok and n_ok and nc_ok and cov_ok
        if not ok:
            issues = []
            if not fp_ok:
                issues.append(f"fingerprint {fingerprint!r} != "
                              f"{ref.get('dataset_fingerprint','?')!r}")
            if not n_ok:
                issues.append(f"num_examples {total_n} != {ref.get('num_examples','?')}")
            if not nc_ok:
                issues.append(f"num_covered {total_cov} != {ref.get('num_covered','?')}")
            if not cov_ok:
                issues.append(f"coverage diff {cov_diff:.2e}")
            msg = (f"canonical_eval_ctf MISMATCH [{variant_tag}]: "
                   + ", ".join(issues)
                   + ". Model must not change coverage/candidates/fingerprint.")
            if fail_on_mismatch:
                raise RuntimeError(msg)
            print(f"  WARNING: {msg}")

    return results


# ── Gated global eval for CTF ─────────────────────────────────────────────────

@torch.no_grad()
def gated_global_eval_ctf(
    model: CandidateSetTransformerRefiner,
    val_dir: str,
    tok_emb_w: torch.Tensor,
    r2s_np: np.ndarray,
    device,
    gate_filter_name: str,
    filter_kwargs: Dict,
    official_baseline: Optional[Dict] = None,
    fail_on_mismatch: bool = False,
    variant_tag: str = "",
    eval_batch_size: int = 64,
) -> Dict:
    """
    Full-val pass with gating.  selection_mode=topm_no_gold  eval_force_include_gold=false
    gated_scores[b] = model_scores[b] if gate[b] else base_scores[b].
    Primary metric: gated_covered_nll. Asserts gold_force_included_rate == 0.
    Fingerprint/counts/coverage asserted vs official_baseline.
    """
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt in {val_dir}")

    emb_w = tok_emb_w.float().to(device)
    model.eval()

    total_n = total_cov = 0
    sum_cand_counts = sum_gold_idx_cov = sum_gold_tok = 0
    gated_ce = gated_nc = 0
    ig_m_ce = ig_b_ce = ig_nc = ig_nt = 0
    og_ce = og_nc = 0
    gold_forced_total = gold_forced_covered = 0

    for path in paths:
        shard   = torch.load(path, map_location="cpu", weights_only=True)
        N       = len(shard["covered"])
        has_gtk = "gold_token" in shard

        try:
            gate_shard = compute_filter_mask(shard, gate_filter_name, **filter_kwargs)
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"gated_global_eval_ctf gate mask failed: {e}") from e

        cf_np    = shard["cand_fine"].numpy().astype(np.int64).clip(min=0)
        cs_np    = r2s_np[cf_np].astype(np.int64)
        cs_np[shard["cand_fine"].numpy() < 0] = 0
        cs_shard = torch.from_numpy(cs_np)

        for start in range(0, N, eval_batch_size):
            end     = min(start + eval_batch_size, N)
            h       = shard["h_prime"][start:end].float().to(device)
            ct      = shard["cand_tok"][start:end].long().to(device)
            cf      = shard["cand_fine"][start:end].long().to(device)
            cs      = cs_shard[start:end].long().to(device)
            g_idx   = shard["gold_cand_idx"][start:end].long().to(device)
            covered = shard["covered"][start:end].bool().to(device)
            gate    = gate_shard[start:end].to(device)
            B, C    = ct.shape

            cmask   = (ct >= 0)
            tok_e   = F.embedding(ct.clamp(min=0), emb_w)
            base_r  = (h.unsqueeze(1) * tok_e).sum(-1)
            base_sc = base_r.masked_fill(~cmask, float("-inf"))

            if gate.any():
                r_reg    = shard["router_topk_reg"][start:end].long().to(device)
                r_prb    = shard["router_topk_prb"][start:end].float().to(device)
                m_reg    = shard["mem_topk_reg"][start:end].long().to(device)
                m_prb    = shard["mem_topk_prb"][start:end].float().to(device)
                r_margin = shard["router_margin"][start:end].float().to(device)
                m_margin = shard["mem_margin"][start:end].float().to(device)
                mdl_sc, _ = model(
                    h, ct, cf, cs, cmask, emb_w,
                    r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
                )
                gold_forced_total   += int(covered.sum())
                gold_forced_covered += int(model._last_sel_stats["gold_forced"].sum())
            else:
                mdl_sc = base_sc

            gate_3d  = gate.unsqueeze(1).expand(-1, C)
            gated_sc = torch.where(gate_3d, mdl_sc, base_sc)

            n_cov_b  = int(covered.sum())
            if n_cov_b > 0:
                ar   = torch.arange(n_cov_b, device=device)
                gi_c = g_idx[covered]

                lp_g = F.log_softmax(gated_sc[covered], dim=-1)
                gated_ce += float(-lp_g[ar, gi_c].sum())
                gated_nc += n_cov_b

                ig_mask = covered & gate
                n_ig    = int(ig_mask.sum())
                if n_ig > 0:
                    ar_ig  = torch.arange(n_ig, device=device)
                    gi_ig  = g_idx[ig_mask]
                    ig_m_ce += float(-F.log_softmax(mdl_sc[ig_mask], dim=-1)[ar_ig, gi_ig].sum())
                    ig_b_ce += float(-F.log_softmax(base_sc[ig_mask], dim=-1)[ar_ig, gi_ig].sum())
                    ig_nc   += n_ig

                og_mask = covered & ~gate
                n_og    = int(og_mask.sum())
                if n_og > 0:
                    ar_og  = torch.arange(n_og, device=device)
                    og_ce += float(-F.log_softmax(base_sc[og_mask], dim=-1)[ar_og, g_idx[og_mask]].sum())
                    og_nc += n_og

                sum_gold_idx_cov += int(gi_c.sum())

            ig_nt    += int(gate.sum())
            total_n  += B
            total_cov += n_cov_b
            sum_cand_counts += int(cmask.sum())
            if has_gtk:
                sum_gold_tok += int(shard["gold_token"][start:end].long().sum())

    model.train()

    fp_data = {
        "num_shards": len(paths), "total_n": total_n, "total_cov": total_cov,
        "sum_cand_counts": sum_cand_counts,
        "sum_gold_idx_cov": sum_gold_idx_cov,
        "sum_gold_tok": sum_gold_tok,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fp_data, sort_keys=True).encode()
    ).hexdigest()[:16]

    results = {
        "gated_covered_nll":     gated_ce  / max(gated_nc, 1),
        "inside_gate_model_nll": ig_m_ce   / max(ig_nc,    1),
        "inside_gate_base_nll":  ig_b_ce   / max(ig_nc,    1),
        "outside_gate_nll":      og_ce     / max(og_nc,    1),
        "gate_rate":             ig_nt     / max(total_n,  1),
        "covered_gate_rate":     ig_nc     / max(total_cov, 1),
        "coverage":              total_cov / max(total_n,  1),
        "dataset_fingerprint":   fingerprint,
        "num_examples":          total_n,
        "num_covered":           total_cov,
        "inside_gate_n_total":   ig_nt,
        "inside_gate_n_cov":     ig_nc,
        "gold_force_included_rate":  gold_forced_covered / max(gold_forced_total, 1),
        "selection_mode":            "topm_no_gold",
        "eval_force_include_gold":   False,
    }

    gf_rate = results["gold_force_included_rate"]
    if gf_rate > 0.0:
        raise RuntimeError(
            f"gated_global_eval_ctf [{variant_tag}]: gold_force_included_rate={gf_rate:.6f} > 0. "
            "This is a bug — selection_mode=topm_no_gold must never force gold in."
        )

    if official_baseline is not None:
        ref   = official_baseline
        fp_ok = fingerprint == ref.get("dataset_fingerprint", "")
        n_ok  = total_n    == ref.get("num_examples", -1)
        nc_ok = total_cov  == ref.get("num_covered",  -1)
        cd    = abs(results["coverage"] - ref.get("coverage", -1.0))
        ok    = fp_ok and n_ok and nc_ok and cd < 1e-6
        if not ok:
            issues = []
            if not fp_ok: issues.append(f"fingerprint {fingerprint!r}")
            if not n_ok:  issues.append(f"num_examples {total_n}")
            if not nc_ok: issues.append(f"num_covered {total_cov}")
            if cd >= 1e-6: issues.append(f"coverage diff {cd:.2e}")
            msg = (f"gated_global_eval_ctf MISMATCH [{variant_tag}]: "
                   + ", ".join(issues))
            if fail_on_mismatch:
                raise RuntimeError(msg)
            print(f"  WARNING: {msg}")

    return results


# ── Local subset eval for CTF ─────────────────────────────────────────────────

@torch.no_grad()
def local_subset_eval_ctf(
    model: CandidateSetTransformerRefiner,
    val_dir: str,
    tok_emb_w: torch.Tensor,
    r2s_np: np.ndarray,
    device,
    filter_kwargs: Dict,
    eval_batch_size: int = 64,
) -> Dict[str, Dict]:
    """One pass over full val — NLL + acc@1/5 for model and base across all EVAL_SUBSETS."""
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt in {val_dir}")

    emb_w = tok_emb_w.float().to(device)
    model.eval()

    acc = {s: {"n_total": 0, "n_cov": 0,
               "m_ce": 0.0, "b_ce": 0.0,
               "m_a1": 0, "m_a5": 0,
               "b_a1": 0, "b_a5": 0}
           for s in EVAL_SUBSETS}

    for path in paths:
        shard  = torch.load(path, map_location="cpu", weights_only=True)
        N      = len(shard["covered"])
        smasks = {}
        for sub in EVAL_SUBSETS:
            try:
                smasks[sub] = compute_filter_mask(shard, sub, **filter_kwargs)
            except (KeyError, ValueError):
                smasks[sub] = torch.zeros(N, dtype=torch.bool)

        cf_np    = shard["cand_fine"].numpy().astype(np.int64).clip(min=0)
        cs_np    = r2s_np[cf_np].astype(np.int64)
        cs_np[shard["cand_fine"].numpy() < 0] = 0
        cs_shard = torch.from_numpy(cs_np)

        for start in range(0, N, eval_batch_size):
            end     = min(start + eval_batch_size, N)
            h       = shard["h_prime"][start:end].float().to(device)
            ct      = shard["cand_tok"][start:end].long().to(device)
            cf      = shard["cand_fine"][start:end].long().to(device)
            cs      = cs_shard[start:end].long().to(device)
            g_idx   = shard["gold_cand_idx"][start:end].long().to(device)
            covered = shard["covered"][start:end].bool().to(device)
            B, C    = ct.shape

            cmask   = (ct >= 0)
            tok_e   = F.embedding(ct.clamp(min=0), emb_w)
            base_sc = (h.unsqueeze(1) * tok_e).sum(-1).masked_fill(~cmask, float("-inf"))

            r_reg    = shard["router_topk_reg"][start:end].long().to(device)
            r_prb    = shard["router_topk_prb"][start:end].float().to(device)
            m_reg    = shard["mem_topk_reg"][start:end].long().to(device)
            m_prb    = shard["mem_topk_prb"][start:end].float().to(device)
            r_margin = shard["router_margin"][start:end].float().to(device)
            m_margin = shard["mem_margin"][start:end].float().to(device)

            mdl_sc, _ = model(
                h, ct, cf, cs, cmask, emb_w,
                r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
            )

            k5 = min(5, C)
            for sub in EVAL_SUBSETS:
                bm    = smasks[sub][start:end].to(device)
                cov_s = covered & bm
                nc    = int(cov_s.sum())
                acc[sub]["n_total"] += int(bm.sum())
                acc[sub]["n_cov"]   += nc
                if nc == 0:
                    continue
                ar    = torch.arange(nc, device=device)
                gi_s  = g_idx[cov_s]
                gi_e  = gi_s.unsqueeze(1)

                acc[sub]["m_ce"] += float(-F.log_softmax(mdl_sc[cov_s], dim=-1)[ar, gi_s].sum())
                acc[sub]["b_ce"] += float(-F.log_softmax(base_sc[cov_s], dim=-1)[ar, gi_s].sum())

                m5 = mdl_sc[cov_s].topk(k5, dim=-1).indices
                b5 = base_sc[cov_s].topk(k5, dim=-1).indices
                acc[sub]["m_a1"] += int((m5[:, :1] == gi_e).any(1).sum())
                acc[sub]["m_a5"] += int((m5         == gi_e).any(1).sum())
                acc[sub]["b_a1"] += int((b5[:, :1] == gi_e).any(1).sum())
                acc[sub]["b_a5"] += int((b5         == gi_e).any(1).sum())

    model.train()
    out = {}
    for sub, s in acc.items():
        nc = s["n_cov"]
        out[sub] = {
            "n_total":   s["n_total"],
            "n_covered": nc,
            "model_nll": s["m_ce"] / max(nc, 1),
            "base_nll":  s["b_ce"] / max(nc, 1),
            "delta_nll": (s["b_ce"] - s["m_ce"]) / max(nc, 1),
            "model_acc1": s["m_a1"] / max(nc, 1),
            "model_acc5": s["m_a5"] / max(nc, 1),
            "base_acc1":  s["b_a1"] / max(nc, 1),
            "base_acc5":  s["b_a5"] / max(nc, 1),
        }
    return out


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_ctf_loss(
    model: CandidateSetTransformerRefiner,
    batch: Dict,
    device,
    lambda_kl: float,
    lambda_delta: float,
    drop_gold_not_selected: bool = False,
) -> Tuple[Optional[torch.Tensor], Dict]:
    h         = batch["h_prime"].to(device)
    ct        = batch["cand_tok"].to(device)
    cf        = batch["cand_fine"].to(device)
    cs        = batch["cand_super"].to(device)
    cmask     = batch["cand_mask"].to(device)
    gold_idx  = batch["gold_idx"].to(device)
    covered   = batch["covered"].to(device)
    r_reg     = batch["r_topk_reg"].to(device)
    r_prb     = batch["r_topk_prb"].to(device)
    m_reg     = batch["m_topk_reg"].to(device)
    m_prb     = batch["m_topk_prb"].to(device)
    r_margin  = batch["r_margin"].to(device)
    m_margin  = batch["m_margin"].to(device)

    scores, base_sc = model(
        h, ct, cf, cs, cmask, model._tok_emb_w,
        r_reg, r_prb, m_reg, m_prb, r_margin, m_margin,
    )

    cov_mask = covered.bool()
    if drop_gold_not_selected:
        # Only train on positions where gold is naturally in selected top-M
        sel_idx = model._last_sel_stats.get("sel_idx")
        if sel_idx is not None:
            gold_in_topM = (sel_idx == gold_idx.unsqueeze(1)).any(dim=1)
            cov_mask = cov_mask & gold_in_topM

    n_cov    = int(cov_mask.sum())
    if n_cov == 0:
        return None, {}

    sc  = scores[cov_mask]
    bc  = base_sc[cov_mask]
    gi  = gold_idx[cov_mask]

    loss_ce = F.cross_entropy(sc, gi)

    p_pred  = F.softmax(sc, dim=-1)
    p_base  = F.softmax(bc.detach(), dim=-1)
    loss_kl = (p_pred * (torch.log(p_pred + 1e-9) - torch.log(p_base + 1e-9))).sum(-1).mean()

    delta      = (scores - base_sc)[cmask]
    loss_delta = delta.pow(2).mean()

    total = loss_ce + lambda_kl * loss_kl + lambda_delta * loss_delta
    return total, {"ce": loss_ce.item(), "kl": loss_kl.item(), "delta": loss_delta.item()}


# ── Training ──────────────────────────────────────────────────────────────────

def train_ctf(
    args,
    d_model: int,
    n_fine: int,
    n_super: int,
    r2s_np: np.ndarray,
    tok_emb_w: torch.Tensor,
    device: torch.device,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    model = CandidateSetTransformerRefiner(
        d_model=d_model, n_fine=n_fine, n_super=n_super,
        refiner_dim=args.refiner_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, ff_mult=args.ff_mult,
        dropout=args.dropout, selected_M=args.selected_M,
    ).to(device)
    model.register_buffer("_tok_emb_w", tok_emb_w.float().to(device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] CTF  M={args.selected_M}  D={args.refiner_dim}  "
          f"L={args.num_layers}  H={args.num_heads}  params={n_params:,}")
    print(f"[train] train_filter={args.train_filter}  gate_filter={args.gate_filter}")

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    fail_hard     = getattr(args, "fail_on_baseline_mismatch", False)
    official_bl: Optional[Dict] = None
    if args.official_baseline and os.path.isfile(args.official_baseline):
        with open(args.official_baseline) as f:
            official_bl = json.load(f)
        print()
        print("=== CANONICAL EVAL ENABLED ===")
        print(f"  baseline nll        : {official_bl['covered_nll']:.6f}")
        print(f"  baseline fingerprint: {official_bl['dataset_fingerprint']}")
        print(f"  num_examples        : {official_bl['num_examples']:,}")
        if fail_hard:
            print("  --fail_on_baseline_mismatch: training aborts on mismatch.")
        print()

    filter_kwargs = {"margin_thresh": args.margin_thresh,
                     "entropy_thresh": args.entropy_thresh}
    variant_tag   = f"CTF-M{args.selected_M}/{args.train_filter}"
    bl_nll        = official_bl["covered_nll"] if official_bl else float("nan")

    # ── Safety guard: eval_selection_mode must be topm_no_gold ───────────────
    eval_sel_mode  = getattr(args, "eval_selection_mode",  "topm_no_gold")
    train_sel_mode = getattr(args, "train_selection_mode", "topm_no_gold")
    drop_train_no_sel = getattr(args, "drop_train_gold_not_selected", False)
    if eval_sel_mode != "topm_no_gold":
        raise RuntimeError(
            f"train_ctf: eval_selection_mode={eval_sel_mode!r} — only 'topm_no_gold' "
            "is allowed for canonical eval. Oracle modes must never be used for training."
        )
    if train_sel_mode != "topm_no_gold":
        raise RuntimeError(
            f"train_ctf: train_selection_mode={train_sel_mode!r} — only 'topm_no_gold' allowed."
        )
    print(f"[train] selection_mode        = {eval_sel_mode} (eval and train)")
    print(f"[train] eval_force_include_gold= false")
    print(f"[train] drop_train_gold_not_selected = {drop_train_no_sel}")

    run_filter_audit(args.train_dir, args.val_dir, filter_kwargs, args.output_dir)

    # ── Step-0 checks ─────────────────────────────────────────────────────────
    if getattr(args, "eval_before_train", False):
        tok_dev = model._tok_emb_w
        print("\n[train] === step-0 eval (eval_before_train) ===")

        # Force-zero full val (no model call)
        print("  [step 0 / force_zero] full val ...")
        m0_zero = canonical_eval_refiner(
            None, args.val_dir, tok_dev, r2s_np, device,
            force_zero=True, eval_batch_size=args.eval_batch_size,
            official_baseline=official_bl, fail_on_mismatch=fail_hard,
            variant_tag=variant_tag,
        )
        fp0 = m0_zero["dataset_fingerprint"]
        print(f"  [step 0 / force_zero]  nll={m0_zero['covered_nll']:.6f}  "
              f"cov={m0_zero['coverage']:.6f}  fp={fp0}")
        if official_bl:
            check_baseline_match(m0_zero, official_bl, fail_hard,
                                 context=f"force_zero/{variant_tag}", check_nll=True)

        # Init identity check (asserts max|delta|<1e-5, NLL diff<1e-4)
        print()
        print("  [init identity] zero_out_layer + scale=1.0 → delta=0 at step 0 ...")
        check_init_identity(model, args.val_dir, tok_dev, r2s_np, device,
                            print_delta_stats=True)

        # With-delta full val using CTF eval (topm_no_gold — inference valid)
        print()
        print("  [step 0 / with_delta] full val (CTF eval, selection_mode=topm_no_gold) ...")
        m0_with = canonical_eval_ctf(
            model, args.val_dir, tok_dev, r2s_np, device,
            eval_batch_size=args.eval_batch_size,
            official_baseline=official_bl, fail_on_mismatch=fail_hard,
            variant_tag=variant_tag,
        )
        print(f"  [step 0 / with_delta]  nll={m0_with['covered_nll']:.6f}  "
              f"cov={m0_with['coverage']:.6f}  fp={m0_with['dataset_fingerprint']}")
        print(f"    gold_force_included_rate = {m0_with['gold_force_included_rate']:.4f}")

        nll_diff0 = abs(m0_with["covered_nll"] - m0_zero["covered_nll"])
        fp_ok0    = m0_with["dataset_fingerprint"] == fp0
        cov_ok0   = abs(m0_with["coverage"] - m0_zero["coverage"]) < 1e-6
        if nll_diff0 >= 1e-4 or not fp_ok0 or not cov_ok0:
            raise RuntimeError(
                f"Step-0 identity FAIL [{variant_tag}]: "
                f"with_delta={m0_with['covered_nll']:.6f}  "
                f"force_zero={m0_zero['covered_nll']:.6f}  "
                f"nll_diff={nll_diff0:.2e}  fp_ok={fp_ok0}  cov_ok={cov_ok0}."
            )
        print(f"  [step 0] identity PASS  nll_diff={nll_diff0:.2e}")
        model.train()

    # ── Dataset + optimiser ───────────────────────────────────────────────────
    train_ds  = FilteredShardStreamDataset(
        args.train_dir, r2s_np, args.train_filter,
        filter_kwargs=filter_kwargs, shuffle=True,
    )
    train_inf = make_infinite(train_ds, args.batch_size)

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler("cuda")
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                         eta_min=args.lr * 0.1)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path   = os.path.join(args.output_dir, "train_log.csv")
    log_fields = ["step", "ce", "kl", "delta",
                  "gated_covered_nll", "inside_gate_model_nll", "inside_gate_base_nll",
                  "outside_gate_nll", "gate_rate", "covered_gate_rate",
                  "coverage", "dataset_fingerprint", "num_examples", "num_covered",
                  "gold_force_included_rate", "delta_abs_mean", "delta_abs_max",
                  "residual_scale"]
    log_file   = open(log_path, "w", newline="")
    log_csv    = csv.DictWriter(log_file, fieldnames=log_fields, extrasaction="ignore")
    log_csv.writeheader()

    sub_path   = os.path.join(args.output_dir, "local_subset_eval.csv")
    sub_fields = ["step", "subset", "n_total", "n_covered",
                  "model_nll", "base_nll", "delta_nll",
                  "model_acc1", "model_acc5", "base_acc1", "base_acc5"]
    sub_file   = open(sub_path, "w", newline="")
    sub_csv    = csv.DictWriter(sub_file, fieldnames=sub_fields, extrasaction="ignore")
    sub_csv.writeheader()

    best_path = os.path.join(args.output_dir, "best_refiner.pt")
    best_nll  = float("inf")
    best_step = 0
    ema_ce    = None
    t0        = time.time()
    model.train()

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in range(1, args.steps + 1):
        batch = next(train_inf)

        with autocast("cuda"):
            loss, info = compute_ctf_loss(model, batch, device,
                                          args.lambda_kl, args.lambda_delta,
                                          drop_gold_not_selected=drop_train_no_sel)
        if loss is None:
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()

        ema_ce = info["ce"] if ema_ce is None else 0.98 * ema_ce + 0.02 * info["ce"]

        if step % 100 == 0:
            rscale = float(model.residual_scale)
            print(f"  step {step:6d}/{args.steps}  ce={ema_ce:.4f}  "
                  f"kl={info['kl']:.4f}  scale={rscale:.4f}  "
                  f"t={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] step={step} ···")

            g_m = gated_global_eval_ctf(
                model, args.val_dir, model._tok_emb_w, r2s_np, device,
                gate_filter_name=args.gate_filter,
                filter_kwargs=filter_kwargs,
                official_baseline=official_bl,
                fail_on_mismatch=fail_hard,
                variant_tag=variant_tag,
                eval_batch_size=args.eval_batch_size,
            )
            l_r = local_subset_eval_ctf(
                model, args.val_dir, model._tok_emb_w, r2s_np, device,
                filter_kwargs=filter_kwargs,
                eval_batch_size=args.eval_batch_size,
            )

            rscale = float(model.residual_scale)
            row    = {"step": step, **info, **g_m, "residual_scale": rscale}
            log_csv.writerow(row)
            log_file.flush()
            for sub, m in l_r.items():
                sub_csv.writerow({"step": step, "subset": sub, **m})
            sub_file.flush()

            gated_nll   = g_m["gated_covered_nll"]
            delta_vs_bl = bl_nll - gated_nll

            print(f"  [eval/gated_global] step={step}")
            print(f"    gated_covered_nll     = {gated_nll:.6f}  "
                  f"(baseline={bl_nll:.6f}  delta={delta_vs_bl:+.6f})")
            print(f"    inside_gate_model_nll = {g_m['inside_gate_model_nll']:.6f}")
            print(f"    inside_gate_base_nll  = {g_m['inside_gate_base_nll']:.6f}")
            print(f"    gate_rate             = {g_m['gate_rate']:.4f}")
            print(f"    coverage              = {g_m['coverage']:.6f}")
            print(f"    fingerprint           = {g_m['dataset_fingerprint']}")
            print(f"    num_examples          = {g_m['num_examples']:,}")
            print(f"    num_covered           = {g_m['num_covered']:,}")
            print(f"    gold_force_incl_rate  = {g_m['gold_force_included_rate']:.4f}")
            print(f"    residual_scale        = {rscale:.4f}")

            print(f"  [eval/local_subsets]")
            print(f"    {'subset':30s}  {'n_cov':>7}  {'model_nll':>9}  "
                  f"{'delta':>7}  {'acc@1':>5}  {'base_acc@1':>10}")
            for sub in ["hard_union", "hard_union_small", "boundary",
                        "type_A", "router_top8_miss", "all"]:
                if sub not in l_r:
                    continue
                m = l_r[sub]
                if m["n_covered"] == 0:
                    continue
                print(f"    {sub:30s}  {m['n_covered']:7,}  "
                      f"{m['model_nll']:9.4f}  {m['delta_nll']:+7.4f}  "
                      f"{m['model_acc1']:5.3f}  {m['base_acc1']:10.3f}")

            if gated_nll < best_nll:
                best_nll  = gated_nll
                best_step = step
                best_metrics = {
                    "eval_mode":               "gated_global_val",
                    "selection_mode":          "topm_no_gold",
                    "eval_force_include_gold": False,
                    "step":                    step,
                    "variant":                 "CandidateTransformer",
                    "train_filter":            args.train_filter,
                    "gate_filter":             args.gate_filter,
                    "selected_M":              args.selected_M,
                    "refiner_dim":             args.refiner_dim,
                    "num_layers":              args.num_layers,
                    "drop_train_gold_not_selected": drop_train_no_sel,
                    "official_baseline_nll":   bl_nll,
                    "gated_covered_nll":       gated_nll,
                    "delta_vs_baseline":       delta_vs_bl,
                    "inside_gate_model_nll":   g_m["inside_gate_model_nll"],
                    "inside_gate_base_nll":    g_m["inside_gate_base_nll"],
                    "outside_gate_nll":        g_m["outside_gate_nll"],
                    "gate_rate":               g_m["gate_rate"],
                    "covered_gate_rate":       g_m["covered_gate_rate"],
                    "coverage":                g_m["coverage"],
                    "fingerprint":             g_m["dataset_fingerprint"],
                    "num_examples":            g_m["num_examples"],
                    "num_covered":             g_m["num_covered"],
                    "gold_force_included_rate": g_m["gold_force_included_rate"],
                    "residual_scale":          rscale,
                    "local_subsets":           l_r,
                }
                torch.save({"step": step, "model": model.state_dict(),
                            "metrics": best_metrics, "args": vars(args)}, best_path)
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"  [eval] *** new best  gated_nll={best_nll:.6f}  "
                      f"delta={delta_vs_bl:+.6f}  → {best_path}")

    torch.save({"step": args.steps, "model": model.state_dict(),
                "metrics": {}, "args": vars(args)},
               os.path.join(args.output_dir, "last_refiner.pt"))
    log_file.close()
    sub_file.close()
    print(f"\n[train] done  filter={args.train_filter}  "
          f"best_gated_nll={best_nll:.6f}  best_step={best_step}")


# ── Debug mode ────────────────────────────────────────────────────────────────

def run_debug(args, backbone, d_model, n_fine, n_super, r2s_np,
              tok_emb_w, device) -> None:
    """
    Smoke-test the CTF pipeline end-to-end:
      1. Build model, verify init identity (delta=0 at step 0)
      2. Force-zero full-val  → assert == baseline
      3. With-delta full-val  → assert == force-zero (identity)
      4. Report gold force-include stats on one shard
    """
    print("\n[debug] Building CTF model ...")
    model = CandidateSetTransformerRefiner(
        d_model=d_model, n_fine=n_fine, n_super=n_super,
        refiner_dim=args.refiner_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, ff_mult=args.ff_mult,
        dropout=0.0, selected_M=args.selected_M,
    ).to(device)
    model.register_buffer("_tok_emb_w", tok_emb_w.float().to(device))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}  M={args.selected_M}  D={args.refiner_dim}")

    tok_dev      = model._tok_emb_w
    fail_hard    = getattr(args, "fail_on_baseline_mismatch", False)
    official_bl  = None
    if args.official_baseline and os.path.isfile(args.official_baseline):
        with open(args.official_baseline) as f:
            official_bl = json.load(f)

    # 1. Small-batch identity check
    print("\n[debug] Init identity check ...")
    check_init_identity(model, args.val_dir, tok_dev, r2s_np, device,
                        print_delta_stats=True)

    # 2. Gold coverage in top-M (diagnostic — uses oracle mode for rate reporting only)
    print("\n[debug] Gold naturally-in-top-M rate (first val shard) ...")
    paths = sorted(glob.glob(os.path.join(args.val_dir, "shard_*.pt")))
    shard = torch.load(paths[0], map_location="cpu", weights_only=True)
    N     = min(512, len(shard["covered"]))
    h     = shard["h_prime"][:N].float().to(device)
    ct    = shard["cand_tok"][:N].long().to(device)
    cmask = (ct >= 0)
    g_idx   = shard["gold_cand_idx"][:N].long().to(device)
    covered = shard["covered"][:N].bool().to(device)
    emb_w   = tok_dev.float()
    tok_e   = F.embedding(ct.clamp(min=0), emb_w)
    base_r  = (h.unsqueeze(1) * tok_e).sum(-1).masked_fill(~cmask, float("-inf"))
    # Use oracle mode only to measure how many positions would need force-inclusion
    sel_idx, sel_mask, gold_would_need_force = select_candidates_topm(
        base_r, cmask, args.selected_M,
        mode="oracle_force_gold_last", gold_idx=g_idx, covered=covered,
    )
    n_cov = int(covered.sum())
    n_natural_in = n_cov - int(gold_would_need_force.sum())
    print(f"  positions                    : {N}")
    print(f"  covered                      : {n_cov}  ({n_cov/N*100:.1f}%)")
    print(f"  gold naturally in top-M      : {n_natural_in}  ({n_natural_in/max(n_cov,1)*100:.1f}% of covered)")
    print(f"  gold NOT in top-M (excluded) : {int(gold_would_need_force.sum())}  "
          f"({int(gold_would_need_force.sum())/max(n_cov,1)*100:.1f}% of covered)")
    print(f"  mean valid sel               : {sel_mask.float().sum(1).mean():.1f} / {args.selected_M}")
    print(f"  NOTE: training uses topm_no_gold — gold outside top-M is EXCLUDED, not inserted.")

    # 3. Force-zero full-val
    print("\n[debug] Force-zero full-val ...")
    m0_zero = canonical_eval_refiner(
        None, args.val_dir, tok_dev, r2s_np, device,
        force_zero=True, eval_batch_size=args.eval_batch_size,
        official_baseline=official_bl, fail_on_mismatch=fail_hard,
        variant_tag="CTF-debug",
    )
    print(f"  force_zero_nll = {m0_zero['covered_nll']:.6f}")
    print(f"  coverage       = {m0_zero['coverage']:.6f}")
    print(f"  fingerprint    = {m0_zero['dataset_fingerprint']}")
    if official_bl:
        check_baseline_match(m0_zero, official_bl, fail_hard,
                             context="CTF-debug/force_zero", check_nll=True)

    # 4. With-delta full-val (CTF eval, selection_mode=topm_no_gold)
    print("\n[debug] With-delta full-val (CTF eval, selection_mode=topm_no_gold) ...")
    m0_with = canonical_eval_ctf(
        model, args.val_dir, tok_dev, r2s_np, device,
        eval_batch_size=args.eval_batch_size,
        official_baseline=official_bl, fail_on_mismatch=fail_hard,
        variant_tag="CTF-debug",
    )
    print(f"  with_delta_nll           = {m0_with['covered_nll']:.6f}")
    print(f"  coverage                 = {m0_with['coverage']:.6f}")
    print(f"  fingerprint              = {m0_with['dataset_fingerprint']}")
    print(f"  max_abs_delta            = {m0_with['delta_abs_max']:.2e}")
    print(f"  gold_force_included_rate = {m0_with['gold_force_included_rate']:.4f}  "
          f"(must be 0.0000 — selection_mode=topm_no_gold)")

    nll_diff = abs(m0_with["covered_nll"] - m0_zero["covered_nll"])
    fp_match = m0_with["dataset_fingerprint"] == m0_zero["dataset_fingerprint"]
    print(f"\n[debug] identity check: nll_diff={nll_diff:.2e}  fp_match={fp_match}")
    if nll_diff < 1e-4 and fp_match:
        print("[debug] PASS — CTF is exact identity at step 0")
    else:
        print("[debug] FAIL — identity not satisfied at step 0")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device(args.device)

    print(f"[main] loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    if hasattr(backbone, "token_emb"):
        tok_emb_w = backbone.token_emb.weight.detach().cpu()
    elif hasattr(backbone, "transformer"):
        tok_emb_w = backbone.transformer.wte.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot locate token embedding in backbone")

    cfg_path = os.path.join(args.val_dir, "dataset_config.json")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(args.train_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            ds_cfg = json.load(f)
        n_fine  = ds_cfg.get("n_fine",  128)
        n_super = ds_cfg.get("n_super",  24)
        print(f"[main] n_fine={n_fine}  n_super={n_super} (from dataset_config)")
    else:
        n_fine  = args.n_fine
        n_super = args.n_super

    r2s_np = np.zeros(n_fine, dtype=np.int32)
    if args.super_map and os.path.isfile(args.super_map):
        r2s_np  = load_r2s(args.super_map, n_fine)
        n_super = int(r2s_np.max()) + 1
        print(f"[main] loaded super_map  n_super={n_super}")

    if getattr(args, "debug_only", False):
        run_debug(args, backbone, d_model, n_fine, n_super, r2s_np, tok_emb_w, device)
        return

    if not args.train_dir:
        raise RuntimeError("--train_dir required for training")
    if not args.output_dir:
        raise RuntimeError("--output_dir required for training")

    train_ctf(args, d_model, n_fine, n_super, r2s_np, tok_emb_w, device)


def _parse():
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--train_dir",   default="")
    p.add_argument("--val_dir",     required=True)
    p.add_argument("--small_ckpt",  required=True)
    p.add_argument("--super_map",   default=None)
    p.add_argument("--output_dir",  default="")
    p.add_argument("--official_baseline", default=None)
    # Architecture
    p.add_argument("--selected_M",  type=int, default=256)
    p.add_argument("--refiner_dim", type=int, default=256)
    p.add_argument("--num_layers",  type=int, default=2)
    p.add_argument("--num_heads",   type=int, default=4)
    p.add_argument("--ff_mult",     type=int, default=4)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--n_fine",      type=int, default=128)
    p.add_argument("--n_super",     type=int, default=24)
    # Filter
    p.add_argument("--train_filter",   default="boundary")
    p.add_argument("--gate_filter",    default=None)
    p.add_argument("--margin_thresh",  type=float, default=0.1)
    p.add_argument("--entropy_thresh", type=float, default=2.0)
    # Training
    p.add_argument("--steps",          type=int,   default=10_000)
    p.add_argument("--eval_every",     type=int,   default=1_000)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--eval_batch_size",type=int,   default=32)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--lambda_kl",      type=float, default=0.01)
    p.add_argument("--lambda_delta",   type=float, default=1e-4)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    # Selection mode (must remain topm_no_gold for all canonical evals)
    p.add_argument("--train_selection_mode", default="topm_no_gold",
                   choices=["topm_no_gold"],
                   help="Candidate selection mode for training. Only topm_no_gold is valid.")
    p.add_argument("--eval_selection_mode",  default="topm_no_gold",
                   choices=["topm_no_gold"],
                   help="Candidate selection mode for eval. Only topm_no_gold is valid.")
    p.add_argument("--drop_train_gold_not_selected", action="store_true",
                   help="Only train on covered positions where gold is naturally in top-M "
                        "(no oracle insertion). Positions where gold is outside top-M are skipped.")
    # Flags
    p.add_argument("--fail_on_baseline_mismatch", action="store_true")
    p.add_argument("--eval_before_train",         action="store_true")
    p.add_argument("--debug_only",                action="store_true",
                   help="Run identity + baseline smoke tests, then exit.")
    p.add_argument("--device", default="cuda")

    args = p.parse_args()
    if args.gate_filter is None:
        args.gate_filter = args.train_filter
    return args


if __name__ == "__main__":
    run(_parse())
