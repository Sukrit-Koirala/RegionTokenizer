#!/usr/bin/env python3
"""
Set-transformer refiner (STR, variant=SetTransformer).

Treats the candidate set as an UNORDERED SET rather than a sequence.

Key difference from the candidate transformer (CTF):

  CTF:  [CTX][ROUTER][MEM][CAND_0 .. CAND_{M-1}]  — unified sequence,
         CAND gets type_emb + rank_emb  (position-dependent, not equivariant)

  STR:  context tokens [CTX, ROUTER, MEM] and candidate set are SEPARATE streams.
         Each STR layer:
           1. MAB(cand ← ctx)  — candidates cross-attend to 3 context tokens
           2. SAB(cand ← cand) — candidates self-attend within the set
         NO rank_emb, NO candidate type_emb → output is PERMUTATION-EQUIVARIANT.

Permutation equivariance guarantee: if you permute the order of candidates fed
into the model, each candidate receives the same delta regardless of its slot
position — only its features (base_logit, token, region, router/mem probs) matter.

Building blocks:
  MAB(Q, KV)  Pre-LN multihead attention block: Q attends to KV, then FFN
  SAB(X)      Set attention block: MAB(X, X) — self-attention in the set
  SetBlock    One layer: MAB(cand, ctx) then SAB(cand)

Candidate score features (5-dim, no rank_norm — preserves permutation invariance):
  [base_logit, r_prob, m_prob, is_router_region, is_mem_region]

Robust init: out.weight=zeros, out.bias=zeros, residual_scale=1.0
  → delta=0 at step 0, inner-layer gradients flow from step 1.

Canonical eval requirements (identical to CTF):
  fingerprint  = f57cabcdc46d69ce
  num_examples = 239,362
  num_covered  = 227,017
  coverage     = 0.948425

Usage:
    python scripts/train_set_transformer_refiner.py \\
        --small_ckpt runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --train_dir  runs/path_refiner_clean/data/train_hgrid_K24 \\
        --val_dir    runs/path_refiner_clean/data/val_hgrid_K24 \\
        --super_map  runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --official_baseline runs/path_refiner_clean/baselines/saved_candidate_baseline.json \\
        --output_dir runs/path_refiner_set_transformer/variant_STR_boundary_M256 \\
        --train_filter boundary --gate_filter boundary \\
        --selected_M 256 --refiner_dim 256 --num_layers 2 --num_heads 4 \\
        --steps 10000 --eval_every 1000 --device cuda
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.train_clean_path_refiner import (
    canonical_eval_refiner,
    check_baseline_match,
    check_init_identity,
    load_r2s,
    make_infinite,
)
from scripts.train_hard_position_refiner import (
    EVAL_SUBSETS,
    FilteredShardStreamDataset,
    compute_filter_mask,
    run_filter_audit,
)
from scripts.train_candidate_transformer_refiner import (
    _build_fine_indicator,
    _build_fine_probs,
    canonical_eval_ctf,
    compute_ctf_loss,
    gated_global_eval_ctf,
    local_subset_eval_ctf,
    select_candidates_topm,
)


# ── Set Transformer building blocks ───────────────────────────────────────────

class MAB(nn.Module):
    """
    Pre-LN Multi-head Attention Block.

    Q attends to KV (cross-attention when Q≠KV, self-attention when Q=KV).
    Separate LayerNorm for Q and KV allows Q/KV to come from different streams.
    """

    def __init__(self, dim: int, num_heads: int,
                 ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2   = nn.LayerNorm(dim)
        self.ff      = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        Q:       torch.Tensor,                    # (B, Lq,  D)
        KV:      torch.Tensor,                    # (B, Lkv, D)
        kv_mask: Optional[torch.Tensor] = None,   # (B, Lkv) bool, True=ignore
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.norm_q(Q), self.norm_kv(KV), self.norm_kv(KV),
            key_padding_mask=kv_mask, need_weights=False,
        )
        Q = Q + self.drop(attn_out)
        Q = Q + self.drop(self.ff(self.norm2(Q)))
        return Q


class SetBlock(nn.Module):
    """
    One layer of the set transformer encoder.

    Step 1: MAB(cand ← ctx)  — candidates attend to the 3 fixed context tokens.
    Step 2: SAB(cand ← cand) — candidates attend to each other within the set.

    Context tokens (CTX/ROUTER/MEM) are never padded, so cross-attention needs
    no key-padding mask. The within-set SAB masks out padding candidate slots.
    """

    def __init__(self, dim: int, num_heads: int,
                 ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.cross = MAB(dim, num_heads, ff_mult, dropout)   # cand ← ctx
        self.self_ = MAB(dim, num_heads, ff_mult, dropout)   # cand ← cand

    def forward(
        self,
        cand:     torch.Tensor,                   # (B, M, D)
        ctx:      torch.Tensor,                   # (B, 3, D)
        cand_kpm: Optional[torch.Tensor] = None,  # (B, M) True=padding
    ) -> torch.Tensor:
        cand = self.cross(cand, ctx)                           # ctx never masked
        cand = self.self_(cand, cand, kv_mask=cand_kpm)
        return cand


# ── Model ──────────────────────────────────────────────────────────────────────

class SetTransformerRefiner(nn.Module):
    """
    Permutation-equivariant set transformer over the selected top-M candidates.

    Differences from CandidateSetTransformerRefiner (CTF):
      - No rank_emb:     candidates are an unordered set; position in the set
                         does not affect the output delta.
      - No type_emb for candidates: type embedding is only for CTX/ROUTER/MEM.
      - score_proj takes 5 features instead of 6 (rank_norm removed).
      - Context stream and candidate stream are kept separate; cross-attention
        (MAB) feeds context information into the candidate representations,
        after which candidates self-attend (SAB) within the set.

    Init: out.weight=0, out.bias=0, residual_scale=1.0 → delta=0 at step 0.
    Gradients flow through inner MAB/SAB weights from step 1.
    """

    def __init__(
        self,
        d_model:     int,
        n_fine:      int,
        n_super:     int,
        refiner_dim: int   = 256,
        num_layers:  int   = 2,
        num_heads:   int   = 4,
        ff_mult:     int   = 4,
        dropout:     float = 0.0,
        selected_M:  int   = 256,
    ) -> None:
        super().__init__()
        self.d_model     = d_model
        self.n_fine      = n_fine
        self.n_super     = n_super
        self.refiner_dim = refiner_dim
        self.selected_M  = selected_M

        # ── Context encoders ────────────────────────────────────────────────
        self.ctx_proj    = nn.Linear(d_model, refiner_dim)
        self.router_proj = nn.Sequential(nn.Linear(5, refiner_dim), nn.GELU())
        self.mem_proj    = nn.Sequential(nn.Linear(5, refiner_dim), nn.GELU())
        # Type embeddings distinguish the three context tokens (not used for cands)
        self.ctx_type_emb = nn.Embedding(3, refiner_dim)   # 0=CTX 1=ROUTER 2=MEM

        # ── Candidate encoders (NO rank_emb, NO cand type_emb) ──────────────
        self.token_proj  = nn.Linear(d_model, refiner_dim, bias=False)
        self.fine_emb    = nn.Embedding(n_fine  + 1, refiner_dim, padding_idx=n_fine)
        self.super_emb   = nn.Embedding(n_super + 1, refiner_dim, padding_idx=n_super)
        # 5 scalar features: [base_logit, r_prob, m_prob, is_r, is_m]
        self.score_proj  = nn.Linear(5, refiner_dim)

        # ── Set transformer layers ──────────────────────────────────────────
        self.set_layers  = nn.ModuleList([
            SetBlock(refiner_dim, num_heads, ff_mult, dropout)
            for _ in range(num_layers)
        ])

        # ── Output ──────────────────────────────────────────────────────────
        self.norm_out       = nn.LayerNorm(refiner_dim)
        self.out            = nn.Linear(refiner_dim, 1)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

        self._last_sel_stats: Dict = {}
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.ctx_proj.weight,        std=0.02)
        nn.init.zeros_(self.ctx_proj.bias)
        nn.init.normal_(self.router_proj[0].weight,  std=0.02)
        nn.init.zeros_(self.router_proj[0].bias)
        nn.init.normal_(self.mem_proj[0].weight,     std=0.02)
        nn.init.zeros_(self.mem_proj[0].bias)
        nn.init.normal_(self.ctx_type_emb.weight,    std=0.02)
        nn.init.normal_(self.token_proj.weight,      std=0.02)
        nn.init.normal_(self.fine_emb.weight,        std=0.02)
        nn.init.normal_(self.super_emb.weight,       std=0.02)
        nn.init.normal_(self.score_proj.weight,      std=0.01)
        nn.init.zeros_(self.score_proj.bias)
        # Zero-init output head → delta=0 at step 0
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ── Candidate selection ──────────────────────────────────────────────────

    def select_candidates(
        self,
        base:      torch.Tensor,  # (B, C) -inf at invalid
        cand_mask: torch.Tensor,  # (B, C) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-M by base logit only — inference valid. Gold is never consulted.
        Returns sel_idx (B, M) and sel_mask (B, M).
        """
        sel_idx, sel_mask, _ = select_candidates_topm(
            base, cand_mask, self.selected_M, mode="topm_no_gold"
        )
        return sel_idx, sel_mask

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        h_prime:     torch.Tensor,           # (B, d_model)
        cand_tok:    torch.Tensor,           # (B, C)
        cand_fine:   torch.Tensor,           # (B, C)
        cand_super:  torch.Tensor,           # (B, C)
        cand_mask:   torch.Tensor,           # (B, C) bool
        token_emb_w: torch.Tensor,           # (vocab, d_model)
        r_topk_reg:  torch.Tensor,           # (B, K)
        r_topk_prb:  torch.Tensor,           # (B, K)
        m_topk_reg:  torch.Tensor,           # (B, K)
        m_topk_prb:  torch.Tensor,           # (B, K)
        r_margin:    torch.Tensor,           # (B,)
        m_margin:    torch.Tensor,           # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C   = cand_tok.shape
        M      = self.selected_M
        D      = self.refiner_dim
        device = h_prime.device
        emb_w  = token_emb_w.float()

        # ── Base scores ──────────────────────────────────────────────────────
        tok_e    = F.embedding(cand_tok.clamp(min=0), emb_w)         # (B, C, d_model)
        base_raw = (h_prime.float().unsqueeze(1) * tok_e).sum(-1)    # (B, C)
        base     = base_raw.masked_fill(~cand_mask, float("-inf"))

        # ── Candidate selection (topm_no_gold — inference valid) ─────────────
        sel_idx, sel_mask = self.select_candidates(base, cand_mask)
        self._last_sel_stats = {
            "gold_forced":     torch.zeros(B, dtype=torch.bool, device=device),
            "sel_valid_count": sel_mask.float().sum(1),
            "sel_idx":         sel_idx.detach(),
        }

        # ── Context tokens (B, 3, D) ─────────────────────────────────────────
        ctx_tok = self.ctx_proj(h_prime.float())                      # (B, D)
        ctx_tok = ctx_tok + self.ctx_type_emb.weight[0]

        r_prb_4 = r_topk_prb[:, :4].float()
        if r_prb_4.size(1) < 4:
            r_prb_4 = F.pad(r_prb_4, (0, 4 - r_prb_4.size(1)))
        r_feat  = torch.cat([r_margin.float().unsqueeze(-1), r_prb_4], dim=-1)
        r_tok   = self.router_proj(r_feat) + self.ctx_type_emb.weight[1]

        m_prb_4 = m_topk_prb[:, :4].float()
        if m_prb_4.size(1) < 4:
            m_prb_4 = F.pad(m_prb_4, (0, 4 - m_prb_4.size(1)))
        m_feat  = torch.cat([m_margin.float().unsqueeze(-1), m_prb_4], dim=-1)
        m_tok   = self.mem_proj(m_feat) + self.ctx_type_emb.weight[2]

        ctx_seq = torch.stack([ctx_tok, r_tok, m_tok], dim=1)         # (B, 3, D)

        # ── Candidate tokens (B, M, D) — no rank, no type_emb ────────────────
        sel_clamped = sel_idx.clamp(min=0)

        sel_tok_id  = cand_tok.clamp(min=0).gather(1, sel_clamped)    # (B, M)
        sel_fine    = cand_fine.gather(1,  sel_clamped)               # (B, M)
        sel_sup     = cand_super.gather(1, sel_clamped)               # (B, M)
        sel_base    = base_raw.gather(1,   sel_clamped)               # (B, M)

        sel_tok_e   = F.embedding(sel_tok_id, emb_w)                  # (B, M, d_model)
        tok_part    = self.token_proj(sel_tok_e.float())               # (B, M, D)
        fine_part   = self.fine_emb(sel_fine.clamp(min=0, max=self.n_fine))   # (B,M,D)
        super_part  = self.super_emb(sel_sup.clamp(min=0, max=self.n_super))  # (B,M,D)

        # Per-candidate router / memory features
        p_r  = _build_fine_probs(r_topk_reg, r_topk_prb, self.n_fine)
        p_m  = _build_fine_probs(m_topk_reg, m_topk_prb, self.n_fine)
        in_r = _build_fine_indicator(r_topk_reg, self.n_fine)
        in_m = _build_fine_indicator(m_topk_reg, self.n_fine)

        sf_c       = sel_fine.clamp(min=0, max=self.n_fine - 1)
        r_prob_sel = p_r.gather(1, sf_c)                              # (B, M)
        m_prob_sel = p_m.gather(1, sf_c)
        is_r_sel   = in_r.gather(1, sf_c).float()
        is_m_sel   = in_m.gather(1, sf_c).float()

        # 5-dim features — rank_norm omitted to preserve permutation invariance
        score_feat = torch.stack(
            [sel_base, r_prob_sel, m_prob_sel, is_r_sel, is_m_sel], dim=-1
        )                                                              # (B, M, 5)
        score_part = self.score_proj(score_feat)                       # (B, M, D)

        cand_feat  = tok_part + fine_part + super_part + score_part   # (B, M, D)
        # Zero out padding slots so they don't pollute self-attention keys
        cand_feat  = cand_feat * sel_mask.float().unsqueeze(-1)

        # ── Set transformer layers ───────────────────────────────────────────
        cand_kpm = ~sel_mask                                           # (B, M) True=pad
        for layer in self.set_layers:
            cand_feat = layer(cand_feat, ctx_seq, cand_kpm=cand_kpm)

        # ── Output + scatter ─────────────────────────────────────────────────
        cand_out  = self.norm_out(cand_feat)                          # (B, M, D)
        delta_sel = self.out(cand_out).squeeze(-1)                    # (B, M)
        delta_sel = delta_sel * self.residual_scale
        delta_sel = delta_sel * sel_mask.float()                      # zero invalid slots

        full_delta = torch.zeros(B, C, device=device)
        full_delta.scatter_(1, sel_clamped, delta_sel)

        scores = (base_raw + full_delta).masked_fill(~cand_mask, float("-inf"))
        return scores, base


# ── Training ──────────────────────────────────────────────────────────────────

def train_str(
    args,
    d_model: int,
    n_fine:  int,
    n_super: int,
    r2s_np:     np.ndarray,
    tok_emb_w:  torch.Tensor,
    device:     torch.device,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    model = SetTransformerRefiner(
        d_model=d_model, n_fine=n_fine, n_super=n_super,
        refiner_dim=args.refiner_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, ff_mult=args.ff_mult,
        dropout=args.dropout, selected_M=args.selected_M,
    ).to(device)
    model.register_buffer("_tok_emb_w", tok_emb_w.float().to(device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] STR  M={args.selected_M}  D={args.refiner_dim}  "
          f"L={args.num_layers}  H={args.num_heads}  params={n_params:,}")
    print(f"[train] train_filter={args.train_filter}  gate_filter={args.gate_filter}")
    print(f"[train] architecture: set-transformer (permutation-equivariant, no rank_emb)")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    fail_hard    = getattr(args, "fail_on_baseline_mismatch", False)
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
    variant_tag   = f"STR-M{args.selected_M}/{args.train_filter}"
    bl_nll        = official_bl["covered_nll"] if official_bl else float("nan")

    # ── Safety guard: eval_selection_mode must be topm_no_gold ───────────────
    eval_sel_mode  = getattr(args, "eval_selection_mode",  "topm_no_gold")
    train_sel_mode = getattr(args, "train_selection_mode", "topm_no_gold")
    drop_train_no_sel = getattr(args, "drop_train_gold_not_selected", False)
    if eval_sel_mode != "topm_no_gold":
        raise RuntimeError(
            f"train_str: eval_selection_mode={eval_sel_mode!r} — only 'topm_no_gold' "
            "is allowed for canonical eval."
        )
    if train_sel_mode != "topm_no_gold":
        raise RuntimeError(
            f"train_str: train_selection_mode={train_sel_mode!r} — only 'topm_no_gold' allowed."
        )
    print(f"[train] selection_mode         = {eval_sel_mode} (eval and train)")
    print(f"[train] eval_force_include_gold= false")
    print(f"[train] drop_train_gold_not_selected = {drop_train_no_sel}")

    run_filter_audit(args.train_dir, args.val_dir, filter_kwargs, args.output_dir)

    # ── Step-0 checks ─────────────────────────────────────────────────────────
    if getattr(args, "eval_before_train", False):
        tok_dev = model._tok_emb_w
        print("\n[train] === step-0 eval (eval_before_train) ===")

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

        print()
        print("  [init identity] zero_out_layer + scale=1.0 → delta=0 at step 0 ...")
        check_init_identity(model, args.val_dir, tok_dev, r2s_np, device,
                            print_delta_stats=True)

        print()
        print("  [step 0 / with_delta] full val (STR eval, selection_mode=topm_no_gold) ...")
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
                  "gold_force_included_rate", "residual_scale"]
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
            tok_dev = model._tok_emb_w

            g_m = gated_global_eval_ctf(
                model, args.val_dir, tok_dev, r2s_np, device,
                gate_filter_name=args.gate_filter,
                filter_kwargs=filter_kwargs,
                official_baseline=official_bl,
                fail_on_mismatch=fail_hard,
                variant_tag=variant_tag,
                eval_batch_size=args.eval_batch_size,
            )
            l_r = local_subset_eval_ctf(
                model, args.val_dir, tok_dev, r2s_np, device,
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
                    "eval_mode":             "gated_global_val",
                    "step":                  step,
                    "variant":               "SetTransformer",
                    "train_filter":          args.train_filter,
                    "gate_filter":           args.gate_filter,
                    "selected_M":            args.selected_M,
                    "refiner_dim":           args.refiner_dim,
                    "num_layers":            args.num_layers,
                    "num_heads":             args.num_heads,
                    "architecture":          "set_transformer_no_rank_emb",
                    "official_baseline_nll": bl_nll,
                    "gated_covered_nll":     gated_nll,
                    "delta_vs_baseline":     delta_vs_bl,
                    "inside_gate_model_nll": g_m["inside_gate_model_nll"],
                    "inside_gate_base_nll":  g_m["inside_gate_base_nll"],
                    "outside_gate_nll":      g_m["outside_gate_nll"],
                    "gate_rate":             g_m["gate_rate"],
                    "covered_gate_rate":     g_m["covered_gate_rate"],
                    "coverage":              g_m["coverage"],
                    "fingerprint":           g_m["dataset_fingerprint"],
                    "num_examples":          g_m["num_examples"],
                    "num_covered":           g_m["num_covered"],
                    "gold_force_included_rate": g_m["gold_force_included_rate"],
                    "residual_scale":        rscale,
                    "local_subsets":         l_r,
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

def run_str_debug(args, d_model, n_fine, n_super, r2s_np, tok_emb_w, device) -> None:
    """
    Smoke-test the STR pipeline:
      1. Build model, print param count
      2. Init identity check (delta=0 at step 0)
      3. Gold coverage in top-M on first val shard
      4. Force-zero full-val  → assert == baseline
      5. With-delta full-val  → assert == force-zero (identity PASS)
    """
    print("\n[debug] Building SetTransformerRefiner ...")
    model = SetTransformerRefiner(
        d_model=d_model, n_fine=n_fine, n_super=n_super,
        refiner_dim=args.refiner_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, ff_mult=args.ff_mult,
        dropout=0.0, selected_M=args.selected_M,
    ).to(device)
    model.register_buffer("_tok_emb_w", tok_emb_w.float().to(device))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}  M={args.selected_M}  D={args.refiner_dim}  "
          f"L={args.num_layers}  H={args.num_heads}")
    print(f"  architecture: set-transformer (permutation-equivariant, no rank_emb)")

    tok_dev   = model._tok_emb_w
    fail_hard = getattr(args, "fail_on_baseline_mismatch", False)
    official_bl = None
    if args.official_baseline and os.path.isfile(args.official_baseline):
        with open(args.official_baseline) as f:
            official_bl = json.load(f)

    # 1. Init identity check (small batch)
    print("\n[debug] Init identity check ...")
    check_init_identity(model, args.val_dir, tok_dev, r2s_np, device,
                        print_delta_stats=True)

    # 2. Gold naturally-in-top-M rate (diagnostic — oracle mode for reporting only)
    print("\n[debug] Gold naturally-in-top-M rate (first val shard) ...")
    paths = sorted(glob.glob(os.path.join(args.val_dir, "shard_*.pt")))
    shard   = torch.load(paths[0], map_location="cpu", weights_only=True)
    N       = min(512, len(shard["covered"]))
    h       = shard["h_prime"][:N].float().to(device)
    ct      = shard["cand_tok"][:N].long().to(device)
    cmask   = (ct >= 0)
    emb_w   = tok_dev.float()
    tok_e   = F.embedding(ct.clamp(min=0), emb_w)
    base_r  = (h.unsqueeze(1) * tok_e).sum(-1).masked_fill(~cmask, float("-inf"))
    g_idx   = shard["gold_cand_idx"][:N].long().to(device)
    covered = shard["covered"][:N].bool().to(device)
    # Use oracle mode only to measure how many positions would need force-inclusion
    sel_idx, sel_mask, gold_would_need_force = select_candidates_topm(
        base_r, cmask, args.selected_M,
        mode="oracle_force_gold_last", gold_idx=g_idx, covered=covered,
    )
    n_cov   = int(covered.sum())
    n_natural_in = n_cov - int(gold_would_need_force.sum())
    print(f"  positions                    : {N}")
    print(f"  covered                      : {n_cov}  ({n_cov/N*100:.1f}%)")
    print(f"  gold naturally in top-M      : {n_natural_in}  ({n_natural_in/max(n_cov,1)*100:.1f}% of covered)")
    print(f"  gold NOT in top-M (excluded) : {int(gold_would_need_force.sum())}  "
          f"({int(gold_would_need_force.sum())/max(n_cov,1)*100:.1f}% of covered)")
    print(f"  mean valid sel               : {sel_mask.float().sum(1).mean():.1f} / {args.selected_M}")
    print(f"  NOTE: training uses topm_no_gold — gold outside top-M is EXCLUDED, not inserted.")

    # 3. Force-zero full val
    print("\n[debug] Force-zero full-val ...")
    m0_zero = canonical_eval_refiner(
        None, args.val_dir, tok_dev, r2s_np, device,
        force_zero=True, eval_batch_size=args.eval_batch_size,
        official_baseline=official_bl, fail_on_mismatch=fail_hard,
        variant_tag="STR-debug",
    )
    print(f"  force_zero_nll = {m0_zero['covered_nll']:.6f}")
    print(f"  coverage       = {m0_zero['coverage']:.6f}")
    print(f"  fingerprint    = {m0_zero['dataset_fingerprint']}")
    if official_bl:
        check_baseline_match(m0_zero, official_bl, fail_hard,
                             context="STR-debug/force_zero", check_nll=True)

    # 4. With-delta full val (STR eval, gold force-included)
    print("\n[debug] With-delta full-val (STR eval) ...")
    m0_with = canonical_eval_ctf(
        model, args.val_dir, tok_dev, r2s_np, device,
        eval_batch_size=args.eval_batch_size,
        official_baseline=official_bl, fail_on_mismatch=fail_hard,
        variant_tag="STR-debug",
    )
    print(f"  with_delta_nll           = {m0_with['covered_nll']:.6f}")
    print(f"  coverage                 = {m0_with['coverage']:.6f}")
    print(f"  fingerprint              = {m0_with['dataset_fingerprint']}")
    print(f"  max_abs_delta            = {m0_with['delta_abs_max']:.2e}")
    print(f"  gold_force_included_rate = {m0_with['gold_force_included_rate']:.4f}")

    nll_diff = abs(m0_with["covered_nll"] - m0_zero["covered_nll"])
    fp_match = m0_with["dataset_fingerprint"] == m0_zero["dataset_fingerprint"]
    print(f"\n[debug] identity check: nll_diff={nll_diff:.2e}  fp_match={fp_match}")
    if nll_diff < 1e-4 and fp_match:
        print("[debug] PASS — STR is exact identity at step 0")
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
        run_str_debug(args, d_model, n_fine, n_super, r2s_np, tok_emb_w, device)
        return

    if not args.train_dir:
        raise RuntimeError("--train_dir required for training")
    if not args.output_dir:
        raise RuntimeError("--output_dir required for training")

    train_str(args, d_model, n_fine, n_super, r2s_np, tok_emb_w, device)


def _parse():
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--train_dir",        default="")
    p.add_argument("--val_dir",          required=True)
    p.add_argument("--small_ckpt",       required=True)
    p.add_argument("--super_map",        default=None)
    p.add_argument("--output_dir",       default="")
    p.add_argument("--official_baseline",default=None)
    # Architecture
    p.add_argument("--selected_M",  type=int,   default=256)
    p.add_argument("--refiner_dim", type=int,   default=256)
    p.add_argument("--num_layers",  type=int,   default=2)
    p.add_argument("--num_heads",   type=int,   default=4)
    p.add_argument("--ff_mult",     type=int,   default=4)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--n_fine",      type=int,   default=128)
    p.add_argument("--n_super",     type=int,   default=24)
    # Filter
    p.add_argument("--train_filter",   default="boundary")
    p.add_argument("--gate_filter",    default=None)
    p.add_argument("--margin_thresh",  type=float, default=0.1)
    p.add_argument("--entropy_thresh", type=float, default=2.0)
    # Training
    p.add_argument("--steps",           type=int,   default=10_000)
    p.add_argument("--eval_every",      type=int,   default=1_000)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--eval_batch_size", type=int,   default=32)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--lambda_kl",       type=float, default=0.01)
    p.add_argument("--lambda_delta",    type=float, default=1e-4)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    # Selection mode (must remain topm_no_gold for all canonical evals)
    p.add_argument("--train_selection_mode", default="topm_no_gold",
                   choices=["topm_no_gold"],
                   help="Candidate selection mode for training. Only topm_no_gold is valid.")
    p.add_argument("--eval_selection_mode",  default="topm_no_gold",
                   choices=["topm_no_gold"],
                   help="Candidate selection mode for eval. Only topm_no_gold is valid.")
    p.add_argument("--drop_train_gold_not_selected", action="store_true",
                   help="Only train on covered positions where gold is naturally in top-M.")
    # Flags
    p.add_argument("--fail_on_baseline_mismatch", action="store_true")
    p.add_argument("--eval_before_train",         action="store_true")
    p.add_argument("--debug_only",                action="store_true",
                   help="Run identity + baseline smoke tests, then exit.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
