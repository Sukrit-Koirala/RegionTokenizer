#!/usr/bin/env python3
"""
Path-conditioned refiner variants for the region-tokenizer pipeline.

All variants share the same forward signature:
  (h_prime, cand_tok, cand_fine, cand_super, cand_mask, token_emb) -> scores (B, C)

Returned scores always include the base candidate logit
  base_logit = h_prime @ token_emb.weight[cand_tok]
plus a learned residual initialized near zero so training starts from the
frozen backbone's predictions.

Padding positions (cand_mask=False) are filled with -1e9 before return.

Variants
--------
A  BiasOnlyRefiner   — per-region additive bias; fewest parameters
B  DotProductRefiner — low-rank dot-product residual conditioned on path context
C  MLPRefiner        — per-candidate MLP delta; most expressive
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Variant A ─────────────────────────────────────────────────────────────────

class BiasOnlyRefiner(nn.Module):
    """
    score = base_logit + fine_bias[cand_fine] + super_bias[cand_super]

    Parameters: n_fine + n_super  (typically ~200 scalars).
    """

    def __init__(self, n_fine: int, n_super: int) -> None:
        super().__init__()
        # padding_idx absorbs -1 region IDs (mapped via clamp to n_fine/n_super)
        self.fine_bias  = nn.Embedding(n_fine  + 1, 1, padding_idx=n_fine)
        self.super_bias = nn.Embedding(n_super + 1, 1, padding_idx=n_super)
        nn.init.zeros_(self.fine_bias.weight)
        nn.init.zeros_(self.super_bias.weight)

    def forward(
        self,
        h_prime:    torch.Tensor,   # (B, d_model)
        cand_tok:   torch.Tensor,   # (B, C) long  — token IDs, -1=pad
        cand_fine:  torch.Tensor,   # (B, C) long  — fine region IDs, -1=pad
        cand_super: torch.Tensor,   # (B, C) long  — super-region IDs, -1=pad
        cand_mask:  torch.Tensor,   # (B, C) bool  — True where valid
        token_emb:  nn.Embedding,   # (V, d_model) frozen
    ) -> torch.Tensor:              # (B, C) scores
        tok_e  = token_emb(cand_tok.clamp(min=0))                       # (B, C, d)
        base   = (h_prime.unsqueeze(1) * tok_e).sum(-1)                 # (B, C)
        f_bias = self.fine_bias(cand_fine.clamp(min=0)).squeeze(-1)     # (B, C)
        s_bias = self.super_bias(cand_super.clamp(min=0)).squeeze(-1)   # (B, C)
        return (base + f_bias + s_bias).masked_fill(~cand_mask, -1e9)


# ── Variant B ─────────────────────────────────────────────────────────────────

class DotProductRefiner(nn.Module):
    """
    path_ctx = masked mean of super_emb[cand_super] over valid candidates
    q        = Wq(h_prime + path_ctx)
    k        = Wk(token_emb[cand])
    residual = (q · k) / sqrt(d_head)
    score    = base_logit + exp(log_scale) * residual

    log_scale starts at -2 so exp(-2) ≈ 0.14 — residual begins small.
    """

    def __init__(self, d_model: int, n_fine: int, n_super: int,
                 d_head: int = 64) -> None:
        super().__init__()
        self.d_head    = d_head
        self.Wq        = nn.Linear(d_model, d_head, bias=False)
        self.Wk        = nn.Linear(d_model, d_head, bias=False)
        self.super_emb = nn.Embedding(n_super + 1, d_model, padding_idx=n_super)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.Wq.weight, std=0.01)
        nn.init.normal_(self.Wk.weight, std=0.01)
        nn.init.zeros_(self.super_emb.weight)

    def forward(
        self,
        h_prime:    torch.Tensor,
        cand_tok:   torch.Tensor,
        cand_fine:  torch.Tensor,
        cand_super: torch.Tensor,
        cand_mask:  torch.Tensor,
        token_emb:  nn.Embedding,
    ) -> torch.Tensor:
        tok_e = token_emb(cand_tok.clamp(min=0))               # (B, C, d)
        base  = (h_prime.unsqueeze(1) * tok_e).sum(-1)         # (B, C)

        # path context: masked mean of super-region embeddings
        s_emb    = self.super_emb(cand_super.clamp(min=0))     # (B, C, d)
        mask_f   = cand_mask.float().unsqueeze(-1)              # (B, C, 1)
        path_ctx = (s_emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)  # (B, d)

        q        = self.Wq(h_prime + path_ctx)                 # (B, d_head)
        k        = self.Wk(tok_e)                              # (B, C, d_head)
        residual = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(self.d_head)  # (B, C)
        scores   = base + self.log_scale.exp() * residual
        return scores.masked_fill(~cand_mask, -1e9)


# ── Variant C ─────────────────────────────────────────────────────────────────

class MLPRefiner(nn.Module):
    """
    feat  = concat(h_prime, tok_emb[cand], fine_emb[cand_fine], super_emb[cand_super])
    delta = MLP(feat)   — single hidden layer, GELU
    score = base_logit + delta

    All weights start near zero; fine/super embeddings zero-initialized.
    """

    def __init__(self, d_model: int, n_fine: int, n_super: int,
                 d_region: int = 32, d_hidden: int = 128) -> None:
        super().__init__()
        self.fine_emb  = nn.Embedding(n_fine  + 1, d_region, padding_idx=n_fine)
        self.super_emb = nn.Embedding(n_super + 1, d_region, padding_idx=n_super)
        feat_dim = d_model + d_model + d_region + d_region
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, d_hidden, bias=True),
            nn.GELU(),
            nn.Linear(d_hidden, 1, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.01)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        nn.init.zeros_(self.fine_emb.weight)
        nn.init.zeros_(self.super_emb.weight)

    def forward(
        self,
        h_prime:    torch.Tensor,
        cand_tok:   torch.Tensor,
        cand_fine:  torch.Tensor,
        cand_super: torch.Tensor,
        cand_mask:  torch.Tensor,
        token_emb:  nn.Embedding,
    ) -> torch.Tensor:
        B, C   = cand_tok.shape
        tok_e  = token_emb(cand_tok.clamp(min=0))               # (B, C, d_model)
        fine_e = self.fine_emb(cand_fine.clamp(min=0))          # (B, C, d_region)
        sup_e  = self.super_emb(cand_super.clamp(min=0))        # (B, C, d_region)
        h_exp  = h_prime.unsqueeze(1).expand(-1, C, -1)         # (B, C, d_model)
        base   = (h_prime.unsqueeze(1) * tok_e).sum(-1)         # (B, C)
        feat   = torch.cat([h_exp, tok_e, fine_e, sup_e], dim=-1)  # (B, C, feat_dim)
        delta  = self.mlp(feat).squeeze(-1)                     # (B, C)
        return (base + delta).masked_fill(~cand_mask, -1e9)


# ── Factory ───────────────────────────────────────────────────────────────────

REFINER_VARIANTS = {"A": BiasOnlyRefiner, "B": DotProductRefiner, "C": MLPRefiner}


def build_refiner(variant: str, d_model: int, n_fine: int, n_super: int,
                  **kwargs) -> nn.Module:
    """
    variant: 'A', 'B', or 'C'
    kwargs forwarded to variant constructor (d_head for B; d_region, d_hidden for C).
    """
    cls = REFINER_VARIANTS.get(variant.upper())
    if cls is None:
        raise ValueError(f"Unknown variant {variant!r}; choose A, B, or C.")
    if cls is BiasOnlyRefiner:
        return cls(n_fine=n_fine, n_super=n_super)
    return cls(d_model=d_model, n_fine=n_fine, n_super=n_super, **kwargs)
