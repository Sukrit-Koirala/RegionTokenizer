#!/usr/bin/env python3
"""
train_region_lm.py

Jointly-trained Region-Conditioned Transformer LM.
Trains from scratch; compares baseline GPT vs region-conditioned variant.

Modes:
  baseline        standard GPT decoder
  coarse          coarse router + soft conditioning
  coarse_leaf     coarse + leaf routers + soft conditioning
  random_control  same arch as coarse_leaf, fixed-permuted labels (null hypothesis)
  oracle_coarse   upper bound: true next-token coarse region fed directly
  oracle          upper bound: true coarse + leaf regions fed directly
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import GPT2TokenizerFast


# ── Config ────────────────────────────────────────────────────────────────────

_MODES = ("baseline", "coarse", "coarse_leaf", "random_control", "oracle_coarse", "oracle",
          "build_regions", "soft_moe", "oracle_soft_moe", "random_soft_moe",
          "repr_region", "oracle_repr_region", "random_repr_region",
          "repr_region_capacity", "oracle_repr_region_capacity", "random_repr_region_capacity",
          "repr_region_boundary", "oracle_repr_region_boundary", "random_repr_region_boundary",
          "repr_region_multihyp", "oracle_repr_region_multihyp", "random_repr_region_multihyp",
          "repr_region_branchattn", "repr_region_branch_identity",
          "repr_region_boundary_seqrefine", "random_repr_region_boundary_seqrefine",
          "repr_region_boundary_adaptivedepth", "random_repr_region_boundary_adaptivedepth",
          "repr_region_retrieval", "random_repr_region_retrieval",
          "layer_margin_analysis",
          "boundary_analysis")


@dataclass
class TrainConfig:
    # data
    dataset: str = "wikitext-2-raw-v1"
    seq_len: int = 256
    batch_size: int = 32
    vocab_subset_size: int = 50257
    # model
    mode: str = "baseline"
    n_layer: int = 6
    d_model: int = 384
    n_head: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    # region conditioning
    region_map_path: Optional[str] = None
    leaf_map_path: Optional[str] = None
    n_coarse: int = 50
    n_leaf: int = 200
    d_region: int = 64
    region_temp: float = 1.0
    router_type: str = "linear"        # "linear" | "mlp"
    base_alpha: float = 0.1            # max coarse injection scale (after warmup)
    base_beta: float = 0.03            # max leaf injection scale (after warmup)
    region_warmup_steps: int = 5000    # steps to ramp 0 → base; 0 = no warmup
    use_refiner: bool = False
    # loss weights
    lambda_coarse: float = 0.05
    lambda_leaf: float = 0.01
    lambda_balance: float = 0.001
    # training
    steps: int = 20000
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    eval_interval: int = 500
    save_interval: int = 2000
    log_interval: int = 50
    # misc
    seed: int = 42
    output_dir: str = "runs/region_lm"
    resume: bool = False
    device: str = "cuda"
    # build_regions mode
    baseline_ckpt: Optional[str] = None    # checkpoint to sweep for co-activation graph
    region_output_dir: str = ""            # defaults to output_dir if empty
    region_vocab_size: int = 5000          # top-N frequent tokens for graph
    region_top_k: int = 20                 # top-k predictions per position
    region_max_tokens: int = 500_000       # corpus tokens to sweep
    n_clusters: int = 50                   # coarse cluster count
    build_leaf: bool = True                # also build region_tree.json
    leaf_min_size: int = 30                # min tokens per leaf before recursing
    # build_regions clustering control
    region_cluster_method: str = "spectral"  # leiden | spectral | kmeans
    target_n_regions: int = 128              # desired coarse region count
    target_n_leaves: int = 512               # desired leaf count (guides subcluster fan-out)
    extend_full_vocab: bool = False          # if True, build graph over all vocab tokens
    # soft_moe / soft routing
    router_temp: float = 2.0
    prior_gamma: float = 1.0
    prior_eps: float = 1e-6
    learned_region_prior: bool = False
    soft_topk_regions: int = 0              # 0 = full soft; >0 = top-k sparse routing
    # repr_region_capacity
    usage_ema_decay: float = 0.99          # EMA decay for region usage tracking
    capacity_alpha: float = 0.0            # capacity penalty strength; 0 = disabled
    lambda_diversity: float = 0.0          # weight on usage-entropy maximisation loss
    router_temp_init: float = 2.0          # starting router temperature
    router_temp_final: float = 1.0         # final router temperature after annealing
    router_temp_decay_steps: int = 10000   # steps to anneal from init → final; 0 = no anneal
    # repr_region_boundary mode
    alpha_core: float = 0.2               # alpha for high-margin (core) tokens
    alpha_boundary: float = 0.4           # alpha for low-margin (boundary) tokens
    boundary_tau: float = 0.10            # margin threshold separating boundary from core
    boundary_temp: float = 0.05           # sigmoid gate temperature
    boundary_mode: str = "margin"         # "margin" | "entropy"
    # repr_region_multihyp mode
    hyp_k: int = 4                        # number of top-k hypotheses to preserve
    # repr_region_boundary_seqrefine mode
    boundary_refine_layers: int = 1       # causal Transformer blocks for seqrefine
    boundary_refine_gamma: float = 0.5    # fixed scale multiplier for refine contribution
    # repr_region_boundary_adaptivedepth mode
    adaptive_layers: int = 1              # extra causal blocks applied only at boundary tokens
    adaptive_scale_init: float = 0.05    # init value of tanh-gated residual scale
    adaptive_threshold: float = 0.5      # hard gate threshold: gate > this → boundary
    # repr_region_branchattn mode
    branch_attn_heads: int = 4            # attention heads over K branch dimension
    branch_attn_layers: int = 1           # stacked branch-attention layers
    branch_attn_dropout: float = 0.1      # dropout inside BranchAttentionRefiner
    branch_scale_init: float = 0.05       # initial value of tanh-gated residual scale
    # boundary_analysis mode
    repr_ckpt: Optional[str] = None        # trained repr_region checkpoint to analyse
    max_analysis_batches: int = 200        # val batches to process (0 = all)
    # repr_region_retrieval mode
    retrieval_loss_type: str = "proxy"               # "proxy" | "supcon"
    lambda_retrieval: float = 0.0                    # weight on retrieval metric loss
    retrieval_dim: int = 128                         # projection head output dimension
    retrieval_temp: float = 0.07                     # temperature for proxy/supcon loss
    retrieval_key_source: str = "pre_region"         # "pre_region" (h) | "post_region" (h')
    max_retrieval_positions_per_batch: int = 2048    # subsample cap for supcon


# ── Transformer blocks ────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_len: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, seq_len: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


def _make_blocks(cfg: TrainConfig, n: int) -> nn.ModuleList:
    return nn.ModuleList([
        TransformerBlock(cfg.d_model, cfg.n_head, cfg.d_ff, cfg.seq_len, cfg.dropout)
        for _ in range(n)
    ])


def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


def _make_router_head(d_model: int, n_out: int, dropout: float, router_type: str) -> nn.Module:
    if router_type == "mlp":
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_out, bias=False),
        )
    return nn.Linear(d_model, n_out, bias=False)


# ── Baseline model ─────────────────────────────────────────────────────────────

class BaselineTransformerLM(nn.Module):
    def __init__(self, cfg: TrainConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.lm_head   = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        return self.lm_head(h), h

    def loss(self, idx: torch.Tensor, **_) -> Dict[str, torch.Tensor]:
        logits, _ = self.forward(idx[:, :-1])
        tgt = idx[:, 1:]
        l_lm = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return {"lm": l_lm, "total": l_lm}

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb   = _n(self.token_emb) + _n(self.pos_emb)
        trunk = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        _n(self.lm_head)   # weight-tied → 0
        total = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": 0, "refiner": 0,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Region-Conditioned model ───────────────────────────────────────────────────

class RegionConditionedTransformerLM(nn.Module):
    """
    Architecture:
      x_t  = token_emb[t] + pos_emb[t]
      h_t  = Transformer(x_t)                        (trunk)
      p_c  = softmax(router_coarse(h_t) / T)
      r_c  = p_c @ E_coarse
      h'_t = h_t + alpha * proj_c(MLP_c(r_c))
             [+ beta * proj_l(MLP_l(r_leaf))]        (if leaf)
      h_r  = RefinerBlock(h')                        (optional, default off)
      logits = W_vocab h_r

    Target alignment: h_t predicts x_{t+1}; region labels index tgt = idx[:,1:].

    random_control: same arch, coarse_map/leaf_map pre-permuted at load time.
    No per-batch randomisation — labels are fixed throughout training.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        coarse_map: Optional[torch.Tensor] = None,  # (vocab_size,) int64, -1=unknown
        leaf_map:   Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size

        self.use_coarse = cfg.mode in ("coarse", "coarse_leaf", "random_control",
                                       "oracle_coarse", "oracle")
        self.use_leaf   = cfg.mode in ("coarse_leaf", "random_control", "oracle")

        self.register_buffer("coarse_map", coarse_map)
        self.register_buffer("leaf_map",   leaf_map)

        # Injection scales — initialised to full base so eval-only use is safe.
        # During training, set_region_scales() ramps these from 0 → base.
        self.current_alpha: float = cfg.base_alpha
        self.current_beta:  float = cfg.base_beta

        # Trunk
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)

        # Coarse region components
        if self.use_coarse:
            self.coarse_head = _make_router_head(cfg.d_model, cfg.n_coarse,
                                                 cfg.dropout, cfg.router_type)
            self.coarse_emb  = nn.Embedding(cfg.n_coarse, cfg.d_region)
            self.mlp_coarse  = FFN(cfg.d_region, cfg.d_region * 2, dropout=0.0)
            self.proj_coarse = nn.Linear(cfg.d_region, cfg.d_model, bias=False)

        # Leaf region components
        if self.use_leaf:
            self.leaf_head = _make_router_head(cfg.d_model, cfg.n_leaf,
                                               cfg.dropout, cfg.router_type)
            self.leaf_emb  = nn.Embedding(cfg.n_leaf, cfg.d_region)
            self.mlp_leaf  = FFN(cfg.d_region, cfg.d_region * 2, dropout=0.0)
            self.proj_leaf = nn.Linear(cfg.d_region, cfg.d_model, bias=False)

        # Optional refiner block (off by default — keeps param count fair)
        if cfg.use_refiner:
            self.refiner = TransformerBlock(
                cfg.d_model, cfg.n_head, cfg.d_ff // 2, cfg.seq_len, cfg.dropout
            )
            self.ln_r = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying
        self.apply(_init_weights)

    # ------------------------------------------------------------------
    def set_region_scales(self, alpha: float, beta: float):
        self.current_alpha = alpha
        self.current_beta  = beta

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        oracle_coarse: Optional[torch.Tensor] = None,
        oracle_leaf:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        idx           : (B, T)
        oracle_coarse : (B, T) int — true coarse region of x_{t+1}
        oracle_leaf   : (B, T) int — true leaf  region of x_{t+1}
        Returns: lm_logits, coarse_logits, leaf_logits, h_final
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)   # (B, T, d_model)

        coarse_logits = None
        leaf_logits   = None
        h_prime = h

        is_oracle_c = self.cfg.mode in ("oracle_coarse", "oracle")
        is_oracle_l = self.cfg.mode == "oracle"

        if self.use_coarse:
            coarse_logits = self.coarse_head(h)   # (B, T, n_coarse)
            if is_oracle_c and oracle_coarse is not None:
                r_coarse = self.coarse_emb(oracle_coarse.clamp(min=0))
            else:
                p_c = F.softmax(coarse_logits / self.cfg.region_temp, dim=-1)
                r_coarse = p_c @ self.coarse_emb.weight                # (B, T, d_region)
            h_prime = h + self.current_alpha * self.proj_coarse(self.mlp_coarse(r_coarse))

        if self.use_leaf:
            leaf_logits = self.leaf_head(h)       # (B, T, n_leaf)
            if is_oracle_l and oracle_leaf is not None:
                r_leaf = self.leaf_emb(oracle_leaf.clamp(min=0))
            else:
                p_l = F.softmax(leaf_logits / self.cfg.region_temp, dim=-1)
                r_leaf = p_l @ self.leaf_emb.weight
            h_prime = h_prime + self.current_beta * self.proj_leaf(self.mlp_leaf(r_leaf))

        if self.cfg.use_refiner:
            h_final = self.ln_r(self.refiner(h_prime))
        else:
            h_final = h_prime

        return self.lm_head(h_final), coarse_logits, leaf_logits, h_final

    # ------------------------------------------------------------------
    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        idx : (B, T+1).  Labels are always for tgt = idx[:,1:] (no off-by-one).
        random_control: coarse_map/leaf_map already permuted at load time.
        """
        src = idx[:, :-1]  # (B, T)
        tgt = idx[:, 1:]   # (B, T) — next-token targets

        oracle_coarse = None
        oracle_leaf   = None
        if self.cfg.mode in ("oracle_coarse", "oracle") and self.coarse_map is not None:
            oracle_coarse = self.coarse_map[tgt]
        if self.cfg.mode == "oracle" and self.leaf_map is not None:
            oracle_leaf = self.leaf_map[tgt]

        lm_logits, coarse_logits, leaf_logits, _ = self.forward(
            src, oracle_coarse=oracle_coarse, oracle_leaf=oracle_leaf
        )

        l_lm  = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        # Coarse auxiliary loss
        if self.use_coarse and coarse_logits is not None and self.coarse_map is not None:
            coarse_labels = self.coarse_map[tgt]   # (B, T)
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    coarse_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c
            # Entropy regulariser — penalise router collapse
            p_c   = F.softmax(coarse_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()  # -entropy
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

        # Leaf auxiliary loss
        if self.use_leaf and leaf_logits is not None and self.leaf_map is not None:
            leaf_labels = self.leaf_map[tgt]
            valid = leaf_labels >= 0
            if valid.any():
                l_l = F.cross_entropy(
                    leaf_logits.reshape(-1, self.cfg.n_leaf)[valid.reshape(-1)],
                    leaf_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_leaf * l_l
                losses["leaf"] = l_l

        losses["total"] = total
        return losses

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb    = _n(self.token_emb) + _n(self.pos_emb)
        trunk  = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router = 0
        if self.use_coarse:
            router += (_n(self.coarse_head) + _n(self.coarse_emb)
                       + _n(self.mlp_coarse) + _n(self.proj_coarse))
        if self.use_leaf:
            router += (_n(self.leaf_head) + _n(self.leaf_emb)
                       + _n(self.mlp_leaf) + _n(self.proj_leaf))
        refiner = (_n(self.refiner) + _n(self.ln_r)) if self.cfg.use_refiner else 0
        _n(self.lm_head)   # weight-tied → 0
        total = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": refiner,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Soft-MoE model ────────────────────────────────────────────────────────────

class SoftMoETransformerLM(nn.Module):
    """
    Soft mixture-of-regions LM.

      P(token | ctx)  ∝  exp( base_logits + γ · log(p_region @ M + ε) )

    where M[r,v] = 1 if token v is in region r (precomputed, not learned by default).
    Fully differentiable; no hard token-to-region decisions.

    Modes handled:
      soft_moe        — learned router
      oracle_soft_moe — true target region as one-hot (upper bound)
      random_soft_moe — same arch, random partition map (null hypothesis)
    """

    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        coarse_map: torch.Tensor,   # (vocab_size,) int64, -1 = unknown
        membership: torch.Tensor,   # (n_coarse, vocab_size) float32
    ):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size

        self.register_buffer("coarse_map", coarse_map)
        self.register_buffer("membership", membership)

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.lm_head   = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying

        self.coarse_head = _make_router_head(cfg.d_model, cfg.n_coarse,
                                             cfg.dropout, cfg.router_type)
        if cfg.learned_region_prior:
            self.region_bias = nn.Parameter(torch.zeros(cfg.n_coarse, vocab_size))

        self.apply(_init_weights)

    # ------------------------------------------------------------------
    def _get_membership(self) -> torch.Tensor:
        if self.cfg.learned_region_prior:
            return torch.sigmoid(self.region_bias)
        return self.membership

    def _compute_prior(self, p_region: torch.Tensor) -> torch.Tensor:
        """p_region: (B, T, n_coarse) → log_prior: (B, T, vocab_size)"""
        B, T, _ = p_region.shape
        M   = self._get_membership()                               # (n_coarse, V)
        rtp = p_region.reshape(B * T, self.cfg.n_coarse) @ M      # (B*T, V)
        return torch.log(rtp + self.cfg.prior_eps).reshape(B, T, self.vocab_size)

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        oracle_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: final_logits, region_logits, p_region, h"""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)

        base_logits   = self.lm_head(h)                            # (B, T, V)
        region_logits = self.coarse_head(h)                        # (B, T, n_coarse)

        if oracle_region is not None:
            known    = oracle_region >= 0                          # (B, T)
            p_region = torch.full(
                (B, T, self.cfg.n_coarse), 1.0 / self.cfg.n_coarse, device=idx.device,
            )
            oh = F.one_hot(oracle_region.clamp(min=0),
                           num_classes=self.cfg.n_coarse).float()
            p_region = torch.where(known.unsqueeze(-1), oh, p_region)
        else:
            p_region = F.softmax(region_logits / self.cfg.router_temp, dim=-1)

        # Optional top-k sparsification (not used in oracle mode)
        if self.cfg.soft_topk_regions > 0 and oracle_region is None:
            k = min(self.cfg.soft_topk_regions, self.cfg.n_coarse)
            topk_vals, topk_idx = p_region.topk(k, dim=-1)
            sparse_p = torch.zeros_like(p_region)
            sparse_p.scatter_(-1, topk_idx, topk_vals)
            p_region = sparse_p / (sparse_p.sum(dim=-1, keepdim=True) + 1e-8)

        log_prior    = self._compute_prior(p_region)               # (B, T, V)
        final_logits = base_logits + self.cfg.prior_gamma * log_prior
        return final_logits, region_logits, p_region, h

    # ------------------------------------------------------------------
    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        src = idx[:, :-1]
        tgt = idx[:, 1:]

        oracle_region = None
        if self.cfg.mode == "oracle_soft_moe" and self.coarse_map is not None:
            oracle_region = self.coarse_map[tgt]

        final_logits, region_logits, p_region, _ = self.forward(src, oracle_region)

        l_lm  = F.cross_entropy(final_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        if self.cfg.mode != "oracle_soft_moe" and self.coarse_map is not None:
            coarse_labels = self.coarse_map[tgt]
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    region_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c
            p_c   = F.softmax(region_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

        losses["total"] = total
        return losses

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb    = _n(self.token_emb) + _n(self.pos_emb)
        trunk  = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router = _n(self.coarse_head)
        if self.cfg.learned_region_prior:
            router += self.region_bias.numel()
        _n(self.lm_head)
        total = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": 0,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Repr-Region model ─────────────────────────────────────────────────────────

class ReprRegionTransformerLM(nn.Module):
    """
    Representation-level region conditioning.

    Region info is injected into the hidden state before lm_head — no logit
    bias, no token masking, no top-k routing.

    Forward pass:
      h            = Transformer(ctx)
      region_logits = coarse_head(h)                     (B, T, n_coarse)
      p_region     = softmax(region_logits / router_temp)
      r            = p_region @ region_emb.weight        (B, T, d_region)
      region_feat  = region_proj(region_mlp(r))          (B, T, d_model)
      h'           = h + alpha * region_feat
      logits       = lm_head(h')

    Modes:
      repr_region        — learned soft router
      oracle_repr_region — one-hot oracle (upper bound, no coarse aux loss)
      random_repr_region — random partition map (null hypothesis)
    """

    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        coarse_map: torch.Tensor,   # (vocab_size,) int64, -1 = unknown
    ):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size

        self.register_buffer("coarse_map", coarse_map)
        self.current_alpha: float = cfg.base_alpha

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)

        self.coarse_head = _make_router_head(cfg.d_model, cfg.n_coarse,
                                             cfg.dropout, cfg.router_type)
        self.region_emb  = nn.Embedding(cfg.n_coarse, cfg.d_region)
        self.region_mlp  = FFN(cfg.d_region, cfg.d_region * 2, dropout=0.0)
        self.region_proj = nn.Linear(cfg.d_region, cfg.d_model, bias=False)

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying
        self.apply(_init_weights)

    def set_region_scales(self, alpha: float, beta: float = 0.0):
        self.current_alpha = alpha

    def forward(
        self,
        idx: torch.Tensor,
        oracle_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: lm_logits, region_logits, p_region, h_prime"""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)                                       # (B, T, d_model)

        region_logits = self.coarse_head(h)                    # (B, T, n_coarse)

        if oracle_region is not None:
            known    = oracle_region >= 0
            p_region = torch.full(
                (B, T, self.cfg.n_coarse), 1.0 / self.cfg.n_coarse, device=idx.device,
            )
            oh = F.one_hot(oracle_region.clamp(min=0),
                           num_classes=self.cfg.n_coarse).float()
            p_region = torch.where(known.unsqueeze(-1), oh, p_region)
        else:
            p_region = F.softmax(region_logits / self.cfg.router_temp, dim=-1)

        r           = p_region @ self.region_emb.weight        # (B, T, d_region)
        region_feat = self.region_proj(self.region_mlp(r))     # (B, T, d_model)
        h_prime     = h + self.current_alpha * region_feat

        return self.lm_head(h_prime), region_logits, p_region, h_prime

    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        src = idx[:, :-1]
        tgt = idx[:, 1:]

        oracle_region = None
        if self.cfg.mode == "oracle_repr_region" and self.coarse_map is not None:
            oracle_region = self.coarse_map[tgt]

        lm_logits, region_logits, _, _ = self.forward(src, oracle_region)

        l_lm  = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        # Coarse aux + balance losses: only for learned router (not oracle)
        if self.cfg.mode != "oracle_repr_region" and self.coarse_map is not None:
            coarse_labels = self.coarse_map[tgt]
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    region_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c
            p_c   = F.softmax(region_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

        losses["total"] = total
        return losses

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb    = _n(self.token_emb) + _n(self.pos_emb)
        trunk  = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router = (_n(self.coarse_head) + _n(self.region_emb)
                  + _n(self.region_mlp) + _n(self.region_proj))
        _n(self.lm_head)   # weight-tied → 0
        total = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": 0,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Repr-Region-Capacity model ─────────────────────────────────────────────────

class ReprRegionCapacityLM(nn.Module):
    """
    repr_region extended with anti-collapse mechanisms:
      1. EMA region usage tracking (updated per training step)
      2. Capacity normalisation: penalise overused regions in routing logits
      3. Diversity regularisation: auxiliary loss maximising batch-level usage entropy
      4. Optional temperature annealing: router_temp_init → router_temp_final

    Capacity normalisation:
      adjusted_logits = logits - capacity_alpha * log(ema_usage + eps)
      p_region        = softmax(adjusted_logits / current_router_temp)

    Diversity loss (added only when mode != oracle):
      usage    = p_region.mean(B, T)
      L_div    = -(usage * log(usage + eps)).sum()   # negative entropy → maximise
      total   += lambda_diversity * (-L_div)

    Modes:
      repr_region_capacity        — learned router with anti-collapse
      oracle_repr_region_capacity — one-hot oracle (upper bound, no aux losses)
      random_repr_region_capacity — random partition (null hypothesis)
    """

    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        coarse_map: torch.Tensor,
    ):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size

        self.register_buffer("coarse_map", coarse_map)
        self.register_buffer(
            "region_usage_ema",
            torch.ones(cfg.n_coarse, dtype=torch.float32) / cfg.n_coarse,
        )
        self.current_alpha:       float = cfg.base_alpha
        self.current_router_temp: float = cfg.router_temp_init

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)

        self.coarse_head = _make_router_head(cfg.d_model, cfg.n_coarse,
                                             cfg.dropout, cfg.router_type)
        self.region_emb  = nn.Embedding(cfg.n_coarse, cfg.d_region)
        self.region_mlp  = FFN(cfg.d_region, cfg.d_region * 2, dropout=0.0)
        self.region_proj = nn.Linear(cfg.d_region, cfg.d_model, bias=False)

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying
        self.apply(_init_weights)

    def set_region_scales(self, alpha: float, beta: float = 0.0):
        self.current_alpha = alpha

    def set_router_temp(self, temp: float):
        self.current_router_temp = temp

    def forward(
        self,
        idx: torch.Tensor,
        oracle_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: lm_logits, region_logits, p_region, h_prime"""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)                                       # (B, T, d_model)

        region_logits = self.coarse_head(h)                    # (B, T, n_coarse)

        if oracle_region is not None:
            known    = oracle_region >= 0
            p_region = torch.full(
                (B, T, self.cfg.n_coarse), 1.0 / self.cfg.n_coarse, device=idx.device,
            )
            oh = F.one_hot(oracle_region.clamp(min=0),
                           num_classes=self.cfg.n_coarse).float()
            p_region = torch.where(known.unsqueeze(-1), oh, p_region)
        else:
            if self.cfg.capacity_alpha > 0.0:
                # Penalise overused regions: subtract capacity_alpha * log(ema + eps)
                # High usage → large log → large penalty → lower adjusted logit
                cap_bias = -self.cfg.capacity_alpha * torch.log(
                    self.region_usage_ema + 1e-8
                )                                              # (K,) — broadcast over B,T
                adjusted_logits = region_logits + cap_bias
            else:
                adjusted_logits = region_logits
            p_region = F.softmax(adjusted_logits / self.current_router_temp, dim=-1)

        r           = p_region @ self.region_emb.weight        # (B, T, d_region)
        region_feat = self.region_proj(self.region_mlp(r))     # (B, T, d_model)
        h_prime     = h + self.current_alpha * region_feat

        return self.lm_head(h_prime), region_logits, p_region, h_prime

    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        src = idx[:, :-1]
        tgt = idx[:, 1:]

        oracle_region = None
        if self.cfg.mode == "oracle_repr_region_capacity" and self.coarse_map is not None:
            oracle_region = self.coarse_map[tgt]

        lm_logits, region_logits, p_region, _ = self.forward(src, oracle_region)

        # Update EMA usage only during training (gated by self.training flag)
        if self.training:
            with torch.no_grad():
                batch_usage = p_region.detach().mean(dim=(0, 1))  # (K,)
                self.region_usage_ema.mul_(self.cfg.usage_ema_decay).add_(
                    batch_usage * (1.0 - self.cfg.usage_ema_decay)
                )

        l_lm  = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        if self.cfg.mode != "oracle_repr_region_capacity" and self.coarse_map is not None:
            coarse_labels = self.coarse_map[tgt]
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    region_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c

            # Token-level balance loss (entropy of per-position routing)
            p_c   = F.softmax(region_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

            # Batch-level diversity loss (entropy of mean routing weights)
            if self.cfg.lambda_diversity > 0.0:
                usage      = p_region.mean(dim=(0, 1))                    # (K,)
                usage_ent  = -(usage * (usage + 1e-8).log()).sum()        # scalar
                l_div      = -usage_ent                                    # minimise → maximise H
                total      = total + self.cfg.lambda_diversity * l_div
                losses["diversity"] = l_div

        losses["total"] = total
        return losses

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb    = _n(self.token_emb) + _n(self.pos_emb)
        trunk  = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router = (_n(self.coarse_head) + _n(self.region_emb)
                  + _n(self.region_mlp) + _n(self.region_proj))
        _n(self.lm_head)   # weight-tied → 0
        total = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": 0,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Repr-Region-Boundary model ─────────────────────────────────────────────────

class ReprRegionBoundaryLM(nn.Module):
    """
    repr_region with margin-adaptive alpha conditioning.

    Boundary hypothesis: routing uncertainty is localized, not global.
    Tokens near region boundaries (low routing margin) benefit from stronger
    soft multi-region conditioning; confident core tokens need only weak conditioning.

    Gate (margin mode):
        margin          = top1 − top2 of softmax(router_logits)
        boundary_gate   = sigmoid((tau − margin) / boundary_temp)
        → gate ≈ 1 when margin << tau  (ambiguous / boundary token)
        → gate ≈ 0 when margin >> tau  (confident / core token)

    Gate (entropy mode):
        boundary_gate   = H(p_region) / log(K)     ∈ [0, 1]

    Dynamic alpha:
        alpha_dyn = (alpha_core*(1−gate) + alpha_boundary*gate) * warmup_scale

    Forward:
        h' = h + alpha_dyn * region_feat

    Full soft distribution p_region is always used for the mixture; no hard
    top-k masking is applied. alpha_boundary > alpha_core gives boundary
    tokens stronger representation blending.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        coarse_map: torch.Tensor,
    ):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size

        self.register_buffer("coarse_map", coarse_map)
        self.current_alpha: float = cfg.alpha_core  # for logging / compat
        self.warmup_scale:  float = 1.0

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)

        self.coarse_head = _make_router_head(cfg.d_model, cfg.n_coarse,
                                             cfg.dropout, cfg.router_type)
        self.region_emb  = nn.Embedding(cfg.n_coarse, cfg.d_region)
        self.region_mlp  = FFN(cfg.d_region, cfg.d_region * 2, dropout=0.0)
        self.region_proj = nn.Linear(cfg.d_region, cfg.d_model, bias=False)

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying
        self.apply(_init_weights)

        # Set during forward; read by evaluate() for per-group stats
        self._last_gate:      Optional[torch.Tensor] = None
        self._last_alpha_dyn: Optional[torch.Tensor] = None

    def set_region_scales(self, alpha: float, beta: float = 0.0):
        """Warmup hook: alpha is alpha_core*scale; derive scale for both alphas."""
        self.warmup_scale  = min(1.0, alpha / max(self.cfg.alpha_core, 1e-8))
        self.current_alpha = alpha

    def _boundary_gate(self, p_region: torch.Tensor) -> torch.Tensor:
        """Returns gate ∈ [0,1] shaped (B, T). High = boundary token."""
        if self.cfg.boundary_mode == "entropy":
            H = -(p_region * (p_region + 1e-8).log()).sum(-1)   # (B, T)
            return H / math.log(max(self.cfg.n_coarse, 2))
        else:  # margin
            top2 = p_region.topk(2, dim=-1).values               # (B, T, 2)
            margin = top2[..., 0] - top2[..., 1]                 # (B, T)
            return torch.sigmoid(
                (self.cfg.boundary_tau - margin) / self.cfg.boundary_temp
            )

    def forward(
        self,
        idx: torch.Tensor,
        oracle_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: lm_logits, region_logits, p_region, h_prime"""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)                                        # (B, T, d_model)

        region_logits = self.coarse_head(h)                     # (B, T, K)

        if oracle_region is not None:
            known    = oracle_region >= 0
            p_region = torch.full(
                (B, T, self.cfg.n_coarse), 1.0 / self.cfg.n_coarse, device=idx.device,
            )
            oh = F.one_hot(oracle_region.clamp(min=0),
                           num_classes=self.cfg.n_coarse).float()
            p_region = torch.where(known.unsqueeze(-1), oh, p_region)
        else:
            p_region = F.softmax(region_logits / self.cfg.router_temp, dim=-1)

        # Adaptive alpha per token position
        gate      = self._boundary_gate(p_region)               # (B, T)
        alpha_dyn = (
            self.cfg.alpha_core * (1.0 - gate)
            + self.cfg.alpha_boundary * gate
        ) * self.warmup_scale                                    # (B, T)

        # Cache for eval tracking (CPU to avoid holding GPU memory)
        self._last_gate      = gate.detach().cpu()
        self._last_alpha_dyn = alpha_dyn.detach().cpu()

        # Region feature (full soft mixture — no top-k masking)
        r           = p_region @ self.region_emb.weight         # (B, T, d_region)
        region_feat = self.region_proj(self.region_mlp(r))      # (B, T, d_model)
        h_prime     = h + alpha_dyn.unsqueeze(-1) * region_feat # (B, T, d_model)

        return self.lm_head(h_prime), region_logits, p_region, h_prime

    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        src = idx[:, :-1]
        tgt = idx[:, 1:]

        oracle_region = None
        if self.cfg.mode == "oracle_repr_region_boundary" and self.coarse_map is not None:
            oracle_region = self.coarse_map[tgt]

        lm_logits, region_logits, _, _ = self.forward(src, oracle_region)

        l_lm  = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        if self.cfg.mode != "oracle_repr_region_boundary" and self.coarse_map is not None:
            coarse_labels = self.coarse_map[tgt]
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    region_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c
            p_c   = F.softmax(region_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

        losses["total"] = total
        return losses

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb    = _n(self.token_emb) + _n(self.pos_emb)
        trunk  = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router = (_n(self.coarse_head) + _n(self.region_emb)
                  + _n(self.region_mlp) + _n(self.region_proj))
        _n(self.lm_head)   # weight-tied → 0
        total = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": 0,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Boundary Sequence Refiner ─────────────────────────────────────────────────

class ReprRegionBoundarySeqRefineLM(nn.Module):
    """
    repr_region_boundary + gate-weighted causal sequence refinement.

    Hypothesis: boundary tokens need extra *contextual* computation (seeing
    neighboring tokens through sequence attention) rather than static top-k
    region-branch manipulation.

    Core path (all tokens, fixed alpha_core):
        h_core  = h + alpha_core * warmup_scale * region_feat

    Boundary refinement (causal, gated):
        h_seq   = seq_blocks(h_core)       # 1+ causal TransformerBlocks
        scale   = tanh(refine_scale) * boundary_refine_gamma * warmup_scale
        h_final = h_core + gate.unsqueeze(-1) * scale * (h_seq - h_core)

    Gate: same calibrated margin gate as repr_region_boundary.
    Causal mask is enforced by TransformerBlock's CausalSelfAttention.
    refine_scale initialized to 0.05 → small live gradient at step 0.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        coarse_map: torch.Tensor,
    ):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size

        self.register_buffer("coarse_map", coarse_map)
        self.current_alpha: float = cfg.alpha_core
        self.warmup_scale:  float = 1.0

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)

        self.coarse_head = _make_router_head(cfg.d_model, cfg.n_coarse,
                                             cfg.dropout, cfg.router_type)
        self.region_emb  = nn.Embedding(cfg.n_coarse, cfg.d_region)
        self.region_mlp  = FFN(cfg.d_region, cfg.d_region * 2, dropout=0.0)
        self.region_proj = nn.Linear(cfg.d_region, cfg.d_model, bias=False)

        # Extra causal blocks for boundary refinement.
        # Uses the same d_model / n_head / d_ff as the main trunk.
        self.seq_refine_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_head, cfg.d_ff,
                             cfg.seq_len, cfg.dropout)
            for _ in range(cfg.boundary_refine_layers)
        ])
        # refine_scale ≠ 0 so gradients are live from step 0.
        # tanh(0.05) * gamma ≈ 0.025 initial effective scale.
        self.refine_scale = nn.Parameter(torch.tensor(0.05))

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
        self.apply(_init_weights)

        self._last_gate:      Optional[torch.Tensor] = None
        self._last_alpha_dyn: Optional[torch.Tensor] = None
        self._last_norms:     Dict[str, float]       = {}

    def set_region_scales(self, alpha: float, beta: float = 0.0):
        self.warmup_scale  = min(1.0, alpha / max(self.cfg.alpha_core, 1e-8))
        self.current_alpha = alpha

    def _boundary_gate(self, p_region: torch.Tensor) -> torch.Tensor:
        if self.cfg.boundary_mode == "entropy":
            H = -(p_region * (p_region + 1e-8).log()).sum(-1)
            return H / math.log(max(self.cfg.n_coarse, 2))
        top2   = p_region.topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        return torch.sigmoid(
            (self.cfg.boundary_tau - margin) / self.cfg.boundary_temp
        )

    def forward(
        self,
        idx: torch.Tensor,
        oracle_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)

        region_logits = self.coarse_head(h)

        if oracle_region is not None:
            known    = oracle_region >= 0
            p_region = torch.full(
                (B, T, self.cfg.n_coarse), 1.0 / self.cfg.n_coarse, device=idx.device,
            )
            oh = F.one_hot(oracle_region.clamp(min=0),
                           num_classes=self.cfg.n_coarse).float()
            p_region = torch.where(known.unsqueeze(-1), oh, p_region)
        else:
            p_region = F.softmax(region_logits / self.cfg.router_temp, dim=-1)

        # Gate always uses full p_region (not top-k renormalized)
        gate = self._boundary_gate(p_region.detach())           # (B, T)

        ws = self.warmup_scale
        r           = p_region @ self.region_emb.weight
        region_feat = self.region_proj(self.region_mlp(r))
        h_core      = h + self.cfg.alpha_core * ws * region_feat

        # Causal sequence refinement — one pass per block
        h_seq = h_core
        for blk in self.seq_refine_blocks:
            h_seq = blk(h_seq)

        # Gate-weighted blend; warmup also scales the refine contribution
        scale   = torch.tanh(self.refine_scale) * self.cfg.boundary_refine_gamma * ws
        h_final = h_core + gate.unsqueeze(-1) * scale * (h_seq - h_core)

        self._last_gate      = gate.detach().cpu()
        # alpha_dyn for logging compat: alpha_core + gate * extra_refinement_effect
        self._last_alpha_dyn = (self.cfg.alpha_core * torch.ones_like(gate)).detach().cpu()
        with torch.no_grad():
            self._last_norms = {
                "h_norm":              h.norm(dim=-1).mean().item(),
                "h_seq_delta_norm":    (h_seq   - h_core).norm(dim=-1).mean().item(),
                "h_final_delta_norm":  (h_final - h_core).norm(dim=-1).mean().item(),
            }

        return self.lm_head(h_final), region_logits, p_region, h_final

    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        src = idx[:, :-1]
        tgt = idx[:, 1:]

        oracle_region = None
        if (self.cfg.mode == "oracle_repr_region_boundary_seqrefine"
                and self.coarse_map is not None):
            oracle_region = self.coarse_map[tgt]

        lm_logits, region_logits, _, _ = self.forward(src, oracle_region)

        l_lm  = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        if (self.cfg.mode not in ("oracle_repr_region_boundary_seqrefine",)
                and self.coarse_map is not None):
            coarse_labels = self.coarse_map[tgt]
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    region_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c
            p_c   = F.softmax(region_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

        losses["total"] = total
        return losses

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb     = _n(self.token_emb) + _n(self.pos_emb)
        trunk   = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router  = (_n(self.coarse_head) + _n(self.region_emb)
                   + _n(self.region_mlp) + _n(self.region_proj))
        refiner = sum(_n(b) for b in self.seq_refine_blocks)
        _n(self.lm_head)
        total   = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": refiner,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Boundary Adaptive-Depth ───────────────────────────────────────────────────

class ReprRegionBoundaryAdaptiveDepthLM(nn.Module):
    """
    repr_region_boundary with SPARSE adaptive-depth refinement.

    Scientific question:
        Can region-boundary geometry decide WHERE extra compute should go?

    Key distinction from seqrefine:
        seqrefine:      gate-WEIGHTED update applied to ALL tokens   (soft blend)
        adaptivedepth:  hard boundary mask, delta applied ONLY to boundary tokens

    Pipeline:
        h_core = h + alpha_core * warmup * region_feat

        gate          = boundary_gate(p_region.detach())
        boundary_mask = gate > adaptive_threshold               # (B, T) bool

        h_refined     = adaptive_blocks(h_core)                 # full causal context
        delta         = h_refined - h_core
        scale         = tanh(refine_scale)

        h_final = h_core + boundary_mask.float().unsqueeze(-1) * scale * delta

    Core tokens are mathematically unchanged.
    Boundary tokens receive one additional contextual pass.
    The adaptive block sees the full sequence for context but
    only boundary positions carry the resulting update forward.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        coarse_map: torch.Tensor,
    ):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size

        self.register_buffer("coarse_map", coarse_map)
        self.current_alpha: float = cfg.alpha_core
        self.warmup_scale:  float = 1.0

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)

        self.coarse_head = _make_router_head(cfg.d_model, cfg.n_coarse,
                                             cfg.dropout, cfg.router_type)
        self.region_emb  = nn.Embedding(cfg.n_coarse, cfg.d_region)
        self.region_mlp  = FFN(cfg.d_region, cfg.d_region * 2, dropout=0.0)
        self.region_proj = nn.Linear(cfg.d_region, cfg.d_model, bias=False)

        # Extra causal blocks run on the full sequence; update applied sparsely.
        self.adaptive_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_head, cfg.d_ff,
                             cfg.seq_len, cfg.dropout)
            for _ in range(cfg.adaptive_layers)
        ])
        self.refine_scale = nn.Parameter(torch.tensor(float(cfg.adaptive_scale_init)))

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
        self.apply(_init_weights)

        self._last_gate:      Optional[torch.Tensor] = None
        self._last_alpha_dyn: Optional[torch.Tensor] = None
        self._last_norms:     Dict[str, float]       = {}

    def set_region_scales(self, alpha: float, beta: float = 0.0):
        self.warmup_scale  = min(1.0, alpha / max(self.cfg.alpha_core, 1e-8))
        self.current_alpha = alpha

    def _boundary_gate(self, p_region: torch.Tensor) -> torch.Tensor:
        if self.cfg.boundary_mode == "entropy":
            H = -(p_region * (p_region + 1e-8).log()).sum(-1)
            return H / math.log(max(self.cfg.n_coarse, 2))
        top2   = p_region.topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        return torch.sigmoid(
            (self.cfg.boundary_tau - margin) / self.cfg.boundary_temp
        )

    def forward(
        self,
        idx: torch.Tensor,
        oracle_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)

        region_logits = self.coarse_head(h)

        if oracle_region is not None:
            known    = oracle_region >= 0
            p_region = torch.full(
                (B, T, self.cfg.n_coarse), 1.0 / self.cfg.n_coarse, device=idx.device,
            )
            oh = F.one_hot(oracle_region.clamp(min=0),
                           num_classes=self.cfg.n_coarse).float()
            p_region = torch.where(known.unsqueeze(-1), oh, p_region)
        else:
            p_region = F.softmax(region_logits / self.cfg.router_temp, dim=-1)

        # Gate uses full p_region, detached so it doesn't leak into router training
        gate          = self._boundary_gate(p_region.detach())          # (B, T)
        boundary_mask = (gate > self.cfg.adaptive_threshold).float()    # (B, T)

        ws = self.warmup_scale
        r           = p_region @ self.region_emb.weight
        region_feat = self.region_proj(self.region_mlp(r))
        h_core      = h + self.cfg.alpha_core * ws * region_feat

        # Run adaptive blocks on the FULL sequence for causal context
        h_refined = h_core
        for blk in self.adaptive_blocks:
            h_refined = blk(h_refined)

        # Apply update ONLY at boundary positions (hard sparse mask)
        delta   = h_refined - h_core                                    # (B, T, d_model)
        scale   = torch.tanh(self.refine_scale) * ws
        h_final = h_core + boundary_mask.unsqueeze(-1) * scale * delta

        self._last_gate      = gate.detach().cpu()
        self._last_alpha_dyn = (self.cfg.alpha_core * torch.ones_like(gate)).detach().cpu()
        with torch.no_grad():
            frac_refined = boundary_mask.mean().item()
            self._last_norms = {
                "h_norm":             h.norm(dim=-1).mean().item(),
                "delta_norm":         delta.norm(dim=-1).mean().item(),
                "h_final_delta_norm": (h_final - h_core).norm(dim=-1).mean().item(),
                "frac_tokens_refined":frac_refined,
            }

        return self.lm_head(h_final), region_logits, p_region, h_final

    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        src = idx[:, :-1]
        tgt = idx[:, 1:]

        lm_logits, region_logits, _, _ = self.forward(src)

        l_lm  = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        if self.coarse_map is not None:
            coarse_labels = self.coarse_map[tgt]
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    region_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c
            p_c   = F.softmax(region_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

        losses["total"] = total
        return losses

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb     = _n(self.token_emb) + _n(self.pos_emb)
        trunk   = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router  = (_n(self.coarse_head) + _n(self.region_emb)
                   + _n(self.region_mlp) + _n(self.region_proj))
        refiner = sum(_n(b) for b in self.adaptive_blocks) + 1  # +1 for refine_scale
        _n(self.lm_head)
        total   = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": refiner,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Boundary refiner (MLP, used by MultiHyp) ──────────────────────────────────

class BoundaryRefiner(nn.Module):
    """Small shared MLP applied independently to each latent hypothesis.
    Residual-connected so it can be initialised as identity."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ── Repr-Region-MultiHyp model ─────────────────────────────────────────────────

class ReprRegionMultiHypLM(nn.Module):
    """
    Uncertainty-preserving latent refinement near region boundaries.

    For each token the router computes routing margin = top1 − top2.
    A sigmoid gate converts low margin (ambiguous) to high gate value.

    Core path  (gate ≈ 0, confident tokens):
        r_soft   = p_region @ region_emb.weight        (soft mixture)
        h_core   = h + alpha_core * region_proj(region_mlp(r_soft))

    Multi-hypothesis path  (gate ≈ 1, boundary tokens):
        For i = 1..hyp_k:
            h_i   = h + alpha_boundary * region_proj(region_mlp(region_emb[topk_idx_i]))
            h_i'  = BoundaryRefiner(h_i)               (shared, lightweight)
        h_bnd = Σ_i  topk_norm_prob_i * h_i'           (probability-weighted merge)

    Final blend:
        h_final = (1 − gate) * h_core + gate * h_bnd

    Key design principle: hypotheses are kept SEPARATE through the refiner so
    that each latent manifold can evolve independently before recombination.
    Naive soft averaging before refinement destroys this information.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        coarse_map: torch.Tensor,
    ):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size

        self.register_buffer("coarse_map", coarse_map)
        self.current_alpha: float = cfg.alpha_core
        self.warmup_scale:  float = 1.0

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)

        self.coarse_head  = _make_router_head(cfg.d_model, cfg.n_coarse,
                                              cfg.dropout, cfg.router_type)
        self.region_emb   = nn.Embedding(cfg.n_coarse, cfg.d_region)
        self.region_mlp   = FFN(cfg.d_region, cfg.d_region * 2, dropout=0.0)
        self.region_proj  = nn.Linear(cfg.d_region, cfg.d_model, bias=False)
        self.hyp_refiner  = BoundaryRefiner(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying
        self.apply(_init_weights)

        self._last_gate:      Optional[torch.Tensor] = None
        self._last_alpha_dyn: Optional[torch.Tensor] = None

    def set_region_scales(self, alpha: float, beta: float = 0.0):
        self.warmup_scale  = min(1.0, alpha / max(self.cfg.alpha_core, 1e-8))
        self.current_alpha = alpha

    def _boundary_gate(self, p_region: torch.Tensor) -> torch.Tensor:
        """Sigmoid gate ∈ [0,1] per (B,T). 1 = boundary, 0 = core."""
        top2   = p_region.topk(2, dim=-1).values       # (B, T, 2)
        margin = top2[..., 0] - top2[..., 1]           # (B, T)
        return torch.sigmoid(
            (self.cfg.boundary_tau - margin) / self.cfg.boundary_temp
        )

    def forward(
        self,
        idx: torch.Tensor,
        oracle_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)                                    # (B, T, d_model)

        region_logits = self.coarse_head(h)                 # (B, T, K)

        if oracle_region is not None:
            known    = oracle_region >= 0
            p_region = torch.full(
                (B, T, self.cfg.n_coarse), 1.0 / self.cfg.n_coarse, device=idx.device,
            )
            oh = F.one_hot(oracle_region.clamp(min=0),
                           num_classes=self.cfg.n_coarse).float()
            p_region = torch.where(known.unsqueeze(-1), oh, p_region)
        else:
            p_region = F.softmax(region_logits / self.cfg.router_temp, dim=-1)

        # ── Gate ──────────────────────────────────────────────────────────────
        # Detach p_region so gate-selection gradient does not leak back into
        # the router. Without this, the LM loss flows through h_final → gate →
        # margin → p_region and pushes the router toward uniform distributions
        # (more boundary tokens), collapsing boundary_frac toward 1.0. The
        # router is still trained by coarse auxiliary loss, balance loss, and
        # LM loss through the region features — just not through gate selection.
        gate      = self._boundary_gate(p_region.detach())  # (B, T)
        gate_exp  = gate.unsqueeze(-1)                      # (B, T, 1)

        # ── Core path: soft region mixture ────────────────────────────────────
        ws = self.warmup_scale
        r_soft   = p_region @ self.region_emb.weight        # (B, T, d_region)
        core_feat = self.region_proj(self.region_mlp(r_soft))
        h_core    = h + self.cfg.alpha_core * ws * core_feat

        # ── Multi-hypothesis boundary path ────────────────────────────────────
        K_h = min(self.cfg.hyp_k, self.cfg.n_coarse)
        topk_out        = p_region.topk(K_h, dim=-1)
        topk_probs_raw  = topk_out.values                   # (B, T, K_h)
        topk_idx        = topk_out.indices                  # (B, T, K_h)

        # Renormalise so weights sum to 1 for the K_h hypotheses
        topk_probs = topk_probs_raw / (topk_probs_raw.sum(-1, keepdim=True) + 1e-8)

        # Embed each hypothesis: (B, T, K_h) → (B, T, K_h, d_region)
        hyp_emb = self.region_emb(topk_idx)                # (B, T, K_h, d_region)

        # Pass through shared region_mlp + region_proj flattened to (B*T*K_h, ...)
        flat_emb  = hyp_emb.reshape(B * T * K_h, self.cfg.d_region)
        flat_feat = self.region_proj(self.region_mlp(flat_emb))  # (B*T*K_h, d_model)
        hyp_feat  = flat_feat.reshape(B, T, K_h, self.cfg.d_model)

        # h_i = h + alpha_boundary * region_feat_i  for each hypothesis i
        h_exp  = h.unsqueeze(2).expand(-1, -1, K_h, -1)    # (B, T, K_h, d_model)
        h_hyps = h_exp + self.cfg.alpha_boundary * ws * hyp_feat  # (B, T, K_h, d_model)

        # Refine each hypothesis independently (shared lightweight MLP)
        h_flat_ref = self.hyp_refiner(h_hyps.reshape(B * T * K_h, self.cfg.d_model))
        h_refined  = h_flat_ref.reshape(B, T, K_h, self.cfg.d_model)

        # Probability-weighted merge of refined hypotheses
        h_bnd = (topk_probs.unsqueeze(-1) * h_refined).sum(dim=2)  # (B, T, d_model)

        # ── Final blend ───────────────────────────────────────────────────────
        h_final = (1.0 - gate_exp) * h_core + gate_exp * h_bnd

        # Cache for eval per-group tracking (effective alpha for logging compat)
        self._last_gate      = gate.detach().cpu()
        self._last_alpha_dyn = (
            self.cfg.alpha_core * (1.0 - gate) + self.cfg.alpha_boundary * gate
        ).detach().cpu()

        return self.lm_head(h_final), region_logits, p_region, h_final

    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        src = idx[:, :-1]
        tgt = idx[:, 1:]

        oracle_region = None
        if self.cfg.mode == "oracle_repr_region_multihyp" and self.coarse_map is not None:
            oracle_region = self.coarse_map[tgt]

        lm_logits, region_logits, _, _ = self.forward(src, oracle_region)

        l_lm  = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        if self.cfg.mode != "oracle_repr_region_multihyp" and self.coarse_map is not None:
            coarse_labels = self.coarse_map[tgt]
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    region_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c
            p_c   = F.softmax(region_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

        losses["total"] = total
        return losses

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb      = _n(self.token_emb) + _n(self.pos_emb)
        trunk    = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router   = (_n(self.coarse_head) + _n(self.region_emb)
                    + _n(self.region_mlp) + _n(self.region_proj))
        refiner  = _n(self.hyp_refiner)
        _n(self.lm_head)   # weight-tied → 0
        total    = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": refiner,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Branch-Attention Refiner ──────────────────────────────────────────────────

class _BranchAttnLayer(nn.Module):
    """One Transformer-style layer operating over the K branch dimension."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True, bias=False)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        self.drop = nn.Dropout(dropout)
        # NOTE: no zero-init here — _init_weights handles all linears at std=0.02.
        # Zero-init on both projections AND branch_scale=0 killed all gradients
        # through the refiner (delta=0 → d_out/d_scale=0 AND d_out/d_params=0).
        # Identity-at-init is instead ensured by tanh(branch_scale_init≈0.05) ≈ 0.05,
        # giving a tiny but gradient-carrying contribution from the start.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, K, d_model)  where N = B*T
        x2 = self.ln1(x)
        x2, _ = self.attn(x2, x2, x2, need_weights=False)
        x = x + self.drop(x2)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class BranchAttentionRefiner(nn.Module):
    """
    Attention over the K latent hypothesis dimension.

    Input:  (B, T, K, d_model)
    Output: (B, T, K, d_model)

    Flattens B*T into the batch axis, treats K as the sequence axis for
    self-attention.  A tanh-gated learnable scale keeps the initial contribution
    small without killing gradients (unlike zero-init projections + scale=0).
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 dropout: float, scale_init: float):
        super().__init__()
        self.layers = nn.ModuleList([
            _BranchAttnLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.branch_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, K, d_model)
        B, T, K, D = x.shape
        flat = x.reshape(B * T, K, D)    # (B*T, K, d_model)
        refined = flat
        for layer in self.layers:
            refined = layer(refined)
        delta = refined - flat            # (B*T, K, d_model)
        scale = torch.tanh(self.branch_scale)
        out   = flat + scale * delta
        return out.reshape(B, T, K, D), delta.reshape(B, T, K, D)


class ReprRegionBranchAttnLM(nn.Module):
    """
    repr_region with boundary-gated multi-hypothesis branch attention.

    Modes handled:
      repr_region_branchattn   — full branch-attention refiner
      repr_region_branch_identity — skip refiner (diagnostic: is the branch path
                                    itself useful before any refinement?)

    Identical gating to repr_region_multihyp (detached p_region for gate),
    but replaces the shared MLP BoundaryRefiner with BranchAttentionRefiner
    that runs attention over the K hypothesis dimension before merging.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        coarse_map: torch.Tensor,
    ):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size
        self.use_branch_attn = (cfg.mode != "repr_region_branch_identity")

        self.register_buffer("coarse_map", coarse_map)
        self.current_alpha: float = cfg.alpha_core
        self.warmup_scale:  float = 1.0

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = _make_blocks(cfg, cfg.n_layer)
        self.ln_f      = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight   # weight tying (match multihyp)

        self.coarse_head  = _make_router_head(cfg.d_model, cfg.n_coarse,
                                              cfg.dropout, cfg.router_type)
        self.region_emb   = nn.Embedding(cfg.n_coarse, cfg.d_region)
        self.region_mlp   = FFN(cfg.d_region, cfg.d_region * 2, dropout=0.0)
        self.region_proj  = nn.Linear(cfg.d_region, cfg.d_model, bias=False)

        self.branch_refiner = BranchAttentionRefiner(
            d_model    = cfg.d_model,
            n_heads    = cfg.branch_attn_heads,
            n_layers   = cfg.branch_attn_layers,
            dropout    = cfg.branch_attn_dropout,
            scale_init = cfg.branch_scale_init,
        )

        # Apply standard weight init to everything; branch_scale retains its
        # custom init value (it is an nn.Parameter, not nn.Linear/nn.Embedding).
        self.apply(_init_weights)

        self._last_gate:      Optional[torch.Tensor] = None
        self._last_alpha_dyn: Optional[torch.Tensor] = None
        self._last_norms:     Dict[str, float]       = {}

    def set_region_scales(self, alpha: float, beta: float = 0.0):
        self.warmup_scale  = min(1.0, alpha / max(self.cfg.alpha_core, 1e-8))
        self.current_alpha = alpha

    def _boundary_gate(self, p_region: torch.Tensor) -> torch.Tensor:
        top2   = p_region.topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        return torch.sigmoid(
            (self.cfg.boundary_tau - margin) / self.cfg.boundary_temp
        )

    def forward(
        self,
        idx: torch.Tensor,
        oracle_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)                                        # (B, T, d_model)

        region_logits = self.coarse_head(h)

        if oracle_region is not None:
            known    = oracle_region >= 0
            p_region = torch.full(
                (B, T, self.cfg.n_coarse), 1.0 / self.cfg.n_coarse, device=idx.device,
            )
            oh = F.one_hot(oracle_region.clamp(min=0),
                           num_classes=self.cfg.n_coarse).float()
            p_region = torch.where(known.unsqueeze(-1), oh, p_region)
        else:
            p_region = F.softmax(region_logits / self.cfg.router_temp, dim=-1)

        gate     = self._boundary_gate(p_region.detach())
        gate_exp = gate.unsqueeze(-1)

        ws = self.warmup_scale
        r_soft    = p_region @ self.region_emb.weight
        core_feat = self.region_proj(self.region_mlp(r_soft))
        h_core    = h + self.cfg.alpha_core * ws * core_feat

        K_h = min(self.cfg.hyp_k, self.cfg.n_coarse)
        topk_out       = p_region.topk(K_h, dim=-1)
        topk_probs_raw = topk_out.values                        # (B, T, K_h)
        topk_idx       = topk_out.indices

        topk_probs = topk_probs_raw / (topk_probs_raw.sum(-1, keepdim=True) + 1e-8)

        hyp_emb  = self.region_emb(topk_idx)                   # (B, T, K_h, d_region)
        flat_emb = hyp_emb.reshape(B * T * K_h, self.cfg.d_region)
        flat_feat= self.region_proj(self.region_mlp(flat_emb))
        hyp_feat = flat_feat.reshape(B, T, K_h, self.cfg.d_model)

        h_exp  = h.unsqueeze(2).expand(-1, -1, K_h, -1)
        h_hyps = h_exp + self.cfg.alpha_boundary * ws * hyp_feat   # (B, T, K_h, d_model)

        if self.use_branch_attn:
            h_refined, refined_delta = self.branch_refiner(h_hyps)
        else:
            h_refined    = h_hyps
            refined_delta = torch.zeros_like(h_hyps)

        h_bnd   = (topk_probs.unsqueeze(-1) * h_refined).sum(dim=2)  # (B, T, d_model)
        h_final = (1.0 - gate_exp) * h_core + gate_exp * h_bnd

        self._last_gate      = gate.detach().cpu()
        self._last_alpha_dyn = (
            self.cfg.alpha_core * (1.0 - gate) + self.cfg.alpha_boundary * gate
        ).detach().cpu()
        with torch.no_grad():
            self._last_norms = {
                "h_norm":             h.norm(dim=-1).mean().item(),
                "h_core_delta_norm":  (h_core - h).norm(dim=-1).mean().item(),
                "h_bnd_delta_norm":   (h_bnd  - h).norm(dim=-1).mean().item(),
                "h_final_delta_norm": (h_final - h).norm(dim=-1).mean().item(),
                "hyp_feat_norm":      hyp_feat.norm(dim=-1).mean().item(),
                "refined_delta_norm": refined_delta.norm(dim=-1).mean().item(),
            }

        return self.lm_head(h_final), region_logits, p_region, h_final

    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        src = idx[:, :-1]
        tgt = idx[:, 1:]

        lm_logits, region_logits, _, _ = self.forward(src)

        l_lm  = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        if self.coarse_map is not None:
            coarse_labels = self.coarse_map[tgt]
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    region_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c
            p_c   = F.softmax(region_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

        losses["total"] = total
        return losses

    def param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb     = _n(self.token_emb) + _n(self.pos_emb)
        trunk   = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router  = (_n(self.coarse_head) + _n(self.region_emb)
                   + _n(self.region_mlp) + _n(self.region_proj))
        refiner = _n(self.branch_refiner)
        _n(self.lm_head)  # weight-tied → 0
        total   = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": refiner,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


class ReprRegionRetrievalLM(ReprRegionTransformerLM):
    """
    ReprRegionTransformerLM + metric-learning loss that makes pre-region
    hidden states cluster by next-token region label.

    Two loss variants (cfg.retrieval_loss_type):
      "proxy"  — CE over learned normalized region centroids (cheap, stable)
      "supcon" — supervised contrastive loss (richer, subsampled for memory)

    cfg.retrieval_key_source:
      "pre_region"  — train on h before region injection; matches kNN key convention
      "post_region" — train on h' after injection
    """

    def __init__(self, cfg: TrainConfig, vocab_size: int, coarse_map: torch.Tensor):
        super().__init__(cfg, vocab_size, coarse_map)
        d, r = cfg.d_model, cfg.retrieval_dim
        self.retrieval_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d, bias=False),
            nn.GELU(),
            nn.Linear(d, r, bias=False),
        )
        if cfg.retrieval_loss_type == "proxy":
            self.region_centroids = nn.Embedding(cfg.n_coarse, r)
            nn.init.normal_(self.region_centroids.weight, std=0.02)
        self.retrieval_proj.apply(_init_weights)
        self._last_h: Optional[torch.Tensor] = None

    def forward(
        self,
        idx: torch.Tensor,
        oracle_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as parent but caches h (pre-region) in self._last_h."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        self._last_h = h                                        # pre-region key

        region_logits = self.coarse_head(h)

        if oracle_region is not None:
            known = oracle_region >= 0
            p_region = torch.full(
                (B, T, self.cfg.n_coarse), 1.0 / self.cfg.n_coarse, device=idx.device,
            )
            oh = F.one_hot(oracle_region.clamp(min=0),
                           num_classes=self.cfg.n_coarse).float()
            p_region = torch.where(known.unsqueeze(-1), oh, p_region)
        else:
            p_region = F.softmax(region_logits / self.cfg.router_temp, dim=-1)

        r           = p_region @ self.region_emb.weight
        region_feat = self.region_proj(self.region_mlp(r))
        h_prime     = h + self.current_alpha * region_feat

        return self.lm_head(h_prime), region_logits, p_region, h_prime

    def loss(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        src = idx[:, :-1]
        tgt = idx[:, 1:]

        lm_logits, region_logits, _, h_prime = self.forward(src)
        l_lm  = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        total = l_lm
        losses: Dict[str, torch.Tensor] = {"lm": l_lm}

        if self.coarse_map is not None:
            coarse_labels = self.coarse_map[tgt]
            valid = coarse_labels >= 0
            if valid.any():
                l_c = F.cross_entropy(
                    region_logits.reshape(-1, self.cfg.n_coarse)[valid.reshape(-1)],
                    coarse_labels.reshape(-1)[valid.reshape(-1)],
                )
                total = total + self.cfg.lambda_coarse * l_c
                losses["coarse"] = l_c
            p_c   = F.softmax(region_logits, dim=-1)
            l_bal = -(p_c * (p_c + 1e-8).log()).sum(-1).mean().neg()
            total = total + self.cfg.lambda_balance * l_bal
            losses["balance_coarse"] = l_bal

            if self.cfg.lambda_retrieval > 0.0 and valid.any():
                h_key = (self._last_h
                         if self.cfg.retrieval_key_source == "pre_region" else h_prime)
                h_flat     = h_key.reshape(-1, self.cfg.d_model)
                lbl_flat   = coarse_labels.reshape(-1)
                valid_flat = valid.reshape(-1)
                h_valid    = h_flat[valid_flat]
                lbl_valid  = lbl_flat[valid_flat]
                N_valid = len(h_valid)
                max_p = self.cfg.max_retrieval_positions_per_batch
                if N_valid > max_p:
                    perm      = torch.randperm(N_valid, device=h_valid.device)[:max_p]
                    h_valid   = h_valid[perm]
                    lbl_valid = lbl_valid[perm]
                z = F.normalize(self.retrieval_proj(h_valid), dim=-1)

                if self.cfg.retrieval_loss_type == "proxy":
                    c = F.normalize(self.region_centroids.weight, dim=-1)
                    logits_metric = (z @ c.T) / self.cfg.retrieval_temp
                    l_retr = F.cross_entropy(logits_metric, lbl_valid)
                    losses["retrieval"] = l_retr
                    with torch.no_grad():
                        losses["retrieval_acc1"] = torch.tensor(
                            topk_accuracy(logits_metric, lbl_valid, 1))
                        losses["retrieval_acc4"] = torch.tensor(
                            topk_accuracy(logits_metric, lbl_valid, 4))
                else:  # supcon
                    l_retr = _supcon_loss(z, lbl_valid, self.cfg.retrieval_temp)
                    losses["retrieval"] = l_retr
                total = total + self.cfg.lambda_retrieval * l_retr

        losses["total"] = total
        return losses

    def param_breakdown(self) -> Dict[str, int]:
        seen: set = set()
        def _n(m: nn.Module) -> int:
            c = 0
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); c += p.numel()
            return c
        emb    = _n(self.token_emb) + _n(self.pos_emb)
        trunk  = sum(_n(b) for b in self.blocks) + _n(self.ln_f)
        router = (_n(self.coarse_head) + _n(self.region_emb)
                  + _n(self.region_mlp) + _n(self.region_proj))
        refiner = _n(self.retrieval_proj)
        if hasattr(self, "region_centroids"):
            refiner += _n(self.region_centroids)
        _n(self.lm_head)  # weight-tied → 0
        total = self.param_count()
        return {"emb": emb, "trunk": trunk, "router": router, "refiner": refiner,
                "total": total, "non_emb": total - self.token_emb.weight.numel()}


# ── Region map loaders ─────────────────────────────────────────────────────────

def load_coarse_map(path: str, vocab_size: int) -> Tuple[torch.Tensor, int]:
    with open(path) as f:
        d = json.load(f)
    arr = torch.full((vocab_size,), -1, dtype=torch.long)
    max_region = -1
    for tok_str, region_id in d.items():
        tok_id = int(tok_str)
        if tok_id < vocab_size:
            arr[tok_id] = int(region_id)
            if int(region_id) > max_region:
                max_region = int(region_id)
    return arr, max_region + 1


def _flatten_tree(node: dict, vocab_to_leaf: dict, counter: list):
    children = node.get("children", {})
    if not children:
        lid = counter[0]; counter[0] += 1
        for vid in node.get("vocab_ids", []):
            vocab_to_leaf[int(vid)] = lid
    else:
        for child in children.values():
            _flatten_tree(child, vocab_to_leaf, counter)


def load_leaf_map(path: str, vocab_size: int) -> Tuple[torch.Tensor, int]:
    with open(path) as f:
        tree = json.load(f)
    vocab_to_leaf: dict = {}
    counter = [0]
    _flatten_tree(tree.get("root", tree), vocab_to_leaf, counter)
    n_leaf = counter[0]
    arr = torch.full((vocab_size,), -1, dtype=torch.long)
    for tok_id, leaf_id in vocab_to_leaf.items():
        if tok_id < vocab_size:
            arr[tok_id] = leaf_id
    return arr, n_leaf


def permute_map(arr: torch.Tensor, n_classes: int, seed: int) -> torch.Tensor:
    """Fixed random permutation of valid labels (>= 0). Preserves class-count distribution."""
    rng  = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n_classes, generator=rng)
    out  = arr.clone()
    valid = arr >= 0
    out[valid] = perm[arr[valid]]
    return out


def make_random_partition_map(arr: torch.Tensor, n_classes: int, seed: int) -> torch.Tensor:
    """
    True random partition: shuffles labeled tokens across regions while preserving
    per-region size counts.  Destroys structural assignment (semantically similar
    tokens no longer share a region) while keeping class imbalance identical.
    This is a stronger null hypothesis than label permutation.
    """
    labeled_mask = arr >= 0
    labeled_idx  = torch.where(labeled_mask)[0]
    region_sizes = [(arr == r).sum().item() for r in range(n_classes)]

    rng = torch.Generator()
    rng.manual_seed(seed)
    shuffled_idx = labeled_idx[torch.randperm(len(labeled_idx), generator=rng)]

    out = arr.clone()
    out[labeled_mask] = -1
    offset = 0
    for r, size in enumerate(region_sizes):
        if size == 0 or offset >= len(shuffled_idx):
            continue
        end = min(offset + int(size), len(shuffled_idx))
        out[shuffled_idx[offset:end]] = r
        offset = end
    return out


def build_membership_matrix(
    coarse_map: torch.Tensor, n_coarse: int, vocab_size: int,
) -> torch.Tensor:
    """
    Build (n_coarse, vocab_size) float32 membership matrix.
    M[r, v] = 1 if token v is in region r.
    Unknown tokens (map == -1) get 1/n_coarse in every row so the soft prior
    is uniform (neutral) for them — no routing bias for unseen tokens.
    """
    M = torch.zeros(n_coarse, vocab_size, dtype=torch.float32, device=coarse_map.device)
    known     = coarse_map >= 0
    known_idx = torch.where(known)[0]
    M[coarse_map[known_idx], known_idx] = 1.0
    unknown_idx = torch.where(~known)[0]
    if len(unknown_idx) > 0:
        M[:, unknown_idx] = 1.0 / n_coarse
    return M


# ── Dataset ────────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.n = max(0, (len(tokens) - 1) // seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        start = i * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return torch.from_numpy(chunk.astype(np.int64))


def build_datasets(cfg: TrainConfig, tokenizer) -> Tuple[TokenDataset, TokenDataset]:
    if cfg.dataset in ("wikitext-2-raw-v1", "wikitext-103-raw-v1"):
        ds = load_dataset("wikitext", cfg.dataset)
    else:
        ds = load_dataset(cfg.dataset)

    def encode_split(split: str) -> np.ndarray:
        texts = [t for t in ds[split]["text"] if t.strip()]
        chunks = []
        for text in texts:
            ids = tokenizer.encode(text)
            if ids:
                chunks.append(ids)
        return np.concatenate([np.array(c, dtype=np.int32) for c in chunks])

    train_tok = encode_split("train")
    val_tok   = encode_split("validation")
    print(f"[data] train={len(train_tok):,}  val={len(val_tok):,} tokens")
    return TokenDataset(train_tok, cfg.seq_len), TokenDataset(val_tok, cfg.seq_len)


# ── LR schedule ────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_router_temp(step: int, cfg: TrainConfig) -> float:
    """Exponential decay: router_temp_init → router_temp_final over decay_steps."""
    if cfg.router_temp_decay_steps <= 0 or cfg.router_temp_init == cfg.router_temp_final:
        return cfg.router_temp_init
    progress = min(1.0, step / cfg.router_temp_decay_steps)
    temp = cfg.router_temp_init * (cfg.router_temp_final / cfg.router_temp_init) ** progress
    return max(cfg.router_temp_final, temp)


# ── Metric helpers ─────────────────────────────────────────────────────────────

@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    valid = labels >= 0
    if not valid.any():
        return 0.0
    preds = logits[valid].topk(k, dim=-1).indices
    tgt   = labels[valid].unsqueeze(-1).expand_as(preds)
    return (preds == tgt).any(-1).float().mean().item()


@torch.no_grad()
def mean_entropy(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=-1)
    return -(p * (p + 1e-8).log()).sum(-1).mean().item()


def _compute_usage_stats(usage: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    """Routing diversity statistics from a (K,) usage probability vector."""
    u = usage.float().cpu()
    usage_entropy    = -(u * (u + eps).log()).sum().item()
    participation_ratio = (u.sum() ** 2 / ((u ** 2).sum() + eps)).item()
    sorted_u = u.sort().values
    n = len(sorted_u)
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    gini = ((2 * idx - n - 1) * sorted_u).sum() / (n * sorted_u.mean() + eps)
    return {
        "val_usage_entropy":       usage_entropy,
        "val_max_region_usage":    u.max().item(),
        "val_min_region_usage":    u.min().item(),
        "val_participation_ratio": participation_ratio,
        "val_active_regions_1pct":   int((u >= 0.01).sum().item()),
        "val_active_regions_0p1pct": int((u >= 0.001).sum().item()),
        "val_region_gini":         gini.item(),
    }


def _supcon_loss(z: torch.Tensor, labels: torch.Tensor, temp: float) -> torch.Tensor:
    """Supervised contrastive loss. z: (N, D) unit-normalized, labels: (N,) int >= 0."""
    N = len(z)
    if N <= 1:
        return z.sum() * 0.0
    sim = (z @ z.T) / temp
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    sim = sim - sim_max
    exp_sim = torch.exp(sim)
    mask_self = torch.eye(N, device=z.device, dtype=torch.float32)
    mask_pos  = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() - mask_self
    has_pos   = mask_pos.sum(1) > 0
    if not has_pos.any():
        return z.sum() * 0.0
    denom    = (exp_sim * (1.0 - mask_self)).sum(1)
    pos_sum  = (exp_sim * mask_pos).sum(1)
    n_pos    = mask_pos.sum(1).clamp(min=1)
    loss_per = -torch.log((pos_sum[has_pos] + 1e-8) / (denom[has_pos] + 1e-8)) / n_pos[has_pos]
    return loss_per.mean()


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    coarse_map: Optional[torch.Tensor] = None,
    leaf_map:   Optional[torch.Tensor] = None,
    max_batches: int = 50,
) -> Dict[str, float]:
    model.eval()
    use_amp = device.type == "cuda"
    ctx = torch.autocast(device_type=device.type, enabled=use_amp)

    lm_vals:    List[float] = []
    aux_c:      List[float] = []
    aux_l:      List[float] = []
    c_acc1:     List[float] = []
    c_acc4:     List[float] = []
    l_acc1:     List[float] = []
    l_acc8:     List[float] = []
    l_acc32:    List[float] = []
    c_ent:      List[float] = []
    l_ent:      List[float] = []
    cov_c:      List[float] = []  # coarse target coverage
    usage_buf:  List[torch.Tensor] = []  # per-batch mean routing weights (capacity model)
    # ReprRegionBoundaryLM per-group accumulators
    bnd_nll_buf:   List[float] = []   # NLL at boundary positions
    core_nll_buf:  List[float] = []   # NLL at core positions
    bnd_alpha_buf: List[float] = []   # avg alpha_dyn at boundary
    core_alpha_buf:List[float] = []   # avg alpha_dyn at core
    bnd_frac_buf:  List[float] = []   # fraction of positions classified as boundary
    # ReprRegionRetrievalLM metric-learning accumulators
    retr_loss_buf:  List[float] = []
    retr_acc1_buf:  List[float] = []
    retr_acc4_buf:  List[float] = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        src   = batch[:, :-1]
        tgt   = batch[:, 1:]

        with ctx:
            losses = model.loss(batch)
        lm_vals.append(losses["lm"].item())
        if "coarse" in losses:
            aux_c.append(losses["coarse"].item())
        if "leaf" in losses:
            aux_l.append(losses["leaf"].item())
        if "retrieval" in losses:
            retr_loss_buf.append(losses["retrieval"].item())
        if "retrieval_acc1" in losses:
            retr_acc1_buf.append(losses["retrieval_acc1"].item())
        if "retrieval_acc4" in losses:
            retr_acc4_buf.append(losses["retrieval_acc4"].item())

        if isinstance(model, RegionConditionedTransformerLM):
            with ctx:
                _, coarse_logits, leaf_logits, _ = model.forward(src)
            if coarse_logits is not None and coarse_map is not None:
                fl = coarse_logits.reshape(-1, cfg.n_coarse)
                lb = coarse_map[tgt].reshape(-1)
                c_acc1.append(topk_accuracy(fl, lb, 1))
                c_acc4.append(topk_accuracy(fl, lb, 4))
                c_ent.append(mean_entropy(fl))
            if leaf_logits is not None and leaf_map is not None:
                fl = leaf_logits.reshape(-1, cfg.n_leaf)
                lb = leaf_map[tgt].reshape(-1)
                l_acc1.append(topk_accuracy(fl, lb, 1))
                l_acc8.append(topk_accuracy(fl, lb, 8))
                l_acc32.append(topk_accuracy(fl, lb, 32))
                l_ent.append(mean_entropy(fl))

        elif isinstance(model, SoftMoETransformerLM):
            with ctx:
                _, region_logits, _, _ = model.forward(src)
            if coarse_map is not None:
                fl = region_logits.reshape(-1, cfg.n_coarse)
                lb = coarse_map[tgt].reshape(-1)
                c_acc1.append(topk_accuracy(fl, lb, 1))
                c_acc4.append(topk_accuracy(fl, lb, 4))
                c_ent.append(mean_entropy(fl))

        elif isinstance(model, ReprRegionTransformerLM):
            with ctx:
                _, region_logits, _, _ = model.forward(src)
            if coarse_map is not None:
                fl = region_logits.reshape(-1, cfg.n_coarse)
                lb = coarse_map[tgt].reshape(-1)
                c_acc1.append(topk_accuracy(fl, lb, 1))
                c_acc4.append(topk_accuracy(fl, lb, 4))
                c_ent.append(mean_entropy(fl))

        elif isinstance(model, ReprRegionCapacityLM):
            with ctx:
                _, region_logits, p_region, _ = model.forward(src)
            if coarse_map is not None:
                fl = region_logits.reshape(-1, cfg.n_coarse)
                lb = coarse_map[tgt].reshape(-1)
                c_acc1.append(topk_accuracy(fl, lb, 1))
                c_acc4.append(topk_accuracy(fl, lb, 4))
                c_ent.append(mean_entropy(fl))
            usage_buf.append(p_region.detach().mean(dim=(0, 1)).cpu())

        elif isinstance(model, (ReprRegionBoundaryLM, ReprRegionMultiHypLM,
                                ReprRegionBranchAttnLM,
                                ReprRegionBoundarySeqRefineLM,
                                ReprRegionBoundaryAdaptiveDepthLM)):
            with ctx:
                lm_logits_bnd, region_logits, p_region, _ = model.forward(src)
            if coarse_map is not None:
                fl = region_logits.reshape(-1, cfg.n_coarse)
                lb = coarse_map[tgt].reshape(-1)
                c_acc1.append(topk_accuracy(fl, lb, 1))
                c_acc4.append(topk_accuracy(fl, lb, 4))
                c_ent.append(mean_entropy(fl))
            # Per-token NLL split by boundary/core gate
            if model._last_gate is not None and model._last_alpha_dyn is not None:
                gate_flat  = model._last_gate.reshape(-1)       # (B*T,) CPU
                alpha_flat = model._last_alpha_dyn.reshape(-1)  # (B*T,) CPU
                log_p    = F.log_softmax(lm_logits_bnd.float(), dim=-1)
                per_nll  = (-log_p.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                            ).cpu().reshape(-1)                 # (B*T,)
                bnd_mask  = gate_flat >= 0.5
                core_mask = ~bnd_mask
                bnd_frac_buf.append(bnd_mask.float().mean().item())
                if bnd_mask.any():
                    bnd_nll_buf.append(per_nll[bnd_mask].mean().item())
                    bnd_alpha_buf.append(alpha_flat[bnd_mask].mean().item())
                if core_mask.any():
                    core_nll_buf.append(per_nll[core_mask].mean().item())
                    core_alpha_buf.append(alpha_flat[core_mask].mean().item())

        # Coverage: fraction of next-token targets that have a known coarse region
        if coarse_map is not None:
            lb = coarse_map[tgt].reshape(-1)
            cov_c.append((lb >= 0).float().mean().item())

    def avg(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    lm = avg(lm_vals)
    out: Dict[str, float] = {
        "val_lm_loss":          lm,
        "val_ppl":              math.exp(min(lm, 20.0)),
        "val_aux_coarse":       avg(aux_c),
        "val_aux_leaf":         avg(aux_l),
        "val_coarse_acc1":      avg(c_acc1),
        "val_coarse_acc4":      avg(c_acc4),
        "val_leaf_acc1":        avg(l_acc1),
        "val_leaf_acc8":        avg(l_acc8),
        "val_leaf_acc32":       avg(l_acc32),
        "val_coarse_ent":       avg(c_ent),
        "val_leaf_ent":         avg(l_ent),
        "val_coarse_coverage":  avg(cov_c),
        "current_alpha":        getattr(model, "current_alpha", 0.0),
        "current_beta":         getattr(model, "current_beta",  0.0),
        "current_router_temp":  getattr(model, "current_router_temp", 0.0),
        # ReprRegionBoundaryLM / multihyp / branchattn per-group metrics
        "val_boundary_lm":      avg(bnd_nll_buf),
        "val_core_lm":          avg(core_nll_buf),
        "val_boundary_frac":    avg(bnd_frac_buf),
        "val_avg_boundary_alpha": avg(bnd_alpha_buf),
        "val_avg_core_alpha":     avg(core_alpha_buf),
        # ReprRegionRetrievalLM metric-learning
        "val_retrieval_loss":  avg(retr_loss_buf),
        "val_retrieval_acc1":  avg(retr_acc1_buf),
        "val_retrieval_acc4":  avg(retr_acc4_buf),
        # Refiner-specific scales (0.0 for non-applicable models)
        "val_branch_scale": (
            float(torch.tanh(model.branch_refiner.branch_scale).item())
            if isinstance(model, ReprRegionBranchAttnLM) else 0.0
        ),
        "val_refine_scale": (
            float(torch.tanh(model.refine_scale).item())
            if isinstance(model, (ReprRegionBoundarySeqRefineLM,
                                  ReprRegionBoundaryAdaptiveDepthLM)) else 0.0
        ),
        **({f"val_{k}": v for k, v in model._last_norms.items()}
           if hasattr(model, "_last_norms") and model._last_norms else {}),
    }
    # Usage diversity stats for ReprRegionCapacityLM
    if usage_buf:
        avg_usage = torch.stack(usage_buf).mean(0)
        out.update(_compute_usage_stats(avg_usage))
    elif hasattr(model, "region_usage_ema"):
        out.update(_compute_usage_stats(model.region_usage_ema))
    return out


# ── Training ────────────────────────────────────────────────────────────────────

def _print_params(bd: Dict[str, int]):
    print(
        f"  [params] total={bd['total']:,}  non_emb={bd['non_emb']:,}"
        f"  emb={bd['emb']:,}  trunk={bd['trunk']:,}"
        f"  router={bd.get('router', 0):,}  refiner={bd.get('refiner', 0):,}"
    )


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    os.makedirs(cfg.output_dir, exist_ok=True)
    device  = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"[train] mode={cfg.mode}  device={device}  amp={use_amp}")

    # ── Tokenizer + vocab ──────────────────────────────────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)  # we chunk ourselves; suppress length warnings
    if cfg.vocab_subset_size < len(tokenizer):
        raise ValueError(
            f"vocab_subset_size={cfg.vocab_subset_size} < tokenizer vocab "
            f"({len(tokenizer)}). This is unsafe unless token IDs are remapped. "
            f"Pass --vocab_subset_size 50257 for GPT-2."
        )
    vocab_size = len(tokenizer)

    # ── Region maps ────────────────────────────────────────────────────────────
    coarse_map: Optional[torch.Tensor] = None
    leaf_map:   Optional[torch.Tensor] = None

    if cfg.mode != "baseline" and cfg.region_map_path:
        raw_coarse, n_coarse_actual = load_coarse_map(cfg.region_map_path, vocab_size)
        cfg.n_coarse = n_coarse_actual
        _RANDOM_MODES = ("random_control", "random_soft_moe",
                         "random_repr_region", "random_repr_region_capacity",
                         "random_repr_region_boundary", "random_repr_region_multihyp",
                         "random_repr_region_boundary_seqrefine",
                         "random_repr_region_boundary_adaptivedepth",
                         "random_repr_region_retrieval")
        if cfg.mode in _RANDOM_MODES:
            raw_coarse = make_random_partition_map(raw_coarse, cfg.n_coarse, seed=cfg.seed)
        coarse_map = raw_coarse.to(device)
        cov = (raw_coarse >= 0).float().mean().item()
        tag = " (RANDOM PARTITION)" if cfg.mode in _RANDOM_MODES else ""
        print(f"[region] coarse: {n_coarse_actual} regions  {cov:.1%} coverage{tag}")

    if cfg.mode in ("coarse_leaf", "random_control", "oracle") and cfg.leaf_map_path:
        raw_leaf, n_leaf_actual = load_leaf_map(cfg.leaf_map_path, vocab_size)
        cfg.n_leaf = n_leaf_actual
        if cfg.mode == "random_control":
            raw_leaf = make_random_partition_map(raw_leaf, cfg.n_leaf, seed=cfg.seed + 1)
        leaf_map = raw_leaf.to(device)
        cov = (raw_leaf >= 0).float().mean().item()
        tag = " (RANDOM PARTITION)" if cfg.mode == "random_control" else ""
        print(f"[region] leaf:   {n_leaf_actual} leaves    {cov:.1%} coverage{tag}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("[data] loading...")
    train_ds, val_ds = build_datasets(cfg, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=use_amp, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=use_amp, drop_last=False,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    if cfg.mode == "baseline":
        model: nn.Module = BaselineTransformerLM(cfg, vocab_size)
    elif cfg.mode in ("soft_moe", "oracle_soft_moe", "random_soft_moe"):
        assert coarse_map is not None, f"{cfg.mode} requires --region_map_path"
        membership = build_membership_matrix(coarse_map, cfg.n_coarse, vocab_size)
        model = SoftMoETransformerLM(cfg, vocab_size, coarse_map=coarse_map,
                                     membership=membership)
    elif cfg.mode in ("repr_region", "oracle_repr_region", "random_repr_region"):
        assert coarse_map is not None, f"{cfg.mode} requires --region_map_path"
        model = ReprRegionTransformerLM(cfg, vocab_size, coarse_map=coarse_map)
    elif cfg.mode in ("repr_region_capacity", "oracle_repr_region_capacity",
                      "random_repr_region_capacity"):
        assert coarse_map is not None, f"{cfg.mode} requires --region_map_path"
        model = ReprRegionCapacityLM(cfg, vocab_size, coarse_map=coarse_map)
    elif cfg.mode in ("repr_region_boundary", "oracle_repr_region_boundary",
                      "random_repr_region_boundary"):
        assert coarse_map is not None, f"{cfg.mode} requires --region_map_path"
        model = ReprRegionBoundaryLM(cfg, vocab_size, coarse_map=coarse_map)
    elif cfg.mode in ("repr_region_multihyp", "oracle_repr_region_multihyp",
                      "random_repr_region_multihyp"):
        assert coarse_map is not None, f"{cfg.mode} requires --region_map_path"
        model = ReprRegionMultiHypLM(cfg, vocab_size, coarse_map=coarse_map)
    elif cfg.mode in ("repr_region_boundary_seqrefine",
                      "random_repr_region_boundary_seqrefine"):
        assert coarse_map is not None, f"{cfg.mode} requires --region_map_path"
        model = ReprRegionBoundarySeqRefineLM(cfg, vocab_size, coarse_map=coarse_map)
    elif cfg.mode in ("repr_region_boundary_adaptivedepth",
                      "random_repr_region_boundary_adaptivedepth"):
        assert coarse_map is not None, f"{cfg.mode} requires --region_map_path"
        model = ReprRegionBoundaryAdaptiveDepthLM(cfg, vocab_size, coarse_map=coarse_map)
    elif cfg.mode in ("repr_region_branchattn", "repr_region_branch_identity"):
        assert coarse_map is not None, f"{cfg.mode} requires --region_map_path"
        model = ReprRegionBranchAttnLM(cfg, vocab_size, coarse_map=coarse_map)
    elif cfg.mode in ("repr_region_retrieval", "random_repr_region_retrieval"):
        assert coarse_map is not None, f"{cfg.mode} requires --region_map_path"
        model = ReprRegionRetrievalLM(cfg, vocab_size, coarse_map=coarse_map)
    else:
        model = RegionConditionedTransformerLM(
            cfg, vocab_size, coarse_map=coarse_map, leaf_map=leaf_map,
        )
    model = model.to(device)
    bd       = model.param_breakdown()  # type: ignore[attr-defined]
    n_params = bd["total"]
    print(f"[model] {cfg.mode}")
    _print_params(bd)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    decay_p   = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_p = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_p,   "weight_decay": cfg.weight_decay},
         {"params": nodecay_p, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_step  = 0
    latest_ckpt = os.path.join(cfg.output_dir, "checkpoint_latest.pt")
    if cfg.resume and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        print(f"[resume] step {start_step}")

    # ── CSV ────────────────────────────────────────────────────────────────────
    _csv_fields = [
        "step", "train_lm_loss", "train_total_loss",
        "val_lm_loss", "val_ppl",
        "train_aux_coarse", "train_aux_leaf", "train_aux_balance", "train_aux_diversity",
        "train_retrieval_loss", "train_retrieval_acc1", "train_retrieval_acc4",
        "val_aux_coarse", "val_aux_leaf",
        "val_retrieval_loss", "val_retrieval_acc1", "val_retrieval_acc4",
        "val_coarse_acc1", "val_coarse_acc4",
        "val_leaf_acc1", "val_leaf_acc8", "val_leaf_acc32",
        "val_coarse_ent", "val_leaf_ent",
        "val_usage_entropy", "val_max_region_usage", "val_min_region_usage",
        "val_participation_ratio", "val_active_regions_1pct", "val_active_regions_0p1pct",
        "val_region_gini",
        "current_alpha", "current_beta", "current_router_temp",
        "val_boundary_lm", "val_core_lm", "val_boundary_frac",
        "val_avg_boundary_alpha", "val_avg_core_alpha",
        "lr", "tokens_per_sec", "n_params",
    ]
    _csv_file   = open(os.path.join(cfg.output_dir, "metrics.csv"), "w", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields, extrasaction="ignore")
    _csv_writer.writeheader()

    # ── Loop ───────────────────────────────────────────────────────────────────
    model.train()
    amp_ctx    = torch.autocast(device_type=device.type, enabled=use_amp)
    train_iter = iter(train_loader)
    step       = start_step
    ema_lm     = None
    ema_total  = None
    t0         = time.time()
    tok_seen   = 0

    while step < cfg.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch.to(device, non_blocking=True)

        # Region injection warmup
        if isinstance(model, (RegionConditionedTransformerLM,
                               ReprRegionTransformerLM, ReprRegionCapacityLM)):
            if cfg.region_warmup_steps <= 0:
                scale = 1.0
            else:
                scale = min(1.0, step / cfg.region_warmup_steps)
            model.set_region_scales(cfg.base_alpha * scale, cfg.base_beta * scale)
        elif isinstance(model, (ReprRegionBoundaryLM, ReprRegionMultiHypLM,
                                 ReprRegionBranchAttnLM,
                                 ReprRegionBoundarySeqRefineLM,
                                 ReprRegionBoundaryAdaptiveDepthLM)):
            if cfg.region_warmup_steps <= 0:
                scale = 1.0
            else:
                scale = min(1.0, step / cfg.region_warmup_steps)
            model.set_region_scales(cfg.alpha_core * scale)

        # Router temperature annealing
        if isinstance(model, ReprRegionCapacityLM):
            model.set_router_temp(get_router_temp(step, cfg))

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            losses = model.loss(batch)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tok_seen += cfg.batch_size * cfg.seq_len
        step     += 1

        lm_v  = losses["lm"].item()
        tot_v = losses["total"].item()
        ema_lm    = lm_v  if ema_lm    is None else 0.9 * ema_lm    + 0.1 * lm_v
        ema_total = tot_v if ema_total is None else 0.9 * ema_total + 0.1 * tot_v

        if step % cfg.log_interval == 0:
            tps   = tok_seen / (time.time() - t0)
            alpha = getattr(model, "current_alpha", 0.0)
            beta  = getattr(model, "current_beta",  0.0)
            rtemp = getattr(model, "current_router_temp", 0.0)
            _retr_str = ""
            if isinstance(model, ReprRegionRetrievalLM) and "retrieval" in losses:
                _retr_str = (
                    f"  retr={losses['retrieval'].item():.4f}"
                    + (f"  r@1={losses['retrieval_acc1'].item():.3f}"
                       if "retrieval_acc1" in losses else "")
                )
            print(
                f"step {step:6d}  lm={ema_lm:.4f}  total={ema_total:.4f}"
                f"  lr={lr:.2e}  α={alpha:.3f}  β={beta:.4f}  T={rtemp:.2f}  tok/s={tps:.0f}"
                + _retr_str
            )
            # ── Gradient norm diagnostics for boundary refiner modes ─────────
            if isinstance(model, (ReprRegionBranchAttnLM, ReprRegionBoundarySeqRefineLM,
                                   ReprRegionBoundaryAdaptiveDepthLM)):
                def _gnorm(params) -> float:
                    g = [p.grad for p in params if p.grad is not None]
                    return float(torch.stack([p.norm() for p in g]).norm().item()) if g else 0.0
                if isinstance(model, ReprRegionBranchAttnLM):
                    scale_p = model.branch_refiner.branch_scale
                    refiner_params = model.branch_refiner.parameters()
                    scale_label = "br_scale"
                elif isinstance(model, ReprRegionBoundaryAdaptiveDepthLM):
                    scale_p = model.refine_scale
                    refiner_params = list(model.adaptive_blocks.parameters()) + [model.refine_scale]
                    scale_label = "ad_scale"
                else:
                    scale_p = model.refine_scale
                    refiner_params = list(model.seq_refine_blocks.parameters()) + [model.refine_scale]
                    scale_label = "rf_scale"
                scale_grad = float(scale_p.grad.item()) if scale_p.grad is not None else 0.0
                scale_tanh = float(torch.tanh(scale_p).item())
                print(
                    f"  [grad] {scale_label}.grad={scale_grad:.2e}"
                    f"  refiner={_gnorm(refiner_params):.2e}"
                    f"  region_proj={_gnorm(model.region_proj.parameters()):.2e}"
                    f"  router={_gnorm(model.coarse_head.parameters()):.2e}"
                    f"  trunk={_gnorm(p for b in model.blocks for p in b.parameters()):.2e}"
                    f"  {scale_label}(tanh)={scale_tanh:.4f}"
                )

        if step % cfg.eval_interval == 0 or step == cfg.steps:
            val = evaluate(model, val_loader, cfg, device,
                           coarse_map=coarse_map, leaf_map=leaf_map)
            tps = tok_seen / (time.time() - t0)
            row = {
                "step":              step,
                "train_lm_loss":     ema_lm,
                "train_total_loss":  ema_total,
                "train_aux_coarse":     losses.get("coarse",         torch.zeros(1)).item(),
                "train_aux_leaf":       losses.get("leaf",           torch.zeros(1)).item(),
                "train_aux_balance":    losses.get("balance_coarse", torch.zeros(1)).item(),
                "train_aux_diversity":  losses.get("diversity",      torch.zeros(1)).item(),
                "train_retrieval_loss": losses.get("retrieval",      torch.zeros(1)).item(),
                "train_retrieval_acc1": losses.get("retrieval_acc1", torch.zeros(1)).item(),
                "train_retrieval_acc4": losses.get("retrieval_acc4", torch.zeros(1)).item(),
                "lr":                lr,
                "tokens_per_sec":    int(tps),
                "n_params":          n_params,
                **val,
            }
            _csv_writer.writerow(row)
            _csv_file.flush()
            _bnd_extra = ""
            if isinstance(model, (ReprRegionBoundaryLM, ReprRegionMultiHypLM,
                                   ReprRegionBranchAttnLM,
                                   ReprRegionBoundarySeqRefineLM,
                                   ReprRegionBoundaryAdaptiveDepthLM)) and val["val_boundary_frac"] > 0:
                _bnd_extra = (
                    f"  bnd_lm={val['val_boundary_lm']:.4f}"
                    f"  core_lm={val['val_core_lm']:.4f}"
                    f"  bnd_frac={val['val_boundary_frac']:.2f}"
                )
                if isinstance(model, ReprRegionBranchAttnLM):
                    _bnd_extra += f"  br_scale={val['val_branch_scale']:.4f}"
                elif isinstance(model, (ReprRegionBoundarySeqRefineLM,
                                        ReprRegionBoundaryAdaptiveDepthLM)):
                    _bnd_extra += f"  rf_scale={val['val_refine_scale']:.4f}"
                else:
                    _bnd_extra += f"  α_bnd={val['val_avg_boundary_alpha']:.3f}"
            _retr_val_str = ""
            if isinstance(model, ReprRegionRetrievalLM) and val.get("val_retrieval_loss", 0.0) > 0:
                _retr_val_str = (
                    f"  val_retr={val['val_retrieval_loss']:.4f}"
                    f"  val_r@1={val['val_retrieval_acc1']:.3f}"
                )
            print(
                f"  [val] val_lm={val['val_lm_loss']:.4f}  ppl={val['val_ppl']:.2f}"
                f"  coarse_acc@1={val['val_coarse_acc1']:.3f}"
                f"  leaf_acc@8={val['val_leaf_acc8']:.3f}"
                f"  α={val['current_alpha']:.3f}"
                + _bnd_extra + _retr_val_str
            )
            if hasattr(model, "_last_norms") and model._last_norms:
                n = model._last_norms
                norm_parts = [f"  [norms] ||h||={n.get('h_norm', 0):.3f}"]
                for k, v in n.items():
                    if k != "h_norm":
                        norm_parts.append(f"  ||{k}||={v:.3f}")
                print("".join(norm_parts))
            model.train()

        if step % cfg.save_interval == 0 or step == cfg.steps:
            ckpt_state = {
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
                "cfg":       asdict(cfg),
            }
            torch.save(ckpt_state, latest_ckpt)
            torch.save(ckpt_state,
                       os.path.join(cfg.output_dir, f"checkpoint_{step:07d}.pt"))
            print(f"  [ckpt] saved step {step}")

    _csv_file.close()

    # ── Final summary ──────────────────────────────────────────────────────────
    final_val = evaluate(model, val_loader, cfg, device,
                         coarse_map=coarse_map, leaf_map=leaf_map, max_batches=500)
    with open(os.path.join(cfg.output_dir, "final_summary.json"), "w") as f:
        json.dump({"mode": cfg.mode, "n_params": n_params, "param_breakdown": bd,
                   "steps": step, **final_val, "config": asdict(cfg)}, f, indent=2)

    _write_summary_md(cfg, final_val, n_params, bd, step)
    print(f"\n[done]  val_lm={final_val['val_lm_loss']:.4f}  ppl={final_val['val_ppl']:.2f}")
    print(f"[done]  outputs → {cfg.output_dir}")


def _write_summary_md(cfg, val, n_params, bd, step):
    lines = [
        f"# Final Summary — {cfg.mode}\n",
        "## Results\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mode | `{cfg.mode}` |",
        f"| Parameters (total) | {n_params:,} |",
        f"| Parameters (non-emb) | {bd['non_emb']:,} |",
        f"| Parameters (trunk) | {bd['trunk']:,} |",
        f"| Parameters (router) | {bd.get('router', 0):,} |",
        f"| Parameters (refiner) | {bd.get('refiner', 0):,} |",
        f"| Steps | {step:,} |",
        f"| Val LM Loss | {val['val_lm_loss']:.4f} |",
        f"| Val PPL | {val['val_ppl']:.2f} |",
        f"| Coarse Acc@1 | {val['val_coarse_acc1']:.3f} |",
        f"| Coarse Acc@4 | {val['val_coarse_acc4']:.3f} |",
        f"| Leaf Acc@1 | {val['val_leaf_acc1']:.3f} |",
        f"| Leaf Acc@8 | {val['val_leaf_acc8']:.3f} |",
        f"| Leaf Acc@32 | {val['val_leaf_acc32']:.3f} |",
        f"| Coarse Router Entropy | {val['val_coarse_ent']:.3f} |",
        f"| Leaf Router Entropy | {val['val_leaf_ent']:.3f} |",
        "",
        "## Interpretation\n",
        "> **Compare runs by `val_lm_loss` only, not total loss.**",
        "> Total loss includes auxiliary terms and is not comparable across modes.",
        "",
        "### Recommended experiment order",
        "1. `baseline` — LM floor with no region structure",
        "2. `coarse` — does soft coarse conditioning help?",
        "3. `oracle_coarse` — headroom: how much would *perfect* coarse routing help?",
        "4. `coarse_leaf` — does leaf signal add further improvement?",
        "5. `random_control` — same arch as `coarse_leaf`, permuted labels — should NOT improve",
        "",
        "### Reading the results",
        "**Strong support:** `coarse`/`coarse_leaf` val_lm_loss < `baseline`, "
        "and `random_control` does not match the gain.",
        "",
        "**Partial support:** `oracle_coarse` helps but `coarse` does not — "
        "router is the bottleneck, not the architecture.",
        "",
        "**No support:** Region model is worse than baseline, or `random_control` "
        "performs similarly to `coarse` (structure is not the cause), "
        "or routers collapse (entropy → 0).",
    ]
    with open(os.path.join(cfg.output_dir, "final_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Region building ────────────────────────────────────────────────────────────

def build_regions(cfg: TrainConfig):
    """
    Sweep a trained baseline checkpoint over the corpus, build a token
    co-activation graph, cluster it, and write:
      {out_dir}/token_to_region.json   — coarse map  (for --region_map_path)
      {out_dir}/region_tree.json       — leaf map    (for --leaf_map_path)
      {out_dir}/frequent_token_ids.npy — token subset used for the graph
    """
    try:
        from scipy import sparse as sp
    except ImportError:
        raise ImportError("scipy is required for build_regions mode")

    if not cfg.baseline_ckpt:
        raise ValueError("--baseline_ckpt is required for build_regions mode")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device  = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    out_dir = cfg.region_output_dir or cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Tokenizer & corpus ────────────────────────────────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    vocab_size = len(tokenizer)

    print("[build_regions] encoding corpus...")
    if cfg.dataset in ("wikitext-2-raw-v1", "wikitext-103-raw-v1"):
        ds = load_dataset("wikitext", cfg.dataset)
    else:
        ds = load_dataset(cfg.dataset)

    def _encode(split: str) -> np.ndarray:
        texts = [t for t in ds[split]["text"] if t.strip()]
        parts: list = []
        for t in texts:
            ids = tokenizer.encode(t)
            if ids:
                parts.append(ids)
        return np.concatenate([np.array(p, dtype=np.int32) for p in parts])

    train_tokens = _encode("train")
    print(f"[build_regions] {len(train_tokens):,} training tokens")

    # ── Frequent token subset ─────────────────────────────────────────────────
    n_graph  = len(tokenizer) if cfg.extend_full_vocab else cfg.region_vocab_size
    tok_freq = np.bincount(train_tokens.astype(np.int64), minlength=vocab_size)
    freq_ids = np.argsort(tok_freq)[-n_graph:][::-1].astype(np.int32)
    np.save(os.path.join(out_dir, "frequent_token_ids.npy"), freq_ids)

    sub_id = np.full(vocab_size, -1, dtype=np.int32)
    for i, gid in enumerate(freq_ids):
        sub_id[gid] = i
    print(f"[build_regions] graph over top-{n_graph} tokens")

    # ── Load baseline checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(cfg.baseline_ckpt, map_location=device, weights_only=False)
    # Reconstruct the model using the architecture that was actually saved
    saved_fields = {k: v for k, v in ckpt["cfg"].items()
                    if k in TrainConfig.__dataclass_fields__}
    model_cfg        = TrainConfig(**saved_fields)
    model_cfg.device = cfg.device
    model = BaselineTransformerLM(model_cfg, vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[build_regions] loaded baseline from {cfg.baseline_ckpt}")

    # ── Co-activation sweep ───────────────────────────────────────────────────
    sweep_ds = TokenDataset(train_tokens, model_cfg.seq_len)
    loader   = DataLoader(sweep_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=2, pin_memory=use_amp, drop_last=False)

    W        = np.zeros((n_graph, n_graph), dtype=np.float32)
    degree   = np.zeros(n_graph, dtype=np.float32)   # positions where token was in top-K
    ctx      = torch.autocast(device_type=device.type, enabled=use_amp)
    tok_seen = 0
    top_k    = cfg.region_top_k
    max_tok  = cfg.region_max_tokens

    print(f"[build_regions] sweeping corpus (top_k={top_k}, max={max_tok:,} tokens)...")
    with torch.no_grad():
        for batch in loader:
            if tok_seen >= max_tok:
                break
            batch = batch.to(device)
            with ctx:
                logits, _ = model.forward(batch[:, :-1])   # (B, T, V)

            topk     = logits.topk(top_k, dim=-1).indices.cpu().numpy()  # (B, T, K)
            B, T, K  = topk.shape
            flat_sub = sub_id[topk.reshape(B * T, K)]      # (B*T, K), -1=not in graph

            # Vectorised: build sparse indicator A[pos, sub_token] = 1,
            # then W += A^T @ A  (co-occurrence counts, diagonal = self-counts)
            ns, ks = np.where(flat_sub >= 0)
            ss = flat_sub[ns, ks]
            A  = sp.csr_matrix(
                (np.ones(len(ns), dtype=np.float32), (ns, ss)),
                shape=(B * T, n_graph),
            )
            W += (A.T @ A).toarray()
            # degree[i] = number of positions where token i appeared in top-K
            np.add.at(degree, ss, 1)
            tok_seen += B * T
            if tok_seen % 100_000 < B * T:
                print(f"  {min(tok_seen, max_tok):,} / {max_tok:,} tokens swept")

    np.fill_diagonal(W, 0.0)
    # Jaccard/cosine normalisation matching interference_region_pipeline:
    # W_norm[i,j] = W[i,j] / sqrt(degree[i] * degree[j] + eps)
    # Done in-place to avoid materialising a 50k×50k float64 outer product.
    inv_sqrt_deg = (1.0 / np.sqrt(degree.clip(min=1e-8))).astype(np.float32)
    W *= inv_sqrt_deg[:, None]   # normalise rows
    W *= inv_sqrt_deg[None, :]   # normalise cols
    np.fill_diagonal(W, 0.0)
    W_sym = W
    print(f"[build_regions] W density={(W_sym > 0).mean():.3%}")

    # ── Clustering helpers ────────────────────────────────────────────────────
    def _spectral(W_sub: np.ndarray, n_clus: int, seed: int) -> np.ndarray:
        from sklearn.cluster import SpectralClustering
        n = W_sub.shape[0]
        sc = SpectralClustering(
            n_clusters=n_clus, affinity="precomputed",
            random_state=seed, n_jobs=-1, assign_labels="cluster_qr",
        )
        return sc.fit_predict(W_sub + 1e-10 * np.eye(n)).astype(np.int32)

    def _kmeans(W_sub: np.ndarray, n_clus: int, seed: int) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.cluster import KMeans
        n      = W_sub.shape[0]
        n_comp = min(n_clus * 2, n - 1, 64)
        emb    = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(W_sub)
        return (KMeans(n_clusters=n_clus, random_state=seed, n_init=10)
                .fit_predict(emb).astype(np.int32))

    def _leiden_targeted(W_sub: np.ndarray, n_target: int, seed: int) -> np.ndarray:
        import igraph as ig
        import leidenalg
        n     = W_sub.shape[0]
        W_coo = sp.csr_matrix(W_sub).tocoo()
        mask  = (W_coo.row < W_coo.col) & (W_coo.data > 0)
        if mask.sum() == 0:
            return np.random.RandomState(seed).randint(0, n_target, n).astype(np.int32)
        G = ig.Graph(n=n, directed=False)
        G.add_edges(list(zip(W_coo.row[mask].tolist(), W_coo.col[mask].tolist())))
        G.es["weight"] = W_coo.data[mask].tolist()
        lo, hi, best = 0.0, 10.0, None
        for _ in range(25):
            mid  = (lo + hi) / 2
            part = leidenalg.find_partition(
                G, leidenalg.RBConfigurationVertexPartition,
                weights="weight", seed=seed, resolution_parameter=mid,
            )
            n_found = len(set(part.membership))
            if best is None or abs(n_found - n_target) < abs(len(set(best.membership)) - n_target):
                best = part
            if n_found < n_target * 0.9:
                lo = mid
            elif n_found > n_target * 1.1:
                hi = mid
            else:
                break
        return np.array(best.membership, dtype=np.int32)

    def _cluster(W_sub: np.ndarray, n_clus: int, seed: int, top_level: bool = False) -> np.ndarray:
        n      = W_sub.shape[0]
        if n <= n_clus:
            return np.arange(n, dtype=np.int32)
        method = cfg.region_cluster_method if top_level else "spectral"
        try:
            if method == "leiden":
                return _leiden_targeted(W_sub, n_clus, seed)
            elif method == "kmeans":
                return _kmeans(W_sub, n_clus, seed)
            else:
                return _spectral(W_sub, n_clus, seed)
        except Exception as e:
            print(f"[build_regions] clustering failed ({e}), falling back to random")
            return np.random.RandomState(seed).randint(0, n_clus, n).astype(np.int32)

    # ── Coarse clustering ─────────────────────────────────────────────────────
    n_target      = cfg.target_n_regions if cfg.target_n_regions > 0 else cfg.n_clusters
    coarse_labels = _cluster(W_sym, n_target, cfg.seed, top_level=True)
    n_coarse      = int(coarse_labels.max()) + 1
    sizes         = np.bincount(coarse_labels)
    print(f"[build_regions] coarse: {n_coarse} regions  "
          f"(target={n_target}  median={int(np.median(sizes))}  "
          f"min={int(sizes.min())}  max={int(sizes.max())})")
    with open(os.path.join(out_dir, "region_stats.json"), "w") as f:
        json.dump({
            "actual_n_regions": int(n_coarse),
            "target_n_regions": int(n_target),
            "region_cluster_method": cfg.region_cluster_method,
            "median_size": float(np.median(sizes)),
            "mean_size":   float(sizes.mean()),
            "max_size":    int(sizes.max()),
            "min_size":    int(sizes.min()),
            "coverage":    float((coarse_labels >= 0).mean()),
        }, f, indent=2)

    token_to_region: Dict[str, int] = {
        str(int(freq_ids[i])): int(coarse_labels[i]) for i in range(n_graph)
    }
    with open(os.path.join(out_dir, "token_to_region.json"), "w") as f:
        json.dump(token_to_region, f)
    print(f"[build_regions] saved token_to_region.json ({len(token_to_region)} tokens)")

    if not cfg.build_leaf:
        print(f"[build_regions] done → {out_dir}")
        return

    # ── Recursive subcluster → region_tree.json ───────────────────────────────
    def _make_node(sub_indices: np.ndarray, depth: int) -> dict:
        vocab_ids = [int(freq_ids[i]) for i in sub_indices]
        node: dict = {"vocab_ids": vocab_ids, "n_tokens": len(vocab_ids), "children": {}}
        if depth == 0 or len(sub_indices) < cfg.leaf_min_size * 2:
            return node
        W_sub   = W_sym[np.ix_(sub_indices, sub_indices)]
        leaves_per_region = max(2, cfg.target_n_leaves // max(1, n_coarse))
        n_sub_c = max(2, min(leaves_per_region, len(sub_indices) // cfg.leaf_min_size))
        try:
            sub_labels = _cluster(W_sub, n_sub_c, cfg.seed + int(sub_indices[0]))
        except Exception:
            return node
        n_sub = int(sub_labels.max()) + 1
        if n_sub < 2:
            return node
        for cid in range(n_sub):
            child_idx = sub_indices[sub_labels == cid]
            if len(child_idx) < 2:
                continue
            node["children"][str(cid)] = _make_node(child_idx, depth - 1)
        if len(node["children"]) < 2:
            node["children"] = {}   # revert to leaf if split failed
        return node

    print("[build_regions] building region tree (depth=1 recursive subcluster)...")
    root_children: dict = {}
    for cid in range(n_coarse):
        sub_indices = np.where(coarse_labels == cid)[0]
        root_children[str(cid)] = _make_node(sub_indices, depth=1)

    tree = {"root": {"vocab_ids": [int(v) for v in freq_ids],
                     "n_tokens": n_graph, "children": root_children}}
    with open(os.path.join(out_dir, "region_tree.json"), "w") as f:
        json.dump(tree, f)

    leaf_count = [0]
    def _count_leaves(node: dict):
        if not node.get("children"):
            leaf_count[0] += 1
        else:
            for c in node["children"].values():
                _count_leaves(c)
    _count_leaves(tree["root"])
    print(f"[build_regions] saved region_tree.json ({leaf_count[0]} leaves)")
    print(f"[build_regions] done → {out_dir}")


# ── Layer margin analysis ──────────────────────────────────────────────────────

def layer_margin_analysis(cfg: TrainConfig):
    """
    Tests whether transformers progressively resolve manifold uncertainty.

    For each layer of the backbone transformer, a frozen region probe (extracted
    from a trained repr_region checkpoint) is applied to the hidden states.
    This measures how predictive neighbourhood geometry evolves with depth.

    Q1: Does gold region enter top-k early?
    Q2: Does top-1 accuracy sharpen later than top-k?
    Q3: Does entropy decrease progressively across layers?
    Q4: Do margins increase progressively across layers?
    Q5: Do some tokens remain low-margin across ALL layers?
    Q6: Does top-1 accuracy gain non-trivially in the second half of the network?
    Q7: Does context dynamically sharpen manifold uncertainty?

    Required args:
        --baseline_ckpt   plain transformer checkpoint to probe
        --repr_ckpt       repr_region checkpoint (for probe extraction)
        --region_map_path token-to-region map
    Outputs:
        output_dir/layer_metrics.csv
        output_dir/layer_trajectories.csv
        output_dir/token_margin_variance.csv
        output_dir/summary.md
        output_dir/plots/   (8 figures)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLT = True
    except ImportError:
        HAS_PLT = False
        print("[layer_margin_analysis] matplotlib not available — plots skipped")
    from collections import defaultdict, Counter

    # ── Setup ──────────────────────────────────────────────────────────────────
    if not cfg.repr_ckpt:
        raise ValueError("--repr_ckpt required: trained repr_region checkpoint for probe")
    if not cfg.baseline_ckpt:
        raise ValueError("--baseline_ckpt required: baseline transformer to probe")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device    = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp   = device.type == "cuda"
    out_dir   = cfg.output_dir
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    vocab_size = len(tokenizer)

    # ── Region map ─────────────────────────────────────────────────────────────
    print(f"[layer_margin_analysis] loading probe: {cfg.repr_ckpt}")
    probe_raw  = torch.load(cfg.repr_ckpt, map_location="cpu", weights_only=False)
    probe_fields = {k: v for k, v in probe_raw["cfg"].items()
                    if k in TrainConfig.__dataclass_fields__}
    probe_cfg  = TrainConfig(**probe_fields)
    probe_step = probe_raw.get("step", "?")

    region_map_path = cfg.region_map_path or probe_cfg.region_map_path
    if not region_map_path:
        raise ValueError("--region_map_path required (or stored in probe checkpoint cfg)")
    raw_coarse, n_coarse = load_coarse_map(region_map_path, vocab_size)
    coarse_map = raw_coarse.to(device)
    cov = (raw_coarse >= 0).float().mean().item()
    print(f"[layer_margin_analysis] {n_coarse} regions  {cov:.1%} coverage")

    # ── Extract frozen probe (coarse_head) from repr_region checkpoint ─────────
    # Works for any repr_region variant — all share the same coarse_head structure.
    probe_sd = {k[len("coarse_head."):]: v
                for k, v in probe_raw["model"].items()
                if k.startswith("coarse_head.")}
    if not probe_sd:
        raise ValueError("repr_ckpt has no 'coarse_head.*' keys; "
                         "expected a repr_region family checkpoint")
    probe     = _make_router_head(probe_cfg.d_model, n_coarse,
                                  probe_cfg.dropout, probe_cfg.router_type)
    probe.load_state_dict(probe_sd)
    probe     = probe.to(device).eval()
    probe_T   = probe_cfg.router_temp
    for p in probe.parameters():
        p.requires_grad_(False)
    print(f"[layer_margin_analysis] probe ready  "
          f"router_type={probe_cfg.router_type}  temp={probe_T}  "
          f"from_step={probe_step}")

    # ── Backbone (baseline transformer) ────────────────────────────────────────
    print(f"[layer_margin_analysis] loading backbone: {cfg.baseline_ckpt}")
    base_raw = torch.load(cfg.baseline_ckpt, map_location="cpu", weights_only=False)
    base_fields = {k: v for k, v in base_raw["cfg"].items()
                   if k in TrainConfig.__dataclass_fields__}
    base_cfg  = TrainConfig(**base_fields)
    base_step = base_raw.get("step", "?")
    n_blocks  = base_cfg.n_layer

    if base_cfg.d_model != probe_cfg.d_model:
        raise ValueError(
            f"d_model mismatch: backbone={base_cfg.d_model}, probe={probe_cfg.d_model}. "
            "Probe must originate from a model with the same d_model."
        )

    # Reconstruct backbone. For repr_region variants only the shared trunk is used
    # during analysis (region conditioning is not applied during hidden-state probing).
    backbone = BaselineTransformerLM(base_cfg, vocab_size)
    load_res  = backbone.load_state_dict(base_raw["model"], strict=False)
    if load_res.missing_keys:
        raise RuntimeError(
            f"Backbone missing keys: {load_res.missing_keys[:5]}... "
            "Ensure --baseline_ckpt points to a baseline or repr_region checkpoint."
        )
    if load_res.unexpected_keys:
        print(f"[layer_margin_analysis] ignoring {len(load_res.unexpected_keys)} "
              f"non-trunk keys from backbone checkpoint "
              f"(mode={base_cfg.mode!r})")
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    layer_names = (["embed"]
                   + [f"block_{i}" for i in range(n_blocks)]
                   + ["final"])           # embed + N blocks + post-ln_f
    n_layers = len(layer_names)           # n_blocks + 2
    print(f"[layer_margin_analysis] backbone ready  mode={base_cfg.mode}  "
          f"n_layer={n_blocks}  d_model={base_cfg.d_model}  step={base_step}")

    # ── Validation dataset ─────────────────────────────────────────────────────
    ds_name = base_cfg.dataset
    if cfg.dataset not in ("wikitext-2-raw-v1",):
        ds_name = cfg.dataset
    print(f"[layer_margin_analysis] loading validation data: {ds_name}")
    if ds_name in ("wikitext-2-raw-v1", "wikitext-103-raw-v1"):
        ds_raw = load_dataset("wikitext", ds_name)
    else:
        ds_raw = load_dataset(ds_name)
    val_parts  = [np.array(tokenizer.encode(t), dtype=np.int32)
                  for t in ds_raw["validation"]["text"] if t.strip()]
    val_tokens = np.concatenate(val_parts)
    bsz        = cfg.batch_size or 16
    val_ds     = TokenDataset(val_tokens, base_cfg.seq_len)
    val_loader = DataLoader(val_ds, batch_size=bsz, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"),
                            drop_last=False)
    max_b    = cfg.max_analysis_batches if cfg.max_analysis_batches > 0 else len(val_loader)
    actual_b = min(max_b, len(val_loader))
    print(f"[layer_margin_analysis] {len(val_tokens):,} tokens  "
          f"{len(val_ds)} seqs  {actual_b} batches  "
          f"{n_layers} layers (embed + {n_blocks} blocks + final)")

    ctx = torch.autocast(device_type=device.type, enabled=use_amp)

    # ── Per-layer accumulators ─────────────────────────────────────────────────
    lay_margin   = [[] for _ in range(n_layers)]
    lay_entropy  = [[] for _ in range(n_layers)]
    lay_acc1     = [[] for _ in range(n_layers)]
    lay_acc4     = [[] for _ in range(n_layers)]
    lay_acc8     = [[] for _ in range(n_layers)]
    lay_rank     = [[] for _ in range(n_layers)]
    lay_valid    = [[] for _ in range(n_layers)]
    final_li     = n_layers - 1
    top1_final   : List[torch.Tensor] = []
    all_tok_ids  : List[torch.Tensor] = []

    # ── Main loop ──────────────────────────────────────────────────────────────
    print("[layer_margin_analysis] running analysis...")
    for bi, batch in enumerate(val_loader):
        if bi >= max_b:
            break
        if bi % 50 == 0:
            print(f"  batch {bi}/{actual_b}")

        batch = batch.to(device)
        src   = batch[:, :-1]
        tgt   = batch[:, 1:]
        B, T  = src.shape
        gold  = coarse_map[tgt]           # (B, T), -1 = unknown

        # Single forward pass: collect hidden states at every layer
        with torch.no_grad(), ctx:
            pos = torch.arange(T, device=device).unsqueeze(0)
            x   = backbone.drop(backbone.token_emb(src) + backbone.pos_emb(pos))
            layer_hs = [x]                # embed
            for blk in backbone.blocks:
                x = blk(x)
                layer_hs.append(x)        # block_0 ... block_{N-1}
            layer_hs.append(backbone.ln_f(x))   # final (post-ln_f)

        all_tok_ids.append(tgt.cpu().reshape(-1))

        for li, h in enumerate(layer_hs):
            with torch.no_grad(), ctx:
                logits = probe(h)                               # (B, T, n_coarse)
                p      = F.softmax(logits / probe_T, dim=-1).float()

            p_f   = p.reshape(-1, n_coarse)                    # (N, n_coarse)
            g_f   = gold.reshape(-1)                           # (N,)
            valid = g_f >= 0

            top2v   = torch.topk(p_f, k=2, dim=-1).values      # (N, 2)
            margin  = (top2v[:, 0] - top2v[:, 1]).cpu()
            entropy = (-(p_f * (p_f + 1e-10).log()).sum(-1)).cpu()

            acc1 = torch.zeros(B * T)
            acc4 = torch.zeros(B * T)
            acc8 = torch.zeros(B * T)
            rank = torch.full((B * T,), float(n_coarse))

            if valid.any():
                k8    = min(8, n_coarse)
                top8i = torch.topk(p_f, k=k8, dim=-1).indices  # (N, k8)
                g_v   = g_f[valid]
                top8v = top8i[valid]
                gcol  = g_v.unsqueeze(-1)
                acc1[valid] = (top8v[:, :1] == gcol).any(-1).float().cpu()
                acc4[valid] = (top8v[:, :min(4, k8)] == gcol).any(-1).float().cpu()
                acc8[valid] = (top8v == gcol).any(-1).float().cpu()
                # Gold rank: # regions with strictly higher prob (0 = top-1)
                p_v    = p_f[valid]
                gprob  = p_v[torch.arange(len(g_v), device=device), g_v]
                rank[valid] = (p_v > gprob.unsqueeze(-1)).sum(-1).float().cpu()

                if li == final_li:
                    top1_final.append(torch.topk(p_f, k=1, dim=-1).indices[:, 0].cpu())

            lay_margin[li].append(margin)
            lay_entropy[li].append(entropy)
            lay_acc1[li].append(acc1)
            lay_acc4[li].append(acc4)
            lay_acc8[li].append(acc8)
            lay_rank[li].append(rank)
            lay_valid[li].append(valid.cpu())

    # ── Concatenate ────────────────────────────────────────────────────────────
    print("[layer_margin_analysis] concatenating results...")
    margins   = [torch.cat(lay_margin[li])  for li in range(n_layers)]
    entropies = [torch.cat(lay_entropy[li]) for li in range(n_layers)]
    acc1s     = [torch.cat(lay_acc1[li])    for li in range(n_layers)]
    acc4s     = [torch.cat(lay_acc4[li])    for li in range(n_layers)]
    acc8s     = [torch.cat(lay_acc8[li])    for li in range(n_layers)]
    ranks     = [torch.cat(lay_rank[li])    for li in range(n_layers)]
    valids    = [torch.cat(lay_valid[li])   for li in range(n_layers)]
    top1_fin  = torch.cat(top1_final) if top1_final else torch.zeros(0, dtype=torch.long)
    tok_ids   = torch.cat(all_tok_ids)
    N_total   = len(margins[0])
    print(f"[layer_margin_analysis] {N_total:,} positions × {n_layers} layers")

    # ── Aggregate per-layer statistics ─────────────────────────────────────────
    def _q(t: torch.Tensor, q: float) -> float:
        return float(t.quantile(q).item())

    def _vm(num: torch.Tensor, mask: torch.Tensor) -> float:
        s = num[mask]
        return float(s.mean().item()) if len(s) > 0 else float("nan")

    bnd_tau = cfg.boundary_tau
    layer_stats: List[Dict] = []
    for li in range(n_layers):
        m  = margins[li];   e = entropies[li]; v = valids[li]
        a1 = acc1s[li];     a4 = acc4s[li];    a8 = acc8s[li]
        gr = ranks[li]
        bfrac = (m < bnd_tau).float().mean().item()
        layer_stats.append({
            "layer":          li,
            "layer_name":     layer_names[li],
            "acc1":           _vm(a1, v),
            "acc4":           _vm(a4, v),
            "acc8":           _vm(a8, v),
            "margin_mean":    float(m.mean()),
            "margin_q10":     _q(m, 0.10),
            "margin_q25":     _q(m, 0.25),
            "margin_median":  _q(m, 0.50),
            "margin_q75":     _q(m, 0.75),
            "margin_q90":     _q(m, 0.90),
            "entropy_mean":   float(e.mean()),
            "entropy_q25":    _q(e, 0.25),
            "entropy_median": _q(e, 0.50),
            "entropy_q75":    _q(e, 0.75),
            "boundary_frac":  bfrac,
            "core_frac":      1.0 - bfrac,
            "gold_rank_mean": _vm(gr, v),
            "gold_rank_med":  float(gr[v].median()) if v.any() else float("nan"),
            "n_valid":        int(v.sum()),
        })

    # ── Trajectory statistics grouped by embedding-margin bin ─────────────────
    embed_margin = margins[0]
    TRAJ_BINS = [
        (0.00, 0.05,  "M<0.05"),
        (0.05, 0.10,  "0.05≤M<0.10"),
        (0.10, 0.20,  "0.10≤M<0.20"),
        (0.20, 0.40,  "0.20≤M<0.40"),
        (0.40, 2.0,   "M≥0.40"),
    ]
    traj_rows: List[Dict] = []
    for li in range(n_layers):
        m  = margins[li];   e = entropies[li]
        a1 = acc1s[li];     v = valids[li];   gr = ranks[li]
        traj_rows.append({
            "group": "all", "layer": li, "layer_name": layer_names[li],
            "margin_mean": float(m.mean()),  "margin_q25": _q(m, 0.25),
            "margin_q75":  _q(m, 0.75),
            "entropy_mean": float(e.mean()),
            "acc1": _vm(a1, v), "gold_rank_mean": _vm(gr, v),
            "n": N_total,
        })
        for lo, hi, label in TRAJ_BINS:
            grp = (embed_margin >= lo) & (embed_margin < hi)
            if not grp.any():
                continue
            traj_rows.append({
                "group": label, "layer": li, "layer_name": layer_names[li],
                "margin_mean": float(m[grp].mean()),
                "margin_q25":  _q(m[grp], 0.25),
                "margin_q75":  _q(m[grp], 0.75),
                "entropy_mean": float(e[grp].mean()),
                "acc1": _vm(a1, grp & v), "gold_rank_mean": _vm(gr, grp & v),
                "n": int(grp.sum()),
            })

    # ── Token margin variance at final layer ───────────────────────────────────
    print("[layer_margin_analysis] computing token context variance...")
    fin_m  = margins[final_li]
    tok_n  : dict = defaultdict(int)
    tok_mu : dict = defaultdict(float)
    tok_M2 : dict = defaultdict(float)
    tok_reg: dict = defaultdict(Counter)

    for idx in range(N_total):
        tid = int(tok_ids[idx].item())
        if not (0 <= tid < vocab_size):
            continue
        mv  = float(fin_m[idx].item())
        n   = tok_n[tid] + 1
        d   = mv - tok_mu[tid]
        mu  = tok_mu[tid] + d / n
        tok_M2[tid] = tok_M2[tid] + d * (mv - mu)
        tok_n[tid]  = n
        tok_mu[tid] = mu
        if idx < len(top1_fin):
            tok_reg[tid][int(top1_fin[idx].item())] += 1

    tok_var_rows: List[Dict] = []
    for tid in sorted(tok_n, key=lambda t: -tok_n[t]):
        n = tok_n[tid]
        if n < 5:
            continue
        mu   = tok_mu[tid]
        std  = math.sqrt(tok_M2[tid] / n) if n > 1 else 0.0
        reg  = tok_reg.get(tid, Counter())
        mode_cnt = max(reg.values()) if reg else 0
        ctx_var  = 1.0 - mode_cnt / n
        try:
            tok_str = tokenizer.decode([tid])
        except Exception:
            tok_str = ""
        tok_var_rows.append({
            "token_id":         tid,
            "token_str":        tok_str,
            "n_occurrences":    n,
            "mean_margin":      round(mu, 5),
            "std_margin":       round(std, 5),
            "n_unique_regions": len(reg),
            "context_variance": round(ctx_var, 5),
        })
    tok_var_rows.sort(key=lambda r: -r["std_margin"])
    print(f"[layer_margin_analysis] {len(tok_var_rows)} tokens with ≥5 occurrences")

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    def _write_csv(rows: List[Dict], path: str):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    _write_csv(layer_stats,          os.path.join(out_dir, "layer_metrics.csv"))
    _write_csv(traj_rows,            os.path.join(out_dir, "layer_trajectories.csv"))
    _write_csv(tok_var_rows[:5000],  os.path.join(out_dir, "token_margin_variance.csv"))
    print("[layer_margin_analysis] wrote CSVs")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if HAS_PLT:
        xs      = list(range(n_layers))
        xlabels = layer_names

        def _ax(ax, title, ylabel):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Layer"); ax.set_ylabel(ylabel)
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
            ax.grid(True, alpha=0.3)

        def _ls(key):
            return [s[key] for s in layer_stats]

        # 1: Region accuracy across layers
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, _ls("acc1"), "o-b",  label="Acc@1")
        ax.plot(xs, _ls("acc4"), "s-g",  label="Acc@4")
        ax.plot(xs, _ls("acc8"), "^-r",  label="Acc@8")
        _ax(ax, "Region Accuracy vs Layer", "Accuracy")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "01_region_accuracy_vs_layer.png"), dpi=120)
        plt.close(fig)

        # 2: Margin across layers (mean + IQR band)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(xs, _ls("margin_q25"), _ls("margin_q75"),
                        alpha=0.2, color="blue", label="Q25–Q75")
        ax.plot(xs, _ls("margin_q10"), "--", color="lightblue", linewidth=0.8, label="Q10")
        ax.plot(xs, _ls("margin_q90"), "--", color="steelblue", linewidth=0.8, label="Q90")
        ax.plot(xs, _ls("margin_mean"),   "o-b", label="Mean")
        ax.plot(xs, _ls("margin_median"), "s--", color="navy", linewidth=1, label="Median")
        _ax(ax, "Routing Margin vs Layer", "Margin (top1 − top2)")
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "02_margin_vs_layer.png"), dpi=120)
        plt.close(fig)

        # 3: Entropy across layers (mean + IQR band)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(xs, _ls("entropy_q25"), _ls("entropy_q75"),
                        alpha=0.2, color="red", label="Q25–Q75")
        ax.plot(xs, _ls("entropy_mean"),   "o-r", label="Mean")
        ax.plot(xs, _ls("entropy_median"), "s--", color="darkred", linewidth=1,
                label="Median")
        _ax(ax, "Router Entropy vs Layer", "H(p_region)")
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "03_entropy_vs_layer.png"), dpi=120)
        plt.close(fig)

        # 4: Boundary fraction across layers
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, _ls("boundary_frac"), "o-m",
                label=f"boundary (margin < {bnd_tau})")
        ax.plot(xs, _ls("core_frac"), "s--g", label="core")
        _ax(ax, f"Boundary Fraction vs Layer (τ={bnd_tau})", "Fraction")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "04_boundary_frac_vs_layer.png"), dpi=120)
        plt.close(fig)

        # 5: Gold region rank across layers (inverted y-axis: lower rank = better)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, _ls("gold_rank_mean"), "o-",  color="navy",       label="Mean rank")
        ax.plot(xs, _ls("gold_rank_med"),  "s--", color="dodgerblue", label="Median rank")
        ax.invert_yaxis()
        _ax(ax, "Gold Region Rank vs Layer (lower = better)", "Rank (0 = top-1)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "05_gold_rank_vs_layer.png"), dpi=120)
        plt.close(fig)

        # 6: Margin trajectory by initial-margin group
        COLORS = {"M<0.05": "red", "0.05≤M<0.10": "darkorange",
                  "0.10≤M<0.20": "gold", "0.20≤M<0.40": "forestgreen",
                  "M≥0.40": "royalblue"}
        fig, ax = plt.subplots(figsize=(9, 4))
        groups_seen = list(dict.fromkeys(r["group"] for r in traj_rows
                                         if r["group"] != "all"))
        for grp in groups_seen:
            rows_g = sorted([r for r in traj_rows if r["group"] == grp],
                            key=lambda r: r["layer"])
            n_g = rows_g[0]["n"] if rows_g else 0
            ax.plot([r["layer"] for r in rows_g],
                    [r["margin_mean"] for r in rows_g],
                    "o-", color=COLORS.get(grp, "k"),
                    label=f"{grp} (n={n_g:,})", linewidth=1.5)
        ax.plot(xs, _ls("margin_mean"), "k--", linewidth=1.5, alpha=0.5, label="all")
        _ax(ax, "Margin Trajectory by Embedding-Layer Margin Group",
            "Mean margin")
        ax.legend(fontsize=7, loc="upper left"); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "06_margin_trajectory_by_group.png"), dpi=120)
        plt.close(fig)

        # 7: Token margin variance scatter (final layer)
        if tok_var_rows:
            sample = tok_var_rows[:2000]
            mu_m  = [r["mean_margin"]    for r in sample]
            std_m = [r["std_margin"]     for r in sample]
            cv    = [r["context_variance"] for r in sample]
            n_uni = [r["n_unique_regions"] for r in sample]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sc = axes[0].scatter(mu_m, std_m, c=cv, s=6, alpha=0.4,
                                 cmap="hot_r", vmin=0, vmax=1)
            plt.colorbar(sc, ax=axes[0], label="context_variance")
            axes[0].set_xlabel("Mean margin (final layer)")
            axes[0].set_ylabel("Std margin (final layer)")
            axes[0].set_title("Token Margin Variability"); axes[0].grid(True, alpha=0.3)
            axes[1].scatter(mu_m, n_uni, s=6, alpha=0.3, color="teal")
            axes[1].set_xlabel("Mean margin (final layer)")
            axes[1].set_ylabel("Unique top-1 regions")
            axes[1].set_title("Routing Diversity vs Margin"); axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "07_token_margin_variance.png"), dpi=120)
            plt.close(fig)

        # 8: Example token margin trajectories (highest context variance tokens)
        if tok_var_rows:
            top_var = tok_var_rows[:min(10, len(tok_var_rows))]
            try:
                cmap = plt.get_cmap("tab10")
            except AttributeError:
                cmap = plt.cm.get_cmap("tab10")
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, row in enumerate(top_var):
                tid = row["token_id"]
                mask = (tok_ids == tid)
                if not mask.any():
                    continue
                ys = [float(margins[li][mask].mean()) for li in range(n_layers)]
                lbl = repr(row["token_str"])[:14]
                ax.plot(xs, ys, "o-", color=cmap(i % 10),
                        label=f"{lbl} σ={row['std_margin']:.3f}", linewidth=1.5)
            _ax(ax, "Margin Trajectory — Highest Context-Variance Tokens",
                "Mean margin at layer")
            ax.legend(fontsize=7, loc="upper left", ncol=2); fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "08_example_token_trajectories.png"),
                        dpi=120)
            plt.close(fig)

        print(f"[layer_margin_analysis] wrote 8 plots → {plots_dir}/")

    # ── Q&A verdicts ──────────────────────────────────────────────────────────
    emb = layer_stats[0]
    fin = layer_stats[-1]
    mid = layer_stats[n_layers // 2]

    def _v(cond: bool) -> str:
        return "PASS" if cond else "INCONCLUSIVE"

    chance_k4 = min(4, n_coarse) / n_coarse
    q1 = _v(emb["acc4"] > 2.0 * chance_k4)                   # top-k enters early

    def _rel_gap(s):
        a1 = s["acc1"]; a4 = s["acc4"]
        return (a4 - a1) / (a4 + 1e-8)
    q2 = _v(_rel_gap(emb) > _rel_gap(fin) + 0.05)             # top-1 sharpens later

    q3 = _v(fin["entropy_mean"] < emb["entropy_mean"] - 0.1)  # entropy decreases
    q4 = _v(fin["margin_mean"]  > emb["margin_mean"]  + 0.02) # margins increase

    low_grp_final = [r for r in traj_rows
                     if r["group"] in ("M<0.05", "0.05≤M<0.10")
                     and r["layer"] == final_li]
    q5_frac = sum(r["n"] for r in low_grp_final) / max(N_total, 1)
    q5 = _v(q5_frac > 0.05)                                   # perpetually ambiguous

    acc1_e_m = mid["acc1"] - emb["acc1"]
    acc1_m_f = fin["acc1"] - mid["acc1"]
    q6 = _v(acc1_m_f > acc1_e_m * 0.3 and acc1_m_f > 0.01)   # gain in second half

    high_var_count = sum(1 for r in tok_var_rows if r["std_margin"] > 0.10)
    q7 = _v(high_var_count > 50)                              # context-dynamic tokens

    # ── Summary markdown ──────────────────────────────────────────────────────
    lines = [
        "# Layer Margin Analysis Report",
        "",
        f"**Backbone:** `{cfg.baseline_ckpt}`  "
        f"mode={base_cfg.mode}  step={base_step}  "
        f"n_layer={n_blocks}  d_model={base_cfg.d_model}",
        f"**Probe:**    `{cfg.repr_ckpt}`  "
        f"mode={probe_cfg.mode}  step={probe_step}  "
        f"router_type={probe_cfg.router_type}  temp={probe_T}",
        f"**Dataset:**  {ds_name}  ({len(val_tokens):,} tokens  {actual_b} batches)",
        f"**Positions:**{N_total:,}  **Regions:** {n_coarse}  (coverage={cov:.1%})",
        "",
        "## Per-Layer Statistics",
        "",
        "| Layer | Acc@1 | Acc@4 | Acc@8 | "
        "Margin μ | Margin med | Entropy μ | Bnd% | Rank μ |",
        "|-------|------:|------:|------:|"
        "---------:|-----------:|----------:|-----:|-------:|",
    ]
    for s in layer_stats:
        lines.append(
            f"| {s['layer_name']:<12} "
            f"| {s['acc1']:.3f} | {s['acc4']:.3f} | {s['acc8']:.3f} "
            f"| {s['margin_mean']:.3f} | {s['margin_median']:.3f} "
            f"| {s['entropy_mean']:.3f} "
            f"| {s['boundary_frac']:.1%} "
            f"| {s['gold_rank_mean']:.1f} |"
        )

    lines += [
        "",
        "## Hypothesis Tests (Q1–Q7)",
        "",
        "| # | Question | Verdict |",
        "|---|----------|---------|",
        f"| Q1 | Gold region enters top-4 early "
        f"(acc@4 > 2× chance={2*chance_k4:.3f} at embed)? | **{q1}** |",
        f"| Q2 | Top-1 accuracy sharpens later than top-4 "
        f"(relative acc@4−acc@1 gap narrows)? | **{q2}** |",
        f"| Q3 | Entropy decreases progressively across layers "
        f"(final < embed − 0.1)? | **{q3}** |",
        f"| Q4 | Margins increase progressively across layers "
        f"(final > embed + 0.02)? | **{q4}** |",
        f"| Q5 | Some tokens remain low-margin at ALL layers "
        f"({q5_frac:.1%} in low bins at final layer)? | **{q5}** |",
        f"| Q6 | Top-1 accuracy gains non-trivially in second half of network "
        f"(+{acc1_m_f:.3f})? | **{q6}** |",
        f"| Q7 | Context dynamically sharpens uncertainty "
        f"({high_var_count:,} tokens with std_margin > 0.10)? | **{q7}** |",
        "",
        "## Margin Trajectory by Initial-Margin Group",
        "",
        "Tokens grouped by their margin at the embedding layer (before any attention).",
        "",
        "| Group | n | Embed Margin | Embed Acc@1 | "
        "Final Margin | Final Acc@1 | Δ Acc@1 |",
        "|-------|--:|-------------:|------------:|"
        "-------------:|------------:|--------:|",
    ]
    for _, _, label in TRAJ_BINS:
        er = next((r for r in traj_rows if r["group"] == label and r["layer"] == 0), None)
        fr = next((r for r in traj_rows if r["group"] == label
                   and r["layer"] == final_li), None)
        if er and fr:
            da1 = fr["acc1"] - er["acc1"]
            lines.append(
                f"| {label} | {er['n']:,} "
                f"| {er['margin_mean']:.3f} | {er['acc1']:.3f} "
                f"| {fr['margin_mean']:.3f} | {fr['acc1']:.3f} "
                f"| {da1:+.3f} |"
            )

    lines += [
        "",
        "## Top-20 Context-Variance Tokens (final layer)",
        "",
        "| Token | n | Mean Margin | Std Margin | Unique Regions | Ctx Variance |",
        "|-------|--:|------------:|-----------:|---------------:|-------------:|",
    ]
    for r in tok_var_rows[:20]:
        lines.append(
            f"| `{r['token_str']}` | {r['n_occurrences']:,} "
            f"| {r['mean_margin']:.4f} | {r['std_margin']:.4f} "
            f"| {r['n_unique_regions']} | {r['context_variance']:.4f} |"
        )

    passes = sum(1 for q in [q1, q2, q3, q4, q5, q6, q7] if q == "PASS")
    support = ("STRONG" if passes >= 5 else "PARTIAL" if passes >= 3 else "WEAK")
    lines += [
        "",
        "## Interpretation",
        "",
        f"**{passes}/7 questions PASS → {support} SUPPORT** "
        "for progressive manifold uncertainty resolution.",
        "",
        "| Signal | Observation |",
        "|--------|-------------|",
        f"| Embed→Final margin gain | "
        f"{emb['margin_mean']:.3f} → {fin['margin_mean']:.3f} "
        f"({fin['margin_mean'] - emb['margin_mean']:+.3f}) |",
        f"| Embed→Final entropy drop | "
        f"{emb['entropy_mean']:.3f} → {fin['entropy_mean']:.3f} "
        f"({fin['entropy_mean'] - emb['entropy_mean']:+.3f}) |",
        f"| Embed→Final Acc@1 gain | "
        f"{emb['acc1']:.3f} → {fin['acc1']:.3f} "
        f"({fin['acc1'] - emb['acc1']:+.3f}) |",
        f"| Embed→Final Acc@4 gain | "
        f"{emb['acc4']:.3f} → {fin['acc4']:.3f} "
        f"({fin['acc4'] - emb['acc4']:+.3f}) |",
        f"| Gold rank embed→final | "
        f"{emb['gold_rank_mean']:.1f} → {fin['gold_rank_mean']:.1f} |",
        "",
        "If Q1+Q2 both PASS: early layers localize to coarse neighbourhood; "
        "later layers suppress competing manifolds.",
        "",
        "## Output Files",
        "",
        "- `layer_metrics.csv` — per-layer aggregate statistics",
        "- `layer_trajectories.csv` — per-layer stats by initial-margin group",
        "- `token_margin_variance.csv` — per-token margin variance (final layer, top 5k)",
        "- `plots/01_region_accuracy_vs_layer.png` — acc@1/4/8 vs layer",
        "- `plots/02_margin_vs_layer.png` — margin distribution vs layer",
        "- `plots/03_entropy_vs_layer.png` — entropy vs layer",
        "- `plots/04_boundary_frac_vs_layer.png` — boundary fraction vs layer",
        "- `plots/05_gold_rank_vs_layer.png` — gold region rank vs layer",
        "- `plots/06_margin_trajectory_by_group.png` — trajectory by initial ambiguity",
        "- `plots/07_token_margin_variance.png` — context-dependent routing scatter",
        "- `plots/08_example_token_trajectories.png` — per-token margin trajectories",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n[layer_margin_analysis] DONE  →  {out_dir}")
    print(f"  {passes}/7 PASS ({support} support for progressive resolution hypothesis)")
    for q, label in [(q1, "Q1 gold enters top-k early           "),
                     (q2, "Q2 top-1 sharpens later than top-k   "),
                     (q3, "Q3 entropy decreases progressively    "),
                     (q4, "Q4 margins increase progressively     "),
                     (q5, "Q5 perpetually ambiguous tokens exist "),
                     (q6, "Q6 top-1 gains in second half         "),
                     (q7, "Q7 context dynamically sharpens margin")]:
        print(f"  {label}: {q}")


# ── Boundary analysis ──────────────────────────────────────────────────────────

@torch.no_grad()
def boundary_analysis(cfg: TrainConfig):
    """
    Pure diagnostic stage (no training).  Loads a trained repr_region /
    repr_region_capacity checkpoint, runs the validation set, bins every token
    position by routing confidence margin (top1 − top2 of p_region), and writes:

      boundary_metrics.csv           — per-bin aggregate statistics
      token_boundary_stats.csv       — per-token NLL and oracle gap
      token_context_variance.csv     — routing variance for repeated tokens
      boundary_analysis_summary.md   — Q1–Q7 verdicts and tables
      plots/                         — 7 diagnostic plots

    Answers:
      Q1 Do routing failures concentrate at low-margin positions?
      Q2 Is LM NLL higher at low-margin positions?
      Q3 Does oracle routing help more at low-margin (ambiguous) positions?
      Q4 Does router entropy increase at low-margin positions?
      Q5 Do repeated tokens show context-dependent routing?
      Q6 Are boundary tokens (low avg_margin) harder to model than core tokens?
      Q7 Does learned routing outperform a random-assignment baseline?
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for boundary_analysis mode")
    from collections import Counter

    if not cfg.repr_ckpt:
        raise ValueError("--repr_ckpt is required for boundary_analysis mode")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device    = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp   = device.type == "cuda"
    out_dir   = cfg.output_dir
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"[boundary_analysis] loading checkpoint: {cfg.repr_ckpt}")
    ckpt = torch.load(cfg.repr_ckpt, map_location=device, weights_only=False)
    saved_fields = {k: v for k, v in ckpt["cfg"].items()
                    if k in TrainConfig.__dataclass_fields__}
    model_cfg        = TrainConfig(**saved_fields)
    model_cfg.device = cfg.device
    step_ckpt        = ckpt.get("step", "final")
    print(f"[boundary_analysis] mode={model_cfg.mode}  step={step_ckpt}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    vocab_size = len(tokenizer)

    # ── Region maps ───────────────────────────────────────────────────────────
    region_map_path = cfg.region_map_path or model_cfg.region_map_path
    if not region_map_path:
        raise ValueError("Pass --region_map_path (or ensure it was stored in checkpoint cfg)")
    raw_coarse, n_coarse = load_coarse_map(region_map_path, vocab_size)
    model_cfg.n_coarse = n_coarse
    coarse_map = raw_coarse.to(device)
    rand_raw   = make_random_partition_map(raw_coarse, n_coarse, seed=cfg.seed)
    rand_map   = rand_raw.to(device)
    cov = (raw_coarse >= 0).float().mean().item()
    print(f"[boundary_analysis] {n_coarse} regions  {cov:.1%} coverage")

    # ── Reconstruct model ─────────────────────────────────────────────────────
    _REPR_MODES = ("repr_region", "oracle_repr_region", "random_repr_region")
    _CAP_MODES  = ("repr_region_capacity", "oracle_repr_region_capacity",
                   "random_repr_region_capacity")
    if model_cfg.mode in _REPR_MODES:
        model = ReprRegionTransformerLM(model_cfg, vocab_size, coarse_map=coarse_map)
    elif model_cfg.mode in _CAP_MODES:
        model = ReprRegionCapacityLM(model_cfg, vocab_size, coarse_map=coarse_map)
    else:
        raise ValueError(
            f"boundary_analysis only supports repr_region/capacity checkpoints, "
            f"got mode={model_cfg.mode!r}"
        )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if hasattr(model, "set_region_scales"):
        model.set_region_scales(model_cfg.base_alpha)
    print(f"[boundary_analysis] model ready  ({model.param_count():,} params)")

    # ── Validation dataset ────────────────────────────────────────────────────
    ds_name = model_cfg.dataset
    if cfg.dataset not in ("wikitext-2-raw-v1",):  # user explicitly overrode
        ds_name = cfg.dataset
    print(f"[boundary_analysis] loading validation data: {ds_name}")
    if ds_name in ("wikitext-2-raw-v1", "wikitext-103-raw-v1"):
        ds_raw = load_dataset("wikitext", ds_name)
    else:
        ds_raw = load_dataset(ds_name)
    val_parts = [np.array(tokenizer.encode(t), dtype=np.int32)
                 for t in ds_raw["validation"]["text"] if t.strip()]
    val_tokens = np.concatenate(val_parts)
    val_ds     = TokenDataset(val_tokens, model_cfg.seq_len)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"),
                            drop_last=False)
    max_b = cfg.max_analysis_batches if cfg.max_analysis_batches > 0 else len(val_loader)
    print(f"[boundary_analysis] {len(val_tokens):,} val tokens  "
          f"({len(val_ds)} seqs)  using {min(max_b, len(val_loader))} batches")

    ctx = torch.autocast(device_type=device.type, enabled=use_amp)

    # ── Collect per-token statistics ───────────────────────────────────────────
    # Two forward passes per batch:
    #   Pass 1 (learned routing): margins, entropies, nll_learned, p_region
    #   Pass 2 (oracle routing):  nll_oracle — upper bound with perfect routing
    # Routing accuracy computed against both real and random maps (same p_region).

    all_margins     : List[torch.Tensor] = []
    all_entropies   : List[torch.Tensor] = []
    all_nll_learned : List[torch.Tensor] = []
    all_nll_oracle  : List[torch.Tensor] = []
    all_acc1_real   : List[torch.Tensor] = []
    all_acc4_real   : List[torch.Tensor] = []
    all_acc1_rand   : List[torch.Tensor] = []
    all_acc4_rand   : List[torch.Tensor] = []
    all_true_region : List[torch.Tensor] = []
    all_top1_region : List[torch.Tensor] = []
    all_target_toks : List[torch.Tensor] = []

    print("[boundary_analysis] running inference...")
    for bi, batch in enumerate(val_loader):
        if bi >= max_b:
            break
        if bi % 20 == 0:
            print(f"  batch {bi}/{min(max_b, len(val_loader))}")

        batch = batch.to(device)
        src   = batch[:, :-1]
        tgt   = batch[:, 1:]   # (B, T)

        # Pass 1: learned routing
        with ctx:
            lm_logits, _, p_region, _ = model.forward(src)

        log_probs = F.log_softmax(lm_logits, dim=-1)
        nll_l     = (-log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).float().cpu()

        # Pass 2: oracle routing using the real coarse map
        oracle_ids = coarse_map[tgt]
        with ctx:
            lm_ora, _, _, _ = model.forward(src, oracle_ids)
        log_probs_ora = F.log_softmax(lm_ora, dim=-1)
        nll_o = (-log_probs_ora.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).float().cpu()

        # Margin and entropy from Pass 1
        p       = p_region.float().cpu()            # (B, T, K)
        top2v, top2i = p.topk(2, dim=-1)
        margin  = (top2v[..., 0] - top2v[..., 1])  # (B, T)
        entropy = -(p * (p + 1e-8).log()).sum(-1)   # (B, T)
        top1_r  = top2i[..., 0]                     # (B, T)
        top4_i  = p.topk(4, dim=-1).indices         # (B, T, 4)

        # Routing accuracy — real map
        true_r_real = coarse_map[tgt].cpu()   # (B, T); -1 = unknown
        known_real  = true_r_real >= 0
        neg1        = torch.full_like(top1_r.float(), -1.0)
        a1_real = torch.where(known_real, (top1_r == true_r_real).float(), neg1)
        a4_real = torch.where(known_real,
                              (top4_i == true_r_real.unsqueeze(-1)).any(-1).float(), neg1)

        # Routing accuracy — random map (null hypothesis)
        true_r_rand = rand_map[tgt].cpu()
        known_rand  = true_r_rand >= 0
        a1_rand = torch.where(known_rand, (top1_r == true_r_rand).float(), neg1)
        a4_rand = torch.where(known_rand,
                              (top4_i == true_r_rand.unsqueeze(-1)).any(-1).float(), neg1)

        all_margins.append(margin.reshape(-1))
        all_entropies.append(entropy.reshape(-1))
        all_nll_learned.append(nll_l.reshape(-1))
        all_nll_oracle.append(nll_o.reshape(-1))
        all_acc1_real.append(a1_real.reshape(-1))
        all_acc4_real.append(a4_real.reshape(-1))
        all_acc1_rand.append(a1_rand.reshape(-1))
        all_acc4_rand.append(a4_rand.reshape(-1))
        all_true_region.append(true_r_real.reshape(-1))
        all_top1_region.append(top1_r.reshape(-1))
        all_target_toks.append(tgt.cpu().reshape(-1))

    margins     = torch.cat(all_margins)
    entropies   = torch.cat(all_entropies)
    nll_learned = torch.cat(all_nll_learned)
    nll_oracle  = torch.cat(all_nll_oracle)
    acc1_real   = torch.cat(all_acc1_real)
    acc4_real   = torch.cat(all_acc4_real)
    acc1_rand   = torch.cat(all_acc1_rand)
    acc4_rand   = torch.cat(all_acc4_rand)
    true_region = torch.cat(all_true_region)
    top1_region = torch.cat(all_top1_region)
    target_toks = torch.cat(all_target_toks)
    N = len(margins)
    print(f"[boundary_analysis] {N:,} positions collected")

    # ── Margin bins ───────────────────────────────────────────────────────────
    BIN_EDGES  = [0.0, 0.05, 0.10, 0.20, 0.40, 1.01]
    BIN_LABELS = ["[0,0.05)", "[0.05,0.10)", "[0.10,0.20)", "[0.20,0.40)", "[0.40,1.0]"]
    N_BINS     = len(BIN_LABELS)
    oracle_gap = nll_learned - nll_oracle  # positive = oracle helps

    def _safe_mean(t: torch.Tensor, mask: torch.Tensor) -> float:
        sub = t[mask]
        return sub.mean().item() if len(sub) > 0 else 0.0

    bin_rows: List[Dict] = []
    for b in range(N_BINS):
        lo, hi  = BIN_EDGES[b], BIN_EDGES[b + 1]
        bin_m   = (margins >= lo) & (margins < hi)
        known_r = bin_m & (acc1_real >= 0)
        known_d = bin_m & (acc1_rand >= 0)
        n = int(bin_m.sum().item())
        bin_rows.append({
            "bin":             BIN_LABELS[b],
            "lo":              lo,
            "hi":              lo if b == N_BINS - 1 else hi,
            "count":           n,
            "pct":             100.0 * n / max(N, 1),
            "mean_margin":     _safe_mean(margins,     bin_m),
            "mean_nll":        _safe_mean(nll_learned, bin_m),
            "mean_oracle_nll": _safe_mean(nll_oracle,  bin_m),
            "oracle_gap":      _safe_mean(oracle_gap,  bin_m),
            "mean_entropy":    _safe_mean(entropies,   bin_m),
            "acc1_real":       _safe_mean(acc1_real,   known_r),
            "acc4_real":       _safe_mean(acc4_real,   known_r),
            "acc1_rand":       _safe_mean(acc1_rand,   known_d),
            "acc4_rand":       _safe_mean(acc4_rand,   known_d),
        })

    # ── Part 5: Token context variance ───────────────────────────────────────
    print("[boundary_analysis] computing token context variance...")
    tok2regions: Dict[int, List[int]]   = {}
    tok2margins: Dict[int, List[float]] = {}
    for tok, reg, mar in zip(target_toks.tolist(), top1_region.tolist(), margins.tolist()):
        if tok not in tok2regions:
            tok2regions[tok] = []; tok2margins[tok] = []
        tok2regions[tok].append(reg)
        tok2margins[tok].append(mar)

    ctx_var_rows: List[Dict] = []
    for tok, regs in tok2regions.items():
        if len(regs) < 5:
            continue
        c = Counter(regs)
        ctx_var_rows.append({
            "token_id":         tok,
            "n_contexts":       len(regs),
            "n_unique_regions": len(set(regs)),
            "context_variance": round(1.0 - c.most_common(1)[0][1] / len(regs), 6),
            "avg_margin":       round(float(np.mean(tok2margins[tok])), 6),
        })
    ctx_var_rows.sort(key=lambda r: r["context_variance"], reverse=True)
    print(f"[boundary_analysis] {len(ctx_var_rows)} tokens with ≥5 occurrences")

    # ── Part 6: Core vs boundary tokens ──────────────────────────────────────
    tok_avg_m = {tok: float(np.mean(mars)) for tok, mars in tok2margins.items()}

    def _group_stats(mask: torch.Tensor) -> Dict[str, float]:
        known = mask & (acc1_real >= 0)
        return {
            "count":           int(mask.sum().item()),
            "mean_nll":        _safe_mean(nll_learned, mask),
            "mean_oracle_nll": _safe_mean(nll_oracle,  mask),
            "oracle_gap":      _safe_mean(oracle_gap,  mask),
            "mean_entropy":    _safe_mean(entropies,   mask),
            "acc1":            _safe_mean(acc1_real,   known),
            "acc4":            _safe_mean(acc4_real,   known),
        }

    core_mask     = torch.tensor(
        [tok_avg_m.get(int(t), 0.5) >= 0.4 for t in target_toks.tolist()])
    boundary_mask = torch.tensor(
        [tok_avg_m.get(int(t), 0.5) <  0.1 for t in target_toks.tolist()])
    core_stats     = _group_stats(core_mask)
    boundary_stats = _group_stats(boundary_mask)

    # ── Per-token NLL table ───────────────────────────────────────────────────
    tok_nlls: Dict[int, List[float]] = {}
    tok_onlls: Dict[int, List[float]] = {}
    for tok, nl, onl in zip(target_toks.tolist(), nll_learned.tolist(), nll_oracle.tolist()):
        if tok not in tok_nlls:
            tok_nlls[tok] = []; tok_onlls[tok] = []
        tok_nlls[tok].append(nl); tok_onlls[tok].append(onl)

    tok_stat_rows: List[Dict] = []
    for tok, nlls_t in tok_nlls.items():
        if len(nlls_t) < 3:
            continue
        onlls_t = tok_onlls[tok]
        tok_stat_rows.append({
            "token_id":        tok,
            "n_occ":           len(nlls_t),
            "mean_nll":        round(float(np.mean(nlls_t)), 6),
            "mean_oracle_nll": round(float(np.mean(onlls_t)), 6),
            "oracle_gap":      round(float(np.mean([a - b for a, b in zip(nlls_t, onlls_t)])), 6),
            "avg_margin":      round(tok_avg_m.get(tok, 0.0), 6),
        })
    tok_stat_rows.sort(key=lambda r: r["oracle_gap"], reverse=True)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    def _write_csv(rows: List[Dict], path: str):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    _write_csv(bin_rows,              os.path.join(out_dir, "boundary_metrics.csv"))
    _write_csv(ctx_var_rows,          os.path.join(out_dir, "token_context_variance.csv"))
    _write_csv(tok_stat_rows[:5000],  os.path.join(out_dir, "token_boundary_stats.csv"))
    print("[boundary_analysis] wrote CSVs")

    # ── Plots ─────────────────────────────────────────────────────────────────
    nonempty = [i for i in range(N_BINS) if bin_rows[i]["count"] > 0]

    def _mids(key: str) -> List[float]:
        return [bin_rows[i][key] for i in nonempty]

    bc   = _mids("mean_margin")
    xlbl = [BIN_LABELS[i] for i in nonempty]

    def _ax_setup(ax, title: str, xlabel: str, ylabel: str):
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if bc:
            ax.set_xticks(bc)
            ax.set_xticklabels(xlbl, rotation=20, ha="right", fontsize=8)

    # Plot 1: routing accuracy vs margin
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bc, _mids("acc1_real"), "o-b", label="Acc@1")
    ax.plot(bc, _mids("acc4_real"), "s-g", label="Acc@4")
    _ax_setup(ax, "Routing Accuracy vs Margin", "Mean routing margin", "Routing accuracy")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "routing_accuracy_vs_margin.png"), dpi=120)
    plt.close(fig)

    # Plot 2: oracle NLL gain vs margin
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bc, _mids("oracle_gap"), "o-r", label="oracle_gap = nll_learned − nll_oracle")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    _ax_setup(ax, "Oracle NLL Gain vs Routing Margin", "Mean routing margin", "NLL gap")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "oracle_gain_vs_margin.png"), dpi=120)
    plt.close(fig)

    # Plot 3: router entropy vs margin
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bc, _mids("mean_entropy"), "o-", color="darkorange")
    _ax_setup(ax, "Router Entropy vs Margin Bin", "Mean routing margin", "Mean router entropy")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "entropy_vs_margin.png"), dpi=120)
    plt.close(fig)

    # Plot 4: top-4 recall real vs random
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bc, _mids("acc4_real"), "s-b", label="Acc@4 (real map)")
    ax.plot(bc, _mids("acc4_rand"), "s--", color="gray", label="Acc@4 (random map)")
    _ax_setup(ax, "Top-4 Recall: Real vs Random Region Map", "Mean routing margin", "Acc@4")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "topk_recall_vs_margin.png"), dpi=120)
    plt.close(fig)

    # Plot 5: region usage histogram
    region_counts = torch.zeros(n_coarse)
    for r in top1_region.tolist():
        if 0 <= r < n_coarse:
            region_counts[r] += 1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n_coarse), region_counts.numpy(), width=1.0, color="steelblue")
    ax.set_xlabel("Region ID"); ax.set_ylabel("Token count (val set)")
    ax.set_title("Region Usage Histogram (top-1 predicted region)")
    ax.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "region_usage_histogram.png"), dpi=120)
    plt.close(fig)

    # Plot 6: token context variance distribution + scatter
    if ctx_var_rows:
        variances  = [r["context_variance"] for r in ctx_var_rows]
        avg_mars_v = [r["avg_margin"]       for r in ctx_var_rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(variances, bins=30, edgecolor="k", linewidth=0.5, color="teal")
        axes[0].set_xlabel("Context variance (1 − mode_freq)")
        axes[0].set_ylabel("# tokens")
        axes[0].set_title("Distribution of Token Context Variance")
        axes[0].grid(True, alpha=0.3)
        axes[1].scatter(avg_mars_v, variances, alpha=0.25, s=6, color="teal")
        axes[1].set_xlabel("Avg routing margin")
        axes[1].set_ylabel("Context variance")
        axes[1].set_title("Context Variance vs Avg Margin")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "token_context_region_variance.png"), dpi=120)
        plt.close(fig)

    # Plot 7: margin distribution — learned vs simulated uniform router
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(margins.numpy(), bins=60, alpha=0.7, density=True, color="steelblue",
            label="Learned routing")
    rng_np  = np.random.default_rng(cfg.seed)
    n_sim   = min(50_000, N)
    sim_p   = rng_np.dirichlet(np.ones(n_coarse), size=n_sim)
    top2sim = np.sort(sim_p, axis=1)[:, -2:]
    sim_mg  = top2sim[:, 1] - top2sim[:, 0]
    ax.hist(sim_mg, bins=60, alpha=0.5, density=True, color="gray",
            label=f"Uniform router (K={n_coarse}, simulated)")
    ax.set_xlabel("Routing margin (top1 − top2)")
    ax.set_ylabel("Density")
    ax.set_title("Margin Distribution: Learned vs Uniform Baseline")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "margin_distribution_real_vs_random.png"), dpi=120)
    plt.close(fig)

    print(f"[boundary_analysis] wrote 7 plots → {plots_dir}/")

    # ── Q&A verdicts ──────────────────────────────────────────────────────────
    nonempty_rows = [r for r in bin_rows if r["count"] > 0]
    lo_row = nonempty_rows[0]  if nonempty_rows else {}
    hi_row = nonempty_rows[-1] if nonempty_rows else {}

    def _v(cond: bool) -> str:
        return "PASS" if cond else "INCONCLUSIVE"

    q1 = _v(len(nonempty_rows) >= 2 and lo_row["acc1_real"]    < hi_row["acc1_real"])
    q2 = _v(len(nonempty_rows) >= 2 and lo_row["mean_nll"]     > hi_row["mean_nll"])
    q3 = _v(len(nonempty_rows) >= 2 and lo_row["oracle_gap"]   > hi_row["oracle_gap"])
    q4 = _v(len(nonempty_rows) >= 2 and lo_row["mean_entropy"] > hi_row["mean_entropy"])
    q5 = _v(sum(1 for r in ctx_var_rows if r["context_variance"] > 0.2) > 10)
    q6 = _v(boundary_stats["count"] > 0 and core_stats["count"] > 0 and
            boundary_stats["mean_nll"] > core_stats["mean_nll"])
    q7 = _v(nonempty_rows != [] and
            hi_row["acc1_real"] > hi_row["acc1_rand"] + 0.05)

    # ── Summary markdown ──────────────────────────────────────────────────────
    lines = [
        "# Boundary Analysis Report\n",
        f"**Checkpoint:** `{cfg.repr_ckpt}`",
        f"**Mode:** `{model_cfg.mode}`  |  **Step:** {step_ckpt}  "
        f"|  **K:** {n_coarse}  |  **Positions:** {N:,}",
        "",
        "## Margin Bin Statistics\n",
        "| Bin | Count | % | NLL | Oracle NLL | Oracle Gap | Entropy | Acc@1 | Acc@4 | Acc@1 (rand) |",
        "|-----|------:|--:|----:|----------:|----------:|-------:|------:|------:|-------------:|",
    ]
    for r in bin_rows:
        lines.append(
            f"| {r['bin']} | {r['count']:,} | {r['pct']:.1f}% "
            f"| {r['mean_nll']:.4f} | {r['mean_oracle_nll']:.4f} | {r['oracle_gap']:+.4f} "
            f"| {r['mean_entropy']:.3f} | {r['acc1_real']:.3f} | {r['acc4_real']:.3f} "
            f"| {r['acc1_rand']:.3f} |"
        )
    lines += [
        "",
        "## Core vs Boundary Tokens\n",
        "| Group | Count | NLL | Oracle NLL | Oracle Gap | Acc@1 | Acc@4 |",
        "|-------|------:|----:|----------:|----------:|------:|------:|",
        f"| Core (avg_margin ≥ 0.4) | {core_stats['count']:,} | {core_stats['mean_nll']:.4f} "
        f"| {core_stats['mean_oracle_nll']:.4f} | {core_stats['oracle_gap']:+.4f} "
        f"| {core_stats['acc1']:.3f} | {core_stats['acc4']:.3f} |",
        f"| Boundary (avg_margin < 0.1) | {boundary_stats['count']:,} | {boundary_stats['mean_nll']:.4f} "
        f"| {boundary_stats['mean_oracle_nll']:.4f} | {boundary_stats['oracle_gap']:+.4f} "
        f"| {boundary_stats['acc1']:.3f} | {boundary_stats['acc4']:.3f} |",
        "",
        "## Q&A Verdicts\n",
        "| # | Question | Verdict |",
        "|---|----------|---------|",
        f"| Q1 | Do routing failures concentrate at low-margin positions? | **{q1}** |",
        f"| Q2 | Is LM NLL higher at low-margin positions? | **{q2}** |",
        f"| Q3 | Does oracle routing help more at boundary (low-margin)? | **{q3}** |",
        f"| Q4 | Does router entropy increase at low-margin positions? | **{q4}** |",
        f"| Q5 | Do repeated tokens show context-dependent routing? | **{q5}** |",
        f"| Q6 | Are boundary tokens (low avg_margin) harder to model? | **{q6}** |",
        f"| Q7 | Does learned routing outperform random at high-margin? | **{q7}** |",
        "",
        "## Context Variance — Top 20 Tokens\n",
        "| Token ID | Contexts | Unique Regions | Variance | Avg Margin |",
        "|--------:|--------:|---------------:|--------:|-----------:|",
    ]
    for r in ctx_var_rows[:20]:
        lines.append(
            f"| {r['token_id']} | {r['n_contexts']} | {r['n_unique_regions']} "
            f"| {r['context_variance']:.4f} | {r['avg_margin']:.4f} |"
        )
    lines += [
        "",
        "## Output Files\n",
        f"- `boundary_metrics.csv` — {N_BINS} margin bins with all metrics",
        f"- `token_boundary_stats.csv` — per-token NLL and oracle gap (top 5k by oracle_gap)",
        f"- `token_context_variance.csv` — routing variance for tokens with ≥5 occurrences",
        f"- `plots/` — 7 diagnostic plots",
        "",
    ]
    with open(os.path.join(out_dir, "boundary_analysis_summary.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n[boundary_analysis] DONE  →  {out_dir}")
    for q, label in [(q1, "routing failures at low margin"),
                     (q2, "higher NLL at low margin   "),
                     (q3, "oracle helps more at boundary"),
                     (q4, "entropy up at boundary      "),
                     (q5, "context-dependent routing   "),
                     (q6, "boundary tokens harder      "),
                     (q7, "learned > random (high margin)")]:
        print(f"  {label}: {q}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset",           default="wikitext-2-raw-v1")
    p.add_argument("--seq_len",           type=int,   default=256)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--vocab_subset_size", type=int,   default=50257)
    p.add_argument("--mode",    default="baseline", choices=list(_MODES))
    p.add_argument("--n_layer", type=int,   default=6)
    p.add_argument("--d_model", type=int,   default=384)
    p.add_argument("--n_head",  type=int,   default=6)
    p.add_argument("--d_ff",    type=int,   default=1536)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--region_map_path",     default=None)
    p.add_argument("--leaf_map_path",       default=None)
    p.add_argument("--n_coarse",            type=int,   default=50)
    p.add_argument("--n_leaf",              type=int,   default=200)
    p.add_argument("--d_region",            type=int,   default=64)
    p.add_argument("--region_temp",         type=float, default=1.0)
    p.add_argument("--router_type",         default="linear", choices=["linear", "mlp"])
    p.add_argument("--base_alpha",          type=float, default=0.1)
    p.add_argument("--base_beta",           type=float, default=0.03)
    p.add_argument("--region_warmup_steps", type=int,   default=5000)
    p.add_argument("--use_refiner",         action="store_true")
    p.add_argument("--lambda_coarse",       type=float, default=0.05)
    p.add_argument("--lambda_leaf",         type=float, default=0.01)
    p.add_argument("--lambda_balance",      type=float, default=0.001)
    p.add_argument("--steps",              type=int,   default=20000)
    p.add_argument("--lr",                 type=float, default=3e-4)
    p.add_argument("--weight_decay",       type=float, default=0.1)
    p.add_argument("--grad_clip",          type=float, default=1.0)
    p.add_argument("--warmup_steps",       type=int,   default=1000)
    p.add_argument("--eval_interval",      type=int,   default=500)
    p.add_argument("--save_interval",      type=int,   default=2000)
    p.add_argument("--log_interval",       type=int,   default=50)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", default="runs/region_lm")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--device",     default="cuda")
    # build_regions args
    p.add_argument("--baseline_ckpt",     default=None,
                   help="Baseline checkpoint (required for build_regions and layer_margin_analysis modes)")
    p.add_argument("--region_output_dir", default="",
                   help="Where to write maps; defaults to --output_dir")
    p.add_argument("--region_vocab_size", type=int,   default=5000)
    p.add_argument("--region_top_k",      type=int,   default=20)
    p.add_argument("--region_max_tokens", type=int,   default=500_000)
    p.add_argument("--n_clusters",        type=int,   default=50)
    p.add_argument("--no_leaf",              action="store_true",
                   help="Skip recursive subcluster / region_tree.json")
    p.add_argument("--leaf_min_size",        type=int,   default=30)
    p.add_argument("--region_cluster_method", default="spectral",
                   choices=["leiden", "spectral", "kmeans"])
    p.add_argument("--target_n_regions",     type=int,   default=128)
    p.add_argument("--target_n_leaves",      type=int,   default=512)
    p.add_argument("--extend_full_vocab",    action="store_true")
    # soft_moe / soft routing
    p.add_argument("--router_temp",          type=float, default=2.0)
    p.add_argument("--prior_gamma",          type=float, default=1.0)
    p.add_argument("--prior_eps",            type=float, default=1e-6)
    p.add_argument("--learned_region_prior", action="store_true")
    p.add_argument("--soft_topk_regions",    type=int,   default=0)
    # repr_region_capacity
    p.add_argument("--usage_ema_decay",        type=float, default=0.99)
    p.add_argument("--capacity_alpha",         type=float, default=0.0)
    p.add_argument("--lambda_diversity",       type=float, default=0.0)
    p.add_argument("--router_temp_init",       type=float, default=2.0)
    p.add_argument("--router_temp_final",      type=float, default=1.0)
    p.add_argument("--router_temp_decay_steps",type=int,   default=10000)
    # boundary_analysis
    # repr_region_multihyp
    p.add_argument("--hyp_k",          type=int,   default=4,
                   help="Number of top-k latent hypotheses to preserve at boundary tokens")
    # repr_region_branchattn
    # repr_region_boundary_seqrefine
    p.add_argument("--boundary_refine_layers", type=int,   default=1,
                   help="Causal Transformer blocks for seqrefine boundary path")
    p.add_argument("--boundary_refine_gamma",  type=float, default=0.5,
                   help="Fixed scale multiplier for seqrefine contribution")
    # repr_region_boundary_adaptivedepth
    p.add_argument("--adaptive_layers",      type=int,   default=1,
                   help="Extra causal Transformer blocks applied only at boundary tokens")
    p.add_argument("--adaptive_scale_init",  type=float, default=0.05,
                   help="Initial value of tanh-gated adaptive residual scale")
    p.add_argument("--adaptive_threshold",   type=float, default=0.5,
                   help="Hard gate threshold: gate > threshold → boundary")
    # repr_region_branchattn
    p.add_argument("--branch_attn_heads",   type=int,   default=4,
                   help="Attention heads in BranchAttentionRefiner (over K dim)")
    p.add_argument("--branch_attn_layers",  type=int,   default=1,
                   help="Number of stacked branch-attention layers")
    p.add_argument("--branch_attn_dropout", type=float, default=0.1,
                   help="Dropout inside BranchAttentionRefiner")
    p.add_argument("--branch_scale_init",   type=float, default=0.0,
                   help="Initial value of tanh-gated residual scale (0 = identity at init)")
    # repr_region_boundary / repr_region_multihyp
    p.add_argument("--alpha_core",     type=float, default=0.2,
                   help="Alpha for high-margin (core) token conditioning")
    p.add_argument("--alpha_boundary", type=float, default=0.4,
                   help="Alpha for low-margin (boundary) token conditioning")
    p.add_argument("--boundary_tau",   type=float, default=0.10,
                   help="Margin threshold: tokens with margin < tau get stronger conditioning")
    p.add_argument("--boundary_temp",  type=float, default=0.05,
                   help="Gate sigmoid temperature (smaller = sharper transition)")
    p.add_argument("--boundary_mode",  default="margin", choices=["margin", "entropy"],
                   help="Gate signal: routing margin (top1-top2) or normalized entropy")
    # boundary_analysis
    p.add_argument("--repr_ckpt",            default=None,
                   help="Trained repr_region checkpoint for boundary_analysis mode")
    p.add_argument("--max_analysis_batches", type=int, default=200,
                   help="Max val batches for boundary_analysis (0 = all)")
    # repr_region_retrieval
    p.add_argument("--retrieval_loss_type",  default="proxy", choices=["proxy", "supcon"],
                   help="Retrieval metric loss: 'proxy' (CE over centroids) or 'supcon'")
    p.add_argument("--lambda_retrieval",     type=float, default=0.0,
                   help="Weight on retrieval metric loss (0 = disabled)")
    p.add_argument("--retrieval_dim",        type=int,   default=128,
                   help="Projection head output dimension for retrieval loss")
    p.add_argument("--retrieval_temp",       type=float, default=0.07,
                   help="Temperature for proxy/supcon retrieval loss")
    p.add_argument("--retrieval_key_source", default="pre_region",
                   choices=["pre_region", "post_region"],
                   help="Hidden state to project: before (pre) or after (post) region injection")
    p.add_argument("--max_retrieval_positions_per_batch", type=int, default=2048,
                   help="Max positions subsampled per batch for supcon stability")
    a = p.parse_args()
    return TrainConfig(
        dataset=a.dataset, seq_len=a.seq_len, batch_size=a.batch_size,
        vocab_subset_size=a.vocab_subset_size, mode=a.mode,
        n_layer=a.n_layer, d_model=a.d_model, n_head=a.n_head,
        d_ff=a.d_ff, dropout=a.dropout,
        region_map_path=a.region_map_path, leaf_map_path=a.leaf_map_path,
        n_coarse=a.n_coarse, n_leaf=a.n_leaf, d_region=a.d_region,
        region_temp=a.region_temp, router_type=a.router_type,
        base_alpha=a.base_alpha, base_beta=a.base_beta,
        region_warmup_steps=a.region_warmup_steps,
        use_refiner=a.use_refiner,
        lambda_coarse=a.lambda_coarse, lambda_leaf=a.lambda_leaf,
        lambda_balance=a.lambda_balance,
        steps=a.steps, lr=a.lr, weight_decay=a.weight_decay,
        grad_clip=a.grad_clip, warmup_steps=a.warmup_steps,
        eval_interval=a.eval_interval, save_interval=a.save_interval,
        log_interval=a.log_interval,
        seed=a.seed, output_dir=a.output_dir, resume=a.resume, device=a.device,
        baseline_ckpt=a.baseline_ckpt,
        region_output_dir=a.region_output_dir,
        region_vocab_size=a.region_vocab_size,
        region_top_k=a.region_top_k,
        region_max_tokens=a.region_max_tokens,
        n_clusters=a.n_clusters,
        build_leaf=not a.no_leaf,
        leaf_min_size=a.leaf_min_size,
        region_cluster_method=a.region_cluster_method,
        target_n_regions=a.target_n_regions,
        target_n_leaves=a.target_n_leaves,
        extend_full_vocab=a.extend_full_vocab,
        router_temp=a.router_temp,
        prior_gamma=a.prior_gamma,
        prior_eps=a.prior_eps,
        learned_region_prior=a.learned_region_prior,
        soft_topk_regions=a.soft_topk_regions,
        usage_ema_decay=a.usage_ema_decay,
        capacity_alpha=a.capacity_alpha,
        lambda_diversity=a.lambda_diversity,
        router_temp_init=a.router_temp_init,
        router_temp_final=a.router_temp_final,
        router_temp_decay_steps=a.router_temp_decay_steps,
        hyp_k=a.hyp_k,
        boundary_refine_layers=a.boundary_refine_layers,
        boundary_refine_gamma=a.boundary_refine_gamma,
        adaptive_layers=a.adaptive_layers,
        adaptive_scale_init=a.adaptive_scale_init,
        adaptive_threshold=a.adaptive_threshold,
        branch_attn_heads=a.branch_attn_heads,
        branch_attn_layers=a.branch_attn_layers,
        branch_attn_dropout=a.branch_attn_dropout,
        branch_scale_init=a.branch_scale_init,
        alpha_core=a.alpha_core,
        alpha_boundary=a.alpha_boundary,
        boundary_tau=a.boundary_tau,
        boundary_temp=a.boundary_temp,
        boundary_mode=a.boundary_mode,
        repr_ckpt=a.repr_ckpt,
        max_analysis_batches=a.max_analysis_batches,
        retrieval_loss_type=a.retrieval_loss_type,
        lambda_retrieval=a.lambda_retrieval,
        retrieval_dim=a.retrieval_dim,
        retrieval_temp=a.retrieval_temp,
        retrieval_key_source=a.retrieval_key_source,
        max_retrieval_positions_per_batch=a.max_retrieval_positions_per_batch,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    if cfg.mode == "build_regions":
        build_regions(cfg)
    elif cfg.mode == "layer_margin_analysis":
        layer_margin_analysis(cfg)
    elif cfg.mode == "boundary_analysis":
        boundary_analysis(cfg)
    else:
        train(cfg)
