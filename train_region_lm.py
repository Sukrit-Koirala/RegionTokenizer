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
          "build_regions")


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

    lm_vals: List[float] = []
    aux_c:   List[float] = []
    aux_l:   List[float] = []
    c_acc1:  List[float] = []
    c_acc4:  List[float] = []
    l_acc1:  List[float] = []
    l_acc8:  List[float] = []
    l_acc32: List[float] = []
    c_ent:   List[float] = []
    l_ent:   List[float] = []

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

    def avg(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    lm = avg(lm_vals)
    out: Dict[str, float] = {
        "val_lm_loss":     lm,
        "val_ppl":         math.exp(min(lm, 20.0)),
        "val_aux_coarse":  avg(aux_c),
        "val_aux_leaf":    avg(aux_l),
        "val_coarse_acc1": avg(c_acc1),
        "val_coarse_acc4": avg(c_acc4),
        "val_leaf_acc1":   avg(l_acc1),
        "val_leaf_acc8":   avg(l_acc8),
        "val_leaf_acc32":  avg(l_acc32),
        "val_coarse_ent":  avg(c_ent),
        "val_leaf_ent":    avg(l_ent),
        "current_alpha":   getattr(model, "current_alpha", 0.0),
        "current_beta":    getattr(model, "current_beta",  0.0),
    }
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
        if cfg.mode == "random_control":
            raw_coarse = permute_map(raw_coarse, cfg.n_coarse, seed=cfg.seed)
        coarse_map = raw_coarse.to(device)
        cov = (raw_coarse >= 0).float().mean().item()
        tag = " (PERMUTED)" if cfg.mode == "random_control" else ""
        print(f"[region] coarse: {n_coarse_actual} regions  {cov:.1%} coverage{tag}")

    if cfg.mode in ("coarse_leaf", "random_control", "oracle") and cfg.leaf_map_path:
        raw_leaf, n_leaf_actual = load_leaf_map(cfg.leaf_map_path, vocab_size)
        cfg.n_leaf = n_leaf_actual
        if cfg.mode == "random_control":
            raw_leaf = permute_map(raw_leaf, cfg.n_leaf, seed=cfg.seed + 1)
        leaf_map = raw_leaf.to(device)
        cov = (raw_leaf >= 0).float().mean().item()
        tag = " (PERMUTED)" if cfg.mode == "random_control" else ""
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
        "train_aux_coarse", "train_aux_leaf", "train_aux_balance",
        "val_aux_coarse", "val_aux_leaf",
        "val_coarse_acc1", "val_coarse_acc4",
        "val_leaf_acc1", "val_leaf_acc8", "val_leaf_acc32",
        "val_coarse_ent", "val_leaf_ent",
        "current_alpha", "current_beta",
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
        if isinstance(model, RegionConditionedTransformerLM):
            if cfg.region_warmup_steps <= 0:
                scale = 1.0
            else:
                scale = min(1.0, step / cfg.region_warmup_steps)
            model.set_region_scales(cfg.base_alpha * scale, cfg.base_beta * scale)

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
            print(
                f"step {step:6d}  lm={ema_lm:.4f}  total={ema_total:.4f}"
                f"  lr={lr:.2e}  α={alpha:.3f}  β={beta:.4f}  tok/s={tps:.0f}"
            )

        if step % cfg.eval_interval == 0 or step == cfg.steps:
            val = evaluate(model, val_loader, cfg, device,
                           coarse_map=coarse_map, leaf_map=leaf_map)
            tps = tok_seen / (time.time() - t0)
            row = {
                "step":              step,
                "train_lm_loss":     ema_lm,
                "train_total_loss":  ema_total,
                "train_aux_coarse":  losses.get("coarse",         torch.zeros(1)).item(),
                "train_aux_leaf":    losses.get("leaf",           torch.zeros(1)).item(),
                "train_aux_balance": losses.get("balance_coarse", torch.zeros(1)).item(),
                "lr":                lr,
                "tokens_per_sec":    int(tps),
                "n_params":          n_params,
                **val,
            }
            _csv_writer.writerow(row)
            _csv_file.flush()
            print(
                f"  [val] val_lm={val['val_lm_loss']:.4f}  ppl={val['val_ppl']:.2f}"
                f"  coarse_acc@1={val['val_coarse_acc1']:.3f}"
                f"  leaf_acc@8={val['val_leaf_acc8']:.3f}"
                f"  α={val['current_alpha']:.3f}"
            )
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
    n_graph  = cfg.region_vocab_size
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

    # ── Clustering helper ─────────────────────────────────────────────────────
    def _cluster(W_sub: np.ndarray, n_clus: int, seed: int) -> np.ndarray:
        n = W_sub.shape[0]
        if n <= n_clus:
            return np.arange(n, dtype=np.int32)

        # Try Leiden
        try:
            import igraph as ig
            import leidenalg
            W_coo = sp.csr_matrix(W_sub).tocoo()
            mask  = (W_coo.row < W_coo.col) & (W_coo.data > 0)
            if mask.sum() > 0:
                G = ig.Graph(n=n, directed=False)
                G.add_edges(list(zip(W_coo.row[mask].tolist(),
                                     W_coo.col[mask].tolist())))
                G.es["weight"] = W_coo.data[mask].tolist()
                part = leidenalg.find_partition(
                    G, leidenalg.RBConfigurationVertexPartition,
                    weights="weight", seed=seed,
                )
                return np.array(part.membership, dtype=np.int32)
        except Exception:
            pass

        # Fallback: SpectralClustering
        try:
            from sklearn.cluster import SpectralClustering
            sc = SpectralClustering(
                n_clusters=n_clus, affinity="precomputed",
                random_state=seed, n_jobs=-1, assign_labels="cluster_qr",
            )
            return sc.fit_predict(W_sub + 1e-10 * np.eye(n)).astype(np.int32)
        except Exception:
            pass

        # Last resort: random
        return np.random.RandomState(seed).randint(0, n_clus, n).astype(np.int32)

    # ── Coarse clustering ─────────────────────────────────────────────────────
    coarse_labels = _cluster(W_sym, cfg.n_clusters, cfg.seed)
    n_coarse      = int(coarse_labels.max()) + 1
    print(f"[build_regions] coarse: {n_coarse} regions")

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
        n_sub_c = max(2, min(8, len(sub_indices) // cfg.leaf_min_size))
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
                   help="Baseline checkpoint to sweep (required for build_regions mode)")
    p.add_argument("--region_output_dir", default="",
                   help="Where to write maps; defaults to --output_dir")
    p.add_argument("--region_vocab_size", type=int,   default=5000)
    p.add_argument("--region_top_k",      type=int,   default=20)
    p.add_argument("--region_max_tokens", type=int,   default=500_000)
    p.add_argument("--n_clusters",        type=int,   default=50)
    p.add_argument("--no_leaf",           action="store_true",
                   help="Skip recursive subcluster / region_tree.json")
    p.add_argument("--leaf_min_size",     type=int,   default=30)
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
    )


if __name__ == "__main__":
    cfg = _parse_args()
    if cfg.mode == "build_regions":
        build_regions(cfg)
    else:
        train(cfg)
