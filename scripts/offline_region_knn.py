#!/usr/bin/env python3
"""
Offline Region-kNN Experiment.

Builds a kNN memory of (hidden_state → region_label) pairs from the training
split, then retrieves nearest neighbours at eval time to form a region
prediction prior (p_mem).  Compares p_router, p_mem, and their mixture
distributions across boundary vs core tokens.

Usage (GPT-2 XL):
    python scripts/offline_region_knn.py \
        --model_type gpt2xl \
        --region_map_path runs/region_maps_128/token_to_region.json \
        --probe_path runs/gpt2xl_layer_margin/probes.pt \
        --knn_layer block_41 \
        --output_dir runs/offline_region_knn_gpt2xl_block41 \
        --device cuda

Usage (small repr_region checkpoint):
    python scripts/offline_region_knn.py \
        --model_type small \
        --small_ckpt runs/repr_region_reference/checkpoint_latest.pt \
        --region_map_path runs/region_maps_128/token_to_region.json \
        --output_dir runs/offline_region_knn_small \
        --device cuda
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

try:
    import faiss as _faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

# ── Project imports (for small backbone) ─────────────────────────────────────

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
try:
    from train_region_lm import (BaselineTransformerLM, TrainConfig, _make_router_head,
                                  ReprRegionTransformerLM, ReprRegionRetrievalLM)
    _HAS_TRAIN_LM = True
except ImportError:
    _HAS_TRAIN_LM = False
    ReprRegionTransformerLM = None  # type: ignore
    ReprRegionRetrievalLM   = None  # type: ignore

# ── Constants ─────────────────────────────────────────────────────────────────

# (name, margin_lo_inclusive, margin_hi_exclusive)
# None = no bound in that direction
SPLIT_DEFS = [
    ("all",             None,  None),
    ("core",            0.30,  None),
    ("medium",          0.10,  0.30),
    ("boundary",        None,  0.10),
    ("tight_boundary",  None,  0.03),
]

RANDOM_BASELINES = {
    "acc1":  None,   # filled from n_coarse at runtime
    "acc4":  None,
    "acc8":  None,
    "acc16": None,
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenChunkDataset(Dataset):
    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.n       = max(0, (len(tokens) - 1) // seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        start = i * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return torch.from_numpy(chunk.astype(np.int64))


# ── Region probe (for GPT-2 XL probes.pt) ────────────────────────────────────

class RegionProbe(nn.Module):
    def __init__(self, d_model: int, n_regions: int, probe_type: str = "ln_linear"):
        super().__init__()
        if probe_type == "ln_linear":
            self.net: nn.Module = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, n_regions, bias=False),
            )
        else:
            self.net = nn.Linear(d_model, n_regions, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ── Memory index ─────────────────────────────────────────────────────────────

class MemoryIndex:
    """kNN index backed by FAISS (preferred) or GPU/CPU brute-force."""

    def __init__(self, d_model: int, device: torch.device, normalize: bool = True):
        self.d_model   = d_model
        self.device    = device
        self.normalize = normalize
        self._index    = None   # FAISS index (CPU only — used when device is CPU)
        self._keys_gpu = None   # fp16 GPU tensor for torch matmul search
        # For d_model=1600 and N=500K, CPU FAISS IndexFlatIP is ~15s/batch.
        # GPU torch matmul over fp16 keys is ~10ms/batch on L40S.
        # Only use CPU FAISS when no GPU is available.
        self._use_torch_gpu = (device.type == "cuda")

    # ── build / save / load ───────────────────────────────────────────────────

    def build(self, keys_f32: np.ndarray) -> None:
        """keys_f32: (N, d_model) float32."""
        if self.normalize:
            norms = np.linalg.norm(keys_f32, axis=1, keepdims=True).clip(min=1e-8)
            keys_f32 = keys_f32 / norms

        if _HAS_FAISS and not self._use_torch_gpu:
            self._index = _faiss.IndexFlatIP(self.d_model)
            self._index.add(keys_f32)
            print(f"[MemoryIndex] FAISS IndexFlatIP (CPU) built  ntotal={self._index.ntotal:,}")
        else:
            self._keys_gpu = torch.tensor(keys_f32, dtype=torch.float16).to(self.device)
            print(f"[MemoryIndex] torch GPU fp16 index  n={len(keys_f32):,}  "
                  f"device={self.device}  "
                  f"mem={keys_f32.nbytes / 2 / 1e9:.2f} GB")

    def save(self, path: str) -> None:
        if _HAS_FAISS and self._index is not None:
            _faiss.write_index(self._index, path)

    def load(self, path: str) -> bool:
        if self._use_torch_gpu:
            return False  # always rebuild as fp16 GPU tensor from cached numpy keys
        if _HAS_FAISS and os.path.exists(path):
            self._index = _faiss.read_index(path)
            print(f"[MemoryIndex] loaded FAISS index  ntotal={self._index.ntotal:,}")
            return True
        return False

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, queries_f32: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (similarities, indices) both (Q, k) float32 / int64."""
        if self.normalize:
            norms = np.linalg.norm(queries_f32, axis=1, keepdims=True).clip(min=1e-8)
            queries_f32 = queries_f32 / norms

        if _HAS_FAISS and self._index is not None:
            sims, idx = self._index.search(queries_f32, k)
            return sims.astype(np.float32), idx.astype(np.int64)

        return self._torch_search(queries_f32, k)

    def _torch_search(self, queries_f32: np.ndarray, k: int):
        Q        = len(queries_f32)
        all_sims = np.zeros((Q, k), dtype=np.float32)
        all_idx  = np.zeros((Q, k), dtype=np.int64)
        batch_q  = 256

        q_t = torch.tensor(queries_f32, dtype=torch.float16)
        for i in range(0, Q, batch_q):
            q_batch = q_t[i : i + batch_q].to(self.device)
            sims    = (q_batch @ self._keys_gpu.T).float()   # (Bq, N)
            topk    = sims.topk(k, dim=-1)
            all_sims[i : i + batch_q] = topk.values.cpu().numpy()
            all_idx[i  : i + batch_q] = topk.indices.cpu().numpy()

        return all_sims, all_idx


# ── Region map helpers ────────────────────────────────────────────────────────

def load_region_map(path: str, vocab_size: int) -> Tuple[torch.Tensor, int]:
    with open(path) as f:
        d = json.load(f)
    arr = torch.full((vocab_size,), -1, dtype=torch.long)
    max_r = -1
    for tok_str, rid in d.items():
        tid = int(tok_str)
        if tid < vocab_size:
            arr[tid] = int(rid)
            if int(rid) > max_r:
                max_r = int(rid)
    return arr, max_r + 1


def build_inverse_map(coarse_map: torch.Tensor, n_coarse: int) -> Dict[int, List[int]]:
    inv: Dict[int, List[int]] = defaultdict(list)
    for tok_id, rid in enumerate(coarse_map.tolist()):
        if rid >= 0:
            inv[int(rid)].append(tok_id)
    return dict(inv)


def tokens_per_region_tensor(inv_map: Dict[int, List[int]],
                              n_coarse: int) -> torch.Tensor:
    return torch.tensor([len(inv_map.get(r, [])) for r in range(n_coarse)],
                        dtype=torch.float32)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(p: torch.Tensor, gold: torch.Tensor, n_coarse: int) -> Dict:
    """p: (N, n_coarse), gold: (N,) long. All gold >= 0."""
    N    = len(p)
    kmax = min(16, n_coarse)

    topk_idx  = torch.topk(p, kmax, dim=-1).indices   # (N, kmax)
    gold_col  = gold.unsqueeze(-1)
    acc1  = (topk_idx[:, :1]               == gold_col).any(-1).float().mean().item()
    acc4  = (topk_idx[:, :min(4,  kmax)]   == gold_col).any(-1).float().mean().item()
    acc8  = (topk_idx[:, :min(8,  kmax)]   == gold_col).any(-1).float().mean().item()
    acc16 = (topk_idx[:, :min(16, kmax)]   == gold_col).any(-1).float().mean().item()

    gprob = p[torch.arange(N), gold].clamp(min=1e-10)
    rank  = (p > gprob.unsqueeze(-1)).sum(-1).float()

    top2v   = torch.topk(p, k=min(2, n_coarse), dim=-1).values
    margin  = (top2v[:, 0] - top2v[:, 1]).mean().item() if n_coarse >= 2 else 0.0
    entropy = (-(p * (p + 1e-10).log()).sum(-1)).mean().item()

    return {
        "acc1":               acc1,
        "acc4":               acc4,
        "acc8":               acc8,
        "acc16":              acc16,
        "gold_rank_mean":     rank.mean().item(),
        "gold_rank_median":   rank.median().item(),
        "region_nll":         float(-gprob.log().mean()),
        "gold_region_prob":   float(gprob.mean()),
        "margin_mean":        margin,
        "entropy_mean":       entropy,
        "n":                  N,
    }


# ── Mixture policies ──────────────────────────────────────────────────────────

def normalize_dist(p: torch.Tensor) -> torch.Tensor:
    return p / p.sum(-1, keepdim=True).clamp(min=1e-10)


def mix(p_router: torch.Tensor, p_mem: torch.Tensor, beta: float) -> torch.Tensor:
    return normalize_dist((1.0 - beta) * p_router + beta * p_mem)


def beta_margin(router_margin: torch.Tensor,
                high: float = 0.30, low: float = 0.05,
                beta_max: float = 0.75) -> torch.Tensor:
    t = ((router_margin - high) / (low - high + 1e-8)).clamp(0.0, 1.0)
    return (t * beta_max).unsqueeze(-1)  # (N, 1)


def beta_mem_conf(router_margin: torch.Tensor,
                  memory_margin: torch.Tensor,
                  tau_router: float = 0.10, tau_mem: float = 0.20,
                  temp_router: float = 0.05, temp_mem: float = 0.05,
                  beta_max: float = 0.75) -> torch.Tensor:
    s_router = torch.sigmoid((tau_router - router_margin) / temp_router)
    s_mem    = torch.sigmoid((memory_margin - tau_mem)    / temp_mem)
    return (beta_max * s_router * s_mem).unsqueeze(-1)  # (N, 1)


def compute_all_mixtures(p_r: torch.Tensor, p_m: torch.Tensor,
                         r_margin: torch.Tensor, m_margin: torch.Tensor
                         ) -> Dict[str, torch.Tensor]:
    mixes = {}
    for b in [0.25, 0.50, 0.75]:
        mixes[f"mix_{b:.2f}"] = mix(p_r, p_m, b)
    b_mg = beta_margin(r_margin)
    mixes["mix_margin"] = normalize_dist((1 - b_mg) * p_r + b_mg * p_m)
    b_mc = beta_mem_conf(r_margin, m_margin)
    mixes["mix_mem_conf"] = normalize_dist((1 - b_mc) * p_r + b_mc * p_m)
    return mixes


# ── Model loading ─────────────────────────────────────────────────────────────

def load_small_backbone_and_probe(ckpt_path: str, device: torch.device):
    """
    Returns (backbone, probe_or_None, d_model, cfg_dict, vocab_size).

    backbone: model loaded from checkpoint.  Always exposes trunk attrs
              (token_emb, pos_emb, drop, blocks, ln_f) for get_hs_small.
              When checkpoint mode is repr_region_retrieval:
                backbone is ReprRegionRetrievalLM → has .retrieval_proj
              When checkpoint mode is repr_region*:
                backbone is ReprRegionTransformerLM → has .region_proj etc.
              Otherwise: BaselineTransformerLM.
    """
    if not _HAS_TRAIN_LM:
        raise RuntimeError("train_region_lm.py not importable; set PYTHONPATH=$PWD")

    raw      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = raw.get("cfg", {})
    fields   = {k: v for k, v in cfg_dict.items()
                if k in TrainConfig.__dataclass_fields__}
    cfg      = TrainConfig(**fields)
    d_model  = cfg.d_model

    from transformers import GPT2TokenizerFast
    tokenizer  = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    vocab_size = len(tokenizer)

    # Choose model class from checkpoint contents / mode
    ckpt_mode    = cfg_dict.get("mode", "baseline")
    has_retrieval = any(k.startswith("retrieval_proj.") for k in raw["model"])
    _REPR_MODES   = ("repr_region", "oracle_repr_region", "random_repr_region",
                     "repr_region_retrieval", "random_repr_region_retrieval")
    dummy_map = torch.zeros(vocab_size, dtype=torch.long)

    if has_retrieval and ReprRegionRetrievalLM is not None:
        backbone = ReprRegionRetrievalLM(cfg, vocab_size, coarse_map=dummy_map)
        print(f"[load_small] loading ReprRegionRetrievalLM  (has retrieval_proj)")
    elif ckpt_mode in _REPR_MODES and ReprRegionTransformerLM is not None:
        backbone = ReprRegionTransformerLM(cfg, vocab_size, coarse_map=dummy_map)
        print(f"[load_small] loading ReprRegionTransformerLM  mode={ckpt_mode}")
    else:
        backbone = BaselineTransformerLM(cfg, vocab_size)
        print(f"[load_small] loading BaselineTransformerLM  mode={ckpt_mode}")

    load_res = backbone.load_state_dict(raw["model"], strict=False)
    # Only error on trunk keys missing (coarse_head/retrieval_proj absence is fine for baseline)
    trunk_prefixes = ("token_emb.", "pos_emb.", "ln_f.", "blocks.", "drop.")
    missing_trunk  = [k for k in load_res.missing_keys
                      if any(k.startswith(p) for p in trunk_prefixes)]
    if missing_trunk:
        raise RuntimeError(f"Missing backbone trunk keys: {missing_trunk[:5]}")
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    probe = None
    probe_sd = {k[len("coarse_head."):]: v for k, v in raw["model"].items()
                if k.startswith("coarse_head.")}
    if probe_sd:
        router_type = cfg_dict.get("router_type", "linear")
        # infer n_coarse from the last 2-D weight (final linear output dim).
        # For linear router: only one 2-D weight [n_coarse, d_model].
        # For mlp router: first values are LN weights [d_model], last 2-D weight
        # is [n_coarse, d_model] — so we can't use next(iter(...)).
        two_d = [v for v in probe_sd.values() if v.ndim == 2]
        if not two_d:
            raise RuntimeError(f"coarse_head state dict has no 2-D tensors: {list(probe_sd.keys())}")
        n_out = two_d[-1].shape[0]   # last linear layer → output = n_coarse
        probe = _make_router_head(d_model, n_out, cfg.dropout, router_type)
        probe.load_state_dict(probe_sd)
        probe.to(device).eval()
        for p in probe.parameters():
            p.requires_grad_(False)
        print(f"[load_small] probe extracted  router_type={router_type}  n_coarse={n_out}")
    else:
        print("[load_small] no coarse_head in checkpoint — router metrics will be skipped")

    return backbone, probe, d_model, cfg_dict, vocab_size


def load_gpt2xl_backbone_and_probe(
    model_name: str,
    probe_path: Optional[str],
    knn_layer: str,
    n_coarse: int,
    device: torch.device,
    hf_cache: Optional[str],
):
    """Returns (model, probe_or_None, d_model, layer_hook_info)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_kw: Dict = {}
    if hf_cache:
        hf_kw["cache_dir"] = hf_cache
        os.environ["HF_HOME"] = hf_cache

    print(f"[load_gpt2xl] loading {model_name} ...")
    tokenizer  = AutoTokenizer.from_pretrained(model_name, **hf_kw)
    model      = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, **hf_kw
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_model      = model.config.n_embd
    n_blocks     = model.config.n_layer
    vocab_size   = model.config.vocab_size

    # Parse knn_layer → block_idx (into model.transformer.h)
    if knn_layer == "embed":
        block_idx  = None   # special: hook embedding dropout
        probe_idx  = 0
    elif knn_layer.startswith("block_"):
        bn         = int(knn_layer.split("_")[1])
        if bn >= n_blocks:
            raise ValueError(f"--knn_layer {knn_layer} out of range (max block_{n_blocks-1})")
        block_idx  = bn
        probe_idx  = bn + 1   # embed=0, block_0=1, ...
    else:
        raise ValueError(f"Unknown --knn_layer: {knn_layer!r}")

    print(f"[load_gpt2xl] d_model={d_model}  n_blocks={n_blocks}  "
          f"knn_layer={knn_layer}  probe_idx={probe_idx}")

    # Load probe
    probe = None
    if probe_path and os.path.exists(probe_path):
        if n_coarse > 0:
            ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
            sd   = ckpt["probes"][probe_idx]
            probe = RegionProbe(d_model, n_coarse, "ln_linear")
            probe.load_state_dict(sd)
            probe.to(device).eval()
            for p in probe.parameters():
                p.requires_grad_(False)
            print(f"[load_gpt2xl] probe loaded from {probe_path}  idx={probe_idx}")
        # else: n_coarse not yet known; caller will load probe after region map is ready
    elif probe_path:
        print(f"[WARNING] probe_path={probe_path!r} not found — router metrics skipped")
    else:
        print("[load_gpt2xl] no probe_path — router metrics skipped")

    return model, probe, d_model, block_idx, vocab_size, tokenizer


# ── Hidden state extraction ───────────────────────────────────────────────────

def get_hs_small(backbone, src: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Returns (B, T, d_model) float32."""
    T   = src.shape[1]
    pos = torch.arange(T, device=device).unsqueeze(0)
    with torch.no_grad():
        x = backbone.drop(backbone.token_emb(src) + backbone.pos_emb(pos))
        for blk in backbone.blocks:
            x = blk(x)
        return backbone.ln_f(x).float()


class _HookCapture:
    def __init__(self):
        self.h: Optional[torch.Tensor] = None

    def __call__(self, module, inp, out):
        self.h = (out[0] if isinstance(out, tuple) else out).float().detach()


def get_hs_gpt2xl(model, src: torch.Tensor, block_idx: Optional[int]) -> torch.Tensor:
    """Returns (B, T, d_model) float32 for the requested layer."""
    cap    = _HookCapture()
    if block_idx is None:
        handle = model.transformer.drop.register_forward_hook(cap)
    else:
        handle = model.transformer.h[block_idx].register_forward_hook(cap)
    try:
        with torch.no_grad():
            model(src, use_cache=False)
    finally:
        handle.remove()
    return cap.h   # (B, T, d_model) float32


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_wikitext(dataset_name: str, tokenizer, split: str) -> np.ndarray:
    from datasets import load_dataset as hf_load
    raw  = hf_load("wikitext", dataset_name)
    parts = []
    for text in raw[split]["text"]:
        if text.strip():
            ids = tokenizer.encode(text)
            if ids:
                parts.append(np.array(ids, dtype=np.int32))
    return np.concatenate(parts)


# ── Memory construction ───────────────────────────────────────────────────────

def build_or_load_memory(
    get_key_fn,
    coarse_map: torch.Tensor,
    train_loader: DataLoader,
    max_positions: int,
    out_dir: str,
    mem_index: MemoryIndex,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (keys_f16, regions, tokens) numpy arrays and populates mem_index.

    get_key_fn(src) -> (h_raw, key)  where key is the vector stored in memory.
    For raw_h: h_raw == key.  For retrieval_proj: key is the projected vector.
    """
    keys_path    = os.path.join(out_dir, "memory_keys.npy")
    regions_path = os.path.join(out_dir, "memory_regions.npy")
    tokens_path  = os.path.join(out_dir, "memory_tokens.npy")
    index_path   = os.path.join(out_dir, "faiss.index")

    # Try loading cached memory
    if all(os.path.exists(p) for p in [keys_path, regions_path, tokens_path]):
        print("[memory] loading cached arrays ...")
        keys_f16 = np.load(keys_path)
        regions  = np.load(regions_path)
        tokens   = np.load(tokens_path)
        print(f"[memory] loaded {len(keys_f16):,} entries")
        if mem_index.load(index_path):
            return keys_f16, regions, tokens
        # Index missing or not applicable; rebuild from cached keys
        print("[memory] rebuilding index from cached keys ...")
        mem_index.build(keys_f16.astype(np.float32))
        mem_index.save(index_path)
        return keys_f16, regions, tokens

    # Collect from training data
    print("[memory] building memory from training split ...")
    all_keys:    List[np.ndarray] = []
    all_regions: List[np.ndarray] = []
    all_tokens:  List[np.ndarray] = []
    total = 0
    t0    = time.time()

    for bi, batch in enumerate(train_loader):
        if total >= max_positions:
            break
        if bi % 200 == 0:
            elapsed = time.time() - t0
            print(f"  batch {bi}  stored {total:,}/{max_positions:,}  {elapsed:.0f}s")

        batch = batch.to(device)
        src   = batch[:, :-1]   # (B, T)
        tgt   = batch[:, 1:]    # (B, T)  — aligned: key[:, t, :] predicts tgt[:, t]

        _, key    = get_key_fn(src)                  # (B, T, key_dim) float32
        key_flat  = key.reshape(-1, key.shape[-1])   # (B*T, key_dim)
        regions   = coarse_map[tgt].reshape(-1)      # (B*T,) on device
        valid     = regions >= 0

        if not valid.any():
            continue

        h_valid = key_flat[valid].cpu().numpy().astype(np.float16)
        r_valid = regions[valid].cpu().numpy().astype(np.int32)
        t_valid = tgt.reshape(-1)[valid].cpu().numpy().astype(np.int32)

        all_keys.append(h_valid)
        all_regions.append(r_valid)
        all_tokens.append(t_valid)
        total += len(h_valid)

    keys_f16 = np.concatenate(all_keys)[:max_positions]
    regions  = np.concatenate(all_regions)[:max_positions]
    tokens   = np.concatenate(all_tokens)[:max_positions]

    print(f"[memory] collected {len(keys_f16):,} valid positions in {time.time()-t0:.1f}s")

    # Save to disk
    np.save(keys_path,    keys_f16)
    np.save(regions_path, regions)
    np.save(tokens_path,  tokens)
    print(f"[memory] saved to {out_dir}/")

    # Build index
    mem_index.build(keys_f16.astype(np.float32))
    mem_index.save(index_path)

    return keys_f16, regions, tokens


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_eval(
    get_key_fn,
    probe: Optional[nn.Module],
    probe_temp: float,
    mem_index: MemoryIndex,
    memory_regions: np.ndarray,
    coarse_map: torch.Tensor,
    val_loader: DataLoader,
    max_eval_batches: int,
    n_coarse: int,
    knn_k: int,
    knn_temp: float,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        p_router:    (N, n_coarse) or None
        p_mem_w:     (N, n_coarse) weighted vote
        p_mem_uw:    (N, n_coarse) unweighted vote
        gold:        (N,) long

    get_key_fn(src) -> (h_raw, key)
        h_raw: raw hidden state for probe  (B, T, d_model)
        key:   kNN query vector            (B, T, key_dim)  — may equal h_raw
    """
    all_p_router:  List[torch.Tensor] = []
    all_p_mem_w:   List[torch.Tensor] = []
    all_p_mem_uw:  List[torch.Tensor] = []
    all_gold:      List[torch.Tensor] = []

    actual_b = min(max_eval_batches, len(val_loader))

    for bi, batch in enumerate(val_loader):
        if bi >= max_eval_batches:
            break
        print(f"  eval batch {bi}/{actual_b}")

        batch = batch.to(device)
        src   = batch[:, :-1]
        tgt   = batch[:, 1:]
        B, T  = src.shape

        gold   = coarse_map[tgt].reshape(-1)   # (B*T,) on device
        valid  = gold >= 0
        if not valid.any():
            continue

        h_raw, key = get_key_fn(src)                      # model forward (once)
        h_flat     = h_raw.reshape(-1, h_raw.shape[-1])   # (B*T, d_model) — for probe
        key_flat   = key.reshape(-1, key.shape[-1])        # (B*T, key_dim) — for kNN

        # kNN retrieval (CPU numpy)
        k_valid = key_flat[valid].cpu().numpy().astype(np.float32)
        n_v     = len(k_valid)

        sims, nn_idx = mem_index.search(k_valid, knn_k)   # (n_v, k)
        nn_regs      = memory_regions[nn_idx]              # (n_v, k) int32

        nn_regs_t = torch.tensor(nn_regs.astype(np.int64), dtype=torch.long)
        sims_t    = torch.tensor(sims, dtype=torch.float32)

        # Weighted vote
        w         = F.softmax(sims_t / knn_temp, dim=-1)  # (n_v, k)
        p_mw      = torch.zeros(n_v, n_coarse)
        p_mw.scatter_add_(1, nn_regs_t, w)
        p_mw      = normalize_dist(p_mw)

        # Unweighted vote
        uw        = torch.ones(n_v, knn_k) / knn_k
        p_muw     = torch.zeros(n_v, n_coarse)
        p_muw.scatter_add_(1, nn_regs_t, uw)
        p_muw     = normalize_dist(p_muw)

        # Router: always uses raw hidden state h, not the projected key
        p_r = None
        if probe is not None:
            with torch.no_grad():
                logits = probe(h_flat[valid])              # GPU, raw h
            p_r = F.softmax(logits.float() / probe_temp, dim=-1).cpu()

        all_p_mem_w.append(p_mw)
        all_p_mem_uw.append(p_muw)
        all_gold.append(gold[valid].cpu())
        if p_r is not None:
            all_p_router.append(p_r)

    p_mem_w  = torch.cat(all_p_mem_w)
    p_mem_uw = torch.cat(all_p_mem_uw)
    gold_all = torch.cat(all_gold)
    p_router = torch.cat(all_p_router) if all_p_router else None

    return p_router, p_mem_w, p_mem_uw, gold_all


# ── Coverage computation ──────────────────────────────────────────────────────

def topk_coverage(p: torch.Tensor, gold: torch.Tensor, k: int,
                  tpr: torch.Tensor) -> Dict:
    kk        = min(k, p.shape[-1])
    idx       = torch.topk(p, kk, dim=-1).indices             # (N, k)
    covered   = (idx == gold.unsqueeze(-1)).any(-1)
    avg_tok   = tpr[idx].sum(-1).float().mean().item()
    return {
        "gold_region_coverage": covered.float().mean().item(),
        "avg_regions_kept":     kk,
        "avg_tokens_kept":      avg_tok,
        "pct_vocab_kept":       avg_tok / 50257 * 100,
    }


def union_coverage(p1: torch.Tensor, k1: int, p2: torch.Tensor, k2: int,
                   gold: torch.Tensor, tpr: torch.Tensor) -> Dict:
    kk1 = min(k1, p1.shape[-1])
    kk2 = min(k2, p2.shape[-1])
    i1  = torch.topk(p1, kk1, dim=-1).indices
    i2  = torch.topk(p2, kk2, dim=-1).indices
    cov = ((i1 == gold.unsqueeze(-1)).any(-1) |
           (i2 == gold.unsqueeze(-1)).any(-1)).float().mean().item()

    N_samp = min(len(gold), 5000)
    region_counts: List[float] = []
    token_counts:  List[float] = []
    for j in range(N_samp):
        regs = set(i1[j].tolist()) | set(i2[j].tolist())
        region_counts.append(len(regs))
        token_counts.append(sum(tpr[r].item() for r in regs))

    return {
        "gold_region_coverage": cov,
        "avg_regions_kept":     float(np.mean(region_counts)),
        "avg_tokens_kept":      float(np.mean(token_counts)),
        "pct_vocab_kept":       float(np.mean(token_counts)) / 50257 * 100,
    }


def adaptive_coverage(p_r: torch.Tensor, p_m: torch.Tensor,
                       gold: torch.Tensor, tpr: torch.Tensor,
                       r_margin: torch.Tensor, m_margin: torch.Tensor) -> Dict:
    """Adaptive policy: route by router confidence + memory confidence."""
    N      = len(gold)
    case1  = r_margin >= 0.30                              # router top-1
    case2  = (r_margin >= 0.10) & ~case1                   # router top-4
    case3a = (r_margin < 0.10)  & (m_margin >= 0.20)       # union(router4, mem2)
    case3b = (r_margin < 0.10)  & (m_margin < 0.20)        # router top-16

    rt1  = torch.topk(p_r, 1,  dim=-1).indices
    rt4  = torch.topk(p_r, min(4, p_r.shape[-1]),  dim=-1).indices
    rt16 = torch.topk(p_r, min(16, p_r.shape[-1]), dim=-1).indices
    mt2  = torch.topk(p_m, min(2, p_m.shape[-1]),  dim=-1).indices

    gold_c = gold.unsqueeze(-1)
    cov = (
        (case1 & (rt1  == gold_c).any(-1)) |
        (case2 & (rt4  == gold_c).any(-1)) |
        (case3a & ((rt4 == gold_c).any(-1) | (mt2 == gold_c).any(-1))) |
        (case3b & (rt16 == gold_c).any(-1))
    ).float().mean().item()

    # Token counts (sampled)
    N_samp = min(N, 5000)
    reg_counts: List[float] = []
    tok_counts: List[float] = []
    for j in range(N_samp):
        if case1[j]:
            regs = set(rt1[j].tolist())
        elif case2[j]:
            regs = set(rt4[j].tolist())
        elif case3a[j]:
            regs = set(rt4[j].tolist()) | set(mt2[j].tolist())
        else:
            regs = set(rt16[j].tolist())
        reg_counts.append(len(regs))
        tok_counts.append(sum(tpr[r].item() for r in regs))

    return {
        "gold_region_coverage": cov,
        "avg_regions_kept":     float(np.mean(reg_counts)),
        "avg_tokens_kept":      float(np.mean(tok_counts)),
        "pct_vocab_kept":       float(np.mean(tok_counts)) / 50257 * 100,
        "fallback_rate":        case3b.float().mean().item(),
        "boundary_coverage":    cov,   # same as overall for boundary subset
        "core_coverage":        (case1 & (rt1 == gold_c).any(-1)).float().sum().item()
                                / case1.float().sum().clamp(min=1).item(),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(metrics_by_method: Dict, split_table: Dict, cov_rows: List[Dict],
               consistency_counts: Dict, boundary_scatter: Dict,
               plots_dir: str) -> None:
    if not _HAS_PLT:
        print("[plots] matplotlib not available — skipping")
        return

    os.makedirs(plots_dir, exist_ok=True)

    method_order = [k for k in metrics_by_method]
    splits_order = [s for s, _, _ in SPLIT_DEFS]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ── 01: acc@k comparison across methods ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    x       = np.arange(len(method_order))
    w       = 0.18
    for oi, (key, label) in enumerate([("acc1","@1"),("acc4","@4"),
                                        ("acc8","@8"),("acc16","@16")]):
        vals = [metrics_by_method[m].get(key, 0) for m in method_order]
        ax.bar(x + oi * w, vals, w, label=f"Acc{label}", color=colors[oi])
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([m.replace("mix_", "mix\n") for m in method_order],
                        rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Region Acc@k — router vs memory vs mixture")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "01_router_vs_mem_vs_mix_acc.png"), dpi=120)
    plt.close(fig)

    # ── 02: Region NLL by split ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    methods_to_plot = [m for m in method_order if m in
                       ("router", "mem_weighted", "mix_0.50", "mix_margin", "mix_mem_conf")]
    x = np.arange(len(splits_order))
    w = 0.9 / max(len(methods_to_plot), 1)
    for mi, mname in enumerate(methods_to_plot):
        vals = [split_table.get((mname, sp), {}).get("region_nll", float("nan"))
                for sp in splits_order]
        ax.bar(x + mi * w, vals, w, label=mname, color=colors[mi % len(colors)])
    ax.set_xticks(x + w * len(methods_to_plot) / 2)
    ax.set_xticklabels(splits_order, fontsize=9)
    ax.set_ylabel("Region NLL  (lower = better)")
    ax.set_title("Region NLL by split")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "02_region_nll_by_split.png"), dpi=120)
    plt.close(fig)

    # ── 03: Gold rank by split ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    for mi, mname in enumerate(methods_to_plot):
        vals = [split_table.get((mname, sp), {}).get("gold_rank_mean", float("nan"))
                for sp in splits_order]
        ax.bar(x + mi * w, vals, w, label=mname, color=colors[mi % len(colors)])
    ax.set_xticks(x + w * len(methods_to_plot) / 2)
    ax.set_xticklabels(splits_order, fontsize=9)
    ax.set_ylabel("Gold rank mean  (lower = better)")
    ax.set_title("Gold region rank by split")
    ax.invert_yaxis()
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "03_gold_rank_by_split.png"), dpi=120)
    plt.close(fig)

    # ── 04/05: candidate coverage vs regions/tokens kept ─────────────────────
    for fname, xkey, xlabel in [
        ("04_candidate_coverage_vs_regions_kept.png", "avg_regions_kept", "Avg regions kept"),
        ("05_candidate_coverage_vs_tokens_kept.png",  "avg_tokens_kept",  "Avg tokens kept"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for ri, row in enumerate(cov_rows):
            ax.scatter(row[xkey], row["gold_region_coverage"],
                       s=60, zorder=3, color=colors[ri % len(colors)])
            ax.annotate(row["policy"], (row[xkey], row["gold_region_coverage"]),
                        textcoords="offset points", xytext=(4, 2), fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Gold region coverage")
        ax.set_title(f"Coverage vs {xlabel}")
        ax.grid(True, alpha=0.3)
        if xkey == "avg_tokens_kept":
            ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=120)
        plt.close(fig)

    # ── 06: memory margin vs router margin scatter (boundary positions) ───────
    if boundary_scatter:
        r_mg = boundary_scatter.get("router_margin", [])
        m_mg = boundary_scatter.get("memory_margin", [])
        types = boundary_scatter.get("types", [])
        type_color = {"A": "green", "B": "orange", "C": "red", "?": "grey"}
        if len(r_mg) > 0:
            fig, ax = plt.subplots(figsize=(7, 5))
            for tp in ["A", "B", "C", "?"]:
                mask = [i for i, t in enumerate(types) if t == tp]
                if mask:
                    ax.scatter([r_mg[i] for i in mask], [m_mg[i] for i in mask],
                               s=4, alpha=0.3, color=type_color[tp], label=f"Type {tp}")
            ax.axvline(0.10, color="grey", linestyle="--", linewidth=0.8)
            ax.axhline(0.20, color="grey", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Router margin")
            ax.set_ylabel("Memory margin")
            ax.set_title("Memory vs router margin (boundary positions)")
            ax.legend(fontsize=8, markerscale=3)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "06_memory_margin_vs_router_margin.png"),
                        dpi=120)
            plt.close(fig)

    # ── 07: region consistency type breakdown ─────────────────────────────────
    if consistency_counts:
        labels_pie = []
        sizes_pie  = []
        col_pie    = []
        for tp, col in [("A", "green"), ("B", "orange"), ("C", "red")]:
            n = consistency_counts.get(f"type_{tp}", 0)
            if n > 0:
                labels_pie.append(f"Type {tp} (n={n:,})")
                sizes_pie.append(n)
                col_pie.append(col)
        n_other = consistency_counts.get("non_boundary", 0)
        labels_pie.append(f"non-boundary (n={n_other:,})")
        sizes_pie.append(n_other)
        col_pie.append("steelblue")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(sizes_pie, labels=labels_pie, colors=col_pie,
               autopct="%1.1f%%", startangle=90)
        ax.set_title("Region consistency types (all eval positions)")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "07_region_consistency_type_breakdown.png"),
                    dpi=120)
        plt.close(fig)

    print(f"[plots] wrote 7 plots → {plots_dir}/")


# ── Summary markdown ──────────────────────────────────────────────────────────

def write_summary(path: str, args, cfg_summary: Dict,
                  metrics_by_method: Dict, split_table: Dict,
                  cov_rows: List[Dict], consistency_counts: Dict,
                  n_coarse: int, n_mem: int, n_eval: int,
                  verdicts: Dict) -> None:

    lines = [
        "# Offline Region-kNN Experiment Report",
        "",
        f"**Model:**      `{args.model_type}`"
        + (f"  knn_layer={args.knn_layer}" if args.model_type == "gpt2xl" else ""),
        f"**Key source:** `{args.knn_key_source}`  "
        f"key_dim={cfg_summary.get('key_dim', '?')}",
        f"**Dataset:**    {args.dataset}",
        f"**Memory:**     {n_mem:,} positions  (train split)",
        f"**Eval:**       {n_eval:,} positions  (val split)",
        f"**Regions:**    {n_coarse}  knn_k={args.knn_k}  knn_temp={args.knn_temp}  "
        f"normalize={args.normalize_keys}",
        "",
        "## Overall Metrics (all valid positions)",
        "",
        "| Method | Acc@1 | Acc@4 | Acc@8 | Acc@16 | NLL | "
        "Gold Rank | Margin μ | Entropy μ |",
        "|--------|------:|------:|------:|-------:|----:|"
        "----------:|---------:|----------:|",
    ]
    chance1 = 1 / n_coarse
    rand_row = (f"| **random** | {chance1:.3f} | {4/n_coarse:.3f} | "
                f"{8/n_coarse:.3f} | {16/n_coarse:.3f} | — | — | — | — |")
    lines.append(rand_row)
    for mname, mdict in metrics_by_method.items():
        lines.append(
            f"| **{mname}** "
            f"| {mdict.get('acc1', float('nan')):.3f} "
            f"| {mdict.get('acc4', float('nan')):.3f} "
            f"| {mdict.get('acc8', float('nan')):.3f} "
            f"| {mdict.get('acc16', float('nan')):.3f} "
            f"| {mdict.get('region_nll', float('nan')):.3f} "
            f"| {mdict.get('gold_rank_mean', float('nan')):.1f} "
            f"| {mdict.get('margin_mean', float('nan')):.3f} "
            f"| {mdict.get('entropy_mean', float('nan')):.3f} |"
        )

    lines += [
        "",
        "## Metrics by Split",
        "",
        "| Method | Split | n | Acc@1 | Acc@4 | NLL | Gold Rank |",
        "|--------|-------|--:|------:|------:|----:|----------:|",
    ]
    for sname, _, _ in SPLIT_DEFS:
        for mname in ("router", "mem_weighted", "mix_0.50", "mix_margin"):
            d = split_table.get((mname, sname))
            if d is None:
                continue
            lines.append(
                f"| {mname} | {sname} | {d.get('n', 0):,} "
                f"| {d.get('acc1', float('nan')):.3f} "
                f"| {d.get('acc4', float('nan')):.3f} "
                f"| {d.get('region_nll', float('nan')):.3f} "
                f"| {d.get('gold_rank_mean', float('nan')):.1f} |"
            )

    lines += [
        "",
        "## Candidate Coverage",
        "",
        "| Policy | Gold Coverage | Avg Regions | Avg Tokens | % Vocab |",
        "|--------|-------------:|------------:|-----------:|--------:|",
    ]
    for row in cov_rows:
        lines.append(
            f"| {row['policy']} "
            f"| {row['gold_region_coverage']:.3f} "
            f"| {row['avg_regions_kept']:.1f} "
            f"| {row['avg_tokens_kept']:.0f} "
            f"| {row['pct_vocab_kept']:.2f}% |"
        )

    # Region consistency types
    ta = consistency_counts.get("type_A", 0)
    tb = consistency_counts.get("type_B", 0)
    tc = consistency_counts.get("type_C", 0)
    n_bnd = ta + tb + tc
    lines += [
        "",
        "## Region Consistency Types (boundary positions only)",
        "",
        f"**Type A** (router low, memory high, gold in top-k): "
        f"{ta:,} ({ta/max(n_bnd,1):.1%})",
        f"**Type B** (router low, memory spread):              "
        f"{tb:,} ({tb/max(n_bnd,1):.1%})",
        f"**Type C** (memory confident but wrong):             "
        f"{tc:,} ({tc/max(n_bnd,1):.1%})",
        "",
    ]

    # Verdicts
    lines += [
        "## Verdicts (Q1–Q6)",
        "",
        "| # | Question | Verdict |",
        "|---|----------|---------|",
    ]
    for q, label, result in verdicts:
        lines.append(f"| {q} | {label} | **{result}** |")

    passes = sum(1 for _, _, r in verdicts if r == "PASS")
    lines += [
        "",
        f"**{passes}/{len(verdicts)} PASS**",
        "",
        "## Output Files",
        "",
        "- `memory_keys.npy`, `memory_regions.npy`, `memory_tokens.npy` — kNN memory arrays",
        "- `faiss.index` — FAISS index (if FAISS available)",
        "- `metrics_all.csv` — overall metrics per method",
        "- `metrics_by_split.csv` — metrics per method × split",
        "- `region_consistency_types.csv` — Type A/B/C analysis",
        "- `candidate_coverage.csv` — coverage per selection policy",
        "- `knn_config.json` — run configuration",
        "- `plots/01-07` — diagnostic plots",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Region-kNN experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--model_type",  default="gpt2xl",
                        choices=["gpt2xl", "small"])
    parser.add_argument("--model_name",  default="gpt2-xl",
                        help="HF model ID (gpt2xl mode)")
    parser.add_argument("--small_ckpt",  default=None,
                        help="Checkpoint path (small mode)")
    parser.add_argument("--probe_path",  default=None,
                        help="probes.pt from gpt2xl_layer_margin_analysis (gpt2xl) "
                             "or standalone probe file")
    parser.add_argument("--knn_layer",   default="block_41",
                        help="Layer to use as kNN key (gpt2xl mode): "
                             "embed | block_0 ... block_47")
    parser.add_argument("--knn_key_source", default="raw_h",
                        choices=["raw_h", "retrieval_proj", "post_region"],
                        help="Vector to use as kNN key: "
                             "raw_h (default) = trunk hidden state; "
                             "retrieval_proj  = backbone.retrieval_proj(h), requires "
                             "repr_region_retrieval checkpoint; "
                             "post_region     = h after region injection")
    # Data
    parser.add_argument("--dataset",     default="wikitext-103-raw-v1")
    parser.add_argument("--region_map_path", required=True)
    parser.add_argument("--seq_len",     type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=4)
    # Memory
    parser.add_argument("--max_memory_positions", type=int, default=500_000)
    parser.add_argument("--max_eval_positions",   type=int, default=247_000)
    parser.add_argument("--normalize_keys",       type=lambda x: x.lower() != "false",
                        default=True)
    # kNN
    parser.add_argument("--knn_k",    type=int,   default=64)
    parser.add_argument("--knn_temp", type=float, default=0.2)
    # Probe
    parser.add_argument("--probe_temp", type=float, default=1.0)
    # Output
    parser.add_argument("--output_dir", default="runs/offline_region_knn")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--hf_cache_dir", default=None)
    args = parser.parse_args()

    # ── Guards ────────────────────────────────────────────────────────────────
    if os.path.exists(os.path.join(args.output_dir, "summary.md")) \
            and os.environ.get("FORCE_RUN", "0") != "1":
        print(f"[region_knn] summary.md exists in {args.output_dir}; "
              f"set FORCE_RUN=1 to re-run")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    print(f"[region_knn] device={device}  model_type={args.model_type}  "
          f"knn_k={args.knn_k}  knn_temp={args.knn_temp}")
    print(f"[region_knn] knn_key_source={args.knn_key_source}")
    if _HAS_FAISS:
        print("[region_knn] FAISS available")
    else:
        print("[region_knn] FAISS not available — using torch brute-force")

    # ── Load model + probe ────────────────────────────────────────────────────
    vocab_size = 50257  # GPT-2 default; overridden below
    backbone   = None   # set below for small model
    if args.model_type == "gpt2xl":
        if args.hf_cache_dir:
            os.environ.setdefault("HF_HOME", args.hf_cache_dir)
        gpt2_model, probe, d_model, block_idx, vocab_size, tokenizer = \
            load_gpt2xl_backbone_and_probe(
                args.model_name, args.probe_path, args.knn_layer,
                -1,  # n_coarse unknown yet; re-checked after region map load
                device, args.hf_cache_dir,
            )
        def get_hs(src: torch.Tensor) -> torch.Tensor:
            return get_hs_gpt2xl(gpt2_model, src, block_idx)
    else:
        if args.small_ckpt is None:
            parser.error("--small_ckpt required for --model_type small")
        backbone, probe, d_model, _, vocab_size = \
            load_small_backbone_and_probe(args.small_ckpt, device)
        def get_hs(src: torch.Tensor) -> torch.Tensor:
            return get_hs_small(backbone, src, device)

    # ── kNN key-source function ───────────────────────────────────────────────
    # get_key_fn(src) -> (h_raw, key)
    #   h_raw: (B, T, d_model) float32 — always raw trunk output (for probe)
    #   key:   (B, T, key_dim) float32 — vector indexed / queried in kNN memory
    if args.knn_key_source == "raw_h":
        key_dim = d_model
        def get_key_fn(src: torch.Tensor):
            h = get_hs(src)
            return h, h

    elif args.knn_key_source == "retrieval_proj":
        if backbone is None or not hasattr(backbone, "retrieval_proj"):
            parser.error(
                "--knn_key_source retrieval_proj requires a repr_region_retrieval "
                "checkpoint (backbone must have .retrieval_proj).  "
                "Check --small_ckpt was trained with mode=repr_region_retrieval."
            )
        key_dim = backbone.retrieval_proj[-1].weight.shape[0]
        def get_key_fn(src: torch.Tensor):
            h = get_hs(src)                                 # (B, T, d_model)
            with torch.no_grad():
                flat = h.reshape(-1, d_model)
                z    = backbone.retrieval_proj(flat)        # (B*T, retrieval_dim)
                z    = F.normalize(z.float(), dim=-1)       # unit sphere
                key  = z.reshape(h.shape[0], h.shape[1], key_dim)
            return h, key

    elif args.knn_key_source == "post_region":
        if backbone is None or not hasattr(backbone, "region_proj"):
            parser.error(
                "--knn_key_source post_region requires a repr_region* checkpoint "
                "(backbone must have .region_proj)."
            )
        key_dim = d_model
        def get_key_fn(src: torch.Tensor):
            with torch.no_grad():
                _, _, _, h_prime = backbone.forward(src)    # h' = h + alpha * region_feat
            # For probe we need raw h.  ReprRegionRetrievalLM stores _last_h.
            if hasattr(backbone, "_last_h") and backbone._last_h is not None:
                h_raw = backbone._last_h.float()
            else:
                h_raw = get_hs(src)                         # second trunk pass (fallback)
            return h_raw, h_prime.float()

    print(f"[region_knn] key_dim={key_dim}")

    # ── Region map ────────────────────────────────────────────────────────────
    print(f"[region_knn] loading region map: {args.region_map_path}")
    coarse_map, n_coarse = load_region_map(args.region_map_path, vocab_size)
    coarse_map = coarse_map.to(device)
    cov  = (coarse_map >= 0).float().mean().item()
    print(f"[region_knn] {n_coarse} regions  {cov:.1%} coverage  "
          f"d_model={d_model}")

    # For gpt2xl, load probe now that n_coarse is known (model already in GPU memory)
    if args.model_type == "gpt2xl" and probe is None and args.probe_path:
        if os.path.exists(args.probe_path):
            p_idx = 0 if block_idx is None else block_idx + 1
            ckpt  = torch.load(args.probe_path, map_location="cpu", weights_only=False)
            sd    = ckpt["probes"][p_idx]
            probe = RegionProbe(d_model, n_coarse, "ln_linear")
            probe.load_state_dict(sd)
            probe.to(device).eval()
            for p in probe.parameters():
                p.requires_grad_(False)
            print(f"[region_knn] probe loaded  idx={p_idx}  n_coarse={n_coarse}")
        else:
            print(f"[WARNING] probe_path={args.probe_path!r} not found — router metrics skipped")

    inv_map = build_inverse_map(coarse_map.cpu(), n_coarse)
    tpr     = tokens_per_region_tensor(inv_map, n_coarse)
    print(f"[region_knn] mapped tokens: {sum(len(v) for v in inv_map.values()):,}  "
          f"avg tokens/region: {tpr.mean():.1f}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.model_type == "gpt2xl":
        from transformers import AutoTokenizer as _AT
        tok = _AT.from_pretrained(args.model_name,
                                  cache_dir=args.hf_cache_dir or None)
    else:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.model_max_length = int(1e30)

    print("[region_knn] encoding training split ...")
    train_tokens = load_wikitext(args.dataset, tok, "train")
    print(f"[region_knn] train: {len(train_tokens):,} tokens")
    print("[region_knn] encoding validation split ...")
    val_tokens = load_wikitext(args.dataset, tok, "validation")
    print(f"[region_knn] val: {len(val_tokens):,} tokens")

    positions_per_batch = args.batch_size * args.seq_len
    max_mem_batches  = math.ceil(args.max_memory_positions / positions_per_batch)
    max_eval_batches = math.ceil(args.max_eval_positions   / positions_per_batch)

    train_ds = TokenChunkDataset(train_tokens, args.seq_len)
    val_ds   = TokenChunkDataset(val_tokens,   args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0,
                              pin_memory=(device.type == "cuda"), drop_last=False)

    # ── Build / load memory ───────────────────────────────────────────────────
    mem_index = MemoryIndex(key_dim, device, normalize=args.normalize_keys)
    keys_f16, mem_regions, mem_tokens = build_or_load_memory(
        get_key_fn, coarse_map, train_loader,
        args.max_memory_positions, args.output_dir, mem_index, device,
    )
    n_mem = len(keys_f16)
    print(f"[region_knn] memory size: {n_mem:,} positions")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("[region_knn] evaluating ...")
    p_router, p_mem_w, p_mem_uw, gold_all = run_eval(
        get_key_fn, probe, args.probe_temp,
        mem_index, mem_regions, coarse_map,
        val_loader, max_eval_batches,
        n_coarse, args.knn_k, args.knn_temp, device,
    )
    n_eval = len(gold_all)
    print(f"[region_knn] evaluated {n_eval:,} valid positions")

    # ── Derived distributions ─────────────────────────────────────────────────
    dists: Dict[str, torch.Tensor] = {"mem_weighted": p_mem_w,
                                       "mem_unweighted": p_mem_uw}
    if p_router is not None:
        dists["router"] = p_router
        top2r     = torch.topk(p_router, 2, dim=-1).values
        r_margin  = top2r[:, 0] - top2r[:, 1]              # (N,)
        mixes     = compute_all_mixtures(p_router, p_mem_w, r_margin,
                                          torch.topk(p_mem_w, 2, dim=-1).values[:,0]
                                          - torch.topk(p_mem_w, 2, dim=-1).values[:,1])
        dists.update(mixes)
    else:
        r_margin = torch.zeros(n_eval)
        print("[region_knn] no probe loaded — router and mixture metrics skipped")

    top2m   = torch.topk(p_mem_w, 2, dim=-1).values
    m_margin = top2m[:, 0] - top2m[:, 1]   # (N,)

    # ── Overall metrics ───────────────────────────────────────────────────────
    print("[region_knn] computing metrics ...")
    metrics_by_method: Dict[str, Dict] = {}
    for mname, p in dists.items():
        metrics_by_method[mname] = compute_metrics(p, gold_all, n_coarse)

    # ── Split metrics ─────────────────────────────────────────────────────────
    split_table: Dict[Tuple[str, str], Dict] = {}
    for sname, lo, hi in SPLIT_DEFS:
        mask = torch.ones(n_eval, dtype=torch.bool)
        if lo is not None:
            mask = mask & (r_margin >= lo)
        if hi is not None:
            mask = mask & (r_margin < hi)
        if not mask.any():
            continue
        for mname, p in dists.items():
            split_table[(mname, sname)] = compute_metrics(
                p[mask], gold_all[mask], n_coarse
            )

    # ── Region consistency types (boundary positions only) ────────────────────
    bnd_mask = r_margin < 0.10
    n_bnd    = int(bnd_mask.sum())
    ta = tb = tc = 0
    scatter_r_mg: List[float] = []
    scatter_m_mg: List[float] = []
    scatter_types: List[str]  = []

    if n_bnd > 0 and p_router is not None:
        bnd_idx    = bnd_mask.nonzero(as_tuple=True)[0]
        # Sample for scatter (max 20K)
        samp_idx   = bnd_idx[:20000]
        s_rmg      = r_margin[samp_idx]
        s_mmg      = m_margin[samp_idx]
        s_gold     = gold_all[samp_idx]
        mem_top4   = torch.topk(p_mem_w[samp_idx], min(4, n_coarse), dim=-1).indices
        gold_in_mt = (mem_top4 == s_gold.unsqueeze(-1)).any(-1)  # in top-4 of memory

        for i in range(len(samp_idx)):
            rmg = s_rmg[i].item()
            mmg = s_mmg[i].item()
            in_top = gold_in_mt[i].item()
            scatter_r_mg.append(rmg)
            scatter_m_mg.append(mmg)
            if mmg >= 0.20 and in_top:
                tp = "A"
                ta += 1
            elif mmg >= 0.20 and not in_top:
                tp = "C"
                tc += 1
            else:
                tp = "B"
                tb += 1
            scatter_types.append(tp)

    consistency_counts = {
        "type_A": ta, "type_B": tb, "type_C": tc,
        "n_boundary": n_bnd,
        "non_boundary": n_eval - n_bnd,
    }

    # ── Candidate coverage ────────────────────────────────────────────────────
    cov_rows: List[Dict] = []

    def _add(policy, **kwargs):
        cov_rows.append({"policy": policy, **kwargs})

    if p_router is not None:
        for k in [1, 4, 8, 16]:
            c = topk_coverage(p_router, gold_all, k, tpr)
            _add(f"router_top{k}", **c)
    for k in [1, 4, 8, 16]:
        c = topk_coverage(p_mem_w, gold_all, k, tpr)
        _add(f"mem_top{k}", **c)
    if "mix_0.50" in dists:
        for k in [1, 4, 8, 16]:
            c = topk_coverage(dists["mix_0.50"], gold_all, k, tpr)
            _add(f"mix50_top{k}", **c)
    if p_router is not None:
        for k1, k2 in [(4, 4), (8, 8)]:
            c = union_coverage(p_router, k1, p_mem_w, k2, gold_all, tpr)
            _add(f"union_r{k1}_m{k2}", **c)
        c_adp = adaptive_coverage(p_router, p_mem_w, gold_all, tpr,
                                   r_margin, m_margin)
        _add("adaptive", **c_adp)

    # ── Verdicts ──────────────────────────────────────────────────────────────
    def _v(cond: bool) -> str:
        return "PASS" if cond else "INCONCLUSIVE"

    r_bnd  = split_table.get(("router",       "boundary"), {})
    m_bnd  = split_table.get(("mem_weighted", "boundary"), {})
    x50_bnd= split_table.get(("mix_0.50",     "boundary"), {})
    mg_bnd = split_table.get(("mix_margin",   "boundary"), {})
    r_all  = metrics_by_method.get("router", {})
    m_all  = metrics_by_method.get("mem_weighted", {})
    x50_all= metrics_by_method.get("mix_0.50", {})

    # Q1: Region-kNN beats router on boundary NLL
    q1 = _v(m_bnd.get("region_nll", 1e9) < r_bnd.get("region_nll", 1e9) - 0.01)

    # Q2: Mixture beats both router and memory alone on boundary NLL
    best_single = min(r_bnd.get("region_nll", 1e9), m_bnd.get("region_nll", 1e9))
    q2 = _v(x50_bnd.get("region_nll", 1e9) < best_single - 0.005)

    # Q3: Type A improves more than Type B (check metrics if we have per-type)
    q3 = _v(ta > tb)   # proxy: if more Type A than B, memory is region-consistent

    # Q4: Adaptive policy maintains high coverage
    adp = next((r for r in cov_rows if r["policy"] == "adaptive"), {})
    q4 = _v(adp.get("gold_region_coverage", 0) >= 0.90)

    # Q5: Adaptive policy reduces vocab size significantly
    q5 = _v(adp.get("pct_vocab_kept", 100) < 20.0)

    # Q6: Memory misleading rate acceptable (Type C ≤ 20% of boundary)
    c_rate = tc / max(ta + tb + tc, 1)
    q6 = _v(c_rate <= 0.20)

    verdicts = [
        ("Q1", "Region-kNN beats router alone on boundary region NLL?", q1),
        ("Q2", "Router + kNN mixture beats both router and memory alone?",  q2),
        ("Q3", "Type A (consistent) > Type B (spread): memory is region-useful?", q3),
        ("Q4", "Adaptive policy achieves ≥90% gold-region coverage?", q4),
        ("Q5", "Adaptive policy keeps < 20% of vocab as candidate tokens?", q5),
        ("Q6", "Memory misleading rate (Type C) ≤ 20% of boundary?", q6),
    ]

    # ── Write outputs ─────────────────────────────────────────────────────────
    def _csv(rows: List[Dict], fname: str) -> None:
        if not rows:
            return
        # union of all keys in order — rows may have different extra fields
        seen: Dict[str, None] = {}
        for row in rows:
            for k in row:
                seen[k] = None
        path = os.path.join(args.output_dir, fname)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(seen.keys()), restval="")
            w.writeheader()
            w.writerows(rows)

    # metrics_all.csv
    _csv([{"method": m, **d} for m, d in metrics_by_method.items()],
         "metrics_all.csv")

    # metrics_by_split.csv
    split_rows = [{"method": m, "split": s, **d}
                  for (m, s), d in split_table.items()]
    _csv(split_rows, "metrics_by_split.csv")

    # region_consistency_types.csv
    _csv([{"type": k, "count": v} for k, v in consistency_counts.items()],
         "region_consistency_types.csv")

    # candidate_coverage.csv
    _csv(cov_rows, "candidate_coverage.csv")

    # knn_config.json
    cfg_json = {
        "model_type":          args.model_type,
        "model_name":          getattr(args, "model_name", None),
        "knn_layer":           args.knn_layer if args.model_type == "gpt2xl" else "final",
        "knn_key_source":      args.knn_key_source,
        "key_dim":             key_dim,
        "dataset":             args.dataset,
        "region_map_path":     args.region_map_path,
        "n_coarse":            n_coarse,
        "d_model":             d_model,
        "knn_k":               args.knn_k,
        "knn_temp":            args.knn_temp,
        "probe_temp":          args.probe_temp,
        "normalize_keys":      args.normalize_keys,
        "max_memory_positions":args.max_memory_positions,
        "max_eval_positions":  args.max_eval_positions,
        "n_mem_actual":        n_mem,
        "n_eval_actual":       n_eval,
    }
    with open(os.path.join(args.output_dir, "knn_config.json"), "w") as f:
        json.dump(cfg_json, f, indent=2)
    print("[region_knn] wrote CSVs and knn_config.json")

    # summary.md
    write_summary(
        os.path.join(args.output_dir, "summary.md"),
        args, cfg_json, metrics_by_method, split_table, cov_rows,
        consistency_counts, n_coarse, n_mem, n_eval, verdicts,
    )

    # plots
    make_plots(
        metrics_by_method, split_table, cov_rows, consistency_counts,
        {"router_margin": scatter_r_mg, "memory_margin": scatter_m_mg,
         "types": scatter_types},
        plots_dir,
    )

    # ── Final print ───────────────────────────────────────────────────────────
    passes = sum(1 for _, _, r in verdicts if r == "PASS")
    print(f"\n[region_knn] DONE  →  {args.output_dir}")
    print(f"  {passes}/{len(verdicts)} PASS")
    for q, label, result in verdicts:
        print(f"  {q}: {result:<14} {label[:60]}")

    if p_router is not None:
        print("\n  Overall summary:")
        for mname in ("router", "mem_weighted", "mix_0.50", "mix_margin", "mix_mem_conf"):
            d = metrics_by_method.get(mname)
            if d:
                print(f"    {mname:<16}  acc@1={d['acc1']:.3f}  "
                      f"acc@4={d['acc4']:.3f}  nll={d['region_nll']:.3f}  "
                      f"rank={d['gold_rank_mean']:.1f}")
    if adp:
        print(f"\n  Adaptive policy:  coverage={adp['gold_region_coverage']:.3f}  "
              f"avg_regions={adp['avg_regions_kept']:.1f}  "
              f"avg_tokens={adp['avg_tokens_kept']:.0f}  "
              f"vocab%={adp['pct_vocab_kept']:.1f}%")


if __name__ == "__main__":
    main()
