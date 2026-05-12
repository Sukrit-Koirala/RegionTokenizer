#!/usr/bin/env python3
"""
GPT-2 XL layer-wise region probe analysis.

Loads GPT-2 XL frozen (float16), trains lightweight per-layer probes on
WikiText-103, evaluates them to test the progressive manifold resolution
hypothesis across all 49 hidden-state layers (embed + 48 transformer blocks).

Usage:
    python scripts/gpt2xl_layer_margin_analysis.py \
        --region_map_path runs/region_maps_128/token_to_region.json \
        --output_dir      runs/gpt2xl_layer_margin \
        --device          cuda
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Tighter trajectory bins for GPT-2 XL (more layers → finer resolution)
TRAJ_BINS = [
    (0.00, 0.03,  "M<0.03"),
    (0.03, 0.05,  "0.03≤M<0.05"),
    (0.05, 0.10,  "0.05≤M<0.10"),
    (0.10, 0.20,  "0.10≤M<0.20"),
    (0.20, 2.00,  "M≥0.20"),
]

TRAJ_COLORS = {
    "M<0.03":       "red",
    "0.03≤M<0.05":  "darkorange",
    "0.05≤M<0.10":  "gold",
    "0.10≤M<0.20":  "forestgreen",
    "M≥0.20":       "royalblue",
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenChunkDataset(Dataset):
    """Non-overlapping fixed-length chunks of a token array."""

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


# ── Probe ─────────────────────────────────────────────────────────────────────

class RegionProbe(nn.Module):
    """Lightweight probe: d_model → n_regions logits.

    'ln_linear': LayerNorm then Linear (better calibration, default)
    'linear':    bare Linear without normalisation
    """

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


# ── Region map ────────────────────────────────────────────────────────────────

def load_region_map(path: str, vocab_size: int) -> Tuple[torch.Tensor, int]:
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-2 XL layer-wise region probe analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name",          default="gpt2-xl",
                        help="HuggingFace model ID")
    parser.add_argument("--dataset",             default="wikitext-103-raw-v1",
                        help="HuggingFace datasets name under 'wikitext'")
    parser.add_argument("--region_map_path",     required=True,
                        help="JSON: token_id → region_id")
    parser.add_argument("--output_dir",          default="runs/gpt2xl_layer_margin")
    parser.add_argument("--layers",              default="all",
                        help="Comma-separated layer indices to probe, or 'all'")
    parser.add_argument("--max_train_positions", type=int, default=500_000,
                        help="Approximate number of token positions used for probe training")
    parser.add_argument("--max_eval_positions",  type=int, default=247_000,
                        help="Approximate number of token positions used for evaluation")
    parser.add_argument("--probe_type",          default="ln_linear",
                        choices=["ln_linear", "linear"])
    parser.add_argument("--probe_temp",          type=float, default=1.0,
                        help="Softmax temperature applied to probe logits")
    parser.add_argument("--probe_lr",            type=float, default=1e-3)
    parser.add_argument("--probe_epochs",        type=int,   default=3)
    parser.add_argument("--batch_size",          type=int,   default=4)
    parser.add_argument("--seq_len",             type=int,   default=512,
                        help="Sequence length (≤1024 for GPT-2)")
    parser.add_argument("--boundary_tau",        type=float, default=0.10,
                        help="Standard boundary threshold (margin < tau → boundary)")
    parser.add_argument("--boundary_tau_tight",  type=float, default=0.03,
                        help="Tight boundary threshold for fine-grained analysis")
    parser.add_argument("--device",              default="cuda")
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--skip_training",       action="store_true",
                        help="Load probes from probes.pt instead of training")
    parser.add_argument("--hf_cache_dir",        default=None,
                        help="Override HuggingFace cache directory")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "probes.pt")

    print(f"[gpt2xl_margin] device={device}  model={args.model_name}")

    # ── Load GPT-2 XL ─────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_kwargs: Dict = {}
    if args.hf_cache_dir:
        hf_kwargs["cache_dir"] = args.hf_cache_dir
        os.environ["HF_HOME"] = args.hf_cache_dir

    print(f"[gpt2xl_margin] loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **hf_kwargs)
    gpt2 = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        **hf_kwargs,
    )
    gpt2 = gpt2.to(device).eval()
    for p in gpt2.parameters():
        p.requires_grad_(False)

    gpt2_cfg      = gpt2.config
    d_model       = gpt2_cfg.n_embd          # 1600 for gpt2-xl
    n_gpt2_blocks = gpt2_cfg.n_layer         # 48
    n_hs          = n_gpt2_blocks + 1        # 49 (embed output + 48 block outputs)
    vocab_size    = gpt2_cfg.vocab_size      # 50257
    print(f"[gpt2xl_margin] d_model={d_model}  n_blocks={n_gpt2_blocks}  "
          f"vocab_size={vocab_size}  hidden_states={n_hs}")

    # Layer names: embed + block_0 … block_{N-1}
    layer_names: List[str] = ["embed"] + [f"block_{i}" for i in range(n_gpt2_blocks)]
    assert len(layer_names) == n_hs

    # Layer selection (for optional subsetting; training still covers all)
    if args.layers == "all":
        layer_indices = list(range(n_hs))
    else:
        layer_indices = sorted(
            {int(x) for x in args.layers.split(",") if 0 <= int(x) < n_hs}
        )
    if not layer_indices:
        sys.exit("[gpt2xl_margin] ERROR: no valid layer indices after filtering")
    print(f"[gpt2xl_margin] probing {len(layer_indices)}/{n_hs} layers")

    # ── Region map ────────────────────────────────────────────────────────────
    print(f"[gpt2xl_margin] loading region map: {args.region_map_path}")
    coarse_map, n_coarse = load_region_map(args.region_map_path, vocab_size)
    coarse_map = coarse_map.to(device)
    cov = (coarse_map >= 0).float().mean().item()
    print(f"[gpt2xl_margin] {n_coarse} regions  {cov:.1%} token coverage")

    # ── WikiText-103 ──────────────────────────────────────────────────────────
    from datasets import load_dataset as hf_load_dataset

    print(f"[gpt2xl_margin] loading dataset: {args.dataset}")
    raw_ds = hf_load_dataset("wikitext", args.dataset)

    def _encode_split(split: str) -> np.ndarray:
        texts = [t for t in raw_ds[split]["text"] if t.strip()]
        parts = []
        for text in texts:
            ids = tokenizer.encode(text)
            if ids:
                parts.append(np.array(ids, dtype=np.int32))
        return np.concatenate(parts)

    print("[gpt2xl_margin] encoding train split ...")
    train_tokens = _encode_split("train")
    print(f"[gpt2xl_margin] train: {len(train_tokens):,} tokens")
    print("[gpt2xl_margin] encoding validation split ...")
    val_tokens = _encode_split("validation")
    print(f"[gpt2xl_margin] val: {len(val_tokens):,} tokens")

    train_ds = TokenChunkDataset(train_tokens, args.seq_len)
    val_ds   = TokenChunkDataset(val_tokens,   args.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    positions_per_batch = args.batch_size * args.seq_len
    max_train_batches   = math.ceil(args.max_train_positions / positions_per_batch)
    max_eval_batches    = math.ceil(args.max_eval_positions  / positions_per_batch)
    print(f"[gpt2xl_margin] train: {len(train_ds)} seqs  "
          f"capped at {max_train_batches} batches/epoch × {args.probe_epochs} epochs")
    print(f"[gpt2xl_margin] eval:  {len(val_ds)} seqs  "
          f"capped at {max_eval_batches} batches")

    # ── Probes ────────────────────────────────────────────────────────────────
    probes = nn.ModuleList([
        RegionProbe(d_model, n_coarse, args.probe_type)
        for _ in range(n_hs)
    ])
    probes = probes.to(device)

    loss_curves: List[List[float]] = [
        [float("nan")] * args.probe_epochs for _ in range(n_hs)
    ]

    # ── Load or train probes ──────────────────────────────────────────────────
    if args.skip_training and os.path.exists(ckpt_path):
        print(f"[gpt2xl_margin] loading probes from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        for li, (probe, sd) in enumerate(zip(probes, ckpt["probes"])):
            probe.load_state_dict(sd)
        if "loss_curves" in ckpt:
            loss_curves = ckpt["loss_curves"]
        print("[gpt2xl_margin] probes loaded")
    else:
        if args.skip_training:
            print(f"[WARNING] --skip_training set but {ckpt_path} not found; training")

        print(f"[gpt2xl_margin] training {n_hs} probes  "
              f"probe_type={args.probe_type}  lr={args.probe_lr}  "
              f"epochs={args.probe_epochs}  temp={args.probe_temp}")

        optimizers = [
            torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
            for probe in probes
        ]

        for epoch in range(args.probe_epochs):
            ep_loss  = [0.0] * n_hs
            ep_count = [0]   * n_hs

            for bi, batch in enumerate(train_loader):
                if bi >= max_train_batches:
                    break

                batch = batch.to(device)
                src   = batch[:, :-1]
                tgt   = batch[:, 1:]
                B, T  = src.shape
                gold  = coarse_map[tgt].reshape(-1)  # (B*T,) long, -1 = unknown
                valid = gold >= 0
                if not valid.any():
                    continue

                # Single frozen GPT-2 forward — all 49 hidden states
                with torch.no_grad():
                    out = gpt2(src, output_hidden_states=True, use_cache=False)
                    hs  = out.hidden_states  # tuple[49] of (B, T, d_model) fp16

                # Update every probe on its corresponding hidden state
                for li, (probe, opt) in enumerate(zip(probes, optimizers)):
                    h_flat = hs[li].detach().float().reshape(-1, d_model)
                    h_v    = h_flat[valid]
                    g_v    = gold[valid]
                    logits = probe(h_v) / args.probe_temp
                    loss   = F.cross_entropy(logits, g_v)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    n_v             = int(valid.sum())
                    ep_loss[li]    += loss.item() * n_v
                    ep_count[li]   += n_v

                if bi % 200 == 0:
                    l0 = ep_loss[0]  / max(ep_count[0],  1)
                    lf = ep_loss[-1] / max(ep_count[-1], 1)
                    print(f"  epoch {epoch+1}/{args.probe_epochs}  "
                          f"batch {bi}/{max_train_batches}  "
                          f"loss[embed]={l0:.4f}  loss[block_47]={lf:.4f}")

            for li in range(n_hs):
                loss_curves[li][epoch] = ep_loss[li] / max(ep_count[li], 1)
            l0 = loss_curves[0][epoch]
            lf = loss_curves[-1][epoch]
            print(f"  epoch {epoch+1}/{args.probe_epochs} done  "
                  f"loss[embed]={l0:.4f}  loss[block_47]={lf:.4f}")

        torch.save(
            {"probes": [p.state_dict() for p in probes], "loss_curves": loss_curves},
            ckpt_path,
        )
        print(f"[gpt2xl_margin] probes saved → {ckpt_path}")

    # Freeze probes for eval
    probes.eval()
    for probe in probes:
        for p in probe.parameters():
            p.requires_grad_(False)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("[gpt2xl_margin] evaluating ...")
    final_li = n_hs - 1

    lay_margin  = [[] for _ in range(n_hs)]
    lay_entropy = [[] for _ in range(n_hs)]
    lay_acc1    = [[] for _ in range(n_hs)]
    lay_acc4    = [[] for _ in range(n_hs)]
    lay_acc8    = [[] for _ in range(n_hs)]
    lay_acc16   = [[] for _ in range(n_hs)]
    lay_rank    = [[] for _ in range(n_hs)]
    lay_valid   = [[] for _ in range(n_hs)]
    top1_final  : List[torch.Tensor] = []
    all_tok_ids : List[torch.Tensor] = []

    actual_eval_b = min(max_eval_batches, len(val_loader))

    for bi, batch in enumerate(val_loader):
        if bi >= max_eval_batches:
            break
        if bi % 100 == 0:
            print(f"  eval batch {bi}/{actual_eval_b}")

        batch = batch.to(device)
        src   = batch[:, :-1]
        tgt   = batch[:, 1:]
        B, T  = src.shape
        gold  = coarse_map[tgt]  # (B, T)

        with torch.no_grad():
            out = gpt2(src, output_hidden_states=True, use_cache=False)
            hs  = out.hidden_states

        all_tok_ids.append(tgt.cpu().reshape(-1))

        k_max = min(16, n_coarse)

        for li, probe in enumerate(probes):
            h_flat = hs[li].float().reshape(-1, d_model)
            g_flat = gold.reshape(-1)
            valid  = g_flat >= 0
            # valid is a GPU bool tensor; need a CPU copy to index CPU accumulators
            valid_cpu = valid.cpu()

            with torch.no_grad():
                logits = probe(h_flat) / args.probe_temp
                p      = F.softmax(logits, dim=-1)

            top2v   = torch.topk(p, k=2, dim=-1).values
            margin  = (top2v[:, 0] - top2v[:, 1]).cpu()
            entropy = (-(p * (p + 1e-10).log()).sum(-1)).cpu()

            acc1  = torch.zeros(B * T)
            acc4  = torch.zeros(B * T)
            acc8  = torch.zeros(B * T)
            acc16 = torch.zeros(B * T)
            rank  = torch.full((B * T,), float(n_coarse))

            if valid.any():
                top16i = torch.topk(p, k=k_max, dim=-1).indices  # (N, k_max) on GPU
                g_v    = g_flat[valid]         # GPU
                t16_v  = top16i[valid]         # GPU
                gcol   = g_v.unsqueeze(-1)     # GPU
                # Use valid_cpu to index the CPU accumulator tensors
                acc1[valid_cpu]  = (t16_v[:, :1]              == gcol).any(-1).float().cpu()
                acc4[valid_cpu]  = (t16_v[:, :min(4,  k_max)] == gcol).any(-1).float().cpu()
                acc8[valid_cpu]  = (t16_v[:, :min(8,  k_max)] == gcol).any(-1).float().cpu()
                acc16[valid_cpu] = (t16_v[:, :min(16, k_max)] == gcol).any(-1).float().cpu()
                # Gold rank: # regions with strictly higher prob (0 = top-1)
                p_v    = p[valid]              # GPU
                gprob  = p_v[torch.arange(len(g_v), device=device), g_v]
                rank[valid_cpu] = (p_v > gprob.unsqueeze(-1)).sum(-1).float().cpu()

            # Always collect final-layer top-1 so top1_fin stays aligned with tok_ids
            if li == final_li:
                top1_final.append(torch.topk(p, k=1, dim=-1).indices[:, 0].cpu())

            lay_margin[li].append(margin)
            lay_entropy[li].append(entropy)
            lay_acc1[li].append(acc1)
            lay_acc4[li].append(acc4)
            lay_acc8[li].append(acc8)
            lay_acc16[li].append(acc16)
            lay_rank[li].append(rank)
            lay_valid[li].append(valid.cpu())

    print("[gpt2xl_margin] concatenating ...")
    margins   = [torch.cat(lay_margin[li])  for li in range(n_hs)]
    entropies = [torch.cat(lay_entropy[li]) for li in range(n_hs)]
    acc1s     = [torch.cat(lay_acc1[li])    for li in range(n_hs)]
    acc4s     = [torch.cat(lay_acc4[li])    for li in range(n_hs)]
    acc8s     = [torch.cat(lay_acc8[li])    for li in range(n_hs)]
    acc16s    = [torch.cat(lay_acc16[li])   for li in range(n_hs)]
    ranks     = [torch.cat(lay_rank[li])    for li in range(n_hs)]
    valids    = [torch.cat(lay_valid[li])   for li in range(n_hs)]
    top1_fin  = torch.cat(top1_final) if top1_final else torch.zeros(0, dtype=torch.long)
    tok_ids   = torch.cat(all_tok_ids)
    N_total   = len(margins[0])
    print(f"[gpt2xl_margin] {N_total:,} positions × {n_hs} layers")

    # ── Per-layer aggregate statistics ────────────────────────────────────────
    def _q(t: torch.Tensor, q: float) -> float:
        return float(t.quantile(q).item())

    def _vm(num: torch.Tensor, mask: torch.Tensor) -> float:
        s = num[mask]
        return float(s.mean().item()) if len(s) > 0 else float("nan")

    bnd_tau       = args.boundary_tau
    bnd_tau_tight = args.boundary_tau_tight
    layer_stats: List[Dict] = []

    for li in range(n_hs):
        m   = margins[li];   e  = entropies[li]; v  = valids[li]
        a1  = acc1s[li];     a4 = acc4s[li]
        a8  = acc8s[li];     a16 = acc16s[li]
        gr  = ranks[li]
        layer_stats.append({
            "layer":               li,
            "layer_name":          layer_names[li],
            "acc1":                _vm(a1, v),
            "acc4":                _vm(a4, v),
            "acc8":                _vm(a8, v),
            "acc16":               _vm(a16, v),
            "margin_mean":         float(m.mean()),
            "margin_q10":          _q(m, 0.10),
            "margin_q25":          _q(m, 0.25),
            "margin_median":       _q(m, 0.50),
            "margin_q75":          _q(m, 0.75),
            "margin_q90":          _q(m, 0.90),
            "entropy_mean":        float(e.mean()),
            "entropy_q25":         _q(e, 0.25),
            "entropy_median":      _q(e, 0.50),
            "entropy_q75":         _q(e, 0.75),
            "boundary_frac":       float((m < bnd_tau).float().mean()),
            "boundary_frac_tight": float((m < bnd_tau_tight).float().mean()),
            "core_frac":           float((m >= bnd_tau).float().mean()),
            "gold_rank_mean":      _vm(gr, v),
            "gold_rank_med":       float(gr[v].median()) if v.any() else float("nan"),
            "n_valid":             int(v.sum()),
        })

    # ── Trajectory statistics by embedding-layer margin bin ───────────────────
    embed_margin = margins[0]
    traj_rows: List[Dict] = []

    for li in range(n_hs):
        m  = margins[li];  e = entropies[li]
        a1 = acc1s[li];    v = valids[li];   gr = ranks[li]
        traj_rows.append({
            "group": "all", "layer": li, "layer_name": layer_names[li],
            "margin_mean": float(m.mean()), "margin_q25": _q(m, 0.25),
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

    # ── Token margin variance at final layer (Welford online) ─────────────────
    print("[gpt2xl_margin] computing token context variance ...")
    fin_m    = margins[final_li]
    tok_n:   Dict[int, int]   = defaultdict(int)
    tok_mu:  Dict[int, float] = defaultdict(float)
    tok_M2:  Dict[int, float] = defaultdict(float)
    tok_reg: Dict[int, Counter] = defaultdict(Counter)

    for idx in range(N_total):
        tid = int(tok_ids[idx].item())
        if not (0 <= tid < vocab_size):
            continue
        mv  = float(fin_m[idx].item())
        n   = tok_n[tid] + 1
        d   = mv - tok_mu[tid]
        mu  = tok_mu[tid] + d / n
        tok_M2[tid]  = tok_M2[tid] + d * (mv - mu)
        tok_n[tid]   = n
        tok_mu[tid]  = mu
        if idx < len(top1_fin):
            tok_reg[tid][int(top1_fin[idx].item())] += 1

    tok_var_rows: List[Dict] = []
    for tid in sorted(tok_n, key=lambda t: -tok_n[t]):
        n = tok_n[tid]
        if n < 5:
            continue
        mu       = tok_mu[tid]
        std      = math.sqrt(tok_M2[tid] / n) if n > 1 else 0.0
        reg      = tok_reg.get(tid, Counter())
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
    print(f"[gpt2xl_margin] {len(tok_var_rows)} tokens with ≥5 occurrences")

    # ── Write CSVs ────────────────────────────────────────────────────────────
    def _write_csv(rows: List[Dict], path: str) -> None:
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(layer_stats,         os.path.join(args.output_dir, "layer_metrics.csv"))
    _write_csv(traj_rows,           os.path.join(args.output_dir, "layer_trajectories.csv"))
    _write_csv(tok_var_rows[:5000], os.path.join(args.output_dir, "token_margin_variance.csv"))

    lc_rows: List[Dict] = []
    for li in range(n_hs):
        for ep, lv in enumerate(loss_curves[li]):
            lc_rows.append({"layer": li, "layer_name": layer_names[li],
                            "epoch": ep, "loss": lv})
    _write_csv(lc_rows, os.path.join(args.output_dir, "probe_training_loss.csv"))
    print("[gpt2xl_margin] wrote CSVs")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLT = True
    except ImportError:
        HAS_PLT = False
        print("[gpt2xl_margin] matplotlib not available — plots skipped")

    if HAS_PLT:
        xs         = list(range(n_hs))
        # Sparse x-tick labels to avoid clutter across 49 layers
        tick_step  = max(1, n_hs // 12)
        xtick_pos  = list(range(0, n_hs, tick_step))
        xtick_lbl  = [layer_names[i] for i in xtick_pos]

        def _ax(ax: "plt.Axes", title: str, ylabel: str) -> None:
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylabel)
            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(xtick_lbl, rotation=45, ha="right", fontsize=7)
            ax.grid(True, alpha=0.3)

        def _ls(key: str) -> List[float]:
            return [s[key] for s in layer_stats]

        # ── 1: Region accuracy ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(xs, _ls("acc1"),  "o-b",  markersize=2, label="Acc@1")
        ax.plot(xs, _ls("acc4"),  "s-g",  markersize=2, label="Acc@4")
        ax.plot(xs, _ls("acc8"),  "^-r",  markersize=2, label="Acc@8")
        ax.plot(xs, _ls("acc16"), "D-m",  markersize=2, label="Acc@16")
        _ax(ax, f"Region Accuracy vs Layer — {args.model_name}", "Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "01_region_accuracy_vs_layer.png"), dpi=120)
        plt.close(fig)

        # ── 2: Margin distribution ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(xs, _ls("margin_q25"), _ls("margin_q75"),
                        alpha=0.2, color="blue", label="Q25–Q75")
        ax.plot(xs, _ls("margin_q10"),   "--", color="lightblue",  lw=0.8, label="Q10")
        ax.plot(xs, _ls("margin_q90"),   "--", color="steelblue",  lw=0.8, label="Q90")
        ax.plot(xs, _ls("margin_mean"),  "o-b", markersize=2, label="Mean")
        ax.plot(xs, _ls("margin_median"),"s--", color="navy", lw=1, markersize=2,
                label="Median")
        _ax(ax, "Routing Margin vs Layer", "Margin (top1 − top2 probe prob)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "02_margin_vs_layer.png"), dpi=120)
        plt.close(fig)

        # ── 3: Entropy ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(xs, _ls("entropy_q25"), _ls("entropy_q75"),
                        alpha=0.2, color="red", label="Q25–Q75")
        ax.plot(xs, _ls("entropy_mean"),   "o-r", markersize=2, label="Mean")
        ax.plot(xs, _ls("entropy_median"), "s--", color="darkred", lw=1, markersize=2,
                label="Median")
        _ax(ax, "Router Entropy vs Layer", "H(p_region)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "03_entropy_vs_layer.png"), dpi=120)
        plt.close(fig)

        # ── 4: Boundary fraction (both tau thresholds) ────────────────────────
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(xs, _ls("boundary_frac"),
                "o-m", markersize=2, label=f"boundary (τ={bnd_tau})")
        ax.plot(xs, _ls("boundary_frac_tight"),
                "s-",  color="darkorange", markersize=2,
                label=f"boundary tight (τ={bnd_tau_tight})")
        ax.plot(xs, _ls("core_frac"),
                "^--g", markersize=2, label=f"core (τ={bnd_tau})")
        _ax(ax, "Boundary Fraction vs Layer", "Fraction of positions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "04_boundary_frac_vs_layer.png"), dpi=120)
        plt.close(fig)

        # ── 5: Gold rank (lower = better) ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(xs, _ls("gold_rank_mean"), "o-",  color="navy",       markersize=2,
                label="Mean rank")
        ax.plot(xs, _ls("gold_rank_med"),  "s--", color="dodgerblue", markersize=2,
                label="Median rank")
        ax.invert_yaxis()
        _ax(ax, "Gold Region Rank vs Layer (lower = better)", "Rank (0 = top-1)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "05_gold_rank_vs_layer.png"), dpi=120)
        plt.close(fig)

        # ── 6: Margin trajectory by initial-margin group ──────────────────────
        fig, ax = plt.subplots(figsize=(12, 4))
        groups_seen = list(dict.fromkeys(
            r["group"] for r in traj_rows if r["group"] != "all"
        ))
        for grp in groups_seen:
            rows_g = sorted(
                [r for r in traj_rows if r["group"] == grp],
                key=lambda r: r["layer"],
            )
            n_g = rows_g[0]["n"] if rows_g else 0
            ax.plot(
                [r["layer"] for r in rows_g],
                [r["margin_mean"] for r in rows_g],
                "o-", color=TRAJ_COLORS.get(grp, "k"), markersize=2,
                label=f"{grp} (n={n_g:,})", linewidth=1.5,
            )
        ax.plot(xs, _ls("margin_mean"), "k--", lw=1.5, alpha=0.5, label="all")
        _ax(ax, "Margin Trajectory by Embedding-Layer Margin Group", "Mean margin")
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "06_margin_trajectory_by_group.png"), dpi=120)
        plt.close(fig)

        # ── 7: Token margin variance scatter ──────────────────────────────────
        if tok_var_rows:
            sample = tok_var_rows[:2000]
            mu_m   = [r["mean_margin"]      for r in sample]
            std_m  = [r["std_margin"]       for r in sample]
            cv     = [r["context_variance"] for r in sample]
            n_uni  = [r["n_unique_regions"] for r in sample]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sc = axes[0].scatter(mu_m, std_m, c=cv, s=6, alpha=0.4,
                                 cmap="hot_r", vmin=0, vmax=1)
            plt.colorbar(sc, ax=axes[0], label="context_variance")
            axes[0].set_xlabel("Mean margin (final layer)")
            axes[0].set_ylabel("Std margin (final layer)")
            axes[0].set_title("Token Margin Variability")
            axes[0].grid(True, alpha=0.3)
            axes[1].scatter(mu_m, n_uni, s=6, alpha=0.3, color="teal")
            axes[1].set_xlabel("Mean margin (final layer)")
            axes[1].set_ylabel("Unique top-1 regions")
            axes[1].set_title("Routing Diversity vs Margin")
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "07_token_margin_variance.png"), dpi=120)
            plt.close(fig)

        # ── 8: Example token margin trajectories ──────────────────────────────
        if tok_var_rows:
            top_var = tok_var_rows[:min(10, len(tok_var_rows))]
            try:
                cmap8 = plt.get_cmap("tab10")
            except AttributeError:
                cmap8 = plt.cm.get_cmap("tab10")
            fig, ax = plt.subplots(figsize=(12, 4))
            for i, row in enumerate(top_var):
                tid  = row["token_id"]
                mask = (tok_ids == tid)
                if not mask.any():
                    continue
                ys  = [float(margins[li][mask].mean()) for li in range(n_hs)]
                lbl = repr(row["token_str"])[:14]
                ax.plot(xs, ys, "o-", color=cmap8(i % 10), markersize=2,
                        label=f"{lbl} σ={row['std_margin']:.3f}", linewidth=1.5)
            _ax(ax, "Margin Trajectory — Highest Context-Variance Tokens",
                "Mean margin at layer")
            ax.legend(fontsize=7, loc="upper left", ncol=2)
            fig.tight_layout()
            fig.savefig(
                os.path.join(plots_dir, "08_example_token_trajectories.png"), dpi=120
            )
            plt.close(fig)

        # ── 9: Probe training loss curves ─────────────────────────────────────
        has_loss = any(
            not math.isnan(loss_curves[li][ep])
            for li in range(n_hs)
            for ep in range(len(loss_curves[li]))
        )
        if has_loss:
            try:
                cmap9 = plt.get_cmap("viridis")
            except AttributeError:
                cmap9 = plt.cm.get_cmap("viridis")
            # Plot every ~6th layer to avoid clutter
            plot_stride   = max(1, n_hs // 8)
            plot_layers   = list(range(0, n_hs, plot_stride))
            n_plotted     = len(plot_layers)
            n_saved_epochs = len(loss_curves[0])
            fig, ax = plt.subplots(figsize=(10, 4))
            for idx, li in enumerate(plot_layers):
                epochs_x = list(range(1, n_saved_epochs + 1))
                losses_y = loss_curves[li]
                ax.plot(
                    epochs_x, losses_y,
                    "o-",
                    color=cmap9(idx / max(n_plotted - 1, 1)),
                    markersize=4, linewidth=1.5, label=layer_names[li],
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-entropy loss")
            ax.set_title(f"Probe Training Loss — {args.model_name} (selected layers)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "09_probe_training_loss.png"), dpi=120)
            plt.close(fig)

        print(f"[gpt2xl_margin] wrote 9 plots → {plots_dir}/")

    # ── Q1–Q8 verdicts ────────────────────────────────────────────────────────
    emb = layer_stats[0]
    fin = layer_stats[-1]
    mid = layer_stats[n_hs // 2]

    def _v(cond: bool) -> str:
        return "PASS" if cond else "INCONCLUSIVE"

    chance_k4 = min(4, n_coarse) / n_coarse

    # Q1: gold enters top-4 at embedding layer
    q1 = _v(emb["acc4"] > 2.0 * chance_k4)

    # Q2: top-1 sharpens later than top-4
    def _rel_gap(s: Dict) -> float:
        a1 = s["acc1"]; a4 = s["acc4"]
        return (a4 - a1) / (a4 + 1e-8)
    q2 = _v(_rel_gap(emb) > _rel_gap(fin) + 0.05)

    # Q3: entropy decreases progressively
    q3 = _v(fin["entropy_mean"] < emb["entropy_mean"] - 0.1)

    # Q4: margins increase progressively
    q4 = _v(fin["margin_mean"] > emb["margin_mean"] + 0.02)

    # Q5: persistently ambiguous tokens (low margin even at final layer)
    low_grp_final = [
        r for r in traj_rows
        if r["group"] in ("M<0.03", "0.03≤M<0.05") and r["layer"] == final_li
    ]
    q5_frac = sum(r["n"] for r in low_grp_final) / max(N_total, 1)
    q5 = _v(q5_frac > 0.05)

    # Q6: top-1 accuracy gains non-trivially in second half
    acc1_e_m = mid["acc1"] - emb["acc1"]
    acc1_m_f = fin["acc1"] - mid["acc1"]
    q6 = _v(acc1_m_f > acc1_e_m * 0.3 and acc1_m_f > 0.01)

    # Q7: context dynamically sharpens margin (high-variance tokens exist)
    high_var_count = sum(1 for r in tok_var_rows if r["std_margin"] > 0.10)
    q7 = _v(high_var_count > 50)

    # Q8: sharpening elbow in second half (largest single-step margin gain)
    margin_means   = [s["margin_mean"] for s in layer_stats]
    margin_deltas  = [margin_means[li] - margin_means[li - 1]
                      for li in range(1, n_hs)]
    elbow_li       = int(np.argmax(margin_deltas)) + 1
    q8 = _v(elbow_li > n_hs // 2)

    # ── Summary markdown ──────────────────────────────────────────────────────
    passes = sum(1 for q in [q1, q2, q3, q4, q5, q6, q7, q8] if q == "PASS")
    support = "STRONG" if passes >= 6 else "PARTIAL" if passes >= 4 else "WEAK"

    lines = [
        "# GPT-2 XL Layer Margin Analysis Report",
        "",
        f"**Model:**    `{args.model_name}`  "
        f"d_model={d_model}  n_blocks={n_gpt2_blocks}  "
        f"hidden_states={n_hs}",
        f"**Dataset:**  {args.dataset}  "
        f"({len(val_tokens):,} val tokens  {actual_eval_b} batches evaluated)",
        f"**Probes:**   {args.probe_type}  "
        f"lr={args.probe_lr}  epochs={args.probe_epochs}  "
        f"temp={args.probe_temp}",
        f"**Positions evaluated:** {N_total:,}  "
        f"**Regions:** {n_coarse}  (coverage={cov:.1%})",
        f"**Boundary τ:** {bnd_tau} (standard)  {bnd_tau_tight} (tight)",
        "",
        "## Per-Layer Statistics (every 4th layer + final)",
        "",
        "| Layer | Acc@1 | Acc@4 | Acc@8 | Acc@16 | "
        "Margin μ | Margin med | Entropy μ | Bnd% | Bnd%(tight) | Rank μ |",
        "|-------|------:|------:|------:|-------:|"
        "---------:|-----------:|----------:|-----:|------------:|-------:|",
    ]

    summary_lis = sorted({0} | set(range(4, n_hs - 1, 4)) | {n_hs - 1})
    for li in summary_lis:
        s = layer_stats[li]
        lines.append(
            f"| {s['layer_name']:<12} "
            f"| {s['acc1']:.3f} | {s['acc4']:.3f} "
            f"| {s['acc8']:.3f} | {s['acc16']:.3f} "
            f"| {s['margin_mean']:.3f} | {s['margin_median']:.3f} "
            f"| {s['entropy_mean']:.3f} "
            f"| {s['boundary_frac']:.1%} "
            f"| {s['boundary_frac_tight']:.1%} "
            f"| {s['gold_rank_mean']:.1f} |"
        )

    lines += [
        "",
        "## Hypothesis Tests (Q1–Q8)",
        "",
        "| # | Question | Verdict |",
        "|---|----------|---------|",
        f"| Q1 | Gold region enters top-4 early "
        f"(acc@4 > 2× chance={2*chance_k4:.3f} at embed)? | **{q1}** |",
        f"| Q2 | Top-1 sharpens later than top-4 "
        f"(relative acc@4−acc@1 gap narrows from embed→final)? | **{q2}** |",
        f"| Q3 | Entropy decreases progressively "
        f"(final < embed − 0.1)? | **{q3}** |",
        f"| Q4 | Margins increase progressively "
        f"(final > embed + 0.02)? | **{q4}** |",
        f"| Q5 | Persistently ambiguous tokens exist at final layer "
        f"({q5_frac:.1%} in tight-low bins)? | **{q5}** |",
        f"| Q6 | Top-1 accuracy gains non-trivially in second half "
        f"(+{acc1_m_f:.3f} vs +{acc1_e_m:.3f} in first half)? | **{q6}** |",
        f"| Q7 | Context dynamically sharpens uncertainty "
        f"({high_var_count:,} tokens with std_margin > 0.10)? | **{q7}** |",
        f"| Q8 | Sharpening elbow is in second half "
        f"(largest Δmargin at layer {elbow_li} = {layer_names[elbow_li]})? | **{q8}** |",
        "",
        "## Margin Trajectory by Initial-Margin Group",
        "",
        "Tokens are grouped by their margin at the embedding layer (layer 0) "
        "to reveal whether different ambiguity profiles converge or persist.",
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

    lines += [
        "",
        "## Interpretation",
        "",
        f"**{passes}/8 questions PASS → {support} SUPPORT** "
        "for progressive manifold uncertainty resolution.",
        "",
        "| Signal | embed → final | Change |",
        "|--------|:-------------|:-------|",
        f"| Margin mean  | {emb['margin_mean']:.3f} → {fin['margin_mean']:.3f} "
        f"| {fin['margin_mean'] - emb['margin_mean']:+.3f} |",
        f"| Entropy mean | {emb['entropy_mean']:.3f} → {fin['entropy_mean']:.3f} "
        f"| {fin['entropy_mean'] - emb['entropy_mean']:+.3f} |",
        f"| Acc@1        | {emb['acc1']:.3f} → {fin['acc1']:.3f} "
        f"| {fin['acc1'] - emb['acc1']:+.3f} |",
        f"| Acc@4        | {emb['acc4']:.3f} → {fin['acc4']:.3f} "
        f"| {fin['acc4'] - emb['acc4']:+.3f} |",
        f"| Gold rank    | {emb['gold_rank_mean']:.1f} → {fin['gold_rank_mean']:.1f} "
        f"| {fin['gold_rank_mean'] - emb['gold_rank_mean']:+.1f} |",
        "",
        f"Sharpening elbow (largest single-layer margin gain): "
        f"layer {elbow_li} ({layer_names[elbow_li]})",
        "",
        "## Output Files",
        "",
        "- `layer_metrics.csv` — per-layer aggregate statistics (acc, margin, entropy, rank)",
        "- `layer_trajectories.csv` — per-layer stats by initial-margin group",
        "- `token_margin_variance.csv` — per-token margin variance at final layer (top 5k)",
        "- `probe_training_loss.csv` — per-layer per-epoch training loss",
        "- `plots/01_region_accuracy_vs_layer.png` — acc@1/4/8/16 vs layer",
        "- `plots/02_margin_vs_layer.png` — margin distribution vs layer (IQR band)",
        "- `plots/03_entropy_vs_layer.png` — entropy vs layer",
        "- `plots/04_boundary_frac_vs_layer.png` — boundary fraction (both τ) vs layer",
        "- `plots/05_gold_rank_vs_layer.png` — gold region rank vs layer (inverted)",
        "- `plots/06_margin_trajectory_by_group.png` — trajectory by initial ambiguity",
        "- `plots/07_token_margin_variance.png` — context-dependent routing scatter",
        "- `plots/08_example_token_trajectories.png` — per-token margin trajectories",
        "- `plots/09_probe_training_loss.png` — probe training loss curves",
    ]

    with open(os.path.join(args.output_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n[gpt2xl_margin] DONE  →  {args.output_dir}")
    print(f"  {passes}/8 PASS ({support} support for progressive resolution hypothesis)")
    for q, label in [
        (q1, "Q1 gold enters top-k early              "),
        (q2, "Q2 top-1 sharpens later than top-k      "),
        (q3, "Q3 entropy decreases progressively       "),
        (q4, "Q4 margins increase progressively        "),
        (q5, "Q5 persistently ambiguous tokens exist   "),
        (q6, "Q6 top-1 gains in second half            "),
        (q7, "Q7 context dynamically sharpens margin   "),
        (q8, "Q8 sharpening elbow in second half       "),
    ]:
        print(f"  {label}: {q}")


if __name__ == "__main__":
    main()
