#!/usr/bin/env python3
"""
Phase 3 — Path-conditioned refiner training.

Loads shards produced by build_path_refiner_dataset.py, freezes the backbone
(used only to provide token_emb for base logit computation), and trains one of
three refiner variants (A/B/C).

Key constraints:
  - Base model is never trained; only the refiner parameters are updated.
  - Local CE loss is computed ONLY on positions where gold is in the candidate
    set (covered=True). Uncovered positions contribute to fallback rate metrics
    but not to gradient updates.
  - Base candidate logit (h_prime @ token_emb[cand_tok]) is always included
    in the final score — the refiner learns a residual on top of it.
  - Scores are over the padded candidate set [B, C], never a full-vocab tensor.

Training metrics (per epoch, per split):
  covered_ce        — CE loss on covered positions only
  fallback_rate     — fraction of eval positions uncovered
  ppl_improvement   — exp(full_lm_nll) - exp(fallback_nll)  [check if refiner hurts]

Usage:
    python scripts/train_path_refiner.py \\
        --shard_dir    runs/path_refiner/dataset \\
        --small_ckpt   runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --variant      B \\
        --output_dir   runs/path_refiner/variant_B \\
        --device       cuda

    # Variant A (bias-only, fast):
    python scripts/train_path_refiner.py ... --variant A

    # Variant C (MLP):
    python scripts/train_path_refiner.py ... --variant C --d_hidden 256
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe


# ── Refiner variants (inlined from models/path_refiner.py) ───────────────────

class BiasOnlyRefiner(nn.Module):
    def __init__(self, n_fine: int, n_super: int) -> None:
        super().__init__()
        self.fine_bias  = nn.Embedding(n_fine  + 1, 1, padding_idx=n_fine)
        self.super_bias = nn.Embedding(n_super + 1, 1, padding_idx=n_super)
        nn.init.zeros_(self.fine_bias.weight)
        nn.init.zeros_(self.super_bias.weight)

    def forward(self, h_prime, cand_tok, cand_fine, cand_super, cand_mask, token_emb):
        tok_e  = token_emb(cand_tok.clamp(min=0))
        base   = (h_prime.unsqueeze(1) * tok_e).sum(-1)
        f_bias = self.fine_bias(cand_fine.clamp(min=0)).squeeze(-1)
        s_bias = self.super_bias(cand_super.clamp(min=0)).squeeze(-1)
        return (base + f_bias + s_bias).masked_fill(~cand_mask, -1e9)


class DotProductRefiner(nn.Module):
    def __init__(self, d_model: int, n_fine: int, n_super: int, d_head: int = 64) -> None:
        super().__init__()
        self.d_head    = d_head
        self.Wq        = nn.Linear(d_model, d_head, bias=False)
        self.Wk        = nn.Linear(d_model, d_head, bias=False)
        self.super_emb = nn.Embedding(n_super + 1, d_model, padding_idx=n_super)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.Wq.weight, std=0.01)
        nn.init.normal_(self.Wk.weight, std=0.01)
        nn.init.zeros_(self.super_emb.weight)

    def forward(self, h_prime, cand_tok, cand_fine, cand_super, cand_mask, token_emb):
        tok_e    = token_emb(cand_tok.clamp(min=0))
        base     = (h_prime.unsqueeze(1) * tok_e).sum(-1)
        s_emb    = self.super_emb(cand_super.clamp(min=0))
        mask_f   = cand_mask.float().unsqueeze(-1)
        path_ctx = (s_emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        q        = self.Wq(h_prime + path_ctx)
        k        = self.Wk(tok_e)
        residual = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(self.d_head)
        return (base + self.log_scale.exp() * residual).masked_fill(~cand_mask, -1e9)


class MLPRefiner(nn.Module):
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

    def forward(self, h_prime, cand_tok, cand_fine, cand_super, cand_mask, token_emb):
        B, C   = cand_tok.shape
        tok_e  = token_emb(cand_tok.clamp(min=0))
        fine_e = self.fine_emb(cand_fine.clamp(min=0))
        sup_e  = self.super_emb(cand_super.clamp(min=0))
        h_exp  = h_prime.unsqueeze(1).expand(-1, C, -1)
        base   = (h_prime.unsqueeze(1) * tok_e).sum(-1)
        feat   = torch.cat([h_exp, tok_e, fine_e, sup_e], dim=-1)
        delta  = self.mlp(feat).squeeze(-1)
        return (base + delta).masked_fill(~cand_mask, -1e9)


def build_refiner(variant: str, d_model: int, n_fine: int, n_super: int,
                  **kwargs) -> nn.Module:
    v = variant.upper()
    if v == "A":
        return BiasOnlyRefiner(n_fine=n_fine, n_super=n_super)
    if v == "B":
        return DotProductRefiner(d_model=d_model, n_fine=n_fine, n_super=n_super, **kwargs)
    if v == "C":
        return MLPRefiner(d_model=d_model, n_fine=n_fine, n_super=n_super, **kwargs)
    raise ValueError(f"Unknown variant {variant!r}; choose A, B, or C.")

SPLIT_NAMES = {0: "core", 1: "medium", 2: "boundary", 3: "tight_boundary"}


# ── Shard dataset ─────────────────────────────────────────────────────────────

class ShardDataset(Dataset):
    """
    Loads all shards from shard_dir. Returns ALL positions (covered and not),
    so the training loop can compute fallback_rate. The loss function skips
    uncovered positions internally.
    """

    def __init__(self, shard_dir: str, split_filter: Optional[int] = None) -> None:
        paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
        if not paths:
            raise RuntimeError(f"No shard_*.npz files found in {shard_dir}")
        print(f"[dataset] loading {len(paths)} shards from {shard_dir} ...")
        t0 = time.time()

        arrays: Dict[str, List[np.ndarray]] = {k: [] for k in [
            "h_prime", "cand_tok", "cand_fine", "cand_super", "cand_mask",
            "gold_idx", "covered", "split", "type_arr",
        ]}
        for path in paths:
            d = np.load(path)
            for k in arrays:
                if k in d:
                    arrays[k].append(d[k])
                elif k == "type_arr":
                    arrays[k].append(np.zeros(len(d["covered"]), dtype=np.uint8))

        self.h_prime    = np.concatenate(arrays["h_prime"],    axis=0)  # (N, d)
        self.cand_tok   = np.concatenate(arrays["cand_tok"],   axis=0)  # (N, C)
        self.cand_fine  = np.concatenate(arrays["cand_fine"],  axis=0)  # (N, C)
        self.cand_super = np.concatenate(arrays["cand_super"], axis=0)  # (N, C)
        self.cand_mask  = np.concatenate(arrays["cand_mask"],  axis=0)  # (N, C) bool
        self.gold_idx   = np.concatenate(arrays["gold_idx"],   axis=0)  # (N,) int32
        self.covered    = np.concatenate(arrays["covered"],    axis=0)  # (N,) bool
        self.split      = np.concatenate(arrays["split"],      axis=0)  # (N,) uint8
        self.type_arr   = np.concatenate(arrays["type_arr"],   axis=0)  # (N,) uint8

        if split_filter is not None:
            keep = self.split == split_filter
            self.h_prime    = self.h_prime[keep]
            self.cand_tok   = self.cand_tok[keep]
            self.cand_fine  = self.cand_fine[keep]
            self.cand_super = self.cand_super[keep]
            self.cand_mask  = self.cand_mask[keep]
            self.gold_idx   = self.gold_idx[keep]
            self.covered    = self.covered[keep]
            self.split      = self.split[keep]
            self.type_arr   = self.type_arr[keep]

        N = len(self.covered)
        n_cov = int(self.covered.sum())
        print(f"[dataset] N={N:,}  covered={n_cov:,} ({n_cov/max(N,1):.3f})  "
              f"({time.time()-t0:.1f}s)")

    def __len__(self) -> int:
        return len(self.covered)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "h_prime":    torch.from_numpy(self.h_prime[idx].astype(np.float32)),
            "cand_tok":   torch.from_numpy(self.cand_tok[idx].astype(np.int64)),
            "cand_fine":  torch.from_numpy(self.cand_fine[idx].astype(np.int64)),
            "cand_super": torch.from_numpy(self.cand_super[idx].astype(np.int64)),
            "cand_mask":  torch.from_numpy(self.cand_mask[idx]),
            "gold_idx":   torch.tensor(int(self.gold_idx[idx]), dtype=torch.long),
            "covered":    torch.tensor(bool(self.covered[idx]),  dtype=torch.bool),
            "split":      torch.tensor(int(self.split[idx]),     dtype=torch.long),
            "type_arr":   torch.tensor(int(self.type_arr[idx]),  dtype=torch.long),
        }


# ── Training helpers ──────────────────────────────────────────────────────────

def compute_loss_and_metrics(
    refiner:   nn.Module,
    batch:     Dict[str, torch.Tensor],
    token_emb: nn.Embedding,
    device:    torch.device,
    use_amp:   bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns (loss, metrics_dict).
    Loss is CE on covered positions only; 0-dim zero tensor if none covered.
    """
    h_prime    = batch["h_prime"].to(device)      # (B, d)
    cand_tok   = batch["cand_tok"].to(device)     # (B, C)
    cand_fine  = batch["cand_fine"].to(device)    # (B, C)
    cand_super = batch["cand_super"].to(device)   # (B, C)
    cand_mask  = batch["cand_mask"].to(device)    # (B, C) bool
    gold_idx   = batch["gold_idx"].to(device)     # (B,) long
    covered    = batch["covered"].to(device)      # (B,) bool

    with autocast(enabled=use_amp):
        scores = refiner(h_prime, cand_tok, cand_fine, cand_super,
                         cand_mask, token_emb)     # (B, C)

    # Only compute CE where gold is in candidate set
    n_covered = int(covered.sum())
    if n_covered == 0:
        loss = scores.sum() * 0.0  # zero with grad
        metrics = {"covered_ce": float("nan"), "n_covered": 0,
                   "n_total": len(covered), "fallback_rate": 1.0}
        return loss, metrics

    scores_cov  = scores[covered]      # (n_covered, C)
    gold_cov    = gold_idx[covered]    # (n_covered,)
    loss        = F.cross_entropy(scores_cov, gold_cov)

    with torch.no_grad():
        preds_cov = scores_cov.argmax(-1)
        acc       = float((preds_cov == gold_cov).float().mean())

    metrics = {
        "covered_ce":    float(loss),
        "top1_acc":      acc,
        "n_covered":     n_covered,
        "n_total":       len(covered),
        "fallback_rate": 1.0 - n_covered / len(covered),
    }
    return loss, metrics


@torch.no_grad()
def eval_epoch(
    refiner:    nn.Module,
    loader:     DataLoader,
    token_emb:  nn.Embedding,
    device:     torch.device,
    use_amp:    bool,
) -> Dict[str, Dict]:
    """Returns nested dict: {split_name: {metric: value}}."""
    split_buckets: Dict[int, Dict[str, List]] = {
        sid: {"ce": [], "acc": [], "n_cov": 0, "n_total": 0}
        for sid in SPLIT_NAMES
    }
    all_ce: List[float] = []
    all_n_cov = 0
    all_n_tot = 0

    refiner.eval()
    for batch in loader:
        h_prime    = batch["h_prime"].to(device)
        cand_tok   = batch["cand_tok"].to(device)
        cand_fine  = batch["cand_fine"].to(device)
        cand_super = batch["cand_super"].to(device)
        cand_mask  = batch["cand_mask"].to(device)
        gold_idx   = batch["gold_idx"].to(device)
        covered    = batch["covered"].to(device)
        split_ids  = batch["split"]               # (B,) on cpu

        with autocast(enabled=use_amp):
            scores = refiner(h_prime, cand_tok, cand_fine, cand_super,
                              cand_mask, token_emb)  # (B, C)

        B = len(covered)
        for sid in SPLIT_NAMES:
            sel = (split_ids == sid)
            if not sel.any():
                continue
            sel_dev  = sel.to(device)
            cov_sel  = covered[sel_dev]
            n_cov    = int(cov_sel.sum())
            split_buckets[sid]["n_total"] += int(sel.sum())
            split_buckets[sid]["n_cov"]   += n_cov
            if n_cov == 0:
                continue
            sc_sel   = scores[sel_dev][cov_sel]
            gi_sel   = gold_idx[sel_dev][cov_sel]
            ce_val   = float(F.cross_entropy(sc_sel, gi_sel))
            preds    = sc_sel.argmax(-1)
            acc_val  = float((preds == gi_sel).float().mean())
            split_buckets[sid]["ce"].append(ce_val * n_cov)
            split_buckets[sid]["acc"].append(acc_val * n_cov)

        if covered.any():
            sc_cov = scores[covered]
            gi_cov = gold_idx[covered]
            all_ce.append(float(F.cross_entropy(sc_cov, gi_cov)) * int(covered.sum()))
        all_n_cov += int(covered.sum())
        all_n_tot += B

    refiner.train()
    results: Dict[str, Dict] = {}

    def _agg_bucket(bucket: Dict) -> Dict:
        n_c = bucket["n_cov"]
        n_t = bucket["n_total"]
        return {
            "covered_ce":    sum(bucket["ce"])  / n_c if n_c > 0 else float("nan"),
            "top1_acc":      sum(bucket["acc"]) / n_c if n_c > 0 else float("nan"),
            "fallback_rate": 1.0 - n_c / n_t   if n_t > 0 else float("nan"),
            "n_covered":     n_c,
            "n_total":       n_t,
        }

    results["all"] = {
        "covered_ce":    sum(all_ce)  / all_n_cov if all_n_cov > 0 else float("nan"),
        "fallback_rate": 1.0 - all_n_cov / all_n_tot if all_n_tot > 0 else float("nan"),
        "n_covered":     all_n_cov,
        "n_total":       all_n_tot,
    }
    for sid, sname in SPLIT_NAMES.items():
        results[sname] = _agg_bucket(split_buckets[sid])

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_train(args):
    device  = torch.device(args.device)
    use_amp = (device.type == "cuda") and args.amp
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load dataset metadata ─────────────────────────────────────────────────
    meta_path = os.path.join(args.shard_dir, "dataset_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_fine  = meta.get("n_fine",  args.n_fine)
        n_super = meta.get("n_super", args.n_super)
        d_model = meta.get("d_model", args.d_model)
        print(f"[train] meta: n_fine={n_fine}  n_super={n_super}  d_model={d_model}")
    else:
        n_fine  = args.n_fine
        n_super = args.n_super
        d_model = args.d_model
        print(f"[train] no meta file; using CLI: n_fine={n_fine}  n_super={n_super}")

    # n_super=0 means no super mapping; use n_fine as proxy so embeddings still exist
    effective_n_super = max(n_super, 1)

    # ── Load token embedding (frozen) ─────────────────────────────────────────
    print(f"[train] loading backbone for token_emb: {args.small_ckpt}")
    backbone, _, _, _, vocab_size = load_small_backbone_and_probe(args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    token_emb: nn.Embedding = backbone.token_emb
    token_emb.weight.requires_grad_(False)

    # ── Build refiner ─────────────────────────────────────────────────────────
    refiner_kwargs = {}
    if args.variant.upper() == "B":
        refiner_kwargs["d_head"] = args.d_head
    elif args.variant.upper() == "C":
        refiner_kwargs["d_region"] = args.d_region
        refiner_kwargs["d_hidden"] = args.d_hidden

    refiner = build_refiner(
        variant=args.variant,
        d_model=d_model,
        n_fine=n_fine,
        n_super=effective_n_super,
        **refiner_kwargs,
    ).to(device)

    n_params = sum(p.numel() for p in refiner.parameters())
    print(f"[train] variant={args.variant}  params={n_params:,}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = ShardDataset(args.shard_dir)
    eval_ds  = ShardDataset(args.eval_shard_dir if args.eval_shard_dir else args.shard_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, drop_last=False)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size * 2, shuffle=False,
                               num_workers=0, drop_last=False)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )
    scaler = GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_ce    = float("inf")
    log_rows:  List[Dict] = []
    ckpt_path  = os.path.join(args.output_dir, "best_refiner.pt")

    print(f"[train] starting training  epochs={args.epochs}  "
          f"steps/epoch={len(train_loader)}  amp={use_amp}")

    for epoch in range(1, args.epochs + 1):
        refiner.train()
        ep_ce_sum    = 0.0
        ep_n_cov     = 0
        ep_n_tot     = 0
        t_ep         = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss, m = compute_loss_and_metrics(
                refiner, batch, token_emb, device, use_amp
            )
            if m["n_covered"] == 0:
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_ce_sum += float(loss) * m["n_covered"]
            ep_n_cov  += m["n_covered"]
            ep_n_tot  += m["n_total"]

            if (step + 1) % args.log_every == 0:
                print(f"  epoch {epoch}  step {step+1}/{len(train_loader)}  "
                      f"ce={ep_ce_sum/max(ep_n_cov,1):.4f}  "
                      f"fallback={1-ep_n_cov/max(ep_n_tot,1):.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        train_ce = ep_ce_sum / max(ep_n_cov, 1)
        train_fr = 1.0 - ep_n_cov / max(ep_n_tot, 1)

        # Eval
        eval_results = eval_epoch(refiner, eval_loader, token_emb, device, use_amp)
        eval_ce   = eval_results["all"]["covered_ce"]
        eval_fr   = eval_results["all"]["fallback_rate"]

        print(f"[epoch {epoch}]  train_ce={train_ce:.4f}  train_fb={train_fr:.4f}  "
              f"eval_ce={eval_ce:.4f}  eval_fb={eval_fr:.4f}  "
              f"({time.time()-t_ep:.0f}s)")

        # Per-split print
        for sname in ("core", "medium", "boundary", "tight_boundary"):
            r = eval_results.get(sname, {})
            if r.get("n_total", 0) > 0:
                print(f"  [{sname}]  ce={r.get('covered_ce', float('nan')):.4f}  "
                      f"acc={r.get('top1_acc', float('nan')):.4f}  "
                      f"fallback={r.get('fallback_rate', float('nan')):.4f}  "
                      f"n={r.get('n_total', 0)}")

        # Log row
        row: Dict = {"epoch": epoch, "train_ce": train_ce, "train_fallback": train_fr,
                     "eval_ce": eval_ce, "eval_fallback": eval_fr}
        for sname, r in eval_results.items():
            for k, v in r.items():
                row[f"{sname}_{k}"] = v
        log_rows.append(row)

        # Save best
        if np.isfinite(eval_ce) and eval_ce < best_ce:
            best_ce = eval_ce
            torch.save({
                "epoch":      epoch,
                "eval_ce":    eval_ce,
                "variant":    args.variant,
                "n_fine":     n_fine,
                "n_super":    effective_n_super,
                "d_model":    d_model,
                "refiner":    refiner.state_dict(),
            }, ckpt_path)
            print(f"  [save] best_ce={best_ce:.4f} → {ckpt_path}")

    # ── Write training log ────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, "train_log.csv")
    if log_rows:
        keys: List[str] = []
        seen: set = set()
        for r in log_rows:
            for k in r:
                if k not in seen:
                    keys.append(k); seen.add(k)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
            w.writeheader(); w.writerows(log_rows)
    print(f"[train] done.  best_eval_ce={best_ce:.4f}  log={log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir",      required=True,
                   help="Directory with shard_*.npz training shards.")
    p.add_argument("--eval_shard_dir", default=None,
                   help="Separate eval shard dir (defaults to shard_dir).")
    p.add_argument("--small_ckpt",     required=True,
                   help="Checkpoint for frozen backbone (token_emb).")
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--variant",        default="B", choices=["A", "B", "C"])
    p.add_argument("--n_fine",         type=int, default=128)
    p.add_argument("--n_super",        type=int, default=24)
    p.add_argument("--d_model",        type=int, default=384)
    # Variant B
    p.add_argument("--d_head",         type=int, default=64)
    # Variant C
    p.add_argument("--d_region",       type=int, default=32)
    p.add_argument("--d_hidden",       type=int, default=128)
    # Training
    p.add_argument("--epochs",         type=int,   default=5)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-2)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--amp",            action="store_true", default=True)
    p.add_argument("--no_amp",         action="store_false", dest="amp")
    p.add_argument("--log_every",      type=int,   default=50)
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run_train(_parse())
