#!/usr/bin/env python3
"""
soft_routing_eval.py

Tests soft hierarchical routing as a continuous prior over the vocabulary.

Instead of hard-masking tokens outside predicted regions, we bias GPT-2 logits
with log-probabilities from learned coarse/leaf router probes:

    logits_biased[v] = logits_full[v]
                     + alpha * log p(coarse(v) | h)
                     + beta  * log p(leaf(v)   | h)

The full vocabulary is always kept. No tokens are removed.

Called from full_vocab_region_eval.py via --stage soft_routing_eval, or run
directly:

    python soft_routing_eval.py --output_dir interference_experiment/full_vocab_region_eval

Requires:  cache_hidden and cluster stages to have completed in output_dir.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Sweep grids
# ──────────────────────────────────────────────────────────────────────────────

TEMPERATURES: list[float] = [0.5, 1.0, 1.5, 2.0]
ALPHAS:       list[float] = [0.0, 0.25, 0.5, 1.0, 2.0]
BETAS:        list[float] = [0.0, 0.25, 0.5, 1.0, 2.0]
MARGINS:      list[float] = [0.5, 1.0, 2.0, 4.0]
EVAL_BATCH = 256
EPS = 1e-8

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log(name: str = "sre") -> logging.Logger:
    return logging.getLogger(name)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _ensure(*dirs: str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Region structure
# ──────────────────────────────────────────────────────────────────────────────

def derive_coarse_labels(region_tree: dict, V: int) -> tuple[np.ndarray, int]:
    """
    Extract coarse region IDs from the region tree produced by the cluster stage.

    region_tree["children"][i]["vocab_ids"] = sub-indices belonging to coarse region i.
    Returns:
        coarse_id_arr: (V,) int32  — coarse region id for each sub-index, -1 if unassigned
        n_coarse: int
    """
    coarse_id_arr = np.full(V, -1, dtype=np.int32)
    children = region_tree.get("children", [])
    for coarse_id, child in enumerate(children):
        for sub_idx in child.get("vocab_ids", []):
            if 0 <= sub_idx < V:
                coarse_id_arr[sub_idx] = coarse_id
    return coarse_id_arr, len(children)


# ──────────────────────────────────────────────────────────────────────────────
# Probe
# ──────────────────────────────────────────────────────────────────────────────

class _LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _split_indices(N: int, seed: int, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    split = int(train_frac * N)
    return perm[:split], perm[split:]


def _train_probe(
    H_tr: np.ndarray, y_tr: np.ndarray,
    H_te: np.ndarray, y_te: np.ndarray,
    n_classes: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 2048,
) -> tuple[_LinearProbe, dict]:
    """Train linear probe and return (probe_on_cpu, metrics_dict)."""
    D = H_tr.shape[1]
    Htr = torch.from_numpy(H_tr.astype(np.float32)).to(device)
    ytr = torch.from_numpy(y_tr.astype(np.int64)).to(device)
    Hte = torch.from_numpy(H_te.astype(np.float32)).to(device)
    yte = torch.from_numpy(y_te.astype(np.int64)).to(device)

    probe = _LinearProbe(D, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(len(Htr), device=device)
        for i in range(0, len(Htr), batch_size):
            idx = perm[i: i + batch_size]
            opt.zero_grad()
            ce(probe(Htr[idx]), ytr[idx]).backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        logits_te = probe(Hte)
        ce_loss = float(ce(logits_te, yte).item())
        _, topk_pred = logits_te.topk(min(8, n_classes), dim=1)
        yte_col = yte.unsqueeze(1)
        metrics = {
            "ce": ce_loss,
            "top1": float((topk_pred[:, :1] == yte_col).any(1).float().mean()),
            "top4": float((topk_pred[:, :min(4, n_classes)] == yte_col).any(1).float().mean()),
            "top8": float((topk_pred[:, :min(8, n_classes)] == yte_col).any(1).float().mean()),
            "random_top1": 1.0 / n_classes,
        }

    probe = probe.cpu()
    return probe, metrics


def _get_or_train_probe(
    H: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    save_path: str,
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    label: str,
) -> tuple[_LinearProbe, dict]:
    log = _log()
    tr_idx, te_idx = _split_indices(len(H), seed)

    if os.path.exists(save_path):
        log.info(f"  Loading saved {label} probe from {save_path}")
        D = H.shape[1]
        probe = _LinearProbe(D, n_classes)
        probe.load_state_dict(torch.load(save_path, map_location="cpu"))
        probe.eval()
        # Still compute metrics on test split
        with torch.no_grad():
            Hte = torch.from_numpy(H[te_idx].astype(np.float32))
            yte = torch.from_numpy(labels[te_idx].astype(np.int64))
            logits_te = probe(Hte)
            ce_loss = float(nn.CrossEntropyLoss()(logits_te, yte).item())
            _, topk_pred = logits_te.topk(min(8, n_classes), dim=1)
            yte_col = yte.unsqueeze(1)
            metrics = {
                "ce": ce_loss,
                "top1": float((topk_pred[:, :1] == yte_col).any(1).float().mean()),
                "top4": float((topk_pred[:, :min(4, n_classes)] == yte_col).any(1).float().mean()),
                "top8": float((topk_pred[:, :min(8, n_classes)] == yte_col).any(1).float().mean()),
                "random_top1": 1.0 / n_classes,
            }
    else:
        log.info(f"  Training {label} probe  n_classes={n_classes}  "
                 f"train={len(tr_idx):,}  test={len(te_idx):,}")
        probe, metrics = _train_probe(
            H[tr_idx], labels[tr_idx],
            H[te_idx], labels[te_idx],
            n_classes, device, epochs, lr, batch_size,
        )
        torch.save(probe.state_dict(), save_path)
        log.info(f"  Saved {label} probe → {save_path}")

    log.info(
        f"  {label}: top1={metrics['top1']:.4f}  top4={metrics['top4']:.4f}  "
        f"top8={metrics['top8']:.4f}  random={metrics['random_top1']:.6f}  "
        f"ce={metrics['ce']:.4f}"
    )
    return probe, metrics


# ──────────────────────────────────────────────────────────────────────────────
# LM head utilities
# ──────────────────────────────────────────────────────────────────────────────

def _load_lm_head(model, top_ids: np.ndarray) -> tuple[nn.LayerNorm, torch.Tensor]:
    """Return ln_f and wte_sub, both on CPU float32."""
    ln_f = model.transformer.ln_f.cpu().float()
    wte_sub = model.transformer.wte.weight[top_ids].detach().cpu().float()  # (V, D)
    return ln_f, wte_sub


def _compute_lm_logits(
    H_batch: torch.Tensor,   # (B, D) float32 CPU
    ln_f: nn.LayerNorm,
    wte_sub: torch.Tensor,   # (V, D) float32 CPU
) -> torch.Tensor:           # (B, V) float32 CPU
    with torch.no_grad():
        return ln_f(H_batch) @ wte_sub.T


# ──────────────────────────────────────────────────────────────────────────────
# Per-batch metrics
# ──────────────────────────────────────────────────────────────────────────────

def _batch_metrics(
    logits: torch.Tensor,     # (B, V)
    gold_sub: torch.Tensor,   # (B,) int64 — sub-indices
) -> dict[str, torch.Tensor]:
    """NLL, gold rank (1-indexed), top-1 hit, distribution entropy."""
    lp = F.log_softmax(logits, dim=-1)           # (B, V)
    nll = -lp[torch.arange(len(gold_sub)), gold_sub]   # (B,)

    # rank: number of tokens strictly higher than gold score + 1
    gold_scores = logits[torch.arange(len(gold_sub)), gold_sub].unsqueeze(1)  # (B, 1)
    rank = (logits > gold_scores).sum(dim=1).float() + 1.0                    # (B,)

    top1_hit = (logits.argmax(dim=1) == gold_sub).float()                     # (B,)

    p = F.softmax(logits, dim=-1)
    entropy = -(p * (p + EPS).log()).sum(dim=-1)   # (B,) nats

    return {"nll": nll, "rank": rank, "top1_hit": top1_hit, "entropy": entropy}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — train / load routers
# ──────────────────────────────────────────────────────────────────────────────

def step1_train_routers(
    out_dir: str,
    sr_dir: str,
    target_layers: list[int],
    leaf_labels: np.ndarray,
    coarse_labels: np.ndarray,
    n_leaves: int,
    n_coarse: int,
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> dict[int, tuple[_LinearProbe, _LinearProbe, dict, dict]]:
    """
    Returns {layer: (coarse_probe, leaf_probe, coarse_metrics, leaf_metrics)}.
    Probes are saved/loaded from sr_dir.
    """
    log = _log()
    log.info("Step 1 — Train / load router probes")
    result: dict[int, tuple[_LinearProbe, _LinearProbe, dict, dict]] = {}

    for l in target_layers:
        hpath = os.path.join(out_dir, f"hidden_L{l:02d}.npy")
        if not os.path.exists(hpath):
            log.warning(f"  hidden_L{l:02d}.npy not found — skipping")
            continue
        H = np.load(hpath)  # (N, D) fp16
        log.info(f"Layer {l:02d}  H={H.shape}")

        coarse_path = os.path.join(sr_dir, f"coarse_probe_L{l:02d}.pt")
        leaf_path   = os.path.join(sr_dir, f"leaf_probe_L{l:02d}.pt")

        # Filter to samples that have valid coarse labels
        valid = (coarse_labels >= 0) & (leaf_labels >= 0)
        H_v = H[valid]
        cl_v = coarse_labels[valid]
        ll_v = leaf_labels[valid]

        coarse_probe, coarse_m = _get_or_train_probe(
            H_v, cl_v, n_coarse, coarse_path, device, seed, epochs, lr, batch_size, f"coarse L{l:02d}"
        )
        leaf_probe, leaf_m = _get_or_train_probe(
            H_v, ll_v, n_leaves, leaf_path, device, seed, epochs, lr, batch_size, f"leaf L{l:02d}"
        )
        result[l] = (coarse_probe, leaf_probe, coarse_m, leaf_m)

    # Save router metrics
    rows = []
    for l, (_, _, cm, lm) in result.items():
        rows.append({"layer": l, "router": "coarse", **cm})
        rows.append({"layer": l, "router": "leaf",   **lm})
    if rows:
        _write_csv(os.path.join(sr_dir, "router_probe_metrics.csv"), rows)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — baseline full logits
# ──────────────────────────────────────────────────────────────────────────────

def step2_baseline(
    out_dir: str,
    sr_dir: str,
    final_layer: int,
    te_idx: np.ndarray,
    gold_tokens: np.ndarray,
    ln_f: nn.LayerNorm,
    wte_sub: torch.Tensor,
) -> dict:
    log = _log()
    log.info("Step 2 — Baseline full logits")

    H = np.load(os.path.join(out_dir, f"hidden_L{final_layer:02d}.npy"))
    H_te = H[te_idx]
    gold_te = torch.from_numpy(gold_tokens[te_idx].astype(np.int64))

    all_nll, all_rank, all_top1, all_ent = [], [], [], []

    for i in range(0, len(H_te), EVAL_BATCH):
        h_b = torch.from_numpy(H_te[i:i+EVAL_BATCH].astype(np.float32))
        logits = _compute_lm_logits(h_b, ln_f, wte_sub)
        g_b = gold_te[i:i+EVAL_BATCH]
        m = _batch_metrics(logits, g_b)
        all_nll.append(m["nll"])
        all_rank.append(m["rank"])
        all_top1.append(m["top1_hit"])
        all_ent.append(m["entropy"])

    nll   = torch.cat(all_nll).numpy()
    rank  = torch.cat(all_rank).numpy()
    top1  = torch.cat(all_top1).numpy()
    ent   = torch.cat(all_ent).numpy()

    baseline = {
        "mean_nll":  float(np.mean(nll)),
        "ppl":       float(math.exp(min(float(np.mean(nll)), 100))),
        "top1_acc":  float(np.mean(top1)),
        "mean_rank": float(np.mean(rank)),
        "median_rank": float(np.median(rank)),
        "mean_entropy_nats": float(np.mean(ent)),
        "n_samples": len(nll),
    }
    log.info(
        f"  Baseline: nll={baseline['mean_nll']:.4f}  ppl={baseline['ppl']:.2f}  "
        f"top1={baseline['top1_acc']:.4f}  mean_rank={baseline['mean_rank']:.1f}"
    )

    # Also save per-sample arrays for bucket analysis later
    np.save(os.path.join(sr_dir, "baseline_nll.npy"), nll)
    np.save(os.path.join(sr_dir, "baseline_rank.npy"), rank)
    np.save(os.path.join(sr_dir, "te_idx.npy"), te_idx)

    return baseline


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — soft routing sweep
# ──────────────────────────────────────────────────────────────────────────────

def _router_logits_for_test(
    probe: _LinearProbe,
    H_te: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Run probe over test hidden states; return (N_te, n_classes) float32 CPU."""
    probe = probe.to(device)
    probe.eval()
    chunks = []
    for i in range(0, len(H_te), 1024):
        h = torch.from_numpy(H_te[i:i+1024].astype(np.float32)).to(device)
        with torch.no_grad():
            chunks.append(probe(h).cpu())
    return torch.cat(chunks, dim=0)  # (N_te, n_classes)


def step3_soft_sweep(
    out_dir: str,
    sr_dir: str,
    final_layer: int,
    te_idx: np.ndarray,
    gold_tokens: np.ndarray,
    coarse_probe: _LinearProbe,
    leaf_probe: _LinearProbe,
    coarse_id_arr: np.ndarray,   # (V,) int32
    leaf_id_arr: np.ndarray,     # (V,) int32
    ln_f: nn.LayerNorm,
    wte_sub: torch.Tensor,
    device: torch.device,
    baseline_nll: float,
) -> list[dict]:
    log = _log()
    log.info("Step 3 — Soft routing sweep  "
             f"(T×{len(TEMPERATURES)} α×{len(ALPHAS)} β×{len(BETAS)} = "
             f"{len(TEMPERATURES)*len(ALPHAS)*len(BETAS)} combos)")

    H = np.load(os.path.join(out_dir, f"hidden_L{final_layer:02d}.npy"))
    H_te = H[te_idx]
    gold_te = torch.from_numpy(gold_tokens[te_idx].astype(np.int64))

    # Pre-compute router logits for all test samples (once)
    log.info("  Pre-computing coarse router logits …")
    coarse_rl = _router_logits_for_test(coarse_probe, H_te, device)  # (N_te, n_coarse)
    log.info("  Pre-computing leaf router logits …")
    leaf_rl   = _router_logits_for_test(leaf_probe,   H_te, device)  # (N_te, n_leaves)

    coarse_idx_t = torch.from_numpy(coarse_id_arr.astype(np.int64))  # (V,)
    leaf_idx_t   = torch.from_numpy(leaf_id_arr.astype(np.int64))    # (V,)

    # Accumulator: (temperature_idx, alpha_idx, beta_idx) → running stats
    n_T, n_A, n_B = len(TEMPERATURES), len(ALPHAS), len(BETAS)
    sum_nll  = np.zeros((n_T, n_A, n_B), dtype=np.float64)
    sum_rank = np.zeros((n_T, n_A, n_B), dtype=np.float64)
    sum_top1 = np.zeros((n_T, n_A, n_B), dtype=np.float64)
    sum_ent  = np.zeros((n_T, n_A, n_B), dtype=np.float64)
    counts   = np.zeros((n_T, n_A, n_B), dtype=np.int64)

    for i in range(0, len(H_te), EVAL_BATCH):
        sl = slice(i, i + EVAL_BATCH)
        h_b     = torch.from_numpy(H_te[sl].astype(np.float32))
        g_b     = gold_te[sl]
        c_rl_b  = coarse_rl[sl]   # (B, n_coarse)
        l_rl_b  = leaf_rl[sl]     # (B, n_leaves)
        B       = h_b.shape[0]

        logits_full = _compute_lm_logits(h_b, ln_f, wte_sub)  # (B, V)

        for ti, temp in enumerate(TEMPERATURES):
            # log-probs from router at this temperature
            c_lp = F.log_softmax(c_rl_b / temp, dim=-1)    # (B, n_coarse)
            l_lp = F.log_softmax(l_rl_b / temp, dim=-1)    # (B, n_leaves)

            # Scatter to vocab: each token gets its region's log-prob
            # (B, V) = (B, n_coarse/n_leaves)[:, coarse/leaf_id_arr]
            c_prior = c_lp[:, coarse_idx_t]   # (B, V)
            l_prior = l_lp[:, leaf_idx_t]     # (B, V)

            for ai, alpha in enumerate(ALPHAS):
                for bi, beta in enumerate(BETAS):
                    if alpha == 0.0 and beta == 0.0:
                        logits_b = logits_full
                    elif alpha == 0.0:
                        logits_b = logits_full + beta * l_prior
                    elif beta == 0.0:
                        logits_b = logits_full + alpha * c_prior
                    else:
                        logits_b = logits_full + alpha * c_prior + beta * l_prior

                    m = _batch_metrics(logits_b, g_b)
                    sum_nll [ti, ai, bi] += m["nll"].sum().item()
                    sum_rank[ti, ai, bi] += m["rank"].sum().item()
                    sum_top1[ti, ai, bi] += m["top1_hit"].sum().item()
                    sum_ent [ti, ai, bi] += m["entropy"].sum().item()
                    counts  [ti, ai, bi] += B

    N_te = counts[0, 0, 0]
    rows: list[dict] = []

    for ti, temp in enumerate(TEMPERATURES):
        for ai, alpha in enumerate(ALPHAS):
            for bi, beta in enumerate(BETAS):
                n = int(counts[ti, ai, bi])
                mean_nll  = sum_nll [ti, ai, bi] / n
                mean_rank = sum_rank[ti, ai, bi] / n
                mean_top1 = sum_top1[ti, ai, bi] / n
                mean_ent  = sum_ent [ti, ai, bi] / n
                delta_nll = mean_nll - baseline_nll

                # tag special combinations
                if alpha == 0.0 and beta == 0.0:
                    tag = "full_baseline"
                elif alpha > 0.0 and beta == 0.0:
                    tag = "coarse_only"
                elif alpha == 0.0 and beta > 0.0:
                    tag = "leaf_only"
                elif alpha == 1.0 and beta == 1.0:
                    tag = "hierarchical_mixture"
                else:
                    tag = "coarse+leaf"

                rows.append({
                    "temperature": temp,
                    "alpha": alpha,
                    "beta": beta,
                    "tag": tag,
                    "mean_nll": mean_nll,
                    "ppl": math.exp(min(mean_nll, 100)),
                    "delta_nll": delta_nll,
                    "mean_rank": mean_rank,
                    "top1_acc": mean_top1,
                    "mean_entropy_nats": mean_ent,
                    "n_samples": n,
                })

    _write_csv(os.path.join(sr_dir, "soft_routing_sweep.csv"), rows)

    # Log top-10 by delta_nll
    rows_sorted = sorted(rows, key=lambda r: r["delta_nll"])
    log.info("  Best 10 (alpha, beta, temp) by delta_NLL:")
    for r in rows_sorted[:10]:
        log.info(
            f"    tag={r['tag']:22s}  α={r['alpha']:.2f} β={r['beta']:.2f} "
            f"T={r['temperature']:.1f}  Δnll={r['delta_nll']:+.4f}  "
            f"rank={r['mean_rank']:.1f}  top1={r['top1_acc']:.4f}"
        )

    log.info(f"  Saved soft_routing_sweep.csv  ({len(rows)} rows)")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — oracle soft routing
# ──────────────────────────────────────────────────────────────────────────────

def step4_oracle(
    out_dir: str,
    sr_dir: str,
    final_layer: int,
    te_idx: np.ndarray,
    gold_tokens: np.ndarray,
    coarse_id_arr: np.ndarray,
    leaf_id_arr: np.ndarray,
    ln_f: nn.LayerNorm,
    wte_sub: torch.Tensor,
    baseline_nll: float,
) -> None:
    log = _log()
    log.info("Step 4 — Oracle soft routing")

    H = np.load(os.path.join(out_dir, f"hidden_L{final_layer:02d}.npy"))
    H_te = H[te_idx]
    gold_te = torch.from_numpy(gold_tokens[te_idx].astype(np.int64))

    V = wte_sub.shape[0]
    coarse_t = torch.from_numpy(coarse_id_arr.astype(np.int64))  # (V,)
    leaf_t   = torch.from_numpy(leaf_id_arr.astype(np.int64))    # (V,)

    rows: list[dict] = []

    for margin in MARGINS:
        for mode in ("coarse", "leaf", "coarse+leaf"):
            sum_nll, sum_rank, sum_top1, cnt = 0.0, 0.0, 0.0, 0

            for i in range(0, len(H_te), EVAL_BATCH):
                sl = slice(i, i + EVAL_BATCH)
                h_b  = torch.from_numpy(H_te[sl].astype(np.float32))
                g_b  = gold_te[sl]                                    # (B,)
                B    = h_b.shape[0]

                logits_full = _compute_lm_logits(h_b, ln_f, wte_sub)  # (B, V)

                # Gold region IDs for this batch
                gold_coarse = coarse_id_arr[g_b.numpy()]              # (B,) int
                gold_leaf   = leaf_id_arr  [g_b.numpy()]              # (B,) int

                # Oracle prior: 0 for same-region tokens, -margin otherwise
                # Vectorized: (B, 1) vs (V,) → (B, V) bool
                gc_t = torch.from_numpy(gold_coarse).long().unsqueeze(1)   # (B, 1)
                gl_t = torch.from_numpy(gold_leaf  ).long().unsqueeze(1)   # (B, 1)

                same_coarse = (coarse_t.unsqueeze(0) == gc_t)  # (B, V)
                same_leaf   = (leaf_t  .unsqueeze(0) == gl_t)  # (B, V)

                oracle_coarse = torch.where(same_coarse,
                                            torch.zeros(B, V),
                                            torch.full((B, V), -margin))
                oracle_leaf   = torch.where(same_leaf,
                                            torch.zeros(B, V),
                                            torch.full((B, V), -margin))

                if mode == "coarse":
                    logits_b = logits_full + oracle_coarse
                elif mode == "leaf":
                    logits_b = logits_full + oracle_leaf
                else:
                    logits_b = logits_full + oracle_coarse + oracle_leaf

                m = _batch_metrics(logits_b, g_b)
                sum_nll  += m["nll"].sum().item()
                sum_rank += m["rank"].sum().item()
                sum_top1 += m["top1_hit"].sum().item()
                cnt      += B

            mean_nll  = sum_nll  / cnt
            mean_rank = sum_rank / cnt
            mean_top1 = sum_top1 / cnt
            rows.append({
                "mode":       mode,
                "margin":     margin,
                "mean_nll":   mean_nll,
                "ppl":        math.exp(min(mean_nll, 100)),
                "delta_nll":  mean_nll - baseline_nll,
                "mean_rank":  mean_rank,
                "top1_acc":   mean_top1,
                "n_samples":  cnt,
            })
            log.info(
                f"  oracle {mode:10s}  margin={margin:.1f}  "
                f"Δnll={mean_nll - baseline_nll:+.4f}  rank={mean_rank:.1f}"
            )

    _write_csv(os.path.join(sr_dir, "oracle_soft_routing.csv"), rows)
    log.info("  Saved oracle_soft_routing.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — hard mask baseline (for comparison only)
# ──────────────────────────────────────────────────────────────────────────────

def step5_hard_mask_baseline(
    out_dir: str,
    sr_dir: str,
    final_layer: int,
    te_idx: np.ndarray,
    gold_tokens: np.ndarray,
    coarse_probe: _LinearProbe,
    leaf_probe: _LinearProbe,
    coarse_id_arr: np.ndarray,
    leaf_id_arr: np.ndarray,
    leaf_region_to_tokens: dict[int, list[int]],
    ln_f: nn.LayerNorm,
    wte_sub: torch.Tensor,
    device: torch.device,
    baseline_nll: float,
) -> None:
    log = _log()
    log.info("Step 5 — Hard mask baseline (HARD_MASK_BASELINE)")

    H = np.load(os.path.join(out_dir, f"hidden_L{final_layer:02d}.npy"))
    H_te = H[te_idx]
    gold_te = torch.from_numpy(gold_tokens[te_idx].astype(np.int64))
    V = wte_sub.shape[0]

    coarse_rl = _router_logits_for_test(coarse_probe, H_te, device)
    leaf_rl   = _router_logits_for_test(leaf_probe,   H_te, device)

    r_values = [4, 8, 16, 32, 64]
    rows: list[dict] = []

    for r in r_values:
        for granularity, rl, id_arr in [
            ("coarse", coarse_rl, coarse_id_arr),
            ("leaf",   leaf_rl,   leaf_id_arr),
        ]:
            n_classes = rl.shape[1]
            r_cap = min(r, n_classes)
            _, top_r = rl.topk(r_cap, dim=1)   # (N_te, r_cap)

            cond_nlls, fails, full_nlls = [], [], []

            for i in range(0, len(H_te), EVAL_BATCH):
                sl = slice(i, i + EVAL_BATCH)
                h_b = torch.from_numpy(H_te[sl].astype(np.float32))
                g_b = gold_te[sl]
                lg_full = _compute_lm_logits(h_b, ln_f, wte_sub)   # (B, V)

                top_r_b = top_r[sl].numpy()
                g_np    = g_b.numpy()

                for b_i in range(h_b.shape[0]):
                    # Build candidate mask
                    cand = np.zeros(V, dtype=bool)
                    for region_id in top_r_b[b_i]:
                        if granularity == "leaf":
                            toks = leaf_region_to_tokens.get(int(region_id), [])
                        else:
                            # coarse: all leaf tokens whose coarse_id matches
                            toks = np.where(coarse_id_arr == int(region_id))[0].tolist()
                        cand[toks] = True

                    gold_sub = int(g_np[b_i])
                    m = _batch_metrics(lg_full[b_i:b_i+1], g_b[b_i:b_i+1])
                    full_nlls.append(float(m["nll"][0].item()))

                    if not cand[gold_sub]:
                        fails.append(1)
                        continue
                    fails.append(0)

                    lg_masked = lg_full[b_i].clone()
                    lg_masked[~torch.from_numpy(cand)] = -1e9
                    m2 = _batch_metrics(lg_masked.unsqueeze(0), g_b[b_i:b_i+1])
                    cond_nlls.append(float(m2["nll"][0].item()))

            fail_rate  = float(np.mean(fails))
            full_nll   = float(np.mean(full_nlls))
            cond_nll   = float(np.mean(cond_nlls)) if cond_nlls else float("nan")
            penalty    = math.log(V)
            eff_nll    = (1 - fail_rate) * cond_nll + fail_rate * penalty if cond_nlls else float("nan")

            rows.append({
                "type": "HARD_MASK_BASELINE",
                "granularity": granularity,
                "r": r,
                "full_nll": full_nll,
                "cond_nll": cond_nll,
                "effective_nll": eff_nll,
                "failure_rate": fail_rate,
                "delta_eff_vs_full": eff_nll - baseline_nll if not math.isnan(eff_nll) else float("nan"),
            })
            log.info(
                f"  HARD {granularity:6s} r={r:3d}: "
                f"cond_nll={cond_nll:.4f}  fail={fail_rate:.4f}  "
                f"eff_nll={eff_nll:.4f}"
            )

    _write_csv(os.path.join(sr_dir, "hard_mask_baseline.csv"), rows)
    log.info("  Saved hard_mask_baseline.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — hierarchical mixture factorization
# ──────────────────────────────────────────────────────────────────────────────

def step6_hierarchical_mixture(
    out_dir: str,
    sr_dir: str,
    final_layer: int,
    te_idx: np.ndarray,
    gold_tokens: np.ndarray,
    coarse_probe: _LinearProbe,
    leaf_probe: _LinearProbe,
    coarse_id_arr: np.ndarray,
    leaf_id_arr: np.ndarray,
    ln_f: nn.LayerNorm,
    wte_sub: torch.Tensor,
    device: torch.device,
    baseline_nll: float,
) -> None:
    """
    Explicit hierarchical factorization test.

    score[v] = log P_full(v|h) + alpha*log P(coarse(v)|h) + beta*log P(leaf(v)|h)

    This is identical to the soft sweep at those alpha/beta values, but we also
    compute the "pure factorization" P(v|h) ∝ P_full(v|h)*P(coarse|h)*P(leaf|h)
    without any additive baseline (replace logits_full with log_softmax(logits_full)).
    """
    log = _log()
    log.info("Step 6 — Hierarchical mixture factorization")

    H = np.load(os.path.join(out_dir, f"hidden_L{final_layer:02d}.npy"))
    H_te = H[te_idx]
    gold_te = torch.from_numpy(gold_tokens[te_idx].astype(np.int64))

    coarse_rl = _router_logits_for_test(coarse_probe, H_te, device)
    leaf_rl   = _router_logits_for_test(leaf_probe,   H_te, device)
    coarse_idx_t = torch.from_numpy(coarse_id_arr.astype(np.int64))
    leaf_idx_t   = torch.from_numpy(leaf_id_arr.astype(np.int64))

    # Test temperatures 0.5 and 1.0 for this analysis
    rows: list[dict] = []

    for temp in [0.5, 1.0]:
        # Factorization variants:
        # A. logits_full + log P(coarse) + log P(leaf)   [alpha=beta=1, additive to raw logits]
        # B. log_softmax(logits_full) + log P(coarse) + log P(leaf)  [true log-prob mixture]
        for variant in ("additive_raw", "log_prob_mixture"):
            sum_nll, sum_rank, sum_top1, cnt = 0.0, 0.0, 0.0, 0

            for i in range(0, len(H_te), EVAL_BATCH):
                sl = slice(i, i + EVAL_BATCH)
                h_b    = torch.from_numpy(H_te[sl].astype(np.float32))
                g_b    = gold_te[sl]
                c_rl_b = coarse_rl[sl]
                l_rl_b = leaf_rl[sl]

                lg_full = _compute_lm_logits(h_b, ln_f, wte_sub)  # (B, V)

                c_lp = F.log_softmax(c_rl_b / temp, dim=-1)
                l_lp = F.log_softmax(l_rl_b / temp, dim=-1)
                c_prior = c_lp[:, coarse_idx_t]   # (B, V)
                l_prior = l_lp[:, leaf_idx_t]     # (B, V)

                if variant == "additive_raw":
                    logits_b = lg_full + c_prior + l_prior
                else:
                    lp_full = F.log_softmax(lg_full, dim=-1)
                    logits_b = lp_full + c_prior + l_prior

                m = _batch_metrics(logits_b, g_b)
                sum_nll  += m["nll"].sum().item()
                sum_rank += m["rank"].sum().item()
                sum_top1 += m["top1_hit"].sum().item()
                cnt      += g_b.shape[0]

            mean_nll  = sum_nll  / cnt
            rows.append({
                "variant":     variant,
                "temperature": temp,
                "alpha":       1.0,
                "beta":        1.0,
                "mean_nll":    mean_nll,
                "ppl":         math.exp(min(mean_nll, 100)),
                "delta_nll":   mean_nll - baseline_nll,
                "mean_rank":   sum_rank / cnt,
                "top1_acc":    sum_top1 / cnt,
                "n_samples":   cnt,
            })
            log.info(
                f"  {variant:20s}  T={temp:.1f}  "
                f"Δnll={mean_nll - baseline_nll:+.4f}  "
                f"rank={sum_rank/cnt:.1f}  top1={sum_top1/cnt:.4f}"
            )

    _write_csv(os.path.join(sr_dir, "hierarchical_mixture.csv"), rows)
    log.info("  Saved hierarchical_mixture.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 — bucket analysis
# ──────────────────────────────────────────────────────────────────────────────

def _quantile_buckets(values: np.ndarray, n_buckets: int = 4) -> np.ndarray:
    """Return bucket index (0..n_buckets-1) for each value by equal-frequency binning."""
    quantiles = np.linspace(0, 100, n_buckets + 1)
    edges = np.percentile(values, quantiles)
    edges[-1] += 1e-9   # include max
    return np.digitize(values, edges[1:-1]).astype(np.int32)


def step7_bucket_analysis(
    out_dir: str,
    sr_dir: str,
    final_layer: int,
    te_idx: np.ndarray,
    gold_tokens: np.ndarray,
    coarse_probe: _LinearProbe,
    leaf_probe: _LinearProbe,
    coarse_id_arr: np.ndarray,
    leaf_id_arr: np.ndarray,
    ln_f: nn.LayerNorm,
    wte_sub: torch.Tensor,
    device: torch.device,
    best_alpha: float,
    best_beta: float,
    best_temp: float,
    baseline_nll: np.ndarray,     # per-sample, from step2
) -> None:
    log = _log()
    log.info("Step 7 — Bucket analysis  "
             f"(best α={best_alpha}  β={best_beta}  T={best_temp})")

    H = np.load(os.path.join(out_dir, f"hidden_L{final_layer:02d}.npy"))
    H_te = H[te_idx]
    gold_te = torch.from_numpy(gold_tokens[te_idx].astype(np.int64))
    N_te = len(H_te)

    coarse_rl = _router_logits_for_test(coarse_probe, H_te, device)   # (N_te, n_coarse)
    leaf_rl   = _router_logits_for_test(leaf_probe,   H_te, device)   # (N_te, n_leaves)
    coarse_idx_t = torch.from_numpy(coarse_id_arr.astype(np.int64))
    leaf_idx_t   = torch.from_numpy(leaf_id_arr.astype(np.int64))

    # Router confidence per sample
    c_probs = F.softmax(coarse_rl / best_temp, dim=-1).numpy()   # (N_te, n_coarse)
    l_probs = F.softmax(leaf_rl   / best_temp, dim=-1).numpy()   # (N_te, n_leaves)
    max_coarse_conf = c_probs.max(axis=1)       # (N_te,)
    max_leaf_conf   = l_probs.max(axis=1)       # (N_te,)

    # Gold coarse / leaf rank in router
    gold_np      = gold_te.numpy()
    gold_coarse  = coarse_id_arr[gold_np]                                # (N_te,)
    gold_leaf    = leaf_id_arr  [gold_np]                                # (N_te,)
    gold_c_rank  = np.argsort(np.argsort(-c_probs, axis=1), axis=1)[
        np.arange(N_te), gold_coarse.clip(0)
    ] + 1   # (N_te,) 1-indexed
    gold_l_rank  = np.argsort(np.argsort(-l_probs, axis=1), axis=1)[
        np.arange(N_te), gold_leaf.clip(0)
    ] + 1   # (N_te,)

    # Per-sample biased NLL at best setting
    soft_nll = np.zeros(N_te, dtype=np.float32)
    for i in range(0, N_te, EVAL_BATCH):
        sl = slice(i, i + EVAL_BATCH)
        h_b    = torch.from_numpy(H_te[sl].astype(np.float32))
        g_b    = gold_te[sl]
        c_rl_b = coarse_rl[sl]
        l_rl_b = leaf_rl[sl]
        lg_full = _compute_lm_logits(h_b, ln_f, wte_sub)
        c_lp = F.log_softmax(c_rl_b / best_temp, dim=-1)
        l_lp = F.log_softmax(l_rl_b / best_temp, dim=-1)
        logits_b = lg_full + best_alpha * c_lp[:, coarse_idx_t] + best_beta * l_lp[:, leaf_idx_t]
        m = _batch_metrics(logits_b, g_b)
        soft_nll[sl] = m["nll"].numpy()

    delta_nll = soft_nll - baseline_nll   # (N_te,)

    # Bucketing dimensions
    bucket_dims = {
        "max_coarse_conf":  _quantile_buckets(max_coarse_conf),
        "max_leaf_conf":    _quantile_buckets(max_leaf_conf),
        "gold_coarse_rank": _quantile_buckets(gold_c_rank.astype(float)),
        "gold_leaf_rank":   _quantile_buckets(gold_l_rank.astype(float)),
        "baseline_nll":     _quantile_buckets(baseline_nll),
    }

    rows: list[dict] = []
    for dim_name, buckets in bucket_dims.items():
        for b_id in range(4):
            mask = buckets == b_id
            if mask.sum() == 0:
                continue
            rows.append({
                "bucket_dim":      dim_name,
                "bucket_id":       b_id,
                "n":               int(mask.sum()),
                "mean_delta_nll":  float(delta_nll[mask].mean()),
                "mean_baseline_nll": float(baseline_nll[mask].mean()),
                "mean_soft_nll":   float(soft_nll[mask].mean()),
                "frac_improved":   float((delta_nll[mask] < 0).mean()),
                # representative value for bucket
                "bucket_min": float({"max_coarse_conf": max_coarse_conf,
                                     "max_leaf_conf": max_leaf_conf,
                                     "gold_coarse_rank": gold_c_rank.astype(float),
                                     "gold_leaf_rank": gold_l_rank.astype(float),
                                     "baseline_nll": baseline_nll}[dim_name][mask].min()),
                "bucket_max": float({"max_coarse_conf": max_coarse_conf,
                                     "max_leaf_conf": max_leaf_conf,
                                     "gold_coarse_rank": gold_c_rank.astype(float),
                                     "gold_leaf_rank": gold_l_rank.astype(float),
                                     "baseline_nll": baseline_nll}[dim_name][mask].max()),
            })
            log.info(
                f"  {dim_name:22s} bucket={b_id}  n={mask.sum():6,}  "
                f"Δnll={delta_nll[mask].mean():+.4f}  "
                f"improved={100*(delta_nll[mask]<0).mean():.1f}%"
            )

    _write_csv(os.path.join(sr_dir, "soft_routing_bucket_analysis.csv"), rows)
    log.info("  Saved soft_routing_bucket_analysis.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Step 8 — summary
# ──────────────────────────────────────────────────────────────────────────────

def step8_summary(
    sr_dir: str,
    baseline: dict,
    sweep_rows: list[dict],
    final_layer: int,
) -> None:
    log = _log()
    log.info("Step 8 — Summary")

    def _load(fname: str) -> list[dict]:
        p = os.path.join(sr_dir, fname)
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return list(csv.DictReader(f))

    oracle_rows    = _load("oracle_soft_routing.csv")
    hard_rows      = _load("hard_mask_baseline.csv")
    mixture_rows   = _load("hierarchical_mixture.csv")
    bucket_rows    = _load("soft_routing_bucket_analysis.csv")

    bsl_nll = baseline["mean_nll"]
    bsl_ppl = baseline["ppl"]

    # Best sweep result
    best_row = min(sweep_rows, key=lambda r: r["delta_nll"])
    # Best coarse-only
    best_coarse = min((r for r in sweep_rows if r["tag"] == "coarse_only"),
                      key=lambda r: r["delta_nll"], default=None)
    # Best leaf-only
    best_leaf = min((r for r in sweep_rows if r["tag"] == "leaf_only"),
                    key=lambda r: r["delta_nll"], default=None)
    # Best oracle
    best_oracle = min(oracle_rows, key=lambda r: float(r["delta_nll"]), default=None)
    # Hierarchical mixture (alpha=beta=1, T=1)
    mixture_11 = next(
        (r for r in mixture_rows
         if float(r["alpha"]) == 1.0 and float(r["temperature"]) == 1.0
         and r["variant"] == "additive_raw"),
        None
    )

    # Decision
    b_delta  = float(best_row["delta_nll"])
    b_rank   = float(best_row["mean_rank"])
    bsl_rank = baseline["mean_rank"]
    o_delta  = float(best_oracle["delta_nll"]) if best_oracle else float("nan")

    hard_best_eff = min(
        (float(r["effective_nll"]) for r in hard_rows if r["type"] == "HARD_MASK_BASELINE"),
        default=float("nan")
    )
    hard_best_delta = hard_best_eff - bsl_nll

    if b_delta <= 0 and b_rank < bsl_rank and not math.isnan(o_delta) and o_delta <= -0.1:
        support = "STRONG_SUPPORT"
        reason = (f"Best soft routing Δnll={b_delta:+.4f} ≤ 0, "
                  f"gold rank improved {bsl_rank:.1f}→{b_rank:.1f}, "
                  f"oracle Δnll={o_delta:+.4f} shows clear headroom.")
    elif b_delta <= 0.05 or b_rank < bsl_rank * 0.95:
        support = "PARTIAL_SUPPORT"
        reason = (f"Soft routing shows modest improvement "
                  f"(Δnll={b_delta:+.4f}, rank {bsl_rank:.1f}→{b_rank:.1f}). "
                  f"May help in high-confidence router buckets.")
    elif not math.isnan(o_delta) and o_delta <= -0.2 and b_delta > 0.05:
        support = "PARTIAL_SUPPORT"
        reason = (f"Learned router too weak (Δnll={b_delta:+.4f}) but oracle "
                  f"Δnll={o_delta:+.4f} shows real headroom exists.")
    else:
        support = "NO_SUPPORT"
        reason = (f"All learned soft routing worsens NLL (best Δnll={b_delta:+.4f}). "
                  f"Oracle Δnll={o_delta:+.4f}. "
                  f"Region probabilities are not useful priors at this scale.")

    b = best_row
    lines: list[str] = [
        "# Soft Hierarchical Routing Evaluation — Summary",
        "",
        f"**Layer evaluated:** {final_layer}",
        f"**Baseline NLL:** {bsl_nll:.4f}  PPL: {bsl_ppl:.2f}  "
        f"mean_rank: {bsl_rank:.1f}  top1: {baseline['top1_acc']:.4f}",
        "",
        "---",
        "",
        "## Q1 — Does soft coarse routing improve or preserve NLL?",
    ]
    if best_coarse:
        lines.append(
            f"Best coarse-only: α={best_coarse['alpha']} T={best_coarse['temperature']}  "
            f"Δnll={float(best_coarse['delta_nll']):+.4f}  rank={float(best_coarse['mean_rank']):.1f}"
        )
    lines += [
        "",
        "## Q2 — Does soft leaf routing help or hurt?",
    ]
    if best_leaf:
        lines.append(
            f"Best leaf-only: β={best_leaf['beta']} T={best_leaf['temperature']}  "
            f"Δnll={float(best_leaf['delta_nll']):+.4f}  rank={float(best_leaf['mean_rank']):.1f}"
        )
    lines += [
        "",
        "## Q3 — Is coarse+leaf better than either alone?",
        f"Best combined: α={b['alpha']} β={b['beta']} T={b['temperature']}  "
        f"Δnll={float(b['delta_nll']):+.4f}  rank={float(b['mean_rank']):.1f}  "
        f"top1={float(b['top1_acc']):.4f}",
        "",
        "## Q4 — Best alpha/beta/temperature?",
        f"α={b['alpha']}  β={b['beta']}  T={b['temperature']}  "
        f"(tag: {b['tag']})",
        "",
        "## Q5 — Does soft routing avoid hard-mask failures?",
        "Soft routing always keeps full vocabulary — failure rate = 0 by construction.",
        f"Hard mask best effective NLL: {hard_best_eff:.4f}  (Δ vs baseline: {hard_best_delta:+.4f})",
        f"Soft routing best NLL: {float(b['mean_nll']):.4f}  (Δ vs baseline: {float(b['delta_nll']):+.4f})",
        "",
        "## Q6 — Does oracle soft routing show headroom?",
    ]
    if best_oracle:
        lines.append(
            f"Best oracle: mode={best_oracle['mode']}  margin={best_oracle['margin']}  "
            f"Δnll={float(best_oracle['delta_nll']):+.4f}  rank={float(best_oracle['mean_rank']):.1f}"
        )
    else:
        lines.append("Oracle results not available.")

    lines += [
        "",
        "## Q7 — Is this evidence for region-conditioned softmax?",
    ]
    if support == "STRONG_SUPPORT":
        lines.append(
            "**YES**: Region log-probabilities act as useful continuous priors. "
            "NLL improves and gold rank decreases."
        )
    elif support == "PARTIAL_SUPPORT":
        lines.append(
            "**PARTIAL**: Region priors help in some conditions (see bucket analysis). "
            "Router quality may be the limiting factor."
        )
    else:
        lines.append(
            "**NO**: Region probabilities do not act as useful priors at this scale. "
            "Hard routing failures are not the only problem — the regions themselves "
            "may not align well with the prediction distribution."
        )

    lines += [
        "",
        "---",
        "",
        "## Sweep summary table (top 10 by Δnll)",
        "",
        "| tag | α | β | T | Δnll | rank | top1 |",
        "|-----|---|---|---|------|------|------|",
    ]
    for r in sorted(sweep_rows, key=lambda r: r["delta_nll"])[:10]:
        lines.append(
            f"| {r['tag']:22s} | {r['alpha']:.2f} | {r['beta']:.2f} | "
            f"{r['temperature']:.1f} | {float(r['delta_nll']):+.4f} | "
            f"{float(r['mean_rank']):.1f} | {float(r['top1_acc']):.4f} |"
        )

    if mixture_11:
        lines += [
            "",
            "## Hierarchical mixture (α=β=1, T=1, additive_raw)",
            f"Δnll={float(mixture_11['delta_nll']):+.4f}  "
            f"rank={float(mixture_11['mean_rank']):.1f}  "
            f"top1={float(mixture_11['top1_acc']):.4f}",
        ]

    lines += [
        "",
        "---",
        "",
        f"## Verdict: **{support}**",
        "",
        f"*{reason}*",
        "",
        "**Interpretation note:** "
        "Soft routing cannot 'improve GPT-2' — it can only redistribute probability "
        "mass. A negative Δnll means the region prior aligns with GPT-2's prediction "
        "distribution; a positive Δnll means it conflicts.",
        "",
        "---",
        "*Generated by soft_routing_eval.py*",
    ]

    summary_path = os.path.join(sr_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log.info(f"Saved {summary_path}")
    log.info(f"\nVERDICT: {support}\n{reason}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_soft_routing_eval(args, device: torch.device, out_dir: str) -> None:
    """
    Main entry point called from full_vocab_region_eval.py.

    args must provide: model_name, layers, seed, probe_epochs, probe_lr,
                       probe_batch_size, vocab_subset_size.
    Optional: args.soft_routing_layer (defaults to max(layers)).
    """
    from transformers import AutoModelForCausalLM

    log = _log()
    log.info("=" * 60)
    log.info("STAGE: soft_routing_eval")
    log.info("=" * 60)

    sr_dir = os.path.join(out_dir, "soft_routing_eval")
    _ensure(sr_dir)

    target_layers = [int(x) for x in args.layers.split(",")]
    final_layer = getattr(args, "soft_routing_layer", max(target_layers))

    # ── Load artifacts ──────────────────────────────────────────────────────
    log.info("Loading artifacts …")
    top_ids     = np.load(os.path.join(out_dir, "frequent_token_ids.npy"))
    gold_tokens = np.load(os.path.join(out_dir, "gold_tokens.npy"))
    leaf_labels = np.load(os.path.join(out_dir, "leaf_labels.npy"))
    V = len(top_ids)
    N = len(gold_tokens)

    with open(os.path.join(out_dir, "region_tree.json")) as f:
        region_tree = json.load(f)
    with open(os.path.join(out_dir, "leaf_region_to_tokens.json")) as f:
        leaf_region_to_tokens_str = json.load(f)
    leaf_region_to_tokens: dict[int, list[int]] = {
        int(k): v for k, v in leaf_region_to_tokens_str.items()
    }

    coarse_id_arr, n_coarse = derive_coarse_labels(region_tree, V)
    n_leaves = len(leaf_region_to_tokens)

    # leaf_id_arr: sub-index → leaf_id (-1 if unassigned)
    leaf_id_arr = np.full(V, -1, dtype=np.int32)
    with open(os.path.join(out_dir, "leaf_token_to_region.json")) as f:
        l2r = json.load(f)
    for tok_str, lid in l2r.items():
        si = int(tok_str)
        if 0 <= si < V:
            leaf_id_arr[si] = lid

    # Coarse labels for cached samples (gold_tokens are sub-indices)
    coarse_labels_samples = coarse_id_arr[gold_tokens]   # (N,)

    log.info(
        f"V={V:,}  N={N:,}  n_coarse={n_coarse}  n_leaves={n_leaves}  "
        f"final_layer={final_layer}"
    )
    log.info(
        f"Coarse coverage: {(coarse_id_arr >= 0).sum():,}/{V:,}  "
        f"Leaf coverage: {(leaf_id_arr >= 0).sum():,}/{V:,}"
    )

    # ── Load model for LM head ──────────────────────────────────────────────
    log.info(f"Loading model {args.model_name} for LM head …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map={"": device}
    )
    model.eval()
    ln_f, wte_sub = _load_lm_head(model, top_ids)

    # ── Train/test split ─────────────────────────────────────────────────────
    tr_idx, te_idx = _split_indices(N, args.seed)
    log.info(f"Train/test split: {len(tr_idx):,} / {len(te_idx):,}")

    # ── Step 1 ───────────────────────────────────────────────────────────────
    routers = step1_train_routers(
        out_dir, sr_dir,
        [final_layer],
        leaf_labels, coarse_labels_samples,
        n_leaves, n_coarse,
        device, args.seed,
        args.probe_epochs, args.probe_lr, args.probe_batch_size,
    )

    if final_layer not in routers:
        log.error(f"No router trained for layer {final_layer}. Aborting soft routing eval.")
        return

    coarse_probe, leaf_probe, coarse_m, leaf_m = routers[final_layer]

    # ── Step 2 ───────────────────────────────────────────────────────────────
    baseline = step2_baseline(
        out_dir, sr_dir, final_layer, te_idx,
        gold_tokens, ln_f, wte_sub,
    )
    baseline_nll_per_sample = np.load(os.path.join(sr_dir, "baseline_nll.npy"))

    # ── Step 3 ───────────────────────────────────────────────────────────────
    sweep_rows = step3_soft_sweep(
        out_dir, sr_dir, final_layer, te_idx,
        gold_tokens, coarse_probe, leaf_probe,
        coarse_id_arr, leaf_id_arr,
        ln_f, wte_sub, device,
        baseline["mean_nll"],
    )

    # ── Step 4 ───────────────────────────────────────────────────────────────
    step4_oracle(
        out_dir, sr_dir, final_layer, te_idx,
        gold_tokens, coarse_id_arr, leaf_id_arr,
        ln_f, wte_sub, baseline["mean_nll"],
    )

    # ── Step 5 ───────────────────────────────────────────────────────────────
    step5_hard_mask_baseline(
        out_dir, sr_dir, final_layer, te_idx,
        gold_tokens, coarse_probe, leaf_probe,
        coarse_id_arr, leaf_id_arr,
        leaf_region_to_tokens, ln_f, wte_sub,
        device, baseline["mean_nll"],
    )

    # ── Step 6 ───────────────────────────────────────────────────────────────
    step6_hierarchical_mixture(
        out_dir, sr_dir, final_layer, te_idx,
        gold_tokens, coarse_probe, leaf_probe,
        coarse_id_arr, leaf_id_arr,
        ln_f, wte_sub, device, baseline["mean_nll"],
    )

    # ── Step 7 ───────────────────────────────────────────────────────────────
    best_row = min(sweep_rows, key=lambda r: r["delta_nll"])
    step7_bucket_analysis(
        out_dir, sr_dir, final_layer, te_idx,
        gold_tokens, coarse_probe, leaf_probe,
        coarse_id_arr, leaf_id_arr,
        ln_f, wte_sub, device,
        float(best_row["alpha"]),
        float(best_row["beta"]),
        float(best_row["temperature"]),
        baseline_nll_per_sample,
    )

    # ── Step 8 ───────────────────────────────────────────────────────────────
    step8_summary(sr_dir, baseline, sweep_rows, final_layer)

    log.info("=" * 60)
    log.info("soft_routing_eval complete.")
