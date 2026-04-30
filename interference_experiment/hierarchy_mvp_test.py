#!/usr/bin/env python3
"""
hierarchy_mvp_test.py

Tests whether GPT-2 XL builds hierarchical token partitions across layers.
Three MVP experiments on cached hidden states (no model re-run for Parts A/B):

  Part A — Recursive Predictability
            Linear probes at every cached layer for coarse→fine labels.

  Part B — Information Gain Curves
            Bits recovered per layer per granularity level.

  Part C — Logit Locality
            Top-K final logits: do they concentrate inside the top-1's cluster?

Usage
-----
python interference_experiment/hierarchy_mvp_test.py \\
    --cache_dir  interference_experiment/results/probe_cache \\
    --cluster_map interference_experiment/results/cluster_tokens_coactivation_final.json \\
    --subcluster_map interference_experiment/results/cluster7_subcluster_mapping.json \\
    --labels   interference_experiment/graphs/layer_final_coactivation_labels_louvain.npy \\
    --freq_ids interference_experiment/graphs/layer_final_frequent_token_ids.npy \\
    --output_dir interference_experiment/results/hierarchy_mvp
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
    _tqdm = tqdm
except ImportError:
    _tqdm = lambda x, **kw: x


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("hierarchy")


log = logging.getLogger("hierarchy")


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch linear probe
# ─────────────────────────────────────────────────────────────────────────────

class _LinearProbe:
    """Thin wrapper: StandardScaler + nn.Linear trained with Adam."""

    def __init__(self, linear: nn.Linear, scaler: StandardScaler, device: torch.device) -> None:
        self.linear = linear.eval()
        self.scaler = scaler
        self.device = device

    def predict_proba(self, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
        self.linear.eval()
        X_s = self.scaler.transform(X.astype(np.float32))
        parts = []
        with torch.no_grad():
            t = torch.from_numpy(X_s)
            for i in range(0, len(t), chunk):
                logits = self.linear(t[i : i + chunk].to(self.device)).cpu()
                parts.append(F.softmax(logits, dim=-1).numpy())
        return np.concatenate(parts, axis=0)


def fit_probe(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_classes: int,
    device: torch.device,
    seed: int = 42,
    epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 2048,
    weight_decay: float = 1e-4,
) -> _LinearProbe:
    _set_seed(seed)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_tr.astype(np.float32))
    N = len(X_s)
    linear = nn.Linear(X_s.shape[1], n_classes).to(device)
    X_t = torch.from_numpy(X_s).to(device)
    y_t = torch.from_numpy(y_tr.astype(np.int64)).to(device)
    opt = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss_fn(linear(X_t[idx]), y_t[idx]).backward()
            opt.step()
    return _LinearProbe(linear, scaler, device)


def topk_recall(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    top_k = np.argpartition(probs, -k, axis=1)[:, -k:]
    return float(np.mean([labels[i] in top_k[i] for i in range(len(labels))]))


def ce_bits(probs: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy in bits. Avoids sklearn log_loss class-count mismatch bug."""
    eps = 1e-12
    true_p = probs[np.arange(len(labels)), labels.astype(np.int64)]
    return float(-np.log2(np.clip(true_p, eps, 1.0)).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(
    cache_dir: str,
) -> Tuple[Dict[int, str], np.ndarray, np.ndarray, List[int]]:
    """
    Returns hidden state file paths (lazy), clusters, tokens, layer list.

    hidden_paths[l] = path to hidden_L{l:02d}.npy  — loaded on demand
    clusters[i]     = coarse cluster of gold next-token at position i
    tokens[i]       = vocab ID of gold next-token at position i
    """
    meta = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta):
        log.error("meta.json not found in %s", cache_dir); sys.exit(1)
    with open(meta) as f:
        m = json.load(f)
    layers: List[int] = m["layers"]
    hidden_paths: Dict[int, str] = {}
    for l in layers:
        p = os.path.join(cache_dir, f"hidden_L{l:02d}.npy")
        if not os.path.exists(p):
            log.error("Missing %s", p); sys.exit(1)
        hidden_paths[l] = p
    clusters = np.load(os.path.join(cache_dir, "clusters.npy"))
    tokens   = np.load(os.path.join(cache_dir, "tokens.npy"))
    log.info("Cache: N=%d  layers=%s", len(clusters), layers)
    return hidden_paths, clusters, tokens, layers


def load_vocab_map(path: Optional[str], vocab_size: int = 50257) -> Optional[np.ndarray]:
    """
    Load a JSON cluster/subcluster map and return a dense int32 array of shape
    (vocab_size,) where arr[vocab_id] = label_id, -1 if unmapped.

    Accepts three JSON formats:
      1. Direct:    {"token_id_str": label_id, ...}
      2. Cluster:   {"label_id_str": [{"token_id": X}, ...], ...}
      3. Subcluster: {"vocab_to_subcluster": {"token_id_str": label_id}, ...}
    """
    if path is None or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    arr = np.full(vocab_size, -1, dtype=np.int32)

    # Format 3: subcluster mapping (from recursive_cluster7.py)
    if "vocab_to_subcluster" in data:
        for vid_str, lab in data["vocab_to_subcluster"].items():
            vid = int(vid_str)
            if 0 <= vid < vocab_size:
                arr[vid] = int(lab)

    # Format 2: cluster JSON {"0": [{"token_id": X}, ...]}
    elif all(
        isinstance(v, list) for v in data.values()
    ):
        for lab_str, entries in data.items():
            lab = int(lab_str)
            for e in entries:
                vid = int(e.get("token_id", -1))
                if 0 <= vid < vocab_size:
                    arr[vid] = lab

    # Format 1: direct mapping {"vid_str": label_id}
    else:
        for vid_str, lab in data.items():
            vid = int(vid_str)
            if 0 <= vid < vocab_size:
                arr[vid] = int(lab)

    n_mapped = int((arr >= 0).sum())
    n_classes = int(arr.max()) + 1 if n_mapped > 0 else 0
    log.info("  %s: %d mapped tokens, %d classes", Path(path).name, n_mapped, n_classes)
    return arr


def build_coarse_map_from_npy(
    labels_path: str,
    freq_ids_path: str,
    vocab_size: int = 50257,
) -> np.ndarray:
    """Build vocab→coarse_cluster from labels.npy + freq_ids.npy (full coverage)."""
    labels   = np.load(labels_path).astype(np.int32)
    freq_ids = np.load(freq_ids_path).astype(np.int32)
    arr = np.full(vocab_size, -1, dtype=np.int32)
    for local_idx, vocab_id in enumerate(freq_ids):
        if 0 <= vocab_id < vocab_size:
            arr[vocab_id] = labels[local_idx]
    n_mapped = int((arr >= 0).sum())
    n_classes = int(arr.max()) + 1
    log.info("  labels+freq_ids: %d mapped tokens, %d classes", n_mapped, n_classes)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Task configuration
# ─────────────────────────────────────────────────────────────────────────────

class TaskConfig:
    """Everything needed to run one probe task (coarse / sub / token)."""

    def __init__(
        self,
        name: str,
        labels: np.ndarray,       # (N,) full label array, -1 = invalid
        n_classes: int,
        train_idx: np.ndarray,    # indices into the valid subset for training
        test_idx:  np.ndarray,    # indices into the valid subset for test
        valid_global: np.ndarray, # boolean mask over all N samples
    ) -> None:
        self.name        = name
        self.labels      = labels
        self.n_classes   = n_classes
        self.train_idx   = train_idx
        self.test_idx    = test_idx
        self.valid_global = valid_global

    @property
    def y_tr(self) -> np.ndarray:
        valid_pos = np.where(self.valid_global)[0]
        return self.labels[valid_pos[self.train_idx]]

    @property
    def y_te(self) -> np.ndarray:
        valid_pos = np.where(self.valid_global)[0]
        return self.labels[valid_pos[self.test_idx]]

    def X_tr(self, h: np.ndarray) -> np.ndarray:
        valid_pos = np.where(self.valid_global)[0]
        return h[valid_pos[self.train_idx]]

    def X_te(self, h: np.ndarray) -> np.ndarray:
        valid_pos = np.where(self.valid_global)[0]
        return h[valid_pos[self.test_idx]]

    @property
    def N(self) -> int:
        return int(self.valid_global.sum())


def build_task_configs(
    clusters: np.ndarray,
    tokens: np.ndarray,
    vocab_map_sub: Optional[np.ndarray],
    vocab_map_subsub: Optional[np.ndarray],
    token_vocab_k: int,
    train_frac: float,
    seed: int,
) -> List[TaskConfig]:
    """
    Build one TaskConfig per granularity level with consistent 70/30 splits.
    """
    N = len(clusters)
    rng = np.random.default_rng(seed)

    def _make_task(name, labels, n_classes):
        valid = labels >= 0
        if valid.sum() < 100:
            log.warning("Task '%s': only %d valid samples — skipping.", name, valid.sum())
            return None
        valid_pos = np.where(valid)[0]
        perm = rng.permutation(len(valid_pos))
        n_tr = int(train_frac * len(valid_pos))
        return TaskConfig(
            name=name,
            labels=labels,
            n_classes=n_classes,
            train_idx=perm[:n_tr],
            test_idx=perm[n_tr:],
            valid_global=valid,
        )

    tasks: List[TaskConfig] = []

    # ── Coarse cluster (from cache clusters.npy)
    coarse_labels = clusters.astype(np.int32)
    n_coarse = int(coarse_labels[coarse_labels >= 0].max()) + 1
    t = _make_task("coarse_cluster", coarse_labels, n_coarse)
    if t: tasks.append(t)

    # ── Subcluster (optional)
    if vocab_map_sub is not None:
        sub_labels = vocab_map_sub[tokens.clip(0, len(vocab_map_sub) - 1)].astype(np.int32)
        n_sub = int(sub_labels[sub_labels >= 0].max()) + 1
        t = _make_task("subcluster", sub_labels, n_sub)
        if t: tasks.append(t)

    # ── Sub-subcluster (optional)
    if vocab_map_subsub is not None:
        subsub_labels = vocab_map_subsub[tokens.clip(0, len(vocab_map_subsub) - 1)].astype(np.int32)
        n_subsub = int(subsub_labels[subsub_labels >= 0].max()) + 1
        t = _make_task("subsubcluster", subsub_labels, n_subsub)
        if t: tasks.append(t)

    # ── Exact token (compact vocab of top-K frequent tokens in cache)
    seen, counts = np.unique(tokens, return_counts=True)
    top_k_ids = seen[np.argsort(-counts)[:token_vocab_k]]
    tok_map = np.full(50257, -1, dtype=np.int32)
    for compact, vid in enumerate(top_k_ids):
        tok_map[int(vid)] = compact
    tok_labels = tok_map[tokens.clip(0, 50256)].astype(np.int32)
    K = int(tok_labels[tok_labels >= 0].max()) + 1
    t = _make_task("token", tok_labels, K)
    if t: tasks.append(t)

    for task in tasks:
        log.info(
            "Task %-20s: N=%6d  n_classes=%4d  train=%d  test=%d",
            task.name, task.N, task.n_classes,
            len(task.train_idx), len(task.test_idx),
        )
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Part A — Recursive Predictability
# ─────────────────────────────────────────────────────────────────────────────

def run_part_A(
    hidden_paths: Dict[int, str],
    layers: List[int],
    tasks: List[TaskConfig],
    probe_device: torch.device,
    args,
) -> List[dict]:
    """
    Train one linear probe per (layer, task). Returns flat list of metric dicts.
    Hidden states loaded one layer at a time — never holds full cache in RAM.
    """
    rows: List[dict] = []

    for l in layers:
        log.info("=== Layer %d ===", l)
        h = np.load(hidden_paths[l])   # (N, D) float16

        for task in tasks:
            t0 = time.time()
            X_tr = task.X_tr(h)
            X_te = task.X_te(h)
            y_tr = task.y_tr
            y_te = task.y_te

            probe = fit_probe(
                X_tr, y_tr, task.n_classes, probe_device,
                seed=args.seed, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size,
            )
            probs = probe.predict_proba(X_te)

            top1 = float((np.argmax(probs, axis=1) == y_te).mean())
            top5 = topk_recall(probs, y_te, 5) if task.n_classes > 5 else 1.0
            ce   = ce_bits(probs, y_te)
            rand_ce  = math.log2(task.n_classes)
            rand_top1 = 1.0 / task.n_classes

            row = {
                "layer":     l,
                "task":      task.name,
                "n_classes": task.n_classes,
                "n_test":    len(y_te),
                "top1":      round(top1, 4),
                "top5":      round(top5, 4),
                "ce_bits":   round(ce, 4),
                "rand_ce_bits":  round(rand_ce, 4),
                "rand_top1": round(rand_top1, 4),
                "elapsed_s": round(time.time() - t0, 1),
            }
            rows.append(row)
            log.info(
                "  %-20s  top1=%.3f (rand=%.3f)  top5=%.3f  ce=%.3f bits  (%.1fs)",
                task.name, top1, rand_top1, top5, ce, time.time() - t0,
            )

        del h

    return rows


def save_part_A_csv(rows: List[dict], output_dir: str) -> str:
    path = os.path.join(output_dir, "part_a_metrics.csv")
    fields = ["layer", "task", "n_classes", "n_test",
              "top1", "top5", "ce_bits", "rand_ce_bits", "rand_top1", "elapsed_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("Part A CSV → %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Part B — Information Gain Curves
# ─────────────────────────────────────────────────────────────────────────────

def run_part_B(rows: List[dict], layers: List[int], tasks: List[TaskConfig]) -> List[dict]:
    """
    info_recovered_bits(layer, task) = H_theoretical - CE_probe_bits
    H_theoretical = log2(n_classes)  [uniform-distribution upper bound]

    Positive = probe beats random; max = H_theoretical (perfect probe).
    """
    ig_rows = []
    for r in rows:
        h_theory = math.log2(r["n_classes"])
        info_rec = h_theory - r["ce_bits"]
        info_rec_frac = info_rec / h_theory if h_theory > 0 else 0.0
        ig_rows.append({
            "layer":           r["layer"],
            "task":            r["task"],
            "n_classes":       r["n_classes"],
            "H_theory_bits":   round(h_theory, 4),
            "ce_bits":         r["ce_bits"],
            "info_recovered":  round(info_rec, 4),
            "info_frac":       round(info_rec_frac, 4),
        })
    return ig_rows


def save_part_B_csv(ig_rows: List[dict], output_dir: str) -> str:
    path = os.path.join(output_dir, "part_b_info_gain.csv")
    fields = ["layer", "task", "n_classes", "H_theory_bits", "ce_bits",
              "info_recovered", "info_frac"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ig_rows)
    log.info("Part B CSV → %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Part C — Logit Locality
# ─────────────────────────────────────────────────────────────────────────────

def _load_lm_head(model_name: str, device: torch.device):
    """
    Load only ln_f and the LM head weight from the model.
    GPT-2 uses weight-tying: lm_head.weight == wte.weight (vocab_size × d_model).
    Returns (ln_f_module, lm_weight_tensor) both on device.
    """
    from transformers import AutoModelForCausalLM
    log.info("Loading %s for LM head (will discard after extraction)…", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    ln_f      = model.transformer.ln_f.eval().float()
    lm_weight = model.transformer.wte.weight.detach().float()   # (V, D)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ln_f      = ln_f.to(device)
    lm_weight = lm_weight.to(device)
    log.info("LM head extracted: weight shape %s", tuple(lm_weight.shape))
    return ln_f, lm_weight


def run_part_C(
    hidden_paths: Dict[int, str],
    last_layer: int,
    vocab_map_coarse: Optional[np.ndarray],
    vocab_map_sub: Optional[np.ndarray],
    model_name: str,
    probe_device: torch.device,
    n_samples: int,
    seed: int,
) -> List[dict]:
    """
    Apply LM head to final-layer hidden states → logits.
    For top-K tokens, measure cluster concentration relative to top-1 prediction.

    K values: [10, 50, 100, 500]
    """
    if vocab_map_coarse is None and vocab_map_sub is None:
        log.warning("No vocab map available — skipping Part C.")
        return []

    # Load LM head
    ln_f, lm_weight = _load_lm_head(model_name, probe_device)
    V = lm_weight.shape[0]

    # Load final hidden states
    h_all = np.load(hidden_paths[last_layer]).astype(np.float32)   # (N, D)
    N = len(h_all)

    if n_samples < N:
        idx = np.random.default_rng(seed).choice(N, n_samples, replace=False)
        h_all = h_all[idx]
        log.info("Part C: sampled %d / %d positions", n_samples, N)
    else:
        log.info("Part C: using all %d positions", N)

    ks = [10, 50, 100, 500]
    max_k = max(ks)

    stats: Dict[int, Dict[str, List[float]]] = {
        k: {"same_cluster": [], "same_subcluster": [],
            "unique_clusters": [], "unique_subclusters": []}
        for k in ks
    }

    batch_sz = 256
    n_done = 0

    for start in _tqdm(range(0, len(h_all), batch_sz), desc="Part C logit batches"):
        h_b = torch.from_numpy(h_all[start : start + batch_sz]).to(probe_device)

        with torch.no_grad():
            h_ln = ln_f(h_b)
            logits = (h_ln @ lm_weight.T).cpu()   # (B, V)

        # torch.topk returns sorted descending — index 0 is argmax
        topk_ids = torch.topk(logits, max_k, dim=1).indices.numpy()   # (B, max_k)
        top1_ids = topk_ids[:, 0]                                       # (B,)

        for k in ks:
            topk_k = topk_ids[:, :k]   # (B, k)

            # ── Coarse cluster concentration ────────────────────────────────
            if vocab_map_coarse is not None:
                top1_c = vocab_map_coarse[top1_ids.clip(0, V - 1)]    # (B,)
                topk_c = vocab_map_coarse[topk_k.clip(0, V - 1)]      # (B, k)
                known  = topk_c >= 0                                    # (B, k)
                valid_top1 = top1_c >= 0                               # (B,)

                for bi in np.where(valid_top1)[0]:
                    k_known = topk_c[bi][known[bi]]
                    if len(k_known) == 0:
                        continue
                    stats[k]["same_cluster"].append(
                        float((k_known == top1_c[bi]).mean())
                    )
                    stats[k]["unique_clusters"].append(
                        float(len(np.unique(k_known)))
                    )

            # ── Subcluster concentration ────────────────────────────────────
            if vocab_map_sub is not None:
                top1_s = vocab_map_sub[top1_ids.clip(0, V - 1)]   # (B,)
                topk_s = vocab_map_sub[topk_k.clip(0, V - 1)]     # (B, k)
                known_s = topk_s >= 0
                valid_top1_s = top1_s >= 0

                for bi in np.where(valid_top1_s)[0]:
                    k_known_s = topk_s[bi][known_s[bi]]
                    if len(k_known_s) == 0:
                        continue
                    stats[k]["same_subcluster"].append(
                        float((k_known_s == top1_s[bi]).mean())
                    )
                    stats[k]["unique_subclusters"].append(
                        float(len(np.unique(k_known_s)))
                    )

        n_done += len(h_b)

    del lm_weight, ln_f
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    for k in ks:
        def _mean(lst):
            return round(float(np.mean(lst)), 4) if lst else float("nan")
        row = {
            "K": k,
            "same_cluster_frac":      _mean(stats[k]["same_cluster"]),
            "same_subcluster_frac":   _mean(stats[k]["same_subcluster"]),
            "unique_clusters":        _mean(stats[k]["unique_clusters"]),
            "unique_subclusters":     _mean(stats[k]["unique_subclusters"]),
            "n_samples_cluster":      len(stats[k]["same_cluster"]),
            "n_samples_subcluster":   len(stats[k]["same_subcluster"]),
        }
        rows.append(row)
        log.info(
            "  K=%-4d  same_cluster=%.3f  same_sub=%.3f  "
            "uniq_clusters=%.1f  uniq_sub=%.1f",
            k, row["same_cluster_frac"], row["same_subcluster_frac"],
            row["unique_clusters"], row["unique_subclusters"],
        )
    return rows


def save_part_C_csv(rows: List[dict], output_dir: str) -> str:
    if not rows:
        return ""
    path = os.path.join(output_dir, "part_c_logit_locality.csv")
    fields = ["K", "same_cluster_frac", "same_subcluster_frac",
              "unique_clusters", "unique_subclusters",
              "n_samples_cluster", "n_samples_subcluster"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("Part C CSV → %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

_TASK_STYLE = {
    "coarse_cluster":  ("tab:blue",   "o-",  "Coarse cluster"),
    "subcluster":      ("tab:orange", "s--", "Subcluster"),
    "subsubcluster":   ("tab:green",  "^:",  "Sub-subcluster"),
    "token":           ("tab:red",    "D:",  "Exact token"),
}


def _plot_setup():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return None


def plot_recursive_predictability(
    rows: List[dict],
    layers: List[int],
    output_dir: str,
) -> None:
    plt = _plot_setup()
    if plt is None:
        return

    task_names = list(dict.fromkeys(r["task"] for r in rows))
    data_by_task: Dict[str, Dict[int, dict]] = {t: {} for t in task_names}
    for r in rows:
        data_by_task[r["task"]][r["layer"]] = r

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xlabels = [str(l) for l in layers]

    for task in task_names:
        color, ls, label = _TASK_STYLE.get(
            task, ("tab:gray", "-", task)
        )
        d = data_by_task[task]
        top1 = [d[l]["top1"] if l in d else float("nan") for l in layers]
        top5 = [d[l]["top5"] if l in d else float("nan") for l in layers]
        rand  = d[layers[0]]["rand_top1"]

        axes[0].plot(xlabels, top1, ls, color=color, lw=2, label=label)
        axes[0].axhline(rand, ls=":", color=color, alpha=0.35, lw=1)

        axes[1].plot(xlabels, top5, ls, color=color, lw=2, label=label)

    axes[0].set_title("Top-1 accuracy by layer")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_title("Top-5 accuracy by layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Top-5 accuracy")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    fig.suptitle("Part A — Recursive Predictability: coarse → fine across layers", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "recursive_predictability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Plot → %s", path)


def plot_info_gain(
    ig_rows: List[dict],
    layers: List[int],
    output_dir: str,
) -> None:
    plt = _plot_setup()
    if plt is None:
        return

    task_names = list(dict.fromkeys(r["task"] for r in ig_rows))
    by_task: Dict[str, Dict[int, dict]] = {t: {} for t in task_names}
    for r in ig_rows:
        by_task[r["task"]][r["layer"]] = r

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xlabels = [str(l) for l in layers]

    for task in task_names:
        color, ls, label = _TASK_STYLE.get(task, ("tab:gray", "-", task))
        d = by_task[task]
        h_theory = d[layers[0]]["H_theory_bits"]
        rec  = [d[l]["info_recovered"] if l in d else float("nan") for l in layers]
        frac = [d[l]["info_frac"]      if l in d else float("nan") for l in layers]

        axes[0].plot(xlabels, rec,  ls, color=color, lw=2,
                     label=f"{label}  (H={h_theory:.2f} bits)")
        axes[1].plot(xlabels, frac, ls, color=color, lw=2, label=label)

    axes[0].axhline(0, ls="-", color="gray", lw=0.8)
    axes[0].set_title("Bits recovered  =  H_theory − CE_probe")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Bits recovered")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0, ls="-", color="gray", lw=0.8)
    axes[1].axhline(1, ls=":", color="gray", lw=0.8)
    axes[1].set_title("Fraction of max entropy recovered")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("info_frac  (0=random, 1=perfect)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(-0.1, 1.05)

    fig.suptitle("Part B — Information Gain: bits recovered per level per layer", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "info_gain_bits.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Plot → %s", path)


def plot_logit_locality(part_c_rows: List[dict], output_dir: str) -> None:
    plt = _plot_setup()
    if plt is None or not part_c_rows:
        return

    ks      = [r["K"]                  for r in part_c_rows]
    same_c  = [r["same_cluster_frac"]  for r in part_c_rows]
    same_s  = [r["same_subcluster_frac"] for r in part_c_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, same_c, "o-", color="tab:blue",   lw=2, label="Same coarse cluster as top-1")
    ax.plot(ks, same_s, "s--", color="tab:orange", lw=2, label="Same subcluster as top-1")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("K (top-K logit candidates)")
    ax.set_ylabel("Fraction in same cluster as top-1")
    ax.set_title("Part C — Logit Locality: cluster concentration in top-K")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(output_dir, "logit_locality.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Plot → %s", path)


def plot_candidate_diversity(part_c_rows: List[dict], output_dir: str) -> None:
    plt = _plot_setup()
    if plt is None or not part_c_rows:
        return

    ks       = [r["K"]                for r in part_c_rows]
    uniq_c   = [r["unique_clusters"]  for r in part_c_rows]
    uniq_s   = [r["unique_subclusters"] for r in part_c_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, uniq_c, "o-",  color="tab:blue",   lw=2, label="Unique coarse clusters")
    ax.plot(ks, uniq_s, "s--", color="tab:orange",  lw=2, label="Unique subclusters")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("K (top-K logit candidates)")
    ax.set_ylabel("Unique clusters in top-K")
    ax.set_title("Part C — Candidate Diversity: cluster spread in top-K")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "candidate_diversity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Plot → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Interpretation & summary
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(
    part_a_rows: List[dict],
    ig_rows: List[dict],
    part_c_rows: List[dict],
    tasks: List[TaskConfig],
    layers: List[int],
    output_dir: str,
) -> None:
    lines = [
        "HIERARCHY MVP TEST — INTERPRETATION SUMMARY",
        "=" * 60,
        "",
    ]

    # ── Index Part A by (layer, task) ────────────────────────────────────────
    a_idx: Dict[Tuple, dict] = {(r["layer"], r["task"]): r for r in part_a_rows}
    task_names = [t.name for t in tasks]

    # ── 1. Hierarchy ordering check ─────────────────────────────────────────
    lines.append("1. HIERARCHY ORDERING  (coarse > subcluster > token at each layer?)")
    lines.append("-" * 60)

    ordered_layers = 0
    for l in layers:
        top1s = []
        for t in task_names:
            r = a_idx.get((l, t))
            if r:
                top1s.append((t, r["top1"]))
        ordered = all(top1s[i][1] >= top1s[i + 1][1]
                      for i in range(len(top1s) - 1))
        mark = "✓" if ordered else "✗"
        lines.append(
            f"  Layer {l:2d}: {mark}  "
            + "  ".join(f"{t}={v:.3f}" for t, v in top1s)
        )
        if ordered:
            ordered_layers += 1

    frac_ordered = ordered_layers / len(layers) if layers else 0.0
    lines.append(f"\n  Hierarchy ordering holds at {ordered_layers}/{len(layers)} layers "
                 f"({100*frac_ordered:.0f}%).")
    lines.append("")

    # ── 2. Earliest layer where cluster acc > token acc by ≥ 0.10 ──────────
    lines.append("2. EARLIEST DIVERGENCE  (cluster top-1 − token top-1 ≥ 0.10)")
    lines.append("-" * 60)
    diverge_layer = None
    for l in layers:
        r_coarse = a_idx.get((l, "coarse_cluster"))
        r_token  = a_idx.get((l, "token"))
        if r_coarse and r_token:
            gap = r_coarse["top1"] - r_token["top1"]
            lines.append(f"  Layer {l:2d}: cluster={r_coarse['top1']:.3f}  "
                          f"token={r_token['top1']:.3f}  gap={gap:+.3f}")
            if gap >= 0.10 and diverge_layer is None:
                diverge_layer = l
    if diverge_layer is not None:
        lines.append(f"\n  → First layer with gap ≥ 0.10: Layer {diverge_layer}")
    else:
        lines.append("\n  → Gap never reached 0.10 in sampled layers.")
    lines.append("")

    # ── 3. Information gain ordering ────────────────────────────────────────
    lines.append("3. INFORMATION GAIN  (does coarser label recover more bits earlier?)")
    lines.append("-" * 60)
    ig_idx: Dict[Tuple, dict] = {(r["layer"], r["task"]): r for r in ig_rows}
    for l in layers:
        fracs = []
        for t in task_names:
            r = ig_idx.get((l, t))
            if r:
                fracs.append(f"{t}={r['info_frac']:.3f}")
        lines.append(f"  Layer {l:2d}: " + "  ".join(fracs))
    lines.append("")

    # ── 4. Logit locality ───────────────────────────────────────────────────
    if part_c_rows:
        lines.append("4. LOGIT LOCALITY  (top-K logit concentration in same cluster)")
        lines.append("-" * 60)
        for r in part_c_rows:
            lines.append(
                f"  K={r['K']:4d}:  same_cluster={r['same_cluster_frac']:.3f}  "
                f"same_subcluster={r['same_subcluster_frac']:.3f}  "
                f"unique_clusters={r['unique_clusters']:.1f}  "
                f"unique_subclusters={r['unique_subclusters']:.1f}"
            )
        # Is top-100 mostly in one cluster?
        r100 = next((r for r in part_c_rows if r["K"] == 100), None)
        if r100 and r100["same_cluster_frac"] > 0.70:
            lines.append(
                f"\n  → Top-100 logits are {r100['same_cluster_frac']:.0%} in the same "
                f"coarse cluster as top-1. Flat 50k softmax is wasteful."
            )
        elif r100:
            lines.append(
                f"\n  → Top-100 logits span {r100['unique_clusters']:.1f} clusters on average. "
                f"Moderate locality — partial routing gain possible."
            )
        lines.append("")

    # ── 5. Nested routing verdict ────────────────────────────────────────────
    lines.append("5. NESTED ROUTING HYPOTHESIS VERDICT")
    lines.append("-" * 60)
    supports = []
    against  = []

    if frac_ordered >= 0.75:
        supports.append(f"Hierarchy ordering holds at {100*frac_ordered:.0f}% of layers.")
    else:
        against.append(f"Hierarchy ordering holds at only {100*frac_ordered:.0f}% of layers.")

    if diverge_layer is not None and layers.index(diverge_layer) <= len(layers) // 2:
        supports.append(f"Coarse cluster already easier than exact token by layer {diverge_layer}.")
    elif diverge_layer is None:
        against.append("Cluster never pulls ahead of token by ≥ 0.10 margin.")

    if part_c_rows:
        r100 = next((r for r in part_c_rows if r["K"] == 100), None)
        if r100 and r100["same_cluster_frac"] > 0.70:
            supports.append(f"Final logit top-100 is {r100['same_cluster_frac']:.0%} cluster-local.")
        elif r100:
            against.append(f"Final logit top-100 spans {r100['unique_clusters']:.1f} clusters "
                           f"(locality weak).")

    if supports:
        lines.append("  Evidence FOR nested routing:")
        for s in supports:
            lines.append(f"    + {s}")
    if against:
        lines.append("  Evidence AGAINST nested routing:")
        for s in against:
            lines.append(f"    - {s}")

    if len(supports) > len(against):
        lines.append(
            "\n  CONCLUSION: Results SUPPORT the nested routing hypothesis.\n"
            "  GPT-2 XL appears to resolve coarse token identity before fine identity,\n"
            "  and final logits concentrate in few clusters — motivating hierarchical softmax."
        )
    elif len(supports) == len(against):
        lines.append(
            "\n  CONCLUSION: MIXED evidence. More granular experiments needed."
        )
    else:
        lines.append(
            "\n  CONCLUSION: Results do NOT clearly support nested routing.\n"
            "  Token and cluster probes may saturate simultaneously."
        )

    lines.append("")
    lines.append("=" * 60)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Summary → %s", summary_path)
    print("\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical token partition test — GPT-2 XL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── I/O ────────────────────────────────────────────────────────────────
    parser.add_argument("--cache_dir",       required=True,
                        help="probe_cache from layerwise_region_probe.py")
    parser.add_argument("--cluster_map",     default=None,
                        help="JSON: token_id→coarse_cluster  "
                             "(accepts cluster JSON, direct map, or vocab_to_subcluster format)")
    parser.add_argument("--subcluster_map",  default=None,
                        help="JSON: token_id→subcluster  (from recursive_cluster7.py)")
    parser.add_argument("--subsubcluster_map", default=None,
                        help="JSON: token_id→sub-subcluster  (optional)")
    parser.add_argument("--labels",          default=None,
                        help="labels.npy — preferred over --cluster_map for full coverage")
    parser.add_argument("--freq_ids",        default=None,
                        help="frequent_token_ids.npy — required if --labels provided")
    parser.add_argument("--output_dir",      default="results/hierarchy_mvp")
    parser.add_argument("--model_name",      default="gpt2-xl",
                        help="HuggingFace model name (for Part C LM head)")

    # ── Probe training ──────────────────────────────────────────────────────
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--train_frac",      type=float, default=0.7)
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=1e-2)
    parser.add_argument("--batch_size",      type=int,   default=2048)
    parser.add_argument("--token_vocab_k",   type=int,   default=5000,
                        help="Cap on number of token classes for the exact-token probe")

    # ── Part C ─────────────────────────────────────────────────────────────
    parser.add_argument("--skip_part_c",     action="store_true",
                        help="Skip logit locality test (avoids loading model)")
    parser.add_argument("--logit_samples",   type=int,   default=5000,
                        help="Number of positions to use for Part C")
    parser.add_argument("--logit_layer",     type=int,   default=-1,
                        help="Which cached layer to use for Part C logits (-1=last)")

    args = parser.parse_args()

    _setup_logging()
    _set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", probe_device)

    # ── Load cache ──────────────────────────────────────────────────────────
    log.info("=== Loading probe cache ===")
    hidden_paths, clusters, tokens, layers = load_cache(args.cache_dir)
    N = len(clusters)

    # ── Build coarse cluster vocab map (for Part C) ─────────────────────────
    log.info("=== Building vocab maps ===")
    vocab_map_coarse: Optional[np.ndarray] = None
    if args.labels and args.freq_ids:
        log.info("Building coarse map from labels.npy + freq_ids.npy (full coverage)")
        vocab_map_coarse = build_coarse_map_from_npy(args.labels, args.freq_ids)
    elif args.cluster_map:
        log.info("Building coarse map from --cluster_map JSON (top-20 per cluster)")
        vocab_map_coarse = load_vocab_map(args.cluster_map)

    vocab_map_sub: Optional[np.ndarray]    = load_vocab_map(args.subcluster_map)
    vocab_map_subsub: Optional[np.ndarray] = load_vocab_map(args.subsubcluster_map)

    # ── Build task configs ──────────────────────────────────────────────────
    log.info("=== Building task configurations ===")
    tasks = build_task_configs(
        clusters=clusters,
        tokens=tokens,
        vocab_map_sub=vocab_map_sub,
        vocab_map_subsub=vocab_map_subsub,
        token_vocab_k=args.token_vocab_k,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    if not tasks:
        log.error("No valid tasks built. Check input files."); sys.exit(1)

    # ── Part A ───────────────────────────────────────────────────────────────
    log.info("=== Part A: Recursive Predictability ===")
    part_a_rows = run_part_A(hidden_paths, layers, tasks, probe_device, args)
    save_part_A_csv(part_a_rows, args.output_dir)
    plot_recursive_predictability(part_a_rows, layers, args.output_dir)

    # ── Part B ───────────────────────────────────────────────────────────────
    log.info("=== Part B: Information Gain ===")
    ig_rows = run_part_B(part_a_rows, layers, tasks)
    save_part_B_csv(ig_rows, args.output_dir)
    plot_info_gain(ig_rows, layers, args.output_dir)

    # ── Part C ───────────────────────────────────────────────────────────────
    part_c_rows: List[dict] = []
    if not args.skip_part_c:
        log.info("=== Part C: Logit Locality ===")
        logit_layer = args.logit_layer if args.logit_layer >= 0 else layers[-1]
        if logit_layer not in hidden_paths:
            log.warning("Requested logit_layer=%d not in cache; using %d", logit_layer, layers[-1])
            logit_layer = layers[-1]
        part_c_rows = run_part_C(
            hidden_paths=hidden_paths,
            last_layer=logit_layer,
            vocab_map_coarse=vocab_map_coarse,
            vocab_map_sub=vocab_map_sub,
            model_name=args.model_name,
            probe_device=probe_device,
            n_samples=args.logit_samples,
            seed=args.seed,
        )
        save_part_C_csv(part_c_rows, args.output_dir)
        plot_logit_locality(part_c_rows, args.output_dir)
        plot_candidate_diversity(part_c_rows, args.output_dir)
    else:
        log.info("Part C skipped (--skip_part_c).")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("=== Writing summary ===")
    write_summary(part_a_rows, ig_rows, part_c_rows, tasks, layers, args.output_dir)

    log.info("Done. All outputs in %s/", args.output_dir)


if __name__ == "__main__":
    main()
