#!/usr/bin/env python3
"""
layerwise_region_probe.py

Measures how early in a causal LM hidden states can predict token *region*
(coarse coactivation cluster) compared with exact token identity.

Core question
-------------
Do coarse token regions emerge earlier than exact token identity?
If h_t^(l) -> cluster is easy at layer 8 but h_t^(l) -> token is hard until
layer 40, that supports an early-exit / region-routing architecture.

Usage
-----
python layerwise_region_probe.py \\
    --clusters  results/cluster_tokens_coactivation_final.json \\
    --labels    graphs/layer_final_coactivation_labels_louvain.npy \\
    --freq_ids  graphs/layer_final_frequent_token_ids.npy \\
    --output_dir results/ \\
    --max_samples 100000
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("probe")


log = logging.getLogger("probe")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: torch.device, fp16: bool):
    """Load causal LM + tokenizer, falling back to gpt2-medium on OOM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fallback = "gpt2-medium"
    for name in ([model_name] if model_name != fallback else [model_name]):
        try:
            log.info("Loading tokenizer: %s", name)
            tokenizer = AutoTokenizer.from_pretrained(name)

            dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32
            log.info("Loading model: %s  dtype=%s", name, dtype)
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(device).eval()

            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            log.info("Loaded %s  (%.0f M params)", name, n_params)
            return model, tokenizer, name
        except (OSError, RuntimeError) as exc:
            log.warning("Could not load %s: %s", name, exc)

    # Hard fallback
    log.info("Trying fallback model: %s", fallback)
    tokenizer = AutoTokenizer.from_pretrained(fallback)
    model = AutoModelForCausalLM.from_pretrained(fallback).to(device).eval()
    return model, tokenizer, fallback


def get_n_layers(model) -> int:
    """Return the number of transformer blocks."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return len(model.gpt_neox.layers)
    raise ValueError("Cannot determine number of transformer layers from model.")


def compute_target_layers(n_layers: int, user_spec: Optional[str]) -> List[int]:
    """
    Return sorted list of 0-indexed transformer-block layer numbers to probe.
    -1 in user_spec → n_layers-1.
    Default: ~8 evenly spaced across the full depth, always including final.
    """
    if user_spec is not None:
        raw = [int(x.strip()) for x in user_spec.split(",")]
        resolved = [n_layers - 1 if l < 0 else min(l, n_layers - 1) for l in raw]
        return sorted(set(resolved))

    # Requested anchor points: 0, 4, 8, 12, 16, 20, 24, final
    anchors = [0, 4, 8, 12, 16, 20, 24]
    layers = [l for l in anchors if l < n_layers]
    layers.append(n_layers - 1)
    return sorted(set(layers))


# ---------------------------------------------------------------------------
# Cluster mapping
# ---------------------------------------------------------------------------

def load_cluster_mapping(
    clusters_path: str,
    labels_path: Optional[str],
    freq_ids_path: Optional[str],
    vocab_size: int,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Build token_to_cluster: np.ndarray of shape (vocab_size,) with int16.
    Value = cluster_id for tokens that have a cluster, -1 otherwise.

    Also returns cluster_names dict {cluster_id_str: short description}.

    Priority:
      1. labels.npy + freq_ids.npy → full 5000-token mapping
      2. JSON only → top-20 per cluster (warns; limited coverage)
    """
    with open(clusters_path, encoding="utf-8") as f:
        cluster_json: Dict[str, List[dict]] = json.load(f)

    n_clusters = len(cluster_json)
    log.info("Cluster JSON: %d clusters", n_clusters)

    token_to_cluster = np.full(vocab_size, -1, dtype=np.int16)

    # Auto-detect files next to clusters_path if not given
    cdir = str(Path(clusters_path).parent)

    def _try(path: Optional[str], candidates: List[str]) -> Optional[np.ndarray]:
        for c in ([path] if path else []) + candidates:
            if c and os.path.exists(c):
                arr = np.load(c)
                log.info("  Loaded %s  shape=%s", c, arr.shape)
                return arr
        return None

    # Try to get full mapping
    labels = _try(labels_path, [
        # common naming conventions
        os.path.join(cdir, "../graphs/layer_final_coactivation_labels_louvain.npy"),
        os.path.join(cdir, "../graphs/layer_final_labels_louvain.npy"),
        "graphs/layer_final_coactivation_labels_louvain.npy",
        "graphs/layer_final_labels_louvain.npy",
    ])
    freq_ids = _try(freq_ids_path, [
        os.path.join(cdir, "../graphs/layer_final_frequent_token_ids.npy"),
        os.path.join(cdir, "../graphs/layer_final_frequent_token_ids.npy"),
        "graphs/layer_final_frequent_token_ids.npy",
        "graphs/layer_final_frequent_token_ids.npy",
    ])

    if labels is not None and freq_ids is not None:
        labels = labels.astype(np.int16)
        freq_ids = freq_ids.astype(np.int32)
        for local_idx, tok_id in enumerate(freq_ids):
            if 0 <= tok_id < vocab_size:
                token_to_cluster[tok_id] = labels[local_idx]
        n_mapped = int((token_to_cluster >= 0).sum())
        log.info(
            "Full mapping: %d / %d vocab tokens assigned to clusters "
            "(%.1f%% coverage)", n_mapped, vocab_size, 100 * n_mapped / vocab_size
        )
    else:
        log.warning(
            "labels.npy or freq_ids.npy not found — "
            "falling back to JSON top-20 tokens per cluster. "
            "Coverage will be very limited."
        )
        for cid_str, entries in cluster_json.items():
            cid = int(cid_str)
            for entry in entries:
                tok = int(entry["token_id"])
                if 0 <= tok < vocab_size:
                    token_to_cluster[tok] = cid

    # Build human-readable cluster descriptions from top-3 tokens in JSON
    cluster_names: Dict[str, str] = {}
    for cid_str, entries in cluster_json.items():
        top3 = [e["token"].strip().replace("\n", "\\n") or "<SP>"
                for e in entries[:3]]
        cluster_names[cid_str] = f"C{cid_str}[{','.join(top3)}]"

    return token_to_cluster, cluster_names


# ---------------------------------------------------------------------------
# Corpus tokenisation
# ---------------------------------------------------------------------------

def load_corpus_tokens(tokenizer, max_tokens: int) -> np.ndarray:
    """Load WikiText-2, tokenise, return flat int32 token array."""
    try:
        from datasets import load_dataset
        log.info("Loading wikitext-2-raw-v1 from HuggingFace datasets …")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                          split="train+validation+test",
                          trust_remote_code=False)
        texts = [t for t in ds["text"] if t and len(t.strip()) > 20]
        log.info("%d non-empty documents", len(texts))
    except Exception as exc:
        log.warning("Dataset load failed (%s); using built-in fallback.", exc)
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 2000,
            "Science is the systematic study of structure and behaviour. " * 2000,
            "In the beginning was the Word, and the Word was with God. " * 2000,
        ]

    budget = max_tokens * 2
    ids: List[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) >= budget:
            break

    arr = np.array(ids[:budget], dtype=np.int32)
    log.info("Corpus: %d tokens", len(arr))
    return arr


# ---------------------------------------------------------------------------
# Data collection — single forward pass, all target layers simultaneously
# ---------------------------------------------------------------------------

def collect_hidden_states(
    model,
    tokenizer,
    corpus: np.ndarray,
    target_layers: List[int],
    token_to_cluster: np.ndarray,
    device: torch.device,
    seq_len: int = 512,
    batch_size: int = 4,
    max_samples: int = 100_000,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    Single forward pass over the corpus with output_hidden_states=True.

    Returns
    -------
    hiddens   : {layer_idx: np.ndarray of shape (N, D), float16}
    clusters  : np.ndarray of shape (N,), int16 — cluster ID of each gold next-token
    tokens    : np.ndarray of shape (N,), int32 — raw gold next-token ID
    """
    vocab_size = len(token_to_cluster)
    eos_id = tokenizer.eos_token_id or 0

    # Build windows: each window is seq_len+1 tokens (last = gold for final position)
    windows = [
        corpus[i: i + seq_len + 1]
        for i in range(0, len(corpus) - seq_len - 1, seq_len)
    ]
    log.info("%d non-overlapping windows of len %d", len(windows), seq_len)

    # Per-layer buffers (list of arrays to concatenate at end)
    layer_bufs: Dict[int, List[np.ndarray]] = {l: [] for l in target_layers}
    cluster_buf: List[np.ndarray] = []
    token_buf:   List[np.ndarray] = []
    n_total = 0
    n_accepted = 0

    n_batches = (len(windows) + batch_size - 1) // batch_size

    # Hidden size is inferred from first batch
    hidden_size: Optional[int] = None

    from tqdm import tqdm

    with torch.no_grad():
        for b_idx in tqdm(range(n_batches), desc="Collecting hidden states"):
            if n_accepted >= max_samples:
                break

            batch_wins = windows[b_idx * batch_size: (b_idx + 1) * batch_size]
            if not batch_wins:
                continue

            B = len(batch_wins)
            S = seq_len

            # Pad to seq_len + 1
            padded = np.full((B, S + 1), eos_id, dtype=np.int32)
            for i, w in enumerate(batch_wins):
                L = min(len(w), S + 1)
                padded[i, :L] = w[:L]

            input_ids  = torch.tensor(padded[:, :-1], dtype=torch.long, device=device)  # (B, S)
            gold_next  = padded[:, 1:].flatten().astype(np.int32)  # (B*S,)

            # Determine which positions have a known cluster
            valid_tok  = (gold_next >= 0) & (gold_next < vocab_size)
            clust_ids  = np.where(valid_tok,
                                  token_to_cluster[np.clip(gold_next, 0, vocab_size - 1)],
                                  np.int16(-1))
            mask = (clust_ids >= 0)  # (B*S,) bool

            n_total += B * S
            n_this   = int(mask.sum())
            if n_this == 0:
                continue

            # Forward pass — ask for all hidden states
            out = model(input_ids=input_ids, output_hidden_states=True)
            # out.hidden_states: tuple of (n_layers+1) × (B, S, D)
            # Index 0 = embedding; index l+1 = block-l output

            if hidden_size is None:
                hidden_size = out.hidden_states[-1].shape[-1]
                log.info("Hidden size: %d", hidden_size)

            D = hidden_size

            for l in target_layers:
                h_idx = l + 1   # hidden_states[0] is embedding, [l+1] is block l output
                h_idx = min(h_idx, len(out.hidden_states) - 1)
                h_flat = out.hidden_states[h_idx].reshape(-1, D)  # (B*S, D)

                # Move to CPU as float16 immediately
                h_cpu = h_flat.cpu().to(torch.float16).numpy()
                layer_bufs[l].append(h_cpu[mask])

            # Gold labels for accepted positions
            cluster_buf.append(clust_ids[mask].astype(np.int16))
            token_buf.append(gold_next[mask])

            n_accepted += n_this
            del out, input_ids

            if device.type == "cuda":
                torch.cuda.empty_cache()

    log.info(
        "Collected %d / %d positions  (%.1f%% coverage of corpus, "
        "%.1f%% of positions had a cluster label)",
        n_accepted, n_total,
        100 * n_total / max(len(corpus), 1),
        100 * n_accepted / max(n_total, 1),
    )

    # Concatenate buffers
    hiddens: Dict[int, np.ndarray] = {}
    for l in target_layers:
        hiddens[l] = np.concatenate(layer_bufs[l], axis=0)   # (N, D) float16

    clusters = np.concatenate(cluster_buf, axis=0).astype(np.int16)
    tokens   = np.concatenate(token_buf,   axis=0).astype(np.int32)

    return hiddens, clusters, tokens


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def save_cache(
    cache_dir: str,
    hiddens: Dict[int, np.ndarray],
    clusters: np.ndarray,
    tokens: np.ndarray,
    target_layers: List[int],
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for l, h in hiddens.items():
        np.save(os.path.join(cache_dir, f"hidden_L{l:02d}.npy"), h)
    np.save(os.path.join(cache_dir, "clusters.npy"), clusters)
    np.save(os.path.join(cache_dir, "tokens.npy"), tokens)
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({"layers": target_layers, "n_samples": int(len(clusters))}, f)
    log.info("Cache saved to %s  (N=%d)", cache_dir, len(clusters))


def load_cache(
    cache_dir: str,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, List[int]]:
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    target_layers = meta["layers"]
    hiddens = {}
    for l in target_layers:
        p = os.path.join(cache_dir, f"hidden_L{l:02d}.npy")
        hiddens[l] = np.load(p)
    clusters = np.load(os.path.join(cache_dir, "clusters.npy"))
    tokens   = np.load(os.path.join(cache_dir, "tokens.npy"))
    log.info("Cache loaded from %s  (N=%d, layers=%s)", cache_dir, len(clusters), target_layers)
    return hiddens, clusters, tokens, target_layers


# ---------------------------------------------------------------------------
# Probe training — GPU-accelerated linear probe (replaces sklearn saga)
# ---------------------------------------------------------------------------

class _TorchLinearProbe:
    """
    Thin wrapper around nn.Linear with a sklearn-style predict_proba().
    Processes in chunks to avoid OOM on large test sets.
    """
    def __init__(self, model, device: torch.device):
        self.model  = model.eval()
        self.device = device

    def predict_proba(self, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
        import torch.nn.functional as F
        self.model.eval()
        parts = []
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32))
            for i in range(0, len(t), chunk):
                logits = self.model(t[i: i + chunk].to(self.device)).cpu()
                parts.append(F.softmax(logits, dim=-1).numpy())
        return np.concatenate(parts, axis=0)


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split → (X_train, X_val, X_test, y_train, y_val, y_test)."""
    from sklearn.model_selection import train_test_split as _split

    X_tr, X_tmp, y_tr, y_tmp = _split(
        X, y, test_size=1 - train_frac, random_state=seed, stratify=y
    )
    relative_val = val_frac / (1 - train_frac)
    X_v, X_te, y_v, y_te = _split(
        X_tmp, y_tmp, test_size=1 - relative_val, random_state=seed, stratify=y_tmp
    )
    return X_tr, X_v, X_te, y_tr, y_v, y_te


def fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    probe_device: torch.device,
    seed: int = 42,
    epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 2048,
    weight_decay: float = 1e-4,
) -> Tuple[object, "_TorchLinearProbe"]:
    """
    GPU-accelerated linear probe: StandardScaler + nn.Linear trained with Adam.

    ~5 s for 70k × 1600 → 9 classes on an L40S (vs. 562 s with sklearn saga).
    Returns (scaler, _TorchLinearProbe) — probe has a predict_proba() method.
    """
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train.astype(np.float32))

    in_dim = X_s.shape[1]
    linear = nn.Linear(in_dim, n_classes).to(probe_device)

    X_t = torch.from_numpy(X_s).to(probe_device)
    y_t = torch.from_numpy(y_train.astype(np.int64)).to(probe_device)

    opt     = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    N = len(X_s)

    linear.train()
    for _ in range(epochs):
        perm = torch.randperm(N, device=probe_device)
        for i in range(0, N, batch_size):
            idx = perm[i: i + batch_size]
            opt.zero_grad()
            loss_fn(linear(X_t[idx]), y_t[idx]).backward()
            opt.step()

    return scaler, _TorchLinearProbe(linear, probe_device)


def fit_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Tuple[object, object, object]:
    """
    Fit a 2-layer MLP probe. Returns (scaler, model, device).
    """
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train.astype(np.float32))
    X_v = scaler.transform(X_val.astype(np.float32))

    in_dim = X_s.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        def forward(self, x):
            return self.net(x)

    mlp = MLP().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_tr_t = torch.from_numpy(X_s).to(device)
    y_tr_t = torch.from_numpy(y_train.astype(np.int64)).to(device)
    X_vl_t = torch.from_numpy(X_v).to(device)
    y_vl_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    best_val = float("inf")
    best_state = None
    N = len(X_s)

    for ep in range(epochs):
        mlp.train()
        perm = torch.randperm(N)
        ep_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i: i + batch_size]
            opt.zero_grad()
            logits = mlp(X_tr_t[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= N
        mlp.eval()
        with torch.no_grad():
            val_loss = loss_fn(mlp(X_vl_t), y_vl_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in mlp.state_dict().items()}

    if best_state is not None:
        mlp.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return scaler, mlp, device


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def topk_recall(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction where true label is among top-k predicted classes."""
    # argsort ascending; last k = top-k
    top_k = np.argsort(probs, axis=1)[:, -k:]
    hits = np.array([labels[i] in top_k[i] for i in range(len(labels))], dtype=bool)
    return float(hits.mean())


def probe_metrics(
    scaler,
    probe,
    X: np.ndarray,
    y: np.ndarray,
    is_mlp: bool = False,
    mlp_device: Optional[torch.device] = None,
) -> dict:
    """Compute full metric dict for one probe on one dataset split."""
    from sklearn.metrics import (
        log_loss,
        precision_recall_fscore_support,
    )

    X_s = scaler.transform(X.astype(np.float32))

    if is_mlp:
        # MLP probe (optional --mlp_probe path) — raw nn.Module, needs manual softmax
        probe.eval()
        with torch.no_grad():
            t = torch.from_numpy(X_s).to(mlp_device)
            logits = probe(t).cpu().numpy()
            probs  = np.exp(logits - logits.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
    else:
        # Works for both _TorchLinearProbe and any sklearn probe
        probs = probe.predict_proba(X_s)

    preds = np.argmax(probs, axis=1)
    # All classes the probe knows about (one column per class in y_prob)
    all_classes = list(range(probs.shape[1]))
    # Classes actually present in this split (may be a subset)
    present_classes = sorted(np.unique(y).tolist())

    pr, rc, f1, sup = precision_recall_fscore_support(
        y, preds, labels=present_classes, average=None, zero_division=0
    )

    # log_loss needs labels= to match the number of columns in probs
    ce = float(log_loss(y, probs, labels=all_classes))

    metrics = {
        "top1":           float((preds == y).mean()),
        "top2":           topk_recall(probs, y, 2),
        "top4":           topk_recall(probs, y, 4),
        "cross_entropy":  ce,
        "n":              len(y),
        "per_cluster": {
            str(c): {
                "precision": float(pr[i]),
                "recall":    float(rc[i]),
                "f1":        float(f1[i]),
                "support":   int(sup[i]),
            }
            for i, c in enumerate(present_classes)
        },
    }
    return metrics


def calibration_data(
    scaler,
    probe,
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean_predicted_confidence, fraction_correct) for calibration plot.
    Uses top-1 confidence (max predicted probability) vs. correctness.
    """
    from sklearn.calibration import calibration_curve

    X_s = scaler.transform(X.astype(np.float32))
    probs = probe.predict_proba(X_s)
    conf  = probs.max(axis=1)
    correct = (np.argmax(probs, axis=1) == y).astype(float)

    frac_pos, mean_pred = calibration_curve(
        correct, conf, n_bins=n_bins, strategy="uniform"
    )
    return mean_pred, frac_pos


# ---------------------------------------------------------------------------
# Token baseline probe
# ---------------------------------------------------------------------------

def fit_token_probe(
    hiddens: np.ndarray,
    tokens: np.ndarray,
    probe_device: torch.device,
    top_k_tokens: int = 1000,
    seed: int = 42,
    epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 2048,
) -> Optional[dict]:
    """
    GPU linear probe: h_t -> exact token ID for the top-K most frequent tokens.

    Uses the same torch probe as fit_linear_probe so 1000-class problems
    finish in ~20 s instead of hours with sklearn saga.
    """
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    unique, counts = np.unique(tokens, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k_tokens]
    top_tok_set = set(unique[order].tolist())

    mask = np.isin(tokens, list(top_tok_set))
    if mask.sum() < 200:
        log.warning("Too few examples for token probe (%d). Skipping.", mask.sum())
        return None

    X     = hiddens[mask].astype(np.float32)
    y_raw = tokens[mask]

    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    n_classes = int(len(enc.classes_))

    X_tr, _, X_te, y_tr, _, y_te = train_test_split_stratified(X, y, seed=seed)

    log.info(
        "Token probe: %d classes  train=%d  test=%d",
        n_classes, len(X_tr), len(X_te),
    )

    torch.manual_seed(seed)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    linear  = nn.Linear(X_tr_s.shape[1], n_classes).to(probe_device)
    X_t     = torch.from_numpy(X_tr_s).to(probe_device)
    y_t     = torch.from_numpy(y_tr.astype(np.int64)).to(probe_device)
    opt     = torch.optim.Adam(linear.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    N = len(X_tr_s)

    linear.train()
    for _ in range(epochs):
        perm = torch.randperm(N, device=probe_device)
        for i in range(0, N, batch_size):
            idx = perm[i: i + batch_size]
            opt.zero_grad()
            loss_fn(linear(X_t[idx]), y_t[idx]).backward()
            opt.step()

    probe = _TorchLinearProbe(linear, probe_device)
    probs = probe.predict_proba(X_te_s)
    preds = np.argmax(probs, axis=1)

    return {
        "top1": float((preds == y_te).mean()),
        "top5": topk_recall(probs, y_te, 5),
        "n_classes": n_classes,
        "n_test": int(len(y_te)),
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _layer_xticks(target_layers: List[int]) -> Tuple[List[int], List[str]]:
    labels = [str(l) if l < max(target_layers) else f"{l}(final)"
              for l in target_layers]
    return target_layers, labels


def save_accuracy_plot(
    layer_metrics: Dict[int, dict],
    target_layers: List[int],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: str,
    thresholds: Optional[List[Tuple[float, str]]] = None,
    token_metrics: Optional[Dict[int, dict]] = None,
    token_key: Optional[str] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_vals = [layer_metrics[l][metric_key] for l in target_layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(target_layers, y_vals, marker="o", lw=2.0, color="steelblue",
            markersize=7, label=f"Cluster probe ({metric_key})")

    if token_metrics and token_key:
        tok_vals = [token_metrics.get(l, {}).get(token_key, float("nan"))
                    for l in target_layers]
        ax.plot(target_layers, tok_vals, marker="s", lw=1.5, color="darkorange",
                ls="--", markersize=6, label=f"Token probe ({token_key})")

    if thresholds:
        for thresh_val, thresh_label in thresholds:
            ax.axhline(thresh_val, color="crimson", ls="--", lw=1.1,
                       label=thresh_label, alpha=0.8)

    xticks, xlabels = _layer_xticks(target_layers)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out_path)


def save_calibration_plot(
    calib_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    target_layers: List[int],
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    cmap = cm.get_cmap("plasma", len(target_layers))
    for i, l in enumerate(target_layers):
        if l not in calib_data:
            continue
        conf, acc = calib_data[l]
        label = f"Layer {l}" if l < max(target_layers) else f"Layer {l} (final)"
        ax.plot(conf, acc, marker="o", ms=5, lw=1.5, color=cmap(i), label=label)

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title("Calibration by layer — cluster probe", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out_path)


def save_per_cluster_heatmap(
    layer_metrics: Dict[int, dict],
    target_layers: List[int],
    cluster_names: Dict[str, str],
    out_path: str,
) -> None:
    """Heatmap: rows = clusters, cols = layers, value = per-cluster recall."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Collect cluster IDs from first layer's metrics
    first_m = layer_metrics[target_layers[0]]
    cluster_ids = sorted(first_m["per_cluster"].keys(), key=lambda x: int(x))
    n_clusters  = len(cluster_ids)
    n_layers    = len(target_layers)

    # Matrix: rows=clusters, cols=layers
    mat = np.zeros((n_clusters, n_layers), dtype=np.float32)
    for j, l in enumerate(target_layers):
        m = layer_metrics[l]
        for i, cid in enumerate(cluster_ids):
            mat[i, j] = m["per_cluster"].get(cid, {}).get("recall", 0.0)

    row_labels = [cluster_names.get(cid, f"C{cid}") for cid in cluster_ids]
    col_labels = [str(l) if l < max(target_layers) else f"{l}(final)"
                  for l in target_layers]

    fig, ax = plt.subplots(figsize=(max(10, n_layers * 1.4), max(6, n_clusters * 0.8)))
    sns.heatmap(
        mat, ax=ax,
        xticklabels=col_labels, yticklabels=row_labels,
        cmap="YlOrRd", vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5,
    )
    ax.set_title("Per-cluster recall by layer — linear probe", fontweight="bold")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out_path)


def save_combined_comparison(
    layer_metrics: Dict[int, dict],
    token_metrics: Dict[int, dict],
    target_layers: List[int],
    out_path: str,
) -> None:
    """
    Single figure: cluster top-1, top-4, token top-1, token top-5 — all on one plot.
    This is the primary comparison figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))

    def _vals(src, key):
        return [src.get(l, {}).get(key, float("nan")) for l in target_layers]

    ax.plot(target_layers, _vals(layer_metrics, "top1"),
            "o-", lw=2, color="steelblue", label="Cluster top-1 acc")
    ax.plot(target_layers, _vals(layer_metrics, "top4"),
            "D-", lw=2.5, color="royalblue", label="Cluster top-4 recall  ← KEY")
    ax.plot(target_layers, _vals(token_metrics, "top1"),
            "s--", lw=1.8, color="darkorange", label="Token top-1 acc (top-1000 vocab)")
    ax.plot(target_layers, _vals(token_metrics, "top5"),
            "^--", lw=1.5, color="tomato", label="Token top-5 acc (top-1000 vocab)")

    ax.axhline(0.90, color="crimson", ls=":", lw=1.2, label="90% threshold")
    ax.axhline(0.70, color="gray",    ls=":",  lw=0.9, alpha=0.6)

    xticks, xlabels = _layer_xticks(target_layers)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_xlabel("Transformer layer", fontsize=12)
    ax.set_ylabel("Accuracy / Recall", fontsize=12)
    ax.set_title(
        "Cluster region recall vs. exact token accuracy across layers\n"
        "(GPT-2 XL — linear probe)",
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def find_threshold_layer(
    layer_metrics: Dict[int, dict],
    target_layers: List[int],
    metric: str,
    threshold: float,
) -> Optional[int]:
    """Return the first layer where metric >= threshold, or None."""
    for l in target_layers:
        if layer_metrics.get(l, {}).get(metric, 0.0) >= threshold:
            return l
    return None


def earliest_cluster_layers(
    layer_metrics: Dict[int, dict],
    target_layers: List[int],
    recall_threshold: float = 0.70,
) -> Dict[str, int]:
    """
    For each cluster, return the first layer where recall >= threshold.
    Returns {cluster_id_str: layer_idx} (layer=inf if never reached).
    """
    first_layer: Dict[str, int] = {}
    first_m = layer_metrics[target_layers[0]]
    cluster_ids = list(first_m["per_cluster"].keys())

    for cid in cluster_ids:
        for l in target_layers:
            if layer_metrics[l]["per_cluster"].get(cid, {}).get("recall", 0) >= recall_threshold:
                first_layer[cid] = l
                break
        else:
            first_layer[cid] = 999

    return first_layer


def write_summary(
    layer_metrics: Dict[int, dict],
    token_metrics: Dict[int, dict],
    target_layers: List[int],
    cluster_names: Dict[str, str],
    out_path: str,
    model_name: str,
    n_samples: int,
) -> None:
    lines: List[str] = []
    app = lines.append

    sep = "=" * 66
    final_l = target_layers[-1]

    top4_at_final = layer_metrics.get(final_l, {}).get("top4", 0.0)
    top1_at_final = layer_metrics.get(final_l, {}).get("top1", 0.0)
    tok_top1_final = token_metrics.get(final_l, {}).get("top1", float("nan"))

    app(sep)
    app("  LAYERWISE REGION PROBE — SUMMARY REPORT")
    app(f"  Model: {model_name}")
    app(f"  Samples: {n_samples:,}   Layers probed: {target_layers}")
    app(sep)
    app("")

    # Q1: at what layer do regions become predictable?
    layer_50 = find_threshold_layer(layer_metrics, target_layers, "top1", 0.50)
    layer_70 = find_threshold_layer(layer_metrics, target_layers, "top1", 0.70)
    app("1. AT WHAT LAYER DO REGIONS BECOME PREDICTABLE?")
    app(f"   top-1 cluster accuracy >= 50%: layer {layer_50}")
    app(f"   top-1 cluster accuracy >= 70%: layer {layer_70}")
    for l in target_layers:
        t1 = layer_metrics.get(l, {}).get("top1", 0)
        t4 = layer_metrics.get(l, {}).get("top4", 0)
        app(f"   Layer {l:3d}: top-1={t1:.3f}  top-4={t4:.3f}")
    app("")

    # Q2: does top-4 exceed 90% before final?
    layer_90_top4 = find_threshold_layer(layer_metrics, target_layers, "top4", 0.90)
    non_final_layers = target_layers[:-1]
    layer_90_top4_early = find_threshold_layer(
        {l: layer_metrics[l] for l in non_final_layers}, non_final_layers, "top4", 0.90
    )
    app("2. DOES TOP-4 CLUSTER RECALL EXCEED 90% BEFORE FINAL LAYERS?")
    if layer_90_top4_early is not None:
        app(f"   YES — first reaches 90% at layer {layer_90_top4_early}")
        if layer_90_top4_early <= 12:
            verdict = "EXCELLENT — strongly supports early routing"
        elif layer_90_top4_early <= 18:
            verdict = "GOOD — supports mid-network routing"
        else:
            verdict = "WEAK — routing only viable near output"
        app(f"   Verdict: {verdict}")
    elif layer_90_top4 == final_l:
        app(f"   NO — 90% top-4 recall only reached at final layer ({final_l})")
        app("   Verdict: Supports output-head routing only, not early routing")
    else:
        app(f"   NO — top-4 recall at final layer = {top4_at_final:.3f} (never reaches 90%)")
        app("   Verdict: Clusters may not be actionable from hidden states alone")
    app("")

    # Q3: how much earlier are clusters than exact tokens?
    app("3. HOW MUCH EARLIER ARE CLUSTERS VS EXACT TOKENS?")
    app(f"   Final-layer cluster top-1:  {top1_at_final:.3f}")
    app(f"   Final-layer token   top-1:  {tok_top1_final:.3f}  (top-1000 vocab)")
    for l in target_layers:
        c1 = layer_metrics.get(l, {}).get("top1", 0)
        tk = token_metrics.get(l, {}).get("top1", float("nan"))
        gap = c1 - tk if not np.isnan(tk) else float("nan")
        app(f"   Layer {l:3d}: cluster={c1:.3f}  token={tk:.3f}  gap={gap:+.3f}")
    app("")

    # Q4 & Q5: which clusters emerge earliest / latest?
    early_map = earliest_cluster_layers(layer_metrics, target_layers, recall_threshold=0.70)
    sorted_by_layer = sorted(early_map.items(), key=lambda kv: kv[1])

    app("4. CLUSTERS THAT EMERGE EARLIEST (recall >= 70%):")
    for cid, l in sorted_by_layer[:4]:
        name = cluster_names.get(cid, f"C{cid}")
        lstr = str(l) if l < 999 else "NEVER"
        app(f"   {name:<40s}  first @ layer {lstr}")
    app("")

    app("5. CLUSTERS THAT EMERGE LATEST:")
    for cid, l in sorted_by_layer[-4:]:
        name = cluster_names.get(cid, f"C{cid}")
        lstr = str(l) if l < 999 else "NEVER"
        app(f"   {name:<40s}  first @ layer {lstr}")
    app("")

    # Overall conclusion
    app(sep)
    app("OVERALL CONCLUSION")
    if layer_90_top4_early is not None and layer_90_top4_early <= 12:
        app("  STRONG support for early region-routing architecture.")
        app(f"  Middle layers ({layer_90_top4_early}) already achieve top-4 cluster recall > 90%.")
        app("  Implication: a region router at layer ~{} could route tokens to".format(layer_90_top4_early))
        app("  specialised output heads, potentially saving compute in later layers.")
    elif layer_90_top4_early is not None:
        app("  MODERATE support for mid-network routing.")
        app(f"  Top-4 cluster recall > 90% reached at layer {layer_90_top4_early}.")
    elif layer_90_top4 is not None and layer_90_top4 == final_l:
        app("  Regions are predictable from final hidden states but NOT from early layers.")
        app("  Output-head routing is viable; early routing may not be.")
    else:
        app("  Cluster probing did not reach 90% top-4 recall at any probed layer.")
        app("  Consider: denser layer sampling, more samples, or revisiting cluster quality.")
    app(sep)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    log.info("Summary saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layerwise linear probe: do token regions emerge earlier than "
                    "exact token identity in GPT-2 XL?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input files
    parser.add_argument("--clusters", required=True,
                        help="cluster_tokens JSON (from interference experiment)")
    parser.add_argument("--labels", default=None,
                        help="Louvain labels .npy (auto-detected if omitted)")
    parser.add_argument("--freq_ids", default=None,
                        help="frequent_token_ids .npy (auto-detected if omitted)")
    # Model
    parser.add_argument("--model", default="gpt2-xl",
                        help="HuggingFace causal LM to probe")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable float16 inference (use float32)")
    # Data
    parser.add_argument("--max_samples", type=int, default=100_000,
                        help="Max next-token positions to collect")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Sequences per forward pass (reduce if OOM)")
    # Layers
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices to probe, e.g. 0,4,8,12,24,-1. "
                             "Default: auto-computed based on model depth.")
    # Probe
    parser.add_argument("--mlp_probe", action="store_true",
                        help="Also train a small MLP probe (slower)")
    parser.add_argument("--token_vocab_k", type=int, default=1000,
                        help="Top-K tokens used for exact-token baseline probe")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="Subsample training set to this many examples per layer. "
                             "0 = use all data. 20000 is usually sufficient for 9 clusters.")
    # I/O
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory to cache collected hidden states "
                             "(skips re-extraction on re-runs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    t0_global = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16   = not args.no_fp16
    log.info("Device: %s  fp16: %s", device, fp16)

    # ------------------------------------------------------------------ #
    # Step 1 — Load model                                                  #
    # ------------------------------------------------------------------ #
    log.info("=== Loading model ===")
    model, tokenizer, model_name = load_model_and_tokenizer(args.model, device, fp16)
    n_layers = get_n_layers(model)
    log.info("Model has %d transformer layers  hidden_dim=%d",
             n_layers, model.config.hidden_size)

    target_layers = compute_target_layers(n_layers, args.layers)
    log.info("Target layers: %s", target_layers)

    vocab_size = len(tokenizer)

    # ------------------------------------------------------------------ #
    # Step 2 — Build token → cluster mapping                              #
    # ------------------------------------------------------------------ #
    log.info("=== Building cluster mapping ===")
    token_to_cluster, cluster_names = load_cluster_mapping(
        clusters_path=args.clusters,
        labels_path=args.labels,
        freq_ids_path=args.freq_ids,
        vocab_size=vocab_size,
    )
    n_clusters = int(token_to_cluster.max()) + 1
    log.info("n_clusters=%d", n_clusters)

    # ------------------------------------------------------------------ #
    # Step 3 — Collect hidden states (or load from cache)                 #
    # ------------------------------------------------------------------ #
    cache_ready = (
        args.cache_dir is not None
        and os.path.exists(os.path.join(args.cache_dir, "meta.json"))
    )

    if cache_ready:
        log.info("=== Loading cached hidden states from %s ===", args.cache_dir)
        hiddens, clusters, tokens, target_layers = load_cache(args.cache_dir)
    else:
        log.info("=== Collecting hidden states from model ===")
        corpus = load_corpus_tokens(tokenizer, max_tokens=args.max_samples * 2)

        hiddens, clusters, tokens = collect_hidden_states(
            model=model,
            tokenizer=tokenizer,
            corpus=corpus,
            target_layers=target_layers,
            token_to_cluster=token_to_cluster,
            device=device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )

        if args.cache_dir:
            save_cache(args.cache_dir, hiddens, clusters, tokens, target_layers)

        # Free model memory now that extraction is done
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_samples = len(clusters)
    log.info("Dataset: %d samples  %d clusters", n_samples, n_clusters)

    # Probe device: prefer GPU (fast matrix ops); falls back to CPU
    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # Step 4 — Train probes for each layer                                 #
    # ------------------------------------------------------------------ #
    log.info("=== Training probes (torch linear, device=%s) ===", probe_device)

    layer_metrics:     Dict[int, dict] = {}
    token_metrics:     Dict[int, dict] = {}
    calib_data:        Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    mlp_layer_metrics: Dict[int, dict] = {}

    for l in target_layers:
        log.info("--- Layer %d ---", l)
        X = hiddens[l].astype(np.float32)   # (N, D)
        y = clusters.astype(np.int32)        # (N,)

        # Optional training-set subsample (does not affect test set)
        if args.max_train_samples > 0 and len(X) > args.max_train_samples:
            rng = np.random.default_rng(args.seed)
            sub_idx = rng.choice(len(X), size=args.max_train_samples, replace=False)
            X_sub, y_sub = X[sub_idx], y[sub_idx]
            log.info("  Subsampled train to %d (from %d)", args.max_train_samples, len(X))
        else:
            X_sub, y_sub = X, y

        X_tr, X_v, X_te, y_tr, y_v, y_te = train_test_split_stratified(
            X_sub, y_sub, seed=args.seed
        )

        t_probe = time.time()
        scaler, probe = fit_linear_probe(
            X_tr, y_tr,
            n_classes=n_clusters,
            probe_device=probe_device,
            seed=args.seed,
        )
        log.info("  Linear probe trained in %.1fs", time.time() - t_probe)

        # Always evaluate on the FULL test split (not subsampled)
        X_te_full = X[int(len(X) * 0.85):]
        y_te_full = y[int(len(y) * 0.85):]
        layer_metrics[l] = probe_metrics(scaler, probe, X_te_full, y_te_full)
        log.info(
            "  top1=%.3f  top2=%.3f  top4=%.3f  CE=%.3f",
            layer_metrics[l]["top1"], layer_metrics[l]["top2"],
            layer_metrics[l]["top4"], layer_metrics[l]["cross_entropy"],
        )

        # Calibration
        try:
            conf, acc = calibration_data(scaler, probe, X_te_full, y_te_full)
            calib_data[l] = (conf, acc)
        except Exception as exc:
            log.warning("  Calibration failed for layer %d: %s", l, exc)

        # MLP probe (optional)
        if args.mlp_probe:
            log.info("  Fitting MLP probe …")
            t_mlp = time.time()
            mlp_scaler, mlp, mlp_dev = fit_mlp_probe(
                X_tr, y_tr, X_v, y_v,
                n_classes=n_clusters,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                seed=args.seed,
            )
            log.info("  MLP trained in %.1fs", time.time() - t_mlp)
            mlp_layer_metrics[l] = probe_metrics(
                mlp_scaler, mlp, X_te, y_te, is_mlp=True, mlp_device=mlp_dev
            )
            log.info(
                "  MLP: top1=%.3f  top4=%.3f",
                mlp_layer_metrics[l]["top1"], mlp_layer_metrics[l]["top4"],
            )

        # Token baseline probe
        log.info("  Fitting token baseline probe …")
        tok_result = fit_token_probe(
            X, tokens,
            probe_device=probe_device,
            top_k_tokens=args.token_vocab_k,
            seed=args.seed,
        )
        if tok_result:
            token_metrics[l] = tok_result
            log.info(
                "  Token: top1=%.3f  top5=%.3f  (k=%d classes)",
                tok_result["top1"], tok_result["top5"], tok_result["n_classes"],
            )

    # ------------------------------------------------------------------ #
    # Step 5 — Save metrics JSON                                           #
    # ------------------------------------------------------------------ #
    all_metrics = {
        "model": model_name,
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "target_layers": target_layers,
        "linear_probe": {str(l): layer_metrics[l] for l in target_layers},
        "token_probe":  {str(l): token_metrics.get(l, {}) for l in target_layers},
    }
    if mlp_layer_metrics:
        all_metrics["mlp_probe"] = {str(l): mlp_layer_metrics[l] for l in target_layers}

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    # ------------------------------------------------------------------ #
    # Step 6 — Visualisations                                              #
    # ------------------------------------------------------------------ #
    log.info("=== Generating visualisations ===")

    save_accuracy_plot(
        layer_metrics, target_layers,
        metric_key="top1",
        title="Top-1 cluster accuracy vs. transformer layer (linear probe)",
        ylabel="Top-1 accuracy",
        out_path=os.path.join(args.output_dir, "layer_vs_top1_cluster.png"),
        thresholds=[(0.7, "70% threshold"), (0.9, "90% threshold")],
    )

    save_accuracy_plot(
        layer_metrics, target_layers,
        metric_key="top4",
        title="Top-4 cluster recall vs. transformer layer (linear probe)",
        ylabel="Top-4 recall  ← KEY METRIC",
        out_path=os.path.join(args.output_dir, "layer_vs_top4_cluster.png"),
        thresholds=[(0.9, "90% threshold  ← early-routing threshold")],
    )

    save_accuracy_plot(
        layer_metrics, target_layers,
        metric_key="top1",
        title="Cluster top-1 vs. exact-token top-1 accuracy by layer",
        ylabel="Accuracy",
        out_path=os.path.join(args.output_dir, "layer_vs_exact_token_acc.png"),
        token_metrics=token_metrics,
        token_key="top1",
        thresholds=[(0.9, "90%")],
    )

    if calib_data:
        save_calibration_plot(
            calib_data, target_layers,
            out_path=os.path.join(args.output_dir, "calibration_by_layer.png"),
        )

    try:
        save_per_cluster_heatmap(
            layer_metrics, target_layers, cluster_names,
            out_path=os.path.join(args.output_dir, "per_cluster_recall_heatmap.png"),
        )
    except Exception as exc:
        log.warning("Heatmap failed: %s", exc)

    # Primary comparison figure (bonus, most informative)
    if token_metrics:
        save_combined_comparison(
            layer_metrics, token_metrics, target_layers,
            out_path=os.path.join(args.output_dir, "layer_cluster_vs_token_comparison.png"),
        )

    # ------------------------------------------------------------------ #
    # Step 7 — Written summary                                             #
    # ------------------------------------------------------------------ #
    log.info("=== Writing summary ===")
    write_summary(
        layer_metrics=layer_metrics,
        token_metrics=token_metrics,
        target_layers=target_layers,
        cluster_names=cluster_names,
        out_path=os.path.join(args.output_dir, "summary.txt"),
        model_name=model_name,
        n_samples=n_samples,
    )

    elapsed = time.time() - t0_global
    log.info("Total time: %.1f s (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
