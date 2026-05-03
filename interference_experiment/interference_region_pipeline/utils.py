"""
utils.py — Shared utilities for the interference region pipeline.
"""

import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_model_tokenizer(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log = logging.getLogger("utils")
    log.info("Loading tokenizer: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log.info("Loading model: %s (fp16, device_map=auto)", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return tok, model


def get_corpus_tokens(
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    max_tokens: int,
    local_text_file: Optional[str] = None,
) -> np.ndarray:
    log = logging.getLogger("utils")
    if local_text_file:
        log.info("Reading local text: %s", local_text_file)
        with open(local_text_file, encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text)[:max_tokens]
        log.info("Tokens from file: %d", len(ids))
        return np.array(ids, dtype=np.int32)

    from datasets import load_dataset
    log.info("Loading %s / %s", dataset_name, dataset_config)
    ds = load_dataset(dataset_name, dataset_config, split="train")
    all_ids: List[int] = []
    for row in ds:
        text = row.get("text", "")
        if text.strip():
            all_ids.extend(tokenizer.encode(text))
            if len(all_ids) >= max_tokens:
                break
    ids = np.array(all_ids[:max_tokens], dtype=np.int32)
    log.info("Corpus tokens: %d", len(ids))
    return ids


def top_k_frequent_ids(token_ids: np.ndarray, k: int) -> np.ndarray:
    """Return the k most frequent token IDs, sorted by descending frequency."""
    counts = np.bincount(token_ids.astype(np.int64))
    return np.argsort(counts)[::-1][:k].astype(np.int32)


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


# ── Linear probe (GPU) ────────────────────────────────────────────────────────

class _Probe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def fit_probe(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_classes: int,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-2,
    batch: int = 4096,
) -> Tuple[dict, "_Probe"]:
    """Train and evaluate a linear probe. Returns (metrics_dict, trained_probe)."""
    probe = _Probe(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).long().to(device)

    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(Xt), batch):
            idx = perm[s : s + batch]
            loss = loss_fn(probe(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X_te).to(device)
        yte = torch.from_numpy(y_te).long().to(device)
        logits = probe(Xte)
        top1 = float((logits.argmax(1) == yte).float().mean())
        k2 = min(2, n_classes); k4 = min(4, n_classes)
        top2 = float((logits.topk(k2, 1).indices == yte.unsqueeze(1)).any(1).float().mean())
        top4 = float((logits.topk(k4, 1).indices == yte.unsqueeze(1)).any(1).float().mean())
        ce   = float(loss_fn(logits, yte).item() / np.log(2))   # bits

    metrics = {
        "top1": top1, "top2": top2, "top4": top4,
        "ce_bits": ce, "n_classes": n_classes,
        "random_top1": 1.0 / n_classes,
    }
    return metrics, probe


def probe_predict_topk(probe: "_Probe", X: np.ndarray, k: int, device: torch.device) -> np.ndarray:
    """Returns (N, k) array of top-k predicted class indices."""
    probe.eval()
    with torch.no_grad():
        logits = probe(torch.from_numpy(X).to(device))
        return logits.topk(min(k, logits.shape[1]), dim=1).indices.cpu().numpy()
