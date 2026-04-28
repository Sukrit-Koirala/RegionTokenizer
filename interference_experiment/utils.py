"""
Shared configuration, logging, and mathematical utilities.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # ---- Model ----
    model_name: str = "gpt2-xl"
    fallback_model_name: str = "gpt2-medium"
    device: str = "cuda"
    fp16: bool = True

    # ---- Data ----
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_tokens: int = 50_000   # flat token budget from corpus
    seq_len: int = 512         # context window per forward pass
    batch_size: int = 4        # sequences per GPU batch

    # ---- Extraction ----
    layer_index: int = -1      # -1 → final layer; 0 → first transformer block output
    top_k: int = 20            # predictions kept per position for co-activation graph

    # ---- Graph ----
    # We restrict dense graph operations to the N most-predicted tokens to keep
    # the N×N matrix tractable (5 000 → 100 MB in float32).
    vocab_subset_size: int = 5_000

    # ---- Clustering ----
    n_clusters: int = 50        # for spectral clustering and k-means
    louvain_resolution: float = 1.0

    # ---- Layer comparison (optional) ----
    # Layer indices to compare; -1 always means "final".
    # For GPT-2 XL (48 layers): 0=first, 23=middle, -1=final.
    layer_compare_indices: List[int] = field(default_factory=lambda: [0, 23, -1])

    # ---- Paths ----
    data_dir: str = "data"
    graphs_dir: str = "graphs"
    results_dir: str = "results"

    # ---- Reproducibility ----
    seed: int = 42

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("interference")


def setup_dirs(cfg: ExperimentConfig) -> None:
    for d in [cfg.data_dir, cfg.graphs_dir, cfg.results_dir]:
        os.makedirs(d, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(cfg: ExperimentConfig):
    """Return torch.device, falling back to CPU with a warning."""
    import torch

    if cfg.device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log = logging.getLogger("interference")
        log.info(
            "GPU: %s  VRAM: %.1f GB",
            props.name,
            props.total_memory / 1e9,
        )
        return torch.device("cuda")

    log = logging.getLogger("interference")
    log.warning("CUDA not available — running on CPU (will be slow).")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Linear-algebra utilities
# ---------------------------------------------------------------------------

def cosine_sim_matrix(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute the N×N pairwise cosine-similarity matrix for rows of A (N×D).
    Returns a float32 matrix in [-1, 1].
    """
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    A_hat = (A / norms).astype(np.float32)
    return A_hat @ A_hat.T


def symmetric_normalize(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Graph-Laplacian symmetric normalization: D^{-1/2} W D^{-1/2}.
    Handles zero-degree nodes gracefully.
    """
    d = W.sum(axis=1)
    d_inv_sqrt = np.where(d > eps, 1.0 / np.sqrt(d), 0.0)
    return (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]).astype(np.float32)


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Effective dimension from eigenvalue distribution.
    PR = (Σ λ_i)² / Σ λ_i²  — high PR → spread spectrum (many dims),
    low PR → concentrated (few dims dominate).
    """
    lam = np.abs(eigenvalues)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def cluster_entropy(labels: np.ndarray) -> float:
    """Shannon entropy (nats) of the cluster-size distribution."""
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def intra_inter_weights(
    W: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    """
    Return (mean_intra_weight, mean_inter_weight) for a symmetric weight
    matrix W and integer label vector labels.  Diagonal is excluded.
    """
    same = (labels[:, None] == labels[None, :])
    np.fill_diagonal(same, False)
    diff = ~same
    np.fill_diagonal(diff, False)

    intra = float(W[same].mean()) if same.any() else 0.0
    inter = float(W[diff].mean()) if diff.any() else 0.0
    return intra, inter


def compute_modularity_numpy(W: np.ndarray, labels: np.ndarray) -> float:
    """
    Fast numpy implementation of modularity Q for a weighted undirected graph.
    Q = (1/2m) Σ_{ij} [A_{ij} - k_i k_j / (2m)] δ(c_i, c_j)
    """
    m2 = W.sum()                   # 2m (sum of all edge weights, both directions)
    if m2 == 0:
        return 0.0
    k = W.sum(axis=1)              # degree vector
    Q = 0.0
    n_clusters = labels.max() + 1
    for c in range(n_clusters):
        mask = labels == c
        A_c = W[np.ix_(mask, mask)]
        k_c = k[mask]
        Q += A_c.sum() - (k_c.sum() ** 2) / m2
    return float(Q / m2)


def random_graph_modularity(n_nodes: int, n_clusters: int) -> float:
    """
    Expected modularity of a random partition into equal-sized clusters
    (analytic approximation: Q ≈ 0 for random graphs with uniform partition).
    Returns the Erdos-Renyi baseline ~1/n_clusters.
    """
    return 1.0 / n_clusters
