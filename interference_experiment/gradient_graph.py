"""
Gradient-based interference graph.

The gradient of log p(i | h) with respect to the hidden state h is:

    ∂ log p(i | h) / ∂h  =  w_i  −  E_{j ~ p(·|h)}[w_j]
                          =  w_i  −  Σ_j p_j(h) · w_j

where w_i is the i-th row of the LM-head (unembedding) weight matrix W.

Averaging this gradient over all context positions t:

    v_i  =  w_i  −  (1/T) Σ_t  Σ_j p_j(h_t) · w_j
         =  w_i  −  mean_expected_embedding

This is computed analytically — no autograd required.
`mean_expected_embedding` was accumulated during extraction.

Two tokens i, j interfere in the same hidden-state direction iff their
gradient vectors are similar:

    I[i, j]  =  cosine_similarity(v_i, v_j)

The resulting matrix I ∈ [−1, 1] is our gradient interference graph.

We also optionally build a representational-similarity graph from per-token
mean hidden states (used for layer-comparison analysis).
"""

import argparse
import logging
import os
from typing import Optional

import numpy as np

from utils import (
    ExperimentConfig,
    cosine_sim_matrix,
    save_json,
    setup_dirs,
    setup_logging,
)
from extract_hidden_states import load_extracted


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_gradient_vectors(
    W_unembed: np.ndarray,          # (V, D) float32 — full vocabulary
    frequent_token_ids: np.ndarray, # (N_sub,) int32
    mean_expected_embedding: np.ndarray,  # (D,) float32
) -> np.ndarray:
    """
    Return v_i = w_i − mean_expected_embedding for each token i in
    frequent_token_ids.  Shape: (N_sub, D) float32.
    """
    W_sub = W_unembed[frequent_token_ids.astype(np.int32)]  # (N_sub, D)
    V = W_sub - mean_expected_embedding[None, :]             # broadcast subtract
    return V.astype(np.float32)


def compute_gradient_graph(
    W_unembed: np.ndarray,
    frequent_token_ids: np.ndarray,
    mean_expected_embedding: np.ndarray,
) -> np.ndarray:
    """
    Compute I[i, j] = cosine_similarity(v_i, v_j) for the vocabulary subset.
    Returns (N_sub, N_sub) float32 matrix in [−1, 1].
    """
    V = compute_gradient_vectors(W_unembed, frequent_token_ids, mean_expected_embedding)
    I = cosine_sim_matrix(V)   # (N_sub, N_sub)
    return I


def compute_representational_similarity(
    mean_hidden_per_token: np.ndarray,  # (V, D) float32
    frequent_token_ids: np.ndarray,     # (N_sub,) int32
    seen_counts: Optional[np.ndarray] = None,  # (V,) int64
) -> np.ndarray:
    """
    Build a similarity graph from per-token mean hidden states.
    Used for layer-comparison: tokens whose hidden representations are similar
    tend to be processed alike at that layer.

    Tokens with zero seen_count are masked out (row/col set to 0).
    Returns (N_sub, N_sub) float32.
    """
    ids = frequent_token_ids.astype(np.int32)
    H = mean_hidden_per_token[ids]  # (N_sub, D)

    if seen_counts is not None:
        counts = seen_counts[ids]   # (N_sub,)
        valid = counts > 0
        H[~valid] = 0.0

    S = cosine_sim_matrix(H)

    if seen_counts is not None:
        # Zero out rows/cols for unseen tokens
        mask = (counts > 0).astype(np.float32)
        S = S * mask[:, None] * mask[None, :]

    return S


# ---------------------------------------------------------------------------
# Unembedding matrix loader (model-free, from saved checkpoint)
# ---------------------------------------------------------------------------

def load_unembedding_from_model(model_name: str, fp16: bool = True) -> np.ndarray:
    """
    Load only the unembedding weight from a HuggingFace model.
    We do NOT load the full model — we just grab the head weights.
    Returns (V, D) float32 numpy array.
    """
    import torch
    from transformers import AutoModelForCausalLM

    log = logging.getLogger("interference")
    log.info("Loading unembedding matrix from %s …", model_name)

    dtype = torch.float16 if fp16 else torch.float32
    # Load to CPU to avoid GPU overhead for weight-only access
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W = model.lm_head.weight.detach().float().cpu().numpy()
    elif hasattr(model, "embed_out") and hasattr(model.embed_out, "weight"):
        W = model.embed_out.weight.detach().float().cpu().numpy()
    else:
        raise AttributeError("Cannot locate lm_head.weight in model.")

    if W.shape[0] < W.shape[1]:
        W = W.T   # ensure (V, D)

    del model   # free memory immediately
    log.info("Unembedding matrix: shape=%s", W.shape)
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def build_and_save(
    prefix: str,
    cfg: ExperimentConfig,
    model_name: Optional[str] = None,
    need_hidden: bool = False,
) -> np.ndarray:
    """
    Load extraction artefacts, compute the gradient interference graph I,
    save it, and return the matrix.

    If need_hidden is True, also compute the representational-similarity graph
    from mean hidden states (requires collect_hidden=True during extraction).
    """
    log = logging.getLogger("interference")
    data = load_extracted(prefix, need_hidden=need_hidden)

    frequent_ids = data["frequent_token_ids"]        # (N_sub,)
    mean_emb = data["mean_expected_embedding"]       # (D,)
    layer_tag = data.get("layer_tag", "final")
    model_name_used = model_name or data.get("model_name", "gpt2-xl")

    # Load unembedding weights
    W_unembed = load_unembedding_from_model(model_name_used, fp16=cfg.fp16)

    # ---- Gradient interference graph ----
    log.info("Computing gradient interference graph …")
    I = compute_gradient_graph(W_unembed, frequent_ids, mean_emb)

    out_prefix = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}")
    np.save(f"{out_prefix}_I_gradient.npy", I)
    log.info("Saved gradient graph → %s_I_gradient.npy  shape=%s", out_prefix, I.shape)

    # ---- Diagnostics ----
    diag = {
        "layer_tag": layer_tag,
        "vocab_subset_size": int(len(frequent_ids)),
        "I_mean": float(I.mean()),
        "I_std": float(I.std()),
        "I_min": float(I.min()),
        "I_max": float(I.max()),
        # Off-diagonal mean (exclude self-similarity = 1)
        "I_offdiag_mean": float(
            (I.sum() - np.trace(I)) / max((I.size - len(I)), 1)
        ),
    }
    save_json(f"{out_prefix}_I_meta.json", diag)

    for k, v in diag.items():
        if isinstance(v, float):
            log.info("  %-26s  %.4f", k, v)

    # ---- Representational similarity graph (optional) ----
    if need_hidden and "mean_hidden_per_token" in data:
        log.info("Computing representational similarity graph from hidden states …")
        seen = data.get("token_seen_counts")
        S = compute_representational_similarity(
            data["mean_hidden_per_token"], frequent_ids, seen_counts=seen
        )
        np.save(f"{out_prefix}_S_hidden.npy", S)
        log.info("Saved hidden-state similarity graph → %s_S_hidden.npy", out_prefix)

    return I


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build gradient interference graph.")
    parser.add_argument("--layer_tag", default="final")
    parser.add_argument("--model", default=None,
                        help="Override model name for unembedding weight loading.")
    parser.add_argument("--with_hidden", action="store_true",
                        help="Also build representational similarity graph from hidden states.")
    args = parser.parse_args()

    setup_logging()
    cfg = ExperimentConfig()
    setup_dirs(cfg)

    prefix = os.path.join(cfg.graphs_dir, f"layer_{args.layer_tag}")
    build_and_save(prefix, cfg, model_name=args.model, need_hidden=args.with_hidden)


if __name__ == "__main__":
    main()
