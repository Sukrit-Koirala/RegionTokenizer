#!/usr/bin/env python3
"""
build_interference_graph.py

Run a pretrained LM over a corpus and build a sparse coactivation graph:
    W[i,j] += 1  whenever tokens i and j both appear in top-K logits
                 at the same next-token prediction position.

Restricted to vocab_subset_size most-frequent corpus tokens.

Also caches hidden states at probe_layers for N_probe positions so
evaluate_regions.py can train router probes without rerunning the model.

Outputs:
  graphs_dir/
    frequent_token_ids.npy     (V,) int32  — ordered subset token IDs
    vocab_to_sub.npy           (vocab_size,) int32  — vocab_id → subset_idx (-1 if absent)
    coactivation_W.npz         sparse (V,V) float32 — raw co-occurrence counts
    coactivation_W_norm.npz    sparse (V,V) float32 — Jaccard-normalized, diagonal zeroed

  output_dir/probe_cache/
    hidden_L{l:02d}.npy       (N_probe, D) float16
    probe_token_ids.npy       (N_probe,) int32  — gold next-token vocab ID
    topk_ids.npy              (N_probe, top_k) int32  — top-K predicted token IDs
    meta.json
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_logging, set_seed, ensure_dirs,
    load_model_tokenizer, get_corpus_tokens,
    top_k_frequent_ids, save_json,
)

log = setup_logging("build_graph")

PROBE_LAYERS_DEFAULT = [0, 4, 8, 12, 16, 20, 24, 47]


# ── Vocab subset helpers ──────────────────────────────────────────────────────

def build_vocab_map(freq_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    vocab_to_sub = np.full(vocab_size, -1, dtype=np.int32)
    for sub_idx, vid in enumerate(freq_ids):
        vocab_to_sub[int(vid)] = sub_idx
    return vocab_to_sub


# ── Graph normalization ───────────────────────────────────────────────────────

def normalize_coactivation(W_offdiag: np.ndarray, degree: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    W_norm[i,j] = W[i,j] / sqrt(degree[i] * degree[j] + eps)
    Jaccard/cosine-style: degree[i] = # positions where token i was in top-K.
    """
    d = np.sqrt(np.outer(degree, degree) + eps)
    W_norm = W_offdiag / d
    np.fill_diagonal(W_norm, 0.0)
    return W_norm.astype(np.float32)


# ── Optional embedding similarity ─────────────────────────────────────────────

def embedding_similarity_graph(model, freq_ids: np.ndarray, top_n: int = 20) -> np.ndarray:
    """
    Cosine similarity between LM output embedding vectors for subset tokens,
    sparsified to top_n neighbours per token.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        emb = model.transformer.wte.weight[
            torch.from_numpy(freq_ids.astype(np.int64)).to(device)
        ].float()
        emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        S = (emb @ emb.T).cpu().numpy()

    V = len(freq_ids)
    np.fill_diagonal(S, 0.0)
    for i in range(V):
        row = S[i]
        if V > top_n:
            thresh = np.partition(row, V - top_n - 1)[V - top_n - 1]
            row[row < thresh] = 0.0
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, 0.0)
    return S.astype(np.float32)


def gradient_similarity_stub(model, freq_ids: np.ndarray) -> np.ndarray:
    """Not implemented — returns zeros. Set beta=0 (default) to skip."""
    return np.zeros((len(freq_ids), len(freq_ids)), dtype=np.float32)


# ── Main inference loop ───────────────────────────────────────────────────────

def run_inference(
    model,
    token_ids: np.ndarray,
    freq_ids: np.ndarray,
    vocab_to_sub: np.ndarray,
    top_k: int,
    probe_layers: List[int],
    n_probe: int,
    seq_len: int,
    device: torch.device,
) -> tuple:
    """
    Single-pass over the corpus.  Returns:
      W_offdiag : (V, V) float32 — raw co-occurrence counts (off-diagonal)
      degree    : (V,) float32  — per-token top-K appearance count
      probe_hidden     : {layer -> (N_probe, D) float16}
      probe_token_ids  : (N_probe,) int32
      probe_topk_ids   : (N_probe, top_k) int32
    """
    V = len(freq_ids)
    D = model.config.hidden_size
    W_offdiag = np.zeros((V, V), dtype=np.float32)
    degree    = np.zeros(V, dtype=np.float32)

    n_probe = min(n_probe, len(token_ids) - 1)
    probe_hidden    = {l: np.zeros((n_probe, D), dtype=np.float16) for l in probe_layers}
    probe_token_ids = np.zeros(n_probe, dtype=np.int32)
    probe_topk_ids  = np.zeros((n_probe, top_k), dtype=np.int32)
    probe_filled    = 0

    n_seq_total  = (len(token_ids) - 1) // seq_len
    n_probe_seqs = (n_probe + seq_len - 1) // seq_len
    log.info("Corpus: %d tokens → %d sequences (seq_len=%d)", len(token_ids), n_seq_total, seq_len)
    log.info("Probe: first %d seqs → up to %d positions", n_probe_seqs, n_probe)

    t0 = time.time()
    for seq_idx in range(n_seq_total):
        start = seq_idx * seq_len
        end   = start + seq_len
        if end >= len(token_ids):
            break

        inp_np  = token_ids[start:end].astype(np.int64)
        gold_np = token_ids[start + 1 : end + 1].astype(np.int32)

        do_cache = probe_filled < n_probe
        n_cache  = min(seq_len, n_probe - probe_filled) if do_cache else 0

        inp_t = torch.from_numpy(inp_np).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp_t, output_hidden_states=do_cache, use_cache=False)

        logits = out.logits[0]                              # (seq_len, vocab) fp16
        topk_t = logits.topk(top_k, dim=-1).indices        # (seq_len, top_k)
        topk_np = topk_t.cpu().numpy().astype(np.int64)    # int64 for indexing

        # Map to subset indices: (seq_len, top_k) int32
        topk_sub = vocab_to_sub[topk_np]

        # Accumulate pairs and degree
        rows_acc, cols_acc = [], []
        for pos in range(seq_len):
            sub_pos = topk_sub[pos]
            valid   = sub_pos[sub_pos >= 0]
            if len(valid) == 0:
                continue
            np.add.at(degree, valid, 1)
            if len(valid) >= 2:
                r, c = np.triu_indices(len(valid), k=1)
                rows_acc.append(valid[r])
                cols_acc.append(valid[c])

        if rows_acc:
            rows = np.concatenate(rows_acc).astype(np.int32)
            cols = np.concatenate(cols_acc).astype(np.int32)
            np.add.at(W_offdiag, (rows, cols), 1)
            np.add.at(W_offdiag, (cols, rows), 1)   # symmetric

        # Cache hidden states
        if do_cache and n_cache > 0:
            ep = probe_filled + n_cache
            for l in probe_layers:
                hs = out.hidden_states[l + 1][0, :n_cache].cpu().to(torch.float16).numpy()
                probe_hidden[l][probe_filled:ep] = hs
            probe_token_ids[probe_filled:ep] = gold_np[:n_cache]
            probe_topk_ids[probe_filled:ep]  = topk_np[:n_cache].astype(np.int32)
            probe_filled = ep

        if (seq_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (seq_idx + 1) * (n_seq_total - seq_idx - 1)
            log.info("  [%d/%d]  probe=%d  %.1f seq/s  ETA %.0fs",
                     seq_idx + 1, n_seq_total, probe_filled,
                     (seq_idx + 1) / elapsed, eta)

    log.info("Inference done: %.1fs  probe_filled=%d", time.time() - t0, probe_filled)

    # Trim probe arrays to actual filled count
    pf = probe_filled
    for l in probe_layers:
        probe_hidden[l] = probe_hidden[l][:pf]
    probe_token_ids = probe_token_ids[:pf]
    probe_topk_ids  = probe_topk_ids[:pf]

    return W_offdiag, degree, probe_hidden, probe_token_ids, probe_topk_ids


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args) -> None:
    set_seed(args.seed)
    cache_dir = os.path.join(args.output_dir, "probe_cache")
    ensure_dirs(args.graphs_dir, cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    tok, model = load_model_tokenizer(args.model_name)
    token_ids  = get_corpus_tokens(
        tok, args.dataset_name, args.dataset_config,
        args.max_tokens, args.local_text_file,
    )

    freq_ids    = top_k_frequent_ids(token_ids, args.vocab_subset_size)
    vocab_to_sub = build_vocab_map(freq_ids, tok.vocab_size)
    log.info("Vocab subset: %d tokens  (vocab_size=%d)", len(freq_ids), tok.vocab_size)

    np.save(os.path.join(args.graphs_dir, "frequent_token_ids.npy"), freq_ids)
    np.save(os.path.join(args.graphs_dir, "vocab_to_sub.npy"), vocab_to_sub)

    # ── Inference + graph building ─────────────────────────────────────────
    probe_layers = list(args.probe_layers)
    W_offdiag, degree, probe_hidden, probe_token_ids, probe_topk_ids = run_inference(
        model, token_ids, freq_ids, vocab_to_sub,
        top_k=args.top_k,
        probe_layers=probe_layers,
        n_probe=args.n_probe,
        seq_len=args.seq_len,
        device=device,
    )

    # ── Normalize + optional hybrid ───────────────────────────────────────
    log.info("Normalizing coactivation matrix ...")
    W_norm = normalize_coactivation(W_offdiag, degree)

    if args.gamma > 0.0:
        log.info("Adding embedding similarity graph (gamma=%.2f) ...", args.gamma)
        S_emb = embedding_similarity_graph(model, freq_ids)
        W_norm = args.alpha * W_norm + args.gamma * S_emb
        np.fill_diagonal(W_norm, 0.0)

    # ── Save graphs ───────────────────────────────────────────────────────
    def to_sparse(M: np.ndarray) -> sparse.csr_matrix:
        S = sparse.csr_matrix(M)
        S.eliminate_zeros()
        return S

    # Raw W: add diagonal = degree for reference
    W_full = W_offdiag.copy()
    np.fill_diagonal(W_full, degree)
    sparse.save_npz(os.path.join(args.graphs_dir, "coactivation_W.npz"),      to_sparse(W_full))
    sparse.save_npz(os.path.join(args.graphs_dir, "coactivation_W_norm.npz"), to_sparse(W_norm))
    nnz = to_sparse(W_norm).nnz
    log.info("Graph saved: %d nodes, %d nonzeros (%.2f%%)",
             len(freq_ids), nnz, 100.0 * nnz / (len(freq_ids) ** 2))

    # ── Save probe cache ──────────────────────────────────────────────────
    for l in probe_layers:
        np.save(os.path.join(cache_dir, f"hidden_L{l:02d}.npy"), probe_hidden[l])
    np.save(os.path.join(cache_dir, "probe_token_ids.npy"), probe_token_ids)
    np.save(os.path.join(cache_dir, "topk_ids.npy"), probe_topk_ids)
    save_json(
        {
            "layers": probe_layers,
            "N_probe": int(len(probe_token_ids)),
            "D": int(model.config.hidden_size),
            "model_name": args.model_name,
            "top_k": args.top_k,
            "vocab_subset_size": args.vocab_subset_size,
        },
        os.path.join(cache_dir, "meta.json"),
    )
    log.info("Probe cache: %d positions, %d layers", len(probe_token_ids), len(probe_layers))
    log.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Build interference coactivation graph",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_name",        default="gpt2-xl")
    p.add_argument("--dataset_name",      default="wikitext")
    p.add_argument("--dataset_config",    default="wikitext-2-raw-v1")
    p.add_argument("--max_tokens",        type=int, default=500_000)
    p.add_argument("--vocab_subset_size", type=int, default=5_000)
    p.add_argument("--top_k",             type=int, default=50)
    p.add_argument("--output_dir",        default="interference_region_pipeline/results")
    p.add_argument("--graphs_dir",        default="interference_region_pipeline/graphs")
    p.add_argument("--local_text_file",   default=None)
    p.add_argument("--seq_len",           type=int, default=512)
    p.add_argument("--probe_layers",      type=int, nargs="+", default=PROBE_LAYERS_DEFAULT)
    p.add_argument("--n_probe",           type=int, default=50_000,
                   help="Positions to cache for probe training")
    p.add_argument("--alpha",             type=float, default=1.0)
    p.add_argument("--beta",              type=float, default=0.0,
                   help="Gradient similarity weight (not implemented; leave 0)")
    p.add_argument("--gamma",             type=float, default=0.0,
                   help="Embedding similarity weight")
    p.add_argument("--seed",              type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
