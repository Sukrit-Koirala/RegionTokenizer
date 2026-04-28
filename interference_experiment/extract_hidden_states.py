"""
Core extraction pipeline.

For each token position in the corpus this script accumulates:
  1. top-k predicted token indices  → used by build_graph.py
  2. mean expected embedding         → used by gradient_graph.py
  3. token prediction frequencies    → used to select the vocabulary subset

Optionally, for a layer-comparison run, it also collects per-token mean
hidden states at a chosen intermediate layer.

Nothing is fine-tuned or trained — purely forward-pass inference.
"""

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    ExperimentConfig,
    get_device,
    save_json,
    set_seed,
    setup_dirs,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: ExperimentConfig, device: torch.device):
    """
    Try to load cfg.model_name; fall back to cfg.fallback_model_name on OOM
    or any load error.  Returns (model, tokenizer, model_name_used).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log = logging.getLogger("interference")

    for model_name in [cfg.model_name, cfg.fallback_model_name]:
        try:
            log.info("Loading tokenizer: %s", model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            dtype = (
                torch.float16
                if cfg.fp16 and device.type == "cuda"
                else torch.float32
            )
            log.info("Loading model: %s  dtype=%s", model_name, dtype)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model = model.to(device).eval()

            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            log.info("Loaded %s  (%.0f M params)", model_name, n_params)
            return model, tokenizer, model_name

        except (OSError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            log.warning("Could not load %s: %s — trying fallback.", model_name, exc)

    raise RuntimeError(
        f"Failed to load both {cfg.model_name} and {cfg.fallback_model_name}."
    )


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus_tokens(cfg: ExperimentConfig, tokenizer) -> np.ndarray:
    """
    Load WikiText-2 from HuggingFace datasets, tokenize, and return a flat
    int32 array of at most 2 * cfg.max_tokens tokens.
    """
    log = logging.getLogger("interference")

    try:
        from datasets import load_dataset

        log.info("Fetching %s / %s …", cfg.dataset_name, cfg.dataset_config)
        ds = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config,
            split="train+validation+test",
            trust_remote_code=False,
        )
        texts = [t for t in ds["text"] if t and len(t.strip()) > 20]
        log.info("Loaded %d non-empty documents.", len(texts))
    except Exception as exc:
        log.warning("Dataset load failed (%s); using built-in fallback.", exc)
        texts = _fallback_texts()

    token_budget = cfg.max_tokens * 2   # extra so trimming later is exact
    all_ids: List[int] = []
    for text in tqdm(texts, desc="Tokenising", leave=False):
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= token_budget:
            break

    arr = np.array(all_ids[:token_budget], dtype=np.int32)
    log.info("Corpus: %d tokens", len(arr))
    return arr


def _fallback_texts() -> List[str]:
    base = [
        "The quick brown fox jumps over the lazy dog. " * 500,
        (
            "In the beginning was the Word, and the Word was with God. "
            "The same was in the beginning with God. " * 400
        ),
        (
            "To be, or not to be, that is the question: "
            "Whether 'tis nobler in the mind to suffer "
            "the slings and arrows of outrageous fortune. " * 400
        ),
        "Science is the systematic study of the structure and behaviour of "
        "the physical and natural world through observation and experiment. " * 300,
    ]
    return base * 10   # repeat to get enough tokens


# ---------------------------------------------------------------------------
# Unembedding matrix
# ---------------------------------------------------------------------------

def get_unembedding_matrix(model, device: torch.device) -> torch.Tensor:
    """
    Return the LM-head weight matrix as float32 on CPU, shape (vocab_size, hidden_size).
    Handles weight-tied models (GPT-2) and independent heads (OPT, LLaMA).
    """
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W = model.lm_head.weight.detach().float().cpu()
    elif hasattr(model, "embed_out") and hasattr(model.embed_out, "weight"):
        W = model.embed_out.weight.detach().float().cpu()
    else:
        raise AttributeError(
            "Cannot locate unembedding weight in model.  "
            "Expected model.lm_head.weight or model.embed_out.weight."
        )

    # Ensure shape is (vocab_size, hidden_size), not (hidden_size, vocab_size)
    if W.shape[0] < W.shape[1]:
        W = W.T
    return W  # (V, D)


# ---------------------------------------------------------------------------
# Layer hook helper
# ---------------------------------------------------------------------------

def _get_transformer_layers(model) -> list:
    """Return the list of transformer block modules for GPT-2/LLaMA/GPT-NeoX."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)        # GPT-2 family
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)         # LLaMA / Mistral
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)      # Pythia / GPT-NeoX
    return []


def _register_hidden_hook(model, layer_index: int) -> Tuple[object, dict]:
    """
    Register a forward hook on the specified transformer layer.
    Returns (handle, cache_dict).  cache_dict['h'] is populated after each
    forward pass.
    """
    layers = _get_transformer_layers(model)
    if not layers:
        raise ValueError("Could not locate transformer layers for hook registration.")

    n_layers = len(layers)
    if layer_index < 0:
        layer_index = n_layers + layer_index   # e.g. -1 → last

    if not (0 <= layer_index < n_layers):
        raise IndexError(
            f"layer_index {layer_index} out of range [0, {n_layers - 1}]."
        )

    cache: dict = {}

    def hook(module, inp, output):
        # GPT-2 blocks return (hidden, present_kv, ...) tuples
        h = output[0] if isinstance(output, tuple) else output
        cache["h"] = h.detach()

    handle = layers[layer_index].register_forward_hook(hook)
    return handle, cache, layer_index


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract(
    cfg: ExperimentConfig,
    layer_index: Optional[int] = None,
    collect_mean_hidden: bool = False,
) -> Dict:
    """
    Run inference over the corpus and accumulate:
      - topk_indices          (N_positions, top_k)  uint16
      - token_freqs           (vocab_size,)           int32  — prediction freq
      - mean_expected_emb     (hidden_size,)          float32
      - mean_hidden_per_token (vocab_size, hidden_size) float32  [optional]

    Arguments
    ---------
    layer_index          Layer whose hidden state to collect (overrides cfg).
                         Only relevant when collect_mean_hidden=True.
    collect_mean_hidden  If True, accumulate per-predicted-token mean hidden
                         state (used for layer-comparison analysis).
    """
    log = logging.getLogger("interference")
    device = get_device(cfg)
    set_seed(cfg.seed)

    if layer_index is None:
        layer_index = cfg.layer_index

    model, tokenizer, model_name = load_model_and_tokenizer(cfg, device)
    vocab_size = len(tokenizer)

    # Unembedding matrix lives on CPU in float32 to avoid polluting GPU memory
    W_unembed = get_unembedding_matrix(model, device)  # (V, D), CPU float32
    hidden_size = W_unembed.shape[1]
    log.info("vocab=%d  hidden=%d  top_k=%d", vocab_size, cfg.top_k, cfg.top_k)

    # Move unembedding to GPU once for the batch matmul
    W_unembed_gpu = W_unembed.to(device)

    # Register hook if needed
    hook_handle = None
    hook_cache: dict = {}
    resolved_layer: Optional[int] = None

    if collect_mean_hidden:
        hook_handle, hook_cache, resolved_layer = _register_hidden_hook(
            model, layer_index
        )
        log.info("Collecting hidden states at layer %d", resolved_layer)

    # Tokenise corpus into non-overlapping windows of length seq_len
    corpus = load_corpus_tokens(cfg, tokenizer)
    step = cfg.seq_len
    windows = [
        corpus[i : i + cfg.seq_len + 1]
        for i in range(0, len(corpus) - cfg.seq_len - 1, step)
    ]
    log.info("Windows: %d  (seq_len=%d, batch=%d)", len(windows), cfg.seq_len, cfg.batch_size)

    # Streaming accumulators — never store all activations simultaneously
    all_topk: List[np.ndarray] = []      # each (B*S, top_k) uint16
    token_freqs = np.zeros(vocab_size, dtype=np.int64)
    mean_emb_acc = np.zeros(hidden_size, dtype=np.float64)  # float64 for precision
    n_positions = 0

    # For per-token mean hidden state (layer comparison)
    if collect_mean_hidden:
        h_sum = np.zeros((vocab_size, hidden_size), dtype=np.float64)
        h_cnt = np.zeros(vocab_size, dtype=np.int64)

    eos_id = tokenizer.eos_token_id or 0
    n_batches = (len(windows) + cfg.batch_size - 1) // cfg.batch_size

    with torch.no_grad():
        for b_idx in tqdm(range(n_batches), desc="Extracting"):
            batch_wins = windows[b_idx * cfg.batch_size : (b_idx + 1) * cfg.batch_size]
            if not batch_wins:
                continue

            # ---- Build padded input tensor ----
            max_len = max(len(w) for w in batch_wins)
            padded = np.full((len(batch_wins), max_len), eos_id, dtype=np.int32)
            for i, w in enumerate(batch_wins):
                padded[i, : len(w)] = w

            input_ids = torch.tensor(
                padded[:, :-1], dtype=torch.long, device=device
            )  # (B, S)

            # ---- Forward pass ----
            out = model(input_ids=input_ids)
            logits = out.logits  # (B, S, V)

            B, S, V = logits.shape

            # ---- Top-k indices ----
            topk_t = torch.topk(logits, k=cfg.top_k, dim=-1).indices  # (B, S, K)
            topk_np = topk_t.cpu().to(torch.int32).numpy()             # (B, S, K) int32
            all_topk.append(topk_np.reshape(B * S, cfg.top_k).astype(np.uint16))

            # Update prediction-frequency counts (using top-1 as "predicted token")
            top1_flat = topk_np[:, :, 0].flatten().astype(np.int32)
            np.add.at(token_freqs, top1_flat, 1)

            # ---- Accumulate mean expected embedding ----
            # E_p[W] at each position t = Σ_i p_i(t) * W_i
            # Averaged over all positions gives us mean_expected_embedding,
            # which is the context-averaged baseline for the gradient graph.
            logits_2d = logits.reshape(B * S, V).float()
            probs = torch.softmax(logits_2d, dim=-1)  # (N, V)

            # matmul: (N, V) × (V, D) → (N, D)
            exp_emb = probs @ W_unembed_gpu      # (N, D), GPU float32
            mean_emb_acc += exp_emb.sum(dim=0).cpu().numpy().astype(np.float64)
            n_positions += B * S

            # ---- Collect per-token hidden states (optional) ----
            if collect_mean_hidden and "h" in hook_cache:
                h = hook_cache["h"].float().cpu().numpy()  # (B, S, D)
                h_2d = h.reshape(B * S, hidden_size)
                for pos_idx in range(B * S):
                    tok = int(top1_flat[pos_idx])
                    if 0 <= tok < vocab_size:
                        h_sum[tok] += h_2d[pos_idx]
                        h_cnt[tok] += 1
                hook_cache.clear()

            # Free GPU tensors immediately
            del logits, topk_t, probs, exp_emb, logits_2d
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if hook_handle is not None:
        hook_handle.remove()

    del W_unembed_gpu  # release GPU copy

    # ---- Finalise ----
    topk_indices = np.vstack(all_topk)   # (N_total, K) uint16
    mean_expected_emb = (mean_emb_acc / max(n_positions, 1)).astype(np.float32)

    result = {
        "topk_indices": topk_indices,
        "token_freqs": token_freqs.astype(np.int32),
        "mean_expected_embedding": mean_expected_emb,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "n_positions": n_positions,
        "model_name": model_name,
        "layer_index": resolved_layer if collect_mean_hidden else layer_index,
    }

    if collect_mean_hidden:
        # Compute mean hidden per token (set to zero for unseen tokens)
        mask = h_cnt > 0
        mean_hidden = np.zeros((vocab_size, hidden_size), dtype=np.float32)
        mean_hidden[mask] = (h_sum[mask] / h_cnt[mask, None]).astype(np.float32)
        result["mean_hidden_per_token"] = mean_hidden
        result["token_seen_counts"] = h_cnt

    log.info(
        "Done — %d positions  |  top_k=%d  |  vocab=%d",
        n_positions, cfg.top_k, vocab_size,
    )
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_extracted(data: Dict, cfg: ExperimentConfig, layer_tag: str = "final") -> str:
    """
    Save extraction artefacts to cfg.graphs_dir with a layer_tag prefix.
    Returns the prefix string so callers can construct derived filenames.
    """
    prefix = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}")

    np.save(f"{prefix}_topk_indices.npy", data["topk_indices"])
    np.save(f"{prefix}_token_freqs.npy", data["token_freqs"])
    np.save(f"{prefix}_mean_expected_embedding.npy", data["mean_expected_embedding"])

    # Derive top-N frequent token IDs and save for downstream scripts
    top_ids = np.argsort(data["token_freqs"])[::-1][: cfg.vocab_subset_size]
    np.save(f"{prefix}_frequent_token_ids.npy", top_ids.astype(np.int32))

    if "mean_hidden_per_token" in data:
        np.save(f"{prefix}_mean_hidden_per_token.npy", data["mean_hidden_per_token"])

    meta = {
        "vocab_size": int(data["vocab_size"]),
        "hidden_size": int(data["hidden_size"]),
        "n_positions": int(data["n_positions"]),
        "model_name": data["model_name"],
        "top_k": cfg.top_k,
        "layer_tag": layer_tag,
        "layer_index": data["layer_index"],
        "vocab_subset_size": cfg.vocab_subset_size,
    }
    save_json(f"{prefix}_meta.json", meta)

    log = logging.getLogger("interference")
    log.info("Artefacts saved to %s_*", prefix)
    return prefix


def load_extracted(prefix: str, need_hidden: bool = False) -> Dict:
    """Load artefacts produced by save_extracted."""
    data = {
        "topk_indices": np.load(f"{prefix}_topk_indices.npy"),
        "token_freqs": np.load(f"{prefix}_token_freqs.npy"),
        "mean_expected_embedding": np.load(f"{prefix}_mean_expected_embedding.npy"),
        "frequent_token_ids": np.load(f"{prefix}_frequent_token_ids.npy"),
    }
    from utils import load_json
    data.update(load_json(f"{prefix}_meta.json"))

    if need_hidden:
        path = f"{prefix}_mean_hidden_per_token.npy"
        if os.path.exists(path):
            data["mean_hidden_per_token"] = np.load(path)

    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract top-k predictions and gradients.")
    parser.add_argument("--model", default="gpt2-xl")
    parser.add_argument("--max_tokens", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--layer_index", type=int, default=-1,
                        help="-1 = final layer")
    parser.add_argument("--layer_tag", default="final",
                        help="String label appended to saved filenames.")
    parser.add_argument("--collect_hidden", action="store_true",
                        help="Also collect per-token mean hidden states.")
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()

    setup_logging()
    cfg = ExperimentConfig(
        model_name=args.model,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        layer_index=args.layer_index,
        fp16=not args.no_fp16,
    )
    setup_dirs(cfg)

    data = extract(cfg, layer_index=args.layer_index, collect_mean_hidden=args.collect_hidden)
    save_extracted(data, cfg, layer_tag=args.layer_tag)


if __name__ == "__main__":
    main()
