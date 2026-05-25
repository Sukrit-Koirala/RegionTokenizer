#!/usr/bin/env python3
"""
eval_multitoken_path_oracle_v1.py — Multi-Token Path Oracle Diagnostic.

⚠ ORACLE_DIAGNOSTIC ONLY ⚠
This script intentionally uses future gold continuation tokens.
It MUST NOT be used as real validation metrics.
It MUST NOT influence model training or checkpoint selection.
It is an upper-bound existence test for multi-token path-memory signal.

For each val row, forces each candidate as the next token and scores the true
future continuation.  The question: does the gold candidate consistently produce
better continuation paths?

Usage:
  python scripts/eval_multitoken_path_oracle_v1.py \
    --val_dir runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val \
    --small_ckpt runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \
    --token_to_region runs/region_maps_128/token_to_region.json \
    --super_map runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \
    --output_dir runs/path_memory_v1/multitoken_oracle_v1 \
    --eval_filter bucketA_confuser \
    --candidate_pool_size 16 \
    --path_lens 1,2,4,8 \
    --alpha_grid 0.25,0.5,1.0,2.0 \
    --max_rows 5000 \
    --batch_size 16 \
    --amp \
    --device cuda \
    --seed 42
"""

import argparse
import contextlib
import csv
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

_EPS = 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer (optional — for example decoding only)
# ─────────────────────────────────────────────────────────────────────────────

_enc = None
try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
except Exception:
    pass

def _decode_tok(tid):
    if _enc is not None:
        try:
            return repr(_enc.decode([int(tid)]))[1:-1]
        except Exception:
            pass
    return f"<{int(tid)}>"

def _decode_ids(ids):
    if _enc is not None:
        try:
            return _enc.decode([int(i) for i in ids if 0 <= int(i) < 50257])
        except Exception:
            pass
    return " ".join(_decode_tok(i) for i in ids)

def _classify_token(tok_id):
    """Heuristic type classification for slice analysis."""
    s = _decode_tok(tok_id)
    stripped = s.strip()
    if stripped in {',', '.', '?', '!', ';', ':', "'", '"', '-',
                    '(', ')', '[', ']', '{', '}', '/', '\\', '``', "''"}:
        return "punctuation"
    if stripped.isdigit() or (stripped.replace('.', '', 1).isdigit() and '.' in stripped):
        return "number"
    if s.startswith(' ') and stripped.isalpha():
        return "word_choice"
    if stripped.isalpha() and not s.startswith(' ') and len(stripped) >= 2:
        return "bpe_fragment"
    return "other"

# ─────────────────────────────────────────────────────────────────────────────
# Shard field aliases
# ─────────────────────────────────────────────────────────────────────────────

_TOPK_ALIASES  = ["base_topk_ids",    "base_topk",      "topk_ids"]
_LGT_ALIASES   = ["base_topk_logits", "base_topk_lgt",  "topk_lgt", "topk_logits"]
_GOLD_ALIASES  = ["gold_token",       "gold",           "labels"]
_IDS_ALIASES   = ["input_ids"]
_HCTX_ALIASES  = ["h_ctx",  "h_context",  "hidden_ctx"]
_ROWID_ALIASES = ["row_id", "row_ids"]
_OFF_ALIASES   = ["token_offset", "offsets", "offset"]

def _get(shard, aliases, required=True):
    for a in aliases:
        if a in shard:
            return shard[a]
    if required:
        raise KeyError(f"Missing field. Tried: {aliases}. Have: {list(shard.keys())}")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    """Load frozen small backbone from checkpoint. Returns (backbone, emb_w, seq_len, vocab_size)."""
    from scripts.offline_region_knn import load_small_backbone_and_probe
    print(f"[model] Loading checkpoint: {ckpt_path}")
    backbone, _, d_model, cfg_dict, vocab_size = load_small_backbone_and_probe(
        ckpt_path, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    seq_len = backbone.pos_emb.num_embeddings
    emb_w   = backbone.token_emb.weight.detach().float().cpu()   # [vocab, d]
    print(f"  d_model={d_model}  vocab_size={vocab_size}  seq_len={seq_len}")
    return backbone, emb_w, seq_len, vocab_size, cfg_dict

# ─────────────────────────────────────────────────────────────────────────────
# Corpus loading (val split — for future tokens via token_offset)
# ─────────────────────────────────────────────────────────────────────────────

def load_val_corpus(val_dir, corpus_path_arg=None):
    """
    Returns (corpus_np int32, corpus_len).
    Search order:
      1. --corpus_path argument (if given)
      2. Pipeline stage-00 cache next to val_dir
      3. HuggingFace WikiText-103 validation (last resort)
    """
    if corpus_path_arg and os.path.isfile(corpus_path_arg):
        print(f"[corpus] Loading from --corpus_path: {corpus_path_arg}")
        corp = np.load(corpus_path_arg).astype(np.int32) if corpus_path_arg.endswith(".npy") \
               else np.frombuffer(open(corpus_path_arg, "rb").read(), dtype=np.int32)
        print(f"  corpus length: {len(corp):,}")
        return corp

    # Try to auto-detect from pipeline layout:
    # val_dir = .../01_live_dataset_patched/val
    # manifest/stage-00 report would be at .../00_raw_token_source/report.json
    for steps_up in [2, 3]:
        candidate_root = val_dir
        for _ in range(steps_up):
            candidate_root = os.path.dirname(candidate_root)
        report_path = os.path.join(candidate_root, "00_raw_token_source", "report.json")
        if os.path.isfile(report_path):
            with open(report_path) as f:
                rpt = json.load(f)
            vp = rpt.get("val_tokens_path", "")
            if vp and os.path.isfile(vp):
                print(f"[corpus] Using pipeline cache: {vp}")
                corp = np.load(vp).astype(np.int32) if vp.endswith(".npy") \
                       else np.frombuffer(open(vp, "rb").read(), dtype=np.int32)
                print(f"  corpus length: {len(corp):,}")
                return corp
        # also try a val_tokens.npy next to the report dir
        vp2 = os.path.join(candidate_root, "00_raw_token_source", "val_tokens.npy")
        if os.path.isfile(vp2):
            print(f"[corpus] Using cached val_tokens.npy: {vp2}")
            corp = np.load(vp2).astype(np.int32)
            print(f"  corpus length: {len(corp):,}")
            return corp

    # Last resort: HuggingFace
    print("[corpus] Cached corpus not found. Loading from HuggingFace WikiText-103 (validation)...")
    print("  (This requires internet access and ~10–30s)")
    try:
        from transformers import GPT2TokenizerFast
        from datasets import load_dataset as hf_load
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.model_max_length = int(1e30)
        raw = hf_load("wikitext", "wikitext-103-raw-v1")
        parts = []
        for text in raw["validation"]["text"]:
            if text.strip():
                ids = tok.encode(text)
                if ids:
                    parts.append(np.array(ids, dtype=np.int32))
        corp = np.concatenate(parts)
        print(f"  corpus length: {len(corp):,}")
        return corp
    except Exception as e:
        raise RuntimeError(
            f"Could not load val corpus from HuggingFace: {e}\n"
            "Please provide --corpus_path pointing to the val_tokens.npy file.\n"
            "future continuation tokens unavailable; cannot run path oracle"
        ) from e

# ─────────────────────────────────────────────────────────────────────────────
# Val shard loading
# ─────────────────────────────────────────────────────────────────────────────

def load_val_shards(val_dir, M, max_rows=None):
    """Load all val shards. Returns raw data dict; filtering applied later."""
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {val_dir}")
    print(f"[shards] {len(paths)} val shards")

    bufs = defaultdict(list)
    total = 0
    first = True
    for sp in paths:
        if max_rows is not None and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print("[preflight] val shard keys:")
            for k, v in sh.items():
                print(f"  {k}: shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',type(v).__name__)}")
            first = False

        topk = _get(sh, _TOPK_ALIASES).long()
        lgt  = _get(sh, _LGT_ALIASES).float()
        gold = _get(sh, _GOLD_ALIASES).long()
        ids  = _get(sh, _IDS_ALIASES).long()
        rids = _get(sh, _ROWID_ALIASES, required=False)
        offs = _get(sh, _OFF_ALIASES, required=False)
        B    = topk.shape[0]
        if rids is None:
            rids = torch.arange(total, total + B)
        if offs is None:
            offs = None   # will fail later when corpus lookup is attempted

        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, ids, rids = (x[:keep] for x in (topk, lgt, gold, ids, rids))
            if offs is not None:
                offs = offs[:keep]
            B = keep

        bufs["topk"].append(topk.numpy())
        bufs["lgt"].append(lgt.numpy())
        bufs["gold"].append(gold.numpy())
        bufs["ids"].append(ids.numpy())
        bufs["row_id"].append(rids.numpy())
        if offs is not None:
            bufs["token_offset"].append(offs.numpy())
        total += B

    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    print(f"[shards] loaded {total:,} rows  topk_K={data['topk'].shape[1]}")

    if "token_offset" not in data:
        raise RuntimeError(
            "token_offset field not found in val shards.\n"
            "Cannot reconstruct future continuation tokens.\n"
            "future continuation tokens unavailable; cannot run path oracle\n"
            "Check that shards were built by the live pipeline (stage 01 / run_live_full_pipeline_rebuild.py)."
        )

    # Clip candidates to pool size
    K = data["topk"].shape[1]
    M_use = min(M, K)
    data["cand_ids"] = data["topk"][:, :M_use].astype(np.int64)     # [N, M]
    data["cand_lgt"] = np.where(
        np.isfinite(data["lgt"][:, :M_use]), data["lgt"][:, :M_use], -1e9
    ).astype(np.float32)  # [N, M]
    data["base_top1"] = data["topk"][:, 0].astype(np.int64)         # [N]
    data["M"] = M_use
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Corpus alignment check
# ─────────────────────────────────────────────────────────────────────────────

def verify_corpus_alignment(data, corpus, n_check=500, seed=42):
    """Verify corpus[token_offset[i]] == gold[i] for a sample of rows."""
    rng   = np.random.default_rng(seed)
    N     = len(data["gold"])
    idxs  = rng.choice(N, size=min(n_check, N), replace=False)
    offs  = data["token_offset"][idxs].astype(np.int64)
    golds = data["gold"][idxs].astype(np.int64)
    cl    = len(corpus)
    valid = (offs >= 0) & (offs < cl)
    if not valid.all():
        print(f"  [WARN] {(~valid).sum()} token_offsets out of corpus range [{0}, {cl})")
    corp_golds = np.where(valid, corpus[np.clip(offs, 0, cl - 1)].astype(np.int64), -1)
    mismatches = int(((corp_golds != golds) & valid).sum())
    if mismatches > 0:
        print(f"  [WARN] corpus alignment: {mismatches}/{valid.sum()} mismatches "
              f"(corpus split may not match token_offset split)")
        if mismatches / max(int(valid.sum()), 1) > 0.1:
            raise RuntimeError(
                f"Corpus alignment failure: {mismatches}/{valid.sum()} gold tokens "
                "don't match corpus[token_offset]. "
                "Ensure --corpus_path points to the val split used to build the shards.\n"
                "future continuation tokens unavailable; cannot run path oracle"
            )
    else:
        print(f"  [OK] corpus alignment verified on {valid.sum()} rows")

# ─────────────────────────────────────────────────────────────────────────────
# Region maps
# ─────────────────────────────────────────────────────────────────────────────

def load_maps(t2r_path, super_path=None):
    with open(t2r_path) as f:
        raw = json.load(f)
    t2r = ({i: v for i, v in enumerate(raw) if v is not None} if isinstance(raw, list)
           else {int(k): v for k, v in raw.items()})
    unk_r = int(max(t2r.values())) + 1 if t2r else 1
    r2s = {}; unk_s = 1; sr = False
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f:
            raw2 = json.load(f)
        r2s = ({int(k): v for k, v in raw2.items()} if not isinstance(raw2, list)
               else {i: v for i, v in enumerate(raw2) if v is not None})
        unk_s = int(max(r2s.values())) + 1 if r2s else 1
        sr = True
    V = max(t2r.keys(), default=0) + 2
    tok_arr = np.full(V, unk_r, dtype=np.int32)
    for t, r in t2r.items():
        if 0 <= t < V:
            tok_arr[t] = int(r)
    R = unk_r + 2
    reg_arr = np.full(R, unk_s, dtype=np.int32)
    for r, s in r2s.items():
        if 0 <= int(r) < R:
            reg_arr[int(r)] = int(s)
    return tok_arr, reg_arr, unk_r, unk_s, sr

# ─────────────────────────────────────────────────────────────────────────────
# Slices
# ─────────────────────────────────────────────────────────────────────────────

def build_slices(data, tok_arr, reg_arr, unk_r, unk_s, sr, memory_len=128):
    gold     = data["gold"].astype(np.int64)
    base_top1= data["base_top1"].astype(np.int64)
    cand_ids = data["cand_ids"]
    cand_lgt = data["cand_lgt"]
    inp_ids  = data["ids"]
    N        = len(gold)
    V        = tok_arr.shape[0]; R = reg_arr.shape[0]

    gold_in_pool = (cand_ids == gold[:, None]).any(axis=1)
    base_wrong   = base_top1 != gold

    # Gold in recent context?
    T = min(memory_len, inp_ids.shape[1])
    ctx = inp_ids[:, -T:]
    gold_in_ctx = np.array([(ctx[i] == gold[i]).any() for i in range(N)], dtype=bool)

    # Base logit margin
    if cand_lgt.shape[1] >= 2:
        top2 = np.sort(cand_lgt, axis=1)[:, -2]
        base_margin = cand_lgt.max(axis=1) - top2
    else:
        base_margin = np.zeros(N, dtype=np.float32)
    med_margin = float(np.median(base_margin))

    # Region-based
    base_reg = tok_arr[base_top1.clip(0, V-1)]
    gold_reg  = tok_arr[gold.clip(0, V-1)]
    same_reg  = (base_reg == gold_reg) & (base_reg != unk_r)
    same_sup  = np.zeros(N, dtype=bool)
    if sr:
        base_sup = reg_arr[base_reg.clip(0, R-1)]
        gold_sup = reg_arr[gold_reg.clip(0, R-1)]
        same_sup = (base_sup == gold_sup) & (base_sup != unk_s)

    # Token-type heuristic slices (on gold token)
    types = np.array([_classify_token(int(g)) for g in gold], dtype=object)
    bpe_frag    = types == "bpe_fragment"
    punct       = types == "punctuation"
    number      = types == "number"
    word_choice = types == "word_choice"

    return {
        "all":                       np.ones(N, dtype=bool),
        "bucketA_confuser":          base_wrong & gold_in_pool,
        "gold_not_in_context":       ~gold_in_ctx,
        "gold_in_context":           gold_in_ctx,
        "high_base_margin":          base_margin > med_margin,
        "low_base_margin":           base_margin <= med_margin,
        "same_region_confuser":      base_wrong & gold_in_pool & same_reg,
        "same_superregion_confuser": base_wrong & gold_in_pool & same_sup,
        "bpe_fragment_like":         bpe_frag,
        "punctuation_like":          punct,
        "number_like":               number,
        "word_choice_like":          word_choice,
    }, gold_in_pool, base_wrong, base_margin

# ─────────────────────────────────────────────────────────────────────────────
# Forward pass helper
# ─────────────────────────────────────────────────────────────────────────────

def _forward_logits(backbone, emb_w, seqs_cpu, device, use_amp):
    """
    seqs_cpu: [B, T] int64 numpy
    Returns logits [B, T, vocab] float32 on CPU.
    """
    seqs_t = torch.from_numpy(seqs_cpu).long().to(device)
    T = seqs_t.shape[1]
    pos = torch.arange(T, device=device).unsqueeze(0)

    if use_amp and device.type == "cuda":
        amp_ctx = torch.amp.autocast("cuda")
    else:
        amp_ctx = contextlib.nullcontext()

    with torch.no_grad(), amp_ctx:
        x = backbone.drop(backbone.token_emb(seqs_t) + backbone.pos_emb(pos))
        for blk in backbone.blocks:
            x = blk(x)
        h = backbone.ln_f(x)   # [B, T, d]
    h = h.float().cpu()        # [B, T, d] float32
    # Project to vocab using weight-tied embedding
    B_, T_, d_ = h.shape
    logits = (h.view(B_ * T_, d_) @ emb_w.T).view(B_, T_, -1)   # [B, T, vocab]
    return logits

# ─────────────────────────────────────────────────────────────────────────────
# Continuation NLL computation — ORACLE: uses future gold tokens
# ─────────────────────────────────────────────────────────────────────────────

def compute_path_results(backbone, emb_w, seq_len, data, corpus,
                         path_lens, batch_size, device, use_amp):
    """
    ⚠ ORACLE: reads future gold tokens from corpus via token_offset.

    Returns dict: L -> {"cont_nll": [N, M] float32, "has_future": [N] bool}
    For L=1: cont_nll is all-zero (no continuation scoring needed).
    """
    N         = len(data["gold"])
    M         = data["M"]
    inp_ids   = data["ids"].astype(np.int64)         # [N, ctx_len]
    offsets   = data["token_offset"].astype(np.int64) # [N]
    cand_ids  = data["cand_ids"]                      # [N, M]
    corp_len  = len(corpus)
    ctx_len   = inp_ids.shape[1]

    results = {}
    print(f"\n[path_oracle] ⚠ ORACLE: reading future tokens from corpus. "
          f"N={N:,}  M={M}  seq_len={seq_len}")

    for L in path_lens:
        print(f"\n[path_oracle] path_len={L}")
        if L == 1:
            results[L] = {
                "cont_nll":  np.zeros((N, M), dtype=np.float32),
                "has_future": np.ones(N, dtype=bool),
            }
            print("  L=1: no continuation scoring (immediate logit only)")
            continue

        # T_prefix: how much of inp_ids to use so total seq = T_prefix + 1 + (L-1) = seq_len
        T_prefix  = seq_len - L
        if T_prefix <= 0:
            raise ValueError(
                f"path_len={L} >= seq_len={seq_len}. "
                "Reduce path_len or increase model seq_len.")

        # has_future: need corpus[offset+1 .. offset+L-1] — requires offset+L-1 < corp_len
        has_future = (offsets + L - 1) < corp_len    # [N] bool
        n_ok = int(has_future.sum())
        print(f"  has_future: {n_ok}/{N}  (rows with enough future tokens)")
        if n_ok == 0:
            raise RuntimeError(
                f"No rows have enough future tokens for path_len={L}. "
                "corpus may be misaligned.\n"
                "future continuation tokens unavailable; cannot run path oracle"
            )

        cont_nll_all = np.zeros((N, M), dtype=np.float32)
        t0 = time.time()
        sub_bs = 64   # forward pass sub-batch size (rows × M sequences)

        for b_start in range(0, N, batch_size):
            b_end   = min(b_start + batch_size, N)
            B       = b_end - b_start
            off_b   = offsets[b_start:b_end]     # [B]
            hf_b    = has_future[b_start:b_end]  # [B]
            ids_b   = inp_ids[b_start:b_end]     # [B, ctx_len]
            cand_b  = cand_ids[b_start:b_end]    # [B, M]

            # Trim prefix to T_prefix tokens
            prefix_b = ids_b[:, -T_prefix:] if ctx_len > T_prefix else ids_b  # [B, T_prefix]
            # Pad left if ctx_len < T_prefix (unlikely, but safe)
            if prefix_b.shape[1] < T_prefix:
                pad = np.zeros((B, T_prefix - prefix_b.shape[1]), dtype=np.int64)
                prefix_b = np.concatenate([pad, prefix_b], axis=1)

            # Future tokens: [B, L-1], zeros for rows with has_future=False
            future_b = np.zeros((B, L - 1), dtype=np.int64)
            for i in range(B):
                if hf_b[i]:
                    off = int(off_b[i])
                    future_b[i] = corpus[off + 1: off + L].astype(np.int64)

            # Build sequences [B, M, seq_len]: prefix | candidate | future
            # Axis layout: seqs[i, j, :] = row i, candidate j
            seqs = np.zeros((B, M, seq_len), dtype=np.int64)
            seqs[:, :, :T_prefix] = prefix_b[:, None, :]            # broadcast over M
            for j in range(M):
                seqs[:, j, T_prefix] = cand_b[:, j]
            seqs[:, :, T_prefix + 1:] = future_b[:, None, :]        # broadcast over M

            # Reshape to [B*M, seq_len] for batched forward pass
            seqs_flat = seqs.reshape(B * M, seq_len)

            # Process seqs_flat in sub-batches to bound GPU memory
            cont_nll_flat = np.zeros(B * M, dtype=np.float32)
            for s in range(0, B * M, sub_bs):
                s_end  = min(s + sub_bs, B * M)
                chunk  = seqs_flat[s:s_end]                          # [sub, seq_len]
                logits = _forward_logits(backbone, emb_w, chunk, device, use_amp)
                # logit[t] predicts token t+1
                # Continuation logits: positions T_prefix..T_prefix+L-2 → predict future[0..L-2]
                cont_lgt = logits[:, T_prefix: T_prefix + L - 1, :]  # [sub, L-1, vocab]
                # Targets: tokens at positions T_prefix+1..T_prefix+L-1 = future[0..L-2]
                tgt = torch.from_numpy(
                    seqs_flat[s:s_end, T_prefix + 1: T_prefix + L]
                ).long()                                              # [sub, L-1]
                log_p = F.log_softmax(cont_lgt.float(), dim=-1)
                nll   = -log_p.gather(2, tgt.unsqueeze(2)).squeeze(2)  # [sub, L-1]
                cont_nll_flat[s:s_end] = nll.mean(dim=1).numpy()

            # Check for NaN
            if np.isnan(cont_nll_flat).any():
                raise RuntimeError(
                    f"NaN in continuation NLL at batch {b_start} path_len={L}. "
                    "Check checkpoint and input data.")

            cont_nll_bm = cont_nll_flat.reshape(B, M)
            # Zero out rows where future tokens unavailable (they won't contribute to metrics)
            cont_nll_bm[~hf_b] = 0.0
            cont_nll_all[b_start:b_end] = cont_nll_bm

            if (b_start // batch_size) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {b_end}/{N}  ({elapsed:.1f}s)")

        print(f"  done in {time.time() - t0:.1f}s")
        results[L] = {
            "cont_nll":   cont_nll_all,
            "has_future": has_future,
        }

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Rank utilities
# ─────────────────────────────────────────────────────────────────────────────

def ranks_descending(scores):
    """[N, M] → [N, M] int rank (0 = best = highest score)."""
    return (-scores).argsort(axis=1).argsort(axis=1).astype(np.int32)

def ranks_ascending(scores):
    """[N, M] → [N, M] int rank (0 = best = lowest score)."""
    return scores.argsort(axis=1).argsort(axis=1).astype(np.int32)

# ─────────────────────────────────────────────────────────────────────────────
# Per-config metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(data, path_res_L, L, alpha, slicemask, gold_in_pool, base_wrong):
    """
    Compute all metrics for one (L, alpha) config restricted to slicemask.
    Returns dict of scalar metrics.
    """
    cand_ids  = data["cand_ids"]   # [N, M]
    cand_lgt  = data["cand_lgt"]   # [N, M]
    gold      = data["gold"]       # [N]
    base_top1 = data["base_top1"]  # [N]
    M         = data["M"]
    cont_nll  = path_res_L["cont_nll"]    # [N, M]
    has_fut   = path_res_L["has_future"]  # [N]
    N         = len(gold)
    ar        = np.arange(N)

    # Path scores: higher is better
    path_scores = cand_lgt - alpha * cont_nll   # [N, M]

    # Gold index in pool
    gip_mask  = gold_in_pool
    gold_pidx = np.where(gip_mask,
                         (cand_ids == gold[:, None]).argmax(axis=1),
                         np.zeros(N, dtype=np.intp))  # [N]

    # Ranks
    path_rank = ranks_descending(path_scores)    # [N, M]
    base_rank = ranks_descending(cand_lgt)       # [N, M]
    cont_rank = ranks_ascending(cont_nll)        # [N, M]; lower NLL = better = rank 0

    gold_path_rank = path_rank[ar, gold_pidx]   # [N] (meaningful only if gip_mask)
    gold_base_rank = base_rank[ar, gold_pidx]
    gold_cont_rank = cont_rank[ar, gold_pidx]

    # Selected token under path score
    path_sel_idx = path_scores.argmax(axis=1)   # [N]
    path_sel_tok = cand_ids[ar, path_sel_idx]   # [N]
    path_correct = path_sel_tok == gold         # [N]
    base_correct = base_top1 == gold            # [N]

    changed_to_gold  = (~base_correct) & path_correct
    changed_away     = base_correct & (~path_correct)

    # Restrict to slicemask
    sm     = slicemask
    sm_gip = sm & gip_mask
    sm_hf  = sm & has_fut & gip_mask

    def _rate(arr): return float(arr[sm].mean()) if sm.any() else float("nan")
    def _mean_gip(arr): return float(arr[sm_gip].mean()) if sm_gip.any() else float("nan")
    def _mean_hf(arr):  return float(arr[sm_hf].mean())  if sm_hf.any()  else float("nan")

    n    = int(sm.sum())
    n_gip = int(sm_gip.sum())
    n_hf  = int(sm_hf.sum())

    # Gold continuation NLL (oracle)
    gold_cont_nll = cont_nll[ar, gold_pidx]          # [N]
    base_cont_nll = cont_nll[:, 0]                    # [N] — base_top1 is always cand[0]
    sel_cont_nll  = cont_nll[ar, path_sel_idx]        # [N] path-selected candidate NLL

    # Rank histograms (gold_in_pool rows in slice only)
    gpr = gold_path_rank[sm_gip]  # gold path ranks for gip rows in slice
    le_counts = {k: int((gpr < k).sum()) for k in [1, 3, 5, 10]}

    # frac_gold_best_cont: gold has lowest cont_nll among all candidates
    gold_is_best_cont = (gold_cont_rank[sm_gip] == 0) if sm_gip.any() else np.array([])
    # frac_gold_best_path: gold has highest path_score
    gold_is_best_path = (gold_path_rank[sm_gip] == 0) if sm_gip.any() else np.array([])
    # gold cont_nll < base_top1 cont_nll
    gold_better_than_base = (gold_cont_nll < base_cont_nll)

    return {
        "n": n,
        "n_gold_in_pool": n_gip,
        "n_has_future": n_hf,
        # correction metrics
        "base_acc":          _rate(base_correct),
        "path_acc":          _rate(path_correct),
        "acc_gain":          _rate(path_correct) - _rate(base_correct)
                             if sm.any() else float("nan"),
        "changed_to_gold":   _rate(changed_to_gold),
        "changed_away":      _rate(changed_away),
        "net_correction":    _rate(changed_to_gold) - _rate(changed_away)
                             if sm.any() else float("nan"),
        "selected_gold_given_in_pool": _mean_gip(path_sel_tok == gold),
        # rank metrics (gold_in_pool only)
        "gold_path_rank_mean":   _mean_gip(gold_path_rank),
        "gold_path_rank_median": float(np.median(gold_path_rank[sm_gip]))
                                 if sm_gip.any() else float("nan"),
        "gold_base_rank_mean":   _mean_gip(gold_base_rank),
        "gold_cont_rank_mean":   _mean_hf(gold_cont_rank),
        "gold_path_rank_le_1":   le_counts[1] / max(n_gip, 1),
        "gold_path_rank_le_3":   le_counts[3] / max(n_gip, 1),
        "gold_path_rank_le_5":   le_counts[5] / max(n_gip, 1),
        "gold_path_rank_le_10":  le_counts[10] / max(n_gip, 1),
        # continuation NLL diagnostics (oracle — future tokens used)
        "gold_continuation_nll_mean":      _mean_hf(gold_cont_nll),
        "base_top1_continuation_nll_mean": _mean_hf(base_cont_nll),
        "selected_path_cont_nll_mean":     _mean_hf(sel_cont_nll),
        "gold_minus_base_cont_nll":        _mean_hf(gold_cont_nll - base_cont_nll),
        "frac_gold_lower_cont_nll_than_base":
            float(gold_better_than_base[sm_hf].mean()) if sm_hf.any() else float("nan"),
        "frac_gold_best_cont_nll":
            float(gold_is_best_cont.mean()) if len(gold_is_best_cont) else float("nan"),
        "frac_gold_best_combined_path":
            float(gold_is_best_path.mean()) if len(gold_is_best_path) else float("nan"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Example collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_examples(data, path_res_L, L, alpha, gold_in_pool, corpus, max_per=15, seed=42):
    """
    Returns dict of example lists: helps, fails, gold_best_cont, base_wrong_future_disambig.
    """
    N         = len(data["gold"])
    cand_ids  = data["cand_ids"]
    cand_lgt  = data["cand_lgt"]
    gold      = data["gold"]
    base_top1 = data["base_top1"]
    inp_ids   = data["ids"]
    row_ids   = data["row_id"]
    offsets   = data["token_offset"]
    M         = data["M"]
    cont_nll  = path_res_L["cont_nll"]
    has_fut   = path_res_L["has_future"]

    path_scores = cand_lgt - alpha * cont_nll
    ar          = np.arange(N)
    path_sel    = cand_ids[ar, path_scores.argmax(axis=1)]
    gold_pidx   = np.where(gold_in_pool,
                            (cand_ids == gold[:, None]).argmax(axis=1),
                            np.zeros(N, dtype=np.intp))
    gold_cont_nll_r = cont_nll[ar, gold_pidx]
    base_cont_nll_r = cont_nll[:, 0]

    rng  = random.Random(seed)
    idxs = list(range(N)); rng.shuffle(idxs)

    buckets = {"helps": [], "fails": [], "gold_best_cont": [], "base_wrong_future_disambig": []}

    for i in idxs:
        base_ok  = bool(base_top1[i] == gold[i])
        gip      = bool(gold_in_pool[i])
        hf       = bool(has_fut[i])
        psel     = int(path_sel[i])
        path_ok  = (psel == int(gold[i]))

        # Build candidate detail list
        base_rank_arr = ranks_descending(cand_lgt[i:i+1])[0]  # [M]
        cont_rank_arr = ranks_ascending(cont_nll[i:i+1])[0]
        path_rank_arr = ranks_descending(path_scores[i:i+1])[0]
        cands_detail = []
        for j in range(min(M, 16)):
            cands_detail.append({
                "tok_id": int(cand_ids[i, j]),
                "tok_str": _decode_tok(int(cand_ids[i, j])),
                "base_logit": float(cand_lgt[i, j]),
                "base_rank": int(base_rank_arr[j]),
                "cont_nll_L": float(cont_nll[i, j]),
                "cont_rank": int(cont_rank_arr[j]),
                "path_score": float(path_scores[i, j]),
                "path_rank": int(path_rank_arr[j]),
                "is_gold": int(cand_ids[i, j]) == int(gold[i]),
                "is_base_top1": j == 0,
                "is_path_selected": int(cand_ids[i, j]) == psel,
            })

        # Future continuation text
        off = int(offsets[i])
        future_ids = corpus[off + 1: off + max(8, L)].tolist() if hf else []

        ex = {
            "row_id": int(row_ids[i]),
            "token_offset": off,
            "ctx_tail": inp_ids[i, -64:].tolist(),
            "gold_id": int(gold[i]),
            "gold_str": _decode_tok(int(gold[i])),
            "base_top1_id": int(base_top1[i]),
            "base_top1_str": _decode_tok(int(base_top1[i])),
            "path_sel_id": psel,
            "path_sel_str": _decode_tok(psel),
            "path_len": L,
            "alpha": alpha,
            "gold_cont_nll": float(gold_cont_nll_r[i]) if gip else float("nan"),
            "base_cont_nll": float(base_cont_nll_r[i]),
            "gold_minus_base_cont_nll": float(gold_cont_nll_r[i] - base_cont_nll_r[i])
                                         if gip else float("nan"),
            "has_future": hf,
            "future_str": _decode_ids(future_ids) if future_ids else "(unavailable)",
            "candidates": cands_detail,
        }

        if len(buckets["helps"]) < max_per and not base_ok and gip and path_ok and hf:
            buckets["helps"].append(ex)
        if len(buckets["fails"]) < max_per and base_ok and not path_ok and hf:
            buckets["fails"].append(ex)
        if len(buckets["gold_best_cont"]) < max_per and gip and hf \
                and float(gold_cont_nll_r[i]) < float(base_cont_nll_r[i]):
            buckets["gold_best_cont"].append(ex)
        if len(buckets["base_wrong_future_disambig"]) < max_per and not base_ok and gip and hf \
                and float(gold_cont_nll_r[i]) < float(base_cont_nll_r[i]):
            buckets["base_wrong_future_disambig"].append(ex)
        if all(len(v) >= max_per for v in buckets.values()):
            break

    return buckets

def _cand_table_md(cands):
    hdr = ("| rank | tok | base_lgt | base_rnk | cont_nll | cont_rnk | "
           "path_score | path_rnk | gold? | base? | sel? |")
    sep = ("|------|-----|----------|----------|----------|----------|"
           "------------|----------|-------|-------|------|")
    rows = [hdr, sep]
    for c in cands:
        rows.append(
            f"| {c['path_rank']} "
            f"| `{c['tok_str'][:12]:<12}` "
            f"| {c['base_logit']:7.3f} | {c['base_rank']} "
            f"| {c['cont_nll_L']:.4f} | {c['cont_rank']} "
            f"| {c['path_score']:7.3f} | {c['path_rank']} "
            f"| {'✓' if c['is_gold'] else ''} "
            f"| {'✓' if c['is_base_top1'] else ''} "
            f"| {'✓' if c['is_path_selected'] else ''} |")
    return "\n".join(rows)

def write_examples_md(exs, title, path):
    lines = [f"# ⚠ ORACLE_DIAGNOSTIC: {title}\n",
             "_Future gold tokens are used. Not real validation metrics._\n\n"]
    for idx, ex in enumerate(exs):
        ctx_str = _decode_ids(ex["ctx_tail"])
        lines.append(f"### Example {idx+1}  (row {ex['row_id']}  offset={ex['token_offset']})\n")
        lines.append(
            f"**Gold:** `{ex['gold_str']}`  "
            f"**Base:** `{ex['base_top1_str']}`  "
            f"**PathSel:** `{ex['path_sel_str']}`  "
            f"path_len={ex['path_len']}  alpha={ex['alpha']}\n")
        lines.append(
            f"gold_cont_nll={ex['gold_cont_nll']:.4f}  "
            f"base_cont_nll={ex['base_cont_nll']:.4f}  "
            f"Δ(gold−base)={ex['gold_minus_base_cont_nll']:.4f}\n")
        lines.append(f"\n**Future continuation (gold):** `{ex['future_str'][:200]}`\n")
        lines.append(f"\n**Context tail:**\n```\n{ctx_str[-300:]}\n```\n")
        lines.append("\n**Candidates:**\n" + _cand_table_md(ex["candidates"]) + "\n\n---\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"[md] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# CSV writing
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows, fields, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {path}")

def _f(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(all_metrics, path_lens, alpha_grid, slice_names, out_path, args, N_total):
    def _best_config(sn, metric="frac_gold_best_combined_path"):
        best_val = -1e9; best_key = None
        for (L, alpha, sn2), m in all_metrics.items():
            if sn2 != sn:
                continue
            v = m.get(metric, float("nan"))
            if not math.isnan(v) and v > best_val:
                best_val = v; best_key = (L, alpha)
        return best_key, best_val

    lines = [
        "# ⚠ ORACLE_DIAGNOSTIC: Multi-Token Path Oracle V1\n\n",
        "**WARNING: This report uses future gold continuation tokens. "
        "It is NOT deployable validation. It is an existence test for path-memory signal.**\n\n",
        f"val_dir: `{args.val_dir}`  eval_filter: `{args.eval_filter}`  "
        f"N_eval: {N_total:,}  M={args.candidate_pool_size}\n\n",
    ]

    def _g(sn, L, alpha, key):
        return all_metrics.get((L, alpha, sn), {}).get(key, float("nan"))

    # Q1: Does future continuation make gold easier to identify?
    lines.append("## Q1. Does future continuation make the gold candidate easier to identify?\n")
    bk, bv = _best_config("bucketA_confuser", "frac_gold_best_combined_path")
    bk_base, bv_base = _best_config("bucketA_confuser", "gold_path_rank_le_1")
    lines.append(
        f"Best frac_gold_best_combined_path on bucketA: {_f(bv)} "
        f"at L={bk[0] if bk else '?'} α={bk[1] if bk else '?'}\n")
    base_frac = _g("bucketA_confuser", 1, alpha_grid[0], "gold_path_rank_le_1")
    lines.append(f"Baseline (L=1, path_rank_le_1): {_f(base_frac)}\n")
    ans1 = "**YES**" if bv > 0.3 else ("**WEAK**" if bv > 0.1 else "**NO**")
    lines.append(f"Verdict: {ans1}\n\n")

    # Q2: Which path length works best?
    lines.append("## Q2. Which path length works best?\n")
    lines.append("| L | best_alpha | frac_gold_best_combined (bucketA) | gold_path_rank_le_1 |\n")
    lines.append("|---|-----------|----------------------------------|---------------------|\n")
    for L in path_lens:
        best_a = max(alpha_grid,
                     key=lambda a: _g("bucketA_confuser", L, a, "frac_gold_best_combined_path"))
        lines.append(
            f"| {L} | {best_a} "
            f"| {_f(_g('bucketA_confuser', L, best_a, 'frac_gold_best_combined_path'))} "
            f"| {_f(_g('bucketA_confuser', L, best_a, 'gold_path_rank_le_1'))} |\n")
    lines.append("\n")

    # Q3: Does path scoring improve selected_gold_given_in_pool above base?
    lines.append("## Q3. Does path scoring improve selected_gold_given_in_pool vs base?\n")
    bk3, _ = _best_config("bucketA_confuser", "selected_gold_given_in_pool")
    if bk3:
        sip_path = _g("bucketA_confuser", bk3[0], bk3[1], "selected_gold_given_in_pool")
        sip_base = _g("bucketA_confuser", 1, alpha_grid[0], "base_acc")
        lines.append(
            f"Best sip (L={bk3[0]}, α={bk3[1]}): {_f(sip_path)}  "
            f"vs base_acc: {_f(sip_base)}\n")
        ans3 = "**YES**" if (not math.isnan(sip_path) and not math.isnan(sip_base)
                             and sip_path > sip_base + 0.01) else "**Marginal or NO**"
        lines.append(f"Verdict: {ans3}\n\n")

    # Q4: Does path help same-region confusers?
    lines.append("## Q4. Does path help same-region confusers?\n")
    bk4, bv4 = _best_config("same_region_confuser", "frac_gold_best_combined_path")
    lines.append(
        f"Best frac_gold_best_combined (same_region): {_f(bv4)} "
        f"at L={bk4[0] if bk4 else '?'} α={bk4[1] if bk4 else '?'}\n")
    ans4 = "**YES**" if bv4 > 0.3 else "**NO/Marginal**"
    lines.append(f"Verdict: {ans4}\n\n")

    # Q5: Does path help gold_not_in_context?
    lines.append("## Q5. Does path help gold_not_in_context rows (where pointer cannot)?\n")
    bk5, bv5 = _best_config("gold_not_in_context", "frac_gold_best_combined_path")
    lines.append(
        f"Best frac_gold_best_combined (gold_not_in_context): {_f(bv5)} "
        f"at L={bk5[0] if bk5 else '?'} α={bk5[1] if bk5 else '?'}\n")
    ans5 = "**YES**" if bv5 > 0.25 else "**NO/Marginal**"
    lines.append(f"Verdict: {ans5}\n\n")

    # Q6: Which token classes benefit most?
    lines.append("## Q6. Which token classes benefit most?\n")
    type_slices = ["bpe_fragment_like", "punctuation_like", "number_like", "word_choice_like"]
    lines.append("| slice | best_L | best_alpha | frac_gold_best_combined | gold_minus_base_nll |\n")
    lines.append("|-------|--------|-----------|------------------------|---------------------|\n")
    for sn in type_slices:
        bk6, bv6 = _best_config(sn, "frac_gold_best_combined_path")
        nll_diff = _g(sn, bk6[0], bk6[1], "gold_minus_base_cont_nll") if bk6 else float("nan")
        lines.append(
            f"| {sn} | {bk6[0] if bk6 else '?'} | {bk6[1] if bk6 else '?'} "
            f"| {_f(bv6)} | {_f(nll_diff)} |\n")
    lines.append("\n")

    # Q7: Is oracle upper bound strong enough?
    lines.append("## Q7. Is the oracle upper bound strong enough to justify a non-oracle path model?\n")
    bk7, bv7 = _best_config("all", "frac_gold_best_cont_nll")
    bk7p, bv7p = _best_config("bucketA_confuser", "frac_gold_best_combined_path")
    lines.append(
        f"frac_gold_best_cont_nll (all rows): {_f(bv7)} at L={bk7[0] if bk7 else '?'}\n")
    lines.append(
        f"frac_gold_best_combined_path (bucketA): {_f(bv7p)} at L={bk7p[0] if bk7p else '?'}\n")
    if bv7 > 0.4:
        ans7 = "**YES** — gold produces best continuation in >40% of cases. Path signal is real."
    elif bv7 > 0.2:
        ans7 = "**MARGINAL** — some path signal, but weak. May require longer contexts."
    else:
        ans7 = "**NO** — gold does not reliably produce best continuation. Path signal absent."
    lines.append(f"Verdict: {ans7}\n\n")

    # Q8: Should multi-token path memory become expert #3?
    lines.append("## Q8. Should multi-token path memory become memory expert #3?\n")
    evidence_for, evidence_against = [], []
    if not math.isnan(bv7) and bv7 > 0.35:
        evidence_for.append(f"frac_gold_best_cont_nll={_f(bv7)} — continuation signal exists")
    elif not math.isnan(bv7):
        evidence_against.append(f"frac_gold_best_cont_nll={_f(bv7)} — continuation signal weak")
    if not math.isnan(bv7p) and bv7p > 0.35:
        evidence_for.append(f"frac_gold_best_combined={_f(bv7p)} on bucketA — path selects gold often")
    best_L = min(path_lens[1:], key=lambda L_:
                 -max((_g("bucketA_confuser", L_, a, "frac_gold_best_combined_path")
                       for a in alpha_grid), default=-1e9))
    if best_L in [2, 4]:
        evidence_for.append(f"Best signal at L={best_L} — practical non-oracle path scorer is feasible")
    elif best_L == 8:
        evidence_against.append("Best signal only at L=8 — longer context may be expensive")

    if evidence_for:
        lines.append("**Evidence FOR:**\n")
        for r in evidence_for:
            lines.append(f"- {r}\n")
    if evidence_against:
        lines.append("\n**Evidence AGAINST:**\n")
        for r in evidence_against:
            lines.append(f"- {r}\n")

    strong = len(evidence_for) >= 2 and (not math.isnan(bv7)) and bv7 > 0.3
    if strong:
        verdict = ("**Conditional YES** — path memory signal is meaningful. "
                   "Implement a candidate continuation scorer as a non-oracle expert.")
    elif evidence_for:
        verdict = ("**Marginal** — some signal, but weak. "
                   "Consider longer contexts or higher-capacity models before committing.")
    else:
        verdict = ("**NO** — path memory is not a major error-resolution mechanism for this dataset. "
                   "Focus on other experts.")
    lines.append(f"\n**Verdict:** {verdict}\n")
    lines.append("\n---\n*⚠ ORACLE_DIAGNOSTIC — future gold tokens used. "
                 "Not real validation metrics. Not deployable.*\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"[report] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="ORACLE_DIAGNOSTIC: Multi-Token Path Oracle V1. "
                    "Uses future gold tokens — NOT real eval.")
    p.add_argument("--val_dir",             required=True)
    p.add_argument("--small_ckpt",          required=True)
    p.add_argument("--token_to_region",     default=None)
    p.add_argument("--super_map",           default=None)
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--corpus_path",         default=None,
                   help="Path to val_tokens.npy for future token lookup")
    p.add_argument("--eval_filter",         default="bucketA_confuser",
                   choices=["all", "bucketA_confuser", "same_region_confuser",
                            "pointer_should_not_help"])
    p.add_argument("--candidate_pool_size", type=int, default=16)
    p.add_argument("--path_lens",           default="1,2,4,8")
    p.add_argument("--alpha_grid",          default="0.25,0.5,1.0,2.0")
    p.add_argument("--max_rows",            type=int, default=5000)
    p.add_argument("--batch_size",          type=int, default=16,
                   help="Val rows per outer batch (controls RAM via B×M sequences per forward pass)")
    p.add_argument("--memory_len",          type=int, default=128)
    p.add_argument("--max_examples",        type=int, default=50)
    p.add_argument("--amp",                 action="store_true",
                   help="Use AMP (fp16) for backbone forward passes")
    p.add_argument("--device",              default="cuda")
    p.add_argument("--seed",                type=int, default=42)
    return p, p.parse_args()


def main():
    p, args = _parse()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    path_lens  = [int(x) for x in args.path_lens.split(",")]
    alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
    device     = torch.device(args.device if torch.cuda.is_available()
                              or args.device == "cpu" else "cpu")
    use_amp    = args.amp and device.type == "cuda"

    print("=" * 65)
    print(" ⚠  ORACLE_DIAGNOSTIC: Multi-Token Path Oracle V1")
    print("    Uses FUTURE gold tokens. NOT real validation metrics.")
    print(f" val_dir:    {args.val_dir}")
    print(f" small_ckpt: {args.small_ckpt}")
    print(f" output_dir: {args.output_dir}")
    print(f" eval_filter:{args.eval_filter}  max_rows:{args.max_rows}")
    print(f" path_lens:  {path_lens}  alpha_grid:{alpha_grid}")
    print(f" M={args.candidate_pool_size}  batch_size={args.batch_size}  amp={use_amp}")
    print("=" * 65)

    for d in [args.val_dir]:
        if not os.path.isdir(d):
            print(f"ERROR: directory not found: {d}"); sys.exit(1)
    if not os.path.isfile(args.small_ckpt):
        print(f"ERROR: checkpoint not found: {args.small_ckpt}"); sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    backbone, emb_w, seq_len, vocab_size, ckpt_cfg = load_model(args.small_ckpt, device)

    # ── Load val corpus for future tokens ─────────────────────────────────────
    print("\n[corpus] Loading val corpus for future token reconstruction...")
    corpus = load_val_corpus(args.val_dir, args.corpus_path)

    # ── Load val shards ───────────────────────────────────────────────────────
    print("\n[shards] Loading val shards...")
    data = load_val_shards(args.val_dir, args.candidate_pool_size, max_rows=None)
    N_all = len(data["gold"])

    # ── Verify corpus alignment ───────────────────────────────────────────────
    print("\n[align] Verifying corpus alignment...")
    verify_corpus_alignment(data, corpus)

    # ── Region maps ───────────────────────────────────────────────────────────
    print("\n[maps] Loading region maps...")
    if args.token_to_region and os.path.isfile(args.token_to_region):
        tok_arr, reg_arr, unk_r, unk_s, sr = load_maps(args.token_to_region, args.super_map)
        print(f"  sr_enabled={sr}")
    else:
        print("  [WARN] token_to_region not found — region slices disabled")
        V = int(data["topk"].max()) + 2
        tok_arr = np.zeros(V, dtype=np.int32)
        reg_arr = np.zeros(2, dtype=np.int32)
        unk_r = unk_s = 0; sr = False

    # ── Build slices (on full dataset) ────────────────────────────────────────
    print("\n[slices] Building slices...")
    slices_full, gold_in_pool_full, base_wrong_full, base_margin_full = build_slices(
        data, tok_arr, reg_arr, unk_r, unk_s, sr, args.memory_len)

    # ── Apply eval_filter ─────────────────────────────────────────────────────
    if args.eval_filter == "bucketA_confuser":
        filter_mask = base_wrong_full & gold_in_pool_full
    elif args.eval_filter == "same_region_confuser":
        filter_mask = slices_full["same_region_confuser"]
    elif args.eval_filter == "pointer_should_not_help":
        filter_mask = slices_full["gold_not_in_context"]
    else:  # all
        filter_mask = np.ones(N_all, dtype=bool)

    eval_idxs = np.where(filter_mask)[0]
    if args.max_rows and len(eval_idxs) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        eval_idxs = rng.choice(eval_idxs, size=args.max_rows, replace=False)
        eval_idxs.sort()
    N_eval = len(eval_idxs)
    print(f"\n[filter] eval_filter={args.eval_filter}  N_eval={N_eval:,}")
    if N_eval == 0:
        print("ERROR: No rows pass eval_filter. Check --eval_filter and data.")
        sys.exit(1)

    # ── Subset data to eval rows ──────────────────────────────────────────────
    def _sub(arr):
        return arr[eval_idxs]
    sub_data = {k: _sub(v) for k, v in data.items() if isinstance(v, np.ndarray)}
    sub_data["M"] = data["M"]

    # Rebuild slices on subset
    slices_sub, gold_in_pool_sub, base_wrong_sub, _ = build_slices(
        sub_data, tok_arr, reg_arr, unk_r, unk_s, sr, args.memory_len)

    # ── Compute path results (ORACLE — reads future tokens) ───────────────────
    path_res = compute_path_results(
        backbone, emb_w, seq_len, sub_data, corpus,
        path_lens, args.batch_size, device, use_amp)

    # ── Evaluate all (L, alpha, slice) combos ─────────────────────────────────
    print("\n[eval] Computing metrics for all configs...")
    all_metrics = {}
    slice_names = list(slices_sub.keys())
    summary_rows = []
    _MKEYS = [
        "n", "n_gold_in_pool", "n_has_future",
        "base_acc", "path_acc", "acc_gain", "net_correction",
        "changed_to_gold", "changed_away",
        "selected_gold_given_in_pool",
        "gold_path_rank_mean", "gold_path_rank_median",
        "gold_base_rank_mean", "gold_cont_rank_mean",
        "gold_path_rank_le_1", "gold_path_rank_le_3",
        "gold_path_rank_le_5", "gold_path_rank_le_10",
        "gold_continuation_nll_mean", "base_top1_continuation_nll_mean",
        "selected_path_cont_nll_mean", "gold_minus_base_cont_nll",
        "frac_gold_lower_cont_nll_than_base",
        "frac_gold_best_cont_nll", "frac_gold_best_combined_path",
    ]
    for L in path_lens:
        for alpha in alpha_grid:
            for sn in slice_names:
                smask = slices_sub[sn]
                m = compute_metrics(sub_data, path_res[L], L, alpha,
                                    smask, gold_in_pool_sub, base_wrong_sub)
                all_metrics[(L, alpha, sn)] = m
                row = {"L": L, "alpha": alpha, "slice": sn}
                for k in _MKEYS:
                    v = m.get(k, "")
                    row[k] = f"{v:.6f}" if isinstance(v, float) else str(v)
                summary_rows.append(row)

    write_csv(summary_rows, ["L", "alpha", "slice"] + _MKEYS,
              os.path.join(args.output_dir, "path_oracle_summary.csv"))

    # path_rank_stats.csv — rank distributions for best L across slices
    rank_rows = []
    for sn in slice_names:
        best_L   = path_lens[0]; best_a = alpha_grid[0]; best_val = -1e9
        for L in path_lens:
            for a in alpha_grid:
                v = all_metrics.get((L, a, sn), {}).get("frac_gold_best_combined_path", float("nan"))
                if not math.isnan(v) and v > best_val:
                    best_val = v; best_L = L; best_a = a
        m = all_metrics.get((best_L, best_a, sn), {})
        row = {"slice": sn, "best_L": best_L, "best_alpha": best_a}
        for k in _MKEYS:
            v = m.get(k, "")
            row[k] = f"{v:.6f}" if isinstance(v, float) else str(v)
        rank_rows.append(row)
    write_csv(rank_rows, ["slice", "best_L", "best_alpha"] + _MKEYS,
              os.path.join(args.output_dir, "path_rank_stats.csv"))

    # path_oracle_by_slice.csv — best (L, alpha) per slice
    by_slice_rows = []
    for sn in slice_names:
        best_L = best_a = None; best_val = -1e9
        for L in path_lens:
            for a in alpha_grid:
                v = all_metrics.get((L, a, sn), {}).get("net_correction", float("nan"))
                if not math.isnan(v) and v > best_val:
                    best_val = v; best_L = L; best_a = a
        if best_L is not None:
            m = all_metrics[(best_L, best_a, sn)]
            row = {"slice": sn, "best_L": best_L, "best_alpha": best_a}
            for k in _MKEYS:
                v = m.get(k, "")
                row[k] = f"{v:.6f}" if isinstance(v, float) else str(v)
            by_slice_rows.append(row)
    write_csv(by_slice_rows, ["slice", "best_L", "best_alpha"] + _MKEYS,
              os.path.join(args.output_dir, "path_oracle_by_slice.csv"))

    # ── Examples ──────────────────────────────────────────────────────────────
    print("\n[examples] Collecting examples...")
    # Use best L and alpha on bucketA for examples
    bk_ex = None; bv_ex = -1e9
    for L in path_lens:
        for a in alpha_grid:
            v = all_metrics.get((L, a, "bucketA_confuser"), {}).get(
                "frac_gold_best_combined_path", float("nan"))
            if not math.isnan(v) and v > bv_ex:
                bv_ex = v; bk_ex = (L, a)
    ex_L, ex_alpha = bk_ex if bk_ex else (path_lens[-1], alpha_grid[1])
    max_per = max(args.max_examples // 4, 5)
    buckets = collect_examples(sub_data, path_res[ex_L], ex_L, ex_alpha,
                               gold_in_pool_sub, corpus, max_per, args.seed)
    write_examples_md(buckets["helps"],
                      "Path HELPS — base wrong, gold in pool, path selects gold",
                      os.path.join(args.output_dir, "examples_path_helps.md"))
    write_examples_md(buckets["fails"],
                      "Path FAILS — base correct, path selects wrong",
                      os.path.join(args.output_dir, "examples_path_fails.md"))
    write_examples_md(buckets["gold_best_cont"],
                      "Gold has Best Continuation NLL",
                      os.path.join(args.output_dir, "examples_path_gold_best_continuation.md"))
    write_examples_md(buckets["base_wrong_future_disambig"],
                      "Base Wrong but Future Disambiguates — gold has lower cont_nll than base",
                      os.path.join(args.output_dir,
                                   "examples_path_base_wrong_but_future_disambiguates.md"))

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(all_metrics, path_lens, alpha_grid, slice_names,
                 os.path.join(args.output_dir, "report.md"), args, N_eval)

    # ── Config dump ───────────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({
            "ORACLE_DIAGNOSTIC": True,
            "WARNING": "Uses future gold tokens. NOT real validation metrics.",
            **vars(args),
            "N_eval": N_eval, "M": data["M"],
            "path_lens": path_lens, "alpha_grid": alpha_grid,
            "seq_len": seq_len,
        }, f, indent=2)
    print(f"\n[done] Outputs in: {args.output_dir}")
    print("  ⚠ ORACLE_DIAGNOSTIC — do not use metrics as real validation results.")


if __name__ == "__main__":
    main()
