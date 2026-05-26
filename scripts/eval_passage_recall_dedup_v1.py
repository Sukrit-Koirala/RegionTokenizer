#!/usr/bin/env python3
"""
eval_passage_recall_dedup_v1.py — Passage Recall Dedup Diagnostic V1.

Test whether Type-A/confuser errors are recoverable from near-exact or
source-like train passages.

NON-PARAMETRIC MEMORY DIAGNOSTIC.
  - No training.  No model changes.
  - Val gold used ONLY for metrics AFTER retrieval and scoring.
  - Retrieval keys and candidate votes come from train data only.

For each val row:
  1. Retrieve top-K similar train rows by cosine similarity.
  2. Apply dedup filter (suffix16/suffix32/jaccard03/jaccard05) to remove
     near-duplicate train contexts.
  3. Score each candidate by how often it appears in retrieved continuations.
  4. Evaluate base / passage / combined policies.

Answers:
  Q1. Does train-only passage recall recover gold?
  Q2. Does it help where pointer/fuzzy fail?
  Q3. Does it help exact numbers/entities?
  Q4. Does performance survive dedup/overlap filtering?
  Q5. Genuine reusable passage memory or near-duplicate memorisation?
  Q6. Should passage recall become a memory expert?

Outputs (runs/passage_memory_v1/passage_recall_dedup_v1/eval/):
  passage_recall_report.md
  passage_policy_grid.csv
  passage_slice_metrics.csv
  passage_dedup_stats.csv
  passage_rank_stats.csv
  examples_passage_helps.md
  examples_passage_hurts.md
  examples_dedup_removes_gold.md
  config.json

Usage:
  python scripts/eval_passage_recall_dedup_v1.py \\
    --val_dir  ... --index_dir ... \\
    --token_to_region ... --super_map ... \\
    --output_dir ... \\
    --candidate_pool_size 32 \\
    --retrieval_k_grid 8,16,32,64 \\
    --lambda_grid 0.0,0.25,0.5,1.0,2.0,4.0 \\
    --overlap_filters no_filter,suffix16,suffix32,jaccard03,jaccard05 \\
    --max_examples 50 --seed 42
"""

import argparse
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

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

_EPS = 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer (optional, for example rendering)
# ─────────────────────────────────────────────────────────────────────────────

_enc = None
try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
except Exception:
    pass


def _decode_tok(tid):
    if _enc is not None:
        try: return repr(_enc.decode([int(tid)]))[1:-1]
        except Exception: pass
    return f"<{tid}>"


def _decode_ids(ids):
    if _enc is not None:
        try: return _enc.decode([int(i) for i in ids if 0 <= int(i) < 50257])
        except Exception: pass
    return " ".join(_decode_tok(i) for i in ids)


# ─────────────────────────────────────────────────────────────────────────────
# Shard field aliases
# ─────────────────────────────────────────────────────────────────────────────

_TOPK_ALIASES  = ["base_topk_ids",    "base_topk",     "topk_ids"]
_LGT_ALIASES   = ["base_topk_logits", "base_topk_lgt", "topk_lgt", "topk_logits"]
_GOLD_ALIASES  = ["gold_token",       "gold",          "labels"]
_IDS_ALIASES   = ["input_ids"]
_HCTX_ALIASES  = ["h_ctx",  "h_context", "hidden_ctx"]
_HRAW_ALIASES  = ["h_raw",  "h_backbone","hidden_raw"]
_ROWID_ALIASES = ["row_id", "row_ids"]


def _get(shard, aliases, required=True):
    for a in aliases:
        if a in shard:
            return shard[a]
    if required:
        raise KeyError(f"Missing field. Tried: {aliases}. Have: {list(shard.keys())}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# L2 normalise
# ─────────────────────────────────────────────────────────────────────────────

def l2_norm_rows(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(n, _EPS)


# ─────────────────────────────────────────────────────────────────────────────
# Val shard loading
# ─────────────────────────────────────────────────────────────────────────────

def load_val_shards(val_dir, cand_pool_size, max_rows=None):
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
                print(f"  {k}: shape={getattr(v,'shape',None)}  "
                      f"dtype={getattr(v,'dtype',type(v).__name__)}")
            first = False
        topk = _get(sh, _TOPK_ALIASES).long()
        lgt  = _get(sh, _LGT_ALIASES).float()
        gold = _get(sh, _GOLD_ALIASES).long()
        ids  = _get(sh, _IDS_ALIASES).long()
        rids = _get(sh, _ROWID_ALIASES, required=False)

        B, K = topk.shape
        if rids is None:
            rids = torch.arange(total, total + B)
        if K < cand_pool_size:
            topk = torch.cat([topk, torch.zeros(B, cand_pool_size - K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, cand_pool_size - K), float("nan"))], 1)
        elif K > cand_pool_size:
            topk, lgt = topk[:, :cand_pool_size], lgt[:, :cand_pool_size]

        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, ids = topk[:keep], lgt[:keep], gold[:keep], ids[:keep]
            rids = rids[:keep]
            B = keep

        bufs["topk"].append(topk.numpy())
        bufs["lgt"].append(lgt.numpy())
        bufs["gold"].append(gold.numpy())
        bufs["ids"].append(ids.numpy())
        bufs["row_ids"].append(rids.numpy())
        for alias_list, key in [(_HCTX_ALIASES, "h_ctx"), (_HRAW_ALIASES, "h_raw")]:
            h = _get(sh, alias_list, required=False)
            if h is not None:
                bufs[key].append(h.float().numpy()[:B])
        total += B

    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    data["lgt"] = np.where(np.isfinite(data["lgt"]), data["lgt"], -1e9)
    print(f"[shards] {total:,} val rows  pool={cand_pool_size}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Map loading
# ─────────────────────────────────────────────────────────────────────────────

def load_maps(t2r_path, super_path=None):
    with open(t2r_path) as f:
        raw = json.load(f)
    t2r = ({i: v for i, v in enumerate(raw) if v is not None}
           if isinstance(raw, list)
           else {int(k): v for k, v in raw.items()})
    unk_r = int(max(t2r.values())) + 1 if t2r else 1
    r2s = {}; unk_s = 1; sr = False
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f:
            raw2 = json.load(f)
        r2s = ({int(k): v for k, v in raw2.items()}
               if not isinstance(raw2, list)
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
        if 0 <= r < R:
            reg_arr[int(r)] = int(s)
    return tok_arr, reg_arr, unk_r, unk_s, sr


# ─────────────────────────────────────────────────────────────────────────────
# Index loading
# ─────────────────────────────────────────────────────────────────────────────

def load_index(index_dir, requested_modes, has_hctx, has_hraw):
    cfg_path   = os.path.join(index_dir, "config.json")
    stats_path = os.path.join(index_dir, "build_stats.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"index config.json not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    N              = int(cfg["num_train_rows"])
    h_dim          = cfg.get("h_dim")
    query_len      = int(cfg.get("query_len", 128))
    cont_len       = int(cfg.get("continuation_len", 8))
    feature_dim    = int(cfg.get("feature_dim", 8192))

    modes_built = cfg.get("modes_built", [])
    if not modes_built and os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        modes_built = stats.get("modes_built", [])
        h_dim = h_dim or stats.get("h_dim")
    if not modes_built:
        raise RuntimeError(
            f"modes_built is empty. Check build_stats.json={stats_path}")

    print(f"[index] N={N:,}  h_dim={h_dim}  cont_len={cont_len}  "
          f"query_len={query_len}  modes_built={modes_built}")

    # Load scalar arrays (small — all into RAM)
    gold_arr  = np.load(os.path.join(index_dir, "gold_token.npy"))
    cont_arr  = np.load(os.path.join(index_dir, "continuations.npy"))
    rowid_arr = np.load(os.path.join(index_dir, "row_id.npy"))
    print(f"[index] gold_arr={gold_arr.shape}  cont_arr={cont_arr.shape}")

    # Load train_prefixes (for dedup) — into RAM if possible
    pref_path = os.path.join(index_dir, "train_prefixes.mmap")
    if os.path.isfile(pref_path):
        pref_mmap = np.memmap(pref_path, dtype=np.int32, mode="r",
                              shape=(N, query_len))
        t0 = time.time()
        print(f"[index] loading train_prefixes into RAM "
              f"({N * query_len * 4 / 1e6:.0f} MB)...")
        train_prefs = np.array(pref_mmap)     # load into RAM
        print(f"  done in {time.time()-t0:.1f}s")
    else:
        print("[WARN] train_prefixes.mmap not found — dedup filters disabled")
        train_prefs = None

    # Load key memmaps
    keys = {}
    for m in modes_built:
        if m not in requested_modes:
            continue
        if m == "h_ctx" and not has_hctx:
            print(f"  [WARN] {m}: in index but val shards lack h_ctx — skip"); continue
        if m == "h_raw" and not has_hraw:
            print(f"  [WARN] {m}: in index but val shards lack h_raw — skip"); continue
        D = h_dim if m in ("h_ctx", "h_raw") else feature_dim
        if D is None:
            print(f"  [WARN] unknown dim for {m} — skip"); continue
        mpath = os.path.join(index_dir, f"{m}_keys.mmap")
        if not os.path.isfile(mpath):
            print(f"  [WARN] {mpath} not found — skip"); continue
        keys[m] = np.memmap(mpath, dtype=np.float16, mode="r", shape=(N, int(D)))
        print(f"  [OK] {m}_keys  [{N}, {D}]")

    if not keys:
        raise RuntimeError(
            f"No index keys loaded. requested_modes={requested_modes}  "
            f"modes_built={modes_built}  val_hctx={has_hctx}  val_hraw={has_hraw}")

    return keys, gold_arr, cont_arr, train_prefs, rowid_arr, N, h_dim, query_len, cont_len, feature_dim, cfg


# ─────────────────────────────────────────────────────────────────────────────
# BOW query builder (val side — matches build-time recency_bow)
# ─────────────────────────────────────────────────────────────────────────────

def build_recency_bow_batch(ids_batch, query_len, feature_dim):
    B, seq_len = ids_batch.shape
    T    = min(query_len, seq_len)
    tail = ids_batch[:, -T:].astype(np.int64)
    dists   = np.arange(T - 1, -1, -1, dtype=np.float32)
    weights = np.exp(-dists / max(T / 4.0, 1.0))
    bins    = tail % feature_dim
    vecs    = np.zeros((B, feature_dim), dtype=np.float32)
    row_idx = np.repeat(np.arange(B, dtype=np.int64), T)
    col_idx = bins.ravel()
    w_flat  = np.tile(weights, B)
    np.add.at(vecs, (row_idx, col_idx), w_flat)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, _EPS)
    return vecs.astype(np.float32)


def build_shingle5gram_batch(ids_batch, query_len, feature_dim):
    B, seq_len = ids_batch.shape
    T    = min(query_len, seq_len)
    tail = ids_batch[:, -T:].astype(np.int64)
    n_sh = T - 4
    vecs = np.zeros((B, feature_dim), dtype=np.float32)
    if n_sh <= 0:
        return vecs
    offsets = np.arange(5, dtype=np.int64)
    starts  = np.arange(n_sh, dtype=np.int64)
    idx_mat = starts[:, None] + offsets[None, :]
    shingles = tail[:, idx_mat]
    p       = np.array([1, 31, 961, 29791, 923521], dtype=np.int64)
    hashes  = (shingles * p[None, None, :]).sum(axis=2) % feature_dim
    hashes  = np.abs(hashes).astype(np.int64)
    row_idx = np.repeat(np.arange(B, dtype=np.int64), n_sh)
    col_idx = hashes.ravel()
    np.add.at(vecs, (row_idx, col_idx), 1.0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, _EPS)
    return vecs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Batched cosine retrieval  (O(N_train) reads — same as fuzzy eval)
# ─────────────────────────────────────────────────────────────────────────────

def batch_retrieve_all(query_mat_f32, train_mmap, train_chunk_size, top_k,
                       val_batch_size=4096):
    """
    Scans train memmap in train_chunk_size chunks.  Reads train data exactly
    once total.  Returns indices [N_val, top_k] and sims [N_val, top_k] sorted
    descending.
    """
    N_val   = query_mat_f32.shape[0]
    N_train = train_mmap.shape[0]
    best_sims = np.full((N_val, top_k), -2.0, dtype=np.float32)
    best_idxs = np.zeros((N_val, top_k), dtype=np.int64)

    ptr = 0
    while ptr < N_train:
        end    = min(ptr + train_chunk_size, N_train)
        C      = end - ptr
        chunk  = train_mmap[ptr:end].astype(np.float32)
        k_this = min(top_k, C)

        vptr = 0
        while vptr < N_val:
            vend = min(vptr + val_batch_size, N_val)
            B    = vend - vptr
            q    = query_mat_f32[vptr:vend]
            sims = q @ chunk.T
            ri   = np.arange(B)[:, None]

            if C <= k_this:
                c_ord = np.argsort(sims, axis=1)[:, ::-1]
            else:
                c_ord = np.argpartition(sims, -k_this, axis=1)[:, -k_this:]
                c_ord = c_ord[ri, np.argsort(sims[ri, c_ord], axis=1)[:, ::-1]]

            c_sims = sims[ri, c_ord]
            c_idxs = (ptr + c_ord).astype(np.int64)

            ms = np.concatenate([best_sims[vptr:vend], c_sims], axis=1)
            mi = np.concatenate([best_idxs[vptr:vend], c_idxs], axis=1)
            if ms.shape[1] > top_k:
                keep = np.argpartition(ms, -top_k, axis=1)[:, -top_k:]
                ms   = ms[ri, keep]; mi = mi[ri, keep]
            sord = np.argsort(ms, axis=1)[:, ::-1]
            best_sims[vptr:vend] = ms[ri, sord]
            best_idxs[vptr:vend] = mi[ri, sord]
            vptr = vend

        ptr = end
    return best_idxs, best_sims


# ─────────────────────────────────────────────────────────────────────────────
# Dedup filter
# ─────────────────────────────────────────────────────────────────────────────

def apply_dedup_filter_batch(all_idxs, all_sims, train_prefs, val_ids,
                              query_len, filter_name):
    """
    all_idxs: [N, K] int64
    all_sims:  [N, K] float32
    train_prefs: [N_train, query_len] int32 (loaded into RAM)
    val_ids:     [N, seq_len] int32

    Returns keep_mask [N, K] bool  (True = keep this retrieved row).
    Also returns frac_removed_per_row [N] float.
    """
    N, K = all_idxs.shape

    if train_prefs is None or filter_name == "no_filter":
        return np.ones((N, K), dtype=bool), np.zeros(N, dtype=np.float32)

    # Val prefixes: last query_len tokens of input_ids
    seq_len = val_ids.shape[1]
    T = min(seq_len, query_len)
    if T < query_len:
        val_prefs = np.zeros((N, query_len), dtype=np.int32)
        val_prefs[:, query_len - T:] = val_ids[:, -T:].astype(np.int32)
    else:
        val_prefs = val_ids[:, -query_len:].astype(np.int32)  # [N, query_len]

    # Load all needed train prefixes in one fancy-index: [N*K, query_len]
    flat_idxs  = all_idxs.ravel()                     # [N*K]
    batch_prefs = train_prefs[flat_idxs].reshape(N, K, query_len)  # [N, K, query_len]

    if filter_name == "suffix16":
        # Remove if last 16 tokens of train prefix == last 16 tokens of val prefix
        is_dup = (batch_prefs[:, :, -16:] ==
                  val_prefs[:, None, -16:]).all(axis=2)   # [N, K]
    elif filter_name == "suffix32":
        is_dup = (batch_prefs[:, :, -32:] ==
                  val_prefs[:, None, -32:]).all(axis=2)
    elif filter_name in ("jaccard03", "jaccard05"):
        thresh   = 0.3 if filter_name == "jaccard03" else 0.5
        is_dup   = np.zeros((N, K), dtype=bool)
        # Per-row jaccard (vectorised within each row)
        for i in range(N):
            vset = set(val_prefs[i].tolist())
            vlen = len(vset)
            for j in range(K):
                tset  = set(batch_prefs[i, j].tolist())
                inter = len(vset & tset)
                union = vlen + len(tset) - inter
                if union > 0 and (inter / union) >= thresh:
                    is_dup[i, j] = True
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    keep_mask = ~is_dup
    # fraction removed per row
    frac_removed = is_dup.sum(axis=1).astype(np.float32) / max(K, 1)
    return keep_mask, frac_removed


# ─────────────────────────────────────────────────────────────────────────────
# Passage vote scoring  (vectorised over all val rows at once)
# ─────────────────────────────────────────────────────────────────────────────

def compute_passage_votes_batch(all_idxs, all_sims, keep_mask,
                                cont_arr, cand_ids, cont_lens):
    """
    all_idxs:  [N, K] int64
    all_sims:  [N, K] float32
    keep_mask: [N, K] bool
    cont_arr:  [N_train, cont_len] int32  (loaded into RAM)
    cand_ids:  [N, P] int32

    cont_lens: dict  name -> int, e.g. {'next1': 1, 'next4': 4, 'next8': 8}

    Returns dict  name -> [N, P] float32  (unnormalised passage vote scores)
    """
    N, K   = all_idxs.shape
    P      = cand_ids.shape[1]
    CL     = cont_arr.shape[1]

    eff_sims = (all_sims * keep_mask.astype(np.float32))  # [N, K] zeros out removed

    # Bulk load continuations for all retrieved rows: [N*K, CL] → [N, K, CL]
    flat_conts  = cont_arr[all_idxs.ravel()].reshape(N, K, CL)  # [N, K, CL]

    result = {}
    for name, L in cont_lens.items():
        L = min(L, CL)
        conts_L = flat_conts[:, :, :L]              # [N, K, L]
        valid   = (conts_L != -1)                    # [N, K, L]

        # Match: [N, K, L, 1] == [N, 1, 1, P] → [N, K, L, P]
        match = (conts_L[:, :, :, None] == cand_ids[:, None, None, :]) & valid[:, :, :, None]

        # Weighted sum: eff_sims [N, K] × match [N, K, L, P] → [N, P]
        scores = (eff_sims[:, :, None, None] * match.astype(np.float32)).sum(axis=(1, 2))
        result[name] = scores   # [N, P]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Slice building
# ─────────────────────────────────────────────────────────────────────────────

def build_slices(data, tok_arr, reg_arr, unk_r, unk_s, sr):
    N    = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk"].astype(np.int64)
    lgt  = data["lgt"].astype(np.float32)
    ids  = data["ids"].astype(np.int64)

    P    = topk.shape[1]
    ar   = np.arange(N)

    # Gold in pool
    gold_in_pool  = (topk == gold[:, None]).any(axis=1)     # [N]

    # Base top1
    base_top1     = topk[ar, lgt.argmax(axis=1)]            # [N]

    # Base softmax probabilities
    lgt_shifted   = lgt - lgt.max(axis=1, keepdims=True)
    base_probs    = np.exp(lgt_shifted)
    base_probs   /= base_probs.sum(axis=1, keepdims=True) + _EPS
    base_top1_prob = base_probs[:, 0]                        # prob of base top1
    sorted_probs  = np.sort(base_probs, axis=1)[:, ::-1]
    base_margin   = sorted_probs[:, 0] - sorted_probs[:, 1] if P >= 2 \
                    else sorted_probs[:, 0]

    # Gold in context
    gold_in_ctx   = (ids == gold[:, None]).any(axis=1)       # [N]

    # Region-based slices (only if maps available)
    V = len(tok_arr)
    def tok_region(t):
        t = int(t)
        return tok_arr[t] if 0 <= t < V else unk_r

    def is_bpe_fragment(t):
        if _enc is None: return False
        try:
            s = _enc.decode([int(t)])
            return not s.startswith(" ") and s.isalpha()
        except Exception:
            return False

    def is_number(t):
        if _enc is None: return False
        try:
            return _enc.decode([int(t)]).strip().replace(",", "").replace(".", "").isnumeric()
        except Exception:
            return False

    def is_punct(t):
        if _enc is None: return False
        try:
            s = _enc.decode([int(t)]).strip()
            return len(s) > 0 and all(not c.isalnum() for c in s)
        except Exception:
            return False

    gold_reg  = np.array([tok_region(g) for g in gold])
    top1_reg  = np.array([tok_region(t) for t in topk[:, 0]])
    top2_reg  = np.array([tok_region(t) for t in topk[:, 1]]) if P >= 2 \
                else top1_reg

    same_reg_confuser = (gold_in_pool & (gold_reg == top1_reg) & (base_top1 != gold))

    if sr:
        R = len(reg_arr)
        gold_sr  = np.array([reg_arr[r] if 0 <= r < R else unk_s for r in gold_reg])
        top1_sr  = np.array([reg_arr[r] if 0 <= r < R else unk_s for r in top1_reg])
        same_sr  = (gold_in_pool & (gold_sr == top1_sr) & (base_top1 != gold))
    else:
        same_sr  = same_reg_confuser.copy()

    bpe_frag  = np.array([is_bpe_fragment(g) for g in gold])
    number    = np.array([is_number(g)        for g in gold])
    punct     = np.array([is_punct(g)         for g in gold])
    word      = ~bpe_frag & ~number & ~punct

    slices = {
        "all":                    np.ones(N, dtype=bool),
        "bucketA_confuser":       gold_in_pool,
        "gold_in_context":        gold_in_ctx,
        "gold_not_in_context":    ~gold_in_ctx,
        "high_base_margin":       (base_margin > 0.2),
        "low_base_margin":        (base_margin < 0.05),
        "same_region_confuser":   same_reg_confuser,
        "same_superregion":       same_sr,
        "bpe_fragment":           bpe_frag,
        "number":                 number,
        "punctuation":            punct,
        "word_choice":            word,
    }
    # Sizes
    for s, m in slices.items():
        print(f"  slice [{s:30s}] = {m.sum():6,}/{N:,}")

    extra = {
        "gold_in_pool":  gold_in_pool,
        "base_top1":     base_top1,
        "base_probs":    base_probs,
        "base_margin":   base_margin,
        "base_top1_prob": base_top1_prob,
    }
    return slices, extra


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation for a single (filter, lambda, policy, slice) config
# ─────────────────────────────────────────────────────────────────────────────

def compute_acc(final_top1, gold, mask):
    m = mask & (np.arange(len(gold)) < len(final_top1))
    if m.sum() == 0:
        return float("nan")
    return float((final_top1[m] == gold[m]).mean())


def compute_mrr(cand_ids, combined_scores, gold, mask):
    """Mean reciprocal rank of gold in combined ranking."""
    if mask.sum() == 0:
        return float("nan")
    idxs = np.where(mask)[0]
    rr = []
    for i in idxs:
        g     = int(gold[i])
        scores = combined_scores[i]           # [P]
        rank_order = np.argsort(scores)[::-1] # descending
        pos = np.where(cand_ids[i, rank_order] == g)[0]
        rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
    return float(np.mean(rr)) if rr else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(data, keys, gold_arr, cont_arr, train_prefs,
             tok_arr, reg_arr, unk_r, unk_s, sr,
             slices, extra,
             query_len, cont_len, feature_dim,
             retrieval_k_grid, lambda_grid, overlap_filters,
             conf_gate_thresh, max_examples, seed,
             train_chunk_size, val_batch_size):

    rng   = random.Random(seed)
    N     = len(data["gold"])
    gold  = data["gold"].astype(np.int64)
    topk  = data["topk"].astype(np.int64)    # [N, P]
    lgt   = data["lgt"].astype(np.float32)   # [N, P]
    ids   = data["ids"].astype(np.int64)     # [N, seq_len]
    P     = topk.shape[1]
    ar    = np.arange(N)

    gold_in_pool  = extra["gold_in_pool"]
    base_probs    = extra["base_probs"]
    base_top1_prob = extra["base_top1_prob"]
    base_top1     = extra["base_top1"]

    # log(base_probs) for combining
    log_base = np.log(base_probs + _EPS)     # [N, P]

    cont_lens = {"next1": 1, "next4": 4, "next8": min(8, cont_len)}

    # ── Storage for all metric rows ─────────────────────────────────────
    all_metric_rows = []   # flat dicts for CSV

    # ── Dedup stats: fraction removed per (mode, k, filter) ────────────
    dedup_stats_rows = []

    # ── Example collection ──────────────────────────────────────────────
    examples_helps  = []   # passage changed base wrong → right
    examples_hurts  = []   # passage changed base right → wrong
    examples_dedup  = []   # dedup removed gold-containing passage

    # ═══════════════════════════════════════════════════════════════════
    for mode_name, key_mmap in keys.items():
        print(f"\n[eval] mode={mode_name}  shape={key_mmap.shape}")

        # Build val query matrix
        t0 = time.time()
        if mode_name == "h_ctx":
            if "h_ctx" not in data:
                print("  [WARN] h_ctx not in val data — skip"); continue
            qmat = l2_norm_rows(data["h_ctx"].astype(np.float32))  # [N, D]
        elif mode_name == "h_raw":
            if "h_raw" not in data:
                print("  [WARN] h_raw not in val data — skip"); continue
            qmat = l2_norm_rows(data["h_raw"].astype(np.float32))
        elif mode_name == "recency_bow":
            print(f"  [query] building recency_bow query matrix [{N}, {feature_dim}]...")
            qmat = build_recency_bow_batch(ids.astype(np.int32), query_len, feature_dim)
        elif mode_name == "shingle5gram":
            print(f"  [query] building shingle5gram query matrix [{N}, {feature_dim}]...")
            qmat = build_shingle5gram_batch(ids.astype(np.int32), query_len, feature_dim)
        else:
            print(f"  [WARN] unknown mode {mode_name} — skip"); continue
        print(f"  [query] built in {time.time()-t0:.1f}s")

        for k in retrieval_k_grid:
            print(f"\n  [retrieve] mode={mode_name}  k={k}")
            t0 = time.time()
            all_idxs, all_sims = batch_retrieve_all(
                qmat, key_mmap, train_chunk_size, k, val_batch_size)
            print(f"  [retrieve] done in {time.time()-t0:.1f}s")

            # Pre-fetch continuations for all retrieved: [N, k, cont_len]
            # (cont_arr is already in RAM — fancy index is fast)
            # gold_in_any_cont_nofilt: does gold appear anywhere in retrieved continuations?
            flat_conts_all  = cont_arr[all_idxs.ravel()].reshape(N, k, cont_len)   # [N,k,CL]
            gold_in_any_cont = (flat_conts_all[:, :, :cont_lens["next8"]]
                                == gold[:, None, None]).any(axis=(1, 2))  # [N]

            for filt in overlap_filters:
                print(f"    [filter] {filt}")
                t0 = time.time()
                if train_prefs is not None:
                    keep_mask, frac_rem = apply_dedup_filter_batch(
                        all_idxs, all_sims, train_prefs, ids.astype(np.int32),
                        query_len, filt)
                else:
                    keep_mask = np.ones((N, k), dtype=bool)
                    frac_rem  = np.zeros(N, dtype=np.float32)
                print(f"    [filter] done in {time.time()-t0:.1f}s  "
                      f"avg_removed={frac_rem.mean():.3f}")

                # Dedup stats
                # "dedup_removes_gold" = gold is in no_filter continuations but
                # not in filtered continuations (because the passage was removed)
                flat_conts_kept = (flat_conts_all * keep_mask[:, :, None]).astype(np.int32)
                # Use -1 for removed rows
                flat_conts_kept_masked = np.where(keep_mask[:, :, None], flat_conts_all, -1)
                gold_in_kept = (flat_conts_kept_masked[:, :, :cont_lens["next8"]]
                                == gold[:, None, None]).any(axis=(1, 2))
                dedup_removes_gold = gold_in_any_cont & ~gold_in_kept  # [N]

                dedup_stats_rows.append({
                    "mode":              mode_name,
                    "k":                 k,
                    "filter":            filt,
                    "avg_frac_removed":  float(frac_rem.mean()),
                    "frac_rows_any_removed": float((frac_rem > 0).mean()),
                    "frac_gold_in_retrieved_nofilt": float(gold_in_any_cont.mean()),
                    "frac_gold_in_retrieved_after_filt": float(gold_in_kept.mean()),
                    "frac_dedup_removes_gold": float(dedup_removes_gold.mean()),
                })

                # ── Passage votes ──────────────────────────────────────
                passage_scores = compute_passage_votes_batch(
                    all_idxs, all_sims, keep_mask, cont_arr, topk, cont_lens)
                # passage_scores: dict name -> [N, P]

                # Normalise passage scores to probability space
                passage_probs = {}
                for pname, sc in passage_scores.items():
                    s = sc.copy()
                    s_sum = s.sum(axis=1, keepdims=True)
                    passage_probs[pname] = s / np.maximum(s_sum, _EPS)

                # ── Policies × lambdas ─────────────────────────────────
                for vote_name in ["next1", "next4", "next8"]:
                    log_pp = np.log(passage_probs[vote_name] + _EPS)  # [N, P]

                    for lam in lambda_grid:
                        # Policy 1: base_candidate (no passage)
                        combined_base  = lgt
                        final_base     = topk[ar, combined_base.argmax(axis=1)]

                        # Policy 2: passage_only (ignore base logits)
                        final_pass_only = topk[ar, passage_scores[vote_name].argmax(axis=1)]

                        # Policy 3: base_plus_passage (log-linear blend)
                        combined_blend = lgt + lam * log_pp          # [N, P]
                        final_blend    = topk[ar, combined_blend.argmax(axis=1)]

                        # Policy 4: confidence_gated
                        use_blend = base_top1_prob < conf_gate_thresh  # [N]
                        final_gated = np.where(use_blend, final_blend, final_base)

                        # ── Metrics per slice ──────────────────────────
                        for sname, smask in slices.items():
                            n_slice = int(smask.sum())
                            if n_slice == 0:
                                continue

                            def _acc(pred): return compute_acc(pred, gold, smask)
                            def _mrr(sc): return compute_mrr(topk, sc, gold, smask)

                            row = {
                                "mode":        mode_name,
                                "k":           k,
                                "filter":      filt,
                                "vote":        vote_name,
                                "lambda":      lam,
                                "slice":       sname,
                                "n_slice":     n_slice,
                                # base (same for all filter/vote/lambda — repeated for pivot ease)
                                "base_acc1":   _acc(final_base),
                                # passage only
                                "pass_only_acc1": _acc(final_pass_only),
                                # blended
                                "blend_acc1":  _acc(final_blend),
                                "blend_mrr":   _mrr(combined_blend),
                                # gated
                                "gated_acc1":  _acc(final_gated),
                                # passage coverage
                                "frac_gold_in_retrieved": float(
                                    gold_in_kept[smask].mean()),
                                "passage_gold_vote_prob": float(
                                    passage_probs[vote_name][smask,
                                        np.where(topk[smask] == gold[smask, None],
                                                 *np.where(topk[smask] == gold[smask, None]))
                                    ].mean() if False else
                                    # simplified: mean passage prob for gold position
                                    _passage_gold_prob(passage_probs[vote_name],
                                                       topk, gold, smask)),
                            }
                            all_metric_rows.append(row)

                        # ── Example collection (only for best-config guess) ─
                        if (filt == "no_filter" and vote_name == "next1"
                                and lam == 1.0 and len(examples_helps) < max_examples):
                            for i in range(N):
                                b1 = int(final_base[i])
                                bl = int(final_blend[i])
                                g  = int(gold[i])
                                if b1 != g and bl == g:
                                    examples_helps.append({
                                        "i": i, "mode": mode_name, "k": k,
                                        "gold": g, "base_top1": b1, "blend_top1": bl,
                                        "pass_score": float(passage_scores["next1"][i,
                                            int((topk[i] == g).argmax())] if (topk[i] == g).any() else 0),
                                        "ids": ids[i].tolist(),
                                    })
                                elif b1 == g and bl != g and len(examples_hurts) < max_examples:
                                    examples_hurts.append({
                                        "i": i, "mode": mode_name, "k": k,
                                        "gold": g, "base_top1": b1, "blend_top1": bl,
                                        "ids": ids[i].tolist(),
                                    })

                # Dedup-removes-gold examples
                if filt in ("suffix16", "jaccard03") and len(examples_dedup) < max_examples:
                    for i in np.where(dedup_removes_gold)[0][:max_examples - len(examples_dedup)]:
                        examples_dedup.append({
                            "i": int(i), "mode": mode_name, "k": k, "filter": filt,
                            "gold": int(gold[i]),
                            "frac_removed": float(frac_rem[i]),
                            "ids": ids[i].tolist(),
                        })

    return all_metric_rows, dedup_stats_rows, examples_helps, examples_hurts, examples_dedup


def _passage_gold_prob(passage_probs, topk, gold, mask):
    """Mean passage probability assigned to the gold token, averaged over masked rows."""
    probs = passage_probs[mask]  # [n, P]
    topk_m = topk[mask]          # [n, P]
    gold_m = gold[mask]          # [n]
    vals = []
    for i in range(len(gold_m)):
        pos = np.where(topk_m[i] == gold_m[i])[0]
        if len(pos):
            vals.append(float(probs[i, pos[0]]))
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(all_metric_rows, dedup_stats_rows, examples_helps, examples_hurts,
                 examples_dedup, output_dir, slices, args, N_total, mode_names,
                 conf_gate_thresh):
    os.makedirs(output_dir, exist_ok=True)

    # ── policy_grid.csv ────────────────────────────────────────────────
    if all_metric_rows:
        csv_path = os.path.join(output_dir, "passage_policy_grid.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_metric_rows[0].keys()))
            w.writeheader(); w.writerows(all_metric_rows)
        print(f"[save] {csv_path}")

    # ── dedup_stats.csv ────────────────────────────────────────────────
    if dedup_stats_rows:
        csv_path = os.path.join(output_dir, "passage_dedup_stats.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dedup_stats_rows[0].keys()))
            w.writeheader(); w.writerows(dedup_stats_rows)
        print(f"[save] {csv_path}")

    # ── slice summary: best config per slice ────────────────────────────
    # group by (slice, policy) → find best lambda/filter/mode/k by blend_acc1
    from collections import defaultdict as dd
    best_by_slice = {}
    for row in all_metric_rows:
        key = (row["slice"], row["vote"])
        if key not in best_by_slice or row["blend_acc1"] > best_by_slice[key]["blend_acc1"]:
            best_by_slice[key] = dict(row)

    csv_path = os.path.join(output_dir, "passage_slice_metrics.csv")
    rows_to_write = sorted(best_by_slice.values(),
                           key=lambda r: (r["slice"], r["vote"]))
    if rows_to_write:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_to_write[0].keys()))
            w.writeheader(); w.writerows(rows_to_write)
        print(f"[save] {csv_path}")

    # ── example files ──────────────────────────────────────────────────
    def write_examples(path, examples, header):
        with open(path, "w") as f:
            f.write(f"# {header}\n\n")
            f.write(f"Total examples: {len(examples)}\n\n")
            for ex in examples[:50]:
                f.write(f"---\n")
                f.write(f"Row {ex['i']}  mode={ex.get('mode','?')}  "
                        f"k={ex.get('k','?')}\n")
                f.write(f"  gold:       {_decode_tok(ex['gold'])}  (id={ex['gold']})\n")
                if "base_top1" in ex:
                    f.write(f"  base_top1:  {_decode_tok(ex['base_top1'])}\n")
                if "blend_top1" in ex:
                    f.write(f"  blend_top1: {_decode_tok(ex['blend_top1'])}\n")
                if "frac_removed" in ex:
                    f.write(f"  frac_removed_by_dedup: {ex['frac_removed']:.3f}\n")
                ctx = _decode_ids(ex["ids"][-40:]) if "ids" in ex else ""
                f.write(f"  context: ...{ctx}\n\n")

    write_examples(os.path.join(output_dir, "examples_passage_helps.md"),
                   examples_helps, "Rows where passage recall corrects base error")
    write_examples(os.path.join(output_dir, "examples_passage_hurts.md"),
                   examples_hurts, "Rows where passage recall breaks a correct base prediction")
    write_examples(os.path.join(output_dir, "examples_dedup_removes_gold.md"),
                   examples_dedup, "Rows where dedup filter removed the gold-containing passage")

    # ── report.md ─────────────────────────────────────────────────────
    # Compute summary stats for questions
    def best_blend(rows, slice_name, vote="next1"):
        cands = [r for r in rows if r["slice"] == slice_name and r["vote"] == vote]
        if not cands: return {}
        return max(cands, key=lambda r: r["blend_acc1"])

    def base_acc(rows, slice_name):
        cands = [r for r in rows if r["slice"] == slice_name]
        if not cands: return float("nan")
        return cands[0]["base_acc1"]

    ba_all    = base_acc(all_metric_rows, "all")
    bb_all    = best_blend(all_metric_rows, "all")
    bb_gnic   = best_blend(all_metric_rows, "gold_not_in_context")
    bb_src    = best_blend(all_metric_rows, "same_region_confuser")
    bb_num    = best_blend(all_metric_rows, "number")

    # Dedup Q4/Q5
    nf_row    = next((r for r in dedup_stats_rows
                      if r["filter"] == "no_filter"
                      and r["k"] == max(r["k"] for r in dedup_stats_rows)), None)
    s16_row   = next((r for r in dedup_stats_rows
                      if r["filter"] == "suffix16"
                      and r["k"] == max(r["k"] for r in dedup_stats_rows)), None)

    report_path = os.path.join(output_dir, "passage_recall_report.md")
    with open(report_path, "w") as f:
        f.write("# Passage Recall Dedup Diagnostic V1 — Report\n\n")
        f.write("> NON-PARAMETRIC MEMORY DIAGNOSTIC.  "
                "Val gold used ONLY for metrics after retrieval.\n\n")
        f.write(f"**N_val:** {N_total:,}  |  "
                f"**Modes:** {mode_names}  |  "
                f"**Candidate pool:** {args.candidate_pool_size}\n\n")

        f.write("---\n\n")
        f.write("## Q1: Does train-only passage recall recover gold?\n\n")
        f.write(f"- Base acc@1 (all): **{ba_all:.4f}**\n")
        if bb_all:
            f.write(f"- Best blend acc@1 (all): **{bb_all.get('blend_acc1', float('nan')):.4f}**  "
                    f"(mode={bb_all.get('mode','?')}  k={bb_all.get('k','?')}  "
                    f"filter={bb_all.get('filter','?')}  λ={bb_all.get('lambda','?')}  "
                    f"vote={bb_all.get('vote','?')})\n")
            delta = bb_all.get('blend_acc1', float('nan')) - ba_all
            verdict = ("✅ YES — passage recall improves acc@1" if delta > 0.005
                       else "⚠ MARGINAL" if delta > 0 else "❌ NO — no improvement")
            f.write(f"- **{verdict}** (Δ={delta:+.4f})\n\n")

        f.write("## Q2: Does it help where pointer/fuzzy fail (gold_not_in_context)?\n\n")
        if bb_gnic:
            f.write(f"- Base acc@1 (gold_not_in_context): "
                    f"**{bb_gnic.get('base_acc1', float('nan')):.4f}**\n")
            f.write(f"- Best blend acc@1: **{bb_gnic.get('blend_acc1', float('nan')):.4f}**  "
                    f"(mode={bb_gnic.get('mode','?')}  λ={bb_gnic.get('lambda','?')})\n\n")
        else:
            f.write("- Insufficient data\n\n")

        f.write("## Q3: Does it help exact numbers/entities?\n\n")
        if bb_num:
            f.write(f"- Base acc@1 (number): "
                    f"**{bb_num.get('base_acc1', float('nan')):.4f}**\n")
            f.write(f"- Best blend acc@1: **{bb_num.get('blend_acc1', float('nan')):.4f}**\n\n")
        else:
            f.write("- `number` slice empty\n\n")

        f.write("## Q4: Does performance survive dedup/overlap filtering?\n\n")
        if nf_row and s16_row:
            f.write(f"- Gold in retrieved (no_filter): {nf_row['frac_gold_in_retrieved_nofilt']:.4f}\n")
            f.write(f"- Gold in retrieved (suffix16):  {s16_row['frac_gold_in_retrieved_after_filt']:.4f}\n")
            f.write(f"- Frac dedup removes gold (suffix16): "
                    f"{s16_row['frac_dedup_removes_gold']:.4f}\n")
            surv = bb_all.get('filter', '?') not in ('no_filter',) if bb_all else False
            f.write(f"- Best config uses filter: **{bb_all.get('filter','?') if bb_all else '?'}**\n\n")
        else:
            f.write("- Insufficient dedup stats\n\n")

        f.write("## Q5: Genuine passage memory or near-duplicate memorisation?\n\n")
        if s16_row:
            removed = s16_row.get("avg_frac_removed", float("nan"))
            rg      = s16_row.get("frac_dedup_removes_gold", float("nan"))
            verdict5 = ("⚠ HIGH OVERLAP — results may be near-duplicate memorisation"
                        if removed > 0.3 else
                        "✅ LOW OVERLAP — passage recall appears genuine")
            f.write(f"- suffix16 avg fraction removed: **{removed:.4f}**\n")
            f.write(f"- suffix16 fraction where gold passage was removed: **{rg:.4f}**\n")
            f.write(f"- **{verdict5}**\n\n")

        f.write("## Q6: Should passage recall become a memory expert?\n\n")
        if bb_all:
            delta = bb_all.get('blend_acc1', float('nan')) - ba_all
            low_overlap = (s16_row.get("avg_frac_removed", 1.0) < 0.3 if s16_row else False)
            if delta > 0.01 and low_overlap:
                verdict6 = "✅ YES — meaningful improvement with low duplication signal"
            elif delta > 0.005:
                verdict6 = "⚠ MARGINAL — small gain; investigate duplication first"
            else:
                verdict6 = "❌ NO — insufficient signal to justify expert module"
            f.write(f"- **{verdict6}**\n\n")

        f.write("---\n\n## Best Configs per Slice (vote=next1)\n\n")
        f.write("| slice | n | base_acc1 | blend_acc1 | Δ | mode | k | filter | λ |\n")
        f.write("|-------|---|-----------|------------|---|------|---|--------|---|\n")
        for sname in sorted(slices.keys()):
            bb = best_blend(all_metric_rows, sname, "next1")
            if not bb: continue
            delta_s = bb.get('blend_acc1', float('nan')) - bb.get('base_acc1', float('nan'))
            f.write(f"| {sname} | {bb.get('n_slice','?')} | "
                    f"{bb.get('base_acc1', float('nan')):.4f} | "
                    f"{bb.get('blend_acc1', float('nan')):.4f} | "
                    f"{delta_s:+.4f} | "
                    f"{bb.get('mode','?')} | {bb.get('k','?')} | "
                    f"{bb.get('filter','?')} | {bb.get('lambda','?')} |\n")

        f.write("\n---\n\n## Dedup Statistics\n\n")
        f.write("| mode | k | filter | avg_removed | gold_in_retrieved | dedup_removes_gold |\n")
        f.write("|------|---|--------|-------------|-------------------|--------------------|\n")
        for r in dedup_stats_rows:
            f.write(f"| {r['mode']} | {r['k']} | {r['filter']} | "
                    f"{r['avg_frac_removed']:.4f} | "
                    f"{r['frac_gold_in_retrieved_after_filt']:.4f} | "
                    f"{r['frac_dedup_removes_gold']:.4f} |\n")

        f.write(f"\n\n*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"[save] {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Passage Recall Dedup Diagnostic V1")
    p.add_argument("--val_dir",             required=True)
    p.add_argument("--index_dir",           required=True)
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--token_to_region",     default=None)
    p.add_argument("--super_map",           default=None)
    p.add_argument("--candidate_pool_size", type=int, default=32)
    p.add_argument("--retrieval_k_grid",    default="8,16,32,64")
    p.add_argument("--lambda_grid",         default="0.0,0.25,0.5,1.0,2.0,4.0")
    p.add_argument("--overlap_filters",
                   default="no_filter,suffix16,suffix32,jaccard03,jaccard05")
    p.add_argument("--modes",               default=None,
                   help="Subset of built modes to evaluate (default: all built)")
    p.add_argument("--conf_gate_thresh",    type=float, default=0.3,
                   help="Base confidence threshold for confidence_gated policy")
    p.add_argument("--max_val_rows",        type=int, default=None)
    p.add_argument("--max_examples",        type=int, default=50)
    p.add_argument("--train_chunk_size",    type=int, default=65536)
    p.add_argument("--val_batch_size",      type=int, default=4096)
    p.add_argument("--seed",               type=int, default=42)
    return p, p.parse_args()


def main():
    _, args = _parse()
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 64)
    print(" Passage Recall Dedup Diagnostic V1")
    print(f" val_dir:    {args.val_dir}")
    print(f" index_dir:  {args.index_dir}")
    print(f" output_dir: {args.output_dir}")
    print("=" * 64)

    retrieval_k_grid = [int(x) for x in args.retrieval_k_grid.split(",")]
    lambda_grid      = [float(x) for x in args.lambda_grid.split(",")]
    overlap_filters  = [x.strip() for x in args.overlap_filters.split(",") if x.strip()]
    requested_modes  = ([m.strip() for m in args.modes.split(",") if m.strip()]
                        if args.modes else None)

    # Load val shards
    data = load_val_shards(args.val_dir, args.candidate_pool_size, args.max_val_rows)
    N    = len(data["gold"])

    has_hctx = "h_ctx" in data
    has_hraw = "h_raw" in data

    # Load region maps
    tok_arr = reg_arr = None
    unk_r = unk_s = 1; sr = False
    if args.token_to_region and os.path.isfile(args.token_to_region):
        tok_arr, reg_arr, unk_r, unk_s, sr = load_maps(
            args.token_to_region, args.super_map)
        print(f"[maps] token_to_region loaded  sr={sr}")
    else:
        print("[maps] token_to_region not found — region slices disabled")
        V = int(data["topk"].max()) + 2
        tok_arr  = np.zeros(V, dtype=np.int32)
        reg_arr  = np.zeros(2, dtype=np.int32)

    # Load index
    if requested_modes is None:
        # Load all available modes from index
        cfg_p = os.path.join(args.index_dir, "config.json")
        with open(cfg_p) as f:
            _cfg = json.load(f)
        requested_modes = _cfg.get("modes_built",
                                   _cfg.get("modes_requested", ["h_raw", "h_ctx", "recency_bow"]))
        print(f"[index] auto-selected modes: {requested_modes}")

    (keys, gold_arr, cont_arr, train_prefs, rowid_arr, N_train,
     h_dim, query_len, cont_len, feature_dim, idx_cfg) = load_index(
        args.index_dir, requested_modes, has_hctx, has_hraw)

    # Build slices
    print("[slices] building slices...")
    slices, extra = build_slices(data, tok_arr, reg_arr, unk_r, unk_s, sr)

    # Run evaluation
    print("[eval] starting evaluation...")
    t_eval = time.time()
    (all_metric_rows, dedup_stats_rows,
     examples_helps, examples_hurts, examples_dedup) = run_eval(
        data=data,
        keys=keys,
        gold_arr=gold_arr,
        cont_arr=cont_arr,
        train_prefs=train_prefs,
        tok_arr=tok_arr,
        reg_arr=reg_arr,
        unk_r=unk_r,
        unk_s=unk_s,
        sr=sr,
        slices=slices,
        extra=extra,
        query_len=query_len,
        cont_len=cont_len,
        feature_dim=feature_dim,
        retrieval_k_grid=retrieval_k_grid,
        lambda_grid=lambda_grid,
        overlap_filters=overlap_filters,
        conf_gate_thresh=args.conf_gate_thresh,
        max_examples=args.max_examples,
        seed=args.seed,
        train_chunk_size=args.train_chunk_size,
        val_batch_size=args.val_batch_size,
    )
    print(f"[eval] done in {time.time()-t_eval:.1f}s  "
          f"({len(all_metric_rows):,} metric rows)")

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)

    report_path = write_report(
        all_metric_rows=all_metric_rows,
        dedup_stats_rows=dedup_stats_rows,
        examples_helps=examples_helps,
        examples_hurts=examples_hurts,
        examples_dedup=examples_dedup,
        output_dir=args.output_dir,
        slices=slices,
        args=args,
        N_total=N,
        mode_names=list(keys.keys()),
        conf_gate_thresh=args.conf_gate_thresh,
    )

    # Save config
    cfg_out = {
        "val_dir":             args.val_dir,
        "index_dir":           args.index_dir,
        "candidate_pool_size": args.candidate_pool_size,
        "retrieval_k_grid":    retrieval_k_grid,
        "lambda_grid":         lambda_grid,
        "overlap_filters":     overlap_filters,
        "modes_evaluated":     list(keys.keys()),
        "conf_gate_thresh":    args.conf_gate_thresh,
        "N_val":               N,
        "N_train":             N_train,
        "query_len":           query_len,
        "cont_len":            cont_len,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)

    print("\n" + "=" * 64)
    print(" Passage Recall Dedup Diagnostic V1 complete.")
    print(f" Report: {report_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
