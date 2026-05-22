#!/usr/bin/env python3
"""
run_retrieval_evidence_sweep.py — Diagnostic sweep over retrieval key quality.

Tests 9 different representations as retrieval keys and measures whether each
provides token-level evidence beyond the base model's top-K candidates.

NOT a resolver training script. Purely diagnostic.

Keys tested:
  h_prime              final prediction-ready hidden (saved in patched shards)
  h_raw                pre-region-proj hidden (saved in patched shards)
  layer_early          hidden at ~25% transformer depth
  layer_mid            hidden at ~50% transformer depth
  layer_late           hidden at ~75% transformer depth
  hybrid_early_hprime  concat(normalize(early), normalize(h_prime)) normalized
  lexical_last16       Jaccard overlap of last 16 input tokens
  lexical_last32       Jaccard overlap of last 32 input tokens
  hybrid_mid_lexical   layer_mid dense pre-rank + lexical rerank
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe

try:
    import faiss as _faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

_BASELINE_ADDED_GOLD = 0.0017   # h_prime from limited500k logitfix run

_ALL_KEYS = [
    "h_prime", "h_raw",
    "layer_early", "layer_mid", "layer_late",
    "hybrid_early_hprime",
    "lexical_last16", "lexical_last32",
    "hybrid_mid_lexical",
]
_LEXICAL_KEYS = {"lexical_last16", "lexical_last32"}
_HYBRID_LEXICAL = "hybrid_mid_lexical"
_NEEDS_LIVE = {
    "layer_early", "layer_mid", "layer_late",
    "hybrid_early_hprime", "hybrid_mid_lexical",
}


# ══════════════════════════════════════════════════════════════════════════════
# Transformer block discovery
# ══════════════════════════════════════════════════════════════════════════════

def find_transformer_blocks(backbone):
    """
    Discover transformer block list in backbone.
    Returns (blocks: ModuleList, n_layers: int) or (None, 0).
    """
    for attr_seq in [
        ("transformer", "h"),
        ("transformer", "layers"),
        ("model", "layers"),
        ("decoder", "layers"),
        ("encoder", "layers"),
    ]:
        obj = backbone
        try:
            for a in attr_seq:
                obj = getattr(obj, a)
            if isinstance(obj, nn.ModuleList) and len(obj) > 1:
                return obj, len(obj)
        except AttributeError:
            pass

    for attr in ("blocks", "layers", "h"):
        obj = getattr(backbone, attr, None)
        if obj is not None and isinstance(obj, nn.ModuleList) and len(obj) > 1:
            return obj, len(obj)

    # Two-level search
    for _, child in backbone.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 3:
            return child, len(child)
        for _, gc in child.named_children():
            if isinstance(gc, nn.ModuleList) and len(gc) > 3:
                return gc, len(gc)

    return None, 0


def make_layer_info(backbone, d_model):
    blocks, n = find_transformer_blocks(backbone)
    if blocks is None:
        n = 0
    return {
        "blocks":     blocks,
        "n_layers":   n,
        "early_idx":  max(0, n // 4),
        "mid_idx":    max(0, n // 2),
        "late_idx":   max(0, 3 * n // 4),
        "d_model":    d_model,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Representation extraction
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_repr_live(backbone, src, requested_keys, layer_info):
    """
    Single backbone forward pass; collect requested representations.
    src: (B, T) int64 on device.
    Returns {key: (B, d) float32 cpu tensor}. Key absent if unavailable.
    """
    blocks   = layer_info["blocks"]
    n_layers = layer_info["n_layers"]

    layer_cap = {}
    hooks = []

    def _make_hook(slot):
        def _h(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(o, torch.Tensor):
                layer_cap[slot] = o.detach()
        return _h

    if blocks is not None:
        for slot, idx_key in [("early", "early_idx"),
                               ("mid",   "mid_idx"),
                               ("late",  "late_idx")]:
            needs = {
                "early": {"layer_early", "hybrid_early_hprime"},
                "mid":   {"layer_mid", "hybrid_mid_lexical", "_mid"},
                "late":  {"layer_late"},
            }[slot]
            if any(k in requested_keys for k in needs):
                idx = layer_info[idx_key]
                if 0 <= idx < n_layers:
                    hooks.append(blocks[idx].register_forward_hook(_make_hook(slot)))

    # Hook ln_f output to capture h_raw.
    # ln_f(x) is the normalized residual stream before lm_head — same as
    # get_hs_small() output.  "raw" refers to pre-region-projection, not
    # pre-layernorm.  Consistent with Stage01B shard convention.
    h_raw_cap = [None]
    if "h_raw" in requested_keys and hasattr(backbone, "ln_f"):
        def _hook_ln_f(mod, inp, out):
            h_raw_cap[0] = out.detach()
        hooks.append(backbone.ln_f.register_forward_hook(_hook_ln_f))

    out_tuple = backbone(src)
    for h in hooks:
        h.remove()

    if not (isinstance(out_tuple, (tuple, list)) and len(out_tuple) >= 4):
        raise RuntimeError(
            "backbone.forward() did not return (lm_lgt, rg_lgt, p_r, h_prime). "
            "Check model architecture.")
    h_prime_bt   = out_tuple[3]                       # (B, T, d)
    h_prime_last = h_prime_bt[:, -1, :].float().cpu() # (B, d)

    res = {}

    if "h_prime" in requested_keys:
        res["h_prime"] = h_prime_last

    if "h_raw" in requested_keys and h_raw_cap[0] is not None:
        res["h_raw"] = h_raw_cap[0][:, -1, :].float().cpu()

    for key, slot in [("layer_early", "early"), ("layer_mid", "mid"),
                      ("layer_late",  "late")]:
        if key in requested_keys and slot in layer_cap:
            res[key] = layer_cap[slot][:, -1, :].float().cpu()

    if "hybrid_early_hprime" in requested_keys and "early" in layer_cap:
        e = F.normalize(layer_cap["early"][:, -1, :].float().cpu(), dim=-1)
        p = F.normalize(h_prime_last, dim=-1)
        res["hybrid_early_hprime"] = F.normalize(torch.cat([e, p], dim=-1), dim=-1)

    # Side channel for hybrid_mid_lexical (the dense pre-rank uses layer_mid)
    if "_mid" in requested_keys and "mid" in layer_cap:
        res["_mid"] = layer_cap["mid"][:, -1, :].float().cpu()

    return res


def get_shard_repr(shard, key, start, end, backbone, layer_info,
                   device, live_batch=64):
    """
    Get (B, d) float32 numpy repr for shard rows [start:end].
    Fast path for h_prime/h_raw (read from shard); live inference for others.
    Returns None if unavailable.
    """
    sl = slice(start, end)
    if key == "h_prime":
        h = shard.get("h_ctx")
        if h is not None:
            return h[sl].float().numpy().astype(np.float32)
    if key == "h_raw":
        h = shard.get("h_raw")
        if h is not None:
            return h[sl].float().numpy().astype(np.float32)

    # Live inference
    live_key = "_mid" if key == "hybrid_mid_lexical" else key
    src_cpu = shard["input_ids"][sl].long()
    bufs = []
    for bs in range(0, end - start, live_batch):
        be = min(bs + live_batch, end - start)
        src_b = src_cpu[bs:be].to(device)
        reps  = extract_repr_live(backbone, src_b, {live_key}, layer_info)
        v = reps.get(live_key)
        if v is None:
            return None
        bufs.append(v.numpy().astype(np.float32))
    return np.concatenate(bufs, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Dense index
# ══════════════════════════════════════════════════════════════════════════════

def build_dense_index(key, train_dir, backbone, layer_info,
                      device, max_index_rows, retrieval_backend="auto",
                      live_batch=64):
    """
    Collect train representations and build cosine index.
    Returns index_data dict or raises RuntimeError if key unavailable.
    """
    shards = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    if not shards:
        raise RuntimeError(f"No train shards in {train_dir}")

    all_vecs, all_golds, all_rids = [], [], []
    total = 0
    t0 = time.time()
    extract_key = "_mid" if key == "hybrid_mid_lexical" else key

    for si, sp in enumerate(shards):
        if max_index_rows and total >= max_index_rows:
            break
        s = torch.load(sp, map_location="cpu", weights_only=True)
        N_full = s["gold_token"].shape[0]
        N = min(N_full, max_index_rows - total) if max_index_rows else N_full

        v = get_shard_repr(s, extract_key, 0, N, backbone, layer_info,
                           device, live_batch)
        if v is None:
            raise RuntimeError(f"Key '{key}' unavailable from shard {si}. "
                               f"Layer discovery may have failed or key unsupported.")

        all_vecs.append(v)
        all_golds.append(s["gold_token"][:N].numpy().astype(np.int32))
        rid = s["row_id"][:N].numpy().astype(np.int64) if "row_id" in s \
              else np.arange(total, total + N, dtype=np.int64)
        all_rids.append(rid)
        total += N

        if si % 50 == 0 or (max_index_rows and total >= max_index_rows):
            print(f"  [build_index:{key}] shard {si}/{len(shards)-1}"
                  f"  rows={total:,}  t={time.time()-t0:.0f}s")

    vecs  = np.concatenate(all_vecs,  axis=0)
    golds = np.concatenate(all_golds, axis=0)
    rids  = np.concatenate(all_rids,  axis=0)

    if max_index_rows and len(vecs) > max_index_rows:
        sel   = np.linspace(0, len(vecs) - 1, max_index_rows, dtype=int)
        vecs  = vecs[sel]; golds = golds[sel]; rids = rids[sel]

    norms  = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    normed = (vecs / norms).astype(np.float32)
    if not np.isfinite(normed).all():
        raise RuntimeError(f"Non-finite values in '{key}' vectors after normalization.")

    N, D = normed.shape

    # Quick self-rank-1 sanity
    n_probe   = min(100, N)
    probe_i   = np.random.RandomState(42).choice(N, n_probe, replace=False)
    pv        = torch.tensor(normed[probe_i])
    av        = torch.tensor(normed)
    rank1_ct  = 0
    for i in range(0, n_probe, 64):
        sims = pv[i:i+64] @ av.T
        tops = sims.argmax(1).numpy()
        rank1_ct += int(sum(int(t) == int(p) for t, p in
                            zip(tops, probe_i[i:i+64])))
    rank1_rate = rank1_ct / n_probe
    print(f"  [build_index:{key}] N={N:,}  D={D}  self_rank1={rank1_rate:.4f}")

    faiss_idx = None
    use_faiss = _HAS_FAISS and retrieval_backend in ("auto", "faiss")
    if use_faiss:
        faiss_idx = _faiss.IndexFlatIP(D)
        faiss_idx.add(normed)

    return {
        "key": key, "normed": normed, "golds": golds, "row_ids": rids,
        "N": N, "D": D, "use_faiss": use_faiss, "faiss": faiss_idx,
        "self_rank1": rank1_rate,
    }


def dense_retrieve(q_vecs, index, K, chunk_size, device):
    """
    q_vecs: (B, d) float32, un-normalized.
    Returns (golds: (B,K) int32, scores: (B,K) float32, positions: (B,K) int32).
    """
    B = len(q_vecs)
    norms = np.linalg.norm(q_vecs, axis=1, keepdims=True).clip(min=1e-8)
    q_n   = (q_vecs / norms).astype(np.float32)

    if index["use_faiss"] and index["faiss"] is not None:
        sc, ii = index["faiss"].search(q_n, K)
        N = index["N"]
        ii_safe = np.clip(ii, 0, N - 1)
        golds = np.where(ii >= 0, index["golds"][ii_safe], np.int32(-1)).astype(np.int32)
        return golds, sc.astype(np.float32), ii.astype(np.int32)

    N_idx = index["N"]
    idx_t = torch.tensor(index["normed"], device=device)
    q_t   = torch.tensor(q_n, device=device)
    top_v = torch.full((B, K), -1e9, device=device)
    top_i = torch.zeros((B, K), dtype=torch.long, device=device)

    for cs in range(0, N_idx, chunk_size):
        ce    = min(cs + chunk_size, N_idx)
        sim_c = q_t @ idx_t[cs:ce].T
        cat_v = torch.cat([top_v, sim_c], dim=1)
        cat_i = torch.cat([
            top_i,
            torch.arange(cs, ce, device=device).unsqueeze(0).expand(B, -1)
        ], dim=1)
        top_v, sel = cat_v.topk(K, dim=1)
        top_i = cat_i.gather(1, sel)

    ii = top_i.cpu().numpy()
    sc = top_v.cpu().numpy()
    return index["golds"][ii].astype(np.int32), sc.astype(np.float32), ii.astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Lexical index
# ══════════════════════════════════════════════════════════════════════════════

def build_lexical_index(train_dir, max_postings, max_rows=None):
    """Build inverted token index over train input_ids (last 32 tokens)."""
    shards = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    # Shuffle so common-token posting lists sample uniformly across the corpus,
    # not just the earliest shards (which would bias retrieval toward early text).
    random.Random(42).shuffle(shards)
    inverted     = defaultdict(list)   # tok_id -> [meta_idx, ...]
    row_meta     = []                  # [{gold, set16, set32}, ...]
    row_id_to_meta = {}
    total = 0
    t0 = time.time()

    for si, sp in enumerate(shards):
        if max_rows and total >= max_rows:
            break
        s = torch.load(sp, map_location="cpu", weights_only=True)
        N = min(s["gold_token"].shape[0],
                max_rows - total if max_rows else s["gold_token"].shape[0])
        ids  = s["input_ids"][:N].numpy()
        glds = s["gold_token"][:N].numpy()
        rids = (s["row_id"][:N].numpy() if "row_id" in s
                else np.arange(total, total + N))

        for j in range(N):
            mi   = len(row_meta)
            set16 = frozenset(int(x) for x in ids[j, -16:])
            set32 = frozenset(int(x) for x in ids[j, -32:])
            row_meta.append({"gold": int(glds[j]), "set16": set16, "set32": set32})
            row_id_to_meta[int(rids[j])] = mi
            for tok in set32:
                lst = inverted[tok]
                if len(lst) < max_postings:
                    lst.append(mi)

        total += N
        if si % 100 == 0:
            print(f"  [lex_index] shard={si}/{len(shards)-1}  rows={total:,}"
                  f"  t={time.time()-t0:.0f}s")

    print(f"  [lex_index] done  rows={total:,}  vocab_entries={len(inverted):,}")
    return {"inverted": dict(inverted), "row_meta": row_meta,
            "row_id_to_meta": row_id_to_meta, "n_rows": total}


def lexical_retrieve_single(id_row, lex_data, last_k, K, max_cands):
    inverted  = lex_data["inverted"]
    row_meta  = lex_data["row_meta"]
    query_set = frozenset(int(x) for x in id_row[-last_k:])

    counts = Counter()
    for tok in query_set:
        for mi in inverted.get(tok, []):
            counts[mi] += 1

    if not counts:
        return np.full(K, -1, dtype=np.int32), np.zeros(K, dtype=np.float32)

    cands = (sorted(counts, key=counts.__getitem__, reverse=True)[:max_cands]
             if len(counts) > max_cands else list(counts))

    scored = []
    for mi in cands:
        cand_set = row_meta[mi]["set32"] if last_k >= 32 else row_meta[mi]["set16"]
        union    = len(query_set | cand_set)
        jaccard  = counts[mi] / max(union, 1)
        scored.append((jaccard, mi))
    scored.sort(reverse=True)

    golds  = np.full(K, -1, dtype=np.int32)
    scores = np.zeros(K, dtype=np.float32)
    for rank, (sc, mi) in enumerate(scored[:K]):
        golds[rank]  = row_meta[mi]["gold"]
        scores[rank] = float(sc)
    return golds, scores


def lexical_retrieve_batch(id_arrays, lex_data, last_k, K, max_cands):
    B = len(id_arrays)
    golds  = np.full((B, K), -1, dtype=np.int32)
    scores = np.zeros((B, K), dtype=np.float32)
    for bi, row in enumerate(id_arrays):
        g, s = lexical_retrieve_single(row, lex_data, last_k, K, max_cands)
        golds[bi] = g; scores[bi] = s
    return golds, scores


def hybrid_retrieve_batch(q_vecs, id_arrays, dense_index, lex_data,
                          K, chunk_size, device, lexical_weight,
                          n_prerank=256):
    """
    Dense pre-rank n_prerank via layer_mid, then rerank by
    final_score = dense_cosine + lexical_weight * jaccard(last32).
    """
    _, dense_sc, positions = dense_retrieve(q_vecs, dense_index, n_prerank,
                                             chunk_size, device)
    B          = len(q_vecs)
    row_ids    = dense_index["row_ids"]
    row_meta   = lex_data["row_meta"]
    r2m        = lex_data["row_id_to_meta"]

    golds_out  = np.full((B, K), -1, dtype=np.int32)
    scores_out = np.zeros((B, K), dtype=np.float32)

    for bi in range(B):
        query_set = frozenset(int(x) for x in id_arrays[bi][-32:])
        reranked  = []
        for rank in range(n_prerank):
            pos = int(positions[bi, rank])
            if pos < 0 or pos >= len(row_ids):
                continue
            global_rid = int(row_ids[pos])
            mi = r2m.get(global_rid)
            if mi is not None:
                cand_set = row_meta[mi]["set32"]
                union    = len(query_set | cand_set)
                jaccard  = len(query_set & cand_set) / max(union, 1)
                gold     = row_meta[mi]["gold"]
            else:
                jaccard  = 0.0
                gold     = int(dense_index["golds"][pos])
            final_sc = float(dense_sc[bi, rank]) + lexical_weight * jaccard
            reranked.append((final_sc, gold))
        reranked.sort(reverse=True)
        for out_rank, (sc, g) in enumerate(reranked[:K]):
            golds_out [bi, out_rank] = g
            scores_out[bi, out_rank] = sc

    return golds_out, scores_out


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_row_metrics(golds, base_topk, nbr_golds, nbr_scores, tau, top_k):
    """
    golds      (N,) int32
    base_topk  (N, top_k) int32
    nbr_golds  (N, K) int32
    nbr_scores (N, K) float32
    Returns dict of (N,) arrays for later aggregation.
    """
    N      = len(golds)
    K      = nbr_golds.shape[1]
    golds  = np.asarray(golds, dtype=np.int32)
    base   = np.asarray(base_topk, dtype=np.int32)
    ng     = np.asarray(nbr_golds, dtype=np.int32)
    ns     = np.asarray(nbr_scores, dtype=np.float32)

    gold_in_base = (base == golds[:, None]).any(1)
    valid_mask   = ng >= 0
    gold_in_nbr  = (valid_mask & (ng == golds[:, None])).any(1)
    ret_added    = ~gold_in_base & gold_in_nbr
    base_top1    = base[:, 0]
    base_correct = base_top1 == golds

    # Softmax-weighted support
    ns_safe   = np.where(valid_mask, ns, -1e9).astype(np.float64)
    ns_shifted = ns_safe - ns_safe.max(1, keepdims=True)
    exp_s     = np.exp(ns_shifted / tau)
    exp_s_m   = exp_s * valid_mask.astype(np.float64)
    denom     = exp_s_m.sum(1, keepdims=True)
    denom     = np.where(denom > 0, denom, 1.0)
    weights   = exp_s_m / denom

    sup_gold  = (weights * (ng == golds[:, None]).astype(np.float64)).sum(1)
    sup_top1  = (weights * (ng == base_top1[:, None]).astype(np.float64)).sum(1)
    sup_margin = sup_gold - sup_top1

    # Unique neighbors + entropy
    unique_ct = np.array([len(set(int(g) for g in row if g >= 0)) for row in ng])
    entropy_a = np.zeros(N, dtype=np.float32)
    for i, row in enumerate(ng):
        valid = [int(g) for g in row if g >= 0]
        if not valid:
            continue
        cnt   = Counter(valid)
        total = len(valid)
        entropy_a[i] = float(-sum(
            (c / total) * np.log2(max(c / total, 1e-10)) for c in cnt.values()))

    return {
        "n":               np.ones(N, dtype=np.int32),
        "gold_in_base":    gold_in_base.astype(np.float32),
        "gold_in_nbr":     gold_in_nbr.astype(np.float32),
        "ret_added":       ret_added.astype(np.float32),
        "base_correct":    base_correct.astype(np.float32),
        "sup_gold":        sup_gold.astype(np.float32),
        "sup_top1":        sup_top1.astype(np.float32),
        "sup_margin":      sup_margin.astype(np.float32),
        "unique_nbr":      unique_ct.astype(np.float32),
        "nbr_entropy":     entropy_a,
        "nbr_top1_match":  (ng[:, 0] == golds).astype(np.float32),
        # Masks for sub-populations
        "_base_wrong":     (~base_correct).astype(np.float32),
        "_base_miss":      (~gold_in_base).astype(np.float32),
    }


def aggregate(arrs_list):
    """Concatenate list of per-batch dicts into global means/rates."""
    if not arrs_list:
        raise RuntimeError("aggregate() called with empty batch list — "
                           "no val rows were processed for this key.")
    combined = {}
    for d in arrs_list:
        for k, v in d.items():
            if k not in combined:
                combined[k] = []
            combined[k].append(v)
    merged = {k: np.concatenate(v) for k, v in combined.items()}

    n         = merged["n"].sum()
    n_wrong   = merged["_base_wrong"].sum()
    n_miss    = merged["_base_miss"].sum()

    def _mean(key):
        return float(merged[key].mean()) if n > 0 else 0.0

    def _cond_mean(key, mask_key):
        mask = merged[mask_key].astype(bool)
        arr  = merged[key][mask]
        return float(arr.mean()) if len(arr) > 0 else 0.0

    def _cond_rate(bool_key, mask_key):
        mask = merged[mask_key].astype(bool)
        return float(merged[bool_key][mask].mean()) if mask.any() else 0.0

    return {
        "n_rows":                            int(n),
        "base_top1_acc":                     _mean("base_correct"),
        "gold_in_base_top_k":                _mean("gold_in_base"),
        "gold_in_neighbors":                 _mean("gold_in_nbr"),
        "retrieval_added_gold":              _mean("ret_added"),
        "gold_in_base_or_neighbors":         float((merged["gold_in_base"].astype(bool) | merged["gold_in_nbr"].astype(bool)).mean()),
        "mean_support_gold":                 _mean("sup_gold"),
        "mean_support_base_top1":            _mean("sup_top1"),
        "mean_support_margin":               _mean("sup_margin"),
        "pct_support_gold_gt_top1":          float((merged["sup_margin"] > 0).mean()),
        "mean_unique_nbr_tokens":            _mean("unique_nbr"),
        "nbr_token_entropy_mean":            _mean("nbr_entropy"),
        "nbr_top1_gold_match_rate":          _mean("nbr_top1_match"),
        "n_base_wrong":                      int(n_wrong),
        "n_base_miss":                       int(n_miss),
        "gold_in_nbr_base_wrong":            _cond_rate("gold_in_nbr",   "_base_wrong"),
        "ret_added_base_wrong":              _cond_rate("ret_added",      "_base_wrong"),
        "ret_added_base_miss":               _cond_rate("ret_added",      "_base_miss"),
        "pct_margin_gt0_base_wrong":         float(
            (merged["sup_margin"][merged["_base_wrong"].astype(bool)] > 0).mean()
            if n_wrong > 0 else 0.0),
        "mean_sup_margin_base_wrong":        _cond_mean("sup_margin",    "_base_wrong"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Examples
# ══════════════════════════════════════════════════════════════════════════════

def _decode(ids, tokenizer):
    try:
        return tokenizer.decode(ids)
    except Exception:
        return str(ids)


def generate_examples(val_dir, key, dense_index, lex_data, backbone,
                      layer_info, device, emb_w, tokenizer, args, n=20):
    """Return markdown string with diagnostic examples."""
    shards = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not shards:
        return "_No val shards found._\n"

    lines = [f"# Examples — key: {key}\n"]
    collected = 0
    K = args.num_neighbors

    for sp in shards:
        if collected >= n:
            break
        s = torch.load(sp, map_location="cpu", weights_only=True)
        N = min(s["gold_token"].shape[0], n - collected)

        q_vecs = _get_batch_repr_for_eval(
            s, key, 0, N, backbone, layer_info, device)
        # q_vecs is None for lexical keys (they use raw input_ids) — that's fine.
        # Only bail out for dense/hybrid keys that should have produced a vector.
        if q_vecs is None and key not in _LEXICAL_KEYS and key != _HYBRID_LEXICAL:
            return f"_Key '{key}' unavailable for examples._\n"

        id_arrays = s["input_ids"][:N].numpy()
        golds_arr = s["gold_token"][:N].numpy().astype(np.int32)
        base_topk = s["base_topk_ids"][:N].numpy().astype(np.int32)

        nbr_golds, nbr_scores = _retrieve_for_key(
            key, q_vecs, id_arrays, dense_index, lex_data, args, device)

        for i in range(N):
            ctx_tail = _decode(id_arrays[i, -20:].tolist(), tokenizer)
            gold_str = _decode([int(golds_arr[i])], tokenizer)
            top5_str = " | ".join(
                _decode([int(t)], tokenizer).strip() for t in base_topk[i, :5])
            nbr5_toks = " | ".join(
                _decode([int(nbr_golds[i, r])], tokenizer).strip()
                for r in range(min(5, K)) if nbr_golds[i, r] >= 0)
            nbr5_sc   = " | ".join(
                f"{nbr_scores[i, r]:.3f}"
                for r in range(min(5, K)) if nbr_golds[i, r] >= 0)
            gold_in_nbr = bool((nbr_golds[i] == int(golds_arr[i])).any())
            gold_in_top = bool((base_topk[i] == int(golds_arr[i])).any())
            lines.append(
                f"## Example {collected + 1}\n"
                f"**ctx_tail**: `{ctx_tail.strip()}`\n"
                f"**gold**: `{gold_str.strip()}`  "
                f"in_base_top{args.top_k}={gold_in_top}  "
                f"gold_in_neighbors={gold_in_nbr}\n"
                f"**base_top5**: {top5_str}\n"
                f"**nbr_top5_tokens**: {nbr5_toks}\n"
                f"**nbr_top5_scores**: {nbr5_sc}\n"
            )
            collected += 1
            if collected >= n:
                break

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Shared retrieve dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def _get_batch_repr_for_eval(shard, key, start, end, backbone, layer_info, device):
    """Get (B, d) float32 numpy for a val shard slice. None if unavailable."""
    if key == "hybrid_mid_lexical":
        return get_shard_repr(shard, "hybrid_mid_lexical", start, end,
                              backbone, layer_info, device)
    if key in _LEXICAL_KEYS:
        return None  # lexical uses raw input_ids, not repr vectors
    return get_shard_repr(shard, key, start, end, backbone, layer_info, device)


def _retrieve_for_key(key, q_vecs, id_arrays, dense_index, lex_data, args, device):
    """Dispatch to the appropriate retrieval function."""
    K = args.num_neighbors
    if key == "lexical_last16":
        return lexical_retrieve_batch(
            id_arrays, lex_data, 16, K, args.max_lexical_candidates_per_query)
    if key == "lexical_last32":
        return lexical_retrieve_batch(
            id_arrays, lex_data, 32, K, args.max_lexical_candidates_per_query)
    if key == "hybrid_mid_lexical":
        return hybrid_retrieve_batch(
            q_vecs, id_arrays, dense_index, lex_data,
            K, args.retrieval_chunk_size, device,
            args.lexical_weight)
    # Dense keys
    golds, scores, _ = dense_retrieve(
        q_vecs, dense_index, K, args.retrieval_chunk_size, device)
    return golds, scores


# ══════════════════════════════════════════════════════════════════════════════
# Audit one key
# ══════════════════════════════════════════════════════════════════════════════

def audit_key(key, val_dir, dense_index, lex_data, backbone,
              layer_info, device, args):
    """Evaluate one retrieval key over all val shards. Returns metrics dict."""
    shards = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not shards:
        raise RuntimeError(f"No val shards in {val_dir}")

    tau = args.support_tau_lexical if key in _LEXICAL_KEYS else args.support_tau_dense
    all_batches = []
    total_n = 0
    t0 = time.time()

    for sp in shards:
        if args.max_val_rows and total_n >= args.max_val_rows:
            break
        s = torch.load(sp, map_location="cpu", weights_only=True)
        N_full = s["gold_token"].shape[0]
        N = min(N_full, args.max_val_rows - total_n) if args.max_val_rows else N_full

        golds_full   = s["gold_token"][:N].numpy().astype(np.int32)
        base_topk_full = s["base_topk_ids"][:N].numpy().astype(np.int32)
        id_arrays_full = s["input_ids"][:N].numpy()

        for bs in range(0, N, args.query_batch_size):
            be = min(bs + args.query_batch_size, N)
            B  = be - bs

            golds_b   = golds_full[bs:be]
            base_topk_b = base_topk_full[bs:be]
            id_arrays_b = id_arrays_full[bs:be]

            if key in _LEXICAL_KEYS:
                q_vecs = None
            else:
                q_vecs = _get_batch_repr_for_eval(
                    s, key, bs, be, backbone, layer_info, device)
                if q_vecs is None:
                    raise RuntimeError(
                        f"Key '{key}' unavailable. Layer discovery failed or "
                        f"h_raw not saved in shards.")

            nbr_golds, nbr_scores = _retrieve_for_key(
                key, q_vecs, id_arrays_b, dense_index, lex_data, args, device)

            batch_metrics = compute_row_metrics(
                golds_b, base_topk_b, nbr_golds, nbr_scores, tau, args.top_k)
            all_batches.append(batch_metrics)

        total_n += N
        print(f"  [audit:{key}] rows={total_n:,}  t={time.time()-t0:.0f}s")

    return aggregate(all_batches)


# ══════════════════════════════════════════════════════════════════════════════
# Ranking
# ══════════════════════════════════════════════════════════════════════════════

def rank_keys(all_metrics):
    """
    Rank keys by:
      1. ret_added_base_miss  (primary)
      2. pct_margin_gt0_base_wrong  (secondary)
      3. gold_in_nbr_base_wrong  (tertiary)
      4. retrieval_added_gold  (global fallback)
    """
    def _score(m):
        return (
            m.get("ret_added_base_miss", 0.0),
            m.get("pct_margin_gt0_base_wrong", 0.0),
            m.get("gold_in_nbr_base_wrong", 0.0),
            m.get("retrieval_added_gold", 0.0),
        )
    return sorted(all_metrics, key=lambda x: _score(x[1]), reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = _parse()
    os.makedirs(args.output_root, exist_ok=True)
    keys_requested = [k.strip() for k in args.keys.split(",") if k.strip()]
    unknown = set(keys_requested) - set(_ALL_KEYS)
    if unknown:
        print(f"[WARN] Unknown keys: {unknown}. Valid: {_ALL_KEYS}")
        keys_requested = [k for k in keys_requested if k in set(_ALL_KEYS)]

    print("=" * 70)
    print(f" Retrieval Evidence Sweep — {len(keys_requested)} keys")
    print(f" output_root: {args.output_root}")
    print(f" keys: {keys_requested}")
    print("=" * 70)

    # ── Preflight checks ──
    for f in [args.small_ckpt]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")
    for d in [args.train_dir, args.val_dir]:
        shards = glob.glob(os.path.join(d, "shard_*.pt"))
        if not shards:
            raise RuntimeError(f"No shard_*.pt in {d}")
        print(f"  [OK] {d}  ({len(shards)} shards)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  device: {device}  FAISS: {_HAS_FAISS}")

    # ── Load backbone ──
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    emb_w = backbone.token_emb.weight.detach().float().cpu()
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    # ── Layer discovery ──
    layer_info = make_layer_info(backbone, d_model)
    if layer_info["blocks"] is not None:
        print(f"  transformer blocks found: {layer_info['n_layers']} layers  "
              f"early_idx={layer_info['early_idx']}  "
              f"mid_idx={layer_info['mid_idx']}  "
              f"late_idx={layer_info['late_idx']}")
    else:
        print("  [WARN] Could not find transformer block list. "
              "Layer keys will be unavailable.")
        unavail = _NEEDS_LIVE - {"h_raw"}
        unavail_lex = set()
        keys_requested = [k for k in keys_requested
                          if k not in unavail or k in _LEXICAL_KEYS]
        print(f"  Dropping layer/hybrid keys: {unavail}")

    # ── Tokenizer (for examples) ──
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e30)
    except Exception:
        tokenizer = None
        print("  [WARN] Could not load GPT2 tokenizer; examples will use raw IDs.")

    # ── Build lexical index (once, shared) ──
    needs_lex = any(k in {"lexical_last16", "lexical_last32", "hybrid_mid_lexical"}
                    for k in keys_requested)
    lex_data = None
    if needs_lex:
        print("\n[lex_index] Building lexical index over train ...")
        lex_data = build_lexical_index(
            args.train_dir, args.max_postings_per_token,
            max_rows=args.max_index_rows or None)

    # ── Save config ──
    cfg = vars(args)
    cfg["keys_requested"] = keys_requested
    cfg["device"] = str(device)
    cfg["faiss_available"] = _HAS_FAISS
    cfg["n_layers"] = layer_info["n_layers"]
    cfg["baseline_added_gold"] = _BASELINE_ADDED_GOLD
    with open(os.path.join(args.output_root, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # ── Per-key evaluation ──
    all_results = []      # [(key, metrics_dict)]
    skipped     = []

    for key in keys_requested:
        print(f"\n{'='*60}")
        print(f" Key: {key}")
        print(f"{'='*60}")
        key_dir = os.path.join(args.output_root, "keys", key)
        os.makedirs(key_dir, exist_ok=True)

        try:
            # Build index (or use lex_data)
            dense_index = None
            if key not in _LEXICAL_KEYS:
                print(f"  Building train index for '{key}' ...")
                dense_index = build_dense_index(
                    key, args.train_dir, backbone, layer_info,
                    device, args.max_index_rows, args.retrieval_backend)

            # Evaluate
            print(f"  Evaluating on val ...")
            metrics = audit_key(key, args.val_dir, dense_index, lex_data,
                                 backbone, layer_info, device, args)

            # Augment with key name and baseline delta
            metrics["key"] = key
            metrics["vs_baseline_ret_added"] = (
                metrics.get("retrieval_added_gold", 0.0) - _BASELINE_ADDED_GOLD)
            metrics["vs_baseline_ret_added_miss"] = (
                metrics.get("ret_added_base_miss", 0.0) - _BASELINE_ADDED_GOLD)

            print(f"\n  === {key} ===")
            for m in ["gold_in_neighbors", "retrieval_added_gold",
                      "ret_added_base_miss", "pct_margin_gt0_base_wrong"]:
                v = metrics.get(m, "N/A")
                print(f"  {m:40s} = {v:.4f}" if isinstance(v, float) else
                      f"  {m:40s} = {v}")

            # Save metrics
            with open(os.path.join(key_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            # Generate examples
            if tokenizer is not None:
                print(f"  Generating examples ...")
                ex_txt = generate_examples(
                    args.val_dir, key, dense_index, lex_data,
                    backbone, layer_info, device, emb_w, tokenizer, args,
                    n=args.num_examples)
                with open(os.path.join(key_dir, "examples.md"), "w",
                          encoding="utf-8") as f:
                    f.write(ex_txt)

            all_results.append((key, metrics))

        except RuntimeError as e:
            print(f"  [SKIP] key='{key}' unavailable: {e}")
            skipped.append({"key": key, "error": str(e)})
            with open(os.path.join(key_dir, "metrics.json"), "w") as f:
                json.dump({"key": key, "available": False, "error": str(e)}, f, indent=2)

    # ── Rank and summarize ──
    ranked = rank_keys(all_results)

    print("\n" + "=" * 70)
    print(" RANKING (by ret_added_base_miss, then margin, then global)")
    print("=" * 70)
    header = (f"{'rank':>4}  {'key':25s}  {'ret_add_miss':>12}  "
              f"{'ret_add_global':>14}  {'margin_gt0_wrong':>16}  "
              f"{'gold_in_nbr':>11}")
    print(header)
    for rank, (key, m) in enumerate(ranked, 1):
        print(f"  {rank:2d}.  {key:25s}  "
              f"{m.get('ret_added_base_miss', 0):12.4f}  "
              f"{m.get('retrieval_added_gold', 0):14.4f}  "
              f"{m.get('pct_margin_gt0_base_wrong', 0):16.4f}  "
              f"{m.get('gold_in_neighbors', 0):11.4f}")
    print(f"\n  h_prime baseline: ret_added_global={_BASELINE_ADDED_GOLD:.4f}")

    # ── Write CSV summary ──
    csv_path = os.path.join(args.output_root, "sweep_summary.csv")
    fieldnames = [
        "rank", "key", "n_rows", "base_top1_acc", "gold_in_base_top_k",
        "gold_in_neighbors", "retrieval_added_gold", "ret_added_base_miss",
        "ret_added_base_wrong", "pct_margin_gt0_base_wrong",
        "mean_sup_margin_base_wrong", "mean_unique_nbr_tokens",
        "nbr_token_entropy_mean", "vs_baseline_ret_added",
        "vs_baseline_ret_added_miss", "self_rank1",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rank, (key, m) in enumerate(ranked, 1):
            row = dict(m, rank=rank)
            if hasattr(m, "get"):
                dense_idx_self = m.get("_self_rank1", "N/A")
            row["self_rank1"] = "N/A"
            w.writerow(row)
    print(f"\n  CSV → {csv_path}")

    # ── Write markdown report ──
    md_lines = [
        "# Retrieval Evidence Sweep\n",
        f"output_root: `{args.output_root}`\n",
        f"data_root: `{args.data_root}`\n",
        f"n_keys_evaluated: {len(all_results)}  |  "
        f"n_skipped: {len(skipped)}\n",
        f"h_prime_baseline_added_gold: {_BASELINE_ADDED_GOLD}\n",
        "\n## Ranking\n",
        "| rank | key | ret_add_miss | ret_add_global | margin_gt0_wrong |"
        " gold_in_nbr | vs_baseline |\n",
        "|------|-----|-------------|----------------|-----------------|"
        "------------|------------|\n",
    ]
    for rank, (key, m) in enumerate(ranked, 1):
        delta = m.get("vs_baseline_ret_added", 0.0)
        md_lines.append(
            f"| {rank} | {key} "
            f"| {m.get('ret_added_base_miss', 0):.4f} "
            f"| {m.get('retrieval_added_gold', 0):.4f} "
            f"| {m.get('pct_margin_gt0_base_wrong', 0):.4f} "
            f"| {m.get('gold_in_neighbors', 0):.4f} "
            f"| {delta:+.4f} |\n"
        )
    if skipped:
        md_lines.append("\n## Skipped Keys\n")
        for s in skipped:
            md_lines.append(f"- **{s['key']}**: {s['error']}\n")

    md_lines.append("\n## Interpretation\n")
    if ranked:
        best_key, best_m = ranked[0]
        delta_miss = best_m.get("ret_added_base_miss", 0) - _BASELINE_ADDED_GOLD
        status = "IMPROVEMENT" if delta_miss > 0.002 else "MARGINAL" if delta_miss > 0 else "NO GAIN"
        md_lines.append(
            f"Best key: **{best_key}**  "
            f"ret_added_base_miss={best_m.get('ret_added_base_miss', 0):.4f}  "
            f"vs h_prime baseline: {delta_miss:+.4f}  [{status}]\n"
        )

    report_md = os.path.join(args.output_root, "report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"  MD  → {report_md}")

    report_json = {
        "config": cfg,
        "ranked_keys": [
            {"rank": rank, "key": key, **m}
            for rank, (key, m) in enumerate(ranked, 1)
        ],
        "skipped_keys": skipped,
        "baseline_added_gold": _BASELINE_ADDED_GOLD,
    }
    with open(os.path.join(args.output_root, "report.json"), "w") as f:
        json.dump(report_json, f, indent=2)

    print("\n[sweep] DONE")


def _parse():
    p = argparse.ArgumentParser(description="Retrieval evidence diagnostic sweep")
    p.add_argument("--small_ckpt",        required=True)
    p.add_argument("--data_root",         default=None,
                   help="Root of the logitfix pipeline output (for reference)")
    p.add_argument("--train_dir",         required=True,
                   help="Patched train shard dir (01_live_dataset_patched/train)")
    p.add_argument("--val_dir",           required=True,
                   help="Patched val shard dir (01_live_dataset_patched/val)")
    p.add_argument("--output_root",       default="runs/retrieval_evidence_sweep_limited500k")
    p.add_argument("--token_to_region",   default=None)
    p.add_argument("--super_map",         default=None)
    p.add_argument("--keys",
                   default=",".join(_ALL_KEYS),
                   help="Comma-separated list of keys to test")
    # Index limits
    p.add_argument("--num_neighbors",     type=int,   default=32)
    p.add_argument("--top_k",            type=int,   default=256)
    p.add_argument("--max_index_rows",   type=int,   default=500000)
    p.add_argument("--max_val_rows",     type=int,   default=0,
                   help="Cap on val rows evaluated per key (0 = all)")
    # Retrieval backend
    p.add_argument("--retrieval_backend",     default="auto",
                   choices=["auto", "faiss", "torch_chunked"])
    p.add_argument("--retrieval_chunk_size",  type=int, default=131072)
    p.add_argument("--query_batch_size",      type=int, default=512)
    # Lexical
    p.add_argument("--max_postings_per_token",         type=int, default=5000)
    p.add_argument("--max_lexical_candidates_per_query", type=int, default=20000)
    p.add_argument("--lexical_weight",                  type=float, default=0.5)
    # Metrics
    p.add_argument("--support_tau_dense",   type=float, default=0.2)
    p.add_argument("--support_tau_lexical", type=float, default=1.0)
    p.add_argument("--num_examples",        type=int,   default=20)
    return p.parse_args()


if __name__ == "__main__":
    main()
