#!/usr/bin/env python3
"""
eval_fuzzy_phrase_recall_v1.py — Phase 2B: Evaluate fuzzy phrase recall.

For each val row, retrieves top-R similar train rows by cosine similarity
(using pre-built key matrices) and votes over their next tokens to score
the candidate pool.

NO training. NO model changes. Gold is only used for metrics after retrieval.

Usage:
  python scripts/eval_fuzzy_phrase_recall_v1.py \
    --val_dir ... --index_dir ... --output_dir ... \
    --modes h_ctx,h_raw,recency_bow,bow --phrase_lens 16,32,64 \
    --retrieval_k_grid 8,16,32,64 --temp_grid 0.05,0.1,0.2,0.5 \
    --lambda_grid 0.0,0.25,0.5,1.0,2.0,4.0 \
    --top_vote_thresholds 0.1,0.2,0.3,0.5 \
    --margin_thresholds 0.0,0.05,0.1,0.2 \
    --min_agreement_grid 1,2,4 \
    --chunk_size 4096 --train_chunk_size 65536 \
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
# Tokenizer
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
# Val shard loading
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

def l2_norm_rows(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(n, _EPS)

def load_val_shards(val_dir, top_k, max_rows=None):
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
        sh  = torch.load(sp, map_location="cpu", weights_only=False)
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
        B, K = topk.shape
        if rids is None: rids = torch.arange(total, total + B)
        if K < top_k:
            topk = torch.cat([topk, torch.zeros(B, top_k-K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, top_k-K), float("nan"))], 1)
        elif K > top_k:
            topk, lgt = topk[:, :top_k], lgt[:, :top_k]
        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, ids, rids = (x[:keep] for x in (topk, lgt, gold, ids, rids))
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
    print(f"[shards] {total:,} val rows  K={top_k}")
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Map loading
# ─────────────────────────────────────────────────────────────────────────────

def load_maps(t2r_path, super_path=None):
    with open(t2r_path) as f: raw = json.load(f)
    t2r = ({i: v for i, v in enumerate(raw) if v is not None} if isinstance(raw, list)
           else {int(k): v for k, v in raw.items()})
    unk_r = int(max(t2r.values())) + 1 if t2r else 1
    r2s = {}; unk_s = 1; sr = False
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f: raw2 = json.load(f)
        r2s = ({int(k): v for k, v in raw2.items()} if not isinstance(raw2, list)
               else {i: v for i, v in enumerate(raw2) if v is not None})
        unk_s = int(max(r2s.values())) + 1 if r2s else 1
        sr = True
    V = max(t2r.keys(), default=0) + 2
    tok_arr = np.full(V, unk_r, dtype=np.int32)
    for t, r in t2r.items():
        if 0 <= t < V: tok_arr[t] = int(r)
    R = unk_r + 2
    reg_arr = np.full(R, unk_s, dtype=np.int32)
    for r, s in r2s.items():
        if 0 <= r < R: reg_arr[int(r)] = int(s)
    return tok_arr, reg_arr, unk_r, unk_s, sr

# ─────────────────────────────────────────────────────────────────────────────
# Index loading  (reads modes_built from build_stats.json, not config.json)
# ─────────────────────────────────────────────────────────────────────────────

def load_index(index_dir, requested_modes, phrase_lens, data_has_hctx, data_has_hraw):
    cfg_path   = os.path.join(index_dir, "config.json")
    stats_path = os.path.join(index_dir, "build_stats.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"index config.json not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    N     = int(cfg["num_train_rows"])
    h_dim = cfg.get("h_dim")

    # modes_built lives in build_stats.json (not config.json)
    modes_built = cfg.get("modes_built", [])   # may be present if old format
    if not modes_built and os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        modes_built = stats.get("modes_built", [])
        h_dim = h_dim or stats.get("h_dim")
    if not modes_built:
        raise RuntimeError(
            f"modes_built is empty. build_stats.json={stats_path} — "
            "was the index built successfully?")

    gold_arr = np.load(os.path.join(index_dir, "values_gold_token.npy"))
    rowid_arr = np.load(os.path.join(index_dir, "row_id.npy"))
    if len(gold_arr) != N:
        raise RuntimeError(f"gold_arr length {len(gold_arr)} != N={N}")
    print(f"[index] N={N:,}  h_dim={h_dim}  modes_built={modes_built}")

    feature_dim = cfg.get("feature_dim", 8192)
    keys = {}   # name -> memmap float16 [N, D]

    for name in modes_built:
        skip = False
        # h_ctx / h_raw availability
        if name in ("h_ctx", "h_raw"):
            if name not in requested_modes:
                skip = True
            elif name == "h_ctx" and not data_has_hctx:
                print(f"  [WARN] {name}: in index but val shards lack h_ctx — skipping")
                skip = True
            elif name == "h_raw" and not data_has_hraw:
                print(f"  [WARN] {name}: in index but val shards lack h_raw — skipping")
                skip = True
        elif "bow" in name:
            plen_match = any(f"len{pl}" in name for pl in phrase_lens)
            bow_base   = "recency_bow" if "recency" in name else "bow"
            if bow_base not in requested_modes or not plen_match:
                skip = True
        if skip:
            continue
        mmap_path = os.path.join(index_dir, f"{name}_keys.mmap")
        if not os.path.isfile(mmap_path):
            print(f"  [WARN] memmap not found: {mmap_path} — skipping {name}")
            continue
        D = h_dim if name in ("h_ctx", "h_raw") else feature_dim
        if D is None:
            print(f"  [WARN] unknown dim for {name} — skipping")
            continue
        keys[name] = np.memmap(mmap_path, dtype=np.float16, mode="r", shape=(N, int(D)))
        print(f"  [OK] {name}: shape=({N},{D})")

    if not keys:
        raise RuntimeError(
            "No index keys could be loaded. "
            f"requested_modes={requested_modes}  modes_built={modes_built}  "
            f"val_has_hctx={data_has_hctx}  val_has_hraw={data_has_hraw}. "
            "Check that index was built with matching modes and phrase_lens.")
    return keys, gold_arr, rowid_arr, N, h_dim, cfg

# ─────────────────────────────────────────────────────────────────────────────
# BOW query builder
# ─────────────────────────────────────────────────────────────────────────────

def build_bow_vector(ctx_np, phrase_len, feature_dim, recency=False):
    seq  = len(ctx_np)
    T    = min(phrase_len, seq)
    tail = ctx_np[-T:].astype(np.int64)
    vec  = np.zeros(feature_dim, dtype=np.float32)
    if recency:
        dists   = np.arange(T-1, -1, -1, dtype=np.float32)
        weights = np.exp(-dists / max(T / 4.0, 1.0))
    else:
        weights = np.ones(T, dtype=np.float32)
    bins = tail % feature_dim
    np.add.at(vec, bins, weights)
    nrm = np.linalg.norm(vec)
    if nrm > _EPS: vec /= nrm
    return vec

# ─────────────────────────────────────────────────────────────────────────────
# Batched cosine retrieval  (reads train memmap ONCE per chunk for all val rows)
# ─────────────────────────────────────────────────────────────────────────────

def batch_retrieve_all(query_mat_f32, train_mmap, train_chunk_size, top_k,
                        val_batch_size=4096):
    """
    Scans train memmap in train_chunk_size chunks; for each chunk computes
    similarities for all val queries in val_batch_size sub-batches, then merges
    into a running top-k.  Reads train memmap exactly once total.

    query_mat_f32: [N_val, D] float32, already L2-normalised.
    Returns: indices [N_val, top_k], sims [N_val, top_k] sorted descending.
    """
    N_val   = query_mat_f32.shape[0]
    N_train = train_mmap.shape[0]
    best_sims = np.full((N_val, top_k), -2.0, dtype=np.float32)
    best_idxs = np.zeros((N_val, top_k), dtype=np.int64)

    ptr = 0
    while ptr < N_train:
        end    = min(ptr + train_chunk_size, N_train)
        C      = end - ptr
        chunk  = train_mmap[ptr:end].astype(np.float32)   # [C, D] — one read
        k_this = min(top_k, C)

        vptr = 0
        while vptr < N_val:
            vend   = min(vptr + val_batch_size, N_val)
            B      = vend - vptr
            q      = query_mat_f32[vptr:vend]              # [B, D]
            sims   = q @ chunk.T                           # [B, C]
            ri     = np.arange(B)[:, None]

            # Top-k indices from this chunk
            if C <= k_this:
                c_ord = np.argsort(sims, axis=1)[:, ::-1]
            else:
                c_ord = np.argpartition(sims, -k_this, axis=1)[:, -k_this:]
                c_ord = c_ord[ri, np.argsort(sims[ri, c_ord], axis=1)[:, ::-1]]

            c_sims = sims[ri, c_ord]
            c_idxs = (ptr + c_ord).astype(np.int64)

            # Merge with running best [B, top_k]
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
# Vote computation from neighbors (per row)
# ─────────────────────────────────────────────────────────────────────────────

def compute_votes(neighbor_toks, neighbor_sims, cand_ids, temp):
    """Returns vote_probs [P], agree_cnt [P], top_tok, top_prob, margin, entropy."""
    shifted  = neighbor_sims / max(temp, 1e-6)
    shifted -= shifted.max()
    weights  = np.exp(shifted); weights /= weights.sum() + _EPS
    tok_vote  = defaultdict(float)
    tok_count = defaultdict(int)
    for w, t in zip(weights.tolist(), neighbor_toks.tolist()):
        tok_vote[t]  += w
        tok_count[t] += 1
    P = len(cand_ids)
    vote_probs = np.array([tok_vote.get(int(c), 0.0) for c in cand_ids], dtype=np.float32)
    agree_cnt  = np.array([tok_count.get(int(c), 0)  for c in cand_ids], dtype=np.int32)
    if tok_vote:
        top_tok  = max(tok_vote, key=tok_vote.get)
        top_prob = float(tok_vote[top_tok])
    else:
        top_tok  = -1; top_prob = 0.0
    sorted_vp = np.sort(vote_probs)[::-1]
    margin    = float(sorted_vp[0] - sorted_vp[1]) if P >= 2 else 0.0
    vp_clip   = np.clip(vote_probs, _EPS, None)
    vp_norm   = vp_clip / vp_clip.sum()
    entropy   = float(-(vp_norm * np.log(vp_norm)).sum())
    return vote_probs, agree_cnt, top_tok, top_prob, margin, entropy

# ─────────────────────────────────────────────────────────────────────────────
# Batch surgical edit
# ─────────────────────────────────────────────────────────────────────────────

def batch_surgical_edit(topk_ids, topk_lgt, sel_toks, apply_mask, md_half=0.5):
    """topk_ids [N,K], topk_lgt [N,K], sel_toks [N], apply_mask [N] → final_top1 [N]"""
    N, K = topk_ids.shape
    ar   = np.arange(N)
    ref  = np.where(np.isfinite(topk_lgt), topk_lgt, -1e9).copy()
    ref[:, 0] -= md_half   # penalise base_top1
    edit_rows = np.where(apply_mask)[0]
    if len(edit_rows):
        st  = sel_toks[edit_rows].astype(topk_ids.dtype)
        pos = (topk_ids[edit_rows] == st[:, None]).argmax(1)
        ok  = topk_ids[edit_rows, pos] == st
        ref[edit_rows[ok], pos[ok]] += md_half
    return topk_ids[ar, ref.argmax(1)]

# ─────────────────────────────────────────────────────────────────────────────
# Batch policy application  (all N rows at once — no Python row loop)
# ─────────────────────────────────────────────────────────────────────────────

def apply_policies_batch(policy_names, cand_ids, cand_lgts,
                          vote_probs_all, agree_cnt_all,
                          top_vote_tok, top_vote_prob, vote_margin,
                          gold, topk_ids, topk_lgt, md_half=0.5):
    """
    Returns dict: pname -> (sel_toks [N], apply_mask [N], final_top1 [N])
    policy_names: list of str matching names produced by make_policy_names()
    """
    N, P = cand_ids.shape
    ar   = np.arange(N)
    results = {}

    # Pre-compute: is top_vote_tok in each row's candidate pool?
    tvt_in_pool = (cand_ids == top_vote_tok[:, None]).any(axis=1)   # [N]
    # Find pool idx of top_vote_tok per row (0 if not present — safe because gated by tvt_in_pool)
    tvt_pool_idx = np.where(
        tvt_in_pool,
        (cand_ids == top_vote_tok[:, None]).argmax(axis=1),
        np.zeros(N, dtype=np.intp))

    agree_max = agree_cnt_all.max(axis=1)   # [N]

    for pname in policy_names:
        if pname == "base_candidate":
            sel_idx   = cand_lgts.argmax(axis=1)
            sel_toks  = cand_ids[ar, sel_idx]
            apply     = np.ones(N, dtype=bool)

        elif pname == "fuzzy_only":
            valid    = (top_vote_tok >= 0) & (top_vote_prob > 0) & tvt_in_pool
            base_idx = cand_lgts.argmax(axis=1)
            sel_toks = np.where(valid, top_vote_tok, cand_ids[ar, base_idx]).astype(cand_ids.dtype)
            apply    = valid

        elif pname.startswith("fuzzy_lam"):
            lam_str  = pname[len("fuzzy_lam"):]
            lam      = float(lam_str)
            scores   = cand_lgts + lam * np.log(vote_probs_all + _EPS)
            sel_idx  = scores.argmax(axis=1)
            sel_toks = cand_ids[ar, sel_idx]
            apply    = np.ones(N, dtype=bool)

        elif pname.startswith("gated_"):
            # name format: gated_tv{t}_m{mg}_ag{mac}
            parts = pname.split("_")
            tvt_thresh = float(parts[1][2:])   # "tv0.1" → 0.1
            mg_thresh  = float(parts[2][1:])   # "m0.0"  → 0.0
            mac_thresh = int(parts[3][2:])      # "ag1"   → 1
            gate     = (top_vote_prob >= tvt_thresh) & (vote_margin >= mg_thresh) \
                       & (agree_max >= mac_thresh) & (top_vote_tok >= 0) & tvt_in_pool
            base_idx = cand_lgts.argmax(axis=1)
            sel_toks = np.where(gate, top_vote_tok, cand_ids[ar, base_idx]).astype(cand_ids.dtype)
            apply    = gate

        elif pname == "oracle_DIAGNOSTIC":
            # Apply only when top voted token in pool AND equals gold
            valid    = (top_vote_tok >= 0) & tvt_in_pool
            sel_toks = np.where(valid, top_vote_tok,
                                cand_ids[ar, cand_lgts.argmax(axis=1)]).astype(cand_ids.dtype)
            apply    = valid & (sel_toks == gold.astype(sel_toks.dtype))

        else:
            # Unknown policy — no-op
            sel_toks = cand_ids[ar, cand_lgts.argmax(axis=1)]
            apply    = np.zeros(N, dtype=bool)

        final_top1 = batch_surgical_edit(topk_ids, topk_lgt, sel_toks, apply, md_half)
        results[pname] = (sel_toks, apply, final_top1)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Policy name list (no closures needed — names encode params)
# ─────────────────────────────────────────────────────────────────────────────

def make_policy_names(lambda_grid, top_vote_thresholds, margin_thresholds, min_agreement_grid):
    names = ["base_candidate", "fuzzy_only"]
    for lam in lambda_grid:
        names.append(f"fuzzy_lam{lam:.2f}")
    for tvt in top_vote_thresholds:
        for mg in margin_thresholds:
            for mac in min_agreement_grid:
                names.append(f"gated_tv{tvt}_m{mg}_ag{mac}")
    names.append("oracle_DIAGNOSTIC")
    return names

# ─────────────────────────────────────────────────────────────────────────────
# Per-row retrieval + vote (returns per-row arrays)
# ─────────────────────────────────────────────────────────────────────────────

def eval_mode(train_keys_mmap, train_gold, query_mat_f32,
              cand_ids, gold, top_k, temp, train_chunk_size, val_batch_size=4096):
    """
    Returns dict of per-row arrays [N]:
      vote_probs_all [N,P], agree_cnt_all [N,P],
      top_vote_tok [N], top_vote_prob [N], vote_margin [N], vote_entropy [N],
      top_sim [N], agree_max [N],
      gold_vote_rank [N], gold_vote_prob [N], gold_supported [N], top_is_gold [N]

    Retrieval is done with batch_retrieve_all(): reads train memmap exactly once
    per chunk for all val queries simultaneously via matmul. O(N_train) reads total.
    """
    N  = query_mat_f32.shape[0]
    P  = cand_ids.shape[1]

    vote_probs_all    = np.zeros((N, P), dtype=np.float32)
    agree_cnt_all     = np.zeros((N, P), dtype=np.int32)
    top_vote_tok_all  = np.full(N, -1, dtype=np.int64)
    top_vote_prob_all = np.zeros(N, dtype=np.float32)
    vote_margin_all   = np.zeros(N, dtype=np.float32)
    vote_entropy_all  = np.zeros(N, dtype=np.float32)
    top_sim_all       = np.zeros(N, dtype=np.float32)
    gold_vote_rank_all= np.full(N, P, dtype=np.int32)
    gold_vote_prob_all= np.zeros(N, dtype=np.float32)
    gold_supported_all= np.zeros(N, dtype=bool)

    t0 = time.time()
    print(f"    [retrieve] batched matmul  N_val={N:,}  N_train={train_keys_mmap.shape[0]:,}"
          f"  top_k={top_k}  val_batch={val_batch_size} ...")
    all_idxs, all_sims = batch_retrieve_all(query_mat_f32, train_keys_mmap,
                                             train_chunk_size, top_k, val_batch_size)
    print(f"    [retrieve] done in {time.time()-t0:.1f}s")

    # Bulk-load all neighbor gold tokens in one fancy-index read: [N, top_k]
    nb_toks_batch = train_gold[all_idxs].astype(np.int64)

    # Per-row voting loop — O(N × top_k), fast (no memmap I/O here)
    for i in range(N):
        nb_toks = nb_toks_batch[i]
        nb_sims = all_sims[i].astype(np.float32)
        vp, ac, tvt, tvp, vm, ent = compute_votes(nb_toks, nb_sims, cand_ids[i], temp)

        vote_probs_all[i]    = vp
        agree_cnt_all[i]     = ac
        top_vote_tok_all[i]  = tvt
        top_vote_prob_all[i] = tvp
        vote_margin_all[i]   = vm
        vote_entropy_all[i]  = ent
        top_sim_all[i]       = float(nb_sims[0]) if len(nb_sims) else 0.0

        g     = int(gold[i])
        match = np.where(cand_ids[i] == g)[0]
        if len(match):
            j = match[0]
            gold_vote_prob_all[i]  = float(vp[j])
            gold_vote_rank_all[i]  = int((vp > vp[j]).sum())
            gold_supported_all[i]  = vp[j] > 0

        if i % 5000 == 0 and i > 0:
            print(f"    [vote] {i:,}/{N:,} ...")

    top_is_gold = (top_vote_tok_all == gold.astype(np.int64))
    return {
        "vote_probs":     vote_probs_all,
        "agree_cnt":      agree_cnt_all,
        "top_vote_tok":   top_vote_tok_all,
        "top_vote_prob":  top_vote_prob_all,
        "vote_margin":    vote_margin_all,
        "vote_entropy":   vote_entropy_all,
        "top_sim":        top_sim_all,
        "agree_max":      agree_cnt_all.max(axis=1),
        "gold_vote_rank": gold_vote_rank_all,
        "gold_vote_prob": gold_vote_prob_all,
        "gold_supported": gold_supported_all,
        "top_is_gold":    top_is_gold,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Metric accumulator
# ─────────────────────────────────────────────────────────────────────────────

class MetricAcc:
    __slots__ = ["n","apply","noop","ctg","caw","base_ok","final_ok",
                 "sip_n","sip_match","gip_n"]
    def __init__(self):
        for s in self.__slots__: setattr(self, s, 0)

    def update(self, gold, base_top1, final_top1, apply_mask, sel_tok, gold_in_pool, subset):
        g=gold[subset]; b=base_top1[subset]; ft=final_top1[subset]
        am=apply_mask[subset]; gip=gold_in_pool[subset]; st=sel_tok[subset]
        self.n+=len(g); self.apply+=int(am.sum()); self.noop+=int((~am).sum())
        self.ctg+=int((am&(ft==g)).sum()); self.caw+=int((am&(b==g)&(ft!=g)).sum())
        self.base_ok+=int((b==g).sum()); self.final_ok+=int((ft==g).sum())
        self.gip_n+=int(gip.sum()); self.sip_n+=int(gip.sum())
        self.sip_match+=int((gip&(st==g)).sum())

    def metrics(self):
        n=max(self.n,1); _E=1e-9
        return {
            "n":self.n, "apply_rate":self.apply/n, "noop_rate":self.noop/n,
            "gold_in_pool_rate":self.gip_n/n, "base_acc":self.base_ok/n,
            "policy_acc":self.final_ok/n, "acc_gain":(self.final_ok-self.base_ok)/n,
            "selected_gold_given_in_pool":self.sip_match/max(self.sip_n,1),
            "changed_to_gold":self.ctg/n, "changed_away":self.caw/n,
            "net_correction":(self.ctg-self.caw)/n,
            "benefit_damage_ratio":self.ctg/max(self.caw,_E),
            "applied_precision_ctg":self.ctg/max(self.apply,1),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Rank accumulator
# ─────────────────────────────────────────────────────────────────────────────

class RankAcc:
    def __init__(self, P):
        self.P=P; self.hist=np.zeros(P+1,dtype=np.int64)
        self.sum_gv=self.sum_tp=self.sum_mg=self.sum_ent=self.sum_sim=self.sum_ag=0.0
        self.n=0; self.n_gs=0; self.n_tig=0

    def update(self, gold_vote_rank, gold_in_pool, gold_vote_prob,
               top_vote_prob, vote_margin, vote_entropy, top_sim, agree_max,
               top_is_gold, gold_supported, subset):
        gvr=gold_vote_rank[subset]; m=int(subset.sum())
        for r in gvr: self.hist[min(r,self.P)]+=1
        self.sum_gv +=float(gold_vote_prob[subset].sum())
        self.sum_tp +=float(top_vote_prob[subset].sum())
        self.sum_mg +=float(vote_margin[subset].sum())
        self.sum_ent+=float(vote_entropy[subset].sum())
        self.sum_sim+=float(top_sim[subset].sum())
        self.sum_ag +=float(agree_max[subset].sum())
        self.n+=m
        self.n_gs +=int(gold_supported[subset].sum())
        self.n_tig+=int(top_is_gold[subset].sum())

    def stats(self):
        n=max(self.n,1)
        return {
            "n":self.n,
            "fuzzy_gold_supported_rate":self.n_gs/n,
            "neighbor_top_token_is_gold_rate":self.n_tig/n,
            "gold_vote_rank_le_1":int(self.hist[:2].sum()),
            "gold_vote_rank_le_3":int(self.hist[:4].sum()),
            "gold_vote_rank_le_5":int(self.hist[:6].sum()),
            "gold_vote_rank_le_10":int(self.hist[:11].sum()),
            "mean_gold_vote":self.sum_gv/n,
            "mean_top_vote_prob":self.sum_tp/n,
            "mean_vote_margin":self.sum_mg/n,
            "mean_vote_entropy":self.sum_ent/n,
            "mean_top_neighbor_similarity":self.sum_sim/n,
            "mean_agreement_count":self.sum_ag/n,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Slice builder
# ─────────────────────────────────────────────────────────────────────────────

def build_slices(gold, base_top1, cand_ids, gold_in_pool, cand_lgts,
                 tok_arr, reg_arr, unk_r, unk_s, sr, inp_ids, memory_len,
                 gold_supported, gold_vote_rank, top_vote_tok,
                 top_vote_prob, agree_max):
    N = len(gold); V = tok_arr.shape[0]; R = reg_arr.shape[0]
    base_wrong = base_top1 != gold
    T   = min(memory_len, inp_ids.shape[1])
    ctx = inp_ids[:, -T:]
    gold_in_ctx = np.array([(ctx[i] == gold[i]).any() for i in range(N)], dtype=bool)
    if cand_lgts.shape[1] >= 2:
        top2        = np.sort(cand_lgts, axis=1)[:, -2]
        base_margin = cand_lgts.max(axis=1) - top2
    else:
        base_margin = np.zeros(N, dtype=np.float32)
    med_margin = np.median(base_margin)
    base_reg = tok_arr[base_top1.clip(0,V-1)]
    gold_reg = tok_arr[gold.clip(0,V-1)]
    same_reg = (base_reg == gold_reg) & (base_reg != unk_r)
    same_sup = np.zeros(N, dtype=bool)
    if sr:
        base_sup = reg_arr[base_reg.clip(0,R-1)]
        gold_sup = reg_arr[gold_reg.clip(0,R-1)]
        same_sup = (base_sup == gold_sup) & (base_sup != unk_s)
    fgold_top1 = (gold_vote_rank == 0) & gold_in_pool
    fgold_top3 = (gold_vote_rank <= 2) & gold_in_pool
    fconfident = (top_vote_prob >= 0.3) & (agree_max >= 2)
    return {
        "all":                           np.ones(N, dtype=bool),
        "bucketA_confuser":              base_wrong & gold_in_pool,
        "pointer_should_not_help":       ~gold_in_ctx,
        "gold_not_in_context_and_bucketA": base_wrong & gold_in_pool & ~gold_in_ctx,
        "fuzzy_gold_supported":          gold_supported,
        "fuzzy_gold_top1":               fgold_top1,
        "fuzzy_gold_top3":               fgold_top3,
        "fuzzy_confident":               fconfident,
        "same_region_confuser":          base_wrong & gold_in_pool & same_reg,
        "same_superregion_confuser":     base_wrong & gold_in_pool & same_sup,
        "high_base_margin":              base_margin > med_margin,
        "low_base_margin":               base_margin <= med_margin,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

_MK = ["n","apply_rate","noop_rate","gold_in_pool_rate","base_acc","policy_acc","acc_gain",
       "selected_gold_given_in_pool","changed_to_gold","changed_away","net_correction",
       "benefit_damage_ratio","applied_precision_ctg"]

def write_csv(rows, fields, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"[csv] {path}")

def _f(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "N/A"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)

def _m(metrics, sn, pn, key):
    return metrics.get(sn, {}).get(pn, MetricAcc()).metrics().get(key, float("nan"))

def _cand_table(cands):
    hdr = "| rank | token | base_logit | fuzzy_vote | agree | final_score | gold? | base? | fuzzy_top? |"
    sep = "|------|-------|------------|------------|-------|-------------|-------|-------|------------|"
    rows = [hdr, sep]
    for j, c in enumerate(cands):
        rows.append(
            f"| {j} | `{_decode_tok(c['tok_id']):<10}` "
            f"| {c['base_logit']:8.3f} | {c['fuzzy_vote']:.4f} "
            f"| {c['agree_cnt']} | {c['final_score']:8.3f} "
            f"| {'✓' if c['is_gold'] else ''} "
            f"| {'✓' if c['is_base_top1'] else ''} "
            f"| {'✓' if c['is_fuzzy_top'] else ''} |")
    return "\n".join(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(all_results, policy_names, slice_names, out_path, args, N_total):
    lines = [
        "# Fuzzy Phrase Recall V1 — Report\n",
        f"**val_dir:** `{args.val_dir}`  **index_dir:** `{args.index_dir}`\n",
        f"**N total:** {N_total:,}  **pool_size:** {args.candidate_pool_size}\n", ""]

    def best_pname(res, sn):
        return max(policy_names,
                   key=lambda p: res["metrics"].get(sn,{}).get(p, MetricAcc()).metrics()
                       .get("net_correction", -1e9))

    # Best mode overall
    best_mk  = max(all_results,
                   key=lambda mk: max(
                       (_m(all_results[mk]["metrics"], "all", p, "net_correction")
                        for p in policy_names), default=-1e9))
    best_net = max((_m(all_results[best_mk]["metrics"], "all", p, "net_correction")
                    for p in policy_names), default=float("nan"))
    lines.append(f"**Best mode:** `{best_mk}`  **best net on all:** {_f(best_net)}\n\n")

    # Q1
    lines.append("## Q1. Does fuzzy phrase recall beat exact n-gram recall?\n")
    lines.append("Exact n-gram: n=4 key_hit_all=0.0978, top-recalled=gold 0.2354, global gain tiny.\n")
    ans1 = "**YES — fuzzy improves**" if best_net > 0.001 else ("**Marginal**" if best_net > 0 else "**NO**")
    lines.append(f"Best fuzzy net_correction={_f(best_net)} on 'all': {ans1}\n\n")

    # Q2: mode comparison
    lines.append("## Q2. Which retrieval key works best?\n")
    lines.append("| mode | net_all | sip_all | net_bucketA | gold_supported_rate |")
    lines.append("|------|---------|---------|-------------|---------------------|")
    for mk, res in sorted(all_results.items()):
        bp     = best_pname(res, "all")
        m_all  = res["metrics"].get("all",{}).get(bp, MetricAcc()).metrics()
        m_ba   = res["metrics"].get("bucketA_confuser",{}).get(bp, MetricAcc()).metrics()
        rs     = res["rank_stats"].get("all", RankAcc(1)).stats()
        lines.append(f"| {mk} | {_f(m_all.get('net_correction'))} "
                     f"| {_f(m_all.get('selected_gold_given_in_pool'))} "
                     f"| {_f(m_ba.get('net_correction'))} "
                     f"| {_f(rs.get('fuzzy_gold_supported_rate'))} |")
    lines.append("")

    # Q3 / Q4
    for qn, sn, q_text in [
        ("Q3", "pointer_should_not_help",
         "Does fuzzy recall help when pointer cannot (gold not in context)?"),
        ("Q4", "gold_not_in_context_and_bucketA",
         "Does fuzzy help on bucketA where gold is not in context?"),
    ]:
        lines.append(f"## {qn}. {q_text}\n")
        res = all_results[best_mk]
        bp  = best_pname(res, sn)
        m   = res["metrics"].get(sn,{}).get(bp, MetricAcc()).metrics()
        lines.append(f"Best mode `{best_mk}` policy `{bp}`:")
        lines.append(f"  n={m.get('n',0):,}  net={_f(m.get('net_correction'))}  "
                     f"sip={_f(m.get('selected_gold_given_in_pool'))}  "
                     f"acc_gain={_f(m.get('acc_gain'))}\n")

    # Q5
    lines.append("## Q5. Does fuzzy raise selected_gold_given_in_pool above ~0.246 ceiling?\n")
    lines.append("Stage 2C baseline sip ≈ 0.2457.  Pointer strong slice sip ≈ 0.5348.\n")
    res  = all_results[best_mk]
    bp   = max(policy_names,
               key=lambda p: _m(res["metrics"], "all", p, "selected_gold_given_in_pool"))
    sip  = _m(res["metrics"], "all", bp, "selected_gold_given_in_pool")
    ans5 = "**YES**" if sip > 0.2468 else "**NO — does not beat baseline ceiling**"
    lines.append(f"Best sip on 'all': {_f(sip)} (policy `{bp}`) → {ans5}\n\n")

    # Q6
    lines.append("## Q6. Is there a usable confidence gate?\n")
    res     = all_results[best_mk]
    gated_p = [p for p in policy_names if p.startswith("gated_")]
    lam_p   = [p for p in policy_names if p.startswith("fuzzy_lam")]
    if gated_p:
        best_g  = max(gated_p, key=lambda p: _m(res["metrics"], "all", p, "net_correction"))
        mg_net  = _m(res["metrics"], "all", best_g, "net_correction")
        mg_ar   = _m(res["metrics"], "all", best_g, "apply_rate")
        best_l  = max(lam_p, key=lambda p: _m(res["metrics"], "all", p, "net_correction")) if lam_p else None
        ml_net  = _m(res["metrics"], "all", best_l, "net_correction") if best_l else float("nan")
        lines.append(f"Best gated: `{best_g}`  net={_f(mg_net)}  apply_rate={_f(mg_ar)}")
        if not (math.isnan(mg_net) or math.isnan(ml_net)):
            ans6 = "**YES**" if mg_net > ml_net + 0.0003 else "**NO — gating no better than lambda**"
            lines.append(f"  vs best λ net={_f(ml_net)}: {ans6}")
    else:
        lines.append("No gated policies evaluated.")
    lines.append("")

    # Q7
    lines.append("## Q7. Does h_ctx suggest phrase memory in contextual hidden states?\n")
    hctx_mk = next((mk for mk in all_results if mk.startswith("h_ctx")), None)
    if hctx_mk:
        res_h = all_results[hctx_mk]
        bp_h  = best_pname(res_h, "all")
        m_h   = res_h["metrics"].get("all",{}).get(bp_h, MetricAcc()).metrics()
        lines.append(f"Best h_ctx mode: `{hctx_mk}`  net={_f(m_h.get('net_correction'))}  "
                     f"sip={_f(m_h.get('selected_gold_given_in_pool'))}\n")
    else:
        lines.append("h_ctx mode not available in this run.\n")

    # Q8
    lines.append("## Q8. Should fuzzy phrase recall become memory expert #2?\n")
    fv_for=[]; fv_ag=[]
    if not math.isnan(best_net) and best_net > 0.003:
        fv_for.append(f"net_correction={_f(best_net)} on 'all'")
    elif not math.isnan(best_net):
        fv_ag.append(f"net_correction={_f(best_net)} — near zero on 'all'")
    res = all_results[best_mk]
    or_net = _m(res["metrics"], "all", "oracle_DIAGNOSTIC", "net_correction")
    if not math.isnan(or_net) and or_net > 0.005:
        fv_for.append(f"oracle net={_f(or_net)} — ceiling is meaningful")
    rs_all = res["rank_stats"].get("all", RankAcc(1)).stats()
    if rs_all["fuzzy_gold_supported_rate"] > 0.1:
        fv_for.append(f"fuzzy_gold_supported_rate={_f(rs_all['fuzzy_gold_supported_rate'])}")
    if fv_for:
        lines.append("**Evidence FOR:**")
        for r in fv_for: lines.append(f"- {r}")
    if fv_ag:
        lines.append("\n**Evidence AGAINST:**")
        for r in fv_ag: lines.append(f"- {r}")
    strong = len(fv_for) >= 1 and not math.isnan(best_net) and best_net > 0.003
    if strong:
        verdict = ("**Conditional YES** — fuzzy phrase recall shows signal. "
                   "Implement as a gated memory expert with confidence threshold.")
    elif fv_for:
        verdict = "**Marginal** — signal exists but weak. Use as a soft feature, not standalone expert."
    else:
        verdict = ("**NO** — fuzzy recall insufficient. "
                   "Consider: multi-token path memory; dense passage retrieval; BM25.")
    lines.append(f"\n**Verdict:** {verdict}\n")
    lines.append("---\n*Generated by eval_fuzzy_phrase_recall_v1.py — training-free diagnostic.*\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Evaluate fuzzy phrase recall")
    p.add_argument("--val_dir",              required=True)
    p.add_argument("--index_dir",            required=True)
    p.add_argument("--token_to_region",      default=None)
    p.add_argument("--super_map",            default=None)
    p.add_argument("--exact_eval_dir",       default=None)
    p.add_argument("--output_dir",           required=True)
    p.add_argument("--candidate_pool_size",  type=int, default=32)
    p.add_argument("--top_k",               type=int, default=256)
    p.add_argument("--modes",               default="h_ctx,h_raw,recency_bow,bow")
    p.add_argument("--phrase_lens",         default="16,32,64")
    p.add_argument("--retrieval_k_grid",    default="8,16,32,64")
    p.add_argument("--temp_grid",           default="0.05,0.1,0.2,0.5")
    p.add_argument("--lambda_grid",         default="0.0,0.25,0.5,1.0,2.0,4.0")
    p.add_argument("--top_vote_thresholds", default="0.1,0.2,0.3,0.5")
    p.add_argument("--margin_thresholds",   default="0.0,0.05,0.1,0.2")
    p.add_argument("--min_agreement_grid",  default="1,2,4")
    p.add_argument("--chunk_size",          type=int, default=4096)
    p.add_argument("--train_chunk_size",    type=int, default=65536)
    p.add_argument("--val_batch_size",      type=int, default=4096,
                   help="Val sub-batch size for batched matmul retrieval (peak RAM = "
                        "val_batch_size × train_chunk_size × 4 bytes)")
    p.add_argument("--memory_len",          type=int, default=128)
    p.add_argument("--max_examples",        type=int, default=50)
    p.add_argument("--seed",                type=int, default=42)
    return p, p.parse_args()


def main():
    p, args = _parse()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    modes        = [m.strip() for m in args.modes.split(",")           if m.strip()]
    phrase_lens  = [int(x)    for x in args.phrase_lens.split(",")]
    retrieval_ks = [int(x)    for x in args.retrieval_k_grid.split(",")]
    temps        = [float(x)  for x in args.temp_grid.split(",")]
    lambda_grid  = [float(x)  for x in args.lambda_grid.split(",")]
    tvt_thresh   = [float(x)  for x in args.top_vote_thresholds.split(",")]
    mg_thresh    = [float(x)  for x in args.margin_thresholds.split(",")]
    min_agree    = [int(x)    for x in args.min_agreement_grid.split(",")]

    print("=" * 60)
    print(" Fuzzy Phrase Recall Eval V1")
    print(f" val_dir:   {args.val_dir}")
    print(f" index_dir: {args.index_dir}")
    print(f" modes:     {modes}")
    print("=" * 60)

    for d in [args.val_dir, args.index_dir]:
        if not os.path.isdir(d):
            print(f"ERROR: directory not found: {d}"); sys.exit(1)

    print("\n[data] Loading val shards...")
    data     = load_val_shards(args.val_dir, args.top_k)
    N_total  = len(data["gold"])
    P        = args.candidate_pool_size
    has_hctx = "h_ctx" in data
    has_hraw = "h_raw" in data
    print(f"  val has h_ctx={has_hctx}  h_raw={has_hraw}")

    print("\n[maps] Loading region maps...")
    if args.token_to_region and os.path.isfile(args.token_to_region):
        tok_arr, reg_arr, unk_r, unk_s, sr = load_maps(args.token_to_region, args.super_map)
        print(f"  sr_enabled={sr}")
    else:
        print("  [WARN] token_to_region not found — region slices disabled")
        V = int(data["topk"].max()) + 2
        tok_arr = np.zeros(V, dtype=np.int32); reg_arr = np.zeros(2, dtype=np.int32)
        unk_r = unk_s = 0; sr = False

    print("\n[index] Loading index...")
    key_mmaps, train_gold, train_rowids, N_train, h_dim, idx_cfg = load_index(
        args.index_dir, modes, phrase_lens, has_hctx, has_hraw)

    # Shared arrays
    topk_ids     = data["topk"]
    topk_lgt     = data["lgt"]
    gold         = data["gold"]
    inp_ids      = data["ids"]
    row_ids_v    = data["row_ids"]
    cand_ids     = topk_ids[:, 1:P+1].copy()
    cand_lgts    = np.where(np.isfinite(topk_lgt[:, 1:P+1]), topk_lgt[:, 1:P+1], -1e9)
    base_top1    = topk_ids[:, 0]
    gold_in_pool = (cand_ids == gold[:, None]).any(axis=1)

    feat_dim_bow = int(idx_cfg.get("feature_dim", 8192))
    policy_names = make_policy_names(lambda_grid, tvt_thresh, mg_thresh, min_agree)
    print(f"[eval] {len(policy_names)} policies")

    all_results = {}   # mk -> {metrics, rank_stats, slices}

    for mode_name, train_keys in key_mmaps.items():
        print(f"\n[mode] {mode_name}  train_shape={train_keys.shape}")

        # Build val query matrix
        if mode_name == "h_ctx":
            q_mat = l2_norm_rows(data["h_ctx"].astype(np.float32))
        elif mode_name == "h_raw":
            q_mat = l2_norm_rows(data["h_raw"].astype(np.float32))
        else:
            plen_str = [x for x in mode_name.split("_") if x.startswith("len")]
            plen     = int(plen_str[0][3:]) if plen_str else phrase_lens[0]
            use_rec  = "recency" in mode_name
            print(f"  [bow] phrase_len={plen} recency={use_rec}")
            q_mat = np.stack(
                [build_bow_vector(inp_ids[i], plen, feat_dim_bow, use_rec)
                 for i in range(N_total)], axis=0)

        if np.isnan(q_mat).any() or np.isinf(q_mat).any():
            raise RuntimeError(f"NaN/Inf in query matrix for mode {mode_name}")

        for top_k in retrieval_ks:
            for temp in temps:
                mk = f"{mode_name}_k{top_k}_t{temp:.2f}"
                print(f"  [retrieval] {mk} ...")

                row_res = eval_mode(train_keys, train_gold, q_mat,
                                    cand_ids, gold, top_k, temp,
                                    args.train_chunk_size, args.val_batch_size)

                # Build slices
                sl = build_slices(
                    gold, base_top1, cand_ids, gold_in_pool, cand_lgts,
                    tok_arr, reg_arr, unk_r, unk_s, sr, inp_ids, args.memory_len,
                    row_res["gold_supported"], row_res["gold_vote_rank"],
                    row_res["top_vote_tok"], row_res["top_vote_prob"],
                    row_res["agree_max"])
                slice_names = list(sl.keys())

                # Batch-apply all policies once (vectorised)
                policy_results = apply_policies_batch(
                    policy_names, cand_ids, cand_lgts,
                    row_res["vote_probs"], row_res["agree_cnt"],
                    row_res["top_vote_tok"], row_res["top_vote_prob"],
                    row_res["vote_margin"], gold, topk_ids, topk_lgt)

                # Accumulate into metrics per (slice, policy)
                metrics    = {sn: {pn: MetricAcc() for pn in policy_names} for sn in slice_names}
                rank_stats = {sn: RankAcc(P)       for sn in slice_names}

                for sn in slice_names:
                    sub = sl[sn]
                    if not sub.any(): continue
                    rank_stats[sn].update(
                        row_res["gold_vote_rank"], gold_in_pool,
                        row_res["gold_vote_prob"], row_res["top_vote_prob"],
                        row_res["vote_margin"], row_res["vote_entropy"],
                        row_res["top_sim"], row_res["agree_max"],
                        row_res["top_is_gold"], row_res["gold_supported"], sub)
                    for pname in policy_names:
                        st, am, ft = policy_results[pname]
                        metrics[sn][pname].update(gold, base_top1, ft, am, st, gold_in_pool, sub)

                all_results[mk] = {"metrics": metrics, "rank_stats": rank_stats, "slices": sl}
                slice_names_final = slice_names

    if not all_results:
        print("\nERROR: all_results is empty — no retrieval configs produced results.")
        print("Check that index was built with modes matching --modes and --phrase_lens,")
        print("and that val shards have h_ctx/h_raw if those modes were requested.")
        sys.exit(1)

    # ── Examples ─────────────────────────────────────────────────────────────
    print("\n[examples] Collecting examples...")
    best_mk = max(all_results,
                  key=lambda mk: max(
                      (_m(all_results[mk]["metrics"], "all", p, "net_correction")
                       for p in policy_names), default=-1e9))
    best_res  = all_results[best_mk]
    best_rows = None
    # Re-run retrieval for best_mk to get vote_probs (we didn't store them all)
    # Parse mk: "{mode_name}_k{top_k}_t{temp}"
    # Find the corresponding mode_name
    mk_parts  = best_mk.rsplit("_k", 1)
    mode_name_ex = mk_parts[0]
    kt_str    = mk_parts[1]  # e.g. "32_t0.10"
    top_k_ex  = int(kt_str.split("_t")[0])
    temp_ex   = float(kt_str.split("_t")[1])
    if mode_name_ex in key_mmaps:
        if mode_name_ex == "h_ctx":
            q_ex = l2_norm_rows(data["h_ctx"].astype(np.float32))
        elif mode_name_ex == "h_raw":
            q_ex = l2_norm_rows(data["h_raw"].astype(np.float32))
        else:
            plen_str = [x for x in mode_name_ex.split("_") if x.startswith("len")]
            plen_ex  = int(plen_str[0][3:]) if plen_str else phrase_lens[0]
            use_rec_ex = "recency" in mode_name_ex
            q_ex = np.stack(
                [build_bow_vector(inp_ids[i], plen_ex, feat_dim_bow, use_rec_ex)
                 for i in range(N_total)], axis=0)
        print(f"  Re-running retrieval for examples (mode={mode_name_ex})...")
        best_rows = eval_mode(key_mmaps[mode_name_ex], train_gold, q_ex,
                              cand_ids, gold, top_k_ex, temp_ex,
                              args.train_chunk_size, args.val_batch_size)

    KINDS = ["fuzzy_helps","fuzzy_hurts","fuzzy_should_help_but_fails","fuzzy_retrieves_gold"]
    ex_buckets = {k: [] for k in KINDS}
    max_per    = max(args.max_examples // len(KINDS), 1)
    rng        = random.Random(args.seed)
    idxs       = list(range(N_total)); rng.shuffle(idxs)

    if best_rows is not None:
        bp_ex = max(policy_names,
                    key=lambda p: _m(best_res["metrics"], "all", p, "net_correction"))
        bp_results = apply_policies_batch(
            [bp_ex], cand_ids, cand_lgts,
            best_rows["vote_probs"], best_rows["agree_cnt"],
            best_rows["top_vote_tok"], best_rows["top_vote_prob"],
            best_rows["vote_margin"], gold, topk_ids, topk_lgt)
        bp_ft = bp_results[bp_ex][2]

        for i in idxs:
            bw  = base_top1[i] != gold[i]
            gip = bool(gold_in_pool[i])
            tvt = int(best_rows["top_vote_tok"][i])
            tvp = float(best_rows["top_vote_prob"][i])
            tis = bool(best_rows["top_is_gold"][i])
            cands_ex = [{"tok_id": int(cand_ids[i][j]),
                         "base_logit": float(cand_lgts[i][j]),
                         "fuzzy_vote": float(best_rows["vote_probs"][i][j]),
                         "agree_cnt":  int(best_rows["agree_cnt"][i][j]),
                         "final_score":float(cand_lgts[i][j] + math.log(float(best_rows["vote_probs"][i][j]) + _EPS)),
                         "is_gold": int(cand_ids[i][j]) == int(gold[i]),
                         "is_base_top1": False,
                         "is_fuzzy_top": j == int(best_rows["vote_probs"][i].argmax())}
                        for j in range(min(P, 16))]
            ex = {"row_id": int(row_ids_v[i]), "ctx_ids": inp_ids[i,-128:].tolist(),
                  "gold_id": int(gold[i]), "base_top1_id": int(base_top1[i]),
                  "base_top1_lgt": float(topk_lgt[i,0]),
                  "fuzzy_top_tok": tvt, "fuzzy_top_prob": tvp, "mode": best_mk,
                  "top_sim": float(best_rows["top_sim"][i]),
                  "agree_top": int(best_rows["agree_max"][i]),
                  "gold_vote_rank": int(best_rows["gold_vote_rank"][i]),
                  "gold_vote": float(best_rows["gold_vote_prob"][i]),
                  "candidates": cands_ex}
            if len(ex_buckets["fuzzy_helps"]) < max_per and bw and gip and tvt == gold[i]:
                ex_buckets["fuzzy_helps"].append(ex)
            if len(ex_buckets["fuzzy_hurts"]) < max_per and not bw and tvt >= 0 and tvt != gold[i]:
                ex_buckets["fuzzy_hurts"].append(ex)
            if len(ex_buckets["fuzzy_should_help_but_fails"]) < max_per and bw and gip and tvt != gold[i]:
                ex_buckets["fuzzy_should_help_but_fails"].append(ex)
            if len(ex_buckets["fuzzy_retrieves_gold"]) < max_per and tis:
                ex_buckets["fuzzy_retrieves_gold"].append(ex)
            if all(len(v) >= max_per for v in ex_buckets.values()):
                break

    kind_labels = {
        "fuzzy_helps":                "Fuzzy HELPS (base wrong, gold in pool, fuzzy top = gold)",
        "fuzzy_hurts":                "Fuzzy HURTS (base correct, fuzzy top ≠ gold)",
        "fuzzy_should_help_but_fails":"Should Help But Fails (bucketA + fuzzy top ≠ gold)",
        "fuzzy_retrieves_gold":       "Fuzzy Retrieves Gold (top voted = gold)",
    }
    for kind in KINDS:
        exs   = ex_buckets[kind]
        lines = [f"# {kind_labels[kind]}\n\n"]
        for idx, ex in enumerate(exs):
            ctx_str = _decode_ids(ex["ctx_ids"])
            lines.append(f"### Example {idx+1}  (row {ex['row_id']})\n")
            lines.append(
                f"**Gold:** `{_decode_tok(ex['gold_id'])}`  "
                f"**Base:** `{_decode_tok(ex['base_top1_id'])}` ({ex['base_top1_lgt']:.3f})  "
                f"**FuzzyTop:** `{_decode_tok(ex['fuzzy_top_tok'])}` (prob={ex['fuzzy_top_prob']:.4f})  "
                f"**Mode:** {ex['mode']}\n")
            lines.append(
                f"top_sim={ex['top_sim']:.4f}  agree_top={ex['agree_top']}  "
                f"gold_vote_rank={ex['gold_vote_rank']}  gold_vote={ex['gold_vote']:.4f}\n")
            lines.append("\n**Context tail:**\n```\n" + ctx_str[-400:] + "\n```\n")
            lines.append("\n**Candidates:**\n" + _cand_table(ex["candidates"]) + "\n\n---\n")
        out_ex = os.path.join(args.output_dir, f"examples_{kind}.md")
        with open(out_ex, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        print(f"[md] {out_ex}")

    # ── CSVs ─────────────────────────────────────────────────────────────────
    print("\n[write] Writing CSVs...")
    summary_rows = []
    for mk, res in all_results.items():
        for sn in slice_names_final:
            for pn in policy_names:
                m = res["metrics"].get(sn,{}).get(pn, MetricAcc()).metrics()
                row = {"mode": mk, "policy": pn, "slice": sn,
                       "fraction_of_val": f"{m.get('n',0)/max(N_total,1):.6f}"}
                for k in _MK:
                    v = m.get(k,""); row[k] = f"{v:.6f}" if isinstance(v,float) else v
                summary_rows.append(row)
    write_csv(summary_rows, ["mode","policy","slice","fraction_of_val"]+_MK,
              os.path.join(args.output_dir, "fuzzy_slice_metrics.csv"))

    _RK = ["n","fuzzy_gold_supported_rate","neighbor_top_token_is_gold_rate",
           "gold_vote_rank_le_1","gold_vote_rank_le_3","gold_vote_rank_le_5","gold_vote_rank_le_10",
           "mean_gold_vote","mean_top_vote_prob","mean_vote_margin","mean_vote_entropy",
           "mean_top_neighbor_similarity","mean_agreement_count"]
    rank_rows = []
    for mk, res in all_results.items():
        for sn in slice_names_final:
            s  = res["rank_stats"].get(sn, RankAcc(P)).stats()
            nn = max(s["n"],1)
            row = {"mode": mk, "slice": sn}
            for k in _RK:
                v = s.get(k,0)
                row[k] = (f"{v/nn:.4f}" if k.startswith("gold_vote_rank_le")
                          else f"{v:.4f}" if isinstance(v,float) else str(v))
            rank_rows.append(row)
    write_csv(rank_rows, ["mode","slice"]+_RK,
              os.path.join(args.output_dir, "fuzzy_retrieval_rank_stats.csv"))

    best_rows_csv = []
    for sn in slice_names_final:
        best_mk2 = best_pn2 = None; best_net2 = -1e9
        for mk, res in all_results.items():
            for pn in policy_names:
                net = _m(res["metrics"], sn, pn, "net_correction")
                if not math.isnan(net) and net > best_net2:
                    best_net2 = net; best_mk2 = mk; best_pn2 = pn
        if best_mk2:
            m = all_results[best_mk2]["metrics"].get(sn,{}).get(best_pn2, MetricAcc()).metrics()
            row = {"slice": sn, "best_mode": best_mk2, "best_policy": best_pn2}
            for k in _MK:
                v = m.get(k,""); row[k] = f"{v:.6f}" if isinstance(v,float) else v
            best_rows_csv.append(row)
    write_csv(best_rows_csv, ["slice","best_mode","best_policy"]+_MK,
              os.path.join(args.output_dir, "fuzzy_best_by_slice.csv"))

    write_report(all_results, policy_names, slice_names_final,
                 os.path.join(args.output_dir, "fuzzy_phrase_report.md"), args, N_total)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n[done] Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
