#!/usr/bin/env python3
"""
eval_geometry_evidence_interaction_v2.py — Geometry + Evidence Interaction Diagnostic V2.

Analyzes why previous memory/expert methods worked or failed from a vector/geometry
standpoint. Diagnoses the design of Memory Evidence Mixer V1.

GOLD USED ONLY FOR DIAGNOSTICS/METRICS.  NO TRAINING.  NO NEW EXPERT.

Parts:
  1.  Load unembedding geometry from checkpoint
  2.  Hidden state audit (h_raw / h_ctx vs saved logits)
  3.  Base near-miss geometry (d_gold, gap, cosine, delta_scale)
  4.  Correction crowding geometry (relative projections along d_gold)
  5.  Candidate relative geometry (per-candidate features vs base)
  6.  Load / reconstruct expert evidence
  7.  Expert logit-space geometry (support, entropy, sharpness)
  8.  Expert vs base geometry (successful / harmful / w2w edits)
  9.  Expert agreement geometry (multi-expert rows)
  10. Method failure explanation table
  11. Near-miss bucket classification
  12. Slice summaries + report
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import torch

def _nanmean(arr):
    """np.nanmean that returns nan silently when arr is empty or all-nan."""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(a))

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

_EPS    = 1e-9
VS      = 50257          # GPT-2 vocab size
GEO_B   = 256            # rows per U-lookup batch
CSAMP   = 5000           # max rows in candidate_geometry_metrics.csv

# ── shard field aliases ────────────────────────────────────────────────────────
_TOPK  = ["base_topk_ids", "base_topk", "topk_ids"]
_LGT   = ["base_topk_logits", "base_topk_lgt", "topk_lgt", "topk_logits"]
_GOLD  = ["gold_token", "gold", "labels"]
_IDS   = ["input_ids"]
_RID   = ["row_id", "row_ids"]
_OFF   = ["token_offset", "offsets", "offset"]
_HRAW  = ["h_raw", "hidden_raw"]
_HCTX  = ["h_ctx", "hidden_ctx"]
_GREG  = ["gold_region"]
_RREG  = ["router_topk_reg"]
_RPRB  = ["router_topk_prb"]


def _get(sh, aliases, required=True):
    for a in aliases:
        if a in sh:
            return sh[a]
    if required:
        raise KeyError(f"Need one of {aliases}; have {list(sh.keys())}")
    return None


# ── tiktoken (optional) ───────────────────────────────────────────────────────
_enc = None
try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
except Exception:
    pass


def _dtok(tid):
    if _enc:
        try:
            return repr(_enc.decode([int(tid)]))[1:-1]
        except Exception:
            pass
    return f"<{tid}>"


def _dids(ids):
    if _enc:
        try:
            return _enc.decode([int(i) for i in ids if 0 <= int(i) < VS])
        except Exception:
            pass
    return " ".join(_dtok(i) for i in ids)


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Shard loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_val_shards(val_dir, pool_size, max_rows=None):
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {val_dir}")
    print(f"[shards] {len(paths)} val shards")
    bufs = defaultdict(list)
    total = 0
    first = True
    for sp in paths:
        if max_rows and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print("[preflight] first shard keys:")
            for k, v in sh.items():
                print(f"  {k:28s}: shape={getattr(v,'shape',None)}"
                      f"  dtype={getattr(v,'dtype',type(v).__name__)}")
            first = False
        topk = _get(sh, _TOPK).long()
        lgt  = _get(sh, _LGT).float()
        gold = _get(sh, _GOLD).long()
        ids  = _get(sh, _IDS).long()
        rids = _get(sh, _RID,  required=False)
        off  = _get(sh, _OFF,  required=False)
        hraw = _get(sh, _HRAW, required=False)
        hctx = _get(sh, _HCTX, required=False)
        greg = _get(sh, _GREG, required=False)
        B, K = topk.shape
        if rids is None:
            rids = torch.arange(total, total + B)
        if off is None:
            off = torch.full((B,), -1, dtype=torch.long)
        # Adjust pool size
        if K < pool_size:
            topk = torch.cat([topk, torch.zeros(B, pool_size - K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, pool_size - K), -1e9)], 1)
        elif K > pool_size:
            topk, lgt = topk[:, :pool_size], lgt[:, :pool_size]
        if max_rows and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, ids, rids, off = (x[:keep] for x in (topk, lgt, gold, ids, rids, off))
            if hraw is not None: hraw = hraw[:keep]
            if hctx is not None: hctx = hctx[:keep]
            if greg is not None: greg = greg[:keep]
            B = keep
        bufs["topk"].append(topk.numpy())
        bufs["lgt"].append(lgt.numpy())
        bufs["gold"].append(gold.numpy())
        bufs["ids"].append(ids.numpy())
        bufs["row_ids"].append(rids.numpy())
        bufs["token_offset"].append(off.numpy())
        if hraw is not None:
            bufs["h_raw"].append(hraw.float().numpy())
        if hctx is not None:
            bufs["h_ctx"].append(hctx.float().numpy())
        if greg is not None:
            bufs["gold_region"].append(greg.numpy())
        total += B
    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    data["lgt"] = np.where(np.isfinite(data["lgt"]), data["lgt"], -1e9)
    print(f"[shards] {total:,} rows  pool_size={pool_size}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Unembedding matrix loading
# ═══════════════════════════════════════════════════════════════════════════════

# Ordered list of exact key names to try (after prefix stripping)
_U_EXACT_KEYS = [
    "lm_head.weight",
    "tok_emb.weight",
    "token_emb.weight",
    "wte.weight",
    "transformer.wte.weight",
    "model.lm_head.weight",
    "model.token_emb.weight",
    "model.tok_emb.weight",
    "model.transformer.wte.weight",
    "embedding.weight",
    "model.embed_tokens.weight",
]
# Suffix patterns — match any key that ends with one of these
_U_SUFFIX_KEYS = [
    "lm_head.weight",
    "token_emb.weight",
    "tok_emb.weight",
    "wte.weight",
]
# Prefix strips applied iteratively until stable
_U_PREFIXES = ["module.", "_orig_mod.", "model.model.", "model.", "_model."]


def _strip_prefixes(k):
    """Strip known DDP/compile prefixes until the key stabilises."""
    changed = True
    while changed:
        changed = False
        for pfx in _U_PREFIXES:
            if k.startswith(pfx):
                k = k[len(pfx):]
                changed = True
    return k


def _pick_embedding(w):
    """Return (w, V, d) where w is [V, d] float32."""
    w = w.float().numpy()
    if w.ndim != 2:
        return None, 0, 0
    V, d = w.shape
    if d > V:           # stored as [d, V] — transpose
        w = w.T
        V, d = w.shape
    return w, V, d


def load_unembedding(ckpt_path):
    """
    Returns (U [V, d] float32, info dict).
    Raises RuntimeError if checkpoint exists but no embedding matrix can be found.
    Returns (None, {}) only when checkpoint path is absent (caller decides what to do).
    """
    if not ckpt_path:
        print("[unembedding] no checkpoint path supplied")
        return None, {"unembedding_source": "none", "d_model": "?"}
    if not os.path.isfile(ckpt_path):
        print(f"[unembedding] checkpoint not found: {ckpt_path}")
        return None, {"unembedding_source": "missing", "d_model": "?"}

    print(f"[unembedding] loading {ckpt_path}  ({os.path.getsize(ckpt_path)/1e6:.1f} MB)")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Unwrap common wrappers
    state = raw
    for wrap in ("state_dict", "model_state_dict", "model"):
        if isinstance(raw, dict) and wrap in raw:
            candidate = raw[wrap]
            if isinstance(candidate, dict) and len(candidate) > 0:
                state = candidate
                print(f"[unembedding] unwrapped via '{wrap}'")
                break

    if not isinstance(state, dict):
        raise RuntimeError(
            f"[unembedding] checkpoint is {type(state).__name__}, not a state_dict. "
            f"Cannot extract embedding matrix.")

    print(f"[unembedding] state_dict has {len(state)} keys")
    # Print all keys for diagnosis
    tensor_keys = [k for k, v in state.items() if isinstance(v, torch.Tensor)]
    print(f"[unembedding] tensor keys ({len(tensor_keys)}):")
    for k in tensor_keys:
        print(f"  {k}: {state[k].shape}")

    # Build stripped → original key map
    stripped = {}
    for k in tensor_keys:
        sk = _strip_prefixes(k)
        if sk not in stripped:
            stripped[sk] = k          # first match wins

    print(f"[unembedding] stripped keys: {list(stripped.keys())[:30]}")

    def _try(key_list, label):
        for key in key_list:
            orig = stripped.get(key)
            if orig is not None:
                w, V, d = _pick_embedding(state[orig])
                if w is not None:
                    return w, V, d, key
        return None, 0, 0, None

    # 1. Exact match on stripped keys
    w, V, d, found_key = _try(_U_EXACT_KEYS, "exact")

    # 2. Suffix match on stripped keys
    if w is None:
        for sk, orig in stripped.items():
            for sfx in _U_SUFFIX_KEYS:
                if sk.endswith(sfx):
                    w2, V2, d2 = _pick_embedding(state[orig])
                    if w2 is not None:
                        w, V, d, found_key = w2, V2, d2, sk
                        break
            if w is not None:
                break

    # 3. Suffix match on original (un-stripped) keys
    if w is None:
        for k in tensor_keys:
            for sfx in _U_SUFFIX_KEYS:
                if k.endswith(sfx):
                    w2, V2, d2 = _pick_embedding(state[k])
                    if w2 is not None:
                        w, V, d, found_key = w2, V2, d2, k
                        break
            if w is not None:
                break

    if w is None:
        raise RuntimeError(
            f"[unembedding] FATAL: no embedding matrix found in checkpoint.\n"
            f"  Tried exact keys: {_U_EXACT_KEYS}\n"
            f"  Tried suffix patterns: {_U_SUFFIX_KEYS}\n"
            f"  Available stripped keys: {list(stripped.keys())}\n"
            f"  Checkpoint path: {ckpt_path}\n"
            f"  Fix: add the actual key name to _U_EXACT_KEYS in the script.")

    info = {
        "unembedding_source": found_key,
        "vocab_size":         V,
        "d_model":            d,
        "dtype":              str(w.dtype),
        "U_shape":            list(w.shape),
        "ckpt_path":          ckpt_path,
    }
    norm_mean = float(np.linalg.norm(w, axis=1).mean())
    print(f"[unembedding] FOUND '{found_key}'")
    print(f"  shape={w.shape}  vocab={V}  d_model={d}  dtype={w.dtype}")
    print(f"  row-norm mean={norm_mean:.4f}  min={w.min():.4f}  max={w.max():.4f}")
    return w, info


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Region maps
# ═══════════════════════════════════════════════════════════════════════════════

def load_region_maps(t2r_path, super_path=None):
    if not t2r_path or not os.path.isfile(t2r_path):
        V = VS
        return np.zeros(V, np.int32), np.zeros(2, np.int32), 1, 1, False
    with open(t2r_path) as f:
        raw = json.load(f)
    t2r = ({i: v for i, v in enumerate(raw) if v is not None}
           if isinstance(raw, list) else {int(k): v for k, v in raw.items()})
    unk_r = int(max(t2r.values())) + 1 if t2r else 1
    V = max(t2r.keys(), default=0) + 2
    tok_arr = np.full(V, unk_r, dtype=np.int32)
    for t, r in t2r.items():
        tok_arr[int(t)] = int(r)
    r2s = {}; unk_s = 1; sr = False
    if super_path and os.path.isfile(super_path):
        with open(super_path) as f:
            raw2 = json.load(f)
        r2s = ({int(k): v for k, v in raw2.items()} if not isinstance(raw2, list)
               else {i: v for i, v in enumerate(raw2) if v is not None})
        unk_s = int(max(r2s.values())) + 1 if r2s else 1
        sr = True
    R = max(r2s.keys(), default=0) + 2
    reg_arr = np.full(R, unk_s, dtype=np.int32)
    for r, s in r2s.items():
        reg_arr[int(r)] = int(s)
    return tok_arr, reg_arr, unk_r, unk_s, sr


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Hidden state audit
# ═══════════════════════════════════════════════════════════════════════════════

def audit_hidden_states(data, U, audit_rows=2000):
    """Check whether h_raw/h_ctx reproduce saved base_topk logits."""
    result = {
        "h_raw_available": "h_raw" in data,
        "h_ctx_available": "h_ctx" in data,
        "h_raw_top1_match_rate":     float("nan"),
        "h_ctx_top1_match_rate":     float("nan"),
        "h_raw_topk_overlap":        float("nan"),
        "h_ctx_topk_overlap":        float("nan"),
        "h_raw_logit_corr":          float("nan"),
        "h_ctx_logit_corr":          float("nan"),
    }
    if U is None:
        result["note"] = "U not available — skipping audit"
        return result

    N  = min(len(data["gold"]), audit_rows)
    tk = data["topk"][:N].astype(np.int64)
    lg = data["lgt"][:N].astype(np.float32)
    saved_top1 = tk[:, 0]

    for h_key, prefix in [("h_raw", "h_raw"), ("h_ctx", "h_ctx")]:
        if h_key not in data:
            continue
        H = data[h_key][:N].astype(np.float32)        # [N, d]
        # Compute logits in batches
        pred_top1  = np.empty(N, dtype=np.int64)
        corrs      = []
        overlaps   = []
        for s in range(0, N, GEO_B):
            e  = min(s + GEO_B, N)
            Hb = H[s:e]                               # [B, d]
            # full_lgt: [B, V] — too large; compute only top-K logits
            # Compare with saved topk via targeted dot product
            # lgt_topk: [B, K]
            K   = tk.shape[1]
            U_k = U[tk[s:e].ravel()].reshape(e - s, K, U.shape[1])  # [B, K, d]
            lgt_topk = (Hb[:, None, :] * U_k).sum(axis=2)           # [B, K]
            # Also compute top1 over full vocab for top1 match
            lgt_full  = Hb @ U.T                                      # [B, V]
            pred_top1[s:e] = lgt_full.argmax(axis=1)
            # Correlation on saved topk logits
            for i in range(e - s):
                mask = np.isfinite(lg[s + i]) & (lg[s + i] > -1e8)
                if mask.sum() < 2:
                    continue
                c = np.corrcoef(lgt_topk[i][mask], lg[s + i][mask])
                corrs.append(float(c[0, 1]) if np.isfinite(c[0, 1]) else float("nan"))
                # Top-k overlap: fraction of saved topk found in pred top-K from h
                topk_h = lgt_full[i].argsort()[::-1][:K]
                overlap = len(set(topk_h) & set(tk[s + i][mask])) / max(mask.sum(), 1)
                overlaps.append(overlap)
        match_rate = float((pred_top1 == saved_top1).mean())
        result[f"{prefix}_top1_match_rate"] = match_rate
        result[f"{prefix}_topk_overlap"]    = float(np.nanmean(overlaps)) if overlaps else float("nan")
        result[f"{prefix}_logit_corr"]      = float(np.nanmean(corrs))    if corrs    else float("nan")
        print(f"[audit] {prefix}: top1_match={match_rate:.4f}  "
              f"topk_overlap={result[prefix+'_topk_overlap']:.4f}  "
              f"logit_corr={result[prefix+'_logit_corr']:.4f}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3-5. Core geometry computation  (batched over rows)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_geometry(data, U, tok_arr, reg_arr, unk_r, unk_s, sr, topm):
    """
    Returns three arrays:
      row_geom:   structured array [N] — per-row geometry metrics
      cand_geom:  dict of arrays [N, topm] — per-candidate geometry
      oracle_sim: dict of arrays [N] — oracle simulation results (gold-in-pool rows only)
    """
    N   = len(data["gold"])
    P   = data["topk"].shape[1]
    M   = min(P, topm)
    gold_np = data["gold"].astype(np.int64)
    topk_np = data["topk"].astype(np.int64)
    lgt_np  = data["lgt"].astype(np.float32)
    ar      = np.arange(N)

    # Row-level gold_in_pool and gold_rank
    gip  = (topk_np == gold_np[:, None]).any(axis=1)           # [N]
    rank_mat = (topk_np == gold_np[:, None]).astype(np.int32)  # [N, P]
    gold_rank = np.where(gip, rank_mat.argmax(axis=1), P).astype(np.int32)  # [N]

    base_top1  = topk_np[:, 0]                                 # [N]
    base_logit = lgt_np[:, 0]                                  # [N]
    gold_logit = np.where(gip, lgt_np[ar, np.clip(gold_rank, 0, P - 1)], float("nan"))

    # Region labels
    Vt = len(tok_arr)
    gc = np.clip(gold_np,  0, Vt - 1)
    bc = np.clip(base_top1, 0, Vt - 1)
    gold_reg = tok_arr[gc]
    base_reg = tok_arr[bc]
    same_reg = (gold_reg == base_reg) & (gold_reg != unk_r)
    same_sr  = same_reg.copy()
    if sr:
        Rr = len(reg_arr)
        gs = reg_arr[np.clip(gold_reg, 0, Rr - 1)]
        bs = reg_arr[np.clip(base_reg, 0, Rr - 1)]
        same_sr = (gs == bs) & (gs != unk_s)

    # Candidate region entropy
    cand_regs   = tok_arr[np.clip(topk_np[:, :M], 0, Vt - 1)]  # [N, M]
    # Per-row region distribution
    def region_entropy(regs_2d):
        B, MM = regs_2d.shape
        ent = np.zeros(B, dtype=np.float32)
        for i in range(B):
            vals, cnts = np.unique(regs_2d[i], return_counts=True)
            probs = cnts / cnts.sum()
            ent[i] = float(-np.sum(probs * np.log(probs + _EPS)))
        return ent
    cand_reg_ent = region_entropy(cand_regs)

    # ── Batched U-lookup geometry ────────────────────────────────────────────
    # Per-row: base_gap, cosine_gold_base, target_norm, delta_scale
    # Per-cand: relative_projection, cosine_cand_base
    # Oracle: adjusted logits after target delta

    row_base_gap    = np.full(N, float("nan"), dtype=np.float32)
    row_cos_gb      = np.full(N, float("nan"), dtype=np.float32)
    row_dg_norm     = np.full(N, float("nan"), dtype=np.float32)
    row_ds          = np.full(N, float("nan"), dtype=np.float32)
    row_req_dn      = np.full(N, float("nan"), dtype=np.float32)

    # Crowding metrics
    row_max_rel_proj   = np.full(N, float("nan"), dtype=np.float32)
    row_mean_rel_proj  = np.full(N, float("nan"), dtype=np.float32)
    row_n_above_0p5    = np.zeros(N, dtype=np.int32)
    row_n_above_0p8    = np.zeros(N, dtype=np.int32)
    row_n_above_1p0    = np.zeros(N, dtype=np.int32)

    # Oracle simulation
    oracle_new_top1    = topk_np[:, 0].copy()
    oracle_gold_rank   = np.full(N, P, dtype=np.int32)
    oracle_gold_wins   = np.zeros(N, dtype=bool)
    oracle_wrong_steals= np.zeros(N, dtype=bool)
    oracle_margin_after= np.full(N, float("nan"), dtype=np.float32)

    # Candidate geometry: [N, M] arrays
    cand_gap    = np.full((N, M), float("nan"), dtype=np.float32)
    cand_cos_cb = np.full((N, M), float("nan"), dtype=np.float32)
    cand_rel_pr = np.full((N, M), float("nan"), dtype=np.float32)
    cand_sreg   = np.zeros((N, M), dtype=bool)
    cand_ssr    = np.zeros((N, M), dtype=bool)

    # Only compute U-based geometry for gip rows to save memory
    gip_idx = np.where(gip)[0]

    # ── base_gap from saved logits (no U needed) ─────────────────────────────
    # Do this first so base_gap is always populated for gip rows, even if U
    # fails to load or the U block is skipped.
    print(f"[geometry] computing base_gap from saved logits for {len(gip_idx):,} gip rows...")
    for bi in range(0, len(gip_idx), GEO_B):
        idxs = gip_idx[bi: bi + GEO_B]
        gr_b = gold_rank[idxs]
        gold_lgt_b = lgt_np[idxs, np.clip(gr_b, 0, P - 1)]
        row_base_gap[idxs] = lgt_np[idxs, 0] - gold_lgt_b

    # Sanity-check base_gap
    n_bg_nan = int(np.isnan(row_base_gap[gip_idx]).sum())
    if n_bg_nan > 0:
        print(f"[geometry] WARNING: {n_bg_nan} gip rows have nan base_gap "
              f"(check that lgt array has no -inf/-1e9 sentinels at gold position)")
    else:
        print(f"[geometry] base_gap OK: mean={float(np.nanmean(row_base_gap[gip_idx])):.4f}  "
              f"min={float(np.nanmin(row_base_gap[gip_idx])):.4f}  "
              f"max={float(np.nanmax(row_base_gap[gip_idx])):.4f}")

    if U is not None:
        print(f"[geometry] U-based geometry: processing {len(gip_idx):,} gip rows "
              f"in batches of {GEO_B}...")
        d = U.shape[1]
        for bi in range(0, len(gip_idx), GEO_B):
            idxs = gip_idx[bi: bi + GEO_B]
            B    = len(idxs)

            # Gather topk ids and logits for this batch
            tk_b   = topk_np[idxs, :M]          # [B, M]
            lg_b   = lgt_np[idxs, :M]           # [B, M]
            gr_b   = gold_rank[idxs]             # [B]
            gid_b  = gold_np[idxs]               # [B]
            bid_b  = tk_b[:, 0]                  # [B] base ids

            # Lookup embeddings
            u_base = U[np.clip(bid_b, 0, len(U) - 1)]                          # [B, d]
            u_gold = U[np.clip(gid_b, 0, len(U) - 1)]                          # [B, d]
            u_cand = U[np.clip(tk_b.ravel(), 0, len(U) - 1)].reshape(B, M, d)  # [B, M, d]

            d_gold     = u_gold - u_base                         # [B, d]
            dg_norm2   = (d_gold ** 2).sum(axis=1)               # [B]
            dg_norm    = np.sqrt(dg_norm2 + _EPS)                 # [B]

            # base_gap already in row_base_gap[idxs]; reuse it here
            bg = row_base_gap[idxs]                               # [B]
            ds = bg / (dg_norm2 + _EPS)                           # [B] delta scale

            # Cosine gold-base
            ub_n = np.sqrt((u_base ** 2).sum(axis=1) + _EPS)     # [B]
            ug_n = np.sqrt((u_gold ** 2).sum(axis=1) + _EPS)     # [B]
            cos_gb = (u_base * u_gold).sum(axis=1) / (ub_n * ug_n)

            # row_base_gap[idxs] already set before U block
            row_cos_gb[idxs]    = cos_gb
            row_dg_norm[idxs]   = dg_norm
            row_ds[idxs]        = ds
            row_req_dn[idxs]    = np.abs(ds) * dg_norm

            # Candidate gap and cosine to base
            cand_gap[idxs]    = lg_b[:, 0:1] - lg_b              # [B, M] base_logit - cand_logit
            uc_n = np.sqrt((u_cand ** 2).sum(axis=2) + _EPS)      # [B, M]
            cos_cb = (u_cand * u_base[:, None, :]).sum(axis=2) / (ub_n[:, None] * uc_n)
            cand_cos_cb[idxs] = cos_cb

            # Candidate region membership
            cand_sreg[idxs] = (tok_arr[np.clip(tk_b, 0, Vt-1)] == gold_reg[idxs, None])
            if sr:
                Rr = len(reg_arr)
                cand_regids = tok_arr[np.clip(tk_b, 0, Vt-1)]
                cand_srids  = reg_arr[np.clip(cand_regids, 0, Rr-1)]
                gold_sr_b   = reg_arr[np.clip(gold_reg[idxs], 0, Rr-1)]
                cand_ssr[idxs] = (cand_srids == gold_sr_b[:, None])

            # Relative projections (crowding)
            u_diff = u_cand - u_base[:, None, :]                  # [B, M, d]
            proj   = (d_gold[:, None, :] * u_diff).sum(axis=2)    # [B, M]
            rel_pr = proj / (dg_norm2[:, None] + _EPS)            # [B, M]
            cand_rel_pr[idxs] = rel_pr

            # Exclude gold and base from crowding stats
            is_gold_cand = (tk_b == gid_b[:, None])               # [B, M]
            is_base_cand = np.zeros_like(is_gold_cand)
            is_base_cand[:, 0] = True
            wrong_mask = ~is_gold_cand & ~is_base_cand             # [B, M]
            masked_rp  = np.where(wrong_mask, rel_pr, -np.inf)
            row_max_rel_proj[idxs]  = masked_rp.max(axis=1)
            mean_rp = np.where(wrong_mask, rel_pr, float("nan"))
            row_mean_rel_proj[idxs] = np.nanmean(mean_rp, axis=1)
            row_n_above_0p5[idxs]   = (np.where(wrong_mask, rel_pr, 0) > 0.5).sum(axis=1)
            row_n_above_0p8[idxs]   = (np.where(wrong_mask, rel_pr, 0) > 0.8).sum(axis=1)
            row_n_above_1p0[idxs]   = (np.where(wrong_mask, rel_pr, 0) > 1.0).sum(axis=1)

            # Oracle simulation: add delta = ds * d_gold to hidden state
            # gold_in_M: gold present within the top-M slice (may differ from full gip)
            gold_in_M    = (tk_b == gid_b[:, None]).any(axis=1)   # [B]
            gold_M_idx   = (tk_b == gid_b[:, None]).argmax(axis=1) # [B] pos in tk_b

            delta         = ds[:, None] * d_gold                   # [B, d]
            delta_lgt     = (u_cand * delta[:, None, :]).sum(axis=2)  # [B, M]
            adj_lgt       = lg_b + delta_lgt                       # [B, M]
            oracle_top1_i = adj_lgt.argmax(axis=1)                 # [B]
            oracle_new_top1[idxs]  = tk_b[np.arange(B), oracle_top1_i]
            oracle_gold_wins[idxs] = gold_in_M & (oracle_new_top1[idxs] == gid_b)
            # Find gold rank in adjusted logits (only when gold is in top-M)
            adj_sort_idx = np.argsort(-adj_lgt, axis=1)
            for ii in range(B):
                if gold_in_M[ii]:
                    rank_hits = np.where(adj_sort_idx[ii] == gold_M_idx[ii])[0]
                    if len(rank_hits) > 0:
                        oracle_gold_rank[idxs[ii]] = int(rank_hits[0])
            # Wrong steals: clamp gold index to M when gold not in top-M (safe sentinel)
            safe_gi   = np.where(gold_in_M, gold_M_idx, 0)
            gold_adj  = adj_lgt[np.arange(B), safe_gi]
            max_wrong_adj = np.where(~is_gold_cand, adj_lgt, -np.inf).max(axis=1)
            oracle_wrong_steals[idxs] = gold_in_M & (max_wrong_adj > gold_adj)
            oracle_margin_after[idxs] = np.where(
                gold_in_M, gold_adj - max_wrong_adj, float("nan"))

        # Also compute for non-gip rows (cand geometry, no oracle)
        non_gip = np.where(~gip)[0]
        if U is not None and len(non_gip) > 0:
            for bi in range(0, len(non_gip), GEO_B):
                idxs = non_gip[bi: bi + GEO_B]
                B    = len(idxs)
                tk_b = topk_np[idxs, :M]
                lg_b = lgt_np[idxs, :M]
                bid_b = tk_b[:, 0]
                u_base = U[np.clip(bid_b, 0, len(U) - 1)]
                u_cand = U[np.clip(tk_b.ravel(), 0, len(U) - 1)].reshape(B, M, d)
                ub_n = np.sqrt((u_base ** 2).sum(axis=1) + _EPS)
                uc_n = np.sqrt((u_cand ** 2).sum(axis=2) + _EPS)
                cand_gap[idxs]    = lg_b[:, 0:1] - lg_b
                cand_cos_cb[idxs] = (u_cand * u_base[:, None, :]).sum(axis=2) / (ub_n[:, None] * uc_n)
                cand_sreg[idxs]   = (tok_arr[np.clip(tk_b, 0, Vt-1)] == gold_reg[idxs, None])
                if sr:
                    Rr = len(reg_arr)
                    cand_regids = tok_arr[np.clip(tk_b, 0, Vt-1)]
                    cand_srids  = reg_arr[np.clip(cand_regids, 0, Rr-1)]
                    gold_sr_b   = reg_arr[np.clip(gold_reg[idxs], 0, Rr-1)]
                    cand_ssr[idxs] = (cand_srids == gold_sr_b[:, None])

    # ── Summary objects ──────────────────────────────────────────────────────
    row_geom = {
        "gip":             gip,
        "gold_rank":       gold_rank,
        "base_gap":        row_base_gap,
        "cosine_gb":       row_cos_gb,
        "dg_norm":         row_dg_norm,
        "req_dn":          row_req_dn,
        "delta_scale":     row_ds,
        "same_reg":        same_reg,
        "same_sr":         same_sr,
        "gold_reg":        gold_reg,
        "base_reg":        base_reg,
        "cand_reg_ent":    cand_reg_ent,
        "max_rel_proj":    row_max_rel_proj,
        "mean_rel_proj":   row_mean_rel_proj,
        "n_above_0p5":     row_n_above_0p5,
        "n_above_0p8":     row_n_above_0p8,
        "n_above_1p0":     row_n_above_1p0,
        "oracle_top1":     oracle_new_top1,
        "oracle_rank":     oracle_gold_rank,
        "oracle_wins":     oracle_gold_wins,
        "oracle_wrong_steals": oracle_wrong_steals,
        "oracle_margin":   oracle_margin_after,
    }
    cand_geom = {
        "gap":     cand_gap,
        "cos_cb":  cand_cos_cb,
        "rel_pr":  cand_rel_pr,
        "sreg":    cand_sreg,
        "ssr":     cand_ssr,
    }
    return row_geom, cand_geom


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Format scores (inline recompute)  — simple; already designed
# ═══════════════════════════════════════════════════════════════════════════════

_FMT_CATS = {
    "is_comma":      lambda s: s.strip() in (',',),
    "is_period":     lambda s: s.strip() in ('.', '…', '...'),
    "is_sent_end":   lambda s: s.strip() in ('.', '!', '?', '…', '...'),
    "is_quote":      lambda s: s.strip() in ('"', "'", '`', '"', '"', ''', ''', '``', "''"),
    "is_open_p":     lambda s: s.strip() == '(',
    "is_close_p":    lambda s: s.strip() == ')',
    "is_open_b":     lambda s: s.strip() == '[',
    "is_close_b":    lambda s: s.strip() == ']',
    "is_open_c":     lambda s: s.strip() == '{',
    "is_close_c":    lambda s: s.strip() == '}',
    "is_wiki_eq":    lambda s: s.strip() in ('=', '==', '===', '====', '====='),
    "is_at":         lambda s: '@' in s.strip(),
    "is_newline":    lambda s: ('\n' in s or '\r' in s) and not s.strip(),
    "is_format":     lambda s: False,  # filled below
}

def _build_fmt_arr(vocab_size=VS):
    arr = {k: np.zeros(vocab_size, bool) for k in _FMT_CATS}
    for tid in range(vocab_size):
        s = _dtok(tid)
        for k, fn in _FMT_CATS.items():
            if k != "is_format":
                arr[k][tid] = fn(s)
        arr["is_format"][tid] = any(
            arr[k][tid] for k in arr if k not in ("is_format",))
    return arr


def compute_format_scores_inline(topk_ids, prefix_ids, pool_size):
    """
    Returns format_scores [N, P] float32 and fires [N, 6] float32.
    Uses 6-heuristic system identical to eval_format_syntax_diagnostic_v1.
    """
    arr = _build_fmt_arr()
    N, P = topk_ids.shape
    M    = min(prefix_ids.shape[1], 128)
    ctx  = np.clip(prefix_ids[:, -M:], 0, VS - 1)

    def _s(k, c):  return arr[k][c].sum(axis=1).astype(np.int32)

    unm_p   = _s("is_open_p",  ctx) - _s("is_close_p", ctx)
    unm_b   = _s("is_open_b",  ctx) - _s("is_close_b", ctx)
    unm_c   = _s("is_open_c",  ctx) - _s("is_close_c", ctx)
    q_cnt   = _s("is_quote",   ctx)
    rec_eq  = _s("is_wiki_eq", ctx[:, -min(16, M):])
    rec_at  = _s("is_at",      ctx[:, -min(8,  M):])
    rec_com = _s("is_comma",   ctx[:, -min(16, M):])
    last    = ctx[:, -1]
    l_sent  = arr["is_sent_end"][last]
    l_nl    = arr["is_newline"][last]
    l_com   = arr["is_comma"][last]

    fires = np.zeros((N, 6), np.float32)
    fires[:, 0] = ((unm_p > 0) | (unm_b > 0) | (unm_c > 0)).astype(np.float32)
    fires[:, 1] = (q_cnt % 2).astype(np.float32)
    fires[:, 2] = (rec_eq > 0).astype(np.float32)
    fires[:, 3] = (rec_at > 0).astype(np.float32)
    fires[:, 4] = (l_sent | l_nl).astype(np.float32)
    fires[:, 5] = ((rec_com > 0) & ~l_com).astype(np.float32)

    # heuristic_applies [V, 6]
    close_any = arr["is_close_p"] | arr["is_close_b"] | arr["is_close_c"]
    H = np.stack([close_any, arr["is_quote"], arr["is_wiki_eq"],
                  arr["is_at"], arr["is_sent_end"], arr["is_comma"]], axis=1).astype(np.float32)

    topk_c  = np.clip(topk_ids, 0, VS - 1)
    applies = H[topk_c]                                   # [N, P, 6]
    scores  = (applies * fires[:, None, :]).sum(axis=2)   # [N, P]
    return scores.astype(np.float32), fires


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. Load expert aggregate metrics from files
# ═══════════════════════════════════════════════════════════════════════════════

def _read_csv_safe(path):
    if not path or not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _find_file(dirpath, patterns):
    """Find first existing file matching any pattern."""
    if not dirpath or not os.path.isdir(dirpath):
        return None
    for pat in patterns:
        matches = glob.glob(os.path.join(dirpath, pat))
        if matches:
            return matches[0]
    return None


def _extract_method_metrics(rows, method_name, policy_field="policy",
                             slice_filter="all", best_field="acc_gain"):
    """From a slice-metrics CSV, extract the best row for method on 'all' slice."""
    cands = [r for r in rows if r.get("slice", r.get("mode", "")) == slice_filter]
    if not cands:
        cands = rows
    if not cands:
        return None
    # Try to find best policy row
    for r in cands:
        for f in ["policy", "mode"]:
            if r.get(f, "").lower() in (method_name.lower(), "prefix_plus_doc_past",
                                         "base_plus_format", "base_plus_passage"):
                return r
    return cands[0] if cands else None


def load_expert_summaries(args):
    """
    Load aggregate metrics from each expert output directory.
    Returns dict: expert_name -> {acc_gain, changed_to_gold, changed_away,
                                   net_correction, bdr, source_file, per_row_available}
    """
    experts = {}

    # ── Format ─────────────────────────────────────────────────────────────
    fmt_path = _find_file(args.format_dir,
                          ["format_slice_metrics.csv", "format_policy_grid.csv"])
    if fmt_path:
        rows = _read_csv_safe(fmt_path)
        best = _extract_method_metrics(rows, "base_plus_format", slice_filter="all")
        if best:
            experts["format"] = {
                "source":        fmt_path,
                "per_row":       False,  # aggregate only; scores recomputed inline
                "acc_gain":      float(best.get("acc_gain", "nan")),
                "changed_to_gold": float(best.get("changed_to_gold", "nan")),
                "changed_away":  float(best.get("changed_away", "nan")),
                "net_correction": float(best.get("net_correction", "nan")),
                "bdr":           float(best.get("benefit_damage_ratio", "nan")),
            }
            print(f"[expert] format loaded from {fmt_path}")
    else:
        print(f"[expert] format: no metrics file in {args.format_dir!r} — will use inline scores")
        experts["format"] = {"source": "inline", "per_row": True,
                              "acc_gain": float("nan"), "changed_to_gold": float("nan"),
                              "changed_away": float("nan"), "net_correction": float("nan"),
                              "bdr": float("nan")}

    # ── Pointer ─────────────────────────────────────────────────────────────
    ptr_path = _find_file(args.pointer_dir,
                          ["*.csv", "pointer_*.csv", "slice_metrics.csv"])
    if ptr_path:
        rows = _read_csv_safe(ptr_path)
        best = _extract_method_metrics(rows, "pointer", slice_filter="all")
        if best is None and rows:
            best = rows[0]
        experts["pointer"] = {
            "source": ptr_path, "per_row": False,
            "acc_gain":       float(best.get("acc_gain", "nan")) if best else float("nan"),
            "changed_to_gold": float(best.get("changed_to_gold", "nan")) if best else float("nan"),
            "changed_away":   float(best.get("changed_away", "nan")) if best else float("nan"),
            "net_correction": float(best.get("net_correction", "nan")) if best else float("nan"),
            "bdr":            float(best.get("benefit_damage_ratio", "nan")) if best else float("nan"),
        }
        print(f"[expert] pointer loaded from {ptr_path}")
    else:
        print(f"[expert] pointer: no file in {args.pointer_dir!r}")

    # ── Fuzzy phrase ─────────────────────────────────────────────────────────
    fz_path = _find_file(args.fuzzy_dir, ["*.csv", "fuzzy_*.csv", "phrase_*.csv"])
    if fz_path:
        rows = _read_csv_safe(fz_path)
        best = _extract_method_metrics(rows, "fuzzy", slice_filter="all")
        if best is None and rows:
            best = rows[0]
        experts["fuzzy_phrase"] = {
            "source": fz_path, "per_row": False,
            "acc_gain":       float(best.get("acc_gain", "nan")) if best else float("nan"),
            "changed_to_gold": float(best.get("changed_to_gold", "nan")) if best else float("nan"),
            "changed_away":   float(best.get("changed_away", "nan")) if best else float("nan"),
            "net_correction": float(best.get("net_correction", "nan")) if best else float("nan"),
            "bdr":            float(best.get("benefit_damage_ratio", "nan")) if best else float("nan"),
        }
        print(f"[expert] fuzzy_phrase loaded from {fz_path}")
    else:
        print(f"[expert] fuzzy_phrase: no file in {args.fuzzy_dir!r}")

    # ── Detail memory ────────────────────────────────────────────────────────
    dt_path = _find_file(args.detail_dir,
                         ["ablation_results.csv", "eval_by_slice.csv", "final_metrics.json"])
    if dt_path:
        if dt_path.endswith(".json"):
            try:
                fm = json.load(open(dt_path))
                abl = fm.get("ablation", {})
                base_acc = abl.get("base_only", float("nan"))
                main_acc = abl.get("prefix_plus_doc_past", float("nan"))
                experts["detail_memory"] = {
                    "source": dt_path, "per_row": False,
                    "acc_gain": main_acc - base_acc if np.isfinite(main_acc) and np.isfinite(base_acc) else float("nan"),
                    "changed_to_gold": float("nan"), "changed_away": float("nan"),
                    "net_correction": float("nan"), "bdr": float("nan"),
                    "real_vs_shuffled": abl.get("prefix_plus_doc_past", float("nan")) -
                                        abl.get("shuffled_doc_past_control", float("nan")),
                }
                print(f"[expert] detail_memory loaded from {dt_path}")
            except Exception as e:
                print(f"[expert] detail_memory json parse error: {e}")
        else:
            rows = _read_csv_safe(dt_path)
            best = _extract_method_metrics(rows, "prefix_plus_doc_past", slice_filter="all")
            if best is None and rows:
                best = rows[0]
            experts["detail_memory"] = {
                "source": dt_path, "per_row": False,
                "acc_gain":       float(best.get("acc_gain", best.get("delta_acc", "nan"))) if best else float("nan"),
                "changed_to_gold": float("nan"), "changed_away": float("nan"),
                "net_correction":  float("nan"), "bdr": float("nan"),
            }
            print(f"[expert] detail_memory loaded from {dt_path}")
    else:
        print(f"[expert] detail_memory: no file in {args.detail_dir!r}")

    # ── Passage recall ───────────────────────────────────────────────────────
    pa_path = _find_file(args.passage_dir,
                         ["passage_slice_metrics.csv", "passage_policy_grid.csv"])
    if pa_path:
        rows = _read_csv_safe(pa_path)
        best = _extract_method_metrics(rows, "passage", slice_filter="all")
        if best is None and rows:
            best = rows[0]
        experts["passage_recall"] = {
            "source": pa_path, "per_row": False,
            "acc_gain":       float(best.get("acc_gain", "nan")) if best else float("nan"),
            "changed_to_gold": float(best.get("changed_to_gold", "nan")) if best else float("nan"),
            "changed_away":   float(best.get("changed_away", "nan")) if best else float("nan"),
            "net_correction": float(best.get("net_correction", "nan")) if best else float("nan"),
            "bdr":            float(best.get("benefit_damage_ratio", "nan")) if best else float("nan"),
        }
        print(f"[expert] passage_recall loaded from {pa_path}")
    else:
        print(f"[expert] passage_recall: no file in {args.passage_dir!r}")

    # ── Path oracle ──────────────────────────────────────────────────────────
    po_path = _find_file(args.path_dir, ["*.csv", "path_*.csv"])
    if po_path:
        rows = _read_csv_safe(po_path)
        best = _extract_method_metrics(rows, "path", slice_filter="all")
        if best is None and rows:
            best = rows[0]
        experts["path_oracle"] = {
            "source": po_path, "per_row": False,
            "acc_gain":       float(best.get("acc_gain", "nan")) if best else float("nan"),
            "changed_to_gold": float(best.get("changed_to_gold", "nan")) if best else float("nan"),
            "changed_away":   float(best.get("changed_away", "nan")) if best else float("nan"),
            "net_correction": float(best.get("net_correction", "nan")) if best else float("nan"),
            "bdr":            float(best.get("benefit_damage_ratio", "nan")) if best else float("nan"),
        }
        print(f"[expert] path_oracle loaded from {po_path}")
    else:
        print(f"[expert] path_oracle: no file in {args.path_dir!r}")

    return experts


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Expert logit-space geometry for format (inline scores available)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_format_expert_geometry(data, fmt_scores, fmt_fires, row_geom, lam=1.0):
    """
    Compute per-row expert geometry for the format heuristic.
    Returns dict of per-row arrays.
    """
    N     = len(data["gold"])
    gold  = data["gold"].astype(np.int64)
    ar    = np.arange(N)

    # Truncate topk/lgt to match the fmt_scores column count (topm_eval, not full pool)
    M     = fmt_scores.shape[1]
    topk  = data["topk"][:, :M].astype(np.int64)
    lgt   = data["lgt"][:, :M].astype(np.float32)

    # base_top1 from the full pool (index 0 is always the same)
    base_top1    = data["topk"][:, 0].astype(np.int64)
    base_correct = (base_top1 == gold)
    gip          = row_geom["gip"]
    # gold_rank may exceed M if gold is ranked beyond topm — clamp for score lookup
    gold_rank    = row_geom["gold_rank"]
    gold_in_M    = gip & (gold_rank < M)
    safe_gr      = np.clip(gold_rank, 0, M - 1)

    # Combined logits (both [N, M])
    combined = lgt + lam * fmt_scores
    fmt_pred  = topk[ar, combined.argmax(axis=1)]

    # Per-row support: expert score delta for gold vs base
    # Only meaningful when gold is within the top-M window
    gold_fmt_score = np.where(gold_in_M, fmt_scores[ar, safe_gr], float("nan"))
    base_fmt_score = fmt_scores[:, 0]
    support_gold_minus_base = gold_fmt_score - base_fmt_score

    # Expert sharpness
    exp_top1_idx = fmt_scores.argmax(axis=1)
    exp_top1     = fmt_scores[ar, exp_top1_idx]
    sorted_fs    = np.sort(fmt_scores, axis=1)[:, ::-1]
    exp_top2     = sorted_fs[:, 1] if M >= 2 else sorted_fs[:, 0]
    exp_margin   = exp_top1 - exp_top2
    exp_entropy  = -(fmt_scores * np.log(np.clip(fmt_scores + 1e-4, 1e-4, None))).sum(axis=1)
    exp_std      = fmt_scores.std(axis=1)

    # Max wrong support
    is_gold_c = (topk == gold[:, None])
    wrong_fs  = np.where(is_gold_c, -np.inf, fmt_scores)
    max_wrong = wrong_fs.max(axis=1)

    # Edit labels
    changed_to_gold  = ~base_correct & (fmt_pred == gold)
    changed_away     = base_correct  & (fmt_pred != gold)
    no_change        = (fmt_pred == base_top1)
    w2w              = ~base_correct & (fmt_pred != gold) & (fmt_pred != base_top1)

    return {
        "fmt_pred":               fmt_pred,
        "support_gold_minus_base": support_gold_minus_base,
        "max_wrong_support":      max_wrong,
        "exp_margin":             exp_margin,
        "exp_entropy":            exp_entropy,
        "exp_std":                exp_std,
        "changed_to_gold":        changed_to_gold,
        "changed_away":           changed_away,
        "no_change":              no_change,
        "w2w":                    w2w,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Near-miss bucket classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_near_miss_buckets(data, row_geom, fmt_geom):
    N    = len(data["gold"])
    gip  = row_geom["gip"]
    bg   = row_geom["base_gap"]
    mrp  = row_geom["max_rel_proj"]
    ow   = row_geom["oracle_wrong_steals"]
    ow_wins = row_geom["oracle_wins"]
    cos  = row_geom["cosine_gb"]
    sr   = row_geom["same_reg"]
    rdn  = row_geom["req_dn"]
    rank = row_geom["gold_rank"]

    low_gap  = gip & (bg < 2.0)
    high_cos = gip & (cos > 0.9)
    low_rdn  = gip & (rdn < 1.0)

    # Format support signal
    if fmt_geom is not None:
        fmt_sup  = fmt_geom["support_gold_minus_base"] > 0
        fmt_bad  = fmt_geom["changed_away"]
    else:
        fmt_sup  = np.zeros(N, bool)
        fmt_bad  = np.zeros(N, bool)

    buckets = np.full(N, "not_candidate_solved", dtype=object)
    buckets[~gip]                            = "not_candidate_solved"

    far_wrong = gip & ~low_gap & ~high_cos & (rank > 8)
    buckets[far_wrong]                       = "far_wrong"

    crowded = gip & (ow | (mrp > 0.8))
    buckets[crowded]                         = "crowded_near_miss"

    clean = gip & low_gap & ow_wins & ~crowded & (mrp < 0.5) & (sr)
    buckets[clean]                           = "clean_near_miss"

    ev_sup = gip & fmt_sup & ~clean & ~crowded
    buckets[ev_sup]                          = "evidence_supported_near_miss"

    ev_conf = gip & fmt_bad & ~clean
    buckets[ev_conf]                         = "evidence_conflicted_near_miss"

    # Remaining gip rows without a label → clean or crowded based on projections
    unlabeled = gip & (buckets == "not_candidate_solved")
    buckets[unlabeled & (mrp < 0.5)]        = "clean_near_miss"
    buckets[unlabeled & (mrp >= 0.5)]       = "crowded_near_miss"

    return buckets


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Slice building
# ═══════════════════════════════════════════════════════════════════════════════

def build_slices(data, row_geom, fmt_geom, buckets):
    N    = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk"].astype(np.int64)
    lgt  = data["lgt"].astype(np.float32)
    ar   = np.arange(N)

    base_top1    = topk[ar, lgt.argmax(axis=1)]
    base_correct = (base_top1 == gold)
    gip          = row_geom["gip"]
    rank         = row_geom["gold_rank"]

    slices = {
        "all":                      np.ones(N, bool),
        "base_correct":             base_correct,
        "bucketA_confuser":         ~base_correct & gip,
        "gold_in_pool":             gip,
        "gold_rank_1_5":            gip & (rank < 5),
        "gold_rank_6_32":           gip & (rank >= 5) & (rank < 32),
        "gold_rank_33_256":         gip & (rank >= 32),
        "same_region_confuser":     ~base_correct & gip & row_geom["same_reg"],
        "same_superregion_confuser": ~base_correct & gip & row_geom["same_sr"],
        "different_region_confuser": ~base_correct & gip & ~row_geom["same_reg"],
        "clean_near_miss":          buckets == "clean_near_miss",
        "crowded_near_miss":        buckets == "crowded_near_miss",
        "evidence_supported_near_miss": buckets == "evidence_supported_near_miss",
        "evidence_conflicted_near_miss": buckets == "evidence_conflicted_near_miss",
        "far_wrong":                buckets == "far_wrong",
        "not_candidate_solved":     buckets == "not_candidate_solved",
    }
    if fmt_geom is not None:
        slices["format_supported"]   = fmt_geom["changed_to_gold"]
        slices["format_hurt"]        = fmt_geom["changed_away"]

    print("[slices]")
    for s, m in slices.items():
        print(f"  {s:40s} = {m.sum():6,}/{N:,}")
    return slices


# ═══════════════════════════════════════════════════════════════════════════════
# Example collection
# ═══════════════════════════════════════════════════════════════════════════════

def collect_examples(data, row_geom, cand_geom, fmt_geom, buckets, slices, topm,
                     max_ex=50, seed=42):
    rng  = random.Random(seed)
    N    = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk"].astype(np.int64)
    lgt  = data["lgt"].astype(np.float32)
    ar   = np.arange(N)
    M    = min(topk.shape[1], topm)

    base_top1    = topk[ar, lgt.argmax(axis=1)]
    base_correct = (base_top1 == gold)

    def sample(mask, n):
        idx = np.where(mask)[0].tolist()
        rng.shuffle(idx)
        return idx[:n]

    def make_ex(i):
        prefix = data["ids"][i, -24:]
        cands  = []
        for j in range(min(M, 16)):
            cands.append({
                "rank":   j,
                "tok":    _dtok(topk[i, j]),
                "tid":    int(topk[i, j]),
                "lgt":    float(lgt[i, j]),
                "gap_from_base": float(cand_geom["gap"][i, j]) if j < cand_geom["gap"].shape[1] else float("nan"),
                "cos_cb": float(cand_geom["cos_cb"][i, j]) if j < cand_geom["cos_cb"].shape[1] else float("nan"),
                "rel_pr": float(cand_geom["rel_pr"][i, j]) if j < cand_geom["rel_pr"].shape[1] else float("nan"),
                "sreg":   bool(cand_geom["sreg"][i, j]) if j < cand_geom["sreg"].shape[1] else False,
                "fmt":    float(fmt_geom["support_gold_minus_base"][i]) if fmt_geom and j == row_geom["gold_rank"][i] else float("nan"),
                "is_gold": int(topk[i, j]) == int(gold[i]),
                "is_base": j == 0,
            })
        if fmt_geom is not None:
            adj = lgt[i, :M] + fmt_geom["support_gold_minus_base"][i] if False else lgt[i, :M]
        return {
            "i":          i,
            "row_id":     int(data["row_ids"][i]),
            "offset":     int(data["token_offset"][i]),
            "prefix":     _dids(prefix),
            "gold_tok":   _dtok(gold[i]),
            "base_tok":   _dtok(base_top1[i]),
            "gold_rank":  int(row_geom["gold_rank"][i]),
            "base_gap":   float(row_geom["base_gap"][i]),
            "cosine_gb":  float(row_geom["cosine_gb"][i]),
            "req_dn":     float(row_geom["req_dn"][i]),
            "same_reg":   bool(row_geom["same_reg"][i]),
            "same_sr":    bool(row_geom["same_sr"][i]),
            "max_rp":     float(row_geom["max_rel_proj"][i]),
            "oracle_top1": _dtok(row_geom["oracle_top1"][i]),
            "oracle_wins": bool(row_geom["oracle_wins"][i]),
            "oracle_wrong_steals": bool(row_geom["oracle_wrong_steals"][i]),
            "bucket":     str(buckets[i]),
            "candidates": cands,
        }

    def fmt_ex(ex):
        lines = [
            f"**Row {ex['i']}** row_id={ex['row_id']} offset={ex['offset']} bucket={ex['bucket']}",
            f"- prefix: `{ex['prefix'][:80]}`",
            f"- gold=`{ex['gold_tok']}` base=`{ex['base_tok']}` gold_rank={ex['gold_rank']}",
            f"- base_gap={ex['base_gap']:.3f} cos_gb={ex['cosine_gb']:.3f} "
            f"req_dn={ex['req_dn']:.3f} same_reg={ex['same_reg']} max_rp={ex['max_rp']:.3f}",
            f"- oracle_top1=`{ex['oracle_top1']}` oracle_wins={ex['oracle_wins']} "
            f"oracle_wrong_steals={ex['oracle_wrong_steals']}",
            "",
            "| rank | tok | lgt | gap_from_base | cos_cb | rel_pr | sreg | gold | base |",
            "|------|-----|-----|---------------|--------|--------|------|------|------|",
        ]
        for c in ex["candidates"]:
            lines.append(
                f"| {c['rank']} | `{c['tok'][:12]}` | {c['lgt']:.3f} | "
                f"{c['gap_from_base']:.3f} | {c['cos_cb']:.3f} | "
                f"{c['rel_pr']:.3f} | {c['sreg']} | "
                f"{'yes' if c['is_gold'] else ''} | {'yes' if c['is_base'] else ''} |"
            )
        return "\n".join(lines) + "\n"

    def write_exs(path, mask, header):
        exs = [make_ex(i) for i in sample(mask, max_ex)]
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {header}\n\n({len(exs)} examples)\n\n")
            for ex in exs:
                f.write("---\n")
                f.write(fmt_ex(ex))
                f.write("\n")
        print(f"[save] {path}")

    return write_exs


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _wcsv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {path}")


def _nanf(x):
    return float(x) if x is not None and np.isfinite(float(x)) else float("nan")


def build_output_rows(data, row_geom, cand_geom, fmt_geom, buckets, slices, topm):
    N    = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk"].astype(np.int64)
    lgt  = data["lgt"].astype(np.float32)
    ar   = np.arange(N)
    M    = min(topk.shape[1], topm)

    # row_geometry_metrics.csv
    row_rows = []
    for i in range(N):
        row_rows.append({
            "row_id":          int(data["row_ids"][i]),
            "token_offset":    int(data["token_offset"][i]),
            "gold_id":         int(gold[i]),
            "base_id":         int(topk[i, 0]),
            "gold_rank":       int(row_geom["gold_rank"][i]),
            "gold_in_pool":    bool(row_geom["gip"][i]),
            "base_gap":        _nanf(row_geom["base_gap"][i]),
            "cosine_gb":       _nanf(row_geom["cosine_gb"][i]),
            "dg_norm":         _nanf(row_geom["dg_norm"][i]),
            "req_dn":          _nanf(row_geom["req_dn"][i]),
            "delta_scale":     _nanf(row_geom["delta_scale"][i]),
            "same_reg":        bool(row_geom["same_reg"][i]),
            "same_sr":         bool(row_geom["same_sr"][i]),
            "cand_reg_ent":    _nanf(row_geom["cand_reg_ent"][i]),
            "max_rel_proj":    _nanf(row_geom["max_rel_proj"][i]),
            "mean_rel_proj":   _nanf(row_geom["mean_rel_proj"][i]),
            "n_above_0p5":     int(row_geom["n_above_0p5"][i]),
            "n_above_0p8":     int(row_geom["n_above_0p8"][i]),
            "n_above_1p0":     int(row_geom["n_above_1p0"][i]),
            "oracle_top1_id":  int(row_geom["oracle_top1"][i]),
            "oracle_gold_rank": int(row_geom["oracle_rank"][i]),
            "oracle_wins":     bool(row_geom["oracle_wins"][i]),
            "oracle_wrong_steals": bool(row_geom["oracle_wrong_steals"][i]),
            "oracle_margin":   _nanf(row_geom["oracle_margin"][i]),
            "near_miss_bucket": str(buckets[i]),
        })

    # candidate_geometry_metrics.csv  (sample)
    sample_n = min(N, CSAMP)
    rng_c    = np.random.default_rng(42)
    sample_i = rng_c.choice(N, sample_n, replace=False) if N > sample_n else np.arange(N)
    cand_rows = []
    for i in sample_i:
        for j in range(M):
            cand_rows.append({
                "row_id":       int(data["row_ids"][i]),
                "cand_rank":    j,
                "cand_id":      int(topk[i, j]),
                "base_id":      int(topk[i, 0]),
                "gold_id":      int(gold[i]),
                "cand_gap":     _nanf(cand_geom["gap"][i, j]),
                "cos_cb":       _nanf(cand_geom["cos_cb"][i, j]),
                "rel_pr":       _nanf(cand_geom["rel_pr"][i, j]),
                "sreg_as_base": bool(cand_geom["sreg"][i, j]),
                "ssr_as_base":  bool(cand_geom["ssr"][i, j]),
                "is_gold":      int(topk[i, j]) == int(gold[i]),
                "is_base":      j == 0,
                "fmt_score":    _nanf(fmt_geom["support_gold_minus_base"][i]) if fmt_geom and (int(topk[i, j]) == int(gold[i])) else float("nan"),
            })

    # slice summaries
    gold_np = gold
    topk_np = topk
    lgt_np  = lgt
    base_t1 = topk_np[ar, lgt_np.argmax(axis=1)]
    base_ok  = (base_t1 == gold_np)

    slice_rows = []
    for sname, mask in slices.items():
        n = int(mask.sum())
        if n == 0:
            slice_rows.append({"slice": sname, "n": n, "base_acc": float("nan"),
                                "gip_rate": float("nan"), "mean_base_gap": float("nan"),
                                "mean_cosine_gb": float("nan"), "mean_max_rp": float("nan"),
                                "frac_oracle_wins": float("nan"),
                                "frac_oracle_wrong_steals": float("nan"),
                                "frac_same_reg": float("nan")})
            continue
        slice_rows.append({
            "slice":           sname,
            "n":               n,
            "base_acc":        float(base_ok[mask].mean()),
            "gip_rate":        float(row_geom["gip"][mask].mean()),
            "mean_base_gap":   _nanmean(row_geom["base_gap"][mask]),
            "mean_cosine_gb":  _nanmean(row_geom["cosine_gb"][mask]),
            "mean_max_rp":     _nanmean(row_geom["max_rel_proj"][mask]),
            "frac_oracle_wins": float(row_geom["oracle_wins"][mask].mean()),
            "frac_oracle_wrong_steals": float(row_geom["oracle_wrong_steals"][mask].mean()),
            "frac_same_reg":   float(row_geom["same_reg"][mask].mean()),
        })

    return row_rows, cand_rows, slice_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(args, data, row_geom, fmt_geom, experts, slices, buckets, audit, uinfo, out,
                 validity=None):
    N    = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk"].astype(np.int64)
    lgt  = data["lgt"].astype(np.float32)
    ar   = np.arange(N)
    base_ok = (topk[ar, lgt.argmax(axis=1)] == gold)
    gip     = row_geom["gip"]

    def pct(mask):
        return 100.0 * int(mask.sum()) / max(N, 1)

    def mnan(arr, mask=None):
        v = arr[mask] if mask is not None else arr
        return _nanmean(v)

    rpt = os.path.join(out, "geometry_evidence_interaction_report.md")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("# Geometry + Evidence Interaction Diagnostic V2 — Report\n\n")
        f.write("> GOLD USED FOR DIAGNOSTICS/METRICS ONLY. Non-deployable oracle sections marked.\n\n")

        # VALIDITY STATUS block (always the first thing in the report)
        v = validity or {}
        v_ok = v.get("validity_ok", True)
        v_status = "PASS" if v_ok else "FAIL"
        _rpt_gip_n = int(row_geom["gip"].sum())  # local to write_report scope
        f.write("## VALIDITY STATUS\n\n")
        f.write(f"**{v_status}**\n\n")
        f.write(f"| field | value |\n|-------|-------|\n")
        f.write(f"| unembedding_source | `{uinfo.get('unembedding_source','?')}` |\n")
        f.write(f"| U_shape | {uinfo.get('U_shape', '?')} |\n")
        f.write(f"| d_model | {uinfo.get('d_model','?')} |\n")
        f.write(f"| N_val | {v.get('N_val', N):,} |\n")
        f.write(f"| gold_in_pool | {v.get('gip_n', _rpt_gip_n):,} / {N:,} "
                f"({100*v.get('gip_n', _rpt_gip_n)/max(N,1):.1f}%) |\n")
        f.write(f"| oracle_wins | {v.get('n_oracle_wins', int(row_geom['oracle_wins'].sum())):,} |\n")
        f.write(f"| oracle_wrong_steals | {v.get('n_oracle_steals', int(row_geom['oracle_wrong_steals'].sum())):,} |\n")
        nan_rates = v.get("nan_rates", {})
        for fname, nr in nan_rates.items():
            flag = " ⚠" if (isinstance(nr, float) and np.isfinite(nr) and nr > 0.001) else ""
            f.write(f"| nan_rate[{fname}] | {nr:.5f}{flag} |\n")
        if v.get("validity_messages"):
            f.write("\n**Validity issues:**\n")
            for msg in v["validity_messages"]:
                f.write(f"- {msg}\n")
        f.write("\n---\n\n")

        f.write(f"**N_val:** {N:,}  |  **pool_size:** {args.candidate_pool_size}  |  "
                f"**topm_eval:** {args.topm_eval}  |  "
                f"**unembedding:** {uinfo.get('unembedding_source','?')}  |  "
                f"**d_model:** {uinfo.get('d_model','?')}\n\n---\n\n")

        # ── Q1 ───────────────────────────────────────────────────────────────
        f.write("## Q1: Are Type-A errors mostly almost-right base hypotheses?\n\n")
        n_ba = int((~base_ok & gip).sum())
        n_gip = int(gip.sum())
        f.write(f"- gold_in_pool rows: {n_gip:,} / {N:,} ({pct(gip):.1f}%)\n")
        f.write(f"- BucketA (base_wrong & gold_in_pool): {n_ba:,} ({100*n_ba/max(N,1):.1f}%)\n")
        f.write(f"- Of BucketA: mean base_gap={mnan(row_geom['base_gap'], ~base_ok & gip):.3f}\n")
        f.write(f"- Of BucketA: mean cosine_gb={mnan(row_geom['cosine_gb'], ~base_ok & gip):.3f}\n")
        f.write(f"- Of BucketA: frac same_region={mnan(row_geom['same_reg'].astype(float), ~base_ok & gip):.3f}\n")
        thresh = 2.0
        low_gap = gip & ~base_ok & (row_geom["base_gap"] < thresh)
        f.write(f"- Of BucketA: frac low_gap (<{thresh}): "
                f"{100*int(low_gap.sum())/max(n_ba,1):.1f}%\n")
        q1 = "YES" if pct(gip) > 40 else ("PARTIAL" if pct(gip) > 20 else "NO")
        f.write(f"- **{q1}** — "
                f"{'Most errors have gold in the candidate pool' if q1=='YES' else 'Many errors lack gold in pool'}\n\n")

        # ── Q2 ───────────────────────────────────────────────────────────────
        f.write("## Q2: When gold is in pool, is it a small correction from base?\n\n")
        n_oracle_win = int(row_geom["oracle_wins"].sum())
        n_wrong_steal = int(row_geom["oracle_wrong_steals"].sum())
        f.write(f"- (ORACLE — non-deployable) oracle_wins: {n_oracle_win:,} / {n_gip:,} "
                f"({100*n_oracle_win/max(n_gip,1):.1f}%)\n")
        f.write(f"- oracle_wrong_steals: {n_wrong_steal:,} / {n_gip:,} "
                f"({100*n_wrong_steal/max(n_gip,1):.1f}%)\n")
        f.write(f"- mean req_dn on gip rows: {mnan(row_geom['req_dn'], gip):.3f}\n")
        q2 = ("YES — correction often clean" if 100*n_oracle_win/max(n_gip,1) > 50
              else "PARTIAL — many oracle corrections are crowded")
        f.write(f"- **{q2}**\n\n")

        # ── Q3 ───────────────────────────────────────────────────────────────
        f.write("## Q3: Are correction directions crowded?\n\n")
        f.write(f"- mean max_wrong_relative_projection (gip): "
                f"{mnan(row_geom['max_rel_proj'], gip):.3f}\n")
        f.write(f"- frac with max_rp > 0.8: "
                f"{100*int((row_geom['max_rel_proj'][gip]>0.8).sum())/max(n_gip,1):.1f}%\n")
        f.write(f"- frac with max_rp > 1.0 (wrong candidate leapfrogs): "
                f"{100*int((row_geom['max_rel_proj'][gip]>1.0).sum())/max(n_gip,1):.1f}%\n")
        crowded_pct = 100 * int((buckets == "crowded_near_miss").sum()) / max(n_gip, 1)
        f.write(f"- crowded_near_miss bucket: {crowded_pct:.1f}% of gip rows\n")
        clean_pct   = 100 * int((buckets == "clean_near_miss").sum())   / max(n_gip, 1)
        f.write(f"- clean_near_miss bucket:   {clean_pct:.1f}% of gip rows\n")
        q3 = "HIGH" if crowded_pct > 40 else ("MODERATE" if crowded_pct > 20 else "LOW")
        f.write(f"- **Crowding level: {q3}** — "
                f"mixer must account for collateral logit changes.\n\n")

        # ── Q4 ───────────────────────────────────────────────────────────────
        f.write("## Q4: Why did pointer and fuzzy phrase work better?\n\n")
        for ename in ["pointer", "fuzzy_phrase"]:
            if ename in experts:
                e = experts[ename]
                f.write(f"- {ename}: acc_gain={_nanf(e.get('acc_gain')):.4f}  "
                        f"changed_to_gold={_nanf(e.get('changed_to_gold')):.4f}  "
                        f"changed_away={_nanf(e.get('changed_away')):.4f}  "
                        f"BDR={_nanf(e.get('bdr')):.2f}\n")
            else:
                f.write(f"- {ename}: NOT AVAILABLE\n")
        f.write("\n*Hypothesis (verify with per-row scores):*\n")
        f.write("Pointer works because pointer-match evidence is highly candidate-specific"
                " (token X appears literally in the pointer context); "
                "this creates sharp support for exactly one candidate.\n")
        f.write("Fuzzy phrase works on the same-neighborhood recall: the retrieved phrase "
                "pins the correction to within the same region, reducing wrong-steal risk.\n\n")

        # ── Q5 ───────────────────────────────────────────────────────────────
        f.write("## Q5: Do successful experts have sharper evidence?\n\n")
        if fmt_geom is not None:
            f.write(f"- Format (inline): mean_support_gold_minus_base="
                    f"{mnan(fmt_geom['support_gold_minus_base'], gip):.4f}  "
                    f"mean_max_wrong_support={mnan(fmt_geom['max_wrong_support'], gip):.4f}  "
                    f"mean_entropy={mnan(fmt_geom['exp_entropy']):.4f}\n")
            ctg  = fmt_geom["changed_to_gold"].sum()
            caw  = fmt_geom["changed_away"].sum()
            f.write(f"  changed_to_gold={ctg:,}  changed_away={caw:,}  "
                    f"BDR={ctg/max(caw,1):.2f}\n")
        f.write("- (Per-row scores unavailable for pointer/fuzzy/passage/detail "
                "from saved files — aggregate metrics only.)\n\n")

        # ── Q6 ───────────────────────────────────────────────────────────────
        f.write("## Q6: Do failed experts cause changed_away due to broad/low-margin evidence?\n\n")
        for ename in ["passage_recall", "detail_memory", "path_oracle"]:
            if ename in experts:
                e = experts[ename]
                f.write(f"- {ename}: acc_gain={_nanf(e.get('acc_gain')):.4f}  "
                        f"changed_away={_nanf(e.get('changed_away')):.4f}  "
                        f"BDR={_nanf(e.get('bdr')):.2f}  "
                        f"real_vs_shuffled={_nanf(e.get('real_vs_shuffled',float('nan'))):.4f}\n")
            else:
                f.write(f"- {ename}: NOT AVAILABLE\n")
        f.write("\n*Hypothesis:* Failed experts either have diffuse candidate scores "
                "(high entropy) or their top-1 selection is not the gold token, "
                "leading to changed_away damage that outweighs changed_to_gold gains.\n")
        if "detail_memory" in experts:
            rvs = _nanf(experts["detail_memory"].get("real_vs_shuffled", float("nan")))
            if np.isfinite(rvs) and abs(rvs) < 0.001:
                f.write("- detail_memory: real ~= shuffled — memory NOT causal.\n")
            elif np.isfinite(rvs):
                f.write(f"- detail_memory: real - shuffled = {rvs:+.4f} — "
                        f"{'memory IS causal' if rvs > 0.002 else 'marginal causality'}.\n")
        f.write("\n")

        # ── Q7 ───────────────────────────────────────────────────────────────
        f.write("## Q7: Should Mixer V1 be a conservative relative-correction controller?\n\n")
        f.write(f"- Crowding level: **{q3}**\n")
        f.write(f"- Oracle wrong-steal rate: {100*n_wrong_steal/max(n_gip,1):.1f}%\n")
        f.write(f"- Base accuracy (overall): {100*int(base_ok.sum())/max(N,1):.1f}%\n")
        if q3 == "HIGH":
            f.write("- **YES** — crowding is HIGH. Absolute reranking risks boosting wrong "
                    "candidates that are geometrically aligned with the correction direction. "
                    "The mixer should predict whether to apply a *relative* score boost to a "
                    "specific candidate, not rerank from scratch. Default to no-op unless "
                    "expert evidence is sharp and candidate-specific.\n\n")
        elif q3 == "MODERATE":
            f.write("- **LIKELY YES** — crowding is MODERATE. Some correction directions "
                    "will drag wrong candidates; mixer should still prefer relative correction "
                    "with evidence sharpness gating.\n\n")
        else:
            f.write("- **UNCERTAIN** — crowding is LOW. Oracle corrections mostly succeed "
                    "without wrong-steals. A conservative mixer is still safe; revisit if "
                    "crowding grows with larger pools or different domains.\n\n")

        # ── Near-miss bucket counts ─────────────────────────────────────────
        f.write("---\n\n## Near-miss Bucket Summary\n\n")
        f.write("| bucket | count | % of gip |\n|--------|-------|----------|\n")
        for bname in ["clean_near_miss", "crowded_near_miss",
                       "evidence_supported_near_miss", "evidence_conflicted_near_miss",
                       "far_wrong", "not_candidate_solved"]:
            cnt = int((buckets == bname).sum())
            f.write(f"| {bname:40s} | {cnt:6,} | {100*cnt/max(n_gip,1):.1f}% |\n")

        # ── Geometry slice summary ──────────────────────────────────────────
        f.write("\n---\n\n## Key Geometry Metrics by Slice\n\n")
        f.write("| slice | n | base_acc | gip_rate | mean_gap | mean_cos | "
                "mean_max_rp | oracle_wins |\n")
        f.write("|-------|---|----------|----------|----------|----------|"
                "------------|-------------|\n")
        for sname, mask in slices.items():
            n = int(mask.sum())
            if n == 0:
                continue
            f.write(f"| {sname[:30]:30s} | {n:5,} | "
                    f"{float(base_ok[mask].mean()):.4f} | "
                    f"{float(row_geom['gip'][mask].mean()):.4f} | "
                    f"{_nanmean(row_geom['base_gap'][mask]):.3f} | "
                    f"{_nanmean(row_geom['cosine_gb'][mask]):.3f} | "
                    f"{_nanmean(row_geom['max_rel_proj'][mask]):.3f} | "
                    f"{float(row_geom['oracle_wins'][mask].mean()):.3f} |\n")

        # ── Method comparison ───────────────────────────────────────────────
        f.write("\n---\n\n## Method Comparison Table\n\n")
        f.write("| method | acc_gain | changed_to_gold | changed_away | BDR | source |\n")
        f.write("|--------|----------|-----------------|--------------|-----|--------|\n")
        for ename, e in sorted(experts.items()):
            f.write(f"| {ename:20s} | {_nanf(e.get('acc_gain')):.4f} | "
                    f"{_nanf(e.get('changed_to_gold')):.4f} | "
                    f"{_nanf(e.get('changed_away')):.4f} | "
                    f"{_nanf(e.get('bdr')):.2f} | "
                    f"{os.path.basename(str(e.get('source','?')))} |\n")

        # ── Mixer V1 feature recommendations ────────────────────────────────
        f.write("\n---\n\n## Mixer V1 Feature Recommendations\n\n")
        f.write("### Design Principle\n")
        f.write("> Mixer V1 should **predict whether to move from base to candidate**, "
                "not predict the token from scratch. Default to no-op unless evidence is "
                "sharp and base-relative.\n\n")
        f.write("### Base-relative geometry features\n")
        for feat in ["candidate_rank (logit position relative to base)",
                     "candidate_vs_base_logit_gap (base_logit - cand_logit)",
                     "base_margin (top1 - top2 logit gap)",
                     "cosine_candidate_base (U[cand] · U[base])",
                     "same_region_as_base",
                     "same_superregion_as_base",
                     "candidate_crowding_risk (max_rel_proj for moving to this cand)",
                     "required_delta_norm_proxy (|base_logit - cand_logit| / ||U[cand]-U[base]||)"]:
            f.write(f"- {feat}\n")
        f.write("\n### Expert relative evidence features\n")
        for feat in ["pointer_score_candidate_minus_base",
                     "fuzzy_score_candidate_minus_base",
                     "format_score_candidate_minus_base",
                     "detail_score_candidate_minus_base (if available)",
                     "passage_vote_candidate_minus_base (if available)"]:
            f.write(f"- {feat}\n")
        f.write("\n### Expert reliability features\n")
        for feat in ["expert_margin (top1 - top2 expert score)",
                     "expert_entropy (low = sharp = more reliable)",
                     "expert_agreement_count (# experts boosting this candidate)",
                     "expert_disagreement_score (std of expert scores)",
                     "real_vs_shuffled_gap (if applicable — causal signal)"]:
            f.write(f"- {feat}\n")
        f.write("\n### No-op safety features\n")
        for feat in ["base_confidence (softmax of base logits)",
                     "edit_risk_score (base_margin * crowding_proxy)",
                     "changed_away_risk_proxy (cand not in same region as base, low expert margin)"]:
            f.write(f"- {feat}\n")
        f.write("\n### Architecture\n")
        f.write("- Input: `[candidate_features, expert_scores, base_features]` per candidate\n")
        f.write("- Output: `delta_score[candidate]` — additive correction to base logit\n")
        f.write("- Training: only on rows where gold in pool AND "
                "at least one expert has positive signal\n")
        f.write("- Loss: CE over `base_logit + delta` where delta=0 at init (identity)\n")
        f.write("- Regularize: L1 or sparsity on delta to encourage no-op default\n\n")

        f.write(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"[save] {rpt}")
    return rpt


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse():
    p = argparse.ArgumentParser(description="Geometry + Evidence Interaction Diagnostic V2")
    p.add_argument("--val_dir",             required=True)
    p.add_argument("--small_ckpt",          default=None)
    p.add_argument("--token_to_region",     default=None)
    p.add_argument("--super_map",           default=None)
    # Expert dirs (all optional)
    p.add_argument("--pointer_dir",         default=None)
    p.add_argument("--fuzzy_dir",           default=None)
    p.add_argument("--format_dir",          default=None)
    p.add_argument("--detail_dir",          default=None)
    p.add_argument("--passage_dir",         default=None)
    p.add_argument("--path_dir",            default=None)
    p.add_argument("--output_dir",          required=True)
    # Params
    p.add_argument("--candidate_pool_size", type=int, default=256)
    p.add_argument("--topm_eval",           type=int, default=64)
    p.add_argument("--max_val_rows",        type=int, default=None)
    p.add_argument("--example_count",       type=int, default=50)
    p.add_argument("--seed",                type=int, default=42)
    return p, p.parse_args()


def main():
    _, args = _parse()
    np.random.seed(args.seed)
    random.seed(args.seed)
    t0 = time.time()

    print("=" * 64)
    print(" Geometry + Evidence Interaction Diagnostic V2")
    print(f" val_dir:    {args.val_dir}")
    print(f" output_dir: {args.output_dir}")
    print("=" * 64)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 0. Load shards ───────────────────────────────────────────────────
    print("\n[step 0] loading val shards...")
    data = load_val_shards(args.val_dir, args.candidate_pool_size, args.max_val_rows)
    N    = len(data["gold"])

    # ── 1. Unembedding ───────────────────────────────────────────────────
    print("\n[step 1] loading unembedding matrix...")
    U, uinfo = load_unembedding(args.small_ckpt)
    if U is not None:
        print(f"[U] shape={U.shape}  norm_mean={np.linalg.norm(U, axis=1).mean():.4f}")

    # If h_raw available and U available, compute full top-P logits
    if U is not None and "h_raw" in data:
        print("[step 1b] h_raw detected — computing full logits for top candidate expansion")
        P = args.candidate_pool_size
        d = U.shape[1]
        if data["h_raw"].shape[1] == d:
            new_topk = np.zeros((N, P), dtype=np.int64)
            new_lgt  = np.full((N, P), -1e9, dtype=np.float32)
            for s in range(0, N, GEO_B):
                e   = min(s + GEO_B, N)
                Hb  = data["h_raw"][s:e].astype(np.float32)
                fl  = Hb @ U.T                                    # [B, V]
                topP_idx = np.argpartition(-fl, P, axis=1)[:, :P]
                for ii in range(e - s):
                    order = np.argsort(-fl[ii, topP_idx[ii]])
                    new_topk[s + ii] = topP_idx[ii][order]
                    new_lgt[s + ii]  = fl[ii, new_topk[s + ii]]
            data["topk"] = new_topk
            data["lgt"]  = new_lgt
            print(f"[step 1b] updated topk to top-{P} from h_raw  "
                  f"gold_in_pool={( (new_topk == data['gold'][:, None]).any(1) ).mean():.4f}")
        else:
            print(f"[step 1b] h_raw dim {data['h_raw'].shape[1]} != U dim {d} — skipping")

    # ── 2. Hidden state audit ────────────────────────────────────────────
    print("\n[step 2] hidden state audit...")
    audit = audit_hidden_states(data, U)
    with open(os.path.join(args.output_dir, "hidden_state_audit.json"), "w") as f:
        json.dump({**audit, **uinfo}, f, indent=2, default=str)
    print(f"[save] {args.output_dir}/hidden_state_audit.json")

    # ── Region maps ──────────────────────────────────────────────────────
    print("\n[step 3] loading region maps...")
    tok_arr, reg_arr, unk_r, unk_s, sr = load_region_maps(
        args.token_to_region, args.super_map)
    print(f"[maps] tok_arr={len(tok_arr)}  sr={sr}")

    # ── 3-5. Core geometry ───────────────────────────────────────────────
    print(f"\n[step 4] computing geometry (N={N:,}, pool={args.candidate_pool_size}, "
          f"topm={args.topm_eval})...")
    row_geom, cand_geom = compute_geometry(
        data, U, tok_arr, reg_arr, unk_r, unk_s, sr, args.topm_eval)
    gip_n = int(row_geom["gip"].sum())
    n_oracle_wins   = int(row_geom["oracle_wins"].sum())
    n_oracle_steals = int(row_geom["oracle_wrong_steals"].sum())
    print(f"[geometry] gip={gip_n:,}  "
          f"oracle_wins={n_oracle_wins:,}  "
          f"oracle_wrong_steals={n_oracle_steals:,}")

    # ── NaN guard: all required gip-row fields must be finite ────────────────
    print("\n[step 4b] NaN validity audit...")
    gip_mask = row_geom["gip"]
    _req_fields = {
        "base_gap":        row_geom["base_gap"],
        "cosine_gb":       row_geom["cosine_gb"],
        "req_dn":          row_geom["req_dn"],
        "max_rel_proj":    row_geom["max_rel_proj"],
    }
    nan_rates = {}
    validity_ok = True
    validity_msgs = []
    for fname, arr in _req_fields.items():
        if gip_n > 0:
            nr = float(np.isnan(arr[gip_mask]).mean())
        else:
            nr = float("nan")
        nan_rates[fname] = nr
        print(f"  nan_rate[{fname}] = {nr:.5f}")
        if gip_n > 0 and nr > 0.001:
            msg = f"FAIL: {fname} nan_rate={nr:.5f} > 0.001 on {gip_n:,} gip rows"
            print(f"  [VALIDITY] {msg}")
            validity_msgs.append(msg)
            validity_ok = False

    # Extra check: if U is loaded but cosine is all-nan, something went wrong
    if U is not None and gip_n > 0 and nan_rates.get("cosine_gb", 1.0) > 0.999:
        msg = "FAIL: U loaded but cosine_gb is all-nan — U dims may not match hidden states"
        print(f"  [VALIDITY] {msg}")
        validity_msgs.append(msg)
        validity_ok = False

    # Oracle debug: if gip_n > 0 but oracle counts are both 0, something is wrong
    if gip_n > 0 and n_oracle_wins == 0 and n_oracle_steals == 0 and U is not None:
        print("[VALIDITY] WARNING: oracle_wins=0 AND oracle_wrong_steals=0 with gip_n>0")
        print("  Printing 20 debug rows to diagnose:")
        gold_np_  = data["gold"].astype(np.int64)
        topk_np_  = data["topk"].astype(np.int64)
        lgt_np_   = data["lgt"].astype(np.float32)
        gip_idx_  = np.where(gip_mask)[0][:20]
        for ii in gip_idx_:
            gr = int(row_geom["gold_rank"][ii])
            bg = float(row_geom["base_gap"][ii])
            rdn = float(row_geom["req_dn"][ii])
            cos = float(row_geom["cosine_gb"][ii])
            mrp = float(row_geom["max_rel_proj"][ii])
            ow  = bool(row_geom["oracle_wins"][ii])
            print(f"  row={ii} gold_rank={gr} base_gap={bg:.3f} req_dn={rdn:.3f} "
                  f"cos={cos:.3f} max_rp={mrp:.3f} oracle_wins={ow}")
        validity_msgs.append(
            "WARNING: oracle_wins=0 AND oracle_wrong_steals=0 — "
            "this is suspicious if gip_n is large")

    # Write validity audit JSON
    validity_audit = {
        "validity_ok":        validity_ok,
        "gip_n":              gip_n,
        "N_val":              N,
        "n_oracle_wins":      n_oracle_wins,
        "n_oracle_steals":    n_oracle_steals,
        "nan_rates":          nan_rates,
        "U_loaded":           U is not None,
        "U_shape":            list(U.shape) if U is not None else None,
        "unembedding_source": uinfo.get("unembedding_source", "?"),
        "d_model":            uinfo.get("d_model", "?"),
        "validity_messages":  validity_msgs,
    }
    with open(os.path.join(args.output_dir, "geometry_validity_audit.json"), "w") as f:
        json.dump(validity_audit, f, indent=2, default=str)
    print(f"[save] {args.output_dir}/geometry_validity_audit.json")

    # Write unembedding audit JSON
    with open(os.path.join(args.output_dir, "unembedding_audit.json"), "w") as f:
        json.dump(uinfo, f, indent=2, default=str)
    print(f"[save] {args.output_dir}/unembedding_audit.json")

    if not validity_ok:
        raise RuntimeError(
            "Geometry validity check FAILED. Refusing to write interpretive report.\n"
            + "\n".join(validity_msgs))

    # ── 6. Format scores (inline) ────────────────────────────────────────
    print("\n[step 5] computing format scores (inline)...")
    fmt_scores, fmt_fires = compute_format_scores_inline(
        data["topk"][:, :args.topm_eval],
        data["ids"],
        args.topm_eval)
    fmt_geom = analyze_format_expert_geometry(
        data, fmt_scores, fmt_fires, row_geom, lam=1.0)
    print(f"[format] changed_to_gold={fmt_geom['changed_to_gold'].sum():,}  "
          f"changed_away={fmt_geom['changed_away'].sum():,}")

    # ── 6b. Expert aggregate metrics ────────────────────────────────────
    print("\n[step 6] loading expert summaries...")
    experts = load_expert_summaries(args)
    print(f"[experts] loaded: {list(experts.keys())}")

    # ── 11. Near-miss buckets ────────────────────────────────────────────
    print("\n[step 7] classifying near-miss buckets...")
    buckets = classify_near_miss_buckets(data, row_geom, fmt_geom)
    for bn in ["clean_near_miss", "crowded_near_miss", "evidence_supported_near_miss",
               "evidence_conflicted_near_miss", "far_wrong", "not_candidate_solved"]:
        print(f"  {bn:40s}: {int((buckets==bn).sum()):6,}")

    # ── 12. Slices ───────────────────────────────────────────────────────
    print("\n[step 8] building slices...")
    slices = build_slices(data, row_geom, fmt_geom, buckets)

    # ── Build + write CSVs ───────────────────────────────────────────────
    print("\n[step 9] writing output files...")
    row_rows, cand_rows, slice_rows = build_output_rows(
        data, row_geom, cand_geom, fmt_geom, buckets, slices, args.topm_eval)
    _wcsv(os.path.join(args.output_dir, "row_geometry_metrics.csv"), row_rows)
    _wcsv(os.path.join(args.output_dir, "candidate_geometry_metrics.csv"), cand_rows)
    _wcsv(os.path.join(args.output_dir, "geometry_slice_summary.csv"), slice_rows)

    # Near-miss bucket summary CSV
    bkt_rows = []
    for bn in ["clean_near_miss", "crowded_near_miss", "evidence_supported_near_miss",
               "evidence_conflicted_near_miss", "far_wrong", "not_candidate_solved"]:
        cnt = int((buckets == bn).sum())
        bkt_rows.append({"bucket": bn, "count": cnt,
                          "pct_of_gip": 100*cnt/max(gip_n,1),
                          "pct_of_all": 100*cnt/max(N,1)})
    _wcsv(os.path.join(args.output_dir, "near_miss_bucket_summary.csv"), bkt_rows)

    # Crowding summary
    crowd_rows = []
    for sname, mask in slices.items():
        n = int(mask.sum())
        if n == 0:
            continue
        crowd_rows.append({
            "slice": sname, "n": n,
            "mean_max_rp": _nanmean(row_geom["max_rel_proj"][mask]),
            "mean_mean_rp": _nanmean(row_geom["mean_rel_proj"][mask]),
            "frac_above_0p5": float((row_geom["n_above_0p5"][mask] > 0).mean()),
            "frac_above_0p8": float((row_geom["n_above_0p8"][mask] > 0).mean()),
            "frac_above_1p0": float((row_geom["n_above_1p0"][mask] > 0).mean()),
            "oracle_wins_rate": float(row_geom["oracle_wins"][mask].mean()),
            "oracle_wrong_steal_rate": float(row_geom["oracle_wrong_steals"][mask].mean()),
        })
    _wcsv(os.path.join(args.output_dir, "correction_crowding_summary.csv"), crowd_rows)

    # Oracle simulation summary
    oracle_rows = []
    for sname, mask in [("all", np.ones(N, bool)), ("gold_in_pool", row_geom["gip"]),
                         ("bucketA", ~(data["topk"][:, 0] == data["gold"].astype(np.int64)) & row_geom["gip"])]:
        n = int(mask.sum())
        oracle_rows.append({
            "slice": sname, "n": n,
            "oracle_wins_rate": float(row_geom["oracle_wins"][mask].mean()),
            "oracle_wrong_steals_rate": float(row_geom["oracle_wrong_steals"][mask].mean()),
            "mean_oracle_margin_after": _nanmean(row_geom["oracle_margin"][mask]),
            "mean_req_dn": _nanmean(row_geom["req_dn"][mask]),
        })
    _wcsv(os.path.join(args.output_dir, "target_direction_simulation.csv"), oracle_rows)

    # Expert score geometry CSV (format only for now)
    exp_geom_rows = []
    if fmt_geom is not None:
        for sname, mask in slices.items():
            n = int(mask.sum())
            if n == 0:
                continue
            m_ftg = fmt_geom
            exp_geom_rows.append({
                "expert": "format",
                "slice": sname, "n": n,
                "mean_support_gold_minus_base": _nanmean(m_ftg["support_gold_minus_base"][mask]),
                "mean_max_wrong_support": _nanmean(m_ftg["max_wrong_support"][mask]),
                "mean_exp_margin": _nanmean(m_ftg["exp_margin"][mask]),
                "mean_exp_entropy": _nanmean(m_ftg["exp_entropy"][mask]),
                "changed_to_gold": int(m_ftg["changed_to_gold"][mask].sum()),
                "changed_away": int(m_ftg["changed_away"][mask].sum()),
            })
    _wcsv(os.path.join(args.output_dir, "expert_score_geometry.csv"), exp_geom_rows)

    # Expert method comparison CSV
    exp_cmp_rows = []
    for ename, e in sorted(experts.items()):
        exp_cmp_rows.append({
            "method": ename,
            "acc_gain": _nanf(e.get("acc_gain")),
            "changed_to_gold": _nanf(e.get("changed_to_gold")),
            "changed_away": _nanf(e.get("changed_away")),
            "net_correction": _nanf(e.get("net_correction")),
            "bdr": _nanf(e.get("bdr")),
            "real_vs_shuffled": _nanf(e.get("real_vs_shuffled", float("nan"))),
            "per_row_available": bool(e.get("per_row", False)),
            "source": str(e.get("source", "?")),
        })
    _wcsv(os.path.join(args.output_dir, "expert_method_comparison.csv"), exp_cmp_rows)
    _wcsv(os.path.join(args.output_dir, "expert_agreement_summary.csv"),
          [{"note": "multi-expert row-level agreement unavailable: "
                    "per-row scores not found in expert output dirs"}])

    # ── Examples ─────────────────────────────────────────────────────────
    print("\n[step 10] collecting examples...")
    gold_np  = data["gold"].astype(np.int64)
    topk_np  = data["topk"].astype(np.int64)
    lgt_np   = data["lgt"].astype(np.float32)
    ar       = np.arange(N)
    base_ok  = (topk_np[ar, lgt_np.argmax(axis=1)] == gold_np)
    write_exs = collect_examples(
        data, row_geom, cand_geom, fmt_geom, buckets, slices,
        args.topm_eval, args.example_count, args.seed)
    write_exs(os.path.join(args.output_dir, "examples_clean_near_miss.md"),
              buckets == "clean_near_miss", "Clean Near-miss Examples")
    write_exs(os.path.join(args.output_dir, "examples_crowded_near_miss.md"),
              buckets == "crowded_near_miss", "Crowded Near-miss Examples")
    write_exs(os.path.join(args.output_dir, "examples_evidence_supported_near_miss.md"),
              buckets == "evidence_supported_near_miss", "Evidence-Supported Near-miss")
    write_exs(os.path.join(args.output_dir, "examples_evidence_conflicted_near_miss.md"),
              buckets == "evidence_conflicted_near_miss", "Evidence-Conflicted Near-miss")
    write_exs(os.path.join(args.output_dir, "examples_harmful_expert_edits.md"),
              fmt_geom["changed_away"] if fmt_geom else np.zeros(N, bool),
              "Harmful Expert Edits (format changed_away)")
    write_exs(os.path.join(args.output_dir, "examples_target_direction_wrong_steals.md"),
              row_geom["oracle_wrong_steals"], "Oracle Target Direction — Wrong Steals")

    # ── Expert failure explanations ───────────────────────────────────────
    fail_path = os.path.join(args.output_dir, "expert_failure_explanations.md")
    with open(fail_path, "w", encoding="utf-8") as f:
        f.write("# Expert Failure Explanations\n\n")
        f.write("> Based on geometry diagnostics and available aggregate metrics.\n\n")

        f.write("## Format heuristic\n")
        if fmt_geom:
            ctg  = int(fmt_geom["changed_to_gold"].sum())
            caw  = int(fmt_geom["changed_away"].sum())
            f.write(f"- changed_to_gold={ctg:,}  changed_away={caw:,}  BDR={ctg/max(caw,1):.2f}\n")
            f.write(f"- mean_support_gold_minus_base (gip)="
                    f"{_nanmean(fmt_geom['support_gold_minus_base'][row_geom['gip']]):.4f}\n")
            f.write(f"- mean_max_wrong_support="
                    f"{_nanmean(fmt_geom['max_wrong_support'][row_geom['gip']]):.4f}\n")
            status = ("WORKS" if ctg > caw * 1.5 else ("MARGINAL" if ctg > caw else "HURTS"))
            f.write(f"- Assessment: **{status}**\n\n")

        for ename, interp in [
            ("pointer", "Sharp candidate-specific evidence from pointer context matching. "
             "Low entropy; correct candidate receives exclusive boosting."),
            ("fuzzy_phrase", "Neighborhood-level recall aligns correction with same-region "
             "candidates; reduces crowding risk on supported rows."),
            ("passage_recall", "Passage-level similarity gives diffuse vote over multiple candidates "
             "in the retrieved passage; insufficient to discriminate gold from neighboring tokens."),
            ("detail_memory", "Cross-attention over doc_past fails to outperform shuffled control, "
             "suggesting doc_past provides little causal signal for next-token prediction. "
             "The learned delta may be fitting noise rather than genuine memory patterns."),
            ("path_oracle", "Path oracle may have high changed_away rate on non-path rows; "
             "evidence scope too broad for isolated expert use."),
        ]:
            e = experts.get(ename, {})
            f.write(f"## {ename}\n")
            if e:
                f.write(f"- acc_gain={_nanf(e.get('acc_gain')):.4f}  "
                        f"changed_away={_nanf(e.get('changed_away')):.4f}  "
                        f"BDR={_nanf(e.get('bdr')):.2f}\n")
            else:
                f.write("- Source file NOT AVAILABLE — no aggregate metrics loaded.\n")
            f.write(f"- Interpretation: {interp}\n\n")
    print(f"[save] {fail_path}")

    # ── Report ────────────────────────────────────────────────────────────
    print("\n[step 11] writing main report...")
    rpt_path = write_report(
        args, data, row_geom, fmt_geom, experts, slices, buckets, audit, uinfo,
        args.output_dir, validity_audit)

    # ── config.json ───────────────────────────────────────────────────────
    cfg = {**vars(args), **uinfo,
           "N_val": N, "gip_n": gip_n,
           "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"[save] {args.output_dir}/config.json")

    print("\n" + "=" * 64)
    print(f" Geometry + Evidence Interaction Diagnostic V2 complete")
    print(f" elapsed: {time.time()-t0:.1f}s")
    print(f" report:  {rpt_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
