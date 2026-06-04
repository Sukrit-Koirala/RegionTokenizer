#!/usr/bin/env python3
"""
audit_v2_region_map_formation.py — Phase 1B.1

V2 Region Map Formation Audit.

Evaluation/audit only. No training. No weight updates.
No semantic labels. No manual categories.

Analyzes:
  1. Size/cost comparison (old vs new)
  2. Old→new token fragmentation
  3. New region purity relative to old map
  4. Representative token reports
  5. Router miss analysis (sibling vs cross-parent)
  6. New region confusion matrix
  7. Merge simulation / upper bound
  8. Old vs new recall-per-cost curve
  9. Formation verdict

Usage:
  python scripts/audit_v2_region_map_formation.py \\
    --map_name V2A_K256 \\
    --old_map  runs/region_maps_128/token_to_region.json \\
    --new_map  runs/cheap_ai/phase1B_compute_aware_region_maps/V2A_K256/token_to_region.json \\
    --new_map_dir runs/cheap_ai/phase1B_compute_aware_region_maps/V2A_K256 \\
    --router_ckpt runs/cheap_ai/phase1B_compute_aware_region_maps/router_runs/V2A_K256/best_real_region_router.pt \\
    --router_run_dir runs/cheap_ai/phase1B_compute_aware_region_maps/router_runs/V2A_K256 \\
    --train_dir runs/.../train --val_dir runs/.../val \\
    --output_dir runs/cheap_ai/phase1B_compute_aware_region_maps/formation_audit/V2A_K256 \\
    --tokenizer_name gpt2 --context_len 128 \\
    --d_model 256 --n_layers 2 --n_heads 4 --batch_size 512 --seed 42
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
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=RuntimeWarning)

_EPS = 1e-9
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from train_static_region_router import (
    StaticRegionRouter,
    LastTokenMLPBaseline,
    MeanEmbeddingMLPBaseline,
)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, dec=5):
    if isinstance(v, float): return f"{v:.{dec}f}" if v == v else "nan"
    return str(v)

def _wcsv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[save] {path}")

def _write(path, text):
    with open(path, "w", encoding="utf-8") as f: f.write(text)
    print(f"[save] {path}")

def _safediv(a, b, default=float("nan")):
    return a / b if (b and b > 0) else default

def _entropy(counts: np.ndarray) -> float:
    s = counts.sum()
    if s == 0: return 0.0
    p = counts[counts > 0] / s
    return float(-(p * np.log(p + _EPS)).sum())

def _gini(sizes: np.ndarray) -> float:
    n = len(sizes); s = sizes.sum()
    if s == 0 or n == 0: return 0.0
    ss = np.sort(sizes)
    return float((2 * np.arange(1, n+1) @ ss - (n+1) * s) / (n * s + _EPS))


# ══════════════════════════════════════════════════════════════════════════════
# Map loading
# ══════════════════════════════════════════════════════════════════════════════

def load_map_arr(path: str, vocab_size: int = 65536) -> Tuple[np.ndarray, int]:
    """Returns (tok_arr, n_regions). tok_arr[t] = n_regions means unknown."""
    with open(path) as f: raw = json.load(f)
    if isinstance(raw, list):
        t2r = {i: v for i, v in enumerate(raw) if v is not None}
    else:
        t2r = {int(k): v for k, v in raw.items()}
    if not t2r: return np.zeros(vocab_size, np.int32), 1
    n_regions = int(max(t2r.values())) + 1
    V = min(max(t2r.keys()) + 2, vocab_size)
    arr = np.full(vocab_size, n_regions, dtype=np.int32)
    for t, r in t2r.items():
        if int(t) < vocab_size: arr[int(t)] = int(r)
    return arr, n_regions


# ══════════════════════════════════════════════════════════════════════════════
# Shard loading
# ══════════════════════════════════════════════════════════════════════════════

_GOLD_AL = ["gold_token", "gold", "labels"]
_TOPK_AL = ["base_topk_ids", "base_topk", "topk_ids"]
_IDS_AL  = ["input_ids"]
_RID_AL  = ["row_id", "row_ids"]
_TOFF_AL = ["token_offset"]

def _get_key(sh, aliases, required=True):
    for a in aliases:
        if a in sh: return sh[a]
    if required: raise KeyError(f"Need one of {aliases}")
    return None

def load_val_data(val_dir: str, max_rows: Optional[int]) -> dict:
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths: raise FileNotFoundError(f"No shards in {val_dir}")
    bufs = defaultdict(list)
    total = 0
    for sp in paths:
        if max_rows and total >= max_rows: break
        sh   = torch.load(sp, map_location="cpu", weights_only=False)
        ids  = _get_key(sh, _IDS_AL).long()
        gold = _get_key(sh, _GOLD_AL).long()
        topk = _get_key(sh, _TOPK_AL, required=False)
        rids = _get_key(sh, _RID_AL, required=False)
        toff = _get_key(sh, _TOFF_AL, required=False)
        B, T = ids.shape
        if rids is None: rids = torch.arange(total, total+B)
        if toff is None: toff = torch.zeros(B, dtype=torch.long)
        if max_rows and total+B > max_rows:
            keep = max_rows-total
            ids, gold, rids, toff = ids[:keep], gold[:keep], rids[:keep], toff[:keep]
            if topk is not None: topk = topk[:keep]
            B = keep
        bufs["input_ids"].append(ids.numpy().astype(np.int32))
        bufs["gold"].append(gold.numpy().astype(np.int32))
        bufs["row_ids"].append(rids.numpy().astype(np.int32))
        bufs["token_offset"].append(toff.numpy().astype(np.int32))
        if topk is not None:
            bufs["base_topk_ids"].append(topk.numpy().astype(np.int32))
        total += B
    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    data["has_base_topk"] = "base_topk_ids" in data
    data["seq_len"] = int(data["input_ids"].shape[1])
    data["n_rows"]  = total
    print(f"[val] {total:,} rows  T={data['seq_len']}  has_topk={data['has_base_topk']}")
    return data

def load_train_gold_freq(train_dir: str, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
    paths = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    gold_freq = np.zeros(vocab_size, dtype=np.int64)
    topk_freq = np.zeros(vocab_size, dtype=np.int64)
    for sp in paths:
        sh   = torch.load(sp, map_location="cpu", weights_only=False)
        gold = _get_key(sh, _GOLD_AL).numpy().astype(np.int64)
        topk = _get_key(sh, _TOPK_AL, required=False)
        valid_g = gold[(gold >= 0) & (gold < vocab_size)]
        np.add.at(gold_freq, valid_g, 1)
        if topk is not None:
            valid_t = topk.numpy().ravel()
            valid_t = valid_t[(valid_t >= 0) & (valid_t < vocab_size)].astype(np.int64)
            np.add.at(topk_freq, valid_t, 1)
    print(f"[train] gold_freq sum={gold_freq.sum():,}  topk_freq sum={topk_freq.sum():,}")
    return gold_freq, topk_freq


# ══════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        print(f"[tokenizer] loaded: {name}")
        return tok
    except Exception as e:
        print(f"[tokenizer] WARN: {e}")
        return None

def decode_tok(tid: int, tokenizer) -> str:
    if tokenizer is None: return f"<{tid}>"
    try: return repr(tokenizer.decode([tid]))
    except Exception: return f"<{tid}>"


# ══════════════════════════════════════════════════════════════════════════════
# Map analysis helpers
# ══════════════════════════════════════════════════════════════════════════════

def size_diagnostics(tok_arr: np.ndarray, n_regions: int, label: str) -> dict:
    sizes = np.bincount(tok_arr[tok_arr < n_regions], minlength=n_regions).astype(np.float64)
    mn = sizes.mean()
    return {
        "map": label, "n_regions": n_regions,
        "vocab_assigned": int((tok_arr < n_regions).sum()),
        "min_size":  int(sizes.min()),   "max_size":   int(sizes.max()),
        "mean_size": round(mn, 1),       "median_size": round(float(np.median(sizes)), 1),
        "p90_size":  round(float(np.percentile(sizes, 90)), 1),
        "p95_size":  round(float(np.percentile(sizes, 95)), 1),
        "p99_size":  round(float(np.percentile(sizes, 99)), 1),
        "size_gini": round(_gini(sizes), 4),
        "num_over_2x_mean": int((sizes > 2*mn).sum()),
        "num_over_4x_mean": int((sizes > 4*mn).sum()),
    }


def build_frag_matrix(old_arr: np.ndarray, new_arr: np.ndarray,
                       n_old: int, n_new: int, vocab_size: int) -> np.ndarray:
    """frag[old_r, new_r] = number of vocab tokens assigned to both."""
    frag = np.zeros((n_old, n_new), dtype=np.int32)
    V = min(vocab_size, len(old_arr), len(new_arr))
    for t in range(V):
        ov = int(old_arr[t]); nv = int(new_arr[t])
        if ov < n_old and nv < n_new:
            frag[ov, nv] += 1
    return frag


def build_frag_matrix_fast(old_arr: np.ndarray, new_arr: np.ndarray,
                             n_old: int, n_new: int, vocab_size: int) -> np.ndarray:
    """Vectorised fragmentation matrix."""
    V = min(vocab_size, len(old_arr), len(new_arr))
    ov = old_arr[:V].astype(np.int64)
    nv = new_arr[:V].astype(np.int64)
    valid = (ov < n_old) & (nv < n_new)
    idx = ov[valid] * n_new + nv[valid]
    frag_flat = np.bincount(idx, minlength=n_old * n_new).astype(np.int32)
    return frag_flat.reshape(n_old, n_new)


def fragmentation_stats(frag: np.ndarray, old_r: int,
                         gold_freq: np.ndarray, topk_freq: np.ndarray,
                         old_arr: np.ndarray, n_old: int) -> dict:
    row   = frag[old_r].astype(np.float64)
    total = row.sum()
    if total == 0:
        return {"old_region": old_r, "old_size": 0, "num_children": 0,
                "split_entropy": 0.0, "effective_children": 1.0,
                "largest_child_frac": 0.0, "second_child_frac": 0.0,
                "is_old_huge": False, "is_old_frequent": False}
    top2 = np.sort(row)[::-1][:2]
    old_tokens = np.where(old_arr == old_r)[0]
    freq_sum = int(gold_freq[old_tokens].sum()) if len(old_tokens) > 0 else 0
    mean_sz = frag.sum(axis=1).mean()
    return {
        "old_region":           old_r,
        "old_size":             int(total),
        "num_children":         int((row > 0).sum()),
        "split_entropy":        round(_entropy(row), 4),
        "effective_children":   round(float(np.exp(_entropy(row))), 3),
        "largest_child_frac":   round(float(top2[0] / total), 4),
        "second_child_frac":    round(float(top2[1] / total) if len(top2) > 1 else 0.0, 4),
        "train_gold_freq":      freq_sum,
        "is_old_huge":          bool(total > 4 * mean_sz),
        "is_old_frequent":      bool(freq_sum > np.percentile([gold_freq[np.where(old_arr == r)[0]].sum() for r in range(frag.shape[0]) if (old_arr == r).any()], 75) if frag.shape[0] > 0 else False),
    }


def purity_stats(frag: np.ndarray, new_r: int,
                  old_arr: np.ndarray, new_arr: np.ndarray,
                  n_old: int, vocab_size: int) -> dict:
    col  = frag[:, new_r].astype(np.float64)
    total= col.sum()
    if total == 0:
        return {"new_region": new_r, "new_size": 0, "dominant_old": -1,
                "dominant_old_frac": 0.0, "old_parent_entropy": 0.0,
                "effective_old_parents": 1.0}
    dom  = int(col.argmax())
    H    = _entropy(col)
    return {
        "new_region":            new_r,
        "new_size":              int(total),
        "dominant_old":          dom,
        "dominant_old_frac":     round(float(col[dom] / total), 4),
        "old_parent_entropy":    round(H, 4),
        "effective_old_parents": round(float(np.exp(H)), 3),
        "top3_old_regions":      str(list(np.argsort(-col)[:3].tolist())),
        "top3_old_fracs":        str([round(float(col[i]/total), 3) for i in np.argsort(-col)[:3]]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Router loading and inference
# ══════════════════════════════════════════════════════════════════════════════

def load_router_model(ckpt_path: str, run_dir: Optional[str], args,
                       vocab_size: int, n_regions: int, n_super: int,
                       device: torch.device):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[router] no checkpoint at {ckpt_path} — skipping router parts")
        return None, {}
    saved   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = saved.get("variant", "real_region_router")
    sd      = saved.get("state_dict", saved)
    cfg = {}
    if run_dir and os.path.isfile(os.path.join(run_dir, "config.json")):
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
    d_model    = cfg.get("d_model",     args.d_model)
    n_layers   = cfg.get("n_layers",    args.n_layers)
    n_heads    = cfg.get("n_heads",     args.n_heads)
    max_seq_len= cfg.get("max_seq_len", args.context_len)
    d_ff       = 4 * d_model
    if "last_token" in variant:
        model = LastTokenMLPBaseline(vocab_size, n_regions, d_model, 0.0)
    elif "mean_emb" in variant or "mean_embedding" in variant:
        model = MeanEmbeddingMLPBaseline(vocab_size, n_regions, d_model, 0.0)
    else:
        model = StaticRegionRouter(
            vocab_size=vocab_size, n_regions=n_regions, n_super=n_super,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=0.0, max_seq_len=max_seq_len,
            use_super_aux=False)
    model.load_state_dict(sd, strict=False)
    model.to(device); model.eval()
    print(f"[router] loaded variant={variant}  params={sum(p.numel() for p in model.parameters()):,}")
    return model, cfg


def run_router_inference(model, data: dict, context_len: int,
                          n_regions: int, device: torch.device,
                          batch_size: int) -> np.ndarray:
    """Returns region_logits [N, n_regions]."""
    N   = data["n_rows"]
    ids = data["input_ids"]
    avail_T = ids.shape[1]
    ctx_len = min(context_len, avail_T)
    ids_ctx = ids[:, -ctx_len:]
    logits  = np.zeros((N, n_regions), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e    = min(s + batch_size, N)
            ids_t= torch.from_numpy(ids_ctx[s:e].astype(np.int64)).to(device)
            rl, _= model(ids_t)
            logits[s:e] = rl.cpu().float().numpy()
    if np.isnan(logits).any():
        raise RuntimeError("NaN in router logits!")
    return logits


# ══════════════════════════════════════════════════════════════════════════════
# Region representatives
# ══════════════════════════════════════════════════════════════════════════════

def region_reps(r: int, tok_arr: np.ndarray, n_regions: int,
                gold_freq: np.ndarray, topk_freq: np.ndarray,
                tokenizer, top_n: int = 30) -> dict:
    tids = np.where(tok_arr == r)[0]
    if len(tids) == 0:
        return {"region": r, "size": 0, "top_ids": [], "top_strs": [], "top_scores": []}
    scores = np.log1p(gold_freq[tids].astype(np.float64)) + \
             0.5 * np.log1p(topk_freq[tids].astype(np.float64))
    top    = np.argsort(-scores)[:top_n]
    top_ids= tids[top].tolist()
    top_sc = scores[top].tolist()
    top_str= [decode_tok(t, tokenizer) for t in top_ids]
    return {"region": r, "size": int(len(tids)),
            "top_ids": top_ids, "top_strs": top_str,
            "top_scores": [round(s, 3) for s in top_sc]}


def format_region_block(r: int, reps_d: dict, purity_d: dict, frag_row: dict) -> str:
    lines = [
        f"### Region {r}  (size={reps_d['size']})",
        f"  dominant_old={purity_d.get('dominant_old','?')}  "
        f"dom_frac={purity_d.get('dominant_old_frac','?')}  "
        f"eff_parents={purity_d.get('effective_old_parents','?')}",
        f"  top3_old={purity_d.get('top3_old_regions','?')}  "
        f"top3_fracs={purity_d.get('top3_old_fracs','?')}",
        f"  rep_tokens: {reps_d['top_strs'][:12]}",
        "",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Merge simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_merge_policies(region_logits: np.ndarray,
                              new_gold_region: np.ndarray,
                              new_arr: np.ndarray,
                              old_arr: np.ndarray,
                              frag: np.ndarray,
                              n_new: int,
                              n_old: int,
                              new_region_sz: np.ndarray,
                              vocab_size: int,
                              known_new_mask: np.ndarray) -> List[dict]:
    """Simulate merge policies using predicted region probabilities."""
    N, R = region_logits.shape
    ranked  = np.argsort(-region_logits, axis=1)
    km      = known_new_mask

    # Precompute: dominant old parent for each new region
    new_to_dom_old = np.argmax(frag, axis=0)  # [n_new]: dominant old region for each new r

    # Precompute: old_r → set of new children
    old_to_new_children: List[np.ndarray] = []
    for ov in range(n_old):
        children = np.where(frag[ov] > 0)[0].astype(np.int32)
        old_to_new_children.append(children)

    def _eval(effective_new_per_row: List[set], label: str, fallback_rate: float = float("nan")) -> dict:
        covered   = np.zeros(N, dtype=bool)
        cand_sizes= np.zeros(N, dtype=np.float64)
        for i in range(N):
            if not km[i]: continue
            chosen = effective_new_per_row[i]
            chosen = {r for r in chosen if r < n_new}
            if new_gold_region[i] in chosen:
                covered[i] = True
            cand_sizes[i] = sum(int(new_region_sz[r]) for r in chosen if r < len(new_region_sz))
        N_km   = int(km.sum())
        recall = float(covered[km].mean()) if N_km > 0 else float("nan")
        avg_cs = float(cand_sizes[km].mean()) if N_km > 0 else float("nan")
        med_cs = float(np.median(cand_sizes[km])) if N_km > 0 else float("nan")
        p90_cs = float(np.percentile(cand_sizes[km], 90)) if N_km > 0 else float("nan")
        n_known_vocab = int((new_arr < n_new).sum()) if len(new_arr) > 0 else vocab_size
        frac   = _safediv(avg_cs, n_known_vocab)
        avg_nr = float(np.mean([len(s) for s in effective_new_per_row]))
        return {
            "policy": label,
            "fallback_rate": round(fallback_rate, 5),
            "effective_recall": round(recall, 5),
            "avg_candidate_size": round(avg_cs, 1),
            "median_candidate_size": round(med_cs, 1),
            "p90_candidate_size": round(p90_cs, 1),
            "candidate_fraction": round(frac, 5),
            "avg_num_regions_selected": round(avg_nr, 1),
        }

    rows = []

    # Policy 1: raw top-K for k in [4,8,16]
    for k in [4, 8, 16]:
        k_ = min(k, R)
        regions_per_row = [set(ranked[i, :k_].tolist()) for i in range(N)]
        rows.append(_eval(regions_per_row, f"new_top{k}_raw"))

    # Policy 2: old_parent_expand_top8
    k8_regions = [set(ranked[i, :min(8, R)].tolist()) for i in range(N)]
    expanded = []
    for i in range(N):
        top8 = k8_regions[i]
        dom_olds = {int(new_to_dom_old[r]) for r in top8 if r < n_new}
        exp = set(top8)
        for ov in dom_olds:
            if ov < len(old_to_new_children):
                exp |= set(int(x) for x in old_to_new_children[ov])
        expanded.append(exp)
    rows.append(_eval(expanded, "old_parent_expand_top8"))

    # Policy 3: sibling_expand_limited (top-2 siblings per old parent)
    k8_limited = []
    for i in range(N):
        top8 = k8_regions[i]
        dom_olds = {int(new_to_dom_old[r]) for r in top8 if r < n_new}
        exp = set(top8)
        for ov in dom_olds:
            if ov >= len(old_to_new_children): continue
            children = old_to_new_children[ov]
            # add top-2 largest siblings not already in top8
            sibs = [c for c in children if c not in top8]
            sibs_sorted = sorted(sibs, key=lambda c: int(new_region_sz[c]) if c < len(new_region_sz) else 0, reverse=True)
            exp |= set(sibs_sorted[:2])
        k8_limited.append(exp)
    rows.append(_eval(k8_limited, "sibling_expand_top8_limited"))

    # Policy 4: oracle old grouping (diagnostic only — gold old parent determines group)
    oracle = []
    for i in range(N):
        g_tok = int(0)  # placeholder; we derive from new_gold_region
        g_new = int(new_gold_region[i])
        if g_new < n_new:
            gold_old = int(new_to_dom_old[g_new])
            oracle_set = set(int(x) for x in old_to_new_children[gold_old]) if gold_old < len(old_to_new_children) else set()
        else:
            oracle_set = set()
        oracle.append(oracle_set)
    rows.append(_eval(oracle, "old_oracle_grouping_DIAGNOSTIC_ONLY"))

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Report writing
# ══════════════════════════════════════════════════════════════════════════════

def write_formation_report(args,
                            old_diag: dict, new_diag: dict,
                            frag_stats: List[dict],
                            pur_stats: List[dict],
                            miss_decomp: dict,
                            merge_rows: List[dict],
                            cov_old: dict, cov_new: dict,
                            out_dir: str) -> str:

    def _v(d, k): return d.get(k, float("nan")) if d else float("nan")

    # Purity aggregate stats
    dom_fracs = [r["dominant_old_frac"] for r in pur_stats if r["new_size"] > 0]
    pct_high_purity = _safediv(sum(1 for f in dom_fracs if f > 0.8), len(dom_fracs))
    pct_low_purity  = _safediv(sum(1 for f in dom_fracs if f < 0.5), len(dom_fracs))
    avg_dom_frac    = float(np.mean(dom_fracs)) if dom_fracs else float("nan")

    # Fragmentation aggregate
    n_children = [r["num_children"] for r in frag_stats if r["old_size"] > 0]
    avg_children = float(np.mean(n_children)) if n_children else float("nan")

    # Old region 52 stats
    old52 = next((r for r in frag_stats if r["old_region"] == 52), {})

    # Merge simulation best (non-oracle)
    best_merge = max(
        (r for r in merge_rows if "DIAGNOSTIC" not in r["policy"]),
        key=lambda r: r.get("effective_recall", 0), default={})

    # Recommendation logic
    sibling_pct = miss_decomp.get("pct_sibling_miss", 0.0)
    cross_pct   = miss_decomp.get("pct_hard_cross_old_miss", 0.0)
    r8_new      = cov_new.get("recall@8", 0.0)
    r8_old      = cov_old.get("recall@8", 0.82189)
    frac8_new   = cov_new.get("cand_frac@8", 1.0)

    if r8_new >= r8_old and frac8_new < 0.10:
        rec = "KEEP_V2A_AND_TRAIN_MORE"
    elif pct_low_purity > 0.5:
        rec = "DO_NOT_USE_V2A"
    elif sibling_pct > 0.40:
        rec = "BUILD_V3_HIERARCHICAL_OLD_PARENT_SPLIT"
    elif sibling_pct > 0.25:
        rec = "BUILD_V3_SELECTIVE_SPLIT"
    elif r8_new < 0.60:
        rec = "TRY_V2B_BPE_AWARE"
    elif r8_new < r8_old * 0.90:
        rec = "TRY_V2C_HEAD_TAIL"
    else:
        rec = "TRY_V2B_BPE_AWARE"

    lines = [
        f"# Phase 1B.1 V2 Region Map Formation Audit — {args.map_name}",
        "",
        f"**old_map:** {args.old_map}  |  **new_map:** {args.new_map}",
        f"**router_ckpt:** {args.router_ckpt or 'not loaded'}",
        "", "---", "",
        "## Size Comparison", "",
        "| metric | old_K128 | new_map |",
        "|---|---|---|",
    ]
    for k in ["n_regions", "vocab_assigned", "min_size", "max_size", "p95_size", "size_gini",
              "num_over_2x_mean", "num_over_4x_mean"]:
        lines.append(f"| {k} | {_v(old_diag,k)} | {_v(new_diag,k)} |")

    lines += ["", "---", "", "## Coverage Curve Comparison", "",
              "| k | old_recall@k | old_frac@k | new_recall@k | new_frac@k | "
              "old_score | new_score |",
              "|---|---|---|---|---|---|---|"]
    for k in [4, 8, 16, 32, 64]:
        or_ = _fmt(cov_old.get(f"recall@{k}", float("nan")))
        of_ = _fmt(cov_old.get(f"cand_frac@{k}", float("nan")))
        nr_ = _fmt(cov_new.get(f"recall@{k}", float("nan")))
        nf_ = _fmt(cov_new.get(f"cand_frac@{k}", float("nan")))
        try: os_ = _fmt(_safediv(float(or_), float(of_))) if float(of_) > 0 else "nan"
        except Exception: os_ = "nan"
        try: ns_ = _fmt(_safediv(float(nr_), float(nf_))) if float(nf_) > 0 else "nan"
        except Exception: ns_ = "nan"
        lines.append(f"| {k} | {or_} | {of_} | {nr_} | {nf_} | {os_} | {ns_} |")

    lines += ["", "---", "", "## Miss Decomposition (top8 misses)", ""]
    for k, v in miss_decomp.items():
        lines.append(f"- **{k}:** {_fmt(v)}")

    lines += ["", "---", "", "## Merge Simulation", "",
              "| policy | recall | cand_frac | avg_regions |",
              "|---|---|---|---|"]
    for r in merge_rows:
        lines.append(f"| {r['policy']} | {r['effective_recall']} "
                     f"| {r['candidate_fraction']} | {r['avg_num_regions_selected']} |")

    lines += ["", "---", "", "## Q&A", ""]
    def _q(n, q, ans, detail=""):
        lines.append(f"\n### Q{n}: {q}")
        lines.append(f"**{ans}**")
        if detail: lines.append(f"\n{detail}")

    _q(1, "Did V2A_K256 successfully remove huge regions?",
       f"old_max={_v(old_diag,'max_size')}  new_max={_v(new_diag,'max_size')}",
       f"old over_2x={_v(old_diag,'num_over_2x_mean')}  new over_2x={_v(new_diag,'num_over_2x_mean')}")
    _q(2, f"What happened to old region 52 (old_size={old52.get('old_size','?')})?",
       f"Split into {old52.get('num_children','?')} new regions  "
       f"eff_children={old52.get('effective_children','?')}  "
       f"largest_child_frac={old52.get('largest_child_frac','?')}",
       f"entropy={old52.get('split_entropy','?')}")
    _q(3, "Did V2 mostly split old regions or globally remix them?",
       f"avg_dom_frac={_fmt(avg_dom_frac)}  pct_high_purity(>0.8)={_fmt(pct_high_purity)}  "
       f"pct_low_purity(<0.5)={_fmt(pct_low_purity)}",
       "LOW purity → global remix. HIGH purity → clean splits.")
    _q(4, "Are new regions coherent by representative tokens?",
       "See new_region_representatives.md for token examples per region")
    _q(5, "Are new router misses mostly sibling misses?",
       f"pct_sibling_miss={_fmt(sibling_pct)}  pct_hard_cross={_fmt(cross_pct)}",
       "sibling > 0.3 → regions too finely split; hard cross → structurally wrong map")
    _q(6, "Are high-frequency regions still dominating top-k slots?",
       "See new_region_representatives.md and top_new_region_confusions.md")
    _q(7, "Does merge simulation recover recall?",
       f"best_policy={best_merge.get('policy','?')}  recall={best_merge.get('effective_recall','?')}  "
       f"frac={best_merge.get('candidate_fraction','?')}")
    _q(8, "Should V3 use global balanced clustering again?",
       f"pct_low_purity={_fmt(pct_low_purity)} → "
       + ("NO — global remix hurts structure" if pct_low_purity > 0.4 else "MAYBE — purity acceptable"))
    _q(9, "Should V3 use selective recursive splitting of old regions?",
       f"sibling_miss_pct={_fmt(sibling_pct)} → "
       + ("YES — sibling misses dominate" if sibling_pct > 0.25 else "NOT PRIMARY CAUSE"))
    _q(10, "Is V2A_K256 worth continuing with?",
       f"recall@8={_fmt(r8_new)} vs old={r8_old}  frac@8={_fmt(frac8_new)}")

    lines += ["", "---", "",
              f"## FINAL RECOMMENDATION: {rec}", "",
              f"*Based on purity={_fmt(avg_dom_frac)}, sibling_miss={_fmt(sibling_pct)}, "
              f"recall@8={_fmt(r8_new)}*", ""]

    _write(os.path.join(out_dir, "formation_audit_report.md"), "\n".join(lines))
    return rec


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 1B.1: V2 Region Map Formation Audit")
    p.add_argument("--map_name",      default="V2A_K256")
    p.add_argument("--old_map",       required=True)
    p.add_argument("--new_map",       required=True)
    p.add_argument("--new_map_dir",   default=None)
    p.add_argument("--router_ckpt",   default=None)
    p.add_argument("--router_run_dir", default=None)
    p.add_argument("--train_dir",     required=True)
    p.add_argument("--val_dir",       required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--tokenizer_name", default="gpt2")
    p.add_argument("--no_tokenizer",  action="store_true")
    p.add_argument("--context_len",   type=int, default=128)
    p.add_argument("--d_model",       type=int, default=256)
    p.add_argument("--n_layers",      type=int, default=2)
    p.add_argument("--n_heads",       type=int, default=4)
    p.add_argument("--batch_size",    type=int, default=512)
    p.add_argument("--max_val_rows",  type=int, default=None)
    p.add_argument("--vocab_size",    type=int, default=50257)
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    t0 = time.time()

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = None if args.no_tokenizer else load_tokenizer(args.tokenizer_name)
    V = args.vocab_size

    # ── Load maps ─────────────────────────────────────────────────────────────
    print("\n[step 1] Loading maps...")
    old_arr, n_old = load_map_arr(args.old_map, V)
    new_arr, n_new = load_map_arr(args.new_map, V)
    print(f"  old: n_regions={n_old}  assigned={int((old_arr<n_old).sum()):,}")
    print(f"  new: n_regions={n_new}  assigned={int((new_arr<n_new).sum()):,}")

    old_diag = size_diagnostics(old_arr, n_old, "old_K128")
    new_diag = size_diagnostics(new_arr, n_new, args.map_name)

    old_region_sz = np.bincount(old_arr[old_arr < n_old], minlength=n_old).astype(np.int64)
    new_region_sz = np.bincount(new_arr[new_arr < n_new], minlength=n_new).astype(np.int64)

    # ── Train frequencies ─────────────────────────────────────────────────────
    print("\n[step 2] Loading train gold frequencies...")
    gold_freq, topk_freq = load_train_gold_freq(args.train_dir, V)

    # ── Fragmentation matrix ──────────────────────────────────────────────────
    print("\n[step 3] Building fragmentation matrix...")
    frag = build_frag_matrix_fast(old_arr, new_arr, n_old, n_new, V)
    print(f"  frag matrix: {frag.shape}  nnz={int((frag>0).sum()):,}")

    # ── Part 1: Size comparison ───────────────────────────────────────────────
    print("\n[PART 1] Size comparison...")
    _wcsv(os.path.join(args.output_dir, "size_comparison.csv"), [old_diag, new_diag])

    def _largest_regions_md(tok_arr, n_reg, sz_arr, label, gold_freq, topk_freq, top_n=20):
        lines = [f"# {label} — Largest Regions\n"]
        top20 = np.argsort(-sz_arr)[:top_n]
        for r in top20:
            tids = np.where(tok_arr == r)[0]
            rep  = region_reps(r, tok_arr, n_reg, gold_freq, topk_freq, tokenizer, top_n=10)
            lines.append(f"## Region {r}  size={sz_arr[r]}")
            lines.append(f"  rep_tokens: {rep['top_strs'][:10]}")
            lines.append("")
        return "\n".join(lines)

    _write(os.path.join(args.output_dir, "largest_regions_old.md"),
           _largest_regions_md(old_arr, n_old, old_region_sz, "Old K128 Map",
                                gold_freq, topk_freq))
    _write(os.path.join(args.output_dir, "largest_regions_new.md"),
           _largest_regions_md(new_arr, n_new, new_region_sz, f"New {args.map_name}",
                                gold_freq, topk_freq))

    # ── Part 2: Fragmentation ─────────────────────────────────────────────────
    print("\n[PART 2] Old→new fragmentation...")
    frag_stat_rows = [fragmentation_stats(frag, r, gold_freq, topk_freq, old_arr, n_old)
                      for r in range(n_old)]
    _wcsv(os.path.join(args.output_dir, "old_to_new_fragmentation.csv"), frag_stat_rows)

    problem_regs = [52, 127, 125, 121, 126, 124, 90, 80]

    def _problem_split_md():
        lines = [f"# Problem Old Region Splits — {args.map_name}\n"]
        for old_r in problem_regs:
            row = frag[old_r]
            total = row.sum()
            lines.append(f"## Old Region {old_r}  (old_size={total})")
            top_children = np.argsort(-row)[:10]
            for nc in top_children:
                if row[nc] == 0: continue
                child_rep = region_reps(nc, new_arr, n_new, gold_freq, topk_freq, tokenizer, 8)
                frac = float(row[nc]) / (total + _EPS)
                lines.append(f"  → new_region={nc}  n={row[nc]}  frac={frac:.3f}  "
                              f"reps={child_rep['top_strs'][:6]}")
            sr = frag_stat_rows[old_r] if old_r < len(frag_stat_rows) else {}
            lines.append(f"  num_children={sr.get('num_children','?')}  "
                         f"entropy={sr.get('split_entropy','?')}  "
                         f"eff_children={sr.get('effective_children','?')}  "
                         f"largest_frac={sr.get('largest_child_frac','?')}")
            lines.append("")
        return "\n".join(lines)

    _write(os.path.join(args.output_dir, "old_problem_region_splits.md"),
           _problem_split_md())

    # Most/least fragmented
    sorted_by_children = sorted(frag_stat_rows, key=lambda r: -r["num_children"])
    _write(os.path.join(args.output_dir, "most_fragmented_old_regions.md"),
           "# Most Fragmented Old Regions\n\n" +
           "\n".join(f"old={r['old_region']}  old_size={r['old_size']}  "
                     f"children={r['num_children']}  eff={r['effective_children']}  "
                     f"ent={r['split_entropy']}"
                     for r in sorted_by_children[:20]))
    huge_rows = [r for r in frag_stat_rows if r["is_old_huge"]]
    _write(os.path.join(args.output_dir, "huge_old_region_splits.md"),
           "# Huge Old Region Splits\n\n" +
           "\n".join(f"old={r['old_region']}  size={r['old_size']}  "
                     f"children={r['num_children']}  eff={r['effective_children']}  "
                     f"largest_frac={r['largest_child_frac']}"
                     for r in huge_rows))

    # ── Part 3: New region purity ─────────────────────────────────────────────
    print("\n[PART 3] New region purity analysis...")
    pur_stat_rows = [purity_stats(frag, r, old_arr, new_arr, n_old, V)
                     for r in range(n_new)]
    _wcsv(os.path.join(args.output_dir, "new_region_old_parent_purity.csv"), pur_stat_rows)

    high_pur = sorted([r for r in pur_stat_rows if r["dominant_old_frac"] >= 0.8 and r["new_size"]>0],
                      key=lambda r: -r["dominant_old_frac"])
    low_pur  = sorted([r for r in pur_stat_rows if r["dominant_old_frac"] <  0.5 and r["new_size"]>0],
                      key=lambda r: r["dominant_old_frac"])

    def _pur_md(rows, title):
        lines = [f"# {title}\n"]
        for r in rows[:30]:
            lines.append(f"new={r['new_region']}  size={r['new_size']}  "
                         f"dom_old={r['dominant_old']}  dom_frac={r['dominant_old_frac']}  "
                         f"eff_parents={r['effective_old_parents']}  "
                         f"top3_old={r['top3_old_regions']}")
        return "\n".join(lines)

    _write(os.path.join(args.output_dir, "new_regions_high_old_purity.md"),
           _pur_md(high_pur, "New Regions with High Old Purity (dom_frac >= 0.8)"))
    _write(os.path.join(args.output_dir, "new_regions_low_old_purity.md"),
           _pur_md(low_pur, "New Regions with Low Old Purity (dom_frac < 0.5)"))

    large_mixed = sorted([r for r in pur_stat_rows if r["dominant_old_frac"] < 0.6
                           and r["new_size"] > int(new_region_sz.mean() * 0.5)],
                          key=lambda r: -r["new_size"])
    _write(os.path.join(args.output_dir, "new_regions_large_mixed.md"),
           _pur_md(large_mixed, "Large Mixed New Regions (dom_frac < 0.6)"))

    # ── Part 4: Representatives ───────────────────────────────────────────────
    print("\n[PART 4] Building region representatives...")
    all_reps = [region_reps(r, new_arr, n_new, gold_freq, topk_freq, tokenizer, 30)
                for r in range(n_new)]
    with open(os.path.join(args.output_dir, "new_region_representatives.json"),
              "w", encoding="utf-8") as f:
        json.dump([{**rep, "purity": pur_stat_rows[rep["region"]]["dominant_old_frac"]}
                   for rep in all_reps], f, indent=2)

    rep_md_lines = [f"# New Region Representatives — {args.map_name}\n"]
    for r, rep in enumerate(all_reps):
        pur = pur_stat_rows[r] if r < len(pur_stat_rows) else {}
        rep_md_lines.append(format_region_block(r, rep, pur, frag_stat_rows[pur.get("dominant_old", 0)] if pur else {}))
    _write(os.path.join(args.output_dir, "new_region_representatives.md"),
           "\n".join(rep_md_lines))

    # Per-old-problem files
    for old_r in problem_regs:
        children = sorted(np.where(frag[old_r] > 0)[0].tolist(), key=lambda c: -frag[old_r, c])
        lines = [f"# Children of Old Region {old_r} in {args.map_name}\n",
                 f"old_size={frag[old_r].sum()}  num_children={len(children)}\n"]
        for nc in children[:15]:
            rep = all_reps[nc] if nc < len(all_reps) else {"top_strs": [], "size": 0}
            pur = pur_stat_rows[nc] if nc < len(pur_stat_rows) else {}
            n_tok_from_old = int(frag[old_r, nc])
            lines.append(f"## new_region={nc}  n_from_old={n_tok_from_old}  "
                         f"frac={n_tok_from_old/(frag[old_r].sum()+_EPS):.3f}  "
                         f"total_size={rep['size']}")
            lines.append(f"  dom_frac_from_old52={pur.get('dominant_old_frac','?')}")
            lines.append(f"  rep_tokens={rep['top_strs'][:12]}")
            lines.append("")
        fname = f"representatives_children_of_old_{old_r}.md"
        _write(os.path.join(args.output_dir, fname), "\n".join(lines))

    # ── Part 5: Router miss analysis ──────────────────────────────────────────
    cov_new: dict = {}
    miss_decomp: dict = {}
    pred_rows: List[dict] = []
    region_logits_all: Optional[np.ndarray] = None

    print("\n[PART 5] Loading val data and running router inference...")
    val_data = load_val_data(args.val_dir, args.max_val_rows)
    N = val_data["n_rows"]
    gold_np   = val_data["gold"].astype(np.int64)
    gold_clipped = np.clip(gold_np, 0, V - 1)
    old_gold_r = old_arr[gold_clipped].astype(np.int64)
    new_gold_r = new_arr[gold_clipped].astype(np.int64)
    old_known  = old_gold_r < n_old
    new_known  = new_gold_r < n_new

    # Precompute new-to-dominant-old
    new_to_dom_old = np.argmax(frag, axis=0).astype(np.int32)  # [n_new]

    model, router_cfg = load_router_model(
        args.router_ckpt, args.router_run_dir, args,
        V, n_new, 1, device)

    if model is not None:
        print("[PART 5] Running inference...")
        region_logits_all = run_router_inference(
            model, val_data, args.context_len, n_new, device, args.batch_size)
        probs = torch.from_numpy(region_logits_all).float()
        probs_sm = F.softmax(probs, dim=-1).numpy()

        ranked = np.argsort(-region_logits_all, axis=1)   # [N, n_new]
        R = region_logits_all.shape[1]

        top1p = probs_sm[np.arange(N), ranked[:, 0]]
        top2p = probs_sm[np.arange(N), ranked[:, 1]] if R >= 2 else np.zeros(N)
        margin = top1p - top2p
        entropy= -(probs_sm * np.log(probs_sm + _EPS)).sum(axis=1)

        # Coverage metrics
        for k in [1, 2, 4, 8, 16, 32, 64]:
            k_ = min(k, R)
            top_r = ranked[:, :k_]  # [N, k_]
            hits = np.array([
                (new_known[i] and int(new_gold_r[i]) in top_r[i].tolist())
                for i in range(N)
            ])
            nm = new_known
            recall = float(hits[nm].mean()) if nm.any() else float("nan")
            cand_sz = np.array([new_region_sz[top_r[i]].sum() for i in range(N)], dtype=np.float64)
            n_known_v = int((new_arr < n_new).sum())
            frac   = _safediv(float(cand_sz[nm].mean()) if nm.any() else 0.0, n_known_v)
            cov_new[f"recall@{k}"]   = round(recall, 5)
            cov_new[f"cand_frac@{k}"]= round(frac, 5)
            cov_new[f"avg_cand@{k}"] = round(float(cand_sz[nm].mean()) if nm.any() else float("nan"), 1)

        # Gold rank
        gold_rank = np.full(N, -1, dtype=np.int64)
        for i in np.where(new_known)[0]:
            gr = int(new_gold_r[i])
            if gr < R:
                pos = np.where(ranked[i] == gr)[0]
                if len(pos) > 0: gold_rank[i] = int(pos[0])

        # Miss decomposition (top8 misses)
        k8 = min(8, R)
        top8_per_row = ranked[:, :k8]    # [N, k8]

        miss_mask = new_known & (gold_rank >= 8) & (gold_rank >= 0)
        n_miss    = int(miss_mask.sum())
        n_known_nm= int(new_known.sum())

        sibling_miss = 0; same_old_top8 = 0; hard_cross = 0
        rank_9_16 = 0; rank_17_32 = 0; rank_33p = 0

        miss_diag_rows = []
        for i in np.where(miss_mask)[0]:
            gn = int(new_gold_r[i])
            gold_dom_old = int(new_to_dom_old[gn]) if gn < n_new else -1
            top8_srs = [int(new_to_dom_old[r]) if r < n_new else -1 for r in top8_per_row[i]]
            same_old_in_top8 = gold_dom_old in top8_srs
            is_sibling = same_old_in_top8 and gn not in top8_per_row[i].tolist()
            is_hard    = not same_old_in_top8
            if same_old_in_top8: same_old_top8 += 1
            if is_sibling:       sibling_miss  += 1
            if is_hard:          hard_cross     += 1
            gr = int(gold_rank[i])
            if 8 <= gr <= 15:    rank_9_16  += 1
            elif 16 <= gr <= 31: rank_17_32 += 1
            elif gr >= 32:       rank_33p   += 1
            miss_diag_rows.append({
                "row_idx": int(i), "new_gold_region": int(gn),
                "gold_rank": gr, "old_gold_region": int(old_gold_r[i]),
                "gold_dom_old": gold_dom_old,
                "same_old_in_top8": same_old_in_top8,
                "is_sibling_miss": is_sibling,
                "is_hard_cross": is_hard,
                "top8_dom_olds": str(top8_srs),
                "top1_prob": round(float(top1p[i]), 5),
                "margin": round(float(margin[i]), 5),
                "entropy": round(float(entropy[i]), 4),
            })

        _wcsv(os.path.join(args.output_dir, "v2_top8_miss_diagnostics.csv"),
              miss_diag_rows[:5000])

        miss_decomp = {
            "total_top8_misses":         n_miss,
            "pct_sibling_miss":          round(_safediv(sibling_miss, n_miss), 5),
            "pct_same_old_parent_top8":  round(_safediv(same_old_top8, n_miss), 5),
            "pct_hard_cross_old_miss":   round(_safediv(hard_cross, n_miss), 5),
            "pct_gold_rank_9_16":        round(_safediv(rank_9_16, n_miss), 5),
            "pct_gold_rank_17_32":       round(_safediv(rank_17_32, n_miss), 5),
            "pct_gold_rank_33_plus":     round(_safediv(rank_33p, n_miss), 5),
        }
        _wcsv(os.path.join(args.output_dir, "v2_miss_decomposition.csv"), [miss_decomp])
        miss_md_lines = ["# V2 Top8 Miss Decomposition\n"]
        for k, v in miss_decomp.items():
            miss_md_lines.append(f"- **{k}:** {_fmt(v)}")
        _write(os.path.join(args.output_dir, "v2_miss_decomposition.md"),
               "\n".join(miss_md_lines))

        # Save val predictions
        pred_save_rows = []
        for i in range(min(N, 20000)):  # save sample
            if not new_known[i]: continue
            pred_save_rows.append({
                "row_idx": i, "row_id": int(val_data["row_ids"][i]),
                "gold_token": int(gold_np[i]),
                "old_gold_region": int(old_gold_r[i]),
                "new_gold_region": int(new_gold_r[i]),
                "new_gold_rank": int(gold_rank[i]),
                "top1_prob": round(float(top1p[i]), 5),
                "margin": round(float(margin[i]), 5),
                "entropy": round(float(entropy[i]), 4),
                "top4_new_regions": str(ranked[i, :4].tolist()),
            })
        _wcsv(os.path.join(args.output_dir, "v2_router_val_predictions.csv"),
              pred_save_rows)

    else:
        print("[PART 5] No router checkpoint — skipping inference parts")
        miss_decomp = {}

    # ── Part 6: Confusion matrix ──────────────────────────────────────────────
    print("\n[PART 6] Building confusion matrix...")
    if region_logits_all is not None:
        top1_pred = ranked[:, 0]
        conf: Dict[Tuple[int,int], int] = defaultdict(int)
        for i in np.where(new_known)[0]:
            conf[(int(new_gold_r[i]), int(top1_pred[i]))] += 1
        conf_rows = sorted(
            [{"gold_new": k[0], "pred_new": k[1], "count": v,
              "gold_dom_old": int(new_to_dom_old[k[0]]) if k[0]<n_new else -1,
              "pred_dom_old": int(new_to_dom_old[k[1]]) if k[1]<n_new else -1,
              "same_dom_old": bool(new_to_dom_old[k[0]] == new_to_dom_old[k[1]]) if k[0]<n_new and k[1]<n_new else False,
              "gold_reps": str(all_reps[k[0]]["top_strs"][:4]) if k[0]<len(all_reps) else "[]",
              "pred_reps": str(all_reps[k[1]]["top_strs"][:4]) if k[1]<len(all_reps) else "[]"}
             for k, v in conf.items() if k[0] != k[1]],
            key=lambda r: -r["count"])
        _wcsv(os.path.join(args.output_dir, "new_region_confusion_pairs.csv"),
              conf_rows[:500])
        conf_md = ["# Top New Region Confusions\n",
                   "| gold_new | pred_new | count | same_dom_old | gold_dom | pred_dom |",
                   "|---|---|---|---|---|---|"]
        for r in conf_rows[:50]:
            conf_md.append(f"| {r['gold_new']} | {r['pred_new']} | {r['count']} "
                           f"| {r['same_dom_old']} | {r['gold_dom_old']} | {r['pred_dom_old']} |")
        _write(os.path.join(args.output_dir, "top_new_region_confusions.md"),
               "\n".join(conf_md))

        sib_confs = [r for r in conf_rows if r["same_dom_old"]]
        sib_md = ["# Sibling Confusions (same dominant old parent)\n",
                  "| gold_new | pred_new | count | dom_old | gold_reps | pred_reps |",
                  "|---|---|---|---|---|---|"]
        for r in sib_confs[:30]:
            sib_md.append(f"| {r['gold_new']} | {r['pred_new']} | {r['count']} "
                          f"| {r['gold_dom_old']} | {r['gold_reps'][:40]} | {r['pred_reps'][:40]} |")
        _write(os.path.join(args.output_dir, "sibling_confusions.md"),
               "\n".join(sib_md))

    # ── Part 7: Merge simulation ──────────────────────────────────────────────
    print("\n[PART 7] Merge simulation...")
    merge_rows: List[dict] = []
    if region_logits_all is not None:
        merge_rows = simulate_merge_policies(
            region_logits_all, new_gold_r, new_arr, old_arr,
            frag, n_new, n_old, new_region_sz, V, new_known)
        _wcsv(os.path.join(args.output_dir, "merge_simulation_policies.csv"), merge_rows)
        merge_md = ["# Merge Simulation Policies\n",
                    "| policy | recall | cand_frac | avg_cand_size | avg_regions |",
                    "|---|---|---|---|---|"]
        for r in merge_rows:
            merge_md.append(f"| {r['policy']} | {r['effective_recall']} "
                            f"| {r['candidate_fraction']} | {r['avg_candidate_size']} "
                            f"| {r['avg_num_regions_selected']} |")
        _write(os.path.join(args.output_dir, "merge_simulation.md"),
               "\n".join(merge_md))

    # ── Part 8: Old vs new curve ──────────────────────────────────────────────
    print("\n[PART 8] Coverage curve comparison...")
    # Old map known numbers from Phase 1A
    cov_old = {"recall@1": 0.25, "cand_frac@1": 0.0185,
               "recall@4": 0.603, "cand_frac@4": 0.0895,
               "recall@8": 0.82189, "cand_frac@8": 0.1775,
               "recall@16": 0.89512, "cand_frac@16": 0.3504,
               "recall@32": 0.95624, "cand_frac@32": 0.5559,
               "recall@64": 0.99005, "cand_frac@64": 0.7861}

    curve_rows = []
    for k in [1, 2, 4, 8, 16, 32, 64]:
        nr = cov_new.get(f"recall@{k}", float("nan"))
        nf = cov_new.get(f"cand_frac@{k}", float("nan"))
        or_ = cov_old.get(f"recall@{k}", float("nan"))
        of_ = cov_old.get(f"cand_frac@{k}", float("nan"))
        try: ns = round(_safediv(float(nr), float(nf)), 3) if float(nf) > 0 else float("nan")
        except Exception: ns = float("nan")
        try: os_ = round(_safediv(float(or_), float(of_)), 3) if float(of_) > 0 else float("nan")
        except Exception: os_ = float("nan")
        curve_rows.append({
            "k": k,
            "old_recall": or_, "old_cand_frac": of_, "old_score": os_,
            "new_recall": nr,  "new_cand_frac": nf,  "new_score": ns,
            "delta_recall": round(float(nr-or_), 5) if (nr==nr and or_==or_) else float("nan"),
            "delta_frac":   round(float(nf-of_), 5) if (nf==nf and of_==of_) else float("nan"),
        })
    _wcsv(os.path.join(args.output_dir, "old_vs_new_curve.csv"), curve_rows)

    curve_md = ["# Old vs New Coverage Curve\n",
                "| k | old_recall | old_frac | old_score | new_recall | new_frac | new_score | "
                "delta_recall | delta_frac |",
                "|---|---|---|---|---|---|---|---|---|"]
    for r in curve_rows:
        curve_md.append(
            f"| {r['k']} | {_fmt(r['old_recall'])} | {_fmt(r['old_cand_frac'])} "
            f"| {_fmt(r['old_score'])} | {_fmt(r['new_recall'])} | {_fmt(r['new_cand_frac'])} "
            f"| {_fmt(r['new_score'])} | {_fmt(r['delta_recall'])} | {_fmt(r['delta_frac'])} |")
    _write(os.path.join(args.output_dir, "old_vs_new_curve.md"),
           "\n".join(curve_md))

    # ── Part 9: Formation report ──────────────────────────────────────────────
    print("\n[PART 9] Writing formation audit report...")
    recommendation = write_formation_report(
        args, old_diag, new_diag, frag_stat_rows, pur_stat_rows,
        miss_decomp, merge_rows, cov_old, cov_new, args.output_dir)

    # ── Console summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    dom_fracs_arr = np.array([r["dominant_old_frac"] for r in pur_stat_rows if r["new_size"]>0])
    avg_dom = float(dom_fracs_arr.mean()) if len(dom_fracs_arr) > 0 else float("nan")
    pct_hi  = float((dom_fracs_arr > 0.8).mean()) if len(dom_fracs_arr) > 0 else float("nan")
    pct_lo  = float((dom_fracs_arr < 0.5).mean()) if len(dom_fracs_arr) > 0 else float("nan")
    n_ch    = [r["num_children"] for r in frag_stat_rows if r["old_size"]>0]
    avg_ch  = float(np.mean(n_ch)) if n_ch else float("nan")
    old52   = next((r for r in frag_stat_rows if r["old_region"]==52), {})

    best_merge_row = max((r for r in merge_rows if "DIAGNOSTIC" not in r["policy"]),
                         key=lambda r: r.get("effective_recall", 0), default={})

    print(f"\n{'='*65}")
    print(f" PHASE 1B.1 V2 REGION FORMATION AUDIT  ({elapsed/60:.1f} min)")
    print(f"{'='*65}")
    print(f"  Map:         {args.map_name}")
    print(f"  old_regions: {n_old}   new_regions: {n_new}")
    print()
    print(f"  Size:")
    print(f"    old max_size:  {old_diag.get('max_size','?')}")
    print(f"    new max_size:  {new_diag.get('max_size','?')}")
    print(f"    old p95:       {old_diag.get('p95_size','?')}")
    print(f"    new p95:       {new_diag.get('p95_size','?')}")
    print()
    print(f"  Fragmentation:")
    print(f"    avg old->new children: {_fmt(avg_ch)}")
    print(f"    old region 52 children: {old52.get('num_children','?')}")
    print(f"    old region 52 largest child frac: {old52.get('largest_child_frac','?')}")
    print()
    print(f"  New purity (relative to old map):")
    print(f"    avg dominant old fraction: {_fmt(avg_dom)}")
    print(f"    pct with dom_frac > 0.8:   {_fmt(pct_hi)}")
    print(f"    pct with dom_frac < 0.5:   {_fmt(pct_lo)}")
    print()
    print(f"  Coverage curve (new map):")
    for k in [8, 16, 32]:
        print(f"    recall@{k}={_fmt(cov_new.get(f'recall@{k}',float('nan')))}  "
              f"frac@{k}={_fmt(cov_new.get(f'cand_frac@{k}',float('nan')))}")
    print()
    if miss_decomp:
        print(f"  Top8 miss decomposition:")
        print(f"    sibling miss:        {_fmt(miss_decomp.get('pct_sibling_miss',float('nan')))}")
        print(f"    hard cross-old miss: {_fmt(miss_decomp.get('pct_hard_cross_old_miss',float('nan')))}")
        print(f"    rank 9-16:           {_fmt(miss_decomp.get('pct_gold_rank_9_16',float('nan')))}")
        print(f"    rank 17-32:          {_fmt(miss_decomp.get('pct_gold_rank_17_32',float('nan')))}")
        print(f"    rank 33+:            {_fmt(miss_decomp.get('pct_gold_rank_33_plus',float('nan')))}")
    print()
    if best_merge_row:
        print(f"  Merge simulation best (non-oracle):")
        print(f"    policy:           {best_merge_row.get('policy','?')}")
        print(f"    recall:           {best_merge_row.get('effective_recall','?')}")
        print(f"    candidate frac:   {best_merge_row.get('candidate_fraction','?')}")
    print()
    print(f"  Recommendation: {recommendation}")
    print(f"{'='*65}")
    print(f"\n  Outputs: {args.output_dir}/")


if __name__ == "__main__":
    main()
