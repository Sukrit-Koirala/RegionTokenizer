#!/usr/bin/env python3
"""
analyze_retrieval_failure_examples.py

Diagnostic bucket analysis of retrieval failure cases.
Iterates val shards, recomputes retrieval for the requested key,
and writes human-readable markdown reports per failure category.

Reuses retrieval logic from run_retrieval_evidence_sweep.py.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.run_retrieval_evidence_sweep import (
    build_dense_index,
    dense_retrieve,
    build_lexical_index,
    get_shard_repr,
    make_layer_info,
    _LEXICAL_KEYS,
    _HYBRID_LEXICAL,
)

# ══════════════════════════════════════════════════════════════════════════════
# Bucket definitions
# ══════════════════════════════════════════════════════════════════════════════

_BUCKET_META = {
    "A_base_miss_retrieval_miss": (
        "Base miss + retrieval miss",
        "Gold not in base top-K AND not in retrieval neighbors. "
        "Unrecoverable by this pipeline — rare token? factual recall? "
        "tokenization artifact?",
    ),
    "B_base_miss_retrieval_hit": (
        "Base miss + retrieval hit",
        "Gold not in base top-K but IS found in retrieval neighbors. "
        "Cases the resolver could rescue — what characterizes them?",
    ),
    "C_reinforces_wrong": (
        "Base wrong + retrieval reinforces wrong",
        "Base top-1 incorrect, gold in base top-K, "
        "but retrieval support for base top-1 > support for gold. "
        "Retrieval doubling down on the base error.",
    ),
    "D_supports_gold": (
        "Base wrong + retrieval supports gold",
        "Base top-1 incorrect, gold in base top-K, "
        "retrieval support for gold > support for base top-1. "
        "Ideal resolver training cases.",
    ),
    "E_ambiguous": (
        "Ambiguous / many valid",
        "Base top-1 wrong, gold in base top-K, high neighbor entropy, "
        "low support for any single token. Genuinely multi-valid contexts.",
    ),
    "F_high_entropy": (
        "Retrieval high entropy",
        "Neighbor next-tokens scatter across many different values. "
        "Context similarity is not a strong discriminative signal here.",
    ),
    "G_low_entropy_wrong": (
        "Retrieval confident but wrong",
        "Neighbor entropy low — retrieval strongly agrees — but on the "
        "wrong token. Systematic retrieval error or domain mismatch?",
    ),
    "H_low_entropy_gold": (
        "Retrieval confident and correct",
        "Neighbor entropy low and neighbors agree on the gold token. "
        "Best resolver training candidates.",
    ),
    "I_region_right_token_wrong": (
        "Region correct, token wrong",
        "Gold and base top-1 share the same region, but exact token differs. "
        "Region-level signal succeeds; fine-grained selection fails.",
    ),
}

_SEP = "─" * 72


# ══════════════════════════════════════════════════════════════════════════════
# Small utilities
# ══════════════════════════════════════════════════════════════════════════════

def _decode(ids, tokenizer):
    if tokenizer is None:
        if isinstance(ids, (int, np.integer)):
            return f"[{int(ids)}]"
        return " ".join(f"[{int(i)}]" for i in ids)
    try:
        if isinstance(ids, (int, np.integer)):
            ids = [int(ids)]
        return tokenizer.decode([int(i) for i in ids])
    except Exception:
        return str(ids)


def _compute_support(nbr_golds, nbr_scores, gold, base_top1, tau):
    """
    Softmax-weighted support for gold vs base_top1 token.
    Returns (sup_gold, sup_top1, sup_margin, entropy, nbr_top1_tok, unique_ct).
    """
    K = len(nbr_golds)
    valid = nbr_golds >= 0
    if not valid.any():
        return 0.0, 0.0, 0.0, 0.0, -1, 0

    scores = nbr_scores.astype(np.float64)
    scores[~valid] = -1e9
    scores -= scores[valid].max()
    exp_s = np.exp(scores / max(tau, 1e-6))
    exp_s[~valid] = 0.0
    denom = exp_s.sum()
    weights = exp_s / max(denom, 1e-12)

    sup_gold  = float(weights[nbr_golds == gold].sum())
    sup_top1  = float(weights[nbr_golds == base_top1].sum())

    valid_golds = [int(g) for g in nbr_golds if g >= 0]
    cnt   = Counter(valid_golds)
    total = len(valid_golds)
    entropy = float(-sum(
        (c / total) * math.log2(max(c / total, 1e-10))
        for c in cnt.values()
    )) if total > 0 else 0.0

    unique_ct = len(cnt)
    nbr_top1  = int(cnt.most_common(1)[0][0]) if cnt else -1
    return sup_gold, sup_top1, sup_gold - sup_top1, entropy, nbr_top1, unique_ct


def _gold_rank(gold, base_topk):
    matches = np.where(np.asarray(base_topk) == int(gold))[0]
    return int(matches[0]) if len(matches) > 0 else None


def _approx_probs(logits, n=10):
    if logits is None:
        return [None] * n
    lgt = np.asarray(logits[:n], dtype=np.float64)
    lgt -= lgt.max()
    e = np.exp(lgt)
    return (e / e.sum()).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Train row lookup + shard cache
# ══════════════════════════════════════════════════════════════════════════════

def build_train_row_lookup(train_dir, max_rows=None):
    """Build {row_id: (shard_path, local_idx)} from train shards."""
    shards = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    lookup = {}
    total  = 0
    t0     = time.time()
    for si, sp in enumerate(shards):
        if max_rows and total >= max_rows:
            break
        s = torch.load(sp, map_location="cpu", weights_only=True)
        N = s["gold_token"].shape[0]
        if max_rows:
            N = min(N, max_rows - total)
        rids = (s["row_id"][:N].numpy().astype(np.int64) if "row_id" in s
                else np.arange(total, total + N, dtype=np.int64))
        for j in range(N):
            lookup[int(rids[j])] = (sp, int(j))
        total += N
        if si % 100 == 0:
            print(f"  [train_lookup] shard={si}/{len(shards)-1}"
                  f"  rows={total:,}  t={time.time()-t0:.0f}s")
    print(f"  [train_lookup] done  {total:,} rows indexed")
    return lookup


class _ShardCache:
    def __init__(self, max_size=20):
        self._cache = OrderedDict()
        self._max   = max_size

    def load(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        s = torch.load(path, map_location="cpu", weights_only=True)
        self._cache[path] = s
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return s


def fetch_neighbor_contexts(nbr_row_ids, train_lookup, shard_cache, n_tail=64):
    """Return list of {input_ids, gold, row_id} dicts (or None if not found)."""
    out = []
    for rid in nbr_row_ids:
        loc = train_lookup.get(int(rid))
        if loc is None:
            out.append(None)
            continue
        sp, li = loc
        s    = shard_cache.load(sp)
        ids  = s["input_ids"][li].numpy()
        gold = int(s["gold_token"][li])
        out.append({"input_ids": ids[-n_tail:], "gold": gold, "row_id": int(rid)})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Region helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_region_map(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}
    except Exception as e:
        print(f"  [WARN] region map load failed: {e}")
        return None


def _same_region(tok_a, tok_b, region_map):
    if region_map is None or tok_a < 0 or tok_b < 0:
        return None
    ra = region_map.get(int(tok_a))
    rb = region_map.get(int(tok_b))
    if ra is None or rb is None:
        return None
    return ra == rb


# ══════════════════════════════════════════════════════════════════════════════
# Lexical retrieval with row_id return
# ══════════════════════════════════════════════════════════════════════════════

def _lexical_retrieve_with_row_ids(id_arrays, lex_data, last_k, K, max_cands):
    """Like lexical_retrieve_batch but also returns global row_ids of matches."""
    inverted = lex_data["inverted"]
    row_meta = lex_data["row_meta"]

    # Build reverse map meta_idx -> row_id once, cached in lex_data dict
    if "_meta_to_rid" not in lex_data:
        m2r = {}
        for rid, mi in lex_data["row_id_to_meta"].items():
            m2r[mi] = rid
        lex_data["_meta_to_rid"] = m2r
    meta_to_rid = lex_data["_meta_to_rid"]

    B       = len(id_arrays)
    golds   = np.full((B, K), -1, dtype=np.int32)
    scores  = np.zeros((B, K), dtype=np.float32)
    row_ids = np.full((B, K), -1, dtype=np.int64)

    for bi, id_row in enumerate(id_arrays):
        query_set = frozenset(int(x) for x in id_row[-last_k:])
        counts    = Counter()
        for tok in query_set:
            for mi in inverted.get(tok, []):
                counts[mi] += 1
        if not counts:
            continue
        cands = (sorted(counts, key=counts.__getitem__, reverse=True)[:max_cands]
                 if len(counts) > max_cands else list(counts))
        scored = []
        for mi in cands:
            cand_set = row_meta[mi]["set32"] if last_k >= 32 else row_meta[mi]["set16"]
            union    = len(query_set | cand_set)
            scored.append((counts[mi] / max(union, 1), mi))
        scored.sort(reverse=True)
        for rank, (sc, mi) in enumerate(scored[:K]):
            golds[bi, rank]   = row_meta[mi]["gold"]
            scores[bi, rank]  = float(sc)
            row_ids[bi, rank] = meta_to_rid.get(mi, -1)

    return golds, scores, row_ids


# ══════════════════════════════════════════════════════════════════════════════
# Bucket assignment
# ══════════════════════════════════════════════════════════════════════════════

def assign_buckets(row, e_p90, e_p25, e_median, region_map):
    gold      = row["gold_token"]
    base_top1 = row["base_top1"]
    gold_in_b = row["gold_in_base"]
    gold_in_n = row["gold_in_nbr"]
    sup_gold  = row["sup_gold"]
    sup_top1  = row["sup_top1"]
    entropy   = row["nbr_entropy"]
    nbr_top1  = row["nbr_top1_token"]
    correct   = base_top1 == gold
    unique_ct = row["unique_nbr_ct"]

    buckets = []

    if not gold_in_b and not gold_in_n:
        buckets.append("A_base_miss_retrieval_miss")

    if not gold_in_b and gold_in_n:
        buckets.append("B_base_miss_retrieval_hit")

    if not correct and gold_in_b and sup_top1 > sup_gold:
        buckets.append("C_reinforces_wrong")

    if not correct and gold_in_b and sup_gold > sup_top1:
        buckets.append("D_supports_gold")

    if (not correct and gold_in_b
            and entropy > e_median
            and sup_gold < 0.15 and sup_top1 < 0.15):
        buckets.append("E_ambiguous")

    if entropy > e_p90:
        buckets.append("F_high_entropy")

    if (entropy < e_p25 and nbr_top1 != gold
            and not correct and unique_ct <= 3):
        buckets.append("G_low_entropy_wrong")

    if entropy < e_p25 and (nbr_top1 == gold or sup_gold > 0.5):
        buckets.append("H_low_entropy_gold")

    if region_map is not None:
        sr = _same_region(gold, base_top1, region_map)
        if sr is True and not correct:
            buckets.append("I_region_right_token_wrong")

    return buckets


# ══════════════════════════════════════════════════════════════════════════════
# Core collection loop
# ══════════════════════════════════════════════════════════════════════════════

def _get_query_vecs(shard, key, start, end, backbone, layer_info, device):
    if key in _LEXICAL_KEYS:
        return None
    if key == _HYBRID_LEXICAL:
        # Use layer_mid as the dense component for analysis purposes
        return get_shard_repr(shard, "layer_mid", start, end,
                              backbone, layer_info, device)
    return get_shard_repr(shard, key, start, end, backbone, layer_info, device)


def _dense_retrieve_shard(shard, key, N, index, backbone, layer_info,
                          device, K, chunk_size, batch_size):
    """Run dense retrieval for all N rows of a shard. Returns (golds, scores, positions)."""
    g_list, s_list, p_list = [], [], []
    for bs in range(0, N, batch_size):
        be = min(bs + batch_size, N)
        q  = _get_query_vecs(shard, key, bs, be, backbone, layer_info, device)
        if q is None:
            raise RuntimeError(
                f"Key '{key}' returned None query vectors — "
                "layer discovery failed or key is lexical.")
        g, s, p = dense_retrieve(q, index, K, chunk_size, device)
        g_list.append(g); s_list.append(s); p_list.append(p)
    return (np.concatenate(g_list),
            np.concatenate(s_list),
            np.concatenate(p_list))


def collect_all_rows(val_dir, key, compare_key, primary_index, compare_index,
                     lex_data, backbone, layer_info, device, args):
    shards = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not shards:
        raise RuntimeError(f"No val shards in {val_dir}")

    tau   = args.support_tau
    K     = args.num_neighbors
    all_rows = []
    total    = 0
    t0       = time.time()

    for sp in shards:
        if args.max_val_rows and total >= args.max_val_rows:
            break
        s      = torch.load(sp, map_location="cpu", weights_only=True)
        N_full = s["gold_token"].shape[0]
        N      = (min(N_full, args.max_val_rows - total)
                  if args.max_val_rows else N_full)

        golds_arr = s["gold_token"][:N].numpy().astype(np.int32)
        base_topk = s["base_topk_ids"][:N].numpy().astype(np.int32)
        base_lgt  = (s["base_topk_lgt"][:N].numpy().astype(np.float32)
                     if "base_topk_lgt" in s else None)
        id_arrays = s["input_ids"][:N].numpy()
        row_ids_a = (s["row_id"][:N].numpy().astype(np.int64) if "row_id" in s
                     else np.arange(total, total + N, dtype=np.int64))

        # ── Primary key retrieval ──────────────────────────────────────────
        if key in _LEXICAL_KEYS:
            lk = 16 if "last16" in key else 32
            p_golds, p_scores, p_rids = _lexical_retrieve_with_row_ids(
                id_arrays, lex_data, lk, K, args.max_lexical_candidates_per_query)
        else:
            p_golds, p_scores, p_pos = _dense_retrieve_shard(
                s, key, N, primary_index, backbone, layer_info,
                device, K, args.retrieval_chunk_size, args.query_batch_size)
            N_idx  = primary_index["N"]
            pos_ok = np.clip(p_pos, 0, N_idx - 1)
            p_rids = np.where(p_pos >= 0,
                              primary_index["row_ids"][pos_ok],
                              np.int64(-1))

        # ── Compare key retrieval (optional) ──────────────────────────────
        c_golds = c_scores = c_rids = None
        if compare_key is not None:
            if compare_key in _LEXICAL_KEYS and lex_data is not None:
                lk = 16 if "last16" in compare_key else 32
                c_golds, c_scores, c_rids = _lexical_retrieve_with_row_ids(
                    id_arrays, lex_data, lk, K,
                    args.max_lexical_candidates_per_query)
            elif compare_index is not None:
                c_golds, c_scores, c_pos = _dense_retrieve_shard(
                    s, compare_key, N, compare_index, backbone, layer_info,
                    device, K, args.retrieval_chunk_size, args.query_batch_size)
                N_ci   = compare_index["N"]
                pos_ok = np.clip(c_pos, 0, N_ci - 1)
                c_rids = np.where(c_pos >= 0,
                                  compare_index["row_ids"][pos_ok],
                                  np.int64(-1))

        # ── Build per-row dicts ────────────────────────────────────────────
        for i in range(N):
            g  = int(golds_arr[i])
            bt = base_topk[i]
            ng = p_golds[i]
            ns = p_scores[i]

            sg, st, sm, ent, ntop, uct = _compute_support(ng, ns, g, int(bt[0]), tau)

            gold_in_base = bool((bt == g).any())
            gold_in_nbr  = bool(((ng >= 0) & (ng == g)).any())

            row = {
                "row_id":           int(row_ids_a[i]),
                "shard_path":       sp,
                "local_idx":        int(i),
                "input_ids":        id_arrays[i],
                "gold_token":       g,
                "base_topk_ids":    bt,
                "base_topk_lgt":    base_lgt[i] if base_lgt is not None else None,
                "base_top1":        int(bt[0]),
                "gold_in_base":     gold_in_base,
                "base_top1_correct": int(bt[0]) == g,
                "gold_rank":        _gold_rank(g, bt),
                # Primary
                "nbr_golds":        ng,
                "nbr_scores":       ns,
                "nbr_row_ids":      p_rids[i],
                "gold_in_nbr":      gold_in_nbr,
                "ret_added":        not gold_in_base and gold_in_nbr,
                "sup_gold":         sg,
                "sup_top1":         st,
                "sup_margin":       sm,
                "nbr_entropy":      ent,
                "nbr_top1_token":   ntop,
                "unique_nbr_ct":    uct,
                # Compare
                "cmp_nbr_golds":    c_golds[i]  if c_golds  is not None else None,
                "cmp_nbr_scores":   c_scores[i] if c_scores is not None else None,
                "cmp_nbr_row_ids":  c_rids[i]   if c_rids   is not None else None,
                "cmp_sup_gold":     None,
                "cmp_sup_top1":     None,
                "cmp_sup_margin":   None,
                "cmp_entropy":      None,
                # Filled later
                "buckets":          [],
                "_nbr_contexts":    None,
            }

            if c_golds is not None:
                csg, cst, csm, ce, _, _ = _compute_support(
                    c_golds[i], c_scores[i], g, int(bt[0]), tau)
                row["cmp_sup_gold"]   = csg
                row["cmp_sup_top1"]   = cst
                row["cmp_sup_margin"] = csm
                row["cmp_entropy"]    = ce

            all_rows.append(row)

        total += N
        print(f"  [collect] rows={total:,}  t={time.time()-t0:.0f}s")

    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# Markdown formatting
# ══════════════════════════════════════════════════════════════════════════════

def _sr_str(val):
    if val is None:
        return "N/A"
    return "yes" if val else "no"


def format_example(idx, row, tokenizer, region_map, key, compare_key, args,
                   bucket_name):
    g        = row["gold_token"]
    bt_ids   = row["base_topk_ids"]
    bt_lgt   = row["base_topk_lgt"]
    base_t1  = row["base_top1"]
    K        = args.num_neighbors
    probs    = _approx_probs(bt_lgt, min(10, len(bt_ids)))
    ctx_tail = _decode(row["input_ids"][-64:], tokenizer)
    gold_str = _decode(g, tokenizer)
    nbr_ctx  = row.get("_nbr_contexts") or []

    cmp_hdr = f" | {compare_key}" if compare_key else ""
    cmp_sep = " |---------|" if compare_key else ""

    lines = [
        f"## Example {idx}",
        "",
        f"**Bucket:** {bucket_name}",
        f"**Key:** `{key}`" + (f"   **Compare:** `{compare_key}`" if compare_key else ""),
        f"**Row ID:** {row['row_id']}",
        "",
        "**Context tail (last 64 tokens):**",
        "```text",
        ctx_tail.strip(),
        "```",
        "",
        f"**Gold token:** `{gold_str.strip()}` (id={g})",
        (f"**Gold in base top-{args.top_k}:** {row['gold_in_base']}"
         + (f"  rank={row['gold_rank']}" if row['gold_rank'] is not None
            else "  not present")),
        f"**Base top-1 correct:** {row['base_top1_correct']}",
        "",
        "**Base top-10:**",
        "",
        "| rank | token | id | logit | top_k_prob | same_region_as_gold |",
        "|------|-------|----|-------|------------|---------------------|",
    ]

    for r in range(min(10, len(bt_ids))):
        tok_id  = int(bt_ids[r])
        tok_str = _decode(tok_id, tokenizer).strip().replace("|", "\\|")
        lgt_s   = f"{float(bt_lgt[r]):.3f}" if bt_lgt is not None else "N/A"
        prob_s  = f"{probs[r]:.4f}" if probs[r] is not None else "N/A"
        sr      = _sr_str(_same_region(g, tok_id, region_map))
        lines.append(f"| {r+1} | `{tok_str}` | {tok_id} | {lgt_s} | {prob_s} | {sr} |")

    lines += [
        "",
        "**Retrieval summary:**",
        "",
        f"| metric | {key}{cmp_hdr} |",
        f"|--------|---------|{cmp_sep}",
    ]

    def _r(v):
        return f"{v:.4f}" if isinstance(v, float) else ("N/A" if v is None else str(v))

    rows_table = [
        ("gold_in_neighbors",    row["gold_in_nbr"],  None),
        ("retrieval_added_gold", row["ret_added"],     None),
        ("sup_gold",             row["sup_gold"],      row["cmp_sup_gold"]),
        ("sup_base_top1",        row["sup_top1"],      row["cmp_sup_top1"]),
        ("sup_margin",           row["sup_margin"],    row["cmp_sup_margin"]),
        ("nbr_entropy",          row["nbr_entropy"],   row["cmp_entropy"]),
        ("unique_nbr_tokens",    row["unique_nbr_ct"], None),
    ]
    for label, pval, cval in rows_table:
        if compare_key and cval is not None:
            lines.append(f"| {label} | {_r(pval)} | {_r(cval)} |")
        else:
            lines.append(f"| {label} | {_r(pval)} |")

    lines += ["", f"**Top-{min(10, K)} neighbors ({key}):**", ""]

    for nr in range(min(10, K)):
        ng  = int(row["nbr_golds"][nr])
        if ng < 0:
            break
        ns  = float(row["nbr_scores"][nr])
        nrid = int(row["nbr_row_ids"][nr]) if row["nbr_row_ids"] is not None else -1
        ntok = _decode(ng, tokenizer).strip()
        is_gold = ng == g
        is_top1 = ng == base_t1
        sr      = _sr_str(_same_region(g, ng, region_map))

        ctx_line = "_context unavailable_"
        if nr < len(nbr_ctx) and nbr_ctx[nr] is not None:
            ctx_dec  = _decode(nbr_ctx[nr]["input_ids"].tolist(), tokenizer).strip()
            ctx_line = f"`...{ctx_dec[-100:]}`"

        lines += [
            f"  **{nr+1}.** score={ns:.4f}  row_id={nrid}  "
            f"next=`{ntok}` (id={ng})  "
            f"is_gold={is_gold}  is_base_top1={is_top1}  same_region={sr}",
            f"  ctx: {ctx_line}",
            "",
        ]

    lines += [
        "",
        "**Human diagnosis:**",
        "- [ ] local syntax / grammatical completion",
        "- [ ] factual / entity recall",
        "- [ ] long-range context dependency",
        "- [ ] tokenization artifact",
        "- [ ] genuinely ambiguous",
        "- [ ] retrieval junk (wrong context type retrieved)",
        "- [ ] retrieval useful but resolver would need to act",
        "- [ ] other: ",
        "",
        _SEP,
        "",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Report writing
# ══════════════════════════════════════════════════════════════════════════════

def write_bucket_report(bucket_name, examples, tokenizer, region_map,
                        key, compare_key, args, output_dir):
    title, description = _BUCKET_META.get(bucket_name, (bucket_name, ""))

    header = [
        f"# Bucket: {bucket_name}",
        "",
        f"**{title}**",
        "",
        description,
        "",
        f"Key: `{key}`" + (f"   |   Compare: `{compare_key}`" if compare_key else ""),
        f"n_examples: {len(examples)}",
        "",
        _SEP,
        "",
    ]

    body = []
    for idx, row in enumerate(examples, 1):
        body.append(format_example(
            idx, row, tokenizer, region_map, key, compare_key, args, bucket_name))

    path = os.path.join(output_dir, f"bucket_{bucket_name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
        f.write("\n".join(body))
    print(f"  [report] {path}  ({len(examples)} examples)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = _parse()
    os.makedirs(args.output_root, exist_ok=True)

    print("=" * 70)
    print(f" Retrieval Failure Analysis")
    print(f" key={args.key}  compare_key={args.compare_key}")
    print(f" output: {args.output_root}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    layer_info = make_layer_info(backbone, d_model)
    if layer_info["blocks"] is not None:
        print(f"  layers={layer_info['n_layers']}  "
              f"early={layer_info['early_idx']}  "
              f"mid={layer_info['mid_idx']}  "
              f"late={layer_info['late_idx']}")
    else:
        print("  [WARN] No transformer blocks found; layer keys unavailable.")

    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e30)
        print("  Tokenizer: GPT-2")
    except Exception:
        tokenizer = None
        print("  [WARN] Tokenizer unavailable; raw IDs shown.")

    region_map = load_region_map(args.token_to_region)
    if region_map:
        print(f"  Region map: {len(region_map)} tokens")
    else:
        print("  No region map — bucket I will be skipped.")

    # ── Lexical index (shared for both key and compare_key if needed) ──────
    needs_lex = (args.key in _LEXICAL_KEYS or
                 (args.compare_key and args.compare_key in _LEXICAL_KEYS))
    lex_data = None
    if needs_lex:
        print("\n[lex_index] Building lexical index ...")
        lex_data = build_lexical_index(
            args.train_dir, args.max_postings_per_token,
            max_rows=args.max_index_rows or None)

    # ── Dense index for primary key ────────────────────────────────────────
    primary_index = None
    if args.key not in _LEXICAL_KEYS:
        print(f"\n[index] Building dense index: key='{args.key}' ...")
        primary_index = build_dense_index(
            args.key, args.train_dir, backbone, layer_info,
            device, args.max_index_rows, args.retrieval_backend)

    # ── Dense index for compare key ────────────────────────────────────────
    compare_index = None
    if args.compare_key and args.compare_key not in _LEXICAL_KEYS:
        print(f"\n[index] Building dense index: compare_key='{args.compare_key}' ...")
        compare_index = build_dense_index(
            args.compare_key, args.train_dir, backbone, layer_info,
            device, args.max_index_rows, args.retrieval_backend)

    # ── Train lookup for neighbor context display ──────────────────────────
    print("\n[train_lookup] Indexing train rows ...")
    train_lookup = build_train_row_lookup(args.train_dir, args.max_index_rows)
    shard_cache  = _ShardCache(max_size=20)

    # ── Collect all val rows ───────────────────────────────────────────────
    print("\n[collect] Iterating val shards ...")
    all_rows = collect_all_rows(
        args.val_dir, args.key, args.compare_key,
        primary_index, compare_index, lex_data,
        backbone, layer_info, device, args)

    print(f"  Total val rows: {len(all_rows):,}")

    # ── Compute entropy percentiles for bucket thresholds ─────────────────
    entropies = np.array([r["nbr_entropy"] for r in all_rows])
    e_p25  = float(np.percentile(entropies, 25))
    e_med  = float(np.median(entropies))
    e_p75  = float(np.percentile(entropies, 75))
    e_p90  = float(np.percentile(entropies, 90))
    print(f"\n  Entropy: p25={e_p25:.3f}  median={e_med:.3f}"
          f"  p75={e_p75:.3f}  p90={e_p90:.3f}")

    # ── Assign buckets ─────────────────────────────────────────────────────
    for row in all_rows:
        row["buckets"] = assign_buckets(row, e_p90, e_p25, e_med, region_map)

    # ── Select examples ────────────────────────────────────────────────────
    target   = args.num_examples_per_bucket
    buckets  = {b: [] for b in _BUCKET_META}
    for row in all_rows:
        for b in row["buckets"]:
            if b in buckets and len(buckets[b]) < target:
                buckets[b].append(row)

    print("\n  Bucket fill:")
    for b, ex in buckets.items():
        skip_note = ""
        if b == "I_region_right_token_wrong" and not region_map:
            skip_note = " [no region map — skipped]"
        print(f"    {b:42s}: {len(ex):4d}{skip_note}")

    # ── Load neighbor contexts for selected rows ───────────────────────────
    print("\n[contexts] Fetching neighbor contexts ...")
    seen_rows = set()
    for ex_list in buckets.values():
        for row in ex_list:
            rid = row["row_id"]
            if rid in seen_rows:
                continue
            seen_rows.add(rid)
            nbr_rids = row.get("nbr_row_ids")
            if nbr_rids is not None and primary_index is not None:
                row["_nbr_contexts"] = fetch_neighbor_contexts(
                    nbr_rids[:10], train_lookup, shard_cache)
            else:
                row["_nbr_contexts"] = []
    print(f"  Loaded contexts for {len(seen_rows):,} unique rows.")

    # ── Write bucket reports ───────────────────────────────────────────────
    print("\n[reports] Writing bucket markdown files ...")
    written = {}
    for b, examples in buckets.items():
        if not examples:
            if b == "I_region_right_token_wrong" and not region_map:
                continue
        write_bucket_report(
            b, examples, tokenizer, region_map,
            args.key, args.compare_key, args, args.output_root)
        written[b] = len(examples)

    # ── Global stats ──────────────────────────────────────────────────────
    global_stats = {
        "base_top1_acc":        float(np.mean([r["base_top1_correct"] for r in all_rows])),
        "gold_in_base_top_k":   float(np.mean([r["gold_in_base"]      for r in all_rows])),
        "gold_in_neighbors":    float(np.mean([r["gold_in_nbr"]       for r in all_rows])),
        "retrieval_added_gold": float(np.mean([r["ret_added"]         for r in all_rows])),
        "mean_sup_gold":        float(np.mean([r["sup_gold"]          for r in all_rows])),
        "mean_sup_margin":      float(np.mean([r["sup_margin"]        for r in all_rows])),
        "mean_nbr_entropy":     float(np.mean(entropies)),
    }

    summary = {
        "key":               args.key,
        "compare_key":       args.compare_key,
        "n_val_rows":        len(all_rows),
        "entropy_thresholds": {
            "p25": e_p25, "median": e_med, "p75": e_p75, "p90": e_p90,
        },
        "global_stats":      global_stats,
        "bucket_counts":     written,
    }
    summary_path = os.path.join(args.output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f" Analysis complete.  {args.output_root}/")
    print("=" * 70)
    print(f"\n  Global stats (key={args.key}):")
    for k, v in global_stats.items():
        print(f"    {k:35s} = {v:.4f}")
    print(f"\n  Bucket reports:")
    for b, n in written.items():
        if n > 0:
            print(f"    {n:3d} examples  →  bucket_{b}.md")


def _parse():
    p = argparse.ArgumentParser(
        description="Diagnostic bucket analysis of retrieval failure examples")
    p.add_argument("--small_ckpt",      required=True)
    p.add_argument("--train_dir",       required=True)
    p.add_argument("--val_dir",         required=True)
    p.add_argument("--sweep_root",      default=None)
    p.add_argument("--output_root",     default="runs/retrieval_failure_analysis")
    p.add_argument("--key",             default="h_prime")
    p.add_argument("--compare_key",     default=None)
    p.add_argument("--token_to_region", default=None)
    p.add_argument("--num_examples_per_bucket", type=int, default=50)
    p.add_argument("--num_neighbors",           type=int, default=32)
    p.add_argument("--top_k",                   type=int, default=256)
    p.add_argument("--max_index_rows",          type=int, default=500000)
    p.add_argument("--max_val_rows",            type=int, default=0,
                   help="Cap on val rows evaluated (0 = all)")
    p.add_argument("--retrieval_backend",       default="auto",
                   choices=["auto", "faiss", "torch_chunked"])
    p.add_argument("--retrieval_chunk_size",    type=int, default=131072)
    p.add_argument("--query_batch_size",        type=int, default=512)
    p.add_argument("--support_tau",             type=float, default=0.2)
    p.add_argument("--max_postings_per_token",          type=int, default=5000)
    p.add_argument("--max_lexical_candidates_per_query", type=int, default=20000)
    return p.parse_args()


if __name__ == "__main__":
    main()
