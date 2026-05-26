#!/usr/bin/env python3
"""
eval_format_syntax_diagnostic_v1.py — Format/Syntax Memory Diagnostic V1.

Test whether some Type-A/detail errors are really formatting or syntax-state
errors (comma vs period, quote vs colon, closing bracket, wiki headings, @-@
markup, sentence boundaries, etc.) and whether crude heuristic format-state
features can improve them.

NO training.  NO model changes.
Val gold used ONLY for metrics after scoring.
Gold never used to build features or choose candidates.

Steps:
  1. Inspect first val shard (keys/shapes).
  2. Classify all GPT-2 tokens into format categories.
  3. Build format slices from val data.
  4. Extract prefix format-state features from last memory_len input tokens.
  5. Evaluate base_candidate and base_plus_format_score policies.
  6. Collect examples (helps/hurts/confusers).
  7. Write reports and CSVs.

Answers:
  Q1. What fraction of Bucket A errors are format/syntax-like?
  Q2. Which format confusers are most common?
  Q3. Do simple format-state features improve target slices?
  Q4. Are changed_away errors controlled?
  Q5. Should we train a learned FormatStateExpert next?

Usage:
  python scripts/eval_format_syntax_diagnostic_v1.py \\
    --val_dir runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val \\
    --token_to_region runs/region_maps_128/token_to_region.json \\
    --super_map runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
    --output_dir runs/format_memory_v1/format_syntax_diagnostic_v1 \\
    --candidate_pool_size 32 --memory_len 128 \\
    --lambda_grid 0,0.25,0.5,1,2,4 \\
    --max_examples 50 --seed 42
"""

import argparse
import csv
import glob
import json
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
VOCAB_SIZE = 50257

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic definitions
# ─────────────────────────────────────────────────────────────────────────────

HEURISTIC_NAMES = [
    "h_close_paren",   # 0: any unmatched '('→boost ')'  '[' →']'  '{'→'}'
    "h_quote_close",   # 1: odd quote count in context → boost quote-like
    "h_wiki_equal",    # 2: recent '=' in last 16 tokens → boost '='
    "h_at_hyphen",     # 3: recent '@' in last 8 tokens → boost '@'/'hyphen_markup'
    "h_sentence_end",  # 4: last ctx token is sentence-end or newline → boost '.'/'!'/'?'
    "h_list_comma",    # 5: recent comma in last 16 AND last isn't comma → boost ','
]
N_HEURISTICS = len(HEURISTIC_NAMES)

# ─────────────────────────────────────────────────────────────────────────────
# Shard field aliases
# ─────────────────────────────────────────────────────────────────────────────

_TOPK_ALIASES  = ["base_topk_ids",    "base_topk",     "topk_ids"]
_LGT_ALIASES   = ["base_topk_logits", "base_topk_lgt", "topk_lgt", "topk_logits"]
_GOLD_ALIASES  = ["gold_token",       "gold",          "labels"]
_IDS_ALIASES   = ["input_ids"]
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
# Tokenizer helpers (optional — degrades gracefully if tiktoken absent)
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
# Token classification table
# ─────────────────────────────────────────────────────────────────────────────

def build_token_tables(vocab_size=VOCAB_SIZE):
    """
    Decode all GPT-2 tokens and build boolean indicator arrays.

    Returns:
      indicator_arrays: dict  name -> np.ndarray[V] bool
      token_classes:    dict  tid  -> frozenset of class strings
      token_strings:    dict  tid  -> decoded str
    """
    arr = {k: np.zeros(vocab_size, dtype=bool) for k in [
        "is_open_paren",    "is_close_paren",
        "is_open_bracket",  "is_close_bracket",
        "is_open_curly",    "is_close_curly",
        "is_quote",         "is_wiki_equal",
        "is_at_hyphen",     "is_comma",
        "is_period",        "is_sentence_end",
        "is_newline",       "is_colon",
        "is_format",        "is_punctuation",
        "is_bpe_fragment",
    ]}
    token_classes = {}
    token_strings = {}

    for tid in range(vocab_size):
        if _enc is not None:
            try:
                s = _enc.decode([tid])
            except Exception:
                s = ""
        else:
            s = f"<{tid}>"
        token_strings[tid] = s
        stripped = s.strip()
        cats = set()

        if not stripped:
            if '\n' in s or '\r' in s:
                cats.update(["newline_like", "format_like"])
                arr["is_newline"][tid] = True
                arr["is_format"][tid]  = True
        elif stripped == ',':
            cats.update(["comma_like", "punctuation", "format_like"])
            arr["is_comma"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped in ('.', '…', '...'):
            cats.update(["period_like", "sentence_end", "punctuation", "format_like"])
            arr["is_period"][tid] = arr["is_sentence_end"][tid] = True
            arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped in ('!', '?'):
            cats.update(["sentence_end", "punctuation", "format_like"])
            arr["is_sentence_end"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped in ('"', "'", '`', '“', '”', '‘', '’', "``", "''"):
            cats.update(["quote_like", "punctuation", "format_like"])
            arr["is_quote"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped in (':', ';'):
            cats.update(["colon_semicolon", "punctuation", "format_like"])
            arr["is_colon"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped == '(':
            cats.update(["bracket", "open_paren", "punctuation", "format_like"])
            arr["is_open_paren"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped == ')':
            cats.update(["bracket", "close_paren", "punctuation", "format_like"])
            arr["is_close_paren"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped == '[':
            cats.update(["bracket", "open_bracket", "punctuation", "format_like"])
            arr["is_open_bracket"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped == ']':
            cats.update(["bracket", "close_bracket", "punctuation", "format_like"])
            arr["is_close_bracket"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped == '{':
            cats.update(["bracket", "open_curly", "punctuation", "format_like"])
            arr["is_open_curly"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped == '}':
            cats.update(["bracket", "close_curly", "punctuation", "format_like"])
            arr["is_close_curly"][tid] = arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif stripped in ('=', '==', '===', '====', '====='):
            cats.update(["wiki_heading", "format_like"])
            arr["is_wiki_equal"][tid] = arr["is_format"][tid] = True
        elif '@' in stripped:
            cats.update(["hyphen_markup", "format_like"])
            arr["is_at_hyphen"][tid] = arr["is_format"][tid] = True
        elif stripped in ('-', '--', '–', '—'):
            cats.update(["punctuation", "format_like"])
            arr["is_punctuation"][tid] = arr["is_format"][tid] = True
        elif '\n' in s:
            cats.update(["newline_like", "format_like"])
            arr["is_newline"][tid] = arr["is_format"][tid] = True
        else:
            if (not s.startswith(' ') and stripped.isalpha() and 1 <= len(stripped) <= 4):
                cats.add("bpe_fragment_like")
                arr["is_bpe_fragment"][tid] = True

        if not cats:
            cats.add("word_or_other")
        token_classes[tid] = frozenset(cats)

    print(f"[token_tables] {vocab_size} tokens classified")
    for k in ["is_comma", "is_period", "is_quote", "is_wiki_equal",
              "is_at_hyphen", "is_open_paren", "is_close_paren",
              "is_sentence_end", "is_newline", "is_format"]:
        print(f"  {k:25s}: {arr[k].sum()} tokens")
    return arr, token_classes, token_strings


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic applies table  [V, H]
# ─────────────────────────────────────────────────────────────────────────────

def build_heuristic_applies(arr, vocab_size=VOCAB_SIZE):
    """
    [V, H] float32.
    heuristic_applies[tok, h] = 1 if tok is a target token for heuristic h.
    """
    close_any = arr["is_close_paren"] | arr["is_close_bracket"] | arr["is_close_curly"]
    H = np.stack([
        close_any,              # H0: close_paren/bracket/curly
        arr["is_quote"],        # H1: quote_close
        arr["is_wiki_equal"],   # H2: wiki_equal
        arr["is_at_hyphen"],    # H3: at_hyphen
        arr["is_sentence_end"], # H4: sentence_end
        arr["is_comma"],        # H5: list_comma
    ], axis=1).astype(np.float32)   # [V, 6]
    return H


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
            print("[preflight] first val shard keys:")
            for k, v in sh.items():
                print(f"  {k:30s}: shape={getattr(v,'shape',None)}"
                      f"  dtype={getattr(v,'dtype',type(v).__name__)}")
            first = False
        topk = _get(sh, _TOPK_ALIASES).long()
        lgt  = _get(sh, _LGT_ALIASES).float()
        gold = _get(sh, _GOLD_ALIASES).long()
        ids  = _get(sh, _IDS_ALIASES).long()
        rids = _get(sh, _ROWID_ALIASES, required=False)
        off  = _get(sh, _OFF_ALIASES, required=False)
        B, K = topk.shape
        if rids is None: rids = torch.arange(total, total + B)
        if off  is None: off  = torch.full((B,), -1, dtype=torch.long)
        if K < cand_pool_size:
            topk = torch.cat([topk, torch.zeros(B, cand_pool_size-K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, cand_pool_size-K), float("nan"))], 1)
        elif K > cand_pool_size:
            topk, lgt = topk[:, :cand_pool_size], lgt[:, :cand_pool_size]
        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, ids, rids, off = (x[:keep] for x in (topk, lgt, gold, ids, rids, off))
            B = keep
        bufs["topk"].append(topk.numpy())
        bufs["lgt"].append(lgt.numpy())
        bufs["gold"].append(gold.numpy())
        bufs["ids"].append(ids.numpy())
        bufs["row_ids"].append(rids.numpy())
        bufs["token_offset"].append(off.numpy())
        total += B
    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    data["lgt"] = np.where(np.isfinite(data["lgt"]), data["lgt"], -1e9)
    print(f"[shards] {total:,} val rows  pool_size={cand_pool_size}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Region maps
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
# Prefix format-state feature extraction  (vectorised over all N rows)
# ─────────────────────────────────────────────────────────────────────────────

def extract_state_features(ids_batch, arr, memory_len, vocab_size=VOCAB_SIZE):
    """
    ids_batch: [N, seq_len] int64
    arr:       indicator_arrays dict  name -> [V] bool
    Returns dict of [N] arrays.
    """
    N, seq_len = ids_batch.shape
    M   = min(seq_len, memory_len)
    ctx = np.clip(ids_batch[:, -M:], 0, vocab_size - 1)   # [N, M]

    def _sum(key, ctx_): return arr[key][ctx_].sum(axis=1).astype(np.int32)

    unmatched_paren   = (_sum("is_open_paren",   ctx) - _sum("is_close_paren",   ctx))
    unmatched_bracket = (_sum("is_open_bracket",  ctx) - _sum("is_close_bracket", ctx))
    unmatched_curly   = (_sum("is_open_curly",    ctx) - _sum("is_close_curly",   ctx))
    quote_count       = _sum("is_quote", ctx)

    T16 = min(16, M);  T8 = min(8, M)
    ctx16  = ctx[:, -T16:]
    ctx8   = ctx[:, -T8:]
    recent_eq    = _sum("is_wiki_equal", ctx16)
    recent_at    = _sum("is_at_hyphen",  ctx8)
    recent_comma = _sum("is_comma",      ctx16)

    last = ctx[:, -1] if M > 0 else np.zeros(N, dtype=np.int64)
    last_is_sent = arr["is_sentence_end"][last]
    last_is_nl   = arr["is_newline"][last]
    last_is_com  = arr["is_comma"][last]
    last_is_fmt  = arr["is_format"][last]

    # format density in context
    fmt_count    = _sum("is_format", ctx)
    fmt_density  = fmt_count.astype(np.float32) / max(M, 1)

    return {
        "unmatched_paren":     unmatched_paren,
        "unmatched_bracket":   unmatched_bracket,
        "unmatched_curly":     unmatched_curly,
        "quote_count":         quote_count,
        "recent_eq":           recent_eq,
        "recent_at":           recent_at,
        "recent_comma":        recent_comma,
        "last_is_sentence_end": last_is_sent,
        "last_is_newline":     last_is_nl,
        "last_is_comma":       last_is_com,
        "last_is_format":      last_is_fmt,
        "fmt_density":         fmt_density,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic firing matrix  [N, H]
# ─────────────────────────────────────────────────────────────────────────────

def compute_fires(state):
    N = len(state["unmatched_paren"])
    fires = np.zeros((N, N_HEURISTICS), dtype=np.float32)
    fires[:, 0] = ((state["unmatched_paren"]   > 0) |
                   (state["unmatched_bracket"]  > 0) |
                   (state["unmatched_curly"]    > 0)).astype(np.float32)
    fires[:, 1] = (state["quote_count"] % 2).astype(np.float32)
    fires[:, 2] = (state["recent_eq"]  > 0).astype(np.float32)
    fires[:, 3] = (state["recent_at"]  > 0).astype(np.float32)
    fires[:, 4] = (state["last_is_sentence_end"] | state["last_is_newline"]).astype(np.float32)
    fires[:, 5] = ((state["recent_comma"] > 0) & ~state["last_is_comma"]).astype(np.float32)
    return fires


# ─────────────────────────────────────────────────────────────────────────────
# Format score computation  [N, P]  — fully vectorised
# ─────────────────────────────────────────────────────────────────────────────

def compute_format_scores(fires, topk, heuristic_applies, vocab_size=VOCAB_SIZE):
    """
    fires:             [N, H] float32
    topk:              [N, P] int64  (candidate token IDs)
    heuristic_applies: [V, H] float32
    Returns:           [N, P] float32 format scores
    """
    V = heuristic_applies.shape[0]
    topk_c  = np.clip(topk, 0, V - 1)
    applies = heuristic_applies[topk_c]              # [N, P, H]
    scores  = (applies * fires[:, None, :]).sum(axis=2)  # [N, P]
    return scores.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Slice builder
# ─────────────────────────────────────────────────────────────────────────────

def build_slices(data, arr, tok_arr, reg_arr, unk_r, unk_s, sr):
    N     = len(data["gold"])
    gold  = data["gold"].astype(np.int64)
    topk  = data["topk"].astype(np.int64)
    lgt   = data["lgt"].astype(np.float32)
    P     = topk.shape[1]
    ar    = np.arange(N)
    V     = len(next(v for v in arr.values()))

    base_top1    = topk[ar, lgt.argmax(axis=1)]
    gold_in_pool = (topk == gold[:, None]).any(axis=1)

    shifted   = lgt - lgt.max(axis=1, keepdims=True)
    base_probs = np.exp(shifted); base_probs /= base_probs.sum(axis=1, keepdims=True) + _EPS
    sp         = np.sort(base_probs, axis=1)[:, ::-1]
    base_margin = sp[:, 0] - (sp[:, 1] if P >= 2 else sp[:, 0])

    gc  = np.clip(gold,      0, V-1)
    bc  = np.clip(base_top1, 0, V-1)

    def gi(key): return arr[key][gc]
    def bi(key): return arr[key][bc]

    gold_is_fmt    = gi("is_format")
    base_is_fmt    = bi("is_format")
    gold_is_punct  = gi("is_punctuation")
    base_is_punct  = bi("is_punctuation")
    gold_is_comma  = gi("is_comma")
    base_is_comma  = bi("is_comma")
    gold_is_period = gi("is_period")
    base_is_period = bi("is_period")
    gold_is_quote  = gi("is_quote")
    base_is_quote  = bi("is_quote")
    gold_is_colon  = gi("is_colon")
    base_is_colon  = bi("is_colon")
    gold_is_brk    = (arr["is_open_paren"] | arr["is_close_paren"] |
                      arr["is_open_bracket"] | arr["is_close_bracket"] |
                      arr["is_open_curly"] | arr["is_close_curly"])[gc]
    base_is_brk    = (arr["is_open_paren"] | arr["is_close_paren"] |
                      arr["is_open_bracket"] | arr["is_close_bracket"] |
                      arr["is_open_curly"] | arr["is_close_curly"])[bc]
    gold_is_wiki   = gi("is_wiki_equal")
    base_is_wiki   = bi("is_wiki_equal")
    gold_is_at     = gi("is_at_hyphen")
    base_is_at     = bi("is_at_hyphen")
    gold_is_sen    = (arr["is_sentence_end"] | arr["is_newline"])[gc]

    # candidate pool has at least one format token
    cand_has_fmt   = arr["is_format"][np.clip(topk, 0, V-1)].any(axis=1)

    # region slices
    Vr = len(tok_arr)
    gold_reg = tok_arr[np.clip(gc, 0, Vr-1)]
    base_reg = tok_arr[np.clip(bc, 0, Vr-1)]
    same_reg = (gold_reg == base_reg) & (gold_reg != unk_r)
    same_sr  = same_reg.copy()
    if sr:
        Rr = len(reg_arr)
        gs = reg_arr[np.clip(gold_reg, 0, Rr-1)]
        bs = reg_arr[np.clip(base_reg, 0, Rr-1)]
        same_sr = (gs == bs) & (gs != unk_s)

    base_wrong = (base_top1 != gold)
    fmt_wrong  = gold_is_fmt & base_is_fmt & base_wrong

    slices = {
        "all":                        np.ones(N, dtype=bool),
        "bucketA_confuser":           base_wrong & gold_in_pool,
        "gold_format":                gold_is_fmt,
        "base_format_wrong":          fmt_wrong,
        "candidate_has_format":       cand_has_fmt,
        "punctuation_confuser":       gold_is_punct & base_is_punct & base_wrong & gold_in_pool,
        "comma_vs_period":            ((gold_is_comma & base_is_period) |
                                       (gold_is_period & base_is_comma)) & gold_in_pool,
        "quote_colon_confuser":       ((gold_is_quote | gold_is_colon) &
                                       (base_is_quote | base_is_colon) &
                                       base_wrong & gold_in_pool),
        "bracket_confuser":           gold_is_brk & base_is_brk & base_wrong & gold_in_pool,
        "wiki_heading_confuser":      gold_is_wiki & ~base_is_wiki & gold_in_pool,
        "hyphen_markup_confuser":     gold_is_at   & ~base_is_at   & gold_in_pool,
        "sentence_boundary":          gold_is_sen,
        "same_region_format_confuser":   same_reg & fmt_wrong,
        "same_superregion_format_confuser": same_sr & fmt_wrong,
        "high_base_margin":           base_margin > 0.2,
        "low_base_margin":            base_margin < 0.05,
    }

    print("[slices]")
    for s, m in slices.items():
        print(f"  {s:40s} = {m.sum():6,}/{N:,}")

    extra = {
        "base_top1":     base_top1,
        "gold_in_pool":  gold_in_pool,
        "base_margin":   base_margin,
        "base_probs":    base_probs,
        "gold_is_fmt":   gold_is_fmt,
        "cand_has_fmt":  cand_has_fmt,
    }
    return slices, extra


# ─────────────────────────────────────────────────────────────────────────────
# Policy metrics (per slice)
# ─────────────────────────────────────────────────────────────────────────────

def compute_policy_metrics(topk, lgt, format_scores, gold, base_top1, gold_in_pool,
                            slicemask, lam):
    N  = len(gold)
    ar = np.arange(N)

    combined   = lgt + lam * format_scores        # [N, P]
    fmt_pred   = topk[ar, combined.argmax(axis=1)]  # [N]
    apply_mask = fmt_pred != base_top1             # [N]

    m = slicemask
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "base_acc": float("nan"), "policy_acc": float("nan"),
                "acc_gain": float("nan"), "changed_to_gold": float("nan"),
                "changed_away": float("nan"), "net_correction": float("nan"),
                "benefit_damage_ratio": float("nan"), "apply_rate": float("nan"),
                "selected_gold_given_in_pool": float("nan")}

    g   = gold[m]; bp  = base_top1[m]
    fp  = fmt_pred[m]; am  = apply_mask[m]
    gip = gold_in_pool[m]

    base_acc   = float((bp == g).mean())
    policy_acc = float((fp == g).mean())
    acc_gain   = policy_acc - base_acc
    ctg        = float(((fp == g) & (bp != g)).mean())
    caw        = float(((fp != g) & (bp == g)).mean())
    net_corr   = ctg - caw
    bdr        = min(ctg / max(caw, _EPS), 999.0)
    apply_rate = float(am.mean())
    n_gip      = int(gip.sum())
    sggip      = float((fp[gip] == g[gip]).mean()) if n_gip > 0 else float("nan")

    return {
        "n":                          n,
        "base_acc":                   base_acc,
        "policy_acc":                 policy_acc,
        "acc_gain":                   acc_gain,
        "changed_to_gold":            ctg,
        "changed_away":               caw,
        "net_correction":             net_corr,
        "benefit_damage_ratio":       bdr,
        "apply_rate":                 apply_rate,
        "selected_gold_given_in_pool": sggip,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_heuristic_stats(fires, data, extra, heuristic_applies):
    gold      = data["gold"].astype(np.int64)
    base_top1 = extra["base_top1"]
    gip       = extra["gold_in_pool"]
    V         = heuristic_applies.shape[0]
    gc        = np.clip(gold, 0, V-1)
    bucketA   = gip & (base_top1 != gold)

    rows = []
    for h, name in enumerate(HEURISTIC_NAMES):
        fire_h  = fires[:, h].astype(bool)
        gold_target_h = heuristic_applies[gc, h].astype(bool)
        n_all   = len(gold); n_ba = max(bucketA.sum(), 1)
        rows.append({
            "heuristic":                name,
            "fire_rate_all":            float(fire_h.mean()),
            "fire_rate_bucketA":        float(fire_h[bucketA].mean()) if bucketA.sum() > 0 else float("nan"),
            "gold_is_target_all":       float(gold_target_h.mean()),
            "gold_is_target_bucketA":   float(gold_target_h[bucketA].mean()) if bucketA.sum() > 0 else float("nan"),
            "fires_and_gold_target":    float((fire_h & gold_target_h).mean()),
            "fires_and_gold_in_pool":   float((fire_h & gip).mean()),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Feature summary (per slice)
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_summary(state, slices):
    rows = []
    for sname, smask in slices.items():
        n = int(smask.sum())
        if n == 0:
            continue
        rows.append({
            "slice":             sname,
            "n":                 n,
            "frac_unmatched_paren": float((state["unmatched_paren"][smask] > 0).mean()),
            "mean_unmatched_paren": float(state["unmatched_paren"][smask].mean()),
            "frac_inside_quote":    float((state["quote_count"][smask] % 2 == 1).mean()),
            "mean_quote_count":     float(state["quote_count"][smask].mean()),
            "frac_recent_eq":       float((state["recent_eq"][smask] > 0).mean()),
            "frac_recent_at":       float((state["recent_at"][smask] > 0).mean()),
            "frac_recent_comma":    float((state["recent_comma"][smask] > 0).mean()),
            "mean_fmt_density":     float(state["fmt_density"][smask].mean()),
            "frac_last_is_sentend": float(state["last_is_sentence_end"][smask].mean()),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Example collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_examples(data, format_scores, fires, extra, slices,
                     token_strings, max_examples, best_lam, seed):
    rng = random.Random(seed)
    N   = len(data["gold"])
    ar  = np.arange(N)
    P   = data["topk"].shape[1]
    lgt = data["lgt"]
    topk = data["topk"]
    gold = data["gold"].astype(np.int64)
    base_top1 = extra["base_top1"]

    combined = lgt + best_lam * format_scores   # [N, P]
    fmt_pred = topk[ar, combined.argmax(axis=1)]

    def tok_name(tid):
        s = token_strings.get(int(tid), f"<{tid}>")
        return f"'{repr(s)[1:-1]}' (id={tid})"

    def make_ex(i):
        cand_rows = []
        for j in range(min(P, 16)):
            tid = int(topk[i, j])
            cand_rows.append({
                "token": tok_name(tid),
                "base_lgt": float(lgt[i, j]),
                "fmt_score": float(format_scores[i, j]),
                "combined": float(combined[i, j]),
                "is_gold": tid == int(gold[i]),
            })
        cand_rows.sort(key=lambda r: r["combined"], reverse=True)
        fired = [HEURISTIC_NAMES[h] for h in range(N_HEURISTICS) if fires[i, h] > 0]
        return {
            "i":            int(i),
            "row_id":       int(data["row_ids"][i]),
            "token_offset": int(data["token_offset"][i]),
            "ctx_tail":     data["ids"][i, -20:].tolist(),
            "gold":         tok_name(gold[i]),
            "base_top1":    tok_name(base_top1[i]),
            "fmt_top1":     tok_name(fmt_pred[i]),
            "fired":        fired,
            "lambda":       best_lam,
            "cands":        cand_rows,
        }

    helps  = []   # base wrong → fmt right
    hurts  = []   # base right → fmt wrong
    confus = []   # both gold and base are format tokens

    bucketA = slices["bucketA_confuser"]
    gold_fmt = extra["gold_is_fmt"]

    indices = list(range(N))
    rng.shuffle(indices)
    for i in indices:
        b1  = int(base_top1[i]); fp = int(fmt_pred[i]); g = int(gold[i])
        if b1 != g and fp == g and len(helps)  < max_examples:
            helps.append(make_ex(i))
        elif b1 == g and fp != g and len(hurts) < max_examples:
            hurts.append(make_ex(i))
        elif (gold_fmt[i] and extra["base_top1"][i] != gold[i]
              and extra["gold_in_pool"][i] and len(confus) < max_examples):
            confus.append(make_ex(i))
        if len(helps) >= max_examples and len(hurts) >= max_examples and len(confus) >= max_examples:
            break

    return {"helps": helps, "hurts": hurts, "confusers": confus}


# ─────────────────────────────────────────────────────────────────────────────
# Report / output writer
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ex_md(ex, token_strings):
    lines = [
        f"**Row** {ex['row_id']}  offset={ex['token_offset']}",
        f"  context: `...{_decode_ids(ex['ctx_tail'])}`",
        f"  gold:     {ex['gold']}",
        f"  base:     {ex['base_top1']}",
        f"  fmt(λ={ex['lambda']}): {ex['fmt_top1']}",
        f"  fired:    {ex['fired']}",
        "",
        "  | token | base_lgt | fmt_score | combined | gold? |",
        "  |-------|----------|-----------|----------|-------|",
    ]
    for r in ex["cands"][:8]:
        g = "✓" if r["is_gold"] else ""
        lines.append(f"  | {r['token']:30s} | {r['base_lgt']:8.3f} | "
                     f"{r['fmt_score']:9.3f} | {r['combined']:8.3f} | {g} |")
    return "\n".join(lines) + "\n"


def write_outputs(all_metric_rows, heuristic_rows, feature_rows,
                  examples, slices, data, extra, args, N_total,
                  lambda_grid, token_strings):
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ── policy_grid.csv ─────────────────────────────────────────────────
    if all_metric_rows:
        path = os.path.join(out, "format_policy_grid.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_metric_rows[0].keys()))
            w.writeheader(); w.writerows(all_metric_rows)
        print(f"[save] {path}")

    # ── slice_metrics.csv  (best lambda per slice) ───────────────────────
    best_by_slice = {}
    for row in all_metric_rows:
        key = (row["slice"], row["policy"])
        if row["policy"] == "base_candidate":
            best_by_slice[key] = row
        elif (key not in best_by_slice
              or (not np.isnan(row.get("acc_gain", float("nan")))
                  and row["acc_gain"] > best_by_slice[key].get("acc_gain", -1e9))):
            best_by_slice[key] = row
    slice_rows = sorted(best_by_slice.values(), key=lambda r: (r["slice"], r["policy"]))
    if slice_rows:
        path = os.path.join(out, "format_slice_metrics.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(slice_rows[0].keys()))
            w.writeheader(); w.writerows(slice_rows)
        print(f"[save] {path}")

    # ── heuristic_stats.csv ──────────────────────────────────────────────
    if heuristic_rows:
        path = os.path.join(out, "format_heuristic_stats.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(heuristic_rows[0].keys()))
            w.writeheader(); w.writerows(heuristic_rows)
        print(f"[save] {path}")

    # ── feature_summary.csv ──────────────────────────────────────────────
    if feature_rows:
        path = os.path.join(out, "format_feature_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
            w.writeheader(); w.writerows(feature_rows)
        print(f"[save] {path}")

    # ── example files ────────────────────────────────────────────────────
    def write_ex(path, exs, header):
        with open(path, "w") as f:
            f.write(f"# {header}\n\n({len(exs)} examples)\n\n")
            for ex in exs:
                f.write("---\n")
                f.write(_fmt_ex_md(ex, token_strings))
                f.write("\n")
        print(f"[save] {path}")

    write_ex(os.path.join(out, "examples_format_helps.md"),
             examples["helps"], "Format score corrected base prediction")
    write_ex(os.path.join(out, "examples_format_hurts.md"),
             examples["hurts"], "Format score broke a correct base prediction")
    write_ex(os.path.join(out, "examples_format_confusers.md"),
             examples["confusers"], "Format token confuser (gold is format, base wrong)")

    # ── report.md ────────────────────────────────────────────────────────
    N  = N_total

    def get_base_acc(sname):
        r = next((r for r in all_metric_rows if r["slice"] == sname
                  and r["policy"] == "base_candidate"), None)
        return r["base_acc"] if r else float("nan")

    def get_best_lam_acc(sname):
        cands = [r for r in all_metric_rows
                 if r["slice"] == sname and r["policy"] == "base_plus_format"]
        if not cands: return float("nan"), float("nan"), float("nan")
        best = max(cands, key=lambda r: r.get("policy_acc", -1e9))
        return best.get("policy_acc", float("nan")), best.get("lambda", 0), best.get("acc_gain", float("nan"))

    gold = data["gold"]; base_top1 = extra["base_top1"]; gip = extra["gold_in_pool"]
    n_bucketA = int(slices["bucketA_confuser"].sum())
    n_gold_fmt_in_bucketA = int((slices["bucketA_confuser"] & extra["gold_is_fmt"]).sum())
    frac_bucketA_is_fmt   = n_gold_fmt_in_bucketA / max(n_bucketA, 1)

    # confuser counts
    confuser_counts = {}
    for sname in ["comma_vs_period", "quote_colon_confuser", "bracket_confuser",
                  "wiki_heading_confuser", "hyphen_markup_confuser",
                  "punctuation_confuser", "sentence_boundary"]:
        confuser_counts[sname] = int(slices[sname].sum())

    report_path = os.path.join(out, "format_syntax_report.md")
    with open(report_path, "w") as f:
        f.write("# Format / Syntax Memory Diagnostic V1 — Report\n\n")
        f.write("> NON-PARAMETRIC DIAGNOSTIC. Val gold used only for metrics.\n\n")
        f.write(f"**N_val:** {N:,}  |  **Candidate pool:** {data['topk'].shape[1]}  |  "
                f"**λ grid:** {lambda_grid}  |  **memory_len:** {args.memory_len}\n\n")
        f.write("---\n\n")

        f.write("## Q1: What fraction of Bucket A errors are format/syntax-like?\n\n")
        f.write(f"- Bucket A confuser rows (base_wrong ∩ gold_in_pool): **{n_bucketA:,}**\n")
        f.write(f"- Of those, gold token is format-like: **{n_gold_fmt_in_bucketA:,}** "
                f"({frac_bucketA_is_fmt:.1%})\n")
        q1 = ("✅ Substantial" if frac_bucketA_is_fmt > 0.15 else
              "⚠ Moderate" if frac_bucketA_is_fmt > 0.05 else "❌ Low")
        f.write(f"- **{q1}** fraction of errors are format/syntax-related\n\n")

        f.write("## Q2: Which format confusers are most common?\n\n")
        f.write("| Confuser type | Count |\n|---|---|\n")
        for sname, cnt in sorted(confuser_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {sname} | {cnt:,} |\n")
        f.write("\n")

        f.write("## Q3: Do simple format-state features improve target slices?\n\n")
        f.write("| slice | base_acc | best_acc | Δ | best_λ |\n")
        f.write("|-------|----------|----------|---|--------|\n")
        for sname in ["gold_format", "base_format_wrong", "punctuation_confuser",
                      "comma_vs_period", "quote_colon_confuser", "bracket_confuser",
                      "wiki_heading_confuser", "bucketA_confuser", "all"]:
            ba = get_base_acc(sname)
            pa, bl, ag = get_best_lam_acc(sname)
            f.write(f"| {sname} | {ba:.4f} | {pa:.4f} | {ag:+.4f} | {bl} |\n")
        f.write("\n")

        f.write("## Q4: Are changed_away errors controlled?\n\n")
        for sname in ["gold_format", "bucketA_confuser", "all"]:
            rows_s = [r for r in all_metric_rows
                      if r["slice"] == sname and r["policy"] == "base_plus_format"]
            if not rows_s: continue
            best = max(rows_s, key=lambda r: r.get("net_correction", -1e9))
            bdr = best.get("benefit_damage_ratio", float("nan"))
            verdict = ("✅ Controlled (BDR>2)" if bdr > 2 else
                       "⚠ Marginal (BDR 1-2)" if bdr > 1 else
                       "❌ More hurt than help")
            f.write(f"- **{sname}** (λ={best.get('lambda','?')}): "
                    f"ctg={best.get('changed_to_gold','?'):.4f}  "
                    f"caw={best.get('changed_away','?'):.4f}  "
                    f"BDR={bdr:.2f}  → **{verdict}**\n")
        f.write("\n")

        f.write("## Q5: Should we train a learned FormatStateExpert next?\n\n")
        pa_all, bl_all, ag_all = get_best_lam_acc("gold_format")
        bdr_vals = [r.get("benefit_damage_ratio", 0)
                    for r in all_metric_rows
                    if r["slice"] == "gold_format" and r["policy"] == "base_plus_format"]
        max_bdr = max(bdr_vals) if bdr_vals else 0.0
        if frac_bucketA_is_fmt > 0.10 and max_bdr > 1.5 and not np.isnan(ag_all) and ag_all > 0.005:
            verdict5 = ("✅ YES — meaningful format fraction + heuristic gain + controlled damage. "
                        "Train a learned FormatStateExpert.")
        elif frac_bucketA_is_fmt > 0.05 and max_bdr > 1.0:
            verdict5 = "⚠ MAYBE — moderate signal; try rule refinement before training."
        else:
            verdict5 = "❌ NO — insufficient signal or heuristics cause more harm than good."
        f.write(f"- Format fraction of BucketA: {frac_bucketA_is_fmt:.1%}\n")
        f.write(f"- Best acc gain on gold_format slice: {ag_all:+.4f} (λ={bl_all})\n")
        f.write(f"- Max BDR on gold_format: {max_bdr:.2f}\n")
        f.write(f"- **{verdict5}**\n\n")

        f.write("---\n\n## Heuristic Fire Rates\n\n")
        f.write("| heuristic | fire_rate_all | fire_rate_bucketA | "
                "gold_target_all | gold_target_bucketA |\n")
        f.write("|-----------|---------------|-------------------|"
                "----------------|--------------------|\n")
        for r in heuristic_rows:
            f.write(f"| {r['heuristic']:20s} | {r['fire_rate_all']:.4f} | "
                    f"{r['fire_rate_bucketA']:.4f} | "
                    f"{r['gold_is_target_all']:.4f} | "
                    f"{r['gold_is_target_bucketA']:.4f} |\n")

        f.write(f"\n\n*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"[save] {report_path}")

    # ── config.json ──────────────────────────────────────────────────────
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({
            "val_dir":            args.val_dir,
            "candidate_pool_size": args.candidate_pool_size,
            "memory_len":         args.memory_len,
            "lambda_grid":        lambda_grid,
            "N_val":              N_total,
            "heuristics":         HEURISTIC_NAMES,
        }, f, indent=2)

    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Format/Syntax Memory Diagnostic V1")
    p.add_argument("--val_dir",             required=True)
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--token_to_region",     default=None)
    p.add_argument("--super_map",           default=None)
    p.add_argument("--candidate_pool_size", type=int, default=32)
    p.add_argument("--memory_len",          type=int, default=128)
    p.add_argument("--lambda_grid",         default="0,0.25,0.5,1,2,4")
    p.add_argument("--max_val_rows",        type=int, default=None)
    p.add_argument("--max_examples",        type=int, default=50)
    p.add_argument("--seed",               type=int, default=42)
    return p, p.parse_args()


def main():
    _, args = _parse()
    np.random.seed(args.seed)
    random.seed(args.seed)
    t0 = time.time()

    print("=" * 64)
    print(" Format/Syntax Memory Diagnostic V1")
    print(f" val_dir:    {args.val_dir}")
    print(f" output_dir: {args.output_dir}")
    print("=" * 64)

    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]

    # ── Step 1-2: token classification ──────────────────────────────────
    print("\n[step 1-2] building token tables...")
    arr, token_classes, token_strings = build_token_tables()
    heur_applies = build_heuristic_applies(arr)

    # ── Val shards ───────────────────────────────────────────────────────
    print("\n[step] loading val shards...")
    data = load_val_shards(args.val_dir, args.candidate_pool_size, args.max_val_rows)
    N    = len(data["gold"])

    # ── Region maps ──────────────────────────────────────────────────────
    tok_arr = reg_arr = None; unk_r = unk_s = 1; sr = False
    if args.token_to_region and os.path.isfile(args.token_to_region):
        tok_arr, reg_arr, unk_r, unk_s, sr = load_maps(args.token_to_region, args.super_map)
        print(f"[maps] loaded  sr={sr}")
    else:
        print("[maps] token_to_region not found — region slices disabled")
        V = int(data["topk"].max()) + 2
        tok_arr  = np.zeros(V, dtype=np.int32)
        reg_arr  = np.zeros(2, dtype=np.int32)

    # ── Step 3: slices ───────────────────────────────────────────────────
    print("\n[step 3] building slices...")
    slices, extra = build_slices(data, arr, tok_arr, reg_arr, unk_r, unk_s, sr)

    # ── Step 4: prefix state features ───────────────────────────────────
    print("\n[step 4] extracting prefix state features...")
    state = extract_state_features(data["ids"], arr, args.memory_len)
    fires = compute_fires(state)
    print(f"  any_fires: {(fires.any(axis=1)).mean():.3f} of rows have ≥1 heuristic fire")
    for h, name in enumerate(HEURISTIC_NAMES):
        print(f"  {name}: fire_rate={fires[:, h].mean():.4f}")

    # ── Step 5: format scores ────────────────────────────────────────────
    print("\n[step 5] computing format scores...")
    format_scores = compute_format_scores(fires, data["topk"], heur_applies)
    print(f"  non-zero format score: {(format_scores.max(axis=1) > 0).mean():.4f} of rows")

    # ── Step 6: evaluate policies × lambda grid × slices ────────────────
    print("\n[step 6] evaluating policies...")
    all_metric_rows = []
    best_lam_overall = 1.0   # track best lambda on gold_format for examples
    best_acc_gold_fmt = -1.0

    for lam in lambda_grid:
        policy = "base_candidate" if lam == 0 else "base_plus_format"
        for sname, smask in slices.items():
            row = compute_policy_metrics(
                data["topk"], data["lgt"], format_scores, data["gold"],
                extra["base_top1"], extra["gold_in_pool"], smask, lam)
            row["policy"] = policy
            row["lambda"] = lam
            row["slice"]  = sname
            all_metric_rows.append(row)

            if sname == "gold_format" and policy == "base_plus_format":
                pa = row.get("policy_acc", -1.0)
                if not np.isnan(pa) and pa > best_acc_gold_fmt:
                    best_acc_gold_fmt = pa
                    best_lam_overall  = lam

    print(f"  {len(all_metric_rows):,} metric rows  best_lam={best_lam_overall}")

    # ── Heuristic stats ──────────────────────────────────────────────────
    print("\n[step] computing heuristic stats...")
    heuristic_rows = compute_heuristic_stats(fires, data, extra, heur_applies)

    # ── Feature summary ──────────────────────────────────────────────────
    feature_rows = compute_feature_summary(state, slices)

    # ── Step 7: examples ─────────────────────────────────────────────────
    print("\n[step 7] collecting examples...")
    examples = collect_examples(data, format_scores, fires, extra, slices,
                                token_strings, args.max_examples,
                                best_lam_overall, args.seed)
    print(f"  helps={len(examples['helps'])}  hurts={len(examples['hurts'])}  "
          f"confusers={len(examples['confusers'])}")

    # ── Step 8: write outputs ────────────────────────────────────────────
    print("\n[step 8] writing outputs...")
    report_path = write_outputs(
        all_metric_rows, heuristic_rows, feature_rows, examples,
        slices, data, extra, args, N, lambda_grid, token_strings)

    elapsed = time.time() - t0
    print(f"\n{'='*64}")
    print(f" Format/Syntax Diagnostic V1 complete.  ({elapsed:.1f}s)")
    print(f" Report: {report_path}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
