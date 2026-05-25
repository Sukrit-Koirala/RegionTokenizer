#!/usr/bin/env python3
"""
audit_bucketA_attention_patterns.py — BucketA/H attention diagnostic audit.

Diagnostic only — NO training, NO model changes, NO gold leakage into forward pass.
Gold is used ONLY after model outputs exist, to label examples and compare attentions.

Answers:
  Q1. Does evidence for the gold candidate exist in attention / memory?
  Q2. Does the resolver attend to useful context but fail to use it?
  Q3. Or is the signal absent entirely?
  Q4. Are changed_to_gold examples different from missed_gold?
  Q5. Are changed_away examples caused by misleading context attention?

Usage:
  python scripts/audit_bucketA_attention_patterns.py \
    --run_dir  runs/txl_detail_resolver_v1/txlmem_selector_gate_v1 \
    --small_ckpt runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \
    --val_dir  runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val \
    --token_to_region runs/region_maps_128/token_to_region.json \
    --super_map runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \
    --output_dir runs/txl_detail_resolver_v1/txlmem_selector_gate_v1/bucketA_attention_audit \
    --max_examples_per_group 25
"""

import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Project root + import training module
# ─────────────────────────────────────────────────────────────────────────────

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

_TRAIN_SCRIPT = os.path.join(_PROJ_ROOT, "scripts", "train_txl_detail_resolver_v1.py")

def _import_train_module():
    spec = importlib.util.spec_from_file_location("_txl_train", _TRAIN_SCRIPT)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_T = _import_train_module()

TXLDetailMemory       = _T.TXLDetailMemory
LocalDetailSelectorGate = _T.LocalDetailSelectorGate
load_backbone         = _T.load_backbone
load_maps             = _T.load_maps
load_shards           = _T.load_shards
build_candidate_pool  = _T.build_candidate_pool
build_tok_arr         = _T.build_tok_arr
build_reg_arr         = _T.build_reg_arr
_apply_surgical       = _T._apply_surgical

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer  (GPT-2 / tiktoken, optional)
# ─────────────────────────────────────────────────────────────────────────────

_enc = None
try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
except Exception:
    pass

def _decode_tok(tid: int) -> str:
    if _enc is not None:
        try:
            return repr(_enc.decode([int(tid)]))[1:-1]   # strip outer quotes
        except Exception:
            pass
    return f"<{tid}>"

def _decode_ids(ids) -> str:
    if _enc is not None:
        try:
            clean = [int(i) for i in ids if 0 <= int(i) < 50257]
            return _enc.decode(clean)
        except Exception:
            pass
    return " ".join(_decode_tok(i) for i in ids)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)

def _ent(w_np: np.ndarray) -> float:
    """Shannon entropy (nats) of a probability vector."""
    w = w_np.astype(np.float64)
    w = w / (w.sum() + 1e-30)
    w = np.clip(w, 1e-30, None)
    return float(-np.sum(w * np.log(w)))

def _top_positions(w_np: np.ndarray, n: int = 10):
    """Return indices of top-n attention positions, sorted by weight descending."""
    idx = np.argsort(w_np)[::-1][:n]
    return idx.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Config / checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_run_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(path) as f:
        return json.load(f)


def find_checkpoint(run_dir: str) -> str:
    for name in ("best_txl_detail_resolver.pt",
                 "latest_txl_detail_resolver.pt",
                 "best_selector.pt",
                 "latest_selector.pt"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No checkpoint found in {run_dir}. "
        "Expected best_txl_detail_resolver.pt or latest_txl_detail_resolver.pt.")


def build_model(cfg: dict, tok_w, tok_arr_np, reg_arr_np,
                d_model, vocab_size, n_regions, n_supers,
                sr_enabled, unk_region, unk_super, device):
    model = LocalDetailSelectorGate(
        token_emb_weight = tok_w.to(device),
        tok_arr          = tok_arr_np,
        reg_arr          = reg_arr_np,
        d_model          = d_model,
        n_regions        = n_regions,
        n_supers         = n_supers,
        sr_enabled       = sr_enabled,
        unk_region       = unk_region,
        unk_super        = unk_super,
        top_k            = cfg.get("top_k", 256),
        pool_size        = cfg.get("candidate_pool_size", 32),
        memory_len       = cfg.get("memory_len", 128),
        resolver_dim     = cfg.get("resolver_dim", 256),
        hidden_dim       = cfg.get("hidden_dim", 512),
        attention_heads  = cfg.get("attention_heads", 4),
        dropout          = 0.0,          # eval mode — no dropout
        region_emb_dim   = cfg.get("region_emb_dim", 64),
        super_emb_dim    = cfg.get("super_emb_dim", 32),
        memory_backend   = cfg.get("memory_backend", "token"),
        mem_dim          = cfg.get("mem_dim", 256),
        txl_layers       = cfg.get("txl_layers", 2),
        txl_heads        = cfg.get("txl_heads", 4),
        txl_ff_dim       = cfg.get("txl_ff_dim", 1024),
        num_memory_slots = cfg.get("num_memory_slots", 16),
        pos_encoding     = cfg.get("pos_encoding", "alibi"),
    ).to(device)
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Bucket definitions
# ─────────────────────────────────────────────────────────────────────────────

BUCKETS = [
    "changed_to_gold",          # model corrected to gold
    "missed_gold",              # gold in pool, base wrong, model failed to fix
    "selected_wrong_confuser",  # gold in pool, model edited but picked wrong cand
    "changed_away",             # base was gold, model edited away (harm contrast)
    "high_gap_small",           # base_wrong, gold in pool, small logit gap
    "same_region_confuser",     # base_wrong, gold in pool, same fine region
]

# ─────────────────────────────────────────────────────────────────────────────
# Main collection pass
# ─────────────────────────────────────────────────────────────────────────────

def collect_examples(model, val_data, tok_arr_t, reg_arr_t,
                     unk_region, unk_super, sr_enabled, cfg, device,
                     md_half, max_per_group, seed=42):
    """
    Single pass over val data.  For each batch, run forward_eval, determine
    buckets, and keep up to max_per_group examples per bucket (first-seen).

    Gold is used ONLY after model outputs are computed, for labelling only.
    """
    rng = random.Random(seed)

    N   = val_data["h_prime"].shape[0]
    K   = val_data["topk_ids"].shape[1]
    BSZ = 256
    P   = cfg.get("candidate_pool_size", 32)
    mem_len = cfg.get("memory_len", 128)
    num_slots = cfg.get("num_memory_slots", 16) if cfg.get("memory_backend","token") == "txl" else 0
    vs  = tok_arr_t.shape[0]
    rlen = reg_arr_t.shape[0] - 1

    # small_logit_gap_thr: rows where gold is within this many logit units of base_top1
    small_gap_thr = cfg.get("_audit_small_gap_thr", 3.0)

    buckets: dict[str, list] = {b: [] for b in BUCKETS}
    counts: dict[str, int]   = {b: 0  for b in BUCKETS}

    model.eval()
    with torch.no_grad():
        for start in range(0, N, BSZ):
            # Check if all buckets are full
            if all(len(buckets[b]) >= max_per_group for b in BUCKETS):
                break

            end = min(start + BSZ, N)
            B   = end - start

            h_prime  = val_data["h_prime"][start:end].to(device)
            topk_ids = val_data["topk_ids"][start:end].to(device)
            topk_lgt = val_data["topk_lgt"][start:end].to(device)
            gold_t   = val_data["gold"][start:end].to(device)
            input_ids = val_data["ids"][start:end].to(device)

            # ── Build candidate pool (no gold forced in) ──────────────────────
            cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups = build_candidate_pool(
                topk_ids, topk_lgt, tok_arr_t, reg_arr_t,
                unk_region, unk_super, sr_enabled,
                pool_size=P, candidate_filter="top_rank", device=device)

            # ── Forward eval  (no gold involved) ─────────────────────────────
            sel_scores, edit_logit, attn_w, gate_cand_idx = model.forward_eval(
                h_prime, input_ids,
                topk_ids[:, 0],   # base_top1_ids
                topk_lgt,
                cand_ids, cand_lgts, cand_rnks, cand_regs, cand_sups)
            # attn_w: [B, P, M]  — candidate × memory cross-attention (head-averaged)

            # ── Post-forward  (gold allowed here for labelling) ───────────────
            ar         = torch.arange(B, device=device)
            gate_prob  = torch.sigmoid(edit_logit)                    # [B]
            action_edit = gate_prob >= 0.5                            # [B] bool

            selected_toks = cand_ids[ar, gate_cand_idx]               # [B]

            # Simulate surgical edit
            ref_lgts = _apply_surgical(
                topk_lgt.cpu(), topk_ids.cpu(),
                selected_toks.cpu(),
                action_edit.cpu(), md_half)
            ref_top1_idx = ref_lgts.argmax(1)
            ref_top1     = topk_ids.cpu()[torch.arange(B), ref_top1_idx]  # [B]

            final_top1 = torch.where(action_edit.cpu(),
                                     ref_top1,
                                     topk_ids.cpu()[:, 0])

            gold_cpu      = gold_t.cpu()
            base_top1_cpu = topk_ids.cpu()[:, 0]
            base_wrong    = (base_top1_cpu != gold_cpu)

            # Gold in candidate pool
            gold_in_pool_mask = (cand_ids.cpu() == gold_cpu.unsqueeze(1))  # [B,P]
            gold_in_pool      = gold_in_pool_mask.any(1)                   # [B]
            gold_pool_idx_arr = gold_in_pool_mask.long().argmax(1)         # [B]
            gold_pool_idx_arr = torch.where(
                gold_in_pool, gold_pool_idx_arr, torch.full_like(gold_pool_idx_arr, -1))

            # Gold logit gap in topk
            gold_in_topk_mask = (topk_ids.cpu() == gold_cpu.unsqueeze(1))
            gold_found_topk   = gold_in_topk_mask.any(1)
            gold_topk_pos     = gold_in_topk_mask.long().argmax(1)
            gold_logit        = topk_lgt.cpu()[ar.cpu(), gold_topk_pos]
            base_logit_gap_gold = topk_lgt.cpu()[:, 0] - gold_logit  # positive = base better

            # Region info for gold vs base_top1
            base_reg  = tok_arr_t.cpu()[base_top1_cpu.clamp(0, vs - 1)]
            gold_reg  = tok_arr_t.cpu()[gold_cpu.clamp(0, vs - 1)]
            same_reg  = ((base_reg == gold_reg) & (base_reg != unk_region))

            same_sup  = torch.zeros(B, dtype=torch.bool)
            if sr_enabled:
                base_sup = reg_arr_t.cpu()[base_reg.clamp(0, rlen)]
                gold_sup = reg_arr_t.cpu()[gold_reg.clamp(0, rlen)]
                same_sup = ((base_sup == gold_sup) & (base_sup != unk_super))
            else:
                base_sup = torch.zeros(B, dtype=torch.long)
                gold_sup = torch.zeros(B, dtype=torch.long)

            # ── Bucket assignment per row ─────────────────────────────────────
            for i in range(B):
                bw  = bool(base_wrong[i])
                gip = bool(gold_in_pool[i])
                ae  = bool(action_edit[i])
                ft  = int(final_top1[i])
                g   = int(gold_cpu[i])
                b   = int(base_top1_cpu[i])
                st  = int(selected_toks[i])

                ctg_i = bw and ae and ft == g
                caw_i = (not bw) and ae and ft != g   # base was gold, edited away
                sel_wrong_i = bw and gip and ae and st != g
                missed_i    = bw and gip and ft != g
                gap_small_i = bw and gip and bool(gold_found_topk[i]) and \
                              float(base_logit_gap_gold[i]) < small_gap_thr
                same_reg_i  = bw and gip and bool(same_reg[i])

                row_buckets = []
                if ctg_i:        row_buckets.append("changed_to_gold")
                if missed_i:     row_buckets.append("missed_gold")
                if sel_wrong_i:  row_buckets.append("selected_wrong_confuser")
                if caw_i:        row_buckets.append("changed_away")
                if gap_small_i:  row_buckets.append("high_gap_small")
                if same_reg_i:   row_buckets.append("same_region_confuser")

                for bucket in row_buckets:
                    counts[bucket] += 1
                    if len(buckets[bucket]) < max_per_group:
                        mem_ids_i = input_ids.cpu()[i, -mem_len:].tolist()
                        ex = {
                            "row_idx":          start + i,
                            "bucket":           bucket,
                            # ── Context ──────────────────────────────────────
                            "memory_ids":       mem_ids_i,
                            "memory_len":       mem_len,
                            "num_memory_slots": num_slots,
                            # ── Gold-free model state ─────────────────────────
                            "base_top1_id":     b,
                            "base_top1_logit":  float(topk_lgt.cpu()[i, 0]),
                            "gate_cand_idx":    int(gate_cand_idx[i]),
                            "gate_prob":        float(gate_prob[i]),
                            "action":           "EDIT" if ae else "NO_OP",
                            "sel_scores_np":    sel_scores.cpu()[i].numpy().copy(),  # [P]
                            "attn_w_np":        attn_w.cpu()[i].numpy().copy(),      # [P, M]
                            "cand_ids_np":      cand_ids.cpu()[i].numpy().copy(),
                            "cand_lgts_np":     cand_lgts.cpu()[i].numpy().copy(),
                            "cand_rnks_np":     cand_rnks.cpu()[i].numpy().copy(),
                            "cand_regs_np":     cand_regs.cpu()[i].numpy().copy(),
                            # ── Post-forward labels (gold allowed) ────────────
                            "gold_id":          g,
                            "gold_logit":       float(gold_logit[i]) if bool(gold_found_topk[i]) else float("nan"),
                            "gold_in_pool":     gip,
                            "gold_pool_idx":    int(gold_pool_idx_arr[i]),
                            "final_top1_id":    ft,
                            "selected_tok_id":  st,
                            "changed_to_gold":  ctg_i,
                            "changed_away":     caw_i,
                            "selected_is_gold": st == g,
                            "base_logit_gap_gold": float(base_logit_gap_gold[i])
                                                    if bool(gold_found_topk[i]) else float("nan"),
                            "gold_region":      int(gold_reg[i]),
                            "base_region":      int(base_reg[i]),
                            "same_region":      bool(same_reg[i]),
                            "same_superregion": bool(same_sup[i]),
                        }
                        buckets[bucket].append(ex)

    print(f"[collect] Total examples qualified per bucket:")
    for b in BUCKETS:
        print(f"  {b:<30} total={counts[b]:5d}  sampled={len(buckets[b]):3d}")

    return buckets

# ─────────────────────────────────────────────────────────────────────────────
# Per-example attention metrics
# ─────────────────────────────────────────────────────────────────────────────

def _attn_for_cand(attn_w_np, cand_idx, num_memory_slots, memory_len):
    """
    Return head-averaged attention weights [M] for a given candidate index.
    Also returns (real_mass, slot_mass) split.
    """
    w = attn_w_np[cand_idx]          # [M]
    real_mass = float(w[:memory_len].sum())
    slot_mass = float(w[memory_len:].sum()) if num_memory_slots > 0 else 0.0
    return w, real_mass, slot_mass


def _top_attended_tokens(w, memory_ids, num_memory_slots, memory_len, n=10):
    """
    Return list of (pos, token_id, token_str, weight, is_slot) for top-n positions.
    """
    top_idx = np.argsort(w)[::-1][:n]
    result = []
    for idx in top_idx:
        if idx < memory_len:
            tid = int(memory_ids[idx]) if idx < len(memory_ids) else 0
            tok_str = _decode_tok(tid)
            is_slot = False
        else:
            tid     = -1
            tok_str = f"<SLOT_{idx - memory_len}>"
            is_slot = True
        result.append({
            "pos":     int(idx),
            "tok_id":  tid,
            "tok_str": tok_str,
            "weight":  float(w[idx]),
            "is_slot": is_slot,
        })
    return result


def _overlap_top_k(w1, w2, k=10):
    """Jaccard-like overlap of top-k attended positions."""
    s1 = set(np.argsort(w1)[::-1][:k].tolist())
    s2 = set(np.argsort(w2)[::-1][:k].tolist())
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def enrich_example(ex: dict) -> dict:
    """
    Compute all attention metrics for a single example.
    Adds new keys starting with 'gold_' / 'selected_' / 'base_'.
    """
    attn_w_np   = ex["attn_w_np"]       # [P, M]
    cand_ids_np = ex["cand_ids_np"]
    sel_scores  = ex["sel_scores_np"]
    P, M        = attn_w_np.shape
    mem_len     = ex["memory_len"]
    num_slots   = ex["num_memory_slots"]
    mem_ids     = ex["memory_ids"]
    gold_id     = ex["gold_id"]
    gate_idx    = ex["gate_cand_idx"]

    # Find gold candidate index
    gold_cand_match = np.where(cand_ids_np == gold_id)[0]
    gold_cand_idx   = int(gold_cand_match[0]) if len(gold_cand_match) > 0 else -1
    ex["gold_cand_idx"] = gold_cand_idx

    # Selector probabilities
    sel_probs  = torch.softmax(torch.tensor(sel_scores), dim=0).numpy()
    ex["sel_prob_selected"] = float(sel_probs[gate_idx]) if gate_idx < len(sel_probs) else float("nan")
    ex["sel_prob_gold"]     = float(sel_probs[gold_cand_idx]) if gold_cand_idx >= 0 else float("nan")
    ex["sel_prob_gap"]      = ex["sel_prob_selected"] - ex["sel_prob_gold"] \
                              if (not math.isnan(ex["sel_prob_gold"])) else float("nan")

    # ── Selected candidate attention ──────────────────────────────────────────
    w_sel, sel_real, sel_slot = _attn_for_cand(attn_w_np, gate_idx, num_slots, mem_len)
    ex["selected_attention_entropy"]    = _ent(w_sel)
    ex["selected_attention_max"]        = float(w_sel.max())
    ex["selected_attention_mass_real"]  = sel_real
    ex["selected_attention_mass_slot"]  = sel_slot
    ex["selected_top_attended"]         = _top_attended_tokens(w_sel, mem_ids, num_slots, mem_len)

    # ── Gold candidate attention ──────────────────────────────────────────────
    if gold_cand_idx >= 0:
        w_gold, gold_real, gold_slot = _attn_for_cand(attn_w_np, gold_cand_idx, num_slots, mem_len)
        ex["gold_attention_entropy"]    = _ent(w_gold)
        ex["gold_attention_max"]        = float(w_gold.max())
        ex["gold_attention_mass_real"]  = gold_real
        ex["gold_attention_mass_slot"]  = gold_slot
        ex["gold_top_attended"]         = _top_attended_tokens(w_gold, mem_ids, num_slots, mem_len)
        # Overlap of gold vs selected top-10 positions
        ex["gold_selected_overlap_top10"] = _overlap_top_k(w_gold, w_sel, k=10)
    else:
        ex["gold_attention_entropy"]    = float("nan")
        ex["gold_attention_max"]        = float("nan")
        ex["gold_attention_mass_real"]  = float("nan")
        ex["gold_attention_mass_slot"]  = float("nan")
        ex["gold_top_attended"]         = []
        ex["gold_selected_overlap_top10"] = float("nan")

    # ── Base_top1 candidate — first pool slot is rank-2 (base is rank-1, not in pool) ─
    # The pool starts at topk rank 1 (0-indexed), i.e. cand_rnks[0]=1 is base_top1's
    # runner-up. There is no base_top1 candidate in the pool by design.
    # We skip base-top1 separate attention (not in pool).
    ex["base_top1_in_pool"] = False

    return ex


# ─────────────────────────────────────────────────────────────────────────────
# Automatic verdict
# ─────────────────────────────────────────────────────────────────────────────

def auto_verdict(ex: dict) -> str:
    """
    Simple heuristic verdict.
    SIGNAL_PRESENT     — gold candidate attends to plausible real tokens, selector ignores it
    SIGNAL_ABSENT      — gold attention is diffuse or dominated by slots
    MISLEADING_SIGNAL  — selected candidate attends strongly to different real tokens
    MEMORY_SLOT_DOMINATED — majority of attention on learned slots
    UNCLEAR
    """
    slot_mass   = ex.get("selected_attention_mass_slot", float("nan"))
    gold_max    = ex.get("gold_attention_max",          float("nan"))
    sel_max     = ex.get("selected_attention_max",      float("nan"))
    gold_ent    = ex.get("gold_attention_entropy",      float("nan"))
    sel_ent     = ex.get("selected_attention_entropy",  float("nan"))
    overlap     = ex.get("gold_selected_overlap_top10", float("nan"))

    if math.isnan(slot_mass):
        return "UNCLEAR"

    # Slot dominated
    if slot_mass > 0.5:
        return "MEMORY_SLOT_DOMINATED"

    # Both nan → can't assess
    if math.isnan(gold_max):
        return "UNCLEAR"

    # Gold has a clear peak on real tokens
    gold_sharp = gold_max > 0.10 or (not math.isnan(gold_ent) and gold_ent < 2.5)
    sel_sharp  = sel_max  > 0.10 or (not math.isnan(sel_ent)  and sel_ent  < 2.5)

    # Different evidence (low overlap)
    diff_evidence = not math.isnan(overlap) and overlap < 0.3

    if gold_sharp and diff_evidence:
        # Gold has signal but selector attended to different evidence
        if sel_sharp:
            return "MISLEADING_SIGNAL"
        return "SIGNAL_PRESENT"

    if not gold_sharp:
        return "SIGNAL_ABSENT"

    return "UNCLEAR"

# ─────────────────────────────────────────────────────────────────────────────
# CSV outputs
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "row_idx", "bucket",
    "gold_id", "gold_tok", "base_top1_id", "base_tok",
    "selected_tok_id", "selected_tok",
    "action", "changed_to_gold", "changed_away", "selected_is_gold",
    "gold_in_pool", "gold_pool_idx", "gold_cand_idx",
    "base_logit_gap_gold", "gate_prob",
    "sel_prob_gold", "sel_prob_selected", "sel_prob_gap",
    "gold_attention_entropy", "gold_attention_max",
    "gold_attention_mass_real", "gold_attention_mass_slot",
    "selected_attention_entropy", "selected_attention_max",
    "selected_attention_mass_real", "selected_attention_mass_slot",
    "gold_selected_overlap_top10",
    "same_region", "same_superregion",
    "gold_region", "base_region",
    "verdict",
]

def write_csv(examples: list, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for ex in examples:
            row = dict(ex)
            row["gold_tok"]      = _decode_tok(ex["gold_id"])
            row["base_tok"]      = _decode_tok(ex["base_top1_id"])
            row["selected_tok"]  = _decode_tok(ex["selected_tok_id"])
            row["verdict"]       = ex.get("verdict", "")
            for k in _CSV_FIELDS:
                v = row.get(k, "")
                if isinstance(v, float):
                    row[k] = "" if math.isnan(v) else f"{v:.6f}"
                elif isinstance(v, bool):
                    row[k] = str(v)
            w.writerow({k: row.get(k, "") for k in _CSV_FIELDS})
    print(f"[csv] {out_path}")


def write_group_stats_csv(all_examples: list, out_path: str):
    from collections import defaultdict
    stats = defaultdict(list)
    for ex in all_examples:
        stats[ex["bucket"]].append(ex)

    fields = [
        "bucket", "n",
        "mean_gold_attn_entropy", "mean_sel_attn_entropy",
        "mean_gold_attn_max",     "mean_sel_attn_max",
        "mean_gold_real_mass",    "mean_sel_real_mass",
        "mean_gold_slot_mass",    "mean_sel_slot_mass",
        "mean_overlap_top10",
        "mean_sel_prob_gold",     "mean_sel_prob_selected",
        "mean_sel_prob_gap",
        "mean_gate_prob",
        "mean_base_logit_gap_gold",
        "same_region_rate",       "same_superregion_rate",
        "verdict_SIGNAL_PRESENT",
        "verdict_SIGNAL_ABSENT",
        "verdict_MISLEADING_SIGNAL",
        "verdict_MEMORY_SLOT_DOMINATED",
        "verdict_UNCLEAR",
    ]

    def _mean(lst):
        clean = [v for v in lst if not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(clean)) if clean else float("nan")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for bucket in BUCKETS:
            exs = stats.get(bucket, [])
            n   = len(exs)
            if n == 0:
                w.writerow({"bucket": bucket, "n": 0})
                continue
            verd = [ex.get("verdict","UNCLEAR") for ex in exs]
            row = {
                "bucket": bucket,
                "n":      n,
                "mean_gold_attn_entropy":   _f(_mean([ex.get("gold_attention_entropy",float("nan")) for ex in exs])),
                "mean_sel_attn_entropy":    _f(_mean([ex.get("selected_attention_entropy",float("nan")) for ex in exs])),
                "mean_gold_attn_max":       _f(_mean([ex.get("gold_attention_max",float("nan")) for ex in exs])),
                "mean_sel_attn_max":        _f(_mean([ex.get("selected_attention_max",float("nan")) for ex in exs])),
                "mean_gold_real_mass":      _f(_mean([ex.get("gold_attention_mass_real",float("nan")) for ex in exs])),
                "mean_sel_real_mass":       _f(_mean([ex.get("selected_attention_mass_real",float("nan")) for ex in exs])),
                "mean_gold_slot_mass":      _f(_mean([ex.get("gold_attention_mass_slot",float("nan")) for ex in exs])),
                "mean_sel_slot_mass":       _f(_mean([ex.get("selected_attention_mass_slot",float("nan")) for ex in exs])),
                "mean_overlap_top10":       _f(_mean([ex.get("gold_selected_overlap_top10",float("nan")) for ex in exs])),
                "mean_sel_prob_gold":       _f(_mean([ex.get("sel_prob_gold",float("nan")) for ex in exs])),
                "mean_sel_prob_selected":   _f(_mean([ex.get("sel_prob_selected",float("nan")) for ex in exs])),
                "mean_sel_prob_gap":        _f(_mean([ex.get("sel_prob_gap",float("nan")) for ex in exs])),
                "mean_gate_prob":           _f(_mean([ex.get("gate_prob",float("nan")) for ex in exs])),
                "mean_base_logit_gap_gold": _f(_mean([ex.get("base_logit_gap_gold",float("nan")) for ex in exs])),
                "same_region_rate":         _f(sum(ex.get("same_region",False) for ex in exs) / n),
                "same_superregion_rate":    _f(sum(ex.get("same_superregion",False) for ex in exs) / n),
                "verdict_SIGNAL_PRESENT":      sum(v=="SIGNAL_PRESENT"      for v in verd),
                "verdict_SIGNAL_ABSENT":       sum(v=="SIGNAL_ABSENT"       for v in verd),
                "verdict_MISLEADING_SIGNAL":   sum(v=="MISLEADING_SIGNAL"   for v in verd),
                "verdict_MEMORY_SLOT_DOMINATED": sum(v=="MEMORY_SLOT_DOMINATED" for v in verd),
                "verdict_UNCLEAR":             sum(v=="UNCLEAR"             for v in verd),
            }
            w.writerow(row)
    print(f"[csv] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown example files
# ─────────────────────────────────────────────────────────────────────────────

def _attn_table(top_attended: list) -> str:
    lines = ["| pos | token | weight | slot? |",
             "|-----|-------|--------|-------|"]
    for e in top_attended:
        lines.append(
            f"| {e['pos']:<4} | `{e['tok_str']:<12}` | {e['weight']:.4f} | "
            f"{'✓' if e['is_slot'] else ''} |")
    return "\n".join(lines)


def _cand_table(ex: dict) -> str:
    cand_ids  = ex["cand_ids_np"]
    cand_lgts = ex["cand_lgts_np"]
    cand_regs = ex.get("cand_regs_np", np.zeros(len(cand_ids), dtype=np.int32))
    sel_probs = torch.softmax(torch.tensor(ex["sel_scores_np"]), dim=0).numpy()
    gold_id   = ex["gold_id"]
    gate_idx  = ex["gate_cand_idx"]
    P = len(cand_ids)

    lines = [
        "| rank | token | base_logit | sel_prob | gate? | region | is_gold | is_selected |",
        "|------|-------|------------|----------|-------|--------|---------|-------------|",
    ]
    for j in range(min(P, 16)):    # show at most 16 candidates
        tid    = int(cand_ids[j])
        tok    = _decode_tok(tid)
        lgt    = float(cand_lgts[j])
        prob   = float(sel_probs[j])
        reg    = int(cand_regs[j])
        is_g   = "✓" if tid == gold_id else ""
        is_s   = "✓" if j == gate_idx  else ""
        lines.append(
            f"| {j:<4} | `{tok:<10}` | {lgt:8.3f} | {prob:.4f} | "
            f"{'GATE' if j == gate_idx else '':<6} | {reg:<6} | "
            f"{is_g:<7} | {is_s:<11} |")
    return "\n".join(lines)


def write_md_examples(examples: list, bucket_name: str, out_path: str):
    lines = [f"# BucketA Attention Audit — {bucket_name}\n",
             f"Total examples: {len(examples)}\n",
             "---\n"]

    for idx, ex in enumerate(examples):
        ctx_tail   = _decode_ids(ex["memory_ids"])
        gold_str   = _decode_tok(ex["gold_id"])
        base_str   = _decode_tok(ex["base_top1_id"])
        sel_str    = _decode_tok(ex["selected_tok_id"])
        verdict    = ex.get("verdict", "UNCLEAR")
        gate_prob  = float(ex["gate_prob"])
        action     = ex["action"]

        lines.append(f"## Example {idx+1}  (row {ex['row_idx']})\n")
        lines.append(f"**Verdict:** `{verdict}`\n")
        lines.append(f"**Action:** `{action}`  gate_prob={gate_prob:.4f}  "
                     f"ctg={ex['changed_to_gold']}  caw={ex['changed_away']}\n")
        lines.append(f"**Gold token:** `{gold_str}` (id={ex['gold_id']}  "
                     f"region={ex['gold_region']})\n")
        lines.append(f"**Base top-1:**  `{base_str}` (id={ex['base_top1_id']}  "
                     f"logit={_f(ex['base_top1_logit'])}  "
                     f"region={ex['base_region']})\n")
        lines.append(f"**Selected:**    `{sel_str}` (id={ex['selected_tok_id']}  "
                     f"pool_idx={ex['gate_cand_idx']})\n")
        lines.append(f"**same_region:** {ex['same_region']}  "
                     f"**same_super:** {ex['same_superregion']}  "
                     f"**logit_gap_gold:** {_f(ex.get('base_logit_gap_gold', float('nan')))}\n")

        lines.append("\n### Context tail (last 128 tokens)\n")
        lines.append("```\n" + ctx_tail[-300:] + "\n```\n")

        lines.append("\n### Candidate pool\n")
        lines.append(_cand_table(ex) + "\n")

        lines.append("\n### Gold candidate — top attended memory positions\n")
        if ex.get("gold_top_attended"):
            lines.append(f"entropy={_f(ex['gold_attention_entropy'])}  "
                         f"max={_f(ex['gold_attention_max'])}  "
                         f"real_mass={_f(ex['gold_attention_mass_real'])}  "
                         f"slot_mass={_f(ex['gold_attention_mass_slot'])}\n")
            lines.append(_attn_table(ex["gold_top_attended"]) + "\n")
        else:
            lines.append("*Gold not in candidate pool.*\n")

        lines.append("\n### Selected candidate — top attended memory positions\n")
        lines.append(f"entropy={_f(ex['selected_attention_entropy'])}  "
                     f"max={_f(ex['selected_attention_max'])}  "
                     f"real_mass={_f(ex['selected_attention_mass_real'])}  "
                     f"slot_mass={_f(ex['selected_attention_mass_slot'])}\n")
        lines.append(_attn_table(ex["selected_top_attended"]) + "\n")

        if ex.get("gold_cand_idx", -1) >= 0:
            ov = ex.get("gold_selected_overlap_top10", float("nan"))
            lines.append(f"\n**Gold–Selected attention overlap (top-10 Jaccard):** {_f(ov)}\n")
            lines.append(f"**Selector prob (gold):** {_f(ex.get('sel_prob_gold'))}  "
                         f"**Selector prob (selected):** {_f(ex.get('sel_prob_selected'))}  "
                         f"**Gap:** {_f(ex.get('sel_prob_gap'))}\n")

        lines.append("\n---\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[md] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(all_examples: list, out_path: str, args, cfg: dict):
    from collections import defaultdict

    by_bucket = defaultdict(list)
    for ex in all_examples:
        by_bucket[ex["bucket"]].append(ex)

    def _mean(lst):
        clean = [v for v in lst if not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(clean)) if clean else float("nan")

    def _gstat(bucket, key):
        return _mean([ex.get(key, float("nan")) for ex in by_bucket.get(bucket, [])])

    def _vcount(bucket, verdict):
        return sum(ex.get("verdict","") == verdict for ex in by_bucket.get(bucket, []))

    lines = [
        "# BucketA/H Attention Pattern Audit Report\n",
        f"**Run dir:** `{args.run_dir}`\n",
        f"**Memory backend:** `{cfg.get('memory_backend','?')}`  "
        f"mem_dim={cfg.get('mem_dim','?')}  "
        f"txl_layers={cfg.get('txl_layers','?')}  "
        f"num_memory_slots={cfg.get('num_memory_slots','?')}\n",
        f"**Examples per group:** {args.max_examples_per_group}\n",
        "",
    ]

    # ── Bucket sizes ──────────────────────────────────────────────────────────
    lines.append("## 0. Example Counts\n")
    lines.append("| bucket | n |")
    lines.append("|--------|---|")
    for b in BUCKETS:
        lines.append(f"| {b} | {len(by_bucket.get(b,[]))} |")
    lines.append("")

    # ── Aggregate attention stats ─────────────────────────────────────────────
    lines.append("## 1. Attention Summary by Bucket\n")
    lines.append("| bucket | gold_ent | sel_ent | gold_max | sel_max | gold_real | sel_real | slot_mass | overlap | sel_prob_gap |")
    lines.append("|--------|----------|---------|----------|---------|-----------|----------|-----------|---------|--------------|")
    for b in BUCKETS:
        lines.append(
            f"| {b:<30} "
            f"| {_f(_gstat(b,'gold_attention_entropy'))} "
            f"| {_f(_gstat(b,'selected_attention_entropy'))} "
            f"| {_f(_gstat(b,'gold_attention_max'))} "
            f"| {_f(_gstat(b,'selected_attention_max'))} "
            f"| {_f(_gstat(b,'gold_attention_mass_real'))} "
            f"| {_f(_gstat(b,'selected_attention_mass_real'))} "
            f"| {_f(_gstat(b,'selected_attention_mass_slot'))} "
            f"| {_f(_gstat(b,'gold_selected_overlap_top10'))} "
            f"| {_f(_gstat(b,'sel_prob_gap'))} |"
        )
    lines.append("")

    # ── Verdict distribution ──────────────────────────────────────────────────
    lines.append("## 2. Verdict Distribution by Bucket\n")
    lines.append("| bucket | SIGNAL_PRESENT | SIGNAL_ABSENT | MISLEADING | SLOT_DOM | UNCLEAR |")
    lines.append("|--------|----------------|---------------|------------|----------|---------|")
    for b in BUCKETS:
        lines.append(
            f"| {b:<30} "
            f"| {_vcount(b,'SIGNAL_PRESENT'):<14} "
            f"| {_vcount(b,'SIGNAL_ABSENT'):<13} "
            f"| {_vcount(b,'MISLEADING_SIGNAL'):<10} "
            f"| {_vcount(b,'MEMORY_SLOT_DOMINATED'):<8} "
            f"| {_vcount(b,'UNCLEAR')} |"
        )
    lines.append("")

    # ── Q&A section ───────────────────────────────────────────────────────────
    lines.append("## 3. Diagnostic Questions\n")

    def _ans(bucket, q_key, q_text, good_key, threshold, higher_is_yes=True):
        v = _gstat(bucket, q_key)
        if math.isnan(v):
            ans = "**N/A** (no data)"
        elif higher_is_yes:
            ans = "**YES**" if v > threshold else "**NO**"
        else:
            ans = "**YES**" if v < threshold else "**NO**"
        return f"- {q_text}: {ans} ({q_key}={_f(v)})"

    miss = "missed_gold"
    ctg  = "changed_to_gold"
    caw  = "changed_away"

    lines.append("### Q1. In missed_gold, does gold candidate attend to meaningful real context tokens?")
    lines.append(_ans(miss, "gold_attention_max", "gold_attn_max > 0.10", "x", 0.10))
    lines.append(_ans(miss, "gold_attention_mass_real", "gold_real_mass > 0.5", "x", 0.5))
    lines.append("")

    lines.append("### Q2. Are missed_gold examples attention-diffuse (high entropy)?")
    lines.append(_ans(miss, "gold_attention_entropy", "gold_entropy > 3.0", "x", 3.0))
    lines.append(_ans(miss, "selected_attention_entropy", "sel_entropy > 3.0", "x", 3.0))
    lines.append("")

    lines.append("### Q3. Do changed_to_gold examples show sharper/lower-entropy attention?")
    ctg_ent  = _gstat(ctg,  "gold_attention_entropy")
    miss_ent = _gstat(miss, "gold_attention_entropy")
    if not (math.isnan(ctg_ent) or math.isnan(miss_ent)):
        sharper = ctg_ent < miss_ent - 0.2
        lines.append(f"- ctg gold_entropy={_f(ctg_ent)} vs missed_gold gold_entropy={_f(miss_ent)}: "
                     f"{'**YES** (sharper in ctg)' if sharper else '**NO** (similar or reversed)'}")
    else:
        lines.append("- **N/A** (missing data)")
    lines.append("")

    lines.append("### Q4. Are changed_away examples attending to misleading tokens?")
    n_mislead = _vcount(caw, "MISLEADING_SIGNAL")
    n_caw     = len(by_bucket.get(caw, []))
    lines.append(f"- MISLEADING_SIGNAL in changed_away: {n_mislead}/{n_caw} "
                 f"({'**YES**' if n_caw > 0 and n_mislead / max(n_caw,1) > 0.3 else '**UNCLEAR**'})")
    lines.append(_ans(caw, "gold_selected_overlap_top10", "low gold–selected overlap (< 0.3)",
                       "x", 0.3, higher_is_yes=False))
    lines.append("")

    lines.append("### Q5. Are learned memory slots dominating attention?")
    all_slot = _mean([ex.get("selected_attention_mass_slot", float("nan")) for ex in all_examples])
    lines.append(f"- Mean selected slot_mass across all examples: {_f(all_slot)}")
    n_slot_dom = sum(ex.get("verdict","") == "MEMORY_SLOT_DOMINATED" for ex in all_examples)
    lines.append(f"- MEMORY_SLOT_DOMINATED examples: {n_slot_dom}/{len(all_examples)}")
    dom_flag = "**YES** (slot-heavy)" if (not math.isnan(all_slot) and all_slot > 0.5) \
               else ("**NO**" if not math.isnan(all_slot) else "**N/A**")
    lines.append(f"- Verdict: {dom_flag}")
    lines.append("")

    lines.append("### Q6. Does attention contain a signal the selector fails to exploit?")
    sp = _gstat(miss, "sel_prob_gold")
    lines.append(f"- Mean selector_prob(gold) in missed_gold: {_f(sp)}")
    lines.append(f"- Mean selector_prob(selected) in missed_gold: {_f(_gstat(miss,'sel_prob_selected'))}")
    n_present = sum(ex.get("verdict","") == "SIGNAL_PRESENT"
                    for ex in by_bucket.get(miss, []))
    n_miss = len(by_bucket.get(miss, []))
    lines.append(f"- SIGNAL_PRESENT in missed_gold: {n_present}/{n_miss} "
                 f"({'**YES — signal exists but selector under-uses it**' if n_miss > 0 and n_present/max(n_miss,1) > 0.3 else '**UNCLEAR/NO**'})")
    lines.append("")

    lines.append("### Q7. Or is the signal absent from this memory representation?")
    n_absent = sum(ex.get("verdict","") == "SIGNAL_ABSENT"
                   for ex in by_bucket.get(miss, []))
    lines.append(f"- SIGNAL_ABSENT in missed_gold: {n_absent}/{n_miss}")
    if n_miss > 0:
        absent_rate = n_absent / n_miss
        if absent_rate > 0.5:
            lines.append("- **YES** — majority of missed_gold examples show diffuse/absent signal. "
                         "Memory representation may lack usable detail.")
        elif n_present / max(n_miss, 1) > 0.3:
            lines.append("- **NO** — signal is present; problem is in readout/gate.")
        else:
            lines.append("- **UNCLEAR** — mixed evidence.")
    lines.append("")

    # ── Final recommendation ──────────────────────────────────────────────────
    lines.append("## 4. Recommendation\n")

    n_miss_total = len(by_bucket.get(miss, []))
    n_present_miss = sum(ex.get("verdict","") == "SIGNAL_PRESENT"
                         for ex in by_bucket.get(miss, []))
    n_absent_miss  = sum(ex.get("verdict","") == "SIGNAL_ABSENT"
                         for ex in by_bucket.get(miss, []))
    slot_dom_rate  = n_slot_dom / max(len(all_examples), 1)
    miss_ent_val   = _gstat(miss, "gold_attention_entropy")
    ctg_ent_val    = _gstat(ctg,  "gold_attention_entropy")

    recs = []

    if slot_dom_rate > 0.4:
        recs.append(
            "**Memory slot dominance detected.** Learned slots absorb attention mass. "
            "Try reducing `num_memory_slots` to 0 or applying slot-attention regularisation.")

    if n_miss_total > 0 and n_present_miss / max(n_miss_total,1) > 0.3:
        recs.append(
            "**Signal present but ignored.** Gold candidate attends to meaningful tokens "
            "in >30% of missed_gold examples, but the selector fails to exploit this. "
            "Recommendations: (a) use per-head attention (not head-averaged) as a gate feature; "
            "(b) add attention sharpness / entropy as an explicit gate scalar; "
            "(c) train with a contrastive attention objective.")

    if n_miss_total > 0 and n_absent_miss / max(n_miss_total,1) > 0.5:
        recs.append(
            "**Signal absent in memory representation.** Majority of missed_gold examples "
            "show diffuse attention with no identifiable evidence peaks. "
            "Recommendations: (a) switch from frozen token-embedding memory to backbone "
            "hidden-state memory (h_prime at each context position); "
            "(b) try a deeper TXL encoder; "
            "(c) add cross-position attention between candidates and h_prime sequence.")

    if not math.isnan(ctg_ent_val) and not math.isnan(miss_ent_val) and ctg_ent_val < miss_ent_val - 0.3:
        recs.append(
            "**Attention sharpness predicts correction success.** changed_to_gold examples "
            "show lower entropy than missed_gold. Add attention entropy / max as a gate feature.")

    n_mislead_caw = _vcount(caw, "MISLEADING_SIGNAL")
    n_caw_total   = len(by_bucket.get(caw, []))
    if n_caw_total > 0 and n_mislead_caw / max(n_caw_total,1) > 0.3:
        recs.append(
            "**Misleading attention causes changed_away.** The model attends to tokens "
            "that support the wrong candidate when the base is already correct. "
            "Consider adding a 'base-correct confidence' scalar to the gate features.")

    if not recs:
        recs.append(
            "**Insufficient evidence from this sample.** Increase --max_examples_per_group "
            "for a stronger diagnostic.")

    for r in recs:
        lines.append(f"- {r}\n")

    lines.append("\n---\n")
    lines.append("*Generated by audit_bucketA_attention_patterns.py. "
                 "Treat as diagnostic evidence, not ground truth.*\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="BucketA/H Attention Audit")
    p.add_argument("--run_dir",           required=True)
    p.add_argument("--small_ckpt",        required=True)
    p.add_argument("--val_dir",           required=True)
    p.add_argument("--token_to_region",   required=True)
    p.add_argument("--super_map",         default=None)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--candidate_pool_size", type=int, default=32)
    p.add_argument("--memory_len",          type=int, default=128)
    p.add_argument("--max_examples_per_group", type=int, default=25)
    p.add_argument("--device",            default="cuda")
    p.add_argument("--amp",               action="store_true")
    p.add_argument("--seed",              type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[audit] device={device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load run config ───────────────────────────────────────────────────────
    print(f"\n[config] Loading from {args.run_dir}/config.json ...")
    cfg = load_run_config(args.run_dir)
    # Override pool_size / memory_len from CLI if given
    cfg["candidate_pool_size"] = args.candidate_pool_size
    cfg["memory_len"]          = args.memory_len
    md_half = 0.5 * cfg.get("margin_delta", 1.0)
    print(f"  memory_backend={cfg.get('memory_backend','?')}  "
          f"mem_dim={cfg.get('mem_dim','?')}  "
          f"txl_layers={cfg.get('txl_layers','?')}  "
          f"num_memory_slots={cfg.get('num_memory_slots','?')}  "
          f"md_half={md_half}")

    # ── Backbone ──────────────────────────────────────────────────────────────
    print("\n[backbone] Loading token embeddings...")
    tok_w, d_model, vocab_size = load_backbone(args.small_ckpt, device)
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    # ── Maps ──────────────────────────────────────────────────────────────────
    print("\n[maps] Loading region/super maps...")
    t2r, r2s, unk_region, unk_super, sr_enabled = load_maps(
        args.token_to_region, args.super_map)
    tok_arr_np = build_tok_arr(t2r, unk_region, vocab_size)
    reg_arr_np = build_reg_arr(r2s, unk_super, unk_region)
    tok_arr_t  = torch.from_numpy(tok_arr_np).long().to(device)
    reg_arr_t  = torch.from_numpy(reg_arr_np).long().to(device)
    n_regions  = unk_region
    n_supers   = unk_super if sr_enabled else 1
    print(f"  n_regions={n_regions}  n_supers={n_supers}  sr_enabled={sr_enabled}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n[model] Building model from config...")
    model = build_model(cfg, tok_w, tok_arr_np, reg_arr_np,
                        d_model, vocab_size, n_regions, n_supers,
                        sr_enabled, unk_region, unk_super, device)

    ckpt_path = find_checkpoint(args.run_dir)
    print(f"[model] Loading checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Loaded step={ck.get('step','?')}  params={n_p:,}")

    # ── Val data ──────────────────────────────────────────────────────────────
    print("\n[data] Loading val shards...")
    val_data = load_shards(args.val_dir, cfg.get("top_k", 256), "val", max_rows=None)

    # ── Collection pass ───────────────────────────────────────────────────────
    print(f"\n[collect] Scanning val set for BucketA/H examples "
          f"(max {args.max_examples_per_group}/group)...")
    with torch.no_grad():
        buckets = collect_examples(
            model, val_data, tok_arr_t, reg_arr_t,
            unk_region, unk_super, sr_enabled,
            cfg, device, md_half,
            max_per_group=args.max_examples_per_group,
            seed=args.seed)

    # ── Enrich (attention metrics) ────────────────────────────────────────────
    print("\n[enrich] Computing attention metrics and verdicts...")
    all_examples = []
    for b in BUCKETS:
        for ex in buckets[b]:
            enrich_example(ex)
            ex["verdict"] = auto_verdict(ex)
            all_examples.append(ex)
    print(f"  Total enriched: {len(all_examples)}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\n[write] Writing outputs...")

    write_csv(all_examples,
              os.path.join(args.output_dir, "bucketA_attention_summary.csv"))

    write_group_stats_csv(all_examples,
                          os.path.join(args.output_dir, "bucketA_attention_group_stats.csv"))

    for bucket in BUCKETS:
        exs = buckets[bucket]
        if not exs:
            continue
        fname = f"bucketA_examples_{bucket}.md"
        write_md_examples(exs, bucket,
                          os.path.join(args.output_dir, fname))

    write_report(all_examples,
                 os.path.join(args.output_dir, "bucketA_attention_report.md"),
                 args, cfg)

    print(f"\n[audit] Done. Outputs in: {args.output_dir}")
    print(f"  Key file: {args.output_dir}/bucketA_attention_report.md")


if __name__ == "__main__":
    main()
