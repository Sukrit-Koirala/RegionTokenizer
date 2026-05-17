#!/usr/bin/env python3
"""
Phase 2 — Clean masked-softmax baseline eval.

For each candidate policy, evaluates the masked-softmax upper bound on the
WikiText-103 val split using per_position.npz region metadata.

Metrics (per policy, global + per-split + per-type):
  coverage        : fraction of positions where gold token is in candidate set
  covered_nll     : mean NLL under masked softmax (covered positions only)
  fallback_nll    : mean NLL under full-vocab softmax (uncovered positions)
  strict_nll      : covered_nll for covered + log(V) penalty for uncovered
  mean_cand_toks  : mean number of candidate tokens per position

Outputs (--output_dir):
  masked_softmax_eval.csv
  masked_softmax_report.md

Usage:
    python scripts/eval_clean_masked_softmax.py \
        --small_ckpt    runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \
        --knn_run_dir   runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20 \
        --region_map    runs/region_maps_128/token_to_region.json \
        --super_map     runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \
        --output_dir    runs/path_refiner_clean/eval \
        --pareto_csv    runs/hier_policy_tuning/tuned_pareto_frontier.csv \
        --max_positions 300000 \
        --device        cuda
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import (
    TokenChunkDataset, load_region_map, build_inverse_map,
    load_wikitext, load_small_backbone_and_probe,
)

FULL_VOCAB   = 50257
N_FINE       = 128
MISS_PENALTY = float(np.log(FULL_VOCAB))

SPLIT_NAMES = {0: "core", 1: "medium", 2: "boundary", 3: "tight"}
TYPE_NAMES  = {0: "other", 1: "A", 2: "B", 3: "C"}


# ── Region / super helpers (inlined from dataset builder) ─────────────────────

def load_r2s(path: str, n_fine: int) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    r2s = np.full(n_fine, -1, dtype=np.int32)
    for k, v in d.items():
        fid = int(k)
        if 0 <= fid < n_fine:
            r2s[fid] = int(v)
    return r2s


def build_super_children(r2s, n_super):
    from collections import defaultdict
    inv = defaultdict(list)
    for fine, sup in enumerate(r2s):
        if sup >= 0:
            inv[int(sup)].append(fine)
    return dict(inv)


def aggregate_to_super(topk_reg, topk_prb, r2s, n_super):
    N, K = topk_reg.shape
    sp   = np.zeros((N, n_super), dtype=np.float32)
    valid = topk_reg >= 0
    fine  = topk_reg.astype(np.int32).clip(min=0)
    sups  = r2s[fine]
    sups[~valid] = -1
    valid2 = sups >= 0
    rows = np.repeat(np.arange(N), K)[valid2.ravel()]
    cols = sups.ravel()[valid2.ravel()]
    vals = topk_prb.astype(np.float32).ravel()[valid2.ravel()]
    np.add.at(sp, (rows, cols), vals)
    return sp


HGRID_MID_K_LOOKUP = {(60, 30): 8, (50, 25): 6, (40, 20): 6, (30, 10): 8}


def parse_policy(policy: str) -> Dict:
    import re
    p = policy.strip()
    if p.startswith("router_top"):
        return {"type": "router", "K": int(p[len("router_top"):])}
    if p.startswith("union_r"):
        rest = p[len("union_r"):]
        Kr, Km = int(rest.split("m")[0]), int(rest.split("m")[1])
        return {"type": "union", "Kr": Kr, "Km": Km}
    if p.startswith("hgrid_K"):
        cfg: Dict = {"type": "hgrid"}
        m = re.search(r"hgrid_K(\d+)", p);     cfg["K_super"]     = int(m.group(1)) if m else 24
        m = re.search(r"src(\w+?)_", p);       cfg["src"]         = m.group(1) if m else "combined"
        m = re.search(r"nts(\d+)", p);         cfg["n_top_super"] = int(m.group(1)) if m else 1
        m = re.search(r"bkr(\d+)", p);         cfg["base_kr"]     = int(m.group(1)) if m else 12
        m = re.search(r"_fb(\d+)", p);         cfg["fallback_k"]  = int(m.group(1)) if m else 24
        m = re.search(r"cm([\d.]+)", p);       cfg["cm_thr"]      = float(m.group(1)) if m else 0.15
        m = re.search(r"en([\d.]+)", p);       cfg["en_thr"]      = float(m.group(1)) if m else 1.25
        m = re.search(r"corec(\d+)m(\d+)", p)
        cfg["core_high"] = int(m.group(1)) / 100 if m else 0.60
        cfg["core_mid"]  = int(m.group(2)) / 100 if m else 0.30
        key = (int(cfg["core_high"] * 100), int(cfg["core_mid"] * 100))
        cfg["mid_k"] = HGRID_MID_K_LOOKUP.get(key, 8)
        return cfg
    raise ValueError(f"Unknown policy: {policy!r}")


def select_regions_hgrid(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
                          r_margin, m_entropy, r2s, super_children, cfg, K_stored):
    N        = len(r_margin)
    base_kr  = cfg["base_kr"];  mid_k = cfg["mid_k"];  fb_k = cfg["fallback_k"]
    cm_thr   = cfg["cm_thr"];   en_thr = cfg["en_thr"]
    n_ts     = cfg["n_top_super"]
    core_h   = cfg["core_high"]; core_m = cfg["core_mid"]
    src      = cfg["src"];       n_super = cfg["K_super"]

    sp_r = aggregate_to_super(r_topk_reg, r_topk_prb, r2s, n_super)
    sp_m = aggregate_to_super(m_topk_reg, m_topk_prb, r2s, n_super)
    sp   = 0.5 * sp_r + 0.5 * sp_m if src == "combined" else sp_m

    top2    = np.partition(-sp, kth=min(1, n_super - 1), axis=1)[:, :2] * -1
    cm_mg   = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    cm_en   = -(sp * np.log(sp + 1e-10)).sum(1)
    top_sups = np.argsort(-sp, axis=1)[:, :n_ts]

    rmg   = r_margin.astype(np.float32)
    K_out = max(base_kr, fb_k) + N_FINE
    sel   = np.full((N, K_out), -1, dtype=np.int16)

    for i in range(N):
        chosen: set = set()
        if rmg[i] >= core_h:
            for k in range(min(base_kr, K_stored)):
                r = int(r_topk_reg[i, k])
                if r >= 0: chosen.add(r)
        elif rmg[i] >= core_m:
            for k in range(min(mid_k, K_stored)):
                r = int(r_topk_reg[i, k])
                if r >= 0: chosen.add(r)
        else:
            if cm_mg[i] >= cm_thr and cm_en[i] <= en_thr:
                for k in range(min(base_kr, K_stored)):
                    r = int(r_topk_reg[i, k])
                    if r >= 0: chosen.add(r)
                for s in range(n_ts):
                    for fine in super_children.get(int(top_sups[i, s]), []):
                        chosen.add(fine)
            else:
                for k in range(min(fb_k, K_stored)):
                    r = int(r_topk_reg[i, k])
                    if r >= 0: chosen.add(r)
        ch = list(chosen)[:K_out]
        sel[i, :len(ch)] = ch
    return sel


def select_regions(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
                   r_margin, m_entropy, cfg, r2s, super_children) -> np.ndarray:
    K_stored = r_topk_reg.shape[1]
    if cfg["type"] == "router":
        K = min(cfg["K"], K_stored)
        return r_topk_reg[:, :K]
    if cfg["type"] == "union":
        Kr = min(cfg["Kr"], K_stored)
        Km = min(cfg["Km"], K_stored)
        return np.concatenate([r_topk_reg[:, :Kr], m_topk_reg[:, :Km]], axis=1)
    if cfg["type"] == "hgrid":
        return select_regions_hgrid(
            r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
            r_margin, m_entropy, r2s, super_children, cfg, K_stored)
    raise ValueError(cfg["type"])


# ── Pareto CSV reader ─────────────────────────────────────────────────────────

def load_pareto_policies(csv_path: str, voc_limits: List[float]) -> Dict[str, str]:
    """
    Returns {label: policy_str} for best policies under each vocab % budget.
    voc_limits: list of vocab % upper bounds (e.g. [8.0, 15.0]).
    """
    if not os.path.isfile(csv_path):
        return {}
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def _flt(r, k, default=None):
        try: return float(r[k])
        except (KeyError, ValueError, TypeError): return default

    result = {}
    for limit in voc_limits:
        candidates = [r for r in rows if _flt(r, "vocab_percent", 999) <= limit]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: _flt(r, "gold_region_coverage", 0.0))
        policy_str = best.get("policy", "")
        if policy_str:
            label = f"pareto_best_under_{int(limit)}pct"
            result[label] = policy_str
    return result


# ── Region token mask precomputation ─────────────────────────────────────────

def build_region_token_mask(inv_map: Dict[int, List[int]], n_fine: int,
                             device: torch.device) -> torch.Tensor:
    """Returns (n_fine, FULL_VOCAB) bool tensor."""
    mask = torch.zeros(n_fine, FULL_VOCAB, dtype=torch.bool)
    for r, toks in inv_map.items():
        if 0 <= r < n_fine:
            idx = torch.tensor(toks, dtype=torch.long)
            idx = idx[idx < FULL_VOCAB]
            mask[r, idx] = True
    return mask.to(device)


# ── Per-batch masked-softmax eval ─────────────────────────────────────────────

class PolicyAccumulator:
    """Accumulates NLL stats per (split, type) for one policy."""
    def __init__(self, label: str):
        self.label = label
        # keys: (split_id, type_id) → [covered_nll, fallback_nll, strict_nll, covered_n, total_n, cand_toks_sum]
        self.stats: Dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0, 0.0])

    def add(self, covered: np.ndarray, covered_nll: np.ndarray,
            fallback_nll: np.ndarray, cand_counts: np.ndarray,
            split: np.ndarray, type_arr: np.ndarray):
        N = len(covered)
        for i in range(N):
            key = (int(split[i]), int(type_arr[i]))
            s = self.stats[key]
            s[4] += 1
            s[5] += float(cand_counts[i])
            if covered[i]:
                s[0] += float(covered_nll[i])
                s[3] += 1
                s[2] += float(covered_nll[i])
            else:
                s[1] += float(fallback_nll[i])
                s[2] += MISS_PENALTY

    def summary(self) -> Dict[str, float]:
        total_n     = sum(s[4] for s in self.stats.values())
        cov_n       = sum(s[3] for s in self.stats.values())
        uncov_n     = total_n - cov_n
        sum_cov_nll = sum(s[0] for s in self.stats.values())
        sum_fb_nll  = sum(s[1] for s in self.stats.values())
        row = {
            "policy":         self.label,
            "total_n":        total_n,
            "coverage":       cov_n / max(total_n, 1),
            # NLL under masked softmax (covered positions only)
            "covered_nll":    sum_cov_nll / max(cov_n, 1),
            # Full-LM NLL for uncovered positions (not a penalty — real model NLL)
            "fallback_nll":   sum_fb_nll  / max(uncov_n, 1),
            # Global: masked CE for covered, full-LM CE for uncovered (user formula)
            "mixed_nll":      (sum_cov_nll + sum_fb_nll) / max(total_n, 1),
            # Strict upper bound: masked CE for covered, log(V) penalty for uncovered
            "strict_nll":     sum(s[2] for s in self.stats.values()) / max(total_n, 1),
            "mean_cand_toks": sum(s[5] for s in self.stats.values()) / max(total_n, 1),
        }
        # Per split
        for sid, sname in SPLIT_NAMES.items():
            sub = [s for k, s in self.stats.items() if k[0] == sid]
            tn  = sum(s[4] for s in sub)
            cn  = sum(s[3] for s in sub)
            row[f"cov_{sname}"] = cn / max(tn, 1)
            row[f"cnll_{sname}"] = sum(s[0] for s in sub) / max(cn, 1)
        # Per type
        for tid, tname in TYPE_NAMES.items():
            sub = [s for k, s in self.stats.items() if k[1] == tid]
            tn  = sum(s[4] for s in sub)
            cn  = sum(s[3] for s in sub)
            row[f"cov_type{tname}"] = cn / max(tn, 1)
        return row


def eval_policy_batch(logits: torch.Tensor, gold_tok: torch.Tensor,
                      sel_fine: np.ndarray, rtm: torch.Tensor,
                      coarse_map_t: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    logits:       (N, V) float32 GPU
    gold_tok:     (N,)   int64   GPU
    sel_fine:     (N, K) int16   numpy  (-1 = pad)
    rtm:          (n_fine, V) bool GPU  region-token mask
    coarse_map_t: (V,)   int64 GPU  fine region of each vocab token

    Returns: covered, covered_nll, fallback_nll, cand_counts — all numpy (N,)
    """
    N = logits.shape[0]
    sel_t = torch.from_numpy(sel_fine.astype(np.int64)).to(logits.device)  # (N, K)
    valid_sel = sel_t >= 0  # (N, K)

    # Coverage check: is gold_fine in selected regions?
    gold_fine = coarse_map_t[gold_tok]             # (N,)  int64
    covered_t = (gold_fine.unsqueeze(1) == sel_t).any(1)  # (N,) bool

    # Candidate count per position: union of tokens in selected regions
    sel_clamp = sel_t.clamp(min=0)  # (N, K)
    reg_masks = rtm[sel_clamp]      # (N, K, V) bool
    reg_masks = reg_masks & valid_sel.unsqueeze(-1)
    token_mask = reg_masks.any(1)   # (N, V) bool
    cand_counts = token_mask.sum(1).cpu().numpy().astype(np.float32)

    # Masked softmax NLL for covered positions
    covered_np = covered_t.cpu().numpy()
    if covered_t.any():
        cov_idx = covered_t.nonzero(as_tuple=True)[0]
        ml  = logits[cov_idx].clone()
        ml[~token_mask[cov_idx]] = -1e9
        lp  = F.log_softmax(ml, dim=-1)
        gt  = gold_tok[cov_idx]
        cnll = -lp[torch.arange(len(cov_idx), device=lp.device), gt].cpu().numpy()
    else:
        cnll = np.empty(0, dtype=np.float32)

    # Full-vocab NLL for uncovered positions
    uncovered_t = ~covered_t
    if uncovered_t.any():
        unc_idx = uncovered_t.nonzero(as_tuple=True)[0]
        lp_full  = F.log_softmax(logits[unc_idx], dim=-1)
        gt_unc   = gold_tok[unc_idx]
        fnll = -lp_full[torch.arange(len(unc_idx), device=lp_full.device), gt_unc].cpu().numpy()
    else:
        fnll = np.empty(0, dtype=np.float32)

    return covered_np, cnll, fnll, cand_counts


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load backbone
    print(f"[eval] loading checkpoint: {args.small_ckpt}")
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Region maps
    coarse_map, n_coarse = load_region_map(args.region_map, vocab_size)
    if isinstance(coarse_map, torch.Tensor):
        coarse_map = coarse_map.numpy()
    inv_map = build_inverse_map(coarse_map, n_coarse)
    coarse_map_t = torch.from_numpy(coarse_map.astype(np.int64)).to(device)

    # Super map
    r2s: Optional[np.ndarray] = None
    super_children = None
    if args.super_map and os.path.isfile(args.super_map):
        r2s = load_r2s(args.super_map, N_FINE)
        n_super = int(r2s.max()) + 1
        super_children = build_super_children(r2s, n_super)
        print(f"[eval] super map loaded  n_super={n_super}")

    # per_position.npz
    for fname in ("per_position.npz", "per_position_topk.npz"):
        pp_path = os.path.join(args.knn_run_dir, fname)
        if os.path.isfile(pp_path):
            break
    else:
        raise RuntimeError(f"per_position.npz not found in {args.knn_run_dir}")
    print(f"[eval] loading {pp_path}")
    pp = dict(np.load(pp_path))
    has_type = "type" in pp
    N_pp = len(pp["gold_region"])
    print(f"[eval] per_position: N={N_pp:,}  has_type={has_type}")

    # Region-token mask tensor (n_fine, V) on GPU
    rtm = build_region_token_mask(inv_map, n_coarse, device)

    # Build policy list
    base_policies = ["router_top8", "router_top12", "router_top16",
                     "hgrid_K24_srccombined_nts1_bkr12_fb24_cm0.15_en1.25_corec60m30",
                     "union_r16m4", "union_r12m4"]
    if args.pareto_csv:
        pareto = load_pareto_policies(args.pareto_csv, [8.0, 15.0])
        for label, pol_str in pareto.items():
            base_policies.append(pol_str)
            print(f"[eval] pareto policy {label}: {pol_str}")

    policies = []
    for pol_str in base_policies:
        try:
            cfg = parse_policy(pol_str)
            policies.append((pol_str, cfg))
        except Exception as e:
            print(f"[eval] WARNING: skipping policy {pol_str!r}: {e}")

    print(f"[eval] evaluating {len(policies)} policies")
    accumulators = {pol: PolicyAccumulator(pol) for pol, _ in policies}

    # Val data loader
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    val_tokens = load_wikitext(args.dataset, tokenizer, "validation")
    safe_seq   = min(args.seq_len, backbone.pos_emb.num_embeddings)
    val_ds     = TokenChunkDataset(val_tokens, safe_seq)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    pp_offset = 0
    total_pos = 0
    t0 = time.time()

    for bi, batch in enumerate(val_loader):
        if pp_offset >= min(N_pp, args.max_positions):
            break

        batch = batch.to(device)
        src   = batch[:, :-1]
        tgt   = batch[:, 1:]
        if src.size(1) > safe_seq:
            src = src[:, :safe_seq]; tgt = tgt[:, :safe_seq]

        gold_regions_flat = coarse_map[tgt.cpu().reshape(-1).numpy()]  # numpy
        valid_np = (gold_regions_flat >= 0)                            # numpy bool
        n_valid  = int(valid_np.sum())
        if n_valid == 0:
            continue

        # Cap at remaining per_position.npz entries
        remaining = min(N_pp, args.max_positions) - pp_offset
        if n_valid > remaining:
            cumsum = np.cumsum(valid_np)
            cutoff = int(np.searchsorted(cumsum, remaining + 1))
            valid_np[cutoff:] = False
            n_valid = int(valid_np.sum())
        if n_valid == 0:
            break

        valid_t = torch.from_numpy(valid_np).to(device)

        with torch.no_grad():
            out = backbone(src)
            lm_logits_bt = out[0] if isinstance(out, (tuple, list)) else out
            lm_flat = lm_logits_bt.reshape(-1, lm_logits_bt.size(-1)).float()
            logits  = lm_flat[valid_t]                   # (n_valid, V)
            gold    = tgt.reshape(-1)[valid_t].long()    # (n_valid,)

        sl = slice(pp_offset, pp_offset + n_valid)
        r_topk_reg = pp["router_topk_regions"][sl]
        r_topk_prb = pp["router_topk_probs"][sl]
        m_topk_reg = pp["mem_topk_regions"][sl]
        m_topk_prb = pp["mem_topk_probs"][sl]
        r_margin   = pp["router_margin"][sl]
        m_entropy  = pp.get("mem_entropy",  np.zeros(N_pp, np.float16))[sl]
        split_c    = pp["split"][sl]
        type_c     = pp["type"][sl] if has_type else np.zeros(n_valid, np.uint8)

        for pol_str, cfg in policies:
            sel = select_regions(r_topk_reg, r_topk_prb, m_topk_reg, m_topk_prb,
                                 r_margin, m_entropy, cfg, r2s, super_children)
            cov, cnll, fnll, cands = eval_policy_batch(
                logits, gold, sel, rtm, coarse_map_t)

            acc = accumulators[pol_str]
            # Build full N-length NLL arrays (zeros for the "other" split)
            cnll_full = np.zeros(n_valid, np.float32)
            fnll_full = np.zeros(n_valid, np.float32)
            cov_cursor = 0; unc_cursor = 0
            for i in range(n_valid):
                if cov[i]:
                    cnll_full[i] = cnll[cov_cursor]; cov_cursor += 1
                else:
                    fnll_full[i] = fnll[unc_cursor]; unc_cursor += 1
            acc.add(cov, cnll_full, fnll_full, cands, split_c, type_c)

        pp_offset += n_valid
        total_pos += n_valid

        if bi % 20 == 0:
            print(f"  batch {bi:4d}  pos={total_pos:,}  t={time.time()-t0:.0f}s")

    print(f"[eval] done.  total_pos={total_pos:,}  t={time.time()-t0:.1f}s")

    # Summarise
    rows = [acc.summary() for acc in accumulators.values()]
    rows.sort(key=lambda r: r.get("strict_nll", 999))

    # Write CSV
    csv_path = os.path.join(args.output_dir, "masked_softmax_eval.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[eval] wrote {csv_path}")

    # Write markdown report
    md_path = os.path.join(args.output_dir, "masked_softmax_report.md")
    with open(md_path, "w") as f:
        f.write("# Masked-Softmax Baseline Eval\n\n")
        f.write(f"Positions evaluated: {total_pos:,}  \n")
        f.write(f"Policies evaluated: {len(rows)}  \n\n")
        f.write("## Global Summary\n\n")
        f.write("covered_nll = masked softmax NLL (covered only)  \n")
        f.write("fallback_nll = full-LM NLL for uncovered positions (real model NLL, not penalty)  \n")
        f.write("mixed_nll = coverage×covered_nll + (1-coverage)×fallback_nll  [primary global metric]  \n")
        f.write("strict_nll = same but uncovered penalised at log(V)=10.82  \n\n")
        f.write("| policy | coverage | covered_nll | fallback_nll | mixed_nll | strict_nll | mean_cands |\n")
        f.write("|--------|----------|-------------|--------------|-----------|------------|------------|\n")
        for r in rows:
            f.write(f"| {r['policy'][:55]} "
                    f"| {r['coverage']:.4f} "
                    f"| {r['covered_nll']:.4f} "
                    f"| {r['fallback_nll']:.4f} "
                    f"| {r['mixed_nll']:.4f} "
                    f"| {r['strict_nll']:.4f} "
                    f"| {r['mean_cand_toks']:.0f} |\n")
        f.write("\n## Per-Split Breakdown (coverage)\n\n")
        f.write("| policy | core_cov | medium_cov | boundary_cov | tight_cov |\n")
        f.write("|--------|----------|------------|--------------|----------|\n")
        for r in rows:
            f.write(f"| {r['policy'][:60]} "
                    f"| {r.get('cov_core', 0):.4f} "
                    f"| {r.get('cov_medium', 0):.4f} "
                    f"| {r.get('cov_boundary', 0):.4f} "
                    f"| {r.get('cov_tight', 0):.4f} |\n")
    print(f"[eval] wrote {md_path}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",    required=True)
    p.add_argument("--knn_run_dir",   required=True)
    p.add_argument("--region_map",    required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--super_map",     default=None)
    p.add_argument("--pareto_csv",    default=None)
    p.add_argument("--dataset",       default="wikitext-103-raw-v1")
    p.add_argument("--seq_len",       type=int,   default=128)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--max_positions", type=int,   default=300_000)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
