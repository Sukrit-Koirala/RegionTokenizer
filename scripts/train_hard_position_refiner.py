#!/usr/bin/env python3
"""
Hard-position-only path refiner experiment.

Trains a Variant-C RicherMLPRefiner on a filtered subset of positions (boundary,
tight boundary, type A/B, router top-k miss, etc.).  At eval time, applies a
gated global eval: delta=0 outside the gate, model scores inside.

All canonical full-val evals assert fingerprint/counts/coverage vs baseline.

Robust init: residual_scale=1.0 + zero final MLP layer → delta=0 at step 0
while gradients flow through inner layers from step 1 onward.

Filters (from shard metadata):
  boundary             — split in {2, 3}
  tight_boundary       — split == 3
  type_A               — type_arr == 1
  type_B               — type_arr == 2
  type_C               — type_arr == 3
  type_A_or_B_resolvable — type_arr in {1, 2}
  router_top8_miss     — gold_region not in router_topk_reg[:, :8]
  router_top4_miss     — gold_region not in router_topk_reg[:, :4]
  router_low_margin    — router_margin < margin_thresh
  hard_union_small     — type_A | tight_boundary | router_top8_miss
  hard_union_medium    — type_A | type_B | boundary | router_top8_miss
  hard_union           — boundary | type_A | type_B | router_low_margin | router_top8_miss
  all                  — everything (no filtering)

Output:
  <output_dir>/best_refiner.pt
  <output_dir>/last_refiner.pt
  <output_dir>/train_log.csv
  <output_dir>/local_subset_eval.csv   — written every eval step
  <output_dir>/best_metrics.json
  <output_dir>/filter_audit.json       — written at startup
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.train_clean_path_refiner import (
    RicherMLPRefiner,
    canonical_eval_refiner,
    check_baseline_match,
    check_init_identity,
    _aggregate_eval_stats,
    load_r2s,
    make_infinite,
    compute_loss,
    SPLIT_NAMES,
    TYPE_NAMES,
)

# ── Filter registry ───────────────────────────────────────────────────────────

EVAL_SUBSETS = [
    "all",
    "boundary",
    "tight_boundary",
    "type_A",
    "type_B",
    "type_C",
    "type_A_or_B_resolvable",
    "router_top8_miss",
    "router_top4_miss",
    "router_low_margin",
    "hard_union_small",
    "hard_union_medium",
    "hard_union",
]

TRAIN_FILTER_CHOICES = [
    "all",
    "boundary",
    "tight_boundary",
    "type_A",
    "type_B",
    "type_A_or_B_resolvable",
    "router_top8_miss",
    "router_low_margin",
    "hard_union_small",
    "hard_union_medium",
    "hard_union",
]


def _top8_miss_mask(shard: Dict) -> torch.Tensor:
    gold_reg = shard["gold_region"].long()              # (N,)
    topk_reg = shard["router_topk_reg"].long()[:, :8]  # (N, 8)
    gold_exp = gold_reg.unsqueeze(1).expand_as(topk_reg)
    in_top8  = ((topk_reg == gold_exp) & (topk_reg >= 0)).any(dim=1)
    return ~in_top8


def _top4_miss_mask(shard: Dict) -> torch.Tensor:
    gold_reg = shard["gold_region"].long()              # (N,)
    topk_reg = shard["router_topk_reg"].long()[:, :4]  # (N, 4)
    gold_exp = gold_reg.unsqueeze(1).expand_as(topk_reg)
    in_top4  = ((topk_reg == gold_exp) & (topk_reg >= 0)).any(dim=1)
    return ~in_top4


def compute_filter_mask(
    shard: Dict,
    filter_name: str,
    *,
    margin_thresh: float = 0.1,
    entropy_thresh: float = 2.0,
) -> torch.Tensor:
    """Return a (N,) bool tensor of positions that pass filter_name."""
    split    = shard["split"].long()
    type_arr = (shard["type_arr"].long() if "type_arr" in shard
                else torch.zeros(len(split), dtype=torch.long))

    if filter_name == "all":
        return torch.ones(len(split), dtype=torch.bool)
    if filter_name == "boundary":
        return (split == 2) | (split == 3)
    if filter_name == "tight_boundary":
        return split == 3
    if filter_name == "type_A":
        return type_arr == 1
    if filter_name == "type_B":
        return type_arr == 2
    if filter_name == "type_C":
        return type_arr == 3
    if filter_name == "type_A_or_B_resolvable":
        return (type_arr == 1) | (type_arr == 2)
    if filter_name == "router_top8_miss":
        return _top8_miss_mask(shard)
    if filter_name == "router_top4_miss":
        return _top4_miss_mask(shard)
    if filter_name == "router_low_margin":
        return shard["router_margin"].float() < margin_thresh
    if filter_name == "router_high_entropy":
        if "router_entropy" not in shard:
            raise KeyError("router_entropy not in shard")
        return shard["router_entropy"].float() > entropy_thresh
    if filter_name == "hard_union_small":
        return (split == 3) | (type_arr == 1) | _top8_miss_mask(shard)
    if filter_name == "hard_union_medium":
        boundary = (split == 2) | (split == 3)
        type_ab  = (type_arr == 1) | (type_arr == 2)
        return boundary | type_ab | _top8_miss_mask(shard)
    if filter_name == "hard_union":
        boundary   = (split == 2) | (split == 3)
        type_ab    = (type_arr == 1) | (type_arr == 2)
        low_margin = shard["router_margin"].float() < margin_thresh
        return boundary | type_ab | low_margin | _top8_miss_mask(shard)
    raise ValueError(f"Unknown filter: {filter_name!r}")


# ── Filtered dataset ──────────────────────────────────────────────────────────

class FilteredShardStreamDataset(IterableDataset):
    """Streams only positions that pass filter_name."""

    def __init__(self, shard_dir: str, r2s_np: np.ndarray,
                 filter_name: str, filter_kwargs: Optional[Dict] = None,
                 shuffle: bool = True) -> None:
        paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
        if not paths:
            raise RuntimeError(f"No shard_*.pt files in {shard_dir}")
        self.paths        = paths
        self.r2s          = r2s_np
        self.filter_name  = filter_name
        self.filter_kwargs = filter_kwargs or {}
        self.shuffle      = shuffle
        print(f"[FilteredDataset] filter={filter_name!r}  {len(paths)} shards")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        paths = list(self.paths)
        if self.shuffle:
            random.shuffle(paths)
        for path in paths:
            shard   = torch.load(path, map_location="cpu", weights_only=True)
            N       = len(shard["covered"])
            try:
                mask = compute_filter_mask(shard, self.filter_name, **self.filter_kwargs)
            except (KeyError, ValueError) as e:
                print(f"  WARNING: skipping shard {path}: {e}")
                continue
            idxs = torch.where(mask)[0].tolist()
            if not idxs:
                continue
            if self.shuffle:
                random.shuffle(idxs)

            cf_np      = shard["cand_fine"].numpy().astype(np.int64).clip(min=0)
            cs_np      = self.r2s[cf_np].astype(np.int64)
            cs_np[shard["cand_fine"].numpy() < 0] = 0
            cand_super = torch.from_numpy(cs_np)
            cand_mask  = (shard["cand_tok"] >= 0)
            has_t      = "type_arr" in shard

            for i in idxs:
                yield {
                    "h_prime":    shard["h_prime"][i].float(),
                    "cand_tok":   shard["cand_tok"][i].long(),
                    "cand_fine":  shard["cand_fine"][i].long(),
                    "cand_super": cand_super[i],
                    "cand_mask":  cand_mask[i],
                    "gold_idx":   shard["gold_cand_idx"][i].long(),
                    "covered":    shard["covered"][i].bool(),
                    "r_topk_reg": shard["router_topk_reg"][i].long(),
                    "r_topk_prb": shard["router_topk_prb"][i].float(),
                    "m_topk_reg": shard["mem_topk_reg"][i].long(),
                    "m_topk_prb": shard["mem_topk_prb"][i].float(),
                    "r_margin":   shard["router_margin"][i].float(),
                    "m_margin":   shard["mem_margin"][i].float(),
                    "split":      shard["split"][i].long(),
                    "type_arr":   (shard["type_arr"][i].long() if has_t
                                   else torch.zeros(1, dtype=torch.long).squeeze()),
                }


# ── Filter audit ─────────────────────────────────────────────────────────────

def run_filter_audit(
    train_dir: str,
    val_dir: str,
    filter_kwargs: Dict,
    output_dir: str,
) -> Dict:
    """Scan shards and report how many positions each filter selects."""
    print("\n[audit] Filter statistics:")
    audit = {}
    for split_name, shard_dir in [("train", train_dir), ("val", val_dir)]:
        if not shard_dir:
            continue
        paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
        if not paths:
            print(f"  [{split_name}] no shards")
            continue
        total_n   = 0
        total_cov = 0
        fcounts    = {f: 0 for f in EVAL_SUBSETS}
        fcov       = {f: 0 for f in EVAL_SUBSETS}

        for path in paths:
            shard   = torch.load(path, map_location="cpu", weights_only=True)
            covered = shard["covered"].bool()
            N       = len(covered)
            total_n   += N
            total_cov += int(covered.sum())
            for fname in EVAL_SUBSETS:
                try:
                    m = compute_filter_mask(shard, fname, **filter_kwargs)
                    fcounts[fname] += int(m.sum())
                    fcov[fname]    += int((m & covered).sum())
                except (KeyError, ValueError):
                    pass

        print(f"\n  [{split_name}] total={total_n:,}  covered={total_cov:,}")
        split_stats = {}
        for fname in EVAL_SUBSETS:
            n  = fcounts[fname]
            nc = fcov[fname]
            print(f"    {fname:30s}  n={n:8,}  ({n/max(total_n,1)*100:5.1f}%)  "
                  f"n_cov={nc:7,}  cov_rate_in_filter={nc/max(n,1)*100:5.1f}%")
            split_stats[fname] = {"n": n, "n_cov": nc, "total_n": total_n, "total_cov": total_cov}
        audit[split_name] = split_stats

    os.makedirs(output_dir, exist_ok=True)
    audit_path = os.path.join(output_dir, "filter_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  [audit] saved → {audit_path}")
    return audit


# ── Gated global eval ─────────────────────────────────────────────────────────

@torch.no_grad()
def gated_global_eval(
    model,
    val_dir: str,
    tok_emb_w: torch.Tensor,
    r2s_np: np.ndarray,
    device,
    gate_filter_name: str,
    filter_kwargs: Dict,
    official_baseline: Optional[Dict] = None,
    fail_on_mismatch: bool = False,
    variant_tag: str = "",
    eval_batch_size: int = 64,
) -> Dict:
    """
    Single pass over full val set with gating.

    gated_scores[b] = model_scores[b] if gate[b] else base_scores[b]
    NLL computed using gated_scores over all covered positions.
    Fingerprint computed identically to canonical_eval_refiner → must match baseline.
    """
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt in {val_dir}")

    emb_w = tok_emb_w.float().to(device)
    model.eval()

    total_n = total_cov = 0
    sum_cand_counts = sum_gold_idx_cov = sum_gold_tok = 0

    gated_ce  = gated_nc = 0
    ig_m_ce   = ig_b_ce  = ig_nc = ig_nt = 0   # inside gate
    og_ce     = og_nc    = 0                    # outside gate

    for path in paths:
        shard   = torch.load(path, map_location="cpu", weights_only=True)
        N       = len(shard["covered"])
        has_gtk = "gold_token" in shard

        try:
            gate_shard = compute_filter_mask(shard, gate_filter_name, **filter_kwargs)
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"gated_global_eval gate mask failed: {e}") from e

        cf_np      = shard["cand_fine"].numpy().astype(np.int64).clip(min=0)
        cs_np      = r2s_np[cf_np].astype(np.int64)
        cs_np[shard["cand_fine"].numpy() < 0] = 0
        cs_shard   = torch.from_numpy(cs_np)

        for start in range(0, N, eval_batch_size):
            end     = min(start + eval_batch_size, N)
            B       = end - start

            h       = shard["h_prime"][start:end].float().to(device)
            ct      = shard["cand_tok"][start:end].long().to(device)
            cf      = shard["cand_fine"][start:end].long().to(device)
            cs      = cs_shard[start:end].long().to(device)
            g_idx   = shard["gold_cand_idx"][start:end].long().to(device)
            covered = shard["covered"][start:end].bool().to(device)
            gate    = gate_shard[start:end].to(device)
            _, C    = ct.shape

            cmask   = (ct >= 0)
            tok_e   = F.embedding(ct.clamp(min=0), emb_w)
            base_r  = (h.unsqueeze(1) * tok_e).sum(-1)
            base_sc = base_r.masked_fill(~cmask, float("-inf"))

            if gate.any():
                r_reg    = shard["router_topk_reg"][start:end].long().to(device)
                r_prb    = shard["router_topk_prb"][start:end].float().to(device)
                m_reg    = shard["mem_topk_reg"][start:end].long().to(device)
                m_prb    = shard["mem_topk_prb"][start:end].float().to(device)
                r_margin = shard["router_margin"][start:end].float().to(device)
                m_margin = shard["mem_margin"][start:end].float().to(device)
                mdl_sc, _ = model(h, ct, cf, cs, cmask, emb_w,
                                  r_reg, r_prb, m_reg, m_prb, r_margin, m_margin)
            else:
                mdl_sc = base_sc

            gate_3d  = gate.unsqueeze(1).expand(-1, C)
            gated_sc = torch.where(gate_3d, mdl_sc, base_sc)

            n_cov_b  = int(covered.sum())
            if n_cov_b > 0:
                ar   = torch.arange(n_cov_b, device=device)
                gi_c = g_idx[covered]

                # Global gated NLL
                lp_g  = F.log_softmax(gated_sc[covered], dim=-1)
                gated_ce += float(-lp_g[ar, gi_c].sum())
                gated_nc += n_cov_b

                # Inside gate covered
                ig_mask = covered & gate
                n_ig    = int(ig_mask.sum())
                if n_ig > 0:
                    ar_ig  = torch.arange(n_ig, device=device)
                    gi_ig  = g_idx[ig_mask]
                    lp_m   = F.log_softmax(mdl_sc[ig_mask], dim=-1)
                    ig_m_ce += float(-lp_m[ar_ig, gi_ig].sum())
                    lp_b   = F.log_softmax(base_sc[ig_mask], dim=-1)
                    ig_b_ce += float(-lp_b[ar_ig, gi_ig].sum())
                    ig_nc  += n_ig

                # Outside gate covered
                og_mask = covered & ~gate
                n_og    = int(og_mask.sum())
                if n_og > 0:
                    ar_og  = torch.arange(n_og, device=device)
                    gi_og  = g_idx[og_mask]
                    lp_og  = F.log_softmax(base_sc[og_mask], dim=-1)
                    og_ce += float(-lp_og[ar_og, gi_og].sum())
                    og_nc += n_og

                sum_gold_idx_cov += int(gi_c.sum())

            ig_nt    += int(gate.sum())
            total_n  += B
            total_cov += n_cov_b
            sum_cand_counts += int(cmask.sum())
            if has_gtk:
                sum_gold_tok += int(shard["gold_token"][start:end].long().sum())

    model.train()

    fp_data = {
        "num_shards":       len(paths),
        "total_n":          total_n,
        "total_cov":        total_cov,
        "sum_cand_counts":  sum_cand_counts,
        "sum_gold_idx_cov": sum_gold_idx_cov,
        "sum_gold_tok":     sum_gold_tok,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fp_data, sort_keys=True).encode()
    ).hexdigest()[:16]

    results = {
        "gated_covered_nll":     gated_ce  / max(gated_nc, 1),
        "inside_gate_model_nll": ig_m_ce   / max(ig_nc,    1),
        "inside_gate_base_nll":  ig_b_ce   / max(ig_nc,    1),
        "outside_gate_nll":      og_ce     / max(og_nc,    1),
        "gate_rate":             ig_nt     / max(total_n,  1),
        "covered_gate_rate":     ig_nc     / max(total_cov, 1),
        "coverage":              total_cov / max(total_n,  1),
        "dataset_fingerprint":   fingerprint,
        "num_examples":          total_n,
        "num_covered":           total_cov,
        "inside_gate_n_total":   ig_nt,
        "inside_gate_n_cov":     ig_nc,
    }

    if official_baseline is not None:
        ref      = official_baseline
        fp_ok    = fingerprint == ref.get("dataset_fingerprint", "")
        n_ok     = total_n    == ref.get("num_examples", -1)
        nc_ok    = total_cov  == ref.get("num_covered",  -1)
        cov_diff = abs(results["coverage"] - ref.get("coverage", -1.0))
        cov_ok   = cov_diff < 1e-6
        ok       = fp_ok and n_ok and nc_ok and cov_ok
        if not ok:
            issues = []
            if not fp_ok:
                issues.append(f"fingerprint {fingerprint!r} != "
                              f"{ref.get('dataset_fingerprint','?')!r}")
            if not n_ok:
                issues.append(f"num_examples {total_n} != {ref.get('num_examples','?')}")
            if not nc_ok:
                issues.append(f"num_covered {total_cov} != {ref.get('num_covered','?')}")
            if not cov_ok:
                issues.append(f"coverage diff {cov_diff:.2e}")
            msg = (f"gated_global_eval MISMATCH [{variant_tag}]: "
                   + ", ".join(issues)
                   + ". Model must not change coverage/candidates/fingerprint.")
            if fail_on_mismatch:
                raise RuntimeError(msg)
            print(f"  WARNING: {msg}")

    return results


# ── Local subset eval ─────────────────────────────────────────────────────────

@torch.no_grad()
def local_subset_eval(
    model,
    val_dir: str,
    tok_emb_w: torch.Tensor,
    r2s_np: np.ndarray,
    device,
    filter_kwargs: Dict,
    eval_batch_size: int = 64,
) -> Dict[str, Dict]:
    """
    One pass over full val — evaluate model and baseline on every EVAL_SUBSET.
    Returns: {subset_name: {n_total, n_covered, model_nll, base_nll, delta_nll,
                             model_acc1, model_acc5, base_acc1, base_acc5}}
    """
    paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not paths:
        raise RuntimeError(f"No shard_*.pt in {val_dir}")

    emb_w = tok_emb_w.float().to(device)
    model.eval()

    stats: Dict[str, Dict] = {
        s: {"n_total": 0, "n_covered": 0,
            "m_ce": 0.0, "b_ce": 0.0,
            "m_acc1": 0, "m_acc5": 0,
            "b_acc1": 0, "b_acc5": 0}
        for s in EVAL_SUBSETS
    }

    for path in paths:
        shard  = torch.load(path, map_location="cpu", weights_only=True)
        N      = len(shard["covered"])

        # Compute all subset masks once per shard (CPU)
        smasks: Dict[str, torch.Tensor] = {}
        for sub in EVAL_SUBSETS:
            try:
                smasks[sub] = compute_filter_mask(shard, sub, **filter_kwargs)
            except (KeyError, ValueError):
                smasks[sub] = torch.zeros(N, dtype=torch.bool)

        cf_np      = shard["cand_fine"].numpy().astype(np.int64).clip(min=0)
        cs_np      = r2s_np[cf_np].astype(np.int64)
        cs_np[shard["cand_fine"].numpy() < 0] = 0
        cs_shard   = torch.from_numpy(cs_np)

        for start in range(0, N, eval_batch_size):
            end = min(start + eval_batch_size, N)
            B   = end - start

            h       = shard["h_prime"][start:end].float().to(device)
            ct      = shard["cand_tok"][start:end].long().to(device)
            cf      = shard["cand_fine"][start:end].long().to(device)
            cs      = cs_shard[start:end].long().to(device)
            g_idx   = shard["gold_cand_idx"][start:end].long().to(device)
            covered = shard["covered"][start:end].bool().to(device)
            _, C    = ct.shape

            cmask   = (ct >= 0)
            tok_e   = F.embedding(ct.clamp(min=0), emb_w)
            base_r  = (h.unsqueeze(1) * tok_e).sum(-1)
            base_sc = base_r.masked_fill(~cmask, float("-inf"))

            r_reg    = shard["router_topk_reg"][start:end].long().to(device)
            r_prb    = shard["router_topk_prb"][start:end].float().to(device)
            m_reg    = shard["mem_topk_reg"][start:end].long().to(device)
            m_prb    = shard["mem_topk_prb"][start:end].float().to(device)
            r_margin = shard["router_margin"][start:end].float().to(device)
            m_margin = shard["mem_margin"][start:end].float().to(device)
            mdl_sc, _ = model(h, ct, cf, cs, cmask, emb_w,
                              r_reg, r_prb, m_reg, m_prb, r_margin, m_margin)

            k5 = min(5, C)

            for sub in EVAL_SUBSETS:
                bmask   = smasks[sub][start:end].to(device)
                cov_sub = covered & bmask
                nc      = int(cov_sub.sum())
                stats[sub]["n_total"]   += int(bmask.sum())
                stats[sub]["n_covered"] += nc
                if nc == 0:
                    continue

                ar    = torch.arange(nc, device=device)
                gi_s  = g_idx[cov_sub]

                # Model NLL
                m_lp  = F.log_softmax(mdl_sc[cov_sub], dim=-1)
                stats[sub]["m_ce"] += float(-m_lp[ar, gi_s].sum())

                # Base NLL
                b_lp  = F.log_softmax(base_sc[cov_sub], dim=-1)
                stats[sub]["b_ce"] += float(-b_lp[ar, gi_s].sum())

                # Model acc@1, acc@5
                gi_exp     = gi_s.unsqueeze(1)
                m_top5_idx = mdl_sc[cov_sub].topk(k5, dim=-1).indices
                stats[sub]["m_acc1"] += int((m_top5_idx[:, :1] == gi_exp).any(1).sum())
                stats[sub]["m_acc5"] += int((m_top5_idx         == gi_exp).any(1).sum())

                # Base acc@1, acc@5
                b_top5_idx = base_sc[cov_sub].topk(k5, dim=-1).indices
                stats[sub]["b_acc1"] += int((b_top5_idx[:, :1] == gi_exp).any(1).sum())
                stats[sub]["b_acc5"] += int((b_top5_idx         == gi_exp).any(1).sum())

    model.train()

    out: Dict[str, Dict] = {}
    for sub, s in stats.items():
        nc  = s["n_covered"]
        out[sub] = {
            "n_total":   s["n_total"],
            "n_covered": nc,
            "model_nll": s["m_ce"] / max(nc, 1),
            "base_nll":  s["b_ce"] / max(nc, 1),
            "delta_nll": (s["b_ce"] - s["m_ce"]) / max(nc, 1),  # positive = improvement
            "model_acc1": s["m_acc1"] / max(nc, 1),
            "model_acc5": s["m_acc5"] / max(nc, 1),
            "base_acc1":  s["b_acc1"] / max(nc, 1),
            "base_acc5":  s["b_acc5"] / max(nc, 1),
        }
    return out


# ── Training ──────────────────────────────────────────────────────────────────

def train_hard_refiner(
    args,
    d_model: int,
    n_fine: int,
    n_super: int,
    r2s_np: np.ndarray,
    tok_emb_w: torch.Tensor,
    device: torch.device,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build model with robust hard-position init ────────────────────────────
    model = RicherMLPRefiner(d_model=d_model, n_fine=n_fine, n_super=n_super,
                             d_region=args.d_region, d_hidden=args.d_hidden)
    # residual_scale=1.0 + zero final layer → delta=0 at init, gradients flow
    # through inner layers from step 1 onward (unlike scale=0.0 which stalls them)
    model.residual_scale.data.fill_(1.0)
    nn.init.zeros_(model.mlp[-1].weight)
    nn.init.zeros_(model.mlp[-1].bias)
    model = model.to(device)
    model.register_buffer("_tok_emb_w", tok_emb_w.float().to(device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] train_filter={args.train_filter}  gate_filter={args.gate_filter}")
    print(f"[train] variant=C  d_region={args.d_region}  d_hidden={args.d_hidden}  "
          f"params={n_params:,}")

    # ── Official baseline ─────────────────────────────────────────────────────
    fail_hard: bool   = getattr(args, "fail_on_baseline_mismatch", False)
    official_bl: Optional[Dict] = None
    if args.official_baseline and os.path.isfile(args.official_baseline):
        with open(args.official_baseline) as f:
            official_bl = json.load(f)
        print()
        print("=== CANONICAL EVAL ENABLED ===")
        print(f"  baseline nll        : {official_bl['covered_nll']:.6f}")
        print(f"  baseline coverage   : {official_bl['coverage']:.6f}")
        print(f"  baseline fingerprint: {official_bl['dataset_fingerprint']}")
        print(f"  num_examples        : {official_bl['num_examples']:,}")
        print(f"  num_covered         : {official_bl['num_covered']:,}")
        if fail_hard:
            print("  --fail_on_baseline_mismatch: training aborts on any mismatch.")
        print()

    filter_kwargs = {
        "margin_thresh":  args.margin_thresh,
        "entropy_thresh": args.entropy_thresh,
    }
    variant_tag = f"C/{args.train_filter}"

    # ── Filter audit ──────────────────────────────────────────────────────────
    run_filter_audit(args.train_dir, args.val_dir, filter_kwargs, args.output_dir)

    # ── Step-0 checks ─────────────────────────────────────────────────────────
    if getattr(args, "eval_before_train", False):
        tok_emb_dev = model._tok_emb_w
        print("\n[train] === step-0 eval (eval_before_train) ===")

        print("  [step 0 / force_zero] full val ...")
        m0_zero = canonical_eval_refiner(
            None, args.val_dir, tok_emb_dev, r2s_np, device,
            force_zero=True, eval_batch_size=args.eval_batch_size,
            official_baseline=official_bl, fail_on_mismatch=fail_hard,
            variant_tag=variant_tag,
        )
        fp0 = m0_zero["dataset_fingerprint"]
        print(f"  [step 0 / force_zero]  covered_nll={m0_zero['covered_nll']:.6f}  "
              f"cov={m0_zero['coverage']:.6f}  fingerprint={fp0}")
        if official_bl:
            check_baseline_match(m0_zero, official_bl, fail_hard,
                                 context=f"force_zero/{variant_tag}", check_nll=True)

        print()
        print("  [init identity] checking residual_scale=1.0 + zero_final_layer → delta=0 ...")
        check_init_identity(model, args.val_dir, tok_emb_dev, r2s_np, device,
                            print_delta_stats=True)

        print()
        print("  [step 0 / gated_global] full val ...")
        m0_gated = gated_global_eval(
            model, args.val_dir, tok_emb_dev, r2s_np, device,
            gate_filter_name=args.gate_filter,
            filter_kwargs=filter_kwargs,
            official_baseline=official_bl,
            fail_on_mismatch=fail_hard,
            variant_tag=variant_tag,
            eval_batch_size=args.eval_batch_size,
        )
        print(f"  [step 0 / gated_global]  gated_nll={m0_gated['gated_covered_nll']:.6f}  "
              f"cov={m0_gated['coverage']:.6f}  fingerprint={m0_gated['dataset_fingerprint']}")

        nll_diff0 = abs(m0_gated["gated_covered_nll"] - m0_zero["covered_nll"])
        fp_ok0    = m0_gated["dataset_fingerprint"] == fp0
        cov_ok0   = abs(m0_gated["coverage"] - m0_zero["coverage"]) < 1e-6
        if nll_diff0 >= 1e-4 or not fp_ok0 or not cov_ok0:
            raise RuntimeError(
                f"Step-0 gated != force_zero [{variant_tag}]: "
                f"gated={m0_gated['gated_covered_nll']:.6f}  "
                f"force_zero={m0_zero['covered_nll']:.6f}  diff={nll_diff0:.2e}  "
                f"fp_ok={fp_ok0}  cov_ok={cov_ok0}. "
                f"Robust init (scale=1.0 + zero final layer) should give delta=0 at step 0."
            )
        print(f"  [step 0] gated == force_zero: PASS  nll_diff={nll_diff0:.2e}")
        model.train()

    # ── Dataset + optimiser ───────────────────────────────────────────────────
    train_ds  = FilteredShardStreamDataset(
        args.train_dir, r2s_np, args.train_filter,
        filter_kwargs=filter_kwargs, shuffle=True,
    )
    train_inf = make_infinite(train_ds, args.batch_size)

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler("cuda")
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                         eta_min=args.lr * 0.1)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path    = os.path.join(args.output_dir, "train_log.csv")
    log_fields  = ["step", "ce", "kl", "delta",
                   "gated_covered_nll", "inside_gate_model_nll", "inside_gate_base_nll",
                   "outside_gate_nll", "gate_rate", "covered_gate_rate",
                   "coverage", "dataset_fingerprint", "num_examples", "num_covered",
                   "inside_gate_n_total", "inside_gate_n_cov"]
    log_file    = open(log_path, "w", newline="")
    log_csv     = csv.DictWriter(log_file, fieldnames=log_fields, extrasaction="ignore")
    log_csv.writeheader()

    sub_path    = os.path.join(args.output_dir, "local_subset_eval.csv")
    sub_fields  = ["step", "subset", "n_total", "n_covered",
                   "model_nll", "base_nll", "delta_nll",
                   "model_acc1", "model_acc5", "base_acc1", "base_acc5"]
    sub_file    = open(sub_path, "w", newline="")
    sub_csv     = csv.DictWriter(sub_file, fieldnames=sub_fields, extrasaction="ignore")
    sub_csv.writeheader()

    best_path  = os.path.join(args.output_dir, "best_refiner.pt")
    last_path  = os.path.join(args.output_dir, "last_refiner.pt")
    best_nll   = float("inf")
    best_step  = 0
    ema_ce     = None
    t0         = time.time()
    bl_nll     = official_bl["covered_nll"] if official_bl else float("nan")

    model.train()

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in range(1, args.steps + 1):
        batch = next(train_inf)

        with autocast("cuda"):
            loss, info = compute_loss(model, batch, device,
                                      args.lambda_kl, args.lambda_delta)
        if loss is None:
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()

        ema_ce = info["ce"] if ema_ce is None else 0.98 * ema_ce + 0.02 * info["ce"]

        if step % 100 == 0:
            print(f"  step {step:6d}/{args.steps}  ce={ema_ce:.4f}  "
                  f"kl={info['kl']:.4f}  t={time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] step={step} ···")

            g_metrics = gated_global_eval(
                model, args.val_dir, model._tok_emb_w, r2s_np, device,
                gate_filter_name=args.gate_filter,
                filter_kwargs=filter_kwargs,
                official_baseline=official_bl,
                fail_on_mismatch=fail_hard,
                variant_tag=variant_tag,
                eval_batch_size=args.eval_batch_size,
            )

            l_results = local_subset_eval(
                model, args.val_dir, model._tok_emb_w, r2s_np, device,
                filter_kwargs=filter_kwargs,
                eval_batch_size=args.eval_batch_size,
            )

            row = {"step": step, **info, **g_metrics}
            log_csv.writerow(row)
            log_file.flush()

            for sub, m in l_results.items():
                sub_csv.writerow({"step": step, "subset": sub, **m})
            sub_file.flush()

            gated_nll   = g_metrics["gated_covered_nll"]
            delta_vs_bl = bl_nll - gated_nll

            print(f"  [eval/gated_global] step={step}")
            print(f"    gated_covered_nll   = {gated_nll:.6f}  "
                  f"(baseline={bl_nll:.6f}  delta={delta_vs_bl:+.6f})")
            print(f"    inside_gate_model   = {g_metrics['inside_gate_model_nll']:.6f}")
            print(f"    inside_gate_base    = {g_metrics['inside_gate_base_nll']:.6f}")
            print(f"    outside_gate_nll    = {g_metrics['outside_gate_nll']:.6f}")
            print(f"    gate_rate           = {g_metrics['gate_rate']:.4f}  "
                  f"({g_metrics['inside_gate_n_total']:,} / {g_metrics['num_examples']:,})")
            print(f"    coverage            = {g_metrics['coverage']:.6f}")
            print(f"    fingerprint         = {g_metrics['dataset_fingerprint']}")
            print(f"    num_examples        = {g_metrics['num_examples']:,}")
            print(f"    num_covered         = {g_metrics['num_covered']:,}")

            print(f"  [eval/local_subsets]  (model vs base, covered positions)")
            print(f"    {'subset':30s}  {'n_cov':>7}  {'model_nll':>9}  "
                  f"{'base_nll':>8}  {'delta':>7}  {'acc@1':>5}  {'base_acc@1':>10}")
            for sub in ["hard_union", "hard_union_small", "hard_union_medium",
                        "type_A", "type_B", "boundary", "router_top8_miss", "all"]:
                if sub not in l_results:
                    continue
                m = l_results[sub]
                if m["n_covered"] == 0:
                    continue
                print(f"    {sub:30s}  {m['n_covered']:7,}  "
                      f"{m['model_nll']:9.4f}  {m['base_nll']:8.4f}  "
                      f"{m['delta_nll']:+7.4f}  {m['model_acc1']:5.3f}  "
                      f"{m['base_acc1']:10.3f}")

            if gated_nll < best_nll:
                best_nll  = gated_nll
                best_step = step
                best_metrics = {
                    "eval_mode":             "gated_global_val",
                    "step":                  step,
                    "train_filter":          args.train_filter,
                    "gate_filter":           args.gate_filter,
                    "official_baseline_nll": bl_nll,
                    "gated_covered_nll":     gated_nll,
                    "delta_vs_baseline":     delta_vs_bl,
                    "inside_gate_model_nll": g_metrics["inside_gate_model_nll"],
                    "inside_gate_base_nll":  g_metrics["inside_gate_base_nll"],
                    "outside_gate_nll":      g_metrics["outside_gate_nll"],
                    "gate_rate":             g_metrics["gate_rate"],
                    "covered_gate_rate":     g_metrics["covered_gate_rate"],
                    "coverage":              g_metrics["coverage"],
                    "fingerprint":           g_metrics["dataset_fingerprint"],
                    "num_examples":          g_metrics["num_examples"],
                    "num_covered":           g_metrics["num_covered"],
                    "local_subsets":         l_results,
                }
                torch.save({"step": step, "model": model.state_dict(),
                            "metrics": best_metrics, "args": vars(args)}, best_path)
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
                print(f"  [eval] *** new best  gated_nll={best_nll:.6f}  "
                      f"delta_vs_baseline={delta_vs_bl:+.6f}  saved → {best_path}")

    torch.save({"step": args.steps, "model": model.state_dict(),
                "metrics": {}, "args": vars(args)}, last_path)
    log_file.close()
    sub_file.close()
    print(f"\n[train] done  filter={args.train_filter}  "
          f"best_gated_nll={best_nll:.6f}  best_step={best_step}  ckpt={best_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device(args.device)

    print(f"[main] loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    if hasattr(backbone, "token_emb"):
        token_emb_w = backbone.token_emb.weight.detach().cpu()
    elif hasattr(backbone, "transformer"):
        token_emb_w = backbone.transformer.wte.weight.detach().cpu()
    else:
        raise RuntimeError("Cannot locate token embedding in backbone")

    # Region config
    cfg_path = os.path.join(args.val_dir, "dataset_config.json")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(args.train_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            ds_cfg = json.load(f)
        n_fine  = ds_cfg.get("n_fine",  128)
        n_super = ds_cfg.get("n_super",  24)
        print(f"[main] n_fine={n_fine}  n_super={n_super} (from dataset_config)")
    else:
        n_fine  = args.n_fine
        n_super = args.n_super

    r2s_np = np.zeros(n_fine, dtype=np.int32)
    if args.super_map and os.path.isfile(args.super_map):
        r2s_np  = load_r2s(args.super_map, n_fine)
        n_super = int(r2s_np.max()) + 1
        print(f"[main] loaded super_map  n_super={n_super}")

    if getattr(args, "audit_only", False):
        filter_kwargs = {"margin_thresh": args.margin_thresh,
                         "entropy_thresh": args.entropy_thresh}
        os.makedirs(args.output_dir, exist_ok=True)
        run_filter_audit(args.train_dir, args.val_dir, filter_kwargs, args.output_dir)
        return

    if not args.train_dir:
        raise RuntimeError("--train_dir is required for training")
    if not args.output_dir:
        raise RuntimeError("--output_dir is required for training")

    train_hard_refiner(args, d_model, n_fine, n_super, r2s_np, token_emb_w, device)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",    default="")
    p.add_argument("--val_dir",      required=True)
    p.add_argument("--small_ckpt",   required=True)
    p.add_argument("--super_map",    default=None)
    p.add_argument("--output_dir",   default="")

    p.add_argument("--train_filter", default="hard_union",
                   choices=TRAIN_FILTER_CHOICES,
                   help="Filter applied to training positions.")
    p.add_argument("--gate_filter",  default=None,
                   help="Filter used for gated global eval (default: same as train_filter).")
    p.add_argument("--margin_thresh",  type=float, default=0.1)
    p.add_argument("--entropy_thresh", type=float, default=2.0)

    p.add_argument("--d_region",    type=int,   default=32)
    p.add_argument("--d_hidden",    type=int,   default=256)
    p.add_argument("--n_fine",      type=int,   default=128)
    p.add_argument("--n_super",     type=int,   default=24)

    p.add_argument("--steps",       type=int,   default=10_000)
    p.add_argument("--eval_every",  type=int,   default=1_000)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--lambda_kl",   type=float, default=0.01)
    p.add_argument("--lambda_delta", type=float, default=1e-4)

    p.add_argument("--official_baseline", default=None)
    p.add_argument("--fail_on_baseline_mismatch", action="store_true")
    p.add_argument("--eval_before_train", action="store_true")
    p.add_argument("--audit_only",        action="store_true",
                   help="Only run filter audit; skip training.")

    p.add_argument("--device", default="cuda")

    args = p.parse_args()
    if args.gate_filter is None:
        args.gate_filter = args.train_filter
    return args


if __name__ == "__main__":
    run(_parse())
