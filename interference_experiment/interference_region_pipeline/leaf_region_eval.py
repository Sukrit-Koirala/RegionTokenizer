#!/usr/bin/env python3
"""
leaf_region_eval.py

Evaluates whether leaf nodes of the recursive region hierarchy are better
routing units than coarse top-level regions.

A leaf is any terminal node in region_tree.json with no children.
If a top-level region was not recursively split, it is itself a leaf.

Steps:
  1. Flatten region tree → leaf predictive interference regions
  2. Build per-sample leaf labels from probe cache
  3. Train linear router probes per layer (h_layer → leaf_id)
  4. Gold recall under top-r leaf routing (r = 1,2,4,8,16,32)
  5. Routed softmax simulation (NLL / PPL)
  6. Logit locality over leaves
  7. Plots
  8. Summary
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_logging, set_seed, ensure_dirs,
    load_json, save_json,
    fit_probe, probe_predict_topk,
)

log = setup_logging("leaf_eval")


# ── Step 1: Flatten tree to leaves ───────────────────────────────────────────

def _traverse(node: dict, leaf_list: list) -> None:
    """DFS: collect vocab_ids only at leaf nodes (nodes with no children)."""
    children = node.get("children", {})
    if not children:
        leaf_list.append(node.get("vocab_ids", []))
    else:
        for child in children.values():
            _traverse(child, leaf_list)


def flatten_tree_to_leaves(tree: dict) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Returns:
      leaf_to_vocab  : leaf_id (0-indexed) -> [vocab_ids]
      vocab_to_leaf  : vocab_id -> leaf_id  (-1 if not in any leaf)
    """
    root     = tree.get("root", tree)
    children = root.get("children", {})
    if not children:
        # Tree has no hierarchy — treat root itself as a single leaf
        log.warning("Tree has no children at root; treating root as a single leaf")
        leaf_list = [root.get("vocab_ids", [])]
    else:
        leaf_list = []
        for child in children.values():
            _traverse(child, leaf_list)

    leaf_to_vocab: Dict[int, List[int]] = {}
    vocab_to_leaf: Dict[int, int] = {}
    for lid, vids in enumerate(leaf_list):
        leaf_to_vocab[lid] = [int(v) for v in vids]
        for v in vids:
            vocab_to_leaf[int(v)] = lid

    return leaf_to_vocab, vocab_to_leaf


def save_leaf_reports(
    leaf_to_vocab: Dict[int, List[int]],
    vocab_to_leaf: Dict[int, int],
    tokenizer,
    output_dir: str,
) -> None:
    save_json({str(k): v for k, v in leaf_to_vocab.items()},
              os.path.join(output_dir, "leaf_region_to_tokens.json"))
    save_json({str(k): v for k, v in vocab_to_leaf.items()},
              os.path.join(output_dir, "leaf_token_to_region.json"))

    sizes  = [len(v) for v in leaf_to_vocab.values()]
    n_leaf = len(leaf_to_vocab)
    log.info("Leaves: %d  |  sizes min=%d  mean=%.1f  max=%d  total=%d",
             n_leaf, min(sizes), np.mean(sizes), max(sizes), sum(sizes))

    with open(os.path.join(output_dir, "leaf_region_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total leaf predictive interference regions: {n_leaf}\n")
        f.write(f"Sizes — min={min(sizes)}  mean={np.mean(sizes):.1f}  "
                f"max={max(sizes)}  total={sum(sizes)}\n\n")
        for lid in sorted(leaf_to_vocab):
            vids = leaf_to_vocab[lid]
            f.write(f"Leaf {lid}  (size={len(vids)})\n")
            shown = vids[:20]
            if tokenizer:
                decoded = [repr(tokenizer.decode([v])) for v in shown]
                f.write("  " + "  ".join(decoded) + "\n")
            else:
                f.write("  " + " ".join(str(v) for v in shown) + "\n")
            f.write("\n")


# ── Step 2: Build sample labels ───────────────────────────────────────────────

def load_probe_cache(cache_dir: str):
    meta    = load_json(os.path.join(cache_dir, "meta.json"))
    layers  = meta["layers"]
    hidden  = {l: np.load(os.path.join(cache_dir, f"hidden_L{l:02d}.npy")) for l in layers}
    tok_ids = np.load(os.path.join(cache_dir, "probe_token_ids.npy"))
    topk    = np.load(os.path.join(cache_dir, "topk_ids.npy"))
    return hidden, tok_ids, topk, layers, meta


def build_leaf_labels(
    probe_token_ids: np.ndarray,
    vocab_to_leaf: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map gold next-token vocab IDs → compact leaf labels [0..n_leaves-1].
    Returns (labels, valid_mask) where valid_mask selects positions with a known leaf.
    """
    # Raw labels (some may be -1 if vocab_id not in any leaf)
    raw = np.array([vocab_to_leaf.get(int(t), -1) for t in probe_token_ids], dtype=np.int32)
    valid = raw >= 0

    # compact remap: leaf IDs are already 0-indexed from flatten_tree_to_leaves
    n_covered  = int(valid.sum())
    n_classes  = int(raw[valid].max()) + 1 if n_covered > 0 else 0
    log.info("Probe samples: total=%d  covered=%d (%.1f%%)  n_leaf_classes=%d",
             len(raw), n_covered, 100 * n_covered / max(len(raw), 1), n_classes)

    return raw, valid


# ── Step 3: Linear router probes ─────────────────────────────────────────────

def train_leaf_probes(
    hidden: Dict[int, np.ndarray],
    labels: np.ndarray,
    valid: np.ndarray,
    n_classes: int,
    layers: List[int],
    device: torch.device,
    rng: np.random.Generator,
    epochs: int,
) -> Tuple[List[dict], np.ndarray, np.ndarray, object]:
    """
    Returns (per_layer_metrics, tr_idx, te_idx, final_probe).
    final_probe is the probe trained on layers[-1] — used for Steps 4+5.
    """
    valid_idx = np.where(valid)[0]
    perm  = rng.permutation(len(valid_idx))
    n_tr  = int(0.7 * len(perm))
    tr_idx = valid_idx[perm[:n_tr]]
    te_idx = valid_idx[perm[n_tr:]]
    y_tr   = labels[tr_idx]
    y_te   = labels[te_idx]

    results   = []
    final_probe = None

    for l in layers:
        h = hidden[l].astype(np.float32)
        X_tr = h[tr_idx]; X_te = h[te_idx]
        log.info("  Layer %2d  n_tr=%d  n_te=%d  n_classes=%d",
                 l, len(X_tr), len(X_te), n_classes)

        m, probe = fit_probe(X_tr, y_tr, X_te, y_te, n_classes, device,
                             epochs=epochs)

        # top-8 / top-16
        k8  = min(8,  n_classes)
        k16 = min(16, n_classes)
        probe.eval()
        with torch.no_grad():
            logits = probe(torch.from_numpy(X_te).to(device))
            yte_t  = torch.from_numpy(y_te).long().to(device)
            top8   = float((logits.topk(k8,  1).indices == yte_t.unsqueeze(1)).any(1).float().mean())
            top16  = float((logits.topk(k16, 1).indices == yte_t.unsqueeze(1)).any(1).float().mean())

        # Frequency baseline: always predict the most common class
        freq_base = float(np.bincount(y_te, minlength=n_classes).max()) / len(y_te)

        m.update({"layer": l, "top8": top8, "top16": top16,
                  "freq_baseline": freq_base})
        log.info("    top1=%.3f  top4=%.3f  top8=%.3f  top16=%.3f  CE=%.3f bits",
                 m["top1"], m["top4"], top8, top16, m["ce_bits"])
        results.append(m)

        if l == layers[-1]:
            final_probe = probe

    return results, tr_idx, te_idx, final_probe


# ── Step 4: Gold recall under top-r leaf routing ─────────────────────────────

def gold_recall_top_r(
    probe: nn.Module,
    hidden_final: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    leaf_to_vocab: Dict[int, List[int]],
    n_leaves: int,
    device: torch.device,
    rng: np.random.Generator,
    te_idx: np.ndarray,
    r_values: List[int],
) -> List[dict]:
    """
    Predict top-r leaves per sample; check if gold leaf is among them.
    Candidate size = sum of leaf sizes for predicted leaves (leaves are disjoint).
    """
    log.info("=== Step 4: Gold recall under top-r leaf routing ===")

    leaf_sizes = torch.tensor(
        [len(leaf_to_vocab.get(l, [])) for l in range(n_leaves)],
        dtype=torch.float32, device=device,
    )                                                              # (n_leaves,)

    h_te = torch.from_numpy(hidden_final[te_idx].astype(np.float32)).to(device)
    y_te = torch.from_numpy(labels[te_idx]).long().to(device)    # (N_te,)

    probe.eval()
    with torch.no_grad():
        logits = probe(h_te)                                       # (N_te, n_leaves)

    max_r = max(r_values)
    pred_topk = logits.topk(min(max_r, n_leaves), dim=1).indices  # (N_te, max_r)

    total_vocab = sum(len(v) for v in leaf_to_vocab.values())
    results = []

    for r in r_values:
        r_clamped = min(r, n_leaves)
        pred_r = pred_topk[:, :r_clamped]                         # (N_te, r)

        # Gold leaf in predicted top-r?
        hit = (pred_r == y_te.unsqueeze(1)).any(dim=1)            # (N_te,) bool
        recall = float(hit.float().mean())

        # Candidate set size = sum of selected leaf sizes (leaves are disjoint)
        cand_sizes = leaf_sizes[pred_r].sum(dim=1)                 # (N_te,)
        avg_cand = float(cand_sizes.mean())
        med_cand = float(cand_sizes.median())

        # Oracle: always select the correct leaf → size = that leaf's size
        oracle_cand = leaf_sizes[y_te].mean()
        oracle_recall = 1.0   # by definition

        # Random baseline: draw r leaves uniformly
        rand_perm  = torch.from_numpy(
            rng.integers(0, n_leaves, size=(len(te_idx), r_clamped))
        ).to(device)
        rand_hit   = (rand_perm == y_te.unsqueeze(1)).any(dim=1)
        rand_recall = float(rand_hit.float().mean())
        rand_cand   = float(leaf_sizes[rand_perm].sum(dim=1).mean())

        results.append({
            "r_leaves":        r,
            "gold_recall":     recall,
            "avg_cand_size":   avg_cand,
            "med_cand_size":   med_cand,
            "cand_frac_vocab": avg_cand / max(total_vocab, 1),
            "oracle_recall":   oracle_recall,
            "oracle_cand":     float(oracle_cand),
            "random_recall":   rand_recall,
            "random_cand":     rand_cand,
        })
        log.info("  r=%2d  recall=%.3f  cand=%.0f (%.2f%% vocab)  "
                 "random_recall=%.3f  oracle_cand=%.1f",
                 r, recall, avg_cand, 100 * avg_cand / max(total_vocab, 1),
                 rand_recall, float(oracle_cand))

    return results


# ── Optional coarse-region comparison ────────────────────────────────────────

def coarse_gold_recall(
    probe_token_ids: np.ndarray,
    region_to_tokens: Dict[str, List[int]],
    token_to_region: Dict[str, int],
    te_idx: np.ndarray,
    hidden_final: np.ndarray,
    labels_coarse: np.ndarray,
    n_classes_coarse: int,
    device: torch.device,
    rng: np.random.Generator,
    r_values: List[int],
    epochs: int,
) -> List[dict]:
    """Run the same gold-recall experiment on coarse regions for comparison."""
    from utils import fit_probe as _fit
    valid_c = labels_coarse >= 0
    valid_idx_c = np.where(valid_c)[0]
    perm   = rng.permutation(len(valid_idx_c))
    n_tr   = int(0.7 * len(perm))
    tr_c   = valid_idx_c[perm[:n_tr]]
    te_c   = valid_idx_c[perm[n_tr:]]

    H = hidden_final.astype(np.float32)
    _, probe_c = _fit(H[tr_c], labels_coarse[tr_c], H[te_c], labels_coarse[te_c],
                      n_classes_coarse, device, epochs=epochs)

    region_sizes = torch.tensor(
        [len(region_to_tokens.get(str(i), [])) for i in range(n_classes_coarse)],
        dtype=torch.float32, device=device,
    )
    h_te = torch.from_numpy(H[te_c].astype(np.float32)).to(device)
    y_te = torch.from_numpy(labels_coarse[te_c]).long().to(device)
    probe_c.eval()
    with torch.no_grad():
        logits = probe_c(h_te)
    max_r = max(r_values)
    pred_topk = logits.topk(min(max_r, n_classes_coarse), dim=1).indices

    total_vocab = sum(len(v) for v in region_to_tokens.values())
    results = []
    for r in r_values:
        r_c = min(r, n_classes_coarse)
        pred_r = pred_topk[:, :r_c]
        hit = (pred_r == y_te.unsqueeze(1)).any(dim=1)
        cand_sizes = region_sizes[pred_r].sum(dim=1)
        results.append({
            "r_regions":    r,
            "gold_recall":  float(hit.float().mean()),
            "avg_cand":     float(cand_sizes.mean()),
            "cand_frac":    float(cand_sizes.mean()) / max(total_vocab, 1),
        })
    return results


# ── Step 5: Routed softmax simulation ─────────────────────────────────────────

def routed_softmax_sim(
    hidden_final: np.ndarray,
    labels: np.ndarray,
    probe_token_ids: np.ndarray,
    leaf_to_vocab: Dict[int, List[int]],
    n_leaves: int,
    te_idx: np.ndarray,
    probe: nn.Module,
    model_name: str,
    device: torch.device,
    rng: np.random.Generator,
    n_eval: int,
    r_values: List[int],
) -> List[dict]:
    """
    Applies LM head (ln_f + wte.T) to cached final-layer hidden states.
    No second forward pass through the transformer required.
    """
    log.info("=== Step 5: Routed softmax simulation ===")
    log.info("  Loading LM head from %s ...", model_name)
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    ln_f      = lm.transformer.ln_f.eval().float().to(device)
    lm_weight = lm.transformer.wte.weight.detach().float().to(device)   # (vocab_size, D)
    vocab_size = lm_weight.shape[0]
    del lm
    torch.cuda.empty_cache()

    # Build leaf → full-vocab boolean mask on CPU, then move to GPU
    log.info("  Building leaf vocab masks (n_leaves=%d, vocab_size=%d) ...",
             n_leaves, vocab_size)
    leaf_vocab_mask = torch.zeros(n_leaves, vocab_size, dtype=torch.bool)
    for lid, vids in leaf_to_vocab.items():
        for v in vids:
            if v < vocab_size:
                leaf_vocab_mask[lid, v] = True
    leaf_vocab_mask = leaf_vocab_mask.to(device)   # (n_leaves, vocab_size)

    # Subsample from test set
    n_eval  = min(n_eval, len(te_idx))
    ev_idx  = te_idx[rng.choice(len(te_idx), n_eval, replace=False)]
    gold_vids = probe_token_ids[ev_idx].astype(np.int64)

    # Predict top-r leaves for eval positions
    H = hidden_final.astype(np.float32)
    probe.eval()
    with torch.no_grad():
        logits_probe = probe(torch.from_numpy(H[ev_idx]).to(device))   # (N_ev, n_leaves)
    max_r    = max(r_values)
    pred_top = logits_probe.topk(min(max_r, n_leaves), dim=1).indices  # (N_ev, max_r)

    results = []
    B = 256

    for r in r_values:
        r_c = min(r, n_leaves)
        pred_r = pred_top[:, :r_c]                                     # (N_ev, r)

        # Candidate mask for each eval position: (N_ev, vocab_size)
        # candidate_mask[i] = OR of leaf_vocab_mask[pred_r[i]]
        # Build in batches to avoid OOM
        full_nlls: List[float] = []
        routed_nlls: List[float] = []
        oracle_nlls: List[float] = []
        rand_nlls:   List[float] = []
        failures = 0

        # Random leaf baseline (same r, uniformly drawn)
        rand_leaves = torch.from_numpy(
            rng.integers(0, n_leaves, size=(n_eval, r_c))
        ).to(device)

        for s in range(0, n_eval, B):
            e = min(s + B, n_eval)
            Bs = e - s

            h_b = torch.from_numpy(H[ev_idx[s:e]]).float().to(device)    # (Bs, D)
            with torch.no_grad():
                h_ln = ln_f(h_b)
                logits_b = h_ln @ lm_weight.T                            # (Bs, vocab_size)

            gold_b = torch.from_numpy(gold_vids[s:e]).long().to(device)  # (Bs,)

            # Full softmax NLL
            lp_full = logits_b.log_softmax(dim=-1)
            nll_full = -lp_full.gather(1, gold_b.unsqueeze(1)).squeeze(1)  # (Bs,)

            # Predicted-region masked NLL
            pred_b = pred_r[s:e]                                         # (Bs, r)
            cand_b = leaf_vocab_mask[pred_b].any(dim=1)                  # (Bs, vocab_size)
            logits_m = logits_b.clone()
            logits_m[~cand_b] = -1e9
            lp_routed = logits_m.log_softmax(dim=-1)
            # NLL: -inf if gold excluded
            in_cand = cand_b.gather(1, gold_b.unsqueeze(1)).squeeze(1)   # (Bs,) bool
            nll_routed = -lp_routed.gather(1, gold_b.unsqueeze(1)).squeeze(1)
            nll_routed[~in_cand] = float("inf")
            failures += int((~in_cand).sum())

            # Oracle: mask to the gold token's own leaf
            gold_labels_b = torch.from_numpy(labels[ev_idx[s:e]]).long().to(device)
            cand_oracle = leaf_vocab_mask[gold_labels_b]                  # (Bs, vocab_size)
            logits_o = logits_b.clone()
            logits_o[~cand_oracle] = -1e9
            lp_oracle = logits_o.log_softmax(dim=-1)
            nll_oracle = -lp_oracle.gather(1, gold_b.unsqueeze(1)).squeeze(1)

            # Random leaf baseline
            rand_b = rand_leaves[s:e]                                    # (Bs, r)
            cand_rand = leaf_vocab_mask[rand_b].any(dim=1)
            logits_rd = logits_b.clone()
            logits_rd[~cand_rand] = -1e9
            lp_rand = logits_rd.log_softmax(dim=-1)
            nll_rand = -lp_rand.gather(1, gold_b.unsqueeze(1)).squeeze(1)

            for v in nll_full:   full_nlls.append(float(v))
            for v in nll_routed: routed_nlls.append(float(v))
            for v in nll_oracle: oracle_nlls.append(float(v))
            for v in nll_rand:   rand_nlls.append(float(v))

        def _mean_finite(vs):
            fs = [v for v in vs if v != float("inf") and not np.isnan(v)]
            return float(np.mean(fs)) if fs else float("nan")

        full_nll    = _mean_finite(full_nlls)
        routed_nll  = _mean_finite(routed_nlls)
        oracle_nll  = _mean_finite(oracle_nlls)
        rand_nll    = _mean_finite(rand_nlls)

        # Effective routed NLL: penalise failures with full-vocab NLL + large penalty
        fail_rate = failures / max(n_eval, 1)
        n_success = n_eval - failures
        eff_routed = (
            (n_success * routed_nll + failures * (full_nll + 10.0)) / n_eval
            if n_eval > 0 and not np.isnan(routed_nll) else float("nan")
        )

        results.append({
            "r_leaves":         r,
            "full_nll":         full_nll,
            "full_ppl":         float(np.exp(full_nll)) if not np.isnan(full_nll) else float("nan"),
            "routed_nll":       routed_nll,
            "routed_ppl":       float(np.exp(routed_nll)) if not np.isnan(routed_nll) else float("nan"),
            "eff_routed_nll":   eff_routed,
            "eff_routed_ppl":   float(np.exp(eff_routed)) if not np.isnan(eff_routed) else float("nan"),
            "oracle_nll":       oracle_nll,
            "oracle_ppl":       float(np.exp(oracle_nll)) if not np.isnan(oracle_nll) else float("nan"),
            "random_nll":       rand_nll,
            "nll_delta":        routed_nll - full_nll,
            "failure_rate":     fail_rate,
        })
        log.info("  r=%2d  full_ppl=%.1f  routed_ppl=%.1f  "
                 "nll_delta=%+.3f  failures=%.1f%%",
                 r, results[-1]["full_ppl"], results[-1]["routed_ppl"],
                 results[-1]["nll_delta"], 100 * fail_rate)

    return results


# ── Step 6: Logit locality over leaves ───────────────────────────────────────

def logit_locality(
    topk_ids: np.ndarray,
    probe_token_ids: np.ndarray,
    vocab_to_leaf: Dict[int, int],
    ks: List[int],
) -> List[dict]:
    log.info("=== Step 6: Logit locality ===")
    T = topk_ids.shape[0]
    n_leaves = max(vocab_to_leaf.values()) + 1 if vocab_to_leaf else 1

    results = []
    for K in ks:
        K = min(K, topk_ids.shape[1])
        same_top1, same_gold, n_unique, entropies = [], [], [], []

        for i in range(T):
            top1_leaf = vocab_to_leaf.get(int(topk_ids[i, 0]), -1)
            gold_leaf  = vocab_to_leaf.get(int(probe_token_ids[i]), -1)
            topk_vids  = topk_ids[i, :K]
            topk_leaves = np.array([vocab_to_leaf.get(int(v), -1) for v in topk_vids], dtype=np.int32)
            valid = topk_leaves >= 0
            if not valid.any():
                continue
            vl = topk_leaves[valid]

            counts = np.bincount(vl, minlength=n_leaves)
            pk = counts / counts.sum()
            nz = pk[pk > 0]
            H  = float(-np.sum(nz * np.log(nz + 1e-12)) / np.log(2))

            same_top1.append(float((vl == top1_leaf).mean()) if top1_leaf >= 0 else float("nan"))
            same_gold.append( float((vl == gold_leaf ).mean()) if gold_leaf  >= 0 else float("nan"))
            n_unique.append(len(np.unique(vl)))
            entropies.append(H)

        results.append({
            "K":               K,
            "same_top1_frac":  float(np.nanmean(same_top1)),
            "same_gold_frac":  float(np.nanmean(same_gold)),
            "avg_unique":      float(np.mean(n_unique)),
            "avg_entropy_bits":float(np.mean(entropies)),
        })
        log.info("  K=%4d  same_top1=%.3f  same_gold=%.3f  unique=%.1f  H=%.3f bits",
                 K, results[-1]["same_top1_frac"], results[-1]["same_gold_frac"],
                 results[-1]["avg_unique"], results[-1]["avg_entropy_bits"])
    return results


# ── Step 7: Plots ─────────────────────────────────────────────────────────────

def save_plots(
    leaf_to_vocab: Dict[int, List[int]],
    probe_results: List[dict],
    recall_results: List[dict],
    softmax_results: List[dict],
    locality_results: List[dict],
    coarse_recall: Optional[List[dict]],
    output_dir: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return

    # 1. Leaf size distribution
    sizes = sorted([len(v) for v in leaf_to_vocab.values()])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(sizes)), sizes, color="steelblue", width=1.0)
    ax.set_xlabel("Leaf region (sorted by size)")
    ax.set_ylabel("Number of tokens")
    ax.set_title(f"Leaf region size distribution ({len(sizes)} leaves)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "leaf_size_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. Router recall vs layer
    if probe_results:
        ls = [str(r["layer"]) for r in probe_results]
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, label in [("top1","top-1"),("top4","top-4"),("top8","top-8"),("top16","top-16")]:
            ax.plot(ls, [r[key] for r in probe_results], "o-", lw=2, label=label)
        ax.axhline(probe_results[0]["random_top1"], ls="--", color="gray", lw=1, label="random")
        ax.set_xlabel("Layer"); ax.set_ylabel("Recall")
        ax.set_title("Leaf router probe recall vs layer")
        ax.legend(ncol=2); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "leaf_router_recall_vs_layer.png"), dpi=150)
        plt.close(fig)

    # 3. Gold recall vs candidate size
    if recall_results:
        cand = [r["avg_cand_size"] for r in recall_results]
        rec  = [r["gold_recall"]   for r in recall_results]
        rand = [r["random_recall"] for r in recall_results]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(cand, rec,  "o-",  lw=2, label="leaf routing")
        ax.plot(cand, rand, "s--", lw=1.5, color="gray", label="random leaves")
        if coarse_recall:
            ax.plot([r["avg_cand"] for r in coarse_recall],
                    [r["gold_recall"] for r in coarse_recall],
                    "^:", lw=1.5, color="tab:orange", label="coarse routing")
        ax.set_xlabel("Avg candidate set size")
        ax.set_ylabel("Gold recall")
        ax.set_title("Gold recall vs candidate set size")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "leaf_gold_recall_vs_candidate_size.png"), dpi=150)
        plt.close(fig)

    # 4. Routed PPL vs candidate size
    if softmax_results:
        cand_s = [r.get("avg_cand_size", r["r_leaves"]) for r in softmax_results]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r["r_leaves"] for r in softmax_results],
                [r["eff_routed_ppl"] for r in softmax_results], "o-", lw=2, label="routed (effective)")
        ax.plot([r["r_leaves"] for r in softmax_results],
                [r["full_ppl"] for r in softmax_results], "--", color="gray", lw=1.5, label="full softmax")
        ax.set_xlabel("r (leaves in candidate set)")
        ax.set_ylabel("Perplexity")
        ax.set_title("Routed softmax PPL vs number of routing leaves")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "leaf_routed_ppl_vs_candidate_size.png"), dpi=150)
        plt.close(fig)

    # 5. Logit locality
    if locality_results:
        ks    = [r["K"] for r in locality_results]
        s_t1  = [r["same_top1_frac"] for r in locality_results]
        s_gld = [r["same_gold_frac"] for r in locality_results]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, s_t1,  "o-",  lw=2, label="same leaf as top-1")
        ax.plot(ks, s_gld, "s--", lw=2, label="same leaf as gold")
        ax.set_xscale("log")
        ax.set_xlabel("K"); ax.set_ylabel("Fraction in same leaf")
        ax.set_title("Logit locality over leaf regions")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "leaf_logit_locality.png"), dpi=150)
        plt.close(fig)

    log.info("Plots → %s/", output_dir)


# ── Step 8: Summary ───────────────────────────────────────────────────────────

def write_summary(
    leaf_to_vocab: Dict[int, List[int]],
    probe_results: List[dict],
    recall_results: List[dict],
    softmax_results: List[dict],
    locality_results: List[dict],
    coarse_recall: Optional[List[dict]],
    output_dir: str,
) -> None:
    n_leaf = len(leaf_to_vocab)
    sizes  = [len(v) for v in leaf_to_vocab.values()]

    lines = ["# Leaf Region Evaluation — Summary\n"]
    lines.append(f"## 1. Leaf regions\n{n_leaf} leaf predictive interference regions  "
                 f"|  sizes min={min(sizes)} mean={np.mean(sizes):.1f} max={max(sizes)}\n")

    if probe_results:
        last = probe_results[-1]
        first = probe_results[0]
        lines.append(f"## 3. Predictability from hidden states")
        lines.append(f"Layer {last['layer']}: top-1={last['top1']:.3f}  top-4={last['top4']:.3f}  "
                     f"top-8={last['top8']:.3f}  top-16={last['top16']:.3f}  "
                     f"(random={last['random_top1']:.4f})")
        lines.append(f"Layer {first['layer']}: top-1={first['top1']:.3f}  "
                     f"top-4={first['top4']:.3f}\n")

    if recall_results:
        lines.append("## 4. Gold recall under top-r leaf routing")
        for r in recall_results:
            lines.append(f"  r={r['r_leaves']:2d}  recall={r['gold_recall']:.3f}  "
                         f"cand={r['avg_cand_size']:.0f} ({100*r['cand_frac_vocab']:.2f}% vocab)  "
                         f"random={r['random_recall']:.3f}")
        lines.append("")

    if coarse_recall:
        lines.append("## 6. Leaf vs coarse routing (candidate size comparison)")
        for lf, cr in zip(recall_results[:len(coarse_recall)], coarse_recall):
            lines.append(f"  r={cr['r_regions']:2d}  coarse_cand={cr['avg_cand']:.0f}  "
                         f"leaf_cand={lf['avg_cand_size']:.0f}  "
                         f"coarse_recall={cr['gold_recall']:.3f}  "
                         f"leaf_recall={lf['gold_recall']:.3f}")
        lines.append("")

    if softmax_results:
        lines.append("## 7. Routed softmax stability")
        for r in softmax_results:
            lines.append(f"  r={r['r_leaves']:2d}  full_ppl={r['full_ppl']:.1f}  "
                         f"routed_ppl={r['routed_ppl']:.1f}  "
                         f"eff_ppl={r['eff_routed_ppl']:.1f}  "
                         f"nll_delta={r['nll_delta']:+.3f}  "
                         f"failures={100*r['failure_rate']:.1f}%")
        lines.append("")

    # Go / no-go
    lines.append("## Go / No-Go\n")
    go = []
    if probe_results and probe_results[-1]["top8"] > 0.90:
        go.append(f"✓ Leaf router top-8 = {probe_results[-1]['top8']:.3f} > 0.90")
    if recall_results:
        for r in recall_results:
            if r["gold_recall"] > 0.80 and r["cand_frac_vocab"] < 0.15:
                go.append(f"✓ recall={r['gold_recall']:.3f} with {100*r['cand_frac_vocab']:.1f}% vocab at r={r['r_leaves']}")
                break
    if softmax_results and softmax_results[-1]["failure_rate"] < 0.05:
        go.append(f"✓ Failure rate {100*softmax_results[-1]['failure_rate']:.1f}% < 5% at r={softmax_results[-1]['r_leaves']}")
    if probe_results:
        rand_base = probe_results[-1]["random_top1"]
        lift = probe_results[-1]["top4"] / rand_base
        if lift > 5:
            go.append(f"✓ Router lift over random = {lift:.1f}x")

    verdict = ("**STRONG GO — leaf routing is architecture-relevant.**"   if len(go) >= 3 else
               "**WEAK GO — promising; tune leaf granularity.**"           if len(go) >= 2 else
               "**NO-GO — leaf regions do not provide useful routing.**")
    lines.append(verdict)
    lines += go

    lines.append("\n## 8. Architecture relevance")
    lines.append("Leaf routing reduces candidate set more than coarse routing. "
                 "If recall is maintained, this validates recursive region hierarchy "
                 "as a routing mechanism: context → coarse region → leaf region → candidate set.")

    path = os.path.join(output_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + "\n")
    log.info("Summary → %s", path)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _save_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    log.info("→ %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args) -> None:
    set_seed(args.seed)
    ensure_dirs(args.output_dir)
    rng    = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load tokenizer (optional, for readable output)
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        pass

    # ── Step 1: Flatten tree ──────────────────────────────────────────────
    log.info("=== Step 1: Flatten region tree to leaves ===")
    if not os.path.exists(args.region_tree_json):
        log.error("region_tree.json not found: %s\n"
                  "Run recursive_subcluster.py first.", args.region_tree_json)
        sys.exit(1)
    tree = load_json(args.region_tree_json)
    leaf_to_vocab, vocab_to_leaf = flatten_tree_to_leaves(tree)
    n_leaves = len(leaf_to_vocab)
    save_leaf_reports(leaf_to_vocab, vocab_to_leaf, tok, args.output_dir)

    # ── Step 2: Build labels ──────────────────────────────────────────────
    log.info("=== Step 2: Build sample labels ===")
    hidden, probe_token_ids, topk_ids, layers, meta = load_probe_cache(args.probe_cache_dir)
    labels, valid = build_leaf_labels(probe_token_ids, vocab_to_leaf)
    n_classes = int(labels[valid].max()) + 1 if valid.any() else 0

    if n_classes < 2:
        log.error("Fewer than 2 leaf classes in probe cache — cannot train probes.")
        sys.exit(1)

    # Restrict to probe_layers arg if specified
    if args.probe_layers:
        layers = [l for l in layers if l in args.probe_layers]
    if not layers:
        log.error("No layers overlap between cache and --probe_layers")
        sys.exit(1)

    # ── Step 3: Train probes ──────────────────────────────────────────────
    log.info("=== Step 3: Train leaf router probes ===")
    probe_results, tr_idx, te_idx, final_probe = train_leaf_probes(
        hidden, labels, valid, n_classes, layers,
        device, rng, epochs=args.probe_epochs,
    )
    _save_csv(probe_results, os.path.join(args.output_dir, "leaf_router_metrics.csv"))

    hidden_final = hidden[layers[-1]]   # (N_probe, D) float16

    # ── Step 4: Gold recall ───────────────────────────────────────────────
    log.info("=== Step 4: Gold recall under top-r leaf routing ===")
    recall_results = gold_recall_top_r(
        final_probe, hidden_final, labels, valid,
        leaf_to_vocab, n_leaves,
        device, rng, te_idx, args.r_values,
    )
    _save_csv(recall_results, os.path.join(args.output_dir, "leaf_gold_recall.csv"))

    # Optional coarse comparison
    coarse_recall_results = None
    coarse_region_path = os.path.join(
        os.path.dirname(args.region_tree_json), "region_to_tokens.json"
    )
    coarse_labels_path = os.path.join(
        os.path.dirname(args.region_tree_json), "token_to_region.json"
    )
    if os.path.exists(coarse_region_path) and os.path.exists(coarse_labels_path):
        log.info("Running coarse-region comparison ...")
        region_to_tokens = load_json(coarse_region_path)
        token_to_region  = load_json(coarse_labels_path)
        n_coarse = max(int(v) for v in token_to_region.values()) + 1
        coarse_raw = np.array([token_to_region.get(str(int(t)), -1)
                                for t in probe_token_ids], dtype=np.int32)
        try:
            coarse_recall_results = coarse_gold_recall(
                probe_token_ids, region_to_tokens, token_to_region,
                te_idx, hidden_final, coarse_raw, n_coarse,
                device, rng, args.r_values[:4], args.probe_epochs,
            )
            _save_csv(coarse_recall_results,
                      os.path.join(args.output_dir, "coarse_vs_leaf_recall.csv"))
        except Exception as e:
            log.warning("Coarse comparison failed: %s", e)

    # ── Step 5: Routed softmax ────────────────────────────────────────────
    softmax_results: List[dict] = []
    if not args.skip_softmax:
        softmax_results = routed_softmax_sim(
            hidden_final, labels, probe_token_ids,
            leaf_to_vocab, n_leaves, te_idx, final_probe,
            args.model_name, device, rng,
            n_eval=args.n_routed_eval, r_values=args.r_values,
        )
        _save_csv(softmax_results, os.path.join(args.output_dir, "leaf_routed_softmax.csv"))

    # ── Step 6: Logit locality ────────────────────────────────────────────
    locality_results = logit_locality(topk_ids, probe_token_ids, vocab_to_leaf, args.logit_ks)
    _save_csv(locality_results, os.path.join(args.output_dir, "leaf_logit_locality.csv"))

    # ── Step 7: Plots ─────────────────────────────────────────────────────
    save_plots(leaf_to_vocab, probe_results, recall_results, softmax_results,
               locality_results, coarse_recall_results, args.output_dir)

    # ── Step 8: Summary ───────────────────────────────────────────────────
    write_summary(leaf_to_vocab, probe_results, recall_results, softmax_results,
                  locality_results, coarse_recall_results, args.output_dir)

    log.info("Leaf evaluation complete → %s/", args.output_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate leaf predictive interference regions as routing units",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--region_tree_json",
                   default="interference_region_pipeline/results/region_tree.json")
    p.add_argument("--graph_dir",
                   default="interference_region_pipeline/graphs")
    p.add_argument("--probe_cache_dir",
                   default="interference_region_pipeline/results/probe_cache")
    p.add_argument("--output_dir",
                   default="interference_region_pipeline/results/leaf_region_eval")
    p.add_argument("--model_name",   default="gpt2-xl")
    p.add_argument("--probe_layers", type=int, nargs="+",
                   default=[0, 4, 8, 12, 16, 20, 24, 47])
    p.add_argument("--probe_epochs", type=int,   default=200)
    p.add_argument("--r_values",     type=int,   nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--logit_ks",     type=int,   nargs="+", default=[10, 50, 100, 500])
    p.add_argument("--n_routed_eval",type=int,   default=5_000)
    p.add_argument("--skip_softmax", action="store_true",
                   help="Skip routed softmax simulation (faster, no model reload needed)")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
