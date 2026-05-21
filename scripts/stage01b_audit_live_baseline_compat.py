#!/usr/bin/env python3
"""
Stage 01B — Live Baseline Compatibility Audit.

Root-cause analysis: the canonical cand-shard h_prime was extracted from
positions 0..seq_len-2 within seq_len=128 batches (using src=batch[:,:-1]).
Stage-01 context windows use ctx_len=256, pushing the prediction position to
position embedding 255 — far outside the trained range → NLL degrades to ~7.74.

This script finds which ctx_len reproduces the cached baseline and selects the
best live-forward convention. Stage 06 must not train until this passes.

Canonical values (from train_bridge_residual_adapter.py):
  full_vocab_base_nll_all     = 3.754938
  full_vocab_base_nll_covered = 3.481444
  gold_in_top256_all          ≈ measured in Part A
  dataset_fingerprint         = 09b0a71955cc9c43
  num_examples                = 239362

Parts:
  A: Recompute cached baseline from cand h_prime @ emb_w.T
  B: Live forward sweep over ctx_lens × hidden positions
  C: Off-by-one / target sanity checks
  D: Hidden-state cosine comparison (h_live vs h_prime)
  E: Decoded examples markdown (20 rows)

Pass condition:
  min |live_nll_all - cached_nll_all| < 0.10
  AND corresponding |live_top256 - cached_top256| < 0.05
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe, get_hs_small
from scripts.train_hard_position_refiner import compute_filter_mask

# ── Canonical targets ─────────────────────────────────────────────────────────

CANONICAL_NLL_ALL     = 3.754938
CANONICAL_NLL_COV     = 3.481444
CANONICAL_MASKED_CAND = 3.378606
CANONICAL_N           = 239362
CANONICAL_N_COV       = 227017
CANONICAL_COV         = 0.948425
CANONICAL_FP          = "09b0a71955cc9c43"
TOP_K                 = 256

FILTER_KWARGS = {"margin_thresh": 0.1, "entropy_thresh": 2.0}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shard_pairs(cand_dir: str, ctx_dir: str) -> List[Tuple[str, str]]:
    cand_paths = sorted([
        os.path.join(cand_dir, f) for f in os.listdir(cand_dir)
        if f.startswith("shard_") and f.endswith(".pt")
    ])
    if not cand_paths:
        raise RuntimeError(f"No shard_*.pt in {cand_dir}")
    pairs = []
    for cp in cand_paths:
        si = int(os.path.basename(cp).replace("shard_", "").replace(".pt", ""))
        xp = os.path.join(ctx_dir, f"shard_{si:05d}.pt")
        if not os.path.exists(xp):
            raise RuntimeError(f"Missing ctx shard: {xp}")
        pairs.append((cp, xp))
    return pairs


def _nll(ce_sum: float, n: int) -> float:
    return ce_sum / max(n, 1)


# ── Part A: Cached baseline from h_prime ─────────────────────────────────────

@torch.no_grad()
def audit_part_a(cand_dir: str, emb_w: torch.Tensor, device,
                 batch_size: int = 64) -> Dict:
    """Recompute the canonical cached baseline from stored h_prime vectors."""
    emb_w = emb_w.float().to(device)
    V     = emb_w.shape[0]
    K     = TOP_K

    ce_all = ce_cov = 0.0
    nc_all = nc_cov = 0
    top1_all = topK_all = 0
    top1_cov = topK_cov = 0
    hp_dim   = None

    cand_paths = sorted([
        os.path.join(cand_dir, f) for f in os.listdir(cand_dir)
        if f.startswith("shard_") and f.endswith(".pt")
    ])

    for cp in cand_paths:
        cs   = torch.load(cp, map_location="cpu", weights_only=True)
        if "h_prime" not in cs:
            raise RuntimeError(f"No h_prime in {cp}. Cannot compute cached baseline.")
        hp   = cs["h_prime"].float()     # (N, d)
        gt   = cs["gold_token"].long()   # (N,)
        cov  = cs["covered"].bool()      # (N,)
        N    = gt.shape[0]
        hp_dim = hp.shape[1]

        gate = compute_filter_mask(cs, "boundary", **FILTER_KWARGS)

        for s in range(0, N, batch_size):
            e  = min(s + batch_size, N)
            h  = hp[s:e].to(device)
            g  = gt[s:e].to(device)
            cv = cov[s:e].to(device)

            lgt  = h @ emb_w.T                  # (B, V)
            B    = lgt.shape[0]

            ce_all += float(F.cross_entropy(lgt, g, reduction="sum"))
            nc_all += B
            t1 = lgt.argmax(1)
            top1_all += int((t1 == g).sum())
            topK_all += int((lgt.topk(K, dim=1).indices == g.unsqueeze(1)).any(1).sum())

            if cv.any():
                lgt_c = lgt[cv]; g_c = g[cv]
                ce_cov += float(F.cross_entropy(lgt_c, g_c, reduction="sum"))
                nc_cov += int(cv.sum())
                top1_cov += int((lgt_c.argmax(1) == g_c).sum())
                topK_cov += int((lgt_c.topk(K, dim=1).indices == g_c.unsqueeze(1)).any(1).sum())

    return {
        "nll_all":           _nll(ce_all, nc_all),
        "nll_covered":       _nll(ce_cov, nc_cov),
        "top1_acc_all":      top1_all / max(nc_all, 1),
        "topK_rate_all":     topK_all / max(nc_all, 1),
        "topK_rate_covered": topK_cov / max(nc_cov, 1),
        "n_all":             nc_all,
        "n_covered":         nc_cov,
        "h_prime_dim":       hp_dim,
    }


# ── Part B: Live forward per convention ───────────────────────────────────────

@torch.no_grad()
def audit_one_convention(
    backbone,
    emb_w:      torch.Tensor,
    pairs:      List[Tuple[str, str]],
    device,
    ctx_len:    int,
    hidden_pos: int,
    batch_size: int = 64,
) -> Dict:
    """
    Compute live NLL for one (ctx_len, hidden_pos) pair.
    input_ids = ctx["input_ids"][:, -ctx_len:]   (last ctx_len tokens)
    h_live = get_hs_small(backbone, input_ids, device)[:, hidden_pos, :]
    """
    emb_w    = emb_w.float().to(device)
    max_pos  = backbone.pos_emb.num_embeddings
    K        = TOP_K

    if ctx_len > max_pos:
        return {"error": f"ctx_len={ctx_len} > max_pos={max_pos}", "skip": True}

    ce_all = ce_cov = 0.0
    nc_all = nc_cov = 0
    top1_all = topK_all = 0
    top1_cov = topK_cov = 0

    for cp, xp in pairs:
        cs = torch.load(cp, map_location="cpu", weights_only=True)
        cx = torch.load(xp, map_location="cpu", weights_only=True)
        gt  = cs["gold_token"].long()
        cov = cs["covered"].bool()
        ids = cx["input_ids"].long()     # (N, stored_ctx_len)
        N   = gt.shape[0]

        for s in range(0, N, batch_size):
            e   = min(s + batch_size, N)
            g   = gt[s:e].to(device)
            cv  = cov[s:e].to(device)
            src = ids[s:e, -ctx_len:].to(device)   # crop to ctx_len

            h_all  = get_hs_small(backbone, src, device)   # (B, T, d)
            h_live = h_all[:, hidden_pos, :].float()        # (B, d)
            lgt    = h_live @ emb_w.T                       # (B, V)

            ce_all += float(F.cross_entropy(lgt, g, reduction="sum"))
            nc_all += (e - s)
            top1_all += int((lgt.argmax(1) == g).sum())
            topK_all += int((lgt.topk(K, dim=1).indices == g.unsqueeze(1)).any(1).sum())

            if cv.any():
                lgt_c = lgt[cv]; g_c = g[cv]
                ce_cov += float(F.cross_entropy(lgt_c, g_c, reduction="sum"))
                nc_cov += int(cv.sum())
                top1_cov += int((lgt_c.argmax(1) == g_c).sum())
                topK_cov += int((lgt_c.topk(K, dim=1).indices == g_c.unsqueeze(1)).any(1).sum())

    return {
        "ctx_len":           ctx_len,
        "hidden_pos":        hidden_pos,
        "nll_all":           _nll(ce_all, nc_all),
        "nll_covered":       _nll(ce_cov, nc_cov),
        "top1_acc_all":      top1_all / max(nc_all, 1),
        "topK_rate_all":     topK_all / max(nc_all, 1),
        "topK_rate_covered": topK_cov / max(nc_cov, 1),
        "n_all":             nc_all,
        "nll_diff_from_canonical": abs(_nll(ce_all, nc_all) - CANONICAL_NLL_ALL),
        "skip": False,
    }


# ── Part C: Off-by-one / target sanity ───────────────────────────────────────

def audit_part_c(pairs: List[Tuple[str, str]], n_check: int = 1000) -> Dict:
    """
    Check whether gold_token is the last token of input_ids (off-by-one bug).
    Also check first few rows of first shard.
    """
    result = {
        "gold_in_ids_last":      0,
        "gold_in_ids_second_last": 0,
        "checked_n":             0,
    }
    for cp, xp in pairs:
        cs  = torch.load(cp, map_location="cpu", weights_only=True)
        cx  = torch.load(xp, map_location="cpu", weights_only=True)
        gt  = cs["gold_token"].long()
        ids = cx["input_ids"].long()    # (N, ctx_len)
        N   = min(n_check, gt.shape[0])

        last   = ids[:N, -1]
        second = ids[:N, -2] if ids.shape[1] >= 2 else ids[:N, -1]
        result["gold_in_ids_last"]       += int((gt[:N] == last).sum())
        result["gold_in_ids_second_last"] += int((gt[:N] == second).sum())
        result["checked_n"]              += N
        if result["checked_n"] >= n_check:
            break

    n = result["checked_n"]
    result["gold_in_ids_last_rate"]       = result["gold_in_ids_last"] / max(n, 1)
    result["gold_in_ids_second_last_rate"] = result["gold_in_ids_second_last"] / max(n, 1)
    result["alignment_suspicious"] = result["gold_in_ids_last_rate"] > 0.5
    return result


# ── Part D: Hidden-state cosine comparison ────────────────────────────────────

@torch.no_grad()
def audit_part_d(
    backbone,
    pairs:      List[Tuple[str, str]],
    device,
    ctx_len:    int,
    hidden_pos: int = -1,
    n_compare:  int = 2000,
    batch_size: int = 64,
) -> Dict:
    """Compare h_live to cand h_prime. Reports cosine sim and L2 distance."""
    max_pos = backbone.pos_emb.num_embeddings
    if ctx_len > max_pos:
        return {"error": f"ctx_len={ctx_len} > max_pos={max_pos}", "skip": True}

    cos_sims  = []
    l2_dists  = []
    h_live_norms = []
    h_prime_norms = []
    collected = 0

    for cp, xp in pairs:
        if collected >= n_compare:
            break
        cs  = torch.load(cp, map_location="cpu", weights_only=True)
        cx  = torch.load(xp, map_location="cpu", weights_only=True)
        if "h_prime" not in cs:
            return {"error": "h_prime not in cand shard", "skip": True}

        hp  = cs["h_prime"].float()
        ids = cx["input_ids"].long()
        N   = min(hp.shape[0], n_compare - collected)

        for s in range(0, N, batch_size):
            e   = min(s + batch_size, N)
            src = ids[s:e, -ctx_len:].to(device)
            h_all  = get_hs_small(backbone, src, device)
            h_live = h_all[:, hidden_pos, :].float().cpu()   # (B, d)
            h_prim = hp[s:e]                                  # (B, d_hp)

            if h_live.shape[1] != h_prim.shape[1]:
                return {"error": f"dim mismatch: h_live {h_live.shape[1]} vs h_prime {h_prim.shape[1]}",
                        "h_live_dim": h_live.shape[1], "h_prime_dim": h_prim.shape[1], "skip": True}

            # Cosine similarity per row
            h_l_n = F.normalize(h_live, dim=1)
            h_p_n = F.normalize(h_prim, dim=1)
            cos   = (h_l_n * h_p_n).sum(1)                  # (B,)
            l2    = (h_live - h_prim).norm(dim=1)            # (B,)
            cos_sims.append(cos.numpy())
            l2_dists.append(l2.numpy())
            h_live_norms.append(h_live.norm(dim=1).numpy())
            h_prime_norms.append(h_prim.norm(dim=1).numpy())

        collected += N

    cos_sims  = np.concatenate(cos_sims)
    l2_dists  = np.concatenate(l2_dists)
    hlive_n   = np.concatenate(h_live_norms)
    hprime_n  = np.concatenate(h_prime_norms)

    return {
        "ctx_len":        ctx_len,
        "hidden_pos":     hidden_pos,
        "n_compared":     len(cos_sims),
        "cos_mean":       float(cos_sims.mean()),
        "cos_median":     float(np.median(cos_sims)),
        "cos_p10":        float(np.percentile(cos_sims, 10)),
        "cos_p90":        float(np.percentile(cos_sims, 90)),
        "l2_mean":        float(l2_dists.mean()),
        "h_live_norm_mean":  float(hlive_n.mean()),
        "h_prime_norm_mean": float(hprime_n.mean()),
        "skip": False,
    }


# ── Part E: Decoded examples ──────────────────────────────────────────────────

@torch.no_grad()
def audit_part_e(
    backbone,
    emb_w:      torch.Tensor,
    pairs:      List[Tuple[str, str]],
    device,
    ctx_len_best: int,
    cached_nll_all: float,
    n_examples:  int = 20,
    batch_size:  int = 64,
) -> List[Dict]:
    """Generate decoded example comparison rows for the markdown report."""
    from transformers import GPT2TokenizerFast
    try:
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.model_max_length = int(1e30)
    except Exception:
        return []

    emb_w  = emb_w.float().to(device)
    K_show = 5
    rows   = []

    for cp, xp in pairs:
        if len(rows) >= n_examples:
            break
        cs   = torch.load(cp, map_location="cpu", weights_only=True)
        cx   = torch.load(xp, map_location="cpu", weights_only=True)
        si   = int(os.path.basename(cp).replace("shard_", "").replace(".pt", ""))
        gt   = cs["gold_token"].long()
        ids  = cx["input_ids"].long()
        hp   = cs["h_prime"].float() if "h_prime" in cs else None
        N    = min(gt.shape[0], n_examples - len(rows))
        ri   = cs["row_id"] if "row_id" in cs else torch.arange(gt.shape[0])

        for s in range(0, N, batch_size):
            if len(rows) >= n_examples:
                break
            e = min(s + batch_size, N)
            g   = gt[s:e].to(device)
            src_live = ids[s:e, -ctx_len_best:].to(device)

            h_all  = get_hs_small(backbone, src_live, device)
            h_live = h_all[:, -1, :].float()
            live_lgt = h_live @ emb_w.T                # (B, V)

            cached_lgt = None
            if hp is not None:
                cached_lgt = hp[s:e].to(device) @ emb_w.T  # (B, V)

            for i in range(e - s):
                g_tok   = int(gt[s + i].item())
                row_id  = int(ri[s + i].item())
                tail_ids = ids[s + i, -30:].tolist()
                tail_ids = [t for t in tail_ids if t != 0]  # strip padding

                # Live stats
                live_probs  = F.softmax(live_lgt[i], dim=0)
                live_top5v, live_top5i = live_probs.topk(K_show)
                live_rank   = int((live_lgt[i] > live_lgt[i, g_tok]).sum().item()) + 1

                # Cached stats
                if cached_lgt is not None:
                    cac_probs   = F.softmax(cached_lgt[i], dim=0)
                    cac_top5v, cac_top5i = cac_probs.topk(K_show)
                    cac_rank    = int((cached_lgt[i] > cached_lgt[i, g_tok]).sum().item()) + 1
                else:
                    cac_top5v = cac_top5i = None
                    cac_rank  = -1

                def _decode(t_list):
                    try:
                        return tok.decode(t_list, skip_special_tokens=False)
                    except Exception:
                        return str(t_list)

                rows.append({
                    "shard_id":     si,
                    "row_in_shard": s + i,
                    "row_id":       row_id,
                    "input_tail":   _decode(tail_ids),
                    "gold_token":   g_tok,
                    "gold_str":     _decode([g_tok]),
                    "live_rank":    live_rank,
                    "cached_rank":  cac_rank,
                    "live_top5":    [(int(live_top5i[j].item()),
                                      _decode([int(live_top5i[j].item())]),
                                      float(live_top5v[j].item()))
                                     for j in range(K_show)],
                    "cached_top5":  ([(int(cac_top5i[j].item()),
                                       _decode([int(cac_top5i[j].item())]),
                                       float(cac_top5v[j].item()))
                                      for j in range(K_show)]
                                     if cac_top5i is not None else []),
                    "ctx_len_used": ctx_len_best,
                })
                if len(rows) >= n_examples:
                    break

    return rows


# ── Markdown writers ──────────────────────────────────────────────────────────

def write_examples_md(rows: List[Dict], path: str) -> None:
    lines = ["# Live vs Cached Token Prediction Examples\n"]
    for r in rows:
        lines.append(f"## Row {r['row_id']}  (shard {r['shard_id']}, pos {r['row_in_shard']})\n")
        lines.append(f"**Context tail:** `{r['input_tail']}`\n\n")
        lines.append(f"**Gold:** `{r['gold_str']!r}` (id {r['gold_token']})  "
                     f"live_rank={r['live_rank']}  cached_rank={r['cached_rank']}  "
                     f"ctx_len={r['ctx_len_used']}\n\n")
        if r["cached_top5"]:
            lines.append("| Rank | Cached token | Cached prob | Live token | Live prob |\n")
            lines.append("|------|--------------|-------------|------------|-----------|\n")
            for j in range(len(r["cached_top5"])):
                ct = r["cached_top5"][j]
                lt = r["live_top5"][j]
                lines.append(f"| {j+1} | `{ct[1]!r}` | {ct[2]:.4f} | `{lt[1]!r}` | {lt[2]:.4f} |\n")
        else:
            lines.append("| Rank | Live token | Live prob |\n")
            lines.append("|------|------------|-----------|\n")
            for j, lt in enumerate(r["live_top5"]):
                lines.append(f"| {j+1} | `{lt[1]!r}` | {lt[2]:.4f} |\n")
        lines.append("\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def write_audit_md(report: Dict, path: str) -> None:
    r = report
    lines = [
        "# Live Baseline Compatibility Audit\n\n",
        f"**identity_pass**: {r['live_baseline_compatible']}\n\n",
        "## Part A — Cached Baseline (h_prime)\n\n",
        f"| Metric | Value | Canonical |\n|--------|-------|----------|\n",
        f"| NLL all | {r['part_a']['nll_all']:.6f} | {CANONICAL_NLL_ALL} |\n",
        f"| NLL covered | {r['part_a']['nll_covered']:.6f} | {CANONICAL_NLL_COV} |\n",
        f"| top1 acc | {r['part_a']['top1_acc_all']:.4f} | — |\n",
        f"| top256 rate | {r['part_a']['topK_rate_all']:.4f} | — |\n",
        f"| h_prime dim | {r['part_a']['h_prime_dim']} | — |\n\n",
        "## Part B — Live Forward Conventions\n\n",
        "| ctx_len | hidden_pos | NLL_all | NLL_cov | top256 | diff_from_canonical |\n",
        "|---------|-----------|---------|---------|--------|--------------------|\n",
    ]
    for c in r.get("part_b_conventions", []):
        if c.get("skip"):
            lines.append(f"| {c.get('ctx_len','?')} | {c.get('hidden_pos','?')} | SKIP | | | |\n")
        else:
            lines.append(f"| {c['ctx_len']} | {c['hidden_pos']} | "
                         f"{c['nll_all']:.6f} | {c['nll_covered']:.6f} | "
                         f"{c['topK_rate_all']:.4f} | {c['nll_diff_from_canonical']:.4f} |\n")
    lines.append("\n")
    if r.get("part_c"):
        c3 = r["part_c"]
        lines.append("## Part C — Off-by-one Check\n\n")
        lines.append(f"gold_in_ids_last_rate = {c3['gold_in_ids_last_rate']:.4f}  "
                     f"(>0.5 → alignment bug)\n\n")
        lines.append(f"gold_in_ids_second_last_rate = {c3['gold_in_ids_second_last_rate']:.4f}\n\n")
        if c3["alignment_suspicious"]:
            lines.append("**WARNING: Gold token appears to be inside input_ids! "
                         "Off-by-one bug in Stage 01.**\n\n")
    if r.get("best_convention"):
        bc = r["best_convention"]
        lines.append("## Best Convention\n\n")
        lines.append(f"```json\n{json.dumps(bc, indent=2)}\n```\n\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load backbone ─────────────────────────────────────────────────────────
    print(f"[stage01b] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    max_pos = backbone.pos_emb.num_embeddings
    print(f"  d_model={d_model}  vocab_size={vocab_size}  max_pos_emb={max_pos}")

    if hasattr(backbone, "token_emb"):
        emb_w = backbone.token_emb.weight.detach().float().cpu()  # (V, d)
    else:
        raise RuntimeError("Cannot locate token_emb in backbone")

    # ── Parse test ctx_lens ───────────────────────────────────────────────────
    test_ctx_lens = [int(x) for x in args.test_ctx_lens.split(",") if x.strip()]
    test_ctx_lens = [c for c in test_ctx_lens if c <= max_pos]
    if not test_ctx_lens:
        raise RuntimeError(f"No valid ctx_len <= max_pos={max_pos}")
    print(f"  Testing ctx_lens: {test_ctx_lens}")

    # ── Pair shards ───────────────────────────────────────────────────────────
    pairs = _shard_pairs(args.val_cand_dir, args.val_ctx_dir)
    print(f"  {len(pairs)} shard pairs")

    report = {
        "backbone_max_pos": max_pos,
        "d_model":          d_model,
        "vocab_size":       vocab_size,
        "canonical_nll_all":     CANONICAL_NLL_ALL,
        "canonical_nll_covered": CANONICAL_NLL_COV,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Part A: Cached baseline from h_prime
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[stage01b] Part A: Cached baseline from h_prime ...")
    pa = audit_part_a(args.val_cand_dir, emb_w.to(device), device,
                      args.eval_batch_size)
    report["part_a"] = pa
    print(f"  h_prime dim       = {pa['h_prime_dim']}")
    print(f"  cached_nll_all    = {pa['nll_all']:.6f}  (canonical {CANONICAL_NLL_ALL})")
    print(f"  cached_nll_cov    = {pa['nll_covered']:.6f}  (canonical {CANONICAL_NLL_COV})")
    print(f"  cached_top256     = {pa['topK_rate_all']:.4f}")
    print(f"  cached_top1_acc   = {pa['top1_acc_all']:.4f}")
    print(f"  n_all={pa['n_all']}  n_covered={pa['n_covered']}")

    part_a_ok = (abs(pa["nll_all"] - CANONICAL_NLL_ALL) < 0.1 and
                 abs(pa["n_all"] - CANONICAL_N) / max(CANONICAL_N, 1) < 0.02)
    if not part_a_ok:
        print(f"  WARNING: Part A NLL {pa['nll_all']:.4f} far from canonical {CANONICAL_NLL_ALL}")
        print(f"  This may indicate the cand shards changed or h_prime is from a different model.")
    report["part_a_ok"] = part_a_ok
    cached_top256 = pa["topK_rate_all"]

    # ═══════════════════════════════════════════════════════════════════════
    # Part B: Live forward sweep
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[stage01b] Part B: Live forward sweep ...")
    hidden_positions = [-1, -2]
    conventions = []
    best_conv   = None
    best_diff   = float("inf")

    for ctx_len in test_ctx_lens:
        for hp in hidden_positions:
            print(f"  ctx_len={ctx_len}  hidden_pos={hp} ...", end="  ", flush=True)
            t0 = time.time()
            c = audit_one_convention(
                backbone, emb_w.to(device), pairs, device,
                ctx_len=ctx_len, hidden_pos=hp,
                batch_size=args.eval_batch_size,
            )
            conventions.append(c)
            if c.get("skip"):
                print(f"SKIP ({c.get('error','')})")
                continue
            diff = c["nll_diff_from_canonical"]
            print(f"nll={c['nll_all']:.4f}  top256={c['topK_rate_all']:.4f}"
                  f"  diff={diff:.4f}  t={time.time()-t0:.0f}s")
            if diff < best_diff:
                best_diff = diff
                best_conv = c

    report["part_b_conventions"] = conventions

    # ═══════════════════════════════════════════════════════════════════════
    # Part C: Off-by-one check
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[stage01b] Part C: Off-by-one sanity check ...")
    pc = audit_part_c(pairs)
    report["part_c"] = pc
    print(f"  gold_in_ids_last_rate       = {pc['gold_in_ids_last_rate']:.4f}")
    print(f"  gold_in_ids_second_last_rate = {pc['gold_in_ids_second_last_rate']:.4f}")
    if pc["alignment_suspicious"]:
        print("  WARNING: Gold looks like it's inside input_ids (off-by-one)!")

    # ═══════════════════════════════════════════════════════════════════════
    # Part D: Hidden-state comparison for best convention
    # ═══════════════════════════════════════════════════════════════════════
    best_ctx = best_conv["ctx_len"] if best_conv else test_ctx_lens[0]
    best_hpos = best_conv["hidden_pos"] if best_conv else -1

    print(f"\n[stage01b] Part D: Hidden-state comparison (ctx_len={best_ctx}, hpos={best_hpos}) ...")
    pd = audit_part_d(backbone, pairs, device,
                      ctx_len=best_ctx, hidden_pos=best_hpos,
                      n_compare=2000, batch_size=args.eval_batch_size)
    report["part_d"] = pd
    if pd.get("skip"):
        print(f"  SKIP: {pd.get('error','')}")
    else:
        print(f"  cos_mean={pd['cos_mean']:.4f}  cos_median={pd['cos_median']:.4f}"
              f"  l2_mean={pd['l2_mean']:.4f}")
        print(f"  h_live_norm={pd['h_live_norm_mean']:.4f}  h_prime_norm={pd['h_prime_norm_mean']:.4f}")

    # Also run Part D for current (broken) ctx_len=256 for contrast
    if 256 in test_ctx_lens and 256 != best_ctx and 256 <= max_pos:
        print(f"  (contrast: ctx_len=256 for comparison)")
        pd256 = audit_part_d(backbone, pairs, device,
                             ctx_len=256, hidden_pos=-1,
                             n_compare=500, batch_size=args.eval_batch_size)
        report["part_d_ctx256"] = pd256
        if not pd256.get("skip"):
            print(f"  ctx256: cos_mean={pd256['cos_mean']:.4f}  l2_mean={pd256['l2_mean']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Part E: Decoded examples
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[stage01b] Part E: Decoded examples (n={args.num_examples_to_print}) ...")
    examples = audit_part_e(
        backbone, emb_w.to(device), pairs, device,
        ctx_len_best=best_ctx,
        cached_nll_all=pa["nll_all"],
        n_examples=args.num_examples_to_print,
        batch_size=args.eval_batch_size,
    )
    ex_path = os.path.join(args.output_dir, "examples_live_vs_cached.md")
    write_examples_md(examples, ex_path)
    print(f"  Examples → {ex_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Determine pass/fail and best convention
    # ═══════════════════════════════════════════════════════════════════════
    if best_conv is not None:
        top256_diff = abs(best_conv["topK_rate_all"] - cached_top256)
        nll_diff    = best_conv["nll_diff_from_canonical"]
        compatible  = (nll_diff < 0.10 and top256_diff < 0.05)
    else:
        compatible  = False
        nll_diff    = float("inf")
        top256_diff = float("inf")

    best_convention_dict = None
    if best_conv is not None:
        best_convention_dict = {
            "ctx_len":             best_conv["ctx_len"],
            "hidden_position":     best_conv["hidden_pos"],
            "apply_final_norm":    True,
            "logits_source":       "manual_lm_head",
            "live_base_nll_all":   best_conv["nll_all"],
            "live_base_nll_cov":   best_conv["nll_covered"],
            "live_top256_all":     best_conv["topK_rate_all"],
            "cached_base_nll_all": pa["nll_all"],
            "nll_diff_from_canonical": nll_diff,
            "top256_diff_from_cached": top256_diff,
        }

    report.update({
        "live_baseline_compatible":   compatible,
        "best_nll_diff":              nll_diff if best_conv else None,
        "best_top256_diff":           top256_diff if best_conv else None,
        "best_convention":            best_convention_dict,
        "total_seconds":              time.time() - t_start,
        "diagnosis": _diagnose(test_ctx_lens, conventions, max_pos, pc),
    })

    # ═══════════════════════════════════════════════════════════════════════
    # Write per-convention CSV
    # ═══════════════════════════════════════════════════════════════════════
    csv_path = os.path.join(args.output_dir, "per_convention_metrics.csv")
    fieldnames = ["ctx_len", "hidden_pos", "nll_all", "nll_covered",
                  "topK_rate_all", "nll_diff_from_canonical", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(conventions)
    print(f"\n  CSV → {csv_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Write JSON report
    # ═══════════════════════════════════════════════════════════════════════
    json_path = os.path.join(args.output_dir, "live_baseline_audit.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON → {json_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Write markdown report
    # ═══════════════════════════════════════════════════════════════════════
    md_path = os.path.join(args.output_dir, "live_baseline_audit.md")
    write_audit_md(report, md_path)
    print(f"  MD   → {md_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[stage01b] ══ SUMMARY ══")
    print(f"  backbone max_pos_emb   = {max_pos}")
    print(f"  cached_nll_all         = {pa['nll_all']:.6f}  (canonical {CANONICAL_NLL_ALL})")
    if best_conv:
        print(f"  best live convention   = ctx_len={best_ctx}  hidden_pos={best_hpos}")
        print(f"  best live_nll_all      = {best_conv['nll_all']:.6f}")
        print(f"  best nll_diff          = {nll_diff:.4f}  (pass < 0.10)")
        print(f"  top256_diff            = {top256_diff:.4f}  (pass < 0.05)")
    else:
        print("  No working convention found!")
    print(f"  live_baseline_compatible = {compatible}")

    diag = report.get("diagnosis", {})
    if diag:
        print(f"\n[stage01b] Diagnosis: {diag.get('likely_cause', 'unknown')}")
        print(f"  {diag.get('recommendation', '')}")

    if not compatible and args.fail_if_incompatible:
        raise RuntimeError(
            "Live baseline compatibility failed. Do not train Stage 06. "
            f"Best live NLL={best_conv['nll_all'] if best_conv else 'N/A'}, "
            f"canonical={CANONICAL_NLL_ALL}. "
            f"See {json_path}"
        )

    print("\n[stage01b] " + ("PASS" if compatible else "FAIL (not failing script)"))


def _diagnose(test_ctx_lens, conventions, max_pos, part_c):
    """Produce a text diagnosis based on the convention results."""
    nlls = {(c["ctx_len"], c["hidden_pos"]): c.get("nll_all", float("inf"))
            for c in conventions if not c.get("skip")}
    if not nlls:
        return {"likely_cause": "No valid conventions tested", "recommendation": ""}

    # Check if short contexts are much better than long ones
    short_best = min((v for (c, h), v in nlls.items() if c <= 128),
                     default=float("inf"))
    long_nll   = nlls.get((256, -1), float("inf"))

    if part_c.get("alignment_suspicious"):
        return {
            "likely_cause": "OFF-BY-ONE: gold_token is inside input_ids. Stage 01 context convention is wrong.",
            "recommendation": "Patch stage01: context must end BEFORE gold_token position.",
        }

    if max_pos < 256:
        return {
            "likely_cause": f"POSITION OVERFLOW: backbone max_pos_emb={max_pos} < ctx_len=256. "
                            "Stage 01 must use ctx_len <= backbone max_pos_emb.",
            "recommendation": f"Set --ctx_len {max_pos} (or smaller) in Stage 01.",
        }

    if long_nll > short_best + 1.0 and short_best < CANONICAL_NLL_ALL + 0.5:
        best_ctx = min((c for (c, h), v in nlls.items() if v == short_best),
                       default=test_ctx_lens[0])
        return {
            "likely_cause": f"POSITION EMBEDDING MISMATCH: model trained on seq_len ~128, "
                            f"but ctx_len=256 puts last token at untrained position 255. "
                            f"ctx_len={best_ctx} works (NLL={short_best:.4f}), ctx_len=256 broken (NLL={long_nll:.4f}).",
            "recommendation": f"Rebuild Stage 01 with --ctx_len {best_ctx}. "
                               f"Also update all downstream stages to use ctx_len={best_ctx}.",
        }

    best_v = min(nlls.values())
    if abs(best_v - CANONICAL_NLL_ALL) < 0.1:
        return {
            "likely_cause": "Compatible convention found.",
            "recommendation": "Use the best convention in Stage 06.",
        }

    return {
        "likely_cause": "Unknown. All tested contexts give NLL far from canonical. "
                        "Possibly wrong backbone checkpoint or h_prime from a different model.",
        "recommendation": "Check that --small_ckpt is the same checkpoint used to generate cand shards.",
    }


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",          required=True)
    p.add_argument("--val_cand_dir",        required=True)
    p.add_argument("--val_ctx_dir",         required=True)
    p.add_argument("--baseline_json",       default=None)
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--test_ctx_lens",       default="32,64,128,127,256",
                   help="Comma-separated context lengths to test")
    p.add_argument("--eval_batch_size",     type=int, default=64)
    p.add_argument("--num_examples_to_print", type=int, default=20)
    p.add_argument("--fail_if_incompatible", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
