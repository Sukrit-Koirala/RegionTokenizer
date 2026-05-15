#!/usr/bin/env python3
"""
Phase 1 — Path-refiner masked-softmax candidate evaluation.

Measures masked NLL, coverage, and fallback rate for a set of candidate
selection policies on the validation set, using:
  - frozen backbone forward pass for LM logits + gold token IDs
  - per_position.npz for region selections (no kNN re-run)

Policies evaluated (using stored router / mem topk from per_position.npz):
  router_top8, router_top12, router_top16, router_top24
  union_r8m4, union_r12m4, union_r16m8

Per-split breakdown: core (r_mg>=0.30), medium (0.10-0.30),
                     boundary (<0.10), tight_boundary (<0.03)

Metrics per (policy, split):
  full_lm_nll           — NLL under frozen base model (no masking)
  strict_masked_nll     — NLL after masking (only on covered positions)
  fallback_nll          — masked NLL if covered, else full NLL
  gold_token_coverage   — fraction where gold token is in candidate set
  gold_region_coverage  — fraction where gold region is in candidate set
  fallback_rate         — 1 - coverage
  avg_candidate_tokens  — mean |candidate set|
  mapped_vocab_pct      — avg_candidate_tokens / vocab_size * 100

Outputs (in --output_dir):
  phase1_policy_summary.csv    per-policy overall metrics
  phase1_policy_detail.csv     one row per (policy, split)

Usage:
    python scripts/eval_path_refiner_masked_softmax.py \\
        --small_ckpt   runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \\
        --knn_run_dir  runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20 \\
        --region_map   runs/region_maps_128/token_to_region.json \\
        --output_dir   runs/path_refiner \\
        --device       cuda
"""

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import (
    TokenChunkDataset, load_region_map, build_inverse_map,
    tokens_per_region_tensor, load_wikitext,
    load_small_backbone_and_probe, get_hs_small,
)

# ── Policy definitions ────────────────────────────────────────────────────────

POLICIES: List[Tuple[str, Dict]] = [
    ("router_top8",  {"type": "router", "K": 8}),
    ("router_top12", {"type": "router", "K": 12}),
    ("router_top16", {"type": "router", "K": 16}),
    ("router_top24", {"type": "router", "K": 24}),
    ("union_r8m4",   {"type": "union",  "Kr": 8,  "Km": 4}),
    ("union_r12m4",  {"type": "union",  "Kr": 12, "Km": 4}),
    ("union_r16m8",  {"type": "union",  "Kr": 16, "Km": 8}),
]

SPLIT_NAMES = {0: "core", 1: "medium", 2: "boundary", 3: "tight_boundary"}


# ── Token mask construction ───────────────────────────────────────────────────

def build_region_tokens_tensor(inv_map: Dict[int, List[int]],
                                n_coarse: int) -> torch.Tensor:
    """(n_coarse, max_tpr) long, -1=padding."""
    max_len = max((len(v) for v in inv_map.values()), default=1)
    rt = torch.full((n_coarse, max_len), -1, dtype=torch.long)
    for r, toks in inv_map.items():
        rt[r, :len(toks)] = torch.tensor(toks, dtype=torch.long)
    return rt


def selected_to_token_mask(sel_regs: np.ndarray,
                            region_tokens: torch.Tensor,
                            vocab_size: int,
                            device: torch.device) -> torch.Tensor:
    """
    sel_regs: (N, K_sel) int16/int32, negative=padding
    Returns bool (N, vocab_size): True where token is in candidate set.
    """
    N      = sel_regs.shape[0]
    sel_t  = torch.from_numpy(sel_regs.astype(np.int64)).to(device)  # (N, K)
    valid  = sel_t >= 0
    safe   = sel_t.clamp(min=0)
    rt_dev = region_tokens.to(device)

    tok_ids = rt_dev[safe]                                # (N, K, max_tpr)
    tok_ids[~valid.unsqueeze(-1).expand_as(tok_ids)] = -1
    tok_ids = tok_ids.reshape(N, -1)                      # (N, K*max_tpr)

    mask     = torch.zeros(N, vocab_size, dtype=torch.bool, device=device)
    valid_t  = tok_ids >= 0
    safe_ids = tok_ids.clamp(min=0)
    mask.scatter_(1, safe_ids * valid_t.long(), valid_t)
    return mask


# ── Policy region selection ───────────────────────────────────────────────────

def select_regions(pp: Dict[str, np.ndarray], start: int, n: int,
                   cfg: Dict) -> np.ndarray:
    """Return (n, K_sel) int16 of fine-region IDs from per_position.npz slice."""
    if cfg["type"] == "router":
        return pp["router_topk_regions"][start:start + n, :cfg["K"]]
    elif cfg["type"] == "union":
        r = pp["router_topk_regions"][start:start + n, :cfg["Kr"]]
        m = pp["mem_topk_regions"][start:start + n, :cfg["Km"]]
        return np.concatenate([r, m], axis=1)
    raise ValueError(f"Unknown policy type: {cfg['type']!r}")


# ── CSV helper ────────────────────────────────────────────────────────────────

def _write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen: set = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
        w.writeheader(); w.writerows(rows)
    print(f"[phase1] wrote {path}  ({len(rows)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eval(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[phase1] loading checkpoint: {args.small_ckpt}")
    backbone, probe, d_model, cfg_dict, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # ── Load region map ───────────────────────────────────────────────────────
    print(f"[phase1] loading region map: {args.region_map}")
    coarse_map, n_coarse = load_region_map(args.region_map, vocab_size)
    inv_map       = build_inverse_map(coarse_map, n_coarse)
    region_tokens = build_region_tokens_tensor(inv_map, n_coarse)   # (n_coarse, max_tpr)

    # ── Load per_position.npz ─────────────────────────────────────────────────
    for fname in ("per_position.npz", "per_position_topk.npz"):
        pp_path = os.path.join(args.knn_run_dir, fname)
        if os.path.isfile(pp_path):
            break
    print(f"[phase1] loading per_position: {pp_path}")
    pp   = dict(np.load(pp_path))
    N_pp = len(pp["gold_region"])
    print(f"[phase1] per_position N={N_pp:,}")
    for key in ("router_topk_regions", "mem_topk_regions", "split"):
        if key not in pp:
            raise RuntimeError(f"per_position.npz missing key: {key!r}")

    # ── Load val data ─────────────────────────────────────────────────────────
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30)
    val_tokens  = load_wikitext(args.dataset, tokenizer, "validation")
    safe_seq    = min(args.seq_len, backbone.pos_emb.num_embeddings)
    val_ds      = TokenChunkDataset(val_tokens, safe_seq)
    val_loader  = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, drop_last=False)
    max_batches = math.ceil(args.max_positions / (args.batch_size * safe_seq))

    # ── Pass 1: collect logits + gold tokens + split (single model forward) ───
    print("[phase1] pass 1 — collecting LM logits ...")
    t0           = time.time()
    lm_logits_buf: List[np.ndarray] = []  # float16 (n_valid_chunk, vocab_size)
    gold_tok_buf:  List[np.ndarray] = []  # int32   (n_valid_chunk,)
    split_buf:     List[np.ndarray] = []  # uint8   (n_valid_chunk,)
    full_nll_buf:  List[np.ndarray] = []  # float32 (n_valid_chunk,)
    pp_offset = 0

    for bi, batch in enumerate(val_loader):
        if bi >= max_batches or pp_offset >= N_pp:
            break
        batch = batch.to(device)
        src   = batch[:, :-1]
        tgt   = batch[:, 1:]
        if src.size(1) > safe_seq:
            src = src[:, :safe_seq]; tgt = tgt[:, :safe_seq]

        gold_regions = coarse_map[tgt.cpu()].reshape(-1)  # (B*T,) on cpu
        valid        = (gold_regions >= 0).numpy()         # bool (B*T,)
        n_valid      = int(valid.sum())
        if n_valid == 0:
            continue

        # Cap to remaining pp entries
        if pp_offset + n_valid > N_pp:
            # truncate: keep only first (N_pp - pp_offset) valid entries
            need    = N_pp - pp_offset
            cumsum  = np.cumsum(valid)
            cutoff  = np.searchsorted(cumsum, need + 1)  # first index where we'd exceed
            valid[cutoff:] = False
            n_valid = int(valid.sum())

        valid_t = torch.from_numpy(valid).to(device)

        with torch.no_grad():
            if hasattr(backbone, "retrieval_proj"):
                lm_logits_bt, _, _, _ = backbone(src)
            else:
                h = get_hs_small(backbone, src, device)
                lm_logits_bt = backbone.lm_head(h)
            lm_flat = lm_logits_bt.reshape(-1, vocab_size)  # (B*T, V)

        gold_tok_flat = tgt.reshape(-1).to(device)          # (B*T,)
        lm_valid      = lm_flat[valid_t]                    # (n_valid, V)
        gt_valid      = gold_tok_flat[valid_t]              # (n_valid,)

        full_nll_v = F.cross_entropy(lm_valid.float(), gt_valid,
                                     reduction="none").cpu().numpy().astype(np.float32)

        lm_logits_buf.append(lm_valid.cpu().float().numpy().astype(np.float16))
        gold_tok_buf.append(gt_valid.cpu().numpy().astype(np.int32))
        split_buf.append(pp["split"][pp_offset: pp_offset + n_valid])
        full_nll_buf.append(full_nll_v)
        pp_offset += n_valid

    N_eval       = pp_offset
    lm_logits_np = np.concatenate(lm_logits_buf, axis=0)  # (N_eval, vocab_size) float16
    gold_tok_np  = np.concatenate(gold_tok_buf,  axis=0)  # (N_eval,) int32
    split_np     = np.concatenate(split_buf,     axis=0)  # (N_eval,) uint8
    full_nll_np  = np.concatenate(full_nll_buf,  axis=0)  # (N_eval,) float32
    print(f"[phase1] collected N_eval={N_eval:,}  ({time.time()-t0:.1f}s)")

    # ── Pass 2: evaluate each policy ─────────────────────────────────────────
    CHUNK = 1024  # bound GPU memory: 1024 * vocab_size * 4 bytes ≈ 200 MB
    all_summary: List[Dict] = []
    all_detail:  List[Dict] = []

    for pol_name, pol_cfg in POLICIES:
        print(f"[phase1] policy={pol_name}", end="  ", flush=True)
        t_p = time.time()

        covered_list: List[bool]  = []
        n_cand_list:  List[int]   = []
        strict_list:  List[float] = []
        fallbk_list:  List[float] = []

        for cs in range(0, N_eval, CHUNK):
            ce = min(cs + CHUNK, N_eval)
            cn = ce - cs

            sel = select_regions(pp, cs, cn, pol_cfg)  # (cn, K_sel) int16
            tok_mask = selected_to_token_mask(sel, region_tokens, vocab_size, device)
            # (cn, vocab_size) bool on device

            logits_c  = torch.from_numpy(
                lm_logits_np[cs:ce].astype(np.float32)).to(device)  # (cn, V)
            gold_tok_c = torch.from_numpy(
                gold_tok_np[cs:ce].astype(np.int64)).to(device)     # (cn,)
            full_nll_c = torch.from_numpy(full_nll_np[cs:ce]).to(device)  # (cn,)

            # Token-level coverage
            covered_c = tok_mask[torch.arange(cn, device=device), gold_tok_c]  # (cn,) bool
            n_cand_c  = tok_mask.sum(-1).cpu()

            # Masked logits: non-candidates → -inf
            masked_c = logits_c.clone()
            masked_c[~tok_mask] = float("-inf")

            # Strict masked NLL (only where covered; uncovered → +inf)
            strict_c = F.cross_entropy(masked_c.float(), gold_tok_c,
                                        reduction="none")       # (cn,)
            strict_c[~covered_c] = float("inf")

            # Fallback NLL: use masked if covered, else full
            fallbk_c = full_nll_c.clone()
            fallbk_c[covered_c] = strict_c[covered_c]

            covered_list.extend(covered_c.cpu().tolist())
            n_cand_list.extend(n_cand_c.tolist())
            strict_list.extend(strict_c.cpu().tolist())
            fallbk_list.extend(fallbk_c.cpu().tolist())

        covered_np  = np.array(covered_list,  dtype=bool)
        n_cand_np   = np.array(n_cand_list,   dtype=np.float32)
        strict_np   = np.array(strict_list,   dtype=np.float64)
        fallbk_np   = np.array(fallbk_list,   dtype=np.float64)

        def _row(label: str, sel_mask: np.ndarray) -> Optional[Dict]:
            N = int(sel_mask.sum())
            if N == 0:
                return None
            cov_sel     = covered_np[sel_mask]
            nc_sel      = n_cand_np[sel_mask]
            strict_sel  = strict_np[sel_mask]
            fallbk_sel  = fallbk_np[sel_mask]
            full_sel    = full_nll_np[sel_mask]
            cov_rate    = float(cov_sel.mean())
            # strict NLL only on covered positions (exclude +inf)
            finite_mask = np.isfinite(strict_sel)
            strict_mean = float(strict_sel[finite_mask].mean()) if finite_mask.any() else float("nan")
            return {
                "policy":               pol_name,
                "split":                label,
                "n":                    N,
                "full_lm_nll":         float(full_sel.mean()),
                "strict_masked_nll":   strict_mean,
                "fallback_nll":        float(fallbk_sel.mean()),
                "gold_token_coverage": cov_rate,
                "fallback_rate":       1.0 - cov_rate,
                "avg_candidate_tokens":float(nc_sel.mean()),
                "mapped_vocab_pct":    float(nc_sel.mean() / vocab_size * 100),
            }

        r_all = _row("all", np.ones(N_eval, dtype=bool))
        if r_all:
            all_summary.append(r_all)
            all_detail.append(r_all)
        for sid, sname in SPLIT_NAMES.items():
            r = _row(sname, split_np == sid)
            if r:
                all_detail.append(r)

        print(f"cov={covered_np.mean():.4f}  "
              f"fallback={1-covered_np.mean():.4f}  "
              f"cand={n_cand_np.mean():.0f}  "
              f"({time.time()-t_p:.1f}s)")

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_csv(all_summary, os.path.join(args.output_dir, "phase1_policy_summary.csv"))
    _write_csv(all_detail,  os.path.join(args.output_dir, "phase1_policy_detail.csv"))
    print("[phase1] done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",    required=True,
                   help="Path to small model checkpoint.")
    p.add_argument("--knn_run_dir",   required=True,
                   help="Dir with per_position.npz.")
    p.add_argument("--region_map",    required=True,
                   help="token_to_region.json path.")
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--dataset",       default="wikitext-103-raw-v1")
    p.add_argument("--seq_len",       type=int, default=128)
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--max_positions", type=int, default=500_000)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run_eval(_parse())
