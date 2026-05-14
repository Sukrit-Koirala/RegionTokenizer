#!/usr/bin/env python3
"""
Region-Memory Masked-Softmax Evaluation.

Loads a trained small model checkpoint + cached kNN memory and evaluates
whether controller-selected candidate regions can preserve LM loss with
a reduced candidate-token vocabulary.

For each eval position:
  1. Run forward pass → full LM logits
  2. Get p_router from probe, p_mem from kNN memory
  3. Apply controller to select candidate regions
  4. Derive candidate tokens = all tokens in selected regions
  5. Compute masked logits (non-candidates → -inf)
  6. strict_nll  = nll(masked logits, gold_token)  if gold covered else +inf
  7. fallback_nll = nll(full logits,  gold_token)  (always computable)

Reports per split:
  full_lm_nll          — baseline (no masking)
  strict_masked_nll    — NLL with masking, +inf if uncovered
  fallback_nll         — NLL with masking, fall back to full vocab if uncovered
  gold_token_coverage  — fraction of positions where gold token is in candidate set
  gold_region_coverage — fraction of positions where gold region is in candidate set
  avg_candidate_tokens
  mapped_vocab_percent
  fallback_rate

Controller is loaded from best_controller.json. Policies supported:
  P0 (router_topK), P1 (fixed union), P2 (confidence-gated), P3 (type-A), P4 (logistic).

Usage:
    python scripts/eval_region_memory_masked_softmax.py \
        --small_ckpt   runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt \
        --knn_run_dir  runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20 \
        --region_map_path runs/region_maps_128/token_to_region.json \
        --controller_cfg  runs/region_memory_controller_proxy010/best_controller.json \
        --output_dir   runs/region_memory_masked_softmax_proxy010 \
        --device cuda
"""

import argparse
import csv
import json
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

try:
    from scripts.offline_region_knn import (
        TokenChunkDataset, load_region_map, build_inverse_map,
        tokens_per_region_tensor, load_wikitext,
        MemoryIndex, load_small_backbone_and_probe,
        validate_input_ids, get_hs_small,
    )
    _HAS_KNN = True
except ImportError as e:
    print(f"[WARNING] could not import from offline_region_knn: {e}")
    _HAS_KNN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


# ── Region-to-token lookup ────────────────────────────────────────────────────

def build_candidate_token_tensor(inv_map: Dict[int, List[int]],
                                  n_coarse: int, vocab_size: int
                                  ) -> torch.Tensor:
    """
    Returns LongTensor of shape (n_coarse, max_tokens_per_region), padded with -1.
    region_tokens[r] = token IDs in region r (padded with -1).
    """
    max_len = max((len(v) for v in inv_map.values()), default=1)
    region_tokens = torch.full((n_coarse, max_len), -1, dtype=torch.long)
    for r, toks in inv_map.items():
        region_tokens[r, :len(toks)] = torch.tensor(toks, dtype=torch.long)
    return region_tokens


# ── Controller policy ─────────────────────────────────────────────────────────

def apply_controller(
    r_topk_reg: torch.Tensor,   # (N, K) router top regions
    m_topk_reg: torch.Tensor,   # (N, K) mem top regions
    r_margin: torch.Tensor,     # (N,)
    m_margin: torch.Tensor,     # (N,)
    m_entropy: torch.Tensor,    # (N,)
    ctrl_cfg: Dict,
) -> torch.Tensor:
    """
    Returns selected_regions (N, max_K) with values ∈ [0, n_coarse) and -1 for padding.
    Implements Policy 0/1/2/3 from best_controller.json.
    """
    N  = len(r_margin)
    nc = r_topk_reg.shape[-1]
    pc = ctrl_cfg.get("policy_class", "P0")

    mem_thr  = float(ctrl_cfg.get("mem_margin_thr", 0.20))
    ent_thr  = float(ctrl_cfg.get("entropy_thr",    1.50))
    Kr_core  = int(ctrl_cfg.get("Kr_core",   4))
    Kr_mid   = int(ctrl_cfg.get("Kr_mid",    8))
    Kr_bnd   = int(ctrl_cfg.get("Kr_bnd",    8))
    Km_bnd   = int(ctrl_cfg.get("Km_bnd",    4))
    Kr_fall  = int(ctrl_cfg.get("Kr_fallback", 16))

    if pc == "P0":
        K = min(int(ctrl_cfg.get("avg_regions", 8)), nc)
        return r_topk_reg[:, :K]

    if pc == "P1":
        Kr = min(ctrl_cfg.get("Kr", Kr_bnd), nc)
        Km = min(ctrl_cfg.get("Km", Km_bnd), nc)
        # concat and deduplicate below
        return torch.cat([r_topk_reg[:, :Kr], m_topk_reg[:, :Km]], dim=-1)

    # P2 / P3: confidence-gated
    use_mem = (r_margin < 0.10) & (m_margin > mem_thr) & (m_entropy < ent_thr)

    max_out = max(Kr_core, Kr_mid, Kr_bnd + Km_bnd, Kr_fall)
    out = torch.full((N, max_out), -1, dtype=torch.long)

    c1 = r_margin >= 0.30
    c2 = (r_margin >= 0.10) & ~c1
    c3 = (r_margin < 0.10) & use_mem
    c4 = (r_margin < 0.10) & ~use_mem

    # Fill per case
    for case_mask, r_k, m_k in [
        (c1, Kr_core, 0),
        (c2, Kr_mid,  0),
        (c3, Kr_bnd,  Km_bnd),
        (c4, Kr_fall, 0),
    ]:
        n_c = int(case_mask.sum())
        if n_c == 0:
            continue
        r_sel = r_topk_reg[case_mask, :r_k]
        if m_k > 0:
            m_sel = m_topk_reg[case_mask, :m_k]
            combined = torch.cat([r_sel, m_sel], dim=-1)
        else:
            combined = r_sel
        out[case_mask, :combined.shape[-1]] = combined

    return out


def selected_to_token_mask(selected_regions: torch.Tensor,
                             region_tokens: torch.Tensor,
                             vocab_size: int,
                             device: torch.device) -> torch.Tensor:
    """
    selected_regions: (N, max_K)  region ids (-1 = padding)
    region_tokens:    (n_coarse, max_tok_per_region)  token ids (-1 = padding)
    Returns bool mask (N, vocab_size): True = candidate token.
    """
    N = selected_regions.shape[0]
    mask = torch.zeros(N, vocab_size, dtype=torch.bool, device=device)

    # Gather all token ids for selected regions
    valid_regs = selected_regions.clamp(min=0)  # replace -1 padding with 0 temporarily
    pad_regs   = selected_regions < 0           # (N, K) which entries were padding

    # region_tokens[valid_regs]: (N, K, max_tpr)
    tok_ids = region_tokens[valid_regs]  # (N, K, max_tpr)
    tok_ids[pad_regs] = -1               # zero out padding regions
    tok_ids = tok_ids.reshape(N, -1)     # (N, K * max_tpr)

    valid_tok = tok_ids >= 0             # (N, K*max_tpr)
    for i in range(N):
        tids = tok_ids[i][valid_tok[i]]
        if len(tids):
            mask[i, tids] = True
    return mask


def selected_to_token_mask_scatter(selected_regions: torch.Tensor,
                                    region_tokens: torch.Tensor,
                                    vocab_size: int,
                                    device: torch.device) -> torch.Tensor:
    """
    Vectorised version of selected_to_token_mask using scatter.
    """
    N = selected_regions.shape[0]
    valid_regs = selected_regions.clamp(min=0)
    pad_regs   = (selected_regions < 0)

    # region_tokens[valid_regs]: (N, K, max_tpr)
    tok_ids = region_tokens.to(device)[valid_regs]  # (N, K, max_tpr)
    # zero out padding
    tok_ids[pad_regs.unsqueeze(-1).expand_as(tok_ids)] = -1
    tok_ids = tok_ids.reshape(N, -1)    # (N, K*max_tpr)

    mask = torch.zeros(N, vocab_size, dtype=torch.bool, device=device)
    valid = tok_ids >= 0
    # Clamp invalid to 0 so scatter doesn't crash; valid mask corrects
    safe_ids = tok_ids.clamp(min=0)
    mask.scatter_(1, safe_ids * valid.long(), valid)
    return mask


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run_masked_eval(
    backbone,
    probe,
    mem_index: "MemoryIndex",
    mem_regions: np.ndarray,
    coarse_map: torch.Tensor,
    inv_map: Dict[int, List[int]],
    region_tokens: torch.Tensor,
    val_loader: DataLoader,
    max_eval_batches: int,
    n_coarse: int,
    knn_k: int,
    knn_temp: float,
    probe_temp: float,
    ctrl_cfg: Dict,
    device: torch.device,
    vocab_size: int,
) -> Dict[str, List]:
    """
    Returns dict of per-position lists:
      full_nll, strict_nll, fallback_nll, covered (bool),
      n_candidate_tokens, r_margin
    """
    full_nll_all:     List[float] = []
    strict_nll_all:   List[float] = []
    fallback_nll_all: List[float] = []
    covered_all:      List[bool]  = []
    n_cand_all:       List[int]   = []
    r_margin_all:     List[float] = []
    gold_reg_cov_all: List[bool]  = []

    # LM head — use the model's lm_head
    n_batches = min(max_eval_batches, len(val_loader))
    d_model   = backbone.token_emb.embedding_dim
    max_ctx   = backbone.pos_emb.num_embeddings

    for bi, batch in enumerate(val_loader):
        if bi >= max_eval_batches:
            break
        print(f"  eval batch {bi}/{n_batches}")

        batch = batch.to(device)
        src   = batch[:, :-1]    # (B, T)
        tgt   = batch[:, 1:]     # (B, T)

        # Clamp to max_ctx (defensive)
        if src.size(1) > max_ctx:
            src = src[:, :max_ctx]
            tgt = tgt[:, :max_ctx]

        B, T = src.shape

        gold_regions = coarse_map[tgt].reshape(-1)  # (B*T,) region or -1
        gold_tokens  = tgt.reshape(-1)              # (B*T,)
        valid        = gold_regions >= 0
        if not valid.any():
            continue

        # Hidden states
        h = get_hs_small(backbone, src, device)     # (B, T, d_model)

        # Full LM logits
        with torch.no_grad():
            h_flat   = h.reshape(-1, d_model)            # (B*T, d_model)
            lm_logits = backbone.lm_head(h_flat)         # (B*T, vocab_size)

        full_nll_v = F.cross_entropy(
            lm_logits[valid], gold_tokens[valid], reduction="none"
        ).cpu()

        # Router distribution (probe over raw h)
        p_router = None
        if probe is not None:
            with torch.no_grad():
                r_logits = probe(h_flat[valid])
            p_router = F.softmax(r_logits.float() / probe_temp, dim=-1).cpu()

        # kNN retrieval
        key_valid = h_flat[valid].cpu().numpy().astype(np.float32)
        if hasattr(backbone, "retrieval_proj"):
            with torch.no_grad():
                z = backbone.retrieval_proj(h_flat[valid].float())
                z = F.normalize(z, dim=-1)
            key_valid = z.cpu().numpy().astype(np.float32)

        sims, nn_idx  = mem_index.search(key_valid, knn_k)
        nn_regs_t     = torch.tensor(mem_regions[nn_idx].astype(np.int64))
        sims_t        = torch.tensor(sims)
        w             = F.softmax(sims_t / knn_temp, dim=-1)
        p_mem         = torch.zeros(valid.sum(), n_coarse)
        p_mem.scatter_add_(1, nn_regs_t, w)
        p_mem         = p_mem / p_mem.sum(-1, keepdim=True).clamp(min=1e-10)

        # Margins / entropy
        top2m     = torch.topk(p_mem, min(2, n_coarse), dim=-1).values
        m_margin  = top2m[:, 0] - top2m[:, 1]
        m_entropy = (-(p_mem * (p_mem + 1e-10).log()).sum(-1))

        if p_router is not None:
            top2r    = torch.topk(p_router, min(2, n_coarse), dim=-1).values
            r_margin = (top2r[:, 0] - top2r[:, 1])
            r_topk_reg = torch.topk(p_router, min(32, n_coarse), dim=-1).indices
        else:
            r_margin   = torch.zeros(valid.sum())
            r_topk_reg = torch.zeros(valid.sum(), 1, dtype=torch.long)

        m_topk_reg = torch.topk(p_mem, min(32, n_coarse), dim=-1).indices

        # Controller → selected regions
        sel_regs = apply_controller(
            r_topk_reg, m_topk_reg, r_margin, m_margin, m_entropy, ctrl_cfg,
        )  # (N_valid, K_sel) on CPU

        sel_regs_dev = sel_regs.to(device)
        region_tokens_dev = region_tokens.to(device)

        # Token mask (vectorised scatter)
        tok_mask = selected_to_token_mask_scatter(
            sel_regs_dev, region_tokens_dev, vocab_size, device
        )  # (N_valid, vocab_size)

        # Candidate-token count
        n_cand = tok_mask.sum(-1).cpu().long()

        # Gold region coverage
        gold_reg_v  = gold_regions[valid].cpu()
        sel_regs_cpu = sel_regs.cpu()
        gold_reg_cov = (sel_regs_cpu == gold_reg_v.unsqueeze(-1)).any(-1)

        # Gold token coverage
        gold_tok_v   = gold_tokens[valid].cpu()
        covered      = tok_mask.cpu()[torch.arange(valid.sum()), gold_tok_v].bool()

        # Strict NLL: mask to candidates, +inf if uncovered
        masked_logits = lm_logits[valid].clone()
        masked_logits[~tok_mask] = float("-inf")
        strict_nll_v = F.cross_entropy(
            masked_logits, gold_tokens[valid], reduction="none"
        ).cpu()
        strict_nll_v[~covered] = float("inf")

        # Fallback NLL: full vocab if uncovered
        fallback_nll_v = full_nll_v.clone()
        if covered.any():
            fallback_nll_v[covered] = strict_nll_v[covered]

        full_nll_all.extend(full_nll_v.tolist())
        strict_nll_all.extend(strict_nll_v.tolist())
        fallback_nll_all.extend(fallback_nll_v.tolist())
        covered_all.extend(covered.tolist())
        n_cand_all.extend(n_cand.tolist())
        r_margin_all.extend(r_margin.tolist())
        gold_reg_cov_all.extend(gold_reg_cov.tolist())

    return {
        "full_nll":      full_nll_all,
        "strict_nll":    strict_nll_all,
        "fallback_nll":  fallback_nll_all,
        "covered":       covered_all,
        "n_cand":        n_cand_all,
        "r_margin":      r_margin_all,
        "gold_reg_cov":  gold_reg_cov_all,
    }


# ── Metrics aggregation ───────────────────────────────────────────────────────

def aggregate(data: Dict[str, List], mask=None) -> Dict:
    if mask is not None:
        data = {k: [v for v, m in zip(data[k], mask) if m] for k in data}
    N = len(data["full_nll"])
    if N == 0:
        return {"n": 0}

    full_nll = np.array(data["full_nll"], dtype=np.float32)
    fallback  = np.array(data["fallback_nll"], dtype=np.float32)
    covered   = np.array(data["covered"], dtype=bool)
    n_cand    = np.array(data["n_cand"], dtype=np.float32)
    grg_cov   = np.array(data["gold_reg_cov"], dtype=bool)

    strict_arr = np.array(data["strict_nll"], dtype=np.float32)
    strict_fin = strict_arr[np.isfinite(strict_arr)]

    return {
        "n":                    N,
        "full_lm_nll":          round(float(full_nll.mean()), 4),
        "fallback_nll":         round(float(fallback.mean()), 4),
        "strict_masked_nll":    round(float(strict_fin.mean()), 4) if len(strict_fin) else float("nan"),
        "gold_token_coverage":  round(float(covered.mean()), 4),
        "gold_region_coverage": round(float(grg_cov.mean()), 4),
        "avg_candidate_tokens": round(float(n_cand.mean()), 1),
        "fallback_rate":        round(float((~covered).mean()), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Region-Memory Masked-Softmax Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--small_ckpt",      required=True)
    parser.add_argument("--knn_run_dir",     required=True,
                        help="kNN run dir with memory_keys.npy and knn_config.json")
    parser.add_argument("--region_map_path", required=True)
    parser.add_argument("--controller_cfg",  default=None,
                        help="best_controller.json from eval_region_memory_controller")
    parser.add_argument("--dataset",         default="wikitext-103-raw-v1")
    parser.add_argument("--max_eval_positions", type=int, default=50_000)
    parser.add_argument("--batch_size",      type=int, default=4)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    if not _HAS_KNN:
        print("ERROR: could not import from offline_region_knn.py — check PYTHONPATH")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[msoftmax] device={device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[msoftmax] loading checkpoint: {args.small_ckpt}")
    backbone, probe, d_model, cfg_dict, vocab_size = \
        load_small_backbone_and_probe(args.small_ckpt, device)

    if not hasattr(backbone, "lm_head"):
        print("ERROR: backbone has no lm_head — is this a causal LM checkpoint?")
        sys.exit(1)

    max_ctx = backbone.pos_emb.num_embeddings
    print(f"[msoftmax] d_model={d_model}  max_ctx={max_ctx}  vocab_size={vocab_size}")

    # ── Load kNN config & memory ──────────────────────────────────────────────
    knn_cfg_path = os.path.join(args.knn_run_dir, "knn_config.json")
    knn_cfg = json.load(open(knn_cfg_path)) if os.path.exists(knn_cfg_path) else {}
    knn_k    = knn_cfg.get("knn_k", 64)
    knn_temp = knn_cfg.get("knn_temp", 0.20)
    key_dim  = knn_cfg.get("key_dim", d_model)
    probe_temp = knn_cfg.get("probe_temp", 1.0)
    print(f"[msoftmax] knn_k={knn_k}  knn_temp={knn_temp}  key_dim={key_dim}")

    keys_path    = os.path.join(args.knn_run_dir, "memory_keys.npy")
    regions_path = os.path.join(args.knn_run_dir, "memory_regions.npy")
    if not os.path.exists(keys_path):
        print(f"ERROR: memory_keys.npy not found in {args.knn_run_dir}")
        sys.exit(1)

    keys_f16  = np.load(keys_path)
    mem_regs  = np.load(regions_path)
    print(f"[msoftmax] memory: {len(keys_f16):,} entries  key_dim={keys_f16.shape[1]}")

    mem_index = MemoryIndex(keys_f16.shape[1], device, normalize=True)
    mem_index.build(keys_f16.astype(np.float32))

    # ── Region map ────────────────────────────────────────────────────────────
    coarse_map, n_coarse = load_region_map(args.region_map_path, vocab_size)
    coarse_map = coarse_map.to(device)
    inv_map    = build_inverse_map(coarse_map.cpu(), n_coarse)
    tpr        = tokens_per_region_tensor(inv_map, n_coarse)
    total_mapped = int(tpr.sum().item())
    print(f"[msoftmax] n_coarse={n_coarse}  total_mapped={total_mapped:,}")

    region_tokens = build_candidate_token_tensor(inv_map, n_coarse, vocab_size)

    # ── Controller config ─────────────────────────────────────────────────────
    ctrl_cfg: Dict = {"policy_class": "P0", "avg_regions": 8}
    if args.controller_cfg and os.path.exists(args.controller_cfg):
        ctrl_cfg = json.load(open(args.controller_cfg))
        print(f"[msoftmax] controller: {ctrl_cfg.get('policy_name','?')}  "
              f"class={ctrl_cfg.get('policy_class')}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = int(1e30)

    safe_seq_len = min(512, max_ctx)
    print(f"[msoftmax] encoding val split (seq_len={safe_seq_len}) ...")
    val_tokens = load_wikitext(args.dataset, tok, "validation")
    print(f"[msoftmax] val tokens: {len(val_tokens):,}")

    ppb = args.batch_size * safe_seq_len
    max_eval_batches = math.ceil(args.max_eval_positions / ppb)
    val_ds     = TokenChunkDataset(val_tokens, safe_seq_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"[msoftmax] running masked-softmax eval  ({max_eval_batches} batches) ...")
    t0 = time.time()
    results = run_masked_eval(
        backbone, probe, mem_index, mem_regs, coarse_map,
        inv_map, region_tokens, val_loader, max_eval_batches,
        n_coarse, knn_k, knn_temp, probe_temp, ctrl_cfg, device, vocab_size,
    )
    print(f"[msoftmax] eval done in {time.time()-t0:.1f}s  N={len(results['full_nll']):,}")

    # ── Aggregate overall + per split ─────────────────────────────────────────
    r_mg = np.array(results["r_margin"], dtype=np.float32)
    split_masks = {
        "all":            np.ones(len(r_mg), dtype=bool),
        "core":           r_mg >= 0.30,
        "medium":         (r_mg >= 0.10) & (r_mg < 0.30),
        "boundary":       r_mg < 0.10,
        "tight_boundary": r_mg < 0.03,
    }

    agg_rows: List[Dict] = []
    print("\n[msoftmax] Results:")
    print(f"  {'split':<16}  {'n':>7}  {'full_nll':>8}  "
          f"{'fallback_nll':>12}  {'tok_cov':>7}  {'avg_cand':>8}  {'fallback_rate':>13}")
    for sname, smask in split_masks.items():
        agg = aggregate(results, mask=smask.tolist())
        agg["split"] = sname
        if total_mapped > 0:
            agg["mapped_vocab_pct"] = round(agg.get("avg_candidate_tokens", 0) / total_mapped * 100, 2)
        agg_rows.append(agg)
        print(f"  {sname:<16}  {agg.get('n',0):>7,}  "
              f"{agg.get('full_lm_nll', float('nan')):>8.4f}  "
              f"{agg.get('fallback_nll', float('nan')):>12.4f}  "
              f"{agg.get('gold_token_coverage', float('nan')):>7.3f}  "
              f"{agg.get('avg_candidate_tokens', float('nan')):>8.0f}  "
              f"{agg.get('fallback_rate', float('nan')):>13.3f}")

    # ── Pass/fail decision ────────────────────────────────────────────────────
    overall = next((r for r in agg_rows if r["split"] == "all"), {})
    full_nll     = overall.get("full_lm_nll", float("nan"))
    fallback_nll = overall.get("fallback_nll", float("nan"))
    tok_cov      = overall.get("gold_token_coverage", 0.0)
    mapped_pct   = overall.get("mapped_vocab_pct", 100.0)

    nll_delta = fallback_nll - full_nll
    strong  = tok_cov >= 0.90 and mapped_pct <= 8.0  and nll_delta < 0.05
    moderate= tok_cov >= 0.95 and mapped_pct <= 15.0 and nll_delta < 0.10
    print(f"\n[msoftmax] Pass/fail:")
    print(f"  gold_token_coverage={tok_cov:.3f}  mapped_vocab_pct={mapped_pct:.1f}%  "
          f"fallback-full delta={nll_delta:+.4f}")
    if strong:
        print("  STRONG PASS — proceed to local refiner (Phase 2B)")
    elif moderate:
        print("  MODERATE PASS — candidate expansion is viable; consider Phase 2B")
    else:
        print("  FAIL — controller cannot meet coverage/cost criteria; re-examine controller")

    # ── Write outputs ─────────────────────────────────────────────────────────
    def _write_csv(rows, path):
        if not rows:
            return
        keys = {}
        for r in rows:
            for k in r:
                keys[k] = None
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keys.keys()), restval="")
            w.writeheader()
            w.writerows(rows)

    _write_csv(agg_rows, os.path.join(args.output_dir, "masked_softmax_results.csv"))

    # Summary
    cfg_out = {**knn_cfg, "controller": ctrl_cfg,
               "max_eval_positions": args.max_eval_positions,
               "overall": overall}
    with open(os.path.join(args.output_dir, "masked_softmax_config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2, default=str)

    # Report
    md = [
        "# Region-Memory Masked-Softmax Evaluation", "",
        f"Checkpoint: `{args.small_ckpt}`",
        f"kNN run:    `{args.knn_run_dir}`",
        f"Controller: `{ctrl_cfg.get('policy_name','?')}`  (class={ctrl_cfg.get('policy_class')})",
        "",
        "## Overall Results", "",
        "| Metric | Value |", "|--------|------:|",
        f"| full_lm_nll | {full_nll:.4f} |",
        f"| fallback_nll | {fallback_nll:.4f} |",
        f"| nll_delta (fallback - full) | {nll_delta:+.4f} |",
        f"| gold_token_coverage | {tok_cov:.3f} |",
        f"| avg_candidate_tokens | {overall.get('avg_candidate_tokens',0):.0f} |",
        f"| mapped_vocab_pct | {mapped_pct:.1f}% |",
        f"| fallback_rate | {overall.get('fallback_rate',0):.3f} |",
        "",
        "## Per-Split Results", "",
        "| Split | N | full NLL | fallback NLL | tok_cov | avg_cand | mapped% |",
        "|-------|--:|---------:|-------------:|--------:|---------:|--------:|",
    ]
    for r in agg_rows:
        md.append(
            f"| {r['split']} | {r.get('n',0):,} "
            f"| {r.get('full_lm_nll', float('nan')):.4f} "
            f"| {r.get('fallback_nll', float('nan')):.4f} "
            f"| {r.get('gold_token_coverage', float('nan')):.3f} "
            f"| {r.get('avg_candidate_tokens', float('nan')):.0f} "
            f"| {r.get('mapped_vocab_pct', float('nan')):.1f}% |"
        )
    md += ["",
           "## Decision",
           "",
           ("**STRONG PASS** — proceed to Phase 2B local refiner." if strong else
            "**MODERATE PASS** — Phase 2B viable." if moderate else
            "**FAIL** — do not proceed to local refiner."),
           ]
    with open(os.path.join(args.output_dir, "masked_softmax_report.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if _HAS_PLT and len(agg_rows) > 1:
        snames = [r["split"] for r in agg_rows]
        f_nlls = [r.get("full_lm_nll", float("nan")) for r in agg_rows]
        fb_nll = [r.get("fallback_nll", float("nan")) for r in agg_rows]
        x = np.arange(len(snames))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.2, f_nlls, 0.35, label="full LM NLL", color="steelblue")
        ax.bar(x + 0.2, fb_nll, 0.35, label="fallback NLL", color="orange")
        ax.set_xticks(x)
        ax.set_xticklabels(snames)
        ax.set_ylabel("NLL")
        ax.set_title("Masked-softmax fallback NLL vs full LM NLL by split")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "masked_softmax_nll_by_split.png"), dpi=120)
        plt.close(fig)

    print(f"\n[msoftmax] DONE → {args.output_dir}/")


if __name__ == "__main__":
    main()
