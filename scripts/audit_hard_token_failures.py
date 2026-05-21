#!/usr/bin/env python3
"""
Hard-Token Failure Anatomy + Recoverability Audit.

Offline diagnostic over the validation set. No training. No model updates.

Answers:
  Q1.  Boundary loss mass fraction
  Q2.  Gold rank distribution among boundary rows
  Q3.  Gold region in router/memory top-k
  Q4.  Region failure vs token failure breakdown
  Q5.  Memory signal not in router
  Q6.  Router signal not in memory
  Q7.  Bridge improvement on recoverable vs unrecoverable
  Q8.  Which subset does Bridge hurt
  Q9.  Gate breadth analysis
  Q10. Context/input_ids availability
  Q11. Next architecture recommendation

Outputs in --output_dir/:
  config.json, context_availability.json/.md
  val_failure_audit.parquet, val_failure_audit.csv.gz
  bucket_summary.csv/.md, loss_mass_report.md
  top_100_bridge_wins.csv, top_100_bridge_losses.csv  (if refiner loaded)
  top_examples.md, final_report.md
"""

import argparse
import glob
import gzip
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

from scripts.offline_region_knn import load_small_backbone_and_probe
from scripts.train_clean_path_refiner import load_r2s
from scripts.train_hard_position_refiner import compute_filter_mask, EVAL_SUBSETS
from scripts.train_bridge_residual_adapter import (
    CANONICAL_FINGERPRINT, CANONICAL_NUM_EXAMPLES, CANONICAL_NUM_COVERED,
    CANONICAL_COVERAGE, MASKED_CAND_BASELINE_NLL,
)

KNOWN_BASE_NLL_ALL     = 3.754938
KNOWN_BASE_NLL_COVERED = 3.481444
KNOWN_MASKED_CAND_NLL  = 3.378606

TOPK_THRESHOLDS    = [1, 5, 10, 50, 100, 500, 1000]
REGION_THRESHOLDS  = [1, 4, 8, 12]
SPLIT_NAMES        = {0: "core", 1: "medium", 2: "boundary", 3: "tight"}
TYPE_NAMES         = {0: "other", 1: "A", 2: "B", 3: "C"}

AUDIT_SUBSETS = [
    "all", "boundary", "tight_boundary", "type_A", "type_B", "type_C",
    "router_top8_miss", "router_top4_miss", "router_low_margin",
    "hard_union_small", "hard_union",
]


# ── Context availability ──────────────────────────────────────────────────────

def check_context_availability(cand_dir: str, feat_dir: str) -> Dict:
    context_keys = {"input_ids", "context_ids", "x", "tokens", "token_ids",
                    "sequence", "window", "idx", "position", "pos", "doc_id", "offset"}
    result: Dict = {
        "context_available": False,
        "keys_found": [],
        "can_reconstruct_live_forward": False,
        "cand_keys": [],
        "feat_keys": [],
        "reason": "",
    }
    for label, d in [("cand", cand_dir), ("feat", feat_dir)]:
        shards = sorted(glob.glob(os.path.join(d, "shard_*.pt")))
        if not shards:
            continue
        try:
            s = torch.load(shards[0], map_location="cpu", weights_only=True)
            keys = list(s.keys())
            result[f"{label}_keys"] = keys
            result["keys_found"].extend(k for k in keys if k.lower() in context_keys)
        except Exception as e:
            result["reason"] += f"Cannot read {label} shard: {e}. "

    result["keys_found"] = sorted(set(result["keys_found"]))
    if result["keys_found"]:
        result["context_available"] = True
        result["can_reconstruct_live_forward"] = any(
            k in result["keys_found"] for k in ("input_ids", "context_ids"))
        result["reason"] = f"Context-related keys found: {result['keys_found']}"
    else:
        result["reason"] = (
            "No context/input_ids found in candidate or feature shards. "
            "Feature shards store per-position hidden states only (N, n_layers, d_model). "
            "Live causal injection cannot be tested from current dataset. "
            "Current artifacts only support cached hidden-state / final-patch experiments. "
            "Need a new dataset with input_ids/context window and row alignment."
        )
    return result


# ── Token-to-region lookup ────────────────────────────────────────────────────

def load_token_to_region(region_map_path: str, vocab_size: int) -> Optional[np.ndarray]:
    if not region_map_path or not os.path.isfile(region_map_path):
        return None
    with open(region_map_path) as f:
        raw = json.load(f)
    t2r = np.full(vocab_size, -1, dtype=np.int32)
    for k, v in raw.items():
        tok = int(k)
        if 0 <= tok < vocab_size:
            t2r[tok] = int(v)
    return t2r


# ── Optional refiner loading ──────────────────────────────────────────────────

def try_load_refiner(
    ckpt_path: str,
    d_model: int,
    n_fine: int,
    n_super: int,
    r2s_np: np.ndarray,
    feat_layer_ids: List[int],
    device,
) -> Tuple[Optional[str], Optional[object]]:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None, None
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved_args = ck.get("args", {})
        state_dict = ck.get("model", {})

        # Detect architecture from saved args
        if "write_mode" in saved_args:
            arch = f"running_bridge_{saved_args['write_mode']}"
            from scripts.train_running_bridge import RunningBridgeAdapter, _resolve_update_layer_idxs
            a = saved_args
            uidxs = _resolve_update_layer_idxs(a.get("update_layers", "2,4"), feat_layer_ids)
            model = RunningBridgeAdapter(
                d_model=d_model,
                bridge_dim=a.get("bridge_dim", 256),
                bridge_heads=a.get("bridge_heads", 4),
                bridge_update_layers=a.get("bridge_update_layers", 1),
                refiner_dim=a.get("refiner_dim", 256),
                refiner_heads=a.get("refiner_heads", 4),
                refiner_layers=a.get("refiner_layers", 2),
                ff_mult=a.get("ff_mult", 4),
                n_fine=n_fine, n_super=n_super,
                top_fine_k=a.get("top_fine_regions", 24),
                top_super_k=a.get("top_superregions", 8),
                update_layer_idxs=uidxs,
                num_path_tokens=a.get("num_path_tokens", 4),
                write_mode=saved_args["write_mode"],
                r2s_np=r2s_np,
            )
        elif "refiner_layers" in saved_args and "insert_after_block" in saved_args:
            arch = "explicit_bridge_refiner"
            from scripts.train_explicit_bridge_refiner import ExplicitBridgeRefiner
            a = saved_args
            iab = a.get("insert_after_block", 4)
            insert_idx = feat_layer_ids.index(iab) if iab in feat_layer_ids else len(feat_layer_ids) - 1
            model = ExplicitBridgeRefiner(
                d_model=d_model,
                bridge_dim=a.get("bridge_dim", 256),
                bridge_heads=a.get("bridge_heads", 4),
                bridge_layers=a.get("bridge_layers", 1),
                refiner_dim=a.get("refiner_dim", 256),
                refiner_heads=a.get("refiner_heads", 4),
                refiner_layers=a.get("refiner_layers", 2),
                ff_mult=a.get("ff_mult", 4),
                n_fine=n_fine, n_super=n_super,
                top_fine_k=a.get("top_fine_regions", 24),
                top_super_k=a.get("top_superregions", 8),
                num_path_tokens=a.get("num_path_tokens", 4),
                insert_layer_idx=insert_idx,
                r2s_np=r2s_np,
            )
        elif "insert_after_block" in saved_args:
            arch = "midlayer_bridge"
            from scripts.train_midlayer_bridge_adapter import MidLayerBridgeAdapter
            a = saved_args
            iab = a.get("insert_after_block", 4)
            insert_idx = feat_layer_ids.index(iab) if iab in feat_layer_ids else len(feat_layer_ids) - 1
            model = MidLayerBridgeAdapter(
                d_model=d_model,
                bridge_dim=a.get("bridge_dim", 256),
                num_heads=a.get("num_heads", 4),
                num_layers=a.get("num_bridge_layers", 2),
                ff_mult=a.get("ff_mult", 4),
                n_fine=n_fine, n_super=n_super,
                top_fine_k=a.get("top_fine_regions", 24),
                top_super_k=a.get("top_superregions", 8),
                num_path_tokens=a.get("num_path_tokens", 1),
                insert_layer_idx=insert_idx,
                r2s_np=r2s_np,
            )
        else:
            arch = "bridge_v1"
            from scripts.train_bridge_residual_adapter import BridgeResidualAdapter
            a = saved_args
            model = BridgeResidualAdapter(
                d_model=d_model,
                bridge_dim=a.get("bridge_dim", 256),
                num_heads=a.get("num_heads", 4),
                num_layers=a.get("num_bridge_layers", 2),
                ff_mult=a.get("ff_mult", 4),
                n_fine=n_fine, n_super=n_super,
                top_fine_k=a.get("top_fine_regions", 24),
                top_super_k=a.get("top_superregions", 8),
                num_path_tokens=a.get("num_path_tokens", 1),
                r2s_np=r2s_np,
            )

        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"  [refiner] Loaded {arch} from {ckpt_path}")
        return arch, model
    except Exception as e:
        print(f"  WARNING: Could not load refiner from {ckpt_path}: {e}")
        return None, None


def run_refiner_forward(arch: str, model, h_prime, h_layers, ct, cf, cmask, emb_w,
                         r_reg, r_prb, m_reg, m_prb, r_mar, m_mar) -> torch.Tensor:
    """Returns h_refined (B, d_model). Handles all architecture variants."""
    with torch.no_grad():
        if arch.startswith("running"):
            h_ref, _, _, _ = model(h_prime, h_layers, ct, cf, cmask, emb_w,
                                   r_reg, r_prb, m_reg, m_prb, r_mar, m_mar)
        elif arch == "explicit_bridge_refiner":
            h_ref, _, _ = model(h_prime, h_layers, ct, cf, cmask, emb_w,
                                r_reg, r_prb, m_reg, m_prb, r_mar, m_mar)
        else:
            h_ref, _ = model(h_prime, h_layers, ct, cf, cmask, emb_w,
                             r_reg, r_prb, m_reg, m_prb, r_mar, m_mar)
    return h_ref.float()


# ── Main audit loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def audit_val_rows(
    tok_emb_w:   torch.Tensor,
    r2s_np:      np.ndarray,
    t2r:         Optional[np.ndarray],
    refiners:    List[Tuple[str, object]],
    args,
    device,
) -> List[Dict]:
    emb_w = tok_emb_w.float().to(device)
    V     = emb_w.shape[0]
    rows: List[Dict] = []
    total_rows = 0
    t0 = time.time()

    cand_paths = sorted(glob.glob(os.path.join(args.val_cand_dir, "shard_*.pt")))
    if not cand_paths:
        raise RuntimeError(f"No shard_*.pt in {args.val_cand_dir}")
    print(f"[audit] {len(cand_paths)} shards  V={V}  refiners={[r[0] for r in refiners]}")

    for cand_path in cand_paths:
        si        = int(os.path.basename(cand_path).replace("shard_", "").replace(".pt", ""))
        feat_path = os.path.join(args.val_feat_dir, f"shard_{si:05d}.pt")
        cs        = torch.load(cand_path, map_location="cpu", weights_only=True)
        N         = cs["gold_token"].shape[0]
        has_feat  = os.path.exists(feat_path)
        fs        = (torch.load(feat_path, map_location="cpu", weights_only=True)
                     if has_feat else None)
        has_mem   = "mem_topk_reg" in cs

        # Split/type masks (per full shard, not sliced)
        split_arr    = cs["split"].long().numpy()   if "split"    in cs else np.zeros(N, dtype=np.int64)
        type_arr     = cs["type_arr"].long().numpy() if "type_arr" in cs else np.zeros(N, dtype=np.int64)
        gold_region_shard = cs["gold_region"].long().numpy() if "gold_region" in cs else np.full(N, -1, dtype=np.int64)

        filter_kwargs = {"margin_thresh": args.margin_thresh,
                         "entropy_thresh": args.entropy_thresh}
        subset_masks = {}
        for name in AUDIT_SUBSETS:
            try:
                subset_masks[name] = compute_filter_mask(cs, name, **filter_kwargs).numpy().astype(bool)
            except Exception:
                subset_masks[name] = np.ones(N, dtype=bool) if name == "all" else np.zeros(N, dtype=bool)

        for start in range(0, N, args.eval_batch_size):
            end = min(start + args.eval_batch_size, N)
            sl  = slice(start, end)
            B   = end - start

            h_p   = cs["h_prime"][sl].float().to(device)
            gt_b  = cs["gold_token"][sl].long()
            ct    = cs["cand_tok"][sl].long()
            cf    = cs["cand_fine"][sl].long()
            cov_b = cs["covered"][sl].bool().numpy().astype(bool)
            r_reg = cs["router_topk_reg"][sl].long()
            r_prb = cs["router_topk_prb"][sl].float()
            r_mar = cs["router_margin"][sl].float()
            gidx  = cs["gold_cand_idx"][sl].long()

            m_reg = cs["mem_topk_reg"][sl].long()  if has_mem else torch.zeros_like(r_reg)
            m_prb = cs["mem_topk_prb"][sl].float() if has_mem else torch.zeros_like(r_prb)
            m_mar = cs["mem_margin"][sl].float()   if has_mem else torch.zeros(B)

            h_l = fs["h_layers"][sl].float().to(device) if has_feat else torch.zeros(B, 1, h_p.shape[-1], device=device)

            # ── Base full-vocab ──────────────────────────────────────────────
            base_lgt  = h_p.float() @ emb_w.T              # (B, V)
            log_probs = F.log_softmax(base_lgt, dim=1)      # (B, V)
            gt_dev    = gt_b.to(device)

            gold_nlls  = (-log_probs[torch.arange(B, device=device), gt_dev]).cpu().numpy()
            gold_probs = log_probs[torch.arange(B, device=device), gt_dev].exp().cpu().numpy()
            gold_lgts  = base_lgt[torch.arange(B, device=device), gt_dev].cpu().numpy()

            # Efficient gold rank: count logits strictly greater
            gold_lgts_dev = base_lgt[torch.arange(B, device=device), gt_dev].unsqueeze(1)
            gold_rank_t   = 1 + (base_lgt > gold_lgts_dev).sum(1)
            gold_ranks    = gold_rank_t.cpu().numpy()

            # Top-k token IDs
            K = min(args.topk_rank, V)
            topk_vals, topk_ids = base_lgt.topk(K, dim=1)  # (B, K)
            top1_ids   = topk_ids[:, 0].cpu().numpy()
            top1_lgts  = topk_vals[:, 0].cpu().numpy()
            top1_nlls  = (-log_probs[torch.arange(B, device=device), topk_ids[:, 0].to(device)]).cpu().numpy()
            top1_probs = np.exp(-top1_nlls)
            margin_top1_minus_gold = top1_lgts - gold_lgts

            # In-top-k (vectorized)
            topk_cpu   = topk_ids.cpu()
            gt_b_col   = gt_b.unsqueeze(1)  # (B, 1)
            in_topk    = {k: (topk_cpu[:, :k] == gt_b_col).any(1).numpy() for k in TOPK_THRESHOLDS}

            # ── Regions ─────────────────────────────────────────────────────
            gold_reg  = gold_region_shard[start:end].astype(np.int32)
            gold_sup  = np.where(gold_reg >= 0,
                                  r2s_np[np.clip(gold_reg, 0, len(r2s_np) - 1)], -1).astype(np.int32)

            if t2r is not None:
                top1_reg = t2r[np.clip(top1_ids, 0, len(t2r) - 1)].astype(np.int32)
                top1_sup = np.where(top1_reg >= 0,
                                     r2s_np[np.clip(top1_reg, 0, len(r2s_np) - 1)], -1).astype(np.int32)
            else:
                top1_reg = np.full(B, -1, dtype=np.int32)
                top1_sup = np.full(B, -1, dtype=np.int32)

            top1_same_fine  = (top1_reg == gold_reg)
            top1_same_super = (top1_sup == gold_sup)

            # ── Router metrics (vectorized) ──────────────────────────────────
            r_reg_np = r_reg.numpy()  # (B, K_r)
            r_prb_np = r_prb.numpy()
            r_mar_np = r_mar.numpy()

            router_ent = -(r_prb_np * np.log(r_prb_np + 1e-9)).sum(1)
            # Gold region rank in router (0=not found → -1, else 1-indexed)
            match_r = (r_reg_np == gold_reg[:, None])        # (B, K_r)
            r_gold_rank = np.where(match_r.any(1),
                                   match_r.argmax(1) + 1, -1).astype(np.int32)
            r_in_topk = {k: match_r[:, :k].any(1) for k in REGION_THRESHOLDS}

            # ── Memory metrics (vectorized) ──────────────────────────────────
            m_reg_np = m_reg.numpy()
            m_prb_np = m_prb.numpy()
            m_mar_np = m_mar.numpy()

            memory_ent = -(m_prb_np * np.log(m_prb_np + 1e-9)).sum(1)
            match_m = (m_reg_np == gold_reg[:, None])
            m_gold_rank = np.where(match_m.any(1),
                                   match_m.argmax(1) + 1, -1).astype(np.int32)
            m_in_topk = {k: match_m[:, :k].any(1) for k in REGION_THRESHOLDS}

            # Router-memory agreement
            rm_top1_agree = (r_reg_np[:, 0] == m_reg_np[:, 0])
            rm_top4_ov = np.array([len(set(r_reg_np[i, :4]) & set(m_reg_np[i, :4])) / 4.0
                                   for i in range(B)], dtype=np.float32)
            rm_top8_ov = np.array([len(set(r_reg_np[i, :8]) & set(m_reg_np[i, :8])) / 8.0
                                   for i in range(B)], dtype=np.float32)

            r_ok = r_in_topk[8]
            m_ok = m_in_topk[8]
            r_ok_m_wrong  = r_ok & ~m_ok
            m_ok_r_wrong  = m_ok & ~r_ok
            both_ok       = r_ok & m_ok
            both_wrong    = ~r_ok & ~m_ok

            # ── Candidate metrics ────────────────────────────────────────────
            cmask    = (ct >= 0).numpy()
            ct_np    = ct.numpy()
            gidx_np  = gidx.numpy()
            cand_cnt = cmask.sum(1)

            # Gold cand base logit
            ct_dev    = ct.to(device)
            gidx_dev  = gidx.clamp(min=0).to(device)
            gold_cand_tok = ct_dev.gather(1, gidx_dev.unsqueeze(1)).squeeze(1).clamp(min=0)
            gold_cand_lgt = base_lgt.gather(1, gold_cand_tok.unsqueeze(1)).squeeze(1).cpu().numpy()
            gold_cand_lgt = np.where(gidx_np >= 0, gold_cand_lgt, np.nan)

            # Candidate same-region counts (vectorized)
            if t2r is not None:
                cand_reg = t2r[np.clip(ct_np, 0, len(t2r) - 1)]     # (B, C)
                cand_reg = np.where(cmask, cand_reg, -1)
                cand_sup = np.where(cand_reg >= 0,
                                     r2s_np[np.clip(cand_reg, 0, len(r2s_np) - 1)], -1)
                cand_same_fine  = (cand_reg == gold_reg[:, None]).sum(1).astype(np.int32)
                cand_same_super = (cand_sup == gold_sup[:, None]).sum(1).astype(np.int32)
            else:
                cand_same_fine  = np.zeros(B, dtype=np.int32)
                cand_same_super = np.zeros(B, dtype=np.int32)

            # ── Optional refiner NLLs ────────────────────────────────────────
            refiner_nlls: Dict[str, np.ndarray] = {}
            if has_feat and refiners:
                ct_mask  = (ct >= 0).to(device)
                r_reg_d  = r_reg.to(device)
                r_prb_d  = r_prb.to(device)
                m_reg_d  = m_reg.to(device)
                m_prb_d  = m_prb.to(device)
                r_mar_d  = r_mar.to(device)
                m_mar_d  = m_mar.to(device)
                cf_d     = cf.to(device)
                for arch, refiner_model in refiners:
                    try:
                        h_ref = run_refiner_forward(
                            arch, refiner_model,
                            h_p, h_l, ct_dev, cf_d, ct_mask, emb_w,
                            r_reg_d, r_prb_d, m_reg_d, m_prb_d, r_mar_d, m_mar_d)
                        ref_lgt    = h_ref.float() @ emb_w.T
                        ref_log_p  = F.log_softmax(ref_lgt, dim=1)
                        ref_nlls_b = (-ref_log_p[torch.arange(B, device=device), gt_dev]).cpu().numpy()
                        refiner_nlls[arch] = ref_nlls_b
                    except Exception as e:
                        print(f"  WARNING: refiner {arch} forward failed on shard {si}: {e}")

            # ── Assemble rows ────────────────────────────────────────────────
            sp_sl  = split_arr[start:end]
            ty_sl  = type_arr[start:end]

            for i in range(B):
                row: Dict = {
                    "row_id":             int(total_rows + start + i),
                    "shard_id":           int(si),
                    "row_in_shard":       int(start + i),
                    "gold_token_id":      int(gt_b[i]),
                    "covered":            bool(cov_b[i]),
                    "split_type":         SPLIT_NAMES.get(int(sp_sl[i]), "other"),
                    "type_label":         TYPE_NAMES.get(int(ty_sl[i]), "other"),
                    "candidate_count":    int(cand_cnt[i]),
                    # Base full-vocab
                    "base_full_vocab_nll":       float(gold_nlls[i]),
                    "base_gold_prob":            float(gold_probs[i]),
                    "base_gold_logit":           float(gold_lgts[i]),
                    "base_top1_token_id":        int(top1_ids[i]),
                    "base_top1_prob":            float(top1_probs[i]),
                    "base_top1_logit":           float(top1_lgts[i]),
                    "base_margin_top1_minus_gold": float(margin_top1_minus_gold[i]),
                    # Gold rank
                    "gold_rank_full_vocab":      int(gold_ranks[i]),
                    "gold_in_top1":              bool(in_topk[1][i]),
                    "gold_in_top5":              bool(in_topk[5][i]),
                    "gold_in_top10":             bool(in_topk[10][i]),
                    "gold_in_top50":             bool(in_topk[50][i]),
                    "gold_in_top100":            bool(in_topk[100][i]),
                    "gold_in_top500":            bool(in_topk[500][i]),
                    "gold_in_top1000":           bool(in_topk[1000][i]),
                    # Region
                    "gold_fine_region":          int(gold_reg[i]),
                    "gold_superregion":          int(gold_sup[i]),
                    "base_top1_fine_region":     int(top1_reg[i]),
                    "base_top1_superregion":     int(top1_sup[i]),
                    "base_top1_same_fine":       bool(top1_same_fine[i]),
                    "base_top1_same_super":      bool(top1_same_super[i]),
                    # Router
                    "router_top1_region":        int(r_reg_np[i, 0]),
                    "router_top1_prob":          float(r_prb_np[i, 0]),
                    "router_entropy":            float(router_ent[i]),
                    "router_margin":             float(r_mar_np[i]),
                    "gold_rank_in_router":       int(r_gold_rank[i]),
                    "gold_in_router_top1":       bool(r_in_topk[1][i]),
                    "gold_in_router_top4":       bool(r_in_topk[4][i]),
                    "gold_in_router_top8":       bool(r_in_topk[8][i]),
                    "gold_in_router_top12":      bool(r_in_topk[12][i]),
                    # Memory
                    "memory_top1_region":        int(m_reg_np[i, 0]),
                    "memory_top1_prob":          float(m_prb_np[i, 0]),
                    "memory_entropy":            float(memory_ent[i]),
                    "memory_margin":             float(m_mar_np[i]),
                    "gold_rank_in_memory":       int(m_gold_rank[i]),
                    "gold_in_memory_top1":       bool(m_in_topk[1][i]),
                    "gold_in_memory_top4":       bool(m_in_topk[4][i]),
                    "gold_in_memory_top8":       bool(m_in_topk[8][i]),
                    "gold_in_memory_top12":      bool(m_in_topk[12][i]),
                    # Router-memory agreement
                    "router_memory_top1_agree":  bool(rm_top1_agree[i]),
                    "router_memory_top4_overlap": float(rm_top4_ov[i]),
                    "router_memory_top8_overlap": float(rm_top8_ov[i]),
                    "router_correct_memory_wrong": bool(r_ok_m_wrong[i]),
                    "memory_correct_router_wrong": bool(m_ok_r_wrong[i]),
                    "both_correct_region":        bool(both_ok[i]),
                    "both_wrong_region":          bool(both_wrong[i]),
                    # Candidates
                    "gold_in_candidate_set":     bool(cov_b[i]),
                    "gold_candidate_idx":        int(gidx_np[i]),
                    "gold_cand_rank":            int(gidx_np[i] + 1) if gidx_np[i] >= 0 else -1,
                    "gold_cand_base_logit":      float(gold_cand_lgt[i]) if not np.isnan(gold_cand_lgt[i]) else None,
                    "cand_same_fine_region":     int(cand_same_fine[i]),
                    "cand_same_super_region":    int(cand_same_super[i]),
                }
                # Refiner NLLs and delta_nlls
                for arch, _ in refiners:
                    if arch in refiner_nlls:
                        rnll = float(refiner_nlls[arch][i])
                        row[f"refined_nll_{arch}"]   = rnll
                        row[f"delta_nll_{arch}"]     = float(gold_nlls[i]) - rnll
                    else:
                        row[f"refined_nll_{arch}"]   = None
                        row[f"delta_nll_{arch}"]     = None
                rows.append(row)

        total_rows += N
        print(f"  shard {si:05d}  n={N}  total={total_rows:,}  {time.time()-t0:.1f}s")

    print(f"[audit] Complete. {total_rows:,} rows  {time.time()-t0:.1f}s")
    return rows


# ── Recoverability labels ─────────────────────────────────────────────────────

def add_recoverability_labels(df) -> None:
    """Add derived boolean label columns in-place."""
    import pandas as pd

    hard_mask = df["split_type"].isin(["boundary", "tight"])

    df["recover_base_top10"]   = df["gold_rank_full_vocab"] <= 10
    df["recover_base_top50"]   = df["gold_rank_full_vocab"] <= 50
    df["recover_base_top100"]  = df["gold_rank_full_vocab"] <= 100
    df["recover_base_top500"]  = df["gold_rank_full_vocab"] <= 500
    df["recover_base_top1000"] = df["gold_rank_full_vocab"] <= 1000

    df["recover_router_top8"] = df["gold_in_router_top8"]
    df["recover_memory_top8"] = df["gold_in_memory_top8"]
    df["recover_any_region_top8"] = df["gold_in_router_top8"] | df["gold_in_memory_top8"]

    df["router_right_memory_wrong"] = df["router_correct_memory_wrong"]
    df["memory_right_router_wrong"] = df["memory_correct_router_wrong"]
    df["router_memory_disagree"]    = ~df["router_memory_top1_agree"]

    df["region_right_token_wrong"] = (
        (df["gold_in_router_top8"] | df["gold_in_memory_top8"]) &
        (df["gold_rank_full_vocab"] > 10)
    )
    df["region_wrong"] = (
        ~df["gold_in_router_top12"] & ~df["gold_in_memory_top12"]
    )
    df["memory_useful"] = (
        df["memory_correct_router_wrong"] |
        (df["gold_in_memory_top8"] & ~df["gold_in_router_top8"])
    )
    df["router_useful"] = (
        df["router_correct_memory_wrong"] |
        (df["gold_in_router_top8"] & ~df["gold_in_memory_top8"])
    )

    df["hard_recoverable"] = hard_mask & (
        (df["gold_rank_full_vocab"] <= 1000) |
        df["gold_in_router_top8"] |
        df["gold_in_memory_top8"] |
        df["covered"]
    )
    df["hard_unrecoverable"] = hard_mask & ~df["hard_recoverable"]


# ── Bucket summary ────────────────────────────────────────────────────────────

def _bucket_stats(df_bucket, df_all, refiner_arches: List[str]) -> Dict:
    n    = len(df_bucket)
    frac = n / max(len(df_all), 1)
    if n == 0:
        return {"n": 0, "frac_of_val": 0.0}

    stats: Dict = {
        "n":               n,
        "frac_of_val":     float(frac),
        "mean_base_nll":   float(df_bucket["base_full_vocab_nll"].mean()),
        "total_nll":       float(df_bucket["base_full_vocab_nll"].sum()),
        "loss_mass_share": float(df_bucket["base_full_vocab_nll"].sum() /
                                  max(df_all["base_full_vocab_nll"].sum(), 1e-9)),
        "mean_gold_rank":  float(df_bucket["gold_rank_full_vocab"].mean()),
        "pct_top10":       float(df_bucket["gold_in_top10"].mean()),
        "pct_top100":      float(df_bucket["gold_in_top100"].mean()),
        "pct_top1000":     float(df_bucket["gold_in_top1000"].mean()),
        "coverage":        float(df_bucket["covered"].mean()),
        "mean_cand_count": float(df_bucket["candidate_count"].mean()),
        "mean_router_ent": float(df_bucket["router_entropy"].mean()),
        "mean_router_margin": float(df_bucket["router_margin"].mean()),
        "mean_memory_ent": float(df_bucket["memory_entropy"].mean()),
        "mean_memory_margin": float(df_bucket["memory_margin"].mean()),
        "pct_router_gold_top8":  float(df_bucket["gold_in_router_top8"].mean()),
        "pct_memory_gold_top8":  float(df_bucket["gold_in_memory_top8"].mean()),
        "pct_both_region_wrong": float(df_bucket["both_wrong_region"].mean()),
        "pct_region_rt_tok_wrong": float(df_bucket["region_right_token_wrong"].mean()),
    }
    for arch in refiner_arches:
        col = f"delta_nll_{arch}"
        if col in df_bucket.columns:
            valid = df_bucket[col].dropna()
            if len(valid) > 0:
                stats[f"mean_delta_nll_{arch}"]  = float(valid.mean())
                stats[f"win_rate_{arch}"]         = float((valid > 0).mean())
                stats[f"big_win_rate_{arch}"]     = float((valid > 0.1).mean())
                stats[f"big_loss_rate_{arch}"]    = float((valid < -0.1).mean())
    return stats


def compute_bucket_summary(df, refiner_arches: List[str]):
    import pandas as pd

    BUCKETS = {
        "all":                    df["covered"] | True,
        "core":                   df["split_type"] == "core",
        "medium":                 df["split_type"] == "medium",
        "boundary":               df["split_type"].isin(["boundary", "tight"]),
        "tight":                  df["split_type"] == "tight",
        "type_A":                 df["type_label"] == "A",
        "type_B":                 df["type_label"] == "B",
        "type_C":                 df["type_label"] == "C",
        "covered":                df["covered"],
        "uncovered":              ~df["covered"],
        "base_gold_top10":        df["gold_rank_full_vocab"] <= 10,
        "base_gold_top100":       df["gold_rank_full_vocab"] <= 100,
        "base_gold_top1000":      df["gold_rank_full_vocab"] <= 1000,
        "base_gold_far_gt1000":   df["gold_rank_full_vocab"] > 1000,
        "router_gold_top8":       df["gold_in_router_top8"],
        "memory_gold_top8":       df["gold_in_memory_top8"],
        "router_or_memory_top8":  df["gold_in_router_top8"] | df["gold_in_memory_top8"],
        "router_memory_both_wrong": df["both_wrong_region"],
        "region_right_token_wrong": df["region_right_token_wrong"],
        "region_wrong":           df["region_wrong"],
        "memory_useful":          df["memory_useful"],
        "router_useful":          df["router_useful"],
        "router_memory_disagree": df["router_memory_disagree"],
        "both_region_right":      df["both_correct_region"],
        "both_region_wrong":      df["both_wrong_region"],
        "hard_recoverable":       df["hard_recoverable"],
        "hard_unrecoverable":     df["hard_unrecoverable"],
    }

    summary_rows = []
    for name, mask in BUCKETS.items():
        sub   = df[mask]
        stats = _bucket_stats(sub, df, refiner_arches)
        stats["bucket"] = name
        summary_rows.append(stats)

    cols = ["bucket", "n", "frac_of_val", "mean_base_nll", "loss_mass_share",
            "mean_gold_rank", "pct_top10", "pct_top100", "pct_top1000",
            "coverage", "mean_cand_count",
            "mean_router_ent", "mean_router_margin",
            "mean_memory_ent", "mean_memory_margin",
            "pct_router_gold_top8", "pct_memory_gold_top8",
            "pct_both_region_wrong", "pct_region_rt_tok_wrong"]
    for arch in refiner_arches:
        cols += [f"mean_delta_nll_{arch}", f"win_rate_{arch}",
                 f"big_win_rate_{arch}", f"big_loss_rate_{arch}"]

    bdf = pd.DataFrame(summary_rows)
    for c in cols:
        if c not in bdf.columns:
            bdf[c] = None
    return bdf[[c for c in cols if c in bdf.columns]]


# ── Loss mass report ──────────────────────────────────────────────────────────

def build_loss_mass_report(bucket_df) -> str:
    all_row  = bucket_df[bucket_df["bucket"] == "all"]
    mean_nll = all_row["mean_base_nll"].values
    n_all    = all_row["n"].values
    mean_all   = float(mean_nll[0]) if len(mean_nll) else float("nan")
    total_loss = mean_all * float(n_all[0]) if len(n_all) else float("nan")

    REPORT_BUCKETS = ["boundary", "tight", "medium", "core",
                      "hard_recoverable", "hard_unrecoverable",
                      "region_right_token_wrong", "region_wrong",
                      "memory_useful", "both_region_wrong"]

    lines = [
        "# Loss Mass Report",
        "",
        f"Total validation NLL mass: {total_loss:.1f}  |  Mean NLL: {mean_all:.6f}",
        "",
        f"| Bucket | Row share | Loss mass share | Mean NLL | Excess vs all-mean |",
        f"|--------|-----------|-----------------|----------|--------------------|",
    ]
    for bname in REPORT_BUCKETS:
        row = bucket_df[bucket_df["bucket"] == bname]
        if row.empty:
            continue
        row = row.iloc[0]
        n_share   = float(row.get("frac_of_val",    0))
        lm_share  = float(row.get("loss_mass_share", 0))
        m_nll     = float(row.get("mean_base_nll",   float("nan")))
        excess    = m_nll - mean_all
        lines.append(
            f"| {bname:<30} | {n_share:>9.3f} | {lm_share:>15.3f} "
            f"| {m_nll:>8.4f} | {excess:>+18.4f} |"
        )
    return "\n".join(lines) + "\n"


# ── Top wins/losses ───────────────────────────────────────────────────────────

def build_top_examples(df, refiner_arch: str, tokenizer, n: int = 100):
    col = f"delta_nll_{refiner_arch}"
    if col not in df.columns:
        return None, None, None

    valid = df[df[col].notna()].copy()
    if len(valid) == 0:
        return None, None, None

    key_cols = ["row_id", "gold_token_id", "base_full_vocab_nll",
                f"refined_nll_{refiner_arch}", col,
                "gold_rank_full_vocab", "gold_fine_region",
                "gold_in_router_top8", "gold_in_memory_top8",
                "split_type", "type_label", "candidate_count",
                "router_margin", "memory_margin",
                "base_top1_token_id"]

    available = [c for c in key_cols if c in valid.columns]
    wins    = valid.nlargest(n, col)[available].copy()
    losses  = valid.nsmallest(n, col)[available].copy()
    neutral = valid[(valid[col].abs() < 0.01)].head(n)[available].copy()

    def add_strings(sub):
        if tokenizer is not None:
            sub["gold_token_str"]  = sub["gold_token_id"].apply(
                lambda t: tokenizer.decode([int(t)]))
            sub["base_top1_str"]   = sub["base_top1_token_id"].apply(
                lambda t: tokenizer.decode([int(t)]))
        return sub

    wins    = add_strings(wins)
    losses  = add_strings(losses)
    neutral = add_strings(neutral)
    return wins, losses, neutral


def build_examples_md(wins, losses, tokenizer) -> str:
    lines = ["# Top Bridge Examples", ""]
    if wins is None:
        lines += ["No refiner comparison available.", ""]
        if tokenizer is None:
            lines += [
                "NOTE: context unavailable because current dataset lacks input_ids/context tokens.",
                "Gold token strings shown where tokenizer is loaded.",
            ]
        return "\n".join(lines)

    lines += ["## Top 20 Wins (largest NLL decrease)", ""]
    for _, row in wins.head(20).iterrows():
        gold_str = row.get("gold_token_str", f"[{row['gold_token_id']}]")
        top1_str = row.get("base_top1_str",  f"[{row['base_top1_token_id']}]")
        lines.append(
            f"- gold=`{gold_str}`  base_top1=`{top1_str}`  "
            f"rank={row['gold_rank_full_vocab']}  "
            f"delta={row.get(list(row.keys())[-1], '?'):+.4f}  "
            f"split={row.get('split_type','?')}  "
            f"r8={row.get('gold_in_router_top8','?')}  m8={row.get('gold_in_memory_top8','?')}"
        )
    lines += ["", "## Top 20 Losses (largest NLL increase)", ""]
    for _, row in losses.head(20).iterrows():
        gold_str = row.get("gold_token_str", f"[{row['gold_token_id']}]")
        top1_str = row.get("base_top1_str",  f"[{row['base_top1_token_id']}]")
        lines.append(
            f"- gold=`{gold_str}`  base_top1=`{top1_str}`  "
            f"rank={row['gold_rank_full_vocab']}  "
            f"delta={row.get(list(row.keys())[-1], '?'):+.4f}  "
            f"split={row.get('split_type','?')}  "
            f"r8={row.get('gold_in_router_top8','?')}  m8={row.get('gold_in_memory_top8','?')}"
        )
    lines += [
        "",
        "NOTE: context unavailable because current dataset lacks input_ids/context tokens.",
        "      No surrounding text shown. Only token IDs, ranks, and region signals available.",
    ]
    return "\n".join(lines) + "\n"


# ── Final report ──────────────────────────────────────────────────────────────

def generate_final_report(df, bucket_df, context_avail: Dict,
                           refiner_arches: List[str], args) -> str:
    import pandas as pd

    def bval(bucket, col, default=float("nan")):
        rows = bucket_df[bucket_df["bucket"] == bucket]
        return float(rows[col].values[0]) if (not rows.empty and col in rows.columns) else default

    n_total     = len(df)
    n_boundary  = int(df["split_type"].isin(["boundary", "tight"]).sum())
    n_hard_rec  = int(df["hard_recoverable"].sum())
    n_hard_unrec = int(df["hard_unrecoverable"].sum())

    boundary_lm   = bval("boundary",        "loss_mass_share")
    tight_lm      = bval("tight",           "loss_mass_share")
    boundary_top10 = bval("boundary",       "pct_top10")
    boundary_top100 = bval("boundary",      "pct_top100")
    boundary_top1000 = bval("boundary",     "pct_top1000")
    boundary_router8 = bval("boundary",     "pct_router_gold_top8")
    boundary_mem8    = bval("boundary",     "pct_memory_gold_top8")
    both_wrong_lm   = bval("both_region_wrong", "loss_mass_share")
    rrtw_lm         = bval("region_right_token_wrong", "loss_mass_share")
    mem_useful_pct  = bval("memory_useful", "frac_of_val")
    rtr_useful_pct  = bval("router_useful", "frac_of_val")

    has_refiner = len(refiner_arches) > 0
    refiner_summary = ""
    for arch in refiner_arches:
        bw_delta  = bval("hard_recoverable",   f"mean_delta_nll_{arch}")
        ubw_delta = bval("hard_unrecoverable",  f"mean_delta_nll_{arch}")
        bnd_win   = bval("boundary",            f"win_rate_{arch}")
        bnd_bloss = bval("boundary",            f"big_loss_rate_{arch}")
        refiner_summary += (
            f"\n### {arch}\n"
            f"- hard_recoverable mean delta_nll = {bw_delta:+.4f}\n"
            f"- hard_unrecoverable mean delta_nll = {ubw_delta:+.4f}\n"
            f"- boundary win_rate = {bnd_win:.3f}  big_loss_rate = {bnd_bloss:.3f}\n"
        )

    # --- infer answers from data ---
    def q_answer(condition, yes, no):
        return yes if condition else no

    q1 = f"Yes ({boundary_lm:.1%} of total NLL)." if boundary_lm > 0.15 else f"Moderate ({boundary_lm:.1%} of total NLL)."
    q2 = (f"Top-10: {boundary_top10:.1%}, top-100: {boundary_top100:.1%}, "
          f"top-1000: {boundary_top1000:.1%} of boundary rows have gold in those ranks.")
    q3 = (f"Router top-8: {boundary_router8:.1%}, Memory top-8: {boundary_mem8:.1%} "
          f"of boundary rows.")

    if boundary_router8 > 0.6 and rrtw_lm > 0.05:
        q4 = "Mostly token disambiguation (region routing is often correct, but token prediction fails)."
    elif boundary_router8 < 0.4 and both_wrong_lm > 0.1:
        q4 = "Mostly region routing failure (both router and memory miss gold region)."
    else:
        q4 = "Mixed: both region routing and token disambiguation failures."

    q5 = f"Memory is independently useful in {mem_useful_pct:.1%} of val rows."
    q6 = f"Router is independently useful in {rtr_useful_pct:.1%} of val rows."

    if has_refiner:
        if all(bval("hard_recoverable",   f"mean_delta_nll_{a}") >
               bval("hard_unrecoverable", f"mean_delta_nll_{a}") for a in refiner_arches):
            q7 = "YES — Bridge gains are concentrated in hard_recoverable rows, not hard_unrecoverable."
        else:
            q7 = "Inconclusive — Bridge gains are similar for both recoverable and unrecoverable rows."
        worst_bucket = max(
            [(a, bval("boundary", f"big_loss_rate_{a}")) for a in refiner_arches],
            key=lambda x: x[1], default=("?", 0)
        )
        q8 = f"Bridge most often hurts on boundary rows where big_loss_rate = {worst_bucket[1]:.3f} for {worst_bucket[0]}."
    else:
        q7 = "No refiner loaded — cannot answer."
        q8 = "No refiner loaded — cannot answer."

    q9 = (f"Boundary gate covers {n_boundary/max(n_total,1):.1%} of val rows. "
          f"{'Broad — consider tightening to tight_boundary.' if n_boundary/max(n_total,1) > 0.20 else 'Reasonable breadth.'}")
    q10 = ("NO — " if not context_avail["context_available"] else "YES — ") + context_avail["reason"]

    if not context_avail["context_available"]:
        q11 = (
            "Before more architecture work: (1) build a dataset variant that saves input_ids "
            "alongside h_layers so causal multi-write can be tested. "
            "(2) Based on the audit: " +
            ("focus on token disambiguation (memory-neighbor next-tokens, contrastive loss) "
             if rrtw_lm > both_wrong_lm else
             "focus on region routing (stronger memory-router conflict resolver, larger retrieval k).")
        )
    else:
        q11 = "Context is available — test true causal_multi_write."

    lines = [
        "# Hard-Token Failure Anatomy — Final Report",
        f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Executive Summary",
        "",
        f"- Validation set: {n_total:,} rows ({CANONICAL_NUM_EXAMPLES:,} canonical).",
        f"- Boundary rows: {n_boundary:,} ({n_boundary/max(n_total,1):.1%} of val).",
        f"- Hard-recoverable: {n_hard_rec:,}  Hard-unrecoverable: {n_hard_unrec:,}.",
        f"- Full-vocab base NLL all: {df['base_full_vocab_nll'].mean():.6f} "
        f"  (canonical: {KNOWN_BASE_NLL_ALL}).",
        f"- Refiners evaluated: {refiner_arches if refiner_arches else 'none'}.",
        "",
        "## 2. Dataset / Context Availability",
        "",
        f"- context_available = {context_avail['context_available']}",
        f"- can_reconstruct_live_forward = {context_avail['can_reconstruct_live_forward']}",
        f"- Reason: {context_avail['reason']}",
        f"- Cand shard keys: {context_avail['cand_keys']}",
        f"- Feat shard keys: {context_avail['feat_keys']}",
        "",
        "## 3. Loss Mass: Which Rows Dominate NLL?",
        "",
        build_loss_mass_report(bucket_df),
        "",
        "## 4. Base Recoverability by Gold Rank",
        "",
        f"Boundary rows: top-10={boundary_top10:.3f}, top-100={boundary_top100:.3f}, "
        f"top-1000={boundary_top1000:.3f}",
        f"All rows: top-10={bval('all','pct_top10'):.3f}, top-100={bval('all','pct_top100'):.3f}, "
        f"top-1000={bval('all','pct_top1000'):.3f}",
        "",
        "## 5. Region Recoverability by Router/Memory",
        "",
        f"- Boundary: router_top8={boundary_router8:.3f}, memory_top8={boundary_mem8:.3f}",
        f"- All: router_top8={bval('router_gold_top8','frac_of_val'):.3f}, "
        f"memory_top8={bval('memory_gold_top8','frac_of_val'):.3f}",
        f"- Both region wrong (all): {bval('both_region_wrong','frac_of_val'):.3f}",
        f"- Both region right (all): {bval('both_region_right','frac_of_val'):.3f}",
        "",
        "## 6. Router vs Memory Conflict Analysis",
        "",
        f"- Memory useful (memory right, router wrong): {mem_useful_pct:.3f} of val",
        f"- Router useful (router right, memory wrong): {rtr_useful_pct:.3f} of val",
        f"- Router-memory disagree (top-1): {bval('router_memory_disagree','frac_of_val'):.3f} of val",
        "",
        "## 7. Region-Right-Token-Wrong vs Region-Wrong",
        "",
        f"- region_right_token_wrong loss mass: {rrtw_lm:.3f}",
        f"- both_region_wrong loss mass:        {both_wrong_lm:.3f}",
        "",
        "## 8. Bridge/Refiner Win-Loss Analysis",
        "",
        refiner_summary if has_refiner else "No refiner loaded.",
        "",
        "## 9. Failure Buckets",
        "",
        "See bucket_summary.csv for full table.",
        "",
        "## 10. Diagnostic Q&A",
        "",
        f"**Q1.  Boundary rows large share of loss mass?**  {q1}",
        f"**Q2.  Gold rank among boundary rows?**  {q2}",
        f"**Q3.  Gold region in router/memory top-8?**  {q3}",
        f"**Q4.  Nature of hard-token failure?**  {q4}",
        f"**Q5.  Does memory provide unique signal?**  {q5}",
        f"**Q6.  Does router provide unique signal?**  {q6}",
        f"**Q7.  Bridge better on recoverable subset?**  {q7}",
        f"**Q8.  Which subset does Bridge hurt?**  {q8}",
        f"**Q9.  Is boundary gate too broad?**  {q9}",
        f"**Q10. Is live causal forward possible?**  {q10}",
        f"**Q11. Next architecture/data action?**  {q11}",
        "",
        "## 11. Concrete Next Actions",
        "",
    ]

    # Interpret pattern
    if not context_avail["context_available"]:
        lines += [
            "1. **Build context-id dataset**: Save `input_ids` alongside `h_layers` to enable "
            "   true causal_multi_write and richer token-level signals.",
        ]
    if rrtw_lm > both_wrong_lm:
        lines += [
            "2. **Address token disambiguation**: Add memory-neighbor next-token distribution, "
            "   contrastive top-confuser loss, or token×candidate interaction.",
        ]
    else:
        lines += [
            "2. **Address region routing**: Strengthen memory-router conflict resolution, "
            "   increase retrieval k, or train a learned recoverability gate.",
        ]
    if has_refiner:
        worst_bname = max(
            [(a, bval("hard_unrecoverable", f"big_loss_rate_{a}")) for a in refiner_arches],
            key=lambda x: x[1], default=("?", 0)
        )
        lines += [
            f"3. **Gate the refiner**: Don't apply Bridge to hard_unrecoverable rows "
            f"(big_loss_rate={worst_bname[1]:.3f} on that subset for {worst_bname[0]}).",
        ]
    lines += [
        "4. **Do not launch new architecture** until context-id dataset is available "
        "   OR token-disambiguation signals are added.",
    ]
    return "\n".join(lines) + "\n"


# ── Writers ───────────────────────────────────────────────────────────────────

def write_context_availability(ca: Dict, out_dir: str) -> None:
    with open(os.path.join(out_dir, "context_availability.json"), "w") as f:
        json.dump(ca, f, indent=2)
    lines = [
        "# Context Availability",
        "",
        f"- **context_available**: {ca['context_available']}",
        f"- **can_reconstruct_live_forward**: {ca['can_reconstruct_live_forward']}",
        f"- **keys_found**: {ca['keys_found']}",
        f"- **cand_keys**: {ca['cand_keys']}",
        f"- **feat_keys**: {ca['feat_keys']}",
        "",
        f"**Reason**: {ca['reason']}",
    ]
    with open(os.path.join(out_dir, "context_availability.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_bucket_summary(bucket_df, out_dir: str) -> None:
    bucket_df.to_csv(os.path.join(out_dir, "bucket_summary.csv"), index=False)
    lines = ["# Bucket Summary", "", ""]
    # Markdown table
    cols = list(bucket_df.columns)
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in bucket_df.iterrows():
        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    with open(os.path.join(out_dir, "bucket_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args) -> None:
    import pandas as pd

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = {**vars(args), "run_start": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"[audit] Output dir: {args.output_dir}")

    # ── Context availability ──────────────────────────────────────────────────
    print("[audit] Checking context availability ...")
    ca = check_context_availability(args.val_cand_dir, args.val_feat_dir)
    write_context_availability(ca, args.output_dir)
    print(f"  context_available={ca['context_available']}  "
          f"can_live_forward={ca['can_reconstruct_live_forward']}")
    if not ca["context_available"]:
        print(f"  {ca['reason']}")

    # ── Backbone ──────────────────────────────────────────────────────────────
    device = torch.device(args.device)
    print(f"[audit] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, cfg_dict, _ = load_small_backbone_and_probe(args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    tok_emb_w = (backbone.token_emb.weight.detach().cpu()
                 if hasattr(backbone, "token_emb")
                 else backbone.transformer.wte.weight.detach().cpu())
    V = tok_emb_w.shape[0]
    print(f"  d_model={d_model}  vocab={V}")

    # ── Region maps ───────────────────────────────────────────────────────────
    n_fine, n_super = 128, 24
    if args.val_cand_dir:
        cfg_path = os.path.join(args.val_cand_dir, "dataset_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                dc = json.load(f)
            n_fine  = dc.get("n_fine",  n_fine)
            n_super = dc.get("n_super", n_super)
    r2s_np = load_r2s(args.super_map, n_fine) if args.super_map else np.zeros(n_fine, dtype=np.int32)
    n_super = int(r2s_np.max()) + 1 if r2s_np.max() >= 0 else n_super
    t2r = load_token_to_region(args.region_map, V)
    print(f"  n_fine={n_fine}  n_super={n_super}  t2r={'loaded' if t2r is not None else 'not found'}")

    # ── Feature layer IDs ─────────────────────────────────────────────────────
    feat_cfg_path = os.path.join(args.val_feat_dir, "config.json")
    feat_layer_ids: List[int] = [0, 2, 4, 5, -1]
    if os.path.isfile(feat_cfg_path):
        with open(feat_cfg_path) as f:
            fc = json.load(f)
        feat_layer_ids = fc.get("layer_ids", feat_layer_ids)
    print(f"  feat_layer_ids={feat_layer_ids}")

    # ── Optional refiners ─────────────────────────────────────────────────────
    refiner_ckpts = [p for p in [
        args.bridge_v1_ckpt,
        args.midlayer_ckpt,
        args.explicit_refiner_ckpt,
        args.running_bridge_ckpt,
    ] if p]
    refiners: List[Tuple[str, object]] = []
    if refiner_ckpts:
        print("[audit] Loading optional refiners ...")
        for ckpt in refiner_ckpts:
            arch, model = try_load_refiner(ckpt, d_model, n_fine, n_super, r2s_np,
                                           feat_layer_ids, device)
            if arch and model:
                refiners.append((arch, model))
    if not refiners:
        print("[audit] Optional refiner comparison skipped.")

    # ── Tokenizer (for string labels) ────────────────────────────────────────
    tokenizer = None
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        print("[audit] GPT-2 tokenizer loaded.")
    except Exception:
        print("[audit] Tokenizer not available — token strings will be skipped.")

    # ── Main audit ────────────────────────────────────────────────────────────
    print("[audit] Running row-level audit ...")
    rows = audit_val_rows(tok_emb_w, r2s_np, t2r, refiners, args, device)

    # ── DataFrame ─────────────────────────────────────────────────────────────
    print("[audit] Building DataFrame ...")
    df = pd.DataFrame(rows)

    # Add token strings if tokenizer available
    if tokenizer is not None:
        df["gold_token_str"]  = df["gold_token_id"].apply(lambda t: tokenizer.decode([int(t)]))
        df["base_top1_str"]   = df["base_top1_token_id"].apply(lambda t: tokenizer.decode([int(t)]))

    add_recoverability_labels(df)

    # Write parquet + csv.gz
    pq_path  = os.path.join(args.output_dir, "val_failure_audit.parquet")
    csv_path = os.path.join(args.output_dir, "val_failure_audit.csv.gz")
    print(f"[audit] Writing {pq_path} ...")
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"  WARNING: parquet write failed ({e}). Install pyarrow.")
    print(f"[audit] Writing {csv_path} ...")
    df.to_csv(csv_path, index=False, compression="gzip")

    # ── Bucket summary ────────────────────────────────────────────────────────
    refiner_arches = [a for a, _ in refiners]
    print("[audit] Computing bucket summary ...")
    bucket_df = compute_bucket_summary(df, refiner_arches)
    write_bucket_summary(bucket_df, args.output_dir)

    # ── Loss mass report ──────────────────────────────────────────────────────
    lm_report = build_loss_mass_report(bucket_df)
    lm_path = os.path.join(args.output_dir, "loss_mass_report.md")
    with open(lm_path, "w") as f:
        f.write(lm_report)
    print(f"[audit] {lm_path}")

    # ── Top examples ─────────────────────────────────────────────────────────
    if args.write_examples:
        primary_arch = refiner_arches[0] if refiner_arches else None
        if primary_arch:
            wins, losses, neutral = build_top_examples(df, primary_arch, tokenizer)
            if wins is not None:
                wins.to_csv(os.path.join(args.output_dir, "top_100_bridge_wins.csv"), index=False)
                losses.to_csv(os.path.join(args.output_dir, "top_100_bridge_losses.csv"), index=False)
                if neutral is not None:
                    neutral.to_csv(os.path.join(args.output_dir, "top_100_bridge_neutral.csv"), index=False)
        else:
            wins, losses = None, None
        ex_md = build_examples_md(wins, losses if primary_arch else None, tokenizer)
        with open(os.path.join(args.output_dir, "top_examples.md"), "w") as f:
            f.write(ex_md)

    # ── Final report ──────────────────────────────────────────────────────────
    print("[audit] Generating final report ...")
    report = generate_final_report(df, bucket_df, ca, refiner_arches, args)
    report_path = os.path.join(args.output_dir, "final_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("  HARD-TOKEN FAILURE AUDIT COMPLETE")
    print("=" * 70)
    n_bnd = int(df["split_type"].isin(["boundary", "tight"]).sum())
    print(f"  Total rows          : {len(df):,}")
    print(f"  Boundary rows       : {n_bnd:,} ({n_bnd/max(len(df),1):.1%})")
    bnd = bucket_df[bucket_df["bucket"] == "boundary"]
    if not bnd.empty:
        print(f"  Boundary loss mass  : {bnd.iloc[0]['loss_mass_share']:.3f}")
        print(f"  Boundary top-10 %   : {bnd.iloc[0]['pct_top10']:.3f}")
        print(f"  Router top-8 %      : {bnd.iloc[0]['pct_router_gold_top8']:.3f}")
        print(f"  Memory top-8 %      : {bnd.iloc[0]['pct_memory_gold_top8']:.3f}")
    print(f"  Context available   : {ca['context_available']}")
    print(f"  Total elapsed       : {time.time()-t0:.1f}s")
    print(f"\n  Output: {args.output_dir}/final_report.md")
    print("=" * 70)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",     required=True)
    p.add_argument("--val_cand_dir",   required=True)
    p.add_argument("--val_feat_dir",   required=True)
    p.add_argument("--baseline_json",  default=None)
    p.add_argument("--super_map",      default=None)
    p.add_argument("--region_map",     default=None)
    p.add_argument("--output_dir",     required=True)

    # Optional refiner checkpoints
    p.add_argument("--bridge_v1_ckpt",        default=None)
    p.add_argument("--midlayer_ckpt",         default=None)
    p.add_argument("--explicit_refiner_ckpt", default=None)
    p.add_argument("--running_bridge_ckpt",   default=None)

    p.add_argument("--eval_batch_size", type=int,   default=64)
    p.add_argument("--topk_rank",       type=int,   default=1000,
                   help="Top-k tokens to materialise for in-topk metrics. "
                        "Gold rank uses exact count, not sort.")
    p.add_argument("--margin_thresh",   type=float, default=0.1)
    p.add_argument("--entropy_thresh",  type=float, default=2.0)
    p.add_argument("--include_context_check", action="store_true", default=True)
    p.add_argument("--write_examples",  action="store_true")
    p.add_argument("--device",          default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
