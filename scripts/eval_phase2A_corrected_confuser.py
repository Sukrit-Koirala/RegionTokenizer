#!/usr/bin/env python3
"""
eval_phase2A_corrected_confuser.py — Phase 2A.1: Corrected Region-Meaning Confuser Diagnostic

Fixes the split-region scoring bug in the original Phase 2A evaluation.

Old (flawed) metric:
  score_gold = dot(U[gold], Z[variant_map[gold]])
  score_base = dot(U[base], Z[variant_map[base]])
  → shuffled/random can "win" by scoring base through a separate fake-region anchor

Corrected (forced same-anchor) metric:
  anchor = variant_map[gold]
  score_gold = dot(U[gold], Z[anchor])
  score_base = dot(U[base], Z[anchor])      ← same Z slot for both tokens
  → shuffled/random cannot cheat by splitting confuser pair into separate slots

All confuser slices defined with REAL tok_arr.
No training. No gradient updates.
"""

import argparse
import csv
import json
import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import torch

# ── Import model + helpers from training script ───────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from train_contextual_region_meaning_learner import (
    ContextualRegionMeaningLearner,
    load_shards,
    load_unembedding,
    load_region_maps,
    eval_router_baseline,
    _nanmean,
    _wcsv,
    _fmt,
)

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)

_EPS = 1e-9
_N_EXAMPLES = 50   # examples per report category


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_variant_model(variant, ckpt_path, phase2a_dir,
                       tok_arr_v, reg_arr, U_frozen,
                       n_regions, n_super, device):
    """Reconstruct model from config.json + checkpoint."""
    cfg_path = os.path.join(phase2a_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found in {phase2a_dir}")
    cfg = json.load(open(cfg_path))

    d_in       = max(int(cfg.get("d_in", 1)), 1)
    d_model    = int(cfg.get("d_model", 256))
    rem_dim    = int(cfg.get("region_emb_dim", 64))
    sem_dim    = int(cfg.get("super_emb_dim",  32))
    n_layers   = int(cfg.get("num_region_layers", 2))
    n_heads    = int(cfg.get("num_heads", 4))
    ff_mult    = int(cfg.get("ff_mult", 4))
    dropout    = 0.0   # eval: no dropout
    use_ctx    = (variant != "static_region_meaning")

    model = ContextualRegionMeaningLearner(
        tok_arr=tok_arr_v, reg_arr=reg_arr, U_frozen=U_frozen,
        n_regions=n_regions, n_super=n_super, d_in=d_in,
        d_model=d_model, region_emb_dim=rem_dim, super_emb_dim=sem_dim,
        num_region_layers=n_layers, num_heads=n_heads, ff_mult=ff_mult,
        dropout=dropout, use_context=use_ctx,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[load] {variant}: loaded {ckpt_path}  params={sum(p.numel() for p in model.parameters()):,}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Batch utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _make_eval_batch(data, idx, M):
    d_in = max(data.get("d_model", 1), 1)
    b = {
        "topk_ids": data["topk_ids"][idx],
        "topk_lgt": data["topk_lgt"][idx],
        "h": (data["h"][idx] if data["has_h"]
              else np.zeros((len(idx), d_in), np.float32)),
    }
    if data["has_router"]:
        b["router_reg"] = data["router_reg"][idx]
        b["router_prb"] = data["router_prb"][idx]
    return b


def _to_device(batch, device):
    return {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
            for k, v in batch.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Core evaluation for one variant
# ═══════════════════════════════════════════════════════════════════════════════

def eval_variant(variant, model,
                 data, M, R,
                 tok_arr_real, reg_arr_real, n_regions_real,
                 tok_arr_variant, n_regions_variant,
                 args, device):
    """
    Returns dict with per-row metric arrays and aggregate metrics.
    Confuser slices always use tok_arr_real.
    Pair scores use tok_arr_variant as the anchor for 'forced_gold_anchor'.
    """
    N     = len(data["gold"])
    gold  = data["gold"].astype(np.int64)
    topk  = data["topk_ids"].astype(np.int64)
    lgt   = data["topk_lgt"].astype(np.float32)
    Vr    = len(tok_arr_real)
    Vv    = len(tok_arr_variant)
    Rr    = len(reg_arr_real)
    sqrt_d = math.sqrt(model.d_model)

    # Pre-compute real region labels
    real_gold_reg = tok_arr_real[np.clip(gold, 0, Vr-1)].astype(np.int64)
    real_gold_reg[real_gold_reg >= n_regions_real] = -1

    base_top1    = topk[:, 0]
    real_base_reg= tok_arr_real[np.clip(base_top1, 0, Vr-1)].astype(np.int64)
    real_base_reg[real_base_reg >= n_regions_real] = -1

    real_gold_sreg= reg_arr_real[np.clip(np.clip(real_gold_reg, 0, Rr-1), 0, Rr-1)].astype(np.int64)
    real_base_sreg= reg_arr_real[np.clip(np.clip(real_base_reg, 0, Rr-1), 0, Rr-1)].astype(np.int64)
    unk_r  = n_regions_real; unk_s = int(reg_arr_real.max())

    base_correct = (base_top1 == gold)
    gip          = (topk == gold[:, None]).any(axis=1)
    gold_rank_a  = np.where(gip, (topk == gold[:, None]).argmax(axis=1), -1)
    bucketA      = ~base_correct & gip

    same_reg_real  = (real_gold_reg == real_base_reg) & (real_gold_reg >= 0) & (real_base_reg >= 0)
    same_sreg_real = ((real_gold_sreg == real_base_sreg)
                      & (real_gold_reg >= 0) & (real_base_reg >= 0)
                      & (real_gold_sreg < unk_s) & (real_base_sreg < unk_s))

    # Per-row result arrays
    old_pair          = np.full(N, np.nan)
    forced_pair       = np.full(N, np.nan)
    forced_real_pair  = np.full(N, np.nan)
    old_margin_arr    = np.full(N, np.nan)
    forced_margin_arr = np.full(N, np.nan)
    forced_real_margin= np.full(N, np.nan)
    base_pair_arr     = np.full(N, np.nan)   # base logit pair
    own_cand_acc_arr  = np.full(N, np.nan)   # split-sensitive own-region cand acc
    own_cand_nll_arr  = np.full(N, np.nan)   # split-sensitive own-region cand nll
    within_real_acc   = np.full(N, np.nan)
    within_real_nll   = np.full(N, np.nan)
    within_real_rank  = np.full(N, np.nan)
    base_wr_acc       = np.full(N, np.nan)
    base_wr_nll       = np.full(N, np.nan)
    base_wr_rank      = np.full(N, np.nan)
    # Region prediction
    region_pred_arr   = np.full(N, -1, dtype=np.int64)
    r_topk_arr        = np.empty((N, 16), dtype=np.int64)

    # Example accumulators
    ex_real_beats_shuf  = []   # rows where forced_pair=True AND base_pair=False
    ex_shuf_cheat       = []   # rows where old_pair=True but forced_pair=False
    ex_real_fails       = []   # rows where forced_pair=False (real model)

    model.eval()
    with torch.no_grad():
        for s in range(0, N, args.batch_size):
            e   = min(s + args.batch_size, N)
            idx = np.arange(s, e)
            B_  = e - s

            batch  = _to_device(_make_eval_batch(data, idx, M), device)
            Z, rl, cs, cr = model(**batch)
            Z  = Z.float()        # [B_, R, d_model]
            rl = rl.float()       # [B_, R]
            cs = cs.float()       # [B_, M]
            cr = cr.cpu().numpy().astype(np.int32)  # [B_, M]

            gold_b    = gold[idx]; base_b = topk[idx, 0]
            lgt_b     = lgt[idx]
            gold_t    = torch.from_numpy(gold_b.astype(np.int64)).to(device)
            base_t    = torch.from_numpy(base_b.astype(np.int64)).to(device)
            topk_b    = topk[idx]
            gip_b     = gip[idx]; bktA_b = bucketA[idx]

            # U_proj for gold and base tokens
            U_gold = model._compute_U_proj(gold_t.unsqueeze(1)).squeeze(1).float().cpu().numpy()  # [B_, d_model]
            U_base = model._compute_U_proj(base_t.unsqueeze(1)).squeeze(1).float().cpu().numpy()

            Z_np  = Z.cpu().numpy()
            rl_np = rl.cpu().numpy()
            cs_np = cs.cpu().numpy()

            ar = np.arange(B_)

            # Variant-map regions for gold and base
            var_gold_reg = tok_arr_variant[np.clip(gold_b,  0, Vv-1)].astype(np.int64)
            var_base_reg = tok_arr_variant[np.clip(base_b,  0, Vv-1)].astype(np.int64)
            var_gold_reg = np.clip(var_gold_reg, 0, R-1)
            var_base_reg = np.clip(var_base_reg, 0, R-1)

            rgr_b = real_gold_reg[idx]
            rbr_b = real_base_reg[idx]
            rgr_b_clamped = np.clip(rgr_b, 0, R-1)

            # ── Pair metrics ────────────────────────────────────────────────

            # Old split-sensitive
            z_gv = Z_np[ar, var_gold_reg]   # [B_, d_model] — Z at gold's variant slot
            z_bv = Z_np[ar, var_base_reg]   # [B_, d_model] — Z at base's variant slot
            sg_old = (U_gold * z_gv).sum(-1) / sqrt_d
            sb_old = (U_base * z_bv).sum(-1) / sqrt_d
            # Forced same-anchor: both through gold's variant slot
            sb_forced = (U_base * z_gv).sum(-1) / sqrt_d
            # Forced real anchor: both through real gold region slot
            z_rv = Z_np[ar, rgr_b_clamped]
            sg_real = (U_gold * z_rv).sum(-1) / sqrt_d
            sb_real = (U_base * z_rv).sum(-1) / sqrt_d

            # Base logit pair (raw logit comparison)
            gold_idx_b = np.where(gip_b,
                                  (topk_b == gold_b[:, None]).argmax(axis=1), 0)
            base_lgt_gold = lgt_b[ar, np.clip(gold_idx_b, 0, M-1)]
            base_lgt_base = lgt_b[:, 0]

            # Own-region candidate scoring (split-sensitive): each cand through its OWN variant region
            topk_t = torch.from_numpy(topk_b.astype(np.int64)).to(device)
            U_cand = model._compute_U_proj(topk_t).float().cpu().numpy()  # [B_, M, d_model]
            own_scores = np.zeros((B_, M), np.float32)
            for c_idx in range(M):
                z_slot = Z_np[ar, cr[:B_, c_idx]]      # [B_, d_model]
                own_scores[:, c_idx] = (U_cand[:, c_idx, :] * z_slot).sum(-1) / sqrt_d

            # Store in per-row arrays
            valid_rgr = (rgr_b >= 0) & gip_b
            old_pair[s:e]          = np.where(bktA_b, sg_old > sb_old, np.nan)
            forced_pair[s:e]       = np.where(bktA_b, sg_old > sb_forced, np.nan)
            forced_real_pair[s:e]  = np.where(bktA_b & valid_rgr, sg_real > sb_real, np.nan)
            old_margin_arr[s:e]    = np.where(bktA_b, sg_old - sb_old, np.nan)
            forced_margin_arr[s:e] = np.where(bktA_b, sg_old - sb_forced, np.nan)
            forced_real_margin[s:e]= np.where(bktA_b & valid_rgr, sg_real - sb_real, np.nan)
            base_pair_arr[s:e]     = np.where(bktA_b, base_lgt_gold > base_lgt_base, np.nan)

            # Own-region candidate accuracy + NLL (gip rows)
            if gip_b.any():
                own_pred = own_scores.argmax(axis=1)
                own_pred_id = topk_b[ar, own_pred]
                own_cand_acc_arr[s:e] = np.where(gip_b, (own_pred_id == gold_b).astype(float), np.nan)
                for bi_ in range(B_):
                    if not gip_b[bi_]: continue
                    gi_ = gold_idx_b[bi_]
                    sc_ = own_scores[bi_].astype(np.float64)
                    sc_ = sc_ - sc_.max()
                    ls_ = sc_ - np.log(np.exp(sc_).sum() + _EPS)
                    own_cand_nll_arr[s + bi_] = float(-ls_[gi_])

            # ── Region prediction ───────────────────────────────────────────
            region_pred_arr[s:e] = rl_np.argmax(axis=1)
            top16 = np.argsort(rl_np, axis=1)[:, -16:][:, ::-1]
            r_topk_arr[s:e] = top16

            # ── Within-real-region ranking (per-row loop) ───────────────────
            for bi in range(B_):
                row_i = s + bi
                if not gip_b[bi]: continue
                rgr_i = rgr_b[bi]
                if rgr_i < 0: continue

                # Candidates in real gold region
                cand_real_regs = tok_arr_real[np.clip(topk_b[bi], 0, Vr-1)].astype(np.int64)
                in_real = (cand_real_regs == rgr_i)
                idxs_r = np.where(in_real)[0]
                if len(idxs_r) < 2: continue

                # Gold full index
                gf_matches = np.where(topk_b[bi] == gold_b[bi])[0]
                if len(gf_matches) == 0: continue
                gf = gf_matches[0]
                if not in_real[gf]: continue
                gw_matches = np.where(idxs_r == gf)[0]
                if len(gw_matches) == 0: continue
                gw = gw_matches[0]

                # Score candidates in real-region through gold's variant anchor
                anch = var_gold_reg[bi]
                z_anch = Z_np[bi, anch]     # [d_model]
                U_r = U_cand[bi, idxs_r]    # [n_r, d_model]
                scores_r = (U_r * z_anch).sum(-1) / sqrt_d

                sr_ = scores_r.astype(np.float64); sr_ = sr_ - sr_.max()
                log_sum_ = np.log(np.exp(sr_).sum() + _EPS)
                nll_r = float(-(sr_[gw] - log_sum_))
                acc_r = float(scores_r.argmax() == gw)
                rank_r = int((scores_r > scores_r[gw]).sum())

                within_real_acc[row_i]  = acc_r
                within_real_nll[row_i]  = float(nll_r)
                within_real_rank[row_i] = rank_r

                # Base logit baseline on same subset
                lgt_r = lgt_b[bi, idxs_r].astype(np.float64)
                lgt_r = lgt_r - lgt_r.max()
                lgt_log_sum = np.log(np.exp(lgt_r).sum() + _EPS)
                b_nll_r = float(-(lgt_r[gw] - lgt_log_sum))
                b_acc_r = float(lgt_r.argmax() == gw)
                b_rank_r = int((lgt_r > lgt_r[gw]).sum())
                base_wr_acc[row_i]  = b_acc_r
                base_wr_nll[row_i]  = b_nll_r
                base_wr_rank[row_i] = b_rank_r

            # ── Collect examples (per batch) ─────────────────────────────────
            rgrs = real_gold_reg[idx]; rbrs = real_base_reg[idx]
            for bi in range(B_):
                if not bktA_b[bi]: continue
                row_i = s + bi
                forced_win = bool(sg_old[bi] > sb_forced[bi])
                base_win   = bool(base_lgt_gold[bi] > base_lgt_base[bi])
                ex_row = {
                    "row_id": int(row_i), "gold": int(gold_b[bi]),
                    "base": int(base_b[bi]),
                    "real_gold_reg": int(rgrs[bi]), "real_base_reg": int(rbrs[bi]),
                    "var_gold_reg": int(var_gold_reg[bi]),
                    "var_base_reg": int(var_base_reg[bi]),
                    "sg_old": float(sg_old[bi]), "sb_old": float(sb_old[bi]),
                    "sg_forced": float(sg_old[bi]), "sb_forced": float(sb_forced[bi]),
                    "base_lgt_gold": float(base_lgt_gold[bi]),
                    "base_lgt_base": float(base_lgt_base[bi]),
                    "top10_cand": topk_b[bi, :10].tolist(),
                    "top10_lgt": lgt_b[bi, :10].tolist(),
                }
                if len(ex_real_beats_shuf) < _N_EXAMPLES:
                    if forced_win and not base_win:
                        ex_real_beats_shuf.append(ex_row)
                if len(ex_shuf_cheat) < _N_EXAMPLES:
                    if (sg_old[bi] > sb_old[bi]) and not forced_win:
                        ex_shuf_cheat.append(ex_row)
                if len(ex_real_fails) < _N_EXAMPLES:
                    if not forced_win:
                        ex_real_fails.append(ex_row)

    # ── Aggregate metrics ────────────────────────────────────────────────────

    # Region prediction under variant's map (internal)
    var_gold_reg_all = tok_arr_variant[np.clip(gold, 0, Vv-1)].astype(np.int64)
    var_gold_reg_all[var_gold_reg_all >= n_regions_variant] = -1
    valid_vgr = (var_gold_reg_all >= 0)
    def _recall_k(k):
        if valid_vgr.sum() == 0: return float("nan")
        top_k = np.argsort(
            np.zeros((valid_vgr.sum(), R), np.float32), axis=1)  # placeholder
        # Use r_topk_arr
        hits = (r_topk_arr[valid_vgr, :k] == var_gold_reg_all[valid_vgr, None]).any(axis=1)
        return float(hits.mean())
    # Correct recall computation using stored top16
    def _recall(k):
        if valid_vgr.sum() == 0: return float("nan")
        kk = min(k, 16)
        hits = (r_topk_arr[valid_vgr, :kk] == var_gold_reg_all[valid_vgr, None]).any(axis=1)
        return float(hits.mean())

    # Real-region diagnostic (always vs real gold region)
    valid_rgr_all = (real_gold_reg >= 0)
    def _recall_real(k):
        if valid_rgr_all.sum() == 0: return float("nan")
        kk = min(k, 16)
        hits = (r_topk_arr[valid_rgr_all, :kk] == real_gold_reg[valid_rgr_all, None]).any(axis=1)
        return float(hits.mean())

    # Slice masks
    slices = {
        "bucketA":                     bucketA,
        "real_same_region_confuser":   bucketA & same_reg_real,
        "real_same_superregion_conf":  bucketA & same_sreg_real,
        "real_different_region_conf":  bucketA & ~same_reg_real & gip,
        "real_same_reg_2plus_cands":   gip & (
            np.array([((tok_arr_real[np.clip(topk[n], 0, Vr-1)] == real_gold_reg[n]).sum() >= 2
                       if real_gold_reg[n] >= 0 else False)
                      for n in range(N)])),
        "gold_rank_1_5":   gip & (gold_rank_a >= 0) & (gold_rank_a < 5),
        "gold_rank_6_32":  gip & (gold_rank_a >= 5)  & (gold_rank_a < 32),
        "gold_rank_33_M":  gip & (gold_rank_a >= 32),
    }

    def _slice_metrics(mask):
        n = int(mask.sum()); m = mask
        if n == 0:
            return {"n": 0, "old_pair_acc": float("nan"),
                    "forced_pair_acc": float("nan"), "forced_real_pair_acc": float("nan"),
                    "forced_margin_mean": float("nan"), "forced_margin_median": float("nan"),
                    "within_real_acc": float("nan"), "within_real_nll": float("nan"),
                    "within_real_rank_mean": float("nan"),
                    "base_wr_acc": float("nan"), "base_wr_nll": float("nan")}
        return {
            "n": n,
            "old_pair_acc":          _nanmean(old_pair[m]),
            "forced_pair_acc":       _nanmean(forced_pair[m]),
            "forced_real_pair_acc":  _nanmean(forced_real_pair[m]),
            "forced_margin_mean":    _nanmean(forced_margin_arr[m]),
            "forced_margin_median":  float(np.nanmedian(forced_margin_arr[m])) if m.sum() > 0 else float("nan"),
            "within_real_acc":       _nanmean(within_real_acc[m]),
            "within_real_nll":       _nanmean(within_real_nll[m]),
            "within_real_rank_mean": _nanmean(within_real_rank[m]),
            "base_wr_acc":           _nanmean(base_wr_acc[m]),
            "base_wr_nll":           _nanmean(base_wr_nll[m]),
        }

    slice_results = {sn: _slice_metrics(mk) for sn, mk in slices.items()}

    # Top-level aggregates
    wr_mask      = ~np.isnan(within_real_acc)
    wr_both_mask = wr_mask & ~np.isnan(base_wr_acc)
    agg = {
        "n_val":                       N,
        "n_bucketA":                   int(bucketA.sum()),
        "n_real_same_region_confuser": int((bucketA & same_reg_real).sum()),
        "n_real_same_sreg_confuser":   int((bucketA & same_sreg_real).sum()),
        "n_within_real_region_rows":   int(wr_mask.sum()),
        # Region prediction
        "region_recall@8_internal":         _recall(8),
        "region_recall@8_real_diagnostic":  _recall_real(8),
        "region_recall@1_internal":         _recall(1),
        "region_recall@4_internal":         _recall(4),
        "region_recall@16_internal":        _recall(16),
        # Pair metrics (bucketA)
        "old_split_pair_acc_bucketA":        _nanmean(old_pair[bucketA]),
        "forced_anchor_pair_acc_bucketA":    _nanmean(forced_pair[bucketA]),
        "forced_real_anchor_pair_acc_bucketA": _nanmean(forced_real_pair[bucketA]),
        # Same-region confuser
        "old_split_pair_acc_same_reg":       _nanmean(old_pair[bucketA & same_reg_real]),
        "forced_anchor_pair_acc_same_reg":   _nanmean(forced_pair[bucketA & same_reg_real]),
        "forced_real_anchor_pair_acc_same_reg": _nanmean(forced_real_pair[bucketA & same_reg_real]),
        "forced_anchor_margin_mean_same_reg":   _nanmean(forced_margin_arr[bucketA & same_reg_real]),
        "forced_anchor_margin_median_same_reg": (float(np.nanmedian(forced_margin_arr[bucketA & same_reg_real]))
                                                  if (bucketA & same_reg_real).sum() > 0 else float("nan")),
        # Same-superregion confuser
        "old_split_pair_acc_same_sreg":      _nanmean(old_pair[bucketA & same_sreg_real]),
        "forced_anchor_pair_acc_same_sreg":  _nanmean(forced_pair[bucketA & same_sreg_real]),
        "forced_real_anchor_pair_acc_same_sreg": _nanmean(forced_real_pair[bucketA & same_sreg_real]),
        # Within-real-region ranking
        "within_real_region_anchor_acc":     _nanmean(within_real_acc),
        "within_real_region_anchor_nll":     _nanmean(within_real_nll),
        "within_real_region_rank_mean":      _nanmean(within_real_rank),
        "within_real_region_rank_median":    (float(np.nanmedian(within_real_rank[wr_mask]))
                                               if wr_mask.sum() > 0 else float("nan")),
        "base_within_real_region_acc":       _nanmean(base_wr_acc),
        "base_within_real_region_nll":       _nanmean(base_wr_nll),
        "within_real_acc_gain_vs_base":      (_nanmean(within_real_acc[wr_both_mask] - base_wr_acc[wr_both_mask])
                                              if wr_both_mask.sum() > 0 else float("nan")),
        "within_real_nll_gain_vs_base":      (_nanmean(base_wr_nll[wr_both_mask] - within_real_nll[wr_both_mask])
                                              if wr_both_mask.sum() > 0 else float("nan")),
        # Own-region split-sensitive candidate metric
        "split_sensitive_own_region_cand_acc": _nanmean(own_cand_acc_arr[gip]),
        "split_sensitive_own_region_cand_nll": _nanmean(own_cand_nll_arr[gip]),
        # Base pair
        "base_logit_pair_acc_bucketA":       _nanmean(base_pair_arr[bucketA]),
    }

    return agg, slice_results, {
        "ex_real_beats_shuf": ex_real_beats_shuf,
        "ex_shuf_cheat":      ex_shuf_cheat,
        "ex_real_fails":      ex_real_fails,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Example reports
# ═══════════════════════════════════════════════════════════════════════════════

def write_example_report(path, title, examples, headers):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if not examples:
            f.write("No examples found.\n")
            return
        for i, ex in enumerate(examples[:_N_EXAMPLES]):
            f.write(f"## Example {i+1}\n\n")
            for k, v in ex.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")
    print(f"[save] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main report
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(args, results, out_dir):
    variants = list(results.keys())

    def _v(variant, k): return results[variant]["agg"].get(k, float("nan"))

    real   = results.get("contextual_region_meaning_real",    {}).get("agg", {})
    static = results.get("static_region_meaning",             {}).get("agg", {})
    shuf   = results.get("contextual_region_meaning_shuffled",{}).get("agg", {})
    rand   = results.get("contextual_region_meaning_random",  {}).get("agg", {})

    def _nv(d, k): return d.get(k, float("nan"))

    # Case analysis
    real_forced   = _nv(real,   "forced_anchor_pair_acc_same_reg")
    shuf_forced   = _nv(shuf,   "forced_anchor_pair_acc_same_reg")
    rand_forced   = _nv(rand,   "forced_anchor_pair_acc_same_reg")
    static_forced = _nv(static, "forced_anchor_pair_acc_same_reg")

    real_wr    = _nv(real,   "within_real_region_anchor_acc")
    shuf_wr    = _nv(shuf,   "within_real_region_anchor_acc")
    rand_wr    = _nv(rand,   "within_real_region_anchor_acc")
    static_wr  = _nv(static, "within_real_region_anchor_acc")

    real_old   = _nv(real,  "old_split_pair_acc_same_reg")
    shuf_old   = _nv(shuf,  "old_split_pair_acc_same_reg")
    rand_old   = _nv(rand,  "old_split_pair_acc_same_reg")

    def _ok(a, b): return a == a and b == b and a > b + 0.005

    cheating_detected     = _ok(shuf_old, shuf_forced) or _ok(rand_old, rand_forced)
    real_beats_shuf_forced = _ok(real_forced, shuf_forced) or _ok(real_forced, rand_forced)
    real_beats_shuf_wr     = _ok(real_wr,    shuf_wr)   or _ok(real_wr,    rand_wr)
    real_beats_static      = _ok(real_forced, static_forced)
    wr_nontrivial          = real_wr == real_wr and real_wr > 0.1
    wr_gain                = _nv(real, "within_real_acc_gain_vs_base")
    wr_gain_pos            = wr_gain == wr_gain and wr_gain > 0.0

    if real_beats_shuf_forced and real_beats_shuf_wr and wr_nontrivial:
        verdict = "PROCEED_TO_PHASE_2B"
    elif real_beats_shuf_forced and cheating_detected:
        verdict = "PARTIAL_GO_BUT_IMPROVE_REGION_MEANING"
    elif real_beats_static and not real_beats_shuf_forced:
        verdict = "PARTIAL_GO_BUT_IMPROVE_REGION_MEANING"
    else:
        verdict = "DO_NOT_PROCEED_TO_PHASE_2B"

    rpt = os.path.join(out_dir, "phase2A_corrected_confuser_report.md")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("# Phase 2A.1: Corrected Region-Meaning Confuser Diagnostic\n\n")
        f.write(f"**selected_M:** {args.selected_M}  |  "
                f"**phase2a_dir:** {args.phase2a_dir}\n\n---\n\n")

        f.write("## Corrected Comparison Table\n\n")
        f.write("Primary corrected metric: `forced_anchor_pair_acc_same_reg` "
                "(gold and base both scored through gold's region anchor)\n\n")
        cols = ["variant",
                "old_split_pair_acc_same_reg", "forced_anchor_pair_acc_same_reg",
                "forced_real_anchor_pair_acc_same_reg",
                "within_real_region_anchor_acc", "within_real_acc_gain_vs_base",
                "region_recall@8_internal"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for v in variants:
            agg = results[v]["agg"]
            f.write("| " + " | ".join(
                v if c == "variant" else _fmt(agg.get(c, float("nan")))
                for c in cols) + " |\n")

        f.write("\n---\n\n## Q&A\n\n")

        def _q(n, q, ans, detail=""):
            f.write(f"### Q{n}: {q}\n\n**{ans}**\n\n{detail}\n\n")

        _q(1, "Did old split-sensitive scoring allow shuffled/random to cheat?",
           "YES" if cheating_detected else "UNCLEAR",
           f"shuf old={_fmt(shuf_old)} vs forced={_fmt(shuf_forced)}  "
           f"rand old={_fmt(rand_old)} vs forced={_fmt(rand_forced)}")
        _q(2, "Does real beat shuffled/random under forced same-anchor pair scoring?",
           "YES" if real_beats_shuf_forced else "NO",
           f"real={_fmt(real_forced)}  shuf={_fmt(shuf_forced)}  rand={_fmt(rand_forced)}")
        _q(3, "Does real beat shuffled/random on within-real-region ranking?",
           "YES" if real_beats_shuf_wr else "NO",
           f"real={_fmt(real_wr)}  shuf={_fmt(shuf_wr)}  rand={_fmt(rand_wr)}")
        _q(4, "Does contextual real beat static under corrected metrics?",
           "YES" if real_beats_static else "NO",
           f"real forced={_fmt(real_forced)}  static forced={_fmt(static_forced)}  "
           f"real wr={_fmt(real_wr)}  static wr={_fmt(static_wr)}")
        _q(5, "Are learned real region states identity-resolving or only coarse?",
           "IDENTITY_RESOLVING" if (real_beats_shuf_forced and wr_nontrivial) else
           "COARSE_ONLY" if (not wr_nontrivial) else "PARTIAL",
           f"within_real_acc={_fmt(real_wr)}  wr_gain_vs_base={_fmt(wr_gain)}")
        _q(6, "Should Phase 2A verdict be revised?",
           "YES — UPGRADE TO PARTIAL GO" if cheating_detected and real_beats_shuf_forced else
           "NO — original verdict stands",
           "Shuffled/random confuser advantage was a split-region scoring artifact." if cheating_detected else "")
        _q(7, "Should we proceed to Phase 2B?",
           verdict,
           "" if verdict != "DO_NOT_PROCEED_TO_PHASE_2B" else
           "Region meanings distinguish coarse regions but cannot resolve token identity within a region.")

        f.write("---\n\n## PHASE 2A.1 VERDICT\n\n```\n")
        for v in variants:
            agg = results[v]["agg"]
            f.write(f"{v:42s} "
                    f"old={_fmt(agg.get('old_split_pair_acc_same_reg',float('nan')))}  "
                    f"forced={_fmt(agg.get('forced_anchor_pair_acc_same_reg',float('nan')))}  "
                    f"wr={_fmt(agg.get('within_real_region_anchor_acc',float('nan')))}  "
                    f"wr_gain={_fmt(agg.get('within_real_acc_gain_vs_base',float('nan')))}\n")
        f.write(f"\nrecommendation: {verdict}\n```\n")

    print(f"[save] {rpt}")
    return rpt, verdict


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 2A.1: Corrected Confuser Diagnostic")
    p.add_argument("--phase2a_dir",   required=True)
    p.add_argument("--val_dir",       required=True)
    p.add_argument("--small_ckpt",    required=True)
    p.add_argument("--token_to_region", required=True)
    p.add_argument("--super_map",     default=None)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--selected_M",    type=int,   default=64)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--max_val_rows",  type=int,   default=None)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--amp",           action="store_true")
    args = p.parse_args()

    import random, numpy as np, torch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.isdir(args.phase2a_dir):
        raise FileNotFoundError(f"phase2a_dir not found: {args.phase2a_dir}")
    if not os.path.isfile(args.small_ckpt):
        raise FileNotFoundError(f"small_ckpt not found: {args.small_ckpt}")
    if not os.path.isfile(args.token_to_region):
        raise FileNotFoundError(f"token_to_region not found: {args.token_to_region}")

    shuffled_map_path = os.path.join(args.phase2a_dir, "shuffled_token_to_region.json")
    random_map_path   = os.path.join(args.phase2a_dir, "random_token_to_region.json")
    for p_ in [shuffled_map_path, random_map_path]:
        if not os.path.isfile(p_): raise FileNotFoundError(f"Map not found: {p_}")

    # Checkpoint paths (fail loudly if missing)
    ckpt_map = {
        "static_region_meaning":             os.path.join(args.phase2a_dir, "best_static_region_meaning.pt"),
        "contextual_region_meaning_real":    os.path.join(args.phase2a_dir, "best_contextual_region_meaning_real.pt"),
        "contextual_region_meaning_shuffled":os.path.join(args.phase2a_dir, "best_contextual_region_meaning_shuffled.pt"),
        "contextual_region_meaning_random":  os.path.join(args.phase2a_dir, "best_contextual_region_meaning_random.pt"),
    }
    for v, cp in ckpt_map.items():
        if not os.path.isfile(cp):
            raise FileNotFoundError(f"Checkpoint missing for {v}: {cp}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[step 1] loading val shards...")
    val_data = load_shards(args.val_dir, args.selected_M, args.max_val_rows, "val")
    M = args.selected_M

    # ── Unembedding ───────────────────────────────────────────────────────────
    print("\n[step 2] loading unembedding...")
    U, uinfo = load_unembedding(args.small_ckpt)
    with open(os.path.join(args.output_dir, "unembedding_audit.json"), "w") as f:
        json.dump(uinfo, f, indent=2)

    # ── Region maps ───────────────────────────────────────────────────────────
    print("\n[step 3] loading region maps...")
    tok_arr_real, reg_arr_real, n_regions_real, n_super_real = load_region_maps(
        args.token_to_region, args.super_map)

    def _load_json_map(path, base_arr, n_regions):
        d = json.load(open(path))
        out = base_arr.copy()
        for k, v in d.items():
            i = int(k)
            if 0 <= i < len(out): out[i] = int(v)
        return out

    tok_arr_shuf = _load_json_map(shuffled_map_path, tok_arr_real, n_regions_real)
    tok_arr_rand = _load_json_map(random_map_path,   tok_arr_real, n_regions_real)
    print(f"  real n_regions={n_regions_real}  n_super={n_super_real}")

    variant_maps = {
        "static_region_meaning":              tok_arr_real,
        "contextual_region_meaning_real":     tok_arr_real,
        "contextual_region_meaning_shuffled": tok_arr_shuf,
        "contextual_region_meaning_random":   tok_arr_rand,
    }

    # ── Router baseline ───────────────────────────────────────────────────────
    print("\n[router baseline]")
    router_vm = eval_router_baseline(val_data, tok_arr_real, n_regions_real)
    if router_vm["router_available"]:
        print(f"  router recall@8={router_vm['gold_region_recall@8']:.4f}")
    else:
        print("  router: unavailable (no router_topk_reg in shards)")

    # ── Evaluate each variant ─────────────────────────────────────────────────
    all_results = {}
    for variant, ckpt_path in ckpt_map.items():
        print(f"\n{'='*60}\n[variant] {variant}\n{'='*60}")
        tok_arr_v = variant_maps[variant]
        model = load_variant_model(
            variant, ckpt_path, args.phase2a_dir,
            tok_arr_v, reg_arr_real, U,
            n_regions_real, n_super_real, device)

        agg, slices, examples = eval_variant(
            variant, model,
            val_data, M, n_regions_real,
            tok_arr_real, reg_arr_real, n_regions_real,
            tok_arr_v, n_regions_real,
            args, device)

        all_results[variant] = {"agg": agg, "slices": slices, "examples": examples}
        print(f"  old_pair_same_reg     = {_fmt(agg.get('old_split_pair_acc_same_reg'))}")
        print(f"  forced_pair_same_reg  = {_fmt(agg.get('forced_anchor_pair_acc_same_reg'))}")
        print(f"  within_real_acc       = {_fmt(agg.get('within_real_region_anchor_acc'))}")
        print(f"  wr_gain_vs_base       = {_fmt(agg.get('within_real_acc_gain_vs_base'))}")
        print(f"  region_recall@8       = {_fmt(agg.get('region_recall@8_internal'))}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Comparison CSV ────────────────────────────────────────────────────────
    comp_cols = [
        "variant","n_val","n_bucketA","n_real_same_region_confuser",
        "n_real_same_sreg_confuser","n_within_real_region_rows",
        "region_recall@8_internal","region_recall@8_real_diagnostic",
        "old_split_pair_acc_bucketA","forced_anchor_pair_acc_bucketA",
        "forced_real_anchor_pair_acc_bucketA",
        "old_split_pair_acc_same_reg","forced_anchor_pair_acc_same_reg",
        "forced_real_anchor_pair_acc_same_reg",
        "forced_anchor_margin_mean_same_reg","forced_anchor_margin_median_same_reg",
        "old_split_pair_acc_same_sreg","forced_anchor_pair_acc_same_sreg",
        "forced_real_anchor_pair_acc_same_sreg",
        "within_real_region_anchor_acc","within_real_region_anchor_nll",
        "within_real_region_rank_mean",
        "base_within_real_region_acc","base_within_real_region_nll",
        "within_real_acc_gain_vs_base","within_real_nll_gain_vs_base",
        "split_sensitive_own_region_cand_acc",
        "split_sensitive_own_region_cand_nll",
        "base_logit_pair_acc_bucketA",
    ]
    # Router baseline row — fill with nan for model-specific metrics
    router_row = {"variant": "router_baseline"}
    for c in comp_cols[1:]:
        router_row[c] = float("nan")
    if router_vm["router_available"]:
        router_row["region_recall@8_internal"] = router_vm["gold_region_recall@8"]
        router_row["region_recall@8_real_diagnostic"] = router_vm["gold_region_recall@8"]
    comp_rows = [router_row]
    for v, res in all_results.items():
        row = {"variant": v}
        for c in comp_cols[1:]:
            row[c] = res["agg"].get(c, float("nan"))
        comp_rows.append(row)
    _wcsv(os.path.join(args.output_dir, "corrected_confuser_comparison.csv"), comp_rows)

    # ── Slice metrics CSV ─────────────────────────────────────────────────────
    slice_rows = []
    for v, res in all_results.items():
        for sname, sm in res["slices"].items():
            slice_rows.append({"variant": v, "slice": sname, **sm})
    _wcsv(os.path.join(args.output_dir, "corrected_slice_metrics.csv"), slice_rows)

    # ── Example reports ───────────────────────────────────────────────────────
    shuf_res  = all_results.get("contextual_region_meaning_shuffled", {})
    real_res  = all_results.get("contextual_region_meaning_real",     {})

    write_example_report(
        os.path.join(args.output_dir, "examples_real_beats_shuffled_same_anchor.md"),
        "Examples: Real Model Beats Base-Logit Under Forced Same-Anchor (forced=True, base_pair=False)",
        real_res.get("examples", {}).get("ex_real_beats_shuf", []),
        [])
    write_example_report(
        os.path.join(args.output_dir, "examples_shuffled_old_metric_cheat.md"),
        "Examples: Shuffled/Random Old-Metric Cheating (old=True, forced=False)",
        shuf_res.get("examples", {}).get("ex_shuf_cheat", []),
        [])
    write_example_report(
        os.path.join(args.output_dir, "examples_real_fails_same_region.md"),
        "Examples: Real Model Fails Same-Anchor Pair (forced=False)",
        real_res.get("examples", {}).get("ex_real_fails", []),
        [])

    # ── Report ────────────────────────────────────────────────────────────────
    rpt, verdict = write_report(args, all_results, args.output_dir)

    # ── Final console output ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(" PHASE 2A.1 CORRECTED CONFUSER VERDICT:")
    print(f"{'='*60}")
    for v in all_results:
        agg = all_results[v]["agg"]
        print(f"  {v:42s}")
        print(f"    old_split_pair_same_reg  = {_fmt(agg.get('old_split_pair_acc_same_reg'))}")
        print(f"    forced_anchor_same_reg   = {_fmt(agg.get('forced_anchor_pair_acc_same_reg'))}")
        print(f"    within_real_region_acc   = {_fmt(agg.get('within_real_region_anchor_acc'))}")
        print(f"    wr_acc_gain_vs_base      = {_fmt(agg.get('within_real_acc_gain_vs_base'))}")
    print(f"\n  recommendation: {verdict}")
    print(f"  report: {rpt}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
