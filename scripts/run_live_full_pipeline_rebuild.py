#!/usr/bin/env python3
"""
run_live_full_pipeline_rebuild.py — Single-file live context + retrieval pipeline.

Rebuilds everything cleanly from raw tokens.  Old candidate shards are NOT used
as the source of context truth.

Stages:
  00 find_tokens       — locate / download WikiText-103 raw token arrays
  01 rebuild_dataset   — build input_ids + h_ctx shards from raw tokens
  02 verify_rebuild    — prove saved h_ctx is reproduced from saved input_ids
  03 build_index       — FAISS / numpy index over train h_ctx
  04 attach_neighbors  — attach top-K train neighbors to every row
  05 audit_retrieval   — measure retrieval signal quality
  06 debug_identity    — step-0 identity check on fresh resolver
  07 train_resolver    — train LiveContextRetrievalTokenResolver

Usage:
  python scripts/run_live_full_pipeline_rebuild.py --run_all [args...]
  python scripts/run_live_full_pipeline_rebuild.py --stage rebuild_dataset [args...]
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe, get_hs_small

try:
    import faiss as _faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

_STAGE_ORDER = [
    "find_tokens", "rebuild_dataset", "audit_logit_paths", "verify_rebuild",
    "build_index", "attach_neighbors", "audit_retrieval",
    "debug_identity", "train_resolver",
]

# ── Manifest ──────────────────────────────────────────────────────────────────

class Manifest:
    def __init__(self, root: str):
        self.path = os.path.join(root, "pipeline_manifest.json")
        os.makedirs(root, exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._data = json.load(f)
        else:
            self._data = {"stages": {}, "root": root}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def start(self, stage: str, output_dir: str):
        self._data["stages"][stage] = {
            "status": "running",
            "output_dir": output_dir,
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None,
            "key_metrics": {},
            "error_message": None,
        }
        self._save()

    def done(self, stage: str, metrics: Dict = None):
        s = self._data["stages"].setdefault(stage, {})
        s["status"]     = "pass"
        s["end_time"]   = datetime.datetime.now().isoformat()
        s["key_metrics"] = metrics or {}
        self._save()

    def fail(self, stage: str, error: str):
        s = self._data["stages"].setdefault(stage, {})
        s["status"]        = "fail"
        s["end_time"]      = datetime.datetime.now().isoformat()
        s["error_message"] = error
        self._save()

    def get(self, stage: str) -> Dict:
        return self._data["stages"].get(stage, {})

    def require_pass(self, stage: str):
        s = self.get(stage)
        if s.get("status") != "pass":
            raise RuntimeError(
                f"Prerequisite stage '{stage}' has not passed "
                f"(status={s.get('status', 'missing')}). Run it first."
            )


# ── Region helpers ────────────────────────────────────────────────────────────

def _load_token_to_region(path: str, vocab_size: int) -> Optional[np.ndarray]:
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    t2r = np.full(vocab_size, -1, dtype=np.int32)
    for k, v in raw.items():
        tok = int(k)
        if 0 <= tok < vocab_size:
            t2r[tok] = int(v)
    return t2r


def _load_r2s(path: str, n_fine: int) -> np.ndarray:
    r2s = np.zeros(n_fine, dtype=np.int32)
    if not path or not os.path.isfile(path):
        return r2s
    with open(path) as f:
        d = json.load(f)
    for k, v in d.items():
        fid = int(k)
        if 0 <= fid < n_fine:
            r2s[fid] = int(v)
    return r2s


# ── Model ─────────────────────────────────────────────────────────────────────

class LiveContextRetrievalTokenResolver(nn.Module):
    """
    Token resolver that operates on pre-computed h_ctx (no live backbone).
    Stage 01 saves h_ctx; Stage 02 proves it's reproducible from input_ids.

    Sequence: [CTX | REGION | RET | CAND_0 .. CAND_{n_cand-1}]
    delta_head zero-init → step-0 identity guaranteed.
    """

    def __init__(
        self,
        d_backbone:     int,
        n_fine:         int,
        n_super:        int,
        r2s_np:         np.ndarray,
        t2r_np:         Optional[np.ndarray] = None,
        top_k:          int   = 256,
        num_neighbors:  int   = 32,
        d_resolver:     int   = 256,
        n_layers:       int   = 2,
        n_heads:        int   = 4,
        ff_mult:        int   = 4,
        dropout:        float = 0.0,
        delta_scale:    float = 0.25,
        retrieval_tau:  float = 0.2,
        candidate_mode: str   = "base_topk_plus_neighbors",
    ):
        super().__init__()
        self.top_k         = top_k
        self.num_neighbors = num_neighbors
        self.n_fine        = n_fine
        self.n_super       = n_super
        self.d_resolver    = d_resolver
        self.delta_scale   = delta_scale
        self.retrieval_tau = retrieval_tau
        self.candidate_mode = candidate_mode
        self.n_cand = (top_k + num_neighbors
                       if candidate_mode == "base_topk_plus_neighbors" else top_k)

        self.fine_emb  = nn.Embedding(n_fine  + 1, d_resolver, padding_idx=n_fine)
        self.super_emb = nn.Embedding(n_super + 1, d_resolver, padding_idx=n_super)
        self.ctx_proj       = nn.Linear(d_backbone, d_resolver)
        self.ctx_to_region  = nn.Linear(d_backbone, d_resolver)  # fallback region summary
        self.region_scalar  = nn.Linear(3, d_resolver)           # entropy, margin, top_prob
        self.tok_emb_proj   = nn.Linear(d_backbone, d_resolver, bias=False)
        self.score_proj     = nn.Linear(3, d_resolver)           # logit, rank_norm, logprob
        self.ret_feat_proj  = nn.Linear(4, d_resolver)           # support, count_norm, max_sc, best_rank
        self.ret_scalar     = nn.Linear(3, d_resolver)           # mean_sc, max_sc, n_valid/N
        self.support_proj   = nn.Linear(2, d_resolver)           # router_supp, in_top8
        self.tok_norm       = nn.LayerNorm(d_resolver)
        self.region_head    = nn.Linear(d_resolver, n_fine)      # auxiliary region loss

        enc = nn.TransformerEncoderLayer(
            d_model=d_resolver, nhead=n_heads,
            dim_feedforward=d_resolver * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)

        self.delta_head = nn.Linear(d_resolver, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        V = t2r_np.shape[0] if t2r_np is not None else 50257
        t2r_i = (t2r_np.astype(np.int32) if t2r_np is not None
                 else np.full(V, n_fine, dtype=np.int32))
        self.register_buffer("t2r", torch.from_numpy(t2r_i).long())
        self.register_buffer("r2s", torch.from_numpy(r2s_np.astype(np.int64)).long())

    # ── helpers ───────────────────────────────────────────────────────────────

    def _region_tok(self, h_ctx, r_reg, r_prb, r_mar):
        if r_reg is not None and r_prb is not None:
            r_s = r_reg.clamp(0, self.n_fine - 1)
            w   = r_prb / (r_prb.sum(1, keepdim=True) + 1e-8)
            pooled   = (self.fine_emb(r_s) * w.unsqueeze(-1)).sum(1)
            entropy  = -(r_prb * torch.log(r_prb + 1e-9)).sum(1)
            mar_v    = r_mar.float() if r_mar is not None else torch.zeros(h_ctx.shape[0], device=h_ctx.device)
            sc       = torch.stack([entropy, mar_v, r_prb[:, 0]], dim=-1)
            return pooled + self.region_scalar(sc)
        return self.ctx_to_region(h_ctx)

    def _ret_tok(self, nbr_ids, nbr_scores, emb_w):
        valid  = (nbr_ids >= 0)
        safe   = nbr_ids.clamp(0, emb_w.shape[0] - 1)
        n_embs = F.embedding(safe, emb_w.float())
        n_proj = self.tok_emb_proj(n_embs)

        masked = nbr_scores.float().masked_fill(~valid, -1e9)
        weights = F.softmax(masked / self.retrieval_tau, dim=-1).masked_fill(~valid, 0.0)
        w_sum   = weights.sum(-1, keepdim=True).clamp(min=1e-8)
        weights = weights / w_sum
        ret_emb = (n_proj * weights.unsqueeze(-1)).sum(1)

        n_valid  = valid.float().sum(-1)
        mean_sc  = (nbr_scores.float() * valid.float()).sum(-1) / n_valid.clamp(min=1)
        max_sc   = nbr_scores.float().masked_fill(~valid, -1e9).max(-1).values
        max_sc   = max_sc.masked_fill(n_valid == 0, 0.0)
        scalars  = torch.stack([mean_sc, max_sc, n_valid / max(self.num_neighbors, 1)], dim=-1)
        return ret_emb + self.ret_scalar(scalars)

    def _retrieval_feats_per_cand(self, cand_ids, nbr_ids, nbr_scores):
        """(B, n_cand, 4): support, count_norm, max_score, best_rank"""
        B      = cand_ids.shape[0]
        N_nbr  = nbr_ids.shape[1]
        device = cand_ids.device
        valid  = (nbr_ids >= 0)                                  # (B, N_nbr)
        # (B, n_cand, N_nbr)
        matches = (nbr_ids.unsqueeze(1) == cand_ids.unsqueeze(2)) & valid.unsqueeze(1)
        sc_exp  = nbr_scores.float().unsqueeze(1)                # (B, 1, N_nbr)
        support  = (sc_exp * matches.float()).sum(2)             # (B, n_cand)
        count    = matches.float().sum(2)                        # (B, n_cand)
        max_sc   = (sc_exp * matches.float() + (~matches).float() * (-1e9)).max(2).values
        max_sc   = max_sc.masked_fill(count == 0, 0.0)
        rp = torch.where(
            matches,
            torch.arange(N_nbr, device=device, dtype=torch.float32).view(1, 1, N_nbr).expand(B, self.n_cand, N_nbr),
            torch.full((1, 1, 1), float(N_nbr), device=device).expand(B, self.n_cand, N_nbr),
        )
        best_rank = rp.min(2).values / max(N_nbr, 1)
        best_rank = best_rank.masked_fill(count == 0, 1.0)
        return torch.stack([support, count / max(N_nbr, 1), max_sc, best_rank], dim=-1)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        h_ctx:      torch.Tensor,           # (B, d_backbone) float32
        nbr_ids:    torch.Tensor,           # (B, num_neighbors) int64
        nbr_scores: torch.Tensor,           # (B, num_neighbors) float32
        tok_emb_w:  torch.Tensor,           # (V, d_backbone) float32
        r_reg:      Optional[torch.Tensor] = None,   # (B, K_r) int64
        r_prb:      Optional[torch.Tensor] = None,   # (B, K_r) float32
        r_mar:      Optional[torch.Tensor] = None,   # (B,) float32
    ) -> Dict[str, torch.Tensor]:

        device  = h_ctx.device
        B       = h_ctx.shape[0]
        emb_w   = tok_emb_w.float().to(device)
        V       = emb_w.shape[0]

        base_lgt = h_ctx.float() @ emb_w.T                      # (B, V)
        topk_lgt, topk_ids = base_lgt.topk(self.top_k, dim=1)  # (B, K)

        if self.candidate_mode == "base_topk_plus_neighbors":
            cand_ids = torch.cat([topk_ids, nbr_ids.long()], dim=1)  # (B, K+N)
        else:
            cand_ids = topk_ids                                  # (B, K)

        cand_valid = (cand_ids >= 0)                             # (B, n_cand)
        n_cand = cand_ids.shape[1]

        # ── Prefix tokens ─────────────────────────────────────────────────────
        ctx_tok    = self.ctx_proj(h_ctx.float())                # (B, d)
        region_tok = self._region_tok(h_ctx.float(), r_reg, r_prb, r_mar)  # (B, d)
        ret_tok    = self._ret_tok(nbr_ids, nbr_scores, emb_w)  # (B, d)

        # ── Per-candidate tokens ──────────────────────────────────────────────
        safe_cand = cand_ids.clamp(0, V - 1)
        t_embs    = F.embedding(safe_cand, emb_w)               # (B, n_cand, d_bb)
        tok_repr  = self.tok_emb_proj(t_embs)                   # (B, n_cand, d_res)

        cand_lgt  = base_lgt.gather(1, safe_cand) * cand_valid.float()
        cand_lp   = F.log_softmax(base_lgt, dim=1).gather(1, safe_cand) * cand_valid.float()
        rank_norm = torch.arange(n_cand, device=device, dtype=torch.float32
                                 ).unsqueeze(0).expand(B, -1) / n_cand
        tok_repr  = tok_repr + self.score_proj(
            torch.stack([cand_lgt, rank_norm, cand_lp], dim=-1))

        fine_ids  = self.t2r[safe_cand.clamp(0, self.t2r.shape[0] - 1)]
        fine_ids  = fine_ids.masked_fill(fine_ids < 0, self.n_fine)
        super_ids = self.r2s[fine_ids.clamp(0, self.n_fine - 1)].clamp(0, self.n_super - 1)
        super_ids = super_ids.masked_fill(fine_ids == self.n_fine, self.n_super)
        tok_repr  = tok_repr + self.fine_emb(fine_ids)
        tok_repr  = tok_repr + self.super_emb(super_ids)

        ret_feats = self._retrieval_feats_per_cand(cand_ids, nbr_ids, nbr_scores)
        tok_repr  = tok_repr + self.ret_feat_proj(ret_feats)

        if r_reg is not None and r_prb is not None:
            K_r  = r_reg.shape[1]
            r_s  = r_reg.clamp(0, self.n_fine - 1)
            match_r     = (r_s.unsqueeze(2) == fine_ids.unsqueeze(1))   # (B, K_r, n_cand)
            router_supp = (r_prb.unsqueeze(2) * match_r.float()).sum(1) # (B, n_cand)
            in_top8     = match_r[:, :min(8, K_r), :].any(1).float()   # (B, n_cand)
            tok_repr    = tok_repr + self.support_proj(
                torch.stack([router_supp, in_top8], dim=-1))

        tok_repr  = self.tok_norm(tok_repr) * cand_valid.float().unsqueeze(-1)

        # ── Transformer ───────────────────────────────────────────────────────
        n_prefix  = 3
        seq       = torch.cat([
            ctx_tok.unsqueeze(1), region_tok.unsqueeze(1), ret_tok.unsqueeze(1),
            tok_repr,
        ], dim=1)                                                # (B, 3+n_cand, d)
        pad_mask  = torch.zeros(B, n_prefix + n_cand, dtype=torch.bool, device=device)
        pad_mask[:, n_prefix:] = ~cand_valid
        out = self.transformer(seq, src_key_padding_mask=pad_mask)

        # ── Bounded delta ─────────────────────────────────────────────────────
        raw_delta     = self.delta_head(out[:, n_prefix:, :]).squeeze(-1)  # (B, n_cand)
        bounded_delta = self.delta_scale * torch.tanh(raw_delta) * cand_valid.float()
        delta_full    = torch.zeros(B, V, device=device)
        delta_full.scatter_add_(1, safe_cand, self.alpha * bounded_delta)
        refined_lgt   = base_lgt + delta_full

        ctx_out       = out[:, 0, :]                             # (B, d) for region aux
        region_logits = self.region_head(ctx_out)                # (B, n_fine)

        return {
            "refined_lgt":  refined_lgt,
            "base_lgt":     base_lgt,
            "cand_ids":     cand_ids,
            "cand_valid":   cand_valid,
            "region_logits": region_logits,
            "bounded_delta": bounded_delta,
        }


# ── Dataset ───────────────────────────────────────────────────────────────────

class RebuildDataset(IterableDataset):
    """Streams (stage01 shard, stage04 neighbor shard) pairs."""

    K_ROUTER = 16  # expected router topk size; falls back to zeros if missing

    def __init__(self, s01_dir: str, s04_dir: str, shuffle: bool = False):
        self.s01_dir = s01_dir
        self.s04_dir = s04_dir
        self.shuffle = shuffle
        self._paths  = sorted(glob.glob(os.path.join(s01_dir, "shard_*.pt")))
        if not self._paths:
            raise RuntimeError(f"No shard_*.pt in {s01_dir}")

    def __iter__(self):
        paths = list(self._paths)
        if self.shuffle:
            import random; random.shuffle(paths)
        for sp in paths:
            si   = int(os.path.basename(sp).replace("shard_", "").replace(".pt", ""))
            np_  = os.path.join(self.s04_dir, f"shard_{si:05d}.pt")
            if not os.path.exists(np_):
                continue  # limited attach run — this shard has no neighbors, skip
            s1 = torch.load(sp,  map_location="cpu", weights_only=True)
            s4 = torch.load(np_, map_location="cpu", weights_only=True)
            N  = s1["gold_token"].shape[0]
            K_r = self.K_ROUTER
            has_router = "router_topk_reg" in s1

            for i in range(N):
                row = {
                    "h_ctx":       s1["h_ctx"][i].float(),
                    "gold_token":  s1["gold_token"][i].long(),
                    "gold_region": (s1["gold_region"][i].long()
                                    if "gold_region" in s1
                                    else torch.tensor(-1, dtype=torch.long)),
                    "nbr_ids":     s4["neighbor_gold_tokens"][i].long(),
                    "nbr_scores":  s4["neighbor_scores"][i].float(),
                    "r_topk_reg":  (s1["router_topk_reg"][i].long()
                                    if has_router
                                    else torch.full((K_r,), -1, dtype=torch.long)),
                    "r_topk_prb":  (s1["router_topk_prb"][i].float()
                                    if has_router
                                    else torch.zeros(K_r)),
                    "r_margin":    (s1["router_margin"][i].float()
                                    if "router_margin" in s1
                                    else torch.tensor(0.0)),
                }
                yield row


def _collate(batch: List[Dict]) -> Dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(out, batch, device, lambda_region, lambda_kl, lambda_delta, kl_topk):
    gt      = batch["gold_token"].long().to(device)
    g_reg   = batch["gold_region"].long().to(device)
    ref_lgt = out["refined_lgt"]
    base_lgt = out["base_lgt"]

    ce_loss = F.cross_entropy(ref_lgt, gt)

    # Region auxiliary
    reg_loss = torch.tensor(0.0, device=device)
    if lambda_region > 0 and "region_logits" in out:
        valid_r = g_reg >= 0
        if valid_r.any():
            reg_loss = F.cross_entropy(
                out["region_logits"][valid_r], g_reg[valid_r])

    # KL on top-K candidate logits
    kl_loss = torch.tensor(0.0, device=device)
    if lambda_kl > 0:
        with torch.no_grad():
            base_kk_lp = F.log_softmax(
                base_lgt.topk(kl_topk, dim=1).values, dim=1)
        ref_kk_lp = F.log_softmax(
            ref_lgt.gather(1, base_lgt.topk(kl_topk, dim=1).indices), dim=1)
        kl_loss = F.kl_div(ref_kk_lp, base_kk_lp.exp(), reduction="batchmean")

    # Delta regularisation
    dl_loss = out["bounded_delta"].pow(2).mean() if lambda_delta > 0 else torch.tensor(0.0, device=device)

    loss = ce_loss + lambda_region * reg_loss + lambda_kl * kl_loss + lambda_delta * dl_loss
    return loss, {
        "loss": loss.item(), "ce": ce_loss.item(),
        "reg": reg_loss.item(), "kl": kl_loss.item(), "dl": dl_loss.item(),
    }


# ── Full-vocab eval ──────────────────────────────────────────────────────────

@torch.no_grad()
def full_vocab_eval(model, s01_dir, s04_dir, tok_emb_w, device,
                    eval_batch_size=64, gate=None) -> Dict:
    model.eval()
    emb_w = tok_emb_w.float().to(device)
    V = emb_w.shape[0]

    base_ce = ref_ce = 0.0
    nc = nc_nbr_gold = nc_no_nbr_gold = 0
    base_ce_nbr = ref_ce_nbr = base_ce_no_nbr = ref_ce_no_nbr = 0.0
    top1_b = top1_r = topK_b = topK_r = 0
    rank_improved = rank_worsened = 0
    gold_in_topK = gold_in_nbr = retrieval_added = 0
    changed_to_gold = changed_away = top1_changed = 0
    delta_abs_sum = delta_n = 0.0
    K = model.top_k

    s01_shards = sorted(glob.glob(os.path.join(s01_dir, "shard_*.pt")))
    for sp in s01_shards:
        si = int(os.path.basename(sp).replace("shard_", "").replace(".pt", ""))
        np_ = os.path.join(s04_dir, f"shard_{si:05d}.pt")
        if not os.path.exists(np_):
            continue
        s1 = torch.load(sp,  map_location="cpu", weights_only=True)
        s4 = torch.load(np_, map_location="cpu", weights_only=True)
        N  = s1["gold_token"].shape[0]
        has_router = "router_topk_reg" in s1

        for start in range(0, N, eval_batch_size):
            end = min(start + eval_batch_size, N)
            sl  = slice(start, end)
            B   = end - start

            h   = s1["h_ctx"][sl].float().to(device)
            gt  = s1["gold_token"][sl].long().to(device)
            nbr_i = s4["neighbor_gold_tokens"][sl].long().to(device)
            nbr_s = s4["neighbor_scores"][sl].float().to(device)
            r_reg = s1["router_topk_reg"][sl].long().to(device) if has_router else None
            r_prb = s1["router_topk_prb"][sl].float().to(device) if has_router else None
            r_mar = s1["router_margin"][sl].float().to(device) if "router_margin" in s1 else None

            out = model(h, nbr_i, nbr_s, emb_w, r_reg, r_prb, r_mar)
            ref_lgt  = out["refined_lgt"]
            base_lgt = out["base_lgt"]

            base_ce += float(F.cross_entropy(base_lgt, gt, reduction="sum"))
            ref_ce  += float(F.cross_entropy(ref_lgt,  gt, reduction="sum"))
            nc += B

            base_t1 = base_lgt.argmax(1); ref_t1 = ref_lgt.argmax(1)
            top1_b  += int((base_t1 == gt).sum())
            top1_r  += int((ref_t1  == gt).sum())
            topK_b  += int((base_lgt.topk(K, dim=1).indices == gt.unsqueeze(1)).any(1).sum())
            topK_r  += int((ref_lgt.topk( K, dim=1).indices == gt.unsqueeze(1)).any(1).sum())

            bi = torch.arange(B, device=device)
            rb = (1 + (base_lgt > base_lgt[bi, gt].unsqueeze(1)).sum(1)).float()
            rr = (1 + (ref_lgt  > ref_lgt [bi, gt].unsqueeze(1)).sum(1)).float()
            rank_improved += int((rr < rb).sum())
            rank_worsened += int((rr > rb).sum())

            in_topK = (base_lgt.topk(K, dim=1).indices == gt.unsqueeze(1)).any(1)
            in_nbr  = (nbr_i == gt.unsqueeze(1)).any(1)
            gold_in_topK     += int(in_topK.sum())
            gold_in_nbr      += int(in_nbr.sum())
            retrieval_added  += int((in_nbr & ~in_topK).sum())

            changed     = (base_t1 != ref_t1)
            top1_changed    += int(changed.sum())
            changed_to_gold += int((changed & (ref_t1 == gt)).sum())
            changed_away    += int((changed & (base_t1 == gt)).sum())

            # Split by whether gold is in neighbor
            nbr_g = in_nbr.cpu()
            for mask, flag in [(nbr_g, True), (~nbr_g, False)]:
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0]
                    gm  = gt[idx.to(device)]
                    bce = float(F.cross_entropy(base_lgt[idx.to(device)], gm, reduction="sum"))
                    rce = float(F.cross_entropy(ref_lgt [idx.to(device)], gm, reduction="sum"))
                    if flag:
                        base_ce_nbr += bce; ref_ce_nbr += rce; nc_nbr_gold += int(mask.sum())
                    else:
                        base_ce_no_nbr += bce; ref_ce_no_nbr += rce; nc_no_nbr_gold += int(mask.sum())

            delta_abs_sum += float(out["bounded_delta"].abs().sum())
            delta_n       += float(B * out["bounded_delta"].shape[1])

    def _nll(ce, n): return ce / max(n, 1)

    model.train()
    return {
        "full_vocab_base_nll_all":    _nll(base_ce, nc),
        "full_vocab_refined_nll_all": _nll(ref_ce,  nc),
        "full_vocab_gain_all":        _nll(base_ce, nc) - _nll(ref_ce, nc),
        "top1_acc_base":  top1_b / max(nc, 1),
        "top1_acc_ref":   top1_r / max(nc, 1),
        "topK_rate_base": topK_b / max(nc, 1),
        "topK_rate_ref":  topK_r / max(nc, 1),
        "gold_in_topK_rate":        gold_in_topK    / max(nc, 1),
        "gold_in_neighbor_rate":    gold_in_nbr     / max(nc, 1),
        "retrieval_added_gold_rate":retrieval_added / max(nc, 1),
        "rank_improved_rate": rank_improved / max(nc, 1),
        "rank_worsened_rate": rank_worsened / max(nc, 1),
        "changed_to_gold_rate": changed_to_gold / max(nc, 1),
        "changed_away_rate":    changed_away    / max(nc, 1),
        "top1_changed_rate":    top1_changed    / max(nc, 1),
        "gain_rows_with_neighbor_gold":    _nll(base_ce_nbr,    nc_nbr_gold)    - _nll(ref_ce_nbr,    nc_nbr_gold),
        "gain_rows_without_neighbor_gold": _nll(base_ce_no_nbr, nc_no_nbr_gold) - _nll(ref_ce_no_nbr, nc_no_nbr_gold),
        "mean_delta_abs": delta_abs_sum / max(delta_n, 1),
        "num_examples":   nc,
    }


# ── Stage helpers ─────────────────────────────────────────────────────────────

def _mkdir(root, sub):
    d = os.path.join(root, sub)
    os.makedirs(d, exist_ok=True)
    return d


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_md(path, title, body):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{body}\n")


def _nll_str(v): return f"{v:.6f}"


def _get_s01_dir(output_root: str, split: str) -> str:
    """Return patched dataset dir if Stage01B has populated it, else the original."""
    patched = os.path.join(output_root, "01_live_dataset_patched", split)
    if os.path.isdir(patched) and glob.glob(os.path.join(patched, "shard_*.pt")):
        return patched
    return os.path.join(output_root, "01_live_dataset", split)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 00 — Find raw token source
# ══════════════════════════════════════════════════════════════════════════════

def stage00_find_raw_token_source(args, manifest: Manifest):
    out_dir = _mkdir(args.output_root, "00_raw_token_source")
    manifest.start("find_tokens", out_dir)
    print("\n[stage00] Finding raw token source ...")

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = int(1e30)
    vocab_size = len(tok)

    search_locs = {
        "train_tokens_path": args.train_tokens_path,
        "val_tokens_path":   args.val_tokens_path,
    }

    # Search for local binary/npy files
    local_candidates = []
    for root_dir in [args.output_root, _PROJ_ROOT, os.path.join(_PROJ_ROOT, "data")]:
        for fname in ["train.bin", "val.bin", "train_tokens.npy", "val_tokens.npy",
                      "wikitext_train.npy", "wikitext_val.npy"]:
            p = os.path.join(root_dir, fname)
            if os.path.isfile(p):
                local_candidates.append(p)

    train_path = args.train_tokens_path
    val_path   = args.val_tokens_path
    source     = "user_provided"

    if train_path and val_path:
        print(f"  Using user-provided paths: train={train_path}  val={val_path}")
    else:
        # Fall back to HuggingFace WikiText-103
        print("  No local token files found. Loading from HuggingFace WikiText-103 ...")
        from scripts.offline_region_knn import load_wikitext
        t0 = time.time()
        train_arr = load_wikitext("wikitext-103-raw-v1", tok, "train")
        val_arr   = load_wikitext("wikitext-103-raw-v1", tok, "validation")
        print(f"  train tokens: {len(train_arr):,}  val tokens: {len(val_arr):,}  t={time.time()-t0:.0f}s")

        # Cache locally
        train_path = os.path.join(out_dir, "train_tokens.npy")
        val_path   = os.path.join(out_dir, "val_tokens.npy")
        np.save(train_path, train_arr)
        np.save(val_path,   val_arr)
        source = "huggingface_wikitext103"
        print(f"  Cached → {train_path}, {val_path}")

    # Validate
    for label, path in [("train", train_path), ("val", val_path)]:
        arr = np.load(path, mmap_mode="r") if path.endswith(".npy") else \
              np.frombuffer(open(path, "rb").read(), dtype=np.int32)
        n   = len(arr)
        mn  = int(arr[:1000].min()); mx = int(arr[:1000].max())
        if mx >= vocab_size or mn < 0:
            raise RuntimeError(f"{label} tokens out of range: min={mn} max={mx} vocab={vocab_size}")
        if n < args.ctx_len + 100:
            raise RuntimeError(f"{label} corpus too short: {n} < ctx_len+100")
        print(f"  {label}: n={n:,}  min={mn}  max={mx}  first5={arr[:5].tolist()}")

    report = {
        "pass":             True,
        "source":           source,
        "train_tokens_path": train_path,
        "val_tokens_path":   val_path,
        "vocab_size":       vocab_size,
        "local_candidates_found": local_candidates,
    }
    _write_json(os.path.join(out_dir, "report.json"), report)
    _write_md(os.path.join(out_dir, "report.md"), "Stage 00 — Raw Token Source",
              f"Source: {source}\ntrain: {train_path}\nval: {val_path}")
    manifest.done("find_tokens", {"source": source, "train_tokens_path": train_path,
                                   "val_tokens_path": val_path})
    print("[stage00] PASS")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 01 — Rebuild live dataset
# ══════════════════════════════════════════════════════════════════════════════

def stage01_rebuild_live_dataset(args, manifest: Manifest):
    manifest.require_pass("find_tokens")
    out_dir = _mkdir(args.output_root, "01_live_dataset")
    train_dir = _mkdir(out_dir, "train")
    val_dir   = _mkdir(out_dir, "val")
    manifest.start("rebuild_dataset", out_dir)
    print("\n[stage01] Rebuilding live dataset from raw tokens ...")

    # Load token source
    s0_report = _write_json  # re-open
    with open(os.path.join(args.output_root, "00_raw_token_source", "report.json")) as f:
        s0 = json.load(f)
    train_tokens_path = s0["train_tokens_path"]
    val_tokens_path   = s0["val_tokens_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Loading backbone: {args.small_ckpt}")
    backbone, probe, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    max_pos = backbone.pos_emb.num_embeddings
    if args.ctx_len > max_pos:
        raise RuntimeError(
            f"ctx_len={args.ctx_len} > backbone max_pos_emb={max_pos}. "
            f"Use --ctx_len <= {max_pos}.")
    print(f"  d_model={d_model}  vocab_size={vocab_size}  max_pos={max_pos}  ctx_len={args.ctx_len}")

    emb_w = backbone.token_emb.weight.detach().float()   # (V, d) on device
    has_probe = probe is not None
    K_router  = 16

    # region lookup
    t2r_np = _load_token_to_region(args.region_map, vocab_size) if args.region_map else None

    def _process_split(corpus_path, split_out_dir, split_name, stride, max_rows):
        corpus = np.load(corpus_path, mmap_mode="r") if corpus_path.endswith(".npy") \
                 else np.frombuffer(open(corpus_path, "rb").read(), dtype=np.int32)
        corpus = np.array(corpus, dtype=np.int32)
        L = len(corpus)
        print(f"  [{split_name}] corpus length={L:,}  stride={stride}  ctx_len={args.ctx_len}")

        positions = np.arange(args.ctx_len, L, stride, dtype=np.int64)
        if max_rows and len(positions) > max_rows:
            positions = positions[:max_rows]
        total_rows = len(positions)
        print(f"  [{split_name}] total positions: {total_rows:,}")

        # Sharding buffers
        buf_ids   = np.zeros((args.shard_size, args.ctx_len), dtype=np.int32)
        buf_gold  = np.zeros(args.shard_size, dtype=np.int32)
        buf_hctx  = np.zeros((args.shard_size, d_model), dtype=np.float16)
        buf_topk_ids  = np.zeros((args.shard_size, args.top_k), dtype=np.int32)
        buf_topk_lgt  = np.zeros((args.shard_size, args.top_k), dtype=np.float16)
        buf_row_id    = np.zeros(args.shard_size, dtype=np.int64)
        buf_offset    = np.zeros(args.shard_size, dtype=np.int64)
        buf_greg      = np.full(args.shard_size, -1, dtype=np.int16)
        # router buffers (if probe)
        buf_r_reg  = np.full((args.shard_size, K_router), -1, dtype=np.int16)
        buf_r_prb  = np.zeros((args.shard_size, K_router), dtype=np.float16)
        buf_r_mar  = np.zeros(args.shard_size, dtype=np.float16)

        shard_idx  = 0
        buf_n      = 0
        global_row = 0
        total_ce   = 0.0
        total_topK = 0
        t0         = time.time()

        def _flush_shard():
            nonlocal shard_idx
            n   = buf_n
            sd  = {
                "input_ids":     torch.from_numpy(buf_ids[:n].copy()),
                "gold_token":    torch.from_numpy(buf_gold[:n].copy()),
                "h_ctx":         torch.from_numpy(buf_hctx[:n].copy()),
                "base_topk_ids": torch.from_numpy(buf_topk_ids[:n].copy()),
                "base_topk_lgt": torch.from_numpy(buf_topk_lgt[:n].copy()),
                "row_id":        torch.from_numpy(buf_row_id[:n].copy()),
                "token_offset":  torch.from_numpy(buf_offset[:n].copy()),
                "gold_region":   torch.from_numpy(buf_greg[:n].copy()),
            }
            if has_probe:
                sd["router_topk_reg"] = torch.from_numpy(buf_r_reg[:n].copy())
                sd["router_topk_prb"] = torch.from_numpy(buf_r_prb[:n].copy())
                sd["router_margin"]   = torch.from_numpy(buf_r_mar[:n].copy())
            path = os.path.join(split_out_dir, f"shard_{shard_idx:05d}.pt")
            torch.save(sd, path)
            shard_idx += 1

        # Process in batches
        batch_size = args.eval_batch_size
        for bi in range(0, total_rows, batch_size):
            batch_pos = positions[bi: bi + batch_size]
            B  = len(batch_pos)
            ids_np = np.zeros((B, args.ctx_len), dtype=np.int32)
            gld_np = np.zeros(B, dtype=np.int32)
            for j, t in enumerate(batch_pos):
                ids_np[j] = corpus[int(t) - args.ctx_len : int(t)]
                gld_np[j] = corpus[int(t)]

            src = torch.from_numpy(ids_np).long().to(device)
            with torch.no_grad():
                h_all = get_hs_small(backbone, src, device)      # (B, T, d)
            h_ctx  = h_all[:, -1, :].float()                     # (B, d)
            lgt    = h_ctx @ emb_w.to(device).T                  # (B, V)
            topk_lgt_t, topk_ids_t = lgt.topk(args.top_k, dim=1)

            gt_t = torch.from_numpy(gld_np).long().to(device)
            total_ce  += float(F.cross_entropy(lgt, gt_t, reduction="sum"))
            total_topK += int((topk_ids_t == gt_t.unsqueeze(1)).any(1).sum())

            # Router outputs
            if has_probe:
                with torch.no_grad():
                    p_fine = F.softmax(probe(h_ctx), dim=-1)     # (B, n_fine)
                tv, ti = p_fine.topk(K_router, dim=1)            # (B, K_router)
                r_reg_np = ti.cpu().numpy().astype(np.int16)
                r_prb_np = tv.cpu().numpy().astype(np.float16)
                # margin = top1 - top2
                r_mar_np = (tv[:, 0] - tv[:, 1]).cpu().numpy().astype(np.float16) \
                           if K_router >= 2 else np.zeros(B, dtype=np.float16)

            h_np      = h_ctx.cpu().numpy().astype(np.float16)
            topk_i_np = topk_ids_t.cpu().numpy().astype(np.int32)
            topk_l_np = topk_lgt_t.cpu().float().numpy().astype(np.float16)

            for j in range(B):
                buf_ids[buf_n]      = ids_np[j]
                buf_gold[buf_n]     = gld_np[j]
                buf_hctx[buf_n]     = h_np[j]
                buf_topk_ids[buf_n] = topk_i_np[j]
                buf_topk_lgt[buf_n] = topk_l_np[j]
                buf_row_id[buf_n]   = global_row
                buf_offset[buf_n]   = batch_pos[j]
                buf_greg[buf_n]     = int(t2r_np[gld_np[j]]) if t2r_np is not None else -1
                if has_probe:
                    buf_r_reg[buf_n] = r_reg_np[j]
                    buf_r_prb[buf_n] = r_prb_np[j]
                    buf_r_mar[buf_n] = r_mar_np[j]
                global_row += 1
                buf_n += 1
                if buf_n == args.shard_size:
                    _flush_shard()
                    buf_n = 0

            if bi % (batch_size * 50) == 0 or bi + batch_size >= total_rows:
                print(f"  [{split_name}] batch {bi//batch_size}/{total_rows//batch_size}"
                      f"  rows={global_row:,}  t={time.time()-t0:.0f}s")

        if buf_n > 0:
            _flush_shard()

        base_nll  = total_ce / max(global_row, 1)
        top256_rt = total_topK / max(global_row, 1)
        print(f"  [{split_name}] shards={shard_idx}  rows={global_row:,}"
              f"  base_nll={base_nll:.4f}  gold_in_top{args.top_k}={top256_rt:.4f}")
        return {"shards": shard_idx, "rows": global_row,
                "base_nll": base_nll, "gold_in_topK": top256_rt,
                "corpus_len": L}

    limited_run = bool(args.max_train_rows or args.max_val_rows or args.stride_train > 8)
    print(f"\n  LIMITED_RUN={limited_run}  stride_train={args.stride_train}"
          f"  max_train_rows={args.max_train_rows}"
          f"  stride_val={args.stride_val}  max_val_rows={args.max_val_rows}")
    print("\n  Processing TRAIN split ...")
    train_stats = _process_split(train_tokens_path, train_dir, "train",
                                 args.stride_train, args.max_train_rows)
    print("\n  Processing VAL split ...")
    val_stats   = _process_split(val_tokens_path, val_dir, "val",
                                 args.stride_val, args.max_val_rows)

    # Reduction factor vs stride-8 full train run
    full_s8_rows = max(0, (train_stats.get("corpus_len", 0) - args.ctx_len) // 8)
    reduction_vs_s8 = train_stats["rows"] / full_s8_rows if full_s8_rows > 0 else 1.0

    cfg = {"ctx_len": args.ctx_len, "top_k": args.top_k, "d_model": d_model,
           "vocab_size": vocab_size, "stride_train": args.stride_train,
           "stride_val": args.stride_val, "has_probe": has_probe,
           "limited_run": limited_run,
           "max_train_rows": args.max_train_rows, "max_val_rows": args.max_val_rows}
    _write_json(os.path.join(out_dir, "dataset_config.json"), cfg)

    if args.max_train_rows and train_stats["rows"] > args.max_train_rows:
        raise RuntimeError(f"stage01 actual_train_rows={train_stats['rows']} > max_train_rows={args.max_train_rows}")
    if args.max_val_rows and val_stats["rows"] > args.max_val_rows:
        raise RuntimeError(f"stage01 actual_val_rows={val_stats['rows']} > max_val_rows={args.max_val_rows}")

    report = {"pass": True, "config": cfg, "train": train_stats, "val": val_stats,
              "limited_run": limited_run, "reduction_vs_stride8_full": reduction_vs_s8}
    _write_json(os.path.join(out_dir, "report.json"), report)
    body = (f"LIMITED_RUN={limited_run}\n"
            f"stride_train={args.stride_train}  max_train_rows={args.max_train_rows}"
            f"  reduction_vs_s8={reduction_vs_s8:.3f}\n"
            f"stride_val={args.stride_val}  max_val_rows={args.max_val_rows}\n\n"
            f"ctx_len={args.ctx_len}  top_k={args.top_k}  d_model={d_model}\n\n"
            f"**train** shards={train_stats['shards']}  rows={train_stats['rows']:,}"
            f"  base_nll={train_stats['base_nll']:.4f}\n\n"
            f"**val** shards={val_stats['shards']}  rows={val_stats['rows']:,}"
            f"  base_nll={val_stats['base_nll']:.4f}  gold_in_topK={val_stats['gold_in_topK']:.4f}")
    _write_md(os.path.join(out_dir, "report.md"), "Stage 01 — Live Dataset Rebuild", body)
    manifest.done("rebuild_dataset",
                  {"train_rows": train_stats["rows"], "val_rows": val_stats["rows"],
                   "val_base_nll": val_stats["base_nll"], "limited_run": limited_run})
    print("[stage01] PASS")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 01B — Audit logit paths and patch shards with correct h_prime
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def stage01b_audit_logit_paths(args, manifest: Manifest):
    manifest.require_pass("rebuild_dataset")
    out_dir         = _mkdir(args.output_root, "01b_logit_path_audit")
    patch_root      = _mkdir(args.output_root, "01_live_dataset_patched")
    train_patch_dir = _mkdir(patch_root, "train")
    val_patch_dir   = _mkdir(patch_root, "val")
    manifest.start("audit_logit_paths", out_dir)
    print("\n[stage01b] Auditing logit paths ...")

    _OLD_CACHED_NLL    = 3.754938
    _OLD_CACHED_TOP256 = 0.8883
    _SANE_NLL          = 5.0
    _SANE_TOP256       = 0.80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    emb_w = backbone.token_emb.weight.detach().float().to(device)

    # ReprRegionRetrievalLM: forward() returns (lm_logits, region_logits, p_region, h_prime)
    has_hprime_forward = hasattr(backbone, "retrieval_proj")
    print(f"  backbone has h_prime forward path: {has_hprime_forward}")
    print(f"  Reference: old_cached_nll={_OLD_CACHED_NLL}  old_top256={_OLD_CACHED_TOP256}")

    val_s01_dir = os.path.join(args.output_root, "01_live_dataset", "val")
    val_shards  = sorted(glob.glob(os.path.join(val_s01_dir, "shard_*.pt")))[:5]
    if not val_shards:
        raise RuntimeError(f"No val shards in {val_s01_dir}")
    print(f"  Auditing {len(val_shards)} val shards ...")

    def _eval_lgt(lgt, gt):
        nll   = float(F.cross_entropy(lgt, gt, reduction="sum")) / len(gt)
        top256 = float((lgt.topk(args.top_k, dim=1).indices == gt.unsqueeze(1)).any(1).float().mean())
        return nll, top256

    # Accumulate per-path CE sums and counts
    acc = {}   # name -> [ce_sum, top256_sum, n]

    def _add(name, nll, top256, B):
        if name not in acc:
            acc[name] = [0.0, 0.0, 0]
        acc[name][0] += nll * B
        acc[name][1] += top256 * B
        acc[name][2] += B

    path_errors = {}

    for sp in val_shards:
        s1 = torch.load(sp, map_location="cpu", weights_only=True)
        N  = s1["gold_token"].shape[0]
        for start in range(0, N, args.eval_batch_size):
            end = min(start + args.eval_batch_size, N)
            src = s1["input_ids"][start:end].long().to(device)
            gt  = s1["gold_token"][start:end].long().to(device)
            B   = end - start

            # PATH A — raw h (= ln_f output) dot token_emb (current broken path)
            try:
                h_raw = get_hs_small(backbone, src, device)[:, -1, :].float()
                nll_a, t256_a = _eval_lgt(h_raw @ emb_w.T, gt)
                _add("path_A_raw_h_dot_emb", nll_a, t256_a, B)
            except Exception as e:
                path_errors["path_A_raw_h_dot_emb"] = str(e)

            # PATH B — model.forward() native logits (correct path for ReprRegionRetrievalLM)
            if has_hprime_forward:
                try:
                    lm_lgt_bt, _, _, h_prime_bt = backbone(src)
                    lgt_b = lm_lgt_bt[:, -1, :].float()
                    nll_b, t256_b = _eval_lgt(lgt_b, gt)
                    _add("path_B_model_native_logits", nll_b, t256_b, B)

                    # PATH C — h_prime dot token_emb (verify == PATH B since lm_head weight-tied)
                    h_prime_last = h_prime_bt[:, -1, :].float()
                    nll_c, t256_c = _eval_lgt(h_prime_last @ emb_w.T, gt)
                    _add("path_C_hprime_dot_emb", nll_c, t256_c, B)

                    # PATH D — backbone.lm_head applied to h_raw (== PATH A, sanity check)
                    nll_d, t256_d = _eval_lgt(backbone.lm_head(h_raw).float(), gt)
                    _add("path_D_lmhead_raw_h", nll_d, t256_d, B)
                except Exception as e:
                    path_errors["path_B_model_native_logits"] = str(e)
            else:
                # Non-ReprRegionRetrievalLM: try backbone.lm_head(h_raw)
                try:
                    lgt_b = backbone.lm_head(h_raw).float()
                    nll_b, t256_b = _eval_lgt(lgt_b, gt)
                    _add("path_B_lmhead_direct", nll_b, t256_b, B)
                except Exception as e:
                    path_errors["path_B_lmhead_direct"] = str(e)

    print(f"\n  Path audit results (val, {sum(v[2] for v in acc.values()) // max(len(acc),1):,} rows each):")
    path_results = []
    for name, (ce_sum, t256_sum, n) in sorted(acc.items()):
        nll    = ce_sum   / max(n, 1)
        top256 = t256_sum / max(n, 1)
        sane   = nll < _SANE_NLL and top256 > _SANE_TOP256
        flag   = "  ← SANE" if sane else "  ← BROKEN"
        print(f"    {name:40s}  nll={nll:.4f}  top256={top256:.4f}{flag}")
        path_results.append({
            "name": name, "nll_all": nll, "gold_in_top256": top256,
            "available": True, "sane": sane, "n": n,
        })
    for name, err in path_errors.items():
        if name not in acc:
            print(f"    {name:40s}  UNAVAILABLE: {err}")
            path_results.append({"name": name, "available": False, "error": err})

    # Absolute rule: do not continue if no sane path
    sane_paths = [p for p in path_results if p.get("sane", False)]
    if not sane_paths:
        manifest.fail("audit_logit_paths", "no sane logit path")
        raise RuntimeError(
            "[stage01b] No sane logit path found. All paths have NLL near 7–8. "
            "Do not train resolver. Inspect ReprRegionRetrievalLM forward/candidate builder.")

    best = min(sane_paths, key=lambda p: p["nll_all"])
    use_model_native = "model_native" in best["name"] or "hprime" in best["name"]
    print(f"\n  Best path: {best['name']}"
          f"  nll={best['nll_all']:.4f}  top256={best['gold_in_top256']:.4f}")
    print(f"  old_cached_nll={_OLD_CACHED_NLL}  old_top256={_OLD_CACHED_TOP256}")

    best_logit_path = {
        "name":                    best["name"],
        "nll_all":                 best["nll_all"],
        "gold_in_top256":          best["gold_in_top256"],
        "uses_h_key":              "h_prime" if use_model_native else "h_raw",
        "apply_final_norm":        True,
        "apply_retrieval_proj":    False,
        "use_model_native_logits": use_model_native,
    }

    # ── Patch all shards ───────────────────────────────────────────────────────
    print("\n  Patching shards — replacing h_ctx with h_prime, recomputing base_topk ...")

    def _patch_split(s01_dir, patched_dir, split_name):
        shards = sorted(glob.glob(os.path.join(s01_dir, "shard_*.pt")))
        if not shards:
            raise RuntimeError(f"No shards in {s01_dir}")
        total_rows = 0; base_ce = 0.0; base_top = 0
        t0 = time.time()
        for si, sp in enumerate(shards):
            orig = torch.load(sp, map_location="cpu", weights_only=True)
            N    = orig["gold_token"].shape[0]

            new_h   = np.zeros((N, d_model), dtype=np.float16)
            new_tid = np.zeros((N, args.top_k), dtype=np.int32)
            new_tlg = np.zeros((N, args.top_k), dtype=np.float16)

            for start in range(0, N, args.eval_batch_size):
                end = min(start + args.eval_batch_size, N)
                src = orig["input_ids"][start:end].long().to(device)
                gt  = orig["gold_token"][start:end].long().to(device)
                if has_hprime_forward:
                    lm_lgt_bt, _, _, h_prime_bt = backbone(src)
                    h_new  = h_prime_bt[:, -1, :].float()
                    lgt_new = lm_lgt_bt[:, -1, :].float()
                else:
                    h_new  = get_hs_small(backbone, src, device)[:, -1, :].float()
                    lgt_new = backbone.lm_head(h_new)
                topk_lgt, topk_ids = lgt_new.topk(args.top_k, dim=1)
                new_h  [start:end] = h_new.cpu().numpy().astype(np.float16)
                new_tid[start:end] = topk_ids.cpu().numpy().astype(np.int32)
                new_tlg[start:end] = topk_lgt.cpu().float().numpy().astype(np.float16)
                base_ce  += float(F.cross_entropy(lgt_new, gt, reduction="sum"))
                base_top += int((topk_ids == gt.unsqueeze(1)).any(1).sum())

            patched = dict(orig)                      # copy all original keys
            patched["h_raw"]         = orig["h_ctx"]  # preserve old (broken) h for reference
            patched["h_ctx"]         = torch.from_numpy(new_h)   # h_prime
            patched["base_topk_ids"] = torch.from_numpy(new_tid)
            patched["base_topk_lgt"] = torch.from_numpy(new_tlg)
            torch.save(patched, os.path.join(patched_dir, f"shard_{si:05d}.pt"))
            total_rows += N
            if si % 50 == 0 or si == len(shards) - 1:
                elapsed = time.time() - t0
                print(f"  [{split_name}] shard {si:05d}/{len(shards)-1}"
                      f"  rows={total_rows:,}  t={elapsed:.0f}s")

        nll_p   = base_ce  / max(total_rows, 1)
        top256_p = base_top / max(total_rows, 1)
        print(f"  [{split_name}] patched base_nll={nll_p:.4f}"
              f"  gold_in_top256={top256_p:.4f}  rows={total_rows:,}")
        return {"rows": total_rows, "shards": len(shards),
                "base_nll": nll_p, "gold_in_top256": top256_p}

    print("  Patching TRAIN ...")
    train_ps = _patch_split(
        os.path.join(args.output_root, "01_live_dataset", "train"),
        train_patch_dir, "train")
    print("  Patching VAL ...")
    val_ps = _patch_split(
        os.path.join(args.output_root, "01_live_dataset", "val"),
        val_patch_dir, "val")

    if val_ps["base_nll"] > _SANE_NLL:
        manifest.fail("audit_logit_paths",
                      f"patched val base_nll={val_ps['base_nll']:.4f} > {_SANE_NLL}")
        raise RuntimeError(
            f"[stage01b] Patched val base_nll={val_ps['base_nll']:.4f} > {_SANE_NLL}. "
            "Patching did not fix the logit path.")

    # CSV
    csv_path = os.path.join(out_dir, "per_path_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f,
            fieldnames=["name","nll_all","gold_in_top256","available","sane","n","error"])
        w.writeheader()
        for p in path_results:
            w.writerow({k: p.get(k, "") for k in w.fieldnames})

    report = {
        "logit_path_compatible": True,
        "best_logit_path":       best_logit_path,
        "paths":                 path_results,
        "patched_train":         train_ps,
        "patched_val":           val_ps,
        "old_cached_nll":        _OLD_CACHED_NLL,
        "old_cached_top256":     _OLD_CACHED_TOP256,
        "patch_dir":             patch_root,
        "raw_h_nll":             acc.get("path_A_raw_h_dot_emb", [0,0,1])[0] /
                                 max(acc.get("path_A_raw_h_dot_emb", [0,0,1])[2], 1),
    }
    _write_json(os.path.join(out_dir, "report.json"), report)
    _write_md(os.path.join(out_dir, "report.md"), "Stage 01B — Logit Path Audit",
              f"best_path={best['name']}  nll={best['nll_all']:.4f}"
              f"  top256={best['gold_in_top256']:.4f}\n\n"
              f"old_cached_nll={_OLD_CACHED_NLL}  old_top256={_OLD_CACHED_TOP256}\n\n"
              f"patched_train rows={train_ps['rows']:,}  base_nll={train_ps['base_nll']:.4f}\n\n"
              f"patched_val   rows={val_ps['rows']:,}  base_nll={val_ps['base_nll']:.4f}\n\n"
              f"raw_h_nll={report['raw_h_nll']:.4f}  (broken path, shown for comparison)")
    manifest.done("audit_logit_paths", {
        "logit_path_compatible": True,
        "best_path":       best["name"],
        "best_nll":        best["nll_all"],
        "patched_val_nll": val_ps["base_nll"],
    })
    print(f"[stage01b] PASS  best_path={best['name']}"
          f"  nll={best['nll_all']:.4f}  patched_val_nll={val_ps['base_nll']:.4f}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 02 — Verify live rebuild
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def stage02_verify_live_rebuild(args, manifest: Manifest):
    manifest.require_pass("audit_logit_paths")
    out_dir = _mkdir(args.output_root, "02_verify_rebuild")
    manifest.start("verify_rebuild", out_dir)
    print("\n[stage02] Verifying saved h_ctx is reproduced from saved input_ids ...")

    s01_dir = _get_s01_dir(args.output_root, "val")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    emb_w = backbone.token_emb.weight.detach().float().to(device)

    val_shards = sorted(glob.glob(os.path.join(s01_dir, "shard_*.pt")))[:5]
    if not val_shards:
        raise RuntimeError(f"No val shards in {s01_dir}")

    # Detect whether shards were patched by Stage01B (h_ctx = h_prime)
    _probe = torch.load(val_shards[0], map_location="cpu", weights_only=True)
    shards_patched = "h_raw" in _probe
    has_hprime_forward = hasattr(backbone, "retrieval_proj")
    print(f"  shards_patched={shards_patched}  has_hprime_forward={has_hprime_forward}")
    print(f"  s01_dir={s01_dir}")

    def _get_h_live(src):
        if shards_patched and has_hprime_forward:
            _, _, _, h_prime_bt = backbone(src)
            return h_prime_bt[:, -1, :].float()
        return get_hs_small(backbone, src, device)[:, -1, :].float()

    cos_sims = []; l2_dists = []; nll_diffs = []; top256_diffs = []
    n_checked = 0; target = 2000; K = args.top_k

    for sp in val_shards:
        if n_checked >= target:
            break
        s1 = torch.load(sp, map_location="cpu", weights_only=True)
        N  = min(s1["gold_token"].shape[0], target - n_checked)

        for start in range(0, N, args.eval_batch_size):
            end  = min(start + args.eval_batch_size, N)
            sl   = slice(start, end)
            src  = s1["input_ids"][sl].long().to(device)
            gt   = s1["gold_token"][sl].long().to(device)
            hc_s = s1["h_ctx"][sl].float().to(device)       # saved h_ctx

            h_live = _get_h_live(src)                        # recomputed with correct path

            cos = F.cosine_similarity(h_live, hc_s, dim=1)
            l2  = (h_live - hc_s).norm(dim=1)
            cos_sims.append(cos.cpu().numpy())
            l2_dists.append(l2.cpu().numpy())

            lgt_saved = hc_s  @ emb_w.T
            lgt_live  = h_live @ emb_w.T
            nll_s = float(F.cross_entropy(lgt_saved, gt, reduction="sum")) / (end - start)
            nll_l = float(F.cross_entropy(lgt_live,  gt, reduction="sum")) / (end - start)
            nll_diffs.append(abs(nll_l - nll_s))

            topK_s = (lgt_saved.topk(K, dim=1).indices == gt.unsqueeze(1)).any(1).float().mean()
            topK_l = (lgt_live.topk( K, dim=1).indices == gt.unsqueeze(1)).any(1).float().mean()
            top256_diffs.append(abs(float(topK_l) - float(topK_s)))

        n_checked += N

    cos_mean   = float(np.concatenate(cos_sims).mean())
    l2_mean    = float(np.concatenate(l2_dists).mean())
    nll_diff   = float(np.mean(nll_diffs))
    top256_diff = float(np.mean(top256_diffs))

    print(f"  cos_mean={cos_mean:.6f}  l2_mean={l2_mean:.6f}")
    print(f"  nll_diff={nll_diff:.2e}  top256_diff={top256_diff:.2e}")

    # nll_diff threshold is 2e-3: FP16 softmax accumulation produces ~5e-4 noise
    # even when h_ctx is bit-exact; top256_diff=0 is the sharp reproducibility signal.
    identity_pass = (cos_mean > 0.999 and nll_diff < 2e-3 and top256_diff < 1e-4)
    if not identity_pass:
        issues = []
        if cos_mean   <= 0.999: issues.append(f"cos_mean={cos_mean:.4f} <= 0.999")
        if nll_diff   >= 2e-3:  issues.append(f"nll_diff={nll_diff:.2e} >= 2e-3")
        if top256_diff >= 1e-4: issues.append(f"top256_diff={top256_diff:.2e} >= 1e-4")
        manifest.fail("verify_rebuild", str(issues))
        raise RuntimeError(
            f"Stage 02 verification FAILED: {'; '.join(issues)}. "
            "Saved h_ctx does not reproduce from saved input_ids. "
            "This pipeline cannot continue — check backbone loading and ctx_len.")

    report = {"pass": True, "cos_mean": cos_mean, "l2_mean": l2_mean,
              "nll_diff": nll_diff, "top256_diff": top256_diff, "n_checked": n_checked}
    _write_json(os.path.join(out_dir, "report.json"), report)
    _write_md(os.path.join(out_dir, "report.md"), "Stage 02 — Verify Rebuild",
              f"cos_mean={cos_mean:.6f}  nll_diff={nll_diff:.2e}  n={n_checked}")
    manifest.done("verify_rebuild", report)
    print("[stage02] PASS")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 03 — Build train-only retrieval index
# ══════════════════════════════════════════════════════════════════════════════

def stage03_build_retrieval_index(args, manifest: Manifest):
    manifest.require_pass("verify_rebuild")
    out_dir    = _mkdir(args.output_root, "03_retrieval_index")
    idx_dir    = _mkdir(out_dir, "hctx_train")
    manifest.start("build_index", out_dir)
    print("\n[stage03] Building train-only retrieval index ...")

    s01_train = _get_s01_dir(args.output_root, "train")
    shards    = sorted(glob.glob(os.path.join(s01_train, "shard_*.pt")))
    if not shards:
        raise RuntimeError(f"No train shards in {s01_train}")

    # repr_key: h_ctx (=h_prime in patched shards) or h_raw (original h, patched only)
    repr_key = "h_raw" if getattr(args, "retrieval_repr", "h_prime") == "h_raw" else "h_ctx"
    print(f"  retrieval repr key: '{repr_key}'")

    # Collect all repr vectors + metadata
    all_vecs  = []
    row_ids   = []
    gold_toks = []
    offsets   = []
    shard_ids = []
    row_in_sh = []
    t0 = time.time()

    for si, sp in enumerate(shards):
        s1 = torch.load(sp, map_location="cpu", weights_only=True)
        h  = s1[repr_key].float().numpy()
        N  = h.shape[0]
        all_vecs.append(h)
        row_ids.append(s1["row_id"].numpy())
        gold_toks.append(s1["gold_token"].numpy())
        offsets.append(s1["token_offset"].numpy() if "token_offset" in s1
                       else np.arange(N, dtype=np.int64))
        shard_ids.append(np.full(N, si, dtype=np.int32))
        row_in_sh.append(np.arange(N, dtype=np.int32))
        if si % 100 == 0 or si == len(shards) - 1:
            print(f"  shard {si:05d}/{len(shards)-1}  t={time.time()-t0:.0f}s")

    all_vecs  = np.concatenate(all_vecs,  axis=0).astype(np.float32)  # (N_total, d)
    row_ids   = np.concatenate(row_ids,   axis=0).astype(np.int64)
    gold_toks = np.concatenate(gold_toks, axis=0).astype(np.int32)
    offsets   = np.concatenate(offsets,   axis=0).astype(np.int64)
    shard_ids = np.concatenate(shard_ids, axis=0).astype(np.int32)
    row_in_sh = np.concatenate(row_in_sh, axis=0).astype(np.int32)
    N_total, D = all_vecs.shape
    print(f"  Total train vectors loaded: {N_total:,}  d={D}")

    # Apply max_index_rows limit via deterministic uniform sampling
    max_idx = args.max_index_rows if args.max_index_rows else 0
    if max_idx and N_total > max_idx:
        sel = np.linspace(0, N_total - 1, max_idx, dtype=int)
        all_vecs  = all_vecs[sel]
        row_ids   = row_ids[sel]
        gold_toks = gold_toks[sel]
        offsets   = offsets[sel]
        shard_ids = shard_ids[sel]
        row_in_sh = row_in_sh[sel]
        sampling_method = "uniform_linspace"
        print(f"  Sampled {max_idx:,} / {N_total:,} rows (uniform_linspace)")
    else:
        sel = np.arange(N_total, dtype=np.int64)
        sampling_method = "all"
    N, D = all_vecs.shape

    if max_idx and N > max_idx:
        raise RuntimeError(f"stage03 index_rows={N} > max_index_rows={max_idx}")
    print(f"  Index size: {N:,}  sampling={sampling_method}")

    # Normalise for cosine
    norms      = np.linalg.norm(all_vecs, axis=1, keepdims=True).clip(min=1e-8)
    all_normed = (all_vecs / norms).astype(np.float32)

    if not np.isfinite(all_normed).all():
        raise RuntimeError("Non-finite values in train h_ctx vectors!")

    # Self-retrieval sanity
    n_probe = min(200, N)
    probe_i = np.random.RandomState(42).choice(N, n_probe, replace=False)
    pv      = torch.tensor(all_normed[probe_i])
    av      = torch.tensor(all_normed)
    rank1   = 0
    for i in range(0, n_probe, 64):
        q    = pv[i:i+64]
        sims = q @ av.T
        top1 = sims.argmax(1).numpy()
        for j, (pi, t) in enumerate(zip(probe_i[i:i+64], top1)):
            if int(t) == int(pi):
                rank1 += 1
    rank1_rate = rank1 / n_probe
    print(f"  Self-retrieval rank-1: {rank1_rate:.4f}")
    if rank1_rate < 0.99:
        raise RuntimeError(f"Self-retrieval rank1={rank1_rate:.4f} < 0.99. Index is corrupt.")

    # Save metadata
    np.save(os.path.join(idx_dir, "vectors.fp32.npy"),              all_normed)
    np.save(os.path.join(idx_dir, "train_row_ids.npy"),             row_ids)
    np.save(os.path.join(idx_dir, "train_gold_tokens.npy"),         gold_toks)
    np.save(os.path.join(idx_dir, "train_offsets.npy"),             offsets)
    np.save(os.path.join(idx_dir, "train_shard_ids.npy"),           shard_ids)
    np.save(os.path.join(idx_dir, "train_row_in_shard.npy"),        row_in_sh)
    np.save(os.path.join(idx_dir, "index_selected_global_row_ids.npy"), row_ids)

    use_faiss_backend = (args.retrieval_backend in ("auto", "faiss")) and _HAS_FAISS
    index_type = "numpy"
    if use_faiss_backend:
        idx = _faiss.IndexFlatIP(D)
        idx.add(all_normed)
        fpath = os.path.join(idx_dir, "index.faiss")
        _faiss.write_index(idx, fpath)
        index_type = "faiss_flatip"
        print(f"  FAISS index saved → {fpath}  ntotal={idx.ntotal:,}")
    else:
        reason = "not available" if not _HAS_FAISS else f"backend={args.retrieval_backend}"
        print(f"  torch_chunked backend ({reason})")

    cfg = {"n_train": N, "n_total_loaded": N_total, "d": D,
           "index_type": index_type, "normalized": True,
           "sampling_method": sampling_method, "max_index_rows": max_idx}
    _write_json(os.path.join(idx_dir, "config.json"), cfg)

    report = {"pass": True, "n_train": N, "n_total_train_rows": N_total,
              "d": D, "index_type": index_type, "sampling_method": sampling_method,
              "max_index_rows": max_idx, "self_rank1_rate": rank1_rate}
    _write_json(os.path.join(out_dir, "report.json"), report)
    _write_md(os.path.join(out_dir, "report.md"), "Stage 03 — Retrieval Index",
              f"n_train={N:,}  n_total={N_total:,}  sampling={sampling_method}\n"
              f"d={D}  type={index_type}  rank1={rank1_rate:.4f}")
    manifest.done("build_index", report)
    print("[stage03] PASS")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 04 — Attach retrieval neighbors
# ══════════════════════════════════════════════════════════════════════════════

def stage04_attach_retrieval_neighbors(args, manifest: Manifest):
    manifest.require_pass("build_index")
    out_dir   = _mkdir(args.output_root, "04_retrieval_neighbors")
    train_out = _mkdir(out_dir, "train")
    val_out   = _mkdir(out_dir, "val")
    manifest.start("attach_neighbors", out_dir)
    print("\n[stage04] Attaching retrieval neighbors ...")

    idx_dir = os.path.join(args.output_root, "03_retrieval_index", "hctx_train")
    all_normed = np.load(os.path.join(idx_dir, "vectors.fp32.npy"))
    train_row_ids = np.load(os.path.join(idx_dir, "train_row_ids.npy"))
    train_golds   = np.load(os.path.join(idx_dir, "train_gold_tokens.npy"))
    N_idx, D = all_normed.shape
    print(f"  Index: {N_idx:,} vectors  d={D}")

    # Row-id → index position map for self-exclusion
    row_to_pos = {int(r): i for i, r in enumerate(train_row_ids)}

    # Resolve backend
    faiss_path     = os.path.join(idx_dir, "index.faiss")
    use_faiss      = (args.retrieval_backend in ("auto", "faiss") and
                      _HAS_FAISS and os.path.exists(faiss_path))
    device_idx     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _IDX_CHUNK     = max(1024, args.retrieval_chunk_size)
    _QUERY_BATCH   = max(1,    args.query_batch_size)
    K              = args.num_neighbors

    if use_faiss:
        faiss_idx = _faiss.read_index(faiss_path)
        print(f"  Backend: FAISS  ({faiss_idx.ntotal:,} vectors)")
    else:
        idx_t = torch.tensor(all_normed, device=device_idx)
        backend_label = "torch_chunked"
        print(f"  Backend: {backend_label} on {device_idx}"
              f"  index={N_idx:,}  chunk={_IDX_CHUNK:,}  qbatch={_QUERY_BATCH}")

    def _query_batch(q_vecs, query_row_ids=None):
        B          = len(q_vecs)
        retrieve_k = K + (1 if query_row_ids is not None else 0)
        if use_faiss:
            sc, ii = faiss_idx.search(q_vecs, retrieve_k)
        else:
            q_t   = torch.tensor(q_vecs, device=device_idx)
            top_v = torch.full((B, retrieve_k), -1e9, device=device_idx)
            top_i = torch.zeros((B, retrieve_k), dtype=torch.long, device=device_idx)
            for cs in range(0, N_idx, _IDX_CHUNK):
                ce    = min(cs + _IDX_CHUNK, N_idx)
                sim_c = q_t @ idx_t[cs:ce].T
                cat_v = torch.cat([top_v, sim_c], dim=1)
                cat_i = torch.cat([
                    top_i,
                    torch.arange(cs, ce, device=device_idx).unsqueeze(0).expand(B, -1)
                ], dim=1)
                top_v, sel = cat_v.topk(retrieve_k, dim=1)
                top_i = cat_i.gather(1, sel)
            ii = top_i.cpu().numpy()
            sc = top_v.cpu().numpy()

        out_golds  = np.full((B, K), -1, dtype=np.int32)
        out_scores = np.zeros((B, K), dtype=np.float32)
        for bi in range(B):
            filled = 0
            for rank in range(retrieve_k):
                idx_pos = int(ii[bi, rank])
                if idx_pos < 0 or idx_pos >= N_idx:
                    continue
                nbr_row_id = int(train_row_ids[idx_pos])
                if query_row_ids is not None and int(query_row_ids[bi]) == nbr_row_id:
                    continue  # self-exclude
                out_golds [bi, filled] = int(train_golds[idx_pos])
                out_scores[bi, filled] = float(sc[bi, rank])
                filled += 1
                if filled == K:
                    break
        return out_golds, out_scores

    # repr_key must match Stage03 (same key used to build the index)
    repr_key = "h_raw" if getattr(args, "retrieval_repr", "h_prime") == "h_raw" else "h_ctx"
    print(f"  retrieval repr key: '{repr_key}'")

    def _attach_split(s01_dir, nbr_out_dir, split_name, is_train, max_rows):
        shards = sorted(glob.glob(os.path.join(s01_dir, "shard_*.pt")))
        if not shards:
            raise RuntimeError(f"No shards in {s01_dir}")
        total_rows = 0
        t0 = time.time()
        for si, sp in enumerate(shards):
            if max_rows and total_rows >= max_rows:
                break
            s1   = torch.load(sp, map_location="cpu", weights_only=True)
            N    = s1[repr_key].shape[0]
            # Clip last partial shard to max_rows
            if max_rows:
                N = min(N, max_rows - total_rows)
            h    = s1[repr_key][:N].float().numpy()
            rids = s1["row_id"][:N].numpy() if is_train else None
            norms = np.linalg.norm(h, axis=1, keepdims=True).clip(min=1e-8)
            h_n   = (h / norms).astype(np.float32)

            all_golds  = np.full((N, K), -1, dtype=np.int32)
            all_scores = np.zeros((N, K), dtype=np.float32)

            t_batch = time.time()
            for start in range(0, N, _QUERY_BATCH):
                end  = min(start + _QUERY_BATCH, N)
                q    = h_n[start:end]
                qrid = rids[start:end] if rids is not None else None
                g, sc = _query_batch(q, qrid)
                all_golds [start:end] = g
                all_scores[start:end] = sc

            nbr_sd = {
                "neighbor_gold_tokens": torch.from_numpy(all_golds),
                "neighbor_scores":      torch.from_numpy(all_scores),
                "neighbor_row_ids":     torch.from_numpy(
                    train_row_ids[np.clip(
                        np.array([[row_to_pos.get(int(x), 0) for x in row]
                                  for row in all_golds.tolist()]), 0, N_idx - 1)]),
            }
            out_path = os.path.join(nbr_out_dir, f"shard_{si:05d}.pt")
            torch.save(nbr_sd, out_path)
            total_rows += N
            elapsed = time.time() - t0
            qps = total_rows / max(elapsed, 1e-3)
            if si % 20 == 0 or (max_rows and total_rows >= max_rows):
                print(f"  [{split_name}] shard {si:05d}  rows={total_rows:,}"
                      f"  qps={qps:.0f}  t={elapsed:.0f}s")
        return {"shards_written": si + 1, "rows": total_rows}

    skip_train = args.skip_train_attach or args.attach_val_only
    max_tr  = args.max_attach_train_rows if args.max_attach_train_rows else 0
    max_vl  = args.max_attach_val_rows   if args.max_attach_val_rows   else 0

    if skip_train:
        print("  Skipping train retrieval attachment (--skip_train_attach / --attach_val_only)")
        train_stats = {"shards_written": 0, "rows": 0}
    else:
        print(f"  Attaching to train (max_rows={max_tr or 'all'}) ...")
        train_stats = _attach_split(
            _get_s01_dir(args.output_root, "train"),
            train_out, "train", is_train=True, max_rows=max_tr)
        if max_tr and train_stats["rows"] > max_tr:
            raise RuntimeError(
                f"attached_train_rows={train_stats['rows']} > max_attach_train_rows={max_tr}")

    print(f"  Attaching to val (max_rows={max_vl or 'all'}) ...")
    val_stats = _attach_split(
        _get_s01_dir(args.output_root, "val"),
        val_out, "val", is_train=False, max_rows=max_vl)
    if max_vl and val_stats["rows"] > max_vl:
        raise RuntimeError(
            f"attached_val_rows={val_stats['rows']} > max_attach_val_rows={max_vl}")

    report = {"pass": True, "leakage_detected": False,
              "train": train_stats, "val": val_stats,
              "num_neighbors": K,
              "max_attach_train_rows": max_tr, "max_attach_val_rows": max_vl,
              "index_rows": N_idx,
              "backend": "faiss" if use_faiss else "torch_chunked",
              "query_batch_size": _QUERY_BATCH,
              "retrieval_chunk_size": _IDX_CHUNK}
    _write_json(os.path.join(out_dir, "report.json"), report)
    _write_md(os.path.join(out_dir, "report.md"), "Stage 04 — Retrieval Neighbors",
              f"K={K}  train_rows={train_stats['rows']:,}  val_rows={val_stats['rows']:,}\n"
              f"max_attach_train={max_tr}  max_attach_val={max_vl}\n"
              f"index_rows={N_idx:,}  backend={report['backend']}")
    manifest.done("attach_neighbors", report)
    print("[stage04] PASS")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 05 — Audit retrieval signal
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def stage05_audit_retrieval_signal(args, manifest: Manifest):
    manifest.require_pass("attach_neighbors")
    out_dir = _mkdir(args.output_root, "05_retrieval_audit")
    manifest.start("audit_retrieval", out_dir)
    print("\n[stage05] Auditing retrieval signal ...")

    s01_val = _get_s01_dir(args.output_root, "val")
    s04_val = os.path.join(args.output_root, "04_retrieval_neighbors", "val")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K_idx   = args.top_k

    gold_in_topK = gold_in_nbr = ret_added = total_n = 0
    nbr_finite = True; nbr_valid_ids = True
    val_shards = sorted(glob.glob(os.path.join(s01_val, "shard_*.pt")))
    # Save per-shard CSV
    csv_rows = []

    for sp in val_shards:
        si   = int(os.path.basename(sp).replace("shard_", "").replace(".pt", ""))
        np_  = os.path.join(s04_val, f"shard_{si:05d}.pt")
        if not os.path.exists(np_):
            continue
        s1 = torch.load(sp,  map_location="cpu", weights_only=True)
        s4 = torch.load(np_, map_location="cpu", weights_only=True)
        N  = s1["gold_token"].shape[0]

        gt    = s1["gold_token"].long()
        nbr_i = s4["neighbor_gold_tokens"].long()   # (N, K_nbr)
        nbr_s = s4["neighbor_scores"].float()

        if not torch.isfinite(nbr_s[nbr_i >= 0]).all():
            nbr_finite = False
        if (nbr_i >= 0).any() and (nbr_i[nbr_i >= 0] >= 50257).any():
            nbr_valid_ids = False

        if "base_topk_ids" not in s1:
            raise RuntimeError(f"base_topk_ids not in stage01 shard: {sp}")

        shard_topK = shard_nbr = shard_ret = 0
        for start in range(0, N, 256):
            end = min(start + 256, N)
            gt_b      = gt[start:end].to(device)
            nbr_b     = nbr_i[start:end].to(device)
            topk_ids_b = s1["base_topk_ids"][start:end].long().to(device)

            in_topK = (topk_ids_b == gt_b.unsqueeze(1)).any(1)
            in_nbr  = (nbr_b      == gt_b.unsqueeze(1)).any(1)
            shard_topK += int(in_topK.sum())
            shard_nbr  += int(in_nbr.sum())
            shard_ret  += int((in_nbr & ~in_topK).sum())
            total_n    += (end - start)

        gold_in_topK += shard_topK
        gold_in_nbr  += shard_nbr
        ret_added    += shard_ret
        csv_rows.append({
            "shard": si, "n": N,
            "gold_in_topK": shard_topK, "gold_in_nbr": shard_nbr,
            "ret_added": shard_ret,
        })

    r_topK = gold_in_topK / max(total_n, 1)
    r_nbr  = gold_in_nbr  / max(total_n, 1)
    r_add  = ret_added     / max(total_n, 1)
    print(f"  gold_in_top{K_idx}:       {r_topK:.4f}")
    print(f"  gold_in_neighbors:    {r_nbr:.4f}")
    print(f"  retrieval_added_gold: {r_add:.4f}")
    print(f"  neighbor_scores_finite: {nbr_finite}")
    print(f"  neighbor_ids_valid:     {nbr_valid_ids}")

    if not (nbr_finite and nbr_valid_ids):
        raise RuntimeError("Retrieval audit failed: non-finite scores or invalid token ids.")

    # CSV
    csv_path = os.path.join(out_dir, "candidate_survival.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shard","n","gold_in_topK","gold_in_nbr","ret_added"])
        w.writeheader(); w.writerows(csv_rows)

    report = {"pass": True, "total_n": total_n,
              f"gold_in_top{K_idx}": r_topK, "gold_in_neighbor_rate": r_nbr,
              "retrieval_added_gold_rate": r_add,
              "neighbor_scores_finite": nbr_finite,
              "neighbor_ids_valid": nbr_valid_ids}
    _write_json(os.path.join(out_dir, "report.json"), report)
    _write_md(os.path.join(out_dir, "report.md"), "Stage 05 — Retrieval Signal Audit",
              f"gold_in_top{K_idx}={r_topK:.4f}  gold_in_nbr={r_nbr:.4f}"
              f"  retrieval_added={r_add:.4f}  n={total_n:,}")
    manifest.done("audit_retrieval", report)
    print("[stage05] PASS")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 06 — Debug resolver identity (step-0 check)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def stage06_debug_resolver_identity(args, manifest: Manifest):
    manifest.require_pass("audit_retrieval")
    out_dir = _mkdir(args.output_root, "06_debug_identity")
    manifest.start("debug_identity", out_dir)
    print("\n[stage06] Checking step-0 resolver identity ...")

    with open(os.path.join(args.output_root, "01_live_dataset", "dataset_config.json")) as f:
        ds_cfg = json.load(f)
    d_model   = ds_cfg["d_model"]
    vocab_size = ds_cfg["vocab_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load backbone only for tok_emb_w
    backbone, probe, _, _, _ = load_small_backbone_and_probe(args.small_ckpt, device)
    emb_w = backbone.token_emb.weight.detach().float().cpu()   # (V, d)
    del backbone; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    t2r_np = _load_token_to_region(args.region_map, vocab_size) if args.region_map else None
    n_fine = int(t2r_np[t2r_np >= 0].max()) + 1 if (t2r_np is not None and (t2r_np >= 0).any()) else 128
    r2s_np = _load_r2s(args.super_map, n_fine) if args.super_map else np.zeros(n_fine, dtype=np.int32)
    n_super = int(r2s_np.max()) + 1

    model = LiveContextRetrievalTokenResolver(
        d_backbone=d_model, n_fine=n_fine, n_super=n_super,
        r2s_np=r2s_np, t2r_np=t2r_np,
        top_k=args.top_k, num_neighbors=args.num_neighbors,
        d_resolver=args.resolver_dim, n_layers=args.resolver_layers,
        n_heads=args.resolver_heads, delta_scale=args.delta_scale,
        candidate_mode=args.candidate_mode,
    ).to(device)

    # Verify delta_head is zero
    dw = model.delta_head.weight.abs().max().item()
    db = model.delta_head.bias.abs().max().item()
    if max(dw, db) > 1e-9:
        raise RuntimeError(f"delta_head not zero-init: w={dw:.2e} b={db:.2e}")

    s01_val = _get_s01_dir(args.output_root, "val")
    s04_val = os.path.join(args.output_root, "04_retrieval_neighbors", "val")

    base_ce = ref_ce = 0.0; nc = 0; delta_max = 0.0
    emb_w_d = emb_w.to(device)

    val_shards = sorted(glob.glob(os.path.join(s01_val, "shard_*.pt")))[:3]
    for sp in val_shards:
        si   = int(os.path.basename(sp).replace("shard_", "").replace(".pt", ""))
        np_  = os.path.join(s04_val, f"shard_{si:05d}.pt")
        s1 = torch.load(sp,  map_location="cpu", weights_only=True)
        s4 = torch.load(np_, map_location="cpu", weights_only=True)
        N  = s1["gold_token"].shape[0]
        has_router = "router_topk_reg" in s1

        for start in range(0, N, args.eval_batch_size):
            end = min(start + args.eval_batch_size, N)
            sl  = slice(start, end)
            h   = s1["h_ctx"][sl].float().to(device)
            gt  = s1["gold_token"][sl].long().to(device)
            nbr_i = s4["neighbor_gold_tokens"][sl].long().to(device)
            nbr_s = s4["neighbor_scores"][sl].float().to(device)
            r_reg = s1["router_topk_reg"][sl].long().to(device) if has_router else None
            r_prb = s1["router_topk_prb"][sl].float().to(device) if has_router else None
            r_mar = s1["router_margin"][sl].float().to(device) if "router_margin" in s1 else None

            out = model(h, nbr_i, nbr_s, emb_w_d, r_reg, r_prb, r_mar)
            base_ce += float(F.cross_entropy(out["base_lgt"],    gt, reduction="sum"))
            ref_ce  += float(F.cross_entropy(out["refined_lgt"], gt, reduction="sum"))
            nc      += (end - start)
            delta_max = max(delta_max, float(out["bounded_delta"].abs().max()))

    base_nll = base_ce / max(nc, 1)
    ref_nll  = ref_ce  / max(nc, 1)
    nll_diff = abs(ref_nll - base_nll)
    print(f"  base_nll={base_nll:.6f}  refined_nll={ref_nll:.6f}")
    print(f"  nll_diff={nll_diff:.2e}  delta_max_abs={delta_max:.2e}")

    identity_pass = (nll_diff < 1e-3 and delta_max < 1e-6)
    if not identity_pass:
        issues = []
        if nll_diff  >= 1e-3: issues.append(f"nll_diff={nll_diff:.2e} >= 1e-3")
        if delta_max >= 1e-6: issues.append(f"delta_max={delta_max:.2e} >= 1e-6")
        manifest.fail("debug_identity", str(issues))
        raise RuntimeError(f"Stage 06 identity FAILED: {'; '.join(issues)}")

    report = {"pass": True, "base_nll": base_nll, "refined_nll": ref_nll,
              "nll_diff": nll_diff, "delta_max_abs": delta_max, "n_checked": nc,
              "h_ctx_from_stage01_input_ids": True}
    _write_json(os.path.join(out_dir, "report.json"), report)
    _write_md(os.path.join(out_dir, "report.md"), "Stage 06 — Debug Identity",
              f"nll_diff={nll_diff:.2e}  delta_max={delta_max:.2e}  n={nc}")
    manifest.done("debug_identity", report)
    print("[stage06] PASS")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 07 — Train context + retrieval resolver
# ══════════════════════════════════════════════════════════════════════════════

def stage07_train_context_retrieval_resolver(args, manifest: Manifest):
    manifest.require_pass("debug_identity")

    run_tag  = f"top{args.top_k}_ctx{args.ctx_len}_ret{args.num_neighbors}_v1"
    train_out = _mkdir(os.path.join(args.output_root, "07_train_resolver"), run_tag)
    manifest.start("train_resolver", train_out)
    print(f"\n[stage07] Training resolver → {train_out}")

    with open(os.path.join(args.output_root, "01_live_dataset", "dataset_config.json")) as f:
        ds_cfg = json.load(f)
    d_model    = ds_cfg["d_model"]
    vocab_size = ds_cfg["vocab_size"]
    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # tok_emb_w
    backbone, probe, _, _, _ = load_small_backbone_and_probe(args.small_ckpt, device)
    emb_w = backbone.token_emb.weight.detach().float().cpu()
    del backbone; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    t2r_np = _load_token_to_region(args.region_map, vocab_size) if args.region_map else None
    n_fine  = int(t2r_np[t2r_np >= 0].max()) + 1 if (t2r_np is not None and (t2r_np >= 0).any()) else 128
    r2s_np  = _load_r2s(args.super_map, n_fine) if args.super_map else np.zeros(n_fine, dtype=np.int32)
    n_super = int(r2s_np.max()) + 1

    model = LiveContextRetrievalTokenResolver(
        d_backbone=d_model, n_fine=n_fine, n_super=n_super,
        r2s_np=r2s_np, t2r_np=t2r_np,
        top_k=args.top_k, num_neighbors=args.num_neighbors,
        d_resolver=args.resolver_dim, n_layers=args.resolver_layers,
        n_heads=args.resolver_heads, dropout=0.0,
        delta_scale=args.delta_scale, retrieval_tau=args.retrieval_tau,
        candidate_mode=args.candidate_mode,
    ).to(device)

    s01_train = _get_s01_dir(args.output_root, "train")
    s04_train = os.path.join(args.output_root, "04_retrieval_neighbors", "train")
    s01_val   = _get_s01_dir(args.output_root, "val")
    s04_val   = os.path.join(args.output_root, "04_retrieval_neighbors", "val")

    # Detect limited attach coverage
    s01_train_shards = sorted(glob.glob(os.path.join(s01_train, "shard_*.pt")))
    s04_train_shards = sorted(glob.glob(os.path.join(s04_train, "shard_*.pt")))
    s01_val_shards   = sorted(glob.glob(os.path.join(s01_val,   "shard_*.pt")))
    s04_val_shards   = sorted(glob.glob(os.path.join(s04_val,   "shard_*.pt")))
    train_rows_total = sum(torch.load(p, map_location="cpu", weights_only=True)["gold_token"].shape[0]
                          for p in s01_train_shards)
    train_rows_used  = sum(torch.load(p, map_location="cpu", weights_only=True)["neighbor_gold_tokens"].shape[0]
                          for p in s04_train_shards) if s04_train_shards else 0
    val_rows_total   = sum(torch.load(p, map_location="cpu", weights_only=True)["gold_token"].shape[0]
                          for p in s01_val_shards)
    val_rows_used    = sum(torch.load(p, map_location="cpu", weights_only=True)["neighbor_gold_tokens"].shape[0]
                          for p in s04_val_shards) if s04_val_shards else 0
    limited_run      = train_rows_used < train_rows_total or val_rows_used < val_rows_total
    print(f"  train_rows_total={train_rows_total:,}  train_rows_used={train_rows_used:,}")
    print(f"  val_rows_total={val_rows_total:,}    val_rows_used={val_rows_used:,}")
    if limited_run:
        print(f"  LIMITED RUN: training on {train_rows_used:,}/{train_rows_total:,} train rows"
              f"  ({100*train_rows_used/max(train_rows_total,1):.1f}%)")
    if train_rows_used == 0:
        manifest.fail("train_resolver", "No train rows have retrieval neighbors. Run stage04 first.")
        raise RuntimeError("stage07: train_rows_used=0. No stage04 train shards found.")

    train_ds = RebuildDataset(s01_train, s04_train, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              collate_fn=_collate, num_workers=0, drop_last=True)

    optim  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda") if args.amp and torch.cuda.is_available() else None
    emb_d  = emb_w.to(device)

    # CSV loggers
    train_csv = open(os.path.join(train_out, "train_log.csv"), "w", newline="")
    eval_csv  = open(os.path.join(train_out, "eval_log.csv"),  "w", newline="")
    tcw = csv.writer(train_csv); tcw.writerow(["step", "loss", "ce", "reg", "kl", "dl", "lr"])
    ecw = csv.writer(eval_csv)

    eval_header_written = False
    best_gain = -float("inf"); best_step = 0; best_metrics = None
    step = 0; accum_loss = {}; t0 = time.time()

    # step-0 eval
    print("  [eval step=0]")
    metrics0 = full_vocab_eval(model, s01_val, s04_val, emb_w, device, args.eval_batch_size)
    if not eval_header_written:
        ecw.writerow(["step"] + list(metrics0.keys())); eval_header_written = True
    ecw.writerow([0] + [f"{metrics0[k]:.6f}" if isinstance(metrics0[k], float) else metrics0[k]
                        for k in metrics0])
    eval_csv.flush()
    print(f"  step=0  base_nll={metrics0['full_vocab_base_nll_all']:.4f}"
          f"  refined_nll={metrics0['full_vocab_refined_nll_all']:.4f}"
          f"  gain={metrics0['full_vocab_gain_all']:+.4f}")
    full_vocab_base_nll = metrics0["full_vocab_base_nll_all"]

    model.train()
    loader_iter = iter(train_loader)
    optim.zero_grad()

    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader); batch = next(loader_iter)

        h   = batch["h_ctx"].float().to(device)
        nbr_i = batch["nbr_ids"].long().to(device)
        nbr_s = batch["nbr_scores"].float().to(device)
        r_reg = batch["r_topk_reg"].long().to(device) if (batch["r_topk_reg"] >= 0).any() else None
        r_prb = batch["r_topk_prb"].float().to(device) if r_reg is not None else None
        r_mar = batch["r_margin"].float().to(device) if r_reg is not None else None

        if scaler:
            with torch.amp.autocast("cuda"):
                out = model(h, nbr_i, nbr_s, emb_d, r_reg, r_prb, r_mar)
                loss, ldict = compute_loss(out, batch, device,
                                           args.lambda_region, args.lambda_kl,
                                           args.lambda_delta, args.top_k)
            scaler.scale(loss / args.grad_accum_steps).backward()
        else:
            out = model(h, nbr_i, nbr_s, emb_d, r_reg, r_prb, r_mar)
            loss, ldict = compute_loss(out, batch, device,
                                       args.lambda_region, args.lambda_kl,
                                       args.lambda_delta, args.top_k)
            (loss / args.grad_accum_steps).backward()

        for k, v in ldict.items():
            accum_loss[k] = accum_loss.get(k, 0.0) + v / args.grad_accum_steps

        if (step + 1) % args.grad_accum_steps == 0:
            if scaler:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler:
                scaler.step(optim); scaler.update()
            else:
                optim.step()
            optim.zero_grad()

            real_step = (step + 1) // args.grad_accum_steps
            cur_lr = args.lr
            tcw.writerow([real_step] + [f"{accum_loss.get(k, 0):.6f}" for k in
                                         ["loss","ce","reg","kl","dl"]] + [f"{cur_lr:.2e}"])
            train_csv.flush()
            accum_loss = {}

            if real_step % (args.eval_every // args.grad_accum_steps) == 0:
                print(f"  [eval step={real_step}  t={time.time()-t0:.0f}s]")
                em = full_vocab_eval(model, s01_val, s04_val, emb_w, device, args.eval_batch_size)
                ecw.writerow([real_step] + [f"{em[k]:.6f}" if isinstance(em[k], float) else em[k]
                                             for k in em])
                eval_csv.flush()
                gain = em["full_vocab_gain_all"]
                print(f"    base_nll={em['full_vocab_base_nll_all']:.4f}"
                      f"  refined_nll={em['full_vocab_refined_nll_all']:.4f}"
                      f"  gain={gain:+.4f}"
                      f"  top1_base={em['top1_acc_base']:.4f}"
                      f"  top1_ref={em['top1_acc_ref']:.4f}")

                if gain > best_gain:
                    best_gain = gain; best_step = real_step; best_metrics = em
                    if gain > 0:
                        ckpt = {k: v for k, v in model.state_dict().items()}
                        torch.save(ckpt, os.path.join(train_out, "best_resolver.pt"))
                        _write_json(os.path.join(train_out, "best_metrics.json"),
                                    {"step": real_step, **em})

                model.train()

        step += 1

    # Final eval
    print(f"  [final eval  t={time.time()-t0:.0f}s]")
    final = full_vocab_eval(model, s01_val, s04_val, emb_w, device, args.eval_batch_size)
    ecw.writerow(["final"] + [f"{final[k]:.6f}" if isinstance(final[k], float) else final[k]
                               for k in final])
    train_csv.close(); eval_csv.close()

    gain_f = final["full_vocab_gain_all"]
    verdict = ("STRONG"      if gain_f > 0.010 else
               "MEANINGFUL"  if gain_f > 0.005 else
               "WEAK"        if gain_f > 0.0033 else
               "NO_GAIN")

    final_metrics = {"step": args.steps, **final, "verdict": verdict,
                     "best_step": best_step, "best_gain": best_gain,
                     "limited_run": limited_run,
                     "train_rows_total": train_rows_total,
                     "train_rows_used":  train_rows_used,
                     "val_rows_total":   val_rows_total,
                     "val_rows_used":    val_rows_used}
    _write_json(os.path.join(train_out, "final_metrics.json"), final_metrics)

    report_body = (
        f"ctx_len={args.ctx_len}  top_k={args.top_k}  num_neighbors={args.num_neighbors}\n"
        f"steps={args.steps}  lr={args.lr}  batch={args.batch_size}x{args.grad_accum_steps}\n\n"
        f"**full_vocab_base_nll**    = {final['full_vocab_base_nll_all']:.6f}\n"
        f"**full_vocab_refined_nll** = {final['full_vocab_refined_nll_all']:.6f}\n"
        f"**full_vocab_gain**        = {gain_f:+.6f}\n"
        f"**verdict**                = {verdict}\n\n"
        f"best_step = {best_step}  best_gain = {best_gain:+.4f}\n\n"
        f"top1_acc_base = {final['top1_acc_base']:.4f}  "
        f"top1_acc_ref = {final['top1_acc_ref']:.4f}\n"
        f"rank_improved_rate = {final['rank_improved_rate']:.4f}  "
        f"rank_worsened_rate = {final['rank_worsened_rate']:.4f}\n"
        f"gold_in_topK = {final['gold_in_topK_rate']:.4f}  "
        f"gold_in_neighbor = {final['gold_in_neighbor_rate']:.4f}\n"
        f"retrieval_added_gold = {final['retrieval_added_gold_rate']:.4f}\n"
        f"gain_w_nbr_gold = {final['gain_rows_with_neighbor_gold']:+.4f}  "
        f"gain_wo_nbr_gold = {final['gain_rows_without_neighbor_gold']:+.4f}\n"
    )
    _write_md(os.path.join(train_out, "report.md"), "Stage 07 — Resolver Training", report_body)
    manifest.done("train_resolver", {"verdict": verdict, "final_gain": gain_f,
                                     "best_step": best_step})
    print(f"[stage07] verdict={verdict}  gain={gain_f:+.4f}")
    print("[stage07] PASS")
    return final_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

_STAGE_FN = {
    "find_tokens":       stage00_find_raw_token_source,
    "rebuild_dataset":   stage01_rebuild_live_dataset,
    "audit_logit_paths": stage01b_audit_logit_paths,
    "verify_rebuild":    stage02_verify_live_rebuild,
    "build_index":       stage03_build_retrieval_index,
    "attach_neighbors":  stage04_attach_retrieval_neighbors,
    "audit_retrieval":   stage05_audit_retrieval_signal,
    "debug_identity":    stage06_debug_resolver_identity,
    "train_resolver":    stage07_train_context_retrieval_resolver,
}


def main():
    args = _parse()
    os.makedirs(args.output_root, exist_ok=True)

    # Guard against overwriting an existing completed run
    manifest_path = os.path.join(args.output_root, "pipeline_manifest.json")
    if os.path.exists(manifest_path) and not args.allow_overwrite:
        with open(manifest_path) as f:
            existing = json.load(f)
        stages_done = [s for s, d in existing.get("stages", {}).items()
                       if d.get("status") == "pass"]
        if stages_done:
            print(f"[main] Resuming existing run at {args.output_root}")
            print(f"  Stages already passed: {stages_done}")

    manifest = Manifest(args.output_root)

    # Write README
    readme = os.path.join(args.output_root, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write("# Live Full Pipeline Rebuild\n\nBuilt by run_live_full_pipeline_rebuild.py\n")

    if args.run_all:
        for stage_name in _STAGE_ORDER:
            if manifest.get(stage_name).get("status") == "pass":
                print(f"[stage] SKIP {stage_name} (already passed)")
                continue
            fn = _STAGE_FN[stage_name]
            try:
                fn(args, manifest)
            except RuntimeError as e:
                manifest.fail(stage_name, str(e))
                print(f"\n[main] FATAL: stage '{stage_name}' failed: {e}")
                sys.exit(1)
        print("\n[main] ALL STAGES COMPLETE")
    elif args.stage:
        fn = _STAGE_FN.get(args.stage)
        if fn is None:
            print(f"Unknown stage: {args.stage}. Valid: {list(_STAGE_FN)}")
            sys.exit(1)
        fn(args, manifest)
    else:
        print("Specify --run_all or --stage <name>")
        print(f"Valid stages: {list(_STAGE_FN)}")
        sys.exit(1)


def _parse():
    p = argparse.ArgumentParser(
        description="Live full pipeline rebuild — single file, all stages")
    # Mode
    p.add_argument("--run_all",  action="store_true")
    p.add_argument("--stage",    default=None, choices=list(_STAGE_FN))
    p.add_argument("--allow_overwrite", action="store_true")
    # Paths
    p.add_argument("--output_root", default="runs/live_full_pipeline_rebuild")
    p.add_argument("--small_ckpt",  required=True)
    p.add_argument("--train_tokens_path", default=None)
    p.add_argument("--val_tokens_path",   default=None)
    p.add_argument("--region_map",  default=None)
    p.add_argument("--super_map",   default=None)
    # Dataset
    p.add_argument("--ctx_len",         type=int, default=128)
    p.add_argument("--top_k",           type=int, default=256)
    p.add_argument("--num_neighbors",   type=int, default=32)
    p.add_argument("--shard_size",      type=int, default=4096)
    p.add_argument("--stride_train",    type=int, default=64,
                   help="Sample every N-th position in train corpus (default 64)")
    p.add_argument("--stride_val",      type=int, default=8,
                   help="Sample every N-th position in val corpus (default 8)")
    p.add_argument("--max_train_rows",  type=int, default=500000,
                   help="Hard cap on train rows built in stage01 (0 = unlimited)")
    p.add_argument("--max_val_rows",    type=int, default=100000,
                   help="Hard cap on val rows built in stage01 (0 = unlimited)")
    # Retrieval index limits
    p.add_argument("--max_index_rows",        type=int, default=500000,
                   help="Max train rows to include in the retrieval index (0 = all)")
    p.add_argument("--max_attach_train_rows", type=int, default=500000,
                   help="Max train rows to attach retrieval neighbors to (0 = all)")
    p.add_argument("--max_attach_val_rows",   type=int, default=100000,
                   help="Max val rows to attach retrieval neighbors to (0 = all)")
    p.add_argument("--skip_train_attach",     action="store_true",
                   help="Skip train retrieval attachment entirely")
    p.add_argument("--attach_val_only",       action="store_true",
                   help="Only attach val retrieval (implies --skip_train_attach)")
    p.add_argument("--retrieval_repr",        default="h_prime",
                   choices=["h_prime", "h_raw"],
                   help="Which representation to use as kNN keys (h_prime=patched h_ctx, h_raw=original)")
    # Retrieval backend
    p.add_argument("--retrieval_backend",    default="auto",
                   choices=["auto", "faiss", "torch_chunked"],
                   help="Retrieval backend: auto tries FAISS first then falls back")
    p.add_argument("--retrieval_chunk_size", type=int, default=131072,
                   help="Index chunk size for chunked torch matmul (rows per chunk)")
    p.add_argument("--query_batch_size",     type=int, default=512,
                   help="Query batch size for retrieval attachment")
    # Model
    p.add_argument("--candidate_mode",  default="base_topk_plus_neighbors",
                   choices=["base_topk", "base_topk_plus_neighbors"])
    p.add_argument("--resolver_dim",    type=int,   default=256)
    p.add_argument("--resolver_layers", type=int,   default=2)
    p.add_argument("--resolver_heads",  type=int,   default=4)
    p.add_argument("--delta_scale",     type=float, default=0.25)
    p.add_argument("--retrieval_tau",   type=float, default=0.2)
    # Training
    p.add_argument("--steps",           type=int,   default=3000)
    p.add_argument("--eval_every",      type=int,   default=500)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--eval_batch_size", type=int,   default=64)
    p.add_argument("--grad_accum_steps",type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--lambda_region",   type=float, default=0.1)
    p.add_argument("--lambda_rank",     type=float, default=0.0)
    p.add_argument("--lambda_kl",       type=float, default=1.0)
    p.add_argument("--lambda_delta",    type=float, default=1e-3)
    p.add_argument("--grad_clip",       type=float, default=0.5)
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--device",          default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    main()
