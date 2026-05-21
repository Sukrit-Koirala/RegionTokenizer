#!/usr/bin/env python3
"""
Stage 02 — Build Train-Only Retrieval Index.

Runs frozen backbone on train input_ids (Stage 01 output) → h_ctx vectors.
Builds FAISS (preferred) or numpy nearest-neighbour index over TRAIN rows only.
Val rows are NEVER indexed.

Hard pass conditions:
  n_train_vectors == total train rows from Stage 01
  vectors normalised
  random self-query retrieves self at rank 1
  no val rows in index (by construction)

Outputs:
  output_dir/vectors.npy           float32 (N, d_model)
  output_dir/train_row_ids.npy     int64   (N,)
  output_dir/faiss.index           FAISS IndexFlatIP (if available)
  output_dir/index_report.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.offline_region_knn import load_small_backbone_and_probe, get_hs_small

try:
    import faiss as _faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    print("[stage02] WARNING: faiss not available; will use numpy index only")


# ── Dataset ───────────────────────────────────────────────────────────────────

class CtxShardDataset(Dataset):
    """Streams input_ids and row_ids from a single context shard."""
    def __init__(self, shard_path: str):
        data = torch.load(shard_path, map_location="cpu", weights_only=True)
        self.input_ids = data["input_ids"].long()      # (N, ctx_len)
        self.row_ids   = data["row_id"].long()         # (N,)

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "row_id": self.row_ids[i]}


# ── Normalise ─────────────────────────────────────────────────────────────────

def normalise_np(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    return vecs / norms


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[stage02] Loading backbone: {args.small_ckpt}")
    backbone, _, d_model, _, vocab_size = load_small_backbone_and_probe(
        args.small_ckpt, device
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    print(f"  d_model={d_model}  vocab_size={vocab_size}")

    # ── Collect train h_ctx vectors ───────────────────────────────────────────
    ctx_paths = sorted([
        os.path.join(args.train_ctx_dir, f)
        for f in os.listdir(args.train_ctx_dir)
        if f.startswith("shard_") and f.endswith(".pt")
    ])
    if not ctx_paths:
        raise RuntimeError(f"No shard_*.pt in {args.train_ctx_dir}")

    print(f"\n[stage02] Encoding {len(ctx_paths)} train context shards ...")
    all_vecs    = []
    all_row_ids = []
    t0          = time.time()

    for si, ctx_path in enumerate(ctx_paths):
        ds     = CtxShardDataset(ctx_path)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)
        shard_vecs = []

        for batch in loader:
            ids = batch["input_ids"].to(device)  # (B, ctx_len)
            with torch.no_grad():
                h_all = get_hs_small(backbone, ids, device)   # (B, T, d)
            h_ctx = h_all[:, -1, :].float().cpu().numpy()     # (B, d)
            shard_vecs.append(h_ctx)
            all_row_ids.append(batch["row_id"].numpy())

        shard_vecs = np.concatenate(shard_vecs, axis=0)
        all_vecs.append(shard_vecs)

        if si % 5 == 0 or si == len(ctx_paths) - 1:
            print(f"  shard {si:05d}/{len(ctx_paths)-1}  "
                  f"n={len(shard_vecs)}  t={time.time()-t0:.0f}s")

    all_vecs    = np.concatenate(all_vecs,    axis=0).astype(np.float32)  # (N, d)
    all_row_ids = np.concatenate(all_row_ids, axis=0).astype(np.int64)   # (N,)
    N, D = all_vecs.shape
    print(f"\n[stage02] Total vectors: {N:,}  d_model={D}")

    # ── Normalise ─────────────────────────────────────────────────────────────
    if args.normalize:
        all_vecs_normed = normalise_np(all_vecs)
    else:
        all_vecs_normed = all_vecs

    # ── Save raw vectors + row IDs ────────────────────────────────────────────
    np.save(os.path.join(args.output_dir, "vectors.npy"),       all_vecs_normed)
    np.save(os.path.join(args.output_dir, "train_row_ids.npy"), all_row_ids)
    print(f"  Saved vectors.npy  train_row_ids.npy")

    # ── Build FAISS index (if available) ──────────────────────────────────────
    index_type = "numpy"
    if args.use_faiss_if_available and _HAS_FAISS:
        print("[stage02] Building FAISS IndexFlatIP ...")
        index = _faiss.IndexFlatIP(D)
        index.add(all_vecs_normed)
        faiss_path = os.path.join(args.output_dir, "faiss.index")
        _faiss.write_index(index, faiss_path)
        index_type = "faiss_flatip"
        print(f"  FAISS index saved → {faiss_path}  ntotal={index.ntotal:,}")
    else:
        if args.use_faiss_if_available and not _HAS_FAISS:
            print("  (FAISS not available; using numpy index)")

    # ── Self-retrieval verification ────────────────────────────────────────────
    n_probe     = min(200, N)
    probe_idx   = np.random.RandomState(42).choice(N, n_probe, replace=False)
    probe_vecs  = all_vecs_normed[probe_idx]      # (n_probe, D)

    # Cosine similarities via matrix multiply
    probe_t  = torch.tensor(probe_vecs)
    all_t    = torch.tensor(all_vecs_normed)
    batch_sz = 64
    self_rank1_count = 0

    for i in range(0, n_probe, batch_sz):
        q = probe_t[i:i + batch_sz]
        sims = (q @ all_t.T)                      # (Bq, N)
        top1 = sims.argmax(dim=1).cpu().numpy()   # (Bq,)
        for j, (pi, top) in enumerate(zip(probe_idx[i:i + batch_sz], top1)):
            if int(top) == int(pi):
                self_rank1_count += 1

    self_rank1_rate = self_rank1_count / n_probe
    print(f"\n[stage02] Self-retrieval rank-1 rate: {self_rank1_rate:.4f}  "
          f"({self_rank1_count}/{n_probe})")

    index_pass = (self_rank1_rate > 0.99) and (N > 0)

    # ── Cross-check with train_cand_dir row count ─────────────────────────────
    cand_paths = sorted([
        os.path.join(args.train_cand_dir, f)
        for f in os.listdir(args.train_cand_dir)
        if f.startswith("shard_") and f.endswith(".pt")
    ])
    expected_N = 0
    for cp in cand_paths:
        cs = torch.load(cp, map_location="cpu", weights_only=True)
        expected_N += cs["gold_token"].shape[0]
    count_pass = (N == expected_N)
    if not count_pass:
        print(f"  WARNING: N={N:,} != expected {expected_N:,} (from cand shards)")

    index_pass = index_pass and count_pass

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "n_train_vectors":     N,
        "d_model":             D,
        "index_type":          index_type,
        "normalized":          args.normalize,
        "metric":              args.metric,
        "self_rank1_rate":     self_rank1_rate,
        "n_probe_vectors":     n_probe,
        "count_vs_cand":       count_pass,
        "index_pass":          index_pass,
        "faiss_available":     _HAS_FAISS,
    }
    rpath = os.path.join(args.output_dir, "index_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[stage02] index_pass = {index_pass}")
    print(f"[stage02] Report → {rpath}")

    if not index_pass:
        raise RuntimeError("Stage 02 index FAILED. See report.")
    print("[stage02] PASS")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--small_ckpt",           required=True)
    p.add_argument("--train_ctx_dir",        required=True)
    p.add_argument("--train_cand_dir",       required=True)
    p.add_argument("--output_dir",           required=True)
    p.add_argument("--ctx_len",              type=int,   default=256)
    p.add_argument("--batch_size",           type=int,   default=64)
    p.add_argument("--normalize",            action="store_true")
    p.add_argument("--metric",               default="cosine",
                   choices=["cosine", "dot"])
    p.add_argument("--use_faiss_if_available", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
