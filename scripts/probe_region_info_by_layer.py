#!/usr/bin/env python3
"""
Layer-wise region-information probe diagnostic.

For each backbone layer, trains simple linear and MLP probes to predict:
  1. gold_fine_region  (128-class)
  2. gold_superregion  (24-class, from r2s mapping)
  3. is_boundary       (split >= 2, binary)
  4. router_top8_miss  (gold_region not in router top-8, binary)

Reports acc@1, acc@4, acc@8 (for classification) and cross-entropy per
layer and probe type. Answers:
  Q1. Which layer best predicts gold region?
  Q2. Does mid-layer information exceed final h_prime?
  Q3. Are boundary/router-miss positions better represented in earlier layers?

Usage:
    python scripts/probe_region_info_by_layer.py \\
        --train_candidate_dir runs/path_refiner_clean/data/train_hgrid_K24 \\
        --val_candidate_dir   runs/path_refiner_clean/data/val_hgrid_K24 \\
        --train_features_dir  runs/path_refiner_residual_interface/features/train_multilayer \\
        --val_features_dir    runs/path_refiner_residual_interface/features/val_multilayer \\
        --super_map           runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \\
        --output_dir          runs/path_refiner_residual_interface/probes \\
        --steps               2000 \\
        --device              cuda
"""

import argparse
import csv
import glob
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

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from scripts.train_clean_path_refiner import load_r2s


# ── Probe architectures ───────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_class: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, n_class))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, n_class: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data loading ──────────────────────────────────────────────────────────────

def iter_paired_shards(
    cand_dir: str,
    feat_dir: str,
) -> Iterator[Tuple[Dict, Dict]]:
    """Yields (cand_shard, feat_shard) pairs in sorted shard order."""
    cand_paths = sorted(glob.glob(os.path.join(cand_dir, "shard_*.pt")))
    feat_paths = sorted(glob.glob(os.path.join(feat_dir, "shard_*.pt")))
    if len(cand_paths) != len(feat_paths):
        raise RuntimeError(
            f"Shard count mismatch: cand={len(cand_paths)} feat={len(feat_paths)}"
        )
    for cp, fp in zip(cand_paths, feat_paths):
        cs = torch.load(cp, map_location="cpu", weights_only=True)
        fs = torch.load(fp, map_location="cpu", weights_only=True)
        if cs["gold_token"].shape[0] != fs["gold_token"].shape[0]:
            raise RuntimeError(
                f"Row count mismatch: {cp} ({cs['gold_token'].shape[0]}) "
                f"vs {fp} ({fs['gold_token'].shape[0]})"
            )
        if not torch.equal(cs["gold_token"].long(), fs["gold_token"].long()):
            raise RuntimeError(
                f"gold_token mismatch between {cp} and {fp} — alignment broken"
            )
        yield cs, fs


def load_features_for_layer(
    cand_dir: str,
    feat_dir: str,
    layer_pos: int,           # index into h_layers dim 1
    r2s_np: np.ndarray,
    max_rows: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Loads all shard pairs and returns a dict of tensors for one layer:
      h:                (N, d_model) float32
      fine_region:      (N,) int64
      superregion:      (N,) int64
      is_boundary:      (N,) int64  (0 or 1)
      router_top8_miss: (N,) int64  (0 or 1)
    """
    h_list, fine_list, sup_list, bnd_list, miss_list = [], [], [], [], []
    total = 0

    for cs, fs in iter_paired_shards(cand_dir, feat_dir):
        N = cs["gold_token"].shape[0]
        if max_rows is not None and total + N > max_rows:
            N = max_rows - total

        h_layer = fs["h_layers"][:N, layer_pos].float()  # (N, D)
        fine_reg = cs["gold_region"][:N].long()           # (N,) fine region
        split    = cs["split"][:N].long()                 # (N,)

        # Superregion
        fine_np  = fine_reg.numpy().clip(min=0)
        sup_reg  = torch.from_numpy(r2s_np[fine_np].astype(np.int64))
        sup_reg[fine_reg < 0] = -1

        # is_boundary: split code >= 2 (boundary or tight_boundary)
        is_bnd = (split >= 2).long()

        # router_top8_miss: gold fine region not in top-8 router regions
        r_reg  = cs["router_topk_reg"][:N].long()  # (N, K)
        top8   = r_reg[:, :8]                       # (N, 8)
        miss   = ~((top8 == fine_reg.unsqueeze(1)) | (fine_reg.unsqueeze(1) < 0)).any(dim=1)
        miss   = miss.long()
        # For uncovered positions (fine_reg < 0), router_miss is undefined → set 0
        miss[fine_reg < 0] = 0

        h_list.append(h_layer)
        fine_list.append(fine_reg)
        sup_list.append(sup_reg)
        bnd_list.append(is_bnd)
        miss_list.append(miss)

        total += N
        if max_rows is not None and total >= max_rows:
            break

    return {
        "h":                torch.cat(h_list),
        "fine_region":      torch.cat(fine_list),
        "superregion":      torch.cat(sup_list),
        "is_boundary":      torch.cat(bnd_list),
        "router_top8_miss": torch.cat(miss_list),
    }


# ── Probe training ────────────────────────────────────────────────────────────

def train_probe(
    model: nn.Module,
    train_h: torch.Tensor,        # (N_train, D)
    train_y: torch.Tensor,        # (N_train,)
    steps: int,
    batch_size: int,
    lr: float,
    device,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N   = train_h.shape[0]

    for step in range(steps):
        idx  = torch.randint(0, N, (batch_size,))
        h    = train_h[idx].to(device)
        y    = train_y[idx].to(device)
        loss = F.cross_entropy(model(h), y)
        loss.backward()
        opt.step()
        opt.zero_grad()


@torch.no_grad()
def eval_probe(
    model: nn.Module,
    h: torch.Tensor,              # (N, D)
    y: torch.Tensor,              # (N,)
    n_class: int,
    device,
    batch_size: int = 2048,
) -> Dict[str, float]:
    model.eval()
    ce_sum = 0.0
    a1 = a4 = a8 = 0
    N   = h.shape[0]
    K   = min(8, n_class)

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        h_b   = h[start:end].to(device)
        y_b   = y[start:end].to(device)
        logit = model(h_b)
        ce_sum += float(F.cross_entropy(logit, y_b, reduction="sum"))
        preds  = logit.topk(K, dim=-1).indices
        a1 += int((preds[:, :1] == y_b.unsqueeze(1)).any(1).sum())
        a4 += int((preds[:, :min(4, K)] == y_b.unsqueeze(1)).any(1).sum())
        a8 += int((preds[:, :K] == y_b.unsqueeze(1)).any(1).sum())

    return {
        "ce":    ce_sum / N,
        "acc@1": a1 / N,
        "acc@4": a4 / N,
        "acc@8": a8 / N,
        "n":     N,
    }


# ── Binary classification probe ───────────────────────────────────────────────

@torch.no_grad()
def eval_binary_probe(
    model: nn.Module,
    h: torch.Tensor,
    y: torch.Tensor,
    device,
    batch_size: int = 2048,
) -> Dict[str, float]:
    model.eval()
    ce_sum = 0.0
    correct = 0
    N = h.shape[0]

    for start in range(0, N, batch_size):
        end    = min(start + batch_size, N)
        h_b    = h[start:end].to(device)
        y_b    = y[start:end].to(device)
        logit  = model(h_b)
        ce_sum += float(F.cross_entropy(logit, y_b, reduction="sum"))
        correct += int((logit.argmax(dim=-1) == y_b).sum())

    return {"ce": ce_sum / N, "acc": correct / N, "n": N}


# ── Main probe runner ─────────────────────────────────────────────────────────

def run_all_probes(args, layer_ids: List[int], d_model: int, r2s_np: np.ndarray) -> List[Dict]:
    device = torch.device(args.device)
    n_fine  = int(r2s_np.shape[0])
    n_super = int(r2s_np.max()) + 1 if r2s_np.max() >= 0 else 24

    tasks = [
        ("fine_region",     n_fine,  "multiclass"),
        ("superregion",     n_super, "multiclass"),
        ("is_boundary",     2,       "binary"),
        ("router_top8_miss",2,       "binary"),
    ]
    probe_types = [("linear", LinearProbe), ("mlp", MLPProbe)]

    all_rows = []

    for layer_pos, lid in enumerate(layer_ids):
        layer_tag = f"h_final" if lid == -1 else f"block_{lid}"
        print(f"\n[probe] Layer {layer_tag} ({layer_pos+1}/{len(layer_ids)}) ...")

        print("  Loading train features ...")
        tr = load_features_for_layer(
            args.train_candidate_dir, args.train_features_dir,
            layer_pos, r2s_np, max_rows=args.max_train_rows,
        )
        print("  Loading val features ...")
        va = load_features_for_layer(
            args.val_candidate_dir, args.val_features_dir,
            layer_pos, r2s_np, max_rows=None,
        )

        tr_h = tr["h"]
        va_h = va["h"]
        print(f"  train={tr_h.shape[0]:,}  val={va_h.shape[0]:,}  d={tr_h.shape[1]}")

        for task_name, n_class, task_type in tasks:
            tr_y = tr[task_name]
            va_y = va[task_name]

            # Only evaluate on valid labels (fine_region < 0 = uncovered)
            if task_name in ("fine_region", "superregion"):
                tr_valid = tr_y >= 0
                va_valid = va_y >= 0
                tr_h_t, tr_y_t = tr_h[tr_valid], tr_y[tr_valid]
                va_h_t, va_y_t = va_h[va_valid], va_y[va_valid]
            else:
                tr_h_t, tr_y_t = tr_h, tr_y
                va_h_t, va_y_t = va_h, va_y

            if tr_h_t.shape[0] == 0 or va_h_t.shape[0] == 0:
                print(f"  [{task_name}] skipped (no valid labels)")
                continue

            for probe_name, ProbeClass in probe_types:
                model = ProbeClass(d_model, n_class).to(device)
                t0    = time.time()
                train_probe(model, tr_h_t, tr_y_t, args.steps,
                            args.batch_size, args.lr, device)
                elapsed = time.time() - t0

                if task_type == "binary":
                    m = eval_binary_probe(model, va_h_t, va_y_t, device)
                    row = {
                        "layer": layer_tag, "layer_pos": layer_pos, "layer_id": lid,
                        "task": task_name, "probe": probe_name,
                        "n_class": n_class, "n_val": m["n"],
                        "val_ce": f"{m['ce']:.4f}", "val_acc1": f"{m['acc']:.4f}",
                        "val_acc4": "", "val_acc8": "", "train_s": f"{elapsed:.1f}",
                    }
                else:
                    m = eval_probe(model, va_h_t, va_y_t, n_class, device)
                    row = {
                        "layer": layer_tag, "layer_pos": layer_pos, "layer_id": lid,
                        "task": task_name, "probe": probe_name,
                        "n_class": n_class, "n_val": m["n"],
                        "val_ce": f"{m['ce']:.4f}", "val_acc1": f"{m['acc@1']:.4f}",
                        "val_acc4": f"{m['acc@4']:.4f}", "val_acc8": f"{m['acc@8']:.4f}",
                        "train_s": f"{elapsed:.1f}",
                    }
                all_rows.append(row)
                print(f"  [{task_name}][{probe_name}]  "
                      f"ce={row['val_ce']}  acc@1={row['val_acc1']}  "
                      f"acc@4={row['val_acc4']}  t={elapsed:.0f}s")

    return all_rows


# ── Report generation ─────────────────────────────────────────────────────────

def write_report(rows: List[Dict], layer_ids: List[int], output_dir: str) -> None:
    """Writes layer_probe_results.csv and layer_probe_report.md."""
    csv_path = os.path.join(output_dir, "layer_probe_results.csv")
    fields   = ["layer", "layer_pos", "layer_id", "task", "probe",
                "n_class", "n_val", "val_ce", "val_acc1", "val_acc4", "val_acc8", "train_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[report] CSV written: {csv_path}")

    # Build report
    lines = [
        "# Layer-wise Region Information Probe Report",
        "",
        "## Summary by layer (linear probe, fine_region acc@1)",
        "",
        "| Layer | fine_acc@1 | super_acc@1 | boundary_acc | router_miss_acc |",
        "|-------|-----------|------------|-------------|----------------|",
    ]
    by_layer: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_layer[r["layer"]][f"{r['task']}/{r['probe']}"] = r

    # Linear probe summary
    for lid in layer_ids:
        ltag = "h_final" if lid == -1 else f"block_{lid}"
        fine_acc  = by_layer[ltag].get("fine_region/linear",     {}).get("val_acc1", "—")
        super_acc = by_layer[ltag].get("superregion/linear",     {}).get("val_acc1", "—")
        bnd_acc   = by_layer[ltag].get("is_boundary/linear",     {}).get("val_acc1", "—")
        miss_acc  = by_layer[ltag].get("router_top8_miss/linear",{}).get("val_acc1", "—")
        lines.append(f"| {ltag:20s} | {fine_acc:>10} | {super_acc:>10} | {bnd_acc:>11} | {miss_acc:>15} |")

    lines += [
        "",
        "## Diagnostic Questions",
        "",
        "### Q1. Which layer best predicts gold fine region?",
        "",
    ]
    best_row = max(
        (r for r in rows if r["task"] == "fine_region" and r["probe"] == "linear"),
        key=lambda r: float(r["val_acc1"] or 0),
        default=None,
    )
    if best_row:
        lines.append(f"Best: **{best_row['layer']}** linear probe  "
                     f"acc@1={best_row['val_acc1']}  acc@4={best_row['val_acc4']}")

    lines += [
        "",
        "### Q2. Does region information appear before final h_prime?",
        "",
        "See acc@1 column above. If mid-layer acc@1 > h_final acc@1, then YES.",
        "",
        "### Q3. Does final h_prime lose/blur region information?",
        "",
        "Compare h_final row vs best mid-layer row.",
        "",
        "### Q4. Are boundary/router-miss positions better represented in mid layers?",
        "",
        "See is_boundary/router_top8_miss rows — compare by layer.",
        "",
        "## Raw Results (all tasks, all probe types)",
        "",
        "See `layer_probe_results.csv`.",
    ]

    md_path = os.path.join(output_dir, "layer_probe_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] Markdown written: {md_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # Load feature config to discover layer_ids
    cfg_path = os.path.join(args.val_features_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"config.json not found in {args.val_features_dir}. "
                           "Run build_multilayer_residual_features.py first.")
    with open(cfg_path) as f:
        feat_cfg = json.load(f)
    layer_ids = feat_cfg["layer_ids"]
    d_model   = feat_cfg["d_model"]
    print(f"[main] layer_ids={layer_ids}  d_model={d_model}")

    # Load superregion map
    print(f"[main] Loading super_map: {args.super_map}")
    n_fine = 128
    with open(args.super_map) as f:
        r2s_dict = json.load(f)
    r2s_np = np.full(n_fine, -1, dtype=np.int32)
    for k, v in r2s_dict.items():
        fid = int(k)
        if 0 <= fid < n_fine:
            r2s_np[fid] = int(v)

    rows = run_all_probes(args, layer_ids, d_model, r2s_np)
    write_report(rows, layer_ids, args.output_dir)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_candidate_dir", required=True)
    p.add_argument("--val_candidate_dir",   required=True)
    p.add_argument("--train_features_dir",  required=True)
    p.add_argument("--val_features_dir",    required=True)
    p.add_argument("--super_map",           required=True)
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--steps",       type=int,   default=2000)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--max_train_rows", type=int, default=None,
                   help="Cap train rows per layer (for speed). Default: all.")
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
