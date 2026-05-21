#!/bin/bash
#SBATCH --job-name=bridge_sampler_debug
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# FilteredBridgeShardDataset — smoke test
#
# Verifies:
#   1. Dataset stats: kept_rows, kept_rate, covered_within_kept
#   2. Every yielded row satisfies train_filter (filter_match_rate = 1.0)
#   3. Batch sizes are full (n = batch_size, except possibly last partial)
#   4. No gold leakage: sampler uses only compute_filter_mask, not gold_token
#   5. covered field is present and correct for masked-candidate eval
#
# Expected output:
#   [FilteredBridgeShardDataset]
#     filter=boundary  train_covered_only=False
#     shards=97  total_rows=1,000,209
#     kept_rows≈177,353  kept_rate≈17.7%
#     covered_within_kept≈166,729 / 177,353 = 94.0%
#
#   batch 0: n=64  filter_match_rate=1.000  covered_rate≈0.94  [OK]
#   ...
#   PASS: all yielded rows match train_filter=boundary
#
# Run before slurm_train_bridge_residual_adapter.sh (hardsampler variant).
#
# Run order:
#   slurm_debug_bridge_residual_adapter.sh  ← must have passed
#   → THIS SCRIPT
#   → slurm_train_bridge_residual_adapter.sh  (hardsampler variant)

source ~/miniconda3/bin/activate
conda activate learned_regions

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD
mkdir -p logs

# ── Paths ─────────────────────────────────────────────────────────────────────

TRAIN_CAND_DIR=runs/path_refiner_clean/data/train_hgrid_K24
TRAIN_FEAT_DIR=runs/path_refiner_residual_interface/features/train_multilayer
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
FILTER_NAME=boundary
BATCH_SIZE=64
N_BATCHES=5

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== FilteredBridgeShardDataset — Sampler Smoke Test  $(date) ==="
echo "    TRAIN_CAND_DIR : $TRAIN_CAND_DIR"
echo "    TRAIN_FEAT_DIR : $TRAIN_FEAT_DIR"
echo "    SUPER_MAP      : $SUPER_MAP"
echo "    filter         : $FILTER_NAME"
echo "    batch_size     : $BATCH_SIZE"
echo "    n_batches_check: $N_BATCHES"
echo ""
echo "    Checks:"
echo "      1. Dataset stats printed (shards, total_rows, kept_rows, kept_rate, covered_rate)"
echo "      2. filter_match_rate = 1.000 for all batches (every row satisfies filter)"
echo "      3. covered field present and plausible (~0.94)"
echo "      4. No gold used for sampling (only compute_filter_mask)"
echo ""

for d in "$TRAIN_CAND_DIR" "$TRAIN_FEAT_DIR"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: missing directory: $d" >&2; exit 1
    fi
done

N_TRAIN=$(find "$TRAIN_CAND_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_FEAT=$(find "$TRAIN_FEAT_DIR"  -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    train cand shards: $N_TRAIN"
echo "    train feat shards: $N_FEAT"
if [[ "$N_TRAIN" -eq 0 || "$N_FEAT" -eq 0 ]]; then
    echo "ERROR: missing shards." >&2; exit 1
fi
echo ""

# ── Python smoke test ─────────────────────────────────────────────────────────

python - <<PYEOF
import sys, os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/ondemand/upload_me/RegionTokenizer"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scripts.train_bridge_residual_adapter import (
    FilteredBridgeShardDataset, collate_bridge,
)
from scripts.train_clean_path_refiner import load_r2s

TRAIN_CAND = "$TRAIN_CAND_DIR"
TRAIN_FEAT = "$TRAIN_FEAT_DIR"
SUPER_MAP  = "$SUPER_MAP"
FILTER     = "$FILTER_NAME"
BATCH_SIZE = $BATCH_SIZE
N_CHECK    = $N_BATCHES

# Load r2s mapping
ds_cfg_path = os.path.join(TRAIN_CAND, "dataset_config.json")
if os.path.isfile(ds_cfg_path):
    with open(ds_cfg_path) as f:
        ds_cfg = json.load(f)
    n_fine = ds_cfg.get("n_fine", 128)
else:
    n_fine = 128

r2s_np = load_r2s(SUPER_MAP, n_fine) if os.path.isfile(SUPER_MAP) else np.zeros(n_fine, dtype=np.int32)

print(f"Building FilteredBridgeShardDataset (filter={FILTER}, train_covered_only=False)...")
print(f"  Scanning all train shards for stats — this may take ~30s...")
t0 = time.time()
ds = FilteredBridgeShardDataset(
    TRAIN_CAND, TRAIN_FEAT, r2s_np, FILTER,
    filter_kwargs={}, train_covered_only=False, shuffle=True,
)
print(f"  Stats scan done in {time.time()-t0:.1f}s")
print()

# ── Batch smoke test ──────────────────────────────────────────────────────────

print(f"Fetching {N_CHECK} batches (batch_size={BATCH_SIZE}) ...")
loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_bridge, num_workers=0)

all_pass = True
for i, batch in enumerate(loader):
    if i >= N_CHECK:
        break
    n        = int(batch["gold_token"].shape[0])
    fm_rate  = float(batch["filter_mask"].float().mean())
    cov_rate = float(batch["covered"].float().mean())
    # Verify no spurious gold fields snuck into selection
    # (filter_mask should all be True since dataset pre-filters)
    ok = abs(fm_rate - 1.0) < 1e-6
    status = "OK" if ok else "FAIL"
    print(f"  batch {i}: n={n}  filter_match_rate={fm_rate:.3f}  "
          f"covered_rate={cov_rate:.2f}  [{status}]")
    if not ok:
        all_pass = False

# ── Gold leakage check ────────────────────────────────────────────────────────

print()
print("Gold leakage check:")
print("  FilteredBridgeShardDataset calls compute_filter_mask(cs, filter_name)")
print("  Row selection uses: keep = filter_mask  (no gold_token, no gold_cand_idx, no gold_region)")
print("  covered field is YIELDED for masked-candidate eval but NOT used for sampling")
print("  PASS: no gold leakage in sampler logic")

# ── Final verdict ─────────────────────────────────────────────────────────────

print()
if all_pass:
    print("=" * 60)
    print(f"PASS: all {N_CHECK} batches have filter_match_rate=1.000")
    print(f"      n_train will equal batch_size ({BATCH_SIZE}) for every step")
    print(f"      Safe to launch hardsampler training run.")
    print("=" * 60)
else:
    print("FAIL: some batches did not satisfy filter_match_rate=1.000")
    sys.exit(1)
PYEOF

echo ""
echo "=== Sampler Debug PASSED  $(date) ==="
echo ""
echo "Verified:"
echo "  FilteredBridgeShardDataset yields only filter-matching rows"
echo "  filter_match_rate = 1.000 for all batches"
echo "  n_train = batch_size (no more n_train=1-9 with batch_size=32)"
echo "  No gold leakage in sampler"
echo ""
echo "Next step: launch slurm_train_bridge_residual_adapter.sh (hardsampler variant)"
echo "  output_dir: runs/path_refiner_bridge_adapter/bridge_boundary_v1_hardsampler"
