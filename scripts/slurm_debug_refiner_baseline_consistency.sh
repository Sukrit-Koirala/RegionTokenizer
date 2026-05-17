#!/bin/bash
#SBATCH --job-name=debug_refiner_baseline
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Debug: verify force-zero baseline consistency across all refiner variants.
#
# Step 1 — Compute the official saved-candidate baseline (architecture-independent).
# Step 2 — Check that Variant C, D3-small, D3-base all reproduce it exactly.
#
# Expected output after the fix:
#   Official saved-candidate baseline: covered_nll=X  coverage=Y  fingerprint=Z
#   Variant C     : covered_nll=X  coverage=Y  fingerprint=Z  PASS
#   D3-small      : covered_nll=X  coverage=Y  fingerprint=Z  PASS
#   D3-base       : covered_nll=X  coverage=Y  fingerprint=Z  PASS
#
# If any variant prints FAIL, do not run slurm_clean_train_refiners.sh.

source ~/miniconda3/bin/activate
conda activate learned_regions

set -eo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME="$SLURM_SUBMIT_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD
mkdir -p logs

# ── Configurable paths ────────────────────────────────────────────────────────

SMALL_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
OUTPUT_ROOT=runs/path_refiner_clean

VAL_DIR=$OUTPUT_ROOT/data/val_hgrid_K24
BASELINE_DIR=$OUTPUT_ROOT/baselines
BASELINE_JSON=$BASELINE_DIR/saved_candidate_baseline.json

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Refiner Baseline Consistency Debug  $(date) ==="
echo "    VAL_DIR      : $VAL_DIR"
echo "    BASELINE_DIR : $BASELINE_DIR"

for f in "$SMALL_CKPT" "$SUPER_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
if [[ "$N_VAL" -eq 0 ]]; then
    echo "ERROR: no val shards in $VAL_DIR" >&2
    echo "       Run slurm_clean_build_datasets.sh first." >&2
    exit 1
fi
echo "    val shards: $N_VAL"

mkdir -p "$BASELINE_DIR" "$OUTPUT_ROOT/debug"

# ── Step 1: Official baseline ─────────────────────────────────────────────────

echo ""
echo "=== Step 1: Official saved-candidate baseline  $(date) ==="

python scripts/eval_saved_candidate_baseline.py \
    --val_dir        "$VAL_DIR"        \
    --small_ckpt     "$SMALL_CKPT"    \
    --output_dir     "$BASELINE_DIR"  \
    --eval_batch_size 64              \
    --device          cuda

echo ""
echo "    Baseline written to: $BASELINE_JSON"
if [[ -f "$BASELINE_JSON" ]]; then
    python -c "
import json
with open('$BASELINE_JSON') as f:
    b = json.load(f)
print(f\"    fingerprint  : {b['dataset_fingerprint']}\")
print(f\"    covered_nll  : {b['covered_nll']:.6f}\")
print(f\"    coverage     : {b['coverage']:.6f}\")
print(f\"    num_examples : {b['num_examples']:,}\")
"
fi

# ── Step 2: Variant C ─────────────────────────────────────────────────────────

echo ""
echo "=== Step 2a: Variant C force-zero  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --val_dir                 "$VAL_DIR"        \
    --small_ckpt              "$SMALL_CKPT"     \
    --super_map               "$SUPER_MAP"      \
    --variant                 C                 \
    --eval_only                                 \
    --force_zero_delta                          \
    --official_baseline       "$BASELINE_JSON"  \
    --fail_on_baseline_mismatch                 \
    --eval_batch_size         64                \
    --device                  cuda

# ── Step 3: D3-small ──────────────────────────────────────────────────────────

echo ""
echo "=== Step 2b: D3-small force-zero  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --val_dir                 "$VAL_DIR"        \
    --small_ckpt              "$SMALL_CKPT"     \
    --super_map               "$SUPER_MAP"      \
    --variant                 D3                \
    --d3_size                 small             \
    --eval_only                                 \
    --force_zero_delta                          \
    --official_baseline       "$BASELINE_JSON"  \
    --fail_on_baseline_mismatch                 \
    --eval_batch_size         64                \
    --device                  cuda

# ── Step 4: D3-base ───────────────────────────────────────────────────────────

echo ""
echo "=== Step 2c: D3-base force-zero  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --val_dir                 "$VAL_DIR"        \
    --small_ckpt              "$SMALL_CKPT"     \
    --super_map               "$SUPER_MAP"      \
    --variant                 D3                \
    --d3_size                 base              \
    --eval_only                                 \
    --force_zero_delta                          \
    --official_baseline       "$BASELINE_JSON"  \
    --fail_on_baseline_mismatch                 \
    --eval_batch_size         64                \
    --device                  cuda

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo ""
echo "If all three steps above printed PASS, the baseline is consistent."
echo "You can now run: sbatch scripts/slurm_clean_train_refiners.sh"
echo ""
echo "Baseline files:"
echo "  $BASELINE_DIR/saved_candidate_baseline.json"
echo "  $BASELINE_DIR/saved_candidate_baseline.md"
echo "  $BASELINE_DIR/saved_candidate_baseline_by_split.csv"
