#!/bin/bash
#SBATCH --job-name=debug_periodic_eval
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Verify that the canonical evaluator produces identical fingerprint/coverage/counts
# for all three refiner variants, at init and after one training step.
#
# Tests:
#   1. force_zero full-val  — must match official baseline exactly (NLL + fp + counts)
#   2. with_delta full-val at init — must equal force_zero (identity check)
#   3. after 1 training step, periodic eval — fingerprint/coverage/counts must remain
#      identical across C / D3-small / D3-base
#
# Expected results for all variants:
#   coverage     = 0.948425
#   fingerprint  = f57cabcdc46d69ce
#   num_examples = 239,362
#   num_covered  = 227,017
#
# Run order:
#   slurm_clean_build_datasets.sh
#   → slurm_debug_refiner_baseline_consistency.sh
#   → slurm_debug_refiner_init_identity.sh
#   → this script
#   → slurm_clean_train_refiners.sh

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
TRAIN_DIR=$OUTPUT_ROOT/data/train_hgrid_K24
BASELINE_JSON=$OUTPUT_ROOT/baselines/saved_candidate_baseline.json
DEBUG_DIR=$OUTPUT_ROOT/debug/periodic_eval_consistency

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Periodic Eval Consistency Debug  $(date) ==="
echo "    VAL_DIR      : $VAL_DIR"
echo "    BASELINE_JSON: $BASELINE_JSON"

for f in "$SMALL_CKPT" "$SUPER_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done
if [[ ! -f "$BASELINE_JSON" ]]; then
    echo "ERROR: $BASELINE_JSON missing — run slurm_debug_refiner_baseline_consistency.sh first." >&2
    exit 1
fi

N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    val shards   : $N_VAL"
echo "    train shards : $N_TRAIN"

if [[ "$N_VAL" -eq 0 || "$N_TRAIN" -eq 0 ]]; then
    echo "ERROR: missing shards. Run slurm_clean_build_datasets.sh first." >&2; exit 1
fi

mkdir -p "$DEBUG_DIR"

# ── Shared args ───────────────────────────────────────────────────────────────

BASELINE_ARG="--official_baseline $BASELINE_JSON --fail_on_baseline_mismatch"

# ── Test 1: eval_only force_zero + check_init_identity, all variants ──────────
#
# Each call must print:
#   [init identity check] PASS
#   Force-zero result: PASS  (NLL + fp + counts match baseline)

echo ""
echo "=== Test 1: force_zero + init identity, Variant C  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --val_dir             "$VAL_DIR"     \
    --small_ckpt          "$SMALL_CKPT" \
    --super_map           "$SUPER_MAP"  \
    --variant             C             \
    --d_region            32            \
    --d_hidden            256           \
    --eval_only                         \
    --force_zero_delta                  \
    --check_init_identity               \
    --print_delta_stats                 \
    $BASELINE_ARG                       \
    --device              cuda

echo ""
echo "=== Test 1: force_zero + init identity, D3-small  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --val_dir             "$VAL_DIR"     \
    --small_ckpt          "$SMALL_CKPT" \
    --super_map           "$SUPER_MAP"  \
    --variant             D3            \
    --d3_size             small         \
    --eval_only                         \
    --force_zero_delta                  \
    --check_init_identity               \
    --print_delta_stats                 \
    $BASELINE_ARG                       \
    --device              cuda

echo ""
echo "=== Test 1: force_zero + init identity, D3-base  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --val_dir             "$VAL_DIR"     \
    --small_ckpt          "$SMALL_CKPT" \
    --super_map           "$SUPER_MAP"  \
    --variant             D3            \
    --d3_size             base          \
    --eval_only                         \
    --force_zero_delta                  \
    --check_init_identity               \
    --print_delta_stats                 \
    $BASELINE_ARG                       \
    --device              cuda

# ── Test 2: 1 training step + periodic eval, all variants ─────────────────────
#
# steps=1  eval_every=1  → runs one step then immediately runs periodic eval.
# All variants must report identical fingerprint/coverage/counts.
# NLL may differ slightly (one gradient step) but must be non-NaN.

echo ""
echo "=== Test 2: 1 step + periodic eval, Variant C  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --train_dir         "$TRAIN_DIR"                          \
    --val_dir           "$VAL_DIR"                            \
    --small_ckpt        "$SMALL_CKPT"                         \
    --super_map         "$SUPER_MAP"                          \
    --variant           C                                     \
    --d_region          32                                    \
    --d_hidden          256                                   \
    --output_dir        "$DEBUG_DIR/variant_C"                \
    --steps             1                                     \
    --eval_every        1                                     \
    --batch_size        64                                    \
    --eval_batch_size   64                                    \
    --lr                3e-4                                  \
    --eval_before_train                                       \
    $BASELINE_ARG                                             \
    --device            cuda

echo ""
echo "=== Test 2: 1 step + periodic eval, D3-small  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --train_dir         "$TRAIN_DIR"                          \
    --val_dir           "$VAL_DIR"                            \
    --small_ckpt        "$SMALL_CKPT"                         \
    --super_map         "$SUPER_MAP"                          \
    --variant           D3                                    \
    --d3_size           small                                 \
    --output_dir        "$DEBUG_DIR/d3_small"                 \
    --steps             1                                     \
    --eval_every        1                                     \
    --batch_size        32                                    \
    --eval_batch_size   64                                    \
    --lr                3e-4                                  \
    --eval_before_train                                       \
    $BASELINE_ARG                                             \
    --device            cuda

echo ""
echo "=== Test 2: 1 step + periodic eval, D3-base  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --train_dir         "$TRAIN_DIR"                          \
    --val_dir           "$VAL_DIR"                            \
    --small_ckpt        "$SMALL_CKPT"                         \
    --super_map         "$SUPER_MAP"                          \
    --variant           D3                                    \
    --d3_size           base                                  \
    --output_dir        "$DEBUG_DIR/d3_base"                  \
    --steps             1                                     \
    --eval_every        1                                     \
    --batch_size        16                                    \
    --eval_batch_size   64                                    \
    --lr                1e-4                                  \
    --eval_before_train                                       \
    $BASELINE_ARG                                             \
    --device            cuda

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo ""
echo "Check that all variants above printed:"
echo "  [init identity check] PASS"
echo "  Force-zero result: PASS"
echo "  [step 0] identity PASS"
echo "  [eval/canonical_full_val] coverage = 0.948425  fingerprint = f57cabcdc46d69ce"
echo ""
echo "If all pass, run: sbatch scripts/slurm_clean_train_refiners.sh"
