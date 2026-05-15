#!/bin/bash
#SBATCH --job-name=train_refiner
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Phase 3 only — train path-conditioned refiner variants A/B/C.
# Requires shard dataset already built by build_path_refiner_dataset.py.

source ~/miniconda3/bin/activate
conda activate learned_regions

set -eo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME="$SLURM_SUBMIT_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD
mkdir -p logs

# ── Configurable paths ────────────────────────────────────────────────────────

SMALL_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
DATASET_DIR=runs/path_refiner/dataset
OUTPUT_ROOT=runs/path_refiner
N_SUPER=24

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Train Path Refiner  $(date) ==="
echo "    SMALL_CKPT  : $SMALL_CKPT"
echo "    DATASET_DIR : $DATASET_DIR"
echo "    OUTPUT_ROOT : $OUTPUT_ROOT"

if [[ ! -f "$SMALL_CKPT" ]]; then
    echo "ERROR: checkpoint not found: $SMALL_CKPT" >&2; exit 1
fi

N_SHARDS=$(ls "$DATASET_DIR"/shard_*.npz 2>/dev/null | wc -l)
if [[ "$N_SHARDS" -eq 0 ]]; then
    echo "ERROR: no shard_*.npz files found in $DATASET_DIR" >&2
    echo "       Run build_path_refiner_dataset.py first." >&2
    exit 1
fi
echo "    shards found: $N_SHARDS"

mkdir -p "$OUTPUT_ROOT/variant_A" "$OUTPUT_ROOT/variant_B" "$OUTPUT_ROOT/variant_C"

# ── Variant A: bias-only ──────────────────────────────────────────────────────

echo ""
echo "=== Variant A (BiasOnly)  $(date) ==="

python scripts/train_path_refiner.py \
    --shard_dir   "$DATASET_DIR"           \
    --small_ckpt  "$SMALL_CKPT"            \
    --variant     A                        \
    --n_super     "$N_SUPER"               \
    --output_dir  "$OUTPUT_ROOT/variant_A" \
    --epochs      5                        \
    --batch_size  512                      \
    --lr          1e-3                     \
    --device      cuda

# ── Variant B: dot-product ────────────────────────────────────────────────────

echo ""
echo "=== Variant B (DotProduct)  $(date) ==="

python scripts/train_path_refiner.py \
    --shard_dir   "$DATASET_DIR"           \
    --small_ckpt  "$SMALL_CKPT"            \
    --variant     B                        \
    --n_super     "$N_SUPER"               \
    --d_head      64                       \
    --output_dir  "$OUTPUT_ROOT/variant_B" \
    --epochs      5                        \
    --batch_size  256                      \
    --lr          3e-4                     \
    --device      cuda

# ── Variant C: MLP ────────────────────────────────────────────────────────────

echo ""
echo "=== Variant C (MLP)  $(date) ==="

python scripts/train_path_refiner.py \
    --shard_dir   "$DATASET_DIR"           \
    --small_ckpt  "$SMALL_CKPT"            \
    --variant     C                        \
    --n_super     "$N_SUPER"               \
    --d_region    32                       \
    --d_hidden    128                      \
    --output_dir  "$OUTPUT_ROOT/variant_C" \
    --epochs      5                        \
    --batch_size  256                      \
    --lr          3e-4                     \
    --device      cuda

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo ""
echo "Checkpoints:"
echo "  $OUTPUT_ROOT/variant_A/best_refiner.pt"
echo "  $OUTPUT_ROOT/variant_B/best_refiner.pt"
echo "  $OUTPUT_ROOT/variant_C/best_refiner.pt"
echo ""

for var in A B C; do
    log="$OUTPUT_ROOT/variant_$var/train_log.csv"
    if [[ -f "$log" ]]; then
        echo "=== Variant $var — final eval ==="
        tail -1 "$log"
    fi
done
