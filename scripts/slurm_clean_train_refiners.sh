#!/bin/bash
#SBATCH --job-name=clean_train_refiners
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Train all refiner variants over pre-built shard datasets:
#   - Variant C  (RicherMLP,  d_region=32,  d_hidden=256)
#   - D3-small   (RegionTransformerRefiner, d_region=128, L=1, H=4)
#   - D3-base    (RegionTransformerRefiner, d_region=256, L=2, H=4)
#
# Each run uses --eval_before_train, which prints a step-0 sanity check:
#   force_zero_delta NLL should match masked-softmax covered_nll (~3.38).
#
# Run order: slurm_clean_build_datasets.sh → slurm_clean_eval_baseline.sh → this job

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

POLICY=hgrid_K24_srccombined_nts1_bkr12_fb24_cm0.15_en1.25_corec60m30
TRAIN_DIR=$OUTPUT_ROOT/data/train_hgrid_K24
VAL_DIR=$OUTPUT_ROOT/data/val_hgrid_K24
EVAL_DIR=$OUTPUT_ROOT/eval
BASELINE_JSON=$OUTPUT_ROOT/baselines/saved_candidate_baseline.json

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Train Refiners  $(date) ==="
echo "    SMALL_CKPT  : $SMALL_CKPT"
echo "    TRAIN_DIR   : $TRAIN_DIR"
echo "    VAL_DIR     : $VAL_DIR"
echo "    OUTPUT_ROOT : $OUTPUT_ROOT"

for f in "$SMALL_CKPT" "$SUPER_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

N_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_VAL=$(find "$VAL_DIR"   -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    train shards : $N_TRAIN"
echo "    val   shards : $N_VAL"

if [[ ! -f "$BASELINE_JSON" ]]; then
    echo ""
    echo "WARNING: official baseline not found at $BASELINE_JSON"
    echo "         Run slurm_debug_refiner_baseline_consistency.sh first to generate it."
    echo "         Continuing without baseline consistency enforcement."
fi

BASELINE_ARG=""
if [[ -f "$BASELINE_JSON" ]]; then
    BASELINE_ARG="--official_baseline $BASELINE_JSON --fail_on_baseline_mismatch"
    echo "    baseline     : $BASELINE_JSON (will enforce consistency)"
fi

if [[ "$N_TRAIN" -eq 0 ]]; then
    echo "ERROR: no train shards in $TRAIN_DIR" >&2
    echo "       Run slurm_clean_build_datasets.sh first." >&2
    exit 1
fi
if [[ "$N_VAL" -eq 0 ]]; then
    echo "ERROR: no val shards in $VAL_DIR" >&2
    echo "       Run slurm_clean_build_datasets.sh first." >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/variant_C" "$OUTPUT_ROOT/d3_small" "$OUTPUT_ROOT/d3_base"

# ── Variant C (RicherMLP) ─────────────────────────────────────────────────────

echo ""
echo "=== Variant C (RicherMLP)  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --train_dir         "$TRAIN_DIR"              \
    --val_dir           "$VAL_DIR"                \
    --small_ckpt        "$SMALL_CKPT"             \
    --super_map         "$SUPER_MAP"              \
    --variant           C                         \
    --d_region          32                        \
    --d_hidden          256                       \
    --output_dir        "$OUTPUT_ROOT/variant_C"  \
    --steps             20000                     \
    --eval_every        1000                      \
    --batch_size        64                        \
    --eval_batch_size   64                        \
    --lr                3e-4                      \
    --eval_before_train                           \
    $BASELINE_ARG                                 \
    --device            cuda

echo "    Variant C done: $OUTPUT_ROOT/variant_C/best_refiner.pt"

# ── D3-small ──────────────────────────────────────────────────────────────────

echo ""
echo "=== D3-small  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --train_dir         "$TRAIN_DIR"              \
    --val_dir           "$VAL_DIR"                \
    --small_ckpt        "$SMALL_CKPT"             \
    --super_map         "$SUPER_MAP"              \
    --variant           D3                        \
    --d3_size           small                     \
    --output_dir        "$OUTPUT_ROOT/d3_small"   \
    --steps             20000                     \
    --eval_every        1000                      \
    --batch_size        32                        \
    --eval_batch_size   64                        \
    --lr                3e-4                      \
    --eval_before_train                           \
    $BASELINE_ARG                                 \
    --device            cuda

echo "    D3-small done: $OUTPUT_ROOT/d3_small/best_refiner.pt"

# ── D3-base ───────────────────────────────────────────────────────────────────

echo ""
echo "=== D3-base  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --train_dir         "$TRAIN_DIR"              \
    --val_dir           "$VAL_DIR"                \
    --small_ckpt        "$SMALL_CKPT"             \
    --super_map         "$SUPER_MAP"              \
    --variant           D3                        \
    --d3_size           base                      \
    --output_dir        "$OUTPUT_ROOT/d3_base"    \
    --steps             20000                     \
    --eval_every        1000                      \
    --batch_size        16                        \
    --eval_batch_size   64                        \
    --lr                1e-4                      \
    --eval_before_train                           \
    $BASELINE_ARG                                 \
    --device            cuda

echo "    D3-base done: $OUTPUT_ROOT/d3_base/best_refiner.pt"

# ── Final summary ─────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo ""
echo "Checkpoints:"
echo "  Variant C : $OUTPUT_ROOT/variant_C/best_refiner.pt"
echo "  D3-small  : $OUTPUT_ROOT/d3_small/best_refiner.pt"
echo "  D3-base   : $OUTPUT_ROOT/d3_base/best_refiner.pt"
echo ""

echo "=== Training summaries (final eval row) ==="
for run_dir in "$OUTPUT_ROOT/variant_C" "$OUTPUT_ROOT/d3_small" "$OUTPUT_ROOT/d3_base"; do
    log="$run_dir/train_log.csv"
    name=$(basename "$run_dir")
    if [[ -f "$log" ]]; then
        echo "--- $name ---"
        head -1 "$log"
        tail -1 "$log"
    else
        echo "--- $name: no log ---"
    fi
done

echo ""
echo "=== Baseline for comparison ==="
if [[ -f "$EVAL_DIR/masked_softmax_eval.csv" ]]; then
    echo "Masked-softmax ($POLICY):"
    grep "$POLICY" "$EVAL_DIR/masked_softmax_eval.csv" || echo "(policy not found in CSV)"
fi

