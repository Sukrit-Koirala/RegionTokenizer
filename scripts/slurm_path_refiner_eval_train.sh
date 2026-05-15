#!/bin/bash
#SBATCH --job-name=path_refiner
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Path-conditioned refiner — Phases 1, 2, 3 (A/B/C).
#
# Phase 1: Masked-softmax baseline eval for RouterTopK / Union policies.
# Phase 2: Build offline shard dataset (h_prime + candidate tokens).
# Phase 3: Train all three refiner variants (A=bias-only, B=dot-product, C=MLP).
#
# Prerequisites:
#   per_position.npz must exist in KNN_RUN_DIR.
#   Checkpoint must be a ReprRegionRetrievalLM (has retrieval_proj).

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
KNN_RUN_DIR=runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20
REGION_MAP=runs/region_maps_128/token_to_region.json
HIER_DIR=runs/hard_memory_predictive_hierarchy
OUTPUT_ROOT=runs/path_refiner

DATASET_DIR=$OUTPUT_ROOT/dataset
PHASE1_DIR=$OUTPUT_ROOT/phase1

# Super-region mapping for variants B/C (K=24 predictive basins)
SUPER_MAP=$HIER_DIR/region_to_superregion_K24.json
N_SUPER=24

# Candidate policy for dataset build
BUILD_POLICY=router_top16

# ── Phase toggles (set to 0 to skip) ─────────────────────────────────────────
RUN_PHASE1=1   # masked-softmax baseline eval
RUN_PHASE2=1   # build shard dataset
RUN_PHASE3=1   # train variants A/B/C

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Path-conditioned Refiner  $(date) ==="
echo "    SMALL_CKPT   : $SMALL_CKPT"
echo "    KNN_RUN_DIR  : $KNN_RUN_DIR"
echo "    BUILD_POLICY : $BUILD_POLICY"
echo "    OUTPUT_ROOT  : $OUTPUT_ROOT"

for f in "$SMALL_CKPT" "$REGION_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

if [[ ! -f "$KNN_RUN_DIR/per_position.npz" ]] && \
   [[ ! -f "$KNN_RUN_DIR/per_position_topk.npz" ]]; then
    echo "ERROR: per_position.npz not found in $KNN_RUN_DIR" >&2; exit 1
fi

mkdir -p "$PHASE1_DIR" "$DATASET_DIR"

# ── Phase 1: masked-softmax baseline eval ────────────────────────────────────

if [[ "$RUN_PHASE1" == "1" ]]; then
    echo ""
    echo "=== Phase 1: masked-softmax eval  $(date) ==="

    python scripts/eval_path_refiner_masked_softmax.py \
        --small_ckpt    "$SMALL_CKPT"   \
        --knn_run_dir   "$KNN_RUN_DIR"  \
        --region_map    "$REGION_MAP"   \
        --output_dir    "$PHASE1_DIR"   \
        --max_positions 300000          \
        --device        cuda

    echo "Phase 1 outputs:"
    echo "  $PHASE1_DIR/phase1_policy_summary.csv"
    echo "  $PHASE1_DIR/phase1_policy_detail.csv"
else
    echo "=== Phase 1: skipped (RUN_PHASE1=0) ==="
fi

# ── Phase 2: build shard dataset ─────────────────────────────────────────────

if [[ "$RUN_PHASE2" == "1" ]]; then
    echo ""
    echo "=== Phase 2: build dataset  $(date) ==="

    SUPER_MAP_ARG=""
    if [[ -f "$SUPER_MAP" ]]; then
        SUPER_MAP_ARG="--super_map $SUPER_MAP --n_super $N_SUPER"
    fi

    python scripts/build_path_refiner_dataset.py \
        --small_ckpt    "$SMALL_CKPT"   \
        --knn_run_dir   "$KNN_RUN_DIR"  \
        --region_map    "$REGION_MAP"   \
        --output_dir    "$DATASET_DIR"  \
        --policy        "$BUILD_POLICY" \
        $SUPER_MAP_ARG                  \
        --max_positions 500000          \
        --shard_size    10000           \
        --device        cuda

    echo "Dataset built: $DATASET_DIR"
    echo "  $(ls $DATASET_DIR/shard_*.npz 2>/dev/null | wc -l) shards"
else
    echo "=== Phase 2: skipped (RUN_PHASE2=0) ==="
    echo "  using existing dataset: $DATASET_DIR"
    echo "  $(ls $DATASET_DIR/shard_*.npz 2>/dev/null | wc -l) shards found"
fi

if [[ "$RUN_PHASE3" == "1" ]]; then

# ── Phase 3A: train Variant A (bias-only) ────────────────────────────────────

    echo ""
    echo "=== Phase 3A: Variant A (BiasOnly)  $(date) ==="

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

# ── Phase 3B: train Variant B (dot-product) ──────────────────────────────────

    echo ""
    echo "=== Phase 3B: Variant B (DotProduct)  $(date) ==="

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

# ── Phase 3C: train Variant C (MLP) ──────────────────────────────────────────

    echo ""
    echo "=== Phase 3C: Variant C (MLP)  $(date) ==="

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

else
    echo "=== Phase 3: skipped (RUN_PHASE3=0) ==="
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo ""
echo "Outputs:"
echo "  Phase 1 eval  : $PHASE1_DIR/"
echo "  Dataset        : $DATASET_DIR/   ($(ls $DATASET_DIR/shard_*.npz 2>/dev/null | wc -l) shards)"
echo "  Variant A ckpt : $OUTPUT_ROOT/variant_A/best_refiner.pt"
echo "  Variant B ckpt : $OUTPUT_ROOT/variant_B/best_refiner.pt"
echo "  Variant C ckpt : $OUTPUT_ROOT/variant_C/best_refiner.pt"
echo ""

for var in A B C; do
    log="$OUTPUT_ROOT/variant_$var/train_log.csv"
    if [[ -f "$log" ]]; then
        echo "=== Variant $var — final eval ==="
        tail -1 "$log"
    fi
done
