#!/bin/bash
#SBATCH --job-name=clean_build_datasets
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
# Build val and train shard datasets for the clean path-refiner experiment.
# Each split is skipped if shards already exist.
# Runs audit_path_refiner_dataset.py on both splits after building.
#
# Run order: this job → slurm_clean_eval_baseline.sh → slurm_clean_train_refiners.sh

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
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
OUTPUT_ROOT=runs/path_refiner_clean

POLICY=hgrid_K24_srccombined_nts1_bkr12_fb24_cm0.15_en1.25_corec60m30
TRAIN_DIR=$OUTPUT_ROOT/data/train_hgrid_K24
VAL_DIR=$OUTPUT_ROOT/data/val_hgrid_K24

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Build Datasets  $(date) ==="
echo "    SMALL_CKPT  : $SMALL_CKPT"
echo "    KNN_RUN_DIR : $KNN_RUN_DIR"
echo "    POLICY      : $POLICY"
echo "    OUTPUT_ROOT : $OUTPUT_ROOT"

for f in "$SMALL_CKPT" "$REGION_MAP" "$SUPER_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

if [[ ! -f "$KNN_RUN_DIR/per_position.npz" ]] && \
   [[ ! -f "$KNN_RUN_DIR/per_position_topk.npz" ]]; then
    echo "ERROR: per_position.npz not found in $KNN_RUN_DIR" >&2
    echo "       Run offline_region_knn.py first." >&2
    exit 1
fi

mkdir -p "$TRAIN_DIR" "$VAL_DIR"

# ── Build val dataset ─────────────────────────────────────────────────────────

echo ""
echo "=== Val dataset  $(date) ==="

N_SHARDS_VAL=$(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
if [[ "$N_SHARDS_VAL" -gt 0 ]]; then
    echo "    already exists ($N_SHARDS_VAL shards) — skipping build"
else
    python scripts/build_clean_path_refiner_dataset.py \
        --split         val                     \
        --small_ckpt    "$SMALL_CKPT"           \
        --knn_run_dir   "$KNN_RUN_DIR"          \
        --region_map    "$REGION_MAP"           \
        --super_map     "$SUPER_MAP"            \
        --output_dir    "$VAL_DIR"              \
        --policy        "$POLICY"               \
        --max_positions 300000                  \
        --shard_size    10000                   \
        --candidate_cap 8192                    \
        --device        cuda
    echo "    val shards built: $(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" | wc -l)"
fi

# ── Build train dataset ───────────────────────────────────────────────────────

echo ""
echo "=== Train dataset  $(date) ==="

N_SHARDS_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
if [[ "$N_SHARDS_TRAIN" -gt 0 ]]; then
    echo "    already exists ($N_SHARDS_TRAIN shards) — skipping build"
else
    python scripts/build_clean_path_refiner_dataset.py \
        --split         train                   \
        --small_ckpt    "$SMALL_CKPT"           \
        --knn_run_dir   "$KNN_RUN_DIR"          \
        --region_map    "$REGION_MAP"           \
        --super_map     "$SUPER_MAP"            \
        --output_dir    "$TRAIN_DIR"            \
        --policy        "$POLICY"               \
        --max_positions 1000000                 \
        --shard_size    10000                   \
        --candidate_cap 8192                    \
        --device        cuda
    echo "    train shards built: $(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" | wc -l)"
fi

# ── Audit both datasets ───────────────────────────────────────────────────────

echo ""
echo "=== Audit: val  $(date) ==="
python scripts/audit_path_refiner_dataset.py \
    --shard_dir "$VAL_DIR"

echo ""
echo "=== Audit: train  $(date) ==="
python scripts/audit_path_refiner_dataset.py \
    --shard_dir "$TRAIN_DIR"

# ── Final summary ─────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo "  Val   : $VAL_DIR/   ($(find "$VAL_DIR"   -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l) shards)"
echo "  Train : $TRAIN_DIR/ ($(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l) shards)"
echo ""
echo "Next: sbatch scripts/slurm_clean_eval_baseline.sh"
