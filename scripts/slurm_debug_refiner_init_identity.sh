#!/bin/bash
#SBATCH --job-name=debug_refiner_init_identity
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Verify that each refiner variant is an exact identity at init:
#   model(h, cand_tok) scores == h_prime @ tok_emb[cand_tok]  (force-zero baseline)
#
# Uses --check_init_identity which:
#   1. Builds a freshly initialized model
#   2. Runs forward on first 64 examples of the first val shard
#   3. Compares against force-zero scores (h_prime @ tok_emb[cand])
#   4. Asserts NLL diff < 1e-4 and max |delta| < 1e-5
#
# Run order:
#   slurm_clean_build_datasets.sh
#   → slurm_debug_refiner_baseline_consistency.sh
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

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Refiner Init Identity Debug  $(date) ==="
echo "    VAL_DIR : $VAL_DIR"

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

# ── Variant C ─────────────────────────────────────────────────────────────────

echo ""
echo "=== Variant C (RicherMLP)  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --val_dir             "$VAL_DIR"     \
    --small_ckpt          "$SMALL_CKPT" \
    --super_map           "$SUPER_MAP"  \
    --variant             C             \
    --d_region            32            \
    --d_hidden            256           \
    --eval_only                         \
    --check_init_identity               \
    --print_delta_stats                 \
    --device              cuda

# ── D3-small ──────────────────────────────────────────────────────────────────

echo ""
echo "=== D3-small  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --val_dir             "$VAL_DIR"     \
    --small_ckpt          "$SMALL_CKPT" \
    --super_map           "$SUPER_MAP"  \
    --variant             D3            \
    --d3_size             small         \
    --eval_only                         \
    --check_init_identity               \
    --print_delta_stats                 \
    --device              cuda

# ── D3-base ───────────────────────────────────────────────────────────────────

echo ""
echo "=== D3-base  $(date) ==="

python scripts/train_clean_path_refiner.py \
    --val_dir             "$VAL_DIR"     \
    --small_ckpt          "$SMALL_CKPT" \
    --super_map           "$SUPER_MAP"  \
    --variant             D3            \
    --d3_size             base          \
    --eval_only                         \
    --check_init_identity               \
    --print_delta_stats                 \
    --device              cuda

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo ""
echo "If all three variants printed [init identity check] PASS, the fix is correct."
echo "Next: sbatch scripts/slurm_clean_train_refiners.sh"
