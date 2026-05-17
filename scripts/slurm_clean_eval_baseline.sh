#!/bin/bash
#SBATCH --job-name=clean_eval_baseline
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
# Masked-softmax baseline eval for the clean path-refiner experiment.
# Runs the backbone once over the val split and scores all candidate policies.
# Outputs masked_softmax_eval.csv to $EVAL_DIR.
#
# Also re-runs the dataset audit with coverage comparison against the baseline
# so you can verify the val dataset coverage matches before training.
#
# Run order: slurm_clean_build_datasets.sh → this job → slurm_clean_train_refiners.sh

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
PARETO_CSV=runs/hier_policy_tuning/tuned_pareto_frontier.csv
OUTPUT_ROOT=runs/path_refiner_clean

POLICY=hgrid_K24_srccombined_nts1_bkr12_fb24_cm0.15_en1.25_corec60m30
VAL_DIR=$OUTPUT_ROOT/data/val_hgrid_K24
EVAL_DIR=$OUTPUT_ROOT/eval

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Masked-Softmax Baseline Eval  $(date) ==="
echo "    SMALL_CKPT  : $SMALL_CKPT"
echo "    KNN_RUN_DIR : $KNN_RUN_DIR"
echo "    EVAL_DIR    : $EVAL_DIR"

for f in "$SMALL_CKPT" "$REGION_MAP" "$SUPER_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
if [[ "$N_VAL" -eq 0 ]]; then
    echo "ERROR: no val shards found in $VAL_DIR" >&2
    echo "       Run slurm_clean_build_datasets.sh first." >&2
    exit 1
fi
echo "    val shards  : $N_VAL"

mkdir -p "$EVAL_DIR"

# ── Masked-softmax eval ───────────────────────────────────────────────────────

echo ""
echo "=== Running eval  $(date) ==="

PARETO_ARG=""
if [[ -f "$PARETO_CSV" ]]; then
    PARETO_ARG="--pareto_csv $PARETO_CSV"
fi

python scripts/eval_clean_masked_softmax.py \
    --small_ckpt    "$SMALL_CKPT"   \
    --knn_run_dir   "$KNN_RUN_DIR"  \
    --region_map    "$REGION_MAP"   \
    --super_map     "$SUPER_MAP"    \
    --output_dir    "$EVAL_DIR"     \
    $PARETO_ARG                     \
    --max_positions 300000          \
    --device        cuda

# ── Dataset audit with coverage comparison ────────────────────────────────────

echo ""
echo "=== Audit val dataset vs baseline  $(date) ==="

python scripts/audit_path_refiner_dataset.py \
    --shard_dir       "$VAL_DIR"                        \
    --masked_eval_csv "$EVAL_DIR/masked_softmax_eval.csv" \
    --policy          "$POLICY"                         \
    --tol             0.01

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo "  Output: $EVAL_DIR/masked_softmax_eval.csv"
echo ""

if [[ -f "$EVAL_DIR/masked_softmax_eval.csv" ]]; then
    echo "Top-3 policies by strict_nll:"
    head -1 "$EVAL_DIR/masked_softmax_eval.csv"
    tail -n +2 "$EVAL_DIR/masked_softmax_eval.csv" | sort -t',' -k6 -n | head -3
fi

echo ""
echo "Next: sbatch scripts/slurm_clean_train_refiners.sh"
