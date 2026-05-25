#!/bin/bash
#SBATCH --job-name=ptr_support
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Pointer Support Baseline V1 — Phase 1
#
# Training-free diagnostic: tests whether explicit copy/pointer evidence
# exists in the validation data for candidate reranking.
#
# NO GPU required. NO training. NO model changes.
# Gold used ONLY after scores are computed, for metrics only.
#
# Outputs: pointer_support_report.md, pointer_support_summary.csv,
#          pointer_support_bucket_stats.csv, pointer_support_examples.md
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD
mkdir -p logs

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/pointer_sentinel_v1/pointer_support_baseline"

echo "========================================================"
echo " Pointer Support Baseline V1 — Phase 1"
echo " output: ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

echo "[preflight] Checking required inputs..."

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: val_dir not found: $VAL_DIR"; exit 1
fi
N=$(find "$VAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $VAL_DIR"; exit 1
fi
echo "  [OK] $VAL_DIR  ($N shards)"

if [ ! -f "$TOKEN_TO_REGION" ]; then
    echo "ERROR: token_to_region not found: $TOKEN_TO_REGION"; exit 1
fi
echo "  [OK] $TOKEN_TO_REGION"

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map $SUPER_MAP"
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found: $SUPER_MAP — superregion disabled"
fi

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/eval_pointer_support_baseline_v1.py
echo "  [OK] scripts/eval_pointer_support_baseline_v1.py compiles"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting pointer support evaluation..."
echo ""

python scripts/eval_pointer_support_baseline_v1.py \
    --val_dir              "${VAL_DIR}" \
    --token_to_region      "${TOKEN_TO_REGION}" \
    $SUPER_ARG \
    --output_dir           "${OUTPUT_DIR}" \
    --top_k                256 \
    --candidate_pool_size  32 \
    --memory_len           128 \
    --recency_tau          32 \
    --lambda_grid          0.0,0.25,0.5,1.0,2.0,4.0 \
    --max_examples         200 \
    --seed                 42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: eval exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Pointer Support Baseline V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/pointer_support_report.md"
echo "  ${OUTPUT_DIR}/pointer_support_summary.csv"
echo "  ${OUTPUT_DIR}/pointer_support_bucket_stats.csv"
echo "  ${OUTPUT_DIR}/pointer_support_examples.md"
echo "  ${OUTPUT_DIR}/config.json"
echo ""

if [ ! -f "${OUTPUT_DIR}/pointer_support_report.md" ]; then
    echo "ERROR: pointer_support_report.md not found"
    exit 1
fi
echo "[done] Report: ${OUTPUT_DIR}/pointer_support_report.md"
