#!/bin/bash
#SBATCH --job-name=ptr_slices
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Pointer Help Slices V1 — Phase 2
#
# Focused diagnostic: measures whether pointer support helps exactly where
# it is intended to help (copy-supported cases).
#
# NO GPU required. NO training. NO model changes.
# Gold used ONLY after scoring, for metrics only.
#
# Inputs:
#   val shards           runs/live_full_pipeline_rebuild_limited500k_logitfix/…/val
#   phase1 baseline      runs/pointer_sentinel_v1/pointer_support_baseline
#   token_to_region      runs/region_maps_128/token_to_region.json
#   super_map            runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
#
# Outputs:
#   pointer_help_report.md
#   pointer_help_slice_summary.csv
#   pointer_policy_grid.csv
#   pointer_rank_stats.csv
#   examples_{pointer_helps,pointer_hurts,should_help_fails,pointer_danger}.md
#   pointer_help_examples_all.md
#   config.json
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
PHASE1_DIR="runs/pointer_sentinel_v1/pointer_support_baseline"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/pointer_sentinel_v1/pointer_help_slices_v1"

echo "========================================================"
echo " Pointer Help Slices V1 — Phase 2"
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

PHASE1_ARG=""
if [ -d "$PHASE1_DIR" ]; then
    PHASE1_ARG="--phase1_dir $PHASE1_DIR"
    echo "  [OK] $PHASE1_DIR  (phase1 config enabled)"
else
    echo "  [WARN] phase1_dir not found: $PHASE1_DIR — phase1 config skipped"
fi

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map $SUPER_MAP"
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found: $SUPER_MAP — superregion disabled"
fi

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/analyze_pointer_help_slices_v1.py
echo "  [OK] scripts/analyze_pointer_help_slices_v1.py compiles"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting pointer help slice analysis..."
echo ""

python scripts/analyze_pointer_help_slices_v1.py \
    --val_dir              "${VAL_DIR}" \
    $PHASE1_ARG \
    --token_to_region      "${TOKEN_TO_REGION}" \
    $SUPER_ARG \
    --output_dir           "${OUTPUT_DIR}" \
    --top_k                256 \
    --candidate_pool_size  32 \
    --memory_len           128 \
    --recency_tau          32 \
    --lambda_grid          0.0,0.25,0.5,1.0,2.0,4.0 \
    --pointer_conf_thresholds  0.1,0.2,0.3,0.4,0.5,0.7 \
    --pointer_margin_thresholds 0.0,0.05,0.1,0.2,0.5 \
    --max_examples         50 \
    --seed                 42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: analysis exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Pointer Help Slices V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/pointer_help_report.md"
echo "  ${OUTPUT_DIR}/pointer_help_slice_summary.csv"
echo "  ${OUTPUT_DIR}/pointer_policy_grid.csv"
echo "  ${OUTPUT_DIR}/pointer_rank_stats.csv"
echo "  ${OUTPUT_DIR}/pointer_help_examples_all.md"
echo "  ${OUTPUT_DIR}/config.json"
echo ""

if [ ! -f "${OUTPUT_DIR}/pointer_help_report.md" ]; then
    echo "ERROR: pointer_help_report.md not found"
    exit 1
fi
echo "[done] Report: ${OUTPUT_DIR}/pointer_help_report.md"
