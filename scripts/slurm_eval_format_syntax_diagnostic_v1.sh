#!/bin/bash
#SBATCH --job-name=format_diag
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Format/Syntax Memory Diagnostic V1
#
# Tests whether Type-A/detail errors are formatting or syntax-state errors
# (comma vs period, quote vs colon, closing brackets, wiki headings,
# @-@ markup, sentence boundaries) and whether crude heuristic format-state
# features can improve prediction on these slices.
#
# NO training. NO model changes.
# Val gold used ONLY for metrics after scoring.
# Gold never used to build features or choose candidates.
#
# Steps:
#   1. Inspect first val shard (keys/shapes).
#   2. Classify all GPT-2 tokens into format categories.
#   3. Build format slices from val data.
#   4. Extract prefix format-state features from last 128 input tokens.
#   5. Evaluate base_candidate and base_plus_format_score policies.
#   6. Collect examples (helps / hurts / confusers).
#   7. Write reports and CSVs.
#
# Answers:
#   Q1. What fraction of Bucket A errors are format/syntax-like?
#   Q2. Which format confusers are most common?
#   Q3. Do simple format-state features improve target slices?
#   Q4. Are changed_away errors controlled?
#   Q5. Should we train a learned FormatStateExpert next?
#
# Outputs:
#   runs/format_memory_v1/format_syntax_diagnostic_v1/
#     format_syntax_report.md
#     format_slice_metrics.csv
#     format_policy_grid.csv
#     format_heuristic_stats.csv
#     format_feature_summary.csv
#     examples_format_helps.md
#     examples_format_hurts.md
#     examples_format_confusers.md
#     config.json
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
OUTPUT_DIR="runs/format_memory_v1/format_syntax_diagnostic_v1"

echo "========================================================"
echo " Format/Syntax Memory Diagnostic V1"
echo " val_dir:    ${VAL_DIR}"
echo " output_dir: ${OUTPUT_DIR}"
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

T2R_ARG=""
if [ -f "$TOKEN_TO_REGION" ]; then
    T2R_ARG="--token_to_region $TOKEN_TO_REGION"
    echo "  [OK] $TOKEN_TO_REGION"
else
    echo "  [WARN] token_to_region not found: $TOKEN_TO_REGION — region slices disabled"
fi

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map $SUPER_MAP"
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found: $SUPER_MAP — superregion disabled"
fi

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/eval_format_syntax_diagnostic_v1.py
echo "  [OK] scripts/eval_format_syntax_diagnostic_v1.py compiles"
bash -n scripts/slurm_eval_format_syntax_diagnostic_v1.sh
echo "  [OK] slurm_eval_format_syntax_diagnostic_v1.sh syntax"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting format/syntax diagnostic..."
echo "  NON-PARAMETRIC DIAGNOSTIC — val gold used only for metrics"
echo ""

python scripts/eval_format_syntax_diagnostic_v1.py \
    --val_dir              "${VAL_DIR}" \
    $T2R_ARG \
    $SUPER_ARG \
    --output_dir           "${OUTPUT_DIR}" \
    --candidate_pool_size  32 \
    --memory_len           128 \
    --lambda_grid          0,0.25,0.5,1,2,4 \
    --max_examples         50 \
    --seed                 42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: eval exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Format/Syntax Diagnostic V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/format_syntax_report.md"
echo "  ${OUTPUT_DIR}/format_slice_metrics.csv"
echo "  ${OUTPUT_DIR}/format_heuristic_stats.csv"
echo "  ${OUTPUT_DIR}/format_feature_summary.csv"
echo "  ${OUTPUT_DIR}/format_policy_grid.csv"
echo ""

if [ ! -f "${OUTPUT_DIR}/format_syntax_report.md" ]; then
    echo "ERROR: format_syntax_report.md not found — evaluation may have failed"
    exit 1
fi
echo "[done] Report: ${OUTPUT_DIR}/format_syntax_report.md"
