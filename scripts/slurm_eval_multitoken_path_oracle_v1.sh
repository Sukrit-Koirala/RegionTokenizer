#!/bin/bash
#SBATCH --job-name=path_oracle
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Multi-Token Path Oracle Eval V1
#
# ⚠ ORACLE_DIAGNOSTIC ONLY ⚠
# Uses future gold continuation tokens.
# NOT real validation metrics. NOT deployable.
# Upper-bound existence test for path-memory signal.
#
# Questions answered:
#   Q1. Does future continuation make gold easier to identify?
#   Q2. Which path length works best: 2,4,8?
#   Q3. Does path scoring improve selected_gold_given_in_pool?
#   Q4. Does path scoring help same-region confusers?
#   Q5. Does path scoring help gold_not_in_context rows?
#   Q6. Which token classes benefit most?
#   Q7. Is the oracle upper bound strong enough to justify a non-oracle model?
#   Q8. Should multi-token path memory become memory expert #3?
#
# Outputs:
#   runs/path_memory_v1/multitoken_oracle_v1/
#     config.json
#     path_oracle_summary.csv
#     path_oracle_by_slice.csv
#     path_rank_stats.csv
#     examples_path_helps.md
#     examples_path_fails.md
#     examples_path_gold_best_continuation.md
#     examples_path_base_wrong_but_future_disambiguates.md
#     report.md
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
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
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/path_memory_v1/multitoken_oracle_v1"

# Optional: point directly at cached val corpus to avoid HuggingFace download
# Typically at: runs/live_full_pipeline_rebuild_limited500k_logitfix/00_raw_token_source/val_tokens.npy
CORPUS_PATH="runs/live_full_pipeline_rebuild_limited500k_logitfix/00_raw_token_source/val_tokens.npy"

echo "========================================================"
echo " ⚠  ORACLE_DIAGNOSTIC: Multi-Token Path Oracle V1"
echo "    Uses future gold tokens. NOT real validation metrics."
echo " val_dir:   ${VAL_DIR}"
echo " small_ckpt:${SMALL_CKPT}"
echo " output:    ${OUTPUT_DIR}"
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

if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: checkpoint not found: $SMALL_CKPT"; exit 1
fi
echo "  [OK] $SMALL_CKPT"

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

CORPUS_ARG=""
if [ -f "$CORPUS_PATH" ]; then
    CORPUS_ARG="--corpus_path $CORPUS_PATH"
    echo "  [OK] $CORPUS_PATH  (val corpus cache)"
else
    echo "  [WARN] $CORPUS_PATH not found — will attempt HuggingFace download"
fi

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/eval_multitoken_path_oracle_v1.py
echo "  [OK] scripts/eval_multitoken_path_oracle_v1.py compiles"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting multi-token path oracle evaluation..."
echo "  ⚠ ORACLE_DIAGNOSTIC: future gold tokens will be read from corpus"
echo ""

python scripts/eval_multitoken_path_oracle_v1.py \
    --val_dir              "${VAL_DIR}" \
    --small_ckpt           "${SMALL_CKPT}" \
    $T2R_ARG \
    $SUPER_ARG \
    $CORPUS_ARG \
    --output_dir           "${OUTPUT_DIR}" \
    --eval_filter          bucketA_confuser \
    --candidate_pool_size  16 \
    --path_lens            1,2,4,8 \
    --alpha_grid           0.25,0.5,1.0,2.0 \
    --max_rows             5000 \
    --batch_size           16 \
    --amp \
    --device               cuda \
    --seed                 42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: eval exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " ⚠ ORACLE_DIAGNOSTIC complete. $(date)"
echo " Results are NOT real validation metrics."
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/report.md"
echo "  ${OUTPUT_DIR}/path_oracle_summary.csv"
echo "  ${OUTPUT_DIR}/path_oracle_by_slice.csv"
echo "  ${OUTPUT_DIR}/path_rank_stats.csv"
echo ""

if [ ! -f "${OUTPUT_DIR}/report.md" ]; then
    echo "ERROR: report.md not found — evaluation may have failed"
    exit 1
fi
echo "[done] Report: ${OUTPUT_DIR}/report.md"
echo "       ⚠ ORACLE_DIAGNOSTIC — do not use as real validation results."
