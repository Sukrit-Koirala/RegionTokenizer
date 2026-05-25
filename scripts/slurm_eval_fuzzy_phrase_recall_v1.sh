#!/bin/bash
#SBATCH --job-name=fuzzy_eval
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Fuzzy Phrase Recall Eval V1 — Phase 2B
#
# Evaluates fuzzy phrase retrieval on validation data.
# Requires index built by slurm_build_fuzzy_phrase_index_v1.sh.
#
# For each retrieval mode × k × temperature:
#   retrieves top-k similar train rows
#   votes over their gold next tokens
#   scores candidate pool by votes
#   evaluates policies and slices
#
# NO training. NO model changes.
# Gold used ONLY after retrieval/scoring, for metrics only.
#
# Outputs:
#   runs/phrase_memory_v1/fuzzy_phrase_recall/eval/
#     fuzzy_phrase_report.md
#     fuzzy_slice_metrics.csv
#     fuzzy_policy_grid.csv
#     fuzzy_retrieval_rank_stats.csv
#     fuzzy_best_by_slice.csv
#     examples_{fuzzy_helps,fuzzy_hurts,
#               fuzzy_should_help_but_fails,fuzzy_retrieves_gold}.md
#     config.json
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
INDEX_DIR="runs/phrase_memory_v1/fuzzy_phrase_recall/index"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
EXACT_EVAL_DIR="runs/phrase_memory_v1/ngram_phrase_recall/eval"
OUTPUT_DIR="runs/phrase_memory_v1/fuzzy_phrase_recall/eval"

echo "========================================================"
echo " Fuzzy Phrase Recall Eval V1"
echo " index:  ${INDEX_DIR}"
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

if [ ! -d "$INDEX_DIR" ]; then
    echo "ERROR: index_dir not found: $INDEX_DIR"
    echo "  Run slurm_build_fuzzy_phrase_index_v1.sh first."
    exit 1
fi
if [ ! -f "${INDEX_DIR}/config.json" ]; then
    echo "ERROR: ${INDEX_DIR}/config.json not found — index incomplete"; exit 1
fi
if [ ! -f "${INDEX_DIR}/values_gold_token.npy" ]; then
    echo "ERROR: ${INDEX_DIR}/values_gold_token.npy not found — index incomplete"; exit 1
fi
echo "  [OK] $INDEX_DIR"

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

EXACT_ARG=""
if [ -d "$EXACT_EVAL_DIR" ]; then
    EXACT_ARG="--exact_eval_dir $EXACT_EVAL_DIR"
    echo "  [OK] $EXACT_EVAL_DIR  (exact ngram eval available)"
else
    echo "  [WARN] exact_eval_dir not found: $EXACT_EVAL_DIR — comparison skipped"
fi

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/eval_fuzzy_phrase_recall_v1.py
echo "  [OK] scripts/eval_fuzzy_phrase_recall_v1.py compiles"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting fuzzy phrase recall evaluation..."
echo ""

python scripts/eval_fuzzy_phrase_recall_v1.py \
    --val_dir              "${VAL_DIR}" \
    --index_dir            "${INDEX_DIR}" \
    $T2R_ARG \
    $SUPER_ARG \
    $EXACT_ARG \
    --output_dir           "${OUTPUT_DIR}" \
    --candidate_pool_size  32 \
    --modes                h_ctx,h_raw \
    --phrase_lens          16,32,64 \
    --retrieval_k_grid     8,16,32,64 \
    --temp_grid            0.05,0.1,0.2,0.5 \
    --lambda_grid          0.0,0.25,0.5,1.0,2.0,4.0 \
    --top_vote_thresholds  0.1,0.2,0.3,0.5 \
    --margin_thresholds    0.0,0.05,0.1,0.2 \
    --min_agreement_grid   1,2,4 \
    --chunk_size           4096 \
    --train_chunk_size     65536 \
    --val_batch_size       4096 \
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
echo " Fuzzy Phrase Recall Eval complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/fuzzy_phrase_report.md"
echo "  ${OUTPUT_DIR}/fuzzy_slice_metrics.csv"
echo "  ${OUTPUT_DIR}/fuzzy_retrieval_rank_stats.csv"
echo "  ${OUTPUT_DIR}/fuzzy_best_by_slice.csv"
echo "  ${OUTPUT_DIR}/config.json"
echo ""

if [ ! -f "${OUTPUT_DIR}/fuzzy_phrase_report.md" ]; then
    echo "ERROR: fuzzy_phrase_report.md not found"
    exit 1
fi
echo "[done] Report: ${OUTPUT_DIR}/fuzzy_phrase_report.md"
