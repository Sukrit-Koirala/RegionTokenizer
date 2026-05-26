#!/bin/bash
#SBATCH --job-name=passage_build
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
# Passage Recall Index Build V1
#
# Builds train-only passage retrieval index for Passage Recall Dedup Diagnostic V1.
# No GPU computation — GPU allocation is for parity with eval job.
# Reads all train shards; stores per-mode key memmaps + continuations.
#
# ⚠ TRAIN DATA ONLY: val/test data never ingested.
#
# Modes built: h_raw, h_ctx, recency_bow, shingle5gram
#
# Outputs:
#   runs/passage_memory_v1/passage_recall_dedup_v1/index/
#     h_raw_keys.mmap
#     h_ctx_keys.mmap
#     recency_bow_keys.mmap
#     shingle5gram_keys.mmap
#     gold_token.npy
#     continuations.npy
#     train_prefixes.mmap
#     row_id.npy
#     token_offset.npy
#     config.json
#     build_stats.json
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

TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
OUTPUT_DIR="runs/passage_memory_v1/passage_recall_dedup_v1/index"

# Optional: point directly at cached train corpus to avoid derivation
# Typically at: runs/live_full_pipeline_rebuild_limited500k_logitfix/00_raw_token_source/train_tokens.npy
CORPUS_PATH="runs/live_full_pipeline_rebuild_limited500k_logitfix/00_raw_token_source/train_tokens.npy"

echo "========================================================"
echo " Passage Recall Index Build V1"
echo " train_dir:  ${TRAIN_DIR}"
echo " output_dir: ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

echo "[preflight] Checking required inputs..."

if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: train_dir not found: $TRAIN_DIR"; exit 1
fi
N=$(find "$TRAIN_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $TRAIN_DIR"; exit 1
fi
echo "  [OK] $TRAIN_DIR  ($N shards)"

CORPUS_ARG=""
if [ -f "$CORPUS_PATH" ]; then
    CORPUS_ARG="--corpus_path $CORPUS_PATH"
    echo "  [OK] $CORPUS_PATH  (train corpus cache)"
else
    echo "  [WARN] $CORPUS_PATH not found — will try pipeline-cache fallback"
    echo "         (continuations may fall back to gold_token only)"
fi

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/build_passage_recall_index_v1.py
echo "  [OK] scripts/build_passage_recall_index_v1.py compiles"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting passage recall index build..."
echo "  ⚠ TRAIN DATA ONLY — val data never ingested"
echo ""

python scripts/build_passage_recall_index_v1.py \
    --train_dir         "${TRAIN_DIR}" \
    --output_dir        "${OUTPUT_DIR}" \
    $CORPUS_ARG \
    --modes             h_raw,h_ctx,recency_bow,shingle5gram \
    --query_len         128 \
    --continuation_len  8 \
    --feature_dim       8192 \
    --max_train_rows    -1 \
    --seed              42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: build exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Passage Recall Index Build V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/config.json"
echo "  ${OUTPUT_DIR}/build_stats.json"
echo "  ${OUTPUT_DIR}/gold_token.npy"
echo "  ${OUTPUT_DIR}/continuations.npy"
echo "  ${OUTPUT_DIR}/train_prefixes.mmap"
echo ""

if [ ! -f "${OUTPUT_DIR}/config.json" ]; then
    echo "ERROR: config.json not found — build may have failed"
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/build_stats.json" ]; then
    echo "ERROR: build_stats.json not found — build may have failed"
    exit 1
fi
echo "[done] Index: ${OUTPUT_DIR}"
echo "       Run slurm_eval_passage_recall_dedup_v1.sh next."
