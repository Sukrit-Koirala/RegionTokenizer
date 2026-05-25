#!/bin/bash
#SBATCH --job-name=fuzzy_build
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
# Fuzzy Phrase Index Build V1 — Phase 2A
#
# Builds retrieval key matrices from train shards:
#   h_ctx keys     : normalized contextual hidden vectors
#   h_raw keys     : normalized backbone hidden vectors (if available)
#   bow keys       : hashed bag-of-tokens (per phrase_len)
#   recency_bow    : recency-weighted bag-of-tokens (per phrase_len)
#
# Saved as float16 memmaps + gold_token / row_id .npy companions.
#
# NO training. NO model changes.
# Uses train shards only — val data never read.
#
# Outputs:
#   runs/phrase_memory_v1/fuzzy_phrase_recall/index/
#     h_ctx_keys.mmap
#     h_raw_keys.mmap
#     bow_len{16,32,64}_keys.mmap
#     recency_bow_len{16,32,64}_keys.mmap
#     values_gold_token.npy
#     row_id.npy
#     token_offset.npy
#     config.json
#     build_stats.json
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

TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
OUTPUT_DIR="runs/phrase_memory_v1/fuzzy_phrase_recall/index"

echo "========================================================"
echo " Fuzzy Phrase Index Build V1"
echo " output: ${OUTPUT_DIR}"
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

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/build_fuzzy_phrase_index_v1.py
echo "  [OK] scripts/build_fuzzy_phrase_index_v1.py compiles"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting fuzzy phrase index build..."
echo ""

python scripts/build_fuzzy_phrase_index_v1.py \
    --train_dir      "${TRAIN_DIR}" \
    --output_dir     "${OUTPUT_DIR}" \
    --modes          h_ctx,h_raw,recency_bow,bow \
    --phrase_lens    16,32,64 \
    --feature_dim    8192 \
    --max_train_rows -1 \
    --seed           42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: build exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Fuzzy Phrase Index Build complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/h_ctx_keys.mmap"
echo "  ${OUTPUT_DIR}/h_raw_keys.mmap"
echo "  ${OUTPUT_DIR}/values_gold_token.npy"
echo "  ${OUTPUT_DIR}/build_stats.json"
echo "  ${OUTPUT_DIR}/config.json"
echo ""

if [ ! -f "${OUTPUT_DIR}/config.json" ]; then
    echo "ERROR: config.json not found — build may have failed"
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/values_gold_token.npy" ]; then
    echo "ERROR: values_gold_token.npy not found"
    exit 1
fi
echo "[done] Index: ${OUTPUT_DIR}"
