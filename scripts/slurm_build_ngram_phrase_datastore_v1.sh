#!/bin/bash
#SBATCH --job-name=ngram_build
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# N-gram Phrase Datastore Build V1 — Phase 1A
#
# Builds per-suffix-length hash maps:
#   key = tuple(last n tokens of context)
#   value = counts of gold_token
#
# NO GPU required. NO training. NO model changes.
# Uses train shards only. Val data never included.
#
# Outputs:
#   runs/phrase_memory_v1/ngram_phrase_recall/datastore/
#     ngram_4.pkl, ngram_8.pkl, ngram_16.pkl, ngram_32.pkl
#     build_config.json
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
OUTPUT_DIR="runs/phrase_memory_v1/ngram_phrase_recall/datastore"

echo "========================================================"
echo " N-gram Phrase Datastore Build V1"
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
python -m py_compile scripts/build_ngram_phrase_datastore_v1.py
echo "  [OK] scripts/build_ngram_phrase_datastore_v1.py compiles"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting datastore build..."
echo ""

python scripts/build_ngram_phrase_datastore_v1.py \
    --train_dir      "${TRAIN_DIR}" \
    --output_dir     "${OUTPUT_DIR}" \
    --ngram_lengths  4,8,16,32 \
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
echo " N-gram Datastore Build complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/ngram_4.pkl"
echo "  ${OUTPUT_DIR}/ngram_8.pkl"
echo "  ${OUTPUT_DIR}/ngram_16.pkl"
echo "  ${OUTPUT_DIR}/ngram_32.pkl"
echo "  ${OUTPUT_DIR}/build_stats.json"
echo "  ${OUTPUT_DIR}/build_config.json"
echo ""

for N_LEN in 4 8 16 32; do
    if [ ! -f "${OUTPUT_DIR}/ngram_${N_LEN}.pkl" ]; then
        echo "ERROR: ngram_${N_LEN}.pkl not found"
        exit 1
    fi
done
echo "[done] All ngram_*.pkl files present."
