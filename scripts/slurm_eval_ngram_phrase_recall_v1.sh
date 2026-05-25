#!/bin/bash
#SBATCH --job-name=ngram_eval
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# N-gram Phrase Recall Eval V1 — Phase 1B
#
# Evaluates n-gram phrase memory on validation data.
# Requires datastore built by slurm_build_ngram_phrase_datastore_v1.sh.
#
# NO GPU required. NO training. NO model changes.
# Gold used ONLY after lookup/scoring, for metrics only.
#
# Outputs:
#   runs/phrase_memory_v1/ngram_phrase_recall/eval/
#     phrase_recall_report.md
#     phrase_slice_metrics.csv
#     phrase_policy_grid.csv
#     phrase_rank_stats.csv
#     examples_{phrase_helps,phrase_hurts,phrase_should_help_but_fails,
#                phrase_key_hits,phrase_no_hit}.md
#     phrase_examples_all.md
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
DATASTORE_DIR="runs/phrase_memory_v1/ngram_phrase_recall/datastore"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/phrase_memory_v1/ngram_phrase_recall/eval"

echo "========================================================"
echo " N-gram Phrase Recall Eval V1"
echo " datastore: ${DATASTORE_DIR}"
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

if [ ! -d "$DATASTORE_DIR" ]; then
    echo "ERROR: datastore_dir not found: $DATASTORE_DIR"
    echo "  Run slurm_build_ngram_phrase_datastore_v1.sh first."
    exit 1
fi
for N_LEN in 4 8 16 32; do
    if [ ! -f "${DATASTORE_DIR}/ngram_${N_LEN}.pkl" ]; then
        echo "ERROR: ${DATASTORE_DIR}/ngram_${N_LEN}.pkl not found"; exit 1
    fi
done
echo "  [OK] $DATASTORE_DIR  (all ngram_*.pkl present)"

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
python -m py_compile scripts/eval_ngram_phrase_recall_v1.py
echo "  [OK] scripts/eval_ngram_phrase_recall_v1.py compiles"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting n-gram phrase recall evaluation..."
echo ""

python scripts/eval_ngram_phrase_recall_v1.py \
    --val_dir              "${VAL_DIR}" \
    --datastore_dir        "${DATASTORE_DIR}" \
    $T2R_ARG \
    $SUPER_ARG \
    --output_dir           "${OUTPUT_DIR}" \
    --candidate_pool_size  32 \
    --ngram_lengths        4,8,16,32 \
    --lambda_grid          0.0,0.25,0.5,1.0,2.0,4.0 \
    --min_count_grid       1,2,4,8 \
    --prob_thresholds      0.1,0.2,0.3,0.5,0.7 \
    --margin_thresholds    0.0,0.05,0.1,0.2 \
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
echo " N-gram Phrase Recall Eval complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/phrase_recall_report.md"
echo "  ${OUTPUT_DIR}/phrase_slice_metrics.csv"
echo "  ${OUTPUT_DIR}/phrase_policy_grid.csv"
echo "  ${OUTPUT_DIR}/phrase_rank_stats.csv"
echo "  ${OUTPUT_DIR}/phrase_examples_all.md"
echo "  ${OUTPUT_DIR}/config.json"
echo ""

if [ ! -f "${OUTPUT_DIR}/phrase_recall_report.md" ]; then
    echo "ERROR: phrase_recall_report.md not found"
    exit 1
fi
echo "[done] Report: ${OUTPUT_DIR}/phrase_recall_report.md"
