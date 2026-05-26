#!/bin/bash
#SBATCH --job-name=passage_eval
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
# Passage Recall Dedup Diagnostic Eval V1
#
# Evaluates passage recall on validation data using a pre-built index.
# Requires index built by slurm_build_passage_recall_index_v1.sh.
#
# For each retrieval mode × k × overlap_filter:
#   retrieves top-k similar train rows by cosine similarity
#   applies dedup filter (suffix16/suffix32/jaccard03/jaccard05)
#   scores candidate pool by train continuation overlap
#   evaluates base / passage_only / base_plus_passage / confidence_gated policies
#
# NO training.  NO model changes.
# Val gold used ONLY after retrieval/scoring, for metrics only.
#
# Answers:
#   Q1. Does train-only passage recall recover gold?
#   Q2. Does it help where pointer/fuzzy fail?
#   Q3. Does it help exact numbers/entities?
#   Q4. Does performance survive dedup/overlap filtering?
#   Q5. Genuine reusable passage memory or near-duplicate memorisation?
#   Q6. Should passage recall become a memory expert?
#
# Outputs:
#   runs/passage_memory_v1/passage_recall_dedup_v1/eval/
#     passage_recall_report.md
#     passage_policy_grid.csv
#     passage_slice_metrics.csv
#     passage_dedup_stats.csv
#     examples_passage_helps.md
#     examples_passage_hurts.md
#     examples_dedup_removes_gold.md
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
INDEX_DIR="runs/passage_memory_v1/passage_recall_dedup_v1/index"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/passage_memory_v1/passage_recall_dedup_v1/eval"

echo "========================================================"
echo " Passage Recall Dedup Diagnostic Eval V1"
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
    echo "  Run slurm_build_passage_recall_index_v1.sh first."
    exit 1
fi
if [ ! -f "${INDEX_DIR}/config.json" ]; then
    echo "ERROR: ${INDEX_DIR}/config.json not found — index incomplete"; exit 1
fi
if [ ! -f "${INDEX_DIR}/gold_token.npy" ]; then
    echo "ERROR: ${INDEX_DIR}/gold_token.npy not found — index incomplete"; exit 1
fi
if [ ! -f "${INDEX_DIR}/continuations.npy" ]; then
    echo "ERROR: ${INDEX_DIR}/continuations.npy not found — index incomplete"; exit 1
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

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/eval_passage_recall_dedup_v1.py
echo "  [OK] scripts/eval_passage_recall_dedup_v1.py compiles"
bash -n scripts/slurm_eval_passage_recall_dedup_v1.sh
echo "  [OK] slurm_eval_passage_recall_dedup_v1.sh syntax"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting passage recall dedup evaluation..."
echo "  ⚠ NON-PARAMETRIC DIAGNOSTIC — val gold used only for metrics"
echo ""

python scripts/eval_passage_recall_dedup_v1.py \
    --val_dir              "${VAL_DIR}" \
    --index_dir            "${INDEX_DIR}" \
    $T2R_ARG \
    $SUPER_ARG \
    --output_dir           "${OUTPUT_DIR}" \
    --modes                h_raw,h_ctx,recency_bow \
    --candidate_pool_size  32 \
    --retrieval_k_grid     8,16,32,64 \
    --lambda_grid          0.0,0.25,0.5,1.0,2.0,4.0 \
    --overlap_filters      no_filter,suffix16,suffix32,jaccard03,jaccard05 \
    --conf_gate_thresh     0.3 \
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
echo " Passage Recall Dedup Diagnostic Eval V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/passage_recall_report.md"
echo "  ${OUTPUT_DIR}/passage_policy_grid.csv"
echo "  ${OUTPUT_DIR}/passage_slice_metrics.csv"
echo "  ${OUTPUT_DIR}/passage_dedup_stats.csv"
echo ""

if [ ! -f "${OUTPUT_DIR}/passage_recall_report.md" ]; then
    echo "ERROR: passage_recall_report.md not found — evaluation may have failed"
    exit 1
fi
echo "[done] Report: ${OUTPUT_DIR}/passage_recall_report.md"
