#!/bin/bash
#SBATCH --job-name=bucketA_attn
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# BucketA/H Attention Pattern Audit
#
# Diagnostic audit — NO training, NO model changes.
# Inspects whether gold-candidate evidence exists in attention / memory patterns
# for confuser-disambiguation examples from the TXL Detail Resolver V1 run.
#
# Answers:
#   1. Does evidence for the gold candidate exist in model attention?
#   2. Does the resolver attend to useful context but fail to exploit it?
#   3. Or is the signal absent from the memory representation?
#   4. Are changed_to_gold examples different from missed_gold?
#   5. Are changed_away examples caused by misleading context attention?
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

RUN_DIR="runs/txl_detail_resolver_v1/txlmem_selector_gate_v1"
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="${RUN_DIR}/bucketA_attention_audit"

echo "========================================================"
echo " BucketA/H Attention Pattern Audit"
echo " run_dir: ${RUN_DIR}"
echo " output:  ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

echo "[preflight] Checking required inputs..."

if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: backbone checkpoint not found: $SMALL_CKPT"; exit 1
fi
echo "  [OK] $SMALL_CKPT"

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
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found: $SUPER_MAP — superregion disabled"
fi

# Check that a checkpoint exists
if [ ! -f "${RUN_DIR}/best_txl_detail_resolver.pt" ] && \
   [ ! -f "${RUN_DIR}/latest_txl_detail_resolver.pt" ]; then
    echo "ERROR: no checkpoint found in ${RUN_DIR}"
    echo "  Expected: best_txl_detail_resolver.pt or latest_txl_detail_resolver.pt"
    exit 1
fi
echo "  [OK] checkpoint found in ${RUN_DIR}"

if [ ! -f "${RUN_DIR}/config.json" ]; then
    echo "ERROR: config.json not found in ${RUN_DIR}"; exit 1
fi
echo "  [OK] ${RUN_DIR}/config.json"

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/audit_bucketA_attention_patterns.py
echo "  [OK] scripts/audit_bucketA_attention_patterns.py compiles"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting audit..."
echo ""

python scripts/audit_bucketA_attention_patterns.py \
    --run_dir              "${RUN_DIR}" \
    --small_ckpt           "${SMALL_CKPT}" \
    --val_dir              "${VAL_DIR}" \
    --token_to_region      "${TOKEN_TO_REGION}" \
    --super_map            "${SUPER_MAP}" \
    --output_dir           "${OUTPUT_DIR}" \
    --candidate_pool_size  32 \
    --memory_len           128 \
    --max_examples_per_group 25 \
    --device               cuda \
    --amp \
    --seed                 42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: audit exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " BucketA/H Attention Audit complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/bucketA_attention_report.md"
echo "  ${OUTPUT_DIR}/bucketA_attention_summary.csv"
echo "  ${OUTPUT_DIR}/bucketA_attention_group_stats.csv"
echo "  ${OUTPUT_DIR}/bucketA_examples_changed_to_gold.md"
echo "  ${OUTPUT_DIR}/bucketA_examples_missed_gold.md"
echo "  ${OUTPUT_DIR}/bucketA_examples_selected_wrong_confuser.md"
echo "  ${OUTPUT_DIR}/bucketA_examples_changed_away.md"
echo ""

if [ ! -f "${OUTPUT_DIR}/bucketA_attention_report.md" ]; then
    echo "ERROR: bucketA_attention_report.md not found"
    exit 1
fi
echo "[done] Report: ${OUTPUT_DIR}/bucketA_attention_report.md"
