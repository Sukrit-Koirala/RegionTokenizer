#!/bin/bash
#SBATCH --job-name=stage04_audit_ret
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
# Stage 04 — Audit Context + Retrieval Data
#
# Proves the new data is clean and useful before model training.
# Reports gold_in_base_top256, gold_in_neighbors, retrieval_added_gold,
# and boundary-specific versions of all metrics.
#
# Prerequisite: Stage 01 + 03 must have passed
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME="$SLURM_SUBMIT_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
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

SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
VAL_CTX_DIR="runs/live_context_pipeline/stage01_context_data/val_ctx"
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
VAL_RET_DIR="runs/live_context_pipeline/stage03_retrieval_neighbors/val_retrieval"
OUTPUT_DIR="runs/live_context_pipeline/stage04_data_audit"
STAGE03_REPORT="runs/live_context_pipeline/stage03_retrieval_neighbors/retrieval_attach_report.json"

echo "========================================================"
echo " Stage 04 — Audit Context + Retrieval Data"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking Stage 01 + 03 outputs..."

for F in "$SMALL_CKPT" "$STAGE03_REPORT"; do
    if [ ! -f "$F" ]; then echo "ERROR: required file not found: $F"; exit 1; fi
    echo "  [OK] $F"
done

python -c "
import json, sys
r = json.load(open('$STAGE03_REPORT'))
if r.get('leakage_detected', True):
    print('ERROR: Stage 03 detected leakage. Fix Stage 03 first.')
    sys.exit(1)
print('  [OK] Stage 03 leakage_detected = False')
"

for D in "$VAL_CTX_DIR" "$VAL_CAND_DIR" "$VAL_RET_DIR"; do
    if [ ! -d "$D" ]; then echo "ERROR: not found: $D"; exit 1; fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then echo "ERROR: no shards in $D"; exit 1; fi
    echo "  [OK] $D  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_DIR"

python scripts/stage04_audit_live_context_retrieval_data.py \
    --small_ckpt          "$SMALL_CKPT"    \
    --val_ctx_dir         "$VAL_CTX_DIR"   \
    --val_cand_dir        "$VAL_CAND_DIR"  \
    --val_retrieval_dir   "$VAL_RET_DIR"   \
    --output_dir          "$OUTPUT_DIR"    \
    --top_k               256              \
    --num_neighbors       32               \
    --batch_size          64

EXIT=$?
if [ $EXIT -ne 0 ]; then echo "ERROR: stage04 exited with code $EXIT"; exit $EXIT; fi

echo ""
echo "========================================================"
echo " Stage 04 complete. $(date)"
echo "========================================================"
echo ""
echo "Required before Stage 05:"
echo "  $OUTPUT_DIR/retrieval_audit.json  (audit_pass == true)"
echo ""
