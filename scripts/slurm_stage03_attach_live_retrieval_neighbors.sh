#!/bin/bash
#SBATCH --job-name=stage03_attach_ret
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
# Stage 03 — Attach Retrieval Neighbors
#
# Queries index for every train/val row. Train self-excluded. Val train-only.
#
# Hard pass conditions:
#   val retrieves train-only (no val rows in any neighbor list)
#   no train row has itself in final neighbors
#   neighbor scores finite
#   neighbor tokens valid
#
# Prerequisite: Stage 01 + Stage 02 must have passed
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
INDEX_DIR="runs/live_context_pipeline/stage02_retrieval_index/hctx_train"
TRAIN_CTX_DIR="runs/live_context_pipeline/stage01_context_data/train_ctx"
VAL_CTX_DIR="runs/live_context_pipeline/stage01_context_data/val_ctx"
TRAIN_CAND_DIR="runs/path_refiner_clean/data/train_hgrid_K24"
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
OUTPUT_ROOT="runs/live_context_pipeline/stage03_retrieval_neighbors"
STAGE02_REPORT="$INDEX_DIR/index_report.json"

echo "========================================================"
echo " Stage 03 — Attach Retrieval Neighbors"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking Stage 01 + 02 outputs..."

for F in "$SMALL_CKPT" "$STAGE02_REPORT" "$INDEX_DIR/train_row_ids.npy"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: required file not found: $F"; exit 1
    fi
    echo "  [OK] $F"
done

python -c "
import json, sys
r = json.load(open('$STAGE02_REPORT'))
if not r.get('index_pass', False):
    print('ERROR: Stage 02 index_pass is False. Fix Stage 02 first.')
    sys.exit(1)
print('  [OK] Stage 02 index_pass = True')
"

for D in "$TRAIN_CTX_DIR" "$VAL_CTX_DIR" "$TRAIN_CAND_DIR" "$VAL_CAND_DIR"; do
    if [ ! -d "$D" ]; then echo "ERROR: not found: $D"; exit 1; fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then echo "ERROR: no shards in $D"; exit 1; fi
    echo "  [OK] $D  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_ROOT"

python scripts/stage03_attach_live_retrieval_neighbors.py \
    --small_ckpt          "$SMALL_CKPT"       \
    --index_dir           "$INDEX_DIR"         \
    --train_ctx_dir       "$TRAIN_CTX_DIR"     \
    --val_ctx_dir         "$VAL_CTX_DIR"       \
    --train_cand_dir      "$TRAIN_CAND_DIR"    \
    --val_cand_dir        "$VAL_CAND_DIR"      \
    --output_root         "$OUTPUT_ROOT"       \
    --num_neighbors       32                   \
    --exclude_self_for_train                   \
    --batch_size          64

EXIT=$?
if [ $EXIT -ne 0 ]; then echo "ERROR: stage03 exited with code $EXIT"; exit $EXIT; fi

echo ""
echo "========================================================"
echo " Stage 03 complete. $(date)"
echo "========================================================"
echo ""
echo "Required before Stage 04:"
echo "  $OUTPUT_ROOT/retrieval_attach_report.json  (leakage_detected == false)"
echo ""
