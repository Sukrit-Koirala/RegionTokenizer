#!/bin/bash
#SBATCH --job-name=stage02_ret_index
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
# Stage 02 — Build Train-Only Retrieval Index
#
# Runs frozen backbone on train input_ids (from Stage 01) to get h_ctx vectors.
# Builds FAISS (or numpy) nearest-neighbor index over TRAIN rows only.
# Val rows are NEVER indexed.
#
# Hard pass conditions:
#   n_train_vectors == total train rows from Stage 01
#   no val rows in index
#   vectors normalized
#   random self-query retrieves self at rank 1
#
# Prerequisite: Stage 01 must have passed
#   runs/live_context_pipeline/stage01_context_data/alignment_report.json
#   runs/live_context_pipeline/stage01_context_data/train_ctx/shard_*.pt
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
TRAIN_CTX_DIR="runs/live_context_pipeline/stage01_context_data/train_ctx"
TRAIN_CAND_DIR="runs/path_refiner_clean/data/train_hgrid_K24"
STAGE01_REPORT="runs/live_context_pipeline/stage01_context_data/alignment_report.json"
OUTPUT_DIR="runs/live_context_pipeline/stage02_retrieval_index/hctx_train"

echo "========================================================"
echo " Stage 02 — Build Train-Only Retrieval Index"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking Stage 01 outputs..."

for F in "$SMALL_CKPT" "$STAGE01_REPORT"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: required file not found: $F"
        echo "  Run Stage 01 first."
        exit 1
    fi
    echo "  [OK] $F"
done

python -c "
import json, sys
r = json.load(open('$STAGE01_REPORT'))
if not r.get('alignment_pass', False):
    print('ERROR: Stage 01 alignment_pass is False. Fix Stage 01 before running Stage 02.')
    sys.exit(1)
print('  [OK] Stage 01 alignment_pass = True')
"

for D in "$TRAIN_CTX_DIR" "$TRAIN_CAND_DIR"; do
    if [ ! -d "$D" ]; then
        echo "ERROR: required directory not found: $D"
        exit 1
    fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then echo "ERROR: no shard_*.pt in $D"; exit 1; fi
    echo "  [OK] $D  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_DIR"

python scripts/stage02_build_live_retrieval_index.py \
    --small_ckpt          "$SMALL_CKPT"       \
    --train_ctx_dir       "$TRAIN_CTX_DIR"     \
    --train_cand_dir      "$TRAIN_CAND_DIR"    \
    --output_dir          "$OUTPUT_DIR"        \
    --ctx_len             256                  \
    --normalize                                \
    --metric              cosine               \
    --use_faiss_if_available                   \
    --batch_size          64

EXIT=$?
if [ $EXIT -ne 0 ]; then echo "ERROR: stage02 exited with code $EXIT"; exit $EXIT; fi

echo ""
echo "========================================================"
echo " Stage 02 complete. $(date)"
echo "========================================================"
echo ""
echo "Required before Stage 03:"
echo "  $OUTPUT_DIR/index_report.json  (pass == true)"
echo "  $OUTPUT_DIR/train_row_ids.npy"
echo ""
