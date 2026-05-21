#!/bin/bash
#SBATCH --job-name=stage01_live_ctx
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
# Stage 01 — Build Live Context Dataset
#
# Reconstructs exact input_ids windows aligned 1:1 with existing candidate rows.
# Uses WikiText-103 + GPT-2 tokenizer + region_map to find sampled corpus positions.
#
# Context convention: input_ids = tokens[t-ctx_len:t], gold_token = tokens[t]
#
# Hard pass conditions:
#   input_ids shape [N, ctx_len]
#   gold_token alignment with candidate shards
#   row counts match
#   no future leakage (by construction)
#   token IDs in valid range
#
# Outputs:
#   runs/live_context_pipeline/stage01_context_data/alignment_report.json
#   runs/live_context_pipeline/stage01_context_data/train_ctx/shard_*.pt
#   runs/live_context_pipeline/stage01_context_data/val_ctx/shard_*.pt
#
# Prerequisite: candidate shards must exist
#   runs/path_refiner_clean/data/train_hgrid_K24/
#   runs/path_refiner_clean/data/val_hgrid_K24/
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

# ── Paths ────────────────────────────────────────────────────────────────────
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_CAND_DIR="runs/path_refiner_clean/data/train_hgrid_K24"
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
TRAIN_FEAT_DIR="runs/path_refiner_residual_interface/features/train_multilayer"
VAL_FEAT_DIR="runs/path_refiner_residual_interface/features/val_multilayer"
REGION_MAP="runs/region_maps_128/token_to_region.json"
OUTPUT_ROOT="runs/live_context_pipeline/stage01_context_data"

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Stage 01 — Build Live Context Dataset"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking required inputs..."

for F in "$SMALL_CKPT" "$REGION_MAP"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: required file not found: $F"
        exit 1
    fi
    echo "  [OK] $F"
done

for D in "$TRAIN_CAND_DIR" "$VAL_CAND_DIR"; do
    if [ ! -d "$D" ]; then
        echo "ERROR: required directory not found: $D"
        exit 1
    fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then
        echo "ERROR: no shard_*.pt in $D"
        exit 1
    fi
    echo "  [OK] $D  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present."
echo ""

mkdir -p "$OUTPUT_ROOT"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Running stage01_build_live_context_dataset.py"
echo "========================================================"
echo ""

python scripts/stage01_build_live_context_dataset.py \
    --small_ckpt      "$SMALL_CKPT"      \
    --train_cand_dir  "$TRAIN_CAND_DIR"  \
    --val_cand_dir    "$VAL_CAND_DIR"    \
    --region_map      "$REGION_MAP"      \
    --output_root     "$OUTPUT_ROOT"     \
    --ctx_len         256                \
    --fail_on_alignment_error

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo "ERROR: stage01 exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Stage 01 complete. $(date)"
echo "========================================================"
echo ""
echo "Required before Stage 02:"
echo "  $OUTPUT_ROOT/alignment_report.json  (alignment_pass == true)"
echo "  $OUTPUT_ROOT/train_ctx/shard_*.pt"
echo "  $OUTPUT_ROOT/val_ctx/shard_*.pt"
echo ""
