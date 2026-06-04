#!/bin/bash
#SBATCH --job-name=eval_region_v2
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
# Phase 1B: Evaluate Compute-Aware Region Maps (standalone eval)
#
# Evaluates all V2* maps in maps_root for candidate compression quality
# WITHOUT training a router. Uses frequency-prior coverage as a fast proxy.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH="$PWD"
mkdir -p logs

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate learned_regions 2>/dev/null || conda activate base
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
    conda activate learned_regions 2>/dev/null || conda activate base
fi

TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
OLD_MAP="runs/region_maps_128/token_to_region.json"
MAPS_ROOT="runs/cheap_ai/phase1B_compute_aware_region_maps"
OUTPUT_DIR="${MAPS_ROOT}/eval"
SCRIPT="scripts/evaluate_compute_aware_region_maps.py"

echo "========================================================"
echo " Phase 1B: Evaluate Compute-Aware Region Maps"
echo " maps_root:  ${MAPS_ROOT}"
echo " output_dir: ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

echo "[preflight] Checking inputs..."
[ ! -f "$SCRIPT" ] && { echo "ERROR: $SCRIPT not found"; exit 1; }
echo "  [OK] $SCRIPT"
[ ! -d "$TRAIN_DIR" ] && { echo "ERROR: $TRAIN_DIR not found"; exit 1; }
echo "  [OK] $TRAIN_DIR"
[ ! -d "$VAL_DIR" ]   && { echo "ERROR: $VAL_DIR not found"; exit 1; }
echo "  [OK] $VAL_DIR"
[ ! -d "$MAPS_ROOT" ] && { echo "ERROR: $MAPS_ROOT not found — run build first"; exit 1; }
N_MAPS=$(find "$MAPS_ROOT" -maxdepth 1 -type d -name 'V2*' 2>/dev/null | wc -l)
echo "  [OK] $MAPS_ROOT  ($N_MAPS map variants found)"

OLD_ARG=""
[ -f "$OLD_MAP" ] && { OLD_ARG="--old_map ${OLD_MAP}"; echo "  [OK] $OLD_MAP"; }

python -m py_compile "$SCRIPT"
echo "  [OK] $SCRIPT compiles"
echo ""

mkdir -p "$OUTPUT_DIR"

python "$SCRIPT" \
    --maps_root     "$MAPS_ROOT" \
    --train_dir     "$TRAIN_DIR" \
    --val_dir       "$VAL_DIR" \
    $OLD_ARG \
    --output_dir    "$OUTPUT_DIR" \
    --ks            "1,2,4,8,16,32,64" \
    --seed          42

EXIT=$?
[ $EXIT -ne 0 ] && { echo "ERROR: eval exited $EXIT"; exit $EXIT; }

echo ""
echo "========================================================"
echo " Eval complete. $(date)"
echo "========================================================"
echo "  ${OUTPUT_DIR}/map_eval_summary.csv"
echo "  ${OUTPUT_DIR}/map_eval_report.md"
