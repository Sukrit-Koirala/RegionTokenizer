#!/bin/bash
#SBATCH --job-name=tune_hier
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Hierarchical policy tuning (Tasks 1-6).
# Loads per_position.npz + pre-built region_to_superregion_K*.json,
# evaluates full policy grid, writes Pareto frontier + tuned report.
#
# Prerequisites:
#   analyze_hard_memory_and_predictive_hierarchy.py must have completed
#   (region_to_superregion_K*.json files must exist in HIER_DIR).

source ~/miniconda3/bin/activate
conda activate learned_regions

set -eo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME="$SLURM_SUBMIT_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD
mkdir -p logs

# ── Configurable paths ────────────────────────────────────────────────────────

KNN_RUN_DIR=runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20
REGION_MAP=runs/region_maps_128/token_to_region.json
HIER_DIR=runs/hard_memory_predictive_hierarchy
OUTPUT_DIR=runs/hard_memory_predictive_hierarchy

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Hierarchical Policy Tuning  $(date) ==="
echo "    KNN_RUN_DIR : $KNN_RUN_DIR"
echo "    HIER_DIR    : $HIER_DIR"
echo "    OUTPUT_DIR  : $OUTPUT_DIR"

for f in "$REGION_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

for K in 8 16 24 32 48 64; do
    RMAP="$HIER_DIR/region_to_superregion_K${K}.json"
    if [[ ! -f "$RMAP" ]]; then
        echo "ERROR: $RMAP not found." >&2
        echo "       Run analyze_hard_memory_and_predictive_hierarchy.py first." >&2
        exit 1
    fi
done

if [[ ! -f "$KNN_RUN_DIR/per_position.npz" ]] && \
   [[ ! -f "$KNN_RUN_DIR/per_position_topk.npz" ]]; then
    echo "ERROR: per_position.npz not found in $KNN_RUN_DIR" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Run tuning ────────────────────────────────────────────────────────────────

python scripts/tune_hier_policies.py \
    --knn_run_dir  "$KNN_RUN_DIR"  \
    --hier_dir     "$HIER_DIR"     \
    --region_map   "$REGION_MAP"   \
    --output_dir   "$OUTPUT_DIR"   \
    --topk         32

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/tuned_policy_results.csv"
echo "  $OUTPUT_DIR/tuned_pareto_frontier.csv"
echo "  $OUTPUT_DIR/tuned_final_report.md"
echo ""

if [[ -f "$OUTPUT_DIR/tuned_final_report.md" ]]; then
    echo "=== Report excerpt ==="
    grep -A3 "Success Criteria\|beats router\|Strong\|High-cov" \
        "$OUTPUT_DIR/tuned_final_report.md" | head -30
fi
