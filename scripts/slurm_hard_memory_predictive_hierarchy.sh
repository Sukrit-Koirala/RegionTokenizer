#!/bin/bash
#SBATCH --job-name=hard_mem_hier
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Hard-Position Memory Controller + Type-B Predictive Hierarchy Analysis.
#
# Prerequisites:
#   per_position.npz must exist in KNN_RUN_DIR.
#   If missing, re-run offline_region_knn.py with FORCE_RUN=1 (memory loads
#   from cache — fast).
#
# Outputs: runs/hard_memory_predictive_hierarchy/

source ~/miniconda3/bin/activate
conda activate learned_regions

set -eo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
OUTPUT_DIR=runs/hard_memory_predictive_hierarchy

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Hard-Memory Predictive Hierarchy  $(date) ==="
echo "    KNN_RUN_DIR : $KNN_RUN_DIR"
echo "    REGION_MAP  : $REGION_MAP"
echo "    OUTPUT_DIR  : $OUTPUT_DIR"

for f in "$REGION_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

if [[ ! -d "$KNN_RUN_DIR" ]]; then
    echo "ERROR: KNN_RUN_DIR not found: $KNN_RUN_DIR" >&2; exit 1
fi

if [[ ! -f "$KNN_RUN_DIR/per_position.npz" ]]; then
    echo "ERROR: per_position.npz not found in $KNN_RUN_DIR" >&2
    echo "       Re-run offline_region_knn.py with FORCE_RUN=1:" >&2
    echo "         FORCE_RUN=1 python scripts/offline_region_knn.py \\" >&2
    echo "             --output_dir $KNN_RUN_DIR  [same args as original run]" >&2
    echo "       Memory loads from cache (fast — skips expensive build step)." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Run analysis ──────────────────────────────────────────────────────────────

python scripts/analyze_hard_memory_and_predictive_hierarchy.py \
    --knn_run_dir    "$KNN_RUN_DIR"  \
    --region_map_path "$REGION_MAP" \
    --output_dir     "$OUTPUT_DIR"  \
    --topk           32             \
    --device         cuda           \
    --seed           42

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo ""
echo "Outputs:"
echo "  $OUTPUT_DIR/hard_position_controller_results.csv"
echo "  $OUTPUT_DIR/typeB_coarse_conversion.csv"
echo "  $OUTPUT_DIR/typeB_subtype_breakdown.csv"
echo "  $OUTPUT_DIR/hierarchical_candidate_policies.csv"
echo "  $OUTPUT_DIR/final_report.md"
echo "  $OUTPUT_DIR/plots/"
echo ""

if [[ -f "$OUTPUT_DIR/final_report.md" ]]; then
    echo "=== Report excerpt ==="
    grep -A2 "RECOMMENDATION\|Q8\|YES\|NO" "$OUTPUT_DIR/final_report.md" | head -20
fi
