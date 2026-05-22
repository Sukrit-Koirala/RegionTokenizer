#!/bin/bash
#SBATCH --job-name=region_gold_rank
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Region Gold Rank Audit — Pre-reranker diagnostic
#
# Reads patched val shards from Stage01B. NO model loaded. NO retrieval.
# Maps gold/topK tokens through token_to_region → region_to_superregion,
# handling unmapped tokens explicitly as UNK (never dropped).
#
# IMPORTANT RULES enforced by audit_region_gold_rank.py:
#   1. No training, no retrieval, no manual token classes.
#   2. Only uses: base_topk_ids, base_topk_lgt, gold_token, input_ids, row_id.
#   3. Unmapped tokens reported as UNK — never silently dropped.
#   4. All ranks are rank-in-topK (NOT full-vocab rank).
#
# Inputs:
#   val shards:        runs/live_full_pipeline_rebuild_limited500k_logitfix/
#                      01_live_dataset_patched/val/
#   token_to_region:   runs/region_maps_128/token_to_region.json
#   super_map:         runs/region_maps_128/region_to_superregion.json
#
# Output: runs/region_gold_rank_audit/
#   summary.md
#   subset_metrics.csv
#   gold_rank_histograms.csv
#   topk_region_curves.csv
#   examples_gold_rank_1.md
#   examples_covered_wrong_base.md
#   examples_covered_high_region_rank.md
#   examples_not_covered.md
#   audit_results.json
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$SLURM_SUBMIT_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

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
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/region_maps_128/region_to_superregion.json"
OUTPUT_ROOT="runs/region_gold_rank_audit"

echo "========================================================"
echo " Region Gold Rank Audit"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking required inputs..."

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: val shard dir not found: $VAL_DIR"
    echo "  Run slurm_run_live_full_pipeline_rebuild.sh (Stage01B) first."
    exit 1
fi
N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_VAL" -eq 0 ]; then
    echo "ERROR: no shard_*.pt found in $VAL_DIR"
    exit 1
fi
echo "  [OK] $VAL_DIR  ($N_VAL shards)"

if [ ! -f "$TOKEN_TO_REGION" ]; then
    echo "ERROR: token_to_region not found: $TOKEN_TO_REGION"
    exit 1
fi
echo "  [OK] $TOKEN_TO_REGION"

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map $SUPER_MAP"
    echo "  [OK] $SUPER_MAP  (superregion mapping enabled)"
else
    echo "  [WARN] super_map not found: $SUPER_MAP  (superregion metrics will be skipped)"
fi

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_ROOT"

echo "[launch] Starting region gold rank audit..."
echo ""

python scripts/audit_region_gold_rank.py \
    --val_dir           "$VAL_DIR"           \
    --token_to_region   "$TOKEN_TO_REGION"   \
    $SUPER_ARG                               \
    --output_root       "$OUTPUT_ROOT"       \
    --top_k             256                  \
    --ks                1,2,4,8,16,32,64,128,256 \
    --num_examples      50                   \
    --tokenizer         gpt2

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: audit exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Region Gold Rank Audit complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  $OUTPUT_ROOT/summary.md"
echo "  $OUTPUT_ROOT/subset_metrics.csv"
echo "  $OUTPUT_ROOT/gold_rank_histograms.csv"
echo "  $OUTPUT_ROOT/topk_region_curves.csv"
echo ""

python -c "
import json, os, math
jp = '$OUTPUT_ROOT/audit_results.json'
if not os.path.exists(jp):
    print('audit_results.json not found')
    exit(0)
res = json.load(open(jp))
m = res.get('val', {}).get('all', {})
if not m:
    print('no val/all metrics in audit_results.json')
    exit(0)
def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'nan'
    if isinstance(v, float): return f'{v:.4f}'
    return str(v)
print('Val metrics (all rows):')
for k in ['n','base_correct_rate','covered_rate','top1_same_region_rate',
          'top1_same_super_rate','gold_unmapped_rate','top1_unmapped_rate',
          'mean_gold_rank_1b','median_gold_rank_1b',
          'mean_region_local_rank_1b','median_region_local_rank_1b']:
    print(f'  {k:40s} = {fmt(m.get(k))}')
" 2>/dev/null || true
