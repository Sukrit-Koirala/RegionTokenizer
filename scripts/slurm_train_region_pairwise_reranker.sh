#!/bin/bash
#SBATCH --job-name=region_pairwise
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Region Pairwise Reranker V1 — Training
#
# Tests whether pairwise loss + region-aware transformer can flip gold
# above the wrong base-top-1 using only:
#   base logits, token embeddings, h_prime, learned token regions.
#
# No retrieval. No manual token features. No hand-coded categories.
#
# Evidence motivating this experiment:
#   covered_rate          = 0.8807
#   median_gold_rank_1b   = 2.0
#   top1_same_region_rate = 0.5801
#   base_correct_rate     = 0.3638
#
# Input dataset:
#   runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/
#
# Region map:
#   runs/region_maps_128/token_to_region.json
#
# Superregion map:
#   runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
#
# Output: runs/region_pairwise_reranker_v1/same_region_pair_v1/
#
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
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_ROOT="runs/region_pairwise_reranker_v1"

echo "========================================================"
echo " Region Pairwise Reranker V1"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking required inputs..."

if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: backbone checkpoint not found: $SMALL_CKPT"
    exit 1
fi
echo "  [OK] $SMALL_CKPT"

for D in "$TRAIN_DIR" "$VAL_DIR"; do
    if [ ! -d "$D" ]; then
        echo "ERROR: shard dir not found: $D"
        echo "  Run slurm_run_live_full_pipeline_rebuild.sh (Stage01B) first."
        exit 1
    fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then
        echo "ERROR: no shard_*.pt in $D"
        exit 1
    fi
    echo "  [OK] $D  ($N shards)"
done

if [ ! -f "$TOKEN_TO_REGION" ]; then
    echo "ERROR: token_to_region not found: $TOKEN_TO_REGION"
    exit 1
fi
echo "  [OK] $TOKEN_TO_REGION"

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map $SUPER_MAP"
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found: $SUPER_MAP"
    echo "         superregion_enabled = false — region-only mode"
    echo "         Superregion metrics will be skipped (not faked)."
fi

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_ROOT"
mkdir -p logs

echo "[launch] Starting training..."
echo ""

python scripts/train_region_pairwise_reranker.py \
    --small_ckpt        "$SMALL_CKPT"           \
    --train_dir         "$TRAIN_DIR"             \
    --val_dir           "$VAL_DIR"               \
    --token_to_region   "$TOKEN_TO_REGION"       \
    $SUPER_ARG                                   \
    --output_root       "$OUTPUT_ROOT"           \
    --run_name          same_region_pair_v1      \
    --top_k             256                      \
    --train_filter      same_region_or_superregion_confuser \
    --resolver_dim      256                      \
    --resolver_layers   2                        \
    --resolver_heads    4                        \
    --region_emb_dim    64                       \
    --super_emb_dim     32                       \
    --delta_scale       0.25                     \
    --lr                1e-5                     \
    --lambda_pair       2.0                      \
    --lambda_multi      0.5                      \
    --lambda_ce         0.25                     \
    --lambda_kl         1.0                      \
    --lambda_delta      1e-3                     \
    --num_confusers     8                        \
    --batch_size        32                       \
    --grad_accum_steps  2                        \
    --steps             3000                     \
    --eval_every        500                      \
    --eval_batch_size   64                       \
    --grad_clip         0.5                      \
    --eval_full_vocab                            \
    --amp                                        \
    --seed              42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: training exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Region Pairwise Reranker V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  $OUTPUT_ROOT/same_region_pair_v1/report.md"
echo "  $OUTPUT_ROOT/same_region_pair_v1/best_metrics.json"
echo "  $OUTPUT_ROOT/same_region_pair_v1/final_metrics.json"
echo "  $OUTPUT_ROOT/same_region_pair_v1/subset_eval_log.csv"
echo ""

python -c "
import json, os, math
run = '$OUTPUT_ROOT/same_region_pair_v1'
jp  = os.path.join(run, 'final_metrics.json')
if not os.path.exists(jp):
    print('final_metrics.json not found')
    exit(0)
res = json.load(open(jp))
subs = res.get('subsets', {})
fv   = res.get('fv', {})

def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'nan'
    if isinstance(v, float): return f'{v:.4f}'
    return str(v)

print('=== Final Val Metrics ===')
for sname in ['all', 'same_region_or_superregion_confuser', 'base_wrong_covered']:
    m = subs.get(sname, {})
    if not m: continue
    print(f'[{sname}]  n={m.get(\"n\", 0)}')
    for k in ['top1_acc_base','top1_acc_refined','top1_acc_gain',
              'pairwise_win_rate_base','pairwise_win_rate_refined',
              'changed_to_gold_rate','changed_away_rate',
              'candidate_ce_base','candidate_ce_refined','candidate_ce_gain',
              'mean_pair_margin_base','mean_pair_margin_refined']:
        if k in m:
            print(f'  {k:40s} = {fmt(m[k])}')
    print()

if fv:
    print('[full_vocab]')
    for k in ['full_vocab_base_nll','full_vocab_refined_nll','full_vocab_gain',
              'full_vocab_top1_acc_base','full_vocab_top1_acc_refined']:
        if k in fv:
            print(f'  {k:40s} = {fmt(fv[k])}')
" 2>/dev/null || true
