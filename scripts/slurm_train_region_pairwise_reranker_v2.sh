#!/bin/bash
#SBATCH --job-name=region_pair_v2
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
# Region Pairwise Reranker V2 — Gated Delta + No-harm Training
#
# Key changes from V1:
#   - Gated delta: actual_delta = delta_scale * sigmoid(gate) * tanh(raw_delta)
#   - Mixed batches: 50% target (same-region confusers) + 50% noharm (base-correct)
#   - No-harm losses: noharm CE + margin preservation + gate sparsity
#   - Checkpoint metric: noharm_adjusted_score = ctg - 2*caw + 0.25*pwr
#
# V1 reference (same_region_pair_v1):
#   pairwise_win_rate_refined = 0.1379
#   changed_to_gold_rate      = 0.1018
#   full_vocab_gain           = -0.0050
#   top1_acc_drop             = -0.0101
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

RUN_NAME="gated_same_region_pair_v1"
OUT_ROOT="runs/region_pairwise_reranker_v2"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"

echo "========================================================"
echo " Region Pairwise Reranker V2"
echo " run_name: ${RUN_NAME}"
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
    echo "         superregion_enabled = false — superregion metrics skipped, not faked."
fi

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUT_ROOT"
mkdir -p logs

echo "[launch] Starting V2 training..."
echo ""

python scripts/train_region_pairwise_reranker_v2.py \
    --small_ckpt        "$SMALL_CKPT"           \
    --train_dir         "$TRAIN_DIR"             \
    --val_dir           "$VAL_DIR"               \
    --token_to_region   "$TOKEN_TO_REGION"       \
    $SUPER_ARG                                   \
    --output_root       "$OUT_ROOT"              \
    --run_name          "$RUN_NAME"              \
    --top_k             256                      \
    --resolver_dim      256                      \
    --resolver_layers   2                        \
    --resolver_heads    4                        \
    --region_emb_dim    64                       \
    --super_emb_dim     32                       \
    --delta_scale       0.25                     \
    --gate_bias_init    -4.0                     \
    --target_fraction   0.5                      \
    --lr                1e-5                     \
    --lambda_pair       2.0                      \
    --lambda_multi      0.25                     \
    --lambda_ce         0.25                     \
    --lambda_kl         2.0                      \
    --lambda_delta      5e-3                     \
    --lambda_noharm_ce  1.0                      \
    --lambda_noharm_margin 1.0                   \
    --lambda_gate       1e-3                     \
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
echo " Region Pairwise Reranker V2 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${RUN_DIR}/report.md"
echo "  ${RUN_DIR}/best_metrics.json"
echo "  ${RUN_DIR}/final_metrics.json"
echo "  ${RUN_DIR}/subset_eval_log.csv"
echo "  ${RUN_DIR}/examples_target_flipped_to_gold.md"
echo "  ${RUN_DIR}/examples_noharm_changed_away.md"
echo ""

if [ ! -f "${RUN_DIR}/final_metrics.json" ]; then
    echo "WARNING: final_metrics.json not found at ${RUN_DIR}/final_metrics.json"
    exit 1
fi

python -c "
import json, os, math
rd  = '${RUN_DIR}'
jp  = os.path.join(rd, 'final_metrics.json')
res = json.load(open(jp))
subs = res.get('subsets', {})
fv   = res.get('fv', {})

def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'nan'
    if isinstance(v, float): return f'{v:.4f}'
    return str(v)

print('=== Final Val Metrics (V2) ===')
print()
for sname in ['all', 'target_confuser', 'base_correct_covered']:
    m = subs.get(sname, {})
    if not m: continue
    print(f'[{sname}]  n={m.get(\"n\", 0)}')
    keys = []
    if sname == 'all':
        keys = ['top1_acc_base','top1_acc_refined','top1_acc_gain',
                'mean_gate','max_gate','mean_actual_delta_abs']
    elif sname == 'target_confuser':
        keys = ['pairwise_win_rate_base','pairwise_win_rate_refined',
                'changed_to_gold_rate','changed_away_rate',
                'mean_pair_margin_base','mean_pair_margin_refined',
                'mean_gate','mean_actual_delta_abs']
    else:
        keys = ['changed_away_rate','margin_eroded_rate',
                'mean_margin_change','mean_gate','mean_actual_delta_abs']
    for k in keys:
        if k in m:
            print(f'  {k:40s} = {fmt(m[k])}')
    print()

nhs = subs.get('target_confuser', {})
nhnh = subs.get('base_correct_covered', {})
ctg = nhs.get('changed_to_gold_rate', float('nan'))
caw = nhnh.get('changed_away_rate', float('nan'))
pwr = nhs.get('pairwise_win_rate_refined', float('nan'))
if not any(v != v for v in [ctg, caw, pwr]):
    sc = ctg - 2.0*caw + 0.25*pwr
    print(f'noharm_adjusted_score = {sc:.4f}')
    print(f'  (ctg={fmt(ctg)} - 2*caw={fmt(caw)} + 0.25*pwr={fmt(pwr)})')
print()

if fv:
    print('[full_vocab]')
    for k in ['full_vocab_base_nll','full_vocab_refined_nll','full_vocab_gain',
              'full_vocab_top1_acc_base','full_vocab_top1_acc_refined']:
        if k in fv:
            print(f'  {k:40s} = {fmt(fv[k])}')
    print()
    print('[V1 reference]')
    print('  full_vocab_gain (V1)                     = -0.0050')
    print('  top1_acc_drop   (V1)                     = -0.0101')
" 2>/dev/null || true
