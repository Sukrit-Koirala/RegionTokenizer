#!/bin/bash
#SBATCH --job-name=pairkey_knn
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Learned Pair-Key KNN Selector V1
#
# Trains PairKeyNet so that pair keys cluster by correction decision.
# Evaluates both parametric (score_head) and KNN (memory_score) selectors.
#
# Old KNN V1 (random proj): knn_sel_gold_bwcov=0.0070, rank2=0.1452, fv_gain=-0.0116
# V3 oracle: target_ctg=0.2407, fv_gain=+0.1584
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

RUN_NAME="learned_pairkey_hprime_v1"
OUT_ROOT="runs/learned_pairkey_knn_v1"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"

echo "========================================================"
echo " Learned Pair-Key KNN Selector V1"
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
    echo "  [WARN] super_map not found: $SUPER_MAP — superregion disabled"
fi

python -c "import faiss; print('  [OK] faiss available')" 2>/dev/null || \
    echo "  [WARN] faiss not found — using torch_chunked backend (slower)"

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUT_ROOT"
mkdir -p logs

echo "[launch] Starting learned pair-key KNN training..."
echo ""

python scripts/train_learned_pairkey_knn_v1.py \
    --small_ckpt             "$SMALL_CKPT"       \
    --train_dir              "$TRAIN_DIR"         \
    --val_dir                "$VAL_DIR"           \
    --token_to_region        "$TOKEN_TO_REGION"   \
    $SUPER_ARG                                    \
    --output_root            "${OUT_ROOT}"        \
    --run_name               "${RUN_NAME}"        \
    --top_k                  256                  \
    --candidate_filter       top_rank             \
    --candidate_pool_size    32                   \
    --max_train_rows         500000               \
    --max_pair_train_rows    200000               \
    --negatives_per_positive 8                    \
    --key_dim                128                  \
    --hidden_dim             512                  \
    --dropout                0.1                  \
    --batch_size             256                  \
    --lr                     1e-4                 \
    --steps                  5000                 \
    --eval_every             500                  \
    --eval_batch_size        256                  \
    --grad_clip              1.0                  \
    --lambda_margin          0.5                  \
    --target_margin          1.0                  \
    --knn_k                  32                   \
    --knn_tau                0.1                  \
    --thresholds             0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
    --param_thresholds       0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
    --margin_delta           1.0                  \
    --eval_full_vocab                             \
    --amp                                         \
    --seed                   42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: training exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Learned Pair-Key KNN V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${RUN_DIR}/report.md"
echo "  ${RUN_DIR}/final_metrics.json"
echo "  ${RUN_DIR}/param_threshold_sweep.csv"
echo "  ${RUN_DIR}/knn_threshold_sweep.csv"
echo "  ${RUN_DIR}/pair_keys_train.pt"
echo ""

if [ ! -f "${RUN_DIR}/final_metrics.json" ]; then
    echo "ERROR: final_metrics.json not found"
    exit 1
fi
if [ ! -f "${RUN_DIR}/param_threshold_sweep.csv" ]; then
    echo "ERROR: param_threshold_sweep.csv not found"
    exit 1
fi
if [ ! -f "${RUN_DIR}/knn_threshold_sweep.csv" ]; then
    echo "ERROR: knn_threshold_sweep.csv not found"
    exit 1
fi

python -c "
import json, os, math
rd = '${RUN_DIR}'
s  = json.load(open(os.path.join(rd, 'final_metrics.json')))

def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'nan'
    if isinstance(v, float): return f'{v:.4f}'
    return str(v)

print('=== Learned Pair-Key KNN V1 — Final Results ===')
print()

bl = s.get('baselines', {})
print('[baselines]')
for k in ['base_top1_acc','gold_in_pool_rate_val','baseline_rank2_sg_bwcov',
          'baseline_rank2_ctg','baseline_rank2_caw','baseline_oracle_pool_ctg']:
    if k in bl:
        print(f'  {k:40s} = {fmt(bl[k])}')
print()

bp = s.get('best_param', {}) or {}
print('[best parametric selector]')
for k in ['threshold','selected_gold_bwcov','selected_gold_in_pool',
          'changed_to_gold_rate','changed_away_rate','noharm_changed_away',
          'top1_acc_gain','full_vocab_gain']:
    if k in bp:
        print(f'  {k:40s} = {fmt(bp[k])}')
print()

bk = s.get('best_knn', {}) or {}
print('[best KNN selector]')
for k in ['threshold','selected_gold_bwcov','selected_gold_in_pool',
          'changed_to_gold_rate','changed_away_rate','noharm_changed_away',
          'top1_acc_gain','full_vocab_gain']:
    if k in bk:
        print(f'  {k:40s} = {fmt(bk[k])}')
print()

print('[reference]')
print('  old KNN V1  knn_sel_gold_bwcov           = 0.0070')
print('  old KNN V1  rank2_sel_gold_bwcov          = 0.1452')
print('  old KNN V1  best_fv_gain                  = -0.0116')
print('  V3 oracle   target_ctg                    = 0.2407')
print('  V3 oracle   fv_gain                       = +0.1584')
" 2>/dev/null || true
