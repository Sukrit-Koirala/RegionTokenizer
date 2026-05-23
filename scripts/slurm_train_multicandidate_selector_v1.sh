#!/bin/bash
#SBATCH --job-name=mcand_sel
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
# Multi-Candidate Selector V1 — Learned non-oracle candidate selection
#
# Replaces KNN pair memory with a learned MLP that scores:
#   action ∈ {NO_OP, candidate_0 .. candidate_{P-1}}
# No gold is used at eval time.
# Surgical correction: selected += +0.5*md, base_top1 -= 0.5*md.
#
# Reference:
#   KNN V1:  knn_sel_gold_bwcov=0.0070, rank2_sel=0.1452, fv_gain=-0.0116
#   V3 oracle:  target_ctg=0.2407, oracle_fv_gain=+0.1584
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

RUN_NAME="noop_candidate_selector_v1"
OUT_ROOT="runs/multicandidate_selector_v1"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"

echo "========================================================"
echo " Multi-Candidate Selector V1"
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

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUT_ROOT"
mkdir -p logs

echo "[launch] Starting multi-candidate selector training..."
echo ""

python scripts/train_multicandidate_selector_v1.py \
    --small_ckpt            "$SMALL_CKPT"          \
    --train_dir             "$TRAIN_DIR"            \
    --val_dir               "$VAL_DIR"              \
    --token_to_region       "$TOKEN_TO_REGION"      \
    $SUPER_ARG                                      \
    --output_root           "${OUT_ROOT}"           \
    --run_name              "${RUN_NAME}"           \
    --top_k                 256                     \
    --candidate_filter      top_rank                \
    --candidate_pool_size   32                      \
    --architecture          mlp                     \
    --hidden_dim            256                     \
    --layers                3                       \
    --dropout               0.1                     \
    --batch_size            128                     \
    --lr                    1e-4                    \
    --steps                 5000                    \
    --eval_every            500                     \
    --eval_batch_size       256                     \
    --grad_clip             1.0                     \
    --noop_weight           0.5                     \
    --candidate_weight      1.0                     \
    --lambda_margin_loss    0.5                     \
    --target_margin         0.0                     \
    --margin_delta          1.0                     \
    --uncovered_policy      ignore                  \
    --eval_full_vocab                               \
    --amp                                           \
    --seed                  42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: training exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Multi-Candidate Selector V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${RUN_DIR}/report.md"
echo "  ${RUN_DIR}/final_metrics.json"
echo "  ${RUN_DIR}/best_metrics.json"
echo "  ${RUN_DIR}/best_selector.pt"
echo "  ${RUN_DIR}/pretrain_baseline.json"
echo ""

if [ ! -f "${RUN_DIR}/final_metrics.json" ]; then
    echo "ERROR: final_metrics.json not found at ${RUN_DIR}/final_metrics.json"
    exit 1
fi
if [ ! -f "${RUN_DIR}/report.md" ]; then
    echo "ERROR: report.md not found at ${RUN_DIR}/report.md"
    exit 1
fi

python -c "
import json, os, math
rd = '${RUN_DIR}'
jp = os.path.join(rd, 'final_metrics.json')
s  = json.load(open(jp))
subs = s.get('subsets', {})
fv   = s.get('fv', {})

def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'nan'
    if isinstance(v, float): return f'{v:.4f}'
    return str(v)

print('=== Multi-Candidate Selector V1 — Final Results ===')
print()

sa = subs.get('all', {})
si = subs.get('gold_in_pool', {})
sb = subs.get('base_correct', {})
st = subs.get('target_confuser', {})

print('[all]')
for k in ['n','apply_rate','noop_rate','base_top1_acc','refined_top1_acc',
          'top1_acc_gain','changed_to_gold_rate','changed_away_rate']:
    if k in sa:
        print(f'  {k:45s} = {fmt(sa[k])}')
print()

print('[gold_in_pool]')
for k in ['n','selected_gold_rate_in_pool','apply_rate']:
    if k in si:
        print(f'  {k:45s} = {fmt(si[k])}')
print()

print('[base_correct (no-harm)]')
for k in ['n','false_apply_on_base_correct','changed_away_on_base_correct']:
    if k in sb:
        print(f'  {k:45s} = {fmt(sb[k])}')
print()

print('[target_confuser]')
for k in ['n','changed_to_gold_rate','selected_gold_rate_in_pool']:
    if k in st:
        print(f'  {k:45s} = {fmt(st[k])}')
print()

if fv:
    print('[full_vocab]')
    for k in ['full_vocab_gain','full_vocab_base_nll','full_vocab_refined_nll',
              'full_vocab_top1_acc_base','full_vocab_top1_acc_refined']:
        if k in fv:
            print(f'  {k:45s} = {fmt(fv[k])}')
    print()

print('[reference]')
print('  KNN V1  knn_sel_gold_bwcov                    = 0.0070')
print('  KNN V1  rank2_sel_gold_bwcov                  = 0.1452')
print('  KNN V1  best_fv_gain                          = -0.0116')
print('  V3 oracle  target_ctg                         = 0.2407')
print('  V3 oracle  fv_gain                            = +0.1584')

bl_path = os.path.join(rd, 'pretrain_baseline.json')
if os.path.exists(bl_path):
    bl = json.load(open(bl_path))
    print()
    print('[pretrain baselines]')
    for k in ['baseline_oracle_pool_ctg','baseline_rank2_ctg','baseline_rank2_caw']:
        if k in bl and bl[k] is not None:
            print(f'  {k:45s} = {fmt(bl[k])}')
" 2>/dev/null || true
