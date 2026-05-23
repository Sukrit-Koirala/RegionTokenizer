#!/bin/bash
#SBATCH --job-name=local_detail_sel
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
# Local Detail Memory Candidate Selector V1
#
# Stage 2: candidates attend over recent context tokens via cross-attention.
# Fixes Stage 1 missing NO_OP problem: base-correct rows are included in training.
#
# Stage 1 reference: sel_gold_bwcov≈0.246, ctg≈0.040, caw≈0.062
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

RUN_NAME="cand_xattn_tokmem_v1"
OUT_ROOT="runs/local_detail_selector_v1"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"

echo "========================================================"
echo " Local Detail Memory Candidate Selector V1"
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

echo "[launch] Starting local detail selector training..."
echo ""

python scripts/train_local_detail_selector_v1.py \
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
    --memory_len            128                     \
    --resolver_dim          256                     \
    --hidden_dim            512                     \
    --attention_heads       4                       \
    --dropout               0.1                     \
    --region_emb_dim        64                      \
    --super_emb_dim         32                      \
    --batch_size            128                     \
    --candidate_fraction    0.5                     \
    --lr                    1e-4                    \
    --steps                 5000                    \
    --eval_every            500                     \
    --eval_batch_size       256                     \
    --grad_clip             1.0                     \
    --noop_weight           0.5                     \
    --candidate_weight      1.0                     \
    --lambda_margin_loss    0.25                    \
    --target_margin         1.0                     \
    --margin_delta          1.0                     \
    --thresholds            0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
    --uncovered_policy      ignore                  \
    --max_train_rows        500000                  \
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
echo " Local Detail Selector V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${RUN_DIR}/report.md"
echo "  ${RUN_DIR}/final_metrics.json"
echo "  ${RUN_DIR}/threshold_sweep.csv"
echo "  ${RUN_DIR}/best_selector.pt"
echo "  ${RUN_DIR}/examples_attention_debug.md"
echo ""

if [ ! -f "${RUN_DIR}/final_metrics.json" ]; then
    echo "ERROR: final_metrics.json not found at ${RUN_DIR}/final_metrics.json"
    exit 1
fi
if [ ! -f "${RUN_DIR}/report.md" ]; then
    echo "ERROR: report.md not found at ${RUN_DIR}/report.md"
    exit 1
fi
if [ ! -f "${RUN_DIR}/threshold_sweep.csv" ]; then
    echo "ERROR: threshold_sweep.csv not found at ${RUN_DIR}/threshold_sweep.csv"
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

print('=== Local Detail Selector V1 — Final Results ===')
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

ck = s.get('checkpoint', {})
if ck:
    print('[checkpoint provenance]')
    for k in ['best_step_during_training','loaded_best_step_for_final_eval',
              'final_eval_uses_best_checkpoint']:
        if k in ck:
            print(f'  {k:45s} = {fmt(ck[k])}')
    print()

print('[reference]')
print('  Stage 1  sel_gold_bwcov                       = 0.246')
print('  Stage 1  ctg                                  = 0.040')
print('  Stage 1  caw                                  = 0.062')
print('  KNN V1   knn_sel_gold_bwcov                   = 0.0070')
print('  KNN V1   rank2_sel_gold_bwcov                 = 0.1452')
print('  KNN V1   best_fv_gain                         = -0.0116')
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
