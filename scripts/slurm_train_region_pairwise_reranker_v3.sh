#!/bin/bash
#SBATCH --job-name=region_pair_v3
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
# Region Pairwise Reranker V3 — Surgical Oracle Diagnostic
#
# Hypothesis: V1/V2 edited all 256 candidate logits → collateral damage.
# Test: surgically edit ONLY gold vs base_top1 logits on known confuser rows.
#
#   refined[gold_idx]  += +0.5 * margin_delta
#   refined[base_top1] += -0.5 * margin_delta
#   all other logits unchanged
#
# ⚠️  ORACLE DIAGNOSTIC: uses gold_idx at eval time. Not deployable at inference.
#    Purpose: test if pairwise features can learn useful correction when the
#    pair is known. If yes → problem is candidate selection, not pairwise signal.
#
# V1/V2 reference:
#   V1: changed_to_gold=0.1018  pairwise_win_ref=0.1379  fv_gain=-0.0050
#   V2: changed_to_gold=0.0193  pairwise_win_ref=0.0299  fv_gain=-0.0105 (gate saturated)
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

RUN_NAME="surgical_pair_v1"
OUT_ROOT="runs/region_pairwise_reranker_v3"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"

echo "========================================================"
echo " Region Pairwise Reranker V3 — Surgical Oracle"
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

echo "[launch] Starting V3 surgical oracle training..."
echo ""

python scripts/train_region_pairwise_reranker_v3.py \
    --small_ckpt        "$SMALL_CKPT"           \
    --train_dir         "$TRAIN_DIR"             \
    --val_dir           "$VAL_DIR"               \
    --token_to_region   "$TOKEN_TO_REGION"       \
    $SUPER_ARG                                   \
    --output_root       "$OUT_ROOT"              \
    --run_name          "$RUN_NAME"              \
    --top_k             256                      \
    --apply_policy      target_only              \
    --region_emb_dim    64                       \
    --super_emb_dim     32                       \
    --hidden_dim        256                      \
    --n_hidden          3                        \
    --margin_delta_scale 1.0                     \
    --target_fraction   0.75                     \
    --lr                1e-4                     \
    --lambda_pair       2.0                      \
    --lambda_positive   0.1                      \
    --lambda_margin_delta 1e-3                   \
    --batch_size        64                       \
    --grad_accum_steps  1                        \
    --steps             2000                     \
    --eval_every        500                      \
    --eval_batch_size   128                      \
    --grad_clip         1.0                      \
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
echo " Region Pairwise Reranker V3 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${RUN_DIR}/report.md"
echo "  ${RUN_DIR}/best_metrics.json"
echo "  ${RUN_DIR}/final_metrics.json"
echo "  ${RUN_DIR}/subset_eval_log.csv"
echo "  ${RUN_DIR}/examples_oracle_flipped_to_gold.md"
echo "  ${RUN_DIR}/examples_oracle_failed.md"
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

print('=== Final Val Metrics (V3 Surgical Oracle) ===')
print()
for sname in ['all', 'target_confuser', 'base_correct_covered']:
    m = subs.get(sname, {})
    if not m: continue
    print(f'[{sname}]  n={m.get(\"n\", 0)}')
    if sname == 'all':
        keys = ['top1_acc_base','top1_acc_refined','top1_acc_gain',
                'applied_rate','mean_margin_delta','mean_abs_margin_delta']
    elif sname == 'target_confuser':
        keys = ['applied_rate','changed_to_gold_rate','changed_away_rate',
                'pairwise_win_rate_base','pairwise_win_rate_refined',
                'mean_pair_margin_base','mean_pair_margin_refined',
                'mean_margin_delta','mean_abs_margin_delta']
    else:
        keys = ['changed_away_rate','applied_rate']
    for k in keys:
        if k in m:
            print(f'  {k:40s} = {fmt(m[k])}')
    print()

m_tgt = subs.get('target_confuser', {})
m_nh  = subs.get('base_correct_covered', {})
ctg = m_tgt.get('changed_to_gold_rate', float('nan'))
pwr = m_tgt.get('pairwise_win_rate_refined', float('nan'))
caw = m_nh.get('changed_away_rate', float('nan'))
if not any(v != v for v in [ctg, pwr, caw]):
    sc = ctg + 0.25*pwr - 2.0*caw
    print(f'checkpoint_score = {sc:.4f}')
    print(f'  (ctg={fmt(ctg)} + 0.25*pwr={fmt(pwr)} - 2*caw={fmt(caw)})')
print()

if fv:
    print('[full_vocab (oracle policy)]')
    for k in ['oracle_policy_full_vocab_gain',
              'full_vocab_base_nll','full_vocab_refined_nll',
              'full_vocab_top1_acc_base','full_vocab_top1_acc_refined']:
        if k in fv:
            print(f'  {k:40s} = {fmt(fv[k])}')
    print()
    print('[V1/V2 reference]')
    print('  V1  changed_to_gold                     = 0.1018')
    print('  V1  pairwise_win_ref                     = 0.1379')
    print('  V1  full_vocab_gain                      = -0.0050')
    print('  V2  changed_to_gold                      = 0.0193 (gate saturated)')
    print('  V2  full_vocab_gain                      = -0.0105')
" 2>/dev/null || true
