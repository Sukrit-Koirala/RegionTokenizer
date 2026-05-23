#!/bin/bash
#SBATCH --job-name=knn_pair_sel
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# KNN Pair Selector V1 — Non-oracle pairwise candidate selection
#
# Motivation:
#   V3 oracle surgical correction (target_ctg=0.2407, fv_gain=+0.1584)
#   used gold_idx at eval to pick which pair to correct — not deployable.
#   This experiment replaces the oracle with kNN pair memory:
#
#     For each val row:
#       candidates c = region-filtered topK[1:] (no gold used)
#       memory_score(c) = kNN support from train pair (pos/neg) records
#       c* = argmax memory_score
#       if memory_score(c*) >= threshold:
#         c* += +0.5 * margin_delta
#         base_top1 -= -0.5 * margin_delta
#
# Success: knn_sel_gold > rank2_sel_gold + positive full_vocab_gain
# Failure: knn_sel_gold ~ rank2/random baseline → pair memory not enough
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

RUN_NAME="pair_memory_hprime_region_v1"
OUT_ROOT="runs/knn_pair_selector_v1"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"

echo "========================================================"
echo " KNN Pair Selector V1"
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
    echo "  [WARN] super_map not found: $SUPER_MAP"
    echo "         superregion features disabled — region-only mode"
fi

# Check for FAISS (optional)
python -c "import faiss; print('  [OK] faiss available')" 2>/dev/null || \
    echo "  [WARN] faiss not found — using torch_chunked backend (slower)"

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUT_ROOT"
mkdir -p logs

echo "[launch] Starting KNN pair selector..."
echo ""

python scripts/run_knn_pair_selector_v1.py \
    --small_ckpt         "$SMALL_CKPT"          \
    --train_dir          "$TRAIN_DIR"            \
    --val_dir            "$VAL_DIR"              \
    --token_to_region    "$TOKEN_TO_REGION"      \
    $SUPER_ARG                                   \
    --output_root        "${OUT_ROOT}"           \
    --run_name           "${RUN_NAME}"           \
    --top_k              256                     \
    --store_filter       same_region_or_superregion_confuser \
    --candidate_filter   same_region_or_superregion          \
    --candidate_pool_size 32                     \
    --max_store_rows     500000                  \
    --max_pair_records   1000000                 \
    --negatives_per_positive 8                   \
    --key_dim            256                     \
    --knn_k              32                      \
    --knn_tau            0.1                     \
    --backend            auto                    \
    --query_batch_pairs  4096                    \
    --index_chunk_size   131072                  \
    --margin_delta       1.0                     \
    --thresholds         0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8 \
    --eval_full_vocab                            \
    --num_examples       40                      \
    --seed               42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: run exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " KNN Pair Selector V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${RUN_DIR}/summary.md"
echo "  ${RUN_DIR}/threshold_sweep.csv"
echo "  ${RUN_DIR}/pair_keys_train.pt"
echo "  ${RUN_DIR}/examples_selected_gold.md"
echo ""

if [ ! -f "${RUN_DIR}/summary.md" ]; then
    echo "ERROR: summary.md not found at ${RUN_DIR}/summary.md"
    exit 1
fi
if [ ! -f "${RUN_DIR}/threshold_sweep.csv" ]; then
    echo "ERROR: threshold_sweep.csv not found at ${RUN_DIR}/threshold_sweep.csv"
    exit 1
fi

python -c "
import json, os, math
rd  = '${RUN_DIR}'
jp  = os.path.join(rd, 'summary.json')
if not os.path.exists(jp):
    print('summary.json not found'); exit(0)
s = json.load(open(jp))

def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'nan'
    if isinstance(v, float): return f'{v:.4f}'
    return str(v)

print('=== KNN Pair Selector V1 — Final Results ===')
print()
ds = s.get('datastore_meta', {})
print(f'[datastore]  pos={ds.get(\"n_pos\",\"?\")}  neg={ds.get(\"n_neg\",\"?\")}  '
      f'total={ds.get(\"n_total\",\"?\")}  eligible_rows={ds.get(\"n_eligible_rows\",\"?\")}')
print()

rep = s.get('rep_metrics', {})
fv  = s.get('rep_fv', {})
thr = s.get('rep_threshold', '?')
print(f'[threshold={thr}]')
for k in ['apply_rate','changed_to_gold_rate','changed_away_rate',
          'knn_sel_gold_bwcov','rank2_sel_gold_bwcov','random_sel_gold_bwcov',
          'gold_in_pool_rate','gold_top4_mem_rate',
          'target_sel_gold_rate','target_ctg_rate',
          'noharm_changed_away']:
    if k in rep:
        print(f'  {k:40s} = {fmt(rep[k])}')
print()

if fv:
    print('[full_vocab]')
    for k in ['full_vocab_gain','full_vocab_base_nll','full_vocab_refined_nll',
              'full_vocab_top1_acc_base','full_vocab_top1_acc_refined']:
        if k in fv:
            print(f'  {k:40s} = {fmt(fv[k])}')
    print()

print('[V3 oracle reference]')
print('  target_changed_to_gold                   = 0.2407')
print('  oracle_full_vocab_gain                   = +0.1584')
print()
bfv = s.get('best_fv_gain', float('nan'))
bsc = s.get('best_score',   float('nan'))
print(f'best_fv_gain = {fmt(bfv)} at threshold={s.get(\"best_fv_threshold\")}')
print(f'best_ctg-2caw = {fmt(bsc)} at threshold={s.get(\"best_score_threshold\")}')
" 2>/dev/null || true
