#!/bin/bash
#SBATCH --job-name=learn_detail_mem
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
# Learned Detail Memory Resolver V1
#
# Trains a cross-attention model that uses document-past memory
# (corpus[token_offset-2048:token_offset]) and prefix context to resolve
# exact name/entity/number candidates — purely learned, no manual rules.
#
# NO hand-coded heuristics.  NO token-type rules as model inputs.
# Val gold used ONLY for CE loss and metrics.
# Gold NEVER used to construct memory or candidates.
#
# Five ablation modes evaluated each epoch and at final eval:
#   base_only              — delta=0 (sanity check)
#   prefix_memory_only     — memory = prefix[-128:] only
#   doc_past_memory_only   — memory = corpus[offset-2048:offset] only
#   prefix_plus_doc_past   — memory = doc_past + prefix (main mode)
#   shuffled_doc_past_control — roll(doc_past,1,batch) + prefix (control)
#
# Best checkpoint saved only if model_acc > base_acc.
# If no improvement: best_metrics.json contains no_improving_checkpoint_found=true.
#
# Outputs:
#   runs/entity_number_memory_v1/learned_detail_resolver_v1/
#     config.json
#     train_log.csv
#     eval_log.csv
#     eval_by_slice.csv
#     ablation_results.csv
#     best_model.pt           (only written if model beats base)
#     best_metrics.json
#     final_metrics.json
#     report.md
#     examples_detail_helps.md
#     examples_detail_hurts.md
#     examples_attn_memory_top.md
#     examples_shuffled_vs_real.md
#     examples_no_doc_past.md
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1
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

TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
CORPUS_PATH="runs/live_full_pipeline_rebuild_limited500k_logitfix/00_raw_token_source/train_tokens.npy"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/entity_number_memory_v1/learned_detail_resolver_v1"

# Optional: path to GPT-2 checkpoint for frozen token embedding init.
# If absent, random-init embeddings are used (model still trains fine; gains may be lower).
# Common locations:
#   runs/live_full_pipeline_rebuild_limited500k_logitfix/checkpoints/best_model.pt
#   checkpoints/gpt2_base.pt
CHECKPOINT=""   # <- fill in if available; leave empty to skip

echo "========================================================"
echo " Learned Detail Memory Resolver V1"
echo " train_dir:  ${TRAIN_DIR}"
echo " val_dir:    ${VAL_DIR}"
echo " output_dir: ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

echo "[preflight] Checking required inputs..."

if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: train_dir not found: $TRAIN_DIR"; exit 1
fi
N_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_TRAIN" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $TRAIN_DIR"; exit 1
fi
echo "  [OK] $TRAIN_DIR  ($N_TRAIN train shards)"

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: val_dir not found: $VAL_DIR"; exit 1
fi
N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_VAL" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $VAL_DIR"; exit 1
fi
echo "  [OK] $VAL_DIR  ($N_VAL val shards)"

CORPUS_ARG=""
if [ -f "$CORPUS_PATH" ]; then
    CORPUS_ARG="--corpus_path $CORPUS_PATH"
    echo "  [OK] $CORPUS_PATH  (train corpus)"
else
    echo "  [WARN] corpus not found: $CORPUS_PATH"
    echo "         Will try pipeline-cache auto-detect; doc_past may be zeros if both absent."
fi

T2R_ARG=""
if [ -f "$TOKEN_TO_REGION" ]; then
    T2R_ARG="--token_to_region $TOKEN_TO_REGION"
    echo "  [OK] $TOKEN_TO_REGION"
else
    echo "  [WARN] token_to_region not found — region slices disabled"
fi

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map $SUPER_MAP"
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found — superregion slices disabled"
fi

CKPT_ARG=""
if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
    CKPT_ARG="--checkpoint $CHECKPOINT"
    echo "  [OK] $CHECKPOINT  (token emb init)"
else
    echo "  [WARN] no checkpoint supplied — random-init token embeddings"
fi

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/train_learned_detail_memory_resolver_v1.py
echo "  [OK] scripts/train_learned_detail_memory_resolver_v1.py compiles"
bash -n scripts/slurm_train_learned_detail_memory_resolver_v1.sh
echo "  [OK] slurm_train_learned_detail_memory_resolver_v1.sh syntax"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting Learned Detail Memory Resolver V1 training..."
echo "  NOTE: Trains only on rows where gold is naturally in pool."
echo "  NOTE: Val gold used only for CE loss computation and metrics."
echo "  NOTE: No token-type heuristics or manual rules as model inputs."
echo ""

python scripts/train_learned_detail_memory_resolver_v1.py \
    --train_dir              "${TRAIN_DIR}" \
    --val_dir                "${VAL_DIR}" \
    $CORPUS_ARG \
    $T2R_ARG \
    $SUPER_ARG \
    --output_dir             "${OUTPUT_DIR}" \
    --doc_memory_len         2048 \
    --prefix_len             128 \
    --candidate_pool_size    32 \
    $CKPT_ARG \
    --emb_dim                768 \
    --memory_dim             256 \
    --num_heads              4 \
    --n_attn_layers          2 \
    --mlp_hidden             512 \
    --dropout                0.1 \
    --epochs                 5 \
    --lr                     1e-4 \
    --weight_decay           1e-4 \
    --train_batch_size       256 \
    --val_batch_size         512 \
    --max_examples           50 \
    --seed                   42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: training exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Learned Detail Memory Resolver V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/report.md"
echo "  ${OUTPUT_DIR}/ablation_results.csv"
echo "  ${OUTPUT_DIR}/eval_by_slice.csv"
echo "  ${OUTPUT_DIR}/best_metrics.json"
echo "  ${OUTPUT_DIR}/train_log.csv"
echo ""

if [ ! -f "${OUTPUT_DIR}/report.md" ]; then
    echo "ERROR: report.md not found — training may have failed"
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/best_metrics.json" ]; then
    echo "ERROR: best_metrics.json not found — training may have failed"
    exit 1
fi

# Surface the key result
python - <<'PYEOF'
import json, sys, os
d = "runs/entity_number_memory_v1/learned_detail_resolver_v1"
try:
    bm = json.load(open(os.path.join(d, "best_metrics.json")))
    if bm.get("no_improving_checkpoint_found"):
        print("[result] No checkpoint improved over base — model did not beat base accuracy.")
    else:
        print(f"[result] Best checkpoint: epoch={bm.get('epoch')}  "
              f"main_acc={bm.get('main_acc', float('nan')):.4f}  "
              f"base_acc={bm.get('base_acc', float('nan')):.4f}  "
              f"delta={bm.get('delta_acc', float('nan')):+.4f}")
    ar_path = os.path.join(d, "ablation_results.csv")
    if os.path.isfile(ar_path):
        import csv
        rows = list(csv.DictReader(open(ar_path)))
        print("[result] Ablation summary:")
        for r in rows:
            print(f"  {r['mode']:30s}  acc={r['acc_all']}")
except Exception as e:
    print(f"[result] Could not parse outputs: {e}", file=sys.stderr)
PYEOF

echo "[done] Report: ${OUTPUT_DIR}/report.md"
