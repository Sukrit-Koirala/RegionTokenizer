#!/bin/bash
#SBATCH --job-name=live_full_rebuild
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Live Full Pipeline Rebuild — LIMITED FIRST ARCHITECTURE TEST (logit-path fix)
#
# This is a limited, sampled first run to validate the architecture.
# Do NOT run the full 14.7M retrieval until this limited run shows useful gains.
#
# Logit-path fix (Stage01B):
#   Stage01 was saving h_ctx = get_hs_small()[:,-1,:] = h (raw ln_f output)
#   which gave base NLL ~7.95 instead of the correct ~3.75.
#   Stage01B patches all shards: h_ctx → h_prime (backbone.forward()[3][:,-1,:])
#   and recomputes base_topk_ids/lgt from correct lm_head(h_prime) logits.
#   Stages 02–07 transparently use patched shards via _get_s01_dir() fallback.
#
# Limits applied:
#   stride_train=64   max_train_rows=500000  → ~500k train rows
#   stride_val=8      max_val_rows=100000    → ~30k val rows (val corpus is short)
#   max_index_rows=500000                    → index ≤ 500k vectors
#   max_attach_train_rows=500000             → attach neighbors to ≤ 500k train rows
#   max_attach_val_rows=100000               → attach neighbors to ≤ 100k val rows
#
# Output root: runs/live_full_pipeline_rebuild_limited500k_logitfix/
#   (separate from the old broken run at runs/live_full_pipeline_rebuild_limited500k/)
#
# Stages (in order):
#   stage00  find_tokens        — locate/download WikiText-103 tokens
#   stage01  rebuild_dataset    — build input_ids + h_ctx shards (limited, h=ln_f output)
#   stage01b audit_logit_paths  — test 4 logit paths, patch shards with h_prime
#   stage02  verify_rebuild     — cos_mean > 0.999, NLL diff < 2e-3 (uses patched shards)
#   stage03  build_index        — FAISS/torch index over ≤500k patched train vectors
#   stage04  attach_neighbors   — kNN retrieval (limited, train self-excluded)
#   stage05  audit_retrieval    — coverage, recall@K, distributions
#   stage06  debug_identity     — step-0 delta=0 identity check
#   stage07  train_resolver     — training on limited rows
#
# Success thresholds (stage07):
#   Weak:        full_vocab_gain_all > +0.0033
#   Meaningful:  full_vocab_gain_all > +0.005
#   Strong:      full_vocab_gain_all > +0.010  inside_gate_gain > +0.05
#
# Optional env overrides (set before sbatch):
#   TRAIN_TOKENS_PATH   path to train token corpus  (default: auto-discovered)
#   VAL_TOKENS_PATH     path to val token corpus    (default: auto-discovered)
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
OUTPUT_ROOT="runs/live_full_pipeline_rebuild_limited500k_logitfix"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
REGION_MAP="runs/region_maps_128/token_to_region.json"

echo "========================================================"
echo " Live Full Pipeline Rebuild — LIMITED 500k LOGIT-FIX"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking required inputs..."

for F in "$SMALL_CKPT" "$SUPER_MAP" "$REGION_MAP"; do
    if [ ! -f "$F" ]; then echo "ERROR: required file not found: $F"; exit 1; fi
    echo "  [OK] $F"
done

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_ROOT"
mkdir -p logs

EXTRA_TOKEN_ARGS=""
if [ -n "${TRAIN_TOKENS_PATH:-}" ]; then
    echo "  [override] TRAIN_TOKENS_PATH=$TRAIN_TOKENS_PATH"
    EXTRA_TOKEN_ARGS="$EXTRA_TOKEN_ARGS --train_tokens_path $TRAIN_TOKENS_PATH"
fi
if [ -n "${VAL_TOKENS_PATH:-}" ]; then
    echo "  [override] VAL_TOKENS_PATH=$VAL_TOKENS_PATH"
    EXTRA_TOKEN_ARGS="$EXTRA_TOKEN_ARGS --val_tokens_path $VAL_TOKENS_PATH"
fi

echo ""
echo "[launch] Starting limited pipeline rebuild..."
echo ""

python scripts/run_live_full_pipeline_rebuild.py \
    --run_all                                     \
    --small_ckpt           "$SMALL_CKPT"          \
    --output_root          "$OUTPUT_ROOT"         \
    --super_map            "$SUPER_MAP"           \
    --region_map           "$REGION_MAP"          \
    \
    --ctx_len              128                    \
    --top_k                256                    \
    --num_neighbors        32                     \
    \
    --stride_train         64                     \
    --stride_val           8                      \
    --max_train_rows       500000                 \
    --max_val_rows         100000                 \
    \
    --max_index_rows       500000                 \
    --max_attach_train_rows 500000                \
    --max_attach_val_rows  100000                 \
    \
    --retrieval_backend    auto                   \
    --retrieval_chunk_size 131072                 \
    --query_batch_size     512                    \
    \
    --candidate_mode       base_topk_plus_neighbors \
    --resolver_dim         256                    \
    --resolver_layers      2                      \
    --resolver_heads       4                      \
    --delta_scale          0.25                   \
    --retrieval_tau        0.2                    \
    \
    --lr                   1e-5                   \
    --lambda_region        0.1                    \
    --lambda_rank          0.0                    \
    --lambda_kl            1.0                    \
    --lambda_delta         1e-3                   \
    --grad_clip            0.5                    \
    --batch_size           16                     \
    --grad_accum_steps     4                      \
    --steps                3000                   \
    --eval_every           500                    \
    --eval_batch_size      64                     \
    --shard_size           4096                   \
    --amp                                         \
    $EXTRA_TOKEN_ARGS

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: pipeline exited with code $EXIT"
    echo "Check: $OUTPUT_ROOT/pipeline_manifest.json"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Live Full Pipeline Rebuild (limited) complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  $OUTPUT_ROOT/pipeline_manifest.json"
echo "  $OUTPUT_ROOT/07_train_resolver/*/final_metrics.json"
echo "  $OUTPUT_ROOT/07_train_resolver/*/report.md"
echo "  $OUTPUT_ROOT/07_train_resolver/*/best_resolver.pt"
echo ""
echo "Success thresholds:"
echo "  Weak:       full_vocab_gain_all > +0.0033"
echo "  Meaningful: full_vocab_gain_all > +0.005"
echo "  Strong:     full_vocab_gain_all > +0.010"
echo ""

python -c "
import json, os, glob
manifest_path = '$OUTPUT_ROOT/pipeline_manifest.json'
if os.path.exists(manifest_path):
    m = json.load(open(manifest_path))
    stages = m.get('stages', {})
    print('Stage summary:')
    for s, info in stages.items():
        status = info.get('status', '?')
        mark = '[PASS]' if status == 'pass' else '[FAIL]' if status == 'fail' else '[    ]'
        print(f'  {mark} {s}')

metrics_paths = glob.glob('$OUTPUT_ROOT/07_train_resolver/*/final_metrics.json')
if metrics_paths:
    mp = sorted(metrics_paths)[-1]
    r = json.load(open(mp))
    print()
    print(f'Resolver metrics ({os.path.basename(os.path.dirname(mp))}):')
    print(f'  LIMITED_RUN       = {r.get(\"limited_run\")}')
    print(f'  train_rows_used   = {r.get(\"train_rows_used\",\"?\")} / {r.get(\"train_rows_total\",\"?\")}')
    print(f'  val_rows_used     = {r.get(\"val_rows_used\",\"?\")} / {r.get(\"val_rows_total\",\"?\")}')
    base = r.get('full_vocab_base_nll_all')
    nll  = r.get('full_vocab_refined_nll_all')
    gain = r.get('full_vocab_gain_all')
    if base  is not None: print(f'  base NLL          = {base:.6f}')
    if nll   is not None: print(f'  refined NLL       = {nll:.6f}')
    if gain  is not None: print(f'  gain              = {gain:+.6f}')
    print(f'  verdict           = {r.get(\"verdict\")}')
" 2>/dev/null || true
