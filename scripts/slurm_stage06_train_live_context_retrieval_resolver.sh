#!/bin/bash
#SBATCH --job-name=stage06_train_lctx
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Stage 06 — Train Live Context + Retrieval Resolver
#
# Requires all prior stages to have passed.
#
# Model: LiveContextRetrievalTokenResolver
#   - Runs frozen backbone live on input_ids → h_ctx
#   - Soft region planner: p_fine, p_super from h_ctx
#   - Retrieval support: neighbor_gold_tokens + scores
#   - Bounded delta: delta_scale * tanh(raw_delta), delta_head zero-init
#   - candidate_mode: base_topk_plus_neighbors
#
# Output: runs/live_context_pipeline/stage06_train_ctxret_resolver/top256_ctx256_ret32_boundary_v1/
#
# Success thresholds:
#   Weak:        full_vocab_gain_all > +0.0033
#   Meaningful:  full_vocab_gain_all > +0.005
#   Strong:      full_vocab_gain_all > +0.010  inside_gate > +0.05
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
TRAIN_CAND_DIR="runs/path_refiner_clean/data/train_hgrid_K24"
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
TRAIN_CTX_DIR="runs/live_context_pipeline/stage01_context_data/train_ctx"
VAL_CTX_DIR="runs/live_context_pipeline/stage01_context_data/val_ctx"
TRAIN_RET_DIR="runs/live_context_pipeline/stage03_retrieval_neighbors/train_retrieval"
VAL_RET_DIR="runs/live_context_pipeline/stage03_retrieval_neighbors/val_retrieval"
BASELINE_JSON="runs/path_refiner_clean/baselines/saved_candidate_baseline.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
REGION_MAP="runs/region_maps_128/token_to_region.json"
OUTPUT_DIR="runs/live_context_pipeline/stage06_train_ctxret_resolver/top256_ctx256_ret32_boundary_v1"
STAGE05_REPORT="runs/live_context_pipeline/stage05_debug_identity/debug_identity.json"
STAGE01B_AUDIT="runs/live_context_pipeline/stage01b_live_baseline_audit/live_baseline_audit.json"

echo "========================================================"
echo " Stage 06 — Train Live Context + Retrieval Resolver"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking all prior stage outputs..."

for F in "$SMALL_CKPT" "$BASELINE_JSON" "$SUPER_MAP" "$REGION_MAP" \
         "$STAGE05_REPORT" "$STAGE01B_AUDIT"; do
    if [ ! -f "$F" ]; then echo "ERROR: required file not found: $F"; exit 1; fi
    echo "  [OK] $F"
done

python -c "
import json, sys
r = json.load(open('$STAGE05_REPORT'))
if not r.get('identity_pass', False):
    print('ERROR: Stage 05 identity_pass is False. Fix Stage 05 first. Do NOT train.')
    sys.exit(1)
print('  [OK] Stage 05 identity_pass = True')
"

python -c "
import json, sys
r = json.load(open('$STAGE01B_AUDIT'))
if not r.get('live_baseline_compatible', False):
    print('ERROR: Stage 01B live_baseline_compatible is False.')
    print('The live backbone NLL does not match the cached baseline.')
    print('Fix the ctx_len convention in Stage 01 and re-run stages 01-05 first.')
    bc = r.get('best_convention')
    if bc:
        print(f'  Best ctx_len tried: {bc.get(\"ctx_len\")}  NLL: {bc.get(\"live_base_nll_all\")}')
    diag = r.get('diagnosis', {})
    if diag:
        print(f'  Diagnosis: {diag.get(\"likely_cause\", \"?\")}')
    sys.exit(1)
bc = r.get('best_convention', {})
print(f'  [OK] Stage 01B live_baseline_compatible=True  best_ctx_len={bc.get(\"ctx_len\")}')
"

for D in "$TRAIN_CAND_DIR" "$VAL_CAND_DIR" "$TRAIN_CTX_DIR" "$VAL_CTX_DIR" \
         "$TRAIN_RET_DIR" "$VAL_RET_DIR"; do
    if [ ! -d "$D" ]; then echo "ERROR: not found: $D"; exit 1; fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then echo "ERROR: no shards in $D"; exit 1; fi
    echo "  [OK] $D  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present. Stage 05 identity confirmed. Stage 01B compatible."
echo ""
mkdir -p "$OUTPUT_DIR"

python scripts/stage06_train_live_context_retrieval_resolver.py \
    --small_ckpt           "$SMALL_CKPT"       \
    --train_cand_dir       "$TRAIN_CAND_DIR"   \
    --val_cand_dir         "$VAL_CAND_DIR"     \
    --train_ctx_dir        "$TRAIN_CTX_DIR"    \
    --val_ctx_dir          "$VAL_CTX_DIR"      \
    --train_retrieval_dir  "$TRAIN_RET_DIR"    \
    --val_retrieval_dir    "$VAL_RET_DIR"      \
    --baseline_json        "$BASELINE_JSON"    \
    --super_map            "$SUPER_MAP"        \
    --region_map           "$REGION_MAP"       \
    --output_dir           "$OUTPUT_DIR"       \
    --stage01b_audit       "$STAGE01B_AUDIT"   \
    --ctx_len_from_audit                       \
    --ctx_len              256                 \
    --candidate_mode       base_topk_plus_neighbors \
    --confuser_source      base_topk           \
    --top_k                256                 \
    --num_neighbors        32                  \
    --train_filter         boundary            \
    --gate_filter          boundary            \
    --use_filtered_train_loader                \
    --resolver_dim         256                 \
    --resolver_layers      2                   \
    --resolver_heads       4                   \
    --delta_scale          0.25                \
    --retrieval_tau        0.2                 \
    --batch_size           16                  \
    --grad_accum_steps     4                   \
    --steps                3000                \
    --eval_every           500                 \
    --eval_batch_size      64                  \
    --lr                   1e-5                \
    --lambda_region        0.1                 \
    --lambda_rank          0.0                 \
    --lambda_kl            1.0                 \
    --lambda_delta         1e-3                \
    --rank_margin          0.05                \
    --grad_clip            0.5                 \
    --amp                                      \
    --eval_before_train                        \
    --fail_on_baseline_mismatch

EXIT=$?
if [ $EXIT -ne 0 ]; then echo "ERROR: stage06 exited with code $EXIT"; exit $EXIT; fi

echo ""
echo "========================================================"
echo " Stage 06 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  $OUTPUT_DIR/final_metrics.json"
echo "  $OUTPUT_DIR/report.md"
echo "  $OUTPUT_DIR/best_resolver.pt  (only if beats baseline)"
echo ""
echo "Success thresholds:"
echo "  Weak:       full_vocab_gain_all > +0.0033"
echo "  Meaningful: full_vocab_gain_all > +0.005"
echo "  Strong:     full_vocab_gain_all > +0.010"
echo ""
