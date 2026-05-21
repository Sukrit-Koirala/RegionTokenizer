#!/bin/bash
#SBATCH --job-name=stage05_debug_lctx
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
# Stage 05 — Debug Model Identity
#
# Instantiates LiveContextRetrievalTokenResolver and verifies step-0 identity:
#   delta_head zero-init → delta=0 → refined == base logits
#
# Expected output:
#   IDENTITY PASS
#   gold_force_included_rate = 0.0000
#   diff_all < 1e-3
#   outside_gate_diff ≈ 0
#
# Prerequisite: Stages 01, 03, 04 must have passed
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
VAL_CTX_DIR="runs/live_context_pipeline/stage01_context_data/val_ctx"
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
VAL_RET_DIR="runs/live_context_pipeline/stage03_retrieval_neighbors/val_retrieval"
BASELINE_JSON="runs/path_refiner_clean/baselines/saved_candidate_baseline.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
REGION_MAP="runs/region_maps_128/token_to_region.json"
OUTPUT_DIR="runs/live_context_pipeline/stage05_debug_identity"
STAGE04_REPORT="runs/live_context_pipeline/stage04_data_audit/retrieval_audit.json"

echo "========================================================"
echo " Stage 05 — Debug Model Identity"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking Stage 04 outputs..."

for F in "$SMALL_CKPT" "$BASELINE_JSON" "$SUPER_MAP" "$REGION_MAP" "$STAGE04_REPORT"; do
    if [ ! -f "$F" ]; then echo "ERROR: required file not found: $F"; exit 1; fi
    echo "  [OK] $F"
done

python -c "
import json, sys
r = json.load(open('$STAGE04_REPORT'))
if not r.get('audit_pass', False):
    print('ERROR: Stage 04 audit_pass is False. Fix Stage 04 first.')
    sys.exit(1)
print('  [OK] Stage 04 audit_pass = True')
"

for D in "$VAL_CTX_DIR" "$VAL_CAND_DIR" "$VAL_RET_DIR"; do
    if [ ! -d "$D" ]; then echo "ERROR: not found: $D"; exit 1; fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then echo "ERROR: no shards in $D"; exit 1; fi
    echo "  [OK] $D  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_DIR"

python scripts/stage05_debug_live_context_retrieval_resolver.py \
    --small_ckpt        "$SMALL_CKPT"      \
    --val_ctx_dir       "$VAL_CTX_DIR"     \
    --val_cand_dir      "$VAL_CAND_DIR"    \
    --val_retrieval_dir "$VAL_RET_DIR"     \
    --baseline_json     "$BASELINE_JSON"   \
    --super_map         "$SUPER_MAP"       \
    --region_map        "$REGION_MAP"      \
    --output_dir        "$OUTPUT_DIR"      \
    --ctx_len           256                \
    --top_k             256                \
    --num_neighbors     32                 \
    --resolver_dim      256                \
    --resolver_layers   2                  \
    --resolver_heads    4                  \
    --delta_scale       0.25               \
    --gate_filter       boundary           \
    --eval_batch_size   64

EXIT=$?
if [ $EXIT -ne 0 ]; then echo "ERROR: stage05 exited with code $EXIT"; exit $EXIT; fi

echo ""
echo "========================================================"
echo " Stage 05 complete. $(date)"
echo "========================================================"
echo ""
echo "Required before Stage 06:"
echo "  $OUTPUT_DIR/debug_identity.json  (identity_pass == true)"
echo ""
