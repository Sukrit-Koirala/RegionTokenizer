#!/bin/bash
#SBATCH --job-name=bridge_debug
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Bridge Residual Adapter — Identity + Safety Debug
#
# Verifies:
#   1. delta_h_max_abs = 0.0  (out_proj is zero-init)
#   2. h_refined == h_prime   (within fp tolerance)
#   3. logits_refined == logits_base  (within fp tolerance)
#   4. full_vocab_gated_nll == full_vocab_base_nll  (diff < 1e-4)
#   5. gold_force_included_rate = 0.0000  (always — no selection step)
#   6. Canonical fingerprint / coverage / num_examples / num_covered match
#
# All checks are asserted inside train_bridge_residual_adapter.py (--eval_before_train).
# Any failure raises RuntimeError and exits non-zero.
#
# Do NOT launch slurm_train_bridge_residual_adapter.sh until this PASSES.
#
# Run order:
#   slurm_build_multilayer_residual_features.sh  ← must have completed
#   → THIS SCRIPT
#   → slurm_train_bridge_residual_adapter.sh

source ~/miniconda3/bin/activate
conda activate learned_regions

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

# ── Paths ─────────────────────────────────────────────────────────────────────

SMALL_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
VAL_CAND_DIR=runs/path_refiner_clean/data/val_hgrid_K24
VAL_FEAT_DIR=runs/path_refiner_residual_interface/features/val_multilayer
BASELINE_JSON=runs/path_refiner_clean/baselines/saved_candidate_baseline.json
OUTPUT_DIR=runs/path_refiner_bridge_adapter/debug_identity

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Bridge Residual Adapter — Identity Debug  $(date) ==="
echo "    SMALL_CKPT  : $SMALL_CKPT"
echo "    VAL_CAND_DIR: $VAL_CAND_DIR"
echo "    VAL_FEAT_DIR: $VAL_FEAT_DIR"
echo "    BASELINE_JSON: $BASELINE_JSON"
echo "    OUTPUT_DIR  : $OUTPUT_DIR"
echo ""
echo "    Checks:"
echo "      1. delta_h_max_abs = 0.0  (zero-init out_proj)"
echo "      2. full_vocab_gated_nll_all == full_vocab_base_nll_all  (diff < 1e-3)"
echo "      3. full_vocab_gated_nll_covered == full_vocab_base_nll_covered  (diff < 1e-3)"
echo "      4. masked_cand_gated_nll == masked_cand_base_nll  (diff < 1e-3)"
echo "      5. masked_cand_base_nll == 3.378606  (canonical ref, diff < 1e-3)"
echo "      6. outside_gate_gated_nll_all == outside_gate_base_nll_all  (diff < 1e-5)"
echo "      7. gold_force_included_rate = 0.0000  (no selection step)"
echo "      8. Canonical fingerprint / coverage match"
echo ""

for f in "$SMALL_CKPT" "$SUPER_MAP" "$BASELINE_JSON"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

VAL_FEAT_CFG=$VAL_FEAT_DIR/config.json
if [[ ! -f "$VAL_FEAT_CFG" ]]; then
    echo "ERROR: $VAL_FEAT_CFG not found." >&2
    echo "       Run slurm_build_multilayer_residual_features.sh first." >&2
    exit 1
fi

N_VAL=$(find "$VAL_CAND_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_FEAT=$(find "$VAL_FEAT_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    val candidate shards : $N_VAL"
echo "    val feature shards   : $N_FEAT"
if [[ "$N_VAL" -eq 0 || "$N_FEAT" -eq 0 ]]; then
    echo "ERROR: missing shards." >&2; exit 1
fi

LAYER_IDS=$(python -c "import json; c=json.load(open('$VAL_FEAT_CFG')); print(c['layer_ids'])" 2>/dev/null || echo "?")
echo "    layer_ids (features) : $LAYER_IDS"
echo ""

mkdir -p "$OUTPUT_DIR"

# ── Run identity check (--eval_before_train, --steps 0 not supported — use 1 step) ──

echo "Running bridge identity check (steps=1, eval_before_train) ..."
echo "The script will:"
echo "  1. Build BridgeResidualAdapter with zero-init out_proj"
echo "  2. Run step-0 full val eval"
echo "  3. Assert delta_norm_max == 0"
echo "  4. Assert full_vocab_gated_nll == full_vocab_base_nll  (diff < 1e-4)"
echo "  5. Assert gold_force_included_rate == 0.0000"
echo "  6. Assert canonical fingerprint / coverage match"
echo ""

python scripts/train_bridge_residual_adapter.py       \
    --small_ckpt        $SMALL_CKPT                   \
    --train_cand_dir    $VAL_CAND_DIR                 \
    --val_cand_dir      $VAL_CAND_DIR                 \
    --train_feat_dir    $VAL_FEAT_DIR                 \
    --val_feat_dir      $VAL_FEAT_DIR                 \
    --baseline_json     $BASELINE_JSON                \
    --super_map         $SUPER_MAP                    \
    --output_dir        $OUTPUT_DIR                   \
    --train_filter      boundary                      \
    --gate_filter       boundary                      \
    --bridge_dim        256                           \
    --num_bridge_layers 2                             \
    --num_heads         4                             \
    --ff_mult           4                             \
    --top_fine_regions  24                            \
    --top_superregions  8                             \
    --steps             1                             \
    --eval_every        1                             \
    --batch_size        32                            \
    --eval_batch_size   64                            \
    --lr                1e-4                          \
    --lambda_kl         0.1                           \
    --lambda_delta      1e-4                          \
    --kl_topk           512                           \
    --eval_before_train                               \
    --fail_on_baseline_mismatch                       \
    --device            cuda

echo ""
echo "=== Bridge Identity Debug PASSED  $(date) ==="
echo ""
echo "Confirmed:"
echo "  delta_h_max_abs = 0.0  (zero-init out_proj)"
echo "  full_vocab_gated_nll_all == full_vocab_base_nll_all  (diff < 1e-3)"
echo "  full_vocab_gated_nll_covered == full_vocab_base_nll_covered  (diff < 1e-3)"
echo "  masked_cand_gated_nll == masked_cand_base_nll  (diff < 1e-3)"
echo "  masked_cand_base_nll matches canonical 3.378606  (diff < 1e-3)"
echo "  outside_gate_gated_nll_all == outside_gate_base_nll_all  (diff < 1e-5)"
echo "  gold_force_included_rate = 0.0000"
echo "  canonical fingerprint / coverage OK"
echo ""

# ── Print baseline values for reference ──────────────────────────────────────

BL_JSON=$OUTPUT_DIR/full_vocab_baseline.json
if [[ -f "$BL_JSON" ]]; then
    echo "Full-vocab baseline (different from masked-candidate baseline):"
    python -c "
import json
d = json.load(open('$BL_JSON'))
print(f\"  full_vocab_base_nll_all    = {d['full_vocab_base_nll_all']:.6f}  (PRIMARY threshold)\")
print(f\"  full_vocab_base_nll_covered= {d['full_vocab_base_nll_covered']:.6f}  (secondary)\")
print(f\"  masked_cand_base_nll       = {d['masked_cand_base_nll']:.6f}\")
print(f\"  masked_cand_ref            = {d['masked_cand_baseline_ref']:.6f}  (prior refiners used this)\")
print(f\"  coverage                   = {d['coverage']:.6f}\")
print(f\"  fingerprint                = {d['dataset_fingerprint']}\")
print()
print('NOTE: full_vocab_base_nll_all is PRIMARY. masked_cand_base_nll is a different metric.')
print('      Do NOT compare full-vocab NLL to masked-candidate NLL.')
print('      Best checkpoint saved only if full_vocab_gated_nll_all < full_vocab_base_nll_all.')
"
fi

echo ""
echo "Safe to proceed with slurm_train_bridge_residual_adapter.sh"
