#!/bin/bash
#SBATCH --job-name=midlayer_debug
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
# Mid-Layer Bridge Adapter — Identity + Safety Debug
#
# Verifies:
#   1. delta_h_max_abs = 0.0  (out_proj is zero-init)
#   2. h_refined == h_prime   (within fp tolerance)
#   3. logits_refined == logits_base  (within fp tolerance)
#   4. full_vocab_gated_nll_all == full_vocab_base_nll_all  (diff < 1e-3)
#   5. full_vocab_gated_nll_covered == full_vocab_base_nll_covered  (diff < 1e-3)
#   6. masked_cand_gated_nll == masked_cand_base_nll  (diff < 1e-3)
#   7. masked_cand_base_nll == 3.378606  (canonical ref, diff < 1e-3)
#   8. outside_gate_gated_nll_all == outside_gate_base_nll_all  (diff < 1e-5)
#   9. gold_force_included_rate = 0.0000
#  10. insert_after_block=4 → insert_layer_idx resolved correctly from layer_ids
#  11. Canonical fingerprint / coverage match
#
# Primary check: step-0 IDENTITY PASS message in output.
#
# Run order:
#   slurm_debug_bridge_residual_adapter.sh  ← must have passed (V1 identity check)
#   → THIS SCRIPT
#   → slurm_train_midlayer_bridge_adapter.sh

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
OUTPUT_DIR=runs/path_refiner_midlayer_bridge/debug_identity

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Mid-Layer Bridge Adapter — Identity Debug  $(date) ==="
echo "    SMALL_CKPT   : $SMALL_CKPT"
echo "    VAL_CAND_DIR : $VAL_CAND_DIR"
echo "    VAL_FEAT_DIR : $VAL_FEAT_DIR"
echo "    OUTPUT_DIR   : $OUTPUT_DIR"
echo ""
echo "    Checks:"
echo "      1.  delta_h_max_abs = 0.0  (zero-init out_proj)"
echo "      2.  full_vocab_gated_nll_all == full_vocab_base_nll_all  (diff < 1e-3)"
echo "      3.  full_vocab_gated_nll_covered == full_vocab_base_nll_covered  (diff < 1e-3)"
echo "      4.  masked_cand_gated_nll == masked_cand_base_nll  (diff < 1e-3)"
echo "      5.  masked_cand_base_nll == 3.378606  (canonical ref, diff < 1e-3)"
echo "      6.  outside_gate_gated_nll_all == outside_gate_base_nll_all  (diff < 1e-5)"
echo "      7.  gold_force_included_rate = 0.0000"
echo "      8.  insert_after_block=4 → insert_layer_idx resolved via layer_ids"
echo "      9.  Canonical fingerprint / coverage match"
echo "     10.  debug_identity.json written with all invariants"
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
N_LAYERS=$(python -c "import json; c=json.load(open('$VAL_FEAT_CFG')); print(c['n_layers_saved'])" 2>/dev/null || echo "?")
echo "    layer_ids            : $LAYER_IDS"
echo "    n_layers_saved       : $N_LAYERS"
echo ""
echo "    insert_after_block=4 maps to:"
python -c "
import json
c = json.load(open('$VAL_FEAT_CFG'))
layer_ids = c['layer_ids']
iab = 4
if iab in layer_ids:
    idx = layer_ids.index(iab)
    print(f'      layer_ids[{idx}] = {layer_ids[idx]}  (insert_layer_idx={idx})')
else:
    print(f'      WARNING: block 4 not in layer_ids={layer_ids}')
" 2>/dev/null || echo "    (could not resolve)"
echo ""

mkdir -p "$OUTPUT_DIR"

# ── Run identity check ────────────────────────────────────────────────────────

echo "Running mid-layer bridge identity check (steps=1, eval_before_train) ..."
echo "Script will:"
echo "  1. Resolve insert_after_block=4 → insert_layer_idx via layer_ids"
echo "  2. Build MidLayerBridgeAdapter with zero-init out_proj"
echo "  3. Assert INSERT_STATE token is prepended to bridge sequence"
echo "  4. Assert step-0 full_vocab_gated_nll == full_vocab_base_nll (all 3 families)"
echo "  5. Assert canonical fingerprint / coverage"
echo "  6. Write debug_identity.json"
echo ""

python scripts/train_midlayer_bridge_adapter.py   \
    --small_ckpt        $SMALL_CKPT               \
    --train_cand_dir    $VAL_CAND_DIR             \
    --val_cand_dir      $VAL_CAND_DIR             \
    --train_feat_dir    $VAL_FEAT_DIR             \
    --val_feat_dir      $VAL_FEAT_DIR             \
    --baseline_json     $BASELINE_JSON            \
    --super_map         $SUPER_MAP                \
    --output_dir        $OUTPUT_DIR               \
    --insert_after_block 4                        \
    --num_path_tokens   1                         \
    --train_filter      boundary                  \
    --gate_filter       boundary                  \
    --bridge_dim        256                       \
    --num_bridge_layers 2                         \
    --num_heads         4                         \
    --ff_mult           4                         \
    --top_fine_regions  24                        \
    --top_superregions  8                         \
    --steps             1                         \
    --eval_every        1                         \
    --batch_size        32                        \
    --eval_batch_size   64                        \
    --lr                5e-5                      \
    --lambda_kl         0.2                       \
    --lambda_delta      3e-4                      \
    --kl_topk           512                       \
    --eval_before_train                           \
    --fail_on_baseline_mismatch                   \
    --device            cuda

echo ""
echo "=== Mid-Layer Bridge Identity Debug PASSED  $(date) ==="
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
echo "  insert_after_block=4 resolved via layer_ids"
echo ""

# Print identity results for reference
DEBUG_JSON=$OUTPUT_DIR/debug_identity.json
if [[ -f "$DEBUG_JSON" ]]; then
    echo "Identity check results:"
    python -c "
import json
d = json.load(open('$DEBUG_JSON'))
print(f\"  insert_after_block  = {d['insert_after_block']}\")
print(f\"  insert_layer_idx    = {d['insert_layer_idx']}  (h_layers index)\")
print(f\"  layer_ids           = {d['layer_ids']}\")
print()
print(f\"  full_vocab_base_nll_all      = {d['full_vocab_base_nll_all']:.6f}  (PRIMARY threshold)\")
print(f\"  full_vocab_gated_nll_all     = {d['full_vocab_gated_nll_all']:.6f}  (must == base at step 0)\")
print(f\"  full_vocab_base_nll_covered  = {d['full_vocab_base_nll_covered']:.6f}  (secondary)\")
print(f\"  masked_cand_base_nll         = {d['masked_cand_base_nll']:.6f}  (canonical ref = 3.378606)\")
print(f\"  nll_diff_all                 = {d['nll_diff_all']:.2e}  (< 1e-3)\")
print(f\"  nll_diff_covered             = {d['nll_diff_covered']:.2e}  (< 1e-3)\")
print(f\"  mc_diff                      = {d['mc_diff']:.2e}  (< 1e-3)\")
print(f\"  og_gate_diff                 = {d['og_gate_diff']:.2e}  (< 1e-5)\")
print(f\"  delta_norm_max               = {d['delta_norm_max']:.2e}  (must be ~0)\")
print(f\"  alpha                        = {d['alpha']:.4f}\")
print()
print('NOTE: full_vocab_base_nll_all is PRIMARY. masked_cand_base_nll is a different metric.')
print('      Do NOT compare full-vocab NLL to masked-candidate NLL.')
" 2>/dev/null || echo "    (could not read debug_identity.json)"
fi

echo ""
echo "Safe to proceed with slurm_train_midlayer_bridge_adapter.sh"
echo "  → runs/path_refiner_midlayer_bridge/boundary_insert4_v1"
