#!/bin/bash
#SBATCH --job-name=explicit_bridge_debug
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
# Explicit Bridge + Refiner — Identity Debug
#
# Architecture:
#   ExplicitBridge   (translator):
#     h_insert + h_layers + region/router/memory evidence → path_tokens (B, P, bridge_dim)
#     No out_proj. No delta_h production.
#   ExplicitRefiner  (thinker):
#     [RESIDUAL_TOKEN + path_tokens] → transformer → out_proj(zero-init) → delta_h
#   Write: h_prime_refined = h_prime + alpha * delta_h
#   Decode: logits = h_prime_refined @ token_emb.T   (full-vocab, V=50257)
#
# Verifies at step 0 (eval_before_train, steps=1):
#   1. delta_norm_max = 0.0          (Refiner out_proj is zero-initialized)
#   2. h_prime_refined == h_prime    (within fp tolerance)
#   3. full_vocab_gated_nll_all == full_vocab_base_nll_all        (diff < 1e-3)
#   4. full_vocab_gated_nll_covered == full_vocab_base_nll_covered (diff < 1e-3)
#   5. masked_cand_gated_nll == masked_cand_base_nll               (diff < 1e-3)
#   6. masked_cand_base_nll == 3.378606                            (diff < 1e-3)
#   7. outside_gate_gated_nll_all == outside_gate_base_nll_all    (diff < 1e-5)
#   8. gold_force_included_rate = 0.0000
#   9. Canonical fingerprint / coverage match
#  10. insert_after_block=4 → insert_layer_idx resolved via layer_ids
#  11. debug_identity.json written with identity_pass=True
#
# Required output format (from spec):
#   variant = explicit_bridge_refiner
#   insert_after_block = 4
#   num_path_tokens = 4
#   bridge_layers = 1
#   refiner_layers = 2
#   delta_norm_max = 0.0
#   full_vocab_base_nll_all = 3.754938   (value will differ from 3.378606 — different metric)
#   full_vocab_gated_nll_all = 3.754938
#   diff_all < 1e-3
#   IDENTITY PASS
#
# Primary check: debug_identity.json exists with identity_pass=True
#
# Run order:
#   slurm_debug_bridge_residual_adapter.sh       ← must have passed (V1 identity check)
#   slurm_debug_hard_sampler_bridge.sh           ← must have passed (sampler check)
#   slurm_debug_midlayer_bridge_adapter.sh       ← should have passed (MidLayer check)
#   → THIS SCRIPT
#   → slurm_train_explicit_bridge_refiner.sh

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
OUTPUT_DIR=runs/path_refiner_explicit_bridge_refiner/debug_identity

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Explicit Bridge + Refiner — Identity Debug  $(date) ==="
echo ""
echo "    Architecture:"
echo "      ExplicitBridge  : evidence → path_tokens (B, P=4, D=256)  [NO out_proj]"
echo "      ExplicitRefiner : [RESIDUAL + path_tokens] → delta_h       [zero-init out_proj]"
echo "      Write           : h_prime_refined = h_prime + alpha * delta_h"
echo "      Decode          : logits = h_prime_refined @ token_emb.T   (full-vocab)"
echo ""
echo "    SMALL_CKPT   : $SMALL_CKPT"
echo "    VAL_CAND_DIR : $VAL_CAND_DIR"
echo "    VAL_FEAT_DIR : $VAL_FEAT_DIR"
echo "    OUTPUT_DIR   : $OUTPUT_DIR"
echo ""
echo "    Identity checks:"
echo "      1.  delta_norm_max = 0.0  (Refiner out_proj zero-init)"
echo "      2.  full_vocab_gated_nll_all == full_vocab_base_nll_all  (diff < 1e-3)"
echo "      3.  full_vocab_gated_nll_covered == full_vocab_base_nll_covered  (diff < 1e-3)"
echo "      4.  masked_cand_gated_nll == masked_cand_base_nll  (diff < 1e-3)"
echo "      5.  masked_cand_base_nll == 3.378606  (canonical ref, diff < 1e-3)"
echo "      6.  outside_gate_gated_nll_all == outside_gate_base_nll_all  (diff < 1e-5)"
echo "      7.  gold_force_included_rate = 0.0000"
echo "      8.  insert_after_block=4 resolved via layer_ids"
echo "      9.  Canonical fingerprint / coverage match"
echo "     10.  debug_identity.json written with identity_pass=True"
echo ""
echo "    NOTE: full_vocab_base_nll_all (~3.75) is NOT the same as masked_cand_baseline (3.378606)."
echo "          They measure different things. Do NOT compare them."
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

echo "Running explicit bridge + refiner identity check (steps=1, eval_before_train) ..."
echo "Script will:"
echo "  1. Build ExplicitBridge (no out_proj) and ExplicitRefiner (zero-init out_proj)"
echo "  2. Resolve insert_after_block=4 → insert_layer_idx via layer_ids"
echo "  3. Assert step-0 delta_norm_max == 0.0"
echo "  4. Assert step-0 full_vocab_gated_nll == full_vocab_base_nll (all 3 families)"
echo "  5. Assert canonical fingerprint / coverage"
echo "  6. Write debug_identity.json with identity_pass=True"
echo ""

python scripts/train_explicit_bridge_refiner.py \
    --small_ckpt        $SMALL_CKPT             \
    --train_cand_dir    $VAL_CAND_DIR            \
    --val_cand_dir      $VAL_CAND_DIR            \
    --train_feat_dir    $VAL_FEAT_DIR            \
    --val_feat_dir      $VAL_FEAT_DIR            \
    --baseline_json     $BASELINE_JSON           \
    --super_map         $SUPER_MAP               \
    --output_dir        $OUTPUT_DIR              \
    --insert_after_block 4                       \
    --num_path_tokens   4                        \
    --bridge_dim        256                      \
    --bridge_layers     1                        \
    --bridge_heads      4                        \
    --refiner_dim       256                      \
    --refiner_layers    2                        \
    --refiner_heads     4                        \
    --ff_mult           4                        \
    --top_fine_regions  24                       \
    --top_superregions  8                        \
    --train_filter      boundary                 \
    --gate_filter       boundary                 \
    --steps             1                        \
    --eval_every        1                        \
    --batch_size        32                       \
    --eval_batch_size   64                       \
    --lr                5e-5                     \
    --lambda_kl         0.2                      \
    --lambda_delta      3e-4                     \
    --lambda_path       0.0                      \
    --kl_topk           512                      \
    --eval_before_train                          \
    --fail_on_baseline_mismatch                  \
    --device            cuda

# ── Verify debug_identity.json was written ────────────────────────────────────

DEBUG_JSON=$OUTPUT_DIR/debug_identity.json
if [[ ! -f "$DEBUG_JSON" ]]; then
    echo "" >&2
    echo "ERROR: $DEBUG_JSON not found — identity check may have crashed or skipped." >&2
    exit 1
fi

# Verify identity_pass=True
PASS=$(python -c "import json; d=json.load(open('$DEBUG_JSON')); print(d.get('identity_pass', False))" 2>/dev/null || echo "False")
if [[ "$PASS" != "True" ]]; then
    echo "" >&2
    echo "ERROR: identity_pass=$PASS in debug_identity.json — check failed." >&2
    echo "       Run slurm_debug_bridge_residual_adapter.sh to diagnose invariant failures." >&2
    exit 1
fi

echo ""
echo "=== Explicit Bridge + Refiner Identity Debug PASSED  $(date) ==="
echo ""
echo "Confirmed:"
echo "  variant              = explicit_bridge_refiner"
echo "  insert_after_block   = 4"
echo "  num_path_tokens      = 4"
echo "  bridge_layers        = 1   (translator, NO out_proj)"
echo "  refiner_layers       = 2   (thinker, zero-init out_proj)"
echo "  delta_norm_max       = 0.0  (Refiner out_proj zero-init)"
echo "  full_vocab_gated_nll_all == full_vocab_base_nll_all   (diff < 1e-3)"
echo "  full_vocab_gated_nll_covered == full_vocab_base_nll_covered  (diff < 1e-3)"
echo "  masked_cand_gated_nll == masked_cand_base_nll  (diff < 1e-3)"
echo "  masked_cand_base_nll matches canonical 3.378606  (diff < 1e-3)"
echo "  outside_gate_gated_nll_all == outside_gate_base_nll_all  (diff < 1e-5)"
echo "  gold_force_included_rate = 0.0000"
echo "  canonical fingerprint / coverage OK"
echo "  identity_pass = True in debug_identity.json"
echo ""

# Print identity results
python -c "
import json
d = json.load(open('$DEBUG_JSON'))
print('Identity check results:')
print(f\"  variant              = {d.get('variant', 'explicit_bridge_refiner')}\")
print(f\"  insert_after_block   = {d.get('insert_after_block', '?')}\")
print(f\"  insert_layer_idx     = {d.get('insert_layer_idx', '?')}  (h_layers index)\")
print(f\"  layer_ids            = {d.get('layer_ids', '?')}\")
print(f\"  num_path_tokens      = {d.get('num_path_tokens', '?')}\")
print(f\"  bridge_layers        = {d.get('bridge_layers', '?')}\")
print(f\"  refiner_layers       = {d.get('refiner_layers', '?')}\")
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
print(f\"  identity_pass                = {d['identity_pass']}\")
print()
print('NOTE: full_vocab_base_nll_all (~3.75) != masked_cand_base_nll (3.378606).')
print('      These are different metrics. Do NOT compare them.')
" 2>/dev/null || echo "    (could not read debug_identity.json)"

echo ""
echo "Safe to proceed with slurm_train_explicit_bridge_refiner.sh"
echo "  → runs/path_refiner_explicit_bridge_refiner/boundary_insert4_refiner2_path4_v1"
