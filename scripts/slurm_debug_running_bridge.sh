#!/bin/bash
#SBATCH --job-name=running_bridge_debug
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
# Running Bridge Adapter V0 — Identity Debug
#
# Architecture:
#   RunningBridgeUpdate  (per update layer, independent weights):
#     path_state + h_L_token + router/memory/fine/super evidence
#     → Pre-LN transformer → updated path_state (B, P, bridge_dim)
#     update_layers = [2, 4] → two sequential updates
#   RunningBridgeRefiner (final write):
#     [RESIDUAL_TOKEN + path_state] → transformer → out_proj(zero-init) → delta_final
#   Write (final_only):
#     h_refined = h_prime + alpha * delta_final
#   Write (multi_write, additionally):
#     write_projs[i](path_state[:,0,:]) → delta_L  (zero-init)
#     h_refined += sum_i(alpha_L_i * delta_L_i)
#     h_L_effective = h_L + prev_delta  (simulated causal chain, gradient flows)
#   Decode: logits = h_refined @ token_emb.T  (full-vocab, V=50257)
#
# Tests BOTH modes:
#   1. final_only  → debug_identity_finalonly/
#   2. multi_write → debug_identity_multiwrite/
#
# Verifies at step 0 (eval_before_train, steps=1):
#   1. delta_norm_max = 0.0          (Refiner out_proj is zero-initialized)
#   2. all_write_delta_norms = 0.0   (write_projs zero-initialized, multi_write only)
#   3. h_refined == h_prime          (within fp tolerance)
#   4. full_vocab_gated_nll_all == full_vocab_base_nll_all       (diff < 1e-3)
#   5. full_vocab_gated_nll_covered == full_vocab_base_nll_covered (diff < 1e-3)
#   6. masked_cand_gated_nll == masked_cand_base_nll              (diff < 1e-3)
#   7. masked_cand_base_nll == 3.378606                           (diff < 1e-3)
#   8. outside_gate_gated_nll_all == outside_gate_base_nll_all   (diff < 1e-5)
#   9. gold_force_included_rate = 0.0000
#  10. update_layers=[2,4] → update_layer_idxs resolved via layer_ids
#  11. debug_identity.json written with identity_pass=True
#
# Required output format:
#   variant = running_bridge_adapter
#   write_mode = final_only | multi_write
#   update_layers = [2, 4]
#   num_path_tokens = 4
#   delta_norm_max = 0.0
#   all_write_delta_norms = 0.0   (multi_write)
#   full_vocab_base_nll_all = 3.754938   (approx)
#   full_vocab_gated_nll_all = 3.754938
#   diff_all < 1e-3
#   IDENTITY PASS
#
# Run order:
#   slurm_debug_bridge_residual_adapter.sh        ← must have passed (V1)
#   slurm_debug_hard_sampler_bridge.sh            ← must have passed (sampler)
#   slurm_debug_midlayer_bridge_adapter.sh        ← should have passed (MidLayer)
#   slurm_debug_explicit_bridge_refiner.sh        ← should have passed (ExplicitBridge)
#   → THIS SCRIPT
#   → slurm_train_running_bridge.sh

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
OUTPUT_FINALONLY=runs/path_refiner_running_bridge/debug_identity_finalonly
OUTPUT_MULTIWRITE=runs/path_refiner_running_bridge/debug_identity_multiwrite

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Running Bridge Adapter V0 — Identity Debug  $(date) ==="
echo ""
echo "    Architecture:"
echo "      RunningBridgeUpdate  : path_state + h_L → updated path_state  (per layer)"
echo "      RunningBridgeRefiner : [RESIDUAL + path_state] → delta_final  (zero-init out_proj)"
echo "      final_only           : h_refined = h_prime + alpha * delta_final"
echo "      multi_write          : h_refined += sum_i(alpha_Li * delta_Li)  (zero-init write_projs)"
echo "      Decode               : logits = h_refined @ token_emb.T  (full-vocab)"
echo ""
echo "    SMALL_CKPT   : $SMALL_CKPT"
echo "    VAL_CAND_DIR : $VAL_CAND_DIR"
echo "    VAL_FEAT_DIR : $VAL_FEAT_DIR"
echo ""
echo "    Identity checks (both modes):"
echo "      1.  delta_norm_max = 0.0  (Refiner out_proj zero-init)"
echo "      2.  all_write_delta_norms = 0.0  (write_projs zero-init, multi_write)"
echo "      3.  full_vocab_gated_nll_all == full_vocab_base_nll_all  (diff < 1e-3)"
echo "      4.  full_vocab_gated_nll_covered == full_vocab_base_nll_covered  (diff < 1e-3)"
echo "      5.  masked_cand_gated_nll == masked_cand_base_nll  (diff < 1e-3)"
echo "      6.  masked_cand_base_nll == 3.378606  (canonical ref, diff < 1e-3)"
echo "      7.  outside_gate_gated_nll_all == outside_gate_base_nll_all  (diff < 1e-5)"
echo "      8.  gold_force_included_rate = 0.0000"
echo "      9.  update_layers=[2,4] resolved via layer_ids"
echo "     10.  debug_identity.json written with identity_pass=True"
echo ""
echo "    NOTE: full_vocab_base_nll_all (~3.75) != masked_cand_baseline (3.378606)."
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
echo "    update_layers=2,4 maps to:"
python -c "
import json
c = json.load(open('$VAL_FEAT_CFG'))
layer_ids = c['layer_ids']
for ul in [2, 4]:
    if ul in layer_ids:
        idx = layer_ids.index(ul)
        print(f'      layer_ids[{idx}] = {layer_ids[idx]}  (update_layer_idx={idx})')
    else:
        print(f'      WARNING: layer {ul} not in layer_ids={layer_ids}')
" 2>/dev/null || echo "    (could not resolve)"
echo ""

mkdir -p "$OUTPUT_FINALONLY" "$OUTPUT_MULTIWRITE"

# ── Common args ───────────────────────────────────────────────────────────────

COMMON_ARGS=(
    --small_ckpt        $SMALL_CKPT
    --train_cand_dir    $VAL_CAND_DIR
    --val_cand_dir      $VAL_CAND_DIR
    --train_feat_dir    $VAL_FEAT_DIR
    --val_feat_dir      $VAL_FEAT_DIR
    --baseline_json     $BASELINE_JSON
    --super_map         $SUPER_MAP
    --update_layers     2,4
    --num_path_tokens   4
    --bridge_dim        256
    --bridge_update_layers 1
    --bridge_heads      4
    --refiner_dim       256
    --refiner_layers    2
    --refiner_heads     4
    --ff_mult           4
    --top_fine_regions  24
    --top_superregions  8
    --train_filter      boundary
    --gate_filter       boundary
    --steps             1
    --eval_every        1
    --batch_size        32
    --eval_batch_size   64
    --lr                5e-5
    --lambda_kl         0.2
    --lambda_delta      3e-4
    --lambda_path       0.0
    --kl_topk           512
    --eval_before_train
    --fail_on_baseline_mismatch
    --device            cuda
)

# ── Run 1: final_only ─────────────────────────────────────────────────────────

echo "========================================================================"
echo "  RUN 1 / 2 : write_mode = final_only"
echo "  OUTPUT    : $OUTPUT_FINALONLY"
echo "========================================================================"
echo ""
echo "  Identity invariant: refiner.out_proj zero-init → delta_final=0 → h_refined=h_prime"
echo ""

python scripts/train_running_bridge.py \
    "${COMMON_ARGS[@]}" \
    --write_mode  final_only \
    --output_dir  $OUTPUT_FINALONLY

# Verify debug_identity.json
DEBUG_JSON_FO=$OUTPUT_FINALONLY/debug_identity.json
if [[ ! -f "$DEBUG_JSON_FO" ]]; then
    echo "" >&2
    echo "ERROR: $DEBUG_JSON_FO not found — final_only identity check may have crashed." >&2
    exit 1
fi

PASS_FO=$(python -c "import json; d=json.load(open('$DEBUG_JSON_FO')); print(d.get('identity_pass', False))" 2>/dev/null || echo "False")
if [[ "$PASS_FO" != "True" ]]; then
    echo "" >&2
    echo "ERROR: identity_pass=$PASS_FO (final_only) — check failed." >&2
    echo "       Inspect $DEBUG_JSON_FO for details." >&2
    exit 1
fi

echo ""
echo "  final_only identity PASSED."
echo ""

python -c "
import json
d = json.load(open('$DEBUG_JSON_FO'))
print('  final_only results:')
print(f\"    variant              = {d.get('variant', 'running_bridge_adapter')}\")
print(f\"    write_mode           = {d.get('write_mode', '?')}\")
print(f\"    update_layers        = {d.get('update_layers', '?')}\")
print(f\"    update_layer_idxs    = {d.get('update_layer_idxs', '?')}\")
print(f\"    num_path_tokens      = {d.get('num_path_tokens', '?')}\")
print(f\"    bridge_update_layers = {d.get('bridge_update_layers', '?')}\")
print(f\"    refiner_layers       = {d.get('refiner_layers', '?')}\")
print()
print(f\"    full_vocab_base_nll_all      = {d['full_vocab_base_nll_all']:.6f}  (PRIMARY)\")
print(f\"    full_vocab_gated_nll_all     = {d['full_vocab_gated_nll_all']:.6f}  (must == base)\")
print(f\"    full_vocab_base_nll_covered  = {d['full_vocab_base_nll_covered']:.6f}\")
print(f\"    masked_cand_base_nll         = {d['masked_cand_base_nll']:.6f}  (ref=3.378606)\")
print(f\"    nll_diff_all                 = {d['nll_diff_all']:.2e}  (< 1e-3)\")
print(f\"    nll_diff_covered             = {d['nll_diff_covered']:.2e}  (< 1e-3)\")
print(f\"    mc_diff                      = {d['mc_diff']:.2e}  (< 1e-3)\")
print(f\"    og_gate_diff                 = {d['og_gate_diff']:.2e}  (< 1e-5)\")
print(f\"    delta_norm_max               = {d['delta_norm_max']:.2e}  (must be ~0)\")
print(f\"    alpha                        = {d['alpha']:.4f}\")
print(f\"    identity_pass                = {d['identity_pass']}\")
" 2>/dev/null || echo "    (could not read debug_identity.json)"

echo ""

# ── Run 2: multi_write ────────────────────────────────────────────────────────

echo "========================================================================"
echo "  RUN 2 / 2 : write_mode = multi_write"
echo "  OUTPUT    : $OUTPUT_MULTIWRITE"
echo "========================================================================"
echo ""
echo "  Identity invariant:"
echo "    write_projs zero-init   → delta_L=0  → prev_delta=0 → h_L_effective=h_L"
echo "    refiner.out_proj zero-init → delta_final=0"
echo "    h_refined = h_prime + alpha*0 + sum_i(alpha_Li*0) = h_prime"
echo ""

python scripts/train_running_bridge.py \
    "${COMMON_ARGS[@]}" \
    --write_mode  multi_write \
    --output_dir  $OUTPUT_MULTIWRITE

# Verify debug_identity.json
DEBUG_JSON_MW=$OUTPUT_MULTIWRITE/debug_identity.json
if [[ ! -f "$DEBUG_JSON_MW" ]]; then
    echo "" >&2
    echo "ERROR: $DEBUG_JSON_MW not found — multi_write identity check may have crashed." >&2
    exit 1
fi

PASS_MW=$(python -c "import json; d=json.load(open('$DEBUG_JSON_MW')); print(d.get('identity_pass', False))" 2>/dev/null || echo "False")
if [[ "$PASS_MW" != "True" ]]; then
    echo "" >&2
    echo "ERROR: identity_pass=$PASS_MW (multi_write) — check failed." >&2
    echo "       Inspect $DEBUG_JSON_MW for details." >&2
    exit 1
fi

echo ""
echo "  multi_write identity PASSED."
echo ""

python -c "
import json
d = json.load(open('$DEBUG_JSON_MW'))
print('  multi_write results:')
print(f\"    variant              = {d.get('variant', 'running_bridge_adapter')}\")
print(f\"    write_mode           = {d.get('write_mode', '?')}\")
print(f\"    update_layers        = {d.get('update_layers', '?')}\")
print(f\"    update_layer_idxs    = {d.get('update_layer_idxs', '?')}\")
print(f\"    num_path_tokens      = {d.get('num_path_tokens', '?')}\")
print(f\"    bridge_update_layers = {d.get('bridge_update_layers', '?')}\")
print(f\"    refiner_layers       = {d.get('refiner_layers', '?')}\")
print()
print(f\"    full_vocab_base_nll_all      = {d['full_vocab_base_nll_all']:.6f}  (PRIMARY)\")
print(f\"    full_vocab_gated_nll_all     = {d['full_vocab_gated_nll_all']:.6f}  (must == base)\")
print(f\"    full_vocab_base_nll_covered  = {d['full_vocab_base_nll_covered']:.6f}\")
print(f\"    masked_cand_base_nll         = {d['masked_cand_base_nll']:.6f}  (ref=3.378606)\")
print(f\"    nll_diff_all                 = {d['nll_diff_all']:.2e}  (< 1e-3)\")
print(f\"    nll_diff_covered             = {d['nll_diff_covered']:.2e}  (< 1e-3)\")
print(f\"    mc_diff                      = {d['mc_diff']:.2e}  (< 1e-3)\")
print(f\"    og_gate_diff                 = {d['og_gate_diff']:.2e}  (< 1e-5)\")
print(f\"    delta_norm_max               = {d['delta_norm_max']:.2e}  (final write, ~0)\")
all_write_keys = [k for k in d if k.startswith('layer_delta_norm_max_')]
for k in sorted(all_write_keys):
    print(f\"    {k:35s} = {d[k]:.2e}  (~0, write_proj zero-init)\")
print(f\"    alpha                        = {d['alpha']:.4f}\")
print(f\"    identity_pass                = {d['identity_pass']}\")
" 2>/dev/null || echo "    (could not read debug_identity.json)"

echo ""

# ── Final summary ─────────────────────────────────────────────────────────────

echo "========================================================================"
echo "=== Running Bridge Adapter V0 — Identity Debug PASSED  $(date) ==="
echo "========================================================================"
echo ""
echo "Both modes passed:"
echo "  final_only  → $DEBUG_JSON_FO"
echo "  multi_write → $DEBUG_JSON_MW"
echo ""
echo "Confirmed (both modes):"
echo "  variant              = running_bridge_adapter"
echo "  update_layers        = [2, 4]"
echo "  num_path_tokens      = 4"
echo "  bridge_update_layers = 1  (transformer layers per RunningBridgeUpdate step)"
echo "  refiner_layers       = 2  (thinker, zero-init out_proj)"
echo "  delta_norm_max       = 0.0   (Refiner out_proj zero-init)"
echo "  all_write_delta_norms = 0.0  (write_projs zero-init, multi_write)"
echo "  full_vocab_gated_nll_all == full_vocab_base_nll_all   (diff < 1e-3)"
echo "  masked_cand_base_nll matches canonical 3.378606  (diff < 1e-3)"
echo "  gold_force_included_rate = 0.0000"
echo "  canonical fingerprint / coverage OK"
echo "  identity_pass = True in both debug_identity.json files"
echo ""
echo "Safe to proceed with slurm_train_running_bridge.sh"
echo "  → runs/path_refiner_running_bridge/boundary_running_finalonly_v0"
echo "  → runs/path_refiner_running_bridge/boundary_running_multiwrite_v0"
