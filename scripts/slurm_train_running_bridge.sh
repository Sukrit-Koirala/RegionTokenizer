#!/bin/bash
#SBATCH --job-name=running_bridge_train
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
# Running Bridge Adapter V0 — Training
#
# Architecture:
#   RunningBridgeUpdate  : path_state + h_L evidence → updated path_state (per update layer)
#   RunningBridgeRefiner : [RESIDUAL_TOKEN + path_state] → out_proj(zero-init) → delta_final
#   final_only: h_refined = h_prime + alpha * delta_final
#   final_patch_sim: diagnostic — adds all delta_Li to h_prime (NOT causal)
#                    h_L_effective = h_L + prev_delta for gradient flow only
#   Decode: logits = h_refined @ token_emb.T  (full-vocab, V=50257)
#
# Hypothesis: Maintaining persistent path-state across multiple backbone layers (tracking
# the trajectory of how path decisions form) should outperform one-shot adapters.
#
# Runs:
#   1. final_only      → boundary_running_finalonly_v0/
#   2. final_patch_sim → boundary_running_finalpatchsim_v0/
#      (diagnostic: adds multiple deltas to h_prime, NOT causal)
#
# NOTE: causal_multi_write is UNAVAILABLE.
# Dataset stores per-position hidden states (N, n_layers, d_model) only.
# Remaining frozen backbone blocks cannot be re-run from injected residuals
# without input_ids. final_patch_sim is the diagnostic substitute.
# The real causal experiment requires dataset rebuild with input_ids saved.
#
# Loss: CE + lambda_kl * KL_topk(512) + lambda_delta * sum_delta + lambda_path * path_norm
# Eff batch: 32 * 2 = 64 boundary examples per step
# PRIMARY eval metric: full_vocab_gated_nll_all  (vs full_vocab_base_nll_all)
#
# Architecture comparison targets (prior results):
#   V1 BridgeResidualAdapter          : +0.003037 NLL improvement
#   MidLayerBridgeAdapter             : +0.003347 NLL improvement
#   ExplicitBridgeRefiner             : +0.003305 NLL improvement
#   RunningBridge final_only      V0  : target > +0.003347 (beat MidLayer)
#   RunningBridge final_patch_sim V0  : diagnostic — not directly comparable to causal
#
# Prerequisite: slurm_debug_running_bridge.sh must have passed BOTH modes.
#
# Run order:
#   slurm_debug_bridge_residual_adapter.sh        ← must have passed
#   slurm_debug_hard_sampler_bridge.sh            ← must have passed
#   slurm_debug_midlayer_bridge_adapter.sh        ← should have passed
#   slurm_debug_explicit_bridge_refiner.sh        ← should have passed
#   slurm_debug_running_bridge.sh                 ← MUST have passed (both modes)
#   → THIS SCRIPT

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
TRAIN_CAND_DIR=runs/path_refiner_clean/data/train_hgrid_K24
VAL_CAND_DIR=runs/path_refiner_clean/data/val_hgrid_K24
TRAIN_FEAT_DIR=runs/path_refiner_residual_interface/features/train_multilayer
VAL_FEAT_DIR=runs/path_refiner_residual_interface/features/val_multilayer
BASELINE_JSON=runs/path_refiner_clean/baselines/saved_candidate_baseline.json

OUTPUT_FINALONLY=runs/path_refiner_running_bridge/boundary_running_finalonly_v0
OUTPUT_FINALPATCH=runs/path_refiner_running_bridge/boundary_running_finalpatchsim_v0

DEBUG_JSON_FO=runs/path_refiner_running_bridge/debug_identity_finalonly/debug_identity.json
DEBUG_JSON_FP=runs/path_refiner_running_bridge/debug_identity_finalpatchsim/debug_identity.json

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Running Bridge Adapter V0 — Training  $(date) ==="
echo ""
echo "    Architecture:"
echo "      RunningBridgeUpdate  : path_state trajectory over update_layers=[2,4]"
echo "      RunningBridgeRefiner : [RESIDUAL + path_state] → delta_final  (zero-init)"
echo "      final_only           : h_refined = h_prime + alpha * delta_final"
echo "      final_patch_sim : diagnostic — adds all delta_Li to h_prime (NOT causal)"
echo "      Decode               : logits = h_refined @ token_emb.T  (full-vocab)"
echo ""
echo "    SMALL_CKPT      : $SMALL_CKPT"
echo "    TRAIN_CAND_DIR  : $TRAIN_CAND_DIR"
echo "    VAL_CAND_DIR    : $VAL_CAND_DIR"
echo "    TRAIN_FEAT_DIR  : $TRAIN_FEAT_DIR"
echo "    VAL_FEAT_DIR    : $VAL_FEAT_DIR"
echo ""
echo "    Runs:"
echo "      1. final_only      → $OUTPUT_FINALONLY"
echo "      2. final_patch_sim → $OUTPUT_FINALPATCH  (diagnostic, NOT causal)"
echo ""
echo "    Config (both runs):"
echo "      update_layers        = 2,4"
echo "      num_path_tokens      = 4"
echo "      bridge_dim           = 256"
echo "      bridge_update_layers = 1  (transformer layers per RunningBridgeUpdate)"
echo "      bridge_heads         = 4"
echo "      refiner_dim          = 256"
echo "      refiner_layers       = 2"
echo "      refiner_heads        = 4"
echo "      batch_size           = 32  (eff_bs=64 with grad_accum=2)"
echo "      steps                = 5000"
echo "      lr                   = 5e-5"
echo "      lambda_kl            = 0.2"
echo "      lambda_delta         = 3e-4"
echo "      lambda_path          = 0.0"
echo "      kl_topk              = 512"
echo "      amp                  = True"
echo ""
echo "    Architecture comparison targets:"
echo "      V1 BridgeResidualAdapter          : +0.003037 NLL improvement"
echo "      MidLayerBridgeAdapter             : +0.003347 NLL improvement"
echo "      ExplicitBridgeRefiner             : +0.003305 NLL improvement"
echo "      RunningBridge final_only      V0  : target > +0.003347  (beat MidLayer)"
echo "      RunningBridge final_patch_sim V0  : diagnostic (NOT causal_multi_write)"
echo ""
echo "    NOTE: causal_multi_write UNAVAILABLE — dataset has no input_ids."
echo ""
echo "    Success thresholds:"
echo "      Weak        > +0.001"
echo "      Meaningful  > +0.003347  (beat MidLayer)"
echo "      Strong      > +0.005"
echo "      Very strong > +0.010"
echo ""

# Check required files exist
for f in "$SMALL_CKPT" "$SUPER_MAP" "$BASELINE_JSON"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

# Check train/val feature configs
for d in "$TRAIN_FEAT_DIR" "$VAL_FEAT_DIR"; do
    if [[ ! -f "$d/config.json" ]]; then
        echo "ERROR: $d/config.json not found." >&2
        echo "       Run slurm_build_multilayer_residual_features.sh first." >&2
        exit 1
    fi
done

# Require debug identity pass for final_only
if [[ ! -f "$DEBUG_JSON_FO" ]]; then
    echo "ERROR: $DEBUG_JSON_FO not found." >&2
    echo "       Run slurm_debug_running_bridge.sh first." >&2
    exit 1
fi
PASS_FO=$(python -c "import json; d=json.load(open('$DEBUG_JSON_FO')); print(d.get('identity_pass', False))" 2>/dev/null || echo "False")
if [[ "$PASS_FO" != "True" ]]; then
    echo "ERROR: identity_pass=$PASS_FO for final_only — debug check failed." >&2
    echo "       Run slurm_debug_running_bridge.sh to fix invariant failures." >&2
    exit 1
fi

# Require debug identity pass for final_patch_sim
if [[ ! -f "$DEBUG_JSON_FP" ]]; then
    echo "ERROR: $DEBUG_JSON_FP not found." >&2
    echo "       Run slurm_debug_running_bridge.sh first (both modes must pass)." >&2
    exit 1
fi
PASS_FP=$(python -c "import json; d=json.load(open('$DEBUG_JSON_FP')); print(d.get('identity_pass', False))" 2>/dev/null || echo "False")
if [[ "$PASS_FP" != "True" ]]; then
    echo "ERROR: identity_pass=$PASS_FP for final_patch_sim — debug check failed." >&2
    echo "       Run slurm_debug_running_bridge.sh to fix invariant failures." >&2
    exit 1
fi

echo "    Preflight OK — final_only and final_patch_sim debug identity checks passed."
echo ""

# Shard counts
N_TRAIN=$(find "$TRAIN_CAND_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_VAL=$(find "$VAL_CAND_DIR"   -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_TF=$(find "$TRAIN_FEAT_DIR"  -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_VF=$(find "$VAL_FEAT_DIR"    -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    train candidate shards : $N_TRAIN"
echo "    val candidate shards   : $N_VAL"
echo "    train feature shards   : $N_TF"
echo "    val feature shards     : $N_VF"
if [[ "$N_TRAIN" -eq 0 || "$N_VAL" -eq 0 || "$N_TF" -eq 0 || "$N_VF" -eq 0 ]]; then
    echo "ERROR: missing shards." >&2; exit 1
fi
echo ""

mkdir -p "$OUTPUT_FINALONLY" "$OUTPUT_FINALPATCH"

# ── Common args ───────────────────────────────────────────────────────────────

COMMON_ARGS=(
    --small_ckpt             $SMALL_CKPT
    --train_cand_dir         $TRAIN_CAND_DIR
    --val_cand_dir           $VAL_CAND_DIR
    --train_feat_dir         $TRAIN_FEAT_DIR
    --val_feat_dir           $VAL_FEAT_DIR
    --baseline_json          $BASELINE_JSON
    --super_map              $SUPER_MAP
    --update_layers          2,4
    --num_path_tokens        4
    --bridge_dim             256
    --bridge_update_layers   1
    --bridge_heads           4
    --refiner_dim            256
    --refiner_layers         2
    --refiner_heads          4
    --ff_mult                4
    --top_fine_regions       24
    --top_superregions       8
    --train_filter           boundary
    --gate_filter            boundary
    --use_filtered_train_loader
    --batch_size             32
    --grad_accum_steps       2
    --steps                  5000
    --eval_every             1000
    --lr                     5e-5
    --lambda_kl              0.2
    --lambda_delta           3e-4
    --lambda_path            0.0
    --kl_topk                512
    --grad_clip              1.0
    --amp
    --eval_before_train
    --fail_on_baseline_mismatch
    --device                 cuda
)

# ── Run 1: final_only ─────────────────────────────────────────────────────────

echo "========================================================================"
echo "  RUN 1 / 2 : write_mode = final_only"
echo "  OUTPUT    : $OUTPUT_FINALONLY"
echo "========================================================================"
echo ""
echo "  h_refined = h_prime + alpha * delta_final"
echo "  Path state traverses update_layers=[2,4], final write via RunningBridgeRefiner."
echo "  Gradient flows: learned_path_tokens → RunningBridgeUpdate × 2 → RunningBridgeRefiner"
echo ""

python scripts/train_running_bridge.py \
    "${COMMON_ARGS[@]}" \
    --write_mode  final_only \
    --output_dir  $OUTPUT_FINALONLY

echo ""
echo "  Run 1 (final_only) complete."
echo ""

# Print final metrics if available
FO_METRICS=$OUTPUT_FINALONLY/final_metrics.json
if [[ -f "$FO_METRICS" ]]; then
    python -c "
import json
m = json.load(open('$FO_METRICS'))
base = m.get('full_vocab_base_nll_all', float('nan'))
gate = m.get('full_vocab_gated_nll_all', float('nan'))
imp  = base - gate
print('  final_only final metrics:')
print(f'    full_vocab_base_nll_all   = {base:.6f}')
print(f'    full_vocab_gated_nll_all  = {gate:.6f}')
print(f'    NLL improvement           = {imp:+.6f}')
print(f'    best step                 = {m.get(\"best_step\", \"?\")}')
" 2>/dev/null || echo "  (could not read final_metrics.json)"
fi

echo ""

# ── Run 2: final_patch_sim ────────────────────────────────────────────────────
#
# DIAGNOSTIC ONLY. Adds delta_L and delta_final all to h_prime.
# This is NOT causal — remaining backbone blocks do not see the injected deltas.
# causal_multi_write would require re-running backbone blocks from injected residuals,
# which requires input_ids not present in the current dataset.
#

echo "========================================================================"
echo "  RUN 2 / 2 : write_mode = final_patch_sim  (diagnostic, NOT causal)"
echo "  OUTPUT    : $OUTPUT_FINALPATCH"
echo "========================================================================"
echo ""
echo "  DIAGNOSTIC MODE. What it does:"
echo "    At each update layer L:"
echo "      h_L_effective = h_L + prev_delta  (gradient signal, but h_L is cached)"
echo "      path_state = RunningBridgeUpdate_L(path_state, h_L_effective, ...)"
echo "      delta_L = write_projs[L](path_state[:,0,:])  (zero-init)"
echo "      prev_delta = delta_L"
echo "    Final:"
echo "      delta_final = RunningBridgeRefiner(h_insert, path_state)"
echo "      h_refined = h_prime + alpha*delta_final + sum_i(alpha_Li * delta_Li)"
echo ""
echo "  What it does NOT do:"
echo "    Does NOT run backbone block3 from h2_refined."
echo "    Does NOT run backbone block5 from h4_refined."
echo "    All deltas accumulate into final h_prime only."
echo ""
echo "  causal_multi_write status: UNAVAILABLE"
echo "    Dataset stores per-position hidden states, not full-sequence tensors."
echo "    Cannot re-run backbone blocks from injected residuals without input_ids."
echo ""

python scripts/train_running_bridge.py \
    "${COMMON_ARGS[@]}" \
    --write_mode  final_patch_sim \
    --output_dir  $OUTPUT_FINALPATCH

echo ""
echo "  Run 2 (final_patch_sim) complete."
echo ""

# Print final metrics if available
FP_METRICS=$OUTPUT_FINALPATCH/final_metrics.json
if [[ -f "$FP_METRICS" ]]; then
    python -c "
import json
m = json.load(open('$FP_METRICS'))
base = m.get('full_vocab_base_nll_all', float('nan'))
gate = m.get('full_vocab_gated_nll_all', float('nan'))
imp  = base - gate
print('  final_patch_sim final metrics:')
print(f'    full_vocab_base_nll_all   = {base:.6f}')
print(f'    full_vocab_gated_nll_all  = {gate:.6f}')
print(f'    NLL improvement           = {imp:+.6f}')
print(f'    best step                 = {m.get(\"best_step\", \"?\")}')
print('    NOTE: diagnostic only — NOT comparable to causal_multi_write.')
" 2>/dev/null || echo "  (could not read final_metrics.json)"
fi

echo ""

# ── Cross-architecture comparison ─────────────────────────────────────────────

echo "========================================================================"
echo "  Cross-architecture comparison"
echo "========================================================================"
echo ""

python -c "
import json, os

def read_result(path):
    for fname in ['best_metrics.json', 'final_metrics.json']:
        fpath = os.path.join(path, fname)
        if os.path.isfile(fpath):
            try:
                m = json.load(open(fpath))
                base = m.get('full_vocab_base_nll_all', float('nan'))
                gate = m.get('full_vocab_gated_nll_all', float('nan'))
                return base, gate, base - gate, fname
            except Exception:
                pass
    return None, None, None, None

rows = [
    ('V1 BridgeResidualAdapter',
     'runs/path_refiner_bridge_residual_adapter/boundary_insert4_v1'),
    ('MidLayer BridgeAdapter',
     'runs/path_refiner_midlayer_bridge_adapter/boundary_midlayer_insert4_v1'),
    ('ExplicitBridgeRefiner',
     'runs/path_refiner_explicit_bridge_refiner/boundary_insert4_refiner2_path4_v1'),
    ('RunningBridge final_only V0',
     'runs/path_refiner_running_bridge/boundary_running_finalonly_v0'),
    ('RunningBridge final_patch_sim V0',
     'runs/path_refiner_running_bridge/boundary_running_finalpatchsim_v0'),
]

print(f'  {\"Architecture\":<35} {\"base_nll\":>10} {\"gated_nll\":>10} {\"improvement\":>12}')
print(f'  {\"-\"*35} {\"-\"*10} {\"-\"*10} {\"-\"*12}')
for name, path in rows:
    base, gate, imp, src = read_result(path)
    if imp is not None:
        flag = ''
        if imp > 0.010:   flag = '  *** VERY STRONG'
        elif imp > 0.005: flag = '  ** STRONG'
        elif imp > 0.003347: flag = '  * BEATS MIDLAYER'
        elif imp > 0.001: flag = '  (weak)'
        print(f'  {name:<35} {base:>10.6f} {gate:>10.6f} {imp:>+12.6f}{flag}')
    else:
        print(f'  {name:<35} {\"(no result)\":>10}')
" 2>/dev/null || echo "  (could not generate comparison table)"

echo ""

# Q&A section
echo "========================================================================"
echo "  Questions to answer after reviewing results"
echo "========================================================================"
echo ""
echo "  Q1: Does final_only beat MidLayer (+0.003347)?"
echo "      If yes  → running path-state over 2 layers is more informative than"
echo "                the single insert-state snapshot in MidLayer."
echo "      If no   → the trajectory doesn't add information beyond the final snapshot."
echo ""
echo "  Q2: Does final_patch_sim beat final_only?"
echo "      If yes  → multiple delta writes to h_prime help even without true causal injection."
echo "      NOTE: this does NOT test whether backbone blocks see the injected residuals."
echo "      If no   → early writes add noise; the final write alone is sufficient."
echo ""
echo "  Q3: Do layer_delta_norms grow during training for final_patch_sim?"
echo "      If they stay near zero → write_projs not learning; consider separate lr."
echo "      If they grow faster than final_delta → intermediate writes dominate."
echo ""
echo "  Q4: Is the NLL improvement on _covered vs _all gap consistent?"
echo "      Large gap → model learns something, but only for covered positions."
echo "      Small gap → gains are truly general, not just on candidate positions."
echo ""
echo "  Q5: Does alpha grow or shrink during training?"
echo "      Large alpha + small delta_norm → scalar compensating for small deltas."
echo "      alpha near 0 → model not using the refined hidden state."
echo ""
echo "  Q6: Should V1 (update_layers=[4], single snapshot) be re-run with"
echo "      update_layers=[2,4] for a fair comparison with RunningBridge?"
echo "      Only relevant if final_only loses — to isolate whether the gain"
echo "      comes from multi-layer trajectory or from just using block 2."
echo ""

echo "=== Running Bridge Adapter V0 — Training COMPLETE  $(date) ==="
echo ""
echo "Output directories:"
echo "  final_only      : $OUTPUT_FINALONLY"
echo "  final_patch_sim : $OUTPUT_FINALPATCH  (diagnostic)"
echo ""
echo "Key files per run:"
echo "  best_refiner.pt      — best checkpoint (saved only if gated < base NLL)"
echo "  best_metrics.json    — metrics at best checkpoint"
echo "  final_metrics.json   — metrics at final step"
echo "  eval_log.csv         — per-eval-step metrics"
echo "  train_log.csv        — per-step training metrics"
echo "  report.md            — cross-architecture comparison report"
echo "  config.json          — full run configuration"
echo "  debug_identity.json  — step-0 identity check (identity_pass=True)"
