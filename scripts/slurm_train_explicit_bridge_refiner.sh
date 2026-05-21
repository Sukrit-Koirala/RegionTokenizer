#!/bin/bash
#SBATCH --job-name=explicit_bridge_train
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
# Explicit Bridge + Refiner — boundary filter, initial run.
#
# Architecture:
#   ExplicitBridge  (translator, bridge_layers=1, bridge_dim=256, bridge_heads=4):
#     INSERT_STATE token (h_layers[:, insert_layer_idx, :])
#     + layer tokens (5) + ROUTER + MEMORY + fine-region (24) + superregion (8)
#     + path query tokens (num_path_tokens=4)
#     → Transformer → path_tokens (B, 4, 256)
#     NO out_proj: Bridge produces path-state tokens, NOT delta_h.
#
#   ExplicitRefiner (thinker, refiner_layers=2, refiner_dim=256, refiner_heads=4):
#     [RESIDUAL_TOKEN (residual_proj(h_insert)) + path_tokens (4)]
#     → Pre-LN Transformer (5 total tokens)
#     → seq[:, 0, :] → out_proj(zero-init) → delta_h (B, d_model)
#
#   Write:  h_prime_refined = h_prime + alpha * delta_h
#   Decode: logits = h_prime_refined @ token_emb.T   (full-vocab, V=50257)
#
# Primary metric: full_vocab_gated_nll_all  (lower = better, threshold = base NLL at step 0)
# Secondary:      full_vocab_gated_nll_covered, masked_cand_gated_nll
#
# Safety invariants (all asserted at runtime):
#   gold_force_included_rate = 0.0000  — no candidate selection step
#   eval_force_include_gold  = false
#   identity at step 0       — delta_h = 0 (zero-init Refiner out_proj)
#   masked_cand_base_nll == 3.378606 within 1e-3  (canonical dataset check)
#   outside_gate_diff < 1e-5  (gate is clean — outside positions unchanged)
#   best checkpoint ONLY saved if full_vocab_gated_nll_all < full_vocab_base_nll_all
#
# Hyperparameters:
#   batch_size=32  grad_accum=2  eff_batch=64  lr=5e-5
#   lambda_kl=0.2  lambda_delta=3e-4  lambda_path=0.0 (path L2 off by default)
#   steps=5000  eval_every=1000  amp
#
# Baseline comparisons (masked-candidate metric):
#   Masked-candidate baseline          : covered_nll = 3.378606
#   Global MLP (masked candidate gain) : +0.001284
#   Hard-boundary MLP (masked gain)    : +0.000530
#   V1 Bridge Adapter (full-vocab)     : ~+0.003  (expected; check best_metrics.json)
#   MidLayer Bridge   (full-vocab)     : ~+0.0028 (expected; check best_metrics.json)
#
# Success criteria (full_vocab_gain_all):
#   Weak      : > 0.0
#   Meaningful: > +0.003
#   Strong    : > +0.005
#   Very strong: > +0.010
#
# Run order:
#   slurm_debug_bridge_residual_adapter.sh        ← must have passed
#   slurm_debug_hard_sampler_bridge.sh            ← must have passed
#   slurm_debug_explicit_bridge_refiner.sh        ← MUST PASS before this
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

# ── Configurable paths ────────────────────────────────────────────────────────

SMALL_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
TRAIN_CAND_DIR=runs/path_refiner_clean/data/train_hgrid_K24
VAL_CAND_DIR=runs/path_refiner_clean/data/val_hgrid_K24
TRAIN_FEAT_DIR=runs/path_refiner_residual_interface/features/train_multilayer
VAL_FEAT_DIR=runs/path_refiner_residual_interface/features/val_multilayer
BASELINE_JSON=runs/path_refiner_clean/baselines/saved_candidate_baseline.json
OUTPUT_ROOT=runs/path_refiner_explicit_bridge_refiner

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Explicit Bridge + Refiner Training  $(date) ==="
echo ""
echo "    Architecture:"
echo "      ExplicitBridge  : translator — evidence → path_tokens (B, 4, 256)  [NO out_proj]"
echo "      ExplicitRefiner : thinker — [RESIDUAL + path_tokens] → delta_h      [zero-init out_proj]"
echo "      Write           : h_prime_refined = h_prime + alpha * delta_h"
echo "      Decode          : full-vocab (V=50257)"
echo ""
echo "    TRAIN_CAND_DIR: $TRAIN_CAND_DIR"
echo "    VAL_CAND_DIR  : $VAL_CAND_DIR"
echo "    TRAIN_FEAT_DIR: $TRAIN_FEAT_DIR"
echo "    VAL_FEAT_DIR  : $VAL_FEAT_DIR"
echo "    BASELINE_JSON : $BASELINE_JSON"
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

# Require debug_identity.json to exist and have identity_pass=True
DEBUG_JSON=$OUTPUT_ROOT/debug_identity/debug_identity.json
if [[ ! -f "$DEBUG_JSON" ]]; then
    echo "ERROR: $DEBUG_JSON not found." >&2
    echo "       Run slurm_debug_explicit_bridge_refiner.sh first (must PASS)." >&2
    exit 1
fi
PASS=$(python -c "import json; d=json.load(open('$DEBUG_JSON')); print(d.get('identity_pass', False))" 2>/dev/null || echo "False")
if [[ "$PASS" != "True" ]]; then
    echo "ERROR: identity_pass=$PASS in $DEBUG_JSON." >&2
    echo "       Run slurm_debug_explicit_bridge_refiner.sh and ensure IDENTITY PASS." >&2
    exit 1
fi
echo "    identity check : PASSED  (debug_identity.json identity_pass=True)"

N_TR=$(find "$TRAIN_CAND_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_VA=$(find "$VAL_CAND_DIR"   -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_TF=$(find "$TRAIN_FEAT_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_VF=$(find "$VAL_FEAT_DIR"   -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    train cand shards   : $N_TR"
echo "    val cand shards     : $N_VA"
echo "    train feat shards   : $N_TF"
echo "    val feat shards     : $N_VF"
if [[ "$N_TR" -eq 0 || "$N_VA" -eq 0 || "$N_TF" -eq 0 || "$N_VF" -eq 0 ]]; then
    echo "ERROR: missing shards." >&2; exit 1
fi

LAYER_IDS=$(python -c "import json; c=json.load(open('$VAL_FEAT_CFG')); print(c['layer_ids'])" 2>/dev/null || echo "?")
N_LAYERS=$(python -c "import json; c=json.load(open('$VAL_FEAT_CFG')); print(c['n_layers_saved'])" 2>/dev/null || echo "?")
echo "    layer_ids           : $LAYER_IDS"
echo "    n_layers_saved      : $N_LAYERS"
echo ""

# Print step-0 baseline from debug run
python -c "
import json
d = json.load(open('$DEBUG_JSON'))
print('    Step-0 baseline (from debug run):')
print(f\"      full_vocab_base_nll_all     = {d['full_vocab_base_nll_all']:.6f}  (PRIMARY checkpoint threshold)\")
print(f\"      full_vocab_base_nll_covered = {d['full_vocab_base_nll_covered']:.6f}  (secondary)\")
print(f\"      masked_cand_base_nll        = {d['masked_cand_base_nll']:.6f}  (ref = 3.378606, different metric)\")
print()
print('    NOTE: full_vocab_base_nll_all is PRIMARY. masked_cand_base_nll is a different metric.')
print('          Do NOT compare full-vocab NLL to masked-candidate NLL.')
" 2>/dev/null || echo "    (could not read debug_identity.json)"

mkdir -p "$OUTPUT_ROOT"

# ── Run: boundary — hard sampler ──────────────────────────────────────────────
#
# use_filtered_train_loader:
#   Yields only rows matching train_filter=boundary (~17% of positions).
#   Every microbatch has batch_size=32 hard boundary examples (not 3–9).
#   effective_hard_batch_size = batch_size * grad_accum_steps = 32 * 2 = 64.
#
# If GPU OOM: reduce --batch_size to 16 --grad_accum_steps 4
#   (same effective batch, less activation memory per microstep)

echo ""
echo "=== Run: boundary (hard sampler, explicit bridge + refiner)  $(date) ==="
echo "    Output : $OUTPUT_ROOT/boundary_insert4_refiner2_path4_v1"
echo "    bridge : layers=1  heads=4  dim=256  num_path_tokens=4"
echo "    refiner: layers=2  heads=4  dim=256"
echo "    batch_size=32  grad_accum=2  eff_hard_batch=64"
echo "    steps=5000  eval_every=1000  lr=5e-5"
echo "    lambda_kl=0.2  lambda_delta=3e-4  lambda_path=0.0  kl_topk=512"
echo ""

python scripts/train_explicit_bridge_refiner.py \
    --small_ckpt        $SMALL_CKPT             \
    --train_cand_dir    $TRAIN_CAND_DIR          \
    --val_cand_dir      $VAL_CAND_DIR            \
    --train_feat_dir    $TRAIN_FEAT_DIR          \
    --val_feat_dir      $VAL_FEAT_DIR            \
    --baseline_json     $BASELINE_JSON           \
    --super_map         $SUPER_MAP               \
    --output_dir        $OUTPUT_ROOT/boundary_insert4_refiner2_path4_v1 \
    --insert_after_block 4                       \
    --num_path_tokens   4                        \
    --bridge_dim        256                      \
    --bridge_layers     1                        \
    --bridge_heads      4                        \
    --refiner_dim       256                      \
    --refiner_layers    2                        \
    --refiner_heads     4                        \
    --ff_mult           4                        \
    --dropout           0.0                      \
    --top_fine_regions  24                       \
    --top_superregions  8                        \
    --train_filter      boundary                 \
    --gate_filter       boundary                 \
    --steps             5000                     \
    --eval_every        1000                     \
    --batch_size        32                       \
    --grad_accum_steps  2                        \
    --eval_batch_size   64                       \
    --lr                5e-5                     \
    --lambda_kl         0.2                      \
    --lambda_delta      3e-4                     \
    --lambda_path       0.0                      \
    --kl_topk           512                      \
    --grad_clip         1.0                      \
    --amp                                        \
    --eval_before_train                          \
    --fail_on_baseline_mismatch                  \
    --use_filtered_train_loader                  \
    --device            cuda

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Explicit Bridge + Refiner training complete  $(date) ==="
echo ""
echo "Primary metric : full_vocab_gated_nll_all  (lower = better)"
echo "  Threshold    : full_vocab_base_nll_all (computed at step 0)"
echo "  NOTE         : full_vocab_base_nll_all != masked_cand_baseline (3.378606) — different metrics"
echo ""

# ── Per-variant quick results ─────────────────────────────────────────────────

EXPL_DIR=$OUTPUT_ROOT/boundary_insert4_refiner2_path4_v1

echo "Results for boundary_insert4_refiner2_path4_v1 (explicit bridge + refiner):"
bm="$EXPL_DIR/best_metrics.json"
fb="$EXPL_DIR/full_vocab_baseline.json"
if [[ -f "$fb" ]]; then
    fv_base=$(python -c "import json; d=json.load(open('$fb')); print(f\"{d['full_vocab_base_nll_all']:.6f}\")" 2>/dev/null || echo "N/A")
    fv_base_cov=$(python -c "import json; d=json.load(open('$fb')); print(f\"{d['full_vocab_base_nll_covered']:.6f}\")" 2>/dev/null || echo "N/A")
    echo "  full_vocab_base_nll_all     = $fv_base  (PRIMARY threshold)"
    echo "  full_vocab_base_nll_covered = $fv_base_cov  (secondary)"
fi
if [[ -f "$bm" ]]; then
    no_ckpt=$(python -c "import json; d=json.load(open('$bm')); print(d.get('no_improving_checkpoint', False))" 2>/dev/null || echo "?")
    if [[ "$no_ckpt" == "True" ]]; then
        echo "  RESULT: no improving checkpoint — model did NOT beat full_vocab_base_nll_all"
    else
        fv_gated=$(python -c "import json; d=json.load(open('$bm')); print(f\"{d['full_vocab_gated_nll_all']:.6f}\")" 2>/dev/null || echo "N/A")
        fv_gain=$(python -c "import json; d=json.load(open('$bm')); print(f\"{d['full_vocab_gain_all']:+.6f}\")" 2>/dev/null || echo "N/A")
        ig_base=$(python -c "import json; d=json.load(open('$bm')); print(f\"{d['full_vocab_inside_gate_base_nll_all']:.6f}\")" 2>/dev/null || echo "N/A")
        ig_ref=$(python -c "import json; d=json.load(open('$bm')); print(f\"{d['full_vocab_inside_gate_ref_nll_all']:.6f}\")" 2>/dev/null || echo "N/A")
        ig_gain=$(python -c "import json; d=json.load(open('$bm')); print(f\"{d['full_vocab_inside_gate_gain_all']:+.6f}\")" 2>/dev/null || echo "N/A")
        og_diff=$(python -c "import json; d=json.load(open('$bm')); print(f\"{abs(d['full_vocab_outside_gate_gated_nll_all']-d['full_vocab_outside_gate_base_nll_all']):.2e}\")" 2>/dev/null || echo "N/A")
        mc_gain=$(python -c "import json; d=json.load(open('$bm')); print(f\"{d['masked_cand_gain']:+.6f}\")" 2>/dev/null || echo "N/A")
        step=$(python -c "import json; d=json.load(open('$bm')); print(d['step'])" 2>/dev/null || echo "N/A")
        gfir=$(python -c "import json; d=json.load(open('$bm')); print(f\"{d['gold_force_included_rate']:.4f}\")" 2>/dev/null || echo "N/A")
        echo "  RESULT: best checkpoint at step $step"
        echo "  full_vocab_gated_nll_all  = $fv_gated  (gain = $fv_gain)  [PRIMARY]"
        echo "  inside_gate_base_nll_all  = $ig_base"
        echo "  inside_gate_ref_nll_all   = $ig_ref  (gain = $ig_gain)"
        echo "  outside_gate_diff_all     = $og_diff  (must be ~0)"
        echo "  masked_cand_gain          = $mc_gain  (compare: hard-boundary MLP +0.000530, global MLP +0.001284)"
        echo "  gold_forced               = $gfir  (must be 0.0000)"
    fi
else
    echo "  (no best_metrics.json — training may have failed)"
fi

# ── Cross-architecture comparison ─────────────────────────────────────────────

echo ""
echo "Cross-architecture comparison (full_vocab_gain_all):"
echo "  Architecture                          | gain_all   | inside_gate_gain | masked_cand_gain"
echo "  --------------------------------------|------------|------------------|------------------"

# V1 Bridge Adapter
V1_BM=runs/path_refiner_bridge_adapter/bridge_boundary_v1_hardsampler/best_metrics.json
if [[ -f "$V1_BM" ]]; then
    v1_no=$(python -c "import json; d=json.load(open('$V1_BM')); print(d.get('no_improving_checkpoint', False))" 2>/dev/null || echo "?")
    if [[ "$v1_no" == "True" ]]; then
        echo "  V1 BridgeResidualAdapter (boundary)  | no ckpt    |       N/A        |       N/A"
    else
        v1_g=$(python -c "import json; d=json.load(open('$V1_BM')); print(f\"{d['full_vocab_gain_all']:+.6f}\")" 2>/dev/null || echo "   N/A")
        v1_ig=$(python -c "import json; d=json.load(open('$V1_BM')); print(f\"{d['full_vocab_inside_gate_gain_all']:+.6f}\")" 2>/dev/null || echo "   N/A")
        v1_mc=$(python -c "import json; d=json.load(open('$V1_BM')); print(f\"{d['masked_cand_gain']:+.6f}\")" 2>/dev/null || echo "   N/A")
        echo "  V1 BridgeResidualAdapter (boundary)  | $v1_g | $v1_ig | $v1_mc"
    fi
else
    echo "  V1 BridgeResidualAdapter (boundary)  | (no results yet)                              "
fi

# MidLayer Bridge Adapter
ML_BM=runs/path_refiner_midlayer_bridge/boundary_insert4_v1/best_metrics.json
if [[ -f "$ML_BM" ]]; then
    ml_no=$(python -c "import json; d=json.load(open('$ML_BM')); print(d.get('no_improving_checkpoint', False))" 2>/dev/null || echo "?")
    if [[ "$ml_no" == "True" ]]; then
        echo "  MidLayer Bridge (insert4, boundary)  | no ckpt    |       N/A        |       N/A"
    else
        ml_g=$(python -c "import json; d=json.load(open('$ML_BM')); print(f\"{d['full_vocab_gain_all']:+.6f}\")" 2>/dev/null || echo "   N/A")
        ml_ig=$(python -c "import json; d=json.load(open('$ML_BM')); print(f\"{d['full_vocab_inside_gate_gain_all']:+.6f}\")" 2>/dev/null || echo "   N/A")
        ml_mc=$(python -c "import json; d=json.load(open('$ML_BM')); print(f\"{d['masked_cand_gain']:+.6f}\")" 2>/dev/null || echo "   N/A")
        echo "  MidLayer Bridge (insert4, boundary)  | $ml_g | $ml_ig | $ml_mc"
    fi
else
    echo "  MidLayer Bridge (insert4, boundary)  | (no results yet)                              "
fi

# ExplicitBridgeRefiner
if [[ -f "$bm" && "$no_ckpt" != "True" ]]; then
    echo "  ExplicitBridgeRefiner (insert4,bnd)  | $fv_gain | $ig_gain | $mc_gain"
else
    echo "  ExplicitBridgeRefiner (insert4,bnd)  | (see above)                                   "
fi

echo ""
echo "MLP baselines (masked-candidate metric only):"
echo "  Global MLP         : masked_cand_gain = +0.001284"
echo "  Hard-boundary MLP  : masked_cand_gain = +0.000530"
echo ""
echo "Success criteria (full_vocab_gain_all):"
echo "  Weak       : > 0.0"
echo "  Meaningful : > +0.003"
echo "  Strong     : > +0.005"
echo "  Very strong: > +0.010"
echo ""

# ── Questions to answer ───────────────────────────────────────────────────────

echo "Questions to answer from results:"
echo "  Q1: Does full_vocab_gated_nll_all < full_vocab_base_nll_all?    (PRIMARY: any gain)"
echo "  Q2: Does inside_gate_ref_nll_all < inside_gate_base_nll_all?    (residual write helps)"
echo "  Q3: Is outside_gate_diff_all ~0?                                (gate clean — outside unchanged)"
echo "  Q4: Is masked_cand_gain > +0.000530?                            (beats hard-boundary MLP)"
echo "  Q5: Is masked_cand_gain > +0.001284?                            (beats global MLP)"
echo "  Q6: Does ExplicitBridgeRefiner outperform V1 Bridge?            (architecture improvement)"
echo ""
echo "If no checkpoint was saved:"
echo "  → explicit bridge/refiner separation did not improve over frozen backbone"
echo "  → try: more bridge path tokens, larger refiner, higher lambda_kl, or longer training"
echo ""
echo "Report: $OUTPUT_ROOT/boundary_insert4_refiner2_path4_v1/  (train_log.csv, eval_log.csv, local_subset_eval.csv)"
echo "        (also see path_norm_mean, path_norm_max, delta_to_h_norm_ratio in eval_log.csv)"
