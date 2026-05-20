#!/bin/bash
#SBATCH --job-name=bridge_train
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
# Bridge Residual Adapter training — boundary filter, initial run.
#
# Architecture:
#   Bridge sequence (40 tokens):
#     [layer_0..4] [ROUTER] [MEMORY] [fine_0..23] [super_0..7] [PATH]
#   Transformer: L=2  H=4  D=256  ff_mult=4  norm_first=True
#   Delta write: h_refined = h_prime + alpha * out_proj(PATH_out)
#   Decode:      logits = h_refined @ token_emb.T   (full-vocab, V=50257)
#
# Primary metric: full_vocab_gated_nll_all  (full-vocabulary CE over ALL positions, NOT masked)
# Secondary:      full_vocab_gated_nll_covered  (covered positions only)
#                 masked_cand_gated_nll          (for comparison with CTF/MLP refiners)
#
# Safety invariants (all asserted at runtime):
#   gold_force_included_rate = 0.0000  — no candidate selection step
#   selection_mode           = no_candidate_selection
#   eval_force_include_gold  = false
#   identity at step 0       — delta_h = 0 (zero-init out_proj)
#   step-0 identity checked for _all, _covered, masked_cand families
#   masked_cand_base_nll asserted == 3.378606 within 1e-3
#   outside_gate_gated_nll_all asserted == outside_gate_base_nll_all (diff < 1e-5)
#   best checkpoint ONLY saved if full_vocab_gated_nll_all < full_vocab_base_nll_all
#   full_vocab_base_nll_all != masked_cand_baseline (3.378606) — separate metrics
#
# Baseline comparisons (for context):
#   Masked-candidate baseline          : covered_nll = 3.378606
#   Global MLP (masked candidate gain) : +0.001284
#   Hard-boundary MLP (masked gain)    : +0.000530
#   CTF no-force (failed, masked)      : −0.297357  (inside_gate = 5.718 vs base 4.812)
#
# For the bridge adapter, primary success criterion is:
#   full_vocab_gated_nll_all < full_vocab_base_nll_all    (any positive gain)
# Strong success:
#   full_vocab_inside_gate_ref_nll_all < full_vocab_inside_gate_base_nll_all
# Most important:
#   inside-gate improves, outside-gate diff ~0.
#
# Run order:
#   slurm_build_multilayer_residual_features.sh  ← must have completed
#   slurm_debug_bridge_residual_adapter.sh       ← MUST PASS before this
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
OUTPUT_ROOT=runs/path_refiner_bridge_adapter

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Bridge Residual Adapter Training  $(date) ==="
echo "    TRAIN_CAND_DIR: $TRAIN_CAND_DIR"
echo "    VAL_CAND_DIR  : $VAL_CAND_DIR"
echo "    TRAIN_FEAT_DIR: $TRAIN_FEAT_DIR"
echo "    VAL_FEAT_DIR  : $VAL_FEAT_DIR"
echo "    BASELINE_JSON : $BASELINE_JSON"

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

# Require debug to have passed (check for its output)
DEBUG_OUT=runs/path_refiner_bridge_adapter/debug_identity/full_vocab_baseline.json
if [[ ! -f "$DEBUG_OUT" ]]; then
    echo "ERROR: $DEBUG_OUT not found." >&2
    echo "       Run slurm_debug_bridge_residual_adapter.sh first (must PASS)." >&2
    exit 1
fi

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

# Print debug baseline for reference
python -c "
import json
d = json.load(open('$DEBUG_OUT'))
print(f\"    full_vocab_base_nll_all     = {d['full_vocab_base_nll_all']:.6f}  (PRIMARY threshold)\")
print(f\"    full_vocab_base_nll_covered = {d['full_vocab_base_nll_covered']:.6f}  (secondary)\")
print(f\"    masked_cand_base_nll (step-0) = {d['masked_cand_base_nll']:.6f}\")
print(f\"    (masked baseline ref = {d['masked_cand_baseline_ref']:.6f})\")
print()
print('    NOTE: full_vocab_base_nll_all is PRIMARY threshold for best checkpoint.')
print('    full_vocab_base_nll_covered and masked_cand_base_nll are different metrics.')
print('    Do NOT compare full-vocab NLL to masked-candidate NLL.')
" 2>/dev/null || echo "    (could not read debug output)"

mkdir -p "$OUTPUT_ROOT"

# ── Shared args ───────────────────────────────────────────────────────────────

SHARED="
    --small_ckpt        $SMALL_CKPT
    --train_cand_dir    $TRAIN_CAND_DIR
    --val_cand_dir      $VAL_CAND_DIR
    --train_feat_dir    $TRAIN_FEAT_DIR
    --val_feat_dir      $VAL_FEAT_DIR
    --baseline_json     $BASELINE_JSON
    --super_map         $SUPER_MAP
    --bridge_dim        256
    --num_bridge_layers 2
    --num_heads         4
    --ff_mult           4
    --dropout           0.0
    --top_fine_regions  24
    --top_superregions  8
    --steps             5000
    --eval_every        1000
    --batch_size        32
    --eval_batch_size   64
    --lr                1e-4
    --lambda_kl         0.1
    --lambda_delta      1e-4
    --kl_topk           512
    --grad_clip         1.0
    --amp
    --eval_before_train
    --fail_on_baseline_mismatch
    --device            cuda
"

# ── Run 1: boundary (primary) ─────────────────────────────────────────────────

echo ""
echo "=== Run 1/1: boundary  $(date) ==="
echo "    Output: $OUTPUT_ROOT/bridge_boundary_v1"
echo ""

python scripts/train_bridge_residual_adapter.py  \
    $SHARED                                       \
    --train_filter  boundary                      \
    --gate_filter   boundary                      \
    --output_dir    $OUTPUT_ROOT/bridge_boundary_v1

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Bridge training complete  $(date) ==="
echo ""
echo "Primary metric: full_vocab_gated_nll  (lower is better)"
echo "  Threshold  : full_vocab_base_nll (computed at step 0)"
echo "  Note       : full_vocab_base_nll != masked_cand_baseline (3.378606)"
echo ""
echo "Quick summary:"
for d in "$OUTPUT_ROOT"/bridge_boundary_v1; do
    bm="$d/best_metrics.json"
    fm="$d/final_metrics.json"
    fb="$d/full_vocab_baseline.json"
    variant=$(basename "$d")
    if [[ -f "$fb" ]]; then
        fv_base=$(python -c "import json; d=json.load(open('$fb')); print(f\"{d['full_vocab_base_nll_all']:.6f}\")" 2>/dev/null || echo "N/A")
        fv_base_cov=$(python -c "import json; d=json.load(open('$fb')); print(f\"{d['full_vocab_base_nll_covered']:.6f}\")" 2>/dev/null || echo "N/A")
        echo "  ${variant}:"
        echo "    full_vocab_base_nll_all     = $fv_base  (PRIMARY, step-0 threshold)"
        echo "    full_vocab_base_nll_covered = $fv_base_cov  (secondary)"
    fi
    if [[ -f "$bm" ]]; then
        no_ckpt=$(python -c "import json; d=json.load(open('$bm')); print(d.get('no_improving_checkpoint', False))" 2>/dev/null || echo "?")
        if [[ "$no_ckpt" == "True" ]]; then
            echo "    RESULT: no improving checkpoint — bridge did NOT beat full_vocab_base_nll_all"
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
            echo "    RESULT: checkpoint at step $step"
            echo "    full_vocab_gated_nll_all  = $fv_gated  (gain = $fv_gain)  [PRIMARY]"
            echo "    inside_gate_base_nll_all  = $ig_base"
            echo "    inside_gate_ref_nll_all   = $ig_ref  (gain = $ig_gain)"
            echo "    outside_gate_diff_all     = $og_diff  (must be ~0)"
            echo "    masked_cand_gain          = $mc_gain  (compare to MLP +0.001284)"
            echo "    gold_forced               = $gfir  (must be 0.0000)"
        fi
    else
        echo "    (no best_metrics.json — training may have failed)"
    fi
done

echo ""
echo "Questions to answer from results:"
echo "  Q1: Does full_vocab_gated_nll_all < full_vocab_base_nll_all? (PRIMARY: any gain at all)"
echo "  Q2: Does inside_gate_ref_nll_all < inside_gate_base_nll_all? (residual write helps inside gate)"
echo "  Q3: Is outside_gate_diff_all ~0? (gating works — outside unchanged)"
echo "  Q4: Is masked_cand_gain > +0.000530? (beats hard-boundary MLP)"
echo "  Q5: Is masked_cand_gain > +0.001284? (beats global MLP)"
echo "  Q6: Is full_vocab_gain_all > full_vocab_gain_covered? (expect similar)"
echo ""
echo "If no checkpoint was saved:"
echo "  → bridge mechanism did not improve over frozen backbone for this gate/filter"
echo "  → next steps: try harder filters, longer training, or joint training approach"
echo ""
echo "Report: runs/path_refiner_bridge_adapter/bridge_boundary_v1/  (train_log.csv, eval_log.csv, local_subset_eval.csv)"
