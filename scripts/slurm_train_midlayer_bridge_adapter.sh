#!/bin/bash
#SBATCH --job-name=midlayer_train
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
# Mid-Layer Bridge Adapter training — boundary filter, insert_after_block=4.
#
# Architecture:
#   Bridge sequence (1 + n_ctx_layers + 2 + top_fine_k + top_super_k + num_path_tokens):
#     [INSERT_STATE]  ← layer_proj(h_layers[insert_layer_idx]) + insert_marker_emb
#     [layer_0..4]    ← all n_ctx_layers layer tokens (same as V1)
#     [ROUTER] [MEMORY]
#     [fine_0..23] [super_0..7] [PATH]
#   Transformer: L=2  H=4  D=256  ff_mult=4  norm_first=True
#   Delta write: h_refined = h_prime + alpha * out_proj(PATH_out)  (same as V1)
#   Decode:      logits = h_refined @ token_emb.T   (full-vocab, V=50257)
#
# Hypothesis:
#   The bridge conditions on h_layers[insert_layer_idx] = h after block 4.
#   This mid-backbone state may encode richer context before routing uncertainty.
#   Test: does INSERT_STATE conditioning improve over V1 (which only sees all layers equally)?
#
# Primary metric: full_vocab_gated_nll_all (lower is better, must beat step-0 base NLL)
# Compare:
#   V1 (late bridge):  runs/path_refiner_bridge_adapter/bridge_boundary_v1_hardsampler/
#   MidLayer (block4): runs/path_refiner_midlayer_bridge/boundary_insert4_v1/
#   A report.md is generated at end comparing results.
#
# Safety invariants (all asserted at runtime):
#   gold_force_included_rate = 0.0000  — no candidate selection step
#   selection_mode           = no_candidate_selection
#   eval_force_include_gold  = false
#   identity at step 0       — delta_h = 0 (zero-init out_proj)
#   step-0 identity for _all, _covered, masked_cand families
#   masked_cand_base_nll asserted == 3.378606 within 1e-3
#   outside_gate_gated_nll_all asserted == outside_gate_base_nll_all (diff < 1e-5)
#   best checkpoint ONLY saved if full_vocab_gated_nll_all < full_vocab_base_nll_all
#
# Hyperparams vs V1:
#   lr: 5e-5 (V1: 1e-4)  — smaller for tighter convergence
#   lambda_kl: 0.2 (V1: 0.1)  — stronger KL regulariser to stay near base
#   lambda_delta: 3e-4 (V1: 1e-4)  — stronger delta regularisation
#   batch_size=32, grad_accum=2 → eff_bs=64  (same as V1)
#
# Run order:
#   slurm_debug_bridge_residual_adapter.sh    ← V1 must pass
#   slurm_debug_midlayer_bridge_adapter.sh    ← MUST PASS before this
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
OUTPUT_ROOT=runs/path_refiner_midlayer_bridge

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Mid-Layer Bridge Adapter Training  $(date) ==="
echo "    TRAIN_CAND_DIR: $TRAIN_CAND_DIR"
echo "    VAL_CAND_DIR  : $VAL_CAND_DIR"
echo "    TRAIN_FEAT_DIR: $TRAIN_FEAT_DIR"
echo "    VAL_FEAT_DIR  : $VAL_FEAT_DIR"
echo "    BASELINE_JSON : $BASELINE_JSON"
echo "    OUTPUT_ROOT   : $OUTPUT_ROOT"

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

# Require midlayer debug to have passed
DEBUG_OUT=$OUTPUT_ROOT/debug_identity/debug_identity.json
if [[ ! -f "$DEBUG_OUT" ]]; then
    echo "ERROR: $DEBUG_OUT not found." >&2
    echo "       Run slurm_debug_midlayer_bridge_adapter.sh first (must PASS)." >&2
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
print(f\"    insert_after_block  = {d['insert_after_block']}  → insert_layer_idx={d['insert_layer_idx']}\")
print(f\"    layer_ids           = {d['layer_ids']}\")
print(f\"    full_vocab_base_nll_all     = {d['full_vocab_base_nll_all']:.6f}  (PRIMARY threshold)\")
print(f\"    full_vocab_base_nll_covered = {d['full_vocab_base_nll_covered']:.6f}  (secondary)\")
print(f\"    masked_cand_base_nll        = {d['masked_cand_base_nll']:.6f}\")
print()
print('    NOTE: full_vocab_base_nll_all is PRIMARY threshold for best checkpoint.')
print('    Do NOT compare full-vocab NLL to masked-candidate NLL (3.378606 is masked-cand ref).')
" 2>/dev/null || echo "    (could not read debug output)"

mkdir -p "$OUTPUT_ROOT"

# ── Shared args ───────────────────────────────────────────────────────────────
#
# batch_size=32  grad_accum_steps=2  → eff_hard_batch=64  (same as V1 hardsampler)
# Half memory per microstep vs V1's batch_size=64.
# If GPU OOM: reduce batch_size to 16, increase grad_accum to 4.

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
    --num_path_tokens   1
    --steps             10000
    --eval_every        1000
    --batch_size        32
    --grad_accum_steps  2
    --eval_batch_size   64
    --lr                5e-5
    --lambda_kl         0.2
    --lambda_delta      3e-4
    --kl_topk           512
    --grad_clip         1.0
    --amp
    --eval_before_train
    --fail_on_baseline_mismatch
    --use_filtered_train_loader
    --device            cuda
"

# ── Run 1: boundary, insert_after_block=4 ─────────────────────────────────────

echo ""
echo "=== Run 1/1: boundary  insert_after_block=4  $(date) ==="
echo "    Output: $OUTPUT_ROOT/boundary_insert4_v1"
echo "    batch_size=32  grad_accum=2  eff_hard_batch=64"
echo "    INSERT_STATE = h_layers[insert_layer_idx] (block 4 output)"
echo ""

python scripts/train_midlayer_bridge_adapter.py  \
    $SHARED                                       \
    --insert_after_block  4                       \
    --train_filter        boundary                \
    --gate_filter         boundary                \
    --output_dir          $OUTPUT_ROOT/boundary_insert4_v1

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Mid-Layer Bridge training complete  $(date) ==="
echo ""
echo "Primary metric: full_vocab_gated_nll_all  (lower is better)"
echo "  Threshold  : full_vocab_base_nll_all (computed at step 0)"
echo ""
echo "Quick summary:"
for d in "$OUTPUT_ROOT"/boundary_insert4_v1; do
    bm="$d/best_metrics.json"
    fm="$d/final_metrics.json"
    fb="$d/full_vocab_baseline.json"
    variant=$(basename "$d")
    if [[ -f "$fb" ]]; then
        fv_base=$(python -c "import json; d=json.load(open('$fb')); print(f\"{d['full_vocab_base_nll_all']:.6f}\")" 2>/dev/null || echo "N/A")
        ins_blk=$(python -c "import json; d=json.load(open('$fb')); print(d.get('insert_after_block','?'))" 2>/dev/null || echo "?")
        ins_idx=$(python -c "import json; d=json.load(open('$fb')); print(d.get('insert_layer_idx','?'))" 2>/dev/null || echo "?")
        echo "  ${variant}:"
        echo "    insert_after_block      = $ins_blk  (insert_layer_idx=$ins_idx)"
        echo "    full_vocab_base_nll_all = $fv_base  (PRIMARY, step-0 threshold)"
    fi
    if [[ -f "$bm" ]]; then
        no_ckpt=$(python -c "import json; d=json.load(open('$bm')); print(d.get('no_improving_checkpoint', False))" 2>/dev/null || echo "?")
        if [[ "$no_ckpt" == "True" ]]; then
            echo "    RESULT: no improving checkpoint — midlayer bridge did NOT beat full_vocab_base_nll_all"
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
    if [[ -f "$d/report.md" ]]; then
        echo "    Report: $d/report.md"
    fi
done

# Compare to V1 if available
V1_BM=runs/path_refiner_bridge_adapter/bridge_boundary_v1_hardsampler/best_metrics.json
if [[ -f "$V1_BM" ]]; then
    echo ""
    echo "=== Comparison: V1 (late bridge) vs MidLayer (block 4) ==="
    python -c "
import json, os

v1_bm_path = '$V1_BM'
ml_bm_path = '$OUTPUT_ROOT/boundary_insert4_v1/best_metrics.json'

v1 = json.load(open(v1_bm_path)) if os.path.isfile(v1_bm_path) else None
ml = json.load(open(ml_bm_path)) if os.path.isfile(ml_bm_path) else None

def fmtg(d, key):
    if d is None or d.get('no_improving_checkpoint', True): return 'no_ckpt'
    v = d.get(key)
    return f'{v:+.6f}' if v is not None else 'N/A'

keys = [
    ('full_vocab_gain_all',           'fv_gain_all (PRIMARY)'),
    ('full_vocab_inside_gate_gain_all','inside_gate_gain_all'),
    ('full_vocab_gain_covered',        'fv_gain_covered'),
    ('masked_cand_gain',               'masked_cand_gain'),
]
print(f'  {\"Metric\":<32}  {\"V1 (late)\":>14}  {\"MidLayer (blk4)\":>15}  {\"Delta\":>10}')
print(f'  {\"-\"*32}  {\"-\"*14}  {\"-\"*15}  {\"-\"*10}')
for key, label in keys:
    v1_v = fmtg(v1, key)
    ml_v = fmtg(ml, key)
    if v1_v not in ('no_ckpt','N/A') and ml_v not in ('no_ckpt','N/A'):
        dlt = float(ml_v) - float(v1_v)
        dlt_s = f'{dlt:+.6f}'
    else:
        dlt_s = 'N/A'
    print(f'  {label:<32}  {v1_v:>14}  {ml_v:>15}  {dlt_s:>10}')
" 2>/dev/null || echo "    (could not generate comparison)"
fi

echo ""
echo "Questions to answer from results:"
echo "  Q1: Does full_vocab_gated_nll_all < full_vocab_base_nll_all? (PRIMARY: any gain)"
echo "  Q2: Does inside_gate_ref_nll_all < inside_gate_base_nll_all? (residual write helps)"
echo "  Q3: Is outside_gate_diff_all ~0? (gating clean)"
echo "  Q4: Is masked_cand_gain > +0.000530? (beats hard-boundary MLP)"
echo "  Q5: Does midlayer beat V1 on full_vocab_gain_all? (INSERT_STATE helps)"
echo "  Q6: Does midlayer beat V1 on inside_gate_gain_all? (mid-layer signal useful)"
echo ""
echo "If no checkpoint was saved:"
echo "  → INSERT_STATE conditioning at block 4 did not help for this filter/run"
echo "  → Try different insert_after_block (e.g., 2 or 'final') or longer training"
echo ""
echo "Report: $OUTPUT_ROOT/boundary_insert4_v1/report.md"
