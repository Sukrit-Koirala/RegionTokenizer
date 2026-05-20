#!/bin/bash
#SBATCH --job-name=rir_train
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
# Residual Interface Refiner (RIR) training — boundary filter, M=256.
#
# Runs two variants sequentially:
#
#   Variant 1: multilayer_crossattn
#     Context  : n_ctx_layers tokens, each projected from backbone h_layers[:, j, :]
#     Requires : pre-extracted features in features/val_multilayer + features/train_multilayer
#     Context size inferred from features/val_multilayer/config.json at runtime
#
#   Variant 2: region_state_crossattn
#     Context  : [CTX from h_prime] + [top_super_k superregion tokens] + [top_fine_k fine tokens]
#     Requires : no extra feature extraction — built from candidate metadata
#
# Both variants:
#   Selection  : topm_no_gold — top-M by base logit only, gold NEVER inserted
#   Init       : residual_scale=1.0 + zero final Linear → delta=0 at step 0
#   Train mask : covered & gold_in_topM  (drop_train_gold_not_selected=True always)
#   lambda_kl  : 0.1 (stronger than CTF's 0.01 — penalises KL divergence from base)
#
# Safety invariants (both variants):
#   gold_force_included_rate = 0.0000  (RuntimeError if violated)
#   selection_mode           = topm_no_gold
#   eval_force_include_gold  = false
#   best checkpoint ONLY saved if gated_covered_nll < official_baseline_nll (3.378606)
#   If no checkpoint saved: "model never beat baseline" printed — not an error
#
# Canonical eval fingerprint (must match):
#   fingerprint  = f57cabcdc46d69ce
#   num_examples = 239,362
#   num_covered  = 227,017
#   coverage     = 0.948425
#
# Comparison targets:
#   force-zero baseline           : covered_nll = 3.378606
#   global MLP refiner (variant C): gated gain  = +0.001284
#   hard boundary MLP             : gated gain  = +0.000530  local_gain = +0.003209
#   no-force CTF (failed)         : inside_gate_model_nll = 5.718 (catastrophic degradation)
#
# Run order:
#   slurm_build_multilayer_residual_features.sh  ← required for multilayer_crossattn
#   slurm_probe_region_info_by_layer.sh          ← optional but recommended before this
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
DATA_ROOT=runs/path_refiner_clean/data
FEATURES_ROOT=runs/path_refiner_residual_interface/features
OUTPUT_ROOT=runs/path_refiner_residual_interface
BASELINE_JSON=runs/path_refiner_clean/baselines/saved_candidate_baseline.json

VAL_DIR=$DATA_ROOT/val_hgrid_K24
TRAIN_DIR=$DATA_ROOT/train_hgrid_K24

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== RIR Training  $(date) ==="
echo "    TRAIN_DIR    : $TRAIN_DIR"
echo "    VAL_DIR      : $VAL_DIR"
echo "    FEATURES_ROOT: $FEATURES_ROOT"
echo "    BASELINE_JSON: $BASELINE_JSON"

for f in "$SMALL_CKPT" "$SUPER_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done
if [[ ! -f "$BASELINE_JSON" ]]; then
    echo "ERROR: $BASELINE_JSON missing — run baseline eval first." >&2; exit 1
fi

N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    val shards   : $N_VAL"
echo "    train shards : $N_TRAIN"
if [[ "$N_VAL" -eq 0 || "$N_TRAIN" -eq 0 ]]; then
    echo "ERROR: missing candidate shards. Run slurm_clean_build_datasets.sh first." >&2; exit 1
fi

# Check multilayer features exist for variant 1
VAL_FEAT_DIR=$FEATURES_ROOT/val_multilayer
TRAIN_FEAT_DIR=$FEATURES_ROOT/train_multilayer
VAL_FEAT_CFG=$VAL_FEAT_DIR/config.json
if [[ ! -f "$VAL_FEAT_CFG" ]]; then
    echo "ERROR: $VAL_FEAT_CFG not found." >&2
    echo "       Run slurm_build_multilayer_residual_features.sh before this script." >&2
    exit 1
fi

N_VAL_FEAT=$(find "$VAL_FEAT_DIR"  -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_TR_FEAT=$(find "$TRAIN_FEAT_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    val feature shards   : $N_VAL_FEAT"
echo "    train feature shards : $N_TR_FEAT"
if [[ "$N_VAL_FEAT" -eq 0 || "$N_TR_FEAT" -eq 0 ]]; then
    echo "ERROR: missing feature shards. Run slurm_build_multilayer_residual_features.sh first." >&2; exit 1
fi

LAYER_IDS=$(python -c "import json; c=json.load(open('$VAL_FEAT_CFG')); print(c['layer_ids'])" 2>/dev/null || echo "?")
echo "    layer_ids            : $LAYER_IDS"

mkdir -p "$OUTPUT_ROOT"

# ── Shared args (both variants) ───────────────────────────────────────────────

SHARED="
    --small_ckpt             $SMALL_CKPT
    --train_dir              $TRAIN_DIR
    --val_dir                $VAL_DIR
    --super_map              $SUPER_MAP
    --baseline_json          $BASELINE_JSON
    --selected_M             256
    --refiner_dim            256
    --num_heads              4
    --ff_mult                4
    --dropout                0.0
    --steps                  5000
    --eval_every             1000
    --batch_size             32
    --eval_batch_size        64
    --lr                     1e-4
    --lambda_kl              0.1
    --lambda_delta           1e-4
    --grad_clip              1.0
    --train_filter           boundary
    --gate_filter            boundary
    --eval_before_train
    --fail_on_baseline_mismatch
    --device                 cuda
"

# ── Run 1: multilayer_crossattn ───────────────────────────────────────────────

echo ""
echo "=== Run 1/2: multilayer_crossattn  $(date) ==="
echo "    Context: per-layer hidden states (auto_even layers + h_final)"
echo "    Requires: $FEATURES_ROOT"
echo ""

python scripts/train_residual_interface_refiner.py    \
    $SHARED                                            \
    --variant        multilayer_crossattn              \
    --features_root  $FEATURES_ROOT                   \
    --output_dir     $OUTPUT_ROOT/multilayer_boundary_M256

# ── Run 2: region_state_crossattn ─────────────────────────────────────────────

echo ""
echo "=== Run 2/2: region_state_crossattn  $(date) ==="
echo "    Context: [CTX from h_prime] + [8 superregion tokens] + [16 fine-region tokens]"
echo "    No pre-extracted features needed."
echo ""

python scripts/train_residual_interface_refiner.py    \
    $SHARED                                            \
    --variant          region_state_crossattn          \
    --top_super_tokens 8                               \
    --top_fine_tokens  16                              \
    --output_dir       $OUTPUT_ROOT/region_state_boundary_M256

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== All RIR runs complete  $(date) ==="
echo ""
echo "Compare best_metrics.json across runs:"
echo "  Key metric : gated_covered_nll  (lower is better)"
echo "  Baseline   : covered_nll=3.378606  (official force-zero)"
echo "  MLP global : gated gain=+0.001284"
echo "  MLP hard   : gated gain=+0.000530  local_gain=+0.003209 (boundary)"
echo "  NOTE: a missing best_refiner.pt means the model never beat baseline."
echo ""
echo "  All runs: selection_mode=topm_no_gold  gold_force_included_rate=0.0000"
echo ""
echo "Quick summary:"
for d in "$OUTPUT_ROOT/multilayer_boundary_M256" "$OUTPUT_ROOT/region_state_boundary_M256"; do
    f="$d/best_metrics.json"
    variant=$(basename "$d")
    if [[ -f "$f" ]]; then
        nll=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gated_covered_nll']:.6f}\")" 2>/dev/null || echo "N/A")
        delta=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['delta_vs_baseline']:+.6f}\")" 2>/dev/null || echo "N/A")
        step=$(python -c "import json; d=json.load(open('$f')); print(d['step'])" 2>/dev/null || echo "N/A")
        ig_m=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['inside_gate_model_nll']:.6f}\")" 2>/dev/null || echo "N/A")
        ig_b=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['inside_gate_base_nll']:.6f}\")" 2>/dev/null || echo "N/A")
        gfir=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gold_force_included_rate']:.4f}\")" 2>/dev/null || echo "N/A")
        echo "  ${variant}:"
        echo "    gated_nll=${nll}  delta=${delta}  step=${step}"
        echo "    inside_gate_model=${ig_m}  inside_gate_base=${ig_b}"
        echo "    gold_forced=${gfir}  (must be 0.0000)"
    else
        echo "  ${variant}:  no best checkpoint (model never beat baseline=${BASELINE_JSON})"
    fi
done
echo ""
echo "Questions to answer from results:"
echo "  Q1: Does multilayer_crossattn beat boundary MLP? (target: delta > +0.000530)"
echo "  Q2: Does multilayer_crossattn beat global MLP?   (target: delta > +0.001284)"
echo "  Q3: Does region_state_crossattn beat CTF no-force? (CTF was catastrophic)"
echo "  Q4: inside_gate_model_nll vs inside_gate_base_nll — is model helping inside gate?"
echo "  Q5: Did any run beat baseline at all? (check for best_refiner.pt presence)"
echo "  Q6: Consult probe report in ${OUTPUT_ROOT}/probes/layer_probe_report.md for layer analysis"
echo ""
echo "Final report template: runs/path_refiner_residual_interface/final_report.md"
