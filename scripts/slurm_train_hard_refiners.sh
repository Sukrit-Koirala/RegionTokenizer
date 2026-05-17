#!/bin/bash
#SBATCH --job-name=hard_refiners
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Hard-position-only path refiner experiment — 6 filter runs, Variant C.
#
# Each run trains a RicherMLPRefiner ONLY on positions passing a filter
# (boundary / tight_boundary / type_A / type_A_or_B_resolvable /
#  router_top8_miss / hard_union), then evaluates with a gated global eval
# where delta=0 outside the gate.
#
# Robust init: residual_scale=1.0 + zero final MLP layer → delta=0 at step 0.
#
# ALL full-val evals assert:
#   fingerprint  = f57cabcdc46d69ce
#   num_examples = 239,362
#   num_covered  = 227,017
#   coverage     = 0.948425
#
# Run order:
#   slurm_clean_build_datasets.sh
#   → slurm_debug_periodic_eval_consistency.sh   (pass before running this)
#   → this script
#
# Primary metric: gated_covered_nll (full val with delta=0 outside gate).
# Compare across runs via runs/path_refiner_hard/*/best_metrics.json.

source ~/miniconda3/bin/activate
conda activate learned_regions

set -eo pipefail

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
OUTPUT_ROOT=runs/path_refiner_hard

VAL_DIR=$DATA_ROOT/val_hgrid_K24
TRAIN_DIR=$DATA_ROOT/train_hgrid_K24
BASELINE_JSON=runs/path_refiner_clean/baselines/saved_candidate_baseline.json

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Hard Refiner Training  $(date) ==="
echo "    TRAIN_DIR    : $TRAIN_DIR"
echo "    VAL_DIR      : $VAL_DIR"
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
    echo "ERROR: missing shards. Run slurm_clean_build_datasets.sh first." >&2; exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# ── Shared args ───────────────────────────────────────────────────────────────

SHARED="
    --val_dir           $VAL_DIR
    --train_dir         $TRAIN_DIR
    --small_ckpt        $SMALL_CKPT
    --super_map         $SUPER_MAP
    --d_region          32
    --d_hidden          256
    --steps             10000
    --eval_every        1000
    --eval_batch_size   64
    --lr                3e-4
    --lambda_kl         0.01
    --lambda_delta      1e-4
    --eval_before_train
    --official_baseline $BASELINE_JSON
    --fail_on_baseline_mismatch
    --device            cuda
"

# ── Run 1: boundary ───────────────────────────────────────────────────────────

echo ""
echo "=== Run 1/6: boundary  $(date) ==="

python scripts/train_hard_position_refiner.py \
    $SHARED                                   \
    --train_filter  boundary                  \
    --gate_filter   boundary                  \
    --batch_size    64                        \
    --output_dir    $OUTPUT_ROOT/variant_C_boundary

# ── Run 2: tight_boundary ─────────────────────────────────────────────────────

echo ""
echo "=== Run 2/6: tight_boundary  $(date) ==="

python scripts/train_hard_position_refiner.py \
    $SHARED                                   \
    --train_filter  tight_boundary            \
    --gate_filter   tight_boundary            \
    --batch_size    64                        \
    --output_dir    $OUTPUT_ROOT/variant_C_tight_boundary

# ── Run 3: type_A ─────────────────────────────────────────────────────────────

echo ""
echo "=== Run 3/6: type_A  $(date) ==="

python scripts/train_hard_position_refiner.py \
    $SHARED                                   \
    --train_filter  type_A                    \
    --gate_filter   type_A                    \
    --batch_size    64                        \
    --output_dir    $OUTPUT_ROOT/variant_C_type_A

# ── Run 4: type_A_or_B_resolvable ────────────────────────────────────────────

echo ""
echo "=== Run 4/6: type_A_or_B_resolvable  $(date) ==="

python scripts/train_hard_position_refiner.py \
    $SHARED                                   \
    --train_filter  type_A_or_B_resolvable    \
    --gate_filter   type_A_or_B_resolvable    \
    --batch_size    64                        \
    --output_dir    $OUTPUT_ROOT/variant_C_type_A_or_B

# ── Run 5: router_top8_miss ───────────────────────────────────────────────────

echo ""
echo "=== Run 5/6: router_top8_miss  $(date) ==="

python scripts/train_hard_position_refiner.py \
    $SHARED                                   \
    --train_filter  router_top8_miss          \
    --gate_filter   router_top8_miss          \
    --batch_size    64                        \
    --output_dir    $OUTPUT_ROOT/variant_C_router_top8_miss

# ── Run 6: hard_union ────────────────────────────────────────────────────────

echo ""
echo "=== Run 6/6: hard_union  $(date) ==="

python scripts/train_hard_position_refiner.py \
    $SHARED                                   \
    --train_filter  hard_union                \
    --gate_filter   hard_union                \
    --batch_size    64                        \
    --output_dir    $OUTPUT_ROOT/variant_C_hard_union

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== All 6 runs complete  $(date) ==="
echo ""
echo "Compare best_metrics.json across runs:"
echo "  Key metric : gated_covered_nll  (lower is better)"
echo "  Baseline   : covered_nll=3.378606  (official force-zero)"
echo ""
echo "Quick summary:"
for d in "$OUTPUT_ROOT"/variant_C_*; do
    f="$d/best_metrics.json"
    if [[ -f "$f" ]]; then
        filter=$(basename "$d" | sed 's/variant_C_//')
        nll=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gated_covered_nll']:.6f}\")" 2>/dev/null || echo "N/A")
        delta=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['delta_vs_baseline']:+.6f}\")" 2>/dev/null || echo "N/A")
        step=$(python -c "import json; d=json.load(open('$f')); print(d['step'])" 2>/dev/null || echo "N/A")
        gate=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gate_rate']:.4f}\")" 2>/dev/null || echo "N/A")
        echo "  ${filter:30}  gated_nll=${nll}  delta=${delta}  step=${step}  gate_rate=${gate}"
    fi
done
echo ""
echo "For per-subset breakdown, inspect local_subset_eval.csv in each run directory."
