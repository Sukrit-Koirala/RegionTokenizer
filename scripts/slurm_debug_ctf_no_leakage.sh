#!/bin/bash
#SBATCH --job-name=ctf_noleak_debug
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# CTF No-Leakage Smoke Test
#
# Verifies that after the gold-leakage fix:
#   1. selection_mode=topm_no_gold (gold never consulted at eval/train)
#   2. gold_force_included_rate = 0.0000 (hard assertion inside canonical_eval_ctf)
#   3. CTF is exact identity at step 0 (delta=0 → nll matches force-zero)
#   4. Fingerprint, num_examples, num_covered, coverage match official baseline
#
# Run this BEFORE slurm_train_candidate_transformer_refiner.sh to confirm
# the leakage fix is active.
#
# Expected output (PASS conditions):
#   gold_force_included_rate = 0.0000   ← must be exactly 0
#   identity check: PASS  nll_diff < 1e-4
#   with_delta_nll ≈ force_zero_nll ≈ 3.378606
#
# Pre-requisites: slurm_clean_build_datasets.sh must have completed.

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

# ── Paths ────────────────────────────────────────────────────────────────────

SMALL_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
VAL_DIR=runs/path_refiner_clean/data/val_hgrid_K24
BASELINE_JSON=runs/path_refiner_clean/baselines/saved_candidate_baseline.json
OUTPUT_DIR=runs/path_refiner_candidate_transformer/no_leakage_debug

# ── Preflight ────────────────────────────────────────────────────────────────

echo "=== CTF No-Leakage Smoke Test  $(date) ==="
echo "    VAL_DIR     : $VAL_DIR"
echo "    BASELINE    : $BASELINE_JSON"
echo "    OUTPUT_DIR  : $OUTPUT_DIR"
echo ""
echo "    This test verifies that gold_force_included_rate == 0.0000"
echo "    and that CTF is an exact identity at step 0 (delta=0)."
echo ""

for f in "$SMALL_CKPT" "$SUPER_MAP" "$BASELINE_JSON"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    val shards  : $N_VAL"
if [[ "$N_VAL" -eq 0 ]]; then
    echo "ERROR: no val shards in $VAL_DIR" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Run debug (identity + baseline checks, exits after smoke tests) ───────────

echo "Running CTF debug (--debug_only) ..."
echo "Expecting: gold_force_included_rate = 0.0000 (hard assertion)"
echo ""

python scripts/train_candidate_transformer_refiner.py \
    --small_ckpt        $SMALL_CKPT                   \
    --val_dir           $VAL_DIR                      \
    --super_map         $SUPER_MAP                    \
    --official_baseline $BASELINE_JSON                \
    --output_dir        $OUTPUT_DIR                   \
    --selected_M        256                           \
    --refiner_dim       256                           \
    --num_layers        2                             \
    --num_heads         4                             \
    --ff_mult           4                             \
    --eval_batch_size   64                            \
    --train_selection_mode topm_no_gold               \
    --eval_selection_mode  topm_no_gold               \
    --fail_on_baseline_mismatch                       \
    --debug_only                                      \
    --device            cuda

echo ""
echo "=== CTF No-Leakage Smoke Test PASSED  $(date) ==="
echo ""
echo "Confirmed:"
echo "  selection_mode           = topm_no_gold"
echo "  eval_force_include_gold  = false"
echo "  gold_force_included_rate = 0.0000  (assertion passed)"
echo "  identity at step 0       = PASS"
echo ""
echo "Safe to proceed with slurm_train_candidate_transformer_refiner.sh"
