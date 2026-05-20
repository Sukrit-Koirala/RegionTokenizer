#!/bin/bash
#SBATCH --job-name=ctf_debug
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Smoke-test the Candidate-set Transformer Refiner (CTF) before full training.
#
# Checks (in order):
#   1. Import / instantiate model
#   2. Init identity (delta=0 at step 0, max|delta|<1e-5)
#   3. Gold coverage in top-M on first val shard
#   4. Force-zero full-val  → assert == baseline
#   5. With-delta full-val  → assert == force-zero (identity PASS)
#
# Expected terminal output:
#   force_zero_nll           = 3.378606
#   with_delta_nll           = 3.378606
#   coverage                 = 0.948425
#   fingerprint              = f57cabcdc46d69ce
#   max_abs_delta            = 0.0
#   identity PASS
#
# Pre-requisites:
#   slurm_clean_build_datasets.sh  must have run (val shards present)
#   slurm_debug_periodic_eval_consistency.sh  must have passed
#   runs/path_refiner_clean/baselines/saved_candidate_baseline.json  must exist
#
# Run order:
#   slurm_clean_build_datasets.sh
#   → slurm_debug_periodic_eval_consistency.sh
#   → THIS SCRIPT   (pass before running slurm_train_candidate_transformer_refiner.sh)
#   → slurm_train_candidate_transformer_refiner.sh

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
BASELINE_JSON=runs/path_refiner_clean/baselines/saved_candidate_baseline.json

VAL_DIR=$DATA_ROOT/val_hgrid_K24

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== CTF Debug  $(date) ==="
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
echo "    val shards : $N_VAL"
if [[ "$N_VAL" -eq 0 ]]; then
    echo "ERROR: no val shards. Run slurm_clean_build_datasets.sh first." >&2; exit 1
fi

# ── Debug run ─────────────────────────────────────────────────────────────────

echo ""
echo "=== CTF smoke test  $(date) ==="

python scripts/train_candidate_transformer_refiner.py \
    --debug_only                                      \
    --val_dir           $VAL_DIR                      \
    --small_ckpt        $SMALL_CKPT                   \
    --super_map         $SUPER_MAP                    \
    --official_baseline $BASELINE_JSON                \
    --fail_on_baseline_mismatch                       \
    --selected_M        256                           \
    --refiner_dim       256                           \
    --num_layers        2                             \
    --num_heads         4                             \
    --ff_mult           4                             \
    --eval_batch_size   64                            \
    --device            cuda

echo ""
echo "=== CTF debug PASS  $(date) ==="
echo "    All identity and baseline checks passed."
echo "    Safe to launch slurm_train_candidate_transformer_refiner.sh"
