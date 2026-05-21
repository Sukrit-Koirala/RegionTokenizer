#!/bin/bash
#SBATCH --job-name=tcr_top256_bnd_v1
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Token Confuser Resolver V1 — top256_boundary_v1
#
# Architecture:
#   confuser_source = base_topk  (pure base top-K, NO gold force)
#   top_k           = 256
#   resolver_dim    = 256
#   resolver_layers = 2  (TransformerEncoder, pre-LN)
#   resolver_heads  = 4
#
# Training:
#   train_filter = boundary  (hard-token subset only)
#   gate_filter  = boundary  (inside-gate eval = boundary rows)
#   batch_size   = 16  (try 32 if GPU memory allows after identity check)
#   grad_accum   = 4   (effective batch = 64)
#   steps        = 5000
#   eval_every   = 1000
#   lr           = 5e-5  (cosine to 5e-6)
#   lambda_rank  = 0.5
#   lambda_kl    = 0.1
#   lambda_delta = 1e-4
#   rank_margin  = 0.1
#
# Loss:
#   CE(refined_logits, gold)
#   + 0.5  * rank_loss  (only when gold in top-K, hardest wrong token, margin=0.1)
#   + 0.1  * KL(refined_topK || base_topK)
#   + 1e-4 * ||delta||²
#
# Success criteria (vs. full-vocab baseline 3.754938):
#   full_vocab_gain_all  > 0.05  nats  (whole val set)
#   inside_gate_gain     > 0.15  nats  (boundary rows only)
#   outside_gate_diff    < 0.005 nats  (leak budget)
#
# Canonical invariants (asserted at step 0):
#   FINGERPRINT           = 09b0a71955cc9c43
#   num_examples          = 239,362
#   num_covered           = 227,017
#   coverage              = 0.948425
#   MASKED_CAND_BASELINE  = 3.378606
#   full_vocab_base_nll   = 3.754938
#
# Prerequisite: run slurm_debug_token_confuser_resolver.sh first and
# confirm "IDENTITY PASS" before submitting this job.
#
# Run order:
#   slurm_clean_build_datasets.sh
#   slurm_build_multilayer_residual_features.sh
#   slurm_audit_hard_token_failures.sh
#   slurm_debug_token_confuser_resolver.sh   ← confirm IDENTITY PASS
#   → this script
# ─────────────────────────────────────────────────────────────────────────────

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
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_CAND_DIR="runs/path_refiner_clean/data/train_hgrid_K24"
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
TRAIN_FEAT_DIR="runs/path_refiner_residual_interface/features/train_multilayer"
VAL_FEAT_DIR="runs/path_refiner_residual_interface/features/val_multilayer"
BASELINE_JSON="runs/path_refiner_clean/baselines/saved_candidate_baseline.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
REGION_MAP="runs/region_maps_128/token_to_region.json"
OUTPUT_DIR="runs/token_confuser_resolver/top256_boundary_v1"

# ── Preflight checks ─────────────────────────────────────────────────────────
echo "========================================================"
echo " Token Confuser Resolver V1 — top256_boundary_v1"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking required inputs..."

for CHECK_FILE in "$SMALL_CKPT" "$BASELINE_JSON" "$SUPER_MAP" "$REGION_MAP"; do
    if [ ! -f "$CHECK_FILE" ]; then
        echo "ERROR: required file not found: $CHECK_FILE"
        exit 1
    fi
    echo "  [OK] $CHECK_FILE"
done

for CHECK_DIR in "$TRAIN_CAND_DIR" "$VAL_CAND_DIR" "$TRAIN_FEAT_DIR" "$VAL_FEAT_DIR"; do
    if [ ! -d "$CHECK_DIR" ]; then
        echo "ERROR: required directory not found: $CHECK_DIR"
        exit 1
    fi
    N=$(find "$CHECK_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then
        echo "ERROR: no shard_*.pt files in $CHECK_DIR"
        exit 1
    fi
    echo "  [OK] $CHECK_DIR  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present."
echo ""

# ── Activate environment ─────────────────────────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

mkdir -p "$OUTPUT_DIR"

# ── Train ────────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Training token_confuser_resolver"
echo " output_dir: $OUTPUT_DIR"
echo "========================================================"
echo ""

# NOTE: --batch_size 16 is conservative. If GPU memory allows after the
# identity check (monitor nvidia-smi), increase to --batch_size 32 and
# set --grad_accum_steps 2 to keep effective batch = 64.

python scripts/train_token_confuser_resolver.py \
    --small_ckpt         "$SMALL_CKPT"        \
    --train_cand_dir     "$TRAIN_CAND_DIR"    \
    --val_cand_dir       "$VAL_CAND_DIR"      \
    --train_feat_dir     "$TRAIN_FEAT_DIR"    \
    --val_feat_dir       "$VAL_FEAT_DIR"      \
    --baseline_json      "$BASELINE_JSON"     \
    --super_map          "$SUPER_MAP"         \
    --region_map         "$REGION_MAP"        \
    --output_dir         "$OUTPUT_DIR"        \
    --confuser_source    base_topk            \
    --top_k              256                  \
    --train_filter       boundary             \
    --gate_filter        boundary             \
    --use_filtered_train_loader               \
    --resolver_dim       256                  \
    --resolver_layers    2                    \
    --resolver_heads     4                    \
    --batch_size         16                   \
    --grad_accum_steps   4                    \
    --steps              5000                 \
    --eval_every         1000                 \
    --eval_batch_size    64                   \
    --lr                 5e-5                 \
    --lambda_rank        0.5                  \
    --lambda_kl          0.1                  \
    --lambda_delta       1e-4                 \
    --rank_margin        0.1                  \
    --grad_clip          1.0                  \
    --amp                                     \
    --eval_before_train                       \
    --fail_on_baseline_mismatch

TRAIN_EXIT=$?

echo ""
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: training script exited with code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "========================================================"
echo " Training complete."
echo " $(date)"
echo "========================================================"
echo ""

# ── Summarise outputs ─────────────────────────────────────────────────────────
echo "Output files:"
for F in \
    config.json \
    debug_identity.json \
    full_vocab_baseline.json \
    train_log.csv \
    eval_log.csv \
    bucket_eval.csv \
    best_resolver.pt \
    best_metrics.json \
    final_metrics.json \
    report.md
do
    FPATH="$OUTPUT_DIR/$F"
    if [ -f "$FPATH" ]; then
        SIZE=$(du -sh "$FPATH" 2>/dev/null | cut -f1)
        echo "  [OK]   $FPATH  ($SIZE)"
    else
        echo "  [--]   $FPATH  (not written — no improvement or optional)"
    fi
done

echo ""
echo "Key metrics to check in final_metrics.json:"
echo "  best_full_vocab_gain_all   > 0.05  nats  (target: whole-val improvement)"
echo "  best_inside_gate_gain      > 0.15  nats  (target: boundary-row improvement)"
echo "  no_improving_checkpoint    = false        (must have beaten baseline)"
echo ""
echo "If best_resolver.pt was NOT written, the model never beat the baseline."
echo "Diagnose via eval_log.csv: check gold_in_topK_rate_gate and delta_abs_mean."
echo ""
echo "NOTE: Do not compare full-vocab NLL (3.75) to masked-candidate NLL (3.38)."
echo "NOTE: outside_gate rows are untouched — leak budget < 0.005 nats."
echo ""
