#!/bin/bash
#SBATCH --job-name=hard_token_audit
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Hard-Token Failure Anatomy + Recoverability Audit
#
# Offline diagnostic — NO training. Loads cached val shards and
# optional refiner checkpoints, then produces:
#
#   runs/hard_token_failure_audit/
#     config.json                      — run config snapshot
#     context_availability.json/.md    — what's in shard feature keys
#     val_failure_audit.parquet        — full row-level audit (~50 cols × 239K rows)
#     val_failure_audit.csv.gz         — same, gzip-compressed CSV
#     bucket_summary.csv/.md           — 25+ buckets × stats + loss_mass_share
#     loss_mass_report.md              — loss mass breakdown by bucket
#     top_100_bridge_wins.csv          — (if refiner ckpt given) top rows where refiner beats base
#     top_100_bridge_losses.csv        — (if refiner ckpt given) top rows where refiner hurts
#     top_examples.md                  — human-readable per-row examples
#     final_report.md                  — Q1-Q11 diagnostic answers
#
# DO NOT compare full-vocab NLL to masked-candidate NLL as if they are the same.
# gold token is used only for labels/metrics/audit categories, not for model inputs.
# causal_multi_write is UNAVAILABLE (dataset stores per-position h_layers, no input_ids).
#
# Canonical invariants (asserted by the Python script if baseline_json provided):
#   FINGERPRINT               = 09b0a71955cc9c43
#   num_examples              = 239,362
#   num_covered               = 227,017
#   coverage                  = 0.948425
#   MASKED_CAND_BASELINE_NLL  = 3.378606   (masked-candidate metric)
#   full_vocab_base_nll_all   = 3.754938   (full-vocab metric — different scale)
#
# Run order:
#   slurm_clean_build_datasets.sh
#   slurm_build_multilayer_residual_features.sh
#   → this script (no training prerequisite, just reads val shards + optional ckpts)
#
# After interpreting this audit, decide whether to continue architecture search.
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

# ── Paths ───────────────────────────────────────────────────────────────────
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
VAL_FEAT_DIR="runs/path_refiner_residual_interface/features/val_multilayer"
BASELINE_JSON="runs/path_refiner_clean/baselines/saved_candidate_baseline.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
REGION_MAP="runs/region_maps_128/token_to_region.json"
OUTPUT_DIR="runs/hard_token_failure_audit"

# ── Optional refiner checkpoints (leave empty to skip) ──────────────────────
BRIDGE_V1_CKPT=""
MIDLAYER_CKPT=""
EXPLICIT_REFINER_CKPT=""
RUNNING_BRIDGE_CKPT=""

# ── Eval settings ────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE=64
TOPK_RANK=1000
MARGIN_THRESH=0.1
ENTROPY_THRESH=2.0

# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Hard-Token Failure Audit"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight checks ─────────────────────────────────────────────────────────
echo "[preflight] Checking required inputs..."

if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: small_ckpt not found: $SMALL_CKPT"
    exit 1
fi

if [ ! -d "$VAL_CAND_DIR" ]; then
    echo "ERROR: val_cand_dir not found: $VAL_CAND_DIR"
    exit 1
fi

if [ ! -d "$VAL_FEAT_DIR" ]; then
    echo "ERROR: val_feat_dir not found: $VAL_FEAT_DIR"
    exit 1
fi

if [ ! -f "$BASELINE_JSON" ]; then
    echo "ERROR: baseline_json not found: $BASELINE_JSON"
    exit 1
fi

if [ ! -f "$SUPER_MAP" ]; then
    echo "ERROR: super_map not found: $SUPER_MAP"
    exit 1
fi

if [ ! -f "$REGION_MAP" ]; then
    echo "ERROR: region_map not found: $REGION_MAP"
    exit 1
fi

# Check val cand shards exist
CAND_SHARDS=$(find "$VAL_CAND_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$CAND_SHARDS" -eq 0 ]; then
    echo "ERROR: No shard_*.pt files found in $VAL_CAND_DIR"
    exit 1
fi
echo "  val_cand_dir: $CAND_SHARDS shards found"

# Check val feat shards exist
FEAT_SHARDS=$(find "$VAL_FEAT_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$FEAT_SHARDS" -eq 0 ]; then
    echo "ERROR: No shard_*.pt files found in $VAL_FEAT_DIR"
    exit 1
fi
echo "  val_feat_dir: $FEAT_SHARDS shards found"

if [ "$CAND_SHARDS" -ne "$FEAT_SHARDS" ]; then
    echo "WARNING: shard count mismatch — cand=$CAND_SHARDS feat=$FEAT_SHARDS"
    echo "  Audit will continue but cross-shard alignment may fail on mismatch."
fi

# Validate optional checkpoints if provided
for CKPT_VAR in BRIDGE_V1_CKPT MIDLAYER_CKPT EXPLICIT_REFINER_CKPT RUNNING_BRIDGE_CKPT; do
    CKPT_PATH="${!CKPT_VAR}"
    if [ -n "$CKPT_PATH" ] && [ ! -f "$CKPT_PATH" ]; then
        echo "WARNING: $CKPT_VAR not found: $CKPT_PATH — will be skipped"
    elif [ -n "$CKPT_PATH" ]; then
        echo "  $CKPT_VAR: $CKPT_PATH  [OK]"
    fi
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

mkdir -p logs "$OUTPUT_DIR"

# ── Build optional ckpt flags ─────────────────────────────────────────────────
OPTIONAL_FLAGS=""
if [ -n "$BRIDGE_V1_CKPT" ] && [ -f "$BRIDGE_V1_CKPT" ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --bridge_v1_ckpt $BRIDGE_V1_CKPT"
fi
if [ -n "$MIDLAYER_CKPT" ] && [ -f "$MIDLAYER_CKPT" ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --midlayer_ckpt $MIDLAYER_CKPT"
fi
if [ -n "$EXPLICIT_REFINER_CKPT" ] && [ -f "$EXPLICIT_REFINER_CKPT" ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --explicit_refiner_ckpt $EXPLICIT_REFINER_CKPT"
fi
if [ -n "$RUNNING_BRIDGE_CKPT" ] && [ -f "$RUNNING_BRIDGE_CKPT" ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --running_bridge_ckpt $RUNNING_BRIDGE_CKPT"
fi

# ── Run audit ────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Running audit_hard_token_failures.py"
echo " output_dir: $OUTPUT_DIR"
echo "========================================================"
echo ""

python scripts/audit_hard_token_failures.py \
    --small_ckpt        "$SMALL_CKPT"        \
    --val_cand_dir      "$VAL_CAND_DIR"      \
    --val_feat_dir      "$VAL_FEAT_DIR"      \
    --baseline_json     "$BASELINE_JSON"     \
    --super_map         "$SUPER_MAP"         \
    --region_map        "$REGION_MAP"        \
    --output_dir        "$OUTPUT_DIR"        \
    --eval_batch_size   "$EVAL_BATCH_SIZE"   \
    --topk_rank         "$TOPK_RANK"         \
    --margin_thresh     "$MARGIN_THRESH"     \
    --entropy_thresh    "$ENTROPY_THRESH"    \
    --include_context_check                  \
    --write_examples                         \
    $OPTIONAL_FLAGS

AUDIT_EXIT=$?

echo ""
if [ $AUDIT_EXIT -ne 0 ]; then
    echo "ERROR: audit script exited with code $AUDIT_EXIT"
    exit $AUDIT_EXIT
fi

echo "========================================================"
echo " Audit complete."
echo " $(date)"
echo "========================================================"
echo ""

# ── Summarise outputs ─────────────────────────────────────────────────────────
echo "Output files:"
for F in \
    config.json \
    context_availability.json \
    context_availability.md \
    val_failure_audit.parquet \
    val_failure_audit.csv.gz \
    bucket_summary.csv \
    bucket_summary.md \
    loss_mass_report.md \
    top_examples.md \
    final_report.md \
    top_100_bridge_wins.csv \
    top_100_bridge_losses.csv
do
    FPATH="$OUTPUT_DIR/$F"
    if [ -f "$FPATH" ]; then
        SIZE=$(du -sh "$FPATH" 2>/dev/null | cut -f1)
        echo "  [OK]   $FPATH  ($SIZE)"
    else
        echo "  [--]   $FPATH  (not written — optional or skipped)"
    fi
done

echo ""
echo "Key reports to read first:"
echo "  1. $OUTPUT_DIR/context_availability.md  — what's in shard keys"
echo "  2. $OUTPUT_DIR/loss_mass_report.md       — which buckets carry the loss"
echo "  3. $OUTPUT_DIR/bucket_summary.md         — recoverability per bucket"
echo "  4. $OUTPUT_DIR/final_report.md           — Q1-Q11 diagnostic answers"
echo ""
echo "NOTE: Do not compare masked-candidate NLL (3.38) to full-vocab NLL (3.75)."
echo "NOTE: Do not interpret causal conclusions from cached h_layers."
echo "NOTE: Do not continue architecture search until this audit is interpreted."
echo ""
