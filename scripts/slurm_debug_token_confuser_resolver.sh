#!/bin/bash
#SBATCH --job-name=tcr_debug_identity
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Token Confuser Resolver — Step-0 Identity Debug
#
# Runs --steps 0 --eval_before_train to verify:
#   1. delta_head is zero-init → delta=0 → refined_logits == base_logits
#   2. gold_force_included_rate = 0.0 (confuser set is clean base top-K)
#   3. full-vocab NLL matches known baseline (3.754938)
#   4. masked-candidate NLL matches canonical (3.378606)
#   5. outside-gate diff < 1e-5 (no contamination)
#
# Expected output lines:
#   variant               = token_confuser_resolver
#   top_k                 = 256
#   confuser_source       = base_topk
#   gold_force_included_rate = 0.0000
#   delta_max_abs            = 0.0
#   full_vocab_base_nll_all      = 3.754938
#   full_vocab_refined_nll_all   = 3.754938
#   diff_all (< 1e-3)            = <tiny>
#   full_vocab_base_nll_covered  = 3.481444
#   diff_covered (< 1e-3)        = <tiny>
#   masked_cand_base_nll         = 3.378606  (canonical = 3.378606)
#   outside_gate_diff (< 1e-5)   = <tiny>
#   gold_in_topK_rate_all        = ...
#   gold_in_topK_rate_gate       = ...
#   [step-0] IDENTITY PASS — token_confuser_resolver  top_k=256  gold_force_included_rate=0.0000
#
# DO NOT compare full-vocab NLL (3.75) to masked-candidate NLL (3.38).
# DO NOT proceed to slurm_train_token_confuser_resolver.sh unless IDENTITY PASS is printed.
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
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
VAL_FEAT_DIR="runs/path_refiner_residual_interface/features/val_multilayer"
TRAIN_CAND_DIR="runs/path_refiner_clean/data/train_hgrid_K24"
TRAIN_FEAT_DIR="runs/path_refiner_residual_interface/features/train_multilayer"
BASELINE_JSON="runs/path_refiner_clean/baselines/saved_candidate_baseline.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
REGION_MAP="runs/region_maps_128/token_to_region.json"
OUTPUT_DIR="runs/token_confuser_resolver/debug_identity"

# ── Preflight checks ─────────────────────────────────────────────────────────
echo "========================================================"
echo " Token Confuser Resolver — Step-0 Identity Check"
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

for CHECK_DIR in "$VAL_CAND_DIR" "$VAL_FEAT_DIR" "$TRAIN_CAND_DIR" "$TRAIN_FEAT_DIR"; do
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

# ── Run identity check (steps=0, eval_before_train) ──────────────────────────
echo "========================================================"
echo " Running step-0 identity check"
echo " output_dir: $OUTPUT_DIR"
echo "========================================================"
echo ""

python scripts/train_token_confuser_resolver.py \
    --small_ckpt        "$SMALL_CKPT"        \
    --train_cand_dir    "$TRAIN_CAND_DIR"    \
    --val_cand_dir      "$VAL_CAND_DIR"      \
    --train_feat_dir    "$TRAIN_FEAT_DIR"    \
    --val_feat_dir      "$VAL_FEAT_DIR"      \
    --baseline_json     "$BASELINE_JSON"     \
    --super_map         "$SUPER_MAP"         \
    --region_map        "$REGION_MAP"        \
    --output_dir        "$OUTPUT_DIR"        \
    --confuser_source   base_topk            \
    --top_k             256                  \
    --train_filter      boundary             \
    --gate_filter       boundary             \
    --resolver_dim      256                  \
    --resolver_layers   2                    \
    --resolver_heads    4                    \
    --steps             0                    \
    --eval_batch_size   64                   \
    --eval_before_train                      \
    --fail_on_baseline_mismatch              \
    --amp

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo "IDENTITY CHECK FAILED (exit code $EXIT_CODE)"
    echo ""
    echo "Checklist:"
    echo "  - delta_head must be zero-init (check model __init__)"
    echo "  - confuser set must be pure base top-K (no gold force)"
    echo "  - outside-gate rows must be untouched by refined logits"
    exit $EXIT_CODE
fi

echo "========================================================"
echo " Identity check complete."
echo " $(date)"
echo "========================================================"
echo ""

# ── Check output files ────────────────────────────────────────────────────────
echo "Output files:"
for F in config.json debug_identity.json final_metrics.json report.md; do
    FPATH="$OUTPUT_DIR/$F"
    if [ -f "$FPATH" ]; then
        SIZE=$(du -sh "$FPATH" 2>/dev/null | cut -f1)
        echo "  [OK]   $FPATH  ($SIZE)"
    else
        echo "  [--]   $FPATH  (not written)"
    fi
done

echo ""
echo "If IDENTITY PASS was printed above, proceed to:"
echo "  sbatch scripts/slurm_train_token_confuser_resolver.sh"
echo ""
echo "NOTE: Do not compare full-vocab NLL (3.75) to masked-candidate NLL (3.38)."
echo "NOTE: gold_force_included_rate must be 0.0000 — confuser set is base top-K only."
echo ""
