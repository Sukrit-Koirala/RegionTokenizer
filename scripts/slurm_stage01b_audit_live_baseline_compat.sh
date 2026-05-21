#!/bin/bash
#SBATCH --job-name=stage01b_live_audit
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
# Stage 01B — Live Baseline Compatibility Audit
#
# Diagnoses why live-context NLL (7.74) does not match cached baseline (3.75).
# Tests multiple context lengths and hidden positions to find the correct
# live-forward convention, then reports the root cause.
#
# Expected diagnosis: backbone trained with seq_len=128 → position embeddings
# for positions ≥ 128 are untrained → ctx_len=256 degrades NLL to 7.74.
# Fix: use ctx_len ≤ 127 in Stage 01 and all downstream stages.
#
# Output:
#   runs/live_context_pipeline/stage01b_live_baseline_audit/
#     live_baseline_audit.json     (live_baseline_compatible: true/false)
#     live_baseline_audit.md       (human-readable report)
#     per_convention_metrics.csv   (NLL for each ctx_len × hidden_pos)
#     examples_live_vs_cached.md   (20 decoded examples)
#
# Run order: Stage 01 → Stage 01B → Stage 02 → Stage 03 → Stage 04 → Stage 05 → Stage 06
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

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
VAL_CTX_DIR="runs/live_context_pipeline/stage01_context_data/val_ctx"
BASELINE_JSON="runs/path_refiner_clean/baselines/saved_candidate_baseline.json"
OUTPUT_DIR="runs/live_context_pipeline/stage01b_live_baseline_audit"

echo "========================================================"
echo " Stage 01B — Live Baseline Compatibility Audit"
echo " $(date)"
echo "========================================================"
echo ""
echo "Known issue: live NLL=7.74 vs canonical NLL=3.75"
echo "Hypothesis: backbone trained on seq_len=128 → ctx_len=256 broken"
echo ""
echo "[preflight] Checking required inputs..."

for F in "$SMALL_CKPT" "$BASELINE_JSON"; do
    if [ ! -f "$F" ]; then echo "ERROR: required file not found: $F"; exit 1; fi
    echo "  [OK] $F"
done

for D in "$VAL_CAND_DIR" "$VAL_CTX_DIR"; do
    if [ ! -d "$D" ]; then echo "ERROR: not found: $D"; exit 1; fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then echo "ERROR: no shards in $D"; exit 1; fi
    echo "  [OK] $D  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_DIR"

python scripts/stage01b_audit_live_baseline_compat.py \
    --small_ckpt             "$SMALL_CKPT"    \
    --val_cand_dir           "$VAL_CAND_DIR"  \
    --val_ctx_dir            "$VAL_CTX_DIR"   \
    --baseline_json          "$BASELINE_JSON" \
    --output_dir             "$OUTPUT_DIR"    \
    --test_ctx_lens          32,64,127,128,256 \
    --eval_batch_size        64               \
    --num_examples_to_print  20               \
    --fail_if_incompatible

EXIT=$?
if [ $EXIT -ne 0 ]; then echo "ERROR: stage01b exited with code $EXIT"; exit $EXIT; fi

echo ""
echo "========================================================"
echo " Stage 01B complete. $(date)"
echo "========================================================"
echo ""
echo "Key output:"
echo "  $OUTPUT_DIR/live_baseline_audit.json"
echo "  $OUTPUT_DIR/per_convention_metrics.csv"
echo "  $OUTPUT_DIR/examples_live_vs_cached.md"
echo ""
echo "Next step:"
echo "  1. Check diagnosis in live_baseline_audit.json → best_convention"
echo "  2. Re-run Stage 01 with the correct --ctx_len from best_convention"
echo "  3. Re-run Stage 02 → 05 with updated ctx dirs"
echo "  4. Only then run Stage 06"
echo ""

python -c "
import json
r = json.load(open('$OUTPUT_DIR/live_baseline_audit.json'))
bc = r.get('best_convention', {})
if bc:
    print(f'Best ctx_len: {bc.get(\"ctx_len\", \"?\")}')
    print(f'Best NLL:     {bc.get(\"live_base_nll_all\", \"?\")}')
    print(f'Canonical:    {bc.get(\"cached_base_nll_all\", \"?\")}')
    print(f'NLL diff:     {bc.get(\"nll_diff_from_canonical\", \"?\")}')
diag = r.get('diagnosis', {})
if diag:
    print()
    print(f'Diagnosis:      {diag.get(\"likely_cause\", \"?\")}')
    print(f'Recommendation: {diag.get(\"recommendation\", \"?\")}')
print()
print(f'live_baseline_compatible = {r.get(\"live_baseline_compatible\", False)}')
"
