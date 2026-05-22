#!/bin/bash
#SBATCH --job-name=ret_failure_analysis
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
# Retrieval Failure Analysis — Diagnostic bucket analysis
#
# Iterates the val set, recomputes retrieval for the requested key,
# and writes human-readable markdown reports per failure bucket.
#
# Inputs (from logitfix run):
#   patched train: runs/live_full_pipeline_rebuild_limited500k_logitfix/
#                  01_live_dataset_patched/train/
#   patched val:   runs/live_full_pipeline_rebuild_limited500k_logitfix/
#                  01_live_dataset_patched/val/
#
# Key analyzed: h_prime (baseline)
# Compare key:  layer_late (best performer in evidence sweep)
#
# Buckets:
#   A  base_miss + retrieval_miss      — unrecoverable cases
#   B  base_miss + retrieval_hit       — cases retrieval rescues
#   C  base_wrong + reinforces_wrong   — retrieval agrees with base error
#   D  base_wrong + supports_gold      — ideal resolver training cases
#   E  ambiguous / many_valid          — high-entropy, low support
#   F  retrieval_high_entropy          — neighbors scatter widely
#   G  retrieval_low_entropy_wrong     — retrieval confident but wrong
#   H  retrieval_low_entropy_gold      — retrieval confident and correct
#   I  region_right_token_wrong        — region succeeds, token fails
#
# Output: runs/retrieval_failure_analysis/
#   summary.json
#   bucket_A_base_miss_retrieval_miss.md
#   bucket_B_base_miss_retrieval_hit.md
#   bucket_C_reinforces_wrong.md
#   bucket_D_supports_gold.md
#   bucket_E_ambiguous.md
#   bucket_F_high_entropy.md
#   bucket_G_low_entropy_wrong.md
#   bucket_H_low_entropy_gold.md
#   bucket_I_region_right_token_wrong.md  (if region map available)
#
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
DATA_ROOT="runs/live_full_pipeline_rebuild_limited500k_logitfix"
TRAIN_DIR="$DATA_ROOT/01_live_dataset_patched/train"
VAL_DIR="$DATA_ROOT/01_live_dataset_patched/val"
SWEEP_ROOT="runs/retrieval_evidence_sweep_limited500k"
OUTPUT_ROOT="runs/retrieval_failure_analysis"
REGION_MAP="runs/region_maps_128/token_to_region.json"

echo "========================================================"
echo " Retrieval Failure Analysis"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking required inputs..."

if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: checkpoint not found: $SMALL_CKPT"
    exit 1
fi
echo "  [OK] $SMALL_CKPT"

for D in "$TRAIN_DIR" "$VAL_DIR"; do
    if [ ! -d "$D" ]; then
        echo "ERROR: patched shard dir not found: $D"
        echo "  Run slurm_run_live_full_pipeline_rebuild.sh first."
        exit 1
    fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then
        echo "ERROR: no shards in $D"
        exit 1
    fi
    echo "  [OK] $D  ($N shards)"
done

REGION_ARG=""
if [ -f "$REGION_MAP" ]; then
    REGION_ARG="--token_to_region $REGION_MAP"
    echo "  [OK] $REGION_MAP  (bucket I enabled)"
else
    echo "  [WARN] region map not found: $REGION_MAP  (bucket I will be skipped)"
fi

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_ROOT"
mkdir -p logs

echo "[launch] Starting retrieval failure analysis..."
echo ""

python scripts/analyze_retrieval_failure_examples.py \
    --small_ckpt        "$SMALL_CKPT"           \
    --train_dir         "$TRAIN_DIR"             \
    --val_dir           "$VAL_DIR"               \
    --sweep_root        "$SWEEP_ROOT"            \
    --output_root       "$OUTPUT_ROOT"           \
    --key               h_prime                  \
    --compare_key       layer_late               \
    $REGION_ARG                                  \
    \
    --num_examples_per_bucket  50                \
    --num_neighbors            32                \
    --top_k                    256               \
    --max_index_rows           500000            \
    --max_val_rows             0                 \
    \
    --retrieval_backend        auto              \
    --retrieval_chunk_size     131072            \
    --query_batch_size         512               \
    --support_tau              0.2               \
    \
    --max_postings_per_token            5000     \
    --max_lexical_candidates_per_query  20000

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: analysis exited with code $EXIT"
    echo "Check: $OUTPUT_ROOT/summary.json"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Retrieval Failure Analysis complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  $OUTPUT_ROOT/summary.json"
echo "  $OUTPUT_ROOT/bucket_*.md"
echo ""

python -c "
import json, os, glob
sp = '$OUTPUT_ROOT/summary.json'
if not os.path.exists(sp):
    print('summary.json not found')
    exit(0)
s = json.load(open(sp))
gs = s.get('global_stats', {})
print(f'Global stats (key={s[\"key\"]}):')
for k, v in gs.items():
    print(f'  {k:35s} = {v:.4f}')
print()
print('Bucket counts:')
for b, n in s.get('bucket_counts', {}).items():
    print(f'  {b:42s}: {n}')
mds = sorted(glob.glob('$OUTPUT_ROOT/bucket_*.md'))
print()
print('Markdown reports:')
for m in mds:
    print(f'  {m}')
" 2>/dev/null || true
