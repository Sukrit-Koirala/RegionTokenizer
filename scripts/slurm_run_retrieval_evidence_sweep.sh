#!/bin/bash
#SBATCH --job-name=ret_evidence_sweep
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
# Retrieval Evidence Sweep — Diagnostic over retrieval key quality
#
# Tests 9 retrieval representations to find which provides the most
# useful token-level evidence beyond the base model's top-K candidates.
#
# Inputs (from logitfix run):
#   train shards: runs/live_full_pipeline_rebuild_limited500k_logitfix/
#                 01_live_dataset_patched/train/shard_*.pt
#   val shards:   runs/live_full_pipeline_rebuild_limited500k_logitfix/
#                 01_live_dataset_patched/val/shard_*.pt
#
# Keys:
#   h_prime              baseline (saved in patched shards)
#   h_raw                pre-logit hidden (saved in patched shards)
#   layer_early          hidden at ~25% depth (live inference)
#   layer_mid            hidden at ~50% depth (live inference)
#   layer_late           hidden at ~75% depth (live inference)
#   hybrid_early_hprime  concat(norm(early), norm(h_prime))
#   lexical_last16       Jaccard over last 16 input tokens
#   lexical_last32       Jaccard over last 32 input tokens
#   hybrid_mid_lexical   layer_mid dense + lexical rerank
#
# Main metric: ret_added_base_miss
#   (among rows where gold not in base top-256, how often retrieval adds gold)
#   Current h_prime baseline: ~0.0017 globally
#
# Output: runs/retrieval_evidence_sweep_limited500k/
#   config.json         sweep config
#   sweep_summary.csv   all keys ranked
#   report.md           human-readable ranking + interpretation
#   report.json         full metrics JSON
#   keys/<key>/metrics.json    per-key metrics
#   keys/<key>/examples.md     20 diagnostic examples
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
OUTPUT_ROOT="runs/retrieval_evidence_sweep_limited500k"

echo "========================================================"
echo " Retrieval Evidence Sweep"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking required inputs..."

if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: checkpoint not found: $SMALL_CKPT"
    exit 1
fi
echo "  [OK] $SMALL_CKPT"

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: data root not found: $DATA_ROOT"
    echo "  Run the logitfix pipeline first:"
    echo "  sbatch scripts/slurm_run_live_full_pipeline_rebuild.sh"
    exit 1
fi
echo "  [OK] $DATA_ROOT"

for D in "$TRAIN_DIR" "$VAL_DIR"; do
    if [ ! -d "$D" ]; then
        echo "ERROR: patched shard dir not found: $D"
        echo "  Stage01B (audit_logit_paths) must have passed first."
        exit 1
    fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then
        echo "ERROR: no shards in $D"
        exit 1
    fi
    echo "  [OK] $D  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present."
echo ""
mkdir -p "$OUTPUT_ROOT"
mkdir -p logs

echo "[launch] Starting retrieval evidence sweep..."
echo ""

python scripts/run_retrieval_evidence_sweep.py \
    --small_ckpt           "$SMALL_CKPT"        \
    --data_root            "$DATA_ROOT"          \
    --train_dir            "$TRAIN_DIR"          \
    --val_dir              "$VAL_DIR"            \
    --output_root          "$OUTPUT_ROOT"        \
    --token_to_region      runs/region_maps_128/token_to_region.json \
    --super_map            runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json \
    \
    --keys                 h_prime,h_raw,layer_early,layer_mid,layer_late,hybrid_early_hprime,lexical_last16,lexical_last32,hybrid_mid_lexical \
    \
    --num_neighbors        32                    \
    --top_k                256                   \
    --max_index_rows       500000                \
    --max_val_rows         30896                 \
    \
    --retrieval_backend    auto                  \
    --retrieval_chunk_size 131072                \
    --query_batch_size     512                   \
    \
    --support_tau_dense    0.2                   \
    --support_tau_lexical  1.0                   \
    --max_postings_per_token         5000        \
    --max_lexical_candidates_per_query 20000     \
    --lexical_weight       0.5                   \
    \
    --num_examples         20

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: sweep exited with code $EXIT"
    echo "Check: $OUTPUT_ROOT/report.json"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Retrieval Evidence Sweep complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  $OUTPUT_ROOT/report.md"
echo "  $OUTPUT_ROOT/sweep_summary.csv"
echo "  $OUTPUT_ROOT/keys/*/metrics.json"
echo ""
echo "Baseline: h_prime ret_added_global = 0.0017"
echo ""

python -c "
import json, os
rp = '$OUTPUT_ROOT/report.json'
if not os.path.exists(rp):
    print('report.json not found')
    exit(0)
r = json.load(open(rp))
ranked = r.get('ranked_keys', [])
print('Ranking (by ret_added_base_miss):')
for k in ranked[:9]:
    print(f'  {k.get(\"rank\",\"?\"):>2}. {k.get(\"key\",\"?\"):25s}'
          f'  miss={k.get(\"ret_added_base_miss\",0):.4f}'
          f'  global={k.get(\"retrieval_added_gold\",0):.4f}'
          f'  margin_wrong={k.get(\"pct_margin_gt0_base_wrong\",0):.4f}')
skip = r.get('skipped_keys', [])
if skip:
    print(f'Skipped: {[s[\"key\"] for s in skip]}')
" 2>/dev/null || true
