#!/bin/bash
#SBATCH --job-name=rir_build_features
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Build multi-layer residual features for the Residual Interface Refiner experiment.
#
# Processes WikiText-103 (val then train) with the frozen backbone, capturing
# hidden states after every-other transformer block and after ln_f (auto_even).
# Output is aligned to existing candidate shards via gold_token comparison
# (aborts on any alignment mismatch).
#
# Output layout:
#   features/val_multilayer/shard_XXXXX.pt  — (N, n_layers, d_model) float16
#   features/train_multilayer/shard_XXXXX.pt
#   features/{val,train}_multilayer/config.json  — layer_ids, d_model, n_layers_saved
#
# Shard fields:
#   h_layers    float16  (N, n_layers_saved, d_model)
#   layer_ids   int64    (n_layers_saved,)  — block indices, -1 = h_final
#   gold_token  int32    (N,)               — alignment verification field
#
# Run order:
#   slurm_clean_build_datasets.sh
#   → THIS SCRIPT
#   → slurm_probe_region_info_by_layer.sh
#   → slurm_train_residual_interface_refiner.sh

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
REGION_MAP=runs/region_maps_128/token_to_region.json
DATA_ROOT=runs/path_refiner_clean/data
FEATURES_ROOT=runs/path_refiner_residual_interface/features

VAL_CAND_DIR=$DATA_ROOT/val_hgrid_K24
TRAIN_CAND_DIR=$DATA_ROOT/train_hgrid_K24

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Build Multilayer Residual Features  $(date) ==="
echo "    SMALL_CKPT      : $SMALL_CKPT"
echo "    REGION_MAP      : $REGION_MAP"
echo "    VAL_CAND_DIR    : $VAL_CAND_DIR"
echo "    TRAIN_CAND_DIR  : $TRAIN_CAND_DIR"
echo "    FEATURES_ROOT   : $FEATURES_ROOT"
echo ""

for f in "$SMALL_CKPT" "$REGION_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

N_VAL=$(find "$VAL_CAND_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_TRAIN=$(find "$TRAIN_CAND_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    val candidate shards   : $N_VAL"
echo "    train candidate shards : $N_TRAIN"
if [[ "$N_VAL" -eq 0 || "$N_TRAIN" -eq 0 ]]; then
    echo "ERROR: missing candidate shards. Run slurm_clean_build_datasets.sh first." >&2; exit 1
fi

mkdir -p "$FEATURES_ROOT"

# ── Shared args ───────────────────────────────────────────────────────────────
#
# --layers auto_even : every-other block + h_final  (seq_len=128 → 7 states for 6-block model)
# --seq_len 128      : must match the candidate shard build (non-negotiable)
# --batch_size 32    : GPU memory safe for d_model=384

SHARED="
    --small_ckpt  $SMALL_CKPT
    --region_map  $REGION_MAP
    --layers      auto_even
    --seq_len     128
    --batch_size  32
    --device      cuda
"

# ── Step 1: val split ─────────────────────────────────────────────────────────

echo ""
echo "=== Step 1/2: val split  $(date) ==="

python scripts/build_multilayer_residual_features.py \
    $SHARED                                           \
    --candidate_dir $VAL_CAND_DIR                     \
    --output_dir    $FEATURES_ROOT/val_multilayer     \
    --split         val

echo ""
echo "=== Step 1/2: val split DONE  $(date) ==="

# ── Step 2: train split ───────────────────────────────────────────────────────

echo ""
echo "=== Step 2/2: train split  $(date) ==="

python scripts/build_multilayer_residual_features.py \
    $SHARED                                           \
    --candidate_dir $TRAIN_CAND_DIR                   \
    --output_dir    $FEATURES_ROOT/train_multilayer   \
    --split         train

echo ""
echo "=== Step 2/2: train split DONE  $(date) ==="

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Feature build complete  $(date) ==="
echo ""

for d in "$FEATURES_ROOT/val_multilayer" "$FEATURES_ROOT/train_multilayer"; do
    cfg="$d/config.json"
    n_shards=$(find "$d" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
    if [[ -f "$cfg" ]]; then
        n_layers=$(python -c "import json; c=json.load(open('$cfg')); print(c['n_layers_saved'])" 2>/dev/null || echo "?")
        layer_ids=$(python -c "import json; c=json.load(open('$cfg')); print(c['layer_ids'])" 2>/dev/null || echo "?")
        d_model=$(python -c "import json; c=json.load(open('$cfg')); print(c['d_model'])" 2>/dev/null || echo "?")
        total=$(python -c "import json; c=json.load(open('$cfg')); print(c['total_rows'])" 2>/dev/null || echo "?")
        echo "  $(basename $d):  shards=$n_shards  n_layers=$n_layers  d_model=$d_model  rows=$total"
        echo "    layer_ids=$layer_ids"
    else
        echo "  $(basename $d):  shards=$n_shards  (no config.json)"
    fi
done

echo ""
echo "Next step: slurm_probe_region_info_by_layer.sh"
