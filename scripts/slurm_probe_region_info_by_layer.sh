#!/bin/bash
#SBATCH --job-name=rir_probe_layers
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
# Layer-wise region information probes for the Residual Interface Refiner experiment.
#
# Trains linear and MLP probes per backbone layer to predict:
#   fine_region      — gold region index (128-class classification)
#   superregion      — gold superregion index (24-class classification)
#   is_boundary      — position is a boundary position (binary)
#   router_top8_miss — gold region outside router top-8 (binary)
#
# Reads config.json from val_multilayer/ to discover layer_ids and d_model.
# Outputs:
#   probes/layer_probe_results.csv   — all metrics per layer/task/probe_type
#   probes/layer_probe_report.md     — diagnostic summary with Q1-Q4 answered
#
# Questions answered:
#   Q1. Which layer best predicts gold fine region?
#   Q2. Does region information appear before final h_prime?
#   Q3. Does final h_prime lose/blur region information?
#   Q4. Are boundary/router-miss positions better represented in mid layers?
#
# Run order:
#   slurm_build_multilayer_residual_features.sh   ← must have completed
#   → THIS SCRIPT
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

DATA_ROOT=runs/path_refiner_clean/data
FEATURES_ROOT=runs/path_refiner_residual_interface/features
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
PROBE_OUTPUT_DIR=runs/path_refiner_residual_interface/probes

VAL_CAND_DIR=$DATA_ROOT/val_hgrid_K24
TRAIN_CAND_DIR=$DATA_ROOT/train_hgrid_K24
VAL_FEAT_DIR=$FEATURES_ROOT/val_multilayer
TRAIN_FEAT_DIR=$FEATURES_ROOT/train_multilayer

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Layer Region Information Probes  $(date) ==="
echo "    TRAIN_CAND_DIR : $TRAIN_CAND_DIR"
echo "    VAL_CAND_DIR   : $VAL_CAND_DIR"
echo "    TRAIN_FEAT_DIR : $TRAIN_FEAT_DIR"
echo "    VAL_FEAT_DIR   : $VAL_FEAT_DIR"
echo "    SUPER_MAP      : $SUPER_MAP"
echo "    OUTPUT_DIR     : $PROBE_OUTPUT_DIR"
echo ""

for f in "$SUPER_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

for d in "$VAL_CAND_DIR" "$TRAIN_CAND_DIR" "$VAL_FEAT_DIR" "$TRAIN_FEAT_DIR"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: required directory missing: $d" >&2; exit 1
    fi
done

VAL_CFG=$VAL_FEAT_DIR/config.json
if [[ ! -f "$VAL_CFG" ]]; then
    echo "ERROR: $VAL_CFG not found — run slurm_build_multilayer_residual_features.sh first." >&2; exit 1
fi

N_VAL_FEAT=$(find "$VAL_FEAT_DIR"   -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_TR_FEAT=$(find "$TRAIN_FEAT_DIR"  -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    val feature shards   : $N_VAL_FEAT"
echo "    train feature shards : $N_TR_FEAT"
if [[ "$N_VAL_FEAT" -eq 0 || "$N_TR_FEAT" -eq 0 ]]; then
    echo "ERROR: missing feature shards. Run slurm_build_multilayer_residual_features.sh first." >&2; exit 1
fi

LAYER_IDS=$(python -c "import json; c=json.load(open('$VAL_CFG')); print(c['layer_ids'])" 2>/dev/null || echo "?")
N_LAYERS=$(python -c  "import json; c=json.load(open('$VAL_CFG')); print(c['n_layers_saved'])" 2>/dev/null || echo "?")
echo "    layer_ids            : $LAYER_IDS"
echo "    n_layers_saved       : $N_LAYERS"
echo ""

mkdir -p "$PROBE_OUTPUT_DIR"

# ── Run probes ────────────────────────────────────────────────────────────────

echo "Running layer probes ..."
echo "  Tasks: fine_region (128-class), superregion (24-class),"
echo "         is_boundary (binary), router_top8_miss (binary)"
echo "  Probe types: linear, mlp"
echo "  steps=2000  batch_size=512  lr=1e-3"
echo ""

python scripts/probe_region_info_by_layer.py         \
    --train_candidate_dir $TRAIN_CAND_DIR             \
    --val_candidate_dir   $VAL_CAND_DIR               \
    --train_features_dir  $TRAIN_FEAT_DIR             \
    --val_features_dir    $VAL_FEAT_DIR               \
    --super_map           $SUPER_MAP                  \
    --output_dir          $PROBE_OUTPUT_DIR           \
    --steps               2000                        \
    --batch_size          512                         \
    --lr                  1e-3                        \
    --device              cuda

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Layer probes complete  $(date) ==="
echo ""

CSV=$PROBE_OUTPUT_DIR/layer_probe_results.csv
MD=$PROBE_OUTPUT_DIR/layer_probe_report.md

if [[ -f "$CSV" ]]; then
    echo "Results CSV: $CSV"
    echo ""
    echo "Linear probe fine_region acc@1 by layer:"
    python -c "
import csv
rows = list(csv.DictReader(open('$CSV')))
for r in rows:
    if r.get('task') == 'fine_region' and r.get('probe') == 'linear':
        print(f\"  {r['layer']:20s}  acc@1={r['val_acc1']}  acc@4={r['val_acc4']}\")
"
fi

if [[ -f "$MD" ]]; then
    echo ""
    echo "Report: $MD"
fi

echo ""
echo "Next step: slurm_train_residual_interface_refiner.sh"
echo "  Probe results inform which variant is most promising."
echo "  If mid-layer acc > h_final acc: multilayer_crossattn is well-motivated."
echo "  If region info is already in h_prime: region_state_crossattn may suffice."
