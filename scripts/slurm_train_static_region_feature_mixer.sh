#!/bin/bash
#SBATCH --job-name=static_region_mix
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Phase 1: Static Region Feature Mixer — FIXED
#
# Tests whether static region/superregion/router-region features improve
# candidate selection beyond:
#   1. base logits
#   2. logit-only MLP recalibration
#   3. shuffled/random region controls
#
# Variants evaluated:
#   base_only
#   logit_only_mlp
#   hard_region_features
#   static_region_embedding
#   shuffled_region_control
#   random_region_control
#
# Safety:
#   Gold used only for CE loss and metrics.
#   Gold is never force-included.
#   Step-0 identity must pass for every trainable variant.
#   No best checkpoint saved unless candidate NLL beats base.
#   Real-region verdict must beat logit-only and shuffled/random controls.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH="$PWD"
mkdir -p logs

# Robust environment setup: prefer project venv, then learned_regions, then base.
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate learned_regions || conda activate base
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
    conda activate learned_regions || conda activate base
fi

TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/region_coordinate_mixer/phase1_static_region_features"
SCRIPT="scripts/train_static_region_feature_mixer.py"
SLURM_SCRIPT="scripts/slurm_train_static_region_feature_mixer.sh"

echo "========================================================"
echo " Phase 1: Static Region Feature Mixer — FIXED"
echo " train_dir:  ${TRAIN_DIR}"
echo " val_dir:    ${VAL_DIR}"
echo " output_dir: ${OUTPUT_DIR}"
echo " script:     ${SCRIPT}"
echo " $(date)"
echo "========================================================"
echo ""

echo "[preflight] Checking required inputs..."

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: script not found: $SCRIPT"
    exit 1
fi

if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: train_dir not found: $TRAIN_DIR"
    exit 1
fi
N_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_TRAIN" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $TRAIN_DIR"
    exit 1
fi
echo "  [OK] $TRAIN_DIR  ($N_TRAIN train shards)"

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: val_dir not found: $VAL_DIR"
    exit 1
fi
N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_VAL" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $VAL_DIR"
    exit 1
fi
echo "  [OK] $VAL_DIR  ($N_VAL val shards)"

if [ ! -f "$TOKEN_TO_REGION" ]; then
    echo "ERROR: token_to_region not found: $TOKEN_TO_REGION"
    echo "The fixed Phase 1 script requires --token_to_region."
    exit 1
fi
echo "  [OK] $TOKEN_TO_REGION"

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map ${SUPER_MAP}"
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found — superregion features disabled"
fi

echo ""
echo "[preflight] Checking syntax..."
python -m py_compile "$SCRIPT"
echo "  [OK] $SCRIPT compiles"
if [ -f "$SLURM_SCRIPT" ]; then
    bash -n "$SLURM_SCRIPT"
    echo "  [OK] $SLURM_SCRIPT syntax"
else
    echo "  [WARN] $SLURM_SCRIPT not found for bash -n self-check"
fi

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "$OUTPUT_DIR"

echo "[launch] Starting fixed Phase 1 Static Region Feature Mixer..."
echo "  NOTE: includes logit_only_mlp control."
echo "  NOTE: shuffled/random slices are evaluated with REAL region labels."
echo "  NOTE: gold used only for loss/metrics, never as input."
echo "  NOTE: no best checkpoint saved unless candidate NLL beats base."
echo ""

python "$SCRIPT" \
    --train_dir              "$TRAIN_DIR" \
    --val_dir                "$VAL_DIR" \
    --token_to_region        "$TOKEN_TO_REGION" \
    $SUPER_ARG \
    --output_dir             "$OUTPUT_DIR" \
    --selected_M             64 \
    --hidden_dim             128 \
    --dropout                0.1 \
    --steps                  5000 \
    --eval_every             500 \
    --batch_size             256 \
    --lr                     1e-4 \
    --lambda_delta           1e-4 \
    --lambda_gate            1e-3 \
    --lambda_preserve        0.5 \
    --max_base_correct_damage_rate 0.05 \
    --use_gate \
    --gate_init_bias         0.0 \
    --seed                   42 \
    --amp

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: training exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Phase 1 Static Region Feature Mixer complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/phase1_static_region_report.md"
echo "  ${OUTPUT_DIR}/phase1_comparison.csv"
echo "  ${OUTPUT_DIR}/slice_metrics.csv"
echo "  ${OUTPUT_DIR}/best_metrics.json"
echo "  ${OUTPUT_DIR}/final_metrics.json"
echo "  ${OUTPUT_DIR}/train_log.csv"
echo "  ${OUTPUT_DIR}/eval_log.csv"
echo ""

for f in \
    "${OUTPUT_DIR}/phase1_static_region_report.md" \
    "${OUTPUT_DIR}/phase1_comparison.csv" \
    "${OUTPUT_DIR}/best_metrics.json" \
    "${OUTPUT_DIR}/final_metrics.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"
        exit 1
    fi
done

# Surface the key result using the updated fixed-script metric names.
python - <<'PYEOF'
import csv, json, os, re, sys

d = "runs/region_coordinate_mixer/phase1_static_region_features"
comp_path = os.path.join(d, "phase1_comparison.csv")
report_path = os.path.join(d, "phase1_static_region_report.md")

try:
    rows = list(csv.DictReader(open(comp_path)))
    print("[result] Comparison table:")
    for r in rows:
        print(
            f"  {r.get('variant','?'):30s} "
            f"selected={r.get('selected_for_comparison','?'):14s} "
            f"cand_nll={r.get('candidate_nll_given_gold_in_topM','?')} "
            f"nll_gain={r.get('nll_gain_vs_base','?')} "
            f"all_acc={r.get('all_row_model_acc','?')} "
            f"all_gain={r.get('all_row_acc_gain','?')} "
            f"ctg={r.get('changed_to_gold','?')} "
            f"caw={r.get('changed_away','?')} "
            f"BDR={r.get('benefit_damage_ratio','?')} "
            f"bc_dmg={r.get('base_correct_damage_rate','?')}"
        )

    if os.path.isfile(report_path):
        text = open(report_path, encoding="utf-8").read()
        m = re.search(r"recommendation:\s*(PROCEED_TO_PHASE_2|DO_NOT_PROCEED_TO_PHASE_2)", text)
        if m:
            print(f"[result] VERDICT: {m.group(1)}")
        else:
            m2 = re.search(r"\*\*(PROCEED_TO_PHASE_2|DO_NOT_PROCEED_TO_PHASE_2)\*\*", text)
            print(f"[result] VERDICT: {m2.group(1) if m2 else 'UNKNOWN'}")

except Exception as e:
    print(f"[result] Could not parse outputs: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

echo "[done] Report: ${OUTPUT_DIR}/phase1_static_region_report.md"
