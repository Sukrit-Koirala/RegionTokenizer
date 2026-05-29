#!/bin/bash
#SBATCH --job-name=ctx_region_coord
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
# Phase 2: Contextual All-Region Coordinate Mixer
#
# Variants compared:
#   base_only
#   logit_only_mlp
#   static_region_embedding
#   contextual_all_region_real
#   contextual_all_region_shuffled
#   contextual_all_region_random
#
# Safety:
#   Gold used only for CE loss and metrics.
#   Step-0 identity required for every trainable variant.
#   No best checkpoint saved unless candidate NLL beats base.
#   Shuffled/random controls preserve region size distribution.
#   Evaluation slices always use real region map.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH="$PWD"
mkdir -p logs

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate learned_regions 2>/dev/null || conda activate base
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
    conda activate learned_regions 2>/dev/null || conda activate base
fi

TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/region_coordinate_mixer/phase2_contextual_all_region"
SCRIPT="scripts/train_contextual_all_region_coordinate_mixer.py"
SLURM_SCRIPT="scripts/slurm_train_contextual_all_region_coordinate_mixer.sh"

echo "========================================================"
echo " Phase 2: Contextual All-Region Coordinate Mixer"
echo " train_dir:  ${TRAIN_DIR}"
echo " val_dir:    ${VAL_DIR}"
echo " small_ckpt: ${SMALL_CKPT}"
echo " output_dir: ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

echo "[preflight] Checking required inputs..."

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: script not found: $SCRIPT"; exit 1
fi

if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: train_dir not found: $TRAIN_DIR"; exit 1
fi
N_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_TRAIN" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $TRAIN_DIR"; exit 1
fi
echo "  [OK] $TRAIN_DIR  ($N_TRAIN train shards)"

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: val_dir not found: $VAL_DIR"; exit 1
fi
N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_VAL" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $VAL_DIR"; exit 1
fi
echo "  [OK] $VAL_DIR  ($N_VAL val shards)"

if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: small_ckpt not found: $SMALL_CKPT"
    echo "       Phase 2 requires frozen token embeddings from this checkpoint."
    exit 1
fi
echo "  [OK] $SMALL_CKPT"

if [ ! -f "$TOKEN_TO_REGION" ]; then
    echo "ERROR: token_to_region not found: $TOKEN_TO_REGION"; exit 1
fi
echo "  [OK] $TOKEN_TO_REGION"

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map ${SUPER_MAP}"
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found — superregion disabled"
fi

echo ""
echo "[preflight] Checking syntax..."
python -m py_compile "$SCRIPT"
echo "  [OK] $SCRIPT compiles"
if [ -f "$SLURM_SCRIPT" ]; then
    bash -n "$SLURM_SCRIPT"
    echo "  [OK] $SLURM_SCRIPT syntax"
fi

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "$OUTPUT_DIR"

echo "[launch] Starting Phase 2 Contextual All-Region Coordinate Mixer..."
echo "  NOTE: Gold used only for loss/metrics — never as model input."
echo "  NOTE: All 128 regions used for contextual coordinate construction."
echo "  NOTE: Shuffled/random controls use same architecture with permuted maps."
echo "  NOTE: Slices always evaluated with real region labels."
echo ""

python "$SCRIPT" \
    --train_dir              "$TRAIN_DIR" \
    --val_dir                "$VAL_DIR" \
    --small_ckpt             "$SMALL_CKPT" \
    --token_to_region        "$TOKEN_TO_REGION" \
    $SUPER_ARG \
    --output_dir             "$OUTPUT_DIR" \
    --selected_M             64 \
    --model_dim              256 \
    --region_emb_dim         64 \
    --super_emb_dim          32 \
    --coord_proj_dim         64 \
    --hidden_dim             256 \
    --dropout                0.1 \
    --steps                  5000 \
    --eval_every             500 \
    --batch_size             128 \
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
echo " Phase 2 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/phase2_contextual_all_region_report.md"
echo "  ${OUTPUT_DIR}/phase2_comparison.csv"
echo "  ${OUTPUT_DIR}/slice_metrics.csv"
echo "  ${OUTPUT_DIR}/best_metrics.json"
echo "  ${OUTPUT_DIR}/final_metrics.json"
echo "  ${OUTPUT_DIR}/eval_log.csv"
echo ""

for f in \
    "${OUTPUT_DIR}/phase2_contextual_all_region_report.md" \
    "${OUTPUT_DIR}/phase2_comparison.csv" \
    "${OUTPUT_DIR}/best_metrics.json" \
    "${OUTPUT_DIR}/final_metrics.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"
        exit 1
    fi
done

python - <<'PYEOF'
import csv, json, os, re, sys

d = "runs/region_coordinate_mixer/phase2_contextual_all_region"
comp_path   = os.path.join(d, "phase2_comparison.csv")
report_path = os.path.join(d, "phase2_contextual_all_region_report.md")

try:
    rows = list(csv.DictReader(open(comp_path)))
    print("[result] Comparison table:")
    for r in rows:
        print(
            f"  {r.get('variant','?'):35s}"
            f" sel={r.get('selected_for_comparison','?'):14s}"
            f" nll={r.get('candidate_nll_given_gold_in_topM','?')}"
            f" gain={r.get('nll_gain_vs_base','?')}"
            f" all_acc={r.get('all_row_model_acc','?')}"
            f" ctg={r.get('changed_to_gold','?')}"
            f" caw={r.get('changed_away','?')}"
            f" BDR={r.get('benefit_damage_ratio','?')}"
            f" bc_dmg={r.get('base_correct_damage_rate','?')}"
        )
except Exception as e:
    print(f"[result] Could not parse comparison: {e}", file=sys.stderr)

try:
    if os.path.isfile(report_path):
        text = open(report_path, encoding="utf-8").read()
        m = re.search(r"recommendation:\s*(PROCEED_TO_PHASE_3|DO_NOT_PROCEED_TO_PHASE_3)", text)
        if m:
            print(f"\n[result] PHASE 2 VERDICT: {m.group(1)}")
        for line in text.splitlines():
            if "failure reasons" in line.lower():
                print(f"[result] {line.strip()}")
except Exception as e:
    print(f"[result] Could not parse report: {e}", file=sys.stderr)
PYEOF

echo "[done] Report: ${OUTPUT_DIR}/phase2_contextual_all_region_report.md"
