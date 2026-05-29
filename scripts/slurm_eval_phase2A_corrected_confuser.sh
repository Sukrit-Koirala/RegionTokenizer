#!/bin/bash
#SBATCH --job-name=phase2a_corr_eval
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
# Phase 2A.1: Corrected Region-Meaning Confuser Diagnostic
#
# Fixes the split-sensitive confuser metric from Phase 2A by forcing
# both gold and base through the SAME anchor region state.
#
# Primary question:
#   Does the learned region meaning z_gold_region distinguish gold from
#   confusers when both are scored through the SAME anchor?
#
# This is eval-only. No training. No gradient updates.
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

PHASE2A_DIR="runs/region_coordinate_mixer/phase2A_region_meaning"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/region_coordinate_mixer/phase2A_region_meaning_corrected_eval"
SCRIPT="scripts/eval_phase2A_corrected_confuser.py"
SLURM_SCRIPT="scripts/slurm_eval_phase2A_corrected_confuser.sh"

echo "========================================================"
echo " Phase 2A.1: Corrected Region-Meaning Confuser Diagnostic"
echo " phase2a_dir: ${PHASE2A_DIR}"
echo " val_dir:     ${VAL_DIR}"
echo " output_dir:  ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

echo "[preflight] Checking required inputs..."

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: script not found: $SCRIPT"; exit 1
fi

if [ ! -d "$PHASE2A_DIR" ]; then
    echo "ERROR: phase2a_dir not found: $PHASE2A_DIR"; exit 1
fi

for ckpt in \
    "${PHASE2A_DIR}/best_static_region_meaning.pt" \
    "${PHASE2A_DIR}/best_contextual_region_meaning_real.pt" \
    "${PHASE2A_DIR}/best_contextual_region_meaning_shuffled.pt" \
    "${PHASE2A_DIR}/best_contextual_region_meaning_random.pt"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: checkpoint not found: $ckpt"; exit 1
    fi
    echo "  [OK] $ckpt"
done

for map_file in \
    "${PHASE2A_DIR}/shuffled_token_to_region.json" \
    "${PHASE2A_DIR}/random_token_to_region.json"; do
    if [ ! -f "$map_file" ]; then
        echo "ERROR: map not found: $map_file"; exit 1
    fi
    echo "  [OK] $map_file"
done

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: val_dir not found: $VAL_DIR"; exit 1
fi
N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_VAL" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $VAL_DIR"; exit 1
fi
echo "  [OK] $VAL_DIR  ($N_VAL val shards)"

if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: small_ckpt not found: $SMALL_CKPT"; exit 1
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
    echo "  [WARN] super_map not found — superregion features disabled"
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

echo "[launch] Starting Phase 2A.1 Corrected Confuser Diagnostic..."
echo "  NOTE: Eval only — no training, no gradient updates."
echo "  NOTE: Pair scores use forced same-anchor: gold and base both"
echo "        scored through the SAME region state (gold's variant anchor)."
echo "  NOTE: All confuser slices defined with REAL tok_arr."
echo "  NOTE: Gold token used only for metrics after Z is computed."
echo ""

python "$SCRIPT" \
    --phase2a_dir        "$PHASE2A_DIR" \
    --val_dir            "$VAL_DIR" \
    --small_ckpt         "$SMALL_CKPT" \
    --token_to_region    "$TOKEN_TO_REGION" \
    $SUPER_ARG \
    --output_dir         "$OUTPUT_DIR" \
    --selected_M         64 \
    --batch_size         512 \
    --seed               42 \
    --amp

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: eval exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Phase 2A.1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/phase2A_corrected_confuser_report.md"
echo "  ${OUTPUT_DIR}/corrected_confuser_comparison.csv"
echo "  ${OUTPUT_DIR}/corrected_slice_metrics.csv"
echo "  ${OUTPUT_DIR}/examples_real_beats_shuffled_same_anchor.md"
echo "  ${OUTPUT_DIR}/examples_shuffled_old_metric_cheat.md"
echo "  ${OUTPUT_DIR}/examples_real_fails_same_region.md"
echo ""

for f in \
    "${OUTPUT_DIR}/phase2A_corrected_confuser_report.md" \
    "${OUTPUT_DIR}/corrected_confuser_comparison.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"
        exit 1
    fi
done

python - <<'PYEOF'
import csv, json, os, re, sys

d = "runs/region_coordinate_mixer/phase2A_region_meaning_corrected_eval"
comp = os.path.join(d, "corrected_confuser_comparison.csv")
rpt  = os.path.join(d, "phase2A_corrected_confuser_report.md")

try:
    rows = list(csv.DictReader(open(comp)))
    print("[result] Corrected comparison table:")
    for r in rows:
        print(
            f"  {r.get('variant','?'):44s}"
            f" old_sr={r.get('old_split_pair_acc_same_reg','?'):7}"
            f" forced_sr={r.get('forced_anchor_pair_acc_same_reg','?'):7}"
            f" wr_acc={r.get('within_real_region_anchor_acc','?'):7}"
            f" wr_gain={r.get('within_real_acc_gain_vs_base','?')}"
        )
except Exception as e:
    print(f"[result] Could not parse comparison: {e}", file=sys.stderr)

try:
    if os.path.isfile(rpt):
        text = open(rpt, encoding="utf-8").read()
        m = re.search(r"recommendation:\s*(PROCEED_TO_PHASE_2B|PARTIAL_GO_BUT_IMPROVE_REGION_MEANING|DO_NOT_PROCEED_TO_PHASE_2B)", text)
        if m:
            print(f"\n[result] PHASE 2A.1 VERDICT: {m.group(1)}")
        for line in text.splitlines():
            if any(kw in line for kw in ["Failure reason", "failure reason", "CASE", "verdict"]):
                print(f"[result] {line.strip()}")
except Exception as e:
    print(f"[result] Could not parse report: {e}", file=sys.stderr)
PYEOF

echo "[done] Report: ${OUTPUT_DIR}/phase2A_corrected_confuser_report.md"
