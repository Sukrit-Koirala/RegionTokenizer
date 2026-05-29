#!/bin/bash
#SBATCH --job-name=region_meaning
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
# Phase 2A: Contextual Region Meaning Learner
#
# Directly learns contextual region states Z(h) = [z_0(h), ..., z_{R-1}(h)].
# Region states are trained with region-grounded objectives.
#
# Variants:
#   router_baseline                (no training, router recall only)
#   static_region_meaning          (no h_ctx — tests whether context is necessary)
#   contextual_region_meaning_real
#   contextual_region_meaning_shuffled
#   contextual_region_meaning_random
#
# Safety:
#   Gold used only for targets/loss/metrics — never as model input.
#   Candidate summaries use only base top-M candidates.
#   Confuser metrics always evaluated with real tok_arr.
#   Shuffled/random region CE measured under their own map.
#   Fail loudly on NaN.
#
# Proceed to Phase 2B only if:
#   contextual > static on gold_region_recall@8
#   contextual > shuffled/random on same_region_confuser_pair_acc
#   within_true_region_acc is nontrivial (> 0.1)
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
OUTPUT_DIR="runs/region_coordinate_mixer/phase2A_region_meaning"
SCRIPT="scripts/train_contextual_region_meaning_learner.py"
SLURM_SCRIPT="scripts/slurm_train_contextual_region_meaning_learner.sh"

echo "========================================================"
echo " Phase 2A: Contextual Region Meaning Learner"
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
    echo "       Phase 2A requires frozen token embeddings from this checkpoint."
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

echo "[launch] Starting Phase 2A Contextual Region Meaning Learner..."
echo "  NOTE: Region states are trained with region-grounded objectives."
echo "  NOTE: Gold used only for targets/loss/metrics — never as model input."
echo "  NOTE: Confuser metrics always use real tok_arr for definition."
echo "  NOTE: Shuffled/random region CE measured under their own map."
echo ""

python "$SCRIPT" \
    --train_dir              "$TRAIN_DIR" \
    --val_dir                "$VAL_DIR" \
    --small_ckpt             "$SMALL_CKPT" \
    --token_to_region        "$TOKEN_TO_REGION" \
    $SUPER_ARG \
    --output_dir             "$OUTPUT_DIR" \
    --selected_M             64 \
    --d_model                256 \
    --region_emb_dim         64 \
    --super_emb_dim          32 \
    --num_region_layers      2 \
    --num_heads              4 \
    --dropout                0.1 \
    --steps                  5000 \
    --eval_every             500 \
    --batch_size             128 \
    --lr                     1e-4 \
    --lambda_region_ce       1.0 \
    --lambda_candidate_ce    0.5 \
    --lambda_within_region   1.0 \
    --lambda_margin          0.5 \
    --lambda_region_kl       0.0 \
    --lambda_l2              1e-5 \
    --margin                 0.5 \
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
echo " Phase 2A complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/phase2A_region_meaning_report.md"
echo "  ${OUTPUT_DIR}/phase2A_comparison.csv"
echo "  ${OUTPUT_DIR}/best_metrics.json"
echo "  ${OUTPUT_DIR}/final_metrics.json"
echo "  ${OUTPUT_DIR}/eval_log.csv"
echo "  ${OUTPUT_DIR}/train_log.csv"
echo ""

for f in \
    "${OUTPUT_DIR}/phase2A_region_meaning_report.md" \
    "${OUTPUT_DIR}/phase2A_comparison.csv" \
    "${OUTPUT_DIR}/best_metrics.json" \
    "${OUTPUT_DIR}/final_metrics.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"
        exit 1
    fi
done

python - <<'PYEOF'
import csv, json, os, re, sys

d = "runs/region_coordinate_mixer/phase2A_region_meaning"
comp_path   = os.path.join(d, "phase2A_comparison.csv")
report_path = os.path.join(d, "phase2A_region_meaning_report.md")

try:
    rows = list(csv.DictReader(open(comp_path)))
    print("[result] Comparison table:")
    for r in rows:
        print(
            f"  {r.get('variant','?'):40s}"
            f" sel={r.get('selected_for_comparison','?'):14s}"
            f" recall@8={r.get('gold_region_recall@8','?')}"
            f" conf={r.get('same_region_confuser_pair_acc','?')}"
            f" wr_acc={r.get('within_true_region_acc','?')}"
            f" val={r.get('validation_score','?')}"
        )
except Exception as e:
    print(f"[result] Could not parse comparison: {e}", file=sys.stderr)

try:
    if os.path.isfile(report_path):
        text = open(report_path, encoding="utf-8").read()
        m = re.search(r"recommendation:\s*(PROCEED_TO_PHASE_2B|DO_NOT_PROCEED_TO_PHASE_2B)", text)
        if m:
            print(f"\n[result] PHASE 2A VERDICT: {m.group(1)}")
        for line in text.splitlines():
            if "failure reasons" in line.lower():
                print(f"[result] {line.strip()}")
except Exception as e:
    print(f"[result] Could not parse report: {e}", file=sys.stderr)

try:
    bm = json.load(open(os.path.join(d, "best_metrics.json")))
    ctx = bm.get("contextual_region_meaning_real", {})
    if ctx:
        print(f"\n[result] contextual_real best:"
              f" recall@8={ctx.get('gold_region_recall@8', float('nan')):.4f}"
              f" conf_pair={ctx.get('same_region_confuser_pair_acc', float('nan')):.4f}"
              f" wr_acc={ctx.get('within_true_region_acc', float('nan')):.4f}"
              f" val_score={ctx.get('validation_score', float('nan')):.4f}")
except Exception as e:
    print(f"[result] Could not parse best_metrics: {e}", file=sys.stderr)
PYEOF

echo "[done] Report: ${OUTPUT_DIR}/phase2A_region_meaning_report.md"
