#!/bin/bash
#SBATCH --job-name=allreg_identity
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
# Phase 2B-pre: All-Region Token Identity Resolver
#
# Tests whether token identity can be learned as an all-region compatibility
# profile:
#   identity_ctx[c] = soft_attention(q_c, Z_contextual_region_states)
#
# Variants:
#   base_only                              (no model — baseline)
#   logit_only_mlp                         (logit features only)
#   token_context_identity                 (token emb + h_ctx, no region states)
#   home_region_identity_real              (only candidate's home-region state)
#   all_region_identity_real               (full attention over all region states — main model)
#   all_region_identity_shuffled           (shuffled Phase 2A checkpoint — control)
#   all_region_identity_random             (random Phase 2A checkpoint — control)
#   all_region_identity_real_coord_permuted (real Z, permuted order — coord identity control)
#
# Safety:
#   Phase 2A region meaning models are frozen by default.
#   Gold used only for loss/metrics after scores are produced.
#   No gold force-inclusion. No full-vocab scoring.
#   Fail loudly on NaN.
#
# Proceed only if:
#   all_region_identity_real beats token_context and home_region and shuffled/random
#   on candidate_nll_gain_vs_base and within_real_region_nll_gain_vs_base
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
PHASE2A_DIR="runs/region_coordinate_mixer/phase2A_region_meaning"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/region_coordinate_mixer/phase2B_pre_all_region_token_identity"
SCRIPT="scripts/train_all_region_token_identity_resolver.py"
SLURM_SCRIPT="scripts/slurm_train_all_region_token_identity_resolver.sh"

echo "========================================================"
echo " Phase 2B-pre: All-Region Token Identity Resolver"
echo " train_dir:   ${TRAIN_DIR}"
echo " val_dir:     ${VAL_DIR}"
echo " phase2a_dir: ${PHASE2A_DIR}"
echo " output_dir:  ${OUTPUT_DIR}"
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
    echo "ERROR: small_ckpt not found: $SMALL_CKPT"; exit 1
fi
echo "  [OK] $SMALL_CKPT"

if [ ! -d "$PHASE2A_DIR" ]; then
    echo "ERROR: phase2a_dir not found: $PHASE2A_DIR"; exit 1
fi
echo "  [OK] $PHASE2A_DIR"

for ckpt in \
    "${PHASE2A_DIR}/best_contextual_region_meaning_real.pt" \
    "${PHASE2A_DIR}/best_contextual_region_meaning_shuffled.pt" \
    "${PHASE2A_DIR}/best_contextual_region_meaning_random.pt"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Phase 2A checkpoint not found: $ckpt"; exit 1
    fi
    echo "  [OK] $ckpt"
done

for map_file in \
    "${PHASE2A_DIR}/shuffled_token_to_region.json" \
    "${PHASE2A_DIR}/random_token_to_region.json" \
    "${PHASE2A_DIR}/config.json"; do
    if [ ! -f "$map_file" ]; then
        echo "ERROR: file not found: $map_file"; exit 1
    fi
    echo "  [OK] $map_file"
done

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

echo "[launch] Starting Phase 2B-pre All-Region Token Identity Resolver..."
echo "  NOTE: Phase 2A region meaning models are frozen."
echo "  NOTE: Gold used only for loss/metrics after scores are produced."
echo "  NOTE: Only top-M candidates are scored. No full-vocab."
echo "  NOTE: Step-0 final_scores == base_logits (zero-init delta head)."
echo ""

# Auto-detect resume: only skip prior variants if their results are already saved
START_ARG=""
if python -c "
import json, sys
try:
    d = json.load(open('${OUTPUT_DIR}/best_metrics.json'))
    required = ['base_only','logit_only_mlp','token_context_identity','home_region_identity_real']
    missing  = [v for v in required if v not in d]
    if missing:
        print(f'[resume] Missing from best_metrics.json: {missing}', file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
except FileNotFoundError:
    print('[resume] best_metrics.json not found — starting from scratch', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
    START_ARG="--start_from_variant all_region_identity_real"
    echo "  [resume] Prior results found — resuming from all_region_identity_real"
else
    echo "  [fresh]  No complete prior results — running all variants from scratch"
fi

python "$SCRIPT" \
    --train_dir             "$TRAIN_DIR" \
    --val_dir               "$VAL_DIR" \
    --small_ckpt            "$SMALL_CKPT" \
    --phase2a_dir           "$PHASE2A_DIR" \
    --token_to_region       "$TOKEN_TO_REGION" \
    $SUPER_ARG \
    --output_dir            "$OUTPUT_DIR" \
    --selected_M            64 \
    --d_model               256 \
    --hidden_dim            256 \
    --num_heads             4 \
    --dropout               0.1 \
    --steps                 5000 \
    --eval_every            500 \
    --batch_size            128 \
    --lr                    5e-5 \
    --margin                0.5 \
    --lambda_ce             1.0 \
    --lambda_pair           1.0 \
    --lambda_same_region_pair        2.0 \
    --lambda_same_superregion_pair   1.0 \
    --lambda_within_region  1.0 \
    --lambda_preserve       1.0 \
    --lambda_delta          1e-3 \
    --lambda_gate           1e-2 \
    --use_gate \
    --gate_init_bias        -4.0 \
    --freeze_region_meaning \
    --seed                  42 \
    --amp \
    $START_ARG

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: training exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Phase 2B-pre complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/phase2B_pre_all_region_token_identity_report.md"
echo "  ${OUTPUT_DIR}/phase2B_pre_comparison.csv"
echo "  ${OUTPUT_DIR}/slice_metrics.csv"
echo "  ${OUTPUT_DIR}/coordinate_diagnostics.csv"
echo "  ${OUTPUT_DIR}/best_metrics.json"
echo ""

for f in \
    "${OUTPUT_DIR}/phase2B_pre_all_region_token_identity_report.md" \
    "${OUTPUT_DIR}/phase2B_pre_comparison.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"
        exit 1
    fi
done

python - <<'PYEOF'
import csv, json, os, re, sys

d    = "runs/region_coordinate_mixer/phase2B_pre_all_region_token_identity"
comp = os.path.join(d, "phase2B_pre_comparison.csv")
rpt  = os.path.join(d, "phase2B_pre_all_region_token_identity_report.md")

try:
    rows = list(csv.DictReader(open(comp)))
    print("[result] Phase 2B-pre comparison:")
    for r in rows:
        print(
            f"  {r.get('variant','?'):46s}"
            f" nll_g={r.get('candidate_nll_gain_vs_base','?'):8}"
            f" wr_g={r.get('within_real_region_nll_gain_vs_base','?'):8}"
            f" sr={r.get('same_region_pair_acc','?'):7}"
            f" c2g={r.get('changed_to_gold','?'):5}"
            f" caw={r.get('changed_away','?'):5}"
            f" bdr={r.get('benefit_damage_ratio','?'):6}"
            f" val={r.get('validation_score','?')}"
        )
except Exception as e:
    print(f"[result] Could not parse comparison: {e}", file=sys.stderr)

try:
    if os.path.isfile(rpt):
        text = open(rpt, encoding="utf-8").read()
        m = re.search(
            r"recommendation:\s*(PROCEED_TO_REGION_GENERATED_CANDIDATES"
            r"|PARTIAL_GO_IMPROVE_IDENTITY_RESOLVER"
            r"|DO_NOT_PROCEED)",
            text)
        if m:
            print(f"\n[result] PHASE 2B-pre VERDICT: {m.group(1)}")
except Exception as e:
    print(f"[result] Could not parse report: {e}", file=sys.stderr)

try:
    bm = json.load(open(os.path.join(d, "best_metrics.json")))
    real = bm.get("all_region_identity_real", {})
    if real:
        print(f"\n[result] all_region_identity_real best:")
        for k in ["candidate_nll_gain_vs_base",
                  "within_real_region_nll_gain_vs_base",
                  "same_region_pair_acc",
                  "changed_to_gold", "changed_away",
                  "benefit_damage_ratio",
                  "base_correct_damage_rate",
                  "validation_score"]:
            v = real.get(k, float("nan"))
            print(f"  {k:42s} = {v}")
except Exception as e:
    print(f"[result] Could not parse best_metrics: {e}", file=sys.stderr)
PYEOF

echo "[done] Report: ${OUTPUT_DIR}/phase2B_pre_all_region_token_identity_report.md"
