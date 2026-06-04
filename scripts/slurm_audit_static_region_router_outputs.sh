#!/bin/bash
#SBATCH --job-name=router_audit
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
# Phase 1A.4: Router Output / Error Anatomy Audit
#
# Evaluation only. No training. No weight updates.
# Model input: input_ids only.
# Gold used only after prediction for metrics and categorization.
#
# Answers:
#   Q1.  Gold-region rank distribution
#   Q2.  Top8 miss breakdown (rank 9-16 / 17-32 / 33+)
#   Q3.  Gold superregion in top8/top16 when region missed
#   Q4.  Are top8 predictions redundant or diverse?
#   Q5.  Is router uncertain when it misses?
#   Q6.  Can margin/entropy fallback recover misses?
#   Q7.  Candidate fraction needed for 90/95/98/99% recall
#   Q8.  Failure taxonomy
#   Q9.  Best next policy for Phase 1B
#   Q10. Proceed or improve?
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH="$PWD"
mkdir -p logs

# ── Environment activation ────────────────────────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate learned_regions 2>/dev/null || conda activate base
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
    conda activate learned_regions 2>/dev/null || conda activate base
fi

# ── Path configuration ────────────────────────────────────────────────────────
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
ROUTER_CKPT="runs/cheap_ai/phase1A_static_region_router/best_real_region_router.pt"
ROUTER_RUN_DIR="runs/cheap_ai/phase1A_static_region_router"
OUTPUT_DIR="runs/cheap_ai/phase1A_static_region_router_audit"
SCRIPT="scripts/audit_static_region_router_outputs.py"
SLURM_SCRIPT="scripts/slurm_audit_static_region_router_outputs.sh"

# Allow overriding checkpoint via env var for auditing other models:
#   sbatch --export=ALL,ROUTER_CKPT=<path> slurm_audit_...sh
ROUTER_CKPT="${ROUTER_CKPT_OVERRIDE:-${ROUTER_CKPT}}"

echo "========================================================"
echo " Phase 1A.4: Router Output / Error Anatomy Audit"
echo " val_dir:         ${VAL_DIR}"
echo " train_dir:       ${TRAIN_DIR}"
echo " token_to_region: ${TOKEN_TO_REGION}"
echo " router_ckpt:     ${ROUTER_CKPT}"
echo " output_dir:      ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "[preflight] Checking required inputs..."

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: script not found: $SCRIPT"; exit 1
fi
echo "  [OK] $SCRIPT"

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: val_dir not found: $VAL_DIR"; exit 1
fi
N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_VAL" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $VAL_DIR"; exit 1
fi
echo "  [OK] $VAL_DIR  ($N_VAL val shards)"

if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: train_dir not found: $TRAIN_DIR"; exit 1
fi
N_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N_TRAIN" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $TRAIN_DIR"; exit 1
fi
echo "  [OK] $TRAIN_DIR  ($N_TRAIN train shards)"

if [ ! -f "$TOKEN_TO_REGION" ]; then
    echo "ERROR: token_to_region not found: $TOKEN_TO_REGION"; exit 1
fi
echo "  [OK] $TOKEN_TO_REGION"

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map ${SUPER_MAP}"
    echo "  [OK] $SUPER_MAP"
else
    echo "  [WARN] super_map not found: $SUPER_MAP — superregion metrics disabled"
fi

if [ ! -f "$ROUTER_CKPT" ]; then
    echo "ERROR: router checkpoint not found: $ROUTER_CKPT"; exit 1
fi
echo "  [OK] $ROUTER_CKPT"

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

echo "[launch] Starting router output audit..."
echo "  NOTE: Evaluation only. No training. No weight updates."
echo "  NOTE: Model input is input_ids only."
echo "  NOTE: Gold used only after prediction for metrics."
echo ""

python "$SCRIPT" \
    --val_dir           "$VAL_DIR" \
    --train_dir         "$TRAIN_DIR" \
    --token_to_region   "$TOKEN_TO_REGION" \
    $SUPER_ARG \
    --router_ckpt       "$ROUTER_CKPT" \
    --router_run_dir    "$ROUTER_RUN_DIR" \
    --output_dir        "$OUTPUT_DIR" \
    --context_len       128 \
    --d_model           256 \
    --n_layers          2 \
    --n_heads           4 \
    --batch_size        512 \
    --num_examples      30 \
    --tokenizer_name    gpt2 \
    --seed              42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: audit exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Phase 1A.4 audit complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/audit_report.md"
echo "  ${OUTPUT_DIR}/adaptive_fallback_policies.csv"
echo "  ${OUTPUT_DIR}/miss_rank_histogram.csv"
echo "  ${OUTPUT_DIR}/superregion_near_miss.csv"
echo "  ${OUTPUT_DIR}/coverage_by_k.csv"
echo "  ${OUTPUT_DIR}/router_confusion_pairs.csv"
echo "  ${OUTPUT_DIR}/confidence_bins_*.csv"
echo "  ${OUTPUT_DIR}/examples_*.md"
echo ""

# ── Verify expected outputs ───────────────────────────────────────────────────
for f in \
    "${OUTPUT_DIR}/audit_report.md" \
    "${OUTPUT_DIR}/adaptive_fallback_policies.csv" \
    "${OUTPUT_DIR}/miss_rank_histogram.csv" \
    "${OUTPUT_DIR}/coverage_by_k.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"
        exit 1
    fi
done

# ── Print final summary ───────────────────────────────────────────────────────
python - <<'PYEOF'
import csv, json, os, re, sys

d    = "runs/cheap_ai/phase1A_static_region_router_audit"
rpt  = os.path.join(d, "audit_report.md")
hist = os.path.join(d, "miss_rank_histogram.csv")
cov  = os.path.join(d, "coverage_by_k.csv")
fb   = os.path.join(d, "adaptive_fallback_policies.csv")

print("[result] Phase 1A.4 Router Audit Summary:")

try:
    rows = {r["metric"]: r["value"] for r in csv.DictReader(open(cov))}
    print(f"\n  Coverage:")
    for k in [1, 2, 4, 8, 16, 32, 64]:
        rec  = rows.get(f"gold_region_recall@{k}", "?")
        frac = rows.get(f"candidate_fraction@{k}",  "?")
        print(f"    @{k:2d}:  recall={rec:8}  cand_frac={frac}")
except Exception as e:
    print(f"  [WARN] coverage: {e}", file=sys.stderr)

try:
    print(f"\n  Miss rank histogram:")
    for r in csv.DictReader(open(hist)):
        print(f"    {r['bucket']:20s}  n={r['n']:7s}  frac={r['frac']:7s}  "
              f"cum_upper={r['cumulative_recall_upper_bound']}")
except Exception as e:
    print(f"  [WARN] miss hist: {e}", file=sys.stderr)

try:
    fb_rows = sorted(csv.DictReader(open(fb)),
                     key=lambda r: float(r.get("effective_region_recall", 0) or 0),
                     reverse=True)
    non_oracle = [r for r in fb_rows if "ORACLE" not in r.get("policy","")]
    print(f"\n  Best fallback policies (top 5):")
    for r in non_oracle[:5]:
        print(f"    {r.get('policy','?'):<40}  "
              f"recall={r.get('effective_region_recall','?'):7}  "
              f"frac={r.get('candidate_fraction','?'):7}  "
              f"fallback%={r.get('fallback_rate','?')}")
except Exception as e:
    print(f"  [WARN] fallback: {e}", file=sys.stderr)

try:
    if os.path.isfile(rpt):
        text = open(rpt, encoding="utf-8").read()
        m = re.search(r"## FINAL RECOMMENDATION:\s*(\S+)", text)
        if m:
            print(f"\n  PHASE 1A.4 RECOMMENDATION: {m.group(1)}")
except Exception as e:
    print(f"  [WARN] report: {e}", file=sys.stderr)
PYEOF

echo "[done] Audit report: ${OUTPUT_DIR}/audit_report.md"
