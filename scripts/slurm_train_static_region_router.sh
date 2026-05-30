#!/bin/bash
#SBATCH --job-name=static_region_router
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
# Phase 1A: Static Region Router
#
# CORE QUESTION:
#   Can a cheap causal attention router predict the gold-containing static region?
#
# Model input:  input_ids only
# Model target: token_to_region[gold_token]
# No h_ctx, no h_raw, no base_topk_ids as input. No full-vocab scoring.
# No gold force-inclusion. No token-level reranking. Fail loudly on NaN.
#
# Variants:
#   real_region_router         (real token_to_region map — main model)
#   shuffled_region_router     (shuffled map, same region sizes — control)
#   random_region_router       (random map, same region sizes — control)
#
# Baselines:
#   frequency_prior_baseline   (non-neural — rank by training frequency)
#   last_token_mlp_baseline    (embed last token, MLP, region logits)
#   mean_embedding_mlp_baseline (mean-pool embeddings, MLP)
#
# Proceed only if:
#   real_region_router recall@8 > frequency_prior recall@8 by clear margin
#   real_region_router recall@8 > shuffled/random recall@8
#   candidate_fraction@8 meaningfully below 0.25
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
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/cheap_ai/phase1A_static_region_router"
SCRIPT="scripts/train_static_region_router.py"
SLURM_SCRIPT="scripts/slurm_train_static_region_router.sh"

echo "========================================================"
echo " Phase 1A: Static Region Router"
echo " train_dir:         ${TRAIN_DIR}"
echo " val_dir:           ${VAL_DIR}"
echo " token_to_region:   ${TOKEN_TO_REGION}"
echo " output_dir:        ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "[preflight] Checking required inputs..."

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: script not found: $SCRIPT"; exit 1
fi
echo "  [OK] $SCRIPT"

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

if [ ! -f "$TOKEN_TO_REGION" ]; then
    echo "ERROR: token_to_region not found: $TOKEN_TO_REGION"; exit 1
fi
echo "  [OK] $TOKEN_TO_REGION"

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map ${SUPER_MAP}"
    echo "  [OK] $SUPER_MAP  (superregion auxiliary head enabled)"
else
    echo "  [WARN] super_map not found: $SUPER_MAP — superregion aux head disabled"
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

echo "[launch] Starting Phase 1A Static Region Router training..."
echo "  NOTE: Model input is input_ids only."
echo "  NOTE: No h_ctx, h_raw, base_topk_ids/logits used as input."
echo "  NOTE: Gold region computed from token_to_region[gold_token]."
echo "  NOTE: Unknown-region rows excluded from training loss."
echo "  NOTE: Fail loudly on NaN."
echo ""

python "$SCRIPT" \
    --train_dir         "$TRAIN_DIR" \
    --val_dir           "$VAL_DIR" \
    --token_to_region   "$TOKEN_TO_REGION" \
    $SUPER_ARG \
    --output_dir        "$OUTPUT_DIR" \
    --d_model           256 \
    --n_layers          2 \
    --n_heads           4 \
    --dropout           0.1 \
    --max_seq_len       128 \
    --batch_size        256 \
    --steps             10000 \
    --eval_every        500 \
    --lr                3e-4 \
    --weight_decay      0.01 \
    --grad_clip         1.0 \
    --lambda_super      0.2 \
    --seed              42 \
    --amp

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: training exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Phase 1A complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/phase1A_static_region_router_report.md"
echo "  ${OUTPUT_DIR}/phase1A_region_router_comparison.csv"
echo "  ${OUTPUT_DIR}/coverage_by_k.csv"
echo "  ${OUTPUT_DIR}/slice_metrics.csv"
echo "  ${OUTPUT_DIR}/best_metrics.json"
echo ""

# ── Verify expected outputs ───────────────────────────────────────────────────
for f in \
    "${OUTPUT_DIR}/phase1A_static_region_router_report.md" \
    "${OUTPUT_DIR}/phase1A_region_router_comparison.csv" \
    "${OUTPUT_DIR}/coverage_by_k.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"
        exit 1
    fi
done

# ── Print final verdict ───────────────────────────────────────────────────────
python - <<'PYEOF'
import csv, json, os, re, sys

d    = "runs/cheap_ai/phase1A_static_region_router"
comp = os.path.join(d, "phase1A_region_router_comparison.csv")
cov  = os.path.join(d, "coverage_by_k.csv")
rpt  = os.path.join(d, "phase1A_static_region_router_report.md")
bm   = os.path.join(d, "best_metrics.json")

try:
    rows = list(csv.DictReader(open(comp)))
    print("[result] Phase 1A region router comparison:")
    print(f"  {'variant':<36}  acc@1   rec@4   rec@8   rec@16  cfrac@8  b5any@8  flop_red@8")
    print("  " + "─" * 90)
    for r in rows:
        print(
            f"  {r.get('variant','?'):<36}"
            f"  {r.get('region_acc@1','?'):6}"
            f"  {r.get('region_recall@4','?'):6}"
            f"  {r.get('region_recall@8','?'):6}"
            f"  {r.get('region_recall@16','?'):6}"
            f"  {r.get('candidate_fraction@8','?'):7}"
            f"  {r.get('base_top5_any_in_C@8','?'):7}"
            f"  {r.get('theoretical_output_flop_reduction@8','?')}"
        )
except Exception as e:
    print(f"[result] Could not parse comparison: {e}", file=sys.stderr)

try:
    rows = list(csv.DictReader(open(cov)))
    print("\n[result] Coverage by k (real_region_router):")
    print(f"  {'k':>4}  {'recall@k':>9}  {'avg_cand':>9}  {'fraction':>9}  {'flop_red':>9}")
    for r in rows:
        if r.get("variant", "") == "real_region_router" or True:
            print(
                f"  {r.get('k','?'):>4}"
                f"  {r.get('gold_region_recall','?'):>9}"
                f"  {r.get('avg_candidate_set_size','?'):>9}"
                f"  {r.get('candidate_fraction','?'):>9}"
                f"  {r.get('flop_reduction','?'):>9}"
            )
except Exception as e:
    print(f"[result] Could not parse coverage_by_k: {e}", file=sys.stderr)

try:
    if os.path.isfile(rpt):
        text = open(rpt, encoding="utf-8").read()
        m = re.search(
            r"recommendation:\s*(PROCEED_TO_ROUTED_CANDIDATE_SOFTMAX"
            r"|PARTIAL_GO_IMPROVE_ROUTER"
            r"|DO_NOT_PROCEED)",
            text)
        if m:
            print(f"\n[result] PHASE 1A VERDICT: {m.group(1)}")
except Exception as e:
    print(f"[result] Could not parse report: {e}", file=sys.stderr)

try:
    data = json.load(open(bm))
    real = data.get("real_region_router", {})
    if real:
        print(f"\n[result] real_region_router best metrics:")
        for k in ["region_acc@1", "region_recall@4", "region_recall@8", "region_recall@16",
                  "candidate_fraction@8", "avg_candidate_set_size@8",
                  "gold_token_coverage@8", "base_top5_any_in_C@8",
                  "theoretical_output_flop_reduction@8", "router_params"]:
            v = real.get(k)
            print(f"  {k:<46} = {v}")
except Exception as e:
    print(f"[result] Could not parse best_metrics: {e}", file=sys.stderr)
PYEOF

echo "[done] Report: ${OUTPUT_DIR}/phase1A_static_region_router_report.md"
