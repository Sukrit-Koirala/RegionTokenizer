#!/bin/bash
#SBATCH --job-name=router_ctx_ablate
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Phase 1A: Router Capacity + Context-Length Ablation
#
# QUESTION:
#   How much does transformer depth improve gold-region routing?
#   How much does longer context improve gold-region routing?
#   Which model gives the best recall-per-compute tradeoff?
#
# Known baseline (Phase 1A, seq_len=128):
#   last_token_mlp_baseline   recall@8 = 0.78037
#   mean_embedding_mlp        recall@8 = 0.66478
#   real_router_L2_ctx128     recall@8 = 0.82189
#
# Run matrix (20 runs total):
#   context_lens = 16, 32, 64, 128
#   variants = last_token_mlp, mean_embedding_mlp,
#              real_router_L1, real_router_L2, real_router_L4
#
# Safety:
#   Model input is input_ids only.
#   No h_ctx, h_raw, base_topk_ids/logits used as input.
#   context_len > shard length fails loudly (no silent padding).
#   Fail loudly on NaN.
#
# For a fast check, add --quick to run only:
#   context_lens = 32, 128
#   variants = last_token_mlp, real_router_L1, real_router_L2
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
OUTPUT_DIR="runs/cheap_ai/phase1A_router_capacity_context_ablation"
SCRIPT="scripts/train_static_region_router.py"
SLURM_SCRIPT="scripts/slurm_router_capacity_context_ablation.sh"

# ── Optional quick mode ───────────────────────────────────────────────────────
# Pass QUICK=1 to sbatch env to run only context_lens=32,128 and L1/L2:
#   sbatch --export=ALL,QUICK=1 scripts/slurm_router_capacity_context_ablation.sh
QUICK_FLAG=""
if [ "${QUICK:-0}" = "1" ]; then
    QUICK_FLAG="--quick"
    echo "[mode] QUICK mode: context_lens=32,128 variants=last_token_mlp,real_router_L1,real_router_L2"
else
    echo "[mode] FULL mode: context_lens=16,32,64,128 all 5 variants (20 runs)"
fi

echo "========================================================"
echo " Phase 1A: Router Capacity + Context-Length Ablation"
echo " train_dir:       ${TRAIN_DIR}"
echo " val_dir:         ${VAL_DIR}"
echo " token_to_region: ${TOKEN_TO_REGION}"
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
    echo "  [OK] $SUPER_MAP  (superregion aux head enabled)"
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

echo "[launch] Starting capacity + context-length ablation..."
echo "  NOTE: Model input is input_ids only."
echo "  NOTE: No h_ctx, h_raw, base_topk_ids used as input."
echo "  NOTE: context_len > shard length will fail loudly (no silent padding)."
echo "  NOTE: All efficiency numbers are theoretical estimates."
echo ""

python "$SCRIPT" \
    --train_dir                      "$TRAIN_DIR" \
    --val_dir                        "$VAL_DIR" \
    --token_to_region                "$TOKEN_TO_REGION" \
    $SUPER_ARG \
    --output_dir                     "$OUTPUT_DIR" \
    --run_capacity_context_ablation \
    --context_lens                   "16,32,64,128" \
    --variants                       "last_token_mlp,mean_embedding_mlp,real_router_L1,real_router_L2,real_router_L4" \
    --d_model                        256 \
    --n_heads                        4 \
    --dropout                        0.1 \
    --batch_size                     256 \
    --steps                          10000 \
    --eval_every                     500 \
    --lr                             3e-4 \
    --weight_decay                   0.01 \
    --grad_clip                      1.0 \
    --lambda_super                   0.2 \
    --seed                           42 \
    --base_d_model                   384 \
    --amp \
    $QUICK_FLAG

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: ablation exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Phase 1A capacity + context ablation complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/capacity_context_ablation.csv"
echo "  ${OUTPUT_DIR}/capacity_context_ablation.md"
echo "  ${OUTPUT_DIR}/capacity_context_ablation_report.md"
echo "  ${OUTPUT_DIR}/best_by_context.csv"
echo "  ${OUTPUT_DIR}/best_by_compute_tradeoff.csv"
echo ""

# ── Verify expected outputs ───────────────────────────────────────────────────
for f in \
    "${OUTPUT_DIR}/capacity_context_ablation.csv" \
    "${OUTPUT_DIR}/capacity_context_ablation_report.md"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"
        exit 1
    fi
done

# ── Print final verdict ───────────────────────────────────────────────────────
python - <<'PYEOF'
import csv, json, os, re, sys

d    = "runs/cheap_ai/phase1A_router_capacity_context_ablation"
comp = os.path.join(d, "capacity_context_ablation.csv")
rpt  = os.path.join(d, "capacity_context_ablation_report.md")
bctx = os.path.join(d, "best_by_context.csv")

try:
    rows = list(csv.DictReader(open(comp)))
    print("[result] Phase 1A Capacity + Context Ablation:")
    print(f"  {'variant':<28}  {'ctx':>4}  {'params':>9}  "
          f"{'r@4':>7}  {'r@8':>7}  {'r@16':>7}  "
          f"{'cand_f@8':>8}  {'speedup@8':>9}")
    print("  " + "─" * 90)
    for r in rows:
        print(
            f"  {r.get('variant','?'):<28}"
            f"  {r.get('context_len','?'):>4}"
            f"  {r.get('params','?'):>9}"
            f"  {r.get('region_recall@4','?'):>7}"
            f"  {r.get('region_recall@8','?'):>7}"
            f"  {r.get('region_recall@16','?'):>7}"
            f"  {r.get('candidate_fraction@8','?'):>8}"
            f"  {r.get('estimated_speedup_vs_full_lm_head@8','?'):>9}"
        )
except Exception as e:
    print(f"[result] Could not parse ablation CSV: {e}", file=sys.stderr)

try:
    if os.path.isfile(bctx):
        rows = list(csv.DictReader(open(bctx)))
        print("\n[result] Best by context:")
        for r in rows:
            print(f"  ctx={r.get('context_len','?')}  best={r.get('variant','?')}"
                  f"  r@8={r.get('region_recall@8','?')}"
                  f"  r@16={r.get('region_recall@16','?')}")
except Exception as e:
    print(f"[result] Could not parse best_by_context: {e}", file=sys.stderr)

try:
    if os.path.isfile(rpt):
        text = open(rpt, encoding="utf-8").read()
        m = re.search(
            r"Final recommendation:\s*(\S+)",
            text)
        if m:
            print(f"\n[result] PHASE 1A ABLATION FINAL RECOMMENDATION: {m.group(1)}")
        # Also extract recommended router
        mr = re.search(r"Recommended router:\s*(\S+)", text)
        if mr:
            print(f"[result] Recommended router: {mr.group(1)}")
except Exception as e:
    print(f"[result] Could not parse report: {e}", file=sys.stderr)
PYEOF

echo "[done] Report: ${OUTPUT_DIR}/capacity_context_ablation_report.md"
