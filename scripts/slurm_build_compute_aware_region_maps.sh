#!/bin/bash
#SBATCH --job-name=build_region_v2
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Phase 1B: Build Compute-Aware Predictive Interference Region Maps V2
#
# Builds empirical region maps from model behaviour (base_topk coactivation),
# with compute-aware constraints.
#
# NO manual labels. NO semantic categories.
# Only token statistics and graph structure.
#
# Map variants:
#   V2A_K256, V2A_K512       balanced interference, no BPE penalty
#   V2B_K256, V2B_K512       balanced interference + BPE surface penalty
#   V2C_K256_head1024         fixed-head (1024) + routed tail regions
#   V2C_K512_head1024         fixed-head (1024) + K=512 tail regions
#
# Key properties vs old 128-region map:
#   - full 50257 vocab assignment (old map ≈ 20k known tokens)
#   - enforced size balance (no huge bloat regions like old region 52 size 2607)
#   - BPE surface compatibility penalty (reduces subword mixing)
#   - head-tail routing for V2C (head tokens always covered, router focuses on tail)
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
OLD_MAP="runs/region_maps_128/token_to_region.json"
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
AUDIT_DIR="runs/cheap_ai/phase1A_static_region_router_audit"
OUTPUT_ROOT="runs/cheap_ai/phase1B_compute_aware_region_maps"
BUILD_SCRIPT="scripts/build_compute_aware_region_maps.py"
EVAL_SCRIPT="scripts/evaluate_compute_aware_region_maps.py"
SLURM_SCRIPT="scripts/slurm_build_compute_aware_region_maps.sh"

echo "========================================================"
echo " Phase 1B: Build Compute-Aware Region Maps"
echo " train_dir:   ${TRAIN_DIR}"
echo " old_map:     ${OLD_MAP}"
echo " output_root: ${OUTPUT_ROOT}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "[preflight] Checking required inputs..."

if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "ERROR: build script not found: $BUILD_SCRIPT"; exit 1
fi
echo "  [OK] $BUILD_SCRIPT"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: eval script not found: $EVAL_SCRIPT"; exit 1
fi
echo "  [OK] $EVAL_SCRIPT"

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

if [ ! -f "$OLD_MAP" ]; then
    echo "  [WARN] old_map not found: $OLD_MAP — will skip comparison"
    OLD_MAP_ARG=""
else
    OLD_MAP_ARG="--old_map ${OLD_MAP}"
    echo "  [OK] $OLD_MAP"
fi

SMALL_CKPT_ARG=""
if [ -f "$SMALL_CKPT" ]; then
    SMALL_CKPT_ARG="--small_ckpt ${SMALL_CKPT}"
    echo "  [OK] $SMALL_CKPT  (embedding fallback enabled)"
else
    echo "  [WARN] small_ckpt not found — embedding fallback disabled"
fi

AUDIT_ARG=""
if [ -d "$AUDIT_DIR" ]; then
    AUDIT_ARG="--audit_dir ${AUDIT_DIR}"
    echo "  [OK] $AUDIT_DIR  (Phase 1A.4 audit feedback available)"
else
    echo "  [INFO] audit_dir not found — run Phase 1A.4 first for richer diagnostics"
fi

echo ""
echo "[preflight] Checking syntax..."
python -m py_compile "$BUILD_SCRIPT"
echo "  [OK] $BUILD_SCRIPT compiles"
python -m py_compile "$EVAL_SCRIPT"
echo "  [OK] $EVAL_SCRIPT compiles"
if [ -f "$SLURM_SCRIPT" ]; then
    bash -n "$SLURM_SCRIPT"
    echo "  [OK] $SLURM_SCRIPT syntax"
fi

echo ""
echo "[preflight] Checking scientific dependencies..."
python -c "import scipy.sparse; import sklearn; print('  [OK] scipy and sklearn available')"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "$OUTPUT_ROOT"

echo "[launch] Building compute-aware region maps..."
echo "  NOTE: Train data only for map construction."
echo "  NOTE: Val data NOT used for map building."
echo "  NOTE: NO manual labels or semantic categories."
echo "  NOTE: Full 50k vocab assignment via graph→embedding→surface→deterministic fallback."
echo ""

python "$BUILD_SCRIPT" \
    --train_dir         "$TRAIN_DIR" \
    --val_dir           "$VAL_DIR" \
    $OLD_MAP_ARG \
    $SMALL_CKPT_ARG \
    $AUDIT_ARG \
    --output_root       "$OUTPUT_ROOT" \
    --tokenizer_name    gpt2 \
    --vocab_size        50257 \
    --graph_topk        64 \
    --pair_topk         32 \
    --top_neighbors_per_token 128 \
    --graph_embed_dim   64 \
    --variants          "V2A_K256,V2A_K512,V2B_K256,V2B_K512,V2C_K256_head1024,V2C_K512_head1024" \
    --bpe_lambda        0.75 \
    --head_size         1024 \
    --max_size_mult     2.5 \
    --min_size_mult     0.25 \
    --seed              42

BUILD_EXIT=$?
if [ $BUILD_EXIT -ne 0 ]; then
    echo ""; echo "ERROR: build exited with code $BUILD_EXIT"; exit $BUILD_EXIT
fi

echo ""
echo "========================================================"
echo " Map building complete. Running evaluation..."
echo "========================================================"
echo ""

mkdir -p "${OUTPUT_ROOT}/eval"

python "$EVAL_SCRIPT" \
    --maps_root         "$OUTPUT_ROOT" \
    --train_dir         "$TRAIN_DIR" \
    --val_dir           "$VAL_DIR" \
    $OLD_MAP_ARG \
    --output_dir        "${OUTPUT_ROOT}/eval" \
    --ks                "1,2,4,8,16,32,64" \
    --seed              42

EVAL_EXIT=$?
if [ $EVAL_EXIT -ne 0 ]; then
    echo ""; echo "ERROR: eval exited with code $EVAL_EXIT"; exit $EVAL_EXIT
fi

echo ""
echo "========================================================"
echo " Phase 1B complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_ROOT}/phase1B_region_map_report.md"
echo "  ${OUTPUT_ROOT}/map_diagnostics.csv"
echo "  ${OUTPUT_ROOT}/eval/map_eval_summary.csv"
echo "  ${OUTPUT_ROOT}/eval/map_eval_report.md"
echo "  ${OUTPUT_ROOT}/V2A_K256/token_to_region.json"
echo "  ${OUTPUT_ROOT}/V2B_K256/token_to_region.json"
echo "  ${OUTPUT_ROOT}/V2C_K256_head1024/token_to_region.json"
echo "  ${OUTPUT_ROOT}/V2C_K256_head1024/head_token_ids.npy"
echo ""

# ── Verify outputs ────────────────────────────────────────────────────────────
for f in \
    "${OUTPUT_ROOT}/phase1B_region_map_report.md" \
    "${OUTPUT_ROOT}/map_diagnostics.csv" \
    "${OUTPUT_ROOT}/eval/map_eval_summary.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"; exit 1
    fi
done

# ── Print final summary ───────────────────────────────────────────────────────
python - <<'PYEOF'
import csv, json, os, re, sys

root = "runs/cheap_ai/phase1B_compute_aware_region_maps"
diag = os.path.join(root, "map_diagnostics.csv")
eval_s = os.path.join(root, "eval", "map_eval_summary.csv")
rpt    = os.path.join(root, "phase1B_region_map_report.md")

try:
    rows = list(csv.DictReader(open(diag)))
    print("[result] Map diagnostics:")
    print(f"  {'map':<30}  {'K':>4}  {'max_sz':>7}  {'over_2x':>7}  {'gini':>6}  {'intra/inter':>12}")
    print("  " + "─" * 70)
    for r in rows:
        print(f"  {r.get('map_name','?'):<30}  {r.get('n_regions','?'):>4}"
              f"  {r.get('max_size','?'):>7}  {r.get('num_regions_over_2x_mean','?'):>7}"
              f"  {r.get('size_gini','?'):>6}  {r.get('intra_inter_ratio','?'):>12}")
except Exception as e:
    print(f"[result] WARN diagnostics: {e}", file=sys.stderr)

try:
    rows = list(csv.DictReader(open(eval_s)))
    print("\n[result] Eval summary (frequency prior):")
    print(f"  {'map':<30}  {'r@8':>7}  {'frac@8':>7}  {'score@8':>8}  {'r@16':>7}  {'frac@16':>7}")
    print("  " + "─" * 70)
    for r in rows:
        print(f"  {r.get('map_name','?'):<30}"
              f"  {r.get('freq_prior_recall@8','?'):>7}"
              f"  {r.get('candidate_fraction@8','?'):>7}"
              f"  {r.get('recall_per_frac_score@8','?'):>8}"
              f"  {r.get('freq_prior_recall@16','?'):>7}"
              f"  {r.get('candidate_fraction@16','?'):>7}")
except Exception as e:
    print(f"[result] WARN eval: {e}", file=sys.stderr)

try:
    if os.path.isfile(rpt):
        text = open(rpt, encoding="utf-8").read()
        m = re.search(r"## FINAL RECOMMENDATION:\s*(\S+)", text)
        if m: print(f"\n[result] PHASE 1B RECOMMENDATION: {m.group(1)}")
except Exception as e:
    print(f"[result] WARN report: {e}", file=sys.stderr)
PYEOF

echo "[done] Phase 1B: ${OUTPUT_ROOT}/phase1B_region_map_report.md"
echo "       Next: train router via slurm_train_router_on_region_v2_maps.sh"
