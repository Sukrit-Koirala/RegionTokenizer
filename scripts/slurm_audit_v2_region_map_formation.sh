#!/bin/bash
#SBATCH --job-name=v2_region_audit
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
# Phase 1B.1: V2 Region Map Formation Audit
#
# Audit only — no training, no weight updates.
# No semantic labels. Numeric region IDs only.
#
# Analyzes:
#   1. Size/cost comparison (old K128 vs new V2A_K256)
#   2. Old→new token fragmentation
#   3. New region purity relative to old map
#   4. Representative token reports
#   5. Router miss analysis (sibling vs cross-parent misses)
#   6. New region confusion matrix
#   7. Merge simulation / upper bound
#   8. Old vs new recall-per-cost curve
#   9. Formation verdict and recommendation
#
# Override map via env vars:
#   MAP_NAME=V2B_K256
#   NEW_MAP=runs/cheap_ai/phase1B.../V2B_K256/token_to_region.json
#   ROUTER_CKPT=runs/cheap_ai/phase1B.../router_runs/V2B_K256/best_real_region_router.pt
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

# ── Path configuration (override via env vars) ────────────────────────────────
MAP_NAME="${MAP_NAME:-V2A_K256}"
MAPS_ROOT="runs/cheap_ai/phase1B_compute_aware_region_maps"

OLD_MAP="runs/region_maps_128/token_to_region.json"
NEW_MAP="${NEW_MAP:-${MAPS_ROOT}/${MAP_NAME}/token_to_region.json}"
NEW_MAP_DIR="${NEW_MAP_DIR:-${MAPS_ROOT}/${MAP_NAME}}"
ROUTER_CKPT="${ROUTER_CKPT:-${MAPS_ROOT}/router_runs/${MAP_NAME}/best_real_region_router.pt}"
ROUTER_RUN_DIR="${ROUTER_RUN_DIR:-${MAPS_ROOT}/router_runs/${MAP_NAME}}"
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
OUTPUT_DIR="${MAPS_ROOT}/formation_audit/${MAP_NAME}"
SCRIPT="scripts/audit_v2_region_map_formation.py"
SLURM_SCRIPT="scripts/slurm_audit_v2_region_map_formation.sh"

echo "========================================================"
echo " Phase 1B.1: V2 Region Map Formation Audit"
echo " map_name:   ${MAP_NAME}"
echo " old_map:    ${OLD_MAP}"
echo " new_map:    ${NEW_MAP}"
echo " router_ckpt: ${ROUTER_CKPT}"
echo " output_dir: ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "[preflight] Checking required inputs..."

[ ! -f "$SCRIPT" ] && { echo "ERROR: $SCRIPT not found"; exit 1; }
echo "  [OK] $SCRIPT"

[ ! -f "$OLD_MAP" ] && { echo "ERROR: old_map not found: $OLD_MAP"; exit 1; }
echo "  [OK] $OLD_MAP"

[ ! -f "$NEW_MAP" ] && { echo "ERROR: new_map not found: $NEW_MAP"; exit 1; }
echo "  [OK] $NEW_MAP"

[ ! -d "$TRAIN_DIR" ] && { echo "ERROR: train_dir not found: $TRAIN_DIR"; exit 1; }
N_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
[ "$N_TRAIN" -eq 0 ] && { echo "ERROR: no shards in $TRAIN_DIR"; exit 1; }
echo "  [OK] $TRAIN_DIR  ($N_TRAIN shards)"

[ ! -d "$VAL_DIR" ] && { echo "ERROR: val_dir not found: $VAL_DIR"; exit 1; }
N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
[ "$N_VAL" -eq 0 ] && { echo "ERROR: no shards in $VAL_DIR"; exit 1; }
echo "  [OK] $VAL_DIR  ($N_VAL shards)"

CKPT_ARG=""
if [ -f "$ROUTER_CKPT" ]; then
    CKPT_ARG="--router_ckpt ${ROUTER_CKPT} --router_run_dir ${ROUTER_RUN_DIR}"
    echo "  [OK] $ROUTER_CKPT  (router inference enabled)"
else
    echo "  [WARN] router_ckpt not found: $ROUTER_CKPT"
    echo "         Parts 5-8 will be skipped (map analysis will still run)"
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

echo "[launch] Starting V2 region map formation audit..."
echo "  NOTE: Evaluation/audit only. No training."
echo "  NOTE: No semantic labels. Region IDs only."
echo "  NOTE: Gold used only after router prediction."
echo ""

python "$SCRIPT" \
    --map_name          "$MAP_NAME" \
    --old_map           "$OLD_MAP" \
    --new_map           "$NEW_MAP" \
    --new_map_dir       "$NEW_MAP_DIR" \
    $CKPT_ARG \
    --train_dir         "$TRAIN_DIR" \
    --val_dir           "$VAL_DIR" \
    --output_dir        "$OUTPUT_DIR" \
    --tokenizer_name    gpt2 \
    --context_len       128 \
    --d_model           256 \
    --n_layers          2 \
    --n_heads           4 \
    --batch_size        512 \
    --seed              42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""; echo "ERROR: audit exited with code $EXIT"; exit $EXIT
fi

echo ""
echo "========================================================"
echo " Phase 1B.1 audit complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/formation_audit_report.md"
echo "  ${OUTPUT_DIR}/size_comparison.csv"
echo "  ${OUTPUT_DIR}/old_to_new_fragmentation.csv"
echo "  ${OUTPUT_DIR}/new_region_old_parent_purity.csv"
echo "  ${OUTPUT_DIR}/old_problem_region_splits.md"
echo "  ${OUTPUT_DIR}/v2_miss_decomposition.md"
echo "  ${OUTPUT_DIR}/merge_simulation_policies.csv"
echo "  ${OUTPUT_DIR}/old_vs_new_curve.csv"
echo ""

# ── Verify expected outputs ───────────────────────────────────────────────────
for f in \
    "${OUTPUT_DIR}/formation_audit_report.md" \
    "${OUTPUT_DIR}/size_comparison.csv" \
    "${OUTPUT_DIR}/old_to_new_fragmentation.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected output missing: $f"; exit 1
    fi
done

# ── Print final summary ───────────────────────────────────────────────────────
python - <<PYEOF
import csv, json, os, re, sys

d   = "${OUTPUT_DIR}"
rpt = os.path.join(d, "formation_audit_report.md")
sz  = os.path.join(d, "size_comparison.csv")
frg = os.path.join(d, "old_to_new_fragmentation.csv")
pur = os.path.join(d, "new_region_old_parent_purity.csv")
mis = os.path.join(d, "v2_miss_decomposition.csv")
mrg = os.path.join(d, "merge_simulation_policies.csv")
crv = os.path.join(d, "old_vs_new_curve.csv")

try:
    rows = list(csv.DictReader(open(sz)))
    print("[result] Size comparison:")
    for r in rows:
        print(f"  {r.get('map','?'):<15}  K={r.get('n_regions','?'):>4}"
              f"  max={r.get('max_size','?'):>6}  p95={r.get('p95_size','?'):>6}"
              f"  gini={r.get('size_gini','?'):>6}  over_2x={r.get('num_over_2x_mean','?')}")
except Exception as e:
    print(f"[WARN] {e}", file=sys.stderr)

try:
    rows = list(csv.DictReader(open(frg)))
    n_ch = [int(r.get("num_children",0)) for r in rows if int(r.get("old_size",0))>0]
    old52 = next((r for r in rows if r.get("old_region","")=="52"),{})
    print(f"\n[result] Fragmentation:  avg_children={sum(n_ch)/max(len(n_ch),1):.1f}  "
          f"old52_children={old52.get('num_children','?')}  "
          f"old52_eff={old52.get('effective_children','?')}")
except Exception as e:
    print(f"[WARN] {e}", file=sys.stderr)

try:
    rows = list(csv.DictReader(open(pur)))
    fracs = [float(r.get("dominant_old_frac",0)) for r in rows if int(r.get("new_size",0))>0]
    if fracs:
        print(f"[result] New purity:  avg_dom_frac={sum(fracs)/len(fracs):.4f}"
              f"  pct_high(>0.8)={sum(1 for f in fracs if f>0.8)/len(fracs):.3f}"
              f"  pct_low(<0.5)={sum(1 for f in fracs if f<0.5)/len(fracs):.3f}")
except Exception as e:
    print(f"[WARN] {e}", file=sys.stderr)

try:
    if os.path.isfile(mis):
        rows = list(csv.DictReader(open(mis)))
        if rows:
            r = rows[0]
            print(f"\n[result] Miss decomposition:  sibling={r.get('pct_sibling_miss','?')}"
                  f"  hard_cross={r.get('pct_hard_cross_old_miss','?')}"
                  f"  r9_16={r.get('pct_gold_rank_9_16','?')}")
except Exception as e:
    print(f"[WARN] {e}", file=sys.stderr)

try:
    if os.path.isfile(crv):
        rows = list(csv.DictReader(open(crv)))
        print(f"\n[result] Coverage curve (new map):")
        for r in rows:
            if r.get("k","") in ["8","16","32"]:
                print(f"  @{r['k']:2s}: old_recall={r.get('old_recall','?'):7}  old_frac={r.get('old_cand_frac','?'):7}"
                      f"  new_recall={r.get('new_recall','?'):7}  new_frac={r.get('new_cand_frac','?'):7}"
                      f"  delta_recall={r.get('delta_recall','?')}")
except Exception as e:
    print(f"[WARN] {e}", file=sys.stderr)

try:
    if os.path.isfile(mrg):
        rows = sorted(csv.DictReader(open(mrg)),
                      key=lambda r: float(r.get("effective_recall",0) or 0), reverse=True)
        non_oracle = [r for r in rows if "DIAGNOSTIC" not in r.get("policy","")]
        print(f"\n[result] Best merge policy: {non_oracle[0].get('policy','?') if non_oracle else '?'}"
              f"  recall={non_oracle[0].get('effective_recall','?') if non_oracle else '?'}"
              f"  frac={non_oracle[0].get('candidate_fraction','?') if non_oracle else '?'}")
except Exception as e:
    print(f"[WARN] {e}", file=sys.stderr)

try:
    if os.path.isfile(rpt):
        text = open(rpt, encoding="utf-8").read()
        m = re.search(r"## FINAL RECOMMENDATION:\s*(\S+)", text)
        if m: print(f"\n[result] PHASE 1B.1 RECOMMENDATION: {m.group(1)}")
except Exception as e:
    print(f"[WARN] {e}", file=sys.stderr)
PYEOF

echo "[done] Report: ${OUTPUT_DIR}/formation_audit_report.md"
