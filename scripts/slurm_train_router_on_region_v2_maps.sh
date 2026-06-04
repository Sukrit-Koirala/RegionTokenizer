#!/bin/bash
#SBATCH --job-name=router_v2_sweep
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Phase 1B: Train Static Region Router on V2 Maps and Collect Comparison
#
# Trains the same cheap causal transformer router (d_model=256, L=2, H=4)
# on each V2 region map and evaluates candidate compression.
#
# For V2C maps: passes --head_token_ids so router only routes tail tokens.
# Coverage = head_covered OR tail_region_in_topk.
#
# Output per map: runs/cheap_ai/phase1B_compute_aware_region_maps/router_runs/<map_name>/
# Final table:   runs/cheap_ai/phase1B_compute_aware_region_maps/router_v2_comparison.csv
#
# Usage: sbatch scripts/slurm_train_router_on_region_v2_maps.sh
# To run a subset:
#   sbatch --export=ALL,VARIANTS="V2B_K256,V2C_K256_head1024" scripts/slurm_train_router_on_region_v2_maps.sh
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

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
MAPS_ROOT="runs/cheap_ai/phase1B_compute_aware_region_maps"
ROUTER_RUNS="${MAPS_ROOT}/router_runs"
TRAIN_SCRIPT="scripts/train_static_region_router.py"
AUDIT_SCRIPT="scripts/audit_static_region_router_outputs.py"

# Allow override via env var
VARIANTS="${VARIANTS:-V2A_K256 V2A_K512 V2B_K256 V2B_K512 V2C_K256_head1024 V2C_K512_head1024}"

# Router hyperparameters (same as Phase 1A)
D_MODEL=256
N_LAYERS=2
N_HEADS=4
CONTEXT_LEN=128
STEPS=10000
EVAL_EVERY=500
BATCH_SIZE=256
LR=3e-4
LAMBDA_SUPER=0.2
SEED=42

echo "========================================================"
echo " Phase 1B: Train Router on V2 Region Maps"
echo " maps_root: ${MAPS_ROOT}"
echo " variants:  ${VARIANTS}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "[preflight] Checking inputs..."
[ ! -f "$TRAIN_SCRIPT" ] && { echo "ERROR: $TRAIN_SCRIPT"; exit 1; }
echo "  [OK] $TRAIN_SCRIPT"
[ ! -d "$TRAIN_DIR" ]    && { echo "ERROR: $TRAIN_DIR"; exit 1; }
echo "  [OK] $TRAIN_DIR"
[ ! -d "$VAL_DIR" ]      && { echo "ERROR: $VAL_DIR"; exit 1; }
echo "  [OK] $VAL_DIR"
[ ! -d "$MAPS_ROOT" ]    && { echo "ERROR: $MAPS_ROOT — run build first"; exit 1; }
echo "  [OK] $MAPS_ROOT"

python -m py_compile "$TRAIN_SCRIPT"
echo "  [OK] $TRAIN_SCRIPT compiles"

SUPER_ARG=""
[ -f "$SUPER_MAP" ] && SUPER_ARG="--super_map ${SUPER_MAP}"

mkdir -p "$ROUTER_RUNS"

# ── Per-map training loop ─────────────────────────────────────────────────────
RESULTS_CSV="${MAPS_ROOT}/router_v2_comparison.csv"
echo "map_name,K,head_size,effective_recall@8,effective_recall@16,effective_recall@32,candidate_fraction@8,candidate_fraction@16,candidate_fraction@32,avg_candidate_size@8,head_gold_rate,tail_recall@8,params,speedup_est@8,recommendation" > "$RESULTS_CSV"

for VARIANT in $VARIANTS; do
    MAP_DIR="${MAPS_ROOT}/${VARIANT}"
    if [ ! -d "$MAP_DIR" ]; then
        echo "[SKIP] $VARIANT — map directory not found: $MAP_DIR"
        continue
    fi
    TOKEN_TO_REGION="${MAP_DIR}/token_to_region.json"
    if [ ! -f "$TOKEN_TO_REGION" ]; then
        echo "[SKIP] $VARIANT — token_to_region.json not found"
        continue
    fi

    RUN_DIR="${ROUTER_RUNS}/${VARIANT}"
    mkdir -p "$RUN_DIR"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo " Training router on: ${VARIANT}"
    echo " map:     ${TOKEN_TO_REGION}"
    echo " run_dir: ${RUN_DIR}"
    echo "────────────────────────────────────────────────────────────"

    # Check for V2C head file
    HEAD_ARGS=""
    HEAD_FILE="${MAP_DIR}/head_token_ids.npy"
    if [ -f "$HEAD_FILE" ]; then
        HEAD_ARGS="--head_token_ids ${HEAD_FILE} --head_tail_mode"
        echo "  [V2C] head_tail mode  head_file=${HEAD_FILE}"
    fi

    python "$TRAIN_SCRIPT" \
        --train_dir         "$TRAIN_DIR" \
        --val_dir           "$VAL_DIR" \
        --token_to_region   "$TOKEN_TO_REGION" \
        $SUPER_ARG \
        --output_dir        "$RUN_DIR" \
        $HEAD_ARGS \
        --d_model           $D_MODEL \
        --n_layers          $N_LAYERS \
        --n_heads           $N_HEADS \
        --max_seq_len       $CONTEXT_LEN \
        --batch_size        $BATCH_SIZE \
        --steps             $STEPS \
        --eval_every        $EVAL_EVERY \
        --lr                $LR \
        --lambda_super      $LAMBDA_SUPER \
        --seed              $SEED \
        --amp

    TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "  [ERROR] training failed for $VARIANT (exit $TRAIN_EXIT)"
        continue
    fi

    echo "  [done] training complete for $VARIANT"

    # ── Run audit on trained model ────────────────────────────────────────────
    if [ -f "$AUDIT_SCRIPT" ]; then
        AUDIT_OUT="${RUN_DIR}/audit"
        mkdir -p "$AUDIT_OUT"

        # Find best checkpoint
        BEST_CKPT=$(ls "${RUN_DIR}"/best_*.pt 2>/dev/null | head -1)
        if [ -n "$BEST_CKPT" ]; then
            echo "  [audit] running on: $BEST_CKPT"
            python "$AUDIT_SCRIPT" \
                --val_dir           "$VAL_DIR" \
                --train_dir         "$TRAIN_DIR" \
                --token_to_region   "$TOKEN_TO_REGION" \
                $SUPER_ARG \
                --router_ckpt       "$BEST_CKPT" \
                --router_run_dir    "$RUN_DIR" \
                --output_dir        "$AUDIT_OUT" \
                --context_len       $CONTEXT_LEN \
                --d_model           $D_MODEL \
                --n_layers          $N_LAYERS \
                --n_heads           $N_HEADS \
                --batch_size        512 \
                --num_examples      20 \
                --no_tokenizer \
                --seed              $SEED 2>/dev/null || echo "  [WARN] audit failed for $VARIANT (non-fatal)"
        fi
    fi

    # ── Collect results ───────────────────────────────────────────────────────
    python - <<PYEOF
import json, csv, os, sys

run_dir    = "${RUN_DIR}"
map_dir    = "${MAP_DIR}"
variant    = "${VARIANT}"
results_csv= "${RESULTS_CSV}"
head_file  = "${MAP_DIR}/head_token_ids.npy"

# Load best metrics
bm_path = os.path.join(run_dir, "best_metrics.json")
bm = {}
try:
    all_bm = json.load(open(bm_path))
    # find the real_region_router entry, or any entry that's not baseline
    for k, v in all_bm.items():
        if "real" in k or "router" in k: bm = v; break
    if not bm and all_bm: bm = list(all_bm.values())[0]
except Exception as e:
    print(f"  [WARN] could not load best_metrics: {e}", file=sys.stderr)

# Load map config
cfg = {}
try:
    cfg = json.load(open(os.path.join(map_dir, "map_config.json")))
except Exception: pass

K         = cfg.get("K", "?")
head_size = cfg.get("head_size", 0) or 0
head_gold_rate = bm.get("head_gold_rate", "nan") if "head_gold_rate" in bm else "nan"

# Effective recall = head + tail routing
r8  = bm.get("region_recall@8",  bm.get("effective_recall@8",  "nan"))
r16 = bm.get("region_recall@16", bm.get("effective_recall@16", "nan"))
r32 = bm.get("region_recall@32", bm.get("effective_recall@32", "nan"))
f8  = bm.get("candidate_fraction@8",  "nan")
f16 = bm.get("candidate_fraction@16", "nan")
f32 = bm.get("candidate_fraction@32", "nan")
cs8 = bm.get("avg_candidate_set_size@8", "nan")
params = bm.get("router_params", "nan")
flop_red = bm.get("theoretical_output_flop_reduction@8", "nan")

# Speedup estimate: 1/(1-flop_red) approximately
try:
    sp = 1.0 / (1.0 - float(flop_red)) if float(flop_red) < 1.0 else "nan"
    speedup = f"{sp:.3f}"
except Exception:
    speedup = "nan"

# Recommendation
try:
    r8f = float(r8)
    f8f = float(f8)
    if r8f >= 0.88 and f8f <= 0.20:
        rec = "PROCEED_TO_ROUTED_CANDIDATE_SOFTMAX"
    elif r8f >= 0.82:
        rec = "PARTIAL_GO_IMPROVE_ROUTER"
    else:
        rec = "NEED_IMPROVEMENT"
except Exception:
    rec = "UNKNOWN"

row = [variant, K, head_size, r8, r16, r32, f8, f16, f32, cs8,
       head_gold_rate, r8, params, speedup, rec]

with open(results_csv, "a", newline="") as f:
    csv.writer(f).writerow(row)

print(f"  [result] {variant}: recall@8={r8}  frac@8={f8}  speedup@8={speedup}  rec={rec}")
PYEOF

done

# ── Print final comparison ────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Phase 1B Router V2 Comparison  $(date)"
echo "========================================================"
python - <<'PYEOF'
import csv, os

f = "runs/cheap_ai/phase1B_compute_aware_region_maps/router_v2_comparison.csv"
if not os.path.isfile(f):
    print("[WARN] No comparison CSV found")
    exit()

rows = list(csv.DictReader(open(f)))
print(f"\n  {'map':<30}  {'K':>4}  {'r@8':>7}  {'r@16':>7}  {'frac@8':>7}  {'speedup@8':>10}  rec")
print("  " + "─" * 80)
for r in rows:
    print(
        f"  {r.get('map_name','?'):<30}"
        f"  {r.get('K','?'):>4}"
        f"  {r.get('effective_recall@8','?'):>7}"
        f"  {r.get('effective_recall@16','?'):>7}"
        f"  {r.get('candidate_fraction@8','?'):>7}"
        f"  {r.get('speedup_est@8','?'):>10}"
        f"  {r.get('recommendation','?')}")
PYEOF

echo ""
echo "Results: ${MAPS_ROOT}/router_v2_comparison.csv"
echo "Done."
