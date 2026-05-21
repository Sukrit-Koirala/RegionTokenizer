#!/bin/bash
#SBATCH --job-name=tcr_hparam_test
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
# Token Confuser Resolver V1 — Hyperparameter Stability Test
#
# Background:
#   Aggressive run (lr=5e-5, lambda_rank=0.5, lambda_kl=0.1) was badly negative:
#     full_vocab_gain_all   = -0.038068
#     inside_gate_gain_all  = -0.228398
#     changed_to_gold_rate  = 0.0080
#     changed_away_rate     = 0.0197
#   Architecture passed step-0 identity. Logit edits were too aggressive.
#
# This script tests 4 conservative configs to diagnose whether the architecture
# has a useful calibrated training signal.
#
# Configs (2000 steps each, eval every 500):
#   A: stable_no_rank     — CE + strong KL + delta reg, no rank loss
#   B: gentle_rank005     — very gentle rank, strong KL + delta reg
#   C: rank010            — slightly stronger rank, strong KL + delta reg
#   D: extra_regularized  — smallest LR, strongest KL + delta reg
#
# All output under: runs/token_confuser_resolver/hparam_test/
# Does NOT touch:   runs/token_confuser_resolver/top256_boundary_v1
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME="$SLURM_SUBMIT_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD
mkdir -p logs

# ── Activate environment ─────────────────────────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

# ── Paths ────────────────────────────────────────────────────────────────────
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_CAND_DIR="runs/path_refiner_clean/data/train_hgrid_K24"
VAL_CAND_DIR="runs/path_refiner_clean/data/val_hgrid_K24"
TRAIN_FEAT_DIR="runs/path_refiner_residual_interface/features/train_multilayer"
VAL_FEAT_DIR="runs/path_refiner_residual_interface/features/val_multilayer"
BASELINE_JSON="runs/path_refiner_clean/baselines/saved_candidate_baseline.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
REGION_MAP="runs/region_maps_128/token_to_region.json"
OUT_ROOT="runs/token_confuser_resolver/hparam_test"

# ── Preflight checks ─────────────────────────────────────────────────────────
echo "========================================================"
echo " Token Confuser Resolver — Hyperparameter Stability Test"
echo " $(date)"
echo "========================================================"
echo ""
echo "[preflight] Checking required inputs..."

for CHECK_FILE in "$SMALL_CKPT" "$BASELINE_JSON" "$SUPER_MAP" "$REGION_MAP"; do
    if [ ! -f "$CHECK_FILE" ]; then
        echo "ERROR: required file not found: $CHECK_FILE"
        exit 1
    fi
    echo "  [OK] $CHECK_FILE"
done

for CHECK_DIR in "$TRAIN_CAND_DIR" "$VAL_CAND_DIR" "$TRAIN_FEAT_DIR" "$VAL_FEAT_DIR"; do
    if [ ! -d "$CHECK_DIR" ]; then
        echo "ERROR: required directory not found: $CHECK_DIR"
        exit 1
    fi
    N=$(find "$CHECK_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then
        echo "ERROR: no shard_*.pt files in $CHECK_DIR"
        exit 1
    fi
    echo "  [OK] $CHECK_DIR  ($N shards)"
done

echo ""
echo "[preflight] All required inputs present."
echo ""

mkdir -p "$OUT_ROOT"

# ── Check for optional --delta_scale support ─────────────────────────────────
if python scripts/train_token_confuser_resolver.py --help 2>/dev/null | grep -q -- "--delta_scale"; then
    HAS_DELTA_SCALE=1
    echo "[INFO] --delta_scale supported; will pass --delta_scale 0.25 to all runs."
else
    HAS_DELTA_SCALE=0
    echo "[INFO] --delta_scale not supported; running without tanh/clipped delta."
fi
echo ""

# ── Run function ─────────────────────────────────────────────────────────────
# Args: NAME OUT LR LAMBDA_RANK LAMBDA_KL LAMBDA_DELTA MARGIN GRAD_CLIP
run_config() {
    local NAME="$1"
    local OUT="$2"
    local LR="$3"
    local LAMBDA_RANK="$4"
    local LAMBDA_KL="$5"
    local LAMBDA_DELTA="$6"
    local MARGIN="$7"
    local GRAD_CLIP="$8"

    echo "========================================================"
    echo " Running config: $NAME"
    echo " output_dir: $OUT"
    echo " lr=$LR  rank=$LAMBDA_RANK  kl=$LAMBDA_KL  delta=$LAMBDA_DELTA"
    echo " margin=$MARGIN  grad_clip=$GRAD_CLIP"
    echo "========================================================"
    echo ""

    mkdir -p "$OUT"

    local EXTRA_ARGS=""
    if [ "$HAS_DELTA_SCALE" -eq 1 ]; then
        EXTRA_ARGS="--delta_scale 0.25"
    fi

    set +e
    python scripts/train_token_confuser_resolver.py \
        --small_ckpt         "$SMALL_CKPT"        \
        --train_cand_dir     "$TRAIN_CAND_DIR"    \
        --val_cand_dir       "$VAL_CAND_DIR"      \
        --train_feat_dir     "$TRAIN_FEAT_DIR"    \
        --val_feat_dir       "$VAL_FEAT_DIR"      \
        --baseline_json      "$BASELINE_JSON"     \
        --super_map          "$SUPER_MAP"         \
        --region_map         "$REGION_MAP"        \
        --confuser_source    base_topk            \
        --top_k              256                  \
        --train_filter       boundary             \
        --gate_filter        boundary             \
        --use_filtered_train_loader               \
        --resolver_dim       256                  \
        --resolver_layers    2                    \
        --resolver_heads     4                    \
        --batch_size         16                   \
        --grad_accum_steps   4                    \
        --eval_batch_size    64                   \
        --steps              2000                 \
        --eval_every         500                  \
        --amp                                     \
        --eval_before_train                       \
        --fail_on_baseline_mismatch               \
        --output_dir         "$OUT"               \
        --lr                 "$LR"                \
        --lambda_rank        "$LAMBDA_RANK"       \
        --lambda_kl          "$LAMBDA_KL"         \
        --lambda_delta       "$LAMBDA_DELTA"      \
        --rank_margin        "$MARGIN"            \
        --grad_clip          "$GRAD_CLIP"         \
        $EXTRA_ARGS
    STATUS=$?
    set -e

    echo "$STATUS" > "$OUT/run_status.txt"

    if [ "$STATUS" -ne 0 ]; then
        echo "  WARNING: $NAME exited with code $STATUS"
    else
        echo "  $NAME completed (exit 0)"
    fi

    if [ -f "$OUT/final_metrics.json" ]; then
        python -c "
import json, os
m  = json.load(open('$OUT/final_metrics.json'))
bm = json.load(open('$OUT/best_metrics.json')) if os.path.isfile('$OUT/best_metrics.json') else {}
print(f'  no_improving_checkpoint  = {m.get(\"no_improving_checkpoint\", \"?\")}')
print(f'  best_step                = {m.get(\"best_step\", \"?\")}')
print(f'  best_full_vocab_gain_all = {m.get(\"best_full_vocab_gain_all\", float(\"nan\")):.6f}')
print(f'  best_inside_gate_gain    = {m.get(\"best_inside_gate_gain\", float(\"nan\")):.6f}')
if bm:
    print(f'  changed_to_gold_rate     = {bm.get(\"changed_to_gold_rate\", float(\"nan\")):.4f}')
    print(f'  changed_away_rate        = {bm.get(\"changed_away_from_gold_rate\", float(\"nan\")):.4f}')
    print(f'  mean_delta_abs           = {bm.get(\"mean_delta_abs\", float(\"nan\")):.6f}')
"
    else
        echo "  WARNING: final_metrics.json not found for $NAME"
    fi
    echo ""
}

# ── Run A: stable_no_rank ────────────────────────────────────────────────────
run_config \
    "stable_no_rank"              \
    "$OUT_ROOT/top256_stable_no_rank" \
    "1e-5" "0.0" "1.0" "1e-3" "0.1" "0.5"

# ── Run B: gentle_rank005 ────────────────────────────────────────────────────
run_config \
    "gentle_rank005"              \
    "$OUT_ROOT/top256_gentle_rank005" \
    "1e-5" "0.05" "1.0" "1e-3" "0.05" "0.5"

# ── Run C: rank010 ───────────────────────────────────────────────────────────
run_config \
    "rank010"                     \
    "$OUT_ROOT/top256_rank010"    \
    "1e-5" "0.10" "1.0" "1e-3" "0.05" "0.5"

# ── Run D: extra_regularized ─────────────────────────────────────────────────
run_config \
    "extra_regularized"               \
    "$OUT_ROOT/top256_extra_regularized" \
    "5e-6" "0.05" "2.0" "3e-3" "0.05" "0.5"

# ── Generate summary ─────────────────────────────────────────────────────────
echo "========================================================"
echo " Generating hparam summary"
echo "========================================================"
echo ""

python << PYEOF
import csv
import json
import os

out_root = "$OUT_ROOT"

configs = [
    {
        "config":       "stable_no_rank",
        "output_dir":   os.path.join(out_root, "top256_stable_no_rank"),
        "lr":           "1e-5",
        "lambda_rank":  "0.0",
        "lambda_kl":    "1.0",
        "lambda_delta": "1e-3",
        "rank_margin":  "0.1",
        "grad_clip":    "0.5",
    },
    {
        "config":       "gentle_rank005",
        "output_dir":   os.path.join(out_root, "top256_gentle_rank005"),
        "lr":           "1e-5",
        "lambda_rank":  "0.05",
        "lambda_kl":    "1.0",
        "lambda_delta": "1e-3",
        "rank_margin":  "0.05",
        "grad_clip":    "0.5",
    },
    {
        "config":       "rank010",
        "output_dir":   os.path.join(out_root, "top256_rank010"),
        "lr":           "1e-5",
        "lambda_rank":  "0.10",
        "lambda_kl":    "1.0",
        "lambda_delta": "1e-3",
        "rank_margin":  "0.05",
        "grad_clip":    "0.5",
    },
    {
        "config":       "extra_regularized",
        "output_dir":   os.path.join(out_root, "top256_extra_regularized"),
        "lr":           "5e-6",
        "lambda_rank":  "0.05",
        "lambda_kl":    "2.0",
        "lambda_delta": "3e-3",
        "rank_margin":  "0.05",
        "grad_clip":    "0.5",
    },
]

rows = []
for cfg in configs:
    out = cfg["output_dir"]
    row = dict(cfg)

    # run_status
    st_path = os.path.join(out, "run_status.txt")
    row["run_status"] = open(st_path).read().strip() if os.path.isfile(st_path) else ""

    # final_metrics
    fm_path = os.path.join(out, "final_metrics.json")
    if os.path.isfile(fm_path):
        fm = json.load(open(fm_path))
        row["no_improving_checkpoint"]  = fm.get("no_improving_checkpoint", "")
        row["best_step"]                = fm.get("best_step", "")
        row["best_full_vocab_gain_all"] = fm.get("best_full_vocab_gain_all", "")
        row["best_inside_gate_gain"]    = fm.get("best_inside_gate_gain", "")
    else:
        for k in ("no_improving_checkpoint", "best_step",
                  "best_full_vocab_gain_all", "best_inside_gate_gain"):
            row[k] = ""

    # best_metrics
    bm_path = os.path.join(out, "best_metrics.json")
    bm_keys = ("gold_rank_improved_rate", "gold_rank_worsened_rate",
               "top1_acc_base", "top1_acc_ref",
               "changed_to_gold_rate", "changed_away_from_gold_rate",
               "mean_delta_abs", "alpha")
    if os.path.isfile(bm_path):
        bm = json.load(open(bm_path))
        for k in bm_keys:
            row[k] = bm.get(k, "")
    else:
        for k in bm_keys:
            row[k] = ""

    rows.append(row)

# ── CSV ───────────────────────────────────────────────────────────────────────
csv_fields = [
    "config", "output_dir", "run_status", "lr",
    "lambda_rank", "lambda_kl", "lambda_delta", "rank_margin", "grad_clip",
    "no_improving_checkpoint", "best_step",
    "best_full_vocab_gain_all", "best_inside_gate_gain",
    "gold_rank_improved_rate", "gold_rank_worsened_rate",
    "top1_acc_base", "top1_acc_ref",
    "changed_to_gold_rate", "changed_away_from_gold_rate",
    "mean_delta_abs", "alpha",
]
csv_path = os.path.join(out_root, "hparam_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"  CSV written: {csv_path}")

# ── Markdown ──────────────────────────────────────────────────────────────────
def _gain(r):
    try:
        return float(r.get("best_full_vocab_gain_all", ""))
    except (TypeError, ValueError):
        return float("-inf")

def _fmt(v, fmt=".6f"):
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "—" if v == "" else str(v)

sorted_rows = sorted(rows, key=_gain, reverse=True)

md_lines = [
    "# Token Confuser Resolver — Hyperparameter Stability Test",
    "",
    "Sorted by `best_full_vocab_gain_all` (descending). Missing values = run failed or no checkpoint.",
    "",
    "| config | lr | λ_rank | λ_kl | λ_δ | gain_all | ig_gain | step | to_gold | away_gold | Δ_abs | no_ckpt |",
    "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for r in sorted_rows:
    md_lines.append(
        f"| {r['config']} "
        f"| {r['lr']} "
        f"| {r['lambda_rank']} "
        f"| {r['lambda_kl']} "
        f"| {r['lambda_delta']} "
        f"| {_fmt(r.get('best_full_vocab_gain_all', ''))} "
        f"| {_fmt(r.get('best_inside_gate_gain', ''))} "
        f"| {r.get('best_step', '—')} "
        f"| {_fmt(r.get('changed_to_gold_rate', ''), '.4f')} "
        f"| {_fmt(r.get('changed_away_from_gold_rate', ''), '.4f')} "
        f"| {_fmt(r.get('mean_delta_abs', ''), '.4f')} "
        f"| {r.get('no_improving_checkpoint', '—')} |"
    )

md_path = os.path.join(out_root, "hparam_summary.md")
with open(md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"  MD written:  {md_path}")
PYEOF

# ── Interpretation ────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Interpretation Guide"
echo "========================================================"
echo ""
echo "Good signs:"
echo "  full_vocab_gain_all          >= 0"
echo "  inside_gate_gain_all         >= 0"
echo "  changed_to_gold_rate         >= changed_away_from_gold_rate"
echo "  top1_acc_ref                 >= top1_acc_base"
echo "  gold_rank_improved_rate       > gold_rank_worsened_rate"
echo "  mean_delta_abs               controlled (not exploding)"
echo ""
echo "Bad signs:"
echo "  full_vocab_gain_all          strongly negative"
echo "  inside_gate_gain_all         negative"
echo "  changed_away_from_gold_rate  > changed_to_gold_rate"
echo "  top1_acc_ref                 < top1_acc_base"
echo "  mean_delta_abs               too large"
echo ""
echo "This is a stability test, not final optimization."
echo "If all four are negative, the current Token Confuser architecture/features"
echo "are likely flawed or need delta clipping / better token evidence."
echo ""
echo "Summary files:"
echo "  $OUT_ROOT/hparam_summary.csv"
echo "  $OUT_ROOT/hparam_summary.md"
echo ""
echo "========================================================"
echo " Hyperparameter test complete."
echo " $(date)"
echo "========================================================"
echo ""
