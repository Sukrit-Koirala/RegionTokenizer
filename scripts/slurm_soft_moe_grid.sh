#!/bin/bash
#SBATCH --job-name=soft_moe_grid
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# NOTE: 72 grid + 6 oracle/random = 78 screening runs (5k steps each).
# If 24h walltime is tight, submit three separate jobs by K:
#   sbatch --export=RUN_K=32  scripts/slurm_soft_moe_grid.sh
#   sbatch --export=RUN_K=64  scripts/slurm_soft_moe_grid.sh
#   sbatch --export=RUN_K=128 scripts/slurm_soft_moe_grid.sh
# Each job will only build maps and run training for that K.

source ~/miniconda3/bin/activate
conda activate learned_regions

set -eo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME="$SLURM_SUBMIT_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD

# ── Safety checks ──────────────────────────────────────────────────────────────
BASELINE_CKPT=runs/baseline/checkpoint_latest.pt
if [[ ! -f "$BASELINE_CKPT" ]]; then
    echo "ERROR: baseline checkpoint not found: $BASELINE_CKPT" >&2
    exit 1
fi
if [[ ! -f "train_region_lm.py" ]]; then
    echo "ERROR: train_region_lm.py not found in $PWD" >&2
    exit 1
fi

mkdir -p logs models/hf_cache runs/grid_soft_moe

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

# ── Grid axes ──────────────────────────────────────────────────────────────────
REGION_COUNTS=(32 64 128)
ROUTER_TYPES=(linear mlp)
LAMBDAS=(0.05 0.2)
GAMMAS=(0.25 0.5 1.0)
TEMPS=(1.0 2.0)

# ── Shared training args ───────────────────────────────────────────────────────
# Note: --region_map_path and --lambda_coarse are NOT here; they are swept per run.
COMMON_ARGS=(
    --vocab_subset_size   50257
    --dataset             wikitext-103-raw-v1
    --seq_len             256
    --batch_size          32
    --n_layer             6
    --d_model             384
    --n_head              6
    --d_ff                1536
    --steps               5000
    --lr                  3e-4
    --warmup_steps        1000
    --lambda_balance      0.001
    --eval_interval       500
    --save_interval       5000
    --log_interval        100
    --seed                42
    --device              cuda
)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Build region maps for each K
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "PHASE 0: Build region maps"
echo "=============================="

for K in "${REGION_COUNTS[@]}"; do
    [[ -n "$RUN_K" && "$RUN_K" != "$K" ]] && continue

    REGION_MAP_DIR=runs/region_maps_${K}
    REGION_MAP=${REGION_MAP_DIR}/token_to_region.json

    mkdir -p "$REGION_MAP_DIR"

    if [[ -f "$REGION_MAP" && "${FORCE_BUILD:-0}" != "1" ]]; then
        echo "  K=$K: region map exists, skipping (FORCE_BUILD=1 to rebuild)"
    else
        echo "  K=$K: building spectral region map at $(date) ..."
        python train_region_lm.py \
            --mode                  build_regions \
            --dataset               wikitext-103-raw-v1 \
            --vocab_subset_size     50257 \
            --batch_size            64 \
            --baseline_ckpt         "$BASELINE_CKPT" \
            --region_output_dir     "$REGION_MAP_DIR" \
            --region_vocab_size     20000 \
            --region_top_k          50 \
            --region_max_tokens     1000000 \
            --region_cluster_method spectral \
            --target_n_regions      "$K" \
            --target_n_leaves       512 \
            --leaf_min_size         50 \
            --seed                  42 \
            --device                cuda
        echo "  K=$K: build_regions done at $(date)"
    fi

    echo "  K=$K region stats:"
    STATS_FILE="${REGION_MAP_DIR}/region_stats.json"
    python3 - <<PYEOF
import json
s = json.load(open('${STATS_FILE}'))
print(f"    actual={s['actual_n_regions']}  median={s['median_size']:.1f}  "
      f"min={s['min_size']}  max={s['max_size']}  coverage={s['coverage']:.1%}")
PYEOF
    echo ""
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Grid screening runs  (soft_moe only, no top-k)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "PHASE 1: Grid screening  (soft_moe, 5k steps each)"
echo "=============================="

TOTAL_GRID=$(( ${#REGION_COUNTS[@]} * ${#ROUTER_TYPES[@]} * ${#LAMBDAS[@]} * ${#GAMMAS[@]} * ${#TEMPS[@]} ))
RUN_IDX=0

for K in "${REGION_COUNTS[@]}"; do
    [[ -n "$RUN_K" && "$RUN_K" != "$K" ]] && continue

    REGION_MAP=runs/region_maps_${K}/token_to_region.json
    if [[ ! -f "$REGION_MAP" ]]; then
        echo "ERROR: region map missing for K=$K: $REGION_MAP" >&2
        exit 1
    fi

    for ROUTER_TYPE in "${ROUTER_TYPES[@]}"; do
        for LAMBDA in "${LAMBDAS[@]}"; do
            for GAMMA in "${GAMMAS[@]}"; do
                for TEMP in "${TEMPS[@]}"; do
                    RUN_IDX=$(( RUN_IDX + 1 ))

                    # sanitize dots: 0.05 -> 0p05
                    LAMBDA_S="${LAMBDA//./p}"
                    GAMMA_S="${GAMMA//./p}"
                    TEMP_S="${TEMP//./p}"

                    RUN_NAME="K${K}_${ROUTER_TYPE}_lam${LAMBDA_S}_gamma${GAMMA_S}_temp${TEMP_S}"
                    OUT_DIR="runs/grid_soft_moe/${RUN_NAME}"
                    SUMMARY="${OUT_DIR}/final_summary.json"

                    echo "------------------------------"
                    echo "Run ${RUN_IDX}/${TOTAL_GRID}: ${RUN_NAME}"
                    echo "  K=${K}  router_type=${ROUTER_TYPE}  lambda_coarse=${LAMBDA}"
                    echo "  prior_gamma=${GAMMA}  router_temp=${TEMP}"
                    echo "  output_dir: ${OUT_DIR}"
                    echo "------------------------------"

                    if [[ -f "$SUMMARY" && "${FORCE_RUN:-0}" != "1" ]]; then
                        echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
                        echo ""
                        continue
                    fi

                    mkdir -p "$OUT_DIR"

                    python train_region_lm.py \
                        --mode            soft_moe \
                        --output_dir      "$OUT_DIR" \
                        --region_map_path "$REGION_MAP" \
                        --router_type     "$ROUTER_TYPE" \
                        --lambda_coarse   "$LAMBDA" \
                        --prior_gamma     "$GAMMA" \
                        --router_temp     "$TEMP" \
                        "${COMMON_ARGS[@]}"

                    echo "  ${RUN_NAME} done at $(date)"
                    echo ""
                done
            done
        done
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Oracle + random checks per K
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "PHASE 2: Oracle and random checks per K"
echo "=============================="

for K in "${REGION_COUNTS[@]}"; do
    [[ -n "$RUN_K" && "$RUN_K" != "$K" ]] && continue

    REGION_MAP=runs/region_maps_${K}/token_to_region.json

    # ── oracle: upper bound for this K ────────────────────────────────────────
    ORACLE_DIR="runs/grid_soft_moe/K${K}_oracle_gamma0p5"
    echo "--- Oracle  K=${K}  output_dir: ${ORACLE_DIR} ---"
    if [[ -f "${ORACLE_DIR}/final_summary.json" && "${FORCE_RUN:-0}" != "1" ]]; then
        echo "  Already done — skipping"
    else
        mkdir -p "$ORACLE_DIR"
        python train_region_lm.py \
            --mode            oracle_soft_moe \
            --output_dir      "$ORACLE_DIR" \
            --region_map_path "$REGION_MAP" \
            --prior_gamma     0.5 \
            "${COMMON_ARGS[@]}"
        echo "  oracle K=${K} done at $(date)"
    fi
    echo ""

    # ── random: null hypothesis for this K ───────────────────────────────────
    RANDOM_DIR="runs/grid_soft_moe/K${K}_random_mlp_lam0p2_gamma0p5_temp2p0"
    echo "--- Random  K=${K}  output_dir: ${RANDOM_DIR} ---"
    if [[ -f "${RANDOM_DIR}/final_summary.json" && "${FORCE_RUN:-0}" != "1" ]]; then
        echo "  Already done — skipping"
    else
        mkdir -p "$RANDOM_DIR"
        python train_region_lm.py \
            --mode            random_soft_moe \
            --output_dir      "$RANDOM_DIR" \
            --region_map_path "$REGION_MAP" \
            --router_type     mlp \
            --lambda_coarse   0.2 \
            --prior_gamma     0.5 \
            --router_temp     2.0 \
            "${COMMON_ARGS[@]}"
        echo "  random K=${K} done at $(date)"
    fi
    echo ""
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Aggregate summary: CSV + Markdown + console report
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "PHASE 3: Generating grid summary"
echo "=============================="

python3 - <<'PYEOF'
import json, os, csv, re, math

GRID_DIR   = "runs/grid_soft_moe"
BASELINE_F = "runs/baseline/final_summary.json"
CSV_OUT    = f"{GRID_DIR}/grid_summary.csv"
MD_OUT     = f"{GRID_DIR}/grid_summary.md"

# ── baseline ───────────────────────────────────────────────────────────────
baseline_loss = None
if os.path.isfile(BASELINE_F):
    with open(BASELINE_F) as fh:
        bd = json.load(fh)
    baseline_loss = bd.get("val_lm_loss")

# ── collect runs ───────────────────────────────────────────────────────────
FIELDS = [
    "run_name", "mode", "K", "router_type", "lambda_coarse",
    "prior_gamma", "router_temp",
    "val_lm_loss", "val_ppl",
    "val_coarse_acc1", "val_coarse_acc4", "val_coarse_ent",
    "val_coarse_coverage", "n_params", "delta_vs_baseline",
]

rows = []
for run_name in sorted(os.listdir(GRID_DIR)):
    summary_f = os.path.join(GRID_DIR, run_name, "final_summary.json")
    if not os.path.isfile(summary_f):
        continue
    with open(summary_f) as fh:
        d = json.load(fh)

    m = re.match(r"K(\d+)_", run_name)
    K = int(m.group(1)) if m else None

    val_lm = d.get("val_lm_loss")
    delta  = (val_lm - baseline_loss) if (val_lm is not None and baseline_loss is not None) else None

    rows.append({
        "run_name":            run_name,
        "mode":                d.get("mode", ""),
        "K":                   K,
        "router_type":         d.get("router_type", ""),
        "lambda_coarse":       d.get("lambda_coarse", ""),
        "prior_gamma":         d.get("prior_gamma", ""),
        "router_temp":         d.get("router_temp", ""),
        "val_lm_loss":         val_lm,
        "val_ppl":             d.get("val_ppl"),
        "val_coarse_acc1":     d.get("val_coarse_acc1"),
        "val_coarse_acc4":     d.get("val_coarse_acc4"),
        "val_coarse_ent":      d.get("val_coarse_ent"),
        "val_coarse_coverage": d.get("val_coarse_coverage"),
        "n_params":            d.get("n_params"),
        "delta_vs_baseline":   delta,
    })

rows.sort(key=lambda r: (r["val_lm_loss"] is None, r["val_lm_loss"] or math.inf))

# ── CSV ────────────────────────────────────────────────────────────────────
with open(CSV_OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {CSV_OUT}  ({len(rows)} runs)")

# ── Markdown ───────────────────────────────────────────────────────────────
def fmt(v, spec=""):
    if v is None:
        return "—"
    try:
        if spec == ".4f":   return f"{v:.4f}"
        if spec == ".2f":   return f"{v:.2f}"
        if spec == ".1%":   return f"{v:.1%}"
        if spec == "+.4f":  return f"{v:+.4f}"
    except (TypeError, ValueError):
        pass
    return str(v)

md_header = (
    "| run_name | mode | K | router | λ_coarse | γ | temp"
    " | val_lm | ppl | acc@1 | acc@4 | cov | Δbaseline |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)

with open(MD_OUT, "w") as fh:
    fh.write("# Soft-MoE Grid Search Summary\n\n")
    if baseline_loss is not None:
        fh.write(f"**Baseline val_lm_loss:** {baseline_loss:.4f}\n\n")
    fh.write(md_header)
    for r in rows:
        fh.write(
            f"| {r['run_name']} | {r['mode']} | {r['K']} | {r['router_type']}"
            f" | {fmt(r['lambda_coarse'])} | {fmt(r['prior_gamma'])} | {fmt(r['router_temp'])}"
            f" | {fmt(r['val_lm_loss'],'.4f')} | {fmt(r['val_ppl'],'.2f')}"
            f" | {fmt(r['val_coarse_acc1'],'.4f')} | {fmt(r['val_coarse_acc4'],'.4f')}"
            f" | {fmt(r['val_coarse_coverage'],'.1%')} | {fmt(r['delta_vs_baseline'],'+.4f')} |\n"
        )
print(f"Wrote {MD_OUT}")

# ── console: top 10 ────────────────────────────────────────────────────────
print("\n══ Top 10 configs by val_lm_loss ══")
print(f"{'':>4}  {'run_name':<52}  {'val_lm':>8}  {'ppl':>7}  {'Δbaseline':>10}")
print("─" * 88)
for i, r in enumerate(rows[:10], 1):
    print(f"  {i:2d}.  {r['run_name']:<52}  {fmt(r['val_lm_loss'],'.4f'):>8}"
          f"  {fmt(r['val_ppl'],'.2f'):>7}  {fmt(r['delta_vs_baseline'],'+.4f'):>10}")

# ── best per category ──────────────────────────────────────────────────────
def best_of(mode_str):
    subset = [r for r in rows if r["mode"] == mode_str and r["val_lm_loss"] is not None]
    return subset[0] if subset else None

best_real   = best_of("soft_moe")
best_random = best_of("random_soft_moe")
best_oracle = best_of("oracle_soft_moe")

def print_best(label, r):
    if r is None:
        print(f"  {label:<22}  (no results)")
        return
    print(f"  {label:<22}  val_lm={fmt(r['val_lm_loss'],'.4f')}"
          f"  ppl={fmt(r['val_ppl'],'.2f')}"
          f"  Δbaseline={fmt(r['delta_vs_baseline'],'+.4f')}"
          f"  [{r['run_name']}]")

print("\n══ Best per category ══")
bl_str = f"{baseline_loss:.4f}" if baseline_loss is not None else "—"
print(f"  {'Baseline val_lm:':<22}  {bl_str}")
print_best("Best soft_moe:",   best_real)
print_best("Best random_moe:", best_random)
print_best("Best oracle_moe:", best_oracle)

# ── verdict ────────────────────────────────────────────────────────────────
print("\n══ Verdict ══")
if best_real and baseline_loss is not None:
    if best_real["val_lm_loss"] < baseline_loss:
        print("  PASS  Best soft_moe beats baseline")
    else:
        print("  FAIL  Best soft_moe does not beat baseline")

if best_real and best_random and best_real["val_lm_loss"] is not None and best_random["val_lm_loss"] is not None:
    if best_real["val_lm_loss"] < best_random["val_lm_loss"]:
        print("  PASS  Best soft_moe beats random routing")
    else:
        print("  FAIL  Best soft_moe does not beat random routing  (routing bottleneck still present)")

if best_oracle and best_real and best_oracle["val_lm_loss"] is not None and best_real["val_lm_loss"] is not None:
    headroom = best_real["val_lm_loss"] - best_oracle["val_lm_loss"]
    print(f"  INFO  Oracle headroom vs best soft_moe: {headroom:+.4f}")
PYEOF

echo ""
echo "Done at $(date)"
