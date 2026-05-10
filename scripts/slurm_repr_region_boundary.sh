#!/bin/bash
#SBATCH --job-name=repr_rgn_bnd
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# 6 targeted runs (20k steps each) testing the boundary-conditioning hypothesis:
#
#   Run 1: repr_region reference         (existing best, alpha=0.2)
#   Run 2: repr_region_boundary a=0.4    (boundary boost alpha_bnd=0.4, tau=0.10)
#   Run 3: repr_region_boundary a=0.6    (stronger boundary boost, alpha_bnd=0.6)
#   Run 4: repr_region_boundary tau=0.2  (wider boundary region, tau=0.20)
#   Run 5: random_repr_region_boundary   (null hypothesis, same config as Run 2)
#   Run 6: oracle_repr_region_boundary   (upper bound, same config as Run 2)
#
# Hypothesis: routing failures are localised to low-margin (boundary) tokens.
# Adaptive alpha that boosts conditioning exactly there should improve
# boundary_val_lm while leaving core_val_lm stable.
#
# Success criteria:
#   boundary_val_lm improves vs reference    (boundary tokens benefit)
#   core_val_lm stays flat                   (core tokens unaffected)
#   random control does NOT match            (structure matters, not just scale)
#   oracle run remains significantly better  (ceiling is reachable)

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
REGION_MAP=runs/region_maps_128/token_to_region.json

if [[ ! -f "$REGION_MAP" ]]; then
    echo "ERROR: region map missing: $REGION_MAP" >&2
    exit 1
fi
if [[ ! -f "train_region_lm.py" ]]; then
    echo "ERROR: train_region_lm.py not found in $PWD" >&2
    exit 1
fi

mkdir -p logs

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

# ── Shared base args ───────────────────────────────────────────────────────────
BASE_ARGS=(
    --dataset             wikitext-103-raw-v1
    --vocab_subset_size   50257
    --seq_len             256
    --batch_size          32
    --n_layer             6
    --d_model             384
    --n_head              6
    --d_ff                1536
    --steps               20000
    --lr                  3e-4
    --warmup_steps        1000
    --lambda_balance      0.001
    --eval_interval       500
    --save_interval       2000
    --log_interval        100
    --seed                42
    --device              cuda
    --region_map_path     "$REGION_MAP"
    --router_type         mlp
    --lambda_coarse       0.2
    --router_temp         2.0
    --region_warmup_steps 5000
)

# Shared boundary args (Runs 2–6)
BND_ARGS=(
    --boundary_tau   0.10
    --boundary_temp  0.05
    --boundary_mode  margin
    --alpha_core     0.2
)

run_if_needed() {
    local out_dir="$1"
    shift
    local summary="${out_dir}/final_summary.json"
    if [[ -f "$summary" && "${FORCE_RUN:-0}" != "1" ]]; then
        echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
        return 0
    fi
    mkdir -p "$out_dir"
    python train_region_lm.py --output_dir "$out_dir" "$@"
}

# ══════════════════════════════════════════════════════════════════════════════
# Run 1: repr_region reference  (reuse if already done)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 1/6: repr_region reference  (alpha=0.2)"
echo "  output_dir: runs/repr_region_reference"
echo "=============================="
run_if_needed runs/repr_region_reference \
    --mode       repr_region \
    --base_alpha 0.2 \
    "${BASE_ARGS[@]}"
echo "  Run 1 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 2: boundary boost  (alpha_bnd=0.4, tau=0.10)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 2/6: repr_region_boundary  (alpha_core=0.2, alpha_bnd=0.4, tau=0.10)"
echo "  output_dir: runs/repr_region_boundary_a0p4"
echo "=============================="
run_if_needed runs/repr_region_boundary_a0p4 \
    --mode           repr_region_boundary \
    --alpha_boundary 0.4 \
    "${BND_ARGS[@]}" "${BASE_ARGS[@]}"
echo "  Run 2 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 3: stronger boundary boost  (alpha_bnd=0.6, tau=0.10)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 3/6: repr_region_boundary  (alpha_core=0.2, alpha_bnd=0.6, tau=0.10)"
echo "  output_dir: runs/repr_region_boundary_a0p6"
echo "=============================="
run_if_needed runs/repr_region_boundary_a0p6 \
    --mode           repr_region_boundary \
    --alpha_boundary 0.6 \
    "${BND_ARGS[@]}" "${BASE_ARGS[@]}"
echo "  Run 3 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 4: wider boundary region  (alpha_bnd=0.4, tau=0.20)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 4/6: repr_region_boundary  (alpha_core=0.2, alpha_bnd=0.4, tau=0.20)"
echo "  output_dir: runs/repr_region_boundary_tau0p2"
echo "=============================="
run_if_needed runs/repr_region_boundary_tau0p2 \
    --mode           repr_region_boundary \
    --alpha_boundary 0.4 \
    --boundary_tau   0.20 \
    --boundary_temp  0.05 \
    --boundary_mode  margin \
    --alpha_core     0.2 \
    "${BASE_ARGS[@]}"
echo "  Run 4 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 5: random_repr_region_boundary  (null hypothesis)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 5/6: random_repr_region_boundary  (null hypothesis, same config as Run 2)"
echo "  output_dir: runs/random_repr_region_boundary_a0p4"
echo "=============================="
run_if_needed runs/random_repr_region_boundary_a0p4 \
    --mode           random_repr_region_boundary \
    --alpha_boundary 0.4 \
    "${BND_ARGS[@]}" "${BASE_ARGS[@]}"
echo "  Run 5 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 6: oracle_repr_region_boundary  (upper bound)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 6/6: oracle_repr_region_boundary  (upper bound, same config as Run 2)"
echo "  output_dir: runs/oracle_repr_region_boundary_a0p4"
echo "=============================="
run_if_needed runs/oracle_repr_region_boundary_a0p4 \
    --mode           oracle_repr_region_boundary \
    --alpha_boundary 0.4 \
    "${BND_ARGS[@]}" "${BASE_ARGS[@]}"
echo "  Run 6 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Results"
echo "=============================="

python3 - <<'PYEOF'
import json, os, math

RUNS = [
    ("repr_region_reference",              "repr_region",                  "reference"),
    ("repr_region_boundary_a0p4",          "repr_region_boundary",         "bnd a=0.4"),
    ("repr_region_boundary_a0p6",          "repr_region_boundary",         "bnd a=0.6"),
    ("repr_region_boundary_tau0p2",        "repr_region_boundary",         "tau=0.20"),
    ("random_repr_region_boundary_a0p4",   "random_repr_region_boundary",  "random"),
    ("oracle_repr_region_boundary_a0p4",   "oracle_repr_region_boundary",  "oracle"),
]
BASELINE_F = "runs/baseline/final_summary.json"

baseline_loss = None
if os.path.isfile(BASELINE_F):
    with open(BASELINE_F) as fh:
        bd = json.load(fh)
    baseline_loss = bd.get("val_lm_loss")
    print(f"  {'baseline':44s}  val_lm={baseline_loss:.4f}  ppl={math.exp(min(baseline_loss,20)):.2f}")
print()

results = []
for run_dir, mode, label in RUNS:
    f = f"runs/{run_dir}/final_summary.json"
    if not os.path.isfile(f):
        print(f"  [{label:12s}] {run_dir:<44s}  (no results yet)")
        continue
    with open(f) as fh:
        d = json.load(fh)
    val_lm   = d.get("val_lm_loss", float("nan"))
    val_ppl  = d.get("val_ppl",     float("nan"))
    acc1     = d.get("val_coarse_acc1", 0.0)
    bnd_lm   = d.get("val_boundary_lm", float("nan"))
    core_lm  = d.get("val_core_lm",     float("nan"))
    bnd_frac = d.get("val_boundary_frac", 0.0)
    bnd_alph = d.get("val_avg_boundary_alpha", 0.0)
    delta    = (val_lm - baseline_loss) if (baseline_loss and not math.isnan(val_lm)) else None
    delta_s  = f"{delta:+.4f}" if delta is not None else "—"
    print(
        f"  [{label:12s}] {run_dir:<44s}  val_lm={val_lm:.4f}  ppl={val_ppl:.2f}"
        f"  acc@1={acc1:.3f}  bnd_lm={bnd_lm:.4f}  core_lm={core_lm:.4f}"
        f"  bnd_frac={bnd_frac:.2f}  α_bnd={bnd_alph:.3f}  Δ={delta_s}"
    )
    results.append((val_lm, mode, label, run_dir, bnd_lm, core_lm))

print()
print("── Boundary Hypothesis Verdict ──")

ref_rows   = [r for r in results if r[2] == "reference"]
bnd_rows   = [r for r in results if "bnd" in r[2] or "tau" in r[2]]
rand_rows  = [r for r in results if "random" in r[1]]
orac_rows  = [r for r in results if "oracle" in r[1]]

ref_lm     = ref_rows[0][0]  if ref_rows else None
ref_bnd_lm = ref_rows[0][4]  if ref_rows else None
ref_core_lm= ref_rows[0][5]  if ref_rows else None

if bnd_rows and ref_lm is not None:
    best_bnd = min(bnd_rows, key=lambda r: r[0])
    if best_bnd[0] < ref_lm:
        print(f"  PASS  Best boundary run ({best_bnd[2]}) beats reference ({best_bnd[0]:.4f} < {ref_lm:.4f})")
    else:
        print(f"  FAIL  Best boundary run does not beat reference ({best_bnd[0]:.4f} >= {ref_lm:.4f})")

if bnd_rows and ref_bnd_lm is not None:
    best_bnd = min(bnd_rows, key=lambda r: r[4] if not math.isnan(r[4]) else 99)
    if not math.isnan(best_bnd[4]) and best_bnd[4] < ref_bnd_lm:
        print(f"  PASS  Boundary-token NLL improved: {ref_bnd_lm:.4f} → {best_bnd[4]:.4f}")
    else:
        print(f"  INFO  Boundary-token NLL not improved (ref={ref_bnd_lm:.4f})")

if bnd_rows and ref_core_lm is not None:
    best_bnd = min(bnd_rows, key=lambda r: r[0])
    if not math.isnan(best_bnd[5]):
        drift = best_bnd[5] - ref_core_lm
        if abs(drift) < 0.005:
            print(f"  PASS  Core-token NLL stable (drift={drift:+.4f})")
        elif drift > 0:
            print(f"  WARN  Core-token NLL regressed by {drift:+.4f}")
        else:
            print(f"  INFO  Core-token NLL also improved by {drift:+.4f}")

if bnd_rows and rand_rows:
    if min(r[0] for r in bnd_rows) < min(r[0] for r in rand_rows):
        print(f"  PASS  Best boundary run beats random control (structure matters)")
    else:
        print(f"  FAIL  Boundary run does not beat random control")

if orac_rows and bnd_rows:
    headroom = min(r[0] for r in bnd_rows) - orac_rows[0][0]
    print(f"  INFO  Oracle headroom vs best boundary run: {headroom:+.4f}")

PYEOF

echo ""
echo "Done at $(date)"
