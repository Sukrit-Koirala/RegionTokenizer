#!/bin/bash
#SBATCH --job-name=bnd_gate_sweep
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
# Boundary gate sharpness sweep — 6 runs × 20k steps.
#
# Context: prior multihyp run (boundary_frac≈0.83) failed because the gate
# activated too broadly, making the boundary path non-special and disrupting
# normal computation.  This sweep finds a clean, selective gate
# (target boundary_frac ≈ 0.20–0.40) BEFORE re-running multihyp.
#
# Only tau and temp vary.  Everything else is fixed at the successful
# repr_region_reference config (alpha_core=0.2, alpha_boundary=0.4).
#
# Run matrix:
#   Run 1: tau=0.03  temp=0.01
#   Run 2: tau=0.03  temp=0.02
#   Run 3: tau=0.05  temp=0.01
#   Run 4: tau=0.05  temp=0.02
#   Run 5: tau=0.07  temp=0.02
#   Run 6: tau=0.07  temp=0.03
#
# Success criteria (per run):
#   boundary_frac  ≈ 0.20 – 0.40
#   val_lm         ≤ 3.9978   (reference)
#   boundary_lm    improved vs reference
#   core_lm        stable

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

# ── Shared base args (frozen — identical to repr_region_reference) ─────────────
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
    --lambda_coarse       0.2
    --eval_interval       500
    --save_interval       2000
    --log_interval        100
    --seed                42
    --device              cuda
    --region_map_path     "$REGION_MAP"
    --router_type         mlp
    --router_temp         2.0
    --region_warmup_steps 5000
    --mode                repr_region_boundary
    --alpha_core          0.2
    --alpha_boundary      0.4
    --boundary_mode       margin
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
# Run 1: tau=0.03  temp=0.01  (sharpest gate)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 1/6: tau=0.03  temp=0.01"
echo "  output_dir: runs/boundary_tau0p03_temp0p01"
echo "=============================="
run_if_needed runs/boundary_tau0p03_temp0p01 \
    --boundary_tau  0.03 \
    --boundary_temp 0.01 \
    "${BASE_ARGS[@]}"
echo "  Run 1 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 2: tau=0.03  temp=0.02
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 2/6: tau=0.03  temp=0.02"
echo "  output_dir: runs/boundary_tau0p03_temp0p02"
echo "=============================="
run_if_needed runs/boundary_tau0p03_temp0p02 \
    --boundary_tau  0.03 \
    --boundary_temp 0.02 \
    "${BASE_ARGS[@]}"
echo "  Run 2 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 3: tau=0.05  temp=0.01
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 3/6: tau=0.05  temp=0.01"
echo "  output_dir: runs/boundary_tau0p05_temp0p01"
echo "=============================="
run_if_needed runs/boundary_tau0p05_temp0p01 \
    --boundary_tau  0.05 \
    --boundary_temp 0.01 \
    "${BASE_ARGS[@]}"
echo "  Run 3 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 4: tau=0.05  temp=0.02
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 4/6: tau=0.05  temp=0.02"
echo "  output_dir: runs/boundary_tau0p05_temp0p02"
echo "=============================="
run_if_needed runs/boundary_tau0p05_temp0p02 \
    --boundary_tau  0.05 \
    --boundary_temp 0.02 \
    "${BASE_ARGS[@]}"
echo "  Run 4 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 5: tau=0.07  temp=0.02
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 5/6: tau=0.07  temp=0.02"
echo "  output_dir: runs/boundary_tau0p07_temp0p02"
echo "=============================="
run_if_needed runs/boundary_tau0p07_temp0p02 \
    --boundary_tau  0.07 \
    --boundary_temp 0.02 \
    "${BASE_ARGS[@]}"
echo "  Run 5 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 6: tau=0.07  temp=0.03
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 6/6: tau=0.07  temp=0.03"
echo "  output_dir: runs/boundary_tau0p07_temp0p03"
echo "=============================="
run_if_needed runs/boundary_tau0p07_temp0p03 \
    --boundary_tau  0.07 \
    --boundary_temp 0.03 \
    "${BASE_ARGS[@]}"
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

REFERENCE_VAL_LM = 3.9978

RUNS = [
    ("boundary_tau0p03_temp0p01", "tau=0.03 temp=0.01"),
    ("boundary_tau0p03_temp0p02", "tau=0.03 temp=0.02"),
    ("boundary_tau0p05_temp0p01", "tau=0.05 temp=0.01"),
    ("boundary_tau0p05_temp0p02", "tau=0.05 temp=0.02"),
    ("boundary_tau0p07_temp0p02", "tau=0.07 temp=0.02"),
    ("boundary_tau0p07_temp0p03", "tau=0.07 temp=0.03"),
]

# Also pull reference from disk if available (overrides hard-coded value)
ref_f = "runs/repr_region_reference/final_summary.json"
ref_val_lm = REFERENCE_VAL_LM
if os.path.isfile(ref_f):
    with open(ref_f) as fh:
        rd = json.load(fh)
    ref_val_lm = rd.get("val_lm_loss", REFERENCE_VAL_LM)
    ref_bnd_lm  = rd.get("val_boundary_lm", float("nan"))
    ref_core_lm = rd.get("val_core_lm",     float("nan"))
    ref_bnd_frac= rd.get("val_boundary_frac", float("nan"))
    print(f"  {'repr_region_reference':<34}  "
          f"val_lm={ref_val_lm:.4f}  ppl={math.exp(min(ref_val_lm,20)):.2f}  "
          f"bnd_frac={ref_bnd_frac:.2f}  "
          f"bnd_lm={ref_bnd_lm:.4f}  core_lm={ref_core_lm:.4f}  "
          f"α_bnd=—  α_core=—  Δ=0.0000  [REFERENCE]")
else:
    print(f"  {'repr_region_reference':<34}  val_lm={ref_val_lm:.4f}  [REFERENCE, hard-coded]")
    ref_bnd_lm  = float("nan")
    ref_core_lm = float("nan")
    ref_bnd_frac= float("nan")
print()

# ── Header ─────────────────────────────────────────────────────────────────────
print(f"  {'run':<34}  {'val_lm':>7}  {'ppl':>7}  {'bnd_frac':>8}  "
      f"{'bnd_lm':>7}  {'core_lm':>7}  {'α_bnd':>6}  {'α_core':>6}  {'Δ':>8}")
print(f"  {'-'*34}  {'-'*7}  {'-'*7}  {'-'*8}  "
      f"{'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*8}")

results = []
for run_dir, label in RUNS:
    f = f"runs/{run_dir}/final_summary.json"
    if not os.path.isfile(f):
        print(f"  [{label:<32}]  (no results yet)")
        continue
    with open(f) as fh:
        d = json.load(fh)
    val_lm    = d.get("val_lm_loss",           float("nan"))
    val_ppl   = d.get("val_ppl",               float("nan"))
    bnd_frac  = d.get("val_boundary_frac",     float("nan"))
    bnd_lm    = d.get("val_boundary_lm",       float("nan"))
    core_lm   = d.get("val_core_lm",           float("nan"))
    alpha_bnd = d.get("val_avg_boundary_alpha", float("nan"))
    alpha_cor = d.get("val_avg_core_alpha",     float("nan"))
    delta     = val_lm - ref_val_lm if not math.isnan(val_lm) else float("nan")
    delta_s   = f"{delta:+.4f}" if not math.isnan(delta) else "—"

    # Flag whether boundary_frac is in the target range
    in_range = ""
    if not math.isnan(bnd_frac):
        if 0.20 <= bnd_frac <= 0.40:
            in_range = " ✓"
        elif bnd_frac > 0.60:
            in_range = " !"  # too broad

    print(
        f"  {label:<34}  "
        f"{val_lm:>7.4f}  {val_ppl:>7.2f}  {bnd_frac:>8.2f}{in_range:<2}  "
        f"{bnd_lm:>7.4f}  {core_lm:>7.4f}  "
        f"{alpha_bnd:>6.3f}  {alpha_cor:>6.3f}  {delta_s:>8}"
    )
    results.append({
        "label": label, "run_dir": run_dir,
        "val_lm": val_lm, "bnd_frac": bnd_frac,
        "bnd_lm": bnd_lm, "core_lm": core_lm,
        "delta": delta,
    })

# ── Gate sweep verdict ─────────────────────────────────────────────────────────
print()
print("── Gate Sharpness Verdict ──")

if not results:
    print("  (no completed runs)")
else:
    # Runs with boundary_frac in target range
    in_range = [r for r in results if not math.isnan(r["bnd_frac"]) and 0.20 <= r["bnd_frac"] <= 0.40]
    # Runs that beat reference on overall LM
    better_lm = [r for r in results if not math.isnan(r["val_lm"]) and r["val_lm"] < ref_val_lm]
    # Runs with improved boundary NLL
    better_bnd = [r for r in results if not math.isnan(r["bnd_lm"]) and not math.isnan(ref_bnd_lm)
                  and r["bnd_lm"] < ref_bnd_lm]
    # Runs with stable core NLL (|drift| < 0.01)
    stable_core = [r for r in results if not math.isnan(r["core_lm"]) and not math.isnan(ref_core_lm)
                   and abs(r["core_lm"] - ref_core_lm) < 0.01]

    if in_range:
        print(f"  Gate in target range (0.20–0.40):")
        for r in sorted(in_range, key=lambda x: x["val_lm"]):
            print(f"    {r['label']:<34}  bnd_frac={r['bnd_frac']:.2f}  val_lm={r['val_lm']:.4f}")
    else:
        print(f"  No run achieved boundary_frac in 0.20–0.40 — consider lower tau")

    print()
    if better_lm:
        best = min(better_lm, key=lambda r: r["val_lm"])
        print(f"  PASS  Best LM improvement: {best['label']} → val_lm={best['val_lm']:.4f}  ({best['delta']:+.4f})")
    else:
        best_overall = min(results, key=lambda r: r["val_lm"])
        print(f"  FAIL  No run beats reference val_lm={ref_val_lm:.4f}  "
              f"(best: {best_overall['label']} at {best_overall['val_lm']:.4f})")

    if better_bnd:
        best = min(better_bnd, key=lambda r: r["bnd_lm"])
        print(f"  PASS  Boundary-token NLL improved: {ref_bnd_lm:.4f} → {best['bnd_lm']:.4f}  ({best['label']})")
    elif not math.isnan(ref_bnd_lm):
        print(f"  FAIL  No run improved boundary NLL (ref={ref_bnd_lm:.4f})")

    if stable_core:
        print(f"  PASS  {len(stable_core)}/{len(results)} run(s) kept core NLL stable (|drift| < 0.01)")
    else:
        print(f"  WARN  No run kept core NLL within 0.01 of reference")

    # Recommend best config for multihyp reuse
    candidates = [r for r in results
                  if not math.isnan(r["bnd_frac"]) and 0.20 <= r["bnd_frac"] <= 0.40
                  and not math.isnan(r["val_lm"]) and r["val_lm"] <= ref_val_lm]
    print()
    if candidates:
        rec = min(candidates, key=lambda r: r["val_lm"])
        print(f"  RECOMMENDED for multihyp reuse: {rec['label']}")
        print(f"    → bnd_frac={rec['bnd_frac']:.2f}  val_lm={rec['val_lm']:.4f}  "
              f"bnd_lm={rec['bnd_lm']:.4f}  core_lm={rec['core_lm']:.4f}")
    else:
        # Fall back: best bnd_frac regardless of LM quality
        ranked = sorted(
            [r for r in results if not math.isnan(r["bnd_frac"])],
            key=lambda r: abs(r["bnd_frac"] - 0.30)
        )
        if ranked:
            rec = ranked[0]
            print(f"  CANDIDATE for multihyp (closest bnd_frac to 0.30, may not beat LM): {rec['label']}")
            print(f"    → bnd_frac={rec['bnd_frac']:.2f}  val_lm={rec['val_lm']:.4f}")

PYEOF

echo ""
echo "Done at $(date)"
