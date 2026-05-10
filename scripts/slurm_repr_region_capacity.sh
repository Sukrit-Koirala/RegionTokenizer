#!/bin/bash
#SBATCH --job-name=repr_region_cap
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
# 6 targeted runs (20k steps each) testing the routing-collapse hypothesis:
#
#   Run 1: repr_region reference       (existing mode, best alpha=0.2)
#   Run 2: capacity only               (capacity_alpha=0.5, no diversity)
#   Run 3: diversity only              (lambda_diversity=0.01, no capacity)
#   Run 4: capacity + diversity        (both enabled)
#   Run 5: random_repr_region_capacity (null hypothesis, same config as Run 4)
#   Run 6: oracle_repr_region_capacity (upper bound, same config as Run 4)
#
# Hypothesis: router collapse into dominant regions is the failure mode.
# If capacity + diversity helps: usage entropy ↑, participation ratio ↑,
# val_lm_loss ↓, and random control does NOT match.

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
BASELINE_F=runs/baseline/final_summary.json

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

# ── Shared base args (all runs) ────────────────────────────────────────────────
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
    --base_alpha          0.2
    --region_warmup_steps 5000
)

# Shared capacity-mode args (Runs 2–6)
CAP_ARGS=(
    --usage_ema_decay         0.99
    --router_temp_init        2.0
    --router_temp_final       1.0
    --router_temp_decay_steps 10000
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
# Run 1: repr_region reference  (existing mode, alpha=0.2, no capacity args)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 1/6: repr_region reference  (alpha=0.2)"
echo "  output_dir: runs/repr_region_reference"
echo "=============================="
run_if_needed runs/repr_region_reference \
    --mode          repr_region \
    --router_temp   2.0 \
    "${BASE_ARGS[@]}"
echo "  Run 1 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 2: capacity only  (capacity_alpha=0.5, lambda_diversity=0.0)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 2/6: repr_region_capacity  (capacity_alpha=0.5, no diversity)"
echo "  output_dir: runs/repr_region_capacity_alpha0p5"
echo "=============================="
run_if_needed runs/repr_region_capacity_alpha0p5 \
    --mode             repr_region_capacity \
    --capacity_alpha   0.5 \
    --lambda_diversity 0.0 \
    "${BASE_ARGS[@]}" "${CAP_ARGS[@]}"
echo "  Run 2 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 3: diversity only  (capacity_alpha=0.0, lambda_diversity=0.01)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 3/6: repr_region_capacity  (no capacity, lambda_diversity=0.01)"
echo "  output_dir: runs/repr_region_diversity0p01"
echo "=============================="
run_if_needed runs/repr_region_diversity0p01 \
    --mode             repr_region_capacity \
    --capacity_alpha   0.0 \
    --lambda_diversity 0.01 \
    "${BASE_ARGS[@]}" "${CAP_ARGS[@]}"
echo "  Run 3 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 4: capacity + diversity  (both enabled)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 4/6: repr_region_capacity  (capacity_alpha=0.5 + lambda_diversity=0.01)"
echo "  output_dir: runs/repr_region_capacity_diversity"
echo "=============================="
run_if_needed runs/repr_region_capacity_diversity \
    --mode             repr_region_capacity \
    --capacity_alpha   0.5 \
    --lambda_diversity 0.01 \
    "${BASE_ARGS[@]}" "${CAP_ARGS[@]}"
echo "  Run 4 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 5: random_repr_region_capacity  (null hypothesis — random partition)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 5/6: random_repr_region_capacity  (null hypothesis, same config as Run 4)"
echo "  output_dir: runs/random_repr_region_capacity_diversity"
echo "=============================="
run_if_needed runs/random_repr_region_capacity_diversity \
    --mode             random_repr_region_capacity \
    --capacity_alpha   0.5 \
    --lambda_diversity 0.01 \
    "${BASE_ARGS[@]}" "${CAP_ARGS[@]}"
echo "  Run 5 done at $(date)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 6: oracle_repr_region_capacity  (upper bound)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 6/6: oracle_repr_region_capacity  (upper bound)"
echo "  output_dir: runs/oracle_repr_region_capacity"
echo "=============================="
run_if_needed runs/oracle_repr_region_capacity \
    --mode             oracle_repr_region_capacity \
    --capacity_alpha   0.5 \
    --lambda_diversity 0.01 \
    "${BASE_ARGS[@]}" "${CAP_ARGS[@]}"
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
    ("repr_region_reference",               "repr_region",                    "reference"),
    ("repr_region_capacity_alpha0p5",        "repr_region_capacity",           "cap only"),
    ("repr_region_diversity0p01",            "repr_region_capacity",           "div only"),
    ("repr_region_capacity_diversity",       "repr_region_capacity",           "cap+div"),
    ("random_repr_region_capacity_diversity","random_repr_region_capacity",    "random"),
    ("oracle_repr_region_capacity",          "oracle_repr_region_capacity",    "oracle"),
]
BASELINE_F = "runs/baseline/final_summary.json"

baseline_loss = None
if os.path.isfile(BASELINE_F):
    with open(BASELINE_F) as fh:
        bd = json.load(fh)
    baseline_loss = bd.get("val_lm_loss")
    print(f"  {'baseline':42s}  val_lm={baseline_loss:.4f}  ppl={math.exp(min(baseline_loss,20)):.2f}")
print()

results = []
for run_dir, mode, label in RUNS:
    f = f"runs/{run_dir}/final_summary.json"
    if not os.path.isfile(f):
        print(f"  [{label:10s}] {run_dir:<42s}  (no results yet)")
        continue
    with open(f) as fh:
        d = json.load(fh)
    val_lm   = d.get("val_lm_loss")
    val_ppl  = d.get("val_ppl")
    acc1     = d.get("val_coarse_acc1", 0.0)
    ent      = d.get("val_coarse_ent", 0.0)
    u_ent    = d.get("val_usage_entropy", 0.0)
    pr       = d.get("val_participation_ratio", 0.0)
    gini     = d.get("val_region_gini", 0.0)
    act1     = d.get("val_active_regions_1pct", 0)
    delta    = (val_lm - baseline_loss) if (val_lm and baseline_loss) else None
    delta_s  = f"{delta:+.4f}" if delta is not None else "—"
    print(f"  [{label:10s}] {run_dir:<42s}  val_lm={val_lm:.4f}  ppl={val_ppl:.2f}"
          f"  acc@1={acc1:.3f}  router_ent={ent:.3f}  usage_ent={u_ent:.3f}"
          f"  PR={pr:.1f}  gini={gini:.3f}  act@1%={act1:3d}  Δ={delta_s}")
    results.append((val_lm, mode, label, run_dir, delta, pr, u_ent))

print()
print("── Collapse Hypothesis Verdict ──")

real_runs  = [(r[0], r[2]) for r in results if "oracle" not in r[1] and "random" not in r[1]]
rand_runs  = [(r[0], r[2]) for r in results if "random" in r[1]]
orac_runs  = [(r[0], r[2]) for r in results if "oracle" in r[1]]
ref_runs   = [(r[0], r[4], r[5], r[6]) for r in results if r[2] == "reference"]
best_cap   = min([r for r in results if "capacity" in r[1] and "oracle" not in r[1] and "random" not in r[1]],
                 key=lambda r: r[0]) if any("capacity" in r[1] for r in results if "oracle" not in r[1] and "random" not in r[1]) else None

if real_runs and baseline_loss:
    best_real_lm = min(r[0] for r in real_runs)
    if best_real_lm < baseline_loss:
        print(f"  PASS  Best repr_region_capacity beats baseline ({best_real_lm:.4f} < {baseline_loss:.4f})")
    else:
        print(f"  FAIL  Best repr_region_capacity does not beat baseline ({best_real_lm:.4f} >= {baseline_loss:.4f})")

if real_runs and rand_runs:
    if min(r[0] for r in real_runs) < min(r[0] for r in rand_runs):
        print(f"  PASS  Best capacity run beats random control")
    else:
        print(f"  FAIL  Capacity run does not beat random (collapse not the only issue)")

if ref_runs and best_cap:
    ref_lm, ref_delta, ref_pr, ref_ent = ref_runs[0]
    cap_lm, _, cap_label, _, _, cap_pr, cap_ent = best_cap
    if cap_lm < ref_lm:
        print(f"  PASS  {cap_label} improves over repr_region reference ({cap_lm:.4f} < {ref_lm:.4f})")
    else:
        print(f"  FAIL  {cap_label} does not improve over repr_region reference")
    if cap_pr > ref_pr:
        print(f"  INFO  Participation ratio increased: {ref_pr:.1f} → {cap_pr:.1f} (collapse reduced)")
    if cap_ent > ref_ent:
        print(f"  INFO  Usage entropy increased: {ref_ent:.3f} → {cap_ent:.3f} (more distributed routing)")

if orac_runs and real_runs:
    headroom = min(r[0] for r in real_runs) - min(r[0] for r in orac_runs)
    print(f"  INFO  Remaining oracle headroom: {headroom:+.4f}")
PYEOF

echo ""
echo "Done at $(date)"
