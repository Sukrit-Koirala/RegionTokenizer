#!/bin/bash
#SBATCH --job-name=repr_region
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
# 6 targeted runs (20k steps each):
#   repr_region  × alpha ∈ {0.025, 0.05, 0.1, 0.2}   (alpha sweep)
#   random_repr_region  alpha=0.1                      (null hypothesis)
#   oracle_repr_region  alpha=0.1                      (upper bound)
#
# Previous grid showed:
#   soft_moe ≈ worse than baseline  →  logit-level biasing is harmful
#   oracle_soft_moe ≪ baseline      →  region structure has headroom
#   best config: K=128, router_type=mlp, lambda_coarse=0.2
# This sweep tests whether representation-level conditioning captures that headroom.

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
    echo "ERROR: region map not found: $REGION_MAP" >&2
    echo "       Run build_regions (K=128) first or submit slurm_soft_moe_grid.sh" >&2
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

# ── Shared args ────────────────────────────────────────────────────────────────
COMMON_ARGS=(
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

# ══════════════════════════════════════════════════════════════════════════════
# Runs 1–4: repr_region  alpha sweep
# ══════════════════════════════════════════════════════════════════════════════
RUN_IDX=0
for ALPHA in 0.025 0.05 0.1 0.2; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    ALPHA_S="${ALPHA//./p}"
    OUT_DIR="runs/repr_region_128_mlp_alpha${ALPHA_S}"
    SUMMARY="${OUT_DIR}/final_summary.json"

    echo "=============================="
    echo "Run ${RUN_IDX}/6: repr_region  alpha=${ALPHA}"
    echo "  output_dir: ${OUT_DIR}"
    echo "=============================="

    if [[ -f "$SUMMARY" && "${FORCE_RUN:-0}" != "1" ]]; then
        echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
        echo ""
        continue
    fi

    mkdir -p "$OUT_DIR"
    python train_region_lm.py \
        --mode       repr_region \
        --output_dir "$OUT_DIR" \
        --base_alpha "$ALPHA" \
        "${COMMON_ARGS[@]}"
    echo "  repr_region alpha=${ALPHA} done at $(date)"
    echo ""
done

# ══════════════════════════════════════════════════════════════════════════════
# Run 5: random_repr_region  (null hypothesis — random partition, alpha=0.1)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 5/6: random_repr_region  alpha=0.1  (null hypothesis)"
echo "  output_dir: runs/random_repr_region_128_mlp_alpha0p1"
echo "=============================="

OUT_DIR=runs/random_repr_region_128_mlp_alpha0p1
if [[ -f "${OUT_DIR}/final_summary.json" && "${FORCE_RUN:-0}" != "1" ]]; then
    echo "  Already done — skipping"
else
    mkdir -p "$OUT_DIR"
    python train_region_lm.py \
        --mode       random_repr_region \
        --output_dir "$OUT_DIR" \
        --base_alpha 0.1 \
        "${COMMON_ARGS[@]}"
    echo "  random_repr_region done at $(date)"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Run 6: oracle_repr_region  (upper bound — true region one-hot, alpha=0.1)
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================="
echo "Run 6/6: oracle_repr_region  alpha=0.1  (upper bound)"
echo "  output_dir: runs/oracle_repr_region_128_alpha0p1"
echo "=============================="

OUT_DIR=runs/oracle_repr_region_128_alpha0p1
if [[ -f "${OUT_DIR}/final_summary.json" && "${FORCE_RUN:-0}" != "1" ]]; then
    echo "  Already done — skipping"
else
    mkdir -p "$OUT_DIR"
    # Note: oracle does not use router_type / lambda_coarse / router_temp
    python train_region_lm.py \
        --mode                oracle_repr_region \
        --output_dir          "$OUT_DIR" \
        --base_alpha          0.1 \
        --region_warmup_steps 5000 \
        --dataset             wikitext-103-raw-v1 \
        --vocab_subset_size   50257 \
        --seq_len             256 \
        --batch_size          32 \
        --n_layer             6 \
        --d_model             384 \
        --n_head              6 \
        --d_ff                1536 \
        --steps               20000 \
        --lr                  3e-4 \
        --warmup_steps        1000 \
        --eval_interval       500 \
        --save_interval       2000 \
        --log_interval        100 \
        --seed                42 \
        --device              cuda \
        --region_map_path     "$REGION_MAP"
    echo "  oracle_repr_region done at $(date)"
fi
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
    ("repr_region_128_mlp_alpha0p025",     "repr_region",        0.025),
    ("repr_region_128_mlp_alpha0p05",      "repr_region",        0.05),
    ("repr_region_128_mlp_alpha0p1",       "repr_region",        0.1),
    ("repr_region_128_mlp_alpha0p2",       "repr_region",        0.2),
    ("random_repr_region_128_mlp_alpha0p1","random_repr_region", 0.1),
    ("oracle_repr_region_128_alpha0p1",    "oracle_repr_region", 0.1),
]
BASELINE_F = "runs/baseline/final_summary.json"

baseline_loss = None
if os.path.isfile(BASELINE_F):
    with open(BASELINE_F) as fh:
        bd = json.load(fh)
    baseline_loss = bd.get("val_lm_loss")
    print(f"  {'baseline':40s}  val_lm={baseline_loss:.4f}  ppl={math.exp(min(baseline_loss,20)):.2f}")

print()
results = []
for run_dir, mode, alpha in RUNS:
    f = f"runs/{run_dir}/final_summary.json"
    if not os.path.isfile(f):
        print(f"  {run_dir:<40}  (no results yet)")
        continue
    with open(f) as fh:
        d = json.load(fh)
    val_lm = d.get("val_lm_loss")
    val_ppl = d.get("val_ppl")
    acc1 = d.get("val_coarse_acc1", 0.0)
    acc4 = d.get("val_coarse_acc4", 0.0)
    cov  = d.get("val_coarse_coverage", 0.0)
    delta = (val_lm - baseline_loss) if (val_lm and baseline_loss) else None
    delta_str = f"{delta:+.4f}" if delta is not None else "—"
    print(f"  {run_dir:<40}  val_lm={val_lm:.4f}  ppl={val_ppl:.2f}"
          f"  acc@1={acc1:.3f}  acc@4={acc4:.3f}  cov={cov:.1%}  Δbaseline={delta_str}")
    results.append((val_lm, mode, run_dir, delta))

results.sort()

print()
print("── Verdict ──")
real = [r for r in results if r[1] == "repr_region"]
rand = [r for r in results if r[1] == "random_repr_region"]
orac = [r for r in results if r[1] == "oracle_repr_region"]

if real and baseline_loss:
    best_lm = real[0][0]
    if best_lm < baseline_loss:
        print(f"  PASS  Best repr_region ({real[0][2]}) beats baseline ({best_lm:.4f} < {baseline_loss:.4f})")
    else:
        print(f"  FAIL  Best repr_region does not beat baseline ({best_lm:.4f} >= {baseline_loss:.4f})")

if real and rand:
    if real[0][0] < rand[0][0]:
        print(f"  PASS  Best repr_region beats random_repr_region")
    else:
        print(f"  FAIL  Best repr_region does not beat random (representation still not useful)")

if orac and real:
    headroom = real[0][0] - orac[0][0]
    print(f"  INFO  Oracle headroom vs best repr_region: {headroom:+.4f}")
    if headroom > 0.02:
        print(f"        Large headroom → router bottleneck; consider stronger router or longer warmup")
    else:
        print(f"        Small headroom → repr conditioning is near-optimal for this K")
PYEOF

echo ""
echo "Done at $(date)"
