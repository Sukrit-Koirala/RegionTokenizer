#!/bin/bash
#SBATCH --job-name=mhyp_cal_gate
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Single focused test: repr_region_multihyp with calibrated boundary gate.
#
# Context:
#   Prior multihyp run (tau=0.10, temp=0.05) → val_lm=4.0346  boundary_frac≈0.83
#   Gate swept to (tau=0.03, temp=0.01)      → boundary_frac≈0.30  (selective)
#   Calibrated single-path boundary           → val_lm=4.0144
#   Reference repr_region                    → val_lm=3.9978
#
# Isolated question:
#   Did the original multihyp fail mainly because boundary gating was too broad?
#
# Success:  val_lm < 3.9978  AND  boundary_lm improves over 4.0144 single-path
# Partial:  val_lm < 4.0144  (gating helped but multihyp still not beating reference)
# Failure:  val_lm >= 4.0144 (multihyp not helping even with clean gate)

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
OUT_DIR=runs/repr_region_multihyp_k4_tau0p03_temp0p01

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

echo "=============================="
echo "repr_region_multihyp  K=4  tau=0.03  temp=0.01"
echo "  output_dir: $OUT_DIR"
echo "=============================="

SUMMARY="${OUT_DIR}/final_summary.json"
if [[ -f "$SUMMARY" && "${FORCE_RUN:-0}" != "1" ]]; then
    echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
else
    mkdir -p "$OUT_DIR"
    python train_region_lm.py \
        --mode                repr_region_multihyp \
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
        --region_map_path     "$REGION_MAP" \
        --router_type         mlp \
        --router_temp         2.0 \
        --lambda_coarse       0.2 \
        --lambda_balance      0.001 \
        --hyp_k               4 \
        --alpha_core          0.2 \
        --alpha_boundary      0.4 \
        --boundary_tau        0.03 \
        --boundary_temp       0.01 \
        --boundary_mode       margin \
        --region_warmup_steps 5000 \
        --eval_interval       500 \
        --save_interval       2000 \
        --log_interval        100 \
        --seed                42 \
        --device              cuda \
        --output_dir          "$OUT_DIR"
fi

echo "  Training done at $(date)"
echo ""

# ── Results ────────────────────────────────────────────────────────────────────
echo "=============================="
echo "Results"
echo "=============================="

python3 - <<'PYEOF'
import json, os, math

REF_VAL_LM          = 3.9978   # repr_region_reference (hard-coded fallback)
PREV_MULTIHYP_LM    = 4.0346   # repr_region_multihyp_k4 tau=0.10 temp=0.05
CAL_BOUNDARY_LM     = 4.0144   # calibrated single-path boundary tau=0.03 temp=0.01

OUT_DIR = "runs/repr_region_multihyp_k4_tau0p03_temp0p01"

# Pull live reference if available
for path, attr in [
    ("runs/repr_region_reference/final_summary.json",              "ref_val_lm"),
    ("runs/boundary_tau0p03_temp0p01/final_summary.json",          "cal_bnd_lm"),
    ("runs/repr_region_multihyp_k4/final_summary.json",            "prev_mhyp_lm"),
]:
    if os.path.isfile(path):
        with open(path) as fh:
            d = json.load(fh)
        live = d.get("val_lm_loss")
        if live is not None:
            if attr == "ref_val_lm":       REF_VAL_LM       = live
            elif attr == "cal_bnd_lm":     CAL_BOUNDARY_LM  = live
            elif attr == "prev_mhyp_lm":   PREV_MULTIHYP_LM = live

# ── Baselines ──────────────────────────────────────────────────────────────────
print("  Baselines")
print(f"    repr_region_reference (target to beat)   val_lm = {REF_VAL_LM:.4f}")
print(f"    calibrated boundary single-path          val_lm = {CAL_BOUNDARY_LM:.4f}  (partial-success threshold)")
print(f"    prev multihyp tau=0.10 temp=0.05         val_lm = {PREV_MULTIHYP_LM:.4f}  (failed run)")
print()

# ── This run ───────────────────────────────────────────────────────────────────
summary_f = f"{OUT_DIR}/final_summary.json"
if not os.path.isfile(summary_f):
    print("  No results yet — training may not have completed.")
    raise SystemExit(0)

with open(summary_f) as fh:
    d = json.load(fh)

val_lm    = d.get("val_lm_loss",            float("nan"))
val_ppl   = d.get("val_ppl",                float("nan"))
bnd_frac  = d.get("val_boundary_frac",      float("nan"))
bnd_lm    = d.get("val_boundary_lm",        float("nan"))
core_lm   = d.get("val_core_lm",            float("nan"))
alpha_bnd = d.get("val_avg_boundary_alpha", float("nan"))
alpha_cor = d.get("val_avg_core_alpha",     float("nan"))
acc1      = d.get("val_coarse_acc1",        float("nan"))
refiner   = d.get("param_breakdown", {}).get("refiner", 0)

delta_ref  = val_lm - REF_VAL_LM       if not math.isnan(val_lm) else float("nan")
delta_prev = val_lm - PREV_MULTIHYP_LM if not math.isnan(val_lm) else float("nan")
delta_cal  = val_lm - CAL_BOUNDARY_LM  if not math.isnan(val_lm) else float("nan")

fmt = lambda v: f"{v:.4f}" if not math.isnan(v) else "—"
fmtd = lambda v: f"{v:+.4f}" if not math.isnan(v) else "—"

print("  repr_region_multihyp  K=4  tau=0.03  temp=0.01")
print(f"    val_lm              = {fmt(val_lm)}")
print(f"    ppl                 = {fmt(val_ppl)}")
print(f"    boundary_frac       = {fmt(bnd_frac)}")
print(f"    boundary_lm         = {fmt(bnd_lm)}")
print(f"    core_lm             = {fmt(core_lm)}")
print(f"    avg_boundary_alpha  = {fmt(alpha_bnd)}")
print(f"    avg_core_alpha      = {fmt(alpha_cor)}")
print(f"    acc@1 (router)      = {fmt(acc1)}")
print(f"    refiner params      = {refiner:,}")
print()
print(f"    Δ vs reference      = {fmtd(delta_ref)}")
print(f"    Δ vs prev multihyp  = {fmtd(delta_prev)}")
print(f"    Δ vs cal boundary   = {fmtd(delta_cal)}")
print()

# ── Verdict ────────────────────────────────────────────────────────────────────
print("── Verdict ──")

if math.isnan(val_lm):
    print("  (incomplete — val_lm missing)")
elif val_lm < REF_VAL_LM:
    print(f"  STRONG PASS  val_lm={val_lm:.4f} beats reference ({REF_VAL_LM:.4f})")
    print(f"  Calibrated gating fixed the multihyp failure AND beats single-alpha reference.")
elif val_lm < CAL_BOUNDARY_LM:
    print(f"  PARTIAL PASS  val_lm={val_lm:.4f} < calibrated boundary ({CAL_BOUNDARY_LM:.4f})")
    print(f"  Multihyp helps over calibrated single-path but does not beat reference.")
    print(f"  → Consider K=8, stronger alpha_boundary, or longer training.")
else:
    print(f"  FAIL  val_lm={val_lm:.4f} >= calibrated boundary ({CAL_BOUNDARY_LM:.4f})")
    print(f"  Multihyp is not helping even with a clean gate.")
    print(f"  → The multi-hypothesis mechanism itself may not be the right inductive bias.")

# Gate selectivity check
if not math.isnan(bnd_frac):
    if 0.20 <= bnd_frac <= 0.40:
        print(f"  GATE OK     boundary_frac={bnd_frac:.2f}  (target 0.20–0.40)")
    elif bnd_frac < 0.20:
        print(f"  GATE TIGHT  boundary_frac={bnd_frac:.2f}  (may be too selective)")
    else:
        print(f"  GATE BROAD  boundary_frac={bnd_frac:.2f}  (still above target range)")

# Gate improvement check vs previous bad run
if not math.isnan(delta_prev) and delta_prev < 0:
    print(f"  IMPROVED    {abs(delta_prev):.4f} better than broad-gate multihyp run")

PYEOF

echo ""
echo "Done at $(date)"
