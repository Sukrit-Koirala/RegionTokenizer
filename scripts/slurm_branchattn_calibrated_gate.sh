#!/bin/bash
#SBATCH --job-name=branchattn_cal
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
# Focused single test: repr_region_branchattn with calibrated boundary gate.
#
# Context:
#   repr_region_reference                      val_lm = 3.9978  (target to beat)
#   calibrated single-path boundary tau=0.03   val_lm = 4.0144  (partial-success bar)
#   calibrated multihyp MLP  tau=0.03          val_lm = 4.0519  (previous best branch)
#
# Hypothesis:
#   Independent MLP refinement of K branches is insufficient — branches need
#   to communicate (via attention over the K dimension) before merging.
#
# Gate settings (from sweep run 1):
#   boundary_tau = 0.03  boundary_temp = 0.01  → boundary_frac ≈ 0.25–0.30
#
# Success criteria:
#   Strong:  val_lm < 3.9978  (beats reference)
#   Partial: val_lm < 4.0144  (beats single-path calibrated boundary)
#   Weak:    val_lm < 4.0519  (at least beats MLP multihyp)
#   Failure: val_lm >= 4.0519

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
OUT_DIR=runs/repr_region_branchattn_k4_tau0p03_temp0p01

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
echo "repr_region_branchattn  K=4  tau=0.03  temp=0.01"
echo "  branch_attn_heads=4  branch_attn_layers=1"
echo "  output_dir: $OUT_DIR"
echo "=============================="

SUMMARY="${OUT_DIR}/final_summary.json"
if [[ -f "$SUMMARY" && "${FORCE_RUN:-0}" != "1" ]]; then
    echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
else
    mkdir -p "$OUT_DIR"
    python train_region_lm.py \
        --mode                repr_region_branchattn \
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
        --branch_attn_heads   4 \
        --branch_attn_layers  1 \
        --branch_attn_dropout 0.1 \
        --branch_scale_init   0.0 \
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

# Hard-coded fallbacks — overridden by live checkpoints if present
REF_VAL_LM       = 3.9978   # repr_region_reference
CAL_BND_LM       = 4.0144   # calibrated single-path boundary  tau=0.03 temp=0.01
CAL_MHYP_MLP_LM  = 4.0519   # calibrated multihyp MLP          tau=0.03 temp=0.01

OUT_DIR = "runs/repr_region_branchattn_k4_tau0p03_temp0p01"

# Pull live values when available
for path, name, var in [
    ("runs/repr_region_reference/final_summary.json",
     "repr_region_reference", "REF_VAL_LM"),
    ("runs/boundary_tau0p03_temp0p01/final_summary.json",
     "calibrated single-path boundary", "CAL_BND_LM"),
    ("runs/repr_region_multihyp_k4_tau0p03_temp0p01/final_summary.json",
     "calibrated multihyp MLP", "CAL_MHYP_MLP_LM"),
]:
    if os.path.isfile(path):
        with open(path) as fh:
            d = json.load(fh)
        live = d.get("val_lm_loss")
        if live is not None:
            if var == "REF_VAL_LM":       REF_VAL_LM       = live
            elif var == "CAL_BND_LM":     CAL_BND_LM       = live
            elif var == "CAL_MHYP_MLP_LM": CAL_MHYP_MLP_LM = live

# ── Baseline table ─────────────────────────────────────────────────────────────
print("  Baselines")
print(f"    repr_region_reference                   val_lm = {REF_VAL_LM:.4f}  [strong-pass target]")
print(f"    calibrated single-path boundary         val_lm = {CAL_BND_LM:.4f}  [partial-pass target]")
print(f"    calibrated multihyp MLP (tau=0.03)      val_lm = {CAL_MHYP_MLP_LM:.4f}  [weak-pass target]")
print()

# ── This run ───────────────────────────────────────────────────────────────────
summary_f = f"{OUT_DIR}/final_summary.json"
if not os.path.isfile(summary_f):
    print("  No results yet — training may not have completed.")
    raise SystemExit(0)

with open(summary_f) as fh:
    d = json.load(fh)

val_lm      = d.get("val_lm_loss",            float("nan"))
val_ppl     = d.get("val_ppl",                float("nan"))
bnd_frac    = d.get("val_boundary_frac",      float("nan"))
bnd_lm      = d.get("val_boundary_lm",        float("nan"))
core_lm     = d.get("val_core_lm",            float("nan"))
alpha_bnd   = d.get("val_avg_boundary_alpha", float("nan"))
alpha_cor   = d.get("val_avg_core_alpha",     float("nan"))
acc1        = d.get("val_coarse_acc1",        float("nan"))
acc4        = d.get("val_coarse_acc4",        float("nan"))
br_scale    = d.get("val_branch_scale",       float("nan"))
refiner     = d.get("param_breakdown", {}).get("refiner", 0)

fmt  = lambda v: f"{v:.4f}" if not math.isnan(v) else "—"
fmtd = lambda v: f"{v:+.4f}" if not math.isnan(v) else "—"

delta_ref  = val_lm - REF_VAL_LM       if not math.isnan(val_lm) else float("nan")
delta_bnd  = val_lm - CAL_BND_LM       if not math.isnan(val_lm) else float("nan")
delta_mhyp = val_lm - CAL_MHYP_MLP_LM if not math.isnan(val_lm) else float("nan")

print("  repr_region_branchattn  K=4  tau=0.03  temp=0.01")
print(f"    val_lm              = {fmt(val_lm)}")
print(f"    ppl                 = {fmt(val_ppl)}")
print(f"    boundary_frac       = {fmt(bnd_frac)}")
print(f"    boundary_lm         = {fmt(bnd_lm)}")
print(f"    core_lm             = {fmt(core_lm)}")
print(f"    avg_boundary_alpha  = {fmt(alpha_bnd)}")
print(f"    avg_core_alpha      = {fmt(alpha_cor)}")
print(f"    acc@1 (router)      = {fmt(acc1)}")
print(f"    acc@4 (router)      = {fmt(acc4)}")
print(f"    branch_scale (tanh) = {fmt(br_scale)}")
print(f"    refiner params      = {refiner:,}")
print()
print(f"    Δ vs reference              = {fmtd(delta_ref)}")
print(f"    Δ vs calibrated single-path = {fmtd(delta_bnd)}")
print(f"    Δ vs calibrated MLP multihyp= {fmtd(delta_mhyp)}")
print()

# ── Verdict ────────────────────────────────────────────────────────────────────
print("── Verdict ──")

if math.isnan(val_lm):
    print("  (incomplete — val_lm missing)")
elif val_lm < REF_VAL_LM:
    print(f"  STRONG PASS  val_lm={val_lm:.4f} beats reference ({REF_VAL_LM:.4f})")
    print(f"  Branch attention + calibrated gating beats single-alpha reference.")
elif val_lm < CAL_BND_LM:
    print(f"  PARTIAL PASS  val_lm={val_lm:.4f} < single-path ({CAL_BND_LM:.4f})")
    print(f"  Branch interaction helps over single-path but does not beat reference.")
    print(f"  → Consider more layers, larger K, or longer training.")
elif val_lm < CAL_MHYP_MLP_LM:
    print(f"  WEAK PASS  val_lm={val_lm:.4f} < MLP multihyp ({CAL_MHYP_MLP_LM:.4f})")
    print(f"  Branch attention improves over MLP refinement but not over single-path.")
    print(f"  → Interaction helps within branch path; broader conditioning is still missing.")
else:
    print(f"  FAIL  val_lm={val_lm:.4f} >= MLP multihyp ({CAL_MHYP_MLP_LM:.4f})")
    print(f"  Branch attention did not help over independent MLP refinement.")
    print(f"  → Boundary uncertainty likely requires sequence-level context,")
    print(f"     not just interaction among static region branches.")

# Gate selectivity
if not math.isnan(bnd_frac):
    if 0.20 <= bnd_frac <= 0.40:
        print(f"  GATE OK     boundary_frac={bnd_frac:.2f}  (target 0.20–0.40)")
    elif bnd_frac < 0.20:
        print(f"  GATE TIGHT  boundary_frac={bnd_frac:.2f}")
    else:
        print(f"  GATE BROAD  boundary_frac={bnd_frac:.2f}  (above target range)")

# Branch scale check — did the refiner actually activate?
if not math.isnan(br_scale):
    if abs(br_scale) < 0.01:
        print(f"  WARN  branch_scale≈0 ({br_scale:.4f}) — refiner barely activated; "
              f"check gradient flow or increase training time")
    else:
        print(f"  INFO  branch_scale = {br_scale:.4f}  (refiner is active)")

PYEOF

echo ""
echo "Done at $(date)"
