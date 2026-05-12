#!/bin/bash
#SBATCH --job-name=bnd_addepth
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
# Adaptive-depth boundary experiment — PRIMARY hypothesis test.
#
# Hypothesis:
#   Sparse hard-masked extra compute (applied ONLY at genuine boundary tokens)
#   is more efficient and more effective than soft gate-weighted refinement
#   applied globally to all tokens (seqrefine).
#
# Key design difference vs repr_region_boundary_seqrefine:
#   seqrefine:      gate-WEIGHTED delta added to ALL tokens (soft, global)
#   adaptivedepth:  delta added ONLY to tokens where gate > threshold (hard, sparse)
#   Both run adaptive_blocks on full sequence for causal context;
#   difference is in how delta is applied.
#
# Architecture:
#   - HARD boundary_mask = (gate > adaptive_threshold).float()  [sparse ~bnd_frac]
#   - h_core = h + alpha_core * ws * region_feat
#   - h_refined = adaptive_blocks(h_core)   [full sequence, causal context]
#   - delta = h_refined - h_core
#   - scale = tanh(refine_scale) * ws
#   - h_final = h_core + boundary_mask * scale * delta   [only boundary tokens updated]
#
# Key metric: frac_tokens_refined — should equal boundary_frac (~0.25–0.30)
#             if >> boundary_frac, threshold is too low; if near 0, threshold too high.
#
# Runs:
#   1. repr_region_boundary_adaptivedepth    — real region map, sparse hard gate
#   2. random_repr_region_boundary_adaptivedepth — shuffled map null hypothesis
#      If random ≈ real → gain is from extra compute; if real << random → real regions help
#
# Baselines for comparison (from prior runs):
#   repr_region_reference              val_lm ≈ 3.9978  [strong target]
#   boundary single-path tau=0.03      val_lm ≈ 4.0144  [partial target]
#   boundary seqrefine 1-layer         val_lm ≈ ?       [primary comparison]
#   random seqrefine control           val_lm ≈ ?       [extra-compute bar]
#   branch identity K=4                val_lm ≈ ?
#   branch attention K=4 (fixed)       val_lm ≈ ?
#
# Config: 30k steps, seq_len=256, batch=32, tau=0.03, temp=0.01, seed=42

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
REAL_DIR=runs/repr_region_boundary_adaptivedepth_30k
RAND_DIR=runs/random_boundary_adaptivedepth_30k

if [[ ! -f "$REGION_MAP" ]]; then
    echo "ERROR: region map missing: $REGION_MAP" >&2; exit 1
fi
if [[ ! -f "train_region_lm.py" ]]; then
    echo "ERROR: train_region_lm.py not found in $PWD" >&2; exit 1
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

# ─────────────────────────────────────────────────────────────────────────────
# PREFLIGHT — verify both adaptivedepth modes before committing 12h of GPU
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "PREFLIGHT CHECKS"
echo "=============================="

PREFLIGHT_FAILED=0
for MODE in repr_region_boundary_adaptivedepth random_repr_region_boundary_adaptivedepth; do
    echo "--- preflight: $MODE ---"
    if ! python scripts/preflight_boundary.py \
            --mode "$MODE" \
            --region_map_path "$REGION_MAP" \
            --device cuda; then
        echo "PREFLIGHT FAILED for $MODE — aborting job." >&2
        PREFLIGHT_FAILED=1
    fi
done

if [[ "$PREFLIGHT_FAILED" -ne 0 ]]; then
    echo "One or more preflight checks failed. Aborting." >&2
    exit 1
fi

echo ""
echo "All preflight checks passed."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Shared training arguments
# ─────────────────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --dataset             wikitext-103-raw-v1
    --vocab_subset_size   50257
    --seq_len             256
    --batch_size          32
    --n_layer             6
    --d_model             384
    --n_head              6
    --d_ff                1536
    --steps               30000
    --lr                  3e-4
    --warmup_steps        1000
    --region_map_path     "$REGION_MAP"
    --router_type         mlp
    --router_temp         2.0
    --lambda_coarse       0.2
    --lambda_balance      0.001
    --alpha_core          0.2
    --alpha_boundary      0.4
    --boundary_tau        0.03
    --boundary_temp       0.01
    --boundary_mode       margin
    --region_warmup_steps 5000
    --adaptive_layers     1
    --adaptive_scale_init 0.05
    --adaptive_threshold  0.5
    --eval_interval       500
    --save_interval       5000
    --log_interval        100
    --seed                42
    --device              cuda
)

# ─────────────────────────────────────────────────────────────────────────────
# RUN 1 — repr_region_boundary_adaptivedepth (real region map)
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RUN 1: repr_region_boundary_adaptivedepth"
echo "  adaptive_layers=1  adaptive_scale_init=0.05  adaptive_threshold=0.5"
echo "  boundary_tau=0.03  boundary_temp=0.01  → sparse hard gate ~0.25-0.30"
echo "  output_dir: $REAL_DIR"
echo "=============================="

if [[ -f "${REAL_DIR}/final_summary.json" && "${FORCE_RUN:-0}" != "1" ]]; then
    echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
else
    mkdir -p "$REAL_DIR"
    python train_region_lm.py \
        "${COMMON_ARGS[@]}" \
        --mode       repr_region_boundary_adaptivedepth \
        --output_dir "$REAL_DIR"
fi
echo "  Run 1 done at $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# RUN 2 — random_repr_region_boundary_adaptivedepth (null hypothesis)
# Same architecture, randomized region map.
# Any gain here = extra compute benefit; not from real region structure.
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RUN 2: random_repr_region_boundary_adaptivedepth (null hypothesis)"
echo "  Same config; region map randomized — controls for extra compute"
echo "  output_dir: $RAND_DIR"
echo "=============================="

if [[ -f "${RAND_DIR}/final_summary.json" && "${FORCE_RUN:-0}" != "1" ]]; then
    echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
else
    mkdir -p "$RAND_DIR"
    python train_region_lm.py \
        "${COMMON_ARGS[@]}" \
        --mode       random_repr_region_boundary_adaptivedepth \
        --output_dir "$RAND_DIR"
fi
echo "  Run 2 done at $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RESULTS"
echo "=============================="

python3 - <<'PYEOF'
import json, os, math

# ── Hard-coded fallback baselines (overridden by live checkpoints if present) ─
REF_VAL_LM   = 3.9978   # repr_region_reference
BND_VAL_LM   = 4.0144   # calibrated single-path boundary  tau=0.03 temp=0.01
MHYP_VAL_LM  = 4.0519   # calibrated MLP multihyp          tau=0.03 temp=0.01

# Load live baselines from prior runs if available.
# Check both short-run dirs (20k) and long-run dirs (30k).
BASELINE_PATHS = [
    # reference
    ("runs/repr_region_reference/final_summary.json",             "REF_VAL_LM"),
    ("runs_long/repr_region_reference_30k/final_summary.json",    "REF_VAL_LM"),
    # single-path boundary
    ("runs/boundary_tau0p03_temp0p01/final_summary.json",         "BND_VAL_LM"),
    ("runs_long/repr_region_boundary_tau0p03_temp0p01_30k/final_summary.json", "BND_VAL_LM"),
]
for path, varname in BASELINE_PATHS:
    if os.path.isfile(path):
        with open(path) as fh:
            d = json.load(fh)
        v = d.get("val_lm_loss")
        if v is not None:
            if varname == "REF_VAL_LM": REF_VAL_LM = v
            elif varname == "BND_VAL_LM": BND_VAL_LM = v
            elif varname == "MHYP_VAL_LM": MHYP_VAL_LM = v

def load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)

fmt  = lambda v: f"{v:.4f}" if v is not None and not math.isnan(v) else "—"
fmtd = lambda v: f"{v:+.4f}" if v is not None and not math.isnan(v) else "—"
fmts = lambda v: f"{v:.3f}"  if v is not None and not math.isnan(v) else "—"

# ── Load all relevant results ──────────────────────────────────────────────────
RESULT_PATHS = {
    "adaptivedepth (real)":  "runs/repr_region_boundary_adaptivedepth_30k/final_summary.json",
    "adaptivedepth (random)":"runs/random_boundary_adaptivedepth_30k/final_summary.json",
    "seqrefine (real)":      "runs_long/repr_region_boundary_seqrefine_1layer_30k/final_summary.json",
    "seqrefine (random)":    "runs_long/random_boundary_seqrefine_1layer_30k/final_summary.json",
    "branch identity K=4":   "runs_long/repr_region_branch_identity_k4_30k/final_summary.json",
    "branch attention K=4":  "runs_long/repr_region_branchattn_k4_scale0p05_30k/final_summary.json",
}
# Also check short-run dirs for seqrefine if long-run not present
SEQREFINE_ALT = {
    "seqrefine (real)":  "runs/repr_region_boundary_seqrefine_1layer/final_summary.json",
    "seqrefine (random)":"runs/random_boundary_seqrefine_1layer/final_summary.json",
}

results = {}
for label, path in RESULT_PATHS.items():
    r = load(path)
    if r is None and label in SEQREFINE_ALT:
        r = load(SEQREFINE_ALT[label])
    results[label] = r

def lm_of(label): return (results.get(label) or {}).get("val_lm_loss")
def bfrac_of(label): return (results.get(label) or {}).get("val_boundary_frac")

lm_ad   = lm_of("adaptivedepth (real)")
lm_adr  = lm_of("adaptivedepth (random)")
lm_seq  = lm_of("seqrefine (real)")
lm_seqr = lm_of("seqrefine (random)")
lm_idt  = lm_of("branch identity K=4")
lm_bra  = lm_of("branch attention K=4")

# ── Baseline table ─────────────────────────────────────────────────────────────
print()
print("  Baselines")
print(f"    repr_region_reference          val_lm = {REF_VAL_LM:.4f}  [strong-pass target]")
print(f"    calibrated single-path bnd     val_lm = {BND_VAL_LM:.4f}  [partial-pass target]")
print(f"    calibrated MLP multihyp        val_lm = {MHYP_VAL_LM:.4f}")
if lm_seq  is not None: print(f"    seqrefine (real, 30k)          val_lm = {lm_seq:.4f}  [primary comparison]")
if lm_seqr is not None: print(f"    seqrefine (random, 30k)        val_lm = {lm_seqr:.4f}  [extra-compute bar]")
if lm_idt  is not None: print(f"    branch identity K=4 (30k)      val_lm = {lm_idt:.4f}")
if lm_bra  is not None: print(f"    branch attention K=4 (30k)     val_lm = {lm_bra:.4f}")
print()

# ── This experiment ────────────────────────────────────────────────────────────
for label in ("adaptivedepth (real)", "adaptivedepth (random)"):
    r = results.get(label)
    tag = "[REAL]" if "real" in label else "[RANDOM]"
    print(f"  {tag}  {label}")
    if r is None:
        print("    (no result — run may not have completed)")
        print()
        continue
    lm        = r.get("val_lm_loss")
    bnd_frac  = r.get("val_boundary_frac")
    bnd_lm    = r.get("val_boundary_lm")
    core_lm   = r.get("val_core_lm")
    acc1      = r.get("val_coarse_acc1")
    acc4      = r.get("val_coarse_acc4")
    rs        = r.get("val_refine_scale")
    norms     = {k: r.get(f"val_{k}") for k in ("h_norm", "delta_norm",
                                                   "h_final_delta_norm",
                                                   "frac_tokens_refined")}
    print(f"    val_lm              = {fmt(lm)}")
    if lm is not None:
        print(f"    Δ vs reference      = {fmtd(lm - REF_VAL_LM)}")
        print(f"    Δ vs single-path    = {fmtd(lm - BND_VAL_LM)}")
        if lm_seq is not None:
            print(f"    Δ vs seqrefine      = {fmtd(lm - lm_seq)}")
    print(f"    boundary_frac       = {fmts(bnd_frac)}")
    print(f"    frac_tokens_refined = {fmts(norms['frac_tokens_refined'])}",
          "(should ≈ boundary_frac)" if norms["frac_tokens_refined"] is not None else "")
    print(f"    boundary_lm         = {fmt(bnd_lm)}")
    print(f"    core_lm             = {fmt(core_lm)}")
    print(f"    acc@1 (router)      = {fmts(acc1)}")
    print(f"    acc@4 (router)      = {fmts(acc4)}")
    print(f"    refine_scale(tanh)  = {fmts(rs)}")
    if norms["h_norm"]             is not None: print(f"    ||h||               = {norms['h_norm']:.3f}")
    if norms["delta_norm"]         is not None: print(f"    ||delta||           = {norms['delta_norm']:.3f}")
    if norms["h_final_delta_norm"] is not None: print(f"    ||h_final - h_core||= {norms['h_final_delta_norm']:.3f}")
    print()

# ── Verdict ────────────────────────────────────────────────────────────────────
print("── Verdict ──")
print()

if lm_ad is None:
    print("  (no results yet — training may not have completed)")
else:
    # Pass/fail vs baselines
    if lm_ad < REF_VAL_LM:
        print(f"  STRONG PASS  {lm_ad:.4f} < reference {REF_VAL_LM:.4f}")
        print(f"  → Sparse adaptive depth allocation beats the reference model.")
    elif lm_ad < BND_VAL_LM:
        print(f"  PARTIAL PASS {lm_ad:.4f} < single-path boundary {BND_VAL_LM:.4f}")
        print(f"  → Adaptive depth improves over calibrated gate alone.")
    elif lm_ad < MHYP_VAL_LM:
        print(f"  WEAK PASS    {lm_ad:.4f} < MLP multihyp {MHYP_VAL_LM:.4f}")
        print(f"  → Sparse computation beats branch manipulation, but not single-path boundary.")
    else:
        print(f"  FAIL         {lm_ad:.4f} >= MLP multihyp {MHYP_VAL_LM:.4f}")
        print(f"  → Adaptive depth did not help over static branch approaches.")
    print()

    # Primary comparison: hard sparse vs soft global (seqrefine)
    if lm_seq is not None:
        if lm_ad < lm_seq:
            print(f"  SPARSE > SOFT    adaptivedepth={lm_ad:.4f} < seqrefine={lm_seq:.4f}"
                  f"  (Δ={lm_ad-lm_seq:+.4f})")
            print(f"  → Hard sparse masking is more effective than soft global weighting.")
        else:
            print(f"  SPARSE ≤ SOFT    adaptivedepth={lm_ad:.4f} >= seqrefine={lm_seq:.4f}"
                  f"  (Δ={lm_ad-lm_seq:+.4f})")
            print(f"  → Soft global weighting matches or beats hard sparse masking.")
            if abs(lm_ad - lm_seq) < 0.002:
                print(f"  → Near-tie: sparse compute is as efficient as soft at this scale.")
        print()

    # Random control: is gain from real region geometry or just extra compute?
    if lm_adr is not None:
        delta_vs_random = lm_adr - lm_ad
        if delta_vs_random > 0.01:
            print(f"  REAL REGION GAIN   real={lm_ad:.4f} << random={lm_adr:.4f}"
                  f"  (Δ={delta_vs_random:+.4f})")
            print(f"  → Sparse compute exploits real region structure, not just extra FLOPs.")
        elif delta_vs_random > 0.002:
            print(f"  MARGINAL GAIN      real={lm_ad:.4f} < random={lm_adr:.4f}"
                  f"  (Δ={delta_vs_random:+.4f})")
            print(f"  → Small region-structure benefit over pure extra compute.")
        else:
            print(f"  EXTRA-COMPUTE ONLY real≈random  (Δ={delta_vs_random:+.4f})")
            print(f"  → Hard gate selects tokens without real region signal advantage.")
        print()

    # Sparsity check: did the hard threshold actually select ~boundary_frac tokens?
    r_ad = results.get("adaptivedepth (real)") or {}
    frac_ref  = r_ad.get("val_boundary_frac")
    frac_tok  = r_ad.get("val_frac_tokens_refined")
    if frac_ref is not None and frac_tok is not None:
        ratio = frac_tok / frac_ref if frac_ref > 0 else float("nan")
        if 0.8 <= ratio <= 1.2:
            print(f"  SPARSITY OK        frac_tokens_refined={frac_tok:.3f} ≈ boundary_frac={frac_ref:.3f}")
        elif ratio < 0.8:
            print(f"  SPARSITY TOO LOW   frac_tokens_refined={frac_tok:.3f} << boundary_frac={frac_ref:.3f}")
            print(f"  → adaptive_threshold may be too high; many boundary tokens are not refined.")
        else:
            print(f"  SPARSITY TOO HIGH  frac_tokens_refined={frac_tok:.3f} >> boundary_frac={frac_ref:.3f}")
            print(f"  → adaptive_threshold may be too low; non-boundary tokens are also refined.")
        print()

    # refine_scale activation check
    rs = r_ad.get("val_refine_scale")
    if rs is not None:
        if abs(rs) < 0.01:
            print(f"  WARN  refine_scale≈0 ({rs:.4f}) — adaptive refiner barely activated")
        else:
            print(f"  OK    refine_scale={rs:.4f} — adaptive refiner is active")
        print()

# ── Comparison table ───────────────────────────────────────────────────────────
print("── Comparison table ──")
print()
print(f"  {'Model':<38}  {'val_lm':>8}  {'Δref':>8}  {'bnd_frac':>9}")
print(f"  {'-'*38}  {'-'*8}  {'-'*8}  {'-'*9}")

all_rows = [
    ("repr_region_reference",      REF_VAL_LM,  None),
    ("boundary single-path",       BND_VAL_LM,  None),
    ("seqrefine (real)",           lm_seq,       bfrac_of("seqrefine (real)")),
    ("seqrefine (random)",         lm_seqr,      bfrac_of("seqrefine (random)")),
    ("adaptivedepth (real)",       lm_ad,        bfrac_of("adaptivedepth (real)")),
    ("adaptivedepth (random)",     lm_adr,       bfrac_of("adaptivedepth (random)")),
    ("branch identity K=4",        lm_idt,       bfrac_of("branch identity K=4")),
    ("branch attention K=4",       lm_bra,       bfrac_of("branch attention K=4")),
]

for label, lm, bf in all_rows:
    delta = (lm - REF_VAL_LM) if lm is not None else None
    print(f"  {label:<38}  {fmt(lm):>8}  {fmtd(delta):>8}  {fmts(bf):>9}")

print()

PYEOF

echo ""
echo "Done at $(date)"
