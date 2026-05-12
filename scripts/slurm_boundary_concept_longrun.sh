#!/bin/bash
#SBATCH --job-name=bnd_concepts
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Long-run boundary concept comparison.
# All runs share identical config, seed, region map.
#
# Primary question:
#   Do boundary tokens need contextual sequence-level computation
#   rather than static region-branch manipulation?
#
# Run matrix (30k steps each):
#   1. repr_region_reference          — strong baseline
#   2. repr_region_boundary           — calibrated gate, adaptive alpha only
#   3. repr_region_branch_identity    — top-K branch weighted sum, no refiner
#   4. repr_region_branchattn         — top-K branch + cross-branch attention
#   5. repr_region_boundary_seqrefine — gate-weighted extra causal seq block
#   6. random_repr_region_boundary_seqrefine — random-map control for #5
#
# Fixed config:
#   dataset=wikitext-103-raw-v1  seq_len=256  batch=32  d_model=384
#   n_layer=6  n_head=6  d_ff=1536  steps=30000  seed=42
#   boundary_tau=0.03  boundary_temp=0.01  alpha_core=0.2  alpha_boundary=0.4

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
OUTBASE=runs_long

if [[ ! -f "$REGION_MAP" ]]; then
    echo "ERROR: region map missing: $REGION_MAP" >&2; exit 1
fi
if [[ ! -f "train_region_lm.py" ]]; then
    echo "ERROR: train_region_lm.py not found in $PWD" >&2; exit 1
fi

mkdir -p logs "$OUTBASE"

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PREFLIGHT — verify all boundary models before committing 18h of GPU
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "PREFLIGHT CHECKS"
echo "=============================="

PREFLIGHT_MODES=(
    repr_region_boundary
    repr_region_branch_identity
    repr_region_branchattn
    repr_region_boundary_seqrefine
    random_repr_region_boundary_seqrefine
)

PREFLIGHT_FAILED=0
for MODE in "${PREFLIGHT_MODES[@]}"; do
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
    --eval_interval       500
    --save_interval       5000
    --log_interval        100
    --seed                42
    --device              cuda
)

train_run() {
    local LABEL="$1"
    local OUT_DIR="$2"
    shift 2
    local EXTRA_ARGS=("$@")

    echo "=============================="
    echo "RUN: $LABEL"
    echo "  output_dir: $OUT_DIR"
    echo "=============================="

    if [[ -f "${OUT_DIR}/final_summary.json" && "${FORCE_RUN:-0}" != "1" ]]; then
        echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
        return 0
    fi

    mkdir -p "$OUT_DIR"
    python train_region_lm.py \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        --output_dir "$OUT_DIR"
    echo "  Done at $(date)"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# RUN 1 — repr_region reference
# ─────────────────────────────────────────────────────────────────────────────
train_run "repr_region_reference (30k)" \
    "${OUTBASE}/repr_region_reference_30k" \
    --mode       repr_region \
    --base_alpha 0.2

# ─────────────────────────────────────────────────────────────────────────────
# RUN 2 — calibrated boundary single-path
# ─────────────────────────────────────────────────────────────────────────────
train_run "calibrated boundary single-path (30k)" \
    "${OUTBASE}/repr_region_boundary_tau0p03_temp0p01_30k" \
    --mode repr_region_boundary

# ─────────────────────────────────────────────────────────────────────────────
# RUN 3 — branch identity diagnostic
# ─────────────────────────────────────────────────────────────────────────────
train_run "branch identity K=4 (30k)" \
    "${OUTBASE}/repr_region_branch_identity_k4_30k" \
    --mode   repr_region_branch_identity \
    --hyp_k  4

# ─────────────────────────────────────────────────────────────────────────────
# RUN 4 — fixed branch attention
# ─────────────────────────────────────────────────────────────────────────────
train_run "branch attention fixed K=4 (30k)" \
    "${OUTBASE}/repr_region_branchattn_k4_scale0p05_30k" \
    --mode                repr_region_branchattn \
    --hyp_k               4 \
    --branch_attn_heads   4 \
    --branch_attn_layers  1 \
    --branch_attn_dropout 0.1 \
    --branch_scale_init   0.05

# ─────────────────────────────────────────────────────────────────────────────
# RUN 5 — boundary sequence refiner (key new concept)
# ─────────────────────────────────────────────────────────────────────────────
train_run "boundary seqrefine 1-layer (30k)" \
    "${OUTBASE}/repr_region_boundary_seqrefine_1layer_30k" \
    --mode                    repr_region_boundary_seqrefine \
    --boundary_refine_layers  1 \
    --boundary_refine_gamma   0.5

# ─────────────────────────────────────────────────────────────────────────────
# RUN 6 — random control for seqrefine
# ─────────────────────────────────────────────────────────────────────────────
train_run "random seqrefine control (30k)" \
    "${OUTBASE}/random_boundary_seqrefine_1layer_30k" \
    --mode                    random_repr_region_boundary_seqrefine \
    --boundary_refine_layers  1 \
    --boundary_refine_gamma   0.5

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RESULTS"
echo "=============================="

python3 - <<'PYEOF'
import json, os, math

OUTBASE = "runs_long"

RUNS = [
    ("repr_region_reference_30k",                    "repr_region_reference"),
    ("repr_region_boundary_tau0p03_temp0p01_30k",    "boundary single-path"),
    ("repr_region_branch_identity_k4_30k",           "branch identity K=4"),
    ("repr_region_branchattn_k4_scale0p05_30k",      "branch attention K=4"),
    ("repr_region_boundary_seqrefine_1layer_30k",    "boundary seqrefine"),
    ("random_boundary_seqrefine_1layer_30k",         "random seqrefine"),
]

def load(d):
    p = f"{OUTBASE}/{d}/final_summary.json"
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)

fmt  = lambda v: f"{v:.4f}" if v is not None else "—"
fmtd = lambda v: f"{v:+.4f}" if v is not None else "—"
fmts = lambda v: f"{v:.3f}"  if v is not None else "—"

results = {}
for d, label in RUNS:
    r = load(d)
    results[label] = r

ref_lm = results.get("repr_region_reference", {}) or {}
ref_lm = ref_lm.get("val_lm_loss") if ref_lm else None

print()
print(f"  {'Model':<40}  {'val_lm':>8}  {'Δref':>8}  {'bnd_frac':>9}  "
      f"{'bnd_lm':>8}  {'core_lm':>8}  {'acc@1':>6}")
print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*6}")

for d, label in RUNS:
    r = results.get(label)
    if r is None:
        print(f"  {label:<40}  {'—':>8}  {'—':>8}  {'—':>9}  {'—':>8}  {'—':>8}  {'—':>6}")
        continue
    lm       = r.get("val_lm_loss")
    bnd_frac = r.get("val_boundary_frac")
    bnd_lm   = r.get("val_boundary_lm")
    core_lm  = r.get("val_core_lm")
    acc1     = r.get("val_coarse_acc1")
    delta    = (lm - ref_lm) if (lm is not None and ref_lm is not None) else None

    print(f"  {label:<40}  {fmt(lm):>8}  {fmtd(delta):>8}  {fmts(bnd_frac):>9}  "
          f"{fmt(bnd_lm):>8}  {fmt(core_lm):>8}  {fmts(acc1):>6}")

print()

# ── Extra details for seqrefine ────────────────────────────────────────────────
for key in ("boundary seqrefine", "random seqrefine"):
    r = results.get(key)
    if r is None:
        continue
    rs = r.get("val_refine_scale")
    h_norm = r.get("val_h_norm")
    seq_dn = r.get("val_h_seq_delta_norm")
    fin_dn = r.get("val_h_final_delta_norm")
    if any(v is not None for v in [rs, h_norm, seq_dn, fin_dn]):
        print(f"  {key} extra:")
        if rs       is not None: print(f"    refine_scale(tanh)   = {rs:.4f}")
        if h_norm   is not None: print(f"    ||h||                = {h_norm:.3f}")
        if seq_dn   is not None: print(f"    ||h_seq - h_core||   = {seq_dn:.3f}")
        if fin_dn   is not None: print(f"    ||h_final - h_core|| = {fin_dn:.3f}")
        print()

# ── Verdict ─────────────────────────────────────────────────────────────────────
print("── Verdict ──")
print()

ref = results.get("repr_region_reference", {}) or {}
bnd = results.get("boundary single-path",  {}) or {}
idt = results.get("branch identity K=4",   {}) or {}
bra = results.get("branch attention K=4",  {}) or {}
seq = results.get("boundary seqrefine",    {}) or {}
rnd = results.get("random seqrefine",      {}) or {}

def lm_of(d): return d.get("val_lm_loss") if d else None

lm_ref = lm_of(ref); lm_bnd = lm_of(bnd); lm_idt = lm_of(idt)
lm_bra = lm_of(bra); lm_seq = lm_of(seq); lm_rnd = lm_of(rnd)

if lm_seq is not None and lm_ref is not None:
    if lm_seq < lm_ref:
        print(f"  STRONG PASS  seqrefine={lm_seq:.4f} < reference={lm_ref:.4f}")
        print(f"  → Gate-weighted contextual computation exploits region structure.")
    elif lm_bnd is not None and lm_seq < lm_bnd:
        print(f"  PARTIAL PASS seqrefine={lm_seq:.4f} < boundary single-path={lm_bnd:.4f}")
        print(f"  → Extra sequence compute helps over adaptive alpha alone.")
    elif lm_bra is not None and lm_seq < lm_bra:
        print(f"  WEAK PASS    seqrefine={lm_seq:.4f} < branch attention={lm_bra:.4f}")
        print(f"  → Contextual sequence compute beats static branch manipulation.")
    elif lm_idt is not None and lm_seq < lm_idt:
        print(f"  WEAK PASS    seqrefine={lm_seq:.4f} < branch identity={lm_idt:.4f}")
    else:
        print(f"  FAIL         seqrefine={lm_seq:.4f}")
        print(f"  → Contextual sequence refinement did not help boundary tokens.")

# Branch diagnostics
print()
if lm_idt is not None and lm_bnd is not None:
    if lm_idt >= lm_bnd:
        print(f"  BRANCH IDENTITY BAD  ({lm_idt:.4f} ≥ single-path {lm_bnd:.4f})")
        print(f"  → Static top-K branch construction/merge is itself harmful.")
    else:
        print(f"  BRANCH IDENTITY OK   ({lm_idt:.4f} < single-path {lm_bnd:.4f})")

if lm_bra is not None and lm_idt is not None:
    if lm_bra < lm_idt:
        print(f"  BRANCH ATTN HELPS    ({lm_bra:.4f} < identity {lm_idt:.4f})")
    else:
        print(f"  BRANCH ATTN NO HELP  ({lm_bra:.4f} ≥ identity {lm_idt:.4f})")

# Random control: is the seqrefine gain from real regions or just extra compute?
if lm_seq is not None and lm_rnd is not None:
    delta_vs_random = lm_rnd - lm_seq
    if delta_vs_random > 0.01:
        print(f"  REAL REGION GAIN     seqrefine={lm_seq:.4f} < random={lm_rnd:.4f}"
              f"  (Δ={delta_vs_random:+.4f})")
        print(f"  → Gain comes from real region structure, not just extra compute.")
    else:
        print(f"  EXTRA-COMPUTE ONLY   seqrefine≈random  (Δ={delta_vs_random:+.4f})")
        print(f"  → Extra causal block helps regardless of region quality.")

# Best model summary
print()
print("── Best model summary ──")
all_results = {
    "repr_region_reference":  lm_ref,
    "boundary single-path":   lm_bnd,
    "branch identity K=4":    lm_idt,
    "branch attention K=4":   lm_bra,
    "boundary seqrefine":     lm_seq,
    "random seqrefine":       lm_rnd,
}
valid = {k: v for k, v in all_results.items() if v is not None}
if valid:
    best_k = min(valid, key=valid.__getitem__)
    print(f"  Best overall:       {best_k}  val_lm={valid[best_k]:.4f}")
    if lm_ref:
        best_delta = valid[best_k] - lm_ref
        print(f"  Δ vs reference:     {best_delta:+.4f}")

# Boundary LM comparison (who resolves boundary tokens best?)
bnd_lms = {}
for k in ("boundary single-path", "branch identity K=4", "branch attention K=4",
          "boundary seqrefine"):
    r = results.get(k)
    if r and r.get("val_boundary_lm"):
        bnd_lms[k] = r["val_boundary_lm"]
if bnd_lms:
    best_bk = min(bnd_lms, key=bnd_lms.__getitem__)
    print(f"  Best boundary_lm:   {best_bk}  boundary_lm={bnd_lms[best_bk]:.4f}")

PYEOF

echo ""
echo "Done at $(date)"
