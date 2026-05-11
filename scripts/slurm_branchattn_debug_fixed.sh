#!/bin/bash
#SBATCH --job-name=branchattn_dbg
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Debug run following branchattn val_lm=5.3169 failure.
#
# Root causes identified:
#   1. Missing _init_weights → embeddings init at std≈1.0 → huge logits → bad training
#   2. Double zero-init: zero-init projections + branch_scale=0 → dead gradients
#      (grad(branch_scale) = tanh'(0) * refined_delta = 1 * 0 = 0 forever)
#   3. Wrong router (bypassed _make_router_head, missing LayerNorm+Dropout)
#   4. Wrong region_mlp (d_region×d_region vs d_region×(2*d_region))
#
# Fixed in this run:
#   - _init_weights applied to all modules (std=0.02)
#   - zero-init projections removed; branch_scale_init=0.05 (tanh≈0.05, small but live)
#   - router/region_mlp now match repr_region_multihyp exactly
#   - use_branch_attn flag controls identity vs full-attention branch path
#
# Run 1: repr_region_branch_identity  — pure weighted sum of K hypothesis hiddens
#         (no refiner). Isolates: is the branch construction/merge itself useful?
# Run 2: repr_region_branchattn fixed — same construction + branch attention refiner.
#
# Baselines for comparison:
#   repr_region_reference              val_lm = 3.9978
#   calibrated single-path boundary    val_lm = 4.0144
#   calibrated MLP multihyp            val_lm = 4.0519
#   original branchattn BROKEN         val_lm = 5.3169  (all three bugs above)

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
IDENTITY_DIR=runs/repr_region_branch_identity_k4_tau0p03_temp0p01
BRANCHATTN_DIR=runs/repr_region_branchattn_fixed_k4_tau0p03_temp0p01

if [[ ! -f "$REGION_MAP" ]]; then
    echo "ERROR: region map missing: $REGION_MAP" >&2; exit 1
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
# RUN 1 — Branch Identity (diagnostic)
# Purpose: weighted sum of K hypotheses, NO refiner.
# If this is bad → branch construction/merge is the problem.
# If this is similar to single-path boundary → refiner is the question.
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RUN 1: repr_region_branch_identity"
echo "  K=4  tau=0.03  temp=0.01  (no branch attention)"
echo "  output_dir: $IDENTITY_DIR"
echo "=============================="

if [[ -f "${IDENTITY_DIR}/final_summary.json" && "${FORCE_RUN:-0}" != "1" ]]; then
    echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
else
    mkdir -p "$IDENTITY_DIR"
    python train_region_lm.py \
        --mode                repr_region_branch_identity \
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
        --output_dir          "$IDENTITY_DIR"
fi
echo "  Run 1 done at $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# RUN 2 — Branch Attention Fixed
# Purpose: branch identity path + attention refiner over K dim.
# branch_scale_init=0.05 → tanh(0.05)≈0.05 (small live contribution at init).
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RUN 2: repr_region_branchattn (fixed)"
echo "  K=4  tau=0.03  temp=0.01  branch_scale_init=0.05"
echo "  output_dir: $BRANCHATTN_DIR"
echo "=============================="

if [[ -f "${BRANCHATTN_DIR}/final_summary.json" && "${FORCE_RUN:-0}" != "1" ]]; then
    echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
else
    mkdir -p "$BRANCHATTN_DIR"
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
        --branch_scale_init   0.05 \
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
        --output_dir          "$BRANCHATTN_DIR"
fi
echo "  Run 2 done at $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "Results"
echo "=============================="

python3 - <<'PYEOF'
import json, os, math

REF_VAL_LM       = 3.9978
CAL_BND_LM       = 4.0144
CAL_MHYP_MLP_LM  = 4.0519
BROKEN_BRANCHATTN = 5.3169

# Load live baselines if available
for path, varname in [
    ("runs/repr_region_reference/final_summary.json",                       "REF_VAL_LM"),
    ("runs/boundary_tau0p03_temp0p01/final_summary.json",                   "CAL_BND_LM"),
    ("runs/repr_region_multihyp_k4_tau0p03_temp0p01/final_summary.json",   "CAL_MHYP_MLP_LM"),
]:
    if os.path.isfile(path):
        with open(path) as fh:
            d = json.load(fh)
        v = d.get("val_lm_loss")
        if v is not None:
            if varname == "REF_VAL_LM":        REF_VAL_LM       = v
            elif varname == "CAL_BND_LM":      CAL_BND_LM       = v
            elif varname == "CAL_MHYP_MLP_LM": CAL_MHYP_MLP_LM = v

def load_result(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)

fmt  = lambda v: f"{v:.4f}" if v is not None and not math.isnan(v) else "—"
fmtd = lambda v: f"{v:+.4f}" if v is not None and not math.isnan(v) else "—"
fmts = lambda v: f"{v:.3f}"  if v is not None and not math.isnan(v) else "—"

IDENTITY_DIR    = "runs/repr_region_branch_identity_k4_tau0p03_temp0p01"
BRANCHATTN_DIR  = "runs/repr_region_branchattn_fixed_k4_tau0p03_temp0p01"

d_id  = load_result(f"{IDENTITY_DIR}/final_summary.json")
d_bra = load_result(f"{BRANCHATTN_DIR}/final_summary.json")

def extract(d):
    if d is None:
        return {k: None for k in ["lm","bnd_frac","bnd_lm","core_lm","br_scale",
                                   "h_norm","h_core_dn","h_bnd_dn","h_fin_dn","ref_dn"]}
    return {
        "lm":       d.get("val_lm_loss"),
        "bnd_frac": d.get("val_boundary_frac"),
        "bnd_lm":   d.get("val_boundary_lm"),
        "core_lm":  d.get("val_core_lm"),
        "br_scale": d.get("val_branch_scale"),
        "h_norm":   d.get("val_h_norm"),
        "h_core_dn":d.get("val_h_core_delta_norm"),
        "h_bnd_dn": d.get("val_h_bnd_delta_norm"),
        "h_fin_dn": d.get("val_h_final_delta_norm"),
        "ref_dn":   d.get("val_refined_delta_norm"),
    }

ri = extract(d_id)
rb = extract(d_bra)

print("  Baselines")
print(f"    repr_region_reference              val_lm = {fmt(REF_VAL_LM)}  [strong]")
print(f"    calibrated single-path boundary    val_lm = {fmt(CAL_BND_LM)}  [partial]")
print(f"    calibrated MLP multihyp            val_lm = {fmt(CAL_MHYP_MLP_LM)}  [weak]")
print(f"    old branchattn BROKEN              val_lm = {fmt(BROKEN_BRANCHATTN)}")
print()

for tag, r, use_attn in [("branch_identity", ri, False), ("branchattn_fixed", rb, True)]:
    print(f"  {tag}")
    if r["lm"] is None:
        print("    (no result — run may not have completed)")
    else:
        print(f"    val_lm          = {fmt(r['lm'])}  Δref={fmtd(r['lm'] - REF_VAL_LM if r['lm'] else None)}")
        print(f"    boundary_frac   = {fmts(r['bnd_frac'])}")
        print(f"    boundary_lm     = {fmt(r['bnd_lm'])}")
        print(f"    core_lm         = {fmt(r['core_lm'])}")
        if use_attn:
            print(f"    branch_scale    = {fmts(r['br_scale'])}")
        if r["h_norm"] is not None:
            print(f"    ||h||           = {fmts(r['h_norm'])}")
            print(f"    ||Δh_core||     = {fmts(r['h_core_dn'])}")
            print(f"    ||Δh_bnd||      = {fmts(r['h_bnd_dn'])}")
            print(f"    ||Δh_final||    = {fmts(r['h_fin_dn'])}")
        if use_attn and r["ref_dn"] is not None:
            print(f"    ||refined_delta||= {fmts(r['ref_dn'])}")
    print()

# ── Interpretation ─────────────────────────────────────────────────────────────
print("── Interpretation ──")

lm_id  = ri["lm"]
lm_bra = rb["lm"]

if lm_id is None and lm_bra is None:
    print("  (no results yet)")
    raise SystemExit(0)

if lm_id is not None:
    if lm_id >= CAL_MHYP_MLP_LM:
        print(f"  IDENTITY BAD ({fmt(lm_id)}) → branch construction/merge itself is harmful")
        print(f"  → K-hypothesis weighted sum hurts vs single soft-mixture path.")
    elif lm_id >= CAL_BND_LM:
        print(f"  IDENTITY WEAK ({fmt(lm_id)}) → branch merge is neutral/marginal; refiner needed")
    elif lm_id >= REF_VAL_LM:
        print(f"  IDENTITY OK ({fmt(lm_id)}) → branch merge helps over single-path boundary")
    else:
        print(f"  IDENTITY GOOD ({fmt(lm_id)}) → weighted K-hyp merge beats reference alone")
    print()

if lm_bra is not None and lm_id is not None:
    if lm_bra < lm_id:
        print(f"  ATTN HELPS  ({fmt(lm_bra)} < identity {fmt(lm_id)})")
        print(f"  → hypothesis interaction improves over pure weighted sum")
    else:
        print(f"  ATTN NO HELP  ({fmt(lm_bra)} ≥ identity {fmt(lm_id)})")
        print(f"  → branch attention does not help over identity merge")

if lm_bra is not None:
    if lm_bra < REF_VAL_LM:
        print(f"  STRONG PASS  {fmt(lm_bra)} beats reference ({fmt(REF_VAL_LM)})")
    elif lm_bra < CAL_BND_LM:
        print(f"  PARTIAL PASS {fmt(lm_bra)} < single-path ({fmt(CAL_BND_LM)})")
    elif lm_bra < CAL_MHYP_MLP_LM:
        print(f"  WEAK PASS    {fmt(lm_bra)} < MLP multihyp ({fmt(CAL_MHYP_MLP_LM)})")
    else:
        print(f"  FAIL         {fmt(lm_bra)} ≥ MLP multihyp ({fmt(CAL_MHYP_MLP_LM)})")

# Sanity check: branch_scale activation
if lm_bra is not None and rb["br_scale"] is not None:
    bs = rb["br_scale"]
    if abs(bs) < 0.01:
        print(f"  WARN  branch_scale≈0 ({bs:.4f}) — refiner still dead; check init fix")
    else:
        print(f"  OK    branch_scale={bs:.4f} — refiner is active")

PYEOF

echo ""
echo "Done at $(date)"
