#!/bin/bash
#SBATCH --job-name=geom_evidence_fix
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Geometry + Evidence Interaction Diagnostic V2 — FIXED
#
# Changes vs V1:
#   - Robust unembedding loading (more key aliases, suffix matching, raises on failure)
#   - base_gap computed from saved logits before U block (always finite for gip rows)
#   - NaN guard: raises RuntimeError if any required field has nan_rate > 0.001
#   - Oracle debug: prints 20 rows if oracle_wins=0 AND oracle_wrong_steals=0
#   - geometry_validity_audit.json written before report
#   - unembedding_audit.json written explicitly
#   - Report starts with VALIDITY STATUS: PASS/FAIL
#   - No interpretive conclusions written if geometry is invalid
#   - Q7 crowding verdict tied to actual measured crowding level
#
# GOLD USED ONLY FOR DIAGNOSTICS/METRICS. NOT A TRAINING RUN. NOT A NEW EXPERT.
#
# Outputs:
#   runs/geometry_diagnostics/evidence_interaction_v2_fixed/
#     config.json
#     unembedding_audit.json
#     hidden_state_audit.json
#     geometry_validity_audit.json
#     row_geometry_metrics.csv
#     candidate_geometry_metrics.csv
#     correction_crowding_summary.csv
#     target_direction_simulation.csv
#     near_miss_bucket_summary.csv
#     geometry_slice_summary.csv
#     expert_score_geometry.csv
#     expert_method_comparison.csv
#     expert_agreement_summary.csv
#     expert_failure_explanations.md
#     geometry_evidence_interaction_report.md
#     examples_*.md  (6 example files)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD
mkdir -p logs

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"
OUTPUT_DIR="runs/geometry_diagnostics/evidence_interaction_v2_fixed"

# Optional expert output dirs — script continues gracefully if any are absent
POINTER_DIR="runs/pointer_sentinel_v1/pointer_help_slices_v1"
FUZZY_DIR="runs/phrase_memory_v1/fuzzy_phrase_recall"
FORMAT_DIR="runs/format_memory_v1/format_syntax_diagnostic_v1"
DETAIL_DIR="runs/entity_number_memory_v1/learned_detail_resolver_v1"
PASSAGE_DIR="runs/passage_memory_v1/passage_recall_dedup_v1/eval"
PATH_DIR="runs/path_oracle_v1"

echo "========================================================"
echo " Geometry + Evidence Interaction Diagnostic V2 — FIXED"
echo " val_dir:    ${VAL_DIR}"
echo " small_ckpt: ${SMALL_CKPT}"
echo " output_dir: ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

echo "[preflight] Checking required inputs..."

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: val_dir not found: $VAL_DIR"; exit 1
fi
N=$(find "$VAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then
    echo "ERROR: no shard_*.pt in $VAL_DIR"; exit 1
fi
echo "  [OK] $VAL_DIR  ($N shards)"

# Checkpoint is required for geometry in this run.
if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: checkpoint not found: $SMALL_CKPT"
    echo "       Geometry requires the unembedding matrix. Cannot proceed."
    exit 1
fi
echo "  [OK] $SMALL_CKPT"

T2R_ARG=""
if [ -f "$TOKEN_TO_REGION" ]; then
    T2R_ARG="--token_to_region $TOKEN_TO_REGION"
    echo "  [OK] $TOKEN_TO_REGION"
else
    echo "  [WARN] token_to_region not found — region slices disabled"
fi

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map $SUPER_MAP"
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found — superregion slices disabled"
fi

echo ""
echo "[preflight] Checking optional expert dirs..."

PTR_ARG="";  FZ_ARG="";  FMT_ARG="";  DT_ARG="";  PA_ARG="";  PO_ARG=""

[ -d "$POINTER_DIR" ] && { PTR_ARG="--pointer_dir $POINTER_DIR";  echo "  [OK] pointer:  $POINTER_DIR"; } \
                       || echo "  [skip] pointer_dir absent: $POINTER_DIR"
[ -d "$FUZZY_DIR"   ] && { FZ_ARG="--fuzzy_dir $FUZZY_DIR";       echo "  [OK] fuzzy:    $FUZZY_DIR"; } \
                       || echo "  [skip] fuzzy_dir absent: $FUZZY_DIR"
[ -d "$FORMAT_DIR"  ] && { FMT_ARG="--format_dir $FORMAT_DIR";    echo "  [OK] format:   $FORMAT_DIR"; } \
                       || echo "  [skip] format_dir absent: $FORMAT_DIR"
[ -d "$DETAIL_DIR"  ] && { DT_ARG="--detail_dir $DETAIL_DIR";     echo "  [OK] detail:   $DETAIL_DIR"; } \
                       || echo "  [skip] detail_dir absent: $DETAIL_DIR"
[ -d "$PASSAGE_DIR" ] && { PA_ARG="--passage_dir $PASSAGE_DIR";   echo "  [OK] passage:  $PASSAGE_DIR"; } \
                       || echo "  [skip] passage_dir absent: $PASSAGE_DIR"
[ -d "$PATH_DIR"    ] && { PO_ARG="--path_dir $PATH_DIR";         echo "  [OK] path:     $PATH_DIR"; } \
                       || echo "  [skip] path_dir absent: $PATH_DIR"

echo ""
echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/eval_geometry_evidence_interaction_v2.py
echo "  [OK] scripts/eval_geometry_evidence_interaction_v2.py compiles"
bash -n scripts/slurm_eval_geometry_evidence_interaction_v2_fixed.sh
echo "  [OK] slurm_eval_geometry_evidence_interaction_v2_fixed.sh syntax"

echo ""
echo "[preflight] All checks passed."
echo ""

mkdir -p "${OUTPUT_DIR}"

echo "[launch] Starting Geometry + Evidence Interaction Diagnostic V2 (FIXED)..."
echo "  GOLD USED FOR DIAGNOSTICS/METRICS ONLY — NOT DEPLOYABLE ORACLE SECTIONS"
echo ""

python scripts/eval_geometry_evidence_interaction_v2.py \
    --val_dir                "${VAL_DIR}" \
    --small_ckpt             "${SMALL_CKPT}" \
    $T2R_ARG \
    $SUPER_ARG \
    $PTR_ARG \
    $FZ_ARG \
    $FMT_ARG \
    $DT_ARG \
    $PA_ARG \
    $PO_ARG \
    --output_dir             "${OUTPUT_DIR}" \
    --candidate_pool_size    256 \
    --topm_eval              64 \
    --example_count          50 \
    --seed                   42

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: diagnostic exited with code $EXIT"
    exit $EXIT
fi

echo ""
echo "========================================================"
echo " Geometry + Evidence Interaction Diagnostic V2 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT_DIR}/geometry_evidence_interaction_report.md"
echo "  ${OUTPUT_DIR}/geometry_validity_audit.json"
echo "  ${OUTPUT_DIR}/unembedding_audit.json"
echo "  ${OUTPUT_DIR}/row_geometry_metrics.csv"
echo "  ${OUTPUT_DIR}/correction_crowding_summary.csv"
echo "  ${OUTPUT_DIR}/near_miss_bucket_summary.csv"
echo "  ${OUTPUT_DIR}/expert_method_comparison.csv"
echo ""

if [ ! -f "${OUTPUT_DIR}/geometry_validity_audit.json" ]; then
    echo "ERROR: geometry_validity_audit.json not found — diagnostic may have failed"
    exit 1
fi

# Surface validity status from the audit
python - <<'PYEOF'
import json, sys, os
d = "runs/geometry_diagnostics/evidence_interaction_v2_fixed"
try:
    va = json.load(open(os.path.join(d, "geometry_validity_audit.json")))
    status = "PASS" if va.get("validity_ok") else "FAIL"
    print(f"[result] VALIDITY STATUS: {status}")
    print(f"  unembedding_source: {va.get('unembedding_source','?')}")
    print(f"  U_shape:            {va.get('U_shape','?')}")
    print(f"  d_model:            {va.get('d_model','?')}")
    print(f"  gip_n:              {va.get('gip_n',0):,} / {va.get('N_val',0):,}")
    print(f"  oracle_wins:        {va.get('n_oracle_wins',0):,}")
    print(f"  oracle_steals:      {va.get('n_oracle_steals',0):,}")
    for fname, nr in va.get("nan_rates", {}).items():
        flag = "  <-- PROBLEM" if isinstance(nr, float) and nr > 0.001 else ""
        print(f"  nan_rate[{fname}] = {nr:.5f}{flag}")
    for msg in va.get("validity_messages", []):
        print(f"  [msg] {msg}")
except Exception as e:
    print(f"[result] Could not parse validity audit: {e}", file=sys.stderr)
PYEOF

if [ ! -f "${OUTPUT_DIR}/geometry_evidence_interaction_report.md" ]; then
    echo "ERROR: report not found — diagnostic may have failed"
    exit 1
fi
echo "[done] Report: ${OUTPUT_DIR}/geometry_evidence_interaction_report.md"
