#!/bin/bash
#SBATCH --job-name=txl_depth_sweep
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# TXL Depth/Capacity Sweep V1
#
# Tests whether deeper / wider Transformer-XL detail memory improves
# candidate selection over Stage 2C (raw token-embedding cross-attention).
#
# Group A — TXL depth/width:
#   A0: L=2, mem_dim=256, resolver=256, hidden=512  (baseline TXL)
#   A1: L=4, mem_dim=256, resolver=256, hidden=512
#   A2: L=6, mem_dim=256, resolver=256, hidden=512
#   A3: L=4, mem_dim=384, resolver=384, hidden=768
#   A4: L=6, mem_dim=384, resolver=384, hidden=768
#
# Group B — parameter-matched token-embedding controls:
#   B0: token backend, hidden=1024
#   B1: token backend, hidden=1408
#   B2: token backend, resolver=384, hidden=1536
#
# Stage 2C ref:
#   apply_rate=0.0798, ctg=0.0084, caw=0.0089, fv_gain=-0.0017
#   selected_gold_given_in_pool=0.2457, applied_precision_ctg=0.1047
# V3 oracle: target_ctg=0.2407, fv_gain=+0.1584
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

STEPS="${STEPS:-3000}"
EVAL_EVERY="${EVAL_EVERY:-500}"

OUT_ROOT="runs/txl_depth_capacity_sweep_v1"
SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"

FAILED_LOG="${OUT_ROOT}/failed_runs.txt"

mkdir -p "${OUT_ROOT}"
mkdir -p logs
rm -f "${FAILED_LOG}"

echo "========================================================"
echo " TXL Depth/Capacity Sweep V1"
echo " OUT_ROOT: ${OUT_ROOT}"
echo " STEPS=${STEPS}  EVAL_EVERY=${EVAL_EVERY}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "[preflight] Checking required inputs..."

if [ ! -f "$SMALL_CKPT" ]; then
    echo "ERROR: backbone checkpoint not found: $SMALL_CKPT"; exit 1
fi
echo "  [OK] $SMALL_CKPT"

for D in "$TRAIN_DIR" "$VAL_DIR"; do
    if [ ! -d "$D" ]; then
        echo "ERROR: shard dir not found: $D"; exit 1
    fi
    N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then
        echo "ERROR: no shard_*.pt in $D"; exit 1
    fi
    echo "  [OK] $D  ($N shards)"
done

if [ ! -f "$TOKEN_TO_REGION" ]; then
    echo "ERROR: token_to_region not found: $TOKEN_TO_REGION"; exit 1
fi
echo "  [OK] $TOKEN_TO_REGION"

SUPER_ARG=""
if [ -f "$SUPER_MAP" ]; then
    SUPER_ARG="--super_map $SUPER_MAP"
    echo "  [OK] $SUPER_MAP  (superregion enabled)"
else
    echo "  [WARN] super_map not found: $SUPER_MAP — superregion disabled"
fi

echo "[preflight] Checking input_ids in shards..."
TRAIN_SHARD=$(find "$TRAIN_DIR" -maxdepth 1 -name 'shard_*.pt' | head -1)
VAL_SHARD=$(find "$VAL_DIR"   -maxdepth 1 -name 'shard_*.pt' | head -1)

for SHARD in "$TRAIN_SHARD" "$VAL_SHARD"; do
    python -c "
import torch, sys
sh = torch.load('$SHARD', map_location='cpu', weights_only=False)
if 'input_ids' not in sh:
    print('ERROR: input_ids missing from shard: $SHARD')
    print('  keys:', list(sh.keys()))
    sys.exit(1)
print('  [OK] input_ids found in $SHARD')
" || exit 1
done

echo "[preflight] Checking Python syntax..."
python -m py_compile scripts/train_txl_detail_resolver_v1.py
python -m py_compile scripts/analyze_txl_depth_capacity_sweep_v1.py
echo "  [OK] Python scripts compile"

echo ""
echo "[preflight] All required inputs present."
echo ""

# ── Shared hyperparameters ─────────────────────────────────────────────────────
SHARED_ARGS=(
    --small_ckpt              "$SMALL_CKPT"
    --train_dir               "$TRAIN_DIR"
    --val_dir                 "$VAL_DIR"
    --token_to_region         "$TOKEN_TO_REGION"
    --output_root             "${OUT_ROOT}"
    --top_k                   256
    --candidate_filter        top_rank
    --candidate_pool_size     32
    --memory_len              128
    --txl_mem_len             64
    --pos_encoding            alibi
    --dropout                 0.1
    --region_emb_dim          64
    --super_emb_dim           32
    --batch_size              128
    --correctable_fraction    0.5
    --lr                      1e-4
    --steps                   "${STEPS}"
    --eval_every              "${EVAL_EVERY}"
    --eval_batch_size         256
    --grad_clip               1.0
    --lambda_selector         1.0
    --lambda_gate             1.0
    --gate_pos_weight         3.0
    --lambda_noharm_gate      1.0
    --gate_margin             1.0
    --lambda_selector_margin  0.25
    --selector_margin         1.0
    --lambda_memory_contrastive 0.0
    --margin_delta            1.0
    --gate_thresholds         0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
    --uncovered_policy        ignore
    --max_train_rows          500000
    --eval_full_vocab
    --amp
    --seed                    42
)

# ── run_one function ───────────────────────────────────────────────────────────
# Args: run_name backend txl_layers txl_heads mem_dim resolver_dim txl_ff_dim num_slots attn_heads hidden_dim
run_one() {
    local RUN_NAME="$1"
    local BACKEND="$2"
    local TXL_LAYERS="$3"
    local TXL_HEADS="$4"
    local MEM_DIM="$5"
    local RESOLVER_DIM="$6"
    local TXL_FF_DIM="$7"
    local NUM_SLOTS="$8"
    local ATTN_HEADS="$9"
    local HIDDEN_DIM="${10}"

    local RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

    if [ -f "${RUN_DIR}/final_metrics.json" ]; then
        echo "[skip] ${RUN_NAME}  — final_metrics.json exists"
        return 0
    fi

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo " RUN: ${RUN_NAME}"
    echo "   backend=${BACKEND}  txl_layers=${TXL_LAYERS}  txl_heads=${TXL_HEADS}"
    echo "   mem_dim=${MEM_DIM}  resolver_dim=${RESOLVER_DIM}  txl_ff_dim=${TXL_FF_DIM}"
    echo "   num_slots=${NUM_SLOTS}  attention_heads=${ATTN_HEADS}  hidden_dim=${HIDDEN_DIM}"
    echo " $(date)"
    echo "────────────────────────────────────────────────────────"

    python scripts/train_txl_detail_resolver_v1.py \
        "${SHARED_ARGS[@]}" \
        $SUPER_ARG \
        --run_name                "${RUN_NAME}" \
        --memory_backend          "${BACKEND}" \
        --txl_layers              "${TXL_LAYERS}" \
        --txl_heads               "${TXL_HEADS}" \
        --mem_dim                 "${MEM_DIM}" \
        --resolver_dim            "${RESOLVER_DIM}" \
        --txl_ff_dim              "${TXL_FF_DIM}" \
        --num_memory_slots        "${NUM_SLOTS}" \
        --attention_heads         "${ATTN_HEADS}" \
        --hidden_dim              "${HIDDEN_DIM}" \
        "--selector_margin_thresholds=-999,0.0,0.5,1.0,1.5,2.0"

    local EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "  [FAIL] ${RUN_NAME} exited with code $EXIT"
        echo "${RUN_NAME}" >> "${FAILED_LOG}"
    else
        echo "  [DONE] ${RUN_NAME}"
    fi
    return 0
}

# ── Group A — TXL depth/width ──────────────────────────────────────────────────
echo "========================================================"
echo " GROUP A — TXL depth/width runs"
echo "========================================================"

# A0: baseline TXL  (L=2, D=256, ff=1024, slots=16, heads=4, hidden=512)
run_one "sweep_A0_txl_L2_D256"  txl  2  4  256  256  1024  16  4  512

# A1: deeper TXL    (L=4, D=256, ff=1024, slots=16, heads=4, hidden=512)
run_one "sweep_A1_txl_L4_D256"  txl  4  4  256  256  1024  16  4  512

# A2: deeper TXL    (L=6, D=256, ff=1024, slots=16, heads=4, hidden=512)
run_one "sweep_A2_txl_L6_D256"  txl  6  4  256  256  1024  16  4  512

# A3: wider TXL     (L=4, D=384, ff=1536, slots=16, heads=6, hidden=768)
run_one "sweep_A3_txl_L4_D384"  txl  4  6  384  384  1536  16  6  768

# A4: wider+deeper  (L=6, D=384, ff=1536, slots=16, heads=6, hidden=768)
run_one "sweep_A4_txl_L6_D384"  txl  6  6  384  384  1536  16  6  768

# ── Group B — parameter-matched token-embedding controls ──────────────────────
echo ""
echo "========================================================"
echo " GROUP B — token-embedding controls (parameter-matched)"
echo "========================================================"

# B0: token, hidden=1024       (param-matched to A1; txl args valid but unused)
run_one "sweep_B0_token_h1024"       token  0  4  256  256  1024  0  4  1024

# B1: token, hidden=1408       (param-matched to A3)
run_one "sweep_B1_token_h1408"       token  0  4  256  256  1024  0  4  1408

# B2: token, r=384, hidden=1536  (param-matched to A4)
run_one "sweep_B2_token_r384_h1536"  token  0  6  384  384  1536  0  6  1536

# ── Analysis ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Running sweep analysis..."
echo " $(date)"
echo "========================================================"
echo ""

python scripts/analyze_txl_depth_capacity_sweep_v1.py \
    --sweep_root "${OUT_ROOT}" \
    --stage2c_ctg             0.0084 \
    --stage2c_caw             0.0089 \
    --stage2c_fv_gain         -0.0017 \
    --stage2c_sip             0.2457 \
    --stage2c_applied_precision 0.1047

ANALYZE_EXIT=$?
if [ $ANALYZE_EXIT -ne 0 ]; then
    echo "WARNING: analysis script exited with code $ANALYZE_EXIT"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " TXL Depth/Capacity Sweep V1 complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs:"
echo "  ${OUT_ROOT}/sweep_summary.csv"
echo "  ${OUT_ROOT}/sweep_report.md"
echo ""

if [ -f "${FAILED_LOG}" ]; then
    echo "WARNING: Some runs failed:"
    cat "${FAILED_LOG}"
    echo ""
fi

echo "Individual run dirs:"
for RUN_NAME in \
    sweep_A0_txl_L2_D256 \
    sweep_A1_txl_L4_D256 \
    sweep_A2_txl_L6_D256 \
    sweep_A3_txl_L4_D384 \
    sweep_A4_txl_L6_D384 \
    sweep_B0_token_h1024  \
    sweep_B1_token_h1408  \
    sweep_B2_token_r384_h1536; do
    RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
    if [ -f "${RUN_DIR}/final_metrics.json" ]; then
        echo "  [OK] ${RUN_DIR}"
    else
        echo "  [MISSING] ${RUN_DIR}/final_metrics.json"
    fi
done
echo ""
