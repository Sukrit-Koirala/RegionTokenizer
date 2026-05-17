#!/bin/bash
#SBATCH --job-name=clean_refiner_d3
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Clean path-conditioned refiner experiment — D3 architecture.
#
# Phases:
#   1. Build val dataset   (build_clean_path_refiner_dataset.py --split val)
#   2. Build train dataset (build_clean_path_refiner_dataset.py --split train)
#   3. Masked-softmax baseline eval (eval_clean_masked_softmax.py)
#   4. Train Variant C  (RicherMLP)
#   5. Train D3-small   (RegionTransformerRefiner, d_region=128, L=1, H=4)
#   6. Train D3-base    (RegionTransformerRefiner, d_region=256, L=2, H=4)  [if time]
#
# Set RUN_PHASEn=0 to skip a phase (e.g. if dataset already built).

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
mkdir -p logs

# ── Configurable paths ────────────────────────────────────────────────────────

SMALL_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
KNN_RUN_DIR=runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20
REGION_MAP=runs/region_maps_128/token_to_region.json
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
PARETO_CSV=runs/hier_policy_tuning/tuned_pareto_frontier.csv
OUTPUT_ROOT=runs/path_refiner_clean

POLICY=hgrid_K24_srccombined_nts1_bkr12_fb24_cm0.15_en1.25_corec60m30
TRAIN_DIR=$OUTPUT_ROOT/data/train_hgrid_K24
VAL_DIR=$OUTPUT_ROOT/data/val_hgrid_K24
EVAL_DIR=$OUTPUT_ROOT/eval
N_SUPER=24

# ── Phase toggles ─────────────────────────────────────────────────────────────
RUN_PHASE1=1   # build val dataset
RUN_PHASE2=1   # build train dataset
RUN_PHASE3=1   # masked-softmax baseline eval
RUN_PHASE4=1   # train Variant C
RUN_PHASE5=1   # train D3-small
RUN_PHASE6=1   # train D3-base

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Clean Path Refiner D3  $(date) ==="
echo "    SMALL_CKPT  : $SMALL_CKPT"
echo "    KNN_RUN_DIR : $KNN_RUN_DIR"
echo "    POLICY      : $POLICY"
echo "    OUTPUT_ROOT : $OUTPUT_ROOT"

for f in "$SMALL_CKPT" "$REGION_MAP" "$SUPER_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

if [[ ! -f "$KNN_RUN_DIR/per_position.npz" ]] && \
   [[ ! -f "$KNN_RUN_DIR/per_position_topk.npz" ]]; then
    echo "ERROR: per_position.npz not found in $KNN_RUN_DIR" >&2
    echo "       Run offline_region_knn.py first." >&2
    exit 1
fi

mkdir -p "$TRAIN_DIR" "$VAL_DIR" "$EVAL_DIR" \
         "$OUTPUT_ROOT/variant_C" \
         "$OUTPUT_ROOT/d3_small"  \
         "$OUTPUT_ROOT/d3_base"

# ── Phase 1: Build val dataset ────────────────────────────────────────────────

if [[ "$RUN_PHASE1" == "1" ]]; then
    echo ""
    echo "=== Phase 1: build val dataset  $(date) ==="

    N_SHARDS_VAL=$(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
    if [[ "$N_SHARDS_VAL" -gt 0 ]]; then
        echo "    val dataset already exists ($N_SHARDS_VAL shards) — skipping build"
    else
        python scripts/build_clean_path_refiner_dataset.py \
            --split         val                     \
            --small_ckpt    "$SMALL_CKPT"           \
            --knn_run_dir   "$KNN_RUN_DIR"          \
            --region_map    "$REGION_MAP"           \
            --super_map     "$SUPER_MAP"            \
            --output_dir    "$VAL_DIR"              \
            --policy        "$POLICY"               \
            --max_positions 300000                  \
            --shard_size    10000                   \
            --candidate_cap 8192                    \
            --device        cuda
        echo "    val shards: $(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" | wc -l)"
    fi
else
    echo "=== Phase 1: skipped  ($(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l) val shards found) ==="
fi

# ── Phase 2: Build train dataset ─────────────────────────────────────────────

if [[ "$RUN_PHASE2" == "1" ]]; then
    echo ""
    echo "=== Phase 2: build train dataset  $(date) ==="

    N_SHARDS_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
    if [[ "$N_SHARDS_TRAIN" -gt 0 ]]; then
        echo "    train dataset already exists ($N_SHARDS_TRAIN shards) — skipping build"
    else
        python scripts/build_clean_path_refiner_dataset.py \
            --split         train                   \
            --small_ckpt    "$SMALL_CKPT"           \
            --knn_run_dir   "$KNN_RUN_DIR"          \
            --region_map    "$REGION_MAP"           \
            --super_map     "$SUPER_MAP"            \
            --output_dir    "$TRAIN_DIR"            \
            --policy        "$POLICY"               \
            --max_positions 1000000                 \
            --shard_size    10000                   \
            --candidate_cap 8192                    \
            --device        cuda
        echo "    train shards: $(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" | wc -l)"
    fi
else
    echo "=== Phase 2: skipped  ($(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l) train shards found) ==="
fi

# Validate datasets exist before training
for d in "$TRAIN_DIR" "$VAL_DIR"; do
    n=$(find "$d" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
    if [[ "$n" -eq 0 ]]; then
        echo "ERROR: no shard_*.pt files in $d" >&2
        echo "       Set RUN_PHASE1=1 / RUN_PHASE2=1 to build them." >&2
        exit 1
    fi
done

# ── Phase 3: Masked-softmax baseline eval ────────────────────────────────────

if [[ "$RUN_PHASE3" == "1" ]]; then
    echo ""
    echo "=== Phase 3: masked-softmax eval  $(date) ==="

    PARETO_ARG=""
    if [[ -f "$PARETO_CSV" ]]; then
        PARETO_ARG="--pareto_csv $PARETO_CSV"
    fi

    python scripts/eval_clean_masked_softmax.py \
        --small_ckpt    "$SMALL_CKPT"   \
        --knn_run_dir   "$KNN_RUN_DIR"  \
        --region_map    "$REGION_MAP"   \
        --super_map     "$SUPER_MAP"    \
        --output_dir    "$EVAL_DIR"     \
        $PARETO_ARG                     \
        --max_positions 300000          \
        --device        cuda

    echo "    eval outputs: $EVAL_DIR/masked_softmax_eval.csv"
else
    echo "=== Phase 3: skipped ==="
fi

# ── Phase 4: Train Variant C (RicherMLP) ─────────────────────────────────────

if [[ "$RUN_PHASE4" == "1" ]]; then
    echo ""
    echo "=== Phase 4: Variant C (RicherMLP)  $(date) ==="

    python scripts/train_clean_path_refiner.py \
        --train_dir       "$TRAIN_DIR"              \
        --val_dir         "$VAL_DIR"                \
        --small_ckpt      "$SMALL_CKPT"             \
        --super_map       "$SUPER_MAP"              \
        --variant         C                         \
        --d_region        32                        \
        --d_hidden        256                       \
        --output_dir      "$OUTPUT_ROOT/variant_C"  \
        --steps           20000                     \
        --eval_every      1000                      \
        --batch_size      64                        \
        --lr              3e-4                      \
        --eval_before_train                         \
        --device          cuda

    echo "    Variant C done: $OUTPUT_ROOT/variant_C/best_refiner.pt"
else
    echo "=== Phase 4: skipped ==="
fi

# ── Phase 5: Train D3-small ───────────────────────────────────────────────────

if [[ "$RUN_PHASE5" == "1" ]]; then
    echo ""
    echo "=== Phase 5: D3-small  $(date) ==="

    python scripts/train_clean_path_refiner.py \
        --train_dir       "$TRAIN_DIR"              \
        --val_dir         "$VAL_DIR"                \
        --small_ckpt      "$SMALL_CKPT"             \
        --super_map       "$SUPER_MAP"              \
        --variant         D3                        \
        --d3_size         small                     \
        --output_dir      "$OUTPUT_ROOT/d3_small"   \
        --steps           20000                     \
        --eval_every      1000                      \
        --batch_size      32                        \
        --lr              3e-4                      \
        --eval_before_train                         \
        --device          cuda

    echo "    D3-small done: $OUTPUT_ROOT/d3_small/best_refiner.pt"
else
    echo "=== Phase 5: skipped ==="
fi

# ── Phase 6: Train D3-base ────────────────────────────────────────────────────

if [[ "$RUN_PHASE6" == "1" ]]; then
    echo ""
    echo "=== Phase 6: D3-base  $(date) ==="

    python scripts/train_clean_path_refiner.py \
        --train_dir       "$TRAIN_DIR"              \
        --val_dir         "$VAL_DIR"                \
        --small_ckpt      "$SMALL_CKPT"             \
        --super_map       "$SUPER_MAP"              \
        --variant         D3                        \
        --d3_size         base                      \
        --output_dir      "$OUTPUT_ROOT/d3_base"    \
        --steps           20000                     \
        --eval_every      1000                      \
        --batch_size      16                        \
        --lr              1e-4                      \
        --eval_before_train                         \
        --device          cuda

    echo "    D3-base done: $OUTPUT_ROOT/d3_base/best_refiner.pt"
else
    echo "=== Phase 6: skipped ==="
fi

# ── Final comparison report ───────────────────────────────────────────────────

echo ""
echo "=== Done  $(date) ==="
echo ""
echo "Outputs:"
echo "  Val dataset    : $VAL_DIR/   ($(find "$VAL_DIR"   -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l) shards)"
echo "  Train dataset  : $TRAIN_DIR/ ($(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l) shards)"
echo "  Baseline eval  : $EVAL_DIR/masked_softmax_eval.csv"
echo "  Variant C ckpt : $OUTPUT_ROOT/variant_C/best_refiner.pt"
echo "  D3-small ckpt  : $OUTPUT_ROOT/d3_small/best_refiner.pt"
echo "  D3-base  ckpt  : $OUTPUT_ROOT/d3_base/best_refiner.pt"
echo ""

echo "=== Training summaries ==="
for run_dir in "$OUTPUT_ROOT/variant_C" "$OUTPUT_ROOT/d3_small" "$OUTPUT_ROOT/d3_base"; do
    log="$run_dir/train_log.csv"
    name=$(basename "$run_dir")
    if [[ -f "$log" ]]; then
        echo "--- $name (final eval) ---"
        tail -1 "$log"
    else
        echo "--- $name: no log found ---"
    fi
done

echo ""
echo "=== Baseline summary ==="
if [[ -f "$EVAL_DIR/masked_softmax_eval.csv" ]]; then
    echo "Top-3 policies by strict_nll:"
    # Print header + top 3 data rows (skip header, sort by strict_nll col)
    head -1 "$EVAL_DIR/masked_softmax_eval.csv"
    tail -n +2 "$EVAL_DIR/masked_softmax_eval.csv" | sort -t',' -k6 -n | head -3
fi
