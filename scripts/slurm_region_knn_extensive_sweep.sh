#!/bin/bash
#SBATCH --job-name=knn_extensive
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
# Extensive offline Region-kNN sweep.
#
# Stages:
#   A — memory scale at default k/temp (reference, proxy0.05, proxy0.10)
#   B — k/temp grid at best memory size (proxy0.05, proxy0.10, retrieval_proj)
#   C — raw_h vs retrieval_proj control at best config
#   D — candidate policy sweep at best config
#
# Outputs: runs/region_knn_extensive_sweep/
#   aggregate_results.csv, aggregate_by_split.csv, aggregate_candidate_policies.csv,
#   aggregate_type_analysis.csv, final_report.md, plots/01-10
#
# Set FORCE_RUN=1 to rerun completed experiments.
# Set SKIP_AGGREGATE=1 to skip final aggregation.

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

# ── Paths ────────────────────────────────────────────────────────────────────

REGION_MAP=runs/region_maps_128/token_to_region.json
CKPT_REF=runs/repr_region_reference_30k/checkpoint_latest.pt
CKPT_P05=runs/repr_region_retrieval_proxy_lam0p05/checkpoint_latest.pt
CKPT_P10=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
SWEEP_DIR=runs/region_knn_extensive_sweep

mkdir -p "$SWEEP_DIR"

# ── Preflight ─────────────────────────────────────────────────────────────────

for f in "$REGION_MAP" "$CKPT_REF" "$CKPT_P05" "$CKPT_P10"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done
python -c "from scripts.offline_region_knn import _adaptive_coverage_generic; print('imports OK')" \
    || { echo "ERROR: offline_region_knn.py import failed" >&2; exit 1; }

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Date: $(date)"
echo "=============================="

# ── Manifest ─────────────────────────────────────────────────────────────────
# Written before running so aggregate can read it even if some runs fail.

MANIFEST="$SWEEP_DIR/manifest.txt"
: > "$MANIFEST"

# ── run_knn helper ────────────────────────────────────────────────────────────
# run_knn CKPT KEY_SOURCE MEM_K KNN_K KNN_TEMP OUT_DIR
run_knn() {
    local ckpt="$1" key_src="$2" mem="$3" k="$4" temp="$5" out="$6"

    # Register in manifest
    echo "$out" >> "$MANIFEST"

    # Skip if already done (unless forced)
    if [[ "${FORCE_RUN:-0}" != "1" ]] \
       && [[ -f "$out/summary.md" ]] \
       && [[ -f "$out/metrics_all.csv" ]]; then
        echo "[SKIP] $out  (summary.md exists)"
        return 0
    fi

    mkdir -p "$out"
    echo "------------------------------------------------------"
    echo "[RUN ] $out"
    echo "       ckpt=$(basename $ckpt)  src=$key_src  mem=$mem  k=$k  temp=$temp"
    echo "       Start: $(date)"
    echo "------------------------------------------------------"

    python scripts/offline_region_knn.py \
        --model_type          small \
        --small_ckpt          "$ckpt" \
        --dataset             wikitext-103-raw-v1 \
        --region_map_path     "$REGION_MAP" \
        --knn_key_source      "$key_src" \
        --max_memory_positions "$mem" \
        --max_eval_positions  247000 \
        --knn_k               "$k" \
        --knn_temp            "$temp" \
        --probe_temp          1.0 \
        --normalize_keys      true \
        --output_dir          "$out" \
        --batch_size          4 \
        --seq_len             512 \
        --device              cuda \
        --seed                42 \
        && echo "[OK ] $out  $(date)" \
        || echo "[FAIL] $out  exit=$?" | tee -a "$SWEEP_DIR/failed_runs.txt"
}

# ── Stage A — memory scale ───────────────────────────────────────────────────
echo ""
echo "=============================="
echo "STAGE A: Memory scale  (k=64  temp=0.20)"
echo "=============================="

for mem in 50000 100000 500000 1000000 2000000; do
    mtag=$(python3 -c "m=$mem; print(f'{m//1000}k' if m<1000000 else f'{m//1000000}M')")

    run_knn "$CKPT_REF" raw_h        "$mem" 64 0.20 \
        "$SWEEP_DIR/reference_raw_h_mem${mtag}_k64_t0p20"

    run_knn "$CKPT_P05" retrieval_proj "$mem" 64 0.20 \
        "$SWEEP_DIR/proxy005_retrproj_mem${mtag}_k64_t0p20"

    run_knn "$CKPT_P10" retrieval_proj "$mem" 64 0.20 \
        "$SWEEP_DIR/proxy010_retrproj_mem${mtag}_k64_t0p20"
done

echo ""
echo "Stage A done: $(date)"

# ── Stage B — k/temp grid ────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "STAGE B: k/temp grid  (mem=500k)"
echo "=============================="

MEM_B=500000
MTAG_B="500k"

for kv in 16 32 64 128; do
    for tv in 0p05 0p10 0p20 0p50 1p00; do
        # convert tag back to float
        tf=$(python3 -c "print('$tv'.replace('p','.'))")

        run_knn "$CKPT_P05" retrieval_proj "$MEM_B" "$kv" "$tf" \
            "$SWEEP_DIR/proxy005_retrproj_mem${MTAG_B}_k${kv}_t${tv}"

        run_knn "$CKPT_P10" retrieval_proj "$MEM_B" "$kv" "$tf" \
            "$SWEEP_DIR/proxy010_retrproj_mem${MTAG_B}_k${kv}_t${tv}"
    done
done

echo ""
echo "Stage B done: $(date)"

# ── Stage C — raw_h vs retrieval_proj control ────────────────────────────────
echo ""
echo "=============================="
echo "STAGE C: raw_h vs retrieval_proj control  (mem=500k  k=64  temp=0.20)"
echo "=============================="

# reference raw_h already in Stage A
run_knn "$CKPT_P05" raw_h          500000 64 0.20 \
    "$SWEEP_DIR/proxy005_raw_h_mem500k_k64_t0p20"
# proxy005 retrieval_proj already in Stage A
run_knn "$CKPT_P10" raw_h          500000 64 0.20 \
    "$SWEEP_DIR/proxy010_raw_h_mem500k_k64_t0p20"
# proxy010 retrieval_proj already in Stage A

echo ""
echo "Stage C done: $(date)"

# ── Stage D — best config (use proxy010 retrieval_proj at 500k k=64 t=0.20) ──
echo ""
echo "=============================="
echo "STAGE D: Candidate policy sweep at best config"
echo "       (proxy010 retrieval_proj mem=500k k=64 temp=0.20)"
echo "=============================="

# This run shares the output dir from Stage A — already completed.
# The extensive candidate policies (v1-v5, all unions, all router/mem/mix k)
# are already emitted by offline_region_knn.py in candidate_coverage.csv.
# Stage D is just a reminder; re-run with FORCE_RUN=1 to regenerate.

BEST_DIR="$SWEEP_DIR/proxy010_retrproj_mem500k_k64_t0p20"
if [[ -f "$BEST_DIR/candidate_coverage.csv" ]]; then
    echo "[Stage D] candidate_coverage.csv already exists at $BEST_DIR"
    echo "          $(wc -l < "$BEST_DIR/candidate_coverage.csv") rows"
else
    echo "[Stage D] best config run not yet complete — will aggregate what exists"
fi

echo ""
echo "Stage D done: $(date)"

# ── Aggregate ─────────────────────────────────────────────────────────────────
if [[ "${SKIP_AGGREGATE:-0}" == "1" ]]; then
    echo "[SKIP] aggregation disabled (SKIP_AGGREGATE=1)"
else
    echo ""
    echo "=============================="
    echo "AGGREGATING RESULTS"
    echo "=============================="
    python scripts/aggregate_region_knn_sweep.py \
        --sweep_dir  "$SWEEP_DIR" \
        --output_dir "$SWEEP_DIR" \
        && echo "[OK] aggregation complete" \
        || echo "[WARN] aggregation returned non-zero — partial results may exist"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "SWEEP COMPLETE: $(date)"
echo "=============================="
echo ""
echo "--- Failed runs ---"
if [[ -f "$SWEEP_DIR/failed_runs.txt" ]]; then
    cat "$SWEEP_DIR/failed_runs.txt"
else
    echo "(none)"
fi
echo ""
echo "--- Final report ---"
if [[ -f "$SWEEP_DIR/final_report.md" ]]; then
    head -80 "$SWEEP_DIR/final_report.md"
fi
echo ""
echo "Job $SLURM_JOB_ID complete."
