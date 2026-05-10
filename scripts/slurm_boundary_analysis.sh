#!/bin/bash
#SBATCH --job-name=boundary_analysis
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Boundary analysis: pure inference diagnostic for trained repr_region checkpoints.
# No training. Runs ~200 val batches per checkpoint.
#
# Analyses each checkpoint listed in CKPTS below. For each it writes:
#   runs/boundary_analysis_<name>/boundary_metrics.csv
#   runs/boundary_analysis_<name>/token_context_variance.csv
#   runs/boundary_analysis_<name>/token_boundary_stats.csv
#   runs/boundary_analysis_<name>/boundary_analysis_summary.md
#   runs/boundary_analysis_<name>/plots/*.png  (7 plots)
#
# To analyse a specific checkpoint only:
#   FORCE_RUN=1 sbatch --export=ALL,TARGET=repr_region_reference slurm_boundary_analysis.sh

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

# ── Safety checks ──────────────────────────────────────────────────────────────
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

# ── Checkpoints to analyse ─────────────────────────────────────────────────────
# Format: "run_dir_name  label"
# checkpoint_latest.pt is used from each run_dir.
# If TARGET is set (via --export), only that run_dir is analysed.
declare -a CKPTS=(
    "repr_region_reference                 repr_region reference (alpha=0.2)"
    "repr_region_capacity_alpha0p5         capacity only (alpha=0.5)"
    "repr_region_diversity0p01             diversity only (lambda=0.01)"
    "repr_region_capacity_diversity        capacity + diversity"
)

analyse_if_needed() {
    local run_dir="$1"
    local label="$2"
    local ckpt="runs/${run_dir}/checkpoint_latest.pt"
    local out_dir="runs/boundary_analysis_${run_dir}"
    local summary="${out_dir}/boundary_analysis_summary.md"

    echo "=============================="
    echo "Analysing: ${label}"
    echo "  checkpoint: ${ckpt}"
    echo "  output_dir: ${out_dir}"
    echo "=============================="

    if [[ ! -f "$ckpt" ]]; then
        echo "  SKIP — checkpoint not found: $ckpt"
        echo ""
        return 0
    fi

    if [[ -f "$summary" && "${FORCE_RUN:-0}" != "1" ]]; then
        echo "  Already done — skipping (FORCE_RUN=1 to re-run)"
        echo ""
        return 0
    fi

    mkdir -p "$out_dir"
    python train_region_lm.py \
        --mode                  boundary_analysis \
        --repr_ckpt             "$ckpt" \
        --region_map_path       "$REGION_MAP" \
        --output_dir            "$out_dir" \
        --dataset               wikitext-103-raw-v1 \
        --batch_size            32 \
        --max_analysis_batches  0 \
        --seed                  42 \
        --device                cuda

    echo "  Done at $(date)"
    echo ""
}

# ── Run analyses ───────────────────────────────────────────────────────────────
for entry in "${CKPTS[@]}"; do
    run_dir="${entry%%  *}"
    run_dir="${run_dir%"${run_dir##*[! ]}"}"   # trim trailing spaces
    label="${entry#*  }"
    label="${label#"${label%%[! ]*}"}"          # trim leading spaces

    # If TARGET is set, skip everything else
    if [[ -n "${TARGET:-}" && "$run_dir" != "$TARGET" ]]; then
        continue
    fi

    analyse_if_needed "$run_dir" "$label"
done

# ── Cross-run summary ──────────────────────────────────────────────────────────
echo "=============================="
echo "Cross-run verdict summary"
echo "=============================="

python3 - <<'PYEOF'
import os, json

CKPTS = [
    ("repr_region_reference",          "reference"),
    ("repr_region_capacity_alpha0p5",  "cap only"),
    ("repr_region_diversity0p01",      "div only"),
    ("repr_region_capacity_diversity", "cap+div"),
]

print(f"  {'run':<38}  Q1     Q2     Q3     Q4     Q5     Q6     Q7")
print(f"  {'-'*38}  {'----   '*7}")

for run_dir, label in CKPTS:
    md_path = f"runs/boundary_analysis_{run_dir}/boundary_analysis_summary.md"
    if not os.path.isfile(md_path):
        print(f"  [{label:<12}] {run_dir:<38}  (no results yet)")
        continue

    verdicts = {}
    with open(md_path) as f:
        for line in f:
            for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"):
                if f"| {q} |" in line:
                    v = "PASS" if "PASS" in line else "INCONCL"
                    verdicts[q] = v

    row = "  ".join(verdicts.get(f"Q{i}", "?      ")[:6] for i in range(1, 8))
    print(f"  [{label:<12}] {run_dir:<38}  {row}")

PYEOF

echo ""
echo "Done at $(date)"
