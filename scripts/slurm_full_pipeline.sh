#!/bin/bash
#SBATCH --job-name=region_full_pipeline
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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

# ── Paths ──────────────────────────────────────────────────────────────────────
BASELINE_CKPT=runs/baseline/checkpoint_latest.pt
REGION_MAP_DIR=runs/region_maps_full
REGION_MAP=${REGION_MAP_DIR}/token_to_region.json
LEAF_MAP=${REGION_MAP_DIR}/region_tree.json

mkdir -p logs models/hf_cache "$REGION_MAP_DIR"

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

# ── 0. Build region maps from trained baseline (full vocab) ───────────────────
echo "=============================="
echo "Run 0/5: build_regions  (20k-vocab)"
echo "=============================="
python train_region_lm.py \
    --mode               build_regions \
    --dataset            wikitext-103-raw-v1 \
    --vocab_subset_size  50257 \
    --batch_size         64 \
    --baseline_ckpt      "$BASELINE_CKPT" \
    --region_output_dir  "$REGION_MAP_DIR" \
    --region_vocab_size  20000 \
    --region_top_k       50 \
    --region_max_tokens  1000000 \
    --n_clusters         256 \
    --leaf_min_size      50 \
    --seed               42 \
    --device             cuda
echo "build_regions done at $(date)"
echo ""

# ── Shared args across all conditioned runs ────────────────────────────────────
COMMON_ARGS=(
    --vocab_subset_size   50257
    --dataset             wikitext-103-raw-v1
    --seq_len             256
    --batch_size          32
    --n_layer             6
    --d_model             384
    --n_head              6
    --d_ff                1536
    --steps               20000
    --lr                  3e-4
    --warmup_steps        1000
    --region_warmup_steps 5000
    --base_alpha          0.1
    --base_beta           0.03
    --lambda_coarse       0.05
    --lambda_leaf         0.01
    --lambda_balance      0.001
    --eval_interval       500
    --save_interval       2000
    --log_interval        50
    --seed                42
    --device              cuda
    --region_map_path     "$REGION_MAP"
    --leaf_map_path       "$LEAF_MAP"
)

# ── 1. Coarse only ─────────────────────────────────────────────────────────────
echo "=============================="
echo "Run 1/5: coarse"
echo "=============================="
mkdir -p runs/coarse_full
python train_region_lm.py \
    --mode coarse --output_dir runs/coarse_full \
    "${COMMON_ARGS[@]}"
echo "coarse done at $(date)"
echo ""

# ── 2. Oracle coarse (upper bound, coarse only) ───────────────────────────────
echo "=============================="
echo "Run 2/5: oracle_coarse"
echo "=============================="
mkdir -p runs/oracle_coarse_full
python train_region_lm.py \
    --mode oracle_coarse --output_dir runs/oracle_coarse_full \
    "${COMMON_ARGS[@]}"
echo "oracle_coarse done at $(date)"
echo ""

# ── 3. Coarse + leaf ───────────────────────────────────────────────────────────
echo "=============================="
echo "Run 3/5: coarse_leaf"
echo "=============================="
mkdir -p runs/coarse_leaf_full
python train_region_lm.py \
    --mode coarse_leaf --output_dir runs/coarse_leaf_full \
    "${COMMON_ARGS[@]}"
echo "coarse_leaf done at $(date)"
echo ""

# ── 4. Random control (null hypothesis) ───────────────────────────────────────
echo "=============================="
echo "Run 4/5: random_control"
echo "=============================="
mkdir -p runs/random_control_full
python train_region_lm.py \
    --mode random_control --output_dir runs/random_control_full \
    "${COMMON_ARGS[@]}"
echo "random_control done at $(date)"
echo ""

# ── 5. Oracle coarse + leaf (upper bound) ─────────────────────────────────────
echo "=============================="
echo "Run 5/5: oracle"
echo "=============================="
mkdir -p runs/oracle_full
python train_region_lm.py \
    --mode oracle --output_dir runs/oracle_full \
    "${COMMON_ARGS[@]}"
echo "oracle done at $(date)"
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "=============================="
echo "Final val_lm_loss per run:"
echo "=============================="
for run in coarse_full oracle_coarse_full coarse_leaf_full random_control_full oracle_full; do
    f="runs/$run/final_summary.json"
    if [ -f "$f" ]; then
        python -c "
import json
d = json.load(open('$f'))
print(f\"  {d['mode']:16s}  val_lm={d['val_lm_loss']:.4f}  ppl={d['val_ppl']:.2f}  params={d['n_params']:,}\")
"
    fi
done
echo "Done at $(date)"
