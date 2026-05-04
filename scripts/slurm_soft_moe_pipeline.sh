#!/bin/bash
#SBATCH --job-name=soft_moe_pipeline
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
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

BASELINE_CKPT=runs/baseline/checkpoint_latest.pt
REGION_MAP_DIR=runs/region_maps_128
REGION_MAP=${REGION_MAP_DIR}/token_to_region.json

mkdir -p logs models/hf_cache "$REGION_MAP_DIR"

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

# ── 0. Build 128-region map ────────────────────────────────────────────────────
echo "=============================="
echo "Run 0/4: build_regions  (spectral, 128 coarse regions, 20k vocab)"
echo "=============================="
python train_region_lm.py \
    --mode                  build_regions \
    --dataset               wikitext-103-raw-v1 \
    --vocab_subset_size     50257 \
    --batch_size            64 \
    --baseline_ckpt         "$BASELINE_CKPT" \
    --region_output_dir     "$REGION_MAP_DIR" \
    --region_vocab_size     20000 \
    --region_top_k          50 \
    --region_max_tokens     1000000 \
    --region_cluster_method spectral \
    --target_n_regions      128 \
    --target_n_leaves       512 \
    --leaf_min_size         50 \
    --seed                  42 \
    --device                cuda
echo "build_regions done at $(date)"
python -c "import json; s=json.load(open('${REGION_MAP_DIR}/region_stats.json')); print(f\"  actual={s['actual_n_regions']}  median_size={s['median_size']:.1f}  coverage={s['coverage']:.1%}\")"
echo ""

# ── Shared args for all soft-moe training runs ─────────────────────────────────
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
    --lambda_coarse       0.05
    --lambda_balance      0.001
    --eval_interval       500
    --save_interval       2000
    --log_interval        50
    --seed                42
    --device              cuda
    --region_map_path     "$REGION_MAP"
)

# ── 1. Soft MoE — learned router, full soft ────────────────────────────────────
echo "=============================="
echo "Run 1/4: soft_moe  (full soft, gamma=0.5)"
echo "=============================="
mkdir -p runs/soft_moe_128
python train_region_lm.py \
    --mode soft_moe --output_dir runs/soft_moe_128 \
    --router_temp 2.0 --prior_gamma 0.5 \
    "${COMMON_ARGS[@]}"
echo "soft_moe done at $(date)"
echo ""

# ── 2. Soft MoE — top-8 sparse routing ────────────────────────────────────────
echo "=============================="
echo "Run 2/4: soft_moe_top8  (top-8 regions, gamma=0.5)"
echo "=============================="
mkdir -p runs/soft_moe_128_top8
python train_region_lm.py \
    --mode soft_moe --output_dir runs/soft_moe_128_top8 \
    --router_temp 2.0 --prior_gamma 0.5 --soft_topk_regions 8 \
    "${COMMON_ARGS[@]}"
echo "soft_moe_top8 done at $(date)"
echo ""

# ── 3. Oracle soft MoE — upper bound ──────────────────────────────────────────
echo "=============================="
echo "Run 3/4: oracle_soft_moe  (true region as one-hot, gamma=0.5)"
echo "=============================="
mkdir -p runs/oracle_soft_moe_128
python train_region_lm.py \
    --mode oracle_soft_moe --output_dir runs/oracle_soft_moe_128 \
    --prior_gamma 0.5 \
    "${COMMON_ARGS[@]}"
echo "oracle_soft_moe done at $(date)"
echo ""

# ── 4. Random soft MoE — null hypothesis ──────────────────────────────────────
echo "=============================="
echo "Run 4/4: random_soft_moe  (true random partition, gamma=0.5)"
echo "=============================="
mkdir -p runs/random_soft_moe_128
python train_region_lm.py \
    --mode random_soft_moe --output_dir runs/random_soft_moe_128 \
    --router_temp 2.0 --prior_gamma 0.5 \
    "${COMMON_ARGS[@]}"
echo "random_soft_moe done at $(date)"
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "=============================="
echo "Final val_lm_loss per run:"
echo "=============================="
for run in soft_moe_128 soft_moe_128_top8 oracle_soft_moe_128 random_soft_moe_128; do
    f="runs/$run/final_summary.json"
    if [ -f "$f" ]; then
        python -c "
import json
d = json.load(open('$f'))
print(f\"  {d['mode']:18s}  val_lm={d['val_lm_loss']:.4f}  ppl={d['val_ppl']:.2f}  cov={d.get('val_coarse_coverage', 0):.1%}\")
"
    fi
done
echo ""
echo "Baseline reference:"
f="runs/baseline/final_summary.json"
[ -f "$f" ] && python -c "
import json; d=json.load(open('$f'))
print(f\"  {'baseline':18s}  val_lm={d['val_lm_loss']:.4f}  ppl={d['val_ppl']:.2f}\")
"
echo "Done at $(date)"
