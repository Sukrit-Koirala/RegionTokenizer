#!/bin/bash
#SBATCH --job-name=region_lm_baseline
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
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

OUT_DIR=runs/baseline

mkdir -p logs "$OUT_DIR" models/hf_cache

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

python train_region_lm.py \
    --mode                baseline \
    --vocab_subset_size   50257 \
    --dataset             wikitext-103-raw-v1 \
    --seq_len             256 \
    --batch_size          32 \
    --n_layer             6 \
    --d_model             384 \
    --n_head              6 \
    --d_ff                1536 \
    --steps               20000 \
    --lr                  3e-4 \
    --warmup_steps        1000 \
    --eval_interval       500 \
    --save_interval       2000 \
    --log_interval        50 \
    --seed                42 \
    --output_dir          "$OUT_DIR" \
    --device              cuda

echo ""
echo "=============================="
echo "Saved artefacts:"
echo "=============================="
ls -lh "$OUT_DIR/"
echo "Done at $(date)"
