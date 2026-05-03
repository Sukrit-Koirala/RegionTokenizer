#!/bin/bash
#SBATCH --job-name=soft_routing_eval
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
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

OUT_DIR=interference_experiment/full_vocab_region_eval

mkdir -p logs "${OUT_DIR}/soft_routing_eval" models/hf_cache

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

python interference_experiment/full_vocab_region_eval.py \
    --stage              soft_routing_eval \
    --model_name         gpt2-xl \
    --dataset_name       wikitext \
    --dataset_config     wikitext-2-raw-v1 \
    --vocab_subset_size  20000 \
    --layers             0,4,8,12,16,20,24,47 \
    --soft_routing_layer 47 \
    --probe_epochs       100 \
    --probe_lr           1e-3 \
    --probe_batch_size   2048 \
    --output_dir         "${OUT_DIR}" \
    --seed               42

echo ""
echo "=============================="
echo "Saved artefacts:"
echo "=============================="
ls -lh "${OUT_DIR}/soft_routing_eval/"
echo "Done at $(date)"
