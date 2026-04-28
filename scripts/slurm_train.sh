#!/bin/bash
#SBATCH --job-name=agnews_bpe
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/miniconda3/bin/activate
conda activate learned_regions

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/ondemand/upload_me/RegionTokenizer

python run.py \
    --config configs/agnews_bpe.yaml \
    --outdir outputs/runs/agnews_bpe \
    --seed 42


