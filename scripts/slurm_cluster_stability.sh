#!/bin/bash
#SBATCH --job-name=cluster_stab
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-cluster-stab-%j.out
#SBATCH --error=slurm-cluster-stab-%j.err

source ~/miniconda3/bin/activate
conda activate learned_regions

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD

python interference_experiment/cluster_stability.py \
    --cache_dir      interference_experiment/results/probe_cache \
    --output_dir     interference_experiment/results/cluster_stability \
    --min_occurrences 20 \
    --n_pairs        20000 \
    --n_flip_pairs   20 \
    --seed           42
