#!/bin/bash
#SBATCH --job-name=interf_regions
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-interf-regions-%j.out
#SBATCH --error=slurm-interf-regions-%j.err

source ~/miniconda3/bin/activate
conda activate learned_regions

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD

PIPE=interference_experiment/interference_region_pipeline

python ${PIPE}/run_all.py \
    --output_dir       ${PIPE}/results \
    --graphs_dir       ${PIPE}/graphs \
    --model_name       gpt2-xl \
    --dataset_name     wikitext \
    --dataset_config   wikitext-2-raw-v1 \
    --max_tokens       500000 \
    --vocab_subset_size 5000 \
    --top_k            50 \
    --seq_len          512 \
    --n_probe          50000 \
    --probe_layers     0 4 8 12 16 20 24 47 \
    --alpha            1.0 \
    --gamma            0.0 \
    --resolutions      0.5 1.0 1.5 2.0 \
    --recursive_depth  2 \
    --recursive_min_size 200 \
    --min_modularity_gain 0.05 \
    --probe_epochs     200 \
    --logit_ks         10 50 100 500 \
    --r_values         1 2 4 8 \
    --n_routed_eval    5000 \
    --seed             42 \
    --eval_leaf_regions \
    --leaf_r_values    1 2 4 8 16 32
