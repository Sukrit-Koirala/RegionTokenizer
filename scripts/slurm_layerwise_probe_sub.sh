#!/bin/bash
#SBATCH --job-name=subcluster_probe
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-subcluster-probe-%j.out
#SBATCH --error=slurm-subcluster-probe-%j.err

source ~/miniconda3/bin/activate
conda activate learned_regions

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD

# ─────────────────────────────────────────────────────────────────────────────
# Layerwise subcluster probe — cluster 7
#
# Purpose:
#   Determines at which layer subcluster identity (11 subclusters within
#   coarse cluster 7) becomes linearly decodable vs. exact token identity
#   (~1896 classes). The gap between A and B is the key finding.
#
# Prerequisites:
#   1. layerwise_region_probe.py must have run and populated --cache_dir.
#   2. recursive_cluster7.py must have run and produced
#      cluster7_subcluster_mapping.json in --output_dir.
#
# Memory breakdown (approx):
#   GPU : probe linear layers are tiny; GPU used only for Adam training
#         → 4 GB comfortable
#   RAM : reads cached hidden states from disk (no GPT-2 XL in memory)
#         cluster-7 slice of 100k × 1600 × fp16 ≈ ~300 MB
#         + sklearn/numpy working memory ≈ 4 GB
#         → 32 GB requested with headroom
#
# Outputs (results/cluster7_subcluster_probe/):
#   metrics.json
#   layer_vs_subcluster.png
#   layer_vs_token.png
#   comparison.png          — KEY: normalized gain, subcluster vs token
# ─────────────────────────────────────────────────────────────────────────────

python interference_experiment/layerwise_subcluster_probe.py \
    --cache_dir      interference_experiment/results/probe_cache \
    --subcluster_map interference_experiment/results/cluster7_subcluster_mapping.json \
    --output_dir     interference_experiment/results/cluster7_subcluster_probe \
    --seed           42 \
    --train_frac     0.7 \
    --epochs         200 \
    --lr             1e-2 \
    --batch_size     2048