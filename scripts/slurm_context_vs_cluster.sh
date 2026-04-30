#!/bin/bash
#SBATCH --job-name=ctx_cluster
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-ctx-cluster-%j.out
#SBATCH --error=slurm-ctx-cluster-%j.err

source ~/miniconda3/bin/activate
conda activate learned_regions

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD

# ─────────────────────────────────────────────────────────────────────────────
# Context vs Cluster geometry test — GPT-2 XL
#
# Purpose:
#   Tests whether context moves token representations LOCALLY within clusters
#   (small drift) rather than GLOBALLY across clusters (large drift).
#
# Metrics (all normalized by layer-mean hidden-state norm):
#   contextual_drift  — L2 spread of the same token across different contexts
#   intra_cluster     — mean pairwise L2 between per-token means within a cluster
#   inter_cluster     — mean pairwise L2 between cluster centroids
#   cluster_flip_rate — fraction of occurrences where nearest-centroid
#                       classification disagrees with the token's true cluster
#   subcluster_flip_rate — same, at subcluster granularity (cluster-7 only)
#
# Expected signal:
#   inter_cluster >> intra_cluster >= contextual_drift
#   cluster_flip_rate low, subcluster_flip_rate higher
#
# Prerequisites:
#   interference_experiment/results/probe_cache/ must exist
#   interference_experiment/results/cluster7_subcluster_mapping.json must exist
#
# Memory breakdown:
#   GPU: one layer's hidden states in fp32: 101k × 1600 × 4 ≈ 650 MB
#        + centroid matrices (tiny)  → < 1 GB GPU total
#   RAM: one layer loaded at a time  ≈ 325 MB
#        + group index arrays        ≈ 100 MB
#        → 24 GB with wide margin
#
# Outputs (results/context_vs_cluster/):
#   metrics.json         — per-layer: drift, intra, inter, flip rates, avg_norm
#   distances_plot.png   — drift / intra / inter vs layer
#   flip_rates_plot.png  — cluster / subcluster flip rate vs layer
#
# Estimated wall time:
#   8 layers × ~30s (token-mean computation + nearest-centroid over 101k) ≈ 4 min
#   1 h limit gives wide margin
# ─────────────────────────────────────────────────────────────────────────────

python interference_experiment/context_vs_cluster.py \
    --cache_dir      interference_experiment/results/probe_cache \
    --subcluster_map interference_experiment/results/cluster7_subcluster_mapping.json \
    --output_dir     interference_experiment/results/context_vs_cluster \
    --min_occurrences 20 \
    --seed 42
