#!/bin/bash
#SBATCH --job-name=hierarchy_mvp
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-hierarchy-mvp-%j.out
#SBATCH --error=slurm-hierarchy-mvp-%j.err

source ~/miniconda3/bin/activate
conda activate learned_regions

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD

# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy MVP Test — GPT-2 XL
#
# Purpose:
#   Three-part experiment testing whether GPT-2 XL forms hierarchical nested
#   token partitions across layers:
#
#   Part A — Recursive Predictability
#     Linear probes at layers [0,4,8,12,16,20,24,47] for 4 granularities:
#     coarse_cluster (9), subcluster (11), token (~5000).
#     Tests whether coarser labels become linearly decodable earlier.
#
#   Part B — Information Gain Curves
#     bits_recovered = log2(n_classes) - CE_bits per layer per granularity.
#     Plots convergence rate of each level.
#
#   Part C — Logit Locality
#     Applies LM head to cached final-layer hidden states.
#     Measures whether top-K predicted tokens concentrate in the same cluster
#     as the top-1 prediction (K in [10,50,100,500]).
#
# Prerequisites:
#   1. interference_experiment/results/probe_cache/ must exist
#      (run slurm_layerwise_probe.sh first)
#   2. interference_experiment/results/cluster7_subcluster_mapping.json
#      (run slurm_subcluster_probe.sh or recursive_cluster7.py first)
#
# Memory breakdown:
#   GPU: probe weights ~10 MB + LM head for Part C ~300 MB   → < 1 GB GPU
#   RAM: hidden states loaded one layer at a time (101k×1600×2 ≈ 320 MB/layer)
#        + LM head (50257×1600×4 ≈ 320 MB)
#        → 48 GB gives wide margin
#
# Outputs (results/hierarchy_mvp/):
#   part_a_metrics.csv            — per-layer per-task: top1, top5, ce_bits
#   part_b_info_gain.csv          — per-layer per-task: H, info_recovered, info_frac
#   part_c_logit_locality.csv     — per-K: same_cluster_frac, unique_clusters
#   recursive_predictability.png  — KEY: top-1/top-5 curves by granularity
#   info_gain_bits.png            — bits recovered vs layer, all levels
#   logit_locality.png            — cluster concentration in top-K
#   candidate_diversity.png       — unique clusters/subclusters in top-K
#   summary.txt                   — auto-generated interpretation
#
# Estimated wall time:
#   Part A: 8 layers × 4 tasks × ~30s = ~17 min
#   Part B: negligible (reuses Part A CE values)
#   Part C: ~5 min (5k samples × batch LM head application)
#   Total:  ~25 min  (3 h limit gives wide margin)
# ─────────────────────────────────────────────────────────────────────────────

GRAPHS=interference_experiment/graphs
RESULTS=interference_experiment/results

python interference_experiment/hierarchy_mvp_test.py \
    --cache_dir        ${RESULTS}/probe_cache \
    --cluster_map      ${RESULTS}/cluster_tokens_coactivation_final.json \
    --subcluster_map   ${RESULTS}/cluster7_subcluster_mapping.json \
    --labels           ${GRAPHS}/layer_final_coactivation_labels_louvain.npy \
    --freq_ids         ${GRAPHS}/layer_final_frequent_token_ids.npy \
    --output_dir       ${RESULTS}/hierarchy_mvp \
    --model_name       gpt2-xl \
    --token_vocab_k    5000 \
    --logit_samples    5000 \
    --epochs           200 \
    --seed             42
