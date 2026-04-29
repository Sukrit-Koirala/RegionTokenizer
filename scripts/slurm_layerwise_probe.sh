#!/bin/bash
#SBATCH --job-name=layer_probe
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-layer-probe-%j.out
#SBATCH --error=slurm-layer-probe-%j.err

source ~/miniconda3/bin/activate
conda activate learned_regions

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD

# ─────────────────────────────────────────────────────────────────────────────
# Layerwise region probe — GPT-2 XL
#
# Purpose:
#   Determines how early in the transformer coarse token regions (coactivation
#   clusters) become linearly decodable from hidden states, and whether they
#   emerge earlier than exact token identity.
#
# Pipeline:
#   1. Load GPT-2 XL (fp16, frozen).
#   2. Run one forward pass over WikiText-2 with output_hidden_states=True,
#      collecting hidden states at layers [0, 4, 8, 12, 16, 20, 24, final].
#   3. For each layer: fit a LogisticRegression probe h_t^(l) -> cluster.
#   4. Also fit a token-identity probe h_t^(l) -> top-1000 token for baseline.
#   5. Compute top-1 / top-4 cluster recall and exact-token top-1 per layer.
#
# Memory breakdown (approx):
#   GPU  : GPT-2 XL fp16 ~3 GB + forward-pass buffers ~1 GB  → ~4 GB total
#   RAM  : 8 layers × 100k × 1600 × float16 ≈ 2.6 GB collected data
#          + sklearn saga working memory ≈ 8 GB
#          + model on CPU during load ≈ 6 GB
#          → 64 GB requested with comfortable headroom
#
# Outputs (results/):
#   metrics.json                        — all per-layer metrics
#   summary.txt                         — answers the 5 research questions
#   layer_vs_top1_cluster.png
#   layer_vs_top4_cluster.png           — KEY: early-routing threshold plot
#   layer_vs_exact_token_acc.png
#   calibration_by_layer.png
#   per_cluster_recall_heatmap.png
#   layer_cluster_vs_token_comparison.png
#
# Estimated wall time:
#   Extraction  ~5 min   (L40S, batch=4, seq=512, 100k positions)
#   Probes      ~40 min  (saga, 8 layers, 70k train × 1600 dims → 9 classes)
#   Total       ~1 h     (4 h limit gives ample safety margin)
# ─────────────────────────────────────────────────────────────────────────────

python interference_experiment/layerwise_region_probe.py \
    --clusters  interference_experiment/results/cluster_tokens_coactivation_final.json \
    --labels    interference_experiment/graphs/layer_final_coactivation_labels_louvain.npy \
    --freq_ids  interference_experiment/graphs/layer_final_frequent_token_ids.npy \
    --model     gpt2-xl \
    --output_dir interference_experiment/results/ \
    --cache_dir  interference_experiment/results/probe_cache \
    --max_samples 100000 \
    --batch_size  4 \
    --seq_len     512 \
    --token_vocab_k 1000 \
    --seed 42
