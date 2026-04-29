#!/bin/bash
#SBATCH --job-name=sub7_probe
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-sub7-probe-%j.out
#SBATCH --error=slurm-sub7-probe-%j.err

source ~/miniconda3/bin/activate
conda activate learned_regions

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD

# ─────────────────────────────────────────────────────────────────────────────
# Cluster-7 subcluster probe — GPT-2 XL
#
# Purpose:
#   For tokens whose gold next-token falls in coarse cluster 7 (~1896 tokens),
#   determines how early the transformer can distinguish:
#     A. Which subcluster  (11 classes from recursive_cluster7.py)
#     B. Which exact token (up to ~1896 classes)
#
# Pipeline:
#   Step 1 — Re-run recursive_cluster7.py (CPU) to emit the machine-readable
#             subcluster mapping JSON (cluster7_subcluster_mapping.json).
#             Needed even if you ran it before — mapping save was added later.
#   Step 2 — Run layerwise_subcluster_probe.py (GPU):
#             Reads the existing probe cache (no model forward passes needed),
#             trains a PyTorch linear probe per layer for both tasks,
#             saves metrics.json + 3 plots.
#
# Memory breakdown:
#   GPU  : cluster-7 hidden states per layer + probe weights  < 2 GB
#   RAM  : 8 layers × ~12k × 1600 × float16 ≈ 300 MB
#          + full cache loaded for filtering ≈ 2.6 GB
#          → 32 GB with wide margin
#
# Outputs (results/cluster7_subcluster_probe/):
#   metrics.json                — per-layer top-1/top-4 for both probes
#   layer_vs_subcluster.png     — subcluster recall by layer
#   layer_vs_token.png          — exact-token recall by layer
#   comparison.png              — side-by-side raw + normalized gain
#
# Estimated wall time:
#   Step 1 (clustering)  ~5 min   (CPU only, 1896-node subgraph)
#   Step 2 (probes)      ~20 min  (8 layers × 2 probes × ~1 min each on L40S)
#   Total                ~30 min  (2 h limit gives wide margin)
# ─────────────────────────────────────────────────────────────────────────────

GRAPHS_DIR=interference_experiment/graphs
RESULTS_DIR=interference_experiment/results

# ── Step 1: regenerate cluster-7 subcluster mapping JSON ──────────────────
echo "=== Step 1: recursive_cluster7.py ==="
python interference_experiment/recursive_cluster7.py \
    --graph        ${GRAPHS_DIR}/layer_final_W_norm.npy \
    --clusters     ${RESULTS_DIR}/cluster_tokens_coactivation_final.json \
    --labels       ${GRAPHS_DIR}/layer_final_coactivation_labels_louvain.npy \
    --freq_ids     ${GRAPHS_DIR}/layer_final_frequent_token_ids.npy \
    --target_cluster 7 \
    --output_dir   ${RESULTS_DIR} \
    --recursive_depth 2 \
    --no_graph_plot \
    --seed 42

echo "=== Step 1 done ==="

# ── Step 2: layerwise subcluster probe ────────────────────────────────────
echo "=== Step 2: layerwise_subcluster_probe.py ==="
python interference_experiment/layerwise_subcluster_probe.py \
    --cache_dir      ${RESULTS_DIR}/probe_cache \
    --subcluster_map ${RESULTS_DIR}/cluster7_subcluster_mapping.json \
    --output_dir     ${RESULTS_DIR}/cluster7_subcluster_probe \
    --seed 42

echo "=== Step 2 done ==="
