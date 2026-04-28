#!/bin/bash
#SBATCH --job-name=interference_graph
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# -----------------------------------------------------------------------
# Interference graph experiment
# NOTE: convert line endings before submitting from a Windows machine:
#   dos2unix scripts/slurm_interference.sh
#   sbatch  scripts/slurm_interference.sh
#
# Estimated wall time:
#   GPT-2 XL, 50k tokens, 5k vocab subset  ~60-90 min
#   + layer comparison (3 layers)           ~2-3 hr total
# -----------------------------------------------------------------------

# conda activate exits non-zero in strict mode, so set -e comes AFTER
source ~/miniconda3/bin/activate
conda activate learned_regions

set -eo pipefail          # -e: exit on error  -o pipefail: catch pipe errors
                          # NO -u: conda/HF use unset vars internally

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME="$SLURM_SUBMIT_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# ---- Working directory (same as existing slurm_train.sh) ---------------
cd ~/ondemand/upload_me/RegionTokenizer

# logs/ must exist before sbatch is called; create it here as a fallback
mkdir -p logs \
         interference_experiment/data \
         interference_experiment/graphs \
         interference_experiment/results \
         models/hf_cache

# ---- GPU sanity check --------------------------------------------------
echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

# ---- Main experiment (final layer) -------------------------------------
cd interference_experiment

python run_all.py \
    --model        gpt2-xl \
    --max_tokens   50000 \
    --batch_size   4 \
    --vocab_subset 5000 \
    --top_k        20 \
    --n_clusters   50

# ---- Optional: layer comparison ----------------------------------------
# Uncomment to add ~1-2 hr.  Prior artefacts are cached so only the
# extra layer passes run.
#
# python run_all.py \
#     --model        gpt2-xl \
#     --max_tokens   50000 \
#     --batch_size   4 \
#     --vocab_subset 5000 \
#     --top_k        20 \
#     --n_clusters   50 \
#     --layer_compare \
#     --skip_extraction \
#     --skip_graphs \
#     --skip_clustering \
#     --skip_viz

# ---- Artefact manifest -------------------------------------------------
echo ""
echo "=============================="
echo "Saved artefacts:"
echo "=============================="
echo "-- graphs/ --"
ls -lh graphs/
echo "-- results/ --"
ls -lh results/
echo "Done at $(date)"
