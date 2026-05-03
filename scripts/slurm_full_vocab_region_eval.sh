#!/bin/bash
#SBATCH --job-name=full_vocab_region_eval
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
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

mkdir -p logs "${OUT_DIR}" models/hf_cache

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

STAGE="${STAGE:-all}"

python interference_experiment/full_vocab_region_eval.py \
    --stage              "${STAGE}" \
    --model_name         gpt2-xl \
    --dataset_name       wikitext \
    --dataset_config     wikitext-2-raw-v1 \
    --max_tokens         1000000 \
    --vocab_subset_size  20000 \
    --top_k_logits       50 \
    --seq_len            512 \
    --batch_size         4 \
    --layers             0,4,8,12,16,20,24,47 \
    --knn_edges          128 \
    --recursive_depth    2 \
    --recursive_min_size 300 \
    --min_modularity_gain 0.03 \
    --n_probe_max        200000 \
    --probe_epochs       100 \
    --probe_lr           1e-3 \
    --probe_batch_size   2048 \
    --output_dir         "${OUT_DIR}" \
    --seed               42

echo ""
echo "=============================="
echo "Saved artefacts:"
echo "=============================="
ls -lh "${OUT_DIR}/"
echo "Done at $(date)"
