#!/bin/bash
#SBATCH --job-name=agnews_bpe
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-agnews-bpe-%j.out
#SBATCH --error=slurm-agnews-bpe-%j.err

source ~/miniconda3/bin/activate
conda activate learned_regions

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD

# ─────────────────────────────────────────────────────────────────────────────
# BPE baseline — small GPT trained on AG News with a custom BPE tokenizer.
#
# Purpose:
#   Baseline run for the RegionTokenizer project.  Trains a ByteLevel BPE
#   tokenizer from scratch on AG News then fits a small GPT-2 on the same
#   data.  Serves as the comparison point for any region-aware tokenizer runs.
#
# Architecture:
#   n_layers = 6   d_model = 384   n_heads = 6
#   GPT-2 (HuggingFace), randomly initialised, causal LM objective
#
# Training:
#   Dataset   : ag_news (HuggingFace, ~120K train / 7.6K test)
#   Tokenizer : BPE, vocab_size=16K, ByteLevel pre-tokenizer
#               saved to outputs/agnews_bpe/tokenizer.json — skipped on rerun
#   Packing   : all texts concatenated and chunked into 256-token blocks
#   Optimizer : AdamW  wd=0.01
#   LR        : 3e-4, cosine decay with warmup (5% warmup)
#   Batch     : 32 × 256 tokens
#   Epochs    : 20 (train to convergence)
#
# Outputs:
#   outputs/agnews_bpe/tokenizer.json
#   outputs/agnews_bpe/final_model/
#   outputs/agnews_bpe/metrics.json   (eval_loss, perplexity)
# ─────────────────────────────────────────────────────────────────────────────

python scripts/run.py \
    --config configs/agnews_bpe.yaml \
    --outdir outputs/agnews_bpe
