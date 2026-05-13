#!/bin/bash
#SBATCH --job-name=region_knn
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Offline Region-kNN diagnostic experiment.
#
# Hypothesis (kNN region consistency):
#   At the penultimate GPT-2 XL layer (block_41), hidden states that are
#   nearest neighbours in representation space share region membership more
#   often than chance.  A weighted kNN vote over retrieved region labels
#   (p_mem) should complement the router's own p_region — especially for
#   boundary tokens where the router margin is low.
#
# What this script does:
#   1. Loads GPT-2 XL (fp16) + RegionProbe for block_41 from probes.pt.
#   2. Sweeps 500k WikiText-103 training positions to build a FAISS
#      IndexFlatIP memory of (normalized hidden state → region label).
#   3. Evaluates on 247k validation positions:
#      - Retrieves 64 nearest neighbours, computes weighted region vote p_mem.
#      - Tests 5 beta-blend policies (fixed 0.25/0.50/0.75, margin-based,
#        memory-confidence-based) mixing p_router and p_mem.
#      - Splits results by routing margin: all / core / medium / boundary /
#        tight_boundary.
#      - Computes region consistency types A/B/C.
#      - Evaluates adaptive candidate coverage policy.
#   4. Writes: summary.md, metrics_all.csv, metrics_by_split.csv,
#              region_consistency_types.csv, candidate_coverage.csv,
#              knn_config.json, plots/01-07.
#
# Q1-Q6 hypothesis tests in summary.md:
#   Q1  kNN region accuracy > router accuracy overall
#   Q2  kNN region accuracy > router accuracy for boundary tokens
#   Q3  Optimal beta > 0 (p_mem helps even for core tokens)
#   Q4  Margin-adaptive beta outperforms best fixed beta
#   Q5  Memory-confidence beta outperforms best fixed beta
#   Q6  Adaptive candidate coverage improves accuracy vs top1 router alone
#
# Outputs: runs/offline_region_knn_gpt2xl_block41/
#
# Config: 500k memory positions, 247k eval positions, seq_len=512,
#         batch=4, knn_k=64, knn_temp=0.2, probe_temp=1.0,
#         normalize_keys=true, knn_layer=block_41
#
# Memory estimate (L40S 48GB VRAM):
#   GPT-2 XL (fp16):       ~3 GB
#   FAISS index (fp32):    ~3.2 GB  (500k × 1600 × 4 bytes)
#   Eval hidden states:    ~320 MB  per batch (4 × 512 × 1600 × 4 bytes)
#   Total:                 well within 48 GB

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

REGION_MAP=runs/region_maps_128/token_to_region.json
PROBE_PATH=runs/gpt2xl_layer_margin/probes.pt
OUTPUT_DIR=runs/offline_region_knn_gpt2xl_block41

if [[ ! -f "$REGION_MAP" ]]; then
    echo "ERROR: region map missing: $REGION_MAP" >&2; exit 1
fi
if [[ ! -f "$PROBE_PATH" ]]; then
    echo "ERROR: probe checkpoint missing: $PROBE_PATH" >&2
    echo "       Run slurm_gpt2xl_layer_margin_analysis.sh first." >&2
    exit 1
fi
if [[ ! -f "scripts/offline_region_knn.py" ]]; then
    echo "ERROR: scripts/offline_region_knn.py not found in $PWD" >&2; exit 1
fi

mkdir -p logs "$OUTPUT_DIR"

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "
import torch
print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda,
      '| device:', torch.cuda.get_device_name(0))
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('gpt2-xl', cache_dir='$HF_HOME')
print(f'GPT-2 XL config: n_embd={cfg.n_embd}  n_layer={cfg.n_layer}  '
      f'vocab_size={cfg.vocab_size}')
"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# QUICK PREFLIGHT — verify imports, probes, region map before 12h run
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "PREFLIGHT"
echo "=============================="

python - <<'PYEOF'
import sys, json, os, torch

# Check imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import numpy, matplotlib
    print("imports OK")
except ImportError as e:
    print(f"IMPORT ERROR: {e}", file=sys.stderr); sys.exit(1)

try:
    import faiss
    print(f"faiss OK: version={faiss.__version__}")
except ImportError:
    print("faiss not installed — will use torch brute-force fallback")

# Check region map
region_map_path = "runs/region_maps_128/token_to_region.json"
with open(region_map_path) as f:
    d = json.load(f)
n_regions = max(int(v) for v in d.values()) + 1
n_mapped  = len(d)
print(f"region map OK: {n_regions} regions  {n_mapped:,} mapped tokens")

# Check probe checkpoint
probe_path = "runs/gpt2xl_layer_margin/probes.pt"
ckpt = torch.load(probe_path, map_location="cpu")
n_probes = len(ckpt["probes"])
print(f"probes.pt OK: {n_probes} probes loaded")

# Layer block_41 → probe index 42
target_probe_idx = 42
probe_sd = ckpt["probes"][target_probe_idx]
probe_keys = list(probe_sd.keys())
print(f"block_41 probe (idx={target_probe_idx}) keys: {probe_keys}")

# Check script is importable
sys.path.insert(0, "scripts")
from offline_region_knn import MemoryIndex, TokenChunkDataset
print("offline_region_knn imports OK")

# Smoke-test MemoryIndex with tiny data
idx = MemoryIndex(d_model=8, device=torch.device("cpu"), normalize=True)
keys = torch.randn(20, 8)
idx.build(keys)
sims, inds = idx.search(keys[:3], k=5)
assert sims.shape == (3, 5), f"shape mismatch: {sims.shape}"
print(f"MemoryIndex smoke test OK: sims={sims.shape}  inds={inds.shape}")

print("PREFLIGHT PASSED")
PYEOF

PREFLIGHT_EXIT=$?
if [[ "$PREFLIGHT_EXIT" -ne 0 ]]; then
    echo "PREFLIGHT FAILED (exit $PREFLIGHT_EXIT) — aborting job." >&2
    exit "$PREFLIGHT_EXIT"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RUNNING OFFLINE REGION-kNN ANALYSIS"
echo "=============================="
echo "Backbone:        GPT-2 XL (block_41)"
echo "Region map:      $REGION_MAP"
echo "Probe path:      $PROBE_PATH"
echo "Output dir:      $OUTPUT_DIR"
echo "Memory positions: 500,000  |  Eval positions: 247,000"
echo "knn_k=64  knn_temp=0.2  normalize_keys=true"
echo "Start: $(date)"
echo ""

python scripts/offline_region_knn.py \
    --model_type          gpt2xl \
    --model_name          gpt2-xl \
    --dataset             wikitext-103-raw-v1 \
    --region_map_path     "$REGION_MAP" \
    --probe_path          "$PROBE_PATH" \
    --knn_layer           block_41 \
    --max_memory_positions 500000 \
    --max_eval_positions  247000 \
    --knn_k               64 \
    --knn_temp            0.2 \
    --probe_temp          1.0 \
    --normalize_keys      true \
    --output_dir          "$OUTPUT_DIR" \
    --batch_size          4 \
    --seq_len             512 \
    --device              cuda \
    --seed                42 \
    --hf_cache_dir        "$HF_HOME"

ANALYSIS_EXIT=$?

echo ""
echo "=============================="
echo "End: $(date)"
echo "=============================="

if [[ "$ANALYSIS_EXIT" -ne 0 ]]; then
    echo "ANALYSIS FAILED (exit $ANALYSIS_EXIT)" >&2
    exit "$ANALYSIS_EXIT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "RESULTS"
echo "=============================="

if [[ -f "$OUTPUT_DIR/summary.md" ]]; then
    echo "--- Hypothesis test verdicts ---"
    grep -A 14 "Hypothesis Tests" "$OUTPUT_DIR/summary.md" | head -20 || true
    echo ""
    echo "--- Interpretation ---"
    grep -A 8 "## Interpretation" "$OUTPUT_DIR/summary.md" | head -12 || true
fi

echo ""
echo "--- Output files ---"
ls -lh "$OUTPUT_DIR/"*.csv "$OUTPUT_DIR/"*.json "$OUTPUT_DIR/summary.md" 2>/dev/null || true
echo ""
echo "--- Plots ---"
ls -lh "$OUTPUT_DIR/plots/" 2>/dev/null || true

echo ""
echo "--- metrics_all.csv (all rows) ---"
if [[ -f "$OUTPUT_DIR/metrics_all.csv" ]]; then
    cat "$OUTPUT_DIR/metrics_all.csv"
fi

echo ""
echo "--- metrics_by_split.csv (first 20 rows) ---"
if [[ -f "$OUTPUT_DIR/metrics_by_split.csv" ]]; then
    head -21 "$OUTPUT_DIR/metrics_by_split.csv"
fi

echo ""
echo "--- region_consistency_types.csv ---"
if [[ -f "$OUTPUT_DIR/region_consistency_types.csv" ]]; then
    cat "$OUTPUT_DIR/region_consistency_types.csv"
fi

echo ""
echo "--- candidate_coverage.csv ---"
if [[ -f "$OUTPUT_DIR/candidate_coverage.csv" ]]; then
    cat "$OUTPUT_DIR/candidate_coverage.csv"
fi

echo ""
echo "Job $SLURM_JOB_ID COMPLETE."
