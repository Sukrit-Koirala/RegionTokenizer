#!/bin/bash
#SBATCH --job-name=gpt2xl_margin
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
# GPT-2 XL layer-wise region probe analysis.
#
# Hypothesis (progressive manifold resolution):
#   GPT-2 XL uses early layers to coarsely localize token predictions to a
#   neighbourhood in representation space, then later layers to sharpen
#   commitment to the top-1 region.  Measured by probing all 49 hidden
#   states (embed + 48 blocks) with lightweight RegionProbe modules
#   trained on WikiText-103.
#
# What this script does:
#   1. Trains 49 per-layer linear probes (d_model=1600 → n_regions) on
#      500k WikiText-103 training positions (3 epochs, Adam lr=1e-3).
#   2. Evaluates all probes on 247k validation positions.
#   3. Computes: acc@1/4/8/16, margin, entropy, boundary_frac (τ=0.10 and τ=0.03),
#      gold rank, token context variance (Welford), margin trajectories by
#      initial-ambiguity group.
#   4. Writes: layer_metrics.csv, layer_trajectories.csv,
#              token_margin_variance.csv, probe_training_loss.csv,
#              summary.md, plots/01-09.
#
# Q1-Q8 hypothesis tests in summary.md:
#   Q1  Gold region enters top-4 early (above 2× chance at embedding layer)
#   Q2  Top-1 sharpens later than top-4 (relative gap narrows)
#   Q3  Entropy decreases progressively (final < embed − 0.1)
#   Q4  Margins increase progressively (final > embed + 0.02)
#   Q5  Persistently ambiguous tokens remain low-margin at final layer
#   Q6  Top-1 accuracy gains non-trivially in second half of network
#   Q7  Context dynamically sharpens margin (high std_margin tokens)
#   Q8  Sharpening elbow (largest Δmargin) is in second half of network
#
# Outputs: runs/gpt2xl_layer_margin/
#
# Config: 500k train positions, 247k eval positions, seq_len=512,
#         batch=4, probe_epochs=3, lr=1e-3, boundary_tau=0.10,
#         boundary_tau_tight=0.03
#
# GPT-2 XL memory estimate:
#   Model (fp16):  ~3 GB
#   Hidden states: ~320 MB per batch (49 × 4 × 512 × 1600 × 2 bytes)
#   Probes:        ~75 MB (49 probes × ~1.5 MB each)
#   Total:         well within 48 GB L40S VRAM

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
OUTPUT_DIR=runs/gpt2xl_layer_margin

if [[ ! -f "$REGION_MAP" ]]; then
    echo "ERROR: region map missing: $REGION_MAP" >&2; exit 1
fi
if [[ ! -f "scripts/gpt2xl_layer_margin_analysis.py" ]]; then
    echo "ERROR: scripts/gpt2xl_layer_margin_analysis.py not found in $PWD" >&2; exit 1
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
# QUICK PREFLIGHT — verify imports + region map + region count before 12h run
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "PREFLIGHT"
echo "=============================="

python - <<'PYEOF'
import sys, json, torch
import torch.nn as nn

# Check imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import matplotlib; import numpy
    print("imports OK")
except ImportError as e:
    print(f"IMPORT ERROR: {e}", file=sys.stderr); sys.exit(1)

# Check region map
import os
region_map_path = "runs/region_maps_128/token_to_region.json"
if not os.path.exists(region_map_path):
    print(f"ERROR: region map not found: {region_map_path}", file=sys.stderr)
    sys.exit(1)

with open(region_map_path) as f:
    d = json.load(f)
n_regions = max(int(v) for v in d.values()) + 1
n_mapped  = len(d)
print(f"region map OK: {n_regions} regions  {n_mapped:,} mapped tokens")

# Verify RegionProbe class
sys.path.insert(0, "scripts")
from gpt2xl_layer_margin_analysis import RegionProbe, load_region_map, TokenChunkDataset
probe = RegionProbe(1600, n_regions, "ln_linear")
x = torch.randn(4, 1600)
out = probe(x)
assert out.shape == (4, n_regions), f"shape mismatch: {out.shape}"
print(f"RegionProbe OK: d_model=1600  n_regions={n_regions}  "
      f"output={out.shape}")

# Quick parameter count
n_params = sum(p.numel() for p in probe.parameters())
print(f"per-probe params: {n_params:,}  ×49 probes = {n_params * 49:,}")
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
echo "RUNNING GPT-2 XL LAYER MARGIN ANALYSIS"
echo "=============================="
echo "Output dir: $OUTPUT_DIR"
echo "Region map: $REGION_MAP"
echo "Train positions: 500,000  |  Eval positions: 247,000"
echo "Probe: ln_linear  lr=1e-3  epochs=3  temp=1.0"
echo "Boundary τ: 0.10 (standard)  0.03 (tight)"
echo "Start: $(date)"
echo ""

python scripts/gpt2xl_layer_margin_analysis.py \
    --model_name          gpt2-xl \
    --dataset             wikitext-103-raw-v1 \
    --region_map_path     "$REGION_MAP" \
    --output_dir          "$OUTPUT_DIR" \
    --max_train_positions 500000 \
    --max_eval_positions  247000 \
    --probe_type          ln_linear \
    --probe_lr            1e-3 \
    --probe_epochs        3 \
    --probe_temp          1.0 \
    --batch_size          4 \
    --seq_len             512 \
    --boundary_tau        0.10 \
    --boundary_tau_tight  0.03 \
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
    # Print verdict table from summary.md
    echo "--- Hypothesis test verdicts ---"
    grep -A 12 "Hypothesis Tests" "$OUTPUT_DIR/summary.md" | head -20 || true
    echo ""
    echo "--- Interpretation ---"
    grep -A 8 "## Interpretation" "$OUTPUT_DIR/summary.md" | head -12 || true
fi

echo ""
echo "--- Output files ---"
ls -lh "$OUTPUT_DIR/"*.csv "$OUTPUT_DIR/summary.md" 2>/dev/null || true
echo ""
echo "--- Plots ---"
ls -lh "$OUTPUT_DIR/plots/" 2>/dev/null || true

echo ""
echo "--- First 5 rows of layer_metrics.csv ---"
if [[ -f "$OUTPUT_DIR/layer_metrics.csv" ]]; then
    head -6 "$OUTPUT_DIR/layer_metrics.csv"
fi

echo ""
echo "--- Probe training loss (final epoch) ---"
python - <<PYEOF2
import csv, os
path = "$OUTPUT_DIR/probe_training_loss.csv"
if not os.path.exists(path):
    print("probe_training_loss.csv not found")
else:
    rows = list(csv.DictReader(open(path)))
    max_ep = max(int(r["epoch"]) for r in rows)
    final  = [r for r in rows if int(r["epoch"]) == max_ep]
    print(f"Epoch {max_ep+1} cross-entropy losses:")
    for r in final[::6]:  # every 6th layer
        loss_val = float(r["loss"]) if r["loss"] != "nan" else float("nan")
        print(f"  {r['layer_name']:<12}: {loss_val:.4f}")
PYEOF2

echo ""
echo "Job $SLURM_JOB_ID COMPLETE."
