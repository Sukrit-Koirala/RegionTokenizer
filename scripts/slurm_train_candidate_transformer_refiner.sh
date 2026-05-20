#!/bin/bash
#SBATCH --job-name=ctf_train
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
# Candidate-set Transformer Refiner (CTF) training — boundary filter, M=256.
# LEAKAGE-FIXED: selection_mode=topm_no_gold (gold never consulted at eval/train).
#
# Architecture:
#   Sequence  : [CTX][ROUTER][MEM][CAND_0 .. CAND_{M-1}]  length = 3 + M
#   Refiner   : TransformerEncoder  L=2  H=4  D=256  ff_mult=4  (norm_first=True)
#   Init      : residual_scale=1.0 + zero final Linear  → delta=0 at step 0
#   Selection : topm_no_gold — top-M by base logit only. Gold NEVER inserted.
#   Scatter   : full_delta.scatter_(1, sel_idx, delta_sel)  — zero outside M
#
# Gated global eval (primary metric):
#   gate = boundary positions only
#   gated_scores = model_scores inside gate, base_scores outside
#   gated_covered_nll reported; fingerprint/coverage unchanged vs force-zero
#   gold_force_included_rate = 0.0000  (hard assertion — any nonzero aborts)
#
# ALL full-val evals assert:
#   fingerprint  = f57cabcdc46d69ce
#   num_examples = 239,362
#   num_covered  = 227,017
#   coverage     = 0.948425
#
# Comparison targets:
#   force-zero baseline           : covered_nll = 3.378606
#   global MLP refiner (variant C): gated gain  = +0.001284
#   hard boundary MLP             : gated gain  = +0.000530  local_gain = +0.003209
#
# Run order:
#   slurm_clean_build_datasets.sh
#   → slurm_debug_periodic_eval_consistency.sh
#   → slurm_debug_ctf_no_leakage.sh   ← MUST PASS (confirms force_rate=0)
#   → THIS SCRIPT

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
mkdir -p logs

# ── Configurable paths ────────────────────────────────────────────────────────

SMALL_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
DATA_ROOT=runs/path_refiner_clean/data
OUTPUT_ROOT=runs/path_refiner_candidate_transformer
BASELINE_JSON=runs/path_refiner_clean/baselines/saved_candidate_baseline.json

VAL_DIR=$DATA_ROOT/val_hgrid_K24
TRAIN_DIR=$DATA_ROOT/train_hgrid_K24

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== CTF Training  $(date) ==="
echo "    TRAIN_DIR    : $TRAIN_DIR"
echo "    VAL_DIR      : $VAL_DIR"
echo "    BASELINE_JSON: $BASELINE_JSON"

for f in "$SMALL_CKPT" "$SUPER_MAP"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done
if [[ ! -f "$BASELINE_JSON" ]]; then
    echo "ERROR: $BASELINE_JSON missing — run baseline eval first." >&2; exit 1
fi

N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
N_TRAIN=$(find "$TRAIN_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    val shards   : $N_VAL"
echo "    train shards : $N_TRAIN"
if [[ "$N_VAL" -eq 0 || "$N_TRAIN" -eq 0 ]]; then
    echo "ERROR: missing shards. Run slurm_clean_build_datasets.sh first." >&2; exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# ── Shared args ───────────────────────────────────────────────────────────────

SHARED="
    --val_dir                    $VAL_DIR
    --train_dir                  $TRAIN_DIR
    --small_ckpt                 $SMALL_CKPT
    --super_map                  $SUPER_MAP
    --official_baseline          $BASELINE_JSON
    --selected_M                 256
    --refiner_dim                256
    --num_layers                 2
    --num_heads                  4
    --ff_mult                    4
    --dropout                    0.0
    --steps                      10000
    --eval_every                 1000
    --eval_batch_size            64
    --lr                         3e-4
    --lambda_kl                  0.01
    --lambda_delta               1e-4
    --grad_clip                  1.0
    --train_selection_mode       topm_no_gold
    --eval_selection_mode        topm_no_gold
    --drop_train_gold_not_selected
    --eval_before_train
    --fail_on_baseline_mismatch
    --device                     cuda
"

# ── Run 1: boundary (primary) ─────────────────────────────────────────────────

echo ""
echo "=== Run 1/3: boundary  $(date) ==="

python scripts/train_candidate_transformer_refiner.py \
    $SHARED                                           \
    --train_filter  boundary                          \
    --gate_filter   boundary                          \
    --batch_size    32                                \
    --output_dir    $OUTPUT_ROOT/variant_CTF_boundary_M256_noforce

# ── Run 2: hard_union ─────────────────────────────────────────────────────────

echo ""
echo "=== Run 2/3: hard_union  $(date) ==="

python scripts/train_candidate_transformer_refiner.py \
    $SHARED                                           \
    --train_filter  hard_union                        \
    --gate_filter   hard_union                        \
    --batch_size    32                                \
    --output_dir    $OUTPUT_ROOT/variant_CTF_hard_union_M256_noforce

# ── Run 3: type_A ─────────────────────────────────────────────────────────────

echo ""
echo "=== Run 3/3: type_A  $(date) ==="

python scripts/train_candidate_transformer_refiner.py \
    $SHARED                                           \
    --train_filter  type_A                            \
    --gate_filter   type_A                            \
    --batch_size    32                                \
    --output_dir    $OUTPUT_ROOT/variant_CTF_type_A_M256_noforce

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== All CTF runs complete  $(date) ==="
echo ""
echo "Compare best_metrics.json across runs:"
echo "  Key metric : gated_covered_nll  (lower is better)"
echo "  Baseline   : covered_nll=3.378606  (official force-zero)"
echo "  MLP global : gated gain=+0.001284"
echo "  MLP hard   : gated gain=+0.000530  local_gain=+0.003209 (boundary)"
echo ""
echo "  All _noforce runs: selection_mode=topm_no_gold  gold_force_included_rate=0.0000"
echo "  Old _M256 runs (no _noforce suffix) are INVALID due to gold leakage."
echo ""
echo "Quick summary:"
for d in "$OUTPUT_ROOT"/variant_CTF_*_noforce; do
    f="$d/best_metrics.json"
    if [[ -f "$f" ]]; then
        variant=$(basename "$d" | sed 's/variant_CTF_//')
        nll=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gated_covered_nll']:.6f}\")" 2>/dev/null || echo "N/A")
        delta=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['delta_vs_baseline']:+.6f}\")" 2>/dev/null || echo "N/A")
        step=$(python -c "import json; d=json.load(open('$f')); print(d['step'])" 2>/dev/null || echo "N/A")
        gate=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gate_rate']:.4f}\")" 2>/dev/null || echo "N/A")
        gfir=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gold_force_included_rate']:.4f}\")" 2>/dev/null || echo "N/A")
        echo "  ${variant:35}  gated_nll=${nll}  delta=${delta}  step=${step}  gate_rate=${gate}  gold_forced=${gfir}"
    fi
done
echo ""
echo "Questions to answer from results:"
echo "  Q1: Does CTF beat hard boundary MLP? (target: delta > +0.000530)"
echo "  Q2: Does CTF beat global MLP?        (target: delta > +0.001284)"
echo "  Q3: Inside-boundary NLL vs MLP?      (check local_subset_eval.csv)"
echo "  Q4: router_top8_miss improvement?"
echo "  Q5: Gold outside selected M? What fraction?  (gold_force_included_rate)"
echo "  Q6: hard_union CTF vs hard_union MLP?"
echo ""
echo "For per-subset breakdown, inspect local_subset_eval.csv in each run directory."
