#!/bin/bash
#SBATCH --job-name=ctf_gold_leak
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# CTF Gold Leakage Diagnostic
#
# Tests whether the large +0.255 gated gain at step 1000 is caused by
# oracle gold force-inclusion in the candidate selection.
#
# Critical comparison:
#   Mode 1 (oracle_force_gold) — reproduces current eval  [ORACLE / NOT INFERENCE VALID]
#   Mode 2 (no_force_gold)     — true inference metric    [CANONICAL]
#
# If oracle_gain - no_force_gain >= 0.05 → LEAKAGE CONFIRMED
# If gap < 0.01 and no_force beats MLP  → NO LEAKAGE EVIDENT
#
# All 8 modes evaluated on full val set (239,362 examples):
#   0. force_zero            — base logits only (anchor)
#   1. oracle_force_gold     — CTF with gold insertion [ORACLE]
#   2. no_force_gold         — CTF top-M only [TRUE INFERENCE]
#   3. random_force_slot     — CTF with random insertion slot (5 seeds)
#   4. forced_delta_zero     — oracle + zero delta on forced-gold slot
#   5. natural_only          — no-force, NLL only on natural-gold positions
#   6. forced_only_oracle    — oracle, NLL only on forced positions
#   6b. forced_only          — no-force, NLL only on forced positions
#   7. shuffled_gold         — wrong candidate inserted, true gold CE
#
# Outputs:
#   runs/path_refiner_candidate_transformer/gold_leakage_debug/
#     gold_leakage_metrics.csv
#     gold_leakage_subset_metrics.csv
#     gold_leakage_report.md
#     all_modes.json
#     config.json
#
# Pre-requisites: CTF boundary run must have completed and produced best_refiner.pt

source ~/miniconda3/bin/activate
conda activate learned_regions

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME="$SLURM_SUBMIT_DIR/models/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

cd ~/ondemand/upload_me/RegionTokenizer
export PYTHONPATH=$PWD
mkdir -p logs

# ── Paths ────────────────────────────────────────────────────────────────────

SMALL_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
SUPER_MAP=runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json
VAL_DIR=runs/path_refiner_clean/data/val_hgrid_K24
BASELINE_JSON=runs/path_refiner_clean/baselines/saved_candidate_baseline.json
CTF_CKPT=runs/path_refiner_candidate_transformer/variant_CTF_boundary_M256/best_refiner.pt
OUTPUT_DIR=runs/path_refiner_candidate_transformer/gold_leakage_debug

# ── Preflight ────────────────────────────────────────────────────────────────

echo "=== CTF Gold Leakage Diagnostic  $(date) ==="
echo "    CTF_CKPT    : $CTF_CKPT"
echo "    VAL_DIR     : $VAL_DIR"
echo "    OUTPUT_DIR  : $OUTPUT_DIR"

for f in "$SMALL_CKPT" "$SUPER_MAP" "$BASELINE_JSON"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done
if [[ ! -f "$CTF_CKPT" ]]; then
    echo "ERROR: CTF checkpoint missing: $CTF_CKPT" >&2
    echo "       Run slurm_train_candidate_transformer_refiner.sh first." >&2
    exit 1
fi

N_VAL=$(find "$VAL_DIR" -maxdepth 1 -name "shard_*.pt" 2>/dev/null | wc -l)
echo "    val shards  : $N_VAL"
if [[ "$N_VAL" -eq 0 ]]; then
    echo "ERROR: no val shards in $VAL_DIR" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Run diagnostic ───────────────────────────────────────────────────────────

echo ""
echo "Running 8-mode gold leakage diagnostic ..."
echo "(Expect ~30-40 min for all 8 modes on 24 val shards)"
echo ""

python scripts/debug_ctf_gold_leakage.py         \
    --small_ckpt        $SMALL_CKPT               \
    --val_dir           $VAL_DIR                  \
    --baseline_json     $BASELINE_JSON            \
    --ctf_ckpt          $CTF_CKPT                 \
    --output_dir        $OUTPUT_DIR               \
    --super_map         $SUPER_MAP                \
    --selected_M        256                       \
    --gate_filter       boundary                  \
    --batch_size        64                        \
    --fail_on_baseline_mismatch                   \
    --device            cuda

# ── Quick summary ────────────────────────────────────────────────────────────

echo ""
echo "=== Diagnostic complete  $(date) ==="
echo ""
echo "Quick results:"

python - <<'PYEOF'
import json, os

out = "runs/path_refiner_candidate_transformer/gold_leakage_debug"
f   = os.path.join(out, "all_modes.json")
if not os.path.isfile(f):
    print("  all_modes.json not found")
    exit(0)

data = json.load(open(f))
baseline_nll = 3.378606

print(f"\n  {'mode':35s}  {'gated_nll':>10}  {'gain':>8}  {'force_rate':>10}")
print(f"  {'-'*35}  {'-'*10}  {'-'*8}  {'-'*10}")

for mode in ["force_zero", "oracle_force_gold", "no_force_gold",
             "random_force_slot", "forced_delta_zero",
             "natural_only", "forced_only", "forced_only_oracle", "shuffled_gold"]:
    if mode not in data:
        continue
    r      = data[mode]
    gnll   = r.get("gated_covered_nll", r.get("nll", float("nan")))
    gain   = baseline_nll - gnll
    frate  = r.get("gold_force_included_rate", 0.0)
    oracle = " [ORACLE]" if mode in ("oracle_force_gold", "random_force_slot",
                                      "forced_delta_zero", "forced_only_oracle",
                                      "shuffled_gold") else ""
    print(f"  {mode:35s}  {gnll:10.6f}  {gain:+8.6f}  {frate:10.4f}{oracle}")

# Read and print verdict from report
report = os.path.join(out, "gold_leakage_report.md")
if os.path.isfile(report):
    for line in open(report):
        if "LEAKAGE" in line or "EVIDENCE" in line or "INCONCLUSIVE" in line:
            print(f"\n  VERDICT: {line.strip()}")
            break
PYEOF

echo ""
echo "Full report: $OUTPUT_DIR/gold_leakage_report.md"
echo "All metrics: $OUTPUT_DIR/gold_leakage_metrics.csv"
