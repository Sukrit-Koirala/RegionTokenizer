#!/bin/bash
#SBATCH --job-name=reg_retr_lm
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Region-Retrieval LM experiment.
#
# Hypothesis (retrieval-friendly representations):
#   Adding a metric-learning loss on the pre-region hidden states (proxy CE or
#   SupCon) encourages the LM to cluster token hidden states by next-token region,
#   making a kNN memory built from those states more accurate at region prediction.
#   An improved kNN region acc@1 confirms the representation is retrieval-friendly.
#
# What this script does:
#   Run 1 — repr_region reference (30k steps) — skip if checkpoint exists
#   Run 2 — repr_region_retrieval proxy  λ=0.05
#   Run 3 — repr_region_retrieval proxy  λ=0.10
#   Run 4 — repr_region_retrieval supcon λ=0.05
#
#   After each run: offline_region_knn.py (model_type=small) extracts a kNN
#   memory from the trained model's hidden states and reports region acc@1.
#
#   Final: comparison table written to runs/region_retrieval_comparison.md
#
# Success criteria (Q1–Q4):
#   Q1  proxy  λ=0.05 kNN acc@1 > reference kNN acc@1
#   Q2  proxy  λ=0.10 kNN acc@1 > reference kNN acc@1
#   Q3  supcon λ=0.05 kNN acc@1 > reference kNN acc@1
#   Q4  LM val_ppl of best retrieval run ≤ reference val_ppl + 0.5  (minimal regression)
#
# Config: wikitext-103-raw-v1, seq_len=256, batch=32, d_model=384, n_layer=6,
#         30k steps, lr=3e-4, region_map=128 regions, retrieval_dim=128
#
# Outputs: runs/repr_region_reference_30k/
#          runs/repr_region_retrieval_proxy_lam0p05/
#          runs/repr_region_retrieval_proxy_lam0p10/
#          runs/repr_region_retrieval_supcon_lam0p05/
#          runs/region_retrieval_comparison.md
#
# Memory estimate (L40S 48GB VRAM):
#   Small LM (d=384, 6 layers): ~10 MB
#   kNN memory (50k × 384 fp16): ~40 MB
#   Total: well within 48 GB

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
REF_DIR=runs/repr_region_reference_30k
RUN2_DIR=runs/repr_region_retrieval_proxy_lam0p05
RUN3_DIR=runs/repr_region_retrieval_proxy_lam0p10
RUN4_DIR=runs/repr_region_retrieval_supcon_lam0p05

if [[ ! -f "$REGION_MAP" ]]; then
    echo "ERROR: region map missing: $REGION_MAP" >&2; exit 1
fi

mkdir -p logs "$REF_DIR" "$RUN2_DIR" "$RUN3_DIR" "$RUN4_DIR"

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
"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PREFLIGHT
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "PREFLIGHT"
echo "=============================="

python - <<'PYEOF'
import sys, json, torch

try:
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    print("imports OK")
except ImportError as e:
    print(f"IMPORT ERROR: {e}", file=sys.stderr); sys.exit(1)

region_map_path = "runs/region_maps_128/token_to_region.json"
with open(region_map_path) as f:
    d = json.load(f)
n_regions = max(int(v) for v in d.values()) + 1
n_mapped  = len(d)
print(f"region map OK: {n_regions} regions  {n_mapped:,} mapped tokens")

sys.path.insert(0, ".")
from train_region_lm import (
    TrainConfig, ReprRegionTransformerLM, ReprRegionRetrievalLM, _supcon_loss
)
print("train_region_lm imports OK")

# Smoke-test ReprRegionRetrievalLM with proxy loss
import torch.nn.functional as F
cfg = TrainConfig(
    mode="repr_region_retrieval", d_model=64, n_layer=2, n_head=4, d_ff=256,
    n_coarse=n_regions, retrieval_loss_type="proxy", lambda_retrieval=0.05,
    retrieval_dim=32, retrieval_temp=0.07, steps=1,
)
coarse_arr = torch.full((50257,), -1, dtype=torch.long)
for tok_str, reg_id in d.items():
    tid = int(tok_str)
    if tid < 50257:
        coarse_arr[tid] = int(reg_id)
model = ReprRegionRetrievalLM(cfg, vocab_size=50257, coarse_map=coarse_arr)
idx = torch.randint(0, 50257, (2, 17))
losses = model.loss(idx)
assert "lm" in losses and "retrieval" in losses, f"missing keys: {list(losses.keys())}"
print(f"ReprRegionRetrievalLM proxy smoke test OK  "
      f"lm={losses['lm'].item():.4f}  retr={losses['retrieval'].item():.4f}  "
      f"r@1={losses['retrieval_acc1'].item():.3f}")

# Smoke-test supcon loss
z = F.normalize(torch.randn(32, 32), dim=-1)
lbl = torch.randint(0, 4, (32,))
scl = _supcon_loss(z, lbl, temp=0.07)
assert scl.item() > 0, "supcon loss should be > 0"
print(f"_supcon_loss smoke test OK  loss={scl.item():.4f}")

print("PREFLIGHT PASSED")
PYEOF

PREFLIGHT_EXIT=$?
if [[ "$PREFLIGHT_EXIT" -ne 0 ]]; then
    echo "PREFLIGHT FAILED (exit $PREFLIGHT_EXIT) — aborting job." >&2
    exit "$PREFLIGHT_EXIT"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# SHARED TRAINING ARGS
# ─────────────────────────────────────────────────────────────────────────────
COMMON_TRAIN_ARGS=(
    --dataset             wikitext-103-raw-v1
    --region_map_path     "$REGION_MAP"
    --seq_len             256
    --batch_size          32
    --d_model             384
    --n_layer             6
    --n_head              6
    --d_ff                1536
    --dropout             0.1
    --n_coarse            128
    --d_region            64
    --router_type         mlp
    --router_temp         2.0
    --base_alpha          0.1
    --region_warmup_steps 2000
    --lambda_coarse       0.2
    --lambda_balance      0.001
    --steps               30000
    --lr                  3e-4
    --weight_decay        0.1
    --grad_clip           1.0
    --warmup_steps        1000
    --eval_interval       1000
    --save_interval       5000
    --log_interval        100
    --seed                42
    --device              cuda
    --vocab_subset_size   50257
    --retrieval_dim       128
    --retrieval_temp      0.07
    --retrieval_key_source pre_region
    --max_retrieval_positions_per_batch 2048
)

COMMON_KNN_ARGS=(
    --model_type          small
    --dataset             wikitext-103-raw-v1
    --region_map_path     "$REGION_MAP"
    --knn_layer           block_5
    --max_memory_positions 50000
    --max_eval_positions  20000
    --knn_k               32
    --knn_temp            0.2
    --probe_temp          1.0
    --normalize_keys      true
    --batch_size          32
    --seq_len             256
    --device              cuda
    --seed                42
    --hf_cache_dir        "$HF_HOME"
)

# ─────────────────────────────────────────────────────────────────────────────
# RUN 1 — repr_region reference (30k steps)
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RUN 1 — repr_region reference"
echo "=============================="

if [[ -f "$REF_DIR/final_summary.json" ]]; then
    echo "Checkpoint exists at $REF_DIR — skipping training."
else
    echo "Start: $(date)"
    python train_region_lm.py "${COMMON_TRAIN_ARGS[@]}" \
        --mode        repr_region \
        --output_dir  "$REF_DIR"
    echo "End:   $(date)"
fi

echo ""
echo "--- Run 1 final_summary ---"
python -c "
import json
d = json.load(open('$REF_DIR/final_summary.json'))
print(f\"val_lm_loss={d['val_lm_loss']:.4f}  val_ppl={d['val_ppl']:.2f}  \
coarse_acc@1={d['val_coarse_acc1']:.3f}\")
"

echo ""
echo "--- kNN on Run 1 hidden states ---"
python scripts/offline_region_knn.py "${COMMON_KNN_ARGS[@]}" \
    --small_ckpt  "$REF_DIR/checkpoint_latest.pt" \
    --output_dir  "$REF_DIR/knn"
echo ""
echo "--- Run 1 kNN metrics_all.csv ---"
[[ -f "$REF_DIR/knn/metrics_all.csv" ]] && cat "$REF_DIR/knn/metrics_all.csv"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# RUN 2 — repr_region_retrieval proxy λ=0.05
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RUN 2 — repr_region_retrieval proxy λ=0.05"
echo "=============================="
echo "Start: $(date)"
python train_region_lm.py "${COMMON_TRAIN_ARGS[@]}" \
    --mode                 repr_region_retrieval \
    --retrieval_loss_type  proxy \
    --lambda_retrieval     0.05 \
    --output_dir           "$RUN2_DIR"
echo "End:   $(date)"

echo ""
echo "--- Run 2 final_summary ---"
python -c "
import json
d = json.load(open('$RUN2_DIR/final_summary.json'))
print(f\"val_lm_loss={d['val_lm_loss']:.4f}  val_ppl={d['val_ppl']:.2f}  \
coarse_acc@1={d['val_coarse_acc1']:.3f}\")
"

echo ""
echo "--- kNN on Run 2 hidden states ---"
python scripts/offline_region_knn.py "${COMMON_KNN_ARGS[@]}" \
    --small_ckpt  "$RUN2_DIR/checkpoint_latest.pt" \
    --output_dir  "$RUN2_DIR/knn"
echo ""
echo "--- Run 2 kNN metrics_all.csv ---"
[[ -f "$RUN2_DIR/knn/metrics_all.csv" ]] && cat "$RUN2_DIR/knn/metrics_all.csv"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# RUN 3 — repr_region_retrieval proxy λ=0.10
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RUN 3 — repr_region_retrieval proxy λ=0.10"
echo "=============================="
echo "Start: $(date)"
python train_region_lm.py "${COMMON_TRAIN_ARGS[@]}" \
    --mode                 repr_region_retrieval \
    --retrieval_loss_type  proxy \
    --lambda_retrieval     0.10 \
    --output_dir           "$RUN3_DIR"
echo "End:   $(date)"

echo ""
echo "--- Run 3 final_summary ---"
python -c "
import json
d = json.load(open('$RUN3_DIR/final_summary.json'))
print(f\"val_lm_loss={d['val_lm_loss']:.4f}  val_ppl={d['val_ppl']:.2f}  \
coarse_acc@1={d['val_coarse_acc1']:.3f}\")
"

echo ""
echo "--- kNN on Run 3 hidden states ---"
python scripts/offline_region_knn.py "${COMMON_KNN_ARGS[@]}" \
    --small_ckpt  "$RUN3_DIR/checkpoint_latest.pt" \
    --output_dir  "$RUN3_DIR/knn"
echo ""
echo "--- Run 3 kNN metrics_all.csv ---"
[[ -f "$RUN3_DIR/knn/metrics_all.csv" ]] && cat "$RUN3_DIR/knn/metrics_all.csv"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# RUN 4 — repr_region_retrieval supcon λ=0.05
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "RUN 4 — repr_region_retrieval supcon λ=0.05"
echo "=============================="
echo "Start: $(date)"
python train_region_lm.py "${COMMON_TRAIN_ARGS[@]}" \
    --mode                 repr_region_retrieval \
    --retrieval_loss_type  supcon \
    --lambda_retrieval     0.05 \
    --output_dir           "$RUN4_DIR"
echo "End:   $(date)"

echo ""
echo "--- Run 4 final_summary ---"
python -c "
import json
d = json.load(open('$RUN4_DIR/final_summary.json'))
print(f\"val_lm_loss={d['val_lm_loss']:.4f}  val_ppl={d['val_ppl']:.2f}  \
coarse_acc@1={d['val_coarse_acc1']:.3f}\")
"

echo ""
echo "--- kNN on Run 4 hidden states ---"
python scripts/offline_region_knn.py "${COMMON_KNN_ARGS[@]}" \
    --small_ckpt  "$RUN4_DIR/checkpoint_latest.pt" \
    --output_dir  "$RUN4_DIR/knn"
echo ""
echo "--- Run 4 kNN metrics_all.csv ---"
[[ -f "$RUN4_DIR/knn/metrics_all.csv" ]] && cat "$RUN4_DIR/knn/metrics_all.csv"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "COMPARISON TABLE"
echo "=============================="

python - <<'PYEOF'
import json, os, csv

runs = [
    ("reference",        "runs/repr_region_reference_30k"),
    ("proxy λ=0.05",     "runs/repr_region_retrieval_proxy_lam0p05"),
    ("proxy λ=0.10",     "runs/repr_region_retrieval_proxy_lam0p10"),
    ("supcon λ=0.05",    "runs/repr_region_retrieval_supcon_lam0p05"),
]

def _read_summary(d):
    p = os.path.join(d, "final_summary.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))

def _read_knn(d):
    p = os.path.join(d, "knn", "metrics_all.csv")
    if not os.path.exists(p):
        return {}
    rows = list(csv.DictReader(open(p)))
    # router row and mem row
    out = {}
    for r in rows:
        if r.get("policy") == "mem_only" or r.get("label") == "mem_only":
            out["knn_acc1"] = float(r.get("region_acc1", r.get("acc1", 0)))
        if r.get("policy") == "router" or r.get("label") == "router":
            out["router_acc1"] = float(r.get("region_acc1", r.get("acc1", 0)))
    return out

lines = [
    "# Region-Retrieval LM Comparison\n",
    f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
    "",
    "| Run | val_ppl | coarse_acc@1 | kNN acc@1 | router acc@1 |",
    "|-----|---------|--------------|-----------|--------------|",
]

ref_ppl = None
ref_knn = None

for label, d in runs:
    s = _read_summary(d)
    k = _read_knn(d)
    ppl     = s.get("val_ppl",          float("nan"))
    cacc    = s.get("val_coarse_acc1",  float("nan"))
    knn_a1  = k.get("knn_acc1",         float("nan"))
    rtr_a1  = k.get("router_acc1",      float("nan"))
    if label == "reference":
        ref_ppl = ppl
        ref_knn = knn_a1
    lines.append(f"| {label:<18} | {ppl:7.2f} | {cacc:12.3f} | {knn_a1:9.3f} | {rtr_a1:12.3f} |")

lines += [
    "",
    "## Hypothesis Tests",
    "",
    "| Q | Question | Verdict |",
    "|---|----------|---------|",
]

all_data = [(label, _read_summary(d), _read_knn(d)) for label, d in runs]

def _knn(label):
    for l, _, k in all_data:
        if l == label: return k.get("knn_acc1", float("nan"))
    return float("nan")

ref_k = _knn("reference")
q1 = _knn("proxy λ=0.05") > ref_k if ref_k == ref_k else False
q2 = _knn("proxy λ=0.10") > ref_k if ref_k == ref_k else False
q3 = _knn("supcon λ=0.05") > ref_k if ref_k == ref_k else False

best_ppl = min((s.get("val_ppl", float("inf")) for _, s, _ in all_data[1:]), default=float("inf"))
q4 = best_ppl <= (ref_ppl + 0.5) if ref_ppl is not None and ref_ppl == ref_ppl else False

lines += [
    f"| Q1 | proxy λ=0.05 kNN acc@1 > reference | {'PASS' if q1 else 'FAIL'} |",
    f"| Q2 | proxy λ=0.10 kNN acc@1 > reference | {'PASS' if q2 else 'FAIL'} |",
    f"| Q3 | supcon λ=0.05 kNN acc@1 > reference | {'PASS' if q3 else 'FAIL'} |",
    f"| Q4 | best retrieval val_ppl ≤ ref + 0.5  | {'PASS' if q4 else 'FAIL'} |",
    "",
    "## Interpretation",
    "",
]
n_pass = sum([q1, q2, q3, q4])
if n_pass >= 3:
    lines.append("Retrieval-friendly training **improves kNN region accuracy** with minimal LM "
                 "regression. The proxy loss is a viable auxiliary objective.")
elif n_pass >= 2:
    lines.append("Mixed results: some retrieval runs improve kNN accuracy. "
                 "Consider tuning λ or combining proxy + SupCon.")
else:
    lines.append("Retrieval loss did not reliably improve kNN region accuracy at these λ values. "
                 "Consider larger λ, a different projection dim, or post_region key source.")

out_path = "runs/region_retrieval_comparison.md"
os.makedirs("runs", exist_ok=True)
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\nComparison written to {out_path}")
PYEOF

echo ""
echo "=============================="
echo "End: $(date)"
echo "=============================="
echo "Job $SLURM_JOB_ID COMPLETE."
