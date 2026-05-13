#!/bin/bash
#SBATCH --job-name=knn_keysrc
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Region-kNN key-source sweep.
#
# Does NOT retrain models — only reruns offline kNN evaluation.
#
# Scientific question:
#   Did repr_region_retrieval models learn a better region-neighbor space,
#   and must we use z = retrieval_proj(h) as the kNN key to see it?
#
# Runs 7 offline kNN evaluations:
#   1. reference                  raw_h
#   2. proxy λ=0.05               raw_h
#   3. proxy λ=0.05               retrieval_proj
#   4. proxy λ=0.10               raw_h
#   5. proxy λ=0.10               retrieval_proj
#   6. supcon λ=0.05              raw_h
#   7. supcon λ=0.05              retrieval_proj
#
# Hypothesis tests (Q1–Q6) in runs/region_knn_keysource_comparison.md:
#   Q1  proxy λ=0.05  retrieval_proj > proxy λ=0.05  raw_h
#   Q2  proxy λ=0.10  retrieval_proj > proxy λ=0.10  raw_h
#   Q3  supcon λ=0.05 retrieval_proj > supcon λ=0.05 raw_h
#   Q4  best retrieval_proj > reference raw_h
#   Q5  best retrieval_proj mix_mem_conf_nll < reference mix_mem_conf_nll
#   Q6  adaptive coverage improves without >10% token budget increase
#
# Key improvement threshold (Q1-Q3): mem_nll ↓0.05 OR acc@1 ↑0.01 OR acc@4 ↑0.01
#
# Config: wikitext-103-raw-v1, seq_len=256, batch=32,
#         max_memory_positions=500000, max_eval_positions=247000,
#         knn_k=64, knn_temp=0.2, normalize_keys=true
#
# Outputs (per run):
#   runs/<checkpoint_dir>/knn_raw_h_500k/
#   runs/<checkpoint_dir>/knn_retrieval_proj_500k/
# Aggregate:
#   runs/region_knn_keysource_comparison.md
#   runs/region_knn_keysource_comparison.csv

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

REGION_MAP=runs/region_maps_128/token_to_region.json
REF_CKPT=runs/repr_region_reference_30k/checkpoint_latest.pt
P05_CKPT=runs/repr_region_retrieval_proxy_lam0p05/checkpoint_latest.pt
P10_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
SC_CKPT=runs/repr_region_retrieval_supcon_lam0p05/checkpoint_latest.pt

echo "=============================="
echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
echo ""

# ── Verify checkpoints exist ───────────────────────────────────────────────────
for ckpt in "$REF_CKPT" "$P05_CKPT" "$P10_CKPT" "$SC_CKPT"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "ERROR: checkpoint missing: $ckpt" >&2
        echo "       Run slurm_region_retrieval_lm.sh first." >&2
        exit 1
    fi
done
if [[ ! -f "$REGION_MAP" ]]; then
    echo "ERROR: region map missing: $REGION_MAP" >&2; exit 1
fi
echo "All checkpoints present."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PREFLIGHT
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "PREFLIGHT"
echo "=============================="
python - <<'PYEOF'
import sys, torch

sys.path.insert(0, ".")
from train_region_lm import ReprRegionRetrievalLM
from scripts.offline_region_knn import (
    load_small_backbone_and_probe, MemoryIndex, TokenChunkDataset
)
import torch, torch.nn.functional as F

# Verify retrieval_proj is extracted for a retrieval checkpoint
ckpt_path = "runs/repr_region_retrieval_proxy_lam0p05/checkpoint_latest.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone, probe, d_model, _, vocab_size = \
    load_small_backbone_and_probe(ckpt_path, device)

assert hasattr(backbone, "retrieval_proj"), \
    "backbone missing retrieval_proj — check checkpoint mode"
key_dim = backbone.retrieval_proj[-1].weight.shape[0]
print(f"retrieval_proj OK  d_model={d_model}  key_dim={key_dim}")

# Smoke-test key extraction
dummy_src = torch.randint(0, vocab_size, (2, 16), device=device)
from scripts.offline_region_knn import get_hs_small
h = get_hs_small(backbone, dummy_src, device)
with torch.no_grad():
    z = backbone.retrieval_proj(h.reshape(-1, d_model))
    z = F.normalize(z.float(), dim=-1)
print(f"retrieval_proj key shape: {z.shape}  (expected: [{2*16}, {key_dim}])")

# Verify raw_h still works
print(f"raw_h shape: {h.shape}  (expected: [2, 16, {d_model}])")
print("PREFLIGHT PASSED")
PYEOF

PREFLIGHT_EXIT=$?
if [[ "$PREFLIGHT_EXIT" -ne 0 ]]; then
    echo "PREFLIGHT FAILED (exit $PREFLIGHT_EXIT) — aborting job." >&2
    exit "$PREFLIGHT_EXIT"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# SHARED KNN ARGS
# ─────────────────────────────────────────────────────────────────────────────
COMMON_KNN=(
    --model_type            small
    --dataset               wikitext-103-raw-v1
    --region_map_path       "$REGION_MAP"
    --max_memory_positions  500000
    --max_eval_positions    247000
    --knn_k                 64
    --knn_temp              0.2
    --probe_temp            1.0
    --normalize_keys        true
    --batch_size            32
    --seq_len               256
    --device                cuda
    --seed                  42
)

run_knn() {
    local label="$1"
    local ckpt="$2"
    local key_source="$3"
    local out_dir="$4"

    echo "------------------------------"
    echo "[$label]  key_source=$key_source"
    echo "  ckpt:    $ckpt"
    echo "  out_dir: $out_dir"

    if [[ -f "$out_dir/summary.md" ]] && [[ "${FORCE_RUN:-0}" != "1" ]]; then
        echo "  summary.md exists — skipping (set FORCE_RUN=1 to re-run)"
        return
    fi

    echo "  Start: $(date)"
    python scripts/offline_region_knn.py "${COMMON_KNN[@]}" \
        --small_ckpt      "$ckpt" \
        --knn_key_source  "$key_source" \
        --output_dir      "$out_dir"
    echo "  End:   $(date)"

    # Print metrics summary
    if [[ -f "$out_dir/metrics_all.csv" ]]; then
        echo "  --- metrics_all.csv ---"
        cat "$out_dir/metrics_all.csv"
    fi
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# RUN 1 — reference raw_h
# ─────────────────────────────────────────────────────────────────────────────
run_knn "reference raw_h" \
    "$REF_CKPT" \
    "raw_h" \
    "runs/repr_region_reference_30k/knn_raw_h_500k"

# ─────────────────────────────────────────────────────────────────────────────
# RUN 2 — proxy λ=0.05 raw_h
# ─────────────────────────────────────────────────────────────────────────────
run_knn "proxy λ=0.05 raw_h" \
    "$P05_CKPT" \
    "raw_h" \
    "runs/repr_region_retrieval_proxy_lam0p05/knn_raw_h_500k"

# ─────────────────────────────────────────────────────────────────────────────
# RUN 3 — proxy λ=0.05 retrieval_proj
# ─────────────────────────────────────────────────────────────────────────────
run_knn "proxy λ=0.05 retrieval_proj" \
    "$P05_CKPT" \
    "retrieval_proj" \
    "runs/repr_region_retrieval_proxy_lam0p05/knn_retrieval_proj_500k"

# ─────────────────────────────────────────────────────────────────────────────
# RUN 4 — proxy λ=0.10 raw_h
# ─────────────────────────────────────────────────────────────────────────────
run_knn "proxy λ=0.10 raw_h" \
    "$P10_CKPT" \
    "raw_h" \
    "runs/repr_region_retrieval_proxy_lam0p10/knn_raw_h_500k"

# ─────────────────────────────────────────────────────────────────────────────
# RUN 5 — proxy λ=0.10 retrieval_proj
# ─────────────────────────────────────────────────────────────────────────────
run_knn "proxy λ=0.10 retrieval_proj" \
    "$P10_CKPT" \
    "retrieval_proj" \
    "runs/repr_region_retrieval_proxy_lam0p10/knn_retrieval_proj_500k"

# ─────────────────────────────────────────────────────────────────────────────
# RUN 6 — supcon λ=0.05 raw_h
# ─────────────────────────────────────────────────────────────────────────────
run_knn "supcon λ=0.05 raw_h" \
    "$SC_CKPT" \
    "raw_h" \
    "runs/repr_region_retrieval_supcon_lam0p05/knn_raw_h_500k"

# ─────────────────────────────────────────────────────────────────────────────
# RUN 7 — supcon λ=0.05 retrieval_proj
# ─────────────────────────────────────────────────────────────────────────────
run_knn "supcon λ=0.05 retrieval_proj" \
    "$SC_CKPT" \
    "retrieval_proj" \
    "runs/repr_region_retrieval_supcon_lam0p05/knn_retrieval_proj_500k"

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
echo "=============================="
echo "AGGREGATE COMPARISON"
echo "=============================="

python - <<'PYEOF'
import csv, json, os, math
from datetime import datetime

RUNS = [
    # (label, checkpoint_dir, key_source, knn_output_dir)
    ("reference",               "runs/repr_region_reference_30k",
     "raw_h",         "runs/repr_region_reference_30k/knn_raw_h_500k"),
    ("proxy_lam0p05_raw_h",     "runs/repr_region_retrieval_proxy_lam0p05",
     "raw_h",         "runs/repr_region_retrieval_proxy_lam0p05/knn_raw_h_500k"),
    ("proxy_lam0p05_retr_proj", "runs/repr_region_retrieval_proxy_lam0p05",
     "retrieval_proj","runs/repr_region_retrieval_proxy_lam0p05/knn_retrieval_proj_500k"),
    ("proxy_lam0p10_raw_h",     "runs/repr_region_retrieval_proxy_lam0p10",
     "raw_h",         "runs/repr_region_retrieval_proxy_lam0p10/knn_raw_h_500k"),
    ("proxy_lam0p10_retr_proj", "runs/repr_region_retrieval_proxy_lam0p10",
     "retrieval_proj","runs/repr_region_retrieval_proxy_lam0p10/knn_retrieval_proj_500k"),
    ("supcon_lam0p05_raw_h",    "runs/repr_region_retrieval_supcon_lam0p05",
     "raw_h",         "runs/repr_region_retrieval_supcon_lam0p05/knn_raw_h_500k"),
    ("supcon_lam0p05_retr_proj","runs/repr_region_retrieval_supcon_lam0p05",
     "retrieval_proj","runs/repr_region_retrieval_supcon_lam0p05/knn_retrieval_proj_500k"),
]

nan = float("nan")

def _read_summary(d):
    p = os.path.join(d, "final_summary.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))

def _read_metrics(knn_dir):
    p = os.path.join(knn_dir, "metrics_all.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        m = r.get("method", "")
        v = lambda k: float(r.get(k, "nan") or "nan")
        if m == "router":
            out["router_acc1"] = v("acc1")
            out["router_acc4"] = v("acc4")
            out["router_nll"]  = v("region_nll")
        elif m == "mem_weighted":
            out["mem_acc1"] = v("acc1")
            out["mem_acc4"] = v("acc4")
            out["mem_nll"]  = v("region_nll")
        elif m == "mix_mem_conf":
            out["mix_mem_conf_nll"] = v("region_nll")
        elif m == "mix_0.25":
            out["mix_0.25_nll"] = v("region_nll")
    return out

def _read_coverage(knn_dir):
    p = os.path.join(knn_dir, "candidate_coverage.csv")
    if not os.path.exists(p):
        return {}
    for r in csv.DictReader(open(p)):
        if r.get("policy") == "adaptive":
            return {
                "adaptive_coverage": float(r.get("gold_region_coverage", "nan") or "nan"),
                "avg_regions":       float(r.get("avg_regions_kept", "nan") or "nan"),
                "avg_tokens":        float(r.get("avg_tokens_kept", "nan") or "nan"),
                "vocab_percent":     float(r.get("pct_vocab_kept", "nan") or "nan"),
            }
    return {}

def _read_knn_config(knn_dir):
    p = os.path.join(knn_dir, "knn_config.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))

# Collect all data
data = []
for label, ckpt_dir, key_src, knn_dir in RUNS:
    s  = _read_summary(ckpt_dir)
    m  = _read_metrics(knn_dir)
    c  = _read_coverage(knn_dir)
    kc = _read_knn_config(knn_dir)
    data.append({
        "label":       label,
        "ckpt_dir":    ckpt_dir,
        "key_source":  key_src,
        "knn_dir":     knn_dir,
        "val_ppl":     s.get("val_ppl",         nan),
        "val_lm":      s.get("val_lm_loss",      nan),
        "coarse_acc1": s.get("val_coarse_acc1",  nan),
        "key_dim":     kc.get("key_dim",         "?"),
        "n_mem":       kc.get("n_mem_actual",    0),
        "n_eval":      kc.get("n_eval_actual",   0),
        **m, **c,
    })

# ── Markdown table ────────────────────────────────────────────────────────────
def fmt(v, fmt=":.3f"):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    try:
        return format(v, fmt.strip(":"))
    except Exception:
        return str(v)

md_lines = [
    "# Region-kNN Key-Source Sweep",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "## Results",
    "",
    "| Run | key_source | key_dim | val_ppl | router acc@1 | router NLL | "
    "mem acc@1 | mem acc@4 | mem NLL | mix_conf NLL | adaptive_cov | avg_tokens |",
    "|-----|------------|---------|---------|--------------|------------|"
    "----------|----------|---------|-------------|-------------|------------|",
]
for d in data:
    md_lines.append(
        f"| {d['label']:<28} | {d['key_source']:<14} | {d['key_dim']:<7} "
        f"| {fmt(d['val_ppl'], ':.2f'):<7} "
        f"| {fmt(d.get('router_acc1', nan)):<12} "
        f"| {fmt(d.get('router_nll', nan)):<10} "
        f"| {fmt(d.get('mem_acc1', nan)):<8} "
        f"| {fmt(d.get('mem_acc4', nan)):<8} "
        f"| {fmt(d.get('mem_nll', nan)):<7} "
        f"| {fmt(d.get('mix_mem_conf_nll', nan)):<11} "
        f"| {fmt(d.get('adaptive_coverage', nan)):<11} "
        f"| {fmt(d.get('avg_tokens', nan), ':.0f'):<10} |"
    )

# ── Hypothesis verdicts ───────────────────────────────────────────────────────
def _get(label, key):
    for d in data:
        if d["label"] == label:
            return d.get(key, nan)
    return nan

def _beats(retr_label, raw_label):
    """PASS if retrieval_proj run beats raw_h on any of: mem_nll↓0.05, acc@1↑0.01, acc@4↑0.01"""
    nll_r  = _get(retr_label, "mem_nll")
    nll_raw= _get(raw_label,  "mem_nll")
    a1_r   = _get(retr_label, "mem_acc1")
    a1_raw = _get(raw_label,  "mem_acc1")
    a4_r   = _get(retr_label, "mem_acc4")
    a4_raw = _get(raw_label,  "mem_acc4")
    nll_ok = (not math.isnan(nll_r) and not math.isnan(nll_raw)
              and nll_r < nll_raw - 0.05)
    a1_ok  = (not math.isnan(a1_r) and not math.isnan(a1_raw)
              and a1_r > a1_raw + 0.01)
    a4_ok  = (not math.isnan(a4_r) and not math.isnan(a4_raw)
              and a4_r > a4_raw + 0.01)
    return "PASS" if (nll_ok or a1_ok or a4_ok) else "FAIL"

ref_mem_nll       = _get("reference",               "mem_nll")
ref_mix_conf_nll  = _get("reference",               "mix_mem_conf_nll")
ref_avg_tokens    = _get("reference",               "avg_tokens")

best_retr_mem_nll = min(
    (_get(l, "mem_nll") for l in ("proxy_lam0p05_retr_proj",
                                   "proxy_lam0p10_retr_proj",
                                   "supcon_lam0p05_retr_proj")
     if not math.isnan(_get(l, "mem_nll"))),
    default=nan,
)
best_retr_mix_nll = min(
    (_get(l, "mix_mem_conf_nll") for l in ("proxy_lam0p05_retr_proj",
                                            "proxy_lam0p10_retr_proj",
                                            "supcon_lam0p05_retr_proj")
     if not math.isnan(_get(l, "mix_mem_conf_nll"))),
    default=nan,
)

def _q4():
    if math.isnan(best_retr_mem_nll) or math.isnan(ref_mem_nll):
        return "INCONCLUSIVE"
    return "PASS" if best_retr_mem_nll < ref_mem_nll else "FAIL"

def _q5():
    if math.isnan(best_retr_mix_nll) or math.isnan(ref_mix_conf_nll):
        return "INCONCLUSIVE"
    return "PASS" if best_retr_mix_nll < ref_mix_conf_nll else "FAIL"

def _q6():
    """Coverage improves without >10% token budget increase."""
    ref_cov  = _get("reference",               "adaptive_coverage")
    ref_tok  = _get("reference",               "avg_tokens")
    ok = False
    for label in ("proxy_lam0p05_retr_proj",
                  "proxy_lam0p10_retr_proj",
                  "supcon_lam0p05_retr_proj"):
        cov = _get(label, "adaptive_coverage")
        tok = _get(label, "avg_tokens")
        if math.isnan(cov) or math.isnan(tok) or math.isnan(ref_cov) or math.isnan(ref_tok):
            continue
        if cov > ref_cov and tok <= ref_tok * 1.10:
            ok = True
    return "PASS" if ok else "FAIL"

q1 = _beats("proxy_lam0p05_retr_proj",  "proxy_lam0p05_raw_h")
q2 = _beats("proxy_lam0p10_retr_proj",  "proxy_lam0p10_raw_h")
q3 = _beats("supcon_lam0p05_retr_proj", "supcon_lam0p05_raw_h")
q4 = _q4()
q5 = _q5()
q6 = _q6()

verdicts = [(f"Q{i+1}", q) for i, q in enumerate([q1,q2,q3,q4,q5,q6])]
n_pass = sum(1 for _, v in verdicts if v == "PASS")

q_descs = [
    "proxy λ=0.05 retrieval_proj kNN beats proxy λ=0.05 raw_h (↓nll 0.05 OR ↑acc@1 0.01 OR ↑acc@4 0.01)",
    "proxy λ=0.10 retrieval_proj kNN beats proxy λ=0.10 raw_h (same thresholds)",
    "supcon λ=0.05 retrieval_proj kNN beats supcon λ=0.05 raw_h (same thresholds)",
    "best retrieval_proj mem_nll < reference raw_h mem_nll",
    "best retrieval_proj mix_conf_nll < reference raw_h mix_conf_nll",
    "retrieval_proj adaptive coverage improves with ≤10% token budget increase",
]

md_lines += [
    "",
    "## Hypothesis Tests",
    "",
    "| Q | Description | Verdict |",
    "|---|-------------|---------|",
]
for (qn, v), desc in zip(verdicts, q_descs):
    md_lines.append(f"| {qn} | {desc} | **{v}** |")

md_lines += [
    "",
    f"**{n_pass}/6 PASS**",
    "",
    "## Interpretation",
    "",
]

if n_pass >= 4:
    md_lines.append(
        "Strong evidence: the retrieval loss trained embeddings that are meaningfully "
        "better for region-kNN.  **Use z = retrieval_proj(h) as the kNN key in "
        "production**, not raw h."
    )
elif n_pass >= 2:
    md_lines.append(
        "Mixed evidence: some retrieval_proj runs improve kNN, but not consistently "
        "across all λ/loss combinations.  "
        "Consider tuning λ, retrieval_dim, or retrieval_temp before committing to the "
        "retrieval_proj key."
    )
else:
    md_lines.append(
        "The retrieval projection did not produce reliably better region neighbors than "
        "raw hidden states.  Possible causes: λ too small, retrieval_dim mismatch, "
        "proxy loss not a good proxy for kNN geometry.  "
        "Try SupCon with larger λ or a direct kNN-geometry loss (e.g., triplet)."
    )

# ── Write outputs ─────────────────────────────────────────────────────────────
os.makedirs("runs", exist_ok=True)

md_path = "runs/region_knn_keysource_comparison.md"
with open(md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Written: {md_path}")

csv_path = "runs/region_knn_keysource_comparison.csv"
csv_fields = [
    "label", "key_source", "key_dim", "val_ppl", "val_lm", "coarse_acc1",
    "router_acc1", "router_acc4", "router_nll",
    "mem_acc1", "mem_acc4", "mem_nll",
    "mix_0.25_nll", "mix_mem_conf_nll",
    "adaptive_coverage", "avg_regions", "avg_tokens", "vocab_percent",
    "n_mem", "n_eval",
]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore", restval="")
    w.writeheader()
    w.writerows(data)
print(f"Written: {csv_path}")

# Print table to stdout
print("\n" + "\n".join(md_lines))
PYEOF

echo ""
echo "=============================="
echo "End: $(date)"
echo "=============================="
echo "Job $SLURM_JOB_ID COMPLETE."
