#!/bin/bash
#SBATCH --job-name=reg_mem_arch
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Region-Memory Architecture Evaluation Pipeline.
#
# Stages (run sequentially; each stage may gate later stages):
#   1 — Controller eval (Policies 0–4, offline, no GPU needed)
#   2 — Hard-position group analysis (offline, no GPU needed)
#   3 — Masked-softmax token-level eval (GPU, forward-pass evaluation)
#   4 — Final architecture report
#
# Prerequisites:
#   - offline_region_knn.py must have been run with the best config and
#     must have produced per_position.npz in KNN_RUN_DIR.
#   - memory_keys.npy and memory_regions.npy must exist in KNN_RUN_DIR.
#
# Outputs:
#   runs/region_memory_controller/        — controller_results.csv, pareto_frontier.csv,
#                                           best_controller.json, controller_plots/
#   runs/region_memory_hard_positions/    — hard_position_stats.csv, test_results.txt,
#                                           hard_position_plots/
#   runs/region_memory_masked_softmax/    — masked_softmax_results.json, pass_fail.txt
#   runs/region_memory_architecture_report.md  — final 9-question report
#
# Tunable knobs at the top of this file:
#   KNN_RUN_DIR — path to best kNN run (must contain per_position.npz)
#   SMALL_CKPT  — small backbone checkpoint for masked-softmax eval
#   REGION_MAP  — token_to_region.json for the 128-coarse-region map
#   MAX_EVAL    — max positions for masked-softmax eval (default 50k)

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

KNN_RUN_DIR=runs/region_knn_extensive_sweep/proxy010_retrproj_mem500k_k64_t0p20
SMALL_CKPT=runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt
REGION_MAP=runs/region_maps_128/token_to_region.json
DATASET=wikitext-103-raw-v1
MAX_EVAL=50000
BATCH_SIZE=4

CTRL_OUT=runs/region_memory_controller
HARD_OUT=runs/region_memory_hard_positions
SOFT_OUT=runs/region_memory_masked_softmax
REPORT=runs/region_memory_architecture_report.md

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "=== Region-Memory Architecture Eval  $(date) ==="
echo "    KNN_RUN_DIR : $KNN_RUN_DIR"
echo "    SMALL_CKPT  : $SMALL_CKPT"
echo "    REGION_MAP  : $REGION_MAP"

for f in "$REGION_MAP" "$SMALL_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2; exit 1
    fi
done

if [[ ! -d "$KNN_RUN_DIR" ]]; then
    echo "ERROR: KNN_RUN_DIR not found: $KNN_RUN_DIR" >&2; exit 1
fi

if [[ ! -f "$KNN_RUN_DIR/per_position.npz" ]]; then
    echo "ERROR: per_position.npz not found in $KNN_RUN_DIR" >&2
    echo "       Re-run offline_region_knn.py to generate it." >&2
    exit 1
fi

for artifact in memory_keys.npy memory_regions.npy; do
    if [[ ! -f "$KNN_RUN_DIR/$artifact" ]]; then
        echo "ERROR: $artifact not found in $KNN_RUN_DIR" >&2
        echo "       Required by masked-softmax eval for memory index rebuild." >&2
        exit 1
    fi
done

mkdir -p "$CTRL_OUT" "$HARD_OUT" "$SOFT_OUT" "$(dirname "$REPORT")"

# ── Stage 1: Controller Eval ──────────────────────────────────────────────────

echo ""
echo "=== Stage 1: Controller Evaluation (Policies 0–4)  $(date) ==="

CTRL_DONE="$CTRL_OUT/best_controller.json"
if [[ -f "$CTRL_DONE" ]]; then
    echo "    [skip] best_controller.json exists; delete to rerun."
else
    python scripts/eval_region_memory_controller.py \
        --knn_run_dir   "$KNN_RUN_DIR"   \
        --region_map_path "$REGION_MAP"  \
        --output_dir    "$CTRL_OUT"      \
        --train_frac    0.7              \
        --seed          42

    if [[ ! -f "$CTRL_DONE" ]]; then
        echo "ERROR: Stage 1 did not produce best_controller.json" >&2; exit 1
    fi
    echo "    [done] best_controller.json written."
fi

# ── Stage 2: Hard-Position Group Analysis ─────────────────────────────────────

echo ""
echo "=== Stage 2: Hard-Position Group Analysis  $(date) ==="

HARD_DONE="$HARD_OUT/hard_position_stats.csv"
if [[ -f "$HARD_DONE" ]]; then
    echo "    [skip] hard_position_stats.csv exists; delete to rerun."
else
    CTRL_ARG=""
    if [[ -f "$CTRL_DONE" ]]; then
        CTRL_ARG="--best_controller $CTRL_DONE"
    fi

    python scripts/eval_memory_hard_positions.py \
        --knn_run_dir  "$KNN_RUN_DIR" \
        --output_dir   "$HARD_OUT"    \
        $CTRL_ARG

    if [[ ! -f "$HARD_DONE" ]]; then
        echo "ERROR: Stage 2 did not produce hard_position_stats.csv" >&2; exit 1
    fi
    echo "    [done] hard_position_stats.csv written."
fi

# ── Stage 3: Masked-Softmax Token-Level Eval ──────────────────────────────────

echo ""
echo "=== Stage 3: Masked-Softmax Evaluation  $(date) ==="

SOFT_DONE="$SOFT_OUT/masked_softmax_results.json"
if [[ -f "$SOFT_DONE" ]]; then
    echo "    [skip] masked_softmax_results.json exists; delete to rerun."
else
    CTRL_ARG=""
    if [[ -f "$CTRL_DONE" ]]; then
        CTRL_ARG="--controller_cfg $CTRL_DONE"
    fi

    python scripts/eval_region_memory_masked_softmax.py \
        --small_ckpt         "$SMALL_CKPT"   \
        --knn_run_dir        "$KNN_RUN_DIR"  \
        --region_map_path    "$REGION_MAP"   \
        --dataset            "$DATASET"      \
        --max_eval_positions "$MAX_EVAL"     \
        --batch_size         "$BATCH_SIZE"   \
        --output_dir         "$SOFT_OUT"     \
        --device             cuda            \
        --seed               42              \
        $CTRL_ARG

    if [[ ! -f "$SOFT_DONE" ]]; then
        echo "ERROR: Stage 3 did not produce masked_softmax_results.json" >&2; exit 1
    fi
    echo "    [done] masked_softmax_results.json written."
fi

# ── Stage 4: Final Architecture Report ────────────────────────────────────────

echo ""
echo "=== Stage 4: Writing Final Architecture Report  $(date) ==="

# Export shell vars so the Python report generator can read them
export CTRL_OUT HARD_OUT SOFT_OUT REPORT KNN_RUN_DIR

python - <<'PYEOF'
import json, csv, math, os, sys, datetime

CTRL_OUT = os.environ.get("CTRL_OUT", "runs/region_memory_controller")
HARD_OUT = os.environ.get("HARD_OUT", "runs/region_memory_hard_positions")
SOFT_OUT = os.environ.get("SOFT_OUT", "runs/region_memory_masked_softmax")
REPORT   = os.environ.get("REPORT",   "runs/region_memory_architecture_report.md")
KNN_DIR  = os.environ.get("KNN_RUN_DIR", "")


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _flt(d, k, default=float("nan")):
    try:
        return float(d[k])
    except Exception:
        return default


# Load artefacts
best_ctrl   = _load_json(os.path.join(CTRL_OUT, "best_controller.json"))
ctrl_rows   = _load_csv(os.path.join(CTRL_OUT, "controller_results.csv"))
pareto_rows = _load_csv(os.path.join(CTRL_OUT, "pareto_frontier.csv"))
hard_rows   = _load_csv(os.path.join(HARD_OUT, "hard_position_stats.csv"))
soft_res    = _load_json(os.path.join(SOFT_OUT, "masked_softmax_results.json"))

# Derive key numbers from hard-position stats
def _group(name):
    for r in hard_rows:
        if r.get("group") == name:
            return r
    return {}

all_g    = _group("all")
typeA    = _group("type_A")
typeB    = _group("type_B")
typeC    = _group("type_C")
bnd      = _group("boundary")
hi_ent   = _group("high_entropy")
r4miss   = _group("router_top4_miss")
r8miss   = _group("router_top8_miss")
mem_conf = _group("mem_confident")

router_nll = _flt(all_g, "router_nll")
mem_nll    = _flt(all_g, "mem_nll")
mix_nll    = _flt(all_g, "mix_nll")
mix_beats_router = (not math.isnan(mix_nll)) and (not math.isnan(router_nll)) and (mix_nll < router_nll)

typeA_mix_nll    = _flt(typeA, "mix_nll")
typeA_router_nll = _flt(typeA, "router_nll")
typeA_imp = typeA_router_nll - typeA_mix_nll  # positive = better

typeB_mix_nll    = _flt(typeB, "mix_nll")
typeB_router_nll = _flt(typeB, "router_nll")
typeB_hurt = (not math.isnan(typeB_mix_nll)) and (typeB_mix_nll > typeB_router_nll + 0.01)

# Soft-eval numbers
tok_cov     = _flt(soft_res, "token_coverage",  float("nan"))
mapped_pct  = _flt(soft_res, "mapped_vocab_pct", float("nan"))
nll_delta   = _flt(soft_res, "strict_nll_delta", float("nan"))
verdict     = soft_res.get("verdict", "UNKNOWN")

# Best controller summary
ctrl_policy  = best_ctrl.get("policy", "?")
ctrl_cov     = _flt(best_ctrl, "coverage", float("nan"))
ctrl_regions = _flt(best_ctrl, "avg_regions", float("nan"))

# ── Write report ──────────────────────────────────────────────────────────────

lines = []
def h(s): lines.append(f"\n## {s}\n")
def p(s): lines.append(s)
def hr(): lines.append("\n---\n")

lines.append("# Region-Memory Architecture Evaluation Report\n")
lines.append(f"*Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*  \n")
lines.append(f"*kNN run: `{KNN_DIR}`*\n")

hr()
h("Q1. Should memory be used as a distribution mixture or as candidate expansion?")
if mix_beats_router:
    delta = router_nll - mix_nll
    p(f"Mix NLL ({mix_nll:.4f}) beats router NLL ({router_nll:.4f}) by {delta:.4f} bits — "
      "mixture provides marginal global lift.")
else:
    p(f"Mix NLL ({mix_nll:.4f}) does NOT beat router NLL ({router_nll:.4f}) globally. "
      "Mixture adds noise on average.")
if not math.isnan(tok_cov):
    p(f"\nCandidate expansion achieves {tok_cov*100:.1f}% token coverage with "
      f"{mapped_pct:.1f}% mapped vocab and NLL delta {nll_delta:+.4f} ({verdict}).")
p("\n**Recommendation**: use candidate expansion (gated, Type-A only) as primary mode; "
  "optionally add soft mixture only for Type-A positions where it helps.")

h("Q2. For which position types does memory help vs hurt?")
if not math.isnan(typeA_imp):
    p(f"- **Type A** (same-region ambiguity): mix NLL improvement = {typeA_imp:+.4f} "
      f"(router {typeA_router_nll:.4f} → mix {typeA_mix_nll:.4f})")
if not math.isnan(typeB_mix_nll):
    typeB_delta = typeB_mix_nll - typeB_router_nll
    p(f"- **Type B** (multi-region ambiguity): mix NLL change = {typeB_delta:+.4f} "
      f"({'hurts' if typeB_delta > 0 else 'helps'})")
if not math.isnan(_flt(typeC, "mix_nll")):
    typeC_delta = _flt(typeC, "mix_nll") - _flt(typeC, "router_nll")
    p(f"- **Type C** (misleading memory): mix NLL change = {typeC_delta:+.4f} "
      f"({'hurts' if typeC_delta > 0 else 'helps'})")
p(f"\n**Boundary positions**: router NLL = {_flt(bnd, 'router_nll'):.4f}, "
  f"mix NLL = {_flt(bnd, 'mix_nll'):.4f}")
p(f"**High-entropy positions**: router NLL = {_flt(hi_ent, 'router_nll'):.4f}, "
  f"mix NLL = {_flt(hi_ent, 'mix_nll'):.4f}")

h("Q3. What is the best controller policy?")
if best_ctrl:
    p(f"Best policy: **{ctrl_policy}**")
    p(f"  Coverage = {ctrl_cov:.3f}, avg regions kept = {ctrl_regions:.1f}")
    for k, v in best_ctrl.items():
        if k not in ("policy", "coverage", "avg_regions"):
            p(f"  {k} = {v}")
else:
    p("No best_controller.json found — controller eval may not have completed.")

h("Q4. What is the best coverage/cost operating point?")
if pareto_rows:
    p("Pareto-frontier (coverage vs avg_regions):\n")
    p("| Policy | Coverage | Avg Regions | Tokens (est) |")
    p("|--------|----------|-------------|--------------|")
    for r in pareto_rows[:10]:
        p(f"| {r.get('policy','?')} | {_flt(r,'coverage'):.3f} | "
          f"{_flt(r,'avg_regions'):.1f} | {r.get('avg_tokens_kept','?')} |")
else:
    p("Pareto frontier data not available.")

h("Q5. What fraction of router_top4_miss positions does memory recover?")
r4_r8m4 = _flt(r4miss, "union_r8m4_coverage")
r4_r16m4 = _flt(r4miss, "union_r16m4_coverage")
r4_router4 = _flt(r4miss, "router_top4_coverage")
p(f"In the router_top4_miss group ({r4miss.get('count','?')} positions):")
p(f"  router top-4 coverage: {r4_router4:.3f} (should be ~0)")
p(f"  union(r8, m4) coverage: {r4_r8m4:.3f}")
p(f"  union(r16, m4) coverage: {r4_r16m4:.3f}")
p(f"  mix NLL: {_flt(r4miss, 'mix_nll'):.4f} vs router NLL: {_flt(r4miss, 'router_nll'):.4f}")

h("Q6. Does the masked-softmax eval preserve LM loss?")
if math.isnan(tok_cov):
    p("Masked-softmax eval results not available.")
else:
    p(f"- Token coverage: **{tok_cov*100:.1f}%**")
    p(f"- Mapped vocab:   **{mapped_pct:.1f}%** of total tokens have a region mapping")
    p(f"- NLL delta (strict): **{nll_delta:+.4f}** bits")
    p(f"- Verdict: **{verdict}**")
    # Reproduce pass thresholds for clarity
    if verdict == "STRONG PASS":
        p("\nStrong pass criteria met: tok_cov ≥ 0.90, mapped_pct ≤ 8.0, nll_delta < 0.05.")
    elif verdict == "MODERATE PASS":
        p("\nModerate pass criteria met: tok_cov ≥ 0.95, mapped_pct ≤ 15.0, nll_delta < 0.10.")
    else:
        p("\nNeither pass threshold met — do not proceed to online refiner.")

h("Q7. How much compute is saved by candidate restriction?")
full_vocab = 50257
if not math.isnan(mapped_pct) and not math.isnan(tok_cov):
    avg_toks = soft_res.get("avg_candidate_tokens", float("nan"))
    if not math.isnan(avg_toks):
        savings = (1 - avg_toks / full_vocab) * 100
        p(f"Average candidate vocab size: {avg_toks:.0f} / {full_vocab} tokens "
          f"({savings:.1f}% reduction in output projection work at restricted positions).")
    p(f"Restricted positions: those with a region mapping ({mapped_pct:.1f}% of eval set).")
    p(f"For unrestricted positions, full-vocab softmax runs unchanged.")
else:
    p("Compute savings data not available from masked-softmax results.")

h("Q8. Does the Type-A gating correctly avoid hurting non-A positions?")
if typeB_hurt:
    p("WARNING: Type B positions show mix NLL hurt — gating on mem_margin alone is insufficient. "
      "Ensure controller only activates memory expansion for confirmed Type-A positions.")
else:
    p("Type B positions are not significantly hurt by the mixture — gating appears adequate.")
p(f"\nType C (misleading) NLL change: {_flt(typeC,'mix_nll') - _flt(typeC,'router_nll'):+.4f} "
  f"(should be ≤ 0 or small positive after gating).")

h("Q9. Should we proceed to Phase 2B (online local refiner)?")
if verdict in ("STRONG PASS", "MODERATE PASS"):
    p(f"**YES** — masked-softmax eval verdict is {verdict}. The candidate expansion preserves "
      "LM loss within acceptable bounds. Proceed to Phase 2B: online local refiner training.")
    p("\nRecommended next steps:")
    p("1. Implement `RegionMemoryLM` wrapping the small backbone with masked-softmax head.")
    p("2. Fine-tune on wikitext-103 with candidate-restricted cross-entropy for Type-A positions.")
    p("3. Evaluate full LM perplexity on test set vs baseline.")
else:
    p("**NO** — masked-softmax eval did not pass. Do not proceed to online refiner yet.")
    p("\nDiagnostic actions:")
    p("1. Check `mapped_vocab_pct` — if > 15%, the region map covers too many tokens; "
      "tighten the router confidence threshold.")
    p("2. Check `token_coverage` — if < 0.90, increase union(Kr, Km) in the controller.")
    p("3. Check Type-C positions — if memory is misleading often, raise mem_margin threshold.")

hr()
p("*Artefact paths:*\n")
p(f"- Controller:      `{CTRL_OUT}/`")
p(f"- Hard positions:  `{HARD_OUT}/`")
p(f"- Masked softmax:  `{SOFT_OUT}/`")

os.makedirs(os.path.dirname(os.path.abspath(REPORT)), exist_ok=True)
with open(REPORT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Report written: {REPORT}")
PYEOF


echo ""
echo "=== Pipeline Complete  $(date) ==="
echo ""
echo "Outputs:"
echo "  Controller:         $CTRL_OUT/"
echo "  Hard positions:     $HARD_OUT/"
echo "  Masked softmax:     $SOFT_OUT/"
echo "  Architecture report: $REPORT"
echo ""
if [[ -f "$REPORT" ]]; then
    grep -E "^## Q9|^\*\*YES\*\*|^\*\*NO\*\*|Verdict:" "$REPORT" | head -5
fi
