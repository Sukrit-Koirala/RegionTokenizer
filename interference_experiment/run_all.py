"""
Master orchestrator for the interference graph experiment.

Runs the full pipeline end-to-end:

  Step 1  extract_hidden_states  — forward-pass corpus, collect top-k & gradients
  Step 2  build_graph            — co-activation interference graph W
  Step 3  gradient_graph         — analytic gradient interference graph I
  Step 4  clustering             — Louvain / Spectral / K-Means on W and I
  Step 5  visualize              — heatmaps, spectra, cluster plots
  Step 6  layer_comparison       — repeat steps 1–4 for early/mid/final layers
  Step 7  summary                — print structured diagnostic report

Each step caches its outputs; re-running with --skip_* flags is supported.
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np

# Ensure the experiment directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    ExperimentConfig,
    load_json,
    save_json,
    set_seed,
    setup_dirs,
    setup_logging,
)
import extract_hidden_states as ext
import build_graph as bg
import gradient_graph as gg
import clustering as cl
import visualize as viz


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_step1_extraction(
    cfg: ExperimentConfig,
    layer_index: int,
    layer_tag: str,
    collect_hidden: bool = False,
    force: bool = False,
) -> str:
    """Extract top-k and gradient accumulators. Returns artefact prefix."""
    prefix = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}")
    meta_path = f"{prefix}_meta.json"

    if os.path.exists(meta_path) and not force:
        log = logging.getLogger("interference")
        log.info("[Step 1] Cached extraction found for layer_tag=%s — skipping.", layer_tag)
        return prefix

    log = logging.getLogger("interference")
    log.info("[Step 1] Extracting hidden states / top-k (layer_tag=%s, layer=%d)…",
             layer_tag, layer_index)
    t0 = time.time()
    data = ext.extract(cfg, layer_index=layer_index, collect_mean_hidden=collect_hidden)
    ext.save_extracted(data, cfg, layer_tag=layer_tag)
    log.info("[Step 1] Done in %.1f s", time.time() - t0)
    return prefix


def run_step2_coactivation(
    cfg: ExperimentConfig, prefix: str, layer_tag: str, force: bool = False
) -> np.ndarray:
    """Build co-activation graph W. Returns normalised matrix."""
    out = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}_W_norm.npy")
    log = logging.getLogger("interference")

    if os.path.exists(out) and not force:
        log.info("[Step 2] Cached co-activation graph found — skipping.")
        return np.load(out)

    log.info("[Step 2] Building co-activation interference graph …")
    t0 = time.time()
    W = bg.build_and_save(prefix, cfg)
    log.info("[Step 2] Done in %.1f s", time.time() - t0)
    return W


def run_step3_gradient(
    cfg: ExperimentConfig, prefix: str, layer_tag: str, force: bool = False
) -> np.ndarray:
    """Build gradient interference graph I. Returns cosine-similarity matrix."""
    out = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}_I_gradient.npy")
    log = logging.getLogger("interference")

    if os.path.exists(out) and not force:
        log.info("[Step 3] Cached gradient graph found — skipping.")
        return np.load(out)

    log.info("[Step 3] Building gradient interference graph …")
    t0 = time.time()
    I = gg.build_and_save(prefix, cfg)
    log.info("[Step 3] Done in %.1f s", time.time() - t0)
    return I


def run_step4_clustering(
    cfg: ExperimentConfig,
    W: np.ndarray,
    layer_tag: str,
    graph_type: str,
    force: bool = False,
) -> dict:
    """Run community detection. Returns results dict."""
    out = os.path.join(
        cfg.graphs_dir, f"layer_{layer_tag}_{graph_type}_clustering_results.json"
    )
    log = logging.getLogger("interference")

    if os.path.exists(out) and not force:
        log.info("[Step 4] Cached clustering found for %s / %s — skipping.",
                 layer_tag, graph_type)
        return load_json(out)

    log.info("[Step 4] Clustering %s graph (layer=%s) …", graph_type, layer_tag)
    t0 = time.time()
    results = cl.cluster_and_analyse(W, cfg, layer_tag=layer_tag, graph_type=graph_type)
    log.info("[Step 4] Done in %.1f s", time.time() - t0)
    return results


def run_step5_visualise(
    cfg: ExperimentConfig,
    W: np.ndarray,
    graph_type: str,
    layer_tag: str,
    clustering_results: dict,
    draw_graph: bool = True,
) -> None:
    log = logging.getLogger("interference")
    log.info("[Step 5] Generating visualisations for %s / %s …", graph_type, layer_tag)
    t0 = time.time()
    viz.visualise_graph(
        W=W,
        graph_type=graph_type,
        layer_tag=layer_tag,
        cfg=cfg,
        clustering_results=clustering_results,
        draw_cluster_graph=draw_graph,
    )
    log.info("[Step 5] Done in %.1f s", time.time() - t0)


# ---------------------------------------------------------------------------
# Token report — decode cluster labels to readable token strings
# ---------------------------------------------------------------------------

def save_cluster_token_report(
    cfg: ExperimentConfig,
    layer_tag: str = "final",
    graph_type: str = "coactivation",
    top_n_per_cluster: int = 20,
) -> None:
    """
    For each cluster, list the top N token strings ranked by prediction
    frequency.  Saves two files:

      results/cluster_tokens_{graph_type}_{layer_tag}.json
          Full mapping: {cluster_id: [{"token": str, "token_id": int, "freq": int}, ...]}

      results/cluster_tokens_{graph_type}_{layer_tag}.txt
          Human-readable text version for quick inspection.
    """
    from transformers import AutoTokenizer

    log = logging.getLogger("interference")
    prefix_graph = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}")
    prefix_clust = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}_{graph_type}")

    labels_path = f"{prefix_clust}_labels_louvain.npy"
    if not os.path.exists(labels_path):
        log.warning("No Louvain labels found at %s — skipping token report.", labels_path)
        return

    labels = np.load(labels_path)                          # (N_sub,)
    freq_ids = np.load(f"{prefix_graph}_frequent_token_ids.npy")   # (N_sub,)
    token_freqs = np.load(f"{prefix_graph}_token_freqs.npy")       # (V,)

    meta = load_json(f"{prefix_graph}_meta.json")
    model_name = meta.get("model_name", cfg.model_name)

    log.info("Loading tokenizer %s for cluster token report …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_clusters = int(labels.max()) + 1
    report: dict = {}

    for c in range(n_clusters):
        mask = labels == c
        ids_in_cluster = freq_ids[mask]
        freqs_in_cluster = token_freqs[ids_in_cluster]
        # Sort by prediction frequency (most frequent first)
        order = np.argsort(freqs_in_cluster)[::-1][:top_n_per_cluster]
        entries = []
        for i in order:
            tok_id = int(ids_in_cluster[i])
            tok_str = tokenizer.decode([tok_id])
            entries.append({
                "token": tok_str,
                "token_id": tok_id,
                "freq": int(freqs_in_cluster[i]),
            })
        report[str(c)] = entries

    # Save JSON
    json_path = os.path.join(
        cfg.results_dir, f"cluster_tokens_{graph_type}_{layer_tag}.json"
    )
    save_json(json_path, report)

    # Save human-readable text
    txt_path = os.path.join(
        cfg.results_dir, f"cluster_tokens_{graph_type}_{layer_tag}.txt"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Cluster token report — {graph_type} | layer={layer_tag}\n")
        f.write(f"Model: {model_name}  |  {n_clusters} clusters\n")
        f.write("=" * 70 + "\n\n")
        for c in range(n_clusters):
            tokens = [e["token"].replace("\n", "\\n").strip() for e in report[str(c)]]
            size = int((labels == c).sum())
            f.write(f"Cluster {c:3d}  (size={size:4d}):  {' | '.join(tokens[:15])}\n")

    log.info(
        "Cluster token report saved → %s  (and .txt)", json_path
    )


# ---------------------------------------------------------------------------
# Layer comparison
# ---------------------------------------------------------------------------

def run_layer_comparison(cfg: ExperimentConfig, force: bool = False) -> dict:
    """
    Run extraction + gradient graph + clustering for each layer in
    cfg.layer_compare_indices.  Returns dict {layer_tag: modularity}.
    """
    log = logging.getLogger("interference")
    log.info("=== Layer comparison experiment ===")

    n_layers_guess = 48   # GPT-2 XL; will be corrected from first extraction
    layer_results: dict = {}

    layer_tags = []
    for raw_idx in cfg.layer_compare_indices:
        tag = "final" if raw_idx == -1 else f"L{raw_idx:02d}"
        layer_tags.append((raw_idx, tag))

    for layer_index, layer_tag in layer_tags:
        log.info("--- Layer %s (index=%d) ---", layer_tag, layer_index)

        prefix = run_step1_extraction(
            cfg, layer_index=layer_index, layer_tag=layer_tag,
            collect_hidden=True, force=force,
        )
        I = run_step3_gradient(cfg, prefix, layer_tag, force=force)

        # Also build representational similarity from hidden states
        h_path = os.path.join(cfg.graphs_dir, f"layer_{layer_tag}_mean_hidden_per_token.npy")
        if os.path.exists(h_path):
            data = ext.load_extracted(prefix, need_hidden=True)
            freq_ids = data["frequent_token_ids"]
            H = data["mean_hidden_per_token"]
            S = gg.compute_representational_similarity(H, freq_ids)
            np.save(os.path.join(cfg.graphs_dir, f"layer_{layer_tag}_S_hidden.npy"), S)
            cl_res = run_step4_clustering(cfg, S, layer_tag, "hidden_sim", force=force)
            layer_results[layer_tag] = cl_res.get("louvain", {}).get("modularity", 0.0)

    if layer_results:
        viz.visualise_layer_comparison(
            [t for _, t in layer_tags],
            cfg,
            graph_type="hidden_sim",
        )

    return layer_results


# ---------------------------------------------------------------------------
# Final summary printer
# ---------------------------------------------------------------------------

def print_summary(
    coact_results: dict,
    grad_results: dict,
    layer_comparison: Optional[dict] = None,
) -> None:
    """Print structured summary answering the five hypothesis questions."""

    sep = "=" * 66

    def _fmt_q(val: float, q_rand: float) -> str:
        if val > q_rand * 2:
            return f"Q={val:.4f}  [STRONG — >2x random baseline {q_rand:.4f}]"
        elif val > q_rand:
            return f"Q={val:.4f}  [WEAK — >random baseline {q_rand:.4f}]"
        else:
            return f"Q={val:.4f}  [NOT SIGNIFICANT — ≤random {q_rand:.4f}]"

    print(f"\n{sep}")
    print("  INTERFERENCE GRAPH EXPERIMENT — SUMMARY REPORT")
    print(sep)

    # ---- Community structure? ----
    q_co = coact_results.get("louvain", {}).get("modularity", 0.0)
    q_gr = grad_results.get("louvain", {}).get("modularity", 0.0)
    q_rand = coact_results.get("random_baseline_modularity", 0.02)
    strong_community = q_co > q_rand * 2 or q_gr > q_rand * 2
    weak_community = q_co > q_rand or q_gr > q_rand
    q1 = "YES" if strong_community else ("WEAK" if weak_community else "NO")

    print(f"\n1. Does the interference graph show community structure?  {q1}")
    print(f"   Co-activation  {_fmt_q(q_co, q_rand)}")
    print(f"   Gradient       {_fmt_q(q_gr, q_rand)}")

    # ---- Cluster stability? ----
    nmi_co = coact_results.get("nmi", {}).get("louvain_spectral", 0.0)
    nmi_gr = grad_results.get("nmi", {}).get("louvain_spectral", 0.0)
    stable = nmi_co > 0.5 and nmi_gr > 0.5
    q2 = "YES" if stable else ("PARTLY" if (nmi_co > 0.3 or nmi_gr > 0.3) else "NO")
    print(f"\n2. Are clusters stable across methods?  {q2}")
    print(f"   NMI(Louvain, Spectral) — co-act={nmi_co:.3f}  gradient={nmi_gr:.3f}")
    print(f"   (NMI > 0.5 = stable, > 0.3 = partially stable)")

    # ---- Modularity vs random? ----
    beats = coact_results.get("louvain_beats_random", False) or \
            grad_results.get("louvain_beats_random", False)
    q3 = "YES" if beats else "NO"
    print(f"\n3. Does modularity exceed the random baseline?  {q3}")
    print(f"   Random baseline Q ≈ {q_rand:.4f}  (uniform partition)")

    # ---- Low-rank eigenspectrum? ----
    eig_co = coact_results.get("eigenvalues", {})
    eig_gr = grad_results.get("eigenvalues", {})
    pr_co = eig_co.get("participation_ratio", float("nan"))
    pr_gr = eig_gr.get("participation_ratio", float("nan"))
    d90_co = eig_co.get("dims_for_90pct", float("nan"))
    d90_gr = eig_gr.get("dims_for_90pct", float("nan"))
    vocab_sub = coact_results.get("louvain", {}).get("n_clusters", 50)
    n_tokens = coact_results.get("louvain", {}).get("n_clusters", 50)
    # Low-rank if PR << N
    low_rank = (not isinstance(pr_co, float) or pr_co < 200)
    q4 = "YES (low-rank)" if low_rank else "NO (full-rank)"
    print(f"\n4. Does the eigenspectrum suggest low-rank / modular structure?  {q4}")
    print(f"   Co-activation: PR={pr_co:.1f}  dims@90%={d90_co}")
    print(f"   Gradient:      PR={pr_gr:.1f}  dims@90%={d90_gr}")
    print(f"   (Participation ratio << N tokens → low-dimensional effective space)")

    # ---- Layer comparison ----
    if layer_comparison:
        tags = sorted(layer_comparison.keys())
        vals = [layer_comparison[t] for t in tags]
        increasing = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)) if len(vals) > 1 else True
        print(f"\n   Layer comparison modularity:")
        for t, v in zip(tags, vals):
            print(f"     layer={t}  Q={v:.4f}")
        trend = "INCREASES" if increasing else "MIXED"
        print(f"   Trend: modularity {trend} with depth.")

    # ---- Region-based decoding hypothesis ----
    intra_co = coact_results.get("louvain", {}).get("intra_cluster_weight", 0.0)
    inter_co = coact_results.get("louvain", {}).get("inter_cluster_weight", 0.0)
    ratio_co = coact_results.get("louvain", {}).get("intra_inter_ratio", 0.0)

    supported = strong_community and stable and beats
    partly = weak_community and (stable or beats)
    q5 = "SUPPORTED" if supported else ("PARTIALLY SUPPORTED" if partly else "NOT SUPPORTED")

    print(f"\n5. Does this support the region-based decoding hypothesis?  {q5}")
    print(f"   Tokens form {'distinct' if strong_community else 'weak'} interference communities.")
    print(f"   Intra-cluster weight: {intra_co:.4f}")
    print(f"   Inter-cluster weight: {inter_co:.4f}")
    print(f"   Intra/Inter ratio:    {ratio_co:.2f}x  (>2x = meaningful separation)")

    n_co = coact_results.get("louvain", {}).get("n_clusters", "?")
    n_gr = grad_results.get("louvain", {}).get("n_clusters", "?")
    print(f"\n   Louvain communities found: {n_co} (co-activation), {n_gr} (gradient)")

    print(f"\n{sep}\n")

    if supported:
        print("CONCLUSION: Evidence supports region-based decoding.")
        print("  Vocabulary tokens do form interference communities in representation")
        print("  space.  A region-aware decoder that groups tokens by community")
        print("  before scoring should reduce spurious interference.")
    elif partly:
        print("CONCLUSION: Partial evidence for region-based decoding.")
        print("  Some modular structure exists but is not strongly pronounced.")
        print("  Region-based decoding may help but may not dominate standard decoding.")
    else:
        print("CONCLUSION: Evidence does NOT support region-based decoding at this layer.")
        print("  Interference structure is indistinguishable from noise.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end interference graph experiment."
    )
    parser.add_argument("--model", default="gpt2-xl",
                        help="HuggingFace model ID.")
    parser.add_argument("--max_tokens", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--vocab_subset", type=int, default=5_000)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--n_clusters", type=int, default=50)

    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument("--skip_graphs", action="store_true")
    parser.add_argument("--skip_clustering", action="store_true")
    parser.add_argument("--skip_viz", action="store_true")
    parser.add_argument("--layer_compare", action="store_true",
                        help="Also run the layer-comparison experiment.")
    parser.add_argument("--no_cluster_graph", action="store_true",
                        help="Skip the force-directed cluster graph (slow for large N).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite cached artefacts.")
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("interference")
    log.info("=" * 60)
    log.info("Interference Graph Experiment")
    log.info("Model: %s   max_tokens=%d   vocab_subset=%d",
             args.model, args.max_tokens, args.vocab_subset)
    log.info("=" * 60)

    cfg = ExperimentConfig(
        model_name=args.model,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        vocab_subset_size=args.vocab_subset,
        top_k=args.top_k,
        n_clusters=args.n_clusters,
        fp16=not args.no_fp16,
    )
    setup_dirs(cfg)
    set_seed(cfg.seed)

    t_global = time.time()

    # ------------------------------------------------------------------ #
    # Step 1 — Extraction (final layer)                                   #
    # ------------------------------------------------------------------ #
    if not args.skip_extraction:
        prefix = run_step1_extraction(
            cfg, layer_index=-1, layer_tag="final", force=args.force
        )
    else:
        prefix = os.path.join(cfg.graphs_dir, "layer_final")

    # ------------------------------------------------------------------ #
    # Step 2 — Co-activation graph                                        #
    # ------------------------------------------------------------------ #
    if not args.skip_graphs:
        W = run_step2_coactivation(cfg, prefix, "final", force=args.force)
    else:
        W_path = os.path.join(cfg.graphs_dir, "layer_final_W_norm.npy")
        if os.path.exists(W_path):
            W = np.load(W_path)
        else:
            log.error("Co-activation graph not found at %s — run without --skip_graphs first.", W_path)
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 3 — Gradient interference graph                                #
    # ------------------------------------------------------------------ #
    if not args.skip_graphs:
        I = run_step3_gradient(cfg, prefix, "final", force=args.force)
    else:
        I_path = os.path.join(cfg.graphs_dir, "layer_final_I_gradient.npy")
        if os.path.exists(I_path):
            I = np.load(I_path)
        else:
            log.error("Gradient graph not found at %s — run without --skip_graphs first.", I_path)
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 4 — Clustering                                                 #
    # ------------------------------------------------------------------ #
    if not args.skip_clustering:
        coact_results = run_step4_clustering(cfg, W, "final", "coactivation", force=args.force)
        grad_results = run_step4_clustering(cfg, I, "final", "gradient", force=args.force)
    else:
        coact_path = os.path.join(cfg.graphs_dir, "layer_final_coactivation_clustering_results.json")
        grad_path = os.path.join(cfg.graphs_dir, "layer_final_gradient_clustering_results.json")
        coact_results = load_json(coact_path) if os.path.exists(coact_path) else {}
        grad_results = load_json(grad_path) if os.path.exists(grad_path) else {}

    # ------------------------------------------------------------------ #
    # Step 5 — Visualisation                                              #
    # ------------------------------------------------------------------ #
    if not args.skip_viz:
        run_step5_visualise(cfg, W, "coactivation", "final", coact_results,
                            draw_graph=not args.no_cluster_graph)
        run_step5_visualise(cfg, I, "gradient", "final", grad_results,
                            draw_graph=not args.no_cluster_graph)

    # ------------------------------------------------------------------ #
    # Step 6 — Layer comparison (optional)                                #
    # ------------------------------------------------------------------ #
    layer_comparison = None
    if args.layer_compare:
        layer_comparison = run_layer_comparison(cfg, force=args.force)

    # ------------------------------------------------------------------ #
    # Step 6.5 — Decode cluster labels → human-readable token report     #
    # ------------------------------------------------------------------ #
    if not args.skip_clustering:
        save_cluster_token_report(cfg, layer_tag="final")

    # ------------------------------------------------------------------ #
    # Step 7 — Final summary                                              #
    # ------------------------------------------------------------------ #
    print_summary(coact_results, grad_results, layer_comparison)

    elapsed = time.time() - t_global
    log.info("Total wall time: %.1f s (%.1f min)", elapsed, elapsed / 60)

    # Persist full config + summary metrics
    summary = {
        "config": cfg.to_dict(),
        "model": cfg.model_name,
        "max_tokens": cfg.max_tokens,
        "vocab_subset_size": cfg.vocab_subset_size,
        "coactivation_louvain_Q": coact_results.get("louvain", {}).get("modularity"),
        "gradient_louvain_Q": grad_results.get("louvain", {}).get("modularity"),
        "coactivation_nmi_lv_sp": coact_results.get("nmi", {}).get("louvain_spectral"),
        "gradient_nmi_lv_sp": grad_results.get("nmi", {}).get("louvain_spectral"),
        "coactivation_PR": coact_results.get("eigenvalues", {}).get("participation_ratio"),
        "gradient_PR": grad_results.get("eigenvalues", {}).get("participation_ratio"),
        "layer_comparison": layer_comparison,
        "elapsed_s": round(elapsed, 1),
    }
    save_json(os.path.join(cfg.results_dir, "experiment_summary.json"), summary)


if __name__ == "__main__":
    main()
