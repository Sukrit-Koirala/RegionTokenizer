#!/usr/bin/env python3
"""
layerwise_subcluster_probe.py

For tokens whose gold next-token falls in a coarse cluster (default: cluster 7),
runs layerwise linear probes to answer:

  A. At which layer does *subcluster* identity become linearly decodable?
     (11 subclusters from recursive_cluster7.py)

  B. At which layer does *exact token* identity become linearly decodable
     within that cluster?  (~1896 classes)

The gap between A and B is the key finding: does the transformer resolve
"which sub-region?" before "which specific token?", or simultaneously?

Prerequisites
-------------
1. Run layerwise_region_probe.py --cache_dir ... to populate the probe cache.
2. Run recursive_cluster7.py --output_dir ... to generate cluster7_subcluster_mapping.json.

Usage
-----
python interference_experiment/layerwise_subcluster_probe.py \\
    --cache_dir     interference_experiment/results/probe_cache \\
    --subcluster_map interference_experiment/results/cluster7_subcluster_mapping.json \\
    --output_dir    interference_experiment/results/cluster7_subcluster_probe \\
    --seed 42
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("subprobe")


log = logging.getLogger("subprobe")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# PyTorch linear probe (same pattern as layerwise_region_probe.py)
# ---------------------------------------------------------------------------

class _TorchLinearProbe:
    def __init__(self, model: nn.Linear, device: torch.device) -> None:
        self.model = model.eval()
        self.device = device

    def predict_proba(self, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
        import torch.nn.functional as F
        self.model.eval()
        parts = []
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32))
            for i in range(0, len(t), chunk):
                logits = self.model(t[i : i + chunk].to(self.device)).cpu()
                parts.append(F.softmax(logits, dim=-1).numpy())
        return np.concatenate(parts, axis=0)


def fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    device: torch.device,
    seed: int = 42,
    epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 2048,
    weight_decay: float = 1e-4,
) -> Tuple[StandardScaler, _TorchLinearProbe]:
    set_seed(seed)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train.astype(np.float32))
    N = len(X_s)
    linear = nn.Linear(X_s.shape[1], n_classes).to(device)
    X_t = torch.from_numpy(X_s).to(device)
    y_t = torch.from_numpy(y_train.astype(np.int64)).to(device)
    opt = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss_fn(linear(X_t[idx]), y_t[idx]).backward()
            opt.step()
    return scaler, _TorchLinearProbe(linear, device)


def topk_recall(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([labels[i] in top_k[i] for i in range(len(labels))]))


def compute_metrics(
    probs: np.ndarray,
    y: np.ndarray,
    ks: List[int],
    n_classes: int,
) -> dict:
    preds = np.argmax(probs, axis=1)
    out: dict = {
        "top1": float((preds == y).mean()),
        "n": int(len(y)),
        "n_classes": n_classes,
    }
    for k in ks:
        out[f"top{k}"] = topk_recall(probs, y, min(k, n_classes))
    return out


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_cache(
    cache_dir: str,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, List[int]]:
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        log.error("Cache meta not found: %s", meta_path)
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    layers: List[int] = meta["layers"]
    hiddens: Dict[int, np.ndarray] = {}
    for l in layers:
        p = os.path.join(cache_dir, f"hidden_L{l:02d}.npy")
        if not os.path.exists(p):
            log.error("Missing cache file: %s", p)
            sys.exit(1)
        hiddens[l] = np.load(p)
    clusters = np.load(os.path.join(cache_dir, "clusters.npy"))
    tokens   = np.load(os.path.join(cache_dir, "tokens.npy"))
    log.info("Cache: N=%d  layers=%s", len(clusters), layers)
    return hiddens, clusters, tokens, layers


# ---------------------------------------------------------------------------
# Subcluster mapping
# ---------------------------------------------------------------------------

def load_subcluster_map(
    map_path: str,
    vocab_size: int = 50257,
) -> Tuple[np.ndarray, int, int, List[str]]:
    """
    Load cluster{C}_subcluster_mapping.json produced by recursive_cluster7.py.

    Returns
    -------
    vocab_to_sub  : int32 array (vocab_size,), -1 for non-members
    n_subclusters : int
    target_cluster: int
    token_strings : list[str], parallel to member_vocab_ids in the JSON
    """
    if not os.path.exists(map_path):
        log.error("Subcluster mapping not found: %s", map_path)
        log.error("Run recursive_cluster7.py first to generate this file.")
        sys.exit(1)

    with open(map_path, encoding="utf-8") as f:
        data = json.load(f)

    target_cluster = int(data["target_cluster"])
    n_subclusters  = int(data["n_subclusters"])
    token_strings  = data.get("token_strings", [])

    vocab_to_sub = np.full(vocab_size, -1, dtype=np.int32)
    for vid_str, sub_id in data["vocab_to_subcluster"].items():
        vid = int(vid_str)
        if 0 <= vid < vocab_size:
            vocab_to_sub[vid] = int(sub_id)

    n_members = int((vocab_to_sub >= 0).sum())
    log.info(
        "Subcluster map: cluster=%d  n_subclusters=%d  n_members=%d",
        target_cluster, n_subclusters, n_members,
    )
    return vocab_to_sub, n_subclusters, target_cluster, token_strings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layerwise subcluster + token probe for a single coarse cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache_dir",      required=True,
                        help="probe_cache dir from layerwise_region_probe.py")
    parser.add_argument("--subcluster_map", required=True,
                        help="cluster{N}_subcluster_mapping.json from recursive_cluster7.py")
    parser.add_argument("--output_dir",     default="results/cluster7_subcluster_probe",
                        help="Where to write metrics.json and plots")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--train_frac",     type=float, default=0.7,
                        help="Fraction of cluster-7 samples used for training")
    parser.add_argument("--epochs",         type=int, default=200)
    parser.add_argument("--lr",             type=float, default=1e-2)
    parser.add_argument("--batch_size",     type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap training set per probe for speed")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Probe device: %s", probe_device)

    # ------------------------------------------------------------------
    # 1. Load cache
    # ------------------------------------------------------------------
    log.info("=== Loading probe cache ===")
    hiddens, clusters, tokens, layers = load_cache(args.cache_dir)
    N_total = len(clusters)

    # ------------------------------------------------------------------
    # 2. Load subcluster mapping
    # ------------------------------------------------------------------
    log.info("=== Loading subcluster mapping ===")
    vocab_to_sub, n_subclusters, target_cluster, _ = load_subcluster_map(
        args.subcluster_map, vocab_size=50257
    )

    # ------------------------------------------------------------------
    # 3. Filter to target cluster
    # ------------------------------------------------------------------
    cluster_mask = (clusters == target_cluster)
    N_c = int(cluster_mask.sum())
    log.info(
        "Cluster %d: %d / %d samples  (%.1f%% of cache)",
        target_cluster, N_c, N_total, 100 * N_c / N_total,
    )
    if N_c < 200:
        log.error("Too few cluster-%d samples (%d). Check cache and cluster ID.", target_cluster, N_c)
        sys.exit(1)

    tok_c = tokens[cluster_mask].astype(np.int32)   # (N_c,) gold token IDs

    # ------------------------------------------------------------------
    # 4. Map tokens → subcluster labels
    # ------------------------------------------------------------------
    tok_clipped = tok_c.clip(0, len(vocab_to_sub) - 1)
    sub_labels_full = vocab_to_sub[tok_clipped]   # (N_c,), -1 if unmapped

    valid = sub_labels_full >= 0
    coverage = float(valid.mean())
    log.info(
        "Subcluster label coverage: %d / %d  (%.1f%%)  [%d unmapped]",
        valid.sum(), N_c, 100 * coverage, (~valid).sum(),
    )
    if valid.sum() < 100:
        log.error("Too few valid subcluster labels. Check subcluster_map path.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Build compact [0..K-1] token vocab for within-cluster token probe
    # ------------------------------------------------------------------
    tok_valid = tok_c[valid]
    seen_vocab_ids = np.unique(tok_valid)
    K = len(seen_vocab_ids)
    v2c = {int(vid): i for i, vid in enumerate(seen_vocab_ids)}

    compact_labels_full = np.full(N_c, -1, dtype=np.int32)
    for vid, cid in v2c.items():
        compact_labels_full[tok_c == vid] = cid

    # Keep positions where both labels are valid
    both = valid & (compact_labels_full >= 0)
    N_f = int(both.sum())
    log.info(
        "Final sample count after dual filter: %d  (K=%d unique tokens, %d subclusters)",
        N_f, K, n_subclusters,
    )
    if N_f < 100:
        log.error("Insufficient samples after filtering (%d).", N_f)
        sys.exit(1)

    y_sub = sub_labels_full[both].astype(np.int64)       # (N_f,)
    y_tok = compact_labels_full[both].astype(np.int64)   # (N_f,)

    # ------------------------------------------------------------------
    # 6. Train / test split — fixed so both probes see identical splits
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N_f)
    n_train = int(args.train_frac * N_f)
    tr_idx = perm[:n_train]
    te_idx = perm[n_train:]

    y_sub_te = y_sub[te_idx]
    y_tok_te = y_tok[te_idx]

    log.info("Split: train=%d  test=%d", n_train, len(te_idx))

    # Optional cap on training samples
    if args.max_train_samples and n_train > args.max_train_samples:
        cap_idx = rng.choice(n_train, args.max_train_samples, replace=False)
        tr_idx_eff  = tr_idx[cap_idx]
        log.info("Training capped at %d samples", args.max_train_samples)
    else:
        tr_idx_eff = tr_idx

    y_sub_tr = y_sub[tr_idx_eff]
    y_tok_tr = y_tok[tr_idx_eff]

    # Random chance baselines
    rand_sub = {f"top{k}": min(k / n_subclusters, 1.0) for k in [1, 2, 4]}
    rand_tok = {f"top{k}": min(k / K, 1.0) for k in [1, 5, 10, 20, 50]}
    log.info(
        "Random baselines — sub: top1=%.3f  top4=%.3f | tok: top1=%.4f  top10=%.3f",
        rand_sub["top1"], rand_sub["top4"], rand_tok["top1"], rand_tok["top10"],
    )

    # ------------------------------------------------------------------
    # 7. Per-layer probes
    # ------------------------------------------------------------------
    log.info("=== Training probes (PyTorch GPU linear, device=%s) ===", probe_device)
    results: Dict[int, dict] = {}

    for l in layers:
        t0 = time.time()
        log.info("--- Layer %d ---", l)

        # Extract hidden states for cluster-7 valid positions
        h_layer = hiddens[l][cluster_mask][both]   # (N_f, D)
        X_tr = h_layer[tr_idx_eff]
        X_te = h_layer[te_idx]

        # 7a. Subcluster probe  (n_subclusters classes)
        t1 = time.time()
        sc_sub, probe_sub = fit_linear_probe(
            X_tr, y_sub_tr, n_subclusters, probe_device,
            seed=args.seed, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )
        X_te_sub = sc_sub.transform(X_te.astype(np.float32))
        probs_sub = probe_sub.predict_proba(X_te_sub)
        m_sub = compute_metrics(probs_sub, y_sub_te, ks=[2, 4], n_classes=n_subclusters)
        log.info(
            "  Subcluster probe: top1=%.3f  top4=%.3f  (%.1fs)",
            m_sub["top1"], m_sub["top4"], time.time() - t1,
        )

        # 7b. Token probe  (K classes — up to ~1896)
        t2 = time.time()
        sc_tok, probe_tok = fit_linear_probe(
            X_tr, y_tok_tr, K, probe_device,
            seed=args.seed, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )
        X_te_tok = sc_tok.transform(X_te.astype(np.float32))
        probs_tok = probe_tok.predict_proba(X_te_tok)
        m_tok = compute_metrics(probs_tok, y_tok_te, ks=[5, 10, 20, 50], n_classes=K)
        log.info(
            "  Token probe:      top1=%.4f  top10=%.3f  (%.1fs)",
            m_tok["top1"], m_tok["top10"], time.time() - t2,
        )

        results[l] = {
            "subcluster": m_sub,
            "token":      m_tok,
            "elapsed_s":  round(time.time() - t0, 1),
        }

    # ------------------------------------------------------------------
    # 8. Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "config": vars(args),
        "n_samples_cluster": N_f,
        "n_train": n_train,
        "n_test": len(te_idx),
        "n_subclusters": n_subclusters,
        "K_tokens": K,
        "random_baselines": {"subcluster": rand_sub, "token": rand_tok},
        "layers": {str(l): results[l] for l in layers},
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    _plot_results(results, layers, n_subclusters, K, rand_sub, rand_tok, args.output_dir)

    # ------------------------------------------------------------------
    # 10. Summary table
    # ------------------------------------------------------------------
    _print_summary(results, layers, n_subclusters, K, rand_sub, rand_tok)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(
    results: dict,
    layers: List[int],
    n_subclusters: int,
    K: int,
    rand_sub: dict,
    rand_tok: dict,
    output_dir: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return

    xlabels = [str(l) for l in layers]

    # ------------------------------------------------------------------
    # Plot 1: Subcluster probe
    # ------------------------------------------------------------------
    top1_sub = [results[l]["subcluster"]["top1"] for l in layers]
    top4_sub = [results[l]["subcluster"]["top4"] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xlabels, top1_sub, "o-", color="tab:blue",   lw=2, label="top-1")
    ax.plot(xlabels, top4_sub, "s--", color="tab:orange", lw=2, label="top-4")
    ax.axhline(rand_sub["top1"], ls=":", color="tab:blue",   alpha=0.4,
               label=f"random top-1 ({rand_sub['top1']:.2f})")
    ax.axhline(rand_sub["top4"], ls=":", color="tab:orange", alpha=0.4,
               label=f"random top-4 ({rand_sub['top4']:.2f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recall")
    ax.set_title(f"Subcluster probe — {n_subclusters} subclusters, cluster-7 tokens")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "layer_vs_subcluster.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Token probe
    # ------------------------------------------------------------------
    top1_tok  = [results[l]["token"]["top1"]  for l in layers]
    top10_tok = [results[l]["token"]["top10"] for l in layers]
    top50_tok = [results[l]["token"]["top50"] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xlabels, top1_tok,  "o-",  color="tab:blue",   lw=2, label="top-1")
    ax.plot(xlabels, top10_tok, "s--", color="tab:orange",  lw=2, label="top-10")
    ax.plot(xlabels, top50_tok, "^:",  color="tab:green",   lw=2, label="top-50")
    ax.axhline(rand_tok["top10"], ls=":", color="tab:orange", alpha=0.4,
               label=f"random top-10 ({rand_tok['top10']:.4f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recall")
    ax.set_title(f"Exact token probe — {K} classes, cluster-7 tokens")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "layer_vs_token.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: Comparison — subcluster top-1 vs token top-1 on one axis.
    # Both curves are normalized to their own random baseline so they're
    # directly comparable despite very different class counts.
    #
    #   normalized gain = (top1 - random_top1) / (1 - random_top1)
    # ------------------------------------------------------------------
    def norm_gain(vals: List[float], rand: float) -> List[float]:
        denom = max(1.0 - rand, 1e-6)
        return [(v - rand) / denom for v in vals]

    ng_sub = norm_gain(top1_sub, rand_sub["top1"])
    ng_tok = norm_gain(top1_tok, rand_tok["top1"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw top-1
    axes[0].plot(xlabels, top1_sub, "o-", color="tab:blue", lw=2,
                 label=f"sub top-1 ({n_subclusters} cls)")
    axes[0].plot(xlabels, top1_tok, "s--", color="tab:red", lw=2,
                 label=f"tok top-1 ({K} cls)")
    axes[0].axhline(rand_sub["top1"], ls=":", color="tab:blue",  alpha=0.4)
    axes[0].axhline(rand_tok["top1"], ls=":", color="tab:red",   alpha=0.4)
    axes[0].set_title("Raw top-1 recall")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Recall")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Right: normalized gain (same scale)
    axes[1].plot(xlabels, ng_sub, "o-", color="tab:blue", lw=2,
                 label=f"subcluster ({n_subclusters} cls)")
    axes[1].plot(xlabels, ng_tok, "s--", color="tab:red", lw=2,
                 label=f"token ({K} cls)")
    axes[1].axhline(0.0, ls="-", color="gray", lw=0.8)
    axes[1].set_title("Normalized gain  =  (top1 − random) / (1 − random)")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Normalized gain")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Does subcluster emerge before exact token?  (cluster-7 tokens)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    plt.close(fig)

    log.info("Plots saved to %s/", output_dir)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(
    results: dict,
    layers: List[int],
    n_subclusters: int,
    K: int,
    rand_sub: dict,
    rand_tok: dict,
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  CLUSTER-7 SUBCLUSTER PROBE — SUMMARY")
    print(sep)
    print(f"  n_subclusters = {n_subclusters}    K_tokens = {K}")
    print(f"  Random sub top-1={rand_sub['top1']:.3f}  top-4={rand_sub['top4']:.3f}")
    print(f"  Random tok top-1={rand_tok['top1']:.4f}  top-10={rand_tok['top10']:.4f}")
    print()
    print(f"  {'Layer':>6}  {'Sub top-1':>10}  {'Sub top-4':>10}  "
          f"{'Tok top-1':>10}  {'Tok top-10':>11}  {'Tok top-50':>11}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*11}")
    for l in layers:
        s = results[l]["subcluster"]
        t = results[l]["token"]
        print(
            f"  {l:>6}  {s['top1']:10.3f}  {s['top4']:10.3f}  "
            f"{t['top1']:10.4f}  {t['top10']:11.3f}  {t['top50']:11.3f}"
        )
    print()

    # Find first layer that clears 3× random for each probe
    thresh_sub = 3.0 * rand_sub["top1"]
    thresh_tok = 3.0 * rand_tok["top1"]
    first_sub = next((l for l in layers if results[l]["subcluster"]["top1"] > thresh_sub), None)
    first_tok = next((l for l in layers if results[l]["token"]["top1"]      > thresh_tok), None)

    # First layer where subcluster top-1 > 50 %
    first_sub_50 = next((l for l in layers if results[l]["subcluster"]["top1"] > 0.50), None)

    print(f"  First layer with subcluster top-1 > {thresh_sub:.2f} (3× random): {first_sub}")
    print(f"  First layer with subcluster top-1 > 0.50:                         {first_sub_50}")
    print(f"  First layer with token top-1      > {thresh_tok:.4f} (3× random): {first_tok}")
    print()

    if first_sub is not None and first_tok is not None:
        delta = layers.index(first_tok) - layers.index(first_sub)
        if delta > 0:
            print(f"  FINDING: Subcluster identity emerges {delta} layer-slot(s) EARLIER "
                  f"than exact token.")
            print(f"           (sub 3×rand @ layer {first_sub}  vs  tok 3×rand @ layer {first_tok})")
        elif delta == 0:
            print("  FINDING: Subcluster and exact token emerge at the SAME layer slot.")
        else:
            print(f"  FINDING: Exact token emerges {-delta} layer-slot(s) EARLIER "
                  f"than subcluster (unexpected).")
    elif first_sub is None:
        print("  FINDING: Subcluster never reaches 3× random in the sampled layers.")
    elif first_tok is None:
        print("  FINDING: Token never reaches 3× random — subcluster is much easier.")
    print(sep)


if __name__ == "__main__":
    main()
