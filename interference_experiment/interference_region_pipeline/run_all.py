#!/usr/bin/env python3
"""
run_all.py

End-to-end interference region pipeline:

  Step 1: build_interference_graph  — run LM, build coactivation graph
  Step 2: cluster_regions           — Leiden / Louvain community detection
  Step 3: recursive_subcluster      — subcluster large regions
  Step 4: evaluate_regions          — probes, logit locality, gold recall, PPL

Usage:
  python run_all.py [--skip_steps 1 2] [all other args]

--skip_steps  space-separated list of step numbers to skip (useful for re-runs)
"""

import argparse
import os
import sys
import time

# Make imports work when called from any cwd
sys.path.insert(0, os.path.dirname(__file__))

import build_interference_graph as _build
import cluster_regions          as _cluster
import recursive_subcluster     as _subcluster
import evaluate_regions         as _eval
from utils import setup_logging, set_seed

log = setup_logging("run_all")


def run_step(name: str, fn, args, skip: set) -> None:
    step_n = {"build": 1, "cluster": 2, "subcluster": 3, "evaluate": 4}[name]
    if step_n in skip:
        log.info("Skipping step %d (%s)", step_n, name)
        return
    log.info("=" * 60)
    log.info("STEP %d: %s", step_n, name.upper())
    log.info("=" * 60)
    t0 = time.time()
    fn(args)
    log.info("Step %d done: %.1fs", step_n, time.time() - t0)


def build_step_args(args) -> tuple:
    """Create per-step args namespaces from the global args."""

    class NS:
        pass

    build = NS()
    build.model_name        = args.model_name
    build.dataset_name      = args.dataset_name
    build.dataset_config    = args.dataset_config
    build.max_tokens        = args.max_tokens
    build.vocab_subset_size = args.vocab_subset_size
    build.top_k             = args.top_k
    build.output_dir        = args.output_dir
    build.graphs_dir        = args.graphs_dir
    build.local_text_file   = args.local_text_file
    build.seq_len           = args.seq_len
    build.probe_layers      = args.probe_layers
    build.n_probe           = args.n_probe
    build.alpha             = args.alpha
    build.beta              = args.beta
    build.gamma             = args.gamma
    build.seed              = args.seed

    cluster = NS()
    cluster.graphs_dir  = args.graphs_dir
    cluster.output_dir  = args.output_dir
    cluster.model_name  = args.model_name
    cluster.resolutions = args.resolutions
    cluster.seed        = args.seed

    subcluster = NS()
    subcluster.graphs_dir           = args.graphs_dir
    subcluster.output_dir           = args.output_dir
    subcluster.model_name           = args.model_name
    subcluster.recursive_depth      = args.recursive_depth
    subcluster.recursive_min_size   = args.recursive_min_size
    subcluster.min_modularity_gain  = args.min_modularity_gain
    subcluster.seed                 = args.seed

    evaluate = NS()
    evaluate.output_dir      = args.output_dir
    evaluate.graphs_dir      = args.graphs_dir
    evaluate.model_name      = args.model_name
    evaluate.probe_epochs    = args.probe_epochs
    evaluate.logit_ks        = args.logit_ks
    evaluate.r_values        = args.r_values
    evaluate.n_routed_eval   = args.n_routed_eval
    evaluate.seed            = args.seed

    return build, cluster, subcluster, evaluate


def main(args) -> None:
    set_seed(args.seed)
    skip = set(args.skip_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.graphs_dir, exist_ok=True)

    build_args, cluster_args, subcluster_args, eval_args = build_step_args(args)

    t_start = time.time()
    run_step("build",      _build.main,      build_args,      skip)
    run_step("cluster",    _cluster.main,    cluster_args,    skip)
    run_step("subcluster", _subcluster.main, subcluster_args, skip)
    run_step("evaluate",   _eval.main,       eval_args,       skip)
    log.info("Pipeline complete: %.1fs total", time.time() - t_start)


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end interference region pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Pipeline control
    p.add_argument("--skip_steps",         type=int, nargs="*", default=[],
                   help="Step numbers to skip: 1=build 2=cluster 3=subcluster 4=evaluate")

    # Shared paths
    p.add_argument("--output_dir",         default="interference_region_pipeline/results")
    p.add_argument("--graphs_dir",         default="interference_region_pipeline/graphs")
    p.add_argument("--model_name",         default="gpt2-xl")
    p.add_argument("--seed",               type=int, default=42)

    # Step 1: build graph
    p.add_argument("--dataset_name",       default="wikitext")
    p.add_argument("--dataset_config",     default="wikitext-2-raw-v1")
    p.add_argument("--max_tokens",         type=int, default=500_000)
    p.add_argument("--vocab_subset_size",  type=int, default=5_000)
    p.add_argument("--top_k",             type=int, default=50)
    p.add_argument("--local_text_file",    default=None)
    p.add_argument("--seq_len",            type=int, default=512)
    p.add_argument("--probe_layers",       type=int, nargs="+",
                   default=[0, 4, 8, 12, 16, 20, 24, 47])
    p.add_argument("--n_probe",            type=int, default=50_000)
    p.add_argument("--alpha",              type=float, default=1.0,
                   help="Coactivation graph weight")
    p.add_argument("--beta",               type=float, default=0.0,
                   help="Gradient similarity weight (not implemented)")
    p.add_argument("--gamma",              type=float, default=0.0,
                   help="Embedding similarity weight")

    # Step 2: clustering
    p.add_argument("--resolutions",        type=float, nargs="+",
                   default=[0.5, 1.0, 1.5, 2.0])

    # Step 3: recursive subcluster
    p.add_argument("--recursive_depth",    type=int,   default=2)
    p.add_argument("--recursive_min_size", type=int,   default=200)
    p.add_argument("--min_modularity_gain",type=float, default=0.05)

    # Step 4: evaluation
    p.add_argument("--probe_epochs",       type=int,   default=200)
    p.add_argument("--logit_ks",           type=int,   nargs="+", default=[10, 50, 100, 500])
    p.add_argument("--r_values",           type=int,   nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n_routed_eval",      type=int,   default=5_000)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
