# Baseline Consistency Report

## Problem

The force-zero NLL differed across refiner variants on the same val dataset:

| Variant  | force_zero covered_nll | best covered_nll | gain   |
|----------|------------------------|-----------------|--------|
| C        | 3.2405                 | 3.2387          | 0.0018 |
| D3-small | 3.1558                 | 3.1553          | 0.0005 |
| D3-base  | 3.1377                 | ?               | ?      |

Force-zero should be architecture-independent. If it differs, the comparison
between variants is invalid.

## Root Cause

`force_zero_delta` was implemented by patching `model.log_scale.data.fill_(-100.0)`
and then calling the full model forward. This still ran the complete architecture
(including the D3 region transformer). The patched log_scale approach is correct
in principle, but the coverage divergence revealed a separate bug:

**`eval_dataset` used `max_batches=500` counted per-batch, not per-example.**
Each variant had a different training `batch_size`:
- Variant C:  64 × 500 =  32,000 examples evaluated
- D3-small:   32 × 500 =  16,000 examples evaluated
- D3-base:    16 × 500 =   8,000 examples evaluated

The val dataset is not perfectly uniform: coverage and NLL vary across shards.
Evaluating on different numbers of examples from different positions in the
streaming order gave genuinely different numbers — not a model artifact.

**Coverage differences:**
```
C        cov = 0.9487  (32k examples)
D3-small cov = 0.9509  (16k examples)
D3-base  cov = 0.9559  ( 8k examples)
```

The D3 variants happened to be evaluated on a higher-coverage portion of the
dataset (earlier shards), making their numbers look better.

## Why Old D3 Results Are Invalid

D3-small appeared to have a lower force-zero NLL (3.1558 vs 3.2405 for C).
This is **not a real improvement** — it's an artifact of evaluating on a
different, easier subset of the data. Any training gain on top of an already
biased baseline cannot be trusted.

**D3-small and D3-base results from this run must be discarded.**

## Fix

Two changes were made:

### 1. Architecture-independent `eval_force_zero()`

Replaced `eval_dataset(..., force_zero_delta=True)` with a standalone function
that:
- Loads shards **directly** (no ShardStreamDataset, no r2s, no model)
- Evaluates **all examples** (no `max_batches` limit)
- Uses a fixed `eval_batch_size=64` regardless of training batch_size
- Computes: `scores = h_prime @ tok_emb[cand_tok]`, masked, then cross-entropy

### 2. Official saved-candidate baseline

`scripts/eval_saved_candidate_baseline.py` computes the reference numbers from
the same formula. All variants must reproduce these numbers (within 1e-4 NLL,
1e-6 coverage) before training is considered valid.

### 3. Dataset fingerprint

A deterministic fingerprint is computed from:
```
num_shards, total_n, total_cov, sum_cand_counts, sum_gold_idx_cov, sum_gold_tok
```

The fingerprint must match across all eval calls. Any mismatch means something
is filtering or reordering examples incorrectly.

## Verification Procedure

Run `slurm_debug_refiner_baseline_consistency.sh` before any training:

```bash
sbatch scripts/slurm_debug_refiner_baseline_consistency.sh
```

Expected output (after fix):
```
Official saved-candidate baseline:
  covered_nll = X.XXXXXX
  coverage    = 0.XXXXXX
  fingerprint = <16-char hex>

Variant C:     covered_nll = X.XXXXXX  coverage = 0.XXXXXX  fingerprint = <same>  PASS
D3-small:      covered_nll = X.XXXXXX  coverage = 0.XXXXXX  fingerprint = <same>  PASS
D3-base:       covered_nll = X.XXXXXX  coverage = 0.XXXXXX  fingerprint = <same>  PASS
```

## Next Steps

After all checks pass, rerun training with `--eval_before_train` and
`--official_baseline`. The correct comparison table format is:

| Variant  | official_baseline_nll | best_model_nll | delta  | coverage |
|----------|-----------------------|---------------|--------|----------|
| C        | X.XXXX                | X.XXXX        | X.XXXX | X.XXXX   |
| D3-small | X.XXXX                | X.XXXX        | X.XXXX | X.XXXX   |
| D3-base  | X.XXXX (if ran)       | X.XXXX        | X.XXXX | X.XXXX   |

All variants share the same `official_baseline_nll` column. A D3 variant is
better than C only if its `best_model_nll` is lower, not its force-zero NLL.
