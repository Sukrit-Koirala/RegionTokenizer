# Periodic Eval Bug Report

## Problem

The official force-zero / baseline path was correct and consistent across variants:

```
Variant C force_zero:     covered_nll=3.378606  cov=0.948425  fingerprint=f57cabcdc46d69ce  PASS
D3-small  force_zero:     covered_nll=3.378606  cov=0.948425  fingerprint=f57cabcdc46d69ce  PASS
D3-base   force_zero:     covered_nll=3.378606  cov=0.948425  fingerprint=f57cabcdc46d69ce  PASS
```

However, the **periodic training eval** used a different code path (`eval_dataset`) that
reported variant-dependent NLL **and coverage**, making the runs invalid:

```
Variant C   step=1000:  covered_nll=3.2405  cov=0.9487
D3-small    step=1000:  covered_nll=3.1558  cov=0.9509
D3-base     step=1000:  covered_nll=3.1377  cov=0.9559
```

Coverage **must not** vary by architecture — the validation set is fixed.
NLL differences are expected (models learn at different rates), but coverage differences
mean the variants were evaluated on **different subsets** of the data.

## Root Cause

`eval_dataset` was called without `eval_batch_size`:

```python
# BUGGY (line 824-826 before fix):
metrics = eval_dataset(model, args.val_dir, r2s_np,
                       args.batch_size, device,        # ← varies: 64/32/16 by variant
                       max_batches=args.eval_max_batches)  # ← 500 batches
```

With `max_batches=500` and different `batch_size`:

| Variant  | batch_size | examples evaluated | coverage     |
|----------|------------|-------------------|--------------|
| C        | 64         | 500 × 64 = 32,000  | 0.9487       |
| D3-small | 32         | 500 × 32 = 16,000  | 0.9509       |
| D3-base  | 16         | 500 × 16 =  8,000  | 0.9559       |

The shard coverage distribution is non-uniform: earlier shards (evaluated by smaller
batches that stop sooner) happen to have higher coverage than the full dataset.

D3-base's 8,000 examples come from the first few shards, which have 95.6% coverage.
C's 32,000 examples average out closer to the true 94.84% coverage.

## Why the Old Results Are Invalid

**D3-small `best_model_nll=3.1553`** appears better than **C `best_model_nll=3.2391`**
but this comparison is meaningless: D3-small was evaluated on 16k easier examples,
C on 32k harder examples. The models cannot be compared.

Similarly D3-base `best_model_nll≈3.1377` from 8k examples cannot be compared to
either of the above.

## Invalid Checkpoints (DO NOT USE)

```
runs/path_refiner_clean/variant_C/best_refiner.pt       INVALID
runs/path_refiner_clean/d3_small/best_refiner.pt        INVALID
runs/path_refiner_clean/d3_base/best_refiner.pt         INVALID
```

These checkpoints were saved using coverage-biased periodic eval.
Their `best_covered_nll` values are not comparable to each other or to the baseline.
**Do not delete them** (they may be useful for debugging), but do not use them for
conclusions about architecture quality.

## Fix

Replaced `eval_dataset` (limited to `max_batches × batch_size` examples) with
`canonical_eval_refiner` — the single unified evaluator used for everything:

```
force_zero baseline check
step-0 with_delta identity check
periodic training eval every N steps
final eval
best checkpoint selection
```

`canonical_eval_refiner` enforces:
- No `max_batches` cap — evaluates ALL examples (239,362 for hgrid_K24)
- Fixed `eval_batch_size=64` regardless of training batch_size
- Asserts `fingerprint == f57cabcdc46d69ce` if official_baseline provided
- Asserts `num_examples == 239,362` and `num_covered == 227,017`
- Asserts `|coverage - 0.948425| < 1e-6`
- Raises `RuntimeError` on any mismatch when `--fail_on_baseline_mismatch`

Additionally, training now runs TWO full-val evals at step 0:
- `force_zero` (no model) → must match official baseline exactly
- `with_delta` (model at init) → must equal force_zero (identity assertion)

Both must print the same fingerprint/coverage/counts. If they differ, training aborts.

## Verification

```bash
sbatch scripts/slurm_debug_periodic_eval_consistency.sh
```

Expected for all variants:
```
[step 0 / force_zero]  covered_nll=3.378606  cov=0.948425  fingerprint=f57cabcdc46d69ce
[step 0 / with_delta]  covered_nll=3.378606  cov=0.948425  fingerprint=f57cabcdc46d69ce
[step 0] identity PASS: nll_diff=X.XXe-07  fingerprint match=True  coverage match=True

[eval/canonical_full_val] step=1
  eval_mode    = canonical_full_val
  covered_nll  = ...  (may differ by variant — model learned one step)
  coverage     = 0.948425          ← MUST BE IDENTICAL ACROSS ALL VARIANTS
  fingerprint  = f57cabcdc46d69ce  ← MUST BE IDENTICAL ACROSS ALL VARIANTS
  num_examples = 239,362           ← MUST BE IDENTICAL ACROSS ALL VARIANTS
  num_covered  = 227,017           ← MUST BE IDENTICAL ACROSS ALL VARIANTS
```

## Next Steps

After all variants pass the debug script:
1. `sbatch scripts/slurm_clean_train_refiners.sh`
2. Compare `best_metrics.json` across variants — all share same `official_baseline_nll`
3. A variant is better only if its `covered_nll` is lower, not its step-1000 subset NLL
