# Init Identity Report

## Problem

At step-0 with `eval_before_train`, the `with_delta` eval produced a different NLL
than `force_zero`:

```
[step 0 / force_zero] covered_nll=3.378606  cov=0.948425
[step 0 / with_delta] covered_nll=3.240547  cov=0.948719
difference = 0.138059 nats
```

A residual refiner must be an exact identity at initialization (delta=0 → scores=base).
The 0.138-nat gap meant either the residual was non-zero, or the two evals saw
different data — or both.

## Root Causes

### 1. Non-zero residual scale at init

`log_scale = nn.Parameter(torch.tensor(-4.0))` → `exp(-4.0) = 0.0183`.
The MLP/scorer output is multiplied by 0.0183, not 0.0.
So `scores = base + 0.018 * mlp_out ≠ base`.

**Fix:** Replace with `residual_scale = nn.Parameter(torch.tensor(0.0))`.
`scores = base + 0.0 * mlp_out = base` exactly.

### 2. Different val subsets for `force_zero` vs `with_delta`

`eval_force_zero` evaluates **all** ~239k val examples.
`eval_dataset(max_batches=500)` evaluates only `500 × batch_size` examples from
the start of the streaming order (first shards), which happens to have higher
coverage — giving a different (spuriously better) NLL.

Even with a mathematically zero delta this alone would produce a discrepancy.

**Fix:** Replace the step-0 `eval_dataset` call with `check_init_identity`, which
runs both evaluations on the **same** 64 examples and asserts equality within
`nll_diff < 1e-4` and `max|delta| < 1e-5`.
The CSV step-0 row now logs `m0_zero` (force_zero, full dataset).

### 3. Zero-init of final output layer (gradient flow issue)

`nn.init.zeros_` on the final MLP layer (weight) delays learning.
With `residual_scale=0.0` controlling the scale gate, the final layer weight
should use standard init so gradients flow through it from step 1 onward.

**Fix:** Removed `nn.init.zeros_(self.mlp[-1].weight)` and
`nn.init.zeros_(self.scorer.weight)` from both model `_init_weights` methods.
Default Kaiming init applies instead.

## Fix Summary

| Item | Before | After |
|------|--------|-------|
| Scale parameter | `log_scale = Parameter(-4.0)` | `residual_scale = Parameter(0.0)` |
| Forward scale | `* self.log_scale.exp()` | `* self.residual_scale` |
| Final layer weight init | `zeros_` | default (Kaiming) |
| Final layer bias init | `zeros_` | default (zeros, kept) |
| Step-0 delta eval | `eval_dataset(max_batches=500)` | `check_init_identity` (first 64 examples) |
| Step-0 CSV row | `m0_full` (biased 32k subset) | `m0_zero` (full 239k dataset) |

Both `RicherMLPRefiner` and `RegionTransformerRefiner` were updated identically.

## Verification Procedure

```bash
sbatch scripts/slurm_debug_refiner_init_identity.sh
```

Expected output for each variant:

```
[init identity check] PASS
    force_zero NLL : 3.XXXXXX
    model NLL      : 3.XXXXXX  (diff=X.XXe-07  threshold=1e-4)
    max |delta|    : 0.00e+00  (threshold=1e-5)
    mean |delta|   : 0.00e+00
    n_valid_cands  : XXXXX
    n_covered      : XX
```

`max |delta| = 0.00e+00` is expected because `residual_scale=0.0` is a
scalar multiplication — `x * 0.0 = 0.0` exactly in float32.

## Next Steps

1. Run `sbatch scripts/slurm_debug_refiner_init_identity.sh` — all variants must PASS
2. Run `sbatch scripts/slurm_debug_refiner_baseline_consistency.sh` — cross-variant NLL must match
3. Run `sbatch scripts/slurm_clean_train_refiners.sh` — full training

The correct comparison table format after training:

| Variant  | official_baseline_nll | best_model_nll | delta  | coverage |
|----------|-----------------------|----------------|--------|----------|
| C        | X.XXXX                | X.XXXX         | X.XXXX | X.XXXX   |
| D3-small | X.XXXX                | X.XXXX         | X.XXXX | X.XXXX   |
| D3-base  | X.XXXX                | X.XXXX         | X.XXXX | X.XXXX   |

All variants share the same `official_baseline_nll` column. A variant is better
only if its `best_model_nll` is lower, not its step-0 NLL.
