# CTF Gold Leakage — Fix Report

## Status: LEAKAGE CONFIRMED → FIX APPLIED

---

## Diagnostic Results (runs/path_refiner_candidate_transformer/gold_leakage_debug/)

| Mode | gated_nll | gain vs baseline | gold_force_rate |
|------|-----------|-----------------|-----------------|
| force_zero (baseline) | 3.378606 | — | 0.0000 |
| oracle_force_gold | 3.110616 | +0.267990 | 0.0781 |
| no_force_gold | 3.675963 | −0.297357 | 0.0000 |
| gap (oracle − no_force) | | **+0.565347** | |

**Verdict: LEAKAGE CONFIRMED.** Gap ≥ 0.05. All prior checkpoint results are invalid.

### What happened

The `select_candidates()` method inside `CandidateSetTransformerRefiner` was force-inserting
the gold candidate into slot `[-1]` of the selected top-M whenever the gold was not naturally
ranked in the top-256 by base logit. This affected ~7.81% of covered positions.

The model learned to assign a high score delta to this last slot, producing an oracle-level
gain (+0.268) that completely collapses (−0.297 vs baseline) when gold is not inserted at
inference time.

---

## INVALID Checkpoints

The following checkpoint directories were produced **with gold force-inclusion** and are
**NOT INFERENCE VALID**. Their `gated_covered_nll` metrics do not reflect real performance.

| Directory | Status |
|-----------|--------|
| `variant_CTF_boundary_M256/best_refiner.pt` | **INVALID — ORACLE LEAKAGE** |
| `variant_CTF_hard_union_M256/best_refiner.pt` | **INVALID — ORACLE LEAKAGE** (if exists) |
| `variant_CTF_type_A_M256/best_refiner.pt` | **INVALID — ORACLE LEAKAGE** (if exists) |

Do not use these checkpoints for any published results or downstream experiments.

---

## Fix Applied (2026-05-18)

### Code changes

**`scripts/train_candidate_transformer_refiner.py`**:
- Added standalone `select_candidates_topm(base, cand_mask, selected_M, mode, ...)` with
  modes `topm_no_gold` (default), `oracle_force_gold_last`, `oracle_force_gold_random`.
- `CandidateSetTransformerRefiner.select_candidates(base, cand_mask)` — removed `gold_idx`
  and `covered` args; always delegates to `select_candidates_topm(..., mode="topm_no_gold")`.
- `forward()` — removed `gold_idx`/`covered` parameters; `_last_sel_stats["gold_forced"]`
  is always all-zeros.
- `canonical_eval_ctf()` — no longer passes gold to model; asserts `gold_force_included_rate == 0`
  (raises `RuntimeError` if nonzero).
- `gated_global_eval_ctf()` — same assertion.
- `compute_ctf_loss()` — removed gold pass-through; added `drop_gold_not_selected` param.
- `train_ctf()` — added safety guard (`RuntimeError` if eval_selection_mode != topm_no_gold`).
- `_parse()` — added `--train_selection_mode`, `--eval_selection_mode`,
  `--drop_train_gold_not_selected`.

**`scripts/train_set_transformer_refiner.py`**:
- Same fixes applied to `SetTransformerRefiner.select_candidates()` and `forward()`.
- Imports `select_candidates_topm` from CTF script.

### New scripts

- `scripts/slurm_debug_ctf_no_leakage.sh` — smoke test verifying `gold_force_included_rate == 0`
  and identity at step 0.
- `scripts/slurm_train_candidate_transformer_refiner.sh` — updated with `_noforce` output dirs
  and `--train_selection_mode topm_no_gold --eval_selection_mode topm_no_gold
  --drop_train_gold_not_selected`.

---

## Clean Training Runs (to be filled after retraining)

Output dirs: `variant_CTF_*_M256_noforce/`

| Run | Output dir | Status |
|-----|-----------|--------|
| boundary | variant_CTF_boundary_M256_noforce | pending |
| hard_union | variant_CTF_hard_union_M256_noforce | pending |
| type_A | variant_CTF_type_A_M256_noforce | pending |

### Expected behaviour after fix

- `gold_force_included_rate = 0.0000` (asserted)
- At step 0: `gated_covered_nll ≈ 3.378606` (identity guaranteed by zero init)
- Training gain will be real (no oracle boost) — expect smaller delta than the invalid +0.268
- If delta is positive vs MLP baseline after 10k steps, CTF has genuine discriminative power
- If delta ≈ 0, M=256 may be insufficient (gold outside top-256 in ~7.8% of cases)

---

## Runtime Safety Guards

Both training scripts now enforce:

1. `eval_selection_mode` must equal `"topm_no_gold"` — raises `RuntimeError` otherwise.
2. `canonical_eval_ctf` and `gated_global_eval_ctf` assert `gold_force_included_rate == 0`
   and raise `RuntimeError` if violated.
3. `best_metrics.json` always records `selection_mode: "topm_no_gold"` and
   `eval_force_include_gold: false` for traceability.
