# Residual Interface Refiner — Final Report

## Status: PENDING (fill after training completes)

---

## Motivation

The no-force CTF run revealed a catastrophic failure: inside the boundary gate,
the CTF model's NLL was **5.718** vs the base score of **4.812** (−0.906 degradation).
This suggests that h_prime + candidate metadata is insufficient for within-region ranking.

Hypothesis: multi-layer backbone states expose more discriminative features before
the region-injection step smooths them out. This experiment tests whether richer
backbone state access can rescue the refiner.

---

## Experimental Setup

| Setting | Value |
|---------|-------|
| backbone | `runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt` |
| candidate shards | `runs/path_refiner_clean/data/{val,train}_hgrid_K24` |
| multilayer features | `runs/path_refiner_residual_interface/features/` (auto_even layers + h_final) |
| gate/train filter | boundary |
| selected_M | 256 |
| refiner_dim | 256 |
| lambda_kl | 0.1 (stronger than CTF's 0.01) |
| selection_mode | topm_no_gold (gold NEVER consulted) |
| drop_train_gold_not_selected | True |
| best checkpoint gate | gated_nll < official_baseline_nll (3.378606) |

---

## Layer Probe Results (from slurm_probe_region_info_by_layer.sh)

Fill from `probes/layer_probe_results.csv`:

| Layer | fine_acc@1 | super_acc@1 | boundary_acc | router_miss_acc |
|-------|-----------|------------|-------------|----------------|
| block_0 | — | — | — | — |
| block_2 | — | — | — | — |
| block_4 | — | — | — | — |
| h_final | — | — | — | — |

**Q_probe_1**: Which layer best predicts gold fine region? **TBD**

**Q_probe_2**: Does region information appear before final h_prime? **TBD**

**Q_probe_3**: Does h_prime blur/lose region information vs best mid-layer? **TBD**

---

## Training Results

### Variant 1: multilayer_crossattn (boundary, M=256)

| Metric | Value |
|--------|-------|
| gated_covered_nll | — |
| delta_vs_baseline | — |
| inside_gate_model_nll | — |
| inside_gate_base_nll | — |
| gate_rate | — |
| best_step | — |
| gold_force_included_rate | 0.0000 (asserted) |
| selection_mode | topm_no_gold |
| eval_force_include_gold | false |

Best checkpoint saved: **TBD** (only if gated_nll < 3.378606)

### Variant 2: region_state_crossattn (boundary, M=256)

| Metric | Value |
|--------|-------|
| gated_covered_nll | — |
| delta_vs_baseline | — |
| inside_gate_model_nll | — |
| inside_gate_base_nll | — |
| gate_rate | — |
| best_step | — |
| gold_force_included_rate | 0.0000 (asserted) |
| selection_mode | topm_no_gold |
| eval_force_include_gold | false |

Best checkpoint saved: **TBD**

---

## Comparison vs Known Baselines

| Method | gated_covered_nll | delta_vs_baseline | Notes |
|--------|------------------|------------------|-------|
| force-zero baseline | 3.378606 | 0.000000 | official reference |
| global MLP (variant C) | — | +0.001284 | global gate |
| hard boundary MLP | — | +0.000530 | boundary gate; local_gain=+0.003209 |
| CTF no-force (failed) | 3.675963 | −0.297357 | catastrophic; inside_gate=5.718 |
| **RIR multilayer_crossattn** | **TBD** | **TBD** | |
| **RIR region_state_crossattn** | **TBD** | **TBD** | |

---

## Questions Answered

**Q1**: Does multilayer_crossattn beat the hard boundary MLP? (target: delta > +0.000530)
> **TBD**

**Q2**: Does multilayer_crossattn beat the global MLP? (target: delta > +0.001284)
> **TBD**

**Q3**: Does region_state_crossattn improve over the failed no-force CTF?
> **TBD** — CTF inside_gate_model_nll was 5.718 vs base 4.812; any positive result here matters.

**Q4**: Is inside_gate_model_nll < inside_gate_base_nll for either variant?
> **TBD** — This is the key check: does the model help within the boundary gate?

**Q5**: Did any variant beat the official baseline at all?
> **TBD** — If neither saves a checkpoint, the residual interface hypothesis is also wrong.

**Q6**: Which layer carries the most discriminative region signal (from probe report)?
> **TBD** — See `probes/layer_probe_report.md`.

**Q7**: Does the best RIR layer align with the best probe layer?
> **TBD** — If multilayer_crossattn works, check which ctx layer weights are highest.

**Q8**: What is the coverage/fallback rate (gold outside top-256)?
> **TBD** — Expect ~7.8% from CTF diagnostics (same M=256, same base scores).

---

## Failure Modes to Investigate

If no variant beats baseline:
- Check if inside_gate_model_nll > inside_gate_base_nll (model is hurting)
- Reduce selected_M (gold outside top-256 in ~7.8% of cases)
- Try M=512 — gold coverage improves but memory cost doubles
- Check probe acc@1 — if mid-layer probes are weak, residual features may be non-discriminative
- Consider that boundary positions may genuinely require external retrieval (KNN path)

If multilayer_crossattn helps but region_state_crossattn does not:
- Mid-layer states contain discriminative features not recoverable from candidate metadata
- Residual state access is necessary; region-state summarisation loses the signal

If region_state_crossattn helps but multilayer_crossattn does not:
- h_prime carries sufficient region info; per-layer states add noise via cross-attention
- Region-level aggregation is the right abstraction

---

## Next Steps (to be decided after results)

- [ ] If any variant beats baseline: full M=512 run
- [ ] If both fail: investigate KNN-augmented path (bring back retrieval signal)
- [ ] Update comparison table in project-level README
