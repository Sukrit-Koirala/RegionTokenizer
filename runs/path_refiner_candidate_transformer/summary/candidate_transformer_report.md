# Candidate-set Transformer Refiner — Results Report

Fill in each section after runs complete. Reference files:
- `runs/path_refiner_candidate_transformer/variant_CTF_*/best_metrics.json`
- `runs/path_refiner_candidate_transformer/variant_CTF_*/local_subset_eval.csv`
- `runs/path_refiner_hard/variant_C_*/best_metrics.json`  (MLP hard baselines)
- `runs/path_refiner_clean/baselines/saved_candidate_baseline.json`  (force-zero)

---

## Baselines

| Model | Filter | gated_covered_nll | delta_vs_baseline |
|-------|--------|-------------------|-------------------|
| Force-zero | — | 3.378606 | — |
| Global MLP (variant C) | — | _3.377322_ | _+0.001284_ |
| Hard boundary MLP | boundary | _TBD_ | _+0.000530 (global)_ |
| Hard boundary MLP | boundary | — | _+0.003209 (local)_ |

---

## Q1: Does CTF beat hard boundary MLP globally?

**Target**: gated_covered_nll < hard boundary MLP gated_nll

| Model | gated_covered_nll | delta_vs_baseline | step |
|-------|-------------------|-------------------|------|
| Hard boundary MLP | | | |
| CTF boundary M=256 | | | |

**Answer**:

**Why / why not**:

---

## Q2: Does CTF beat global MLP globally?

**Target**: gated_covered_nll < 3.377322 (delta > +0.001284)

| Model | gated_covered_nll | delta_vs_baseline |
|-------|-------------------|-------------------|
| Global MLP (best) | 3.377322 | +0.001284 |
| CTF boundary M=256 | | |
| CTF hard_union M=256 | | |

**Answer**:

**Why / why not**:

---

## Q3: Inside-boundary NLL — CTF vs hard boundary MLP?

Source: `local_subset_eval.csv`, subset=`boundary`

| Model | inside_boundary_nll | base_nll | local_delta |
|-------|---------------------|----------|-------------|
| Hard boundary MLP | | 4.812637 | |
| CTF boundary M=256 | | | |

**Answer**:

---

## Q4: router_top8_miss improvement?

Source: `local_subset_eval.csv`, subset=`router_top8_miss`

| Model | n_covered | model_nll | base_nll | delta_nll |
|-------|-----------|-----------|----------|-----------|
| CTF boundary M=256 | | | | |
| CTF hard_union M=256 | | | | |

**Answer**:

---

## Q5: Gold outside selected M — forced inclusion rate?

Source: `best_metrics.json` → `gold_force_included_rate`

| Model | gold_force_included_rate | notes |
|-------|--------------------------|-------|
| CTF boundary M=256 | | |
| CTF hard_union M=256 | | |
| CTF type_A M=256 | | |

**Answer**: Is M=256 large enough? What fraction of covered positions have gold forced in?

**Implication for M sensitivity**: if rate is high (>5%), try M=512.

---

## Q6: hard_union CTF vs hard_union MLP?

| Model | gate_filter | gated_covered_nll | delta_vs_baseline |
|-------|-------------|-------------------|-------------------|
| Hard union MLP | hard_union | | |
| CTF hard_union M=256 | hard_union | | |

**Answer**:

---

## Q7: type_A CTF results?

Source: `best_metrics.json` + `local_subset_eval.csv`, subset=`type_A`

| Model | gated_covered_nll | inside_type_A_nll | delta_nll |
|-------|-------------------|-------------------|-----------|
| CTF type_A M=256 | | | |

**Answer**: type_A positions are structurally ambiguous (multiple valid region paths). Does CTF exploit candidate competition to resolve them?

---

## Q8: What do the region/super embeddings contribute?

_(Ablation — fill after optional ablation run.)_

Comparison: CTF with region/super embeddings vs CTF with only base-logit + rank.

| Variant | inside_boundary_nll | notes |
|---------|---------------------|-------|
| Full CTF (with region emb) | | |
| Ablated (no region emb) | | |

---

## Q9: Residual scale evolution

Source: `train_log.csv` → `residual_scale` column

| Model | scale @ step 0 | scale @ best step | notes |
|-------|----------------|-------------------|-------|
| CTF boundary | 1.0 | | |
| CTF hard_union | 1.0 | | |

**Answer**: Does residual_scale grow or shrink? What does that indicate about transformer confidence vs base logit?

---

## Q10: Should we train inside the LM (unfreeze transformer)?

Based on these CTF results:

- If delta > +0.003 globally → the refiner can absorb head-level bias; LM training likely won't add much.
- If delta < +0.001 even with transformer → the bottleneck is the candidate set itself, not scoring; consider larger K or different retrieval.
- If gold_force_included_rate > 10% → M=256 is too small; training inside the LM would shift h_prime such that gold is naturally ranked higher.

**Recommendation**:

---

## Training curves (summary)

### CTF boundary M=256

| step | ce | gated_nll | inside_gate_model_nll | inside_gate_base_nll | gold_forced |
|------|----|-----------|-----------------------|----------------------|-------------|
| 0 | — | — | — | — | — |
| 1000 | | | | | |
| 2000 | | | | | |
| ... | | | | | |
| best | | | | | |

---

## Conclusions

1. CTF vs hard MLP:
2. CTF vs global MLP:
3. Gold force-inclusion rate:
4. Type A + router_miss benefit:
5. Recommended next step:
