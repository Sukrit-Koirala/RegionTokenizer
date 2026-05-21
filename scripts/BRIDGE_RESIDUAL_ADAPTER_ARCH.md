# Bridge Residual Adapter — Architecture Summary

## Hypothesis

Prior refiners (CTF, MLP, residual interface) all operated by **patching logits or hidden states after the backbone has already made its routing decision**. The core failure mode: if you patch logits over a top-M candidate set, you are forced to include gold in the candidates to compute a valid training signal, which leaks gold into the selection step and inflates eval metrics.

The bridge adapter tests a different claim:

> A small specialist should not patch logits after the decision.  
> It should build a path-conditioned **residual update** before final token decoding.

By writing `delta_h` back into the residual stream *before* decoding via `h_refined @ emb.T`, the model decodes over the **full vocabulary** (50,257 tokens). There is no candidate set, no gold-force inclusion, and `gold_force_included_rate = 0` structurally.

---

## System Overview

```
  Frozen backbone (ReprRegionRetrievalLM)
       │
       ├─ h_layers  (N, n_layers, d_model) float16  ← pre-extracted, blocks [0,2,4,5,final]
       └─ h_prime   (N, d_model)           float32  ← h + α·region_feat (already has region info)

  BridgeResidualAdapter (trainable, ~D params)
       │
       ├─ Reads: h_layers, h_prime, router top-K, memory top-K, candidate metadata
       ├─ Builds bridge token sequence (40 tokens)
       ├─ Runs Pre-LN TransformerEncoder (L=2, H=4, D=256, ff=4×)
       ├─ Extracts PATH token → delta_h = out_proj(path_out)   [zero at init]
       └─ Writes: h_refined = h_prime + α · delta_h

  Decode
       logits_refined = h_refined @ token_emb.T     (V = 50,257)

  Gate (eval / loss)
       gated_logits[i] = logits_refined[i]  if gate[i]    (in-filter positions)
                       = logits_base[i]     otherwise      (outside-gate positions unchanged)
```

---

## Bridge Token Sequence (length = 40)

| Slots | What | How built |
|---|---|---|
| 0–4 | **Layer tokens** (5) | `layer_proj(h_layers[l]) + layer_id_emb(l)` for l in [0,2,4,5,final] |
| 5 | **ROUTER token** (1) | Weighted pool of router region embeddings + entropy/margin/top1-prob scalars |
| 6 | **MEMORY token** (1) | Same, for memory top-K |
| 7–30 | **Fine-region tokens** (24) | Top-12 from router + top-12 from memory; 7-feature scalars per region: [r_prob, m_prob, in_r, in_m, n_cands_frac, max_base_logit, mean_base_logit] |
| 31–38 | **Superregion tokens** (8) | Scatter-aggregate router+memory probs by superregion; top-8 by evidence |
| 39 | **PATH token** (1) | Learned query; its output → delta_h |

No gold token, gold region, or gold candidate index appears anywhere in the bridge sequence or the adapter forward pass.

---

## Model Components (`BridgeResidualAdapter`)

| Component | Role |
|---|---|
| `layer_proj` (Linear d→D) | Projects each backbone layer state into bridge_dim |
| `layer_id_emb` (Emb n×D) | Positional embedding per layer slot |
| `fine_emb` (Emb R×D) | Fine-region embeddings (shared by router/memory tokens and fine-region tokens) |
| `super_emb` (Emb S×D) | Superregion embeddings |
| `router_scalar_proj` (3→D) | Projects [entropy, margin, top1_prob] for router token |
| `mem_scalar_proj` (3→D) | Same for memory token |
| `reg_scalar_proj` (7→D) | Per-region scalar features → bridge_dim |
| `super_scalar_proj` (3→D) | Per-superregion scalar features → bridge_dim |
| `path_token` (1×D param) | Learned PATH query |
| `transformer` (Pre-LN, L layers) | Full attention over all 40 tokens |
| `out_proj` (D→d_model) | **Zero-initialized** → delta_h = 0 at step 0 |
| `alpha` (scalar param, init=1) | Learnable residual scale |

**Identity invariant**: because `out_proj` is zero-initialized, `delta_h = 0` at step 0, `h_refined = h_prime`, and `logits_refined = logits_base`. Step-0 eval asserts this holds for all three metric families (`_all`, `_covered`, `masked_cand`).

---

## Training Setup

### Data pipeline

```
Candidate shards  (train_hgrid_K24/)         — h_prime, cand_tok/fine/super, router/memory top-K
Feature shards    (features/train_multilayer) — h_layers (float16, aligned by gold_token)
```

Shards are aligned: `gold_token` in both must match exactly (RuntimeError otherwise).

### Dataset modes

| Mode | Class | Row selection |
|---|---|---|
| Default (noisy) | `BridgeShardDataset` | All rows; `train_mask = filter_mask` post-collation → n_train ≈ 3–9 |
| **Hard sampler** | `FilteredBridgeShardDataset` | Pre-filters to only yield rows where `filter_mask == True`; `train_mask = None` (all rows) → n_train = batch_size |

The hard sampler is the correct mode for dense gradient signals. `covered` is not required for row selection (full-vocab has valid gold_token everywhere); `covered` is still yielded for masked-candidate eval.

### Loss

```
L = CE(logits_refined[train_mask], gold_token[train_mask])           ← full-vocab CE
  + λ_kl  · KL(base_topk_probs || refined_topk_log_probs)           ← top-512 KL regularizer
  + λ_Δ   · ||delta_h[train_mask]||²                                 ← delta L2 regularizer
```

Top-K KL (K=512) is a memory-efficient approximation of full KL over 50k vocab.

### Gradient accumulation

```
effective_hard_batch_size = batch_size × grad_accum_steps
```

Default: `batch_size=64, grad_accum_steps=1` → eff_bs=64.  
Memory fallback: `batch_size=32, grad_accum_steps=2` → same eff_bs, half activation memory.

### Filter / gate

`--train_filter boundary` selects positions where the backbone's routing is uncertain (boundary between regions). Same filter used as gate during eval: inside-gate positions use `logits_refined`; outside-gate positions use `logits_base` unchanged.

---

## Evaluation Metrics

### Primary — full-vocab `_all` (every position)

| Metric | Description |
|---|---|
| `full_vocab_base_nll_all` | CE(h_prime @ emb.T, gold) over all N=239,362 val positions |
| `full_vocab_gated_nll_all` | CE(gated_logits, gold) — **threshold for best checkpoint** |
| `full_vocab_gain_all` | base − gated (positive = improvement) |
| `full_vocab_inside_gate_base_nll_all` | CE(base, gold) for in-gate positions |
| `full_vocab_inside_gate_ref_nll_all` | CE(refined, gold) for in-gate positions |
| `full_vocab_outside_gate_gated_nll_all` | Must equal `outside_gate_base_nll_all` (diff < 1e-5) |

### Secondary — full-vocab `_covered` (227,017 positions, 94.8%)

Same metrics with `_covered` suffix. Covered = gold token has a region assignment.

### Secondary — masked-candidate (covered only)

CE restricted to the ~100–300 candidate logits. Used for comparison with prior MLP/CTF refiners.

| Reference | Value |
|---|---|
| `masked_cand_baseline` | 3.378606 (canonical, asserted at step 0) |
| Global MLP gain | +0.001284 |
| Hard-boundary MLP gain | +0.000530 |

### Success criteria

| Question | What to check |
|---|---|
| Any gain? | `full_vocab_gated_nll_all < full_vocab_base_nll_all` |
| Residual write helps? | `inside_gate_ref_nll_all < inside_gate_base_nll_all` |
| Gate is clean? | `outside_gate_diff_all ≈ 0` |
| Beats hard MLP? | `masked_cand_gain > +0.000530` |
| Beats global MLP? | `masked_cand_gain > +0.001284` |

---

## Canonical Invariants

| Constant | Value |
|---|---|
| `CANONICAL_FINGERPRINT` | `09b0a71955cc9c43` |
| `CANONICAL_NUM_EXAMPLES` | 239,362 |
| `CANONICAL_NUM_COVERED` | 227,017 |
| `CANONICAL_COVERAGE` | 0.948425 |
| `MASKED_CAND_BASELINE_NLL` | 3.378606 |

The fingerprint is a SHA-256 hash over `{num_shards, total_n, total_cov, sum_cand_counts, sum_gold_idx_cov, sum_gold_tok}` from the val set. Any dataset drift triggers RuntimeError.

---

## Safety Assertions (all runtime)

1. `delta_norm_max == 0` at step 0 (out_proj zero-init)
2. `full_vocab_gated_nll_all == full_vocab_base_nll_all` at step 0 (diff < 1e-3)
3. `full_vocab_gated_nll_covered == full_vocab_base_nll_covered` at step 0 (diff < 1e-3)
4. `masked_cand_gated_nll == masked_cand_base_nll` at step 0 (diff < 1e-3)
5. `masked_cand_base_nll == 3.378606` (diff < 1e-3) — canonical dataset check
6. `outside_gate_gated_nll_all == outside_gate_base_nll_all` (diff < 1e-5) — gating invariant
7. `gold_force_included_rate == 0.0` — structural (no candidate selection step exists)
8. Canonical fingerprint/coverage match
9. Best checkpoint threshold = `full_vocab_base_nll_all` (not masked baseline, not float("inf"))

---

## Run Order

```
slurm_build_multilayer_residual_features.sh   ← extract h_layers for all shards
slurm_probe_region_info_by_layer.sh           ← (optional) verify region signal in layers
slurm_debug_bridge_residual_adapter.sh        ← MUST PASS: identity check, all 9 assertions
slurm_debug_hard_sampler_bridge.sh            ← MUST PASS: filter_match_rate=1.0 for sampler
slurm_train_bridge_residual_adapter.sh        ← 5000 steps, boundary filter, hard sampler
```

### Output files (per run)

| File | Contents |
|---|---|
| `full_vocab_baseline.json` | Step-0 base NLLs (`_all`, `_covered`, masked), fingerprint |
| `train_log.csv` | Per-step: ce, kl, delta_norm, n_train, alpha, lr |
| `eval_log.csv` | Per-eval: all `_all`/`_covered`/masked metrics |
| `local_subset_eval.csv` | Per-filter-subset breakdown |
| `best_refiner.pt` | Best checkpoint (only exists if model beat base NLL) |
| `best_metrics.json` | All metrics at best checkpoint |
| `final_metrics.json` | Summary: no_improving_checkpoint, best_step, gains |
| `last_refiner.pt` | Final step checkpoint (always saved) |

---

## Comparison to Prior Refiners

| Approach | Selection | Gold leakage? | Primary metric |
|---|---|---|---|
| CTF (failed) | Top-M candidates | Yes (gold forced in) | masked_cand (inflated) |
| Global MLP | Top-M candidates | Yes (eval force-include) | masked_cand +0.001284 |
| Hard-boundary MLP | Top-M candidates | Yes | masked_cand +0.000530 |
| **Bridge Adapter** | None (full vocab) | **No** | full_vocab_all (honest) |

The bridge adapter is the first architecture in this pipeline with a structurally valid eval.
