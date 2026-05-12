# RegionTokenizer — Architecture

## Core Hypothesis

Transformers progressively resolve token prediction uncertainty across layers. At early layers, hidden states localize to a coarse neighbourhood in representation space (high router entropy, low margin); at later layers, commitment sharpens to a single dominant region (low entropy, high margin). Tokens near region decision boundaries — **boundary tokens**, those with low top1−top2 routing margin — remain genuinely ambiguous and benefit from richer multi-region conditioning than confident **core tokens**.

The architecture family tests whether making this structure explicit in the model (via learned region maps, adaptive alpha, and various forms of extra compute at boundary positions) improves language modelling perplexity.

---

## Region Map

A region map is a flat JSON file `token_to_region.json` mapping GPT-2 BPE token IDs to integer region IDs:

```json
{"100": 0, "101": 5, "4312": 0, ...}
```

Loaded at startup into a `(vocab_size,)` long tensor; unknown tokens get `-1`. The number of regions `n_coarse` is inferred from the max region ID + 1.

Region maps are built via `--mode build_regions` which sweeps a baseline checkpoint over a corpus, constructs a token co-activation graph (tokens that co-appear in the model's top-k predictions), and clusters it with spectral clustering / Leiden / k-means. Maps for different cluster counts live in `runs/region_maps_<n>/`.

A hierarchical **leaf map** (`region_tree.json`) is optionally built on top of the coarse map for fine-grained conditioning (legacy coarse_leaf modes only).

---

## Boundary Gate

All boundary-aware models share the same gating mechanism:

```
gate(x) = sigmoid( (boundary_tau − margin(x)) / boundary_temp )
```

where `margin = top1_prob − top2_prob` of the routing softmax at position x.

- `gate ≈ 1` → boundary token (low margin, ambiguous routing)
- `gate ≈ 0` → core token (high margin, confident routing)

Default: `boundary_tau=0.10`, `boundary_temp=0.05`.  
The gate is always computed on `p_region.detach()` to prevent the LM loss from artificially inflating the boundary fraction.

---

## Model Family

### BaselineTransformerLM

Standard GPT-2-style decoder. No regions. Used as the probe backbone in analysis modes and as the comparison anchor for all region-conditioned models.

```
token_emb + pos_emb → Dropout → [N × TransformerBlock] → LayerNorm → lm_head
```

---

### SoftMoETransformerLM  *(modes: soft_moe, oracle_soft_moe, random_soft_moe)*

Region information modulates the **output logit** rather than the hidden state. The router produces a distribution over regions; region membership of each vocabulary token is used as a prior:

```
p_region = softmax(coarse_head(h) / router_temp)          # (B, T, K)
log_prior = p_region @ log(membership).T                   # (B, T, V)
logits    = lm_head(h) + prior_gamma * log_prior
```

`membership` is a `(K, V)` binary matrix where `M[r, v] = 1` iff token v is in region r. Soft blend over region log-likelihoods acts as a learned mixture prior on top of the standard LM head.

---

### ReprRegionTransformerLM  *(modes: repr_region, oracle_repr_region, random_repr_region)*

Region information is injected into the **hidden state** as a soft mixture of learned region embeddings:

```
p_region   = softmax(coarse_head(h) / router_temp)        # (B, T, K)
r          = p_region @ region_emb.weight                  # soft mixture (B, T, d_region)
region_feat = region_proj(region_mlp(r))                   # (B, T, d_model)
h′         = h + α · region_feat                           # α ramped up from 0 over warmup_steps
logits     = lm_head(h′)
```

This is the **reference model** for the boundary family — all boundary variants extend it.

---

### ReprRegionCapacityLM  *(modes: repr_region_capacity, ...)*

Extends `ReprRegionTransformerLM` with three mechanisms to prevent router collapse:

| Mechanism | Description |
|-----------|-------------|
| EMA usage tracking | Per-region usage `(K,)` tracked with exponential moving average |
| Capacity penalty | `adjusted_logits = region_logits − capacity_alpha · log(ema_usage + ε)` — penalises overused regions |
| Temperature annealing | Router temperature decays `router_temp_init → router_temp_final` over `router_temp_decay_steps` steps |

An optional diversity loss maximises the entropy of the mean routing distribution to further discourage collapse.

---

### ReprRegionBoundaryLM  *(modes: repr_region_boundary, oracle_repr_region_boundary, random_repr_region_boundary)*

**Hypothesis:** boundary tokens need stronger region conditioning than core tokens.

Replaces the fixed scalar `α` with a per-token adaptive alpha:

```
gate       = boundary_gate(p_region.detach())              # (B, T) ∈ [0, 1]
α_dyn      = (α_core · (1 − gate) + α_boundary · gate) · warmup_scale
h′         = h + α_dyn.unsqueeze(−1) · region_feat
```

`alpha_core < alpha_boundary` (default 0.2 / 0.4). Core tokens get weaker injection; boundary tokens get stronger.

---

### ReprRegionBoundarySeqRefineLM  *(modes: repr_region_boundary_seqrefine, random_...)*

**Hypothesis:** boundary tokens need contextual (causal attention) computation, not just static feature injection.

```
h_core  = h + α_core · warmup_scale · region_feat             # all tokens
h_seq   = seq_refine_blocks(h_core)                           # full-sequence causal blocks
scale   = tanh(refine_scale) · boundary_refine_gamma · warmup_scale
h_final = h_core + gate.unsqueeze(−1) · scale · (h_seq − h_core)
```

ALL tokens pass through `seq_refine_blocks` (preserving causal context), but only boundary tokens (high gate) carry the resulting delta into the final hidden state. The refine contribution starts near zero at init (`refine_scale` is a learned scalar, initialised small).

---

### ReprRegionBoundaryAdaptiveDepthLM  *(modes: repr_region_boundary_adaptivedepth, random_...)*

**Hypothesis:** sparse hard assignment — extra compute should only flow through token positions that actually cross a decision boundary.

```
boundary_mask = (gate > adaptive_threshold).float()           # hard binary (B, T)
h_core     = h + α_core · warmup_scale · region_feat
h_refined  = adaptive_blocks(h_core)                          # full sequence (causal context)
delta      = h_refined − h_core
scale      = tanh(refine_scale) · warmup_scale
h_final    = h_core + boundary_mask.unsqueeze(−1) · scale · delta
```

Key difference from `seqrefine`: the gate is a **hard threshold** not a continuous weight. Only positions where `gate > 0.5` (default) are mathematically modified. `frac_tokens_refined` should track approximately `boundary_frac` as a sanity check.

---

### ReprRegionMultiHypLM  *(modes: repr_region_multihyp, oracle_..., random_...)*

**Hypothesis:** rather than collapsing to a single soft mixture, retain the top-K most plausible region hypotheses independently for ambiguous tokens.

**Core path** (confident tokens, gate ≈ 0):
```
r_soft  = p_region @ region_emb.weight
h_core  = h + α_core · region_proj(region_mlp(r_soft))
```

**Multi-hypothesis path** (boundary tokens, gate ≈ 1):
```
top-K probs/indices = p_region.topk(hyp_k)                    # renormalised
hyp_feats = region_proj(region_mlp(region_emb(topk_idx)))      # (B, T, K, d_model)
h_hyps    = h.unsqueeze(2) + α_boundary · hyp_feats            # (B, T, K, d_model)
h_refined = BoundaryRefiner(h_hyps)                            # lightweight residual MLP per hyp
h_bnd     = (topk_probs.unsqueeze(−1) · h_refined).sum(dim=2)  # weighted merge
```

**Blend:**
```
h_final = (1 − gate) · h_core + gate · h_bnd
```

`BoundaryRefiner` is a small residual MLP (Linear→GELU→Linear) shared across the K hypothesis dimension; initialised near identity.

---

### ReprRegionBranchAttnLM  *(modes: repr_region_branchattn, repr_region_branch_identity)*

**Hypothesis:** hypothesis-dimension attention allows the model to learn which region hypotheses are relevant given the sequence context.

Shares the multi-hypothesis generation with `ReprRegionMultiHypLM`, then applies a `BranchAttentionRefiner` over the K dimension:

```
h_hyps: (B, T, K, d_model)
→ reshape to (B·T, K, d_model)
→ [branch_attn_layers × standard Transformer block over the K dimension]
→ delta = refined − flat
→ output = flat + tanh(branch_scale) · delta
→ reshape back to (B, T, K, d_model)
```

Attention over K lets each hypothesis attend to the others (cross-hypothesis interaction) before the probability-weighted merge.

`repr_region_branch_identity` uses the same architecture but skips the refiner entirely — a diagnostic to check whether branch construction alone (without cross-hypothesis attention) is beneficial.

---

## Router Head

All models share the same router head factory:

| `router_type` | Structure |
|---------------|-----------|
| `"linear"` | `Linear(d_model, n_coarse, bias=False)` |
| `"mlp"` | `LayerNorm → Linear → GELU → Dropout → Linear` (d_model → d_model → n_coarse) |

The MLP router is used in all recent experiment runs. The router output is always stored as `coarse_head` in the state dict, allowing the standalone probe scripts to extract it by key prefix regardless of model variant.

---

## Training Modes Reference

| Family | Modes | Model Class |
|--------|-------|-------------|
| Baseline | `baseline` | `BaselineTransformerLM` |
| Legacy coarse | `coarse`, `coarse_leaf`, `oracle_coarse`, `oracle`, `random_control` | `RegionConditionedTransformerLM` |
| Soft MoE | `soft_moe`, `oracle_soft_moe`, `random_soft_moe` | `SoftMoETransformerLM` |
| Repr region | `repr_region`, `oracle_repr_region`, `random_repr_region` | `ReprRegionTransformerLM` |
| Capacity | `repr_region_capacity`, `oracle_...`, `random_...` | `ReprRegionCapacityLM` |
| Boundary | `repr_region_boundary`, `oracle_...`, `random_...` | `ReprRegionBoundaryLM` |
| Seq refine | `repr_region_boundary_seqrefine`, `random_...` | `ReprRegionBoundarySeqRefineLM` |
| Adaptive depth | `repr_region_boundary_adaptivedepth`, `random_...` | `ReprRegionBoundaryAdaptiveDepthLM` |
| Multi-hyp | `repr_region_multihyp`, `oracle_...`, `random_...` | `ReprRegionMultiHypLM` |
| Branch attn | `repr_region_branchattn`, `repr_region_branch_identity` | `ReprRegionBranchAttnLM` |
| Build | `build_regions` | — |
| Analysis | `layer_margin_analysis`, `boundary_analysis` | — |

**Oracle variants** feed the true next-token region as a one-hot directly (upper bound). **Random variants** permute the region map while preserving class sizes (null hypothesis: does structure matter, or is it just label consistency?).

---

## Loss Functions

```
L = L_lm + λ_coarse · L_coarse + λ_leaf · L_leaf + λ_balance · L_balance
```

| Term | Formula | Default weight |
|------|---------|---------------|
| `L_lm` | Cross-entropy on next-token prediction | 1.0 |
| `L_coarse` | Cross-entropy between `coarse_head` logits and gold region IDs | `λ_coarse = 0.2` |
| `L_leaf` | Cross-entropy on leaf router (legacy modes only) | `λ_leaf = 0.01` |
| `L_balance` | `−mean(H(p_region))` — penalises low router entropy | `λ_balance = 0.001` |
| `L_diversity` | `−H(mean_batch(p_region))` — maximises usage spread (capacity mode) | `λ_diversity = 0.0` |

The coarse auxiliary loss (`L_coarse`) is the main supervision signal that keeps the router meaningful. Without it, the router degrades toward uniform outputs since LM loss alone doesn't require region structure.

---

## Region Warmup

All region-conditioned models ramp the injection scale from 0 to its target value over `region_warmup_steps` steps (default 5000):

```python
scale = min(1.0, step / region_warmup_steps)   # 0 → 1 linearly
```

This allows the LM trunk to learn a reasonable baseline before region features are switched on, avoiding early instability.

---

## Analysis Modes

### `layer_margin_analysis`

Probes a frozen trained `repr_region` checkpoint's `coarse_head` layer-by-layer against a baseline transformer's hidden states. Tests Q1–Q7 of the progressive manifold resolution hypothesis:

| Q | Question |
|---|----------|
| Q1 | Gold region enters top-4 at the embedding layer (above 2× chance)? |
| Q2 | Top-1 sharpens later than top-4 (relative acc@4−acc@1 gap narrows)? |
| Q3 | Entropy decreases progressively (final < embed − 0.1)? |
| Q4 | Margins increase progressively (final > embed + 0.02)? |
| Q5 | Some tokens remain persistently low-margin even at the final layer? |
| Q6 | Top-1 accuracy gains non-trivially in the second half of the network? |
| Q7 | Context dynamically sharpens routing uncertainty (high std_margin tokens)? |

Outputs: `layer_metrics.csv`, `layer_trajectories.csv`, `token_margin_variance.csv`, `summary.md`, 8 plots.

### `boundary_analysis`

Evaluates a trained `repr_region` checkpoint on a validation set. Bins token positions by routing margin and computes NLL, oracle NLL, and region accuracy in each bin. Tests the same Q1–Q7 but on the final-layer representation.

### `scripts/gpt2xl_layer_margin_analysis.py`

Standalone version of `layer_margin_analysis` using GPT-2 XL (48 layers, d_model=1600) as the backbone. Trains lightweight per-layer `RegionProbe` modules (LayerNorm+Linear, d_model→n_regions) on WikiText-103 from scratch, then evaluates all 49 hidden states. Extends to Q8 (sharpening elbow location) and acc@16. Outputs 9 plots including probe training loss curves.

---

## Key Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `d_model` | 384 | Hidden dimension (experiments use 384; GPT-2 XL uses 1600) |
| `n_layer` | 6 | Transformer depth |
| `n_coarse` | 128 | Number of regions (inferred from map) |
| `router_type` | `mlp` | Router head architecture |
| `router_temp` | 2.0 | Softmax temperature on router logits |
| `base_alpha` | 0.1 | Max region injection scale (repr_region) |
| `alpha_core` | 0.2 | Injection scale for confident core tokens |
| `alpha_boundary` | 0.4 | Injection scale for ambiguous boundary tokens |
| `boundary_tau` | 0.03–0.10 | Margin threshold separating boundary from core |
| `boundary_temp` | 0.01–0.05 | Gate sigmoid sharpness |
| `region_warmup_steps` | 5000 | Steps to ramp injection scale 0 → target |
| `lambda_coarse` | 0.2 | Router auxiliary loss weight |
| `lambda_balance` | 0.001 | Router entropy regulariser weight |
| `hyp_k` | 4 | Number of parallel region hypotheses (multihyp/branchattn) |
| `adaptive_threshold` | 0.5 | Hard gate threshold (adaptive depth) |

---

## Checkpoint Format

All checkpoints are saved as:

```python
{
    "step":  int,
    "model": state_dict,
    "cfg":   dataclasses.asdict(TrainConfig),
    "opt":   optimizer_state_dict,     # omitted in final_summary saves
}
```

The `coarse_head.*` keys in the state dict are the router weights and can be extracted independently to build a standalone probe without reconstructing the full model.
