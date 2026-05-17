# Hard-Position Refiner — Architecture

## Motivation

The global path refiner (Variant C, trained on all positions) achieved only
`+0.001284 nats` gain over the force-zero baseline (`best_nll=3.377323` vs
`baseline_nll=3.378606`).

The reason the gain is so small: most of the 239,362 val positions are easy.
The backbone is already confident on them, and the correct token is already
ranked first by the base score (`h_prime · tok_emb[gold]`). Training on all
positions means the model sees a huge mass of examples where `delta` should
be near zero, and the gradient from those examples pushes `residual_scale` and
the MLP weights toward doing nothing. The model learns a small *average*
correction rather than a targeted one for the cases where the backbone is
actually wrong.

**Hypothesis**: if we train exclusively on positions where the backbone
struggles — boundary positions, off-manifold types, router misses — and apply
the refiner only there (delta=0 on easy positions), we get a larger gain where
it matters without degrading easy ones.

---

## What a Shard Is

Each `.pt` file holds ~500 positions. A "position" is one token-prediction
moment captured from the frozen backbone during a forward pass over text. The
key fields per position `b`:

```
h_prime[b]           (384,)  — backbone hidden state at that position
cand_tok[b]          (C,)    — ~100-200 candidate token IDs (padded to fixed width with -1)
cand_fine[b]         (C,)    — which fine region (0-127) each candidate's token belongs to
gold_cand_idx[b]     scalar  — index into cand_tok that is the correct answer
covered[b]           bool    — True if the correct token appears in cand_tok at all
split[b]             0-3     — core / medium / boundary / tight  (difficulty by region split)
type_arr[b]          0-3     — other / A / B / C  (position type classification)
router_topk_reg[b]   (K,)    — top-K region IDs the router predicted for this position
router_margin[b]     scalar  — margin between router top-1 and top-2 probs (low = uncertain)
gold_region[b]       scalar  — the correct fine region for this position

```

If `covered[b]` is False, the correct token is not in the candidate set — you
cannot compute a meaningful CE loss and cannot train on that position.
All NLL metrics are computed only over covered positions.

---

## System Overview

```
val shards (all 239,362 positions)
         │
         ├─── gate_mask = compute_filter_mask(filter_name)
         │         (precomputed from shard metadata, no model call)
         │
         ├─ inside gate  (~5–50% depending on filter)
         │       model_scores = base + residual_scale * mlp(features)
         │
         └─ outside gate  (remainder)
                 scores = base   (delta forced to 0, no model call at all)

gated_scores → log_softmax → CE over covered positions → gated_covered_nll
```

Training sees ONLY positions inside `train_filter` — the dataset iterator
skips everything else. Validation always covers all 239,362 positions because
the gated eval computes the global NLL using model scores inside the gate and
base scores outside. This separation prevents the model from gaming the metric
by only performing well on whatever subset it was trained on, and it keeps the
primary metric comparable across all six filter runs.

---

## Model: RicherMLPRefiner (Variant C)

Shared with the global refiner experiment — no architectural changes.

### Base Score

For every candidate `c` at position `b`, the base score is a dot product:

```python
base[b, c] = h_prime[b] · tok_emb[cand_tok[b, c]]    # scalar
```

This approximates the backbone's raw logit for that candidate token. It is
what you would get with no refiner at all — the force-zero baseline. The
model learns a residual correction on top of this.

### Feature Assembly

The MLP takes a 134-dim feature vector per `(position b, candidate c)` pair:

```
h_proj(h_prime[b])             → (32,)   compressed hidden state
tok_proj(tok_emb[cand[b,c]])   → (32,)   compressed token embedding
fine_emb(cand_fine[b,c])       → (32,)   learned embedding for candidate's fine region
super_emb(cand_super[b,c])     → (32,)   learned embedding for candidate's super-region
p_router_fine[b,c]             → (1,)    router's probability mass on this candidate's region
p_mem_fine[b,c]                → (1,)    memory system's probability mass on this candidate's region
router_margin[b]               → (1,)    how confident the router was (position-level)
mem_margin[b]                  → (1,)    how confident memory was (position-level)
is_router[b,c]                 → (1,)    1 if candidate's region is in router top-K
is_mem[b,c]                    → (1,)    1 if candidate's region is in memory top-K
────────────────────────────────────────────────────────────────────────────────
total                          → (134,)
```

The feature includes both position-level context (h_prime, margins) and
candidate-level context (what region this specific token lives in, how
much probability mass the router puts there). This lets the model ask
"given where the backbone is pointing, should token X score higher or lower
than its raw dot product suggests?"

### Forward Pass

```python
feat    = cat([h_proj, tok_proj, fine_emb, super_emb,
               p_router, p_mem, r_margin, m_margin,
               is_router, is_mem])           # (B, C, 134)

delta   = mlp(feat).squeeze(-1)             # (B, C) — one scalar per candidate
          * residual_scale                  # learned scalar gate

scores  = (base + delta).masked_fill(~cand_mask, -inf)   # (B, C)
```

The MLP is three linear layers:
```
Linear(134 → 256) → GELU → Linear(256 → 256) → GELU → Linear(256 → 1)
```

`residual_scale` is a single `nn.Parameter` — one number shared across all
positions and candidates. It controls how much the network's output is trusted.
At init (described below) it is set so that `delta = 0` everywhere, making
the model start as pure force-zero.

Trainable parameters: ~170k (d_region=32, d_hidden=256).

---

## Initialization — Robust Hard-Position Init

### Goal

`delta = 0` at step 0 (model = identity = force-zero), AND gradients flow
through inner MLP layers from step 1 onward.

### Why not `residual_scale = 0.0` (the global refiner approach)?

With `residual_scale = Parameter(0.0)`:

```
delta = mlp(feat) * 0.0 = 0   ✓ identity at init
```

But consider the gradient of the loss w.r.t. an inner MLP weight `W_inner`:

```
d(loss)/d(W_inner) = d(loss)/d(delta) * d(delta)/d(mlp) * d(mlp)/d(W_inner)
                   = d(loss)/d(delta) * residual_scale  * d(mlp)/d(W_inner)
                   = d(loss)/d(delta) * 0.0             * d(mlp)/d(W_inner)
                   = 0
```

`residual_scale = 0` kills the gradient to every weight in the MLP. With
sparse filtered batches (e.g., `tight_boundary` is only ~5-8% of positions),
you may go hundreds of steps before the signal is strong enough to make
`residual_scale` meaningfully nonzero, during which the MLP does nothing.

### What we do instead

```python
model.residual_scale.data.fill_(1.0)        # gate is open
nn.init.zeros_(model.mlp[-1].weight)        # last Linear: weight = 0
nn.init.zeros_(model.mlp[-1].bias)          # last Linear: bias = 0
```

Now `mlp(feat)` = `Linear(0)(anything)` = 0, so `delta = 1.0 * 0 = 0`. Identity holds.

But the gradient to inner weights is now:

```
d(loss)/d(W_inner) = d(loss)/d(delta) * 1.0 * d(mlp)/d(W_inner)   ≠ 0
```

The inner layers get real gradients from step 1. The last layer weight
leaves zero and gets a gradient too:

```
d(loss)/d(W_last) = residual_scale * d(loss)/d(delta) * prev_layer_output
                  = 1.0            * nonzero           * nonzero
                  ≠ 0
```

So from step 1 onward, the last layer immediately starts learning.
`residual_scale` itself starts getting gradient from step 2 (once
`mlp(feat)` is nonzero). This is meaningfully faster for sparse filters.

**Verified by**: `check_init_identity` runs on the first 64 val examples
and asserts `max|delta| < 1e-5` and `NLL_diff < 1e-4` before training starts.
Training aborts if this fails.

| Parameter | Value | Effect |
|-----------|-------|--------|
| `residual_scale` | `1.0` | Gate open — gradients flow through immediately |
| `mlp[-1].weight` | `zeros` | Output = 0 → delta = 0 at step 0 |
| `mlp[-1].bias` | `zeros` | Bias also zero |
| Inner MLP layers | Kaiming (default) | Nonzero activations → real gradients from step 1 |

---

## Filter System

All filters read precomputed shard metadata — no model call, no forward pass.
`compute_filter_mask(shard, filter_name)` returns a `(N,)` bool tensor.

### Simple filters

```python
# boundary: positions in region-split level 2 or 3
(split == 2) | (split == 3)

# tight_boundary: only the tightest split level
split == 3

# type_A: positions classified as type A by the offline classifier
type_arr == 1
```

### Router top-K miss

```python
gold_reg = shard["gold_region"].long()               # (N,)
topk_reg = shard["router_topk_reg"].long()[:, :8]    # (N, 8)

# For each position: is gold_region in router's top 8?
gold_exp = gold_reg.unsqueeze(1).expand_as(topk_reg) # (N, 8) — broadcast
in_top8  = ((topk_reg == gold_exp) & (topk_reg >= 0)).any(dim=1)  # (N,) bool

return ~in_top8   # True where gold region is NOT in router top 8
```

These are positions where the router sent probability mass to the wrong regions.
The correct token's region is not among the 8 most likely — the backbone's
region routing was outright wrong.

### Union filters

```python
# hard_union: positions that are hard for any reason
boundary   = (split == 2) | (split == 3)
type_ab    = (type_arr == 1) | (type_arr == 2)
low_margin = router_margin < margin_thresh   # router was uncertain
return boundary | type_ab | low_margin | top8_miss
```

### Filter rates (approximate, from filter_audit.json)

| Filter | Approx. % of positions |
|--------|------------------------|
| `tight_boundary` | 5–10% |
| `type_A` | 10–20% |
| `boundary` | 20–30% |
| `router_top8_miss` | 10–20% |
| `type_A_or_B_resolvable` | 15–30% |
| `hard_union_small` | 20–35% |
| `hard_union_medium` | 30–45% |
| `hard_union` | 35–55% |

Exact rates are printed at startup and saved to `filter_audit.json`.

---

## Training

```
FilteredShardStreamDataset(train_dir, filter=train_filter)
  — iterates all shards, skips positions where filter_mask=False
  — reshuffles shard order and per-shard positions each epoch
  ↓ batch_size=64  (of filtered positions only)
  ↓
compute_loss(model, batch, device, lambda_kl=0.01, lambda_delta=1e-4)
```

### Loss

```python
scores, base_scores = model(...)          # (n_cov, C) for covered positions

loss_ce = cross_entropy(scores, gold_idx)

# KL toward base: prevents the model from drifting far from base distribution
p_pred  = softmax(scores)
p_base  = softmax(base_scores.detach())
loss_kl = (p_pred * (log(p_pred) - log(p_base))).sum(-1).mean()

# Delta reg: L2 penalty on residual over all valid candidates
delta      = scores[cmask] - base_scores[cmask].detach()
loss_delta = delta.pow(2).mean()

total = loss_ce + 0.01 * loss_kl + 1e-4 * loss_delta
```

The KL and delta terms prevent collapse: without them the model can score the
gold token infinitely high and everything else at `-inf`, which minimizes CE
but destroys the underlying probability structure needed for decoding.

### Optimizer

```
AdamW(lr=3e-4, weight_decay=1e-2)
CosineAnnealingLR(T_max=10000, eta_min=3e-5)
GradScaler("cuda") + autocast("cuda")
clip_grad_norm = 1.0
```

---

## Eval System

### Gated Global Eval (primary metric)

One pass over ALL 239,362 val positions with the gate applied:

```python
gate = compute_filter_mask(shard, gate_filter_name)   # (N,) bool, CPU
...
gate_3d      = gate.unsqueeze(1).expand(-1, C)         # (B, C)
gated_scores = torch.where(gate_3d, model_scores, base_scores)

# NLL over covered positions using gated_scores
lp  = F.log_softmax(gated_scores[covered], dim=-1)
nll = -lp[arange, gold_cand_idx[covered]].mean()
```

The model is only called when `gate.any()` is True for a batch. Outside-gate
positions use `base_scores` exactly — no model computation.

**Why `gated_covered_nll` is the right primary metric:**
`inside_gate_model_nll` would be incomparable across filters — `tight_boundary`
(5% gate) and `hard_union` (50% gate) evaluate completely different subsets at
different difficulty levels. `gated_covered_nll` always covers all 239k
positions the same way. Any gain comes only from what happens inside the gate,
so `delta_vs_baseline = 3.378606 - gated_covered_nll` is directly comparable
across all 6 runs.

Additional metrics reported per eval step:

```
inside_gate_model_nll   — model NLL restricted to gated positions
inside_gate_base_nll    — base NLL on same positions  (gap = what the model is correcting)
outside_gate_nll        — base NLL on non-gated positions (sanity: should be stable)
gate_rate               — fraction of val positions inside the gate
covered_gate_rate       — fraction of covered val positions inside the gate
```

### Dataset Integrity

Every eval (gated global and local subset) computes the same fingerprint as
the official baseline:

```python
fp_data = {
    "num_shards": len(paths), "total_n": total_n, "total_cov": total_cov,
    "sum_cand_counts": ..., "sum_gold_idx_cov": ..., "sum_gold_tok": ...,
}
fingerprint = sha256(json.dumps(fp_data, sort_keys=True))[:16]
```

None of these values depend on model scores — they're counts derived from the
shard tensors. So the fingerprint is invariant to what the model predicts.
If it ever changes, something is wrong with the shard iteration (missing shards,
different order-dependent accumulation, etc.). Training aborts on mismatch
(`--fail_on_baseline_mismatch`).

Required: `fingerprint=f57cabcdc46d69ce`, `num_examples=239,362`,
`num_covered=227,017`, `coverage=0.948425`.

### Local Subset Eval

A second full val pass (separate from gated eval) that evaluates model and
base scores across all 13 named subsets in one sweep. For each shard batch,
all 13 filter masks are computed, and per-subset NLL + acc@k accumulate:

```python
for sub in EVAL_SUBSETS:
    cov_sub = covered & subset_mask[sub]   # covered positions in this subset

    # Model NLL on this subset
    m_lp = F.log_softmax(model_scores[cov_sub], dim=-1)
    model_ce += -m_lp[arange, gold_idx[cov_sub]].sum()

    # acc@1 and acc@5
    top5 = model_scores[cov_sub].topk(5, dim=-1).indices   # (n_cov_sub, 5)
    acc1 += (top5[:, :1] == gold_idx[cov_sub].unsqueeze(1)).any(1).sum()
    acc5 += (top5         == gold_idx[cov_sub].unsqueeze(1)).any(1).sum()
```

Written to `local_subset_eval.csv` every eval step. The interesting question:
does a `boundary`-trained model also improve `type_A` positions it never
trained on? Check the `type_A` row in the boundary run's CSV.

---

## Experiment Runs (6 filters, Variant C)

| Run | train_filter | gate_filter | Output dir |
|-----|-------------|-------------|------------|
| 1 | `boundary` | `boundary` | `variant_C_boundary` |
| 2 | `tight_boundary` | `tight_boundary` | `variant_C_tight_boundary` |
| 3 | `type_A` | `type_A` | `variant_C_type_A` |
| 4 | `type_A_or_B_resolvable` | `type_A_or_B_resolvable` | `variant_C_type_A_or_B` |
| 5 | `router_top8_miss` | `router_top8_miss` | `variant_C_router_top8_miss` |
| 6 | `hard_union` | `hard_union` | `variant_C_hard_union` |

All under `runs/path_refiner_hard/`.

---

## Output Files Per Run

```
runs/path_refiner_hard/variant_C_<filter>/
  best_refiner.pt          — checkpoint at lowest gated_covered_nll
  last_refiner.pt          — checkpoint at final step
  filter_audit.json        — how many positions each filter selects (train + val)
  train_log.csv            — per-eval-step: gated NLL, inside/outside NLL,
                             gate_rate, fingerprint, num_examples
  local_subset_eval.csv    — per-eval-step × 13 subsets: NLL + acc@1/5
  best_metrics.json        — all metrics at best checkpoint (primary comparison file)
```

`best_metrics.json` structure:
```json
{
  "eval_mode":             "gated_global_val",
  "step":                  ...,
  "train_filter":          "...",
  "gate_filter":           "...",
  "official_baseline_nll": 3.378606,
  "gated_covered_nll":     ...,
  "delta_vs_baseline":     ...,
  "inside_gate_model_nll": ...,
  "inside_gate_base_nll":  ...,
  "outside_gate_nll":      ...,
  "gate_rate":             ...,
  "covered_gate_rate":     ...,
  "coverage":              0.948425,
  "fingerprint":           "f57cabcdc46d69ce",
  "num_examples":          239362,
  "num_covered":           227017,
  "local_subsets":         { "type_A": {...}, "boundary": {...}, ... }
}
```

---

## Comparison Protocol

Primary metric: `gated_covered_nll`. Lower is better.
`delta_vs_baseline = official_baseline_nll - gated_covered_nll`; positive = improvement.

Global refiner baseline: `delta_vs_baseline = +0.001284`.
Any hard-filter run exceeding this shows the targeted approach is beneficial.

**Do not compare `inside_gate_model_nll` across runs** — different gates
cover different (and differently-difficult) position subsets. Only
`gated_covered_nll` is apples-to-apples.

---

## Relationship to Global Refiner

| Attribute | Global refiner | Hard refiner |
|-----------|---------------|--------------|
| Script | `train_clean_path_refiner.py` | `train_hard_position_refiner.py` |
| Model class | `RicherMLPRefiner` | same (imported) |
| Train positions | all 239k | filtered subset only |
| Val metric | canonical full-val NLL | gated global NLL |
| `residual_scale` init | `0.0` | `1.0` |
| Final layer init | Kaiming (after fix) | zeros |
| delta at step 0 | 0 (via `scale=0`) | 0 (via zero final layer) |
| Inner-layer grad at step 0 | **0** (blocked by scale=0) | **nonzero** (flows through) |
| Best achieved delta | +0.001284 nats | TBD |
