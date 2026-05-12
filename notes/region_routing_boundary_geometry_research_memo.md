# Region Routing / Boundary Geometry Research Memo

## Core summary

The work evolved from a simple routing hypothesis into a broader theory about how transformers may internally organize prediction, uncertainty, and computation.

The original question was whether next-token prediction is really a flat 50k-way classification problem. The emerging hypothesis is more nuanced:

```text
A transformer hidden state may first represent a probability distribution over predictive neighborhoods.
Low-margin states correspond to unresolved competition among overlapping feature/manifold regions.
Contextual computation progressively sharpens this distribution, suppressing incompatible regions and amplifying coherent ones, until exact token logits emerge.
```

In this framing:

```text
regions = predictive neighborhoods
margins = uncertainty geometry
top-k regions = local manifold neighborhood
boundary tokens = unresolved contextual competition
refinement = contextual sharpening / compute allocation
```

---

## 1. Original hypothesis: tokens live in predictive regions

The original intuition was that transformers might not directly select one token out of 50k possibilities. Instead, they may internally organize prediction into coarse regions or predictive neighborhoods.

The early idea was not a semantic hierarchy like:

```text
animal → dog → golden retriever
```

It was a prediction hierarchy:

```text
contextual hidden state
→ group of likely next-token candidates
→ finer candidate group
→ token identity
```

The hypothesis was:

```text
There exist regions of tokens/features that interfere mostly within themselves,
and a transformer hidden state may first locate a region before resolving exact token identity.
```

This was connected to superposition because if features are packed into overlapping subspaces, then exact token identity may emerge late, while coarse predictive structure may emerge earlier.

---

## 2. How the regions were clustered

The regions were not made by clustering raw token embeddings.

A naive pipeline would be:

```text
token embedding vectors
→ cosine similarity
→ k-means
```

The actual idea was closer to:

```text
model behavior / hidden-state interaction / co-activation
→ token graph
→ community detection
→ regions
```

A representative pipeline:

```text
1. Run a trained transformer over corpus.
2. At each position, record top predicted next tokens.
3. Build a co-occurrence graph:
   tokens are nodes,
   edges mean tokens co-occur in the same predictive candidate set.
4. Normalize the graph.
5. Cluster graph into regions.
6. Save token_to_region.json.
```

Earlier GPT-2 XL experiments also involved:
- hidden states from late layers like Layer 47
- co-activation graphs
- gradient/interference graphs
- Louvain clustering
- spectral comparisons

The important distinction:

```text
regions are based on computational/predictive interaction structure,
not just static semantic similarity.
```

So the regions are closer to:

```text
tokens that share predictive/interference behavior
```

rather than:

```text
tokens that merely mean similar things.
```

---

## 3. First major result: regions are predictable

After creating token-to-region maps, routers/probes were trained to predict regions from hidden states:

```text
hidden state h → region distribution p(region | context)
```

The major finding:

```text
Predicting the coarse region is much easier than predicting the exact token.
```

Observed patterns included:
- coarse acc@1 around 0.58 in several 128-region setups
- strong top-k region recall
- random baselines much worse
- earlier 72-class setups showing top-4 recall around ~0.82 compared to random around ~0.05

Interpretation:

```text
The hidden state contains strong information about the correct predictive neighborhood,
even when exact token identity is not yet resolved.
```

This was one of the first pieces of evidence for:

```text
coarse predictive localization
→ local refinement
→ token collapse
```

rather than flat token classification.

---

## 4. Oracle routing and why it mattered

Oracle routing means giving the model the true/gold region of the next token rather than using the learned predicted region.

Oracle results were much better than learned routing.

This showed:

```text
If the model had perfect region information, the task becomes much easier.
```

But learned routing did not approach oracle performance.

This means the issue is not:

```text
regions are useless
```

It is more like:

```text
region information is highly valuable,
but current routers/architectures do not exploit or infer it well enough.
```

This created the central problem:

```text
The structure exists.
The question is how to use it.
```

---

## 5. Output/logit routing versus representation routing

Two ways of using region information were explored.

### A. Output/logit-level routing

This means:

```text
compute transformer hidden state normally
compute logits
then bias/mask/prior logits using region information
```

Examples:
- region logit priors
- candidate restriction
- hard top-k region masking
- hierarchical-softmax-like output routing

This was brittle because it forced the answer late, after the model had already computed its representation. If the region guess was wrong or fuzzy, hard output restriction destroyed useful uncertainty.

### B. Representation-level routing

This means:

```text
region distribution modifies hidden state before final prediction
```

A representative model:

```python
p_region = softmax(router(h))
r = p_region @ region_embeddings
region_feat = region_proj(region_mlp(r))
h_prime = h + alpha * region_feat
logits = lm_head(h_prime)
```

This worked better conceptually and empirically.

Key lesson:

```text
Regions are useful as computational context,
not as hard output filters.
```

So routing should influence representations, not merely final logits.

---

## 6. Capacity/diversity experiments

To test whether routers collapsed onto a few regions, several anti-collapse mechanisms were added:
- EMA region usage tracking
- capacity penalties
- diversity loss
- temperature annealing
- participation ratio metrics
- usage entropy
- Gini / active region counts

The idea:

```text
If regions collapse, rare regions cannot contribute.
Maybe fixing usage improves routing.
```

Results showed:
- usage entropy increased
- participation ratio improved
- collapse reduced
- LM performance did not significantly beat the reference

Lesson:

```text
Router collapse exists,
but reducing collapse alone does not solve prediction.
```

Also, forced uniformity may be wrong because language is naturally non-uniform. The right goal is useful diversity, not equal usage.

---

## 7. Boundary analysis: the major turning point

The key diagnostic became:

```python
margin = p(top1_region) - p(top2_region)
```

High margin:

```text
model strongly prefers one region
```

Low margin:

```text
model sees multiple plausible regions
```

This created:
- core tokens = high margin
- boundary tokens = low margin

Boundary analysis measured:
- margin bins
- entropy
- boundary/core NLL
- oracle gaps
- top-k accuracy
- token-context variance
- repeated-token behavior

Important findings:
- low-margin positions had much higher NLL
- high-margin positions were much easier
- entropy was higher near boundaries
- acc@1 was low near boundaries
- acc@4/top-k remained much stronger than top-1
- repeated tokens could appear in many different regions
- boundary tokens were systematically harder than core tokens

Key conclusion:

```text
Routing failures are not uniform.
They concentrate around fuzzy latent boundaries.
```

The research question changed from:

```text
Can we route better globally?
```

to:

```text
What computation is needed at ambiguous boundary states?
```

---

## 8. Contextual variance: tokens do not have fixed regions

A major result was that the same token could appear across many unique regions depending on context, often with low average margin.

Interpretation:

```text
token identity alone does not determine region.
context determines latent positioning.
```

The naive model:

```text
token → fixed region
```

was replaced by:

```text
(context, hidden state) → distribution over regions
```

Example intuition:

```text
apple
```

can be:
- fruit-like
- company-like
- product-like
- name-like

depending on context.

Therefore the region distribution is a property of the contextual prediction state, not just the token.

---

## 9. Connection to superposition

Superposition suggests that models represent many features in overlapping directions or subspaces because representational capacity is limited.

The region results fit this view.

Boundary states may correspond to:

```text
multiple partially-compatible feature bundles coexisting simultaneously.
```

So instead of clean separated regions:

```text
A | B | C
```

the geometry may look like:

```text
region cores + fuzzy overlapping transitions
```

This explains:
- low margins
- top-k strength
- top-1 brittleness
- contextual drift
- same token across many regions
- hard-routing failures

Superposition-informed interpretation:

```text
A low-margin state is not necessarily confusion.
It may be structured superposition of multiple plausible predictive manifolds.
```

---

## 10. Why top-k became the right signal

Repeated observation:

```text
top-1 is weak near boundaries,
but top-k remains strong.
```

Interpretation:

```text
The model knows the neighborhood, but not exact placement.
```

This is different from:

```text
The router is clueless.
```

It suggests:

```text
coarse localization is easy,
fine resolution is hard.
```

So the region module should not be treated as:

```text
argmax region
```

but as:

```text
a belief distribution over predictive neighborhoods
```

Old router role:

```text
pick region
```

New router role:

```text
estimate where the gold token is likely to live
```

Old refiner role:

```text
predict token from region
```

New refiner role:

```text
use the region distribution to focus/sharpen computation
```

---

## 11. Boundary gate calibration

Boundary-aware alpha scaling used:

```python
gate = sigmoid((tau - margin) / boundary_temp)
alpha_dynamic = alpha_core*(1-gate) + alpha_boundary*gate
```

Initial settings:

```text
tau=0.10
temp=0.05
boundary_frac≈0.8+
```

This overfired: almost everything was treated as boundary.

After a gate sweep:

```text
tau=0.03
temp=0.01
boundary_frac≈0.25–0.30
```

This produced a cleaner picture:

```text
~25–30% boundary
~70–75% core
```

Interpretation:

```text
language is not mostly boundary,
but boundary structure is real and substantial.
```

---

## 12. Single-path boundary scaling did not solve the problem

With calibrated gates, boundary tokens received stronger region conditioning.

This tested:

```text
Do hard tokens simply need stronger region signal?
```

It did not beat the reference.

Conclusion:

```text
Boundary tokens do not just need stronger conditioning.
They need different computation.
```

---

## 13. Multihypothesis branching failed

Next idea:

```text
If top-k regions are meaningful,
preserve top-k hypotheses separately.
```

Architecture idea:

```text
low-margin token
→ top-k region branches
→ separate h_i states
→ refiner
→ weighted merge
```

This was similar to internal beam search.

With calibrated gates, multihypothesis branching still underperformed.

Conclusion:

```text
Static top-k region branch vectors do not solve boundary ambiguity.
```

This showed that preserving multiple hypotheses as static vectors is insufficient.

---

## 14. Branch attention also failed

Next idea:

```text
Maybe branches need to communicate.
```

Branch attention tested:

```text
top-k branch vectors
→ attention over branch dimension
→ merge
```

The first implementation had a scale/gradient bug, but debugging fixed that.

Final concept comparison still showed:
- branch identity worse than boundary single-path
- branch attention worse than branch identity

Conclusion:

```text
static region branches are not good computational primitives.
```

More carefully:

```text
region distribution is useful as a signal,
but branch vectors alone do not contain the missing information needed to resolve ambiguity.
```

---

## 15. Sequence refinement: context beats static branch tricks

Sequence refinement added an extra causal transformer block over the sequence.

This tested:

```text
extra contextual reasoning
```

rather than:

```text
latent branch manipulation
```

Results showed:
- seqrefine beat branch attention
- seqrefine beat branch identity
- contextual compute helped more than static branch tricks

Interpretation:

```text
boundary ambiguity is more contextual than purely geometric.
```

Boundary tokens likely need additional contextual reasoning, not alternate region embeddings.

---

## 16. Random seqrefine nearly matched real seqrefine

The twist:

```text
random seqrefine nearly matched real seqrefine
```

Interpretation:

```text
extra compute helps,
but current region geometry is not yet strongly controlling useful compute allocation.
```

Still, real regions produced:
- lower boundary fraction
- sharper margins
- cleaner uncertainty geometry
- less destabilized hidden updates

Refined conclusion:

```text
real regions are good descriptive uncertainty geometry,
but current architectures are not yet exploiting them optimally.
```

---

## 17. Adaptive depth: current best architecture direction

The next hypothesis:

```text
easy/core tokens: normal depth
boundary tokens: extra transformer depth
```

Previous seqrefine:

```text
run extra block over whole sequence
then gate output
```

Adaptive depth:

```text
use boundary geometry to decide which positions receive extra updates
```

This tests:

```text
Can region-boundary geometry serve as a compute allocation signal?
```

This is more novel than just adding a block.

The key unresolved question:

```text
Can real region geometry tell us where to spend compute better than random/entropy/blanket extra compute?
```

---

## 18. Interpretability pivot: layer-margin analysis

Before more custom architecture, the underlying mechanistic hypothesis should be tested in a standard transformer.

Hypothesis:

```text
Transformers progressively resolve manifold uncertainty across layers.
```

Expected pattern:

```text
early layers:
    high entropy
    low margin
    weak top1
    maybe gold region enters top-k

middle layers:
    top-k neighborhoods stabilize

late layers:
    margin sharpens
    entropy decreases
    gold region rank improves
```

Most important expected result:

```text
gold region enters top-k early,
but top-1/margin sharpens later.
```

If true, it supports:

```text
transformers first localize a predictive neighborhood,
then later resolve ambiguity inside that neighborhood.
```

This is a pure interpretability experiment and does not depend on a custom architecture succeeding.

---

## 19. Current refined theory

### Step 1: Coarse localization

The hidden state forms:

```text
p(region | context)
```

This tells us where the gold token is likely to live.

### Step 2: Uncertainty geometry

Margins and entropy describe how resolved the prediction is.

High margin:

```text
one predictive manifold dominates
```

Low margin:

```text
multiple manifolds/features are competing
```

### Step 3: Contextual sharpening

Layers/context suppress incompatible manifolds and strengthen coherent ones.

Example:

```text
apple
```

alone may be low-margin.

But:

```text
Tim Cook unveiled Apple...
```

pushes the hidden state toward a tech/company manifold and increases margin.

### Step 4: Token collapse

Only late does the model resolve exact token identity.

Overall process:

```text
context
→ fuzzy predictive neighborhood
→ progressive margin sharpening
→ exact token logits
```

---

## 20. Refiner’s job

The refiner is not simply:

```text
region → token
```

A better abstraction:

```text
uncertain predictive neighborhood → contextually sharpened hidden state
```

The refiner’s job may be to:
- suppress wrong neighboring regions
- amplify coherent region features
- allocate compute to ambiguous states
- improve local manifold placement
- increase margin when context resolves ambiguity
- preserve uncertainty when ambiguity is genuine

So the refiner is doing something like:

```text
margin sharpening / uncertainty resolution
```

not just token prediction.

---

## 21. Interpretability value

If layer-margin analysis works, the region system becomes an interpretability lens.

For every token position, you can inspect:

```text
p(region | context)
top-k regions
margin
entropy
gold region rank
boundary/core status
region trajectory across layers
```

This exposes:

```text
what predictive neighborhood the model thinks the next token belongs to
before final token selection
```

Possible analyses:
- when ambiguity appears
- when ambiguity resolves
- which contexts sharpen which tokens
- which tokens remain polysemantic
- which regions compete
- how region belief evolves layer by layer

---

## 22. Current empirical claims

### Strongly supported

```text
Region structure exists in predictive/co-activation space.
```

```text
Region prediction is easier than exact token prediction.
```

```text
Boundary/core distinction is real.
```

```text
Low-margin boundary positions are much harder than high-margin core positions.
```

```text
Context changes region assignment/distribution.
```

```text
Top-k region information is more meaningful than top-1 hard routing.
```

```text
Static top-k branch manipulation does not solve boundary ambiguity.
```

```text
Contextual sequence refinement helps more than static branch interaction.
```

### Moderately supported

```text
Regions act like predictive latent neighborhoods.
```

```text
Margins reflect uncertainty geometry.
```

```text
Transformers may localize coarse neighborhoods before exact token identity.
```

```text
Boundary tokens require extra contextual reasoning.
```

### Still unproven / next tests

```text
Margins sharpen progressively across layers in standard transformers.
```

```text
Gold region enters top-k early and top-1 sharpens later.
```

```text
Region-boundary geometry can usefully allocate sparse compute.
```

```text
Real regions beat entropy/random signals for adaptive depth.
```

```text
Region-guided top-k pruning can reduce compute without hurting loss.
```

---

## 23. Why negative results are valuable

The failed experiments ruled out several naive mechanisms:

```text
Just make routing stronger.
```

```text
Just balance regions more.
```

```text
Just preserve top-k branches.
```

```text
Just let static branches attend to each other.
```

```text
Just add global refinement and assume region geometry matters.
```

Each failure narrowed the theory.

Current refined conclusion:

```text
Regions expose useful uncertainty geometry,
but exploiting that geometry probably requires contextual compute allocation,
not static region-vector manipulation.
```

---

## 24. Clean research narrative

A concise paper-style narrative:

```text
We construct predictive token regions from model-induced co-activation/interference structure.
We show that transformer hidden states predict these regions much more easily than exact tokens.
We introduce margin-based boundary analysis and find that low-margin region-boundary states are systematically harder.
We show that the same token can occupy different predictive neighborhoods under different contexts, suggesting dynamic contextual manifold positioning.
We test several mechanisms for exploiting this structure.
Static region conditioning helps slightly, but hard/logit-level routing and static branch manipulation fail.
Sequence-level contextual refinement outperforms branch manipulation, suggesting boundary uncertainty is resolved through contextual computation.
This motivates viewing regions as probabilistic predictive neighborhoods and margins as uncertainty geometry for adaptive compute allocation.
```

---

## 25. Core insight

The deepest idea:

```text
A transformer hidden state may not directly represent “the next token.”
It may first represent a probability distribution over predictive neighborhoods.
Low-margin states correspond to unresolved competition among overlapping feature/manifold regions.
Contextual computation progressively sharpens this distribution, suppressing incompatible regions and amplifying coherent ones, until exact token logits emerge.
```

The next best proof is standard-transformer layer-margin analysis.
