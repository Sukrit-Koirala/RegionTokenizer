# Live Context + Retrieval Token Resolver Pipeline

## Why

Previous experiments (BridgeResidualAdapter, MidLayerBridge, ExplicitBridgeRefiner,
RunningBridge, TokenConfuserResolver) all plateaued near:

  full_vocab_gain_all  ≈ +0.003
  inside_gate_gain_all ≈ +0.018–0.021

The bottleneck is **information**, not architecture. Cached `h_prime`/`h_layers`
do not contain enough exact-token evidence to confidently flip confusers.

This pipeline adds:
  1. Raw context tokens (input_ids)
  2. Previous similar examples from retrieval
  3. Neighbor next-token evidence

## Stage Order

```
Stage 01  →  Stage 02  →  Stage 03  →  Stage 04  →  Stage 05  →  Stage 06
context      index         neighbors    audit         identity      train
data         build         attach        check         check         model
```

Each stage must pass before the next can run. Stage 06 checks that Stages
01, 03, 04, and 05 all have passing artifacts.

## Stages

### Stage 01 — Build Live Context Dataset

**Script:** `scripts/stage01_build_live_context_dataset.py`
**Slurm:**  `scripts/slurm_stage01_build_live_context_dataset.sh`
**Output:** `runs/live_context_pipeline/stage01_context_data/`

Reconstructs `input_ids` windows aligned 1:1 with existing candidate rows.
Uses WikiText-103 (`wikitext-103-raw-v1`) + GPT-2 tokenizer + region map
to find exactly which corpus positions were sampled. Context convention:
`input_ids = tokens[t-ctx_len:t]`, `gold_token = tokens[t]`.

**Must exist before Stage 02:**
- `stage01_context_data/alignment_report.json` (alignment_pass == true)
- `stage01_context_data/train_ctx/shard_*.pt`
- `stage01_context_data/val_ctx/shard_*.pt`

### Stage 02 — Build Train-Only Retrieval Index

**Script:** `scripts/stage02_build_live_retrieval_index.py`
**Slurm:**  `scripts/slurm_stage02_build_live_retrieval_index.sh`
**Output:** `runs/live_context_pipeline/stage02_retrieval_index/hctx_train/`

Runs frozen backbone on train `input_ids` to get `h_ctx` vectors. Builds
FAISS (or numpy) nearest-neighbor index over train-only vectors. Val rows
are never indexed.

**Must exist before Stage 03:**
- `stage02_retrieval_index/hctx_train/index_report.json` (pass == true)
- `stage02_retrieval_index/hctx_train/train_row_ids.npy`
- `stage02_retrieval_index/hctx_train/train_gold_tokens.npy`

### Stage 03 — Attach Retrieval Neighbors

**Script:** `scripts/stage03_attach_live_retrieval_neighbors.py`
**Slurm:**  `scripts/slurm_stage03_attach_live_retrieval_neighbors.sh`
**Output:** `runs/live_context_pipeline/stage03_retrieval_neighbors/`

Queries index for every train and val row. Train self-neighbor excluded.
Val only retrieves from train index. Stores neighbor gold tokens + scores.

**Must exist before Stage 04:**
- `stage03_retrieval_neighbors/retrieval_attach_report.json` (leakage_detected == false)
- `stage03_retrieval_neighbors/train_retrieval/shard_*.pt`
- `stage03_retrieval_neighbors/val_retrieval/shard_*.pt`

### Stage 04 — Audit Context + Retrieval Data

**Script:** `scripts/stage04_audit_live_context_retrieval_data.py`
**Slurm:**  `scripts/slurm_stage04_audit_live_context_retrieval_data.sh`
**Output:** `runs/live_context_pipeline/stage04_data_audit/`

Proves data is clean and useful before training. Reports gold-in-top256,
gold-in-neighbors, retrieval-added-gold, boundary-specific versions.

**Must exist before Stage 05:**
- `stage04_data_audit/retrieval_audit.json` (audit_pass == true)

### Stage 05 — Debug Model Identity

**Script:** `scripts/stage05_debug_live_context_retrieval_resolver.py`
**Slurm:**  `scripts/slurm_stage05_debug_live_context_retrieval_resolver.sh`
**Output:** `runs/live_context_pipeline/stage05_debug_identity/`

Instantiates `LiveContextRetrievalTokenResolver` (defined in Stage 06) and
verifies step-0 identity: `delta_head` zero-init → `delta=0` → `refined==base`.

**Must pass before Stage 06:**
- `stage05_debug_identity/debug_identity.json` (identity_pass == true)

### Stage 06 — Train Live Context + Retrieval Resolver

**Script:** `scripts/stage06_train_live_context_retrieval_resolver.py`
**Slurm:**  `scripts/slurm_stage06_train_live_context_retrieval_resolver.sh`
**Output:** `runs/live_context_pipeline/stage06_train_ctxret_resolver/top256_ctx256_ret32_boundary_v1/`

Trains the model. Requires all prior stages to have passing artifacts.

## Running the Pipeline

```bash
dos2unix scripts/slurm_stage01_build_live_context_dataset.sh
bash -n scripts/slurm_stage01_build_live_context_dataset.sh
sbatch scripts/slurm_stage01_build_live_context_dataset.sh
# wait for completion, check logs/

dos2unix scripts/slurm_stage02_build_live_retrieval_index.sh
bash -n scripts/slurm_stage02_build_live_retrieval_index.sh
sbatch scripts/slurm_stage02_build_live_retrieval_index.sh

dos2unix scripts/slurm_stage03_attach_live_retrieval_neighbors.sh
bash -n scripts/slurm_stage03_attach_live_retrieval_neighbors.sh
sbatch scripts/slurm_stage03_attach_live_retrieval_neighbors.sh

dos2unix scripts/slurm_stage04_audit_live_context_retrieval_data.sh
bash -n scripts/slurm_stage04_audit_live_context_retrieval_data.sh
sbatch scripts/slurm_stage04_audit_live_context_retrieval_data.sh

dos2unix scripts/slurm_stage05_debug_live_context_retrieval_resolver.sh
bash -n scripts/slurm_stage05_debug_live_context_retrieval_resolver.sh
sbatch scripts/slurm_stage05_debug_live_context_retrieval_resolver.sh

dos2unix scripts/slurm_stage06_train_live_context_retrieval_resolver.sh
bash -n scripts/slurm_stage06_train_live_context_retrieval_resolver.sh
sbatch scripts/slurm_stage06_train_live_context_retrieval_resolver.sh
```

## Failure Modes

If any stage fails:
- The stage writes its error to `pipeline_manifest.json`
- Later stages will abort with a prerequisite check error
- Do NOT continue to the next stage — fix the root cause first

## What NOT to Do

- Do not fake input_ids if corpus reconstruction fails
- Do not let val rows retrieve from val index
- Do not let train rows retrieve themselves
- Do not use gold_token to select retrieval results
- Do not compare full-vocab NLL (≈3.75) to masked-candidate NLL (≈3.38)
- Do not skip identity check before training

## Success Thresholds (Stage 06)

```
Weak:        full_vocab_gain_all > +0.0033  (beats bridge ceiling)
Meaningful:  full_vocab_gain_all > +0.005
Strong:      full_vocab_gain_all > +0.010
             inside_gate_gain_all > +0.05

Token decision success:
  changed_to_gold_rate > changed_away_rate
  top1_acc_ref > top1_acc_base
  gold_rank_improved_rate > gold_rank_worsened_rate

Information success:
  rows with neighbor_gold support improve more than rows without
```

## Do Not Overwrite

```
runs/path_refiner_clean/
runs/path_refiner_residual_interface/
runs/token_confuser_resolver/
runs/context_token_data/
runs/context_retrieval_token_resolver/
```
