    #!/bin/bash
    #SBATCH --job-name=local_detail_2c
    #SBATCH --partition=gpuGeneral
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64GB
    #SBATCH --time=06:00:00
    #SBATCH --gres=gpu:l40s:1
    #SBATCH --output=logs/%x-%j.out
    #SBATCH --error=logs/%x-%j.err
    #
    # Local Detail Selector V1C — Decoupled Selector + Edit Gate
    #
    # Stage 2C: Fixes Stage 2B's NO_OP collapse (apply_rate≈0.0017).
    # Key change: split joint action softmax into:
    #   - Selector (CE, correctable rows only): which candidate?
    #   - Gate (BCE, correctable + base_correct): should we edit?
    #
    # Teacher forcing:
    #   correctable  → gate_cand_idx = gold_pool_idx
    #   base_correct → gate_cand_idx = argmax(sel_scores.detach())
    #
    # Gate labels for correctable: surgery simulation → gate_label=1 if gold becomes top1
    # No-harm gate margin: softplus(gate_margin + edit_logit) on base_correct
    #
    # Stage 1 ref:  ctg≈0.040, caw≈0.062, fv_gain≈-0.0199
    # Stage 2 ref:  ctg≈0.054–0.063, caw≈0.079–0.094, fv_gain≈-0.0173
    # Stage 2B ref: apply_rate≈0.0017 (collapsed)
    # V3 oracle:    target_ctg=0.2407, fv_gain=+0.1584
    # ─────────────────────────────────────────────────────────────────────────────

    set -euo pipefail

    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false

    cd ~/ondemand/upload_me/RegionTokenizer
    export PYTHONPATH=$PWD
    mkdir -p logs

    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate base
    fi

    RUN_NAME="cand_xattn_editgate_v1c"
    OUT_ROOT="runs/local_detail_selector_v1c"
    RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

    SMALL_CKPT="runs/repr_region_retrieval_proxy_lam0p10/checkpoint_latest.pt"
    TRAIN_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/train"
    VAL_DIR="runs/live_full_pipeline_rebuild_limited500k_logitfix/01_live_dataset_patched/val"
    TOKEN_TO_REGION="runs/region_maps_128/token_to_region.json"
    SUPER_MAP="runs/hard_memory_predictive_hierarchy/region_to_superregion_K24.json"

    echo "========================================================"
    echo " Local Detail Selector V1C — Decoupled Selector + Gate"
    echo " run_name: ${RUN_NAME}"
    echo " $(date)"
    echo "========================================================"
    echo ""
    echo "[preflight] Checking required inputs..."

    if [ ! -f "$SMALL_CKPT" ]; then
        echo "ERROR: backbone checkpoint not found: $SMALL_CKPT"
        exit 1
    fi
    echo "  [OK] $SMALL_CKPT"

    for D in "$TRAIN_DIR" "$VAL_DIR"; do
        if [ ! -d "$D" ]; then
            echo "ERROR: shard dir not found: $D"
            exit 1
        fi
        N=$(find "$D" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
        if [ "$N" -eq 0 ]; then
            echo "ERROR: no shard_*.pt in $D"
            exit 1
        fi
        echo "  [OK] $D  ($N shards)"
    done

    if [ ! -f "$TOKEN_TO_REGION" ]; then
        echo "ERROR: token_to_region not found: $TOKEN_TO_REGION"
        exit 1
    fi
    echo "  [OK] $TOKEN_TO_REGION"

    SUPER_ARG=""
    if [ -f "$SUPER_MAP" ]; then
        SUPER_ARG="--super_map $SUPER_MAP"
        echo "  [OK] $SUPER_MAP  (superregion enabled)"
    else
        echo "  [WARN] super_map not found: $SUPER_MAP — superregion disabled"
    fi

    # Check input_ids in at least one train and val shard
    echo "[preflight] Checking input_ids in shards..."
    TRAIN_SHARD=$(find "$TRAIN_DIR" -maxdepth 1 -name 'shard_*.pt' | head -1)
    VAL_SHARD=$(find "$VAL_DIR"   -maxdepth 1 -name 'shard_*.pt' | head -1)

    for SHARD in "$TRAIN_SHARD" "$VAL_SHARD"; do
        python -c "
    import torch, sys
    sh = torch.load('$SHARD', map_location='cpu', weights_only=False)
    if 'input_ids' not in sh:
        print('ERROR: input_ids missing from shard: $SHARD')
        print('  keys:', list(sh.keys()))
        sys.exit(1)
    print('  [OK] input_ids found in $SHARD')
    " || exit 1
    done

    echo ""
    echo "[preflight] All required inputs present."
    echo ""
    mkdir -p "$OUT_ROOT"
    mkdir -p logs

    echo "[launch] Starting local detail selector V1C training..."
    echo ""

    # NOTE: --selector_margin_thresholds uses = form because value starts with -999
    python scripts/train_local_detail_selector_v1c.py \
        --small_ckpt                  "$SMALL_CKPT"          \
        --train_dir                   "$TRAIN_DIR"            \
        --val_dir                     "$VAL_DIR"              \
        --token_to_region             "$TOKEN_TO_REGION"      \
        $SUPER_ARG                                            \
        --output_root                 "${OUT_ROOT}"           \
        --run_name                    "${RUN_NAME}"           \
        --top_k                       256                     \
        --candidate_filter            top_rank                \
        --candidate_pool_size         32                      \
        --memory_len                  128                     \
        --resolver_dim                256                     \
        --hidden_dim                  512                     \
        --attention_heads             4                       \
        --dropout                     0.1                     \
        --region_emb_dim              64                      \
        --super_emb_dim               32                      \
        --batch_size                  128                     \
        --correctable_fraction        0.35                    \
        --lr                          1e-4                    \
        --steps                       5000                    \
        --eval_every                  500                     \
        --eval_batch_size             256                     \
        --grad_clip                   1.0                     \
        --lambda_selector             1.0                     \
        --lambda_gate                 1.0                     \
        --gate_pos_weight             3.0                     \
        --lambda_noharm_gate          1.0                     \
        --gate_margin                 1.0                     \
        --lambda_selector_margin      0.25                    \
        --selector_margin             1.0                     \
        --margin_delta                1.0                     \
        --gate_thresholds             0.3,0.4,0.5,0.6,0.7,0.8 \
        --selector_margin_thresholds="-999,0.0,0.5,1.0,1.5,2.0" \
        --uncovered_policy            ignore                  \
        --max_train_rows              500000                  \
        --eval_full_vocab                                     \
        --amp                                                 \
        --seed                        42

    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo ""
        echo "ERROR: training exited with code $EXIT"
        exit $EXIT
    fi

    echo ""
    echo "========================================================"
    echo " Local Detail Selector V1C complete. $(date)"
    echo "========================================================"
    echo ""
    echo "Key outputs:"
    echo "  ${RUN_DIR}/report.md"
    echo "  ${RUN_DIR}/final_metrics.json"
    echo "  ${RUN_DIR}/threshold_grid.csv"
    echo "  ${RUN_DIR}/best_selector.pt"
    echo "  ${RUN_DIR}/examples_changed_to_gold.md"
    echo "  ${RUN_DIR}/examples_attention_debug.md"
    echo ""

    if [ ! -f "${RUN_DIR}/final_metrics.json" ]; then
        echo "ERROR: final_metrics.json not found at ${RUN_DIR}/final_metrics.json"
        exit 1
    fi
    if [ ! -f "${RUN_DIR}/report.md" ]; then
        echo "ERROR: report.md not found at ${RUN_DIR}/report.md"
        exit 1
    fi
    if [ ! -f "${RUN_DIR}/threshold_grid.csv" ]; then
        echo "ERROR: threshold_grid.csv not found at ${RUN_DIR}/threshold_grid.csv"
        exit 1
    fi

    python -c "
    import json, os, math
    rd = '${RUN_DIR}'
    s  = json.load(open(os.path.join(rd, 'final_metrics.json')))

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)): return 'nan'
        if isinstance(v, float): return f'{v:.4f}'
        return str(v)

    print('=== Local Detail Selector V1C — Final Results ===')
    print()

    am = s.get('argmax_metrics', {})
    bc = s.get('best_conservative', {})
    bn = s.get('best_net_correction', {})
    fv = s.get('full_vocab', {})

    print('[argmax (gt=min, smt=no-margin)]')
    for k in ['apply_rate','noop_acc_on_base_correct','changed_to_gold_rate',
            'changed_away_rate','net_correction','noharm_changed_away',
            'applied_precision_ctg','benefit_damage_ratio','top1_acc_gain',
            'selected_gold_given_in_pool','selector_ce_val','selector_acc_val']:
        if k in am:
            print(f'  {k:52s} = {fmt(am[k])}')
    print()

    print(f'[best conservative  gt={bc.get(\"gate_threshold\",\"?\")}  smt={bc.get(\"selector_margin_threshold\",\"?\")}]')
    for k in ['apply_rate','changed_to_gold_rate','changed_away_rate',
            'net_correction','applied_precision_ctg','benefit_damage_ratio']:
        if k in bc:
            print(f'  {k:52s} = {fmt(bc[k])}')
    print()

    print(f'[best net_correction  gt={bn.get(\"gate_threshold\",\"?\")}  smt={bn.get(\"selector_margin_threshold\",\"?\")}]')
    for k in ['apply_rate','changed_to_gold_rate','changed_away_rate','net_correction']:
        if k in bn:
            print(f'  {k:52s} = {fmt(bn[k])}')
    print()

    if fv:
        print('[full_vocab]')
        for k in ['full_vocab_gain','full_vocab_base_nll','full_vocab_refined_nll',
                'full_vocab_top1_acc_base','full_vocab_top1_acc_refined']:
            if k in fv:
                print(f'  {k:52s} = {fmt(fv[k])}')
        print()

    print('[reference]')
    print('  Stage 1  ctg                                         = 0.040')
    print('  Stage 1  caw                                         = 0.062')
    print('  Stage 2  ctg                                         = 0.054-0.063')
    print('  Stage 2  caw                                         = 0.079-0.094')
    print('  Stage 2  fv_gain                                     = -0.0173')
    print('  Stage 2B apply_rate (collapsed)                      = 0.0017')
    print('  V3 oracle  target_ctg                                = 0.2407')
    print('  V3 oracle  fv_gain                                   = +0.1584')
    " 2>/dev/null || true
