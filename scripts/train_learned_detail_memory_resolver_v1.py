#!/usr/bin/env python3
"""
train_learned_detail_memory_resolver_v1.py — Learned Detail Memory Resolver V1.

Test whether exact name/entity/number/detail memorization can be learned from
memory evidence, without manual rules.

Model: LearnedDetailMemoryResolver
  Candidates = Query; doc_past + prefix context = Key/Value in cross-attention.
  Zero-init final MLP layer -> delta=0 at step 0 (identity = base logits).
  Frozen token embeddings loaded from checkpoint.
  NO rule-based token-type heuristics as model inputs.
  Gold used ONLY for CE loss and metrics; never for memory construction.

Ablation modes (evaluated each epoch + final eval):
  base_only              — delta=0 everywhere (sanity check)
  prefix_memory_only     — memory = prefix_tokens[-128:] only
  doc_past_memory_only   — memory = corpus[offset-2048:offset] only
  prefix_plus_doc_past   — memory = doc_past + prefix (main mode)
  shuffled_doc_past_control — memory = roll(doc_past,1,batch) + prefix (control)

Training:
  Only on rows where gold naturally in pool AND token_offset >= doc_memory_len.
  CE loss over candidate pool logits + learned delta.
  Save best checkpoint ONLY if model_acc > base_acc.
  If no improvement: best_metrics.json contains no_improving_checkpoint_found=true.

Outputs in --output_dir:
  config.json, train_log.csv, eval_log.csv, eval_by_slice.csv,
  ablation_results.csv, best_model.pt (conditional), best_metrics.json,
  final_metrics.json, report.md,
  examples_detail_helps.md, examples_detail_hurts.md,
  examples_attn_memory_top.md, examples_shuffled_vs_real.md,
  examples_no_doc_past.md

Usage:
  python scripts/train_learned_detail_memory_resolver_v1.py \\
    --train_dir runs/.../train --val_dir runs/.../val \\
    --output_dir runs/entity_number_memory_v1/learned_detail_resolver_v1 \\
    --checkpoint <path_to_gpt2_checkpoint.pt>
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

_EPS = 1e-9
VOCAB_SIZE = 50257

# ── Shard field aliases ────────────────────────────────────────────────────────
_TOPK_ALIASES  = ["base_topk_ids",    "base_topk",     "topk_ids"]
_LGT_ALIASES   = ["base_topk_logits", "base_topk_lgt", "topk_lgt", "topk_logits"]
_GOLD_ALIASES  = ["gold_token",       "gold",          "labels"]
_IDS_ALIASES   = ["input_ids"]
_ROWID_ALIASES = ["row_id", "row_ids"]
_OFF_ALIASES   = ["token_offset", "offsets", "offset"]

ABLATION_MODES = [
    "base_only",
    "prefix_memory_only",
    "doc_past_memory_only",
    "prefix_plus_doc_past",
    "shuffled_doc_past_control",
]

# ── Optional tiktoken decoder ──────────────────────────────────────────────────
_enc = None
try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
except Exception:
    pass


def _decode_tok(tid):
    if _enc is not None:
        try:
            return repr(_enc.decode([int(tid)]))[1:-1]
        except Exception:
            pass
    return f"<{tid}>"


def _decode_ids(ids):
    if _enc is not None:
        try:
            return _enc.decode([int(i) for i in ids if 0 <= int(i) < VOCAB_SIZE])
        except Exception:
            pass
    return " ".join(_decode_tok(i) for i in ids)


# ── Shard loading ──────────────────────────────────────────────────────────────

def _get(shard, aliases, required=True):
    for a in aliases:
        if a in shard:
            return shard[a]
    if required:
        raise KeyError(f"Missing field. Tried: {aliases}. Have: {list(shard.keys())}")
    return None


def load_shards(data_dir, cand_pool_size, max_rows=None, split_name="val"):
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {data_dir}")
    print(f"[shards] {len(paths)} {split_name} shards")
    bufs = defaultdict(list)
    total = 0
    first = True
    for sp in paths:
        if max_rows is not None and total >= max_rows:
            break
        sh = torch.load(sp, map_location="cpu", weights_only=False)
        if first:
            print(f"[preflight] first {split_name} shard keys:")
            for k, v in sh.items():
                print(f"  {k:30s}: shape={getattr(v,'shape',None)}"
                      f"  dtype={getattr(v,'dtype',type(v).__name__)}")
            first = False
        topk = _get(sh, _TOPK_ALIASES).long()
        lgt  = _get(sh, _LGT_ALIASES).float()
        gold = _get(sh, _GOLD_ALIASES).long()
        ids  = _get(sh, _IDS_ALIASES).long()
        rids = _get(sh, _ROWID_ALIASES, required=False)
        off  = _get(sh, _OFF_ALIASES,   required=False)
        B, K = topk.shape
        if rids is None:
            rids = torch.arange(total, total + B)
        if off is None:
            off = torch.full((B,), -1, dtype=torch.long)
        if K < cand_pool_size:
            topk = torch.cat([topk, torch.zeros(B, cand_pool_size - K, dtype=torch.long)], 1)
            lgt  = torch.cat([lgt,  torch.full((B, cand_pool_size - K), float("nan"))], 1)
        elif K > cand_pool_size:
            topk = topk[:, :cand_pool_size]
            lgt  = lgt[:,  :cand_pool_size]
        if max_rows is not None and total + B > max_rows:
            keep = max_rows - total
            topk, lgt, gold, ids, rids, off = (
                x[:keep] for x in (topk, lgt, gold, ids, rids, off))
            B = keep
        bufs["topk"].append(topk.numpy())
        bufs["lgt"].append(lgt.numpy())
        bufs["gold"].append(gold.numpy())
        bufs["ids"].append(ids.numpy())
        bufs["row_ids"].append(rids.numpy())
        bufs["token_offset"].append(off.numpy())
        total += B
    data = {k: np.concatenate(v, 0) for k, v in bufs.items()}
    data["lgt"] = np.where(np.isfinite(data["lgt"]), data["lgt"], -1e9)
    print(f"[shards] {total:,} {split_name} rows  pool_size={cand_pool_size}")
    return data


# ── Corpus loading ─────────────────────────────────────────────────────────────

def try_load_corpus(corpus_path_arg, train_dir):
    """
    Three-tier fallback:
      T1: --corpus_path (explicitly supplied)
      T2: .../00_raw_token_source/train_tokens.npy  (pipeline cache)
      T3: None — doc_past feature disabled (zeros)
    """
    if corpus_path_arg and os.path.isfile(corpus_path_arg):
        print(f"[corpus] Tier-1 load: {corpus_path_arg}")
        return np.load(corpus_path_arg, mmap_mode="r")
    # Tier 2: try pipeline-standard path
    parent = os.path.dirname(os.path.dirname(train_dir))
    t2 = os.path.join(parent, "00_raw_token_source", "train_tokens.npy")
    if os.path.isfile(t2):
        print(f"[corpus] Tier-2 load: {t2}")
        return np.load(t2, mmap_mode="r")
    print("[corpus] WARNING: no corpus found — doc_past memory will be zeros")
    return None


def fetch_doc_past_batch(corpus, token_offsets, doc_memory_len):
    """
    Returns numpy [B, doc_memory_len] int32.
    Rows with offset < doc_memory_len or corpus=None are filled with zeros.
    Sorted by offset for sequential mmap reads.
    """
    B = len(token_offsets)
    out = np.zeros((B, doc_memory_len), dtype=np.int32)
    if corpus is None:
        return out
    corpus_len = len(corpus)
    order = np.argsort(token_offsets)
    for bi in order:
        off = int(token_offsets[bi])
        if off >= doc_memory_len:
            start = off - doc_memory_len
            end   = min(off, corpus_len)
            if start < corpus_len and end > start:
                chunk = corpus[start:end].astype(np.int32)
                L = len(chunk)
                out[bi, doc_memory_len - L:] = chunk
    return out


# ── Memory construction ────────────────────────────────────────────────────────

def _pad_to_len(arr, length):
    """Ensure arr [B, L] has exactly L columns; left-pad with zeros if shorter."""
    B, L = arr.shape
    if L == length:
        return arr
    if L > length:
        return arr[:, -length:]
    pad = np.zeros((B, length - L), dtype=arr.dtype)
    return np.concatenate([pad, arr], axis=1)


def build_memory_ids(doc_past_np, prefix_np, mode, doc_memory_len, prefix_len):
    """
    Returns:
      memory_ids : [B, T]  int32 numpy
      pad_mask   : [B, T]  bool numpy  (True = padding, model ignores)
    """
    doc = _pad_to_len(doc_past_np.astype(np.int32), doc_memory_len)
    pre = _pad_to_len(prefix_np.astype(np.int32),   prefix_len)

    if mode == "prefix_memory_only":
        mem = pre
    elif mode == "doc_past_memory_only":
        mem = doc
    elif mode == "prefix_plus_doc_past":
        mem = np.concatenate([doc, pre], axis=1)
    elif mode == "shuffled_doc_past_control":
        doc_shuf = np.roll(doc, 1, axis=0)          # row i gets doc from row i-1
        mem = np.concatenate([doc_shuf, pre], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    pad_mask = (mem == 0)
    return mem, pad_mask


# ── Model ──────────────────────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn     = nn.MultiheadAttention(dim, num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.norm_q   = nn.LayerNorm(dim)
        self.norm_kv  = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, q, kv, key_padding_mask=None, need_weights=False):
        """
        q:   [B, Nq,  dim]
        kv:  [B, Nkv, dim]
        Returns: [B, Nq, dim], attn_weights or None
        """
        q_n  = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        out, attn_w = self.attn(q_n, kv_n, kv_n,
                                key_padding_mask=key_padding_mask,
                                need_weights=need_weights,
                                average_attn_weights=need_weights)
        return self.norm_out(q + out), attn_w


class LearnedDetailMemoryResolver(nn.Module):
    """
    Cross-attention resolver:
      - Token embeddings frozen from checkpoint (or random if no ckpt).
      - Candidates are Query; memory context (doc_past+prefix) is Key/Value.
      - Zero-init final MLP layer -> delta=0 at step 0 (identity).
    """

    def __init__(self,
                 vocab_size=VOCAB_SIZE,
                 emb_dim=768,
                 memory_dim=256,
                 num_heads=4,
                 doc_memory_len=2048,
                 prefix_len=128,
                 n_attn_layers=2,
                 mlp_hidden=512,
                 dropout=0.1):
        super().__init__()
        self.vocab_size     = vocab_size
        self.emb_dim        = emb_dim
        self.memory_dim     = memory_dim
        self.doc_memory_len = doc_memory_len
        self.prefix_len     = prefix_len

        # Frozen token embedding table
        self.token_emb = nn.Embedding(vocab_size, emb_dim)

        # Project to memory_dim
        if emb_dim != memory_dim:
            self.emb_proj = nn.Linear(emb_dim, memory_dim, bias=False)
        else:
            self.emb_proj = nn.Identity()

        # Positional embedding for memory sequence
        max_pos = doc_memory_len + prefix_len + 64
        self.pos_emb = nn.Embedding(max_pos, memory_dim)

        # Cross-attention layers: cands=Q, memory=KV
        self.attn_layers = nn.ModuleList([
            CrossAttentionLayer(memory_dim, num_heads, dropout=dropout)
            for _ in range(n_attn_layers)
        ])

        # Delta MLP: concat(cand_emb_proj, cand_state, base_logit) -> delta
        delta_in = memory_dim * 2 + 1
        self.delta_mlp = nn.Sequential(
            nn.Linear(delta_in, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1, bias=True),
        )
        # Zero-init final layer -> delta=0 at step 0
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)

    def freeze_token_emb(self):
        self.token_emb.requires_grad_(False)

    def forward(self,
                cand_ids,          # [B, M] int64
                memory_ids,        # [B, T] int64
                base_logits,       # [B, M] float32
                memory_pad_mask=None,  # [B, T] bool  True=padding
                return_attn=False):
        """
        Returns:
          final_logits: [B, M]  (base_logits + delta)
          attn_list:    list of [B, M, T] per layer, or None
        """
        B, M = cand_ids.shape
        _, T  = memory_ids.shape

        # Embed candidates
        cand_emb_raw = self.token_emb(cand_ids)          # [B, M, emb_dim]
        cand_emb     = self.emb_proj(cand_emb_raw)        # [B, M, memory_dim]

        # Embed memory
        mem_emb_raw = self.token_emb(memory_ids)          # [B, T, emb_dim]
        mem_emb     = self.emb_proj(mem_emb_raw)           # [B, T, memory_dim]
        pos_ids     = torch.arange(T, device=memory_ids.device)
        mem_emb     = mem_emb + self.pos_emb(pos_ids).unsqueeze(0)  # [B, T, memory_dim]

        # Cross-attention
        q = cand_emb
        attn_list = []
        for layer in self.attn_layers:
            q, attn_w = layer(q, mem_emb,
                              key_padding_mask=memory_pad_mask,
                              need_weights=return_attn)
            if return_attn and attn_w is not None:
                attn_list.append(attn_w)
        cand_state = q                                     # [B, M, memory_dim]

        # Delta MLP
        base_lgt_exp = base_logits.unsqueeze(-1)           # [B, M, 1]
        delta_input  = torch.cat([cand_emb, cand_state, base_lgt_exp], dim=-1)
        delta        = self.delta_mlp(delta_input).squeeze(-1)  # [B, M]

        return base_logits + delta, (attn_list if return_attn else None)


# ── Checkpoint embedding loading ───────────────────────────────────────────────

_EMB_KEYS = [
    "token_emb.weight",
    "embedding.weight",
    "transformer.wte.weight",
    "model.transformer.wte.weight",
    "embeddings.word_embeddings.weight",
    "model.embed_tokens.weight",
]


def load_token_emb_from_checkpoint(ckpt_path, model, device="cpu"):
    """Try to load token embeddings from a checkpoint. Returns True if loaded."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("[emb] No checkpoint supplied — using random-init token embeddings")
        return False
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        cleaned = {}
        for k, v in state.items():
            s = k
            for pfx in ("module.", "model.", "_orig_mod."):
                if s.startswith(pfx):
                    s = s[len(pfx):]
            cleaned[s] = v
        for key in _EMB_KEYS:
            if key in cleaned:
                w = cleaned[key]
                if w.shape == model.token_emb.weight.shape:
                    with torch.no_grad():
                        model.token_emb.weight.copy_(w.to(device))
                    print(f"[emb] Loaded token embeddings from '{key}'  shape={w.shape}")
                    return True
                else:
                    print(f"[emb] Key '{key}' shape mismatch: "
                          f"ckpt={w.shape} model={model.token_emb.weight.shape}")
        print(f"[emb] No matching key found. Tried: {_EMB_KEYS}")
        return False
    except Exception as e:
        print(f"[emb] WARNING: failed to load checkpoint: {e}")
        return False


# ── Slice building ─────────────────────────────────────────────────────────────

def build_val_slices(data, t2r_path=None, super_map_path=None):
    N    = len(data["gold"])
    gold = data["gold"].astype(np.int64)
    topk = data["topk"].astype(np.int64)
    lgt  = data["lgt"].astype(np.float32)
    ar   = np.arange(N)

    base_top1    = topk[ar, lgt.argmax(axis=1)]
    gold_in_pool = (topk == gold[:, None]).any(axis=1)
    base_correct = (base_top1 == gold)

    # Region/superregion maps (optional)
    tok_arr = reg_arr = None
    unk_r = unk_s = 1
    sr = False
    if t2r_path and os.path.isfile(t2r_path):
        with open(t2r_path) as f:
            raw = json.load(f)
        t2r = ({i: v for i, v in enumerate(raw) if v is not None}
               if isinstance(raw, list) else {int(k): v for k, v in raw.items()})
        unk_r = int(max(t2r.values())) + 1 if t2r else 1
        V = max(t2r.keys(), default=0) + 2
        tok_arr = np.full(V, unk_r, dtype=np.int32)
        for tid, rid in t2r.items():
            tok_arr[int(tid)] = int(rid)
        if super_map_path and os.path.isfile(super_map_path):
            with open(super_map_path) as f:
                raw2 = json.load(f)
            r2s = ({int(k): v for k, v in raw2.items()}
                   if not isinstance(raw2, list)
                   else {i: v for i, v in enumerate(raw2) if v is not None})
            unk_s = int(max(r2s.values())) + 1 if r2s else 1
            R = max(r2s.keys(), default=0) + 2
            reg_arr = np.full(R, unk_s, dtype=np.int32)
            for rid, sid in r2s.items():
                reg_arr[int(rid)] = int(sid)
            sr = True
    if tok_arr is None:
        tok_arr = np.zeros(VOCAB_SIZE, dtype=np.int32)
        reg_arr = np.zeros(2, dtype=np.int32)

    Vt = len(tok_arr)
    gc = np.clip(gold,      0, Vt - 1)
    bc = np.clip(base_top1, 0, Vt - 1)
    gold_reg = tok_arr[gc]
    base_reg = tok_arr[bc]
    same_reg = (gold_reg == base_reg) & (gold_reg != unk_r)
    same_sr  = same_reg.copy()
    if sr:
        Rr = len(reg_arr)
        gs = reg_arr[np.clip(gold_reg, 0, Rr - 1)]
        bs = reg_arr[np.clip(base_reg, 0, Rr - 1)]
        same_sr = (gs == bs) & (gs != unk_s)

    off        = data["token_offset"].astype(np.int64)
    has_offset = (off >= 0)

    slices = {
        "all":                np.ones(N, dtype=bool),
        "gold_in_pool":       gold_in_pool,
        "base_correct":       base_correct,
        "base_wrong_in_pool": ~base_correct & gold_in_pool,
        "has_doc_past":       has_offset,
        "same_region":        same_reg,
        "same_superregion":   same_sr,
    }
    print("[slices]")
    for s, m in slices.items():
        print(f"  {s:40s} = {m.sum():6,}/{N:,}")

    extra = {
        "base_top1":    base_top1,
        "gold_in_pool": gold_in_pool,
        "base_correct": base_correct,
    }
    return slices, extra


# ── Eval forward pass ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_mode(model, data, corpus, args, mode, device, batch_size=512):
    """Run model in one ablation mode over all val rows."""
    N    = len(data["gold"])
    topk_all = data["topk"].astype(np.int64)
    lgt_all  = data["lgt"].astype(np.float32)

    if mode == "base_only":
        ar   = np.arange(N)
        pred = topk_all[ar, lgt_all.argmax(axis=1)]
        return {"pred": pred, "final_lgt": lgt_all.copy()}

    final_lgt = np.empty_like(lgt_all, dtype=np.float32)
    model.eval()

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        sl   = slice(start, end)

        topk_np = topk_all[sl]
        lgt_np  = lgt_all[sl]
        ids_np  = data["ids"][sl].astype(np.int64)
        off_np  = data["token_offset"][sl].astype(np.int64)

        doc_past = fetch_doc_past_batch(corpus, off_np, args.doc_memory_len)
        prefix   = ids_np[:, -args.prefix_len:]
        prefix   = _pad_to_len(prefix, args.prefix_len)

        mem_np, pad_np = build_memory_ids(
            doc_past, prefix, mode, args.doc_memory_len, args.prefix_len)

        cand_ids = torch.tensor(topk_np, dtype=torch.long,    device=device)
        mem_ids  = torch.tensor(mem_np,  dtype=torch.long,    device=device)
        base_lgt = torch.tensor(lgt_np,  dtype=torch.float32, device=device)
        pad_mask = torch.tensor(pad_np,  dtype=torch.bool,    device=device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            fl, _ = model(cand_ids, mem_ids, base_lgt,
                          memory_pad_mask=pad_mask, return_attn=False)
        final_lgt[sl] = fl.float().cpu().numpy()

    ar   = np.arange(N)
    pred = topk_all[ar, final_lgt.argmax(axis=1)]
    return {"pred": pred, "final_lgt": final_lgt}


def compute_slice_metrics(data, pred, slices):
    gold = data["gold"].astype(np.int64)
    rows = []
    for sname, mask in slices.items():
        n = int(mask.sum())
        if n == 0:
            rows.append({"slice": sname, "n": 0, "acc": float("nan")})
            continue
        acc = float((pred[mask] == gold[mask]).mean())
        rows.append({"slice": sname, "n": n, "acc": acc})
    return rows


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args, model, train_data, val_data, corpus, device, slices, extra, out):
    gold_in_pool = (train_data["topk"] == train_data["gold"][:, None]).any(axis=1)
    has_offset   = (train_data["token_offset"].astype(np.int64) >= args.doc_memory_len)
    valid_mask   = gold_in_pool & has_offset
    valid_idxs   = np.where(valid_mask)[0]
    print(f"[train] valid rows (gold_in_pool & offset>={args.doc_memory_len}): "
          f"{len(valid_idxs):,} / {len(train_data['gold']):,}")

    if len(valid_idxs) == 0:
        print("[train] WARNING: no valid training rows — skipping training")
        return {}, {"no_improving_checkpoint_found": True}

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt        = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler     = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    base_acc_val = float(extra["base_correct"].mean())
    print(f"[train] base_acc_val = {base_acc_val:.4f}")

    best_model_acc = 0.0
    best_epoch     = -1
    best_metrics   = {"no_improving_checkpoint_found": True}
    train_log_rows = []
    eval_log_rows  = []
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        perm  = rng.permutation(len(valid_idxs))
        idxs  = valid_idxs[perm]
        model.train()
        ep_loss = ep_correct = ep_total = 0

        for start in range(0, len(idxs), args.train_batch_size):
            batch_idxs = idxs[start: start + args.train_batch_size]
            if len(batch_idxs) == 0:
                break

            topk_np = train_data["topk"][batch_idxs].astype(np.int64)
            lgt_np  = train_data["lgt"][batch_idxs].astype(np.float32)
            gold_np = train_data["gold"][batch_idxs].astype(np.int64)
            ids_np  = train_data["ids"][batch_idxs].astype(np.int64)
            off_np  = train_data["token_offset"][batch_idxs].astype(np.int64)

            doc_past = fetch_doc_past_batch(corpus, off_np, args.doc_memory_len)
            prefix   = _pad_to_len(ids_np[:, -args.prefix_len:], args.prefix_len)

            mem_np, pad_np = build_memory_ids(
                doc_past, prefix, "prefix_plus_doc_past",
                args.doc_memory_len, args.prefix_len)

            cand_ids = torch.tensor(topk_np, dtype=torch.long,    device=device)
            mem_ids  = torch.tensor(mem_np,  dtype=torch.long,    device=device)
            base_lgt = torch.tensor(lgt_np,  dtype=torch.float32, device=device)
            pad_mask = torch.tensor(pad_np,  dtype=torch.bool,    device=device)
            gold_t   = torch.tensor(gold_np, dtype=torch.long,    device=device)

            # CE target: index of gold in candidate pool
            gold_idx = (cand_ids == gold_t.unsqueeze(1)).float().argmax(dim=1)

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                fl, _ = model(cand_ids, mem_ids, base_lgt,
                              memory_pad_mask=pad_mask, return_attn=False)
                loss  = nn.functional.cross_entropy(fl, gold_idx)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()

            ep_correct += int((fl.argmax(dim=1) == gold_idx).sum())
            ep_total   += len(batch_idxs)
            ep_loss    += float(loss.item()) * len(batch_idxs)

        train_acc  = ep_correct / max(ep_total, 1)
        train_loss = ep_loss    / max(ep_total, 1)
        print(f"[epoch {epoch}/{args.epochs}] loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        train_log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "n_train": ep_total,
        })

        # Eval all ablation modes
        row = {"epoch": epoch, "base_acc": base_acc_val}
        for mode in ABLATION_MODES:
            r = eval_mode(model, val_data, corpus, args, mode, device,
                          batch_size=args.val_batch_size)
            m_acc = float((r["pred"] == val_data["gold"].astype(np.int64)).mean())
            row[f"{mode}_acc"] = m_acc
        eval_log_rows.append(row)

        main_acc = row["prefix_plus_doc_past_acc"]
        shuf_acc = row["shuffled_doc_past_control_acc"]
        print(f"[epoch {epoch}] val: base={base_acc_val:.4f}  "
              f"main={main_acc:.4f}  shuf={shuf_acc:.4f}")

        # Save best checkpoint only if model beats base
        if main_acc > base_acc_val and main_acc > best_model_acc:
            best_model_acc = main_acc
            best_epoch     = epoch
            ckpt_path      = os.path.join(out, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            best_metrics = {
                "epoch":                      epoch,
                "main_acc":                   main_acc,
                "base_acc":                   base_acc_val,
                "delta_acc":                  main_acc - base_acc_val,
                "no_improving_checkpoint_found": False,
            }
            print(f"[checkpoint] IMPROVED  acc={main_acc:.4f}  saved -> {ckpt_path}")

    _write_csv(os.path.join(out, "train_log.csv"), train_log_rows)
    _write_csv(os.path.join(out, "eval_log.csv"),  eval_log_rows)
    print(f"[train] done  best_epoch={best_epoch}  best_model_acc={best_model_acc:.4f}")
    return {}, best_metrics


# ── Final ablation eval ────────────────────────────────────────────────────────

def eval_ablation(model, val_data, corpus, args, slices, extra, device, out):
    gold = val_data["gold"].astype(np.int64)
    N    = len(gold)

    mode_results = {}
    for mode in ABLATION_MODES:
        print(f"  eval mode: {mode}")
        r = eval_mode(model, val_data, corpus, args, mode, device,
                      batch_size=args.val_batch_size)
        mode_results[mode] = r

    ablation_rows = []
    for mode in ABLATION_MODES:
        pred    = mode_results[mode]["pred"]
        acc_all = float((pred == gold).mean())
        gip     = extra["gold_in_pool"]
        acc_gip = float((pred[gip] == gold[gip]).mean()) if gip.any() else float("nan")
        ablation_rows.append({
            "mode":             mode,
            "acc_all":          acc_all,
            "acc_gold_in_pool": acc_gip,
            "n":                N,
            "n_gold_in_pool":   int(gip.sum()),
        })
    _write_csv(os.path.join(out, "ablation_results.csv"), ablation_rows)

    slice_rows_all = []
    for mode in ABLATION_MODES:
        pred = mode_results[mode]["pred"]
        for srow in compute_slice_metrics(val_data, pred, slices):
            slice_rows_all.append({"mode": mode, **srow})
    _write_csv(os.path.join(out, "eval_by_slice.csv"), slice_rows_all)

    return mode_results, ablation_rows, slice_rows_all


# ── Example collection ─────────────────────────────────────────────────────────

def _attn_note(i, data, model, corpus, args, device):
    """Return top-8 most-attended memory token IDs and positions for row i."""
    try:
        off     = int(data["token_offset"][i])
        doc_past = fetch_doc_past_batch(corpus, np.array([off]), args.doc_memory_len)
        prefix   = _pad_to_len(data["ids"][i:i+1, -args.prefix_len:].astype(np.int64),
                                args.prefix_len)
        mem_np, pad_np = build_memory_ids(
            doc_past, prefix, "prefix_plus_doc_past",
            args.doc_memory_len, args.prefix_len)
        cand_ids = torch.tensor(data["topk"][i:i+1].astype(np.int64),
                                dtype=torch.long, device=device)
        mem_ids  = torch.tensor(mem_np, dtype=torch.long,    device=device)
        base_lgt = torch.tensor(data["lgt"][i:i+1].astype(np.float32),
                                dtype=torch.float32, device=device)
        pad_mask = torch.tensor(pad_np, dtype=torch.bool,    device=device)
        model.eval()
        with torch.no_grad():
            _, attn_list = model(cand_ids, mem_ids, base_lgt,
                                 memory_pad_mask=pad_mask, return_attn=True)
        if not attn_list:
            return "no attn"
        attn = np.stack([a.float().cpu().numpy() for a in attn_list]).mean(axis=0)
        attn = attn[0].mean(axis=0)                    # [T]
        top8 = attn.argsort()[::-1][:8]
        parts = [f"'{_decode_tok(mem_np[0, p])}'(p={p})" for p in top8]
        return "top-attn: " + " ".join(parts)
    except Exception as e:
        return f"attn err: {e}"


def collect_examples(data, mode_results, slices, extra, args, corpus, model, device,
                     max_examples=50, seed=42):
    rng      = random.Random(seed)
    gold     = data["gold"].astype(np.int64)
    topk_all = data["topk"].astype(np.int64)
    lgt_all  = data["lgt"].astype(np.float32)
    ar       = np.arange(len(gold))

    base_pred  = topk_all[ar, lgt_all.argmax(axis=1)]
    main_pred  = mode_results["prefix_plus_doc_past"]["pred"]
    shuf_pred  = mode_results["shuffled_doc_past_control"]["pred"]
    main_lgt   = mode_results["prefix_plus_doc_past"]["final_lgt"]

    def sample_idx(mask, n):
        idx = np.where(mask)[0].tolist()
        rng.shuffle(idx)
        return idx[:n]

    def make_ex(i, with_attn=False):
        prefix_ids = data["ids"][i, -32:]
        cands = []
        for j in range(min(topk_all.shape[1], 16)):
            cands.append({
                "tok":       _decode_tok(topk_all[i, j]),
                "base_lgt":  float(lgt_all[i, j]),
                "model_lgt": float(main_lgt[i, j]),
                "is_gold":   int(topk_all[i, j]) == int(gold[i]),
            })
        ex = {
            "i":             i,
            "row_id":        int(data["row_ids"][i]),
            "token_offset":  int(data["token_offset"][i]),
            "prefix_text":   _decode_ids(prefix_ids),
            "gold_tok":      _decode_tok(gold[i]),
            "gold_tid":      int(gold[i]),
            "base_pred_tok": _decode_tok(base_pred[i]),
            "model_pred_tok": _decode_tok(main_pred[i]),
            "shuf_pred_tok": _decode_tok(shuf_pred[i]),
            "candidates":    cands,
        }
        if with_attn and corpus is not None:
            ex["attn_note"] = _attn_note(i, data, model, corpus, args, device)
        return ex

    helps_mask  = (base_pred != gold) & (main_pred == gold) & slices["gold_in_pool"]
    hurts_mask  = (base_pred == gold) & (main_pred != gold)
    shuf_diff   = (main_pred != shuf_pred) & slices["gold_in_pool"]
    no_doc_mask = slices["gold_in_pool"] & ~slices["has_doc_past"]

    # For attn_memory_top: rows from helps with attn notes (highest model gain)
    helps_idxs  = sample_idx(helps_mask, max_examples * 2)[:max_examples]
    attn_idxs   = helps_idxs[:min(20, max_examples)]   # attn is expensive; cap at 20

    return {
        "helps":        [make_ex(i, with_attn=False) for i in helps_idxs],
        "hurts":        [make_ex(i, with_attn=False) for i in sample_idx(hurts_mask,  max_examples)],
        "attn_top":     [make_ex(i, with_attn=True)  for i in attn_idxs],
        "shuf_vs_real": [make_ex(i, with_attn=True)  for i in sample_idx(shuf_diff,   max_examples)],
        "no_doc":       [make_ex(i, with_attn=False) for i in sample_idx(no_doc_mask, max_examples)],
    }


def _fmt_ex_md(ex):
    lines = [
        f"**Row {ex['i']}**  row_id={ex['row_id']}  offset={ex['token_offset']}",
        f"- Prefix: `{ex['prefix_text'][:80]}`",
        f"- Gold: `{ex['gold_tok']}` (tid={ex['gold_tid']})",
        f"- Base: `{ex['base_pred_tok']}`  Model: `{ex['model_pred_tok']}`"
        f"  Shuf: `{ex['shuf_pred_tok']}`",
    ]
    if "attn_note" in ex:
        lines.append(f"- Attn: {ex['attn_note']}")
    lines.append("\n| cand | base_lgt | model_lgt | gold |")
    lines.append("|------|----------|-----------|------|")
    for c in ex["candidates"][:8]:
        lines.append(f"| `{c['tok'][:14]}` | {c['base_lgt']:.3f} | "
                     f"{c['model_lgt']:.3f} | {'yes' if c['is_gold'] else ''} |")
    return "\n".join(lines) + "\n"


# ── Report writing ────────────────────────────────────────────────────────────

def write_report(args, ablation_rows, slice_rows_all, val_data, extra, best_metrics, out):
    N    = len(val_data["gold"])

    def acc(mode):
        return next((r["acc_all"] for r in ablation_rows if r["mode"] == mode),
                    float("nan"))

    def slice_acc(mode, sname):
        return next((r["acc"] for r in slice_rows_all
                     if r["mode"] == mode and r["slice"] == sname),
                    float("nan"))

    base      = acc("base_only")
    main      = acc("prefix_plus_doc_past")
    pref_only = acc("prefix_memory_only")
    doc_only  = acc("doc_past_memory_only")
    shuf      = acc("shuffled_doc_past_control")
    improved  = not best_metrics.get("no_improving_checkpoint_found", True)

    path = os.path.join(out, "report.md")
    with open(path, "w") as f:
        f.write("# Learned Detail Memory Resolver V1 — Report\n\n")
        f.write("> LEARNED MODEL. Gold used only for CE loss and metrics.\n\n")
        f.write(f"**N_val:** {N:,}  |  **epochs:** {args.epochs}  |  "
                f"**pool:** {args.candidate_pool_size}  |  "
                f"**doc_memory_len:** {args.doc_memory_len}  |  "
                f"**prefix_len:** {args.prefix_len}\n\n---\n\n")

        # Q1
        f.write("## Q1: Does learned memory improve over base on full val?\n\n")
        f.write(f"- base_only acc:          **{base:.4f}**\n")
        f.write(f"- prefix_plus_doc_past:   **{main:.4f}**  (delta={main-base:+.4f})\n")
        q1 = ("YES" if main > base + 0.001 else
              "MARGINAL" if main > base else "NO")
        f.write(f"- **{q1}**\n\n")

        # Q2
        f.write("## Q2: Which memory source contributes most?\n\n")
        f.write("| mode | acc_all | acc_gold_in_pool |\n|------|---------|------------------|\n")
        for r in ablation_rows:
            f.write(f"| {r['mode']:30s} | {r['acc_all']:.4f} | "
                    f"{r['acc_gold_in_pool']:.4f} |\n")
        src = "prefix" if (pref_only - base) > (doc_only - base) else "doc_past"
        f.write(f"\nPrefix-only delta={pref_only-base:+.4f},  "
                f"doc_past-only delta={doc_only-base:+.4f}  => **{src}** dominates\n\n")

        # Q3
        f.write("## Q3: Does shuffled doc_past reveal genuine memory use?\n\n")
        f.write(f"- prefix_plus_doc_past:          {main:.4f}\n")
        f.write(f"- shuffled_doc_past_control:     {shuf:.4f}  (delta={main-shuf:+.4f})\n")
        q3 = ("YES — genuine doc memory use" if main > shuf + 0.001 else
              "MARGINAL" if main > shuf else "NO — no improvement over shuffled")
        f.write(f"- **{q3}**\n\n")

        # Q4
        f.write("## Q4: Does it help where base is wrong but gold in pool?\n\n")
        bwp_base  = slice_acc("base_only",             "base_wrong_in_pool")
        bwp_main  = slice_acc("prefix_plus_doc_past",  "base_wrong_in_pool")
        f.write(f"- base on base_wrong_in_pool:   {bwp_base:.4f}\n")
        f.write(f"- model on base_wrong_in_pool:  {bwp_main:.4f}  "
                f"(delta={bwp_main-bwp_base:+.4f})\n")
        q4 = "YES" if bwp_main > bwp_base + 0.005 else "NO"
        f.write(f"- **{q4}**\n\n")

        # Q5
        f.write("## Q5: Should we deploy as a memory expert?\n\n")
        if improved and (main > shuf + 0.001) and (bwp_main > bwp_base):
            verdict = "YES — beats base, genuine memory use, helps BucketA"
        elif improved:
            verdict = "PARTIAL — checkpoint improved base but investigate ablations"
        else:
            verdict = "NO — no checkpoint improved over base"
        f.write(f"- Best checkpoint improved: {'Yes' if improved else 'No'}\n")
        if improved:
            f.write(f"  - epoch={best_metrics.get('epoch','?')}  "
                    f"acc={best_metrics.get('main_acc', float('nan')):.4f}  "
                    f"delta={best_metrics.get('delta_acc', float('nan')):+.4f}\n")
        f.write(f"- **{verdict}**\n\n")

        # Slice table
        f.write("---\n\n## Slice Breakdown (prefix_plus_doc_past vs base)\n\n")
        f.write("| slice | base | model | delta |\n|-------|------|-------|-------|\n")
        for sname in ["all", "gold_in_pool", "base_correct", "base_wrong_in_pool",
                      "has_doc_past", "same_region", "same_superregion"]:
            bv = slice_acc("base_only",            sname)
            mv = slice_acc("prefix_plus_doc_past", sname)
            f.write(f"| {sname:30s} | {bv:.4f} | {mv:.4f} | {mv-bv:+.4f} |\n")

        f.write(f"\n\n*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"[save] {path}")
    return path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {path}")


def _write_examples(path, exs, header):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n\n({len(exs)} examples)\n\n")
        for ex in exs:
            f.write("---\n")
            f.write(_fmt_ex_md(ex))
            f.write("\n")
    print(f"[save] {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Learned Detail Memory Resolver V1")
    # Data
    p.add_argument("--train_dir",           required=True)
    p.add_argument("--val_dir",             required=True)
    p.add_argument("--corpus_path",         default=None)
    p.add_argument("--token_to_region",     default=None)
    p.add_argument("--super_map",           default=None)
    p.add_argument("--output_dir",          required=True)
    # Memory
    p.add_argument("--doc_memory_len",      type=int,   default=2048)
    p.add_argument("--prefix_len",          type=int,   default=128)
    p.add_argument("--candidate_pool_size", type=int,   default=32)
    # Model arch
    p.add_argument("--checkpoint",          default=None,
                   help="Path to GPT-2 checkpoint for token embedding init")
    p.add_argument("--emb_dim",             type=int,   default=768)
    p.add_argument("--memory_dim",          type=int,   default=256)
    p.add_argument("--num_heads",           type=int,   default=4)
    p.add_argument("--n_attn_layers",       type=int,   default=2)
    p.add_argument("--mlp_hidden",          type=int,   default=512)
    p.add_argument("--dropout",             type=float, default=0.1)
    # Training
    p.add_argument("--epochs",              type=int,   default=5)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--weight_decay",        type=float, default=1e-4)
    p.add_argument("--train_batch_size",    type=int,   default=256)
    p.add_argument("--val_batch_size",      type=int,   default=512)
    p.add_argument("--max_train_rows",      type=int,   default=None)
    p.add_argument("--max_val_rows",        type=int,   default=None)
    # Misc
    p.add_argument("--max_examples",        type=int,   default=50)
    p.add_argument("--seed",                type=int,   default=42)
    return p, p.parse_args()


def main():
    _, args = _parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    t0 = time.time()

    print("=" * 64)
    print(" Learned Detail Memory Resolver V1")
    print(f" train_dir:  {args.train_dir}")
    print(f" val_dir:    {args.val_dir}")
    print(f" output_dir: {args.output_dir}")
    print("=" * 64)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[step 1] loading train shards...")
    train_data = load_shards(args.train_dir, args.candidate_pool_size,
                             args.max_train_rows, split_name="train")
    print("\n[step 2] loading val shards...")
    val_data   = load_shards(args.val_dir, args.candidate_pool_size,
                             args.max_val_rows, split_name="val")

    # ── Corpus ───────────────────────────────────────────────────────────
    print("\n[step 3] loading corpus...")
    corpus = try_load_corpus(args.corpus_path, args.train_dir)

    # ── Val slices ───────────────────────────────────────────────────────
    print("\n[step 4] building val slices...")
    slices, extra = build_val_slices(val_data, args.token_to_region, args.super_map)

    # ── Model ────────────────────────────────────────────────────────────
    print("\n[step 5] building model...")
    model = LearnedDetailMemoryResolver(
        vocab_size     = VOCAB_SIZE,
        emb_dim        = args.emb_dim,
        memory_dim     = args.memory_dim,
        num_heads      = args.num_heads,
        doc_memory_len = args.doc_memory_len,
        prefix_len     = args.prefix_len,
        n_attn_layers  = args.n_attn_layers,
        mlp_hidden     = args.mlp_hidden,
        dropout        = args.dropout,
    )
    loaded_emb = load_token_emb_from_checkpoint(args.checkpoint, model)
    model.freeze_token_emb()
    model = model.to(device)

    n_total    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_total:,} total params  {n_trainable:,} trainable  "
          f"emb_loaded={loaded_emb}")

    # ── Train ────────────────────────────────────────────────────────────
    print("\n[step 6] training...")
    _, best_metrics = train(args, model, train_data, val_data, corpus,
                            device, slices, extra, args.output_dir)

    # ── Load best checkpoint if it improved ──────────────────────────────
    best_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.isfile(best_path):
        print(f"\n[step 7] loading best checkpoint: {best_path}")
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True))
    else:
        print("\n[step 7] no best checkpoint — using final model weights")

    # ── Final ablation eval ──────────────────────────────────────────────
    print("\n[step 8] final ablation eval...")
    mode_results, ablation_rows, slice_rows_all = eval_ablation(
        model, val_data, corpus, args, slices, extra, device, args.output_dir)

    # ── Examples ─────────────────────────────────────────────────────────
    print("\n[step 9] collecting examples...")
    examples = collect_examples(
        val_data, mode_results, slices, extra, args, corpus, model, device,
        max_examples=args.max_examples, seed=args.seed)
    _write_examples(os.path.join(args.output_dir, "examples_detail_helps.md"),
                    examples["helps"],
                    "Learned memory corrected base prediction")
    _write_examples(os.path.join(args.output_dir, "examples_detail_hurts.md"),
                    examples["hurts"],
                    "Learned memory broke correct base prediction")
    _write_examples(os.path.join(args.output_dir, "examples_attn_memory_top.md"),
                    examples["attn_top"],
                    "Top attention-to-memory examples (from helps set)")
    _write_examples(os.path.join(args.output_dir, "examples_shuffled_vs_real.md"),
                    examples["shuf_vs_real"],
                    "Shuffled doc_past vs real doc_past prediction differs")
    _write_examples(os.path.join(args.output_dir, "examples_no_doc_past.md"),
                    examples["no_doc"],
                    "Gold in pool but no doc_past (offset < doc_memory_len)")

    # ── Metrics JSON ─────────────────────────────────────────────────────
    final_metrics = {
        "N_val":       len(val_data["gold"]),
        "ablation":    {r["mode"]: r["acc_all"] for r in ablation_rows},
        "best_metrics": best_metrics,
        "elapsed_s":   round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"[save] {args.output_dir}/final_metrics.json")
    print(f"[save] {args.output_dir}/best_metrics.json")

    # ── Report ───────────────────────────────────────────────────────────
    print("\n[step 10] writing report...")
    report_path = write_report(args, ablation_rows, slice_rows_all,
                               val_data, extra, best_metrics, args.output_dir)

    # ── config.json ──────────────────────────────────────────────────────
    cfg = vars(args).copy()
    cfg.update({"device": str(device), "n_params": n_total,
                "n_trainable": n_trainable, "emb_loaded": loaded_emb})
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[save] {args.output_dir}/config.json")

    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print(f" Learned Detail Memory Resolver V1 complete")
    print(f" elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f" report:  {report_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
