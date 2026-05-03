#!/usr/bin/env python3
"""
decode_region_tokens.py

Converts leaf_region_to_tokens.json (sub-indices) into actual token strings
using frequent_token_ids.npy and the GPT-2 tokenizer.

Produces region_tokens_decoded.json in the same format as
cluster_tokens_coactivation_final.json from the old 5k pipeline.

Usage:
    python interference_experiment/decode_region_tokens.py \
        --output_dir interference_experiment/full_vocab_region_eval \
        --model_name gpt2-xl
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",
                   default="interference_experiment/full_vocab_region_eval")
    p.add_argument("--model_name", default="gpt2-xl")
    p.add_argument("--top_n", type=int, default=None,
                   help="Only keep top-n tokens per region by freq (default: all)")
    args = p.parse_args()

    top_ids   = np.load(os.path.join(args.output_dir, "frequent_token_ids.npy"))
    tok_freqs = np.load(os.path.join(args.output_dir, "token_freqs.npy"))

    with open(os.path.join(args.output_dir, "leaf_region_to_tokens.json")) as f:
        region_to_sub: dict = json.load(f)

    print(f"Loading tokenizer {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    out: dict[str, list[dict]] = {}

    for region_id, sub_indices in region_to_sub.items():
        sub_arr  = np.array(sub_indices, dtype=np.int32)
        vocab_ids = top_ids[sub_arr]
        freqs     = tok_freqs[sub_arr]

        # sort by frequency descending
        order    = np.argsort(freqs)[::-1]
        vocab_ids = vocab_ids[order]
        freqs     = freqs[order]

        if args.top_n is not None:
            vocab_ids = vocab_ids[:args.top_n]
            freqs     = freqs[:args.top_n]

        entries = []
        for vid, freq in zip(vocab_ids.tolist(), freqs.tolist()):
            token_str = tokenizer.decode([vid])
            entries.append({
                "token":    token_str,
                "token_id": vid,
                "freq":     int(freq),
            })

        out[region_id] = entries

    out_path = os.path.join(args.output_dir, "region_tokens_decoded.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Written {out_path}  ({len(out)} regions)")

    # quick sanity print — first 3 regions, top 10 tokens each
    for rid in list(out.keys())[:3]:
        top = [e["token"] for e in out[rid][:10]]
        print(f"  region {rid} (size={len(out[rid])}): {top}")


if __name__ == "__main__":
    main()
