import os, argparse, yaml, random, numpy as np, torch
from src.data import load_agnews_texts, tokenize_dataset
from src.tokenizer import build_or_load_tokenizer
from src.model import build_gpt_model
from src.train import run_training

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.outdir, exist_ok=True)
    seed_all(args.seed)
    train_texts, test_texts = load_agnews_texts()
    tok = build_or_load_tokenizer(cfg, train_texts, args.outdir)
    train_ds = tokenize_dataset(train_texts, tok, cfg['dataset']['seq_len'])
    test_ds = tokenize_dataset(test_texts, tok, cfg['dataset']['seq_len'])
    model = build_gpt_model(cfg, tok)
    run_training(model, tok, train_ds, test_ds, cfg, args.outdir)

if __name__ == '__main__':
    main()