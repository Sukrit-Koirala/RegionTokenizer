from transformers import GPT2Config, GPT2LMHeadModel

def build_gpt_model(cfg, tok):
    m = cfg['model']; seq = cfg['dataset']['seq_len']
    conf = GPT2Config(
        vocab_size=len(tok), n_positions=seq, n_ctx=seq,
        n_embd=m['d_model'], n_layer=m['n_layers'], n_head=m['n_heads'],
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
    return GPT2LMHeadModel(conf)