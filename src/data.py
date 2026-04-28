from datasets import load_dataset

def load_agnews_texts():
    ds = load_dataset('ag_news')
    train = [x['text'] for x in ds['train']]
    test = [x['text'] for x in ds['test']]
    return train, test

from datasets import Dataset

def tokenize_dataset(texts, tokenizer, seq_len):
    all_ids = []
    for t in texts:
        all_ids.extend(tokenizer(t)['input_ids'])
    chunks = [all_ids[i:i + seq_len] for i in range(0, len(all_ids) - seq_len + 1, seq_len)]
    return Dataset.from_list([{'input_ids': c, 'attention_mask': [1] * seq_len, 'labels': c[:]} for c in chunks])