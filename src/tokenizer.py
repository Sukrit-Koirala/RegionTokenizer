import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

def build_or_load_tokenizer(cfg, train_texts, outdir):
    path = os.path.join(outdir, 'tokenizer.json')
    if os.path.exists(path):
        return PreTrainedTokenizerFast(tokenizer_file=path, bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]')
    tok = Tokenizer(BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=cfg['tokenizer']['vocab_size'], special_tokens=['[PAD]','[UNK]','[BOS]','[EOS]'])
    tok.train_from_iterator(train_texts, trainer)
    tok.save(path)
    return PreTrainedTokenizerFast(tokenizer_file=path, bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]')