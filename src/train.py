import math, os, json, torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

def run_training(model, tok, train_ds, test_ds, cfg, outdir):
    t = cfg['training']
    args = TrainingArguments(
        output_dir=outdir,
        num_train_epochs=t['epochs'],
        per_device_train_batch_size=t['batch_size'],
        per_device_eval_batch_size=t['batch_size'],
        learning_rate=float(t['lr']),
        weight_decay=float(t['weight_decay']),
        warmup_steps=int(float(t['warmup_ratio']) * t['epochs'] * 3750),
        lr_scheduler_type='cosine',
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        bf16=torch.cuda.is_available(),
        report_to='none',
        load_best_model_at_end=True,
        save_total_limit=2
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, data_collator=collator)
    trainer.train()
    metrics = trainer.evaluate()
    metrics['perplexity'] = math.exp(metrics['eval_loss'])
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    trainer.save_model(os.path.join(outdir,'final_model'))
    tok.save_pretrained(os.path.join(outdir,'final_model'))