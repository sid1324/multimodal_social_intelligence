"""Bridges-2: LoRA r=16 on the language model of Qwen3-VL-8B-Instruct, 1 epoch, identical recipe for both arms.

Loss: next-token cross-entropy on the answer tokens only (prompt and image tokens masked).
usage: train_lora.py data/train_spatial.jsonl adapters/spatial [--max-steps N] [--seed S]
--seed sets the data order, LoRA initialisation and dropout; the 5,000 items are the same for every seed.
"""
import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model

from common import LORA_TARGETS, ROOT, load, messages

LR, ACCUM, WARMUP = 1e-4, 8, 30


def encode(processor, row):
    full = processor.apply_chat_template(messages(row['images'], row['prompt'], row['answer']), tokenize=True,
                                         return_dict=True, return_tensors='pt')
    prompt = processor.apply_chat_template(messages(row['images'], row['prompt']), tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt')
    n = prompt['input_ids'].shape[1]
    assert torch.equal(full['input_ids'][0, :n], prompt['input_ids'][0]), 'prompt is not a prefix of the full sequence'
    labels = full['input_ids'].clone()
    labels[0, :n] = -100
    full['labels'] = labels
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('out')
    ap.add_argument('--max-steps', type=int)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in (ROOT / args.data).open()]
    random.shuffle(rows)
    if args.max_steps:
        rows = rows[:args.max_steps]
    model, processor = load()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    total = math.ceil(len(rows) / ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / WARMUP) * max(0.0, 1 - s / total))
    model.train()
    log, running, started = [], 0.0, time.time()
    for i, row in enumerate(rows, 1):
        batch = {k: v.to('cuda') for k, v in encode(processor, row).items()}
        loss = model(**batch).loss
        (loss / ACCUM).backward()
        running += loss.item()
        if i % ACCUM == 0 or i == len(rows):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if i % 200 == 0 or i == len(rows):
            log.append({'step': i, 'loss': running / 200 if i % 200 == 0 else running / (i % 200), 'seconds': time.time() - started})
            print(json.dumps(log[-1]), flush=True)
            running = 0.0
    model.save_pretrained(ROOT / args.out)
    (ROOT / args.out / 'train_log.json').write_text(json.dumps({'n': len(rows), 'seed': args.seed, 'lr': LR, 'accum': ACCUM, 'r': 16, 'log': log}))


if __name__ == '__main__':
    main()
