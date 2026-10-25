"""Bridges-2: greedy evaluation on the 200 EgoNormia test tasks with an arm's own prompt format (no labels here).

Raw replies are saved and scored locally (score_rl.py). Uses the same displayed option order as the LoRA pilot
(data/egonormia_tasks.json, positions mapped back locally with lora_pilot/answer_key.json).
usage: eval_rl.py --format social|spatial --adapter base|adapters/<name> --out results/<name>.json [--limit N]
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import rl_common as rc

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
BATCH = 16
MAX_NEW = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', required=True, choices=['social', 'spatial'])
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    if args.adapter != 'base':
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = 'left'
    tasks = json.loads((ROOT / 'data/egonormia_tasks.json').read_text())[:args.limit]
    rows = []
    for s in range(0, len(tasks), BATCH):
        chunk = tasks[s:s + BATCH]
        convs = [[{'role': 'user', 'content': [
            {'type': 'image', 'image': Image.open(ROOT / 'data/egonormia_images' / t['image']).convert('RGB')},
            {'type': 'text', 'text': rc.prompt(args.format, t['action_options'], t['justification_options'])}]}] for t in chunk]
        inputs = processor.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW)
        replies = processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        rows += [{'id': t['id'], 'reply': x} for t, x in zip(chunk, replies)]
        print(s + len(chunk), 'of', len(tasks), flush=True)
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps({'format': args.format, 'adapter': args.adapter, 'max_new_tokens': MAX_NEW, 'rows': rows}, indent=1))
    parsed = sum(rc.parse_answer(r['reply'])[0] is not None for r in rows)
    print('EVAL_DONE', args.adapter, args.format, 'parsed', parsed, '/', len(rows))


if __name__ == '__main__':
    main()
