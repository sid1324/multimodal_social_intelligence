"""Runpod: EgoNormia test replies for base / spatial LoRA under the RGB-only prompt or the combined input.

combined = one user message: RGB strip, its description, then each of depth / segmentation / pose followed by its
description (Gemini 4 descriptions, written without the question), then the unchanged RGB-condition question text.
Same model, LoRA loading, greedy decoding (32 new tokens) and reply parsing as eval_arm.py. No labels here.
usage: eval_combined.py base|adapters/<name> rgb|combined results/<run>.json [--batch 8] [--limit N]
"""
import argparse
import json
import re
import time

import torch
from PIL import Image

from common import ROOT, load
from eval_models import egonormia_prompt

LABEL = {'rgb': 'Video frames', 'depth': 'Depth map (brighter is nearer)', 'segmentation': 'Instance segmentation map',
         'pose': 'Body and hand skeleton map'}


def conversation(t, cond, desc):
    img = lambda p: {'type': 'image', 'image': Image.open(ROOT / p).convert('RGB')}  # noqa: E731
    content = [img(f"data/egonormia_images/{t['image']}")]
    if cond == 'combined':
        d = desc[t['id']]
        if d['rgb']:  # one test item has no RGB description (generation failed twice)
            content.append({'type': 'text', 'text': f"{LABEL['rgb']}: {d['rgb']}"})
        for k in ('depth', 'segmentation', 'pose'):
            content += [img(f"data/grids/{t['id']}_{k}.jpg"), {'type': 'text', 'text': f'{LABEL[k]}: {d[k]}'}]
    content.append({'type': 'text', 'text': egonormia_prompt(t)})
    return [{'role': 'user', 'content': content}]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('cond', choices=['rgb', 'combined'])
    ap.add_argument('out')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    model, processor = load(None if args.model == 'base' else str(ROOT / args.model))
    model.eval()
    processor.tokenizer.padding_side = 'left'
    tasks = json.loads((ROOT / 'data/egonormia_tasks.json').read_text())[:args.limit]
    desc = json.loads((ROOT / 'data/combined_desc.json').read_text())
    t0, rows, tokens = time.time(), [], []
    for s in range(0, len(tasks), args.batch):
        chunk = tasks[s:s + args.batch]
        batch = processor.apply_chat_template([conversation(t, args.cond, desc) for t in chunk], tokenize=True,
                                              add_generation_prompt=True, return_dict=True, return_tensors='pt',
                                              padding=True).to('cuda')
        tokens += batch['attention_mask'].sum(1).tolist()
        out = model.generate(**batch, max_new_tokens=32, do_sample=False)
        for t, x in zip(chunk, processor.batch_decode(out[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)):
            m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', x, re.S)
            rows.append({'id': t['id'], 'reply': x, 'action_position': int(m.group(1)) if m else None,
                         'justification_position': int(m.group(2)) if m else None})
        print(len(rows), round(time.time() - t0), 's', f'{torch.cuda.max_memory_allocated() / 2**30:.1f}GB', flush=True)
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps({'model': args.model, 'cond': args.cond, 'batch': args.batch,
                                             'mean_prompt_tokens': sum(tokens) / len(tokens), 'seconds': time.time() - t0,
                                             'egonormia': rows}, indent=1))
    print('DONE', args.model, args.cond, 'parsed', sum(r['action_position'] is not None for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
