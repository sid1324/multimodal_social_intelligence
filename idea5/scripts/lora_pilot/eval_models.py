"""Bridges-2: greedy evaluation of one model (base or LoRA) on SAT val, SAT real test (circular) and EgoNormia.

One item per prompt; independent prompts are generated in left-padded batches. No answer key is present here:
EgoNormia replies are saved raw and scored locally.
usage: eval_models.py base|adapters/spatial|adapters/control results/<name>.json [--limit N]
"""
import argparse
import json
import re

import torch

from common import ROOT, load, messages

LETTERS = 'ABCDEFGH'
TASK = ('Choose the single most socially appropriate next action for the camera wearer from the action options, '
        'then choose the justification that best fits the action you chose.')
CONTEXT = ("The image is a left-to-right grid of five chronological frames from an egocentric (head-mounted) camera, "
           "captured before the camera wearer's next action. " + TASK)
BATCH = 8


def egonormia_prompt(t):
    lines = [CONTEXT, '', 'Action options:'] + [f'{i}. {o}' for i, o in enumerate(t['action_options'], 1)]
    lines += ['', 'Justification options:'] + [f'{i}. {o}' for i, o in enumerate(t['justification_options'], 1)]
    lines += ['', 'Reply with only this JSON and nothing else: {"action": <1-5>, "justification": <1-5>}']
    return '\n'.join(lines)


@torch.no_grad()
def generate(model, processor, jobs, max_new_tokens):
    """jobs: list of (image_paths, prompt). Returns decoded replies in order."""
    processor.tokenizer.padding_side = 'left'
    replies = []
    for s in range(0, len(jobs), BATCH):
        convs = [messages(paths, text) for paths, text in jobs[s:s + BATCH]]
        batch = processor.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors='pt', padding=True).to('cuda')
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        replies += processor.batch_decode(out[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
    return replies


def letter(reply):
    m = re.search(r'\b([A-H])\b', reply.strip())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    model, processor = load(None if args.model == 'base' else str(ROOT / args.model))
    model.eval()
    result = {'model': args.model}

    val = [json.loads(line) for line in (ROOT / 'data/eval_sat_val.jsonl').open()][:args.limit]
    replies = generate(model, processor, [(r['images'], r['prompt']) for r in val], 8)
    result['sat_val'] = [{'id': r['id'], 'reply': x, 'pred': letter(x), 'answer': r['answer'], 'correct': letter(x) == r['answer'],
                          'question_type': r.get('question_type')} for r, x in zip(val, replies)]

    test = [json.loads(line) for line in (ROOT / 'data/eval_sat_test.jsonl').open()][:args.limit]
    jobs, meta = [], []
    for r in test:
        k = len(r['choices'])
        gold = r['choices'][LETTERS.index(r['answer'])]
        for shift in range(k):
            choices = r['choices'][shift:] + r['choices'][:shift]
            text = r['prompt'].split('\nChoices:')[0] + '\nChoices:\n' + '\n'.join(f'{LETTERS[i]}. {c}' for i, c in enumerate(choices))
            text += '\nAnswer with the letter of the correct choice only.'
            jobs.append((r['images'], text))
            meta.append((r['id'], shift, LETTERS[choices.index(gold)]))
    replies = generate(model, processor, jobs, 8)
    result['sat_test'] = [{'id': i, 'shift': s, 'reply': x, 'pred': letter(x), 'answer': a, 'correct': letter(x) == a}
                          for (i, s, a), x in zip(meta, replies)]

    tasks = json.loads((ROOT / 'data/egonormia_tasks.json').read_text())[:args.limit]
    replies = generate(model, processor, [([f"data/egonormia_images/{t['image']}"], egonormia_prompt(t)) for t in tasks], 32)
    rows = []
    for t, x in zip(tasks, replies):
        m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', x, re.S)
        rows.append({'id': t['id'], 'reply': x, 'action_position': int(m.group(1)) if m else None,
                     'justification_position': int(m.group(2)) if m else None})
    result['egonormia'] = rows

    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(result, indent=1))
    acc = lambda rs: sum(r['correct'] for r in rs) / max(1, len(rs))  # noqa: E731
    print(args.model, 'sat_val', round(acc(result['sat_val']), 3), 'sat_test(per-rotation)', round(acc(result['sat_test']), 3),
          'egonormia parsed', sum(r['action_position'] is not None for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
