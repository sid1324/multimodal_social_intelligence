"""Bridges-2: evaluate one adapter on the 500 held-out SAT static items and the 200 EgoNormia test items.

Same prompts, decoding and parsing as eval_models.py / eval_static.py; EgoNormia replies are scored locally.
usage: eval_arm.py adapters/<name> results/<name>.json
"""
import json
import re
import sys

from common import ROOT, load
from eval_models import egonormia_prompt, generate, letter


def main():
    adapter, out = sys.argv[1], sys.argv[2]
    model, processor = load(str(ROOT / adapter))
    model.eval()
    static = [json.loads(line) for line in (ROOT / 'data/eval_sat_static.jsonl').open()]
    replies = generate(model, processor, [(r['images'], r['prompt']) for r in static], 8)
    result = {'model': adapter,
              'sat_static': [{'id': r['id'], 'reply': x, 'pred': letter(x), 'answer': r['answer'], 'correct': letter(x) == r['answer']}
                             for r, x in zip(static, replies)]}
    tasks = json.loads((ROOT / 'data/egonormia_tasks.json').read_text())
    replies = generate(model, processor, [([f"data/egonormia_images/{t['image']}"], egonormia_prompt(t)) for t in tasks], 32)
    rows = []
    for t, x in zip(tasks, replies):
        m = re.search(r'"action"\s*:\s*([1-5]).*?"justification"\s*:\s*([1-5])', x, re.S)
        rows.append({'id': t['id'], 'reply': x, 'action_position': int(m.group(1)) if m else None,
                     'justification_position': int(m.group(2)) if m else None})
    result['egonormia'] = rows
    (ROOT / out).write_text(json.dumps(result, indent=1))
    print(adapter, 'sat_static', round(sum(r['correct'] for r in result['sat_static']) / len(static), 3),
          'egonormia parsed', sum(r['action_position'] is not None for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
