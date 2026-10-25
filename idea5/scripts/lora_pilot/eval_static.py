"""Bridges-2: manipulation check on held-out SAT `static` items (same distribution as the spatial arm's training data).

Samples 500 single-image static items whose row index is not among the 5,000 training rows, then evaluates
base / spatial / control on them. SAT val and the real test set contain only dynamic question types, so they do
not measure the skill the spatial arm was trained on.
usage: eval_static.py
"""
import gc
import json
import random

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import prep_data
from common import ROOT, load
from eval_models import generate, letter

N = 500


def build():
    path = ROOT / 'data/eval_sat_static.jsonl'
    if path.exists():
        return [json.loads(line) for line in path.open()]
    trained = {int(json.loads(line)['id'].split('_')[1]) for line in (ROOT / 'data/train_spatial.jsonl').open()}
    pf = pq.ParquetFile(hf_hub_download('array/SAT', 'SAT_static.parquet', repo_type='dataset'))
    rng = random.Random(7)
    want = set(rng.sample([i for i in range(pf.metadata.num_rows) if i not in trained], 2 * N))
    rows, offset = [], 0
    for batch in pf.iter_batches(batch_size=16):
        hit = [i - offset for i in range(offset, offset + batch.num_rows) if i in want]
        if hit:
            table = batch.to_pylist()
            for i in hit:
                r = table[i]
                if len(r['image_bytes']) == 1 and r['correct_answer'] in r['answers']:
                    rows.append((offset + i, r))
        offset += batch.num_rows
    rows = sorted(rows)[:N]
    out = []
    for idx, r in rows:
        choices = list(r['answers'])
        rng.shuffle(choices)
        out.append({'id': f'satstaticeval_{idx}', 'images': [prep_data.save_image(r['image_bytes'][0], f'satstaticeval_{idx}')],
                    'question': r['question'], 'choices': choices, 'answer': prep_data.LETTERS[choices.index(r['correct_answer'])],
                    'prompt': prep_data.prompt(r['question'], choices)})
    with path.open('w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    return out


def main():
    rows = build()
    result = {}
    for name, adapter in [('base', None), ('spatial', 'adapters/spatial'), ('control', 'adapters/control')]:
        model, processor = load(str(ROOT / adapter) if adapter else None)
        model.eval()
        replies = generate(model, processor, [(r['images'], r['prompt']) for r in rows], 8)
        result[name] = [{'id': r['id'], 'reply': x, 'pred': letter(x), 'answer': r['answer'], 'correct': letter(x) == r['answer'],
                         'k': len(r['choices'])} for r, x in zip(rows, replies)]
        print(name, 'sat_static_heldout', round(sum(x['correct'] for x in result[name]) / len(rows), 3), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (ROOT / 'results/sat_static_heldout.json').write_text(json.dumps(result, indent=1))
    print('STATIC_DONE')


if __name__ == '__main__':
    main()
