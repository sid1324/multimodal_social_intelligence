"""Bridges-2: build the two training arms and the SAT evaluation sets for the LoRA pilot.

spatial arm: 5,000 single-image items from SAT `static` (fixed seed).
control arm: 5,000 VQAv2 items (HuggingFaceM4/the_cauldron, one shard), questions with spatial words removed, turned
             into multiple choice with the same number of choices as SAT (k drawn from SAT's own k distribution) and
             distractors drawn from other VQAv2 answers of the same kind (yes/no, number, other).
Both arms use the same prompt template and a single-letter target. SAT eval: 500 val items + the 150 real test items.
Writes data/{train_spatial,train_control,eval_sat_val,eval_sat_test}.jsonl and images under data/img/.
"""
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data'
IMG = OUT / 'img'
N_TRAIN = 5000
SEED = 0
LETTERS = 'ABCDEFGH'
SPATIAL = re.compile(r'\b(left|right|behind|front|near|nearer|nearest|far|farther|further|above|below|under|underneath|over|beside|'
                     r'next to|closer|closest|distance|top|bottom|where|between|inside|outside|across|toward|towards|facing|'
                     r'direction|position|side|corner|back|ahead|up|down)\b', re.I)
MAX_SIDE = 640


def prompt(question, choices):
    lines = [question.strip(), 'Choices:'] + [f'{LETTERS[i]}. {c}' for i, c in enumerate(choices)]
    return '\n'.join(lines + ['Answer with the letter of the correct choice only.'])


def save_image(raw, name):
    raw = raw['bytes'] if isinstance(raw, dict) else raw
    im = Image.open(io.BytesIO(raw)).convert('RGB')
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    path = IMG / f'{name}.jpg'
    im.save(path, quality=92)
    return str(path.relative_to(ROOT))


def sat_rows(split, n, rng, single_image_only, name):
    path = hf_hub_download('array/SAT', f'SAT_{split}.parquet', repo_type='dataset')
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    want = set(rng.sample(range(total), min(total, n * 2 if single_image_only else n)))
    rows, offset = [], 0
    # Small batches: whole row groups hold >2 GB of image bytes, which pyarrow cannot convert in one piece.
    for batch in pf.iter_batches(batch_size=16):
        size = batch.num_rows
        hit = [i - offset for i in range(offset, offset + size) if i in want]
        if hit:
            table = batch.to_pylist()
            for i in hit:
                r = table[i]
                if single_image_only and len(r['image_bytes']) != 1:
                    continue
                rows.append((offset + i, r))
        offset += size
    rows.sort()
    rows = rows[:n]
    out = []
    for idx, r in rows:
        choices = list(r['answers'])
        if r['correct_answer'] not in choices:
            continue
        rng.shuffle(choices)
        images = [save_image(im, f'{name}_{idx}_{k}') for k, im in enumerate(r['image_bytes'])]
        out.append({'id': f'{name}_{idx}', 'images': images, 'question': r['question'], 'choices': choices,
                    'answer': LETTERS[choices.index(r['correct_answer'])], 'question_type': r.get('question_type'),
                    'prompt': prompt(r['question'], choices)})
    return out


def kind(answer):
    a = answer.strip().lower()
    return 'yesno' if a in ('yes', 'no') else 'number' if a.isdigit() else 'other'


def control_rows(k_dist, rng):
    files = ['vqav2/train-00000-of-00022-1f40dc68b5c44ca4.parquet', 'vqav2/train-00001-of-00022-a0319a74304abd66.parquet']
    candidates = []
    for f in files:
        pf = pq.ParquetFile(hf_hub_download('HuggingFaceM4/the_cauldron', f, repo_type='dataset'))
        rows_iter = (r for batch in pf.iter_batches(batch_size=64) for r in batch.to_pylist())
        for n, r in enumerate(rows_iter):
            for qa in r['texts']:
                q = qa['user'].split('\n')[0].strip()
                a = qa['assistant'].strip().rstrip('.')
                if not SPATIAL.search(q) and a and len(a.split()) <= 3:
                    candidates.append((f'{f}:{n}', r['images'][0], q, a))
                    break
        if len(candidates) >= 3 * N_TRAIN:
            break
    pool = {k: sorted({a for *_, a in candidates if kind(a) == k}) for k in ('yesno', 'number', 'other')}
    freq = Counter(a for *_, a in candidates)
    pool['other'] = [a for a, _ in freq.most_common() if kind(a) == 'other'][:2000]
    rng.shuffle(candidates)
    out = []
    for key, raw, q, a in candidates[:N_TRAIN]:
        k = 2 if kind(a) == 'yesno' else rng.choice(k_dist)
        others = [x for x in pool[kind(a)] if x.lower() != a.lower()]
        choices = [a] + rng.sample(others, min(k - 1, len(others)))
        rng.shuffle(choices)
        out.append({'id': f'vqa_{len(out)}', 'images': [save_image(raw, f'vqa_{len(out)}')], 'question': q, 'choices': choices,
                    'answer': LETTERS[choices.index(a)], 'prompt': prompt(q, choices), 'source': key})
    return out


def dump(rows, name):
    with (OUT / f'{name}.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(name, len(rows), 'k', Counter(len(r['choices']) for r in rows), 'answer', Counter(r['answer'] for r in rows))


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    spatial = sat_rows('static', N_TRAIN, rng, True, 'satstatic')
    dump(spatial, 'train_spatial')
    k_dist = [len(r['choices']) for r in spatial]
    dump(control_rows(k_dist, random.Random(SEED + 1)), 'train_control')
    dump(sat_rows('val', 500, random.Random(SEED + 2), False, 'satval'), 'eval_sat_val')
    dump(sat_rows('test', 150, random.Random(SEED + 3), False, 'sattest'), 'eval_sat_test')
    for r in spatial[:3]:
        print(json.dumps({k: r[k] for k in ('prompt', 'answer', 'question_type')}))


if __name__ == '__main__':
    main()
