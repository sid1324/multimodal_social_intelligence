"""Bridges-2: narrow non-spatial control arm, 5,000 VQAv2 "what colour" items (one per image), format-matched to SAT static.

Questions with spatial wording are removed with the same filter as the broad control. Each item becomes multiple
choice with k drawn from the spatial arm's own k distribution; distractors are other colour answers. Same prompt
template and single-letter target as the other arms. Shards are read in order until 5,000 items are found.
Writes data/train_color.jsonl.
"""
import json
import random
import re
from collections import Counter

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from prep_data import LETTERS, OUT, SPATIAL, N_TRAIN, dump, prompt, save_image

COLOR = re.compile(r'^what colou?r\b', re.I)


def main():
    rng = random.Random(3)
    k_dist = [len(json.loads(line)['choices']) for line in (OUT / 'train_spatial.jsonl').open()]
    files = sorted(f for f in HfApi().list_repo_files('HuggingFaceM4/the_cauldron', repo_type='dataset') if f.startswith('vqav2/'))
    candidates = []
    for f in files:
        pf = pq.ParquetFile(hf_hub_download('HuggingFaceM4/the_cauldron', f, repo_type='dataset'))
        rows = (r for batch in pf.iter_batches(batch_size=64) for r in batch.to_pylist())
        for n, r in enumerate(rows):
            for qa in r['texts']:
                q = qa['user'].split('\n')[0].strip()
                a = qa['assistant'].strip().rstrip('.')
                if COLOR.search(q) and not SPATIAL.search(q) and a and len(a.split()) <= 3:
                    candidates.append((f'{f}:{n}', r['images'][0], q, a))
                    break
        print(f, len(candidates), flush=True)
        if len(candidates) >= N_TRAIN:
            break
    freq = Counter(a.lower() for *_, a in candidates)
    pool = [a for a, c in freq.most_common() if c >= 2]
    rng.shuffle(candidates)
    out = []
    for key, raw, q, a in candidates[:N_TRAIN]:
        k = rng.choice(k_dist)
        others = [x for x in pool if x != a.lower()]
        choices = [a] + rng.sample(others, k - 1)
        rng.shuffle(choices)
        out.append({'id': f'color_{len(out)}', 'images': [save_image(raw, f'color_{len(out)}')], 'question': q, 'choices': choices,
                    'answer': LETTERS[choices.index(a)], 'prompt': prompt(q, choices), 'source': key})
    dump(out, 'train_color')


if __name__ == '__main__':
    main()
