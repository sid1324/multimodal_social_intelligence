"""Fixed train/test split for the Idea 5 midterm, chosen by hash order before any image or label is read.

test = the 200-item verified split. train = 400 non-verified items, one per source video, from source videos
not in test; train_ext = the next 200 by the same order (used only if time allows).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/EgoNormia'
SEED = 'midterm-v1'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    ids = sorted(json.loads((DATA / 'annotations/final_data.json').read_text()))
    test = sorted(json.loads((DATA / 'annotations/verified_split.json').read_text())['split'])
    test_sources = {i.split('_')[0] for i in test}
    by_source = {}
    for item_id in sorted(ids, key=lambda i: digest(f'{SEED}:{i}')):
        source = item_id.split('_')[0]
        if item_id not in test and source not in test_sources:
            by_source.setdefault(source, item_id)
    pool = sorted(by_source.values(), key=lambda i: digest(f'{SEED}:{i}'))
    split = {'seed': SEED, 'test': test, 'train': pool[:400], 'train_ext': pool[400:600]}
    (ROOT / 'output/idea5-midterm/splits.json').write_text(json.dumps(split, indent=1))
    print({k: len(v) for k, v in split.items() if k != 'seed'}, 'eligible sources', len(pool))


if __name__ == '__main__':
    main()
