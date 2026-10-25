"""Local: GRPO training set = every EgoNormia item whose source video is not a test video (labels included; train only).

Options get a fixed per-item hash order (as lora_pilot/make_egonormia_tasks.py), and the trainer reshuffles them each
epoch. gold is the original index; base_order[p] is the original index shown at position p. Spatial labels for the
training ids are added by add_spatial_labels() once spatial_labels_all.json exists. Test labels are never written here.
usage: make_train_set.py [--labels-only]
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/EgoNormia'
MID = ROOT / 'output/idea5-midterm'
UP = MID / 'rl_pilot/upload'
NONE = {'behaviors': 'None of these actions is appropriate', 'justifications': 'None of these justifications is appropriate'}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def train_ids():
    data = json.loads((DATA / 'annotations/final_data.json').read_text())
    test = set(json.loads((MID / 'splits.json').read_text())['test'])
    test_sources = {i.split('_')[0] for i in test}
    return data, sorted(i for i in data if i.split('_')[0] not in test_sources)


def add_spatial_labels():
    _, ids = train_ids()
    labels = json.loads((MID / 'spatial_labels_all.json').read_text())
    keep = {i: {k: labels[i][k] for k in ('n_people', 'nearest_dist', 'facing', 'hands')} for i in ids if i in labels}
    (UP / 'train_spatial_labels.json').write_text(json.dumps(keep))
    print('spatial labels for', len(keep), 'of', len(ids), 'train ids')


def main():
    if sys.argv[1:] == ['--labels-only']:
        return add_spatial_labels()
    data, ids = train_ids()
    rows = []
    for item_id in ids:
        x = data[item_id]
        order = {f: sorted(range(5), key=lambda i: digest(f'rl-v1:{item_id}:{f}:{i}')) for f in NONE}
        rows.append({'id': item_id, 'image': f'{item_id}_rgb.jpg', 'gold': x['correct'],
                     'behaviors': [t or NONE['behaviors'] for t in x['behaviors']],
                     'justifications': [t or NONE['justifications'] for t in x['justifications']],
                     'base_order': {'action': order['behaviors'], 'justification': order['justifications']}})
        src = MID / 'grids' / f'{item_id}_rgb.jpg'
        if src.exists():
            shutil.copy(src, UP / 'train_images' / src.name)
    (UP / 'train_items.json').write_text(json.dumps(rows))
    missing = [r['id'] for r in rows if not (UP / 'train_images' / r['image']).exists()]
    print(len(rows), 'train items from', len({i.split('_')[0] for i in ids}), 'videos; missing images', len(missing))
    if (MID / 'spatial_labels_all.json').exists():
        add_spatial_labels()


if __name__ == '__main__':
    main()
