"""Local: build the EgoNormia evaluation package for the LoRA pilot (200 verified test items, RGB strip only).

Options are shuffled per item by a fixed hash so the empty option is not always last. Only options and images are
shipped to Bridges-2; the answer key (permutations + gold) stays in output/idea5-midterm/lora_pilot/answer_key.json.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/EgoNormia'
MID = ROOT / 'output/idea5-midterm'
OUT = MID / 'lora_pilot'
NONE = {'behaviors': 'None of these actions is appropriate', 'justifications': 'None of these justifications is appropriate'}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    data = json.loads((DATA / 'annotations/final_data.json').read_text())
    test = json.loads((MID / 'splits.json').read_text())['test']
    tasks, key = [], {}
    for item_id in test:
        x = data[item_id]
        order = {f: sorted(range(5), key=lambda i: digest(f'pilot-v1:{item_id}:{f}:{i}')) for f in NONE}
        tasks.append({'id': item_id, 'image': f'{item_id}_rgb.jpg',
                      'action_options': [x['behaviors'][i] or NONE['behaviors'] for i in order['behaviors']],
                      'justification_options': [x['justifications'][i] or NONE['justifications'] for i in order['justifications']]})
        key[item_id] = {'gold': x['correct'], 'action_order': order['behaviors'], 'justification_order': order['justifications'],
                        'categories': x['taxonomy'].get(str(x['correct']), [])}
        shutil.copy(MID / 'grids' / f'{item_id}_rgb.jpg', OUT / 'egonormia_images' / f'{item_id}_rgb.jpg')
    (OUT / 'egonormia_tasks.json').write_text(json.dumps(tasks, indent=1))
    (OUT / 'answer_key.json').write_text(json.dumps(key, indent=1))
    print(len(tasks), 'tasks')


if __name__ == '__main__':
    main()
