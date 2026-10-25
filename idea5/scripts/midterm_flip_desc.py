"""Counterfactual spatial descriptions: the same scene with only the spatial facts reversed.

GPT-5.6 rewrites each test item's four descriptions (rgb, depth, segmentation, pose), swapping near and far
distances, toward and away facing, left and right, approaching and retreating, and within and out of reach,
while keeping every person, object and activity unchanged. One isolated text-only call per item; the
model never sees the question, options or labels.
Output: output/idea5-midterm/descriptions_flipped/{id}.json  ({"rgb","depth","segmentation","pose"})
--paraphrase instead rewords the notes keeping every fact (spatial ones included), the control for the rewrite
itself; output goes to descriptions_paraphrased/.

usage: midterm_flip_desc.py [--workers 6] [--limit N] [--paraphrase]
"""
import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MID = ROOT / 'output/idea5-midterm'
DESC = MID / 'descriptions'
OUT = MID / 'descriptions_flipped'
KEYS = ['rgb', 'depth', 'segmentation', 'pose']
PROMPT = ('Below are four spatial notes describing the same egocentric scene. Rewrite them so that ONLY the spatial facts '
          'are reversed: people and objects described as near or within reach (under about 1.5 m) become far (3-5 m) '
          'and out of reach, and far ones become near and within reach; facing toward the camera wearer becomes facing '
          'away and vice versa; left and right are swapped; approaching becomes moving away and vice versa. Keep every '
          'person, object, activity, colour and detection-quality remark otherwise identical, and keep each note about '
          'the same length. Return only JSON with the same four keys: {"rgb": "...", "depth": "...", '
          '"segmentation": "...", "pose": "..."}\n\n')
PARA = ('Below are four spatial notes describing the same egocentric scene. Reword them in your own words so they read '
        'differently, but keep every fact exactly the same: every distance, direction, facing, left/right position, '
        'movement, reach judgement, person, object, activity, colour and detection-quality remark. Keep each note about '
        'the same length. Return only JSON with the same four keys: {"rgb": "...", "depth": "...", '
        '"segmentation": "...", "pose": "..."}\n\n')


def run(item_id, out=OUT, prompt=PROMPT):
    dest = out / f'{item_id}.json'
    if dest.exists():
        return 'skip'
    rgb = json.loads((DESC / 'rgb' / f'{item_id}.json').read_text())['descriptions']
    maps = json.loads((DESC / 'maps' / f'{item_id}.json').read_text())['descriptions']
    notes = json.dumps({**rgb, **maps}, indent=1)
    for _ in range(3):
        with tempfile.TemporaryDirectory(prefix='flip_') as tmp:
            out = Path(tmp) / 'reply.txt'
            subprocess.run(['codex', 'exec', '-m', 'gpt-5.6-sol', '--sandbox', 'read-only', '--skip-git-repo-check',
                            '--ephemeral', '-C', tmp, '-o', str(out), '-'], input=prompt + notes, capture_output=True,
                           text=True, timeout=600)
            text = out.read_text() if out.exists() else ''
        try:
            body = json.loads(text[text.find('{'):text.rfind('}') + 1])
            if all(isinstance(body.get(k), str) and body[k] for k in KEYS):
                dest.write_text(json.dumps({'id': item_id, 'model': 'gpt-5.6-sol', **{k: body[k] for k in KEYS}}, indent=1))
                return 'ok'
        except ValueError:
            continue
    return 'fail'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--paraphrase', action='store_true')
    args = ap.parse_args()
    out, prompt = (MID / 'descriptions_paraphrased', PARA) if args.paraphrase else (OUT, PROMPT)
    out.mkdir(parents=True, exist_ok=True)
    ids = [i for i in json.loads((MID / 'splits.json').read_text())['test'] if (DESC / 'rgb' / f'{i}.json').exists()][:args.limit]
    counts = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for n, r in enumerate(ex.map(lambda i: run(i, out, prompt), ids), 1):
            counts[r] = counts.get(r, 0) + 1
            if n % 25 == 0:
                print(n, len(ids), counts, flush=True)
    print('done', counts, flush=True)


if __name__ == '__main__':
    main()
