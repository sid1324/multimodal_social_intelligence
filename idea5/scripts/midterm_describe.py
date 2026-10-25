"""Spatial descriptions of every item's RGB strip and maps with Gemini 4 Argon through the Antigravity CLI.

Two isolated calls per item, each in its own empty directory holding only that call's images:
  rgb:  the RGB strip alone -> {"rgb": ...}
  maps: the RGB strip + depth, segmentation and pose strips -> {"depth": ..., "segmentation": ..., "pose": ...}
The model never sees the question, options or labels. Verified-split items run first. Resumable.

usage: midterm_describe.py [--workers 12] [--model gemini-4-argon-medium] [--limit N]
"""
import argparse
import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/EgoNormia'
GRIDS = ROOT / 'output/idea5-midterm/grids'
OUT = ROOT / 'output/idea5-midterm/descriptions'

STRIP = ('rgb.jpg is a horizontal strip of 5 frames, left to right in time order, sampled from the few seconds of a '
         'head-mounted (egocentric) video; the camera wearer is the viewpoint.')
ASPECTS = ('the layout of the scene; which people and objects are present and roughly how far each is from the camera wearer '
           '(in metres) and from each other; relative positions (left/right/in front/behind, above/below); which way people '
           'face and whether they look toward the camera wearer; what is within arm\'s reach of the camera wearer or of others; '
           'occlusions and anything partly out of view; and how these change across the 5 frames (approach, retreat, turning).')
NO_SOCIAL = 'Describe only spatial facts. Do not guess anyone\'s intentions, social norms, or what anyone should do.'

PROMPTS = {
    'rgb': (f'{STRIP} Describe the spatial understanding this image supports: {ASPECTS} {NO_SOCIAL} '
            'Return only JSON: {"rgb": "..."} with 4-6 sentences.'),
    'maps': (f'{STRIP} depth.jpg, segmentation.jpg and pose.jpg are the same 5 frames rendered as a monocular depth map '
             '(brighter = nearer to the camera wearer), an instance segmentation map (coloured masks with class labels), '
             'and a body/hand skeleton map (yellow = body, magenta = hands) on black. Look at all four images. For EACH of the '
             'three maps, describe the spatial understanding that map supports, using the RGB strip for context: '
             f'{ASPECTS} Say what each map shows that is hard to see in the RGB alone, and where the map looks unreliable. '
             f'{NO_SOCIAL} Return only JSON: {{"depth": "...", "segmentation": "...", "pose": "..."}} with 3-5 sentences each.'),
}
FILES = {'rgb': ['rgb'], 'maps': ['rgb', 'depth', 'segmentation', 'pose']}


def parse(stdout):
    start = stdout.find('{"conversation_id"')
    meta = json.loads(stdout[start:].splitlines()[0])
    text = meta['response'].strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    body = json.loads(text[text.find('{'):text.rfind('}') + 1])
    return meta, body


def codex_call(prompt, images, model, tmp):
    order = ', '.join(f'image {n + 1} = {Path(i).name}' for n, i in enumerate(images))
    out = Path(tmp) / 'reply.txt'
    cmd = ['codex', 'exec', '-m', model, '--sandbox', 'read-only', '--skip-git-repo-check', '--ephemeral', '-C', tmp,
           '-o', str(out)] + [a for i in images for a in ('-i', i)] + ['-']
    t = time.time()
    proc = subprocess.run(cmd, input=f'The attached images are, in order: {order}. {prompt}', capture_output=True,
                          text=True, timeout=600)
    text = out.read_text() if out.exists() else ''
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    body = json.loads(text[text.find('{'):text.rfind('}') + 1])
    return {'status': 'SUCCESS' if proc.returncode == 0 else 'ERROR', 'duration_seconds': round(time.time() - t, 1)}, body


def run(item_id, kind, model):
    dest = OUT / kind / f'{item_id}.json'
    if dest.exists():
        return 'skip'
    for attempt in range(3):
        with tempfile.TemporaryDirectory(prefix='desc_') as tmp:
            for k in FILES[kind]:
                shutil.copy(GRIDS / f'{item_id}_{k}.jpg', Path(tmp) / f'{k}.jpg')
            try:
                if model.startswith('gpt'):
                    meta, body = codex_call(PROMPTS[kind], [str(Path(tmp) / f'{k}.jpg') for k in FILES[kind]], model, tmp)
                else:
                    proc = subprocess.run(['agy', '-p', PROMPTS[kind], '--model', model, '--output-format', 'json'],
                                          cwd=tmp, capture_output=True, text=True, timeout=600)
                    meta, body = parse(proc.stdout)
                want = ['rgb'] if kind == 'rgb' else ['depth', 'segmentation', 'pose']
                if meta.get('status') != 'SUCCESS' or not all(isinstance(body.get(k), str) and body[k] for k in want):
                    raise ValueError('bad response')
            except Exception as e:  # noqa: BLE001
                err = f'{type(e).__name__}: {e}'
                continue
        dest.write_text(json.dumps({'id': item_id, 'kind': kind, 'model': model, 'prompt': PROMPTS[kind],
                                    'descriptions': {k: body[k] for k in want}, 'duration_seconds': meta.get('duration_seconds'),
                                    'usage': meta.get('usage'), 'conversation_id': meta.get('conversation_id')}, indent=1))
        return 'ok'
    (OUT / 'errors').mkdir(parents=True, exist_ok=True)
    (OUT / 'errors' / f'{kind}_{item_id}.txt').write_text(err)
    return 'fail'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--model', default='gemini-4-argon-medium')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--splits', default='test,train,train_ext')
    args = ap.parse_args()
    for kind in FILES:
        (OUT / kind).mkdir(parents=True, exist_ok=True)
    split = json.loads((OUT.parent / 'splits.json').read_text())
    ids = [i for name in args.splits.split(',') for i in split[name]]
    ids = [i for i in ids if all((GRIDS / f'{i}_{k}.jpg').exists() for k in FILES['maps'])][:args.limit]
    jobs = [(i, kind) for i in ids for kind in FILES]
    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, i, kind, args.model) for i, kind in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            counts[f.result()] += 1
            if n % 25 == 0:
                print(n, len(jobs), counts, flush=True)
    print('done', counts)


if __name__ == '__main__':
    main()
