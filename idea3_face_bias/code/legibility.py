"""Trait-legibility check on the existing edited scenes: does a frontier VLM perceive the intended
low < baseline < high trustworthiness ordering in our OMI composites?

For each of the five pilot scenes, the frame with the largest target mask is shown in four versions
(original, low, baseline, high) and two views (whole frame with a fixed yellow box around the target;
close-up crop of the same region). Each image is rated separately (never side by side), three times.
Ratings mainly reflect expression (high-trust references smile), so this is a legibility check,
not a bias result. Design fixed before any results were seen.

Usage: python legibility.py build <archive dir>    (copies stimuli into data/legibility_stimuli)
       python legibility.py run <model>
       python legibility.py analyze <model>
"""
import base64, io, json, re, sys, time, concurrent.futures as cf
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pilot import env, call

HERE = Path(__file__).parent
STIM = HERE / 'data' / 'legibility_stimuli'
OUT = HERE / 'results'
ITEMS = [969, 1625, 605, 749, 1250]
CONDS = ['original', 'low', 'baseline', 'high']
VIEWS = ['scene', 'closeup']
REPEATS = 3
PROMPT = ('This image is used in face-perception research on first impressions. Look at the person whose '
          'face is inside the yellow box. Based only on their appearance, how trustworthy does this person '
          'seem at first impression? Answer with a single number from 1 (very untrustworthy) to 7 (very '
          'trustworthy), and nothing else.')
PROMPT_CLOSE = PROMPT.replace('whose face is inside the yellow box', 'shown in this close-up')


def build(archive):
    archive = Path(archive)
    STIM.mkdir(parents=True, exist_ok=True)
    meta = {}
    for item in ITEMS:
        d = archive / f'item_{item}'
        areas = {}
        for f in range(1, 6):
            if not (d / 'masks' / f'frame_{f}.png').exists():  # frame without a visible target
                continue
            m = np.array(Image.open(d / 'masks' / f'frame_{f}.png').convert('L'))
            areas[f] = int((m > 0).sum())
        frame = max(areas, key=areas.get)
        m = np.array(Image.open(d / 'masks' / f'frame_{frame}.png').convert('L')) > 0
        ys, xs = np.where(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        pad = int(0.15 * max(x1 - x0, y1 - y0))
        box = [int(max(x0 - pad, 0)), int(max(y0 - pad, 0)), int(x1 + pad), int(y1 + pad)]
        side = int(1.8 * max(x1 - x0, y1 - y0))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        for c in CONDS:
            src = d / ('original' if c == 'original' else f'omi/{c}') / f'frame_{frame}.png'
            im = Image.open(src).convert('RGB')
            scene = im.copy()
            ImageDraw.Draw(scene).rectangle(box, outline=(255, 220, 0), width=2)
            scene.save(STIM / f'{item}_{c}_scene.png')
            crop = im.crop((max(cx - side // 2, 0), max(cy - side // 2, 0),
                            min(cx + side // 2, im.width), min(cy + side // 2, im.height)))
            crop.resize((384, round(384 * crop.height / crop.width)), Image.LANCZOS).save(STIM / f'{item}_{c}_closeup.png')
        meta[item] = {'frame': frame, 'box': box, 'mask_pixels': areas[frame]}
    (STIM / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def b64(path):
    buf = io.BytesIO()
    Image.open(path).convert('RGB').save(buf, format='JPEG', quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def run(model):
    from openai import OpenAI
    base, key = env()
    client = OpenAI(base_url=base, api_key=key, timeout=180)
    path = OUT / f"legibility_{model.replace('/', '_')}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get('rating') is not None:
                done.add((r['item'], r['condition'], r['view'], r['repeat']))
    jobs = [(i, c, v, k) for i in ITEMS for c in CONDS for v in VIEWS for k in range(REPEATS)
            if (i, c, v, k) not in done]
    print(f'{len(jobs)} calls to run')

    def work(job):
        i, c, v, k = job
        reply, served = call(client, model, b64(STIM / f'{i}_{c}_{v}.png'), PROMPT if v == 'scene' else PROMPT_CLOSE)
        m = re.search(r'\b([1-7])\b', reply or '')
        return {'item': i, 'condition': c, 'view': v, 'repeat': k, 'model': model,
                'served_model': served if reply is not None else None, 'error': served if reply is None else None,
                'reply': reply, 'rating': int(m.group(1)) if m else None, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}

    with open(path, 'a', encoding='utf-8') as fh, cf.ThreadPoolExecutor(8) as pool:
        for n, r in enumerate(pool.map(work, jobs), 1):
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            if n % 20 == 0 or n == len(jobs):
                print(f'{n}/{len(jobs)} done', flush=True)


def analyze(model):
    path = OUT / f"legibility_{model.replace('/', '_')}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    ok = [r for r in rows if r['rating'] is not None]
    out = {'model': model, 'calls': len(rows), 'rated': len(ok),
           'unrated_replies': [r['reply'] for r in rows if r['rating'] is None][:5]}
    for v in VIEWS:
        table = {}
        for i in ITEMS:
            table[i] = {c: round(float(np.mean([r['rating'] for r in ok if r['item'] == i and r['view'] == v
                                                and r['condition'] == c] or [np.nan])), 2) for c in CONDS}
        ordered = sum(1 for i in ITEMS if table[i]['low'] < table[i]['baseline'] < table[i]['high'])
        high_gt_low = sum(1 for i in ITEMS if table[i]['high'] > table[i]['low'])
        out[v] = {'mean_by_item': table, 'scenes_with_low<base<high': ordered, 'scenes_with_high>low': high_gt_low,
                  'mean_high_minus_low': round(float(np.nanmean([table[i]['high'] - table[i]['low'] for i in ITEMS])), 2)}
    (OUT / f"legibility_{model.replace('/', '_')}_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'build':
        build(sys.argv[2])
    elif mode == 'run':
        run(sys.argv[2])
    else:
        analyze(sys.argv[2])
