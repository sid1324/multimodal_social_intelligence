"""Text-cue pilot: does a sentence about the partner's apparent trustworthiness change a frontier
VLM's EgoNormia answer? The model always sees the same five frames, question and options;
only one added sentence differs between conditions. Design fixed before any results were seen.

Items: 60 drawn (seed 2028) from items whose gold option is non-empty, whose correct-option norms
include a primary (face-invariant) norm, and whose options mix person-directed and other actions.
Conditions: neutral, neutral_repeat (same prompt again: sampling noise), paraphrase (wording noise),
low, high (trust cue). Outcomes: accuracy, flip rate vs neutral, person-directed choice rate.

Usage: python pilot.py run <model> [n_items]     (resumable; appends to results/pilot_<model>.jsonl)
       python pilot.py analyze <model>
Gateway details are read from .env (LITELLM_BASE_URL, LITELLM_API_KEY) and never printed.
"""
import base64, csv, io, json, random, re, sys, time, concurrent.futures as cf
from pathlib import Path
import numpy as np
from PIL import Image
from text_arm import person_directed

HERE = Path(__file__).parent
OUT = HERE / 'results'
PRIMARY = {'Privacy', 'Coordination/Proactivity', 'Communication/Legibility', 'Cooperation'}
SEED, N_DEFAULT = 2028, 60
CUES = {
    'neutral': 'There is another person in the scene.',
    'neutral_repeat': 'There is another person in the scene.',
    'paraphrase': 'Another person is present in the scene.',
    'low': 'The other person in the scene appears untrustworthy.',
    'high': 'The other person in the scene appears trustworthy.',
}
LETTERS = 'ABCDE'


def env():
    vals = {}
    for line in (HERE / '.env').read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals['LITELLM_BASE_URL'], vals['LITELLM_API_KEY']


def select_items(n):
    ann = json.loads((HERE / 'data/official_annotations.json').read_text())
    text = {r['id']: r for r in csv.DictReader(open(OUT / 'text_arm_items.csv'))}
    pool = [k for k in sorted(ann)
            if ann[k]['behaviors'][ann[k]['correct']].strip()
            and set(ann[k]['taxonomy'].get(str(ann[k]['correct']), [])) & PRIMARY
            and text[k]['room_for_trait_cue'] == 'True']
    return pool, random.Random(SEED).sample(pool, n), ann


def image_b64(cid):
    prev = Path((OUT / '.preview_dir').read_text().strip())
    im = Image.open(prev / (cid + '.jpg')).convert('RGB')
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def options(v):
    """Non-empty options in original order -> (letter, original index, text)."""
    opts = [(i, b) for i, b in enumerate(v['behaviors']) if b.strip()]
    return [(LETTERS[j], i, b) for j, (i, b) in enumerate(opts)]


def prompt(v, cue):
    lines = '\n'.join(f'{l}. {b}' for l, _, b in options(v))
    return ('The image shows five frames, in order from left to right, from an egocentric video recorded '
            f'by the camera wearer. {cue} What should the camera wearer do next? Choose the most socially '
            f'appropriate action.\n\n{lines}\n\nRespond with only the letter of your choice and no explanation.')


def call(client, model, img, text):
    for attempt in range(5):
        try:
            r = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img}'}},
                {'type': 'text', 'text': text}]}])
            return r.choices[0].message.content or '', r.model
        except Exception as e:  # rate limits / transient gateway errors
            err = f'{type(e).__name__}: {str(e)[:200]}'
            time.sleep(2 ** attempt * 3)
    return None, err


def parse(reply, n_opts):
    """Accept only an explicit capital option letter; never uppercase the reply (the article 'a'
    would otherwise read as option A). Unparseable replies return None and are reported."""
    s = (reply or '').strip()
    ok = LETTERS[:n_opts]
    m = re.fullmatch(r'[\s\*\(]*([A-E])[\s\*\)\.]*', s)
    if m:
        return m.group(1) if m.group(1) in ok else None
    for pat in (r'(?:answer|option|choice)(?:\s+is)?\s*[:\-]?\s*\**\(?([A-E])\b',
                r'\*\*\(?([A-E])[\.\)]?\**', r'(?<![A-Za-z])([A-E])(?=[\.\):])'):
        hits = [h for h in re.findall(pat, s, flags=re.I if pat.startswith('(?:') else 0) if h.upper() in ok]
        if hits:
            return hits[-1].upper()
    return None


def run(model, n):
    from openai import OpenAI
    base, key = env()
    client = OpenAI(base_url=base, api_key=key, timeout=180)
    _, items, ann = select_items(n)
    path = OUT / f"pilot_{model.replace('/', '_')}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get('letter'):
                done.add((r['id'], r['condition']))
    jobs = [(cid, c) for cid in items for c in CUES if (cid, c) not in done]
    print(f'{len(jobs)} calls to run ({len(done)} already done)')
    imgs = {cid: image_b64(cid) for cid in items}

    def work(job):
        cid, cond = job
        v = ann[cid]
        reply, served = call(client, model, imgs[cid], prompt(v, CUES[cond]))
        letter = parse(reply, len(options(v)))
        return {'id': cid, 'condition': cond, 'model': model, 'served_model': served if reply is not None else None,
                'error': served if reply is None else None, 'reply': reply, 'letter': letter,
                'time': time.strftime('%Y-%m-%dT%H:%M:%S')}

    with open(path, 'a', encoding='utf-8') as fh, cf.ThreadPoolExecutor(8) as pool:
        for i, r in enumerate(pool.map(work, jobs), 1):
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} done', flush=True)


def analyze(model, n_boot=5000):
    path = OUT / f"pilot_{model.replace('/', '_')}.jsonl"
    _, items, ann = select_items(N_DEFAULT)
    res = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get('letter'):
            res[(r['id'], r['condition'])] = r['letter']
    ids = [c for c in items if all((c, k) in res for k in CUES)]
    info = {}
    for cid in ids:
        v = ann[cid]
        opts = {l: (i, b) for l, i, b in options(v)}
        info[cid] = {k: {'correct': opts[res[(cid, k)]][0] == v['correct'],
                         'pd': person_directed(opts[res[(cid, k)]][1]),
                         'letter': res[(cid, k)]} for k in CUES}
    rng = np.random.default_rng(1)

    def stat(f):
        vals = np.array([f(info[c]) for c in ids], float)
        boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
        return round(100 * vals.mean(), 1), [round(100 * x, 1) for x in np.percentile(boots, [2.5, 97.5])]

    out = {'model': model, 'items_complete': len(ids),
           'accuracy': {k: stat(lambda d, k=k: d[k]['correct']) for k in CUES},
           'flip_vs_neutral': {k: stat(lambda d, k=k: d[k]['letter'] != d['neutral']['letter'])
                               for k in CUES if k != 'neutral'},
           'person_directed_choice': {k: stat(lambda d, k=k: d[k]['pd']) for k in CUES},
           'low_minus_high_person_directed': stat(lambda d: d['low']['pd'] - d['high']['pd']),
           'low_minus_neutral_accuracy': stat(lambda d: d['low']['correct'] - d['neutral']['correct']),
           'high_minus_neutral_accuracy': stat(lambda d: d['high']['correct'] - d['neutral']['correct']),
           'unparsed_or_failed': sum(1 for line in path.read_text().splitlines() if not json.loads(line).get('letter'))}
    (OUT / f"pilot_{model.replace('/', '_')}_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    mode, model = sys.argv[1], sys.argv[2]
    if mode == 'run':
        run(model, int(sys.argv[3]) if len(sys.argv) > 3 else N_DEFAULT)
    else:
        analyze(model)
