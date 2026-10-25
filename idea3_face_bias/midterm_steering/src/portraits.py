"""Option 1 vision arm: matched OMI face pairs shown as a second image ("the other person").

Pre-registered matching rule (written before any model saw a face):
  * Trust pairs: trust(high) - trust(low) >= 15 on OMI's 0-100 scale (about 1.2 SD), and
    |happy| <= 5, |age| <= 8, |gender| <= 15, same majority race label (argmax of the 7 race ratings).
  * Equal-trust control pairs: |trust difference| <= 3 with the same calipers. They measure how much
    answers change when only the face identity changes (the face noise floor).
  * Greedy matching on the normalized caliper distance; every face is used at most once across both pair types.
  * Attractiveness and dominance gaps are not matched; they are reported (OMI trust and dominance correlate at -0.81).
Each test item gets one trust pair and one control pair, assigned by a seeded hash of the item id.

Env: OMI_DIR (default $P777/data/omi). Output: results/omi_pairs.json (committed) and $OUT_DIR/omi_pairs_sheet.jpg (images are never committed).
"""
import csv, json, os
from pathlib import Path
from PIL import Image, ImageDraw
from data import HERE, P777, seeded_int

OMI = Path(os.environ.get('OMI_DIR', f'{P777}/data/omi'))
RACES = ['asian', 'middle-eastern', 'hispanic', 'islander', 'native', 'black', 'white']
CAL = {'happy': 5, 'age': 8, 'gender': 15}
GAP, NULL_GAP = 15, 3
PORTRAIT_SIDE = 512  # 16 x 16 = 256 visual tokens


def load_faces():
    faces = {}
    for r in csv.DictReader(open(OMI / 'attribute_means.csv')):
        f = {k: float(v) for k, v in r.items() if k != 'stimulus'}
        f['race'] = max(RACES, key=lambda k: f[k])
        faces[r['stimulus']] = f
    return faces


def _greedy(faces, ok, used):
    cands = []
    for a, x in faces.items():
        for b, y in faces.items():
            if a != b and ok(x, y) and all(abs(x[k] - y[k]) <= c for k, c in CAL.items()) and x['race'] == y['race']:
                cands.append((sum(abs(x[k] - y[k]) / c for k, c in CAL.items()), a, b))
    out = []
    for d, a, b in sorted(cands):
        if a not in used and b not in used:
            used |= {a, b}
            out.append((a, b))
    return out


def build_pairs():
    faces, used = load_faces(), set()
    trust = _greedy(faces, lambda x, y: x['trustworthy'] - y['trustworthy'] >= GAP, used)
    null = _greedy(faces, lambda x, y: 0 <= x['trustworthy'] - y['trustworthy'] <= NULL_GAP, used)
    desc = lambda a, b: {k: round(faces[a][k] - faces[b][k], 1) for k in ('trustworthy', 'happy', 'age', 'gender', 'attractive', 'dominant')}
    return {'rule': {'gap': GAP, 'null_gap': NULL_GAP, 'calipers': CAL, 'same_race': True},
            'trust_pairs': [{'high': a, 'low': b, 'diff': desc(a, b)} for a, b in trust],
            'null_pairs': [{'a': a, 'b': b, 'diff': desc(a, b)} for a, b in null]}


def assign(item_id, pairs):
    t = pairs['trust_pairs'][seeded_int('omi-trust-v1', item_id) % len(pairs['trust_pairs'])]
    n = pairs['null_pairs'][seeded_int('omi-null-v1', item_id) % len(pairs['null_pairs'])]
    return t, n


def portrait(stim):
    return Image.open(OMI / 'images' / f'{stim}.jpg').convert('RGB').resize((PORTRAIT_SIDE, PORTRAIT_SIDE), Image.Resampling.LANCZOS)


def contact_sheet(pairs, path, n=16, side=160):
    tp = pairs['trust_pairs'][:n]
    sheet = Image.new('RGB', (side * 4, side * ((len(tp) + 1) // 2) + 20), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((5, 3), 'Each row: HIGH-trust | LOW-trust (two pairs per row)', fill='black')
    for i, p in enumerate(tp):
        x0, y0 = (i % 2) * side * 2, 20 + (i // 2) * side
        for j, s in enumerate((p['high'], p['low'])):
            sheet.paste(portrait(s).resize((side, side)), (x0 + j * side, y0))
        d.text((x0 + 3, y0 + 3), f"dT={p['diff']['trustworthy']} dH={p['diff']['happy']}", fill='yellow')
    sheet.save(path, quality=90)


if __name__ == '__main__':
    pairs = build_pairs()
    out = HERE / 'results/omi_pairs.json'
    out.write_text(json.dumps(pairs, indent=1))
    mean = lambda ps, k: round(sum(p['diff'][k] for p in ps) / len(ps), 1)
    for name in ('trust_pairs', 'null_pairs'):
        ps = pairs[name]
        print(name, len(ps), {k: mean(ps, k) for k in ps[0]['diff']})
    outdir = Path(os.environ.get('OUT_DIR', f'{P777}/outputs'))
    outdir.mkdir(parents=True, exist_ok=True)
    contact_sheet(pairs, outdir / 'omi_pairs_sheet.jpg')
    print('sheet:', outdir / 'omi_pairs_sheet.jpg')
