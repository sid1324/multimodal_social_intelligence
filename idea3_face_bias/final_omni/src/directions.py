"""Stage 2: per-layer trust directions per channel and level (midterm logic: mean over items and template pairs of
h[low] - h[high], h[neg] - h[pos] at the last prompt token; train split, train wordings), cross-channel and cross-level
cosines, split-half stability, valence control, figure F2 draft. CPU only.
Usage: python src/directions.py --model nemotron --level L2"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, P777, seeded_int
CACHE = Path(P777) / 'cache'
ACTS = Path(os.environ.get('ACTS_DIR', CACHE / 'acts'))  # set to a node-local copy: memmap reads on Lustre stall


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cos(a, b):
    return (unit(a) * unit(b)).sum(-1)


def pairs_for(fam, lv):
    if fam == 'app':
        return [(f'{lv}_app_low_tr{i}', f'{lv}_app_high_tr{i}') for i in range(3)]
    if fam == 'beh':
        return [(f'{lv}_beh_neg_tr{i}', f'{lv}_beh_pos_tr{i}') for i in range(2)]
    return [(f'val_neg{i}', f'val_pos{i}') for i in range(3)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='nemotron'); ap.add_argument('--level', default='L2')
    a = ap.parse_args()
    files = {}
    for part in ('text', 'other'):
        sh = [ACTS / f'g2_{a.model}_{part}_s{k}.npy' for k in (0, 1)]
        if all(p.exists() for p in sh):
            ix = [json.loads(Path(str(p) + '.index.json').read_text()) for p in sh]
            assert ix[0]['cells'] == ix[1]['cells']
            files[part] = ([np.load(p, mmap_mode='r') for p in sh], ix[0]['items'] + ix[1]['items'], {tuple(c): j for j, c in enumerate(ix[0]['cells'])})
    items = files['text'][1]
    order = sorted(range(len(items)), key=lambda i: seeded_int('split-half-v1', items[i]))
    halves = order[::2], order[1::2]

    def D(fam, lv, ch):
        """(per-item mean difference [n, L, H]); None if not collected."""
        part = 'text' if ch == 'text' else 'other'
        if part not in files:
            return None
        arrs, its, pos = files[part]
        assert its == items
        out = None
        ps = pairs_for(fam, lv)
        for x, y in ps:
            if (x, ch) not in pos or (y, ch) not in pos:
                return None
            d = np.concatenate([np.asarray(z[:, pos[(x, ch)]], np.float32) - np.asarray(z[:, pos[(y, ch)]], np.float32) for z in arrs])
            out = d if out is None else out + d
        out = out / len(ps)
        assert np.isfinite(out).all() and (np.abs(out).sum((1, 2)) > 0).all(), (fam, lv, ch, 'missing residuals')
        return out
    dirs, stab = {}, {}
    want = [('app', lv, 'text') for lv in ('L1', 'L2', 'L3')] + [('beh', lv, 'text') for lv in ('L1', 'L2', 'L3')] + \
           [(f, a.level, ch) for f in ('app', 'beh') for ch in ('speech', 'image')] + [('val', '', 'text')]
    for fam, lv, ch in want:
        d = D(fam, lv, ch)
        if d is None:
            continue
        k = f'{fam}_{ch}_{lv}'.rstrip('_')
        dirs[k] = d.mean(0); stab[k] = cos(d[list(halves[0])].mean(0), d[list(halves[1])].mean(0)).round(4).tolist()
    M = a.level
    if f'app_speech_{M}' in dirs and f'app_image_{M}' in dirs:
        dirs['app_avg_' + M] = sum(unit(dirs[f'app_{c}_{M}']) for c in ('text', 'speech', 'image')) / 3
    np.savez(CACHE / f'directions_{a.model}.npz', **dirs)
    C = {}
    def add(name, x, y):
        if x in dirs and y in dirs:
            C[name] = cos(dirs[x], dirs[y]).round(4).tolist()
    add('app text-speech', f'app_text_{M}', f'app_speech_{M}'); add('app text-image', f'app_text_{M}', f'app_image_{M}')
    add('app speech-image', f'app_speech_{M}', f'app_image_{M}')
    for c in ('text', 'speech', 'image'):
        add(f'app-beh {c}', f'app_{c}_{M}', f'beh_{c}_{M}')
    add('app text - valence', f'app_text_{M}', 'val_text'); add('beh text - valence', f'beh_text_{M}', 'val_text')
    for x, y in (('L1', 'L2'), ('L1', 'L3'), ('L2', 'L3')):
        add(f'app text {x}-{y}', f'app_text_{x}', f'app_text_{y}')
    add('app-beh text L1', 'app_text_L1', 'beh_text_L1'); add('app-beh text L3', 'app_text_L3', 'beh_text_L3')
    norms = {k: np.linalg.norm(v, axis=-1).round(3).tolist() for k, v in dirs.items()}
    (HERE / f'results/directions_{a.model}.json').write_text(json.dumps({'level': M, 'n_items': len(items), 'cos': C, 'split_half': stab, 'norm': norms}))
    nL = len(next(iter(C.values())))
    late = slice(round(nL * 0.55), nL - 2)
    for k, v in C.items():
        print(f'{k:22s} mean late layers {np.mean(v[late]):.2f}  max {max(v):.2f} at L{int(np.argmax(v))}')
    for k, v in stab.items():
        print(f'stability {k:18s} min over late layers {min(v[late]):.2f}')


if __name__ == '__main__':
    main()
