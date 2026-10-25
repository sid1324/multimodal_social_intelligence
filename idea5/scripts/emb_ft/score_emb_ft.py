"""Local: score the end-to-end embedding fine-tune (emb_ft.py on Runpod) against the locally held EgoNormia labels.

Reads output/idea5-midterm/emb_ft/{<cond>_s<seed>,zeroshot_<cond>}.json (option scores per test item). The primary
prediction is the argmax over non-empty options, as in midterm_scorer.py's zero-shot ("None" is never gold on test);
the unmasked argmax is reported as well. Contrasts: exact McNemar on joint correctness and an item-bootstrap 95% CI,
seeds pooled pairwise. Writes output/idea5-midterm/emb_ft/scored.json.

usage: score_emb_ft.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
MID = ROOT / 'output/idea5-midterm'
OUT = MID / 'emb_ft'
B = 10000


def correct(path, data, masked=True):
    rows = json.loads(path.read_text())['rows']
    out = {}
    for i, r in rows.items():
        pick = {}
        for head in ('action', 'justification'):
            s = np.array(r[head]) - (1e9 * np.array(r[f'empty_{head}']) if masked else 0)
            pick[head] = int(s.argmax())
        g = data[i]['correct']
        out[i] = {'joint': int(pick['action'] == g and pick['justification'] == g), 'action': int(pick['action'] == g),
                  'justification': int(pick['justification'] == g)}
    return out


def contrast(pairs, ids):
    x = np.array([[a[i]['joint'] for i in ids] for a, _ in pairs], float)
    y = np.array([[b[i]['joint'] for i in ids] for _, b in pairs], float)
    fixed, broken = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    idx = np.random.default_rng(0).integers(0, len(ids), (B, len(ids)))
    diffs = x[:, idx].mean(axis=(0, 2)) - y[:, idx].mean(axis=(0, 2))
    return {'pairs': len(pairs), 'n_items': len(ids), 'acc_a': float(x.mean()), 'acc_b': float(y.mean()), 'diff': float(x.mean() - y.mean()),
            'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], 'fixed': fixed, 'broken': broken,
            'p': binomtest(fixed, fixed + broken, .5).pvalue if fixed + broken else 1.0}


def main():
    data = json.loads((ROOT / 'data/EgoNormia/annotations/final_data.json').read_text())
    result = {'runs': {}, 'contrasts': {}, 'reference': {}}
    for masked in (True, False):
        tag = 'masked' if masked else 'unmasked'
        runs = {p.stem: correct(p, data, masked) for p in sorted(OUT.glob('*.json')) if p.stem != 'scored'}
        ids = sorted(set.intersection(*(set(r) for r in runs.values())))
        for name, rows in runs.items():
            result['runs'].setdefault(name, {})[tag] = {k: float(np.mean([rows[i][k] for i in ids])) for k in ('joint', 'action', 'justification')}
        seeds = lambda c: [runs[n] for n in sorted(runs) if n.startswith(c + '_s')]  # noqa: E731
        plan = {'C5-C2': (seeds('C5'), seeds('C2')), 'C5-X1': (seeds('C5'), seeds('X1')), 'X1-C2': (seeds('X1'), seeds('C2'))}
        for c in ('C2', 'C5', 'X1'):
            if f'zeroshot_{c}' in runs:
                plan[f'{c}ft-{c}zs'] = (seeds(c), [runs[f'zeroshot_{c}']] * len(seeds(c)))
        if 'zeroshot_C5' in runs and 'zeroshot_C2' in runs:
            plan['C5zs-C2zs'] = ([runs['zeroshot_C5']], [runs['zeroshot_C2']])
        for name, (a, b) in plan.items():
            n = min(len(a), len(b))
            if n:
                result['contrasts'].setdefault(name, {})[tag] = contrast(list(zip(a[:n], b[:n])), ids)
        result['n_items'] = len(ids)
    joint = lambda r: float(np.mean([v['action'] == v['gold'] and v['justification'] == v['gold'] for v in r.values()]))  # noqa: E731
    for c in ('C2', 'C5', 'X1'):
        for variant in ('full', 'lowrank'):
            res = json.loads((MID / f'results/{c}_{variant}.json').read_text())
            result['reference'][f'{c}_{variant}'] = {'test': joint(res['test']), 'zero_shot': joint(res['zero_shot'])}
    (OUT / 'scored.json').write_text(json.dumps(result, indent=1))
    for name, v in result['runs'].items():
        print(f"{name:14s} joint {v['masked']['joint']:.3f} (unmasked {v['unmasked']['joint']:.3f})  action {v['masked']['action']:.3f}  "
              f"just {v['masked']['justification']:.3f}")
    for name, v in result['contrasts'].items():
        c = v['masked']
        print(f"{name:11s} pairs={c['pairs']} {100 * c['diff']:+.1f} [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}] "
              f"fixed {c['fixed']} broken {c['broken']} p={c['p']:.3g}   unmasked {100 * v['unmasked']['diff']:+.1f}")
    for k, v in result['reference'].items():
        print(f"reference {k:12s} test {v['test']:.3f} zero-shot {v['zero_shot']:.3f}")


if __name__ == '__main__':
    main()
