"""Local: score midterm_direct_eval.py replies against the lora_pilot answer key.

Per model and condition: joint / action / justification accuracy and parse rate. Contrasts on items both
conditions answered: exact McNemar on joint correctness and a paired bootstrap CI.
Writes output/idea5-midterm/direct_eval/scored.json.

usage: midterm_direct_score.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
MID = ROOT / 'output/idea5-midterm'
OUT = MID / 'direct_eval'
CONDS = ['rgb_cot', 'rgb_spcot', 'blind', 'desc_only', 'desc_only_flip', 'desc_only_para', 'rgb', 'rgb_x4', 'maps', 'maps_wrong', 'desc', 'desc_wrong', 'rgb_ctx', 'rgb_ctx_wrong', 'rgb_forced', 'maps_forced', 'maps_wrong_forced', 'rgb_ctx_forced']
CONTRASTS = [('maps', 'rgb'), ('maps', 'maps_wrong'), ('desc', 'rgb'), ('desc', 'desc_wrong'), ('desc', 'maps'),
             ('rgb', 'blind'), ('desc_only', 'blind'), ('rgb', 'desc_only'), ('maps', 'rgb_x4'), ('rgb_x4', 'rgb'), ('desc_only', 'desc_only_flip'), ('desc_only_para', 'desc_only_flip'), ('desc_only', 'desc_only_para'), ('rgb_spcot', 'rgb_cot'), ('rgb_cot', 'rgb'), ('rgb_spcot', 'rgb'), ('rgb_ctx', 'rgb'), ('rgb_ctx', 'rgb_ctx_wrong'), ('rgb_ctx', 'maps'), ('maps_forced', 'rgb_forced'), ('maps_forced', 'maps_wrong_forced'), ('rgb_ctx_forced', 'rgb_forced')]
CATEGORIES = ['Proxemics', 'Privacy', 'Safety', 'Politeness', 'Cooperation', 'Coordination/Proactivity', 'Communication/Legibility']


def score(path, key):
    out = {}
    for f in path.glob('*.json'):
        r = json.loads(f.read_text())
        k = key[r['id']]
        act = k['action_order'][r['action_position'] - 1] if r['action_position'] else None
        just = k['justification_order'][r['justification_position'] - 1] if r['justification_position'] else None
        out[r['id']] = {'action': int(act == k['gold']), 'justification': int(just == k['gold']),
                        'joint': int(act == k['gold'] and just == k['gold']), 'parsed': r['action_position'] is not None}
    return out


def contrast(a, b, ids, seed=0):
    x, y = np.array([a[i]['joint'] for i in ids]), np.array([b[i]['joint'] for i in ids])
    fixed, broken = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    rng = np.random.default_rng(seed)
    boots = [(x[s] - y[s]).mean() for s in (rng.integers(0, len(ids), len(ids)) for _ in range(10000))]
    return {'diff': float((x - y).mean()), 'ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'fixed': fixed, 'broken': broken, 'p': float(binomtest(fixed, fixed + broken).pvalue) if fixed + broken else 1.0,
            'n': len(ids)}


def main():
    key = json.loads((MID / 'lora_pilot/answer_key.json').read_text())
    result = {}
    for model_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        res = {c: score(model_dir / c, key) for c in CONDS if (model_dir / c).exists()}
        summary = {c: {m: float(np.mean([v[m] for v in r.values()])) for m in ('joint', 'action', 'justification', 'parsed')}
                   | {'n': len(r)} for c, r in res.items() if r}
        contrasts = {}
        for x, y in CONTRASTS:
            if x in res and y in res:
                ids = sorted(set(res[x]) & set(res[y]))
                if ids:
                    contrasts[f'{x}-{y}'] = contrast(res[x], res[y], ids)
                    contrasts[f'{x}-{y}']['by_category'] = {
                        cat: contrast(res[x], res[y], [i for i in ids if cat in key[i]['categories']])
                        for cat in CATEGORIES if any(cat in key[i]['categories'] for i in ids)}
        result[model_dir.name] = {'summary': summary, 'contrasts': contrasts}
        print(f'== {model_dir.name}')
        for c, s in summary.items():
            print(f"  {c:11s} joint {s['joint']:.3f} act {s['action']:.3f} just {s['justification']:.3f} parsed {s['parsed']:.2f} n={s['n']}")
        for name, c in contrasts.items():
            print(f"  {name:16s} {100 * c['diff']:+.1f} [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}] "
                  f"fixed {c['fixed']} broken {c['broken']} p={c['p']:.3f} n={c['n']}")
    (OUT / 'scored.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
