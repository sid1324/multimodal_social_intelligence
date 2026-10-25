"""Local: score the LoRA pilot (base vs spatial-LoRA vs control-LoRA) against the locally held answer key.

Reads output/idea5-midterm/lora_pilot/results/{base,spatial,control}.json (copied back from Bridges-2) and
answer_key.json; writes scored.json with SAT accuracy, EgoNormia accuracy, paired contrasts and categories.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/idea5-midterm/lora_pilot'
MODELS = ['base', 'spatial', 'control']
CATEGORIES = ['Proxemics', 'Privacy', 'Safety', 'Politeness', 'Cooperation', 'Coordination/Proactivity', 'Communication/Legibility']
B = 10000


def egonormia(rows, key):
    out = {}
    for r in rows:
        k = key[r['id']]
        act = k['action_order'][r['action_position'] - 1] if r['action_position'] else None
        just = k['justification_order'][r['justification_position'] - 1] if r['justification_position'] else None
        out[r['id']] = {'action': int(act == k['gold']), 'justification': int(just == k['gold']),
                        'joint': int(act == k['gold'] and just == k['gold']), 'parsed': r['action_position'] is not None}
    return out


def sat_circular(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['id']].append(r['correct'])
    return float(np.mean([all(v) for v in by.values()])), float(np.mean([r['correct'] for r in rows if r['shift'] == 0])), len(by)


def contrast(a, b, ids):
    x, y = np.array([a[i]['joint'] for i in ids]), np.array([b[i]['joint'] for i in ids])
    fixed, broken = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), (B, len(ids)))
    diffs = x[idx].mean(1) - y[idx].mean(1)
    return {'n': len(ids), 'acc_a': float(x.mean()), 'acc_b': float(y.mean()), 'diff': float(x.mean() - y.mean()),
            'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], 'fixed': fixed, 'broken': broken,
            'p': binomtest(fixed, fixed + broken, .5).pvalue if fixed + broken else 1.0}


def main():
    key = json.loads((OUT / 'answer_key.json').read_text())
    res = {m: json.loads((OUT / 'results' / f'{m}.json').read_text()) for m in MODELS if (OUT / 'results' / f'{m}.json').exists()}
    ego = {m: egonormia(r['egonormia'], key) for m, r in res.items()}
    out = {'models': {}, 'contrasts': {}, 'categories': {}}
    for m, r in res.items():
        circ, first, n_test = sat_circular(r['sat_test'])
        e = ego[m]
        out['models'][m] = {'sat_val': float(np.mean([x['correct'] for x in r['sat_val']])), 'n_sat_val': len(r['sat_val']),
                            'sat_test_circular': circ, 'sat_test_first_order': first, 'n_sat_test': n_test,
                            'ego_joint': float(np.mean([v['joint'] for v in e.values()])),
                            'ego_action': float(np.mean([v['action'] for v in e.values()])),
                            'ego_justification': float(np.mean([v['justification'] for v in e.values()])),
                            'ego_parsed': int(sum(v['parsed'] for v in e.values())), 'n_ego': len(e)}
    ids = sorted(key)
    for a, b in [('spatial', 'control'), ('spatial', 'base'), ('control', 'base')]:
        if a in ego and b in ego:
            out['contrasts'][f'{a}-{b}'] = contrast(ego[a], ego[b], ids)
            out['categories'][f'{a}-{b}'] = {c: contrast(ego[a], ego[b], [i for i in ids if c in key[i]['categories']]) for c in CATEGORIES}
    (OUT / 'scored.json').write_text(json.dumps(out, indent=1))
    for m, v in out['models'].items():
        print(f"{m:8} SAT val {v['sat_val']:.3f}  SAT real circ {v['sat_test_circular']:.3f} (first-order {v['sat_test_first_order']:.3f})  "
              f"Ego joint {v['ego_joint']:.3f} act {v['ego_action']:.3f} just {v['ego_justification']:.3f} parsed {v['ego_parsed']}/{v['n_ego']}")
    for name, c in out['contrasts'].items():
        print(f"{name}: {100 * c['diff']:+.1f} [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}] fixed {c['fixed']} broken {c['broken']} p={c['p']:.3f}")
        for cat, d in out['categories'][name].items():
            print(f"   {cat[:14]:14} n={d['n']:3} {100 * d['diff']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] p={d['p']:.3f}")


if __name__ == '__main__':
    main()
