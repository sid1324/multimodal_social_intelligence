"""Local: score the follow-up arms against the locally held answer key and test spatial vs (broad, narrow) controls.

Arms: spatial_s0 / control_s0 (first pilot: results/{spatial,control}.json + results/sat_static_heldout.json),
spatial_s2 / control_s2 / color_s0 (results/<name>.json from eval_arm.py), plus base for reference.
Per arm: held-out SAT static accuracy; EgoNormia joint, action, justification, and consistency (the chosen justification
is the one written for the chosen action). Contrasts: exact McNemar on EgoNormia joint correctness (and on consistency),
pooling the paired items over seeds; item-bootstrap 95% CI of the pooled difference.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/idea5-midterm/lora_pilot'
B = 10000


def load_arms():
    static0 = json.loads((OUT / 'results/sat_static_heldout.json').read_text())
    arms = {}
    for name, src, st in [('base', 'base', 'base'), ('spatial_s0', 'spatial', 'spatial'), ('control_s0', 'control', 'control')]:
        r = json.loads((OUT / 'results' / f'{src}.json').read_text())
        arms[name] = {'egonormia': r['egonormia'], 'sat_static': static0[st]}
    for name in ['spatial_s2', 'control_s2', 'color_s0']:
        path = OUT / 'results' / f'{name}.json'
        if path.exists():
            arms[name] = json.loads(path.read_text())
    return arms


def score(arm, key):
    rows = {}
    for r in arm['egonormia']:
        k = key[r['id']]
        a = k['action_order'][r['action_position'] - 1] if r['action_position'] else None
        j = k['justification_order'][r['justification_position'] - 1] if r['justification_position'] else None
        rows[r['id']] = {'action': int(a == k['gold']), 'justification': int(j == k['gold']),
                         'joint': int(a == k['gold'] and j == k['gold']), 'consistent': int(a is not None and a == j)}
    return rows


def pooled(pairs, field):
    x = np.concatenate([np.array([a[i][field] for i in sorted(a)]) for a, _ in pairs])
    y = np.concatenate([np.array([b[i][field] for i in sorted(a)]) for a, b in pairs])
    fixed, broken = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    rng = np.random.default_rng(0)
    n = len(pairs[0][0])
    # Resample items (the same item set across seeds), keeping seeds paired.
    idx = rng.integers(0, n, (B, n))
    stack_x, stack_y = x.reshape(len(pairs), n), y.reshape(len(pairs), n)
    diffs = stack_x[:, idx].mean(axis=(0, 2)) - stack_y[:, idx].mean(axis=(0, 2))
    return {'pairs': len(pairs), 'n': int(x.size), 'acc_a': float(x.mean()), 'acc_b': float(y.mean()), 'diff': float(x.mean() - y.mean()),
            'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], 'fixed': fixed, 'broken': broken,
            'p_mcnemar_exact': binomtest(fixed, fixed + broken, .5).pvalue if fixed + broken else 1.0}


def main():
    key = json.loads((OUT / 'answer_key.json').read_text())
    arms = load_arms()
    scored = {name: score(arm, key) for name, arm in arms.items()}
    out = {'arms': {}, 'two_seed_means': {}, 'contrasts': {}}
    for name, arm in arms.items():
        s = scored[name]
        out['arms'][name] = {'sat_static': float(np.mean([r['correct'] for r in arm['sat_static']])),
                             **{f: float(np.mean([v[f] for v in s.values()])) for f in ('joint', 'action', 'justification')},
                             'consistent_count': int(sum(v['consistent'] for v in s.values())), 'n': len(s)}
    for family in ('spatial', 'control'):
        names = [f'{family}_s0', f'{family}_s2']
        if all(n in out['arms'] for n in names):
            out['two_seed_means'][family] = {f: float(np.mean([out['arms'][n][f] for n in names]))
                                             for f in ('sat_static', 'joint', 'action', 'justification', 'consistent_count')}
    seeds = [s for s in ('s0', 's2') if f'spatial_{s}' in scored]
    comparisons = {
        'spatial-control': [(scored[f'spatial_{s}'], scored[f'control_{s}']) for s in seeds if f'control_{s}' in scored],
        'spatial-color': [(scored[f'spatial_{s}'], scored['color_s0']) for s in seeds if 'color_s0' in scored],
        'color-base': [(scored['color_s0'], scored['base'])] if 'color_s0' in scored else [],
        'control-base': [(scored[f'control_{s}'], scored['base']) for s in seeds if f'control_{s}' in scored],
        'spatial-base': [(scored[f'spatial_{s}'], scored['base']) for s in seeds],
    }
    for name, pairs in comparisons.items():
        if pairs:
            out['contrasts'][name] = {'joint': pooled(pairs, 'joint'), 'consistent': pooled(pairs, 'consistent')}
            if len(pairs) > 1:
                out['contrasts'][name]['per_seed_joint'] = [pooled([p], 'joint') for p in pairs]
    (OUT / 'scored_followup.json').write_text(json.dumps(out, indent=1))
    print(f"{'arm':12} {'SATstat':>7} {'joint':>6} {'action':>6} {'just':>6} {'consist':>7}")
    for name, v in out['arms'].items():
        print(f"{name:12} {v['sat_static']:7.3f} {v['joint']:6.3f} {v['action']:6.3f} {v['justification']:6.3f} {v['consistent_count']:5d}/{v['n']}")
    for fam, v in out['two_seed_means'].items():
        print(f"mean {fam:7} {v['sat_static']:7.3f} {v['joint']:6.3f} {v['action']:6.3f} {v['justification']:6.3f} {v['consistent_count']:7.1f}")
    for name, c in out['contrasts'].items():
        for field in ('joint', 'consistent'):
            d = c[field]
            print(f"{name:16} {field:10} pairs={d['pairs']} {100 * d['diff']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] "
                  f"fixed {d['fixed']} broken {d['broken']} exact McNemar p={d['p_mcnemar_exact']:.3g}")
        for k, d in enumerate(c.get('per_seed_joint', [])):
            print(f"{'':16} joint seed{('0', '2')[k]}  {100 * d['diff']:+.1f} fixed {d['fixed']} broken {d['broken']} p={d['p_mcnemar_exact']:.3g}")


if __name__ == '__main__':
    main()
