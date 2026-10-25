"""Local: score the GRPO pilot against the locally held answer key and test-item spatial labels.

Reads output/idea5-midterm/rl_pilot/results/<name>.json (raw replies from eval_rl.py), lora_pilot/answer_key.json
and spatial_labels_all.json (test ids only used here, never uploaded). Writes rl_pilot/scored.json and prints:
per run joint/action/justification accuracy, parse-failure rate, spatial-block accuracy per field; contrasts with exact
McNemar on joint correctness and an item-bootstrap 95% CI (seeds pooled pairwise when present); per-category deltas;
train-reward summaries from adapters/<name>/train_log.jsonl.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
import rl_common as rc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MID = ROOT / 'output/idea5-midterm'
OUT = MID / 'rl_pilot'
CATEGORIES = ['Proxemics', 'Privacy', 'Safety', 'Politeness', 'Cooperation', 'Coordination/Proactivity', 'Communication/Legibility']
FIELDS = ['n_people', 'nearest_person', 'facing', 'hands_visible']
B = 10000


def score_run(name, key, labels):
    res = json.loads((OUT / 'results' / f'{name}.json').read_text())
    rows = {}
    for r in res['rows']:
        k = key[r['id']]
        order = {'action': k['action_order'], 'justification': k['justification_order']}
        _, parsed, a_ok, j_ok = rc.norm_score(r['reply'], order, k['gold'])
        row = {'joint': int(a_ok and j_ok), 'action': int(a_ok), 'justification': int(j_ok), 'parsed': int(parsed)}
        if res['format'] == 'spatial':
            block, lab = rc.parse_spatial(r['reply']), labels.get(r['id'])
            row['spatial_parsed'] = int(block is not None)
            row['spatial'] = rc.spatial_score(block, lab)
            for f in FIELDS:
                # Per-field correctness: a block holding only this field; the blanked fields always score wrong.
                only = {g: (block or {}).get(g) if g == f else '__' for g in FIELDS}  # '__' never matches a label
                row[f'sp_{f}'] = int(block is not None and rc.spatial_score(only, lab) > 0)
        rows[r['id']] = row
    return res['format'], rows


def contrast(pairs, ids):
    x = np.concatenate([[a[i]['joint'] for i in ids] for a, _ in pairs]).astype(float)
    y = np.concatenate([[b[i]['joint'] for i in ids] for _, b in pairs]).astype(float)
    fixed, broken = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = len(ids)
    sx, sy = x.reshape(len(pairs), n), y.reshape(len(pairs), n)
    idx = np.random.default_rng(0).integers(0, n, (B, n))
    diffs = sx[:, idx].mean(axis=(0, 2)) - sy[:, idx].mean(axis=(0, 2))
    return {'pairs': len(pairs), 'n_items': n, 'acc_a': float(x.mean()), 'acc_b': float(y.mean()), 'diff': float(x.mean() - y.mean()),
            'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], 'fixed': fixed, 'broken': broken,
            'p': binomtest(fixed, fixed + broken, .5).pvalue if fixed + broken else 1.0}


def curves(name):
    path = OUT / 'adapters' / name / 'train_log.jsonl'
    if not path.exists():
        return None
    log = [json.loads(line) for line in path.open()]
    window = lambda xs: [float(np.mean(xs[max(0, i - 19):i + 1])) for i in range(len(xs))]  # noqa: E731
    keys = ('reward', 'r_norm', 'r_spatial', 'parsed', 'joint', 'len', 'kl', 'frac_flat')
    smooth = {k: window([r[k] for r in log]) for k in keys}
    pick = sorted({0, len(log) // 4, len(log) // 2, 3 * len(log) // 4, len(log) - 1})
    return {'steps': len(log), 'at': [log[i]['step'] for i in pick], **{k: [round(smooth[k][i], 3) for i in pick] for k in keys},
            'sec_per_step': float(np.mean([r['sec'] for r in log]))}


def main():
    key = json.loads((MID / 'lora_pilot/answer_key.json').read_text())
    labels_path = MID / 'spatial_labels_all.json'
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    names = sorted(p.stem for p in (OUT / 'results').glob('*.json') if not p.stem.startswith('smoke'))
    runs = {n: score_run(n, key, labels) for n in names}
    ids = sorted(key)
    out = {'runs': {}, 'contrasts': {}, 'categories': {}, 'curves': {}}
    print(f"{'run':14} {'joint':>6} {'action':>6} {'just':>6} {'parsefail':>9} {'spatial':>7} " + ' '.join(f'{f[:8]:>8}' for f in FIELDS))
    for n, (fmt, rows) in runs.items():
        v = {f: float(np.mean([rows[i][f] for i in ids if i in rows])) for f in ('joint', 'action', 'justification', 'parsed')}
        v['parse_fail'] = 1 - v.pop('parsed')
        v['n'] = len(rows)
        if fmt == 'spatial':
            v['spatial'] = float(np.mean([r['spatial'] for r in rows.values()]))
            v['spatial_parsed'] = float(np.mean([r['spatial_parsed'] for r in rows.values()]))
            for f in FIELDS:
                v[f'sp_{f}'] = float(np.mean([r[f'sp_{f}'] for r in rows.values()]))
        out['runs'][n] = v
        sp = f"{v['spatial']:7.3f} " + ' '.join(f"{v[f'sp_{f}']:8.3f}" for f in FIELDS) if fmt == 'spatial' else ''
        print(f"{n:14} {v['joint']:6.3f} {v['action']:6.3f} {v['justification']:6.3f} {v['parse_fail']:9.3f} {sp}")
    seeds = lambda arm: sorted(n for n in runs if n.startswith(arm + '_s'))  # noqa: E731
    def pairs(a, b):
        A, Bn = seeds(a) if not a.startswith('base') else [a], seeds(b) if not b.startswith('base') else [b]
        if not A or not Bn:
            return []
        if len(A) == len(Bn):
            return [(runs[x][1], runs[y][1]) for x, y in zip(A, Bn)]
        return [(runs[x][1], runs[y][1]) for x in A for y in Bn][:max(len(A), len(Bn))] if len(Bn) == 1 or len(A) == 1 else []
    plan = [('R1', 'R0'), ('R2', 'R0'), ('R1', 'R2'), ('R3', 'base_spatial'), ('R0', 'base_social'),
            ('R1', 'base_spatial'), ('R2', 'base_spatial'), ('base_spatial', 'base_social')]
    for a, b in plan:
        p = pairs(a, b)
        if not p:
            continue
        c = contrast(p, ids)
        out['contrasts'][f'{a}-{b}'] = c
        out['categories'][f'{a}-{b}'] = {cat: contrast(p, [i for i in ids if cat in key[i]['categories']]) for cat in CATEGORIES}
        print(f"{a}-{b:13} pairs={c['pairs']} {100 * c['diff']:+.1f} [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}] "
              f"fixed {c['fixed']} broken {c['broken']} exact McNemar p={c['p']:.3g}")
        print('      ' + '  '.join(f"{cat[:6]} {100 * d['diff']:+.0f}" for cat, d in out['categories'][f'{a}-{b}'].items()))
    for n in runs:
        cv = curves(n)
        if cv:
            out['curves'][n] = cv
            print(f"curve {n}: steps {cv['steps']} at {cv['at']} r_norm {cv['r_norm']} r_spatial {cv['r_spatial']} "
                  f"parsed {cv['parsed']} kl {cv['kl']} len {cv['len']} {cv['sec_per_step']:.0f}s/step")
    (OUT / 'scored.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
