"""Tables, contrasts and the spatial probe for the Idea 5 midterm (PLAN.md sections 0, 5, 6).

Reads results/<cond>.json from midterm_scorer.py and emb/<cond>/ vectors; writes results/stats.json.
Contrasts are paired over items: exact McNemar on joint correctness and a 95% bootstrap interval of the
difference (items resampled for test; source videos resampled for the 800-item CV). Holm over the secondary
contrasts and, separately, over the category contrasts.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/EgoNormia'
OUT = ROOT / 'output/idea5-midterm'
RES = OUT / 'results'
CONDS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'X1', 'X2', 'X3']
PRIMARY = ('C5', 'C2')
SECONDARY = [('C4', 'C2'), ('C3', 'C2'), ('C5', 'X1'), ('C5', 'X2'), ('C5', 'X3'), ('C2', 'C1'), ('C1', 'C0')]
CATEGORIES = ['Proxemics', 'Privacy', 'Safety', 'Politeness', 'Cooperation', 'Coordination/Proactivity', 'Communication/Legibility']
PROBE_LABELS = ['n_people', 'nearest_dist', 'facing', 'hands']
B = 10000


def joint(rows):
    return {i: int(r['action'] == r['gold'] and r['justification'] == r['gold']) for i, r in rows.items()}


def mcnemar(a, b):
    fixed, broken = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(fixed, fixed + broken, .5).pvalue if fixed + broken else 1.0
    return fixed, broken, p


def bootstrap(a, b, groups, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs = np.empty(B)
    for k in range(B):
        take = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])
        diffs[k] = a[take].mean() - b[take].mean()
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def contrast(res, x, y, part, ids=None, grouped=False):
    jx, jy = joint(res[x][part]), joint(res[y][part])
    common = [i for i in jx if i in jy and (ids is None or i in ids)]
    a, b = np.array([jx[i] for i in common]), np.array([jy[i] for i in common])
    groups = np.array([i.split('_')[0] if grouped else i for i in common])
    fixed, broken, p = mcnemar(a, b)
    return {'a': x, 'b': y, 'n': len(common), 'acc_a': float(a.mean()), 'acc_b': float(b.mean()),
            'diff': float(a.mean() - b.mean()), 'ci95': bootstrap(a, b, groups), 'fixed': fixed, 'broken': broken, 'p': p}


def holm(rows):
    order = sorted(range(len(rows)), key=lambda k: rows[k]['p'])
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, rows[k]['p'] * (len(rows) - rank)))
        rows[k]['p_holm'] = running
    return rows


def probe(cond, ids, labels):
    from midterm_scorer import folds  # same grouped folds as the scorer
    keep = [i for i in ids if (OUT / 'emb' / cond / f'{i}.npy').exists()]
    X = np.stack([np.load(OUT / 'emb' / cond / f'{i}.npy') for i in keep]).astype(np.float32)
    f = folds(keep)
    scores = {}
    for name in PROBE_LABELS:
        y = np.array([str(labels[i][name]) for i in keep])
        pred = np.empty_like(y)
        for k in range(5):
            tr, te = f != k, f == k
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
        scores[name] = float(balanced_accuracy_score(y, pred))
    scores['mean'] = float(np.mean([scores[n] for n in PROBE_LABELS]))
    scores['n'] = len(keep)
    return scores


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    data = json.loads((DATA / 'annotations/final_data.json').read_text())
    split = json.loads((OUT / 'splits.json').read_text())
    variant = sys.argv[1] if len(sys.argv) > 1 else 'full'
    res = {c: json.loads((RES / f'{c}_{variant}.json').read_text()) for c in CONDS if (RES / f'{c}_{variant}.json').exists()}
    have = lambda pair: all(c in res for c in pair)  # noqa: E731
    out = {'conditions': {}, 'primary': {}, 'secondary': {}, 'categories': {}, 'probe': {}}
    for c, r in res.items():
        jt, jc, jz = joint(r['test']), joint(r['cv']), joint(r['zero_shot'])
        out['conditions'][c] = {
            'test_joint': float(np.mean(list(jt.values()))), 'n_test': len(jt),
            'test_action': float(np.mean([v['action'] == v['gold'] for v in r['test'].values()])),
            'test_justification': float(np.mean([v['justification'] == v['gold'] for v in r['test'].values()])),
            'cv_joint': float(np.mean(list(jc.values()))), 'n_cv': len(jc),
            'zero_shot_joint': float(np.mean(list(jz.values()))), 'lambda': r['lambda']}
    for part, grouped in (('test', False), ('cv', True)):
        if have(PRIMARY):
            out['primary'][part] = contrast(res, *PRIMARY, part, grouped=grouped)
        out['secondary'][part] = holm([contrast(res, x, y, part, grouped=grouped) for x, y in SECONDARY if have((x, y))])
        if have(PRIMARY):
            rows = []
            for cat in CATEGORIES:
                ids = {i for i in res['C5'][part] if cat in data[i]['taxonomy'].get(str(data[i]['correct']), [])}
                rows.append({'category': cat, **contrast(res, *PRIMARY, part, ids=ids, grouped=grouped)})
            out['categories'][part] = holm(rows)
    labels_path = OUT / 'spatial_labels.json'
    if labels_path.exists() and '--probe' in sys.argv:
        labels = json.loads(labels_path.read_text())
        ids = split['test'] + split['train'] + split['train_ext']
        for c in CONDS:
            if (OUT / 'emb' / c).exists() and c != 'C0':
                out['probe'][c] = probe(c, ids, labels)
    (RES / f'stats_{variant}.json').write_text(json.dumps(out, indent=1))

    print(f"{'cond':5} {'test':>6} {'act':>6} {'just':>6} {'cv':>6} {'zero':>6} {'probe':>6}")
    for c in CONDS:
        if c in out['conditions']:
            r = out['conditions'][c]
            pr = out['probe'].get(c, {}).get('mean', float('nan'))
            print(f"{c:5} {r['test_joint']:6.3f} {r['test_action']:6.3f} {r['test_justification']:6.3f} {r['cv_joint']:6.3f} {r['zero_shot_joint']:6.3f} {pr:6.3f}")
    for part in ('test', 'cv'):
        rows = ([out['primary'][part]] if part in out['primary'] else []) + out['secondary'][part]
        for r in rows:
            print(f"{part:4} {r['a']}-{r['b']}: {100 * r['diff']:+.1f} [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}] "
                  f"fixed {r['fixed']} broken {r['broken']} p={r['p']:.3f} holm={r.get('p_holm', r['p']):.3f} n={r['n']}")
        for r in out['categories'].get(part, []):
            print(f"{part:4}   {r['category'][:12]:12} {100 * r['diff']:+.1f} [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}] "
                  f"n={r['n']} holm={r['p_holm']:.3f}")


if __name__ == '__main__':
    main()
