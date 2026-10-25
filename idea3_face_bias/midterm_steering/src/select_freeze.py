"""Gate metrics and the pre-registered validation selection rule (docs/03_MIDTERM_PLAN.md, Step 3c).

Metrics per method (flip = argmax differs between the two conditions of a pair):
  Acc_clean: accuracy on 'none' over all items with a non-empty gold
  S_0:       flip rate neutral vs neutral_paraphrase (wording noise floor)
  S_app:     flip rate app_low_tr{i} vs app_high_tr{i}, averaged over the 3 train templates
  R_beh:     flip rate beh_neg_tr{j} vs beh_pos_tr{j}, averaged over the 2 train templates
  dlogp_app: mean |log p(gold|low) - log p(gold|high)|; dlogp_0: same for neutral vs paraphrase
CIs: 95% percentile bootstrap, 5,000 replicates, resampling source videos.

Selection (per direction type, written before any sweep result): lowest S_app subject to
(i) Acc_clean >= Acc_clean(base) - 0.02 and (ii) R_beh >= 0.8 * R_beh(base). Ties: smaller alpha, then fewer
layers. If none qualifies: lowest S_app subject to (i) only, flagged.

Usage: python src/select_freeze.py gate     -> results/gate/gate_summary.json
       python src/select_freeze.py sweep    -> results/sweep/sweep_table.json + results/FREEZE_proposal.json
"""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE

APP = [(f'app_low_tr{i}', f'app_high_tr{i}') for i in range(3)]
BEH = [(f'beh_neg_tr{j}', f'beh_pos_tr{j}') for j in range(2)]


def load_rows(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        return [json.loads(l) for l in fh]


def per_item(rows):
    """{method: {item: {condition: row}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        out[r['method']][r['id']][r['condition']] = r
    return out


def item_stats(d):
    """Per-item metric values (None when the item lacks the needed conditions)."""
    def flip(a, b):
        return float(d[a]['pred'] != d[b]['pred']) if a in d and b in d else None
    def dlp(a, b):
        if a in d and b in d:
            g = str(d[a]['gold'])
            return abs(d[a]['logp'][g] - d[b]['logp'][g])
        return None
    mean = lambda xs: (sum(xs) / len(xs)) if xs and None not in xs else None
    return {'acc': float(d['none']['correct']) if 'none' in d else None,
            's0': flip('neutral', 'neutral_paraphrase'),
            's_app': mean([flip(a, b) for a, b in APP]),
            'r_beh': mean([flip(a, b) for a, b in BEH]),
            'dlogp_app': mean([dlp(a, b) for a, b in APP]),
            'dlogp_0': dlp('neutral', 'neutral_paraphrase'),
            'acc_low': mean([float(d[a]['correct']) for a, _ in APP]) if all(a in d for a, _ in APP) else None,
            'acc_high': mean([float(d[b]['correct']) for _, b in APP]) if all(b in d for _, b in APP) else None}


def boot(values_by_video, n=5000, seed=0):
    """values_by_video: {video: [values]} -> (mean, lo, hi) with video-clustered bootstrap."""
    vids = list(values_by_video)
    sums = np.array([sum(values_by_video[v]) for v in vids]); cnts = np.array([len(values_by_video[v]) for v in vids])
    est = sums.sum() / cnts.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), (n, len(vids)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    return [round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def summarize(items, keys=('acc', 's0', 's_app', 'r_beh', 'dlogp_app', 'dlogp_0', 'acc_low', 'acc_high'),
              diffs=(('s_app', 's0'), ('dlogp_app', 'dlogp_0'), ('r_beh', 's0'), ('acc_low', 'acc_high'))):
    st = {k: item_stats(d) for k, d in items.items()}
    out = {}
    for key in keys:
        byv = defaultdict(list)
        for k, s in st.items():
            if s[key] is not None:
                byv[k.split('_')[0]].append(s[key])
        out[key] = boot(byv) if byv else None
        out[f'n_{key}'] = sum(len(v) for v in byv.values())
    for a, b in diffs:  # paired differences on items that have both
        byv = defaultdict(list)
        for k, s in st.items():
            if s[a] is not None and s[b] is not None:
                byv[k.split('_')[0]].append(s[a] - s[b])
        out[f'{a}_minus_{b}'] = boot(byv) if byv else None
    return out


def gate():
    rows = load_rows(HERE / 'results/gate/gate_rows.jsonl')
    s = summarize(per_item(rows)['base'])
    passes_shift = (s['s_app_minus_s0'][1] > 0 and s['s_app_minus_s0'][0] >= 0.05) or s['dlogp_app_minus_dlogp_0'][1] > 0
    beh_ok = s['r_beh_minus_s0'][1] > 0
    s.update(gate_text_effect=bool(passes_shift), gate_behavior_control=bool(beh_ok),
             gate_passed=bool(passes_shift and beh_ok),
             mean_letter_mass=round(float(np.mean([r['letter_mass'] for r in rows])), 4),
             all_finite=all(r['finite'] for r in rows), n_rows=len(rows))
    (HERE / 'results/gate/gate_summary.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def sweep():
    rows = load_rows(HERE / 'results/sweep/rows_v2.jsonl.gz')
    pm = per_item(rows)
    table = {m: summarize(items) for m, items in pm.items()}
    (HERE / 'results/sweep/sweep_table.json').write_text(json.dumps(table, indent=1))
    base = table['base']
    acc_min, rbeh_min = base['acc'][0] - 0.02, 0.8 * base['r_beh'][0]
    proposal = {'rule': 'lowest S_app s.t. Acc_clean >= base-0.02 and R_beh >= 0.8*base; ties: smaller alpha, fewer layers',
                'base': {k: base[k] for k in ('acc', 's_app', 'r_beh', 's0')}, 'acc_min': acc_min, 'r_beh_min': rbeh_min}
    for kind, key in (('naive', 'u_app'), ('refined', 'u_perp')):
        cands = []
        for m, t in table.items():
            if not m.startswith(kind + '_'):
                continue
            ls, a = m[len(kind) + 1:].rsplit('_a', 1)
            from evaluate import LAYER_SETS
            cands.append({'name': m, 'dir': key, 'layers': LAYER_SETS[ls], 'alpha': float(a), 's_app': t['s_app'][0],
                          'acc': t['acc'][0], 'r_beh': t['r_beh'][0], 's0': t['s0'][0]})
        order = lambda c: (c['s_app'], c['alpha'], len(c['layers']))
        ok = sorted([c for c in cands if c['acc'] >= acc_min and c['r_beh'] >= rbeh_min], key=order)
        if ok:
            proposal[kind] = dict(ok[0], flagged=False)
        else:
            fb = sorted([c for c in cands if c['acc'] >= acc_min], key=order) or sorted(cands, key=order)
            assert fb, f'no {kind} rows in the sweep'
            proposal[kind] = dict(fb[0], flagged=True, flag='no config met both constraints; chosen by S_app with accuracy constraint only')
        proposal[f'{kind}_candidates'] = sorted(cands, key=order)
    (HERE / 'results/FREEZE_proposal.json').write_text(json.dumps(proposal, indent=1))
    for kind in ('naive', 'refined'):
        print(kind, json.dumps({k: proposal[kind][k] for k in ('name', 's_app', 'acc', 'r_beh', 'flagged')}))


if __name__ == '__main__':
    {'gate': gate, 'sweep': sweep}[sys.argv[1]]()
