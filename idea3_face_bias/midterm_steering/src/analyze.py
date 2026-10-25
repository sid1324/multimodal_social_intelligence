"""Step 5: test-split metrics, paired comparisons, face-text alignment, per-norm results and the recheck.

All flip rates compare the argmax option between the two conditions of a matched pair. Intervals are 95%
percentile bootstraps (5,000 replicates) that resample source videos. Paired differences (method minus base)
are computed per item first, then bootstrapped.

Metrics
  acc        clean accuracy ('none' condition; all test items with a non-empty gold)
  s0         neutral vs neutral_paraphrase
  s_app_tr   appearance flips, training wordings (mean over 3 pairs)
  s_app_ho   appearance flips, held-out wordings (mean over 2 pairs)
  r_beh      behavior flips, held-out wordings (2 pairs)
  r_state    expressed-state flips (2 pairs)
  face_trust, face_null   flips between OMI trust-pair faces and between equal-trust faces
  s_face     face_trust - face_null
  dlogp_app  mean |log p(gold | low) - log p(gold | high)|, training wordings
  align      cosine between f(x) = logp(.|low-trust face) - logp(.|high-trust face) and
             a(x) = mean over training wordings of logp(.|app_low) - logp(.|app_high)  (same method)

Usage: python src/analyze.py   -> results/test_summary.json, results/per_norm.json, results/recheck_summary.json
"""
import gzip, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, PRIMARY
from select_freeze import load_rows, boot

APP_TR = [(f'app_low_tr{i}', f'app_high_tr{i}') for i in range(3)]
APP_HO = [(f'app_low_ho{i}', f'app_high_ho{i}') for i in range(2)]
BEH_HO = [(f'beh_neg_ho{i}', f'beh_pos_ho{i}') for i in range(2)]
STATE = [(f'state_neg{i}', f'state_pos{i}') for i in range(2)]
METHODS = ['base', 'constitution', 'naive', 'refined']


def vec(r, order):
    return np.array([r['logp'][str(i)] for i in order])


def item_metrics(d):
    """d: {condition: row} for one item and one method."""
    def flips(pairs):
        if not all(a in d and b in d for a, b in pairs):
            return None
        return float(np.mean([d[a]['pred'] != d[b]['pred'] for a, b in pairs]))
    m = {'acc': float(d['none']['correct']) if 'none' in d else None,
         's0': flips([('neutral', 'neutral_paraphrase')]),
         's_app_tr': flips(APP_TR), 's_app_ho': flips(APP_HO), 'r_beh': flips(BEH_HO), 'r_state': flips(STATE),
         'face_trust': flips([('face_trust_high', 'face_trust_low')]),
         'face_null': flips([('face_null_a', 'face_null_b')])}
    m['s_face'] = m['face_trust'] - m['face_null'] if m['face_trust'] is not None and m['face_null'] is not None else None
    if all(a in d and b in d for a, b in APP_TR):
        g = str(d[APP_TR[0][0]]['gold'])
        m['dlogp_app'] = float(np.mean([abs(d[a]['logp'][g] - d[b]['logp'][g]) for a, b in APP_TR]))
        order = sorted(d[APP_TR[0][0]]['logp'], key=int)
        a_vec = np.mean([vec(d[a], order) - vec(d[b], order) for a, b in APP_TR], 0)
        if 'face_trust_low' in d and 'face_trust_high' in d:
            f_vec = vec(d['face_trust_low'], order) - vec(d['face_trust_high'], order)
            den = np.linalg.norm(a_vec) * np.linalg.norm(f_vec)
            m['align'] = float(a_vec @ f_vec / den) if den > 0 else None
            m['face_shift'] = float(np.linalg.norm(f_vec))
    return m


def grouped(rows):
    g = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        g[r['method']][r['id']][r['condition']] = r
    return g


def summarize(per_item, keys):
    out = {}
    for k in keys:
        byv = defaultdict(list)
        for cid, m in per_item.items():
            if m.get(k) is not None:
                byv[cid.split('_')[0]].append(m[k])
        out[k] = boot(byv) if byv else None
        out[f'n_{k}'] = sum(len(v) for v in byv.values())
    return out


def paired(per_a, per_b, keys):
    out = {}
    for k in keys:
        byv = defaultdict(list)
        for cid in per_a:
            if cid in per_b and per_a[cid].get(k) is not None and per_b[cid].get(k) is not None:
                byv[cid.split('_')[0]].append(per_a[cid][k] - per_b[cid][k])
        out[k] = boot(byv) if byv else None
    return out


KEYS = ['acc', 's0', 's_app_tr', 's_app_ho', 'r_beh', 'r_state', 'face_trust', 'face_null', 's_face', 'dlogp_app', 'align', 'face_shift']


def main():
    rows = load_rows(os.environ.get('TEST_ROWS', HERE / 'results/test/rows_v2.jsonl.gz'))  # TEST_ROWS: smoke tests only
    g = grouped(rows)
    per = {m: {cid: item_metrics(d) for cid, d in g[m].items()} for m in METHODS if m in g}
    summary = {'n_rows': len(rows), 'methods': {m: summarize(per[m], KEYS) for m in per},
               'vs_base': {m: paired(per[m], per['base'], KEYS) for m in per if m != 'base'},
               'refined_vs_naive': paired(per['refined'], per['naive'], KEYS) if 'refined' in per and 'naive' in per else None,
               'frozen': json.loads((HERE / 'results/FREEZE.json').read_text())}
    # cross-modal sign test: is the face effect aligned with the text trust axis? (base model)
    al = [m['align'] for m in per['base'].values() if m.get('align') is not None]
    summary['base_align_share_positive'] = round(float(np.mean(np.array(al) > 0)), 4) if al else None
    (HERE / 'results/test_summary.json').write_text(json.dumps(summary, indent=1))

    # per primary vs secondary norm
    norms = {r['id']: r['norms'] for r in rows}
    pn = {}
    for grp in ('primary', 'secondary_only'):
        ids = {cid for cid, ns in norms.items() if (bool(set(ns) & PRIMARY) == (grp == 'primary'))}
        pn[grp] = {m: summarize({c: v for c, v in per[m].items() if c in ids}, ['acc', 's_app_tr', 's_app_ho', 'r_beh']) for m in per}
    (HERE / 'results/per_norm.json').write_text(json.dumps(pn, indent=1))

    # recheck: one-at-a-time rows vs batched test rows on the same items, conditions and methods
    rp = HERE / 'results/recheck/rows_v2.jsonl.gz'
    if rp.exists():
        rec = load_rows(rp)
        test_idx = {(r['id'], r['method'], r['condition']): r for r in rows}
        pairs = [(r, test_idx[(r['id'], r['method'], r['condition'])]) for r in rec if (r['id'], r['method'], r['condition']) in test_idx]
        dp = [max(abs(np.exp(a['logp'][k]) - np.exp(b['logp'][k])) for k in a['logp']) for a, b in pairs]
        rg = grouped(rec)
        ids = set(rg['base'])
        rs = {'n_compared': len(pairs), 'argmax_agree': round(float(np.mean([a['pred'] == b['pred'] for a, b in pairs])), 4),
              'max_abs_dp': round(float(np.max(dp)), 4), 'mean_abs_dp': round(float(np.mean(dp)), 5)}
        for m in ('base', 'refined'):
            one = summarize({c: item_metrics(d) for c, d in rg[m].items()}, ['acc', 's0', 's_app_tr', 's_app_ho'])
            bat = summarize({c: v for c, v in per[m].items() if c in ids}, ['acc', 's0', 's_app_tr', 's_app_ho'])
            rs[m] = {k: {'one_at_a_time': one[k], 'batched': bat[k]} for k in ('acc', 's0', 's_app_tr', 's_app_ho')}
        (HERE / 'results/recheck_summary.json').write_text(json.dumps(rs, indent=1))
        print('recheck:', json.dumps({k: rs[k] for k in ('n_compared', 'argmax_agree', 'max_abs_dp')}))

    pct = lambda v, s=100: 'NA' if v is None else f'{s * v[0]:.1f} [{s * v[1]:.1f}, {s * v[2]:.1f}]'
    for m in per:
        s = summary['methods'][m]
        print(f"{m:13s} acc {pct(s['acc'])}  S0 {pct(s['s0'])}  S_app tr {pct(s['s_app_tr'])} ho {pct(s['s_app_ho'])}  "
              f"R_beh {pct(s['r_beh'])}  R_state {pct(s['r_state'])}  S_face {pct(s['s_face'])}  align {pct(s['align'], 1)}")
    print('base: share of items with face effect aligned to text trust axis:', summary['base_align_share_positive'])


if __name__ == '__main__':
    main()
