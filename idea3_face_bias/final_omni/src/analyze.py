"""Gate, sweep and DPO-validation summaries. Definitions and bootstrap (5,000 replicates over source videos) as at midterm.
Usage: python src/analyze.py g2 --model nemotron | sweep | dpo"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE
from evaluate import read_rows, layer_sets


def boot(byv, n=5000, seed=0):
    vids = list(byv)
    s = np.array([sum(byv[v]) for v in vids]); c = np.array([len(byv[v]) for v in vids])
    idx = np.random.default_rng(seed).integers(0, len(vids), (n, len(vids)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return [round(float(s.sum() / c.sum()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def index(rows):
    out = defaultdict(lambda: defaultdict(dict))   # method -> item -> (cond, channel) -> row
    for r in rows:
        out[r['method']][r['id']][(r['condition'], r['channel'])] = r
    return out


def item_stats(d, lv, ch):
    g = lambda c: d.get((c, ch))
    def flip(ps):
        v = [(g(a), g(b)) for a, b in ps]
        return None if any(x is None or y is None for x, y in v) else float(np.mean([x['pred'] != y['pred'] for x, y in v]))
    def dlp(ps):
        v = [(g(a), g(b)) for a, b in ps]
        return None if any(x is None or y is None for x, y in v) else float(np.mean([abs(x['logp'][str(x['gold'])] - y['logp'][str(y['gold'])]) for x, y in v]))
    app = [(f'{lv}_app_low_tr{i}', f'{lv}_app_high_tr{i}') for i in range(3)]
    beh = [(f'{lv}_beh_neg_tr{i}', f'{lv}_beh_pos_tr{i}') for i in range(2)]
    neu = [(f'{lv}_neutral', f'{lv}_neutral_paraphrase')]
    acc = lambda cs: None if any(g(c) is None for c in cs) else float(np.mean([g(c)['correct'] for c in cs]))
    s = {'acc': float(d[('none', 'text')]['correct']) if ('none', 'text') in d else None, 's0': flip(neu), 's_app': flip(app), 'r_beh': flip(beh),
         'dlp_app': dlp(app), 'dlp_0': dlp(neu), 'acc_low': acc([a for a, _ in app]), 'acc_high': acc([b for _, b in app]), 'acc_neutral': acc([neu[0][0]])}
    diff = lambda a, b: None if s[a] is None or s[b] is None else s[a] - s[b]
    s['app_minus_floor'] = diff('s_app', 's0'); s['beh_minus_floor'] = diff('r_beh', 's0'); s['low_minus_high'] = diff('acc_low', 'acc_high')
    s['dlp_minus_floor'] = diff('dlp_app', 'dlp_0')
    return s


def summarize(per_item):
    out = {}
    for k in next(iter(per_item.values())):
        byv = defaultdict(list)
        for cid, s in per_item.items():
            if s[k] is not None:
                byv[cid.split('_')[0]].append(s[k])
        out[k] = boot(byv) if byv else None
        out['n_' + k] = sum(map(len, byv.values()))
    return out


pct = lambda v: 'NA' if v is None else f'{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]'


def g2(a):
    rows = [r for p in sorted((HERE / 'results/gate').glob(f'g2_{a.model}_*_s?.jsonl')) for r in read_rows(p)]
    ix = index(rows)['base']; S = {}
    for ch, lv in [('text', 'L1'), ('text', 'L2'), ('text', 'L3'), ('speech', a.level), ('image', a.level)]:
        s = summarize({c: item_stats(d, lv, ch) for c, d in ix.items()})
        if not s['s_app']:
            continue
        s['pass'] = bool(s['app_minus_floor'][1] > 0)
        S[f'{ch} {lv}'] = s
        print(f"{ch:6s} {lv} n={s['n_s_app']:3d} app {pct(s['s_app'])} floor {pct(s['s0'])} app-floor {pct(s['app_minus_floor'])} beh {pct(s['r_beh'])} "
              f"acc low/high {pct(s['acc_low'])} / {pct(s['acc_high'])} diff {pct(s['low_minus_high'])} clean {pct(s['acc'])} -> {'PASS' if s['pass'] else 'FAIL'}")
    for ch in ('speech', 'image'):       # exploratory, Nov 24: L1 speech and image (only if scored)
        if any(k == ('L1_neutral', ch) for d in ix.values() for k in d):
            s = summarize({c: item_stats(d, 'L1', ch) for c, d in ix.items()})
            if s['s_app']:
                s['pass'] = bool(s['app_minus_floor'][1] > 0); s['exploratory'] = True; S[f'{ch} L1 (exploratory)'] = s
                print(f"{ch:6s} L1 exploratory n={s['n_s_app']} app {pct(s['s_app'])} floor {pct(s['s0'])} app-floor {pct(s['app_minus_floor'])} beh {pct(s['r_beh'])} acc low/high {pct(s['acc_low'])} / {pct(s['acc_high'])}")
    S['letter_mass'] = float(np.mean([r['letter_mass'] for r in rows])); S['finite'] = all(r['finite'] for r in rows); S['n_rows'] = len(rows)
    print('letter mass', S['letter_mass'], 'finite', S['finite'], 'rows', len(rows))
    (HERE / f'results/gate/g2_{a.model}_summary.json').write_text(json.dumps(S, indent=1))


def choose(table, base, cands, order):
    """Section 6 rule."""
    amin, rmin = base['acc'][0] - 0.02, 0.8 * base['r_beh'][0]
    ok = sorted([c for c in cands if table[c]['acc'][0] >= amin and table[c]['r_beh'][0] >= rmin], key=lambda c: (table[c]['s_app'][0],) + order(c))
    if ok:
        return ok[0], False
    fb = sorted([c for c in cands if table[c]['acc'][0] >= amin], key=lambda c: (table[c]['s_app'][0],) + order(c))
    return (fb or sorted(cands, key=lambda c: table[c]['s_app'][0]))[0], True


def sweep(a):
    ix = index([r for p in sorted((HERE / 'results/sweep').glob(f'{a.model}_*.jsonl')) for r in read_rows(p)])
    T = {m: summarize({c: item_stats(d, a.level, 'text') for c, d in it.items()}) for m, it in ix.items()}
    nl = a.n_layers; ls = layer_sets(nl)
    parse = lambda m: (lambda x: (float(x[1]), len(ls[x[0]])))(m[len('naive_'):].rsplit('_a', 1))
    cands = [m for m in T if m.startswith('naive_')]
    best, flag = choose(T, T['base'], cands, parse)
    for m in ['base'] + cands:
        t = T[m]; print(f"{m:18s} acc {pct(t['acc'])} S_app {pct(t['s_app'])} S0 {pct(t['s0'])} R_beh {pct(t['r_beh'])}" + ('  <- chosen' if m == best else ''))
    nm, al = best[len('naive_'):].rsplit('_a', 1)
    P = {'rule': 'Section 6', 'chosen': best, 'layers': ls[nm], 'alpha': float(al), 'direction': f'app_text_{a.level}', 'fallback_used': flag,
         'acc_min': T['base']['acc'][0] - 0.02, 'r_beh_min': 0.8 * T['base']['r_beh'][0]}
    (HERE / f'results/sweep/{a.model}_summary.json').write_text(json.dumps({'table': T, 'proposal': P}, indent=1)); print(json.dumps(P))


def dpo(a):
    ix = index(read_rows(HERE / f'results/dpo/{a.model}_val_rows.jsonl'))
    T = {}
    for m, it in ix.items():
        T[m] = {ch: summarize({c: item_stats(d, a.level, ch) for c, d in it.items()}) for ch in ('text', 'speech', 'image')}
        pi = {}
        for c, d in it.items():   # all-channel average per item (DPO-multi selection)
            ss = [item_stats(d, a.level, ch) for ch in ('text', 'speech', 'image')]
            pi[c] = {k: (None if any(s[k] is None for s in ss) else float(np.mean([s[k] for s in ss]))) for k in ss[0]}
        T[m]['mean'] = summarize(pi)
    P = {}
    for v, ch in (('text', 'text'), ('multi', 'mean')):
        cands = [m for m in T if m.startswith(f'dpo_{v}_')]
        if cands:
            best, flag = choose({m: T[m][ch] for m in T}, T['base'][ch], cands, lambda m: (int(m[-1]),))
            P[v] = {'chosen': best, 'selection_channel': ch, 'fallback_used': flag}
    for m in sorted(T):
        for ch in ('text', 'speech', 'image', 'mean'):
            t = T[m][ch]; print(f"{m:14s} {ch:6s} acc {pct(t['acc'])} S_app {pct(t['s_app'])} S0 {pct(t['s0'])} R_beh {pct(t['r_beh'])}")
    (HERE / f'results/dpo/{a.model}_val_summary.json').write_text(json.dumps({'table': T, 'proposal': P}, indent=1)); print(json.dumps(P))


def blind(a):
    """Blank-image and wrong-image check on the 300 gate items (train). Real-scene rows come from G2."""
    ix = {'real': index([r for p in sorted((HERE / 'results/gate').glob(f'g2_{a.model}_text_s?.jsonl')) for r in read_rows(p)])['base']}
    for v in ('blank', 'wrong'):
        rows = read_rows(HERE / f'results/gate/blind_{a.model}_{v}.jsonl')
        if rows:
            ix[v] = index(rows)[f'base_{v}']
    S = {}
    have = set.intersection(*[set(d) for d in ix.values()])
    for v, d in ix.items():
        st = {c: item_stats(d[c], 'L2', 'text') for c in have}
        s = summarize(st); S[v] = {k: s[k] for k in ('acc', 'acc_neutral', 's_app', 's0', 'app_minus_floor', 'acc_low', 'acc_high')}
        print(f"{v:6s} n={len(have)} no-remark acc {pct(s['acc'])} neutral acc {pct(s['acc_neutral'])} app flips {pct(s['s_app'])} floor {pct(s['s0'])} app-floor {pct(s['app_minus_floor'])}")
        if v != 'real':
            byv = defaultdict(list)
            for c in have:
                r, x = d[c][('none', 'text')], ix['real'][c][('none', 'text')]
                byv[c.split('_')[0]].append(float(r['correct']) - float(x['correct']))
            S[v]['acc_minus_real'] = boot(byv); print('       accuracy minus real scene', pct(S[v]['acc_minus_real']))
            S[v]['same_answer_as_real_scene'] = float(np.mean([d[c][('none', 'text')]['pred'] == ix['real'][c][('none', 'text')]['pred'] for c in have]))
            print('       same no-remark answer as with real scene', round(100 * S[v]['same_answer_as_real_scene'], 1))
    (HERE / f'results/gate/blind_{a.model}_summary.json').write_text(json.dumps(S, indent=1))


def dpoval(a):
    """DPO on validation, against base: L2 text (pre-registered measure), L1 text and L2 image (descriptive), paired by item."""
    D = HERE / 'results/dpo'
    rows = [dict(r, method='base') for r in read_rows(HERE / 'results/sweep/nemotron_base.jsonl')]
    for p in sorted(D.glob('nemotron_val_*.jsonl')):
        rows += read_rows(p)
    ix = index(rows); S = {}
    for st, lv, ch in (('L2 text', 'L2', 'text'), ('L1 text', 'L1', 'text'), ('L2 image', 'L2', 'image')):
        for m in sorted(ix):
            st_m = {c: item_stats(d, lv, ch) for c, d in ix[m].items()}
            have = {c for c, s in st_m.items() if s['s_app'] is not None and s['s0'] is not None and s['r_beh'] is not None}
            if not have:
                continue
            base = {c: item_stats(d, lv, ch) for c, d in ix['base'].items() if c in have}
            base = {c: s for c, s in base.items() if s['s_app'] is not None}
            common = have & set(base)
            if len(common) < 20:
                continue
            summ = summarize({c: st_m[c] for c in common})
            out = {'n': len(common), **{k: summ[k] for k in ('s_app', 's0', 'r_beh', 'app_minus_floor')}}
            if m != 'base':
                for k in ('s_app', 's0', 'r_beh', 'app_minus_floor'):
                    byv = defaultdict(list)
                    for c in common:
                        byv[c.split('_')[0]].append(st_m[c][k] - base[c][k])
                    out['d_' + k] = boot(byv)
            S[f'{st} {m}'] = out
            print(f"{st:9s} {m:14s} n={out['n']:3d} S_app {pct(out['s_app'])} floor {pct(out['s0'])} app-floor {pct(out['app_minus_floor'])} R_beh {pct(out['r_beh'])}"
                  + (f" | change vs base: S_app {pct(out['d_s_app'])} R_beh {pct(out['d_r_beh'])}" if m != 'base' else ''))
        # accuracy on clean items (no remark) for L2 text files
    acc = {m: summarize({c: {'acc': float(d[('none', 'text')]['correct'])} for c, d in it.items() if ('none', 'text') in d}) for m, it in ix.items()}
    for m, s in acc.items():
        if 'acc' in s and s['acc']:
            print(f"accuracy (no remark) {m:14s} {pct(s['acc'])} n={s['n_acc']}")
    (HERE / 'results/dpo/nemotron_val_prelim.json').write_text(json.dumps(S, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['g2', 'sweep', 'dpo', 'blind', 'dpoval']); ap.add_argument('--model', default='nemotron')
    ap.add_argument('--level', default='L2'); ap.add_argument('--n_layers', type=int, default=52)
    a = ap.parse_args(); {'g2': g2, 'sweep': sweep, 'dpo': dpo, 'blind': blind, 'dpoval': dpoval}[a.phase](a)
