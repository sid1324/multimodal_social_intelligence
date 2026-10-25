"""Failure-analysis data for results/failure_cases.md (test split, frozen methods).

Writes results/failure_data.json with:
  direction      base model: how appearance cues move choices (accuracy and person-directed choice, low vs high)
  still_flip     items whose answer still flips under training-wording appearance cues after steering, by norm group
  beh_lost       items whose behavior response (held-out wordings) disappears under steering
  heldout        items fixed on training wordings that still flip on held-out wordings
  acc_damage     clean items that steering turns from correct to wrong (and the reverse)
  face           OMI arm: flip rates per trust pair vs the pair's attractiveness and dominance gaps; alignment
  examples       candidate qualitative examples with option texts and probabilities
"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, PRIMARY, load_annotations
from select_freeze import load_rows
from analyze import grouped, item_metrics, APP_TR, APP_HO, BEH_HO, METHODS
sys.path.insert(0, str(HERE.parent / 'code'))
from text_arm import person_directed  # A2 lexicon (identical copy of reference/premise_analysis/text_arm.py)


def flips(d, pairs):
    return [d[a]['pred'] != d[b]['pred'] for a, b in pairs if a in d and b in d]


def main():
    ann = load_annotations()
    rows = load_rows(os.environ.get('TEST_ROWS', HERE / 'results/test/rows_v2.jsonl.gz'))  # TEST_ROWS: smoke tests only
    g = grouped(rows)
    pool = [c for c in g['base'] if 'app_low_tr0' in g['base'][c]]
    primary = {c: bool(set(g['base'][c]['none']['norms']) & PRIMARY) for c in g['base'] if 'none' in g['base'][c]}
    pd = lambda c, opt: person_directed(ann[c]['behaviors'][opt])
    out = {}

    # 1. direction of the base effect
    acc_low = np.mean([g['base'][c][a]['correct'] for c in pool for a, _ in APP_TR])
    acc_high = np.mean([g['base'][c][b]['correct'] for c in pool for _, b in APP_TR])
    pd_low = np.mean([pd(c, g['base'][c][a]['pred']) for c in pool for a, _ in APP_TR])
    pd_high = np.mean([pd(c, g['base'][c][b]['pred']) for c in pool for _, b in APP_TR])
    out['direction'] = {'acc_low': round(acc_low, 4), 'acc_high': round(acc_high, 4),
                        'person_directed_low': round(pd_low, 4), 'person_directed_high': round(pd_high, 4)}
    for m in ('base', 'constitution', 'naive', 'refined'):
        out['direction'][f'{m}_acc_low_minus_high'] = round(
            np.mean([g[m][c][a]['correct'] for c in pool for a, _ in APP_TR]) - np.mean([g[m][c][b]['correct'] for c in pool for _, b in APP_TR]), 4)

    # 2. still flipping after steering, by norm group and by person-directed switch
    out['still_flip'] = {}
    for m in ('base', 'constitution', 'naive', 'refined'):
        f = {c: np.mean(flips(g[m][c], APP_TR)) for c in pool}
        by = defaultdict(list)
        for c, v in f.items():
            by['primary' if primary[c] else 'secondary_only'].append(v)
        sw = [pd(c, g[m][c][a]['pred']) != pd(c, g[m][c][b]['pred']) for c in pool for a, b in APP_TR if g[m][c][a]['pred'] != g[m][c][b]['pred']]
        out['still_flip'][m] = {'items_any_flip': int(sum(v > 0 for v in f.values())), 'n': len(f),
                                'rate_primary': round(float(np.mean(by['primary'])), 4), 'n_primary': len(by['primary']),
                                'rate_secondary_only': round(float(np.mean(by['secondary_only'])), 4), 'n_secondary_only': len(by['secondary_only']),
                                'share_of_flips_switching_person_directedness': round(float(np.mean(sw)), 4) if sw else None,
                                'items_all3_flip': sorted(c for c, v in f.items() if v == 1.0)}
    base_f = {c: np.mean(flips(g['base'][c], APP_TR)) for c in pool}
    for m in ('naive', 'refined'):
        f = {c: np.mean(flips(g[m][c], APP_TR)) for c in pool}
        out['still_flip'][m]['fixed'] = sum(1 for c in pool if base_f[c] > 0 and f[c] == 0)
        out['still_flip'][m]['new_flips'] = sum(1 for c in pool if base_f[c] == 0 and f[c] > 0)
        out['still_flip'][m]['unchanged_flipping'] = sum(1 for c in pool if base_f[c] > 0 and f[c] > 0)

    # 3. behavior response lost, 4. held-out wording failures, 5. clean accuracy damage
    out['beh_lost'], out['heldout'], out['acc_damage'] = {}, {}, {}
    clean = [c for c in g['base'] if 'none' in g['base'][c]]
    for m in ('constitution', 'naive', 'refined'):
        bb = {c: np.mean(flips(g['base'][c], BEH_HO)) for c in pool}
        bm = {c: np.mean(flips(g[m][c], BEH_HO)) for c in pool}
        lost = sorted(c for c in pool if bb[c] > 0 and bm[c] == 0)
        out['beh_lost'][m] = {'n_lost': len(lost), 'n_gained': sum(1 for c in pool if bb[c] == 0 and bm[c] > 0), 'items': lost[:15]}
        tr = {c: np.mean(flips(g[m][c], APP_TR)) for c in pool}
        ho = {c: np.mean(flips(g[m][c], APP_HO)) for c in pool}
        out['heldout'][m] = {'fixed_tr_still_flip_ho': sum(1 for c in pool if base_f[c] > 0 and tr[c] == 0 and ho[c] > 0),
                             'mean_tr': round(float(np.mean(list(tr.values()))), 4), 'mean_ho': round(float(np.mean(list(ho.values()))), 4)}
        broke = sorted(c for c in clean if g['base'][c]['none']['correct'] and not g[m][c]['none']['correct'])
        fixed = sorted(c for c in clean if not g['base'][c]['none']['correct'] and g[m][c]['none']['correct'])
        out['acc_damage'][m] = {'correct_to_wrong': len(broke), 'wrong_to_correct': len(fixed), 'items_broken': broke[:15]}

    # 6. face arm: per trust pair, flip rate (base) vs the pair's attribute gaps
    pairs = {(p['high'] + '|' + p['low']): p for p in json.loads((HERE / 'results/omi_pairs.json').read_text())['trust_pairs']}
    out['face'] = {}
    for m in METHODS:
        per_pair = defaultdict(list)
        for c in pool:
            d = g[m][c]
            if 'face_trust_high' in d and 'face_trust_low' in d:
                per_pair[d['face_trust_high']['trust_pair']].append(d['face_trust_high']['pred'] != d['face_trust_low']['pred'])
        rates = {k: float(np.mean(v)) for k, v in per_pair.items()}
        gaps = {att: [pairs[k]['diff'][att] for k in rates] for att in ('trustworthy', 'attractive', 'dominant', 'happy')}
        r = list(rates.values())
        out['face'][m] = {'n_pairs': len(rates), 'mean_flip': round(float(np.mean(r)), 4),
                          'corr_flip_with_gap': {att: round(float(np.corrcoef(r, v)[0, 1]), 3) for att, v in gaps.items() if np.std(v) > 0}}
        # does the low-trust face push the model toward the low-trust text choice? (alignment of argmax shifts)
        same = []
        for c in pool:
            d = g[m][c]
            if d['face_trust_high']['pred'] != d['face_trust_low']['pred'] and d['app_low_tr0']['pred'] != d['app_high_tr0']['pred']:
                same.append(d['face_trust_low']['pred'] == d['app_low_tr0']['pred'])
        out['face'][m]['both_flip_items'] = len(same)
        out['face'][m]['face_low_matches_text_low'] = round(float(np.mean(same)), 4) if same else None

    # 7. qualitative candidates: base flips on all three training wordings; show what each method does
    ex = []
    for c in out['still_flip']['base']['items_all3_flip'][:40]:
        v = ann[c]
        opts = {i: b for i, b in enumerate(v['behaviors']) if b.strip()}
        e = {'id': c, 'norms': g['base'][c]['none']['norms'], 'gold': v['correct'], 'options': opts}
        for m in METHODS:
            d = g[m][c]
            e[m] = {k: {'pred': d[k]['pred'], 'p': {i: round(float(np.exp(x)), 3) for i, x in d[k]['logp'].items()}}
                    for k in ('none', 'app_low_tr0', 'app_high_tr0', 'beh_neg_ho0', 'beh_pos_ho0', 'face_trust_low', 'face_trust_high') if k in d}
        ex.append(e)
    out['examples'] = ex
    (HERE / 'results/failure_data.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
