"""Unified scoring for the clean-60 pilot: every model, one isolated call per (item, condition).

Usage: clean60_score_all.py [answer_key.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-clean60'
CONDS = ['text', 'textforced', 'rgb', 'geo']
MODELS = ['Qwen3-VL-32B', 'Sonnet 5.5', 'Opus 5.5', 'GPT-5.5', 'GPT-5.6', 'GPT-6']


def load_answers():
    answers = {}  # (model, cond, id) -> (action_position, justification_position) or None if unparseable
    for path in (OUT / 'results/claude_single').glob('*.json'):
        name, cond, _ = path.name.split('__')
        r = json.loads(path.read_text())
        answers[({'opus': 'Opus 5.5', 'sonnet': 'Sonnet 5.5'}[name], cond, r['id'])] = (r['action_position'], r['justification_position'])
    for folder, model in [('gpt', 'GPT-5.6'), ('gpt-5.5', 'GPT-5.5'), ('gpt-6-sol', 'GPT-6')]:
        for path in (OUT / 'results' / folder).glob('*.json'):
            r = json.loads(path.read_text())
            answers[(model, r['condition'], r['id'])] = (r['action_position'], r['justification_position'])
    qwen = OUT / 'results/qwen32b.json'
    if qwen.exists():
        for r in json.loads(qwen.read_text()):
            answers[('Qwen3-VL-32B', r['condition'], r['id'])] = (r['action_position'], r['justification_position']) if r['action_position'] else None
    return answers


def mcnemar_exact(fixed, broken):
    """Two-sided exact McNemar p-value from the discordant counts."""
    from math import comb
    n, k = fixed + broken, min(fixed, broken)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    key = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else OUT / 'answer_key.json').read_text())
    tasks = {t['id']: t for t in json.loads((OUT / 'tasks.json').read_text())}
    answers = load_answers()
    ok, none = {}, {}
    for (model, cond, item), pos in answers.items():
        k = key[item]
        if pos is None:
            ok[(model, cond, item)] = (False, False, None)
            none[(model, cond, item)] = False
            continue
        a0, j0 = k['action_order'][pos[0] - 1], k['justification_order'][pos[1] - 1]
        ok[(model, cond, item)] = (a0 == k['correct'], a0 == k['correct'] and j0 == k['correct'], (a0, j0))
        none[(model, cond, item)] = 'None of these' in tasks[item]['action_options'][pos[0] - 1]
    rng = np.random.default_rng(42)
    summary = {}
    for model in MODELS:
        s = {}
        for c in CONDS:
            have = [i for i in key if (model, c, i) in ok]
            if not have:
                continue
            both = np.array([ok[(model, c, i)][1] for i in have], float)
            boots = [both[rng.integers(0, len(both), len(both))].mean() for _ in range(2000)]
            s[c] = {'n': len(have), 'action': int(sum(ok[(model, c, i)][0] for i in have)), 'both': int(both.sum()),
                    'both_ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    'chose_none': int(sum(none[(model, c, i)] for i in have)), 'unparseable': int(sum(answers[(model, c, i)] is None for i in have))}
        for a, b in [('textforced', 'rgb'), ('rgb', 'geo')]:
            ids = [i for i in key if (model, a, i) in ok and (model, b, i) in ok]
            if len(ids) < len(key):
                continue
            fixed = sum(not ok[(model, a, i)][1] and ok[(model, b, i)][1] for i in ids)
            broken = sum(ok[(model, a, i)][1] and not ok[(model, b, i)][1] for i in ids)
            s[f'{a}->{b}'] = {'fixed': fixed, 'broken': broken, 'answer_changed': sum(ok[(model, a, i)][2] != ok[(model, b, i)][2] for i in ids), 'mcnemar_p': mcnemar_exact(fixed, broken),
                              'fixed_ids': [i for i in ids if not ok[(model, a, i)][1] and ok[(model, b, i)][1]]}
        prox = [i for i in key if 'Proxemics' in key[i]['labels']]
        s['proxemics'] = {'n': len(prox), **{c: int(sum(ok[(model, c, i)][1] for i in prox if (model, c, i) in ok)) for c in CONDS if c in s}}
        summary[model] = s
    complete = [m for m in MODELS if all(summary[m].get(c, {}).get('n') == len(key) for c in ['rgb', 'geo'])]
    cross = {'models_complete': complete,
             'items_wrong_on_rgb_for_all': sum(not any(ok[(m, 'rgb', i)][1] for m in complete) for i in key),
             'of_those_right_with_geo_for_any': sum(not any(ok[(m, 'rgb', i)][1] for m in complete) and any(ok[(m, 'geo', i)][1] for m in complete) for i in key),
             'items_fixed_by_geo_in_n_models': {str(n): sum(sum(i in summary[m]['rgb->geo']['fixed_ids'] for m in complete) == n for i in key) for n in range(len(complete) + 1)} if complete else {}}
    pooled = {'fixed': sum(summary[m]['rgb->geo']['fixed'] for m in complete), 'broken': sum(summary[m]['rgb->geo']['broken'] for m in complete)}
    cross['pooled_rgb->geo'] = {**pooled, 'pairs': len(complete) * len(key)}
    # Model-averaged paired differences with a bootstrap over items (items are shared by all models, so they are the resampling unit).
    ids = list(key)
    for name, a, b in [('geo_minus_rgb_pp', 'rgb', 'geo'), ('rgb_minus_textforced_pp', 'textforced', 'rgb')]:
        diff = np.array([[ok[(m, b, i)][1] - ok[(m, a, i)][1] for i in ids] for m in complete], float)  # models x items
        boots = [diff[:, rng.integers(0, len(ids), len(ids))].mean() for _ in range(5000)]
        cross[name] = {'mean': float(100 * diff.mean()), 'ci95': [float(100 * np.percentile(boots, 2.5)), float(100 * np.percentile(boots, 97.5))],
                       'per_model': {m: float(100 * diff[k].mean()) for k, m in enumerate(complete)}}
    prox = [i for i in ids if 'Proxemics' in key[i]['labels']]
    cross['proxemics_pooled'] = {'n_items': len(prox), 'rgb': int(sum(ok[(m, 'rgb', i)][1] for m in complete for i in prox)), 'geo': int(sum(ok[(m, 'geo', i)][1] for m in complete for i in prox)), 'pairs': len(prox) * len(complete)}
    cross['pooled_rgb->geo']['mcnemar_p_descriptive'] = mcnemar_exact(pooled['fixed'], pooled['broken'])
    (OUT / 'scored_all.json').write_text(json.dumps({'summary': summary, 'cross_model': cross, 'rows': [{'model': m, 'condition': c, 'id': i, 'action_correct': v[0], 'both_correct': v[1], 'chose_none': none[(m, c, i)]} for (m, c, i), v in ok.items()]}, indent=1))
    print(f"{'model':14s} " + ' '.join(f'{c:>16s}' for c in CONDS) + '   rgb->geo fixed/broken (p)   none chosen t/tf/rgb/geo')
    for model in MODELS:
        s = summary[model]
        cells = ' '.join((f"{s[c]['both']:2d}/{s[c]['n']} ({s[c]['both'] / s[c]['n']:.0%})".rjust(16) if c in s else ' ' * 16) for c in CONDS)
        t = s.get('rgb->geo')
        print(f"{model:14s} {cells}   " + (f"{t['fixed']}/{t['broken']} (p={t['mcnemar_p']:.2f})".ljust(26) if t else ' ' * 26) + ' ' + '/'.join(str(s[c]['chose_none']) if c in s else '-' for c in CONDS) + (f"  unparseable {sum(s[c]['unparseable'] for c in CONDS if c in s)}" if any(s[c]['unparseable'] for c in CONDS if c in s) else ''))
    print('proxemics (n=%d):' % len([i for i in key if 'Proxemics' in key[i]['labels']]), {m: {c: summary[m]['proxemics'].get(c) for c in ['rgb', 'geo']} for m in MODELS if 'rgb' in summary[m]})
    print('cross-model:', json.dumps(cross))


if __name__ == '__main__':
    main()
