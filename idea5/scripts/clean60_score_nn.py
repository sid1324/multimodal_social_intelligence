"""Score the forced-choice variant of the clean-60 pilot (prompts forbid the "None of these" options).

Reports every statistic twice: on all 60 items, and on the items whose labelled answer is a real option
(the others cannot be answered correctly under a forced choice). Usage: clean60_score_nn.py [answer_key.json]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from clean60_score_all import mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-clean60'
CONDS = ['text_nn', 'rgb_nn', 'geo_nn']
MODELS = ['Qwen3-VL-32B', 'GPT-5.5', 'GPT-5.6', 'GPT-6', 'Sonnet 5.5', 'Opus 5.5']
EXPECTED_EVENTS = {'thread.started', 'turn.started', 'item.completed:agent_message', 'turn.completed'}


def load_answers():
    answers, gpt_events = {}, Counter()
    for path in (OUT / 'results/claude_single').glob('*_nn__*.json'):
        name, cond, _ = path.name.split('__')
        r = json.loads(path.read_text())
        assert r['tools_available'] == 0 and r['mcp_servers'] == 0 and r['num_turns'] == 1, path
        answers[({'opus': 'Opus 5.5', 'sonnet': 'Sonnet 5.5'}[name], cond, r['id'])] = (r['action_position'], r['justification_position'])
    for folder, model in [('gpt', 'GPT-5.6'), ('gpt-5.5', 'GPT-5.5'), ('gpt-6-sol', 'GPT-6')]:
        for path in (OUT / 'results' / folder).glob('*_nn__*.json'):
            r = json.loads(path.read_text())
            answers[(model, r['condition'], r['id'])] = (r['action_position'], r['justification_position'])
            gpt_events['calls'] += 1
            gpt_events['calls_with_unexpected_events'] += bool(set(r['events']) - EXPECTED_EVENTS)
            gpt_events['calls_with_more_than_one_message'] += r['events'].get('item.completed:agent_message', 0) != 1
    qwen = OUT / 'results/qwen32b_nn.json'
    if qwen.exists():
        for r in json.loads(qwen.read_text()):
            answers[('Qwen3-VL-32B', r['condition'], r['id'])] = (r['action_position'], r['justification_position']) if r['action_position'] else None
    return answers, dict(gpt_events)


def summarise(ids, key, tasks, answers, rng):
    ok, none = {}, {}
    for (model, cond, item), pos in answers.items():
        if item not in ids:
            continue
        k = key[item]
        if pos is None:
            ok[(model, cond, item)], none[(model, cond, item)] = (False, False, None), False
            continue
        a0, j0 = k['action_order'][pos[0] - 1], k['justification_order'][pos[1] - 1]
        ok[(model, cond, item)] = (a0 == k['correct'], a0 == k['correct'] and j0 == k['correct'], (a0, j0))
        none[(model, cond, item)] = 'None of these' in tasks[item]['action_options'][pos[0] - 1] or 'None of these' in tasks[item]['justification_options'][pos[1] - 1]
    summary = {}
    for model in MODELS:
        s = {}
        for c in CONDS:
            have = [i for i in ids if (model, c, i) in ok]
            if not have:
                continue
            both = np.array([ok[(model, c, i)][1] for i in have], float)
            boots = [both[rng.integers(0, len(both), len(both))].mean() for _ in range(2000)]
            s[c] = {'n': len(have), 'action': int(sum(ok[(model, c, i)][0] for i in have)), 'both': int(both.sum()),
                    'both_ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    'chose_none': int(sum(none[(model, c, i)] for i in have)), 'unparseable': int(sum(answers[(model, c, i)] is None for i in have))}
        for a, b in [('text_nn', 'rgb_nn'), ('rgb_nn', 'geo_nn')]:
            if not all((model, x, i) in ok for x in (a, b) for i in ids):
                continue
            fixed = [i for i in ids if not ok[(model, a, i)][1] and ok[(model, b, i)][1]]
            broken = [i for i in ids if ok[(model, a, i)][1] and not ok[(model, b, i)][1]]
            s[f'{a}->{b}'] = {'fixed': len(fixed), 'broken': len(broken), 'answer_changed': sum(ok[(model, a, i)][2] != ok[(model, b, i)][2] for i in ids),
                              'mcnemar_p': mcnemar_exact(len(fixed), len(broken)), 'fixed_ids': fixed}
        summary[model] = s
    complete = [m for m in MODELS if all(summary[m].get(c, {}).get('n') == len(ids) for c in CONDS)]
    cross = {'models_complete': complete, 'n_items': len(ids)}
    if complete:
        order = list(ids)
        cross['items_wrong_with_frames_for_all'] = sum(not any(ok[(m, 'rgb_nn', i)][1] for m in complete) for i in order)
        cross['of_those_right_with_maps_for_any'] = sum(not any(ok[(m, 'rgb_nn', i)][1] for m in complete) and any(ok[(m, 'geo_nn', i)][1] for m in complete) for i in order)
        cross['items_fixed_by_maps_in_n_models'] = {str(n): sum(sum(i in summary[m]['rgb_nn->geo_nn']['fixed_ids'] for m in complete) == n for i in order) for n in range(len(complete) + 1)}
        fixed, broken = (sum(summary[m]['rgb_nn->geo_nn'][k] for m in complete) for k in ('fixed', 'broken'))
        cross['pooled_maps_vs_frames'] = {'fixed': fixed, 'broken': broken, 'pairs': len(complete) * len(order)}
        for name, a, b in [('maps_minus_frames_pp', 'rgb_nn', 'geo_nn'), ('frames_minus_options_pp', 'text_nn', 'rgb_nn')]:
            diff = np.array([[ok[(m, b, i)][1] - ok[(m, a, i)][1] for i in order] for m in complete], float)  # models x items
            boots = [diff[:, rng.integers(0, len(order), len(order))].mean() for _ in range(5000)]
            cross[name] = {'mean': float(100 * diff.mean()), 'ci95': [float(100 * np.percentile(boots, 2.5)), float(100 * np.percentile(boots, 97.5))],
                           'per_model': {m: float(100 * diff[k].mean()) for k, m in enumerate(complete)}}
        prox = [i for i in order if 'Proxemics' in key[i]['labels']]
        cross['proxemics'] = {'n_items': len(prox), 'pairs': len(prox) * len(complete), 'frames': int(sum(ok[(m, 'rgb_nn', i)][1] for m in complete for i in prox)),
                              'maps': int(sum(ok[(m, 'geo_nn', i)][1] for m in complete for i in prox))}
    return summary, cross


def main():
    key = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else OUT / 'answer_key.json').read_text())
    tasks = {t['id']: t for t in json.loads((OUT / 'tasks.json').read_text())}
    answers, gpt_events = load_answers()
    result = {'gpt_event_audit': gpt_events}
    # "None of these" is the empty slot of an item (usually index 4, sometimes 3 or 0); an item is unanswerable under a
    # forced choice only if that slot is the labelled answer. In this sample that never happens, so the two sets coincide.
    none_gold = [i for i in key if 'None of these' in tasks[i]['action_options'][key[i]['action_order'].index(key[i]['correct'])]]
    result['items_with_none_as_labelled_answer'] = none_gold
    for name, ids in [('all60', list(key))] + ([('answerable', [i for i in key if i not in none_gold])] if none_gold else []):
        summary, cross = summarise(ids, key, tasks, answers, np.random.default_rng(42))
        result[name] = {'summary': summary, 'cross_model': cross}
        print(f'=== {name}: {len(ids)} items')
        for model in MODELS:
            s = summary[model]
            cells = ' '.join((f"{s[c]['both']:2d}/{s[c]['n']} ({s[c]['both'] / s[c]['n']:.0%})".rjust(14) if c in s else ' ' * 14) for c in CONDS)
            a, b = s.get('text_nn->rgb_nn'), s.get('rgb_nn->geo_nn')
            print(f'{model:13s} {cells}   frames: ' + (f"{a['fixed']}/{a['broken']} p={a['mcnemar_p']:.4f}" if a else '-').ljust(18) + ' maps: ' + (f"{b['fixed']}/{b['broken']} p={b['mcnemar_p']:.2f}" if b else '-').ljust(13)
                  + ' none: ' + '/'.join(str(s[c]['chose_none']) if c in s else '-' for c in CONDS))
        print('cross-model:', json.dumps({k: v for k, v in cross.items()}))
    print('GPT event audit:', gpt_events)
    (OUT / 'scored_nn.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
