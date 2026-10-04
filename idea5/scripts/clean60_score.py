"""Score the clean-60 pilot. Usage: clean60_score.py <answer_key.json>"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-clean60'
CONDS = ['text', 'rgb', 'geo']


def main():
    key = json.loads(Path(sys.argv[1]).read_text())
    answers = {}  # (model, cond, id) -> (action_position, justification_position)
    for model in ['sonnet', 'opus']:
        for path in sorted((OUT / 'results').glob(f'{model}_*.json')):
            cond = path.stem.split('_')[1]
            for r in json.loads(path.read_text())['items']:
                answers[(model, cond, r['id'])] = (r['action_position'], r['justification_position'])
    for path in (OUT / 'results/gpt').glob('*.json'):
        r = json.loads(path.read_text())
        answers[('gpt', r['condition'], r['id'])] = (r['action_position'], r['justification_position'])
    rows = {}
    for (model, cond, item), (a, j) in answers.items():
        k = key[item]
        a0, j0 = k['action_order'][a - 1], k['justification_order'][j - 1]
        rows[(model, cond, item)] = {'action': a0, 'justification': j0, 'action_correct': a0 == k['correct'], 'both_correct': a0 == k['correct'] and j0 == k['correct']}
    summary = {}
    for model in ['sonnet', 'opus', 'gpt']:
        ids = [i for i in key if all((model, c, i) in rows for c in CONDS)]
        s = {'n_complete': len(ids)}
        for c in CONDS:
            have = [i for i in key if (model, c, i) in rows]
            s[c] = {'n': len(have), 'action': sum(rows[(model, c, i)]['action_correct'] for i in have), 'both': sum(rows[(model, c, i)]['both_correct'] for i in have)}
        for a, b in [('text', 'rgb'), ('rgb', 'geo')]:
            s[f'{a}->{b}'] = {'fixed': sum(not rows[(model, a, i)]['both_correct'] and rows[(model, b, i)]['both_correct'] for i in ids),
                              'broken': sum(rows[(model, a, i)]['both_correct'] and not rows[(model, b, i)]['both_correct'] for i in ids),
                              'answer_changed': sum((rows[(model, a, i)]['action'], rows[(model, a, i)]['justification']) != (rows[(model, b, i)]['action'], rows[(model, b, i)]['justification']) for i in ids)}
        prox = [i for i in ids if 'Proxemics' in key[i]['labels']]
        s['proxemics'] = {'n': len(prox), **{c: sum(rows[(model, c, i)]['both_correct'] for i in prox) for c in CONDS}}
        s['rgb_fail_geo_ok_ids'] = [i for i in ids if not rows[(model, 'rgb', i)]['both_correct'] and rows[(model, 'geo', i)]['both_correct']]
        summary[model] = s
    (OUT / 'scored.json').write_text(json.dumps({'summary': summary, 'rows': [{'model': m, 'condition': c, 'id': i, **v} for (m, c, i), v in rows.items()]}, indent=1))
    for model, s in summary.items():
        print(model, 'complete items', s['n_complete'])
        for c in CONDS:
            print(f"   {c:5s} both {s[c]['both']:2d}/{s[c]['n']}  action {s[c]['action']:2d}/{s[c]['n']}")
        print('   text->rgb', s['text->rgb'], '\n   rgb->geo ', s['rgb->geo'], '\n   proxemics', s['proxemics'])
    shared = [set(summary[m]['rgb_fail_geo_ok_ids']) for m in summary]
    print('rgb-fail/geo-ok ids per model:', [len(x) for x in shared], 'shared by >=2 models:', sorted(i[:8] for i in key if sum(i in x for x in shared) >= 2))
    print('gold index distribution:', {v: sum(k['correct'] == v for k in key.values()) for v in range(5)})


if __name__ == '__main__':
    main()
