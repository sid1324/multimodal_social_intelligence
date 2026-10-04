"""Where do answers change? Right/wrong changes in the clean-60 pilot, broken down by norm category.

Usage: clean60_by_category.py [nn|native]   (nn = prompts that forbid "None of these"; native = original prompts)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-clean60'
variant = sys.argv[1] if len(sys.argv) > 1 else 'nn'
OPT, RGB, GEO = ('text_nn', 'rgb_nn', 'geo_nn') if variant == 'nn' else ('textforced', 'rgb', 'geo')
MODELS = ['Qwen3-VL-32B', 'GPT-5.5', 'GPT-5.6', 'GPT-6', 'Sonnet 5.5', 'Opus 5.5']
CATS = ['Proxemics', 'Safety', 'Politeness', 'Cooperation', 'Communication/Legibility', 'Coordination/Proactivity', 'Privacy']

key = json.loads((OUT / 'answer_key.json').read_text())
answers = {}
for path in (OUT / 'results/claude_single').glob('*.json'):
    name, cond, _ = path.name.split('__')
    r = json.loads(path.read_text())
    answers[({'opus': 'Opus 5.5', 'sonnet': 'Sonnet 5.5'}[name], cond, r['id'])] = (r['action_position'], r['justification_position'])
for folder, model in [('gpt', 'GPT-5.6'), ('gpt-5.5', 'GPT-5.5'), ('gpt-6-sol', 'GPT-6')]:
    for path in (OUT / 'results' / folder).glob('*.json'):
        r = json.loads(path.read_text())
        answers[(model, r['condition'], r['id'])] = (r['action_position'], r['justification_position'])
for name in ['qwen32b.json', 'qwen32b_nn.json']:
    if (OUT / 'results' / name).exists():
        for r in json.loads((OUT / 'results' / name).read_text()):
            if r['action_position']:
                answers[('Qwen3-VL-32B', r['condition'], r['id'])] = (r['action_position'], r['justification_position'])


def right(model, cond, item):
    pos = answers.get((model, cond, item))
    if pos is None:
        return None
    k = key[item]
    return k['action_order'][pos[0] - 1] == k['correct'] and k['justification_order'][pos[1] - 1] == k['correct']


models = [m for m in MODELS if all(right(m, c, i) is not None for c in (OPT, RGB, GEO) for i in key)]
print(f'variant={variant}; models with all three conditions complete: {models}')
members = {c: [i for i in key if c in key[i]['labels']] for c in CATS}
members['(no label)'] = [i for i in key if not key[i]['labels']]

print('\nAccuracy by norm category, pooled over models (an item can carry several categories)')
print(f"{'category':26s} {'items':>5s} {'options':>9s} {'+frames':>9s} {'+maps':>9s}   {'maps fixed':>10s} {'maps broke':>10s} {'net':>4s}")
table = {}
for c, ids in members.items():
    if not ids:
        continue
    n = len(ids) * len(models)
    acc = {cond: sum(right(m, cond, i) for m in models for i in ids) for cond in (OPT, RGB, GEO)}
    fixed = sum(not right(m, RGB, i) and right(m, GEO, i) for m in models for i in ids)
    broke = sum(right(m, RGB, i) and not right(m, GEO, i) for m in models for i in ids)
    table[c] = {'items': len(ids), 'pairs': n, 'options': acc[OPT], 'frames': acc[RGB], 'maps': acc[GEO], 'maps_fixed': fixed, 'maps_broke': broke}
    print(f"{c:26s} {len(ids):5d} {acc[OPT] / n:9.0%} {acc[RGB] / n:9.0%} {acc[GEO] / n:9.0%}   {fixed:10d} {broke:10d} {fixed - broke:+4d}")
n = len(key) * len(models)
tot = {cond: sum(right(m, cond, i) for m in models for i in key) for cond in (OPT, RGB, GEO)}
fx = sum(not right(m, RGB, i) and right(m, GEO, i) for m in models for i in key)
bk = sum(right(m, RGB, i) and not right(m, GEO, i) for m in models for i in key)
print(f"{'ALL ITEMS':26s} {len(key):5d} {tot[OPT] / n:9.0%} {tot[RGB] / n:9.0%} {tot[GEO] / n:9.0%}   {fx:10d} {bk:10d} {fx - bk:+4d}")

print('\nPer model: accuracy by category with frames -> with maps (correct / items)')
for m in models:
    cells = [f"{c.split('/')[0][:6]} {sum(right(m, RGB, i) for i in members[c])}->{sum(right(m, GEO, i) for i in members[c])}/{len(members[c])}" for c in CATS]
    print(f'  {m:13s} ' + ' | '.join(cells))

print('\nItems whose answer flips when maps are added, with how many models flip each way')
flips = defaultdict(lambda: [[], []])
for m in models:
    for i in key:
        if not right(m, RGB, i) and right(m, GEO, i):
            flips[i][0].append(m)
        if right(m, RGB, i) and not right(m, GEO, i):
            flips[i][1].append(m)
for i, (f, b) in sorted(flips.items(), key=lambda kv: -(len(kv[1][0]) + len(kv[1][1]))):
    print(f"  {i[:8]}  fixed in {len(f)} ({', '.join(f) or '-'}); broken in {len(b)} ({', '.join(b) or '-'});  labels: {', '.join(key[i]['labels']) or '(none)'}")
print(f'\nitems that never flip: {len(key) - len(flips)} of {len(key)}; flip in both directions (fixed for one model, broken for another): {sum(bool(f) and bool(b) for f, b in flips.values())}')
stable_wrong = [i for i in key if not any(right(m, RGB, i) for m in models)]
print('items wrong with frames for every model:', [(i[:8], key[i]['labels']) for i in stable_wrong])

print('\nOptions-only correctness by category, per model')
for m in models:
    print(f'  {m:13s} ' + ' | '.join(f"{c.split('/')[0][:6]} {sum(right(m, OPT, i) for i in members[c])}/{len(members[c])}" for c in CATS) + f"  | all {sum(right(m, OPT, i) for i in key)}/{len(key)}")
(OUT / f'by_category_{variant}.json').write_text(json.dumps({'models': models, 'by_category': table, 'flips': {i: {'fixed_in': f, 'broken_in': b, 'labels': key[i]['labels']} for i, (f, b) in flips.items()}}, indent=1))
