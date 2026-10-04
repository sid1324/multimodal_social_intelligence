"""Build a local page for inspecting the items that every model gets wrong with frames (forced-choice run)."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-clean60'
MODELS = ['Qwen3-VL-32B', 'GPT-5.5', 'GPT-5.6', 'GPT-6', 'Sonnet 5.5', 'Opus 5.5']
key = json.loads((OUT / 'answer_key.json').read_text())
tasks = {t['id']: t for t in json.loads((OUT / 'tasks.json').read_text())}
data = json.loads((ROOT / 'data/EgoNormia/annotations/final_data.json').read_text())
answers = {}
for path in (OUT / 'results/claude_single').glob('*_nn__*.json'):
    name, cond, _ = path.name.split('__')
    r = json.loads(path.read_text())
    answers[({'opus': 'Opus 5.5', 'sonnet': 'Sonnet 5.5'}[name], cond, r['id'])] = (r['action_position'], r['justification_position'])
for folder, model in [('gpt', 'GPT-5.6'), ('gpt-5.5', 'GPT-5.5'), ('gpt-6-sol', 'GPT-6')]:
    for path in (OUT / 'results' / folder).glob('*_nn__*.json'):
        r = json.loads(path.read_text())
        answers[(model, r['condition'], r['id'])] = (r['action_position'], r['justification_position'])
for r in json.loads((OUT / 'results/qwen32b_nn.json').read_text()):
    if r['action_position']:
        answers[('Qwen3-VL-32B', r['condition'], r['id'])] = (r['action_position'], r['justification_position'])


def chosen(model, cond, item):
    """Original index of the chosen action, and whether action and justification are both right."""
    a, j = answers[(model, cond, item)]
    k = key[item]
    a0, j0 = k['action_order'][a - 1], k['justification_order'][j - 1]
    return a0, a0 == k['correct'] and j0 == k['correct']


hard = [i for i in key if not any(chosen(m, 'rgb_nn', i)[1] for m in MODELS)]
parts = ['<!doctype html><meta charset="utf-8"><title>Items every model misses</title>',
         '<style>body{font:15px/1.45 -apple-system,Helvetica,sans-serif;margin:24px;color:#111;background:#fafaf8;max-width:1500px}'
         'img{width:100%;display:block;margin:4px 0 10px}h2{margin-top:44px;border-top:1px solid #ccc;padding-top:18px}'
         '.gold{background:#d9f2dc;font-weight:600}td,th{padding:3px 8px;text-align:left;vertical-align:top}table{border-collapse:collapse;margin:8px 0}'
         'small{color:#555}.lab{font-size:12px;color:#555;margin-top:8px}</style>',
         f'<h1>The {len(hard)} items every model gets wrong with frames</h1>',
         '<p>For each item: the five frames, the depth and segmentation maps the models saw, the options (labelled answer in green), '
         'and the action each model chose with frames and with frames plus maps.</p>']
for n, item in enumerate(hard, 1):
    x, k, t = data[item], key[item], tasks[item]
    parts.append(f'<h2>{n}. {item} <small>{", ".join(k["labels"]) or "no category"}</small></h2>')
    for kind, label in [('rgb', 'RGB frames (left to right in time)'), ('depth', 'Estimated depth (brighter = nearer)'), ('segmentation', 'Segmentation (labels can be wrong)')]:
        parts.append(f'<div class="lab">{label}</div><img src="grids/{item}_{kind}.jpg">')
    parts.append('<table><tr><th>#</th><th>Action</th><th>Justification</th></tr>')
    for i in range(5):
        if not x['behaviors'][i]:
            continue
        cls = ' class="gold"' if i == k['correct'] else ''
        parts.append(f'<tr{cls}><td>{i}</td><td>{html.escape(x["behaviors"][i])}</td><td>{html.escape(x["justifications"][i])}</td></tr>')
    parts.append('</table><table><tr><th>Model</th><th>Action chosen with frames</th><th>Action chosen with frames + maps</th></tr>')
    for m in MODELS:
        cells = []
        for cond in ['rgb_nn', 'geo_nn']:
            a0, ok = chosen(m, cond, item)
            text = x['behaviors'][a0] or 'none of these'
            cells.append(f'<td{" class=gold" if ok else ""}>{a0}: {html.escape(text)}</td>')
        parts.append(f'<tr><td>{m}</td>{"".join(cells)}</tr>')
    parts.append('</table>')
(OUT / 'always_wrong.html').write_text('\n'.join(parts))
print('items:', [(i[:8], key[i]['labels']) for i in hard])
for item in hard:
    x = data[item]
    print(item[:8], '| gold:', x['behaviors'][key[item]['correct']][:70], '| most chosen with frames:', max(set(chosen(m, 'rgb_nn', item)[0] for m in MODELS), key=[chosen(m, 'rgb_nn', item)[0] for m in MODELS].count))
