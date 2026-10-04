"""Score the repeated-RGB control (rep_nn) against frames (rgb_nn) and maps (geo_nn) for every model."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clean60_score_all import mcnemar_exact

OUT = Path(__file__).resolve().parents[1] / 'output/idea5-clean60'
key = json.loads((OUT / 'answer_key.json').read_text())
answers = {}
for path in (OUT / 'results/claude_single').glob('*_nn__*.json'):
    name, cond, _ = path.name.split('__')
    r = json.loads(path.read_text())
    answers[({'opus': 'Opus 5.5', 'sonnet': 'Sonnet 5.5'}[name], cond, r['id'])] = (r['action_position'], r['justification_position'], r.get('images_sent'))
for folder, model in [('gpt', 'GPT-5.6'), ('gpt-5.5', 'GPT-5.5'), ('gpt-6-sol', 'GPT-6')]:
    for path in (OUT / 'results' / folder).glob('*_nn__*.json'):
        r = json.loads(path.read_text())
        answers[(model, r['condition'], r['id'])] = (r['action_position'], r['justification_position'], None)
for name in ['qwen32b_nn.json', 'qwen32b_controls.json']:
    for r in json.loads((OUT / 'results' / name).read_text()):
        if r['action_position']:
            answers[('Qwen3-VL-32B', r['condition'], r['id'])] = (r['action_position'], r['justification_position'], None)


def right(model, cond, item):
    pos = answers.get((model, cond, item))
    k = key[item]
    return bool(pos) and k['action_order'][pos[0] - 1] == k['correct'] and k['justification_order'][pos[1] - 1] == k['correct']


result = {}
print(f"{'model':13s} {'frames':>7s} {'RGBx3':>6s} {'maps':>5s}   RGBx3 vs frames (fixed/broken, p)   maps vs RGBx3 (fixed/broken, p)   answered")
for model in ['Qwen3-VL-32B', 'GPT-5.5', 'GPT-5.6', 'GPT-6', 'Sonnet 5.5', 'Opus 5.5']:
    n = sum((model, 'rep_nn', i) in answers for i in key)
    if n < len(key):
        print(f'{model:13s} repeated-RGB run incomplete: {n} of {len(key)}')
        continue
    c = {cond: sum(right(model, cond, i) for i in key) for cond in ['rgb_nn', 'rep_nn', 'geo_nn']}
    f1 = sum(not right(model, 'rgb_nn', i) and right(model, 'rep_nn', i) for i in key)
    b1 = sum(right(model, 'rgb_nn', i) and not right(model, 'rep_nn', i) for i in key)
    f2 = sum(not right(model, 'rep_nn', i) and right(model, 'geo_nn', i) for i in key)
    b2 = sum(right(model, 'rep_nn', i) and not right(model, 'geo_nn', i) for i in key)
    result[model] = {'frames': c['rgb_nn'], 'rgb_x3': c['rep_nn'], 'maps': c['geo_nn'], 'rgbx3_vs_frames': [f1, b1, mcnemar_exact(f1, b1)], 'maps_vs_rgbx3': [f2, b2, mcnemar_exact(f2, b2)]}
    print(f"{model:13s} {c['rgb_nn']:7d} {c['rep_nn']:6d} {c['geo_nn']:5d}   {f1}/{b1}, p={mcnemar_exact(f1, b1):.2f}".ljust(66) + f"{f2}/{b2}, p={mcnemar_exact(f2, b2):.2f}".ljust(34) + f'{n}')
if len(result) == 6:
    tot = {k: sum(v[k] for v in result.values()) for k in ['frames', 'rgb_x3', 'maps']}
    print('pooled over six models (of 360):', tot, '| RGBx3 vs frames fixed/broken:', sum(v['rgbx3_vs_frames'][0] for v in result.values()), '/', sum(v['rgbx3_vs_frames'][1] for v in result.values()),
          '| maps vs RGBx3 fixed/broken:', sum(v['maps_vs_rgbx3'][0] for v in result.values()), '/', sum(v['maps_vs_rgbx3'][1] for v in result.values()))
imgs = {a[2] for (m, c, i), a in answers.items() if c == 'rep_nn' and a[2] is not None}
print('images sent per Claude repeated-RGB call:', imgs)
(OUT / 'repeat_control_scored.json').write_text(json.dumps(result, indent=1))
