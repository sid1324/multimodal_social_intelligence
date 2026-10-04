"""Score the depth-only condition (dep_nn: options + depth grid, no RGB) against options only and options + RGB frames."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from clean60_score_all import mcnemar_exact

OUT = Path(__file__).resolve().parents[1] / 'output/idea5-clean60'
MODELS = ['Qwen3-VL-32B', 'GPT-5.5', 'GPT-5.6', 'GPT-6', 'Sonnet 5.5', 'Opus 5.5']
key = json.loads((OUT / 'answer_key.json').read_text())
tasks = {t['id']: t for t in json.loads((OUT / 'tasks.json').read_text())}
answers = {}
for path in (OUT / 'results/claude_single').glob('*_nn__*.json'):
    name, cond, _ = path.name.split('__')
    r = json.loads(path.read_text())
    answers[({'opus': 'Opus 5.5', 'sonnet': 'Sonnet 5.5'}[name], cond, r['id'])] = (r['action_position'], r['justification_position'])
for folder, model in [('gpt', 'GPT-5.6'), ('gpt-5.5', 'GPT-5.5'), ('gpt-6-sol', 'GPT-6')]:
    for path in (OUT / 'results' / folder).glob('*_nn__*.json'):
        r = json.loads(path.read_text())
        answers[(model, r['condition'], r['id'])] = (r['action_position'], r['justification_position'])
for name in ['qwen32b_nn.json', 'qwen32b_depth_only.json']:
    if (OUT / 'results' / name).exists():
        for r in json.loads((OUT / 'results' / name).read_text()):
            answers[('Qwen3-VL-32B', r['condition'], r['id'])] = (r['action_position'], r['justification_position']) if r['action_position'] else (None, None)


def right(model, cond, item):
    a, j = answers[(model, cond, item)]
    k = key[item]
    return a is not None and k['action_order'][a - 1] == k['correct'] and k['justification_order'][j - 1] == k['correct']


def chose_none(model, cond, item):
    a, _ = answers[(model, cond, item)]
    return a is not None and 'None of these' in tasks[item]['action_options'][a - 1]


ids = list(key)
done = [m for m in MODELS if all((m, 'dep_nn', i) in answers for i in ids)]
print('models complete on depth-only:', done)
print(f"{'model':13s} {'options':>8s} {'+depth':>7s} {'+frames':>8s}   depth vs options (fixed/broken, p)   frames vs depth (fixed/broken, p)   none chosen with depth")
result = {}
for m in done:
    c = {cond: sum(right(m, cond, i) for i in ids) for cond in ['text_nn', 'dep_nn', 'rgb_nn']}
    f1 = sum(not right(m, 'text_nn', i) and right(m, 'dep_nn', i) for i in ids)
    b1 = sum(right(m, 'text_nn', i) and not right(m, 'dep_nn', i) for i in ids)
    f2 = sum(not right(m, 'dep_nn', i) and right(m, 'rgb_nn', i) for i in ids)
    b2 = sum(right(m, 'dep_nn', i) and not right(m, 'rgb_nn', i) for i in ids)
    result[m] = {'options': c['text_nn'], 'depth_only': c['dep_nn'], 'frames': c['rgb_nn'], 'depth_vs_options': [f1, b1, mcnemar_exact(f1, b1)], 'frames_vs_depth': [f2, b2, mcnemar_exact(f2, b2)]}
    print(f"{m:13s} {c['text_nn']:8d} {c['dep_nn']:7d} {c['rgb_nn']:8d}   {f1}/{b1}, p={mcnemar_exact(f1, b1):.3f}".ljust(72) + f"{f2}/{b2}, p={mcnemar_exact(f2, b2):.3f}".ljust(36) + str(sum(chose_none(m, 'dep_nn', i) for i in ids)))
if done:
    rng = np.random.default_rng(42)
    for name, a, b in [('depth minus options', 'text_nn', 'dep_nn'), ('frames minus depth', 'dep_nn', 'rgb_nn')]:
        diff = np.array([[right(m, b, i) - right(m, a, i) for i in ids] for m in done], float)
        boots = [diff[:, rng.integers(0, len(ids), len(ids))].mean() for _ in range(5000)]
        print(f"{name}, averaged over {len(done)} models: {100 * diff.mean():+.1f} points [{100 * np.percentile(boots, 2.5):+.1f}, {100 * np.percentile(boots, 97.5):+.1f}]")
        result[name] = [float(100 * diff.mean()), float(100 * np.percentile(boots, 2.5)), float(100 * np.percentile(boots, 97.5))]
    tot = {k: sum(result[m][k] for m in done) for k in ['options', 'depth_only', 'frames']}
    print('pooled correct of', 60 * len(done), ':', tot, '| depth vs options fixed/broken:', sum(result[m]['depth_vs_options'][0] for m in done), '/', sum(result[m]['depth_vs_options'][1] for m in done))
(OUT / 'depth_only_scored.json').write_text(json.dumps(result, indent=1))
