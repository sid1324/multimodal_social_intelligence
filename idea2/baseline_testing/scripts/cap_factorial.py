"""Paired 2×2 model-by-response-cap diagnostic on frozen EgoNormia IDs."""
import json
from pathlib import Path
import numpy as np

from social_rl.common import ROOT, STORE, read_jsonl, write_json
from social_rl.tasks import item_metrics


def cluster_ci(rows, values, seed=777, iterations=2000):
    groups = {}
    for row, value in zip(rows, values):
        groups.setdefault(row['group'], []).append(value)
    if len(groups) < 2:
        return [None, None]
    arrays = [np.asarray(x, dtype=float) for x in groups.values()]
    sums = np.asarray([a.sum() for a in arrays])
    counts = np.asarray([len(a) for a in arrays])
    draws = np.random.default_rng(seed).integers(0, len(arrays), size=(iterations, len(arrays)))
    return np.quantile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [.025, .975]).tolist()


def main():
    panel = read_jsonl(STORE / 'data/manifests/ego_midterm_panel_accessible.jsonl')
    ids = (ROOT / 'egonormia_400_sample_ids.txt').read_text().splitlines()[:40]
    if [r['id'] for r in panel[:40]] != ids:
        raise ValueError('The first-40 cap panel differs from the supplied ID order')
    rows = panel[:40]
    sources = {
        'base192': STORE / 'runs/base_ego/predictions.jsonl',
        'rloo192': STORE / 'runs/rloo_ego/predictions.jsonl',
        'base384': STORE / 'runs/base_ego_cap384/predictions.jsonl',
        'rloo384': STORE / 'runs/rloo_ego_cap384/predictions.jsonl',
    }
    predictions = {}
    for name, path in sources.items():
        data = {p['id']: p for p in read_jsonl(path) if p['condition'] == 'video' and p['id'] in ids}
        if set(data) != set(ids):
            raise ValueError(f'{name} lacks the exact 40-ID panel')
        predictions[name] = data
    scores = {name: np.asarray([item_metrics(row, predictions[name][row['id']])['joint'] for row in rows])
              for name in sources}
    cells = {}
    for name, data in predictions.items():
        invalid = sum(not all(data[uid]['valid'].values()) for uid in ids)
        capped = {task: sum(data[uid]['generation'][task]['hit_token_cap'] for uid in ids)
                  for task in ['action', 'justification', 'sensibility']}
        cells[name] = {'joint_correct': int(scores[name].sum()), 'joint_accuracy': float(scores[name].mean()),
                       'invalid_examples': invalid, 'cap_hits': capped}
    effects = {}
    for label, left, right in [
        ('base_cap_384_minus_192', 'base384', 'base192'),
        ('rloo_cap_384_minus_192', 'rloo384', 'rloo192'),
        ('rloo_minus_base_at_192', 'rloo192', 'base192'),
        ('rloo_minus_base_at_384', 'rloo384', 'base384'),
    ]:
        values = scores[left] - scores[right]
        effects[label] = {'difference': float(values.mean()), 'ci95': cluster_ci(rows, values)}
    interaction = (scores['rloo384'] - scores['base384']) - (scores['rloo192'] - scores['base192'])
    effects['model_by_cap_interaction'] = {'difference': float(interaction.mean()),
                                           'ci95': cluster_ci(rows, interaction)}
    out = ROOT / 'reports/results'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'cap_factorial_40.json', {'n': 40, 'source_groups': len({r['group'] for r in rows}),
                                               'cells': cells, 'paired_effects': effects,
                                               'panel': 'first 40 IDs in supplied order',
                                               'exploratory': True})
    lines = ['# Exploratory response-cap diagnostic', '',
             'The same first 40 supplied IDs are scored in all four cells. Native-style joint accuracy is shown.', '',
             '| Model | Cap | Joint correct / 40 | Invalid examples |', '|---|---:|---:|---:|']
    for name, model, cap in [('base192','Base',192), ('rloo192','RLOO',192),
                             ('base384','Base',384), ('rloo384','RLOO',384)]:
        c = cells[name]
        lines.append(f"| {model} | {cap} | {c['joint_correct']} ({100*c['joint_accuracy']:.1f}%) | {c['invalid_examples']} |")
    lines += ['', 'Paired effects and source-group bootstrap intervals are in `cap_factorial_40.json`.',
              'This slice is exploratory and does not replace the 140-ID primary comparison.']
    (out / 'cap_factorial_40.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'cells': cells, 'effects': effects}, indent=2))


if __name__ == '__main__':
    main()
