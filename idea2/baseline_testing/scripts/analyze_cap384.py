"""Paired source-cluster comparison of the 192- and 384-token 400-ID panels."""
import json

from social_rl.common import ROOT, STORE, read_jsonl, write_json
from social_rl.tasks import item_metrics
from summarize import cluster_ci


def main():
    panel = read_jsonl(STORE / 'data/manifests/ego_midterm_panel_accessible.jsonl')
    ids = (ROOT / 'egonormia_400_sample_ids.txt').read_text().splitlines()
    if len(panel) != 400 or [r['id'] for r in panel] != ids:
        raise ValueError('Panel differs from supplied IDs')
    runs = {}
    for kind in ('base', 'rloo', 'sft'):
        for cap, suffix in ((192, 'ego_video400'), (384, 'ego_cap384_video400')):
            preds = read_jsonl(STORE / 'runs' / f'{kind}_{suffix}' / 'predictions.jsonl')
            if [p['id'] for p in preds] != ids or any(p['condition'] != 'video' for p in preds):
                raise ValueError(f'{kind}/{cap}: incomplete or unordered video run')
            runs[f'{kind}_{cap}'] = [item_metrics(row, pred) for row, pred in zip(panel, preds)]
    comparisons = {}
    for left, right in (
        ('base_384', 'base_192'), ('rloo_384', 'rloo_192'),
        ('sft_384', 'sft_192'), ('rloo_384', 'base_384'),
        ('sft_384', 'base_384'), ('sft_384', 'rloo_384'),
    ):
        comparisons[f'{left} minus {right}'] = {
            metric: {
                'difference': sum(a[metric] - b[metric] for a, b in zip(runs[left], runs[right])) / 400,
                'ci95': cluster_ci(panel, [a[metric] - b[metric]
                                          for a, b in zip(runs[left], runs[right])]),
            }
            for metric in ('action', 'justification', 'joint', 'sensibility_iou')
        }
    out = ROOT / 'reports/results/cap384_video400_paired.json'
    write_json(out, comparisons)
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
