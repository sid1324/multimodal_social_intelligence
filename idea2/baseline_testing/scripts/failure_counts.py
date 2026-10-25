"""Objective failure patterns for midterm discussion; no causal labels inferred."""
import argparse
import json
from collections import Counter
from pathlib import Path

from social_rl.common import read_jsonl, write_json
from social_rl.tasks import item_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = {row['id']: row for row in read_jsonl(args.manifest)}
    if any(row.get('split') == 'final' for row in rows.values()):
        raise ValueError('Reserved final examples cannot enter midterm analysis')
    predictions = {(p['id'], p['condition']): p for p in read_jsonl(Path(args.run) / 'predictions.jsonl')}
    video = {uid: p for (uid, condition), p in predictions.items() if condition == 'video'}
    if not video:
        raise ValueError('No correct-video baseline predictions')
    unknown = set(video) - set(rows)
    if unknown:
        raise ValueError(f'Predictions outside manifest: {sorted(unknown)[:3]}')
    outcomes = Counter()
    invalid = Counter()
    capped = Counter()
    strict = Counter()
    for uid, pred in video.items():
        metric = item_metrics(rows[uid], pred)
        if metric['action'] and metric['justification']:
            outcomes['both_correct'] += 1
        elif metric['action']:
            outcomes['action_only'] += 1
        elif metric['justification']:
            outcomes['justification_only'] += 1
        else:
            outcomes['neither_correct'] += 1
        for task, valid in pred.get('valid', {}).items():
            if not valid:
                invalid[task] += 1
        strict_action = bool(metric['action'] and pred.get('valid', {}).get('action', False))
        strict_justification = bool(metric['justification'] and pred.get('valid', {}).get('justification', False))
        strict_sensibility = bool(pred.get('valid', {}).get('sensibility', False))
        strict['action_correct'] += strict_action
        strict['justification_correct'] += strict_justification
        strict['joint_correct'] += strict_action and strict_justification
        strict['sensibility_valid'] += strict_sensibility
        strict['native_joint_correct_but_invalid'] += bool(metric['joint'] and not (strict_action and strict_justification))
        for task, generation in pred.get('generation', {}).items():
            if generation.get('hit_token_cap'):
                capped[task] += 1
    controls = {}
    for condition in ('blind', 'mismatch'):
        paired = sorted(uid for uid in video if (uid, condition) in predictions)
        if not paired:
            continue
        changes = Counter()
        for uid in paired:
            base = item_metrics(rows[uid], video[uid])['joint']
            control = item_metrics(rows[uid], predictions[uid, condition])['joint']
            changes['both_correct' if base and control else
                    'video_only_correct' if base else
                    'control_only_correct' if control else 'both_incorrect'] += 1
        controls[condition] = {'n_paired': len(paired), **changes}
    result = {'n_panel': len(rows), 'n_video_attempted': len(video),
              'source_groups_attempted': len({rows[uid]['group'] for uid in video}),
              'outcomes': dict(outcomes), 'invalid_by_task': dict(invalid),
              'strict_format_sensitivity': dict(strict),
              'hit_generation_cap_by_task': dict(capped), 'paired_controls': controls}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'failure_counts.json', result)
    lines = ['# Observed failure patterns', '',
             f"Correct-video coverage: {len(video)}/{len(rows)} examples from {result['source_groups_attempted']} source videos.",
             '', '| Action / justification | Count |', '|---|---:|']
    for key, label in [('both_correct', 'Both correct'), ('action_only', 'Action only correct'),
                       ('justification_only', 'Justification only correct'),
                       ('neither_correct', 'Neither correct')]:
        lines.append(f"| {label} | {outcomes[key]} |")
    lines += ['', f"Invalid output counts by task: {dict(invalid)}.",
              f"Strict-format sensitivity counts: {dict(strict)}.",
              f"Generation cap hits by task: {dict(capped)}.", '',
              'These are observable outcomes. Perception, temporal reasoning, norm conflict, and',
              'other causes require inspection of the allowed clip and model responses.']
    if controls:
        lines += ['', 'Paired control outcomes:', '', '| Control | Paired | Video only correct | Control only correct |',
                  '|---|---:|---:|---:|']
        for condition, counts in controls.items():
            lines.append(f"| {condition} | {counts['n_paired']} | {counts.get('video_only_correct', 0)} | "
                         f"{counts.get('control_only_correct', 0)} |")
    (out / 'failure_counts.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
