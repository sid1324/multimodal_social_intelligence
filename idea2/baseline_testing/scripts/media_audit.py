"""Summarize validated media coverage without reading benchmark answer labels."""
from collections import Counter
from pathlib import Path

from social_rl.common import ROOT, STORE, read_jsonl, write_json


def main():
    root = STORE / 'data/manifests'
    history = read_jsonl(STORE / 'data/media_attempts.jsonl')
    latest = {(entry['dataset'], entry['id']): entry for entry in history}
    ego = read_jsonl(root / 'ego_midterm_panel.jsonl')
    r3 = {split: read_jsonl(root / f'r3_{split}_accessible.jsonl') for split in ('train', 'val', 'test')}
    ego_valid = [row for row in ego if latest.get(('ego', row['id']), {}).get('status') == 'ok'
                 and Path(row['video_path']).exists()]
    valid_ids = {row['id'] for row in ego_valid}
    missing = [row for row in ego if row['id'] not in valid_ids]
    r3_attempts = [entry for (dataset, _), entry in latest.items() if dataset == 'r3']
    result = {
        'ego': {'sample_ids': len(ego), 'validated_preceding_clips': len(ego_valid),
                'source_groups_validated': len({row['group'] for row in ego_valid}),
                'verified_validated': sum(row['verified'] for row in ego_valid),
                'missing_or_failed': [{'id': row['id'],
                                       'error': latest.get(('ego', row['id']), {}).get('error', 'not attempted')}
                                      for row in missing]},
        'r3': {'attempted_unique_videos': len(r3_attempts),
               'validated_unique_videos': sum(entry.get('status') == 'ok' for entry in r3_attempts),
               'failure_reasons': dict(Counter(entry.get('error', '')[:100] for entry in r3_attempts
                                               if entry.get('status') != 'ok')),
               'eligible_questions': {split: len(rows) for split, rows in r3.items()},
               'eligible_unique_videos': {split: len({row['video_id'] for row in rows})
                                          for split, rows in r3.items()},
               'eligible_movie_groups': {split: len({row['group'] for row in rows})
                                         for split, rows in r3.items()}},
        'rights': 'R3 annotations CC BY-NC 4.0; source movie/video rights remain original. EgoNormia terms apply.'}
    write_json(ROOT / 'reports/media_audit.json', result)
    print(result)


if __name__ == '__main__':
    main()
