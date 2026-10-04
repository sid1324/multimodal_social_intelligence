"""Run with python -m analysis.idea2 --help from the repository root."""
import argparse
import importlib.metadata
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import digest, inspect, normalize, partition_feasibility, save_json


def main():
    parser = argparse.ArgumentParser(description='VideoNorms Assignment 2: inspect, prepare, annotate, analyze. No training.')
    parser.add_argument('--config', default='analysis/idea2/config.json')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inspect', help='Inspect real field names without assuming semantics')
    p.add_argument('--input', required=True)
    p.add_argument('--records-key')
    for name in ['prepare', 'media', 'analyze', 'language', 'run']:
        sub.add_parser(name)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config['output_dir'])
    for name in ['tables', 'figures', 'embeddings', 'logs']:
        (out / name).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'logs' / 'workflow.log')])
    try:
        if args.command == 'inspect':
            inspect(args.input, out / 'tables', args.records_key)
        elif args.command == 'prepare':
            from .annotations import prepare
            df = normalize(config)
            partition_feasibility(df, config)
            prepare(df, config)
        elif args.command == 'media':
            from .annotations import build_media
            build_media(config)
        else:
            from .statistics import analyze_audit, analyze_humans, merge_humans
            dest = Path(config['data_dir'])
            if not (dest / 'preparation_manifest.json').exists():
                raise ValueError('No real sample prepared yet. See docs/data_access_status.md, then inspect and prepare.')
            manifest = json.loads((dest / 'preparation_manifest.json').read_text())
            # Embedding settings may be changed; annotation inputs and IDs may not.
            for key in ['dataset', 'sampling', 'annotation', 'seed']:
                if config[key] != manifest['config'][key]:
                    raise ValueError(f'{key} changed since preparation. Do not mix study configurations.')
            for filename, key in [('video_norms_sample.csv', 'sample_sha256'), ('human_sample_ids.csv', 'human_ids_sha256')]:
                if digest(dest / filename) != manifest[key]:
                    raise ValueError(f'{filename} changed since preparation. Refusing to analyze altered IDs or gold data.')
            sample = pd.read_csv(dest / 'video_norms_sample.csv', dtype=str, keep_default_na=False)
            if sample.sample_id.duplicated().any():
                raise ValueError('Sample has duplicate record IDs.')
            humans = merge_humans(config, sample)
            if args.command in ['analyze', 'run']:
                analyze_audit(config, sample)
                analyze_humans(config, humans)
            if args.command in ['language', 'run']:
                from .language import embed, language_analysis
                rows, vectors = embed(config, sample, humans)
                language_analysis(config, sample, humans, rows, vectors)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        save_json({'command': args.command, 'time_utc': stamp, 'config': config,
                   'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']}},
                  out / 'logs' / f'{stamp}_{args.command}.json')
    except (ValueError, FileNotFoundError, KeyError) as exc:
        logging.error('%s', exc)
        raise SystemExit(2) from exc


if __name__ == '__main__':
    main()
