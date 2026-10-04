import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from analysis.idea2.data import save_json


def main():
    parser = argparse.ArgumentParser(description='Social Genome public-input feasibility study; no gold labels or training.')
    parser.add_argument('--config', default='analysis/social_genome/config.json')
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('inspect')
    fetch = subs.add_parser('fetch')
    fetch.add_argument('--limit', type=int)
    importer = subs.add_parser('import-responses')
    importer.add_argument('--annotator', required=True)
    importer.add_argument('--file', required=True)
    corrections = subs.add_parser('apply-corrections')
    corrections.add_argument('--file', required=True)
    for name in ['captions', 'audio', 'prepare', 'refresh-pages', 'analyze', 'language', 'run']:
        subs.add_parser(name)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config['output_dir'])
    for name in ['logs', 'tables', 'figures', 'embeddings']:
        (out / name).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'logs/workflow.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    if not config['ta_discussion_confirmed']:
        parser.error('Discuss the revised analyses with your primary TA before proceeding (assignment page 2).')
    from .access import metadata, fetch_media, refresh_captions
    if args.command == 'inspect':
        metadata(config, download=True)
    elif args.command == 'import-responses':
        from .study import import_responses
        import_responses(config, args.annotator, args.file)
    elif args.command == 'apply-corrections':
        from .study import apply_corrections
        apply_corrections(config, args.file)
    elif args.command == 'fetch':
        ready = fetch_media(config, args.limit)
        if ready < (config['sample_size'] if args.limit is None else args.limit):
            raise RuntimeError('Requested media are unavailable; inspect media_access.csv. No replacement or human result invented.')
        refresh_captions(config)
    elif args.command == 'audio':
        from .access import fetch_audio
        fetch_audio(config)
    elif args.command == 'captions':
        refresh_captions(config)
    elif args.command == 'prepare':
        from .study import prepare
        prepare(config)
    elif args.command == 'refresh-pages':
        from .study import refresh_pages
        refresh_pages(config)
    else:
        from .study import load_study, analyze_humans
        sample, humans = load_study(config)
        if args.command in ['analyze', 'run']:
            analyze_humans(config, sample, humans)
        if args.command in ['language', 'run']:
            from .language import analyze_language
            analyze_language(config, sample, humans)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    save_json({'command': args.command, 'time_utc': stamp, 'config': config}, out / 'logs' / f'{stamp}.json')


if __name__ == '__main__':
    main()
