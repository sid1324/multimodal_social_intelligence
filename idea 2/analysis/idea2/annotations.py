"""Blank, allow-listed human packets; human responses are never generated."""
import html
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .data import digest, sample_records, save_json

LOG = logging.getLogger(__name__)
AUDIT_FIELDS = ['evidence_available', 'evidence_modality', 'visual_evidence_present',
                'verbal_evidence_present', 'requires_audio_or_dialogue',
                'requires_temporal_reasoning', 'cue_types', 'clearly_observable',
                'observable_in_planned_input', 'ambiguity_level', 'notes']
RESPONSE_FIELDS = ['predicted_judgment', 'human_evidence', 'confidence', 'evidence_modality', 'response_status']
BLIND_FIELDS = ['sample_id', 'norm_text', 'input_mode', 'media_files', 'transcript']
MODES = {'silent_frames', 'video_audio', 'video_transcript'}


def blind_rows(df, mode):
    if mode not in MODES:
        raise ValueError(f'Select input_mode from {sorted(MODES)}; no modality assumption is made.')
    result = pd.DataFrame({'sample_id': df.sample_id, 'norm_text': df.norm_text,
                           'input_mode': mode, 'media_files': '',
                           'transcript': df.transcript if mode == 'video_transcript' else ''})
    for col in RESPONSE_FIELDS:
        result[col] = ''
    return result


def prepare(df, config):
    dest = Path(config['data_dir'])
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Preparation will not overwrite existing annotation work. Use a new data_dir for a new sample.')
    mode = config['annotation']['input_mode']
    if mode not in MODES:
        raise ValueError('Set annotation.input_mode after deciding the intended VLM input.')
    if df.norm_text.eq('').any() or df.gold_norm_judgment.eq('').any():
        LOG.warning('Incomplete norm/label rows excluded from sampling, see data_issues.csv.')
    eligible = df[df.norm_text.ne('') & df.gold_norm_judgment.ne('')].copy()
    for column, values in config['sampling']['filters'].items():
        if column not in eligible or not isinstance(values, list):
            raise ValueError('Sampling filters require mapped concept names and lists of observed values.')
        eligible = eligible[eligible[column].isin(values)]
    opts = config['sampling']
    sample = sample_records(eligible, opts['size'], config['seed'], opts['one_per_source'])
    sample['native_record_id'] = sample.sample_id
    sample['sample_id'] = sample.sample_id.map(lambda x: 'vn-a2-' + hashlib.sha256(x.encode()).hexdigest()[:16])
    human = sample_records(sample, opts['human_size'], config['seed'] + 1, opts['one_per_source'])
    dest.mkdir(parents=True, exist_ok=True)
    sample.to_csv(dest / 'video_norms_sample.csv', index=False)
    sample[['sample_id', 'native_record_id', 'source_video_id']].to_csv(dest / 'sampled_ids.csv', index=False)
    human[['sample_id']].to_csv(dest / 'human_sample_ids.csv', index=False)
    audit = sample.copy()
    for col in AUDIT_FIELDS:
        audit[col] = ''
    audit.to_csv(dest / 'audit_template.csv', index=False)
    for i, name in enumerate(config['annotation']['annotators']):
        if not name.isidentifier():
            raise ValueError('Annotator names must be simple identifiers.')
        blind = blind_rows(human.sample(frac=1, random_state=config['seed'] + i + 2), mode)
        folder = dest / 'blinded' / name
        folder.mkdir(parents=True)
        blind.to_csv(folder / f'human_annotations_{name}.csv', index=False)
    save_json({'seed': config['seed'], 'dataset_rows': len(df), 'eligible_rows': len(eligible),
               'sample_sha256': digest(dest / 'video_norms_sample.csv'),
               'human_ids_sha256': digest(dest / 'human_sample_ids.csv'),
               'sample_rows': len(sample), 'human_rows': len(human),
               'labels': sample.gold_norm_judgment.value_counts().to_dict(),
               'note': 'Exploratory label-balanced sample; field presence is not observability.',
               'config': config}, dest / 'preparation_manifest.json')
    LOG.info('Prepared %d audit rows and %d blank responses per annotator.', len(sample), len(human))


def build_media(config):
    """Build a separate portable packet for each annotator, without source filenames."""
    dest = Path(config['data_dir'])
    manifest = json.loads((dest / 'preparation_manifest.json').read_text())
    for key in ['dataset', 'sampling', 'annotation', 'seed']:
        if manifest['config'][key] != config[key]:
            raise ValueError('Study configuration changed since preparation. Use the frozen preparation config.')
    if digest(dest / 'video_norms_sample.csv') != manifest['sample_sha256']:
        raise ValueError('Prepared sample changed; regenerate in a new study directory.')
    sample = pd.read_csv(dest / 'video_norms_sample.csv', dtype=str, keep_default_na=False)
    mode = config['annotation']['input_mode']
    cap = config['annotation']['max_media_gb'] * 1024**3
    root = Path(config['dataset']['media_root'])
    issues, used = [], 0
    for name in config['annotation']['annotators']:
        folder = dest / 'blinded' / name
        csv_path = folder / f'human_annotations_{name}.csv'
        blind = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if blind[RESPONSE_FIELDS].ne('').any().any():
            raise ValueError('Media preparation will not rewrite a completed response file.')
        assets = folder / 'media'
        assets.mkdir(exist_ok=True)
        for i, row in blind.iterrows():
            source = sample.set_index('sample_id').loc[row.sample_id]
            path = root / source.video_path
            if not source.video_path or not path.is_file():
                issues.append({'annotator': name, 'sample_id': row.sample_id, 'issue': 'missing_local_video'})
                continue
            if mode == 'video_transcript' and not source.transcript:
                issues.append({'annotator': name, 'sample_id': row.sample_id, 'issue': 'missing_transcript'})
                continue
            # Opaque asset names keep labels/evidence out of media filenames.
            stem = f'item_{i:03d}'
            if mode == 'video_audio':
                target = assets / f'{stem}{path.suffix}'
                if used + path.stat().st_size > cap:
                    raise ValueError('Media budget reached. Increase max_media_gb deliberately or use smaller clips.')
                shutil.copyfile(path, target)
                paths = [target]
            else:
                if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                    raise ValueError('ffmpeg and ffprobe are required for frames or audio-stripped video.')
                if mode == 'video_transcript':
                    target = assets / f'{stem}.mp4'
                    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(path), '-an', '-map_metadata', '-1',
                                    '-vf', 'scale=640:-2', '-c:v', 'libx264', '-crf', '26', str(target)], check=True)
                    paths = [target]
                else:
                    duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                                             '-of', 'default=nw=1:nk=1', str(path)], text=True))
                    paths = []
                    for frame in range(config['annotation']['frame_count']):
                        target = assets / f'{stem}_frame_{frame:02d}.jpg'
                        second = duration * (frame + 0.5) / config['annotation']['frame_count']
                        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(second), '-i', str(path),
                                        '-frames:v', '1', '-vf', 'scale=640:-2', str(target)], check=True)
                        if not target.is_file():
                            raise ValueError(f'Frame extraction produced no image: {row.sample_id}')
                        paths.append(target)
            used += sum(p.stat().st_size for p in paths)
            if used > cap:
                raise ValueError('Media budget exceeded; packet incomplete. Inspect local files before continuing.')
            blind.loc[i, 'media_files'] = '|'.join(str(p.relative_to(folder)) for p in paths)
        blind.to_csv(csv_path, index=False)
        write_interface(blind, folder / 'index.html', name, config['dataset']['label_values'])
    pd.DataFrame(issues, columns=['annotator', 'sample_id', 'issue']).to_csv(dest / 'media_issues.csv', index=False)
    LOG.info('Media packets: %.1f MB, %d unavailable items logged.', used / 1024**2, len(issues))


def write_interface(blind, path, name, labels):
    # Allow-list, rather than removing known gold columns from an untrusted record.
    payload = json.dumps(blind[BLIND_FIELDS].to_dict('records'), ensure_ascii=False).replace('<', '\\u003c')
    label_json = json.dumps(labels, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('annotation.html').read_text()
    template = template.replace('__ROWS__', payload).replace('__LABELS__', label_json)
    template = template.replace('__NAME_JSON__', json.dumps(name)).replace('__NAME__', html.escape(name))
    path.write_text(template, encoding='utf-8')
