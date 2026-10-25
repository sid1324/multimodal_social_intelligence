"""Independent human simulation on public inputs; no accuracy against hidden gold."""
import hashlib
import html
import json
import logging
from pathlib import Path
import shutil

import pandas as pd

from analysis.idea2.data import digest, save_json
from analysis.idea2.statistics import bar_plot, cluster_mean_ci, figure_save, validate_values
from .access import metadata, study_sources

LOG = logging.getLogger(__name__)
INPUTS = ['sample_id', 'question', 'a0', 'a1', 'a2', 'a3', 'transcript', 'media_file', 'input_mode']
RESPONSES = ['response_status', 'selected_answer', 'human_evidence', 'confidence', 'evidence_modality',
             'clearly_observable', 'requires_temporal_reasoning', 'audio_would_help', 'language_understood',
             'cue_types', 'notes']


def response_fields(config):
    return ['audio_used' if f == 'audio_would_help' and config['input_mode'] == 'video_audio_transcript' else f for f in RESPONSES]


def modality_values(config):
    if config['input_mode'] == 'video_audio_transcript':
        return ['visual_only', 'audio_only', 'transcript_only', 'visual_audio', 'visual_transcript', 'audio_transcript', 'all_three', 'unclear']
    return ['visual_only', 'transcript_only', 'both', 'unclear']


def choose_question(rows, seed):
    # Independent of answer options, labels, captions, or future responses.
    return min(rows.to_dict('records'), key=lambda r: hashlib.sha256(f'{seed}:{r["qid"]}'.encode()).hexdigest())


def render_page(config, annotator, inputs):
    audio_enabled = config['input_mode'] == 'video_audio_transcript'
    responses = response_fields(config)
    payload = json.dumps(inputs, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('annotation.html').read_text()
    document = template.replace('__ROWS__', payload).replace('__NAME_JSON__', json.dumps(annotator))
    document = document.replace('__NAME__', html.escape(annotator))
    document = document.replace('__RESPONSE_FIELDS__', json.dumps(responses))
    document = document.replace('__MODALITY_OPTIONS__', ''.join(f'<option>{v}</option>' for v in modality_values(config)))
    document = document.replace('__MEDIA_INSTRUCTION__', 'Watch the video with sound and read its English transcript.' if audio_enabled else 'Watch the silent video and read its English transcript.')
    document = document.replace('__AUDIO_NOTE__', 'Audio is included; turn up your volume before answering.' if audio_enabled else 'Do not interpret silence as behavior: the audio track was deliberately removed.')
    document = document.replace('__VIDEO_MUTED__', '' if audio_enabled else 'muted')
    document = document.replace('__AUDIO_FIELD__', 'audio_used' if audio_enabled else 'audio_would_help')
    document = document.replace('__AUDIO_QUESTION__', 'Did you use information heard in the audio?' if audio_enabled else 'Would hearing the original audio help resolve this question? (Your assessment, not a verified audio fact.)')
    document = document.replace('__CUE_NOTE__', 'You may record tone_prosody or other audible cues you actually heard.' if audio_enabled else 'Do not infer tone/prosody from a silent clip.')
    return document


def refresh_pages(config):
    """Refresh presentation only; never rewrite responses, frozen inputs, or media."""
    dest = Path(config['data_dir'])
    manifest = json.loads((dest / 'manifest.json').read_text())
    if config['input_mode'] != manifest['config']['input_mode'] or config['annotators'] != manifest['config']['annotators']:
        raise ValueError('Page configuration differs from frozen study.')
    for annotator, inputs in manifest['packet_inputs'].items():
        path = dest / 'blinded' / annotator / 'index.html'
        backup = path.with_suffix('.html.before_playback_fix')
        if path.exists() and not backup.exists():
            shutil.copyfile(path, backup)
        path.write_text(render_page(config, annotator, inputs))
    LOG.info('Refreshed HTML only; response CSVs and frozen media unchanged.')


def prepare(config):
    audio_enabled = config['input_mode'] == 'video_audio_transcript'
    clip_name = 'clip_av.mp4' if audio_enabled else 'clip.mp4'
    provenance_name = 'audio_complete.json' if audio_enabled else 'complete.json'
    responses = response_fields(config)
    dest = Path(config['data_dir'])
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Study exists; refusing to overwrite human work. Use a new data_dir for a new study.')
    frame, trims = metadata(config)
    root = Path(config['raw_dir'])
    selected = []
    for vid in study_sources(config, frame):
        folder = root / 'media' / vid
        if not (folder / provenance_name).exists():
            continue
        provenance = json.loads((folder / provenance_name).read_text())
        if digest(folder / clip_name) != provenance['clip_sha256'] or digest(folder / 'transcript.txt') != provenance['transcript_sha256']:
            raise ValueError(f'Media or transcript checksum changed: {vid}')
        row = choose_question(frame[frame.vid_name.eq(vid)], config['seed'])
        selected.append({'sample_id': 'sg-' + hashlib.sha256(row['qid'].encode()).hexdigest()[:16],
                         'qid': row['qid'], 'source_video_id': vid, 'question': row['q'],
                         **{f'a{i}': row[f'a{i}'] for i in range(4)}, 'question_clip_range': row['ts'],
                         'source_start': trims[vid], 'transcript': (folder / 'transcript.txt').read_text(),
                         'caption_source': provenance['caption_source'], 'video_sha256': provenance['clip_sha256'],
                         'transcript_sha256': provenance['transcript_sha256'], 'input_mode': config['input_mode']})
        if len(selected) == config['sample_size']:
            break
    if len(selected) != config['sample_size']:
        raise ValueError(f'Only {len(selected)} verified media pairs; need {config["sample_size"]}. Resolve access or explicitly revise sample_size.')
    sample = pd.DataFrame(selected)
    if config.get('selection_lock'):
        lock = pd.read_csv(config['selection_lock'], dtype=str)
        if sample.qid.tolist() != lock.qid.tolist():
            raise ValueError('Sampled questions do not match frozen selection lock.')
    if not sample.sample_id.is_unique or not sample.source_video_id.is_unique:
        raise ValueError('Sampling must have unique records and one question per source.')
    dest.mkdir(parents=True)
    sample.to_csv(dest / 'sample.csv', index=False)
    sample[['sample_id', 'qid', 'source_video_id', 'source_start', 'question_clip_range']].to_csv(dest / 'sampled_ids.csv', index=False)
    packets = {}
    for i, annotator in enumerate(config['annotators']):
        if not annotator.isidentifier():
            raise ValueError('Annotator names must be simple identifiers.')
        folder = dest / 'blinded' / annotator
        (folder / 'media').mkdir(parents=True)
        shuffled = sample.sample(frac=1, random_state=config['seed']+i+1).reset_index(drop=True)
        blinded = shuffled[['sample_id', 'question', 'a0', 'a1', 'a2', 'a3', 'transcript', 'input_mode']].copy()
        blinded['media_file'] = [f'media/item_{j:02d}.mp4' for j in range(len(blinded))]
        for j, r in shuffled.iterrows():
            shutil.copyfile(root / 'media' / r.source_video_id / clip_name, folder / blinded.iloc[j].media_file)
        for field in responses:
            blinded[field] = ''
        blinded = blinded[INPUTS + responses]
        response = folder / f'human_annotations_{annotator}.csv'
        blinded.to_csv(response, index=False)
        (folder / 'index.html').write_text(render_page(config, annotator, blinded[INPUTS].to_dict('records')))
        packets[annotator] = blinded[INPUTS].to_dict('records')
    save_json({'config': config, 'sample_sha256': digest(dest / 'sample.csv'), 'packet_inputs': packets,
               'n': len(sample), 'selection': 'first available sources in seed-42 shuffled order; one hash-selected question per source',
               'no_gold_labels': True}, dest / 'manifest.json')
    LOG.info('Prepared %d questions with verified media; blank independent packets for %s.', len(sample), ', '.join(config['annotators']))


def load_study(config, response_paths=None):
    responses = response_fields(config)
    dest = Path(config['data_dir'])
    manifest = json.loads((dest / 'manifest.json').read_text())
    for key in ['seed', 'revision', 'sample_size', 'input_mode', 'annotators', 'annotator_languages']:
        if config[key] != manifest['config'][key]:
            raise ValueError(f'Study setting {key} changed; use a new study directory.')
    if digest(dest / 'sample.csv') != manifest['sample_sha256']:
        raise ValueError('Frozen sampled inputs were modified.')
    sample = pd.read_csv(dest / 'sample.csv', dtype=str, keep_default_na=False)
    frames = []
    for annotator in config['annotators']:
        folder = dest / 'blinded' / annotator
        response_path = (response_paths or {}).get(annotator, folder / f'human_annotations_{annotator}.csv')
        df = pd.read_csv(response_path, dtype=str, keep_default_na=False)
        expected = pd.DataFrame(manifest['packet_inputs'][annotator])
        if set(df.columns) != set(INPUTS+responses) or not df.sample_id.is_unique or set(df.sample_id) != set(expected.sample_id):
            raise ValueError(f'{annotator}: unexpected fields, missing IDs or duplicate rows.')
        actual_inputs = df.set_index('sample_id').loc[expected.sample_id, INPUTS[1:]].reset_index()
        if not actual_inputs.equals(expected[INPUTS]):
            raise ValueError(f'{annotator}: question, options, transcript or input configuration was changed.')
        for row in expected.itertuples():
            gold_hash = sample.set_index('sample_id').loc[row.sample_id, 'video_sha256']
            if digest(folder / row.media_file) != gold_hash:
                raise ValueError('Annotation video differs from frozen study.')
        validate_values(df, 'response_status', ['completed', 'unavailable_media', 'language_difficulty', 'cannot_judge'])
        validate_values(df, 'selected_answer', ['A', 'B', 'C', 'D'])
        validate_values(df, 'confidence', ['1', '2', '3', '4', '5'])
        validate_values(df, 'evidence_modality', modality_values(config))
        for field in ['clearly_observable', 'requires_temporal_reasoning', 'audio_used' if config['input_mode'] == 'video_audio_transcript' else 'audio_would_help', 'language_understood']:
            validate_values(df, field, ['yes', 'no', 'uncertain'])
        answered = df.response_status.eq('completed')
        if (answered & (df.selected_answer.eq('') | df.human_evidence.str.strip().eq(''))).any():
            raise ValueError('Completed responses require a selected answer and short observational evidence.')
        if ((~answered) & (df.selected_answer.ne('') | df.human_evidence.str.strip().ne(''))).any():
            raise ValueError('Incomplete/skipped records must have blank answer and evidence fields.')
        df = df[['sample_id']+responses].merge(sample, on='sample_id', validate='one_to_one')
        df['annotator'] = annotator
        frames.append(df)
    return sample, pd.concat(frames, ignore_index=True)


def import_responses(config, annotator, source):
    """Validate an export before replacing the canonical CSV; preserve both versions."""
    from datetime import datetime, timezone
    if annotator not in config['annotators']:
        raise ValueError('Annotator is not assigned to this study.')
    source = Path(source)
    load_study(config, {annotator: source})
    dest = Path(config['data_dir']) / 'blinded' / annotator / f'human_annotations_{annotator}.csv'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = Path(config['data_dir']) / 'annotation_imports' / stamp
    archive.mkdir(parents=True)
    shutil.copyfile(source, archive / 'original_export.csv')
    previous_hash = digest(dest) if dest.exists() else None
    if dest.exists():
        shutil.copyfile(dest, archive / 'previous_canonical.csv')
    if source.resolve() != dest.resolve():
        shutil.copyfile(source, dest)
    save_json({'annotator': annotator, 'source': str(source), 'sha256': digest(source),
               'previous_canonical_sha256': previous_hash, 'time_utc': stamp,
               'validation': 'schema, exact frozen inputs, IDs, values, required responses and media hashes passed'},
              archive / 'provenance.json')
    LOG.info('Imported validated responses for %s; original and previous CSV preserved in %s.', annotator, archive)


def apply_corrections(config, correction_file):
    """Apply only the supplied, human-confirmed field corrections with an audit trail."""
    from datetime import datetime, timezone
    changes = json.loads(Path(correction_file).read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = Path(config['data_dir']) / 'annotation_imports' / (stamp + '_corrections')
    archive.mkdir(parents=True)
    candidates, journal = {}, []
    for change in changes:
        name = change['annotator']
        if name not in config['annotators'] or change['field'] not in response_fields(config) or not change['reason'].strip():
            raise ValueError('Correction requires an assigned annotator, human-response field and documented reason.')
        path = Path(config['data_dir']) / 'blinded' / name / f'human_annotations_{name}.csv'
        if name not in candidates:
            candidates[name] = (path, pd.read_csv(path, dtype=str, keep_default_na=False))
        _, frame = candidates[name]
        index = frame.index[frame.sample_id.eq(change['sample_id'])]
        if len(index) != 1:
            raise ValueError('Correction sample ID must match exactly one record.')
        actual = frame.loc[index[0], change['field']]
        if actual not in [change['old'], change['new']]:
            raise ValueError('Correction does not match current value; refusing to overwrite.')
        frame.loc[index[0], change['field']] = change['new']
        journal.append({**change, 'already_applied': actual == change['new']})
    overrides = {}
    for name, (_, frame) in candidates.items():
        overrides[name] = archive / f'{name}_corrected.csv'
        frame.to_csv(overrides[name], index=False)
    load_study(config, overrides)
    for name, (path, _) in candidates.items():
        shutil.copyfile(path, archive / f'{name}_before.csv')
        shutil.copyfile(overrides[name], path)
    save_json({'time_utc': stamp, 'correction_file': str(correction_file), 'changes': journal}, archive / 'provenance.json')
    LOG.info('Applied %d explicit human-confirmed corrections; originals archived.', len(journal))


def analyze_humans(config, sample, humans):
    out = Path(config['output_dir'])
    tables = out / 'tables'
    humans.to_csv(tables / 'human_responses_merged.csv', index=False)
    completion = humans.assign(status=humans.response_status.replace('', 'pending')).groupby(['annotator', 'status']).size().unstack(fill_value=0)
    completion.to_csv(tables / 'human_completion.csv')
    done = humans[humans.response_status.eq('completed')]
    result = {'status': 'pending_humans' if done.empty else 'exploratory', 'assigned_per_annotator': len(sample),
              'completed_responses': len(done), 'gold_accuracy': None,
              'note': 'Agreement is not correctness. Human cues are observations, not official dataset labels.', 'annotators': {}}
    result['agreement_status'] = 'not_applicable_single_annotator' if len(config['annotators']) == 1 else 'requires_paired_responses'
    for name, group in done.groupby('annotator'):
        stats = {'completed': len(group), 'modality_counts': group.loc[group.evidence_modality.ne(''), 'evidence_modality'].value_counts().to_dict()}
        modality_n = sum(stats['modality_counts'].values())
        stats['modality_recorded'] = modality_n
        stats['modality_blank'] = len(group) - modality_n
        stats['modality_percent'] = {key: 100 * count / modality_n for key, count in stats['modality_counts'].items()}
        for field in ['clearly_observable', 'requires_temporal_reasoning', 'audio_used' if config['input_mode'] == 'video_audio_transcript' else 'audio_would_help', 'language_understood']:
            valid = group[group[field].ne('')]
            stats[field] = {**cluster_mean_ci(valid[field].eq('yes'), valid.source_video_id, config['seed'], config['bootstrap_replicates']),
                            'uncertain': int(valid[field].eq('uncertain').sum()), 'blank': int(group[field].eq('').sum())}
        cues = group[['sample_id', 'cue_types']].assign(cue_types=group.cue_types.str.split(';')).explode('cue_types')
        cues.cue_types = cues.cue_types.str.strip()
        stats['cue_counts'] = cues[cues.cue_types.ne('')].drop_duplicates().cue_types.value_counts().to_dict()
        cue_n = int(group.cue_types.str.strip().ne('').sum())
        stats['cue_recorded'] = cue_n
        stats['cue_percent_among_recorded'] = {key: 100 * count / cue_n for key, count in stats['cue_counts'].items()}
        result['annotators'][name] = stats
        bar_plot(stats['modality_counts'], out, f'A1_modality_{name}', 'Completed responses with modality recorded')
        bar_plot(stats['cue_counts'], out, f'A2_cues_{name}', 'Questions with cue (multiple allowed)')
    if len(config['annotators']) == 2 and len(done):
        wide = done.pivot(index='sample_id', columns='annotator', values='selected_answer')
        if wide.shape[1] == 2:
            wide = wide.dropna()
            if len(wide):
                agree = wide.iloc[:, 0].eq(wide.iloc[:, 1])
                result['answer_agreement'] = cluster_mean_ci(agree, wide.index, config['seed'], config['bootstrap_replicates'])
                wide['agree'] = agree
                wide.to_csv(tables / 'paired_answer_agreement.csv')
                bar_plot({'Agree': int(agree.sum()), 'Disagree': int((~agree).sum())}, out, 'A3_answer_agreement', 'Paired completed questions')
    save_json(result, tables / 'human_statistics.json')
    from .results import audit_tables
    audit_tables(config, humans)
    LOG.info('Human analysis: %d actual responses; no gold accuracy computed.', len(done))
