"""Exploratory estimates with explicit denominators and source-cluster bootstrap."""
import logging
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .annotations import AUDIT_FIELDS, BLIND_FIELDS, RESPONSE_FIELDS
from .data import save_json

LOG = logging.getLogger(__name__)


def cluster_mean_ci(values, groups, seed=42, repeats=2000):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if not len(values):
        return {'n': 0, 'mean': None, 'ci_low': None, 'ci_high': None, 'clusters': 0}
    unique = np.unique(groups)
    parts = [values[groups == x] for x in unique]
    means = []
    rng = np.random.default_rng(seed)
    if len(unique) > 1:
        for _ in range(repeats):
            means.append(float(np.concatenate([parts[j] for j in rng.integers(len(parts), size=len(parts))]).mean()))
    bounds = np.quantile(means, [.025, .975]) if means else [None, None]
    return {'n': len(values), 'mean': float(values.mean()), 'ci_low': None if bounds[0] is None else float(bounds[0]),
            'ci_high': None if bounds[1] is None else float(bounds[1]), 'clusters': len(unique)}


def groups_for(df):
    return np.where(df.source_video_id.ne(''), 'video:' + df.source_video_id, 'record:' + df.sample_id)


def figure_save(fig, out, name):
    out = Path(out) / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def bar_plot(values, out, name, ylabel):
    if len(values) == 0:
        return
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(values)), list(values.values()))
        ax.set_xticks(range(len(values)), list(values), rotation=25, ha='right')
        ax.set_ylabel(ylabel)
        figure_save(fig, out, name)


def validate_values(df, col, allowed):
    bad = set(df[col]) - set(allowed) - {''}
    if bad:
        raise ValueError(f'Invalid {col}: {sorted(bad)}. See annotation instructions.')


def analyze_audit(config, sample):
    audit = pd.read_csv(Path(config['data_dir']) / 'audit_template.csv', dtype=str, keep_default_na=False)
    if audit.sample_id.duplicated().any() or set(audit.sample_id) != set(sample.sample_id):
        raise ValueError('Audit rows must match the sampled IDs exactly.')
    # Use authoritative sample for grouping, not editable copied metadata.
    audit = sample[['sample_id', 'source_video_id']].merge(audit[['sample_id'] + AUDIT_FIELDS], validate='one_to_one')
    result = {'total_sample': len(audit), 'rows_with_any_audit': int(audit[AUDIT_FIELDS].ne('').any(axis=1).sum()), 'metrics': {}}
    binary = ['evidence_available', 'visual_evidence_present', 'verbal_evidence_present',
              'requires_audio_or_dialogue', 'requires_temporal_reasoning', 'clearly_observable', 'observable_in_planned_input']
    for col in binary:
        validate_values(audit, col, ['yes', 'no', 'uncertain'])
        rows = audit[audit[col].ne('')]
        result['metrics'][col] = {**cluster_mean_ci(rows[col].eq('yes'), groups_for(rows), config['seed'], config['bootstrap_replicates']),
                                  'uncertain': int(rows[col].eq('uncertain').sum()), 'blank': int(audit[col].eq('').sum()),
                                  'denominator': 'all answered, including uncertain'}
    validate_values(audit, 'evidence_modality', ['visual_only', 'verbal_only', 'both', 'ambiguous_unavailable'])
    validate_values(audit, 'ambiguity_level', ['low', 'medium', 'high', 'unrecoverable'])
    modalities = audit.loc[audit.evidence_modality.ne(''), 'evidence_modality'].value_counts()
    result['modality_counts'] = modalities.to_dict()
    result['modality_percent_of_answered'] = (100 * modalities / modalities.sum()).to_dict()
    cues = audit.cue_types.str.split(';').explode().str.strip()
    # Count each cue at most once per audited example.
    cue_counts = cues[cues.ne('')].reset_index().drop_duplicates().cue_types.value_counts()
    result['cue_counts'] = cue_counts.to_dict()
    result['status'] = 'pending_human_audit' if result['rows_with_any_audit'] == 0 else 'partial_or_complete_human_audit'
    save_json(result, Path(config['output_dir']) / 'tables' / 'audit_statistics.json')
    bar_plot(result['modality_percent_of_answered'], config['output_dir'], 'A1_evidence_modality', 'Percent of answered audit rows')
    bar_plot(result['cue_counts'], config['output_dir'], 'A2_cue_types', 'Examples with cue (multiple allowed)')
    LOG.info('Audit: %d/%d rows have human entries.', result['rows_with_any_audit'], len(audit))
    return result


def merge_humans(config, sample):
    dest = Path(config['data_dir'])
    expected = pd.read_csv(dest / 'human_sample_ids.csv', dtype=str).sample_id.tolist()
    merged, completion = [], []
    labels = config['dataset']['label_values']
    for name in config['annotation']['annotators']:
        path = dest / 'blinded' / name / f'human_annotations_{name}.csv'
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if set(frame.columns) != set(BLIND_FIELDS + RESPONSE_FIELDS):
            raise ValueError(f'{name}: unexpected columns in blinded response file.')
        if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(expected):
            raise ValueError(f'{name}: IDs must match the human sample exactly once.')
        validate_values(frame, 'predicted_judgment', labels)
        validate_values(frame, 'response_status', ['completed', 'unavailable_media', 'language_difficulty', 'cannot_judge'])
        validate_values(frame, 'confidence', ['1', '2', '3', '4', '5'])
        validate_values(frame, 'evidence_modality', ['visual_only', 'verbal_only', 'both', 'ambiguous_unavailable'])
        if (frame.response_status.eq('completed') & (frame.predicted_judgment.eq('') | frame.human_evidence.eq(''))).any():
            raise ValueError(f'{name}: completed responses require judgment and observational evidence.')
        if (frame.response_status.ne('completed') & (frame.predicted_judgment.ne('') | frame.human_evidence.ne(''))).any():
            raise ValueError(f'{name}: responses require status=completed; skipped items must have blank answers.')
        if frame.input_mode.ne(config['annotation']['input_mode']).any():
            raise ValueError('Input mode changed in annotation file.')
        check = frame[['sample_id', 'norm_text']].merge(sample[['sample_id', 'norm_text']], on='sample_id', suffixes=('_human', '_gold'))
        if check.norm_text_human.ne(check.norm_text_gold).any():
            raise ValueError('Norm prompt changed in annotation file.')
        completion.append({'annotator': name, 'assigned': len(frame), **frame.response_status.replace('', 'pending').value_counts().to_dict()})
        frame = frame[['sample_id'] + RESPONSE_FIELDS].copy()
        frame['annotator'] = name
        merged.append(frame.merge(sample, on='sample_id', how='left', validate='one_to_one'))
    out = pd.concat(merged, ignore_index=True)
    tables = Path(config['output_dir']) / 'tables'
    out.to_csv(tables / 'merged_human_annotations.csv', index=False)
    pd.DataFrame(completion).fillna(0).to_csv(tables / 'human_completion.csv', index=False)
    return out


def analyze_humans(config, merged):
    completed = merged[merged.response_status.eq('completed')].copy()
    completed['correct'] = completed.predicted_judgment.eq(completed.gold_norm_judgment).astype(float)
    results = {'completed': len(completed), 'status': 'pending_human_simulation' if completed.empty else 'exploratory', 'accuracy': {}, 'accuracy_by_evidence_modality': {}}
    for name, rows in completed.groupby('annotator'):
        results['accuracy'][name] = cluster_mean_ci(rows.correct, groups_for(rows), config['seed'], config['bootstrap_replicates'])
        results['accuracy_by_evidence_modality'][name] = {
            modality: cluster_mean_ci(part.correct, groups_for(part), config['seed'], config['bootstrap_replicates'])
            for modality, part in rows[rows.evidence_modality.ne('')].groupby('evidence_modality')}
    if len(config['annotation']['annotators']) == 2:
        wide = completed.pivot(index='sample_id', columns='annotator', values='predicted_judgment').dropna()
        if wide.shape[1] == 2 and len(wide):
            a, b = wide.iloc[:, 0], wide.iloc[:, 1]
            kappa = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else None
            paired = merged.drop_duplicates('sample_id').set_index('sample_id').loc[wide.index].reset_index()
            results['agreement'] = cluster_mean_ci(a.eq(b), groups_for(paired), config['seed'], config['bootstrap_replicates'])
            results['agreement']['cohen_kappa'] = kappa
            results['agreement']['note'] = 'Kappa undefined if the pooled labels contain only one class.'
    save_json(results, Path(config['output_dir']) / 'tables' / 'human_statistics.json')
    if results['accuracy']:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (name, v) in enumerate(results['accuracy'].items()):
            err = None if v['ci_low'] is None else [[v['mean'] - v['ci_low']], [v['ci_high'] - v['mean']]]
            ax.bar(i, v['mean'], yerr=err, capsize=4, label=f'{name}, n={v["n"]}')
        ax.set_xticks(range(len(results['accuracy'])), list(results['accuracy']))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Judgment accuracy (95% bootstrap CI)')
        ax.legend()
        figure_save(fig, config['output_dir'], 'A3_human_accuracy')
    LOG.info('Human analysis: %d completed responses. No blank responses scored.', len(completed))
    return results
