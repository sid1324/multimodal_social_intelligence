"""Descriptive single-annotator audit summaries; no gold or reliability claims."""
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.idea2.data import save_json
from analysis.idea2.statistics import figure_save

CHANNELS = {
    'visual_only': {'visual'}, 'audio_only': {'audio'}, 'transcript_only': {'transcript'},
    'visual_audio': {'visual', 'audio'}, 'visual_transcript': {'visual', 'transcript'},
    'audio_transcript': {'audio', 'transcript'}, 'all_three': {'visual', 'audio', 'transcript'},
    'both': {'visual', 'transcript'}, 'unclear': set(), '': set(),
}


def modality_label(value):
    return 'Visual + audio + transcript' if value == 'all_three' else value.replace('_only', ' only').replace('_', ' + ').capitalize()


def wilson(successes, n):
    """95% binomial score interval; descriptive item uncertainty, not rater reliability."""
    if not n:
        return (None, None)
    z = NormalDist().inv_cdf(.975)
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0., center-half), min(1., center+half)


def rate_record(annotator, metric, count, n, missing=0):
    low, high = wilson(count, n)
    return {'annotator': annotator, 'metric': metric, 'count': int(count), 'n': int(n),
            'percent': 100*count/n if n else None, 'wilson_low_percent': 100*low if n else None,
            'wilson_high_percent': 100*high if n else None, 'missing': int(missing)}


def audit_tables(config, humans):
    out = Path(config['output_dir'])
    tables = out/'tables'
    done = humans[humans.response_status.eq('completed')].copy()
    flags, rates, cue_rows, confidence, lengths = [], [], [], [], []
    for name, group in done.groupby('annotator'):
        for row in group.itertuples():
            channels = CHANNELS[row.evidence_modality]
            audio = getattr(row, 'audio_used', '')
            if row.evidence_modality not in ['', 'unclear'] and audio in ['yes', 'no'] and (('audio' in channels) != (audio == 'yes')):
                flags.append({'annotator': name, 'sample_id': row.sample_id, 'kind': 'audio_fields_conflict',
                              'detail': f'evidence_modality={row.evidence_modality}; audio_used={audio}',
                              'action': 'Preserve both fields; request human clarification. Do not silently recode.'})
            lengths.append({'annotator': name, 'sample_id': row.sample_id,
                            'evidence_words': len(row.human_evidence.split()), 'notes_present': bool(row.notes.strip())})
        for field in ['clearly_observable', 'requires_temporal_reasoning', 'audio_used', 'language_understood']:
            if field not in group:
                continue
            valid = group[group[field].ne('')]
            rec = rate_record(name, field, int(valid[field].eq('yes').sum()), len(valid), len(group)-len(valid))
            rec['uncertain'] = int(valid[field].eq('uncertain').sum())
            rates.append(rec)
        valid_modality = group[~group.evidence_modality.isin(['', 'unclear'])]
        for channel in ['visual', 'audio', 'transcript']:
            count = sum(channel in CHANNELS[m] for m in valid_modality.evidence_modality)
            rates.append(rate_record(name, f'{channel}_included_in_modality', count, len(valid_modality), len(group)-len(valid_modality)))
        for modality, count in group.loc[group.evidence_modality.ne(''), 'evidence_modality'].value_counts().items():
            rates.append(rate_record(name, f'modality_{modality}', count, int(group.evidence_modality.ne('').sum())))
        cue_sets = group.cue_types.map(lambda v: {x.strip() for x in v.split(';') if x.strip()})
        denominator = int(cue_sets.map(bool).sum())
        for cue in sorted(set().union(*cue_sets)):
            cue_rows.append(rate_record(name, cue, sum(cue in cues for cues in cue_sets), denominator, len(group)-denominator))
        numeric = pd.to_numeric(group.loc[group.confidence.ne(''), 'confidence'])
        confidence.append({'annotator': name, 'n': len(numeric), 'missing': len(group)-len(numeric),
                           'median': float(numeric.median()) if len(numeric) else None,
                           'counts': {str(k): int(v) for k, v in numeric.value_counts().sort_index().items()}})
        display = {modality_label(m): int(n)
                   for m, n in group.evidence_modality.value_counts().items() if m}
        _horizontal_counts(display, out, f'A1_modality_{name}', len(group), 'Reported evidence modality')
        cue_display = {r['metric'].replace('_', ' '): r['count'] for r in cue_rows if r['annotator'] == name}
        _horizontal_counts(cue_display, out, f'A2_cues_{name}', denominator, 'Cues reported (multiple per item allowed)')
        binary_rows = [r for r in rates if r['annotator'] == name and r['metric'] in
                       ['clearly_observable', 'requires_temporal_reasoning', 'audio_used', 'language_understood'] and r['n']]
        if binary_rows:
            with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
                fig, ax = plt.subplots(figsize=(8, 3.5))
                y = np.arange(len(binary_rows))
                values = np.array([r['percent'] for r in binary_rows])
                errors = np.array([[r['percent']-r['wilson_low_percent'] for r in binary_rows],
                                   [r['wilson_high_percent']-r['percent'] for r in binary_rows]])
                ax.errorbar(values, y, xerr=np.maximum(errors, 0), fmt='o', capsize=4)
                ax.set(yticks=y, yticklabels=[r['metric'].replace('_', ' ') for r in binary_rows],
                       xlim=(-2, 102), xlabel='Reported yes (%) · 95% Wilson interval')
                ax.invert_yaxis()
                figure_save(fig, out, f'A4_observability_{name}')
    pd.DataFrame(flags, columns=['annotator', 'sample_id', 'kind', 'detail', 'action']).to_csv(tables/'annotation_quality_flags.csv', index=False)
    pd.DataFrame(rates).to_csv(tables/'human_rates.csv', index=False)
    pd.DataFrame(cue_rows).to_csv(tables/'human_cue_rates.csv', index=False)
    pd.DataFrame(lengths).to_csv(tables/'human_evidence_lengths.csv', index=False)
    save_json({'confidence': confidence, 'quality_flag_count': len(flags), 'completed': len(done),
               'note': 'Wilson intervals describe item proportions for this annotator; availability sampling and one rater limit generalization. Audio use is not proof of audio necessity.'},
              tables/'human_descriptive_statistics.json')


def _horizontal_counts(values, out, name, n, title):
    if not values:
        return
    ordered = sorted(values.items(), key=lambda p: p[1], reverse=True)
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(8, max(3, len(ordered)*.35+1)))
        ax.barh([k for k, _ in ordered], [v for _, v in ordered])
        ax.invert_yaxis()
        ax.set(xlabel=f'Items (n={n})', title=title, xlim=(0, max(v for _, v in ordered)*1.24))
        for i, (_, v) in enumerate(ordered):
            ax.text(v+.2, i, f'{v} ({100*v/n:.1f}%)', va='center')
        figure_save(fig, out, name)
