"""Schema inspection and explicitly configured normalization. No inferred columns."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
CONCEPTS = ['sample_id', 'source_video_id', 'norm_text', 'gold_norm_judgment',
            'reference_verbal_evidence', 'reference_nonverbal_evidence', 'reference_evidence',
            'video_path', 'transcript', 'category', 'language', 'split']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read_table(path, records_key=None):
    path = Path(path)
    if path.suffix == '.csv':
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix in {'.parquet', '.pq'}:
        df = pd.read_parquet(path)
    elif path.suffix in {'.jsonl', '.ndjson'}:
        df = pd.DataFrame([json.loads(x) for x in path.read_text().splitlines() if x.strip()])
    elif path.suffix == '.json':
        value = json.loads(path.read_text())
        if records_key:
            value = value[records_key]
        if not isinstance(value, list) or any(not isinstance(r, dict) for r in value):
            raise ValueError('JSON must be a list of records; specify records_key or write a reviewed adapter.')
        df = pd.DataFrame(value)
    else:
        raise ValueError('Supported inputs: CSV, Parquet, JSON list, JSONL.')
    if df.empty:
        raise ValueError('Dataset has no records.')
    return df


def inspect(path, out, records_key=None):
    df = read_table(path, records_key)
    rows = []
    for name in df:
        s = df[name]
        examples = [str(x)[:350] for x in s.dropna().head(3)]
        rows.append({'column': name, 'dtype': str(s.dtype), 'rows': len(s),
                     'missing': int((s.isna() | s.astype(str).str.strip().eq('')).sum()),
                     'unique_as_text': int(s.astype(str).nunique()), 'examples': examples})
    out = Path(out)
    save_json({'path': str(path), 'sha256': digest(path), 'rows': len(df),
               'columns': rows, 'warning': 'Examples include unblinded source data.'}, out / 'schema.json')
    pd.DataFrame(rows).to_csv(out / 'schema.csv', index=False)
    LOG.info('Inspected %d rows and %d columns; no semantic mapping inferred.', len(df), len(df.columns))
    return df


def normalize(config):
    cfg = config['dataset']
    if cfg['name'] != 'VideoNorms':
        raise ValueError('This workflow is restricted to VideoNorms; do not use EgoNormia.')
    if not cfg.get('path') or not cfg.get('schema_reviewed') or not cfg.get('version'):
        raise ValueError('Real VideoNorms path, version and reviewed schema mapping are required. See docs/data_access_status.md.')
    raw = read_table(cfg['path'], cfg.get('records_key'))
    mapping = cfg['columns']
    for required in ['norm_text', 'gold_norm_judgment']:
        if not mapping.get(required):
            raise ValueError(f'Missing reviewed mapping for {required}')
    if not any(mapping.get(x) for x in CONCEPTS if 'evidence' in x):
        raise ValueError('Map at least one actual evidence field.')
    df = pd.DataFrame(index=raw.index)
    for concept in CONCEPTS:
        col = mapping.get(concept)
        if col is None:
            df[concept] = ''
        else:
            if col not in raw:
                raise ValueError(f'Mapped column absent: {concept} -> {col}')
            if raw[col].map(lambda x: isinstance(x, (dict, list, tuple, np.ndarray))).any():
                raise ValueError(f'{col} contains nested values. Inspect and write a release-specific adapter first.')
            df[concept] = raw[col].fillna('').astype(str).str.strip()
    if not mapping.get('sample_id'):
        # Stable for this exact immutable export, not a claimed native ID.
        prefix = digest(cfg['path'])[:12]
        df['sample_id'] = [f'{prefix}:row:{i}' for i in range(len(df))]
    if df.sample_id.eq('').any() or df.sample_id.duplicated().any():
        raise ValueError('Record IDs must be nonempty and unique; distinguish task/annotation records explicitly.')
    labels = cfg.get('label_values', [])
    observed = set(df.gold_norm_judgment) - {''}
    if not labels or not observed.issubset(set(labels)):
        raise ValueError(f'Actual labels {sorted(observed)} must be explicitly enumerated in label_values.')
    separate = [x for x in ['reference_verbal_evidence', 'reference_nonverbal_evidence'] if mapping.get(x)]
    if not mapping.get('reference_evidence'):
        df['reference_evidence'] = df[separate].apply(lambda r: '\n'.join(x for x in r if x), axis=1)
    # Presence is not a human observability finding.
    df['annotation_field_presence'] = df.apply(lambda r: (
        'both_fields' if r.reference_verbal_evidence and r.reference_nonverbal_evidence else
        'verbal_field' if r.reference_verbal_evidence else
        'nonverbal_field' if r.reference_nonverbal_evidence else
        'combined_only' if r.reference_evidence else 'missing'), axis=1)
    problems = []
    root = Path(cfg.get('media_root', '.'))
    for row in df.to_dict('records'):
        for col in ['source_video_id', 'norm_text', 'gold_norm_judgment', 'reference_evidence', 'video_path']:
            if not row[col]:
                problems.append({'sample_id': row['sample_id'], 'issue': f'missing_{col}'})
        if row['video_path'] and not (root / row['video_path']).is_file():
            problems.append({'sample_id': row['sample_id'], 'issue': 'video_not_local_or_missing'})
    out = Path(config['output_dir']) / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(problems, columns=['sample_id', 'issue']).to_csv(out / 'data_issues.csv', index=False)
    LOG.info('Loaded %d real records; %d missing-field/media issues logged.', len(df), len(problems))
    pd.DataFrame([{'concept': c, 'actual_column': mapping.get(c),
                   'derivation': 'export hash + row index' if c == 'sample_id' and not mapping.get(c) else
                   'join mapped evidence fields' if c == 'reference_evidence' and not mapping.get(c) else ''}
                  for c in CONCEPTS]).to_csv(out / 'schema_mapping.csv', index=False)
    save_json({'dataset_version': cfg['version'], 'sha256': digest(cfg['path']),
               'rows': len(df), 'unique_nonempty_source_ids': int(df.loc[df.source_video_id.ne(''), 'source_video_id'].nunique()),
               'labels': df.gold_norm_judgment.value_counts().to_dict(),
               'splits': df.split.value_counts().to_dict(),
               'field_presence_not_observability': df.annotation_field_presence.value_counts().to_dict(),
               'category_counts': df.category.value_counts().to_dict(),
               'language_counts': df.language.value_counts().to_dict(),
               'exact_duplicate_records_excluding_id': int(df.drop(columns='sample_id').duplicated().sum())}, out / 'dataset_summary.json')
    return df


def sample_records(df, n, seed, one_per_source=True):
    """Label quotas, with greedy coverage of real category/presence values."""
    if n < 1:
        raise ValueError('Sample size must be positive.')
    pool = df.sort_values('sample_id').sample(frac=1, random_state=seed).copy()
    if one_per_source and pool.source_video_id.eq('').any():
        raise ValueError('one_per_source requires complete mapped source IDs; do not invent them.')
    labels = sorted(pool.gold_norm_judgment.unique())
    chosen, used, coverage = [], set(), {}
    while len(chosen) < n:
        progressed = False
        for label in labels:
            p = pool[pool.gold_norm_judgment.eq(label) & ~pool.sample_id.isin(chosen)]
            if one_per_source:
                p = p[~p.source_video_id.isin(used)]
            if p.empty:
                continue
            keys = list(zip(p.category, p.annotation_field_presence))
            pos = min(range(len(p)), key=lambda i: coverage.get(keys[i], 0))
            row = p.iloc[pos]
            chosen.append(row.sample_id)
            used.add(row.source_video_id)
            coverage[keys[pos]] = coverage.get(keys[pos], 0) + 1
            progressed = True
            if len(chosen) == n:
                break
        if not progressed:
            raise ValueError(f'Requested {n} records; only {len(chosen)} eligible independent choices. Reduce size explicitly.')
    result = df.set_index('sample_id').loc[chosen].reset_index()
    if result.sample_id.duplicated().any():
        raise ValueError('Duplicate sampled IDs.')
    return result


def partition_groups(df, seed, fractions=(0.6, 0.2, 0.2)):
    if df.source_video_id.eq('').any():
        raise ValueError('Source IDs are incomplete.')
    ids = np.array(sorted(df.source_video_id.unique()))
    if len(ids) < 3:
        raise ValueError('At least three unique sources required.')
    np.random.default_rng(seed).shuffle(ids)
    n1 = max(1, min(len(ids) - 2, int(len(ids) * fractions[0])))
    n2 = max(1, min(len(ids) - n1 - 1, int(len(ids) * fractions[1])))
    return dict(zip(['sft_candidate', 'rl_candidate', 'held_out_candidate'],
                    [set(ids[:n1]), set(ids[n1:n1+n2]), set(ids[n1+n2:])]))


def partition_feasibility(df, config):
    out = Path(config['output_dir']) / 'tables'
    counts = df[df.source_video_id.ne('')].groupby('source_video_id').size().rename('examples').reset_index()
    counts.to_csv(out / 'examples_per_video.csv', index=False)
    valid = not df.source_video_id.eq('').any() and config['dataset'].get('source_id_level') == 'parent_video' and len(counts) >= 3
    rows = []
    if valid:
        for name, ids in partition_groups(df, config['seed']).items():
            part = df[df.source_video_id.isin(ids)]
            rows.append({'partition': name, 'videos': len(ids), 'examples': len(part),
                         'label_counts': json.dumps(part.gold_norm_judgment.value_counts().to_dict()),
                         'status': 'size simulation only; coordinate with Idea 1; no assignments saved'})
    else:
        rows.append({'partition': '', 'videos': len(counts), 'examples': len(df),
                     'status': 'unverified: need complete parent-video IDs and at least three sources'})
    pd.DataFrame(rows).to_csv(Path(config['output_dir']) / 'video_partition_feasibility.csv', index=False)
    if df.split.ne('').any():
        overlap = df[df.source_video_id.ne('') & df.split.ne('')].groupby('source_video_id').split.nunique()
        overlap[overlap > 1].rename('existing_split_count').to_csv(out / 'existing_split_source_overlap.csv')
