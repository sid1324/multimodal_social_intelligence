"""Synthetic software fixtures ONLY. These are not data or human annotations."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.idea2.annotations import AUDIT_FIELDS, BLIND_FIELDS, RESPONSE_FIELDS, blind_rows, prepare, write_interface
from analysis.idea2.data import CONCEPTS, inspect, normalize, partition_groups, sample_records, save_json
from analysis.idea2.statistics import analyze_audit, analyze_humans, cluster_mean_ci, merge_humans
from analysis.idea2.language import build_negatives, cache_key, language_analysis, load_cache, retrieval_rank, text_rows


@pytest.fixture
def frame():
    rows = []
    for i in range(16):
        r = {key: '' for key in CONCEPTS}
        r.update(sample_id=f'test_{i}', source_video_id=f'source_{i//2}', norm_text=f'synthetic norm {i%4}',
                 gold_norm_judgment=['LABEL_A', 'LABEL_B'][i%2], reference_evidence=f'synthetic observation {i}',
                 reference_nonverbal_evidence=f'synthetic observation {i}', category=f'category_{i%3}', language='test',
                 annotation_field_presence='nonverbal_field')
        rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture
def config(tmp_path):
    cfg = json.loads(Path('analysis/idea2/config.json').read_text())
    cfg['data_dir'] = str(tmp_path / 'data')
    cfg['output_dir'] = str(tmp_path / 'outputs')
    (Path(cfg['output_dir']) / 'tables').mkdir(parents=True)
    cfg['annotation']['input_mode'] = 'silent_frames'
    cfg['sampling'].update(size=8, human_size=4)
    cfg['dataset']['label_values'] = ['LABEL_A', 'LABEL_B']
    cfg['bootstrap_replicates'] = 30
    return cfg


def test_sampling_reproducible_unique_and_source_disjoint(frame):
    a = sample_records(frame, 8, 42)
    b = sample_records(frame.sample(frac=1, random_state=2), 8, 42)
    assert a.sample_id.tolist() == b.sample_id.tolist()
    assert a.sample_id.is_unique and a.source_video_id.is_unique
    assert a.gold_norm_judgment.value_counts().to_dict() == {'LABEL_A': 4, 'LABEL_B': 4}
    with pytest.raises(ValueError, match='only'):
        sample_records(frame, 9, 42)


def test_partition_same_video_never_crosses(frame):
    groups = partition_groups(frame, 42)
    parts = list(groups.values())
    assert not parts[0] & parts[1] and not parts[0] & parts[2] and not parts[1] & parts[2]
    assert set.union(*parts) == set(frame.source_video_id)
    assert groups == partition_groups(frame.iloc[::-1], 42)


def test_blinded_allowlist(frame, tmp_path):
    frame['dataset_generated_explanation'] = 'SECRET_GOLD'
    frame['reference_evidence'] = 'SECRET_GOLD'
    blind = blind_rows(frame, 'silent_frames')
    assert set(blind) == set(BLIND_FIELDS + RESPONSE_FIELDS)
    assert blind[RESPONSE_FIELDS].eq('').all().all()
    file = tmp_path / 'index.html'
    write_interface(blind, file, 'test', ['LABEL_A', 'LABEL_B'])
    assert 'SECRET_GOLD' not in file.read_text()
    assert 'gold_norm_judgment' not in file.read_text()


def test_missing_data_logged_and_unknown_mapping_rejected(frame, config, tmp_path):
    raw = tmp_path / 'raw.csv'
    frame.to_csv(raw, index=False)
    config['dataset'].update(path=str(raw), version='SYNTHETIC_UNIT_TEST_ONLY', schema_reviewed=True)
    config['dataset']['columns'] = {x: x for x in CONCEPTS}
    inspect(raw, tmp_path / 'inspection')
    normalized = normalize(config)
    assert len(normalized) == len(frame)
    issues = pd.read_csv(Path(config['output_dir']) / 'tables/data_issues.csv')
    assert (issues.issue == 'missing_video_path').sum() == len(frame)
    config['dataset']['columns']['norm_text'] = 'not_a_real_column'
    with pytest.raises(ValueError, match='Mapped column absent'):
        normalize(config)


def test_no_dataset_fails_honestly(config):
    with pytest.raises(ValueError, match='Real VideoNorms'):
        normalize(config)


def test_hard_negatives_exclude_self_source_duplicate_text(frame):
    rng = np.random.default_rng(42)
    norms = rng.normal(size=(len(frame), 8))
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)
    negatives = build_negatives(frame, norms)
    indexed = frame.set_index('sample_id')
    for p in negatives.itertuples():
        assert p.sample_id != p.negative_id
        assert indexed.loc[p.sample_id, 'source_video_id'] != indexed.loc[p.negative_id, 'source_video_id']
        assert indexed.loc[p.sample_id, 'gold_norm_judgment'] == indexed.loc[p.negative_id, 'gold_norm_judgment']
    frame['reference_evidence'] = 'identical text'
    assert build_negatives(frame, norms).negative_id.eq('').all()


def test_retrieval_conservative_ties():
    refs = np.array([[1., 0.], [1., 0.], [0., 1.]])
    assert retrieval_rank(np.array([1., 0.]), refs, 0) == 2
    assert retrieval_rank(np.array([0., 1.]), refs, 2) == 1


def test_embedding_cache_alignment_and_checksum(tmp_path):
    rows = pd.DataFrame({'key': ['a', 'b'], 'sample_id': ['1', '2'], 'representation': ['e', 'e'], 'text': ['foo', 'bar']})
    vectors = np.eye(2)
    rows.to_csv(tmp_path / 'rows.csv', index=False)
    np.save(tmp_path / 'embeddings.npy', vectors)
    save_json({'fingerprint': cache_key(rows, 'test', 'revision'), 'array_sha256': hashlib.sha256(vectors.tobytes()).hexdigest()}, tmp_path / 'manifest.json')
    assert np.array_equal(load_cache(tmp_path, rows, 'test', 'revision'), vectors)
    with pytest.raises(ValueError, match='stale'):
        load_cache(tmp_path, rows.iloc[::-1], 'test', 'revision')
    np.save(tmp_path / 'embeddings.npy', vectors[::-1])
    with pytest.raises(ValueError, match='checksum'):
        load_cache(tmp_path, rows, 'test', 'revision')


def test_blank_humans_are_pending_not_zero_accuracy(frame, config):
    prepare(frame, config)
    sample = pd.read_csv(Path(config['data_dir']) / 'video_norms_sample.csv', keep_default_na=False)
    merged = merge_humans(config, sample)
    result = analyze_humans(config, merged)
    assert result['completed'] == 0 and result['accuracy'] == {}
    assert analyze_audit(config, sample)['status'] == 'pending_human_audit'
    assert not (Path(config['output_dir']) / 'figures').exists()
    with pytest.raises(ValueError, match='overwrite'):
        prepare(frame, config)


def test_end_to_end_with_explicitly_synthetic_test_responses(frame, config):
    prepare(frame, config)
    dest = Path(config['data_dir'])
    sample = pd.read_csv(dest / 'video_norms_sample.csv', keep_default_na=False)
    for name in config['annotation']['annotators']:
        path = dest / 'blinded' / name / f'human_annotations_{name}.csv'
        responses = pd.read_csv(path, keep_default_na=False)
        responses['predicted_judgment'] = responses.sample_id.map(sample.set_index('sample_id').gold_norm_judgment)
        responses['human_evidence'] = 'SYNTHETIC SOFTWARE TEST INPUT, NOT HUMAN DATA'
        responses['response_status'] = 'completed'
        responses.to_csv(path, index=False)
    merged = merge_humans(config, sample)
    result = analyze_humans(config, merged)
    assert result['agreement']['mean'] == 1
    assert all(x['mean'] == 1 for x in result['accuracy'].values())
    rows = text_rows(sample, merged)
    vectors = np.random.default_rng(42).normal(size=(len(rows), 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    language_analysis(config, sample, merged, rows, vectors)
    ranks = pd.read_csv(Path(config['output_dir']) / 'tables/retrieval_ranks.csv')
    assert ranks['rank'].between(1, len(sample)).all()
    assert len(ranks) == 8
    assert (Path(config['output_dir']) / 'figures/B2_tsne.pdf').exists()


def test_cluster_bootstrap():
    result = cluster_mean_ci([0, 1, 1], ['a', 'b', 'b'], repeats=20)
    assert result['n'] == 3 and result['clusters'] == 2
    assert result['ci_low'] <= result['mean'] <= result['ci_high']
