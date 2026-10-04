"""Synthetic software tests; never source data or actual human responses."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.idea2.data import digest, save_json
from analysis.social_genome.access import metadata, source_order, study_sources, trim_vtt
from analysis.social_genome.study import INPUTS, RESPONSES, response_fields, prepare, refresh_pages, load_study, import_responses, apply_corrections, analyze_humans, choose_question
from analysis.social_genome.language import nearest_other, language_rows, cross_human, human_language
from analysis.social_genome.results import wilson, audit_tables


@pytest.fixture
def cfg(tmp_path):
    config = json.loads(Path('analysis/social_genome/config_legacy_silent.json').read_text())
    config.pop('selection_lock', None)
    config.update(raw_dir=str(tmp_path/'raw'), data_dir=str(tmp_path/'study'), output_dir=str(tmp_path/'out'), sample_size=4, bootstrap_replicates=30)
    root = Path(config['raw_dir']);root.mkdir()
    (Path(config['output_dir'])/'tables').mkdir(parents=True)
    rows = [{'qid': f'test_source{i}_q{j}', 'vid_name': f'test_source{i}', 'ts': '0.00-60.00',
             'q': f'SYNTHETIC TEST question {i} {j}', **{f'a{k}': f'test choice {k}' for k in range(4)}} for i in range(4) for j in range(2)]
    save_json(rows, root/'qa_test.json')
    save_json([f'test_source{i}' for i in range(4)], root/'videos.json')
    save_json({f'test_source{i}': i*10 for i in range(4)}, root/'trims.json')
    for i in range(4):
        p=root/'media'/f'test_source{i}';p.mkdir(parents=True)
        (p/'clip.mp4').write_bytes(b'SYNTHETIC TEST FILE; NOT REAL MEDIA')
        (p/'transcript.txt').write_text(f'[0.00–1.00s] SYNTHETIC TEST transcript {i}')
        save_json({'clip_sha256': digest(p/'clip.mp4'), 'transcript_sha256': digest(p/'transcript.txt'),
                   'caption_source': 'SYNTHETIC_TEST_ONLY'}, p/'complete.json')
    return config


def test_actual_schema_and_no_gold(cfg):
    frame, trims=metadata(cfg)
    assert len(frame)==8 and frame.vid_name.nunique()==4
    result=json.loads((Path(cfg['output_dir'])/'tables/dataset_inspection.json').read_text())
    assert not result['gold_answers_available'] and not result['reference_evidence_available']
    assert source_order(frame,42)==source_order(frame.iloc[::-1],42)
    assert choose_question(frame,42)==choose_question(frame.iloc[::-1],42)


def test_caption_clipping_and_manual_repetition():
    text='WEBVTT\n\n00:00:09.000 --> 00:00:11.000\nHello\n\n00:00:11.000 --> 00:00:13.000\nHello\n\n00:00:70.000 --> 00:00:71.000\nOUTSIDE\n'
    segments=trim_vtt(text,10)
    assert [s['text'] for s in segments]==['Hello','Hello']
    assert segments[0]['start']==0 and segments[0]['end']==1
    assert all(0<=s['start']<s['end']<=60 for s in segments)
    assert len(trim_vtt(text,10,rolling=True))==1


def test_locked_selection(cfg):
    prepare(cfg)
    lock = Path(cfg['data_dir'])/'sampled_ids.csv'
    cfg['selection_lock'] = str(lock)
    frame, _ = metadata(cfg)
    assert study_sources(cfg, frame) == pd.read_csv(lock).source_video_id.tolist()
    data = pd.read_csv(lock)
    data.loc[0, 'qid'] = 'MISSING_SYNTHETIC_QUESTION'
    data.to_csv(lock, index=False)
    with pytest.raises(ValueError, match='pinned public metadata'):
        study_sources(cfg, frame)


def test_blind_packets_blank_and_frozen(cfg):
    prepare(cfg)
    sample, humans=load_study(cfg)
    assert sample.sample_id.is_unique and sample.source_video_id.is_unique
    assert humans[RESPONSES].eq('').all().all()
    result=analyze_humans(cfg,sample,humans)
    stats=json.loads((Path(cfg['output_dir'])/'tables/human_statistics.json').read_text())
    assert stats['status']=='pending_humans' and stats['gold_accuracy'] is None
    path=Path(cfg['data_dir'])/'blinded/annotator1/human_annotations_annotator1.csv'
    packet=pd.read_csv(path,keep_default_na=False)
    assert set(packet)==set(INPUTS+RESPONSES)
    assert not any('gold' in c or 'reference' in c for c in packet)
    packet.loc[0,'question']='TAMPERED'
    packet.to_csv(path,index=False)
    with pytest.raises(ValueError,match='changed'):
        load_study(cfg)


def test_no_overwrite_and_media_integrity(cfg):
    prepare(cfg)
    with pytest.raises(ValueError,match='overwrite'):
        prepare(cfg)
    media=next((Path(cfg['data_dir'])/'blinded/annotator1/media').glob('*.mp4'))
    media.write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='video differs'):
        load_study(cfg)


def test_negative_excludes_same_source_and_self():
    x=np.array([[1.,0.],[1.,0.],[0.,1.]])
    assert nearest_other(x,['a','a','b'])==[2,2,0]
    with pytest.raises(ValueError,match='distinct'):
        nearest_other(x,['a','a','a'])


def test_real_response_gate(cfg):
    prepare(cfg)
    p=Path(cfg['data_dir'])/'blinded/annotator1/human_annotations_annotator1.csv'
    frame=pd.read_csv(p,keep_default_na=False)
    frame.loc[0,'response_status']='completed'
    frame.to_csv(p,index=False)
    with pytest.raises(ValueError,match='require'):
        load_study(cfg)


def test_cross_human_only_synthetic_fixture(cfg):
    prepare(cfg)
    for name in cfg['annotators']:
        p=Path(cfg['data_dir'])/'blinded'/name/f'human_annotations_{name}.csv'
        f=pd.read_csv(p,keep_default_na=False)
        f['response_status']='completed'; f['selected_answer']='A'
        f['human_evidence']='SYNTHETIC UNIT TEST RESPONSE, NOT HUMAN DATA'
        f.to_csv(p,index=False)
    sample,humans=load_study(cfg)
    analyze_humans(cfg,sample,humans)
    rows=language_rows(sample,humans)
    assert rows.key.is_unique
    assert len(rows)==len(sample)*8
    basis={sid: np.eye(4)[i] for i,sid in enumerate(sample.sample_id)}
    lookup={r.key:basis[r.sample_id] for r in rows.itertuples()}
    cross_human(cfg,sample,humans,lookup,[1,2,3,0])
    summary=json.loads((Path(cfg['output_dir'])/'tables/cross_human_statistics.json').read_text())
    assert summary['directions']['annotator1->annotator2']['recall_at_1']['mean']==1
    assert summary['directions']['annotator1->annotator2']['paired_difference']['mean']==1
    identity = np.eye(4)
    human_language(cfg,sample,humans,lookup,[1,2,3,0],identity,identity)
    language=json.loads((Path(cfg['output_dir'])/'tables/human_language_statistics.json').read_text())
    assert language['annotators']['annotator1']['recall_at_1']==1
    assert language['annotators']['annotator1']['own_minus_alternative']['mean']==1
    association=pd.read_csv(Path(cfg['output_dir'])/'tables/human_transcript_association.csv')
    assert association.sample_id.ne(association.alternative_id).all()


def test_single_annotator_audio_packet(cfg):
    cfg.update(input_mode='video_audio_transcript', annotators=['annotator1'])
    for folder in (Path(cfg['raw_dir'])/'media').iterdir():
        (folder/'clip_av.mp4').write_bytes(b'SYNTHETIC AUDIO VIDEO FIXTURE')
        info=json.loads((folder/'complete.json').read_text())
        info['clip_sha256']=digest(folder/'clip_av.mp4')
        save_json(info,folder/'audio_complete.json')
    prepare(cfg)
    sample, humans=load_study(cfg)
    assert len(humans)==4 and humans[response_fields(cfg)].eq('').all().all()
    page=(Path(cfg['data_dir'])/'blinded/annotator1/index.html').read_text()
    assert 'id="audio_used"' in page and 'controls muted' not in page
    assert '<option>audio_only</option>' in page and '__AUDIO_' not in page
    assert not (Path(cfg['data_dir'])/'blinded/annotator2').exists()
    analyze_humans(cfg,sample,humans)
    cross_human(cfg,sample,humans,{},[1,2,3,0])
    stats=json.loads((Path(cfg['output_dir'])/'tables/human_statistics.json').read_text())
    cross=json.loads((Path(cfg['output_dir'])/'tables/cross_human_statistics.json').read_text())
    assert stats['agreement_status']=='not_applicable_single_annotator'
    assert cross['status']=='not_applicable_single_annotator' and cross['directions']=={}


def test_refresh_pages_preserves_saved_work(cfg):
    prepare(cfg)
    root=Path(cfg['data_dir'])
    csv=root/'blinded/annotator1/human_annotations_annotator1.csv'
    rows=pd.read_csv(csv,keep_default_na=False)
    rows.loc[0,'notes']='SYNTHETIC SAVED USER NOTE'
    rows.to_csv(csv,index=False)
    protected=[p for p in root.rglob('*') if p.is_file() and p.suffix!='.html']
    before={str(p):digest(p) for p in protected}
    refresh_pages(cfg)
    assert before=={str(p):digest(p) for p in protected}
    assert (root/'blinded/annotator1/index.html.before_playback_fix').exists()
    load_study(cfg)


def test_import_and_explicit_corrections_preserve_export(cfg, tmp_path):
    prepare(cfg)
    canonical=Path(cfg['data_dir'])/'blinded/annotator1/human_annotations_annotator1.csv'
    export=tmp_path/'export.csv'
    frame=pd.read_csv(canonical,dtype=str,keep_default_na=False)
    frame.loc[0,'notes']='SYNTHETIC HUMAN EXPORT'
    frame.to_csv(export,index=False)
    raw_hash=digest(export)
    import_responses(cfg,'annotator1',export)
    assert digest(canonical)==raw_hash
    correction=tmp_path/'corrections.json'
    save_json([{'annotator':'annotator1','sample_id':frame.loc[0,'sample_id'],'field':'notes',
                'old':'SYNTHETIC HUMAN EXPORT','new':'SYNTHETIC CONFIRMED CORRECTION','reason':'synthetic test confirmation'}],correction)
    apply_corrections(cfg,correction)
    assert digest(export)==raw_hash
    assert pd.read_csv(canonical).loc[0,'notes']=='SYNTHETIC CONFIRMED CORRECTION'
    old=digest(canonical)
    frame.loc[0,'question']='TAMPERED INPUT'
    frame.to_csv(export,index=False)
    with pytest.raises(ValueError,match='changed'):
        import_responses(cfg,'annotator1',export)
    assert digest(canonical)==old


def test_boundary_intervals_and_audio_conflicts(cfg):
    low, high=wilson(24,24)
    assert .86<low<.87 and high==1
    prepare(cfg)
    sample,humans=load_study(cfg)
    humans['response_status']='completed'
    humans['audio_used']='no'
    humans['evidence_modality']='visual_audio'
    audit_tables(cfg,humans)
    flags=pd.read_csv(Path(cfg['output_dir'])/'tables/annotation_quality_flags.csv')
    assert len(flags)==len(humans)
