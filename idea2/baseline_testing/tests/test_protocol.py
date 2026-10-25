import json
import subprocess
import sys
from pathlib import Path

from social_rl.tasks import ego_prompt, item_metrics, mismatch_map, parse_choice, parse_r3, parse_sensible, r3_prompt

def example(uid='a_1', group='a'):
    return dict(id=uid, group=group, dataset='ego', answer=1, sensible=[0, 1],
                behaviors=['first', 'second', 'third', 'fourth', ''],
                justifications=['why first', 'why second', 'why third', 'why fourth', ''])

def test_justification_uses_predicted_action_not_ground_truth():
    row = example()
    prompt = ego_prompt(row, 'justification', action=2)
    assert '"third" is selected' in prompt
    row['answer'] = 0
    row['sensible'] = [3]
    assert ego_prompt(row, 'justification', action=2) == prompt

def test_official_ego_prompts_format_for_all_tasks():
    row = example()
    action = ego_prompt(row, 'action')
    justification = ego_prompt(row, 'justification', action=2)
    sensible = ego_prompt(row, 'sensibility')
    assert 'choose the single most normatively relevant' in action
    assert "['first', 'second', 'third', 'fourth', '']" in action
    assert 'most similar contexts' not in action  # Upstream adds this only for RAG models.
    row['answer'] = 4
    assert ego_prompt(row, 'action') == action
    assert '"third" is selected' in justification
    assert 'choose ALL the sensible actions' in sensible

def test_option_permutation_changes_answer_location():
    row = dict(question='What happened?', options=['red','green','blue','white','black'], answer=2)
    order = [2,4,0,1,3]
    assert 'A. blue' in r3_prompt(row, order)
    assert order[parse_r3('Evidence.\nAnswer: A')] == row['answer']
    assert parse_r3('A or B') == -1
    assert parse_r3('Answer: A\nAnswer: B\nExtra trailing discussion') == -1

def test_native_parser_and_strict_validity():
    assert parse_choice('There are 3 people.\n2') == (1, True)
    assert parse_choice('I see 3 people') == (2, False)
    assert parse_choice('99', action=True) == (4, False)
    assert parse_sensible('[1, 1, 3]') == ([0,2], True)
    assert parse_sensible('[0, 2]') == ([], False)
    assert parse_sensible('[]') == ([], True)

def test_joint_not_product_and_failure_denominator():
    row = example()
    a = item_metrics(row, dict(action=1, justification=0, sensible=[0]))
    b = item_metrics(row, dict(action=0, justification=1, sensible=[0,1,2]))
    assert a['joint'] == b['joint'] == 0
    assert a['sensibility_iou'] == 0.5
    assert b['sensibility_iou'] == 2/3
    assert item_metrics(row, {})['joint'] == 0

def test_mismatch_never_same_source():
    rows = [example('a_1','a'), example('a_2','a'), example('b_1','b')]
    m = mismatch_map(rows)
    assert all(m[r['id']]['group'] != r['group'] for r in rows)
    assert m == mismatch_map(rows)

def test_official_scorer_agrees_on_valid_synthetic_predictions(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scorer = root/'vendor/EgoNormia/src/final_dataset/compute_scores_final.py'
    if not scorer.exists():
        import pytest
        pytest.skip('Official scorer checkout not present')
    data = {'a': {'synthetic': {'best': {'results':[1,1], 'correct':[1,1]},
                               'sensible': {'results':[0], 'correct':[0,1]}}},
            'b': {'synthetic': {'best': {'results':[1,0], 'correct':[1,1]},
                               'sensible': {'results':[0,1], 'correct':[0,1]}}}}
    (tmp_path/'fixture_eval.json').write_text(json.dumps(data))
    result = subprocess.run([sys.executable,str(scorer),'--file','fixture'], cwd=tmp_path,
                            text=True,capture_output=True,check=True)
    assert '50.0 & 100.0 & 50.0 & 75.0' in result.stdout

def test_actual_split_integrity_if_available():
    from social_rl.common import STORE, read_jsonl
    p = STORE/'data/manifests'
    if not (p/'splits.json').exists():
        import pytest
        pytest.skip('Data manifests not prepared')
    r3 = [read_jsonl(p/f'r3_{s}_all.jsonl') for s in ['train','val','test']]
    groups = [{r['group'] for r in rows} for rows in r3]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    mid = read_jsonl(p/'ego_midterm_panel.jsonl')
    supplied = (Path(__file__).resolve().parents[1]/'egonormia_400_sample_ids.txt').read_text().splitlines()
    assert [r['id'] for r in mid] == supplied
    final = read_jsonl(p/'ego_final_RESERVED.jsonl')
    assert not {r['group'] for r in mid}&{r['group'] for r in final}
    assert all(r['video_path'].endswith('/video_prev.mp4') for r in mid)
    assert all('description' not in r for r in mid)
