"""Score participant-supplied action choices; no model or inferred responses."""
from pathlib import Path
import csv
import json
import re
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'human_simulation'
session = json.loads((folder / 'chat_session.json').read_text())
assert session['status'] == 'complete'
items = [{'id': r['item']} for r in csv.DictReader((folder / 'answer_key_do_not_open_before_annotation.csv').open())][:session['planned_item_count']]
raw = json.loads((ROOT / 'data/egonormia/final_data.json').read_text())
keys = {r['item']: int(r['answer_index_1based']) for r in csv.DictReader((folder / 'answer_key_do_not_open_before_annotation.csv').open())}
stages = ['options', 'description', 'frames']
responses = session['responses']
assert len(responses) == len(items) * 3
assert len({(r['item'], r['stage']) for r in responses}) == len(items) * 3
lookup = {(r['item'], r['stage']): r for r in responses}
scored = []
for number, item in enumerate(items, 1):
    gold = raw[item['id']]['correct'] + 1
    assert gold == keys[item['id']]
    for stage in stages:
        r = lookup[item['id'], stage]
        assert r['answer'] in range(1, 6) and r['confidence'] in range(1, 6)
        scored.append(dict(example=number, **r, gold_answer=gold, correct=int(r['answer'] == gold)))

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

# Keep free-text reasons, but normalize optional metadata into a separate source JSON.
fields = ['example', 'annotator', 'item', 'stage', 'answer', 'confidence', 'cue', 'gold_answer', 'correct']
write_csv(folder / 'responses_scored.csv', [{k: r[k] for k in fields} for r in scored])
summary = []
for stage in stages:
    rows = [r for r in scored if r['stage'] == stage]
    summary.append(dict(stage=stage, participants=1, n_items=len(rows), correct=sum(r['correct'] for r in rows),
                        accuracy_percent=100 * mean(r['correct'] for r in rows),
                        mean_confidence=mean(r['confidence'] for r in rows),
                        none_choices=sum(r['answer'] == 5 for r in rows)))
transitions = []
for before, after in [('options', 'description'), ('description', 'frames')]:
    pairs = [(lookup[q['id'], before], lookup[q['id'], after], keys[q['id']]) for q in items]
    transitions.append(dict(before=before, after=after, n_items=len(items),
        changed=sum(a['answer'] != b['answer'] for a, b, g in pairs),
        wrong_to_right=sum(a['answer'] != g and b['answer'] == g for a, b, g in pairs),
        right_to_wrong=sum(a['answer'] == g and b['answer'] != g for a, b, g in pairs),
        wrong_to_wrong_changed=sum(a['answer'] != b['answer'] and a['answer'] != g and b['answer'] != g for a, b, g in pairs)))
write_csv(ROOT / 'outputs/tables/human_stage_summary.csv', summary)
write_csv(ROOT / 'outputs/tables/human_transitions.csv', transitions)
detail = []
for number, q in enumerate(items, 1):
    r = dict(example=number, item=q['id'], gold_answer=keys[q['id']])
    for stage in stages:
        a = lookup[q['id'], stage]
        r[stage + '_answer'] = a['answer']
        r[stage + '_correct'] = int(a['answer'] == keys[q['id']])
        r[stage + '_confidence'] = a['confidence']
    detail.append(r)
write_csv(ROOT / 'outputs/tables/human_item_results.csv', detail)
checks = {'status': 'PASS', 'participants': 1, 'items': len(items), 'responses': len(responses),
          'unique_item_stage_pairs': len(responses), 'gold_matches_raw_and_packet': True,
          'distinct_source_prefixes': len({q['id'].split('_')[0] for q in items}),
          'complete_three_stages': True, 'metric': 'action-only agreement with released gold',
          'independence_limit': 'Twenty repeated items from one participant in two sequential ten-item blocks, not 60 independent observations.',
          'provenance': session['provenance']}
assert checks['distinct_source_prefixes'] == len(items)
(ROOT / 'qa/human_validation.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps({'summary': summary, 'transitions': transitions, 'items': detail}, indent=2))

# Block-specific reporting: the second block followed feedback on the first.
block_rows = []
for block, lower, upper in [('first_10', 1, 10), ('second_10', 11, 20)]:
    for stage in stages:
        rows = [r for r in scored if lower <= r['example'] <= upper and r['stage'] == stage]
        block_rows.append(dict(block=block, stage=stage, n_items=len(rows), correct=sum(r['correct'] for r in rows), accuracy_percent=100*mean(r['correct'] for r in rows), mean_confidence=mean(r['confidence'] for r in rows)))
write_csv(ROOT / 'outputs/tables/human_block_summary.csv', block_rows)
# Full 5x5 action-transition matrices and exact within-participant agreement.
matrix = []
agreement = []
for before, after in [('options', 'description'), ('description', 'frames'), ('options', 'frames')]:
    pairs = [(lookup[q['id'], before]['answer'], lookup[q['id'], after]['answer']) for q in items]
    for a in range(1,6):
        for b in range(1,6):
            matrix.append(dict(before=before, after=after, from_answer=a, to_answer=b, count=sum(x==a and y==b for x,y in pairs)))
    agreement.append(dict(before=before, after=after, n_items=len(items), unchanged=sum(a==b for a,b in pairs), agreement_percent=100*sum(a==b for a,b in pairs)/len(items)))
write_csv(ROOT / 'outputs/tables/human_transition_matrix.csv', matrix)
write_csv(ROOT / 'outputs/tables/human_stage_agreement.csv', agreement)
coding = json.loads((folder / 'evidence_coding.json').read_text())
assert len(coding) == len(responses)
assert len({(r['item'],r['stage']) for r in coding}) == len(responses)
for row in coding:
    assert row['reason'] == lookup[row['item'],row['stage']]['cue']
    assert set(row['evidence_codes']) <= {'text','scene_object','spatial','gesture_gaze','temporal_interaction'}
write_csv(ROOT / 'outputs/tables/human_evidence_counts.csv', [dict(stage=stage, evidence_code=tag, count=sum(r['stage']==stage and tag in r['evidence_codes'] for r in coding), n_reasons=len(items)) for stage in stages for tag in ['text','scene_object','spatial','gesture_gaze','temporal_interaction']])
print(json.dumps({'block_summary':block_rows,'stage_agreement':agreement},indent=2))
