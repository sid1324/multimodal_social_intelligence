from pathlib import Path
import csv,json,hashlib,re
r=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((r/'human_simulation/responses_scored.csv').open()))
expected={'options':([1,1,1,2,2,2,4,2,3,3,2,2,1,2,3,4,1,1,3,4],[1]*20),'description':([5,3,5,1,4,2,5,5,3,5,5,1,1,5,5,2,1,2,5,5],[5,5,4,2,5,5,5,5,5,3,4,4,3,4,4,2,4,3,4,4]),'frames':([4,3,5,1,4,2,2,4,3,5,2,2,1,5,5,4,1,5,5,5],[4,5,4,4,5,4,4,4,5,4,4,4,3,4,4,3,5,3,4,4])}
for stage,(a,c) in expected.items():
 sub=[x for x in rows if x['stage']==stage];assert [int(x['answer']) for x in sub]==a;assert [int(x['confidence']) for x in sub]==c
assert [sum(int(x['correct']) for x in rows if x['stage']==s) for s in expected]==[8,8,9]
# Independent transition counts from the scored rows, not summary CSV arithmetic.
by={(x['item'],x['stage']):x for x in rows};ids=sorted({x['item'] for x in rows});trans=[]
for a,b in [('options','description'),('description','frames')]:
 pairs=[(by[k,a],by[k,b]) for k in ids];got=[sum(x['answer']!=y['answer'] for x,y in pairs),sum(x['correct']=='0' and y['correct']=='1' for x,y in pairs),sum(x['correct']=='1' and y['correct']=='0' for x,y in pairs)];trans.append(got)
assert trans==[[16,5,5],[7,3,2]]
verified=set(json.loads((r/'data/egonormia/verified_split.json').read_text())['split']);assert set(ids)<=verified
# Recompute the original fixed-hash selection and source filtering.
media=set(json.loads((r/'data/media_sample_ids.json').read_text()));selected=[];seen=set()
for k in sorted(verified&media,key=lambda k:hashlib.sha256(('human-20261004:'+k).encode()).hexdigest()):
 source=k.split('_')[0]
 if source not in seen:selected.append(k);seen.add(source)
 if len(selected)==20:break
assert set(selected)==set(ids)
(r/'qa/human_transcription_check.json').write_text(json.dumps({'status':'PASS','choices_checked':60,'confidence_values_checked':60,'stage_correct_counts':[8,8,9],'independent_transition_counts_changed_gained_lost':trans,'all_items_verified':True,'original_hash_selection_reproduced':True,'source_prefixes':len(seen),'method':'Separate message-transcription arrays plus row-level arithmetic, verified membership, and independent hash-selection recomputation.'},indent=2)+'\n')

print('Human transcription, arithmetic and selection checks passed.')
