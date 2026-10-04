"""Read-only numeric verification. Standard Python only; no models or network."""
from pathlib import Path
from collections import Counter
import json,hashlib,statistics,struct
root=Path(__file__).resolve().parents[1]
def read(name):return json.loads((root/name).read_text())
d=read('consolidated_results.json');s=d['summary'];passed=[]
def check(name,test):
 if not test:raise AssertionError(name)
 passed.append(name)
for r in d['input_manifest']:
 p=root/r['file'];check('Input hash '+r['file'],p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
official=read('data/official_annotations.json');ids=set(official)
v1=read('data/screening_v1.json');v2=read('data/region_screening_v2.json');coverage=read('data/coverage_results.json');pilots=read('data/pilot_manifest_original.json');download=read('data/download_records.json')
check('Official population',len(ids)==s['population']==1853)
for name,rows in [('v1',v1['records']),('v2',v2['records']),('downloads',download)]:
 check(name+' IDs',len(rows)==len(ids) and {r['id'] for r in rows}==ids)
check('Five automatic frames per item',all(len(r['frames'])==5 for r in v1['records']) and 5*len(ids)==s['automatic_frames'])
check('No screening failures',all(r['status']=='screened' for r in v1['records']))
check('No download failures',all(r['status']=='downloaded' for r in download))
check('v1 triage',dict(Counter(r['screening_label'] for r in v1['records']))==s['v1_triage'])
check('v2 labels',dict(Counter(r['v2_assessment'] for r in v2['records']))==s['v2_labels'])
check('117 unique audit examples',len(v2['audit'])==len({r['id'] for r in v2['audit']})==117)
all_by_id={r['id']:r for r in v2['records']}
check('Audit labels agree',all(all_by_id[r['id']]['v2_assessment']==r['assessment'] for r in v2['audit']))
check('Original and additional audit group sizes',Counter(r['group'] for r in v2['audit'])=={'original69':69,'additional48':48})
pool={r['id'] for r in v2['records'] if r['v2_assessment'] in ['promising_insertion_region','needs_editing_test']}
check('Pool membership',len(pool)==100 and pool=={r['id'] for r in coverage['records']})
def norms(id):r=official[id];return set(r['taxonomy'].get(str(r['correct']),[]))
def source(id):return id.split('_')[0]
check('1075 full source IDs',len({source(id) for id in ids})==1075)
check('92 pool source IDs',len({source(id) for id in pool})==92)
source_counts=Counter(source(id) for id in pool);repeated=[n for n in source_counts.values() if n>1]
check('Repeated source counts',len(repeated)==7 and sum(repeated)==15)
for row in d['norm_coverage']:
 def predicate(id):return not norms(id) if row['norm']=='No correct-option norm labels' else row['norm'] in norms(id)
 check('Norm '+row['norm'],sum(predicate(id) for id in ids)==row['full'] and sum(predicate(id) for id in pool)==row['pool'])
 for label,key in [('promising_insertion_region','promising'),('needs_editing_test','needs_test')]:check(row['norm']+' '+label,sum(predicate(r['id']) for r in coverage['records'] if r['assessment']==label)==row[key])
primary={'Cooperation','Communication/Legibility','Coordination/Proactivity','Privacy'}
check('Primary-norm union',sum(bool(norms(id)&primary) for id in pool)==81)
check('Pilot membership and source IDs',len(pilots['records'])==5 and len({r['source_video_id'] for r in pilots['records']})==5 and all(r['id'] in pool for r in pilots['records']))
region=[r for r in d['pilot_region_measurements']];check('23 visible region measurements',len(region)==23)
for r in region:
 x0,y0,x1,y1=r['review_rectangle_xyxy'];area=100*(x1-x0)*(y1-y0)/(r['width']*r['height'])
 check('Area '+str(r['item'])+'/'+str(r['frame']),abs(area-r['region_area_percent'])<1e-9 and 0<=x0<x1<=r['width'] and 0<=y0<y1<=r['height'])
stats=d['overall_pilot_region_summary'];check('Region median',abs(statistics.median(r['region_area_percent'] for r in region)-stats['median_area_percent'])<1e-9)
audit=read('data/pilot_audit.json')['frames'];check('Output accounting',len(audit)==90 and sum(r['editable'] for r in audit)==84 and sum(not r['editable'] for r in audit)==6)
check('Retained pixel audit',sum(r['outside_mask_changed_pixels'] for r in audit)==0)
check('Retained generation accounting',len(read('data/pilot_generation_records.json'))==18)
for p in (root/'evidence').glob('*.png'):
 b=p.read_bytes();w,h=struct.unpack('>II',b[16:24]);item=int(p.name.split('_')[1]);check('Evidence dimensions '+p.name,(w,h)==(640,480 if item==605 else 360))
print(json.dumps({'passed':True,'checks':len(passed),'items':1853,'potential_candidates':100,'reviewed_examples':117,'variant_frames':90,'scope':'Numeric record reconciliation; retained pixel audit only. No human validation or Qwen inference.'},indent=2))
