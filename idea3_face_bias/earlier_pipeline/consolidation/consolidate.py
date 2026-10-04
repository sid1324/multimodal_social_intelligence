"""Reconcile retained records only. No detector inference, edits or VLM calls."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,shutil,statistics
from PIL import Image

ROOT=Path('/workspace/scratch/8eb17a085ce3')
SRC=ROOT/'tmp/consolidation/extracted'
OUT=ROOT/'output/analysis/Milestone2_Consolidation'
SEQ=ROOT/'output/analysis/Full_Sequence_Face_Tests'
for folder in ['data','figures','code','evidence']:(OUT/folder).mkdir(parents=True,exist_ok=True)
def load(folder,name):return json.loads((SRC/folder/name).read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
official=load('EgoNormia_All_1853_Screening','official_annotations.json')
v1=load('EgoNormia_All_1853_Screening','screening_results.json')
v2=load('EgoNormia_Region_Screening_V2','region_screening_v2_results.json')
coverage=load('EgoNormia_Chunk2_Coverage','coverage_results.json')
pilot=load('EgoNormia_Chunk3_Pilots','pilot_manifest.json')
download=load('EgoNormia_All_1853_Screening','download_records.json')
source_meta=load('EgoNormia_All_1853_Screening','source_metadata.json')
audit=json.loads((SEQ/'audit.json').read_text())
checks=[]
def require(name,condition):
 if not condition:raise AssertionError(name)
 checks.append(name)
ids=set(official)
require('Official population has 1853 unique item IDs',len(ids)==1853)
for label,rows in [('v1',v1['records']),('v2',v2['records']),('downloads',download)]:
 require(label+' exactly matches official IDs',len(rows)==1853 and {r['id'] for r in rows}==ids)
 require(label+' IDs are unique',len({r['id'] for r in rows})==len(rows))
require('Every automatic record contains 5 frames',all(len(r['frames'])==5 for r in v1['records']))
require('Every automatic record is successfully screened',all(r['status']=='screened' for r in v1['records']))
require('Every input preview is successfully downloaded',all(r['status']=='downloaded' for r in download))
labels=Counter(r['screening_label'] for r in v1['records'])
require('v1 triage counts agree',labels=={'large_face_candidate':69,'possible_region_review':1430,'no_region_detected':354})
counts=Counter(r['v2_assessment'] for r in v2['records'])
expected={'promising_insertion_region':49,'needs_editing_test':51,'no_suitable_region_identified':17,'awaiting_visual_verification':1736}
require('v2 all-item label counts agree',counts==expected)
require('117 unique five-frame audit records',len(v2['audit'])==117 and len({r['id'] for r in v2['audit']})==117)
by_id={r['id']:r for r in v2['records']}
require('All 117 audit decisions agree with all-item labels',all(by_id[r['id']]['v2_assessment']==r['assessment'] for r in v2['audit']))
groups={g:Counter(r['assessment'] for r in v2['audit'] if r['group']==g) for g in ['original69','additional48']}
require('Review group sizes are 69 and 48',sum(groups['original69'].values())==69 and sum(groups['additional48'].values())==48)
pool_ids={r['id'] for r in v2['records'] if r['v2_assessment'] in ['promising_insertion_region','needs_editing_test']}
require('Coverage pool is exactly the 100 v2 potential candidates',len(coverage['records'])==100 and {r['id'] for r in coverage['records']}==pool_ids)
sorted_ids=sorted(ids);item_by_id={id:i+1 for i,id in enumerate(sorted_ids)}
require('Item numbering is the sorted official ID position',all(item_by_id[r['id']]==r['item'] for r in v2['records']))

def norms(id):
 r=official[id];return set(r['taxonomy'].get(str(r['correct']),[]))
def source(id):return id.split('_')[0]
for r in coverage['records']:
 require('Gold norm metadata agrees: '+str(r['item']),set(r['correct_action_norms'])==norms(r['id']))
 require('Gold action metadata agrees: '+str(r['item']),r['correct_action']==official[r['id']]['behaviors'][official[r['id']]['correct']])
norm_names=['Cooperation','Communication/Legibility','Coordination/Proactivity','Privacy','Politeness','Safety','Proxemics']
norm_table=[]
for name in norm_names+['No correct-option norm labels']:
 predicate=lambda id: (name in norms(id)) if name!='No correct-option norm labels' else not norms(id)
 pop=sum(predicate(id) for id in ids)
 promising=sum(predicate(r['id']) for r in coverage['records'] if r['assessment']=='promising_insertion_region')
 needs=sum(predicate(r['id']) for r in coverage['records'] if r['assessment']=='needs_editing_test')
 row=dict(norm=name,promising=promising,needs_test=needs,pool=promising+needs,full=pop,pool_percent=promising+needs,full_percent=100*pop/1853,pool_sources=len({source(id) for id in pool_ids if predicate(id)}))
 old=next(r for r in coverage['norm_coverage'] if r['norm']==name)
 require('Recomputed norm counts match saved coverage: '+name,(promising,needs,pop)==(old['promising_49'],old['needs_test_51'],old['full_dataset_1853']))
 norm_table.append(row)
source_counts=Counter(source(id) for id in pool_ids)
repeated={k:v for k,v in source_counts.items() if v>1}
settings=Counter(r['broad_setting'] for r in coverage['records'])
require('100 candidates span 92 source IDs',len(source_counts)==92)
require('Seven repeated sources contain 15 candidates',len(repeated)==7 and sum(repeated.values())==15)
require('11 coarse setting labels',len(settings)==11)
primary=set(norm_names[:4])
primary_union=sum(bool(norms(id)&primary) for id in pool_ids)
require('81 potential candidates carry a primary norm',primary_union==81)
full_sources=len({source(id) for id in ids})
require('Pinned official snapshot yields 1075 source IDs',full_sources==1075)

region_rows=[];pilot_table=[]
require('Five intended pilot items',pilot['summary']['selected_items']==[969,1625,605,749,1250])
require('Pilots are members of the promising pool',all(r['id'] in pool_ids and by_id[r['id']]['v2_assessment']=='promising_insertion_region' for r in pilot['records']))
require('Pilots have 5 distinct source IDs',len({r['source_video_id'] for r in pilot['records']})==5)
manifest_corrections=[]
for record in pilot['records']:
 item=record['item'];visible=0;areas=[];sides=[]
 for rr in record['review_regions']:
  f=rr['frame_1_based'];im=Image.open(SEQ/f'item_{item}'/'original'/f'frame_{f}.png');w,h=im.size
  if (rr['frame_width'],rr['frame_height'])!=(w,h):manifest_corrections.append(dict(item=item,frame=f,stored_dimensions=[rr['frame_width'],rr['frame_height']],actual_dimensions=[w,h]))
  rect=rr['review_rectangle_xyxy']
  if rect is None:continue
  visible+=1;x0,y0,x1,y1=rect
  require('Review rectangle stays in source bounds: '+str(item)+'/'+str(f),0<=x0<x1<=w and 0<=y0<y1<=h)
  area=100*(x1-x0)*(y1-y0)/(w*h);side=min(x1-x0,y1-y0);areas.append(area);sides.append(side)
  region_rows.append(dict(item=item,frame=f,width=w,height=h,review_rectangle_xyxy=rect,region_area_percent=area,min_region_side_px=side,measurement='AI-reviewed planning rectangle, not segmented facial skin or validated trait visibility'))
 pilot_table.append(dict(item=item,setting=record['broad_setting'],correct_action_norms=record['correct_action_norms'],original_detail='blurred target' if item==1250 else 'visible facial detail',visible_frames=visible,frames=5,omi_outputs=15,additional_sob_reference_outputs=15 if item==969 else 0,median_region_area_percent=statistics.median(areas),min_region_side_px=min(sides),max_region_side_px=max(sides)))
require('23 reviewed visible pilot regions',len(region_rows)==23)
require('90 output records; 84 edited and 6 unchanged',len(audit['frames'])==90 and sum(r['editable'] for r in audit['frames'])==84 and sum(not r['editable'] for r in audit['frames'])==6)
require('Retained independent pixel audit reports zero outside-mask changes',sum(r['outside_mask_changed_pixels'] for r in audit['frames'])==0)
require('Every audited final output exists',all((SEQ/r['file']).is_file() for r in audit['frames']))
findings=json.loads((SEQ/'review_findings.json').read_text())
reg=json.loads((SEQ/'registration.json').read_text())
records=json.loads((SEQ/'generation_records.json').read_text())
require('18 retained sequence-condition records',len(records)==18)
require('84 retained registrations',len(reg)==84)
for row in pilot_table:
 row['limitations']=next(r['note'] for r in findings if r['item']==row['item'] and r['method']=='omi')
 row['experiment_ready']='not validated'
 masks=[]
 for rr in region_rows:
  if rr['item']!=row['item']:continue
  match=next(r for r in audit['frames'] if r['item']==rr['item'] and r['frame']==rr['frame'] and r['method']=='omi' and r['condition']=='baseline')
  rr['mask_support_area_percent']=100*match['mask_pixels']/(rr['width']*rr['height'])
  masks.append(rr['mask_support_area_percent'])
 row['median_mask_support_area_percent']=statistics.median(masks)

# Store sufficient retained inputs to reproduce the consolidation without the 400 MB source archives.
input_manifest=[]
for folder,name,dest in [
 ('EgoNormia_All_1853_Screening','official_annotations.json','official_annotations.json'),
 ('EgoNormia_All_1853_Screening','screening_results.json','screening_v1.json'),
 ('EgoNormia_All_1853_Screening','download_records.json','download_records.json'),
 ('EgoNormia_All_1853_Screening','source_metadata.json','source_metadata.json'),
 ('EgoNormia_Region_Screening_V2','region_screening_v2_results.json','region_screening_v2.json'),
 ('EgoNormia_Chunk2_Coverage','coverage_results.json','coverage_results.json'),
 ('EgoNormia_Chunk3_Pilots','pilot_manifest.json','pilot_manifest_original.json')]:
 p=SRC/folder/name;shutil.copy2(p,OUT/'data'/dest);input_manifest.append(dict(file='data/'+dest,sha256=digest(p),origin_archive=folder+'.zip'))
for name in ['audit.json','registration.json','review_findings.json','generation_records.json']:
 p=SEQ/name;shutil.copy2(p,OUT/'data'/('pilot_'+name));input_manifest.append(dict(file='data/pilot_'+name,sha256=digest(p),origin='Full_Sequence_Face_Tests'))
for item,f in [(605,3),(1625,2),(749,3)]:
 for c in ['original','low','baseline','high']:
  p=SEQ/f'item_{item}'/('original' if c=='original' else 'omi/'+c)/f'frame_{f}.png'
  dest=OUT/'evidence'/f'item_{item}_frame_{f}_{c}.png';shutil.copy2(p,dest)
  input_manifest.append(dict(file=str(dest.relative_to(OUT)),sha256=digest(p),origin='Canonical original' if c=='original' else 'Final registered fixed-mask output'))

summary=dict(population=1853,automatic_frames=9265,distinct_source_ids=full_sources,screening_failures=0,v1_triage=dict(labels),v2_labels=dict(counts),reviewed_examples=117,reviewed_frames=585,potential_candidates=100,potential_pool_sources=92,potential_pool_primary_norm_union=81,primary_promising=37,coarse_visual_settings=11,repeated_source_groups=7,candidates_in_repeated_sources=15,pilot_clips=5,pilot_source_ids=5,pilot_original_frames=25,visible_pilot_regions=23,omi_variant_frames=75,sob_reference_variant_frames=15,total_variant_frames=90,edited_face_instances=84,unchanged_copies=6,outside_mask_changed_pixels=0,validated_condition_sets=0,qwen_trials=0)
result=dict(summary=summary,review_group_counts={k:dict(v) for k,v in groups.items()},norm_coverage=norm_table,settings=dict(settings),pilot_table=pilot_table,pilot_region_measurements=region_rows,overall_pilot_region_summary=dict(n=23,min_area_percent=min(r['region_area_percent'] for r in region_rows),median_area_percent=statistics.median(r['region_area_percent'] for r in region_rows),max_area_percent=max(r['region_area_percent'] for r in region_rows),min_short_side_px=min(r['min_region_side_px'] for r in region_rows),max_short_side_px=max(r['min_region_side_px'] for r in region_rows)),metadata_corrections=manifest_corrections,registration_summary=dict(n=84,min_inliers=min(r['inliers'] for r in reg),max_median_inlier_error=max(r['median_inlier_error'] for r in reg)),passed_checks=checks,input_manifest=input_manifest)
(OUT/'consolidated_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'summary':summary,'region_summary':result['overall_pilot_region_summary'],'pilot_medians':[{k:r[k] for k in ['item','visible_frames','median_region_area_percent','min_region_side_px','max_region_side_px']} for r in pilot_table],'metadata_corrections':manifest_corrections,'checks':len(checks)},indent=2))
