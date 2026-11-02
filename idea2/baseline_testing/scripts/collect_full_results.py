"""Collect only complete, media-repaired full-benchmark model results."""
import datetime,json,csv,pathlib
from social_rl.common import STORE,ROOT,write_json,read_jsonl
from summarize import cluster_ci
root=STORE/'data/full_study';registry=json.load(open(root/'registry.json'));result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':{}}
previous_file=ROOT/'reports/results/full_benchmark_results.json'
previous=json.load(open(previous_file)).get('models',{}) if previous_file.exists() else {}
for spec in registry:
 run=STORE/'runs/full_ego'/spec['name'];c=run/'coverage.json'
 if not c.exists():continue
 cov=json.load(open(c))
 if cov.get('pending_media_repair_shards') or cov.get('predictions')!=1853 or cov.get('media_failures')!=0:continue
 cohorts={}
 for n in ('all1853','previous400','new_source_groups','other_development'):
  summary=json.load(open(run/'analysis'/n/'summary.json'));cohorts[n]=next(iter(summary['conditions'].values()))
 result['models'][spec['name']]={'model_path':spec['model_path'],'cohorts':cohorts}
for name,record in result['models'].items():
 base='qwen38_27b_base' if '3.8' in record['model_path'] else 'qwen35_9b_base'
 if base==name or base not in result['models']:continue
 record['paired_vs_base']={}
 for cohort in ('all1853','previous400','new_source_groups','other_development'):
  def read(n):
   p=STORE/'runs/full_ego'/n/'analysis'/cohort/'per_example.csv'
   return {x['id']:x for x in csv.DictReader(open(p))}
  a,b=read(base),read(name);assert set(a)==set(b)
  ids=sorted(a);group_rows=[{'group':a[i]['group']} for i in ids]
  record['paired_vs_base'][cohort]={}
  for metric in ('action','justification','joint','sensibility_iou'):
   delta=[float(b[i][metric])-float(a[i][metric]) for i in ids]
   record['paired_vs_base'][cohort][metric]={'difference':sum(delta)/len(delta),'ci95':cluster_ci(group_rows,delta)}
write_json(ROOT/'reports/results/full_benchmark_results.json',result)
lines=['# Complete full-EgoNormia results','',f"As of {result['utc']}. Only 1,853/1,853 media-repaired results are included.",'','| Model | All 1853 joint | Previous 400 joint | New source groups (1025) joint | Action | Justification | Sensibility IoU |','|---|---:|---:|---:|---:|---:|---:|']
for name,r in result['models'].items():
 c=r['cohorts'];v=c['all1853'];values=[v['joint'],c['previous400']['joint'],c['new_source_groups']['joint'],v['action'],v['justification'],v['sensibility_iou']]
 lines.append('| '+name+' | '+' | '.join(f'{100*x:.1f}%' for x in values)+' |')
lines+=['','The full release mixes development examples and the formerly reserved cohort. Interpret comparisons using model provenance, paired intervals, and seed variability.']
(ROOT/'reports/FULL_BENCHMARK_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Complete models:',len(result['models']))
for name,r in result['models'].items():
 if name not in previous:
  print('NEW_COMPLETE',name,'all joint',r['cohorts']['all1853']['joint'],'new-source joint',r['cohorts']['new_source_groups']['joint'])
