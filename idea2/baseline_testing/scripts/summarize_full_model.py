"""Require all 1,853 predictions; report full and pre-existing evaluation cohorts."""
import argparse,json,subprocess,sys
from pathlib import Path
from social_rl.common import STORE,ROOT,read_jsonl,write_jsonl,write_json
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();root=STORE/'data/full_study';run=STORE/'runs/full_ego'/a.name
rows=read_jsonl(root/'ego_all.jsonl');pred=sum((read_jsonl(run/f'part{i:02d}'/'predictions.jsonl') for i in range(19)),[])
if len(pred)!=1853 or {x['id'] for x in pred}!={x['id'] for x in rows}:raise ValueError('Incomplete or duplicated full evaluation')
assert len({x['id'] for x in pred})==1853
panel=set((ROOT/'egonormia_400_sample_ids.txt').read_text().splitlines())
cohorts={'all1853':rows,'previous400':[x for x in rows if x['id'] in panel],'new_source_groups':[x for x in rows if x['original_partition']=='final'],'other_development':[x for x in rows if x['original_partition']=='midterm' and x['id'] not in panel]}
for name,items in cohorts.items():
 out=run/'cohorts'/name/a.name;out.mkdir(parents=True,exist_ok=True);ids={x['id'] for x in items};manifest=out/'manifest.jsonl';write_jsonl(manifest,items);write_jsonl(out/'predictions.jsonl',[x for x in pred if x['id'] in ids])
 subprocess.run([sys.executable,'scripts/summarize.py','--manifest',str(manifest),'--runs',str(out),'--output',str(run/'analysis'/name)],check=True,stdout=subprocess.DEVNULL)
write_json(run/'coverage.json',{'expected':1853,'predictions':len(pred),'cohorts':{k:len(v) for k,v in cohorts.items()},'media_failures':sum('media' in x.get('errors',{}) for x in pred)})
print((run/'coverage.json').read_text())
