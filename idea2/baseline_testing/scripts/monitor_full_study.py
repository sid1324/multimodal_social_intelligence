"""One monitoring pass; designed to be invoked at 15-minute intervals."""
import datetime,json,pathlib,subprocess
from social_rl.common import STORE,write_json,append_jsonl
root=STORE/'data/full_study';snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training':{},'evaluation':{}}
snapshot['queue']=subprocess.check_output(['squeue','-h','-u','apatawar','-o','%i|%T|%M|%j|%R'],text=True).splitlines()
for run in (STORE/'runs/full_sft').glob('*'):
 log=run/'train_log.jsonl';last=None;count=0
 if log.exists():
  for line in log.open():
   if line.strip():count+=1;last=json.loads(line)
 snapshot['training'][run.name]={'events':count,'last':last,'latest':json.load(open(run/'latest.json')) if (run/'latest.json').exists() else None,'status':json.load(open(run/'status.json')) if (run/'status.json').exists() else None}
for run in (STORE/'runs/full_ego').glob('*'):
 total=missing=0;shards={}
 for part in run.glob('part*/predictions.jsonl'):
  n=0
  for line in part.open():
   if line.strip():
    n+=1;missing+='media' in json.loads(line).get('errors',{})
  total+=n;shards[part.parent.name]=n
 snapshot['evaluation'][run.name]={'predictions':total,'missing_media_predictions':missing,'shards':shards,'analysis_complete':(run/'coverage.json').exists()}
snapshot['media']=json.load(open(root/'coverage.json'))
write_json(root/'monitor_latest.json',snapshot);append_jsonl(root/'monitor_history.jsonl',snapshot)
print('UTC',snapshot['utc'])
for n,v in snapshot['training'].items():print('TRAIN',n,'step',v['last'].get('step') if v['last'] else None,'complete',v['latest'].get('complete') if v['latest'] else False)
for n,v in snapshot['evaluation'].items():print('EVAL',n,v['predictions'],'missing media',v['missing_media_predictions'],'analyzed',v['analysis_complete'])
print('MEDIA',{k:(v['accessible_examples'],v['annotation_examples']) for k,v in snapshot['media'].items()})
subprocess.run(['/ocean/projects/cis260338p/apatawar/social_norm_rl/envs/data/bin/python','scripts/repair_full_eval_media.py'],check=True)
