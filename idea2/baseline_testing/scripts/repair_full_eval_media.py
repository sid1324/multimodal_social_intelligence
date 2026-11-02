"""Repair completed model evaluations after missing source media are recovered."""
import json,subprocess,pathlib,time
from social_rl.common import STORE,read_jsonl,write_jsonl,write_json,sha256
root=STORE/'data/full_study';rows={x['id']:x for x in read_jsonl(root/'ego_all.jsonl')}
commands=subprocess.check_output(['squeue','-h','-u','apatawar','-o','%o'],text=True)
registry=json.load(open(root/'registry.json'))
for registry_file in ('registry_amendments.json','registry_followups.json'):
 if (root/registry_file).exists():registry+=json.load(open(root/registry_file))['models']
for s in registry:
 run=STORE/'runs/full_ego'/s['name'];coverage=run/'coverage.json'
 spec=root/'specs'/f'{s["name"]}.json'
 if not coverage.exists() or str(spec) in commands:continue
 repaired=[]
 for i in range(19):
  f=run/f'part{i:02d}'/'predictions.jsonl'
  if not f.exists():break
  old=read_jsonl(f);bad={x['id'] for x in old if 'media' in x.get('errors',{}) and pathlib.Path(rows[x['id']]['video_path']).exists()}
  if not bad:continue
  backup=f.with_name(f'predictions.before-media-recovery-{sha256(f)[:12]}.jsonl')
  if not backup.exists():backup.write_bytes(f.read_bytes())
  write_jsonl(f,[x for x in old if x['id'] not in bad]);repaired.append(i)
 if repaired:
  state=json.load(open(coverage));state['pending_media_repair_shards']=repaired;write_json(coverage,state)
  (run/'analysis_dispatched').unlink(missing_ok=True)
  adapter=s.get('adapter','none')
  if s.get('manifest'):adapter=json.load(open(pathlib.Path(s['run'])/'latest.json'))['checkpoint']
  job=subprocess.check_output(['sbatch','--parsable','--array='+','.join(map(str,repaired))+'%2','slurm/full_eval.sbatch',str(spec),adapter],text=True).strip()
  write_json(run/'media_repair_dispatch.json',{'shards':repaired,'job':job,'time_unix':time.time()})
  print(s['name'],job,repaired,flush=True)
