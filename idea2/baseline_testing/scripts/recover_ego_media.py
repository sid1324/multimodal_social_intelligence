"""Recover rate-limited official EgoNormia clips serially, respecting Retry-After."""
import json,os,time,subprocess
from pathlib import Path
import requests,av
from social_rl.common import STORE,read_jsonl,append_jsonl,write_json,write_jsonl,config
root=STORE/'data/full_study';rows=read_jsonl(root/'ego_all.jsonl');began=time.monotonic();streak=0;recovered=0
history=read_jsonl(root/'ego_recovery.jsonl') if (root/'ego_recovery.jsonl').exists() else []
from collections import Counter
permanent=Counter(x['id'] for x in history if x.get('status')=='failed' and ('404' in x.get('error','') or '403' in x.get('error','')))
jobsfile=root/'ego_recovery_jobs.json'
jobs=json.load(open(jobsfile)) if jobsfile.exists() else []
jobs.append(os.environ.get('SLURM_JOB_ID','manual'));write_json(jobsfile,jobs)
for row in rows:
 p=Path(row['video_path'])
 if p.exists():continue
 if permanent[row['id']]>=2:continue
 if time.monotonic()-began>5*3600:break
 url=f"https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/{config()['ego_revision']}/video/{row['id']}/video_prev.mp4"
 tmp=p.with_suffix('.recover');p.parent.mkdir(parents=True,exist_ok=True)
 event={'id':row['id'],'dataset':'ego','path':str(p),'status':'failed'}
 try:
  with requests.get(url,stream=True,timeout=(30,90)) as r:
   if r.status_code==429:
    retry=r.headers.get('Retry-After','60');event['error']='HTTP 429';append_jsonl(root/'ego_recovery.jsonl',event)
    time.sleep(min(900,max(60,int(retry) if retry.isdigit() else 60)));streak+=1
    if streak>=3:break
    continue
   r.raise_for_status()
   with tmp.open('wb') as f:
    for part in r.iter_content(1024*1024):
     if part:f.write(part)
  with av.open(str(tmp)) as c:next(c.decode(video=0))
  tmp.replace(p);event.update(status='ok',bytes=p.stat().st_size);streak=0;recovered+=1
 except Exception as e:event['error']=str(e)[:600];streak+=1
 append_jsonl(root/'ego_recovery.jsonl',event);print(json.dumps(event),flush=True);time.sleep(2)
 if streak>=5:break
available=[x for x in rows if Path(x['video_path']).exists()];write_jsonl(root/'ego_accessible.jsonl',available)
coverage=json.load(open(root/'coverage.json'));coverage['ego']['accessible_examples']=len(available);coverage['ego']['ok_media']=len(available);coverage['ego']['remaining_missing_ids']=[x['id'] for x in rows if not Path(x['video_path']).exists()];write_json(root/'coverage.json',coverage)
write_json(root/'ego_recovery_status.json',{'recovered_this_job':recovered,'available':len(available),'missing':1853-len(available)})
# Automatically retry only after a cooldown; saved files are reused.
if len(available)<1853 and len(jobs)<16 and any(permanent[x['id']]<2 for x in rows if not Path(x['video_path']).exists()):
 print('NEXT_RECOVERY',subprocess.check_output(['sbatch','--parsable','--begin=now+15minutes','slurm/recover_ego_media.sbatch'],text=True),flush=True)
