"""Prepare and acquire full external training and official EgoNormia coverage."""
import argparse,concurrent.futures,json,tarfile
from pathlib import Path
from social_rl.common import STORE,ROOT,read_jsonl,write_json,write_jsonl,sha256
from acquire_media import fetch_ego,fetch_r3
p=argparse.ArgumentParser();p.add_argument('--task',choices=['freeze','r3','ego','finalize'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=16);a=p.parse_args()
out=STORE/'data/full_study';out.mkdir(parents=True,exist_ok=True)
if a.task=='freeze':
 old=STORE/'data/manifests';ego=read_jsonl(old/'ego_midterm_all.jsonl')+read_jsonl(old/'ego_final_RESERVED.jsonl')
 for x in ego:x['original_partition']=x['split'];x['split']='evaluation'
 ego.sort(key=lambda x:x['id']);assert len(ego)==1853 and len({x['id'] for x in ego})==1853
 write_jsonl(out/'ego_all.jsonl',ego)
 r3=sum((read_jsonl(old/f'r3_{s}_all.jsonl') for s in ('train','val','test')),[])
 for x in r3:x['original_partition']=x['split'];x['split']='train'
 assert len(r3)==41529;write_jsonl(out/'r3_all.jsonl',r3)
 root=STORE/'raw/psychemme/3b80a47473802b160b54e760d5dca070efd96f4c';eligible=set((STORE/'data/psychemme/audit/eligible_sft_ids.txt').read_text().splitlines())
 qs={x['id']:x for x in read_jsonl(root/'metadata/questions.jsonl')};psy=[];needed=set()
 for x in read_jsonl(root/'sft/train.jsonl'):
  if x['id'] not in eligible:continue
  target=json.loads(x['messages'][-1]['content']);u=x['messages'][0]['content'];u=u[u.index('Timestamped transcript'):u.index('Slot meanings:')].strip()
  frames=[x['images'][i] for i in (0,4,9,13,18,22,27,31)];needed.update(frames)
  psy.append({'id':x['id'],'dataset':'psychemme','split':'train','prompt':u,'answer':target['answer'],'task_type':qs[x['id']]['task_type'],'observation':next(str(v) for k,v in target.items() if k!='answer' and v),'frames':[str(STORE/'data/psychemme'/f) for f in frames]})
 assert len(psy)==3534
 with tarfile.open(root/'media/frames.tar') as tar:
  for m in tar:
   n=m.name.removeprefix('./');dest=STORE/'data/psychemme'/n
   if n not in needed or dest.exists():continue
   if not m.isfile() or '..' in Path(n).parts:raise ValueError(n)
   dest.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(m) as src,dest.open('wb') as dst:dst.write(src.read())
 write_jsonl(out/'psy_all.jsonl',psy)
 sources=sorted({x['group'] for x in ego});(out/'ego_source_ids.txt').write_text('\n'.join(sources)+'\n')
 write_json(out/'boundary.json',{'authorization':'User requested full EgoNormia evaluation and full external SFT in this session.','ego_manifest_sha256':sha256(out/'ego_all.jsonl'),'ego_examples':1853,'ego_source_count':len(sources),'ego_source_ids_sha256':sha256(out/'ego_source_ids.txt'),'psy_eligible':3534,'psy_excluded_source_overlap':104,'r3_annotations':41529,'source_validation_now_training':True,'generation_cap':1024})
 print((out/'boundary.json').read_text())
elif a.task in ('r3','ego'):
 rows=read_jsonl(out/f'{a.task}_all.jsonl')
 if a.task=='r3':rows=list({x['video_id']:x for x in rows}.values())
 rows=rows[a.shard::a.shards];results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for r in pool.map(fetch_r3 if a.task=='r3' else fetch_ego,rows):
   results.append(r);print(json.dumps(r),flush=True)
   write_jsonl(out/f'{a.task}_media_{a.shard:02d}.jsonl',results)
else:
 report={}
 for kind,n in [('r3',16),('ego',16)]:
  results=sum((read_jsonl(out/f'{kind}_media_{i:02d}.jsonl') for i in range(n)),[])
  ok={x['id'] for x in results if x['status']=='ok'}
  allrows=read_jsonl(out/f'{kind}_all.jsonl');rows=[x for x in allrows if x.get('video_id',x['id']) in ok]
  if not rows:raise RuntimeError(f'No {kind} media')
  write_jsonl(out/f'{kind}_accessible.jsonl',rows)
  report[kind]={'annotation_examples':len(allrows),'accessible_examples':len(rows),'attempted_media':len(results),'ok_media':len(ok),'failures':[x for x in results if x['status']!='ok']}
 write_json(out/'coverage.json',report);print(json.dumps({k:{a:b for a,b in v.items() if a!='failures'} for k,v in report.items()},indent=2))
