"""One-pass external SFT with resumable optimizer state and overflow retries."""
import argparse,json,random,re,time,shutil
from pathlib import Path
import torch
from social_rl.common import STORE,read_jsonl,write_json,append_jsonl,sha256
from social_rl.model import VideoPolicy
from social_rl.tasks import r3_prompt
from social_rl.leakage import check_training
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();spec=json.load(open(a.spec));run=Path(spec['run']);run.mkdir(parents=True,exist_ok=True)
rows=read_jsonl(spec['manifest']);assert rows and all(r['split']=='train' for r in rows)
check_training(rows,run)
meta={'spec':spec,'manifest_sha256':sha256(spec['manifest']),'examples':len(rows)}
if (run/'metadata.json').exists() and json.load(open(run/'metadata.json'))!=meta:raise ValueError('Changed training inputs')
write_json(run/'metadata.json',meta)
latest=json.load(open(run/'latest.json')) if (run/'latest.json').exists() else None
if latest and latest['cursor']>=len(rows):exit()
seed=spec.get('seed',777);torch.manual_seed(seed);order=list(range(len(rows)));random.Random(seed).shuffle(order)
adapter=latest['checkpoint'] if latest else spec.get('init_adapter')
policy=VideoPolicy(adapter=adapter,trainable=True,model_path=spec.get('model_path'))
params=[p for p in policy.model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=5e-6,weight_decay=0.0);scaler=torch.amp.GradScaler('cuda',init_scale=32)
cursor=step=0
if latest:
 state=torch.load(Path(adapter)/'state.pt',map_location='cpu',weights_only=False);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);cursor=state['cursor'];step=state['step'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
def save():
 ck=run/f'checkpoint-{step:06d}';tmp=run/'saving.tmp'
 if tmp.exists():shutil.rmtree(tmp)
 policy.save(tmp);torch.save({'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'cursor':cursor,'step':step,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},tmp/'state.pt');tmp.rename(ck)
 write_json(run/'latest.json',{'checkpoint':str(ck),'cursor':cursor,'step':step,'complete':cursor>=len(rows)})
 old=sorted(run.glob('checkpoint-*'))
 for x in old[:-2]:shutil.rmtree(x)
start=time.monotonic();last_saved=step;overflows=0
while cursor<len(rows) and time.monotonic()-start<3.2*3600:
 row=rows[order[cursor]];rng=random.Random(seed+cursor)
 if row['dataset']=='r3':
  perm=rng.sample(range(5),5);prompt=r3_prompt(row,perm);target=f'Answer: {chr(65+perm.index(row["answer"]))}';inputs=policy.inputs(prompt,row['video_path'],end_seconds=row.get('clip_duration_seconds'))
 else:
  prompt=row['prompt'];answer=row['answer'];observation=row['observation']
  if row['task_type']=='single_choice':
   prefix,options=prompt.split('Options:\n',1);choices=options.strip().splitlines()
   if not all(len(s)>3 and s[1:3]=='. ' for s in choices):raise ValueError(row['id'])
   rng.shuffle(choices);mapping={s[0]:chr(65+i) for i,s in enumerate(choices)}
   def remap(t):return re.sub(r'\b((?:option|choice|answer)\s+)([A-J])\b',lambda m:m[1]+mapping.get(m[2],m[2]),t,flags=re.I)
   prefix=remap(prefix);observation=remap(observation);answer=mapping[answer]
   prompt=prefix+'Options:\n'+'\n'.join(f'{chr(65+i)}. {s[3:]}' for i,s in enumerate(choices))
  prompt+='\nGive one brief supported observation, then the answer. Format: Evidence: ... then Answer: ...'
  target=f'Evidence: {observation}\nAnswer: {answer}';inputs=policy.inputs(prompt,frame_paths=row['frames'])
 ids=policy.processor.tokenizer(target+policy.processor.tokenizer.eos_token,add_special_tokens=False,return_tensors='pt')['input_ids'].to(policy.device)
 if ids.shape[1]>512:raise ValueError(f'Target exceeds 512 tokens: {row["id"]}')
 opt.zero_grad(set_to_none=True);policy.model.train();loss=-policy.token_logps(inputs,ids).mean();scaler.scale(loss).backward();scaler.unscale_(opt)
 if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
  scaler.step(opt);scaler.update();overflows+=1
  append_jsonl(run/'train_log.jsonl',{'id':row['id'],'skipped':'overflow_retry','scale':scaler.get_scale()})
  if overflows>30:raise RuntimeError('Repeated gradient overflow')
  continue
 overflows=0;norm=torch.nn.utils.clip_grad_norm_(params,1.0);scaler.step(opt);scaler.update();cursor+=1;step+=1
 append_jsonl(run/'train_log.jsonl',{'step':step,'id':row['id'],'loss':float(loss.detach()),'tokens':ids.shape[1],**policy.memory()})
 if step%250==0:save();last_saved=step
if step>last_saved:save()
write_json(run/'status.json',{'steps':step,'cursor':cursor,'total':len(rows),'complete':cursor>=len(rows)})
