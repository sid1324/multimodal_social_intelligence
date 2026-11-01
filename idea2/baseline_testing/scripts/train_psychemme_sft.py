"""Single-V100 assistant-only PsycheMME SFT from base or frozen R3 SFT."""
import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import torch

from social_rl.common import STORE, append_jsonl, read_jsonl, write_json
from social_rl.model import VideoPolicy

R3_ADAPTER = STORE / 'runs/sft/checkpoint-00100'


def prompt_and_target(row, seed):
    randomizer = random.Random(seed)
    prefix, options = row['prompt'].split('Options:\n', 1)
    lines = options.strip().splitlines()
    choices = [x for x in lines if len(x)>3 and x[1:3]=='. ']
    if len(choices)!=len(lines) or row['answer'] not in {x[0] for x in choices}:
        raise ValueError(f'Invalid options: {row["id"]}')
    randomizer.shuffle(choices)
    answer = chr(65 + next(i for i,x in enumerate(choices) if x[0]==row['answer']))
    choices = [f'{chr(65+i)}. {x[3:]}' for i,x in enumerate(choices)]
    prompt = (prefix + 'Options:\n' + '\n'.join(choices) +
              '\nGive one brief observation supported by the video, then the best option. '
              'Format: Evidence: ... then Answer: X')
    target = f'Evidence: {row["observation"]}\nAnswer: {answer}'
    return prompt,target


def save(policy, optimizer, scaler, out, step, cursor, seed):
    path=out/f'checkpoint-{step:05d}'
    temp=out/f'checkpoint-{step:05d}.tmp'
    if temp.exists(): shutil.rmtree(temp)
    policy.save(temp)
    torch.save({'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'step':step,'cursor':cursor,'rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()},temp/'training_state.pt')
    temp.rename(path)
    write_json(out/'latest.json',{'checkpoint':str(path),'step':step,'cursor':cursor})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start',choices=['base','r3sft'],required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--steps',type=int,default=300)
    ap.add_argument('--max-hours',type=float,default=3.5)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--seed',type=int,default=777)
    ap.add_argument('--save-every',type=int,default=50)
    args=ap.parse_args()
    manifest=STORE/'data/psychemme/train.jsonl'
    rows=read_jsonl(manifest)
    if not rows or any(x['split']!='train' for x in rows):raise ValueError('Only train split allowed')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    metadata={'start':args.start,'steps':args.steps,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'seed':args.seed,'lr':5e-6,'frames':8,'max_target_tokens':128,
              'save_every':args.save_every}
    old=out/'metadata.json'
    if old.exists() and json.loads(old.read_text())!=metadata:raise ValueError('Run metadata mismatch')
    if (out/'latest.json').exists() and not args.resume:raise ValueError('Existing checkpoint; use --resume')
    write_json(old,metadata)
    torch.manual_seed(args.seed)
    order=list(range(len(rows)));random.Random(args.seed).shuffle(order)
    ckpt=json.loads((out/'latest.json').read_text())['checkpoint'] if args.resume else None
    adapter=ckpt or (str(R3_ADAPTER) if args.start=='r3sft' else None)
    policy=VideoPolicy(adapter=adapter,trainable=True)
    params=[x for x in policy.model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=5e-6,weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    step=cursor=0
    if ckpt:
        s=torch.load(Path(ckpt)/'training_state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(s['optimizer']);scaler.load_state_dict(s['scaler'])
        step=s['step'];cursor=s['cursor'];torch.set_rng_state(s['rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])
    began=time.monotonic();saved=step
    attempts=0
    while step<args.steps and time.monotonic()-began<args.max_hours*3600:
        attempts+=1
        if attempts>args.steps*3:
            raise RuntimeError('Too many nonfinite gradient attempts')
        row=rows[order[cursor%len(order)]];cursor+=1
        prompt,target=prompt_and_target(row,args.seed+cursor)
        inputs=policy.inputs(prompt,frame_paths=row['frames'])
        ids=policy.processor.tokenizer(target+policy.processor.tokenizer.eos_token,
                                       add_special_tokens=False,return_tensors='pt')['input_ids'].to(policy.device)
        if ids.shape[1]>128:raise ValueError(f'Target too long {row["id"]}')
        optimizer.zero_grad(set_to_none=True);policy.model.train()
        loss=-policy.token_logps(inputs,ids).mean()
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            scaler.step(optimizer);scaler.update()
            append_jsonl(out/'train_log.jsonl',{'cursor':cursor,'id':row['id'],
                         'skipped':'nonfinite_gradients','loss':float(loss.detach()),
                         'scale':float(scaler.get_scale()),**policy.memory()})
            continue
        norm=torch.nn.utils.clip_grad_norm_(params,1.0)
        scaler.step(optimizer);scaler.update();step+=1
        event={'step':step,'cursor':cursor,'id':row['id'],'group':row['group'],
               'loss':float(loss.detach()),'target_tokens':ids.shape[1],'grad_norm':float(norm),**policy.memory()}
        append_jsonl(out/'train_log.jsonl',event)
        print(json.dumps(event),flush=True)
        if step%args.save_every==0:
            save(policy,optimizer,scaler,out,step,cursor,777);saved=step
    if step>saved:save(policy,optimizer,scaler,out,step,cursor,777)
    write_json(out/'status.json',{'steps':step,'cursor':cursor,'elapsed_seconds':time.monotonic()-began,
                                  'complete':step>=args.steps})

if __name__=='__main__':main()
