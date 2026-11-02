"""Memory and optimizer smoke on normal and long source prompts with 32 images."""
import argparse,time,torch
from social_rl.common import STORE,read_jsonl,write_json
from social_rl.model import VideoPolicy
from social_rl.frame_images import ordered_image_inputs
p=argparse.ArgumentParser();p.add_argument('--family',required=True);a=p.parse_args();name='Qwen3.5-9B' if a.family=='9b' else 'Qwen3.8-27B';out=STORE/'runs'/f'frame_images_smoke_{a.family}';out.mkdir(exist_ok=True,parents=True)
result={'model':name,'visual_format':'32 ordered image frames, <=256 px, no synthetic timestamps','stages':[]}
try:
 model=VideoPolicy(model_path=STORE/'models'/name,trainable=True);params=[x for x in model.model.parameters() if x.requires_grad];optimizer=torch.optim.AdamW(params,lr=5e-6,weight_decay=0);scaler=torch.amp.GradScaler('cuda',init_scale=1)
 rows=read_jsonl(STORE/'data/full_study/psy_ordered_images32.jsonl');samples=[rows[0],max(rows,key=lambda x:len(x['prompt']))]
 for row in samples:
  inputs=ordered_image_inputs(model,row['prompt'],row['frames']);text=f'Evidence: {row["observation"]}\nAnswer: {row["answer"]}';ids=model.processor.tokenizer(text+model.processor.tokenizer.eos_token,add_special_tokens=False,return_tensors='pt')['input_ids'].to(model.device)
  optimizer.zero_grad(set_to_none=True);model.model.train();loss=-model.token_logps(inputs,ids).mean();scaler.scale(loss).backward();scaler.unscale_(optimizer);finite=all(x.grad is None or torch.isfinite(x.grad).all().item() for x in params);scaler.step(optimizer);scaler.update()
  result['stages'].append({'id':row['id'],'input_tokens':inputs['input_ids'].shape[1],'loss':float(loss.detach()),'finite_gradients':finite,**model.memory()});write_json(out/'status.json',result)
  del inputs,ids,loss
 if not all(x['finite_gradients'] for x in result['stages']):raise RuntimeError('Nonfinite smoke gradient')
 result['complete']=True
except Exception as e:
 result['error']=repr(e);result['complete']=False;raise
finally:write_json(out/'status.json',result)
