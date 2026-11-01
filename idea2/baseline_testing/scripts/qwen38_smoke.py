"""Measure 27B NF4 video inference and a short language-LoRA backward pass."""
import json,time,torch
from social_rl.common import STORE,read_jsonl,write_json
from social_rl.model import VideoPolicy
from train_psychemme_sft import prompt_and_target
out=STORE/'runs/qwen38_v100_smoke';out.mkdir(parents=True,exist_ok=True)
result={'model':json.loads((STORE/'models/Qwen3.8-27B/READY').read_text()),'stages':[]}
try:
 policy=VideoPolicy(model_path=STORE/'models/Qwen3.8-27B',trainable=True)
 result['stages'].append({'stage':'loaded',**policy.memory()});write_json(out/'status.json',result)
 row=read_jsonl(STORE/'data/psychemme/train.jsonl')[0]
 prompt,target=prompt_and_target(row,777)
 inputs=policy.inputs(prompt,frame_paths=row['frames'])
 t=time.monotonic();tokens,text=policy.generate(inputs,max_new_tokens=128)
 result['stages'].append({'stage':'video_generation','seconds':time.monotonic()-t,'tokens':tokens.numel(),'response':text,**policy.memory()})
 write_json(out/'status.json',result)
 ids=policy.processor.tokenizer(target+policy.processor.tokenizer.eos_token,add_special_tokens=False,return_tensors='pt')['input_ids'].to(policy.device)
 policy.model.train();loss=-policy.token_logps(inputs,ids).mean();loss.backward()
 finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in policy.model.parameters())
 result['stages'].append({'stage':'backward','loss':float(loss.detach()),'finite_gradients':finite,**policy.memory()})
 result['complete']=True
except Exception as e:
 result['error']=repr(e);result['complete']=False
 raise
finally:write_json(out/'status.json',result)
