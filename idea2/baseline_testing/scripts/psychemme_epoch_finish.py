"""Select an epoch checkpoint on source validation, then evaluate transfer/retention."""
import argparse,json,subprocess,sys
from pathlib import Path
from social_rl.common import STORE,write_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
r=Path(a.run);results=[]
for step in (750,1500,2250,3032):
 ck=r/f'checkpoint-{step:05d}'
 if not (ck/'adapter_config.json').exists():raise RuntimeError(f'Missing requested checkpoint {ck}')
 out=r/f'val120_{step:05d}'
 if not (out/'summary.json').exists():
  subprocess.run([sys.executable,'scripts/evaluate_psychemme_sft.py','--adapter',str(ck),'--output',str(out),'--limit','120'],check=True)
 result=json.loads((out/'summary.json').read_text());results.append({'step':step,'checkpoint':str(ck),**result})
best=max(results,key=lambda x:(x['accuracy'],-x['step']))
write_json(r/'selected.json',{'rule':'Highest source val120 accuracy; ties select earlier checkpoint','candidates':results,'selected':best})
for name,manifest,cap in [('ego1024_video400','ego_midterm_panel_accessible.jsonl',1024),('r3_retention','r3_test_accessible.jsonl',192)]:
 subprocess.run([sys.executable,'scripts/evaluate.py','--manifest',str(STORE/'data/manifests'/manifest),'--adapter',best['checkpoint'],'--output',str(r/name),'--conditions','video','--max-new-tokens',str(cap),'--max-hours','2.8'],check=True)
subprocess.run([sys.executable,'scripts/evaluate.py','--manifest',str(STORE/'data/manifests/ego_midterm_panel_accessible.jsonl'),'--adapter',best['checkpoint'],'--output',str(r/'visual_controls100'),'--conditions','blind','mismatch','--blind-same-prompt','--limit','100','--max-new-tokens','1024','--max-hours','1.0'],check=True)
