"""Resumable full-EgoNormia shard; unavailable media remain in denominators."""
import argparse,json,subprocess,sys
from pathlib import Path
from social_rl.common import STORE,read_jsonl
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--adapter',default='none');p.add_argument('--shard',type=int,required=True);a=p.parse_args();s=json.load(open(a.spec));out=STORE/'runs/full_ego'/s['name']/f'part{a.shard:02d}'
cmd=[sys.executable,'scripts/evaluate.py','--manifest',str(STORE/'data/full_study/ego_all.jsonl'),'--full-evaluation-protocol',str(STORE/'data/full_study/boundary.json'),'--output',str(out),'--offset',str(a.shard*100),'--limit','100','--conditions','video','--max-new-tokens','1024','--max-hours','3.2','--model-path',s['model_path']]
if a.adapter!='none':cmd+=['--adapter',a.adapter]
subprocess.run(cmd,check=True)
expected=min(100,1853-a.shard*100);rows=read_jsonl(out/'predictions.jsonl')
if len(rows)<expected:
 print('RESUME_SHARD',subprocess.check_output(['sbatch','--parsable',f'--array={a.shard}','slurm/full_eval.sbatch',a.spec,a.adapter],text=True),flush=True)
else:print('SHARD_COMPLETE',s['name'],a.shard,len(rows),flush=True)
# Dispatch analysis only after every shard is complete, with an atomic marker.
if all((STORE/'runs/full_ego'/s['name']/f'part{i:02d}'/'predictions.jsonl').exists() and len(read_jsonl(STORE/'runs/full_ego'/s['name']/f'part{i:02d}'/'predictions.jsonl'))==min(100,1853-i*100) for i in range(19)):
 marker=STORE/'runs/full_ego'/s['name']/'analysis_dispatched'
 try:
  with marker.open('x') as f:f.write(subprocess.check_output(['sbatch','--parsable','slurm/full_analysis.sbatch',s['name']],text=True))
 except FileExistsError:pass
