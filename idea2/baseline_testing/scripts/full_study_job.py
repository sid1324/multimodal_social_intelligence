"""Run one training session and schedule continuation or fixed downstream work."""
import json,subprocess,sys,pathlib
from social_rl.common import STORE,write_json
specfile=pathlib.Path(sys.argv[1]);spec=json.load(open(specfile));root=STORE/'data/full_study'
def submit(*args):return subprocess.check_output(['sbatch','--parsable',*args],text=True).strip()
if spec.get('stage')=='evaluate':
 pass
else:
 subprocess.run([sys.executable,'scripts/train_full_external.py','--spec',str(specfile)],check=True)
 latest=json.load(open(pathlib.Path(spec['run'])/'latest.json'))
 if not latest['complete']:
  print('CONTINUATION',submit('slurm/full_train.sbatch',str(specfile)),flush=True);sys.exit()
 spec['adapter']=latest['checkpoint']
 if spec.get('next_spec'):
  nxt=pathlib.Path(spec['next_spec']);s=json.load(open(nxt));s['init_adapter']=spec['adapter'];write_json(nxt,s);print('NEXT_STAGE',submit('slurm/full_train.sbatch',str(nxt)),flush=True)
 acquisition=json.load(open(root/'acquisition_jobs.json'))
 print('FULL_EVAL',submit('--dependency=afterok:'+acquisition['ego'],'--array=0-18%4','slurm/full_eval.sbatch',str(specfile),spec['adapter']),flush=True)
