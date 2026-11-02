"""Finalize expanded R3 media coverage and dispatch fixed one-pass SFT."""
import json,subprocess
from pathlib import Path
from social_rl.common import STORE,write_json
root=STORE/'data/full_study'
subprocess.run(['/ocean/projects/cis260338p/apatawar/social_norm_rl/envs/data/bin/python','scripts/full_data_prepare.py','--task','finalize'],check=True)
registry=json.load(open(root/'registry.json'))
for s in registry:
 if s.get('start_after_r3_acquisition'):
  print(subprocess.check_output(['sbatch','--parsable','slurm/full_train.sbatch',str(root/'specs'/f'{s["name"]}.json')],text=True),flush=True)
