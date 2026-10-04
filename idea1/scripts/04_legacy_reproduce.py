"""Run unchanged supplied scripts in an isolated temporary tree; never mix legacy tables into current outputs."""
from core import *
import subprocess,sys,tempfile,shutil
from empath import Empath
lex=Empath();cats=sorted(lex.cats);O=option_rows(False)
def features(texts):return np.array([[a.get(c,0.) for c in cats] for a in ((lex.analyze(t,normalize=True) or {}) for t in texts)])
with tempfile.TemporaryDirectory(prefix='analysis1-legacy-') as td:
 root=Path(td)
 for sub in ['scripts','data/egonormia','outputs/tables','outputs/logs','outputs/figures']:(root/sub).mkdir(parents=True,exist_ok=True)
 for f in (ROOT/'legacy').glob('*.py'):shutil.copy(f,root/'scripts'/f.name)
 for name in ['final_data.json','final_data_eval.json','verified_split.json']:shutil.copy(ROOT/'data/egonormia'/name,root/'data/egonormia'/name)
 items=[k for k,v in D.items() if isinstance(v['desc'],str) and v['desc'].strip()]
 np.savez(root/'outputs/logs/empath_cache.npz',E=features(O.text),Ed=features([D[k]['desc'] for k in items]))
 for script,args in [('01_egonormia_audit.py',[]),('02_grounding_ladder.py',[]),('03_language_probe.py',['all']),('03b_artifact_check.py',[]),('05_artifact_x_ladder.py',[])]:
  with open(root/'outputs/logs'/('legacy_'+script+'.log'),'w') as f:subprocess.run([sys.executable,str(root/'scripts'/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',script,flush=True)
 shutil.copytree(root/'outputs',ROOT/'legacy/rerun_current_environment',dirs_exist_ok=True)
