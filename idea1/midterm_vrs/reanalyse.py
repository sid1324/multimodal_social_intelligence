import sys,json,hashlib,os
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--artifact-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
R=Path(a.artifact_root).resolve();O=Path(a.out).resolve();O.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'));os.chdir(R)
from grounded_norm_sft.io_utils import verify_sidecar
from grounded_norm_sft.metrics.normlens import score_model,contrast,per_seed_values
from grounded_norm_sft.metrics.calibration import calibrated_rows
from grounded_norm_sft.report_provenance import validate_hybrid_identity
inventory={}
def check(p,expected=None):
 p=Path(p);p=p if p.is_absolute() else R/p
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if expected:assert h==expected,(str(p),h,expected)
 side=Path(str(p)+'.sha256'); verified=verify_sidecar(p) if side.exists() else None
 inventory[str(p)]={'sha256':h,'sidecar_verified':bool(verified)}
 return p
def js(p):return json.loads(check(p).read_text())
def rows(p):return [json.loads(x) for x in check(p).read_text().splitlines()]
primary=js('provenance/psc_midterm_normlens_primary_v1.json')
manifest='private/prepare_normlens_v1/normlens_examples.jsonl'; check(manifest,primary['inputs_sha256']['manifest']);alllabels=rows(manifest);labels={r['example_id']:r for r in alllabels if r['split']=='test'};ids=sorted(labels)
models={};conditions={};proofs={}
for arm,fn in [('B0','psc_b0_conditions_test_v2.json'),('B1','psc_b1_conditions_test_v1.json'),('H-T','psc_ht_corrected_primary_scored_v1.json'),('H-R','psc_hr_corrected_primary_scored_v1.json')]:
 d=js('provenance/'+fn);conditions[arm]={}
 for cond in ['image','text_only','shuffle']:
  if cond not in d['inputs']:continue
  inp=d['inputs'][cond]
  for v in inp.values():check(v['path'],v['sha256'])
  pred=Path(inp['predictions']['path']); identity=js(pred.parent/'identity.json'); man=js(pred.parent/'manifest.json')
  assert man['complete'] and man['predicted']==291 and man['failed']==0 and not rows(pred.parent/'failures.jsonl')
  assert man['identity']==identity
  rr=rows(pred);assert sorted(r['example_id'] for r in rr)==ids
  if arm.startswith('H-'):
   ex=d['inputs']['export_report'];check(ex['path'],ex['sha256'])
   proof=validate_hybrid_identity(identity,arm=arm,task='normlens',condition=cond,export_report=ex['path'],scoring_seal_sha256='61f71b05ba9b46193dde8a7ab1b48e76c1c53e3f5365004652882980d754f1c6');assert proof['inference_adapter_identity_unverified'] is False;proofs[arm]=proof
   for side in pred.parent.rglob('*.sha256'):check(side.with_name(side.name[:-7]))
  conditions[arm][cond]=score_model(rr,labels,ids,condition=cond)
  if cond=='image':models[arm]=rr
fit=js('provenance/psc_b0_calibration_v1.json');check('provenance/psc_b0_calibration_v1.json',primary['inputs_sha256']['B0-cal fit'])
for a in ['B0','B1']:
 assert hashlib.sha256(Path(js('provenance/'+('psc_b0_conditions_test_v2.json' if a=='B0' else 'psc_b1_conditions_test_v1.json'))['inputs']['image']['predictions']['path']).read_bytes()).hexdigest()==primary['inputs_sha256'][a]
models['B0-cal']=calibrated_rows(models['B0'],fit['bias']);conditions['B0-cal']={'image':score_model(models['B0-cal'],labels,ids)}
contrasts={}
for a,b in [('B1','B0-cal'),('B1','B0'),('H-T','B1'),('H-R','B1'),('H-T','H-R')]:
 contrasts[a+' minus '+b]=contrast(models[a],models[b],labels,ids,b_frozen=b in ['B0','B0-cal'])
# Paired transitions identify regressions as well as improvements.
transitions={}
for a,b in [('B1','B0-cal'),('H-T','B1'),('H-T','H-R')]:
 aa=next(iter(per_seed_values(models[a],labels,ids).values()));bb=next(iter(per_seed_values(models[b],labels,ids).values())); counts={'improved':0,'regressed':0,'both_correct':0,'both_wrong':0,'tied_gold':0}
 for i in ids:
  if aa[i]['gold'] is None:counts['tied_gold']+=1;continue
  ac=aa[i]['pred']==aa[i]['gold'];bc=bb[i]['pred']==bb[i]['gold'];counts['both_correct' if ac and bc else 'improved' if ac else 'regressed' if bc else 'both_wrong']+=1
 transitions[a+' versus '+b]=counts
result={'conditions':conditions,'contrasts':contrasts,'transitions':transitions,'fidelity':proofs,'split_counts':{s:sum(r['split']==s for r in alllabels) for s in ['train','dev','test']},'scope':'Saved-output CPU reanalysis only; no new predictions, calibration fit, checkpoint choice or GPU work. Hybrid contrasts exploratory, no multiplicity correction; image-group CIs do not cover training-seed variation.'}
(O/'recomputed.json').write_text(json.dumps(result,indent=2)+'\n');(O/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
for a,c in conditions.items():
 for k,v in c.items(): print(a,k,v['seed_mean']['cond_nll'],v['seed_mean']['accuracy'])
for k,v in contrasts.items():print(k,v['bootstrap'])
print('Transitions',transitions,'verified_files',len(inventory))
