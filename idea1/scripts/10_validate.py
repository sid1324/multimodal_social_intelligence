"""Independent arithmetic, OOF, release-parity, and report-value checks."""
from core import *
from sklearn.metrics import roc_auc_score
from pathlib import Path
import re,hashlib
checks=[]
def check(name,condition,detail=''):
 if not condition:raise AssertionError(name+': '+detail)
 checks.append(dict(check=name,status='PASS',detail=detail))
expected={'final_data.json':'0f19bc41374bf13b359a55fb65b4ba3e5db5a217c31727bc9ff5b1fa6d32e225','final_data_eval.json':'8d81c140e914aabf448f3962cf3964a42373970d2d8479f30eed16489be1aa9e','verified_split.json':'65820d50f700ca5d8bc2225910db8e1996e7b8eddb20d495a8cefcf732c23cff'}
for n,h in expected.items():check('raw_hash_'+n,hashlib.sha256((ROOT/'data/egonormia'/n).read_bytes()).hexdigest()==h)
L=pd.read_csv(T/'ladder.csv');I=pd.read_csv(T/'matched_item_scores.csv')
for r in L.itertuples():
 ks=I[I.family==r.family].item.tolist();idx=FAMILIES[r.family][r.condition];correct=0
 for k in ks:
  v=E[k][idx]['best']['results'];g=D[k]['correct'];correct+=int(v==[g,g]) if r.metric=='joint' else int(v[0 if r.metric=='action' else 1]==g)
 check(f'ladder_{r.family}_{r.metric}_{r.condition}',correct==r.correct and len(ks)==r.n and abs(correct/len(ks)-r.accuracy)<1e-12)
for r in pd.read_csv(T/'paired.csv').itertuples():check(f'paired_arithmetic_{r.family}_{r.metric}_{r.subset}',abs((r.gained-r.lost)/r.n-r.delta)<1e-12 and r.correct_to-r.correct_from==r.gained-r.lost)
P=pd.read_csv(T/'probes.csv')
for space in ['five_options','legacy_four_nonempty']:
 for name in ['tfidf_source','tfidf_strict','behavior','justification','length']:
  p=pd.read_csv(T/f'probe_items_{space}_{name}.csv');r=P[(P.space==space)&(P.probe==name)].iloc[0]
  check(f'probe_{space}_{name}',len(p)==len(D) and len(set(p.item))==len(D) and p.correct.sum()==r.correct and abs(p.correct.mean()-r.accuracy)<1e-12)
  check(f'probe_folds_{space}_{name}',p.groupby(p.item.map(source)).fold.nunique().max()==1)
# Independently aggregate the final error split.
p=pd.read_csv(T/'probe_items_five_options_tfidf_strict.csv').set_index('item');check('strict_error_count',int((p.outcome==0).sum())==742)
for r in pd.read_csv(T/'artifact_robustness.csv').itertuples():
 f=I[I.family==r.family];q=f[f.item.map(p.outcome).eq(0 if r.subset=='probe_wrong' else 1)];check('error_split_'+r.family+'_'+r.subset,len(q)==r.n and int(q.desc.sum())==r.correct_from and int(q.grid.sum())==r.correct_to)
# Recompute every frozen representation AUROC from saved OOF predictions.
for label in ['sbert_option_norm','sbert_description_norm']:
 q=pd.read_csv(T/(label+'_oof.csv'));tab=pd.read_csv(T/'sbert_norm_auroc.csv');tab=tab[tab.representation==label].set_index('category')
 for c in CATS:
  y=[]
  for k in q.item:
   if label=='sbert_option_norm':k,i=k.rsplit(':',1);y.append(int(c in D[k]['taxonomy'].get(i,[])))
   else:y.append(int(c in labels(k)))
  check('auroc_'+label+'_'+c,abs(roc_auc_score(y,q[c])-tab.loc[c,'auroc'])<1e-12)
for label,file in [('clip_correct_option_norm','clip_correct_option_norm_oof.csv'),('matched_description_tfidf_lsa','matched_description_tfidf_lsa_oof.csv'),('matched_description_mpnet','matched_sbert_description_oof.csv')]:
 q=pd.read_csv(T/file);tab=pd.read_csv(T/'representation_matched_comparison.csv').query('representation==@label').set_index('category')
 check('same_visual_ids_'+label,set(q.item)==set(json.load(open(ROOT/'data/media_sample_ids.json'))))
 for c in CATS:check('auroc_'+label+'_'+c,abs(roc_auc_score([int(c in labels(k)) for k in q.item],q[c])-tab.loc[c,'auroc'])<1e-12)
# Independently check external row and label counts.
x=pd.read_csv(ROOT/'data/external/NormBank.csv');check('normbank_count',len(x)==155423);check('normbank_duplicates',int(x.duplicated().sum())==32059);check('normbank_unique_rows',len(x.drop_duplicates())==123364)
hf=pd.read_parquet(ROOT/'data/external/egonormia_hf.parquet');check('hf_membership',len(hf)==1743 and len(set(D)-set(hf.id))==110 and not(set(hf.id)-set(D)))
for s,n in [('high_agreement',934),('mid_agreement',1049)]:check('normlens_'+s,len(json.load(open(ROOT/'data/external'/(s+'.jsonl'))))==n)
# Enforce manuscript's rounded primary numerical claims.
m=(ROOT/'sections/analysis1_section.tex').read_text();expected_delta=['27.1','16.2','19.0'];tab=pd.read_csv(T/'paired.csv').query("metric=='joint' and subset=='three_way'")
check('manuscript_primary_deltas',[f'{v*100:.1f}' for v in tab.delta]==expected_delta and all(v in m for v in expected_delta))
check('manuscript_error_subset',all(x in m for x in ['742','740','473','489']))
for name in ['analysis1_section','methods_supplement','external_audits','qualitative_cases','idea1_intro','idea1_discussion','ai_disclosure']:
 s=(ROOT/'sections'/(name+'.tex')).read_text();check('no_unresolved_scaffolding_'+name,not re.search(r'\bTODO\b|AUTHORERR|TBD|\?\?\?',s))
json.dump(checks,open(ROOT/'qa/validation_results.json','w'),indent=2);print(f'{len(checks)} checks passed.')
