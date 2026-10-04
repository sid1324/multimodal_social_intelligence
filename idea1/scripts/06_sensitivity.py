from core import *
from scipy.stats import binomtest
items=pd.read_csv(T/'matched_item_scores.csv');shared=set.intersection(*[set(items[items.family==f].item) for f in FAMILIES]);out=[]
for fam in FAMILIES:
 x=items[(items.family==fam)&items.item.isin(shared)].sort_values('item');out.append(dict(family=fam,**paired(x.item.tolist(),x.desc,x.grid)))
pd.DataFrame(out).to_csv(T/'all_families_shared_population.csv',index=False)
# Independent direct scorer: asserts every reported condition numerator matches raw JSON.
checks=[]
for _,r in pd.read_csv(T/'ladder.csv').iterrows():
 fam=r.family;keys=FAMILIES[fam];ids=sorted(k for k in D if all(keys[c] in E.get(k,{}) for c in ['blind','desc','grid']));c=0
 for k in ids:
  pred=E[k][keys[r.condition]]['best']['results'];gold=D[k]['correct'];c+=int(pred==[gold,gold]) if r.metric=='joint' else int(pred[0 if r.metric=='action' else 1]==gold)
 assert c==r.correct and len(ids)==r.n and abs(c/len(ids)-r.accuracy)<1e-12
 checks.append({'check':f'raw_numerator_{fam}_{r.metric}_{r.condition}','status':'PASS','numerator':c,'denominator':len(ids)})
# None-candidate handling and out-of-fold results integrity.
for name in ['source','strict']:
 p=pd.read_csv(T/f'probe_items_five_options_tfidf_{name}.csv');assert len(p)==len(D) and p.item.nunique()==len(D)
 for r in p.itertuples():assert r.outcome==int(r.idx==D[r.item]['correct'])
 assert p.groupby(p.item.map(source)).fold.nunique().max()==1
 checks.append({'check':f'five_choice_{name}_OOF_integrity','status':'PASS'})
# The identical normalized behavior can span sources only within a strict split.
p=pd.read_csv(T/'probe_items_five_options_tfidf_strict.csv').set_index('item');texts={}
import re
for k,v in D.items():
 for b in v['behaviors']:
  if b.strip():texts.setdefault(re.sub('[^a-z ]','',b.lower()).strip(),set()).add(k)
assert all(len({p.loc[k,'fold'] for k in ks})==1 for ks in texts.values())
checks.append({'check':'strict_shared_behavior_fold_integrity','status':'PASS'})
json.dump(checks,open(ROOT/'qa/numerical_assertions.json','w'),indent=2)
print('Checks passed',len(checks));print(pd.DataFrame(out).to_string(index=False))
