from core import *
from folds import fixed_splits
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from empath import Empath

def groups_for(O,strict):
 if not strict:return O.source.values
 parent={k:k for k in O.item.unique()}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def join(items):
  its=list(items)
  for a in its[1:]:parent[find(a)]=find(its[0])
 for _,g in O.groupby('source'):join(g.item.unique())
 # Empty None candidate is shared by every item and must not join all examples.
 for _,g in O[O.b.str.strip().ne('')].groupby(O.b.map(lambda t:re.sub('[^a-z ]','',t.lower()).strip())):join(g.item.unique())
 return O.item.map(find).values

def run(O,kind='tfidf',col='text',strict=False,shuffle=None,features=None):
 groups=groups_for(O,strict);y=O.correct.to_numpy().copy();scores=np.zeros(len(O));folds=np.zeros(len(O),int)
 if shuffle is not None:
  y[:]=0;rng=np.random.default_rng(shuffle)
  for _,g in O.groupby('item'):y[rng.choice(g.index)]=1
 space='five_options' if len(O)==len(D)*5 else 'legacy_four_nonempty'
 item_ids=O.item.drop_duplicates().to_numpy();item_groups=O.assign(grp=groups).groupby('item',sort=False).grp.first().reindex(item_ids).values
 row_folds=np.empty(len(O),int)
 for fold,(_,te_items) in enumerate(fixed_splits(item_ids,item_groups,space+('_strict' if strict else '_source'))):row_folds[O.item.isin(item_ids[te_items])]=fold
 for fold in range(5):
  tr=np.where(row_folds!=fold)[0];te=np.where(row_folds==fold)[0]
  assert not(set(O.item.iloc[tr])&set(O.item.iloc[te]));assert not(set(O.source.iloc[tr])&set(O.source.iloc[te]))
  if kind=='tfidf':
   v=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True);a=v.fit_transform(O[col].iloc[tr]);b=v.transform(O[col].iloc[te])
  else:
   sc=StandardScaler().fit(features[tr]);a=sc.transform(features[tr]);b=sc.transform(features[te])
  clf=LogisticRegression(max_iter=2000,class_weight='balanced',C=1.).fit(a,y[tr]);scores[te]=clf.decision_function(b);folds[te]=fold
 p=O.assign(score=scores,y=y,fold=folds,group=groups);p=p.loc[p.groupby('item').score.idxmax()].copy();p['prediction']=p.idx;p['outcome']=p.y
 lo,hi=bootstrap(p.item,p.outcome)
 return dict(n=len(p),correct=int(p.outcome.sum()),accuracy=p.outcome.mean(),ci_lo=lo,ci_hi=hi,groups=len(set(groups))),p

rows=[]
for full in [False,True]:
 O=option_rows(full);space='five_options' if full else 'legacy_four_nonempty';lens=np.column_stack([O.b.str.split().str.len(),O.j.str.split().str.len()])
 for name,kind,col,strict in [('tfidf_source','tfidf','text',False),('tfidf_strict','tfidf','text',True),('behavior','tfidf','b',False),('justification','tfidf','j',False),('length','numeric','text',False)]:
  res,p=run(O,kind,col,strict,features=lens);rows.append(dict(space=space,probe=name,**res));p[['item','idx','correct','outcome','fold','group','score']].to_csv(T/f'probe_items_{space}_{name}.csv',index=False);print(space,name,res,flush=True)
 perms=[]
 for seed in range(3):
  res,p=run(O,shuffle=seed);rows.append(dict(space=space,probe=f'permutation_seed_{seed}',**res));perms.append(res['accuracy'])
 if full:
  lex=Empath();cats=sorted(lex.cats);X=np.array([[(a or {}).get(c,0.) for c in cats] for a in (lex.analyze(t,normalize=True) for t in O.text)])
  np.savez_compressed(ROOT/'outputs/empath_features.npz',X=X,items=O.item.to_numpy(dtype=str),idx=O.idx.values,categories=cats)
  res,p=run(O,'numeric',features=X);rows.append(dict(space=space,probe='empath',**res))
 pd.DataFrame(rows).to_csv(T/'probes.csv',index=False)
# Robustness subgroups explicitly keyed to the full-task strict probe.
it=pd.read_csv(T/'matched_item_scores.csv');p=pd.read_csv(T/'probe_items_five_options_tfidf_strict.csv').set_index('item');out=[]
for fam in FAMILIES:
 f=it[it.family==fam]
 for name,val in [('probe_correct',1),('probe_wrong',0)]:
  q=f[f.item.map(p.outcome)==val];out.append(dict(family=fam,subset=name,**paired(q.item.tolist(),q.desc,q.grid)))
pd.DataFrame(out).to_csv(T/'artifact_robustness.csv',index=False)
# Legacy reproducibility, including the original 729-item strict-probe failure set.
p=pd.read_csv(T/'probe_items_legacy_four_nonempty_tfidf_strict.csv').set_index('item');out=[]
for fam in FAMILIES:
 f=it[(it.family==fam)&it.item.map(p.outcome).eq(0)];out.append(dict(family=fam,**paired(f.item.tolist(),f.desc,f.grid)))
pd.DataFrame(out).to_csv(T/'legacy_artifact_robustness.csv',index=False)
