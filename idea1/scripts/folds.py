from core import *
from sklearn.model_selection import GroupKFold

def fixed_splits(ids,groups,name):
 """Persist the first realized GroupKFold assignment; replay it across platforms."""
 path=ROOT/'data/folds'/f'{name}.csv';path.parent.mkdir(exist_ok=True)
 ids=np.asarray(ids,dtype=str);groups=np.asarray(groups);assert len(set(ids))==len(ids)
 if path.exists():
  f=pd.read_csv(path).set_index('id');assert set(f.index)==set(ids);fold=f.loc[ids,'fold'].to_numpy()
 else:
  fold=np.empty(len(ids),int)
  for n,(_,te) in enumerate(GroupKFold(5).split(ids,groups=groups)):fold[te]=n
  pd.DataFrame({'id':ids,'fold':fold}).to_csv(path,index=False)
 for n in range(5):
  tr=np.where(fold!=n)[0];te=np.where(fold==n)[0];assert not(set(groups[tr])&set(groups[te]));yield tr,te
