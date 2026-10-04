from core import *
from importlib import import_module
import sys
# Reuse frozen feature matrices without reloading or running pretrained models.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from folds import fixed_splits
cache=ROOT/'outputs/features';ids=json.load(open(cache/'clip_image_ids.json'));allids=json.load(open(cache/'mpnet_description_ids.json'));Xall=np.load(cache/'mpnet_descriptions.npy');index={k:i for i,k in enumerate(allids)};X=Xall[[index[k] for k in ids]];Y=np.array([[int(c in labels(k)) for c in CATS] for k in ids]);pred=np.zeros_like(Y,dtype=float);fr=[]
for fold,(tr,te) in enumerate(fixed_splits(ids,[source(k) for k in ids],'clip_correct_option_norm')):
 sc=StandardScaler().fit(X[tr])
 for ci,c in enumerate(CATS):
  pred[te,ci]=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),Y[tr,ci]).decision_function(sc.transform(X[te]))
  if len(set(Y[te,ci]))==2:fr.append(dict(category=c,fold=fold,auroc=roc_auc_score(Y[te,ci],pred[te,ci])))
pd.DataFrame({'item':ids,**{c:pred[:,i] for i,c in enumerate(CATS)}}).to_csv(T/'matched_sbert_description_oof.csv',index=False)
res=[dict(representation='matched_description_mpnet',category=c,n=len(ids),positive=int(Y[:,i].sum()),auroc=roc_auc_score(Y[:,i],pred[:,i]),mean_fold_auroc=np.mean([r['auroc'] for r in fr if r['category']==c])) for i,c in enumerate(CATS)]
pd.concat([pd.read_csv(T/'visual_matched_norm_auroc.csv'),pd.DataFrame(res)]).to_csv(T/'representation_matched_comparison.csv',index=False)
print(pd.read_csv(T/'representation_matched_comparison.csv').groupby('representation')[['auroc','mean_fold_auroc']].mean())
