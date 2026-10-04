from core import *
from folds import fixed_splits
import os,sys,torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel,CLIPProcessor
from PIL import Image,ImageOps
from contextlib import nullcontext

torch.set_num_threads(6)
MODE=sys.argv[1] if len(sys.argv)>1 else 'text';models=Path(os.environ.get('ANALYSIS_MODELS',str(ROOT/'models')));cache=ROOT/'outputs/features';cache.mkdir(exist_ok=True)
def embed(name,texts):
 f=cache/(name+'.npy')
 if f.exists():return np.load(f)
 model=SentenceTransformer(str(models/'all-mpnet-base-v2'),local_files_only=True,device='cpu')
 x=model.encode(list(texts),batch_size=32,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
 np.save(f,x);return x

def auc_cv(X,Y,groups,label,ids):
 pred=np.zeros_like(Y,dtype=float);foldcol=np.zeros(len(Y),int);foldrows=[]
 for fold,(tr,te) in enumerate(fixed_splits(ids,groups,label)):
  sc=StandardScaler().fit(X[tr]);a=sc.transform(X[tr]);b=sc.transform(X[te]);foldcol[te]=fold
  for c in range(len(CATS)):
   if len(set(Y[tr,c]))<2:pred[te,c]=Y[tr,c].mean()
   else:pred[te,c]=LogisticRegression(max_iter=3000,class_weight='balanced',C=1.).fit(a,Y[tr,c]).decision_function(b)
   if len(set(Y[te,c]))==2:foldrows.append(dict(representation=label,fold=fold,category=CATS[c],auroc=roc_auc_score(Y[te,c],pred[te,c]),n_test=len(te)))
 out=[dict(representation=label,category=c,n=len(Y),positive=int(Y[:,i].sum()),auroc=roc_auc_score(Y[:,i],pred[:,i]),mean_fold_auroc=np.mean([r['auroc'] for r in foldrows if r['category']==c])) for i,c in enumerate(CATS)]
 pd.DataFrame({'item':ids,'fold':foldcol,**{c:pred[:,i] for i,c in enumerate(CATS)}}).to_csv(T/(label+'_oof.csv'),index=False)
 pd.DataFrame(foldrows).to_csv(T/(label+'_fold_auroc.csv'),index=False)
 return out
if MODE=='text':
 O=option_rows(True);X=embed('mpnet_options',O.text)
 np.savez_compressed(cache/'mpnet_option_ids.npz',item=O.item.to_numpy(dtype=str),idx=O.idx.values)
 # Frozen pretrained features; all fitted scaling and classifiers are fold-local.
 from importlib import import_module
 probes=import_module('02_probes') if False else None
 scores=np.zeros(len(O));folds=np.zeros(len(O),int)
 for fold,(tr,te) in enumerate(fixed_splits(O.item+':'+O.idx.astype(str),O.source,'sbert_answer_options')):
  sc=StandardScaler().fit(X[tr]);clf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),O.correct.iloc[tr]);scores[te]=clf.decision_function(sc.transform(X[te]));folds[te]=fold
 p=O.assign(score=scores,fold=folds);p=p.loc[p.groupby('item').score.idxmax()];p[['item','idx','correct','fold','score']].to_csv(T/'sbert_answer_predictions.csv',index=False);lo,hi=bootstrap(p.item,p.correct);pd.DataFrame([dict(model='all-mpnet-base-v2',n=len(p),correct=int(p.correct.sum()),accuracy=p.correct.mean(),ci_lo=lo,ci_hi=hi)]).to_csv(T/'sbert_answer_probe.csv',index=False)
 nonempty=O.b.str.strip().ne('');o=O[nonempty];xx=X[nonempty];Y=np.array([[int(c in D[k]['taxonomy'].get(str(i),[])) for c in CATS] for k,i in zip(o.item,o.idx)])
 res=auc_cv(xx,Y,o.source.values,'sbert_option_norm',o.item+':'+o.idx.astype(str))
 ids=[k for k,v in D.items() if isinstance(v['desc'],str) and v['desc'].strip()];Xd=embed('mpnet_descriptions',[D[k]['desc'] for k in ids]);(cache/'mpnet_description_ids.json').write_text(json.dumps(ids));Yd=np.array([[int(c in labels(k)) for c in CATS] for k in ids]);res+=auc_cv(Xd,Yd,[source(k) for k in ids],'sbert_description_norm',ids);pd.DataFrame(res).to_csv(T/'sbert_norm_auroc.csv',index=False)
 single=np.where(Y.sum(1)==1)[0];Z=TSNE(n_components=2,random_state=0,perplexity=40,init='pca',learning_rate='auto').fit_transform(xx[single]);labs=Y[single].argmax(1)
 fig,ax=plt.subplots(figsize=(8,5));
 for i,c in enumerate(CATS):ax.scatter(*Z[labs==i].T,s=7,alpha=.5,label=c)
 ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1));ax.set(xticks=[],yticks=[],title='Frozen MPNet option embeddings: single-label candidates');fig.tight_layout();fig.savefig(F/'sbert_tsne.png',dpi=200);fig.savefig(F/'sbert_tsne.pdf');plt.close(fig);pd.DataFrame({'item':o.iloc[single].item.values,'option':o.iloc[single].idx.values,'category':[CATS[i] for i in labs],'x':Z[:,0],'y':Z[:,1]}).to_csv(T/'sbert_tsne_points.csv',index=False)
 print('TEXT DONE',flush=True)
else:
 f=cache/'clip_images.npy'
 if f.exists():ids=json.load(open(cache/'clip_image_ids.json'))
 else:
  ids=json.load(open(ROOT/'data/media_sample_ids.json'));assert all((ROOT/'data/media'/f'{k}.jpg').exists() for k in ids), 'Run download_sources.py --media first.'
  ids=[k for k in ids if isinstance(D[k]['desc'],str) and D[k]['desc'].strip()]
 if f.exists():X=np.load(f)
 else:
  model=CLIPModel.from_pretrained(str(models/'clip-vit-base-patch32'),local_files_only=True).eval();processor=CLIPProcessor.from_pretrained(str(models/'clip-vit-base-patch32'),local_files_only=True);parts=[]
  for start in range(0,len(ids),16):
   ims=[Image.open(ROOT/'data/media'/f'{k}.jpg').convert('RGB') for k in ids[start:start+16]]
   # Five equal horizontal tiles; pad each complete tile to square to preserve all pixels.
   tiles=[ImageOps.pad(im.crop((i*im.width//5,0,(i+1)*im.width//5,im.height)),(224,224),color=(0,0,0)) for im in ims for i in range(5)]
   inputs=processor(images=tiles,return_tensors='pt',do_center_crop=False,do_resize=False)
   with torch.inference_mode():features=model.get_image_features(**inputs);features=features.pooler_output if hasattr(features,'pooler_output') else features;parts.append(features.numpy().reshape(len(ims),5,-1).mean(1))
   for im in ims:im.close()
   print('CLIP',start+len(ims),flush=True)
  X=np.concatenate(parts);X=X/np.linalg.norm(X,axis=1,keepdims=True);np.save(f,X)
 (cache/'clip_image_ids.json').write_text(json.dumps(ids));Y=np.array([[int(c in labels(k)) for c in CATS] for k in ids]);groups=[source(k) for k in ids];res=auc_cv(X,Y,groups,'clip_correct_option_norm',ids)
 # Matched scene-description features and targets: avoid comparing to candidate-option labels.
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.decomposition import TruncatedSVD
 texts=np.array([D[k]['desc'] for k in ids]);pred=np.zeros_like(Y,dtype=float);foldau=[]
 for fold,(tr,te) in enumerate(fixed_splits(ids,groups,'clip_correct_option_norm')):
  v=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True);a=v.fit_transform(texts[tr]);b=v.transform(texts[te]);sv=TruncatedSVD(n_components=min(128,len(tr)-1),random_state=0).fit(a);aa=sv.transform(a);bb=sv.transform(b)
  for c in range(7):
   pred[te,c]=LogisticRegression(max_iter=3000,class_weight='balanced').fit(aa,Y[tr,c]).decision_function(bb)
   if len(set(Y[te,c]))==2:foldau.append((c,roc_auc_score(Y[te,c],pred[te,c])))
 res += [dict(representation='matched_description_tfidf_lsa',category=c,n=len(ids),positive=int(Y[:,i].sum()),auroc=roc_auc_score(Y[:,i],pred[:,i]),mean_fold_auroc=np.mean([a for ci,a in foldau if ci==i])) for i,c in enumerate(CATS)]
 pd.DataFrame({'item':ids,**{c:pred[:,i] for i,c in enumerate(CATS)}}).to_csv(T/'matched_description_tfidf_lsa_oof.csv',index=False)
 pd.DataFrame(res).to_csv(T/'visual_matched_norm_auroc.csv',index=False)
 Z=TSNE(n_components=2,random_state=0,perplexity=30,init='pca',learning_rate='auto').fit_transform(X);f,axs=plt.subplots(2,4,figsize=(12,6))
 for i,c in enumerate(CATS):
  ax=axs.flat[i];ax.scatter(*Z[Y[:,i]==0].T,s=10,c='#c5cbd4',alpha=.5);ax.scatter(*Z[Y[:,i]==1].T,s=12,c='#126c83',alpha=.8);ax.set(title=c,xticks=[],yticks=[])
 axs.flat[7].axis('off');f.suptitle('CLIP mean frame embeddings: correct-action labels (overlapping categories)');f.tight_layout();f.savefig(F/'clip_tsne.png',dpi=200);f.savefig(F/'clip_tsne.pdf');pd.DataFrame({'item':ids,'x':Z[:,0],'y':Z[:,1],**{c:Y[:,i] for i,c in enumerate(CATS)}}).to_csv(T/'clip_tsne_points.csv',index=False);print('VISUAL DONE',flush=True)
