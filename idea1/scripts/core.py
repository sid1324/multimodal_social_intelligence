from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; T=ROOT/'outputs/tables'; F=ROOT/'outputs/figures'
D=json.load(open(ROOT/'data/egonormia/final_data.json')); E=json.load(open(ROOT/'data/egonormia/final_data_eval.json'))
CATS=['Safety','Privacy','Proxemics','Politeness','Cooperation','Coordination/Proactivity','Communication/Legibility']
FAMILIES={'Gemini-1.5-Flash':{'blind':'blind_gemini-15-flash-002','desc':'desc_gemini-15-flash-002','grid':'gemini-15-flash-002','frames':'frames_gemini-1.5-flash-002','video':'video_gemini-1.5-flash-002'},'Gemini-1.5-Pro':{'blind':'blind_gemini-15-pro-002','desc':'desc_gemini-15-pro-002','grid':'gemini-15-pro-002','frames':'frames_gemini-1.5-pro-002','video':'video_gemini-1.5-pro-002'},'GPT-4o':{'blind':'blind_gpt-4o-240513','desc':'desc_gpt-4o-240513','grid':'gpt-4o-240513','frames':'frames_gpt-4o'}}
def source(k):return k.split('_')[0]
def labels(k):return D[k]['taxonomy'].get(str(D[k]['correct']),[])
def score(k,key,metric='joint',stored=False):
 r=E[k][key]['best'];p=r.get('results');g=r.get('correct') if stored else [D[k]['correct']]*2
 if not isinstance(p,list) or len(p)!=2 or not isinstance(g,list) or len(g)!=2:return 0
 a=p[0]==g[0];j=p[1]==g[1]
 return int(a and j) if metric=='joint' else int(a if metric=='action' else j)
def bootstrap(ids,values,B=5000,seed=20261004):
 grp=pd.DataFrame({'g':[source(k) for k in ids],'v':np.asarray(values)}).groupby('g').v.agg(['sum','count'])
 rng=np.random.default_rng(seed);out=[]
 for start in range(0,B,250):
  ix=rng.integers(len(grp),size=(min(250,B-start),len(grp)));out.extend((grp['sum'].values[ix].sum(1)/grp['count'].values[ix].sum(1)).tolist())
 return np.percentile(out,[2.5,97.5])/1.
def paired(ids,x,y):
 from scipy.stats import binomtest
 x=np.asarray(x);y=np.asarray(y);d=y-x;gain=int(((x==0)&(y==1)).sum());loss=int(((x==1)&(y==0)).sum());lo,hi=bootstrap(ids,d)
 sums=pd.DataFrame({'g':[source(k) for k in ids],'d':d}).groupby('g').d.sum().values
 rng=np.random.default_rng(20261004);extreme=0;B=20000
 for _ in range(B//250):extreme+=int((abs((rng.choice([-1,1],size=(250,len(sums)))*sums).sum(1))>=abs(sums.sum())).sum())
 return dict(n=len(ids),n_clusters=len(sums),correct_from=int(x.sum()),correct_to=int(y.sum()),from_acc=x.mean(),to_acc=y.mean(),delta=d.mean(),ci_lo=lo,ci_hi=hi,gained=gain,lost=loss,mcnemar_p=binomtest(gain,gain+loss,.5).pvalue if gain+loss else 1.,cluster_swap_p=(extreme+1)/(B+1))
def option_rows(include_empty=True):
 return pd.DataFrame([dict(item=k,source=source(k),idx=i,b=b,j=v['justifications'][i],text=b+' '+v['justifications'][i],correct=int(i==v['correct'])) for k,v in D.items() for i,b in enumerate(v['behaviors']) if include_empty or b.strip()])
