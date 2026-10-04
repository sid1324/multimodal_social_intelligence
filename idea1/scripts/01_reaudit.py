from core import *
from scipy.stats import chisquare
import collections,hashlib
ids=list(D);gold=[v['correct'] for v in D.values()];ver=set(json.load(open(ROOT/'data/egonormia/verified_split.json'))['split'])
unknown=collections.Counter(l for v in D.values() for labs in v['taxonomy'].values() for l in labs if l not in CATS)
changed=set();malformed=[]
for k,v in E.items():
 for key,r in v.items():
  g=r['best'].get('correct');p=r['best'].get('results')
  if isinstance(g,list) and None not in g and g!=[D[k]['correct']]*2:changed.add(k)
  if not isinstance(p,list) or len(p)!=2:malformed.append([k,key,p])
position=np.bincount(gold,minlength=5);chi=chisquare(position)
nonempty_rank=[sum(bool(b.strip()) for b in v['behaviors'][:v['correct']]) for v in D.values() if v['behaviors'][v['correct']].strip()]
rank=np.bincount(nonempty_rank,minlength=4);rch=chisquare(rank)
audit={'items':len(D),'source_id_prefixes':len({source(k) for k in D}),'verified':len(ver),'options_per_item':dict(collections.Counter(len(v['behaviors']) for v in D.values())),'empty_gold_items':sum(not v['behaviors'][v['correct']].strip() for v in D.values()),'missing_description_ids':[k for k,v in D.items() if not isinstance(v['desc'],str) or not v['desc'].strip()],'gold_revision_items':len(changed),'unexpected_label_occurrences':dict(unknown),'position_counts_0based':position.tolist(),'position_uniform_five_chi2':float(chi.statistic),'position_uniform_five_p':float(chi.pvalue),'nonempty_gold_rank_counts_0based':rank.tolist(),'nonempty_gold_rank_chi2':float(rch.statistic),'nonempty_gold_rank_p':float(rch.pvalue),'uniform_five_action_chance':.2,'uniform_nonempty_action_chance_all_items':(len(D)-21)/len(D)/4,'malformed_result_shapes':malformed,'no_correct_option_labels':sum(not labels(k) for k in D)}
json.dump(audit,open(T/'audit.json','w'),indent=2)
pd.DataFrame([{'item':k,'source':source(k),'gold':v['correct'],'none_gold':not v['behaviors'][v['correct']].strip(),'verified':k in ver,'labels':'|'.join(labels(k))} for k,v in D.items()]).to_csv(T/'items.csv',index=False)
pd.DataFrame([{'category':c,'full_correct':sum(c in labels(k) for k in D),'verified_correct':sum(c in labels(k) for k in ver)} for c in CATS]).to_csv(T/'label_counts.csv',index=False)
pd.DataFrame({'item':sorted(changed)}).to_csv(T/'gold_revised_ids.csv',index=False)
rows=[];lad=[];ab=[];sens=[];item=[];norm=[]
for fam,conds in FAMILIES.items():
 common=sorted(k for k in D if all(conds[c] in E.get(k,{}) for c in ['blind','desc','grid']))
 for k in common:item.append({'family':fam,'item':k,'source':source(k),**{c:score(k,conds[c]) for c in ['blind','desc','grid']}})
 for metric in ['joint','action','justification']:
  for c in ['blind','desc','grid']:
   a=np.array([score(k,conds[c],metric) for k in common]);lo,hi=bootstrap(common,a)
   lad.append(dict(family=fam,metric=metric,condition=c,n=len(common),correct=int(a.sum()),accuracy=a.mean(),ci_lo=lo,ci_hi=hi))
  for subset,sids in [('three_way',common),('verified',[k for k in common if k in ver])]:
   r=paired(sids,[score(k,conds['desc'],metric) for k in sids],[score(k,conds['grid'],metric) for k in sids]);rows.append(dict(family=fam,metric=metric,subset=subset,**r))
 for c in conds:
  ks=[k for k in D if conds[c] in E.get(k,{})];sens.append(dict(family=fam,condition=c,n=len(ks),current=sum(score(k,conds[c]) for k in ks)/len(ks),stored=sum(score(k,conds[c],stored=True) for k in ks)/len(ks)))
 for c in ['frames','video']:
  if c in conds:
   ks=[k for k in D if all(conds[q] in E.get(k,{}) for q in ['grid',c])];ab.append(dict(family=fam,condition=c,**paired(ks,[score(k,conds['grid']) for k in ks],[score(k,conds[c]) for k in ks])))
 for c in CATS:
  ks=[k for k in common if c in labels(k)];norm.append(dict(family=fam,category=c,**paired(ks,[score(k,conds['desc']) for k in ks],[score(k,conds['grid']) for k in ks])))
for name,v in [('ladder',lad),('paired',rows),('format_ablation',ab),('gold_sensitivity',sens),('matched_item_scores',item),('norm_by_family',norm)]:pd.DataFrame(v).to_csv(T/(name+'.csv'),index=False)
# Joint source bootstrap for an equally weighted mean over families, preserving overlap.
it=pd.DataFrame(item);vids=sorted(set(it.source));vi={g:i for i,g in enumerate(vids)};rng=np.random.default_rng(20261004);weights=rng.multinomial(len(vids),np.repeat(1/len(vids),len(vids)),size=5000)
nd={};out=[]
for cat in CATS:
 vals=[];points=[];ns=[]
 for fam in FAMILIES:
  f=it[(it.family==fam)&it.item.map(lambda k:cat in labels(k))];index=f.source.map(vi).values;w=weights[:,index];d=(f.grid-f.desc).values;vals.append((w@d)/w.sum(1));points.append(d.mean());ns.append(len(f))
 boot=np.mean(vals,axis=0);nd[cat]=boot;lo,hi=np.percentile(boot,[2.5,97.5]);out.append(dict(category=cat,delta=np.mean(points),ci_lo=lo,ci_hi=hi,n_flash=ns[0],n_pro=ns[1],n_gpt=ns[2]))
pd.DataFrame(out).to_csv(T/'norm_pooled.csv',index=False)
contrast=[]
for c in ['Privacy','Communication/Legibility']:
 a='Coordination/Proactivity';diff=nd[a]-nd[c];lo,hi=np.percentile(diff,[2.5,97.5]);point=next(r['delta'] for r in out if r['category']==a)-next(r['delta'] for r in out if r['category']==c);contrast.append(dict(a=a,b=c,delta_difference=point,ci_lo=lo,ci_hi=hi))
pd.DataFrame(contrast).to_csv(T/'norm_contrasts_exploratory.csv',index=False)
print(json.dumps(audit,indent=2));print(pd.DataFrame(rows).to_string(index=False))
