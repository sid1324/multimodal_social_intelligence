from core import *
import collections,re,hashlib
P=ROOT/'data/external';hf=pd.read_parquet(P/'egonormia_hf.parquet');hs=set(hf.id);gs=set(D);shared=hs&gs
# Canonical conversion drops parquet's padded null taxonomy keys.
def clean(x):
 if isinstance(x,np.ndarray):return [clean(v) for v in x]
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if v is not None}
 if isinstance(x,np.generic):return x.item()
 return x
mapping={'behaviors':'behaviors','justifications':'justifications','correct_idx':'correct','sensible_idx':'sensibles','taxonomy':'taxonomy','description':'desc'}
diff=[]
for _,row in hf.iterrows():
 k=row.id
 if k not in D:continue
 for a,b in mapping.items():
  x=clean(row[a]);y=D[k][b]
  if x!=y:diff.append(dict(item=k,field=b,hf=json.dumps(x),github=json.dumps(y)))
parity={'hf_rows':len(hf),'hf_unique_ids':hf.id.nunique(),'github_rows':len(D),'shared':len(shared),'hf_only':len(hs-gs),'github_only':len(gs-hs),'field_differences':dict(collections.Counter(r['field'] for r in diff)),'note':'The observed row gap is explained by the explicit missing ID list; cause of release omission is not documented by this audit.'}
json.dump(parity,open(T/'hf_github_parity.json','w'),indent=2);pd.DataFrame(diff).to_csv(T/'hf_github_field_differences.csv',index=False);pd.DataFrame([{'item':k,'membership':'github_only'} for k in sorted(gs-hs)]+[{'item':k,'membership':'hf_only'} for k in sorted(hs-gs)]).to_csv(T/'hf_github_unshared_ids.csv',index=False)
x=pd.read_csv(P/'NormBank.csv');missing=x.isna().sum().to_dict();a={'rows':len(x),'columns':list(x.columns),'missing':missing,'exact_duplicate_rows':int(x.duplicated().sum()),'label_counts':x.label.value_counts().to_dict(),'norm_counts':x.norm.value_counts().to_dict(),'split_counts':x.split.value_counts().to_dict(),'unique_settings':x.setting.nunique(),'unique_behaviors':x.behavior.nunique(),'label_norm_cross_tab':pd.crosstab(x.label,x.norm).to_dict()}
key=['setting','behavior','constraints'];a['duplicate_context_keys']=int(x.duplicated(key).sum());a['context_keys_with_conflicting_labels']=int((x.groupby(key,dropna=False).label.nunique()>1).sum())
# Same behavior-context text across provided splits is a leakage risk to inspect.
splits={s:set(map(tuple,g[key].fillna('').values)) for s,g in x.groupby('split')};a['context_overlap_across_splits']={s+'__'+t:len(splits[s]&splits[t]) for i,s in enumerate(splits) for t in list(splits)[i+1:]}
by=x.groupby(['setting','behavior']).norm.nunique();a['setting_behavior_groups']=len(by);a['setting_behavior_groups_with_multiple_norms']=int((by>1).sum())
json.dump(a,open(T/'normbank_audit.json','w'),indent=2,default=int)
rows=[];aud={}
for split in ['high_agreement','mid_agreement']:
 z=json.load(open(P/(split+'.jsonl')))
 for r in z:
  cnt=collections.Counter(r['answer_judgment']);mx=max(cnt.values());winners=[k for k,v in cnt.items() if v==mx];rows.append(dict(split=split,id=r['question_id'],image=r['image'],text=r['text'],text_normalized=re.sub(r'\s+',' ',r['text'].strip().lower()),majority=winners[0] if len(winners)==1 else -1,tied_majority=len(winners)>1,votes=len(r['answer_judgment']),unanimous=len(cnt)==1,judgments=json.dumps(r['answer_judgment'])))
 f=pd.DataFrame([r for r in rows if r['split']==split]);aud[split]={'rows':len(f),'unique_ids':f.id.nunique(),'unique_images':f.image.nunique(),'unique_action_texts':f.text.nunique(),'exact_duplicate_image_text':int(f.duplicated(['image','text']).sum()),'majority_label_counts':f.majority.value_counts().to_dict(),'unanimous_rows':int(f.unanimous.sum()),'vote_counts':f.votes.value_counts().to_dict(),'tied_majorities':int(f.tied_majority.sum()),'missing_by_field':pd.DataFrame(z).isna().sum().to_dict()}
f=pd.DataFrame(rows);f.to_csv(T/'normlens_items.csv',index=False)
# Context association, not a paired before/after causal flip rate. Exclude tied image labels.
for scope,df in [('combined',f),('high_agreement',f[f.split=='high_agreement'])]:
 df=df[df.majority>=0];g=df.groupby('text_normalized');repeat={k:q for k,q in g if q.image.nunique()>1};den=sum(len(q) for q in repeat.values());deviations=sum(len(q)-q.majority.value_counts().max() for q in repeat.values());a={'repeated_text_groups_with_multiple_images':len(repeat),'rows_in_repeated_text_groups':den,'groups_with_different_image_conditioned_majorities':sum(q.majority.nunique()>1 for q in repeat.values()),'rows_not_equal_to_a_text_modal_label_min_count':int(deviations),'deviation_rate':deviations/den if den else None,'text_modal_tie_groups':sum((q.majority.value_counts()==q.majority.value_counts().max()).sum()>1 for q in repeat.values()),'interpretation':'Descriptive inconsistency with the within-dataset modal judgment for identical normalized action text; not comparison against separately annotated text-only gold.'};aud[scope+'_context_diagnostic']=a
json.dump(aud,open(T/'normlens_audit.json','w'),indent=2,default=int)
print(json.dumps(parity,indent=2));print(json.dumps(aud,indent=2,default=int))
