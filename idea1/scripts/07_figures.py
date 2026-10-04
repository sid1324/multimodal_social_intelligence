from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
lad=pd.read_csv(T/'ladder.csv');norm=pd.read_csv(T/'norm_pooled.csv');colors=['#246b83','#d27828','#7655a0'];short=['Gemini Flash','Gemini Pro','GPT-4o']
fig,axs=plt.subplots(1,2,figsize=(9.2,3.2),gridspec_kw={'width_ratios':[1,1.25]})
for i,fam in enumerate(FAMILIES):
 f=lad[(lad.family==fam)&(lad.metric=='joint')].set_index('condition').loc[['blind','desc','grid']];a=f.accuracy.to_numpy()*100;axs[0].errorbar(np.arange(3)+(i-1)*.06,a,yerr=np.vstack([a-f.ci_lo.to_numpy()*100,f.ci_hi.to_numpy()*100-a]),fmt='o-',color=colors[i],lw=1.7,capsize=2,label=f'{short[i]} (n={int(f.n.iloc[0]):,})')
axs[0].set(xticks=range(3),xticklabels=['Options','Description','Frame grid'],ylabel='Joint accuracy (%)',ylim=(0,57),title='A  Same model, same items');axs[0].legend(fontsize=7,loc='upper left');axs[0].grid(axis='y',alpha=.16)
norm=norm.sort_values('delta');ys=np.arange(len(norm));axs[1].barh(ys,norm.delta*100,color='#246b83',height=.6);axs[1].errorbar(norm.delta*100,ys,xerr=np.vstack([(norm.delta-norm.ci_lo)*100,(norm.ci_hi-norm.delta)*100]),fmt='none',ecolor='#172d3a',capsize=3)
axs[1].set(yticks=ys,yticklabels=[c.split('/')[0] for c in norm.category],xlabel='Grid minus description (percentage points)',title='B  Correct-action categories',xlim=(0,35));axs[1].tick_params(axis='y',labelsize=8);axs[1].grid(axis='x',alpha=.16)
fig.tight_layout(w_pad=2);fig.savefig(F/'grounding_ladder.pdf',bbox_inches='tight');fig.savefig(F/'grounding_ladder.png',dpi=220,bbox_inches='tight');plt.close(fig)
# A compact qualitative figure shows frames 1 and 5, selected independently of visual appeal.
cases=json.load(open(ROOT/'qualitative/cases.json'));fig,axs=plt.subplots(2,2,figsize=(9.2,5.3))
for r,c in enumerate(cases):
 im=Image.open(ROOT/'data/media'/(c['item']+'.jpg'))
 for col,i in enumerate([0,4] if r==0 else [1,2]):
  ax=axs[r,col];ax.imshow(im.crop((i*im.width//5,0,(i+1)*im.width//5,im.height)));ax.axis('off');ax.set_title(('A  Visual rescue' if r==0 else 'B  Action fixed, justification wrong')+f' | frame {i+1}',fontsize=10,loc='left')
fig.tight_layout();fig.savefig(F/'qualitative_cases.pdf',bbox_inches='tight');fig.savefig(F/'qualitative_cases.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Input format diagnostic, direction is richer-format minus grid.
a=pd.read_csv(T/'format_ablation.csv');fig,ax=plt.subplots(figsize=(7.2,3));y=np.arange(len(a));ax.errorbar(a.delta*100,y,xerr=np.vstack([(a.delta-a.ci_lo)*100,(a.ci_hi-a.delta)*100]),fmt='o',capsize=4,color='#246b83');ax.axvline(0,color='gray',lw=1);ax.set(yticks=y,yticklabels=[f'{r.family}: {r.condition} (n={r.n})' for r in a.itertuples()],xlabel='Input format minus frame grid (percentage points)',title='Released format ablations: joint accuracy');fig.tight_layout();fig.savefig(F/'format_ablation.pdf');fig.savefig(F/'format_ablation.png',dpi=200);plt.close(fig)
# External data facts are count plots, not a transfer-performance comparison.
b=json.load(open(T/'normbank_audit.json'));n=json.load(open(T/'normlens_audit.json'));fig,axs=plt.subplots(1,2,figsize=(8,3.3));axs[0].bar(list(b['norm_counts']),list(b['norm_counts'].values()),color=colors);axs[0].set(title='NormBank label counts',ylabel='Rows');axs[0].ticklabel_format(axis='y',style='plain')
labels=['0','1','2','-1'];width=.36;x=np.arange(4)
for i,sp in enumerate(['high_agreement','mid_agreement']):axs[1].bar(x+(i-.5)*width,[n[sp]['majority_label_counts'].get(k,0) for k in labels],width,label=sp.replace('_',' '),color=colors[i])
axs[1].set(xticks=x,xticklabels=['Inappropriate','Appropriate','Impossible','Tie'],title='NormLens majority judgments',ylabel='Examples');axs[1].tick_params(axis='x',labelsize=7);axs[1].legend(fontsize=7);fig.tight_layout();fig.savefig(F/'external_data_audit.pdf');fig.savefig(F/'external_data_audit.png',dpi=200)
