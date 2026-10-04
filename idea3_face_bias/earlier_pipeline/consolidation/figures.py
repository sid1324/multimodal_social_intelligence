from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path('/workspace/scratch/8eb17a085ce3')
OUT=ROOT/'output/analysis/Milestone2_Consolidation'
d=json.loads((OUT/'consolidated_results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.titleweight':'bold'})
blue='#2469a0';amber='#ca8b31';grey='#a3adba';ink='#172435'
fig=plt.figure(figsize=(9.4,5.4),facecolor='white')
gs=fig.add_gridspec(2,2,width_ratios=[1.05,1.4],hspace=.68,wspace=.73)
ax=fig.add_subplot(gs[0,0])
labels=['Promising','Needs edit test','No region identified'];values=[49,51,17]
ax.barh(np.arange(3),values,color=[blue,amber,grey],height=.62)
ax.set_yticks(np.arange(3),labels);ax.invert_yaxis();ax.set_xlim(0,60)
for i,v in enumerate(values):ax.text(v+1,i,str(v),va='center',color=ink,fontweight='bold')
ax.set_xlabel('Examples');ax.set_title('A  Five-frame AI audit (n = 117)',loc='left',fontsize=10,pad=10)
ax.tick_params(axis='y',length=0);ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15)
ax=fig.add_subplot(gs[1,0])
items=[605,1625,749,969,1250]
for i,item in enumerate(items):
 rows=[r for r in d['pilot_region_measurements'] if r['item']==item];xs=[r['region_area_percent'] for r in rows]
 ys=i+np.linspace(-.12,.12,len(xs));ax.scatter(xs,ys,s=25,c=blue,alpha=.7,zorder=3)
 med=np.median(xs);ax.plot([med,med],[i-.22,i+.22],color=ink,lw=2)
ax.set_yticks(range(5),[str(i) for i in items]);ax.set_ylabel('Pilot item');ax.invert_yaxis();ax.set_xlim(0,23);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('Review rectangle / frame area (%)')
ax.set_title('C  Pilot region size (23 frames)',loc='left',fontsize=10,pad=10);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax=fig.add_subplot(gs[:,1])
rows=d['norm_coverage'][:7];y=np.arange(len(rows));h=.30
ax.barh(y-h/2,[r['pool_percent'] for r in rows],h,color=blue,label='Potential pool (n = 100)')
ax.barh(y+h/2,[r['full_percent'] for r in rows],h,color=grey,label='Full snapshot (n = 1,853)')
display={'Communication/Legibility':'Communication /\nLegibility','Coordination/Proactivity':'Coordination /\nProactivity'}
ax.set_yticks(y,[display.get(r['norm'],r['norm']) for r in rows]);ax.invert_yaxis();ax.set_xlim(0,69);ax.set_xticks([0,20,40,60]);ax.set_xlabel('Examples with correct-option norm (%)')
for i,r in enumerate(rows):
 ax.text(r['pool_percent']+.8,i-h/2,f"{r['pool_percent']:.0f}%",va='center',fontsize=8)
 ax.text(r['full_percent']+.8,i+h/2,f"{r['full_percent']:.1f}%",va='center',fontsize=8,color='#596578')
ax.set_title('B  Descriptive norm coverage',loc='left',fontsize=10,pad=38);ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.01),frameon=False,fontsize=8)
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
fig.subplots_adjust(left=.155,right=.97,top=.82,bottom=.22)
fig.text(.02,.035,'Purposive review: 1,736 examples remain unverified. Norm labels overlap.\nRegion sizes describe AI planning rectangles in five pilots, not segmented faces or dataset-wide sizes.',fontsize=8,color='#536273')
for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'Figure1_Original_Data_Feasibility.{ext}',dpi=300)
plt.close(fig)

fig,axes=plt.subplots(3,4,figsize=(7.4,5.7),facecolor='white')
selection=[(605,3,'Pole assembly: small region'),(1625,2,'Checkout: forehead seam'),(749,3,'Tubing: expression drift')]
allrows=d['pilot_region_measurements'];conditions=['original','low','baseline','high']
for row,(item,f,reason) in enumerate(selection):
 rr=next(r for r in allrows if r['item']==item and r['frame']==f);x0,y0,x1,y1=rr['review_rectangle_xyxy'];pad=max(7,int((x1-x0)*.12))
 for col,condition in enumerate(conditions):
  im=Image.open(OUT/'evidence'/f'item_{item}_frame_{f}_{condition}.png');box=(max(0,x0-pad),max(0,y0-pad),min(im.width,x1+pad),min(im.height,y1+pad))
  ax=axes[row,col];ax.imshow(im.crop(box));ax.set_xticks([]);ax.set_yticks([])
  for spine in ax.spines.values():spine.set_visible(False)
  if row==0:ax.set_title(['Original','Low reference','Baseline reference','High reference'][col],fontsize=10,pad=10)
  if col==0:ax.set_ylabel(f'{item} / frame {f}',fontsize=9,labelpad=10)
 axes[row,0].set_xlabel(reason,fontsize=8,color='#536273',labelpad=6)
fig.subplots_adjust(left=.1,right=.985,top=.88,bottom=.16,hspace=.46,wspace=.12)
fig.text(.025,.03,'Illustrative enlarged crops of final registered outputs; no recovered detail.\nOutside-mask preservation passed, while accessory, gaze and expression control remain unvalidated.',fontsize=8,color='#536273')
for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'Figure2_Pilot_Construction_and_Limitations.{ext}',dpi=300)
plt.close(fig)
print('Created two figures in PNG, PDF and SVG formats')
