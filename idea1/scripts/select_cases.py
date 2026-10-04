from core import *
import hashlib
from PIL import Image
media=set(json.load(open(ROOT/'data/media_sample_ids.json')));f=pd.read_csv(T/'matched_item_scores.csv');f=f[(f.family=='Gemini-1.5-Flash')&f.item.isin(media)];out=[]
for name,mask in [('visual_rescue',(f.desc==0)&(f.grid==1)),('persistent_failure',(f.desc==0)&(f.grid==0))]:
 ids=sorted(f[mask].item,key=lambda k:hashlib.sha256(('qualitative-20261004:'+k).encode()).hexdigest());k=ids[0];v=D[k]
 out.append({'case':name,'item':k,'eligible_candidates':len(ids),'gold':v['correct'],'desc_prediction':E[k]['desc_gemini-15-flash-002']['best']['results'],'grid_prediction':E[k]['gemini-15-flash-002']['best']['results'],**v})
 im=Image.open(ROOT/'data/media'/(k+'.jpg'));thumb=im.copy();thumb.thumbnail((1600,1600));thumb.save(ROOT/'qualitative'/(name+'.jpg'))
 contact=Image.new('RGB',(1600,1350),'white')
 for i in range(5):
  tile=im.crop((i*im.width//5,0,(i+1)*im.width//5,im.height));tile.thumbnail((800,450));contact.paste(tile,((i%2)*800,(i//2)*450))
 contact.save(ROOT/'qualitative'/(name+'_contact.png'))
json.dump(out,open(ROOT/'qualitative/cases.json','w'),indent=2)
