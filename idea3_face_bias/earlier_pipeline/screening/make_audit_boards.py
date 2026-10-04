"""Create readable evidence boards for reviewing detector suggestions."""
import argparse,json,random
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).parent

def load_records():
 complete=ROOT/'screening_results.json'
 if complete.exists():return json.loads(complete.read_text())['records']
 p=ROOT/'screening_records.jsonl'
 by_id={r['id']:r for r in [json.loads(x) for x in p.read_text().splitlines()]}
 return list(by_id.values())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['candidates','no_region','review'],default='candidates');parser.add_argument('--count',type=int,default=24);args=parser.parse_args()
 rows=load_records()
 label={'candidates':'large_face_candidate','no_region':'no_region_detected','review':'possible_region_review'}[args.group]
 pool=sorted([r for r in rows if r['screening_label']==label],key=lambda r:r['id'])
 if args.group=='candidates':selected=pool
 else:selected=random.Random(1729).sample(pool,min(args.count,len(pool)))
 official=json.loads((ROOT/'data/official_annotations.json').read_text())
 item_index={cid:i+1 for i,cid in enumerate(sorted(official))}
 outdir=ROOT/'audit_boards';outdir.mkdir(exist_ok=True)
 mapping=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
 for start in range(0,len(selected),8):
  group=selected[start:start+8];board=Image.new('RGB',(1520,1720),'#edf2f6');draw=ImageDraw.Draw(board)
  draw.text((20,12),f'Automated screening audit: {args.group} | best suggested frame per item',font=font,fill='#123d59')
  for j,row in enumerate(group):
   x=20+(j%2)*750;y=50+(j//2)*415
   im=Image.open(ROOT/'model_previews'/(row['id']+'.jpg')).convert('RGB');fw=im.width//5
   options=[(face['box_xywh'][2]*face['box_xywh'][3],frame['frame'],face) for frame in row['frames'] for face in frame['faces'] if (face['large_frontal_candidate'] if args.group=='candidates' else True)]
   if options:
    _,frame_idx,face=max(options,key=lambda v:v[0]);bbox=face['box_xywh']
   else:
    frames_with_heads=[f for f in row['frames'] if any(h['plausible_proxy'] for h in f['head_proxies'])]
    frame_idx=frames_with_heads[0]['frame'] if frames_with_heads else 3;bbox=None
   frame=im.crop(((frame_idx-1)*fw,0,frame_idx*fw,im.height));original=frame.copy()
   draw.text((x,y),f"Item {item_index[row['id']]:04d} | frame {frame_idx} | {row['id'][:8]}",font=font,fill='#123d59')
   if bbox:
    xx,yy,ww,hh=bbox;ImageDraw.Draw(frame).rectangle((xx,yy,xx+ww,yy+hh),outline='#00ff70',width=3)
    pad=.12*max(ww,hh);crop=original.crop((max(0,int(xx-pad)),max(0,int(yy-pad)),min(fw,int(xx+ww+pad)),min(im.height,int(yy+hh+pad))))
    crop.thumbnail((180,210));board.paste(crop,(x+525,y+40))
    draw.text((x+525,y+255),'Detector face crop',font=small,fill='#263445')
   frame.thumbnail((510,340));board.paste(frame,(x,y+35))
   norms=', '.join(row.get('correct_action_norms',[]))
   draw.text((x,y+380),norms[:92],font=small,fill='#263445')
   mapping.append({'item_index':item_index[row['id']],'id':row['id'],'screening_label':row['screening_label'],'displayed_frame':frame_idx,'board':f'{args.group}_{start//8+1:02d}.png','panel':j+1})
  path=outdir/f'{args.group}_{start//8+1:02d}.png';part=path.with_suffix('.part');board.save(part,format='PNG');part.replace(path)
 (outdir/f'{args.group}_mapping.json').write_text(json.dumps(mapping,indent=2))
 print(json.dumps({'group':args.group,'reviewed_pool_size':len(pool),'board_items':len(mapping),'boards':(len(mapping)+7)//8}))

if __name__=='__main__':main()
