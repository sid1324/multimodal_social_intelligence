from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json,numpy as np
root=Path('/workspace/scratch/8eb17a085ce3');out=root/'output/analysis/Full_Sequence_Face_Tests';work=root/'tmp/full_sequence_tests'
manifest=json.loads((work/'source_pilot_manifest.json').read_text()); font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for r in manifest['records']:
 item=r['item']
 for method in ['omi']+(['sob_reference'] if item==969 else []):
  board=Image.new('RGB',(1120,960),'#f4f6f8');d=ImageDraw.Draw(board)
  for row,c in enumerate(['original','low','baseline','high']):
   d.text((8,34+row*225),c,fill='black',font=font)
   for f in range(1,6):
    p=work/f'item_{item}_frame_{f}.png' if c=='original' else out/f'item_{item}'/method/c/f'frame_{f}.png'
    im=Image.open(p);box=r['review_regions'][f-1]['review_rectangle_xyxy']
    if box:
     x0,y0,x1,y1=box;pad=max(7,int((x1-x0)*.12));box=(max(0,x0-pad),max(0,y0-pad),min(im.width,x1+pad),min(im.height,y1+pad));im=im.crop(box);scale=min(188/im.width,200/im.height);im=im.resize((int(im.width*scale),int(im.height*scale)),Image.Resampling.LANCZOS);board.paste(im,(125+(f-1)*198+(188-im.width)//2,45+row*225))
    else:d.text((125+(f-1)*198,110+row*225),'No target face',fill='black',font=font)
    if row==0:d.text((125+(f-1)*198,6),f'Frame {f}',fill='black',font=font)
  board.save(work/'qa'/f'item_{item}_{method}_crops.png')
print('Created 6 inspection grids')
