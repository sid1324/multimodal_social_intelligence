"""Build aligned modality grids and descriptions without answer-conditioned captioning."""
import json
import math
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load,generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'output/idea5-audit';OUT=ROOT/'output/idea5-expanded'

def location(x,y):
    return ('left' if x<1/3 else 'right' if x>2/3 else 'center')+'-'+('upper' if y<1/3 else 'lower' if y>2/3 else 'middle')

def create_grid(paths,destination):
    images=[]
    for p in paths:
        image=Image.open(p).convert('RGB');image.thumbnail((384,384),Image.Resampling.LANCZOS);images.append(image)
    grid=Image.new('RGB',(sum(i.width for i in images),max(i.height for i in images)));x=0
    for im in images:grid.paste(im,(x,0));x+=im.width
    grid.save(destination,quality=95)

def pose_caption(frames):
    lines=['Estimated pose map: yellow is retained body landmarks; magenta is hand landmarks. This is derived from RGB, not measured 3D geometry. Missing output does not prove absence.']
    for r in frames:
        bodies=[]
        for points in r['body_landmarks']:
            keep=lambda p:0<=p['x']<=1 and 0<=p['y']<=1 and p['visibility']>=.5 and p['presence']>=.5
            if sum(keep(p) for p in points)<8:continue
            shoulder=[points[j] for j in [11,12] if keep(points[j])];wrists=[location(points[j]['x'],points[j]['y']) for j in [15,16] if keep(points[j])]
            center=location(sum(p['x'] for p in shoulder)/len(shoulder),sum(p['y'] for p in shoulder)/len(shoulder)) if shoulder else 'shoulder position uncertain'
            bodies.append(f"body at {center}, retained wrists: {', '.join(wrists) or 'none'}")
        hands=[location(sum(p['x'] for p in hand)/len(hand),sum(p['y'] for p in hand)/len(hand)) for hand in r['hand_landmarks']]
        lines.append(f"Frame {r['sample_index']+1}: "+('; '.join(bodies[:2]) or 'no body meets retention rule')+f"; {len(hands)} predicted hand(s)"+(f" at {', '.join(hands[:3])}" if hands else '')+'.')
    return '\n'.join(lines)

def seg_caption(frames):
    lines=['Instance segmentation map: colored masks and labels are YOLOv8n-seg predictions over COCO classes; black means unsegmented, not empty space. Colors are not tracked identities. Labels can be wrong.']
    for r in frames:
        objects=sorted(r['instances'],key=lambda o:-o['mask_area_fraction'])[:3]
        descriptions=[f"{o['class']} at {location(*o['center_xy_normalized'])} (score {o['confidence']:.2f})" for o in objects]
        lines.append(f"Frame {r['sample_index']+1}: {r['instance_count']} predicted instance(s); "+('; '.join(descriptions) or 'no accepted masks')+'.')
    return '\n'.join(lines)

def depth_caption(frames):
    lines=['Depth map: estimated relative inverse depth; brighter suggests nearer and darker suggests farther within the image. Values are not meters, and camera-frame scales need not agree. No depth ground truth is available.']
    for r in frames:
        regions=r['normalized_depth_region_medians'];near=max(regions,key=regions.get);far=min(regions,key=regions.get)
        lines.append(f"Frame {r['sample_index']+1}: brightest broad region {near} ({regions[near]:.2f}); darkest broad region {far} ({regions[far]:.2f}); values summarize normalized image intensity, not physical distance.")
    return '\n'.join(lines)

def main():
    frames=json.loads((OLD/'frame_results.json').read_text());geo=json.loads((OUT/'geometry_results.json').read_text());samples=json.loads((OLD/'protocol.json').read_text())['sample']
    assert len(geo)==len(frames)==200
    folder=OUT/'grids';folder.mkdir(exist_ok=True)
    items=[]
    for item in samples:
        item_id=item['id'];ff=sorted([r for r in frames if r['id']==item_id],key=lambda r:r['sample_index']);gg=sorted([r for r in geo if r['id']==item_id],key=lambda r:r['sample_index'])
        image_paths={}
        for kind,paths in [('rgb',[OLD/r['paths']['rgb'] for r in ff]),('pose',[OLD/r['paths']['skeleton'] for r in ff]),('segmentation',[OUT/r['segmentation_path'] for r in gg]),('depth',[OUT/r['depth_path'] for r in gg])]:
            destination=folder/f'{item_id}_{kind}.jpg';create_grid(paths,destination);image_paths[kind]=str(destination.relative_to(OUT))
        items.append({'id':item_id,'source':item['source'],'images':image_paths,'descriptions':{'pose':pose_caption(ff),'segmentation':seg_caption(gg),'depth':depth_caption(gg)}})
    descriptions_path=OUT/'inputs.json';previous=json.loads(descriptions_path.read_text()) if descriptions_path.exists() else {'items':[]};cached={x['id']:x['descriptions'].get('rgb') for x in previous['items']}
    model_info=json.loads((OUT/'model.json').read_text())
    model_path=str(ROOT/model_info['local_path']);model,processor=load(model_path);config=load_config(model_path)
    caption_prompt='The image is a left-to-right chronological grid of five egocentric RGB frames. Describe only observable people, objects, spatial layout, and changes across the frames in at most 120 words. Be explicit when something is unclear. Do not infer intentions, dialogue, social norms, next actions, or hidden events. Do not solve any task or recommend behavior. There are no answer options. Write a neutral visual description.'
    for j,item in enumerate(items):
        if cached.get(item['id']):item['descriptions']['rgb']=cached[item['id']]
        else:
            mx.random.seed(42);prompt=apply_chat_template(processor,config,caption_prompt,num_images=1)
            output=generate(model,processor,prompt,image=[str(OUT/item['images']['rgb'])],max_tokens=200,temperature=0.,verbose=False)
            item['descriptions']['rgb']=output.text if hasattr(output,'text') else str(output)
            item['caption_stats']={'generation_tokens':getattr(output,'generation_tokens',None),'prompt_tokens':getattr(output,'prompt_tokens',None)}
        descriptions_path.write_text(json.dumps({'items':items,'rgb_caption_model':model_info['id'],'rgb_caption_revision':model_info['revision'],'rgb_caption_prompt':caption_prompt,'rgb_caption_generation':'Greedy, seed 42, max 200 tokens; model sees RGB only, no options, gold, taxonomy, dataset descriptions, or processed maps','processed_description_generation':'Deterministic extractor-output summaries, not model-generated normative explanations'},indent=2));mx.clear_cache();print(f'Captioned {j+1}/40 clips',flush=True)
    print('Expanded input grids and descriptions complete',flush=True)

if __name__=='__main__':main()
