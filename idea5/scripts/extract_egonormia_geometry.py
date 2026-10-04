"""Extract instance masks and relative monocular depth on the existing 200 frames."""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]; OLD=ROOT/'output/idea5-audit'; OUT=ROOT/'output/idea5-expanded'
DEPTH_ID='depth-anything/Depth-Anything-V2-Small-hf'

def position(x,y):
    return ('left' if x<1/3 else 'right' if x>2/3 else 'center')+'-'+('upper' if y<1/3 else 'lower' if y>2/3 else 'middle')

def main():
    OUT.mkdir(exist_ok=True)
    frames=json.loads((OLD/'frame_results.json').read_text()); sample=json.loads((OLD/'protocol.json').read_text())['sample']
    path=OUT/'geometry_protocol.json'
    revision=json.loads(path.read_text())['depth_revision'] if path.exists() else HfApi().model_info(DEPTH_ID).sha
    model_path=snapshot_download(DEPTH_ID,revision=revision,local_dir=str(ROOT/'models/Depth-Anything-V2-Small'))
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    processor=AutoImageProcessor.from_pretrained(model_path)
    depth_model=AutoModelForDepthEstimation.from_pretrained(model_path).to(device).eval()
    seg_path=ROOT/'models/yolov8n-seg.pt'
    segmenter=YOLO(str(seg_path))
    protocol={'depth_model':DEPTH_ID,'depth_revision':revision,'depth_type':'Estimated relative inverse depth from RGB; not sensor depth or meters. Larger output is rendered brighter (closer). Network scales can vary between frames. Common 2nd/98th percentile normalization across five frames per clip; clipping outside that range.', 'segmentation_model':'Ultralytics YOLOv8n-seg, COCO-pretrained','segmentation_sha256':hashlib.sha256(seg_path.read_bytes()).hexdigest(),'segmentation_confidence':0.25,'segmentation_iou':0.7,'segmentation_input_size':640,'segmentation_max_detections':40,'segmentation_retina_masks':True,'device':device,'sample':'Same 40 source-disjoint items and five timestamps as original pose audit; no outcome-based filtering','segmentation_render':'Mask-only RGB image on black background, class labels on predicted instance masks; colors identify per-frame instances, not tracked identities','text_descriptions':'RGB captions generated independently without answer options. Processed-image descriptions summarize extractor outputs and their limits; no correct labels or norm categories used.'}
    path.write_text(json.dumps(protocol,indent=2))
    results=[]; started=time.time()
    for si,item in enumerate(sample):
        item_frames=sorted([r for r in frames if r['id']==item['id']],key=lambda r:r['sample_index'])
        folder=OUT/'frames'/item['id'];folder.mkdir(parents=True,exist_ok=True)
        clip_depth=[]; clip_records=[]
        for r in item_frames:
            image=Image.open(OLD/r['paths']['rgb']).convert('RGB');w,h=image.size
            inputs=processor(images=image,return_tensors='pt');inputs={k:v.to(device) for k,v in inputs.items()}
            with torch.inference_mode():
                pred=depth_model(**inputs).predicted_depth
                native=pred[0].detach().cpu().float().numpy()
                depth=torch.nn.functional.interpolate(pred.unsqueeze(1),size=(h,w),mode='bicubic',align_corners=False)[0,0].detach().cpu().float().numpy()
            if not np.isfinite(depth).all():raise ValueError('Non-finite depth prediction')
            np.save(folder/f"{r['sample_index']:02d}_depth_raw.npy",native)
            detections=segmenter.predict(image,conf=.25,iou=.7,imgsz=640,max_det=40,retina_masks=True,device=device,verbose=False)[0]
            canvas=Image.new('RGB',(w,h),'black');draw=ImageDraw.Draw(canvas);ids=np.zeros((h,w),dtype=np.uint16);objects=[]
            if detections.masks is not None:
                masks=detections.masks.data.cpu().numpy(); boxes=detections.boxes.xyxy.cpu().numpy();classes=detections.boxes.cls.cpu().numpy().astype(int);scores=detections.boxes.conf.cpu().numpy()
                # Paint large masks first so smaller detected instances remain visible.
                order=sorted(range(len(masks)),key=lambda k:-float(masks[k].sum()))
                for k in order:
                    mask=masks[k]>.5
                    if mask.shape!=(h,w):mask=np.asarray(Image.fromarray(mask.astype('uint8')*255).resize((w,h),Image.Resampling.NEAREST))>0
                    rng=np.random.default_rng(int(classes[k])*1009+k+7);color=tuple(int(x) for x in rng.integers(70,256,3));bitmap=Image.fromarray(mask.astype('uint8')*255)
                    canvas.paste(Image.new('RGB',(w,h),color),mask=bitmap);ids[mask]=k+1
                    x1,y1,x2,y2=boxes[k];label=detections.names[int(classes[k])]
                    objects.append({'instance_id':k+1,'class':label,'confidence':float(scores[k]),'bbox_xyxy':[float(x) for x in boxes[k]],'center_xy_normalized':[float((x1+x2)/2/w),float((y1+y2)/2/h)],'mask_area_fraction':float(mask.mean()),'median_relative_inverse_depth':float(np.median(depth[mask])) if mask.any() else None})
                draw=ImageDraw.Draw(canvas)
                for obj in objects:
                    x1,y1,_,_=obj['bbox_xyxy'];draw.text((max(0,x1),max(0,y1)),f"{obj['instance_id']}:{obj['class']}",fill='white',stroke_width=1,stroke_fill='black')
            canvas.save(folder/f"{r['sample_index']:02d}_segmentation.png")
            Image.fromarray(ids).save(folder/f"{r['sample_index']:02d}_instance_ids.png")
            rec={'id':r['id'],'source':r['source'],'sample_index':r['sample_index'],'timestamp_seconds':r['timestamp_seconds'],'width':w,'height':h,'instances':objects,'instance_count':len(objects),'segmentation_mask_area_fraction':float((ids>0).mean()),'depth_min':float(depth.min()),'depth_max':float(depth.max()),'depth_raw_path':str((folder/f"{r['sample_index']:02d}_depth_raw.npy").relative_to(OUT)),'segmentation_path':str((folder/f"{r['sample_index']:02d}_segmentation.png").relative_to(OUT))}
            clip_depth.append(depth);clip_records.append(rec)
        lo=float(np.percentile(np.concatenate([d[::8,::8].flatten() for d in clip_depth]),2));hi=float(np.percentile(np.concatenate([d[::8,::8].flatten() for d in clip_depth]),98))
        for r,rec,depth in zip(item_frames,clip_records,clip_depth):
            normalized=np.clip((depth-lo)/max(hi-lo,1e-6),0,1)
            path=folder/f"{r['sample_index']:02d}_depth.png";Image.fromarray((normalized*255).astype('uint8')).convert('RGB').save(path)
            rec.update({'depth_path':str(path.relative_to(OUT)),'depth_normalization_low':lo,'depth_normalization_high':hi,'normalized_depth_region_medians':{position((x+.5)/3,(y+.5)/3):float(np.median(normalized[round(y*rec['height']/3):round((y+1)*rec['height']/3),round(x*rec['width']/3):round((x+1)*rec['width']/3)])) for y in range(3) for x in range(3)}})
            results.append(rec)
        (OUT/'geometry_results.json').write_text(json.dumps(results,indent=2));print(f"Geometry {si+1}/40 clips, {len(results)} frames",flush=True)
    (OUT/'geometry_runtime.json').write_text(json.dumps({'seconds':time.time()-started,'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),'device':device},indent=2))

if __name__=='__main__':main()
