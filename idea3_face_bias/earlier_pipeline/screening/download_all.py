"""Download pinned EgoNormia previous-context previews; retain every official ID."""
import argparse, concurrent.futures, hashlib, json, time, threading
import requests
from pathlib import Path
from PIL import Image

REV='2937df7fa96d8515417e7b5122fbf8eaeeaeee86'
ROOT=Path(__file__).parent
ANNOTATIONS=(ROOT/'data/official_annotations.json')
THREAD=threading.local()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=32);args=parser.parse_args()
 data=json.loads(ANNOTATIONS.read_text());ids=sorted(data)
 raw=ROOT/'raw_previews';small=ROOT/'model_previews';raw.mkdir(exist_ok=True);small.mkdir(exist_ok=True)
 available={}
 def download(cid):
  dest=raw/(cid+'.jpg');scaled=small/(cid+'.jpg')
  if cid in available and available[cid].exists() and not dest.exists():dest.write_bytes(available[cid].read_bytes())
  result={'id':cid,'revision':REV,'status':'pending'}
  for attempt in range(3):
   try:
    if not dest.exists():
     url=f'https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/{REV}/video/{cid}/frame_all_prev.jpg'
     if not hasattr(THREAD,'session'):
      THREAD.session=requests.Session()
      THREAD.session.headers.update({'User-Agent':'EgoNormia-course-feasibility-screen/1.0'})
     response=THREAD.session.get(url,timeout=35)
     response.raise_for_status();payload=response.content
     tmp=dest.with_suffix('.part');tmp.write_bytes(payload);tmp.replace(dest)
    with Image.open(dest) as image:
     image.load();w,h=image.size
     if w%5:raise ValueError('Unexpected preview dimensions: not five equal-width frames')
     fw=w//5;factor=640/max(fw,h);sw=round(fw*factor);sh=round(h*factor)
     if not scaled.exists():
      out=Image.new('RGB',(sw*5,sh))
      for i in range(5):out.paste(image.crop((i*fw,0,(i+1)*fw,h)).convert('RGB').resize((sw,sh),Image.Resampling.LANCZOS),(i*sw,0))
      preview_temp=scaled.with_suffix('.preview.part')
      out.save(preview_temp,format='JPEG',quality=90,subsampling=0)
      preview_temp.replace(scaled)
    result.update(status='downloaded',original_dimensions=[w,h],individual_original_dimensions=[fw,h],model_frame_dimensions=[sw,sh],original_bytes=dest.stat().st_size,model_preview_bytes=scaled.stat().st_size,original_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),preview_sha256=hashlib.sha256(scaled.read_bytes()).hexdigest())
    return result
   except requests.HTTPError as e:
    code=e.response.status_code
    result.update(status='download_failed',error=f'HTTP {code}')
    if code in [403,404]:return result
   except Exception as e:result.update(status='download_failed',error=type(e).__name__+': '+str(e))
   if attempt<2:time.sleep(1+attempt)
  return result
 start=time.time();results={}
 with (ROOT/'download_records.jsonl').open('w') as log, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  jobs={pool.submit(download,cid):cid for cid in ids}
  for future in concurrent.futures.as_completed(jobs):
   r=future.result();results[r['id']]=r;log.write(json.dumps(r)+'\n');log.flush()
   n=len(results)
   if n%50==0 or n==len(ids):
    good=sum(r['status']=='downloaded' for r in results.values())
    print(json.dumps({'completed':n,'total':len(ids),'downloaded':good,'failed':n-good,'elapsed_seconds':round(time.time()-start),'model_preview_MB':round(sum(r.get('model_preview_bytes',0) for r in results.values())/1e6,1)}),flush=True)
 ordered=[results[cid] for cid in ids]
 (ROOT/'download_records.json').write_text(json.dumps(ordered,indent=2))
 (ROOT/'source_metadata.json').write_text(json.dumps({'annotations_repository_commit':'09d8a7c53f06ed582236722ba8496c6a504cd1ab','annotations_sha256':hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest(),'dataset_revision':REV,'population':len(ids),'official_ids':ids,'input':'Five released pre-action frames; no video_during or post-action frames used.','model_preview':'Each frame resized with LANCZOS to longest side 640, then stored JPEG quality 90 with no chroma subsampling. All detectors read these saved model previews.','download_elapsed_seconds':round(time.time()-start)},indent=2))

if __name__=='__main__':main()
