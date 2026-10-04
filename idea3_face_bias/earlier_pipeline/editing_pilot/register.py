from pathlib import Path
import cv2,numpy as np,json
from PIL import Image
root=Path('/workspace/scratch/8eb17a085ce3');work=root/'tmp/full_sequence_tests';out=root/'output/analysis/Full_Sequence_Face_Tests'; results=[]
for check in sorted(out.glob('item_*/*/*/pixel_checks.json')):
 rec=json.loads(check.read_text());raw=Image.open(rec['raw_generation']).convert('RGB'); w,h=raw.size
 for frame in rec['frames']:
  if not frame['editable']:continue
  f=frame['frame'];col=(f-1)%2;row=(f-1)//2
  # Match the exact rounded cell bounds used by the JS compositor.
  tile=raw.crop((int(col*w/2+.5),int(row*h/3+.5),int((col+1)*w/2+.5),int((row+1)*h/3+.5)))
  src=np.asarray(Image.open(frame['source']).convert('RGB'));tile=tile.resize((src.shape[1],src.shape[0]),Image.Resampling.LANCZOS);gen=np.asarray(tile)
  grayS=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY);grayG=cv2.cvtColor(gen,cv2.COLOR_RGB2GRAY)
  featureMask=np.full(grayS.shape,255,np.uint8);x0,y0,x1,y1=frame['region_xyxy'];featureMask[max(0,y0-20):min(src.shape[0],y1+20),max(0,x0-20):min(src.shape[1],x1+20)]=0
  sift=cv2.SIFT_create(nfeatures=4000);ks,ds=sift.detectAndCompute(grayS,featureMask);kg,dg=sift.detectAndCompute(grayG,None)
  pairs=cv2.BFMatcher().knnMatch(ds,dg,k=2);matches=[a for a,b in pairs if a.distance<.72*b.distance]
  ps=np.float32([ks[m.queryIdx].pt for m in matches]);pg=np.float32([kg[m.trainIdx].pt for m in matches]);H,inliers=cv2.findHomography(ps,pg,cv2.RANSAC,3.0,maxIters=5000)
  if H is None:raise RuntimeError('Registration failed '+str(check))
  proj=cv2.perspectiveTransform(ps.reshape(-1,1,2),H).reshape(-1,2);errors=np.linalg.norm(proj-pg,axis=1);ins=inliers[:,0]>0
  center=np.float32([[(x0+x1)/2,(y0+y1)/2]]);mapped=cv2.perspectiveTransform(center.reshape(-1,1,2),H).reshape(-1,2)[0]
  results.append(dict(item=rec['item'],method=rec['method'],condition=rec['condition'],frame=f,source_to_generated=H.tolist(),matches=len(matches),inliers=int(ins.sum()),median_inlier_error=float(np.median(errors[ins])),center_displacement=float(np.linalg.norm(mapped-center[0]))))
(out/'registration.json').write_text(json.dumps(results,indent=2))
print(json.dumps({'registrations':len(results),'minimum_inliers':min(r['inliers'] for r in results),'max_median_error':max(r['median_inlier_error'] for r in results),'max_center_shift':max(r['center_displacement'] for r in results)}))
