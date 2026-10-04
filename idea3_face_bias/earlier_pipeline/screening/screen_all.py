"""Two-detector screening, not a validated face-usability classifier.

Outputs raw per-frame detections, explicit proxy geometry, and provisional
per-clip triage labels. No constitutional/DPO model is evaluated or trained.
"""
import argparse, hashlib, importlib.metadata, json, math, time
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO

ROOT=Path(__file__).parent
ANNOTATIONS=(ROOT/'data/official_annotations.json')
PARAMETERS={
 'frame_long_side':640,'face_detector_score_threshold':0.4,'face_nms_threshold':0.3,
 'pose_person_score_threshold':0.18,'pose_head_keypoint_threshold':0.3,
 'supported_face_score':0.6,'candidate_min_face_side_pixels':48,
 'candidate_min_frames':2,'face_boundary_margin_pixels':4,
 'frontal_eye_span_over_face_width':0.25,
 'purpose':'High-recall candidate triage on five pre-action frames. Thresholds are pragmatic heuristics, not established psychophysical trait-legibility cutoffs.',
 'qualification':'A candidate is a region for manual verification and an insertion pilot. No label confirms trait legibility, identity stability, causal invariance, or absence of a face.'
}

def rounded(value):return round(float(value),3)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--wait-for-downloads',action='store_true');parser.add_argument('--reprocess-ids',nargs='*');args=parser.parse_args()
 cv2.setNumThreads(1);torch.set_num_threads(2)
 pose=YOLO(str(ROOT/'models/yolov8n-pose.pt'))
 detector=cv2.FaceDetectorYN.create(str(ROOT/'models/yunet.onnx'),'',(640,480),.4,.3,5000)
 data=json.loads(ANNOTATIONS.read_text());ids=sorted(data)
 if args.limit:ids=ids[:args.limit]
 processed={}
 log_path=ROOT/'screening_records.jsonl'
 if log_path.exists():
  for line in log_path.read_text().splitlines():
   try:r=json.loads(line);processed[r['id']]=r
   except json.JSONDecodeError:pass
 completed_path=ROOT/'screening_results.json'
 if completed_path.exists():
  # A completed snapshot is authoritative; the streaming log may be partial.
  completed=json.loads(completed_path.read_text())
  for r in completed['records']:processed[r['id']]=r
 for cid in args.reprocess_ids or []:processed.pop(cid,None)
 started=time.time()
 with log_path.open('a') as log:
  while True:
   progress=False
   downloads_complete=(ROOT/'download_records.json').exists()
   downloads={r['id']:r for r in json.loads((ROOT/'download_records.json').read_text())} if downloads_complete else {}
   for cid in ids:
    if cid in processed:continue
    p=ROOT/'model_previews'/(cid+'.jpg')
    if not p.exists():
     if downloads_complete:
      record={'id':cid,'status':'unavailable','screening_label':'download_failed','download_error':downloads.get(cid,{}).get('error','No saved preview'),'frames':[]}
     else:continue
    else:
     payload=p.read_bytes()
     if not payload.endswith(b'\xff\xd9'):continue
     input_sha=hashlib.sha256(payload).hexdigest()
     if downloads_complete and input_sha!=downloads.get(cid,{}).get('preview_sha256'):raise ValueError('Saved input does not match download checksum: '+cid)
     image=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
     if image is None:continue
     h,w=image.shape[:2];fw=w//5
     frames=[image[:,j*fw:(j+1)*fw].copy() for j in range(5)]
     try:
      outputs=pose.predict(frames,imgsz=640,conf=.18,device='cpu',verbose=False)
      frame_records=[]
      for index,(frame,output) in enumerate(zip(frames,outputs)):
       height,width=frame.shape[:2]
       people=[];heads=[]
       if output.boxes is not None and output.keypoints is not None:
        for person_index,(box,points) in enumerate(zip(output.boxes.data.cpu().numpy(),output.keypoints.data.cpu().numpy())):
         x1,y1,x2,y2,score,cls=map(float,box)
         people.append({'box':[rounded(x1),rounded(y1),rounded(x2),rounded(y2)],'score':rounded(score),'head_points':[[rounded(v) for v in point] for point in points[:5]]})
         confident=points[:5,2]>=.3
         nose=points[0];visible=points[:5,:2][confident]
         if confident.sum()>=2 and nose[2]>=.3:
          xx0,yy0=visible.min(axis=0);xx1,yy1=visible.max(axis=0)
          span=max(float(xx1-xx0),float(yy1-yy0),8)
          hx0=float(xx0)-.35*span;hy0=float(yy0)-.55*span
          hx1=float(xx1)+.35*span;hy1=float(yy1)+.65*span
          in_upper_body=float(nose[1])<=y1+.33*max(y2-y1,1)
          inside=0<=float(nose[0])<width and 0<=float(nose[1])<height
          clipped=hx0<4 or hy0<4 or hx1>width-4 or hy1>height-4
          heads.append({'person_index':person_index,'nose':[rounded(v) for v in nose],'head_box_proxy':[rounded(hx0),rounded(hy0),rounded(hx1),rounded(hy1)],'confident_head_points':int(confident.sum()),'in_upper_body':bool(in_upper_body),'inside_image':bool(inside),'clipped_proxy':bool(clipped),'plausible_proxy':bool(in_upper_body and inside),'warning':'Pose-derived proxy, not a detected face; head keypoints can be hallucinated on fabric or cropped bodies.'})
       detector.setInputSize((width,height));_,faces=detector.detect(frame)
       face_records=[]
       for raw_face in ([] if faces is None else faces):
        x,y,bw,bh=map(float,raw_face[:4]);confidence=float(raw_face[-1]);landmarks=raw_face[4:14].reshape(5,2)
        eye0,eye1,nose=landmarks[:3];eye_vec=eye1-eye0;eye_span=float(np.linalg.norm(eye_vec))
        nose_projection=float(np.dot(nose-eye0,eye_vec)/max(eye_span**2,1))
        frontal=eye_span/max(bw,1)>=.25 and .05<=nose_projection<=.95
        clipped=x<4 or y<4 or x+bw>width-4 or y+bh>height-4
        matching=[]
        for head in heads:
         nx,ny,nc=head['nose']
         if head['plausible_proxy'] and x-.15*bw<=nx<=x+1.15*bw and y-.15*bh<=ny<=y+1.15*bh:
          matching.append(head['person_index'])
        supported=confidence>=.6 and bool(matching)
        candidate=supported and min(bw,bh)>=48 and frontal and not clipped
        face_records.append({'box_xywh':[rounded(x),rounded(y),rounded(bw),rounded(bh)],'score':rounded(confidence),'landmarks':[[rounded(v) for v in point] for point in landmarks],'eye_span_over_width':rounded(eye_span/max(bw,1)),'nose_projection_between_eyes':rounded(nose_projection),'frontal_geometry_proxy':bool(frontal),'boundary_clipped':bool(clipped),'matching_pose_person_indices':matching,'supported_by_pose':bool(supported),'large_frontal_candidate':bool(candidate)})
       frame_records.append({'frame':index+1,'dimensions':[width,height],'faces':face_records,'people':people,'head_proxies':heads})
      candidate_frames=[f['frame'] for f in frame_records if any(x['large_frontal_candidate'] for x in f['faces'])]
      face_frames=[f['frame'] for f in frame_records if f['faces']]
      supported_frames=[f['frame'] for f in frame_records if any(x['supported_by_pose'] for x in f['faces'])]
      head_frames=[f['frame'] for f in frame_records if any(x['plausible_proxy'] for x in f['head_proxies'])]
      all_faces=[face for f in frame_records for face in f['faces']]
      if len(candidate_frames)>=2:label='large_face_candidate'
      elif face_frames or head_frames:label='possible_region_review'
      else:label='no_region_detected'
      best=max([min(x['box_xywh'][2:]) for x in all_faces if x['large_frontal_candidate']],default=0)
      rank_score=best*(1+.25*len(candidate_frames))
      ann=data[cid];correct=ann['correct'];norms=ann['taxonomy'].get(str(correct),[])
      record={'id':cid,'status':'screened','input_sha256_during_inference':input_sha,'screening_label':label,'candidate_frames':candidate_frames,'any_face_detection_frames':face_frames,'supported_face_frames':supported_frames,'plausible_head_proxy_frames':head_frames,'largest_face_min_side_640':rounded(max([min(x['box_xywh'][2:]) for x in all_faces],default=0)),'largest_candidate_min_side_640':rounded(best),'candidate_rank_score':rounded(rank_score),'maximum_people_detected':max([len(f['people']) for f in frame_records],default=0),'multiple_people_flag':max([len(f['people']) for f in frame_records],default=0)>1,'correct_action_index':correct,'correct_action':ann['behaviors'][correct] or '[none appropriate]','correct_action_norms':norms,'has_primary_norm_label':bool(set(norms)&{'Privacy','Cooperation','Coordination/Proactivity','Communication/Legibility'}),'face_morphology_invariance':'not assessed','mask_or_occlusion':'not assessed automatically','human_review':'pending','frames':frame_records}
     except Exception as exc:
      record={'id':cid,'status':'processing_failed','screening_label':'processing_failed','error':type(exc).__name__+': '+str(exc),'frames':[]}
    processed[cid]=record;log.write(json.dumps(record)+'\n');log.flush();progress=True
    n=len(processed)
    if n%50==0:
     counts={label:sum(r['screening_label']==label for r in processed.values()) for label in ['large_face_candidate','possible_region_review','no_region_detected','download_failed','processing_failed']}
     print(json.dumps({'screened_or_recorded':n,'target':len(ids),'elapsed_seconds':round(time.time()-started),'counts':counts}),flush=True)
   if all(cid in processed for cid in ids):break
   if not args.wait_for_downloads:break
   if not progress:time.sleep(2)
 ordered=[processed[cid] for cid in ids if cid in processed]
 (ROOT/'screening_results.json').write_text(json.dumps({'parameters':PARAMETERS,'records':ordered},indent=2))
 versions={k:importlib.metadata.version(k) for k in ['opencv-python-headless','torch','torchvision','ultralytics','numpy','Pillow']}
 models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'models').glob('*') if p.is_file()}
 (ROOT/'method_metadata.json').write_text(json.dumps({'parameters':PARAMETERS,'package_versions':versions,'model_sha256':models,'model_sources':{'yunet':'https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet','yolov8n-pose':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt'},'important':'This automated triage does not confirm usability or absence. Manual validation is separate. No source descriptions are used by the detectors or triage rules.'},indent=2))
 print('FINISHED:',len(ordered),flush=True)

if __name__=='__main__':main()
