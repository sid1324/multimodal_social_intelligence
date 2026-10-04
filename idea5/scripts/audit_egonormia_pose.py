"""Fixed-sample body/hand representation audit; no VLM inference or training."""
import argparse
import concurrent.futures
import hashlib
import json
import platform
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/EgoNormia"
OUT = ROOT / "output/idea5-audit"
BODY_EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22)]
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

def digest(value):
    return hashlib.sha256(("egonormia-pose-audit-v1:" + value).encode()).hexdigest()

def select_sample(n=40):
    annotations = json.loads((DATA / "annotations/final_data.json").read_text())
    by_source = {}
    for item_id, item in annotations.items():
        source = item_id.split("_")[0]
        by_source.setdefault(source, []).append(item_id)
    selected = []
    for source in sorted(by_source, key=digest)[:n]:
        item_id = min(by_source[source], key=digest)
        item = annotations[item_id]
        selected.append({"id": item_id, "source": source, "categories": item["taxonomy"].get(str(item["correct"]), []), "annotation": item})
    protocol = {
        "version": "v1", "selection": "First 40 source-video IDs by SHA256(egonormia-pose-audit-v1: + ID); one item per source by the same hash rule. No visual or detector-output selection.",
        "annotation_count": len(annotations), "source_count": len(by_source), "sample": selected,
        "frame_fractions": [0.1,0.3,0.5,0.7,0.9], "clip": "video_prev.mp4 only (pre-action context)",
        "pose_model": "MediaPipe Pose Landmarker Full float16; IMAGE mode; num_poses=4; detection/presence=0.5",
        "hand_model": "MediaPipe Hand Landmarker float16; IMAGE mode; num_hands=4; detection/presence=0.5",
        "retained_landmark": "0<=x<=1 and 0<=y<=1; visibility and presence each >= threshold (0.5 or 0.8). Model confidence is not measured correctness.",
        "body_output_rule": "At least one detected body with >=8 retained landmarks out of 33. This is an operational output-availability rule, not a validated quality standard.",
        "hand_output_rule": "At least one predicted hand. No per-landmark visibility/presence score supplied by this detector; handedness confidence is not detection accuracy.",
        "human_review": "Pending; no precision, recall, pose accuracy, spatial relevance or social-norm accuracy is claimed.",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return protocol

def ensure_clips(protocol):
    manifest = json.loads((DATA / "download_manifest.json").read_text())
    sizes = {x["path"]: x["size"] for x in manifest["files"]}
    def fetch(item):
        relative = f"video/{item['id']}/video_prev.mp4"
        path = DATA / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == sizes[relative]:
            return
        url = f"https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/{manifest['revision']}/{relative}"
        for attempt in range(4):
            try:
                payload = urllib.request.urlopen(url, timeout=120).read()
                if len(payload) != sizes[relative]:
                    raise ValueError("Unexpected clip size")
                temp = path.with_suffix(".audit.part")
                temp.write_bytes(payload)
                temp.replace(path)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, protocol["sample"]))

def retained(lm, threshold):
    return 0 <= lm.x <= 1 and 0 <= lm.y <= 1 and lm.visibility >= threshold and lm.presence >= threshold

def draw_points(img, landmarks, edges, color, threshold=None):
    h,w = img.shape[:2]
    keep = [retained(p,threshold) if threshold is not None else 0 <= p.x <= 1 and 0 <= p.y <= 1 for p in landmarks]
    points = [(round(p.x*(w-1)),round(p.y*(h-1))) for p in landmarks]
    for a,b in edges:
        if keep[a] and keep[b]:
            cv2.line(img, points[a], points[b], color, max(2,w//500), cv2.LINE_AA)
    for point,k in zip(points,keep):
        if k:
            cv2.circle(img,point,max(3,w//300),color,-1,cv2.LINE_AA)

def serialize(sets):
    return [[{"x":p.x,"y":p.y,"z":p.z,"visibility":p.visibility,"presence":p.presence} for p in points] for points in sets]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--sample-size",type=int,default=40); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=select_sample(args.sample_size)
    print("Sample fixed before detector inference; obtaining pre-action clips",flush=True)
    ensure_clips(protocol)
    pose_opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/"models/pose_landmarker_full.task"),delegate=python.BaseOptions.Delegate.CPU), running_mode=vision.RunningMode.IMAGE, num_poses=4, min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5)
    hand_opts=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/"models/hand_landmarker.task"),delegate=python.BaseOptions.Delegate.CPU), running_mode=vision.RunningMode.IMAGE, num_hands=4,min_hand_detection_confidence=0.5,min_hand_presence_confidence=0.5)
    records=[]; started=time.time()
    with vision.PoseLandmarker.create_from_options(pose_opts) as pose, vision.HandLandmarker.create_from_options(hand_opts) as hands:
        for index,item in enumerate(protocol["sample"]):
            clip=DATA / f"video/{item['id']}/video_prev.mp4"
            cap=cv2.VideoCapture(str(clip)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or count<5 or fps<=0:
                raise RuntimeError(f"Cannot decode {clip}")
            folder=OUT/"frames"/item["id"]; folder.mkdir(parents=True,exist_ok=True)
            for fi,fraction in enumerate(protocol["frame_fractions"]):
                frame_index=min(count-1,int(fraction*count)); cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index); ok,bgr=cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {frame_index} of {clip}")
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                t=time.time(); p=pose.detect(image); h=hands.detect(image); latency=time.time()-t
                raw_bodies=len(p.pose_landmarks); raw_hands=len(h.hand_landmarks)
                retained50=[sum(retained(q,0.5) for q in points) for points in p.pose_landmarks]
                retained80=[sum(retained(q,0.8) for q in points) for points in p.pose_landmarks]
                wrists50=[sum(retained(points[j],0.5) for j in [15,16]) for points in p.pose_landmarks]
                overlay=bgr.copy(); skeleton=np.zeros_like(bgr)
                for points in p.pose_landmarks:
                    for image_out in [overlay,skeleton]: draw_points(image_out,points,BODY_EDGES,(0,220,255),0.5)
                for points in h.hand_landmarks:
                    for image_out in [overlay,skeleton]: draw_points(image_out,points,HAND_EDGES,(210,70,255))
                paths={"rgb":folder/f"{fi:02d}_rgb.jpg","overlay":folder/f"{fi:02d}_overlay.jpg","skeleton":folder/f"{fi:02d}_skeleton.jpg"}
                for key,img in [("rgb",bgr),("overlay",overlay),("skeleton",skeleton)]:
                    if not cv2.imwrite(str(paths[key]),img,[cv2.IMWRITE_JPEG_QUALITY,92]): raise RuntimeError("Image write failed")
                records.append({"id":item["id"],"source":item["source"],"categories":item["categories"],"frame_index":frame_index,"sample_index":fi,"timestamp_seconds":frame_index/fps,"clip_frames":count,"clip_fps":fps,"width":bgr.shape[1],"height":bgr.shape[0],"raw_bodies":raw_bodies,"raw_hands":raw_hands,"body_available_50":any(k>=8 for k in retained50),"body_available_80":any(k>=8 for k in retained80),"body_wrists_50":sum(wrists50),"body_retained_50":retained50,"body_retained_80":retained80,"body_landmarks":serialize(p.pose_landmarks),"hand_landmarks":serialize(h.hand_landmarks),"handedness":[[{"label":c.category_name,"score":c.score} for c in cats] for cats in h.handedness],"inference_seconds":latency,"paths":{k:str(v.relative_to(OUT)) for k,v in paths.items()}})
            cap.release()
            print(f"{index+1}/{len(protocol['sample'])} clips; {len(records)} frames",flush=True)
            (OUT/"frame_results.json").write_text(json.dumps(records,indent=2))
    runtime={"elapsed_inference_and_export_seconds":time.time()-started,"python":platform.python_version(),"platform":platform.platform(),"mediapipe":mp.__version__,"opencv":cv2.__version__,"numpy":np.__version__,"model_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"models").glob("*.task")}}
    (OUT/"runtime.json").write_text(json.dumps(runtime,indent=2))
    print(json.dumps(runtime),flush=True)

if __name__=="__main__": main()
