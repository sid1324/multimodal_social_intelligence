"""Pose/hand skeleton strips on a black canvas for every EgoNormia item (same five frames as the other maps).

Also stores the raw landmarks so geometry can be computed later without re-running MediaPipe. Resumable.
"""
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from audit_egonormia_pose import BODY_EDGES, HAND_EDGES, draw_points, serialize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/EgoNormia'
OUT = ROOT / 'output/idea5-midterm/grids'
LANDMARKS = ROOT / 'output/idea5-midterm/pose_landmarks'
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]


def frames_384(item_id):
    cap = cv2.VideoCapture(str(DATA / 'video' / item_id / 'video_prev.mp4'))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in FRACTIONS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, int(total * f)))
        ok, frame = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((384, 384), Image.Resampling.LANCZOS)
        frames.append(np.asarray(im))
    cap.release()
    return frames


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LANDMARKS.mkdir(parents=True, exist_ok=True)
    ids = sorted(json.loads((DATA / 'annotations/final_data.json').read_text()))
    cpu = python.BaseOptions.Delegate.CPU
    pose_opts = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT / 'models/pose_landmarker_full.task'), delegate=cpu),
                                             running_mode=vision.RunningMode.IMAGE, num_poses=4, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5)
    hand_opts = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT / 'models/hand_landmarker.task'), delegate=cpu),
                                             running_mode=vision.RunningMode.IMAGE, num_hands=4, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5)
    failed = []
    with vision.PoseLandmarker.create_from_options(pose_opts) as pose, vision.HandLandmarker.create_from_options(hand_opts) as hands:
        for n, item_id in enumerate(ids, 1):
            if (OUT / f'{item_id}_pose.jpg').exists() and (LANDMARKS / f'{item_id}.json').exists():
                continue
            frames = frames_384(item_id)
            if frames is None:
                failed.append(item_id)
                continue
            tiles, record = [], []
            for rgb in frames:
                p = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
                h = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
                skeleton = np.zeros_like(rgb)
                for points in p.pose_landmarks:
                    draw_points(skeleton, points, BODY_EDGES, (255, 220, 0), 0.5)
                for points in h.hand_landmarks:
                    draw_points(skeleton, points, HAND_EDGES, (255, 70, 210))
                tiles.append(skeleton)
                record.append({'body': serialize(p.pose_landmarks), 'hands': serialize(h.hand_landmarks),
                               'handedness': [[c.category_name for c in cats] for cats in h.handedness]})
            Image.fromarray(np.concatenate(tiles, axis=1)).save(OUT / f'{item_id}_pose.jpg', quality=95)
            (LANDMARKS / f'{item_id}.json').write_text(json.dumps(record))
            if n % 50 == 0:
                print(n, len(ids), flush=True)
    (OUT.parent / 'pose_failed.json').write_text(json.dumps(failed))
    print('done', len(ids), 'failed', len(failed))


if __name__ == '__main__':
    main()
