"""Geometry facts per item from the extractor outputs only (no labels, no descriptions), for the spatial probe.

n_people      {0,1,2,3+}: max over the 5 frames of YOLOv8n person detections (conf >= .4) on the RGB strip tiles
nearest_dist  {none,near,mid,far}: median depth-strip brightness inside the largest person mask, split at terciles
facing        {none,toward,away}: any body with nose visible between visible shoulders in the middle frame -> toward
hands         {0,1+}: hands detected by MediaPipe in any frame
Runs YOLO on CPU so the GPU stays free for the embedder. --all labels every item into spatial_labels_all.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-midterm'
GRIDS = OUT / 'grids'
LANDMARKS = OUT / 'pose_landmarks'


def tiles(path):
    im = Image.open(path).convert('RGB')
    w = im.width // 5
    return [im.crop((k * w, 0, (k + 1) * w, im.height)) for k in range(5)]


def facing(record):
    middle = record[2]['body']
    for body in middle:
        nose, ls, rs = body[0], body[11], body[12]
        if min(nose['visibility'], ls['visibility'], rs['visibility']) >= .5:
            lo, hi = sorted([ls['x'], rs['x']])
            return 'toward' if lo <= nose['x'] <= hi else 'away'
    return 'away' if any(record[k]['body'] for k in range(5)) else 'none'


def main():
    split = json.loads((OUT / 'splits.json').read_text())
    ids = split['test'] + split['train'] + split['train_ext']
    dest = OUT / 'spatial_labels.json'
    if '--all' in sys.argv:  # every item with maps, for the RL rewards; terciles are then over all items
        ids = sorted(p.name[:-len('_pose.jpg')] for p in GRIDS.glob('*_pose.jpg'))
        dest = OUT / 'spatial_labels_all.json'
    yolo = YOLO(str(ROOT / 'models/yolov8n-seg.pt'))
    rows = {}
    for item_id in ids:
        rgb, depth = tiles(GRIDS / f'{item_id}_rgb.jpg'), tiles(GRIDS / f'{item_id}_depth.jpg')
        counts, nearest = [], None
        for im, dm in zip(rgb, depth):
            det = yolo.predict(im, conf=.4, classes=[0], imgsz=640, retina_masks=True, device='cpu', verbose=False)[0]
            counts.append(0 if det.masks is None else len(det.masks))
            if det.masks is not None:
                masks = det.masks.data.numpy() > .5
                biggest = masks[masks.reshape(len(masks), -1).sum(1).argmax()]
                d = np.asarray(dm.convert('L').resize((biggest.shape[1], biggest.shape[0])))
                value = float(np.median(d[biggest]))
                nearest = value if nearest is None else max(nearest, value)  # brighter = nearer
        record = json.loads((LANDMARKS / f'{item_id}.json').read_text())
        rows[item_id] = {'n_people': min(max(counts), 3), 'nearest_brightness': nearest, 'facing': facing(record),
                         'hands': int(any(r['hands'] for r in record))}
    values = [r['nearest_brightness'] for r in rows.values() if r['nearest_brightness'] is not None]
    t1, t2 = np.percentile(values, [100 / 3, 200 / 3])
    for r in rows.values():
        b = r['nearest_brightness']
        r['nearest_dist'] = 'none' if b is None else 'near' if b >= t2 else 'mid' if b >= t1 else 'far'
    dest.write_text(json.dumps(rows, indent=1))
    for key in ('n_people', 'nearest_dist', 'facing', 'hands'):
        print(key, dict(Counter(r[key] for r in rows.values())))


if __name__ == '__main__':
    main()
