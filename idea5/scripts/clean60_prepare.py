"""Select the 60 sharpest human-verified EgoNormia items and build RGB / depth / segmentation grids.

Selection uses image quality only (no labels, no model outcomes): one item per source video from the
released verified split, ranked by the minimum Laplacian variance over five sampled pre-action frames.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/EgoNormia'
OUT = ROOT / 'output/idea5-clean60'
N = 60
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
NONE_ACTION = '[None of these actions is appropriate]'
NONE_JUST = '[None of these justifications is appropriate]'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def permutation(item_id, kind):
    return sorted(range(5), key=lambda i: digest(f'clean60-v1:{item_id}:{kind}:{i}'))


def sample_frames(item_id):
    cap = cv2.VideoCapture(str(DATA / 'video' / item_id / 'video_prev.mp4'))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in FRACTIONS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, int(total * f)))
        ok, frame = cap.read()
        if not ok:
            return None
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    scale = 384 / max(gray.shape)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def strip(images, destination):
    tiles = []
    for im in images:
        im = im.copy()
        im.thumbnail((384, 384), Image.Resampling.LANCZOS)
        tiles.append(im)
    canvas = Image.new('RGB', (sum(t.width for t in tiles), max(t.height for t in tiles)))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0))
        x += t.width
    canvas.save(destination, quality=95)
    return str(destination)


def main():
    (OUT / 'grids').mkdir(parents=True, exist_ok=True)
    data = json.loads((DATA / 'annotations/final_data.json').read_text())
    verified = json.loads((DATA / 'annotations/verified_split.json').read_text())['split']
    # One item per source video, fixed by hash order, before any image is read.
    by_source = {}
    for item_id in sorted(verified, key=lambda i: digest('clean60-v1:' + i)):
        by_source.setdefault(item_id.split('_')[0], item_id)
    scored = []
    for item_id in by_source.values():
        frames = sample_frames(item_id)
        if frames is None:
            continue
        values = [sharpness(f) for f in frames]
        scored.append({'id': item_id, 'min_laplacian_var': min(values), 'frame_laplacian_var': values})
    scored.sort(key=lambda r: -r['min_laplacian_var'])
    chosen = scored[:N]

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    depth_path = str(ROOT / 'models/Depth-Anything-V2-Small')
    processor = AutoImageProcessor.from_pretrained(depth_path)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_path).to(device).eval()
    segmenter = YOLO(str(ROOT / 'models/yolov8n-seg.pt'))

    tasks, key = [], {}
    for n, row in enumerate(chosen, 1):
        item_id = row['id']
        frames = [Image.fromarray(f) for f in sample_frames(item_id)]
        depths, segs = [], []
        for image in frames:
            w, h = image.size
            inputs = {k: v.to(device) for k, v in processor(images=image, return_tensors='pt').items()}
            with torch.inference_mode():
                pred = depth_model(**inputs).predicted_depth
                depths.append(torch.nn.functional.interpolate(pred.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False)[0, 0].cpu().float().numpy())
            det = segmenter.predict(image, conf=.25, iou=.7, imgsz=640, max_det=40, retina_masks=True, device=device, verbose=False)[0]
            canvas = Image.new('RGB', (w, h), 'black')
            if det.masks is not None:
                masks = det.masks.data.cpu().numpy()
                boxes = det.boxes.xyxy.cpu().numpy()
                classes = det.boxes.cls.cpu().numpy().astype(int)
                # Paint large masks first so smaller detected instances remain visible.
                for k in sorted(range(len(masks)), key=lambda k: -float(masks[k].sum())):
                    mask = masks[k] > .5
                    if mask.shape != (h, w):
                        mask = np.asarray(Image.fromarray(mask.astype('uint8') * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
                    rng = np.random.default_rng(int(classes[k]) * 1009 + k + 7)
                    color = tuple(int(x) for x in rng.integers(70, 256, 3))
                    canvas.paste(Image.new('RGB', (w, h), color), mask=Image.fromarray(mask.astype('uint8') * 255))
                draw = ImageDraw.Draw(canvas)
                for k in range(len(masks)):
                    draw.text((max(0, boxes[k][0]), max(0, boxes[k][1])), det.names[int(classes[k])], fill='white', stroke_width=2, stroke_fill='black', font_size=max(14, h // 22))
            segs.append(canvas)
        # Common 2nd/98th percentile scaling per clip; brighter is nearer.
        flat = np.concatenate([d[::8, ::8].flatten() for d in depths])
        lo, hi = float(np.percentile(flat, 2)), float(np.percentile(flat, 98))
        depth_images = [Image.fromarray((np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype('uint8')).convert('RGB') for d in depths]
        paths = {kind: strip(images, OUT / 'grids' / f'{item_id}_{kind}.jpg') for kind, images in [('rgb', frames), ('depth', depth_images), ('segmentation', segs)]}

        x = data[item_id]
        a_order, j_order = permutation(item_id, 'action'), permutation(item_id, 'justification')
        tasks.append({'case_number': n, 'id': item_id,
                      'action_options': [x['behaviors'][i] or NONE_ACTION for i in a_order],
                      'justification_options': [x['justifications'][i] or NONE_JUST for i in j_order],
                      'images': paths})
        key[item_id] = {'correct': x['correct'], 'action_order': a_order, 'justification_order': j_order,
                        'labels': x['taxonomy'].get(str(x['correct']), [])}
    (OUT / 'selection.json').write_text(json.dumps({'rule': 'verified split, one item per source video (hash order), top 60 by minimum Laplacian variance over five frames at 10/30/50/70/90% of video_prev.mp4, frames resized to max side 384', 'candidates': len(scored), 'chosen': chosen, 'not_chosen': scored[N:]}, indent=1))
    (OUT / 'tasks.json').write_text(json.dumps(tasks, indent=1))
    (OUT / 'answer_key.json').write_text(json.dumps(key, indent=1))
    print(json.dumps({'verified': len(verified), 'sources': len(by_source), 'candidates': len(scored), 'chosen': len(chosen),
                      'sharpness_range_chosen': [chosen[-1]['min_laplacian_var'], chosen[0]['min_laplacian_var']],
                      'sharpness_median_rest': float(np.median([r['min_laplacian_var'] for r in scored[N:]])) if scored[N:] else None}))


if __name__ == '__main__':
    main()
