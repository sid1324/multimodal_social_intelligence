"""Build RGB / depth / segmentation strips for every EgoNormia item (same recipe as clean60_prepare.py).

Five frames at 10/30/50/70/90% of video_prev.mp4, max side 384, one horizontal strip per modality.
Resumable: items whose three strips already exist are skipped.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from clean60_prepare import sample_frames, strip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/EgoNormia'
OUT = ROOT / 'output/idea5-midterm/grids'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(json.loads((DATA / 'annotations/final_data.json').read_text()))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    depth_path = str(ROOT / 'models/Depth-Anything-V2-Small')
    processor = AutoImageProcessor.from_pretrained(depth_path)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_path).to(device).eval()
    segmenter = YOLO(str(ROOT / 'models/yolov8n-seg.pt'))
    failed = []
    for n, item_id in enumerate(ids, 1):
        if all((OUT / f'{item_id}_{k}.jpg').exists() for k in ('rgb', 'depth', 'segmentation')):
            continue
        raw = sample_frames(item_id)
        if raw is None:
            failed.append(item_id)
            continue
        frames = []
        for f in raw:
            im = Image.fromarray(f)
            im.thumbnail((384, 384), Image.Resampling.LANCZOS)
            frames.append(im)
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
        for kind, images in [('rgb', frames), ('depth', depth_images), ('segmentation', segs)]:
            strip(images, OUT / f'{item_id}_{kind}.jpg')
        if n % 50 == 0:
            print(n, len(ids), flush=True)
    (OUT.parent / 'prepare_failed.json').write_text(json.dumps(failed))
    print('done', len(ids), 'failed', len(failed))


if __name__ == '__main__':
    main()
