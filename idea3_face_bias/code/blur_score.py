"""Parts 1-2: are detected faces in EgoNormia actually blurred, and how large are they?

For every YuNet face box recorded by the original screen (score >= 0.4), compare fine detail
inside the box with a surrounding ring in the same 640-px detector frame:
    detail_ratio = log10((var(Lap(inner box)) + 1) / (var(Lap(ring)) + 1))
Privacy blur removes detail inside the face but not around it, so blurred faces get low ratios.
The clear/blurred threshold is calibrated against human labels (label_server.py).
"""
import json, sys, hashlib, csv
from pathlib import Path
import numpy as np, cv2

HERE = Path(__file__).parent
PREV = Path(sys.argv[1])  # folder of rebuilt 640-px model_previews (from download_all.py)
RAW = PREV.parent / 'raw_previews'


def frames_of(path):
    img = cv2.imread(str(path))
    w = img.shape[1] // 5
    return [img[:, i * w:(i + 1) * w] for i in range(5)]


def lap_var(gray, x0, y0, x1, y1, hole=None):
    region = gray[y0:y1, x0:x1]
    lap = cv2.Laplacian(region, cv2.CV_64F)
    if hole is None:
        return float(lap[1:-1, 1:-1].var()) if lap.shape[0] > 2 and lap.shape[1] > 2 else 0.0
    m = np.ones(lap.shape, bool)
    hx0, hy0, hx1, hy1 = hole
    m[max(hy0 - y0, 0):max(hy1 - y0, 0), max(hx0 - x0, 0):max(hx1 - x0, 0)] = False
    m[[0, -1], :] = False
    m[:, [0, -1]] = False
    return float(lap[m].var()) if m.sum() > 20 else 0.0


def score_box(gray, box):
    H, W = gray.shape
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    clip = lambda v, hi: int(min(max(v, 0), hi))
    # inner 70% of the face box avoids hair/background at the box edge
    ix0, iy0, ix1, iy1 = clip(cx - .35 * w, W), clip(cy - .35 * h, H), clip(cx + .35 * w, W), clip(cy + .35 * h, H)
    bx0, by0, bx1, by1 = clip(x, W), clip(y, H), clip(x + w, W), clip(y + h, H)
    rx0, ry0, rx1, ry1 = clip(cx - .9 * w, W), clip(cy - .9 * h, H), clip(cx + .9 * w, W), clip(cy + .9 * h, H)
    if ix1 - ix0 < 4 or iy1 - iy0 < 4:
        return None
    vin = lap_var(gray, ix0, iy0, ix1, iy1)
    vring = lap_var(gray, rx0, ry0, rx1, ry1, hole=(bx0, by0, bx1, by1))
    # contrast-normalised detail: fine detail relative to the face's own brightness variation,
    # so dark or low-contrast but sharp faces are not mistaken for blur
    std_in = float(gray[iy0:iy1, ix0:ix1].std())
    return {'lap_inner': vin, 'lap_ring': vring, 'detail_ratio': float(np.log10((vin + 1) / (vring + 1))),
            'inner_std': std_in, 'detail_per_contrast': float(np.log10((vin + 1) / (std_in ** 2 + 1)))}


def main():
    screen = json.loads((HERE / 'data/screening_results.json').read_text())['records']
    dl = {r['id']: r for r in json.loads((HERE / 'data/download_records.json').read_text())}
    rows, hash_ok, hash_bad, missing = [], 0, 0, 0
    for item_no, rec in enumerate(screen, 1):
        cid = rec['id']
        raw = RAW / (cid + '.jpg')
        if raw.exists():
            ok = hashlib.sha256(raw.read_bytes()).hexdigest() == dl[cid]['original_sha256']
            hash_ok += ok
            hash_bad += not ok
        if not any(f['faces'] for f in rec['frames']):
            continue
        p = PREV / (cid + '.jpg')
        if not p.exists():
            missing += 1
            continue
        frames = frames_of(p)
        for f in rec['frames']:
            gray = cv2.cvtColor(frames[f['frame'] - 1], cv2.COLOR_BGR2GRAY).astype(np.float64)
            for j, face in enumerate(f['faces']):
                s = score_box(gray, face['box_xywh'])
                if s is None:
                    continue
                x, y, w, h = face['box_xywh']
                rows.append({'id': cid, 'item': item_no, 'frame': f['frame'], 'face': j,
                             'x': round(x, 1), 'y': round(y, 1), 'w': round(w, 1), 'h': round(h, 1),
                             'min_side': round(min(w, h), 1), 'det_score': face['score'],
                             'pose_supported': face['supported_by_pose'], 'clipped': face['boundary_clipped'],
                             **{k: round(v, 4) for k, v in s.items()}})
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    with open(out / 'face_scores.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    meta = {'faces_scored': len(rows), 'items_with_scored_face': len({r['id'] for r in rows}),
            'raw_preview_sha256_match': hash_ok, 'raw_preview_sha256_mismatch': hash_bad,
            'items_missing_preview': missing}
    (out / 'face_scores_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
