"""Step 0 data: download the pinned EgoNormia pre-action frame strips, rebuild A2's 640-px model
previews with A2's exact recipe (earlier_pipeline/screening/download_all.py), and verify both files
against the SHA-256 hashes A2 recorded in download_records.json. Also checks the annotations file
against the pinned GitHub commit.

Usage: python -I src/download_data.py [--workers 16]
Env:   DATA_DIR (default $P777/data/egonormia), REF_DIR (default ./reference/premise_analysis)
Output: $DATA_DIR/raw_previews/*.jpg, $DATA_DIR/model_previews/*.jpg, $DATA_DIR/verify_report.json
"""
import argparse, concurrent.futures as cf, hashlib, json, os, threading, time
from pathlib import Path
import requests
from PIL import Image

REV = '2937df7fa96d8515417e7b5122fbf8eaeeaeee86'          # HF dataset revision (images)
ANN_COMMIT = '09d8a7c53f06ed582236722ba8496c6a504cd1ab'   # GitHub commit (annotations)
HERE = Path(__file__).resolve().parent.parent
P777 = os.environ.get('P777', '/ocean/projects/cis260338p/ylothe')
DATA = Path(os.environ.get('DATA_DIR', f'{P777}/data/egonormia'))
REF = Path(os.environ.get('REF_DIR', HERE / 'reference/premise_analysis'))
THREAD = threading.local()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def session():
    if not hasattr(THREAD, 's'):
        THREAD.s = requests.Session()
        THREAD.s.headers.update({'User-Agent': 'EgoNormia-11777-idea3-midterm/1.0'})
    return THREAD.s


def make_preview(raw, out):
    """A2 recipe, unchanged: split into 5 equal-width frames, LANCZOS so the longest side is 640,
    paste left to right, save JPEG quality 90 with no chroma subsampling."""
    with Image.open(raw) as im:
        im.load()
        w, h = im.size
        if w % 5:
            raise ValueError(f'width {w} not divisible by 5')
        fw = w // 5
        f = 640 / max(fw, h)
        sw, sh = round(fw * f), round(h * f)
        strip = Image.new('RGB', (sw * 5, sh))
        for i in range(5):
            strip.paste(im.crop((i * fw, 0, (i + 1) * fw, h)).convert('RGB').resize((sw, sh), Image.Resampling.LANCZOS), (i * sw, 0))
    tmp = out.with_suffix('.part')
    strip.save(tmp, format='JPEG', quality=90, subsampling=0)
    tmp.replace(out)


def fetch(rec):
    cid = rec['id']
    raw, prev = DATA / 'raw_previews' / f'{cid}.jpg', DATA / 'model_previews' / f'{cid}.jpg'
    out = {'id': cid}
    for attempt in range(4):
        try:
            if not raw.exists() or sha(raw) != rec['original_sha256']:
                url = f'https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/{REV}/video/{cid}/frame_all_prev.jpg'
                r = session().get(url, timeout=60)
                r.raise_for_status()
                tmp = raw.with_suffix('.part')
                tmp.write_bytes(r.content)
                tmp.replace(raw)
            out['raw_ok'] = sha(raw) == rec['original_sha256']
            if not prev.exists() or sha(prev) != rec['preview_sha256']:
                make_preview(raw, prev)
            out['preview_ok'] = sha(prev) == rec['preview_sha256']
            with Image.open(prev) as im:
                out['preview_size'] = list(im.size)
            return out
        except Exception as e:
            out['error'] = f'{type(e).__name__}: {str(e)[:200]}'
            time.sleep(2 * (attempt + 1))
    return out


def check_annotations():
    """Compare A2's saved annotations with final_data.json at the pinned GitHub commit."""
    url = f'https://raw.githubusercontent.com/open-social-world/EgoNormia/{ANN_COMMIT}/src/final_dataset/final_data.json'
    local = REF / 'data/official_annotations.json'
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        same_bytes = hashlib.sha256(r.content).hexdigest() == sha(local)
        a, b = json.loads(r.content), json.loads(local.read_text())
        same_parsed = a == b
        return {'url': url, 'byte_identical': same_bytes, 'parsed_identical': same_parsed,
                'remote_items': len(a), 'local_items': len(b)}
    except Exception as e:
        return {'url': url, 'error': f'{type(e).__name__}: {str(e)[:200]}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    (DATA / 'raw_previews').mkdir(parents=True, exist_ok=True)
    (DATA / 'model_previews').mkdir(parents=True, exist_ok=True)
    recs = json.loads((REF / 'data/download_records.json').read_text())
    print(f'{len(recs)} records; data dir {DATA}', flush=True)
    t0, results = time.time(), []
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(fetch, recs), 1):
            results.append(r)
            if i % 100 == 0 or i == len(recs):
                print(f'{i}/{len(recs)} done, {time.time() - t0:.0f}s', flush=True)
    by_id = {r['id']: r for r in recs}
    size_ok = sum(1 for r in results if r.get('preview_size') == [by_id[r['id']]['model_frame_dimensions'][0] * 5,
                                                                  by_id[r['id']]['model_frame_dimensions'][1]])
    report = {'n': len(results),
              'raw_sha_ok': sum(bool(r.get('raw_ok')) for r in results),
              'preview_sha_ok': sum(bool(r.get('preview_ok')) for r in results),
              'preview_size_matches_record': size_ok,
              'errors': [r for r in results if 'error' in r and not r.get('raw_ok')],
              'preview_sha_mismatch_ids': [r['id'] for r in results if r.get('raw_ok') and not r.get('preview_ok')],
              'annotations': check_annotations(),
              'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}
    (DATA / 'verify_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, indent=2))


if __name__ == '__main__':
    main()
