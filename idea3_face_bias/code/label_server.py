"""Local labeling page for the human check (scores hidden, random order).

Faces: 60 detected faces with min side >= 24 px, one per item, 10 from each sixth of the
detail_ratio distribution (equal-count strata with equal samples, so the sample is self-weighting).
Text: 20 gold answers (10 lexicon-flagged, 10 not) to measure lexicon accuracy.
Usage: python label_server.py <model_previews dir>   then open http://localhost:8765
Labels are written to results/human_labels.json after every click.
"""
import json, sys, csv, random, http.server, socketserver
from pathlib import Path
import cv2

HERE = Path(__file__).parent
PREV = Path(sys.argv[1])
OUT = HERE / 'results'
CROPS = OUT / 'label_crops'
LABELS = OUT / 'human_labels.json'
SEED = 2026


def build_sample():
    rows = [r for r in csv.DictReader(open(OUT / 'face_scores.csv')) if float(r['min_side']) >= 24]
    rows.sort(key=lambda r: float(r['detail_ratio']))
    rng = random.Random(SEED)
    k = len(rows) // 6
    faces, used = [], set()
    for s in range(6):
        stratum = rows[s * k:(s + 1) * k] if s < 5 else rows[5 * k:]
        rng.shuffle(stratum)
        n = 0
        for r in stratum:
            if r['id'] in used:
                continue
            used.add(r['id'])
            faces.append({**r, 'stratum': s})
            n += 1
            if n == 10:
                break
    CROPS.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(faces):
        img = cv2.imread(str(PREV / (r['id'] + '.jpg')))
        fw = img.shape[1] // 5
        fr = img[:, (int(r['frame']) - 1) * fw:int(r['frame']) * fw].copy()
        x, y, w, h = (float(r[c]) for c in 'xywh')
        cx, cy, half = x + w / 2, y + h / 2, max(w, h) * 0.9
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, fr.shape[1])), int(min(cy + half, fr.shape[0]))
        crop = fr[y0:y1, x0:x1]
        scale = 300 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(CROPS / f'face_{i:02d}_crop.jpg'), crop)
        cv2.rectangle(fr, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
        cv2.imwrite(str(CROPS / f'face_{i:02d}_frame.jpg'), fr)
        r['key'] = f'face_{i:02d}'
    text = list(csv.DictReader(open(OUT / 'text_arm_items.csv')))
    ann = json.loads((HERE / 'data/official_annotations.json').read_text())
    yes = [t for t in text if t['gold_person_directed'] == 'True']
    no = [t for t in text if t['gold_person_directed'] == 'False' and t['gold_is_none_sentinel'] == 'False']
    texts = []
    for t in rng.sample(yes, 10) + rng.sample(no, 10):
        v = ann[t['id']]
        texts.append({'key': 'text_' + t['id'], 'id': t['id'], 'answer': v['behaviors'][v['correct']],
                      'lexicon_person_directed': t['gold_person_directed'] == 'True'})
    rng.shuffle(faces)
    rng.shuffle(texts)
    sample = {'seed': SEED, 'faces': faces, 'texts': texts}
    (OUT / 'label_sample.json').write_text(json.dumps(sample, indent=2))
    return sample


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Face Blur Check</title><style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:20px auto;padding:0 16px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:14px 0}
.imgs{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}.imgs img{max-width:100%}
.frame{width:420px}button{margin:4px;padding:8px 12px;border:1px solid #888;border-radius:6px;background:#fff;cursor:pointer}
button.on{background:#2563eb;color:#fff;border-color:#2563eb}#bar{position:sticky;top:0;background:#fafafa;padding:8px 0;font-weight:600}
.done{border-color:#16a34a}
</style></head><body>
<h2>Human check: face blur (part A) and person-directed answers (part B)</h2>
<p><b>Part A.</b> Look at the face inside the yellow box. <b>Clear</b> = eyes, nose and mouth details visible (even if small or low-quality).
<b>Blurred</b> = deliberately smeared/privacy-blurred. <b>Not a face</b> = detector mistake. <b>Can't tell</b> = too small or dark to judge.
Saves automatically after each click.</p>
<div id=bar></div><div id=faces></div>
<h3>Part B (optional, 20 items)</h3><p>Does this action involve or address <b>another person</b> (handing to, talking to, helping, moving around someone)?</p>
<div id=texts></div>
<script>
let S=null,L={};
const FL=['clear','blurred','not_a_face','cant_tell'],TL=['yes','no','unsure'];
function save(){fetch('/save',{method:'POST',body:JSON.stringify(L)});bar()}
function bar(){const nf=S.faces.filter(f=>L[f.key]).length,nt=S.texts.filter(t=>L[t.key]).length;
document.getElementById('bar').textContent=`Faces ${nf}/${S.faces.length} | Text ${nt}/${S.texts.length} | saved`}
function btns(key,opts,el){opts.forEach(o=>{const b=document.createElement('button');b.textContent=o.replace(/_/g,' ');
 if(L[key]===o)b.className='on';b.onclick=()=>{L[key]=o;[...el.querySelectorAll('button')].forEach(x=>x.className='');b.className='on';el.closest('.card').classList.add('done');save()};el.appendChild(b)})}
fetch('/sample').then(r=>r.json()).then(d=>{S=d.sample;L=d.labels||{};
 S.faces.forEach((f,i)=>{const c=document.createElement('div');c.className='card'+(L[f.key]?' done':'');
  c.innerHTML=`<div>Face ${i+1}</div><div class=imgs><img src="/crops/${f.key}_crop.jpg"><img class=frame src="/crops/${f.key}_frame.jpg"></div><div class=b></div>`;
  document.getElementById('faces').appendChild(c);btns(f.key,FL,c.querySelector('.b'))});
 S.texts.forEach((t,i)=>{const c=document.createElement('div');c.className='card'+(L[t.key]?' done':'');
  c.innerHTML=`<div>${i+1}. ${t.answer.replace(/</g,'&lt;')}</div><div class=b></div>`;
  document.getElementById('texts').appendChild(c);btns(t.key,TL,c.querySelector('.b'))});bar()})
</script></body></html>"""


class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            return self.send(PAGE.encode(), 'text/html; charset=utf-8')
        if self.path == '/sample':
            pub = {'faces': [{'key': f['key']} for f in SAMPLE['faces']],
                   'texts': [{'key': t['key'], 'answer': t['answer']} for t in SAMPLE['texts']]}
            labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
            return self.send(json.dumps({'sample': pub, 'labels': labels}).encode(), 'application/json')
        if self.path.startswith('/crops/'):
            p = CROPS / Path(self.path).name
            if p.exists():
                return self.send(p.read_bytes(), 'image/jpeg')
        self.send_error(404)

    def do_POST(self):
        if self.path == '/save':
            body = self.rfile.read(int(self.headers['Content-Length']))
            LABELS.write_text(json.dumps(json.loads(body), indent=2))
            return self.send(b'ok', 'text/plain')
        self.send_error(404)


if __name__ == '__main__':
    sp = OUT / 'label_sample.json'
    SAMPLE = json.loads(sp.read_text()) if sp.exists() else build_sample()
    print(f"{len(SAMPLE['faces'])} faces, {len(SAMPLE['texts'])} texts. Open http://localhost:8765")
    with socketserver.TCPServer(('127.0.0.1', 8765), H) as srv:
        srv.serve_forever()
