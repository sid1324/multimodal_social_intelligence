"""Round 2 labels: (a) orientation of faces labelled 'clear' in round 1;
(b) item-level usable-face check on a simple random sample of all 1,853 items (seed 2027).
Usage: python label_round2.py <model_previews dir>   then open http://localhost:8766
Labels are written to results/human_labels_round2.json after every click.
"""
import json, sys, random, http.server, socketserver
from pathlib import Path
from PIL import Image

HERE = Path(__file__).parent
PREV = Path(sys.argv[1])
OUT = HERE / 'results'
LABELS = OUT / 'human_labels_round2.json'
SEED, N_ITEMS = 2027, 40


def build():
    s1 = json.loads((OUT / 'label_sample.json').read_text())
    l1 = json.loads((OUT / 'human_labels.json').read_text())
    faces = [f['key'] for f in s1['faces'] if l1.get(f['key']) == 'clear']
    ids = sorted(json.loads((HERE / 'data/official_annotations.json').read_text()))
    rng = random.Random(SEED)
    picks = rng.sample(range(len(ids)), N_ITEMS)
    strips = OUT / 'label_strips'
    strips.mkdir(exist_ok=True)
    items = []
    for i in picks:
        cid = ids[i]
        im = Image.open(PREV / (cid + '.jpg'))
        im.resize((1600, round(im.height * 1600 / im.width))).save(strips / (cid + '.jpg'), quality=88)
        items.append({'key': 'item_' + cid, 'id': cid, 'item': i + 1})
    sample = {'seed': SEED, 'orientation_faces': faces, 'items': items}
    (OUT / 'label_sample_round2.json').write_text(json.dumps(sample, indent=2))
    return sample


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Usable Face Check</title><style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:14px 0}.done{border-color:#16a34a}
img{max-width:100%}button{margin:4px;padding:8px 12px;border:1px solid #888;border-radius:6px;background:#fff;cursor:pointer}
button.on{background:#2563eb;color:#fff;border-color:#2563eb}#bar{position:sticky;top:0;background:#fafafa;padding:8px 0;font-weight:600}
</style></head><body>
<h2>Round 2</h2>
<p><b>Part A.</b> Faces you marked clear. Orientation: <b>frontal</b>, <b>angled visible</b> (turned but eyes/nose/mouth clearly visible),
or <b>turned away</b> (mostly back/side of head).</p><div id=bar></div><div id=A></div>
<h3>Part B: 40 random EgoNormia items (all five frames)</h3>
<p>Pick the best option that applies to <b>any person other than the camera wearer</b> in any frame:<br>
<b>usable face</b> = an unblurred face, frontal or angled with features clearly visible, big enough to judge;<br>
<b>blurred face only</b> = faces exist but are privacy-blurred; <b>turned away/too small</b> = people present, but no face meets the usable bar;
<b>no other person</b> = no other person visible.</p><div id=B></div>
<script>
let S,L={};const AO=['frontal','angled_visible','turned_away'],BO=['usable_face','blurred_face_only','turned_away_or_too_small','no_other_person'];
function save(){fetch('/save',{method:'POST',body:JSON.stringify(L)});bar()}
function bar(){document.getElementById('bar').textContent=`A ${S.orientation_faces.filter(k=>L['o_'+k]).length}/${S.orientation_faces.length} | B ${S.items.filter(i=>L[i.key]).length}/${S.items.length} | saved`}
function card(par,key,html,opts){const c=document.createElement('div');c.className='card'+(L[key]?' done':'');c.innerHTML=html+'<div class=b></div>';
 const el=c.querySelector('.b');opts.forEach(o=>{const b=document.createElement('button');b.textContent=o.replace(/_/g,' ');if(L[key]===o)b.className='on';
 b.onclick=()=>{L[key]=o;[...el.querySelectorAll('button')].forEach(x=>x.className='');b.className='on';c.classList.add('done');save()};el.appendChild(b)});par.appendChild(c)}
fetch('/sample').then(r=>r.json()).then(d=>{S=d.sample;L=d.labels||{};
 S.orientation_faces.forEach(k=>card(document.getElementById('A'),'o_'+k,`<div>${k}</div><img src="/crops/${k}_crop.jpg" style="width:260px"> <img src="/crops/${k}_frame.jpg" style="width:420px">`,AO));
 S.items.forEach((it,i)=>card(document.getElementById('B'),it.key,`<div>Item ${i+1}</div><img src="/strips/${it.id}.jpg">`,BO));bar()})
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
            labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
            pub = {'orientation_faces': SAMPLE['orientation_faces'],
                   'items': [{'key': i['key'], 'id': i['id']} for i in SAMPLE['items']]}
            return self.send(json.dumps({'sample': pub, 'labels': labels}).encode(), 'application/json')
        for prefix, folder in (('/crops/', 'label_crops'), ('/strips/', 'label_strips')):
            if self.path.startswith(prefix):
                p = OUT / folder / Path(self.path).name
                if p.exists():
                    return self.send(p.read_bytes(), 'image/jpeg')
        self.send_error(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        LABELS.write_text(json.dumps(json.loads(body), indent=2))
        self.send(b'ok', 'text/plain')


if __name__ == '__main__':
    sp = OUT / 'label_sample_round2.json'
    SAMPLE = json.loads(sp.read_text()) if sp.exists() else build()
    print('Open http://localhost:8766')
    with socketserver.TCPServer(('127.0.0.1', 8766), H) as srv:
        srv.serve_forever()
