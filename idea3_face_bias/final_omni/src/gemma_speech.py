"""Gemma 4 12B: does steering built from TYPED remarks also reduce appearance bias when the remark is SPOKEN?
Plan: QWEN_TRANSFER_PLAN.md, update Nov 25 23:05 CT (Yash): L1 main, L2 also reported. Nothing here touches the test split.

Stages (each resumable; one prompt at a time; exit 75 = time limit, run again; exit 3 = a check stopped the run, see STOP.md):
  check      20 validation scenes, no remark: letter mass, greedy answer = scored answer, steering hook removes the direction
  perc       30 validation scenes: repeat the spoken remark (and the typed one); speech must contain the trait word in >= 90%
  g2         300 training scenes (midterm gate set), unsteered: text (none, L1, L2, weather valence) and speech (L1, L2);
             saves per-scene difference vectors (untrustworthy - trustworthy, harmful - helpful behavior) for the directions
  gate       CPU: bias above the noise floor per format and level (95% range above 0); a level goes on only if text AND speech pass
  directions CPU: appearance direction per level from typed remarks only (mean over training scenes); speech direction for comparison
  sweep LV   validation, typed remarks only: base plus layer sets x strength (midterm grid); speech is never used to choose
  select LV  CPU: midterm rule (lowest typed appearance flips with accuracy >= base - 2 points and behavior flips >= 80% of base)
  transfer LV validation: the chosen setting on SPOKEN remarks, with controls at the same layers and strength: a direction built from
             spoken remarks (the ceiling; transfer ratio), the weather-valence direction, a shuffled-label direction, 3 random
             directions, and the constitution (reviewer changes A to C, approved by Yash Nov 25 23:55 CT)
  faces      validation, exploratory vision arm: the midterm OMI face pairs (pre-registered, results/omi_pairs.json) as a second
             image; face sensitivity = trust-pair flips minus equal-trust-pair flips; base, chosen L1 steering, constitution
  report     CPU: success rule per level, paired by scene, 95% ranges clustered by source video; layer-wise figure; face arm
Usage: python src/gemma_speech.py STAGE [L1|L2]"""
import gzip, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, load_annotations, load_splits, load_item_table, gate_items, permuted_options, seeded_int, constitution
from channels import build, build_repeat, scene, CUES
from evaluate import level_conds, row_for, read_rows, key, layer_sets
from analyze import boot, index, item_stats, summarize, choose, pct

R = Path(os.environ.get('GS_OUT', str(HERE / 'results/gemma')))   # GS_OUT only for the CPU pipeline test
R.mkdir(parents=True, exist_ok=True)
MODEL = 'gemma'
LEVELS = ('L1', 'L2')                       # L1 main (decided before any Gemma result)
ALPHAS = (0.5, 1.0)
N_RAND = 3
END = float(os.environ.get('STOP_AT', '1e12'))   # epoch seconds; stages stop cleanly before it
N_GATE = int(os.environ.get('GS_N_GATE', 300))     # 300 in every real run; smaller only for the CPU pipeline test with GS_FAKE=1
_M = None


class Fake:
    """CPU pipeline test only (GS_FAKE=1): random letters with a built-in appearance effect, small random residuals."""
    name, n_layers = 'gemma', 48
    load_info = {'missing_keys': [], 'unexpected_keys': []}

    def __init__(self):
        self.rng, self._ab = np.random.default_rng(0), None

    def set_ablation(self, d=None, alpha=1.0, layers=()):
        self._ab = None if d is None else (len(layers), alpha)

    def score_many(self, prompts, n_opts, system=None, capture=False):
        outs, res = [], []
        for p in prompts:
            t = ' '.join(x for k, x in p if k == 'text')
            h = seeded_int('fake', t[t.find('What should'):]) % 1000   # same answer for every remark on one scene
            aud = ' '.join(str(x) for k, x in p if k == 'audio')
            low = any(w in t for w in ('untrustworthy', 'cannot be trusted', 'dishonest')) or (aud and h % 2 == 0 and seeded_int('a', aud) % 2 == 0)
            low = low and h % 2 == 0
            shift = 1 if (low and (self._ab is None or self.rng.random() < 0.5)) else 0
            j = (h + shift) % n_opts
            lp = np.full(n_opts, -3.0); lp[j] = -0.1
            outs.append({'letter_logp': lp.tolist(), 'argmax': 'ABCDE'[j], 'letter_mass': 0.99, 'finite': True, 'n_tokens': 300, 'top_token': 'ABCDE'[j]})
            res.append(self.rng.standard_normal((48, 64)).astype(np.float32) + (0.5 if low else 0))
        return outs, (res if capture else None)

    def generate(self, parts, system=None, max_new_tokens=4):
        if max_new_tokens > 10:
            return ' '.join(x for k, x in parts if k == 'text')[:0] + 'The other person in the scene appears untrustworthy.'
        return self.score_many([parts], 4)[0][0]['argmax']


def model():
    global _M
    if _M is None:
        if os.environ.get('GS_FAKE') == '1':
            _M = Fake()
        else:
            from omni_model import load
            _M = load(MODEL)
    return _M


def stop(msg):
    (R / 'STOP.md').write_text(msg + '\n'); print('STOP:', msg, flush=True); sys.exit(3)


def timeup():
    return time.time() > END - 240


def val_sets():
    sp = load_splits()
    pool = sp['pool']['val']
    return pool, [k for k in sp['clean']['val'] if k not in set(pool)]


def grid(path, method, items, cells, split, system=None):
    """Append one row per forward pass; skip what is done. -> False if it stopped for time."""
    m, ann = model(), load_annotations()
    path = Path(path)
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        path.write_bytes(path.read_bytes().rsplit(b'\n', 1)[0] + b'\n')
    done = {key(r) for r in read_rows(path)}
    t0, n = time.time(), 0
    with open(path, 'a') as fh:
        for i, cid in enumerate(items):
            todo = [c for c in cells if (cid, method, c[1], c[0]) not in done]
            if not todo:
                continue
            if timeup():
                print(f'{method}: time limit at item {i}/{len(items)}', flush=True); return False
            v, img = ann[cid], scene(cid)
            outs, _ = m.score_many([build(v, img, c, ch) for c, ch in todo], len(permuted_options(v)), system)
            fh.write(''.join(json.dumps(row_for(cid, v, c, ch, r, split, method, MODEL)) + '\n' for (c, ch), r in zip(todo, outs))); fh.flush()
            n += len(todo)
            if (i + 1) % 20 == 0:
                print(f'{method}: {i + 1}/{len(items)} scenes, {(time.time() - t0) / max(n, 1):.3f} s/pass', flush=True)
    return True


# ---- GPU stages --------------------------------------------------------------------------------------------------
def check():
    import torch
    m, ann = model(), load_annotations()
    items = sorted(load_splits()['pool']['val'], key=lambda k: seeded_int('g1-v1', k))[:20]
    rows = []
    for cid in items:
        v, img = ann[cid], scene(cid)
        p = build(v, img, 'none', 'text')
        r = m.score_many([p], len(permuted_options(v)))[0][0]
        g = m.generate(p, max_new_tokens=4)
        rows.append({'id': cid, 'argmax': r['argmax'], 'gen': g, 'mass': r['letter_mass'], 'finite': r['finite'], 'n_tokens': r['n_tokens'],
                     'correct': row_for(cid, v, 'none', 'text', r, 'val', 'base', MODEL)['correct']})
        print(rows[-1], flush=True)
    v, img = ann[items[0]], scene(items[0]); no = len(permuted_options(v)); p = build(v, img, 'L1_app_low_tr0', 'speech')
    o0, r0 = m.score_many([p], no, capture=True)
    Ls = [m.n_layers // 4, m.n_layers // 2, m.n_layers - 2]   # hook checked at three depths (reviewer)
    m.set_ablation(r0[0], 1.0, Ls); o1, r1 = m.score_many([p], no, capture=True)
    m.set_ablation(None); o3, _ = m.score_many([p], no)
    ratio = max(abs(float(r1[0][L] @ (r0[0][L] / np.linalg.norm(r0[0][L])))) / abs(float(r0[0][L] @ (r0[0][L] / np.linalg.norm(r0[0][L])))) for L in Ls)
    L = Ls[1]; u = r0[0][L] / np.linalg.norm(r0[0][L])
    first = lambda g: re.sub(r'<[^>]*>', '', g).strip()[:1]
    S = {'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'n_layers': m.n_layers,
         'n_missing_keys': len(m.load_info['missing_keys']), 'n_unexpected_keys': len(m.load_info['unexpected_keys']),
         'mean_letter_mass': float(np.mean([r['mass'] for r in rows])), 'greedy_agree': float(np.mean([first(r['gen']) == r['argmax'] for r in rows])),
         'acc_no_remark_20': float(np.mean([r['correct'] for r in rows])), 'finite': all(r['finite'] for r in rows),
         'mean_tokens': float(np.mean([r['n_tokens'] for r in rows])),
         'hook_proj_before': float(r0[0][L] @ u), 'hook_proj_after': float(r1[0][L] @ u), 'hook_layers': Ls, 'hook_max_after_over_before': ratio,
         'hook_restored_dlp': float(max(abs(a - b) for a, b in zip(o0[0]['letter_logp'], o3[0]['letter_logp']))), 'rows': rows}
    S['pass'] = bool(S['mean_letter_mass'] > 0.5 and S['greedy_agree'] >= 0.95 and S['finite'] and S['n_missing_keys'] == 0
                     and S['hook_max_after_over_before'] < 0.02 and S['hook_restored_dlp'] < 0.05)
    (R / 'check.json').write_text(json.dumps(S, indent=1))
    print('CHECK', json.dumps({k: v for k, v in S.items() if k != 'rows'}), flush=True)
    if not S['pass']:
        stop('model check failed (results/gemma/check.json): letters, greedy agreement, missing weights or the steering hook. Claude looks first.')


def key_word(cue):
    for w in ('untrustworthy', 'trustworthy', 'dishonest', 'honest', 'cannot be trusted', 'can be trusted'):
        if w in cue:
            return w


def perc():
    m, ann = model(), load_annotations()
    items = sorted(load_splits()['pool']['val'], key=lambda k: seeded_int('perc-v1', k))[:30]
    conds = [f'{lv}_app_{s}_tr{i}' for lv in LEVELS for i in range(3) for s in ('low', 'high')]
    out = R / 'perception.jsonl'
    done = {(r['id'], r['channel']) for r in read_rows(out)}
    norm = lambda s: ' '.join(re.sub(r'[^a-z ]', ' ', s.lower().replace('can not', 'cannot').replace("can't", 'cannot')).split())
    with open(out, 'a') as fh:
        for j, cid in enumerate(items):
            v, img, cond = ann[cid], scene(cid), conds[j % len(conds)]
            kw = key_word(CUES[cond])
            for ch in ('speech', 'text'):
                if (cid, ch) in done:
                    continue
                if timeup():
                    sys.exit(75)
                g = m.generate(build_repeat(v, img, cond, ch), max_new_tokens=60)
                hit = re.search(r'\b' + norm(kw) + r'\b', norm(re.sub(r'<[^>]*>', ' ', g))) is not None
                fh.write(json.dumps({'id': cid, 'channel': ch, 'cond': cond, 'cue': CUES[cond], 'key': kw, 'gen': g, 'hit': hit}) + '\n'); fh.flush()
                print(ch, hit, kw, '|', g[:140].replace('\n', ' '), flush=True)
    rows = read_rows(out)
    S = {ch: float(np.mean([r['hit'] for r in rows if r['channel'] == ch])) for ch in ('speech', 'text')}
    (R / 'perception.json').write_text(json.dumps(S, indent=1)); print('PERCEPTION', S, flush=True)
    if S['speech'] < 0.9:
        stop(f"the model repeats the spoken remark correctly in only {S['speech']:.2f} of scenes (need 0.90): it may not hear it. Yash decides.")


def g2_cells():
    text = [('none', 'text')] + [(c, 'text') for lv in LEVELS for c in level_conds(lv)] + [(f'val_{s}{i}', 'text') for s in ('neg', 'pos') for i in range(3)]
    speech = [(c, 'speech') for lv in LEVELS for c in level_conds(lv)]
    return text + speech


def diff_keys():
    """name -> list of (cond_a, cond_b, channel); per-scene difference = mean over pairs of h[a] - h[b] at the last token."""
    K = {}
    for lv in LEVELS:
        for ch in ('text', 'speech'):
            K[f'app_{ch}_{lv}'] = [(f'{lv}_app_low_tr{i}', f'{lv}_app_high_tr{i}', ch) for i in range(3)]
            K[f'beh_{ch}_{lv}'] = [(f'{lv}_beh_neg_tr{i}', f'{lv}_beh_pos_tr{i}', ch) for i in range(2)]
    K['val_text'] = [(f'val_neg{i}', f'val_pos{i}', 'text') for i in range(3)]
    return K


def g2():
    """Unsteered rows on the 300 gate scenes plus per-scene difference vectors (float16 npz, one per scene)."""
    m, ann = model(), load_annotations()
    items = gate_items(load_splits(), load_item_table(), 300)[:N_GATE]
    cells, K = g2_cells(), diff_keys()
    out, adir = R / 'g2_rows.jsonl', R / 'acts'
    adir.mkdir(exist_ok=True)
    done = defaultdict(set)
    for r in read_rows(out):
        done[r['id']].add((r['condition'], r['channel']))
    t0 = time.time()
    with open(out, 'a') as fh:
        for i, cid in enumerate(items):
            if len(done[cid]) == len(cells) and (adir / f'{cid}.npz').exists():
                continue
            if timeup():
                print(f'g2: time limit at scene {i}/{len(items)}', flush=True); sys.exit(75)
            v, img = ann[cid], scene(cid)
            outs, res = m.score_many([build(v, img, c, ch) for c, ch in cells], len(permuted_options(v)), capture=True)
            pos = {c: j for j, c in enumerate(cells)}
            D = {k: np.mean([res[pos[(a, ch)]] - res[pos[(b, ch)]] for a, b, ch in ps], 0).astype(np.float16) for k, ps in K.items()}
            np.savez(adir / f'{cid}.npz.tmp.npz', **D); os.replace(adir / f'{cid}.npz.tmp.npz', adir / f'{cid}.npz')
            fh.write(''.join(json.dumps(row_for(cid, v, c, ch, r, 'train', 'base', MODEL)) + '\n' for (c, ch), r in zip(cells, outs)
                             if (c, ch) not in done[cid])); fh.flush()
            if (i + 1) % 10 == 0:
                print(f'g2: {i + 1}/{len(items)} scenes, {(time.time() - t0) / (i + 1):.1f} s/scene', flush=True)
    print('g2 finished', flush=True)


def sweep(lv):
    pool, clean = val_sets()
    m = model()
    u = np.load(R / 'directions_gemma.npz')[f'app_text_{lv}']
    out = R / f'val_{lv}.jsonl'
    cells = [('none', 'text')] + [(c, 'text') for c in level_conds(lv)]
    meths = [('base', None, 0, ())] + [(f'naive_{nm}_a{al}', u, al, ls) for nm, ls in layer_sets(m.n_layers).items() for al in ALPHAS]
    for nm, d, al, ls in meths:
        m.set_ablation(d, al, ls)
        if not (grid(out, nm, pool, cells, 'val') and grid(out, nm, clean, [('none', 'text')], 'val')):
            m.set_ablation(None); sys.exit(75)
    m.set_ablation(None); print(f'sweep {lv} finished', flush=True)


def rand_dirs(shape, i):
    return np.random.default_rng(1000 + i).standard_normal(shape)


CONTROLS = ('speechdir', 'valence', 'shuffled')   # same layers and strength as the chosen setting


def control_dir(name, lv):
    D = np.load(R / 'directions_gemma.npz')
    return {'speechdir': D[f'app_speech_{lv}'], 'valence': D['val_text'], 'shuffled': D[f'shuf_text_{lv}']}[name]


def transfer(lv):
    P = json.loads((R / f'select_{lv}.json').read_text())
    pool, clean = val_sets()
    m = model()
    u = np.load(R / 'directions_gemma.npz')[f'app_text_{lv}']
    out = R / f'val_{lv}.jsonl'
    sp = [(c, 'speech') for c in level_conds(lv)]
    tx = [('none', 'text')] + [(c, 'text') for c in level_conds(lv)]
    jobs = [('base', None, 0, (), None, sp, False), (P['chosen'], u, P['alpha'], P['layers'], None, sp, False),
            ('constitution', None, 0, (), constitution(), tx + sp, True)]
    jobs += [('speechdir', control_dir('speechdir', lv), P['alpha'], P['layers'], None, sp, False)]
    jobs += [(c, control_dir(c, lv), P['alpha'], P['layers'], None, tx + sp, True) for c in ('valence', 'shuffled')]
    jobs += [(f'rand{i}', rand_dirs(u.shape, i), P['alpha'], P['layers'], None, tx + sp, True) for i in range(N_RAND)]
    for nm, d, al, ls, system, cells, do_clean in jobs:
        m.set_ablation(d, al, ls)
        ok = grid(out, nm, pool, cells, 'val', system) and (not do_clean or grid(out, nm, clean, [('none', 'text')], 'val', system))
        if not ok:
            m.set_ablation(None); sys.exit(75)
    m.set_ablation(None); print(f'transfer {lv} finished', flush=True)


FACE_CONDS = ('face_trust_high', 'face_trust_low', 'face_null_a', 'face_null_b')


def faces():
    """Exploratory vision arm. Scene plus a second image with an OMI face; prompt = the midterm portrait prompt, no remark."""
    from PIL import Image
    from data import prompt_text
    omi = Path(os.environ.get('OMI_DIR', str(Path(os.environ.get('P777', '.')) / 'data/omi')))
    pairs = json.loads((HERE / 'results/omi_pairs.json').read_text())
    def assign(cid):
        t = pairs['trust_pairs'][seeded_int('omi-trust-v1', cid) % len(pairs['trust_pairs'])]
        n = pairs['null_pairs'][seeded_int('omi-null-v1', cid) % len(pairs['null_pairs'])]
        return {'face_trust_high': t['high'], 'face_trust_low': t['low'], 'face_null_a': n['a'], 'face_null_b': n['b']}
    face = lambda s: Image.open(omi / 'images' / f'{s}.jpg').convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
    pool, _ = val_sets(); m, ann = model(), load_annotations()
    lv = 'L1'
    P = json.loads((R / f'select_{lv}.json').read_text()) if (R / f'select_{lv}.json').exists() else None
    u = np.load(R / 'directions_gemma.npz')[f'app_text_{lv}']
    meths = [('base', None, 0, (), None), ('constitution', None, 0, (), constitution())]
    if P:
        meths.insert(1, (P['chosen'], u, P['alpha'], P['layers'], None))
    out = R / 'faces.jsonl'
    done = {(r['id'], r['method'], r['condition']) for r in read_rows(out)}
    with open(out, 'a') as fh:
        for nm, d, al, ls, system in meths:
            m.set_ablation(d, al, ls)
            for i, cid in enumerate(pool):
                todo = [c for c in FACE_CONDS if (cid, nm, c) not in done]
                if not todo:
                    continue
                if timeup():
                    m.set_ablation(None); sys.exit(75)
                v, img, fs = ann[cid], scene(cid), assign(cid)
                prompts = [[('image', img), ('image', face(fs[c])), ('text', prompt_text(v, '', portrait=True))] for c in todo]
                outs, _ = m.score_many(prompts, len(permuted_options(v)), system)
                fh.write(''.join(json.dumps(dict(row_for(cid, v, c, 'face', r, 'val', nm, MODEL), face=fs[c])) + '\n' for c, r in zip(todo, outs))); fh.flush()
            print(f'faces {nm} finished', flush=True)
    m.set_ablation(None)


# ---- CPU stages --------------------------------------------------------------------------------------------------
def gate():
    ix = index(read_rows(R / 'g2_rows.jsonl'))['base']
    S = {}
    for lv in LEVELS:
        for ch in ('text', 'speech'):
            s = summarize({c: item_stats(d, lv, ch) for c, d in ix.items()})
            s['pass'] = bool(s['app_minus_floor'] and s['app_minus_floor'][1] > 0)
            S[f'{ch} {lv}'] = s
            print(f"GATE {ch:6s} {lv} n={s['n_s_app']:3d} app flips {pct(s['s_app'])} floor {pct(s['s0'])} app-floor {pct(s['app_minus_floor'])} "
                  f"behavior {pct(s['r_beh'])} acc(no remark) {pct(s['acc'])} -> {'PASS' if s['pass'] else 'FAIL'}", flush=True)
        S[f'go {lv}'] = S[f'text {lv}']['pass'] and S[f'speech {lv}']['pass']
    S['rule'] = 'appearance flips minus noise floor, 95% range above 0, on the 300 training gate scenes; a level goes on only if text and speech both pass'
    (R / 'gate.json').write_text(json.dumps(S, indent=1))
    print('GATE levels going on:', [lv for lv in LEVELS if S[f'go {lv}']], flush=True)
    if not any(S[f'go {lv}'] for lv in LEVELS):
        stop('no level shows appearance bias above the noise floor in both typed and spoken remarks; nothing to transfer. Yash decides.')


def levels_on():
    g = json.loads((R / 'gate.json').read_text())
    return [lv for lv in LEVELS if g[f'go {lv}']]


def directions():
    files = sorted((R / 'acts').glob('*.npz'))
    assert len(files) == N_GATE, f'{len(files)} scenes with difference vectors (need {N_GATE})'
    ids = [f.name[:-4] for f in files]
    order = sorted(range(len(ids)), key=lambda i: seeded_int('split-half-v1', ids[i]))
    halves = order[::2], order[1::2]
    keys = list(np.load(files[0]).keys())
    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    cos = lambda a, b: (unit(a) * unit(b)).sum(-1)
    dirs, stab = {}, {}
    for k in keys:
        X = np.stack([np.load(f)[k].astype(np.float32) for f in files])
        assert np.isfinite(X).all() and (np.abs(X).sum((1, 2)) > 0).all(), k
        dirs[k] = X.mean(0)
        stab[k] = cos(X[list(halves[0])].mean(0), X[list(halves[1])].mean(0)).round(4).tolist()
    for lv in LEVELS:   # shuffled-label control: each scene's untrustworthy-minus-trustworthy difference gets a random sign
        X = np.stack([np.load(f)[f'app_text_{lv}'].astype(np.float32) for f in files])
        sg = np.array([1.0 if seeded_int(f'shuffle-{lv}-v1', i) % 2 else -1.0 for i in ids], np.float32)
        dirs[f'shuf_text_{lv}'] = (X * sg[:, None, None]).mean(0)
    np.savez(R / 'directions_gemma.npz', **dirs)
    n = dirs[keys[0]].shape[0]; late = slice(round(n * 0.4), n)
    C = {}
    for lv in LEVELS:
        C[f'app text-speech {lv}'] = cos(dirs[f'app_text_{lv}'], dirs[f'app_speech_{lv}']).round(4).tolist()
        C[f'app-beh text {lv}'] = cos(dirs[f'app_text_{lv}'], dirs[f'beh_text_{lv}']).round(4).tolist()
        C[f'app text - valence {lv}'] = cos(dirs[f'app_text_{lv}'], dirs['val_text']).round(4).tolist()
    C['app text L1-L2'] = cos(dirs['app_text_L1'], dirs['app_text_L2']).round(4).tolist()
    for lv in LEVELS:
        C[f'app text - shuffled {lv}'] = cos(dirs[f'app_text_{lv}'], dirs[f'shuf_text_{lv}']).round(4).tolist()
    (R / 'directions.json').write_text(json.dumps({'cos': C, 'split_half': stab, 'norm': {k: np.linalg.norm(v, axis=-1).round(3).tolist() for k, v in dirs.items()}}))
    for k, v in C.items():
        print(f'DIRECTIONS {k:24s} mean over layers {late.start}-{n - 1}: {np.mean(v[late]):.2f}', flush=True)
    for k, v in stab.items():
        print(f'stability {k:16s} min over layers {late.start}-{n - 1}: {min(v[late]):.2f}', flush=True)


def select(lv):
    ix = index(read_rows(R / f'val_{lv}.jsonl'))
    T = {m: summarize({c: item_stats(d, lv, 'text') for c, d in it.items()}) for m, it in ix.items() if m == 'base' or m.startswith('naive_')}
    ls = layer_sets(48)
    parse = lambda m: (lambda x: (float(x[1]), len(ls[x[0]])))(m[len('naive_'):].rsplit('_a', 1))
    cands = sorted(m for m in T if m.startswith('naive_'))
    assert len(cands) == len(ls) * len(ALPHAS), cands
    best, flag = choose(T, T['base'], cands, parse)
    for m in ['base'] + cands:
        t = T[m]; print(f"SWEEP {lv} {m:20s} acc {pct(t['acc'])} app flips {pct(t['s_app'])} floor {pct(t['s0'])} behavior {pct(t['r_beh'])}" + ('  <- chosen' if m == best else ''), flush=True)
    nm, al = best[len('naive_'):].rsplit('_a', 1)
    P = {'rule': 'lowest typed appearance flips with accuracy >= base - 2 points and behavior flips >= 80% of base; ties: smaller strength, fewer layers',
         'chosen': best, 'layers': ls[nm], 'alpha': float(al), 'direction': f'app_text_{lv}', 'fallback_used': flag,
         'acc_min': T['base']['acc'][0] - 0.02, 'r_beh_min': 0.8 * T['base']['r_beh'][0], 'selected_on': 'typed remarks, validation; speech not used'}
    (R / f'select_{lv}.json').write_text(json.dumps({**P, 'table': T}, indent=1)); print('SELECTED', lv, json.dumps(P), flush=True)


def paired(st_m, st_b, k, common):
    byv = defaultdict(list)
    for c in common:
        byv[c.split('_')[0]].append(st_m[c][k] - st_b[c][k])
    return boot(byv)


def boot_ratio(num, den, common, n=5000, seed=0):
    """Transfer ratio = (drop with the text direction) / (drop with the speech direction), bootstrap over source videos."""
    byv = defaultdict(list)
    for c in common:
        byv[c.split('_')[0]].append((num[c], den[c]))
    vids = list(byv); A = np.array([sum(x for x, _ in byv[v]) for v in vids]); B = np.array([sum(y for _, y in byv[v]) for v in vids])
    idx = np.random.default_rng(seed).integers(0, len(vids), (n, len(vids)))
    with np.errstate(divide='ignore', invalid='ignore'):
        bs = A[idx].sum(1) / B[idx].sum(1)
    pt = A.sum() / B.sum() if B.sum() else float('nan')
    return [round(float(pt), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


def layer_figure():
    J = json.loads((R / 'directions.json').read_text())['cos']
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, st in ((f'app text-speech L1', '-'), (f'app text-speech L2', '--'), ('app-beh text L1', ':'), ('app text - valence L1', '-.'), ('app text - shuffled L1', (0, (1, 3)))):
        if k in J:
            ax.plot(J[k], linestyle=st, label=k.replace('app', 'appearance').replace('beh', 'behavior'))
    ax.set_xlabel('layer'); ax.set_ylabel('cosine'); ax.set_ylim(-0.2, 1.0); ax.axhline(0, lw=0.5, color='gray'); ax.legend(fontsize=7, frameon=False)
    ax.set_title('Gemma 4 12B: where typed and spoken appearance directions agree', fontsize=9)
    fig.tight_layout(); fig.savefig(R / 'fig_layers.png', dpi=200); plt.close(fig)


def face_summary():
    rows = read_rows(R / 'faces.jsonl')
    if not rows:
        return None, []
    ix = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        ix[r['method']][r['id']][r['condition']] = r
    S, md = {}, ['## Faces (exploratory vision arm, validation, L1 steering setting)', '',
                 '| Method | n | Trust-pair flips | Equal-trust flips | Face sensitivity | Change vs base |', '|---|---|---|---|---|---|']
    per = {}
    for m, it in ix.items():
        per[m] = {c: {'t': float(d['face_trust_high']['pred'] != d['face_trust_low']['pred']), 'n': float(d['face_null_a']['pred'] != d['face_null_b']['pred'])}
                  for c, d in it.items() if all(k in d for k in FACE_CONDS)}
        for c in per[m]:
            per[m][c]['s'] = per[m][c]['t'] - per[m][c]['n']
    for m in ['base'] + sorted(x for x in per if x != 'base'):
        if m not in per or 'base' not in per:
            continue
        common = set(per[m]) & set(per['base'])
        g = lambda k: summarize({c: {k: per[m][c][k]} for c in common})[k]
        o = {'n': len(common), 'trust_flips': g('t'), 'null_flips': g('n'), 's_face': g('s')}
        if m != 'base':
            o['d_s_face'] = paired({c: {'s': per[m][c]['s']} for c in common}, {c: {'s': per['base'][c]['s']} for c in common}, 's', common)
        S[m] = o
        md.append(f"| {m} | {o['n']} | {pct(o['trust_flips'])} | {pct(o['null_flips'])} | {pct(o['s_face'])} | {pct(o.get('d_s_face'))} |")
    if 'base' in S:
        S['face_bias_exists'] = bool(S['base']['s_face'][1] > 0)
        md += ['', f"Face bias above the equal-trust floor: **{'yes' if S['face_bias_exists'] else 'no'}** (if no, the steering rows are not informative)", '']
    return S, md


def report():
    """Success rule (QWEN_TRANSFER_PLAN.md, speech in place of image; reviewer changes approved Nov 25 23:55 CT), per level; L1 main.
    Rule 3 uses the valence and shuffled-label directions (random directions remove almost nothing in 3,840 dimensions; reported only).
    Rule 4a uses accuracy with a spoken neutral remark; accuracy with no remark is reported too."""
    S, md = {}, ['# Gemma 4 12B: steering built from typed remarks, applied to spoken remarks (validation)', '']
    for lv in levels_on():
        if not (R / f'select_{lv}.json').exists():
            continue
        P = json.loads((R / f'select_{lv}.json').read_text())
        ix = index(read_rows(R / f'val_{lv}.jsonl'))
        steer = P['chosen']
        ctrl = ['valence', 'shuffled'] + [f'rand{i}' for i in range(N_RAND)]
        meths = ['base', steer, 'speechdir', 'constitution'] + ctrl
        if any(m not in ix for m in meths):
            print(f'REPORT {lv}: missing methods', [m for m in meths if m not in ix], flush=True); continue
        out = {'chosen': steer, 'layers': P['layers'], 'alpha': P['alpha']}
        md += [f'## {lv}{" (main)" if lv == "L1" else ""}: chosen setting {steer} (layers {P["layers"]}, strength {P["alpha"]})', '',
               '| Format | Method | Acc, no remark | Acc, neutral remark | Appearance flips | Floor | Behavior flips | Change in appearance flips vs base |',
               '|---|---|---|---|---|---|---|---|']
        for ch in ('text', 'speech'):
            ms = [m for m in meths if not (ch == 'text' and m == 'speechdir')]
            st = {m: {c: item_stats(d, lv, ch) for c, d in ix[m].items()} for m in ms}
            common = set.intersection(*[{c for c, x in st[m].items() if x['s_app'] is not None and x['r_beh'] is not None and x['s0'] is not None and x['acc_neutral'] is not None} for m in ms])
            acc0 = {m: summarize({c: {'acc': float(d[('none', 'text')]['correct'])} for c, d in ix[m].items() if ('none', 'text') in d}) for m in ms if m != 'speechdir'}
            for m in ms:
                sm = summarize({c: st[m][c] for c in common})
                o = {'n': len(common), 'acc_no_remark': acc0.get(m, {}).get('acc'), **{k: sm[k] for k in ('s_app', 's0', 'r_beh', 'app_minus_floor', 'acc_neutral')}}
                if m != 'base':
                    for k in ('s_app', 'r_beh', 'acc_neutral'):
                        o[f'd_{k}'] = paired(st[m], st['base'], k, common)
                if m in ctrl:
                    o['steer_minus_ctrl_s_app'] = paired(st[steer], st[m], 's_app', common)
                out[f'{ch} {m}'] = o
                md.append(f"| {ch} | {m} | {pct(o['acc_no_remark'])} | {pct(o['acc_neutral'])} | {pct(o['s_app'])} | {pct(o['s0'])} | {pct(o['r_beh'])} | {pct(o.get('d_s_app'))} |")
            if ch == 'speech':
                num = {c: st['base'][c]['s_app'] - st[steer][c]['s_app'] for c in common}
                den = {c: st['base'][c]['s_app'] - st['speechdir'][c]['s_app'] for c in common}
                out['transfer_ratio'] = boot_ratio(num, den, common)
        b, s_ = out['speech base'], out[f'speech {steer}']
        rule = {'1 bias exists in speech (appearance flips minus floor, range above 0)': b['app_minus_floor'][1] > 0,
                '2 steering lowers spoken appearance flips (paired, range below 0)': s_['d_s_app'][2] < 0,
                '3 beats the valence and shuffled-label directions (steered minus control, range below 0)': all(out[f'speech {c}']['steer_minus_ctrl_s_app'][2] < 0 for c in ('valence', 'shuffled')),
                '4a accuracy with a spoken neutral remark within 2 points of base': s_['acc_neutral'][0] >= b['acc_neutral'][0] - 0.02,
                '4b spoken behavior flips at least 80% of base': s_['r_beh'][0] >= 0.8 * b['r_beh'][0]}
        k1, k2 = list(rule)[:2]
        verdict = 'SUCCESS' if all(rule.values()) else 'PARTIAL' if rule[k1] and rule[k2] else 'NOT SUCCESSFUL'
        out['rule'] = rule; out['verdict'] = verdict; S[lv] = out
        tr = out['transfer_ratio']
        md += ['', f"**Transfer ratio** (drop in spoken appearance flips with the text-built direction, divided by the drop with a direction built from speech; 1 = full transfer): {tr[0]} [{tr[1]}, {tr[2]}]. Spoken-direction drop: {pct(out['speech speechdir']['d_s_app'])}.", '',
               'Random directions (reported, not in the rule): steered minus random ' + ', '.join(pct(out[f'speech rand{i}']['steer_minus_ctrl_s_app']) for i in range(N_RAND)), '',
               'Success rule (speech):', ''] + [f'- {k}: **{"yes" if v else "no"}**' for k, v in rule.items()] + ['', f'**{lv}: {verdict}**', '']
        print(f'REPORT {lv}: {verdict}', json.dumps({k: bool(v) for k, v in rule.items()}), 'transfer ratio', tr, flush=True)
    F, fmd = face_summary()
    if F:
        S['faces'] = F; md += fmd
    try:
        layer_figure(); md += ['![layers](fig_layers.png)', '']
    except Exception as ex:
        print('figure skipped:', ex, flush=True)
    (R / 'report.json').write_text(json.dumps(S, indent=1, default=bool)); (R / 'report.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    st = sys.argv[1]
    if st in ('sweep', 'select', 'transfer'):
        lvs = sys.argv[2:] or levels_on()
        for lv in lvs:
            if lv in levels_on():
                globals()[st](lv)
            else:
                print(f'{st} {lv}: skipped (gate did not pass at this level)', flush=True)
    else:
        globals()[st]()
