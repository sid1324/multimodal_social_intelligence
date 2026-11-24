"""Grid runner: (item x method x condition x channel), one gzip-jsonl row per forward pass, resumable, optional residuals.
Also the validation sweep (Stage 3) and DPO checkpoint evaluation (Stage 4). Test runs are not implemented yet on
purpose: data.load_splits() hides the test ids until Yash has reviewed FREEZE.md.

Usage: python src/evaluate.py sweep --model nemotron --level L2
       python src/evaluate.py dpo_val --model nemotron --level L2
"""
import argparse, gzip, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, P777, load_annotations, load_splits, permuted_options, video_of, norms_of_gold
from channels import build, scene

CACHE = Path(P777) / 'cache'
# stop cleanly 3 min before the Slurm walltime (or after STAGE_HOURS outside Slurm)
DEADLINE = (float(os.environ['SLURM_JOB_END_TIME']) - 180) if 'SLURM_JOB_END_TIME' in os.environ else time.time() + 3600 * float(os.environ.get('STAGE_HOURS', '0.83'))


def level_conds(lv, families=('neutral', 'app_tr', 'beh_tr')):
    out = []
    for f in families:
        if f == 'neutral':
            out += [f'{lv}_neutral', f'{lv}_neutral_paraphrase']
        elif f.startswith('app_'):
            out += [f'{lv}_app_{s}_{f[4:]}{i}' for s in ('low', 'high') for i in range(3 if f.endswith('tr') else 2)]
        elif f.startswith('beh_'):
            out += [f'{lv}_beh_{s}_{f[4:]}{i}' for s in ('neg', 'pos') for i in range(2)]
    return out


def row_for(cid, v, cond, channel, res, split, method, model):
    perm = permuted_options(v)
    logp = {str(i): res['letter_logp'][j] for j, (_, i, _) in enumerate(perm)}
    pred = next(i for l, i, _ in perm if l == res['argmax'])
    return {'model': model, 'id': cid, 'video': video_of(cid), 'split': split, 'method': method, 'channel': channel,
            'condition': cond, 'logp': logp, 'pred': pred, 'gold': v['correct'], 'correct': pred == v['correct'],
            'order': [i for _, i, _ in perm], 'letter_mass': res['letter_mass'], 'top_token': res['top_token'],
            'finite': res['finite'], 'n_tokens': res['n_tokens'], 'norms': norms_of_gold(v)}


def read_rows(path):
    rows = []
    if Path(path).exists():
        op = gzip.open if str(path).endswith('.gz') else open
        try:
            with op(path, 'rt') as fh:
                for line in fh:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
        except (EOFError, gzip.BadGzipFile):
            pass
    return rows


def key(r):
    return r['id'], r['method'], r['channel'], r['condition']


def run_grid(m, path, method, items, cells, split, system=None, acts=None, chunk=0):
    """cells: list of (condition, channel). acts: optional dict(path=, cells=[...]) -> float32 memmap
    [items, cells, layers, H]. Plain jsonl (append, flushed per item) so an interrupted job loses at most one item.
    Returns True if finished, False if it stopped at the deadline."""
    ann = load_annotations()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.read_bytes().endswith(b'\n') and path.stat().st_size:
        good = path.read_bytes().rsplit(b'\n', 1)[0] + b'\n'   # drop one partial line from an interrupted write
        path.write_bytes(good)
    done = {key(r) for r in read_rows(path)}
    mm = None
    if acts:
        ap = Path(acts['path']); ap.parent.mkdir(parents=True, exist_ok=True)
        shape = (len(items), len(acts['cells']), m.n_layers, m._w32.shape[1])
        mm = np.lib.format.open_memmap(ap, mode='r+' if ap.exists() else 'w+', dtype=np.float32, shape=shape)
        Path(str(ap) + '.index.json').write_text(json.dumps({'items': items, 'cells': acts['cells']}))
        pos = {tuple(c): j for j, c in enumerate(acts['cells'])}
    t0, n = time.time(), 0
    with open(path, 'a') as fh:
        for i, cid in enumerate(items):
            todo = [c for c in cells if (cid, method, c[1], c[0]) not in done]
            if not todo:
                continue
            if time.time() > DEADLINE:
                print(f'deadline reached at item {i}/{len(items)}', flush=True)
                return False
            v, img = ann[cid], scene(cid)
            n_opts = len(permuted_options(v))
            outs, resid = m.score_many([build(v, img, c, ch) for c, ch in todo], n_opts, system, capture=mm is not None)
            buf = ''
            for j, ((c, ch), res) in enumerate(zip(todo, outs)):
                if mm is not None and (c, ch) in pos:
                    mm[i, pos[(c, ch)]] = resid[j]
                buf += json.dumps(row_for(cid, v, c, ch, res, split, method, m.name)) + '\n'
            fh.write(buf); fh.flush()
            n += len(todo)
            if (i + 1) % 20 == 0:
                if mm is not None:
                    mm.flush()
                print(f'{method}: {i + 1}/{len(items)} items, {n} passes, {(time.time() - t0) / max(n, 1):.3f} s/pass', flush=True)
    if mm is not None:
        mm.flush()
    print(f'{method}: finished, {n} new passes in {time.time() - t0:.0f}s', flush=True)
    return True


def layer_sets(n):
    """Midterm rule: single layers at about 25/40/55/70% depth plus all layers from about 40% depth to the end.
    (Midterm, 36 layers: 9, 14, 20, 25, 14-35.)"""
    s = {f'L{round(n * f)}': [round(n * f)] for f in (0.25, 0.40, 0.55, 0.70)}
    a = round(n * 0.40)
    s[f'L{a}-{n - 1}'] = list(range(a, n))
    return s


def sweep(a):
    from omni_model import load
    sp = load_splits()
    pool = sp['pool']['val']
    clean = [k for k in sp['clean']['val'] if k not in set(pool)]
    cells = [('none', 'text')] + [(c, 'text') for c in level_conds(a.level)]
    out = HERE / f'results/sweep/{a.model}_rows.jsonl'
    d = np.load(CACHE / f'directions_{a.model}.npz')
    u = d[f'app_text_{a.level}']
    m = load(a.model)
    methods = [('base', None, 0, ())]
    for nm, layers in layer_sets(m.n_layers).items():
        for al in (0.5, 1.0):
            methods.append((f'naive_{nm}_a{al}', u, al, layers))
    for nm, dirs, al, layers in methods:
        m.set_ablation(dirs, al, layers)
        ok = run_grid(m, out, nm, pool, cells, 'val') and run_grid(m, out, nm, clean, [('none', 'text')], 'val')
        if not ok:
            sys.exit(75)   # resumable: submit again
    print('SWEEP DONE', flush=True)


def dpo_val(a):
    from omni_model import load
    from dpo_train import adapter_dirs
    from peft import PeftModel
    sp = load_splits()
    pool = sp['pool']['val']
    clean = [k for k in sp['clean']['val'] if k not in set(pool)]
    chans = ('text', 'speech', 'image')
    cells = [('none', 'text')] + [(c, ch) for ch in chans for c in level_conds(a.level)]
    out = HERE / f'results/dpo/{a.model}_val_rows.jsonl'
    m = load(a.model)
    if not run_grid(m, out, 'base', pool, cells, 'val') or not run_grid(m, out, 'base', clean, [('none', 'text')], 'val'):
        sys.exit(75)
    pm = None
    for name, path in adapter_dirs(a.model):
        if pm is None:
            pm = PeftModel.from_pretrained(m.lm if a.model == 'nemotron' else m.model, path, adapter_name=name)
        else:
            pm.load_adapter(path, adapter_name=name)
        pm.set_adapter(name); pm.eval()
        # LoRA modules are injected in place, so m.lm / m.model now run with the active adapter
        if not run_grid(m, out, name, pool, cells, 'val') or not run_grid(m, out, name, clean, [('none', 'text')], 'val'):
            sys.exit(75)
    print('DPO VAL DONE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['sweep', 'dpo_val'])
    ap.add_argument('--model', default='nemotron'); ap.add_argument('--level', default='L2')
    a = ap.parse_args()
    {'sweep': sweep, 'dpo_val': dpo_val}[a.phase](a)
