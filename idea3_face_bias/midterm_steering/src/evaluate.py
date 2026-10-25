"""Run (item x condition x method) grids and write one row per run.

Phases
  sweep: validation split. Methods: base, constitution, and 20 ablation configs
         (direction in {naive=u_app, refined=u_perp}) x (layer set) x (alpha in {0.5, 1.0}).
         Cue items: 13 train-wording conditions. Other items with a non-empty gold: 'none' only (clean accuracy).
  test:  test split, run once after results/FREEZE.json exists. Methods: base, constitution, naive, refined.
         Text arm: 21 conditions on cue items; clean accuracy on the rest; portrait arm (OMI faces as a
         second image): trust pair high/low and equal-trust control pair a/b on cue items.
Rows go to results/<phase>/rows.jsonl.gz (appended, resumable).

Usage: python src/evaluate.py --phase sweep|test [--fast] [--limit N]
"""
import argparse, gzip, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (HERE, P777, load_annotations, load_item_table, load_splits, load_cues, constitution,
                  permuted_options, prompt_text, image_path, video_of, norms_of_gold)
from model import Qwen
from portraits import assign, portrait
from run_gate import GATE_CONDS, row_for

CACHE = Path(os.environ.get('CACHE_DIR', f'{P777}/cache'))
N_LAYERS = 36
LAYER_SETS = {'L9': [9], 'L14': [14], 'L20': [20], 'L25': [25], 'L14-35': list(range(14, 36))}  # ~25/40/55/70% depth, 40%->end
ALPHAS = [0.5, 1.0]
TEST_TEXT_CONDS = (['none', 'neutral', 'neutral_paraphrase']
                   + [f'app_{s}_tr{i}' for s in ('low', 'high') for i in range(3)]
                   + [f'app_{s}_ho{i}' for s in ('low', 'high') for i in range(2)]
                   + [f'beh_{s}_ho{i}' for s in ('neg', 'pos') for i in range(2)]
                   + [f'state_{s}{i}' for s in ('neg', 'pos') for i in range(2)])
PORTRAIT_CONDS = ['face_trust_high', 'face_trust_low', 'face_null_a', 'face_null_b']


def sweep_methods():
    ms = [{'name': 'base'}, {'name': 'constitution', 'system': True}]
    for kind, key in (('naive', 'u_app'), ('refined', 'u_perp')):
        for ls, layers in LAYER_SETS.items():
            for a in ALPHAS:
                ms.append({'name': f'{kind}_{ls}_a{a}', 'kind': kind, 'dir': key, 'layers': layers, 'alpha': a})
    return ms


def test_methods():
    frz = json.loads((HERE / 'results/FREEZE.json').read_text())
    ms = [{'name': 'base'}, {'name': 'constitution', 'system': True}]
    for kind in ('naive', 'refined'):
        c = frz[kind]
        ms.append({'name': kind, 'kind': kind, 'dir': c['dir'], 'layers': c['layers'], 'alpha': c['alpha'], 'config': c['name']})
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['sweep', 'test'], required=True)
    ap.add_argument('--fast', action='store_true', help='batched shared-prefix scoring (only if verify passed)')
    ap.add_argument('--limit', type=int, default=0, help='debug: first N items per list')
    ap.add_argument('--methods', default='', help='comma-separated subset of method names')
    args = ap.parse_args()
    if args.fast:
        chk = json.loads((HERE / 'results/fast_check.json').read_text())
        assert chk['ok'], 'fast path did not pass verification'

    ann, table, splits, cues = load_annotations(), load_item_table(), load_splits(), load_cues()
    split = 'val' if args.phase == 'sweep' else 'test'
    pool = splits['pool'][split]
    clean_only = [k for k in splits['clean'][split] if k not in set(pool)]
    if args.limit:
        pool, clean_only = pool[:args.limit], clean_only[:args.limit]
    methods = sweep_methods() if args.phase == 'sweep' else test_methods()
    if args.methods:
        keep = set(args.methods.split(','))
        methods = [m for m in methods if m['name'] in keep]
    text_conds = GATE_CONDS if args.phase == 'sweep' else TEST_TEXT_CONDS
    dirs = np.load(CACHE / 'directions.npz')
    pairs = json.loads((HERE / 'results/omi_pairs.json').read_text()) if args.phase == 'test' else None
    system_text = constitution()

    outdir = HERE / f'results/{args.phase}'
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / 'rows.jsonl.gz'
    done = set()
    if path.exists():
        try:
            with gzip.open(path, 'rt') as fh:
                for line in fh:
                    r = json.loads(line)
                    done.add((r['id'], r['method'], r['condition']))
        except (EOFError, gzip.BadGzipFile):
            pass  # a truncated final member from an interrupted run; earlier rows were read
    n_total = len(methods) * (len(pool) * (len(text_conds) + (len(PORTRAIT_CONDS) if pairs else 0)) + len(clean_only))
    print(f'{args.phase}: {len(methods)} methods, {len(pool)} cue items, {len(clean_only)} clean-only items, '
          f'{n_total} runs, {len(done)} already done, fast={args.fast}', flush=True)

    q = Qwen()
    print(f'model on {torch.cuda.get_device_name(0)}, dtype {q.dtype}', flush=True)
    t0, n_run = time.time(), 0
    with gzip.open(path, 'at') as fh:
        for m in methods:
            q.set_ablation(dirs[m['dir']] if 'dir' in m else None, m.get('alpha', 1.0), m.get('layers', ()))
            system = system_text if m.get('system') else None
            tm = time.time()
            for cid in pool + clean_only:
                v = ann[cid]
                n_opts = len(permuted_options(v))
                conds = text_conds if cid in pool else ['none']
                todo = [c for c in conds if (cid, m['name'], c) not in done]
                scene = None
                if todo:
                    scene = Image.open(image_path(cid)).convert('RGB')
                    outs, _ = q.score_many([scene], [prompt_text(v, cues[c]) for c in todo], n_opts, system, fast=args.fast and len(todo) > 1)
                    for c, res in zip(todo, outs):
                        r = row_for(cid, v, c, res, split, m['name'])
                        r['arm'] = 'text'
                        fh.write(json.dumps(r) + '\n')
                    n_run += len(todo)
                if pairs and cid in pool:
                    tp, npair = assign(cid, pairs)
                    faces = {'face_trust_high': tp['high'], 'face_trust_low': tp['low'], 'face_null_a': npair['a'], 'face_null_b': npair['b']}
                    for c in PORTRAIT_CONDS:
                        if (cid, m['name'], c) in done:
                            continue
                        scene = scene or Image.open(image_path(cid)).convert('RGB')
                        res, _ = q.score([scene, portrait(faces[c])], prompt_text(v, '', portrait=True), n_opts, system)
                        r = row_for(cid, v, c, res, split, m['name'])
                        r.update(arm='portrait', face=faces[c],
                                 trust_pair=f"{tp['high']}|{tp['low']}", null_pair=f"{npair['a']}|{npair['b']}")
                        fh.write(json.dumps(r) + '\n')
                        n_run += 1
            fh.flush()
            el = time.time() - t0
            print(f"method {m['name']} done in {time.time() - tm:.0f}s; total {n_run} runs, {el / max(n_run, 1):.3f} s/run, "
                  f"elapsed {el / 60:.1f} min", flush=True)
    print(f'{args.phase.upper()} DONE: {n_run} runs in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
