"""First GPU job (train split only).
1) Correctness check on the first n_check gate items (no cue): finite logits, greedy letter == scored
   argmax, accuracy, letter probability mass, speed, visual tokens, GPU memory. Stops if it fails.
2) Gate: every gate item x 13 train-wording conditions, scored with the base model. Residuals at the
   last prompt token for all layers are saved for direction estimation (Step 3a).

Usage: python src/run_gate.py [--n_gate 300] [--n_check 20]
Env: OUT_DIR (default $P777/outputs), CACHE_DIR (default $P777/cache)
Outputs: results/gate/check.json, results/gate/gate_rows.jsonl (committed, small);
         $CACHE_DIR/acts/gate_resid.npy (+ index json), not committed.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (HERE, P777, load_annotations, load_item_table, load_splits, gate_items, load_cues,
                  permuted_options, prompt_text, image_path, video_of, norms_of_gold)
from model import Qwen

GATE_CONDS = ['none', 'neutral', 'neutral_paraphrase',
              'app_low_tr0', 'app_low_tr1', 'app_low_tr2', 'app_high_tr0', 'app_high_tr1', 'app_high_tr2',
              'beh_neg_tr0', 'beh_neg_tr1', 'beh_pos_tr0', 'beh_pos_tr1']


def row_for(cid, v, cond, res, split, method='base'):
    perm = permuted_options(v)
    logp = {str(i): res['letter_logp'][j] for j, (_, i, _) in enumerate(perm)}  # original option index -> logp
    pred = next(i for l, i, _ in perm if l == res['argmax'])
    return {'id': cid, 'video': video_of(cid), 'split': split, 'method': method, 'condition': cond,
            'logp': logp, 'pred': pred, 'gold': v['correct'], 'correct': pred == v['correct'],
            'order': [i for _, i, _ in perm], 'letter_mass': res['letter_mass'], 'finite': res['finite'],
            'n_tokens': res['n_tokens'], 'norms': norms_of_gold(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_gate', type=int, default=300)
    ap.add_argument('--n_check', type=int, default=20)
    args = ap.parse_args()
    cache = Path(os.environ.get('CACHE_DIR', f'{P777}/cache'))
    (cache / 'acts').mkdir(parents=True, exist_ok=True)
    outdir = HERE / 'results/gate'
    outdir.mkdir(parents=True, exist_ok=True)

    ann, table, splits, cues = load_annotations(), load_item_table(), load_splits(), load_cues()
    items = gate_items(splits, table, args.n_gate)
    assert all(splits['video_split'][video_of(k)] == 'train' for k in items)
    print(f'{len(items)} gate items (train split)', flush=True)

    t0 = time.time()
    q = Qwen()
    print(f'model loaded in {time.time() - t0:.0f}s, dtype {q.dtype}, layers {q.n_layers}, '
          f'GPU {torch.cuda.get_device_name(0)}', flush=True)

    # ---- 1) correctness check ----------------------------------------------------------------
    chk = []
    torch.cuda.synchronize(); t1 = time.time()
    for cid in items[:args.n_check]:
        v = ann[cid]
        img = Image.open(image_path(cid)).convert('RGB')
        txt = prompt_text(v, '')
        res, _ = q.score([img], txt, len(permuted_options(v)))
        gen = q.greedy([img], txt)
        chk.append({'id': cid, 'greedy_text': gen, 'argmax': res['argmax'], 'agree': gen.strip()[:1] == res['argmax'],
                    'correct': row_for(cid, v, 'none', res, 'train')['correct'], 'finite': res['finite'],
                    'letter_mass': res['letter_mass'], 'n_tokens': res['n_tokens'], 'grid': res['image_grid_thw'],
                    'image_size': list(img.size)})
    torch.cuda.synchronize()
    n = len(chk)
    summary = {'n': n, 'all_finite': all(c['finite'] for c in chk),
               'greedy_agree_rate': sum(c['agree'] for c in chk) / n,
               'accuracy': sum(c['correct'] for c in chk) / n, 'chance': 0.25,
               'min_letter_mass': min(c['letter_mass'] for c in chk),
               'mean_letter_mass': sum(c['letter_mass'] for c in chk) / n,
               'sec_per_item_score_plus_generate': (time.time() - t1) / n,
               'tokens_range': [min(c['n_tokens'] for c in chk), max(c['n_tokens'] for c in chk)],
               'peak_gpu_mem_GB': torch.cuda.max_memory_allocated() / 1e9,
               'dtype': str(q.dtype), 'gpu': torch.cuda.get_device_name(0),
               'example_prompt': prompt_text(ann[items[0]], cues['app_low_tr0'])}
    summary['passed'] = summary['all_finite'] and summary['greedy_agree_rate'] >= 0.95
    (outdir / 'check.json').write_text(json.dumps({'summary': summary, 'items': chk}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'example_prompt'}, indent=1), flush=True)
    if not summary['passed']:
        print('CORRECTNESS CHECK FAILED: stopping before the gate.', flush=True)
        sys.exit(2)

    # ---- 2) gate with residual capture (resumable) ---------------------------------------------
    rows_path = outdir / 'gate_rows.jsonl'
    acts_path = cache / 'acts/gate_resid.npy'
    shape = (len(items), len(GATE_CONDS), q.n_layers, q.model.config.text_config.hidden_size)
    acts = np.lib.format.open_memmap(acts_path, mode='r+' if acts_path.exists() else 'w+', dtype=np.float16, shape=shape)
    (cache / 'acts/gate_resid_index.json').write_text(json.dumps({'items': items, 'conditions': GATE_CONDS}))
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r['id'], r['condition']))
    t2, n_run = time.time(), 0
    with open(rows_path, 'a') as fh:
        for i, cid in enumerate(items):
            v = ann[cid]
            img = Image.open(image_path(cid)).convert('RGB')
            for j, cond in enumerate(GATE_CONDS):
                if (cid, cond) in done:
                    continue
                res, resid = q.score([img], prompt_text(v, cues[cond]), len(permuted_options(v)), capture=True)
                acts[i, j] = resid
                fh.write(json.dumps(row_for(cid, v, cond, res, 'train')) + '\n')
                n_run += 1
            fh.flush()
            if (i + 1) % 25 == 0:
                acts.flush()
                el = time.time() - t2
                print(f'{i + 1}/{len(items)} items, {n_run} passes, {el / max(n_run, 1):.3f} s/pass', flush=True)
    acts.flush()
    print(f'GATE DONE: {n_run} passes in {time.time() - t2:.0f}s', flush=True)


if __name__ == '__main__':
    main()
