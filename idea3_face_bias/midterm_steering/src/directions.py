"""Step 3a: estimate per-layer trust directions from train residuals (last prompt token).

d_app[l] = mean over items and the 3 train templates of (h[app_low_i] - h[app_high_i])
d_beh[l] = mean over items and the 2 train templates of (h[beh_neg_j] - h[beh_pos_j])
u_perp[l] = normalize(d_app - (d_app . u_beh) u_beh)   (refined: appearance part not shared with behavior)
Also reports per layer: cos(d_app, d_beh), |d_app|, |d_beh| and split-half stability (cosine between
directions from two seeded random halves of the items).

Usage: python src/directions.py   (reads $CACHE_DIR/acts/gate_resid.npy + index; writes
$CACHE_DIR/directions.npz and results/directions_stats.json)
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, P777, seeded_int

CACHE = Path(os.environ.get('CACHE_DIR', f'{P777}/cache'))
APP_PAIRS = [(f'app_low_tr{i}', f'app_high_tr{i}') for i in range(3)]
BEH_PAIRS = [(f'beh_neg_tr{j}', f'beh_pos_tr{j}') for j in range(2)]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cos(a, b):
    return (unit(a) * unit(b)).sum(-1)


def diffs(acts, conds, pairs, rows):
    return np.concatenate([acts[rows, conds.index(a)].astype(np.float32) - acts[rows, conds.index(b)].astype(np.float32)
                           for a, b in pairs], 0)  # [len(rows)*len(pairs), L, H]


def main():
    acts = np.load(CACHE / 'acts/gate_resid.npy', mmap_mode='r')
    idx = json.loads((CACHE / 'acts/gate_resid_index.json').read_text())
    items, conds = idx['items'], idx['conditions']
    seen = {}
    for line in (HERE / 'results/gate/gate_rows.jsonl').read_text().splitlines():
        r = json.loads(line)
        seen.setdefault(r['id'], set()).add(r['condition'])
    rows_done = [k for k in items if seen.get(k, set()) >= set(conds)]  # items with every condition scored
    rows = np.array([items.index(k) for k in rows_done])
    assert len(rows) and np.isfinite(acts[rows]).all() and np.abs(acts[rows[-1]]).sum() > 0
    d_app = diffs(acts, conds, APP_PAIRS, rows).mean(0)
    d_beh = diffs(acts, conds, BEH_PAIRS, rows).mean(0)
    u_beh = unit(d_beh)
    d_perp = d_app - (d_app * u_beh).sum(-1, keepdims=True) * u_beh
    # split-half stability
    order = sorted(range(len(rows)), key=lambda i: seeded_int('split-half-v1', rows_done[i]))
    h1, h2 = rows[order[: len(order) // 2]], rows[order[len(order) // 2:]]
    stab_app = cos(diffs(acts, conds, APP_PAIRS, h1).mean(0), diffs(acts, conds, APP_PAIRS, h2).mean(0))
    stab_beh = cos(diffs(acts, conds, BEH_PAIRS, h1).mean(0), diffs(acts, conds, BEH_PAIRS, h2).mean(0))
    np.savez(CACHE / 'directions.npz', u_app=unit(d_app), u_beh=u_beh, u_perp=unit(d_perp), d_app=d_app, d_beh=d_beh)
    stats = {'n_items': int(len(rows)), 'layers': int(d_app.shape[0]),
             'cos_app_beh': cos(d_app, d_beh).round(4).tolist(),
             'norm_app': np.linalg.norm(d_app, axis=-1).round(3).tolist(),
             'norm_beh': np.linalg.norm(d_beh, axis=-1).round(3).tolist(),
             'split_half_cos_app': stab_app.round(4).tolist(), 'split_half_cos_beh': stab_beh.round(4).tolist()}
    (HERE / 'results/directions_stats.json').write_text(json.dumps(stats, indent=1))
    for l in range(0, d_app.shape[0], 4):
        print(f"layer {l:2d}: cos(app,beh)={stats['cos_app_beh'][l]:+.3f}  |d_app|={stats['norm_app'][l]:.2f}  "
              f"stab_app={stats['split_half_cos_app'][l]:.3f}  stab_beh={stats['split_half_cos_beh'][l]:.3f}")


if __name__ == '__main__':
    main()
