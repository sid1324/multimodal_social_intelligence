"""Gate stages.  python src/run_gate.py g1|perception|g2 --model nemotron|qwen
g1: 20 val items, no cue. (a) letter mass, 20 greedy generations, accuracy; (b) batched vs one-at-a-time argmax on
    mixed-condition prompts; (c) residual capture and ablation; (d) 10-step LoRA smoke test (--lora); (e) throughput.
perception: 30 val items per channel, repeat the remark verbatim, check the key trait word.
g2: the midterm's 300 gate items (train), base model. Text at L1/L2/L3, then speech and image at --level.
    Residuals are saved for Stage 2 (app/beh/valence cells)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, load_annotations, load_item_table, load_splits, gate_items, permuted_options, seeded_int
from channels import build, build_repeat, scene, CUES
from evaluate import run_grid, level_conds, CACHE, row_for, DEADLINE


def val20():
    return sorted(load_splits()['pool']['val'], key=lambda k: seeded_int('g1-v1', k))[:20]


def g1(a):
    from omni_model import load
    ann, items = load_annotations(), val20()
    t0 = time.time()
    m = load(a.model)
    info = m.load_info
    S = {'model': a.model, 'load_s': round(time.time() - t0), 'n_layers': m.n_layers, 'block_types': m.block_types,
         'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
         'missing_keys': sorted(info['missing_keys'])[:20], 'unexpected_keys': sorted(info['unexpected_keys'])[:20],
         'n_missing': len(info['missing_keys']), 'n_unexpected': len(info['unexpected_keys']),
         'mem_after_load_GB': [round(torch.cuda.memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())]}
    print(json.dumps(S), flush=True)
    out = HERE / f'results/gate/g1_{a.model}.json'; out.parent.mkdir(parents=True, exist_ok=True)
    save = lambda: out.write_text(json.dumps(S, indent=1))
    save()
    # (a)
    rows = []
    for cid in items:
        v, img = ann[cid], scene(cid)
        p = build(v, img, 'none', 'text')
        r, _ = m.score_many([p], len(permuted_options(v)), fast=False)
        g = m.generate(p, max_new_tokens=3)
        rw = row_for(cid, v, 'none', 'text', r[0], 'val', 'base', a.model)
        rows.append({'id': cid, 'gen': g, 'argmax': r[0]['argmax'], 'top_token': r[0]['top_token'], 'mass': r[0]['letter_mass'],
                     'correct': rw['correct'], 'finite': r[0]['finite'], 'n_tokens': r[0]['n_tokens']})
        print(rows[-1], flush=True)
    S['a'] = {'mean_letter_mass': float(np.mean([r['mass'] for r in rows])), 'min_letter_mass': min(r['mass'] for r in rows),
              'acc': float(np.mean([r['correct'] for r in rows])), 'greedy_agree': float(np.mean([r['gen'].strip()[:1] == r['argmax'] for r in rows])),
              'finite': all(r['finite'] for r in rows), 'rows': rows}
    save()
    # (b)+(e)
    cells = [('none', 'text')] + [(c, 'text') for c in level_conds('L2')] + [(c, ch) for ch in ('speech', 'image') for c in level_conds('L2')[:6]]
    agree = n = 0; dps = []; per = {}
    for cid in items[:6]:
        v, img = ann[cid], scene(cid); no = len(permuted_options(v))
        for ch in ('text', 'speech', 'image'):
            cs = [c for c in cells if c[1] == ch]
            ps = [build(v, img, c, x) for c, x in cs]
            torch.cuda.synchronize(); t = time.time(); one, _ = m.score_many(ps, no, fast=False); t1 = time.time() - t
            t = time.time(); bat, _ = m.score_many(ps, no, fast=True); t2 = time.time() - t
            per.setdefault(ch, []).append((t1 / len(cs), t2 / len(cs)))
            for x, y in zip(one, bat):
                n += 1; agree += x['argmax'] == y['argmax']
                dps.append(max(abs(np.exp(p) - np.exp(q)) for p, q in zip(x['letter_logp'], y['letter_logp'])))
    S['b'] = {'n': n, 'argmax_agree': agree / n, 'max_abs_dp': float(max(dps)), 'mean_abs_dp': float(np.mean(dps))}
    S['e'] = {ch: {'sec_per_fwd_one': float(np.mean([x for x, _ in z])), 'sec_per_fwd_batched': float(np.mean([y for _, y in z]))} for ch, z in per.items()}
    S['peak_mem_GB'] = [round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())]
    save(); print(S['b'], S['e'], S['peak_mem_GB'], flush=True)
    # (c)
    v, img = ann[items[0]], scene(items[0]); no = len(permuted_options(v)); p = build(v, img, 'L2_app_low_tr0', 'text')
    o0, r0 = m.score_many([p], no, capture=True)
    rnd = np.random.default_rng(0).standard_normal(r0[0].shape)
    m.set_ablation(r0[0], 1.0, [m.n_layers // 2])
    o1, r1 = m.score_many([p], no, capture=True)
    m.set_ablation(rnd, 1.0, range(m.n_layers)); o2, _ = m.score_many([p], no)
    m.set_ablation(None); o3, r3 = m.score_many([p], no, capture=True)
    L = m.n_layers // 2
    u = r0[0][L] / np.linalg.norm(r0[0][L])
    S['c'] = {'resid_shape': list(r0[0].shape), 'finite': bool(np.isfinite(r0[0]).all()), 'max_abs': float(np.abs(r0[0]).max()),
              'proj_before': float(r0[0][L] @ u), 'proj_after': float(r1[0][L] @ u),
              'lp_base': o0[0]['letter_logp'], 'lp_ablate_self': o1[0]['letter_logp'], 'lp_ablate_rand_all': o2[0]['letter_logp'],
              'restored_dlp': float(max(abs(x - y) for x, y in zip(o0[0]['letter_logp'], o3[0]['letter_logp'])))}
    save(); print(S['c'], flush=True)
    if a.lora:
        from dpo_train import smoke
        S['d'] = smoke(m, items, ann)
        save(); print(S['d'], flush=True)
    S['pass'] = {'a': S['a']['mean_letter_mass'] > 0.5 and S['a']['finite'] and S['a']['greedy_agree'] >= 0.95, 'b': S['b']['argmax_agree'] >= 0.98,
                 'a_gen_is_letter_only': all(r['gen'].replace('<|im_end|>', '').strip() in list('ABCDE') for r in rows),
                 'c': S['c']['finite'] and abs(S['c']['proj_after']) < 0.02 * abs(S['c']['proj_before']) and S['c']['restored_dlp'] < 0.05,
                 'd': S.get('d', {}).get('moved')}
    save(); print('G1', S['pass'], flush=True)


@torch.no_grad()
def moecheck(a):
    """(1) fast MoE dispatch vs original, one prompt at a time: adopt only if every letter log-prob is identical.
    (2) noise references for G1(b): the same prompt as a duplicated batch of two, and with extra right padding."""
    from omni_model import load
    ann, items = load_annotations(), val20()
    m = load('nemotron')
    P = []
    for cid in items[:6]:
        v, img = ann[cid], scene(cid)
        for ch in ('text', 'speech', 'image'):
            for c in ['L2_neutral', 'L2_neutral_paraphrase'] + [f'L2_app_{s}_tr{i}' for i in range(3) for s in ('low', 'high')]:
                P.append((build(v, img, c, ch), len(permuted_options(v))))
    def run():
        torch.cuda.synchronize(); t = time.time()
        o = [m.score_many([p], n)[0][0] for p, n in P]
        return o, (time.time() - t) / len(P)
    assert not m.fast_moe
    P0, P = P, P[::2]     # 72 prompts
    run(); A, ta = run()     # first pass only warms the embedding caches
    m.enable_fast(); B, tb = run()
    dp = lambda x, y: max(abs(np.exp(p) - np.exp(q)) for p, q in zip(x['letter_logp'], y['letter_logp']))
    S = {'n': len(P), 'fast_argmax_agree': float(np.mean([x['argmax'] == y['argmax'] for x, y in zip(A, B)])),
         'fast_max_abs_dp': float(max(dp(x, y) for x, y in zip(A, B))), 'sec_orig': ta, 'sec_fast': tb, 'fast_calls': m.fast_calls, 'expected_calls': len(P) * m.block_types.count('moe'), 'gpu': torch.cuda.get_device_name(0)}
    S['ok'] = S['fast_max_abs_dp'] == 0.0 and S['fast_calls'] == S['expected_calls']
    (HERE / 'results/gate/moecheck3.json').write_text(json.dumps(S, indent=1)); print('MOECHECK part 1', json.dumps(S), flush=True)


def key_word(cue):
    """Word that must be repeated: the trait word of the appearance cue."""
    for w in ('untrustworthy', 'trustworthy', 'dishonest', 'honest', 'cannot be trusted', 'can be trusted'):
        if w in cue:
            return w


def perception(a):
    import re
    from omni_model import load
    ann = load_annotations()
    items = sorted(load_splits()['pool']['val'], key=lambda k: seeded_int('perc-v1', k))[:30]
    conds = [f'L2_app_{s}_tr{i}' for i in range(3) for s in ('low', 'high')]
    out = HERE / f'results/gate/perception_{a.model}.jsonl'
    done = {(r['id'], r['channel']) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    m = load(a.model)
    norm = lambda s: re.sub(r'[^a-z ]', ' ', s.lower().replace('can not', 'cannot').replace("can't", 'cannot'))
    norm2 = lambda s: ' '.join(norm(s).split())
    with open(out, 'a') as fh:
        for j, cid in enumerate(items):
            v, img, cond = ann[cid], scene(cid), conds[j % 6]
            kw = key_word(CUES[cond])
            for ch in ('text', 'speech', 'image'):
                if (cid, ch) in done:
                    continue
                if time.time() > DEADLINE:
                    sys.exit(75)
                g = m.generate(build_repeat(v, img, cond, ch), max_new_tokens=60)
                hit = re.search(r'\b' + norm2(kw) + r'\b', norm2(g)) is not None
                fh.write(json.dumps({'id': cid, 'channel': ch, 'cond': cond, 'cue': CUES[cond], 'key': kw, 'gen': g, 'hit': hit}) + '\n'); fh.flush()
                print(ch, hit, kw, '|', g[:160].replace('\n', ' '), flush=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    S = {ch: float(np.mean([r['hit'] for r in rows if r['channel'] == ch])) for ch in ('text', 'speech', 'image')}
    (HERE / f'results/gate/perception_{a.model}.json').write_text(json.dumps(S, indent=1)); print('PERCEPTION', S, flush=True)


def g2_cells(level):
    text = [('none', 'text')] + [(c, 'text') for lv in ('L1', 'L2', 'L3') for c in level_conds(lv)] + [(f'val_{s}{i}', 'text') for s in ('neg', 'pos') for i in range(3)]
    other = [(c, ch) for ch in ('speech', 'image') for c in level_conds(level)]
    return text, other


def g2(a):
    from omni_model import load
    items = gate_items(load_splits(), load_item_table(), 300)[a.shard::2]   # two shards, one per interactive partition
    text, other = g2_cells(a.level)
    if a.model != 'nemotron':   # second model: main level only (START_PROMPT overnight step 7), plus none and valence
        text = [c for c in text if c[0].startswith(('none', a.level, 'val_'))]
    cells = text if a.part == 'text' else other
    ac = [c for c in cells if 'neutral' not in c[0] and c[0] != 'none']
    m = load(a.model)
    ok = run_grid(m, HERE / f'results/gate/g2_{a.model}_{a.part}_s{a.shard}.jsonl', 'base', items, cells, 'train',
                  acts={'path': str(CACHE / f'acts/g2_{a.model}_{a.part}_s{a.shard}.npy'), 'cells': ac})
    sys.exit(0 if ok else 75)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['g1', 'perception', 'g2', 'moecheck'])
    ap.add_argument('--model', default='nemotron'); ap.add_argument('--level', default='L2')
    ap.add_argument('--part', default='text', choices=['text', 'other']); ap.add_argument('--lora', action='store_true'); ap.add_argument('--shard', type=int, default=0)
    a = ap.parse_args()
    globals()[a.stage](a)
