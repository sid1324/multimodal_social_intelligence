"""Build DPO pairs (START_PROMPT Section 7) from the base model's TRAIN answers, text channel, L1 and L2 train templates.
Appearance: y(cue) != y(none) -> chosen = y(none), rejected = y(cue). Behavior: y(cue) != y(none) -> chosen = y(cue),
rejected = y(none). Behavior pairs are subsampled (seeded) if they outnumber appearance pairs by more than 1.5x ("to
match": down to the number of appearance pairs). DPO-multi uses the same pairs rendered in all three channels.
Usage: python src/dpo_pairs.py run --model nemotron   (remaining train items, resumable)
       python src/dpo_pairs.py build --model nemotron"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE, CONFIG, load_splits, load_item_table, gate_items
from evaluate import run_grid, read_rows, level_conds

CFG = yaml.safe_load((CONFIG / 'dpo.yaml').read_text())
CONDS = [c for lv in CFG['levels'] for c in level_conds(lv, ('app_tr', 'beh_tr'))]


def run(a):
    from omni_model import load
    sp = load_splits()
    gate = set(gate_items(sp, load_item_table(), 300))
    items = [k for k in sp['pool']['train'] if k not in gate]   # the 300 gate items come from the G2 rows
    ok = run_grid(load(a.model), HERE / f'results/dpo/{a.model}_train_rows.jsonl', 'base', items,
                  [('none', 'text')] + [(c, 'text') for c in CONDS], 'train')
    sys.exit(0 if ok else 75)


def build(a):
    sp = load_splits(); train = set(sp['pool']['train'])
    rows = [r for p in sorted((HERE / 'results/gate').glob(f'g2_{a.model}_text_s?.jsonl')) for r in read_rows(p)] + read_rows(HERE / f'results/dpo/{a.model}_train_rows.jsonl')
    pred = {(r['id'], r['condition']): r for r in rows if r['channel'] == 'text' and r['method'] == 'base'}
    items = sorted({i for i, _ in pred})
    assert set(items) <= train, 'non-train item in DPO rows'
    out = {'app': [], 'beh': []}
    for cid in items:
        y0 = pred[(cid, 'none')]['pred']
        for c in CONDS:
            y = pred[(cid, c)]['pred']
            if y == y0:
                continue
            fam = 'app' if '_app_' in c else 'beh'
            out[fam].append({'id': cid, 'cond': c, 'family': fam, 'side': c.split('_')[2], 'chosen': y0 if fam == 'app' else y,
                             'rejected': y if fam == 'app' else y0, 'gold': pred[(cid, c)]['gold']})
    raw = {f: len(v) for f, v in out.items()}
    sub = False
    if raw['beh'] > CFG['max_behavior_to_appearance'] * raw['app']:
        random.Random(CFG['seed']).shuffle(out['beh'])
        out['beh'] = out['beh'][:raw['app']]; sub = True
    pairs = sorted(out['app'] + out['beh'], key=lambda p: (p['id'], p['cond']))
    cnt = lambda f: dict(sorted(Counter(f(p) for p in pairs).items()))
    S = {'n_items': len(items), 'raw': raw, 'behavior_subsampled': sub, 'n_pairs': len(pairs), 'by_family_side': cnt(lambda p: p['family'] + '_' + p['side']),
         'by_level': cnt(lambda p: p['cond'][:2]), 'app_chosen_is_gold': sum(p['chosen'] == p['gold'] for p in pairs if p['family'] == 'app'),
         'app_rejected_is_gold': sum(p['rejected'] == p['gold'] for p in pairs if p['family'] == 'app')}
    (HERE / f'results/dpo/{a.model}_pairs.json').write_text(json.dumps({'summary': S, 'pairs': pairs}, indent=0))
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['run', 'build']); ap.add_argument('--model', default='nemotron')
    a = ap.parse_args(); {'run': run, 'build': build}[a.phase](a)
