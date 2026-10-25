"""Local: 2x2 of model (base vs SAT spatial LoRA) x input (RGB prompt vs combined RGB + maps + descriptions).

Replies come from eval_combined.py on a Runpod RTX 5090 (lora_pilot/results_combined/<run>.json); the Bridges L40S
RGB runs (results/{base,spatial,spatial_s2}.json) are scored alongside as the original cells. Scoring reuses
score_followup.score / pooled (exact McNemar over seed-paired items, item bootstrap). The interaction resamples items
over the per-item difference of differences, spatial arms averaged over both seeds.
Writes lora_pilot/scored_combined.json.
"""
import json

import numpy as np

from score_followup import B, OUT, pooled, score

FIELDS = ('joint', 'action', 'justification')


def none_picks(rows, tasks):
    """'None of these' choices for action and justification."""
    pick = lambda r, kind, pos: pos and tasks[r['id']][f'{kind}_options'][pos - 1].startswith('None of these')  # noqa: E731
    return {'action': sum(bool(pick(r, 'action', r['action_position'])) for r in rows),
            'justification': sum(bool(pick(r, 'justification', r['justification_position'])) for r in rows)}


def interaction(sp_c, sp_r, b_c, b_r):
    """(spatial combined - spatial rgb) - (base combined - base rgb), spatial lists hold one scored dict per seed."""
    ids = sorted(b_r)
    per = np.array([np.mean([s[i]['joint'] for s in sp_c]) - np.mean([s[i]['joint'] for s in sp_r])
                    - (b_c[i]['joint'] - b_r[i]['joint']) for i in ids])
    idx = np.random.default_rng(0).integers(0, len(ids), (B, len(ids)))
    draws = per[idx].mean(1)
    return {'diff': float(per.mean()), 'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            'p_boot_two_sided': float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))}


def main():
    key = json.loads((OUT / 'answer_key.json').read_text())
    tasks = {t['id']: t for t in json.loads((OUT / 'egonormia_tasks.json').read_text())}
    raw = {name: json.loads((OUT / 'results_combined' / f'{name}.json').read_text())
           for name in ('base_rgb', 'base_combined', 'spatial_s0_combined', 'spatial_s2_combined', 'spatial_s0_rgb', 'spatial_s2_rgb')
           if (OUT / 'results_combined' / f'{name}.json').exists()}
    for name, src in (('bridges_base_rgb', 'base'), ('bridges_spatial_s0_rgb', 'spatial'), ('bridges_spatial_s2_rgb', 'spatial_s2')):
        raw[name] = json.loads((OUT / 'results' / f'{src}.json').read_text())
    scored = {n: score(r, key) for n, r in raw.items()}
    out = {'cells': {}, 'contrasts': {}, 'timing': {}}
    for n, r in raw.items():
        s = scored[n]
        out['cells'][n] = {**{f: float(np.mean([v[f] for v in s.values()])) for f in FIELDS},
                           'parse_fail': sum(x['action_position'] is None for x in r['egonormia']),
                           'none_picks': none_picks(r['egonormia'], tasks), 'n': len(s)}
        if 'seconds' in r:
            out['timing'][n] = {'seconds': round(r['seconds']), 'mean_prompt_tokens': round(r['mean_prompt_tokens'])}
    seeds = ('s0', 's2')
    sp = lambda cond, pre='': [scored[f'{pre}spatial_{s}_{cond}'] for s in seeds]  # noqa: E731
    # 'pod': every cell from the 5090; 'bridges_rgb_cells': RGB cells from the original L40S runs.
    sets = {'pod': ('', scored['base_rgb'])} if all(f'spatial_{s}_rgb' in scored for s in seeds) else {}
    sets['bridges_rgb_cells'] = ('bridges_', scored['bridges_base_rgb'])
    for label, (pre, base_rgb) in sets.items():
        sp_rgb = sp('rgb', pre)
        sp_c = sp('combined')
        b_c = scored['base_combined']
        out['contrasts'][label] = {
            'a_spatial_combined-base_rgb': pooled([(x, base_rgb) for x in sp_c], 'joint'),
            'b_base_combined-base_rgb': pooled([(b_c, base_rgb)], 'joint'),
            'c_spatial_combined-base_combined': pooled([(x, b_c) for x in sp_c], 'joint'),
            'd_spatial_combined-spatial_rgb': pooled(list(zip(sp_c, sp_rgb)), 'joint'),
            'spatial_rgb-base_rgb': pooled([(x, base_rgb) for x in sp_rgb], 'joint'),
            'e_interaction': interaction(sp_c, sp_rgb, b_c, base_rgb)}
    out['two_seed_means'] = {cond: {f: float(np.mean([out['cells'][f'{pre}spatial_{s}_{cond}'][f] for s in seeds])) for f in FIELDS}
                             for cond, pre in (('combined', ''), ('rgb', 'bridges_'))}
    (OUT / 'scored_combined.json').write_text(json.dumps(out, indent=1))
    print(f"{'cell':24} {'joint':>6} {'action':>6} {'just':>6} {'parsefail':>9} {'None(a/j)':>9}")
    for n, v in out['cells'].items():
        print(f"{n:24} {v['joint']:6.3f} {v['action']:6.3f} {v['justification']:6.3f} {v['parse_fail']:9d} "
              f"{v['none_picks']['action']:4d}/{v['none_picks']['justification']}")
    print('two-seed means', json.dumps(out['two_seed_means']))
    for label, cs in out['contrasts'].items():
        for name, d in cs.items():
            extra = (f"fixed {d['fixed']} broken {d['broken']} McNemar p={d['p_mcnemar_exact']:.3g}" if 'fixed' in d
                     else f"boot p={d['p_boot_two_sided']:.3g}")
            print(f"{label:18} {name:34} {100 * d['diff']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] {extra}")
    print('timing', json.dumps(out['timing']))


if __name__ == '__main__':
    main()
