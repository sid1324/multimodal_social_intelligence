"""Random-effects pooling of every spatial-specific contrast in Idea 5.

A contrast is spatial-specific when both arms carry the same non-spatial content and differ only in whether the
spatial signal is correct (or present): correct vs mismatched maps, original vs spatially reversed descriptions,
spatial vs generic reasoning, spatial vs non-spatial fine-tuning or reward. Positive = spatial signal helps.
Each estimate's SE comes from its 95% CI; pooling is DerSimonian-Laird. For item-level direct contrasts it also
pools the high- vs low-spatial-need difference (GPT-5.6 rating >= 2 vs <= 1).
Writes output/idea5-midterm/meta.json.

usage: midterm_meta.py [--no-rl]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from midterm_summary_plot import MID, averaged, boot, direct, embedding

NEED = {i: v['score'] for i, v in json.loads((MID / 'spatial_need.json').read_text()).items()}
NAMES = {'gpt': 'GPT-5.6', 'gemini': 'Gemini 4', 'qwen32': 'Qwen3-VL-32B'}


def dl(est, se):
    est, w = np.array(est), 1 / np.array(se) ** 2
    fixed = (w * est).sum() / w.sum()
    q = (w * (est - fixed) ** 2).sum()
    tau2 = max(0.0, (q - (len(est) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (np.array(se) ** 2 + tau2)
    mu, s = (wr * est).sum() / wr.sum(), wr.sum() ** -0.5
    return {'est': mu, 'lo': mu - 1.96 * s, 'hi': mu + 1.96 * s, 'tau2': tau2, 'Q': q, 'k': len(est)}


def need_split(a, b, m):
    A, Bv = averaged(m, a), averaged(m, b)
    out = {}
    for name, keep in (('high', lambda s: s >= 2), ('low', lambda s: s <= 1)):
        groups = defaultdict(list)
        for i in set(A) & set(Bv):
            if i in NEED and keep(NEED[i]):
                groups[i].append(A[i] - Bv[i])
        out[name] = boot(groups)
    return out


def main():
    rows = []  # (family, label, est, lo, hi, (model, a, b) or None)
    rows.append(('maps', 'Embedding low-rank CV: correct vs mismatched masks', *embedding('C5', 'X1', 'lowrank', 'cv'), None))
    for m in ('gemini', 'gpt', 'qwen32'):
        rows.append(('maps', f'{NAMES[m]}: correct vs mismatched maps', *direct('maps', 'maps_wrong', [m]), (m, 'maps', 'maps_wrong')))
    for m in ('gemini', 'gpt', 'qwen32'):
        if len(averaged(m, 'desc_only_flip')) >= 150:
            # a reworded copy with every fact kept is the closer arm: both went through the same rewrite
            base = 'desc_only_para' if len(averaged(m, 'desc_only_para')) >= 150 else 'desc_only'
            rows.append(('reversal', f'{NAMES[m]}: {"reworded" if base != "desc_only" else "original"} vs spatially reversed descriptions',
                         *direct(base, 'desc_only_flip', [m]), (m, base, 'desc_only_flip')))
    for m in ('gpt', 'qwen32'):
        if len(averaged(m, 'rgb_spcot')) >= 75:
            rows.append(('reasoning', f'{NAMES[m]}: spatial vs generic step-by-step', *direct('rgb_spcot', 'rgb_cot', [m]),
                         (m, 'rgb_spcot', 'rgb_cot')))
    ft = json.loads((MID / 'lora_pilot/scored_followup.json').read_text())['contrasts']['spatial-control']['joint']
    rows.append(('training', 'SFT: spatial vs non-spatial LoRA', 100 * ft['diff'], 100 * ft['ci95'][0], 100 * ft['ci95'][1], None))
    rl = MID / 'rl_pilot/matched40/scored.json'  # R1 and R0 at the same number of steps
    if rl.exists() and '--no-rl' not in sys.argv:
        c = json.loads(rl.read_text())['contrasts'].get('R1-R0')
        if c:
            rows.append(('training', 'GRPO: spatial + norm reward vs norm reward', 100 * c['diff'], 100 * c['ci95'][0],
                         100 * c['ci95'][1], None))

    for r in rows:
        print(f'{r[0]:9s} {r[1]:58s} {r[2]:+5.1f} [{r[3]:+5.1f}, {r[4]:+5.1f}]')
    se = [(r[4] - r[3]) / 3.92 for r in rows]
    pooled = {'all': dl([r[2] for r in rows], se)}
    for fam in sorted({r[0] for r in rows}):
        idx = [k for k, r in enumerate(rows) if r[0] == fam]
        if len(idx) > 1:
            pooled[fam] = dl([rows[k][2] for k in idx], [se[k] for k in idx])
    inter = []
    for r in rows:
        if r[5]:
            s = need_split(r[5][1], r[5][2], r[5][0])
            inter.append({'label': r[1], 'high': s['high'], 'low': s['low']})
    for side in ('high', 'low'):
        pooled[f'need_{side}'] = dl([x[side][0] for x in inter], [(x[side][2] - x[side][1]) / 3.92 for x in inter])
    diff = [x['high'][0] - x['low'][0] for x in inter]
    dse = [np.hypot((x['high'][2] - x['high'][1]) / 3.92, (x['low'][2] - x['low'][1]) / 3.92) for x in inter]
    pooled['need_high_minus_low'] = dl(diff, dse)
    for k, v in pooled.items():
        print(f"pooled {k:20s} {v['est']:+5.1f} [{v['lo']:+5.1f}, {v['hi']:+5.1f}] k={v['k']} tau2={v['tau2']:.2f}")
    (MID / 'meta.json').write_text(json.dumps({'rows': [r[:5] for r in rows], 'pooled': pooled, 'need_split': inter},
                                              indent=1, default=float))


if __name__ == '__main__':
    main()
