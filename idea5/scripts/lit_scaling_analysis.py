"""Across published models, do spatial and social benchmark scores move together, and is that more than general ability?

Inputs (compiled from papers and leaderboards, no model run here):
  output/idea5-midterm/lit_scaling/{spatial_scores,social_scores}.json
  the OpenCompass OpenVLM leaderboard JSON (same evaluator for every model): --openvlm path
Analyses
  A  same-pipeline, OpenVLM: social sub-scores vs spatial sub-scores and vs non-spatial comparators, raw and partial on MMMU
  B  paper benchmarks: each social-reasoning benchmark vs each spatial benchmark (n >= 7 shared models), raw and partial on MMMU
  C  composites: per model, mean z over spatial benchmarks vs mean z over social-reasoning benchmarks
  D  within-family size scaling (>= 3 sizes): Spearman of parameter count with spatial and with social scores
Writes output/idea5-midterm/lit_scaling/analysis.json and figures/lit_scaling.{png,pdf}.

usage: lit_scaling_analysis.py --openvlm <OpenVLM.json>
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / 'output/idea5-midterm/lit_scaling'
FIG = ROOT / 'output/idea5-midterm/figures'
INK, MUTED, GRID, BLUE = '#1f2328', '#6e7781', '#d0d7de', '#2f6fdf'
SPATIAL = ['OC-composite', 'BLINK|OpenVLM', 'MMSI-Bench|paper', 'SpatialScore|paper', 'VSI-Bench|EASI-LB',
           'OmniSpatial|paper', 'VSI-Bench|InternVL3.5-report', 'MindCube-Tiny|EASI-LB', 'ViewSpatial|EASI-LB', 'SITE|EASI-LB']
SOCIAL = ['EgoNormia', 'EgoNormia_verified', 'SIV-Bench', 'EmoBench-M', 'HumanVBench_Behavior', 'MMToM-QA_AutoToM',
          'MuMA-ToM', 'SoMi-ToM_third', 'ToMBench', 'SimpleToM']
COMPOSITE_SPATIAL = ['OC-composite', 'BLINK|OpenVLM', 'MMSI-Bench|paper', 'SpatialScore|paper', 'VSI-Bench|EASI-LB', 'OmniSpatial|paper']
COMPOSITE_SOCIAL = ['EgoNormia', 'SIV-Bench', 'EmoBench-M', 'HumanVBench_Behavior', 'MMToM-QA_AutoToM', 'MuMA-ToM']
MMMU = 'control:MMMU-val|OpenVLM'


def corr(x, y, z=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {'n': len(x), 'spearman': float(stats.spearmanr(x, y)[0]), 'spearman_p': float(stats.spearmanr(x, y)[1]),
           'pearson': float(stats.pearsonr(x, y)[0])}
    if z is not None and len(x) >= 6:
        rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
        res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
        r = float(np.corrcoef(res(rx), res(ry))[0, 1])
        t = r * np.sqrt((len(x) - 3) / max(1e-12, 1 - r ** 2))
        out.update({'partial_spearman_mmmu': r, 'partial_p': float(2 * stats.t.sf(abs(t), len(x) - 3))})
    return out


def table():
    sp = json.loads((LIT / 'spatial_scores.json').read_text())['rows']
    so = json.loads((LIT / 'social_scores.json').read_text())['rows']
    rows = defaultdict(dict)
    params = {}
    for r in sp + so:
        for k, v in r['scores'].items():
            rows[r['model']][k] = v['value']
        if r.get('params_b'):
            params[r['model']] = r['params_b']
    for m, s in rows.items():
        circ = [s.get(f'{b}|OC-circular') for b in ('BLINK', '3DSRBench', 'MMSI-Bench', 'LEGO-Puzzles')]
        if all(v is not None for v in circ):
            s['OC-composite'] = float(np.mean(circ))
    return rows, params


def pairs(rows, a, b, control=False):
    ms = [m for m, s in rows.items() if a in s and b in s and (not control or MMMU in s)]
    return ms, [rows[m][a] for m in ms], [rows[m][b] for m in ms], [rows[m].get(MMMU) for m in ms]


def openvlm(path):
    """Same evaluator for every model: MMBench v1.1 sub-scores plus BLINK and MMMU."""
    res = json.loads(Path(path).read_text())['results']
    out = {}
    for name, v in res.items():
        mb = v.get('MMBench_TEST_EN_V11') or {}
        mt = v.get('MMT-Bench_VAL') or {}
        row = {'Social Relation (MMBench)': mb.get('Social Relation'), 'Spatial Relationship (MMBench)': mb.get('Spatial Relationship'),
               'Human interaction (MMT)': mt.get('human_interaction_understanding'), 'Social relation (MMT)': mt.get('social_relation_recognition'),
               'BLINK': (v.get('BLINK') or {}).get('Overall'), 'MMMU': (v.get('MMMU_VAL') or {}).get('Overall'),
               'OCR (MMBench)': mb.get('Ocr'), 'Celebrity recognition (MMBench)': mb.get('Celebrity Recognition'),
               'Image topic (MMBench)': mb.get('Image Topic')}
        row = {k: float(x) for k, x in row.items() if isinstance(x, (int, float)) or (isinstance(x, str) and re.fullmatch(r'[\d.]+', x))}
        if row:
            out[name] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--openvlm', required=True)
    args = ap.parse_args()
    rows, params = table()
    result = {'A_same_pipeline': {}, 'B_paper_pairs': {}, 'C_composite': {}, 'D_size_scaling': {}}

    # A: one evaluator, hundreds of models. Is social more tied to spatial than to unrelated skills?
    ov = openvlm(args.openvlm)
    for soc in ('Social Relation (MMBench)', 'Human interaction (MMT)', 'Social relation (MMT)'):
        for other in ('Spatial Relationship (MMBench)', 'BLINK', 'OCR (MMBench)', 'Celebrity recognition (MMBench)', 'Image topic (MMBench)'):
            ms = [m for m, r in ov.items() if soc in r and other in r and 'MMMU' in r]
            if len(ms) >= 10:
                result['A_same_pipeline'][f'{soc} ~ {other}'] = corr([ov[m][soc] for m in ms], [ov[m][other] for m in ms],
                                                                     [ov[m]['MMMU'] for m in ms])

    # B: reasoning benchmarks from papers
    for soc in SOCIAL:
        for spa in SPATIAL:
            ms, x, y, _ = pairs(rows, spa, soc)
            if len(ms) < 7:
                continue
            c = corr(x, y)
            msc, xc, yc, zc = pairs(rows, spa, soc, control=True)
            if len(msc) >= 6:
                c.update({k: v for k, v in corr(xc, yc, zc).items() if 'partial' in k})
                c['n_partial'] = len(msc)
            c['models'] = ms
            result['B_paper_pairs'][f'{soc} ~ {spa}'] = c

    # C: composites (z within each benchmark over the models that have it)
    z = defaultdict(dict)
    for key in set(COMPOSITE_SPATIAL + COMPOSITE_SOCIAL):
        ms = [m for m in rows if key in rows[m]]
        if len(ms) < 5:
            continue
        v = np.array([rows[m][key] for m in ms])
        for m, s in zip(ms, (v - v.mean()) / v.std()):
            z[m][key] = s
    comp = {}
    for m, s in z.items():
        a = [s[k] for k in COMPOSITE_SPATIAL if k in s]
        b = [s[k] for k in COMPOSITE_SOCIAL if k in s]
        if a and b:
            comp[m] = {'spatial_z': float(np.mean(a)), 'social_z': float(np.mean(b)), 'k_spatial': len(a), 'k_social': len(b),
                       'mmmu': rows[m].get(MMMU)}
    cms = sorted(comp)
    result['C_composite'] = {'models': comp, 'all': corr([comp[m]['spatial_z'] for m in cms], [comp[m]['social_z'] for m in cms])}
    msc = [m for m in cms if comp[m]['mmmu'] is not None]
    if len(msc) >= 6:
        result['C_composite']['with_mmmu'] = corr([comp[m]['spatial_z'] for m in msc], [comp[m]['social_z'] for m in msc],
                                                  [comp[m]['mmmu'] for m in msc])

    # D: within-family size scaling
    fam = defaultdict(list)
    for m in rows:
        g = re.match(r'(Qwen2-VL|Qwen2\.5-VL|InternVL2|InternVL2\.5|InternVL3|Ovis2|LLaVA-OneVision|Gemma-3|DeepSeek-VL2)-?', m)
        if g and m in params:
            fam[g.group(1)].append(m)
    for f, ms in fam.items():
        out = {}
        for key in ('BLINK|OpenVLM', 'OC-composite', 'MMSI-Bench|paper', 'SpatialScore|paper', 'MMBench_SocialRelation',
                    'EmoBench-M', 'EgoNormia', 'SIV-Bench', MMMU):
            have = [m for m in ms if key in rows[m]]
            if len(have) >= 3:
                rho = stats.spearmanr([params[m] for m in have], [rows[m][key] for m in have])[0]
                out[key] = {'n': len(have), 'rho_size': float(rho), 'scores': {m: rows[m][key] for m in sorted(have, key=params.get)}}
        if out:
            result['D_size_scaling'][f] = out
    (LIT / 'analysis.json').write_text(json.dumps(result, indent=1, default=float))

    print('A same pipeline (OpenVLM): spearman raw | partial on MMMU')
    for k, c in result['A_same_pipeline'].items():
        print(f"  {k:62s} n={c['n']:3d} rho={c['spearman']:+.2f} partial={c.get('partial_spearman_mmmu', float('nan')):+.2f}")
    print('B paper benchmarks')
    for k, c in sorted(result['B_paper_pairs'].items(), key=lambda kv: -kv[1]['n']):
        print(f"  {k:55s} n={c['n']:2d} rho={c['spearman']:+.2f} (p={c['spearman_p']:.3f})"
              + (f" partial(n={c['n_partial']})={c['partial_spearman_mmmu']:+.2f} p={c['partial_p']:.2f}" if 'n_partial' in c else ''))
    c = result['C_composite']
    print('C composite', {k: round(v, 3) if isinstance(v, float) else v for k, v in c['all'].items()},
          {k: round(v, 3) if isinstance(v, float) else v for k, v in c.get('with_mmmu', {}).items()})
    print('D size scaling (Spearman with parameter count)')
    for f, d in result['D_size_scaling'].items():
        print(f'  {f:16s} ' + '  '.join(f"{k.split('|')[0][:14]}(n={v['n']}) {v['rho_size']:+.2f}" for k, v in d.items()))

    # Figure: composite scatter (C) and same-pipeline specificity bars (A)
    FIG.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios': [1.15, 1]})
    xs = [comp[m]['spatial_z'] for m in cms]
    ys = [comp[m]['social_z'] for m in cms]
    ax1.scatter(xs, ys, s=36, color=BLUE, edgecolor='white', linewidth=1.2, zorder=3)
    for m, x, y in zip(cms, xs, ys):
        ax1.annotate(m, (x, y), xytext=(4, 3), textcoords='offset points', fontsize=6.5, color=MUTED)
    a = result['C_composite']['all']
    ax1.set_title(f"Spatial vs social composite across models (n={a['n']}, Spearman {a['spearman']:+.2f})", fontsize=9, color=INK, loc='left')
    ax1.set_xlabel('Spatial composite (mean z over spatial benchmarks)', fontsize=8.5, color=INK)
    ax1.set_ylabel('Social composite (mean z over social-reasoning benchmarks)', fontsize=8.5, color=INK)
    labels, raw, part = [], [], []
    for k, c in result['A_same_pipeline'].items():
        if k.startswith('Social Relation (MMBench)'):
            labels.append(k.split(' ~ ')[1])
            raw.append(c['spearman'])
            part.append(c['partial_spearman_mmmu'])
    yy = np.arange(len(labels))
    ax2.barh(yy - 0.18, raw, height=0.34, color=BLUE, label='raw')
    ax2.barh(yy + 0.18, part, height=0.34, color=GRID, edgecolor=MUTED, label='controlling for MMMU')
    ax2.set_yticks(yy, labels, fontsize=8, color=INK)
    ax2.invert_yaxis()
    n = result['A_same_pipeline']['Social Relation (MMBench) ~ Spatial Relationship (MMBench)']['n']
    ax2.set_title(f'Same evaluator (OpenVLM, n={n}): Social Relation correlates with...', fontsize=9, color=INK, loc='left')
    ax2.set_xlabel('Spearman correlation', fontsize=8.5, color=INK)
    ax2.legend(fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5, colors=MUTED)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'lit_scaling.{ext}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
