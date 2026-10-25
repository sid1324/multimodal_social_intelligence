"""Published models: spatial score vs EgoNormia social-norm score (scatter only; the summary statistics are printed, not drawn).

Every value is read from an archived primary source (output/idea5-midterm/lit_scaling/sources/):
  spatial  = mean of the four circular-evaluation spatial benchmarks on the OpenCompass Spatial leaderboard
             (BLINK, 3DSRBench, MMSI-Bench, LEGO-Puzzles; results[model][<bench>_circular]["Overall"]),
             http://opencompass.openxlab.space/assets/SpatialLB.json, snapshot 20250815133921
  social   = EgoNormia "Both" accuracy (action and justification correct), Video Models rows, full split,
             Table 5 of arXiv 2502.20490v5 (https://arxiv.org/html/2502.20490), parsed from the archived HTML
  control  = MMMU (val) Overall from the OpenCompass OpenVLM leaderboard JSON, snapshot 20250917132916
Models are the ten that appear in both sources. Writes report/frontier_spatial_social.{pdf,png} and
lit_scaling/frontier_spatial_social.csv (one row per model with each value and its source string).

usage: frontier_spatial_social.py
"""
import csv
import html
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / 'output/idea5-midterm/lit_scaling'
SRC = LIT / 'sources'
OUT = ROOT / 'output/idea5-midterm/report'
INK, MUTED, GRID, BLUE = '#1f2328', '#6e7781', '#d0d7de', '#2f6fdf'
BENCHES = ('BLINK_circular', '3DSRBench_circular', 'MMSIBench_circular', 'LEGO_circular')
# display name -> (OpenCompass model key, name as printed in EgoNormia Table 5)
MODELS = {'Gemini 2.5 Pro': ('Gemini-2.5-Pro', 'Gemini 2.5 Pro'), 'Gemini 2.5 Flash': ('Gemini-2.5-Flash', 'Gemini 2.5 Flash'),
          'GPT-4.1': ('GPT-4.1-20250414', 'GPT-4.1'), 'Gemini 1.5 Pro': ('Gemini-1.5-Pro-002', 'Gemini 1.5 Pro'),
          'Gemini 1.5 Flash': ('Gemini-1.5-Flash-002', 'Gemini 1.5 Flash'), 'Qwen2.5-VL-72B': ('Qwen2.5-VL-72B', 'Qwen2.5 VL 72B'),
          'GPT-4o': ('GPT-4o (1120, detail-high)', 'GPT-4o'), 'Gemini 2.0 Flash': ('Gemini-2.0-Flash', 'Gemini 2.0 Flash'),
          'Claude 3.7 Sonnet': ('Claude3.7-Sonnet', 'Claude 3.7 Sonnet'), 'Claude 3.5 Sonnet': ('Claude3.5-Sonnet-20241022', 'Claude 3.5 Sonnet')}
# label offsets in points, chosen so the ten labels do not collide
OFFSET = {'Gemini 2.5 Pro': (-8, 5, 'right'), 'Gemini 2.5 Flash': (-6, 4, 'right'), 'GPT-4.1': (6, -1, 'left'),
          'Gemini 1.5 Pro': (-6, 3, 'right'), 'Gemini 1.5 Flash': (5, 3, 'left'), 'Qwen2.5-VL-72B': (6, 3, 'left'),
          'GPT-4o': (6, -3, 'left'), 'Gemini 2.0 Flash': (3, -9, 'left'), 'Claude 3.7 Sonnet': (-6, 1, 'right'),
          'Claude 3.5 Sonnet': (6, -1, 'left')}


def egonormia_video_rows():
    """'Both' (full split) for each Video Models row of Table 5, parsed from the archived arXiv HTML."""
    page = (SRC / 'egonormia_arxiv_2502.20490v5.html').read_text()
    text = lambda x: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', x))).strip()  # noqa: E731
    table = next(t for t in re.findall(r'<figure[^>]*class="ltx_table"[^>]*>(.*?)</figure>', page, re.S) if 'Table 5:' in text(t))
    rows, in_video = {}, False
    for tr in re.findall(r'<tr.*?</tr>', table, re.S):
        cells = [text(c) for c in re.findall(r'<t[dh].*?</t[dh]>', tr, re.S)]
        cells = [c for c in cells if c]
        if cells and cells[0] == 'Video Models':
            in_video = True
            continue
        if in_video and len(cells) >= 9 and re.fullmatch(r'[\d.]+', cells[1]):
            rows[cells[0]] = float(cells[1])
    return rows


def main():
    spatial = json.loads((SRC / 'opencompass_SpatialLB_20250815133921.json').read_text())['results']
    openvlm = json.loads((SRC / 'opencompass_OpenVLM_20250917132916.json').read_text())['results']
    ego = egonormia_video_rows()
    table = []
    for name, (oc, en) in MODELS.items():
        parts = [spatial[oc][b]['Overall'] for b in BENCHES]
        mmmu = (openvlm.get(oc, {}).get('MMMU_VAL') or {}).get('Overall')
        table.append({'model': name, 'spatial_mean': round(float(np.mean(parts)), 2), **dict(zip(BENCHES, parts)),
                      'egonormia_both': ego[en], 'mmmu_val': mmmu, 'opencompass_key': oc, 'egonormia_row': en,
                      'spatial_source': 'OpenCompass SpatialLB.json snapshot 20250815133921',
                      'egonormia_source': 'arXiv 2502.20490v5 Table 5, Video Models, Both (full split)',
                      'mmmu_source': 'OpenCompass OpenVLM.json snapshot 20250917132916' if mmmu else ''})
    with (LIT / 'frontier_spatial_social.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    x = np.array([r['spatial_mean'] for r in table])
    y = np.array([r['egonormia_both'] for r in table])
    fit = stats.linregress(x, y)
    with_mmmu = [r for r in table if r['mmmu_val'] is not None]
    xm, ym, zm = (np.array([r[k] for r in with_mmmu], float) for k in ('spatial_mean', 'egonormia_both', 'mmmu_val'))
    resid = lambda a: a - np.polyval(np.polyfit(zm, a, 1), zm)  # noqa: E731
    partial = float(np.corrcoef(resid(xm), resid(ym))[0, 1])
    print(f"n={len(x)} slope={fit.slope:.2f} r={fit.rvalue:.2f} p={fit.pvalue:.3f}; spatial~MMMU r={np.corrcoef(xm, zm)[0, 1]:.2f}; "
          f"partial r (MMMU, n={len(with_mmmu)})={partial:.2f}")

    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    ax.scatter(x, y, s=22, color=INK, edgecolor='white', linewidth=0.8, zorder=3)
    for r in table:
        dx, dy, ha = OFFSET[r['model']]
        ax.annotate(r['model'], (r['spatial_mean'], r['egonormia_both']), xytext=(dx, dy), textcoords='offset points',
                    fontsize=5.6, color=MUTED, ha=ha, va='center')
    ax.set_xlim(25, 47.5)
    ax.set_ylim(22, 57)
    ax.set_xticks([25, 30, 35, 40, 45])
    ax.set_xlabel('Spatial score, OpenCompass (%)', fontsize=6.5, color=INK)
    ax.set_ylabel('EgoNormia accuracy (%)', fontsize=6.5, color=INK)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=6, length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    fig.tight_layout(pad=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'frontier_spatial_social.{ext}', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
