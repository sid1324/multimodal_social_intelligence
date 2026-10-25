"""Report-size versions of the Idea 5 figures, drawn at their printed width so text stays legible.

  forest_compact: key contrasts only, one ICML column (3.25 in)
  lit_compact:    cross-model correlations with social scores, raw vs controlling for MMMU, one column
Reads figures/summary_forest.json, meta.json and lit_scaling/analysis.json (run midterm_summary_plot.py,
midterm_meta.py and lit_scaling_analysis.py first). Writes output/idea5-midterm/report/{forest_compact,lit_compact}.{pdf,png}.

usage: midterm_report_figures.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MID = ROOT / 'output/idea5-midterm'
OUT = MID / 'report'
INK, MUTED, GRID, BLUE, RED = '#1f2328', '#6e7781', '#d0d7de', '#2f6fdf', '#c4314b'
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6})

# (group, short label, source label in summary_forest.json)
FOREST = [
    ('Spatial input', 'Embedding: correct vs mismatched masks', ('Embedding, low-rank', 'Correct vs mismatched masks (C5-X1)')),
    ('Spatial input', 'VLMs: correct vs mismatched maps', ('Direct answering', 'Correct vs mismatched maps')),
    ('Spatial input', 'VLMs: maps vs RGB repeated', ('Direct answering', '+ maps vs RGB repeated')),
    ('Spatial input', 'Text only: true vs reversed facts', None),
    ('Spatial training', 'SAT LoRA vs non-spatial LoRA', ('Fine-tuning', 'Spatial LoRA vs non-spatial LoRA')),
    ('Spatial training', 'SAT LoRA + maps vs base (RGB)', ('Spatial LoRA x', 'Spatial LoRA + maps & descriptions vs base + RGB')),
    ('Spatial training', 'SpatialLadder-3B vs base', ('Published', 'SpatialLadder-3B vs Qwen2.5-VL-3B (paper: +23.5 spatial)')),
    ('Spatial training', 'MindCube-RL-3B vs base', ('Published', 'MindCube-RL-3B vs Qwen2.5-VL-3B (paper: +23.5 MindCube)')),
    ('Spatial training', 'SpaceR-7B vs base', ('Published', 'SpaceR-7B vs Qwen2.5-VL-7B (paper: +11.2 VSI-Bench)')),
    ('Spatial training', 'Spatial-SSRL-7B vs base', ('Published', 'Spatial-SSRL-7B vs Qwen2.5-VL-7B (paper: +3.9 avg)')),
    ('Pooled', 'All 10 spatial-specific contrasts', ('Pooled', 'All 10 spatial-specific contrasts')),
]


def style(ax):
    ax.grid(axis='x', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=6.5)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)


def forest():
    src = json.loads((MID / 'figures/summary_forest.json').read_text())
    rev = json.loads((MID / 'meta.json').read_text())['pooled']['reversal']
    rows = []
    for group, label, key in FOREST:
        if key is None:
            rows.append((group, label, rev['est'], rev['lo'], rev['hi']))
            continue
        hit = [r for r in src if r[0].startswith(key[0]) and r[1] == key[1]]
        assert len(hit) == 1, key
        rows.append((group, label, *hit[0][2:]))
    fig, ax = plt.subplots(figsize=(3.25, 2.8))
    y, ticks, labels, last = 0, [], [], None
    for group, label, est, lo, hi in rows:
        if group != last:
            ticks.append(y)
            labels.append(group)
            y += 1
            last = group
        color = BLUE if group == 'Reference' else INK
        ax.plot([lo, hi], [y, y], color=color, lw=1.4, solid_capstyle='round')
        ax.plot(est, y, 'D' if group == 'Pooled' else 'o', ms=4, color=color, mec='white', mew=0.8, zorder=3)
        ticks.append(y)
        labels.append('  ' + label)
        y += 1
    ax.axvline(0, color=MUTED, lw=0.7, zorder=0)
    ax.set_yticks(ticks, labels, fontsize=6.5)
    ax.tick_params(axis='y', labelcolor=INK)
    for t, l in zip(ax.get_yticklabels(), labels):
        if not l.startswith('  '):
            t.set_fontweight('bold')
    ax.set_ylim(y - 0.5, -0.7)
    ax.set_xlabel('Change in joint accuracy (pp, 95% CI)', fontsize=6.5, color=INK)
    style(ax)
    ax.tick_params(axis='y', labelcolor=INK)
    fig.tight_layout(pad=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'forest_compact.{ext}', dpi=300, bbox_inches='tight')


def lit():
    """Dumbbell: each skill's Spearman correlation with social scores, raw and controlling for MMMU."""
    a = json.loads((MID / 'lit_scaling/analysis.json').read_text())
    A, C = a['A_same_pipeline'], a['C_composite']
    rows = [('Spatial composite (30 models)', 'social-reasoning composite, 30 models', C['all']['spearman'], C['with_mmmu']['partial_spearman_mmmu'], True)]
    for key, name, spatial in (('Spatial Relationship (MMBench)', 'Spatial relationship', True), ('BLINK', 'BLINK (spatial)', True),
                               ('OCR (MMBench)', 'OCR', False), ('Celebrity recognition (MMBench)', 'Celebrity recognition', False),
                               ('Image topic (MMBench)', 'Image topic', False)):
        c = A[f'Social Relation (MMBench) ~ {key}']
        rows.append((name, f"MMBench Social Relation, {c['n']} models", c['spearman'], c['partial_spearman_mmmu'], spatial))
    fig, ax = plt.subplots(figsize=(3.25, 2.0))
    for y, (name, sub, raw, part, spatial) in enumerate(rows):
        color = BLUE if spatial else MUTED
        ax.plot([part, raw], [y, y], color=GRID, lw=2.2, zorder=1, solid_capstyle='round')
        ax.plot(raw, y, 'o', ms=5, color=color, zorder=3)
        ax.plot(part, y, 'o', ms=5, mfc='white', mec=color, mew=1.2, zorder=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.5)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(BLUE if r[4] else INK)
    ax.axhline(0.5, color=GRID, lw=0.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel('Spearman correlation with social score', fontsize=6.5, color=INK)
    ax.plot([], [], 'o', ms=5, color=INK, label='raw')
    ax.plot([], [], 'o', ms=5, mfc='white', mec=INK, mew=1.2, label='controlling for MMMU')
    ax.legend(fontsize=6, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3)
    style(ax)
    ax.tick_params(axis='y', labelcolor=INK)
    fig.tight_layout(pad=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'lit_compact.{ext}', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    forest()
    lit()
