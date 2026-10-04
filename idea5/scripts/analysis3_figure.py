"""Single-column figure for Analysis 3: joint accuracy of each frozen model under the three inputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-report'
INK, SECONDARY, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE = '#2a78d6', '#eb6834'
# (summary key, column header, colour, marker, filled, lane offset within a model's row, header x in axes fraction)
SERIES = [('text_nn', 'options', MUTED, 'o', False, 0.25, 1.15),
          ('rgb_nn', '+frames', BLUE, 'o', True, 0.0, 1.43),
          ('geo_nn', '+maps', ORANGE, 'D', True, -0.25, 1.70)]


def marker_style(color, marker, filled):
    return dict(marker=marker, ms=4.2 if marker == 'o' else 3.8, mfc=color if filled else 'white', mec=color, mew=1.0, ls='none')


def main():
    # Forced-choice run: every prompt forbids the "None of these" options.
    models = json.loads((ROOT / 'output/idea5-clean60/scored_nn.json').read_text())['all60']['summary']
    names = sorted([m for m in models if all(models[m].get(k, {}).get('n') == 60 for k, *_ in SERIES)], key=lambda m: (models[m]['rgb_nn']['both'], models[m]['geo_nn']['both']))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(3.25, 1.95))
    ax = fig.add_axes([0.26, 0.195, 0.405, 0.69])
    side = ax.get_yaxis_transform()  # x in axes fraction, y in data units
    for i, m in enumerate(names):
        for key, _, color, marker, filled, dy, x_col in SERIES:
            s = models[m][key]
            value, (lo, hi) = 100 * s['both'] / s['n'], (100 * v for v in s['both_ci95'])
            ax.plot([lo, hi], [i + dy, i + dy], color=color, lw=0.9, alpha=0.55, solid_capstyle='butt')
            ax.plot(value, i + dy, **marker_style(color, marker, filled))
            ax.text(x_col, i, f'{value:.0f}', transform=side, ha='center', va='center', fontsize=6.5, color=INK)
    # Column headers double as the legend: each series' marker sits above its column of values.
    for _, header, color, marker, filled, _, x_col in SERIES:
        ax.text(x_col, len(names) - 0.36, header, transform=side, ha='center', va='bottom', fontsize=5.8, color=SECONDARY)
        ax.plot(x_col, len(names) + 0.32, transform=side, clip_on=False, **marker_style(color, marker, filled))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.6, len(names) - 0.25)
    ax.set_xlabel('action and justification both correct (%)', fontsize=6.5, color=SECONDARY)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=SECONDARY, labelsize=6.5, width=0.6, length=2.5)
    ax.tick_params(axis='y', labelcolor=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'analysis3_figure.{ext}', dpi=500, facecolor='white')
    print('models plotted (bottom to top):', names)


if __name__ == '__main__':
    main()
