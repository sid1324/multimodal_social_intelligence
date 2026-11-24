"""F2 draft: cosine between trust directions by layer. Palette: the midterm's validated categorical slots.
Usage: python src/figures.py f2 [--models nemotron qwen]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8985'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
NAMES = {'nemotron': 'Nemotron 3 Nano Omni', 'qwen': 'Qwen3-Omni'}


def f2(models):
    models = [m for m in models if (HERE / f'results/directions_{m}.json').exists()]
    fig, ax = plt.subplots(len(models), 3, figsize=(15, 4.4 * len(models)), squeeze=False, sharey=True)
    for r, m in enumerate(models):
        J = json.loads((HERE / f'results/directions_{m}.json').read_text()); C = J['cos']
        panels = [('Appearance: same direction across channels?', [('app text-speech', 'text vs speech', BLUE, '-'), ('app text-image', 'text vs image', ORANGE, '-'), ('app speech-image', 'speech vs image', AQUA, '-')]),
                  ('Appearance vs behavior, per channel', [('app-beh text', 'text', BLUE, '-'), ('app-beh speech', 'speech', AQUA, '-'), ('app-beh image', 'image', ORANGE, '-'), ('app text - valence', 'text appearance vs weather (control)', GRAY, '--')]),
                  ('Appearance, text: across cue levels', [('app text L1-L2', 'L1 vs L2', BLUE, '-'), ('app text L1-L3', 'L1 vs L3', ORANGE, '-'), ('app text L2-L3', 'L2 vs L3', AQUA, '-')])]
        for c, (title, lines) in enumerate(panels):
            a = ax[r, c]
            for key, lab, col, ls in lines:
                if key in C:
                    a.plot(C[key], color=col, ls=ls, lw=2.4, label=lab)
            a.axhline(0, color=INK2, lw=0.8); a.grid(axis='y', color=GRID, lw=0.8); a.set_ylim(-0.4, 1.02)
            a.set_title((f'{NAMES[m]}\n' if c == 0 else '') + title, fontsize=13, loc='left', color=INK)
            a.set_xlabel('layer'); a.legend(frameon=False, fontsize=11, loc='best')
        ax[r, 0].set_ylabel('cosine similarity')
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(HERE / f'report/figures/F2_draft.{e}', dpi=300 if e == 'png' else None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('fig'); ap.add_argument('--models', nargs='*', default=['nemotron', 'qwen'])
    a = ap.parse_args(); f2(a.models)
