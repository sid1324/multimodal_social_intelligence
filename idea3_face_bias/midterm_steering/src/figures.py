"""Figure 1 for the Idea 3 section (single ICML column, 3.25 in): (a) cosine between the appearance and
behavior directions by layer, with split-half stability; (b) validation trade-off between appearance flips and
behavior flips for every steering configuration, with base, constitution and the frozen choices marked.
Palette: validated categorical slots 1 to 3 (blue, orange, aqua); identity is also carried by line style or
marker shape and by direct labels.
Usage: python src/figures.py  -> report/figures/idea3_fig1.{pdf,png}
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#ffffff'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.3,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def main():
    st = json.loads((HERE / 'results/directions_stats.json').read_text())
    table = json.loads((HERE / 'results/sweep/sweep_table.json').read_text())
    frz = json.loads((HERE / 'results/FREEZE.json').read_text())
    L = list(range(len(st['cos_app_beh'])))

    fig, (a, b) = plt.subplots(2, 1, figsize=(3.25, 4.35), gridspec_kw={'height_ratios': [1, 1.25], 'hspace': 0.42})
    # (a) cosine by layer
    a.axvspan(14, 35, color=GRID, alpha=0.55, lw=0, zorder=0)
    a.axvline(25, color=INK2, lw=0.7, ls=(0, (2, 2)), zorder=1)
    a.text(14.4, -0.16, 'naive: 14 to 35', color=INK2, fontsize=5.8)
    a.text(25.5, 0.05, 'refined:\nlayer 25', color=INK2, fontsize=5.8)
    a.plot(L, st['split_half_cos_app'], color=ORANGE, lw=1.4, ls=(0, (4, 1.5)), label='stability, appearance')
    a.plot(L, st['split_half_cos_beh'], color=AQUA, lw=1.4, ls=(0, (1, 1.2)), label='stability, behavior')
    a.plot(L, st['cos_app_beh'], color=BLUE, lw=2, label='cos(appearance, behavior)')
    a.text(35.6, st['cos_app_beh'][-1], 'cos(app, beh)', color=INK, fontsize=5.8, va='center')
    a.set_xlim(0, 35); a.set_ylim(-0.2, 1.08)
    a.set_xlabel('Decoder layer'); a.set_ylabel('Cosine similarity')
    a.set_title('(a) Appearance and behavior trust directions', loc='left')
    a.grid(axis='y', color=GRID, lw=0.6); a.set_axisbelow(True)
    a.legend(loc='center left', bbox_to_anchor=(0.0, 0.6), frameon=False, handlelength=2.4, borderaxespad=0.2)

    # (b) validation trade-off
    base, cons = table['base'], table['constitution']
    floor = frz['r_beh_min'] * 100
    b.axhline(floor, color=INK2, lw=0.8, ls=(0, (3, 2)))
    b.text(16.2, floor - 0.95, 'behavior floor (0.8 x base)', color=INK2, fontsize=5.8, ha='left')
    for kind, color, marker in (('naive', BLUE, 'o'), ('refined', ORANGE, '^')):
        for m, t in table.items():
            if not m.startswith(kind + '_'):
                continue
            ok_acc = t['acc'][0] >= frz['acc_min']
            alpha = m.rsplit('_a', 1)[1]
            face = color if alpha == '1.0' else SURF
            b.scatter(t['s_app'][0] * 100, t['r_beh'][0] * 100, s=26 if ok_acc else 18, marker=marker if ok_acc else 'X',
                      facecolor=face if ok_acc else color, edgecolor=color, lw=1.1, zorder=3)
    for name, t, mk in (('base model', base, 's'), ('constitution', cons, 'D')):
        b.scatter(t['s_app'][0] * 100, t['r_beh'][0] * 100, s=26, marker=mk, color=INK, edgecolor=SURF, lw=0.8, zorder=4)
    b.annotate('base model', (base['s_app'][0] * 100, base['r_beh'][0] * 100), xytext=(4, 3), textcoords='offset points', fontsize=6, color=INK)
    b.annotate('constitution', (cons['s_app'][0] * 100, cons['r_beh'][0] * 100), xytext=(4, 2), textcoords='offset points', fontsize=6, color=INK)
    for kind, color in (('naive', BLUE), ('refined', ORANGE)):
        t = table[frz[kind]['name']]
        x, y = t['s_app'][0] * 100, t['r_beh'][0] * 100
        b.scatter(x, y, s=150, marker='*', facecolor='none', edgecolor=INK, lw=0.9, zorder=5)
        b.annotate(f'frozen {kind}', (x, y), xytext=(8, 4) if kind == 'naive' else (-8, 9), ha='left' if kind == 'naive' else 'right', textcoords='offset points', fontsize=6, color=INK)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', mfc=BLUE, mec=BLUE, ms=4.5, label='naive, alpha 1.0'),
         Line2D([], [], marker='o', ls='', mfc=SURF, mec=BLUE, ms=4.5, label='naive, alpha 0.5'),
         Line2D([], [], marker='^', ls='', mfc=ORANGE, mec=ORANGE, ms=4.5, label='refined, alpha 1.0'),
         Line2D([], [], marker='^', ls='', mfc=SURF, mec=ORANGE, ms=4.5, label='refined, alpha 0.5'),
         Line2D([], [], marker='X', ls='', mfc=INK2, mec=INK2, ms=4, label='fails accuracy limit')]
    b.legend(handles=h, loc='lower right', frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)
    b.set_xlabel('Appearance flips, $S_{app}$ (%)  (lower is better)')
    b.set_ylabel('Behavior flips, $R_{beh}$ (%)')
    b.set_title('(b) Validation trade-off of steering configurations', loc='left')
    b.set_xlim(13, 23.5); b.set_ylim(14.5, 31)
    b.grid(color=GRID, lw=0.6); b.set_axisbelow(True)

    out = HERE / 'report/figures'
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'idea3_fig1.{ext}', dpi=300, bbox_inches='tight', facecolor=SURF)
    print('wrote', out / 'idea3_fig1.pdf')


if __name__ == '__main__':
    main()
