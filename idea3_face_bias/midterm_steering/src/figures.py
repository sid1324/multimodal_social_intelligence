"""Figure 1 for the Idea 3 section (single ICML column, 3.25 in): (a) cosine between the appearance and
behavior directions by layer, with split-half stability; (b) validation trade-off between appearance flips and
behavior flips for every steering configuration, with base, constitution and the frozen choices marked.
Palette: validated categorical slots 1 to 3 (blue, orange, aqua); identity is also carried by line style or
marker shape and by direct labels.
Usage: python src/figures.py  -> report/figures/idea3_fig1.{pdf,png}
"""
import json, sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#ffffff'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'STIXGeneral'], 'mathtext.fontset': 'stix',
                     'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
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
    a.text(14.4, -0.16, 'naive: 14 to 35', color=INK2, fontsize=6.6)
    a.text(25.5, 0.05, 'refined:\nlayer 25', color=INK2, fontsize=6.6)
    a.plot(L, st['split_half_cos_app'], color=ORANGE, lw=1.4, ls=(0, (4, 1.5)), label=r'split-half, $d_{app}$')
    a.plot(L, st['split_half_cos_beh'], color=AQUA, lw=1.4, ls=(0, (1, 1.2)), label=r'split-half, $d_{beh}$')
    a.plot(L, st['cos_app_beh'], color=BLUE, lw=2, label=r'$\cos(d_{app}, d_{beh})$')
    a.set_xlim(0, 35); a.set_ylim(-0.2, 1.08)
    a.set_xlabel('Decoder layer'); a.set_ylabel('Cosine similarity')
    a.set_title(r'$\bf{(a)}$ Appearance vs. behavior trust direction', loc='left')
    a.grid(axis='y', color=GRID, lw=0.6); a.set_axisbelow(True)
    a.legend(loc='center left', bbox_to_anchor=(0.0, 0.6), frameon=False, handlelength=2.4, borderaxespad=0.2)

    # (b) validation trade-off
    base, cons = table['base'], table['constitution']
    floor = frz['r_beh_min'] * 100
    b.axhline(floor, color=INK2, lw=0.8, ls=(0, (3, 2)))
    b.text(16.2, floor - 0.95, r'behavior floor (0.8 $\times$ base)', color=INK2, fontsize=6.6, ha='left')
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
    b.annotate('base model', (base['s_app'][0] * 100, base['r_beh'][0] * 100), xytext=(4, 3), textcoords='offset points', fontsize=6.8, color=INK)
    b.annotate('constitution', (cons['s_app'][0] * 100, cons['r_beh'][0] * 100), xytext=(4, 2), textcoords='offset points', fontsize=6.8, color=INK)
    for kind, color in (('naive', BLUE), ('refined', ORANGE)):
        t = table[frz[kind]['name']]
        x, y = t['s_app'][0] * 100, t['r_beh'][0] * 100
        b.scatter(x, y, s=150, marker='*', facecolor='none', edgecolor=INK, lw=0.9, zorder=5)
        b.annotate(f'frozen {kind}', (x, y), xytext=(8, 4) if kind == 'naive' else (0, 10), ha='left' if kind == 'naive' else 'center', textcoords='offset points', fontsize=6.8, color=INK)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', mfc=BLUE, mec=BLUE, ms=4.5, label=r'naive, $\alpha=1$'),
         Line2D([], [], marker='o', ls='', mfc=SURF, mec=BLUE, ms=4.5, label=r'naive, $\alpha=0.5$'),
         Line2D([], [], marker='^', ls='', mfc=ORANGE, mec=ORANGE, ms=4.5, label=r'refined, $\alpha=1$'),
         Line2D([], [], marker='^', ls='', mfc=SURF, mec=ORANGE, ms=4.5, label=r'refined, $\alpha=0.5$'),
         Line2D([], [], marker='X', ls='', mfc=INK2, mec=INK2, ms=4, label='fails accuracy limit')]
    b.legend(handles=h, loc='lower right', frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)
    b.set_xlabel('Appearance flips, $S_{app}$ (%)  (lower is better)')
    b.set_ylabel('Behavior flips, $R_{beh}$ (%)')
    b.set_title(r'$\bf{(b)}$ Validation trade-off (196 items)', loc='left')
    b.set_xlim(13, 23.5); b.set_ylim(14.5, 31)
    b.grid(color=GRID, lw=0.6); b.set_axisbelow(True)

    out = HERE / 'report/figures'
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'idea3_fig1.{ext}', dpi=300, bbox_inches='tight', facecolor=SURF)
    print('wrote', out / 'idea3_fig1.pdf')



def fig_example(item_id, methods=('base', 'naive', 'refined'), out_name='idea3_example'):
    """Appendix: one real test scene with its options and the option probabilities under the two
    appearance cues, for each method. Reads results/test rows."""
    import textwrap
    from PIL import Image
    from data import load_annotations, image_path
    from select_freeze import load_rows
    ann = load_annotations()
    v = ann[item_id]
    rows = [r for r in load_rows(HERE / 'results/test/rows_v2.jsonl.gz') if r['id'] == item_id]
    get = lambda m, c: next(r for r in rows if r['method'] == m and r['condition'] == c)
    opts = [(i, b) for i, b in enumerate(v['behaviors']) if b.strip()]
    letters = 'ABCD'
    fig = plt.figure(figsize=(6.75, 3.15))
    gs = fig.add_gridspec(2, len(methods) + 1, height_ratios=[1.0, 1.25], width_ratios=[1.55] + [1] * len(methods),
                          hspace=0.35, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(Image.open(image_path(item_id)).convert('RGB')); ax.axis('off')
    ax.set_title('Five pre-action frames (EgoNormia test item)', loc='left', fontsize=8)
    tx = fig.add_subplot(gs[1, 0]); tx.axis('off')
    y = 1.0
    for k, (i, b) in enumerate(opts):
        lab = '\n'.join(textwrap.wrap(b, 34))
        tx.text(0, y, f'{letters[k]}.', fontsize=7, weight='bold' if i == v['correct'] else 'normal', va='top')
        tx.text(0.08, y, lab + ('  (gold)' if i == v['correct'] else ''), fontsize=7, va='top', color=INK)
        y -= 0.1 + 0.11 * lab.count('\n') + 0.12
    for j, m in enumerate(methods):
        bx = fig.add_subplot(gs[1, j + 1])
        lo, hi = get(m, 'app_low_tr0'), get(m, 'app_high_tr0')
        ys = np.arange(len(opts))
        pl = [np.exp(lo['logp'][str(i)]) for i, _ in opts]
        ph = [np.exp(hi['logp'][str(i)]) for i, _ in opts]
        bx.barh(ys - 0.19, pl, height=0.36, color=ORANGE, label='"appears untrustworthy"')
        bx.barh(ys + 0.19, ph, height=0.36, color=BLUE, label='"appears trustworthy"')
        bx.set_yticks(ys, letters[:len(opts)] if j == 0 else [''] * len(opts)); bx.invert_yaxis()
        bx.set_xlim(0, 1); bx.set_xticks([0, 0.5, 1]); bx.grid(axis='x', color=GRID, lw=0.6); bx.set_axisbelow(True)
        bx.set_title({'base': 'Base model', 'naive': 'Naive steering', 'refined': 'Refined steering', 'constitution': 'Constitution'}[m], fontsize=7.5)
        bx.set_xlabel('p(option)', fontsize=7)
        if j == 0:
            bx.legend(loc='lower left', bbox_to_anchor=(-0.05, -0.62), ncol=2, frameon=False, fontsize=6.8)
    out = HERE / 'report/figures'
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{out_name}.{ext}', dpi=300, bbox_inches='tight', facecolor=SURF)
    print('wrote', out / f'{out_name}.pdf')


def fig_omi(n=4):
    """Appendix: sample matched OMI pairs (high vs low rated trustworthiness, matched happiness)."""
    from PIL import Image
    import os
    omi = Path(os.environ.get('OMI_DIR', '/ocean/projects/cis260338p/ylothe/data/omi'))
    pairs = json.loads((HERE / 'results/omi_pairs.json').read_text())['trust_pairs']
    import csv
    attrs = {r['stimulus']: r for r in csv.DictReader(open(omi / 'attribute_means.csv'))}
    fig, axes = plt.subplots(2, n, figsize=(3.25, 1.95), gridspec_kw={'hspace': 0.32, 'wspace': 0.06})
    for k, p in enumerate(pairs[:n]):
        for r, key in enumerate(('high', 'low')):
            s = p[key]; ax = axes[r, k]
            ax.imshow(Image.open(omi / 'images' / f'{s}.jpg').convert('RGB').resize((256, 256)))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.set_xlabel(f"T {float(attrs[s]['trustworthy']):.0f}  H {float(attrs[s]['happy']):.0f}", fontsize=6.3, labelpad=1.5)
        if k == 0:
            axes[0, 0].set_ylabel('higher trust', fontsize=6.8); axes[1, 0].set_ylabel('lower trust', fontsize=6.8)
    out = HERE / 'report/figures'
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'idea3_omi_pairs.{ext}', dpi=300, bbox_inches='tight', facecolor=SURF)
    print('wrote', out / 'idea3_omi_pairs.pdf')


if __name__ == '__main__':
    import numpy as np
    globals()['np'] = np
    which = sys.argv[1] if len(sys.argv) > 1 else 'fig1'
    if which == 'fig1':
        main()
    elif which == 'omi':
        fig_omi()
    elif which == 'example':
        fig_example(sys.argv[2])
