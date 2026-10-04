"""Final full-width figure (figure*): A human labels, B person-directed gold answers by norm,
C OMI reference. Every axis is placed in inches so titles, baselines and edges align exactly.
Writes results/face_premise_figure_final.pdf/.png."""
import json, csv, collections as C
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / 'results'
REFS = HERE / 'data' / 'omi_refs'

USABLE, TURNED, BLURRED, ABSENT = '#2a78d6', '#a9a8a3', '#eda100', '#e4e3df'
PRIMARY_C, OTHER_C = '#52514e', '#c9c8c2'
INK, MUTED, RULE = '#1f1f1e', '#5d5c58', '#8a8984'
PRIMARY = {'Privacy', 'Coordination/Proactivity', 'Communication/Legibility', 'Cooperation'}
SHORT = {'Communication/Legibility': 'Communication', 'Coordination/Proactivity': 'Coordination'}

W, H = 6.75, 2.05            # inches; full ICML text width
TOP, BOT = 1.60, 0.66        # plot band shared by panels A and B
TITLE_Y = 1.86


def load():
    s1 = json.loads((OUT / 'label_sample.json').read_text())
    l1 = json.loads((OUT / 'human_labels.json').read_text())
    s2 = json.loads((OUT / 'label_sample_round2.json').read_text())
    l2 = json.loads((OUT / 'human_labels_round2.json').read_text())
    det = C.Counter()
    for f in s1['faces']:
        lab = l1[f['key']]
        if lab == 'clear':
            lab = 'turned' if l2.get('o_' + f['key']) == 'turned_away' else 'usable'
        det[lab] += 1
    items = C.Counter(l2[i['key']] for i in s2['items'])
    ann = json.loads((HERE / 'data/official_annotations.json').read_text())
    text = {r['id']: r for r in csv.DictReader(open(OUT / 'text_arm_items.csv'))}
    n, gp = C.Counter(), C.Counter()
    for k, v in ann.items():
        for nm in set(v['taxonomy'].get(str(v['correct']), [])):
            n[nm] += 1
            gp[nm] += text[k]['gold_person_directed'] == 'True'
    norms = sorted(((nm, 100 * gp[nm] / n[nm]) for nm in n if n[nm] >= 50), key=lambda x: x[1])
    return det, items, norms


def axes_in(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def title(fig, x, letter, text):
    fig.text(x / W, TITLE_Y / H, letter, fontsize=8, fontweight='bold', color=INK, va='baseline')
    fig.text((x + 0.16) / W, TITLE_Y / H, text, fontsize=7.5, color=INK, va='baseline')


def clean(ax):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(RULE)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(axis='x', colors=MUTED, labelsize=6.5, width=0.6, length=2.5, pad=2)
    ax.tick_params(axis='y', length=0)


def chip(fig, x, y, color, label):
    fig.patches.append(plt.Rectangle((x / W, (y - 0.035) / H), 0.075 / W, 0.075 / H, color=color,
                                     transform=fig.transFigure, figure=fig))
    fig.text((x + 0.11) / W, y / H, label, fontsize=6.5, color=INK, va='center')


def main():
    det, items, norms = load()
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(W, H))

    # ---- A: human labels (two samples, one colour meaning) ----
    title(fig, 0.0, 'A', 'Human labels of EgoNormia faces')
    a = axes_in(fig, 0.92, BOT, 1.78, TOP - BOT)
    rows = [('Face detections', 'n = 60', [det['usable'], det['turned'], det['blurred'], det['not_a_face']]),
            ('Random items', 'n = 40', [items['usable_face'], items['turned_away_or_too_small'],
                                        items['blurred_face_only'], items['no_other_person']])]
    cols = [USABLE, TURNED, BLURRED, ABSENT]
    for y, (name, nlab, counts) in zip((1, 0), rows):
        left, tot = 0.0, sum(counts)
        for v, c in zip(counts, cols):
            if not v:
                continue
            w = 100 * v / tot
            a.barh(y, w, left=left, height=0.5, color=c, edgecolor='white', linewidth=1.0)
            a.text(left + w / 2, y, str(v), ha='center', va='center', fontsize=6.8,
                   color='white' if c == USABLE else INK)
            left += w
        a.text(-4, y + 0.07, name, ha='right', va='bottom', fontsize=7, color=INK)
        a.text(-4, y - 0.05, nlab, ha='right', va='top', fontsize=6.3, color=MUTED)
    a.set_xlim(0, 100)
    a.set_ylim(-0.5, 1.5)
    a.set_yticks([])
    a.set_xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '100%'])
    clean(a)
    for i, (c, lab) in enumerate(zip(cols, ['Usable face', 'Turned away / too small', 'Blurred',
                                            'Not a face / no person'])):
        chip(fig, 0.06 + (i % 2) * 1.22, 0.27 - (i // 2) * 0.15, c, lab)

    # ---- B: text-cue pilot, answer disagreement between prompt pairs (60 items) ----
    title(fig, 3.62, 'B', 'Text-cue pilot: answers differ between prompts')
    dis = json.loads((OUT / 'pilot_disagreement.json').read_text())
    rows_b = [('Identical neutral prompts', 'Same prompt twice'), ('Neutral vs. paraphrase', 'Neutral vs. paraphrase'),
              ('Low vs. high trust', 'Low vs. high trust')]
    b = axes_in(fig, 4.86, BOT, 1.48, TOP - BOT)
    for y, (key, lab) in zip((2, 1, 0), rows_b):
        pct, lo, hi = dis[key]
        trait = key.startswith('Low')
        b.barh(y, pct, height=0.56, color=PRIMARY_C if trait else OTHER_C)
        b.plot([lo, hi], [y, y], color=INK, lw=0.8, solid_capstyle='butt', zorder=3)
        for x in (lo, hi):
            b.plot([x, x], [y - 0.1, y + 0.1], color=INK, lw=0.8, zorder=3)
        b.text(hi + 1.5, y, f'{pct:.0f}%', va='center', ha='left', fontsize=6.8,
               color=INK if trait else MUTED, fontweight='bold' if trait else 'normal')
        b.text(-2.5, y, lab, va='center', ha='right', fontsize=7, color=INK)
    b.set_xlim(0, 50)
    b.set_ylim(-0.5, 2.5)
    b.set_yticks([])
    b.set_xticks([0, 10, 20, 30, 40, 50], ['0', '10', '20', '30', '40', '50%'])
    clean(b)
    fig.text(3.68 / W, 0.27 / H, 'Gemini 3.1 Pro, 60 items; whiskers show 95% bootstrap CIs',
             ha='left', va='center', fontsize=6.5, color=MUTED)

    fig.savefig(OUT / 'face_premise_figure_final.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(OUT / 'face_premise_figure_final.png', dpi=600, bbox_inches='tight', pad_inches=0.04)


def faces():
    """Single-column figure: OMI donor references (top) and the edited close-ups Gemini rated (bottom)."""
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42})
    W2, H2 = 3.25, 2.80
    side, gap, x0 = 1.08, 0.07, 0.86
    leg = json.loads((OUT / 'legibility_gemini_gemini-3.1-pro-preview_summary.json').read_text())
    r = leg['closeup']['mean_by_item']['749']
    fig = plt.figure(figsize=(W2, H2))

    def square(im):
        s = min(im.size)
        return im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width - s) // 2 + s, (im.height - s) // 2 + s))

    rows = [('OMI reference', '(donor face)', [Image.open(REFS / f'omi_1284_trustworthy_{k}.jpg') for k in (0, 2)], None),
            ('Edited scene', '(close-up rated)', [Image.open(HERE / f'data/legibility_stimuli/749_{c}_closeup.png')
                                                 for c in ('low', 'high')], (r['low'], r['high']))]
    top = H2 - 0.22
    for j, lab in enumerate(('Low trust', 'High trust')):
        fig.text((x0 + j * (side + gap) + side / 2) / W2, (top + 0.06) / H2, lab, ha='center', va='bottom',
                 fontsize=7.5, color=INK, fontweight='bold')
    for i, (name, sub, ims, ratings) in enumerate(rows):
        y = top - side - i * (side + gap + 0.16)
        fig.text((x0 - 0.08) / W2, (y + side / 2 + 0.06) / H2, name, ha='right', va='bottom', fontsize=7, color=INK)
        fig.text((x0 - 0.08) / W2, (y + side / 2 - 0.02) / H2, sub, ha='right', va='top', fontsize=6.3, color=MUTED)
        for j, im in enumerate(ims):
            x = x0 + j * (side + gap)
            ax = fig.add_axes([x / W2, y / H2, side / W2, side / H2])
            ax.imshow(square(im.convert('RGB')), interpolation='lanczos')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('#d6d5cf')
                sp.set_linewidth(0.5)
            if ratings:
                fig.text((x + side / 2) / W2, (y - 0.04) / H2, f'Gemini rating {ratings[j]:.1f} / 7', ha='center',
                         va='top', fontsize=6.5, color=MUTED)
    fig.savefig(OUT / 'face_trust_examples.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(OUT / 'face_trust_examples.png', dpi=600, bbox_inches='tight', pad_inches=0.04)


if __name__ == '__main__':
    main()
    faces()
