"""Combine human labels, blur scores and text-arm results into summary numbers and the figure.

Threshold: the detail_ratio cutoff that best matches human clear/blurred labels; its accuracy is
reported with leave-one-out so the cutoff is never scored on the label used to choose it.
The label sample is self-weighting (equal draws from equal-count score sixths), so simple
proportions estimate the population of detected faces with min side >= 24 px.
"""
import json, csv, collections as C
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / 'results'
PRIMARY = {'Privacy', 'Coordination/Proactivity', 'Communication/Legibility', 'Cooperation'}


def best_cut(xs, ys):
    """Cutoff on x maximising agreement with y (1 = clear if x >= cut)."""
    cands = np.unique(xs)
    cands = np.concatenate([[cands[0] - 1], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1]])
    acc = [np.mean((xs >= c) == ys) for c in cands]
    i = int(np.argmax(acc))
    return float(cands[i]), float(acc[i])


def main():
    sample = json.loads((OUT / 'label_sample.json').read_text())
    labels = json.loads((OUT / 'human_labels.json').read_text())
    faces = sample['faces']
    lab = C.Counter(labels.get(f['key'], 'unlabeled') for f in faces)
    judged = [f for f in faces if labels.get(f['key']) in ('clear', 'blurred')]
    xs = np.array([float(f['detail_ratio']) for f in judged])
    ys = np.array([labels[f['key']] == 'clear' for f in judged])
    cut, train_acc = best_cut(xs, ys)
    loo = []
    for i in range(len(xs)):
        m = np.arange(len(xs)) != i
        c, _ = best_cut(xs[m], ys[m])
        loo.append((xs[i] >= c) == ys[i])
    # text lexicon check
    tl = [(t['lexicon_person_directed'], labels.get(t['key'])) for t in sample['texts']]
    tl = [(a, b == 'yes') for a, b in tl if b in ('yes', 'no')]

    rows = list(csv.DictReader(open(OUT / 'face_scores.csv')))
    for r in rows:
        r['ms'] = float(r['min_side'])
        r['clear'] = float(r['detail_ratio']) >= cut
    big = [r for r in rows if r['ms'] >= 24]
    ann = json.loads((HERE / 'data/official_annotations.json').read_text())
    items = C.defaultdict(list)
    for r in big:
        items[r['id']].append(r)
    clear48 = {i for i, rs in items.items() if any(r['clear'] and r['ms'] >= 48 for r in rs)}
    blur_only = {i for i, rs in items.items() if not any(r['clear'] for r in rs)}
    text = {t['id']: t for t in csv.DictReader(open(OUT / 'text_arm_items.csv'))}
    room = {i for i, t in text.items() if t['room_for_trait_cue'] == 'True'}
    prim = {k for k, v in ann.items() if set(v['taxonomy'].get(str(v['correct']), [])) & PRIMARY}

    s = {
        'human_face_labels': dict(lab),
        'faces_judged_clear_or_blurred': len(judged),
        'human_clear_share_in_sample': round(float(ys.mean()), 3) if len(ys) else None,
        'cutoff_detail_ratio': round(cut, 3),
        'agreement_at_cutoff_in_sample': round(train_acc, 3),
        'agreement_leave_one_out': round(float(np.mean(loo)), 3) if loo else None,
        'text_lexicon_checked': len(tl),
        'text_lexicon_agreement': round(float(np.mean([a == b for a, b in tl])), 3) if tl else None,
        'faces_min_side_ge24': len(big),
        'faces_ge24_scored_clear': sum(r['clear'] for r in big),
        'items_with_face_ge24': len(items),
        'items_with_clear_face_ge48': len(clear48),
        'items_with_face_ge24_none_clear': len(blur_only),
        'items_clear_ge48_and_room_for_text_cue': len(clear48 & room),
        'items_clear_ge48_room_and_primary_norm': len(clear48 & room & prim),
        'items_total': len(ann),
    }
    (OUT / 'premise_summary.json').write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    figure(rows, cut, judged, labels, s)


def figure(rows, cut, judged, labels, s):
    plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.25, 3.4), gridspec_kw={'height_ratios': [1.25, 1]})
    big = [r for r in rows if r['ms'] >= 24]
    bins = np.logspace(np.log10(24), np.log10(max(r['ms'] for r in big) + 1), 22)
    clear = [r['ms'] for r in big if r['clear']]
    blur = [r['ms'] for r in big if not r['clear']]
    a.hist([clear, blur], bins=bins, stacked=True, color=['#2a6f97', '#c9a227'],
           label=[f'Scored clear ({len(clear)})', f'Scored blurred ({len(blur)})'])
    a.set_xscale('log')
    a.set_xticks([24, 48, 100, 200, 400], ['24', '48', '100', '200', '400'])
    a.minorticks_off()
    a.axvline(48, color='#555', lw=0.8, ls='--')
    
    a.set_xlabel('Detected face size (shorter side, px)')
    a.set_ylabel('Detected faces')
    a.legend(frameon=False, fontsize=6)
    xs = [float(f['detail_ratio']) for f in judged]
    ys = [labels[f['key']] for f in judged]
    rng = np.random.default_rng(0)
    for name, y0, col in (('clear', 1, '#2a6f97'), ('blurred', 0, '#c9a227')):
        v = [x for x, y in zip(xs, ys) if y == name]
        b.scatter(v, y0 + rng.uniform(-0.15, 0.15, len(v)), s=10, color=col, alpha=0.8)
    b.axvline(cut, color='#555', lw=0.8, ls='--')
    b.set_yticks([0, 1], ['Human: blurred', 'Human: clear'])
    b.set_ylim(-0.5, 1.5)
    b.set_xlabel('Detail ratio (log detail inside / around face)')
    fig.tight_layout(h_pad=1.2)
    fig.savefig(OUT / 'face_premise_figure.pdf')
    fig.savefig(OUT / 'face_premise_figure.png', dpi=200)


if __name__ == '__main__':
    main()
