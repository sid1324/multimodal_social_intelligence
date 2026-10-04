"""Language-modality exploration of EgoNormia answer options (no video, no VLM).

Compares a contextual representation (mean-pooled BERT) with interpretable ones
(Empath categories, a hand-written spatial/body lexicon) on three label-related
questions: norm category, Proxemics membership, and correct-vs-distractor.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from empath import Empath
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/idea5-language-v2'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
BERT = 'bert-base-uncased'
CATEGORIES = ['Proxemics', 'Safety', 'Politeness', 'Cooperation', 'Communication/Legibility', 'Privacy', 'Coordination/Proactivity']
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'

# Hand-written, fixed before looking at results. Word-boundary matches, lowercase.
SPATIAL = ['distance', 'space', 'near', 'nearby', 'close', 'closer', 'far', 'away', 'behind', 'front', 'beside', 'next to',
           'between', 'around', 'aside', 'path', 'way', 'room', 'step back', 'step aside', 'move', 'position', 'approach',
           'reach', 'side', 'across', 'ahead', 'crowd', 'crowded', 'clear', 'block', 'blocking', 'proximity', 'boundary', 'boundaries']
BODY = ['hand', 'hands', 'gesture', 'point', 'pointing', 'wave', 'nod', 'eye contact', 'look', 'face', 'facing', 'posture',
        'touch', 'hold', 'holding', 'grab', 'arm', 'arms', 'shoulder', 'body', 'lean', 'turn', 'stand', 'sit']


def has_term(text, terms):
    return any(re.search(r'\b' + re.escape(t) + r'\b', text) for t in terms)


def embed(texts, batch=64):
    tok = AutoTokenizer.from_pretrained(BERT)
    model = AutoModel.from_pretrained(BERT).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=96, return_tensors='pt')
            hidden = model(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
    return np.concatenate(out)


def cv_proba(X, y, groups, C):
    """Out-of-fold class probabilities with source-video-grouped folds."""
    proba = np.zeros((len(y), len(np.unique(y))))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[train], y[train])
        proba[test] = clf.predict_proba(X[test])
    return proba


def boot(values, n=2000):
    rng = np.random.default_rng(SEED)
    values = np.asarray(values, float)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return [float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    data = json.loads((ROOT / 'data/EgoNormia/annotations/final_data.json').read_text())
    lex = Empath()
    options = []  # one row per non-empty answer option
    for item_id, x in data.items():
        gold_is_real = bool(x['behaviors'][x['correct']])  # false when "none of these" is the labelled answer
        for i in range(5):
            if not x['behaviors'][i]:
                continue  # the empty slot is the "none of these" option; it is usually index 4 but not always
            options.append({'item': item_id, 'source': item_id.split('_')[0], 'idx': i, 'is_correct': x['correct'] == i,
                            'answerable': gold_is_real, 'labels': x['taxonomy'].get(str(i), []),
                            'text': f"{x['behaviors'][i]} {x['justifications'][i]}"})
    texts = [o['text'] for o in options]
    cache = OUT / 'bert_embeddings.npy'
    if cache.exists():
        X_bert = np.load(cache)
    else:
        X_bert = embed(texts)
        np.save(cache, X_bert)
    empath_names = sorted(lex.cats.keys())
    X_emp = np.array([[v / max(len(t.split()), 1) for _, v in sorted(lex.analyze(t.lower()).items())] for t in texts])
    spatial = np.array([has_term(t.lower(), SPATIAL) for t in texts])
    body = np.array([has_term(t.lower(), BODY) for t in texts])
    X_lex = np.c_[spatial, body].astype(float)
    groups = np.array([o['source'] for o in options])
    correct = np.array([o['is_correct'] for o in options])
    summary = {'bert': BERT, 'pooling': 'mean over tokens, last layer', 'items': len(data), 'options': len(options),
               'sources': len(set(groups)), 'cv': '5-fold grouped by source video', 'empath_categories': len(empath_names)}

    # Task 1: primary norm category of the correct option (first listed label).
    rows = [i for i, o in enumerate(options) if o['is_correct'] and o['labels'] and o['labels'][0] in CATEGORIES]
    y_cat = np.array([CATEGORIES.index(options[i]['labels'][0]) for i in rows])
    cat = {'n': len(rows), 'majority': float(np.bincount(y_cat).max() / len(rows))}
    for name, X, C in [('bert', X_bert, 0.05), ('empath', X_emp, 1.0)]:
        cat[name] = boot(cv_proba(X[rows], y_cat, groups[rows], C).argmax(1) == y_cat)
    summary['category_6way_primary'] = cat

    # Task 2: is Proxemics among the labels of the correct option?
    rows_all = [i for i, o in enumerate(options) if o['is_correct'] and o['labels']]
    y_prox = np.array(['Proxemics' in options[i]['labels'] for i in rows_all]).astype(int)
    prox = {'n': len(rows_all), 'positive_rate': float(y_prox.mean())}
    for name, X, C in [('bert', X_bert, 0.05), ('empath', X_emp, 1.0), ('spatial_body_lexicon', X_lex, 1.0)]:
        prox[name + '_auc'] = float(roc_auc_score(y_prox, cv_proba(X[rows_all], y_prox, groups[rows_all], C)[:, 1]))
    summary['proxemics_binary'] = prox

    # Task 3: text-only answer selection. Score every option, pick the top one per item.
    ans = np.array([o['answerable'] for o in options])
    items = np.array([o['item'] for o in options])[ans]
    text_only = {'n_items': int(ans.sum() // 4), 'chance': 0.25}
    lengths = np.array([len(t.split()) for t in texts])[ans]
    for name, X, C in [('bert', X_bert, 0.05), ('empath', X_emp, 1.0)]:
        score = cv_proba(X[ans], correct[ans].astype(int), groups[ans], C)[:, 1]
        hit = [correct[ans][items == it][score[items == it].argmax()] for it in dict.fromkeys(items)]
        text_only[name] = boot(hit)
        (OUT / f'probe_hits_{name}.json').write_text(json.dumps({it: bool(h) for it, h in zip(dict.fromkeys(items), hit)}))
    text_only['longest_option'] = boot([correct[ans][items == it][lengths[items == it].argmax()] for it in dict.fromkeys(items)])
    text_only['correct_mean_words'] = float(lengths[correct[ans]].mean())
    text_only['distractor_mean_words'] = float(lengths[~correct[ans]].mean())
    summary['text_only_answer_selection'] = text_only

    # Interpretable view: spatial/body vocabulary by norm category (correct options, multi-label membership).
    lexicon = {}
    for c in CATEGORIES:
        member = [i for i in rows_all if c in options[i]['labels']]
        lexicon[c] = {'n': len(member), 'spatial': boot(spatial[member]), 'body': boot(body[member])}
    lexicon['all_correct'] = {'n': len(rows_all), 'spatial': boot(spatial[rows_all]), 'body': boot(body[rows_all])}
    summary['spatial_body_lexicon'] = {'spatial_terms': SPATIAL, 'body_terms': BODY, 'by_category': lexicon}

    # Interpretable view: Empath categories most over-represented in Proxemics-labelled correct options.
    emp = X_emp[rows_all]
    present = emp > 0
    rate_p, rate_o = present[y_prox == 1].mean(0), present[y_prox == 0].mean(0)
    order = np.argsort(rate_p - rate_o)
    summary['empath_proxemics_contrast'] = {
        'more_in_proxemics': [[empath_names[j], float(rate_p[j]), float(rate_o[j])] for j in order[::-1][:8]],
        'less_in_proxemics': [[empath_names[j], float(rate_p[j]), float(rate_o[j])] for j in order[:8]]}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))

    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.edgecolor': '#c3c2b7', 'text.color': INK})

    # Figure 1: t-SNE of BERT embeddings of correct options, coloured by primary norm category.
    xy = TSNE(n_components=2, perplexity=35, init='pca', random_state=SEED).fit_transform(X_bert[rows])
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor=SURFACE)
    for k, c in enumerate(CATEGORIES):
        m = y_cat == k
        ax.scatter(xy[m, 0], xy[m, 1], s=9, color=COLORS[k], edgecolors=SURFACE, linewidths=0.3, label=f'{c} ({m.sum()})')
    style(ax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('t-SNE 1', color=MUTED, fontsize=8); ax.set_ylabel('t-SNE 2', color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2, markerscale=1.8)
    ax.set_title('BERT embeddings of correct answers, by primary norm category', fontsize=9, loc='left', color=INK)
    fig.savefig(OUT / 'tsne_category.png', dpi=220, bbox_inches='tight'); plt.close(fig)

    # Figure 2: what each representation recovers, one panel per question (separate scales).
    panels = [('Norm category (6-way)', 'accuracy', [('Empath', cat['empath']), ('BERT', cat['bert'])], cat['majority'], 'majority class'),
              ('Proxemics present', 'ROC AUC', [('Lexicon', [prox['spatial_body_lexicon_auc']] * 3), ('Empath', [prox['empath_auc']] * 3), ('BERT', [prox['bert_auc']] * 3)], 0.5, 'chance'),
              ('Pick the answer, text only', 'accuracy', [('Longest', text_only['longest_option']), ('Empath', text_only['empath']), ('BERT', text_only['bert'])], 0.25, 'chance')]
    tone = {'Lexicon': '#86b6ef', 'Longest': '#86b6ef', 'Empath': '#5598e7', 'BERT': '#184f95'}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), facecolor=SURFACE)
    for ax, (title, ylabel, bars, ref, ref_name) in zip(axes, panels):
        for i, (name, (v, lo, hi)) in enumerate(bars):
            ax.bar(i, v, width=0.55, color=tone[name])
            if hi > lo:
                ax.plot([i, i], [lo, hi], color=INK, lw=1)
            ax.text(i, max(v, hi) + 0.025, f'{v:.2f}', ha='center', fontsize=8, color=INK)
        ax.axhline(ref, color=MUTED, lw=1, ls='--')
        ax.set_xlabel(f'dashed line: {ref_name}', color=MUTED, fontsize=7)
        ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], color=INK)
        ax.set_ylim(0, 1); ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
        ax.set_title(title, fontsize=9, loc='left', color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / 'representation_comparison.png', dpi=220, bbox_inches='tight'); plt.close(fig)

    # Figure 3: share of correct answers using spatial vocabulary, by norm category.
    names = sorted(CATEGORIES, key=lambda c: lexicon[c]['spatial'][0])
    fig, ax = plt.subplots(figsize=(5.2, 2.7), facecolor=SURFACE)
    for i, c in enumerate(names):
        v, lo, hi = lexicon[c]['spatial']
        ax.barh(i, v, height=0.55, color='#2a78d6')
        ax.plot([lo, hi], [i, i], color=INK, lw=1)
        ax.text(hi + 0.015, i, f'{v:.0%}', va='center', fontsize=8, color=INK)
    ax.axvline(lexicon['all_correct']['spatial'][0], color=MUTED, lw=1, ls='--')
    ax.text(lexicon['all_correct']['spatial'][0] + 0.01, len(names) - 0.45, 'all correct answers', fontsize=7, color=MUTED)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([f"{c} (n={lexicon[c]['n']})" for c in names], color=INK, fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel('share of correct answers containing a spatial term', color=MUTED, fontsize=8)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    style(ax)
    ax.tick_params(axis='y', labelcolor=INK)
    fig.savefig(OUT / 'spatial_language.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'spatial_body_lexicon'}, indent=1))
    print(json.dumps({c: [round(v['spatial'][0], 3), round(v['body'][0], 3), v['n']] for c, v in lexicon.items()}, indent=1))


if __name__ == '__main__':
    main()
