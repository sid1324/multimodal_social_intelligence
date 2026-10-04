"""03 - Language-modality exploration on EgoNormia.

(a) Answer-artifact probe: can a linear model pick the correct option from the
    option text alone (no scene information)?  Option-level logistic
    regression over TF-IDF / Empath / length features; item-level top-1.
(b) Norm-category separability: how well do interpretable (Empath) and lexical
    (TF-IDF+LSA) representations of a candidate action+justification predict
    that option's norm labels?  And does the scene DESCRIPTION predict the
    labels of the correct option?
(c) Interpretable signal: Empath categories most associated with each norm.
(d) t-SNE of option representations coloured by (single-label) norm.

All splits are GroupKFold by Ego4D source video to avoid leakage between
items cut from the same recording.  Contextual-embedding versions (SBERT) are
in scripts/claude_code/ because model weights are not reachable offline here.
"""
import json, os, collections
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from empath import Empath
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "egonormia")
T = os.path.join(ROOT, "outputs", "tables"); F = os.path.join(ROOT, "outputs", "figures")
CATS = ["Safety", "Privacy", "Proxemics", "Politeness", "Cooperation",
        "Coordination/Proactivity", "Communication/Legibility"]
SHORT = {"Safety": "Safety", "Privacy": "Privacy", "Proxemics": "Proxemics", "Politeness": "Politeness",
         "Cooperation": "Cooperation", "Coordination/Proactivity": "Coordination",
         "Communication/Legibility": "Communication"}
SEED = 0
data = json.load(open(os.path.join(D, "final_data.json")))
lex = Empath()

opts = []
for k, v in data.items():
    for i, b in enumerate(v["behaviors"]):
        if not b.strip():
            continue
        labs = [l for l in v["taxonomy"].get(str(i), []) if l in CATS]
        opts.append(dict(item=k, video=k.split("_")[0], idx=i, behavior=b,
                         justification=v["justifications"][i],
                         text=b + " " + v["justifications"][i],
                         correct=int(i == v["correct"]), labels=labs))
O = pd.DataFrame(opts)
print("options:", len(O), "items:", O.item.nunique())

import sys
PART = sys.argv[1] if len(sys.argv) > 1 else "all"
emp_cats = sorted(lex.cats.keys())
CACHE = os.path.join(ROOT, "outputs", "logs", "empath_cache.npz")
if os.path.exists(CACHE):
    _c = np.load(CACHE); E, Ed_cached = _c["E"], _c["Ed"]
else:
    E = np.array([[lex.analyze(t, normalize=True).get(c, 0.0) or 0.0 for c in emp_cats] for t in O.text])
    _items = [k for k, v in data.items() if isinstance(v["desc"], str) and v["desc"].strip()]
    Ed_cached = np.array([[lex.analyze(data[k]["desc"], normalize=True).get(c, 0.0) or 0.0 for c in emp_cats] for k in _items])
    np.savez(CACHE, E=E, Ed=Ed_cached)
O["len_b"] = O.behavior.str.split().str.len(); O["len_j"] = O.justification.str.split().str.len()
gkf = GroupKFold(n_splits=5)
groups = O.video.values

# ---------------- (a) answer-artifact probe ----------------
def item_top1(scores):
    O2 = O.assign(s=scores)
    pick = O2.loc[O2.groupby("item").s.idxmax()]
    return pick.correct.mean()

def cv_scores(featfn):
    s = np.zeros(len(O))
    for tr, te in gkf.split(O, groups=groups):
        Xtr, Xte = featfn(tr, te)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, O.correct.values[tr]); s[te] = clf.decision_function(Xte)
    return s

def f_tfidf(tr, te):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return v.fit_transform(O.text.values[tr]), v.transform(O.text.values[te])

def f_empath(tr, te):
    sc = StandardScaler().fit(E[tr]); return sc.transform(E[tr]), sc.transform(E[te])

def f_len(tr, te):
    X = O[["len_b", "len_j"]].values.astype(float); sc = StandardScaler().fit(X[tr])
    return sc.transform(X[tr]), sc.transform(X[te])


def part_a():
    # Within-item relative features (deviation from item mean) for TF-IDF
    probe = {}
    for name, fn in [("TF-IDF (1-2gram)", f_tfidf), ("Empath (194 cats)", f_empath), ("Length", f_len)]:
        probe[name] = item_top1(cv_scores(fn))
    # null: permutation of option scores within item -> expected = mean(1/#options)
    probe["Chance (1/#non-empty options)"] = float(O.groupby("item").size().rdiv(1).mean())
    # bootstrap CI over items for the probe accuracies
    rng = np.random.default_rng(SEED)
    pa = pd.DataFrame({"probe": list(probe), "top1_action_acc": list(probe.values())})
    pa.to_csv(os.path.join(T, "lang_answer_probe.csv"), index=False)
    print(pa)


# ---------------- (b) norm-category separability ----------------
Y = np.array([[int(c in labs) for c in CATS] for labs in O.labels])
def auroc_probe(Xfn, Yv, grp, texts_idx=None):
    aucs = {c: [] for c in CATS}; pred = np.zeros_like(Yv, dtype=float)
    for tr, te in gkf.split(Yv, groups=grp):
        Xtr, Xte = Xfn(tr, te)
        for ci, c in enumerate(CATS):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Xtr, Yv[tr, ci]); pred[te, ci] = clf.decision_function(Xte)
    return {c: roc_auc_score(Yv[:, ci], pred[:, ci]) for ci, c in enumerate(CATS)}

def lsa_fn(texts):
    def fn(tr, te):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = v.fit_transform(texts[tr]); Xte = v.transform(texts[te])
        svd = TruncatedSVD(256, random_state=SEED).fit(Xtr)
        return svd.transform(Xtr), svd.transform(Xte)
    return fn


def part_b():
    res = {}
    res["Option text: TF-IDF+LSA"] = auroc_probe(lsa_fn(O.text.values), Y, groups)
    res["Option text: Empath"] = auroc_probe(f_empath, Y, groups)
    res["Behavior only: TF-IDF+LSA"] = auroc_probe(lsa_fn(O.behavior.values), Y, groups)

    # Scene description -> labels of the CORRECT option (item level)
    items = [k for k, v in data.items() if isinstance(v["desc"], str) and v["desc"].strip()]
    desc = np.array([data[k]["desc"] for k in items])
    Yd = np.array([[int(c in data[k]["taxonomy"].get(str(data[k]["correct"]), [])) for c in CATS] for k in items])
    gd = np.array([k.split("_")[0] for k in items])
    res["Scene description -> correct-option labels: TF-IDF+LSA"] = auroc_probe(lsa_fn(desc), Yd, gd)
    Ed = Ed_cached
    def f_empath_d(tr, te):
        sc = StandardScaler().fit(Ed[tr]); return sc.transform(Ed[tr]), sc.transform(Ed[te])
    res["Scene description -> correct-option labels: Empath"] = auroc_probe(f_empath_d, Yd, gd)

    auc = pd.DataFrame(res).T[CATS]; auc["macro"] = auc.mean(axis=1)
    auc.columns = [SHORT.get(c, c) for c in auc.columns]
    auc.to_csv(os.path.join(T, "lang_norm_auroc.csv"))
    print(auc.round(3).to_string())
    prev = pd.Series(Y.mean(0), index=[SHORT[c] for c in CATS]); prev.to_csv(os.path.join(T, "lang_option_label_prevalence.csv"))
    print("option-level label prevalence\n", prev.round(3))


def part_c():
    rows = []
    for ci, c in enumerate(CATS):
        pos, neg = E[Y[:, ci] == 1], E[Y[:, ci] == 0]
        diff = pos.mean(0) - neg.mean(0)
        for j in np.argsort(-diff)[:6]:
            rows.append(dict(category=SHORT[c], empath=emp_cats[j], mean_in=pos[:, j].mean(), mean_out=neg[:, j].mean(), diff=diff[j]))
    emp = pd.DataFrame(rows); emp.to_csv(os.path.join(T, "lang_empath_top_by_norm.csv"), index=False)
    print(emp.groupby("category").empath.apply(lambda s: ", ".join(s)).to_string())


def part_d():
    single = np.where(Y.sum(1) == 1)[0]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xs = TruncatedSVD(100, random_state=SEED).fit_transform(v.fit_transform(O.text.values))[single]
    Z = TSNE(2, random_state=SEED, perplexity=40, init="pca").fit_transform(Xs)
    lab = np.array([CATS[Y[i].argmax()] for i in single])
    pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "label": lab, "option_row": single}).to_csv(os.path.join(T, "lang_tsne_points.csv"), index=False)
    plt.figure(figsize=(4.2, 3.6))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(CATS):
        m = lab == c
        plt.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.6, color=cmap(i), label=f"{SHORT[c]} ({m.sum()})")
    plt.xticks([]); plt.yticks([]); plt.legend(fontsize=6, markerscale=3, loc="best", frameon=False)
    plt.title("t-SNE of option text (TF-IDF+LSA), single-label options", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(F, "lang_tsne.png"), dpi=300); plt.savefig(os.path.join(F, "lang_tsne.pdf"))
    print("single-label options in t-SNE:", len(single))

if __name__ == "__main__":
    for name, fn in [("a", part_a), ("b", part_b), ("c", part_c), ("d", part_d)]:
        if PART in (name, "all"):
            fn()
