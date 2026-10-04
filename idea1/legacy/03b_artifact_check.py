"""03b - Stress tests for the text-only answer artifact found in 03(a).
Checks: behaviour-only vs justification-only; leakage via duplicate texts;
stricter grouping; the most predictive n-grams; per-category accuracy."""
import json, os, re, collections
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "egonormia"); T = os.path.join(ROOT, "outputs", "tables")
data = json.load(open(os.path.join(D, "final_data.json")))
rows = []
for k, v in data.items():
    for i, b in enumerate(v["behaviors"]):
        if b.strip():
            rows.append(dict(item=k, video=k.split("_")[0], idx=i, b=b, j=v["justifications"][i],
                             correct=int(i == v["correct"])))
O = pd.DataFrame(rows)

def run(col, groups, ngram=(1, 2), seed_shuffle=None, n_splits=5):
    y = O.correct.values.copy()
    if seed_shuffle is not None:  # permutation control: shuffle which option is correct within item
        rng = np.random.default_rng(seed_shuffle)
        y = np.zeros(len(O), int)
        for _, g in O.groupby("item"):
            y[rng.choice(g.index.values)] = 1
    s = np.zeros(len(O))
    for tr, te in GroupKFold(n_splits).split(O, groups=groups):
        v = TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True)
        Xtr = v.fit_transform(O[col].values[tr]); Xte = v.transform(O[col].values[te])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
        s[te] = clf.decision_function(Xte)
    O2 = O.assign(s=s, y=y)
    pick = O2.loc[O2.groupby("item").s.idxmax()]
    return pick.y.mean(), pick

O["bj"] = O.b + " " + O.j
res = {}
res["behavior+justification, group=video"] = run("bj", O.video)[0]
res["behavior only, group=video"] = run("b", O.video)[0]
res["justification only, group=video"] = run("j", O.video)[0]
res["behavior+justification, unigrams only"] = run("bj", O.video, ngram=(1, 1))[0]
res["permutation control (random gold), b+j"] = np.mean([run("bj", O.video, seed_shuffle=s)[0] for s in range(3)])

# Stricter grouping: cluster items whose texts share near-duplicate options (exact-normalized match)
norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).strip()
dup_b = O.groupby(O.b.map(norm)).item.nunique()
res["n_behavior_texts_shared_across_items"] = int((dup_b > 1).sum())
# union-find on items sharing any normalized behavior text
parent = {k: k for k in O.item.unique()}
def f(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for key, g in O.groupby(O.b.map(norm)):
    its = g.item.unique()
    for a in its[1:]:
        parent[f(a)] = f(its[0])
vid_root = {}
for v_, g in O.groupby("video"):
    its = g.item.unique()
    for a in its[1:]:
        parent[f(a)] = f(its[0])
O["cluster"] = O.item.map(f)
res["n_item_clusters_(video+shared_text)"] = int(O.cluster.nunique())
res["behavior+justification, group=video+shared-text cluster"] = run("bj", O.cluster)[0]

pd.Series(res).to_csv(os.path.join(T, "lang_artifact_checks.csv"))
for k, v in res.items(): print(f"{k:60s} {v:.4f}" if isinstance(v, float) else f"{k:60s} {v}")

# Most predictive n-grams (fit on all, for inspection only)
v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
X = v.fit_transform(O.bj); clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, O.correct)
coef = pd.Series(clf.coef_[0], index=v.get_feature_names_out()).sort_values()
top = pd.DataFrame({"toward_correct": coef.index[-25:][::-1], "w_pos": coef.values[-25:][::-1].round(2),
                    "toward_distractor": coef.index[:25], "w_neg": coef.values[:25].round(2)})
top.to_csv(os.path.join(T, "lang_artifact_top_ngrams.csv"), index=False)
print(top.to_string(index=False))

# How often do key distractor cues appear in correct vs distractor options?
for pat in [r"\bask\b", r"\bsuggest", r"\boffer\b", r"\bcontinue\b", r"\bignore\b", r"\bloudly\b", r"\bimmediately\b"]:
    m = O.bj.str.lower().str.contains(pat)
    print(f"{pat:16s} in correct: {O[O.correct==1].bj.str.lower().str.contains(pat).mean():.3f}  in distractors: {O[O.correct==0].bj.str.lower().str.contains(pat).mean():.3f}")

# Per-item cross-validated probe correctness (b+j, group=video+shared-text cluster) for downstream analysis
acc, pick = run("bj", O.cluster)
pick[["item", "y"]].rename(columns={"y": "text_probe_correct"}).to_csv(os.path.join(T, "lang_probe_item_correct.csv"), index=False)
print("saved per-item probe correctness; acc =", round(acc, 4))
