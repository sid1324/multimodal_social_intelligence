"""02 - Perceptual-grounding ladder on EgoNormia.

Uses the per-item predictions released with EgoNormia (final_data_eval.json)
to ask, for the SAME model on the SAME items, how normative accuracy changes as
the input becomes more perceptually grounded:

  options only (blind)  ->  verbal scene description (desc)
  ->  stitched frame grid image (grid; EgoNormia default)
  [-> five discrete frames (frames) | native video (video), small subsets]

Condition definitions verified in src/eval/eval_api.py:
  blind : no image, no description ("You are blind ...")
  desc  : image replaced by the dataset's generated text description
  grid  : single stitched image of 1-fps frames (default img_url)
  frames: ablation='discrete_frames' (5 separate frames)
  video : ablation='video' (native mp4)

Scoring uses the CURRENT gold in final_data.json; malformed outputs (-1/None)
count as incorrect, as in the official script. A sensitivity version using
the gold stored alongside each prediction is also written.
Caveat (src/gen/04_filtering.py): items that blind Gemini-1.5-Flash answered
fully correctly were removed during construction, so the blind rung is
deflated by design.
"""
import json, os, collections
import numpy as np, pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "egonormia")
T = os.path.join(ROOT, "outputs", "tables")
RNG = np.random.default_rng(0)
B = 2000
CATS = ["Safety", "Privacy", "Proxemics", "Politeness", "Cooperation",
        "Coordination/Proactivity", "Communication/Legibility"]

data = json.load(open(os.path.join(D, "final_data.json")))
ev = json.load(open(os.path.join(D, "final_data_eval.json")))
verified = set(json.load(open(os.path.join(D, "verified_split.json")))["split"])
video_of = {k: k.split("_")[0] for k in data}

FAMILIES = {
    "Gemini-1.5-Flash": {"blind": "blind_gemini-15-flash-002", "desc": "desc_gemini-15-flash-002",
                         "grid": "gemini-15-flash-002", "frames": "frames_gemini-1.5-flash-002",
                         "video": "video_gemini-1.5-flash-002"},
    "Gemini-1.5-Pro":   {"blind": "blind_gemini-15-pro-002", "desc": "desc_gemini-15-pro-002",
                         "grid": "gemini-15-pro-002", "frames": "frames_gemini-1.5-pro-002",
                         "video": "video_gemini-1.5-pro-002"},
    "GPT-4o":           {"blind": "blind_gpt-4o-240513", "desc": "desc_gpt-4o-240513",
                         "grid": "gpt-4o-240513", "frames": "frames_gpt-4o"},
}
# Secondary: models with blind+desc only (text-only reasoners) and InternVL.
SECONDARY = {
    "o3-mini (text-only)": {"blind": "blind_gpt-o3-mini", "desc": "desc_gpt-o3-mini"},
    "DeepSeek-R1 (text-only)": {"blind": "blind_deepseek-r1-new", "desc": "desc_deepseek-r1-new"},
    "InternVL": {"blind": "blind_intern", "desc": "desc_intern", "grid": "intern"},
}


def score(key, gold="current"):
    """Return DataFrame indexed by item id with columns a, j, both (0/1)."""
    out = {}
    for k, v in ev.items():
        if key not in v:
            continue
        r = v[key]["best"]
        res = r.get("results")
        g = [data[k]["correct"]] * 2 if gold == "current" else r.get("correct")
        ok = isinstance(res, list) and len(res) == 2 and isinstance(g, list) and None not in g
        a = int(ok and res[0] == g[0]); j = int(ok and res[1] == g[1])
        out[k] = (a, j, int(a and j))
    return pd.DataFrame.from_dict(out, orient="index", columns=["a", "j", "both"])


def cluster_boot(ids, fn):
    """Bootstrap over source videos; fn(list_of_ids)->float."""
    vids = collections.defaultdict(list)
    for i in ids:
        vids[video_of[i]].append(i)
    keys = list(vids)
    stats = []
    for _ in range(B):
        samp = RNG.choice(len(keys), len(keys), replace=True)
        stats.append(fn([i for s in samp for i in vids[keys[s]]]))
    return np.percentile(stats, [2.5, 97.5])


def labels_correct(k):
    return set(data[k]["taxonomy"].get(str(data[k]["correct"]), []))


rows_overall, rows_pair, rows_norm, rows_trans = [], [], [], []
item_level = {}

for fam, conds in {**FAMILIES, **SECONDARY}.items():
    S = {c: score(m) for c, m in conds.items()}
    ladder = [c for c in ["blind", "desc", "grid"] if c in S]
    common = sorted(set.intersection(*[set(S[c].index) for c in ladder]))
    item_level[fam] = pd.DataFrame({c: S[c].loc[common, "both"] for c in ladder})
    for c in ladder:
        s = S[c].loc[common]
        lo, hi = cluster_boot(common, lambda ids, s=s: s.loc[ids, "both"].mean())
        rows_overall.append(dict(family=fam, condition=c, n_items=len(common),
                                 both=s.both.mean(), both_lo=lo, both_hi=hi,
                                 action=s.a.mean(), justification=s.j.mean(),
                                 tier="primary" if fam in FAMILIES else "secondary"))
    # pairwise McNemar on matched items (both-correct)
    pairs = [("blind", "desc"), ("desc", "grid"), ("blind", "grid")]
    for c1, c2 in pairs:
        if c1 in S and c2 in S:
            ids = sorted(set(S[c1].index) & set(S[c2].index))
            x, y = S[c1].loc[ids, "both"], S[c2].loc[ids, "both"]
            tab = [[int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())],
                   [int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum())]]
            p = mcnemar(tab, exact=True).pvalue
            d = (y - x)
            lo, hi = cluster_boot(ids, lambda i, d=d: d.loc[i].mean())
            rows_pair.append(dict(family=fam, from_=c1, to=c2, n_items=len(ids),
                                  acc_from=x.mean(), acc_to=y.mean(), delta=d.mean(),
                                  delta_lo=lo, delta_hi=hi, gained=tab[1][0], lost=tab[0][1],
                                  mcnemar_p=p))
    # static vs dynamic subsets: compare frames / video with grid on matched items
    for c in ["frames", "video"]:
        if c in S and "grid" in S:
            ids = sorted(set(S[c].index) & set(S["grid"].index))
            x, y = S["grid"].loc[ids, "both"], S[c].loc[ids, "both"]
            tab = [[int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())],
                   [int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum())]]
            d = y - x
            lo, hi = cluster_boot(ids, lambda i, d=d: d.loc[i].mean())
            rows_pair.append(dict(family=fam, from_="grid", to=c, n_items=len(ids),
                                  acc_from=x.mean(), acc_to=y.mean(), delta=d.mean(),
                                  delta_lo=lo, delta_hi=hi, gained=tab[1][0], lost=tab[0][1],
                                  mcnemar_p=mcnemar(tab, exact=True).pvalue))
    # per-norm accuracy on the matched ladder items (primary families only)
    if fam in FAMILIES:
        for cat in CATS:
            ids = [k for k in common if cat in labels_correct(k)]
            r = dict(family=fam, category=cat, n_items=len(ids))
            for c in ladder:
                r[c] = S[c].loc[ids, "both"].mean()
            rows_norm.append(r)
        # item transitions description -> grid, and blind -> grid
        for c1 in ["blind", "desc"]:
            x, y = S[c1].loc[common, "both"], S["grid"].loc[common, "both"]
            for name, mask in {"both correct": (x == 1) & (y == 1), "gained with image": (x == 0) & (y == 1),
                               "lost with image": (x == 1) & (y == 0), "both wrong": (x == 0) & (y == 0)}.items():
                rows_trans.append(dict(family=fam, from_=c1, outcome=name, share=mask.mean(), n=int(mask.sum())))

ov = pd.DataFrame(rows_overall); pr = pd.DataFrame(rows_pair)
nm = pd.DataFrame(rows_norm); tr = pd.DataFrame(rows_trans)
ov.to_csv(os.path.join(T, "ladder_overall.csv"), index=False)
pr.to_csv(os.path.join(T, "ladder_pairwise.csv"), index=False)
nm.to_csv(os.path.join(T, "ladder_by_norm.csv"), index=False)
tr.to_csv(os.path.join(T, "ladder_transitions.csv"), index=False)

# Pooled per-norm deltas across the three primary families with a joint
# cluster bootstrap (resample source videos, recompute family-averaged delta).
def pooled_delta(cat, c1, c2):
    per_fam = []
    for fam in FAMILIES:
        df = item_level[fam]
        ids = [k for k in df.index if cat in labels_correct(k)]
        per_fam.append(df.loc[ids])
    point = np.mean([(f[c2] - f[c1]).mean() for f in per_fam])
    vids = sorted({video_of[k] for f in per_fam for k in f.index})
    vidx = {v: i for i, v in enumerate(vids)}
    stats = []
    for _ in range(B):
        w = np.bincount(RNG.choice(len(vids), len(vids)), minlength=len(vids))
        vals = []
        for f in per_fam:
            ww = np.array([w[vidx[video_of[k]]] for k in f.index])
            if ww.sum() == 0:
                continue
            vals.append(np.average((f[c2] - f[c1]).values, weights=ww))
        stats.append(np.mean(vals))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    n = int(np.mean([len(f) for f in per_fam]))
    return point, lo, hi, n

rows = []
for cat in CATS + ["ALL"]:
    for c1, c2 in [("blind", "desc"), ("desc", "grid"), ("blind", "grid")]:
        if cat == "ALL":
            per = [(item_level[f][c2] - item_level[f][c1]).mean() for f in FAMILIES]
            rows.append(dict(category=cat, step=f"{c1}->{c2}", delta=np.mean(per), lo=np.nan, hi=np.nan, n_avg=np.nan))
        else:
            p, lo, hi, n = pooled_delta(cat, c1, c2)
            rows.append(dict(category=cat, step=f"{c1}->{c2}", delta=p, lo=lo, hi=hi, n_avg=n))
pdlt = pd.DataFrame(rows)
pdlt.to_csv(os.path.join(T, "ladder_by_norm_pooled_delta.csv"), index=False)

# Sensitivity: stored-gold scoring for primary ladder
sens = []
for fam, conds in FAMILIES.items():
    for c in ["blind", "desc", "grid"]:
        s_cur, s_st = score(conds[c]), score(conds[c], gold="stored")
        sens.append(dict(family=fam, condition=c, both_current_gold=s_cur.both.mean(),
                         both_stored_gold=s_st.both.mean(), n=len(s_cur)))
pd.DataFrame(sens).to_csv(os.path.join(T, "ladder_gold_sensitivity.csv"), index=False)

# Verified-subset robustness for the ladder
ver_rows = []
for fam in FAMILIES:
    df = item_level[fam]
    ids = [k for k in df.index if k in verified]
    ver_rows.append(dict(family=fam, n_items=len(ids), **{c: df.loc[ids, c].mean() for c in df.columns}))
pd.DataFrame(ver_rows).to_csv(os.path.join(T, "ladder_verified_subset.csv"), index=False)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("== Overall ladder (matched items) ==\n", ov.round(3).to_string(index=False))
print("\n== Pairwise ==\n", pr.round(4).to_string(index=False))
print("\n== Per-norm both-accuracy ==\n", nm.round(3).to_string(index=False))
print("\n== Pooled per-norm deltas ==\n", pdlt.round(3).to_string(index=False))
print("\n== Transitions ==\n", tr.round(3).to_string(index=False))
print("\n== Gold sensitivity ==\n", pd.DataFrame(sens).round(3).to_string(index=False))
print("\n== Verified subset ==\n", pd.DataFrame(ver_rows).round(3).to_string(index=False))
