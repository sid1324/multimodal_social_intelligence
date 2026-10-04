"""01 - EgoNormia structure audit.

Verifies counts, schema, label representation, answer-position balance,
sensible-set structure, and description properties directly from the
released JSON (GitHub: Open-Social-World/EgoNormia, src/final_dataset).
Every number printed here is also written to outputs/tables/.
"""
import json, collections, re, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "egonormia")
OUT = os.path.join(ROOT, "outputs", "tables")
CATS = ["Safety", "Privacy", "Proxemics", "Politeness", "Cooperation",
        "Coordination/Proactivity", "Communication/Legibility"]

data = json.load(open(os.path.join(D, "final_data.json")))
verified = json.load(open(os.path.join(D, "verified_split.json")))["split"]

summary = {}
summary["n_items"] = len(data)
summary["n_unique_ids"] = len(set(data))
assert all(k == v["id"] for k, v in data.items()), "key/id mismatch"

# Source video = Ego4D UUID before the first '_' (format <uuid>_<timestamp>)
ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_\d+-\d+$")
summary["ids_matching_uuid_timestamp_format"] = sum(bool(ID_RE.match(k)) for k in data)
src = {k: k.split("_")[0] for k in data}
summary["n_source_videos"] = len(set(src.values()))
per_vid = collections.Counter(src.values())
summary["items_per_video_mean"] = round(float(np.mean(list(per_vid.values()))), 3)
summary["items_per_video_max"] = int(max(per_vid.values()))

summary["n_verified"] = len(verified)
summary["n_verified_unique"] = len(set(verified))
summary["verified_all_in_data"] = all(v in data for v in verified)

# Field presence / schema
fields = collections.Counter(tuple(sorted(v.keys())) for v in data.values())
summary["schemas"] = {"|".join(k): n for k, n in fields.items()}

def desc_text(v):
    return v["desc"] if isinstance(v["desc"], str) else ""
summary["items_with_missing_desc"] = [k for k, v in data.items() if not desc_text(v).strip()]

rows = []
for k, v in data.items():
    b, j = v["behaviors"], v["justifications"]
    nonempty = [i for i, x in enumerate(b) if x.strip()]
    tax = {int(i): labs for i, labs in v["taxonomy"].items()}
    rows.append(dict(
        id=k, video=src[k], verified=k in set(verified),
        n_options=len(b), n_justifications=len(j),
        n_nonempty_options=len(nonempty),
        empty_option_idx=[i for i, x in enumerate(b) if not x.strip()],
        correct=v["correct"], correct_is_empty=not b[v["correct"]].strip(),
        n_sensible=len(v["sensibles"]),
        correct_in_sensible=v["correct"] in v["sensibles"],
        n_labels_correct=len(tax.get(v["correct"], [])),
        labels_correct="|".join(tax.get(v["correct"], [])),
        n_distinct_labels_options=len({l for labs in tax.values() for l in labs}),
        desc_words=len(desc_text(v).split()),
        desc_has_frame_lines=bool(re.search(r"Frame \d+:", desc_text(v))),
        correct_len_words=len(b[v["correct"]].split()),
        mean_distractor_len=np.mean([len(b[i].split()) for i in nonempty if i != v["correct"]]) if len(nonempty) > 1 else np.nan,
    ))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "egonormia_items.csv"), index=False)

summary["options_per_item"] = df.n_options.value_counts().to_dict()
summary["nonempty_options_per_item"] = df.n_nonempty_options.value_counts().to_dict()
summary["empty_option_position"] = collections.Counter(tuple(x) for x in df.empty_option_idx).most_common(5)
summary["correct_is_empty_None_option"] = int(df.correct_is_empty.sum())
summary["correct_in_sensible_rate"] = round(float(df.correct_in_sensible.mean()), 4)
summary["n_sensible_dist"] = df.n_sensible.value_counts().sort_index().to_dict()
summary["labels_on_correct_option_dist"] = df.n_labels_correct.value_counts().sort_index().to_dict()
summary["desc_words_median"] = float(df.desc_words.median())
summary["desc_has_frame_lines_rate"] = round(float(df.desc_has_frame_lines.mean()), 4)

# Answer-position balance (options are NOT shuffled by the official eval:
# eval_api.py uses identity permutations)
pos = df.correct.value_counts().sort_index()
summary["correct_position_counts"] = pos.to_dict()
from scipy.stats import chisquare
nonempty_pos = pos[[i for i in pos.index if i < 4]]
chi = chisquare(nonempty_pos.values)
summary["correct_position_chi2_over_0to3"] = dict(chi2=round(float(chi.statistic), 2), p=float(chi.pvalue))

# Length artifact: is the correct option systematically longer?
from scipy.stats import wilcoxon
d = (df.correct_len_words - df.mean_distractor_len).dropna()
w = wilcoxon(d)
summary["correct_minus_distractor_len_words"] = dict(mean=round(float(d.mean()), 3),
    median=float(d.median()), wilcoxon_p=float(w.pvalue))
# share of items where correct option is the single longest
longest = []
for k, v in data.items():
    L = [len(x.split()) if x.strip() else -1 for x in v["behaviors"]]
    longest.append(int(np.argmax(L) == v["correct"] and L.count(max(L)) == 1))
summary["correct_is_unique_longest_rate"] = round(float(np.mean(longest)), 4)
shortest = []
for k, v in data.items():
    L = [len(x.split()) if x.strip() else 10**6 for x in v["behaviors"]]
    shortest.append(int(np.argmin(L) == v["correct"] and L.count(min(L)) == 1))
summary["correct_is_unique_shortest_rate"] = round(float(np.mean(shortest)), 4)
summary["note_chance_rate_4_nonempty_options"] = 0.25

# Label distribution (correct option and any option), full + verified
def label_counts(sub):
    c_corr, c_any = collections.Counter(), collections.Counter()
    for k in sub:
        v = data[k]
        tax = {int(i): l for i, l in v["taxonomy"].items()}
        for l in set(tax.get(v["correct"], [])):
            c_corr[l] += 1
        for l in {l for labs in tax.values() for l in labs}:
            c_any[l] += 1
    n = len(sub)
    return pd.DataFrame({
        "correct_option_n": [c_corr[c] for c in CATS],
        "correct_option_pct": [round(100 * c_corr[c] / n, 1) for c in CATS],
        "any_option_n": [c_any[c] for c in CATS],
        "any_option_pct": [round(100 * c_any[c] / n, 1) for c in CATS],
    }, index=CATS)

unknown = {l for v in data.values() for labs in v["taxonomy"].values() for l in labs} - set(CATS)
summary["unexpected_label_strings"] = sorted(unknown)
all_lab = collections.Counter(l for v in data.values() for labs in v["taxonomy"].values() for l in labs)
corr_lab = collections.Counter(l for v in data.values() for l in v["taxonomy"].get(str(v["correct"]), []))
summary["unexpected_label_freq_any_option"] = {l: all_lab[l] for l in sorted(unknown)}
summary["unexpected_label_freq_correct_option"] = {l: corr_lab[l] for l in sorted(unknown) if corr_lab[l]}
summary["items_with_unexpected_label_on_correct"] = sum(
    any(l in unknown for l in v["taxonomy"].get(str(v["correct"]), [])) for v in data.values())
lc_full = label_counts(list(data))
lc_ver = label_counts(verified)
lc_full.to_csv(os.path.join(OUT, "egonormia_label_counts_full.csv"))
lc_ver.to_csv(os.path.join(OUT, "egonormia_label_counts_verified.csv"))

# Label co-occurrence on the correct option
co = pd.DataFrame(0, index=CATS, columns=CATS)
for v in data.values():
    labs = v["taxonomy"].get(str(v["correct"]), [])
    labs = [l for l in labs if l in CATS]
    for a in labs:
        for b in labs:
            co.loc[a, b] += 1
co.to_csv(os.path.join(OUT, "egonormia_label_cooccurrence_correct.csv"))

json.dump(summary, open(os.path.join(OUT, "egonormia_audit_summary.json"), "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
print("\nCorrect-option label counts (full):\n", lc_full)
print("\nCorrect-option label counts (verified):\n", lc_ver)
