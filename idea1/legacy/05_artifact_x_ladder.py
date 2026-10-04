"""05 - Does the perceptual gain survive on items a text-only probe cannot solve?
Splits items by the cross-validated text-probe outcome (03b) and recomputes the
ladder for the three primary families."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("lad", os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_grounding_ladder.py"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "outputs", "tables")
D = os.path.join(ROOT, "data", "egonormia")
data = json.load(open(os.path.join(D, "final_data.json"))); ev = json.load(open(os.path.join(D, "final_data_eval.json")))
probe = pd.read_csv(os.path.join(T, "lang_probe_item_correct.csv")).set_index("item").text_probe_correct
FAM = {"Gemini-1.5-Flash": ("blind_gemini-15-flash-002", "desc_gemini-15-flash-002", "gemini-15-flash-002"),
       "Gemini-1.5-Pro": ("blind_gemini-15-pro-002", "desc_gemini-15-pro-002", "gemini-15-pro-002"),
       "GPT-4o": ("blind_gpt-4o-240513", "desc_gpt-4o-240513", "gpt-4o-240513")}
def both(key):
    out = {}
    for k, v in ev.items():
        if key in v:
            r = v[key]["best"]["results"]; g = data[k]["correct"]
            out[k] = int(isinstance(r, list) and len(r) == 2 and r[0] == g and r[1] == g)
    return pd.Series(out)
rows = []
for fam, keys in FAM.items():
    S = [both(k) for k in keys]
    common = sorted(set(S[0].index) & set(S[1].index) & set(S[2].index) & set(probe.index))
    for split, val in [("text-probe correct (artifact-easy)", 1), ("text-probe wrong (artifact-hard)", 0)]:
        ids = [k for k in common if probe[k] == val]
        rows.append(dict(family=fam, split=split, n=len(ids), blind=S[0][ids].mean(), desc=S[1][ids].mean(), grid=S[2][ids].mean()))
df = pd.DataFrame(rows); df["desc_to_grid"] = df.grid - df.desc
df.to_csv(os.path.join(T, "artifact_x_ladder.csv"), index=False)
print(df.round(3).to_string(index=False))
# full-coverage models (grid only) on easy vs hard
rows = []
for key in ["gemini-2.5-flash-preview-04-17", "QwenVL-25", "gemini-15-pro-002", "gpt-4o-240513"]:
    s = both(key); ids = [k for k in s.index if k in probe.index]
    e = [k for k in ids if probe[k] == 1]; h = [k for k in ids if probe[k] == 0]
    rows.append(dict(model=key, n_easy=len(e), acc_easy=s[e].mean(), n_hard=len(h), acc_hard=s[h].mean()))
d2 = pd.DataFrame(rows); d2["gap"] = d2.acc_easy - d2.acc_hard
d2.to_csv(os.path.join(T, "artifact_easy_hard_models.csv"), index=False)
print(d2.round(3).to_string(index=False))
