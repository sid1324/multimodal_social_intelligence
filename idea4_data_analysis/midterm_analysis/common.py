"""
Shared loading and helpers for the Idea 4 midterm analysis (stage 1: omni-model modality ablation).

Every analysis script imports this, so all numbers are computed the same way:
  - drops items whose model-input audio (audio_prev) is silent (status near_silent in the manifest)
  - finds which option is "None" in each item (the blank behavior in EgoNormia's final_data.json)
  - adds the "None"-removed prediction: the best of the 4 real behaviors, ignoring "None"

Reads (in this folder):
    <run>_scores_all.jsonl   per-item, per-condition scores from 11_score_mimo.py (default run: qwen3omni)
    audio_items.csv          item list with norm categories and site
    audio_manifest.csv       audio extraction status (to find silent clips)
    final_data.json          found automatically: EgoNormia/src/final_dataset/final_data.json above this folder

Every script takes an optional run name, e.g.  python 01_accuracy.py mimo
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "outputs"
EGONORMIA_REL = Path("EgoNormia/src/final_dataset/final_data.json")

DEFAULT_RUN = "qwen3omni"
CONDS = ["blind", "video", "audio", "both"]
LABEL = {"blind": "Options only", "video": "Video", "audio": "Audio", "both": "Video + audio"}
CHANCE_RAW, CHANCE_NO_NONE = 0.20, 0.25


def run_name():
    return sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN


def find_final_data():
    for d in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (d / EGONORMIA_REL).exists():
            return d / EGONORMIA_REL
    sys.exit(f"Could not find {EGONORMIA_REL} in any folder above {SCRIPT_DIR}.")


def load(run=None):
    """One row per item x condition, with extra columns for the analyses."""
    run = run or run_name()
    scores_path = SCRIPT_DIR / f"{run}_scores_all.jsonl"
    if not scores_path.exists():
        sys.exit(f"{scores_path.name} not found in {SCRIPT_DIR}")
    rows = pd.DataFrame([json.loads(l) for l in open(scores_path)])
    rows = rows[rows.get("status", pd.Series("ok", index=rows.index)).fillna("ok") == "ok"].copy()

    items = pd.read_csv(SCRIPT_DIR / "audio_items.csv", keep_default_na=False)
    manifest = pd.read_csv(SCRIPT_DIR / "audio_manifest.csv")
    with open(find_final_data()) as f:
        final_data = json.load(f)

    silent = set(manifest[(manifest.window == "audio_prev") & (manifest.status == "near_silent")].id)
    rows = rows[~rows.id.isin(silent)].copy()

    behaviors = {i: [str(b).strip() for b in final_data[i]["behaviors"]] for i in rows.id.unique()}
    rows["none_pos"] = rows.id.map(lambda i: next(k + 1 for k, b in enumerate(behaviors[i]) if not b))
    rows["correct_none"] = (rows.correct == rows.none_pos).astype(int)
    rows["pred_none"] = (rows.pred == rows.none_pos).astype(int)
    rows["p_none"] = [getattr(r, f"p_{r.none_pos}") for r in rows.itertuples()]
    rows["p_correct"] = [getattr(r, f"p_{r.correct}") for r in rows.itertuples()]

    def best_real(r):
        ps = {k: getattr(r, f"p_{k}") for k in range(1, 6) if k != r.none_pos}
        return max(ps, key=ps.get)

    rows["pred_no_none"] = [best_real(r) for r in rows.itertuples()]
    # accuracy with "None" removed is undefined when the correct answer IS "None" (a handful of items)
    rows["correct_no_none"] = np.where(rows.correct_none == 1, np.nan,
                                       (rows.pred_no_none == rows.correct).astype(float))

    rows = rows.merge(items[["id", "correct_norm_categories"]], on="id", how="left")
    rows["categories"] = rows.correct_norm_categories.fillna("").replace("", "(no category)").str.split("|")
    rows["behaviors"] = rows.id.map(lambda i: [b or "None" for b in behaviors[i]])

    print(f"Run '{run}': {rows.id.nunique()} items x {rows.condition.nunique()} conditions "
          f"(excluded {len(silent)} silent items)\n")
    OUT_DIR.mkdir(exist_ok=True)
    return rows


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a, b):
    """Exact McNemar test on paired 0/1 outcomes. Returns (# right only in a, # right only in b, p)."""
    a, b = np.asarray(a), np.asarray(b)
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    p = binomtest(b_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return a_only, b_only, p