"""
Accuracy of each input condition by norm category and by collection site, raw and with "None" removed.

Norm categories are multi-label: an item counts once for each category its correct answer carries.

Writes (in outputs/):
    <run>_by_category.csv
    <run>_by_site.csv

Usage:
    python 03_breakdowns.py            # qwen3omni
    python 03_breakdowns.py mimo
"""

import pandas as pd

from common import CONDS, OUT_DIR, load, run_name


def breakdown(df, key):
    raw = df.pivot_table(index=key, columns="condition", values="is_correct", aggfunc="mean")[CONDS] * 100
    nn = df.pivot_table(index=key, columns="condition", values="correct_no_none", aggfunc="mean")[CONDS] * 100
    out = pd.concat([raw.add_prefix("raw_"), nn.add_prefix("no_none_")], axis=1)
    out.insert(0, "n", df.drop_duplicates(["id", key]).groupby(key).size())
    out["audio_gain_raw"] = out.raw_both - out.raw_video
    out["audio_gain_no_none"] = out.no_none_both - out.no_none_video
    return out.sort_values("n", ascending=False)


def show(table, title):
    print(title)
    cols = ["n"] + [f"raw_{c}" for c in CONDS] + [f"no_none_{c}" for c in CONDS] + ["audio_gain_no_none"]
    print(table[cols].round(1).to_string())
    print()


def main():
    run = run_name()
    rows = load(run)

    by_cat = breakdown(rows.explode("categories"), "categories")
    by_site = breakdown(rows, "site")
    by_cat.to_csv(OUT_DIR / f"{run}_by_category.csv")
    by_site.to_csv(OUT_DIR / f"{run}_by_site.csv")

    show(by_cat, "BY NORM CATEGORY (% accuracy; raw chance 20, no-None chance 25)")
    show(by_site, "BY SITE")
    print("Rough 95% CI half-width: about +/-7.5 points at n=140, +/-5.5 at n=270, +/-12 at n=65.")
    print(f"Wrote {run}_by_category.csv and {run}_by_site.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()