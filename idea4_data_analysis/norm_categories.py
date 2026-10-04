"""
Group EgoNormia items by norm category (audio-covered items only) and draw a random
sample for the human simulation.

Reads:
    p2_data_analysis/egonormia_audio_check.csv          (run check_audio.py first)
    EgoNormia/src/final_dataset/final_data.json         (found automatically)

Writes (in p2_data_analysis/):
    egonormia_audio_by_norm.csv   one row per audio-covered item, 0/1 column per category
    human_sim_sheet.csv           items to annotate, shuffled, NO answers or categories shown
    human_sim_key.csv             group + correct answer per item. Don't open until you finish!

Usage:
    python categorize_by_norm.py
"""

import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

# ---- sampling settings ----
N_PER_GROUP = 10        # 10 per group -> 30 items total
SEED = 42               # fixed so the sample is reproducible; report it in the write-up
MAX_SITE_SHARE = 0.4    # no single collection site supplies more than 40% of a group

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_CHECK_CSV = SCRIPT_DIR / "egonormia_audio_check.csv"
OUT_CSV = SCRIPT_DIR / "egonormia_audio_by_norm.csv"
SHEET_CSV = SCRIPT_DIR / "human_sim_sheet.csv"
KEY_CSV = SCRIPT_DIR / "human_sim_key.csv"
EGONORMIA_REL = Path("src/final_dataset/final_data.json")

COMM = "Communication/Legibility"
CONTRASTS = ["Safety", "Proxemics"]
GROUPS = [COMM, *CONTRASTS]  # each group = tagged with that category and neither of the others


def col_name(category):
    return "cat_" + category.lower().replace("/", "_").replace(" ", "_")


def find_final_data():
    for d in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        root = d if d.name == "EgoNormia" else d / "EgoNormia"
        if (root / EGONORMIA_REL).exists():
            return root / EGONORMIA_REL
    sys.exit("Could not find EgoNormia/src/final_dataset/final_data.json above this script.")


def summarize(df, all_cats):
    print(f"Audio-covered items: {len(df):,}\n")
    print("(1) ITEMS PER CATEGORY (an item counts once for each category it carries)")
    print(f"  {'category':<26} {'items':>6} {'single-category':>16}")
    for cat in sorted(all_cats, key=lambda c: -df[col_name(c)].sum()):
        n = df[col_name(cat)].sum()
        single = ((df[col_name(cat)] == 1) & (df["n_categories"] == 1)).sum()
        print(f"  {cat:<26} {n:>6} {single:>16}")
    print(f"  {'(no category)':<26} {(df['n_categories'] == 0).sum():>6}")

    print("\n(2) CATEGORIES PER ITEM")
    for k, n in df["n_categories"].value_counts().sort_index().items():
        print(f"  {k} categories: {n} items")

    print("\n(3) CONTRAST POOLS (tagged with one side, not the other)")
    for other in CONTRASTS:
        c, o = df[col_name(COMM)], df[col_name(other)]
        print(f"  {COMM} vs. {other}")
        print(f"    {COMM} and not {other}: {((c == 1) & (o == 0)).sum()}")
        print(f"    {other} and not {COMM}: {((o == 1) & (c == 0)).sum()}")
        print(f"    both (excluded):        {((c == 1) & (o == 1)).sum()}")


def sample_groups(df):
    """Per group: shuffle the exclusive pool, then pick greedily with at most one item per
    source video (across all groups) and a cap on how many come from any one site."""
    rng = random.Random(SEED)
    cap = math.ceil(N_PER_GROUP * MAX_SITE_SHARE)
    used_videos, picks = set(), []

    for group in GROUPS:
        others = [g for g in GROUPS if g != group]
        mask = df[col_name(group)] == 1
        for o in others:
            mask &= df[col_name(o)] == 0
        pool = df[mask].to_dict("records")
        rng.shuffle(pool)

        chosen, site_counts = [], {}
        for relax in (False, True):  # second pass drops the site cap if the pool runs short
            for row in pool:
                if len(chosen) == N_PER_GROUP:
                    break
                site = row["video_source"]
                if row["id"] in {c["id"] for c in chosen} or row["video_uid"] in used_videos:
                    continue
                if not relax and site_counts.get(site, 0) >= cap:
                    continue
                chosen.append(row)
                used_videos.add(row["video_uid"])
                site_counts[site] = site_counts.get(site, 0) + 1
            if len(chosen) == N_PER_GROUP:
                break

        print(f"  {group:<26} pool {len(pool):>4} -> picked {len(chosen):>2} | sites: "
              + ", ".join(f"{s} {n}" for s, n in sorted(site_counts.items(), key=lambda kv: -kv[1])))
        picks += [{**row, "group": group} for row in chosen]
    return picks


def write_sheets(picks, final_data):
    rng = random.Random(SEED + 1)
    rng.shuffle(picks)  # mix groups so the category isn't guessable from position

    sheet, key = [], []
    for order, row in enumerate(picks, start=1):
        item = final_data[row["id"]]
        options = {f"option_{i}": (b if b else "(blank)") for i, b in enumerate(item["behaviors"])}
        sheet.append({
            "order": order,
            "id": row["id"],
            **options,
            "text_answer": "", "text_confidence_1to5": "",
            "video_answer": "", "video_confidence_1to5": "",
            "speech_would_change_yes_maybe_no": "",
            "notes_cues_used": "",
        })
        key.append({
            "order": order,
            "id": row["id"],
            "group": row["group"],
            "correct": item["correct"],
            "sensibles": "|".join(map(str, item.get("sensibles", []))),
            "correct_norm_categories": row["correct_norm_categories"],
            "video_source": row["video_source"],
        })
    pd.DataFrame(sheet).to_csv(SHEET_CSV, index=False)
    pd.DataFrame(key).to_csv(KEY_CSV, index=False)


def main():
    if not AUDIO_CHECK_CSV.exists():
        sys.exit(f"{AUDIO_CHECK_CSV.name} not found. Run check_audio.py first.")

    df = pd.read_csv(AUDIO_CHECK_CSV)
    df = df[df["audio_covers_window"].astype(str) == "True"].copy()
    df["categories"] = df["correct_norm_categories"].fillna("").apply(
        lambda s: [c for c in s.split("|") if c])
    df["n_categories"] = df["categories"].str.len()
    all_cats = sorted({c for cats in df["categories"] for c in cats} | set(GROUPS))
    for cat in all_cats:
        df[col_name(cat)] = df["categories"].apply(lambda cats, c=cat: int(c in cats))

    keep = ["id", "video_uid", "timestamp_sec", "video_source", "correct_norm_categories",
            "n_categories", *[col_name(c) for c in all_cats]]
    df[keep].to_csv(OUT_CSV, index=False)

    summarize(df, all_cats)

    print(f"\n(4) HUMAN SIMULATION SAMPLE (seed {SEED}, {N_PER_GROUP} per group, "
          f"one item per video, site cap {math.ceil(N_PER_GROUP * MAX_SITE_SHARE)})")
    picks = sample_groups(df)

    with open(find_final_data()) as f:
        final_data = json.load(f)
    write_sheets(picks, final_data)

    print(f"\nWrote {OUT_CSV.name}")
    print(f"Wrote {SHEET_CSV.name}  <- annotate this one")
    print(f"Wrote {KEY_CSV.name}  <- answers + groups, open only after annotating")


if __name__ == "__main__":
    main()