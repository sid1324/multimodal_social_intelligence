"""
Speech-presence proxy for EgoNormia items, using Ego4DSounds' audio-tagger scores.

For every EgoNormia item whose audio covers the clip window (from check_audio.py),
find Ego4DSounds clips that overlap the window [t - 5 s, t] and summarize their
`speech` scores. These are model scores on the real audio, so this previews whether
communication-heavy norms involve more speech, before any Ego4D download.

Expected layout (found automatically from this script's location):
    <parent>/
        EgoNormia/src/final_dataset/...
        ego4d_sounds.json
        train_clips.csv (or train_clips_1.2m.csv), optionally test_clips_11k.csv
        p2_data_analysis/
            check_audio.py
            egonormia_audio_check.csv   <- run check_audio.py first
            speech_proxy.py             <- this file

Usage:
    python speech_proxy.py                       # find the clips CSV(s) automatically
    python speech_proxy.py path/to/train_clips.csv [more.csv ...]

Outputs:
    p2_data_analysis/egonormia_speech_proxy.csv   one row per audio-bearing item
    printed summary: (1) is the speech score usable, (2) window coverage, (3) by category
"""

import sys
from pathlib import Path

import pandas as pd

# ---- settings you may want to change ----
WINDOW_BEFORE_SEC = 5.0   # video_prev covers the 5 s before the action timestamp
WINDOW_AFTER_SEC = 0.0    # raise this to also include the start of video_during
SPEECH_THRESHOLD = 0.5    # an item "has speech" if any overlapping clip scores above this

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_CHECK_CSV = SCRIPT_DIR / "egonormia_audio_check.csv"
CLIP_CSV_NAMES = ["train_clips.csv", "train_clips_1.2m.csv", "test_clips.csv", "test_clips_11k.csv"]
SCORE_COLS = ["speech", "music", "background_music"]


def find_clip_csvs():
    """Look beside this script and beside the EgoNormia folder for the Ego4DSounds CSVs."""
    search_dirs = [SCRIPT_DIR]
    for d in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (d / "EgoNormia").is_dir():
            search_dirs.append(d)
            break
        if d.name == "EgoNormia":
            search_dirs.append(d.parent)
            break
    found = [d / name for d in search_dirs for name in CLIP_CSV_NAMES if (d / name).exists()]
    if not found:
        sys.exit(
            "Could not find train_clips.csv. Put it next to the EgoNormia folder or in "
            "p2_data_analysis, or pass its path: python speech_proxy.py path/to/train_clips.csv"
        )
    return found


def load_clips(paths, wanted_uids):
    frames = []
    for p in paths:
        with open(p) as f:
            sep = "\t" if "\t" in f.readline() else ","
        df = pd.read_csv(p, sep=sep, usecols=["video_uid", "clip_start", "clip_end", *SCORE_COLS])
        df["source_file"] = p.name
        frames.append(df)
        print(f"Loaded {len(df):,} clips from {p.name}")
    all_clips = pd.concat(frames, ignore_index=True)
    return all_clips, all_clips[all_clips["video_uid"].isin(wanted_uids)]


def main(clip_paths):
    if not AUDIO_CHECK_CSV.exists():
        sys.exit(f"{AUDIO_CHECK_CSV.name} not found. Run check_audio.py first.")

    items = pd.read_csv(AUDIO_CHECK_CSV)
    items = items[items["audio_covers_window"].astype(str) == "True"].copy()
    print(f"EgoNormia items with audio covering the window: {len(items):,}\n")

    all_clips, clips = load_clips(clip_paths, set(items["video_uid"]))
    by_video = {uid: g for uid, g in clips.groupby("video_uid")}

    # ---- (1) is the speech score usable? ----
    s = all_clips["speech"]
    print("\n(1) SPEECH SCORE SANITY CHECK (all Ego4DSounds clips)")
    print(f"  median {s.median():.3f} | 90th pct {s.quantile(0.9):.3f} | "
          f"99th pct {s.quantile(0.99):.3f} | max {s.max():.3f}")
    print(f"  clips above {SPEECH_THRESHOLD}: {(s > SPEECH_THRESHOLD).mean():.1%}")
    if s.max() < SPEECH_THRESHOLD:
        print("  WARNING: no clip scores above the threshold. Speech-heavy clips were "
              "probably filtered out, so treat the category numbers below with caution.")

    # ---- per-item matching ----
    records = []
    for _, it in items.iterrows():
        t = float(it["timestamp_sec"])
        lo, hi = t - WINDOW_BEFORE_SEC, t + WINDOW_AFTER_SEC
        g = by_video.get(it["video_uid"])
        hit = g[(g["clip_start"] < hi) & (g["clip_end"] > lo)] if g is not None else g
        n = 0 if hit is None else len(hit)
        rec = {
            "id": it["id"],
            "video_uid": it["video_uid"],
            "timestamp_sec": t,
            "video_source": it["video_source"],
            "correct_norm_categories": it["correct_norm_categories"],
            "n_clips_in_window": n,
        }
        for col in SCORE_COLS:
            rec[f"max_{col}"] = hit[col].max() if n else None
        rec["mean_speech"] = hit["speech"].mean() if n else None
        rec["has_speech"] = bool(n and rec["max_speech"] > SPEECH_THRESHOLD) if n else None
        records.append(rec)

    out = pd.DataFrame(records)
    out_path = SCRIPT_DIR / "egonormia_speech_proxy.csv"
    out.to_csv(out_path, index=False)

    # ---- (2) window coverage ----
    covered = out[out["n_clips_in_window"] > 0]
    print(f"\n(2) WINDOW COVERAGE")
    print(f"  items with >=1 Ego4DSounds clip in the window: {len(covered):,} / {len(out):,} "
          f"({len(covered) / len(out):.1%})")
    print(f"  clips per covered window: median {covered['n_clips_in_window'].median():.0f}, "
          f"max {covered['n_clips_in_window'].max()}" if len(covered) else "  (none)")

    # ---- (3) by norm category ----
    exploded = covered.assign(
        category=covered["correct_norm_categories"].fillna("(none)").replace("", "(none)").str.split("|")
    ).explode("category")
    table = exploded.groupby("category").agg(
        items=("id", "size"),
        mean_max_speech=("max_speech", "mean"),
        pct_has_speech=("has_speech", "mean"),
    ).sort_values("mean_max_speech", ascending=False)

    print(f"\n(3) SPEECH BY NORM CATEGORY (covered items only, correct answer's categories)")
    print(f"  {'category':<26} {'items':>6} {'mean max speech':>16} {'% above ' + str(SPEECH_THRESHOLD):>13}")
    for cat, r in table.iterrows():
        print(f"  {cat:<26} {int(r['items']):>6} {r['mean_max_speech']:>16.3f} {float(r['pct_has_speech']):>12.1%}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else find_clip_csvs()
    main(paths)