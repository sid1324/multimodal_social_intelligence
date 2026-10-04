"""
Check which EgoNormia items have audio in their Ego4D source video.

Expected layout (paths are found automatically from this script's location):
    <parent>/
        EgoNormia/src/final_dataset/final_data.json
        ego4d_sounds.json
        p2_data_analysis/check_audio.py      (here, or anywhere inside EgoNormia/)

Usage:
    python check_audio.py                              # use the layout above
    python check_audio.py <final_data.json> <ego4d.json>   # override paths

Outputs:
    p2_data_analysis/egonormia_audio_check.csv   one row per EgoNormia item
    printed summary             overall, per norm category, per collection site
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PREV_WINDOW_SEC = 5.0  # video_prev.mp4 covers the 5 s before the action

SCRIPT_DIR = Path(__file__).resolve().parent
EGONORMIA_REL = Path("src/final_dataset/final_data.json")
EGO4D_JSON_NAME = "ego4d_sounds.json"


def find_default_paths():
    """Walk up from this script to find the EgoNormia folder and ego4d_sounds.json beside it."""
    for d in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        egonormia_root = d if d.name == "EgoNormia" else d / "EgoNormia"
        if (egonormia_root / EGONORMIA_REL).exists():
            return egonormia_root / EGONORMIA_REL, egonormia_root.parent / EGO4D_JSON_NAME
    sys.exit(
        "Could not find EgoNormia/src/final_dataset/final_data.json above this script. "
        "Pass both paths explicitly: python check_audio.py <final_data.json> <ego4d.json>"
    )


def parse_id(item_id):
    """'51364391-..._1359-23' -> ('51364391-...', 1359.23)"""
    uid, suffix = item_id.rsplit("_", 1)
    return uid, float(suffix.replace("-", "."))


def interval_bounds(iv):
    """Best-effort read of a redacted interval; returns (start, end) or None."""
    if isinstance(iv, (list, tuple)) and len(iv) == 2:
        return float(iv[0]), float(iv[1])
    if isinstance(iv, dict):
        start = next((iv[k] for k in iv if "start" in k and "sec" in k), None)
        end = next((iv[k] for k in iv if "end" in k and "sec" in k), None)
        if start is not None and end is not None:
            return float(start), float(end)
    return None


def main(egonormia_path, ego4d_path):
    with open(egonormia_path) as f:
        egonormia = json.load(f)
    with open(ego4d_path) as f:
        videos = {v["video_uid"]: v for v in json.load(f)["videos"]}

    rows = []
    for item_id, item in egonormia.items():
        uid, t = parse_id(item_id)
        video = videos.get(uid)
        correct_cats = item["taxonomy"].get(str(item["correct"]), [])

        row = {
            "id": item_id,
            "video_uid": uid,
            "timestamp_sec": t,
            "in_ego4d_json": video is not None,
            "has_audio": False,
            "audio_covers_window": False,
            "video_source": None,
            "scenarios": None,
            "has_redacted_regions": None,
            "window_overlaps_redaction": None,
            "split_av": None,
            "correct_norm_categories": "|".join(correct_cats),
        }

        if video is not None:
            meta = video["video_metadata"]
            a_start, a_dur = meta.get("audio_start_sec"), meta.get("audio_duration_sec")
            row["has_audio"] = a_start is not None and a_dur is not None
            if row["has_audio"]:
                # audio must span from the start of video_prev to the action timestamp
                row["audio_covers_window"] = a_start <= t - PREV_WINDOW_SEC and t <= a_start + a_dur
            row["video_source"] = video.get("video_source")
            row["scenarios"] = "|".join(video.get("scenarios") or [])
            row["has_redacted_regions"] = video.get("has_redacted_regions")
            row["split_av"] = video.get("split_av")  # set if video is in the audio-visual benchmark

            intervals = video.get("redacted_intervals") or []
            if not intervals:
                row["window_overlaps_redaction"] = False
            else:
                bounds = [interval_bounds(iv) for iv in intervals]
                if all(b is not None for b in bounds):
                    lo = t - PREV_WINDOW_SEC
                    row["window_overlaps_redaction"] = any(s < t and e > lo for s, e in bounds)
                # else leave None: interval format not recognized, inspect by hand

        rows.append(row)

    out_path = SCRIPT_DIR / "egonormia_audio_check.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary ----
    n = len(rows)
    found = sum(r["in_ego4d_json"] for r in rows)
    audio = sum(r["has_audio"] for r in rows)
    covered = sum(r["audio_covers_window"] for r in rows)
    usable_uids = {r["video_uid"] for r in rows if r["audio_covers_window"]}

    print(f"EgoNormia items:                   {n}")
    print(f"  source video found in ego4d.json: {found}")
    print(f"  source video has audio:           {audio}")
    print(f"  audio covers the clip window:     {covered}")
    print(f"Source videos to download later:   {len(usable_uids)}")

    def breakdown(title, key_fn):
        totals, hits = defaultdict(int), defaultdict(int)
        for r in rows:
            for key in key_fn(r):
                totals[key] += 1
                hits[key] += r["audio_covers_window"]
        print(f"\n{title}")
        for key in sorted(totals, key=lambda k: -totals[k]):
            pct = 100 * hits[key] / totals[key]
            print(f"  {key:<28} {hits[key]:>5} / {totals[key]:<5} ({pct:5.1f}%)")

    breakdown(
        "Audio coverage by norm category (correct answer):",
        lambda r: r["correct_norm_categories"].split("|") if r["correct_norm_categories"] else ["(none)"],
    )
    breakdown("Audio coverage by collection site:", lambda r: [r["video_source"] or "(not in ego4d.json)"])

    # ---- audio-visual (conversation) benchmark membership ----
    usable = [r for r in rows if r["audio_covers_window"]]
    av = [r for r in usable if r["split_av"]]
    print("\nAudio-visual benchmark (split_av) among items with usable audio:")
    print(f"  items in an AV split:          {len(av)} / {len(usable)}")
    print(f"  source videos in an AV split:  {len({r['video_uid'] for r in av})} / {len(usable_uids)}")
    by_split = defaultdict(int)
    for r in av:
        by_split[r["split_av"]] += 1
    for split, count in sorted(by_split.items(), key=lambda kv: -kv[1]):
        print(f"    split_av = {split:<10} {count} items")
    if av:
        cat_counts = defaultdict(int)
        for r in av:
            for c in (r["correct_norm_categories"].split("|") if r["correct_norm_categories"] else ["(none)"]):
                cat_counts[c] += 1
        print("  AV items by norm category:")
        for c, k in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {c:<28} {k}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        egonormia_file, ego4d_file = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        egonormia_file, ego4d_file = find_default_paths()
    if not ego4d_file.exists():
        sys.exit(f"ego4d metadata not found at {ego4d_file}")
    print(f"EgoNormia data: {egonormia_file}\nEgo4D metadata: {ego4d_file}\n")
    main(egonormia_file, ego4d_file)