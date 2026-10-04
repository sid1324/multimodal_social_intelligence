"""
Copy the clips for the human-simulation items into their own folder.

Reads p2_data_analysis/human_sim_sheet.csv and copies each item's video_prev.mp4 to
p2_data_analysis/human_sim_clips/, named by the sheet's order so you can go row by row:

    human_sim_clips/01__af215e0f-..._4328-60__prev.mp4
    human_sim_clips/02__2dbdd409-..._1502-69__prev.mp4
    ...

Only video_prev is copied by default: video_during shows the correct action being
performed, so watching it would give the answer away.

Usage:
    python collect_sim_clips.py                       # find your clips folder automatically
    python collect_sim_clips.py /path/to/clips_root   # folder that contains one subfolder per item id
"""

import csv
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SHEET_CSV = SCRIPT_DIR / "human_sim_sheet.csv"
OUT_DIR = SCRIPT_DIR / "human_sim_clips"
FILES_TO_COPY = ["video_prev.mp4"]  # add "video_during.mp4" only if you want it later


def find_clips_root(ids):
    """Search the project folder (one level above p2_data_analysis) for a folder named after
    one of the item ids, and return its parent: that's where the clip folders live."""
    search_from = SCRIPT_DIR.parent
    ids = set(ids)
    for dirpath, dirnames, _ in os.walk(search_from):
        if any(d in ids for d in dirnames):
            return Path(dirpath)
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != OUT_DIR.name]
    sys.exit(
        f"Couldn't find any clip folders under {search_from}. Pass the folder that holds the "
        "per-item subfolders: python collect_sim_clips.py /path/to/clips_root"
    )


def main():
    if not SHEET_CSV.exists():
        sys.exit(f"{SHEET_CSV.name} not found. Run categorize_by_norm.py first.")
    with open(SHEET_CSV, newline="") as f:
        items = [(int(r["order"]), r["id"]) for r in csv.DictReader(f)]

    clips_root = Path(sys.argv[1]) if len(sys.argv) > 1 else find_clips_root(i for _, i in items)
    print(f"Clips folder: {clips_root}\nCopying to:   {OUT_DIR}\n")
    OUT_DIR.mkdir(exist_ok=True)

    copied, missing = 0, []
    width = len(str(len(items)))
    for order, item_id in sorted(items):
        for name in FILES_TO_COPY:
            src = clips_root / item_id / name
            if not src.exists():
                missing.append(f"{order:0{width}d}  {item_id}/{name}")
                continue
            suffix = name.replace("video_", "").replace(".mp4", "")
            dst = OUT_DIR / f"{order:0{width}d}__{item_id}__{suffix}.mp4"
            shutil.copy2(src, dst)
            copied += 1

    print(f"Copied {copied} file(s) for {len(items)} items.")
    if missing:
        print(f"\nMissing {len(missing)}:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()