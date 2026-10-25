"""Normalize official annotations and freeze partitions without model outputs."""
import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from social_rl.common import ROOT, STORE, config, sha256, stable_hash, write_json, write_jsonl


def main():
    raw = STORE / "data/raw"
    out = STORE / "data/manifests"
    cfg = config()
    with (raw / "r3_durations.csv").open() as f:
        durations = {r['videoid']: float(r['duration']) for r in csv.DictReader(f)}
    movies = {}
    with (raw / "condensed_clips.csv").open() as f:
        for row in csv.DictReader(f):
            vid = row.get("videoid") or row.get("video_id")
            movie = row.get("imdbid") or row.get("movieid") or row.get("movie_id") or row.get("movie_name") or row.get("title")
            if vid and movie:
                movies[vid] = str(movie)
    r3 = pq.read_table(raw / "r3.parquet").to_pylist()
    splits = {s: [] for s in ["train", "val", "test"]}
    seen, invalid = set(), []
    for row in r3:
        qid = row["QID"]
        choices = row["Options"]
        target = row["Answer_Index"]
        if qid in seen or len(choices) != 5 or target not in range(5):
            invalid.append(qid)
            continue
        seen.add(qid)
        vid = Path(row["video"]).stem
        group = "movie:" + movies[vid] if vid in movies else "video:" + vid
        bucket = stable_hash(group) % 100
        split = "train" if bucket < 80 else "val" if bucket < 90 else "test"
        splits[split].append({"id": qid, "dataset": "r3", "video_id": vid, "group": group,
            "split": split, "question": row["Question"], "options": choices, "answer": target,
            "clip_duration_seconds": durations.get(vid),
            "video_path": str(STORE / "data/r3_videos" / f"{vid}.mp4")})
    for split, rows in splits.items():
        rows.sort(key=lambda r: stable_hash(r["id"]))
        write_jsonl(out / f"r3_{split}_all.jsonl", rows)
    # The team-supplied 400 IDs are the fixed midterm baseline/evaluation panel.
    sample_file = ROOT / "egonormia_400_sample_ids.txt"
    sample_ids = [line.strip() for line in sample_file.read_text().splitlines() if line.strip()]
    if len(sample_ids) != 400 or len(set(sample_ids)) != 400:
        raise ValueError("Expected exactly 400 unique EgoNormia sample IDs")
    # Public preview examples inspected during planning are excluded from the final partition.
    exposed = {"01ccf6b0-5e95-4b18-acd4-a8cc1fd31ddf", "01d301c9-9dbf-4da2-9181-d66a6688f5e6"}
    ego = pq.read_table(raw / "ego.parquet").to_pylist()
    official = json.loads((ROOT / "vendor/EgoNormia/src/final_dataset/final_data.json").read_text())
    unknown = set(sample_ids) - set(official)
    if unknown:
        raise ValueError(f"Sample IDs absent from official repository: {sorted(unknown)[:5]}")
    verified = set(json.loads((ROOT / "vendor/EgoNormia/src/final_dataset/verified_split.json").read_text())["split"])
    exploratory, final, panel = [], [], []
    differences = Counter()
    release_by_id = {row['id']: row for row in ego}
    sample_groups = {uid.split('_')[0] for uid in sample_ids}
    for uid, row in official.items():
        group = uid.split("_")[0]
        split = "midterm" if group in sample_groups or group in exposed else "final"
        taxonomy = row["taxonomy"]
        record = {"id": uid, "dataset": "ego", "group": group, "split": split,
            "behaviors": row["behaviors"], "justifications": row["justifications"],
            "answer": row["correct"], "sensible": row["sensibles"],
            "taxonomy": taxonomy, "verified": uid in verified,
            "video_path": str(STORE / "data/ego_videos" / uid / "video_prev.mp4")}
        (exploratory if split == "midterm" else final).append(record)
        if uid in release_by_id:
            for hf_key, repo_key in [("behaviors", "behaviors"), ("justifications", "justifications"),
                                     ("correct_idx", "correct"), ("sensible_idx", "sensibles")]:
                if release_by_id[uid][hf_key] != row.get(repo_key):
                    differences[hf_key] += 1
    panel = [{"id": uid, "dataset": "ego", "group": uid.split('_')[0], "split": "midterm",
              "behaviors": official[uid]["behaviors"], "justifications": official[uid]["justifications"],
              "answer": official[uid]["correct"], "sensible": official[uid]["sensibles"],
              "taxonomy": official[uid]["taxonomy"], "verified": uid in verified,
              "video_path": str(STORE / "data/ego_videos" / uid / "video_prev.mp4")}
             for uid in sample_ids]
    exploratory.sort(key=lambda r: stable_hash("panel:" + r["id"]))
    final.sort(key=lambda r: r["id"])
    write_jsonl(out / "ego_midterm_all.jsonl", exploratory)
    write_jsonl(out / "ego_midterm_panel.jsonl", panel)
    write_jsonl(out / "ego_final_RESERVED.jsonl", final)
    write_json(out / "splits.json", {"r3": {s: [r["id"] for r in rows] for s, rows in splits.items()},
        "ego_midterm": [r["id"] for r in exploratory], "ego_panel": sample_ids,
        "ego_final_reserved": [r["id"] for r in final]})
    report = {"revisions": {k: v for k, v in cfg.items() if "revision" in k or "commit" in k},
        "checksums": {p.name: sha256(p) for p in raw.iterdir() if p.is_file()},
        "r3_rows": len(r3), "r3_unique_videos": len({r["video"] for r in r3}),
        "r3_movie_mapping_available": len(movies), "r3_invalid_or_duplicate_qids": invalid,
        "r3_split_counts": {s: {"questions": len(rows), "groups": len({r['group'] for r in rows}),
            "videos": len({r['video_id'] for r in rows}), "answer_counts": dict(Counter(r['answer'] for r in rows))}
            for s, rows in splits.items()},
        "ego_release_rows": len(ego), "ego_repository_rows": len(official),
        "ego_release_source_videos": len({r['id'].split('_')[0] for r in ego}),
        "ego_repo_annotation_differences": dict(differences), "ego_verified_release_count": len(verified & set(release_by_id)),
        "ego_sample_sha256": sha256(sample_file), "ego_sample_in_hf_annotations": len(set(sample_ids) & set(release_by_id)),
        "ego_sample_repo_only": len(set(sample_ids) - set(release_by_id)),
        "ego_midterm_count": len(exploratory), "ego_panel_count": len(panel),
        "ego_panel_source_groups": len({r['group'] for r in panel}),
        "ego_panel_answer_counts": dict(Counter(r['answer'] for r in panel)),
        "ego_panel_category_counts": dict(Counter(name for r in panel for name in
            (r['taxonomy'].get(str(r['answer'])) or []))),
        "ego_final_reserved_count": len(final), "ego_panel_verified_count": sum(r['verified'] for r in panel),
        "labels_inspected_for_analysis": "Two records viewed for release schema; no model outputs or error analysis yet"}
    write_json(ROOT / "reports/data_audit.json", report)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
