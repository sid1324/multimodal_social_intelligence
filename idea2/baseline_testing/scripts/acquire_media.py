"""Fetch only intended official preceding clips or public upstream movie clips."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

from social_rl.common import STORE, append_jsonl, config, read_jsonl, stable_hash, write_jsonl


def fetch_ego(row):
    path = Path(row["video_path"])
    result = {"id": row["id"], "dataset": "ego", "path": str(path)}
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/{config()['ego_revision']}/video/{row['id']}/video_prev.mp4"
            tmp = path.with_suffix(".part")
            with requests.head(url, allow_redirects=True, timeout=(20, 30)) as head:
                head.raise_for_status()
                total = int(head.headers["Content-Length"])
            offset = tmp.stat().st_size if tmp.exists() else 0
            if offset > total:
                tmp.unlink()
                offset = 0
            while offset < total:
                end = min(offset + 4 * 1024 * 1024, total) - 1
                last_error = None
                for attempt in range(5):
                    try:
                        with requests.get(url, headers={"Range": f"bytes={offset}-{end}"},
                                          timeout=(20, 30)) as r:
                            r.raise_for_status()
                            if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-{end}/"):
                                raise IOError("Server did not honor the requested byte range")
                            data = r.content
                            if len(data) != end + 1 - offset:
                                raise IOError("Short byte-range response")
                        with tmp.open("ab") as f:
                            f.write(data)
                        offset = end + 1
                        break
                    except (requests.RequestException, OSError) as e:
                        last_error = e
                        time.sleep(min(2 ** attempt, 8))
                else:
                    raise RuntimeError(f"Byte-range download failed: {last_error}")
            tmp.replace(path)
        import av
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            next(container.decode(video=0))
            result.update(status="ok", bytes=path.stat().st_size,
                          width=stream.width, height=stream.height,
                          duration=float(container.duration or 0) / 1e6)
    except Exception as e:
        result.update(status="failed", error=str(e)[:500])
    return result


def fetch_r3(row):
    path = Path(row["video_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {"id": row["video_id"], "dataset": "r3", "path": str(path), "group": row["group"]}
    try:
        if not path.exists():
            cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress", "--retries", "1",
                   "--socket-timeout", "25", "--no-warnings", "-f", "bestvideo[height<=480][ext=mp4][protocol=https][vcodec^=avc1]/bestvideo[height<=480][ext=mp4]/best[height<=480]",
                   "-o", str(path), "https://www.youtube.com/watch?v=" + row["video_id"]]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
            if run.returncode:
                raise RuntimeError(run.stderr[-800:])
        import av
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            next(container.decode(video=0))
            result.update(status="ok", bytes=path.stat().st_size, width=stream.width,
                          height=stream.height, duration=float(container.duration or 0) / 1e6)
    except Exception as e:
        result.update(status="failed", error=str(e)[:1000])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["r3", "ego"])
    p.add_argument("--limit", type=int, default=60)
    args = p.parse_args()
    manifests = STORE / "data/manifests"
    if args.dataset == "ego":
        rows = read_jsonl(manifests / "ego_midterm_panel.jsonl")[:args.limit]
        # HF's large-file endpoint throttles parallel streams on PSC login nodes;
        # bounded sequential byte ranges make progress and preserve partial files.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = list(pool.map(fetch_ego, rows))
    else:
        rows = sum((read_jsonl(manifests / f"r3_{s}_all.jsonl") for s in ["train", "val", "test"]), [])
        unique = {r["video_id"]: r for r in rows}
        ordered = sorted(unique.values(), key=lambda r: stable_hash("download:" + r["video_id"]))[:args.limit]
        results = []
        for row in ordered:
            result = fetch_r3(row)
            results.append(result)
            append_jsonl(STORE / "data/media_attempts.jsonl", result)
            print(json.dumps(result), flush=True)
            # Bound network failures and avoid hammering a blocked service.
            if len(results) >= 5 and all(x["status"] != "ok" for x in results[-5:]):
                print("Stopped after five consecutive acquisition failures.", flush=True)
                break
            time.sleep(1)
    if args.dataset == "ego":
        for result in results:
            append_jsonl(STORE / "data/media_attempts.jsonl", result)
    ok_paths = {x["path"] for x in results if x["status"] == "ok"}
    # Rebuild eligible manifests using all recorded successful validations, not just this invocation.
    history = read_jsonl(STORE / "data/media_attempts.jsonl")
    latest = {x["path"]: x for x in history}
    ok_paths = {p for p, x in latest.items() if x["status"] == "ok" and Path(p).exists()}
    for split in (["train", "val", "test"] if args.dataset == "r3" else ["midterm_panel"]):
        filename = f"r3_{split}_all.jsonl" if args.dataset == "r3" else "ego_midterm_panel.jsonl"
        eligible = [r for r in read_jsonl(manifests / filename) if r["video_path"] in ok_paths]
        write_jsonl(manifests / f"{args.dataset}_{split}_accessible.jsonl", eligible)
        print(f"{args.dataset} {split}: {len(eligible)} accessible questions", flush=True)
    print(json.dumps({"dataset": args.dataset, "attempted": len(results), "ok": sum(x['status'] == 'ok' for x in results)}))

if __name__ == "__main__":
    main()
