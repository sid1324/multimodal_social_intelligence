"""Evaluate six-way egocentric MUSON action selection on a fixed pilot panel."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from social_rl.common import append_jsonl, config, read_jsonl, sha256, stable_hash, write_json
from social_rl.model import VideoPolicy
from social_rl.tasks import MUSON_ACTIONS, muson_prompt, parse_letter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adapter")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--max-hours", type=float, default=2.0)
    args = p.parse_args()
    rows = read_jsonl(args.manifest)
    if not rows or any(r["dataset"] != "muson" or r["split"] == "train" for r in rows):
        raise ValueError("Evaluation requires non-training MUSON manifest")
    rows = sorted(rows, key=lambda r: stable_hash("muson-val-pilot:" + r["id"]))
    if args.limit:
        rows = rows[:args.limit]
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config(), "manifest_sha256": sha256(args.manifest),
                "adapter": args.adapter, "ids": [r["id"] for r in rows]}
    if (run / "metadata.json").exists() and json.loads((run / "metadata.json").read_text()) != metadata:
        raise ValueError("Run metadata changed; refusing incompatible resume")
    write_json(run / "metadata.json", metadata)
    path = run / "predictions.jsonl"
    done = {x["id"] for x in read_jsonl(path)} if path.exists() else set()
    torch.manual_seed(config()["seed"])
    policy = VideoPolicy(adapter=args.adapter)
    started = time.monotonic()
    for row in rows:
        if row["id"] in done:
            continue
        if time.monotonic() - started > args.max_hours * 3600:
            print("TIME_BUDGET_REACHED: resumable predictions saved", flush=True)
            break
        tick = time.monotonic()
        inputs = policy.inputs(muson_prompt(), image_path=row["image_path"])
        tokens, answer = policy.generate(inputs)
        pred = parse_letter(answer, 6)
        record = {"id": row["id"], "group": row["group"], "target": row["answer"],
                  "prediction": pred, "valid": pred in range(6), "correct": pred == row["answer"],
                  "raw": answer, "output_tokens": tokens.numel(),
                  "hit_token_cap": tokens.numel() >= config()["max_new_tokens"],
                  "seconds": time.monotonic()-tick, **policy.memory()}
        append_jsonl(path, record)
        print(json.dumps({k: v for k, v in record.items() if k != "raw"}), flush=True)
    a = read_jsonl(path)
    per_class = {MUSON_ACTIONS[c]: {"n": sum(x["target"] == c for x in a),
                  "accuracy": sum(x["correct"] and x["target"] == c for x in a)/max(1, sum(x["target"] == c for x in a))}
                 for c in range(6)}
    write_json(run / "summary.json", {"attempted": len(a), "panel": len(rows),
               "accuracy": sum(x["correct"] for x in a)/len(a) if a else None,
               "invalid": sum(not x["valid"] for x in a),
               "cap_hits": sum(x["hit_token_cap"] for x in a),
               "prediction_counts": dict(Counter(MUSON_ACTIONS[x["prediction"]] if x["valid"] else "invalid" for x in a)),
               "per_class": per_class})


if __name__ == "__main__":
    main()
