"""Evaluate frozen or adapted VLM on image-disjoint NormLens split."""
import argparse
import json
import time
from pathlib import Path

import torch

from social_rl.common import append_jsonl, config, read_jsonl, sha256, write_json
from social_rl.model import VideoPolicy
from social_rl.tasks import normlens_prompt, parse_r3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adapter")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=2.0)
    args = p.parse_args()
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    if not rows or any(r["dataset"] != "normlens" or r["split"] == "train" for r in rows):
        raise ValueError("Evaluation requires a non-training NormLens split")
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
        inputs = policy.inputs(normlens_prompt(row), image_path=row["image_path"])
        tokens, answer = policy.generate(inputs)
        pred = parse_r3(answer)
        record = {"id": row["id"], "group": row["group"], "target": row["answer"],
                  "prediction": pred, "valid": pred in range(3), "correct": pred == row["answer"],
                  "raw": answer, "output_tokens": tokens.numel(),
                  "hit_token_cap": tokens.numel() >= config()["max_new_tokens"],
                  "seconds": time.monotonic()-tick, **policy.memory()}
        append_jsonl(path, record)
        print(json.dumps({k: v for k, v in record.items() if k != "raw"}), flush=True)
    a = read_jsonl(path)
    write_json(run / "summary.json", {"attempted": len(a), "panel": len(rows),
               "accuracy": sum(x["correct"] for x in a)/len(a) if a else None,
               "invalid": sum(not x["valid"] for x in a),
               "cap_hits": sum(x["hit_token_cap"] for x in a)})


if __name__ == "__main__":
    main()
