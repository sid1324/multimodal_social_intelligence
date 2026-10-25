import argparse
import json
import time
from pathlib import Path
import torch

from social_rl.common import ROOT, STORE, append_jsonl, config, read_jsonl, sha256, write_json
from social_rl.model import VideoPolicy
from social_rl.tasks import ego_prompt, r3_prompt, parse_choice, parse_r3, parse_sensible, mismatch_map

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adapter")
    p.add_argument("--conditions", nargs="+", default=["video"], choices=["video", "blind", "mismatch"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=0,
                   help="Generation cap override for a separately labeled ablation")
    p.add_argument("--max-hours", type=float, default=3.5)
    args = p.parse_args()
    rows = read_jsonl(args.manifest)
    if any(r.get("split") == "final" for r in rows):
        raise ValueError("Final EgoNormia is reserved: this midterm evaluator refuses it")
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No accessible evaluation examples")
    if len({r['dataset'] for r in rows}) != 1:
        raise ValueError("Mixed datasets are not supported")
    if rows[0]['dataset'] == 'ego':
        sample_ids = set((ROOT/'egonormia_400_sample_ids.txt').read_text().splitlines())
        if any(r['id'] not in sample_ids for r in rows):
            raise ValueError('EgoNormia midterm evaluation must use the team-supplied 400-ID sample')
    torch.manual_seed(config()["seed"])
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config(), "manifest_sha256": sha256(args.manifest), "adapter": args.adapter,
                "conditions": args.conditions, "ids": [r["id"] for r in rows]}
    if args.max_new_tokens:
        metadata["max_new_tokens_override"] = args.max_new_tokens
    if (run / "metadata.json").exists() and json.loads((run / "metadata.json").read_text()) != metadata:
        raise ValueError("Cannot resume an output directory with different experiment settings")
    write_json(run / "metadata.json", metadata)
    path = run / "predictions.jsonl"
    previous = read_jsonl(path) if path.exists() else []
    done = {(r["id"], r["condition"]) for r in previous}
    mismatches = mismatch_map(rows) if "mismatch" in args.conditions else {}
    policy = VideoPolicy(adapter=args.adapter)
    started = time.monotonic()
    for row in rows:
        for condition in args.conditions:
            if (row["id"], condition) in done:
                continue
            if time.monotonic() - started > args.max_hours * 3600:
                print("TIME_BUDGET_REACHED: resumable predictions saved", flush=True)
                return
            result = {"id": row["id"], "group": row["group"], "dataset": row["dataset"],
                      "condition": condition, "raw": {}, "valid": {}, "errors": {}}
            video = None if condition == "blind" else (mismatches[row["id"]]["video_path"] if condition == "mismatch" else row["video_path"])
            result["input_video_id"] = None if condition == "blind" else (mismatches[row["id"]]["id"] if condition == "mismatch" else row["id"])
            before = time.monotonic()
            tasks = ["r3"] if row["dataset"] == "r3" else ["action", "justification", "sensibility"]
            for task in tasks:
                try:
                    prompt = r3_prompt(row) if task == "r3" else ego_prompt(row, task, result.get("action"), condition == "blind")
                    video_row = mismatches[row['id']] if condition == 'mismatch' else row
                    inputs = policy.inputs(prompt, video, end_seconds=video_row.get('clip_duration_seconds'))
                    tokens, text = policy.generate(inputs, max_new_tokens=args.max_new_tokens or None)
                    result["raw"][task] = text
                    result.setdefault("generation", {})[task] = {"input_tokens": inputs['input_ids'].shape[1],
                        "output_tokens": tokens.numel(),
                        "hit_token_cap": tokens.numel() >= (args.max_new_tokens or config()["max_new_tokens"])}
                    if task == "r3":
                        result["answer"] = parse_r3(text)
                        result["valid"][task] = result["answer"] != -1
                    elif task == "sensibility":
                        result["sensible"], result["valid"][task] = parse_sensible(text)
                    else:
                        result[task], result["valid"][task] = parse_choice(text, action=task == "action")
                    del inputs, tokens
                except Exception as e:
                    result["errors"][task] = f"{type(e).__name__}: {str(e)[:600]}"
                    result["valid"][task] = False
                    torch.cuda.empty_cache()
            result["seconds"] = time.monotonic() - before
            result.update(policy.memory())
            append_jsonl(path, result)
            print(json.dumps({"id": row["id"], "condition": condition, "seconds": round(result['seconds'], 2),
                              "valid": result['valid'], "errors": result['errors']}), flush=True)
            # Abort systemic failures without producing hundreds of empty rows.
            if len(result["errors"]) == len(tasks):
                raise RuntimeError("All tasks failed for an example; inspect saved errors before resuming")

if __name__ == "__main__":
    main()
