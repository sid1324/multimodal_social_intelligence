"""Audit PsycheMME training metadata and source overlap, without EgoNormia labels.

This produces candidate IDs, not a training/validation split. Fetch metadata
from the pinned Hugging Face revision before running; no media is downloaded.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


REVISION = "3b80a47473802b160b54e760d5dca070efd96f4c"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ego-split-ids", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    download = json.loads((args.root / "download_manifest.json").read_text())
    if download["revision"] != REVISION:
        raise ValueError("Unexpected source revision")
    for name, info in download["files"].items():
        if digest(args.root / name) != info["sha256"]:
            raise ValueError(f"Metadata changed: {name}")
    sft = read_jsonl(args.root / "sft/train.jsonl")
    grpo = read_jsonl(args.root / "grpo/train.jsonl")
    episodes = read_jsonl(args.root / "metadata/episodes.jsonl")
    questions = read_jsonl(args.root / "metadata/questions.jsonl")
    ep = {x["episode_id"]: x for x in episodes}
    qs = {x["id"]: x for x in questions}
    if len(qs) != len(questions) or len({x['id'] for x in sft}) != len(sft):
        raise ValueError("Duplicate question IDs")
    if {x['id'] for x in sft} != set(qs):
        raise ValueError("SFT and question metadata differ")
    if not all(x['episode_id'] in ep for x in questions):
        raise ValueError("Question missing source mapping")
    # Only the pre-existing ID-only split file is read. No final questions,
    # answers, model predictions, or videos enter this exclusion audit.
    splits = json.loads(args.ego_split_ids.read_text())
    overlap = {}
    excluded_episodes = set()
    for split in ("ego_panel", "ego_midterm", "ego_final_reserved"):
        groups = {x.split('_')[0] for x in splits[split]}
        matched = [x for x in episodes if x['source_dataset'] == 'Ego4D'
                   and x['source_video_id'] in groups]
        matched_ids = {x['episode_id'] for x in matched}
        excluded_episodes |= matched_ids
        overlap[split] = {
            "source_videos": len({x['source_video_id'] for x in matched}),
            "episodes": len(matched),
            "questions": sum(x['episode_id'] in matched_ids for x in questions),
        }
    excluded = [x['id'] for x in sft if qs[x['id']]['episode_id'] in excluded_episodes]
    candidates = [x for x in sft if qs[x['id']]['episode_id'] not in excluded_episodes]
    answers, slots, active_counts, option_counts = Counter(), Counter(), Counter(), Counter()
    for row in sft:
        target = json.loads(row['messages'][-1]['content'])
        active = [k for k, v in target.items() if k != 'answer' and v]
        slots.update(active)
        active_counts[len(active)] += 1
        if qs[row['id']]['task_type'] == 'single_choice':
            answers[target['answer']] += 1
        option_counts[len(re.findall(r'^([A-Z])\. ', row['messages'][0]['content'], re.M))] += 1
    source = lambda row: ep[qs[row['id']]['episode_id']]['source_dataset']
    result = {
        "repository": "psychemme/PsycheMME-Training", "revision": REVISION,
        "metadata_sha256": {n: i['sha256'] for n, i in download['files'].items()},
        "ego_id_only_split_sha256": digest(args.ego_split_ids),
        "sft_examples": len(sft), "grpo_pool_examples": len(grpo),
        "metadata_episodes": len(episodes),
        "used_sft_episodes": len({qs[x['id']]['episode_id'] for x in sft}),
        "metadata_source_video_groups": len({(x['source_dataset'], x['source_video_id']) for x in episodes}),
        "task_types": dict(Counter(x['task_type'] for x in questions)),
        "categories": dict(Counter(x['category'] for x in questions)),
        "question_sources": dict(Counter(source(x) for x in sft)),
        "single_choice_answers": dict(answers), "option_counts": dict(option_counts),
        "active_target_slots": dict(slots), "nonempty_slots_per_target": dict(active_counts),
        "frames_per_question": dict(Counter(len(x['images']) for x in sft)),
        "asr_records": len(read_jsonl(args.root / 'metadata/asr.jsonl')),
        "ego_source_overlap": overlap,
        "excluded_sft_examples": len(excluded), "eligible_sft_examples": len(candidates),
        "eligible_categories": dict(Counter(qs[x['id']]['category'] for x in candidates)),
        "eligible_sources": dict(Counter(source(x) for x in candidates)),
        "eligibility_note": "Candidate IDs only; session-aware source splitting and quality audit remain required.",
        "overlap_limit": "Exact source-video identifiers only; not a perceptual duplicate or recording-session audit.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'psychemme_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'eligible_sft_ids.txt').write_text(''.join(x['id'] + '\n' for x in candidates))
    (args.output / 'excluded_sft_ids.txt').write_text(''.join(x + '\n' for x in excluded))
    print(json.dumps({k: v for k, v in result.items() if k != 'metadata_sha256'}, indent=2))


if __name__ == '__main__':
    main()
