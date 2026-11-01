"""Prepare pinned MUSON v1.0 action-choice manifests without EgoNormia data.

The release has no source-trajectory IDs. Sequential 100-image blocks are
kept together in deterministic splits to reduce likely adjacent-frame
leakage, but this cannot certify trajectory disjointness.
"""
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from social_rl.common import STORE, write_json, write_jsonl


ARCHIVE = STORE / "data/external/MUSON-v1.0.zip"
EXPECTED_SHA256 = "69eeead852191531f06eb0dd474d322dda9bb104ac7aac91b534384ec1c30b06"
OUT = STORE / "data/muson"
ACTIONS = ("move forward", "move forward right", "move forward left",
           "stop", "turn right", "turn left")


def split_for_block(block):
    value = int(hashlib.sha256(f"social-rl-muson-v1:{block}".encode()).hexdigest()[:12], 16) / 16**12
    return "train" if value < 0.8 else "val" if value < 0.9 else "test"


def main():
    digest = hashlib.sha256()
    with open(ARCHIVE, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError("MUSON archive checksum mismatch")
    rows = {k: [] for k in ("train", "val", "test")}
    with zipfile.ZipFile(ARCHIVE) as z:
        records = json.loads(z.read("MUSON/MUSON.json"))
        if len(records) != 10110:
            raise ValueError(f"Unexpected MUSON record count {len(records)}")
        for i, rec in enumerate(records, 1):
            name = f"MUSON_{i:06d}.jpg"
            if rec["image"] != "images/" + name or rec["action"] not in ACTIONS:
                raise ValueError(f"Unexpected MUSON record {i}: {rec['image']}, {rec['action']}")
            target = OUT / "images" / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open("MUSON/images/" + name) as source, target.open("wb") as dest:
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
            with Image.open(target) as im:
                im.verify()
            block = (i - 1) // 100
            split = split_for_block(block)
            rows[split].append({"dataset": "muson", "split": split, "id": f"muson-{i:06d}",
                                "group": f"block-{block:03d}", "image_path": str(target),
                                "answer": ACTIONS.index(rec["action"]),
                                "action": rec["action"], "explanation": rec["explanation"]})
    if any({r['group'] for r in rows[a]} & {r['group'] for r in rows[b]}
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("Sequential blocks cross split boundaries")
    for split, items in rows.items():
        write_jsonl(OUT / f"{split}.jsonl", items)
    audit = {"archive_sha256": EXPECTED_SHA256, "records": len(records),
             "action_order": list(ACTIONS), "split_method": "fixed SHA256 on sequential 100-image blocks",
             "source_trajectory_ids_available": False,
             "counts": {k: {"examples": len(v), "blocks": len({r['group'] for r in v}),
                            "actions": dict(Counter(r['action'] for r in v))} for k, v in rows.items()}}
    write_json(OUT / "audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
