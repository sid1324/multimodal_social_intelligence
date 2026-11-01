"""Prepare image-disjoint, high-agreement NormLens manifests on Ocean.

NormLens is a secondary normative source for a later RL experiment. The
high-agreement labels are unanimous human judgments: 0 wrong, 1 okay,
2 impossible. All artifacts stay outside Git; EgoNormia is never read here.
"""
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from social_rl.common import STORE, write_json, write_jsonl


ARCHIVE = STORE / "data/external/normlens_dataset.zip"
OUT = STORE / "data/normlens"
EXPECTED_SHA256 = "b92f190bca09b748f97a94b2fe058908a93fa3e4ca0025d90f324cd0ed08eba7"


def image_split(name):
    value = int(hashlib.sha256(("social-rl-normlens-v1:" + name).encode()).hexdigest()[:12], 16) / 16**12
    return "train" if value < 0.8 else "val" if value < 0.9 else "test"


def main():
    observed = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if observed != EXPECTED_SHA256:
        raise ValueError(f"Archive checksum mismatch: {observed}")
    with zipfile.ZipFile(ARCHIVE) as z:
        records = json.loads(z.read("dataset/high_agreement.jsonl"))
        rows = {key: [] for key in ("train", "val", "test")}
        for rec in records:
            votes = rec["answer_judgment"]
            if len(set(votes)) != 1 or votes[0] not in (0, 1, 2):
                raise ValueError(f"Not a high-agreement label: {rec['question_id']}")
            name = Path(rec["image"]).name
            if name != rec["image"] or not name.endswith(".jpg"):
                raise ValueError(f"Unsafe image name: {rec['image']}")
            member = "dataset/image/" + name
            target = OUT / "images" / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, target.open("wb") as dest:
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
            with Image.open(target) as im:
                im.verify()
            split = image_split(name)
            rows[split].append({
                "dataset": "normlens", "split": split,
                "id": f"normlens-ha-{rec['question_id']}", "group": name,
                "image_path": str(target), "action_text": rec["text"],
                "answer": votes[0], "n_votes": len(votes),
                "source": rec.get("image_src"),
            })
    all_images = [{r["group"] for r in rs} for rs in rows.values()]
    if any(a & b for i, a in enumerate(all_images) for b in all_images[i + 1:]):
        raise AssertionError("Image leakage across splits")
    for split, items in rows.items():
        write_jsonl(OUT / f"{split}.jsonl", items)
    audit = {"source": "NormLens public archive at pinned upstream commit",
             "archive_sha256": observed, "label_meanings": {"0": "wrong", "1": "okay", "2": "impossible"},
             "split_key": "image filename", "split_method": "fixed SHA256 hash 80/10/10",
             "counts": {k: {"examples": len(v), "images": len({r['group'] for r in v}),
                            "labels": dict(Counter(r['answer'] for r in v))} for k, v in rows.items()}}
    write_json(OUT / "audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
