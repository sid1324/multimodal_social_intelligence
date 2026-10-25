"""NormLens canonicalization (plan A1, A2).

Actual release schema (inspected read-only 2026-10-22): high_agreement.jsonl / mid_agreement.jsonl
are JSON arrays of objects with keys question_id, image (file name under image/), text (action),
answer_judgment, answer_explanation, image_src, caption. Only image + text are model inputs;
explanations/captions/image_src/agreement/question IDs are provenance only, never inputs/targets.

Whitespace/Unicode policy (explicit):
  * duplicate identity uses NFC + outer strip only (`canonical_action`);
  * model input text is the raw `text` string of the first provenance record, byte-for-byte
    unmodified (no strip, no NFC). Distinct raw variants of one identity are listed.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from ..constants import NORMLENS_CLASSES
from ..hashing import compact_json_bytes, sha256_bytes, sha256_file, stable_id
from .grouping import split_for_group


class SchemaError(RuntimeError):
    pass


@dataclass(frozen=True)
class NormLensFieldMap:
    image: str = "image"
    action: str = "text"
    votes: str = "answer_judgment"
    explanations: str = "answer_explanation"
    question_id: str = "question_id"


RELEASE_FILES = (("high_agreement.jsonl", "HA"), ("mid_agreement.jsonl", "MA"))


@dataclass
class RawRecord:
    file: str
    file_sha256: str
    index: int
    agreement: str  # "HA" | "MA"
    data: dict


def load_release_file(path: str | Path, agreement: str) -> list[RawRecord]:
    """The *.jsonl release files are JSON arrays despite the extension (A1); validate that."""
    if agreement not in ("HA", "MA"):
        raise ValueError("agreement must be HA or MA")
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path} is not a single JSON document (expected JSON array)") from exc
    if not isinstance(data, list):
        raise SchemaError(f"{path}: top-level JSON is {type(data).__name__}, expected array")
    digest = sha256_file(path)
    out = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise SchemaError(f"{path}[{i}] is not an object")
        out.append(RawRecord(file=path.name, file_sha256=digest, index=i, agreement=agreement, data=rec))
    return out


def load_release(dataset_dir: str | Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    for name, agreement in RELEASE_FILES:
        records.extend(load_release_file(Path(dataset_dir) / name, agreement))
    return records


def check_fields_present(records: list[RawRecord], fm: NormLensFieldMap) -> None:
    keys: Counter = Counter()
    for r in records:
        keys.update(r.data.keys())
    required = [fm.image, fm.action, fm.votes, fm.explanations, fm.question_id]
    missing = [f for f in required if keys.get(f, 0) == 0]
    if missing:
        raise SchemaError(f"field(s) {missing} absent from every record; available keys: {sorted(keys)}")


def validate_record(rec: RawRecord, fm: NormLensFieldMap) -> str | None:
    """Return a quarantine reason or None. No repairs are attempted."""
    d = rec.data
    img = d.get(fm.image)
    if not isinstance(img, str) or not img.strip():
        return "missing_or_invalid_image"
    act = d.get(fm.action)
    if not isinstance(act, str) or not act.strip():
        return "missing_or_empty_action"
    votes = d.get(fm.votes)
    if not isinstance(votes, list) or len(votes) == 0:
        return "missing_or_empty_votes"
    if any(type(v) is not int or v not in NORMLENS_CLASSES for v in votes):
        return "vote_not_int_in_0_1_2"
    expl = d.get(fm.explanations)
    if not isinstance(expl, list):
        return "missing_explanations_list"
    if len(expl) != len(votes):
        return "explanation_vote_length_mismatch"
    return None


def canonical_action(text: str) -> str:
    """Duplicate-identity normalization only: Unicode NFC and outer strip."""
    return unicodedata.normalize("NFC", text).strip()


def vote_distribution(votes: list[int]) -> tuple[list[int], list[float], int | None]:
    counts = [0, 0, 0]
    for v in votes:
        counts[v] += 1
    total = sum(counts)
    top = max(counts)
    modes = [c for c in NORMLENS_CLASSES if counts[c] == top]
    return counts, [c / total for c in counts], (modes[0] if len(modes) == 1 else None)


def normalized_entropy(probs) -> float:
    h = -sum(p * math.log(p) for p in probs if p > 0) / math.log(3)
    if h > 1 + 1e-12:
        raise ValueError("normalized entropy exceeds 1")
    return min(max(h, 0.0), 1.0)


@dataclass
class CanonicalResult:
    examples: list[dict]
    quarantine: list[dict]
    summary: dict = field(default_factory=dict)


def canonicalize(records: list[RawRecord], fm: NormLensFieldMap,
                 image_identity: Callable[[str], Mapping]) -> CanonicalResult:
    """Canonical image-action inputs (before grouping).

    image_identity(image_field) -> mapping with rgb_sha256, file_sha256, width, height, phash, path."""
    check_fields_present(records, fm)
    quarantine = []
    by_input: dict[tuple[str, str], list[tuple[RawRecord, dict]]] = defaultdict(list)
    for rec in records:
        reason = validate_record(rec, fm)
        if reason is None:
            try:
                ident = dict(image_identity(rec.data[fm.image]))
            except Exception as exc:  # unreadable image: quarantined with reason, never repaired
                reason = f"image_decode_failed:{type(exc).__name__}"
        if reason:
            quarantine.append({"file": rec.file, "index": rec.index, "agreement": rec.agreement,
                               "question_id": rec.data.get(fm.question_id), "reason": reason})
            continue
        by_input[(ident["rgb_sha256"], canonical_action(rec.data[fm.action]))].append((rec, ident))

    examples = []
    for (rgb, canon), items in sorted(by_input.items()):
        items = sorted(items, key=lambda x: (x[0].agreement != "HA", x[0].file, x[0].index))
        vectors: list[tuple[tuple, str]] = []
        provenance = []
        for rec, ident in items:
            votes = tuple(rec.data[fm.votes])
            expl_hash = sha256_bytes(compact_json_bytes(rec.data[fm.explanations]))
            if (votes, expl_hash) not in vectors:
                vectors.append((votes, expl_hash))
            provenance.append({
                "file": rec.file, "file_sha256": rec.file_sha256, "index": rec.index,
                "agreement": rec.agreement, "question_id": rec.data.get(fm.question_id),
                "image": rec.data[fm.image], "image_file_sha256": ident["file_sha256"],
                "action_raw": rec.data[fm.action], "votes": list(votes), "explanations_sha256": expl_hash,
            })
        all_votes = [v for vec in vectors for v in vec[0]]
        counts, probs, mode = vote_distribution(all_votes)
        agreements = sorted({p["agreement"] for p in provenance})
        ident0 = items[0][1]
        examples.append({
            "example_id": stable_id("normlens_input_v1", rgb, canon),
            "rgb_sha256": rgb,
            "image": items[0][0].data[fm.image],
            "images": sorted({p["image"] for p in provenance}),
            "image_file_sha256s": sorted({p["image_file_sha256"] for p in provenance}),
            "width": ident0["width"], "height": ident0["height"],
            "phash": None if ident0.get("phash") is None else f"{int(ident0['phash']):016x}",
            "action": provenance[0]["action_raw"],
            "action_canonical": canon,
            "action_variants": sorted({p["action_raw"] for p in provenance}),
            "votes": all_votes, "vote_counts": counts, "vote_probs": probs, "unique_mode": mode,
            "normalized_entropy": normalized_entropy(probs),
            "agreement": agreements,
            "agreement_stratum": agreements[0] if len(agreements) == 1 else "mixed",
            "num_source_records": len(provenance),
            "num_distinct_vectors": len(vectors),
            "unknown_rater_overlap": len(vectors) > 1,
            "source_records": provenance,
        })
    ids = [e["example_id"] for e in examples]
    if len(set(ids)) != len(ids):
        raise SchemaError("example_id collision")
    summary = {
        "raw_records": len(records),
        "raw_by_agreement": dict(Counter(r.agreement for r in records)),
        "quarantined": len(quarantine),
        "quarantine_reasons": dict(Counter(q["reason"] for q in quarantine)),
        "canonical_inputs": len(examples),
        "distinct_rgb_images": len({e["rgb_sha256"] for e in examples}),
        "merged_identical_duplicate_records": sum(e["num_source_records"] - e["num_distinct_vectors"]
                                                  for e in examples),
        "multi_vector_examples_unknown_rater_overlap": [e["example_id"] for e in examples
                                                         if e["unknown_rater_overlap"]],
    }
    return CanonicalResult(examples=examples, quarantine=quarantine, summary=summary)


def attach_groups_and_splits(examples: list[dict], group_of: Mapping[str, str]) -> None:
    for e in examples:
        e["group_id"] = group_of[e["rgb_sha256"]]
        e["split"] = split_for_group(e["group_id"])


def split_summary(examples: list[dict]) -> dict:
    out = {}
    for split in ("train", "dev", "test"):
        rows = [e for e in examples if e.get("split") == split]
        modes = Counter("tie" if e["unique_mode"] is None else e["unique_mode"] for e in rows)
        out[split] = {
            "inputs": len(rows), "groups": len({e["group_id"] for e in rows}),
            "images": len({e["rgb_sha256"] for e in rows}),
            "wrong_unique_mode": modes.get(0, 0), "okay_unique_mode": modes.get(1, 0),
            "impossible_unique_mode": modes.get(2, 0), "tied_mode": modes.get("tie", 0),
            "agreement_strata": dict(sorted(Counter(e["agreement_stratum"] for e in rows).items())),
        }
    return out


def apply_ego_exclusion(examples: list[dict], overlap_hits: list[dict]) -> tuple[list[dict], dict]:
    """Exclude train/dev groups matching Ego pre-action frames; log retained test matches."""
    hit_groups = {h["normlens_group_id"] for h in overlap_hits}
    kept = [e for e in examples if not (e["group_id"] in hit_groups and e["split"] in ("train", "dev"))]
    return kept, {
        "overlap_hits": len(overlap_hits), "matched_groups": sorted(hit_groups),
        "excluded_train_dev_examples": sorted(e["example_id"] for e in examples if e not in kept),
        "matched_test_examples_retained": sorted(e["example_id"] for e in examples
                                                 if e["group_id"] in hit_groups and e["split"] == "test"),
    }
