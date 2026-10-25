"""NormBank canonicalization (plan A1, A2).

Inputs: setting, behavior, FULL constraints. Native label mapping taboo 0 / normal 1 /
expected 2 is verified, never assumed from equal numeric IDs. Field strings are never eval'ed:
a constraints field is re-serialized deterministically only if it is genuinely JSON
(json.loads succeeds with a list/dict); otherwise the decoded string is preserved verbatim.
Actual header (inspected read-only 2026-10-22): setting, behavior, setting-behavior, constraints,
constraints_given, constraint_predict, norm, label, split; split values include train/dev/test
(a "val" alias is mapped to dev if present). constraints_given/constraint_predict are not used.
Model text: setting/behavior are outer-stripped; constraints are the verbatim decoded string.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..constants import NORMBANK_LABEL_MAP, NORMBANK_SPLIT_PRIORITY, RETENTION_MAX, RETENTION_SORT_PREFIX
from ..hashing import canonical_json_bytes, sha256_bytes, sha256_file, sort_key_sha256


class NormBankSchemaError(RuntimeError):
    pass


@dataclass(frozen=True)
class NormBankFieldMap:
    setting: str = "setting"
    behavior: str = "behavior"
    constraints: str = "constraints"
    norm: str | None = "norm"     # string label (taboo/normal/expected)
    label: str | None = "label"   # numeric label, if present must agree with norm
    split: str = "split"
    split_aliases: tuple[tuple[str, str], ...] = (("train", "train"), ("dev", "dev"), ("val", "dev"),
                                                  ("validation", "dev"), ("test", "test"))


def _norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def constraints_text(raw: str) -> tuple[str, bool]:
    """Return (model text, is_structured_json). Never eval/literal_eval."""
    stripped = raw.strip()
    if stripped[:1] in ("[", "{"):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return raw, False
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(", ", ": ")), True
    return raw, False


def resolve_label(row: dict, fm: NormBankFieldMap) -> int:
    lab_from_norm = None
    if fm.norm and fm.norm in row:
        name = row[fm.norm].strip().lower()
        if name not in NORMBANK_LABEL_MAP:
            raise NormBankSchemaError(f"unknown norm label {row[fm.norm]!r}")
        lab_from_norm = NORMBANK_LABEL_MAP[name]
    lab_num = None
    if fm.label and fm.label in row and row[fm.label] != "":
        raw = row[fm.label].strip()
        if raw.lower() in NORMBANK_LABEL_MAP:
            lab_num = NORMBANK_LABEL_MAP[raw.lower()]
        else:
            if raw not in ("0", "1", "2"):
                raise NormBankSchemaError(f"unknown numeric label {raw!r}")
            lab_num = int(raw)
    if lab_from_norm is None and lab_num is None:
        raise NormBankSchemaError("row has neither norm nor label field")
    if lab_from_norm is not None and lab_num is not None and lab_from_norm != lab_num:
        raise NormBankSchemaError(f"norm/label disagreement: {row.get(fm.norm)!r} vs {row.get(fm.label)!r}")
    return lab_from_norm if lab_from_norm is not None else lab_num


def canonical_input_id(setting: str, behavior: str, constraints: str) -> str:
    return sha256_bytes(canonical_json_bytes({
        "setting": _norm_text(setting), "behavior": _norm_text(behavior), "constraints": _norm_text(constraints)}))


def load_rows(path: str | Path) -> tuple[list[dict], list[str], str]:
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:  # tolerate (and drop) a UTF-8 BOM only
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, header, sha256_file(path)


def canonicalize(rows: list[dict], header: list[str], fm: NormBankFieldMap) -> dict:
    for col in (fm.setting, fm.behavior, fm.constraints, fm.split):
        if col not in header:
            raise NormBankSchemaError(f"column {col!r} missing; header={header}")
    aliases = dict(fm.split_aliases)
    raw_count = len(rows)

    # 1. exact complete-row dedupe, provenance retained
    seen: dict[tuple, int] = {}
    unique_rows = []
    dup_rows = 0
    for i, row in enumerate(rows):
        key = tuple((c, row.get(c)) for c in header)
        if key in seen:
            unique_rows[seen[key]]["provenance_rows"].append(i)
            dup_rows += 1
            continue
        seen[key] = len(unique_rows)
        unique_rows.append({"row": row, "provenance_rows": [i]})

    # 2. canonical inputs with split priority test > dev > train
    by_id: dict[str, list[dict]] = defaultdict(list)
    for item in unique_rows:
        row = item["row"]
        split = aliases.get(row[fm.split].strip().lower())
        if split is None:
            raise NormBankSchemaError(f"unknown split value {row[fm.split]!r}")
        cid = canonical_input_id(row[fm.setting], row[fm.behavior], row[fm.constraints])
        by_id[cid].append({**item, "split": split, "label": resolve_label(row, fm)})

    examples, quarantined = [], []
    sb_splits: dict[tuple[str, str], set] = defaultdict(set)
    for cid, items in sorted(by_id.items()):
        best = max(NORMBANK_SPLIT_PRIORITY[it["split"]] for it in items)
        keep = [it for it in items if NORMBANK_SPLIT_PRIORITY[it["split"]] == best]
        for it in items:
            if NORMBANK_SPLIT_PRIORITY[it["split"]] < best:
                quarantined.append({"canonical_id": cid, "split": it["split"], "rows": it["provenance_rows"],
                                    "reason": "cross_split_lower_priority_copy"})
        counts = [0, 0, 0]
        for it in keep:
            counts[it["label"]] += 1
        total = sum(counts)
        row0 = keep[0]["row"]
        ctext, structured = constraints_text(row0[fm.constraints])
        split = keep[0]["split"]
        sb_splits[(_norm_text(row0[fm.setting]), _norm_text(row0[fm.behavior]))].add(split)
        examples.append({
            "example_id": cid,
            "split": split,
            "setting": row0[fm.setting].strip(),
            "behavior": row0[fm.behavior].strip(),
            "constraints": ctext,
            "constraints_structured_json": structured,
            "label_counts": counts,
            "label_probs": [c / total for c in counts],
            "conflicting_labels": sum(1 for c in counts if c) > 1,
            "provenance_rows": sorted(r for it in keep for r in it["provenance_rows"]),
        })
    violations = sorted([list(k) for k, v in sb_splits.items() if len(v) > 1])
    summary = {
        "raw_rows": raw_count,
        "exact_duplicate_rows": dup_rows,
        "canonical_inputs": len(examples),
        "by_split": dict(Counter(e["split"] for e in examples)),
        "cross_split_quarantined_copies": len(quarantined),
        "conflicting_label_inputs": sum(e["conflicting_labels"] for e in examples),
        "setting_behavior_split_violations": len(violations),
    }
    return {"examples": examples, "quarantine": quarantined, "summary": summary,
            "setting_behavior_violations": violations}


def retention_subset(examples: list[dict]) -> list[str]:
    """<=2000 official-test canonical inputs by SHA256('retention20261021|'+id)."""
    test_ids = [e["example_id"] for e in examples if e["split"] == "test"]
    return sorted(test_ids, key=lambda i: sort_key_sha256(RETENTION_SORT_PREFIX, i))[:RETENTION_MAX]
