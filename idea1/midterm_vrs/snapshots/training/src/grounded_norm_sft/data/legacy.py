"""Comparison with the provisional EXACT-RGB-ONLY manifest (plan A2.1: reaudit all RGB hashes).

Supported legacy schema `legacy_exact_rgb_v1` (canonical_examples_EXACT_RGB_ONLY_PROVISIONAL.jsonl):
  id (64-hex), group (exact-RGB identity hash; exact-only grouping), action, images (file names),
  split, q (per-class fraction strings), votes, mode (int|null), plus provenance fields.

Policy: any RGB-identity difference (changed decoder output, missing/extra file, conflicting file
mapping) or any example identity/vote difference is BLOCKING; nothing is auto-blessed. Split
changes caused by pHash near-duplicate merging are expected under the final rule but are listed
individually (no silent split drift). The legacy `id` derivation is not reproduced; IDs are
compared only through the (RGB hash, canonical action) identity.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Iterable, Mapping

from ..io_utils import iter_jsonl
from .grouping import split_for_group
from .normlens import canonical_action

LEGACY_SCHEMA = "legacy_exact_rgb_v1"
LEGACY_FIELDS = ("id", "group", "action", "images", "split", "q", "votes", "mode")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class LegacySchemaError(RuntimeError):
    pass


def load_legacy(path) -> list[dict]:
    rows = list(iter_jsonl(path))
    for i, r in enumerate(rows):
        missing = [f for f in LEGACY_FIELDS if f not in r]
        if missing:
            raise LegacySchemaError(f"row {i}: missing {missing} (not {LEGACY_SCHEMA})")
        if not _HEX64.fullmatch(str(r["group"])):
            raise LegacySchemaError(f"row {i}: group is not a 64-hex RGB hash")
        if not isinstance(r["images"], list) or not r["images"]:
            raise LegacySchemaError(f"row {i}: images must be a nonempty list")
        if r["split"] not in ("train", "dev", "test"):
            raise LegacySchemaError(f"row {i}: invalid split {r['split']!r}")
        if not isinstance(r["q"], list) or len(r["q"]) != 3:
            raise LegacySchemaError(f"row {i}: q must have three entries")
    return rows


def compare_rgb_identities(current_file_rgb: Mapping[str, str], legacy_rows: Iterable[dict],
                           expected_images: int | None = None) -> dict:
    """Compare every image file's exact RGB identity with the legacy group hash."""
    legacy_file_rgb: dict[str, str] = {}
    conflicts = []
    legacy_rgbs = set()
    for r in legacy_rows:
        legacy_rgbs.add(r["group"])
        for f in r["images"]:
            prev = legacy_file_rgb.setdefault(f, r["group"])
            if prev != r["group"]:
                conflicts.append(f)
    common = sorted(set(legacy_file_rgb) & set(current_file_rgb))
    mismatched = [f for f in common if legacy_file_rgb[f] != current_file_rgb[f]]
    current_rgbs = set(current_file_rgb.values())
    report = {
        "legacy_files": len(legacy_file_rgb), "current_files": len(current_file_rgb),
        "files_only_in_legacy": sorted(set(legacy_file_rgb) - set(current_file_rgb)),
        "files_only_in_current": sorted(set(current_file_rgb) - set(legacy_file_rgb)),
        "legacy_file_conflicts": sorted(set(conflicts)),
        "rgb_mismatched_files": mismatched,
        "legacy_distinct_rgb": len(legacy_rgbs), "current_distinct_rgb": len(current_rgbs),
        "rgb_sets_equal": legacy_rgbs == current_rgbs,
        "expected_distinct_rgb": expected_images,
    }
    ok = (not mismatched and not conflicts and report["rgb_sets_equal"]
          and not report["files_only_in_legacy"] and not report["files_only_in_current"]
          and (expected_images is None or len(current_rgbs) == expected_images))
    report["status"] = "IDENTICAL" if ok else "BLOCKED_RGB_DIFFERENCE"
    return report


def _fractions(q: list) -> list[Fraction]:
    return [Fraction(str(x)) for x in q]


def compare_examples(current_examples: list[dict], legacy_rows: list[dict]) -> dict:
    """Example identities, votes/q/mode, legacy split-rule reproduction and split drift."""
    cur = {(e["rgb_sha256"], e["action_canonical"]): e for e in current_examples}
    old = {}
    dup_legacy = []
    for r in legacy_rows:
        key = (r["group"], canonical_action(r["action"]))
        if key in old:
            dup_legacy.append(r["id"])
        old[key] = r
    common = sorted(set(cur) & set(old))
    vote_diffs, q_diffs, mode_diffs, drift = [], [], [], []
    for k in common:
        e, r = cur[k], old[k]
        if sorted(e["votes"]) != sorted(r["votes"]):
            vote_diffs.append(r["id"])
        total = sum(e["vote_counts"])
        if _fractions(r["q"]) != [Fraction(c, total) for c in e["vote_counts"]]:
            q_diffs.append(r["id"])
        if r["mode"] != e["unique_mode"]:
            mode_diffs.append(r["id"])
        if "split" in e and e["split"] != r["split"]:
            drift.append({"legacy_id": r["id"], "example_id": e["example_id"], "legacy_split": r["split"],
                          "current_split": e["split"], "legacy_group": r["group"], "current_group": e.get("group_id")})
    rule_mismatch = [r["id"] for r in legacy_rows if split_for_group(r["group"]) != r["split"]]
    report = {
        "legacy_examples": len(legacy_rows), "current_examples": len(current_examples),
        "legacy_duplicate_identities": dup_legacy,
        "only_in_legacy": sorted(old[k]["id"] for k in set(old) - set(cur)),
        "only_in_current": sorted(cur[k]["example_id"] for k in set(cur) - set(old)),
        "vote_multiset_differences": vote_diffs, "q_differences": q_diffs, "mode_differences": mode_diffs,
        "legacy_split_rule_not_reproduced": rule_mismatch,
        "split_drift": drift, "split_drift_count": len(drift),
    }
    blocking = (dup_legacy or report["only_in_legacy"] or report["only_in_current"] or vote_diffs
                or q_diffs or mode_diffs or rule_mismatch)
    report["status"] = "BLOCKED_EXAMPLE_DIFFERENCE" if blocking else "IDENTICAL_IDENTITIES"
    return report
