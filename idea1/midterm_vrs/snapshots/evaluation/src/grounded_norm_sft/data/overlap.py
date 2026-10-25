"""Plan A2.7: NormLens / EgoNormia pre-action-frame overlap exclusion (post-Ego-overlap manifest).

Inputs (all read-only):
  * a prepared NormLens directory (write-once `normlens_examples.jsonl` with RGB identity, pHash,
    group and split per example; sidecar-verified);
  * the frozen Ego ID manifest (`egonormia_eval_ids/v1`, pinned file SHA256) and the media chain
    produced by scripts/egonormia_media.py: plan -> download receipt -> decode report
    (`egonormia_media/v2`, decode protocol `egonormia_a5_sheet/v2`). Every link is re-hashed and
    cross-checked; the decode report must cover every frozen ID, in order, with explicit status.
No Ego gold/labels are read (the frozen manifest's label-field *names* are not used).

Match rule (consistent with A2.5): a NormLens image matches an ORIGINAL selected decoded Ego frame
if the exact RGB identities are equal OR the 64-bit pHash Hamming distance is <= 4. Every matched
train/dev NormLens group is excluded IN FULL (all images, all actions); matched test groups are kept
and reported. Splits are never redrawn or reassigned and nothing is excluded by label. Decode or
download failures are recorded in a separate coverage ledger; they are not compared, not replaced,
and make the result INCOMPLETE. This output is a Seal A prerequisite only, not an approval.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..constants import EGO_MEDIA_REVISION, EGO_SOURCE_SHA256, PHASH_MAX_HAMMING
from ..hashing import compact_json_bytes, sha256_bytes, sha256_file
from ..io_utils import read_json, read_jsonl, write_once_json, write_once_jsonl

OVERLAP_PROTOCOL = "normlens_ego_overlap/v1"
FROZEN_SCHEMA = "egonormia_eval_ids/v1"
FROZEN_MANIFEST_SHA256 = "51d2b3b34712b5e04f423b3bf01373c4b2b4bf7b8add3776db7a693792c9a079"
FROZEN_COUNT = 400
MEDIA_SCHEMA = "egonormia_media/v2"
DECODE_PROTOCOL = "egonormia_a5_sheet/v2"
DECODE_STATUSES = ("ok", "decode_failed", "excluded_download_failed")
FRAMES = 8
REPO = "open-social-world/EgoNormia"
HF = "https://huggingface.co"
INDEX_RULE = "floor(k*(T-1)/7+0.5)"
# Decoder pins of the v2 decode protocol (scripts/egonormia_media.py; import/distribution versions).
PINNED_PACKAGES = {
    "av": {"import": "18.1.0", "distribution": "18.1.0"},
    "pillow": {"import": "13.0.0", "distribution": "13.0.0"},
    "imagehash": {"import": "4.3.2", "distribution": "4.3.2"},
    "numpy": {"import": "1.26.4", "distribution": "1.26.4"},
    "scipy": {"import": "1.14.1", "distribution": "1.14.1"},
    "pywavelets": {"import": "1.8.0", "distribution": "1.9.0"},
}
PINNED_FFMPEG = {"avutil": "60.26.102", "avcodec": "62.28.102", "avformat": "62.12.102", "avdevice": "62.3.102",
                 "avfilter": "11.14.102", "swscale": "9.5.102", "swresample": "6.3.102"}


def media_path(ego_id: str) -> str:
    return f"video/{ego_id}/video_prev.mp4"


def media_url(ego_id: str) -> str:
    return f"{HF}/datasets/{REPO}/resolve/{EGO_MEDIA_REVISION}/{media_path(ego_id)}"


def _valid_source(sha, size) -> bool:
    return isinstance(sha, str) and bool(_HEX64.fullmatch(sha)) and type(size) is int and size > 0
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX16 = re.compile(r"[0-9a-f]{16}")


class OverlapError(RuntimeError):
    pass


def _strict_json_bytes(raw: bytes):
    def hook(pairs):
        keys = [k for k, _ in pairs]
        if len(set(keys)) != len(keys):
            raise OverlapError("duplicate JSON object keys")
        return dict(pairs)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OverlapError(f"not strict UTF-8 JSON: {exc}") from exc


def _load(path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    obj = _strict_json_bytes(raw)
    if not isinstance(obj, dict):
        raise OverlapError(f"{path}: top-level JSON object required")
    return obj, sha256_bytes(raw)


def _expect(obj: dict, static: dict, what: str) -> None:
    bad = [k for k, v in static.items() if obj.get(k) != v]
    if bad:
        raise OverlapError(f"{what} does not match the pinned protocol: fields {bad}")


# ------------------------------------------------------------------ Ego chain validation
def validate_ego_chain(frozen_path, plan_path, receipt_path, decode_path, *,
                       expected_frozen_sha256: str = FROZEN_MANIFEST_SHA256,
                       expected_count: int = FROZEN_COUNT, phash_field: str = "frame_phash") -> dict:
    frozen, frozen_sha = _load(frozen_path)
    if frozen_sha != expected_frozen_sha256:
        raise OverlapError(f"frozen manifest SHA256 {frozen_sha} != accepted {expected_frozen_sha256}")
    _expect(frozen, {"schema": FROZEN_SCHEMA, "media_revision": EGO_MEDIA_REVISION}, "frozen manifest")
    if (frozen.get("annotation") or {}).get("sha256") != EGO_SOURCE_SHA256:
        raise OverlapError("frozen manifest annotation SHA256 is not the pinned production source")
    ids = frozen.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        raise OverlapError("frozen ids must be a list of nonempty strings")
    if len(set(ids)) != len(ids):
        raise OverlapError("duplicate frozen IDs")
    if len(ids) != expected_count or (frozen.get("counts") or {}).get("frozen") != len(ids):
        raise OverlapError(f"frozen manifest must hold exactly {expected_count} IDs")
    ids_sha = sha256_bytes(compact_json_bytes(ids))
    if frozen.get("ids_sha256") != ids_sha:
        raise OverlapError("frozen ids_sha256 does not match its ID list")
    frozen_ref = {"filename": Path(frozen_path).name, "file_sha256": frozen_sha, "ids_sha256": ids_sha,
                  "count": len(ids)}

    plan, plan_sha = _load(plan_path)
    _expect(plan, {"schema": MEDIA_SCHEMA, "kind": "plan", "repo": REPO, "media_revision": EGO_MEDIA_REVISION,
                   "annotation": frozen["annotation"]}, "media plan")
    pf = plan.get("frozen") or {}
    if (pf.get("file_sha256"), pf.get("ids_sha256"), pf.get("count")) != (frozen_sha, ids_sha, len(ids)):
        raise OverlapError("media plan does not reference this frozen manifest")
    p_entries = plan.get("entries")
    if not isinstance(p_entries, list) or [e.get("id") if isinstance(e, dict) else None for e in p_entries] != ids:
        raise OverlapError("media plan entries must equal the frozen IDs in order")
    for p in p_entries:
        if (p.get("path"), p.get("url")) != (media_path(p["id"]), media_url(p["id"])) or not _valid_source(
                p.get("lfs_sha256"), p.get("lfs_size")):
            raise OverlapError(f"{p['id']}: plan entry is not a pinned pre-action source (path/url/sha256/size)")
    if plan.get("total_bytes") != sum(p["lfs_size"] for p in p_entries):
        raise OverlapError("media plan total_bytes does not equal the sum of entry sizes")

    receipt, receipt_sha = _load(receipt_path)
    _expect(receipt, {"schema": MEDIA_SCHEMA, "kind": "download_receipt", "repo": REPO,
                      "media_revision": EGO_MEDIA_REVISION, "annotation": frozen["annotation"],
                      "plan_sha256": plan_sha, "frozen": plan["frozen"]}, "download receipt")
    r_entries = receipt.get("entries")
    if not isinstance(r_entries, list) or [e.get("id") if isinstance(e, dict) else None for e in r_entries] != ids:
        raise OverlapError("download receipt must cover every frozen ID in order")
    for r, p in zip(r_entries, p_entries):
        if (r.get("path"), r.get("expected_lfs_sha256"), r.get("expected_lfs_size")) != (
                p["path"], p["lfs_sha256"], p["lfs_size"]):
            raise OverlapError(f"{r['id']}: receipt expected source differs from the plan entry")
        if r.get("status") == "verified":
            if (r.get("observed_sha256"), r.get("observed_size")) != (p["lfs_sha256"], p["lfs_size"]):
                raise OverlapError(f"{r['id']}: receipt marks verified without matching observed source")
        elif r.get("status") != "failed":
            raise OverlapError("download receipt statuses must be verified or failed")
    rc = Counter(r["status"] for r in r_entries)
    if receipt.get("counts") != {"requested": len(ids), "verified": rc["verified"], "failed": rc["failed"]}:
        raise OverlapError("download receipt counts are inconsistent with its entries")

    decode, decode_sha = _load(decode_path)
    _expect(decode, {"schema": MEDIA_SCHEMA, "kind": "decode_report", "decode_protocol": DECODE_PROTOCOL,
                     "media_revision": EGO_MEDIA_REVISION, "receipt_sha256": receipt_sha, "plan_sha256": plan_sha,
                     "frozen": plan["frozen"], "annotation": frozen["annotation"],
                     "pins": {"packages": PINNED_PACKAGES, "ffmpeg_libraries": PINNED_FFMPEG}}, "decode report")
    _check_versions(decode.get("versions"))
    geometry = decode.get("geometry") or {}
    if (geometry.get("frames"), geometry.get("index_rule")) != (FRAMES, INDEX_RULE):
        raise OverlapError("decode report geometry is not the A5 eight-frame rule")
    entries = decode.get("entries")
    if not isinstance(entries, list) or [e.get("id") if isinstance(e, dict) else None for e in entries] != ids:
        raise OverlapError("decode report must cover every frozen ID exactly once, in frozen order")
    counts = Counter()
    for e, r, p in zip(entries, r_entries, p_entries):
        st = e.get("status")
        if st not in DECODE_STATUSES:
            raise OverlapError(f"{e['id']}: unknown decode status {st!r}")
        if e.get("download_status") != r["status"] or (st == "excluded_download_failed") != (r["status"] == "failed"):
            raise OverlapError(f"{e['id']}: decode status inconsistent with the download receipt")
        if (e.get("expected_lfs_sha256"), e.get("expected_lfs_size")) != (p["lfs_sha256"], p["lfs_size"]):
            raise OverlapError(f"{e['id']}: decode expected source differs from the plan entry")
        counts[st] += 1
        if st == "ok":
            if (e.get("observed_source_sha256"), e.get("observed_source_size")) != (p["lfs_sha256"], p["lfs_size"]):
                raise OverlapError(f"{e['id']}: decoded source identity differs from the pinned plan entry")
            _check_frames(e, phash_field)
    want = {"requested": len(ids), **{s: counts[s] for s in DECODE_STATUSES}}
    if decode.get("counts") != want:
        raise OverlapError("decode report counts are inconsistent with its entries")
    return {"ids": ids, "entries": entries, "phash_field": phash_field, "counts": want,
            "inputs": {"frozen_manifest": frozen_ref, "plan_sha256": plan_sha, "download_receipt_sha256": receipt_sha,
                       "decode_report_sha256": decode_sha, "decode_protocol": DECODE_PROTOCOL,
                       "media_schema": MEDIA_SCHEMA, "media_revision": EGO_MEDIA_REVISION,
                       "source_sha256": EGO_SOURCE_SHA256, "decoder_versions": decode.get("versions")}}


def _check_versions(versions) -> None:
    """Observed decoder versions must equal the v2 protocol pins (python version is recorded only)."""
    if not isinstance(versions, dict):
        raise OverlapError("decode report lacks decoder versions")
    pk, libs = versions.get("packages") or {}, versions.get("ffmpeg_libraries") or {}
    bad = [k for k, v in PINNED_PACKAGES.items() if pk.get(k) != v]
    bad += [k for k, v in PINNED_FFMPEG.items() if libs.get(k) != v]
    if bad:
        raise OverlapError(f"decode report decoder versions differ from the v2 pins: {bad}")


def _check_frames(e: dict, phash_field: str) -> None:
    from .egonormia import frame_indices

    idx, sizes = e.get("frame_indices"), e.get("frame_sizes")
    rgb, ph = e.get("frame_rgb_identity"), e.get(phash_field)
    for name, v in (("frame_indices", idx), ("frame_sizes", sizes), ("frame_rgb_identity", rgb), (phash_field, ph)):
        if not isinstance(v, list) or len(v) != FRAMES:
            raise OverlapError(f"{e['id']}: {name} must list exactly {FRAMES} selected frames")
    total = e.get("decoded_frames")
    if type(total) is not int or total < 1 or idx != frame_indices(total):
        raise OverlapError(f"{e['id']}: frame_indices are not the A5 rule for decoded_frames={total!r}")
    if any(not isinstance(s, list) or len(s) != 2 or any(type(v) is not int or v <= 0 for v in s) for s in sizes):
        raise OverlapError(f"{e['id']}: frame_sizes must be [width, height] positive ints")
    if any(not isinstance(s, str) or not _HEX64.fullmatch(s) for s in rgb):
        raise OverlapError(f"{e['id']}: frame_rgb_identity must be 64-hex SHA256 values")
    if any(not isinstance(s, str) or not _HEX16.fullmatch(s) for s in ph):
        raise OverlapError(f"{e['id']}: {phash_field} must be 16-hex 64-bit pHashes")
    first: dict[int, tuple] = {}
    for k in range(FRAMES):  # every repeat of a source frame index must carry identical identities
        ident = (rgb[k], ph[k], tuple(sizes[k]))
        if first.setdefault(idx[k], ident) != ident:
            raise OverlapError(f"{e['id']}: repeated frame index with different identities")


# ------------------------------------------------------------------ NormLens side
def load_prepared_normlens(prepared_dir) -> dict:
    prepared_dir = Path(prepared_dir)
    examples = read_jsonl(prepared_dir / "normlens_examples.jsonl", verify=True)
    summary = read_json(prepared_dir / "normlens_summary.json", verify=True)
    if not examples:
        raise OverlapError("prepared NormLens manifest is empty")
    phash: dict[str, int] = {}
    group: dict[str, str] = {}
    group_split: dict[str, str] = {}
    for e in examples:
        if group_split.setdefault(e["group_id"], e["split"]) != e["split"]:
            raise OverlapError(f"group {e['group_id']} spans more than one split (whole-group exclusion "
                               "requires one assigned split)")
        p = e.get("phash")
        if not isinstance(p, str) or not _HEX16.fullmatch(p):
            raise OverlapError(f"{e.get('example_id')}: prepared example lacks a 16-hex pHash")
        if phash.setdefault(e["rgb_sha256"], int(p, 16)) != int(p, 16):
            raise OverlapError("one RGB identity carries two different pHashes")
        if group.setdefault(e["rgb_sha256"], e["group_id"]) != e["group_id"]:
            raise OverlapError("one RGB identity is in two groups")
        if e["split"] not in ("train", "dev", "test"):
            raise OverlapError(f"{e['example_id']}: invalid split")
    return {"examples": examples, "phash": phash, "group": group,
            "manifest_sha256": (prepared_dir / "normlens_examples.jsonl.sha256").read_text().split()[0],
            "prepared_status": summary.get("status")}


# ------------------------------------------------------------------ matching / exclusion
def find_matches(normlens: dict, chain: dict, max_hamming: int = PHASH_MAX_HAMMING) -> list[dict]:
    split_of = {e["group_id"]: e["split"] for e in normlens["examples"]}
    nl = sorted(normlens["phash"].items())
    cache: dict[tuple[str, int], list[tuple[str, str, int]]] = {}
    out = []
    for e in chain["entries"]:
        if e["status"] != "ok":
            continue
        for k in range(FRAMES):
            frgb, fph = e["frame_rgb_identity"][k], int(e[chain["phash_field"]][k], 16)
            key = (frgb, fph)
            if key not in cache:
                hits = []
                for rgb, ph in nl:
                    d = bin(ph ^ fph).count("1")
                    if rgb == frgb or d <= max_hamming:
                        hits.append((rgb, "exact_rgb" if rgb == frgb else "phash_hamming", d))
                cache[key] = hits
            for rgb, reason, d in cache[key]:
                g = normlens["group"][rgb]
                out.append({"ego_id": e["id"], "sheet_position": k, "frame_index": e["frame_indices"][k],
                            "frame_rgb_identity": frgb, "frame_phash": f"{fph:016x}", "normlens_rgb_sha256": rgb,
                            "normlens_group_id": g, "normlens_split": split_of[g], "reason": reason, "hamming": d})
    return sorted(out, key=lambda m: (m["normlens_group_id"], m["normlens_rgb_sha256"], m["ego_id"],
                                      m["sheet_position"]))


def apply_exclusion(examples: list[dict], matches: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove every example of every matched train/dev group; test groups are retained."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        by_group[m["normlens_group_id"]].append(m)
    excluded_groups = {g for g, ms in by_group.items() if ms[0]["normlens_split"] in ("train", "dev")}
    kept = [e for e in examples if e["group_id"] not in excluded_groups]
    log = [{"example_id": e["example_id"], "group_id": e["group_id"], "split": e["split"],
            "rgb_sha256": e["rgb_sha256"],
            "traced_matches": [{k: m[k] for k in ("ego_id", "sheet_position", "frame_index", "normlens_rgb_sha256",
                                                  "reason", "hamming")} for m in by_group[e["group_id"]]]}
           for e in examples if e["group_id"] in excluded_groups]
    return kept, sorted(log, key=lambda r: (r["group_id"], r["example_id"]))


def _split_counts(examples: list[dict]) -> dict:
    return {s: {"examples": sum(e["split"] == s for e in examples),
                "groups": len({e["group_id"] for e in examples if e["split"] == s}),
                "images": len({e["rgb_sha256"] for e in examples if e["split"] == s})}
            for s in ("train", "dev", "test")}


def run_overlap(prepared_dir, frozen_path, plan_path, receipt_path, decode_path, out_dir, **chain_kwargs) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise OverlapError(f"output directory {out_dir} is not empty (outputs are write-once)")
    chain = validate_ego_chain(frozen_path, plan_path, receipt_path, decode_path, **chain_kwargs)
    nl = load_prepared_normlens(prepared_dir)
    matches = find_matches(nl, chain)
    kept, log = apply_exclusion(nl["examples"], matches)
    ledger = [{"order": i, "ego_id": e["id"], "status": e["status"], "compared": e["status"] == "ok",
               "reason": e.get("reason")} for i, e in enumerate(chain["entries"])]
    not_compared = [r for r in ledger if not r["compared"]]
    groups_by_split = {s: sorted({m["normlens_group_id"] for m in matches if m["normlens_split"] == s})
                       for s in ("train", "dev", "test")}
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {
        "post_exclusion_manifest": write_once_jsonl(out_dir / "normlens_examples_post_ego_overlap.jsonl", kept),
        "matches": write_once_jsonl(out_dir / "ego_overlap_matches.jsonl", matches),
        "exclusion_log": write_once_jsonl(out_dir / "ego_overlap_exclusion_log.jsonl", log),
        "coverage_ledger": write_once_jsonl(out_dir / "ego_overlap_coverage_ledger.jsonl", ledger),
    }
    complete = not not_compared
    summary = {
        "protocol": OVERLAP_PROTOCOL, "match_rule": f"exact RGB identity OR pHash64 Hamming <= {PHASH_MAX_HAMMING} "
        "against each original selected decoded Ego frame (all 8 sheet positions incl. repeats)",
        "exclusion_rule": "every matched train/dev group excluded in full; test groups retained and reported; "
                          "splits never redrawn; no label-based exclusion",
        "inputs": {**chain["inputs"], "phash_field": chain["phash_field"],
                   "prepared_normlens_manifest_sha256": nl["manifest_sha256"],
                   "prepared_status": nl["prepared_status"]},
        "coverage": {**chain["counts"], "compared": len(ledger) - len(not_compared),
                     "not_compared": len(not_compared), "complete": complete},
        "matches": {"total": len(matches), "by_reason": dict(sorted(Counter(m["reason"] for m in matches).items())),
                    "ego_ids_with_match": len({m["ego_id"] for m in matches}),
                    "groups_by_split": groups_by_split},
        "excluded": {"groups": len(groups_by_split["train"]) + len(groups_by_split["dev"]), "examples": len(log)},
        "counts_before": _split_counts(nl["examples"]), "counts_after": _split_counts(kept),
        "test_overlap_retained_examples": sum(1 for e in kept if e["group_id"] in set(groups_by_split["test"])),
        "outputs": digests,
        "status": "POST_EGO_OVERLAP_COMPLETE_PRE_SEAL_A" if complete else "POST_EGO_OVERLAP_INCOMPLETE_PRE_SEAL_A",
        "claims": "Seal A prerequisite only; does not approve Seal A, data rights or GPU use. No clean-transfer "
                  "claim until the overlap policy and coverage (incl. any not-compared Ego items) are resolved.",
    }
    summary["summary_sha256"] = write_once_json(out_dir / "ego_overlap_summary.json", summary)
    return summary
