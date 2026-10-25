"""Baseline evaluation data artifacts (plan A4): image-shuffle maps and the NormBank retention subset.

Uses the existing locked algorithms unchanged:
  * `grouping.image_shuffle_map` per held-out split (sorted group IDs, fresh PCG64(20261022), each group
    receives the NEXT group's representative image = its lexicographically smallest RGB identity);
  * `normbank.retention_subset` (first <=2000 official-test canonical IDs by the locked hash sort).

Inputs are explicit manifest paths with expected SHA256 values (sidecars also checked); originals
are read only. Outputs are write-once in a new directory. Nothing is conditioned on labels, nothing
is retried, and no score is computed. These artifacts do not declare Seal A or final split
clearance; the source manifest status is recorded as given.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..constants import PHASH_MAX_HAMMING, SHUFFLE_SEED
from ..hashing import compact_json_bytes, sha256_bytes, sha256_file
from ..io_utils import read_json, read_jsonl, write_once_json
from .grouping import image_shuffle_map
from .normbank import retention_subset

ARTIFACT_PROTOCOL = "baseline_eval_artifacts/v1"
HELD_OUT = ("dev", "test")


class EvalManifestError(RuntimeError):
    pass


def _load_verified(path, expected_sha256: str) -> list[dict]:
    path = Path(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise EvalManifestError(f"{path}: SHA256 {actual} != expected {expected_sha256}")
    try:
        rows = read_jsonl(path, verify=True)
    except Exception as exc:
        raise EvalManifestError(f"{path}: sidecar verification failed: {exc}") from exc
    ids = [r.get("example_id") for r in rows]
    if not rows or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise EvalManifestError(f"{path}: manifest must be nonempty with unique string example IDs")
    return rows


def _source_status(path) -> str | None:
    """Status recorded by the step that produced the manifest (overlap summary or prep summary)."""
    d = Path(path).parent
    for name in ("ego_overlap_summary.json", "normlens_summary.json", "normbank_summary.json"):
        if (d / name).exists():
            return read_json(d / name).get("status")
    return None


def shuffle_maps(examples: list[dict]) -> dict:
    """Per held-out split: {example_id: {"image", "rgb_sha256"}} plus the group-level map."""
    split_of: dict[str, str] = {}
    members: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        if split_of.setdefault(e["group_id"], e["split"]) != e["split"]:
            raise EvalManifestError(f"group {e['group_id']} spans more than one split")
        members[e["group_id"]].append(e)
    out = {}
    for split in HELD_OUT:
        groups = sorted(g for g, s in split_of.items() if s == split)
        if len(groups) < 2:
            raise EvalManifestError(f"{split}: image shuffle requires at least two groups (found {len(groups)})")
        rep = {}
        for g in groups:
            files = sorted({f for e in members[g] if e["rgb_sha256"] == g for f in e.get("images") or [e["image"]]})
            if not files:
                raise EvalManifestError(f"{split}: group {g} has no image with its representative RGB identity")
            rep[g] = files[0]
        group_map = image_shuffle_map(groups, SHUFFLE_SEED)
        for g, donor in group_map.items():
            _check_donor(members[g], members[donor], g, donor)
        mapping = {e["example_id"]: {"image": rep[group_map[e["group_id"]]], "rgb_sha256": group_map[e["group_id"]]}
                   for g in groups for e in members[g]}
        out[split] = {"mapping": dict(sorted(mapping.items())), "group_map": dict(sorted(group_map.items())),
                      "groups": len(groups), "examples": len(mapping)}
    return out


def _check_donor(recipient: list[dict], donor: list[dict], g: str, d: str) -> None:
    if g == d:
        raise EvalManifestError(f"group {g} mapped to itself")
    if {e["rgb_sha256"] for e in recipient} & {e["rgb_sha256"] for e in donor}:
        raise EvalManifestError(f"donor group {d} shares an exact image with recipient {g}")
    rp = {int(e["phash"], 16) for e in recipient if e.get("phash")}
    dp = {int(e["phash"], 16) for e in donor if e.get("phash")}
    if any(bin(a ^ b).count("1") <= PHASH_MAX_HAMMING for a in rp for b in dp):
        raise EvalManifestError(f"donor group {d} is a pHash near-duplicate of recipient {g}")


def build_eval_manifests(normlens_path, normlens_sha256: str, normbank_path, normbank_sha256: str, out_dir) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise EvalManifestError(f"output directory {out_dir} is not empty (outputs are write-once)")
    nl = _load_verified(normlens_path, normlens_sha256)
    nb = _load_verified(normbank_path, normbank_sha256)
    if any(e.get("split") not in ("train", "dev", "test") for e in nl + nb):
        raise EvalManifestError("every manifest row needs split train/dev/test")
    maps = shuffle_maps(nl)
    retention = retention_subset(nb)
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for split in HELD_OUT:
        digests[f"shuffle_map_{split}"] = write_once_json(out_dir / f"shuffle_map_{split}.json", maps[split]["mapping"])
        digests[f"shuffle_group_map_{split}"] = write_once_json(out_dir / f"shuffle_group_map_{split}.json",
                                                                maps[split]["group_map"])
    digests["normbank_retention_ids"] = write_once_json(out_dir / "normbank_retention_ids.json", {
        "ids": retention, "count": len(retention), "ids_sha256": sha256_bytes(compact_json_bytes(retention))})
    summary = {
        "protocol": ARTIFACT_PROTOCOL,
        "status": "BASELINE_EVAL_ARTIFACTS_PRE_SEAL_A",
        "claims": "data artifacts only; no scores; does not declare Seal A or final split clearance",
        "inputs": {
            "normlens_manifest": {"path": str(normlens_path), "sha256": normlens_sha256,
                                  "source_status": _source_status(normlens_path)},
            "normbank_manifest": {"path": str(normbank_path), "sha256": normbank_sha256,
                                  "source_status": _source_status(normbank_path)}},
        "shuffle_rule": f"per held-out split: sorted group IDs, numpy PCG64({SHUFFLE_SEED}) permutation, each group "
                        "receives the next group's representative image (smallest RGB identity); no retries",
        "retention_rule": "first <=2000 official-test canonical IDs by sha256('retention20261021|'+id)",
        "counts": {**{split: {"examples": maps[split]["examples"], "groups": maps[split]["groups"]}
                      for split in HELD_OUT},
                   "normbank_official_test": sum(1 for e in nb if e["split"] == "test"),
                   "normbank_retention": len(retention)},
        "outputs": digests,
    }
    summary["summary_sha256"] = write_once_json(out_dir / "eval_manifests_summary.json", summary)
    return summary
