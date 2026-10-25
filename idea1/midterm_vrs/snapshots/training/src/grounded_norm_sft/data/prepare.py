"""Real-data preparation entry points (write-once outputs into a private directory).

prepare_normlens: decode/identify images (pinned Pillow + ImageHash), canonicalize, pHash
connected components, locked grouped split, optional blocking legacy exact-RGB comparison.
compare_legacy: exact-RGB + example-identity reaudit only (no pHash, ImageHash not needed).
prepare_normbank: canonical inputs, split priority, empirical label distributions.

Ego overlap exclusion (A2.7) cannot run: pre-action MP4s are absent. Outputs are therefore
labeled PROVISIONAL_PRE_SEAL_A / pre-Ego-exclusion and are not final for clean-transfer claims.
"""

from __future__ import annotations

from pathlib import Path

from ..hashing import compact_json_bytes, sha256_bytes
from ..io_utils import write_once_json, write_once_jsonl
from . import legacy, normbank, normlens
from .contact_sheets import write_phash_contact_sheets
from .grouping import component_report, connected_components, phash_edges
from .images import decoder_inventory, identify_image, phash_inventory, resolve_image_path


class PreparationBlocked(RuntimeError):
    pass


def _check_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise PreparationBlocked(f"output directory {out_dir} is not empty (outputs are write-once)")
    out_dir.mkdir(parents=True, exist_ok=True)


def _identify_all(dataset_dir: Path, records, with_phash: bool):
    image_dir = dataset_dir / "image"
    cache: dict[str, dict] = {}

    def ident(name: str) -> dict:
        if name not in cache:
            cache[name] = identify_image(resolve_image_path(image_dir, name), with_phash=with_phash).as_dict()
        return cache[name]

    result = normlens.canonicalize(records, normlens.NormLensFieldMap(), ident)
    on_disk = sorted(p.name for p in image_dir.iterdir() if p.is_file())
    listing = {"files_on_disk": len(on_disk),
               "unreferenced_files": sorted(set(on_disk) - set(cache)),
               "file_listing_sha256": sha256_bytes(compact_json_bytes(
                   [[n, cache[n]["file_sha256"]] for n in sorted(cache)]))}
    return result, cache, listing


def _legacy_reports(result, cache, legacy_path, expected_images):
    rows = legacy.load_legacy(legacy_path)
    rgb = legacy.compare_rgb_identities({n: v["rgb_sha256"] for n, v in cache.items()}, rows, expected_images)
    ex = legacy.compare_examples(result.examples, rows)
    return {"schema": legacy.LEGACY_SCHEMA, "legacy_path": str(legacy_path), "rgb": rgb, "examples": ex,
            "blocked": rgb["status"] != "IDENTICAL" or ex["status"] != "IDENTICAL_IDENTITIES"}


def compare_legacy(dataset_dir, legacy_path, out_dir, expected_images: int | None = 1742) -> dict:
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    _check_out_dir(out_dir)
    records = normlens.load_release(dataset_dir)
    result, cache, listing = _identify_all(dataset_dir, records, with_phash=False)
    report = _legacy_reports(result, cache, legacy_path, expected_images)
    report.update(decoder=decoder_inventory(), image_listing=listing, canonical_summary=result.summary,
                  note="exact-RGB reaudit only; pHash grouping not computed")
    write_once_json(out_dir / "legacy_comparison.json", report)
    return report


def prepare_normlens(dataset_dir, out_dir, legacy_path=None, expected_images: int | None = 1742) -> dict:
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    _check_out_dir(out_dir)
    inventory = {"decoder": decoder_inventory(), "phash": phash_inventory()}
    records = normlens.load_release(dataset_dir)
    result, cache, listing = _identify_all(dataset_dir, records, with_phash=True)
    phashes: dict[str, int] = {}
    for v in cache.values():
        prev = phashes.setdefault(v["rgb_sha256"], v["phash"])
        if prev != v["phash"]:
            raise PreparationBlocked("identical RGB identity produced different pHashes")
    edges = phash_edges(phashes)
    group_of = connected_components(phashes, edges)
    normlens.attach_groups_and_splits(result.examples, group_of)

    if legacy_path is not None:
        report = _legacy_reports(result, cache, legacy_path, expected_images)
        write_once_json(out_dir / "legacy_comparison.json", report)
        if report["blocked"]:
            raise PreparationBlocked("legacy exact-RGB/identity comparison found differences; manifest NOT "
                                     f"written (see {out_dir / 'legacy_comparison.json'})")
    from ..metrics.normlens import context_strata

    summary = {
        "status": "PROVISIONAL_PRE_SEAL_A_PRE_EGO_OVERLAP",
        "ego_overlap_exclusion": "NOT_RUN: EgoNormia pre-action MP4s unavailable; not final for clean-transfer claims",
        "inventory": inventory, "image_listing": listing,
        "input_files": sorted({(r.file, r.file_sha256) for r in records}),
        "canonical": result.summary, "components": component_report(group_of, edges),
        "splits": normlens.split_summary(result.examples),
        "context_strata_descriptive": context_strata(result.examples),
        "legacy_comparison": "legacy_comparison.json" if legacy_path is not None else "NOT_RUN",
    }
    manifest_sha = write_once_jsonl(out_dir / "normlens_examples.jsonl", result.examples)
    write_once_jsonl(out_dir / "normlens_quarantine.jsonl", result.quarantine)
    write_once_jsonl(out_dir / "normlens_phash_edges.jsonl",
                     [{"a": a, "b": b, "hamming": d} for a, b, d in edges])
    sheets = write_phash_contact_sheets(result.examples, edges, dataset_dir / "image", out_dir / "contact_sheets")
    summary["contact_sheets"] = {"dir": "contact_sheets", "groups": sheets["num_multi_image_groups"],
                                 "metadata_sha256": sheets["metadata_sha256"]}
    summary["manifest_sha256"] = manifest_sha
    write_once_json(out_dir / "normlens_summary.json", summary)
    return summary


def contact_sheets_from_prepared(dataset_dir, prepared_dir, out_dir) -> dict:
    """Contact sheets for an existing (immutable) prepared directory, written to a NEW directory."""
    from ..io_utils import read_jsonl

    out_dir, prepared_dir = Path(out_dir), Path(prepared_dir)
    _check_out_dir(out_dir)
    examples = read_jsonl(prepared_dir / "normlens_examples.jsonl", verify=True)
    edges = read_jsonl(prepared_dir / "normlens_phash_edges.jsonl", verify=True)
    meta = write_phash_contact_sheets(examples, edges, Path(dataset_dir) / "image", out_dir)
    return {**meta, "source_prepared_dir": str(prepared_dir)}


def prepare_normbank(csv_path, out_dir) -> dict:
    out_dir = Path(out_dir)
    _check_out_dir(out_dir)
    rows, header, digest = normbank.load_rows(csv_path)
    res = normbank.canonicalize(rows, header, normbank.NormBankFieldMap())
    manifest_sha = write_once_jsonl(out_dir / "normbank_examples.jsonl", res["examples"])
    write_once_jsonl(out_dir / "normbank_quarantine.jsonl", res["quarantine"])
    summary = {"status": "PROVISIONAL_PRE_SEAL_A", "csv_sha256": digest, "header": header,
               **res["summary"], "setting_behavior_violations": res["setting_behavior_violations"],
               "retention_subset_ids": normbank.retention_subset(res["examples"]),
               "manifest_sha256": manifest_sha}
    write_once_json(out_dir / "normbank_summary.json", summary)
    return summary
