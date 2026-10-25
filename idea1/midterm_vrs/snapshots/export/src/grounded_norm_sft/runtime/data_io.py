"""Manifest access for the runtime (schemas from data/normlens.py and data/normbank.py, read-only).

Inputs and labels are separated here: model-facing records are whitelisted input fields only
(no votes, labels, explanations, captions, image_src, agreement, group or split), so gold can
never reach a prompt. Labels are exposed separately to the training loss and to the scorer.
Images are resolved inside the declared image directory and their decoded RGB identity must
equal the manifest hash; a missing image raises MissingImageError and is never substituted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..io_utils import read_json, read_jsonl
from .errors import ImageIdentityError, MissingImageError, RuntimeContractError

NORMLENS_INPUT_FIELDS = ("example_id", "image", "action", "rgb_sha256")
NORMBANK_INPUT_FIELDS = ("example_id", "setting", "behavior", "constraints")
NORMLENS_REQUIRED = NORMLENS_INPUT_FIELDS + ("split", "vote_probs", "unique_mode", "group_id")
NORMBANK_REQUIRED = NORMBANK_INPUT_FIELDS + ("split", "label_probs")


def _read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path, verify=True)
    obj = read_json(path, verify=True)
    rows = obj.get("examples") if isinstance(obj, dict) else obj
    if not isinstance(rows, list):
        raise RuntimeContractError(f"{path}: expected a list of examples or {{'examples': [...]}}")
    return rows


def load_manifest(path: str | Path, expected_sha256: str, required: tuple[str, ...]) -> list[dict]:
    """Rows of a sealed manifest; the SHA256 sidecar and the launch hash must both match."""
    from ..hashing import sha256_file

    if sha256_file(path) != expected_sha256:
        raise RuntimeContractError(f"{path}: sha256 differs from the launch/Seal A value")
    rows = _read_rows(path)
    ids = [r.get("example_id") for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeContractError(f"{path}: duplicate example_id")
    for r in rows:
        missing = [k for k in required if k not in r]
        if missing:
            raise RuntimeContractError(f"{path}: example {r.get('example_id')} lacks {missing}")
    return rows


def load_normlens(path, sha) -> list[dict]:
    return load_manifest(path, sha, NORMLENS_REQUIRED)


def load_normbank(path, sha) -> list[dict]:
    return load_manifest(path, sha, NORMBANK_REQUIRED)


def split_rows(rows: list[dict], split: str) -> list[dict]:
    return sorted((r for r in rows if r["split"] == split), key=lambda r: r["example_id"])


def normlens_input(ex: Mapping) -> dict:
    return {k: ex[k] for k in NORMLENS_INPUT_FIELDS}


def normbank_input(ex: Mapping) -> dict:
    return {k: ex[k] for k in NORMBANK_INPUT_FIELDS}


def normlens_target(ex: Mapping) -> list[float]:
    return [float(x) for x in ex["vote_probs"]]


def normbank_target(ex: Mapping) -> list[float]:
    return [float(x) for x in ex["label_probs"]]


def scorer_labels(rows: list[dict]) -> dict:
    """Label view for metrics.normlens (joined only at scoring)."""
    keep = ("vote_probs", "unique_mode", "group_id", "agreement_stratum", "normalized_entropy")
    return {r["example_id"]: {k: r[k] for k in keep if k in r} for r in rows}


class ImageLoader:
    """Resolve, decode (data.images policy) and identity-check NormLens images."""

    def __init__(self, image_dir: str | Path):
        self.image_dir = Path(image_dir)

    def load(self, record: Mapping):
        from ..data.images import decode_rgb, resolve_image_path, rgb_hash_of_image

        path = resolve_image_path(self.image_dir, record["image"])
        if not path.is_file():
            raise MissingImageError(f"{record['example_id']}: image {record['image']!r} not found under {self.image_dir}")
        img = decode_rgb(path)
        want = record.get("rgb_sha256")
        if want is not None and rgb_hash_of_image(img) != want:
            raise ImageIdentityError(f"{record['example_id']}: decoded RGB hash differs from manifest")
        return img
