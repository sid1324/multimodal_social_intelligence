"""EgoNormia transfer data: schema validation, locked subset, pre-action frame sheets (A5).

Only `video/{id}/video_prev.mp4` (pre-action) media and native candidate text enter inference.
Gold indices, taxonomy and descriptions are scorer-only. The hosting split name "train" is
packaging, not a training assignment for this experiment.

Production source (user correction 2026-10-22): the verified 1,853-record `final_data.json`
(constants.EGO_SOURCE_*), media revision constants.EGO_MEDIA_REVISION. The plan v1.2 A1 tuple
(1,743-record HF parquet, revision bc43...) is historical only. Frozen evaluation IDs are owned
by the media session (provenance/egonormia_eval_ids_v1.json); this module only provides the rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..constants import (EGO_DIAG_COHORT, EGO_FRAMES, EGO_MAX_ITEMS, EGO_NUM_OPTIONS, EGO_SHEET_H,
                         EGO_SHEET_W, EGO_SORT_PREFIX, EGO_TILE)
from ..hashing import rgb_identity_hash, sha256_file, sort_key_sha256


class EgoSchemaError(RuntimeError):
    pass


@dataclass(frozen=True)
class EgoItem:
    id: str
    behaviors: tuple[str, ...]
    justifications: tuple[str, ...]
    correct_idx: int          # zero-based; scorer-only
    sensible_idx: frozenset   # zero-based; scorer-only
    source_group: str | None  # verified source-video group, else None

    @property
    def none_index(self) -> int | None:
        empties = [i for i, b in enumerate(self.behaviors) if b == ""]
        return empties[0] if len(empties) == 1 else None


@dataclass(frozen=True)
class EgoSourceSchema:
    """An explicit, versioned raw-source schema. The index base is a declared property of the
    source version, never inferred per record; schemas are never mixed in one load."""
    name: str
    correct_field: str
    sensible_field: str
    index_base: int | None   # None = unverified: the caller must declare it explicitly
    production: bool


SOURCE_SCHEMAS = {
    # Production (user-verified 2026-10-22): JSON object keyed by id, record["id"] == key,
    # `correct` int and `sensibles` list, both zero-based.
    "egonormia_final_data_1853_v1": EgoSourceSchema("egonormia_final_data_1853_v1", "correct", "sensibles", 0, True),
    # Historical only (plan v1.2 A1 HF parquet, 1,743 records); index base never verified.
    "egonormia_hf_bc43_1743_historical": EgoSourceSchema("egonormia_hf_bc43_1743_historical", "correct_idx",
                                                         "sensible_idx", None, False),
}


def _index_list(value, name: str) -> list[int]:
    if type(value) is int:
        value = [value]
    if not isinstance(value, list) or any(type(v) is not int for v in value):
        raise EgoSchemaError(f"{name} must be int or list of int")
    return value


def validate_item(rec: dict, schema: str, index_base: int | None = None,
                  source_group_field: str | None = None) -> EgoItem:
    """Transform one raw record of the named source schema into an EgoItem (zero-based gold).

    Production schema has a verified base; a historical schema requires an explicit index_base."""
    if schema not in SOURCE_SCHEMAS:
        raise EgoSchemaError(f"unknown Ego source schema {schema!r}")
    s = SOURCE_SCHEMAS[schema]
    if s.index_base is not None:
        if index_base not in (None, s.index_base):
            raise EgoSchemaError(f"{schema} has verified index base {s.index_base}; got {index_base}")
        index_base = s.index_base
    elif index_base not in (0, 1):
        raise EgoSchemaError(f"{schema} has no verified index base; declare index_base explicitly")
    required = ("id", "behaviors", "justifications", s.correct_field, s.sensible_field)
    missing = [f for f in required if f not in rec]
    if missing:
        raise EgoSchemaError(f"missing field(s) {missing}; record keys: {sorted(rec)}")
    if not isinstance(rec["id"], str) or not rec["id"]:
        raise EgoSchemaError("id must be a nonempty string")
    beh, jus = rec["behaviors"], rec["justifications"]
    for name, lst in (("behaviors", beh), ("justifications", jus)):
        if not isinstance(lst, list) or len(lst) != EGO_NUM_OPTIONS or not all(isinstance(x, str) for x in lst):
            raise EgoSchemaError(f"{name} must be a list of exactly five strings")
    if sum(1 for b in beh if b == "") > 1:
        raise EgoSchemaError("more than one empty (None) behavior candidate")
    correct = _index_list(rec[s.correct_field], s.correct_field)
    if len(correct) != 1:
        raise EgoSchemaError(f"{s.correct_field} must identify exactly one option")
    sensible = _index_list(rec[s.sensible_field], s.sensible_field)
    zero = [c - index_base for c in correct] + [x - index_base for x in sensible]
    if any(not 0 <= z < EGO_NUM_OPTIONS for z in zero):
        raise EgoSchemaError("index out of range under the schema's index base")
    group = rec.get(source_group_field) if source_group_field else None
    return EgoItem(
        id=rec["id"], behaviors=tuple(beh), justifications=tuple(jus),
        correct_idx=correct[0] - index_base,
        sensible_idx=frozenset(x - index_base for x in sensible),
        source_group=str(group) if group is not None else None,
    )


def load_final_data_1853(path: str | Path, verify_pin: bool = True) -> dict:
    """Load the production `final_data.json` (object keyed by id) into EgoItems.

    Every key must equal its record's `id`. Schema-invalid records are returned with reasons (they
    are excluded from the eligible set, never repaired). With verify_pin, the file SHA256 and the
    record count must equal the pinned production tuple."""
    import json

    from ..constants import EGO_SOURCE_RECORDS, EGO_SOURCE_SCHEMA, EGO_SOURCE_SHA256

    digest = sha256_file(path)
    if verify_pin and digest != EGO_SOURCE_SHA256:
        raise EgoSchemaError(f"Ego source SHA256 {digest} != pinned {EGO_SOURCE_SHA256}")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise EgoSchemaError("production Ego source must be a JSON object keyed by id")
    if verify_pin and len(raw) != EGO_SOURCE_RECORDS:
        raise EgoSchemaError(f"{len(raw)} records != pinned {EGO_SOURCE_RECORDS}")
    items, invalid = {}, []
    for key, rec in raw.items():
        if not isinstance(rec, dict) or rec.get("id") != key:
            raise EgoSchemaError(f"record keyed {key!r} does not carry id == key")
        try:
            items[key] = validate_item(rec, EGO_SOURCE_SCHEMA)
        except EgoSchemaError as exc:
            invalid.append({"id": key, "reason": str(exc)})
    return {"schema": EGO_SOURCE_SCHEMA, "sha256": digest, "records": len(raw), "items": items,
            "schema_invalid": invalid}


def locked_subset(eligible_ids: list[str], limit: int = EGO_MAX_ITEMS) -> list[str]:
    """First <=400 schema-valid eligible IDs sorted by SHA256('ego_midterm20261021|'+id), ties by ID.
    Eligibility is schema validity only (not media availability)."""
    if len(set(eligible_ids)) != len(eligible_ids):
        raise EgoSchemaError("duplicate Ego IDs")
    return sorted(eligible_ids, key=lambda i: (sort_key_sha256(EGO_SORT_PREFIX, i), i))[:limit]


def profile_ids(locked_ids: list[str]) -> list[str]:
    """First 5 locked IDs: runtime/length profile only, never accuracy-scored for selection."""
    from ..constants import EGO_PROFILE_IDS

    return list(locked_ids[:EGO_PROFILE_IDS])


def diagnostic_cohort(locked_ids: list[str]) -> list[str]:
    """First 32 IDs of the locked order (or all if fewer)."""
    return list(locked_ids[:EGO_DIAG_COHORT])


def frame_indices(total_frames: int, k_frames: int = EGO_FRAMES) -> list[int]:
    """floor(k*(T-1)/7 + 0.5), exact integer arithmetic; repeats frames when T < 8."""
    if total_frames <= 0:
        raise ValueError("no decodable frames")
    denom = k_frames - 1
    return [(2 * k * (total_frames - 1) + denom) // (2 * denom) for k in range(k_frames)]


def fit_size(w: int, h: int, box: int = EGO_TILE) -> tuple[int, int]:
    """Aspect-preserving fit within box x box (upscale or downscale), round-half-up, >=1px."""
    if w <= 0 or h <= 0:
        raise ValueError("invalid frame size")
    if w >= h:
        return box, max(1, (2 * h * box + w) // (2 * w))
    return max(1, (2 * w * box + h) // (2 * h)), box


def tile_offset(w: int, h: int, box: int = EGO_TILE) -> tuple[int, int]:
    """Center on black; any odd extra pixel goes right/bottom."""
    return (box - w) // 2, (box - h) // 2


def build_sheet(frames: list):
    """Arrange 8 RGB Pillow frames row-major (4x2) into a 1024x512 unlabeled sheet."""
    from PIL import Image

    if len(frames) != EGO_FRAMES:
        raise ValueError("exactly eight frames required")
    sheet = Image.new("RGB", (EGO_SHEET_W, EGO_SHEET_H), (0, 0, 0))
    cols = EGO_SHEET_W // EGO_TILE
    for k, frame in enumerate(frames):
        frame = frame.convert("RGB")
        w, h = fit_size(frame.width, frame.height)
        tile = frame.resize((w, h), Image.LANCZOS)
        ox, oy = tile_offset(w, h)
        col, row = k % cols, k // cols
        sheet.paste(tile, (col * EGO_TILE + ox, row * EGO_TILE + oy))
    return sheet


def decoder_inventory() -> dict:
    try:
        import av
    except ImportError:
        return {"av": None}
    return {"av": av.__version__, "ffmpeg": getattr(av, "library_versions", None)}


def decode_all_frames(video_path: str | Path) -> list:
    """Decode every frame of the pinned pre-action MP4 with PyAV (pinned in the manifest)."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV (av) is required for Ego frame decoding and must be pinned") from exc
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_image().convert("RGB"))
    return frames


def make_sheet_for_item(video_root: str | Path, ego_id: str, out_dir: str | Path) -> dict:
    """Return a cache record; decode failures are reported (common exclusion), never replaced.
    Frame pHashes (same pinned ImageHash as NormLens) feed the NormLens/Ego overlap audit.
    NOT YET EXERCISED: pre-action MP4s are absent locally; untested pending media access."""
    from .images import phash_int

    path = Path(video_root) / "video" / ego_id / "video_prev.mp4"
    record = {"id": ego_id, "source_path": str(path)}
    if not path.exists():
        return {**record, "status": "decode_failed", "reason": "missing_pre_action_video"}
    try:
        frames = decode_all_frames(path)
    except Exception as exc:
        return {**record, "status": "decode_failed", "reason": f"{type(exc).__name__}: {exc}"}
    if not frames:
        return {**record, "status": "decode_failed", "reason": "zero_frames"}
    idx = frame_indices(len(frames))
    sheet = build_sheet([frames[i] for i in idx])
    out = Path(out_dir) / f"{ego_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, format="PNG")
    return {**record, "status": "ok", "source_sha256": sha256_file(path), "total_frames": len(frames),
            "frame_indices": idx,
            "frame_rgb_sha256": [rgb_identity_hash(frames[i].width, frames[i].height, frames[i].tobytes())
                                 for i in idx],
            "frame_phash": [f"{phash_int(frames[i]):016x}" for i in idx],
            "sheet_path": str(out), "sheet_sha256": sha256_file(out)}
