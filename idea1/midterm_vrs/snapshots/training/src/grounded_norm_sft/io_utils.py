"""Write-once artifact IO with SHA256 sidecars (immutable predictions/manifests)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from .hashing import sha256_file


class ImmutableArtifactError(RuntimeError):
    pass


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_" + path.name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_once_bytes(path: str | Path, data: bytes) -> str:
    """Write a new immutable artifact; refuse to overwrite. Returns SHA256 hex."""
    path = Path(path)
    if path.exists() or _sidecar(path).exists():
        raise ImmutableArtifactError(f"refusing to overwrite immutable artifact {path}")
    _atomic_write_bytes(path, data)
    digest = sha256_file(path)
    _atomic_write_bytes(_sidecar(path), (digest + "  " + path.name + "\n").encode())
    return digest


def write_once_json(path: str | Path, obj: Any) -> str:
    data = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return write_once_bytes(path, data)


def write_once_jsonl(path: str | Path, rows: Iterable[dict]) -> str:
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
    return write_once_bytes(path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    """Mutable (non-sealed) JSON, e.g. logs and resumable state descriptors."""
    data = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write_bytes(Path(path), data)


def verify_sidecar(path: str | Path) -> str:
    path = Path(path)
    side = _sidecar(path)
    if not side.exists():
        raise ImmutableArtifactError(f"missing SHA256 sidecar for {path}")
    expected = side.read_text().split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise ImmutableArtifactError(f"hash mismatch for {path}: sidecar {expected} actual {actual}")
    return actual


def read_json(path: str | Path, verify: bool = False) -> Any:
    if verify:
        verify_sidecar(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str | Path, verify: bool = False) -> list[dict]:
    if verify:
        verify_sidecar(path)
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSONL") from exc
