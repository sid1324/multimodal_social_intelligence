"""Pickle-free tensor files via the standard `safetensors` library (lazy import; pinned runtime
dependency). Writes are write-once with a SHA256 sidecar (io_utils); integrity is verified by the
callers (checkpoint MANIFEST / export report) before any load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..io_utils import write_once_bytes


def save_tensors(path: str | Path, tensors: Mapping[str, "object"], metadata: Mapping[str, str] | None = None) -> str:
    """Write-once safetensors file; returns its SHA256."""
    from safetensors.torch import save

    data = save({k: v.detach().cpu().contiguous() for k, v in tensors.items()},
                metadata=dict(metadata) if metadata else None)
    return write_once_bytes(path, data)


def load_tensors(path: str | Path) -> tuple[dict, dict]:
    """Return ({name: CPU tensor}, metadata)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as fh:
        meta = fh.metadata() or {}
    return load_file(str(path), device="cpu"), dict(meta)
