"""Deterministic hashing and seed derivation used throughout the plan."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compact_json_bytes(obj: Any) -> bytes:
    """Compact UTF-8 JSON exactly as the plan specifies: separators=(",",":"), ensure_ascii=False.

    Key order is preserved (the plan's seed payloads are JSON arrays)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON for identity hashing of mappings (sorted keys, compact, UTF-8)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode("utf-8")


def first64_unsigned(digest: bytes) -> int:
    return int.from_bytes(digest[:8], "big", signed=False)


def seed64_from_json(obj: Any) -> int:
    """First 64 bits (unsigned, big-endian) of SHA256(compact UTF-8 JSON)."""
    return first64_unsigned(hashlib.sha256(compact_json_bytes(obj)).digest())


def unit_interval_from_text(text: str) -> float:
    """u = first64(SHA256(text)) / 2**64 (Python float division, as in the A2 split rule)."""
    return first64_unsigned(hashlib.sha256(text.encode("utf-8")).digest()) / 2**64


def sort_key_sha256(prefix: str, identifier: str) -> str:
    return sha256_text(prefix + identifier)


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def rgb_identity_hash(width: int, height: int, rgb_bytes: bytes) -> str:
    """Image identity: SHA256(struct.pack('>II', width, height) + RGB bytes)."""
    if len(rgb_bytes) != width * height * 3:
        raise ValueError("RGB byte length does not match width*height*3")
    return sha256_bytes(struct.pack(">II", width, height) + rgb_bytes)


def stable_id(tag: str, *parts: Any) -> str:
    """Full 64-hex SHA256 of compact JSON [tag, *parts] (no truncation)."""
    return sha256_bytes(compact_json_bytes([tag, *parts]))
