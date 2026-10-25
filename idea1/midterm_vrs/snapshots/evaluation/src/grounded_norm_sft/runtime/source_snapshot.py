"""Verify an explicitly staged immutable source release, without touching data trees."""
from pathlib import Path
import re

from ..constants import REMOTE_ROOT
from ..hashing import sha256_file
from .errors import GateError


def verify_source_snapshot(code_root: str, expected_sha256: str | None = None) -> str:
    def refuse(message):
        raise GateError([f"GATE source snapshot: {message}"])

    root = Path(code_root)
    releases = Path(REMOTE_ROOT) / "releases"
    if not root.is_absolute() or root.parent != releases or re.fullmatch(r"[0-9a-f]{64}", root.name) is None:
        refuse("code-root must be approved remote_root/releases/<64hex archive SHA256>")
    if root.resolve() != root or any(p.is_symlink() for p in (root, *root.parents)):
        refuse("code-root may not traverse symlinks")
    marker = root / "SOURCE_SNAPSHOT_SHA256"
    manifest = root / "SHA256SUMS"
    if not marker.is_file() or not manifest.is_file() or marker.is_symlink() or manifest.is_symlink():
        refuse("release marker and SHA256SUMS are required")
    digest = marker.read_text().strip()
    if digest != root.name or (expected_sha256 is not None and digest != expected_sha256):
        refuse("release identity differs from launch or directory")
    covered = set()
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            refuse("malformed SHA256SUMS")
        checksum, relative = match.groups()
        rel = Path(relative)
        target = root / rel
        if rel.is_absolute() or ".." in rel.parts or relative in covered or target.resolve() != target:
            refuse("unsafe, duplicate, or symlinked manifest entry")
        if not target.is_file() or sha256_file(target) != checksum:
            refuse(f"hash mismatch: {relative}")
        covered.add(relative)
    required = set()
    for pattern in ("src/**/*.py", "scripts/*.py", "configs/*.json", "provenance/native_source/**/*"):
        required.update(str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    if not required or not required <= covered:
        refuse("manifest must cover source, scripts, configs and native provenance")
    return digest
