"""Image identity (exact RGB hash), file hash and production 64-bit pHash (pinned ImageHash).

Decoding policy (A2.1): Pillow decode -> convert("RGB"), no resize and no EXIF transpose.
Identity = SHA256(struct.pack(">II", width, height) + RGB bytes), the same rule as the
provisional exact-RGB audit (which used Pillow 10.2.0). A different Pillow/libjpeg build must be
re-audited against the provisional hashes (`gnsft compare-legacy`); any difference blocks
preparation and requires a new manifest/split version. Nothing is auto-blessed.

pHash: ImageHash `phash(img, hash_size=8, highfreq_factor=4)` is the only production backend;
its version is recorded with every manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..hashing import rgb_identity_hash, sha256_file


class DecoderUnavailable(RuntimeError):
    pass


def decoder_inventory() -> dict:
    try:
        import PIL
        from PIL import features
    except ImportError as exc:
        raise DecoderUnavailable("Pillow is required (see requirements-cpu.txt)") from exc
    inv = {"pillow": PIL.__version__}
    for name, query in (("libjpeg_turbo", lambda: features.version("libjpeg_turbo")),
                        ("jpg_codec", lambda: features.version_codec("jpg"))):
        try:
            inv[name] = query()
        except Exception:  # feature absent in this build; recorded as None, not guessed
            inv[name] = None
    return inv


def phash_inventory() -> dict:
    try:
        import imagehash
    except ImportError as exc:
        raise DecoderUnavailable("ImageHash is required for production pHash (see requirements-cpu.txt)") from exc
    return {"imagehash": imagehash.__version__, "phash_call": "phash(img, hash_size=8, highfreq_factor=4)"}


def decode_rgb(path: str | Path):
    """Pillow RGB image, no resize, no EXIF transpose."""
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


def rgb_hash_of_image(img) -> str:
    if img.mode != "RGB":
        raise ValueError("expected RGB image")
    return rgb_identity_hash(img.width, img.height, img.tobytes())


def phash_int(img) -> int:
    import imagehash

    value = 0
    for bit in imagehash.phash(img, hash_size=8, highfreq_factor=4).hash.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming64(a: int, b: int) -> int:
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")


@dataclass(frozen=True)
class ImageIdentity:
    path: str
    file_sha256: str
    rgb_sha256: str
    width: int
    height: int
    phash: int | None

    def as_dict(self) -> dict:
        return asdict(self)


def identify_image(path: str | Path, with_phash: bool = True) -> ImageIdentity:
    img = decode_rgb(path)
    return ImageIdentity(path=str(path), file_sha256=sha256_file(path), rgb_sha256=rgb_hash_of_image(img),
                         width=img.width, height=img.height, phash=phash_int(img) if with_phash else None)


def resolve_image_path(image_dir: str | Path, name: str) -> Path:
    """Join a release image field to the image directory; reject path traversal."""
    root = Path(image_dir).resolve()
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError(f"image field escapes image directory: {name!r}")
    return path
