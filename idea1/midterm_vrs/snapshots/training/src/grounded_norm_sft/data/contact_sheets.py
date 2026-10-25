"""pHash near-duplicate audit contact sheets (plan A2.5). No labels are read or written.

One PNG per multi-image component (tiles in sorted RGB-identity order, aspect-preserving
thumbnails on black) plus write-once metadata mapping tile index -> RGB identity -> file names,
and the component's pHash edges with Hamming distances.
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

from ..io_utils import write_once_bytes, write_once_json
from .images import resolve_image_path


def _edge(e) -> tuple[str, str, int]:
    if isinstance(e, dict):
        return e["a"], e["b"], int(e["hamming"])
    a, b, d = e
    return a, b, int(d)


def write_phash_contact_sheets(examples: list[dict], edges, image_dir, out_dir, thumb: int = 192) -> dict:
    from PIL import Image

    out_dir = Path(out_dir)
    files: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    group_of: dict[str, str] = {}
    for e in examples:
        group_of[e["rgb_sha256"]] = e["group_id"]
        for f in e["images"]:
            files[e["group_id"]][e["rgb_sha256"]].add(f)
    edges_by_group: dict[str, list] = defaultdict(list)
    for a, b, d in map(_edge, edges):
        if group_of.get(a) is None or group_of[a] != group_of.get(b):
            raise ValueError("pHash edge endpoints are not in one manifest component")
        edges_by_group[group_of[a]].append({"a": a, "b": b, "hamming": d})

    records = []
    for gid in sorted(g for g, members in files.items() if len(members) > 1):
        rgbs = sorted(files[gid])
        sheet = Image.new("RGB", (thumb * len(rgbs), thumb), (0, 0, 0))
        tiles = []
        for k, rgb in enumerate(rgbs):
            names = sorted(files[gid][rgb])
            with Image.open(resolve_image_path(image_dir, names[0])) as im:
                tile = im.convert("RGB")
            tile.thumbnail((thumb, thumb))
            sheet.paste(tile, (k * thumb + (thumb - tile.width) // 2, (thumb - tile.height) // 2))
            tiles.append({"index": k, "rgb_sha256": rgb, "files": names})
        buf = io.BytesIO()
        sheet.save(buf, format="PNG")
        name = f"group_{gid}.png"
        digest = write_once_bytes(out_dir / name, buf.getvalue())
        records.append({"group_id": gid, "sheet": name, "sheet_sha256": digest, "tiles": tiles,
                        "edges": sorted(edges_by_group[gid], key=lambda x: (x["a"], x["b"]))})
    meta = {"thumb_px": thumb, "num_multi_image_groups": len(records), "groups": records,
            "note": "audit aid only; contains no labels; edges are never removed using labels"}
    meta["metadata_sha256"] = write_once_json(out_dir / "contact_sheets.json", meta)
    return meta
