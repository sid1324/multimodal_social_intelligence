"""Exact-RGB + pHash connected components, locked grouped split, shuffle map, Ego overlap."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping

import numpy as np

from ..constants import PHASH_MAX_HAMMING, SHUFFLE_SEED, SPLIT_DEV_UPPER, SPLIT_SALT, SPLIT_TRAIN_UPPER
from ..hashing import unit_interval_from_text
from .images import hamming64


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller id becomes root
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def phash_edges(phashes: Mapping[str, int], max_hamming: int = PHASH_MAX_HAMMING) -> list[tuple[str, str, int]]:
    """All pairs of distinct RGB identities with pHash Hamming distance <= max_hamming.

    Exact-image identity is already merged by keying on RGB hash. O(n^2) is fine for ~2k images.
    Edges are never removed using labels."""
    keys = sorted(phashes)
    edges = []
    for i, a in enumerate(keys):
        ha = phashes[a]
        for b in keys[i + 1:]:
            d = hamming64(ha, phashes[b])
            if d <= max_hamming:
                edges.append((a, b, d))
    return edges


def connected_components(rgb_ids: Iterable[str], edges: Iterable[tuple[str, str, int]]) -> dict[str, str]:
    """Map each RGB identity to its group ID = lexicographically smallest RGB hash in component."""
    ids = sorted(set(rgb_ids))
    uf = UnionFind(ids)
    for a, b, _ in edges:
        uf.union(a, b)
    members: dict[str, list[str]] = defaultdict(list)
    for x in ids:
        members[uf.find(x)].append(x)
    group_of = {}
    for comp in members.values():
        gid = min(comp)
        for x in comp:
            group_of[x] = gid
    return group_of


def component_report(group_of: Mapping[str, str], edges: list[tuple[str, str, int]]) -> dict:
    sizes = Counter(group_of.values())
    size_dist = Counter(sizes.values())
    largest = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))[0] if sizes else None
    return {
        "num_images": len(group_of),
        "num_groups": len(sizes),
        "num_phash_edges": len(edges),
        "edge_hamming_distribution": dict(sorted(Counter(d for _, _, d in edges).items())),
        "component_size_distribution": {str(k): v for k, v in sorted(size_dist.items())},
        "largest_component": {"group_id": largest[0], "size": largest[1]} if largest else None,
        "multi_image_groups": sorted(g for g, s in sizes.items() if s > 1),
    }


def split_for_group(group_id: str) -> str:
    """A2.6: u = first64(SHA256('20261021|'+group_id))/2**64; <.70 train, <.85 dev, else test."""
    u = unit_interval_from_text(SPLIT_SALT + group_id)
    if u < SPLIT_TRAIN_UPPER:
        return "train"
    if u < SPLIT_DEV_UPPER:
        return "dev"
    return "test"


def image_shuffle_map(group_ids: Iterable[str], seed: int = SHUFFLE_SEED) -> dict[str, str]:
    """A4: order group IDs, permute with PCG64(seed), assign each group the NEXT group's
    representative image (group ID == smallest RGB hash). Requires >= 2 groups.
    Returns {group_id: donor_group_id}. A fresh generator is used per held-out split."""
    ordered = sorted(set(group_ids))
    if len(ordered) < 2:
        raise ValueError("image shuffle requires at least two groups")
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(len(ordered))
    seq = [ordered[i] for i in perm]
    mapping = {seq[i]: seq[(i + 1) % len(seq)] for i in range(len(seq))}
    for g, donor in mapping.items():
        if g == donor:
            raise AssertionError("shuffle mapped a group to itself")
    return mapping


def ego_overlap(
    normlens_phash: Mapping[str, int],
    normlens_group_of: Mapping[str, str],
    ego_frames: Mapping[str, list[tuple[str, int]]],
    max_hamming: int = PHASH_MAX_HAMMING,
) -> list[dict]:
    """Compare NormLens images with decoded pre-action Ego frames (exact RGB or pHash<=4).

    ego_frames: {ego_id: [(frame_rgb_sha256, frame_phash), ...]}. No Ego labels are used."""
    hits = []
    for ego_id in sorted(ego_frames):
        for frame_rgb, frame_ph in ego_frames[ego_id]:
            for rgb in sorted(normlens_phash):
                exact = rgb == frame_rgb
                d = hamming64(normlens_phash[rgb], frame_ph)
                if exact or d <= max_hamming:
                    hits.append({
                        "ego_id": ego_id,
                        "frame_rgb_sha256": frame_rgb,
                        "normlens_rgb_sha256": rgb,
                        "normlens_group_id": normlens_group_of[rgb],
                        "exact": exact,
                        "hamming": d,
                    })
    return hits
