"""Cyclic option-position diagnostic and legal-answer candidate algebra (plan A5.1).

Rotation r: p_r[j] = (j + r) % 5, i.e. displayed candidate j is original candidate p_r[j].
Candidates: bitmasks m = 0..31 rendered exactly as repr([i + 1 for i in range(5) if (m >> i) & 1]).
"""

from __future__ import annotations

from typing import Sequence

N = 5


def rotation(r: int) -> list[int]:
    if not 0 <= r < N:
        raise ValueError("rotation must be in 0..4")
    return [(j + r) % N for j in range(N)]


def rotate_options(options: Sequence[str], r: int) -> list[str]:
    """Displayed order; strings (including the actual empty None candidate) are unchanged."""
    if len(options) != N:
        raise ValueError("five options required")
    return [options[p] for p in rotation(r)]


def derotate(displayed_indices, r: int) -> frozenset:
    p = rotation(r)
    return frozenset(p[j] for j in displayed_indices)


def set_jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def candidate_text(mask: int) -> str:
    if not 0 <= mask < 2 ** N:
        raise ValueError("bitmask out of range")
    return repr([i + 1 for i in range(N) if (mask >> i) & 1])


def candidate_set(mask: int) -> frozenset:
    return frozenset(i for i in range(N) if (mask >> i) & 1)


def select_candidate(logprob_by_mask: Sequence[float]) -> int:
    """Maximal summed log-probability; exact ties choose the smallest bitmask."""
    if len(logprob_by_mask) != 2 ** N:
        raise ValueError("32 candidate scores required")
    best = 0
    for m in range(1, 2 ** N):
        if logprob_by_mask[m] > logprob_by_mask[best]:
            best = m
    return best


def check_prefix_free(token_seqs: Sequence[Sequence[int]]) -> None:
    """The 32 candidate token sequences (incl. terminator) must be distinct and prefix-free."""
    seqs = [tuple(s) for s in token_seqs]
    if len(set(seqs)) != len(seqs):
        raise ValueError("candidate token sequences are not distinct")
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            if i != j and len(a) < len(b) and b[:len(a)] == a:
                raise ValueError("candidate token sequences are not prefix-free")
