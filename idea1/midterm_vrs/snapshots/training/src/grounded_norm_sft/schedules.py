"""Deterministic, label-independent schedules (plan 5.2, 5.3, 9.C.10, A3).

Plan-specified seeds use SHA256(compact JSON) -> first 64 bits unsigned big-endian -> PCG64.
Two seeds are implementation-defined here because the plan only says "deterministically from
run seed": the NormLens epoch order and the NormBank draw stream (tags *_v1 below). They are
frozen with Seal A and shared across arms for a given optimization seed.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .constants import (EFFECTIVE_BATCH, FULL_ATTENTION_LAYERS, INTERVENTION_SCHEDULE_TAG, POTENCY_MAX_INPUTS,
                        POTENCY_MIN_INPUTS, POTENCY_SCHEDULE_TAG, POTENCY_SORT_PREFIX, RANDOM_SUPPORT_TAG,
                        SCHEDULE_SEED_BASE, SUPPORT_FRACTION)
from .hashing import seed64_from_json, sort_key_sha256

NORMLENS_ORDER_TAG = "normlens_epoch_order_v1"
NORMBANK_DRAW_TAG = "normbank_draw_v1"
FAMILIES = ("FA", "D")


def _gen(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


# ------------------------------------------------------------------ support
def support_k(n: int) -> int:
    """k = max(1, min(n-1, ceil(.25 n))); requires n >= 2 image tokens."""
    if n < 2:
        raise ValueError("support requires at least two image tokens (example is unsupported)")
    return max(1, min(n - 1, math.ceil(SUPPORT_FRACTION * n)))


def topk_support(scores: Sequence[float]) -> list[int]:
    """Targeted support from per-image-token affinity scores (relative indices).

    Highest scores first; deterministic tie break by smaller token index. Nonfinite -> error."""
    arr = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("nonfinite selector scores (example is unsupported)")
    k = support_k(arr.size)
    order = sorted(range(arr.size), key=lambda i: (-arr[i], i))
    return sorted(order[:k])


def random_support_seed(example_id: str) -> int:
    return seed64_from_json([RANDOM_SUPPORT_TAG, SCHEDULE_SEED_BASE, example_id])


def random_support(example_id: str, n: int) -> list[int]:
    k = support_k(n)
    perm = _gen(random_support_seed(example_id)).permutation(n)[:k]
    return sorted(int(i) for i in perm)


# ------------------------------------------------------------------ intervention schedule
def intervention_base_entries() -> list[tuple[int, int, str]]:
    return [(b, d, f) for b in range(8) for d in range(3) for f in ["FA", "D"]]


def intervention_schedule(num_slots: int) -> list[tuple[int, int, str]]:
    """Successive permutation(48) cycles from one persistent PCG64; final partial cycle truncated."""
    base = intervention_base_entries()
    rng = _gen(seed64_from_json([INTERVENTION_SCHEDULE_TAG, SCHEDULE_SEED_BASE, 0]))
    out: list[tuple[int, int, str]] = []
    while len(out) < num_slots:
        out.extend(base[int(i)] for i in rng.permutation(len(base)))
    return out[:num_slots]


def site_for_arm(entry: tuple[int, int, str], arm: str) -> tuple[int, str]:
    """Map a base entry to (layer, family) for an arm. Family-only arms force the family while
    reusing the same block/offset sequence."""
    b, d, f = entry
    if arm in ("H-T", "H-R"):
        family = f
    elif arm == "F-T":
        family = "FA"
    elif arm == "D-T":
        family = "D"
    else:
        raise ValueError(f"arm {arm!r} has no intervention schedule")
    layer = 4 * b + 3 if family == "FA" else 4 * b + d
    assert (layer in FULL_ATTENTION_LAYERS) == (family == "FA")
    return layer, family


# ------------------------------------------------------------------ potency
def potency_cohort(eligible_dev_ids: Sequence[str]) -> list[str]:
    ordered = sorted(eligible_dev_ids, key=lambda i: sort_key_sha256(POTENCY_SORT_PREFIX, i))
    cohort = ordered[:min(POTENCY_MAX_INPUTS, len(ordered))]
    if len(cohort) < POTENCY_MIN_INPUTS:
        raise ValueError(f"only {len(cohort)} eligible dev records (<{POTENCY_MIN_INPUTS}); defer potency screen")
    return cohort


def potency_schedule(num_inputs: int) -> list[tuple[int, int]]:
    base = [(b, d) for b in range(8) for d in range(3)]
    rng = _gen(seed64_from_json([POTENCY_SCHEDULE_TAG, SCHEDULE_SEED_BASE, 0]))
    out: list[tuple[int, int]] = []
    while len(out) < num_inputs:
        out.extend(base[int(i)] for i in rng.permutation(len(base)))
    return out[:num_inputs]


# ------------------------------------------------------------------ data order / accumulation
def accumulation_windows(n: int, batch: int = EFFECTIVE_BATCH) -> list[int]:
    """Window sizes for one epoch. Remainder merged into the preceding window; N<batch -> one."""
    if n <= 0:
        raise ValueError("empty training set")
    if n < batch:
        return [n]
    full, r = divmod(n, batch)
    sizes = [batch] * full
    if r:
        sizes[-1] += r
    return sizes


def normlens_epoch_orders(n: int, run_seed: int, epochs: int) -> list[list[int]]:
    rng = _gen(seed64_from_json([NORMLENS_ORDER_TAG, SCHEDULE_SEED_BASE, run_seed]))
    return [[int(i) for i in rng.permutation(n)] for _ in range(epochs)]


def normbank_draws(train_ids: Sequence[str], total: int, run_seed: int) -> list[str]:
    """Uniform draws of canonical inputs without replacement until exhaustion, then reshuffle."""
    ids = sorted(train_ids)
    if not ids:
        raise ValueError("no eligible NormBank train inputs")
    rng = _gen(seed64_from_json([NORMBANK_DRAW_TAG, SCHEDULE_SEED_BASE, run_seed]))
    out: list[str] = []
    while len(out) < total:
        out.extend(ids[int(i)] for i in rng.permutation(len(ids)))
    return out[:total]


def build_training_schedule(normlens_ids: Sequence[str], normbank_ids: Sequence[str], run_seed: int,
                            epochs: int, arm: str) -> dict:
    """Full microbatch schedule. NormLens ids are given in canonical (sorted) order."""
    nl = sorted(normlens_ids)
    n = len(nl)
    windows = accumulation_windows(n)
    orders = normlens_epoch_orders(n, run_seed, epochs)
    total = n * epochs
    nb = normbank_draws(normbank_ids, total, run_seed)
    entries = intervention_schedule(total) if arm != "B1" else [None] * total
    rows = []
    ordinal = 0
    update = 0
    for epoch in range(epochs):
        pos = 0
        for w_index, size in enumerate(windows):
            for j in range(size):
                entry = entries[ordinal]
                site = site_for_arm(entry, arm) if entry is not None else None
                rows.append({
                    "ordinal": ordinal, "epoch": epoch, "update": update, "window_in_epoch": w_index,
                    "window_size": size, "position_in_window": j,
                    "normlens_id": nl[orders[epoch][pos]], "normbank_id": nb[ordinal],
                    "base_entry": list(entry) if entry is not None else None,
                    "site_layer": site[0] if site else None, "site_family": site[1] if site else None,
                })
                ordinal += 1
                pos += 1
            update += 1
    return {
        "arm": arm, "run_seed": run_seed, "epochs": epochs, "num_normlens": n,
        "windows_per_epoch": len(windows), "window_sizes": windows,
        "total_updates": update, "total_microbatches": ordinal, "rows": rows,
    }


def schedule_counts(schedule: dict) -> dict:
    """Counts by site/family/block; used for logs (no implication every example saw every layer)."""
    from collections import Counter

    rows = [r for r in schedule["rows"] if r["site_layer"] is not None]
    return {
        "by_layer": dict(sorted(Counter(r["site_layer"] for r in rows).items())),
        "by_family": dict(Counter(r["site_family"] for r in rows)),
        "by_block": dict(sorted(Counter(r["site_layer"] // 4 for r in rows).items())),
        "complete_cycles": len(rows) // 48, "remainder_entries": len(rows) % 48,
    }
