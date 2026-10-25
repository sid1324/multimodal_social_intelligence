"""Paired cluster bootstrap (A4, A5): 2000 replicates, NumPy PCG64 seed 20261021, cluster IDs
sorted, clusters resampled with replacement carrying all their examples (with multiplicity),
example-weighted mean statistic, linear-interpolation 2.5/97.5 percentiles.

Draw definition: one call rng.integers(0, G, size=(reps, G)) on the sorted cluster list.
Values are per-example paired differences (or per-example scores), so pairing is preserved.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..constants import BOOTSTRAP_PERCENTILES, BOOTSTRAP_REPS, BOOTSTRAP_SEED


def cluster_bootstrap(values: Sequence[float], clusters: Sequence[str], reps: int = BOOTSTRAP_REPS,
                      seed: int = BOOTSTRAP_SEED, percentiles=BOOTSTRAP_PERCENTILES) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) != len(clusters):
        raise ValueError("values and clusters differ in length")
    if len(vals) == 0:
        return {"status": "N/A", "reason": "no data", "n_examples": 0, "n_clusters": 0}
    if not np.all(np.isfinite(vals)):
        raise ValueError("nonfinite value in bootstrap input")
    ids = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(ids)}
    sums = np.zeros(len(ids))
    counts = np.zeros(len(ids))
    for v, c in zip(vals, clusters):
        sums[pos[c]] += v
        counts[pos[c]] += 1
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, len(ids), size=(reps, len(ids)))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(stats, list(percentiles), method="linear")
    out = {
        "status": "ok", "point": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "percentiles": list(percentiles), "reps": reps, "seed": seed,
        "n_examples": int(len(vals)), "n_clusters": len(ids),
    }
    if len(ids) == 1:
        out["warning"] = "single cluster: interval is degenerate and uninformative"
    return out
