"""Grouped bootstrap: resample source photographs, not images (edits of one photo correlate)."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence

import numpy as np


def grouped_bootstrap(
    groups: dict[Hashable, Sequence[float]],
    stat: Callable[[np.ndarray], float] = np.mean,
    n: int = 1000,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point estimate, lower, upper) of ``stat`` over all values, resampling groups."""
    rng = rng or np.random.default_rng(0)
    keys = list(groups)
    arrays = [np.asarray(groups[k], dtype=float) for k in keys]
    point = float(stat(np.concatenate(arrays))) if arrays else float("nan")
    if not keys or n <= 0:
        return point, float("nan"), float("nan")
    est = np.empty(n)
    for i in range(n):
        idx = rng.integers(len(keys), size=len(keys))
        est[i] = stat(np.concatenate([arrays[j] for j in idx]))
    lo, hi = np.nanpercentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
