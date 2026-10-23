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


def grouped_mean_ci(
    groups: dict[Hashable, Sequence[float]],
    n: int = 1000,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean over all values with a grouped-bootstrap percentile CI, ignoring NaN.

    Same estimate as ``grouped_bootstrap`` with ``np.nanmean``, but vectorised over per-group sums
    and counts: Task 3 needs a few thousand CIs over ~500 photos each.
    """
    rng = rng or np.random.default_rng(0)
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[~np.isnan(a)] for a in arrays]
    sums = np.array([a.sum() for a in arrays])
    counts = np.array([a.size for a in arrays], dtype=float)
    if counts.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(sums.sum() / counts.sum())
    if n <= 0:
        return point, float("nan"), float("nan")
    idx = rng.integers(len(arrays), size=(n, len(arrays)))
    with np.errstate(invalid="ignore", divide="ignore"):
        est = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.nanpercentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def grouped_bootstrap_fn(
    groups: dict[Hashable, object],
    fn: Callable[[list], float],
    n: int = 1000,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Any statistic ``fn(list of group values)`` with a grouped-bootstrap percentile CI (e.g. an
    AUROC, whose groups hold (positives, negatives) of one source photo)."""
    rng = rng or np.random.default_rng(0)
    values = list(groups.values())
    point = float(fn(values)) if values else float("nan")
    if not values or n <= 0:
        return point, float("nan"), float("nan")
    est = np.array(
        [fn([values[j] for j in rng.integers(len(values), size=len(values))]) for _ in range(n)]
    )
    lo, hi = np.nanpercentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
