"""Origin discrimination metrics (ARCHITECTURE §3.11 `origin.py`)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score of a positive > score of a negative), ties counting 1/2 (Mann–Whitney U).

    0.5 = no separation; below 0.5 = positives score *lower*. NaN if either side is empty.
    """
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if not pos.size or not neg.size:
        return float("nan")
    values, inverse, counts = np.unique(
        np.concatenate([pos, neg]), return_inverse=True, return_counts=True
    )
    mid_rank = np.cumsum(counts) - (counts - 1) / 2.0  # average 1-based rank of each value
    ranks = mid_rank[inverse]
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))
