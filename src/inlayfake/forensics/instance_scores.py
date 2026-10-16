"""Turn a forensic map into image-level and per-instance scores (ARCHITECTURE.md §3.7)."""

from __future__ import annotations

import numpy as np


def image_score(m: np.ndarray) -> dict[str, float]:
    """Image-level summaries: how extreme the most suspicious region is relative to the median."""
    q50, q95, q99 = np.percentile(m, [50, 95, 99])
    return {
        "mean": float(m.mean()),
        "p95": float(q95),
        "p99": float(q99),
        "p99_minus_median": float(q99 - q50),
    }


def ring_mask(
    bbox: list[float], shape: tuple[int, int], frac: float, inside: np.ndarray
) -> np.ndarray:
    """Context band: the bbox expanded by ``frac`` of its size (min 4 px), minus the object."""
    H, W = shape
    x, y, w, h = bbox
    m = max(4.0, frac * max(w, h))
    x0, y0 = int(max(np.floor(x - m), 0)), int(max(np.floor(y - m), 0))
    x1, y1 = int(min(np.ceil(x + w + m), W)), int(min(np.ceil(y + h + m), H))
    ring = np.zeros((H, W), dtype=bool)
    ring[y0:y1, x0:x1] = True
    return ring & ~inside


def bbox_mask(bbox: list[float], shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    x, y, w, h = bbox
    m = np.zeros((H, W), dtype=bool)
    m[
        int(max(y, 0)) : int(min(np.ceil(y + h), H)), int(max(x, 0)) : int(min(np.ceil(x + w), W))
    ] = True
    return m


def instance_score(m: np.ndarray, mask: np.ndarray, bbox: list[float], ring_frac: float) -> dict:
    """Region-pooled (mean, p95 inside the object) and instance-conditioned (inside − context)."""
    if not mask.any():
        return {
            "mean_in": float("nan"),
            "p95_in": float("nan"),
            "mean_ring": float("nan"),
            "contrast": float("nan"),
        }
    inside = m[mask]
    ring = ring_mask(bbox, m.shape, ring_frac, mask)
    mean_ring = float(m[ring].mean()) if ring.any() else float("nan")
    mean_in = float(inside.mean())
    return {
        "mean_in": mean_in,
        "p95_in": float(np.percentile(inside, 95)),
        "mean_ring": mean_ring,
        "contrast": mean_in - mean_ring,
    }
