"""Insertion planning: box sizes from COCO statistics, free-space and occluding placement.

Boxes are [x, y, w, h] in pixels (COCO convention). See ARCHITECTURE.md §3.3.
"""

from __future__ import annotations

import numpy as np

from inlayfake.data.coco import CocoIndex

Box = list[float]


def box_area(b: Box) -> float:
    return max(b[2], 0.0) * max(b[3], 0.0)


def intersection(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def expand(b: Box, margin: float) -> Box:
    return [b[0] - margin, b[1] - margin, b[2] + 2 * margin, b[3] + 2 * margin]


def size_stats(index: CocoIndex, categories: list[str], max_area_frac: float) -> dict:
    """Per category: array of (area_frac, aspect=w/h) pairs from non-crowd instances.

    Very small instances (< 1% of the image) are excluded: inpainters rarely produce a recognisable
    object that small, and we want insertions a verifier can confirm.
    """
    stats: dict[str, np.ndarray] = {}
    for cat in categories:
        cid = index.cat_id[cat]
        pairs = []
        for a in index.anns.values():
            if a["category_id"] != cid or a.get("iscrowd", 0):
                continue
            im = index.images[a["image_id"]]
            frac = a["bbox"][2] * a["bbox"][3] / (im["width"] * im["height"])
            if 0.01 <= frac <= max_area_frac and a["bbox"][3] > 0:
                pairs.append((frac, a["bbox"][2] / a["bbox"][3]))
        if not pairs:
            raise ValueError(f"no usable size statistics for category {cat!r}")
        stats[cat] = np.asarray(pairs, dtype=float)
    return stats


def sample_size(
    pairs: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
    min_side: int,
    max_area_frac: float,
) -> tuple[float, float]:
    """Sample (w, h) from one real instance's relative area and aspect ratio."""
    frac, aspect = pairs[rng.integers(len(pairs))]
    area = min(frac, max_area_frac) * width * height
    w = float(np.sqrt(area * aspect))
    h = float(np.sqrt(area / aspect))
    scale = max(1.0, min_side / min(w, h))  # enforce the minimum side, keep the aspect ratio
    w, h = w * scale, h * scale
    shrink = min(1.0, 0.9 * width / w, 0.9 * height / h)  # must fit inside the image
    return w * shrink, h * shrink


def place_free(
    width: int,
    height: int,
    w: float,
    h: float,
    gt_boxes: list[Box],
    margin: float,
    rng: np.random.Generator,
    max_tries: int,
) -> Box | None:
    """Box inside the image that does not touch any (margin-expanded) GT box (condition C4)."""
    if w > width or h > height:
        return None
    blocked = [expand(b, margin) for b in gt_boxes]
    for _ in range(max_tries):
        cand = [float(rng.uniform(0, width - w)), float(rng.uniform(0, height - h)), w, h]
        if all(intersection(cand, b) == 0 for b in blocked):
            return cand
    return None


def place_occluding(
    width: int,
    height: int,
    w: float,
    h: float,
    target: Box,
    others: list[Box],
    frac_range: tuple[float, float],
    rng: np.random.Generator,
    max_tries: int,
    max_other_frac: float = 0.05,
) -> Box | None:
    """Box covering ``frac_range`` of ``target``'s box area (condition C5).

    Other GT boxes may be covered by at most ``max_other_frac`` of their area so that only one
    original object is deliberately occluded.
    """
    if w > width or h > height:
        return None
    lo, hi = frac_range
    t_area = box_area(target)
    for _ in range(max_tries):
        # sample the candidate's centre near the target's border region
        cx = rng.uniform(target[0] - w / 2, target[0] + target[2] + w / 2)
        cy = rng.uniform(target[1] - h / 2, target[1] + target[3] + h / 2)
        x = float(np.clip(cx - w / 2, 0, width - w))
        y = float(np.clip(cy - h / 2, 0, height - h))
        cand = [x, y, w, h]
        if not lo <= intersection(cand, target) / t_area <= hi:
            continue
        if all(intersection(cand, o) <= max_other_frac * box_area(o) for o in others):
            return cand
    return None


def plan_source(
    source: dict,
    anns: list[dict],
    category_order: list[str],
    stats: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> dict | None:
    """One shared geometric plan for a source photo (used by C2, C3, C4 and C5).

    Tries categories in ``category_order`` until both a free box and an occluding box are found.
    Returns None if no category fits.
    """
    W, H = source["width"], source["height"]
    gt = [a for a in anns if not a.get("iscrowd", 0)]
    gt_boxes = [a["bbox"] for a in anns]  # crowd regions also block free placement
    # occlusion targets: reasonably large, non-crowd originals
    targets = [a for a in gt if box_area(a["bbox"]) >= (2 * cfg["min_side_px"]) ** 2]
    for cat in category_order:
        w, h = sample_size(stats[cat], W, H, rng, cfg["min_side_px"], cfg["max_area_frac"])
        free = place_free(W, H, w, h, gt_boxes, cfg["margin_px"], rng, cfg["max_tries"])
        if free is None:
            continue
        occ, occ_ann = None, None
        for t in rng.permutation(len(targets)) if targets else []:
            target = targets[t]
            others = [a["bbox"] for a in anns if a["id"] != target["id"]]
            occ = place_occluding(
                W,
                H,
                w,
                h,
                target["bbox"],
                others,
                tuple(cfg["occlusion_frac"]),
                rng,
                cfg["max_tries"],
            )
            if occ is not None:
                occ_ann = target["id"]
                break
        return {
            "source_id": source["source_id"],
            "split": source["split"],
            "category": cat,
            "box_free": free,
            "box_occ": occ,  # None -> C5 skipped for this source
            "occluded_ann_id": occ_ann,
        }
    return None
