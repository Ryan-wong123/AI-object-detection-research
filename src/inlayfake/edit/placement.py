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
    support=None,
) -> Box | None:
    """Box inside the image that does not touch any (margin-expanded) GT box (condition C4).

    With ``support`` (an ``edit.support.Support``), the box is anchored on a surface pixel and
    must rest on that surface.
    """
    if w > width or h > height:
        return None
    blocked = [expand(b, margin) for b in gt_boxes]
    for _ in range(max_tries):
        if support is None:
            cand = [float(rng.uniform(0, width - w)), float(rng.uniform(0, height - h)), w, h]
        else:
            cand = support.sample_box(w, h, rng)
            if cand is None:
                return None
            if not _inside(cand, width, height) or not support.accepts(cand):
                continue
        if all(intersection(cand, b) == 0 for b in blocked):
            return cand
    return None


def _inside(b: Box, width: int, height: int) -> bool:
    return b[0] >= 0 and b[1] >= 0 and b[0] + b[2] <= width and b[1] + b[3] <= height


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
    support=None,
) -> Box | None:
    """Box covering ``frac_range`` of ``target``'s box area (condition C5).

    Other GT boxes may be covered by at most ``max_other_frac`` of their area so that only one
    original object is deliberately occluded. With ``support``, the box must also rest on it.
    """
    if w > width or h > height:
        return None
    lo, hi = frac_range
    t_area = box_area(target)
    # candidate centres lie in the target's box grown by half the insert size
    cx0, cx1 = target[0] - w / 2, target[0] + target[2] + w / 2
    cy0, cy1 = target[1] - h / 2, target[1] + target[3] + h / 2
    for _ in range(max_tries):
        if support is None:
            cx, cy = rng.uniform(cx0, cx1), rng.uniform(cy0, cy1)
            x = float(np.clip(cx - w / 2, 0, width - w))
            y = float(np.clip(cy - h / 2, 0, height - h))
            cand = [x, y, w, h]
        else:  # anchor = bottom-centre ("on") or centre ("inside") of the box
            dy = h / 2 if support.mode == "on" else 0.0
            cand = support.sample_box(w, h, rng, (cx0, cy0 + dy, cx1, cy1 + dy))
            if cand is None:
                return None
            if not _inside(cand, width, height) or not support.accepts(cand):
                continue
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
    supports_for=None,
    occ_only_rng: np.random.Generator | None = None,
) -> dict | None:
    """One shared geometric plan for a source photo (used by C2, C3, C4 and C5).

    Tries categories in ``category_order`` until a free box is found (the occluding box is
    optional). With ``occ_only_rng``, a photo with no free space for any class may still get an
    occluding box alone (``box_free`` None: C5 only); that search uses its own generator, so the
    shared ``rng`` and therefore every other photo's plan are unchanged. Returns None if nothing
    fits.

    ``supports_for(category) -> [Support]`` (from ``edit.support``) switches on surface
    placement: the free and the occluding box must rest on one of that category's surfaces, tried
    in order. Without it, any free space is allowed (the original design).
    """
    W, H = source["width"], source["height"]
    gt = [a for a in anns if not a.get("iscrowd", 0)]
    gt_boxes = [a["bbox"] for a in anns]  # crowd regions also block free placement
    # occlusion targets: reasonably large, non-crowd originals
    targets = [a for a in gt if box_area(a["bbox"]) >= (2 * cfg["min_side_px"]) ** 2]
    for cat in category_order:
        w, h = sample_size(stats[cat], W, H, rng, cfg["min_side_px"], cfg["max_area_frac"])
        supports = [None] if supports_for is None else supports_for(cat)
        free, sup = None, None
        for sup in supports:
            free = place_free(
                W, H, w, h, gt_boxes, cfg["margin_px"], rng, cfg["max_tries"], support=sup
            )
            if free is not None:
                break
        if free is None:
            continue
        occ, occ_ann = _occluding(W, H, w, h, targets, anns, cfg, rng, sup)
        plan = {
            "source_id": source["source_id"],
            "split": source["split"],
            "category": cat,
            "box_free": free,
            "box_occ": occ,  # None -> C5 skipped for this source
            "occluded_ann_id": occ_ann,
        }
        if sup is not None:
            plan["support"] = sup.name  # the surface both boxes rest on
        return plan
    if occ_only_rng is not None:  # no free space for any class: an occluding box still gives C5
        r2 = occ_only_rng
        for cat in category_order:
            w, h = sample_size(stats[cat], W, H, r2, cfg["min_side_px"], cfg["max_area_frac"])
            occ, occ_ann = _occluding(W, H, w, h, targets, anns, cfg, r2, None)
            if occ is None:  # the class's usual size may be far smaller than the photo's objects
                aspect = float(np.median(stats[cat][:, 1]))
                occ, occ_ann = occluding_sized_to_target(W, H, aspect, targets, anns, cfg, r2)
            if occ is not None:
                return {
                    "source_id": source["source_id"],
                    "split": source["split"],
                    "category": cat,
                    "box_free": None,  # -> no C2, C3, C4 for this source
                    "box_occ": occ,
                    "occluded_ann_id": occ_ann,
                }
    return None


def place_anywhere(
    W: int, H: int, w: float, h: float, gt: list[dict], rng: np.random.Generator
) -> tuple[Box, object]:
    """Last-resort box: anywhere inside the image, with no overlap rule (it may cover any real
    objects; `verify` updates their visible masks). Returns the box and the id of the original
    instance it covers most (None if it covers none)."""
    w, h = min(w, 0.95 * W), min(h, 0.95 * H)
    box = [float(rng.uniform(0, W - w)), float(rng.uniform(0, H - h)), float(w), float(h)]
    covered = [(intersection(box, a["bbox"]), a["id"]) for a in gt if not a.get("iscrowd", 0)]
    covered = [c for c in covered if c[0] > 0]
    return box, (max(covered)[1] if covered else None)


def occluding_sized_to_target(
    W: int,
    H: int,
    aspect: float,
    targets: list[dict],
    anns: list[dict],
    cfg: dict,
    rng: np.random.Generator,
    sup=None,
    tries_per_target: int = 4,
) -> tuple[Box | None, object]:
    """C5 box whose size comes from the object it must cover: area 30–60% of the target's box
    (so a 20–50% overlap is reachable), with the inserted class's aspect ratio (w/h), kept within
    ``max_area_frac`` of the image and ``min_side_px``. Used where the class's usual size is far
    too small for the photo's objects (e.g. one train filling the frame)."""
    for t in rng.permutation(len(targets)) if targets else []:
        target = targets[t]
        others = [a["bbox"] for a in anns if a["id"] != target["id"]]
        for _ in range(tries_per_target):
            area = box_area(target["bbox"]) * float(rng.uniform(0.3, 0.6))
            area = min(area, cfg["max_area_frac"] * W * H)
            w, h = float(np.sqrt(area * aspect)), float(np.sqrt(area / aspect))
            if min(w, h) < cfg["min_side_px"]:
                s = cfg["min_side_px"] / min(w, h)
                w, h = w * s, h * s
            if w > 0.95 * W or h > 0.95 * H:
                k = min(0.95 * W / w, 0.95 * H / h)
                w, h = w * k, h * k
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
                support=sup,
            )
            if occ is not None:
                return occ, target["id"]
    return None, None


def _occluding(W, H, w, h, targets, anns, cfg, rng, sup) -> tuple[Box | None, object]:
    """First occluding box over a random target (C5), and that target's annotation id."""
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
            support=sup,
        )
        if occ is not None:
            return occ, target["id"]
    return None, None
