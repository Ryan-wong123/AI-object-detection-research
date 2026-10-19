"""Natural placement geometry (``placement.natural``, author decision 2026-10-19).

A VLM points at spots where another object of a class already in the photo could naturally be
(``inlayfake.edit.scene``); this module turns a point into a box and checks it:

* prompts: one fixed template per class from ``configs/placement_prompts.yaml``;
* size: from the photo's own object of that class (the reference), scaled by the ratio of
  *nearness* (relative inverse depth from a monocular depth model) at the new spot and at the
  reference's base, limited to ``size_ratio`` (single-image depth is approximate);
* position: the point is where the object touches its support (bottom centre of the box); for
  classes that may fly (``air``) a point on sky pixels is the object's centre instead;
* rules: inside the image, never on a person, at most ``max_cover_real`` of any real object's
  area covered, and not resting on window / mirror / ceiling / sky pixels (COCO-Stuff), except
  sky for ``air`` classes.

Boxes are [x, y, w, h] in pixels; points are (x, y) in pixels. Pure functions, unit-tested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inlayfake.edit.placement import Box, box_area, intersection

# COCO-Stuff classes an object must not rest on (or, for "sky", must not rest on unless it flies)
BLOCKED_STUFF = (
    "window-blind",
    "window-other",
    "mirror-stuff",
    "ceiling-other",
    "ceiling-tile",
    "sky-other",
    "clouds",
)
SKY_STUFF = ("sky-other", "clouds")


# ------------------------------------------------------------------------------------- prompts
def load_prompts(path: str | Path) -> dict:
    """The prompt file as {template, check, rules, classes: {class: {where, how, air, group}}}."""
    import yaml

    y = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    classes = {}
    for group, g in y["groups"].items():
        for c in g["classes"]:
            if c in classes:
                raise ValueError(f"class {c!r} is in two prompt groups")
            classes[c] = {
                "group": group,
                "where": g["where"],
                "how": g["how"],
                "air": bool(g.get("air", False)),
            }
    return {
        "template": y["template"],
        "check": y["check"],
        "rules": y.get("box_rules", {}),
        "classes": classes,
    }


def pointing_prompt(prompts: dict, category: str) -> str:
    c = prompts["classes"][category]
    return prompts["template"].format(where=c["where"], how=c["how"], **{"class": category})


def check_questions(prompts: dict, category: str) -> list[tuple[str, str, str]]:
    """[(id, full question, expected answer)] for the realism check of one class."""
    chk = prompts["check"]
    air = chk.get("air_clause", "") if prompts["classes"][category]["air"] else ""
    head = chk["instruction"].format(**{"class": category})
    out = []
    for q in chk["questions"]:
        text = q["text"].format(air_clause=air, **{"class": category})
        out.append((q["id"], f"{head} {text}", q["expected"]))
    return out


# ------------------------------------------------------------------------------------ geometry
def spread_points(
    points: list[tuple[float, float]], W: int, H: int, k: int = 5, min_dist: float = 0.1
) -> list[tuple[float, float]]:
    """Up to ``k`` points in the VLM's order, skipping any closer than ``min_dist`` × the image
    diagonal to one already kept (Molmo often lists many points along one line; retries should
    try different spots)."""
    d = min_dist * float(np.hypot(W, H))
    kept: list[tuple[float, float]] = []
    for p in points:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= d for q in kept):
            kept.append(p)
        if len(kept) == k:
            break
    return kept


def nearness_at(depth: np.ndarray, x: float, y: float, r: int = 3) -> float:
    """Median nearness in a (2r+1)² window around a pixel (clipped to the image)."""
    H, W = depth.shape
    xi, yi = int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))
    win = depth[max(yi - r, 0) : yi + r + 1, max(xi - r, 0) : xi + r + 1]
    return float(np.median(win))


def usable_references(
    same: list[dict], others: list[dict], W: int, H: int, max_hidden: float = 0.3
) -> list[dict]:
    """Same-class real objects whose size can be trusted: not touching the image border (likely
    cut off), at least 16 px, and at most ``max_hidden`` of the box overlapped by other labels."""
    out = []
    for a in same:
        x, y, w, h = a["bbox"]
        if a.get("iscrowd", 0) or min(w, h) < 16:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        hidden = sum(intersection(a["bbox"], o["bbox"]) for o in others if o is not a)
        if hidden > max_hidden * w * h:
            continue
        out.append(a)
    return out


def box_from_point(
    point: tuple[float, float],
    depth: np.ndarray,
    refs: list[dict],
    fallback_wh: tuple[float, float],
    on_sky: bool,
    size_ratio: tuple[float, float] = (0.5, 2.0),
    max_area_frac: float = 0.25,
    min_side: float = 32,
) -> tuple[Box, dict]:
    """A box for one VLM point and how it was sized.

    With references: the one whose base nearness is closest to the point's; height = its height
    × clamp(nearness(point) / nearness(ref base)); its width/height ratio is kept. Without one:
    ``fallback_wh`` (typical COCO size of the class), scaled by nearness(point) / median
    nearness of the image. ``on_sky`` (a flying class pointed at sky): the point is the centre
    and no depth scaling is applied (sky has no usable depth)."""
    H, W = depth.shape
    px, py = point
    near_p = nearness_at(depth, px, py)
    lo, hi = size_ratio
    info: dict = {"near_point": round(near_p, 4)}
    if refs:
        bases = [
            nearness_at(depth, r["bbox"][0] + r["bbox"][2] / 2, r["bbox"][1] + r["bbox"][3])
            for r in refs
        ]
        k = int(np.argmin([abs(b - near_p) for b in bases]))
        ref, near_r = refs[k], bases[k]
        rw, rh = ref["bbox"][2], ref["bbox"][3]
        raw = 1.0 if on_sky else near_p / max(near_r, 1e-6)
        info.update(ref_ann_id=ref.get("id"), near_ref=round(near_r, 4), size_from="reference")
    else:
        rw, rh = fallback_wh
        raw = 1.0 if on_sky else near_p / max(float(np.median(depth)), 1e-6)
        info.update(ref_ann_id=None, size_from="coco_typical")
    ratio = float(np.clip(raw, lo, hi))
    h = rh * ratio
    w = h * (rw / rh)
    # limits: at most max_area_frac of the image, at least min_side on the short side
    k_max = np.sqrt(max_area_frac * W * H / max(w * h, 1e-6))
    if k_max < 1:
        w, h = w * k_max, h * k_max
    if min(w, h) < min_side:
        k_min = min_side / min(w, h)
        w, h = w * k_min, h * k_min
    if on_sky:
        box = [px - w / 2, py - h / 2, w, h]
    else:
        box = [px - w / 2, py - h, w, h]
    info.update(ratio_raw=round(raw, 4), ratio=round(ratio, 4))
    return [float(v) for v in box], info


def fit_inside(box: Box, W: int, H: int, max_shift: float = 0.15) -> Box | None:
    """Shift a box into the image if it sticks out by at most ``max_shift`` of its size (a point
    near the border); None if it would have to move further or is larger than the image."""
    x, y, w, h = box
    if w > W or h > H:
        return None
    dx = -x if x < 0 else min(0.0, W - (x + w))
    dy = -y if y < 0 else min(0.0, H - (y + h))
    if abs(dx) > max_shift * w or abs(dy) > max_shift * h:
        return None
    return [x + dx, y + dy, w, h]


def check_box(
    box: Box,
    anns: list[dict],
    blocked: np.ndarray | None,
    rules: dict,
    band: tuple[float, float] = (0.1, 0.05),
    max_blocked: float = 0.5,
) -> tuple[bool, str]:
    """(ok, reason). ``anns``: the photo's labels with ``category`` names (COCO + audit adds).
    ``blocked``: pixels the object must not rest on (None: no COCO-Stuff labels)."""
    x, y, w, h = box
    H, W = blocked.shape if blocked is not None else (None, None)
    for a in anns:
        if a["category"] == "person" and intersection(box, a["bbox"]) > 0:
            if rules.get("max_cover_person", 0.0) <= 0:
                return False, "covers a person"
    max_real = rules.get("max_cover_real", 0.15)
    for a in anns:
        if a["category"] == "person" or a.get("iscrowd", 0):
            continue
        area = box_area(a["bbox"])
        if area > 0 and intersection(box, a["bbox"]) > max_real * area + 1e-6:
            frac = intersection(box, a["bbox"]) / area
            return False, f"covers {frac:.0%} of a real {a['category']}"
    if blocked is not None:
        # the band where the object's base would touch its support
        c0, c1 = int(max(x + 0.1 * w, 0)), int(min(x + 0.9 * w, W))
        r0, r1 = int(max(y + h - band[0] * h, 0)), int(min(y + h + band[1] * h, H))
        if c1 > c0 and r1 > r0 and blocked[r0:r1, c0:c1].mean() > max_blocked:
            return False, "rests on window, mirror, ceiling or sky"
    return True, "ok"
