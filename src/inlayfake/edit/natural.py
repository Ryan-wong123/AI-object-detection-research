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
    sets = y.get("surface_sets", {})
    supports = y.get("supports", {})
    classes = {}
    for group, g in y["groups"].items():
        sup = supports.get(group, {})
        for c in g["classes"]:
            if c in classes:
                raise ValueError(f"class {c!r} is in two prompt groups")
            classes[c] = {
                "group": group,
                "where": g["where"],
                "how": g["how"],
                "air": bool(g.get("air", False)),
                # allowed support under the point (fix 1): COCO-Stuff names and COCO classes
                "surfaces": {n for k in sup.get("surfaces", []) for n in sets[k]},
                "on_objects": set(sup.get("on_objects", [])),
                "any_surface": bool(sup.get("any_surface", False)),
            }
    return {
        "template": y["template"],
        "box_template": y.get("box_template"),
        "box_retry_template": y.get("box_retry_template"),
        "last_resort_template": y.get("last_resort_template"),
        "size_hint_template": y.get("size_hint_template"),
        "size_retry_template": y.get("size_retry_template"),
        "any_class_template": y.get("any_class_template"),
        "any_class_alt_template": y.get("any_class_alt_template"),
        "check": y["check"],
        "rules": y.get("box_rules", {}),
        "classes": classes,
    }


def pointing_prompt(prompts: dict, category: str) -> str:
    c = prompts["classes"][category]
    return prompts["template"].format(where=c["where"], how=c["how"], **{"class": category})


def _where_short(where: str) -> str:
    """A group's ``where`` without the leading "empty spots on"."""
    for lead in ("empty spots on the ", "empty spots on ", "empty spots in ", "empty spots "):
        if where.startswith(lead):
            return "the " + where[len(lead) :] if lead.endswith("the ") else where[len(lead) :]
    return where


def elongation(w: float, h: float) -> float:
    """Long side / short side (orientation-free: a lying and a standing bottle are alike)."""
    return max(w, h) / max(min(w, h), 1e-6)


def shape_problem(
    box: list[float], W: int, H: int, typical_aspect: float | None, sc: dict
) -> str | None:
    """Why a box cannot be generated well (None: fine): ``too small`` (short side under
    ``min_side_px`` or area under ``min_area_frac`` of the image) or ``too thin`` (elongation more
    than ``max_elongation_factor`` x the class's typical COCO elongation)."""
    _, _, w, h = box
    if min(w, h) < sc.get("min_side_px", 32) or w * h < sc.get("min_area_frac", 0.005) * W * H:
        return "too small"
    if typical_aspect:
        typ = elongation(typical_aspect, 1.0)
        if elongation(w, h) > sc.get("max_elongation_factor", 3.0) * typ:
            return "too thin"
    return None


def size_hint(prompts: dict, category: str, min_side_px: float, W: int, H: int) -> str:
    return prompts["size_hint_template"].format(
        min_side=min_side_grid(min_side_px, W, H), **{"class": category}
    )


def size_retry_text(
    prompts: dict, category: str, failed: list[tuple[list[float], str]], W: int, H: int
) -> str:
    """Feedback sentence listing the failed boxes (0-1000 grid) and why."""
    fb = "; ".join(f"{grid_box(b, W, H)} {why}" for b, why in failed)
    return prompts["size_retry_template"].format(feedback=fb, **{"class": category})


def last_resort_prompt(
    prompts: dict, category: str, existing: list[list[float]], W: int, H: int
) -> str:
    """Last-resort question: same class, may be cut off by the photo's edge or further back; no
    "no place" answer."""
    return box_prompt(prompts, category, existing, W, H, key="last_resort_template")


def box_prompt(
    prompts: dict,
    category: str,
    existing: list[list[float]] | None = None,
    W: int = 1,
    H: int = 1,
    key: str = "box_template",
) -> str:
    """Box-mode question (``box_template``) for one class; ``where_short`` is the group's
    ``where`` without the leading "empty spots on"; ``existing`` = the photo's own objects of the
    class ([x, y, w, h] pixels), given to the VLM on its 0-1000 grid so it does not box them."""
    c = prompts["classes"][category]
    where = _where_short(c["where"])
    boxes = [
        [
            round(x / W * 1000),
            round(y / H * 1000),
            round((x + w) / W * 1000),
            round((y + h) / H * 1000),
        ]
        for x, y, w, h in (existing or [])
    ]
    if not boxes:
        listed, them = f"no {category}", "anything"
    elif len(boxes) == 1:
        listed, them = f"one {category} at bbox_2d {boxes[0]}", "it"
    else:
        listed = f"{len(boxes)} of them at bbox_2d " + ", ".join(str(b) for b in boxes)
        them = "them"
    return prompts[key].format(
        where_short=where, how=c["how"], existing=listed, them=them, **{"class": category}
    )


def grid_box(box: list[float], W: int, H: int) -> list[int]:
    """[x, y, w, h] pixels -> [x1, y1, x2, y2] on the VLM's 0-1000 grid."""
    x, y, w, h = box
    return [
        round(x / W * 1000),
        round(y / H * 1000),
        round((x + w) / W * 1000),
        round((y + h) / H * 1000),
    ]


def _taken_text(taken: list[list[float]], W: int, H: int, max_n: int = 30) -> str:
    """The largest ``max_n`` taken boxes as a list on the 0-1000 grid ("none" if empty)."""
    big = sorted(taken, key=lambda b: -b[2] * b[3])[:max_n]
    return ", ".join(str(grid_box(b, W, H)) for b in big) or "none"


def min_side_grid(min_side_px: float, W: int, H: int) -> int:
    """Minimum box side in pixels on the 0-1000 grid of the shorter image side (conservative)."""
    return round(min_side_px / min(W, H) * 1000)


def box_retry_prompt(
    prompts: dict, category: str, taken: list[list[float]], min_side_px: float, W: int, H: int
) -> str:
    """Second-round question: taken places (labelled objects, rejected boxes), minimum size."""
    c = prompts["classes"][category]
    return prompts["box_retry_template"].format(
        taken=_taken_text(taken, W, H),
        where_short=_where_short(c["where"]),
        how=c["how"],
        min_side=min_side_grid(min_side_px, W, H),
        **{"class": category},
    )


def any_class_prompt(
    prompts: dict,
    classes: list[str],
    taken: list[list[float]],
    min_side_px: float,
    W: int,
    H: int,
    alt: bool = False,
) -> str:
    """Any-class question: the VLM picks one of ``classes`` that fits the scene (``alt``: the
    wording that starts from free space)."""
    key = "any_class_alt_template" if alt else "any_class_template"
    return prompts[key].format(
        taken=_taken_text(taken, W, H),
        class_list=", ".join(classes),
        min_side=min_side_grid(min_side_px, W, H),
    )


def dedupe_boxes(boxes: list[list[float]], max_n: int, max_iou: float = 0.5) -> list[list[float]]:
    """Boxes in the VLM's order without near-duplicates (IoU > ``max_iou`` with an earlier box),
    at most ``max_n``."""
    out: list[list[float]] = []
    for b in boxes:
        if all(_iou(b, o) <= max_iou for o in out):
            out.append(b)
        if len(out) == max_n:
            break
    return out


def _iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


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
def point_support(stuff_label: str | None, thing: str | None, entry: dict) -> tuple[bool, str]:
    """Is the surface under a VLM point one this class may rest on? (``supports`` in the prompt
    file.) ``thing``: class of the labelled object under the point (None: no object);
    ``stuff_label``: COCO-Stuff class there ("other" marks object pixels; None: no COCO-Stuff
    labels for this photo, so nothing can be checked)."""
    if entry.get("any_surface"):  # pole-mounted classes: no support check
        return True, "ok"
    if thing is not None:
        if thing in entry["on_objects"]:
            return True, "ok"
        return False, f"point on a real {thing}"
    if stuff_label is None:
        return True, "ok"
    if stuff_label == "other":
        return False, "point on an unlabelled object"
    if stuff_label in entry["surfaces"]:
        return True, "ok"
    return False, f"point on {stuff_label}"


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
    min_area_frac: float = 0.0,
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
    # at least min_area_frac of the image (fix 2: SDXL fills small boxes poorly; in the first
    # natural pilot the median box was 1.5% of the image and its object filled 28% of it)
    if min_area_frac > 0 and w * h < min_area_frac * W * H:
        k_area = np.sqrt(min_area_frac * W * H / max(w * h, 1e-6))
        w, h = w * k_area, h * k_area
    if min(w, h) < min_side:
        k_min = min_side / min(w, h)
        w, h = w * k_min, h * k_min
    if on_sky:
        box = [px - w / 2, py - h / 2, w, h]
    else:
        box = [px - w / 2, py - h, w, h]
    info.update(ratio_raw=round(raw, 4), ratio=round(ratio, 4))
    return [float(v) for v in box], info


def fit_inside(
    box: Box, W: int, H: int, max_shift: float = 0.15, max_side_shift: float = 0.5
) -> Box | None:
    """Shift a box into the image; None if that is not possible without losing the spot.

    Sideways it may move up to ``max_side_shift`` of its width: the VLM point (the contact point)
    then still lies on the box's base, just not at its centre (pilot 2026-10-19: with 15%, 46 of
    208 candidates near the left or right edge were rejected). Up or down only ``max_shift`` of
    its height, so the object keeps standing on its surface."""
    x, y, w, h = box
    if w > W or h > H:
        return None
    dx = -x if x < 0 else min(0.0, W - (x + w))
    dy = -y if y < 0 else min(0.0, H - (y + h))
    if abs(dx) > max_side_shift * w or abs(dy) > max_shift * h:
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
    max_person = rules.get("max_cover_person", 0.0)
    for a in anns:
        if a["category"] != "person":
            continue
        cover = intersection(box, a["bbox"])
        if max_person <= 0 and cover > 0:
            return False, "covers a person"
        # 2026-10-20 (author): a person may be partly covered, at most max_cover_person of
        # their box (also crowd regions of people)
        if max_person > 0 and cover > max_person * box_area(a["bbox"]) + 1e-6:
            return False, f"covers {cover / box_area(a['bbox']):.0%} of a person"
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
