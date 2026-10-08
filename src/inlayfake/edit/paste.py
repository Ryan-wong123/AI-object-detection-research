"""Real-object paste control (condition C3, ARCHITECTURE.md §3.4)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from inlayfake.data.coco import CocoIndex
from inlayfake.edit.compose import composite
from inlayfake.edit.placement import Box


def choose_donor(
    index: CocoIndex,
    category: str,
    exclude_image_id: int,
    target: Box,
    rng: np.random.Generator,
    min_side: int = 48,
    aspect_tol: float = 0.25,
) -> dict | None:
    """Pick a same-category, non-crowd, non-truncated instance from another photo whose aspect
    ratio is within ``aspect_tol`` of the target box (relaxed if nothing matches)."""
    cid = index.cat_id[category]
    t_aspect = target[2] / target[3]
    cands, loose = [], []
    for a in index.anns.values():
        if a["category_id"] != cid or a.get("iscrowd", 0) or a["image_id"] == exclude_image_id:
            continue
        x, y, w, h = a["bbox"]
        im = index.images[a["image_id"]]
        if min(w, h) < min_side:
            continue
        if x <= 1 or y <= 1 or x + w >= im["width"] - 1 or y + h >= im["height"] - 1:
            continue  # touching the border: probably truncated
        loose.append(a)
        if abs(np.log((w / h) / t_aspect)) <= np.log(1 + aspect_tol):
            cands.append(a)
    pool = sorted(cands or loose, key=lambda a: a["id"])
    if not pool:
        return None
    return pool[int(rng.integers(len(pool)))]


def paste_instance(
    base: np.ndarray,
    donor_img: np.ndarray,
    donor_mask: np.ndarray,
    donor_bbox: Box,
    target: Box,
    feather_px: int,
) -> tuple[np.ndarray, np.ndarray, Box]:
    """Scale the donor to fit inside ``target`` (aspect kept, bottom-centre aligned) and
    alpha-paste it. Returns (image, pasted mask bool, tight bbox of the pasted mask)."""
    H, W = base.shape[:2]
    x, y, w, h = (int(round(v)) for v in donor_bbox)
    patch = donor_img[y : y + h, x : x + w]
    pmask = donor_mask[y : y + h, x : x + w]
    s = min(target[2] / w, target[3] / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    px = int(round(target[0] + (target[2] - nw) / 2))
    py = int(round(target[1] + target[3] - nh))
    px, py = min(max(px, 0), W - nw), min(max(py, 0), H - nh)

    patch_r = np.asarray(Image.fromarray(patch).resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
    m = Image.fromarray((pmask * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    alpha = np.asarray(m, dtype=np.float32) / 255.0
    if feather_px > 0:
        from PIL import ImageFilter

        blurred = m.filter(ImageFilter.GaussianBlur(feather_px / 2))
        alpha = np.minimum(alpha, np.asarray(blurred, dtype=np.float32) / 255.0)

    out = base.copy()
    region = out[py : py + nh, px : px + nw]
    out[py : py + nh, px : px + nw] = composite(region, patch_r, alpha)
    full = np.zeros((H, W), dtype=bool)
    full[py : py + nh, px : px + nw] = alpha > 0.5
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        bbox = [float(px), float(py), float(nw), float(nh)]
    else:
        bbox = [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1),
        ]
    return out, full, bbox
