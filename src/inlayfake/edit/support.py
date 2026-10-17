"""Supporting surfaces for insertion boxes (ARCHITECTURE.md §3.3, `placement.support`).

The first SDXL pilot (task2_pilot) placed boxes in any free space and SDXL mostly filled them with
background (an umbrella in empty sky, a chair mid-road). Here a box must *rest on* a plausible
surface: its bottom band lies on floor / ground / table pixels from the COCO-Stuff human labels.
Wall-mounted classes (clock) may instead sit *inside* a wall region.

Masks are boolean HxW arrays. Boxes are [x, y, w, h] in pixels.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inlayfake.edit.placement import Box

ON, INSIDE = "on", "inside"


@dataclass
class StuffIndex:
    """COCO-Stuff annotations by image id, with category names."""

    anns_by_image: dict[int, list[dict]]
    cat_name: dict[int, str]

    @classmethod
    def load(cls, path: str | Path) -> StuffIndex:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        by_image: dict[int, list[dict]] = defaultdict(list)
        for a in data["annotations"]:
            by_image[a["image_id"]].append(a)
        return cls(dict(by_image), {c["id"]: c["name"] for c in data["categories"]})

    def mask(self, image_id: int, classes: set[str], height: int, width: int) -> np.ndarray:
        """Union of the given stuff classes in one image."""
        out = np.zeros((height, width), dtype=bool)
        for a in self.anns_by_image.get(image_id, []):
            if self.cat_name.get(a["category_id"]) in classes:
                out |= _decode(a["segmentation"], height, width)
        return out


def _decode(seg, height: int, width: int) -> np.ndarray:
    from inlayfake.data.coco import ann_to_mask

    if isinstance(seg, dict) and isinstance(seg.get("counts"), str):
        seg = {"size": seg["size"], "counts": seg["counts"].encode("ascii")}
    return ann_to_mask({"segmentation": seg}, height, width)


def support_frac(box: Box, mask: np.ndarray, mode: str, band: tuple[float, float]) -> float:
    """Share of surface pixels that supports ``box``.

    ``on``: the band from ``band[0]·h`` above the box bottom to ``band[1]·h`` below it, over the
    central 80% of the box width (where an object's base touches the surface).
    ``inside``: the whole box.
    """
    H, W = mask.shape
    x, y, w, h = box
    if mode == INSIDE:
        x0, x1, y0, y1 = x, x + w, y, y + h
    else:
        x0, x1 = x + 0.1 * w, x + 0.9 * w
        y0, y1 = y + h - band[0] * h, y + h + band[1] * h
    c0, c1 = int(np.floor(max(x0, 0))), int(np.ceil(min(x1, W)))
    r0, r1 = int(np.floor(max(y0, 0))), int(np.ceil(min(y1, H)))
    if c1 <= c0 or r1 <= r0:
        return 0.0
    return float(mask[r0:r1, c0:c1].mean())


@dataclass
class Support:
    """One candidate surface for a box: a mask, how the box must relate to it, and a name."""

    name: str
    mask: np.ndarray
    mode: str
    band: tuple[float, float]
    min_frac: float

    def accepts(self, box: Box) -> bool:
        return support_frac(box, self.mask, self.mode, self.band) >= self.min_frac

    def sample_box(
        self,
        w: float,
        h: float,
        rng: np.random.Generator,
        region: tuple[float, float, float, float] | None = None,
    ) -> Box | None:
        """A w×h box anchored on a random surface pixel: its bottom-centre (``on``) or centre
        (``inside``). ``region`` (x0, y0, x1, y1) restricts the anchor; None if no pixel fits."""
        # the surface pixels are computed once per Support: re-scanning the full mask on every
        # try made re-planning hundreds of boxes take many minutes (same pixels, same order)
        if getattr(self, "_pixels", None) is None:
            self._pixels = np.nonzero(self.mask)
        ys, xs = self._pixels
        if region is not None:
            keep = (xs >= region[0]) & (xs <= region[2]) & (ys >= region[1]) & (ys <= region[3])
            ys, xs = ys[keep], xs[keep]
        if len(xs) == 0:
            return None
        i = rng.integers(len(xs))
        ax, ay = float(xs[i]) + 0.5, float(ys[i]) + 0.5
        top = ay - h if self.mode == ON else ay - h / 2
        return [ax - w / 2, top, w, h]


def category_supports(
    image_id: int, height: int, width: int, category: str, stuff: StuffIndex, scfg: dict
) -> list[Support]:
    """Surfaces allowed for ``category`` in this image, in config order (empty masks dropped).
    Classes not listed in ``categories`` use ``default_surfaces``."""
    out = []
    band = tuple(scfg["band"])
    for name in scfg["categories"].get(category, scfg.get("default_surfaces", [])):
        group = scfg["groups"][name]
        m = stuff.mask(image_id, set(group["classes"]), height, width)
        if m.any():
            out.append(Support(name, m, group.get("mode", ON), band, scfg["min_frac"]))
    return out
