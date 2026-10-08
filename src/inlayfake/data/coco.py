"""Minimal COCO annotation index (plain JSON; pycocotools only needed for RLE masks)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class CocoIndex:
    images: dict[int, dict]
    anns_by_image: dict[int, list[dict]]
    anns: dict[int, dict]
    cat_name: dict[int, str]
    cat_id: dict[str, int]
    image_dir: Path

    @classmethod
    def load(cls, annotation_file: str | Path, image_dir: str | Path) -> CocoIndex:
        data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        return cls.from_dict(data, image_dir)

    @classmethod
    def from_dict(cls, data: dict, image_dir: str | Path) -> CocoIndex:
        images = {im["id"]: im for im in data["images"]}
        anns = {a["id"]: a for a in data["annotations"]}
        by_image: dict[int, list[dict]] = defaultdict(list)
        for a in data["annotations"]:
            by_image[a["image_id"]].append(a)
        cat_name = {c["id"]: c["name"] for c in data["categories"]}
        return cls(
            images=images,
            anns_by_image=dict(by_image),
            anns=anns,
            cat_name=cat_name,
            cat_id={v: k for k, v in cat_name.items()},
            image_dir=Path(image_dir),
        )

    def image_path(self, image_id: int) -> Path:
        return self.image_dir / self.images[image_id]["file_name"]

    def image_anns(self, image_id: int) -> list[dict]:
        return self.anns_by_image.get(image_id, [])

    def ann_mask(self, ann: dict) -> np.ndarray:
        im = self.images[ann["image_id"]]
        return ann_to_mask(ann, im["height"], im["width"])


def ann_to_mask(ann: dict, height: int, width: int) -> np.ndarray:
    """Boolean HxW mask. Polygons are rasterised with PIL; RLE needs pycocotools."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for poly in seg:
            if len(poly) >= 6:
                draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
        return np.asarray(canvas, dtype=bool)
    from pycocotools import mask as mask_utils  # RLE (mostly iscrowd annotations)

    rle = seg
    if isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)
    return mask_utils.decode(rle).astype(bool)
