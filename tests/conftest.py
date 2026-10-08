"""Synthetic COCO fixture: small JPEGs with rectangle-polygon annotations for all categories."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

CATS = [
    "person",
    "chair",
    "cup",
    "bottle",
    "potted plant",
    "vase",
    "clock",
    "bench",
    "umbrella",
    "backpack",
    "dog",
    "tv",
]


def rect_poly(x, y, w, h):
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


@pytest.fixture
def coco_dir(tmp_path):
    rng = np.random.default_rng(0)
    img_dir = tmp_path / "coco" / "val2017"
    ann_dir = tmp_path / "coco" / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    images, anns = [], []
    aid = 1
    for i in range(1, 25):
        W, H = 400, 320
        arr = rng.integers(60, 200, size=(H, W, 3), dtype=np.uint8)
        objs = [(CATS[1 + (i % 10)], 30, 40, 90, 100), (CATS[1 + ((i + 3) % 10)], 250, 150, 70, 90)]
        if i == 1:  # an image dominated by a person -> ineligible
            objs.append(("person", 0, 0, 380, 300))
        for cat, x, y, w, h in objs:
            arr[y : y + h, x : x + w] = rng.integers(0, 255, size=3)
            anns.append(
                {
                    "id": aid,
                    "image_id": i,
                    "category_id": CATS.index(cat) + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": rect_poly(x, y, w, h),
                }
            )
            aid += 1
        fn = f"{i:012d}.jpg"
        Image.fromarray(arr).save(img_dir / fn, quality=90)
        images.append({"id": i, "file_name": fn, "width": W, "height": H})
    data = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": k + 1, "name": c} for k, c in enumerate(CATS)],
    }
    (ann_dir / "instances_val2017.json").write_text(json.dumps(data))
    return tmp_path / "coco"


@pytest.fixture
def cfg(coco_dir, tmp_path):
    from pathlib import Path

    from inlayfake.config import deep_merge, load_config

    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "dryrun.yaml")
    return deep_merge(
        base,
        {
            "sampling": {"n_sources": 12},
            "paths": {
                "coco_annotations": str(coco_dir / "annotations" / "instances_val2017.json"),
                "coco_images": str(coco_dir / "val2017"),
                "work": str(tmp_path / "work"),
            },
        },
    )
