"""Select source photographs and assign splits deterministically (ARCHITECTURE.md §3.1)."""

from __future__ import annotations

import hashlib

import numpy as np

from inlayfake.data.coco import CocoIndex


def assign_split(source_id: int | str, fractions: dict[str, float], salt: str = "inlayfake") -> str:
    """Hash-based split: stable across runs and independent of sampling order."""
    h = hashlib.sha256(f"{salt}:{source_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    acc = 0.0
    for name in ("train", "val", "test"):
        acc += fractions[name]
        if u < acc:
            return name
    return "test"


def is_eligible(index: CocoIndex, image_id: int, cfg: dict) -> bool:
    im = index.images[image_id]
    area = im["width"] * im["height"]
    if min(im["width"], im["height"]) < cfg.get("min_image_side", 256):
        return False
    anns = [a for a in index.image_anns(image_id) if not a.get("iscrowd", 0)]
    if not anns:
        return False
    person = index.cat_id.get("person")
    person_area = sum(a["area"] for a in anns if a["category_id"] == person)
    return person_area / area <= cfg["max_person_area_frac"]


def sample_sources(index: CocoIndex, cfg: dict, rng: np.random.Generator) -> list[dict]:
    """Return ``cfg['n_sources']`` eligible images (sorted ids, shuffled with ``rng``)."""
    ids = sorted(i for i in index.images if is_eligible(index, i, cfg))
    rng.shuffle(ids)
    chosen = ids[: cfg["n_sources"]]
    out = []
    for i in sorted(chosen):
        im = index.images[i]
        out.append(
            {
                "source_id": i,
                "file_name": im["file_name"],
                "width": im["width"],
                "height": im["height"],
                "split": assign_split(i, cfg["split"]),
            }
        )
    return out
