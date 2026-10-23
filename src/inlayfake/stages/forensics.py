"""Stage `forensics`: hand-crafted forensic maps → image and per-instance scores.

Output: runs/<run>/forensics/<method>.jsonl, one row per (image, level, instance). Maps are not
stored (too large); set ``forensics.save_maps_for`` to a list of image ids to keep a few as .npy
for figures. Resumable per method.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image
from tqdm import tqdm

from inlayfake.edit.encode import load_rgb
from inlayfake.forensics.handcrafted import METHODS
from inlayfake.forensics.instance_scores import bbox_mask, image_score, instance_score
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import done_ids, load_coco, load_records, open_append, run_dir


def _instance_masks(rec, index, shape) -> dict[str, np.ndarray]:
    masks = {}
    for inst in rec.instances:
        if inst.iscrowd:
            continue
        if inst.mask_path:
            masks[inst.instance_id] = np.asarray(Image.open(inst.mask_path)) > 127
        elif inst.instance_id.startswith("coco:"):
            ann = index.anns[int(inst.instance_id.split(":", 1)[1])]
            masks[inst.instance_id] = index.ann_mask(ann)
        else:
            masks[inst.instance_id] = bbox_mask(inst.bbox, shape)
    return masks


def run(cfg: dict) -> dict:
    recs = load_records(cfg)
    index = load_coco(cfg)
    rd = run_dir(cfg) / "forensics"
    rd.mkdir(parents=True, exist_ok=True)
    methods = cfg["forensics"]["handcrafted"]
    ring = cfg["forensics"]["context_ring_frac"]
    keep_maps = set(cfg["forensics"].get("save_maps_for") or [])
    done = {m: done_ids(rd / f"{m}.jsonl") for m in methods}
    todo = [r for r in recs if any(r.image_id not in done[m] for m in methods)]
    files = {m: open_append(rd / f"{m}.jsonl") for m in methods}
    try:
        for rec in tqdm(todo, desc="forensics"):
            img = load_rgb(rec.path)
            masks = _instance_masks(rec, index, img.shape[:2])
            base = {
                "image_id": rec.image_id,
                "source_id": rec.source_id,
                "split": rec.split,
                "condition": rec.condition.value,
                "generator": rec.generator,
            }
            for m in methods:
                if rec.image_id in done[m]:
                    continue
                fmap = METHODS[m](img)
                if rec.image_id in keep_maps:
                    np.save(rd / f"{m}__{rec.image_id}.npy", fmap.astype(np.float16))
                rows = [base | {"level": "image", "scores": image_score(fmap)}]
                for inst in rec.instances:
                    if inst.instance_id not in masks:
                        continue
                    mk = masks[inst.instance_id]
                    rows.append(
                        base
                        | {
                            "level": "instance",
                            "instance_id": inst.instance_id,
                            "category": inst.category,
                            "origin": inst.origin.value,
                            "area": int(mk.sum()),
                            "visible_fraction": inst.visible_fraction,
                            "scores": instance_score(fmap, mk, inst.bbox, ring),
                        }
                    )
                files[m].write("".join(json.dumps(r) + "\n" for r in rows))
    finally:
        for f in files.values():
            f.close()
    write_run_info(run_dir(cfg), "forensics", cfg, {"images": len(todo), "methods": methods})
    return {"images_processed": len(todo), "methods": methods}
