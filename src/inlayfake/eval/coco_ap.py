"""COCO AP for ledger ground truth vs detector outputs (pycocotools), with grouped bootstrap."""

from __future__ import annotations

import contextlib
import io

import numpy as np

from inlayfake.ledger import ImageRecord


def build_coco(
    records: list[ImageRecord],
    dets: dict[str, list[dict]],
    categories: list[str],
    origins: set[str] | None = None,
) -> tuple[dict, list[dict]]:
    """GT and result dicts in COCO format. Instances whose origin is not in ``origins`` become
    ignore (crowd) regions, so detecting them is neither rewarded nor penalised."""
    cat_id = {c: i + 1 for i, c in enumerate(categories)}
    images, anns, results = [], [], []
    aid = 1
    for k, r in enumerate(records):
        iid = k + 1
        images.append({"id": iid, "width": r.width, "height": r.height})
        for inst in r.instances:
            if inst.category not in cat_id:
                continue
            ignore = inst.iscrowd or (origins is not None and inst.origin.value not in origins)
            x, y, w, h = inst.bbox
            anns.append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": cat_id[inst.category],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": int(ignore),
                }
            )
            aid += 1
        for reg in r.ignore_regions:
            cat, box = reg["category"], reg["bbox"]
            if cat in cat_id:
                anns.append(
                    {
                        "id": aid,
                        "image_id": iid,
                        "category_id": cat_id[cat],
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 1,
                    }
                )
                aid += 1
        for d in dets.get(r.image_id, []):
            if d["category"] in cat_id:
                results.append(
                    {
                        "image_id": iid,
                        "category_id": cat_id[d["category"]],
                        "bbox": d["bbox"],
                        "score": d["score"],
                    }
                )
    gt = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": i, "name": c} for c, i in cat_id.items()],
    }
    return gt, results


def coco_ap(gt: dict, results: list[dict]) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        if not results:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]), "AR100": float(s[8])}


def ap_with_ci(
    records: list[ImageRecord],
    dets: dict[str, list[dict]],
    categories: list[str],
    n_boot: int,
    rng: np.random.Generator,
    origins: set[str] | None = None,
) -> dict[str, float]:
    """AP point estimate plus a 95% CI from resampling source photographs."""
    out = coco_ap(*build_coco(records, dets, categories, origins))
    if n_boot <= 0:
        return out
    by_src: dict = {}
    for r in records:
        by_src.setdefault(r.source_id, []).append(r)
    keys = list(by_src)
    aps = []
    for _ in range(n_boot):
        sample = [by_src[keys[j]] for j in rng.integers(len(keys), size=len(keys))]
        # duplicate records need distinct image ids: build_coco numbers images by position
        recs = [r for group in sample for r in group]
        aps.append(coco_ap(*build_coco(recs, dets, categories, origins))["AP"])
    out["AP_lo"], out["AP_hi"] = (float(v) for v in np.percentile(aps, [2.5, 97.5]))
    return out
