"""Stage `forensics`: forensic maps and scores → image and per-instance scores.

Output: runs/<run>/forensics/<method>.jsonl, one row per (image, level, instance). Maps are not
stored (too large); set ``forensics.save_maps_for`` to a list of image ids to keep a few as .npy
for figures. Resumable per method.

Hand-crafted methods (``forensics.handcrafted``) share one pass over the images. Learned methods
(``forensics.run_learned``, Task 3: trufor, univfd) get one pass each, so only one model is on the
GPU at a time:
  trufor  pixel map → the same image / instance scores as the hand-crafted maps, plus the
          model's own image score ``det``
  univfd  image score (``prob``, ``logit``) and per object the score of its crop (``crop_prob``,
          ``crop_logit``) and ``contrast`` = crop logit − image logit. The 768-d CLIP features of
          the image and of every crop are kept in forensics/univfd_features/<image_id>.npz for the
          crop classifier (L3, week 4).
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image
from tqdm import tqdm

from inlayfake.edit.encode import load_rgb
from inlayfake.forensics.handcrafted import METHODS
from inlayfake.forensics.instance_scores import bbox_mask, image_score, instance_score
from inlayfake.forensics.learned import build_learned, clip_view, object_crop, sigmoid
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


def _base(rec) -> dict:
    return {
        "image_id": rec.image_id,
        "source_id": rec.source_id,
        "split": rec.split,
        "condition": rec.condition.value,
        "generator": rec.generator,
    }


def _instance_row(base: dict, inst, area: int, scores: dict) -> dict:
    return base | {
        "level": "instance",
        "instance_id": inst.instance_id,
        "category": inst.category,
        "origin": inst.origin.value,
        "area": area,
        "visible_fraction": inst.visible_fraction,
        "scores": scores,
    }


def _map_rows(base, rec, fmap, masks, ring, extra=None) -> list[dict]:
    rows = [base | {"level": "image", "scores": image_score(fmap) | (extra or {})}]
    for inst in rec.instances:
        if inst.instance_id in masks:
            mk = masks[inst.instance_id]
            scores = instance_score(fmap, mk, inst.bbox, ring)
            rows.append(_instance_row(base, inst, int(mk.sum()), scores))
    return rows


def _univfd_rows(model, base, rec, img, masks, expand, feat_dir) -> list[dict]:
    insts = [i for i in rec.instances if i.instance_id in masks]
    views = [clip_view(img)] + [object_crop(img, i.bbox, expand) for i in insts]
    feats = model.features(views)
    logits = model.logits(feats)
    np.savez(  # before the JSONL row, so a written row always has its features
        feat_dir / f"{rec.image_id}.npz",
        ids=np.array(["image"] + [i.instance_id for i in insts]),
        features=feats.astype(np.float16),
    )
    img_logit = float(logits[0])
    rows = [
        base | {"level": "image", "scores": {"prob": float(sigmoid(img_logit)), "logit": img_logit}}
    ]
    for inst, lg in zip(insts, logits[1:], strict=True):
        scores = {
            "crop_prob": float(sigmoid(lg)),
            "crop_logit": float(lg),
            "contrast": float(lg) - img_logit,
        }
        rows.append(_instance_row(base, inst, int(masks[inst.instance_id].sum()), scores))
    return rows


def _run_learned(cfg, recs, index, rd) -> dict[str, int]:
    fcfg = cfg["forensics"]
    ring, expand = fcfg["context_ring_frac"], fcfg["crop_expand"]
    processed = {}
    for name in fcfg.get("run_learned") or []:
        out = rd / f"{name}.jsonl"
        done = done_ids(out)
        todo = [r for r in recs if r.image_id not in done]
        processed[name] = len(todo)
        if not todo:
            continue
        model = build_learned(name, cfg)
        feat_dir = rd / "univfd_features"
        if name == "univfd":
            feat_dir.mkdir(exist_ok=True)
        with open_append(out) as f:
            for rec in tqdm(todo, desc=name):
                img = load_rgb(rec.path)
                masks = _instance_masks(rec, index, img.shape[:2])
                base = _base(rec)
                if name == "trufor":
                    fmap, det = model(img)
                    rows = _map_rows(base, rec, fmap, masks, ring, {"det": det})
                else:
                    rows = _univfd_rows(model, base, rec, img, masks, expand, feat_dir)
                f.write("".join(json.dumps(r) + "\n" for r in rows))
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    return processed


def _run_edit_boxes(cfg, recs, rd) -> dict[str, int]:
    """Scores of each method inside the planned edit box of every edited image (C2/C3/C4) →
    <method>__editbox.jsonl, level ``edit_box``. The C2 box holds no object, only repainted
    background: the share of C2 boxes a method flags is the edited-background flag rate (§6), which
    tells detecting AI objects apart from detecting any regenerated region. One pass per method,
    resumable; maps are recomputed (they are not stored)."""
    fcfg = cfg["forensics"]
    ring, expand = fcfg["context_ring_frac"], fcfg["crop_expand"]
    edited = [r for r in recs if r.edit is not None]
    processed = {}
    for name in list(fcfg["handcrafted"]) + list(fcfg.get("run_learned") or []):
        out = rd / f"{name}__editbox.jsonl"
        done = done_ids(out)
        todo = [r for r in edited if r.image_id not in done]
        processed[name] = len(todo)
        if not todo:
            continue
        model = None if name in METHODS else build_learned(name, cfg)
        with open_append(out) as f:
            for rec in tqdm(todo, desc=f"{name} edit box"):
                img = load_rgb(rec.path)
                box = rec.edit.plan_box
                if model is None or name == "trufor":
                    fmap = METHODS[name](img) if model is None else model(img)[0]
                    mask = bbox_mask(box, img.shape[:2])
                    scores = instance_score(fmap, mask, box, ring)
                else:
                    feats = model.features([clip_view(img), object_crop(img, box, expand)])
                    img_logit, lg = (float(v) for v in model.logits(feats))
                    scores = {
                        "crop_prob": float(sigmoid(lg)),
                        "crop_logit": lg,
                        "contrast": lg - img_logit,
                    }
                row = _base(rec) | {"level": "edit_box", "box": box, "scores": scores}
                f.write(json.dumps(row) + "\n")
        del model
    return processed


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
            base = _base(rec)
            for m in methods:
                if rec.image_id in done[m]:
                    continue
                fmap = METHODS[m](img)
                if rec.image_id in keep_maps:
                    np.save(rd / f"{m}__{rec.image_id}.npy", fmap.astype(np.float16))
                rows = _map_rows(base, rec, fmap, masks, ring)
                files[m].write("".join(json.dumps(r) + "\n" for r in rows))
    finally:
        for f in files.values():
            f.close()
    learned = _run_learned(cfg, recs, index, rd)
    boxes = _run_edit_boxes(cfg, recs, rd) if cfg["forensics"].get("edit_box_rows") else {}
    write_run_info(
        run_dir(cfg),
        "forensics",
        cfg,
        {
            "images": len(todo),
            "methods": methods,
            "learned_images": learned,
            "edit_box_images": boxes,
        },
    )
    return {
        "images_processed": len(todo),
        "methods": methods,
        "learned_processed": learned,
        "edit_box_processed": boxes,
    }
