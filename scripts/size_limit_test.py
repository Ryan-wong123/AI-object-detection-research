"""Which box sizes survive generation + verification? (author request, 2026-10-21)

The box-mode shape check rejects boxes under 32 px (short side) / 0.5% of the image as too small
for SDXL and the verifier; that limit was an estimate. This test takes Qwen's real boxes from the
no-rules pilot (data/inlayfake_pilot_natural_qwen_free), groups them by short side, generates the
AI object with the pipeline's own C4 code (SDXL inpainting, pilot edit settings, 3 seeds) and
checks it with the pipeline's own C4 verification (OWLv2 + SAM, pilot verify settings). A box
passes if any seed passes, as in the pipeline.

    python scripts/size_limit_test.py            # resumable; writes <work>/size_test/
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from inlayfake.config import load_config
from inlayfake.edit.compose import crop_window, local_edit
from inlayfake.edit.encode import load_rgb, save_policy
from inlayfake.edit.placement import box_area, intersection
from inlayfake.edit.verify import best_candidate, mask_bbox
from inlayfake.forensics.instance_scores import bbox_mask
from inlayfake.stages.common import load_coco
from inlayfake.stages.spots import spot_id

CFG = "configs/pilot_natural_qwen_free.yaml"
BANDS = [(10, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 40), (40, 48)]
PER_BAND = 6
SEEDS = (11, 22, 33)
WIDE = {153229, 273715, 408774, 428218, 441286}  # the 5 wide sports photos without a box


def pick_boxes(wd: Path, index) -> list[dict]:
    rows = []
    for name in ("natural_candidates.jsonl", "natural_candidates_no_shape.jsonl"):
        p = wd / name
        if p.exists():
            rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    seen, cands = set(), []
    for r in rows:
        if not r.get("box") or r.get("category") is None:
            continue
        k = spot_id(r)
        if k in seen:
            continue
        seen.add(k)
        side = min(r["box"][2:])
        if not 10 <= side < 48:
            continue
        # leave out boxes on an existing object of the same class: the verifier ignores those
        # whatever their size, which would confound the size test
        on_own = any(
            index.cat_name[a["category_id"]] == r["category"]
            and intersection(r["box"], a["bbox"]) > 0.5 * box_area(a["bbox"])
            for a in index.image_anns(r["source_id"])
        )
        if not on_own:
            cands.append(r | {"side": side})
    picked = []
    for lo, hi in BANDS:
        band = [r for r in cands if lo <= r["side"] < hi]
        band.sort(key=lambda r: (r["source_id"] not in WIDE, r["source_id"], r["side"]))
        used, out = set(), []
        for r in band:  # wide photos first, then one box per photo
            if r["source_id"] in used and r["source_id"] not in WIDE:
                continue
            used.add(r["source_id"])
            out.append(r)
            if len(out) == PER_BAND:
                break
        picked += [r | {"band": f"{lo}-{hi}"} for r in out]
    return picked


def main() -> int:
    cfg = load_config(CFG)
    index = load_coco(cfg)
    wd = Path(cfg["paths"]["work"])
    out = wd / "size_test"
    (out / "images").mkdir(parents=True, exist_ok=True)
    boxes_path = out / "boxes.json"
    if boxes_path.exists():
        boxes = json.loads(boxes_path.read_text())
    else:
        boxes = pick_boxes(wd, index)
        boxes_path.write_text(json.dumps(boxes, indent=1))
    print(
        f"{len(boxes)} boxes:",
        dict(
            sorted({b["band"]: sum(x["band"] == b["band"] for x in boxes) for b in boxes}.items())
        ),
        flush=True,
    )
    ecfg, vcfg, enc = cfg["edit"], cfg["verify"], cfg["encoding"]

    # --- phase 1: generate (SDXL only on the GPU)
    todo = [
        (i, s)
        for i in range(len(boxes))
        for s in SEEDS
        if not (out / "images" / f"{i}_{s}.jpg").exists()
    ]
    if todo:
        from inlayfake.edit.generators import build_inpainter

        inp = build_inpainter(cfg["generators"]["G1"], ecfg, cfg["hardware"])
        for n, (i, s) in enumerate(todo, 1):
            b = boxes[i]
            img = load_rgb(index.image_path(b["source_id"]))
            res, intended, _ = local_edit(
                img,
                b["box"],
                inp,
                inp.resolution,
                ecfg["prompt"].format(category=b["category"]),
                ecfg["negative_prompt"],
                s,
                ecfg["window_scale"],
                ecfg["mask_dilate_px"],
                ecfg["feather_px"],
                ecfg.get("composite", "mask"),
                ecfg.get("window_min_px", 256),
            )
            Image.fromarray((intended * 255).astype(np.uint8)).save(
                out / "images" / f"{i}_{s}_mask.png"
            )
            save_policy(res, out / "images" / f"{i}_{s}.jpg", enc)
            print(f"generated {n}/{len(todo)}", flush=True)
        del inp
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    # --- phase 2: verify with the pipeline's C4 rules (OWLv2 + SAM)
    from inlayfake.edit.verify import build_verifier

    finder, segmenter = build_verifier(vcfg, cfg["hardware"]["device"])
    results = []
    for i, b in enumerate(boxes):
        sid, cat, box = b["source_id"], b["category"], b["box"]
        originals = [
            a["bbox"]
            for a in index.image_anns(sid)
            if index.cat_name[a["category_id"]] == cat and not a.get("iscrowd", 0)
        ]
        seeds = []
        for s in SEEDS:
            img = load_rgb(out / "images" / f"{i}_{s}.jpg")
            H, W = img.shape[:2]
            x0, y0, x1, y1 = crop_window(
                box, W, H, vcfg.get("window_scale", 2.5), vcfg.get("window_min_px", 256)
            )
            crop = img[y0:y1, x0:x1]
            cands = [
                ([c[0] + x0, c[1] + y0, c[2] - c[0], c[3] - c[1]], sc)
                for c, sc in finder(crop, cat)
            ]
            best = best_candidate(
                cands,
                box,
                vcfg["min_iou"],
                vcfg["min_score"],
                vcfg.get("min_inside"),
                originals,
                vcfg.get("original_iou", 0.5),
            )
            if best is None:
                seeds.append({"seed": s, "passed": False, "reason": "not found"})
                continue
            intended = np.asarray(Image.open(out / "images" / f"{i}_{s}_mask.png")) > 127
            bx, by, bw, bh = best[0]
            m = np.zeros((H, W), dtype=bool)
            m[y0:y1, x0:x1] = segmenter(crop, [bx - x0, by - y0, bx - x0 + bw, by - y0 + bh])
            m &= intended
            if m.sum() < 0.05 * bw * bh:
                m = bbox_mask(best[0], (H, W)) & intended
            if not m.any():
                seeds.append({"seed": s, "passed": False, "reason": "empty mask"})
                continue
            ox, oy, ow, oh = mask_bbox(m)
            fill = ow * oh / max(box[2] * box[3], 1e-6)
            ok = fill >= vcfg.get("min_box_fill", 0) if vcfg.get("min_box_fill") else True
            seeds.append(
                {
                    "seed": s,
                    "passed": bool(ok),
                    "score": round(best[1], 3),
                    "fill": round(fill, 3),
                    "reason": "ok" if ok else "fill too low",
                }
            )
        results.append(b | {"seeds": seeds, "passed": any(x["passed"] for x in seeds)})
        print(f"verified {i + 1}/{len(boxes)}", flush=True)

    sizes = {
        x["source_id"]: x["width"] * x["height"]
        for x in map(json.loads, (wd / "sources.jsonl").read_text().splitlines())
    }
    by = defaultdict(list)
    for r in results:
        by[r["band"]].append(r)
    summary = {
        band: {
            "boxes": len(rs),
            "box_passed_any_seed": sum(r["passed"] for r in rs),
            "seed_pass_rate": round(
                sum(x["passed"] for r in rs for x in r["seeds"]) / (3 * len(rs)), 3
            ),
            "median_area_pct": round(
                float(
                    np.median([100 * r["box"][2] * r["box"][3] / sizes[r["source_id"]] for r in rs])
                ),
                2,
            ),
        }
        for band, rs in sorted(by.items(), key=lambda kv: int(kv[0].split("-")[0]))
    }
    wide = [
        {
            "source_id": r["source_id"],
            "category": r["category"],
            "side": round(r["side"], 1),
            "passed": r["passed"],
        }
        for r in results
        if r["source_id"] in WIDE
    ]
    (out / "results.json").write_text(
        json.dumps({"bands": summary, "wide_photos": wide, "boxes": results}, indent=1)
    )
    print(json.dumps({"bands": summary, "wide_photos": wide}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
