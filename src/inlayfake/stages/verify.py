"""Stage `verify`: check every AI edit, annotate it and write the final ledger (ARCHITECTURE §3.5).

For each generated C2/C4/C5 image (all attempts in <work>/ledger_generated.jsonl):
  * C4/C5: OWLv2 (an independent verifier, not one of the evaluated detectors) must find the target
    category in the crop window with IoU >= verify.min_iou to the planned box and score >=
    verify.min_score. SAM, prompted with that box and clipped to the intended edit mask, gives the
    inserted object's mask and tight box. Original objects the insertion covers get their visible
    mask, box and visible_fraction updated; below 10% visible they become ignore regions.
  * C2 (sham): passes only if OWLv2 does NOT find the target category there (no object appeared).
  * all: the actual changed region |edited − C1| > verify.changed_pixel_threshold is saved, and an
    original object whose remaining area overlaps it by > verify.mixed_modified_overlap becomes
    origin "mixed_modified" (the editor altered a real object).

Outputs (under <work>):
  verify.jsonl              one row per verified image (passed / reason / score); `generate` reads
                            it and regenerates failed edits with a new seed (max edit.max_attempts)
  verified/<image_id>.json  annotated records that passed
  masks/<id>_actual.png, <id>_inserted.png, <id>_<instance>_visible.png
  ledger.jsonl              FINAL ledger: C0/C1/C3 unchanged + passed C2/C4/C5 (read by later
                            stages)
  audit_edits.html / .csv   random sample (verify.audit_fraction) for the human check
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from inlayfake.edit.encode import load_rgb
from inlayfake.edit.verify import (
    best_candidate,
    build_verifier,
    changed_mask,
    mask_bbox,
    occlusion_update,
)
from inlayfake.forensics.instance_scores import bbox_mask
from inlayfake.ledger import ImageRecord, Origin, read_ledger, write_ledger
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_coco, read_jsonl, work_dir

EDITED = ("C2", "C4", "C5")

AUDIT_PAGE_HEAD = (
    "<!doctype html><meta charset=utf-8><title>Edit audit</title><style>"
    "body{font:14px sans-serif;margin:16px}section{margin:0 0 24px}"
    "div{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0}"
    "img{width:min(380px,44vw);border:1px solid #ccc}</style>"
    "<h2>Human audit of AI edits</h2><p>For each id, fill audit_edits.csv with y / n: "
    "<b>object_ok</b> (C4/C5: the target object is clearly there; C2: nothing new appeared), "
    "<b>realistic</b> (it blends into the photo), <b>other_change</b> (anything outside the "
    "edited box changed).</p>"
)


def job_key(rec: ImageRecord) -> str:
    return f"{rec.source_id}|{rec.condition.value}|{rec.generator}"


def verify_state(wd: Path) -> dict[str, list[dict]]:
    """key -> verification rows (oldest first); used by `generate` to schedule retries."""
    path = wd / "verify.jsonl"
    state: dict[str, list[dict]] = defaultdict(list)
    if path.exists():
        for row in read_jsonl(path):
            state[row["key"]].append(row)
    return state


def next_attempt(state: dict, key: str, max_attempts: int) -> int | None:
    """Attempt index to generate for a job, or None if it passed or ran out of attempts."""
    rows = state.get(key, [])
    if any(r["passed"] for r in rows) or len(rows) >= max_attempts:
        return None
    return len(rows)


def _save_mask(mask: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)
    return path.as_posix()


def _original_mask(inst, index, shape) -> np.ndarray:
    if inst.mask_path:
        return np.asarray(Image.open(inst.mask_path)) > 127
    if inst.instance_id.startswith("coco:"):
        return index.ann_mask(index.anns[int(inst.instance_id.split(":", 1)[1])])
    return bbox_mask(inst.bbox, shape)


def verify_one(rec, img, ref, finder, segmenter, vcfg, index, wd) -> tuple[bool, str, dict]:
    """Returns (passed, reason, info) and, if passed, writes the annotated record."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in rec.edit.crop_box)
    crop = img[y0:y1, x0:x1]
    cat = rec.edit.target_category
    cands = [
        ([b[0] + x0, b[1] + y0, b[2] - b[0], b[3] - b[1]], s) for b, s in finder(crop, cat, rec)
    ]
    best = best_candidate(cands, rec.edit.plan_box, vcfg["min_iou"], vcfg["min_score"])
    info = {"n_candidates": len(cands), "score": best[1] if best else None}

    actual = changed_mask(img, ref, vcfg["changed_pixel_threshold"])
    actual_path = _save_mask(actual, wd / "masks" / f"{rec.image_id}_actual.png")
    info["changed_pixels"] = int(actual.sum())

    cond = rec.condition.value
    if cond == "C2":
        if best is not None:
            return False, "sham edit produced the target object", info
        inserted = np.zeros((H, W), dtype=bool)
    else:
        if best is None:
            return False, "inserted object not found by verifier", info
        intended = np.asarray(Image.open(rec.edit.intended_mask_path)) > 127
        bx, by, bw, bh = best[0]
        box_crop = [bx - x0, by - y0, bx - x0 + bw, by - y0 + bh]
        m = np.zeros((H, W), dtype=bool)
        m[y0:y1, x0:x1] = segmenter(crop, box_crop)
        m &= intended
        if m.sum() < 0.05 * bw * bh:  # SAM failed: fall back to the verifier box inside the mask
            m = bbox_mask(best[0], (H, W)) & intended
        if not m.any():
            return False, "empty inserted mask", info
        inserted = m
        info["inserted_area"] = int(m.sum())

    insts, ignores = [], list(rec.ignore_regions)
    for inst in rec.instances:
        if inst.origin is Origin.AI_INSERTED:
            ins_path = _save_mask(inserted, wd / "masks" / f"{rec.image_id}_inserted.png")
            insts.append(replace(inst, bbox=mask_bbox(inserted), mask_path=ins_path))
            continue
        if inst.iscrowd:
            insts.append(inst)
            continue
        orig = _original_mask(inst, index, (H, W))
        new = inst
        if (orig & inserted).any():  # occluded by the insertion (C5, or rare spill-over)
            visible, frac = occlusion_update(orig, inserted)
            if frac < vcfg.get("min_visible_fraction", 0.1):
                ignores.append({"category": inst.category, "bbox": inst.bbox})
                continue
            tag = inst.instance_id.replace(":", "-")
            vpath = _save_mask(visible, wd / "masks" / f"{rec.image_id}_{tag}_visible.png")
            new = replace(
                inst, bbox=mask_bbox(visible), visible_fraction=round(frac, 4), mask_path=vpath
            )
            orig = visible
        area = orig.sum()
        if area and (orig & actual).sum() > vcfg["mixed_modified_overlap"] * area:
            new = replace(new, origin=Origin.MIXED_MODIFIED)
        insts.append(new)

    edit = replace(rec.edit, actual_mask_path=actual_path, verifier_score=info["score"])
    out = replace(rec, instances=insts, ignore_regions=ignores, edit=edit)
    p = wd / "verified" / f"{rec.image_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out.to_dict(), indent=1), encoding="utf-8")
    return True, "ok", info


def build_final_ledger(wd: Path, generated: list[ImageRecord], state: dict) -> list[ImageRecord]:
    """C0/C1 unchanged; C3 with its pasted mask; one passed record per C2/C4/C5 job."""
    final = []
    for r in generated:
        cond = r.condition.value
        if cond in ("C0", "C1"):
            final.append(r)
        elif cond == "C3":
            insts = [
                replace(i, mask_path=r.edit.intended_mask_path)
                if i.origin is Origin.REAL_PASTED
                else i
                for i in r.instances
            ]
            final.append(replace(r, instances=insts))
    passed = {row["image_id"] for rows in state.values() for row in rows if row["passed"]}
    for image_id in sorted(passed):
        p = wd / "verified" / f"{image_id}.json"
        if p.exists():
            final.append(ImageRecord(**json.loads(p.read_text(encoding="utf-8"))))
    return final


def audit_sheet(wd: Path, records: list[ImageRecord], fraction: float, seed: int) -> int:
    """HTML + CSV for the human check of a random sample of edits. Never overwrites the CSV."""
    edits = sorted((r for r in records if r.condition.value in EDITED), key=lambda r: r.image_id)
    if not edits:
        return 0
    rng = np.random.default_rng(seed)
    k = max(1, int(round(fraction * len(edits))))
    sample = [edits[i] for i in sorted(rng.choice(len(edits), size=k, replace=False))]
    c1 = {r.source_id: r for r in records if r.condition.value == "C1"}
    crops = wd / "audit_crops"
    crops.mkdir(exist_ok=True)
    cards = []
    for r in sample:
        ref = c1.get(r.source_id)
        x0, y0, x1, y1 = (int(round(v)) for v in r.edit.crop_box)
        figs = []
        for tag, path in (("before", ref.path if ref else r.path), ("after", r.path)):
            out = crops / f"{r.image_id}_{tag}.jpg"
            if not out.exists():
                Image.fromarray(load_rgb(path)[y0:y1, x0:x1]).save(out, quality=90)
            figs.append(
                f'<figure><img src="audit_crops/{html.escape(out.name)}"><figcaption>{tag}'
                f"</figcaption></figure>"
            )
        cards.append(
            f"<section><h3>{html.escape(r.image_id)}</h3><p>{r.condition.value} · {r.generator} · "
            f"target: {html.escape(r.edit.target_category)}</p><div>{''.join(figs)}</div></section>"
        )
    (wd / "audit_edits.html").write_text(
        AUDIT_PAGE_HEAD + "".join(cards),
        encoding="utf-8",
    )
    csv_path = wd / "audit_edits.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "image_id",
                    "condition",
                    "generator",
                    "category",
                    "object_ok",
                    "realistic",
                    "other_change",
                    "notes",
                ]
            )
            for r in sample:
                w.writerow(
                    [
                        r.image_id,
                        r.condition.value,
                        r.generator,
                        r.edit.target_category,
                        "",
                        "",
                        "",
                        "",
                    ]
                )
    return len(sample)


def run(cfg: dict) -> dict:
    wd = work_dir(cfg)
    vcfg = cfg["verify"]
    max_attempts = cfg["edit"]["max_attempts"]
    generated = list(read_ledger(wd / "ledger_generated.jsonl"))
    state = verify_state(wd)
    checked = {row["image_id"] for rows in state.values() for row in rows}
    c1 = {r.source_id: r for r in generated if r.condition.value == "C1"}
    todo = [r for r in generated if r.condition.value in EDITED and r.image_id not in checked]

    if todo:
        index = load_coco(cfg)
        finder, segmenter = build_verifier(vcfg, cfg["hardware"]["device"])
        with (wd / "verify.jsonl").open("a", encoding="utf-8") as f:
            for r in tqdm(todo, desc="verify"):
                img, ref = load_rgb(r.path), load_rgb(c1[r.source_id].path)
                passed, reason, info = verify_one(r, img, ref, finder, segmenter, vcfg, index, wd)
                row = {
                    "image_id": r.image_id,
                    "key": job_key(r),
                    "condition": r.condition.value,
                    "generator": r.generator,
                    "attempt": r.edit.attempts,
                    "passed": passed,
                    "reason": reason,
                } | info
                f.write(json.dumps(row) + "\n")
                state[row["key"]].append(row)
        del finder, segmenter
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    final = build_final_ledger(wd, generated, state)
    write_ledger(final, wd / "ledger.jsonl")
    n_audit = audit_sheet(wd, final, vcfg.get("audit_fraction", 0.1), cfg["seed"] + 7)

    by_cond: dict[str, Counter] = defaultdict(Counter)
    for rows in state.values():
        cond = rows[-1]["condition"]
        if any(r["passed"] for r in rows):
            by_cond[cond]["passed"] += 1
            by_cond[cond][f"passed_on_attempt_{len(rows)}"] += 1
        elif len(rows) >= max_attempts:
            by_cond[cond]["dropped"] += 1
        else:
            by_cond[cond]["needs_retry"] += 1
    summary = {
        "verified_now": len(todo),
        "final_ledger_records": len(final),
        "by_condition": {c: dict(v) for c, v in sorted(by_cond.items())},
        "needs_retry": sum(v["needs_retry"] for v in by_cond.values()),
        "audit_sample": n_audit,
    }
    write_run_info(wd / "runs", "verify", cfg, summary)
    return summary
