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

from inlayfake.edit.compose import crop_window
from inlayfake.edit.encode import load_rgb
from inlayfake.edit.placement import intersection
from inlayfake.edit.verify import (
    best_candidate,
    build_verifier,
    changed_mask,
    mask_bbox,
    occlusion_update,
)
from inlayfake.forensics.instance_scores import bbox_mask
from inlayfake.ledger import Condition, ImageRecord, Origin, read_ledger, write_ledger
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_coco, read_jsonl, work_dir
from inlayfake.stages.replan import (
    ADOPT,
    FREE,
    adopted_c5,
    attempts_for,
    load_versions,
    merged,
    merged_pending,
    merged_source,
    merged_version,
    next_job,
    rows_at,
    track_of,
    track_version,
)

EDITED = ("C2", "C4", "C5")

AUDIT_PAGE_HEAD = (
    "<!doctype html><meta charset=utf-8><title>Edit audit</title><style>"
    "body{font:14px sans-serif;margin:16px}section{margin:0 0 24px}"
    "div{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0}"
    "img{width:min(380px,44vw);border:1px solid #ccc;display:block}"
    ".w{position:relative;display:inline-block}"
    ".b{position:absolute;border:2px solid;box-sizing:border-box;pointer-events:none}"
    ".plan{border-color:#fc0;border-style:dashed}.ins{border-color:#f22}"
    ".k{display:inline-block;width:14px;height:10px;border:2px solid;margin:0 4px}</style>"
    "<h2>Human audit of AI edits</h2><p>Each card shows the region the verifier looked at, "
    "before (C1) and after the edit. "
    '<span class="k" style="border-color:#fc0;border-style:dashed"></span>planned box (where '
    'the object was to be drawn), <span class="k" style="border-color:#f22"></span>the AI '
    "object as the verifier found it (after only). For each id, fill the same row of "
    "audit_edits.csv with y / n: <b>object_ok</b> (C4: one object of the target class is "
    "clearly there; C2: no new object appeared), <b>realistic</b> (the edited area blends into "
    "the photo), <b>other_change</b> (something outside the edited box changed). Use "
    "<b>notes</b> for anything else.</p>"
)


def _box_span(box, x0: int, y0: int, x1: int, y1: int, css: str) -> str:
    """A box [x, y, w, h] in image pixels as an overlay on the crop (x0, y0, x1, y1), in % of the
    crop so it scales with the image; clipped to the crop."""
    W, H = x1 - x0, y1 - y0
    bx0, by0 = max(box[0], x0), max(box[1], y0)
    bx1, by1 = min(box[0] + box[2], x1), min(box[1] + box[3], y1)
    if bx1 <= bx0 or by1 <= by0:
        return ""
    return (
        f'<span class="b {css}" style="left:{100 * (bx0 - x0) / W:.2f}%;'
        f"top:{100 * (by0 - y0) / H:.2f}%;width:{100 * (bx1 - bx0) / W:.2f}%;"
        f'height:{100 * (by1 - by0) / H:.2f}%"></span>'
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
    from inlayfake.stages.natural import apply_realism

    apply_realism(wd, state)  # natural placement: edits the VLM judged implausible count as failed
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


def verify_window(rec, W: int, H: int, vcfg: dict) -> list[int]:
    """Region OWLv2 and SAM look at: a square of ``verify.window_scale`` (2.5×) the planned box,
    at least ``verify.window_min_px`` (256), independent of the generation window. With the
    default generation window it equals the edit's crop box."""
    return crop_window(
        rec.edit.plan_box, W, H, vcfg.get("window_scale", 2.5), vcfg.get("window_min_px", 256)
    )


def verify_one(rec, img, ref, finder, segmenter, vcfg, index, wd) -> tuple[bool, str, dict]:
    """Returns (passed, reason, info) and, if passed, writes the annotated record."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = verify_window(rec, W, H, vcfg)
    crop = img[y0:y1, x0:x1]
    cat = rec.edit.target_category
    cands = [
        ([b[0] + x0, b[1] + y0, b[2] - b[0], b[3] - b[1]], s) for b, s in finder(crop, cat, rec)
    ]
    originals = [
        i.bbox
        for i in rec.instances
        if i.category == cat and i.origin is not Origin.AI_INSERTED and not i.iscrowd
    ]
    best = best_candidate(
        cands,
        rec.edit.plan_box,
        vcfg["min_iou"],
        vcfg["min_score"],
        vcfg.get("min_inside"),
        originals,
        vcfg.get("original_iou", 0.5),
    )
    share = vcfg.get("sham_original_overlap")
    if rec.condition.value == "C2" and best is not None and share is not None:
        # a sham repaints a box that may cover part of a real object of the class; a detection
        # lying >= `share` of its area on such an object is that real object, not a new one
        # (author, 2026-10-18: 4 shams failed every seed on the real train/umbrella/bus/book)
        new = [
            c
            for c in cands
            if all(intersection(c[0], o) < share * c[0][2] * c[0][3] for o in originals)
        ]
        best = best_candidate(
            new,
            rec.edit.plan_box,
            vcfg["min_iou"],
            vcfg["min_score"],
            vcfg.get("min_inside"),
            originals,
            vcfg.get("original_iou", 0.5),
        )
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


def build_final_ledger(
    wd: Path, generated: list[ImageRecord], state: dict, cfg: dict | None = None
) -> list[ImageRecord]:
    """C0/C1 unchanged; C3 with its pasted mask; one passed record per C2/C4/C5 job. With
    re-plans, C3 and every job count only on the source's current box version."""
    final = []
    versions = load_versions(wd)
    for r in generated:
        cond = r.condition.value
        if cond in ("C0", "C1"):
            final.append(r)
        elif cond == "C3":
            v = track_version(cfg, state, r.source_id, FREE, versions) if cfg else 0
            if r.edit.plan_version != v:
                continue  # pasted at a box that was later re-planned
            insts = [
                replace(i, mask_path=r.edit.intended_mask_path)
                if i.origin is Origin.REAL_PASTED
                else i
                for i in r.instances
            ]
            final.append(replace(r, instances=insts))
    # one passed record per job: its first passing attempt on the job's current box
    passed = set()
    is_merged = cfg is not None and merged(cfg)
    for key, rows in state.items():
        sid, cond, _ = key.split("|")
        if is_merged and cond == "C5":
            continue  # merged condition: a C5 enters only when adopted (below)
        v = track_version(cfg, state, int(sid), track_of(cond), versions) if cfg else 0
        ok = [row for row in rows_at(rows, v) if row["passed"]]
        if ok:
            passed.add(ok[0]["image_id"])
    adopted = {}
    if is_merged:  # photos whose AI edit is their verified C5: relabelled as the AI condition
        for sid in {int(k.split("|")[0]) for k in state}:
            if merged_version(cfg, state, sid, versions) == ADOPT:
                adopted[adopted_c5(cfg, state, sid)["image_id"]] = sid
        passed |= set(adopted)
    for image_id in sorted(passed):
        p = wd / "verified" / f"{image_id}.json"
        if p.exists():
            rec = ImageRecord(**json.loads(p.read_text(encoding="utf-8")))
            if is_merged and rec.condition.value in ("C4", "C5"):
                v = rec.edit.plan_version
                src = "occluding" if image_id in adopted else ("anywhere" if v > ADOPT else "free")
                rec = replace(
                    rec, condition=Condition.C4, edit=replace(rec.edit, merged_source=src)
                )
            final.append(rec)
    return final


AUDIT_COLUMNS = [
    "image_id",
    "condition",
    "generator",
    "category",
    "in_random_sample",
    "object_ok",
    "realistic",
    "other_change",
    "notes",
]


def audit_sheet(
    wd: Path, records: list[ImageRecord], fraction: float, seed: int, vcfg: dict | None = None
) -> int:
    """HTML + CSV for the human check of the edits. Never overwrites the CSV.

    A random ``fraction`` of the edits (C2, C4) is drawn with ``seed`` (``in_random_sample: y``,
    the unbiased estimate of the verifier's error rate). Conditions in
    ``verify.audit_all_conditions`` are added in full (author, 2026-10-18: every C4 is checked).
    ``verify.audit_only_conditions`` keeps only those conditions on the sheet (author,
    2026-10-18: C4 only; the random draw is unchanged, so its C4 rows stay a random sample).
    If the CSV exists, the HTML is built from its rows, so page and sheet always list the same
    edits in the same order (before, the page was redrawn with a new sample at every verify).
    """
    by_id = {r.image_id: r for r in records}
    csv_path = wd / "audit_edits.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        sample = [by_id[r["image_id"]] for r in rows if r["image_id"] in by_id]
        in_random = {r["image_id"]: r.get("in_random_sample", "y") == "y" for r in rows}
    else:
        edits = sorted(
            (r for r in records if r.condition.value in EDITED), key=lambda r: r.image_id
        )
        if not edits:
            return 0
        rng = np.random.default_rng(seed)
        k = max(1, int(round(fraction * len(edits))))
        drawn = [edits[i] for i in sorted(rng.choice(len(edits), size=k, replace=False))]
        in_random = {r.image_id: True for r in drawn}
        full = set((vcfg or {}).get("audit_all_conditions") or [])
        extra = [r for r in edits if r.condition.value in full and r.image_id not in in_random]
        only = (vcfg or {}).get("audit_only_conditions")
        if only:
            drawn = [r for r in drawn if r.condition.value in only]
        order = ["C4", "C5", "C2"]  # AI insertions first
        sample = sorted(drawn + extra, key=lambda r: (order.index(r.condition.value), r.source_id))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(AUDIT_COLUMNS)
            for r in sample:
                flag = "y" if in_random.get(r.image_id) else "n"
                w.writerow(
                    [r.image_id, r.condition.value, r.generator, r.edit.target_category, flag]
                    + [""] * 4
                )
    c1 = {r.source_id: r for r in records if r.condition.value == "C1"}
    crops = wd / "audit_crops"
    crops.mkdir(exist_ok=True)
    cards = []
    for k, r in enumerate(sample, 1):  # k = row k of audit_edits.csv (after the header)
        ref = c1.get(r.source_id)
        x0, y0, x1, y1 = verify_window(r, r.width, r.height, vcfg or {})
        figs = []
        for tag, path in (("before", ref.path if ref else r.path), ("after", r.path)):
            out = crops / f"{r.image_id}_{tag}.jpg"
            if not out.exists():
                Image.fromarray(load_rgb(path)[y0:y1, x0:x1]).save(out, quality=90)
            boxes = _box_span(r.edit.plan_box, x0, y0, x1, y1, "plan")
            if tag == "after":  # the inserted object as verified (SAM box), not on the C1 crop
                for i in r.instances:
                    if i.origin is Origin.AI_INSERTED:
                        boxes += _box_span(i.bbox, x0, y0, x1, y1, "ins")
            figs.append(
                f'<figure><span class="w"><img src="audit_crops/{html.escape(out.name)}" '
                f'loading="lazy">{boxes}</span><figcaption>{tag}</figcaption></figure>'
            )
        rand = " · random sample" if in_random.get(r.image_id) else ""
        cards.append(
            f"<section><h3>{k}. {html.escape(r.image_id)}</h3><p>{r.condition.value} · "
            f"{r.generator} · target: {html.escape(r.edit.target_category)}{rand}</p>"
            f"<div>{''.join(figs)}</div></section>"
        )
    (wd / "audit_edits.html").write_text(AUDIT_PAGE_HEAD + "".join(cards), encoding="utf-8")
    return len(sample)


def run(cfg: dict) -> dict:
    wd = work_dir(cfg)
    vcfg = cfg["verify"]
    generated = list(read_ledger(wd / "ledger_generated.jsonl"))
    state = verify_state(wd)
    checked = {row["image_id"] for rows in state.values() for row in rows}
    c1 = {r.source_id: r for r in generated if r.condition.value == "C1"}
    # attempt order per job, so that a re-verification (e.g. after a rule change) stops at the
    # first passing attempt and later attempts of that job are left out
    todo = sorted(
        (r for r in generated if r.condition.value in EDITED and r.image_id not in checked),
        key=lambda r: (job_key(r), r.edit.plan_version, r.edit.attempts),
    )

    n_verified = 0
    if todo:
        index = load_coco(cfg)
        finder, segmenter = build_verifier(vcfg, cfg["hardware"]["device"])
        with (wd / "verify.jsonl").open("a", encoding="utf-8") as f:
            for r in tqdm(todo, desc="verify"):
                done_here = rows_at(state.get(job_key(r), []), r.edit.plan_version)
                limit = attempts_for(cfg, r.condition.value, r.edit.plan_version)
                if any(x["passed"] for x in done_here) or len(done_here) >= limit:
                    continue  # this box already passed (or is out of attempts): not needed
                n_verified += 1
                img, ref = load_rgb(r.path), load_rgb(c1[r.source_id].path)
                passed, reason, info = verify_one(r, img, ref, finder, segmenter, vcfg, index, wd)
                row = {
                    "image_id": r.image_id,
                    "key": job_key(r),
                    "condition": r.condition.value,
                    "generator": r.generator,
                    "attempt": r.edit.attempts,
                    "version": r.edit.plan_version,
                    "passed": passed,
                    "reason": reason,
                } | info
                f.write(json.dumps(row) + "\n")
                f.flush()  # an interrupted run keeps every finished row
                state[row["key"]].append(row)
        del finder, segmenter
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    realism = {}
    from inlayfake.stages.natural import enabled as natural_enabled

    if natural_enabled(cfg) and cfg["placement"]["natural"].get("check", True):
        from inlayfake.stages.natural import realism_pass

        realism = realism_pass(cfg, wd, state)  # after OWLv2/SAM are off the GPU

    final = build_final_ledger(wd, generated, state, cfg)
    write_ledger(final, wd / "ledger.jsonl")
    n_audit = audit_sheet(wd, final, vcfg.get("audit_fraction", 0.1), cfg["seed"] + 7, vcfg)

    versions = load_versions(wd)
    by_cond: dict[str, Counter] = defaultdict(Counter)
    for key, rows in state.items():
        sid, cond, gen = key.split("|")
        v = track_version(cfg, state, int(sid), track_of(cond), versions)
        here = rows_at(rows, v)
        if any(r["passed"] for r in here):
            by_cond[cond]["passed"] += 1
            by_cond[cond][f"passed_on_attempt_{len(rows)}"] += 1  # total tries, all boxes
            if cfg["edit"].get("max_replans", 0):
                by_cond[cond][f"passed_on_box_{v}"] += 1  # 0 = original plan box
        elif next_job(cfg, state, int(sid), cond, gen, versions) is None:
            by_cond[cond]["dropped"] += 1
        else:
            by_cond[cond]["needs_retry"] += 1
    summary = {
        "verified_now": n_verified,
        "final_ledger_records": len(final),
        "by_condition": {c: dict(v) for c, v in sorted(by_cond.items())},
        "needs_retry": sum(v["needs_retry"] for v in by_cond.values()),
        "audit_sample": n_audit,
    }
    if realism:
        summary["realism_check"] = realism
    if merged(cfg):  # one AI condition per photo; C2 may wait for its AI edit (no rows yet)
        plans = read_jsonl(wd / "plans.jsonl")
        src = Counter(
            merged_source(cfg, state, p["source_id"], versions) or "none yet" for p in plans
        )
        summary["merged_ai_source"] = dict(sorted(src.items()))
        summary["needs_retry"] = merged_pending(cfg, state, plans, versions)
    write_run_info(wd / "runs", "verify", cfg, summary)
    return summary
