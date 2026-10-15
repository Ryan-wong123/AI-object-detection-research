"""Stage `viewer`: a local web page comparing the COCO labels with every detector's boxes.

Writes runs/<run>/viewer/index.html (template: viewer_page.html) plus data.js. Open index.html
in a browser (no server needed).
For each photo it shows one panel with the ground truth (COCO labels, audit additions, ignore
regions) and one panel per detector, with a score-threshold slider, a C0/C1 switch, filters and
pagination. Images are referenced by relative path, so nothing is copied.

Box colours in detector panels follow the same matching rule as `evaluate` (same class, IoU >=
eval.iou_threshold, one-to-one, highest score first): green = matches a labelled object;
orange = matches nothing (a false positive or an object COCO did not label); grey = falls on an
ignore region.
Labelled objects that no shown box matches are drawn dashed, so misses are visible too.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from inlayfake.edit.placement import iou
from inlayfake.ledger import ImageRecord
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_records, read_jsonl, run_dir, work_dir
from inlayfake.stages.detect import load_detections

IGNORE = -2  # detection overlapping an ignore/crowd region
UNMATCHED = -1


def ground_truth(rec: ImageRecord) -> list[list]:
    """[x, y, w, h, label, kind] with kind in coco | audit | inserted | pasted | crowd | ignore."""
    out = []
    for inst in rec.instances:
        if inst.iscrowd:
            kind = "crowd"
        elif inst.instance_id.startswith("audit:"):
            kind = "audit"
        elif inst.origin.value == "ai_inserted":
            kind = "inserted"
        elif inst.origin.value == "real_pasted":
            kind = "pasted"
        else:
            kind = "coco"
        out.append([*(round(v, 1) for v in inst.bbox), inst.category, kind])
    for reg in rec.ignore_regions:
        out.append([*(round(v, 1) for v in reg["bbox"]), reg["category"], "ignore"])
    return out


def match_detections(dets: list[dict], gt: list[list], iou_thr: float, min_score: float) -> list:
    """Greedy one-to-one matching by descending score, as in COCO evaluation.

    Returns [x, y, w, h, label, score, gt_index] for detections with score >= min_score, where
    gt_index is the matched ground-truth row, UNMATCHED or IGNORE. Because matching runs from the
    highest score down, a box's status does not change when the viewer hides lower-scored boxes.
    """
    scorable = [i for i, g in enumerate(gt) if g[5] not in ("crowd", "ignore")]
    ignores = [g for g in gt if g[5] in ("crowd", "ignore")]
    taken: set[int] = set()
    out = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if d["score"] < min_score:
            break
        best, best_iou = UNMATCHED, iou_thr
        for i in scorable:
            if i in taken or gt[i][4] != d["category"]:
                continue
            o = iou(d["bbox"], gt[i][:4])
            if o >= best_iou:
                best, best_iou = i, o
        if best >= 0:
            taken.add(best)
        elif any(g[4] == d["category"] and iou(d["bbox"], g[:4]) >= iou_thr for g in ignores):
            best = IGNORE
        out.append([*(round(v, 1) for v in d["bbox"]), d["category"], round(d["score"], 3), best])
    return out


def build_data(cfg: dict, out_dir: Path) -> dict:
    recs = load_records(cfg)
    vcfg = cfg.get("viewer", {})
    min_score = vcfg.get("min_score", 0.3)
    iou_thr = cfg["eval"]["iou_threshold"]
    detectors = list(cfg["detect"]["detectors"])
    dets = {}
    for name in detectors:
        try:
            dets[name] = load_detections(cfg, name)
        except FileNotFoundError:
            continue
    detectors = [d for d in detectors if d in dets]

    plans = {}
    plan_file = work_dir(cfg) / "plans.jsonl"
    if plan_file.exists():
        plans = {p["source_id"]: p["category"] for p in read_jsonl(plan_file)}

    by_src: dict = defaultdict(dict)
    meta = {}
    root = out_dir.resolve()
    for r in recs:
        key = f"{r.condition.value}" + (f"/{r.generator}" if r.generator else "")
        gt = ground_truth(r)
        rel = os.path.relpath(Path(r.path).resolve(), root).replace(os.sep, "/")
        by_src[r.source_id][key] = {
            "src": quote(rel, safe="/._-"),
            "gt": gt,
            "det": {
                n: match_detections(dets[n].get(r.image_id, []), gt, iou_thr, min_score)
                for n in detectors
            },
        }
        meta[r.source_id] = {
            "sid": r.source_id,
            "split": r.split,
            "w": r.width,
            "h": r.height,
            "plan": plans.get(r.source_id),
        }
    images = [meta[s] | {"cond": by_src[s]} for s in sorted(by_src, key=str)]
    conditions = sorted({k for s in by_src.values() for k in s})
    return {
        "run": cfg.get("run_name"),
        "detectors": detectors,
        "conditions": conditions,
        "min_score": min_score,
        "default_threshold": vcfg.get("default_threshold", 0.5),
        "iou_threshold": iou_thr,
        "images": images,
    }


def run(cfg: dict) -> dict:
    out = run_dir(cfg) / "viewer"
    out.mkdir(parents=True, exist_ok=True)
    data = build_data(cfg, out)
    (out / "data.js").write_text(
        "window.VIEWER_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8"
    )
    page = (Path(__file__).parent / "viewer_page.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(page, encoding="utf-8")
    summary = {
        "viewer": (out / "index.html").as_posix(),
        "images": len(data["images"]),
        "detectors": data["detectors"],
        "conditions": data["conditions"],
    }
    write_run_info(run_dir(cfg), "viewer", cfg, summary)
    return summary
