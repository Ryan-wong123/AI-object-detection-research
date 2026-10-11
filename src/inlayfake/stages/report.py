"""Stage `report`: a per-image table of the photos a run used and what each detector found.

Writes, for the original photos (C0) of one run:
  runs/<run>/tables/images_used.csv
  runs/<run>/tables/images_used.md     a Markdown table to paste into briefings

One row per source photo: split, the category planned for insertion, the labelled objects (COCO
labels plus audit additions, crowd boxes excluded) and, per detector, how many of those objects it
found. "Found" uses the same rule as `evaluate`'s recall: a same-class prediction with
IoU >= eval.iou_threshold and score >= eval.recall_score.
"""

from __future__ import annotations

import csv
from collections import Counter

from inlayfake.ledger import ImageRecord
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_records, read_jsonl, run_dir, work_dir
from inlayfake.stages.detect import load_detections
from inlayfake.stages.evaluate import detected


def coco_url(source_id: int | str) -> str:
    return f"http://images.cocodataset.org/val2017/{int(source_id):012d}.jpg"


def label_summary(categories: list[str], top: int = 3) -> str:
    """'person ×3, cup, dog' for the `top` most common labels, ties broken alphabetically."""
    counts = sorted(Counter(categories).items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return ", ".join(f"{c} ×{n}" if n > 1 else c for c, n in counts)


def image_rows(
    records: list[ImageRecord],
    dets_by_detector: dict[str, dict[str, list[dict]]],
    planned: dict[str, str],
    iou_thr: float,
    score_thr: float,
) -> list[dict]:
    """One row per record (sorted by source id) with per-detector found counts."""
    rows = []
    for r in sorted(records, key=lambda r: int(r.source_id)):
        objs = [i for i in r.instances if not i.iscrowd]
        row = {
            "source_id": r.source_id,
            "split": r.split,
            "planned_insert": planned.get(str(r.source_id), ""),
            "n_objects": len(objs),
            "labels": label_summary([i.category for i in objs]),
        }
        for name, dets in dets_by_detector.items():
            d = dets.get(r.image_id, [])
            row[f"found_{name}"] = sum(detected(i, d, iou_thr, score_thr) for i in objs)
        rows.append(row)
    return rows


def rows_markdown(rows: list[dict], detectors: list[str], labels: dict[str, str]) -> str:
    head = ["COCO image", "Split", "Planned insert", "Labelled objects", "Most common labels"]
    head += [labels.get(n, n) for n in detectors]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [
            f"[{r['source_id']}]({coco_url(r['source_id'])})",
            r["split"],
            r["planned_insert"],
            str(r["n_objects"]),
            r["labels"],
        ] + [f"{r[f'found_{n}']}/{r['n_objects']}" for n in detectors]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


DETECTOR_LABELS = {
    "faster_rcnn": "Faster R-CNN",
    "yolo": "YOLO11-m",
    "rtdetr": "RT-DETR",
    "grounding_dino": "Grounding DINO",
}


def run(cfg: dict) -> dict:
    ecfg = cfg["eval"]
    detectors = list(cfg["detect"]["detectors"])
    c0 = [r for r in load_records(cfg) if r.condition.value == "C0"]
    planned = {
        str(p["source_id"]): p["category"] for p in read_jsonl(work_dir(cfg) / "plans.jsonl")
    }
    dets = {n: load_detections(cfg, n) for n in detectors}
    rows = image_rows(c0, dets, planned, ecfg["iou_threshold"], ecfg["recall_score"])
    out = run_dir(cfg) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "images_used.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["n_objects"] for r in rows)
    found = {n: sum(r[f"found_{n}"] for r in rows) for n in detectors}
    intro = (
        f"# Photos used in run `{cfg.get('run_name')}`\n\n"
        f"{len(rows)} original photos (C0), {total} labelled objects (COCO labels plus audit "
        f"additions). A detector *found* an object when it predicted the same class with "
        f"IoU ≥ {ecfg['iou_threshold']} and score ≥ {ecfg['recall_score']}. Totals found: "
        + ", ".join(f"{DETECTOR_LABELS.get(n, n)} {found[n]}/{total}" for n in detectors)
        + ".\n\n"
    )
    (out / "images_used.md").write_text(
        intro + rows_markdown(rows, detectors, DETECTOR_LABELS), encoding="utf-8"
    )
    write_run_info(run_dir(cfg), "report", cfg, {"n_images": len(rows), "n_objects": total})
    return {"table": (out / "images_used.md").as_posix(), "n_images": len(rows), "n_objects": total}
