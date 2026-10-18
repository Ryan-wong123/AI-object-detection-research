"""Label audit (ARCHITECTURE.md §3.2): find likely-missing COCO labels and apply decisions.

A candidate is a confident detection that several detectors agree on but that matches no ground
truth box. The author reviews the candidates and marks each one in audit_decisions.csv:
  add     → a real object COCO missed; becomes a photographed ground-truth instance
  ignore  → ambiguous; becomes an ignore region (neither rewarded nor penalised)
  reject  → a detector error; nothing changes
  ?       → not reviewed; treated like ignore (the conservative choice)
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from inlayfake.edit.placement import iou
from inlayfake.ledger import ImageRecord, Instance, Origin

FIELDS = [
    "candidate_id",
    "source_id",
    "category",
    "x",
    "y",
    "w",
    "h",
    "n_detectors",
    "max_score",
    "decision",
]


def find_candidates(
    record: ImageRecord,
    dets_by_detector: dict[str, list[dict]],
    min_score: float,
    min_agree: int,
    same_class_iou: float = 0.3,
    any_class_iou: float = 0.5,
) -> list[dict]:
    gt = [i for i in record.instances]
    confident = [
        (name, d)
        for name, dets in dets_by_detector.items()
        for d in dets
        if d["score"] >= min_score
    ]
    confident.sort(key=lambda t: -t[1]["score"])
    out: list[dict] = []
    for name, d in confident:
        same = [g for g in gt if g.category == d["category"]]
        if any(iou(d["bbox"], g.bbox) >= same_class_iou for g in same):
            continue
        if any(iou(d["bbox"], g.bbox) >= any_class_iou for g in gt):
            continue  # overlaps a labelled object of another class: a class confusion, not a miss
        if any(c["category"] == d["category"] and iou(c["bbox"], d["bbox"]) >= 0.5 for c in out):
            continue  # already covered by a stronger candidate
        agree = {name} | {
            other
            for other, od in confident
            if other != name
            and od["category"] == d["category"]
            and iou(od["bbox"], d["bbox"]) >= 0.5
        }
        if len(agree) >= min_agree:
            out.append(
                {
                    "category": d["category"],
                    "bbox": [round(v, 1) for v in d["bbox"]],
                    "n_detectors": len(agree),
                    "max_score": round(d["score"], 3),
                }
            )
    return out


def write_decisions_template(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            x, y, bw, bh = r["bbox"]
            w.writerow(
                {
                    "candidate_id": r["candidate_id"],
                    "source_id": r["source_id"],
                    "category": r["category"],
                    "x": x,
                    "y": y,
                    "w": bw,
                    "h": bh,
                    "n_detectors": r["n_detectors"],
                    "max_score": r["max_score"],
                    "decision": "?",
                }
            )


def read_decisions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["decision"] = (r.get("decision") or "?").strip().lower()
        if r["decision"] not in {"add", "ignore", "reject", "?"}:
            raise ValueError(f"bad decision {r['decision']!r} for candidate {r['candidate_id']}")
    return rows


def apply_decisions(records: list[ImageRecord], decisions: list[dict]) -> list[ImageRecord]:
    """Return copies of ``records`` with audited labels added and ignore regions set. Geometry is
    identical across conditions of one source, so each decision applies to all of them."""
    by_src: dict[str, list[dict]] = {}
    for d in decisions:
        by_src.setdefault(str(d["source_id"]), []).append(d)
    out = []
    for r in records:
        ds = by_src.get(str(r.source_id), [])
        if not ds:
            out.append(r)
            continue
        insts = list(r.instances)
        ignores = list(r.ignore_regions)
        for d in ds:
            box = [float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"])]
            if d["decision"] == "add":
                insts.append(
                    Instance(f"audit:{d['candidate_id']}", d["category"], box, Origin.PHOTOGRAPHED)
                )
            elif d["decision"] in ("ignore", "?"):
                ignores.append({"category": d["category"], "bbox": box})
        out.append(replace(r, instances=insts, ignore_regions=ignores))
    return out
