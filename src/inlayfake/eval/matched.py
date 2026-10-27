"""Per-instance detector analysis on the edited images (H1, H2; PRD §2, ARCHITECTURE §3.11).

Both analyses use oracle localisation: a ground-truth object's *score* is that of its best
same-class detection with IoU >= ``eval.iou_threshold`` (0 if none), and it is *detected* when
that score is >= ``eval.recall_score`` (the rule of ``evaluate.detected``, the viewer and the
report). Nothing is fitted, so all splits are used.

H1 (origin): in each C4 image, the AI-inserted object against the photographed objects of the
same class in that photo ("twins"), against all its photographed objects, and against the real
object pasted at the same box in C3.

H2 (collateral): each original object is followed from C1 (the matched-encoding reference) into
C0, C2, C3 and C4 of the same photo. C2–C4 share one edit box per photo, so an object has the same
geometric stratum in every condition:
  covered         its box overlaps the edit footprint (planned box + mask dilation + one 8-px
                  JPEG block, the pixels an edit can change)
  uncovered_near  no overlap, gap to the footprint < the footprint's longer side
  uncovered_far   further away
  modified        labelled mixed_modified in that condition (the editor repainted > 10% of it)
  no_edit         photo without any edited condition (C0 only)
Detectors are deterministic (identical output on the same image in two runs, checked
2026-10-23), so any C1 → Cx change on an uncovered object is caused by the edit elsewhere.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from inlayfake.edit.placement import expand, intersection, iou
from inlayfake.eval.bootstrap import grouped_bootstrap_fn, grouped_mean_ci
from inlayfake.eval.origin import auroc
from inlayfake.ledger import ImageRecord, Origin

JPEG_BLOCK_PX = 8
EDITED = ("C4", "C2", "C3")  # conditions that carry the photo's edit box (all the same box)


def best_match(
    bbox: list[float], category: str, dets: list[dict], iou_thr: float
) -> tuple[float, list[float] | None]:
    """Score and box of the best same-class detection with IoU >= iou_thr; (0.0, None) if none."""
    score, box = 0.0, None
    for d in dets:
        if d["category"] == category and d["score"] > score and iou(d["bbox"], bbox) >= iou_thr:
            score, box = d["score"], d["bbox"]
    return score, box


def top_label(bbox: list[float], dets: list[dict], iou_thr: float, score_thr: float) -> str | None:
    """Class of the highest-scoring detection of any class on this object (IoU and score rule)."""
    score, label = -1.0, None
    for d in dets:
        if d["score"] >= score_thr and d["score"] > score and iou(d["bbox"], bbox) >= iou_thr:
            score, label = d["score"], d["category"]
    return label


def covered_fraction(bbox: list[float], region: list[float]) -> float:
    area = bbox[2] * bbox[3]
    return intersection(bbox, region) / area if area > 0 else 0.0


def box_gap(a: list[float], b: list[float]) -> float:
    """Shortest distance between two boxes (0 when they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return math.hypot(dx, dy)


def edit_footprint(recs: dict[str, ImageRecord], dilate_px: float) -> list[float] | None:
    for cond in EDITED:
        r = recs.get(cond)
        if r is not None and r.edit is not None:
            return expand(r.edit.plan_box, dilate_px + JPEG_BLOCK_PX)
    return None


def stratum(c1_bbox: list[float], cx_origin: Origin, footprint: list[float] | None) -> str:
    if cx_origin is not Origin.PHOTOGRAPHED:
        return "modified"
    if footprint is None:
        return "no_edit"
    if covered_fraction(c1_bbox, footprint) > 0:
        return "covered"
    near = box_gap(c1_bbox, footprint) < max(footprint[2], footprint[3])
    return "uncovered_near" if near else "uncovered_far"


def by_source(records: list[ImageRecord]) -> dict:
    out: dict = defaultdict(dict)
    for r in records:
        out[r.source_id][r.condition.value] = r
    return out


# ---------------------------------------------------------------- H2: matched instances


def matched_rows(
    sources: dict,
    dets: dict[str, list[dict]],
    conditions: list[str],
    iou_thr: float,
    score_thr: float,
    dilate_px: float,
) -> list[dict]:
    """One row per (original object, condition other than C1) of every photo with a C1."""
    rows = []
    for sid, recs in sources.items():
        r1 = recs.get("C1")
        if r1 is None:
            continue
        footprint = edit_footprint(recs, dilate_px)
        d1 = dets.get(r1.image_id, [])
        for cond in conditions:
            rx = recs.get(cond)
            if cond == "C1" or rx is None:
                continue
            dx = dets.get(rx.image_id, [])
            in_x = {i.instance_id: i for i in rx.instances}
            for inst in r1.instances:
                ix = in_x.get(inst.instance_id)
                if inst.iscrowd or inst.origin is not Origin.PHOTOGRAPHED or ix is None:
                    continue
                s1, b1 = best_match(inst.bbox, inst.category, d1, iou_thr)
                sx, bx = best_match(ix.bbox, inst.category, dx, iou_thr)
                det1, detx = s1 >= score_thr, sx >= score_thr
                rows.append(
                    {
                        "source_id": sid,
                        "split": r1.split,
                        "condition": cond,
                        "instance_id": inst.instance_id,
                        "category": inst.category,
                        "stratum": stratum(inst.bbox, ix.origin, footprint),
                        "covered_frac": (
                            covered_fraction(inst.bbox, footprint) if footprint else 0.0
                        ),
                        "det_c1": det1,
                        "det_cx": detx,
                        "score_c1": s1,
                        "score_cx": sx,
                        "class_c1": top_label(inst.bbox, d1, iou_thr, score_thr),
                        "class_cx": top_label(ix.bbox, dx, iou_thr, score_thr),
                        "box_iou": iou(b1, bx) if det1 and detx else float("nan"),
                    }
                )
    return rows


def _values(r: dict) -> dict[str, float]:
    """Per-object change indicators C1 → Cx (NaN where the quantity is undefined)."""
    cat = r["category"]
    return {
        "recall_c1": float(r["det_c1"]),
        "recall_cx": float(r["det_cx"]),
        "lost": float(r["det_c1"] and not r["det_cx"]),
        "gained": float(r["det_cx"] and not r["det_c1"]),
        "flip": float(r["det_c1"] != r["det_cx"]),
        # correct top class in C1, another class on top in Cx
        "class_flip": float(r["class_c1"] == cat and r["class_cx"] not in (None, cat)),
        "score_delta": r["score_cx"] - r["score_c1"],
        "abs_score_delta": abs(r["score_cx"] - r["score_c1"]),
        "box_drift": 1.0 - r["box_iou"],  # NaN unless detected in both
    }


METRICS = (
    "recall_c1",
    "recall_cx",
    "lost",
    "gained",
    "flip",
    "class_flip",
    "score_delta",
    "abs_score_delta",
    "box_drift",
)


def _groups_of(stratum_name: str) -> list[str]:
    out = [stratum_name, "all"]
    if stratum_name.startswith("uncovered"):
        out.append("uncovered")
    return out


def summarize_matched(rows: list[dict], n_boot: int, rng: np.random.Generator) -> list[dict]:
    """Change rates per (condition, stratum group) with grouped-bootstrap 95% CIs."""
    cells: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        v = _values(r)
        for g in _groups_of(r["stratum"]):
            for m in METRICS:
                cells[(r["condition"], g)][m][r["source_id"]].append(v[m])
    out = []
    for (cond, g), per_metric in sorted(cells.items()):
        row = {
            "condition": cond,
            "stratum": g,
            "n_objects": sum(len(v) for v in per_metric["flip"].values()),
            "n_sources": len(per_metric["flip"]),
        }
        for m in METRICS:
            p, lo, hi = grouped_mean_ci(per_metric[m], n=n_boot, rng=rng)
            row |= {m: p, f"{m}_lo": lo, f"{m}_hi": hi}
        out.append(row)
    return out


CONTRAST_METRICS = ("flip", "lost", "gained", "class_flip", "abs_score_delta")


def contrast_rows(
    rows: list[dict],
    n_boot: int,
    rng: np.random.Generator,
    a: str = "C4",
    controls: tuple[str, ...] = ("C2", "C3", "C0"),
    group: str = "uncovered",
) -> list[dict]:
    """Paired difference ``a − control`` of each change rate on the *same* objects: present and in
    ``group`` in both conditions (H2's primary test: AI insertion against the sham edit, the
    real-object paste and the plain re-encoding)."""
    by_obj: dict = defaultdict(dict)
    for r in rows:
        if group in _groups_of(r["stratum"]):
            by_obj[(r["source_id"], r["instance_id"])][r["condition"]] = _values(r)
    out = []
    for ctrl in controls:
        pairs = {k: v for k, v in by_obj.items() if a in v and ctrl in v}
        if not pairs:
            continue
        row = {
            "contrast": f"{a} - {ctrl}",
            "a": a,
            "control": ctrl,
            "stratum": group,
            "n_objects": len(pairs),
            "n_sources": len({k[0] for k in pairs}),
        }
        for m in CONTRAST_METRICS:
            diffs: dict = defaultdict(list)
            vals_a: dict = defaultdict(list)
            vals_c: dict = defaultdict(list)
            for (sid, _), v in pairs.items():
                diffs[sid].append(v[a][m] - v[ctrl][m])
                vals_a[sid].append(v[a][m])
                vals_c[sid].append(v[ctrl][m])
            p, lo, hi = grouped_mean_ci(diffs, n=n_boot, rng=rng)
            row |= {
                f"{m}_a": grouped_mean_ci(vals_a, n=0)[0],
                f"{m}_control": grouped_mean_ci(vals_c, n=0)[0],
                f"{m}_diff": p,
                f"{m}_diff_lo": lo,
                f"{m}_diff_hi": hi,
            }
        out.append(row)
    return out


# ---------------------------------------------------------------- H1: origin of the object


def origin_rows(sources: dict, dets: dict[str, list[dict]], iou_thr: float) -> list[dict]:
    """Scores of the AI object of every C4 image, the photographed objects in the same image
    (``twin`` = same class as the AI object, ``other_photo`` = any other class), and the real
    object pasted at the same box in C3 (``pasted``)."""
    rows = []
    for sid, recs in sources.items():
        r4 = recs.get("C4")
        if r4 is None:
            continue
        ai = [i for i in r4.instances if i.origin is Origin.AI_INSERTED]
        ai_classes = {i.category for i in ai}
        found = [(r4, i, "ai") for i in ai]
        found += [
            (r4, i, "twin" if i.category in ai_classes else "other_photo")
            for i in r4.instances
            if i.origin is Origin.PHOTOGRAPHED and not i.iscrowd
        ]
        r3 = recs.get("C3")
        if r3 is not None:
            found += [(r3, i, "pasted") for i in r3.instances if i.origin is Origin.REAL_PASTED]
        for rec, inst, group in found:
            score, _ = best_match(inst.bbox, inst.category, dets.get(rec.image_id, []), iou_thr)
            rows.append(
                {
                    "source_id": sid,
                    "split": rec.split,
                    "condition": rec.condition.value,
                    "group": group,
                    "instance_id": inst.instance_id,
                    "category": inst.category,
                    "area": inst.bbox[2] * inst.bbox[3],
                    "score": score,
                }
            )
    return rows


def _pooled_auroc(groups: list[tuple[list[float], list[float]]]) -> float:
    pos = [s for g in groups for s in g[0]]
    neg = [s for g in groups for s in g[1]]
    return auroc(pos, neg)


def _paired_diff(groups: list[tuple[list[float], list[float]]]) -> float:
    """Mean of the positives minus mean of the negatives, over photos that have both."""
    both = [g for g in groups if g[0] and g[1]]
    if not both:
        return float("nan")
    return float(
        np.mean([s for g in both for s in g[0]]) - np.mean([s for g in both for s in g[1]])
    )


def summarize_origin(
    rows: list[dict], score_thr: float, n_boot: int, rng: np.random.Generator
) -> list[dict]:
    """Per comparison (AI object vs twins / all photographed / the C3 paste): recall of each side,
    the recall difference on photos with both, and the AUROC of the detector score for telling
    the AI object from the other side (0.5 = no separation; > 0.5 = AI objects score higher)."""
    per: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["group"]][r["source_id"]].append(r["score"])
    sides = {
        "twin": ("twin",),
        "photographed": ("twin", "other_photo"),
        "pasted": ("pasted",),
    }
    out = []
    for name, neg_groups in sides.items():
        groups = {}
        for sid in per["ai"]:
            neg = [s for g in neg_groups for s in per[g].get(sid, [])]
            groups[sid] = (per["ai"][sid], neg)
        rec = {
            sid: ([float(s >= score_thr) for s in p], [float(s >= score_thr) for s in n])
            for sid, (p, n) in groups.items()
        }
        row = {
            "comparison": f"ai vs {name}",
            "n_ai": sum(len(p) for p, _ in groups.values()),
            "n_other": sum(len(n) for _, n in groups.values()),
            "n_sources_both": sum(1 for p, n in groups.values() if p and n),
            "recall_ai": grouped_mean_ci({k: v[0] for k, v in rec.items()}, n=0)[0],
            "recall_other": grouped_mean_ci({k: v[1] for k, v in rec.items()}, n=0)[0],
            "mean_score_ai": grouped_mean_ci({k: v[0] for k, v in groups.items()}, n=0)[0],
            "mean_score_other": grouped_mean_ci({k: v[1] for k, v in groups.items()}, n=0)[0],
        }
        p, lo, hi = grouped_bootstrap_fn(rec, _paired_diff, n=n_boot, rng=rng)
        row |= {"recall_diff": p, "recall_diff_lo": lo, "recall_diff_hi": hi}
        p, lo, hi = grouped_bootstrap_fn(groups, _pooled_auroc, n=n_boot, rng=rng)
        row |= {"auroc": p, "auroc_lo": lo, "auroc_hi": hi}
        out.append(row)
    return out
