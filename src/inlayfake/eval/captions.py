"""Origin-claim metrics for VLM outputs (H5; ARCHITECTURE.md §3.11 `captions.py`).

Every ground-truth object of an image (photographed, real_pasted or ai_inserted; crowd and
mixed/modified objects are left out) is matched to the claim whose box overlaps it most, if that
IoU is >= ``eval.iou_threshold`` (class-agnostic: the origin statement is scored, not the
class). Its status is the claim's origin (ai / photographed / unknown / none) or
``not_mentioned``. Rates per (model, setting, condition, object origin), with grouped-bootstrap
CIs over photos:
  false-AI     share of real objects (photographed, pasted) called ai
  false-real   share of AI-inserted objects called photographed
  detected-AI  share of AI-inserted objects called ai
  abstain      share called unknown
  no statement share described without an origin statement (none) or not mentioned at all
and per image: the share of images with any ai claim (on C1 a false alarm).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from inlayfake.edit.placement import iou
from inlayfake.eval.bootstrap import grouped_mean_ci
from inlayfake.ledger import ImageRecord, Origin

SCORED = {
    Origin.PHOTOGRAPHED: "photographed",
    Origin.REAL_PASTED: "real_pasted",
    Origin.AI_INSERTED: "ai_inserted",
}
STATUSES = ("ai", "photographed", "unknown", "none", "not_mentioned")


def object_statuses(rec: ImageRecord, claims: list[dict], iou_thr: float) -> list[dict]:
    out = []
    boxed = [c for c in claims if c.get("bbox")]
    for inst in rec.instances:
        if inst.iscrowd or inst.origin not in SCORED:
            continue
        best, status = 0.0, "not_mentioned"
        for c in boxed:
            v = iou(c["bbox"], inst.bbox)
            if v >= iou_thr and v > best:
                best, status = v, c["origin"]
        out.append(
            {
                "instance_id": inst.instance_id,
                "object": SCORED[inst.origin],
                "category": inst.category,
                "status": status,
                "iou": best,
            }
        )
    return out


def claim_rows(rows: list[dict], recs: dict[str, ImageRecord], iou_thr: float) -> list[dict]:
    """Per-object status rows for every VLM output row whose image is in ``recs``."""
    out = []
    for r in rows:
        rec = recs.get(r["image_id"])
        if rec is None:
            continue
        for s in object_statuses(rec, r["claims"], iou_thr):
            out.append(
                {k: r[k] for k in ("model", "setting", "image_id", "source_id", "condition")} | s
            )
    return out


def summarize_claims(
    obj_rows: list[dict], img_rows: list[dict], n_boot: int, rng: np.random.Generator
) -> tuple[list[dict], list[dict]]:
    """(per-object rates, per-image any-AI rates); two tables, since their columns differ."""
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in obj_rows:
        key = (r["model"], r["setting"], r["condition"], r["object"])
        groups[key][r["source_id"]].append(r["status"])
    out = []
    for (model, setting, cond, obj), per_src in sorted(groups.items()):
        row = {
            "model": model,
            "setting": setting,
            "condition": cond,
            "object": obj,
            "n": sum(len(v) for v in per_src.values()),
            "n_sources": len(per_src),
        }
        for st in STATUSES:
            p, lo, hi = grouped_mean_ci(
                {s: [float(x == st) for x in v] for s, v in per_src.items()}, n=n_boot, rng=rng
            )
            row |= {st: p, f"{st}_lo": lo, f"{st}_hi": hi}
        out.append(row)
    per_image = []
    images: dict = defaultdict(lambda: defaultdict(list))
    for r in img_rows:
        any_ai = float(any(c["origin"] == "ai" for c in r["claims"]))
        images[(r["model"], r["setting"], r["condition"])][r["source_id"]].append(any_ai)
    for (model, setting, cond), per_src in sorted(images.items()):
        p, lo, hi = grouped_mean_ci(per_src, n=n_boot, rng=rng)
        per_image.append(
            {
                "model": model,
                "setting": setting,
                "condition": cond,
                "n": sum(len(v) for v in per_src.values()),
                "n_sources": len(per_src),
                "any_ai": p,
                "any_ai_lo": lo,
                "any_ai_hi": hi,
            }
        )
    return out, per_image
