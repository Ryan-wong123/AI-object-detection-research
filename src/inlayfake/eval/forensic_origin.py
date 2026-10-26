"""H3: per-object origin metrics of the forensic methods (oracle localisation; PAPER.md §6).

Objects: AI-inserted objects of C4 are positives; photographed objects of C1–C4 and the real
objects pasted in C3 are negatives. C0 is left out (original bytes, not the benchmark encoding;
golden rule 4); mixed/modified objects are left out (partly AI-repainted, neither class).

Scoring modes (ARCHITECTURE §3.7) per method:
  map methods (F1–F5, TruFor): image (the image's p99 − median, shared by all its objects),
    region_mean / region_p95 (inside the object mask), contrast (inside − context ring);
    TruFor also image_det (its own image score)
  UnivFD: image (logit of the whole image), crop (logit of the object crop), contrast
    (crop − image)

Everything that is fitted is fitted on val, then applied unchanged to test (golden rule 6):
the score's direction (flipped if its val AUROC < 0.5), the threshold for 5% FPR on val
negatives, and the threshold for 80% TPR on val positives. Test metrics: AUROC, AUPRC (base rate
= share of positives), TPR at the val 5%-FPR threshold, and at the val 80%-TPR threshold the TPR
reached and the **false-accusation rate** (share of real objects called AI), overall and per kind
of real object. Grouped-bootstrap 95% CIs over test photos.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from inlayfake.eval.bootstrap import grouped_bootstrap_fn, grouped_mean_ci
from inlayfake.eval.origin import auroc

CONDITIONS = ("C1", "C2", "C3", "C4")
MAP_MODES = {
    "image": ("image", "p99_minus_median"),
    "region_mean": ("instance", "mean_in"),
    "region_p95": ("instance", "p95_in"),
    "contrast": ("instance", "contrast"),
}
UNIVFD_MODES = {
    "image": ("image", "logit"),
    "crop": ("instance", "crop_logit"),
    "contrast": ("instance", "contrast"),
}


def modes_for(method: str) -> dict[str, tuple[str, str]]:
    if method == "univfd":
        return UNIVFD_MODES
    if method == "trufor":
        return MAP_MODES | {"image_det": ("image", "det")}
    return MAP_MODES


def negative_kind(origin: str, condition: str) -> str | None:
    if origin == "real_pasted":
        return "pasted (C3)"
    if origin == "photographed":
        return f"photographed {condition}"
    return None


def object_scores(rows: list[dict], method: str) -> list[dict]:
    """One entry per scored object: split, source, label (1 = AI), negative kind, mode scores."""
    image = {r["image_id"]: r["scores"] for r in rows if r["level"] == "image"}
    out = []
    for r in rows:
        if r["level"] != "instance" or r["condition"] not in CONDITIONS:
            continue
        if r["origin"] == "ai_inserted":
            label, kind = 1, "ai_inserted"
        else:
            kind = negative_kind(r["origin"], r["condition"])
            if kind is None:
                continue
            label = 0
        scores = {}
        for mode, (level, key) in modes_for(method).items():
            src = image[r["image_id"]] if level == "image" else r["scores"]
            v = src.get(key, float("nan"))
            if v == v:
                scores[mode] = float(v)
        out.append(
            {
                "split": r["split"],
                "source_id": r["source_id"],
                "label": label,
                "kind": kind,
                "scores": scores,
            }
        )
    return out


def edit_box_scores(box_rows: list[dict], rows: list[dict], method: str) -> list[dict]:
    """Scores of each edit box (<method>__editbox.jsonl) in the same modes as the objects: region
    modes from the box, image modes from the image's own row."""
    image = {r["image_id"]: r["scores"] for r in rows if r["level"] == "image"}
    out = []
    for r in box_rows:
        if r["image_id"] not in image:
            continue
        scores = {}
        for mode, (level, key) in modes_for(method).items():
            src = image[r["image_id"]] if level == "image" else r["scores"]
            v = src.get(key, float("nan"))
            if v == v:
                scores[mode] = float(v)
        out.append(
            {
                "split": r["split"],
                "source_id": r["source_id"],
                "condition": r["condition"],
                "scores": scores,
            }
        )
    return out


def auprc(pos, neg) -> float:
    """Average precision of ranking positives above negatives (ties broken pessimistically)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not pos.size or not neg.size:
        return float("nan")
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.lexsort((y, -s))  # by score desc; on ties negatives first
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / pos.size)


def _split(objs, split, mode, sign):
    sel = [o for o in objs if o["split"] == split and mode in o["scores"]]
    return [(o, sign * o["scores"][mode]) for o in sel]


def _pooled(groups, fn):
    pos = [s for g in groups for s in g[0]]
    neg = [s for g in groups for s in g[1]]
    return fn(pos, neg)


def method_rows(
    objs: list[dict],
    method: str,
    n_boot: int,
    rng,
    fpr: float = 0.05,
    tpr: float = 0.80,
    boxes: list[dict] | None = None,
) -> list[dict]:
    rows = []
    for mode in modes_for(method):
        val = _split(objs, "val", mode, 1.0)
        vpos = [s for o, s in val if o["label"] == 1]
        vneg = [s for o, s in val if o["label"] == 0]
        if not vpos or not vneg:
            continue
        val_auc = auroc(vpos, vneg)
        sign = -1.0 if val_auc < 0.5 else 1.0
        vpos = [sign * s for s in vpos]
        vneg = [sign * s for s in vneg]
        thr_fpr = float(np.quantile(vneg, 1 - fpr))  # 5% of val negatives above it
        thr_tpr = float(np.quantile(vpos, 1 - tpr))  # 80% of val positives at or above it
        test = _split(objs, "test", mode, sign)
        groups: dict = defaultdict(lambda: ([], []))
        for o, s in test:
            groups[o["source_id"]][0 if o["label"] == 1 else 1].append(s)
        row = {
            "method": method,
            "mode": mode,
            "direction": "higher=AI" if sign > 0 else "lower=AI",
            "val_auroc": max(val_auc, 1 - val_auc),
            "n_pos": sum(len(g[0]) for g in groups.values()),
            "n_neg": sum(len(g[1]) for g in groups.values()),
        }
        for name, fn in (("auroc", auroc), ("auprc", auprc)):
            p, lo, hi = grouped_bootstrap_fn(
                groups, lambda gs, f=fn: _pooled(gs, f), n=n_boot, rng=rng
            )
            row |= {name: p, f"{name}_lo": lo, f"{name}_hi": hi}
        row["base_rate"] = row["n_pos"] / max(row["n_pos"] + row["n_neg"], 1)
        # AI objects against the real objects pasted at the same boxes (C3): both composited
        # into the photo, so this isolates AI generation from compositing (§7.6 control)
        pasted: dict = defaultdict(lambda: ([], []))
        for o, s in test:
            if o["label"] == 1 or o["kind"] == "pasted (C3)":
                pasted[o["source_id"]][0 if o["label"] == 1 else 1].append(s)
        p, lo, hi = grouped_bootstrap_fn(pasted, lambda gs: _pooled(gs, auroc), n=n_boot, rng=rng)
        row |= {"auroc_vs_pasted": p, "auroc_vs_pasted_lo": lo, "auroc_vs_pasted_hi": hi}
        pos_hit = defaultdict(list)
        for o, s in test:
            if o["label"] == 1:
                pos_hit[o["source_id"]].append(float(s > thr_fpr))
        row["tpr_at_val_fpr5"] = grouped_mean_ci(pos_hit, n=0)[0]
        pos80 = defaultdict(list)
        neg80 = defaultdict(list)
        kinds: dict = defaultdict(lambda: defaultdict(list))
        for o, s in test:
            flag = float(s >= thr_tpr)
            if o["label"] == 1:
                pos80[o["source_id"]].append(flag)
            else:
                neg80[o["source_id"]].append(flag)
                kinds[o["kind"]][o["source_id"]].append(flag)
        row["tpr_at_val_tpr80"] = grouped_mean_ci(pos80, n=0)[0]
        p, lo, hi = grouped_mean_ci(neg80, n=n_boot, rng=rng)
        row |= {"false_accusation": p, "false_accusation_lo": lo, "false_accusation_hi": hi}
        for kind in sorted(kinds):
            row[f"false_accusation {kind}"] = grouped_mean_ci(kinds[kind], n=0)[0]
        # edited-background flag rate: C2 boxes (repainted background, no object) at the same
        # threshold; C4 boxes for reference (the AI object's own box)
        for cond, key in (("C2", "edited_background_flag"), ("C4", "c4_box_flag")):
            flags: dict = defaultdict(list)
            for b in boxes or []:
                if b["split"] == "test" and b["condition"] == cond and mode in b["scores"]:
                    flags[b["source_id"]].append(float(sign * b["scores"][mode] >= thr_tpr))
            p, lo, hi = grouped_mean_ci(flags, n=n_boot if cond == "C2" else 0, rng=rng)
            row |= {key: p, f"{key}_lo": lo, f"{key}_hi": hi}
        rows.append(row)
    return rows


def h3_tables(fdir: Path, methods: list[str], n_boot: int, rng) -> list[dict]:
    rows = []
    for m in methods:
        path = fdir / f"{m}.jsonl"
        if not path.exists():
            continue
        data = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        objs = object_scores(data, m)
        if not any(o["label"] == 1 for o in objs):
            continue
        bpath = fdir / f"{m}__editbox.jsonl"
        boxes = None
        if bpath.exists():
            brows = [json.loads(line) for line in bpath.open(encoding="utf-8") if line.strip()]
            boxes = edit_box_scores(brows, data, m)
        print(f"[evaluate] H3 origin: {m}", flush=True)
        rows += method_rows(objs, m, n_boot, rng, boxes=boxes)
    if not rows:
        return rows
    kinds = sorted({k for r in rows for k in r if k.startswith("false_accusation ")})
    base = [k for k in rows[0] if not k.startswith("false_accusation ")]
    return [{k: r.get(k, float("nan")) for k in base + kinds} for r in rows]  # one CSV header
