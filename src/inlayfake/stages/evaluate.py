"""Stage `evaluate` (Task 1: real-only baseline).

Reads the ledger, detections and forensic scores of one run and writes:
  runs/<run>/metrics.json
  runs/<run>/tables/detector_ap.csv          AP / AP50 / AP75 / AR100 per detector × condition
  runs/<run>/tables/detector_c0_vs_c1.csv    per-instance detection flips caused by resaving
  runs/<run>/tables/forensics_real.csv       score distributions on real photos
  runs/<run>/tables/forensics_resave_shift.csv  paired C1 − C0 image-score change
  runs/<run>/summary.md                      human-readable summary to paste back

Task 1 contains only photographed objects, so it cannot produce AUROC. Forensic false-alarm rates
need a threshold, and that threshold is fixed later on validation images that contain AI objects
(Task 3). The real-photo score files written here are re-used then.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from inlayfake.edit.placement import iou
from inlayfake.eval.bootstrap import grouped_bootstrap
from inlayfake.eval.coco_ap import ap_with_ci
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_coco, load_records, run_dir
from inlayfake.stages.detect import load_detections


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def detected(inst, dets: list[dict], iou_thr: float, score_thr: float) -> bool:
    return any(
        d["category"] == inst.category
        and d["score"] >= score_thr
        and iou(d["bbox"], inst.bbox) >= iou_thr
        for d in dets
    )


def detector_tables(cfg, recs, categories, rng) -> tuple[list[dict], list[dict]]:
    ecfg = cfg["eval"]
    by_cond = defaultdict(list)
    for r in recs:
        by_cond[r.condition.value].append(r)
    ap_rows, flip_rows = [], []
    for name in cfg["detect"]["detectors"]:
        dets = load_detections(cfg, name)
        for cond, rs in sorted(by_cond.items()):
            m = ap_with_ci(rs, dets, categories, ecfg.get("ap_bootstrap", 0), rng)
            ap_rows.append({"detector": name, "condition": cond, "n_images": len(rs)} | m)
        if {"C0", "C1"} <= set(by_cond):
            c1 = {r.source_id: r for r in by_cond["C1"]}
            lost, gained, hit0, hit1 = (
                defaultdict(list),
                defaultdict(list),
                defaultdict(list),
                defaultdict(list),
            )
            for r0 in by_cond["C0"]:
                r1 = c1.get(r0.source_id)
                if r1 is None:
                    continue
                for inst in r0.instances:
                    if inst.iscrowd:
                        continue
                    a = detected(
                        inst, dets.get(r0.image_id, []), ecfg["iou_threshold"], ecfg["recall_score"]
                    )
                    b = detected(
                        inst, dets.get(r1.image_id, []), ecfg["iou_threshold"], ecfg["recall_score"]
                    )
                    hit0[r0.source_id].append(a)
                    hit1[r0.source_id].append(b)
                    lost[r0.source_id].append(a and not b)
                    gained[r0.source_id].append(b and not a)
            n = ecfg["bootstrap_resamples"]
            row = {"detector": name, "n_instances": sum(len(v) for v in hit0.values())}
            for key, g in (
                ("recall_C0", hit0),
                ("recall_C1", hit1),
                ("lost_C0_to_C1", lost),
                ("gained_C0_to_C1", gained),
            ):
                p, lo, hi = grouped_bootstrap(g, n=n, rng=rng)
                row |= {key: p, f"{key}_lo": lo, f"{key}_hi": hi}
            flip_rows.append(row)
    return ap_rows, flip_rows


def forensic_tables(cfg, rng) -> tuple[list[dict], list[dict]]:
    fdir = run_dir(cfg) / "forensics"
    n = cfg["eval"]["bootstrap_resamples"]
    dist_rows, shift_rows = [], []
    for method in cfg["forensics"]["handcrafted"]:
        path = fdir / f"{method}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        groups = defaultdict(lambda: defaultdict(list))  # (cond, level, score) -> source -> values
        image_score = defaultdict(dict)  # source -> cond -> image score
        for r in rows:
            for k, v in r["scores"].items():
                if v == v:  # skip NaN
                    groups[(r["condition"], r["level"], k)][r["source_id"]].append(v)
            if r["level"] == "image":
                image_score[r["source_id"]][r["condition"]] = r["scores"]["p99_minus_median"]
        for (cond, level, score), g in sorted(groups.items()):
            vals = np.concatenate([np.asarray(v) for v in g.values()])
            mean, lo, hi = grouped_bootstrap(g, n=n, rng=rng)
            dist_rows.append(
                {
                    "method": method,
                    "condition": cond,
                    "level": level,
                    "score": score,
                    "n": int(vals.size),
                    "mean": mean,
                    "mean_lo": lo,
                    "mean_hi": hi,
                    "median": float(np.median(vals)),
                    "p95": float(np.percentile(vals, 95)),
                    "std": float(vals.std()),
                }
            )
        diffs = {s: [d["C1"] - d["C0"]] for s, d in image_score.items() if {"C0", "C1"} <= set(d)}
        if diffs:
            p, lo, hi = grouped_bootstrap(diffs, n=n, rng=rng)
            c0 = np.array([image_score[s]["C0"] for s in diffs])
            shift_rows.append(
                {
                    "method": method,
                    "n_sources": len(diffs),
                    "mean_C0": float(c0.mean()),
                    "mean_diff_C1_minus_C0": p,
                    "diff_lo": lo,
                    "diff_hi": hi,
                    "relative_diff": p / (abs(float(c0.mean())) + 1e-12),
                }
            )
    return dist_rows, shift_rows


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _summary_md(cfg, ap_rows, flip_rows, dist_rows, shift_rows) -> str:
    lines = [
        f"# Task 1 summary: run `{cfg.get('run_name')}`",
        "",
        "## Detector AP on real photos (C0 original, C1 resave)",
        "",
        "| detector | cond | AP [95% CI] | AP50 | AR100 |",
        "|---|---|---|---|---|",
    ]
    for r in ap_rows:
        ci = f" [{r['AP_lo']:.3f}, {r['AP_hi']:.3f}]" if "AP_lo" in r else ""
        lines.append(
            _row(
                r["detector"],
                r["condition"],
                f"{r['AP']:.3f}{ci}",
                f"{r['AP50']:.3f}",
                f"{r['AR100']:.3f}",
            )
        )
    lines += [
        "",
        "## Effect of resaving (C0 → C1) on per-object detection",
        "",
        "| detector | objects | recall C0 | recall C1 | lost | gained |",
        "|---|---|---|---|---|---|",
    ]
    for r in flip_rows:
        lines.append(
            _row(
                r["detector"],
                r["n_instances"],
                f"{r['recall_C0']:.3f}",
                f"{r['recall_C1']:.3f}",
                f"{r['lost_C0_to_C1']:.4f}",
                f"{r['gained_C0_to_C1']:.4f}",
            )
        )
    lines += [
        "",
        "## Forensic image score on real photos (p99 − median of the map)",
        "",
        "| method | cond | mean [95% CI] | median | p95 |",
        "|---|---|---|---|---|",
    ]
    for r in dist_rows:
        if r["level"] == "image" and r["score"] == "p99_minus_median":
            lines.append(
                f"| {r['method']} | {r['condition']} | {r['mean']:.4g} [{r['mean_lo']:.4g}, "
                f"{r['mean_hi']:.4g}] | {r['median']:.4g} | {r['p95']:.4g} |"
            )
    lines += [
        "",
        "## Per-object contrast on real objects (inside − context)",
        "",
        "| method | cond | n | mean | std | p95 |",
        "|---|---|---|---|---|---|",
    ]
    for r in dist_rows:
        if r["level"] == "instance" and r["score"] == "contrast":
            lines.append(
                _row(
                    r["method"],
                    r["condition"],
                    r["n"],
                    f"{r['mean']:.4g}",
                    f"{r['std']:.4g}",
                    f"{r['p95']:.4g}",
                )
            )
    lines += [
        "",
        "## Resave shift of the image score (C1 − C0)",
        "",
        "| method | sources | mean diff [95% CI] | relative |",
        "|---|---|---|---|",
    ]
    for r in shift_rows:
        lines.append(
            f"| {r['method']} | {r['n_sources']} | {r['mean_diff_C1_minus_C0']:.4g} "
            f"[{r['diff_lo']:.4g}, {r['diff_hi']:.4g}] | {r['relative_diff']:+.1%} |"
        )
    return "\n".join(lines) + "\n"


def run(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    recs = load_records(cfg)
    categories = sorted(load_coco(cfg).cat_id)
    rd = run_dir(cfg)
    ap_rows, flip_rows = detector_tables(cfg, recs, categories, rng)
    dist_rows, shift_rows = forensic_tables(cfg, rng)
    _write_csv(ap_rows, rd / "tables" / "detector_ap.csv")
    _write_csv(flip_rows, rd / "tables" / "detector_c0_vs_c1.csv")
    _write_csv(dist_rows, rd / "tables" / "forensics_real.csv")
    _write_csv(shift_rows, rd / "tables" / "forensics_resave_shift.csv")
    metrics = {
        "detector_ap": ap_rows,
        "detector_c0_vs_c1": flip_rows,
        "forensics_resave_shift": shift_rows,
    }
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (rd / "summary.md").write_text(
        _summary_md(cfg, ap_rows, flip_rows, dist_rows, shift_rows), encoding="utf-8"
    )
    write_run_info(rd, "evaluate", cfg, {"n_records": len(recs)})
    return {"summary": (rd / "summary.md").as_posix(), "n_records": len(recs)}
