"""Stage `evaluate` (Task 1 real-only baseline; Task 3 detector metrics on the edited images).

Reads the ledger, detections and forensic scores of one run and writes:
  runs/<run>/metrics.json
  runs/<run>/tables/detector_ap.csv          AP / AP50 / AP75 / AR100 per detector × condition
                                             (all objects; `eval.all_objects_ap_conditions`)
  runs/<run>/tables/detector_c0_vs_c1.csv    per-instance detection flips caused by resaving
  runs/<run>/tables/forensics_real.csv       score distributions on real photos
  runs/<run>/tables/forensics_resave_shift.csv  paired C1 − C0 image-score change
  runs/<run>/summary.md                      human-readable summary to paste back

With edited conditions (C2–C4) in the ledger, also (Task 3, `eval/matched.py`):
  tables/detector_ap_subsets.csv   AP on original objects per condition, on AI-inserted (C4) and
                                   pasted (C3) objects, over the photos that have every condition
  tables/h1_origin.csv             AI object vs same-class twins / all photographed / C3 paste:
                                   recall, recall difference, AUROC of the detector score
  tables/h2_matched.csv            C1 -> Cx change rates of original objects per stratum
  tables/h2_contrasts.csv          paired C4 - C2 / C3 / C0 differences on uncovered objects
  tables/h1_instances.csv, h2_instances.csv   per-object rows behind them

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
from inlayfake.eval.captions import claim_rows, summarize_claims
from inlayfake.eval.coco_ap import ap_with_ci
from inlayfake.eval.forensic_origin import h3_tables
from inlayfake.eval.matched import (
    by_source,
    contrast_rows,
    matched_rows,
    origin_rows,
    summarize_matched,
    summarize_origin,
)
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
    # all-objects AP mixes origins once edits exist; the edited conditions get subset AP instead
    ap_conds = set(ecfg.get("all_objects_ap_conditions", ["C0", "C1"]))
    for name in cfg["detect"]["detectors"]:
        dets = load_detections(cfg, name)
        for cond, rs in sorted(by_cond.items()):
            if cond not in ap_conds:
                continue
            print(f"[evaluate] AP: {name} / {cond} ({len(rs)} images)", flush=True)
            m = ap_with_ci(
                rs, dets, categories, ecfg.get("ap_bootstrap", 0), rng, desc=f"{name} {cond}"
            )
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


EDITED_CONDITIONS = ("C2", "C3", "C4")
ORIGINALS = {"photographed"}


def task3_tables(cfg, recs, categories, rng) -> dict[str, list[dict]]:
    """H1/H2 detector tables for a run with edited images (empty dict without them)."""
    if not any(r.condition.value in EDITED_CONDITIONS for r in recs):
        return {}
    ecfg = cfg["eval"]
    conds = list(cfg["conditions"])
    sources = by_source(recs)
    complete = sorted((s for s, d in sources.items() if set(conds) <= set(d)), key=str)
    n_ap, n_boot = ecfg.get("ap_bootstrap", 0), ecfg["bootstrap_resamples"]
    iou_thr, score_thr = ecfg["iou_threshold"], ecfg["recall_score"]
    subsets = [("original", c, ORIGINALS) for c in conds]
    subsets += [("ai_inserted", "C4", {"ai_inserted"}), ("real_pasted", "C3", {"real_pasted"})]
    out: dict[str, list[dict]] = defaultdict(list)
    for name in cfg["detect"]["detectors"]:
        dets = load_detections(cfg, name)
        for objects, cond, origins in subsets:
            rs = [sources[s][cond] for s in complete]
            if not rs:
                continue
            print(f"[evaluate] subset AP: {name} / {cond} / {objects} ({len(rs)})", flush=True)
            m = ap_with_ci(
                rs, dets, categories, n_ap, rng, origins=origins, desc=f"{name} {cond} {objects}"
            )
            out["detector_ap_subsets"].append(
                {"detector": name, "condition": cond, "objects": objects, "n_images": len(rs)} | m
            )
        print(f"[evaluate] H1 origin / H2 matched: {name}", flush=True)
        h1 = origin_rows(sources, dets, iou_thr)
        out["h1_instances"] += [{"detector": name} | r for r in h1]
        out["h1_origin"] += [
            {"detector": name} | r for r in summarize_origin(h1, score_thr, n_boot, rng)
        ]
        h2 = matched_rows(sources, dets, conds, iou_thr, score_thr, cfg["edit"]["mask_dilate_px"])
        out["h2_instances"] += [{"detector": name} | r for r in h2]
        out["h2_matched"] += [{"detector": name} | r for r in summarize_matched(h2, n_boot, rng)]
        out["h2_contrasts"] += [{"detector": name} | r for r in contrast_rows(h2, n_boot, rng)]
    return dict(out)


def vlm_tables(cfg, recs, rng) -> dict[str, list[dict]]:
    """H5 claim tables from runs/<run>/vlm/*.jsonl (empty dict before the `vlm` stage ran)."""
    vdir = run_dir(cfg) / "vlm"
    paths = sorted(vdir.glob("*.jsonl")) if vdir.exists() else []
    if not paths:
        return {}
    rows = [json.loads(line) for p in paths for line in p.open(encoding="utf-8") if line.strip()]
    by_id = {r.image_id: r for r in recs}
    objects = claim_rows(rows, by_id, cfg["eval"]["iou_threshold"])
    summary, images = summarize_claims(objects, rows, cfg["eval"]["bootstrap_resamples"], rng)
    return {"vlm_objects": summary, "vlm_images": images, "vlm_object_rows": objects}


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


def _ci(r: dict, key: str, fmt: str = ".3f") -> str:
    lo, hi = r.get(f"{key}_lo"), r.get(f"{key}_hi")
    ci = f" [{lo:{fmt}}, {hi:{fmt}}]" if lo is not None and lo == lo else ""
    return f"{r[key]:{fmt}}{ci}"


def _task3_md(t3: dict[str, list[dict]]) -> list[str]:
    lines = [
        "",
        "## Subset AP on the photos with every condition (Task 3)",
        "",
        "Original objects only (AI-inserted, pasted and mixed/modified objects are ignore",
        "regions), then AI-inserted objects (C4) and pasted real objects (C3) only.",
        "",
        "| detector | cond | objects | AP [95% CI] | AP50 | AR100 |",
        "|---|---|---|---|---|---|",
    ]
    for r in t3["detector_ap_subsets"]:
        lines.append(
            _row(
                r["detector"],
                r["condition"],
                r["objects"],
                _ci(r, "AP"),
                f"{r['AP50']:.3f}",
                f"{r['AR100']:.3f}",
            )
        )
    lines += [
        "",
        "## H1: detector score on the AI object vs real objects (oracle localisation)",
        "",
        "Recall at score >= eval.recall_score; recall diff = AI − other on photos with both;",
        "AUROC of the score for telling the AI object from the other side (0.5 = none).",
        "",
        "| detector | comparison | n AI / other | recall AI | recall other | recall diff [CI] "
        "| AUROC [CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in t3["h1_origin"]:
        lines.append(
            _row(
                r["detector"],
                r["comparison"],
                f"{r['n_ai']} / {r['n_other']}",
                f"{r['recall_ai']:.3f}",
                f"{r['recall_other']:.3f}",
                _ci(r, "recall_diff"),
                _ci(r, "auroc"),
            )
        )
    lines += [
        "",
        "## H2: C1 → Cx changes on original objects",
        "",
        "flip = detected in exactly one of C1 and Cx; abs score change = mean |Δ| of the best",
        "same-class score. Strata by the photo's edit footprint (see eval/matched.py).",
        "",
        "| detector | cond | stratum | objects | recall C1 → Cx | flip [CI] | lost | gained "
        "| class flip | abs score change [CI] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t3["h2_matched"]:
        if r["stratum"] in ("uncovered", "uncovered_near", "covered", "modified"):
            lines.append(
                _row(
                    r["detector"],
                    r["condition"],
                    r["stratum"],
                    r["n_objects"],
                    f"{r['recall_c1']:.3f} → {r['recall_cx']:.3f}",
                    _ci(r, "flip", ".4f"),
                    f"{r['lost']:.4f}",
                    f"{r['gained']:.4f}",
                    f"{r['class_flip']:.4f}",
                    _ci(r, "abs_score_delta", ".4f"),
                )
            )
    lines += [
        "",
        "## H2: paired contrasts on the same uncovered original objects",
        "",
        "| detector | contrast | objects | flip diff [CI] | abs score change diff [CI] |",
        "|---|---|---|---|---|",
    ]
    for r in t3["h2_contrasts"]:
        lines.append(
            _row(
                r["detector"],
                r["contrast"],
                r["n_objects"],
                _ci(r, "flip_diff", ".4f"),
                _ci(r, "abs_score_delta_diff", ".4f"),
            )
        )
    return lines


def _h3_md(rows: list[dict]) -> list[str]:
    lines = [
        "",
        "## H3: forensic origin per object (test split; direction and thresholds fit on val)",
        "",
        "AI-inserted objects vs real objects (photographed C1-C4, pasted C3). Base rate of",
        "AUPRC = share of AI objects. TPR@5% = AI objects caught at the val 5%-FPR threshold;",
        "false accusation = real objects called AI at the val 80%-TPR threshold. AUROC vs",
        "pasted = AI objects against the real objects pasted at the same boxes (C3). C2 box",
        "= share of sham-edit boxes (repainted background, no object) flagged at that threshold.",
        "",
        "| method | mode | dir | AUROC [CI] | AUROC vs pasted [CI] | AUPRC (base) | TPR@5% "
        "| TPR@80% thr | false accusation [CI] | of pasted | C2 box flagged [CI] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            _row(
                r["method"],
                r["mode"],
                r["direction"],
                _ci(r, "auroc"),
                _ci(r, "auroc_vs_pasted"),
                f"{r['auprc']:.3f} ({r['base_rate']:.3f})",
                f"{r['tpr_at_val_fpr5']:.3f}",
                f"{r['tpr_at_val_tpr80']:.3f}",
                _ci(r, "false_accusation"),
                f"{r.get('false_accusation pasted (C3)', float('nan')):.3f}",
                _ci(r, "edited_background_flag"),
            )
        )
    return lines


def _vlm_md(t3: dict[str, list[dict]]) -> list[str]:
    lines = [
        "",
        "## H5: VLM origin claims per object (Stage 2)",
        "",
        "Share of objects the model calls AI / photographed / unknown, describes without an",
        "origin statement (none) or does not mention (no claim box at IoU >= 0.5).",
        "",
        "| model | setting | cond | objects | n | AI [CI] | photographed | unknown | none "
        "| not mentioned |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t3["vlm_objects"]:
        lines.append(
            _row(
                r["model"],
                r["setting"],
                r["condition"],
                r["object"],
                r["n"],
                _ci(r, "ai"),
                f"{r['photographed']:.3f}",
                f"{r['unknown']:.3f}",
                f"{r['none']:.3f}",
                f"{r['not_mentioned']:.3f}",
            )
        )
    lines += [
        "",
        "## H5: images with any AI claim",
        "",
        "| model | setting | cond | images | any AI claim [CI] |",
        "|---|---|---|---|---|",
    ]
    for r in t3["vlm_images"]:
        lines.append(_row(r["model"], r["setting"], r["condition"], r["n"], _ci(r, "any_ai")))
    return lines


def _summary_md(cfg, ap_rows, flip_rows, dist_rows, shift_rows, t3=None) -> str:
    t3 = t3 or {}
    lines = [
        f"# Evaluation summary: run `{cfg.get('run_name')}`",
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
    if "detector_ap_subsets" in t3:
        lines += _task3_md(t3)
    if "h3_origin" in t3:
        lines += _h3_md(t3["h3_origin"])
    if "vlm_objects" in t3:
        lines += _vlm_md(t3)
    return "\n".join(lines) + "\n"


def run(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    recs = load_records(cfg)
    categories = sorted(load_coco(cfg).cat_id)
    rd = run_dir(cfg)
    ap_rows, flip_rows = detector_tables(cfg, recs, categories, rng)
    t3 = task3_tables(cfg, recs, categories, rng)
    t3 |= vlm_tables(cfg, recs, rng)
    if "detector_ap_subsets" in t3:
        fcfg = cfg["forensics"]
        methods = list(fcfg["handcrafted"]) + list(fcfg.get("run_learned") or [])
        h3 = h3_tables(rd / "forensics", methods, cfg["eval"]["bootstrap_resamples"], rng)
        if h3:
            t3["h3_origin"] = h3
    print("[evaluate] forensic score tables", flush=True)
    dist_rows, shift_rows = forensic_tables(cfg, rng)
    _write_csv(ap_rows, rd / "tables" / "detector_ap.csv")
    _write_csv(flip_rows, rd / "tables" / "detector_c0_vs_c1.csv")
    _write_csv(dist_rows, rd / "tables" / "forensics_real.csv")
    _write_csv(shift_rows, rd / "tables" / "forensics_resave_shift.csv")
    for key, rows in t3.items():
        _write_csv(rows, rd / "tables" / f"{key}.csv")
    metrics = {
        "detector_ap": ap_rows,
        "detector_c0_vs_c1": flip_rows,
        "forensics_resave_shift": shift_rows,
    } | {k: v for k, v in t3.items() if not k.endswith(("_instances", "_rows"))}
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (rd / "summary.md").write_text(
        _summary_md(cfg, ap_rows, flip_rows, dist_rows, shift_rows, t3), encoding="utf-8"
    )
    write_run_info(rd, "evaluate", cfg, {"n_records": len(recs)})
    return {"summary": (rd / "summary.md").as_posix(), "n_records": len(recs)}
