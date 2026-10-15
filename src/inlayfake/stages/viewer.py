"""Stage `viewer`: a local web page showing each photo's labels next to every detector's boxes.

Writes runs/<run>/viewer/index.html (template: viewer_page.html) plus data.js. Open index.html
in a browser (no server needed).

The page shows results, not a tuning tool: boxes are drawn at exactly the operating point used by
`evaluate` (recall) and `report` (images_used.md): score >= eval.recall_score and, for a match,
same class with IoU >= eval.iou_threshold. Each detector panel's "found X/N" therefore equals the
corresponding cell of images_used.md. The run's summary.md and the images_used.md totals are
rendered at the top of the page.

Two output modes:
  local (default): runs/<run>/viewer/. Images are referenced in place from data/ (fast, private).
  --share: docs/viewer/<run>/. Self-contained page + data.js + downscaled JPEG copies
    (viewer.share_max_side / share_quality), so the folder can be committed, zipped or served
    with GitHub Pages.

Per photo: a ground-truth panel (COCO labels, audit additions, ignore/crowd regions, inserted or
pasted objects) and one panel per detector. Box colours: green = the box matches a labelled object;
orange = it matches no label (a false positive or an object COCO did not label); grey = it falls on
an ignore region; dashed = a labelled object the detector did not find.
"""

from __future__ import annotations

import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from inlayfake.edit.encode import load_rgb
from inlayfake.edit.placement import iou
from inlayfake.ledger import ImageRecord
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_records, read_jsonl, run_dir, work_dir
from inlayfake.stages.detect import load_detections

MATCHED, UNMATCHED, IGNORED = 1, 0, 2
NOT_SCORED = ("crowd", "ignore")


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


def score_panel(dets: list[dict], gt: list[list], iou_thr: float, score_thr: float) -> dict:
    """Detections and found labels at the evaluation operating point.

    Uses the same rule as ``evaluate.detected`` (and therefore ``report``): a labelled object is
    found if any detection of the same class with score >= score_thr overlaps it with
    IoU >= iou_thr. A detection is MATCHED if it overlaps such a label, IGNORED if it instead
    overlaps an ignore/crowd region of its class, else UNMATCHED.
    """

    def hits(d: dict, g: list) -> bool:
        return g[4] == d["category"] and iou(d["bbox"], g[:4]) >= iou_thr

    shown = [d for d in dets if d["score"] >= score_thr]
    scorable = [i for i, g in enumerate(gt) if g[5] not in NOT_SCORED]
    found = sorted(i for i in scorable if any(hits(d, gt[i]) for d in shown))
    boxes = []
    for d in sorted(shown, key=lambda d: -d["score"]):
        if any(hits(d, gt[i]) for i in scorable):
            status = MATCHED
        elif any(hits(d, g) for g in gt if g[5] in NOT_SCORED):
            status = IGNORED
        else:
            status = UNMATCHED
        boxes.append(
            [*(round(v, 1) for v in d["bbox"]), d["category"], round(d["score"], 3), status]
        )
    return {"boxes": boxes, "found": found, "n": len(scorable)}


def markdown_to_html(md: str) -> str:
    """Small Markdown renderer for the run's own tables: headings, tables, paragraphs, lists,
    `code`, **bold** and [links](url). HTML comments are dropped."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)

    def inline(text: str) -> str:
        t = html.escape(text, quote=False)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", t)
        return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', t)

    out, para, rows, items = [], [], [], []

    def flush() -> None:
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
        if rows:
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            body = [r for r in cells[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            head = "".join(f"<th>{inline(c)}</th>" for c in cells[0])
            trs = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body
            )
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{trs}</tbody></table>")
            rows.clear()

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("|"):
            if para or items:
                saved = list(rows)
                rows.clear()
                flush()
                rows.extend(saved)
            rows.append(s)
            continue
        if rows:
            flush()
        m = re.match(r"(#{1,4})\s+(.*)", s)
        if m:
            flush()
            level = min(len(m.group(1)) + 1, 5)  # page title is h1
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
        elif s.startswith(("- ", "* ")):
            if para:
                saved_items = list(items)
                items.clear()
                flush()
                items.extend(saved_items)
            items.append(s[2:])
        elif not s:
            flush()
        else:
            para.append(s)
    flush()
    return "\n".join(out)


def results_html(cfg: dict) -> str:
    """summary.md and the intro of tables/images_used.md of this run, rendered to HTML."""
    rd = run_dir(cfg)
    parts = []
    summary = rd / "summary.md"
    if summary.exists():
        parts.append(markdown_to_html(summary.read_text(encoding="utf-8")))
    used = rd / "tables" / "images_used.md"
    if used.exists():
        text = used.read_text(encoding="utf-8")
        intro = text.split("\n|", 1)[0]  # heading + totals paragraph, not the 544-row table
        parts.append(markdown_to_html(intro))
    if not parts:
        return "<p>No summary.md yet: run <code>evaluate</code> (and <code>report</code>).</p>"
    return "\n".join(parts)


def share_copy(src: Path, dst: Path, max_side: int, quality: int) -> None:
    """Downscaled JPEG copy for the shareable viewer. Boxes stay in original pixel coordinates;
    the page scales them with the SVG viewBox, so no coordinate change is needed."""
    from PIL import Image

    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(load_rgb(src))
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    im.save(dst, format="JPEG", quality=quality)


def build_data(cfg: dict, out_dir: Path, share: bool = False) -> dict:
    recs = load_records(cfg)
    iou_thr = cfg["eval"]["iou_threshold"]
    score_thr = cfg["eval"]["recall_score"]
    detectors = []
    dets = {}
    for name in cfg["detect"]["detectors"]:
        try:
            dets[name] = load_detections(cfg, name)
            detectors.append(name)
        except FileNotFoundError:
            continue

    plans = {}
    plan_file = work_dir(cfg) / "plans.jsonl"
    if plan_file.exists():
        plans = {str(p["source_id"]): p["category"] for p in read_jsonl(plan_file)}

    by_src: dict = defaultdict(dict)
    meta = {}
    root = out_dir.resolve()
    for r in recs:
        key = r.condition.value + (f"/{r.generator}" if r.generator else "")
        gt = ground_truth(r)
        if share:
            vcfg = cfg.get("viewer", {})
            rel = f"images/{r.image_id}.jpg"
            share_copy(
                Path(r.path),
                out_dir / rel,
                vcfg.get("share_max_side", 640),
                vcfg.get("share_quality", 80),
            )
        else:
            rel = os.path.relpath(Path(r.path).resolve(), root).replace(os.sep, "/")
        by_src[r.source_id][key] = {
            "src": quote(rel, safe="/._-"),
            "gt": gt,
            "det": {
                n: score_panel(dets[n].get(r.image_id, []), gt, iou_thr, score_thr)
                for n in detectors
            },
        }
        meta[r.source_id] = {
            "sid": r.source_id,
            "split": r.split,
            "w": r.width,
            "h": r.height,
            "plan": plans.get(str(r.source_id)),
        }

    def order(s):
        return (0, int(s)) if str(s).isdigit() else (1, str(s))

    images = [meta[s] | {"cond": by_src[s]} for s in sorted(by_src, key=order)]
    return {
        "run": cfg.get("run_name"),
        "detectors": detectors,
        "conditions": sorted({k for s in by_src.values() for k in s}),
        "score_threshold": score_thr,
        "iou_threshold": iou_thr,
        "results_html": results_html(cfg),
        "images": images,
    }


def run(cfg: dict, share: bool = False) -> dict:
    if share:
        out = Path(cfg.get("viewer", {}).get("share_dir", "docs/viewer")) / cfg.get(
            "run_name", "run"
        )
    else:
        out = run_dir(cfg) / "viewer"
    out.mkdir(parents=True, exist_ok=True)
    data = build_data(cfg, out, share=share)
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
        "shareable": share,
    }
    if share:
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        summary["size_mb"] = round(size / 2**20, 1)
    write_run_info(run_dir(cfg), "viewer", cfg, summary)
    return summary
