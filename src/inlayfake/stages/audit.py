"""Stage `audit`: candidate missing labels on C0 → <work>/audit_decisions.csv + audit.html.

Needs `detect` to have run on C0. Open audit.html, then edit the `decision` column of
audit_decisions.csv (add / ignore / reject). Later stages apply the decisions automatically.
Re-running the stage never overwrites a decisions file you have started editing.
"""

from __future__ import annotations

import html
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from inlayfake.data.audit import find_candidates, write_decisions_template
from inlayfake.edit.encode import load_rgb
from inlayfake.ledger import read_ledger
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import ledger_path, run_dir, work_dir
from inlayfake.stages.detect import load_detections


def _crop(img: np.ndarray, box: list[float], out_path, size: int = 220) -> None:
    H, W = img.shape[:2]
    x, y, w, h = box
    m = 0.6 * max(w, h) + 16
    x0, y0 = int(max(x - m, 0)), int(max(y - m, 0))
    x1, y1 = int(min(x + w + m, W)), int(min(y + h + m, H))
    pil = Image.fromarray(img[y0:y1, x0:x1].copy())
    ImageDraw.Draw(pil).rectangle(
        [x - x0, y - y0, x - x0 + w, y - y0 + h], outline=(255, 0, 0), width=3
    )
    pil.thumbnail((size, size))
    pil.save(out_path, quality=90)


def run(cfg: dict, review: bool = False, share: bool = False) -> dict:
    if review:  # only (re)build the review page from the current decisions
        return review_page(cfg, share=share)
    acfg = cfg["audit"]
    wd = work_dir(cfg)
    c0 = [r for r in read_ledger(ledger_path(cfg)) if r.condition.value == "C0"]
    dets = {name: load_detections(cfg, name) for name in cfg["detect"]["detectors"]}
    crops = wd / "audit_crops"
    crops.mkdir(exist_ok=True)
    rows, cards = [], []
    for rec in c0:
        cands = find_candidates(
            rec,
            {n: d.get(rec.image_id, []) for n, d in dets.items()},
            acfg["min_score"],
            acfg["min_agree"],
        )
        if not cands:
            continue
        img = load_rgb(rec.path)
        for k, c in enumerate(cands):
            cid = f"{rec.source_id}-{k}"
            rows.append(c | {"candidate_id": cid, "source_id": rec.source_id})
            _crop(img, c["bbox"], crops / f"{cid}.jpg")
            cards.append(
                f'<figure><img src="audit_crops/{cid}.jpg"><figcaption><b>{cid}</b> '
                f"{html.escape(c['category'])} · {c['n_detectors']} detectors · "
                f"{c['max_score']:.2f}</figcaption></figure>"
            )
    dec = wd / "audit_decisions.csv"
    if dec.exists():
        status = "kept existing audit_decisions.csv (not overwritten)"
    else:
        write_decisions_template(rows, dec)
        status = "wrote audit_decisions.csv (all decisions '?')"
    (wd / "audit_candidates.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (wd / "audit.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Label audit</title><style>body{font:13px "
        "sans-serif;margin:16px}main{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0;"
        "width:230px}</style><h2>Candidate missing labels (red box)</h2><p>For each id, set "
        "<code>decision</code> in audit_decisions.csv to <b>add</b> (real object COCO missed), "
        "<b>ignore</b> (unsure) or <b>reject</b> (detector mistake).</p><main>"
        + "".join(cards)
        + "</main>",
        encoding="utf-8",
    )
    summary = {
        "c0_images": len(c0),
        "candidates": len(rows),
        "images_with_candidates": len({r["source_id"] for r in rows}),
        "status": status,
    }
    write_run_info(run_dir(cfg), "audit", cfg, summary)
    return summary


DECISION_COLOR = {
    "add": "var(--tp)",
    "ignore": "var(--ign)",
    "reject": "var(--fp)",
    "?": "var(--mod)",
}
DECISION_TEXT = {
    "add": "add: a real object COCO missed (becomes a label)",
    "ignore": "ignore: unsure (neither rewarded nor penalised)",
    "reject": "reject: detector mistake (nothing changes)",
    "?": "? not decided (counts as ignore)",
}


def _svg_boxes(boxes: list[tuple[list[float], str, str, bool]], scale: float) -> str:
    """SVG rects (and labels) in image pixels; ``scale`` sets stroke and font size."""
    out = []
    sw, fs = max(1.5, scale / 220), max(9.0, scale / 30)
    for (x, y, w, h), color, label, dashed in boxes:
        dash = f' stroke-dasharray="{3 * sw:.1f},{2 * sw:.1f}"' if dashed else ""
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="{sw:.1f}"{dash}/>'
        )
        if label:
            ty = y - 3 if y - fs - 3 >= 0 else y + fs
            out.append(
                f'<text x="{x + 2:.1f}" y="{ty:.1f}" font-size="{fs:.1f}" fill="{color}" '
                f'paint-order="stroke" stroke="#000" stroke-width="{fs / 6:.1f}" '
                f'font-family="system-ui,sans-serif">{html.escape(label)}</text>'
            )
    return "".join(out)


def review_page(cfg: dict, share: bool = False) -> dict:
    """<work>/audit_review.html: every label-audit candidate in the order of
    audit_decisions.csv (card k = row k), with its current decision, a zoomed crop and the whole
    photo. Orange: the candidate (a confident detection that matches no label); green: the
    labels COCO already has, with their class. A Show dropdown filters by decision. Read-only:
    decisions are still changed in the CSV (author request 2026-10-18: re-check decisions).

    ``share=True`` (CLI ``audit --review --share``) writes the page as ``audit.html`` into the
    week's website folder (``inlayfake.site.share_dir``, docs/results/week1_real_baseline/) and
    uses the photo copies there (``images/<C0 image id>.jpg``, made by ``viewer --share``; any
    missing one is added), so it works on GitHub Pages. Rebuild it after changing a decision."""
    from inlayfake.data.audit import detector_matches, read_decisions
    from inlayfake.site import DETECTOR_NAMES, SITE_CSS, page_title
    from inlayfake.stages.common import decisions_path

    wd = work_dir(cfg)
    path, _ = decisions_path(cfg)
    decisions = read_decisions(path)
    c0 = {r.source_id: r for r in read_ledger(ledger_path(cfg)) if r.condition.value == "C0"}
    counts = Counter(d["decision"] for d in decisions)
    if share:
        from inlayfake.site import share_dir

        out_dir = share_dir(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
    names = cfg["detect"]["detectors"]
    dets = {}
    for n in names:  # saved C0 detections; a missing file shows as "not run"
        try:
            dets[n] = load_detections(cfg, n)
        except FileNotFoundError:
            dets[n] = None
    min_score = cfg["audit"]["min_score"]
    n_mismatch = 0
    cards = []
    for k, d in enumerate(decisions, 1):
        rec = c0.get(int(d["source_id"]))
        if rec is None:
            continue
        box = [float(d[c]) for c in ("x", "y", "w", "h")]
        W, H = rec.width, rec.height
        labels = [
            (i.bbox, "var(--gt)", i.category, False)
            for i in rec.instances
            if i.instance_id.startswith("coco:")
        ]
        cand = [(box, "var(--fp)", d["category"], False)]
        x, y, w, h = box
        m = 0.6 * max(w, h) + 16
        x0, y0 = max(x - m, 0), max(y - m, 0)
        x1, y1 = min(x + w + m, W), min(y + h + m, H)
        if share:
            name = f"images/{rec.image_id}.jpg"
            if not (out_dir / name).exists():
                from inlayfake.stages.viewer import share_copy

                vcfg = cfg.get("viewer", {})
                share_copy(
                    Path(rec.path),
                    out_dir / name,
                    vcfg.get("share_max_side", 640),
                    vcfg.get("share_quality", 80),
                )
            src = html.escape(name)
        else:
            src = html.escape(os.path.relpath(rec.path, wd).replace(os.sep, "/"))
        crop = (
            f'<svg viewBox="{x0:.1f} {y0:.1f} {x1 - x0:.1f} {y1 - y0:.1f}" class="zoom">'
            f'<image href="{src}" width="{W}" height="{H}"/>'
            f"{_svg_boxes(labels + cand, max(x1 - x0, y1 - y0))}</svg>"
        )
        whole = (
            f'<svg viewBox="0 0 {W} {H}" class="whole"><image href="{src}" width="{W}" '
            f'height="{H}"/>{_svg_boxes(labels + cand, max(W, H))}</svg>'
        )
        found = detector_matches(
            box, d["category"], {n: (v or {}).get(rec.image_id, []) for n, v in dets.items()}
        )
        cells = []
        for n in names:
            label = html.escape(DETECTOR_NAMES.get(n, n))
            sc = found.get(n)
            if dets[n] is None:
                cells.append(f'<span class="det no">{label}: not run</span>')
            elif sc is not None and sc >= min_score:
                cells.append(f'<span class="det yes">✓ {label} {sc:.2f}</span>')
            elif sc is not None:
                cells.append(f'<span class="det low">{label} {sc:.2f} (below {min_score})</span>')
            else:
                cells.append(f'<span class="det no">{label}: no box here</span>')
        agree = sum(1 for n in names if (found.get(n) or 0) >= min_score)
        n_mismatch += agree != int(d["n_detectors"])
        dec = d["decision"]
        cards.append(
            f'<section class="card" data-dec="{html.escape(dec)}"><h2>{k}. '
            f"{html.escape(d['candidate_id'])} <span>· candidate: {html.escape(d['category'])} · "
            f"{d['n_detectors']} detectors agree · score {float(d['max_score']):.2f} · COCO "
            f'{rec.source_id}</span></h2><p class="dec" style="border-color:'
            f'{DECISION_COLOR.get(dec, "var(--line)")}">current decision: <b>'
            f"{html.escape(DECISION_TEXT.get(dec, dec))}</b></p>"
            f'<p class="dets">Detectors ({html.escape(d["category"])} box at IoU ≥ 0.5): '
            f"{''.join(cells)}</p>"
            f'<div class="row"><div class="panel"><div class="t"><b>Zoom</b><i>around the '
            f'candidate</i></div><div class="wrap">{crop}</div></div><div class="panel">'
            f'<div class="t"><b>Whole photo</b><i>{W}×{H}</i></div><div class="wrap">{whole}'
            "</div></div></div></section>"
        )
    options = [("all", f"all candidates ({len(decisions)})")] + [
        (v, f"{v} ({counts[v]})") for v in ("add", "ignore", "reject", "?") if counts[v]
    ]
    select = "".join(f'<option value="{v}">{html.escape(t)}</option>' for v, t in options)
    title = page_title(cfg, "Label audit") + " · label audit review"
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{SITE_CSS}"
        ".wrap svg{position:static;width:100%;height:auto;display:block;border-radius:4px}"
        ".dec{margin:0 0 8px;padding:4px 8px;border-left:4px solid;background:var(--bg)}"
        ".dets{margin:0 0 8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;"
        "color:var(--muted)}.det{padding:1px 8px;border-radius:10px;border:1px solid var(--line)}"
        ".det.yes{color:var(--ink);border-color:var(--tp);font-weight:600}"
        ".det.low{color:var(--ink)}.det.no{opacity:.7}"
        "</style></head><body>"
        f'<header><h1>{html.escape(title)}</h1><label>Show <select id="f">{select}'
        '</select></label><span class="muted" id="n"></span></header><main>'
        '<p class="note">Each card is one row of <code>'
        f"{html.escape(path.as_posix())}</code> (card k = row k after the header). A candidate "
        "is a confident detection (score ≥ 0.7, at least 2 of 4 detectors agree) that matches no "
        "COCO label. To change a decision, edit the <code>decision</code> column of that row "
        "(add / ignore / reject) and save; then see the README (re-run evaluate and report).</p>"
        '<div class="legend"><span><i class="sw" style="border-color:var(--fp)"></i>candidate '
        '(possible missing label)</span><span><i class="sw" style="border-color:var(--gt)">'
        "</i>label COCO already has</span></div>"
        + "".join(cards)
        + "<script>const sel=document.getElementById('f'),n=document.getElementById('n');"
        "function f(){const v=sel.value;let k=0;document.querySelectorAll('section.card')"
        ".forEach(s=>{const show=v==='all'||s.dataset.dec===v;s.style.display=show?'':'none';"
        "k+=show});n.textContent=k+' shown'}sel.onchange=f;f()</script></main></body></html>"
    )
    out = out_dir / "audit.html" if share else wd / "audit_review.html"
    out.write_text(page, encoding="utf-8")
    return {
        "review": out.as_posix(),
        "candidates": len(cards),
        "decisions": dict(counts),
        # recomputed agreement differs from the stored n_detectors (should be 0)
        "detector_count_mismatches": n_mismatch,
    }
