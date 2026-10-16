"""Stage `audit`: candidate missing labels on C0 → <work>/audit_decisions.csv + audit.html.

Needs `detect` to have run on C0. Open audit.html, then edit the `decision` column of
audit_decisions.csv (add / ignore / reject). Later stages apply the decisions automatically.
Re-running the stage never overwrites a decisions file you have started editing.
"""

from __future__ import annotations

import html
import json

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


def run(cfg: dict) -> dict:
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
