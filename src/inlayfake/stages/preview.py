"""Stage `preview`: HTML contact sheet of the conditions per source, for visual checks.

Open <work>/preview.html in a browser. Planned boxes are drawn in yellow, inserted/pasted
instances in red. Images are referenced by relative path (nothing is copied).
"""

from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from pathlib import Path

from inlayfake.stages.common import work_dir

ORDER = ["C0", "C1", "C2", "C3", "C4", "C5"]


def _panel(rec: dict, wd: Path, thumb: int) -> str:
    scale = thumb / rec["width"]
    rel = os.path.relpath(rec["path"], wd).replace(os.sep, "/")
    boxes = []
    if rec.get("edit"):
        boxes.append((rec["edit"]["plan_box"], "#fc0"))
    for inst in rec["instances"]:
        if inst["origin"] in ("ai_inserted", "real_pasted"):
            boxes.append((inst["bbox"], "#f33"))
    divs = "".join(
        f'<div class="b" style="left:{b[0] * scale:.0f}px;top:{b[1] * scale:.0f}px;'
        f'width:{b[2] * scale:.0f}px;height:{b[3] * scale:.0f}px;border-color:{c}"></div>'
        for b, c in boxes
    )
    label = f"{rec['condition']} {rec.get('generator') or ''}"
    return (
        f'<figure><div class="w"><img src="{html.escape(rel)}" width="{thumb}">{divs}</div>'
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
    )


def run(cfg: dict, limit: int | None = 50, thumb: int = 240) -> dict:
    wd = work_dir(cfg)
    by_src: dict[int, list[dict]] = defaultdict(list)
    for line in (wd / "ledger_generated.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_src[r["source_id"]].append(r)
    rows = []
    for sid in sorted(by_src)[:limit]:
        recs = sorted(
            by_src[sid], key=lambda r: (ORDER.index(r["condition"]), r["generator"] or "")
        )
        cat = next((r["edit"]["target_category"] for r in recs if r.get("edit")), "")
        panels = "".join(_panel(r, wd, thumb) for r in recs)
        rows.append(f"<section><h3>source {sid} · {html.escape(cat)}</h3>{panels}</section>")
    page = (
        "<!doctype html><meta charset=utf-8><title>InlayFake preview</title><style>"
        "body{font:13px sans-serif;margin:16px}section{display:flex;flex-wrap:wrap;gap:6px;"
        "margin-bottom:18px}h3{width:100%;margin:4px 0}figure{margin:0}.w{position:relative}"
        ".b{position:absolute;border:2px solid}figcaption{text-align:center}</style>"
        + "".join(rows)
    )
    out = wd / "preview.html"
    out.write_text(page, encoding="utf-8")
    return {"preview": out.as_posix(), "sources": len(rows)}
