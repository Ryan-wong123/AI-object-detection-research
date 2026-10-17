"""Stage `preview`: HTML contact sheet of the conditions per source, for visual checks.

Open <work>/preview.html in a browser. Planned boxes are drawn in yellow, inserted/pasted
instances in red. Images are referenced by relative path (nothing is copied).

Default: only the images that are in the benchmark (``ledger.jsonl``): C0, C1, and the C2-C5
edits that passed `verify`, with their verified boxes. Each photo's heading names the edits it
does not have (yet).

``all_attempts=True`` (CLI ``--all-attempts``) shows every generated attempt of C2/C4/C5 with its
verification status, so that a failed attempt (e.g. SDXL painted background) is never mistaken
for a benchmark image:
  green "in benchmark"   passed `verify`; the red box is the verified (SAM) object box
  grey "rejected"        failed `verify`; dimmed, not in ledger.jsonl (regenerated or dropped)
  "not verified yet"     `verify` has not run on it
"""

from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from pathlib import Path

from inlayfake.stages.common import read_jsonl, work_dir

ORDER = ["C0", "C1", "C2", "C3", "C4", "C5"]
EDITED = ("C2", "C4", "C5")


def _status(
    rec: dict, verified: dict[str, dict], passed_keys: set[str], final_ids: set[str]
) -> tuple[str, str]:
    """(css class, caption suffix) for one record."""
    if rec["condition"] == "C3" and final_ids and rec["image_id"] not in final_ids:
        return "rej", "superseded: the box was re-planned · not in benchmark"
    if rec["condition"] not in EDITED:
        return "", ""
    row = verified.get(rec["image_id"])
    attempt = f"attempt {rec['edit'].get('attempts', 1)}"
    if rec["edit"].get("plan_version"):
        attempt += f" · box {rec['edit']['plan_version']}"
    if row is None:
        key = f"{rec['source_id']}|{rec['condition']}|{rec['generator']}"
        if key in passed_keys:
            return "rej", f"not needed: an earlier attempt passed · {attempt} · not in benchmark"
        return "pending", f"not verified yet · {attempt}"
    if row["passed"]:
        return "ok", f"in benchmark · {attempt}"
    return "rej", f"rejected: {row['reason']} · {attempt} · not in benchmark"


def _panel(rec: dict, wd: Path, thumb: int, css: str, note: str) -> str:
    scale = thumb / rec["width"]
    rel = os.path.relpath(rec["path"], wd).replace(os.sep, "/")
    boxes = []
    if rec.get("edit"):
        boxes.append((rec["edit"]["plan_box"], "#fc0"))
    for inst in rec["instances"]:
        if inst["origin"] in ("ai_inserted", "real_pasted") and css != "rej":
            boxes.append((inst["bbox"], "#f33"))
    divs = "".join(
        f'<div class="b" style="left:{b[0] * scale:.0f}px;top:{b[1] * scale:.0f}px;'
        f'width:{b[2] * scale:.0f}px;height:{b[3] * scale:.0f}px;border-color:{c}"></div>'
        for b, c in boxes
    )
    label = f"{rec['condition']} {rec.get('generator') or ''}"
    extra = f'<br><span class="s">{html.escape(note)}</span>' if note else ""
    return (
        f'<figure class="{css}"><div class="w"><img src="{html.escape(rel)}" width="{thumb}" '
        f'loading="lazy">'
        f"{divs}</div><figcaption>{html.escape(label)}{extra}</figcaption></figure>"
    )


STYLE = (
    "<!doctype html><meta charset=utf-8><title>InlayFake preview</title><style>"
    "body{font:13px sans-serif;margin:16px}section{display:flex;flex-wrap:wrap;gap:6px;"
    "margin-bottom:18px}h3{width:100%;margin:4px 0}h3 small{font-weight:400;color:#777}"
    "figure{margin:0;max-width:240px}"
    ".w{position:relative}.b{position:absolute;border:2px solid}"
    "figcaption{text-align:center}.s{font-size:11px}.ok .s{color:#080;font-weight:600}"
    ".rej img{opacity:.4}.rej .s{color:#777}.pending .s{color:#b60}</style>"
)


def _benchmark_only(cfg: dict, wd: Path, limit: int | None, thumb: int) -> dict:
    """Only the images in the final ledger: what week 3 evaluates."""
    lpath = wd / "ledger.jsonl"
    recs = read_jsonl(lpath) if lpath.exists() else []
    by_src: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        by_src[r["source_id"]].append(r)
    plans = wd / "plans.jsonl"
    planned = {p["source_id"]: p for p in read_jsonl(plans)} if plans.exists() else {}
    rows, n_edits = [], 0
    for sid in sorted(set(by_src) | set(planned))[:limit]:
        have = sorted(
            by_src.get(sid, []), key=lambda r: (ORDER.index(r["condition"]), r["generator"] or "")
        )
        conds = {r["condition"] for r in have}
        p = planned.get(sid, {})
        expected = []
        if p.get("box_free") is not None:
            expected += ["C2", "C3", "C4"]
        if p.get("box_occ") is not None:
            expected.append("C5")
        missing = [c for c in expected if c not in conds]
        cat = next(
            (r["edit"]["target_category"] for r in have if r.get("edit")), p.get("category", "")
        )
        panels = []
        for r in have:
            note = ""
            if r["condition"] in EDITED:
                n_edits += 1
                note = f"attempt {r['edit'].get('attempts', 1)}"
                if r["edit"].get("plan_version"):
                    note += f" · box {r['edit']['plan_version']}"
                if not r["edit"].get("class_in_photo", True):
                    note += " · class not in photo"
                if r["edit"].get("placement_rule", "standard") != "standard":
                    note += " · placed anywhere"
            panels.append(_panel(r, wd, thumb, "ok" if note else "", note))
        miss = f" <small>· not in benchmark (yet): {', '.join(missing)}</small>" if missing else ""
        rows.append(
            f"<section><h3>source {sid} · {html.escape(cat)}{miss}</h3>{''.join(panels)}</section>"
        )
    page = (
        STYLE
        + "<p>Only images in the benchmark (<code>ledger.jsonl</code>): the original (C0), the "
        "re-save (C1), and the edits that passed the check. Yellow: planned box. Red: inserted "
        "or pasted object (verified box). Rejected attempts are not shown; build with "
        "<code>--all-attempts</code> to see them.</p>"
        + ("" if lpath.exists() else "<p><b>No edits checked yet.</b></p>")
        + "".join(rows)
    )
    out = wd / "preview.html"
    out.write_text(page, encoding="utf-8")
    return {"preview": out.as_posix(), "sources": len(rows), "edits_in_benchmark": n_edits}


def run(cfg: dict, limit: int | None = 50, thumb: int = 240, all_attempts: bool = False) -> dict:
    wd = work_dir(cfg)
    if not all_attempts:
        return _benchmark_only(cfg, wd, limit, thumb)
    vpath = wd / "verify.jsonl"
    verified = {r["image_id"]: r for r in read_jsonl(vpath)} if vpath.exists() else {}
    passed_keys = {r["key"] for r in verified.values() if r["passed"]}
    lpath = wd / "ledger.jsonl"
    final_ids = {r["image_id"] for r in read_jsonl(lpath)} if lpath.exists() else set()
    by_src: dict[int, list[dict]] = defaultdict(list)
    for line in (wd / "ledger_generated.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            vp = wd / "verified" / f"{r['image_id']}.json"
            if r["image_id"] in verified and verified[r["image_id"]]["passed"] and vp.exists():
                r = json.loads(vp.read_text(encoding="utf-8"))  # verified boxes
            by_src[r["source_id"]].append(r)
    rows = []
    for sid in sorted(by_src)[:limit]:
        recs = sorted(
            by_src[sid],
            key=lambda r: (
                ORDER.index(r["condition"]),
                r["generator"] or "",
                (r.get("edit") or {}).get("attempts", 0),
            ),
        )
        cat = next((r["edit"]["target_category"] for r in recs if r.get("edit")), "")
        panels = "".join(
            _panel(r, wd, thumb, *_status(r, verified, passed_keys, final_ids)) for r in recs
        )
        rows.append(f"<section><h3>source {sid} · {html.escape(cat)}</h3>{panels}</section>")
    page = (
        STYLE + "<p>All attempts. Yellow: planned box. Red: inserted or pasted object "
        "(verified box for passed edits). "
        "Only edits marked <b style=color:#080>in benchmark</b> enter ledger.jsonl; greyed "
        "attempts were rejected by the verifier and regenerated with a new seed or dropped.</p>"
        + "".join(rows)
    )
    out = wd / "preview.html"
    out.write_text(page, encoding="utf-8")
    n_ok = sum(1 for r in verified.values() if r["passed"])
    return {"preview": out.as_posix(), "sources": len(rows), "edits_in_benchmark": n_ok}
