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
    ".rej img{opacity:.4}.rej .s{color:#777}.pending .s{color:#b60}.flag .s{color:#b60}</style>"
)


REASON = {
    "unplanned": "not planned (no room for its box)",
    "retrying": "still retrying",
    "dropped": "dropped (every allowed attempt failed)",
}

# (value, label) of the "Show" dropdown; the label explains the reason
FILTERS = [
    ("all", "all photos"),
    ("complete", "all conditions C0–C5 in the benchmark"),
    ("any", "any missing C (not passed yet, dropped, or not planned)"),
    ("planned", "photos missing a planned edit (planned, not in the benchmark yet)"),
    ("c4-unplanned", "C4 not planned: no free space for a box that touches no real object"),
    ("c4-retrying", "C4 still retrying: planned, no attempt passed yet"),
    ("c4-dropped", "C4 dropped: planned, every allowed attempt failed or no box fits"),
    ("c5-unplanned", "C5 not planned: no real object it can cover 20–50% of"),
    ("c5-retrying", "C5 still retrying: planned, no attempt passed yet"),
    ("c5-dropped", "C5 dropped: planned, every allowed attempt failed or no box fits"),
]


def _edit_status(cfg, cond, sid, p, conds, state, versions) -> str:
    """passed / retrying / dropped / unplanned for one condition of one photo."""
    from inlayfake.stages.replan import next_job

    box = p.get("box_occ") if cond == "C5" else p.get("box_free")
    if box is None:
        return "unplanned"
    if cond in conds:
        return "passed"
    if cond == "C3":  # no generator: it follows the C4 box; missing only while C4 moves
        c4 = _edit_status(cfg, "C4", sid, p, conds, state, versions)
        return "retrying" if c4 == "retrying" else "dropped"
    roles = cfg["generation"]
    key = {"C2": "sham_generators", "C4": "insert_generators", "C5": "occlusion_generators"}
    enabled = cfg.get("generators_enabled")
    gens = [g for g in roles[key[cond]] if enabled is None or g in enabled]
    if gens and next_job(cfg, state, sid, cond, gens[0], versions) is not None:
        return "retrying"
    return "dropped"


# merged AI condition (placement.merged): one AI edit per photo, shown as C4
SOURCE_LABEL = {
    "C4": "from C4: free space",
    "C5": "from C5: covers a real object",
    "anywhere": "new: placed anywhere",
}
MERGED_REASON = REASON | {"waiting": "waits for the AI edit to be verified"}
MERGED_FILTERS = [
    ("all", "all photos"),
    ("complete", "all conditions C0–C4 in the benchmark"),
    ("missing", "any missing C (AI edit, C2 or C3)"),
    ("ai-C4", "AI edit from the verified C4 (free space)"),
    ("ai-C5", "AI edit from the verified C5 (covers a real object)"),
    ("ai-anywhere", "AI edit newly placed anywhere"),
    ("ai-retrying", "AI edit still retrying (placed anywhere, no attempt passed yet)"),
    ("ai-dropped", "AI edit dropped: every allowed attempt failed"),
    ("c2-missing", "C2 sham missing (retrying, waiting or dropped)"),
    ("c3-missing", "C3 paste missing"),
]


def _merged_status(cfg, sid, conds, state, versions) -> tuple[str | None, dict[str, str]]:
    """(AI source, {C4/C2/C3: passed / retrying / waiting / dropped}) of one photo."""
    from inlayfake.stages.replan import gen_of, merged_source, next_job

    src = merged_source(cfg, state, sid, versions)
    st = {}
    if src is not None:
        st["C4"] = "passed"
    elif next_job(cfg, state, sid, "C4", gen_of(cfg, "C4"), versions) is not None:
        st["C4"] = "retrying"
    else:
        st["C4"] = "dropped"
    if "C2" in conds:
        st["C2"] = "passed"
    elif next_job(cfg, state, sid, "C2", gen_of(cfg, "C2"), versions) is not None:
        st["C2"] = "retrying"
    else:
        st["C2"] = "waiting" if st["C4"] == "retrying" else "dropped"
    if "C3" in conds:
        st["C3"] = "passed"
    elif st["C4"] == "retrying":
        st["C3"] = "waiting"
    else:  # made together with C2 at the chosen box; missing for good if no donor fits
        st["C3"] = "retrying" if st["C2"] == "retrying" else "dropped"
    return src, st


def _merged_only(cfg: dict, wd: Path, limit: int | None, thumb: int) -> dict:
    """Benchmark view of the merged AI condition: C0, C1, C2, C3 and one AI edit (C4)."""
    from inlayfake.stages.replan import load_versions
    from inlayfake.stages.verify import verify_state

    lpath = wd / "ledger.jsonl"
    recs = read_jsonl(lpath) if lpath.exists() else []
    by_src: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        by_src[r["source_id"]].append(r)
    plans = wd / "plans.jsonl"
    planned = {p["source_id"]: p for p in read_jsonl(plans)} if plans.exists() else {}
    state, versions = verify_state(wd), load_versions(wd)
    rows, n_edits = [], 0
    tally: dict[str, int] = defaultdict(int)
    for sid in sorted(set(by_src) | set(planned))[:limit]:
        have = sorted(
            by_src.get(sid, []), key=lambda r: (ORDER.index(r["condition"]), r["generator"] or "")
        )
        conds = {r["condition"] for r in have}
        src, status = _merged_status(cfg, sid, conds, state, versions)
        ai = next((r for r in have if r["condition"] == "C4"), None)
        cat = ai["edit"]["target_category"] if ai else planned.get(sid, {}).get("category", "")
        panels = []
        for r in have:
            note, css = "", ""
            if r["condition"] in EDITED:
                n_edits += 1
            if r["condition"] == "C4":
                note = "AI · " + SOURCE_LABEL.get(r["edit"].get("merged_source") or "", "")
                if not r["edit"].get("class_in_photo", True):
                    note += " · class not in photo"
                    css = "flag"
            panels.append(_panel(r, wd, thumb, css, note))
        groups: dict[str, list[str]] = defaultdict(list)
        for c in ("C4", "C2", "C3"):
            if status[c] != "passed":
                groups[status[c]].append("AI edit (C4)" if c == "C4" else c)
        miss = "".join(
            f" <small>· {', '.join(groups[s])}: {MERGED_REASON[s]}</small>"
            for s in ("retrying", "waiting", "dropped")
            if groups[s]
        )
        complete = all(v == "passed" for v in status.values())
        tags = {
            "complete": complete,
            "missing": not complete,
            f"ai-{src}": src is not None,
            f"ai-{status['C4']}": status["C4"] != "passed",
            "c2-missing": status["C2"] != "passed",
            "c3-missing": status["C3"] != "passed",
        }
        for t, on in tags.items():
            tally[t] += on
        rows.append(
            f'<section data-tags="{" ".join(t for t, on in tags.items() if on)}">'
            f"<h3>source {sid} · {html.escape(cat)}{miss}</h3>{''.join(panels)}</section>"
        )
    tally["all"] = len(rows)
    options = "".join(
        f'<option value="{v}"{" selected" if v == "all" else ""}>'
        f"{html.escape(label)} ({tally[v]})</option>"
        for v, label in MERGED_FILTERS
    )
    controls = (
        f"<p>{len(rows)} photos: <b>{tally['complete']}</b> have C0–C4 in the benchmark, "
        f"<b>{tally['missing']}</b> still miss the AI edit, C2 or C3. AI edit from C4: "
        f"{tally['ai-C4']}, from C5: {tally['ai-C5']}, newly placed anywhere: "
        f"{tally['ai-anywhere']}.</p>"
        f'<p><label>Show: <select id="f">{options}</select></label> '
        f'<b id="n">{len(rows)} photos shown</b></p>'
        "<script>const sel=document.getElementById('f');sel.onchange=()=>{const v=sel.value;"
        "let k=0;document.querySelectorAll('section').forEach(s=>{"
        "const show=v==='all'||s.dataset.tags.split(' ').includes(v);"
        "s.style.display=show?'':'none';k+=show});"
        "document.getElementById('n').textContent=k+' photos shown'}</script>"
    )
    page = (
        STYLE
        + "<p>Only images in the benchmark (<code>ledger.jsonl</code>), merged AI condition: the "
        "original (C0), the re-save (C1), the sham (C2), the real paste (C3) and <b>one AI edit "
        "(C4)</b> of a class already in the photo. A photo keeps its verified C4 (free space), "
        "else its verified C5 (covers a real object), else gets an AI object placed anywhere; "
        "C2 and C3 sit at the AI edit's box. Yellow: planned box. Red: inserted or pasted object "
        "(verified box).</p>"
        + ("" if lpath.exists() else "<p><b>No edits checked yet.</b></p>")
        + controls
        + "".join(rows)
    )
    out = wd / "preview.html"
    out.write_text(page, encoding="utf-8")
    return {
        "preview": out.as_posix(),
        "sources": len(rows),
        "edits_in_benchmark": n_edits,
        "sources_complete": tally["complete"],
        "sources_missing": tally["missing"],
        "ai_source": {k: tally[f"ai-{k}"] for k in ("C4", "C5", "anywhere")},
        "ai_retrying": tally["ai-retrying"],
        "ai_dropped": tally["ai-dropped"],
        "c2_missing": tally["c2-missing"],
        "c3_missing": tally["c3-missing"],
    }


def _benchmark_only(cfg: dict, wd: Path, limit: int | None, thumb: int) -> dict:
    """Only the images in the final ledger: what week 3 evaluates."""
    from inlayfake.stages.replan import load_versions
    from inlayfake.stages.verify import verify_state

    lpath = wd / "ledger.jsonl"
    recs = read_jsonl(lpath) if lpath.exists() else []
    by_src: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        by_src[r["source_id"]].append(r)
    plans = wd / "plans.jsonl"
    planned = {p["source_id"]: p for p in read_jsonl(plans)} if plans.exists() else {}
    state, versions = verify_state(wd), load_versions(wd)
    rows, n_edits, n_missing = [], 0, 0
    tally: dict[str, int] = defaultdict(int)
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
                # only the last-resort flags are shown (author: no "attempt n" wording)
                flags = []
                if not r["edit"].get("class_in_photo", True):
                    flags.append("class not in photo")
                if r["edit"].get("placement_rule", "standard") != "standard":
                    flags.append("placed anywhere")
                note = " · ".join(flags)
            panels.append(_panel(r, wd, thumb, "flag" if note else "", note))
        status = {
            c: _edit_status(cfg, c, sid, p, conds, state, versions)
            for c in ("C2", "C3", "C4", "C5")
        }
        not_planned = [c for c, s in status.items() if s == "unplanned"]
        # heading note: every condition the photo lacks, grouped by reason
        groups: dict[str, list[str]] = defaultdict(list)
        for c, s in status.items():
            if s != "passed":
                groups[s].append(c)
        miss = "".join(
            f" <small>· {', '.join(groups[s])}: {REASON[s]}</small>"
            for s in ("retrying", "dropped", "unplanned")
            if groups[s]
        )
        n_missing += bool(missing)
        complete = all(s == "passed" for s in status.values())
        tags = {
            "missing": bool(missing),
            "unplanned": bool(not_planned),
            "complete": complete,
            f"c4-{status['C4']}": True,
            f"c5-{status['C5']}": True,
        }
        for t, on in tags.items():
            tally[t] += on
        rows.append(
            f'<section data-missing="{int(bool(missing))}" '
            f'data-unplanned="{int(bool(not_planned))}" '
            f'data-tags="{" ".join(t for t, on in tags.items() if on)}">'
            f"<h3>source {sid} · {html.escape(cat)}{miss}</h3>{''.join(panels)}</section>"
        )
    n_unplanned = tally["unplanned"]
    options = "".join(
        f'<option value="{v}"{" selected" if v == "all" else ""}>{html.escape(label)}</option>'
        for v, label in FILTERS
    )
    controls = (
        f"<p>{len(rows)} photos: <b>{tally['complete']}</b> have all of C0–C5 in the benchmark, "
        f"<b>{n_missing}</b> are still missing a planned edit, and {n_unplanned} have a "
        "condition that was never planned (no room for its box).</p>"
        f'<p><label>Show: <select id="f">{options}</select></label> '
        f'<b id="n">{len(rows)} photos shown</b></p>'
        "<script>const sel=document.getElementById('f');sel.onchange=()=>{const v=sel.value;"
        "let k=0;document.querySelectorAll('section').forEach(s=>{"
        "const t=s.dataset.tags.split(' '),m=t.includes('missing'),u=t.includes('unplanned');"
        "const show=v==='all'||(v==='any'&&(m||u))||(v==='planned'&&m)||t.includes(v);"
        "s.style.display=show?'':'none';k+=show});"
        "document.getElementById('n').textContent=k+' photos shown'}</script>"
    )
    page = (
        STYLE
        + "<p>Only images in the benchmark (<code>ledger.jsonl</code>): the original (C0), the "
        "re-save (C1), and the edits that passed the check. Yellow: planned box. Red: inserted "
        "or pasted object (verified box). Rejected attempts are not shown; build with "
        "<code>--all-attempts</code> to see them.</p>"
        + ("" if lpath.exists() else "<p><b>No edits checked yet.</b></p>")
        + controls
        + "".join(rows)
    )
    out = wd / "preview.html"
    out.write_text(page, encoding="utf-8")
    return {
        "preview": out.as_posix(),
        "sources": len(rows),
        "edits_in_benchmark": n_edits,
        "sources_missing_a_planned_edit": n_missing,
        "sources_with_an_unplanned_condition": n_unplanned,
        "sources_complete": tally["complete"],
        "c4": {s: tally[f"c4-{s}"] for s in ("passed", "retrying", "dropped", "unplanned")},
        "c5": {s: tally[f"c5-{s}"] for s in ("passed", "retrying", "dropped", "unplanned")},
    }


def run(cfg: dict, limit: int | None = 50, thumb: int = 240, all_attempts: bool = False) -> dict:
    wd = work_dir(cfg)
    if not all_attempts:
        from inlayfake.stages.replan import merged

        if merged(cfg):
            return _merged_only(cfg, wd, limit, thumb)
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
