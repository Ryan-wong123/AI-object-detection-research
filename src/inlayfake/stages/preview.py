"""Stage `preview`: HTML contact sheet of the conditions per source, for visual checks.

Open <work>/preview.html in a browser. Planned boxes are drawn dashed yellow, inserted/pasted
instances purple (shared site style, ``inlayfake.site``, as on every week's results page).
Images are referenced by relative path (nothing is copied); ``--share`` writes a self-contained
copy to docs/results/week<N>_<topic>/ for the results website.

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

from inlayfake.site import CONDITION_NAMES, SITE_CSS, page_title, share_dir
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
    """One image as on every results page: caption, photo, SVG boxes in image pixels (dashed
    yellow: planned box; purple: inserted or pasted object)."""
    rel = os.path.relpath(rec["path"], wd).replace(os.sep, "/")
    W, H = rec["width"], rec["height"]
    sw = max(1.5, W / 320)
    rects = []
    if rec.get("edit"):
        x, y, w, h = rec["edit"]["plan_box"]
        rects.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" '
            f'stroke="var(--plan)" stroke-width="{sw:.1f}" stroke-dasharray="{3 * sw:.1f},'
            f'{2 * sw:.1f}"/>'
        )
    for inst in rec["instances"]:
        if inst["origin"] in ("ai_inserted", "real_pasted") and css != "rej":
            x, y, w, h = inst["bbox"]
            rects.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" '
                f'stroke="var(--ins)" stroke-width="{sw:.1f}"/>'
            )
    label = CONDITION_NAMES.get(rec["condition"], rec["condition"])
    return (
        f'<div class="panel {css}"><div class="t"><b>{html.escape(label)}</b>'
        f"<i>{html.escape(note)}</i></div>"
        f'<div class="wrap"><img loading="lazy" src="{html.escape(rel)}" '
        f'alt="{html.escape(label)}, photo {rec["source_id"]}">'
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">{"".join(rects)}</svg></div></div>'
    )


PREVIEW_CSS = (
    ".panel.ok .t i{color:var(--tp);font-weight:600}.panel.rej img{opacity:.4}"
    ".panel.pending .t i,.panel.flag .t i{color:var(--mod)}"
)

LEGEND = (
    '<div class="legend"><span><i class="sw" style="border-color:var(--plan);'
    'border-style:dashed"></i>planned box</span><span><i class="sw" '
    'style="border-color:var(--ins)"></i>inserted or pasted object (verified box)</span></div>'
)


def _page(cfg: dict, controls: str, body: str) -> str:
    """Full page in the shared site style: sticky header (title, run, controls) and main."""
    title = page_title(cfg, "Benchmark preview")
    run = html.escape(cfg.get("run_name", ""))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{SITE_CSS}{PREVIEW_CSS}</style></head><body>"
        f"<header><h1>{html.escape(title)} <span>· {run}</span></h1>{controls}</header>"
        f"<main>{body}</main></body></html>"
    )


def _card(sid, cat: str, miss: str, panels: str, attrs: str = "") -> str:
    return (
        f'<section class="card"{attrs}><h2>source {sid} <span>· {html.escape(cat)}{miss}</span>'
        f'</h2><div class="row">{panels}</div></section>'
    )


FILTER_JS = (
    "<script>const sel=document.getElementById('f');sel.onchange=()=>{const v=sel.value;"
    "let k=0;document.querySelectorAll('section.card').forEach(s=>{{TEST}"
    "s.style.display=show?'':'none';k+=show});"
    "document.getElementById('n').textContent=k+' photos shown'}</script>"
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


# merged AI condition (placement.merged): one AI edit per photo, shown as plain C4 (author,
# 2026-10-18: no placement wording on the page, three Show options)
MERGED_REASON = REASON | {"waiting": "waits for the AI edit to be verified"}
MERGED_FILTERS = [
    ("all", "all photos"),
    ("complete", "all conditions C0–C4 in the benchmark"),
    ("missing", "any missing C (C2, C3 or C4)"),
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
            if r["condition"] in EDITED:
                n_edits += 1
            panels.append(_panel(r, wd, thumb, "", ""))
        groups: dict[str, list[str]] = defaultdict(list)
        for c in ("C4", "C2", "C3"):
            if status[c] != "passed":
                groups[status[c]].append(c)
        miss = "".join(
            f" · {', '.join(groups[s])}: {MERGED_REASON[s]}"
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
        attrs = f' data-tags="{" ".join(t for t, on in tags.items() if on)}"'
        rows.append(_card(sid, cat, miss, "".join(panels), attrs))
    tally["all"] = len(rows)
    options = "".join(
        f'<option value="{v}"{" selected" if v == "all" else ""}>'
        f"{html.escape(label)} ({tally[v]})</option>"
        for v, label in MERGED_FILTERS
    )
    controls = (
        f'<label>Show <select id="f">{options}</select></label>'
        f'<span class="muted" id="n">{len(rows)} photos shown</span>'
    )
    body = (
        '<p class="note">Only images in the benchmark (<code>ledger.jsonl</code>): the original '
        "(C0), the re-save (C1), the sham edit (C2), the real paste (C3) and the AI insertion (C4) "
        "of a class already in the photo. C2 and C3 sit at the C4 box. "
        f"{len(rows)} photos: <b>{tally['complete']}</b> have C0–C4 in the benchmark, "
        f"<b>{tally['missing']}</b> still miss C2, C3 or C4.</p>"
        + ("" if lpath.exists() else '<p class="note"><b>No edits checked yet.</b></p>')
        + LEGEND
        + "".join(rows)
        + FILTER_JS.replace(
            "{TEST}", "const show=v==='all'||s.dataset.tags.split(' ').includes(v);"
        )
    )
    page = _page(cfg, controls, body)
    out = wd / "preview.html"
    out.write_text(page, encoding="utf-8")
    return {
        "preview": out.as_posix(),
        "sources": len(rows),
        "edits_in_benchmark": n_edits,
        "sources_complete": tally["complete"],
        "sources_missing": tally["missing"],
        "ai_source": {k: tally[f"ai-{k}"] for k in ("free", "occluding", "anywhere")},
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
                if r["edit"].get("placement_rule", "standard") == "anywhere":
                    flags.append("placed anywhere")
                note = " · ".join(flags)
            panels.append(_panel(r, wd, thumb, "flag" if note else "", note))
        status = {
            c: _edit_status(cfg, c, sid, p, conds, state, versions)
            for c in ("C2", "C3", "C4", "C5")
            if c in cfg["conditions"]  # natural placement has no C5
        }
        not_planned = [c for c, s in status.items() if s == "unplanned"]
        # heading note: every condition the photo lacks, grouped by reason
        groups: dict[str, list[str]] = defaultdict(list)
        for c, s in status.items():
            if s != "passed":
                groups[s].append(c)
        miss = "".join(
            f" · {', '.join(groups[s])}: {REASON[s]}"
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
        }
        if "C5" in status:
            tags[f"c5-{status['C5']}"] = True
        for t, on in tags.items():
            tally[t] += on
        attrs = (
            f' data-missing="{int(bool(missing))}" data-unplanned="{int(bool(not_planned))}" '
            f'data-tags="{" ".join(t for t, on in tags.items() if on)}"'
        )
        rows.append(_card(sid, cat, miss, "".join(panels), attrs))
    n_unplanned = tally["unplanned"]
    options = "".join(
        f'<option value="{v}"{" selected" if v == "all" else ""}>{html.escape(label)}</option>'
        for v, label in FILTERS
        if "C5" in cfg["conditions"] or not v.startswith("c5-")
    )
    controls = (
        f'<label>Show <select id="f">{options}</select></label>'
        f'<span class="muted" id="n">{len(rows)} photos shown</span>'
    )
    body = (
        '<p class="note">Only images in the benchmark (<code>ledger.jsonl</code>): the original '
        "(C0), the re-save (C1), and the edits that passed the check. Rejected attempts are not "
        "shown; build with <code>--all-attempts</code> to see them. "
        f"{len(rows)} photos: <b>{tally['complete']}</b> have all conditions in the benchmark, "
        f"<b>{n_missing}</b> are still missing a planned edit, and {n_unplanned} have a "
        "condition that was never planned (no room for its box).</p>"
        + ("" if lpath.exists() else '<p class="note"><b>No edits checked yet.</b></p>')
        + LEGEND
        + "".join(rows)
        + FILTER_JS.replace(
            "{TEST}",
            "const t=s.dataset.tags.split(' '),m=t.includes('missing'),u=t.includes('unplanned');"
            "const show=v==='all'||(v==='any'&&(m||u))||(v==='planned'&&m)||t.includes(v);",
        )
    )
    page = _page(cfg, controls, body)
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


def share_page(cfg: dict, wd: Path, page: Path) -> dict:
    """Self-contained copy of a preview page in the week's results folder
    (``inlayfake.site.share_dir``: docs/results/week<N>_<topic>/), to commit or send:
    index.html plus downscaled JPEG copies of every image it shows (``preview.share_max_side``,
    ``share_quality``). Boxes are SVG in original pixel coordinates, scaled with the image, so
    the smaller copies need no change.
    Like the week-1 viewer (``viewer --share``), the copies include COCO photos, downscaled."""
    import re

    from inlayfake.stages.viewer import share_copy

    pcfg = cfg.get("preview", {})
    out = share_dir(cfg)
    side, quality = pcfg.get("share_max_side", 480), pcfg.get("share_quality", 80)
    html_text = page.read_text(encoding="utf-8")
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        src = (wd / html.unescape(m.group(1))).resolve()
        name = src.name if src.suffix.lower() in (".jpg", ".jpeg") else src.stem + ".jpg"
        share_copy(src, out / "images" / name, side, quality)
        n += 1
        return f'src="images/{html.escape(name)}"'

    html_text = re.sub(r'src="([^"]+)"', repl, html_text)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html_text, encoding="utf-8")
    return {"shared": (out / "index.html").as_posix(), "shared_images": n}


def run(
    cfg: dict,
    limit: int | None = 50,
    thumb: int = 240,
    all_attempts: bool = False,
    share: bool = False,
) -> dict:
    wd = work_dir(cfg)
    if share:  # the shareable copy shows every photo of the benchmark view
        if all_attempts:
            raise SystemExit("--share builds the benchmark view; drop --all-attempts")
        out = run(cfg, limit=None, thumb=thumb)
        return out | share_page(cfg, wd, Path(out["preview"]))
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
        rows.append(_card(sid, cat, "", panels))
    body = (
        '<p class="note">All attempts. Only edits marked <b>in benchmark</b> enter ledger.jsonl; '
        "greyed attempts were rejected by the verifier and regenerated with a new seed or "
        "dropped.</p>" + LEGEND + "".join(rows)
    )
    page = _page(cfg, "", body)
    out = wd / "preview.html"
    out.write_text(page, encoding="utf-8")
    n_ok = sum(1 for r in verified.values() if r["passed"])
    return {"preview": out.as_posix(), "sources": len(rows), "edits_in_benchmark": n_ok}
