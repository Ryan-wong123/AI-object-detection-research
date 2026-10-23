"""Clickable review of every edited photo: accept or reject C2, C3 and C4 (author request,
2026-10-22).

``python -m inlayfake editaudit --config X --serve`` builds ``<work>/edit_audit.html`` from the
current final ledger: one card per photo with the matched resave (C1, before) and the photo's
current sham (C2), real paste (C3) and AI insertion (C4), cropped around the box, each with
Accept and Reject; notes per photo. It serves the work folder on 127.0.0.1 and saves every click
at once to ``<work>/condition_review_clicks.json`` and ``<work>/condition_review.csv`` (one row
per reviewed or current image; rejected images stay listed).

A rejected image is redone by the next ``insert`` (``verify.apply_review_rejections``): C2/C4
count as a failed verification (new seed at the same box; after max_attempts the next box, and
C2/C3 follow the C4 box); C3 is re-pasted with another donor. Rejected images never enter the
final ledger. New images appear on the page as "to review" when it is built again.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections import defaultdict
from pathlib import Path

from PIL import Image

from inlayfake.edit.encode import load_rgb
from inlayfake.ledger import Origin, read_ledger
from inlayfake.site import SITE_CSS, page_title
from inlayfake.stages.common import work_dir, write_text_atomic
from inlayfake.stages.verify import (
    REVIEW_CLICKS,
    _box_span,
    in_audit_sample,
    read_review,
    verify_window,
)

PAGE_FILE = Path(__file__).with_name("editaudit_page.html")
CSV_NAME = "condition_review.csv"
COLUMNS = [
    "source_id",
    "condition",
    "image_id",
    "category",
    "plan_version",
    "in_random_sample",
    "current",
    "decision",
    "notes",
]
REVIEWED = ("C2", "C3", "C4")
_LOCK = threading.Lock()


def _photo_key(sid: int) -> str:
    return f"photo:{sid}"


def write_review_csv(wd: Path, current: dict[str, dict]) -> int:
    """One row per current C2/C3/C4 image and per image reviewed before (rejected ones stay)."""
    review = read_review(wd)
    rows = dict(current)
    old = wd / CSV_NAME
    if old.exists():  # keep images reviewed before that are no longer current
        with old.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["image_id"] not in rows and r["image_id"] in review:
                    rows[r["image_id"]] = r | {"current": "n"}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for iid, r in sorted(rows.items(), key=lambda kv: (int(kv[1]["source_id"]), kv[0])):
        d = review.get(iid, {})
        notes = review.get(_photo_key(int(r["source_id"])), {}).get("notes", "")
        w.writerow(r | {"decision": d.get("decision", ""), "notes": notes})
    write_text_atomic(wd / CSV_NAME, buf.getvalue())
    return len(rows)


def _missing_cards(cfg: dict, wd: Path, with_c4: set[int], crops: Path) -> list[dict]:
    """Photos without a verified AI object (author request, 2026-10-23: show them too): the photo
    with the model's planned boxes, and every failed C4 attempt with the verifier's reason.
    View only; nothing from them is in the benchmark."""
    from inlayfake.ledger import ImageRecord
    from inlayfake.stages.common import load_coco, read_jsonl

    vcfg = cfg["verify"]
    sources = read_jsonl(wd / "sources.jsonl")
    only = cfg["sampling"].get("only_sources")
    if only:
        keep = {r["source_id"] for r in read_jsonl(Path(only))}
        sources = [x for x in sources if x["source_id"] in keep]
    plans = {r["source_id"]: r for r in read_jsonl(wd / "plans.jsonl")}
    boxes: dict[int, list[dict]] = defaultdict(list)
    cands = wd / "natural_candidates.jsonl"
    for r in read_jsonl(cands) if cands.exists() else []:
        if r.get("ok"):
            boxes[r["source_id"]].append(r)
    fails: dict[int, list[dict]] = defaultdict(list)
    vpath = wd / "verify.jsonl"
    for r in read_jsonl(vpath) if vpath.exists() else []:
        if r["condition"] == "C4" and not r["passed"]:
            fails[int(r["key"].split("|")[0])].append(r)
    index = None
    out = []
    for src in sources:
        sid = src["source_id"]
        if sid in with_c4:
            continue
        W, H = src["width"], src["height"]
        full = crops / f"{sid}_full.jpg"
        if not full.exists():
            index = index or load_coco(cfg)
            im = Image.open(index.image_path(sid)).convert("RGB")
            k = 720 / max(im.size)
            im.resize((round(im.width * k), round(im.height * k))).save(full, quality=85)
        planned = sorted(boxes.get(sid, []), key=lambda r: r["valid_rank"])
        overlay = "".join(_box_span(r["box"], 0, 0, W, H, "plan") for r in planned)
        tries = []
        for row in sorted(fails.get(sid, []), key=lambda r: (r.get("version", 0), r["attempt"])):
            rp = wd / "records" / f"{row['image_id']}.json"
            if not rp.exists():
                continue
            rec = ImageRecord(**json.loads(rp.read_text(encoding="utf-8")))
            x0, y0, x1, y1 = verify_window(rec, rec.width, rec.height, vcfg)
            crop = crops / f"{rec.image_id}_crop.jpg"
            if not crop.exists():
                Image.fromarray(load_rgb(rec.path)[y0:y1, x0:x1]).save(crop, quality=90)
            tries.append(
                {
                    "src": f"audit_crops/{crop.name}",
                    "boxes": _box_span(rec.edit.plan_box, x0, y0, x1, y1, "plan"),
                    "label": f"box {row.get('version', 0) + 1}, seed {row['attempt']}: "
                    + row["reason"],
                }
            )
        if sid not in plans:
            reason = "no box from planning (Qwen found no usable place)"
        else:
            reason = f"{len(planned)} box(es) tried, {len(tries)} images, none verified"
        out.append(
            {
                "kind": "missing",
                "sid": sid,
                "category": plans[sid]["category"] if sid in plans else "no box",
                "reason": reason,
                "full": {"src": f"audit_crops/{full.name}", "boxes": overlay},
                "fails": tries,
            }
        )
    return out


def build_page(cfg: dict) -> dict:
    wd = work_dir(cfg)
    vcfg = cfg["verify"]
    seed, frac = cfg["seed"] + 7, vcfg.get("audit_fraction", 0.1)
    final = list(read_ledger(wd / "ledger.jsonl"))
    by_sid: dict[int, dict[str, object]] = defaultdict(dict)
    for r in final:
        by_sid[r.source_id][r.condition.value] = r
    crops = wd / "audit_crops"
    crops.mkdir(exist_ok=True)
    cards, current = [], {}
    for sid in sorted(by_sid):
        conds = by_sid[sid]
        edits = {c: conds[c] for c in REVIEWED if c in conds}
        if not edits:
            continue
        lead = edits.get("C4") or next(iter(edits.values()))
        x0, y0, x1, y1 = verify_window(lead, lead.width, lead.height, vcfg)
        figs = []
        for cond in ("C1",) + REVIEWED:
            r = conds.get(cond)
            if r is None:
                figs.append({"cond": cond, "missing": True})
                continue
            out = crops / f"{r.image_id}_crop.jpg"
            if not out.exists():
                Image.fromarray(load_rgb(r.path)[y0:y1, x0:x1]).save(out, quality=90)
            boxes = _box_span(lead.edit.plan_box, x0, y0, x1, y1, "plan")
            for i in r.instances:
                if i.origin in (Origin.AI_INSERTED, Origin.REAL_PASTED):
                    boxes += _box_span(i.bbox, x0, y0, x1, y1, "ins")
            fig = {"cond": cond, "src": f"audit_crops/{out.name}", "boxes": boxes}
            if cond in REVIEWED:
                rand = cond == "C4" and in_audit_sample(r.image_id, seed, frac)
                fig |= {"id": r.image_id, "random": rand}
                current[r.image_id] = {
                    "source_id": str(sid),
                    "condition": cond,
                    "image_id": r.image_id,
                    "category": r.edit.target_category if r.edit else "",
                    "plan_version": str(r.edit.plan_version if r.edit else 0),
                    "in_random_sample": "y" if rand else "n",
                    "current": "y",
                }
            figs.append(fig)
        cards.append(
            {
                "sid": sid,
                "category": lead.edit.target_category,
                "photo": _photo_key(sid),
                "figs": figs,
            }
        )
    n_rows = write_review_csv(wd, current)
    with_c4 = {sid for sid, conds in by_sid.items() if "C4" in conds}
    missing = _missing_cards(cfg, wd, with_c4, crops)
    cards += missing
    page = (
        PAGE_FILE.read_text(encoding="utf-8")
        .replace("__CSS__", SITE_CSS)
        .replace("__TITLE__", page_title(cfg, "Edit review"))
        .replace("__DATA__", json.dumps(cards))
    )
    write_text_atomic(wd / "edit_audit.html", page)
    return {
        "page": str(wd / "edit_audit.html"),
        "photos": len(cards) - len(missing),
        "photos_without_ai_object": len(missing),
        "csv_rows": n_rows,
    }


def save_click(wd: Path, key: str, field: str, value: str) -> dict:
    """Store one decision (accept/reject of an image, or notes of a photo) and rewrite the CSV."""
    if field not in ("decision", "notes"):
        raise ValueError(f"unknown field {field!r}")
    if field == "decision" and value not in ("", "accept", "reject"):
        raise ValueError(f"unknown decision {value!r}")
    with _LOCK:
        review = read_review(wd)
        row = review.setdefault(key, {})
        if value:
            row[field] = value
        else:
            row.pop(field, None)
        if not row:
            review.pop(key)
        write_text_atomic(wd / REVIEW_CLICKS, json.dumps(review, indent=1, sort_keys=True))
        current = {}
        p = wd / CSV_NAME
        if p.exists():
            with p.open(encoding="utf-8") as f:
                current = {r["image_id"]: r for r in csv.DictReader(f) if r["current"] == "y"}
        write_review_csv(wd, current)
    return row


def serve(cfg: dict, port: int = 8769) -> None:
    import functools
    import http.server

    wd = work_dir(cfg)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.path = "/edit_audit.html"
            if path.endswith("/review.json"):
                body = json.dumps(read_review(wd)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):  # noqa: N802
            if not self.path.endswith("/review-click"):
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(n) or b"{}")
            try:
                save_click(wd, msg["id"], msg["field"], str(msg.get("value", "")))
            except (KeyError, ValueError) as e:
                self.send_error(400, str(e))
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):  # quiet console
            pass

    handler = functools.partial(Handler, directory=str(wd))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Edit review: http://127.0.0.1:{port}/  (decisions -> {wd / CSV_NAME})", flush=True)
    srv.serve_forever()


def run(cfg: dict, serve_page: bool = False, port: int = 8769) -> dict:
    res = build_page(cfg)
    if serve_page:
        print(json.dumps(res, indent=2), flush=True)
        serve(cfg, port)
    return res
