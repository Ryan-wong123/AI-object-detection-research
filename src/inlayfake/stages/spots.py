"""Human review of planned natural spots before generation (author request, 2026-10-21).

``python -m inlayfake spots --config X`` builds ``<work>/spot_review/index.html``: every photo
with its valid natural spots (green = the planned one, yellow = backups), each with Accept and
Reject buttons. With ``--serve`` it also runs a local web server (127.0.0.1 only) that saves every
click at once to the review file (``placement.natural.review.file``, default
``<work>/spot_review.json``), so a review can stop and continue at any time.

``plan`` then applies the review (``apply_review``) when ``placement.natural.review.enabled``:
a spot is used only if accepted (``require_accept``, default true) or, with
``require_accept: false``, if not rejected. A photo's plan is its first usable spot in the
original order; a photo with none gets no AI edit. Decisions are keyed by ``spot_id`` (photo,
class, box), so a re-plan that gives the same box keeps its decision.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from inlayfake.site import SITE_CSS, page_title
from inlayfake.stages.common import load_coco, read_jsonl, work_dir, write_text_atomic


def spot_id(row: dict) -> str:
    """Stable key of one spot: photo, class and box (whole pixels)."""
    x, y, w, h = (round(v) for v in row["box"])
    return f"{row['source_id']}|{row['category']}|{x},{y},{w},{h}"


def review_path(cfg: dict) -> Path:
    r = cfg["placement"].get("natural", {}).get("review") or {}
    return Path(r["file"]) if r.get("file") else work_dir(cfg) / "spot_review.json"


def read_review(path: Path) -> dict[str, str]:
    """{spot_id: "accept" | "reject"} (empty if the file does not exist yet)."""
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_review(
    rows: list[dict],
    plans: list[dict],
    failed: list[dict],
    decisions: dict[str, str],
    require_accept: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Mark valid spots that the review did not allow as not ok (reason "rejected in review" or
    "not reviewed"), renumber ``valid_rank`` per photo and rebuild the plans from each photo's
    first remaining spot. Returns (plans, failed)."""
    by_sid: dict[int, list[dict]] = {}
    for r in rows:
        if not r.get("ok"):
            continue
        d = decisions.get(spot_id(r))
        if d == "reject" or (require_accept and d != "accept"):
            r["ok"] = False
            r["reason"] = "rejected in review" if d == "reject" else "not reviewed"
            r.pop("valid_rank", None)
            r["review"] = d
        else:
            by_sid.setdefault(r["source_id"], []).append(r)
            r["review"] = d
    for spots in by_sid.values():
        spots.sort(key=lambda r: r["valid_rank"])
        for i, r in enumerate(spots):
            r["valid_rank"] = i
    new_plans, new_failed = [], list(failed)
    for p in plans:
        spots = by_sid.get(p["source_id"])
        if not spots:
            new_failed.append({"source_id": p["source_id"], "reason": "no reviewed spot"})
            continue
        first = spots[0]
        new_plans.append(
            p
            | {
                "category": first["category"],
                "box_free": first["box"],
                "class_in_photo": first.get("class_in_photo", True),
            }
        )
    return new_plans, new_failed


# ------------------------------------------------------------------------------- the page
# the page (template with __CSS__, __TITLE__, __DATA__), like viewer_page.html
PAGE_FILE = Path(__file__).with_name("spots_page.html")


def build_page(cfg: dict) -> dict:
    """Write ``<work>/spot_review/`` (page + downscaled photos) from the current candidates."""
    wd = work_dir(cfg)
    out = wd / "spot_review"
    (out / "img").mkdir(parents=True, exist_ok=True)
    index = load_coco(cfg)
    src = {s["source_id"]: s for s in read_jsonl(wd / "sources.jsonl")}
    by_sid: dict[int, list[dict]] = {}
    # the review is of the plan *before* review filtering: every spot that passed the rules
    cands = read_jsonl(wd / "natural_candidates.jsonl")
    for r in cands:
        if r.get("ok") or r.get("reason") in ("rejected in review", "not reviewed"):
            by_sid.setdefault(r["source_id"], []).append(r)
    photos = []
    for n, (sid, spots) in enumerate(sorted(by_sid.items()), 1):
        order = {1: 0, 2: 1, "any": 2, "any2": 3}
        spots.sort(key=lambda r: (order.get(r.get("round", 1), 4), r.get("rank") or 0))
        s = src[sid]
        p = out / "img" / f"{sid}.jpg"
        if not p.exists():
            im = Image.open(index.image_path(sid)).convert("RGB")
            k = 720 / max(im.size)
            im.resize((round(im.width * k), round(im.height * k))).save(p, quality=85)
        photos.append(
            {
                "n": n,
                "sid": sid,
                "split": s["split"],
                "W": s["width"],
                "H": s["height"],
                "spots": [
                    {
                        "id": spot_id(r),
                        "cat": r["category"],
                        "box": r["box"],
                        "round": r.get("round", 1),
                        "in_photo": r.get("class_in_photo", True),
                    }
                    for r in spots
                ],
            }
        )
    page = (
        PAGE_FILE.read_text(encoding="utf-8")
        .replace("__CSS__", SITE_CSS)
        .replace("__TITLE__", page_title(cfg, "Natural placement"))
        .replace("__DATA__", json.dumps(photos))
    )
    write_text_atomic(out / "index.html", page)
    return {"page": str(out / "index.html"), "photos": len(photos)}


def serve(cfg: dict, port: int = 8767) -> None:
    """Serve the review page on 127.0.0.1 and save every decision to the review file."""
    import functools
    import http.server

    out = work_dir(cfg) / "spot_review"
    path = review_path(cfg)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            if self.path.split("?")[0].endswith("/decisions.json"):
                body = json.dumps(read_review(path)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):  # noqa: N802
            if not self.path.endswith("/decision"):
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(n) or b"{}")
            dec = read_review(path)
            if msg.get("decision") in ("accept", "reject"):
                dec[msg["id"]] = msg["decision"]
            else:
                dec.pop(msg.get("id"), None)
            write_text_atomic(path, json.dumps(dec, indent=1, sort_keys=True))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):  # quiet console
            pass

    handler = functools.partial(Handler, directory=str(out))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(
        f"Spot review: http://127.0.0.1:{port}/  (decisions -> {path}); Ctrl+C to stop", flush=True
    )
    srv.serve_forever()


def run(cfg: dict, serve_page: bool = False, port: int = 8767) -> dict:
    res = build_page(cfg)
    res["decisions"] = str(review_path(cfg))
    if serve_page:
        print(json.dumps(res, indent=2), flush=True)
        serve(cfg, port)
    return res
