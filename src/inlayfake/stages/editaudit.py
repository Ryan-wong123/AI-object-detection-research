"""Clickable human audit of the AI edits (author request, 2026-10-22).

``python -m inlayfake editaudit --config X --serve`` rebuilds the audit sheet from the current
ledger (``verify.audit_sheet``: every C4 edit plus the random sample, crops before/after with the
planned and verified boxes), writes ``<work>/edit_audit.html`` and serves the work folder on
127.0.0.1. Each card has buttons for the sheet's columns (object present, realistic, other change)
and a notes field; every click is saved at once into ``<work>/audit_edits_clicks.json`` and merged
into ``audit_edits.csv`` (the pipeline's audit sheet), so the review can stop and continue at any
time, also while the run is still verifying (``audit_sheet`` merges the clicks whenever it rewrites
the CSV).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from inlayfake.ledger import read_ledger
from inlayfake.site import SITE_CSS, page_title
from inlayfake.stages.common import work_dir, write_text_atomic
from inlayfake.stages.verify import (
    AUDIT_CLICKS,
    audit_sheet,
    read_audit_clicks,
    write_audit_csv,
)

PAGE_FILE = Path(__file__).with_name("editaudit_page.html")
FIELDS = ("object_ok", "realistic", "other_change", "notes")
_LOCK = threading.Lock()


def build_page(cfg: dict) -> dict:
    wd = work_dir(cfg)
    vcfg = cfg["verify"]
    final = list(read_ledger(wd / "ledger.jsonl"))
    n = audit_sheet(wd, final, vcfg.get("audit_fraction", 0.1), cfg["seed"] + 7, vcfg)
    cards = json.loads((wd / "audit_cards.json").read_text(encoding="utf-8")) if n else []
    page = (
        PAGE_FILE.read_text(encoding="utf-8")
        .replace("__CSS__", SITE_CSS)
        .replace("__TITLE__", page_title(cfg, "Edit audit"))
        .replace("__DATA__", json.dumps(cards))
    )
    write_text_atomic(wd / "edit_audit.html", page)
    return {"page": str(wd / "edit_audit.html"), "edits": len(cards)}


def save_click(wd: Path, image_id: str, field: str, value: str) -> dict:
    """Store one decision and merge all clicks into audit_edits.csv."""
    import csv

    if field not in FIELDS:
        raise ValueError(f"unknown field {field!r}")
    with _LOCK:
        clicks = read_audit_clicks(wd)
        row = clicks.setdefault(image_id, {})
        if value:
            row[field] = value
        else:
            row.pop(field, None)
        write_text_atomic(wd / AUDIT_CLICKS, json.dumps(clicks, indent=1, sort_keys=True))
        with (wd / "audit_edits.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        write_audit_csv(wd, rows)
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
            if path.endswith("/audit-clicks.json"):
                body = json.dumps(read_audit_clicks(wd)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):  # noqa: N802
            if not self.path.endswith("/audit-click"):
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
    print(f"Edit audit: http://127.0.0.1:{port}/  (clicks -> {wd / 'audit_edits.csv'})", flush=True)
    srv.serve_forever()


def run(cfg: dict, serve_page: bool = False, port: int = 8769) -> dict:
    res = build_page(cfg)
    if serve_page:
        print(json.dumps(res, indent=2), flush=True)
        serve(cfg, port)
    return res
