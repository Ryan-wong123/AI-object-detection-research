"""Shared look and location of the results website (docs/, served by GitHub Pages).

Every weekly results page (week 1 detector viewer, week 2 benchmark preview, later weeks) uses
``SITE_CSS`` and a header titled ``page_title(cfg)``, and is written to ``share_dir(cfg)``:
``docs/results/week<N>_<topic>/`` from the config's ``site`` block. ``docs/index.html`` shows the
weeks as tabs. A new week page: build it with these helpers and add a tab entry to
docs/index.html (its colour tokens are a copy of the ones below).

Shared building blocks: a sticky ``header`` (h1 + controls), ``main``, ``.card`` per photo with
an ``h2`` and a ``.row`` grid of ``.panel`` images (``.t`` caption, ``.wrap`` with ``img`` and an
``svg`` box overlay in image pixels), ``.legend`` with ``.sw`` swatches, ``.note`` text.
"""

from __future__ import annotations

from pathlib import Path

SITE_CSS = """
:root {
  --bg: #f7f7f5; --panel: #ffffff; --ink: #1d1f24; --muted: #5f6470; --line: #d9dbe0;
  --accent: #2463c2; --gt: #1f7a4d; --audit: #2463c2; --tp: #1f9d55; --fp: #e0662b;
  --ign: #8a8f98; --miss: #c2185b; --ins: #8e44ad; --mod: #b7791f; --plan: #c9a400;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15171c; --panel: #1e2128; --ink: #e8e9ec; --muted: #a3a8b3; --line: #343842;
    --accent: #5b9bf0; --gt: #3fbf7f; --audit: #5b9bf0; --tp: #3ccf7a; --fp: #ff8a4c;
    --ign: #9aa0aa; --miss: #ff5c9a; --ins: #c084fc; --mod: #f0b429; --plan: #ffd84d;
  }
}
:root[data-theme="dark"] {
  --bg: #15171c; --panel: #1e2128; --ink: #e8e9ec; --muted: #a3a8b3; --line: #343842;
  --accent: #5b9bf0; --gt: #3fbf7f; --audit: #5b9bf0; --tp: #3ccf7a; --fp: #ff8a4c;
  --ign: #9aa0aa; --miss: #ff5c9a; --ins: #c084fc; --mod: #f0b429; --plan: #ffd84d;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--bg);
  border-bottom: 1px solid var(--line); padding: 10px 16px; display: flex;
  flex-wrap: wrap; gap: 10px 18px; align-items: center; }
header h1 { font-size: 16px; margin: 0 8px 0 0; }
header h1 span, .muted { color: var(--muted); font-weight: 400; }
label { color: var(--muted); display: inline-flex; align-items: center; gap: 6px; }
select, input[type=text], button { font: inherit; color: var(--ink); background: var(--panel);
  border: 1px solid var(--line); border-radius: 6px; padding: 4px 8px; }
button { cursor: pointer; }
a { color: var(--accent); }
main { padding: 8px 16px 40px; max-width: 1600px; }
.note { color: var(--muted); margin: 10px 0; max-width: 900px; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; padding: 6px 0; color: var(--muted);
  font-size: 13px; }
.sw { display: inline-block; width: 14px; height: 10px; border: 2px solid; margin-right: 5px;
  vertical-align: -1px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  margin: 12px 0; padding: 10px 12px; }
.card h2 { font-size: 14px; margin: 0 0 8px; font-weight: 600; }
.card h2 span { color: var(--muted); font-weight: 400; }
.row { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
.panel .t { font-size: 12px; margin-bottom: 3px; display: flex; justify-content: space-between;
  gap: 6px; }
.panel .t b { font-weight: 600; }
.panel .t i { font-style: normal; color: var(--muted); }
.wrap { position: relative; width: 100%; line-height: 0; }
.wrap img { width: 100%; height: auto; display: block; border-radius: 4px; }
.wrap svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.empty { color: var(--muted); padding: 30px; text-align: center; }
@media (max-width: 600px) { .row { grid-template-columns: 1fr; } }
"""

# condition names used on every page
CONDITION_NAMES = {
    "C0": "C0 original",
    "C1": "C1 resave",
    "C2": "C2 sham edit",
    "C3": "C3 real paste",
    "C4": "C4 AI insert",
}


DETECTOR_NAMES = {
    "faster_rcnn": "Faster R-CNN",
    "yolo": "YOLO11-m",
    "rtdetr": "RT-DETR",
    "grounding_dino": "Grounding DINO",
}


def site_cfg(cfg: dict) -> dict:
    return cfg.get("site") or {}


def share_dir(cfg: dict) -> Path:
    """docs/results/<folder>: the week's folder (``site.folder``, e.g. week2_ai_insertion);
    falls back to the run name for configs that are not a week of the site (pilots, tests)."""
    s = site_cfg(cfg)
    return Path(s.get("dir") or "docs/results") / (s.get("folder") or cfg.get("run_name", "run"))


def page_title(cfg: dict, fallback: str) -> str:
    """'Week 2 · AI insertion' from the config's site block, else the fallback."""
    s = site_cfg(cfg)
    if s.get("week") and s.get("title"):
        return f"Week {s['week']} · {s['title']}"
    return fallback
