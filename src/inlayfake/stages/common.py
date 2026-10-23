"""Shared helpers for pipeline stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from inlayfake.data.coco import CocoIndex


def work_dir(cfg: dict) -> Path:
    p = Path(cfg["paths"]["work"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_coco(cfg: dict) -> CocoIndex:
    paths = cfg["paths"]
    return CocoIndex.load(paths["coco_annotations"], paths["coco_images"])


def derive_seed(*parts: object) -> int:
    """Stable 31-bit seed from arbitrary parts (independent of Python's hash randomisation)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def write_text_atomic(path: Path, text: str) -> None:
    """Write a whole file so that a crash or power loss leaves the old or the new version, never
    a half-written one: write a temporary file next to it, then rename it in one step."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def open_append(path: Path):
    """Open a JSON-lines file for appending, crash-safely. A power cut while appending can leave
    a cut-off last line without a newline; appending after it would glue the next row onto that
    line, which ``read_jsonl`` then cannot read (a broken line that is no longer the last one).
    So a cut-off last line is removed first (that item is redone)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            with path.open("r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)  # 0 if the only line was cut off
    return path.open("a", encoding="utf-8")


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    lines = [json.dumps(r) + "\n" for r in rows]
    write_text_atomic(path, "".join(lines))
    n = len(lines)
    return n


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSON-lines file. A cut-off *last* line (a crash or power loss while appending)
    is skipped, so that item is simply redone; a broken line anywhere else is an error."""
    with path.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
    return out


def run_dir(cfg: dict) -> Path:
    """runs/<run_name>: where predictions, scores and metrics of one experiment go."""
    p = Path(cfg["paths"]["runs"]) / cfg.get("run_name", "default")
    p.mkdir(parents=True, exist_ok=True)
    return p


def ledger_path(cfg: dict) -> Path:
    """The verified ledger if `verify` has run, else the provisional one from `generate`."""
    wd = work_dir(cfg)
    final = wd / "ledger.jsonl"
    return final if final.exists() else wd / "ledger_generated.jsonl"


def decisions_path(cfg: dict) -> tuple[Path, bool]:
    """(path, configured) of the label-audit decisions applied when scoring.

    One file for every run on the same photos (author, 2026-10-18): ``audit.decisions``, else
    ``placement.audit_decisions`` (the file the week-2 class choice used), else the run's own
    ``<work>/audit_decisions.csv`` (week 1, where the `audit` stage writes it). Before, scoring
    always used the work dir, so a run in another work dir (week 2: data/inlayfake_v2) would
    silently have been scored without the week-1 labels.
    """
    explicit = cfg.get("audit", {}).get("decisions") or cfg.get("placement", {}).get(
        "audit_decisions"
    )
    if explicit:
        return Path(explicit), True
    return work_dir(cfg) / "audit_decisions.csv", False


def load_records(cfg: dict):
    """Ledger records restricted to the configured conditions, with audit decisions applied."""
    from inlayfake.data.audit import apply_decisions, read_decisions
    from inlayfake.ledger import read_ledger

    conds = set(cfg["conditions"])
    recs = [r for r in read_ledger(ledger_path(cfg)) if r.condition.value in conds]
    path, configured = decisions_path(cfg)
    if configured and not path.exists():  # a named file must exist: never score without it
        raise SystemExit(f"audit decisions file not found: {path} (audit.decisions)")
    decisions = read_decisions(path)
    return apply_decisions(recs, decisions) if decisions else recs


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["image_id"] for line in path.open(encoding="utf-8") if line.strip()}
