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


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def load_records(cfg: dict):
    """Ledger records restricted to the configured conditions, with audit decisions applied."""
    from inlayfake.data.audit import apply_decisions, read_decisions
    from inlayfake.ledger import read_ledger

    conds = set(cfg["conditions"])
    recs = [r for r in read_ledger(ledger_path(cfg)) if r.condition.value in conds]
    decisions = read_decisions(work_dir(cfg) / "audit_decisions.csv")
    return apply_decisions(recs, decisions) if decisions else recs


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["image_id"] for line in path.open(encoding="utf-8") if line.strip()}
