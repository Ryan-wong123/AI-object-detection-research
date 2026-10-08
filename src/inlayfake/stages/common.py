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
