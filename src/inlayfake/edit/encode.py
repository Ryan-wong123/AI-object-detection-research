"""Encoding policy (ARCHITECTURE.md §3.4): identical decode → edit → JPEG for C1–C5.

Every image a model sees, whether real or edited, is written by ``save_policy`` with the same
encoder, quality and chroma subsampling. File format therefore cannot reveal origin.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    """Decode to uint8 RGB HxWx3. EXIF orientation is NOT applied: COCO boxes refer to the
    stored pixel grid."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def save_policy(arr: np.ndarray, path: str | Path, enc: dict) -> str:
    """Write ``arr`` with the benchmark encoding policy and return the file's sha256."""
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected uint8 HxWx3 RGB array")
    if enc.get("format", "JPEG").upper() != "JPEG":
        raise ValueError("only the JPEG policy is supported")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(
        path,
        format="JPEG",
        quality=int(enc["quality"]),
        subsampling=int(enc["subsampling"]),
        optimize=False,
    )
    return sha256_file(path)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
