"""Provenance ledger: the ground-truth data contract (see ARCHITECTURE.md §2).

One ``ImageRecord`` per final image a model sees. Records are stored as JSON Lines.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class Origin(str, Enum):
    PHOTOGRAPHED = "photographed"
    AI_INSERTED = "ai_inserted"
    REAL_PASTED = "real_pasted"
    MIXED_MODIFIED = "mixed_modified"
    UNRESOLVED = "unresolved"


class Condition(str, Enum):
    C0 = "C0"  # untouched source bytes
    C1 = "C1"  # matched resave (reference for all contrasts)
    C2 = "C2"  # sham edit
    C3 = "C3"  # real-object paste
    C4 = "C4"  # AI insertion, no occlusion
    C5 = "C5"  # AI insertion with controlled occlusion


SPLITS = ("train", "val", "test")


@dataclass
class Instance:
    instance_id: str
    category: str
    bbox: list[float]  # [x, y, w, h] visible extent in the final image
    origin: Origin
    visible_fraction: float = 1.0
    iscrowd: bool = False
    mask_path: str | None = None

    def __post_init__(self) -> None:
        self.origin = Origin(self.origin)
        if len(self.bbox) != 4 or self.bbox[2] < 0 or self.bbox[3] < 0:
            raise ValueError(f"invalid bbox {self.bbox} for {self.instance_id}")
        if not 0.0 <= self.visible_fraction <= 1.0:
            raise ValueError(f"visible_fraction out of range for {self.instance_id}")


@dataclass
class EditInfo:
    target_category: str
    seed: int
    plan_box: list[float]
    prompt: str | None = None
    negative_prompt: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    steps: int | None = None
    guidance: float | None = None
    crop_box: list[float] | None = None
    intended_mask_path: str | None = None
    actual_mask_path: str | None = None
    donor: dict | None = None  # C3: {"image_id": ..., "ann_id": ...}
    attempts: int = 1
    verifier_score: float | None = None


@dataclass
class ImageRecord:
    image_id: str
    source_id: int
    split: str
    condition: Condition
    path: str
    width: int
    height: int
    generator: str | None = None
    sha256: str | None = None
    edit: EditInfo | None = None
    instances: list[Instance] = field(default_factory=list)
    ignore_regions: list[list[float]] = field(default_factory=list)
    postproc: dict | None = None

    def __post_init__(self) -> None:
        self.condition = Condition(self.condition)
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}")
        if isinstance(self.edit, dict):
            self.edit = EditInfo(**self.edit)
        self.instances = [i if isinstance(i, Instance) else Instance(**i) for i in self.instances]
        ids = [i.instance_id for i in self.instances]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate instance ids in {self.image_id}")
        if self.condition in (Condition.C4, Condition.C5) and not any(
            i.origin is Origin.AI_INSERTED for i in self.instances
        ):
            msg = f"{self.image_id}: {self.condition.value} needs an ai_inserted instance"
            raise ValueError(msg)

    @staticmethod
    def make_id(source_id: int, condition: str, generator: str | None, seed: int | None) -> str:
        return f"{source_id}__{Condition(condition).value}__{generator or 'none'}__s{seed or 0}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["condition"] = self.condition.value
        for inst in d["instances"]:
            inst["origin"] = Origin(inst["origin"]).value
        return d


def write_ledger(records: Iterable[ImageRecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_ledger(path: str | Path) -> Iterator[ImageRecord]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield ImageRecord(**json.loads(line))
