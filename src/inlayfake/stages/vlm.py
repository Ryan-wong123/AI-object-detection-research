"""Stage `vlm` (Stage 2, H5): origin claims of VLMs on edited and unedited images.

Images: the ``vlm.split`` (test) photos that have every configured condition, in
``vlm.conditions`` (C1 unedited, C3 real paste, C4 AI insertion), at most ``vlm.max_sources``
photos. For each model in ``vlm.run_models`` (one on the GPU at a time) and each prompt setting in
``vlm.prompt_settings`` (plain, grounded) one row per image goes to
runs/<run>/vlm/<model>__<setting>.jsonl: the raw reply, the caption, and the parsed per-object
claims (``vlm/claims.py``). Resumable per (model, setting).

Grounded boxes come from the frozen detector ``vlm.grounding.detector`` (its detections of this
run, score >= ``vlm.grounding.score``, at most ``vlm.grounding.max_boxes``, highest first), so the
model sees only the final image and a detector's output, never the ledger (golden rule 3).
"""

from __future__ import annotations

import json
import time

from tqdm import tqdm

from inlayfake.edit.encode import load_rgb
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import done_ids, load_records, open_append, run_dir
from inlayfake.stages.detect import load_detections
from inlayfake.vlm.models import FakeVLM, build_vlm_model


def select_records(cfg: dict, recs: list) -> list:
    vcfg = cfg["vlm"]
    conds = list(vcfg["conditions"])
    by_src: dict = {}
    for r in recs:
        by_src.setdefault(r.source_id, {})[r.condition.value] = r
    need = set(cfg["conditions"])
    sources = sorted(
        (s for s, d in by_src.items() if need <= set(d) and d["C1"].split == vcfg["split"]),
        key=str,
    )
    if vcfg.get("max_sources"):
        sources = sources[: vcfg["max_sources"]]
    return [by_src[s][c] for s in sources for c in conds]


def grounding_boxes(dets: list[dict], score: float, max_boxes: int) -> list[dict]:
    kept = sorted((d for d in dets if d["score"] >= score), key=lambda d: -d["score"])[:max_boxes]
    return [
        {
            "id": k + 1,
            "category": d["category"],
            "bbox": [round(v, 1) for v in d["bbox"]],
            "score": round(d["score"], 3),
        }
        for k, d in enumerate(kept)
    ]


def run(cfg: dict) -> dict:
    vcfg = cfg["vlm"]
    recs = select_records(cfg, load_records(cfg))
    g = vcfg["grounding"]
    dets = load_detections(cfg, g["detector"])
    out_dir = run_dir(cfg) / "vlm"
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = {}
    for name in vcfg["run_models"]:
        todo = {}
        for setting in vcfg["prompt_settings"]:
            path = out_dir / f"{name}__{setting}.jsonl"
            done = done_ids(path)
            todo[setting] = [r for r in recs if r.image_id not in done]
            processed[f"{name}__{setting}"] = len(todo[setting])
        if not any(todo.values()):
            continue
        model = build_vlm_model(name, cfg)
        for setting, rs in todo.items():
            if not rs:
                continue
            with open_append(out_dir / f"{name}__{setting}.jsonl") as f:
                for r in tqdm(rs, desc=f"{name} {setting}"):
                    img = load_rgb(r.path)
                    boxes = grounding_boxes(dets.get(r.image_id, []), g["score"], g["max_boxes"])
                    if isinstance(model, FakeVLM):
                        model.truth = [
                            {"category": i.category, "bbox": i.bbox}
                            for i in r.instances
                            if not i.iscrowd
                        ]
                    t = time.time()
                    out = model.plain(img) if setting == "plain" else model.grounded(img, boxes)
                    row = {
                        "image_id": r.image_id,
                        "source_id": r.source_id,
                        "split": r.split,
                        "condition": r.condition.value,
                        "model": name,
                        "setting": setting,
                        "boxes": boxes if setting == "grounded" else [],
                        "seconds": round(time.time() - t, 2),
                    } | out
                    f.write(json.dumps(row) + "\n")
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    write_run_info(run_dir(cfg), "vlm", cfg, {"images": len(recs), "processed": processed})
    return {"images": len(recs), "processed": processed}
