"""Stage `detect`: run each configured detector on every image → runs/<run>/detections/<det>.jsonl.

One detector at a time on the GPU; resumable (images already in the output file are skipped).
"""

from __future__ import annotations

import json

from tqdm import tqdm

from inlayfake.detect import Detection, FakeDetector, build_detector
from inlayfake.edit.encode import load_rgb
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import done_ids, load_coco, load_records, open_append, run_dir


def run(cfg: dict) -> dict:
    recs = load_records(cfg)
    categories = sorted(load_coco(cfg).cat_id)  # the 80 COCO names
    rd = run_dir(cfg)
    summary = {}
    for name in cfg["detect"]["detectors"]:
        out = rd / "detections" / f"{name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = done_ids(out)
        todo = [r for r in recs if r.image_id not in done]
        if not todo:
            summary[name] = 0
            continue
        det = build_detector(name, cfg, categories, cfg["hardware"]["device"])
        allowed = set(categories)
        with open_append(out) as f:
            for r in tqdm(todo, desc=name):
                if isinstance(det, FakeDetector):
                    det.set_truth(
                        [Detection(i.category, i.bbox, 1.0) for i in r.instances if not i.iscrowd]
                    )
                dets = [d.to_dict() for d in det(load_rgb(r.path)) if d.category in allowed]
                f.write(json.dumps({"image_id": r.image_id, "detections": dets}) + "\n")
        summary[name] = len(todo)
        del det
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    write_run_info(rd, "detect", cfg, {"images_processed": summary})
    return {"run_dir": rd.as_posix(), "images_processed": summary}


def load_detections(cfg: dict, name: str) -> dict[str, list[dict]]:
    path = run_dir(cfg) / "detections" / f"{name}.jsonl"
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            out[row["image_id"]] = row["detections"]
    return out
