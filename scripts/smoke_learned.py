"""Check that TruFor and UnivFD load and run on one C4 image (2026-10-26), before `forensics`.

    python scripts/smoke_learned.py configs/task3_detection_natural.yaml

Prints the map size and score range for TruFor and the image and object scores for UnivFD on
the first C4 image of the ledger. Nothing is written.
"""

from __future__ import annotations

import sys
import time

from inlayfake.cli import full_speed_in_background
from inlayfake.config import load_config
from inlayfake.edit.encode import load_rgb
from inlayfake.forensics.learned import build_learned, clip_view, object_crop, sigmoid
from inlayfake.stages.common import load_records


def main() -> int:
    cfg = load_config(sys.argv[1])
    full_speed_in_background()
    rec = next(r for r in load_records(cfg) if r.condition.value == "C4")
    img = load_rgb(rec.path)
    print(f"image {rec.image_id} {img.shape}")
    for name in cfg["forensics"].get("run_learned") or []:
        t = time.time()
        model = build_learned(name, cfg)
        print(f"{name}: loaded in {time.time() - t:.1f} s")
        t = time.time()
        if name == "trufor":
            fmap, det = model(img)
            print(f"  map {fmap.shape} min {fmap.min():.3f} max {fmap.max():.3f}; score {det:.3f}")
        else:
            insts = [i for i in rec.instances if not i.iscrowd]
            views = [clip_view(img)] + [object_crop(img, i.bbox, 1.3) for i in insts]
            logits = model.logits(model.features(views))
            print(f"  image P(fake) {float(sigmoid(logits[0])):.3f}")
            for inst, lg in zip(insts, logits[1:], strict=True):
                print(f"  {inst.origin.value:13s} {inst.category:15s} P(fake) {sigmoid(lg):.3f}")
        print(f"  ran in {time.time() - t:.2f} s")
        del model
    return 0


if __name__ == "__main__":
    sys.exit(main())
