"""Diagnostic only: does a tighter window (box fills more of SDXL's canvas) make SDXL insert?

12 random failed C4 jobs of task2_pilot_surface, new seed, strength 0.99, project local_edit and
OWLv2 rule. Window scale 2.5 (current), 1.5 and 1.2; minimum window side lowered to 64 for the
test so small boxes are affected too. Writes nothing to data/.
"""
import gc
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

import inlayfake.edit.compose as compose
from inlayfake.edit.encode import load_rgb
from inlayfake.edit.generators import DiffusersInpainter
from inlayfake.edit.verify import OwlV2Finder, best_candidate

_cw = compose.crop_window
compose.crop_window = lambda box, W, H, scale, min_side=64: _cw(box, W, H, scale, min_side)

out_dir = sys.argv[1]
wd = "data/inlayfake_pilot_surface"
rows = [json.loads(l) for l in open(f"{wd}/verify.jsonl")]
fail = [r for r in rows if not r["passed"] and r["condition"] == "C4"]
pick = [fail[i] for i in sorted(np.random.default_rng(0).choice(len(fail), 12, replace=False))]
recs = [json.load(open(f"{wd}/records/{r['image_id']}.json")) for r in pick]
SCALES = (2.5, 1.5, 1.2)

gen = DiffusersInpainter("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 1024, strength=0.99)
outs = {}
for rec in recs:
    img = load_rgb(f"{wd}/images/{rec['source_id']}__C1__none__s0.jpg")
    e = rec["edit"]
    for sc in SCALES:
        outs[(sc, rec["image_id"])] = compose.local_edit(
            img, e["plan_box"], gen, 1024, e["prompt"], e["negative_prompt"], e["seed"] + 1, sc, 6, 3
        )
del gen
gc.collect()
torch.cuda.empty_cache()

finder = OwlV2Finder("google/owlv2-base-patch16-ensemble", "cuda", 0.05)
tiles, res = [], dict.fromkeys(SCALES, 0)
for rec in recs:
    e = rec["edit"]
    # verify on the same 2.5x region for every variant, as the real verify stage would
    x0, y0, x1, y1 = (int(v) for v in rec["edit"]["crop_box"])
    for sc in SCALES:
        out = outs[(sc, rec["image_id"])][0]
        crop = out[y0:y1, x0:x1]
        c = [([b[0] + x0, b[1] + y0, b[2] - b[0], b[3] - b[1]], s) for b, s in finder(crop, e["target_category"])]
        best = best_candidate(c, e["plan_box"], 0.3, 0.2)
        res[sc] += best is not None
        print(rec["image_id"], e["target_category"], sc, "PASS" if best else "fail")
        t = Image.fromarray(crop).resize((256, 256))
        ImageDraw.Draw(t).text((3, 3), f"{e['target_category']} win={sc} {'PASS' if best else 'fail'}", fill="yellow")
        tiles.append(t)
print("passed of", len(recs), res)
W = Image.new("RGB", (256 * 6, 256 * 6))
for i, t in enumerate(tiles):
    W.paste(t, (256 * (i % 6), 256 * (i // 6)))
W.save(f"{out_dir}/diag3.jpg")
