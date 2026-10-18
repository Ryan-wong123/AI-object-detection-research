"""Diagnostic only: why does OWLv2 reject visible insertions? Re-score round-1 failures on CPU.

For every rejected C4/C5 edit of task2_pilot_inphoto (round 1), list OWLv2 candidates of the
target class and report: best score with IoU >= 0.3 to the plan box (the current rule), and best
score of a candidate lying mostly inside the plan box (>= 80% of its area), at any IoU.
"""
import json
import sys

from inlayfake.edit.encode import load_rgb
from inlayfake.edit.placement import intersection, iou
from inlayfake.edit.verify import OwlV2Finder
from inlayfake.ledger import ImageRecord
from inlayfake.stages.verify import verify_window

wd = "data/inlayfake_pilot_inphoto"
rows = [json.loads(l) for l in open(f"{wd}/verify.jsonl")][:116]  # round 1 only
fail = [r for r in rows if r["condition"] in ("C4", "C5") and not r["passed"]]
finder = OwlV2Finder("google/owlv2-base-patch16-ensemble", "cpu", 0.05)
out = []
for r in fail:
    rec = ImageRecord(**json.load(open(f"{wd}/records/{r['image_id']}.json")))
    x0, y0, x1, y1 = verify_window(rec, rec.width, rec.height, {})
    img = load_rgb(rec.path)
    cands = [
        ([b[0] + x0, b[1] + y0, b[2] - b[0], b[3] - b[1]], s)
        for b, s in finder(img[y0:y1, x0:x1], rec.edit.target_category)
    ]
    plan = rec.edit.plan_box
    rule = max((s for b, s in cands if iou(b, plan) >= 0.3), default=0.0)
    inside = [(s, iou(b, plan)) for b, s in cands if intersection(b, plan) >= 0.8 * b[2] * b[3]]
    best_in = max(inside, default=(0.0, 0.0))
    out.append((r["image_id"], rec.edit.target_category, rule, best_in[0], best_in[1]))
    print(f"{r['image_id']:40s} {rec.edit.target_category:14s} rule_best={rule:.2f} "
          f"inside_best={best_in[0]:.2f} (iou {best_in[1]:.2f})", flush=True)
n = len(out)
print("failures", n,
      "| inside-candidate score>=0.2:", sum(o[3] >= 0.2 for o in out),
      "| of those IoU<0.3:", sum(o[3] >= 0.2 and o[4] < 0.3 for o in out),
      "| IoU-ok candidate with score 0.1-0.2:", sum(0.1 <= o[2] < 0.2 for o in out))
