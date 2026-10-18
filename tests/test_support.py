"""Surface placement (placement.support): boxes rest on floor / table pixels from COCO-Stuff."""

import json

import numpy as np

from inlayfake.edit.placement import intersection, place_free, place_occluding
from inlayfake.edit.support import INSIDE, ON, StuffIndex, Support, support_frac


def floor_mask(H=300, W=400, top=200):
    m = np.zeros((H, W), dtype=bool)
    m[top:] = True  # "floor": every row from `top` down
    return m


def test_support_frac_hand_computed():
    m = floor_mask()
    # box bottom at y=220, h=100: band rows 205..225 (15 above, 5 below), cols 10..90 -> all floor
    assert support_frac([0, 120, 100, 100], m, ON, (0.15, 0.05)) == 1.0
    # box bottom at y=200: rows 185..205 -> 5 of 20 rows are floor
    assert support_frac([0, 100, 100, 100], m, ON, (0.15, 0.05)) == 0.25
    # floating box well above the floor
    assert support_frac([0, 0, 100, 100], m, ON, (0.15, 0.05)) == 0.0
    # "inside": box rows 150..250 -> half of them are floor
    assert support_frac([0, 150, 100, 100], m, INSIDE, (0.15, 0.05)) == 0.5


def test_place_free_rests_on_surface_and_avoids_gt():
    rng = np.random.default_rng(0)
    sup = Support("ground", floor_mask(), ON, (0.15, 0.05), 0.5)
    gt = [[0, 180, 150, 120]]  # an object standing on the left part of the floor
    for _ in range(30):
        b = place_free(400, 300, 40, 60, gt, 8, rng, 200, support=sup)
        assert b is not None
        assert b[1] + b[3] > 200  # bottom edge is on the floor, not in the air
        assert support_frac(b, sup.mask, ON, (0.15, 0.05)) >= 0.5
        assert 0 <= b[0] and b[0] + b[2] <= 400 and b[1] + b[3] <= 300
        assert intersection(b, [-8, 172, 166, 136]) == 0
    empty = Support("ground", np.zeros((300, 400), dtype=bool), ON, (0.15, 0.05), 0.5)
    assert place_free(400, 300, 40, 60, [], 8, rng, 50, support=empty) is None


def test_place_occluding_with_support():
    rng = np.random.default_rng(1)
    sup = Support("ground", floor_mask(), ON, (0.15, 0.05), 0.5)
    target = [150, 160, 100, 100]  # stands on the floor (bottom at 260)
    for _ in range(20):
        b = place_occluding(400, 300, 60, 60, target, [], (0.2, 0.5), rng, 1000, support=sup)
        assert b is not None
        assert 0.2 <= intersection(b, target) / 10000 <= 0.5
        assert support_frac(b, sup.mask, ON, (0.15, 0.05)) >= 0.5


def test_stuff_index_decodes_rle(tmp_path):
    from pycocotools import mask as mask_utils

    m = np.zeros((20, 30), dtype=np.uint8, order="F")
    m[12:, :] = 1
    rle = mask_utils.encode(m)
    rle["counts"] = rle["counts"].decode("ascii")  # COCO-Stuff JSON stores counts as str
    data = {
        "images": [],
        "categories": [{"id": 1, "name": "floor-wood"}, {"id": 2, "name": "sky-other"}],
        "annotations": [
            {"image_id": 5, "category_id": 1, "segmentation": rle},
            {"image_id": 5, "category_id": 2, "segmentation": rle},
        ],
    }
    p = tmp_path / "stuff.json"
    p.write_text(json.dumps(data))
    s = StuffIndex.load(p)
    got = s.mask(5, {"floor-wood"}, 20, 30)
    assert got.sum() == 8 * 30 and got[12:].all()
    assert not s.mask(5, {"grass"}, 20, 30).any()
    assert not s.mask(99, {"floor-wood"}, 20, 30).any()


def test_plan_stage_with_support(cfg, tmp_path):
    """End to end: every plan rests on the fixture's floor (lower 40% of each 400×320 photo)."""
    from pycocotools import mask as mask_utils

    from inlayfake.config import deep_merge
    from inlayfake.stages import plan, sample

    m = np.zeros((320, 400), dtype=np.uint8, order="F")
    m[192:, :] = 1
    rle = mask_utils.encode(m)
    rle["counts"] = rle["counts"].decode("ascii")
    stuff = {
        "images": [],
        "categories": [{"id": 1, "name": "floor-wood"}, {"id": 2, "name": "table"}],
        "annotations": [
            {"image_id": i, "category_id": c, "segmentation": rle}
            for i in range(1, 25)
            for c in (1, 2)
        ],
    }
    p = tmp_path / "stuff.json"
    p.write_text(json.dumps(stuff))
    cfg = deep_merge(
        cfg, {"placement": {"support": {"enabled": True, "stuff_annotations": str(p)}}}
    )
    sample.run(cfg)
    summary = plan.run(cfg)
    assert summary["n_plans"] > 0
    rows = [json.loads(line) for line in (tmp_path / "work" / "plans.jsonl").open()]
    for r in rows:
        assert r["support"] in ("ground", "tabletop")
        x, y, w, h = r["box_free"]
        assert y + h > 192
