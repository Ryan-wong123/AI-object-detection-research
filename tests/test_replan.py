"""Re-plan on failure (edit.max_replans): a job whose seeds all failed gets a new box."""

import json

from inlayfake.config import deep_merge
from inlayfake.ledger import Origin, read_ledger
from inlayfake.stages import insert, plan, sample
from inlayfake.stages.replan import FREE, job_seed, next_job, track_version


def _cfg(cfg, **verify):
    return deep_merge(
        dict(cfg),
        {"generators_enabled": ["G1"], "edit": {"max_replans": 2}, "verify": verify},
    )


def test_track_version_and_next_job_hand_example():
    cfg = {
        "edit": {"max_attempts": 3, "max_replans": 2},
        "conditions": ["C2", "C3", "C4", "C5"],
        "generation": {"insert_generators": ["G1"], "occlusion_generators": ["G1"]},
        "generators_enabled": ["G1"],
    }
    fail = {"passed": False, "version": 0}
    state = {"7|C4|G1": [fail, fail]}
    assert track_version(cfg, state, 7, FREE) == 0
    assert next_job(cfg, state, 7, "C4", "G1") == (0, 2)
    state = {"7|C4|G1": [fail] * 3, "7|C2|G1": [{"passed": True, "version": 0}]}
    assert track_version(cfg, state, 7, FREE) == 1  # 3 seeds failed -> box 1
    assert next_job(cfg, state, 7, "C4", "G1") == (1, 0)
    assert next_job(cfg, state, 7, "C2", "G1") == (1, 0)  # sham redone at the new box
    v1 = [{"passed": False, "version": 1}] * 3
    v2 = [{"passed": False, "version": 2}] * 3
    state = {"7|C4|G1": [fail] * 3 + v1 + v2}
    assert track_version(cfg, state, 7, FREE) == 2  # max_replans reached
    assert next_job(cfg, state, 7, "C4", "G1") is None  # dropped
    # version 0 keeps the original seeds (existing images reused); later versions differ
    base = {"seed": 5}
    assert job_seed(base, 7, "C4", "G1", 1, 0) != job_seed(base, 7, "C4", "G1", 1, 1)


def test_replan_moves_c2_c3_with_c4(cfg, tmp_path):
    """C4 fails on its original box, passes on box 1; C2 and C3 follow it to box 1."""
    sample.run(cfg)
    plan.run(cfg)
    wd = tmp_path / "work"
    first = json.loads((wd / "plans.jsonl").open().readline())
    sid = first["source_id"]
    cfg = _cfg(cfg, fake_fail_until={f"{sid}|C4": 1})
    out = insert.run(cfg)
    assert out["final"]["needs_retry"] == 0
    assert out["final"]["by_condition"]["C4"].get("passed_on_box_1", 0) >= 1
    final = [r for r in read_ledger(wd / "ledger.jsonl") if r.source_id == sid]
    c4 = next(r for r in final if r.condition.value == "C4")
    c2 = next(r for r in final if r.condition.value == "C2")
    c3 = [r for r in final if r.condition.value == "C3"]
    assert c4.edit.plan_version == 1 and c4.edit.plan_box != first["box_free"]
    assert c2.edit.plan_version == 1 and c2.edit.plan_box == c4.edit.plan_box
    assert len(c3) == 1 and c3[0].edit.plan_version == 1
    assert c3[0].edit.plan_box == c4.edit.plan_box
    assert any(i.origin is Origin.AI_INSERTED for i in c4.instances)
    rows = [json.loads(line) for line in (wd / "replans.jsonl").open()]
    assert any(r["source_id"] == sid and r["track"] == "free" and r["version"] == 1 for r in rows)


def test_version_without_box_is_skipped():
    cfg = {
        "edit": {"max_attempts": 1, "max_replans": 1, "max_class_switches": 2},
        "conditions": ["C4"],
        "generation": {"insert_generators": ["G1"], "occlusion_generators": ["G1"]},
        "generators_enabled": ["G1"],
    }
    state = {"7|C4|G1": [{"passed": False, "version": 0}]}
    versions = {(7, FREE, 1): {"box": None}, (7, FREE, 2): {"box": [0, 0, 9, 9]}}
    assert track_version(cfg, state, 7, FREE, versions) == 2  # v1 had no box -> v2
    assert next_job(cfg, state, 7, "C4", "G1", versions) == (2, 0)
    versions[(7, FREE, 2)] = {"box": None}
    versions[(7, FREE, 3)] = {"box": None}
    assert track_version(cfg, state, 7, FREE, versions) == 3
    assert next_job(cfg, state, 7, "C4", "G1", versions) is None  # nothing left: dropped


def test_class_switch_after_boxes_fail(cfg, tmp_path):
    """C4 fails on the planned class's boxes, passes with another class already in the photo;
    C2 and C3 switch class with it."""
    from inlayfake.data.coco import CocoIndex

    cfg = deep_merge(cfg, {"placement": {"class_choice": "in_photo"}})
    sample.run(cfg)
    plan.run(cfg)
    wd = tmp_path / "work"
    first = json.loads((wd / "plans.jsonl").open().readline())
    sid = first["source_id"]
    cfg = deep_merge(
        cfg,
        {
            "generators_enabled": ["G1"],
            "edit": {"max_replans": 1, "max_class_switches": 1},
            "verify": {"fake_fail_until": {f"{sid}|C4": 2}},  # box 0 and box 1 fail
        },
    )
    insert.run(cfg)
    final = [r for r in read_ledger(wd / "ledger.jsonl") if r.source_id == sid]
    c4 = next(r for r in final if r.condition.value == "C4")
    c2 = next(r for r in final if r.condition.value == "C2")
    c3 = next(r for r in final if r.condition.value == "C3")
    index = CocoIndex.load(cfg["paths"]["coco_annotations"], cfg["paths"]["coco_images"])
    own = {index.cat_name[a["category_id"]] for a in index.image_anns(sid)}
    assert c4.edit.plan_version == 2
    assert c4.edit.target_category != first["category"] and c4.edit.target_category in own
    assert c2.edit.target_category == c3.edit.target_category == c4.edit.target_category
    assert c2.edit.plan_box == c3.edit.plan_box == c4.edit.plan_box


def test_occlusion_only_plan_keeps_other_plans():
    """A photo without free space gets a C5-only plan from its own generator; the shared
    generator (every later photo's plan) is untouched."""
    import numpy as np

    from inlayfake.edit.placement import plan_source

    pcfg = {
        "min_side_px": 32,
        "max_area_frac": 0.25,
        "margin_px": 8,
        "occlusion_frac": [0.2, 0.5],
        "max_tries": 200,
    }
    stats = {"cup": np.array([[0.2, 1.0]])}  # a ~155 px square box
    full = [  # two objects fill the frame: no free space, but a box can cover 40% of one
        {"id": 1, "bbox": [0, 0, 200, 300], "iscrowd": 0, "category_id": 2},
        {"id": 2, "bbox": [200, 0, 200, 300], "iscrowd": 0, "category_id": 2},
    ]
    src = {"source_id": 5, "split": "test", "width": 400, "height": 300}
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    assert plan_source(src, full, ["cup"], stats, pcfg, a) is None
    p = plan_source(src, full, ["cup"], stats, pcfg, b, occ_only_rng=np.random.default_rng(9))
    assert p["box_free"] is None and p["box_occ"] is not None and p["occluded_ann_id"] in (1, 2)
    assert a.random() == b.random()  # same state of the shared generator afterwards


def test_occluding_box_sized_to_a_frame_filling_object():
    """One train filling 2/3 of the frame: a typical-size box (2% of the image) can never cover
    20% of it, a box sized from the train can."""
    import numpy as np

    from inlayfake.edit.placement import intersection, occluding_sized_to_target, place_occluding

    pcfg = {
        "min_side_px": 32,
        "max_area_frac": 0.25,
        "occlusion_frac": [0.2, 0.5],
        "max_tries": 200,
    }
    train = {"id": 1, "bbox": [20, 40, 589, 328], "iscrowd": 0}  # 589x328 in 640x453
    rng = np.random.default_rng(0)
    small = place_occluding(640, 453, 76, 76, train["bbox"], [], (0.2, 0.5), rng, 200)
    assert small is None  # 76x76 = 3% of the train box at most
    for seed in range(10):
        box, tid = occluding_sized_to_target(
            640, 453, 2.0, [train], [train], pcfg, np.random.default_rng(seed)
        )
        assert tid == 1 and box is not None
        frac = intersection(box, train["bbox"]) / (589 * 328)
        assert 0.2 <= frac <= 0.5
        assert box[2] * box[3] <= 0.25 * 640 * 453 + 1e-6
        assert 0 <= box[0] and box[0] + box[2] <= 640 and 0 <= box[1] and box[1] + box[3] <= 453


def test_replan_drops_after_all_boxes(cfg, tmp_path):
    sample.run(cfg)
    plan.run(cfg)
    wd = tmp_path / "work"
    sid = json.loads((wd / "plans.jsonl").open().readline())["source_id"]
    cfg = _cfg(cfg, fake_fail_until={f"{sid}|C4": 99})
    out = insert.run(cfg)
    assert out["final"]["needs_retry"] == 0
    rows = [json.loads(line) for line in (wd / "verify.jsonl").open()]
    mine = [r for r in rows if r["key"] == f"{sid}|C4|G1"]
    assert len(mine) == 9 and not any(r["passed"] for r in mine)  # 3 boxes x 3 seeds
    final = [r for r in read_ledger(wd / "ledger.jsonl") if r.source_id == sid]
    assert not any(r.condition.value == "C4" for r in final)
    assert sum(r.condition.value == "C3" for r in final) == 1  # only the last box's paste
