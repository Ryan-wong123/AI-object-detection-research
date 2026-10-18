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


class _Idx:
    cat_name = {1: "person", 2: "train", 3: "dog"}

    def image_anns(self, sid):
        return [
            {"id": 1, "category_id": 2, "bbox": [0, 0, 100, 100], "iscrowd": 0},
            {"id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
        ]


def test_outside_classes_and_version_layout():
    from inlayfake.stages.replan import outside_classes, version_class

    cfg = {
        "seed": 1,
        "edit": {"max_replans": 2, "max_class_switches": 1, "max_outside_classes": 2},
        "placement": {"exclude_classes": ["person"], "outside_classes": None},
        "sampling": {"categories": ["chair", "train", "person", "dog", "cup"]},
    }
    outs = outside_classes(cfg, _Idx(), 7, [])
    assert sorted(outs) == ["chair", "cup", "dog"]  # not train (in photo), not person
    assert outs == outside_classes(cfg, _Idx(), 7, [])  # deterministic per source
    plan = {"source_id": 7, "category": "train"}
    assert version_class(cfg, _Idx(), plan, 1, []) == ("train", True)  # new box, same class
    assert version_class(cfg, _Idx(), plan, 2, []) == ("train", True)
    assert version_class(cfg, _Idx(), plan, 3, []) == (None, True)  # no other class in photo
    assert version_class(cfg, _Idx(), plan, 4, []) == (outs[0], False)  # not in the photo
    assert version_class(cfg, _Idx(), plan, 5, []) == (outs[1], False)
    assert version_class(cfg, _Idx(), plan, 6, []) == (None, False)


def test_outside_class_is_marked_in_ledger(cfg, tmp_path):
    """Every box of the photo's own classes fails for C4 and C5; a class not in the photo then
    passes and the ledger says class_in_photo = False for C4 and the C2/C3 that follow it."""
    from inlayfake.data.coco import CocoIndex

    cfg = deep_merge(cfg, {"placement": {"class_choice": "in_photo"}})
    sample.run(cfg)
    plan.run(cfg)
    wd = tmp_path / "work"
    sid = json.loads((wd / "plans.jsonl").open().readline())["source_id"]
    cfg = deep_merge(
        cfg,
        {
            "generators_enabled": ["G1"],
            "edit": {"max_replans": 0, "max_class_switches": 1, "max_outside_classes": 1},
            # own class and the class switch fail, for both edits of this photo
            "verify": {"fake_fail_until": {f"{sid}|C4": 2, f"{sid}|C5": 2}},
        },
    )
    insert.run(cfg)
    final = [r for r in read_ledger(wd / "ledger.jsonl") if r.source_id == sid]
    c4 = next(r for r in final if r.condition.value == "C4")
    index = CocoIndex.load(cfg["paths"]["coco_annotations"], cfg["paths"]["coco_images"])
    own = {index.cat_name[a["category_id"]] for a in index.image_anns(sid)}
    assert c4.edit.plan_version == 2 and c4.edit.class_in_photo is False
    assert c4.edit.target_category not in own
    for cond in ("C2", "C3"):
        r = next(x for x in final if x.condition.value == cond)
        assert r.edit.class_in_photo is False and r.edit.target_category == c4.edit.target_category
    others = [r for r in read_ledger(wd / "ledger.jsonl") if r.source_id != sid and r.edit]
    assert all(r.edit.class_in_photo for r in others)


def test_no_outside_class_when_photo_already_has_an_ai_object():
    """C4 ran out of in-photo versions, but C5 passed with the photo's own class: C4 must not
    move on to a class that is not in the photo (it is dropped instead)."""
    cfg = {
        "edit": {
            "max_attempts": 1,
            "max_replans": 0,
            "max_class_switches": 1,
            "max_outside_classes": 2,
        },
        "conditions": ["C4", "C5"],
        "generation": {"insert_generators": ["G1"], "occlusion_generators": ["G1"]},
        "generators_enabled": ["G1"],
    }
    c4_fail = [{"passed": False, "version": 0}, {"passed": False, "version": 1}]
    state = {"7|C4|G1": c4_fail, "7|C5|G1": [{"passed": True, "version": 0}]}
    assert track_version(cfg, state, 7, FREE, {}) == 1  # capped at the last in-photo version
    assert next_job(cfg, state, 7, "C4", "G1", {}) is None  # dropped
    # C5 still trying with the photo's own classes: wait, do not go outside yet
    state["7|C5|G1"] = [{"passed": False, "version": 0}]
    assert track_version(cfg, state, 7, FREE, {}) == 1
    # C5 exhausted too: now C4 may use a class that is not in the photo
    state["7|C5|G1"] = [{"passed": False, "version": 0}, {"passed": False, "version": 1}]
    assert track_version(cfg, state, 7, FREE, {}) == 2
    # a verified outside-class edit is kept even if C5 later passes with an in-photo class
    state["7|C4|G1"] = [*c4_fail, {"passed": True, "version": 2}]
    state["7|C5|G1"] = [{"passed": True, "version": 0}]
    assert track_version(cfg, state, 7, FREE, {}) == 2
    assert next_job(cfg, state, 7, "C4", "G1", {}) is None  # done: nothing more to try
    # ...and while C5 is still trying
    state["7|C5|G1"] = [{"passed": False, "version": 0}]
    assert track_version(cfg, state, 7, FREE, {}) == 2


def test_place_anywhere_hand_example():
    import numpy as np

    from inlayfake.edit.placement import place_anywhere

    gt = [
        {"id": 1, "bbox": [0, 0, 400, 300], "iscrowd": 0},  # fills the frame
        {"id": 2, "bbox": [0, 0, 10, 10], "iscrowd": 1},  # crowd: never the covered object
    ]
    for seed in range(20):
        box, occ = place_anywhere(400, 300, 80, 60, gt, np.random.default_rng(seed))
        assert 0 <= box[0] and box[0] + box[2] <= 400 and 0 <= box[1] and box[1] + box[3] <= 300
        assert box[2:] == [80.0, 60.0] and occ == 1
    big, _ = place_anywhere(400, 300, 900, 900, gt, np.random.default_rng(0))
    assert big[2] <= 0.95 * 400 and big[3] <= 0.95 * 300  # shrunk to fit


def test_every_photo_gets_a_box_when_anywhere_is_allowed(cfg, tmp_path):
    """Rules no box can satisfy (huge margin, impossible occlusion): every photo still gets a
    C5 plan placed anywhere, with one of its own classes, flagged in the plan and the ledger."""
    from inlayfake.data.coco import CocoIndex

    cfg = deep_merge(
        cfg,
        {
            "generators_enabled": ["G1"],
            "placement": {
                "class_choice": "in_photo",
                "allow_anywhere": True,
                "margin_px": 10000,  # no free space anywhere
                "occlusion_frac": [0.99, 1.0],  # no rule-abiding occlusion either
            },
        },
    )
    sample.run(cfg)
    summary = plan.run(cfg)
    n = len([1 for _ in (tmp_path / "work" / "sources.jsonl").open()])
    assert summary["n_plans"] == n  # every photo has a box
    assert summary["n_placed_anywhere"] >= 1
    index = CocoIndex.load(cfg["paths"]["coco_annotations"], cfg["paths"]["coco_images"])
    rows = [json.loads(line) for line in (tmp_path / "work" / "plans.jsonl").open()]
    rows = [r for r in rows if r.get("placement_rule") == "anywhere"]
    assert len(rows) == summary["n_placed_anywhere"]
    for r in rows:
        own = {index.cat_name[a["category_id"]] for a in index.image_anns(r["source_id"])}
        assert r["placement_rule"] == "anywhere" and r["box_free"] is None
        assert r["category"] in own and r["class_in_photo"]
    insert.run(cfg)
    anywhere_ids = {r["source_id"] for r in rows}
    c5 = [r for r in read_ledger(tmp_path / "work" / "ledger.jsonl") if r.condition.value == "C5"]
    assert {r.source_id for r in c5 if r.edit.placement_rule == "anywhere"} == anywhere_ids


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


def test_merged_version_hand_example():
    """Merged AI condition: keep a verified C4, else adopt a verified C5, else boxes anywhere."""
    from inlayfake.stages.replan import ADOPT, merged_version

    cfg = {
        "edit": {"max_attempts": 2, "max_replans": 1},
        "conditions": ["C2", "C3", "C4", "C5"],
        "placement": {"merged": {"enabled": True, "versions": 2}},
        "generation": {
            "insert_generators": ["G1"],
            "occlusion_generators": ["G1"],
            "sham_generators": ["G1"],
        },
        "generators_enabled": ["G1"],
    }
    ok = lambda v: {"passed": True, "version": v, "image_id": f"x{v}"}  # noqa: E731
    no = lambda v: {"passed": False, "version": v}  # noqa: E731
    # verified C4 kept (even with a verified C5); C5 never generated again
    state = {"7|C4|G1": [no(0), ok(0)], "7|C5|G1": [ok(0)]}
    assert merged_version(cfg, state, 7) == 0
    assert next_job(cfg, state, 7, "C5", "G1") is None
    # no C4, verified C5: adopted; no new C4, C2 is made at the C5 box
    state = {"7|C4|G1": [no(0)] * 2 + [no(1)], "7|C5|G1": [no(0), ok(0)]}
    assert merged_version(cfg, state, 7) == ADOPT
    assert next_job(cfg, state, 7, "C4", "G1") is None
    assert next_job(cfg, state, 7, "C2", "G1") == (ADOPT, 0)
    # neither: boxes anywhere; C2 waits until the AI edit there is verified
    state = {"7|C4|G1": [no(0)], "7|C5|G1": [no(0)] * 2}
    assert next_job(cfg, state, 7, "C4", "G1") == (ADOPT + 1, 0)
    assert next_job(cfg, state, 7, "C2", "G1") is None
    state["7|C4|G1"] += [no(ADOPT + 1)] * 2
    assert next_job(cfg, state, 7, "C4", "G1") == (ADOPT + 2, 0)
    state["7|C4|G1"] += [ok(ADOPT + 2)]
    assert next_job(cfg, state, 7, "C2", "G1") == (ADOPT + 2, 0)
    state = {"7|C4|G1": [no(ADOPT + 1)] * 2 + [no(ADOPT + 2)] * 2}
    assert next_job(cfg, state, 7, "C4", "G1") is None  # every anywhere box failed: dropped


def test_merged_condition_end_to_end(cfg, tmp_path):
    """Run with C4/C5 first, then switch to the merged condition: photo A (only C5 verified)
    adopts its C5 and gets C2/C3 at that box; photo B (neither) gets a box anywhere; photo C
    keeps its C4. The ledger has one AI edit (labelled C4) per photo and no C5."""
    from inlayfake.data.coco import CocoIndex
    from inlayfake.stages.replan import ADOPT

    cfg = deep_merge(cfg, {"placement": {"class_choice": "in_photo"}})
    sample.run(cfg)
    plan.run(cfg)
    wd = tmp_path / "work"
    plans = [json.loads(line) for line in (wd / "plans.jsonl").open()]
    both = [p for p in plans if p["box_free"] is not None and p.get("box_occ") is not None]
    a, b, c = (p["source_id"] for p in both[:3])
    fail = {f"{a}|C4": 99, f"{b}|C4": ADOPT + 2, f"{b}|C5": 99}
    insert.run(_cfg(cfg, fake_fail_until=fail))
    mcfg = deep_merge(_cfg(cfg, fake_fail_until=fail), {"placement": {"merged": {"enabled": True}}})
    out = insert.run(mcfg)
    assert out["final"]["needs_retry"] == 0
    final = list(read_ledger(wd / "ledger.jsonl"))
    assert not any(r.condition.value == "C5" for r in final)
    by = {s: [r for r in final if r.source_id == s] for s in (a, b, c)}
    for s, recs in by.items():
        ai = [r for r in recs if r.condition.value == "C4"]
        c2 = [r for r in recs if r.condition.value == "C2"]
        c3 = [r for r in recs if r.condition.value == "C3"]
        assert len(ai) == len(c2) == len(c3) == 1, s
        assert ai[0].edit.plan_box == c2[0].edit.plan_box == c3[0].edit.plan_box
        assert c2[0].edit.plan_version == c3[0].edit.plan_version
    ai = {s: next(r for r in recs if r.condition.value == "C4") for s, recs in by.items()}
    assert ai[a].edit.merged_source == "C5" and "|C5|" not in ai[a].image_id
    assert "C5" in ai[a].image_id and ai[a].edit.plan_box == both[0]["box_occ"]
    assert ai[b].edit.merged_source == "anywhere" and ai[b].edit.plan_version == ADOPT + 2
    assert ai[b].edit.placement_rule == "anywhere"
    index = CocoIndex.load(cfg["paths"]["coco_annotations"], cfg["paths"]["coco_images"])
    own = {index.cat_name[x["category_id"]] for x in index.image_anns(b)}
    assert ai[b].edit.target_category in own
    assert ai[c].edit.merged_source == "C4" and ai[c].edit.plan_version == 0
    assert out["final"]["merged_ai_source"].get("C5", 0) >= 1


def test_merged_preview(cfg, tmp_path):
    """The merged preview shows one AI edit (C4) per photo with its source, and no C5."""
    from inlayfake.stages import preview

    cfg = deep_merge(cfg, {"placement": {"class_choice": "in_photo"}})
    sample.run(cfg)
    plan.run(cfg)
    mcfg = deep_merge(
        cfg, {"generators_enabled": ["G1"], "placement": {"merged": {"enabled": True}}}
    )
    insert.run(mcfg)
    out = preview.run(mcfg, limit=None)
    page = (tmp_path / "work" / "preview.html").read_text(encoding="utf-8")
    assert out["sources_complete"] == out["sources"]
    assert out["ai_source"]["C4"] + out["ai_source"]["anywhere"] == out["sources"]
    # a fresh run has no verified C4/C5, so every AI edit is placed anywhere
    assert "C5 G1" not in page and "new: placed anywhere" in page
    assert 'value="ai-C5"' in page


def test_merged_box_avoids_real_same_class_object(monkeypatch):
    """From placement.merged.avoid_same_class_from on, a merged box covers at most 30% of its
    area with a real object of its class; earlier boxes are unchanged."""
    import numpy as np

    from inlayfake.edit.placement import intersection
    from inlayfake.stages.replan import ADOPT, make_merged_version

    class Index:
        cat_name = {1: "train"}

        def image_anns(self, sid):
            return [{"id": 1, "category_id": 1, "bbox": [0, 0, 600, 400], "iscrowd": 0}]

    cfg = {
        "seed": 3,
        "placement": {
            "exclude_classes": ["person"],
            "min_side_px": 16,
            "max_area_frac": 0.25,
            "merged": {"enabled": True, "versions": 12, "avoid_same_class_from": 7},
        },
    }
    plan = {"source_id": 9, "category": "train", "class_in_photo": True}
    import inlayfake.stages.replan as rp

    monkeypatch.setattr(rp, "_size", lambda cfg, index, anns, cat, W, H, rng: (200.0, 150.0))
    real = [0, 0, 600, 400]
    for v in range(ADOPT + 7, ADOPT + 13, 2):
        row = make_merged_version(cfg, Index(), {"width": 640, "height": 480}, plan, v, {}, {}, [])
        b = row["box"]
        assert intersection(b, real) <= 0.3 * b[2] * b[3] + 1e-6, (v, b)
    assert np.isfinite(b).all()
