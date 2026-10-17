"""placement.class_choice = in_photo: insert a class already in the photo's audited GT."""

import json
from collections import Counter

import numpy as np

from inlayfake.stages.plan import audit_anns, balanced_order, present_classes


class _Index:
    cat_name = {1: "person", 2: "cow", 3: "sheep", 4: "chair"}


def test_present_classes_excludes_person_crowd_and_counts_audit_adds():
    anns = [
        {"category_id": 1, "iscrowd": 0},  # person: excluded
        {"category_id": 2, "iscrowd": 0},
        {"category_id": 2, "iscrowd": 0},
        {"category_id": 3, "iscrowd": 1},  # crowd only: not a usable instance
    ]
    extra = [
        {"category": "chair", "decision": "add"},
        {"category": "dog", "decision": "ignore"},  # unresolved: not present
    ]
    assert present_classes(_Index(), anns, extra, {"person"}) == ["chair", "cow"]


def test_balanced_order_least_used_first():
    rng = np.random.default_rng(0)
    order = balanced_order(["cow", "chair", "sheep"], Counter({"cow": 3, "chair": 1}), rng)
    assert order == ["sheep", "chair", "cow"]


def test_audit_anns_block_but_skip_rejects(tmp_path):
    p = tmp_path / "audit.csv"
    p.write_text(
        "candidate_id,source_id,category,x,y,w,h,n_detectors,max_score,decision\n"
        "7-0,7,chair,1,2,3,4,2,0.9,add\n"
        "7-1,7,cup,5,6,7,8,2,0.9,reject\n"
        "8-0,8,dog,0,0,9,9,2,0.8,?\n",
        encoding="utf-8",
    )
    out = audit_anns(str(p))
    assert [a["category"] for a in out[7]] == ["chair"]
    assert out[7][0]["bbox"] == [1.0, 2.0, 3.0, 4.0] and out[7][0]["iscrowd"] == 1
    assert out[8][0]["decision"] == "?"
    assert audit_anns(None) == {}


def test_only_sources_restricts_the_plan(cfg, tmp_path):
    """sampling.only_sources keeps exactly the listed photos (e.g. the week-1 baseline set)."""
    from inlayfake.config import deep_merge
    from inlayfake.stages import plan, sample

    sample.run(cfg)
    ids = [json.loads(line)["source_id"] for line in (tmp_path / "work" / "sources.jsonl").open()]
    keep = ids[::2]
    lst = tmp_path / "keep.jsonl"
    lst.write_text("".join(json.dumps({"source_id": i}) + "\n" for i in keep))
    cfg = deep_merge(cfg, {"sampling": {"only_sources": str(lst)}})
    plan.run(cfg)
    got = {json.loads(line)["source_id"] for line in (tmp_path / "work" / "plans.jsonl").open()}
    assert got and got <= set(keep)


def test_occlusion_fill_adds_c5_and_leaves_other_plans_unchanged(cfg, tmp_path):
    """placement.occlusion_fill gives C5 boxes to plans without one; every other plan and every
    C2/C3/C4 box is byte-identical; each added box follows the C5 rules."""
    from inlayfake.config import deep_merge
    from inlayfake.data.coco import CocoIndex
    from inlayfake.edit.placement import box_area, intersection
    from inlayfake.stages import plan, sample

    # small objects and an impossible-for-one-size occlusion range: some plans lack C5
    base = deep_merge(
        cfg, {"placement": {"class_choice": "in_photo", "occlusion_frac": [0.3, 0.5]}}
    )
    sample.run(base)
    plan.run(base)
    wd = tmp_path / "work"
    before = {p["source_id"]: p for p in map(json.loads, (wd / "plans.jsonl").open())}
    filled_cfg = deep_merge(
        base, {"placement": {"occlusion_fill": {"enabled": True, "min_target_px": 32}}}
    )
    summary = plan.run(filled_cfg)
    after = {p["source_id"]: p for p in map(json.loads, (wd / "plans.jsonl").open())}
    assert set(after) == set(before)
    index = CocoIndex.load(cfg["paths"]["coco_annotations"], cfg["paths"]["coco_images"])
    gained = 0
    for sid, p in after.items():
        b = before[sid]
        if b["box_occ"] is not None:
            assert p == b  # plans that had C5 are untouched
            continue
        for k in ("category", "box_free", "split"):
            assert p[k] == b[k]  # the C2/C3/C4 box never moves
        if p["box_occ"] is None:
            continue
        gained += 1
        assert p["occ_filled"] is True
        anns = {a["id"]: a for a in index.image_anns(sid)}
        tgt = anns[p["occluded_ann_id"]]["bbox"]
        assert box_area(tgt) >= 32**2
        cover = intersection(p["box_occ"], tgt) / box_area(tgt)
        assert 0.3 - 1e-9 <= cover <= 0.5 + 1e-9
        for aid, a in anns.items():
            if aid != p["occluded_ann_id"]:
                assert intersection(p["box_occ"], a["bbox"]) <= 0.05 * box_area(a["bbox"]) + 1e-6
    assert summary["n_occlusion_filled"] == gained


def test_plan_stage_in_photo(cfg, tmp_path):
    """Every plan's class is one of that photo's own GT classes, never person."""
    from inlayfake.config import deep_merge
    from inlayfake.data.coco import CocoIndex
    from inlayfake.stages import plan, sample

    cfg = deep_merge(cfg, {"placement": {"class_choice": "in_photo"}})
    sample.run(cfg)
    summary = plan.run(cfg)
    assert summary["n_plans"] > 0
    index = CocoIndex.load(cfg["paths"]["coco_annotations"], cfg["paths"]["coco_images"])
    rows = [json.loads(line) for line in (tmp_path / "work" / "plans.jsonl").open()]
    for r in rows:
        own = {index.cat_name[a["category_id"]] for a in index.image_anns(r["source_id"])}
        assert r["category"] in own and r["category"] != "person"
