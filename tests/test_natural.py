"""Natural placement (placement.natural): geometry, prompts, and the plan → insert flow with fake
models (VLM, depth, inpainter, verifier)."""

import json
from pathlib import Path

import numpy as np

from inlayfake.config import deep_merge
from inlayfake.edit.natural import (
    box_from_point,
    check_box,
    check_questions,
    fit_inside,
    load_prompts,
    pointing_prompt,
    usable_references,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "configs" / "placement_prompts.yaml"


def test_prompts_cover_every_class_once():
    p = load_prompts(PROMPTS)
    assert len(p["classes"]) == 79 and "person" not in p["classes"]
    assert pointing_prompt(p, "cup") == (
        "Point to empty spots on a table, counter, desk or shelf where another cup could "
        "naturally rest."
    )
    qs = check_questions(p, "bird")
    assert [q[0] for q in qs] == ["supported", "not_glass", "size", "whole"]
    assert "flying in open sky also counts" in qs[0][1]  # air class
    assert "flying" not in check_questions(p, "dog")[0][1]
    assert p["rules"]["max_cover_real"] == 0.15


def test_box_from_point_hand_example():
    """Reference dog 120x90 whose base has nearness 0.6; a spot with nearness 0.4 gives a dog
    2/3 as tall, same shape, standing on the point."""
    depth = np.full((480, 640), 0.6, dtype=np.float32)
    depth[300:, :] = 0.6
    depth[:300, :] = 0.4
    ref = {"id": 7, "bbox": [400.0, 310.0, 120.0, 90.0]}  # base at y=400 -> 0.6
    box, info = box_from_point((200.0, 250.0), depth, [ref], (50, 50), False)
    assert info["ratio"] == round(0.4 / 0.6, 4) and info["ref_ann_id"] == 7
    assert np.allclose(box, [200 - 40, 250 - 60, 80, 60])  # 80x60, bottom centre on the point
    # the depth ratio is limited to [0.5, 2]
    _, info = box_from_point(
        (200.0, 250.0), depth * np.where(depth < 0.5, 0.1, 1), [ref], (50, 50), False
    )
    assert info["ratio"] == 0.5
    # a flying class pointed at sky: centred on the point, no depth scaling
    box, info = box_from_point((200.0, 100.0), depth, [ref], (50, 50), True)
    assert np.allclose(box, [140, 55, 120, 90]) and info["ratio"] == 1.0


def test_references_skip_cut_off_and_hidden_objects():
    a = {"id": 1, "bbox": [0, 50, 100, 100]}  # touches the left border
    b = {"id": 2, "bbox": [200, 50, 100, 100]}
    c = {"id": 3, "bbox": [210, 60, 80, 80]}  # hides 64% of b
    d = {"id": 4, "bbox": [400, 200, 50, 60]}
    refs = usable_references([a, b, d], [a, b, c, d], 640, 480)
    assert [r["id"] for r in refs] == [4]


def test_box_rules():
    rules = {"max_cover_real": 0.15, "max_cover_person": 0.0}
    anns = [
        {"category": "person", "bbox": [0, 0, 50, 50]},
        {"category": "chair", "bbox": [100, 100, 100, 100]},
    ]
    assert check_box([10, 10, 20, 20], anns, None, rules) == (False, "covers a person")
    ok, why = check_box([180, 180, 50, 50], anns, None, rules)  # 20x20 = 4% of the chair
    assert ok, why
    ok, why = check_box([150, 150, 50, 50], anns, None, rules)  # 25% of the chair
    assert not ok and "25% of a real chair" in why
    blocked = np.zeros((300, 300), dtype=bool)
    blocked[250:, :] = True  # window pixels under the box base
    ok, why = check_box([10, 200, 40, 55], [], blocked, rules)
    assert not ok and "window" in why
    assert fit_inside([-5, 10, 100, 100], 300, 300) == [0, 10, 100, 100]
    assert fit_inside([-50, 10, 100, 100], 300, 300) == [0, 10, 100, 100]  # sideways: up to 50%
    assert fit_inside([-60, 10, 100, 100], 300, 300) is None
    assert fit_inside([10, -20, 100, 100], 300, 300) is None  # up/down: at most 15%


def _natural_cfg(cfg, **natural):
    return deep_merge(
        cfg,
        {
            "conditions": ["C0", "C1", "C2", "C3", "C4"],
            "generators_enabled": ["G1"],
            "placement": {
                "class_choice": "in_photo",
                "natural": {
                    "enabled": True,
                    "prompts": str(PROMPTS),
                    "vlm": {"backend": "fake"} | natural.pop("vlm", {}),
                    "depth": {"model_id": "fake"},
                }
                | natural,
            },
            "edit": {"max_replans": 9, "max_class_switches": 0, "max_outside_classes": 0},
        },
    )


def test_natural_plan_and_insert(cfg, tmp_path):
    """Plans come from VLM points; C2/C3/C4 share the box; the ledger has no C5; every plan's
    candidates and reasons are saved."""
    from inlayfake.ledger import read_ledger
    from inlayfake.stages import insert, plan, sample

    cfg = _natural_cfg(cfg)
    sample.run(cfg)
    summary = plan.run(cfg)
    wd = tmp_path / "work"
    plans = [json.loads(x) for x in (wd / "plans.jsonl").read_text().splitlines()]
    cands = [json.loads(x) for x in (wd / "natural_candidates.jsonl").read_text().splitlines()]
    assert plans and summary["n_plans"] == len(plans)
    assert all(p["placement_rule"] == "natural" and p["box_occ"] is None for p in plans)
    assert summary["candidates"] == len(cands) and all("reason" in c for c in cands)
    first = {c["source_id"]: c for c in cands if c.get("valid_rank") == 0}
    for p in plans:
        assert p["box_free"] == first[p["source_id"]]["box"]
        assert p["category"] == first[p["source_id"]]["category"]
    out = insert.run(cfg)
    assert out["final"]["needs_retry"] == 0
    final = list(read_ledger(wd / "ledger.jsonl"))
    assert not any(r.condition.value == "C5" for r in final)
    for p in plans:
        recs = {r.condition.value: r for r in final if r.source_id == p["source_id"]}
        assert recs["C4"].edit.placement_rule == "natural"
        assert recs["C2"].edit.plan_box == recs["C3"].edit.plan_box == recs["C4"].edit.plan_box
    assert out["final"]["realism_check"]["checked"] >= len(plans)


def test_realism_failure_moves_to_the_next_spot(cfg, tmp_path):
    """A C4 the VLM judges implausible counts as a failed attempt; after 3 seeds the photo moves
    to its next valid natural spot (version 1), and C2/C3 follow."""
    from inlayfake.ledger import read_ledger
    from inlayfake.stages import insert, plan, sample
    from inlayfake.stages.natural import load_valid

    cfg = _natural_cfg(cfg)
    sample.run(cfg)
    plan.run(cfg)
    wd = tmp_path / "work"
    valid = load_valid(wd)
    sid = next(s for s, v in valid.items() if len(v) >= 2)
    # every C4 of this photo on its first spot fails the realism check (fake judge: 3 seeds)
    from inlayfake.ledger import ImageRecord
    from inlayfake.stages.replan import job_seed

    fail = [
        ImageRecord.make_id(sid, "C4", "G1", job_seed(cfg, sid, "C4", "G1", a, 0)) for a in range(3)
    ]
    cfg = deep_merge(
        cfg, {"placement": {"natural": {"vlm": {"backend": "fake", "fake_fail_ids": fail}}}}
    )
    insert.run(cfg)
    real = [json.loads(x) for x in (wd / "realism.jsonl").read_text().splitlines()]
    assert sum(1 for r in real if r["image_id"] in fail and not r["passed"]) == 3
    recs = {r.condition.value: r for r in read_ledger(wd / "ledger.jsonl") if r.source_id == sid}
    assert recs["C4"].edit.plan_version == 1
    assert recs["C4"].edit.plan_box == valid[sid][1]["box"]
    assert recs["C2"].edit.plan_box == recs["C3"].edit.plan_box == recs["C4"].edit.plan_box


def test_spread_points_skips_near_duplicates():
    from inlayfake.edit.natural import spread_points

    pts = [(24, 383), (26, 337), (27, 311), (300, 200), (310, 205), (600, 400)]
    # diagonal of 640x480 is 800 -> min distance 80 px
    # (26, 337) is 46 px and (27, 311) 72 px from (24, 383); (310, 205) is 11 px from (300, 200)
    assert spread_points(pts, 640, 480, k=5) == [(24, 383), (300, 200), (600, 400)]
    assert spread_points(pts, 640, 480, k=2) == [(24, 383), (300, 200)]
