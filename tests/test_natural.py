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
    assert [q[0] for q in qs] == ["supported", "through_glass", "size", "whole"]
    assert "or flying in the sky" in qs[0][1]  # air class
    assert qs[1][2] == "no"  # seen through glass: must be "no"
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


def test_point_support_rules():
    """Fix 1: a point must be on a surface the class rests on, or on an object that carries it."""
    from inlayfake.edit.natural import point_support

    p = load_prompts(PROMPTS)
    horse, cup, bird = p["classes"]["horse"], p["classes"]["cup"], p["classes"]["bird"]
    assert point_support("grass", None, horse) == (True, "ok")
    assert point_support("wall-other", None, horse) == (False, "point on wall-other")
    assert point_support("other", None, horse) == (False, "point on an unlabelled object")
    assert point_support("other", "tv", horse) == (False, "point on a real tv")
    assert point_support("other", "dining table", cup) == (True, "ok")  # cup on a table object
    assert point_support("counter", None, cup) == (True, "ok")
    assert point_support("road", None, cup)[0] is False
    assert point_support("sky-other", None, bird) == (True, "ok")
    assert point_support(None, None, horse) == (True, "ok")  # no COCO-Stuff labels: unchecked


def test_minimum_box_area():
    """Fix 2: a box is enlarged (shape kept) to at least min_area_frac of the image."""
    depth = np.full((400, 500), 0.5, dtype=np.float32)
    ref = {"id": 1, "bbox": [100.0, 100.0, 40.0, 20.0]}  # 800 px² = 0.4% of the image
    box, _ = box_from_point(
        (250.0, 300.0), depth, [ref], (10, 10), False, (0.7, 1.4), 0.25, 32, 0.02
    )
    w, h = box[2], box[3]
    assert np.isclose(w * h, 0.02 * 500 * 400) and np.isclose(w / h, 2.0)
    assert np.isclose(box[1] + h, 300.0)  # still standing on the point


def test_box_fill_rule_rejects_tiny_objects(tmp_path):
    """Fix 3: verify rejects an AI object filling less than verify.min_box_fill of its box."""
    from PIL import Image

    from inlayfake.ledger import EditInfo, ImageRecord, Instance, Origin
    from inlayfake.stages.verify import verify_one

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mpath = tmp_path / "m.png"
    Image.fromarray(np.full((200, 200), 255, dtype=np.uint8)).save(mpath)
    rec = ImageRecord(
        "x",
        1,
        "train",
        "C4",
        "x.jpg",
        200,
        200,
        generator="G1",
        edit=EditInfo("cup", 0, [40, 40, 100, 100], intended_mask_path=str(mpath)),
        instances=[Instance("ins:x", "cup", [40, 40, 100, 100], Origin.AI_INSERTED)],
    )
    finder = lambda crop, cat, r: [([60, 60, 120, 120], 0.9)]  # noqa: E731  in-box detection

    def seg(size):  # SAM stand-in: a size×size object in the box's corner
        def f(crop, box):
            m = np.zeros(crop.shape[:2], dtype=bool)
            x0, y0 = int(box[0]), int(box[1])
            m[y0 : y0 + size, x0 : x0 + size] = True
            return m

        return f

    vcfg = {
        "min_iou": 0.3,
        "min_score": 0.2,
        "min_inside": 0.8,
        "changed_pixel_threshold": 4,
        "mixed_modified_overlap": 0.1,
        "window_scale": 2.5,
        "window_min_px": 64,
        "min_box_fill": 0.35,
    }
    ok, why, info = verify_one(rec, img, img, finder, seg(40), vcfg, None, tmp_path)  # 16% fill
    assert not ok and why == "object fills only 16% of its box" and info["box_fill"] == 0.16
    ok, _, info = verify_one(rec, img, img, finder, seg(60), vcfg, None, tmp_path)  # 36%
    assert ok and info["box_fill"] == 0.36


def test_person_overlap_limit():
    """max_cover_person > 0: a person may be covered up to that share of their box."""
    rules = {"max_cover_real": 0.30, "max_cover_person": 0.30}
    anns = [{"category": "person", "bbox": [0, 0, 100, 100]}]
    assert check_box([80, 0, 50, 50], anns, None, rules) == (True, "ok")  # 1000 px² = 10%
    ok, why = check_box([60, 0, 50, 100], anns, None, rules)  # 4000 px² = 40%
    assert not ok and why == "covers 40% of a person"
    assert check_box([80, 0, 50, 50], anns, None, {"max_cover_person": 0.0}) == (
        False,
        "covers a person",
    )


def test_natural_plan_resumes_with_identical_plans(cfg, tmp_path, monkeypatch):
    """A planning run cut off after some photos continues from natural_plan_progress.jsonl:
    finished photos are not asked again, and the plans equal those of an uninterrupted run."""
    import inlayfake.edit.scene as scene
    from inlayfake.stages import plan, sample

    cfg = _natural_cfg(cfg)
    sample.run(cfg)
    plan.run(cfg)
    wd = tmp_path / "work"
    full_plans = (wd / "plans.jsonl").read_text()
    full_cands = (wd / "natural_candidates.jsonl").read_text()
    lines = (wd / "natural_plan_progress.jsonl").read_text().splitlines()
    assert len(lines) >= 4
    # simulate a crash after 3 photos, the last line cut off mid-write
    (wd / "natural_plan_progress.jsonl").write_text("\n".join(lines[:3]) + "\n" + lines[3][:20])
    (wd / "plans.jsonl").unlink()
    (wd / "natural_points.jsonl").unlink()  # nor were the later photos' VLM answers saved
    calls = []
    real_point = scene.FakeVLM.point

    def counting_point(self, img, prompt):
        calls.append(prompt)
        return real_point(self, img, prompt)

    monkeypatch.setattr(scene.FakeVLM, "point", counting_point)
    plan.run(cfg)
    assert (wd / "plans.jsonl").read_text() == full_plans
    assert (wd / "natural_candidates.jsonl").read_text() == full_cands
    n_done = len({json.loads(x)["source_id"] for x in lines[:3]})
    total = len({json.loads(x)["source_id"] for x in lines})
    assert 0 < len(calls) and n_done > 0 and total > n_done  # only the unfinished photos asked


def test_rule_change_rechecks_saved_photos_without_asking_the_vlm(cfg, tmp_path, monkeypatch):
    """After a planning run, changing a box rule re-checks every saved photo with the new rule
    using the saved VLM answers (no new question), and gives the same plans as a fresh run with
    the new rule."""
    import inlayfake.edit.scene as scene
    from inlayfake.stages import plan, sample

    base = _natural_cfg(cfg)
    sample.run(base)
    plan.run(base)
    loose = deep_merge(base, {"placement": {"natural": {"box_rules": {"max_cover_real": 0.9}}}})
    calls = []
    real_point = scene.FakeVLM.point

    def counting_point(self, img, prompt):
        calls.append(prompt)
        return real_point(self, img, prompt)

    monkeypatch.setattr(scene.FakeVLM, "point", counting_point)
    summary = plan.run(loose)
    assert calls == [] and summary["vlm_questions"]["asked"] == 0
    assert summary["photos_rechecked"] > 0
    rows = [json.loads(x) for x in (tmp_path / "work" / "natural_candidates.jsonl").open()]
    real = [r["reason"] for r in rows if " of a real " in (r["reason"] or "")]
    assert all(int(x.split()[1].rstrip("%")) > 90 for x in real)  # the new 90% limit applied

    fresh = deep_merge(loose, {"paths": {"work": str(tmp_path / "fresh")}})
    sample.run(fresh)
    plan.run(fresh)
    for name in ("plans.jsonl", "natural_candidates.jsonl"):
        assert (tmp_path / "work" / name).read_text() == (tmp_path / "fresh" / name).read_text()
    # the same settings again: every photo is reused as saved
    again = plan.run(loose)
    assert again["photos_rechecked"] == 0 and again["vlm_questions"] == {"asked": 0, "reused": 0}


def test_saved_molmo_answers_are_read_back_from_old_progress_rows(tmp_path):
    """Progress rows saved before the answer cache existed still spare the VLM question: the
    Molmo answer text is parsed again (percent of width/height)."""
    from inlayfake.stages.natural import _load_points, _point_key, _vlm_id

    vcfg = {"backend": "molmo", "model_id": "m"}
    row = {
        "source_id": 7,
        "category": "cup",
        "prompt": "Point to a table",
        "answer": '<point x="50.0" y="25.0" alt="x">x</point>',
        "vlm": "molmo",
    }
    cache = _load_points(tmp_path, [row], {7: (200, 100)}, vcfg)
    hit = cache[_point_key(7, "cup", "Point to a table", _vlm_id(vcfg))]
    assert hit["points"] == [(100.0, 25.0)] and hit["vlm"] == "molmo"


def test_plan_settings_do_not_depend_on_set_order(cfg):
    """The saved settings must compare equal across Python processes: the prompt file's surface
    sets are stored sorted (set order changes with each process's string hashing)."""
    from inlayfake.edit.natural import load_prompts
    from inlayfake.stages.natural import plan_settings

    c = _natural_cfg(cfg)
    prompts = load_prompts(PROMPTS)
    a = plan_settings(c, prompts, {})
    flipped = {
        **prompts,
        "classes": {
            k: {**v, "surfaces": set(sorted(v["surfaces"], reverse=True))}
            for k, v in prompts["classes"].items()
        },
    }
    assert plan_settings(c, flipped, {}) == a
    s = next(iter(a["prompts"]["classes"].values()))["surfaces"]
    assert isinstance(s, list) and s == sorted(s)


def test_parse_bbox_2d_and_dedupe():
    """Qwen3-VL boxes on the 0-1000 grid become pixel [x, y, w, h]; near-duplicates are dropped."""
    from inlayfake.edit.natural import dedupe_boxes
    from inlayfake.edit.scene import parse_bbox_2d

    text = '[{"bbox_2d": [100, 500, 300, 700], "label": "cup"}, {"bbox_2d": [0, 0, 1200, 50]}]'
    boxes = parse_bbox_2d(text, 640, 480)
    assert boxes[0] == [64.0, 240.0, 128.0, 96.0]
    assert boxes[1] == [0.0, 0.0, 640.0, 24.0]  # clipped to the image
    assert parse_bbox_2d("[]", 640, 480) == []
    a, b, c = [0, 0, 100, 100], [5, 5, 100, 100], [300, 300, 50, 50]
    assert dedupe_boxes([a, b, c], 5) == [a, c]
    assert dedupe_boxes([a, b, c], 1) == [a]


def test_box_prompt_fills_the_class_slots():
    from inlayfake.edit.natural import box_prompt, load_prompts

    p = box_prompt(load_prompts(PROMPTS), "tennis racket", [[64, 240, 128, 96]], 640, 480)
    assert "new tennis racket" in p and "lie on the ground" in p and "bbox_2d" in p
    assert "one tennis racket at bbox_2d [100, 500, 300, 700]" in p  # existing one, 0-1000 grid
    assert "{" not in p.replace('[{"bbox_2d"', "").replace("}]", "")  # every slot filled


def test_box_mode_plans_use_the_vlm_boxes(cfg, tmp_path):
    """mode: box takes the VLM's own boxes (after the same rules), records size_from 'vlm', and
    reuses the saved answers on a second run."""
    import inlayfake.edit.scene as scene
    from inlayfake.stages import plan, sample

    c = _natural_cfg(cfg, mode="box", check_point_surface=False)
    sample.run(c)
    summary = plan.run(c)
    wd = tmp_path / "work"
    cands = [json.loads(x) for x in (wd / "natural_candidates.jsonl").read_text().splitlines()]
    ok = [r for r in cands if r["ok"]]
    assert ok and all(r["size_from"] == "vlm" for r in ok)
    plans = [json.loads(x) for x in (wd / "plans.jsonl").read_text().splitlines()]
    assert summary["n_plans"] == len(plans) > 0
    for p in plans:  # the plan's box is one of the fake VLM's flat boxes (scaled to the photo)
        x, y, w, h = p["box_free"]
        assert w > h
    calls = []
    orig = scene.FakeVLM.boxes
    scene.FakeVLM.boxes = lambda self, img, prompt: calls.append(1) or orig(self, img, prompt)
    try:
        again = plan.run(c)
    finally:
        scene.FakeVLM.boxes = orig
    assert calls == [] and again["n_plans"] == summary["n_plans"]


def test_surface_fixes_animals_in_plants_and_pole_classes():
    """Ground animals may stand in bushes or tall plants; traffic lights and stop signs (on a pole)
    skip the support check (author, 21 Oct)."""
    from inlayfake.edit.natural import load_prompts, point_support

    c = load_prompts(PROMPTS)["classes"]
    assert point_support("bush", None, c["elephant"]) == (True, "ok")
    assert point_support("sky-other", None, c["traffic light"]) == (True, "ok")
    assert point_support(None, "car", c["traffic light"]) == (True, "ok")
    assert point_support("bush", None, c["car"])[0] is False


def _tiny_then(good_when: str):
    """Fake box answers: a tiny (rejected) box, unless the question contains ``good_when``."""
    import json as _json

    def boxes(self, img, prompt):
        H, W = img.shape[:2]
        if good_when in prompt:
            b = [0.35 * W, 0.8 * H, 0.3 * W, 0.15 * H]
            lab = "chair" if "Choose only from" in prompt else "new object"
        else:
            b, lab = [0.5 * W, 0.9 * H, 4.0, 4.0], "new object"
        g = [round(b[0] / W * 1000), round(b[1] / H * 1000)]
        g += [round((b[0] + b[2]) / W * 1000), round((b[1] + b[3]) / H * 1000)]
        return [b], _json.dumps([{"bbox_2d": g, "label": lab}])

    return boxes


def test_box_mode_retry_round_finds_a_spot(cfg, tmp_path, monkeypatch):
    """retry: when the first boxes all fail, the second question (taken places + minimum size)
    is asked and its box can become the plan."""
    import inlayfake.edit.scene as scene
    from inlayfake.stages import plan, sample

    monkeypatch.setattr(scene.FakeVLM, "boxes", _tiny_then("already taken"))
    c = _natural_cfg(cfg, mode="box", check_point_surface=False, retry=True)
    sample.run(c)
    summary = plan.run(c)
    rows = [json.loads(x) for x in (tmp_path / "work" / "natural_candidates.jsonl").open()]
    assert summary["n_plans"] > 0
    assert all(r["round"] == 2 for r in rows if r["ok"])
    assert any(r["round"] == 1 and r["reason"] == "too small" for r in rows)
    retry = next(r for r in rows if r["round"] == 2)
    assert "at least" in retry["prompt"] and "already taken" in retry["prompt"]


def test_box_mode_any_class_round_is_flagged(cfg, tmp_path, monkeypatch):
    """any_class: when no class of the photo works, the VLM may pick a listed class; the plan is
    flagged class_in_photo false unless that class is in the photo."""
    import inlayfake.edit.scene as scene
    from inlayfake.stages import plan, sample

    monkeypatch.setattr(scene.FakeVLM, "boxes", _tiny_then("Choose only from"))
    c = _natural_cfg(cfg, mode="box", check_point_surface=False, retry=True, any_class=True)
    sample.run(c)
    summary = plan.run(c)
    wd = tmp_path / "work"
    plans = [json.loads(x) for x in (wd / "plans.jsonl").read_text().splitlines()]
    rows = [json.loads(x) for x in (wd / "natural_candidates.jsonl").open()]
    assert summary["n_plans"] == len(plans) > 0
    assert all(p["category"] == "chair" for p in plans)
    ok = [r for r in rows if r["ok"]]
    assert all(r["round"] == "any" for r in ok)
    for p in plans:
        r = next(r for r in ok if r["source_id"] == p["source_id"])
        assert p["class_in_photo"] == r["class_in_photo"]
    assert "person" not in next(r for r in rows if r["round"] == "any")["prompt"].split(
        "Choose only from:"
    )[1].split(".")[0].split(", ")


def test_apply_review_uses_first_accepted_spot():
    """Only accepted spots stay valid (require_accept); the plan moves to the first accepted
    one; a photo with none is dropped; with require_accept false only rejects are removed."""
    import copy

    from inlayfake.stages.spots import apply_review, spot_id

    rows = [
        {"source_id": 1, "category": "cup", "box": [0, 0, 10, 10], "ok": True, "valid_rank": 0},
        {"source_id": 1, "category": "cup", "box": [50, 0, 10, 10], "ok": True, "valid_rank": 1},
        {"source_id": 2, "category": "dog", "box": [0, 0, 20, 20], "ok": True, "valid_rank": 0},
        {
            "source_id": 2,
            "category": "dog",
            "box": [9, 9, 5, 5],
            "ok": False,
            "reason": "too small",
        },
    ]
    plans = [
        {"source_id": 1, "category": "cup", "box_free": [0, 0, 10, 10], "split": "val"},
        {"source_id": 2, "category": "dog", "box_free": [0, 0, 20, 20], "split": "test"},
    ]
    dec = {spot_id(rows[0]): "reject", spot_id(rows[1]): "accept"}
    r1 = copy.deepcopy(rows)
    new_plans, failed = apply_review(r1, copy.deepcopy(plans), [], dec)
    assert [p["source_id"] for p in new_plans] == [1]
    assert new_plans[0]["box_free"] == [50, 0, 10, 10] and new_plans[0]["split"] == "val"
    assert r1[1]["valid_rank"] == 0 and r1[0]["reason"] == "rejected in review"
    assert r1[2]["reason"] == "not reviewed" and failed == [
        {"source_id": 2, "reason": "no reviewed spot"}
    ]
    loose, _ = apply_review(copy.deepcopy(rows), copy.deepcopy(plans), [], dec, False)
    assert [p["source_id"] for p in loose] == [1, 2]


def test_spot_review_page_server_and_plan(cfg, tmp_path):
    """spots builds the page; the local server saves clicks; plan then keeps only accepted
    spots."""
    import threading
    import urllib.request

    from inlayfake.stages import plan, sample, spots

    c = _natural_cfg(cfg, mode="box", check_point_surface=False)
    sample.run(c)
    first = plan.run(c)
    res = spots.run(c)
    page = Path(res["page"]).read_text(encoding="utf-8")
    assert "__DATA__" not in page and "Accept" in page
    data = json.loads(page.split("const PHOTOS = ")[1].split(";\nlet DEC")[0])
    assert len(data) == first["n_plans"]

    import http.server as hs

    real = hs.ThreadingHTTPServer
    started = {}

    class Grab(real):
        def __init__(self, addr, handler):
            super().__init__(("127.0.0.1", 0), handler)
            started["srv"] = self

    hs.ThreadingHTTPServer = Grab
    try:
        t = threading.Thread(target=spots.serve, args=(c,), daemon=True)
        t.start()
        while "srv" not in started:
            pass
        port = started["srv"].server_address[1]
        keep = data[0]["spots"][0]["id"]
        body = json.dumps({"id": keep, "decision": "accept"}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/decision", data=body, method="POST")
        assert urllib.request.urlopen(req).read() == b"ok"
        got = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/decisions.json").read())
        assert got == {keep: "accept"}
    finally:
        hs.ThreadingHTTPServer = real
        started["srv"].shutdown()
    c2 = deep_merge(c, {"placement": {"natural": {"review": {"enabled": True}}}})
    after = plan.run(c2)
    assert after["n_plans"] == 1


def test_more_questions_only_for_photos_without_an_accepted_spot(cfg, tmp_path, monkeypatch):
    """more_questions: a photo whose spots were all rejected in review gets every class and both
    any-class questions; a photo with an accepted spot is planned exactly as before and asks
    nothing new."""
    import inlayfake.edit.scene as scene
    from inlayfake.stages import plan, sample
    from inlayfake.stages.spots import spot_id

    c = _natural_cfg(
        cfg,
        mode="box",
        check_point_surface=False,
        retry=True,
        any_class=True,
        stop_when_valid=True,
        classes_per_photo=99,
    )
    sample.run(c)
    plan.run(c)
    wd = tmp_path / "work"
    before = [json.loads(x) for x in (wd / "natural_candidates.jsonl").open()]
    sids = sorted({r["source_id"] for r in before if r["ok"]})
    assert len(sids) >= 2
    bad, good = sids[0], sids[1]
    dec = {spot_id(r): "reject" for r in before if r["ok"] and r["source_id"] == bad}
    dec |= {spot_id(r): "accept" for r in before if r["ok"] and r["source_id"] == good}
    (wd / "spot_review.json").write_text(json.dumps(dec))
    asked = []
    orig = scene.FakeVLM.boxes

    def spy(self, img, prompt):
        asked.append(prompt)
        return orig(self, img, prompt)

    monkeypatch.setattr(scene.FakeVLM, "boxes", spy)
    more = deep_merge(c, {"placement": {"natural": {"more_questions": True}}})
    plan.run(more)
    after = [json.loads(x) for x in (wd / "natural_candidates.jsonl").open()]
    rounds_bad = {r.get("round") for r in after if r["source_id"] == bad}
    assert {"any", "any2"} <= rounds_bad
    assert [r for r in after if r["source_id"] == good] == [
        r for r in before if r["source_id"] == good
    ]
    assert asked and any("empty areas" in p for p in asked)


def test_more_question_photos_keep_their_questions_after_an_accept(cfg, tmp_path):
    """A photo that got the extra questions keeps them after its new spot is accepted, so the
    accepted spot is still a candidate (and the plan) on the next run."""
    from inlayfake.stages import plan, sample
    from inlayfake.stages.spots import spot_id

    c = _natural_cfg(
        cfg,
        mode="box",
        check_point_surface=False,
        retry=True,
        any_class=True,
        stop_when_valid=True,
        classes_per_photo=99,
    )
    sample.run(c)
    plan.run(c)
    wd = tmp_path / "work"
    before = [json.loads(x) for x in (wd / "natural_candidates.jsonl").open()]
    bad = sorted({r["source_id"] for r in before if r["ok"]})[0]
    dec = {spot_id(r): "reject" for r in before if r["ok"] and r["source_id"] == bad}
    (wd / "spot_review.json").write_text(json.dumps(dec))
    more = deep_merge(c, {"placement": {"natural": {"more_questions": True}}})
    plan.run(more)
    mid = [json.loads(x) for x in (wd / "natural_candidates.jsonl").open()]
    extra = next(r for r in mid if r["source_id"] == bad and r["ok"] and r["round"] == "any")
    dec[spot_id(extra)] = "accept"
    (wd / "spot_review.json").write_text(json.dumps(dec))
    reviewed = deep_merge(more, {"placement": {"natural": {"review": {"enabled": True}}}})
    plan.run(reviewed)
    plans = {json.loads(x)["source_id"]: json.loads(x) for x in (wd / "plans.jsonl").open()}
    assert plans[bad]["box_free"] == extra["box"]


def test_free_mode_keeps_every_vlm_box(cfg, tmp_path, monkeypatch):
    """free: no rules; every box the VLM draws is a valid spot, even a tiny one or one over a
    person, and the first class with a box is used."""
    import inlayfake.edit.scene as scene
    from inlayfake.stages import plan, sample

    monkeypatch.setattr(scene.FakeVLM, "boxes", _tiny_then("never in a prompt"))
    c = _natural_cfg(cfg, mode="box", free=True, stop_when_valid=True, classes_per_photo=99)
    sample.run(c)
    summary = plan.run(c)
    rows = [json.loads(x) for x in (tmp_path / "work" / "natural_candidates.jsonl").open()]
    assert summary["n_plans"] > 0 and set(summary["failed_reasons"]) <= {"no insertable class"}
    assert rows and all(r["ok"] for r in rows if r.get("box"))  # tiny 4 px boxes included
    for sid in {r["source_id"] for r in rows}:  # the first class with a box is used
        assert len({r["category"] for r in rows if r["source_id"] == sid}) == 1


def test_last_resort_same_class_when_no_place(cfg, tmp_path, monkeypatch):
    """last_resort: when every class of the photo got no box, the same classes are asked again
    (cut off by the edge or further back allowed); the box is tagged round 'edge' and the class
    stays one of the photo's own."""
    import inlayfake.edit.scene as scene
    from inlayfake.stages import plan, sample

    def boxes(self, img, prompt):
        H, W = img.shape[:2]
        if "cut off by" not in prompt:
            return [], "[]"
        return [[0.0, 0.5 * H, 0.2 * W, 0.3 * H]], '[{"bbox_2d": [0, 500, 200, 800]}]'

    monkeypatch.setattr(scene.FakeVLM, "boxes", boxes)
    c = _natural_cfg(
        cfg, mode="box", free=True, stop_when_valid=True, classes_per_photo=99, last_resort=True
    )
    sample.run(c)
    summary = plan.run(c)
    wd = tmp_path / "work"
    rows = [json.loads(x) for x in (wd / "natural_candidates.jsonl").open()]
    plans = [json.loads(x) for x in (wd / "plans.jsonl").read_text().splitlines()]
    assert summary["n_plans"] == len(plans) > 0
    ok = [r for r in rows if r["ok"]]
    assert ok and all(r["round"] == "edge" and r["class_in_photo"] for r in ok)
    asked1 = {(r["source_id"], r["category"]) for r in rows if r.get("round") == 1}
    assert {(r["source_id"], r["category"]) for r in ok} <= asked1  # same classes as before
    off = deep_merge(c, {"placement": {"natural": {"last_resort": False}}})
    assert plan.run(off)["n_plans"] == 0


def test_shape_problem_small_and_thin():
    """Too small: short side or area under the limits; too thin: much more elongated than the
    class usually is, whichever way round (a lying bottle is fine)."""
    from inlayfake.edit.natural import shape_problem

    sc = {"min_side_px": 32, "min_area_frac": 0.005, "max_elongation_factor": 3.0}
    assert shape_problem([0, 0, 20, 100], 640, 480, 1.0, sc) == "too small"
    assert shape_problem([0, 0, 36, 36], 640, 480, 1.0, sc) == "too small"  # 0.42% of the image
    assert shape_problem([0, 222, 630, 34], 640, 480, 2.0, sc) == "too thin"  # railing strip
    assert shape_problem([0, 0, 160, 60], 640, 480, 0.4, sc) is None  # lying bottle (2.7 vs 2.5)
    assert shape_problem([0, 0, 120, 100], 640, 480, None, sc) is None


def test_size_check_reasks_with_feedback(cfg, tmp_path, monkeypatch):
    """shape_check: a tiny box is rejected and the same class is asked again with the failed box
    and reason; a good box from the re-ask becomes the plan (round '1+size1'); the question
    carries the size hint."""
    import inlayfake.edit.scene as scene
    from inlayfake.stages import plan, sample

    asked = []

    def boxes(self, img, prompt):
        asked.append(prompt)
        H, W = img.shape[:2]
        if "cannot be used" in prompt:
            return [[0.3 * W, 0.6 * H, 0.3 * W, 0.25 * H]], '[{"bbox_2d": [300, 600, 600, 850]}]'
        return [[0.5 * W, 0.9 * H, 6.0, 6.0]], '[{"bbox_2d": [500, 900, 510, 912]}]'

    monkeypatch.setattr(scene.FakeVLM, "boxes", boxes)
    c = _natural_cfg(
        cfg,
        mode="box",
        free=True,
        stop_when_valid=True,
        classes_per_photo=99,
        shape_check={"min_side_px": 32, "min_area_frac": 0.005, "retries": 2},
    )
    sample.run(c)
    summary = plan.run(c)
    rows = [json.loads(x) for x in (tmp_path / "work" / "natural_candidates.jsonl").open()]
    assert summary["n_plans"] > 0
    assert all(r["round"] == "1+size1" for r in rows if r["ok"])
    assert any(r["reason"] == "too small" and r["round"] == 1 for r in rows)
    assert all("normal proportions" in p for p in asked)
    assert any("too small" in p and "cannot be used" in p for p in asked)
