"""Week 2: verification, retries and the final ledger (CPU, fake inpainter + fake verifier)."""

import csv
import json

import numpy as np

from inlayfake.config import deep_merge
from inlayfake.edit.verify import best_candidate, changed_mask, occlusion_update
from inlayfake.ledger import Origin, read_ledger
from inlayfake.stages import generate, insert, plan, sample
from inlayfake.stages.verify import next_attempt


def test_changed_mask_ignores_isolated_noise():
    ref = np.zeros((40, 40, 3), np.uint8)
    edit = ref.copy()
    edit[10:20, 10:20] = 50  # a real change
    edit[30, 30] = 50  # a single noisy pixel
    m = changed_mask(edit, ref, 4)
    assert m[10:20, 10:20].all() and not m[30, 30] and m.sum() == 100


def test_occlusion_and_best_candidate():
    orig = np.zeros((10, 10), bool)
    orig[0:10, 0:4] = True
    ins = np.zeros((10, 10), bool)
    ins[0:5, 0:10] = True
    visible, frac = occlusion_update(orig, ins)
    assert frac == 0.5 and visible[5:, :4].all() and not visible[:5].any()
    cands = [([0, 0, 10, 10], 0.9), ([50, 50, 10, 10], 0.95), ([1, 1, 10, 10], 0.1)]
    assert best_candidate(cands, [0, 0, 10, 10], 0.3, 0.2)[0] == [0, 0, 10, 10]
    assert best_candidate(cands, [80, 80, 5, 5], 0.3, 0.2) is None


def test_best_candidate_accepts_small_object_inside_box():
    from inlayfake.edit.verify import best_candidate, inside_frac

    plan = [100, 100, 200, 100]  # 200 x 100 box
    small = [150, 140, 20, 20]  # a small object fully inside: IoU = 400 / 20000 = 0.02
    assert inside_frac(small, plan) == 1.0
    assert best_candidate([(small, 0.7)], plan, 0.3, 0.2) is None  # IoU rule alone rejects it
    got = best_candidate([(small, 0.7)], plan, 0.3, 0.2, min_inside=0.8)
    assert got is not None and got[0] == small and abs(got[2] - 0.02) < 1e-9
    # half outside the box: 50% inside -> still rejected
    straddle = [290, 140, 20, 20]
    assert best_candidate([(straddle, 0.7)], plan, 0.3, 0.2, min_inside=0.8) is None
    # too weak a detection never counts
    assert best_candidate([(small, 0.1)], plan, 0.3, 0.2, min_inside=0.8) is None
    # the photo's own same-class object (C5: the occluded original) is not the insertion
    orig = [140, 130, 30, 40]
    on_orig = [142, 132, 28, 36]  # IoU with orig ≈ 0.84
    assert best_candidate([(on_orig, 0.9)], plan, 0.3, 0.2, 0.8, [orig]) is None
    both = best_candidate([(on_orig, 0.9), (small, 0.5)], plan, 0.3, 0.2, 0.8, [orig])
    assert both[0] == small


def test_next_attempt():
    state = {"k": [{"passed": False}]}
    assert next_attempt({}, "k", 3) == 0
    assert next_attempt(state, "k", 3) == 1
    assert next_attempt({"k": [{"passed": False}] * 3}, "k", 3) is None
    assert next_attempt({"k": [{"passed": True}]}, "k", 3) is None


def _cfg(cfg, **verify_overrides):
    return deep_merge(dict(cfg), {"generators_enabled": ["G1"], "verify": verify_overrides})


def test_insert_end_to_end(cfg, tmp_path):
    cfg = _cfg(cfg)
    sample.run(cfg)
    plan.run(cfg)
    out = insert.run(cfg)
    assert out["final"]["needs_retry"] == 0
    wd = tmp_path / "work"
    final = list(read_ledger(wd / "ledger.jsonl"))
    conds = {r.condition.value for r in final}
    assert {"C0", "C1", "C2", "C3", "C4", "C5"} <= conds

    # C0/C1 are carried over unchanged, so week-1 results are unaffected
    gen = {r.image_id: r for r in read_ledger(wd / "ledger_generated.jsonl")}
    for r in final:
        if r.condition.value in ("C0", "C1"):
            assert r == gen[r.image_id]

    c4 = next(r for r in final if r.condition.value == "C4")
    ins = next(i for i in c4.instances if i.origin is Origin.AI_INSERTED)
    assert ins.mask_path and c4.edit.actual_mask_path and c4.edit.verifier_score == 0.9
    px, py, pw, ph = c4.edit.plan_box
    assert ins.bbox[0] >= px - 1 and ins.bbox[0] + ins.bbox[2] <= px + pw + 1  # SAM box in plan

    c5 = next(r for r in final if r.condition.value == "C5")
    assert any(i.visible_fraction < 1 for i in c5.instances if i.origin is Origin.PHOTOGRAPHED) or (
        c5.ignore_regions
    )
    c3 = next(r for r in final if r.condition.value == "C3")
    assert next(i for i in c3.instances if i.origin is Origin.REAL_PASTED).mask_path

    rows = list(csv.DictReader((wd / "audit_edits.csv").open()))
    assert rows and (wd / "audit_edits.html").exists()


def test_failed_edit_is_regenerated_with_new_seed(cfg, tmp_path):
    cfg = _cfg(cfg)
    sample.run(cfg)
    plan.run(cfg)
    generate.run(cfg)
    wd = tmp_path / "work"
    first_c4 = next(
        r for r in read_ledger(wd / "ledger_generated.jsonl") if r.condition.value == "C4"
    )
    cfg = _cfg(cfg, fake_fail_ids=[first_c4.image_id])  # the verifier rejects attempt 1 only
    out = insert.run(cfg)
    assert out["final"]["needs_retry"] == 0
    rows = [json.loads(line) for line in (wd / "verify.jsonl").open()]
    key = f"{first_c4.source_id}|C4|G1"
    mine = [r for r in rows if r["key"] == key]
    assert [r["passed"] for r in mine] == [False, True]
    assert mine[0]["image_id"] != mine[1]["image_id"]  # new seed -> new image
    final_ids = {r.image_id for r in read_ledger(wd / "ledger.jsonl")}
    assert mine[1]["image_id"] in final_ids and first_c4.image_id not in final_ids

    # the preview never presents the rejected attempt as a benchmark image
    from inlayfake.stages import preview

    preview.run(cfg)
    page = (wd / "preview.html").read_text(encoding="utf-8")
    fig = page.split(f"{first_c4.image_id}.jpg")[0].rsplit("<figure", 1)[1]
    assert 'class="rej"' in fig
    after = page.split(f"{first_c4.image_id}.jpg")[1].split("</figure>")[0]
    assert "rejected" in after and "not in benchmark" in after
    ok = page.split(f"{mine[1]['image_id']}.jpg")[1].split("</figure>")[0]
    assert "in benchmark" in ok and "rejected" not in ok


def test_tight_window_changes_generation_not_verification(cfg, tmp_path):
    """edit.window_scale 1.2 shrinks the generator window; verify still uses the 2.5x region."""
    from inlayfake.edit.compose import crop_window
    from inlayfake.stages.verify import verify_window

    cfg = deep_merge(_cfg(cfg), {"edit": {"window_scale": 1.2, "window_min_px": 64}})
    sample.run(cfg)
    plan.run(cfg)
    insert.run(cfg)
    wd = tmp_path / "work"
    edits = [r for r in read_ledger(wd / "ledger_generated.jsonl") if r.condition.value == "C4"]
    assert edits
    for r in edits:
        x, y, w, h = r.edit.plan_box
        side = r.edit.crop_box[2] - r.edit.crop_box[0]
        assert (
            side
            == crop_window(r.edit.plan_box, r.width, r.height, 1.2, 64)[2]
            - (crop_window(r.edit.plan_box, r.width, r.height, 1.2, 64)[0])
        )
        assert side < 2.5 * max(w, h)
        v = verify_window(r, r.width, r.height, cfg["verify"])
        assert v == crop_window(r.edit.plan_box, r.width, r.height, 2.5, 256)


def test_viewer_shows_edited_conditions(cfg, tmp_path):
    """After verify, the viewer lists C2-C5 and marks detectors not yet run on them."""
    from inlayfake.stages import detect, viewer

    cfg = _cfg(cfg)
    sample.run(cfg)
    plan.run(cfg)
    insert.run(cfg)
    detect.run(dict(cfg, conditions=["C0", "C1"]))  # week 1: detectors on real photos only
    viewer.run(cfg)
    data = json.loads(
        (tmp_path / "runs/dryrun/viewer/data.js").read_text().split("=", 1)[1].rstrip().rstrip(";")
    )
    assert {"C2/G1", "C3", "C4/G1", "C5/G1"} <= set(data["conditions"])
    im = next(i for i in data["images"] if "C4/G1" in i["cond"])
    assert im["cond"]["C4/G1"]["det"]["fake_a"] is None  # not run yet
    assert im["cond"]["C0"]["det"]["fake_a"] is not None
    assert any(g[5] == "inserted" for g in im["cond"]["C4/G1"]["gt"])
