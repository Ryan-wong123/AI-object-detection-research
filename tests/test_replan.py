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
