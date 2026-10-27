"""Task 3 detector metrics (H1, H2): hand-computed examples."""

import math

import numpy as np
import pytest

from inlayfake.eval.bootstrap import grouped_bootstrap_fn, grouped_mean_ci
from inlayfake.eval.coco_ap import _evaluated, _stats, build_coco, coco_ap
from inlayfake.eval.matched import (
    box_gap,
    by_source,
    contrast_rows,
    covered_fraction,
    matched_rows,
    origin_rows,
    stratum,
    summarize_matched,
    summarize_origin,
)
from inlayfake.eval.origin import auroc
from inlayfake.ledger import ImageRecord, Origin

PLAN = [100, 100, 40, 40]  # edit box shared by C2/C3/C4; footprint = PLAN grown by 6 + 8 px


def _inst(iid, cat, bbox, origin="photographed"):
    return {"instance_id": iid, "category": cat, "bbox": bbox, "origin": origin}


def _rec(cond, extra=(), modified=False):
    far = _inst("coco:1", "cup", [300, 300, 40, 40])  # far from the edit
    covered = _inst(
        "coco:2", "dog", [90, 90, 30, 30], "mixed_modified" if modified else "photographed"
    )
    edit = {"target_category": "cup", "seed": 1, "plan_box": PLAN} if cond != "C1" else None
    return ImageRecord(
        f"7__{cond}",
        7,
        "test",
        cond,
        "x.jpg",
        400,
        400,
        edit=edit,
        instances=[far, covered, *extra],
    )


def test_auroc_ties_count_half():
    # pairs (pos, neg): (1,0)=1 (1,2)=0 (2,0)=1 (2,2)=0.5 -> 2.5 / 4
    assert auroc([1, 2], [0, 2]) == pytest.approx(0.625)
    assert auroc([3], [1, 2]) == 1.0 and auroc([0], [1]) == 0.0
    assert math.isnan(auroc([], [1]))


def test_grouped_mean_ci_and_fn():
    groups = {"a": [1.0, 0.0, float("nan")], "b": [1.0]}
    p, lo, hi = grouped_mean_ci(groups, n=500, rng=np.random.default_rng(0))
    assert p == pytest.approx(2 / 3) and lo <= p <= hi
    assert grouped_mean_ci(groups, n=0)[0] == pytest.approx(2 / 3)
    p, _, _ = grouped_bootstrap_fn(groups, lambda vs: float(len(vs)), n=10)
    assert p == 2.0


def test_geometry_and_strata():
    assert covered_fraction([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(0.5)
    assert box_gap([0, 0, 10, 10], [13, 14, 5, 5]) == pytest.approx(5.0)  # dx 3, dy 4
    fp = [86, 86, 68, 68]  # PLAN grown by 14
    assert stratum([90, 90, 30, 30], Origin.PHOTOGRAPHED, fp) == "covered"
    assert stratum([160, 90, 20, 20], Origin.PHOTOGRAPHED, fp) == "uncovered_near"  # gap 6
    assert stratum([300, 300, 40, 40], Origin.PHOTOGRAPHED, fp) == "uncovered_far"
    assert stratum([90, 90, 30, 30], Origin.MIXED_MODIFIED, fp) == "modified"
    assert stratum([90, 90, 30, 30], Origin.PHOTOGRAPHED, None) == "no_edit"


def _det(cat, bbox, score):
    return {"category": cat, "bbox": bbox, "score": score}


def _fixture():
    ai = _inst("ins:7", "cup", [105, 105, 30, 30], "ai_inserted")
    paste = _inst("paste:7", "cup", [105, 105, 30, 30], "real_pasted")
    recs = [
        _rec("C1"),
        _rec("C2", modified=True),
        _rec("C3", extra=[paste]),
        _rec("C4", extra=[ai]),
    ]
    far, dog = [300, 300, 40, 40], [90, 90, 30, 30]
    dets = {
        "7__C1": [_det("cup", far, 0.9), _det("dog", dog, 0.8)],
        "7__C2": [_det("cup", far, 0.85), _det("cat", dog, 0.7)],
        "7__C3": [_det("cup", far, 0.9), _det("dog", dog, 0.8), _det("cup", paste["bbox"], 0.3)],
        # the far cup is lost (0.9 -> 0.4) and the dog's top class becomes "cat"
        "7__C4": [
            _det("cup", far, 0.4),
            _det("cat", dog, 0.9),
            _det("dog", dog, 0.6),
            _det("cup", ai["bbox"], 0.95),
        ],
    }
    return by_source(recs), dets


def test_matched_rows_follow_each_original_from_c1():
    sources, dets = _fixture()
    rows = matched_rows(sources, dets, ["C1", "C2", "C3", "C4"], 0.5, 0.5, 6)
    got = {(r["condition"], r["instance_id"]): r for r in rows}
    assert len(rows) == 6  # 2 originals × C2, C3, C4
    assert got[("C2", "coco:2")]["stratum"] == "modified"
    assert got[("C4", "coco:2")]["stratum"] == "covered"
    assert got[("C4", "coco:1")]["stratum"] == "uncovered_far"
    r = got[("C4", "coco:1")]
    assert r["det_c1"] and not r["det_cx"] and r["score_cx"] == pytest.approx(0.4)
    assert got[("C4", "coco:2")]["class_cx"] == "cat"  # dog still found (0.6) but cat on top

    summary = {(s["condition"], s["stratum"]): s for s in summarize_matched(rows, 0, None)}
    s = summary[("C4", "uncovered")]
    assert s["n_objects"] == 1 and s["lost"] == 1.0 and s["flip"] == 1.0
    assert s["abs_score_delta"] == pytest.approx(0.5)
    s = summary[("C4", "covered")]
    assert s["flip"] == 0.0 and s["class_flip"] == 1.0 and s["score_delta"] == pytest.approx(-0.2)
    assert s["box_drift"] == pytest.approx(0.0)

    (c2, c3) = contrast_rows(rows, 0, None, controls=("C2", "C3"))
    assert c2["contrast"] == "C4 - C2" and c2["n_objects"] == 1
    assert c2["flip_diff"] == 1.0  # C4 lost the far cup, C2 did not
    assert c2["abs_score_delta_diff"] == pytest.approx(0.5 - 0.05)
    assert c3["flip_diff"] == 1.0 and c3["abs_score_delta_diff"] == pytest.approx(0.5)
    assert c2.keys() == c3.keys()  # one CSV header for every contrast


def test_origin_rows_and_summary():
    sources, dets = _fixture()
    rows = origin_rows(sources, dets, 0.5)
    groups = sorted((r["group"], r["category"], r["score"]) for r in rows)
    assert groups == [
        ("ai", "cup", 0.95),
        ("other_photo", "dog", 0.6),
        ("pasted", "cup", 0.3),
        ("twin", "cup", 0.4),
    ]
    out = {r["comparison"]: r for r in summarize_origin(rows, 0.5, 0, None)}
    t = out["ai vs twin"]
    assert t["recall_ai"] == 1.0 and t["recall_other"] == 0.0 and t["recall_diff"] == 1.0
    assert t["auroc"] == 1.0
    p = out["ai vs photographed"]
    assert p["n_other"] == 2 and p["recall_other"] == pytest.approx(0.5)
    assert out["ai vs pasted"]["auroc"] == 1.0


def test_fast_ap_resample_equals_rebuilding_duplicates():
    sources, dets = _fixture()
    recs = [sources[7][c] for c in ("C1", "C2", "C3", "C4")]
    cats = ["cat", "cup", "dog"]
    for origins in (None, {"photographed"}, {"ai_inserted"}):
        ev = _evaluated(*build_coco(recs, dets, cats, origins))
        for idx in ([0, 0, 3, 1], [2, 2, 2, 2], [3, 1]):
            slow = coco_ap(*build_coco([recs[i] for i in idx], dets, cats, origins))
            assert _stats(ev, idx) == pytest.approx(slow)
