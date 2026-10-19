import numpy as np
import pytest

from inlayfake.data.audit import apply_decisions, find_candidates
from inlayfake.eval.bootstrap import grouped_bootstrap
from inlayfake.eval.coco_ap import build_coco, coco_ap
from inlayfake.ledger import ImageRecord, Origin


def _rec(sid=1, cond="C0"):
    return ImageRecord(
        ImageRecord.make_id(sid, cond, None, None),
        sid,
        "test",
        cond,
        "x.jpg",
        200,
        200,
        instances=[
            {
                "instance_id": "coco:1",
                "category": "cup",
                "bbox": [10, 10, 40, 40],
                "origin": "photographed",
            },
            {
                "instance_id": "coco:2",
                "category": "dog",
                "bbox": [100, 100, 60, 50],
                "origin": "photographed",
            },
        ],
    )


def test_perfect_detections_give_ap_one():
    r = _rec()
    dets = {
        r.image_id: [{"category": i.category, "bbox": i.bbox, "score": 0.9} for i in r.instances]
    }
    m = coco_ap(*build_coco([r], dets, ["cup", "dog"]))
    assert m["AP"] == pytest.approx(1.0) and m["AR100"] == pytest.approx(1.0)


def test_missing_detection_lowers_ap_and_origin_filter_ignores():
    r = _rec()
    dets = {r.image_id: [{"category": "cup", "bbox": [10, 10, 40, 40], "score": 0.9}]}
    assert coco_ap(*build_coco([r], dets, ["cup", "dog"]))["AP"] < 1.0
    # restricting scoring to an origin no instance has → all become ignore → no penalty
    gt, res = build_coco([r], dets, ["cup", "dog"], origins={"ai_inserted"})
    assert all(a["iscrowd"] == 1 for a in gt["annotations"])


def test_grouped_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    groups = {i: list(rng.normal(1.0, 0.1, size=3)) for i in range(50)}
    p, lo, hi = grouped_bootstrap(groups, n=300, rng=rng)
    assert lo < p < hi and abs(p - 1.0) < 0.05


def test_audit_candidates_and_decisions():
    r = _rec()
    unlabeled = {"category": "chair", "bbox": [150, 10, 30, 40], "score": 0.9}
    known = {"category": "cup", "bbox": [10, 10, 40, 40], "score": 0.95}
    lone = {"category": "vase", "bbox": [60, 150, 20, 30], "score": 0.8}
    cands = find_candidates(r, {"a": [unlabeled, known, lone], "b": [unlabeled]}, 0.7, 2)
    assert [c["category"] for c in cands] == ["chair"]  # known → matched, lone → no agreement

    decisions = [
        {
            "candidate_id": "1-0",
            "source_id": "1",
            "category": "chair",
            "x": 150,
            "y": 10,
            "w": 30,
            "h": 40,
            "decision": "add",
        },
        {
            "candidate_id": "1-1",
            "source_id": "1",
            "category": "vase",
            "x": 60,
            "y": 150,
            "w": 20,
            "h": 30,
            "decision": "?",
        },
    ]
    (r0,) = apply_decisions([r], decisions)
    assert any(
        i.instance_id == "audit:1-0" and i.origin is Origin.PHOTOGRAPHED for i in r0.instances
    )
    assert r0.ignore_regions == [{"category": "vase", "bbox": [60.0, 150.0, 20.0, 30.0]}]
    assert len(r.instances) == 2  # original record untouched


def test_report_rows_count_found_objects_per_detector():
    from inlayfake.stages.report import image_rows, label_summary, rows_markdown

    r = _rec(sid=7)  # cup [10,10,40,40] and dog [100,100,60,50]
    dets = {
        # finds the cup; the dog box is the right class but scores below 0.5
        "a": {
            r.image_id: [
                {"category": "cup", "bbox": [12, 10, 40, 40], "score": 0.9},
                {"category": "dog", "bbox": [100, 100, 60, 50], "score": 0.4},
            ]
        },
        # wrong class on the cup, finds the dog
        "b": {
            r.image_id: [
                {"category": "bowl", "bbox": [10, 10, 40, 40], "score": 0.9},
                {"category": "dog", "bbox": [102, 101, 58, 50], "score": 0.8},
            ]
        },
    }
    rows = image_rows([r], dets, {"7": "chair"}, iou_thr=0.5, score_thr=0.5)
    assert rows == [
        {
            "source_id": 7,
            "split": "test",
            "planned_insert": "chair",
            "n_objects": 2,
            "labels": "cup, dog",
            "found_a": 1,
            "found_b": 1,
        }
    ]
    assert label_summary(["person", "cup", "person", "dog", "person"]) == "person ×3, cup, dog"
    md = rows_markdown(rows, ["a", "b"], {"a": "A"})
    assert "[7](http://images.cocodataset.org/val2017/000000000007.jpg)" in md
    assert md.strip().endswith("| 1/2 | 1/2 |")
