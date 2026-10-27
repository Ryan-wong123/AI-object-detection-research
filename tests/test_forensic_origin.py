"""H3 origin metrics: hand-computed examples."""

import pytest

from inlayfake.eval.forensic_origin import auprc, method_rows, object_scores


def test_auprc_by_hand():
    assert auprc([3], [1, 2]) == 1.0  # the positive ranks first
    assert auprc([1], [2, 3]) == pytest.approx(1 / 3)  # third: precision 1/3
    assert auprc([2, 1], [3]) == pytest.approx((1 / 2 + 2 / 3) / 2)


def _objs(sign=1.0):
    objs = []
    for s in (6, 7, 8, 9, 10):  # val positives
        objs.append(
            {
                "split": "val",
                "source_id": f"v{s}",
                "label": 1,
                "kind": "ai_inserted",
                "scores": {"contrast": sign * s},
            }
        )
    for s in (0, 1, 2, 3, 4):  # val negatives
        objs.append(
            {
                "split": "val",
                "source_id": f"v{s}",
                "label": 0,
                "kind": "photographed C1",
                "scores": {"contrast": sign * s},
            }
        )
    for s, lab, kind in (
        (7, 1, "ai_inserted"),
        (9, 1, "ai_inserted"),
        (5, 0, "photographed C1"),
        (8, 0, "pasted (C3)"),
    ):
        objs.append(
            {
                "split": "test",
                "source_id": "t",
                "label": lab,
                "kind": kind,
                "scores": {"contrast": sign * s},
            }
        )
    return objs


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_method_rows_thresholds_from_val(sign):
    (row,) = [r for r in method_rows(_objs(sign), "ela", 0, None) if r["mode"] == "contrast"]
    assert row["direction"] == ("higher=AI" if sign > 0 else "lower=AI")  # chosen on val
    assert row["val_auroc"] == 1.0
    assert row["auroc"] == pytest.approx(0.75)  # (7>5) (7<8) (9>5) (9>8) -> 3 of 4 pairs
    # val 5%-FPR threshold = 95th pct of val negatives = 3.8 -> both test positives above
    assert row["tpr_at_val_fpr5"] == 1.0
    # val 80%-TPR threshold = 20th pct of val positives = 6.8 -> 7, 9 caught; 8 accused, 5 not
    assert row["tpr_at_val_tpr80"] == 1.0
    assert row["false_accusation"] == pytest.approx(0.5)
    assert row["false_accusation pasted (C3)"] == 1.0
    assert row["false_accusation photographed C1"] == 0.0
    assert row["auroc_vs_pasted"] == 0.5  # 7 < 8 < 9: one of two pairs


def test_object_scores_modes_and_exclusions():
    base = {"image_id": "a", "split": "test", "source_id": 1}
    rows = [
        base | {"level": "image", "condition": "C4", "scores": {"p99_minus_median": 2.0}},
        base
        | {
            "level": "instance",
            "condition": "C4",
            "origin": "ai_inserted",
            "scores": {"mean_in": 1.0, "p95_in": 3.0, "contrast": 0.5},
        },
        base
        | {
            "level": "instance",
            "condition": "C4",
            "origin": "mixed_modified",
            "scores": {"mean_in": 1.0, "p95_in": 3.0, "contrast": 0.5},
        },
        base
        | {
            "image_id": "b",
            "level": "instance",
            "condition": "C0",
            "origin": "photographed",
            "scores": {"mean_in": 1.0, "p95_in": 3.0, "contrast": 0.5},
        },
    ]
    (obj,) = object_scores(rows, "ela")  # mixed/modified and C0 are left out
    assert obj["label"] == 1
    assert obj["scores"] == {"image": 2.0, "region_mean": 1.0, "region_p95": 3.0, "contrast": 0.5}


def test_edited_background_flag_rate_on_c2_boxes():
    boxes = [
        {"split": "test", "source_id": "t", "condition": "C2", "scores": {"contrast": 8.0}},
        {"split": "test", "source_id": "u", "condition": "C2", "scores": {"contrast": 5.0}},
        {"split": "val", "source_id": "v", "condition": "C2", "scores": {"contrast": 9.0}},
        {"split": "test", "source_id": "t", "condition": "C4", "scores": {"contrast": 9.0}},
    ]
    rows = method_rows(_objs(), "ela", 0, None, boxes=boxes)
    (row,) = [r for r in rows if r["mode"] == "contrast"]
    # threshold 6.8 (val 80% TPR): test C2 boxes 8 flagged, 5 not; the val box is not used
    assert row["edited_background_flag"] == pytest.approx(0.5)
    assert row["c4_box_flag"] == 1.0
