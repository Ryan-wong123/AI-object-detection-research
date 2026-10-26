"""Stage 2 (H5): claim parsing, object matching and claim rates, by hand."""

import pytest

from inlayfake.eval.captions import claim_rows, object_statuses, summarize_claims
from inlayfake.ledger import ImageRecord
from inlayfake.stages.vlm import grounding_boxes
from inlayfake.vlm.claims import (
    boxes_prompt_lines,
    caption_text,
    normalize_origin,
    parse_json_claims,
    text_origin,
)


def test_origin_words():
    assert normalize_origin("ai_generated") == "ai"
    assert normalize_origin("AI") == "ai"
    assert normalize_origin("photographed") == "photographed"
    assert normalize_origin("uncertain") == "unknown"  # contains "ai" inside a word
    assert normalize_origin("") == "unknown"
    assert text_origin("a red chair that was digitally inserted") == "ai"
    assert text_origin("a photo of a red chair on the grass") == "none"


def test_parse_json_claims_scales_and_keeps_ids():
    text = (
        "A kitchen with two cups.\n```json\n"
        '[{"label": "cup", "bbox_2d": [10, 20, 30, 60], "origin": "ai_generated"},\n'
        ' {"id": 2, "origin": "photographed"},\n'
        ' {"label": "noise"}]\n```'
    )
    claims = parse_json_claims(text, sx=2.0, sy=0.5, W=1000, H=1000)
    assert claims == [
        {"label": "cup", "bbox": [20.0, 10.0, 40.0, 20.0], "box_id": None, "origin": "ai"},
        {"label": "", "bbox": None, "box_id": 2, "origin": "photographed"},
    ]  # an object with neither a box nor an id is not a claim
    assert caption_text(text) == "A kitchen with two cups."
    # truncated reply: the complete objects still count
    assert len(parse_json_claims('[{"id": 1, "origin": "unknown"}, {"id": 2, "or')) == 1


def test_boxes_prompt_lines_and_grounding_boxes():
    dets = [
        {"category": "cup", "bbox": [0, 0, 10, 10], "score": 0.6},
        {"category": "dog", "bbox": [50, 50, 20, 10], "score": 0.9},
        {"category": "cat", "bbox": [5, 5, 5, 5], "score": 0.3},
    ]
    boxes = grounding_boxes(dets, 0.5, 5)
    assert [(b["id"], b["category"]) for b in boxes] == [(1, "dog"), (2, "cup")]
    assert grounding_boxes(dets, 0.5, 1)[0]["category"] == "dog"
    assert boxes_prompt_lines(boxes, 2.0, 1.0) == "1: dog [100, 50, 140, 60]\n2: cup [0, 0, 20, 10]"


def _rec(cond="C4"):
    insts = [
        {
            "instance_id": "coco:1",
            "category": "cup",
            "bbox": [10, 10, 40, 40],
            "origin": "photographed",
        },
        {
            "instance_id": "coco:2",
            "category": "dog",
            "bbox": [200, 10, 40, 40],
            "origin": "mixed_modified",
        },
    ]
    if cond == "C4":
        insts.append(
            {
                "instance_id": "ins:1",
                "category": "cup",
                "bbox": [100, 100, 40, 40],
                "origin": "ai_inserted",
            }
        )
    return ImageRecord(f"7__{cond}", 7, "test", cond, "x.jpg", 300, 300, instances=insts)


def test_object_statuses_match_by_iou_and_skip_modified():
    claims = [
        {"label": "cup", "bbox": [102, 100, 40, 40], "box_id": None, "origin": "ai"},
        {"label": "dog", "bbox": [200, 10, 40, 40], "box_id": None, "origin": "ai"},
        {"label": "x", "bbox": None, "box_id": 3, "origin": "ai"},  # unboxed: ignored
    ]
    st = {s["instance_id"]: s["status"] for s in object_statuses(_rec(), claims, 0.5)}
    assert st == {"coco:1": "not_mentioned", "ins:1": "ai"}  # mixed/modified dog not scored


def test_claim_rates():
    r4, r1 = _rec("C4"), _rec("C1")
    rows = [
        {
            "model": "m",
            "setting": "plain",
            "image_id": r4.image_id,
            "source_id": 7,
            "condition": "C4",
            "claims": [
                {"bbox": [100, 100, 40, 40], "origin": "photographed"},
                {"bbox": [10, 10, 40, 40], "origin": "ai"},
            ],
        },
        {
            "model": "m",
            "setting": "plain",
            "image_id": r1.image_id,
            "source_id": 7,
            "condition": "C1",
            "claims": [{"bbox": [10, 10, 40, 40], "origin": "unknown"}],
        },
    ]
    objects = claim_rows(rows, {r4.image_id: r4, r1.image_id: r1}, 0.5)
    per_obj, per_img = summarize_claims(objects, rows, 0, None)
    got = {(r["condition"], r["object"]): r for r in per_obj}
    assert got[("C4", "ai_inserted")]["photographed"] == 1.0  # a false-real claim
    assert got[("C4", "photographed")]["ai"] == 1.0  # a false-AI claim
    assert got[("C1", "photographed")]["unknown"] == 1.0
    assert {(r["condition"], r["any_ai"]) for r in per_img} == {("C4", 1.0), ("C1", 0.0)}
    assert (
        pytest.approx(
            sum(
                got[("C4", "ai_inserted")][s]
                for s in ("ai", "photographed", "unknown", "none", "not_mentioned")
            )
        )
        == 1.0
    )
