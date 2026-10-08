import pytest

from inlayfake.ledger import ImageRecord, Instance, Origin, read_ledger, write_ledger


def _record(**kw):
    base = dict(
        image_id=ImageRecord.make_id(139, "C4", "G1", 7),
        source_id=139,
        split="test",
        condition="C4",
        generator="G1",
        path="images/139__C4__G1__s7.jpg",
        width=640,
        height=426,
        edit={"target_category": "chair", "seed": 7, "plan_box": [10, 20, 50, 80]},
        instances=[
            {
                "instance_id": "coco:1",
                "category": "tv",
                "bbox": [1, 2, 3, 4],
                "origin": "photographed",
            },
            {
                "instance_id": "ins:139__C4__G1__s7",
                "category": "chair",
                "bbox": [10, 20, 50, 80],
                "origin": "ai_inserted",
            },
        ],
    )
    base.update(kw)
    return ImageRecord(**base)


def test_roundtrip(tmp_path):
    rec = _record()
    assert write_ledger([rec], tmp_path / "l.jsonl") == 1
    (back,) = list(read_ledger(tmp_path / "l.jsonl"))
    assert back == rec
    assert back.instances[1].origin is Origin.AI_INSERTED


def test_make_id():
    assert ImageRecord.make_id(5, "C1", None, None) == "5__C1__none__s0"


@pytest.mark.parametrize(
    "kw",
    [
        {"split": "dev"},
        {"condition": "C9"},
        {
            "instances": [
                {
                    "instance_id": "coco:1",
                    "category": "tv",
                    "bbox": [0, 0, 1, 1],
                    "origin": "photographed",
                }
            ]
        },  # C4 without an inserted instance
    ],
)
def test_validation(kw):
    with pytest.raises(ValueError):
        _record(**kw)


def test_instance_validation():
    with pytest.raises(ValueError):
        Instance("a", "cup", [0, 0, -1, 1], Origin.PHOTOGRAPHED)
    with pytest.raises(ValueError):
        Instance("a", "cup", [0, 0, 1, 1], "fake")
