import numpy as np

from inlayfake.detect import decode_grounding, gdino_prompt, token_category_map


def test_prompt_spans():
    text, spans = gdino_prompt(["cup", "potted plant", "dog"])
    assert text == "cup. potted plant. dog."
    assert [text[a:b] for a, b in spans] == ["cup", "potted plant", "dog"]


def test_token_map_and_decode():
    text, spans = gdino_prompt(["cup", "potted plant"])
    # BERT-like offsets: [CLS], cup, ., potted, plant, ., [SEP]
    offsets = [(0, 0), (0, 3), (3, 4), (5, 11), (12, 17), (17, 18), (0, 0)]
    tc = token_category_map(offsets, spans)
    assert tc.tolist() == [-1, 0, -1, 1, 1, -1, -1]

    probs = np.zeros((3, 7))
    probs[0, 1] = 0.8  # query 0 -> cup
    probs[1, 4] = 0.6  # query 1 -> "plant" token -> potted plant
    probs[2, 2] = 0.9  # query 2 only fires on the separator -> no category, below threshold
    boxes = np.array([[0.5, 0.5, 0.2, 0.4]] * 3)
    out = decode_grounding(probs, boxes, tc, 2, 0.3, 200, 100)
    assert [(k, round(s, 2)) for k, _, s in out] == [(0, 0.8), (1, 0.6)]
    assert out[0][1] == [80.0, 30.0, 40.0, 40.0]


def test_decode_handles_padded_logits():
    tc = np.array([-1, 0, -1])
    probs = np.zeros((1, 256))  # logits padded to max text length
    probs[0, 1] = 0.5
    out = decode_grounding(probs, np.array([[0.5, 0.5, 1, 1]]), tc, 1, 0.3, 10, 10)
    assert len(out) == 1
