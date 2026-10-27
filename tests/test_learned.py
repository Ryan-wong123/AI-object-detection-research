"""Learned forensics (TruFor, UnivFD): geometry by hand, UnivFD's real code path on a tiny CLIP."""

import numpy as np
import pytest

from inlayfake.forensics.learned import (
    CROP_SIZE,
    FakeTruFor,
    clip_view,
    crop_box,
    object_crop,
    sigmoid,
)


def test_crop_box_square_and_inside():
    # box 20x10 at (10,10): side 1.3 * 20 = 26 around the centre (20, 15)
    assert crop_box([10, 10, 20, 10], 1.3, 100, 100) == (7, 2, 33, 28)
    # at the corner the square is shifted inside the image
    assert crop_box([90, 90, 10, 10], 1.3, 100, 100) == (87, 87, 100, 100)
    # tiny boxes still give >= 8 px; huge ones are clipped to the image
    x0, y0, x1, y1 = crop_box([50, 50, 2, 2], 1.3, 100, 100)
    assert x1 - x0 == 8 and y1 - y0 == 8
    assert crop_box([0, 0, 100, 60], 1.3, 100, 60) == (0, 0, 100, 60)


def test_views_are_224_rgb():
    img = np.random.default_rng(0).integers(0, 255, (60, 90, 3), dtype=np.uint8)
    assert object_crop(img, [5, 5, 20, 30], 1.3).shape == (CROP_SIZE, CROP_SIZE, 3)
    assert clip_view(img).shape == (CROP_SIZE, CROP_SIZE, 3)


def test_fake_trufor_map_and_score():
    img = np.full((8, 8, 3), 255, np.uint8)
    fmap, score = FakeTruFor()(img)
    assert fmap.shape == (8, 8) and score == pytest.approx(1.0)


def test_univfd_linear_probe_on_tiny_clip(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from inlayfake.forensics.learned import UnivFD

    cfg = transformers.CLIPConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": CROP_SIZE,
            "patch_size": 32,
        },
        projection_dim=8,
    )
    torch.manual_seed(0)
    transformers.CLIPModel(cfg).save_pretrained(tmp_path / "clip")
    w = torch.arange(8, dtype=torch.float32).view(1, 8) / 10
    torch.save({"weight": w, "bias": torch.tensor([0.5])}, tmp_path / "fc.pth")

    m = UnivFD(tmp_path / "fc.pth", str(tmp_path / "clip"), device="cpu", batch_size=2)
    views = [np.full((CROP_SIZE, CROP_SIZE, 3), v, np.uint8) for v in (0, 128, 255)]
    feats = m.features(views)
    assert feats.shape == (3, 8)  # three views across two batches
    logits = m.logits(feats)
    assert logits == pytest.approx(feats @ (np.arange(8) / 10) + 0.5, rel=1e-5)
    assert 0 < float(sigmoid(logits[0])) < 1
