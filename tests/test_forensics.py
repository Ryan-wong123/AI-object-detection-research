import io

import numpy as np
import pytest
from PIL import Image

from inlayfake.forensics.handcrafted import METHODS, box_mean, ela, jpeg_bag, noise_level
from inlayfake.forensics.instance_scores import bbox_mask, image_score, instance_score


def _jpeg(arr, q):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=q)
    return np.asarray(Image.open(buf).convert("RGB"))


def _smooth_noise(rng, h, w):
    base = rng.normal(128, 30, size=(h // 8, w // 8, 3))
    img = np.asarray(
        Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    )
    return np.clip(img + rng.normal(0, 4, size=img.shape), 0, 255).astype(np.uint8)


def _region_vs_rest(m, sl):
    inside = np.zeros(m.shape, bool)
    inside[sl] = True
    return m[inside].mean(), m[~inside].mean()


@pytest.fixture
def spliced():
    """JPEG(q=60) photo-like image with a 64x64 patch that never went through that JPEG."""
    rng = np.random.default_rng(0)
    img = _jpeg(_smooth_noise(rng, 256, 256), 60).copy()
    raw = _smooth_noise(np.random.default_rng(1), 256, 256)
    sl = (slice(96, 160), slice(96, 160))
    img[sl] = raw[sl]
    return img, sl


def test_box_mean_matches_naive():
    x = np.random.default_rng(0).random((20, 30)).astype(np.float32)
    k = 5
    naive = np.zeros_like(x)
    for i in range(20):
        for j in range(30):
            naive[i, j] = x[max(i - 2, 0) : i + 3, max(j - 2, 0) : j + 3].mean()
    assert np.allclose(box_mean(x, k), naive, atol=1e-5)


@pytest.mark.parametrize("name", list(METHODS))
def test_maps_shape_and_finite(name):
    img = _smooth_noise(np.random.default_rng(2), 150, 210)  # not a multiple of the block size
    m = METHODS[name](img)
    assert m.shape == (150, 210) and m.dtype == np.float32 and np.isfinite(m).all()


def test_ela_and_bag_flag_uncompressed_patch(spliced):
    img, sl = spliced
    for fn in (ela, jpeg_bag):
        inside, outside = _region_vs_rest(fn(img), sl)
        assert inside > outside, fn.__name__


def test_noise_level_flags_noisier_patch():
    rng = np.random.default_rng(3)
    img = _smooth_noise(rng, 256, 256).astype(float)
    img[96:160, 96:160] += rng.normal(0, 12, size=(64, 64, 3))
    m = noise_level(np.clip(img, 0, 255).astype(np.uint8))
    inside, outside = _region_vs_rest(m, (slice(96, 160), slice(96, 160)))
    assert inside > 2 * outside


def test_instance_and_image_scores():
    m = np.zeros((100, 100), np.float32)
    m[40:60, 40:60] = 1.0
    box = [40, 40, 20, 20]
    s = instance_score(m, bbox_mask(box, m.shape), box, 0.15)
    assert s["mean_in"] == 1.0 and s["mean_ring"] == 0.0 and s["contrast"] == 1.0
    assert image_score(m)["p99_minus_median"] == pytest.approx(1.0)


def test_jpeg_bag_bounded_on_flat_regions():
    """Flat/saturated regions used to divide ~0 by ~0 (Task 1 saw image scores in the thousands)."""
    rng = np.random.default_rng(4)
    img = _jpeg(_smooth_noise(rng, 256, 256), 80).copy()
    img[:128, :128] = 255  # saturated block
    img[128:, 128:] = 0  # black block
    m = jpeg_bag(_jpeg(img, 80))
    assert np.isfinite(m).all() and m.max() < 10
