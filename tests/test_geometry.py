import numpy as np

from inlayfake.data.sampling import assign_split
from inlayfake.edit.compose import crop_window, feathered_alpha, local_edit, rect_mask
from inlayfake.edit.generators import FakeInpainter
from inlayfake.edit.placement import intersection, iou, place_free, place_occluding


def test_assign_split_is_deterministic_and_balanced():
    fr = {"train": 0.6, "val": 0.2, "test": 0.2}
    assert assign_split(42, fr) == assign_split(42, fr)
    counts = {s: 0 for s in fr}
    for i in range(20000):
        counts[assign_split(i, fr)] += 1
    assert abs(counts["train"] / 20000 - 0.6) < 0.02
    assert abs(counts["test"] / 20000 - 0.2) < 0.02


def test_box_ops():
    assert intersection([0, 0, 10, 10], [5, 5, 10, 10]) == 25
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0


def test_place_free_avoids_gt_with_margin():
    rng = np.random.default_rng(0)
    gt = [[0, 0, 200, 300], [250, 0, 50, 50]]
    for _ in range(50):
        b = place_free(400, 300, 40, 40, gt, 8, rng, 500)
        assert b is not None
        assert 0 <= b[0] and b[0] + b[2] <= 400 and 0 <= b[1] and b[1] + b[3] <= 300
        assert all(intersection(b, [g[0] - 8, g[1] - 8, g[2] + 16, g[3] + 16]) == 0 for g in gt)
    assert place_free(100, 100, 50, 50, [[0, 0, 100, 100]], 0, rng, 50) is None


def test_place_occluding_fraction():
    rng = np.random.default_rng(1)
    target = [100, 100, 100, 100]
    for _ in range(30):
        b = place_occluding(400, 400, 60, 60, target, [], (0.2, 0.5), rng, 1000)
        assert b is not None
        assert 0.2 <= intersection(b, target) / 10000 <= 0.5


def test_crop_window_square_and_inside():
    for box in ([0, 0, 30, 30], [600, 400, 39, 20], [300, 200, 100, 50]):
        x0, y0, x1, y1 = crop_window(box, 640, 427, 2.5)
        assert x1 - x0 == y1 - y0
        assert x0 >= 0 and y0 >= 0 and x1 <= 640 and y1 <= 427
        assert x0 <= box[0] and y0 <= box[1]  # box lies inside the window


def test_feathered_alpha_zero_outside_rect():
    rect = [10, 20, 50, 60]
    a = feathered_alpha(rect, 80, 80, 3)
    outside = np.ones_like(a, dtype=bool)
    outside[20:60, 10:50] = False
    assert a[outside].max() == 0
    assert a[40, 30] == 1.0


def test_local_edit_only_changes_inside_mask():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8)
    box = [150, 100, 60, 50]
    out, intended, win = local_edit(
        img, box, FakeInpainter(256), 256, "a photo of a cup", None, 7, 2.5, 6, 3
    )
    assert intended.sum() == (50 + 12) * (60 + 12)
    assert np.array_equal(out[~intended], img[~intended])
    assert not np.array_equal(out[intended], img[intended])
    assert rect_mask(box, 400, 300, 6) == [144, 94, 216, 156]
