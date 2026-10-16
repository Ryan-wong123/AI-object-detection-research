"""Hand-crafted ("traditional") forensic maps (ARCHITECTURE.md §3.7, F1–F5).

Each function takes an RGB uint8 image HxWx3 and returns a float32 map HxW where *higher means
more suspicious*. None of them has a natural "AI / not AI" threshold. Thresholds are chosen later on
the validation split (Task 3), and Task 1 records how the maps behave on real photographs.

Only numpy and PIL are used, so the methods run identically on Windows and Linux.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def to_gray(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def box_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Mean over a k×k window (same size output, edge windows truncated), via integral image."""
    H, W = x.shape
    r = k // 2
    ii = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys, xs = np.arange(H), np.arange(W)
    y0, y1 = np.clip(ys - r, 0, H), np.clip(ys + r + 1, 0, H)
    x0, x1 = np.clip(xs - r, 0, W), np.clip(xs + r + 1, 0, W)
    s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return (s / area).astype(np.float32)


def conv2d_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    p = np.pad(x, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="reflect")
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] != 0:
                out += kernel[i, j] * p[i : i + x.shape[0], j : j + x.shape[1]]
    return out


def _block_upsample(blocks: np.ndarray, block: int, H: int, W: int) -> np.ndarray:
    return np.kron(blocks, np.ones((block, block), dtype=np.float32))[:H, :W]


def _pad_to_multiple(x: np.ndarray, m: int) -> np.ndarray:
    H, W = x.shape
    return np.pad(x, ((0, (-H) % m), (0, (-W) % m)), mode="reflect")


# F1 -------------------------------------------------------------------------------------------
def ela(img: np.ndarray, quality: int = 90, smooth: int = 7) -> np.ndarray:
    """Error Level Analysis [Krawetz 2007]: |image − JPEG(image, q)|, channel mean, smoothed."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    resaved = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    diff = np.abs(img.astype(np.float32) - resaved).mean(axis=2)
    return box_mean(diff, smooth)


# F2 -------------------------------------------------------------------------------------------
SRM_KERNELS = [
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) / 2,
    np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) / 2,
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4,
]


def srm_residual(img: np.ndarray, window: int = 16) -> np.ndarray:
    """Local variance of SRM high-pass residuals [Fridrich & Kodovsky 2012], log-scaled."""
    g = to_gray(img)
    out = np.zeros_like(g)
    for k in SRM_KERNELS:
        r = conv2d_same(g, k)
        var = box_mean(r * r, window) - box_mean(r, window) ** 2
        out += np.maximum(var, 0)
    return np.log1p(out / len(SRM_KERNELS)).astype(np.float32)


# F3 -------------------------------------------------------------------------------------------
def noise_level(img: np.ndarray, block: int = 32) -> np.ndarray:
    """Blockwise noise σ from the Haar HH sub-band, σ = MAD/0.6745 [Donoho; Mahdian & Saic 2009].

    Returned as |σ_block − median σ|, so blocks whose noise differs from the rest are high.
    """
    g = _pad_to_multiple(to_gray(img), block)
    hh = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    b = block // 2
    Hb, Wb = hh.shape[0] // b, hh.shape[1] // b
    tiles = hh[: Hb * b, : Wb * b].reshape(Hb, b, Wb, b).transpose(0, 2, 1, 3).reshape(Hb, Wb, -1)
    sigma = np.median(np.abs(tiles), axis=2) / 0.6745
    dev = np.abs(sigma - np.median(sigma)).astype(np.float32)
    return _block_upsample(dev, block, img.shape[0], img.shape[1])


# F4 -------------------------------------------------------------------------------------------
def fft_highfreq(img: np.ndarray, block: int = 32, cutoff: float = 0.5) -> np.ndarray:
    """Per-patch share of spectral energy above ``cutoff``×Nyquist [cf. Durall et al. 2020].

    Returned as |ratio − median ratio| so that patches with unusual spectra stand out.
    """
    g = _pad_to_multiple(to_gray(img), block)
    Hb, Wb = g.shape[0] // block, g.shape[1] // block
    tiles = g.reshape(Hb, block, Wb, block).transpose(0, 2, 1, 3)
    tiles = tiles - tiles.mean(axis=(2, 3), keepdims=True)
    win = np.outer(np.hanning(block), np.hanning(block)).astype(np.float32)
    power = np.abs(np.fft.fft2(tiles * win)) ** 2
    f = np.fft.fftfreq(block)
    radius = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2) / 0.5  # 1.0 = Nyquist along an axis
    high = power[..., radius > cutoff].sum(-1)
    total = power[..., radius > 0].sum(-1) + 1e-8
    ratio = high / total
    dev = np.abs(ratio - np.median(ratio)).astype(np.float32)
    return _block_upsample(dev, block, img.shape[0], img.shape[1])


# F5 -------------------------------------------------------------------------------------------
def jpeg_bag(img: np.ndarray, block: int = 32) -> np.ndarray:
    """JPEG blocking-artifact-grid strength per region [cf. Li et al. 2009].

    For each region, the ratio of mean |second difference| on the 8×8 grid boundary to that off the
    boundary. Regions whose grid strength departs from the image median (e.g. content that lacks
    the original photo's compression history) score high.
    """
    g = to_gray(img)
    dh = np.zeros_like(g)
    dv = np.zeros_like(g)
    dh[:, 1:-1] = np.abs(2 * g[:, 1:-1] - g[:, :-2] - g[:, 2:])
    dv[1:-1, :] = np.abs(2 * g[1:-1, :] - g[:-2, :] - g[2:, :])
    on_h = (np.arange(g.shape[1]) % 8 == 7)[None, :]
    on_v = (np.arange(g.shape[0]) % 8 == 7)[:, None]
    on = (dh * on_h) + (dv * on_v)
    off = (dh * ~on_h) + (dv * ~on_v)
    on_p, off_p = _pad_to_multiple(on, block), _pad_to_multiple(off, block)
    Hb, Wb = on_p.shape[0] // block, on_p.shape[1] // block

    def blocks(x):
        return x.reshape(Hb, block, Wb, block).mean(axis=(1, 3))

    # Boundary positions are 1/8 of each axis; weight so a grid-free region gives ≈ 0 after the log.
    # The +1 (grey levels) keeps flat or saturated regions, where both terms are ~0, from exploding:
    # with a plain ratio, Task 1 saw image scores in the thousands from a few such regions.
    ratio = np.log((blocks(on_p) * 7 + 1.0) / (blocks(off_p) + 1.0))
    dev = np.abs(ratio - np.median(ratio)).astype(np.float32)
    return _block_upsample(dev, block, img.shape[0], img.shape[1])


METHODS = {
    "ela": ela,
    "srm_residual": srm_residual,
    "noise_level": noise_level,
    "fft_highfreq": fft_highfreq,
    "jpeg_bag": jpeg_bag,
}
