"""Local inpainting and compositing (ARCHITECTURE.md §3.4).

The generator sees a square window around the planned box, resized to its native resolution.
Only pixels inside the (dilated) intended mask are composited back, with a linear feather that
reaches zero exactly at the mask border, so every pixel outside the intended mask is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from inlayfake.edit.placement import Box

# inpaint_fn(image_pil, mask_pil, prompt, negative_prompt, seed) -> PIL image (same size)
InpaintFn = Callable[[Image.Image, Image.Image, str, str | None, int], Image.Image]


def crop_window(box: Box, width: int, height: int, scale: float, min_side: int = 256) -> list[int]:
    """Square window [x0, y0, x1, y1] of side ``scale * max(w, h)`` around ``box``, kept inside
    the image (shifted, not clipped, so it stays square)."""
    side = int(round(max(box[2], box[3]) * scale))
    side = min(max(side, min_side), width, height)
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    x0 = int(round(min(max(cx - side / 2, 0), width - side)))
    y0 = int(round(min(max(cy - side / 2, 0), height - side)))
    return [x0, y0, x0 + side, y0 + side]


def rect_mask(box: Box, width: int, height: int, dilate: float) -> list[int]:
    """Integer rectangle [x0, y0, x1, y1] of the dilated box, clipped to the image."""
    x0 = int(np.floor(max(box[0] - dilate, 0)))
    y0 = int(np.floor(max(box[1] - dilate, 0)))
    x1 = int(np.ceil(min(box[0] + box[2] + dilate, width)))
    y1 = int(np.ceil(min(box[1] + box[3] + dilate, height)))
    return [x0, y0, x1, y1]


def feathered_alpha(rect: list[int], width: int, height: int, feather: float) -> np.ndarray:
    """Float HxW alpha: 0 outside ``rect``, ramping linearly to 1 within ``feather`` px inside."""
    x0, y0, x1, y1 = rect
    xs = np.arange(width) + 0.5
    ys = np.arange(height) + 0.5
    dx = np.minimum(xs - x0, x1 - xs)
    dy = np.minimum(ys - y0, y1 - ys)
    d = np.minimum(dy[:, None], dx[None, :])
    if feather <= 0:
        return (d > 0).astype(np.float32)
    return np.clip(d / feather, 0.0, 1.0).astype(np.float32)


def composite(base: np.ndarray, top: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    out = base.astype(np.float32) * (1 - a) + top.astype(np.float32) * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def local_edit(
    image: np.ndarray,
    box: Box,
    inpaint: InpaintFn,
    resolution: int,
    prompt: str,
    negative_prompt: str | None,
    seed: int,
    window_scale: float,
    dilate: float,
    feather: float,
    composite_mode: str = "mask",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Inpaint ``box`` and return (edited image, intended mask bool HxW, crop window).

    ``composite_mode="window"`` is the ablation that keeps the generator output for the whole
    window (including VAE re-encoding outside the mask).
    """
    H, W = image.shape[:2]
    win = crop_window(box, W, H, window_scale)
    rect = rect_mask(box, W, H, dilate)
    intended = np.zeros((H, W), dtype=bool)
    intended[rect[1] : rect[3], rect[0] : rect[2]] = True

    x0, y0, x1, y1 = win
    crop = image[y0:y1, x0:x1]
    side = x1 - x0
    crop_pil = Image.fromarray(crop).resize((resolution, resolution), Image.LANCZOS)
    mask_crop = (intended[y0:y1, x0:x1] * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_crop).resize((resolution, resolution), Image.NEAREST)

    gen = inpaint(crop_pil, mask_pil, prompt, negative_prompt, seed)
    gen_arr = np.asarray(gen.convert("RGB").resize((side, side), Image.LANCZOS), dtype=np.uint8)

    out = image.copy()
    if composite_mode == "window":
        out[y0:y1, x0:x1] = gen_arr
        intended[:] = False
        intended[y0:y1, x0:x1] = True
        return out, intended, win
    alpha = feathered_alpha(rect, W, H, feather)[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = composite(crop, gen_arr, alpha)
    return out, intended, win
