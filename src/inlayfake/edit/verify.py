"""Insertion verification and annotation (ARCHITECTURE.md §3.5).

Pure functions (testable on CPU) plus thin wrappers around OWLv2 (open-vocabulary detector used as
an *independent* verifier: it is not one of the evaluated detectors) and SAM (box-prompted mask).

Boxes: [x, y, w, h] in pixels unless a name says xyxy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter

from inlayfake.edit.placement import Box, intersection, iou

# finder(crop_rgb, category, record) -> [(xyxy_in_crop, score)]; segmenter(crop_rgb, xyxy) -> mask
Finder = Callable[[np.ndarray, str, object], list[tuple[list[float], float]]]
Segmenter = Callable[[np.ndarray, list[float]], np.ndarray]


# ---------------------------------------------------------------- pure geometry / masks
def xyxy_to_xywh(b) -> list[float]:
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    ]


def changed_mask(edited: np.ndarray, reference: np.ndarray, threshold: int) -> np.ndarray:
    """Pixels whose max per-channel difference exceeds ``threshold``, after a 3×3 opening that
    removes isolated JPEG-noise pixels."""
    diff = np.abs(edited.astype(np.int16) - reference.astype(np.int16)).max(axis=2) > threshold
    m = Image.fromarray((diff * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return np.asarray(m) > 127


def inside_frac(box: Box, region: Box) -> float:
    """Share of ``box``'s area that lies inside ``region``."""
    a = box[2] * box[3]
    return intersection(box, region) / a if a > 0 else 0.0


def best_candidate(
    candidates: list[tuple[list[float], float]],
    plan_box: Box,
    min_iou: float,
    min_score: float,
    min_inside: float | None = None,
    originals: list[Box] | None = None,
    original_iou: float = 0.5,
) -> tuple[Box, float, float] | None:
    """Highest-scoring candidate (xywh, full-image coords) that matches the planned box.

    A candidate matches if IoU ≥ ``min_iou`` with the plan, or, when ``min_inside`` is set, if
    at least that share of its area lies inside the plan (an object smaller than the box).
    Candidates with IoU ≥ ``original_iou`` to one of ``originals`` (same-class objects already in
    the photo) are the real object, not the insertion, and never match.
    """
    best = None
    for box, score in candidates:
        if score < min_score:
            continue
        if any(iou(box, o) >= original_iou for o in originals or []):
            continue
        o = iou(box, plan_box)
        ok = o >= min_iou or (min_inside is not None and inside_frac(box, plan_box) >= min_inside)
        if ok and (best is None or score > best[1]):
            best = (box, score, o)
    return best


def crop_to_full(box_xyxy, crop_box) -> Box:
    x0, y0 = crop_box[0], crop_box[1]
    return xyxy_to_xywh([box_xyxy[0] + x0, box_xyxy[1] + y0, box_xyxy[2] + x0, box_xyxy[3] + y0])


def occlusion_update(orig_mask: np.ndarray, inserted: np.ndarray) -> tuple[np.ndarray, float]:
    """Visible part of an original object after an insertion, and its visible fraction."""
    area = orig_mask.sum()
    visible = orig_mask & ~inserted
    return visible, (float(visible.sum() / area) if area else 0.0)


# ---------------------------------------------------------------- model wrappers
class OwlV2Finder:
    """OWLv2 queried with "a photo of a {category}" on the (square) crop window."""

    def __init__(self, model_id: str, device: str = "cuda", threshold: float = 0.05) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).eval().to(device)
        self.device, self.thr, self.torch = device, threshold, torch

    def __call__(self, crop: np.ndarray, category: str, record=None):
        text = [[f"a photo of a {category}"]]
        inputs = self.processor(text=text, images=crop, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        # OWLv2 pads to a square; our crops are square, so the padded size is the crop size.
        side = max(crop.shape[:2])
        post = getattr(self.processor, "post_process_grounded_object_detection", None)
        if post is not None:  # transformers >= 4.51 / 5.x
            res = post(out, threshold=self.thr, target_sizes=[(side, side)])[0]
        else:  # older transformers
            res = self.processor.post_process_object_detection(
                out, threshold=self.thr, target_sizes=[(side, side)]
            )[0]
        return [
            ([float(v) for v in b], float(s))
            for b, s in zip(res["boxes"].cpu(), res["scores"].cpu(), strict=True)
        ]


class SamSegmenter:
    """SAM prompted with one box; returns a boolean mask the size of the crop."""

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        import torch
        from transformers import SamModel, SamProcessor

        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).eval().to(device)
        self.device, self.torch = device, torch

    def __call__(self, crop: np.ndarray, box_xyxy: list[float]) -> np.ndarray:
        inputs = self.processor(crop, input_boxes=[[box_xyxy]], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return masks[0][0][0].numpy().astype(bool)


class FakeFinder:
    """Test double: 'finds' the planned box in C4/C5 (unless told to fail) and nothing in C2."""

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()

    def __call__(self, crop, category, record):
        if record.condition.value == "C2" or record.image_id in self.fail_ids:
            return []
        x, y, w, h = record.edit.plan_box
        cx, cy = record.edit.crop_box[0], record.edit.crop_box[1]
        return [([x - cx, y - cy, x - cx + w, y - cy + h], 0.9)]


class FakeSegmenter:
    """Test double: an ellipse inside the box."""

    def __call__(self, crop, box_xyxy):
        from PIL import ImageDraw

        canvas = Image.new("L", (crop.shape[1], crop.shape[0]), 0)
        ImageDraw.Draw(canvas).ellipse([int(v) for v in box_xyxy], fill=255)
        return np.asarray(canvas) > 0


def build_verifier(vcfg: dict, device: str) -> tuple[Finder, Segmenter]:
    if vcfg["model_id"] == "fake":
        return FakeFinder(set(vcfg.get("fake_fail_ids", []))), FakeSegmenter()
    return (
        OwlV2Finder(vcfg["model_id"], device, vcfg.get("candidate_threshold", 0.05)),
        SamSegmenter(vcfg["sam_model_id"], device),
    )
