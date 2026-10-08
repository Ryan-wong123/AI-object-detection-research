"""Object detector wrappers with one interface (ARCHITECTURE.md §3.6).

``build_detector(name, cfg, categories)`` returns a callable
``(rgb uint8 HxWx3) -> list[Detection]`` whose categories are COCO category names.
Predictions with names outside ``categories`` are dropped. Heavy libraries are imported lazily.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Detection:
    category: str
    bbox: list[float]  # [x, y, w, h]
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _xyxy_to_xywh(b) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in b)
    return [x0, y0, x1 - x0, y1 - y0]


class FakeDetector:
    """Test double: returns the boxes given to ``set_truth`` with a little jitter."""

    def __init__(self, seed: int = 0, **_: object) -> None:
        self.rng = np.random.default_rng(seed)
        self.truth: list[Detection] = []

    def set_truth(self, dets: list[Detection]) -> None:
        self.truth = dets

    def __call__(self, image: np.ndarray) -> list[Detection]:
        out = []
        for d in self.truth:
            j = self.rng.normal(0, 1.0, size=4)
            out.append(Detection(d.category, [float(v) for v in np.add(d.bbox, j)], 0.9))
        return out


class TorchvisionFasterRCNN:
    def __init__(self, device: str = "cuda", score_threshold: float = 0.001, **_: object) -> None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.names = weights.meta["categories"]
        self.model = (
            fasterrcnn_resnet50_fpn_v2(
                weights=weights, box_score_thresh=score_threshold, box_detections_per_img=100
            )
            .eval()
            .to(device)
        )
        self.device = device
        self.torch = torch

    def __call__(self, image: np.ndarray) -> list[Detection]:
        t = self.torch.from_numpy(image).permute(2, 0, 1).float().div(255).to(self.device)
        with self.torch.inference_mode():
            out = self.model([t])[0]
        return [
            Detection(self.names[int(lbl)], _xyxy_to_xywh(b), float(s))
            for b, lbl, s in zip(
                *(out[k].cpu() for k in ("boxes", "labels", "scores")), strict=True
            )
        ]


class UltralyticsYOLO:
    def __init__(
        self,
        weights: str = "yolo11m.pt",
        device: str = "cuda",
        score_threshold: float = 0.001,
        **_: object,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = 0 if device == "cuda" else device
        self.conf = score_threshold

    def __call__(self, image: np.ndarray) -> list[Detection]:
        # ultralytics expects BGR numpy arrays
        res = self.model.predict(
            image[..., ::-1], conf=self.conf, device=self.device, max_det=100, verbose=False
        )[0]
        names = res.names
        b = res.boxes
        return [
            Detection(names[int(c)], _xyxy_to_xywh(xy), float(s))
            for xy, c, s in zip(*(t.cpu().numpy() for t in (b.xyxy, b.cls, b.conf)), strict=True)
        ]


class HFRTDETR:
    def __init__(
        self,
        model_id: str = "PekingU/rtdetr_r50vd",
        device: str = "cuda",
        score_threshold: float = 0.001,
        **_: object,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, RTDetrForObjectDetection

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).eval().to(device)
        self.device, self.thr, self.torch = device, score_threshold, torch

    def __call__(self, image: np.ndarray) -> list[Detection]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            out, target_sizes=[image.shape[:2]], threshold=self.thr
        )[0]
        id2label = self.model.config.id2label
        dets = [
            Detection(id2label[int(lbl)], _xyxy_to_xywh(b), float(s))
            for b, lbl, s in zip(
                *(res[k].cpu() for k in ("boxes", "labels", "scores")), strict=True
            )
        ]
        return sorted(dets, key=lambda d: -d.score)[:100]


def gdino_prompt(names: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Prompt "a. b c. d." and the character span of each name in it."""
    spans, parts, pos = [], [], 0
    for n in names:
        spans.append((pos, pos + len(n)))
        parts.append(n)
        pos += len(n) + 2  # ". "
    return ". ".join(parts) + ".", spans


def token_category_map(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> np.ndarray:
    """Category index of each text token (-1 = separator or special token)."""
    out = np.full(len(offsets), -1, dtype=int)
    for t, (a, b) in enumerate(offsets):
        if b <= a:  # special tokens ([CLS], [SEP]) have empty offsets
            continue
        for k, (s0, s1) in enumerate(spans):
            if a >= s0 and b <= s1:
                out[t] = k
                break
    return out


def decode_grounding(
    probs: np.ndarray,  # (queries, text_tokens) sigmoid scores
    boxes_cxcywh: np.ndarray,  # (queries, 4) normalised
    token_cat: np.ndarray,  # (text_tokens,) category index per token, -1 = none
    n_cats: int,
    threshold: float,
    width: int,
    height: int,
) -> list[tuple[int, list[float], float]]:
    """Per query: score of each category = max over its tokens; keep the best category if its score
    is >= threshold. Returns (category index, [x, y, w, h] in pixels, score)."""
    n_tok = min(probs.shape[1], len(token_cat))
    cat_scores = np.full((probs.shape[0], n_cats), -np.inf)
    for k in range(n_cats):
        toks = np.nonzero(token_cat[:n_tok] == k)[0]
        if len(toks):
            cat_scores[:, k] = probs[:, toks].max(axis=1)
    best = cat_scores.argmax(axis=1)
    score = cat_scores[np.arange(len(best)), best]
    out = []
    for q in np.nonzero(score >= threshold)[0]:
        cx, cy, w, h = boxes_cxcywh[q]
        box = [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]
        out.append((int(best[q]), [float(v) for v in box], float(score[q])))
    return out


class HFGroundingDINO:
    """Open-vocabulary detector prompted with COCO class names in chunks.

    Grounding DINO's text encoder has a 256-token limit, so the 80 names are split into chunks and
    the model runs once per chunk. Labels are decoded directly from the token-level logits (see
    ``decode_grounding``) instead of the processor's post-processing, whose phrase output changed
    across transformers versions and could return more labels than boxes.
    """

    def __init__(
        self,
        categories: list[str],
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        score_threshold: float = 0.05,
        chunk: int = 20,
        **_: object,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval().to(device)
        self.device, self.torch = device, torch
        self.thr = max(score_threshold, 0.05)  # very low thresholds give only noise for GDINO
        self.chunks = []
        for i in range(0, len(categories), chunk):
            names = categories[i : i + chunk]
            text, spans = gdino_prompt(names)
            enc = self.processor.tokenizer(text, return_offsets_mapping=True)
            self.chunks.append((names, text, token_category_map(enc["offset_mapping"], spans)))

    def __call__(self, image: np.ndarray) -> list[Detection]:
        H, W = image.shape[:2]
        dets: list[Detection] = []
        for names, text, token_cat in self.chunks:
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                out = self.model(**inputs)
            probs = out.logits[0].sigmoid().float().cpu().numpy()
            boxes = out.pred_boxes[0].float().cpu().numpy()
            for k, box, score in decode_grounding(
                probs, boxes, token_cat, len(names), self.thr, W, H
            ):
                dets.append(Detection(names[k], box, score))
        return sorted(dets, key=lambda d: -d.score)[:100]


def build_detector(name: str, cfg: dict, categories: list[str], device: str):
    dcfg = cfg["detectors"][name]
    impl = dcfg["impl"]
    thr = cfg["detectors"].get("score_threshold", 0.001)
    if impl == "fake":
        return FakeDetector()
    if impl == "torchvision":
        return TorchvisionFasterRCNN(device=device, score_threshold=thr)
    if impl == "ultralytics":
        return UltralyticsYOLO(weights=dcfg["weights"], device=device, score_threshold=thr)
    if impl == "transformers" and "grounding" in dcfg["model_id"].lower():
        return HFGroundingDINO(
            categories, model_id=dcfg["model_id"], device=device, score_threshold=thr
        )
    if impl == "transformers":
        return HFRTDETR(model_id=dcfg["model_id"], device=device, score_threshold=thr)
    raise ValueError(f"unknown detector impl {impl!r}")
