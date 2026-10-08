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


class HFGroundingDINO:
    """Open-vocabulary detector prompted with COCO class names in chunks.

    Grounding DINO's text encoder has a 256-token limit, so the 80 names are split into chunks and
    the model runs once per chunk. A predicted phrase is mapped to the category name it equals
    (or contains).
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
        self.chunks = [categories[i : i + chunk] for i in range(0, len(categories), chunk)]

    @staticmethod
    def _match(phrase: str, names: list[str]) -> str | None:
        phrase = phrase.strip().lower()
        if phrase in names:
            return phrase
        hits = [n for n in names if n in phrase]
        return max(hits, key=len) if hits else None

    def __call__(self, image: np.ndarray) -> list[Detection]:
        dets: list[Detection] = []
        for names in self.chunks:
            text = ". ".join(names) + "."
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                out = self.model(**inputs)
            kwargs = {"target_sizes": [image.shape[:2]], "text_threshold": self.thr}
            try:  # transformers >= 4.51 renamed box_threshold -> threshold
                res = self.processor.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=self.thr, **kwargs
                )[0]
            except TypeError:
                res = self.processor.post_process_grounded_object_detection(
                    out, inputs.input_ids, box_threshold=self.thr, **kwargs
                )[0]
            labels = res.get("text_labels", res.get("labels"))
            for b, phrase, s in zip(res["boxes"].cpu(), labels, res["scores"].cpu(), strict=True):
                name = self._match(str(phrase), names)
                if name is not None:
                    dets.append(Detection(name, _xyxy_to_xywh(b), float(s)))
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
