"""Learned forensic methods (ARCHITECTURE.md §3.7, L1 TruFor and L2 UnivFD).

L1 TruFor [Guillaro et al. 2023] runs from the authors' own code and weights, which are not
redistributed (non-commercial licence): the author clones github.com/grip-unina/TruFor and
downloads TruFor_weights.zip; ``forensics.models.trufor.src`` / ``weights`` point to them. It
gives a pixel map (probability of manipulation) and an image score. Inference follows the
authors' ``test_docker/src/trufor_test.py``: RGB / 256, ``softmax(pred)[1]`` as the map,
``sigmoid(det)`` as the score.

L2 UnivFD [Ojha et al. 2023]: frozen CLIP ViT-L/14 image features (``encode_image``, 768-d) and
the authors' linear probe (``pretrained_weights/fc_weights.pth`` from
github.com/WisconsinAIVision/UniversalFakeDetect, MIT licence); sigmoid of the logit = P(fake).
CLIP comes from Hugging Face (``openai/clip-vit-large-patch14``): ``get_image_features`` is the
same projected embedding as OpenAI's ``encode_image``. The image level uses CLIP's standard
preprocessing (resize + centre crop 224); object crops are the box expanded
``forensics.crop_expand`` (1.3×), squared and resized to 224.

Heavy imports (torch, transformers, the TruFor code) happen inside the classes; the fakes run
on CPU for the tests (``model_id: fake``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CROP_SIZE = 224


def crop_box(
    bbox: list[float], expand: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Square crop around a box: side = expand × the longer box side (≥ 8 px), centred on the
    box and shifted to stay inside the image where possible. Returns (x0, y0, x1, y1)."""
    x, y, w, h = bbox
    side = max(8.0, expand * max(w, h))
    side = min(side, float(max(width, height)))
    cx, cy = x + w / 2, y + h / 2
    x0 = min(max(cx - side / 2, 0.0), max(width - side, 0.0))
    y0 = min(max(cy - side / 2, 0.0), max(height - side, 0.0))
    x1, y1 = min(x0 + side, width), min(y0 + side, height)
    return (
        int(round(x0)),
        int(round(y0)),
        max(int(round(x1)), int(round(x0)) + 1),
        max(int(round(y1)), int(round(y0)) + 1),
    )


def object_crop(img: np.ndarray, bbox: list[float], expand: float) -> np.ndarray:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = crop_box(bbox, expand, W, H)
    crop = Image.fromarray(img[y0:y1, x0:x1])
    return np.asarray(crop.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC))


def clip_view(img: np.ndarray) -> np.ndarray:
    """CLIP's own preprocessing geometry: shorter side to 224 (bicubic), centre crop 224."""
    H, W = img.shape[:2]
    s = CROP_SIZE / min(H, W)
    nw, nh = max(CROP_SIZE, round(W * s)), max(CROP_SIZE, round(H * s))
    im = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BICUBIC))
    y0, x0 = (nh - CROP_SIZE) // 2, (nw - CROP_SIZE) // 2
    return im[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- L1 TruFor


class TruFor:
    def __init__(self, src: str | Path, weights: str | Path, device: str = "cuda") -> None:
        import torch

        src, weights = Path(src).resolve(), Path(weights).resolve()
        for p, what in ((src / "trufor.yaml", "TruFor code"), (weights, "TruFor weights")):
            if not p.exists():
                raise SystemExit(f"{what} not found: {p} (see forensics.models.trufor)")
        spec = importlib.util.spec_from_file_location("trufor_config", src / "config.py")
        tcfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tcfg)
        cfg = tcfg._C.clone()
        cfg.defrost()
        cfg.merge_from_file(str(src / "trufor.yaml"))
        cfg.freeze()
        if str(src) not in sys.path:  # the model imports `models.DnCNN` at construction time
            sys.path.insert(0, str(src))
        from models.cmx.builder_np_conf import myEncoderDecoder

        model = myEncoderDecoder(cfg=cfg)
        try:
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        except Exception:  # older checkpoints hold non-tensor objects; official file (MD5 in docs)
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.to(device).eval()
        self.device = device
        self.torch = torch

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """(map H×W in [0, 1], image score in [0, 1]) for an RGB uint8 image."""
        torch = self.torch
        x = torch.tensor(img.transpose(2, 0, 1)[None], dtype=torch.float32) / 256.0
        with torch.no_grad():
            pred, _conf, det, _npp = self.model(x.to(self.device))
        fmap = torch.softmax(pred[0], dim=0)[1].float().cpu().numpy()
        score = float(torch.sigmoid(det).item()) if det is not None else float("nan")
        return fmap.astype(np.float32), score


class FakeTruFor:
    """CPU stand-in: a smooth brightness-based map and its mean as the score."""

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        fmap = img.astype(np.float32).mean(axis=2) / 255.0
        return fmap, float(fmap.mean())


# ---------------------------------------------------------------- L2 UnivFD


class UnivFD:
    def __init__(
        self,
        fc_weights: str | Path,
        clip_model_id: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import CLIPModel

        fc_weights = Path(fc_weights)
        if not fc_weights.exists():
            raise SystemExit(f"UnivFD weights not found: {fc_weights} (forensics.models.univfd)")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.clip = CLIPModel.from_pretrained(clip_model_id, torch_dtype=dtype).to(device).eval()
        sd = torch.load(fc_weights, map_location="cpu", weights_only=True)
        self.w = sd["weight"].float().numpy().reshape(-1)  # (1, 768) linear probe
        self.b = float(sd["bias"].float().numpy().reshape(-1)[0])
        self.device, self.dtype, self.batch_size, self.torch = device, dtype, batch_size, torch
        self.mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)

    def features(self, views: list[np.ndarray]) -> np.ndarray:
        """768-d CLIP image embeddings (float32) of 224×224 RGB uint8 views."""
        torch = self.torch
        out = []
        for i in range(0, len(views), self.batch_size):
            x = torch.tensor(np.stack(views[i : i + self.batch_size]).transpose(0, 3, 1, 2))
            x = ((x.float() / 255.0) - self.mean) / self.std
            with torch.no_grad():
                f = self.clip.get_image_features(pixel_values=x.to(self.device, self.dtype))
            if not isinstance(f, torch.Tensor):  # newer transformers return an output object
                f = f.pooler_output
            out.append(f.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.w.size), np.float32)

    def logits(self, feats: np.ndarray) -> np.ndarray:
        return feats @ self.w + self.b


class FakeUnivFD:
    """CPU stand-in: 8-d colour statistics as features, their mean as the logit."""

    def features(self, views: list[np.ndarray]) -> np.ndarray:
        return np.array(
            [
                np.concatenate([v.reshape(-1, 3).mean(0), v.reshape(-1, 3).std(0), [0, 0]]) / 255.0
                for v in views
            ],
            dtype=np.float32,
        ).reshape(len(views), 8)

    def logits(self, feats: np.ndarray) -> np.ndarray:
        return feats.mean(axis=1) * 4 - 1


def build_learned(name: str, cfg: dict):
    mcfg = cfg["forensics"]["models"][name]
    device = cfg["hardware"]["device"]
    fake = mcfg.get("model_id") == "fake"
    if name == "trufor":
        return FakeTruFor() if fake else TruFor(mcfg["src"], mcfg["weights"], device)
    if name == "univfd":
        if fake:
            return FakeUnivFD()
        return UnivFD(mcfg["fc_weights"], mcfg["clip_model_id"], device, mcfg.get("batch_size", 32))
    raise ValueError(f"unknown learned forensic method {name!r} (implemented: trufor, univfd)")
