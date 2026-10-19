"""Scene models for natural placement (``placement.natural``): a VLM that points at spots and
answers yes/no questions, and a monocular depth model.

Backends (``placement.natural.vlm.backend``):
  molmo     allenai/Molmo-7B-D-0924 (Apache 2.0), trained to answer "Point to ..." with points;
            its own model code (trust_remote_code), loaded in 4-bit to fit 8 GB.
  internvl  OpenGVLab/InternVL3_5-4B-HF (Apache 2.0), built into transformers; asked for points
            as [x, y] on a 0-1000 grid. Fallback if Molmo does not load.
  fake      deterministic stand-in for CPU tests.
Neither is one of the VLMs evaluated in week 3 (Florence-2, Qwen2.5-VL-3B), so the placement
model cannot bias that test toward its own taste.

Depth (``placement.natural.depth``): Depth Anything V2 Small (built into transformers); returns
relative *nearness* (inverse depth: larger = closer), resized to the image.

Heavy imports happen inside the builders (tests and the CLI need neither model).
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------- answer parsing
_MOLMO_POINT = re.compile(r'x\d*="([\d.]+)"\s+y\d*="([\d.]+)"')
_GRID_POINT = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")


def parse_molmo_points(text: str, W: int, H: int) -> list[tuple[float, float]]:
    """Molmo writes <point x="61.5" y="40.6" ...> or <points x1=".." y1=".." x2=..>, in percent
    of the image width/height."""
    return [(float(x) / 100 * W, float(y) / 100 * H) for x, y in _MOLMO_POINT.findall(text)]


def parse_grid_points(text: str, W: int, H: int) -> list[tuple[float, float]]:
    """[x, y] pairs on a 0-1000 grid (InternVL convention)."""
    return [(float(x) / 1000 * W, float(y) / 1000 * H) for x, y in _GRID_POINT.findall(text)]


def parse_yes_no(text: str) -> str:
    """'yes', 'no' or 'unclear' from the start of an answer."""
    t = text.strip().lower().lstrip("\"'*`- ")
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "unclear"


# ---------------------------------------------------------------------------------------- VLMs
class FakeVLM:
    """CPU stand-in: points along the lower part of the image (where floors and tables usually
    are), answers 'yes' unless the record id is in ``fail_ids``."""

    name = "fake"

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()

    def point(self, img: np.ndarray, prompt: str) -> tuple[list[tuple[float, float]], str]:
        H, W = img.shape[:2]
        pts = [(0.5 * W, 0.97 * H), (0.2 * W, 0.95 * H), (0.8 * W, 0.95 * H)]
        return pts, "fake points"

    def ask(self, img: np.ndarray, question: str, record_id: str = "") -> str:
        return "no" if record_id in self.fail_ids else "yes"


class MolmoVLM:
    name = "molmo"

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = True) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        kw: dict = {"trust_remote_code": True, "device_map": "auto"}
        if load_in_4bit and device.startswith("cuda"):
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            kw["torch_dtype"] = torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()

    def _generate(self, img: np.ndarray, text: str, max_new_tokens: int) -> str:
        import torch
        from transformers import GenerationConfig

        inputs = self.processor.process(images=[Image.fromarray(img)], text=text)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        if "images" in inputs:
            inputs["images"] = inputs["images"].to(torch.float16)
        with torch.inference_mode():
            out = self.model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>", do_sample=False
                ),
                tokenizer=self.processor.tokenizer,
            )
        new = out[0, inputs["input_ids"].size(1) :]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True)

    def point(self, img: np.ndarray, prompt: str) -> tuple[list[tuple[float, float]], str]:
        text = self._generate(img, prompt, 200)
        H, W = img.shape[:2]
        return parse_molmo_points(text, W, H), text

    def ask(self, img: np.ndarray, question: str, record_id: str = "") -> str:
        return self._generate(img, question, 8)


class InternVLVLM:
    name = "internvl"
    GRID = " Answer only with a list of points as [x, y], with x and y between 0 and 1000."

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = True) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        kw: dict = {"device_map": "auto"}
        if load_in_4bit and device.startswith("cuda"):
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            kw["torch_dtype"] = torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        self.model.eval()

    def _generate(self, img: np.ndarray, text: str, max_new_tokens: int) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(img)},
                    {"type": "text", "text": text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def point(self, img: np.ndarray, prompt: str) -> tuple[list[tuple[float, float]], str]:
        text = self._generate(img, prompt + self.GRID, 200)
        H, W = img.shape[:2]
        return parse_grid_points(text, W, H), text

    def ask(self, img: np.ndarray, question: str, record_id: str = "") -> str:
        return self._generate(img, question + " Answer yes or no.", 8)


def build_vlm(vcfg: dict, device: str):
    backend = vcfg.get("backend", "molmo")
    if backend == "fake":
        return FakeVLM(set(vcfg.get("fake_fail_ids", [])))
    if backend == "molmo":
        return MolmoVLM(vcfg["model_id"], device, vcfg.get("load_in_4bit", True))
    if backend == "internvl":
        return InternVLVLM(vcfg["model_id"], device, vcfg.get("load_in_4bit", True))
    raise ValueError(f"unknown VLM backend {backend!r}")


# --------------------------------------------------------------------------------------- depth
class FakeDepth:
    """CPU stand-in: nearness grows linearly towards the bottom of the image (ground plane)."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        return np.tile(np.linspace(0.2, 1.0, H, dtype=np.float32)[:, None], (1, W))


class DepthAnything:
    def __init__(self, model_id: str, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import torch

        inputs = self.processor(images=Image.fromarray(img), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            pred = self.model(**inputs).predicted_depth  # relative inverse depth (nearness)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        )[0, 0]
        out = pred.float().cpu().numpy()
        return out - min(float(out.min()), 0.0) + 1e-3  # positive, larger = closer


def build_depth(dcfg: dict, device: str):
    if dcfg.get("model_id") == "fake":
        return FakeDepth()
    return DepthAnything(dcfg["model_id"], device)
