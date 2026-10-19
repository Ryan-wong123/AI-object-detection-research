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
    are) and answers the realism questions as a judge that finds every edit natural."""

    name = "fake"

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()

    def point(self, img: np.ndarray, prompt: str) -> tuple[list[tuple[float, float]], str]:
        H, W = img.shape[:2]
        pts = [(0.5 * W, 0.97 * H), (0.2 * W, 0.95 * H), (0.8 * W, 0.95 * H)]
        return pts, "fake points"

    def ask(self, img: np.ndarray, question: str, record_id: str = "") -> str:
        """Answers as a judge that finds the edit natural ("no" only to the seen-through-glass
        question); 'unclear' (a failing answer) for records in ``fail_ids``."""
        if record_id in self.fail_ids:
            return "unclear"
        return "no" if "seen through" in question else "yes"


class MolmoVLM:
    name = "molmo"

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = True) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        # one GPU: the whole model on it (transformers 5 cannot balance Molmo's own model code)
        kw: dict = {"trust_remote_code": True, "device_map": {"": device}}
        if load_in_4bit and device.startswith("cuda"):
            # the vision encoder and output layer stay in 16 bit (usual for Molmo in 4-bit); naming
            # them also spares transformers 5 from inspecting tied weights, which Molmo's own
            # (transformers-4) model code does not declare
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=["model.vision_backbone", "model.transformer.ff_out"],
            )
            kw["torch_dtype"] = torch.float16
        else:
            kw["torch_dtype"] = torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module("modeling_molmo.MolmoForCausalLM", model_id)
        if not hasattr(cls, "all_tied_weights_keys"):
            # transformers 5 expects this map of tied weights; Molmo-7B-D ties none
            # (config weight_tying: false), so it is empty
            cls.all_tied_weights_keys = {}
        if not getattr(cls, "_inlayfake_patched", False):
            # transformers 5 passes new keyword arguments to tie_weights; Molmo's version takes
            # none (and ties nothing here), so they are dropped
            original = cls.tie_weights
            cls.tie_weights = lambda self, *a, **k: original(self)
            cls._inlayfake_patched = True
        from transformers import GenerationMixin

        if not issubclass(cls, GenerationMixin):
            # transformers 5 models get generation helpers only from GenerationMixin
            cls = type(cls.__name__, (cls, GenerationMixin), {})
        self.model = cls.from_pretrained(model_id, **kw)
        self.model.eval()

    def _generate(self, img: np.ndarray, text: str, max_new_tokens: int) -> str:
        """Greedy decoding with Molmo's own forward pass and its tuple key/value cache.

        transformers 5's generate() hands models a DynamicCache, which Molmo's code (written for
        transformers 4) cannot read; this loop repeats what Molmo's generate_from_batch and
        prepare_inputs_for_generation do: position ids from the attention mask, the image
        inputs on the first step only, then one token at a time with the cache."""
        import torch

        inputs = self.processor.process(images=[Image.fromarray(img)], text=text)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        if "images" in inputs:
            inputs["images"] = inputs["images"].to(torch.float16)
        input_ids = inputs["input_ids"]
        eos = self.processor.tokenizer.eos_token_id
        step: dict = {"use_cache": True}
        if self.model.config.use_position_ids:
            mask = input_ids != -1
            position_ids = torch.clamp(torch.cumsum(mask.to(torch.int32), dim=-1) - 1, min=0)
            attention_mask = torch.cat([mask, mask.new_ones((1, max_new_tokens))], dim=1)
            step.update(attention_mask=attention_mask, last_logits_only=True)
            first = {
                "position_ids": position_ids,
                "append_last_valid_logits": mask.long().sum(dim=-1) - 1,
            }
        else:
            position_ids, first = None, {}
        for k in ("images", "image_masks", "image_input_idx"):
            if k in inputs:
                first[k] = inputs[k]
        new: list[int] = []
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, past_key_values=None, **step, **first)
            for _ in range(max_new_tokens):
                tok = int(out.logits[0, -1].argmax())
                if tok == eos:
                    break
                new.append(tok)
                nxt = {"past_key_values": out.past_key_values}
                if position_ids is not None:
                    position_ids = position_ids[:, -1:] + 1
                    nxt["position_ids"] = position_ids
                ids = torch.tensor([[tok]], device=input_ids.device)
                out = self.model(input_ids=ids, **step, **nxt)
        text_out = self.processor.tokenizer.decode(new, skip_special_tokens=True)
        return text_out.split("<|endoftext|>")[0]

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
