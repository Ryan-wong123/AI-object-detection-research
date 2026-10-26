"""VLM back ends for the origin-claim study (H5): Qwen2.5-VL-3B (4-bit) and Florence-2-large.

Every back end answers two prompt settings on the final image only (golden rule 3):
  plain     the model finds the objects itself and states each one's origin
  grounded  it is given the boxes of a frozen detector (``vlm.grounding``) and states the origin
            of each box
The third setting of the study (origin-aware: boxes plus forensic evidence) needs the week-4
origin scores and is not implemented yet.

Qwen2.5-VL follows instructions and answers in JSON (``bbox_2d`` in absolute pixels of its
resized input). Florence-2 does not follow free-form instructions: its plain setting is
``<DENSE_REGION_CAPTION>`` (its own regions with captions), its grounded setting is
``<REGION_TO_DESCRIPTION>`` for each detector box, and a region's origin is read from its text
(``claims.text_origin``: an AI/editing word, else no statement).

Heavy imports happen inside the classes; ``FakeVLM`` runs on CPU for the tests.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from inlayfake.vlm.claims import boxes_prompt_lines, caption_text, parse_json_claims, text_origin


class QwenVL25:
    name = "qwen25vl"

    def __init__(
        self,
        model_id: str,
        device: str,
        prompts: dict,
        load_in_4bit: bool = True,
        max_new_tokens: int = 768,
    ) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        kw: dict = {"device_map": {"": device}}
        if load_in_4bit and device.startswith("cuda"):
            kw["quantization_config"] = BitsAndBytesConfig(  # vision encoder stays 16-bit
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual", "lm_head"],
            )
            kw["dtype"] = torch.float16
        else:
            kw["dtype"] = torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw).eval()
        self.prompts, self.max_new_tokens, self.torch = prompts, max_new_tokens, torch

    def _input_size(self, img: np.ndarray) -> tuple[int, int]:
        """(W, H) of the image as the model sees it (resized to multiples of 28)."""
        out = self.processor.image_processor(images=[Image.fromarray(img)], return_tensors="pt")
        _, gh, gw = (int(v) for v in out["image_grid_thw"][0])
        patch = self.processor.image_processor.patch_size
        return gw * patch, gh * patch

    def _generate(self, img: np.ndarray, text: str) -> str:
        torch = self.torch
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
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def plain(self, img: np.ndarray) -> dict:
        H, W = img.shape[:2]
        wi, hi = self._input_size(img)
        raw = self._generate(img, self.prompts["plain"])
        claims = parse_json_claims(raw, W / wi, H / hi, W, H)
        return {"raw": raw, "caption": caption_text(raw), "claims": claims}

    def grounded(self, img: np.ndarray, boxes: list[dict]) -> dict:
        H, W = img.shape[:2]
        wi, hi = self._input_size(img)
        lines = boxes_prompt_lines(boxes, wi / W, hi / H)
        raw = self._generate(img, self.prompts["grounded"].format(boxes=lines))
        by_id = {b["id"]: b for b in boxes}
        claims = []
        for c in parse_json_claims(raw, W / wi, H / hi, W, H):
            b = by_id.get(c["box_id"])
            if b is not None:  # answers about an id that was not given are dropped
                claims.append(c | {"bbox": b["bbox"], "label": c["label"] or b["category"]})
        return {"raw": raw, "caption": caption_text(raw), "claims": claims}


class Florence2:
    name = "florence2"

    def __init__(self, model_id: str, device: str, max_new_tokens: int = 256) -> None:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
        self.model = self.model.to(device).eval()
        self.device, self.dtype, self.torch, self.max_new_tokens = (
            device,
            dtype,
            torch,
            max_new_tokens,
        )

    def _run(self, img: np.ndarray, task: str, text: str | None = None):
        torch = self.torch
        H, W = img.shape[:2]
        inputs = self.processor(text=text or task, images=Image.fromarray(img), return_tensors="pt")
        inputs = inputs.to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, num_beams=3, do_sample=False
            )
        raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        return raw, self.processor.post_process_generation(raw, task=task, image_size=(W, H))[task]

    def plain(self, img: np.ndarray) -> dict:
        raw_cap, caption = self._run(img, "<MORE_DETAILED_CAPTION>")
        raw, parsed = self._run(img, "<DENSE_REGION_CAPTION>")
        claims = [
            {
                "label": lab,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "box_id": None,
                "origin": text_origin(lab),
            }
            for (x1, y1, x2, y2), lab in zip(parsed["bboxes"], parsed["labels"], strict=False)
        ]
        return {
            "raw": raw,
            "caption": str(caption),
            "caption_origin": text_origin(str(caption)),
            "claims": claims,
        }

    def grounded(self, img: np.ndarray, boxes: list[dict]) -> dict:
        H, W = img.shape[:2]
        claims, raws = [], []
        for b in boxes:
            x, y, w, h = b["bbox"]
            loc = [
                int(np.clip(v, 0, 999))
                for v in (x / W * 999, y / H * 999, (x + w) / W * 999, (y + h) / H * 999)
            ]
            text = "<REGION_TO_DESCRIPTION>" + "".join(f"<loc_{v}>" for v in loc)
            raw, desc = self._run(img, "<REGION_TO_DESCRIPTION>", text)
            raws.append(raw)
            claims.append(
                {
                    "label": str(desc),
                    "bbox": b["bbox"],
                    "box_id": b["id"],
                    "origin": text_origin(str(desc)),
                }
            )
        return {"raw": "\n".join(raws), "caption": "", "claims": claims}


class FakeVLM:
    """CPU stand-in: claims every given (or ground-truth-like) box; calls boxes whose left edge
    is in the right half of the image ``ai``, the others ``photographed``."""

    name = "fake"

    def __init__(self) -> None:
        self.truth: list[dict] = []

    def _origin(self, bbox, W) -> str:
        return "ai" if bbox[0] > W / 2 else "photographed"

    def plain(self, img: np.ndarray) -> dict:
        W = img.shape[1]
        claims = [
            {
                "label": t["category"],
                "bbox": t["bbox"],
                "box_id": None,
                "origin": self._origin(t["bbox"], W),
            }
            for t in self.truth
        ]
        return {"raw": "", "caption": "fake", "claims": claims}

    def grounded(self, img: np.ndarray, boxes: list[dict]) -> dict:
        W = img.shape[1]
        claims = [
            {
                "label": b["category"],
                "bbox": b["bbox"],
                "box_id": b["id"],
                "origin": self._origin(b["bbox"], W),
            }
            for b in boxes
        ]
        return {"raw": "", "caption": "fake", "claims": claims}


def build_vlm_model(name: str, cfg: dict):
    mcfg = cfg["vlm"]["models"][name]
    device = cfg["hardware"]["device"]
    impl = mcfg["impl"]
    if impl == "fake":
        return FakeVLM()
    if impl == "qwen25vl":
        return QwenVL25(
            mcfg["model_id"],
            device,
            cfg["vlm"]["prompts"],
            mcfg.get("load_in_4bit", True),
            mcfg.get("max_new_tokens", 768),
        )
    if impl == "florence2":
        return Florence2(mcfg["model_id"], device, mcfg.get("max_new_tokens", 256))
    raise ValueError(f"unknown VLM impl {impl!r}")
