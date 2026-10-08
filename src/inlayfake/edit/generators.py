"""Inpainting generator wrappers (G1 SDXL, G2 SD2, G3 FLUX Fill, fallback Kandinsky).

All wrappers are callables ``(image, mask, prompt, negative_prompt, seed) -> PIL.Image`` at the
generator's native square resolution. Heavy libraries are imported lazily.
"""

from __future__ import annotations

import inspect

import numpy as np
from PIL import Image, ImageDraw


class FakeInpainter:
    """Deterministic stand-in for tests and dry runs: paints a noisy ellipse inside the mask."""

    def __init__(self, resolution: int = 256, **_: object) -> None:
        self.resolution = resolution
        self.model_id = "fake"

    def __call__(self, image, mask, prompt, negative_prompt, seed):
        rng = np.random.default_rng(seed)
        out = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        m = np.asarray(mask.convert("L")) > 127
        if "empty" not in prompt:  # sham prompt -> background-like fill, no "object"
            ys, xs = np.nonzero(m)
            if len(xs):
                canvas = Image.new("L", image.size, 0)
                ImageDraw.Draw(canvas).ellipse(
                    [xs.min() + 2, ys.min() + 2, xs.max() - 2, ys.max() - 2], fill=255
                )
                obj = np.asarray(canvas) > 0
                color = rng.integers(0, 256, size=3)
                noise = rng.integers(-20, 21, size=out.shape)
                out[obj] = np.clip(color + noise[obj], 0, 255).astype(np.uint8)
                return Image.fromarray(out)
        mean = out[~m].mean(axis=0) if (~m).any() else np.array([127, 127, 127])
        out[m] = np.clip(mean + rng.normal(0, 3, size=(m.sum(), 3)), 0, 255).astype(np.uint8)
        return Image.fromarray(out)


class DiffusersInpainter:
    """Wraps a diffusers inpainting pipeline with 8 GB-VRAM-friendly defaults."""

    def __init__(
        self,
        model_id: str,
        resolution: int,
        steps: int = 30,
        guidance: float = 7.5,
        strength: float | None = 0.99,
        device: str = "cuda",
        dtype: str = "float16",
        offload: str = "model",  # model | sequential | none
        gguf_transformer: str | None = None,  # FLUX only: quantised transformer file/URL
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.resolution = resolution
        self.steps = steps
        self.guidance = guidance
        self.strength = strength
        self.device = device
        self.dtype = dtype
        self.offload = offload
        self.gguf_transformer = gguf_transformer
        self.revision = revision
        self._pipe = None

    @property
    def is_flux(self) -> bool:
        return "flux" in self.model_id.lower()

    def _load(self):
        import torch

        dtype = getattr(torch, self.dtype)
        if self.is_flux:
            from diffusers import FluxFillPipeline

            kwargs = {"torch_dtype": torch.bfloat16, "revision": self.revision}
            if self.gguf_transformer:
                from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

                kwargs["transformer"] = FluxTransformer2DModel.from_single_file(
                    self.gguf_transformer,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    config=self.model_id,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                )
            pipe = FluxFillPipeline.from_pretrained(self.model_id, **kwargs)
        else:
            from diffusers import AutoPipelineForInpainting

            try:
                pipe = AutoPipelineForInpainting.from_pretrained(
                    self.model_id, torch_dtype=dtype, variant="fp16", revision=self.revision
                )
            except (OSError, ValueError):  # repo without an fp16 variant
                pipe = AutoPipelineForInpainting.from_pretrained(
                    self.model_id, torch_dtype=dtype, revision=self.revision
                )
        pipe.set_progress_bar_config(disable=True)
        if self.offload == "model":
            pipe.enable_model_cpu_offload()
        elif self.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)
        self._pipe = pipe
        self._params = set(inspect.signature(pipe.__call__).parameters)
        return pipe

    def __call__(self, image, mask, prompt, negative_prompt, seed):
        import torch

        pipe = self._pipe or self._load()
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask_image": mask,
            "height": self.resolution,
            "width": self.resolution,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance,
            "strength": self.strength,
            "generator": torch.Generator("cpu").manual_seed(int(seed)),
        }
        kwargs = {k: v for k, v in kwargs.items() if k in self._params and v is not None}
        return pipe(**kwargs).images[0]


def build_inpainter(gen_cfg: dict, edit_cfg: dict, hw_cfg: dict):
    """Create the wrapper for one generator entry of ``configs/*.yaml: generators``."""
    model_id = gen_cfg["model_id"]
    if model_id == "fake":
        return FakeInpainter(resolution=gen_cfg.get("resolution", 256))
    return DiffusersInpainter(
        model_id=model_id,
        resolution=gen_cfg["resolution"],
        steps=gen_cfg.get("steps", edit_cfg["steps"]),
        guidance=gen_cfg.get("guidance", edit_cfg["guidance"]),
        strength=gen_cfg.get("strength", 0.99),
        device=hw_cfg["device"],
        dtype=hw_cfg["dtype"],
        offload=gen_cfg.get("offload", "model" if hw_cfg.get("cpu_offload", True) else "none"),
        gguf_transformer=gen_cfg.get("gguf_transformer"),
        revision=gen_cfg.get("revision"),
    )
