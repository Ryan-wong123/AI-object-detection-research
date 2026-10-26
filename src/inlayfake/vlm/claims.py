"""Turn VLM output text into per-object origin claims.

A claim is ``{"label", "bbox" ([x, y, w, h] in original pixels, or None), "box_id" (grounded
prompts: the id of the detector box it answers, or None), "origin"}`` with origin one of
  ai            the model says the object was made or inserted by an AI editor
  photographed  the model says it was photographed with the scene
  unknown       the model says it cannot tell (an explicit abstention)
  none          the model describes the object but makes no origin statement (Florence-2)
"""

from __future__ import annotations

import json
import re

ORIGINS = ("ai", "photographed", "unknown", "none")

# words that make an origin statement in free text (Florence-2 captions); "photo of" is not one
AI_WORDS = re.compile(
    r"\b(ai[- ]?(generated|made|created)|artificial(ly)?|computer[- ]generated|cgi|synthetic|"
    r"generated|fake|photoshopp\w*|manipulated|edited|inserted|pasted|rendered|rendering|"
    r"digitally (added|altered|created|inserted))\b",
    re.IGNORECASE,
)

_OBJ = re.compile(r"\{[^{}]*\}")
_NUM = r"-?\d+(?:\.\d+)?"
_BBOX = re.compile(
    rf'"bbox_2d"\s*:\s*\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]'
)
_FIELD = '"{}"\\s*:\\s*"([^"]*)"'
_ID = re.compile(r'"id"\s*:\s*"?(\d+)"?')


def normalize_origin(text: str) -> str:
    t = text.strip().lower()
    if not t:
        return "unknown"
    if re.search(r"\bai\b|\bai[_ -]|generat|fake|insert|synthetic|edit", t):  # not "uncertain"
        return "ai"
    if any(w in t for w in ("photo", "real", "camera", "original", "natural")):
        return "photographed"
    return "unknown"


def text_origin(text: str) -> str:
    """Origin statement in free text: ``ai`` if an AI/editing word occurs, else ``none``."""
    return "ai" if AI_WORDS.search(text) else "none"


def parse_json_claims(
    text: str, sx: float = 1.0, sy: float = 1.0, W: float = 1e9, H: float = 1e9
) -> list[dict]:
    """Claims from the JSON objects in a reply (tolerates code fences, prose and truncation: each
    ``{...}`` is read on its own). ``bbox_2d`` [x1, y1, x2, y2] is scaled by (sx, sy) into
    original pixels and clipped to W×H."""
    claims = []
    for obj in _OBJ.findall(text):
        label = re.search(_FIELD.format("label"), obj)
        origin = re.search(_FIELD.format("origin"), obj)
        box_id = _ID.search(obj)
        bbox = None
        m = _BBOX.search(obj)
        if m:
            x1, y1, x2, y2 = (float(v) for v in m.groups())
            x1, x2 = sorted((min(max(x1 * sx, 0), W), min(max(x2 * sx, 0), W)))
            y1, y2 = sorted((min(max(y1 * sy, 0), H), min(max(y2 * sy, 0), H)))
            bbox = [x1, y1, x2 - x1, y2 - y1] if x2 - x1 >= 1 and y2 - y1 >= 1 else None
        if bbox is None and box_id is None:
            continue
        claims.append(
            {
                "label": label.group(1).strip() if label else "",
                "bbox": bbox,
                "box_id": int(box_id.group(1)) if box_id else None,
                "origin": normalize_origin(origin.group(1)) if origin else "unknown",
            }
        )
    return claims


def caption_text(text: str) -> str:
    """The prose before the first JSON block or code fence."""
    cut = min([i for i in (text.find("```"), text.find("[")) if i >= 0], default=len(text))
    return text[:cut].strip()


def boxes_prompt_lines(boxes: list[dict], sx: float = 1.0, sy: float = 1.0) -> str:
    """``id: class [x1, y1, x2, y2]`` lines for a grounded prompt, boxes scaled into the model's
    input pixels (Qwen2.5-VL answers and reads absolute coordinates of its resized input)."""
    lines = []
    for b in boxes:
        x, y, w, h = b["bbox"]
        c = [round(x * sx), round(y * sy), round((x + w) * sx), round((y + h) * sy)]
        lines.append(f"{b['id']}: {b['category']} {json.dumps(c)}")
    return "\n".join(lines)
