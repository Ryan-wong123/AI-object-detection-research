"""Natural placement stages (``placement.natural``, author decision 2026-10-19).

``plan_natural`` (called by `plan`): per photo, a VLM is asked one fixed "Point to ..." question
(``configs/placement_prompts.yaml``) for up to ``classes_per_photo`` classes already in the photo;
every point becomes a depth-scaled box (``inlayfake.edit.natural``) and is checked against the box
rules. All candidates go to ``<work>/natural_candidates.jsonl``; the first valid one is the plan's
box (``box_free``: C2, C3 and C4 share it; no C5). The other valid candidates are the re-plan
versions 1, 2, ... (``natural_version``), so a failed edit moves to the next natural spot instead
of a random one. A photo with no valid candidate gets no plan (no random fallback).

``realism_pass`` (called by `verify`): every C4 attempt OWLv2 accepted is shown to the VLM with the
object boxed in red and the fixed yes/no questions; any answer other than the expected one turns
the attempt into a failure (``reason: implausible: ...``), so `insert` regenerates it.
Answers go to ``<work>/realism.jsonl``; ``verify.verify_state`` applies them.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from inlayfake.edit.encode import load_rgb
from inlayfake.edit.natural import (
    BLOCKED_STUFF,
    SKY_STUFF,
    box_from_point,
    check_box,
    check_questions,
    fit_inside,
    load_prompts,
    pointing_prompt,
    usable_references,
)

CANDIDATES = "natural_candidates.jsonl"
REALISM = "realism.jsonl"


def enabled(cfg: dict) -> bool:
    return bool(cfg.get("placement", {}).get("natural", {}).get("enabled", False))


def _free_gpu() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def _labels(index, sid: int, extra: list[dict]) -> list[dict]:
    """The photo's real objects with class names: COCO labels plus week-1 audit rows ("add" are
    real objects; ignore / "?" are unresolved regions, treated like objects here so a box never
    hides them either)."""
    out = [
        {
            "id": a["id"],
            "category": index.cat_name[a["category_id"]],
            "bbox": [float(v) for v in a["bbox"]],
            "iscrowd": a.get("iscrowd", 0),
        }
        for a in index.image_anns(sid)
    ]
    out += [
        {"id": e["id"], "category": e["category"], "bbox": e["bbox"], "iscrowd": 0} for e in extra
    ]
    return out


def _typical_wh(stats: dict, cat: str, W: int, H: int) -> tuple[float, float]:
    """Median COCO size of a class, as (w, h) in this image."""
    frac, aspect = np.median(stats[cat][:, 0]), np.median(stats[cat][:, 1])
    w = float(np.sqrt(frac * W * H * aspect))
    return w, w / aspect


def plan_natural(cfg, index, sources: list[dict], extra: dict, stats: dict, wd: Path) -> dict:
    """Natural-placement plans for ``sources``; returns {plans, failed, candidates}."""
    from inlayfake.edit.scene import build_depth, build_vlm
    from inlayfake.edit.support import StuffIndex
    from inlayfake.stages.common import derive_seed
    from inlayfake.stages.plan import balanced_order, present_classes

    pcfg, ncfg = cfg["placement"], cfg["placement"]["natural"]
    prompts = load_prompts(ncfg["prompts"])
    rules = prompts["rules"]
    exclude = set(pcfg.get("exclude_classes", ["person"]))
    spath = pcfg.get("support", {}).get("stuff_annotations")
    stuff = StuffIndex.load(spath) if spath and Path(spath).exists() else None
    device = cfg["hardware"]["device"]
    depth_model = build_depth(ncfg["depth"], device)
    vlm = build_vlm(ncfg["vlm"], device)
    rng = np.random.default_rng(derive_seed(cfg["seed"], "natural_plan"))
    used: Counter = Counter()
    plans, failed, rows = [], [], []
    for src in tqdm(sources, desc="natural placement"):
        sid, W, H = src["source_id"], src["width"], src["height"]
        labels = _labels(index, sid, extra.get(sid, []))
        classes = [
            c
            for c in present_classes(index, index.image_anns(sid), extra.get(sid, []), exclude)
            if c in stats and c in prompts["classes"]
        ]
        if not classes:
            failed.append({"source_id": sid, "reason": "no insertable class"})
            continue
        img = load_rgb(index.image_path(sid))
        depth = depth_model(img)
        blocked_all = stuff.mask(sid, set(BLOCKED_STUFF), H, W) if stuff else None
        sky = stuff.mask(sid, set(SKY_STUFF), H, W) if stuff else None
        valid = []
        for cat in balanced_order(classes, used, rng)[: ncfg.get("classes_per_photo", 2)]:
            air = prompts["classes"][cat]["air"]
            blocked = blocked_all & ~sky if (air and blocked_all is not None) else blocked_all
            same = [lb for lb in labels if lb["category"] == cat]
            refs = usable_references(same, labels, W, H)
            prompt = pointing_prompt(prompts, cat)
            points, answer = vlm.point(img, prompt)
            if not points:
                rows.append(
                    {
                        "source_id": sid,
                        "category": cat,
                        "rank": None,
                        "prompt": prompt,
                        "answer": answer,
                        "ok": False,
                        "reason": "no point in the answer",
                    }
                )
            for rank, (px, py) in enumerate(points[: rules.get("max_candidates", 5)]):
                on_sky = bool(
                    air
                    and sky is not None
                    and sky[int(np.clip(py, 0, H - 1)), int(np.clip(px, 0, W - 1))]
                )
                box, info = box_from_point(
                    (px, py),
                    depth,
                    refs,
                    _typical_wh(stats, cat, W, H),
                    on_sky,
                    tuple(ncfg.get("size_ratio", (0.5, 2.0))),
                    pcfg["max_area_frac"],
                    pcfg["min_side_px"],
                )
                fitted = fit_inside(box, W, H)
                if fitted is None:
                    ok, reason = False, "outside the image"
                else:
                    ok, reason = check_box(fitted, labels, blocked, rules)
                row = {
                    "source_id": sid,
                    "category": cat,
                    "rank": rank,
                    "point": [round(px, 1), round(py, 1)],
                    "on_sky": on_sky,
                    "box": [round(v, 1) for v in (fitted or box)],
                    "ok": ok,
                    "reason": reason,
                    "prompt": prompt,
                    "answer": answer,
                    "vlm": vlm.name,
                } | info
                if ok:
                    row["valid_rank"] = len(valid)
                    valid.append(row)
                rows.append(row)
        if not valid:
            failed.append({"source_id": sid, "reason": "no natural spot"})
            continue
        first = valid[0]
        used[first["category"]] += 1
        plans.append(
            {
                "source_id": sid,
                "split": src["split"],
                "category": first["category"],
                "box_free": first["box"],
                "box_occ": None,
                "occluded_ann_id": None,
                "class_in_photo": True,
                "placement_rule": "natural",
            }
        )
    with (wd / CANDIDATES).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    del vlm, depth_model
    _free_gpu()
    return {"plans": plans, "failed": failed, "candidates": rows}


def load_valid(wd: Path) -> dict[int, list[dict]]:
    """Valid candidates per source, in plan order (version v = valid[v])."""
    p = wd / CANDIDATES
    out: dict[int, list[dict]] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else None
            if r and r.get("ok"):
                out.setdefault(r["source_id"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["valid_rank"])
    return out


def natural_version(valid: dict[int, list[dict]], sid: int, track: str, version: int) -> dict:
    """Re-plan row for version ``version``: the next valid natural spot (box None if none is
    left, which ends the ladder)."""
    cands = valid.get(sid, [])
    row = {
        "source_id": sid,
        "track": track,
        "version": version,
        "category": None,
        "class_in_photo": True,
        "box": None,
        "occluded_ann_id": None,
        "support": None,
        "placement_rule": "natural",
    }
    if version < len(cands):
        c = cands[version]
        row.update(category=c["category"], box=c["box"])
    return row


# ---------------------------------------------------------------------------- realism check
def read_realism(wd: Path) -> dict[str, dict]:
    p = wd / REALISM
    if not p.exists():
        return {}
    return {
        r["image_id"]: r
        for r in (json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip())
    }


def _boxed(img: np.ndarray, box: list[float]) -> np.ndarray:
    pil = Image.fromarray(img.copy())
    x, y, w, h = box
    t = max(2, round(max(pil.size) / 200))
    ImageDraw.Draw(pil).rectangle(
        [x - t, y - t, x + w + t, y + h + t], outline=(255, 0, 0), width=t
    )
    return np.asarray(pil)


def realism_pass(cfg: dict, wd: Path, state: dict) -> dict:
    """VLM yes/no check of every C4 attempt OWLv2 accepted and not yet checked; failures are
    turned into failed attempts in ``state`` (and on disk via realism.jsonl)."""
    from inlayfake.edit.scene import build_vlm, parse_yes_no
    from inlayfake.ledger import Origin

    ncfg = cfg["placement"]["natural"]
    done = read_realism(wd)
    todo = [
        r
        for rows in state.values()
        for r in rows
        if r["condition"] == "C4" and r["passed"] and r["image_id"] not in done
    ]
    if not todo:
        return {"checked": 0, "rejected": 0}
    from inlayfake.ledger import ImageRecord

    prompts = load_prompts(ncfg["prompts"])
    vlm = build_vlm(ncfg["vlm"], cfg["hardware"]["device"])
    rejected = 0
    with (wd / REALISM).open("a", encoding="utf-8") as f:
        for row in tqdm(todo, desc="realism check"):
            p = wd / "verified" / f"{row['image_id']}.json"
            if not p.exists():
                continue
            rec = ImageRecord(**json.loads(p.read_text(encoding="utf-8")))
            ins = next(
                (i.bbox for i in rec.instances if i.origin is Origin.AI_INSERTED),
                rec.edit.plan_box,
            )
            img = _boxed(load_rgb(rec.path), ins)
            answers, raw = {}, {}
            questions = check_questions(prompts, rec.edit.target_category)
            for qid, q, _ in questions:
                text = vlm.ask(img, q, rec.image_id)
                raw[qid], answers[qid] = text, parse_yes_no(text)
            bad = [qid for qid, _, exp in questions if answers[qid] != exp]
            out = {
                "image_id": row["image_id"],
                "key": row["key"],
                "version": row.get("version", 0),
                "passed": not bad,
                "failed": bad,
                "answers": answers,
                "raw": raw,
                "vlm": vlm.name,
            }
            f.write(json.dumps(out) + "\n")
            f.flush()
            if bad:
                rejected += 1
                row["passed"] = False
                row["reason"] = "implausible: " + ", ".join(bad)
    del vlm
    _free_gpu()
    return {"checked": len(todo), "rejected": rejected}


def apply_realism(wd: Path, state: dict) -> None:
    """Turn attempts the realism check rejected into failures (used when state is loaded)."""
    done = read_realism(wd)
    if not done:
        return
    for rows in state.values():
        for r in rows:
            d = done.get(r["image_id"])
            if d is not None and not d["passed"]:
                r["passed"] = False
                r["reason"] = "implausible: " + ", ".join(d["failed"])
