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
import os
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
    box_prompt,
    check_box,
    check_questions,
    dedupe_boxes,
    fit_inside,
    load_prompts,
    point_support,
    pointing_prompt,
    spread_points,
    usable_references,
)

CANDIDATES = "natural_candidates.jsonl"
PROGRESS = "natural_plan_progress.jsonl"  # one line per planned photo, so planning can resume
POINTS = "natural_points.jsonl"  # the VLM's answer per (photo, class, prompt): asked only once
REALISM = "realism.jsonl"


def enabled(cfg: dict) -> bool:
    return bool(cfg.get("placement", {}).get("natural", {}).get("enabled", False))


def _free_gpu() -> None:
    """Release GPU memory of models that were just deleted. Large transformers models keep
    reference cycles, so ``del`` alone leaves them in memory until the garbage collector runs:
    in pilot attempt 2 (19 Oct) Molmo stayed on the GPU after planning and SDXL, in the same
    process, slowed from ~20 s to ~11 min per image."""
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def _free_cuda_cache() -> None:
    """Release cached GPU memory between photos (no-op without torch or CUDA)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _stuff_label_map(stuff, sid: int, H: int, W: int) -> tuple[np.ndarray, list[str]]:
    """Per-pixel COCO-Stuff class index (0 = none) and the class names."""
    from inlayfake.edit.support import _decode

    names = [""]
    lab = np.zeros((H, W), dtype=np.int16)
    for a in stuff.anns_by_image.get(sid, []):
        names.append(stuff.cat_name.get(a["category_id"], ""))
        lab[_decode(a["segmentation"], H, W)] = len(names) - 1
    return lab, names


def _label_below(lab: np.ndarray, names: list[str], x: float, y: float, r: int = 3) -> str | None:
    """Most common COCO-Stuff class in a small window just at/below a point (where the object's
    base meets its support); None if nothing is labelled there."""
    H, W = lab.shape
    xi, yi = int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y) + r, 0, H - 1))
    win = lab[max(yi - r, 0) : yi + r + 1, max(xi - r, 0) : xi + r + 1].ravel()
    win = win[win > 0]
    if win.size == 0:
        return None
    return names[int(np.bincount(win).argmax())]


def _thing_masks(index, sid: int) -> list[tuple[str, np.ndarray, float]]:
    """(class, mask, area) of the photo's COCO objects, smallest first."""
    out = []
    for a in index.image_anns(sid):
        m = index.ann_mask(a)
        out.append((index.cat_name[a["category_id"]], m, float(m.sum())))
    return sorted(out, key=lambda t: t[2])


def _thing_at(things, x: float, y: float) -> str | None:
    """Class of the smallest labelled object whose mask contains the point (None: none)."""
    for cat, m, _ in things:
        H, W = m.shape
        if m[int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))]:
            return cat
    return None


def _typical_wh(stats: dict, cat: str, W: int, H: int) -> tuple[float, float]:
    """Median COCO size of a class, as (w, h) in this image."""
    frac, aspect = np.median(stats[cat][:, 0]), np.median(stats[cat][:, 1])
    w = float(np.sqrt(frac * W * H * aspect))
    return w, w / aspect


def plan_settings(cfg: dict, prompts: dict, rules: dict) -> dict:
    """Everything except the VLM's answers that decides a photo's candidates. A saved photo is
    reused only if it was planned with the same settings; otherwise it is re-checked (cheap: the
    VLM's answers come from ``POINTS``)."""
    pcfg, ncfg = cfg["placement"], cfg["placement"]["natural"]
    keys = (
        "mode",
        "check_point_surface",
        "min_box_frac",
        "min_box_side_px",
        "size_ratio",
        "classes_per_photo",
        "max_candidates",
        "min_point_dist",
    )
    out = {k: ncfg.get(k) for k in keys}
    out |= {
        "rules": rules,
        "max_area_frac": pcfg["max_area_frac"],
        "min_side_px": pcfg["min_side_px"],
        "depth": ncfg["depth"].get("model_id", ncfg["depth"].get("backend")),
        "prompts": prompts,
    }

    # JSON round trip: tuples/lists and key order compare equal to a saved copy; sets (surface
    # lists in the prompt file) sorted, since their order changes between Python processes
    def plain(o):
        return sorted(o) if isinstance(o, (set, frozenset)) else str(o)

    return json.loads(json.dumps(out, sort_keys=True, default=plain))


def _point_key(sid: int, cat: str, prompt: str, vlm_id: str) -> str:
    return json.dumps([sid, cat, prompt, vlm_id])


def _vlm_id(vcfg: dict) -> str:
    return f"{vcfg.get('backend', 'molmo')}:{vcfg.get('model_id', '')}"


def _load_points(wd: Path, old_rows: list[dict], sizes: dict, vcfg: dict) -> dict:
    """Saved VLM answers: ``POINTS``, plus answers kept in progress rows saved before this cache
    existed (Molmo answers are re-read from their text with the same parser as when asked)."""
    from inlayfake.edit.scene import parse_molmo_points
    from inlayfake.stages.common import read_jsonl

    vid = _vlm_id(vcfg)
    cache: dict[str, dict] = {}
    if vcfg.get("backend", "molmo") == "molmo":
        for r in old_rows:
            if r.get("vlm") == "molmo" and "answer" in r and r["source_id"] in sizes:
                W, H = sizes[r["source_id"]]
                pts = parse_molmo_points(r["answer"], W, H)
                k = _point_key(r["source_id"], r["category"], r["prompt"], vid)
                cache[k] = {"key": k, "points": pts, "answer": r["answer"], "vlm": "molmo"}
    if (wd / POINTS).exists():
        for r in read_jsonl(wd / POINTS):
            cache[r["key"]] = r
    return cache


def plan_natural(cfg, index, sources: list[dict], extra: dict, stats: dict, wd: Path) -> dict:
    """Natural-placement plans for ``sources``; returns {plans, failed, candidates}."""
    from inlayfake.edit.scene import build_depth, build_vlm
    from inlayfake.edit.support import StuffIndex
    from inlayfake.stages.common import derive_seed
    from inlayfake.stages.plan import balanced_order, present_classes

    pcfg, ncfg = cfg["placement"], cfg["placement"]["natural"]
    prompts = load_prompts(ncfg["prompts"])
    # box rules from the prompt file, overridden per run by placement.natural.box_rules
    rules = {**prompts["rules"], **(ncfg.get("box_rules") or {})}
    exclude = set(pcfg.get("exclude_classes", ["person"]))
    spath = pcfg.get("support", {}).get("stuff_annotations")
    stuff = StuffIndex.load(spath) if spath and Path(spath).exists() else None
    settings = plan_settings(cfg, prompts, rules)
    device = cfg["hardware"]["device"]
    depth_model = vlm = None  # loaded when first needed (the VLM only if an answer is not saved)
    rng = np.random.default_rng(derive_seed(cfg["seed"], "natural_plan"))
    used: Counter = Counter()
    plans, failed, rows = [], [], []
    # resume (author, 2026-10-20): every planned photo is saved at once, so a stopped or crashed
    # run continues where it stopped; finished photos are replayed (same random draws and class
    # balance as an uninterrupted run, so the plans are identical)
    # Rule changes (author, 2026-10-20): a photo saved with other settings is re-checked with the
    # current ones. The VLM's answers are saved separately (POINTS) and reused, so a re-check
    # costs the depth model and the box rules (about a second per photo), not a new VLM question.
    done, old_rows = {}, []
    if (wd / PROGRESS).exists():
        for line in (wd / PROGRESS).read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a line cut off by a crash: that photo is redone
                continue
            old_rows.extend(rec["rows"])
            if rec.get("settings") == settings:  # a later line for the same photo wins
                done[rec["source_id"]] = rec
            else:
                done.pop(rec["source_id"], None)
    seen = {r["source_id"] for r in old_rows}
    sizes = {s["source_id"]: (s["width"], s["height"]) for s in sources}
    vid = _vlm_id(ncfg["vlm"])
    points_cache = _load_points(wd, old_rows, sizes, ncfg["vlm"])
    n_asked = n_reused = n_rechecked = 0
    points_file = (wd / POINTS).open("a", encoding="utf-8")
    progress = (wd / PROGRESS).open("a", encoding="utf-8")
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
        if sid in done:  # planned before a stop: replay it
            balanced_order(classes, used, rng)  # the same random draw as the first time
            rec = done[sid]
            rows.extend(rec["rows"])
            if rec["plan"] is not None:
                used[rec["plan"]["category"]] += 1
                plans.append(rec["plan"])
            else:
                failed.append(rec["failed"])
            continue
        if depth_model is None:
            # CUDA's own allocator fragments less: Molmo's reserved memory 7.3 -> 6.9 GB on the
            # 8 GB GPU (20 Oct); read when CUDA first allocates, which happens after this point
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
            depth_model = build_depth(ncfg["depth"], ncfg["depth"].get("device", device))
        n_rechecked += sid in seen
        n_rows = len(rows)
        img = load_rgb(index.image_path(sid))
        depth = depth_model(img)
        blocked_all = stuff.mask(sid, set(BLOCKED_STUFF), H, W) if stuff else None
        check_surface = ncfg.get("check_point_surface", False)
        if check_surface:
            stuff_map, stuff_names = _stuff_label_map(stuff, sid, H, W) if stuff else (None, [])
            things = _thing_masks(index, sid)
        sky = stuff.mask(sid, set(SKY_STUFF), H, W) if stuff else None
        valid = []
        box_mode = ncfg.get("mode", "point") == "box"
        n_max = ncfg.get("max_candidates", rules.get("max_candidates", 5))
        for cat in balanced_order(classes, used, rng)[: ncfg.get("classes_per_photo", 2)]:
            air = prompts["classes"][cat]["air"]
            blocked = blocked_all & ~sky if (air and blocked_all is not None) else blocked_all
            same = [lb for lb in labels if lb["category"] == cat]
            refs = usable_references(same, labels, W, H)
            prompt = (
                box_prompt(prompts, cat, [lb["bbox"] for lb in same], W, H)
                if box_mode
                else pointing_prompt(prompts, cat)
            )
            key = _point_key(sid, cat, prompt, vid)
            if key in points_cache:  # asked before (this run or an earlier one): reuse
                hit = points_cache[key]
                points = [tuple(pt) for pt in hit.get("points", [])]
                vboxes = hit.get("boxes", [])
                answer, vlm_name = hit["answer"], hit["vlm"]
                n_reused += 1
            else:
                if vlm is None:
                    vlm = build_vlm(ncfg["vlm"], device)
                if box_mode:
                    vboxes, answer = vlm.boxes(img, prompt)
                    points = []
                else:
                    points, answer = vlm.point(img, prompt)
                    vboxes = []
                vlm_name = vlm.name
                hit = {
                    "key": key,
                    "points": [list(pt) for pt in points],
                    "boxes": [list(b) for b in vboxes],
                    "answer": answer,
                    "vlm": vlm_name,
                }
                points_cache[key] = hit
                points_file.write(json.dumps(hit) + "\n")
                points_file.flush()
                n_asked += 1
            if box_mode:
                # the VLM's own boxes; the contact point is the bottom centre (the centre for a
                # flying class whose centre is in open sky), used for the surface check
                vboxes = dedupe_boxes(vboxes, n_max)
                points = []
                for bx, by, bw, bh in vboxes:
                    cx, cy = bx + bw / 2, by + bh / 2
                    in_sky = bool(
                        air
                        and sky is not None
                        and sky[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))]
                    )
                    points.append((cx, cy) if in_sky else (cx, by + bh - 1))
            if not points:
                rows.append(
                    {
                        "source_id": sid,
                        "category": cat,
                        "rank": None,
                        "prompt": prompt,
                        "answer": answer,
                        "ok": False,
                        "reason": "no box in the answer" if box_mode else "no point in the answer",
                    }
                )
            # spots per class: placement.natural.max_candidates (default: the prompt file's
            # box_rules), at least min_point_dist x the image diagonal apart
            spots = (
                points
                if box_mode
                else spread_points(points, W, H, n_max, ncfg.get("min_point_dist", 0.1))
            )
            for rank, (px, py) in enumerate(spots):
                on_sky = bool(
                    air
                    and sky is not None
                    and sky[int(np.clip(py, 0, H - 1)), int(np.clip(px, 0, W - 1))]
                )
                if check_surface:  # fix 1: the point must be on a surface this class rests on
                    label = _label_below(stuff_map, stuff_names, px, py) if stuff else None
                    thing = _thing_at(things, px, py)
                    sup_ok, sup_why = point_support(label, thing, prompts["classes"][cat])
                    if not sup_ok:
                        rows.append(
                            {
                                "source_id": sid,
                                "category": cat,
                                "rank": rank,
                                "point": [round(px, 1), round(py, 1)],
                                "surface": label,
                                "on_object": thing,
                                "ok": False,
                                "reason": sup_why,
                                "prompt": prompt,
                                "answer": answer,
                                "vlm": vlm_name,
                            }
                        )
                        continue
                if box_mode:
                    box, info = list(vboxes[rank]), {"size_from": "vlm"}
                    fitted = box
                    area = box[2] * box[3]
                    min_side = ncfg.get("min_box_side_px", pcfg["min_side_px"])
                    if area < ncfg.get("min_box_frac", 0.0) * W * H or min(box[2:]) < min_side:
                        ok, reason = False, "too small"
                    elif area > pcfg["max_area_frac"] * W * H:
                        ok, reason = False, "too large"
                    else:
                        ok, reason = check_box(box, labels, blocked, rules)
                else:
                    box, info = box_from_point(
                        (px, py),
                        depth,
                        refs,
                        _typical_wh(stats, cat, W, H),
                        on_sky,
                        tuple(ncfg.get("size_ratio", (0.5, 2.0))),
                        pcfg["max_area_frac"],
                        ncfg.get("min_box_side_px", pcfg["min_side_px"]),
                        ncfg.get("min_box_frac", 0.0),
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
                    "vlm": vlm_name,
                } | info
                if ok:
                    row["valid_rank"] = len(valid)
                    valid.append(row)
                rows.append(row)
        rec = {
            "source_id": sid,
            "rows": rows[n_rows:],
            "plan": None,
            "failed": None,
            "settings": settings,
        }
        if not valid:
            rec["failed"] = {"source_id": sid, "reason": "no natural spot"}
            failed.append(rec["failed"])
        else:
            first = valid[0]
            used[first["category"]] += 1
            rec["plan"] = {
                "source_id": sid,
                "split": src["split"],
                "category": first["category"],
                "box_free": first["box"],
                "box_occ": None,
                "occluded_ann_id": None,
                "class_in_photo": True,
                "placement_rule": "natural",
            }
            plans.append(rec["plan"])
        progress.write(json.dumps(rec) + "\n")
        progress.flush()
        # free the photo's activations and Molmo's generation cache: on the 8 GB GPU, memory left
        # reserved between photos spilled into shared system memory (~80 s instead of ~15 s/photo)
        _free_cuda_cache()
    progress.close()
    points_file.close()
    with (wd / CANDIDATES).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    del vlm, depth_model
    _free_gpu()
    return {
        "plans": plans,
        "failed": failed,
        "candidates": rows,
        "vlm_questions": {"asked": n_asked, "reused": n_reused},
        "photos_rechecked": n_rechecked,
    }


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
