"""Re-plan on failure: give a job a new box, then another class, when every seed failed.

Each source has two box *tracks*, each with versions 0, 1, 2, â€¦:
  free  the C4 box, shared by C2 (sham) and C3 (paste) so the controls stay matched;
  occ   the C5 box (its own placement).
Version 0 is the original plan. A track moves to version n+1 when its *leader* job (C4 or C5 of
the first generator in ``generation.insert_generators`` / ``occlusion_generators``) failed
``edit.max_attempts`` seeds at version n. Followers (C2, C3, C4 of the other generators) are
redone at the leader's current version; a pass at an older version no longer counts for them.

Versions 1 â€¦ ``edit.max_replans`` keep the planned class and only move the box. Versions after
that (up to ``edit.max_class_switches`` more) switch to **another class already in the photo**
(``placement.class_choice: in_photo`` rules: audited GT, never person), smallest first, one class
per version. A version where no box fits is skipped. The track's class is stored per version
(``category``); C2/C3 follow the free track's class, C5 uses the occ track's class.

A new box rests on one of the class's surfaces when COCO-Stuff labels are available
(``placement.support.stuff_annotations``; free space otherwise) and is sized like one of the
photo's own instances of that class (Ã—0.8â€“1.2), falling back to COCO size statistics. Versions
are stored in ``<work>/replans.jsonl`` (one row per source, track and version), so a re-run reuses
them. With ``edit.max_replans: 0`` and ``max_class_switches: 0`` (default) nothing changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inlayfake.edit.placement import (
    occluding_sized_to_target,
    occlusion_min_side,
    occlusion_targets,
    place_free,
    place_occluding,
    sample_size,
    size_stats,
)
from inlayfake.stages.common import derive_seed, read_jsonl

FREE, OCC = "free", "occ"


def leader(cfg: dict, track: str) -> tuple[str, str] | None:
    """(condition, generator) whose failures move the track to a new box."""
    roles = cfg["generation"]
    if track == FREE:
        cond, gens = "C4", roles["insert_generators"]
    else:
        cond, gens = "C5", roles["occlusion_generators"]
    enabled = cfg.get("generators_enabled")
    gens = [g for g in gens if enabled is None or g in enabled]
    if cond not in cfg["conditions"] or not gens:
        return None
    return cond, gens[0]


def track_of(cond: str) -> str:
    return OCC if cond == "C5" else FREE


def rows_at(rows: list[dict], version: int) -> list[dict]:
    return [r for r in rows if r.get("version", 0) == version]


def max_version(cfg: dict) -> int:
    e = cfg["edit"]
    return (
        e.get("max_replans", 0) + e.get("max_class_switches", 0) + e.get("max_outside_classes", 0)
    )


def _no_box(versions: dict | None, sid: int, track: str, v: int) -> bool:
    if not v or versions is None:
        return False
    row = versions.get((sid, track, v))
    return row is not None and row["box"] is None


def _core_version(cfg, rows, sid, track, versions, top) -> int:
    """Version of a track counting only versions â‰¤ ``top``."""
    rows = [r for r in rows if r.get("version", 0) <= top]
    passed = [r for r in rows if r["passed"]]
    if passed:
        return passed[0].get("version", 0)
    v = max((r.get("version", 0) for r in rows), default=0)
    if len(rows_at(rows, v)) >= cfg["edit"]["max_attempts"] and v < top:
        v += 1
    while _no_box(versions, sid, track, v) and v < top:
        v += 1
    return v


def _exhausted(cfg, rows, sid, track, versions, top) -> bool:
    """True if the track has no pass up to ``top`` and cannot go on within it."""
    if any(r["passed"] for r in rows if r.get("version", 0) <= top):
        return False
    v = _core_version(cfg, rows, sid, track, versions, top)
    out_of_tries = len(rows_at(rows, v)) >= cfg["edit"]["max_attempts"]
    return v == top and (out_of_tries or _no_box(versions, sid, track, v))


def _leader_rows(cfg, state, sid, track) -> list[dict] | None:
    lead = leader(cfg, track)
    return None if lead is None else state.get(f"{sid}|{lead[0]}|{lead[1]}", [])


def track_version(
    cfg: dict, state: dict, sid: int, track: str, versions: dict | None = None
) -> int:
    """Current box version of one track of one source (versions without a box are skipped).

    Versions with a class that is not in the photo (``max_outside_classes``) are used only if
    the photo has no verified AI object of its own classes: this track is exhausted within its
    in-photo versions, and the other track has no in-photo pass and is not still trying.
    """
    rows = _leader_rows(cfg, state, sid, track)
    if rows is None:
        return 0
    e = cfg["edit"]
    top_in = e.get("max_replans", 0) + e.get("max_class_switches", 0)
    v_in = _core_version(cfg, rows, sid, track, versions, top_in)
    if not e.get("max_outside_classes", 0) or not _exhausted(
        cfg, rows, sid, track, versions, top_in
    ):
        return v_in
    other = OCC if track == FREE else FREE
    orows = _leader_rows(cfg, state, sid, other) or []
    if any(r["passed"] for r in orows if r.get("version", 0) <= top_in):
        return v_in  # the photo already has an AI object of its own class
    if orows and not _exhausted(cfg, orows, sid, other, versions, top_in):
        return v_in  # the other edit may still succeed with the photo's own classes
    return _core_version(cfg, rows, sid, track, versions, max_version(cfg))


def next_job(
    cfg: dict, state: dict, sid: int, cond: str, gen: str, versions: dict | None = None
) -> tuple[int, int] | None:
    """(version, attempt) of the next image this job needs, or None (passed, out of tries, or
    no box could be placed for the last version)."""
    track = track_of(cond)
    v = track_version(cfg, state, sid, track, versions)
    if _no_box(versions, sid, track, v):
        return None
    rows = rows_at(state.get(f"{sid}|{cond}|{gen}", []), v)
    if any(r["passed"] for r in rows) or len(rows) >= cfg["edit"]["max_attempts"]:
        return None
    return v, len(rows)


def job_seed(cfg: dict, sid: int, cond: str, gen: str | None, attempt: int, version: int) -> int:
    """Version 0 keeps the original seeds, so existing images are reused unchanged."""
    if cond == "C3":
        parts = (
            (cfg["seed"], sid, "C3") if version == 0 else (cfg["seed"], sid, "C3", f"v{version}")
        )
    else:
        parts = (cfg["seed"], sid, cond, gen, attempt)
        if version:
            parts = (*parts, f"v{version}")
    return derive_seed(*parts)


# ---------------------------------------------------------------- box versions
def load_versions(wd: Path) -> dict[tuple[int, str, int], dict]:
    p = wd / "replans.jsonl"
    rows = read_jsonl(p) if p.exists() else []
    return {(r["source_id"], r["track"], r["version"]): r for r in rows}


def track_plan(plan: dict, track: str) -> dict:
    """The plan as seen by one track. The C4 fill (``placement.free_fill``) may give the free
    track (C2, C3, C4) another class of the photo (``free_category``); C5 keeps the plan class."""
    if track == FREE and plan.get("free_category"):
        return dict(plan, category=plan["free_category"], class_in_photo=True)
    return plan


def effective_plan(plan: dict, versions: dict, v_free: int, v_occ: int) -> dict:
    """The plan with the boxes and classes of the given versions (box None if none fits).
    ``category`` is the free track's class (C2, C3, C4), ``category_occ`` the C5 class."""
    base_in = plan.get("class_in_photo", True)
    free_plan = track_plan(plan, FREE)  # the C4 fill may have chosen another class for C2-C4
    free_in = free_plan.get("class_in_photo", True)
    p = dict(
        plan,
        free_version=v_free,
        occ_version=v_occ,
        category=free_plan["category"],
        category_occ=plan["category"],
        class_in_photo=free_in,
        class_in_photo_occ=base_in,
        placement_rule_occ=plan.get("placement_rule", "standard"),
    )
    sid = plan["source_id"]
    if v_free:
        row = versions.get((sid, FREE, v_free))
        if row is None or row["box"] is None:
            p["box_free"] = None
        else:
            p["box_free"] = row["box"]
            p["category"] = row.get("category") or free_plan["category"]
            p["class_in_photo"] = row.get("class_in_photo", free_in)
    if v_occ:
        row = versions.get((sid, OCC, v_occ))
        if row is None or row["box"] is None:
            p["box_occ"], p["occluded_ann_id"] = None, None
        else:
            p["box_occ"], p["occluded_ann_id"] = row["box"], row["occluded_ann_id"]
            p["category_occ"] = row.get("category") or plan["category"]
            p["class_in_photo_occ"] = row.get("class_in_photo", base_in)
            p["placement_rule_occ"] = row.get("placement_rule", "standard")
    return p


def _own(index, anns: list[dict], cat: str) -> list[dict]:
    return [
        a
        for a in anns
        if not a.get("iscrowd", 0) and index.cat_name.get(a.get("category_id")) == cat
    ]


def _size(cfg, index, anns, cat, W, H, rng) -> tuple[float, float]:
    pcfg = cfg["placement"]
    own = _own(index, anns, cat)
    if own:
        x, y, w, h = own[int(rng.integers(len(own)))]["bbox"]
        s = float(rng.uniform(0.8, 1.2))
        w, h = max(w * s, pcfg["min_side_px"]), max(h * s, pcfg["min_side_px"])
        k = min(1.0, np.sqrt(pcfg["max_area_frac"] * W * H / (w * h)), 0.9 * W / w, 0.9 * H / h)
        return w * k, h * k
    stats = size_stats(index, [cat], pcfg["max_area_frac"])
    return sample_size(stats[cat], W, H, rng, pcfg["min_side_px"], pcfg["max_area_frac"])


def alternate_classes(cfg, index, sid: int, base: str, extra: list[dict]) -> list[str]:
    """Other classes already in the photo (same rules as the in_photo plan), smallest first by
    the median box area of their instances in this photo."""
    from inlayfake.stages.plan import present_classes

    exclude = set(cfg["placement"].get("exclude_classes", ["person"]))
    anns = index.image_anns(sid)
    classes = [c for c in present_classes(index, anns, extra, exclude) if c != base]

    def area(c: str) -> float:
        boxes = [a["bbox"] for a in _own(index, anns, c)]
        boxes += [e["bbox"] for e in extra if e["category"] == c and e["decision"] == "add"]
        return float(np.median([b[2] * b[3] for b in boxes])) if boxes else float("inf")

    return sorted(classes, key=lambda c: (area(c), c))


def outside_classes(cfg, index, sid: int, extra: list[dict], seed_parts=()) -> list[str]:
    """Last-resort classes that are NOT in the photo: ``placement.outside_classes`` (default:
    the 10 original benchmark categories), minus the photo's classes and excluded ones, in a
    seeded order per source."""
    from inlayfake.stages.plan import present_classes

    pcfg = cfg["placement"]
    pool = list(pcfg.get("outside_classes") or cfg["sampling"]["categories"])
    exclude = set(pcfg.get("exclude_classes", ["person"]))
    own = set(present_classes(index, index.image_anns(sid), extra, set()))
    pool = [c for c in pool if c not in own and c not in exclude]
    rng = np.random.default_rng(derive_seed(cfg["seed"], sid, "outside", *seed_parts))
    return [pool[i] for i in rng.permutation(len(pool))]


def version_class(
    cfg, index, plan: dict, version: int, extra: list[dict]
) -> tuple[str | None, bool]:
    """(class, class_in_photo) of a version. Versions 1..max_replans keep the planned class;
    the next max_class_switches use other classes in the photo; the last max_outside_classes
    use classes that are not in the photo."""
    e = cfg["edit"]
    k = version - e.get("max_replans", 0) - 1
    base_in = plan.get("class_in_photo", True)
    if k < 0:
        return plan["category"], base_in
    if k < e.get("max_class_switches", 0):
        alts = alternate_classes(cfg, index, plan["source_id"], plan["category"], extra)
        return (alts[k], True) if k < len(alts) else (None, True)
    j = k - e.get("max_class_switches", 0)
    if j >= e.get("max_outside_classes", 0):
        return None, False
    outs = [
        c for c in outside_classes(cfg, index, plan["source_id"], extra) if c != plan["category"]
    ]
    return (outs[j], False) if j < len(outs) else (None, False)


def make_version(cfg, index, src: dict, plan: dict, track: str, version: int, stuff, extra) -> dict:
    """A new box (and class) for one track; box None if nothing fits. Deterministic per
    (source, track, version)."""
    from inlayfake.edit.support import category_supports

    sid, W, H = plan["source_id"], src["width"], src["height"]
    rng = np.random.default_rng(derive_seed(cfg["seed"], sid, "replan", track, version))
    pcfg = cfg["placement"]
    anns = index.image_anns(sid) + extra
    cat, in_photo = version_class(cfg, index, plan, version, extra)
    row = {
        "source_id": sid,
        "track": track,
        "version": version,
        "category": cat,
        "class_in_photo": in_photo,
        "box": None,
        "occluded_ann_id": None,
        "support": None,
    }
    if cat is None:  # no other class left in the photo
        return row
    supports = []
    if stuff is not None:
        supports = category_supports(sid, H, W, cat, stuff, pcfg["support"])
    for _ in range(5):  # a few sizes; each tried on every surface, then in free space
        try:
            w, h = _size(cfg, index, anns, cat, W, H, rng)
        except ValueError:  # no usable size statistics for this class
            return row
        for sup in [*supports, None]:
            if track == FREE:
                box = place_free(
                    W,
                    H,
                    w,
                    h,
                    [a["bbox"] for a in anns],
                    pcfg["margin_px"],
                    rng,
                    pcfg["max_tries"],
                    support=sup,
                )
                occ_id = None
            else:
                targets = occlusion_targets(anns, occlusion_min_side(pcfg))
                box, occ_id = None, None
                for t in rng.permutation(len(targets)) if targets else []:
                    tgt = targets[t]
                    others = [a["bbox"] for a in anns if a["id"] != tgt["id"]]
                    box = place_occluding(
                        W,
                        H,
                        w,
                        h,
                        tgt["bbox"],
                        others,
                        tuple(pcfg["occlusion_frac"]),
                        rng,
                        pcfg["max_tries"],
                        support=sup,
                    )
                    if box is not None:
                        occ_id = tgt["id"]
                        break
            if box is not None:
                row.update(
                    box=[float(v) for v in box],
                    occluded_ann_id=occ_id,
                    support=sup.name if sup else None,
                )
                return row
    if track == OCC:  # last try: size the box from the object it must cover
        targets = occlusion_targets(anns, occlusion_min_side(pcfg))
        own = _own(index, anns, cat)
        if own:
            aspect = float(np.median([a["bbox"][2] / a["bbox"][3] for a in own]))
        else:
            aspect = float(np.median(size_stats(index, [cat], pcfg["max_area_frac"])[cat][:, 1]))
        for sup in [*supports, None]:
            box, occ_id = occluding_sized_to_target(W, H, aspect, targets, anns, pcfg, rng, sup)
            if box is not None:
                row.update(
                    box=[float(v) for v in box],
                    occluded_ann_id=occ_id,
                    support=sup.name if sup else None,
                )
                return row
        if pcfg.get("allow_anywhere", False):  # very last resort: no overlap rule at all
            from inlayfake.edit.placement import place_anywhere

            w, h = _size(cfg, index, anns, cat, W, H, rng)
            box, occ_id = place_anywhere(W, H, w, h, index.image_anns(sid), rng)
            row.update(box=box, occluded_ann_id=occ_id, placement_rule="anywhere")
    return row


def ensure_versions(cfg, index, sources, plans, state, wd: Path) -> dict:
    """Create the box versions the current state needs; returns all versions."""
    versions = load_versions(wd)
    if not max_version(cfg):
        return versions
    from inlayfake.stages.plan import audit_anns

    scfg = cfg["placement"].get("support", {})
    stuff = None
    new = []
    extra = audit_anns(cfg["placement"].get("audit_decisions"))
    for plan in plans:
        sid = plan["source_id"]
        for track in (FREE, OCC):
            if (track == OCC and plan.get("box_occ") is None) or (
                track == FREE and plan.get("box_free") is None
            ):
                continue  # never planned for this source
            while True:  # a version without a box moves the track on: create until stable
                v = track_version(cfg, state, sid, track, versions)
                missing = [ver for ver in range(1, v + 1) if (sid, track, ver) not in versions]
                if not missing:
                    break
                if (
                    stuff is None
                    and scfg.get("stuff_annotations")
                    and Path(scfg["stuff_annotations"]).exists()
                ):
                    from inlayfake.edit.support import StuffIndex

                    stuff = StuffIndex.load(scfg["stuff_annotations"])
                for ver in missing:
                    row = make_version(
                        cfg,
                        index,
                        sources[sid],
                        track_plan(plan, track),
                        track,
                        ver,
                        stuff,
                        extra.get(sid, []),
                    )
                    versions[(sid, track, ver)] = row
                    new.append(row)
    if new:
        with (wd / "replans.jsonl").open("a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
    return versions
