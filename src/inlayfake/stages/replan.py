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
    intersection,
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

# Merged AI condition (``placement.merged.enabled``, author decision 2026-10-18): C4 and C5 become
# one condition, "an AI object of a class already in the photo, placed anywhere". Per photo the
# free track (C4 leader, C2/C3 followers) carries it:
#   a verified C4 (any version <= max_version) is kept;
#   else a verified C5 is adopted: free version ADOPT copies the C5 box and class, and C2/C3 are
#   made there (no new C4 is generated);
#   else versions ADOPT+1 .. ADOPT+placement.merged.versions place a box anywhere (no overlap
#   rule), cycling through the photo's classes.
# C5 is no longer generated. Followers (C2, C3) wait until the AI edit at their box is verified.
ADOPT = 100


def merged(cfg: dict) -> bool:
    return bool(cfg.get("placement", {}).get("merged", {}).get("enabled", False))


def _passed_upto(rows: list[dict] | None, top: int) -> list[dict]:
    return [r for r in rows or [] if r["passed"] and r.get("version", 0) <= top]


def adopted_c5(cfg: dict, state: dict, sid: int) -> dict | None:
    """The verify row of the first verified C5 of a photo (None if it has none)."""
    ok = _passed_upto(_leader_rows(cfg, state, sid, OCC), max_version(cfg))
    return ok[0] if ok else None


def merged_version(cfg: dict, state: dict, sid: int, versions: dict | None = None) -> int:
    """Free-track version of the merged AI condition (see ADOPT above)."""
    rows = _leader_rows(cfg, state, sid, FREE) or []
    old = _passed_upto(rows, max_version(cfg))
    if old:
        return old[0].get("version", 0)
    if adopted_c5(cfg, state, sid) is not None:
        return ADOPT
    new = [r for r in rows if r.get("version", 0) > ADOPT]
    ok = [r for r in new if r["passed"]]
    if ok:
        return ok[0]["version"]
    top = ADOPT + cfg["placement"]["merged"].get("versions", 6)
    v = max((r["version"] for r in new), default=ADOPT + 1)
    while v < top and (
        len(rows_at(new, v)) >= cfg["edit"]["max_attempts"] or _no_box(versions, sid, FREE, v)
    ):
        v += 1
    return v


def attempts_for(cfg: dict, cond: str, version: int) -> int:
    """Seeds a job may use at one box. Merged mode: C2 at a merged box may get more
    (``placement.merged.c2_attempts``, author 2026-10-18: 4 shams failed 3 seeds)."""
    if cond == "C2" and version >= ADOPT and merged(cfg):
        return cfg["placement"]["merged"].get("c2_attempts") or cfg["edit"]["max_attempts"]
    return cfg["edit"]["max_attempts"]


def leader_passed(cfg: dict, state: dict, sid: int, v: int) -> bool:
    """Merged mode: is the AI edit at free version ``v`` verified (C2/C3 may follow)?"""
    if v == ADOPT:
        return True
    return any(r["passed"] for r in rows_at(_leader_rows(cfg, state, sid, FREE) or [], v))


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

    Versions with a class that is not in the photo (``max_outside_classes``) are *tried* only if
    the photo has no verified AI object of its own classes: this track is exhausted within its
    in-photo versions, and the other track has no in-photo pass and is not still trying.
    Once an edit has passed, at any version, it is kept (author decision 2026-10-18): the rule
    decides what to try next, it never withdraws a verified edit. Before that change, 13 C4
    edits with an outside class stopped counting when the C5 fill later gave their photos an
    in-photo C5.
    """
    if merged(cfg):
        if track == FREE:
            return merged_version(cfg, state, sid, versions)
        ok = adopted_c5(cfg, state, sid)  # C5 is frozen: its verified version, if any
        return ok.get("version", 0) if ok else 0
    rows = _leader_rows(cfg, state, sid, track)
    if rows is None:
        return 0
    e = cfg["edit"]
    passed = [r for r in rows if r["passed"] and r.get("version", 0) <= max_version(cfg)]
    if passed:  # a verified edit stays in the benchmark
        return passed[0].get("version", 0)
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
    if merged(cfg) and cond == "C5":
        return None  # merged condition: C5 is no longer generated
    v = track_version(cfg, state, sid, track, versions)
    if _no_box(versions, sid, track, v):
        return None
    if merged(cfg) and v >= ADOPT:
        if cond == "C4" and v == ADOPT:
            return None  # the verified C5 is the photo's AI edit
        if cond != "C4" and not leader_passed(cfg, state, sid, v):
            return None  # C2 waits for a verified AI edit at this box
    rows = rows_at(state.get(f"{sid}|{cond}|{gen}", []), v)
    if any(r["passed"] for r in rows) or len(rows) >= attempts_for(cfg, cond, v):
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
        placement_rule_free="standard",
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
            p["placement_rule_free"] = row.get("placement_rule", "standard")
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
    # a few sizes, alternating the photo's own instance size and a typical COCO size (the size
    # the original planner uses): in crowded photos the own instances can be too big for any free
    # spot, which left 7 C4 edits without a new box in the full run (2026-10-18); each size is
    # tried on every surface, then in free space
    for i in range(6):
        try:
            if i % 2 == 0:
                w, h = _size(cfg, index, anns, cat, W, H, rng)
            else:
                st = size_stats(index, [cat], pcfg["max_area_frac"])[cat]
                w, h = sample_size(st, W, H, rng, pcfg["min_side_px"], pcfg["max_area_frac"])
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


def merged_classes(cfg, index, plan: dict, extra: list[dict]) -> tuple[list[str], bool]:
    """Merged condition: (classes, in_photo). The photo's own classes (audited GT, never person),
    the free track's class first; the plan's class alone (in_photo False) if the photo has none
    it may use."""
    from inlayfake.stages.plan import present_classes

    first = track_plan(plan, FREE)["category"]
    exclude = set(cfg["placement"].get("exclude_classes", ["person"]))
    own = present_classes(index, index.image_anns(plan["source_id"]), extra, exclude)
    if not own:
        return [first], False
    return ([first] if first in own else []) + [c for c in own if c != first], True


def make_merged_version(cfg, index, src: dict, plan: dict, version: int, state, versions, extra):
    """Free-track box of the merged condition: ADOPT copies the verified C5's box and class;
    later versions place a box of one of the photo's classes anywhere (no overlap rule)."""
    from inlayfake.edit.placement import place_anywhere

    sid, W, H = plan["source_id"], src["width"], src["height"]
    row = {"source_id": sid, "track": FREE, "version": version, "support": None}
    if version == ADOPT:
        c5 = adopted_c5(cfg, state, sid)
        p = effective_plan(plan, versions, 0, c5.get("version", 0))
        return row | {
            "category": p["category_occ"],
            "class_in_photo": p["class_in_photo_occ"],
            "box": p["box_occ"],
            "occluded_ann_id": p["occluded_ann_id"],
            "placement_rule": "adopted_c5",
            "adopted_image_id": c5["image_id"],
        }
    k = version - ADOPT - 1
    classes, in_photo = merged_classes(cfg, index, plan, extra)
    cat = classes[k % len(classes)]
    rng = np.random.default_rng(derive_seed(cfg["seed"], sid, "merged", version))
    pcfg = cfg["placement"]
    anns = index.image_anns(sid) + extra
    row |= {"category": cat, "class_in_photo": in_photo, "box": None, "occluded_ann_id": None}
    try:  # alternate the photo's own instance size and a typical COCO size
        if k % 2 == 0:
            w, h = _size(cfg, index, anns, cat, W, H, rng)
        else:
            st = size_stats(index, [cat], pcfg["max_area_frac"])[cat]
            w, h = sample_size(st, W, H, rng, pcfg["min_side_px"], pcfg["max_area_frac"])
    except ValueError:  # no usable size statistics for this class: version skipped
        return row
    gt = index.image_anns(sid)
    box, occ_id = place_anywhere(W, H, w, h, gt, rng)
    start = pcfg["merged"].get("avoid_same_class_from")
    if start is not None and k + 1 >= start:
        # author, 2026-10-18 (re-run of the 7 dropped photos): 59 of their 210 failed attempts
        # had the box on the photo's real object of the same class, which the verifier can never
        # accept (a detection on a real object does not count). Still anywhere in the image, but
        # the box may cover at most 30% of its area with a real object of the same class; it
        # shrinks (x0.8) if no such spot is found, else the plain anywhere box is kept.
        same = [
            a["bbox"]
            for a in gt
            if not a.get("iscrowd", 0) and index.cat_name.get(a.get("category_id")) == cat
        ]
        min_side = pcfg["min_side_px"]
        for _ in range(6):
            found = None
            for _ in range(200):
                b, o = place_anywhere(W, H, w, h, gt, rng)
                if all(intersection(b, s) <= 0.3 * b[2] * b[3] for s in same):
                    found = (b, o)
                    break
            if found:
                box, occ_id = found
                break
            if min(w, h) * 0.8 < min_side:
                break
            w, h = w * 0.8, h * 0.8
    return row | {
        "box": [float(v) for v in box],
        "occluded_ann_id": occ_id,
        "placement_rule": "anywhere",
    }


def _ensure_merged(cfg, index, sources, plans, state, wd: Path, versions: dict) -> dict:
    from inlayfake.stages.plan import audit_anns

    extra = audit_anns(cfg["placement"].get("audit_decisions"))
    new = []
    for plan in plans:
        sid = plan["source_id"]
        while True:  # a version without a box moves the track on: create until stable
            v = merged_version(cfg, state, sid, versions)
            if v < ADOPT:
                break
            want = [ADOPT] if v == ADOPT else range(ADOPT + 1, v + 1)
            missing = [ver for ver in want if (sid, FREE, ver) not in versions]
            if not missing:
                break
            for ver in missing:
                row = make_merged_version(
                    cfg, index, sources[sid], plan, ver, state, versions, extra.get(sid, [])
                )
                versions[(sid, FREE, ver)] = row
                new.append(row)
    if new:
        with (wd / "replans.jsonl").open("a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
    return versions


def ensure_versions(cfg, index, sources, plans, state, wd: Path) -> dict:
    """Create the box versions the current state needs; returns all versions."""
    versions = load_versions(wd)
    if merged(cfg):
        return _ensure_merged(cfg, index, sources, plans, state, wd, versions)
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


def merged_source(cfg: dict, state: dict, sid: int, versions: dict | None = None) -> str | None:
    """Merged condition: "C4", "C5" or "anywhere" for a photo with a verified AI edit, else None."""
    v = merged_version(cfg, state, sid, versions)
    if v == ADOPT:
        return "C5"
    if not leader_passed(cfg, state, sid, v):
        return None
    return "anywhere" if v > ADOPT else "C4"


def gen_of(cfg: dict, cond: str) -> str | None:
    """First enabled generator of a condition's role."""
    key = {"C2": "sham_generators", "C4": "insert_generators", "C5": "occlusion_generators"}[cond]
    enabled = cfg.get("generators_enabled")
    gens = [g for g in cfg["generation"][key] if enabled is None or g in enabled]
    return gens[0] if gens else None


def merged_pending(cfg: dict, state: dict, plans: list[dict], versions: dict) -> int:
    """Merged mode: AI edits and C2 shams `generate` would still make (C2 waits for its AI
    edit, so it may have no verify rows yet)."""
    n = 0
    for p in plans:
        for cond in ("C2", "C4"):
            g = gen_of(cfg, cond)
            if g and cond in cfg["conditions"]:
                n += next_job(cfg, state, p["source_id"], cond, g, versions) is not None
    return n
