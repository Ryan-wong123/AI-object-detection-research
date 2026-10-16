"""Re-plan on failure: give a job a new box when every seed failed on the old one.

Each source has two box *tracks*, each with versions 0, 1, … ``edit.max_replans``:
  free  the C4 box, shared by C2 (sham) and C3 (paste) so the controls stay matched;
  occ   the C5 box (its own placement).
Version 0 is the original plan. A track moves to version n+1 when its *leader* job (C4 or C5 of
the first generator in ``generation.insert_generators`` / ``occlusion_generators``) failed
``edit.max_attempts`` seeds at version n. Followers (C2, C3, C4 of the other generators) are
redone at the leader's current version; a pass at an older version no longer counts for them.

A new box rests on one of the class's surfaces when COCO-Stuff labels are configured
(``placement.support.stuff_annotations``; free space otherwise) and is sized like one of the
photo's own instances of that class (×0.8–1.2), falling back to COCO size statistics. Versions
are stored in ``<work>/replans.jsonl`` (one row per source, track and version), so a re-run reuses
them. With ``edit.max_replans: 0`` (default) nothing changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inlayfake.edit.placement import box_area, place_free, place_occluding, sample_size, size_stats
from inlayfake.stages.common import derive_seed, read_jsonl

FREE, OCC = "free", "occ"


def leader(cfg: dict, track: str) -> tuple[str, str] | None:
    """(condition, generator) whose failures move the track to a new box."""
    roles = cfg["generation"]
    cond, gens = (
        ("C4", roles["insert_generators"])
        if track == FREE
        else (
            "C5",
            roles["occlusion_generators"],
        )
    )
    enabled = cfg.get("generators_enabled")
    gens = [g for g in gens if enabled is None or g in enabled]
    if cond not in cfg["conditions"] or not gens:
        return None
    return cond, gens[0]


def track_of(cond: str) -> str:
    return OCC if cond == "C5" else FREE


def rows_at(rows: list[dict], version: int) -> list[dict]:
    return [r for r in rows if r.get("version", 0) == version]


def track_version(cfg: dict, state: dict, sid: int, track: str) -> int:
    """Current box version of one track of one source."""
    lead = leader(cfg, track)
    if lead is None:
        return 0
    rows = state.get(f"{sid}|{lead[0]}|{lead[1]}", [])
    passed = [r for r in rows if r["passed"]]
    if passed:
        return passed[0].get("version", 0)
    v = max((r.get("version", 0) for r in rows), default=0)
    if len(rows_at(rows, v)) >= cfg["edit"]["max_attempts"] and v < cfg["edit"].get(
        "max_replans", 0
    ):
        v += 1
    return v


def next_job(
    cfg: dict, state: dict, sid: int, cond: str, gen: str, versions: dict | None = None
) -> tuple[int, int] | None:
    """(version, attempt) of the next image this job needs, or None (passed, out of tries, or
    no box could be placed for the current version)."""
    track = track_of(cond)
    v = track_version(cfg, state, sid, track)
    if v and versions is not None:
        row = versions.get((sid, track, v))
        if row is not None and row["box"] is None:
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


def effective_plan(plan: dict, versions: dict, v_free: int, v_occ: int) -> dict:
    """The plan with the boxes of the given versions (a box is None if none could be placed)."""
    p = dict(plan, free_version=v_free, occ_version=v_occ)
    sid = plan["source_id"]
    if v_free:
        row = versions.get((sid, FREE, v_free))
        if row is None or row["box"] is None:
            p["box_free"] = None
        else:
            p["box_free"] = row["box"]
    if v_occ:
        row = versions.get((sid, OCC, v_occ))
        if row is None or row["box"] is None:
            p["box_occ"], p["occluded_ann_id"] = None, None
        else:
            p["box_occ"], p["occluded_ann_id"] = row["box"], row["occluded_ann_id"]
    return p


def _size(cfg, index, anns, cat, W, H, rng) -> tuple[float, float]:
    pcfg = cfg["placement"]
    own = [
        a
        for a in anns
        if not a.get("iscrowd", 0) and index.cat_name.get(a.get("category_id")) == cat
    ]
    if own:
        x, y, w, h = own[int(rng.integers(len(own)))]["bbox"]
        s = float(rng.uniform(0.8, 1.2))
        w, h = max(w * s, pcfg["min_side_px"]), max(h * s, pcfg["min_side_px"])
        k = min(1.0, np.sqrt(pcfg["max_area_frac"] * W * H / (w * h)), 0.9 * W / w, 0.9 * H / h)
        return w * k, h * k
    stats = size_stats(index, [cat], pcfg["max_area_frac"])
    return sample_size(stats[cat], W, H, rng, pcfg["min_side_px"], pcfg["max_area_frac"])


def make_version(cfg, index, src: dict, plan: dict, track: str, version: int, stuff, extra) -> dict:
    """A new box for one track (None box if nothing fits). Deterministic per (source, track,
    version)."""
    from inlayfake.edit.support import category_supports

    sid, W, H = plan["source_id"], src["width"], src["height"]
    rng = np.random.default_rng(derive_seed(cfg["seed"], sid, "replan", track, version))
    pcfg = cfg["placement"]
    anns = index.image_anns(sid) + extra
    cat = plan["category"]
    supports = []
    if stuff is not None:
        supports = category_supports(sid, H, W, cat, stuff, pcfg["support"])
    row = {
        "source_id": sid,
        "track": track,
        "version": version,
        "box": None,
        "occluded_ann_id": None,
        "support": None,
    }
    for _ in range(5):  # a few sizes; each tried on every surface, then in free space
        w, h = _size(cfg, index, anns, cat, W, H, rng)
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
                targets = [
                    a
                    for a in anns
                    if not a.get("iscrowd", 0)
                    and box_area(a["bbox"]) >= (2 * pcfg["min_side_px"]) ** 2
                ]
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
    return row


def ensure_versions(cfg, index, sources, plans, state, wd: Path) -> dict:
    """Create the box versions the current state needs; returns all versions."""
    versions = load_versions(wd)
    if not cfg["edit"].get("max_replans", 0):
        return versions
    from inlayfake.stages.plan import audit_anns

    scfg = cfg["placement"].get("support", {})
    stuff = None
    new = []
    extra = audit_anns(cfg["placement"].get("audit_decisions"))
    for plan in plans:
        sid = plan["source_id"]
        for track in (FREE, OCC):
            if track == OCC and plan.get("box_occ") is None:
                continue  # C5 was never planned for this source
            v = track_version(cfg, state, sid, track)
            for ver in range(1, v + 1):
                if (sid, track, ver) in versions:
                    continue
                if (
                    stuff is None
                    and scfg.get("stuff_annotations")
                    and Path(scfg["stuff_annotations"]).exists()
                ):
                    from inlayfake.edit.support import StuffIndex

                    stuff = StuffIndex.load(scfg["stuff_annotations"])
                row = make_version(
                    cfg, index, sources[sid], plan, track, ver, stuff, extra.get(sid, [])
                )
                versions[(sid, track, ver)] = row
                new.append(row)
    if new:
        with (wd / "replans.jsonl").open("a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
    return versions
