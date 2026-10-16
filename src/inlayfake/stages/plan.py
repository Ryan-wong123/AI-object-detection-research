"""Stage `plan`: one shared insertion plan per source → <work>/plans.jsonl.

Which class to insert (``placement.class_choice``):
  fixed     cycle through ``sampling.categories`` (balanced over the benchmark; the original
            design, used by week 1);
  in_photo  a class that is already in the photo's audited ground truth (COCO labels plus
            week-1 audit "add" rows), never one in ``placement.exclude_classes`` (person). The
            least-used such class so far is tried first, so classes stay as balanced as the
            photos allow. Every edited photo then holds a photographed and an AI object of the
            same class.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from inlayfake.data.audit import read_decisions
from inlayfake.edit.placement import plan_source, size_stats
from inlayfake.edit.support import StuffIndex, category_supports
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_coco, read_jsonl, work_dir, write_jsonl


def audit_anns(path: str | None) -> dict[int, list[dict]]:
    """Week-1 audit rows as annotation-like dicts per source: "add" rows are real objects, and
    ignore / "?" rows are unresolved regions. Both block free placement like crowd regions."""
    out: dict[int, list[dict]] = {}
    for d in read_decisions(Path(path)) if path else []:
        if d["decision"] == "reject":
            continue
        out.setdefault(int(d["source_id"]), []).append(
            {
                "id": f"audit:{d['candidate_id']}",
                "category": d["category"],
                "decision": d["decision"],
                "bbox": [float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"])],
                "iscrowd": 1,  # blocks placement, never an occlusion target
            }
        )
    return out


def present_classes(index, anns: list[dict], extra: list[dict], exclude: set[str]) -> list[str]:
    """Sorted non-excluded classes with a non-crowd GT instance or an audit "add" in the photo."""
    names = {index.cat_name[a["category_id"]] for a in anns if not a.get("iscrowd", 0)}
    names |= {e["category"] for e in extra if e["decision"] == "add"}
    return sorted(names - exclude)


def balanced_order(classes: list[str], used: Counter, rng: np.random.Generator) -> list[str]:
    """Least-used classes first; ties in a seeded random order."""
    shuffled = [classes[i] for i in rng.permutation(len(classes))]
    return sorted(shuffled, key=lambda c: used[c])


def run(cfg: dict) -> dict:
    index = load_coco(cfg)
    wd = work_dir(cfg)
    sources = read_jsonl(wd / "sources.jsonl")
    pcfg = cfg["placement"]
    in_photo = pcfg.get("class_choice", "fixed") == "in_photo"
    exclude = set(pcfg.get("exclude_classes", ["person"]))
    extra = audit_anns(pcfg.get("audit_decisions")) if in_photo else {}
    rng = np.random.default_rng(cfg["seed"] + 1)
    scfg = pcfg.get("support", {})
    stuff = StuffIndex.load(scfg["stuff_annotations"]) if scfg.get("enabled") else None

    if in_photo:
        wanted = set()
        for src in sources:
            sid = src["source_id"]
            wanted.update(
                present_classes(index, index.image_anns(sid), extra.get(sid, []), exclude)
            )
        stats = {}
        for cat in sorted(wanted):
            try:
                stats.update(size_stats(index, [cat], pcfg["max_area_frac"]))
            except ValueError:  # no COCO instance of a usable size: cannot be inserted
                pass
        cats = []
    else:
        cats = list(cfg["sampling"]["categories"])
        stats = size_stats(index, cats, pcfg["max_area_frac"])

    plans, failed, no_class = [], [], []
    used: Counter = Counter()
    for i, src in enumerate(sources):
        sid = src["source_id"]
        anns = index.image_anns(sid) + extra.get(sid, [])
        if in_photo:
            own = present_classes(index, index.image_anns(sid), extra.get(sid, []), exclude)
            classes = [c for c in own if c in stats]
            if not classes:
                no_class.append(sid)
                failed.append(sid)
                continue
            order = balanced_order(classes, used, rng)
        else:
            # balanced categories: rotate the list so each source starts at a different one
            start = i % len(cats)
            order = cats[start:] + cats[:start]
        supports_for = None
        if stuff is not None:

            def supports_for(cat, src=src):
                return category_supports(
                    src["source_id"], src["height"], src["width"], cat, stuff, scfg
                )

        p = plan_source(src, anns, order, stats, pcfg, rng, supports_for)
        if p is None:
            failed.append(sid)
        else:
            plans.append(p)
            used[p["category"]] += 1
    write_jsonl(plans, wd / "plans.jsonl")
    summary = {
        "n_plans": len(plans),
        "n_failed": len(failed),
        "n_with_occlusion": sum(p["box_occ"] is not None for p in plans),
        "categories": dict(Counter(p["category"] for p in plans)),
    }
    if in_photo:
        summary["n_no_insertable_class"] = len(no_class)
    if stuff is not None:
        summary["supports"] = dict(Counter(p["support"] for p in plans))
    write_run_info(wd / "runs", "plan", cfg, summary | {"failed_sources": failed})
    return summary
