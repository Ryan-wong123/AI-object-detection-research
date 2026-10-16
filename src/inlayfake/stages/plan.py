"""Stage `plan`: one shared insertion plan per source → <work>/plans.jsonl."""

from __future__ import annotations

from collections import Counter

import numpy as np

from inlayfake.edit.placement import plan_source, size_stats
from inlayfake.edit.support import StuffIndex, category_supports
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_coco, read_jsonl, work_dir, write_jsonl


def run(cfg: dict) -> dict:
    index = load_coco(cfg)
    wd = work_dir(cfg)
    sources = read_jsonl(wd / "sources.jsonl")
    cats = list(cfg["sampling"]["categories"])
    pcfg = cfg["placement"]
    stats = size_stats(index, cats, pcfg["max_area_frac"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    scfg = pcfg.get("support", {})
    stuff = StuffIndex.load(scfg["stuff_annotations"]) if scfg.get("enabled") else None

    plans, failed = [], []
    for i, src in enumerate(sources):
        # balanced categories: rotate a shuffled list so each source starts at a different one
        start = i % len(cats)
        order = cats[start:] + cats[:start]
        supports_for = None
        if stuff is not None:

            def supports_for(cat, src=src):
                return category_supports(
                    src["source_id"], src["height"], src["width"], cat, stuff, scfg
                )

        p = plan_source(
            src, index.image_anns(src["source_id"]), order, stats, pcfg, rng, supports_for
        )
        if p is None:
            failed.append(src["source_id"])
        else:
            plans.append(p)
    write_jsonl(plans, wd / "plans.jsonl")
    summary = {
        "n_plans": len(plans),
        "n_failed": len(failed),
        "n_with_occlusion": sum(p["box_occ"] is not None for p in plans),
        "categories": dict(Counter(p["category"] for p in plans)),
    }
    if stuff is not None:
        summary["supports"] = dict(Counter(p["support"] for p in plans))
    write_run_info(wd / "runs", "plan", cfg, summary | {"failed_sources": failed})
    return summary
