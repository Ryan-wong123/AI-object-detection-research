"""Stage `sample`: choose source photos from COCO val2017 → <work>/sources.jsonl."""

from __future__ import annotations

from collections import Counter

import numpy as np

from inlayfake.data.sampling import sample_sources
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_coco, work_dir, write_jsonl


def run(cfg: dict) -> dict:
    index = load_coco(cfg)
    rng = np.random.default_rng(cfg["seed"])
    sources = sample_sources(index, cfg["sampling"], rng)
    out = work_dir(cfg) / "sources.jsonl"
    write_jsonl(sources, out)
    summary = {"n_sources": len(sources), "splits": dict(Counter(s["split"] for s in sources))}
    write_run_info(work_dir(cfg) / "runs", "sample", cfg, summary)
    return summary
