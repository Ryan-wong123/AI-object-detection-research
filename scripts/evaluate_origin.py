"""H3 (forensic origin) and H5 (VLM claims) tables only, without the ~50-min AP bootstrap of a
full `evaluate` (2026-10-26). Same functions and settings as `evaluate`, which also writes these
tables in its final run.

    python scripts/evaluate_origin.py configs/task3_detection_natural.yaml

Writes runs/<run>/tables/{h3_origin,vlm_objects,vlm_images,vlm_object_rows}.csv and
runs/<run>/summary_origin.md.
"""

from __future__ import annotations

import sys

import numpy as np

from inlayfake.cli import full_speed_in_background
from inlayfake.config import load_config
from inlayfake.eval.forensic_origin import h3_tables
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_records, run_dir
from inlayfake.stages.evaluate import _h3_md, _vlm_md, _write_csv, vlm_tables


def main() -> int:
    cfg = load_config(sys.argv[1])
    full_speed_in_background()
    rng = np.random.default_rng(cfg["seed"])
    rd = run_dir(cfg)
    fcfg = cfg["forensics"]
    methods = list(fcfg["handcrafted"]) + list(fcfg.get("run_learned") or [])
    tables = {
        "h3_origin": h3_tables(rd / "forensics", methods, cfg["eval"]["bootstrap_resamples"], rng)
    }
    tables |= vlm_tables(cfg, load_records(cfg), rng)
    lines = [f"# Origin summary (H3, H5): run `{cfg.get('run_name')}`"]
    for key, rows in tables.items():
        _write_csv(rows, rd / "tables" / f"{key}.csv")
    if tables["h3_origin"]:
        lines += _h3_md(tables["h3_origin"])
    if "vlm_objects" in tables:
        lines += _vlm_md(tables)
    (rd / "summary_origin.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_info(rd, "evaluate_origin", cfg, {k: len(v) for k, v in tables.items()})
    print(f"wrote {rd / 'summary_origin.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
