"""Run pipeline stages one after another, resumably (author request, 2026-10-21).

    python scripts/run_stages.py configs/insertion_natural_box.yaml

Runs sample -> plan -> insert -> preview, each as its own process (so one model's GPU memory is
freed before the next stage), and asks Windows not to sleep while it runs (closing the lid or
shutting down still stops it). Every stage skips work that is already saved, so after a
shutdown, crash or power cut the same command continues where it stopped:

- plan: Qwen's answers (natural_points.jsonl) and every finished photo
  (natural_plan_progress.jsonl) are saved as they are made; finished photos are replayed.
- insert: an image counts as made only once its record is written (after the image file, in one
  step); verification results are logged per image; a half-written last log line is dropped
  and that item redone.

Progress: <work>/run_stages.log (which stage, start/end) and <work>/run_progress.log (bars).
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

from inlayfake.config import load_config

STAGES = (["sample"], ["plan"], ["insert"], ["preview", "--limit", "100000"])


def main() -> int:
    cfg_path = sys.argv[1]
    wd = Path(load_config(cfg_path)["paths"]["work"])
    wd.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while running
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    log = (wd / "run_stages.log").open("a", encoding="utf-8")
    with (wd / "run_progress.log").open("a", encoding="utf-8") as progress:
        for stage in STAGES:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start {stage[0]}\n")
            log.flush()
            cmd = [sys.executable, "-m", "inlayfake", stage[0], "--config", cfg_path, *stage[1:]]
            code = subprocess.run(cmd, stdout=log, stderr=progress).returncode
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} end {stage[0]} (exit {code})\n")
            log.flush()
            if code != 0:
                log.write("stopped: fix the error, then run the same command again\n")
                return code
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} all stages done\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
