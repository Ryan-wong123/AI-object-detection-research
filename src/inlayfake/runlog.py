"""Run provenance (ARCHITECTURE.md §5): what code, config and environment produced an output."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path

from inlayfake.config import config_hash

PACKAGES = ("numpy", "pillow", "torch", "diffusers", "transformers", "ultralytics", "pycocotools")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name} ({p.total_memory / 2**30:.1f} GiB)"
    except ImportError:
        pass
    return None


def run_info(stage: str, cfg: dict) -> dict:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return {
        "stage": stage,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_hash": config_hash(cfg),
        "seed": cfg.get("seed"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": _gpu(),
        "packages": versions,
    }


def write_run_info(out_dir: str | Path, stage: str, cfg: dict, extra: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = run_info(stage, cfg) | (extra or {})
    path = out_dir / f"run_{stage}_{info['utc'].replace(':', '')}.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out_dir / f"config_{info['config_hash']}.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8"
    )
    return path
