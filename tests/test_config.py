from pathlib import Path

from inlayfake.cli import main
from inlayfake.config import config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_inherits_default():
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    assert cfg["sampling"]["n_sources"] == 50
    assert len(cfg["sampling"]["categories"]) == 10  # inherited, deep-merged
    assert cfg["encoding"]["quality"] == 95
    assert config_hash(cfg) != config_hash(load_config(ROOT / "configs" / "default.yaml"))


def test_full_run_uses_the_pilot_settings():
    """configs/insertion_inphoto.yaml = pilot_inphoto settings on all 600 sources, new work dir."""
    pilot = load_config(ROOT / "configs" / "pilot_inphoto.yaml")
    full = load_config(ROOT / "configs" / "insertion_inphoto.yaml")
    for key in ("edit", "verify", "generators_enabled", "encoding", "generation"):
        assert full[key] == pilot[key], key
    # placement: identical except the C5 fill added to the full run on 2026-10-18
    strip = lambda p: {k: v for k, v in p.items() if k != "occlusion_fill"}  # noqa: E731
    assert strip(full["placement"]) == strip(pilot["placement"])
    assert full["placement"]["occlusion_fill"]["enabled"] is True
    assert pilot["placement"]["occlusion_fill"]["enabled"] is False
    assert full["sampling"]["n_sources"] == 600 and pilot["sampling"]["n_sources"] == 50
    assert full["paths"]["work"] not in ("data/inlayfake", pilot["paths"]["work"])
    assert full["run_name"] == "task2_insertion"
    assert full["sampling"]["only_sources"] == "data/inlayfake/plans.jsonl"  # week-1 photos


def test_cli_help():
    assert main([]) == 0
