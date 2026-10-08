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


def test_cli_help():
    assert main([]) == 0
