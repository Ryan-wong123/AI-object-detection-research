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
    for key in ("edit", "generators_enabled", "encoding", "generation"):
        assert full[key] == pilot[key], key
    # verify: identical except the sham rule and full C4 audit added to the full run (2026-10-18)
    added = ("sham_original_overlap", "audit_all_conditions", "audit_only_conditions")
    full_v = {k: v for k, v in full["verify"].items() if k not in added}
    assert full_v == {k: v for k, v in pilot["verify"].items() if k not in added}
    # placement: identical except the C5 fill added to the full run on 2026-10-18
    # fills and the merged AI condition: added to the full run on 2026-10-18
    fills = ("occlusion_fill", "free_fill", "merged")
    strip = lambda p: {k: v for k, v in p.items() if k not in fills}  # noqa: E731
    assert strip(full["placement"]) == strip(pilot["placement"])
    for f in fills:
        assert full["placement"][f]["enabled"] is True
        assert pilot["placement"][f]["enabled"] is False
    assert full["sampling"]["n_sources"] == 600 and pilot["sampling"]["n_sources"] == 50
    assert full["paths"]["work"] not in ("data/inlayfake", pilot["paths"]["work"])
    assert full["run_name"] == "task2_insertion"
    assert full["sampling"]["only_sources"] == "data/inlayfake/plans.jsonl"  # week-1 photos


def test_cli_help():
    assert main([]) == 0


def test_natural_full_run_uses_pilot_v4_settings():
    """configs/insertion_natural.yaml = pilot v4's placement, edit and verify settings (v2's method
    with looser person/overlap/fill rules) on the 544 week-1 photos; only the photo set, folders,
    website block, audit sheet and the sham rule from the first full run differ."""
    full = load_config(ROOT / "configs" / "insertion_natural.yaml")
    pilot = load_config(ROOT / "configs" / "pilot_natural_v4.yaml")
    for key in ("edit", "generators_enabled", "encoding", "generation", "conditions"):
        assert full[key] == pilot[key], key
    # overlap limits raised from v4's 30% to 50% after the planning-only pass (author, 21 Oct)
    rules = full["placement"]["natural"].pop("box_rules")
    assert rules == {"max_cover_person": 0.50, "max_cover_real": 0.50}
    pilot["placement"]["natural"].pop("box_rules")
    assert full["placement"] == pilot["placement"]
    extra = ("audit_all_conditions", "audit_only_conditions", "sham_original_overlap")
    strip = lambda v: {k: x for k, x in v.items() if k not in extra}  # noqa: E731
    assert strip(full["verify"]) == strip(pilot["verify"])
    assert full["sampling"]["only_sources"] == "data/inlayfake/plans.jsonl"
    assert full["paths"]["work"] == "data/inlayfake_v3"


def test_box_mode_full_run_uses_the_pilot_settings():
    """configs/insertion_natural_box.yaml = the no-rules box-mode pilot's placement, edit,
    generation and encoding on the 544 week-1 photos; verify differs only by the full-run audit
    sheet and the sham rule from the first full run."""
    full = load_config(ROOT / "configs" / "insertion_natural_box.yaml")
    pilot = load_config(ROOT / "configs" / "pilot_natural_qwen_free.yaml")
    for key in ("placement", "edit", "generators_enabled", "encoding", "generation", "conditions"):
        assert full[key] == pilot[key], key
    extra = ("audit_all_conditions", "audit_only_conditions", "sham_original_overlap")
    strip = lambda v: {k: x for k, x in v.items() if k not in extra}  # noqa: E731
    assert strip(full["verify"]) == strip(pilot["verify"])
    assert full["sampling"]["only_sources"] == "data/inlayfake/plans.jsonl"
    assert full["paths"]["work"] == "data/inlayfake_v4"
    assert full["placement"]["natural"]["review"]["enabled"] is False  # author: no gate
