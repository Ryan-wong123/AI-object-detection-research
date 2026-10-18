"""End-to-end CPU dry run: sample → plan → generate → preview with the fake inpainter."""

import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import get_sampling

from inlayfake.edit.encode import load_rgb
from inlayfake.ledger import Origin, read_ledger
from inlayfake.stages import generate, plan, preview, sample


def test_dry_run(cfg, tmp_path):
    s = sample.run(cfg)
    assert s["n_sources"] == 12  # 24 images, one ineligible (person), capped at 12
    p = plan.run(cfg)
    assert p["n_plans"] > 0
    g = generate.run(cfg)
    recs = list(read_ledger(tmp_path / "work" / "ledger_generated.jsonl"))
    assert len(recs) == g["ledger_records"]
    by = {}
    for r in recs:
        by.setdefault(r.source_id, {})[r.condition.value] = r
    sid, conds = next((k, v) for k, v in by.items() if "C5" in v)
    assert {"C0", "C1", "C2", "C3", "C4", "C5"} <= set(conds)

    # every edited image is a policy JPEG (quality 95, 4:4:4)
    for c in ("C1", "C2", "C3", "C4", "C5"):
        with Image.open(conds[c].path) as im:
            assert im.format == "JPEG" and get_sampling(im) == 0

    # inserted instance present in C4/C5, pasted in C3
    assert any(i.origin is Origin.AI_INSERTED for i in conds["C4"].instances)
    assert any(i.origin is Origin.REAL_PASTED for i in conds["C3"].instances)

    # outside the intended mask (+1 JPEG block), C4 decodes identically to C1
    c1, c4 = load_rgb(conds["C1"].path), load_rgb(conds["C4"].path)
    m = np.asarray(Image.open(conds["C4"].edit.intended_mask_path)) > 0
    ys, xs = np.nonzero(m)
    far = np.ones(m.shape, dtype=bool)
    far[max(ys.min() - 16, 0) : ys.max() + 17, max(xs.min() - 16, 0) : xs.max() + 17] = False
    assert np.array_equal(c1[far], c4[far])
    assert not np.array_equal(c1[~far], c4[~far])

    # resumable: a second run creates nothing new
    g2 = generate.run(cfg)
    assert g2["non_generative"] == 0 and g2.get("G1", 0) == 0

    out = preview.run(cfg)
    assert out["sources"] > 0


def test_real_only_run_has_no_ai(cfg, tmp_path):
    cfg = dict(cfg, conditions=["C0", "C1"], generators_enabled=[])
    sample.run(cfg)
    plan.run(cfg)
    generate.run(cfg)
    recs = list(read_ledger(tmp_path / "work" / "ledger_generated.jsonl"))
    assert recs and {r.condition.value for r in recs} == {"C0", "C1"}
    assert all(r.generator is None and r.edit is None for r in recs)
    assert all(i.origin is Origin.PHOTOGRAPHED for r in recs for i in r.instances)


def test_task1_real_baseline_end_to_end(cfg, tmp_path):
    """sample → plan → generate (C0/C1) → detect → forensics → audit → evaluate on CPU."""
    from inlayfake.stages import audit, detect, evaluate, forensics

    cfg = dict(cfg, conditions=["C0", "C1"], generators_enabled=[])
    for stage in (sample, plan, generate):
        stage.run(cfg)
    d = detect.run(cfg)
    assert set(d["images_processed"]) == {"fake_a", "fake_b"}
    assert detect.run(cfg)["images_processed"] == {"fake_a": 0, "fake_b": 0}  # resumable
    forensics.run(cfg)
    assert forensics.run(cfg)["images_processed"] == 0
    a = audit.run(cfg)
    assert a["candidates"] == 0  # fake detectors only echo ground truth
    evaluate.run(cfg)

    rd = tmp_path / "runs" / "dryrun"
    summary = (rd / "summary.md").read_text()
    assert "Detector AP" in summary and "ela" in summary
    import csv

    ap = list(csv.DictReader((rd / "tables" / "detector_ap.csv").open()))
    assert {r["condition"] for r in ap} == {"C0", "C1"}
    assert all(float(r["AP50"]) > 0.9 for r in ap)  # jittered GT boxes
    shift = list(csv.DictReader((rd / "tables" / "forensics_resave_shift.csv").open()))
    assert {r["method"] for r in shift} == {
        "ela",
        "srm_residual",
        "noise_level",
        "fft_highfreq",
        "jpeg_bag",
    }


def test_viewer_page(cfg, tmp_path):
    """The viewer writes index.html + data.js; per-panel found counts equal report's table."""
    import json

    from inlayfake.stages import detect, evaluate, report, viewer

    cfg = dict(cfg, conditions=["C0", "C1"], generators_enabled=[])
    for stage in (sample, plan, generate, detect, evaluate, report):
        stage.run(cfg)
    out = viewer.run(cfg)
    vdir = tmp_path / "runs" / "dryrun" / "viewer"
    assert (vdir / "index.html").exists() and out["conditions"] == ["C0", "C1"]
    data = json.loads((vdir / "data.js").read_text().split("=", 1)[1].rstrip().rstrip(";"))
    assert "Detector AP" in data["results_html"] and "<table>" in data["results_html"]

    import csv

    rows = {
        r["source_id"]: r
        for r in csv.DictReader((tmp_path / "runs/dryrun/tables/images_used.csv").open())
    }
    for im in data["images"]:
        view = im["cond"]["C0"]
        assert (vdir / view["src"]).resolve().exists()  # relative image path resolves
        for name in ("fake_a", "fake_b"):
            assert len(view["det"][name]["found"]) == int(rows[str(im["sid"])][f"found_{name}"])


def test_score_panel_rule():
    from inlayfake.stages.viewer import IGNORED, MATCHED, UNMATCHED, score_panel

    gt = [
        [0, 0, 10, 10, "cup", "coco"],
        [50, 50, 10, 10, "dog", "ignore"],
        [80, 0, 10, 10, "cat", "coco"],
    ]
    dets = [
        {"category": "cup", "bbox": [0, 0, 10, 10], "score": 0.9},
        {"category": "cup", "bbox": [1, 1, 10, 10], "score": 0.8},  # duplicate: still on a label
        {"category": "dog", "bbox": [50, 50, 10, 10], "score": 0.7},  # on an ignore region
        {"category": "vase", "bbox": [30, 30, 5, 5], "score": 0.6},  # no label
        {"category": "cat", "bbox": [80, 0, 10, 10], "score": 0.4},  # below threshold: hidden
    ]
    p = score_panel(dets, gt, 0.5, 0.5)
    assert [b[6] for b in p["boxes"]] == [MATCHED, MATCHED, IGNORED, UNMATCHED]
    assert p["found"] == [0] and p["n"] == 2  # the cat is not found at score >= 0.5


def test_markdown_to_html():
    from inlayfake.stages.viewer import markdown_to_html

    h = markdown_to_html("# T\n\nText **b** `c`\n\n| a | b |\n|---|---|\n| 1 | 2 |\n<!-- x -->")
    assert "<h2>T</h2>" in h and "<b>b</b>" in h and "<code>c</code>" in h
    assert "<th>a</th>" in h and "<td>2</td>" in h and "x -->" not in h


def test_viewer_share_is_self_contained(cfg, tmp_path):
    """--share writes page, data and downscaled photo copies into one folder."""
    import json

    from PIL import Image

    from inlayfake.stages import detect, viewer

    share_dir = tmp_path / "share"
    cfg = dict(
        cfg,
        conditions=["C0", "C1"],
        generators_enabled=[],
        site={"dir": str(share_dir)},
        viewer={"share_max_side": 200, "share_quality": 80},
    )
    for stage in (sample, plan, generate, detect):
        stage.run(cfg)
    out = viewer.run(cfg, share=True)
    vdir = share_dir / "dryrun"
    assert out["shareable"] and (vdir / "index.html").exists()
    data = json.loads((vdir / "data.js").read_text().split("=", 1)[1].rstrip().rstrip(";"))
    for im in data["images"]:
        for view in im["cond"].values():
            assert view["src"].startswith("images/")
            with Image.open(vdir / view["src"]) as p:
                assert max(p.size) <= 200
        assert im["w"] == 400  # boxes stay in original pixel coordinates


def test_scoring_uses_the_configured_audit_decisions(cfg, tmp_path):
    """audit.decisions: a run in another work dir is scored with the same decisions file (no
    copy in its own work dir); a named file that is missing stops the run."""
    import csv

    import pytest

    from inlayfake.config import deep_merge
    from inlayfake.stages.common import decisions_path, load_records

    cfg = dict(cfg, conditions=["C0", "C1"], generators_enabled=[])
    for stage in (sample, plan, generate):
        stage.run(cfg)
    sid = load_records(cfg)[0].source_id
    shared = tmp_path / "week1" / "audit_decisions.csv"
    shared.parent.mkdir()
    with shared.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "candidate_id",
                "source_id",
                "category",
                "x",
                "y",
                "w",
                "h",
                "n_detectors",
                "max_score",
                "decision",
            ]
        )
        w.writerow([f"{sid}-0", sid, "cup", 1, 2, 30, 40, 2, 0.9, "add"])
    assert decisions_path(cfg) == (tmp_path / "work" / "audit_decisions.csv", False)
    assert not any(
        i.instance_id.startswith("audit:") for r in load_records(cfg) for i in r.instances
    )
    scored = deep_merge(cfg, {"audit": {"decisions": str(shared)}})
    recs = [r for r in load_records(scored) if r.source_id == sid]
    assert recs and all(  # the added label is on every condition of that photo
        any(i.instance_id == f"audit:{sid}-0" for i in r.instances) for r in recs
    )
    with pytest.raises(SystemExit):
        load_records(deep_merge(cfg, {"audit": {"decisions": str(tmp_path / "missing.csv")}}))
