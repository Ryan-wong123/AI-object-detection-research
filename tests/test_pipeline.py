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
