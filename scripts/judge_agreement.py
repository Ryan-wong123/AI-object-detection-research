"""Can Qwen judge its own boxes? Agreement with the author's review (author request, 2026-10-21).

Every valid spot of the no-rules pilot (data/inlayfake_pilot_natural_qwen_free) is shown to
Qwen3-VL-8B with the box drawn in red, with two fixed wordings of the question "would a {class}
in this box look natural and make sense here?"; Qwen's P(yes) is saved per spot
(<work>/judge.jsonl, resumable). With the author's decisions in <work>/spot_review.json, it then
reports AUROC, the best agreement over cut-offs, and agreement at 0.5, per wording.

    python scripts/judge_agreement.py           # score (GPU) + compare if the review exists
    python scripts/judge_agreement.py --compare # compare only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from inlayfake.config import load_config
from inlayfake.edit.encode import load_rgb
from inlayfake.stages.common import load_coco, read_jsonl
from inlayfake.stages.natural import _boxed
from inlayfake.stages.spots import read_review, review_path, spot_id

CFG = "configs/pilot_natural_qwen_free.yaml"
QUESTIONS = {
    "full": (
        "The red box marks where one more, new {cls} would be added to this photo. Would a "
        "{cls} filling exactly this red box look natural and make sense in this scene: resting "
        "on a suitable surface (or in open sky if it can fly), with a realistic size and "
        "orientation for its distance, and not on or in front of a person or another object?"
    ),
    "short": (
        "If a {cls} were painted into the red box, would the photo still look like a natural, "
        "believable scene?"
    ),
}


def score(cfg, wd: Path) -> None:
    index = load_coco(cfg)
    out = wd / "judge.jsonl"
    done = {(r["spot"], r["q"]) for r in read_jsonl(out)} if out.exists() else set()
    spots = [
        r
        for r in read_jsonl(wd / "natural_candidates.jsonl")
        if r.get("ok") or r.get("reason") in ("rejected in review", "not reviewed")
    ]
    todo = [(r, q) for r in spots for q in QUESTIONS if (spot_id(r), q) not in done]
    print(f"{len(spots)} spots, {len(todo)} questions to ask", flush=True)
    if not todo:
        return
    from inlayfake.edit.scene import build_vlm

    vlm = build_vlm(cfg["placement"]["natural"]["vlm"], cfg["hardware"]["device"])
    imgs: dict[int, np.ndarray] = {}
    with out.open("a", encoding="utf-8") as f:
        for n, (r, q) in enumerate(todo, 1):
            sid = r["source_id"]
            if sid not in imgs:
                imgs = {sid: load_rgb(index.image_path(sid))}
            p = vlm.yes_prob(_boxed(imgs[sid], r["box"]), QUESTIONS[q].format(cls=r["category"]))
            f.write(json.dumps({"spot": spot_id(r), "q": q, "p_yes": round(p, 4)}) + "\n")
            f.flush()
            print(f"{n}/{len(todo)}", flush=True)


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of an accepted spot > score of a rejected one), ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def compare(cfg, wd: Path) -> dict:
    dec = read_review(review_path(cfg))
    rows = read_jsonl(wd / "judge.jsonl")
    res = {
        "decisions": {
            "accept": sum(v == "accept" for v in dec.values()),
            "reject": sum(v == "reject" for v in dec.values()),
        }
    }
    for q in QUESTIONS:
        pairs = [(r["p_yes"], dec[r["spot"]]) for r in rows if r["q"] == q and r["spot"] in dec]
        if not pairs:
            continue
        pos = [p for p, d in pairs if d == "accept"]
        neg = [p for p, d in pairs if d == "reject"]
        best = max(
            (
                (t, np.mean([(p >= t) == (d == "accept") for p, d in pairs]))
                for t in sorted({p for p, _ in pairs} | {0.5})
            ),
            key=lambda x: x[1],
        )
        at = 0.5
        tp = sum(p >= at and d == "accept" for p, d in pairs)
        tn = sum(p < at and d == "reject" for p, d in pairs)
        res[q] = {
            "n": len(pairs),
            "auroc": round(auroc(pos, neg), 3),
            "agreement_at_0.5": round((tp + tn) / len(pairs), 3),
            "rejects_caught_at_0.5": f"{tn} of {len(neg)}",
            "accepts_kept_at_0.5": f"{tp} of {len(pos)}",
            "best_cutoff": round(float(best[0]), 3),
            "agreement_at_best": round(float(best[1]), 3),
            "always_accept_baseline": round(len(pos) / len(pairs), 3),
        }
    (wd / "judge_agreement.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main() -> int:
    cfg = load_config(CFG)
    wd = Path(cfg["paths"]["work"])
    if "--compare" not in sys.argv:
        score(cfg, wd)
    if review_path(cfg).exists():
        compare(cfg, wd)
    else:
        print("no review yet: run again with --compare after the author's review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
