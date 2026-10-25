"""Figure 2 (H2): flip rate of original objects by distance to the edit (author, 2026-10-25).

    python scripts/figure_h2_distance.py configs/task3_detection_natural.yaml

Reads `runs/<run>/tables/h2_instances.csv` (written by `evaluate`) and the ledger, puts every
original object in a distance bin, and writes
  paper/results/task3_h2_by_distance.csv   flip rate per detector × condition × bin (95% CI) and the
                                           paired C4 − C2 difference on the same objects
  paper/figures/fig2_h2_distance.{png,pdf}

Distance = shortest gap between the object's C1 box and the photo's edit footprint (planned box +
mask dilation + one JPEG block, `eval.matched.edit_footprint`), in units of the footprint's longer
side. "covered" = the box overlaps the footprint (not mixed/modified). Mixed/modified objects are
left out: they were repainted, so their change measures the edit, not its influence.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from inlayfake.config import load_config
from inlayfake.eval.bootstrap import grouped_mean_ci
from inlayfake.eval.matched import box_gap, by_source, edit_footprint
from inlayfake.stages.common import load_records, run_dir

BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, float("inf"))]
LABELS = ["covered", "<0.5", "0.5–1", "1–2", "2–4", "≥4"]
CONDITIONS = ["C4", "C2", "C3", "C0"]
NAMES = {
    "faster_rcnn": "Faster R-CNN v2",
    "yolo": "YOLO11-m",
    "rtdetr": "RT-DETR-R50",
    "grounding_dino": "Grounding DINO-T",
}
# reference palette (dataviz skill), fixed order: AI insert first; C0 is a grey reference line
STYLE = {
    "C4": {"color": "#2a78d6", "marker": "o", "ls": "-", "label": "C4 AI insertion"},
    "C2": {"color": "#eb6834", "marker": "s", "ls": "-", "label": "C2 sham edit"},
    "C3": {"color": "#1baf7a", "marker": "^", "ls": "-", "label": "C3 real paste"},
    "C0": {
        "color": "#8a8985",
        "marker": "D",
        "ls": "--",
        "label": "C0 original bytes (re-encoding)",
    },
}


def distance_bin(stratum: str, d: float) -> str | None:
    if stratum == "covered":
        return "covered"
    if not stratum.startswith("uncovered"):
        return None
    for (lo, hi), label in zip(BINS, LABELS[1:], strict=True):
        if lo <= d < hi:
            return label
    return None


def main() -> int:
    cfg = load_config(sys.argv[1])
    rng = np.random.default_rng(cfg["seed"])
    n_boot = cfg["eval"]["bootstrap_resamples"]
    dist = {}
    for sid, recs in by_source(load_records(cfg)).items():
        fp = edit_footprint(recs, cfg["edit"]["mask_dilate_px"])
        if fp is None or "C1" not in recs:
            continue
        for inst in recs["C1"].instances:
            dist[(str(sid), inst.instance_id)] = box_gap(inst.bbox, fp) / max(fp[2], fp[3])

    flips: dict = defaultdict(lambda: defaultdict(list))  # (det, cond, bin) -> source -> flips
    by_obj: dict = defaultdict(dict)  # (det, bin, source, instance) -> cond -> flip
    path = run_dir(cfg) / "tables" / "h2_instances.csv"
    for r in csv.DictReader(path.open(encoding="utf-8")):
        key = (r["source_id"], r["instance_id"])
        b = distance_bin(r["stratum"], dist.get(key, float("nan")))
        if b is None:
            continue
        flip = float(r["det_c1"] != r["det_cx"])
        flips[(r["detector"], r["condition"], b)][r["source_id"]].append(flip)
        by_obj[(r["detector"], b, *key)][r["condition"]] = flip

    rows = []
    for det in NAMES:
        for b in LABELS:
            pairs: dict = defaultdict(list)
            for (d, bb, sid, _), v in by_obj.items():
                if d == det and bb == b and "C4" in v and "C2" in v:
                    pairs[sid].append(v["C4"] - v["C2"])
            diff = grouped_mean_ci(pairs, n=n_boot, rng=rng)
            for cond in CONDITIONS:
                g = flips.get((det, cond, b))
                if not g:
                    continue
                p, lo, hi = grouped_mean_ci(g, n=n_boot, rng=rng)
                rows.append(
                    {
                        "detector": det,
                        "condition": cond,
                        "distance_bin": b,
                        "n_objects": sum(len(v) for v in g.values()),
                        "flip": p,
                        "flip_lo": lo,
                        "flip_hi": hi,
                        "C4_minus_C2": diff[0],
                        "C4_minus_C2_lo": diff[1],
                        "C4_minus_C2_hi": diff[2],
                    }
                )
    out = Path("paper/results/task3_h2_by_distance.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})
    plot(rows, Path("paper/figures/fig2_h2_distance"))
    print(f"wrote {out} and paper/figures/fig2_h2_distance.png/.pdf")
    return 0


def plot(rows: list[dict], stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.edgecolor": "#52514e",
            "axes.labelcolor": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=True, layout="constrained")
    x = np.arange(len(LABELS))
    ymax = max(r["flip_hi"] for r in rows) * 100
    for ax, det in zip(axes, NAMES, strict=True):
        ax.axvspan(-0.5, 0.5, color="#f0efec", zorder=0)  # covered: occlusion, not influence
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
        for cond in CONDITIONS:
            st = STYLE[cond]
            pts = {
                r["distance_bin"]: r
                for r in rows
                if r["detector"] == det and r["condition"] == cond
            }
            xs = [i for i, b in enumerate(LABELS) if b in pts]
            ys = [pts[LABELS[i]]["flip"] * 100 for i in xs]
            if cond == "C4":
                lo = [pts[LABELS[i]]["flip_lo"] * 100 for i in xs]
                hi = [pts[LABELS[i]]["flip_hi"] * 100 for i in xs]
                ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15, linewidth=0, zorder=1)
            ax.plot(
                xs,
                ys,
                color=st["color"],
                ls=st["ls"],
                lw=1.5,
                marker=st["marker"],
                ms=4,
                mec="white",
                mew=0.8,
                label=st["label"],
                zorder=3 if cond == "C4" else 2,
            )
        ax.set_title(NAMES[det], fontsize=8.5, color="#0b0b0b")
        ax.set_xticks(x, LABELS, rotation=45, ha="right")
        ax.set_xlim(-0.5, len(LABELS) - 0.5)
        ax.set_ylim(0, ymax * 1.05)
        ax.tick_params(length=2)
    axes[0].set_ylabel("objects flipped C1 → Cx (%)")
    fig.supxlabel("distance from the edit footprint (× its longer side)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False, fontsize=7.5)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
