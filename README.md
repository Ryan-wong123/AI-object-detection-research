# InlayFake

Research code for **InlayFake: A Benchmark for Mixed-Origin Object Understanding**: can detectors and
image-forensic methods tell, *per object*, whether an object in a real photograph was photographed or
inserted by an AI image editor?

- `PRD.md`: goals, hypotheses, scope, timeline
- `ARCHITECTURE.md`: pipeline and data contracts
- `paper/PAPER.md`: living paper draft
- `docs/RESEARCH_LOG.md`: experiment log
- `AGENTS.md` / `CLAUDE.md`: instructions for AI coding agents

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (PowerShell)
# source .venv/bin/activate                              # Linux / WSL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[all]"
pytest -q
python -m inlayfake --help
```

## Viewing the detector results

The detector viewer is a web page that shows each photo's COCO labels next to every detector's
boxes, with the run's result tables at the top. Boxes are drawn at the same operating point as the
reported numbers: score ≥ 0.5, and an object counts as found with IoU ≥ 0.5.

### Just look at it (no setup)

The week-1 run is committed as a self-contained copy (page + data + downscaled photos):

```powershell
git clone https://github.com/Ryan-wong123/AI-object-detection-research.git
cd AI-object-detection-research
git checkout week-1-coco-dataset
start docs\viewer\task1_real_baseline\index.html      # Windows
# xdg-open docs/viewer/task1_real_baseline/index.html  # Linux
```

### Re-run the viewer from your own results

Needs the run's outputs in `runs/<run>/` and the photos in `data/` (both are produced by the
pipeline and are not in git). Nothing is re-detected, so this takes well under a minute on a laptop.

| Step | Command | Time | What it does |
|---|---|---|---|
| 1 | `python -m inlayfake report --config configs/real_baseline.yaml` | ~5–10 s | per-photo table `runs/task1_real_baseline/tables/images_used.md` (only needed if it is missing) |
| 2 | `python -m inlayfake viewer --config configs/real_baseline.yaml` | ~5–15 s | reads the saved boxes for 1,088 images (544 photos × C0 original / C1 resave) and writes the page |
| 3 | `start runs\task1_real_baseline\viewer\index.html` | instant | opens it in your browser |

The page in step 2 loads the photos from `data/`, so it only works on the machine that has them.

### Make a shareable copy

```powershell
python -m inlayfake viewer --config configs/real_baseline.yaml --share
git add docs/viewer/task1_real_baseline
git commit -m "Update shared detector viewer"
git push
```

`--share` writes `docs/viewer/<run>/` with the page, its data and downscaled copies of the photos
(longest side 640 px; the week-1 folder is about 70 MB), so the folder works anywhere: after a
clone, zipped, or through GitHub Pages (Settings → Pages → deploy from the branch, folder `/docs`).
Do not commit `runs/` or `data/` themselves; see `AGENTS.md` rule 8.
