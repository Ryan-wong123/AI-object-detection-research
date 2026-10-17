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

## Week 2: AI insertion run

Inserts an SDXL-generated object into each of the 544 week-1 baseline photos (conditions C2–C5)
and keeps only edits in which the verifier (OWLv2) finds the object. The settings come from the
same-class pilot; see `configs/insertion_inphoto.yaml` and `docs/RESEARCH_LOG.md` (17 Oct 2026).
Output goes to `data\inlayfake_v2` (git-ignored); the week-1 data in `data\inlayfake` is untouched.

Commands (PowerShell, in the repo folder). First activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

### Start the run (pick one way)

**A. In a PowerShell window.** Simple, but the window must stay open for the whole run. Progress
and the final summary both go to `insert.log`:

```powershell
python -m inlayfake insert --config configs/insertion_inphoto.yaml *> data\inlayfake_v2\insert.log
```

**B. As a background process.** Keeps running after you close the window or this app, and asks
Windows not to idle-sleep while it runs (a closed lid still sleeps). Progress goes to
`insert_progress.log`, the final summary to `insert.log`, and the process ID to `insert.pid`:

```powershell
$code = "import ctypes, runpy, sys; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); sys.argv=['inlayfake','insert','--config','configs/insertion_inphoto.yaml']; runpy.run_module('inlayfake', run_name='__main__')"
$p = Start-Process -FilePath (Resolve-Path .venv\Scripts\python.exe) -ArgumentList @('-c', "`"$code`"") -WorkingDirectory (Get-Location).Path -RedirectStandardOutput data\inlayfake_v2\insert.log -RedirectStandardError data\inlayfake_v2\insert_progress.log -WindowStyle Hidden -PassThru
$p.Id | Out-File data\inlayfake_v2\insert.pid -Encoding ascii
```

Never start a second run while one is still running: both would write to the same folder.

### Check on it (from any PowerShell window)

Progress (way B; for way A use `insert.log` instead):

```powershell
Get-Content data\inlayfake_v2\insert_progress.log -Tail 2 -Encoding utf8
```

A line like `G1: 85% | 1145/1355 [5:47:52<1:05:04, 18.59s/it]` means: SDXL (G1) has made 1,145 of
this round's 1,355 images, 5 h 48 min so far, about 1 h 5 min left in this round, 18.6 s per
image. Without `-Encoding utf8` the bar shows as `â–ˆ`, which is harmless. A line starting with
`verify:` is OWLv2 checking the round's images (a few minutes). Right after a round reaches 100%,
the log can stay unchanged for a minute or two while OWLv2 and SAM load.

Is it still running? (way B; for way A, the window is still busy):

```powershell
Get-Process -Id (Get-Content data\inlayfake_v2\insert.pid)
```

This is the small `.venv` launcher, so its CPU time stays near 0; the real work runs in a child
Python process. If this shows an error, the run has ended: finished if `insert.log` ends with
the JSON summary, otherwise it stopped (restart it, below).

### How many edits need a retry

Round 1 makes one image per edit; OWLv2 then checks them, and only the failed edits go into the
next round (new seed, then new box, then another class; see `ARCHITECTURE.md` §3.5). After a
round's check:

1. **The verify summary** of the newest round. The last `needs_retry` line is the total going
   into the next round; `passed` lines are edits already in the benchmark:

   ```powershell
   Get-Content (Get-ChildItem data\inlayfake_v2\runs -Filter run_verify_*.json | Sort-Object Name | Select-Object -Last 1).FullName | Select-String '"(passed|needs_retry|dropped)"'
   ```

2. **Passed and failed images per condition** so far (all rounds together):

   ```powershell
   python -c "import json,collections; rows=[json.loads(l) for l in open('data/inlayfake_v2/verify.jsonl')]; c=collections.Counter((r['condition'], 'passed' if r['passed'] else 'failed') for r in rows); [print(k, v) for k, v in sorted(c.items())]"
   ```

3. **The next round's size**: when it starts, its progress bar shows the total, e.g.
   `G1: 0% | 0/640` means 640 images in that round (same `Get-Content ... -Tail 2` command as
   above).

### Stop or restart

Stop (way B; for way A press Ctrl+C in its window):

```powershell
Stop-Process -Id (Get-Content data\inlayfake_v2\insert.pid)
Get-Process python
```

If `Get-Process python` still lists a Python process using the GPU, stop that one too
(`Stop-Process -Id <id>`). To continue after a stop, a reboot or a crash, start it again with
way A or B: finished images and checks are kept, and it continues where it stopped.

### See the results so far: the preview page

The preview can be built at any time, also while the run is going (it only reads saved results;
a few seconds on the CPU). Without `--limit` it shows only the first 50 photos, so pass 544:

```powershell
python -m inlayfake preview --config configs/insertion_inphoto.yaml --limit 544
start data\inlayfake_v2\preview.html
```

The page shows **only images that are in the benchmark** (`ledger.jsonl`): per photo the C0
original, the C1 resave, and the C2 / C3 / C4 / C5 edits that passed the check. Yellow box =
planned box, red box = inserted or pasted object (its verified box). Only edits that used a
last resort carry a note (`class not in photo` or `placed anywhere`). A grey note in a photo's
heading names what it lacks: `not in benchmark (yet): C4, C5` (planned, not passed yet) or
`not planned (no room): C2, C3, C4` (the photo never had room for that box, e.g. C5-only photos).

At the top, a summary counts complete and incomplete photos, and a **Show** dropdown filters the
page instantly (no rebuild); the number of photos shown appears next to it:

| Option | Shows photos where |
|---|---|
| all photos | everything |
| all conditions C0–C5 in the benchmark | every condition passed |
| any missing C | any condition is missing (not passed yet, dropped, or not planned) |
| photos missing a planned edit | a planned edit is not in the benchmark yet |
| C4 not planned | there is no free space for a box that touches no real object |
| C4 still retrying | C4 is planned, no attempt has passed yet |
| C4 dropped | C4 is planned, every allowed attempt failed or no further box fits |
| C5 not planned | no real object can be covered 20–50% under the rules |
| C5 still retrying / C5 dropped | as for C4 |

Each photo's heading names what it lacks and why, e.g. `C2, C3, C4: still retrying ·
C5: not planned (no room for its box)`. "Still retrying" exists only while the run is going.

To see every attempt, including the rejected ones (greyed, with the reason), add
`--all-attempts`:

```powershell
python -m inlayfake preview --config configs/insertion_inphoto.yaml --limit 544 --all-attempts
```

**The page is a snapshot:** it shows the benchmark at the moment it was built and does not update
itself. You do not need to rebuild it after every round (the run does not use it). To see the
latest progress, rebuild it, then refresh the browser tab with **Ctrl+F5**. This takes a few
seconds and does not disturb the running job:

```powershell
python -m inlayfake preview --config configs/insertion_inphoto.yaml --limit 544
```

New results appear only after a round's check (the `verify:` line reaching 100%), so rebuilding
more than once per round shows nothing new. When the run has finished, rebuild it once more:
that version shows the final benchmark and is the one to use for the human audit.

### When it's done

Build the preview as above, then fill the human-audit sheet `data\inlayfake_v2\audit_edits.csv`
(y/n for object_ok, realistic, other_change; before/after crops in `audit_edits.html`).

Notes:

- **Time:** about 28 h of GPU time on the RTX 4070 laptop (estimate from the pilot: ~5,100 SDXL
  images at ~20 s each; the first round of 1,355 images takes ~7.5 h), so plan for about two
  nights. Keep the laptop plugged in and stop Windows from sleeping.
- **Already done:** `sample` and `plan` for this run (544 of 544 photos planned). To redo them:
  `python -m inlayfake sample --config configs/insertion_inphoto.yaml`, then `plan` the same way.
- The final counts are in the JSON at the end of `insert.log`; record them in the research log,
  the paper and the week-2 briefing.
- The run started on 17 Oct 2026, 02:21, used way B.

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
