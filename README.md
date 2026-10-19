# InlayFake

> Terminology: C4-occ = the occlusion placement rule of the AI insertion before 18 Oct 2026 (cover 20–50% of one real object); since then it is part of the single AI condition C4.

Research code for **InlayFake: A Benchmark for Mixed-Origin Object Understanding**: can detectors and
image-forensic methods tell, *per object*, whether an object in a real photograph was photographed or
inserted by an AI image editor?

**View the results:** [ryan-wong123.github.io/AI-object-detection-research](https://ryan-wong123.github.io/AI-object-detection-research/)
(one page per week, chosen with the tabs at the top: week 1 real-photo baseline, week 2 AI
insertion). The link works once GitHub Pages is switched on for this repo (see
[Results website](#results-website-github-pages)).

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

## Results website (GitHub Pages)

`docs/index.html` joins the weekly result pages into one site with a tab per week at the top:

| Tab | Page shown | Built by |
|---|---|---|
| Week 1 | `docs/results/week1_real_baseline/` | `python -m inlayfake viewer --config configs/real_baseline.yaml --share` |
| Week 1 · Label audit | `docs/results/week1_real_baseline/audit.html` | `python -m inlayfake audit --config configs/real_baseline.yaml --review --share` |
| Week 2 | `docs/results/week2_ai_insertion/` | `python -m inlayfake preview --config configs/insertion_inphoto.yaml --share` |

Each week's page stays self-contained; the tab bar only shows it below. All week pages share
one design (header, photo cards, image grid, box colours, light/dark), defined once in
`src/inlayfake/site.py`. The address remembers the
tab (`index.html#week1`, `#week2`), and "open in a new tab" opens the week's page on its own.

**Look at it locally** (a small web server, so it works exactly as on GitHub):

```powershell
python -m http.server 8765 --directory docs
start http://localhost:8765
```

Stop the server with Ctrl+C.

**Publish it as the repo's GitHub Pages site** (once, in the browser): on GitHub open the repo,
then **Settings → Pages → Build and deployment → Source: Deploy from a branch**, choose the branch
(`week-2-ai-insertion` now, `main` once the weeks are merged) and the folder **`/docs`**, and click
**Save**. After a minute or two the site is at
`https://ryan-wong123.github.io/AI-object-detection-research/`. Every push to that branch updates
it. `docs/.nojekyll` tells GitHub to serve the files as they are. The same link is at the top of
this README.

**The site is public** even if the repo is private (on the free plan, Pages needs a public repo).
It shows downscaled COCO photos and names the repo owner, so go through
`docs/PUBLISHING_CHECKLIST.md` (photo licences, double-blind review) before switching it on.

**Add a week** (e.g. week 3):

1. In the week's config, add a `site` block: `week: 3`, `title: <short topic>`,
   `folder: week3_<topic>`. Its `--share` output then goes to `docs/results/week3_<topic>/`.
2. Build the week's page with the shared style: `SITE_CSS`, `page_title(cfg)` and
   `share_dir(cfg)` from `src/inlayfake/site.py` (as `viewer.py` and `preview.py` do), so it
   looks like the other weeks.
3. Add one entry to the `WEEKS` list in `docs/index.html` (id, label, short description,
   path `results/week3_<topic>/index.html`).

## Week 2: AI insertion run

Inserts an SDXL-generated object into each of the 544 week-1 baseline photos (conditions C2–C4)
and keeps only edits in which the verifier (OWLv2) finds the object. **Since 18 Oct 2026, C4 and
C4-occ are one AI condition (C4):** an AI object of a class already in the photo, placed anywhere.
A photo keeps its verified C4, else its verified C4-occ, else gets new boxes anywhere; C2 and C3 sit
at the same box (`placement.merged` in the config; `docs/RESEARCH_LOG.md`, 18 Oct). The settings come from the
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
original, the C1 resave, the C2 sham, the C3 paste and the C4 AI insertion that passed the check.
Yellow box = planned box, red box = inserted or pasted object (its verified box). A photo's
heading names what it lacks and why, e.g. `C4: still retrying · C2, C3: waits for the AI edit to
be verified`.

At the top, a summary counts complete photos, and a **Show** dropdown filters the page instantly
(no rebuild). Each option shows its count, and the number of photos shown appears next to it:

| Option | Shows photos where |
|---|---|
| all photos | everything |
| all conditions C0–C4 in the benchmark | C0, C1, C2, C3 and C4 are all there |
| any missing C (C2, C3 or C4) | C2, C3 or C4 is missing |

"Still retrying" and "waits" exist only while the run is going.

**Share it on GitHub.** `--share` writes a self-contained copy of the whole page (all 544
photos) to `docs/results/week2_ai_insertion/` (index.html + downscaled 480 px JPEG copies, about
95 MB), next to the week-1 viewer in `docs/results/week1_real_baseline/`. Rebuild it after the benchmark changes, then
commit the folder:

```powershell
python -m inlayfake preview --config configs/insertion_inphoto.yaml --share
git add docs/results/week2_ai_insertion
```

On GitHub, open `docs/results/week2_ai_insertion/index.html` after downloading or cloning the repo
(GitHub shows HTML files as source, not as a page).

**Update the shared copy after the benchmark changes** (for example, after the human audit removes
some C4 edits, or after a re-run adds edits). The copy is a snapshot and does not update itself.
Rebuild it, check the counts it prints, then commit and push:

```powershell
python -m inlayfake preview --config configs/insertion_inphoto.yaml --share
git add docs/results/week2_ai_insertion
git commit -m "Update shared week-2 preview"
git push
```

Images of edits that left the benchmark stay in `docs/results/week2_ai_insertion/images/` but are no
longer linked from the page; delete that folder before rebuilding if you want only the current
images in the repo.

Week-1 label-audit decisions live in one file, `data\inlayfake\audit_decisions.csv`, used by
both weeks (and week 3). To re-check them, build and open the review page (card *k* = row *k* of
the CSV; current decision; which detectors found the object and their scores (✓ = agrees,
score ≥ 0.7); zoom on the candidate in orange, COCO's own labels in green; whole photo; Show
dropdown by decision):

```powershell
python -m inlayfake audit --config configs/real_baseline.yaml --review
start data\inlayfake\audit_review.html
```

The same page is on the results website (tab *Week 1 · Label audit*,
`docs/results/week1_real_baseline/audit.html`).

To change a decision, edit the `decision` column of that row (`add` / `ignore` / `reject`), keep a
copy of the old file, and rebuild the page to see the new decision. Then:

1. If a row changed to or from `add`: remove that photo's rows from
   `runs\task1_real_baseline\forensics\*.jsonl` and run
   `python -m inlayfake forensics --config configs/real_baseline.yaml` (seconds; only that photo
   is redone).
2. `python -m inlayfake evaluate --config configs/real_baseline.yaml` (about 30–90 min, CPU).
3. `python -m inlayfake report --config configs/real_baseline.yaml` (about 10 s), and the week-1
   page with `viewer --config configs/real_baseline.yaml --share`.
4. Update the website copy of the audit page and push (seconds):

   ```powershell
   python -m inlayfake audit --config configs/real_baseline.yaml --review --share
   git add docs/results/week1_real_baseline
   git commit -m "Update week-1 label audit decisions"
   git push
   ```

   The decisions file itself is in `data\` (not in git); the website page is the shared
   record of the current decisions.

Week 2's images pick the change up when they are scored (week 3).
Before making the repo or the previews public, see
`docs/PUBLISHING_CHECKLIST.md` (licences of the COCO photos in these copies).

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

Build the preview as above, then do the **human audit of every C4 AI insertion** (544 rows, one
per photo). Column `in_random_sample` (`y` for 52 rows) marks a random subset (do not edit it);
the paper uses it for an unbiased error rate of the automatic verifier, and the full check to
remove every wrong AI label. C2 shams are not audited (author, 2026-10-18).

```powershell
start data\inlayfake_v2\audit_edits.html
start excel data\inlayfake_v2\audit_edits.csv
```

Card *k* in the HTML page is row *k* of the CSV (after the header). Each card shows the region the
verifier looked at, before (C1) and after the edit. A dashed yellow box marks the planned box (where the
object was to be drawn); on the *after* image a red box marks the AI object as the verifier
found it. Fill each row with `y` or `n`:

| Column | Fill with `y` / `n` |
|---|---|
| `object_ok` | one object of the target class is clearly there |
| `realistic` | the object blends into the photo |
| `other_change` | something outside the edited box changed |
| `notes` | anything else (e.g. "half a bed", "blurry blob") |

Save as CSV (keep the file name and columns). The audit sheet is rebuilt only if the CSV is
missing, so a filled sheet is never overwritten.

Notes:

- **Time:** about 28 h of GPU time on the RTX 4070 laptop (estimate from the pilot: ~5,100 SDXL
  images at ~20 s each; the first round of 1,355 images takes ~7.5 h), so plan for about two
  nights. Keep the laptop plugged in and stop Windows from sleeping.
- **Already done:** `sample` and `plan` for this run (544 of 544 photos planned). To redo them:
  `python -m inlayfake sample --config configs/insertion_inphoto.yaml`, then `plan` the same way.
- The final counts are in the JSON at the end of `insert.log`; record them in the research log,
  the paper and the week-2 briefing.
- The run started on 17 Oct 2026, 02:21, used way B.

## Week 2b: natural placement (pilot)

The first full run placed most AI objects in a random free spot, so some look wrong (a skateboard
in the air, a bird in a window pane). The natural-placement pipeline (`placement.natural`,
`configs/pilot_natural.yaml`) places each object where it would really be:

1. **Where:** a local VLM (Molmo-7B-D, 4-bit; not one of the VLMs week 3 evaluates) answers one
   fixed question per class from `configs/placement_prompts.yaml`, e.g. *"Point to empty spots on
   a table, counter, desk or shelf where another cup could naturally rest."* Each point is where
   the object touches its surface.
2. **Size:** from the photo's own object of that class, scaled by depth (Depth Anything V2 Small):
   further back means smaller (limited to ×0.5–2).
3. **Rules:** never on a person, at most 15% of any real object covered, not resting on window,
   mirror, ceiling or sky (sky allowed for bird, airplane, kite, frisbee). All candidates and
   reasons: `natural_candidates.jsonl` in the work folder.
4. **Generate** C4 (SDXL), C3 (real paste) and C2 (sham) at that box, as before.
5. **Check:** OWLv2 must find the object, then the VLM answers four yes/no questions (resting on
   something, not behind glass, plausible size, one complete object); a failure is regenerated
   (3 seeds per spot, then the next natural spot; no random fallback). Answers: `realism.jsonl`.

Pilot (50 photos, about 2–3 h of GPU time; the first run downloads Molmo, 32 GB):

```powershell
python -m inlayfake sample --config configs/pilot_natural.yaml
python -m inlayfake plan   --config configs/pilot_natural.yaml
python -m inlayfake insert --config configs/pilot_natural.yaml
python -m inlayfake preview --config configs/pilot_natural.yaml --limit 50
start data\inlayfake_pilot_natural\preview.html
```

### Full natural-placement run (544 photos, pilot v2 settings)

`configs/insertion_natural.yaml` runs the pilot-v2 method on the 544 week-1 photos into
`data\inlayfake_v3` (the random-placement run in `data\inlayfake_v2` stays as the comparison).
Expected: about 14–27 h of GPU time (best guess ~18 h: planning ~4.5–8 h, generation ~9–19 h);
about 190–200 photos with a verified natural AI edit (estimate from the pilot). Close the browser
and use a high-performance mode (MSI Center): other programs on the GPU slowed the pilots 2–4×.
Start it in the background (keeps running if this window closes; asks Windows not to idle-sleep):

```powershell
python -m inlayfake sample --config configs/insertion_natural.yaml
$code = "import ctypes, subprocess, sys; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); [subprocess.run([sys.executable, '-m', 'inlayfake'] + a) for a in (['plan','--config','configs/insertion_natural.yaml'], ['insert','--config','configs/insertion_natural.yaml'], ['preview','--config','configs/insertion_natural.yaml','--limit','544'])]"
$p = Start-Process -FilePath (Resolve-Path .venv\Scripts\python.exe) -ArgumentList @('-c', "`"$code`"") -WorkingDirectory (Get-Location).Path -RedirectStandardOutput data\inlayfake_v3\run.log -RedirectStandardError data\inlayfake_v3\run_progress.log -WindowStyle Hidden -PassThru
$p.Id | Out-File data\inlayfake_v3\run.pid -Encoding ascii
```

Check progress (progress bar, then passed / retrying / dropped counts):

```powershell
((Get-Content data\inlayfake_v3\run_progress.log -Raw -Encoding utf8) -split "`r|`n" | Where-Object { $_ -match '^(natural placement|G1|verify)' })[-1]
python -c "import json,glob,os; f=max(glob.glob('data/inlayfake_v3/runs/run_verify_*.json'),key=os.path.getmtime); d=json.load(open(f)); print({c:{k:v for k,v in d['by_condition'][c].items() if not k.startswith('passed_on')} for c in d['by_condition']}, 'needs_retry:', d['needs_retry'])"
```

The run is resumable, also after a crash or power loss: start the same command again. Planning
saves every photo as it finishes (`natural_plan_progress.jsonl`), generation every image, and
verification every check, so at most the photo or image in progress is redone. Files are written
crash-safely (a cut-off last log line is skipped; records and plans are replaced in one step).

## Viewing the detector results

The detector viewer is a web page that shows each photo's COCO labels next to every detector's
boxes, with the run's result tables at the top. Boxes are drawn at the same operating point as the
reported numbers: score ≥ 0.5, and an object counts as found with IoU ≥ 0.5.

### Just look at it (no setup)

The week-1 run is committed as a self-contained copy (page + data + downscaled photos):

```powershell
git clone https://github.com/Ryan-wong123/AI-object-detection-research.git
cd AI-object-detection-research
git checkout week-2-ai-insertion
start docs\results\week1_real_baseline\index.html      # Windows
# xdg-open docs/results/week1_real_baseline/index.html  # Linux
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
git add docs/results/week1_real_baseline
git commit -m "Update shared detector viewer"
git push
```

`--share` writes the week's folder `docs/results/week1_real_baseline/` (from the `site` block of
the config) with the page, its data and downscaled copies of the photos
(longest side 640 px; the week-1 folder is about 70 MB), so the folder works anywhere: after a
clone, zipped, or through GitHub Pages (Settings → Pages → deploy from the branch, folder `/docs`).
Do not commit `runs/` or `data/` themselves; see `AGENTS.md` rule 8.
