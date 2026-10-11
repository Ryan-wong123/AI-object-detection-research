# Research log

Dated record of every experiment, decision and finding. Newest entries at the top.
Format: date · run ID (if any) · what · key numbers · interpretation · next step.

---

## 2026-10-12: Decision: weekly briefings are kept current like the paper

- **Decision (author):** the briefing for the current week (`docs/briefings/weekN/`:
  `briefing_full.md`, `one_page.md`, `slides.html`) is updated after every run, finding, decision or
  code change for the week's task, and created at the start of a new week. Added as
  `AGENTS.md` §Weekly briefing protocol, as step 6 of the experiment workflow, and in `CLAUDE.md`.
  Why: the briefings had fallen behind `briefing_full.md` and the paper between sessions.
- **Found while applying it:** week 1's `one_page.md` and `slides.html` still say the 56 unplanned
  photos were "too cluttered" and that there are 37 unit tests; `briefing_full.md` already describes
  them as close-ups (Key finding 4) and the log records 38 tests. Waiting on the author before
  syncing.

---

## 2026-10-11: Task 1 final evaluate with 200 bootstrap resamples; week 1 complete

- **Run:** `task1_real_baseline`, `evaluate` (run_evaluate_2026-10-11T150104Z.json, git 780a397)
  with `eval.ap_bootstrap: 200` (`configs/real_baseline.yaml`), on the RTX 4070 laptop. Reuses the
  saved detections and forensic scores and the confirmed audit decisions (77 add / 26 ignore /
  4 reject). Then `report` for the per-photo table. Summary saved as
  `paper/results/task1_real_baseline_summary_v4_200boot.md`.
- **Check:** every point estimate (AP, AP50, AR100, recall, lost/gained, forensic means, medians,
  p95, object contrast) is identical to v3 (`..._v3_audited.md`). Only the confidence intervals
  changed, so the inputs are unchanged.
- **Final AP 95% CIs (C0 / C1):** Faster R-CNN 0.500 [0.460, 0.515] / 0.498 [0.460, 0.510];
  YOLO11-m 0.530 [0.484, 0.547] / 0.528 [0.487, 0.547]; RT-DETR 0.512 [0.479, 0.527] /
  0.511 [0.476, 0.532]; Grounding DINO 0.515 [0.481, 0.532] / 0.515 [0.476, 0.525]. The intervals
  are wider than the 20-resample ones and overlap heavily across detectors: on 544 photos, the AP
  ranking of the four detectors is not resolved.
- **Resave shift (C1 − C0) CIs:** ELA +0.188 [0.170, 0.206], SRM −0.106 [−0.114, −0.099], noise
  −0.089 [−0.142, −0.038], FFT +0.0045 [0.0029, 0.0064], JPEG grid −0.025 [−0.027, −0.022]. All
  still exclude zero, so Findings (ii) in §7.1 stand. They moved slightly because the random
  generator is shared with the AP bootstrap.
- **Per-photo table:** `docs/briefings/week1/images_used.md`. Totals found / 3,959 reproduce recall
  C0 exactly (2938 → 0.742, 2053 → 0.519, 2387 → 0.603, 1826 → 0.461).
- **Updated:** PAPER §7.1 Tables 2a/2b and Status box, PRD Status, briefing Key finding 5.
- **Note:** PAPER §6 says CIs use 1000 resamples; Task 1 uses 200 (stated in Table 2a). Decide
  before submission whether the main experiments use 1000 or §6 should say 200.
- **Next:** Task 2 (AI insertion): SDXL pilot on 50 sources with `configs/pilot.yaml`.

---

## 2026-10-11: The 56 photos without a plan are close-ups; new `report` stage

- **Finding:** the 56 sources that failed `plan` are not busy scenes. Their median number of
  labelled objects is 4, the same as the 544 kept photos, but the union of their object boxes covers
  a median 97% of the frame (kept: 32%). In 52 of 56, one object covers more than half the image,
  most often a dining table (18), bowl (6), bed (5) or person (5). Split: 39 train, 7 val, 10 test.
  Computed from the plan stage's `failed_sources` list (seed 20261008) and the COCO val2017
  annotations. Interpretation: the benchmark under-represents close-up, frame-filling shots, not
  crowded scenes. The paper's wording ("too cluttered", "very crowded scenes", §4 and §8) should
  change; waiting for the author's go-ahead and a check of the list on the RTX 4070.
- **Added:** the full list of 56 (COCO IDs linked to the official images, split, object count,
  coverage, main labels) as Key finding 4 in `docs/briefings/week1/briefing_full.md`.
- **New stage `report`:** `python -m inlayfake report --config ...` writes
  `runs/<run>/tables/images_used.{csv,md}`, one row per C0 photo with split, planned insert,
  labelled objects and the number found per detector (the same rule as `evaluate`'s recall, so the
  totals reproduce recall C0). Why: the author wants the briefing to list every photo used and what
  was detected in it. Unit test with a hand-checked example; 38 tests pass. The real table is
  produced on the RTX 4070 (added to `TASK.md`).

---

## 2026-10-11: Decision: Stage 2 (VLM origin claims, H5) is a must-have

- **Decision (author):** the VLM caption study is no longer optional (P1). It is part of the P0
  scope, and H5 is a full hypothesis rather than a stretch goal. This reverses the 2026-10-08
  decision that reduced Stage 2 to P1.
- **Why:** Stage 2 is part of the professor's brief, and the author wants the paper to cover it.
- **Changed:** PRD (H5 no longer "stretch"; Stage 2 moved into the P0 list; success criteria need a
  finding for H5; new risk row; timeline: `vlm` stage, caption metrics and the plain/grounded prompts
  in week 3, origin-aware prompts and caption evaluation in week 4). PAPER (abstract gains an H5
  placeholder; new contribution 4; "(P1)" removed from §2 and §7.7). ARCHITECTURE (P1 labels removed
  from `vlm` and `captions.py`). Week-1 briefing files updated.
- **Scope kept small to fit the schedule:** two small VLMs (Florence-2, Qwen2.5-VL-3B in 4-bit),
  three prompt settings, about 200 test images plus unedited controls. If time slips, fewer images,
  not dropping the study.
- **Next:** write the `vlm` stage in week 3 on the second laptop; runs that feed the paper happen on
  the RTX 4070.

---

## 2026-10-11: Label audit re-checked; config set for the final `evaluate`

- **Audit:** the author re-checked all 107 drafted decisions (77 add / 26 ignore / 4 reject) on a
  review page showing each crop next to its decision, and confirmed every one. No change to
  `audit_decisions.csv`, so Table 2a's point estimates stand.
- **Changed:** `configs/real_baseline.yaml` now sets `eval.ap_bootstrap: 200`, the setting for the
  paper's final tables (`default.yaml` keeps 20 for quick runs). Why in this config: the author only
  needs to re-run `evaluate` on run `task1_real_baseline`; `detect`/`forensics` outputs are reused.
- **PRD status** corrected: Task 1 has run; only the final `evaluate` remains.
- **Next:** on the RTX 4070 laptop, `git pull` then
  `python -m inlayfake evaluate --config configs/real_baseline.yaml`; record the 200-resample CIs in
  Table 2a/2b.

---

## 2026-10-09: Label audit applied, final Task 1 `evaluate` (run `task1_real_baseline`)

Source: `paper/results/task1_real_baseline_summary_v3_audited.md`, decisions in
`paper/results/task1_audit_decisions.csv`.

- **Audit decisions (107 candidates):** 77 add, 26 ignore, 4 reject. Drafted from a visual review of
  every crop in `audit.html`. **The author has not re-checked them yet.** If any change, re-run
  `evaluate` (≈ 5 min) and update Table 2a.
  - Reject: `19742-0` (a candle, not a cup), `253742-0/1` (deer, not sheep; deer is not a COCO
    class), `560474-0` (a doll's face, not a person).
  - Ignore: too small or dark to judge (10), possible duplicates of an existing label, e.g. `26204-1`
    overlaps `26204-0` (4), class doesn't clearly fit, e.g. statue cow, plush clownfish, cocoa tin as
    cup, poster person (7), other unclear (5).
  - Why ignore rather than guess: an ignore region is neither rewarded nor penalised, so an
    uncertain call cannot bias AP either way.
- **Effect on detectors:** 3,959 GT objects (was 3,882). AP rises by 0.004–0.006 for every detector
  and condition (e.g. Faster R-CNN C0 0.494 → 0.500, YOLO11-m 0.526 → 0.530). Resave flips are
  unchanged (0.40–0.76%). Interpretation: the missing labels made detectors look slightly worse
  than they are; the ranking and the C0≈C1 conclusion do not change.
- **Forensic tables unchanged:** per-object contrast is computed in the `forensics` stage over the
  original 3,882 labels, so the 77 additions are not in Table 2b. They are 2% of objects, so this
  is unlikely to matter; re-run `forensics` before the final version if we want them included.
- **Pre-audit outputs kept** as `runs/task1_real_baseline/summary_preaudit.md` and
  `metrics_preaudit.json`.
- **Paper:** §4 Sources filled (77/26/4), Table 2a and finding (iv) updated, Status box updated.
- **Week 1 done** apart from the author's re-check of the audit and 200-resample AP CIs.

---

## 2026-10-08: Task 1 re-run after fixes (v2, run `task1_real_baseline`)

Source: `paper/results/task1_real_baseline_summary_v2.md`.

- **jpeg_bag fixed:** image score mean 0.532 / median 0.484 / p95 0.99 (was 428.7 / 0.667). Object
  contrast −0.003 ± 0.053 (sd was 61.8). Resave shift −0.025 [−0.028, −0.022] (−4.7%). The method
  now behaves like the others and is sensitive to re-encoding.
- **AP CIs (basic bootstrap):** e.g. Faster R-CNN C0 0.494 [0.462, 0.506]. Intervals now contain the
  point estimate but are skewed downward: the resample mean is about 0.015 above the point estimate
  (the category-dropout bias), and the reflection moves that skew below. 20 resamples only; use 200
  for the final tables.
- **New observation:** F4 (FFT) and F5 (blocking grid) have ≈ 0 mean object-vs-context contrast on
  real objects, unlike ELA/SRM/noise (0.25–0.51). They may be less confounded by object texture;
  check this when AI objects arrive (Task 3).
- All other numbers are identical to v1, as expected (only jpeg_bag and the CI method changed).
- **Paper:** Tables 2a/2b complete (F5 row and AP CIs filled); findings (ii) and (iii) updated.
- **Week 1 remaining:** audit review of 107 candidates, then a final `evaluate`.

---

## 2026-10-08: Task 1 results (v1) and two fixes (run `task1_real_baseline`)

Source: author's `summary.md`, saved as `paper/results/task1_real_baseline_summary_v1.md`.

- **Detectors on real photos:** AP C0 / C1: Faster R-CNN 0.494 / 0.493, YOLO11-m 0.526 / 0.524,
  RT-DETR 0.508 / 0.507, Grounding DINO 0.511 / 0.510. These are in the range expected for these
  models on COCO val, a sanity check that the wrappers, the category mapping and the evaluation work.
- **Resave effect (C0→C1):** AP changes ≤ 0.002. Per-object flips are 0.41–0.77% lost and gained,
  roughly symmetric (3,882 objects). This is the noise floor for H2.
- **Recall at score ≥ 0.5 differs a lot by detector** (0.46–0.74), mostly because of score
  calibration: YOLO and Grounding DINO give lower scores. Use it only for within-detector
  comparisons (flips), never to rank detectors.
- **Forensics on real photos:** all methods shift significantly with resaving (ELA +3.7%, SRM −3.5%,
  noise −0.9%, FFT +1.2%), so C1, not C0, must be the reference. Real objects have positive mean
  contrast against their context for ELA (0.50), SRM (0.30) and noise (0.25): texture makes real
  objects stand out, so calibration has to be against real objects.
- **Problem 1, jpeg_bag outliers:** mean image score 428 vs median 0.67, and object-contrast sd 62.
  The ratio on/off-grid divided ~0 by ~0 in flat or saturated regions. **Fix:** log ratio with +1 grey
  level in numerator and denominator; regression test added (fails on the old code). Requires
  re-running `forensics` for jpeg_bag only.
- **Problem 2, AP CIs biased:** all 8 percentile intervals sit mostly above the point estimate
  (e.g. 0.494 [0.479, 0.527]). COCO AP averages over the categories present, and rare, low-AP
  categories drop out of resamples, so resampled APs are inflated. **Fix:** basic (reverse-percentile)
  bootstrap interval; the resample mean is also stored. Requires re-running `evaluate`.
- **Paper:** §7.1 drafted with Table 2a/2b and four findings; F5 row and AP CIs left as TBD until
  the re-run.
- **Next:** delete `forensics/jpeg_bag.jsonl`, re-run `forensics` and `evaluate`, then review the
  audit.

---

## 2026-10-08: Task 1 `forensics`, `audit`, `evaluate` complete (run `task1_real_baseline`)

- **forensics:** 1,088 images × 5 methods in 2 min 55 s (6.2 img/s on the laptop CPU).
- **audit:** 107 candidate missing labels in 70 of 544 C0 images (12.9% of images have at least one
  confident object, agreed by ≥ 2 detectors, that COCO did not label). This is consistent with the
  label-incompleteness finding of *Object Detection Benchmarks are Incomplete*. No decisions yet: all
  are "?" and are therefore treated as ignore regions in this evaluation.
- **evaluate:** finished; `runs/task1_real_baseline/summary.md` written (1,088 records). The numbers
  will be recorded once the author shares the summary.
- **Next:** author pastes `summary.md`; reviews `audit.html` and fills `audit_decisions.csv`; re-run
  `evaluate`.

---

## 2026-10-08: `evaluate` looked stuck: progress output added

- **What:** `evaluate` printed nothing for a long time on the laptop. The cause is the AP bootstrap:
  4 detectors × 2 conditions × (1 + 50 resamples) ≈ 400 full COCOevals on 544 images, with no
  progress output.
- **Fix:** progress messages per detector/condition and a progress bar for the resamples. The default
  `eval.ap_bootstrap` is now 20 for quick runs; it will be raised to 200 for the paper's final tables
  (wider-than-final CIs until then).

---

## 2026-10-08: `detect` complete for all four detectors (run `task1_real_baseline`)

- **Ran (author's laptop):** after the fix, Grounding DINO processed all 1,088 images in 22 min 13 s
  (1.23 s/img, 4 prompt chunks per image). The other three were skipped (already complete).
- **State:** detections exist for faster_rcnn, yolo, rtdetr and grounding_dino on 544 × (C0, C1).
- **Next:** `forensics`, `audit`, `evaluate`.

---

## 2026-10-08: `detect` run: 3 of 4 detectors done; Grounding DINO bug fixed

- **Ran (author's laptop):** `detect` on 1,088 images. Faster R-CNN (1 min 56 s, 9.3 img/s), YOLO11-m
  (26 s, 41 img/s) and RT-DETR (59 s, 18 img/s) completed. Grounding DINO crashed after 10 images
  (≈1.4 s/img): `zip() argument 2 is longer than argument 1`. The installed transformers'
  `post_process_grounded_object_detection` returned more phrase labels than boxes.
- **Fix:** Grounding DINO labels are now decoded directly from the token logits. Each query's score
  for a class is the max sigmoid over that class name's tokens, and the query keeps the best class if
  that score is ≥ 0.05. This no longer depends on the processor's post-processing (which changes
  across versions). Unit tests cover the prompt spans, token→class mapping and decoding.
- **Action for the author:** delete the partial `grounding_dino.jsonl` (its 10 rows used the old
  decoding) and re-run `detect`; the other three detectors are skipped automatically.

---

## 2026-10-08: Task 1 data stages run on the laptop (run `task1_real_baseline`)

- **Ran (author, RTX 4070 laptop):** `sample`, `plan` and `generate` with `configs/real_baseline.yaml`.
- **Numbers (as printed by the CLI):**
  - sample: 600 sources; train 362 / val 114 / test 124 (60.3 / 19.0 / 20.7%).
  - plan: 544 planned, **56 failed (9.3%)**, 363 of 544 with a C5 occlusion box (66.7%).
    Categories: chair 62, bottle 56, potted plant 49, vase 50, clock 57, bench 51, backpack 65,
    dog 44, cup 59, umbrella 51.
  - generate: 1,088 records = 544 × (C0 + C1), real only; 27 s.
- **Interpretation:** The splits match the 60/20/20 target. The failed plans are photos with no free
  region for a new object (crowded scenes). All later conditions use the same 544 sources, so the
  benchmark under-represents very cluttered scenes; this has been added to the paper's limitations.
  C5 will cover only about 2/3 of sources; acceptable, but report C5 counts separately.
- **Next:** `detect` (GPU; first run downloads the four detectors' weights), then `forensics`,
  `audit`, `evaluate`.

---

## 2026-10-08: Windows test-environment fix

- **What:** On the author's laptop, `pytest -q` gave 28 passed and 5 errors. All 5 were
  `PermissionError [WinError 5]` on `%TEMP%\pytest-of-USER`: pytest could not open its Windows temp
  folder, so every test that uses `tmp_path` failed during setup. This was not a bug in the
  project code.
- **Fix:** `pyproject.toml` now sets `--basetemp=.pytest_tmp` (a git-ignored folder inside the repo).
- **Next:** the author re-runs `pytest -q` (expect 33 passed), then the Task 1 commands.

---

## 2026-10-08: Task 1 code complete (branch `week-1-coco-dataset`)

- **Implemented:** `detect` (Faster R-CNN v2, YOLO11-m, RT-DETR-R50, Grounding DINO-T), `forensics`
  (F1 ELA, F2 SRM residual variance, F3 wavelet noise-level deviation, F4 FFT high-frequency
  deviation, F5 JPEG blocking-grid deviation), `audit` (candidate missing labels → HTML +
  decisions CSV) and `evaluate` (Task 1 tables + `summary.md`).
- **Checks:** 33 CPU tests pass. Forensic tests confirm that ELA and the blocking grid score higher
  on a patch that skipped the JPEG history, and the noise method scores higher on a noisier patch.
  The detector code paths were smoke-tested on CPU: YOLO with real weights; Faster R-CNN and RT-DETR
  with random weights (their weight hosts are blocked in the cloud sandbox). Grounding DINO's
  post-processing signature was checked against the installed transformers 5.x. Grounding DINO is
  the least-tested wrapper, so watch it on the first laptop run.
- **Speed:** all five forensic maps take ≈0.12 s per 640×480 image on CPU.
- **Method note:** hand-crafted maps have no natural "AI" threshold. Task 1 records their behaviour
  on real photos; the false-alarm rate is computed in Task 3 at a threshold fixed on validation data
  with AI objects (TPR 80%). This was stated to the author.
- **Computer use:** the author offered control of their laptop, but no desktop-control tools are
  connected to this session. The workflow stays the same: code is written here and the author runs
  the commands.
- **Next:** the author runs Task 1 on the laptop and pastes `summary.md`, then reviews
  `audit.html`.

---

## 2026-10-08: Work split into tasks; Task 1 = real-only COCO baseline

- **Decision (author):** split the work. Task 1 runs only on real COCO photos, with no AI generation,
  to establish how detectors and forensic methods behave on genuine photographs first.
- **Interpretation recorded:** COCO (Flickr, about 2014) predates diffusion editors, so its photos
  are real by provenance. Forensic methods cannot *prove* an image is non-AI. What Task 1 measures is
  each method's **false-alarm rate on real photos**, plus detector AP on C0 vs C1 (the resave
  effect) and the label audit.
- **Changed:** `configs/real_baseline.yaml` (C0 and C1 only, no generators; same work dir, so Task 2
  reuses the images). Fixed `generators_enabled: []` (it previously fell back to *all* generators).
  New test confirms that a real-only run contains no AI records. PRD timeline reordered (Task 1 → 2 →
  3). PAPER §7.1 placeholder for the real-photo baseline.
- **Next:** implement the `detect` stage (4 detectors) and the hand-crafted forensic methods so that
  Task 1 can run on the laptop.

---

## 2026-10-08: Generation pipeline implemented (week 1)

- **Hardware recorded:** NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB (from the author's `nvidia-smi`).
  This matches the 8 GB plan, so no config change was needed beyond recording it.
- **Implemented:** stages `sample`, `plan`, `generate`, `preview`; COCO index; encoding policy;
  local inpainting + feathered compositing; real-object paste with donor selection; diffusers
  wrappers for SDXL / SD2 / Kandinsky / FLUX Fill (lazy, 8 GB defaults); run provenance (`runlog`).
- **Design decisions:**
  - Rectangular dilated mask with a linear feather that is exactly 0 at the mask border. Why: it
    guarantees that pixels outside the intended mask are unchanged, so any detector or forensic
    difference outside the edit is a JPEG-block effect we can measure, not generator leakage.
  - G3 (held-out FLUX) only on test-split sources (~120). Why: a held-out generator must never be
    seen in train/val. The earlier "300 sources" figure was inconsistent with a 20% test split.
  - The insertion size is sampled from real COCO instances of the category, excluding those under 1%
    of the image area. Why: tiny inserts rarely produce a recognisable object.
- **Verified on CPU** (fake inpainter, synthetic COCO): 18 tests pass. They check that the policy
  JPEGs are q95 4:4:4, that C4 decodes identically to C1 outside the mask (±1 JPEG block), that every
  condition is produced, and that a second run is a no-op (resumable).
- **No experimental results yet.**
- **Next:** the author runs the 50-source pilot with SDXL on the laptop and checks `preview.html`.
  Then the `verify` stage (OWLv2 + SAM) and `audit`.

---

## 2026-10-08: Added held-out source domain (decision)

- **What:** The author asked for a small second-dataset test in week 4.
- **Decision:** About 100 **Open Images V7** validation photos, test-only, conditions C1–C4, generators
  G1 (seen) and G3 (held out). Every model and threshold is fit on COCO and transferred unchanged.
- **Why Open Images, not Objects365:** Grounding DINO is pretrained on Objects365, so Objects365 would
  not be unseen for one of our detectors. LVIS reuses COCO images.
- **Caveat recorded:** Open Images boxes are exhaustive only for image-level-verified classes. We score
  verified classes only; the rest become ignore regions.
- **Changed:** PRD (scope, risks, week 4), ARCHITECTURE (adapter stage, ledger `source_dataset`), config
  `ood:` block, ledger code + tests, PAPER §4/§7.4/§8.

---

## 2026-10-08: Project setup (week 1, day 1)

- **What:** Read the professor's InlayFake brief. Scope agreed with the author: both detector study and
  forensic origin detection ("traditional" = off-the-shelf detectors **and** hand-crafted forensics,
  plus learned forensic baselines). Target: CVPR 2027 (submission 16 Nov 2026 AoE per official CFP
  via search; confirm, including the registration date).
- **Decisions:**
  - 5-week plan: benchmark + baseline study (the brief's suggested "constrained first paper"). Stage 2
    (VLM captions) reduced to P1.
  - Source pool = COCO val2017 (unseen by COCO-trained detectors).
  - Generators: SDXL-Inpainting, SD-2-Inpainting (seen); FLUX.1-Fill-dev (held out).
  - Verifier for insertions = OWLv2, excluded from the evaluated detectors to avoid selection bias.
  - Identical JPEG q=95 encoding for C1–C5 to remove format shortcuts.
- **Created:** AGENTS.md, CLAUDE.md, PRD.md, ARCHITECTURE.md, paper/PAPER.md, config and ledger
  schema.
- **Open:** laptop GPU model and VRAM; professor sign-off on the scope.
- **Next:** COCO download + `sample` and `plan` stages; generator wrapper for G1; 50-source pilot.
