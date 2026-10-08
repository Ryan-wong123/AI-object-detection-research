# Research log

Dated record of every experiment, decision and finding. Newest entries at the top.
Format: date · run ID (if any) · what · key numbers · interpretation · next step.

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
