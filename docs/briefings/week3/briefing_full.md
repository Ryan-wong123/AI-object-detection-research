# InlayFake: Week 3 progress (22–28 Oct 2026)

Last updated 26 Oct 2026 · Ryan Wong

Briefing for the project professor. Branch `week-3-detection`. GPU stages run on the RTX 4070
laptop (8 GB). Run `task3_detection_natural` (config `configs/task3_detection_natural.yaml`).

**Goal for the week (PRD §8, Task 3):** run the four detectors and the forensic methods on the
edited images (C2–C4) as well as the real ones (C0, C1); add the learned forensics TruFor and
UnivFD; compute the semantic and matched-instance metrics for H1 and H2; write the Stage 2 `vlm`
stage and run its plain and grounded prompts.

## The benchmark used (frozen 23 Oct)

Ledger `data/inlayfake_v4/ledger.jsonl` from the natural-placement run
`task2_insertion_natural_box`: 2,315 images. C0 518, C1 518, C2 425, C3 427, C4 427. 425 photos
have all five conditions (train/val/test 247/88/90). <!-- file:paper/results/task2_benchmark_table1.csv, task2_benchmark_summary.json -->
The author decided not to regenerate. The known limitations stay and are stated in the paper:
some boxes partly repaint a real object (87 C2 and 72 C4 images, labelled mixed/modified), C3
pastes do not update the labels they cover, and 117 photos have no AI object.

## Plan agreed with the author (23 Oct)

| Step | What | Where | Estimate |
|---|---|---|---|
| 1 | `detect` (4 detectors), then hand-crafted `forensics`, on all 2,315 images | GPU, then CPU, detached and resumable | ~1 h (from week-1 rates; measured time below) |
| 2 | Learned forensics TruFor and UnivFD: code + CPU tests, then the author's GPU run | GPU | ~1–2 h (estimate) |
| 3 | `vlm` stage (Florence-2, Qwen2.5-VL-3B 4-bit), plain and grounded prompts, ~200 test images + C1 controls | GPU, overnight | ~1–2 h (estimate) |
| 4 | `evaluate` with the semantic + matched-instance metrics (H1, H2); written during step 1 | CPU | — |

## Done

### 1. Detection and hand-crafted forensics complete (23 Oct, run `task3_detection_natural`)

`scripts/run_stages.py` now accepts a list of stages, so detection and forensics run one after
the other in separate processes with sleep blocked, and resume after an interruption. Started
23 Oct 16:47 (laptop time):

```
.venv\Scripts\python.exe scripts\run_stages.py configs\task3_detection_natural.yaml detect forensics
```

Progress: `data\inlayfake_v4\run_stages.log` (start/end per stage) and `run_progress.log`.

Both stages finished at 18:23 (exit 0). Every detector and every hand-crafted method has exactly
one result for each of the 2,315 images (none missing, no duplicates). The run stopped once, at
17:19 (Grounding DINO at 1,438 of 2,315), because it had been started as a child of the agent
app, which restarted for an update; there was no sleep or driver reset. It was restarted at 18:00
as an independent process and resumed from the next image without losing results. Lesson: start
long GPU runs from the author's own PowerShell window, not from the app.

### 2. H1/H2 metrics written and evaluated (25 Oct, run `task3_detection_natural`)

`evaluate` now computes, on the 425 photos with all five conditions: AP on original objects per
condition and on AI-inserted (C4) and pasted (C3) objects; for H1, each object's detector score
(best same-class detection) for the AI object against same-class photographed objects in the same
photo ("twins") and the real object pasted at the same box; for H2, every real object followed
from C1 into C0/C2/C3/C4 (detected ↔ missed flips, class flips, score change), split by whether
the edit box covers it, with paired C4 − C2 / C3 / C0 differences on the same objects. 1,000
grouped-bootstrap resamples over photos (200 for AP). The detectors are deterministic (identical
output on identical images in two runs), so any change is caused by the edit.

The run took 50 min (25 Oct). Two earlier attempts were lost: a console window was closed, and
Windows throttled the windowless run ~5× and then restarted for the 26H2 update. Each stage now
opts out of that throttling.

### 3. Figure 2: H2 by distance to the edit (25 Oct)

`scripts/figure_h2_distance.py` bins every real object by its distance to the edit
(`paper/figures/fig2_h2_distance.png`). Flips concentrate on and right next to the edit for every
detector. Faster R-CNN changes nothing beyond half a box length; Grounding DINO's extra flips
over the sham edit persist 1–4 box lengths away.

### 4. Learned forensics TruFor and UnivFD written (26 Oct)

Order decided with the author: learned forensics before the VLM stage, because all of week 4
(H3, crop classifier, joint AP, calibration, robustness) needs per-object origin scores, and
TruFor's own code and the 8 GB GPU are the riskiest unknowns. TruFor runs from the authors' code
and weights (non-commercial licence, kept out of git) and gives a pixel map plus an image score;
UnivFD scores the image and a crop around every object with CLIP ViT-L/14 and the authors' linear
probe. The CLIP features of every crop are saved for next week's crop classifier. Tested on CPU;
both then ran on all 2,315 images on 26 Oct in 7 minutes (TruFor's weights checked against the authors' checksum). Their origin scores (H3) are computed next week.

## Results

| Item | Value |
|---|---|
| Records loaded by the Task 3 config | 2,315 (checked 23 Oct) |
| `detect` wall time per detector (2,315 images) | Faster R-CNN 3:01 (12.8 img/s), YOLO11-m 0:39 (59 img/s), RT-DETR 1:32 (25 img/s), Grounding DINO 41:45 in two parts (26:14 for 1,440 + 15:31 for 877; 1.08 s/img); about 47 min of detector time plus model loading <!-- run:task3_detection_natural, file:data/inlayfake_v4/run_progress.log --> |
| TruFor / UnivFD wall time (2,315 images) | 4:16 (9.0 img/s) / 2:38 (14.6 img/s) <!-- run:task3_detection_natural, file:data/inlayfake_v4/run_progress.log --> |
| Hand-crafted `forensics` wall time | 5:05 for 2,315 images × 5 methods (7.6 img/s, CPU) <!-- run:task3_detection_natural, file:data/inlayfake_v4/run_progress.log --> |
| AP on original objects, C1 → C4 (425 photos) | Faster R-CNN 0.502 → 0.493, YOLO 0.536 → 0.528, RT-DETR 0.522 → 0.514, Grounding DINO 0.516 → 0.509 (drops inside CIs) <!-- run:task3_detection_natural, file:paper/results/task3_evaluate_summary.md --> |
| AP on AI-inserted (C4) / pasted real (C3) objects | 0.38–0.41 / 0.34–0.43 <!-- run:task3_detection_natural, file:paper/results/task3_evaluate_summary.md --> |
| Recall, AI object vs same-class photographed in the same photo (H1) | 6–9 points lower for AI objects (e.g. RT-DETR 0.522 vs 0.613; Grounding DINO's CI includes 0); vs the C3 paste at the same box: 0–8 points higher <!-- run:task3_detection_natural, file:paper/results/task3_evaluate_summary.md --> |
| H2 by distance: Grounding DINO C4 − C2 at 1–2 / 2–4 box lengths | +1.4 [0.7, 2.4] / +1.1 [0.4, 2.0] pp; Faster R-CNN 0–0.3% flips beyond half a box length <!-- run:task3_detection_natural, file:paper/results/task3_h2_by_distance.csv --> |
| Origin AUROC of detector confidence (H1) | vs same-class photographed 0.41–0.45; vs C3 paste 0.51–0.57 (0.5 = chance) <!-- run:task3_detection_natural, file:paper/results/task3_evaluate_summary.md --> |
| Flip rate of uncovered real objects C1 → C4; excess over C2 sham (H2) | Faster R-CNN 0.20% (−0.04 pp, n.s.), YOLO 0.68% (+0.28 pp, n.s.), RT-DETR 1.27% (+0.52 pp [0.08, 0.97]), Grounding DINO 1.67% (+0.96 pp [0.45, 1.48]); re-encoding the photo (C0 vs C1) flips 1.0–1.2% <!-- run:task3_detection_natural, file:paper/results/task3_evaluate_summary.md --> |

## Next steps

1. Write the `vlm` stage and caption metrics; plain and grounded prompts on the GPU.
2. The author's audit of C4 edits can continue in parallel. A rejected edit means regenerating it,
   then re-running detection and forensics for the changed images only.

## To discuss

- **H1 reading.** Detector confidence is a weak origin signal (AUROC 0.41–0.45), and its sign
  flips with the comparison: AI objects score below the photo's own same-class objects but above
  real objects pasted into the same box. Is "compositing, not generation, explains the gap" the
  right message? The verifier (OWLv2) passed every AI object, which favours AI recall.
- **H2 reading.** A small excess effect over the sham edit only for the transformer detectors
  (+0.5 / +1.0 percentage points), no larger than one JPEG re-encode of the whole photo. Report as
  "partly supported"?

- C3 pastes do not update the labels of the real objects they cover. The H2 contrast with C3
  must therefore leave those covered objects out (they are the same set as the box overlaps).
