# InlayFake: Week 3 progress (22–28 Oct 2026)

Last updated 23 Oct 2026 · Ryan Wong

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

## Results

| Item | Value |
|---|---|
| Records loaded by the Task 3 config | 2,315 (checked 23 Oct) |
| `detect` wall time per detector (2,315 images) | Faster R-CNN 3:01 (12.8 img/s), YOLO11-m 0:39 (59 img/s), RT-DETR 1:32 (25 img/s), Grounding DINO 41:45 in two parts (26:14 for 1,440 + 15:31 for 877; 1.08 s/img); about 47 min of detector time plus model loading <!-- run:task3_detection_natural, file:data/inlayfake_v4/run_progress.log --> |
| Hand-crafted `forensics` wall time | 5:05 for 2,315 images × 5 methods (7.6 img/s, CPU) <!-- run:task3_detection_natural, file:data/inlayfake_v4/run_progress.log --> |
| Detector AP on original objects, per condition C0–C4 (H2) | [[TBD]] |
| Recall on AI-inserted vs photographed objects of the same class (H1) | [[TBD]] |
| Origin AUROC of detector confidence (H1) | [[TBD]] |
| Matched-instance change rates C4 vs C1/C2/C3 on uncovered originals (H2) | [[TBD]] |

## Next steps

1. Write the H1/H2 metrics in `evaluate`, then run it on these detections.
2. Write TruFor and UnivFD (CPU tests with fakes); give the author the download and run commands.
3. Write the `vlm` stage and caption metrics; plain and grounded prompts on the GPU.
4. The author's audit of C4 edits can continue in parallel. A rejected edit means regenerating it,
   then re-running detection and forensics for the changed images only.

## To discuss

- C3 pastes do not update the labels of the real objects they cover. The H2 contrast with C3
  must therefore leave those covered objects out (they are the same set as the box overlaps).
