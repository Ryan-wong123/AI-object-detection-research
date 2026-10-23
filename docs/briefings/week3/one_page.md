# InlayFake: Week 3 progress (22–28 Oct 2026)

23 Oct 2026 · Ryan Wong

Week 3 measures how detectors and forensic methods behave on the edited images. The benchmark
was frozen on 23 Oct: 2,315 images (C0 518, C1 518, C2 425, C3 427, C4 427; 425 photos with all
five conditions) from the natural-placement run. The four detectors and the hand-crafted
forensics ran on all of them on 23 Oct (run `task3_detection_natural`): about 47 min of detector time (Grounding DINO 42 min of it) and 5 min of forensics; it stopped once when the agent app restarted and resumed without loss.

## Done this week

| Area | What was delivered |
|---|---|
| Benchmark | Frozen ledger `data/inlayfake_v4/ledger.jsonl`, 2,315 images; Task 3 config loads all of them |
| Plan | Agreed order: detection + hand-crafted forensics → TruFor/UnivFD → VLM prompts → evaluate |
| Runs | `detect` then `forensics` complete 23 Oct 18:23: 4 detectors × 2,315 images, 5 hand-crafted methods × 2,315 images |

## Results

| Item | Value |
|---|---|
| Detection / forensics wall time | Faster R-CNN 3:01, YOLO11-m 0:39, RT-DETR 1:32, Grounding DINO 41:45; forensics 5:05 <!-- run:task3_detection_natural --> |
| Detector AP on originals per condition (H2) | [[TBD]] |
| Recall and confidence AUROC, AI vs photographed (H1) | [[TBD]] |

## Next week and points to discuss

- Next: H1/H2 metrics in `evaluate`; TruFor and UnivFD; `vlm` stage with plain and grounded prompts.
- To discuss: C3 pastes do not update the labels they cover, so H2 leaves the covered objects out of the C3 contrast.
