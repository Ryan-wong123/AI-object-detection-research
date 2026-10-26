# InlayFake: Week 3 progress (22–28 Oct 2026)

26 Oct 2026 · Ryan Wong

Week 3 measures how detectors and forensic methods behave on the edited images. The benchmark
was frozen on 23 Oct: 2,315 images (C0 518, C1 518, C2 425, C3 427, C4 427; 425 photos with all
five conditions) from the natural-placement run. The four detectors and the hand-crafted
forensics ran on all of them on 23 Oct (run `task3_detection_natural`): about 47 min of detector time (Grounding DINO 42 min of it) and 5 min of forensics; it stopped once when the agent app restarted and resumed without loss. The H1/H2 metrics ran on 25 Oct: detector confidence is a weak origin signal (H1), and AI insertions change predictions on untouched real objects only slightly more than a sham edit, and only for the two transformer detectors (H2).

## Done this week

| Area | What was delivered |
|---|---|
| Benchmark | Frozen ledger `data/inlayfake_v4/ledger.jsonl`, 2,315 images; Task 3 config loads all of them |
| Plan | Agreed order: detection + hand-crafted forensics → TruFor/UnivFD → VLM prompts → evaluate |
| Runs | `detect` then `forensics` complete 23 Oct 18:23: 4 detectors × 2,315 images, 5 hand-crafted methods × 2,315 images |
| Figure 2 | H2 flip rates by distance to the edit (`paper/figures/fig2_h2_distance.png`) |
| Learned forensics | TruFor and UnivFD written, tested and run on all 2,315 images (26 Oct, 7 min) |
| Metrics | `evaluate`: subset AP, H1 origin (recall, AUROC), H2 matched-instance changes with paired contrasts; run 25 Oct (50 min) |

## Results

| Item | Value |
|---|---|
| Detection / forensics wall time | Faster R-CNN 3:01, YOLO11-m 0:39, RT-DETR 1:32, Grounding DINO 41:45; forensics 5:05 <!-- run:task3_detection_natural --> |
| AP on original objects C1 → C4 | drops of 0.007–0.009, inside CIs <!-- run:task3_detection_natural, file:paper/results/task3_evaluate_summary.md --> |
| H1: confidence AUROC, AI vs same-class photographed / vs pasted | 0.41–0.45 / 0.51–0.57 <!-- run:task3_detection_natural, file:paper/results/task3_evaluate_summary.md --> |
| H2 by distance | local for Faster R-CNN (nothing beyond half a box length); Grounding DINO's excess persists 1–4 box lengths away <!-- run:task3_detection_natural, file:paper/results/task3_h2_by_distance.csv --> |
| H2: extra flips of untouched real objects, C4 − C2 sham | RT-DETR +0.5 pp, Grounding DINO +1.0 pp (significant); CNNs n.s.; re-encode alone 1.0–1.2% <!-- run:task3_detection_natural, file:paper/results/task3_evaluate_summary.md --> |

## Next week and points to discuss

- Next: `vlm` stage with plain and grounded prompts.
- To discuss: H1 sign flips with the comparison group (compositing, not generation?); H2 "partly supported".
- To discuss: C3 pastes do not update the labels they cover, so H2 leaves the covered objects out of the C3 contrast.
