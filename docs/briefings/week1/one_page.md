# InlayFake: Week 1 progress (8–14 Oct 2026)

11 Oct 2026 · Ryan Wong

Week 1 delivered the real-only baseline: four detectors and five forensic methods measured on 544
untouched COCO photographs and their matched resaves, before any AI content is generated. All numbers
come from run `task1_real_baseline`.

## Done this week

| Area | What was delivered |
| --- | --- |
| Pipeline | Staged, resumable code: sample → plan → generate → detect → forensics → audit → evaluate. Seeded and logged per run; 37 CPU unit tests. |
| Sources | 600 COCO val2017 photos, split by photo (train 362 / val 114 / test 124). 544 have an insertion plan; 56 (9.3%) were too cluttered. |
| Real conditions | C0 = original file, C1 = matched resave (JPEG q95, 4:4:4): 1,088 images. |
| Detectors | Faster R-CNN v2, YOLO11-m, RT-DETR-R50, Grounding DINO-T (fixed prompt of all 80 COCO names). |
| Forensics | ELA, SRM residual, wavelet noise level, FFT high-frequency, JPEG blocking grid. |
| Label audit | 107 objects missing from COCO flagged in 70 photos (12.9%); 77 added, 26 ignored, 4 rejected after review by eye. |

## Detector AP on real photos

All four detectors land at AP 0.50–0.53, the expected range on COCO, which confirms the setup and
evaluation are correct.

| Detector | AP C0 | AP C1 |
| --- | --- | --- |
| YOLO11-m | 0.530 | 0.528 |
| Grounding DINO-T | 0.515 | 0.515 |
| RT-DETR-R50 | 0.512 | 0.511 |
| Faster R-CNN v2 | 0.500 | 0.498 |

Ground truth includes the 77 audit-added labels. 95% confidence intervals (bootstrap over source
photos, 200 resamples) are in Table 2a of the paper draft.

## Key findings

1. **Resaving barely changes detection.** AP moves by at most 0.002, yet 0.4–0.8% of objects flip
   between detected and missed. This flip rate is the noise floor any insertion effect on real objects
   (H2) must exceed.
2. **Every forensic method reacts to resaving.** Scores shift by −4.7% to +3.7%, with confidence
   intervals excluding zero. Edited images must be compared with the resave C1, not the original C0,
   which validates the matched-control design.
3. **Real objects already look "suspicious" to ELA, SRM and noise level.** Objects carry more texture
   than their surroundings, so a naive "stands out = fake" rule would accuse real objects. FFT and
   blocking grid show no such bias.
4. **COCO labels are incomplete.** Adding the 77 confirmed labels raises every detector's AP by
   0.004–0.006.

## Fixed along the way, and done early

- **Fixed:** a Grounding DINO crash (labels now read from the model's token scores), blocking-grid
  outliers from near-zero division (now a log ratio), and upward-biased AP intervals (now the basic
  bootstrap). Each fix has a unit test.
- **Ahead of schedule:** the AI-insertion pipeline (SDXL, SD2 and FLUX Fill inpainting, compositing,
  real-object paste control) is written; a held-out Open Images test domain is planned; paper sections
  1–6 and 8 are drafted and §7.1 holds these results.

## Next week and points to discuss

**Week 2 (Task 2, AI insertion):** a 50-photo SDXL pilot with a visual check; the verification stage
(OWLv2 confirms each inserted object, SAM gives its mask); a full overnight generation run with FLUX
held out on the test split; a human audit of 10% of edits; start the Open Images download.

**To discuss:**

- Whether excluding the 9% most cluttered photos is acceptable as a stated limitation.
