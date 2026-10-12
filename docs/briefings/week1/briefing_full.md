# InlayFake: Week 1 progress (8–14 Oct 2026)

Briefing for the project professor. All numbers come from run `task1_real_baseline` (RTX 4070 laptop).

**Goal for the week:** a real-only baseline. Before generating any AI content, measure how object
detectors and forensic methods behave on genuine COCO photographs. COCO dates from about 2014, before
diffusion editors existed, so every object in it is photographed.

## Done

### 1. Infrastructure
- A research codebase with a staged pipeline: `sample` → `plan` → `generate` → `detect` →
  `forensics` → `audit` → `evaluate`.
- **Each stage is reproducible.** Every random choice uses a seed from the config, and each run
  records its git commit, config, model versions and package versions.
- **Each stage can be resumed** if it is interrupted.
- 38 unit tests run on the CPU without model weights.
- Hardware: RTX 4070 Laptop, 8 GB.

### 2. The benchmark's sources
- 600 photographs from COCO val2017, chosen because COCO-trained detectors never saw them during
  training.
- **Split by source photo:** train 362, val 114, test 124. All edits of one photo stay in the same
  split.
- **544 photos have an insertion plan** (a target category and a free location for the object). The
  other 56 (9.3%) had no free space for a new object, mostly because one large object fills the
  frame (see Key finding 4). That is a stated limitation: close-up, frame-filling shots are
  under-represented.
- **Two real conditions per photo:** C0 is the original file, and C1 is the photo saved again with the
  benchmark's own JPEG settings (quality 95, 4:4:4). That gives 1,088 images.

### 3. Detectors on real photos
Four families: two-stage, one-stage, transformer and open-vocabulary.

| Detector | AP C0 | AP C1 |
|---|---|---|
| Faster R-CNN v2 | 0.500 | 0.498 |
| YOLO11-m | 0.530 | 0.528 |
| RT-DETR-R50 | 0.512 | 0.511 |
| Grounding DINO-T | 0.515 | 0.515 |

**What "AP" means here:** the standard COCO detection metric, which is a *mean* average precision
(often written mAP). It is computed with `pycocotools` and averages precision over (1) all recall levels,
(2) the 80 COCO categories present in the images and (3) ten box-overlap (IoU) thresholds from 0.50 to
0.95. AP50 is the same average at the single IoU threshold 0.50. It is not "accuracy": a score of 0.50
does not mean that half of the objects were found. C0 = original photo, C1 = matched resave.

- These values are in the expected range for these models on COCO, which confirms the setup and
  evaluation are correct.
- The confidence intervals are 95% bootstraps over source photos with 200 resamples (Table 2a in the
  paper draft).
- Grounding DINO gets the same fixed text prompt for every image (all 80 COCO class names), so nothing
  about the edits leaks into its input.

### 4. Hand-crafted forensics on real photos
Five methods: error level analysis (ELA), SRM noise residual, wavelet noise level, FFT
high-frequency, and JPEG blocking grid.

### 5. Label audit
- At least two detectors agreed on a confident object that COCO does not label in 107 places, across
  70 photos (12.9%).
- Every one was reviewed by eye: 77 added as missing labels, 26 marked ignore (ambiguous), 4 rejected
  as detector errors (for example, deer detected as sheep).
- Adding the confirmed labels raised every detector's AP by 0.004–0.006. Unaudited COCO labels
  slightly understate detector accuracy.

## Key findings so far

1. **Resaving barely changes detection.** AP moves by at most 0.002, but 0.4–0.8% of individual
   objects flip between detected and missed. That flip rate is the noise floor for H2: any effect of
   insertions on nearby real objects must clearly exceed it.
2. **Every forensic method reacts to resaving.** Scores shift by −4.7% to +3.7%, and the confidence
   intervals exclude zero. So edited images must be compared against the resaved C1, not the original
   C0. This validates the matched-control design.
3. **Real objects already look "suspicious" to three methods.** ELA, SRM and noise score real objects
   higher than their surroundings, because objects have more texture and edges. A naive "this object
   stands out, so it is fake" rule would falsely accuse real objects. Origin scores have to be
   calibrated against real objects. FFT and blocking-grid show no such bias, which is worth watching
   once AI objects exist.
4. **The 56 photos without an insertion plan are close-ups, not busy scenes.** They have the same
   median number of labelled objects as the 544 kept photos (4), but their object boxes cover a
   median of 97% of the frame, against 32% for the kept photos. In 52 of the 56, one object fills
   more than half the picture, most often a dining table (18), a bowl (6), a bed (5) or a person (5).
   So the benchmark under-represents close-up, frame-filling shots rather than crowded scenes. By
   split: 39 train, 7 val, 10 test.

<details>
<summary>The 56 photos (click to expand). Each ID links to the image on the COCO website.</summary>

Sorted by box coverage. "Coverage" is the share of the image covered by the union of labelled
object boxes.

| COCO image ID | Split | Labelled objects | Coverage | Most common labels |
|---|---|---|---|---|
| [253452](http://images.cocodataset.org/val2017/000000253452.jpg) | train | 15 | 100% | donut ×6, knife ×2, carrot ×2 |
| [557916](http://images.cocodataset.org/val2017/000000557916.jpg) | test | 4 | 100% | bed, person, teddy bear |
| [78170](http://images.cocodataset.org/val2017/000000078170.jpg) | train | 10 | 100% | handbag ×3, cell phone ×2, person ×2 |
| [72795](http://images.cocodataset.org/val2017/000000072795.jpg) | train | 15 | 100% | person ×13, baseball bat, baseball glove |
| [527695](http://images.cocodataset.org/val2017/000000527695.jpg) | train | 9 | 100% | broccoli ×7, dining table, bowl |
| [261097](http://images.cocodataset.org/val2017/000000261097.jpg) | train | 27 | 100% | person ×14, bench ×11, tennis racket |
| [256916](http://images.cocodataset.org/val2017/000000256916.jpg) | test | 9 | 99% | pizza ×4, chair ×2, knife ×2 |
| [493799](http://images.cocodataset.org/val2017/000000493799.jpg) | train | 6 | 99% | carrot ×4, dining table, fork |
| [157390](http://images.cocodataset.org/val2017/000000157390.jpg) | train | 23 | 99% | carrot ×14, broccoli ×7, bowl |
| [304545](http://images.cocodataset.org/val2017/000000304545.jpg) | train | 1 | 99% | suitcase |
| [96493](http://images.cocodataset.org/val2017/000000096493.jpg) | train | 3 | 99% | couch, person, remote |
| [61960](http://images.cocodataset.org/val2017/000000061960.jpg) | test | 4 | 99% | bed ×3, person |
| [340015](http://images.cocodataset.org/val2017/000000340015.jpg) | test | 2 | 99% | dining table, pizza |
| [23272](http://images.cocodataset.org/val2017/000000023272.jpg) | test | 2 | 98% | cat, car |
| [513041](http://images.cocodataset.org/val2017/000000513041.jpg) | train | 10 | 98% | wine glass ×4, person ×3, dining table |
| [523175](http://images.cocodataset.org/val2017/000000523175.jpg) | train | 6 | 98% | carrot ×3, spoon, bowl |
| [389933](http://images.cocodataset.org/val2017/000000389933.jpg) | train | 2 | 98% | dog, couch |
| [143931](http://images.cocodataset.org/val2017/000000143931.jpg) | train | 2 | 98% | bus, person |
| [188906](http://images.cocodataset.org/val2017/000000188906.jpg) | train | 1 | 98% | banana |
| [348708](http://images.cocodataset.org/val2017/000000348708.jpg) | train | 3 | 98% | bowl, banana, dining table |
| [2592](http://images.cocodataset.org/val2017/000000002592.jpg) | train | 3 | 98% | dining table, cup, knife |
| [119452](http://images.cocodataset.org/val2017/000000119452.jpg) | train | 6 | 98% | banana ×2, orange ×2, apple ×2 |
| [308430](http://images.cocodataset.org/val2017/000000308430.jpg) | test | 18 | 98% | carrot ×14, bowl, book |
| [382030](http://images.cocodataset.org/val2017/000000382030.jpg) | train | 4 | 98% | umbrella, potted plant, dining table |
| [18575](http://images.cocodataset.org/val2017/000000018575.jpg) | val | 8 | 98% | bottle ×3, dining table, knife |
| [139260](http://images.cocodataset.org/val2017/000000139260.jpg) | train | 6 | 97% | banana ×3, spoon, bowl |
| [244833](http://images.cocodataset.org/val2017/000000244833.jpg) | train | 21 | 97% | chair ×14, person ×4, motorcycle ×2 |
| [273642](http://images.cocodataset.org/val2017/000000273642.jpg) | train | 2 | 97% | dog, remote |
| [222299](http://images.cocodataset.org/val2017/000000222299.jpg) | train | 9 | 97% | bottle ×3, book ×3, cell phone |
| [491757](http://images.cocodataset.org/val2017/000000491757.jpg) | val | 2 | 97% | cat, bed |
| [378284](http://images.cocodataset.org/val2017/000000378284.jpg) | train | 7 | 96% | person ×2, knife ×2, fork |
| [155179](http://images.cocodataset.org/val2017/000000155179.jpg) | train | 2 | 96% | cake, dining table |
| [355325](http://images.cocodataset.org/val2017/000000355325.jpg) | val | 11 | 96% | dining table ×2, wine glass ×2, spoon ×2 |
| [541634](http://images.cocodataset.org/val2017/000000541634.jpg) | val | 11 | 95% | cup ×3, broccoli ×3, spoon ×2 |
| [312192](http://images.cocodataset.org/val2017/000000312192.jpg) | train | 11 | 95% | wine glass ×3, bottle ×2, fork ×2 |
| [530052](http://images.cocodataset.org/val2017/000000530052.jpg) | train | 14 | 95% | carrot ×14 |
| [261116](http://images.cocodataset.org/val2017/000000261116.jpg) | test | 11 | 94% | cake ×5, fork ×2, person ×2 |
| [66926](http://images.cocodataset.org/val2017/000000066926.jpg) | train | 12 | 94% | donut ×12 |
| [530466](http://images.cocodataset.org/val2017/000000530466.jpg) | train | 3 | 94% | chair, train, tv |
| [485130](http://images.cocodataset.org/val2017/000000485130.jpg) | train | 2 | 93% | cake, dining table |
| [312213](http://images.cocodataset.org/val2017/000000312213.jpg) | val | 2 | 91% | cat, laptop |
| [486112](http://images.cocodataset.org/val2017/000000486112.jpg) | test | 2 | 91% | person, elephant |
| [416104](http://images.cocodataset.org/val2017/000000416104.jpg) | train | 36 | 90% | person ×11, chair ×10, bottle ×8 |
| [421455](http://images.cocodataset.org/val2017/000000421455.jpg) | test | 5 | 88% | dog, cell phone, person |
| [434479](http://images.cocodataset.org/val2017/000000434479.jpg) | val | 4 | 88% | pizza ×2, dining table, chair |
| [458255](http://images.cocodataset.org/val2017/000000458255.jpg) | train | 5 | 87% | bed ×2, cat, person |
| [346905](http://images.cocodataset.org/val2017/000000346905.jpg) | train | 3 | 87% | elephant ×3 |
| [464786](http://images.cocodataset.org/val2017/000000464786.jpg) | train | 2 | 86% | hot dog, person |
| [53909](http://images.cocodataset.org/val2017/000000053909.jpg) | val | 2 | 86% | cell phone, person |
| [455267](http://images.cocodataset.org/val2017/000000455267.jpg) | train | 15 | 83% | broccoli ×9, carrot ×5, oven |
| [66135](http://images.cocodataset.org/val2017/000000066135.jpg) | train | 1 | 82% | train |
| [303566](http://images.cocodataset.org/val2017/000000303566.jpg) | train | 56 | 81% | person ×14, apple ×14, orange ×14 |
| [338428](http://images.cocodataset.org/val2017/000000338428.jpg) | test | 4 | 79% | truck ×3, bus |
| [125850](http://images.cocodataset.org/val2017/000000125850.jpg) | train | 2 | 77% | cat, bowl |
| [537802](http://images.cocodataset.org/val2017/000000537802.jpg) | train | 2 | 74% | teddy bear ×2 |
| [76731](http://images.cocodataset.org/val2017/000000076731.jpg) | train | 1 | 67% | laptop |

</details>

5. **All 544 photos used, and what each detector found in them.** For every photo:
   its split, the category planned for insertion, its labelled objects, and how many of them each of
   the four detectors found (same class, IoU ≥ 0.5, score ≥ 0.5). Full table:
   [images_used.md](images_used.md). Of the 3,959 labelled objects (COCO labels plus audit
   additions), the detectors found: Faster R-CNN 2,938 (recall 0.742), RT-DETR 2,387 (0.603),
   YOLO11-m 2,053 (0.519), Grounding DINO 1,826 (0.461). These equal recall C0 in run
   `task1_real_baseline`. Recall at a fixed score ≥ 0.5 ranks the detectors differently from AP,
   because AP averages over all score thresholds.


## Also done (ahead of schedule)

- The AI-insertion pipeline is already written: inpainting wrappers for SDXL, SD2 and FLUX Fill,
  compositing, and the real-object-paste control condition.
- A held-out source domain is planned: about 100 Open Images photos for the week-4 test, all
  test-only.
- Paper draft: sections 1–6 and 8 are drafted, and §7.1 (baseline results) is filled in.

## Next week (Task 2: AI insertion)

- A 50-photo SDXL pilot with a visual check.
- Build the verification stage: OWLv2 checks that each inserted object is visible, and SAM produces
  its mask.
- A full overnight generation run: two seen generators, plus FLUX held out on the test split only.
- A human audit of 10% of the edits.
- Start the Open Images download.

## To discuss

- The VLM caption study (Stage 2, H5) stays a must-have: plain and grounded prompts in week 3,
  origin-aware prompts and caption evaluation in week 4. Confirm this plan.
- Confirm the CVPR 2027 registration and submission dates.
- Whether excluding the 9% of photos with no free space (mostly close-ups) is acceptable as a stated
  limitation.
