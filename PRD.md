# PRD: InlayFake (research product requirements)

> Terminology: C4-occ = the occlusion placement rule of the AI insertion before 18 Oct 2026 (cover 20–50% of one real object); since then it is part of the single AI condition C4.

| | |
|---|---|
| Owner | Ryan Wong |
| Target | CVPR 2027 main track: paper submission **16 Nov 2026 23:59 AoE** (per the official call for papers as found by search; paper registration is typically about 1 week earlier; **confirm both on cvpr.thecvf.com**) |
| Status | Week 1 complete (run `task1_real_baseline`). Week 2 in progress: SDXL pilots done; full SDXL insertion run `task2_insertion` on the 544 week-1 photos running since 17 Oct; scope reduced to SDXL only (17 Oct) |

## 1. Problem

Image editors can now insert photorealistic AI-generated objects into real photographs. Existing work
detects AI images at the **image level** (real or fake) or the **pixel level** (manipulation masks).
Object detectors answer *what* and *where*, but not *where the object came from*. No standard
evaluation asks, for **each object instance** in an edited photo: is it photographed or AI-inserted,
how confident should we be, and did the insertion disturb recognition of the real objects around it?

A detector calling an inserted chair "chair" is **correct**: the chair is visible. The missing
capability is **object-origin estimation**.

## 2. Research questions and hypotheses

| ID | Hypothesis (each can fail; report it either way) | Primary measurement |
|---|---|---|
| H1 | High semantic detection confidence does **not** imply origin discrimination: detector confidence and features separate AI-inserted from photographed objects poorly. | Origin AUROC of detector confidence; recall parity between origin groups |
| H2 | Local insertions change detector predictions on **untouched original objects** beyond what matched processing (resave, sham edit, real paste) and occlusion explain. | Matched-instance change rates C4 vs C1/C2/C3, on originals the AI object does not cover (merged C4, 2026-10-18: covered objects analysed separately, by visible fraction) |
| H3 | Per-object forensic evidence (instance-conditioned scoring) beats image-level and naive region-pooled scoring for object origin, with fewer false accusations of real objects. | Origin AUROC/AUPRC, FPR on real objects at fixed TPR, joint origin-aware AP |
| H4 | Any origin-detection gain survives post-processing (JPEG, resize, blur) and a **held-out source domain** (Open Images). *Held-out generator axis removed 2026-10-17 (SDXL only, see §3).* | Same metrics on held-out splits; degradation curves |
| H5 | Grounding origin evidence into VLM captions reduces false origin claims compared with direct prompting. | Caption origin error rates (Stage 2) |

## 3. Scope

### In scope (must-have for the submission, P0)
1. **InlayFake benchmark v1**: about 600 COCO val2017 source photos, same-class insertions (see §4), 6 matched
   conditions, one inpainting generator (SDXL; scope reduced 2026-10-17 from 3 generators, see
   "Scope change" below). Includes a per-instance provenance ledger
   and an audited sample.
2. **Stage 1: detector study**: 4 frozen detector families on all conditions; semantic AP per origin
   subset; matched-instance analysis (H1, H2).
3. **Origin-detection baselines**:
   - *Hand-crafted ("traditional") forensics*: Error Level Analysis, noise-residual (SRM) variance,
     local noise-level inconsistency, frequency (FFT) spectral energy, JPEG ghost / blocking-grid.
   - *Learned forensics*: TruFor (pixel localisation), UnivFD-style CLIP probe (image level, applied
     to crops), and our trained per-object crop classifier.
   - Each one is scored per instance in three ways: image-level, region-pooled, and
     instance-conditioned (object vs. context contrast).
4. **Detector + forensics composition**: the joint system (find the object, then score its origin),
   evaluated with joint origin-aware AP, calibration (ECE, Brier) and risk–coverage with abstention.
5. **Robustness**: JPEG q ∈ {95, 75, 50}, 0.5× resize, Gaussian blur σ=1 (held-out generator
   removed 2026-10-17).
6. **Held-out source domain (week 4)**: about 100 photos from **Open Images V7** (validation or test
   split), used **only for testing**. Conditions C1, C2, C3, C4 with G1 (SDXL). No
   training, calibration or threshold tuning happens on it: every model and threshold is fit on COCO
   and then transferred. This answers "do the findings hold outside COCO?" (H4, source-domain axis).
   Open Images is chosen over Objects365 because Grounding DINO was pretrained on Objects365.
7. **Stage 2: VLM origin claims (H5)**: Florence-2 and Qwen2.5-VL-3B with three prompt settings
   (plain, grounded, origin-aware) on about 200 test images, plus unedited controls. Structured JSON
   output, scored for false-AI and false-real claims.
8. Paper draft in `paper/PAPER.md`, converted to the CVPR LaTeX template in week 5.

### Should-have (P1)
- Human audit of 10% of edits (insertion success, unintended changes). **Extended (author,
  2026-10-18): every C4 AI insertion is audited**; the C4s of the random 10% draw are flagged
  (`in_random_sample`) for an unbiased estimate of the verifier's error rate. C2 shams are not
  audited (author); their check rests on the verifier alone.

### Out of scope (future work, stated in the paper)
- A new trained architecture beyond the crop classifier; FakeShield/SIDA re-runs (7B+ LMMs, do not fit
  the laptop: compare qualitatively and cite); faces and people as insertion targets; video; C2PA /
  credential tracks.

## 4. Benchmark specification (summary; details in ARCHITECTURE.md)

| Code | Condition | Purpose |
|---|---|---|
| C0 | Untouched source (original bytes) | Label audit, baseline |
| C1 | Matched resave (same decode → encode as all edited images) | Encoding control; the reference for every contrast |
| C2 | Sham edit: same mask, same generator, "empty background" prompt | Editor side effects without a new object |
| C3 | Real-object paste: same-class COCO instance from another photo, same box | Compositing artifacts without AI origin |
| C4 | **AI insertion** (merged 2026-10-18): one AI object of a class already in the photo, placed anywhere | Origin and collateral effects |

- **Merged AI condition (scope change, author, 2026-10-18):** the free-space rule (C4) and the occlusion rule (C4-occ) become **one AI
  condition** (kept under the code C4). Its only placement rule is that the class is already in
  the photo; the box may go anywhere. Per photo: a verified C4 (free space) is kept; otherwise
  the photo's verified C4-occ (covers 20–50% of one object) is used; otherwise a new box of one of
  the photo's classes is placed anywhere (no overlap rule; up to 6 boxes × 3 seeds, OWLv2 must
  still confirm the object). Photos with both a C4 and a C4-occ use the C4 (author's choice: their
  C2/C3 already exist), so those occluding edits leave the benchmark (kept on disk). C2 and C3 always sit at
  the chosen AI box. The ledger records where each AI edit came from (`merged_source`: C4, C4-occ,
  anywhere). Reason: after 8 rounds, 44 photos still had no AI object under the separate C4/C4-occ
  rules. Consequence: H2 no longer has a dedicated occlusion condition; occlusion effects are
  analysed by splitting the merged AI edits into those that cover a real object and those that
  do not (from the verified masks), which is observational, not a controlled contrast.

- **Inserted class (changed 2026-10-16, author decision):** each photo gets an AI object of a
  class that is **already in that photo's audited week-1 ground truth** (COCO labels plus audit
  "add" rows), any COCO class except *person*. Every edited photo therefore holds a photographed
  and an AI instance of the same class, which matches the origin comparison (H1, H3) within a
  class. Among the classes present, the least-used so far is chosen. Photos whose only
  non-person classes cannot fit a box get no insertion. The earlier design (10 fixed categories:
  chair, cup, bottle, potted plant, vase, clock, bench, umbrella, backpack, dog, cycled for
  balance) remains as `placement.class_choice: fixed` and produced the week-1 plan. Trade-off:
  classes are no longer balanced and include large or non-rigid ones (bus, giraffe, surfboard).
  **Exception (author, 2026-10-17):** a photo with no verified AI object of its own classes
  (after new boxes and class switches) may get one of the 10 original categories that is not in
  it; such edits carry `class_in_photo: false` and are excluded from same-class analyses. Once
  such an edit has passed it is kept, even if the photo later gains an in-photo AI object
  (author, 2026-10-18).
  **Second exception (author, 2026-10-17):** if no box satisfies the placement rules at all, a
  C4-occ box may be placed anywhere in the image with no overlap rule (`placement_rule: anywhere`),
  so every photo gets a box. C4 always keeps its no-occlusion rule.
  **C4-occ fill (author, 2026-10-18):** photos whose plan has no C4-occ box get one where the C4-occ rules
  allow it, with several box sizes tried and covered objects down to 32 px (originally 64 px);
  the other C4-occ rules are unchanged. **C4 fill (author, 2026-10-18):** photos without a C4 box
  get one where the unchanged C4 rules allow it (several box sizes, the photo's other classes
  if needed; 8 px margin kept).
- **Generator:** G1 SDXL-Inpainting only (used for training, calibration and test).
  **Scope change (author, 2026-10-17):** the originally planned G2 SD-2-Inpainting (second seen
  generator) and G3 FLUX.1-Fill-dev (held-out generator, fallback Kandinsky 2.2) are dropped for
  lack of time. Consequences: H4 keeps post-processing and the Open Images domain but loses the
  held-out-generator axis; the crop classifier (L3) is trained on SDXL edits only, so its results
  describe SDXL-generated objects; the paper states "one editor (SDXL)" and lists other generators
  as future work. The G2/G3 code and config entries remain for that future work.
- **Splits (by source image):** train 60% / val 20% / test 20%.
- **Held-out source domain:** about 100 Open Images V7 photos with instance masks for the mapped
  categories (Chair, Mug/Coffee cup, Bottle, Houseplant, Vase, Clock, Bench, Umbrella, Backpack, Dog).
  `source_dataset = "openimages"`, test split only. C4-occ is skipped (time), and C3 donors come from other
  Open Images photos.
- **Ledger:** one JSONL record per (source, condition, generator) with all instances and their origin,
  intended mask, actual changed mask, seed, prompt, model revision and file hashes.

## 5. Success criteria

- Benchmark: ≥ 85% verified insertion success in C4 (audited sample). Sham edits introduce a
  detectable new object in < 5% of cases.
- Every P0 table in the paper is filled from runs, with 95% bootstrap CIs.
- At least one clear, defensible finding for each of H1, H2, H4 and H5, whether positive or negative.
- The paper is complete in CVPR format.

## 6. Novelty positioning (what we may and may not claim)

- **May claim:** an object-level origin task with matched causal controls and a provenance ledger; a
  systematic instance-level comparison of hand-crafted vs learned forensics combined with detectors;
  evidence about collateral effects on real objects.
- **May not claim:** first local AI-forgery detector, first explainable forensic model, first
  object-insertion robustness test (prior work: OpenSDI, BR-Gen, FakeShield, SIDA, MetaOD, SOC, MIML,
  B-Free, TruFor).

## 7. Constraints and risks

| Risk | Impact | Mitigation |
|---|---|---|
| Laptop VRAM (assume 8 GB) | FLUX is slow or does not fit | Resolved 2026-10-17 by scope: SDXL only (no G2/G3) |
| Single generator | Findings may be SDXL-specific; no held-out-generator test | Stated as a limitation; post-processing and Open Images axes of H4 kept; other generators as future work |
| Inpainting often fails to insert the object (it fills background instead) | Noisy labels | Verification with OWLv2 (not one of the evaluated detectors) + SAM mask + retry up to 3 seeds; human audit |
| Format shortcut (PNG vs JPEG, resolution) | Spurious "detection" | Identical encoding policy for C1–C4 |
| COCO labels are incomplete (Incomplete Benchmarks, arXiv:2609.21822) | Real unlabeled objects count as false positives | Label audit on C0 with a detector ensemble; flagged regions become "ignore" |
| 5-week deadline | Incomplete experiments | P0/P1 split; pilot first; tables designed now; write in parallel |
| Stage 2 adds GPU time in weeks 3–4 | Squeezes week 4 | Small models (Florence-2, Qwen2.5-VL-3B in 4-bit), about 200 test images, runs overnight; plain and grounded prompts run in week 3 as soon as detections exist; if time slips, reduce the number of images, never drop the study |
| Open Images labels are not exhaustive (boxes cover only image-level-verified classes) | Real objects counted as false positives | Score only classes verified present or absent at image level; unverified classes become ignore regions; same label audit as COCO |
| Verification uses a detector → selection bias | Inflated detector recall on inserts | Verifier (OWLv2) is excluded from the evaluated set; report the verifier-pass rate |

## 8. Timeline (5 weeks)

The work is split into tasks that run in order. **Task 1 uses only real COCO photos.** Its purpose is
to measure, before any AI image exists, how the detectors and forensic methods behave on genuine
photographs. COCO was collected from Flickr around 2014, before diffusion-based editors, so its
photos are real by provenance. Any "AI" flag a forensic method raises on them is a **false alarm**.
That false-alarm rate is the baseline every later result is compared against.

| Week | Deliverables |
|---|---|
| 1 | **Task 1: real-only COCO baseline (no AI).** Repo, environment, COCO download; sample 600 sources; C0/C1 only (`configs/real_baseline.yaml`); label audit; run the 4 detectors and the hand-crafted forensic methods on the real photos → detector AP on real photos and **each forensic method's score distribution on real photos** (the false-alarm rate itself needs a threshold fixed on validation data with AI objects, so it is computed in Task 3) |
| 2  | **Task 2: AI insertion.** 50-source SDXL pilot through C2–C4 + visual check; verification + SAM masks; full generation run (SDXL on the 544 week-1 sources; G2/G3 dropped 2026-10-17); ledger finalised; human audit of 10% |
| 3  | **Task 3: detection on edited images.** Detectors and forensics on C2–C4 (C4-occ is part of C4); learned forensics (TruFor, UnivFD); semantic + matched-instance metrics (H1, H2); **Stage 2:** `vlm` stage and caption metrics written, plain and grounded prompts run |
| 4  | Crop classifier training; joint AP, calibration, abstention; robustness (H3, H4); **Open Images held-out domain: sample 100 (start download in week 2), generate C1–C4 with G1 (SDXL), run detectors + forensics, evaluate**; **Stage 2:** origin-aware prompts (need the week-4 origin scores) and caption evaluation (H5); figures |
| 5  | Writing, LaTeX conversion, supplementary material, related-work verification |

