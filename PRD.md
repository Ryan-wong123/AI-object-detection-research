# PRD: InlayFake (research product requirements)

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
| H2 | Local insertions change detector predictions on **untouched original objects** beyond what matched processing (resave, sham edit, real paste) and occlusion explain. | Matched-instance change rates C4 vs C1/C2/C3 |
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
- Human audit of 10% of edits (insertion success, unintended changes).

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
| C4 | AI insertion, no occlusion of original objects | Clean origin and collateral effects |
| C5 | AI insertion with controlled occlusion (20–50% of one original object) | Visibility-mediated effects with updated labels |

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
  it; such edits carry `class_in_photo: false` and are excluded from same-class analyses.
  **Second exception (author, 2026-10-17):** if no box satisfies the placement rules at all, a
  C5 box may be placed anywhere in the image with no overlap rule (`placement_rule: anywhere`),
  so every photo gets a box. C4 always keeps its no-occlusion rule.
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
  `source_dataset = "openimages"`, test split only. C5 is skipped (time), and C3 donors come from other
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
| Format shortcut (PNG vs JPEG, resolution) | Spurious "detection" | Identical encoding policy for C1–C5 |
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
| 2  | **Task 2: AI insertion.** 50-source SDXL pilot through C2–C5 + visual check; verification + SAM masks; full generation run (SDXL on the 544 week-1 sources; G2/G3 dropped 2026-10-17); ledger finalised; human audit of 10% |
| 3  | **Task 3: detection on edited images.** Detectors and forensics on C2–C5; learned forensics (TruFor, UnivFD); semantic + matched-instance metrics (H1, H2); **Stage 2:** `vlm` stage and caption metrics written, plain and grounded prompts run |
| 4  | Crop classifier training; joint AP, calibration, abstention; robustness (H3, H4); **Open Images held-out domain: sample 100 (start download in week 2), generate C1–C4 with G1 (SDXL), run detectors + forensics, evaluate**; **Stage 2:** origin-aware prompts (need the week-4 origin scores) and caption evaluation (H5); figures |
| 5  | Writing, LaTeX conversion, supplementary material, related-work verification |

