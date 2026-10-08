# PRD: InlayFake (research product requirements)

| | |
|---|---|
| Owner | Ryan Wong (sole author; supervisor: project professor) |
| Target | CVPR 2027 main track: paper submission **16 Nov 2026 23:59 AoE** (per the official call for papers as found by search; paper registration is typically about 1 week earlier; **confirm both on cvpr.thecvf.com**) |
| Source brief | `InlayFake_Project_Description` PDF from the professor (01 Oct 2026) |
| Status | Week 1: setup |

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
| H4 | Any origin-detection gain survives a **held-out generator** and post-processing (JPEG, resize, blur). | Same metrics on held-out splits; degradation curves |
| H5 (stretch) | Grounding origin evidence into VLM captions reduces false origin claims compared with direct prompting. | Caption origin error rates (Stage 2) |

H1, H2 and H4 come from the brief. H3 is our sharpened version of the "object-conditioned estimator"
idea. H5 is the brief's H3, reduced to a small study because of time.

## 3. Scope

### In scope (must-have for the submission, P0)
1. **InlayFake benchmark v1**: about 600 COCO val2017 source photos, 10 categories, 6 matched
   conditions, 3 inpainting generators (2 seen, 1 held out). Includes a per-instance provenance ledger
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
5. **Robustness**: held-out generator; JPEG q ∈ {95, 75, 50}, 0.5× resize, Gaussian blur σ=1.
6. Paper draft in `paper/PAPER.md`, converted to the CVPR LaTeX template in week 5.

### Should-have (P1)
- Stage 2 mini-study: Florence-2 and Qwen2.5-VL-3B with three prompt settings (plain, grounded,
  origin-aware) on about 200 test images, plus unedited controls. Structured JSON output, scored for
  false-AI and false-real claims.
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

- **Categories (10):** chair, cup, bottle, potted plant, vase, clock, bench, umbrella, backpack, dog.
  These are frequent in COCO, mostly rigid and non-human.
- **Generators:** G1 SDXL-Inpainting and G2 SD-2-Inpainting (seen: used to train the crop
  classifier and to calibrate); G3 FLUX.1-Fill-dev (held out, different architecture). Fallback for
  G3 if it does not fit in VRAM: Kandinsky 2.2 inpainting.
- **Splits (by source image):** train 60% / val 20% / test 20%; G3 images appear **only** in test.
- **Ledger:** one JSONL record per (source, condition, generator) with all instances and their origin,
  intended mask, actual changed mask, seed, prompt, model revision and file hashes.

## 5. Success criteria

- Benchmark: ≥ 85% verified insertion success in C4 (audited sample). Sham edits introduce a
  detectable new object in < 5% of cases.
- Every P0 table in the paper is filled from runs, with 95% bootstrap CIs.
- At least one clear, defensible finding for each of H1, H2, H4, whether positive or negative.
- The paper is complete in CVPR format by 13 Nov (3-day buffer).

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
| Laptop VRAM (assume 8 GB) | FLUX is slow or does not fit | CPU offload / quantization; fallback to Kandinsky; held-out set uses fewer sources (300) |
| Inpainting often fails to insert the object (it fills background instead) | Noisy labels | Verification with OWLv2 (not one of the evaluated detectors) + SAM mask + retry up to 3 seeds; human audit |
| Format shortcut (PNG vs JPEG, resolution) | Spurious "detection" | Identical encoding policy for C1–C5 |
| COCO labels are incomplete (Incomplete Benchmarks, arXiv:2609.21822) | Real unlabeled objects count as false positives | Label audit on C0 with a detector ensemble; flagged regions become "ignore" |
| 5-week deadline | Incomplete experiments | P0/P1 split; pilot first; tables designed now; write in parallel |
| Verification uses a detector → selection bias | Inflated detector recall on inserts | Verifier (OWLv2) is excluded from the evaluated set; report the verifier-pass rate |

## 8. Timeline (5 weeks)

| Week | Dates | Deliverables |
|---|---|---|
| 1 | 8–14 Oct | Repo, docs, environment on laptop; COCO download; source sampling + placement; generator wrappers; **pilot of 50 sources through C1–C5 with G1**; first visual check |
| 2 | 15–21 Oct | Label audit; verification + SAM masks; full generation run (600 sources × G1, G2; 300 × G3) overnight; ledger finalised; human audit of 10% |
| 3 | 22–28 Oct | Detector runs (4 families × all conditions); semantic + matched-instance metrics (H1, H2); hand-crafted forensics; TruFor; UnivFD |
| 4 | 29 Oct–4 Nov | Crop classifier training; joint AP, calibration, abstention; robustness + held-out (H3, H4); P1 Stage 2 if on track; figures |
| 5 | 5–11 Nov | Writing, LaTeX conversion, supplementary material, related-work verification, professor review; **paper registration (≈10 Nov, confirm)** |
| buffer | 12–16 Nov | Final fixes, submission by 16 Nov AoE |

## 9. Open decisions

- [ ] Exact GPU model and VRAM of the MSI laptop (run `nvidia-smi` and record it here).
- [ ] Professor's sign-off on the scope cut (Stage 2 reduced to P1).
- [ ] Confirm the CVPR 2027 registration and submission dates on the official site.
- [ ] Licences: COCO images are CC-BY (per-image). FLUX.1-dev licence is non-commercial (fine for
      research; check redistribution before releasing generated images).
