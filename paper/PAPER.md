# InlayFake: A Benchmark for Mixed-Origin Object Understanding in Real Photographs

> **Status box (kept current by agents; delete before submission)**
>
> | Item | State |
> |---|---|
> | Last updated | 2026-10-08 |
> | Drafted | Abstract (skeleton), §1 Intro, §2 Related work, §3 Task, §4 Benchmark construction (design), §5 Methods evaluated, §6 Metrics, §8 Limitations/Ethics |
> | Waiting on results | All numbers in Abstract and §7; Figures 1–5; Tables 2–6 |
> | Experiments done | Task 1 data stages done on the laptop (600 sampled → 544 planned sources; 1,088 real C0/C1 images). Detect/forensics/audit/evaluate pending |
> | Approx. length | ~3,300 words of text (CVPR limit ≈ 8 pages ≈ 6,000–6,500 words with figures) |
> | To verify | every reference marked [verify]; CVPR 2027 dates; dataset licence statement |
>
> Conventions: `[[TBD: ...]]` = result not yet available. Every number carries
> `<!-- run:<run_id> -->`. Nothing in this draft is a measured result unless it is tagged.

---

## Abstract

Generative editors can insert photorealistic objects into real photographs. The result is a
*mixed-origin* scene in which photographed and AI-generated objects coexist. Object detectors
correctly recognise such inserted objects, since they are visible, but they do not say where an
object came from. Meanwhile, image forensics operates at the level of whole images or pixel masks, not
semantic instances. We introduce **InlayFake**, a benchmark for *mixed-origin object understanding*:
localising and recognising every object in an edited photograph and estimating, with calibrated
uncertainty, whether each one was photographed or AI-inserted. InlayFake builds
[[TBD: N]] edited images from [[TBD: 600]] COCO photographs under six matched conditions (resave,
sham edit, real-object paste, AI insertion with and without occlusion) using three inpainting
generators, one held out, plus a source-domain-held-out test set from Open Images. Every instance carries a provenance record. We evaluate four detector
families and [[TBD: 9]] hand-crafted and learned forensic methods at the instance level. We find that
[[TBD: H1 finding: detector confidence does / does not separate origin]],
that insertions [[TBD: H2 finding: do / do not alter predictions on untouched real objects beyond matched controls]],
and that instance-conditioned forensic scoring [[TBD: H3/H4 finding incl. held-out generator]].
We release the benchmark protocol, provenance ledger and baselines.

---

## 1. Introduction

Text-guided inpainting [Rombach et al. 2022; Podell et al. 2024; BFL 2024] lets anyone add a
plausible chair, cup or dog to a real photograph in seconds. Most such edits are benign. They are
creative, decorative or commercial. Still, they raise a question that existing vision systems do not
answer: **which objects in this photo were actually there?**

Two lines of work come close but answer different questions. *Object detectors* [Ren et al. 2015;
Zhao et al. 2024; Liu et al. 2024] are trained to report what is visible, and an inserted chair *is*
visible. Labelling it "chair" is semantically correct, not a failure. *Image forensics* decides
whether a whole image is synthetic [Wang et al. 2020; Ojha et al. 2023; Guillaro et al. 2025] or
localises manipulated pixels [Guillaro et al. 2023; Qu et al. 2024; Wang et al. 2025]. Recent
multimodal systems also explain these decisions in language [Xu et al. 2025; Huang et al. 2025].
However, a manipulation mask is not an object: editors alter background pixels, shadows and
neighbouring objects, and a mask says nothing about which *instance* is synthetic. The
capability that is missing is **object-origin estimation**: a per-instance, calibrated decision
of *photographed* vs *AI-inserted*, with the option to abstain.

Measuring this capability is subtle. A naive benchmark that inserts AI objects and compares them
against untouched photos confounds origin with many nuisances. Generators re-encode images, change
resolution, leave compositing seams and modify pixels outside the requested region. Any of these can
be exploited as a shortcut [Guillaro et al. 2025]. Moreover, standard detection labels are themselves
incomplete [Incomplete Benchmarks 2026], so an unlabelled real object can be mistaken for an
insertion-induced false positive.

InlayFake addresses these issues with **matched controls**. For every source photograph, we generate
an identical-encoding resave, a *sham* edit (same mask and generator, no object), a *real-object paste*
(a photographed object composited into the same box) and AI insertions with and without occlusion of
original objects. All of these share one geometric plan. Contrasting the conditions separates the
effect of AI origin from encoding, editing and compositing. A per-instance **provenance ledger**
records intended and actual edit masks, so ground truth is auditable.

**Contributions.**
1. We formalise *mixed-origin object understanding*, a per-instance origin task with calibrated
   abstention, and separate it from semantic detection (§3).
2. We build **InlayFake**, a benchmark with six matched conditions, three inpainting generators (one
   held out), audited labels and a provenance ledger (§4).
3. We run an instance-level study of four detector families and of hand-crafted and learned forensic
   methods, scored image-level, region-pooled and instance-conditioned, alone and composed with
   detectors (§5–§7).
4. We report [[TBD: key findings, 1–2 sentences each, after experiments]].

---

## 2. Related work

**Object detection and its evaluation.** Two-stage [Ren et al. 2015], one-stage [Redmon et al. 2016;
Jocher & Qiu 2024], transformer-based [Carion et al. 2020; Zhang et al. 2023; Zhao et al. 2024] and
open-vocabulary detectors [Liu et al. 2024; Minderer et al. 2023] are evaluated on COCO [Lin et al.
2014] by semantic correctness alone. Recent re-annotation of COCO, VOC, Cityscapes and KITTI shows that
missing labels and annotation ambiguity change conclusions [Incomplete Benchmarks 2026]. We follow
that lesson by auditing the labels of our source photographs. Metamorphic testing inserts objects to
probe detector robustness [Wang & Su 2020], and synthetic object compositions are used to *train*
detectors [SOC 2026]. We instead ask whether inserted objects can be *identified as inserted*, and
whether they disturb recognition of the real objects around them.

**AI-generated image detection.** Detectors of fully synthetic images exploit generator fingerprints
and frequency artefacts [Wang et al. 2020; Durall et al. 2020; Corvi et al. 2023] or foundation-model
features [Ojha et al. 2023]. Benchmarks such as GenImage [Zhu et al. 2023] test generalisation across
generators. B-Free [Guillaro et al. 2025] shows that such detectors learn format and content shortcuts
and controls them with matched training data. Our encoding policy and sham/paste conditions apply the
same principle to evaluation.

**Manipulation localisation.** Classical forensics detects local inconsistencies in compression and
noise: error level analysis [Krawetz 2007], JPEG ghosts [Farid 2009], blocking-artifact grids [Li et
al. 2009], noise-level inconsistencies [Mahdian & Saic 2009] and rich residual models [Fridrich &
Kodovsky 2012; Cozzolino et al. 2015]. Learned localisers include ManTra-Net [Wu et al. 2019], CAT-Net
[Kwon et al. 2022] and TruFor [Guillaro et al. 2023]. MIML [Qu et al. 2024] scales manipulation
annotation. OpenSDI [Wang et al. 2025] and BR-Gen [BR-Gen 2026] localise *diffusion-generated* regions.
These methods output pixel masks. We link their evidence to semantic instances, and we evaluate the
false-accusation rate on intact photographed objects.

**Explainable forensics and captioning.** FakeShield [Xu et al. 2025] and SIDA [Huang et al. 2025] pair
localisation with language explanations. Captioning benchmarks measure object hallucination [Rohrbach
et al. 2018; Li et al. 2023], but a caption mentioning an inserted chair is not hallucinating. Our
(P1) study evaluates *origin claims* in grounded captions from Florence-2 [Xiao et al. 2024] and
Qwen2.5-VL [Bai et al. 2025].

---

## 3. Task: mixed-origin object understanding

An edited image $I$ contains instances $\{o_i\}$, each with a class $c_i$, a visible extent $b_i$
(box and mask) and an origin $y_i \in \{\text{photographed}, \text{AI-inserted}\}$. We also annotate
two non-binary states: *real-pasted* (photographed, but from another scene) and *mixed/modified*
(an original object altered by the editor). We report them separately. A system receives **only the
final pixels** of $I$. It outputs a set of predictions
$\{(\hat c_j, \hat b_j, s_j, \hat p_j, \hat d_j)\}$, where $s_j$ is a detection score, $\hat p_j$ the
probability of AI origin and $\hat d_j \in \{\text{photographed}, \text{AI}, \text{abstain}\}$ a
decision. Abstention is a prediction, never a ground-truth label. Source images, masks, prompts and
generator identities are evaluation metadata only.

We distinguish three error types: (i) **semantic** errors (missed or misclassified objects,
regardless of origin); (ii) **false accusations**, where a photographed object is labelled AI; and
(iii) **missed insertions**, where an AI object is labelled photographed. Error (ii) is a first-class
harm metric.

---

## 4. The InlayFake benchmark

**Sources.** We sample [[TBD: 600]] photographs from COCO val2017 [Lin et al. 2014], which is unseen
by COCO-trained detectors. We exclude images dominated by people, and we split train/val/test
(60/20/20) by source photograph. We audit the original labels with a detector ensemble and manual
review. Confirmed missing objects are added and unresolved regions are ignored
([[TBD: number of added labels / ignore regions]]).

**Insertion plan.** Of 600 sampled photographs, 544 admitted a free-space box for the
target category <!-- run:task1_real_baseline (sample/plan/generate, author's laptop, 2026-10-08) -->. The remaining 56 (9.3%) were too cluttered and are excluded from every
condition. The benchmark therefore under-represents very crowded scenes (§8). Of the 544, 363 also
admitted a controlled-occlusion box for C5 <!-- run:task1_real_baseline (sample/plan/generate, author's laptop, 2026-10-08) -->.
 For each source we choose one of ten categories (chair, cup, bottle, potted plant,
vase, clock, bench, umbrella, backpack, dog) and a box whose size follows that category's COCO size
distribution. The box is placed either in free space or overlapping 20–50% of one original object.
One plan is shared by every condition of a source.

**Conditions.** (Table 1) C0 untouched; C1 matched resave; C2 sham edit; C3 real-object paste; C4 AI
insertion without occlusion; C5 AI insertion with controlled occlusion.

**Generators.** SDXL-Inpainting [Podell et al. 2024] and SD-2-Inpainting [Rombach et al. 2022] are
*seen* generators. FLUX.1-Fill [BFL 2024], a flow-matching transformer, is *held out* and appears only
in test. Inpainting operates on a square window of 2.5× the planned box at the generator's native
resolution. Only pixels inside the dilated rectangular mask are composited back, with a feather that
vanishes at the mask border, so every pixel outside the intended mask is identical to the matched
resave C1. FLUX.1-Fill is applied only to test-split sources.

**Held-out source domain.** To test whether conclusions transfer beyond COCO, we build a small
test-only set of [[TBD: ~100]] Open Images V7 [Kuznetsova et al. 2020] photographs with conditions C1–C4,
using one seen (SDXL) and one held-out (FLUX.1-Fill) generator. No model, calibrator or threshold is fit
on this set. Because Open Images boxes are exhaustive only for classes verified at image level, we
score only verified classes and ignore the rest. We use Open Images rather than Objects365 because
the latter is part of Grounding DINO's pretraining data [Liu et al. 2024].

**Encoding policy.** All images in C1–C5 pass through the same decode → edit → JPEG (q=95, 4:4:4)
pipeline, so file format cannot reveal origin.

**Verification and annotation.** An open-vocabulary verifier (OWLv2 [Minderer et al. 2023]), which is
not among the evaluated detectors, confirms each insertion. SAM [Kirillov et al. 2023] provides its
mask. Failed insertions are retried with new seeds or discarded
([[TBD: success rate]]). We record the actual changed region separately from the intended mask, and
we mark original objects overlapped by more than 10% as *mixed/modified*. A human audit of
[[TBD: 10%]] of edits finds [[TBD: audit results]].

**Table 1.** Conditions and statistics. [[TBD: images, instances per origin, per condition]]

| Cond. | Description | #images | #photographed | #AI-inserted | #real-pasted |
|---|---|---|---|---|---|
| C0 | untouched | 544 <!-- run:task1_real_baseline (sample/plan/generate, author's laptop, 2026-10-08) --> | [[TBD]] | – | – |
| C1 | matched resave | 544 <!-- run:task1_real_baseline (sample/plan/generate, author's laptop, 2026-10-08) --> | [[TBD]] | – | – |
| C2 | sham edit | [[TBD]] | [[TBD]] | – | – |
| C3 | real paste | [[TBD]] | [[TBD]] | – | [[TBD]] |
| C4 | AI insert | [[TBD]] | [[TBD]] | [[TBD]] | – |
| C5 | AI insert + occlusion | [[TBD]] | [[TBD]] | [[TBD]] | – |

**Figure 1.** [[TBD: example source with C1–C5 panels and instance origin overlays]]

---

## 5. Methods evaluated

**Detectors (frozen, COCO-trained).** Faster R-CNN R50-FPN v2 [Ren et al. 2015], YOLO11-m [Jocher &
Qiu 2024], RT-DETR-R50 [Zhao et al. 2024], and Grounding DINO-T [Liu et al. 2024] prompted with COCO
class names.

**Hand-crafted forensics.** Each produces a dense map in which higher means more suspicious. For
F3–F5 the map is the absolute deviation of a local statistic from the image median. F1 error level
analysis [Krawetz 2007]; F2 SRM high-pass residual variance
[Fridrich & Kodovsky 2012]; F3 local noise-level inconsistency (wavelet MAD) [Mahdian & Saic 2009];
F4 high-frequency spectral energy [Durall et al. 2020]; F5 JPEG blocking-artifact grid [Li et al.
2009].

**Learned forensics.** L1 TruFor [Guillaro et al. 2023]; L2 a UnivFD-style CLIP probe [Ojha et al.
2023]; L3 a CLIP-feature logistic probe trained on object crops from the seen generators of our train
split (an in-domain upper reference); L4 B-Free [Guillaro et al. 2025] [[TBD: if runnable]].

**Instance scoring.** Each map-producing method is converted to a per-instance score in three ways:
*image-level* (one score per image), *region-pooled* (mean / 95th percentile inside the instance mask)
and *instance-conditioned* (interior minus a surrounding context ring). Scores are calibrated on val.

**Composed system.** Detector predictions are paired with an origin scorer evaluated on the
*predicted* region. Two thresholds chosen on val define an abstention band.

---

## 6. Metrics

*Semantic*: COCO AP/AR [Lin et al. 2014], reported separately for original and inserted subsets.
*Matched-instance change*: for each original instance, the detected→missed rate, class flips, box IoU
drift and score change from C1 to each condition, stratified by distance to the edit and by
visibility. *Origin*: AUROC, AUPRC, TPR at 5% FPR and the **real-object false-accusation rate** at 80%
TPR, computed both with oracle localisation and end to end. *Joint origin-aware AP*: a true positive
requires the class, IoU ≥ 0.5 and the correct origin. *Calibration*: ECE, Brier score and
risk–coverage (AURC) [Guo et al. 2017; Geifman & El-Yaniv 2017]. All numbers have 95% bootstrap CIs
over source photographs (1000 resamples), because edits of one photograph are correlated.

---

## 7. Results

### 7.1 Baseline: unedited photographs
Before any insertion, we run every detector and forensic method on the untouched (C0) and resaved
(C1) photographs. COCO predates diffusion-based editing, so every object is photographed and every
"AI" flag is a false alarm.
**Table 2a.** Detector AP on C0 vs C1 with 95% CIs, and per-object detection flips caused only by
resaving (C0→C1). [[TBD: run task1_real_baseline]]
**Table 2b.** Hand-crafted forensic scores on real photographs: image-level distribution, per-object
contrast (inside vs context) and the paired resave shift C1 − C0. The false-alarm rate of each method
is reported in §7.4 at the threshold fixed on validation images (TPR = 80%).
[[TBD: run task1_real_baseline]]
**Finding.** [[TBD]]

### 7.2 Do detectors see inserted objects, and does confidence reveal origin? (H1)
**Table 2.** AP / AR on original vs inserted instances per detector and condition. [[TBD]]
**Finding.** [[TBD]]

### 7.3 Do insertions disturb real objects? (H2)
**Figure 2.** Matched-instance change rates (C2, C3, C4, C5 vs C1) by distance to the edit. [[TBD]]
**Finding.** [[TBD]]

### 7.4 Can forensics tell origin per object? (H3)
**Table 3.** Origin AUROC / AUPRC / false-accusation rate per method × scoring mode, oracle
localisation, seen generators. [[TBD]]
**Table 4.** End-to-end joint origin-aware AP and calibration of composed systems. [[TBD]]
**Finding.** [[TBD]]

### 7.5 Generalisation and robustness (H4)
**Table 5.** Held-out generator (FLUX.1-Fill) and held-out source domain (Open Images), in a 2×2 grid: seen/held-out generator × COCO/Open Images. [[TBD]]
**Figure 3.** AUROC under JPEG 95/75/50, 0.5× resize, blur, screenshot. [[TBD]]
**Finding.** [[TBD]]

### 7.6 Controls: what are the detectors actually responding to?
C3 (real paste) and C2 (sham) results that separate origin from compositing and editing. [[TBD]]

### 7.7 (P1) Origin claims in grounded captions
**Table 6.** False-AI / false-real / abstention rates for Florence-2 and Qwen2.5-VL across three
prompt settings, including unedited images. [[TBD: or move to supplementary / drop]]

---

## 8. Limitations, ethics and broader impact

InlayFake excludes very cluttered photographs (9.3% of sampled sources had no free space for an
insertion), and covers ten non-human object categories, one main source domain (COCO) with a small held-out
Open Images test set, and three open inpainting models. Commercial editors and other domains may behave differently. Pixel evidence cannot recover an
edit history with certainty, so models should be able to abstain, and our labels describe *how the
image was made*, not whether anyone intended to deceive. Benign creative edits are in scope, and
calling a real object fake is treated as a harm, not merely an error. We do not generate people or
identifiable individuals. Source images keep their COCO licences. Generated images inherit the
generators' licences; [[TBD: release terms]]. Cryptographic provenance (e.g., content credentials) is
complementary and is not evaluated here.

---

## 9. Conclusion

[[TBD: 3–4 sentences summarising the findings for H1–H4 and the released assets.]]

---

## References

All entries must be checked against the published versions before submission. Entries marked
[verify] have uncertain author lists or venues.

- [Bai et al. 2025] S. Bai et al. Qwen2.5-VL Technical Report. arXiv 2025.
- [BFL 2024] Black Forest Labs. FLUX.1 (incl. FLUX.1-Fill-dev). 2024. [verify citation form]
- [BR-Gen 2026] Zooming In on Fakes: A Novel Dataset for Localized AI-Generated Image Detection with Forgery Amplification Approach. AAAI 2026. [verify authors]
- [Carion et al. 2020] N. Carion et al. End-to-End Object Detection with Transformers. ECCV 2020.
- [Corvi et al. 2023] R. Corvi et al. On the Detection of Synthetic Images Generated by Diffusion Models. ICASSP 2023.
- [Cozzolino et al. 2015] D. Cozzolino, G. Poggi, L. Verdoliva. Splicebuster: A New Blind Image Splicing Detector. WIFS 2015.
- [Durall et al. 2020] R. Durall, M. Keuper, J. Keuper. Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions. CVPR 2020.
- [Farid 2009] H. Farid. Exposing Digital Forgeries from JPEG Ghosts. IEEE TIFS 2009.
- [Fridrich & Kodovsky 2012] J. Fridrich, J. Kodovský. Rich Models for Steganalysis of Digital Images. IEEE TIFS 2012.
- [Geifman & El-Yaniv 2017] Y. Geifman, R. El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017.
- [Guillaro et al. 2023] F. Guillaro et al. TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization. CVPR 2023.
- [Guillaro et al. 2025] F. Guillaro et al. A Bias-Free Training Paradigm for More General AI-generated Image Detection (B-Free). CVPR 2025.
- [Guo et al. 2017] C. Guo et al. On Calibration of Modern Neural Networks. ICML 2017.
- [Huang et al. 2025] Z. Huang et al. SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model. CVPR 2025. [verify authors]
- [Incomplete Benchmarks 2026] Object Detection Benchmarks are Incomplete: The Role of Label Errors and Annotation Uncertainty. arXiv:2609.21822, 2026. [verify authors and venue]
- [Jocher & Qiu 2024] G. Jocher, J. Qiu. Ultralytics YOLO11. Software, 2024.
- [Kirillov et al. 2023] A. Kirillov et al. Segment Anything. ICCV 2023.
- [Krawetz 2007] N. Krawetz. A Picture's Worth: Digital Image Analysis and Forensics. Black Hat Briefings 2007.
- [Kuznetsova et al. 2020] A. Kuznetsova et al. The Open Images Dataset V4: Unified Image Classification, Object Detection, and Visual Relationship Detection at Scale. IJCV 2020.
- [Kwon et al. 2022] M.-J. Kwon et al. Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization (CAT-Net v2). IJCV 2022. [verify]
- [Li et al. 2009] W. Li, Y. Yuan, N. Yu. Passive Detection of Doctored JPEG Image via Block Artifact Grid Extraction. Signal Processing 2009. [verify]
- [Li et al. 2023] Y. Li et al. Evaluating Object Hallucination in Large Vision-Language Models (POPE). EMNLP 2023.
- [Lin et al. 2014] T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014.
- [Liu et al. 2024] S. Liu et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ECCV 2024.
- [Mahdian & Saic 2009] B. Mahdian, S. Saic. Using Noise Inconsistencies for Blind Image Forensics. Image and Vision Computing 2009.
- [Minderer et al. 2023] M. Minderer, A. Gritsenko, N. Houlsby. Scaling Open-Vocabulary Object Detection (OWLv2). NeurIPS 2023.
- [Ojha et al. 2023] U. Ojha, Y. Li, Y. J. Lee. Towards Universal Fake Image Detectors that Generalize Across Generative Models. CVPR 2023.
- [Podell et al. 2024] D. Podell et al. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. ICLR 2024.
- [Qu et al. 2024] C. Qu et al. Towards Modern Image Manipulation Localization: A Large-Scale Dataset and Novel Methods (MIML). CVPR 2024.
- [Redmon et al. 2016] J. Redmon et al. You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016.
- [Ren et al. 2015] S. Ren et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS 2015.
- [Rohrbach et al. 2018] A. Rohrbach et al. Object Hallucination in Image Captioning. EMNLP 2018.
- [Rombach et al. 2022] R. Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
- [SOC 2026] Synthetic Object Compositions for Scalable and Accurate Learning in Detection, Segmentation, and Grounding. CVPR 2026. [verify authors]
- [Wang & Su 2020] S. Wang, Z. Su. Metamorphic Object Insertion for Testing Object Detection Systems. ASE 2020.
- [Wang et al. 2020] S.-Y. Wang et al. CNN-Generated Images Are Surprisingly Easy to Spot… for Now. CVPR 2020.
- [Wang et al. 2025] Y. Wang et al. OpenSDI: Spotting Diffusion-Generated Images in the Open World. CVPR 2025. [verify authors]
- [Wu et al. 2019] Y. Wu, W. AbdAlmageed, P. Natarajan. ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries with Anomalous Features. CVPR 2019.
- [Xiao et al. 2024] B. Xiao et al. Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks. CVPR 2024.
- [Xu et al. 2025] Z. Xu et al. FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models. ICLR 2025.
- [Zhang et al. 2023] H. Zhang et al. DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. ICLR 2023.
- [Zhao et al. 2024] Y. Zhao et al. DETRs Beat YOLOs on Real-time Object Detection (RT-DETR). CVPR 2024.
- [Zhu et al. 2023] M. Zhu et al. GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image. NeurIPS 2023 (Datasets & Benchmarks).
