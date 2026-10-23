# InlayFake: A Benchmark for Mixed-Origin Object Understanding in Real Photographs

> **Status box (kept current by agents; delete before submission)**
>
> | Item | State |
> |---|---|
> | Last updated | 2026-10-21 |
> | Natural placement in §4 | 2026-10-21: rewritten for the benchmark version: Qwen3-VL-8B box mode with one merged placement question (Box 1, full prompt), three checks only (15 px / 0.1% measured; elongation ≤ 3× the class's; ≤ 30% of any labelled object or person), one re-ask with the failures, next class, no fallback; no review gate (author), hand audit of every C4 afterwards; self-judge AUROC 0.44/0.50 (dropped). Table 1c replaced by the benchmark's rules. Pilot 44 of 50 (run task2_pilot_natural_qwen_free). Full run configs/insertion_natural_box.yaml not started [[TBD]]. Box 1 (~250 words) may move to the supplementary for the page limit. Earlier Molmo and rule-based versions are in docs/RESEARCH_LOG.md |
> | Full natural run done | 2026-10-23 (run task2_insertion_natural_box): 427 of 544 photos with a verified AI object (boxes 1/2/3: 350/57/20), 425 complete C0–C4, 76 classes; author review of C2/C3/C4 pending [[TBD]] |
> | Natural-placement pilot v3 | 2026-10-20 (run task2_pilot_natural_v3): more spots/classes per photo: 19 of 50 photos (v2 18), same quality (paper/results/task2_pilot_natural_four_way.json) |
> | Natural-placement pilot v2 | 2026-10-20 (run task2_pilot_natural_v2): with surface rule, larger boxes and box-fill rule: 18 of 50 photos with a verified AI edit, 72% on a fitting surface (random 46%), objects median 1.8% of the image (random 0.8%) (paper/results/task2_pilot_natural_three_way.json; pilot, not yet a paper result) |
> | Natural-placement pilot | 2026-10-19 (run task2_pilot_natural, 50 photos): VLM-pointed, depth-scaled boxes; 33/50 photos get a natural spot; VLM realism check unusable (off); 28 of 50 photos with a verified natural AI edit; vs random placement on the same photos: on a fitting surface 64% vs 46%, objects smaller (median 0.5% vs 0.8% of the image) (paper/results/task2_pilot_natural_vs_random.json; pilot, not yet a paper result). If adopted, §4 placement text and a new full run follow |
> | Insertion run done | 2026-10-18 15:02: all 544 photos complete C0–C4 (AI edits: 464 free-space, 38 occluding, 42 anywhere; the bed and train close-ups pass only in the bottom-left corner at threshold scores, flagged), 1,088 edited images (paper/results/task2_insertion_final.json). Human audit (all 544 C4 edits; C2 not audited) and Table 1 counts pending |
> | Merged run | 2026-10-18: 537 of 544 photos with a verified AI edit (464 free-space, 38 occluding, 35 anywhere), 7 dropped and re-run with 6 more boxes off the real same-class object (paper/results/task2_insertion_merged_round_final.json) |
> | Scope change 2 | 2026-10-18 (author): the two AI placement rules (free space, occlusion) merged into one condition C4 (a class already in the photo, placed anywhere; verified free-space edit kept, else verified occluding edit, else new box anywhere; C2/C3 at the chosen box). 44 photos had no AI object under the separate rules. §4 conditions, rationale, Tables 1/1b, verification text, §6 and H2 analysis plan rewritten; H2 loses its controlled occlusion condition (covered objects analysed observationally). Merged run in progress |
> | Scope change | 2026-10-17 (author): SDXL is the only generator; SD2 and the held-out FLUX/Kandinsky generator are dropped for time. H4 keeps post-processing + Open Images; held-out-generator results (old Table 5 grid) removed; Abstract, §1, §4, §5, §7.5, §8 reworded |
> | Drafted | Abstract (skeleton), §1 Intro, §2 Related work, §3 Task, §4 Benchmark construction (design), §5 Methods evaluated, §6 Metrics, §8 Limitations/Ethics |
> | Waiting on results | All numbers in Abstract and §7; Figures 1–5; Tables 2–6 |
> | Experiments done | Task 1 complete (run task1_real_baseline, final evaluate 2026-10-11): 4 detectors and 5 hand-crafted methods on 544 × (C0, C1) real photos; §7.1 Tables 2a/2b final with 200-resample bootstrap CIs; label audit applied (77 add / 26 ignore / 4 reject, Table 2a uses audited labels), re-checked and confirmed by the author. Task 2: insertion + verification code complete and CPU-tested (branch week-2-ai-insertion). SDXL pilot (run task2_pilot, 2026-10-16, 46 planned sources, attempt 1 only): verifier-confirmed insertion 4/46 C4 and 1/38 C4-occ, sham 46/46 pass, 21.6 s per SDXL image (paper/results/task2_pilot_round1_*.json). Insertion design must change before the full run; §4 success-rate TBD left open on purpose because the pilot settings will not be the final ones. Second pilot with surface placement (run task2_pilot_surface): C4 0/35, C4-occ 4/19, so placement is not the cause; a 12-job diagnostic shows the edit-window size is (window 2.5× / 1.5× / 1.2× → 1 / 5 / 7 of 12 verified; paper/results/task2_window_scale_diagnostic.txt). Window size awaits the author's decision. Design change (author, 2026-10-16): the inserted class is now one already in the photo's audited GT (any class except person) instead of ten fixed categories; §4 rewritten; pilot re-planned (run task2_pilot_inphoto: 41/50 plans, 34 with C4-occ, 31 classes). Same-class pilot generated at window 1.2× with the looser verifier (IoU ≥ 0.3 or ≥ 80% inside the box): C4 27/41, C4-occ 17/34 verified after ≤ 3 attempts, C2 41/41 (paper/results/task2_pilot_inphoto_final_verify.json); §4 verification text updated. With re-plan on failure (up to 3 new boxes): C4 38/41, C4-occ 25/34 verified, C2 39/41 (paper/results/task2_pilot_inphoto_replan_verify.json). With class switch and C4-occ-only plans (2026-10-17): C4 39/41, C4-occ 35/41; 46 of 50 photos have a verified AI object (paper/results/task2_pilot_inphoto_classswitch_verify.json). With C4-occ boxes sized from the covered object: C4-occ 37/42, 48 of 50 photos (paper/results/task2_pilot_inphoto_targetsize_verify.json). With a class not in the photo as last resort: 49 of 50 photos, one such edit (paper/results/task2_pilot_inphoto_outside_verify.json). With a C4-occ box anywhere as very last resort: 50 of 50 photos (one doubtful tiny object; paper/results/task2_pilot_inphoto_anywhere_verify.json). Full run task2_insertion planned on the 544 week-1 photos (544 of 544, configs/insertion_inphoto.yaml), generation pending; success rate stays TBD until it finishes |
> | Approx. length | ~3,800 words of text (CVPR limit ≈ 8 pages ≈ 6,000–6,500 words with figures); §4 condition rationale added 2026-10-17 (~450 words, may need shortening for the page limit) |
> | Design decision | 2026-10-18: §4 explains how C3 is built and why C0, C3 and C4 alone would not suffice (C1 matched-encoding reference, C2 separates AI objects from AI-repainted pixels; ~370 words added). 2026-10-17 (author): C2 sham edit kept as a control; its justification and scoring (never an AI object; separate edited-background flag rate) written into §4, §6, §7.6 |
> | To verify | every reference marked [verify]; CVPR 2027 dates; dataset licence statement |
>
> Terminology (status box only): C4-occ = the occlusion placement rule of the AI insertion before 18 Oct 2026 (cover 20–50% of one real object); since then it is part of the single AI condition C4.
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
[[TBD: N]] edited images from 544 COCO photographs under five matched conditions (original,
resave, sham edit, real-object paste, AI insertion of a class already in the photograph) using the SDXL inpainting
model, plus a source-domain-held-out test set from Open Images. Every instance carries a provenance record. We evaluate four detector
families and [[TBD: 9]] hand-crafted and learned forensic methods at the instance level. We find that
[[TBD: H1 finding: detector confidence does / does not separate origin]],
that insertions [[TBD: H2 finding: do / do not alter predictions on untouched real objects beyond matched controls]],
that instance-conditioned forensic scoring [[TBD: H3/H4 finding incl. post-processing and held-out domain]],
and that grounding origin evidence in VLM captions [[TBD: H5 finding: does / does not reduce false origin claims]].
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
(a photographed object composited into the same box) and an AI insertion of a class already in the
photograph, which may or may not cover original objects. All of these share one box. Contrasting the conditions separates the
effect of AI origin from encoding, editing and compositing. A per-instance **provenance ledger**
records intended and actual edit masks, so ground truth is auditable.

**Contributions.**
1. We formalise *mixed-origin object understanding*, a per-instance origin task with calibrated
   abstention, and separate it from semantic detection (§3).
2. We build **InlayFake**, a benchmark with six matched conditions, verified AI insertions from an
   open inpainting model (SDXL), audited labels and a provenance ledger (§4).
3. We run an instance-level study of four detector families and of hand-crafted and learned forensic
   methods, scored image-level, region-pooled and instance-conditioned, alone and composed with
   detectors (§5–§7).
4. We measure the origin claims that vision-language models make when describing mixed-origin
   images, with and without grounded origin evidence (§7.7).
5. We report [[TBD: key findings, 1–2 sentences each, after experiments]].

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

**AI object edits in COCO photographs.** Several recent datasets inpaint objects into COCO
images: CocoGlide [Guillaro et al. 2023], TGIF [TGIF 2024], COCO-Inpaint [COCO-Inpaint 2025] and
COinCO [COinCO 2026]. COinCO, the closest to our setting, replaces one object per image and uses an
object detector to keep only edits where the new object appears. TGIF releases each edit both
spliced into the original and fully regenerated. INP-X [INP-X 2026] shows that whole-image detectors
largely key on the global artefacts of the generator's autoencoder, which is the confound our
resave, sham-edit and mask-only composite controls are designed to remove. These works label the
edited region or the edited object. We additionally label the origin of every object, include a
real-object paste control and a same-class real counterpart in the same photograph, and measure
how frozen detectors and forensic methods treat inserted and photographed objects side by side.

**Explainable forensics and captioning.** FakeShield [Xu et al. 2025] and SIDA [Huang et al. 2025] pair
localisation with language explanations. Captioning benchmarks measure object hallucination [Rohrbach
et al. 2018; Li et al. 2023], but a caption mentioning an inserted chair is not hallucinating. Our
study evaluates *origin claims* in grounded captions from Florence-2 [Xiao et al. 2024] and
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

**Sources.** All photographs come from COCO val2017 [Lin et al. 2014] (5,000 images, 80 object
categories). We use val2017 rather than train2017 because the COCO-trained detectors we evaluate
were trained on train2017, so val2017 photographs are unseen by them. Selection proceeds in four
deterministic steps:

1. *Eligibility.* A photograph is eligible if its shorter side is at least 256 px (excludes 42),
   it has at least one non-crowd labelled object (excludes a further 47), and people cover at most
   30% of its area (excludes a further 490; edits should not centre on people, §8). This leaves
   4,421 eligible photographs <!-- computed 2026-10-17 from the COCO val2017 annotations with the sampling settings of configs/default.yaml (min_image_side 256, max_person_area_frac 0.30), filters applied in this order -->.
2. *Random sample.* The eligible image ids are sorted, shuffled with a random generator seeded
   with the project seed, and the first 600 are taken. The same seed reproduces the same 600.
3. *Splits.* Each photograph is assigned to train / val / test (60/20/20) by hashing its id, so
   all conditions of a photograph share a split and the assignment does not depend on sampling
   order: 362 / 114 / 124 photographs <!-- run:task2_insertion (sample; identical to task1_real_baseline), data/inlayfake_v2/sources.jsonl -->.
4. *Planning* (below) keeps the photographs that admit an insertion plan.

We audit the original labels with a detector ensemble and manual review. Confirmed missing
objects are added and unresolved regions are ignored.
Of 107 candidates in 70 of 544 photographs, we add 77 labels, keep 26 as ignore regions and reject 4 as detector errors <!-- run:task1_real_baseline, file:paper/results/task1_audit_decisions.csv -->.

**Insertion plan.** The real-only baseline (§7.1) fixed the photograph set with a first plan:
for each photograph, one of ten fixed categories (chair, cup, bottle, potted plant, vase, clock,
bench, umbrella, backpack, dog), rotated so that each is used about equally, with a box sized
like a typical COCO instance of that category, placed in free space (no overlap with any labelled
object, 8 px margin, up to 200 random positions); if no position fits, the next category is
tried. Of the 600 sampled photographs, 544 admitted such a free-space box <!-- run:task1_real_baseline (sample/plan/generate, author's laptop, 2026-10-08) -->. The remaining 56 (9.3%) had no free space and are excluded from every
condition. They are close-ups rather than crowded scenes: they have the same median number of labelled
objects as the kept photographs (4), but their object boxes cover a median 97% of the frame (kept: 32%),
and in 52 of the 56 a single object covers more than half the image <!-- run:task1_real_baseline (plan failed_sources, seed 20261008; COCO val2017 annotations), see docs/RESEARCH_LOG.md 2026-10-11 -->.
The benchmark therefore under-represents close-up, frame-filling shots (§8). Of the 544, 363 also
admitted a controlled-occlusion box <!-- run:task1_real_baseline (sample/plan/generate, author's laptop, 2026-10-08) -->.
These 544 photographs are the benchmark's photograph set: the real-only baseline and the
insertion run use exactly the same photographs, so results on unedited and edited images are
comparable. For each photograph the insertion run adds one object of a class that is
**already present** in that photograph's audited ground truth (any COCO class except *person*),
so every edited photograph contains a photographed and an AI-generated instance of the same
class. Among the classes present, the one used least so far is tried first, which keeps classes
as balanced as the photographs allow. Where the new object goes, and how large it is, is decided
by the natural placement described below; one plan (class and box) is shared by every condition
of a photograph.

**Conditions.** (Table 1) C0 untouched; C1 matched resave; C2 sham edit; C3 real-object paste; C4 AI
insertion of a class already in the photograph, at a place chosen by natural placement. An
earlier rule-based placement of the same photographs is kept as a comparison (see
*Random-placement comparison run* below).

*Why each condition.* The conditions form a ladder in which neighbouring rungs differ in roughly
one factor, so that any change in detector or forensic output can be attributed to one cause.
C0 is the photograph exactly as distributed by COCO. It anchors the benchmark to ordinary,
unedited images (detector accuracy and forensic false alarms on real photographs), and it is
where the label audit is done. It is not used as the reference for edits, because its encoding
differs from that of every edited image. C1 applies only the benchmark's decode–encode pipeline
and is therefore the reference for every contrast (re-encoding alone already shifts every
hand-crafted forensic score, §7.1); C0 versus C1 measures that re-encoding effect. C4 adds an AI-generated
object at a natural place in the scene. Three controls isolate what in C4 is specific to *AI origin*:

- *C2 (sham edit)* runs the same generator on the same box with the same pipeline, but with the
  prompt "empty background" (and the object class as a negative prompt); a verifier confirms that
  no object of the class appeared. The box therefore contains AI-repainted background and **no
  object**. C2 is the analogue of a placebo: it receives the editing procedure without the
  object. Without it, a forensic method that flags the C4 box could be responding to regenerated
  pixels rather than to an AI-made object, and a detector change near the box could be caused by
  repainting alone. C2 boxes are never scored as AI objects (§6); they measure how often methods
  flag edited background, and they provide the "editing without a new object" control for H2.
- *C3 (real paste)* composites a *photographed* object of the same class into the same box with
  the same feathered blending. C3 versus C4 contrasts two objects that are both composited, one
  real and one AI-made, so compositing seams alone cannot explain an origin difference.
  No generator is involved: a donor instance of the same class is drawn (with a fixed seed) from
  another COCO photograph, excluding crowd annotations, instances under 48 px on their shorter
  side and instances touching their image border (likely truncated), and preferring donors whose
  aspect ratio is within 25% of the box. The donor is cut out with its COCO segmentation mask,
  scaled with its aspect ratio kept to fit the box (centred horizontally, bottom-aligned, as if
  standing on the ground), and alpha-blended with the same 3 px feather as the AI edits; every
  pixel outside the pasted mask is the original photograph, and the result is saved with the
  same JPEG settings. C3 therefore contains only camera-recorded pixels, with the usual traces of
  cut-and-paste (resampling, a soft seam, possibly mismatched lighting and noise). Its object is
  a real-object negative: calling it "AI" counts as a false accusation. One difference from C4
  remains, and C2 covers it: the background inside a C3 box is untouched, whereas in C4 it is
  regenerated together with the object.
- *C1* versus C2, C3 and C4 separates re-encoding from editing.

*Why not only C0, C3 and C4?* The task is to decide, per object, whether it is AI-made, so one
might expect the original (C0), a real inserted object (C3) and an AI inserted object (C4) to
suffice. Two confounds rule this out. First, C0 is COCO's original file with its own JPEG
history, whereas C3 and C4 have passed through the benchmark's decode–encode pipeline; a method
could separate C0 from C3/C4 by detecting re-encoding alone, which already shifts every
hand-crafted forensic score (§7.1). C1 is not a resize but the same photograph re-saved with
exactly the settings of every edited image, so it is the real, unedited reference with matched
encoding. Second, in C4 the generator regenerates the whole box, background included, not only
the object, while C3 contains no generated pixels; a method that flags C4 but not C3 may detect
*AI-repainted pixels* rather than an *AI-made object*. C2 applies the same generator to the same
box with no object, so a method that flags C2 boxes as often as C4 boxes detects repainting, not
AI objects. C2 also provides the "editing without a new object" control for H2 and a false-alarm
rate on edited real backgrounds. Neither control changes the headline task: positives are only
the AI objects of C4, negatives are photographed objects (including those pasted in C3), and C2
boxes are never scored as objects (§6). Without C1 and C2, a separation of C3 from C4 objects
could only be claimed as "pasted-real versus AI-inserted", not as evidence that methods respond
to AI origin.

C2, C3 and C4 share **one box** per photograph, so they differ only in what fills it: nothing
(C2), a real object (C3) or an AI object (C4). When the AI insertion fails verification and moves
to the placement model's next box, C2 and C3 are regenerated at that box to keep the triple matched.

*Covered real objects.* A new object may cover up to 30% of a labelled real object or person
(Table 1c), as when one object stands partly in front of another; it never covers more, so every
edit adds an object rather than replacing or hiding one. Whether an AI object covers part of a
real object is therefore not assigned by design. We evaluate H2 on original objects that the AI
object does not cover, and analyse covered objects separately, stratified by their recomputed
visible fraction; that analysis is observational, because which photographs received a covering
edit depends on where the placement model put the box and on which edits the verifier accepted.
Partly covered real objects are where a false "AI" accusation is most likely, so origin metrics
are also reported for covered and uncovered real objects. When the AI object covers part of a
real object, so does the C2 sham at the same box: the sham repaints part of that object, which is
then marked *mixed/modified*.

*Natural placement.* Random free-space boxes produced implausible edits (a skateboard in mid-air,
a bird inside a window pane), so the insertion run lets a vision–language model place each new
object where it makes sense in the scene. For every photograph, the classes already present in
its labels (COCO annotations and audited additions, never *person*) are tried in order, the class
used least often so far first. For the first class, Qwen3-VL-8B [[verify: citation]] receives the
whole photograph and one fixed question (Box 1) that lists every labelled object and person as an
occupied place, names the photograph's own objects of the class, states where the class rests and
how (one of 22 class groups, e.g. "lie on the ground, grass, floor or a bench" for a tennis
racket), asks for the whole object at its normal proportions and a minimum size, and gives an
order of preference: a whole object in free space; a place nearer the camera if it would
otherwise be very small; and only when no free space remains, an object cut off by the image
border or partly behind something. The model returns three boxes, best first. The wording is
identical for every photograph of a class, so placement involves no per-image tuning. Each box
is then checked against three limits only (Table 1c): it must be large enough to generate and
verify, not much thinner than the class usually is, and cover at most 30% of any labelled object
or person, so that every edit adds an object instead of replacing or hiding a real one. If all
three boxes fail, the model is asked once more with the failed boxes and the reasons ("covers 85%
of the existing car"); if that fails too, the next class is tried, and a photograph without a
usable box receives no AI edit (there is no random or forced fallback). The first box is
generated; the next ones are used only if the verifier rejects it. No automatic plausibility
filter is applied after placement: asking the same model whether an object in a drawn box would
look natural did not agree with the first author's judgement on the pilot boxes (AUROC 0.44 and
0.50 for two wordings) <!-- run:task2_pilot_natural_qwen_free, file:paper/results/task2_judge_agreement.json -->,
and a yes/no realism check after generation was dropped for the same reason (§7.6
[[TBD: limitation text]]); instead every C4 edit is audited by hand (see *Verification* below). The VLM origin-claim study (§5, Table 6) uses Qwen2.5-VL-3B; the placement model is a
later, larger model of the same family that only chooses the location, which we disclose, and we
compare that judge on natural and random placements [[TBD: VLM check on natural vs random
placement]]. An earlier version let Molmo-7B-D point at a spot and sized the box from the
photograph's own object and monocular depth; its boxes were often upright for objects that should
lie flat (39 of 73 such plans, e.g. a tennis racket standing in mid-air) and it was replaced
<!-- run:task2_insertion_natural, file:paper/results/task2_insertion_natural_plan_50pct.json; count from data/inlayfake_v3/natural_candidates.jsonl, 2026-10-21 -->.

> **Box 1. Placement question** (one per photograph and class; braces are filled per photograph:
> occupied places and the class's own objects as boxes on a 0–1000 grid, the minimum size of
> 15 px on that grid, *where* and *how* from the class group).
>
> *Look at the whole photo and understand the scene: what is happening, where the surfaces are,
> and how big things are at each distance. These places are already taken by people and objects
> (bbox_2d, 0-1000 grid): {taken}. The photo already shows {existing}. Add one more, new {class}
> so that it looks completely natural and makes sense in this exact scene. It should {how} on
> {where}, rest on a real surface (or be in open sky if it can fly), have a realistic size and
> orientation for its distance, fit what is happening in the photo, and not overlap any taken
> place. Draw each box around the whole new {class} with its normal proportions (not a thin
> strip), at least {min} wide and {min} high on the 0-1000 grid. Choose places in this order of
> preference: (1) a whole {class} in free space; (2) if it would be smaller than that there, a
> place closer to the camera, in the lower part of the photo, where it appears larger; (3) only if
> there is no free space at all, a {class} partly cut off by the edge of the photo or partly behind
> something, with the box covering only its visible part. Give 3 different places, best first, as
> JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "new {class}"}]. Always give 3 places.*
>
> If all three boxes fail a check, the same question is asked again with: *Your earlier boxes for
> the new {class} cannot be used: {box} {reason}; … Give different boxes that follow these
> instructions.*

**Table 1c.** Placement and verification rules of the benchmark (Qwen3-VL box mode). Every other
judgement of plausibility is left to the model's reading of the scene and to the hand audit.

| Step | Rule | Value |
|---|---|---|
| class | already in the photograph (COCO labels and audited additions), never *person*; least-used class first | next class only if no usable box |
| question | one fixed question per class group (22 groups), Box 1 | three boxes, best first |
| check | minimum box (what the generator and verifier handle; measured, see below) | short side ≥ 15 px and ≥ 0.1% of the image |
| check | not thinner than the class usually is (long/short side, either orientation) | ≤ 3 × the class's median COCO elongation |
| check | share of any labelled object or person the box may cover (insertion, not replacement) | ≤ 30% |
| check | maximum box | 25% of the image |
| re-ask | if all three boxes fail a check: the same question plus the failed boxes and reasons | once per class |
| generate | box 1 first; boxes 2 and 3 only after the verifier rejects the previous one | 3 seeds per box |
| verify | the verifier finds the class at the box: score ≥ 0.2; IoU ≥ 0.3 or ≥ 80% inside; not the photograph's own object of the class (IoU ≥ 0.5 ignored) | yes |
| verify | the object's segmented outline fills at least this share of the box | 25% |
| audit | every C4 edit checked by hand after generation | yes |

On the 50 pilot photographs, this procedure gave a usable box for 44 photographs; 36 were settled
by one question and 37 plans come from the first answer, 7 from the re-ask <!-- run:task2_pilot_natural_qwen_free, run_plan_2026-10-21T091847Z, file:paper/results/task2_pilot_natural_qwen_merged_plan.json -->. The six
photographs without a box are crowded or framed tightly (a vase display, a surfer holding a
board, a ski race, a busy motorcycle street, an airplane seen through a window, a zebra herd).
The minimum box size was set from a test rather than assumed: on 42 boxes the model drew in the
pilot (short sides 10–48 px, 3 seeds each), the share of boxes whose edit passed verification did
not fall with size (4–5 of 6 per band from 10–16 px to 40–48 px), and passing edits down to about
15 px were recognisable by eye <!-- run:task2_pilot_natural_qwen_free, file:paper/results/task2_size_limit_test.json -->
[[TBD: confirm on the full run's verification rates by object size]]. For comparison on the same
photographs, the earlier point-based version placed a verified object in 18 of 50 (72% resting on
ground, a table or water, median 1.8% of the image) and random free-space placement in 50 (46%,
0.8%) <!-- run:task2_pilot_natural_v2, run:task2_pilot_inphoto, file:paper/results/task2_pilot_natural_three_way.json -->.
Photographs without a usable box receive no AI edit, so the AI condition covers a subset of the
photographs (C0 and C1 cover all); in the full run, 518 of the 544 photographs received a box <!-- run:task2_insertion_natural_box, run_plan_2026-10-21T115944Z, file:paper/results/task2_insertion_natural_box_plan.json --> (451 from the first answer, 67 after the re-ask); 427 of the 544 photographs received a verified AI object (350 on the model's first box, 57 on the second, 20 on the third), and 425 have all five conditions <!-- run:task2_insertion_natural_box, run_verify_2026-10-23T070146Z, file:paper/results/task2_insertion_natural_box_final.json -->.

**Table 1b.** What each condition tests.

| Cond. | Changed relative to the previous rung | Comparison | What it tests | Used for |
|---|---|---|---|---|
| C0 | — (original file) | C0 | detectors and forensics on unedited real photographs; label audit | baseline, false alarms on real photos |
| C1 | re-encoding with the benchmark pipeline | C1 vs C0 | effect of re-encoding alone | reference for all edits |
| C2 | AI repainting of the box, no object | C2 vs C1 | effect of AI editing without a new object; false "AI" flags on edited background | H2, H3 (control) |
| C3 | a real object composited into the box | C3 vs C2, C4 | compositing a photographed object; real-object negatives in the same box as C4 | H1, H3 (control) |
| C4 | an AI object of a class in the photo, at a natural place (natural placement, ≤ 30% of any real object covered) | C4 vs C1, C2, C3; covered vs not covered | origin of an AI object; occlusion effects (observational) | H1, H2, H3, H4 |

**Generator.** All insertions use SDXL-Inpainting [Podell et al. 2024]. A second seen generator and
a held-out generator were planned but are left to future work (§8). Inpainting operates on a square
window of 1.2× the planned box at the generator's native resolution (a wider 2.5× window mostly
made the generator fill the box with background in our pilot); verification looks at a 2.5×
region. Only pixels inside the dilated rectangular mask are composited back, with a feather that
vanishes at the mask border, so every pixel outside the intended mask is identical to the matched
resave C1.

**Held-out source domain.** To test whether conclusions transfer beyond COCO, we build a small
test-only set of [[TBD: ~100]] Open Images V7 [Kuznetsova et al. 2020] photographs with conditions C1–C4,
using the same generator (SDXL). No model, calibrator or threshold is fit
on this set. Because Open Images boxes are exhaustive only for classes verified at image level, we
score only verified classes and ignore the rest. We use Open Images rather than Objects365 because
the latter is part of Grounding DINO's pretraining data [Liu et al. 2024].

**Encoding policy.** All images in C1–C4 pass through the same decode → edit → JPEG (q=95, 4:4:4)
pipeline, so file format cannot reveal origin.

**Verification and annotation.** An open-vocabulary verifier (OWLv2 [Minderer et al. 2023]), which is
not among the evaluated detectors, confirms each insertion: it must find the target class around the edit with score ≥ 0.2 and
either IoU ≥ 0.3 to the planned box or at least 80% of the detection inside it (inserted objects
are often smaller than the box); a detection that coincides with a same-class original object
does not count, and the object's segmented outline must fill at least a quarter of the box. A sham
edit (C2) passes only if the verifier finds no such object; a detection lying for at least a
quarter of its area on a real object of the class is attributed to that object, not to the sham.
A failed edit is regenerated with a new seed, up to three attempts per box; then the next of the
model's boxes for the photograph is used (the sham and paste controls move with it). A photograph
whose boxes all fail receives no AI edit; there is no fallback to random boxes, other classes or
placement anywhere. In the natural run, 427 of the 518 photographs with a box received a verified AI object, 286 on the first seed of their box; the inserted objects cover a median 0.82% of the image <!-- run:task2_insertion_natural_box, run_verify_2026-10-23T070146Z, file:paper/results/task2_insertion_natural_box_final.json --> [[TBD: verification rate by object size]]. SAM [Kirillov et al. 2023], prompted with the
verifier's box and clipped to the edit mask, provides the inserted object's mask and box. Original
objects covered by an insertion keep their visible mask and visible fraction; below 10% visible they
become ignore regions. We record the actual changed region separately from the intended mask, and
we mark original objects overlapped by more than 10% as *mixed/modified*. A person audits
every AI insertion, judging object presence, realism, placement and changes outside the box; a
random subset gives an unbiased estimate of the verifier's error rate [[TBD: audit results]].
Insertions the auditor rejects are removed from the benchmark. Sham edits are checked by the
verifier only.

*Random-placement comparison run.* Before natural placement, the same 544 photographs were edited
with boxes placed by rules alone (free space or occluding a real object, then anywhere). With
re-planning, every photograph received a verified AI object: 464 from the free-space rule, 38 from
the occlusion rule and 42 placed anywhere; close-ups of a single bed and a single train passed only
with boxes in the image corner, at verifier scores at the threshold
<!-- run:task2_insertion (final verify 2026-10-18 15:02), file:paper/results/task2_insertion_final.json -->.
Many of these edits were implausible (objects in mid-air or inside windows), which motivated the
natural placement above; the run is kept to compare detection and forensics on natural and random
placements [[TBD: decide whether it is reported as a separate condition]].

**Table 1.** Conditions and statistics. [[TBD: images, instances per origin, per condition]]

| Cond. | Description | #images | #photographed | #AI-inserted | #real-pasted |
|---|---|---|---|---|---|
| C0 | untouched | 544 <!-- run:task1_real_baseline (sample/plan/generate, author's laptop, 2026-10-08) --> | [[TBD]] | – | – |
| C1 | matched resave | 544 <!-- run:task1_real_baseline (sample/plan/generate, author's laptop, 2026-10-08) --> | [[TBD]] | – | – |
| C2 | sham edit | [[TBD]] | [[TBD]] | – | – |
| C3 | real paste | [[TBD]] | [[TBD]] | – | [[TBD]] |
| C4 | AI insert | [[TBD]] | [[TBD]] | [[TBD]] | – |

**Figure 1.** [[TBD: example source with C1–C4 panels and instance origin overlays]]

---

## 5. Methods evaluated

**Detectors (frozen, COCO-trained).** Faster R-CNN R50-FPN v2 [Ren et al. 2015], YOLO11-m [Jocher &
Qiu 2024], RT-DETR-R50 [Zhao et al. 2024], and Grounding DINO-T [Liu et al. 2024]. Grounding DINO
receives the same text for every image: the 80 COCO class names, sorted alphabetically and split into
four prompts of 20 names (e.g. "airplane. apple. backpack. …"), one forward pass per prompt because of
its text-length limit. A box's score for a class is the maximum token score over that class name's
tokens; boxes are kept at score ≥ 0.05, and the top 100 per image over all four prompts are evaluated.
The prompt never depends on the image or its edits.

**Hand-crafted forensics.** Each produces a dense map in which higher means more suspicious. For
F3–F5 the map is the absolute deviation of a local statistic from the image median. F1 error level
analysis [Krawetz 2007]; F2 SRM high-pass residual variance
[Fridrich & Kodovsky 2012]; F3 local noise-level inconsistency (wavelet MAD) [Mahdian & Saic 2009];
F4 high-frequency spectral energy [Durall et al. 2020]; F5 JPEG blocking-artifact grid [Li et al.
2009].

**Learned forensics.** L1 TruFor [Guillaro et al. 2023]; L2 a UnivFD-style CLIP probe [Ojha et al.
2023]; L3 a CLIP-feature logistic probe trained on SDXL object crops of our train split (an in-domain
upper reference); L4 B-Free [Guillaro et al. 2025] [[TBD: if runnable]].

**Instance scoring.** Each map-producing method is converted to a per-instance score in three ways:
*image-level* (one score per image), *region-pooled* (mean / 95th percentile inside the instance mask)
and *instance-conditioned* (interior minus a surrounding context ring). Scores are calibrated on val.

**Composed system.** Detector predictions are paired with an origin scorer evaluated on the
*predicted* region. Two thresholds chosen on val define an abstention band.

---

## 6. Metrics

*Semantic*: COCO AP/AR [Lin et al. 2014]. Following COCO, "AP" denotes average precision averaged
over the 80 categories and over IoU thresholds 0.50:0.05:0.95 (i.e. COCO mAP); AP50 uses IoU 0.50 only.
It is reported separately for original and inserted subsets.
*Matched-instance change*: for each original instance, the detected→missed rate, class flips, box IoU
drift and score change from C1 to each condition, stratified by distance to the edit and by
visibility. *Origin*: AUROC, AUPRC, TPR at 5% FPR and the **real-object false-accusation rate** at 80%
TPR, computed both with oracle localisation and end to end. Origin metrics are computed **over
objects only**: positives are AI-inserted objects (C4); negatives are photographed objects,
including the real pasted objects of C3. A C2 box contains no object and is never a positive or a
negative object; instead we report an *edited-background flag rate*, the share of C2 boxes a
method scores above its operating threshold. A method that flags C4 and C2 boxes alike detects
regenerated pixels rather than AI objects, which the main metric alone cannot reveal.
*Joint origin-aware AP*: a true positive
requires the class, IoU ≥ 0.5 and the correct origin. *Calibration*: ECE, Brier score and
risk–coverage (AURC) [Guo et al. 2017; Geifman & El-Yaniv 2017]. All numbers have 95% bootstrap CIs
over source photographs (1000 resamples), because edits of one photograph are correlated.

---

## 7. Results

### 7.1 Baseline: unedited photographs
Before any insertion, we run every detector and forensic method on the untouched (C0) and resaved
(C1) photographs. COCO predates diffusion-based editing, so every object is photographed and every
"AI" flag is a false alarm.
**Table 2a.** Detector AP on 544 unedited COCO val2017 photographs (C0) and their matched resaves
(C1), with per-object detection flips caused only by resaving. Ground truth includes the 77 audit-added labels. <!-- run:task1_real_baseline, file:paper/results/task1_real_baseline_summary_v4_200boot.md -->

| Detector | AP C0 [95% CI] | AP C1 [95% CI] | AP50 C0 | AR100 C0 | lost C0→C1 | gained C0→C1 |
|---|---|---|---|---|---|---|
| Faster R-CNN v2 | 0.500 [0.460, 0.515] | 0.498 [0.460, 0.510] | 0.702 | 0.628 | 0.58% | 0.66% |
| YOLO11-m | 0.530 [0.484, 0.547] | 0.528 [0.487, 0.547] | 0.692 | 0.651 | 0.53% | 0.40% |
| RT-DETR-R50 | 0.512 [0.479, 0.527] | 0.511 [0.476, 0.532] | 0.675 | 0.635 | 0.66% | 0.45% |
| Grounding DINO-T | 0.515 [0.481, 0.532] | 0.515 [0.476, 0.525] | 0.664 | 0.685 | 0.76% | 0.73% |

Flips are over 3,959 photographed objects (3,882 COCO labels + 77 audit additions). An object counts as detected with a same-class prediction
at IoU ≥ 0.5 and score ≥ 0.5. AP intervals are basic bootstrap intervals over source photographs
(200 resamples).

**Table 2b.** Hand-crafted forensic scores on real photographs. Image score is the 99th percentile
minus the median of the map. Per-object contrast is the mean inside the object minus the mean in a
surrounding context ring, over the 3,882 original COCO labels. <!-- run:task1_real_baseline, file:paper/results/task1_real_baseline_summary_v4_200boot.md -->

| Method | image score C0 (mean) | resave shift C1 − C0 [95% CI] | object contrast C1 (mean ± sd) |
|---|---|---|---|
| F1 ELA | 5.05 | +0.188 [0.170, 0.206] (+3.7%) | 0.515 ± 1.33 |
| F2 SRM residual | 3.05 | −0.106 [−0.114, −0.099] (−3.5%) | 0.290 ± 0.81 |
| F3 noise level | 10.38 | −0.089 [−0.142, −0.038] (−0.9%) | 0.254 ± 1.62 |
| F4 FFT high-freq. | 0.365 | +0.0045 [0.0029, 0.0064] (+1.2%) | −0.002 ± 0.045 |
| F5 JPEG blocking grid | 0.532 | −0.025 [−0.027, −0.022] (−4.7%) | −0.003 ± 0.052 |

**Findings.** (i) The detectors reach AP 0.50–0.53 on our 544 sources, and resaving with the
benchmark encoding changes AP by at most 0.002. Resaving alone flips 0.4–0.8% of photographed
objects between detected and missed, in both directions. This is the noise floor against which
insertion-induced changes to real objects (H2) must be judged. (ii) Every hand-crafted method
responds to resaving, with CIs excluding zero: ELA rises by 3.7%, while SRM and the blocking-grid
score fall by 3.5% and 4.7%. Comparing
edited images against the original C0 rather than the matched resave C1 would therefore confound
editing with re-encoding, which justifies C1 as the reference condition. (iii) Real objects are
already more "suspicious" than their surroundings for ELA, SRM and noise level (positive mean
contrast, sd ≈ 1–1.6). Object interiors carry more texture and edges than their context, so a naive
"object stands out" rule would accuse real objects. Origin scores must be calibrated against
photographed objects of similar content, not against zero. The frequency (F4) and blocking-grid
(F5) scores show no such bias (mean contrast ≈ 0), so they may be less confounded by object texture. Fixed-threshold false-alarm rates are
reported in §7.4.
(iv) In 70 of 544 photographs (12.9%), at least two detectors agree on a confident object that COCO
does not label (107 candidates), consistent with [Incomplete Benchmarks 2026]. Visual review confirms 77 as real
missing objects, leaves 26 ambiguous (ignore regions) and rejects 4 (e.g. deer detected as sheep, a
candle as a cup). Adding the confirmed labels raises AP by 0.004–0.006 for every detector, so
unaudited COCO labels slightly understate detector accuracy. <!-- run:task1_real_baseline, file:paper/results/task1_real_baseline_summary_v3_audited.md -->

### 7.2 Do detectors see inserted objects, and does confidence reveal origin? (H1)
**Table 2.** AP / AR on original vs inserted instances per detector and condition. [[TBD]]
**Finding.** [[TBD]]

### 7.3 Do insertions disturb real objects? (H2)
**Figure 2.** Matched-instance change rates (C2, C3, C4 vs C1) by distance to the edit, on original
objects the AI object does not cover; covered objects shown separately by visible fraction. [[TBD]]
**Finding.** [[TBD]]

### 7.4 Can forensics tell origin per object? (H3)
**Table 3.** Origin AUROC / AUPRC / false-accusation rate per method × scoring mode, oracle
localisation. [[TBD]]
**Table 4.** End-to-end joint origin-aware AP and calibration of composed systems. [[TBD]]
**Finding.** [[TBD]]

### 7.5 Generalisation and robustness (H4)
**Table 5.** Held-out source domain: COCO test split vs Open Images (SDXL insertions in both). [[TBD]]
**Figure 3.** AUROC under JPEG 95/75/50, 0.5× resize, blur, screenshot. [[TBD]]
**Finding.** [[TBD]]

### 7.6 Controls: what are the detectors actually responding to?
For each forensic method, the origin score on C4 objects against the edited-background flag rate
on C2 boxes (does it detect AI objects or any regenerated region?), and C4 against C3 objects
(AI-made versus real objects composited the same way). For detectors, false detections inside
empty C2 boxes, and matched-instance changes under C2 versus C4 (does repainting alone disturb
nearby real objects?). [[TBD]]

### 7.7 Origin claims in grounded captions (H5)
**Table 6.** False-AI / false-real / abstention rates for Florence-2 and Qwen2.5-VL across three
prompt settings, including unedited images. [[TBD]]
**Finding.** [[TBD]]

---

## 8. Limitations, ethics and broader impact

InlayFake under-represents close-up, frame-filling photographs (9.3% of sampled sources had no free
space for an insertion, mostly because one object fills the frame), one main source domain (COCO)
with a small held-out Open Images test set, and **a single inpainting model (SDXL)**. Our findings
may therefore be specific to SDXL; whether they transfer to other inpainting models (e.g.
SD-2-Inpainting, FLUX.1-Fill) or commercial editors is not tested here and is left to future work.
Other domains may also behave differently. Pixel evidence cannot recover an
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
- [COinCO 2026] Common Inpainted Objects In-N-Out of Context. CVPR 2026. [verify authors]
- [COCO-Inpaint 2025] COCO-Inpaint: A Benchmark for Detecting and Localizing Inpainting-Based Image Manipulations. arXiv 2504.18361, 2025. [verify venue]
- [INP-X 2026] AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange. arXiv 2602.00192, 2026. [verify]
- [TGIF 2024] TGIF: Text-Guided Inpainting Forgery Dataset. arXiv 2407.11566, 2024. [verify venue]
