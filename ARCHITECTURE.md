# ARCHITECTURE: InlayFake pipeline

> Terminology: C4-occ = the occlusion placement rule of the AI insertion before 18 Oct 2026 (cover 20–50% of one real object); since then it is part of the single AI condition C4.

This document describes how the system is built. `PRD.md` says what and why. Module paths refer to
`src/inlayfake/`. Status markers: ✅ implemented · 🚧 in progress · ⬜ planned.

## 1. Pipeline overview

```
COCO val2017 ──► [1 sample] ──► [2 audit] ──► [3 plan] ──► [4 generate C1–C4] ──► [5 verify+annotate]
                                                                                        │
                                                       ledger.jsonl (ground truth) ◄────┘
                                                                                        │
            ┌──────────────────────────────────┬────────────────────────────────────────┤
            ▼                                  ▼                                        ▼
   [6 detect] 4 detector families     [7 forensics] hand-crafted + learned     [9 vlm] (Stage 2)
            │                                  │                                        │
            └──────────────► [8 compose: detector + origin scorer] ◄────────────────────┘
                                               │
                      [10 postproc robustness] ─► [11 evaluate] ─► metrics.json, tables/*.csv
                                                                    ─► paper/results, paper/figures
```

Each stage is a CLI subcommand: `python -m inlayfake <stage> --config configs/<name>.yaml`. Stages are
idempotent and resumable (skip outputs that already exist).

We use COCO **val2017** as the source pool: COCO-trained detectors were trained on train2017, so
val2017 photos are unseen by them.

**Held-out source domain (week 4).** About 100 Open Images V7 validation/test photos run through the
same stages (`data/openimages.py` adapter → same ledger format with `source_dataset="openimages"`,
test split only). Conditions C1–C4 with G1 (SDXL; G3 dropped 2026-10-17). All classifiers, calibrators and thresholds are fit
on COCO only and applied unchanged.

## 2. Data contract: the provenance ledger ✅ (`ledger.py`)

Each edited image is one `ImageRecord` (JSONL line):

```
ImageRecord
  image_id        "<coco_id>__<condition>__<generator|none>__s<seed>"
  source_id       COCO image id (bootstrap / split grouping key)
  split           train | val | test
  condition       C0 | C1 | C2 | C3 | C4   (C4 covers the former occlusion rule C4-occ, 2026-10-18)
  generator       G1 | G2 | G3 | null
  path, sha256    final image (what models see)
  width, height
  edit            EditInfo | null
     target_category, prompt, negative_prompt, seed, model_id, model_revision, steps, guidance,
     crop_box (inpainting window), intended_mask_path, actual_mask_path,
     donor (C3: donor coco image id + annotation id), attempts, verifier_score
  instances       [Instance]
  ignore_regions  [{category, bbox}]  (label audit: ignore/unreviewed; scored as COCO crowd regions)
  postproc        null | {"jpeg": 75} | {"resize": 0.5} | ...
  source_dataset  coco | openimages   (openimages ⇒ split must be test)
```

```
Instance
  instance_id       stable across conditions for original objects ("coco:<ann_id>"); "ins:<image_id>" for inserted
  category          COCO category name
  bbox              final-image visible extent [x, y, w, h]
  mask_path / rle   final visible mask (optional)
  origin            photographed | ai_inserted | real_pasted | mixed_modified | unresolved
  visible_fraction  1.0 unless occluded by an insertion (C4-occ)
  iscrowd           COCO crowd flag (ignored in scoring)
```

Rules (from the brief, §4):
- *Intended* edit mask and *actual* changed-region mask are stored separately.
- `real_pasted` is not AI, but it is not from the source scene either: a separate origin.
- `mixed_modified` marks original objects that the editor altered (actual mask overlaps them by
  > 10% of their area). These are excluded from the primary binary origin metric and reported
  separately.
- Model abstention is a **prediction**, never a ground-truth origin.

## 3. Stages

### 3.1 `sample` ✅ (`data/coco.py`, `data/sampling.py`)
Select sources from val2017 that have: ≥ 1 non-crowd annotated object; no person instance covering
> 30% of the image (keeps people out of the edit focus); and a free region that fits the planned
insertion. Assign the split by hashing `source_id` (deterministic). Output: `data/sources.jsonl`.

### 3.1b `sample --dataset openimages` ⬜ (`data/openimages.py`)
Download only what is needed: the class-description CSV, the segmentation and box CSVs for the
validation split, and the selected images (by ID, from the public `open-images-dataset` bucket, or
FiftyOne's zoo with `classes=` and `max_samples=`). Map the 10 Open Images classes to COCO names
(`configs/default.yaml: ood.class_map`). Keep photos with ≥ 1 mapped instance that has a mask. Use
the image-level labels to build ignore regions for unverified classes.

### 3.2 `audit` ✅ (`data/audit.py`, `stages/audit.py`)
Implemented rule: a detection with score ≥ 0.7 that ≥ 2 detectors agree on (same class, IoU ≥ 0.5)
and that matches no GT box (same-class IoU < 0.3 and any-class IoU < 0.5). It outputs
`audit.html` (crops) and `audit_decisions.csv` (add / ignore / reject / ?; "?" counts as ignore).
Scoring stages apply **one** decisions file to every run on the same photos
(`common.decisions_path`): `audit.decisions`, else `placement.audit_decisions`, else the run's own
`<work>/audit_decisions.csv`. The week-2 config names the week-1 file
(`data/inlayfake/audit_decisions.csv`), so a changed decision reaches both weeks at their next
`evaluate`; a named file that is missing stops the run.
Decisions are applied on load to every condition of that source (`stages/common.load_records`).

Run a detector ensemble on C0. A high-confidence detection (≥ 0.7) with IoU < 0.3 against every GT
box becomes a **candidate missing label**. The author reviews the candidates in a simple HTML
contact sheet. Confirmed ones are added to the GT as `photographed`; unresolved ones become
`ignore_regions`. (Lesson from *Object Detection Benchmarks are Incomplete*.)

### 3.3 `plan` ✅ (`edit/placement.py`)
For each source, pick a target category, then a box. `placement.class_choice: in_photo`
(`configs/pilot_inphoto.yaml`; the decided design from 2026-10-16): a class already in the photo's
audited GT (non-crowd COCO labels plus week-1 audit "add" rows from `placement.audit_decisions`),
excluding `placement.exclude_classes` (person); the least-used class so far is tried first, ties
in seeded random order. Audit add / ignore boxes also block free placement. `fixed` (week 1,
`default.yaml`): cycle through `sampling.categories`, balanced; with it the code reproduces the
week-1 and first-pilot plans byte for byte. Classes not in `support.categories` use
`support.default_surfaces`. Box rules:
- size: sampled from that category's COCO area distribution (relative to image area), clipped to
  [32², 0.25·HW] pixels; aspect ratio from the category's median.
- **C4**: box with IoU = 0 against every GT box and an 8-px margin; inside the image.
- **C4-occ**: box that covers 20–50% of the area of one chosen original instance (the "occluded" one).
- C2 and C3 reuse the C4 box, so all conditions share **one geometric plan**.
- **Surface placement** (`placement.support`, `edit/support.py`; off in `default.yaml`, on in
  `configs/pilot_surface.yaml`): the C4 and C4-occ boxes must also *rest on* a surface from the
  COCO-Stuff human labels (`stuff_val2017.json`). "On": at least 50% of the band from 15% of the
  box height above its bottom edge to 5% below it (central 80% of the width) is surface. Groups:
  ground (floor, carpet, rug, grass, dirt, sand, pavement, …), tabletop (table, desk, counter,
  shelf), wall ("inside": 50% of the box is wall). Allowed groups per class are in the config
  (e.g. chair → ground; cup → tabletop; clock → wall, tabletop). The box is anchored on a random
  surface pixel (bottom-centre for "on"). Why: in the first SDXL pilot, free-space boxes in sky
  or mid-road were mostly filled with background (5/84 verified). Thing surfaces (dining table,
  bed) are not used, because C4 must not overlap any GT box.

### 3.4 `generate` ✅ (`edit/`, `stages/generate.py`)
- `encode.py` **Encoding policy** (critical): every image a model sees in C1–C4 is produced by
  `decode(original) → float RGB → [edit] → uint8 → JPEG q=95, 4:4:4, PIL`. C1 is exactly this with no
  edit. The format and quality are therefore identical across origins.
- `compose.py` **Local inpainting**: crop a window of 2.5× the target box (square, clipped), resize
  it to the generator's native resolution (1024 for SDXL/FLUX, 512 for SD2), inpaint with a dilated,
  rectangular mask (box + 6 px), resize back, and alpha-composite **only inside the mask** with a
  linear 3-px feather that reaches 0 at the mask border. Pixels outside the dilated mask are
  bit-identical to C1 before encoding, and after JPEG they decode identically except in the 8×8
  blocks that touch the mask (this is tested). Ablation `edit.composite: window` keeps the generator
  output for the whole window, i.e. VAE re-encoding outside the mask.
- `generators.py`: wrappers with one interface, `inpaint(image, mask, prompt, seed) -> image`:
  - G1 `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` — **the only generator in the benchmark
    since 2026-10-17** (scope change, `PRD.md` §4). G2 and G3 below stay in the code and config for
    future work and are not run (`generation.insert_generators: [G1]`).
  - G2 `stabilityai/stable-diffusion-2-inpainting` (verify availability on HF; fallback
    `kandinsky-community/kandinsky-2-2-decoder-inpaint`)
  - G3 (held out) `black-forest-labs/FLUX.1-Fill-dev` (gated). It runs on **test-split sources
    only**. On 8 GB it needs `offload: sequential` (slow) or a GGUF-quantised transformer
    (`gguf_transformer`); guidance 30, 28 steps, no negative prompt.
  - Loop order is generator-major (one model on the GPU at a time). The stage is resumable: each
    image writes `records/<image_id>.json`, and existing records are skipped.
  - `FakeInpainter` (`model_id: fake`, `configs/dryrun.yaml`) lets the whole pipeline run on CPU.
  - Prompt template: `"a photo of a {category}, realistic, natural lighting"`. Negative: `"text,
    watermark, cartoon, painting"`. 30 steps; per-category guidance from the config.
- C2 (sham, in `stages/generate.py`): same window, mask and generator; prompt `"empty background, nothing"`.
- `paste.py` (C3): donor = a COCO instance of the same category from another val2017 image, with
  enough area and not truncated. It is cut out with its GT mask, scaled to the planned box and
  alpha-pasted with a 3-px feather. The aspect ratio is kept (within 25% of the plan box; fitted
  inside the box, bottom-centre aligned).
- `stages/preview.py` ✅: `<work>/preview.html` contact sheet (planned boxes in yellow, inserted or
  pasted instances in red) for visual checks of the pilot.

### 3.5 `verify` + `annotate` ✅ (`edit/verify.py`, `stages/verify.py`, `stages/insert.py`)
Implemented as below, with these specifics: `verify.jsonl` logs one row per checked image;
`generate` reads it and regenerates failed jobs with seed attempt n+1 (max `edit.max_attempts` = 3),
skipping model loading when nothing is pending. `insert` runs generate → verify in rounds. Sham
edits (C2) pass only if OWLv2 finds *no* target object. The final `ledger.jsonl` carries C0/C1
unchanged (tested), C3 with its pasted mask, and one passed record per C2/C4/C4-occ job. The human audit
is `audit_edits.html` + `audit_edits.csv` (before/after crops; object_ok / realistic / other_change).
Verifier and segmenter are injectable (`model_id: fake` in `configs/dryrun.yaml` for CPU tests).

- **Insertion success**: OWLv2 (`google/owlv2-base-patch16-ensemble`, **not** an evaluated detector)
  queried with the category name inside a 2.5× region around the plan (`verify.window_scale`,
  independent of the generation window `edit.window_scale`). Pass = a box with score ≥ τ (0.2)
  and either IoU ≥ 0.3 to the plan or ≥ 80% of its area inside the plan (`verify.min_inside`,
  added 2026-10-16 for objects smaller than the box); a box with IoU ≥ 0.5 to a same-class
  original object never counts (`verify.original_iou`). On failure, retry with a new seed
  (max 3), then drop and log it. Attempts are verified in order; the first pass ends the job.
- **Re-plan on failure** (`stages/replan.py`, `edit.max_replans`, 0 by default, 3 in
  `pilot_inphoto.yaml`): after all seeds fail on a box, the job gets a new box version (stored in
  `<work>/replans.jsonl`, recorded as `EditInfo.plan_version`). Tracks: *free* (C4 leads, C2 and
  C3 follow so the controls stay on the C4 box) and *occ* (C4-occ). New boxes rest on a surface of
  the class when COCO-Stuff is available (free space otherwise) and are sized like the photo's
  own instance of the class (×0.8–1.2). Only the current box counts in the final ledger.
  After the new boxes, `edit.max_class_switches` further versions switch to another class
  already in the photo (smallest first; class stored per version; C2/C3 follow C4). Versions
  where no box fits are skipped. In `in_photo` plans, a photo without free space may get a
  C4-occ-only plan (`box_free` null), searched with its own generator so other plans are unchanged.
  When the class's usual size cannot cover 20% of any object (e.g. one train filling the frame),
  the C4-occ box is sized from the object it covers (`placement.occluding_sized_to_target`: 30–60% of
  that object's box area, the class's aspect ratio).
  Last resort, `edit.max_outside_classes`: classes not in the photo (`placement.outside_classes`,
  default the 10 original categories), only if the photo has no in-photo pass in C4 or C4-occ and both
  edits are exhausted; also used by the plan stage for photos with no insertable class. A
  verified edit is always kept (`track_version` returns its passing version first); the
  restriction only decides what to try next.
  `EditInfo.class_in_photo` is False for these edits.
  C4-occ fill, `placement.occlusion_fill` (2026-10-18): after the main search, plans without a C4-occ
  box get one (6 sizes, then a box sized from the covered object; covered objects ≥ 32 px instead
  of 64; 20–50% of one object, ≤ 5% of any other), with an own seeded generator per photo, so
  existing plans stay identical. C4-occ re-plans use the same minimum.
  C4 fill, `placement.free_fill` (2026-10-18): plans without a C4 box get one (plan class first,
  then the photo's other classes; 6 sizes per class; C4 rules and `margin_px` unchanged; own
  generator per photo). Another class is stored as `free_category`, used by C2/C3/C4 and their
  re-plans (`replan.track_plan`), while C4-occ keeps the plan class.
  Very last resort, `placement.allow_anywhere`: if no box satisfies the rules, a C4-occ box anywhere
  in the image with no overlap rule (`placement.place_anywhere`; plan stage and C4-occ re-plans;
  `EditInfo.placement_rule = "anywhere"`). C4 is never placed this way.
- **Natural placement** (`placement.natural.enabled`, `stages/natural.py`, `edit/natural.py`,
  `edit/scene.py`; author decision 2026-10-19; pilot `configs/pilot_natural.yaml`): `plan` loads
  Depth Anything V2 Small and a pointing VLM (Molmo-7B-D 4-bit; `internvl` and `fake` backends)
  once and, for up to `classes_per_photo` classes already in the photo, asks the fixed prompt
  `Point to {where} where another {class} could naturally {how}.` (`configs/placement_prompts.yaml`:
  22 class groups covering all 79 non-person COCO classes). Each point (contact point; centre for
  `air` classes pointed at sky) becomes a box: reference = the photo's same-class object not cut
  off and at most 30% hidden, nearest in depth; height × clamp(nearness(point) /
  nearness(reference base), 0.5, 2), reference shape kept; COCO-typical size if no reference;
  `min_side_px`, `max_area_frac`, shifted inside the image by at most 15%. Rules (`box_rules`):
  no person, ≤ 15% of any real object (COCO + audit rows), base band not on window / mirror /
  ceiling / sky COCO-Stuff pixels (sky allowed for `air`). Valid candidates in order are the free
  track's versions 0, 1, 2, … (`replan._ensure_natural`), so retries move to the next natural
  spot; no candidate means no plan. `verify` then runs `realism_pass`: the VLM answers four yes/no
  questions on the edit with the AI object boxed in red; any unexpected answer turns the attempt
  into a failure (`realism.jsonl`, applied by `verify_state`). C5 is not used.
- **Merged AI condition** (`placement.merged.enabled`, full run since 2026-10-18; author scope
  change): C4 and C4-occ become one AI condition, labelled C4 in the final ledger. The free track
  carries it (`replan.merged_version`): a verified C4 at any version ≤ `max_version` is kept;
  else the first verified C4-occ is adopted as free version 100 (`replan.ADOPT`, box and class copied
  from the C4-occ version, `placement_rule: adopted_c5`), and no new C4 is generated; else versions
  101 … 100 + `placement.merged.versions` (6) place a box anywhere (`place_anywhere`, size
  alternating the photo's own instance ×0.8–1.2 and a typical COCO size, classes cycling through
  the photo's own classes, free-track class first), 3 seeds each, verified as usual. C4-occ is no
  longer generated. C2 and C3 at a merged version wait until its AI edit is verified
  (`replan.leader_passed`), then follow its box. `verify` writes one AI record per photo,
  relabelled C4, with `EditInfo.merged_source` = free / occluding / anywhere; other C4-occ records stay out
  of the ledger. `verify` reports `merged_ai_source`, and `needs_retry` counts pending AI and C2
  jobs (`replan.merged_pending`). `preview` shows the merged view (Show dropdown by AI source and
  missing condition). `placement.merged.avoid_same_class_from` (7 in the full run, 2026-10-18):
  from that box on, a merged box may cover at most 30% of its area with a real object of its
  class (shrinks ×0.8 if no spot is found), since a box on the real object cannot pass verify.
  `placement.merged.avoid_same_class_until` (18 in the full run): later boxes are plain anywhere.
  `placement.merged.corner_from` (22 in the full run): from that box on, the box sits in the
  bottom-left corner of the image (any overlap; row field `corner: bottom_left`).
  `placement.merged.c2_attempts` (6 in the full run): seeds for C2 at a merged box
  (`replan.attempts_for`). `verify.sham_original_overlap` (0.25 in the full run): a C2 detection
  lying at least that share of its area on a real object of the class does not count as a
  produced object (a sham box may cover part of a real object since the merge).
- **Inserted-object GT**: SAM (ViT-B) prompted with the OWLv2 box, intersected with the dilated edit
  mask → mask → tight bbox.
- **Actual changed mask**: `|decode(Cx) − decode(C1)| > 4` (per channel, max), morphological
  open.
- **Occlusion update (C4-occ)**: occluded instance's visible mask = original mask − inserted mask;
  recompute bbox and visible_fraction; if < 0.1, mark it ignore.
- **mixed_modified**: an original instance whose actual-changed overlap is > 10% of its area.
- Human audit: `annotate/contact_sheet.py` builds an HTML grid of 10% random edits with checkboxes;
  the results are stored in `data/audit_results.csv`.

### 3.6 `detect` ✅ (`detect/`, `stages/detect.py`; output `runs/<run>/detections/<det>.jsonl`)
Common interface: `predict(image) -> list[Detection(bbox, category, score, logits?)]`, mapped to COCO
category names.

| Family | Model | Source |
|---|---|---|
| Two-stage CNN | Faster R-CNN R50-FPN v2 | torchvision (COCO weights) |
| One-stage | YOLO11-m | ultralytics |
| DETR-style | RT-DETR R50 | HF `PekingU/rtdetr_r50vd` |
| Open-vocab | Grounding DINO tiny | HF `IDEA-Research/grounding-dino-tiny`, prompted with the 80 COCO names (alphabetical, 4 chunks of 20, `"a. b. c."`, same for every image); labels decoded from token logits (max over a name's tokens), score ≥ 0.05, top 100 |

Score threshold 0.001 for AP; outputs are saved as COCO results JSON per (run, condition).

### 3.7 `forensics` ✅ hand-crafted F1–F5 · ⬜ learned L1–L4 (`forensics/`, `stages/forensics.py`)
Every method outputs either a **pixel map** `M ∈ [0,1]^{H×W}` or an **image/crop score**.

Hand-crafted ("traditional") (`forensics/handcrafted.py`, numpy + PIL only, about 0.1 s per 640×480
image for all five). F3–F5 are reported as |local value − image median| so that "unusual relative to
the rest of this photo" scores high:
| ID | Method | Map |
|---|---|---|
| F1 | Error Level Analysis (resave q=90, abs diff) | ELA energy |
| F2 | SRM noise residual (3 high-pass kernels) local variance | residual variance |
| F3 | Local noise level: wavelet HH MAD (Donoho) in 32-px blocks | noise σ per block |
| F4 | Frequency: per-patch high-frequency FFT energy ratio | spectral ratio |
| F5 | JPEG blocking-artifact grid (BAG) / JPEG ghost | grid misalignment |

Learned (`forensics/learned/`):
| ID | Method | Output |
|---|---|---|
| L1 | TruFor (official code + weights) | pixel map + image score |
| L2 | UnivFD-style: CLIP ViT-L/14 features + linear probe | crop score |
| L3 | **Ours-baseline**: CLIP ViT-L/14 (frozen) + logistic regression on object crops vs real-object crops from train split (G1 SDXL only since 2026-10-17) | crop score |
| L4 | (optional) B-Free, if public weights run on the laptop | crop score |

**Per-instance origin score** (`forensics/instance_scores.py`). For an instance mask `m`:
- *image-level*: one score per image, shared by every instance (naive baseline);
- *region-pooled*: `mean(M[m])` and `p95(M[m])`;
- *instance-conditioned (contrast)*: `mean(M[m]) − mean(M[ring(m)])`, where `ring` is a band of 15% of
  the box size around the mask (outside it), plus interior vs boundary-band split for ablation;
- crop methods: crop = bbox expanded 1.3×, resized to 224.
Scores map to probabilities via **calibration fit on val** (Platt or isotonic).

### 3.8 `compose` ⬜ (`compose.py`)
End-to-end system = detector predictions → for each detection, an origin score computed on the
*predicted* box/mask (SAM-prompted) → record `{class, box, p_ai, decision ∈ {photographed, ai, abstain}}`.
Abstain when `p_ai ∈ (τ_lo, τ_hi)`, with thresholds chosen on val.

### 3.9 `vlm` ⬜ (`vlm/`)
Florence-2-large (`<OD>`, `<DENSE_REGION_CAPTION>`) and Qwen2.5-VL-3B (4-bit), three prompts:
plain caption, grounded caption, origin-aware caption requesting JSON
`[{id, class, box, origin: photographed|ai|unknown, evidence}]` followed by prose. Unedited C1 images
are included at 50%.

### 3.10 `postproc` ⬜ (`postproc.py`)
Applied on test images only, after generation: JPEG {95, 75, 50}, resize {0.5}, Gaussian blur σ=1,
"screenshot" (resize 0.8 + PNG). Each one creates a derived ImageRecord with `postproc` set.

### 3.10b `viewer` ✅ (`stages/viewer.py`, template `stages/viewer_page.html`)
Writes `runs/<run>/viewer/index.html` + `data.js`, a static page opened straight from disk (no
server). It shows results, not a tuning tool. At the top it renders the run's `summary.md` and the
totals from `tables/images_used.md`. Below that, per photo (sorted by COCO id, 10 per page), it shows
a ground-truth panel (COCO labels, audit additions, ignore/crowd regions, inserted/pasted objects)
and one panel per detector. Boxes are drawn only at the evaluation operating point
(score ≥ `eval.recall_score` = 0.5), using the same found-rule as `evaluate.detected` and `report`
(same class, IoU ≥ `eval.iou_threshold`). Each panel's "found X/N" therefore equals the
corresponding `images_used.md` cell; a test enforces this. Colours: green = box on a label, orange =
box on no label (false positive or unlabelled object), grey = box on an ignore region, dashed = label
not found. Controls: condition (C0/C1, later C2–C4 per generator), COCO id search, pages,
click-to-enlarge.
`viewer --share` writes a self-contained copy to the week's folder `docs/results/week<N>_<topic>/`
(config `site` block; `inlayfake.site.share_dir`) instead: the same page and
data plus downscaled JPEG copies of the photos (max side 640, quality 80), so it can be committed,
zipped or served with GitHub Pages without the full `data/` and `runs/` folders.

### 3.11 `evaluate` 🚧 (Task 1 ✅, Task 3 detector metrics ✅, origin/joint/calibration ⬜) (`eval/`, `stages/evaluate.py`)
Task 1 version ✅: AP/AP50/AP75/AR100 per detector × condition (95% CI from grouped bootstrap over
sources, `eval.ap_bootstrap` resamples), C0→C1 per-object detection flips, forensic score
distributions on real photos, and paired C1 − C0 shift. Writes `summary.md`, `metrics.json` and
`tables/*.csv`. Origin metrics (AUROC, joint AP, calibration) come in Task 3, once AI images exist.
Task 3 detector metrics ✅ (`eval/matched.py`, `eval/origin.py`, 2026-10-23; run only when the
ledger has C2–C4): subset AP on the photos with every condition (original objects per
condition; AI-inserted in C4; pasted in C3; the other origins are ignore regions); H1 = per-object
score of the best same-class detection (oracle localisation, IoU ≥ 0.5) for the AI object vs its
same-class photographed "twins", all photographed objects of the image, and the C3 paste at the
same box: recall at `eval.recall_score`, paired recall difference, AUROC; H2 = every original
object followed C1 → C0/C2/C3/C4 (lost, gained, flip, class flip, score change, box drift),
stratified by the edit footprint (plan box + `edit.mask_dilate_px` + one 8-px JPEG block:
covered / uncovered near / uncovered far / mixed-modified), and paired C4 − C2 / C3 / C0
contrasts on the same uncovered objects. All CIs: grouped bootstrap over source photos
(`grouped_mean_ci`, `grouped_bootstrap_fn`). AP bootstrap re-accumulates one COCOeval per
resample instead of re-evaluating (identical numbers, tested; ~5× faster). All-object AP is kept
for `eval.all_objects_ap_conditions` (C0, C1) only, since it mixes origins in edited images.
| Module | Metric |
|---|---|
| `semantic.py` | COCO AP/AP50/recall via pycocotools, computed on **subsets**: original instances only, inserted only (other GT set to ignore) |
| `matched.py` | For each original instance across C1 → Cx: detected/missed flip, class flip, box IoU drift, score delta, duplicates; stratified by distance to edit and visible_fraction |
| `origin.py` | AUROC, AUPRC, TPR@FPR=5%, **real-object false-accusation rate** at fixed TPR=80%; on GT instances (oracle localisation) and end-to-end |
| `joint_ap.py` | Origin-aware AP: TP needs class match, IoU ≥ 0.5 and correct origin; score = det_score × p(predicted origin); macro over {photographed, ai_inserted} |
| `calibration.py` | ECE (15 bins), Brier, risk–coverage curve, AURC |
| `captions.py` | false-AI claims on real objects, false-real claims on inserted objects, missed origin statements, abstention coverage |
| `stages/report.py` | per-image table of a run's C0 photos: split, planned insert, labelled objects, objects found per detector (same rule as recall) → `tables/images_used.{csv,md}` |
| `bootstrap.py` | 1000 resamples grouped by `source_id` → 95% CI for every reported number |

## 4. Configuration

`configs/default.yaml` holds every path, seed, model ID, threshold and sample size.
`configs/pilot.yaml` overrides it for the 50-source pilot. A run copies the resolved config into
`runs/<run_id>/config.yaml`.

## 5. Run provenance (`runlog.py`) ✅
`runs/<run_id>/run.json`: git commit and dirty flag, UTC time, resolved config hash, seeds, model IDs
and revisions, package versions (torch, diffusers, transformers, ultralytics), GPU name and VRAM,
wall time.

## 6. Compute budget (estimates to validate in the pilot)

| Item | Count | Est. time on an 8 GB laptop GPU |
|---|---|---|
| G1 SDXL (C2, C4, C4-occ) | 544 sources: 1,355 in round 1 + retries (~5,100 est.) | measured ~18–19 s each; round 1 took 6 h 53 min (run task2_insertion) |
| G2 SD2, G3 FLUX Fill | — | dropped 2026-10-17 (SDXL only) |
| Detectors | ~5k images × 4 | ~1–2 h |
| Forensics hand-crafted | ~5k (CPU) | ~1 h |
| TruFor / CLIP | ~5k | ~1–2 h |

Run generation overnight and resume with the idempotent stages.

## 7. Testing strategy
`tests/` covers the ledger round-trip, placement geometry, compositing (pixels outside the mask are
unchanged), the encoding policy, every metric against hand-computed values, and grouped bootstrap.
The tests run without GPU or weights; model wrappers are tested with tiny fake models.
