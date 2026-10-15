# ARCHITECTURE: InlayFake pipeline

This document describes how the system is built. `PRD.md` says what and why. Module paths refer to
`src/inlayfake/`. Status markers: ✅ implemented · 🚧 in progress · ⬜ planned.

## 1. Pipeline overview

```
COCO val2017 ──► [1 sample] ──► [2 audit] ──► [3 plan] ──► [4 generate C1–C5] ──► [5 verify+annotate]
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
test split only). Conditions C1–C4 with G1 and G3. All classifiers, calibrators and thresholds are fit
on COCO only and applied unchanged.

## 2. Data contract: the provenance ledger ✅ (`ledger.py`)

Each edited image is one `ImageRecord` (JSONL line):

```
ImageRecord
  image_id        "<coco_id>__<condition>__<generator|none>__s<seed>"
  source_id       COCO image id (bootstrap / split grouping key)
  split           train | val | test
  condition       C0 | C1 | C2 | C3 | C4 | C5
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
  visible_fraction  1.0 unless occluded by an insertion (C5)
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
Decisions are applied on load to every condition of that source (`stages/common.load_records`).

Run a detector ensemble on C0. A high-confidence detection (≥ 0.7) with IoU < 0.3 against every GT
box becomes a **candidate missing label**. The author reviews the candidates in a simple HTML
contact sheet. Confirmed ones are added to the GT as `photographed`; unresolved ones become
`ignore_regions`. (Lesson from *Object Detection Benchmarks are Incomplete*.)

### 3.3 `plan` ✅ (`edit/placement.py`)
For each source, pick a target category (balanced across categories), then a box:
- size: sampled from that category's COCO area distribution (relative to image area), clipped to
  [32², 0.25·HW] pixels; aspect ratio from the category's median.
- **C4**: box with IoU = 0 against every GT box and an 8-px margin; inside the image.
- **C5**: box that covers 20–50% of the area of one chosen original instance (the "occluded" one).
- C2 and C3 reuse the C4 box, so all conditions share **one geometric plan**.

### 3.4 `generate` ✅ (`edit/`, `stages/generate.py`)
- `encode.py` **Encoding policy** (critical): every image a model sees in C1–C5 is produced by
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
  - G1 `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
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
unchanged (tested), C3 with its pasted mask, and one passed record per C2/C4/C5 job. The human audit
is `audit_edits.html` + `audit_edits.csv` (before/after crops; object_ok / realistic / other_change).
Verifier and segmenter are injectable (`model_id: fake` in `configs/dryrun.yaml` for CPU tests).

- **Insertion success**: OWLv2 (`google/owlv2-base-patch16-ensemble`, **not** an evaluated detector)
  queried with the category name inside the crop window. Pass = a box with IoU ≥ 0.3 to the plan and
  score ≥ τ. On failure, retry with a new seed (max 3), then drop and log it.
- **Inserted-object GT**: SAM (ViT-B) prompted with the OWLv2 box, intersected with the dilated edit
  mask → mask → tight bbox.
- **Actual changed mask**: `|decode(Cx) − decode(C1)| > 4` (per channel, max), morphological
  open.
- **Occlusion update (C5)**: occluded instance's visible mask = original mask − inserted mask;
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
| L3 | **Ours-baseline**: CLIP ViT-L/14 (frozen) + logistic regression on object crops vs real-object crops from train split (G1, G2) | crop score |
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
not found. Controls: condition (C0/C1, later C2–C5 per generator), COCO id search, pages,
click-to-enlarge.
`viewer --share` writes a self-contained copy to `docs/viewer/<run>/` instead: the same page and
data plus downscaled JPEG copies of the photos (max side 640, quality 80), so it can be committed,
zipped or served with GitHub Pages without the full `data/` and `runs/` folders.

### 3.11 `evaluate` 🚧 (`eval/`, `stages/evaluate.py`)
Task 1 version ✅: AP/AP50/AP75/AR100 per detector × condition (95% CI from grouped bootstrap over
sources, `eval.ap_bootstrap` resamples), C0→C1 per-object detection flips, forensic score
distributions on real photos, and paired C1 − C0 shift. Writes `summary.md`, `metrics.json` and
`tables/*.csv`. Origin metrics (AUROC, joint AP, calibration) come in Task 3, once AI images exist.
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
| G1 SDXL (C2, C4, C5) | 600 × 3 + retries | ~15 s each → ~9 h |
| G2 SD2 (C4) | 600 + retries | ~5 s each → ~1 h |
| G3 FLUX Fill (C4, test split only) | ~120 | 60–180 s each (sequential offload) → 2–6 h |
| Detectors | ~5k images × 4 | ~1–2 h |
| Forensics hand-crafted | ~5k (CPU) | ~1 h |
| TruFor / CLIP | ~5k | ~1–2 h |

Run generation overnight and resume with the idempotent stages.

## 7. Testing strategy
`tests/` covers the ledger round-trip, placement geometry, compositing (pixels outside the mask are
unchanged), the encoding policy, every metric against hand-computed values, and grouped bootstrap.
The tests run without GPU or weights; model wrappers are tested with tiny fake models.
