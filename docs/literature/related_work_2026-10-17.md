# Related work scan and positioning (2026-10-17)

Three web searches on 2026-10-17: (1) CVPR 2026, (2) local/partial AI-edit datasets and methods
2022–2026, (3) adjacent areas (detectors on inserted objects, MLLM forensics, calibration,
matched comparisons, provenance).

**Access limits:** OpenReview, openaccess.thecvf.com, cvpr.thecvf.com and arxiv.org could not be
opened from the cloud sandbox, so venues and numbers come from search-engine snippets of those
pages (and a few GitHub READMEs). Nothing here has been checked against the PDFs. Every entry
must stay `[verify]` until someone reads the paper.
Venue "unconfirmed" = no proceedings page seen.

## 1. Closest prior work (must be cited and contrasted)

| Work | Venue | What it does | Same as InlayFake | Different from InlayFake |
|---|---|---|---|---|
| **COinCO**: Common Inpainted Objects In-N-Out of Context | CVPR 2026 | 97,722 COCO images, one object replaced by SD inpainting with a COCO class; an object detector checks the new object appeared (regenerate on failure); LVLMs label it in/out of context; context helps localizers (CAT-Net, TruFor, PSCC-Net, ManTra-Net) | COCO, object-level, detector in the generation loop | **replaces** an existing object (we **add** a same-class object in free space); detector used only as a filter (we evaluate frozen detectors and use an independent verifier, OWLv2); no matched controls; no per-object origin for every object; no false-accusation or abstention metrics |
| **TGIF / TGIF2**: text-guided inpainting forgery | arXiv 2407.11566 / 2603.28613 (venue unconfirmed) | COCO val2017, SD2/SDXL/Firefly (TGIF2 adds FLUX); each edit released **spliced** (inpainted pixels pasted back) and **fully regenerated** | the spliced-vs-regenerated pairing ≈ our mask-only composite vs `composite: window` | no real-object paste, no sham edit, no per-object labels, no detectors |
| **INP-X**: detectors over-rely on global artifacts | arXiv 2602.00192 (unconfirmed) | restoring original pixels outside the mask drops commercial detectors from >91% to ~55%: they key on the VAE's global spectral shift | direct evidence for our matched-control design (C1 resave, C2 sham, mask-only composite) | image-level only |
| **COCO-Inpaint** | arXiv 2504.18361 (ACM MM 2025?) | 258K COCO inpaints, 6 inpainters incl. SDXL and FLUX.1-Fill, 4 mask strategies | COCO + SDXL inpainting localisation | pixel masks only, no controls, no detectors |
| **FragFake** ("Can VLMs detect and localize fine-grained AI-edited images?") | arXiv 2505.15644 (unconfirmed) | object addition/replacement on COCO/ADE20K; the VLM names the edited object | object-level "which object was added" | only the edited object is labelled; untouched objects are not scored; no controls |
| **OpenSDI** | CVPR 2025 | 300K real + 450K fake, local and global edits, train SD1.5 / test 5 generators incl. FLUX | held-out generator, local edits | Megalith reals, pixel masks, no controls |
| **SIDA / SID-Set** | CVPR 2025 | real / synthetic / tampered + mask + explanation | object replacement + explanation | image-level 3-way label |
| **Semi-Truths** | NeurIPS 2024 D&B | 27.6K reals, graded local AI edits with metadata | graded local edits | image-level detection |
| **MetaOD** | ASE 2020 | pastes real objects to test detectors | real-object paste + frozen detectors | no AI objects, no forensics |
| **Object Detection Benchmarks are Incomplete** | arXiv 2609.21822 (NeurIPS 2026 per poster page) | COCO missing ~40% objects | our week-1 label audit (77 added) | detection labels only |

Also relevant: **B-Free** (CVPR 2025; aligned real/fake training from self-reconstructions and
inpaints of COCO/RAISE: closest in spirit to matched controls, image level); **GIM** (AAAI 2025;
degradations applied to both sides); **DiffSeg30k**, **Impostor** (LVIS), **LocateEdit-Bench**,
**DEAL-300K** (authentic-only split), **DiQuID/SAGI-D** (all arXiv, local-edit localisation);
**Pandolfini et al.** arXiv 2512.16688 (small inpaints are hard for whole-image detectors).

## 2. CVPR 2026 papers found (main conference unless marked)

- **Locate-Then-Examine** (Ji et al.): VLM proposes suspicious boxes then re-examines; TRACE dataset
  with COCO/ImageNet reals; targets whole-image fakes, not insertions.
- **IFA-Net** (Zhang et al.): frozen MAE "realness" prior, pixel localisation of diffusion
  inpainting; tested on CocoGlide and others.
- **FFIM** (Chen et al.): makes diffusion edits *easier* to detect (forensic-friendly editing).
- **DeepfakeJudge** (Kuckreja et al.): MLLM-as-judge for forensic explanations; Open Images reals.
- **Omni-Fake**: real / partially manipulated / synthetic + masks, multimodal.
- **DanceText**: instance-level localisation of AI-counterfeited *text* images.
- **SOC**: synthetic object compositions to *train* detectors.
- **COinCO**: see table.
- Findings track: **IFDL-VLM** (VLM priors barely help, or hurt, forgery localisation: relevant to
  our Stage-2 VLM study); **UniShield**; **ForenDeX**; **RITA/HSIM**; **CDFL**; **Source-Asymmetry
  Confidence Gap** (detector confidence higher on unseen real than unseen fake).
- Workshops: **DiffusionPrint** (AIMS; inpainting fingerprints into TruFor), **QuAD** (APAI;
  quality-aware calibration).
- None found does per-object origin with matched controls, measures frozen detectors on inserted
  vs real objects, or reports abstention / false accusations of real objects.

## 3. Gaps none of the found work covers (InlayFake's edge)

1. **Origin label for every object in the photo**, untouched real ones included, so the
   **real-object false-accusation rate** next to an insertion is measurable. Prior work labels the
   edited region (mask) or the one edited object.
2. **Real-object paste control (C3)** against AI insertion (C4) through one encoding pipeline:
   separates "spliced" evidence from "AI-generated" evidence. Not found anywhere. Fragments: TGIF
   (splice vs regenerate), INP-X (pixel exchange), B-Free (aligned training).
3. **Full matched ladder** C1 resave, C2 sham regeneration, C3 real paste, C4 insert, C5 insert with
   occlusion. No benchmark has C3 or an occlusion arm.
4. **Same-class, same-photo pairs**: a photographed dog and an AI dog in the same image (lighting,
   camera, class held fixed). Not found.
5. **Frozen COCO detectors on AI-inserted vs photographed objects** (H1 confidence vs origin, H2
   collateral flips beyond the resave noise floor). Not measured anywhere; COinCO uses a detector
   only as a filter.
6. **Calibrated abstention / risk–coverage for object-level forensics.** Only image-level
   rejection (Sci. Rep. 2025) and threshold calibration (AAAI 2026) exist.
7. **Hand-crafted + learned forensics in one per-object study under matched encoding**; recent
   benchmarks test mostly learned localisers and MLLMs.
8. **Origin-aware grounded captioning scored on false accusations** (Stage 2). Not found;
   IFDL-VLM (CVPR 2026 Findings) suggests VLMs are weak here, which makes a careful negative or
   positive result informative.

**No longer an edge after the 2026-10-17 scope change (SDXL only):** held-out *generator*
evaluation. OpenSDI, So-Fake and the AAAI 2026 calibration paper already do it at image level.
The held-out *source domain* (Open Images) and post-processing axes remain.

## 4. Risks for the paper

- COCO-based object inpainting itself is **not new** (COinCO, COCO-Inpaint, TGIF, CocoGlide,
  DiffSeg30k, FragFake, BR-Gen). The contribution must be framed as the combination in §3, with
  no "first" (AGENTS.md rule 9).
- COinCO is the reviewer's obvious comparison: state explicitly *insert vs replace*, *same-class
  pairing*, *independent verifier*, *matched controls*, *per-object false accusations*.
- A single generator invites "does this generalise?". Mitigations: report post-processing and
  Open Images results; consider evaluating TruFor/UnivFD-style detectors on a *small* public
  COinCO or COCO-Inpaint subset as an external check (if licences allow).
- All entries above are from search snippets; verify each before citing.
