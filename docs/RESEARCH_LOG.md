# Research log

Dated record of every experiment, decision and finding. Newest entries at the top.
Format: date · run ID (if any) · what · key numbers · interpretation · next step.

---

## 2026-10-08: Generation pipeline implemented (week 1)

- **Hardware recorded:** NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB (from the author's `nvidia-smi`).
  This matches the 8 GB plan, so no config change was needed beyond recording it.
- **Implemented:** stages `sample`, `plan`, `generate`, `preview`; COCO index; encoding policy;
  local inpainting + feathered compositing; real-object paste with donor selection; diffusers
  wrappers for SDXL / SD2 / Kandinsky / FLUX Fill (lazy, 8 GB defaults); run provenance (`runlog`).
- **Design decisions:**
  - Rectangular dilated mask with a linear feather that is exactly 0 at the mask border. Why: it
    guarantees that pixels outside the intended mask are unchanged, so any detector or forensic
    difference outside the edit is a JPEG-block effect we can measure, not generator leakage.
  - G3 (held-out FLUX) only on test-split sources (~120). Why: a held-out generator must never be
    seen in train/val. The earlier "300 sources" figure was inconsistent with a 20% test split.
  - The insertion size is sampled from real COCO instances of the category, excluding those under 1%
    of the image area. Why: tiny inserts rarely produce a recognisable object.
- **Verified on CPU** (fake inpainter, synthetic COCO): 18 tests pass. They check that the policy
  JPEGs are q95 4:4:4, that C4 decodes identically to C1 outside the mask (±1 JPEG block), that every
  condition is produced, and that a second run is a no-op (resumable).
- **No experimental results yet.**
- **Next:** the author runs the 50-source pilot with SDXL on the laptop and checks `preview.html`.
  Then the `verify` stage (OWLv2 + SAM) and `audit`.

---

## 2026-10-08: Added held-out source domain (decision)

- **What:** The author asked for a small second-dataset test in week 4.
- **Decision:** About 100 **Open Images V7** validation photos, test-only, conditions C1–C4, generators
  G1 (seen) and G3 (held out). Every model and threshold is fit on COCO and transferred unchanged.
- **Why Open Images, not Objects365:** Grounding DINO is pretrained on Objects365, so Objects365 would
  not be unseen for one of our detectors. LVIS reuses COCO images.
- **Caveat recorded:** Open Images boxes are exhaustive only for image-level-verified classes. We score
  verified classes only; the rest become ignore regions.
- **Changed:** PRD (scope, risks, week 4), ARCHITECTURE (adapter stage, ledger `source_dataset`), config
  `ood:` block, ledger code + tests, PAPER §4/§7.4/§8.

---

## 2026-10-08: Project setup (week 1, day 1)

- **What:** Read the professor's InlayFake brief. Scope agreed with the author: both detector study and
  forensic origin detection ("traditional" = off-the-shelf detectors **and** hand-crafted forensics,
  plus learned forensic baselines). Target: CVPR 2027 (submission 16 Nov 2026 AoE per official CFP
  via search; confirm, including the registration date).
- **Decisions:**
  - 5-week plan: benchmark + baseline study (the brief's suggested "constrained first paper"). Stage 2
    (VLM captions) reduced to P1.
  - Source pool = COCO val2017 (unseen by COCO-trained detectors).
  - Generators: SDXL-Inpainting, SD-2-Inpainting (seen); FLUX.1-Fill-dev (held out).
  - Verifier for insertions = OWLv2, excluded from the evaluated detectors to avoid selection bias.
  - Identical JPEG q=95 encoding for C1–C5 to remove format shortcuts.
- **Created:** AGENTS.md, CLAUDE.md, PRD.md, ARCHITECTURE.md, paper/PAPER.md, config and ledger
  schema.
- **Open:** laptop GPU model and VRAM; professor sign-off on the scope.
- **Next:** COCO download + `sample` and `plan` stages; generator wrapper for G1; 50-source pilot.
