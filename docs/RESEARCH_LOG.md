# Research log

Dated record of every experiment, decision and finding. Newest entries at the top.
Format: date · run ID (if any) · what · key numbers · interpretation · next step.

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
