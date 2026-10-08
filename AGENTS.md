# AGENTS.md: instructions for AI coding agents

This file is the canonical guide for any AI agent (Claude Code, Codex, Cursor, etc.) working in this
repository. `CLAUDE.md` imports it. Read `PRD.md` (what and why) and `ARCHITECTURE.md` (how) before
changing code.

## Project in one paragraph

**InlayFake** is a research project targeting **CVPR 2027** (paper deadline 16 Nov 2026 AoE; confirm
on the official site). We insert AI-generated objects into real COCO photographs under matched control
conditions. We then measure (1) how off-the-shelf object detectors behave on inserted and on untouched
real objects, and (2) whether hand-crafted and learned image-forensic methods can tell, **per object**,
whether it was photographed or AI-inserted. The output is a benchmark, a baseline study and a paper
(`paper/PAPER.md`). The author works alone, on a 5-week schedule, on a single MSI laptop GPU.

## Golden rules (non-negotiable)

1. **Never fabricate results.** Every number in `paper/PAPER.md` must come from a run in `runs/` and be
   tagged with its run ID (see "Paper update protocol"). If a result does not exist yet, write
   `[[TBD: description]]`, never a plausible-looking number.
2. **Report negative findings.** If a hypothesis (H1–H4 in `PRD.md`) fails, the paper says so.
3. **Model inputs = final image only.** Source photos, edit masks, prompts and generator IDs are
   evaluation metadata. Never let them leak into a model's input in the main track.
4. **Matched controls.** Every comparison of AI-inserted against real goes through the same encoding
   pipeline (see `ARCHITECTURE.md` §Encoding policy). A format or resolution difference that the
   model can exploit is a bug, not a finding.
5. **Split by source image.** All edits of one COCO photo stay in the same split. Confidence intervals
   use a bootstrap over source images.
6. **Thresholds and calibration are fit on `val` only.** Never tune on `test`.
7. **Reproducibility.** Every run records the git commit, config, seeds, model checkpoint IDs and
   package versions (handled by `inlayfake.runlog`). Do not hard-code paths; use the config.
8. **No data or weights in git.** `data/`, `runs/` and model weights are git-ignored. Small result
   tables (CSV/JSON < 1 MB) that the paper cites are copied into `paper/results/`.
9. **Do not overclaim novelty.** Never write "first" in the paper (see PDF §2 and `PRD.md` §Novelty).
10. **Ethics.** No identifiable people are used as insertion targets. Do not generate humans by
    default. Treat false "AI" accusations against real objects as a first-class harm metric.

## Repository map

```
AGENTS.md, CLAUDE.md     agent instructions
PRD.md                   requirements, hypotheses, scope, timeline
ARCHITECTURE.md          pipeline, data contracts, module design
paper/PAPER.md           the CVPR paper draft (kept up to date by agents)
paper/results/           small result tables the paper cites (tracked)
paper/figures/           figures the paper uses (tracked)
docs/RESEARCH_LOG.md     dated log of every experiment, decision and finding
configs/                 YAML configs (default.yaml, pilot.yaml, ...)
src/inlayfake/           Python package (see ARCHITECTURE.md for module list)
tests/                   pytest tests; must run on CPU without model weights
data/                    (ignored) COCO, generated images, ledgers
runs/                    (ignored) predictions, scores, metrics per run ID
```

## Environment

- Python 3.11 (3.10–3.12 OK). PyTorch with CUDA on the author's MSI laptop (Windows 11; WSL2 also
  works).
- Install: `pip install -e ".[dev]"` for tests only; `pip install -e ".[all]"` on the GPU machine.
- **Assume about 8 GB of VRAM** unless `configs/` says otherwise: use fp16, `enable_model_cpu_offload()`,
  batch size 1 for generators, and a single model on the GPU at a time.
- Cloud agent sessions have **no GPU**. Agents there write code and tests and run them on CPU with tiny
  fixtures. GPU stages are run by the author, who then reports the outputs back.
- Use `pathlib`, not shell-specific scripts; code must run on Windows and Linux.

## Commands

```bash
pytest -q                                   # unit tests (CPU, no weights)
ruff check src tests && ruff format src tests
python -m inlayfake --help                  # CLI entry point (stages listed in ARCHITECTURE.md)
python -m inlayfake <stage> --config configs/pilot.yaml
```

## Coding conventions

- Type hints everywhere; dataclasses for records; no global state.
- Each pipeline stage is idempotent: it reads inputs from the config and writes outputs under
  `data/` or `runs/<run_id>/`, skipping items already done (so overnight jobs can resume).
- Heavy model imports (`diffusers`, `transformers`, `ultralytics`) happen **inside** functions so
  that tests and the CLI help work without them installed.
- Every random choice takes an explicit `numpy.random.Generator` seeded from the config.
- Prefer small, pure functions that can be tested (placement, compositing, metrics) over large
  scripts.
- New metric → unit test with a hand-computed example.

## Experiment workflow

1. Add or adjust a config in `configs/`.
2. Run the stage. It creates `runs/<YYYY-MM-DD>_<short-name>/` with `run.json` (provenance) and
   outputs.
3. Run `python -m inlayfake evaluate ...`. It writes `metrics.json` and `tables/*.csv`.
4. Add an entry to `docs/RESEARCH_LOG.md`: date, run ID, what was run, key numbers, interpretation,
   surprises, next step.
5. Update `paper/PAPER.md` (see below).
6. Commit code, configs, log, paper and small tables together.

## Paper update protocol

`paper/PAPER.md` is the living paper draft. Update it **every time** something is done or a finding
appears:

- Put numbers in tables or text followed by an HTML comment giving the source, e.g.
  `AUROC 0.71 <!-- run:2026-10-24_forensics_test, file:paper/results/origin_auroc.csv -->`.
- Replace `[[TBD: ...]]` placeholders as results arrive; do not delete placeholders without filling
  them.
- Keep the **Status box** at the top of `PAPER.md` current (what is drafted, what is waiting on
  results).
- If a finding contradicts the current narrative, rewrite the narrative. Do not bend the result.
- Citations: only cite papers in the reference list; mark any unverified entry `[verify]`.
- CVPR limit: 8 pages excluding references. Track an approximate word budget in the Status box.

## Git

- One branch per week of the plan (`week-1-coco-dataset`, then e.g. `week-2-ai-insertion`).
  Commit small, descriptive messages.
- Never commit `data/`, `runs/`, weights, or COCO images.
