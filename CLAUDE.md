# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- Read `PRD.md` and `ARCHITECTURE.md` before starting any task. Check the timeline in `PRD.md`
  §Timeline for what the current week should deliver.
- After **any** completed task or finding: update `docs/RESEARCH_LOG.md` **and** `paper/PAPER.md`
  (including its Status box), then commit. This is a standing instruction from the author.
- Cloud sessions have no GPU (4 CPU, 15 GB RAM). Do not try to download diffusion or VLM weights
  there. Write the code, test it on CPU with synthetic fixtures, and give the author the exact command
  to run on their laptop.
- When the author pastes results (metrics JSON, logs, screenshots), treat them as the source of
  truth: record them in `paper/results/`, the log and the paper, citing the run ID they give.
- The author is a student new to research code. Explain *why* in commit messages and the log, and
  give copy-paste commands for Windows (PowerShell) when you hand over a GPU step.
- Never put a model name or identifier of the assistant in commits, code or the paper.
