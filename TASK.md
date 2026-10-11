# TASK: finish week 1 on the RTX 4070 laptop

Handover note for the next work session, written 2026-10-11. **Delete this file when the task is done.**

- **For the author:** follow "Steps for the author" below.
- **For an AI agent in a new session:** you have no earlier context. Read `CLAUDE.md` / `AGENTS.md`,
  then this whole file. Wait until the author pastes `summary.md`, then do "Steps for the agent".

---

## Where things stand

- Branch `week-1-coco-dataset`. Task 1 (real-only COCO baseline) has been run on this laptop as run
  `task1_real_baseline`. Outputs are in `runs\task1_real_baseline\` and `data\inlayfake\`. Both are
  git-ignored and exist only on this laptop.
- Paper §7.1 (Tables 2a and 2b) is filled from that run, but the AP confidence intervals use only
  **20** bootstrap resamples. The text says `[[TBD: 200 resamples for the final version]]`.
- The label audit is done. The author re-checked all 107 decisions (77 add / 26 ignore / 4 reject)
  and confirmed every one, so `data\inlayfake\audit_decisions.csv` stays as it is. It matches
  `paper\results\task1_audit_decisions.csv`.
- `configs\real_baseline.yaml` now sets `eval.ap_bootstrap: 200`, so no config edit is needed.
- **Only remaining week-1 step:** re-run `evaluate` with 200 resamples and put the final intervals
  in the paper.

**Laptop roles (author's decision):** this RTX 4070 laptop produces every result: all generation and
every run that feeds the paper. The other laptop is only used for writing code, CPU tests and paper
text. Do not mix results from the two machines.

---

## Steps for the author (PowerShell, in the repo folder)

1. Get the latest code, config and this file:

   ```powershell
   git pull
   ```

2. Activate your Python environment, the same one you used for Task 1. For example:

   ```powershell
   .venv\Scripts\activate
   ```

3. Run the final evaluate. It reuses the saved detections and forensic scores and runs on the CPU.
   Expect a few hours: it computes 4 detectors × 2 conditions × 201 COCO evaluations. You can leave it
   running.

   ```powershell
   python -m inlayfake evaluate --config configs/real_baseline.yaml
   ```

4. Start Claude Code in this repo and say something like:
   *"Read TASK.md. Here is my summary.md:"*, then paste the full contents of
   `runs\task1_real_baseline\summary.md`.

---

## Steps for the agent (after the author pastes `summary.md`)

1. Save the pasted text as `paper/results/task1_real_baseline_summary_v4_200boot.md`. Treat the
   pasted text as the source of truth.
2. Check that the point estimates (AP, AP50, AR100, recall, lost/gained, forensic means and medians)
   equal those in `paper/results/task1_real_baseline_summary_v3_audited.md`. Only the confidence
   intervals should differ. If any point estimate differs, stop and tell the author before editing
   the paper, because it would mean the inputs changed.
3. Update `paper/PAPER.md` §7.1:
   - **Table 2a:** replace the AP C0/C1 intervals with the 200-resample ones. Change the source
     comment to
     `<!-- run:task1_real_baseline, file:paper/results/task1_real_baseline_summary_v4_200boot.md -->`.
   - Replace `(20 resamples) [[TBD: 200 resamples for the final version]].` with `(200 resamples).`
   - **Table 2b:** update the resave-shift intervals (C1 − C0) from the new file and point its source
     comment at the v4 file. They can change slightly, because the random generator is shared with
     the AP bootstrap.
   - Check the Findings text (i)–(iv) still agrees with the numbers.
4. **Status box** in `PAPER.md`: set *Last updated*. Change *Experiments done* to say Task 1 is
   complete with final 200-resample CIs, and remove "Pending: 200-resample AP CIs".
5. `PRD.md` *Status* row: week 1 complete; next is Task 2 (AI insertion).
6. `docs/RESEARCH_LOG.md`: add a dated entry at the top with the run ID, what was run, the key
   intervals, and a note that the point estimates were unchanged.
7. Delete this `TASK.md`. Ask the author before committing. Then suggest merging
   `week-1-coco-dataset` into `main` and creating `week-2-ai-insertion` (see `AGENTS.md` §Git).

---

## After this: week 2 (Task 2, AI insertion)

See `PRD.md` §8 Timeline, week 2. Rough order:

1. **SDXL pilot on 50 sources** (`configs/pilot.yaml`; stages `sample` → `plan` → `generate` →
   `preview`). The author checks `preview.html` by eye.
2. **`verify` stage:** not implemented yet (`python -m inlayfake --help` marks it `[not yet]`). It
   checks each insertion with OWLv2 and makes SAM masks (`ARCHITECTURE.md`). The code can be written
   and tested on either laptop. The runs that produce results happen on this one.
3. **Full generation run overnight:** G1 and G2 on all sources, and G3 (FLUX, held out) on the test
   split only.
4. **Human audit of 10% of edits** with a contact sheet.
5. **Start the Open Images download** for week 4.
