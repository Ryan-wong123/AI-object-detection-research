# Before publishing: checklist

Things to settle before the repo, the shared previews or the paper go public (CVPR submission,
camera-ready, or making the GitHub repo visible to others). Tick each item and note the date and
how it was resolved. Keep this list current: add an item whenever something is deferred "until
before publishing".

## Licences and shared images

- [ ] **COCO photo licences for the shared previews.** `docs/viewer/task1_real_baseline/` (week 1)
  and `docs/preview/task2_insertion/` (week 2) contain downscaled COCO photos, plus edited
  versions of them. COCO images carry their own Flickr licences (the COCO annotations are a
  separate licence). Confirm the terms allow redistributing these copies, edited ones included,
  in a public repo. If not, keep the repo private, or remove those folders (and their history)
  before it goes public. Noted 2026-10-18, `docs/RESEARCH_LOG.md`.
- [ ] **Dataset licence statement in the paper** (paper Status box, "To verify"): COCO images and
  annotations, COCO-Stuff, Open Images (week 4), and the licence of the released benchmark.
- [ ] **Generator and model licences:** SDXL inpainting, OWLv2, SAM and the evaluated detectors
  allow the released edited images and their use in the paper.

## Anonymity (CVPR review is double-blind)

- [ ] The submission must not reveal the authors: no GitHub user name or repo URL in the paper or
  supplementary. Use an anonymised copy of the code/data if a link is needed.
- [ ] Briefings (`docs/briefings/`) name the author; they are internal and must not be linked
  from the submission.

## Git history

- [ ] **Assistant co-author lines in commit messages.** `CLAUDE.md` says no model name or
  assistant identifier may appear in commits. 24 commits on `week-1-coco-dataset` and
  `week-2-ai-insertion` (none on `main` yet) end with a `Co-Authored-By:` line naming the
  assistant (checked 2026-10-18 with
  `git log --all --format=%b | Select-String -Pattern "Co-Authored-By: Claude"`). Removing them
  means rewriting those branches' history and force-pushing. That is the author's decision, best
  done before the branches are merged into `main` or shared.
- [ ] No data, weights or `runs/` in git (`git ls-files data runs` must list nothing), and no file
  over GitHub's 100 MB limit.

## Paper

- [ ] Every `[[TBD: ...]]` filled from a run, or the claim removed.
- [ ] Every reference marked `[verify]` checked against the actual paper; CVPR 2027 dates checked.
- [ ] The word "first" appears nowhere (AGENTS.md rule 9).
- [ ] Status box deleted; within the 8-page limit.
- [ ] The human audit of the C4 edits is done, rejected edits are removed from the benchmark, and
  all counts (§4, Table 1) and shared previews are rebuilt after that removal.
