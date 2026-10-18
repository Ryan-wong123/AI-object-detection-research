# Research log

Dated record of every experiment, decision and finding. Newest entries at the top.
Format: date · run ID (if any) · what · key numbers · interpretation · next step.

---

## 2026-10-18: Re-run done: 542 of 544 photos have an AI edit (run `task2_insertion` finished)

- **Result** (run_verify_2026-10-18T031732Z, 11:17 local; 18 rounds of 3–7 images): 5 of the 7
  dropped photos passed with boxes 7–12: toilet (box 7), tv (box 8), chair (box 9), bed in 65074
  (box 10), surfboard (box 12); their C2 and C3 passed too. Still without an AI edit: 10995
  (bed) and 151857 (train), both close-ups whose only class is the object filling the frame.
- **Final benchmark:** AI edit for 542 of 544 photos (C4 464, C5 38, anywhere 40); 538 have every
  condition C0–C4; 4 lack only C2 (sham attempts kept "producing" the target, 108440, 203580,
  259571, 262938); 1,080 edited images in the ledger (`paper/results/task2_insertion_final.json`).
- **Next:** human audit of 10% (`audit_edits.csv`), Table 1 counts, then week 3.

---

## 2026-10-18: Merged run done (537 of 544); re-run of the 7 dropped photos (restart 6)

- **Merged run result** (run_verify_2026-10-17T222301Z, 06:23 local; 10 rounds after restart 5):
  AI edit for 537 of 544 photos: from C4 464, from C5 38, newly placed anywhere 35; 7 dropped
  after 6 boxes × 3 seeds. 533 photos have every condition C0–C4; 4 more lack only C2: in all 3 sham
  attempts OWLv2 found the target class at the box (108440 at an adopted C5 box, 203580, 259571,
  262938 at anywhere boxes; probably the real object partly inside the box, not yet checked)
  (`paper/results/task2_insertion_merged_round_final.json`). Anywhere boxes passed 35 of the 42
  photos; per round the pass rate fell from 7/42 to 0–4 per round.
- **Why the 7 were dropped** (OWLv2 re-run, read-only, on all 210 failed C4/C5/anywhere attempts
  of the 7): 59 found only the photo's real object of that class (never accepted), 93 a confident
  detection outside the box (mostly the real object as well), 32 only weak detections
  (score < 0.2, best 0.16–0.19), 26 nothing. Two groups: close-ups of one large object (bed,
  train, toilet, beach umbrella), where the box lands on the real object or dark background; and
  cluttered photos with small classes (remote, book, knife), where SDXL's object is not
  recognised.
- **Decision (author):** re-run the 7, still anywhere in the image. `placement.merged.versions`
  6 → 12; from box 7 (`avoid_same_class_from: 7`) a box may cover at most 30% of its area with
  a real object of the same class (shrinks ×0.8 if no spot is found), because a box on the real
  object cannot pass. Boxes 1–6 and all other photos are unchanged (scratch dry run: only the 7
  get jobs). 1 new test; 80 pass. Restart 6 at 10:28, PID 3304, 7 images per round.
- **Next:** record the result of boxes 7–12; final Table 1 counts.

---

## 2026-10-18: Scope change: C4 and C5 merged into one AI condition (restart 5)

- **Why:** after the round-7 check, 44 of 544 photos still had no verified AI object under the
  separate C4 (free space) and C5 (covers 20–50% of one object) rules; 284 had both, 175 only C4,
  41 only C5 (`paper/results/task2_insertion_merge_decision.json`, read-only recomputation from
  the verify state before the merge).
- **Decision (author):** one AI condition, kept under the code C4: "an AI object of a class
  already in the photo, placed anywhere"; that class rule is the only placement rule. Verified C4
  and C5 edits are kept. Photos with both use the C4 (author's choice over a random pick: their
  C2/C3 already exist, no extra GPU time). C2 and C3 sit at the chosen AI box.
- **Code:** `placement.merged` (on in `configs/insertion_inphoto.yaml`). `replan.merged_version`:
  verified C4 → kept; else verified C5 adopted as free version 100 (box/class copied,
  `placement_rule: adopted_c5`); else versions 101–106 place a box anywhere (photo's classes,
  free-track class first; size alternates own instance and typical COCO size), 3 seeds each,
  OWLv2 still has to confirm the object. C5 is no longer generated; C2/C3 at a merged box wait
  for its verified AI edit. The final ledger has one AI record per photo, relabelled C4, with
  `EditInfo.merged_source` (C4 / C5 / anywhere). Preview has a merged view (dropdown: AI source,
  retrying/dropped, C2/C3 missing, with counts). 3 new tests (hand example, end-to-end switch
  from separate to merged, preview); 79 tests pass. Dry run on a scratch copy: 459 C4 kept,
  41 C5 adopted, 44 anywhere boxes, 86 jobs.
- **Restart 5:** run stopped in round 8 (75 of 242 images made). Those attempts were verified
  first with the merged config (run_verify_2026-10-17T194502Z, 114 rows): AI edit from C4 464,
  from C5 38, none yet 42 (`merge_decision.json`). Merged `insert` started 03:46, PID 31272,
  round 1: 81 jobs (42 AI anywhere, 38 C2 at adopted C5 boxes, 1 C2 retry). Backups:
  `replans_before_merge.jsonl`, `verify_before_merge.jsonl`, `ledger_before_merge.jsonl`,
  `insert_progress_part5.log`, `insert_part5.log`.
- **Consequences (paper §4, PRD):** H2 loses its controlled occlusion condition; H2 is evaluated
  on originals the AI object does not cover, covered objects separately by visible fraction
  (observational). 298 verified C5 edits (336 photos have one, 38 adopted) leave the benchmark because their photos use the C4. C2 at an
  adopted C5 box repaints part of the covered real object, which verify marks mixed/modified.
- **Next:** let the merged run finish (anywhere boxes up to 6 × 3 seeds, then C2/C3), check
  success, fill §4 counts and Table 1.

---

## 2026-10-17: Related-work scan (CVPR 2026 and 2022–2026) and positioning

- **What:** three web searches (CVPR 2026; local AI-edit datasets and methods; detectors on
  inserted objects, MLLM forensics, calibration, matched comparisons, provenance). Summary and gap
  analysis in `docs/literature/related_work_2026-10-17.md`. OpenReview, CVF and arXiv pages could
  not be opened from the cloud sandbox: venues and facts come from search snippets and are
  `[verify]` until the PDFs are read.
- **Closest prior work:** COinCO (CVPR 2026: COCO, one object *replaced* by SD inpainting, a
  detector used only as a regeneration filter, in/out-of-context labels). Also TGIF (spliced vs
  regenerated), INP-X (detectors rely on global VAE artefacts), COCO-Inpaint, FragFake, OpenSDI.
- **Gaps not covered by anything found:** origin label for every object and a real-object
  false-accusation rate; a real-object paste control (C3) and occlusion arm (C5); same-class
  same-photo pairs; frozen COCO detectors on inserted vs photographed objects; calibrated
  abstention at object level; hand-crafted + learned forensics under matched encoding;
  origin-aware captions scored on false accusations.
- **No longer an edge** after the SDXL-only scope change: held-out generator (done at image level
  by OpenSDI and others).
- **Paper:** §2 gains a paragraph on AI object edits in COCO (COinCO, TGIF, COCO-Inpaint, INP-X);
  4 references added, all `[verify]`.

---

## 2026-10-18: Round-7 check; verified edits are now always kept (restart 4)

- **Round-7 check** (run_verify_2026-10-17T190409Z.json; 495 images made across restarts 1–3):
  new C4 (fill) passed 35 of 57 on the first try, new C5 (fill) 34 of 128; older C4 on a new
  box 13 passed / 48 failed; older C5 on a new box 15 / 116. Totals C4 446, C5 325, C2 515 clean;
  281 edits into round 8 (210 C5, 70 C4, 1 C2).
- **Found:** 13 C4 edits that had passed with a class not in the photo stopped counting. The
  rule "outside classes only if the photo has no in-photo AI object" was re-evaluated at every
  step, and the C5 fill had given all 13 photos an in-photo C5 (3 passed, 10 retrying).
- **Decision (author):** keep them. `replan.track_version` now returns the version of the first
  passing attempt before applying the outside-class restriction, so the rule decides what to
  try, never withdraws a verified edit. Test updated; re-plan and verify tests pass (20).
- **Read-only check:** under the new rule C4 459 (+13), C2 528 (+13), C5 unchanged, no edit
  removed; their C3 pastes follow the same box. The images were never deleted, so nothing is
  regenerated.
- **Restart:** stopped Sun 03:16 (round 8 at 39/281, kept), restarted 03:17 (process 25208);
  logs `insert_progress_part4.log` / `insert_part4.log`.

---

## 2026-10-18: Re-plans also try typical COCO sizes; 7 dropped C4 edits back (restart 3)

- **Finding:** 7 C4 edits (photos 137294 oven, 140929 clock, 458054 toilet, 491130 snowboard,
  569700 vase, 230008 motorcycle, 529105 horse) were dropped: their original box failed 3 seeds,
  and no new box fit at steps 1–6. Re-plans sized boxes only like the photo's own instance
  (×0.8–1.2); in these crowded photos those instances are large. Outside classes were not allowed
  (5 photos already have a C5; 2 have C5 still retrying).
- **Read-only test:** with a typical COCO size (the size the original planner used), 6 of 7 get
  a new box under unchanged C4 rules; the own-instance size gives none.
- **Fix (author approved):** re-plans now alternate the own-instance size and a typical COCO size
  (6 sizes instead of 5 own-instance sizes); C4/C5 rules unchanged. Re-plan and plan tests pass.
- **Scratch check:** with the 7 edits' empty free-track rows removed (42 rows), recomputing adds
  9 rows, all for these 7: every one now gets a new box (toilet on step 3, on the ground).
- **Restart:** stopped Sun 02:21 (that round at 174/271, saved, not checked yet), removed the 42
  rows from `replans.jsonl` (backup `replans_before_size_fix.jsonl`), restarted 02:22 (process
  31232); logs of the previous part kept as `insert_progress_part3.log` / `insert_part3.log`.
  All images made since the last check are checked together when this round ends.

---

## 2026-10-18: C4 fill: 57 more photos get a C4 box (run `task2_insertion` restarted again)

- **Question (author):** 70 photos had no C4 box; relax the margin to 0 px and drop the surface
  preference? Answered: the surface preference never blocks a box (it falls back to free space),
  and 0 px would let the edit (box + 6 px dilation, 3 px feather) repaint real objects.
- **Read-only test on the 70 photos** (every class in the photo, as the planner does): current
  rules with 1 size per class 32 fit; **current rules with 6 sizes 53**; margin 6 px 33; margin
  6 px + 6 sizes 54; margin 0 px + 6 sizes 54. So again the search, not the rule: the planner
  draws one random size per class. Labelled boxes cover a median 61% of these frames (16 over 75%).
- **Decision (author):** C4 fill with several sizes, 8 px margin kept. Implemented as
  `placement.free_fill` (off by default; on in `configs/insertion_inphoto.yaml`): plans without a
  C4 box get one, the plan class first, then the photo's other classes; 6 sizes per class; C4
  rules unchanged; own seeded generator per photo. A different class is stored as
  `free_category` and used by C2/C3/C4 (and their re-plans), while C5 keeps the plan class.
  2 new tests; plan/config/re-plan/verify tests pass (31).
- **Check before applying:** re-planned in a scratch copy and compared with the live plan
  (which holds the C5 fill): 487 plans byte-identical, 57 gained a C4 box (5 with another class),
  nothing else changed in them, C5 fill reproduced (128). 13 photos stay C5-only.
- **Restart:** stopped at Sun 01:27 (that round at 145/302) and restarted at 01:28 (process
  13064). Plans now: 531 of 544 with a C4 box, 535 with a C5 box
  (run_plan_2026-10-17T172754Z.json, paper/results/task2_insertion_plan_fills.json). Kept:
  `plans_before_free_fill.jsonl`, `insert_progress_part2.log`, `insert_part2.log`.
- **Expected cost (estimate):** 57 new C4 jobs, each with its C2 (and C3, no GPU): roughly
  1.5–2 h more.

---

## 2026-10-18: C5 fill: 128 more photos get a C5 box (run `task2_insertion` restarted)

- **Question (author):** why do 137 of the 544 photos have no C5 box? A read-only test re-ran
  the C5 box search on them with one rule relaxed at a time: with the same rules but three box
  sizes, 84 fit; box sized from the covered object 103; other objects covered up to 20% (not 5%)
  95; no limit 96; cover 10–50% of the target 102; covered objects down to 32 px (not 64) 114.
  So the main cause was the search (the plan tries one C5 box size when the photo also has a C4
  box), not the 5% rule, which I had wrongly named as the cause before checking.
- **Decision (author):** allow covered objects down to 32 px. Implemented as
  `placement.occlusion_fill` (off by default; on in `configs/insertion_inphoto.yaml`): after the
  main search, plans without a C5 box get one, trying 6 sizes (typical COCO size, then the photo's
  own instance size) and finally a box sized from the covered object; covered object ≥ 32 px; the
  other C5 rules unchanged (20–50% of one object, ≤ 5% of any other). Own seeded generator per
  photo. C5 re-plans now use the same 32 px minimum. The main search keeps the original 64 px
  rule so that re-running `plan` reproduces earlier plans. New test; plan/config/re-plan tests
  pass.
- **Check before applying:** re-planned in a scratch copy and compared with the live plan: 416
  plans byte-identical, 128 gained a C5 box, 0 other changes, every C2/C3/C4 box and class
  identical; 9 photos still without C5.
- **Restart:** the run (process 14000, round 7 at 65/239) was stopped on Sun 00:39 and restarted
  at 00:40 (process 13792) with the new plan (535 of 544 with a C5 box). Kept:
  `plans_before_occlusion_fill.jsonl`, `insert_progress_part1.log`, `insert_part1.log`. All
  finished images and checks are kept; the new C5 jobs start at their first seed.
- **Expected cost (estimate):** 128 new C5 jobs at C5's retry rates so far, roughly 2–4 h more.

---

## 2026-10-17: Decision: keep C2; condition rationale written into the paper

- **Question (author):** is C2 useful, since its box holds no object and is AI-repainted like
  C4? Could C3/C4/C5 alone test real vs AI objects?
- **Answer recorded:** the core real-vs-AI object test needs only objects (C1/C3 real, C4/C5 AI),
  but C2 is the placebo-like control: same generator, box and pipeline, no object. It shows
  whether a forensic method detects AI *objects* or any *regenerated region* (H3), and whether
  repainting alone disturbs nearby real objects (H2). It is never scored as an AI object.
- **Decision (author):** keep C2. Remaining cost about 1 h of the run (re-done shams at moved
  C4 boxes).
- **Paper:** §4 now explains why each condition exists, why C2/C3/C4 share one box (and move
  together on re-plan) while C5 has its own (no box can avoid all objects and cover 20–50% of one),
  and why C5 matters (realistic overlap, hardest case, occlusion control for H2). §6 states that
  origin metrics are over objects only and that C2 gives a separate edited-background flag rate;
  §7.6 lists the planned control analyses. About 450 words added; may need shortening later.
- **Seen in the preview:** some C2 shams are flat grey patches (e.g. source 42296), an obvious
  editing trace; to be marked in the human audit (realistic = n) and discussed if frequent.

---

## 2026-10-17: Decision: scope reduced to one generator (SDXL)

- **Decision (author):** not enough time for the planned second seen generator (G2 SD-2-Inpainting)
  and held-out generator (G3 FLUX.1-Fill, fallback Kandinsky 2.2); the benchmark uses SDXL only.
- **Consequences, stated in PRD §2/§3/§4/§7 and the paper:** H4 keeps its post-processing and
  held-out-source-domain (Open Images) axes but loses the held-out-generator axis (old Table 5
  grid removed); the crop classifier L3 is trained on SDXL crops only; Open Images (week 4) uses
  SDXL; the paper claims "one inpainting model (SDXL)" and names other generators as future work
  in §8. PRD success criterion "a finding for H4" is kept with the narrowed H4.
- **Config:** `generation.insert_generators: [G1]`, `ood.generators: [G1]`; the G2/G3 entries and
  code stay for future work. The running `task2_insertion` process was started with
  `generators_enabled: [G1]` and is unaffected (its config was resolved at start). 73 tests pass.
- **Also fixed in the paper:** §4 said the generator sees a 2.5× window; since the pilots it is
  1.2× (verification still uses 2.5×).
- **Round-1 result of the full run** (run_verify_2026-10-17T012532Z.json, interim): C2 472/474
  clean, C4 269/474, C5 123/407 passed on attempt 1; 491 edits went into round 2. Round 1: 1,355
  images in 6 h 53 min.

---

## 2026-10-17: Decision: the full run uses only the 544 week-1 baseline photos

- **Decision (author):** week 1 sampled 600 photos but its baseline (detectors, forensics,
  label audit on C0/C1) used the 544 that had a plan; the full insertion run must use the same
  544, so weeks 1–3 share one photo set.
- **Implemented:** `sampling.only_sources` (a jsonl with `source_id` rows) restricts the plan
  stage; `configs/insertion_inphoto.yaml` sets it to `data/inlayfake/plans.jsonl` (the week-1
  plans). Test added; `pytest -q`: 73 passed.
- **Re-plan** (run_plan_2026-10-16T180641Z.json, paper/results/task2_insertion_plan.json;
  nothing had been generated yet): 544 of 544 planned, all inside the week-1 set; 474 with a
  free box, 407 with an occlusion box, 70 C5-only, 2 with a class not in the photo, 1 anywhere.
  First round: 474 C2 + 474 C4 + 407 C5 = 1,355 images (~7.5 h); total at the pilot's rates
  ~5,100 images, **~28 h** (estimate).
- **Resolved:** the caveat about 56 unaudited close-ups no longer applies (they are out).

---

## 2026-10-17: Full insertion run set up (run `task2_insertion`, plan only so far)

- **Config** `configs/insertion_inphoto.yaml`: the `pilot_inphoto` settings (test: placement,
  edit, verify, generators and encoding identical) on all 600 sampled sources, work dir
  `data/inlayfake_v2` (week-1 `data/inlayfake` untouched), SDXL (G1) only. SD2 and FLUX /
  Kandinsky are later runs in the same work dir once tested with these settings. 72 tests pass.
  The old `configs/insertion.yaml` (week-1 plans, fixed classes) is kept but superseded.
- **sample:** 600 sources, identical to week 1 (362 train / 114 val / 124 test).
- **plan** (run_plan_2026-10-16T171841Z.json, paper/results/task2_insertion_plan.json): 600 of
  600 planned; 474 with a free box (C2/C3/C4), 465 with an occlusion box, 126 C5-only, 4 with a
  class not in the photo, 7 placed anywhere. First-attempt jobs: 474 C2 + 474 C4 + 465 C5 =
  1,413 SDXL images.
- **Time estimate (from the same-class pilot, not measured on this run):** the pilot generated
  469 images for 125 first-attempt jobs (3.75×; C2 1.6, C4 3.6, C5 6.0 — includes some images
  wasted by the rule changes mid-pilot), at ~19–20 s per image in large rounds. Applied here:
  ~5,200 images, **~28–30 h** of GPU time (±20%); round 1 alone ~8 h. Not a single night;
  the run is resumable.
- **Caveat:** 56 of the 600 photos were not planned in week 1 (the close-ups), so their labels
  were not in the week-1 audit (77 add / 26 ignore / 4 reject covered the 544). They need the
  label audit before week-3 evaluation.
- **Next:** author starts `insert` on the laptop; then preview, audit, and log the final counts.

---

## 2026-10-17: Decision: a box anywhere as the very last resort: 50 of 50 photos

- **Decision (author):** relax the rule so every photo always has a valid box: the AI object
  may be placed anywhere in the image.
- **Implemented as the last fallback only** (`placement.allow_anywhere`, false by default, true
  in `pilot_inphoto.yaml`): when no box satisfies the rules after every other fallback, a C5 box
  is placed anywhere inside the image with no overlap rule (`placement.place_anywhere`). Class:
  the photo's smallest insertable class, else an outside class. Used by the plan stage (own
  generator; earlier plans byte-identical: week 1 544/544, pilot 1 46/46, same-class 49/49) and
  by C5 re-plans whose box cannot be placed. C4 stays rule-abiding (no occlusion), because H2
  relies on it. New ledger field `EditInfo.placement_rule` = "anywhere" so week-3 analyses can
  include or exclude these. `verify` already recomputes the visible part of every covered real
  object. 2 new tests; `pytest -q`: 71 passed.
- **Result** (run_plan_2026-10-16T170747Z.json, run_verify_2026-10-16T171152Z.json,
  paper/results/task2_pilot_inphoto_anywhere_verify.json): all 50 photos planned; 421455 (car
  interior) got an anywhere C5 truck that passed on box 2 (OWLv2 0.34). C4 39/41, C5 39/43,
  C2 39/41; final ledger 256, audit sample 12. **50 of 50 photos have a verified AI object.**
- **Doubtful:** by eye the 421455 "truck" is a 21×13-px dark blob in the window reflection, sized
  like the photo's own tiny truck; it passes the verifier but is not a convincing object and will
  probably fail the human audit. The other last-resort edit (151857 clock) is clear but
  implausible.

---

## 2026-10-17: Decision: a class not in the photo as the very last resort: 49 of 50 photos

- **Decision (author):** for photos whose own classes cannot be inserted, allow a class that is
  not in the photo.
- **Implemented:** `edit.max_outside_classes` (0 by default, 3 in `pilot_inphoto.yaml`): after the
  in-photo versions (new boxes, class switches), up to 3 classes from `placement.outside_classes`
  (default: the 10 original categories) that are not in the photo, seeded order per source. The
  plan stage uses the same fallback for photos with no insertable class (own generators: earlier
  plans byte-identical, checked for week 1, pilot 1 and the same-class pilot). New ledger field
  `EditInfo.class_in_photo` (False for these edits; C2/C3 follow C4), so the same-class analysis
  can exclude them.
- **Own mistake, found and fixed before recording:** the first version let *any* failed edit
  switch to an outside class. That added out-of-photo objects to 6 photos that already had an
  in-photo AI object (bottle in the elephant photo, chair in an airport window, clock on a bed,
  bench on a ski slope, bottle in a dark room, vase above a bus), beyond the author's request.
  Now an edit may use an outside class only if the photo has no in-photo pass in C4 or C5 and the
  other edit is exhausted too; a pass at an outside version stops counting otherwise. Test added;
  `pytest -q`: 69 passed.
- **Result** (run_verify_2026-10-16T165640Z.json,
  paper/results/task2_pilot_inphoto_outside_verify.json): C4 39/41, C5 38/42, C2 39/41; final
  ledger 253 records, audit sample 12. **49 of 50 photos have a verified AI object.** Only one
  edit uses a class not in the photo: 151857, a **clock** (OWLv2 0.64) painted over the train's
  windscreen — clearly there and tonally matched, but implausible; flagged for the audit.
- **Still without one (1):** 421455 (car interior): the "car" box covers 88% of the frame, so no
  box satisfies the C5 rule (≤ 5% of any other object); the class is not the problem.

---

## 2026-10-17: C5 boxes sized from the covered object: 48 of 50 photos have an AI object

- **Question (author):** how to give the 4 remaining photos a box. Diagnosis per photo (COCO
  boxes): 151857 one train covering 67% of the frame; 562443 one zebra box 61%; 457848 three
  overlapping motorcycles (61%, 52%, 22%) and 5 people, boxes cover 83%; 421455 a car-interior
  shot whose "car" box covers 88% with dog, phone and person inside. None has free space for C4.
- **Bug found:** C5 boxes were sized from the class's usual COCO size (often ~2% of an image),
  which can never cover 20% of an object filling most of the frame. Fix:
  `placement.occluding_sized_to_target` sizes the C5 box from the object it must cover (30–60%
  of its box area, the class's aspect ratio, ≤ `max_area_frac`), used when the usual size fails,
  in the C5-only plan fallback and in C5 re-plans. The C5 rules (20–50% of one object, ≤ 5% of
  any other) are unchanged. Test with a frame-filling train; `pytest -q`: 66 passed. Plans
  re-checked: week 1 544/544, pilot 1 46/46, same-class pilot 48/48 identical; 562443 gained a
  plan (run_plan_2026-10-16T162944Z.json).
- **Correction:** 151857 and 457848 already had C5-only plans; their objects failed. The 6
  re-plan rows of 457848 were empty because they were computed with the old sizing; they were
  removed (backup `data/inlayfake_pilot_inphoto/replans_before_targetsize.jsonl`) and
  recomputed.
- **Run** (run_verify_2026-10-16T163304Z.json,
  paper/results/task2_pilot_inphoto_targetsize_verify.json; 3 min): 562443 zebra passed on
  attempt 2 (an AI zebra head covering the grazing zebra's head, convincing); 457848 passed on
  box 2 (a small AI motorcycle, real but small for the scene). Totals: C4 39/41, **C5 37/42**,
  C2 39/41; final ledger 252, audit sample 12. **48 of 50 photos have a verified AI object.**
- **Still without one (2):** 151857, 12 attempts at a train over the real train: SDXL extends
  the real train, which the original-object guard correctly refuses, and there is no other class
  in the photo; 421455, no box satisfies the C5 rules. Fixing these would need a rule change
  (another class not in the photo, or a different occlusion rule), so they are left out.

---

## 2026-10-17: Class switch and C5-only plans (run `task2_pilot_inphoto`): 46 of 50 photos have an AI object

- **Author's request:** as many photos as possible must get an AI-inserted object.
- **Class switch** (`edit.max_class_switches`, 0 by default, 3 in `pilot_inphoto.yaml`): after the
  re-planned boxes of the planned class fail, a track moves on to another class already in the
  photo (same rules: audited GT, never person), smallest median instance area in the photo
  first, one class per version. Versions where no box fits are skipped instead of ending the
  job. The class is stored per version (`category` in `replans.jsonl`); C2/C3 follow the C4
  class and box, C5 uses its own track's class.
- **C5-only plans:** in `in_photo` mode a photo with no free space for any class may still get
  an occluding box (no C2/C3/C4). That search uses its own seeded generator, so the other plans
  are unchanged: checked byte for byte — week 1 544/544, pilot 1 46/46, same-class pilot 41/41
  identical; 7 photos gained a C5-only plan (48 of 50 planned; run_plan_2026-10-16T155841Z.json).
- **Tests:** 3 new (version skipping, class switch with matched C2/C3, C5-only plan leaves the
  shared generator untouched); `pytest -q`: 65 passed.
- **Run** (run_verify_2026-10-16T162120Z.json,
  paper/results/task2_pilot_inphoto_classswitch_verify.json; 22 min): **C4 39/41**, **C5 35/41**
  (6 of them after a class switch: motorcycle ×2, bus, surfboard, dining table, handbag),
  **C2 39/41** clean, final ledger 248 records, audit sample 11.
- **Per photo:** **46 of 50** sampled photos now have at least one verified AI object (C4 or
  C5); 39 have a C4. Before the change: 39. The 4 without one: a photo with a single train, one
  with a single zebra, and two crowded street scenes (dog, phone, car, truck; 3 motorcycles among
  5 people) where no box fits for any class.
- **Visual check of the 10 new C5 objects:** 8 clear (toilet, motorcycles, giraffe, handbag,
  spoon, dining table), 2 weak (a tiny surfboard on the sand, a very small bus).
- **Next:** author audit (11 rows); full-run configuration (re-plan of all 600 sources).

---

## 2026-10-16: Re-plan on failure (run `task2_pilot_inphoto`): 93% of C4 with a verified object

- **Author's request:** the photos must get an AI-inserted object. Loosening the verifier
  further was advised against and not done: all 31 dropped jobs were checked by eye and their last
  attempts contain background (or, under C5 boxes, the photo's own object), so passing them would
  put empty `ai_inserted` boxes into the ground truth. Instead, failed jobs get new boxes.
- **Implemented** (`stages/replan.py`, `edit.max_replans`, default 0 = unchanged): when all
  `max_attempts` seeds fail on a box, the job gets box version n+1 (up to `max_replans`). Two
  tracks per source: *free* (C4; C2 sham and C3 paste follow it, so controls stay matched to the
  C4 box) and *occ* (C5). A new box rests on a surface of the class (COCO-Stuff; free space if
  none fits) and is sized like one of the photo's own instances of that class (×0.8–1.2).
  Version 0 keeps the original seeds, so existing images are reused; versions are stored in
  `replans.jsonl`; `EditInfo.plan_version` records the box. `verify` and the final ledger count
  only the current box (one C3 per source). Preview shows "box n" and marks superseded C3.
  Fake verifier gained `fake_fail_until` for tests; 3 new tests; `pytest -q`: 62 passed.
- **Run** (`configs/pilot_inphoto.yaml`, `max_replans: 3`, window 1.2×, looser verifier;
  run_verify_2026-10-16T130208Z.json, paper/results/task2_pilot_inphoto_replan_verify.json;
  boxes in paper/results/task2_pilot_inphoto_replans.jsonl; 65 min wall time):
  31 re-plans created (13 on a surface, 14 free space, 4 with no box that fits).
  **C4 38/41 (93%)** — box 0: 27, box 1: 4, box 2: 7, 3 dropped; **C5 25/34 (74%)** — box 0: 17,
  box 1: 4, box 2: 1, box 3: 3, 9 dropped; **C2 39/41** clean (the 2 missing are sources where no
  new box fit, which also have no C4). Final ledger 223 records, audit sample 10.
- **Visual check of the 19 edits that passed on a new box:** about 16 clearly show the object
  (dog, elephant, train, book on the floor, horse, skis, vase, dining table, traffic light,
  airplane, surfboard on the beach); 3 are weak (a tiny boat, a blurred car, a tiny surfboard);
  a few are still implausible (umbrella above the sea, bed in a hedge).
- **Limits:** when no new box fits (4 crowded photos) the source loses its C2/C3 at the old box
  as well; 85% target met for C4 (93%), not yet for C5 (74%); human audit pending.
- **Next:** author audit (`audit_edits.csv`, 10 rows); then decide whether to run the full
  benchmark with this configuration (re-plan of all 600 sources with the same-class rule).

---

## 2026-10-16: Same-class pilot at window 1.2× with a looser verifier (run `task2_pilot_inphoto`)

- **Author's requirement:** the AI object must show. Generation can never guarantee it, so the
  guarantee is on the benchmark side: only verified edits enter `ledger.jsonl`. Changes made to
  raise the share of photos with a verified object:
  1. **Window 1.2×, minimum 64 px** (`configs/pilot_inphoto.yaml`; new `edit.window_min_px`,
     default 256 unchanged), the best setting of the 12-edit diagnostic. Verification keeps
     looking at the 2.5× region (`verify.window_scale`), which equals the old crop box, so the
     earlier pilots are unaffected.
  2. **Looser verifier rule (author decision):** a target-class detection with score ≥ 0.2
     passes if IoU ≥ 0.3 with the box **or ≥ 80% of it lies inside the box**
     (`verify.min_inside: 0.8`, now the default); a detection with IoU ≥ 0.5 to a same-class
     original object never counts (`verify.original_iou`), so the occluded real object in C5
     is not mistaken for the insertion. Why: a CPU re-scoring of the 58 round-1 rejections
     (paper/results/task2_verifier_iou_diagnostic.{py,txt}) found 14 with a confident
     (≥ 0.2) detection inside the box and IoU < 0.3 (frisbee 0.74 at IoU 0.01, bus 0.77 at
     0.08, toothbrush 0.68, cups 0.56): objects smaller than the box. C2 uses the same rule, so
     the sham check is stricter.
  3. **Preview** marks every attempt: "in benchmark" (verified box), "rejected … not in
     benchmark" (dimmed) or "not needed: an earlier attempt passed". Earlier previews rebuilt.
  4. **Safe re-verification:** `verify` checks attempts in order and stops at the first pass per
     job; the final ledger keeps one passed record per job; rows are flushed as written.
  Tests: 59 passed, ruff clean.
- **Round 1 under the old rule** (run_verify_2026-10-16T104526Z.json): C2 41/41, C4 10/41,
  C5 7/34 (17/75 = 23%; pilot 1 at 2.5×: 6%). SDXL 116 images in 50:07 (25.9 s each). Round 2
  (58 images, 38:31, slowed by a CPU diagnostic running at the same time) was generated and
  then the run was stopped for the rule change; the old-rule results are kept in
  `data/inlayfake_pilot_inphoto/verify_oldrule/`.
- **Final, new rule, up to 3 attempts** (run_verify_2026-10-16T114020Z.json,
  paper/results/task2_pilot_inphoto_final_verify.json): **C4 27/41 (66%)** — 20 on attempt 1,
  6 on 2, 1 on 3, 14 dropped; **C5 17/34 (50%)** — 8 / 9 / 0, 17 dropped; **C2 41/41**; final
  ledger 208 records; audit sample 8. Round 3: 33 + 2 images (19.0 and 22.0 s each).
- **Visual check of all 44 passed C4/C5:** nearly all show a real object of the class (dogs,
  sheep, horse, giraffe, book, broccoli, cups, cat, traffic lights, backpacks, handbag, train).
  Weak: a dark laptop (0.34), a faint umbrella (0.30), a thin surfboard line, a tiny baseball
  glove (0.28). Realism is now the main problem: several float in implausible places (bus in the
  sky, car in the trees, handbag under a hangar roof, tiny giraffe on a hilltop). The dropped
  jobs are mostly such placements and very large classes (train, bed, dining table, surfboard).
- **Interpretation:** the verified-object rate rose from 6% (pilot 1) to 66% C4 / 50% C5,
  through the tighter window, the looser rule and 3 attempts together; the 85% PRD target is not
  met yet. Photos without a verified C4/C5 simply have none in the benchmark.
- **Next:** author's look at `preview.html` and the 8-row `audit_edits.csv`; options for the
  rest: surface placement with the same-class rule (against floating objects), box size from the
  photo's own instance, more attempts.

---

## 2026-10-16: Decision: insert a class already in the photo (run `task2_pilot_inphoto`, plan only)

- **Decision (author):** the AI object must be of a class that is already in the photo's audited
  week-1 ground truth, any COCO class except person, instead of cycling through 10 fixed
  categories. Prompted by source 23937 (5 cows, 1 sheep) being planned as "chair". Why it helps:
  each edited photo then has a photographed and an AI instance of the same class, a matched
  within-class comparison for H1/H3.
- **Before the change (week-1 plan, 544 photos):** only 196 contained any of the 10 fixed
  classes, and the planned class was already in the photo in 20 of 544.
- **Implemented:** `placement.class_choice: in_photo` (`stages/plan.py`). Present classes =
  non-crowd COCO labels + week-1 audit "add" rows (`data/inlayfake/audit_decisions.csv`), minus
  `exclude_classes: [person]`; least-used class first, ties by the seeded RNG; audit add / ignore
  boxes block free placement. `fixed` stays the default in `default.yaml`, and it reproduces the
  week-1 (544) and first-pilot plans byte for byte (checked). New `configs/pilot_inphoto.yaml`
  (free-space placement, window 2.5×). 4 new tests; `pytest -q`: 57 passed; ruff clean.
- **Re-plan of the pilot** (same 50 sources; paper/results/task2_pilot_inphoto_plan.json):
  41 planned, 9 failed, 34 with an occlusion box (pilot 1: 46 / 4 / 38); 31 classes, the most
  frequent surfboard 4, car 3. Source 23937 → sheep. The 9 failures are photos whose only
  non-person classes are large or crowded (train, giraffes, zebra, 10 motorcycles, a toilet).
- **Seen on the plan sheet:** placement is still any free space, so some boxes sit in the sky
  again (handbag at a hangar ceiling, elephant, car, surfboard); box size comes from COCO-wide
  statistics, not from the same-class object already in the photo. Not changed.
- **Implications:** classes are no longer balanced and include large or non-rigid ones (bus,
  giraffe); the week-1 counts in PAPER §4 (544 planned, 363 with C5) come from the old plan and
  must be recomputed before the full run (`insertion.yaml` still reuses the week-1 plans).
- **Not run yet:** generation. Open: window size (1.2× / 1.5×), surface placement, size from the
  photo's own instance.

---

## 2026-10-16: Surface placement pilot (run `task2_pilot_surface`): placement is not the cause; window size is

- **Decision (author):** try boxes that rest on a supporting surface. Implemented as
  `placement.support` (`edit/support.py`; ARCHITECTURE §3.3): COCO-Stuff human labels
  (`stuff_val2017.json`, downloaded with the author's OK from images.cocodataset.org; only that file
  kept). A box rests on ground (floor, carpet, grass, dirt, sand, pavement, …) or a tabletop
  (table, desk, counter, shelf) when ≥ 50% of the band from 15% of its height above the bottom
  edge to 5% below it is surface; clocks may also sit inside a wall region. Classes → surfaces in
  `configs/default.yaml` (off there, so week-1 plans and `insertion.yaml` are unchanged); on in
  `configs/pilot_surface.yaml`. 5 new unit tests (hand-computed band fractions, RLE decoding, an
  end-to-end plan); `pytest -q`: 53 passed.
- **Plan** (same 50 sources as `task2_pilot`; paper/results/task2_pilot_surface_plan.json):
  35 planned (pilot 1: 46), 15 failed, 19 with an occlusion box (pilot 1: 38). Surfaces: ground
  28, wall 4, tabletop 3. Cost of the rule: fewer plannable photos, C5 halved, cup only 1.
- **Round-1 verify** (run_verify_2026-10-16T085229Z.json; code was uncommitted at run time and is
  committed unchanged with this entry): `C2 passed 35/35` · `C4 passed 0/35` · `C5 passed 4/19`
  → **4/54 (7%)**, no better than pilot 1 (5/84). SDXL: 89 images in 38:07, 25.7 s each.
  Retries stopped after round 1 again (state clean).
- **Visual check:** the same failure in plausible places: a bench box on a lawn becomes lawn, a
  plant box on a counter becomes counter. So implausible placement was not the main cause.
- **Window-scale diagnostic** (scratch only; script and output in
  `paper/results/task2_window_scale_diagnostic.{py,txt}`): 12 random failed C4 jobs of this
  pilot, new seed, strength 0.99, the project's `local_edit` and OWLv2 rule, verified on the
  same region. The square window around the box (`edit.window_scale`, now 2.5×, so the box is
  only 40% of SDXL's canvas side) was varied, with the 256-px minimum window lowered to 64 for
  the test: **2.5× → 1/12, 1.5× → 5/12, 1.2× → 7/12**. The objects look real (dog, backpacks,
  potted plants, wall clocks, chairs) and colours match; no tinted rectangle as with strength
  1.0. Interpretation: when the box is a small part of the canvas, SDXL completes the
  surrounding scene; when it fills most of the canvas, SDXL paints the prompted object.
- **Also seen:** OWLv2 misses some real insertions (a chair at 1.5× and 1.2×, a wall clock at
  1.2×, a stopwatch at 1.5× are visible but fail), and one "bench" pass is really a chair. So the
  verifier under-counts in places and is not a perfect label; the human audit matters. Tighter
  windows give less context: one dog is large for its spot, one umbrella floats.
- **Open for the author:** change `edit.window_scale` (and the 256-px minimum) to 1.2–1.5, keep
  or drop surface placement (it costs 11 of 46 plans and half of C5), and whether to look at the
  OWLv2 misses (threshold 0.2, query "a photo of a {category}").
- **Next:** author decision, then a third pilot before the full run.

---

## 2026-10-16: SDXL pilot (run `task2_pilot`): insertion mostly fails, full run on hold

- **Setup:** RTX 4070 Laptop (8 GB), 32 GB RAM (`systeminfo`: 32,471 MB; now `hardware.ram_gb`),
  torch 2.6.0+cu124, diffusers 0.41.0, transformers 5.19.0. `pytest -q`: 48 passed. Work dir
  `data/inlayfake_pilot`, git 0261677 (dirty only by the `ram_gb` config line).
- **sample / plan:** 50 sources (33 train / 10 val / 7 test); 46 planned, 4 failed, 38 with an
  occlusion box. Jobs: 46 C2 + 46 C4 + 38 C5 = 130 SDXL edits; 46 C3 pastes.
- **Speed:** smoke test `generate --limit 3`: images 2–8 took 190 s (≈27 s each; the first
  included the 7 GB download). Round 1 of `insert`: 122 SDXL images in 43:57, **21.6 s per
  image** including model load (run_generate_2026-10-16T074120Z.json). OWLv2 + SAM verify: 130
  images in 42 s.
- **Round-1 verify** (run_verify_2026-10-16T074553Z.json, attempt 1 only):
  `C2: passed 46/46` · `C4: passed 4/46, needs_retry 42` · `C5: passed 1/38, needs_retry 37` ·
  final ledger 189 records · audit sample 5. Insertion success on attempt 1 is therefore
  **5/84 (6%)**, far below the PRD target of ≥ 85%.
- **Retries stopped on purpose:** `insert` was stopped at the start of round 2 (no round-2 image
  was written, so the state is clean and resumable). With 6% per attempt, 3 attempts cannot reach
  the target, and retries would use up the attempt budget before the author decides what to change.
- **Bug hunt, no bug found:** the verifier agrees with a visual check on all 8 smoke-test edits
  (the 2 visible insertions passed, the 6 empty boxes failed), and a 20-tile contact sheet of
  failed C4/C5 edits shows plausible background in every box, no missed object. `compose.py`
  (mask white = inpaint, image and mask both resized to 1024, mask-only composite) is correct.
  So the failure is SDXL behaviour under the current settings, not a code error.
- **Diagnostic (scratch only, not in `data/`):** 12 random failed jobs re-run with a new seed
  through the project's `local_edit` and the OWLv2 rule: **strength 0.99 → 2/12 pass, strength
  1.0 → 3/12**. Strength 1.0 does not fix it and adds two problems: SDXL's colour shift then
  lands inside the mask, so the box is a visibly tinted rectangle (a forensic shortcut), and one
  output painted a person (ethics rule 10). Guidance 12 and no negative prompt also gave background
  on the two cases tried.
- **Interpretation:** the failures share implausible placements: free-space boxes in empty sky,
  mid-road, on a wall or at a night streetlight (umbrella, backpack, chair, cup). SDXL-inpainting
  completes the scene instead of adding an object that does not belong there. The 5 passes
  (bench, chair ×2, dog, bottle) look like real objects, but two sit in odd places (a floating
  bottle, a chair in water).
- **Sham caveat:** C2 passes when OWLv2 finds no object *of the target class*. Sham 31050
  (target bottle) produced a white bird and still passed; the 0/46 "sham produced the target"
  figure does not mean that no object appeared.
- **Open for the author:** (a) visual judgement in `preview.html` / `audit_edits.html` and the
  audit CSV; (b) what to change before the full run. Options: placement on supporting surfaces
  (box bottom on a floor/table/ground region) instead of any free space; larger minimum box for
  small classes; a prompt with scene context; or a generator better at object insertion.
  These are research settings, so nothing was changed.
- **Next:** author decision, then re-run the pilot (`insert` resumes) before
  `configs/insertion.yaml`.

---

## 2026-10-15: Week 2 started: verification stage, insert loop, viewer for C2–C5

- **Branch:** `week-2-ai-insertion` (from `week-1-coco-dataset`). Briefing folder
  `docs/briefings/week2/` created (full, one-page, slides; results all TBD).
- **verify** (new): OWLv2 must find the target class in the crop window (IoU ≥ 0.3 to the plan,
  score ≥ 0.2); C2 sham passes only if it does not. SAM, prompted with the OWLv2 box and clipped to
  the intended mask, gives the inserted mask and box. Actual changed mask (|edit − C1| > 4, 3×3
  opening). Occluded real objects get visible mask, box and fraction (< 10% → ignore region). Real
  objects overlapped > 10% by the changed region → `mixed_modified`. 10% human-audit sheet.
- **Why OWLv2 and not one of the four detectors:** verifying with an evaluated detector would keep
  only insertions that detector sees and inflate its recall on inserted objects.
- **Retries:** failed edits are regenerated with a new seed (attempt n+1, max 3); `insert` loops
  generate → verify. `generate` no longer loads a 7 GB model when no job is pending.
- **Final ledger** `ledger.jsonl`: C0/C1 unchanged (tested, so week-1 results are unaffected), C3
  with its pasted mask, one passed record per C2/C4/C5 job.
- **Bug caught before the GPU run:** transformers 5.x removed `Owlv2Processor.post_process_object_
  detection` (now `post_process_grounded_object_detection`); the wrapper supports both. The SAM wrapper
  was run with random weights on CPU; real OWLv2/SAM weights cannot be downloaded in the sandbox.
- **Viewer:** lists C2–C5 per generator; inserted objects purple, `mixed_modified` amber; detectors
  not yet run on an image show "not run on this image yet" instead of zero detections.
- **Tests:** 48 pass (end-to-end insert; forced verifier failure regenerated with a new seed).
- **Open:** laptop system RAM (FLUX needs about 32 GB on an 8 GB GPU, else Kandinsky 2.2).
- **Next:** SDXL pilot on the laptop (`configs/pilot.yaml`), then the overnight `insert` run.

---

## 2026-10-15: README: how to view and re-run the detector viewer

- The author committed the shareable week-1 viewer (`docs/viewer/task1_real_baseline/`: 1,088
  downscaled photos, about 70 MB).
- README now explains how to open it without setup, how to regenerate the local viewer (`report` →
  `viewer` → open, with timings), and how to rebuild and push the shareable copy.

---

## 2026-10-15: Shareable viewer (`viewer --share`)

- **Why:** the author wants others to view the runs. Committing `runs/` is not enough: the page
  loads photos from the ignored `data/` folder, and the raw run files are large and regenerable.
- **What:** `python -m inlayfake viewer --config <cfg> --share` writes `docs/viewer/<run>/` with
  the page, `data.js` and downscaled JPEG copies (max side 640, q80) of every shown image. This
  folder is self-contained and can be committed, zipped or served with GitHub Pages. AGENTS.md
  rule 8 now names it as the way to share runs.
- **Test:** the share folder only references its own `images/`, and copies are ≤ max side.

---

## 2026-10-15: Viewer shows results only (slider removed)

- **Why:** the author wants the page to show the results as computed, not an adjustable view: a
  free score slider could display a different operating point from the reported numbers.
- **Change:** boxes are drawn only at the evaluation operating point (score ≥ `eval.recall_score`
  = 0.5, IoU ≥ 0.5). The found-rule is now identical to `evaluate.detected` / `report`, so each
  panel's "found X/N" equals the `images_used.md` cell; an end-to-end test checks this for every
  photo. The run's `summary.md` and the `images_used.md` totals are rendered at the top. Category,
  split and "show only" filters were removed; the C0/C1 switch, id search and pages remain.
- **No new results.**

---

## 2026-10-15: Detector viewer web page (`viewer` stage)

- **Why:** the author wanted to see, per photo, the original COCO image with its labels next to what
  each detector found, rather than only AP tables.
- **What:** `python -m inlayfake viewer --config configs/real_baseline.yaml` writes
  `runs/task1_real_baseline/viewer/index.html` (+ `data.js`), opened directly from disk. The page
  has a ground-truth panel plus one panel per detector, coloured with the same matching rule as
  `evaluate` (green matched, orange unmatched, grey on an ignore region, dashed = label missed). It
  has a score slider, a C0/C1 switch, filters (category, split, photos with misses / unmatched boxes /
  audit changes, id) and click-to-enlarge.
- **Checks:** unit tests for greedy one-to-one matching (duplicates → unmatched, ignore regions)
  and an end-to-end CPU test; rendered in headless Chromium on the synthetic fixture.
- **No new results:** this is a visual tool; numbers in the paper are unchanged.

---

## 2026-10-12: Decision: weekly briefings are kept current like the paper

- **Decision (author):** the briefing for the current week (`docs/briefings/weekN/`:
  `briefing_full.md`, `one_page.md`, `slides.html`) is updated after every run, finding, decision or
  code change for the week's task, and created at the start of a new week. Added as
  `AGENTS.md` §Weekly briefing protocol, as step 6 of the experiment workflow, and in `CLAUDE.md`.
  Why: the briefings had fallen behind `briefing_full.md` and the paper between sessions.
- **Found while applying it:** week 1's `one_page.md` and `slides.html` still say the 56 unplanned
  photos were "too cluttered" and that there are 37 unit tests; `briefing_full.md` already describes
  them as close-ups (Key finding 4) and the log records 38 tests.
- **Fixed (author approved):** the close-up wording now appears in PAPER §4 (Insertion plan, with the
  median object coverage 97% vs 32%) and §8, and in the one-pager and slides. The test count reads
  38 everywhere (`pytest -q`: 38 passed on 2026-10-12). The one-pager also gains the close-up and
  per-photo detection findings and the VLM and CVPR-dates discussion points, so all three week-1
  files agree.

---

## 2026-10-11: Task 1 final evaluate with 200 bootstrap resamples; week 1 complete

- **Run:** `task1_real_baseline`, `evaluate` (run_evaluate_2026-10-11T150104Z.json, git 780a397)
  with `eval.ap_bootstrap: 200` (`configs/real_baseline.yaml`), on the RTX 4070 laptop. Reuses the
  saved detections and forensic scores and the confirmed audit decisions (77 add / 26 ignore /
  4 reject). Then `report` for the per-photo table. Summary saved as
  `paper/results/task1_real_baseline_summary_v4_200boot.md`.
- **Check:** every point estimate (AP, AP50, AR100, recall, lost/gained, forensic means, medians,
  p95, object contrast) is identical to v3 (`..._v3_audited.md`). Only the confidence intervals
  changed, so the inputs are unchanged.
- **Final AP 95% CIs (C0 / C1):** Faster R-CNN 0.500 [0.460, 0.515] / 0.498 [0.460, 0.510];
  YOLO11-m 0.530 [0.484, 0.547] / 0.528 [0.487, 0.547]; RT-DETR 0.512 [0.479, 0.527] /
  0.511 [0.476, 0.532]; Grounding DINO 0.515 [0.481, 0.532] / 0.515 [0.476, 0.525]. The intervals
  are wider than the 20-resample ones and overlap heavily across detectors: on 544 photos, the AP
  ranking of the four detectors is not resolved.
- **Resave shift (C1 − C0) CIs:** ELA +0.188 [0.170, 0.206], SRM −0.106 [−0.114, −0.099], noise
  −0.089 [−0.142, −0.038], FFT +0.0045 [0.0029, 0.0064], JPEG grid −0.025 [−0.027, −0.022]. All
  still exclude zero, so Findings (ii) in §7.1 stand. They moved slightly because the random
  generator is shared with the AP bootstrap.
- **Per-photo table:** `docs/briefings/week1/images_used.md`. Totals found / 3,959 reproduce recall
  C0 exactly (2938 → 0.742, 2053 → 0.519, 2387 → 0.603, 1826 → 0.461).
- **Updated:** PAPER §7.1 Tables 2a/2b and Status box, PRD Status, briefing Key finding 5.
- **Note:** PAPER §6 says CIs use 1000 resamples; Task 1 uses 200 (stated in Table 2a). Decide
  before submission whether the main experiments use 1000 or §6 should say 200.
- **Next:** Task 2 (AI insertion): SDXL pilot on 50 sources with `configs/pilot.yaml`.

---

## 2026-10-11: The 56 photos without a plan are close-ups; new `report` stage

- **Finding:** the 56 sources that failed `plan` are not busy scenes. Their median number of
  labelled objects is 4, the same as the 544 kept photos, but the union of their object boxes covers
  a median 97% of the frame (kept: 32%). In 52 of 56, one object covers more than half the image,
  most often a dining table (18), bowl (6), bed (5) or person (5). Split: 39 train, 7 val, 10 test.
  Computed from the plan stage's `failed_sources` list (seed 20261008) and the COCO val2017
  annotations. Interpretation: the benchmark under-represents close-up, frame-filling shots, not
  crowded scenes. The paper's wording ("too cluttered", "very crowded scenes", §4 and §8) should
  change; waiting for the author's go-ahead and a check of the list on the RTX 4070.
- **Added:** the full list of 56 (COCO IDs linked to the official images, split, object count,
  coverage, main labels) as Key finding 4 in `docs/briefings/week1/briefing_full.md`.
- **New stage `report`:** `python -m inlayfake report --config ...` writes
  `runs/<run>/tables/images_used.{csv,md}`, one row per C0 photo with split, planned insert,
  labelled objects and the number found per detector (the same rule as `evaluate`'s recall, so the
  totals reproduce recall C0). Why: the author wants the briefing to list every photo used and what
  was detected in it. Unit test with a hand-checked example; 38 tests pass. The real table is
  produced on the RTX 4070 (added to `TASK.md`).

---

## 2026-10-11: Decision: Stage 2 (VLM origin claims, H5) is a must-have

- **Decision (author):** the VLM caption study is no longer optional (P1). It is part of the P0
  scope, and H5 is a full hypothesis rather than a stretch goal. This reverses the 2026-10-08
  decision that reduced Stage 2 to P1.
- **Why:** Stage 2 is part of the professor's brief, and the author wants the paper to cover it.
- **Changed:** PRD (H5 no longer "stretch"; Stage 2 moved into the P0 list; success criteria need a
  finding for H5; new risk row; timeline: `vlm` stage, caption metrics and the plain/grounded prompts
  in week 3, origin-aware prompts and caption evaluation in week 4). PAPER (abstract gains an H5
  placeholder; new contribution 4; "(P1)" removed from §2 and §7.7). ARCHITECTURE (P1 labels removed
  from `vlm` and `captions.py`). Week-1 briefing files updated.
- **Scope kept small to fit the schedule:** two small VLMs (Florence-2, Qwen2.5-VL-3B in 4-bit),
  three prompt settings, about 200 test images plus unedited controls. If time slips, fewer images,
  not dropping the study.
- **Next:** write the `vlm` stage in week 3 on the second laptop; runs that feed the paper happen on
  the RTX 4070.

---

## 2026-10-11: Label audit re-checked; config set for the final `evaluate`

- **Audit:** the author re-checked all 107 drafted decisions (77 add / 26 ignore / 4 reject) on a
  review page showing each crop next to its decision, and confirmed every one. No change to
  `audit_decisions.csv`, so Table 2a's point estimates stand.
- **Changed:** `configs/real_baseline.yaml` now sets `eval.ap_bootstrap: 200`, the setting for the
  paper's final tables (`default.yaml` keeps 20 for quick runs). Why in this config: the author only
  needs to re-run `evaluate` on run `task1_real_baseline`; `detect`/`forensics` outputs are reused.
- **PRD status** corrected: Task 1 has run; only the final `evaluate` remains.
- **Next:** on the RTX 4070 laptop, `git pull` then
  `python -m inlayfake evaluate --config configs/real_baseline.yaml`; record the 200-resample CIs in
  Table 2a/2b.

---

## 2026-10-09: Label audit applied, final Task 1 `evaluate` (run `task1_real_baseline`)

Source: `paper/results/task1_real_baseline_summary_v3_audited.md`, decisions in
`paper/results/task1_audit_decisions.csv`.

- **Audit decisions (107 candidates):** 77 add, 26 ignore, 4 reject. Drafted from a visual review of
  every crop in `audit.html`. **The author has not re-checked them yet.** If any change, re-run
  `evaluate` (≈ 5 min) and update Table 2a.
  - Reject: `19742-0` (a candle, not a cup), `253742-0/1` (deer, not sheep; deer is not a COCO
    class), `560474-0` (a doll's face, not a person).
  - Ignore: too small or dark to judge (10), possible duplicates of an existing label, e.g. `26204-1`
    overlaps `26204-0` (4), class doesn't clearly fit, e.g. statue cow, plush clownfish, cocoa tin as
    cup, poster person (7), other unclear (5).
  - Why ignore rather than guess: an ignore region is neither rewarded nor penalised, so an
    uncertain call cannot bias AP either way.
- **Effect on detectors:** 3,959 GT objects (was 3,882). AP rises by 0.004–0.006 for every detector
  and condition (e.g. Faster R-CNN C0 0.494 → 0.500, YOLO11-m 0.526 → 0.530). Resave flips are
  unchanged (0.40–0.76%). Interpretation: the missing labels made detectors look slightly worse
  than they are; the ranking and the C0≈C1 conclusion do not change.
- **Forensic tables unchanged:** per-object contrast is computed in the `forensics` stage over the
  original 3,882 labels, so the 77 additions are not in Table 2b. They are 2% of objects, so this
  is unlikely to matter; re-run `forensics` before the final version if we want them included.
- **Pre-audit outputs kept** as `runs/task1_real_baseline/summary_preaudit.md` and
  `metrics_preaudit.json`.
- **Paper:** §4 Sources filled (77/26/4), Table 2a and finding (iv) updated, Status box updated.
- **Week 1 done** apart from the author's re-check of the audit and 200-resample AP CIs.

---

## 2026-10-08: Task 1 re-run after fixes (v2, run `task1_real_baseline`)

Source: `paper/results/task1_real_baseline_summary_v2.md`.

- **jpeg_bag fixed:** image score mean 0.532 / median 0.484 / p95 0.99 (was 428.7 / 0.667). Object
  contrast −0.003 ± 0.053 (sd was 61.8). Resave shift −0.025 [−0.028, −0.022] (−4.7%). The method
  now behaves like the others and is sensitive to re-encoding.
- **AP CIs (basic bootstrap):** e.g. Faster R-CNN C0 0.494 [0.462, 0.506]. Intervals now contain the
  point estimate but are skewed downward: the resample mean is about 0.015 above the point estimate
  (the category-dropout bias), and the reflection moves that skew below. 20 resamples only; use 200
  for the final tables.
- **New observation:** F4 (FFT) and F5 (blocking grid) have ≈ 0 mean object-vs-context contrast on
  real objects, unlike ELA/SRM/noise (0.25–0.51). They may be less confounded by object texture;
  check this when AI objects arrive (Task 3).
- All other numbers are identical to v1, as expected (only jpeg_bag and the CI method changed).
- **Paper:** Tables 2a/2b complete (F5 row and AP CIs filled); findings (ii) and (iii) updated.
- **Week 1 remaining:** audit review of 107 candidates, then a final `evaluate`.

---

## 2026-10-08: Task 1 results (v1) and two fixes (run `task1_real_baseline`)

Source: author's `summary.md`, saved as `paper/results/task1_real_baseline_summary_v1.md`.

- **Detectors on real photos:** AP C0 / C1: Faster R-CNN 0.494 / 0.493, YOLO11-m 0.526 / 0.524,
  RT-DETR 0.508 / 0.507, Grounding DINO 0.511 / 0.510. These are in the range expected for these
  models on COCO val, a sanity check that the wrappers, the category mapping and the evaluation work.
- **Resave effect (C0→C1):** AP changes ≤ 0.002. Per-object flips are 0.41–0.77% lost and gained,
  roughly symmetric (3,882 objects). This is the noise floor for H2.
- **Recall at score ≥ 0.5 differs a lot by detector** (0.46–0.74), mostly because of score
  calibration: YOLO and Grounding DINO give lower scores. Use it only for within-detector
  comparisons (flips), never to rank detectors.
- **Forensics on real photos:** all methods shift significantly with resaving (ELA +3.7%, SRM −3.5%,
  noise −0.9%, FFT +1.2%), so C1, not C0, must be the reference. Real objects have positive mean
  contrast against their context for ELA (0.50), SRM (0.30) and noise (0.25): texture makes real
  objects stand out, so calibration has to be against real objects.
- **Problem 1, jpeg_bag outliers:** mean image score 428 vs median 0.67, and object-contrast sd 62.
  The ratio on/off-grid divided ~0 by ~0 in flat or saturated regions. **Fix:** log ratio with +1 grey
  level in numerator and denominator; regression test added (fails on the old code). Requires
  re-running `forensics` for jpeg_bag only.
- **Problem 2, AP CIs biased:** all 8 percentile intervals sit mostly above the point estimate
  (e.g. 0.494 [0.479, 0.527]). COCO AP averages over the categories present, and rare, low-AP
  categories drop out of resamples, so resampled APs are inflated. **Fix:** basic (reverse-percentile)
  bootstrap interval; the resample mean is also stored. Requires re-running `evaluate`.
- **Paper:** §7.1 drafted with Table 2a/2b and four findings; F5 row and AP CIs left as TBD until
  the re-run.
- **Next:** delete `forensics/jpeg_bag.jsonl`, re-run `forensics` and `evaluate`, then review the
  audit.

---

## 2026-10-08: Task 1 `forensics`, `audit`, `evaluate` complete (run `task1_real_baseline`)

- **forensics:** 1,088 images × 5 methods in 2 min 55 s (6.2 img/s on the laptop CPU).
- **audit:** 107 candidate missing labels in 70 of 544 C0 images (12.9% of images have at least one
  confident object, agreed by ≥ 2 detectors, that COCO did not label). This is consistent with the
  label-incompleteness finding of *Object Detection Benchmarks are Incomplete*. No decisions yet: all
  are "?" and are therefore treated as ignore regions in this evaluation.
- **evaluate:** finished; `runs/task1_real_baseline/summary.md` written (1,088 records). The numbers
  will be recorded once the author shares the summary.
- **Next:** author pastes `summary.md`; reviews `audit.html` and fills `audit_decisions.csv`; re-run
  `evaluate`.

---

## 2026-10-08: `evaluate` looked stuck: progress output added

- **What:** `evaluate` printed nothing for a long time on the laptop. The cause is the AP bootstrap:
  4 detectors × 2 conditions × (1 + 50 resamples) ≈ 400 full COCOevals on 544 images, with no
  progress output.
- **Fix:** progress messages per detector/condition and a progress bar for the resamples. The default
  `eval.ap_bootstrap` is now 20 for quick runs; it will be raised to 200 for the paper's final tables
  (wider-than-final CIs until then).

---

## 2026-10-08: `detect` complete for all four detectors (run `task1_real_baseline`)

- **Ran (author's laptop):** after the fix, Grounding DINO processed all 1,088 images in 22 min 13 s
  (1.23 s/img, 4 prompt chunks per image). The other three were skipped (already complete).
- **State:** detections exist for faster_rcnn, yolo, rtdetr and grounding_dino on 544 × (C0, C1).
- **Next:** `forensics`, `audit`, `evaluate`.

---

## 2026-10-08: `detect` run: 3 of 4 detectors done; Grounding DINO bug fixed

- **Ran (author's laptop):** `detect` on 1,088 images. Faster R-CNN (1 min 56 s, 9.3 img/s), YOLO11-m
  (26 s, 41 img/s) and RT-DETR (59 s, 18 img/s) completed. Grounding DINO crashed after 10 images
  (≈1.4 s/img): `zip() argument 2 is longer than argument 1`. The installed transformers'
  `post_process_grounded_object_detection` returned more phrase labels than boxes.
- **Fix:** Grounding DINO labels are now decoded directly from the token logits. Each query's score
  for a class is the max sigmoid over that class name's tokens, and the query keeps the best class if
  that score is ≥ 0.05. This no longer depends on the processor's post-processing (which changes
  across versions). Unit tests cover the prompt spans, token→class mapping and decoding.
- **Action for the author:** delete the partial `grounding_dino.jsonl` (its 10 rows used the old
  decoding) and re-run `detect`; the other three detectors are skipped automatically.

---

## 2026-10-08: Task 1 data stages run on the laptop (run `task1_real_baseline`)

- **Ran (author, RTX 4070 laptop):** `sample`, `plan` and `generate` with `configs/real_baseline.yaml`.
- **Numbers (as printed by the CLI):**
  - sample: 600 sources; train 362 / val 114 / test 124 (60.3 / 19.0 / 20.7%).
  - plan: 544 planned, **56 failed (9.3%)**, 363 of 544 with a C5 occlusion box (66.7%).
    Categories: chair 62, bottle 56, potted plant 49, vase 50, clock 57, bench 51, backpack 65,
    dog 44, cup 59, umbrella 51.
  - generate: 1,088 records = 544 × (C0 + C1), real only; 27 s.
- **Interpretation:** The splits match the 60/20/20 target. The failed plans are photos with no free
  region for a new object (crowded scenes). All later conditions use the same 544 sources, so the
  benchmark under-represents very cluttered scenes; this has been added to the paper's limitations.
  C5 will cover only about 2/3 of sources; acceptable, but report C5 counts separately.
- **Next:** `detect` (GPU; first run downloads the four detectors' weights), then `forensics`,
  `audit`, `evaluate`.

---

## 2026-10-08: Windows test-environment fix

- **What:** On the author's laptop, `pytest -q` gave 28 passed and 5 errors. All 5 were
  `PermissionError [WinError 5]` on `%TEMP%\pytest-of-USER`: pytest could not open its Windows temp
  folder, so every test that uses `tmp_path` failed during setup. This was not a bug in the
  project code.
- **Fix:** `pyproject.toml` now sets `--basetemp=.pytest_tmp` (a git-ignored folder inside the repo).
- **Next:** the author re-runs `pytest -q` (expect 33 passed), then the Task 1 commands.

---

## 2026-10-08: Task 1 code complete (branch `week-1-coco-dataset`)

- **Implemented:** `detect` (Faster R-CNN v2, YOLO11-m, RT-DETR-R50, Grounding DINO-T), `forensics`
  (F1 ELA, F2 SRM residual variance, F3 wavelet noise-level deviation, F4 FFT high-frequency
  deviation, F5 JPEG blocking-grid deviation), `audit` (candidate missing labels → HTML +
  decisions CSV) and `evaluate` (Task 1 tables + `summary.md`).
- **Checks:** 33 CPU tests pass. Forensic tests confirm that ELA and the blocking grid score higher
  on a patch that skipped the JPEG history, and the noise method scores higher on a noisier patch.
  The detector code paths were smoke-tested on CPU: YOLO with real weights; Faster R-CNN and RT-DETR
  with random weights (their weight hosts are blocked in the cloud sandbox). Grounding DINO's
  post-processing signature was checked against the installed transformers 5.x. Grounding DINO is
  the least-tested wrapper, so watch it on the first laptop run.
- **Speed:** all five forensic maps take ≈0.12 s per 640×480 image on CPU.
- **Method note:** hand-crafted maps have no natural "AI" threshold. Task 1 records their behaviour
  on real photos; the false-alarm rate is computed in Task 3 at a threshold fixed on validation data
  with AI objects (TPR 80%). This was stated to the author.
- **Computer use:** the author offered control of their laptop, but no desktop-control tools are
  connected to this session. The workflow stays the same: code is written here and the author runs
  the commands.
- **Next:** the author runs Task 1 on the laptop and pastes `summary.md`, then reviews
  `audit.html`.

---

## 2026-10-08: Work split into tasks; Task 1 = real-only COCO baseline

- **Decision (author):** split the work. Task 1 runs only on real COCO photos, with no AI generation,
  to establish how detectors and forensic methods behave on genuine photographs first.
- **Interpretation recorded:** COCO (Flickr, about 2014) predates diffusion editors, so its photos
  are real by provenance. Forensic methods cannot *prove* an image is non-AI. What Task 1 measures is
  each method's **false-alarm rate on real photos**, plus detector AP on C0 vs C1 (the resave
  effect) and the label audit.
- **Changed:** `configs/real_baseline.yaml` (C0 and C1 only, no generators; same work dir, so Task 2
  reuses the images). Fixed `generators_enabled: []` (it previously fell back to *all* generators).
  New test confirms that a real-only run contains no AI records. PRD timeline reordered (Task 1 → 2 →
  3). PAPER §7.1 placeholder for the real-photo baseline.
- **Next:** implement the `detect` stage (4 detectors) and the hand-crafted forensic methods so that
  Task 1 can run on the laptop.

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
