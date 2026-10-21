# InlayFake: Week 2 progress (15â€“21 Oct 2026)

> Terminology: C4-occ = the occlusion placement rule of the AI insertion before 18 Oct 2026 (cover 20–50% of one real object); since then it is part of the single AI condition C4.

Last updated 21 Oct 2026 · Ryan Wong

Briefing for the project professor. Branch `week-2-ai-insertion`. Generation runs on the RTX 4070
laptop. The SDXL pilot (run `task2_pilot`, `data/inlayfake_pilot`) ran on 16 Oct; full-run
numbers (run `task2_insertion`) are still `[[TBD]]`.

**Goal for the week:** turn the 544 planned real photos into the benchmark by adding AI-inserted
objects under matched controls, and check that each insertion really contains the requested object.

## How an insertion works

(Since 16 Oct the inserted class is one already in the photo; see Done Â§6.) For each source
photo, week 1 already fixed one plan: a target category (10 classes, balanced), a
box size drawn from real COCO instances of that class, a free-space box and, for 363 photos, a box
covering 20â€“50% of one real object.

1. Crop a square window around the box: 1.2Ã— the box since 16 Oct (2.5Ã— before; the wider window
   mostly produced background, see Done Â§5).
2. Resize the window to SDXL's native size (1024 px).
3. Inpaint the box (+6 px) with the prompt "a photo of a {category}, realistic, natural lighting".
4. Resize back and composite **only inside the mask**, with a 3-px feather that is zero at the mask
   border. Every pixel outside the mask is identical to the matched resave C1 (unit-tested).
5. Save with the benchmark JPEG settings (q95, 4:4:4), the same as C1, so file format cannot reveal
   origin.
6. Record the edit in the provenance ledger: generator, seed, prompt, intended mask and plan box.

| Condition | What it is | Generator |
|---|---|---|
| C0 / C1 | original / matched resave (week 1) | none |
| C2 sham | same window and mask, prompt "empty background, nothing": the editor runs but adds nothing | SDXL |
| C3 real paste | a real COCO object of the same class, cut from another photo and pasted in the same box | none |
| C4 AI insert | **one AI object of a class already in the photo, placed anywhere** (C4 and C4-occ merged 18 Oct, see Done §20) | SDXL (SD2 and FLUX dropped 17 Oct, see Done §14) |
| ~~C4-occ~~ | ~~AI object covering 20–50% of one real object~~: merged into C4 | – |

## Done

### 1. Verification stage (`verify`): new code
- **Does the object exist?** OWLv2 looks for the target class in the crop window. OWLv2 is an
  independent open-vocabulary detector, deliberately *not* one of the four evaluated detectors, so
  the check does not favour them. An edit passes when OWLv2 finds the class with IoU â‰¥ 0.3 to the
  planned box and score â‰¥ 0.2.
- **Sham check.** A C2 edit passes only if OWLv2 does *not* find the class there.
- **Retries.** A failed edit is regenerated with a new seed, up to 3 attempts, then dropped and
  logged. `insert` runs generate â†’ verify in rounds until nothing is left to retry.
- **Exact object outline.** SAM, prompted with the OWLv2 box and clipped to the edit mask, gives the
  inserted object's mask and tight box for the ground truth.
- **What actually changed.** |edited âˆ’ C1| > 4 grey levels (after a 3Ã—3 opening) is saved per image.
- **Occlusion (C4-occ).** Real objects covered by the insertion get their visible mask, box and visible
  fraction updated; below 10% visible they become ignore regions.
- **Altered real objects.** A real object whose area overlaps the changed region by > 10% is
  re-labelled `mixed_modified` and reported separately.
- **Human audit.** A random 10% of passed edits go on `audit_edits.html` (before/after crops) with
  `audit_edits.csv` to mark y/n: object present, realistic, anything else changed.
- **Final ledger.** `ledger.jsonl` keeps C0/C1 exactly as in week 1 (test-checked, so week-1
  results are unchanged), adds C3 with its pasted mask, and one passed record per C2/C4/C4-occ job.

### 2. Viewer
The detector viewer now lists C2â€“C4-occ per generator, outlines inserted/pasted objects (purple) and
real objects altered by the editor (amber), and shows "not run on this image yet" for detectors not
yet run on the edited images (week 3).

### 3. Checks
48 CPU unit tests pass, including an end-to-end insert run and a forced verifier failure that is
regenerated with a new seed. The SAM wrapper was run with random weights on CPU. Checking the OWLv2
wrapper against the installed transformers 5.x caught a renamed post-processing function, now
handled for both old and new versions. The real OWLv2 and SAM weights cannot be downloaded in the
cloud sandbox, so their first real run is the pilot.

### 4. SDXL pilot (16 Oct, run `task2_pilot`)
- 50 sampled photos, 46 with a plan (38 with an occlusion box): 130 SDXL edits and 46 real pastes.
- **Most insertions failed.** On the first attempt, OWLv2 confirmed the object in 4 of 46 C4
  edits and 1 of 38 C4-occ edits. A visual check agrees: in the failed edits SDXL filled the box with
  matching background (grass, sky, road) instead of the object. The verifier is not the problem.
- **Not a code bug.** The compositing and mask handling were checked. A diagnostic on 12 failed
  edits passed 2/12 at the current setting and 3/12 at full denoising strength; full strength
  also leaves a visibly tinted rectangle and once painted a person, so it is not a fix.
- **Likely cause: placement.** Failed boxes sit where the object would not be: an umbrella in
  empty sky, a chair mid-road, a backpack at a streetlight. The 5 successes (bench, two chairs,
  dog, bottle) look like real objects, although two are in odd places.
- **Sham caveat.** All 46 sham edits passed, but the check only looks for the target class: one
  sham (target bottle) produced a bird.
- Retries were stopped after round 1 so that no attempts are wasted before the design changes.

### 5. Second pilot: boxes on supporting surfaces (16 Oct, run `task2_pilot_surface`)
- **New option `placement.support`:** each box must rest on a floor, ground or table surface
  from the COCO-Stuff human labels (clocks may hang on a wall). Off by default, so week-1 plans
  are unchanged; 5 new unit tests, 53 pass.
- Same 50 photos: 35 get a plan (was 46) and 19 an occlusion box (was 38).
- **Result: no better.** C4 0/35, C4-occ 4/19 (7%). SDXL again filled boxes with background, now
  in plausible places (a bench box on a lawn became lawn). Placement was not the main cause.
- **The cause is the edit window.** The generator sees a square of 2.5Ã— the box, so the object is
  only 40% of its canvas side. On 12 failed edits, a tighter window gave many more insertions:
  window 2.5Ã— â†’ 1/12, 1.5Ã— â†’ 5/12, 1.2Ã— â†’ 7/12, with natural colours (a diagnostic, not a run).
- The check also showed OWLv2 missing some objects that are clearly there (a chair, a wall clock),
  so the verifier under-counts in places.

### 6. Decision: insert a class that is already in the photo (16 Oct)
- **Change:** the AI object is now of a class already in the photo's audited week-1 labels (any
  COCO class except person), not one of 10 fixed categories. Example: photo 23937 (cows, sheep)
  was planned as "chair" and now gets a sheep. Each edited photo then has a real and an AI object
  of the same class, a fairer comparison of origin within a class.
- **Why it was needed:** in the week-1 plan, only 196 of 544 photos contained any of the 10 fixed
  classes, and the planned class was in the photo in only 20.
- **Pilot re-planned** (run `task2_pilot_inphoto`, plan only): 41 of 50 photos planned (was 46),
  34 with an occlusion box, 31 different classes. The 9 failures contain only large or crowded
  classes (train, giraffes, zebra).
- **Trade-offs:** classes are no longer balanced, and some boxes are again in the sky because
  placement is still any free space. The old design stays available and reproduces the week-1
  plan exactly. 57 tests pass.

### 7. Same-class pilot: tighter window and a looser check (16 Oct, run `task2_pilot_inphoto`)
- **Goal:** the AI object must show. A generator cannot guarantee it, so the benchmark does:
  only edits in which the verifier finds the object are kept; the rest are retried with a new
  seed and, after 3 attempts, dropped. The preview now labels every attempt "in benchmark" or
  "rejected", so a failed attempt is no longer mistaken for a benchmark image.
- **Changes:** generator window 1.2Ã— the box (was 2.5Ã—); the verifier now also accepts an object
  that lies mostly inside the box even if it is much smaller than the box. Why: 14 of 58
  rejections had a confident detection inside the box (e.g. a frisbee at score 0.74) that failed
  only the overlap rule. A detection on the photo's own same-class object never counts.
- **Result:** a verified AI object in **27 of 41 C4** edits (66%) and **17 of 34 C4-occ** edits (50%),
  up from 6% in the first pilot; all 41 sham edits stay clean. 59 tests pass.
- **What is left:** below the 85% target; several objects float in implausible places (a bus
  in the sky, a car in the trees), and very large classes (train, bed) usually fail.

### 8. Re-plan on failure (16 Oct, run `task2_pilot_inphoto`)
- **Why not loosen the check further:** the 31 dropped edits were checked by eye and contain
  only background, so accepting them would put empty "AI object" labels into the benchmark.
- **Instead:** when all 3 seeds fail on a box, the edit gets a **new box** resting on a ground
  or table surface and sized like the photo's own object of that class, up to 3 new boxes. The
  sham and paste controls move with the C4 box, so they stay matched.
- **Result:** a verified AI object in **38 of 41 C4** edits (93%) and **25 of 34 C4-occ** edits
  (74%); 39 of 41 shams clean. Most of the 19 objects gained on new boxes are clear (dog,
  elephant, train, horse, airplane); 3 are weak and a few still sit in odd places.
- **Remaining:** 3 C4 and 9 C4-occ edits without an object (4 crowded photos had no room for a new
  box); human audit pending. 62 tests pass.

### 9. Another class as a last resort (17 Oct, run `task2_pilot_inphoto`)
- **Change:** if the new boxes also fail, the edit switches to another class already in the
  photo, smallest first (sham and paste switch with it). Photos with no free space at all may
  still get a C4-occ edit (an object partly covering a real one). Week-1 and earlier pilot plans are
  unchanged (checked). 65 tests pass.
- **Result:** **46 of the 50 photos** now have at least one verified AI object (was 39);
  C4 39/41, C4-occ 35/41. The new objects include a toilet, motorcycles, a giraffe, a handbag and a
  spoon; 2 are very small.
- **Still without one (4):** a single train, a single zebra, and two crowded street scenes where
  no box fits for any class.

### 10. The last four photos (17 Oct)
- **Fix:** the C4-occ box was sized like a typical object of the class, far too small to cover a
  train or zebra that fills most of the frame. It is now sized from the object it covers (same
  occlusion rules). The zebra photo gained an AI zebra head; the motorcycle photo a small AI
  motorcycle. 66 tests pass; earlier plans unchanged.
- **Result: 48 of 50 photos** have a verified AI object; C4 39/41, C4-occ 37/42.
- **Left out (2):** an AI train painted over the real train merges with it and cannot be told
  apart, and a car-interior shot has no valid box. Both would need a rule change.

### 11. Decision: a class not in the photo as the very last resort (17 Oct)
- **Rule:** only when a photo has no verified AI object of its own classes, it may get one of
  the 10 original categories that it does not contain. These edits are flagged
  (`class_in_photo: false`) and left out of the same-class comparison. A first version applied
  this too broadly (6 photos that already had an object got an extra one); fixed before use.
- **Result: 49 of 50 photos** have a verified AI object. The train photo got a clock painted
  over the windscreen: clearly there, but implausible, so it is flagged for the audit.
- **Still without one (1):** the car-interior shot, where the class is not the problem: no box
  fits the occlusion rule. 69 tests pass.

### 24. Natural-placement pilot (19 Oct, run `task2_pilot_natural`)
- **Why:** many full-run AI objects are placed unnaturally (a skateboard in the air, a bird in a
  window pane), because most boxes were random free-space spots.
- **New method:** a local VLM (Molmo-7B-D) points at natural spots for a class already in the photo
  (one fixed "Point to …" prompt per class), the box is sized from the photo's own object of that
  class and scaled by depth, and it may cover no person and at most 15% of any real object.
- **Plan:** 33 of the 50 pilot photos got a natural spot; 17 had none that met the rules (mostly
  people or other objects in the way).
- **Automatic realism check did not work:** Molmo answered "not natural" for every edit, including
  clearly fine ones, so it is switched off; edits are judged by eye.
- **Generation (final):** 28 of 33 photos have a verified AI object, 5 dropped; 28 of the 50 pilot
  photos <!-- run:task2_pilot_natural, run_verify_2026-10-19T151526Z -->.
- **Pilot v2 with three fixes** (point on an allowed surface, larger boxes, reject objects filling
  < 35% of their box): 18 of 50 photos with a verified AI edit; on a fitting surface 72%, none on a
  wall, objects median 1.8% of the image (random 0.8%, v1 0.5%); by eye about 14 of 18 natural
  <!-- run:task2_pilot_natural_v2, file:paper/results/task2_pilot_natural_three_way.json -->.
  Better quality, lower coverage; decision needed.
- **Pilot v3** (same rules, 3 classes and 10 spots per class): 19 of 50 photos (v2: 18), same
  quality; more tries per photo do not raise coverage, because people and other objects block the
  spots and some objects still fail verification <!-- run:task2_pilot_natural_v3, file:paper/results/task2_pilot_natural_four_way.json -->.
- **Decision (20 Oct):** the full natural run uses the v2 method with looser rules (up to 30% of a
  person or of a real object covered, object filling at least 25% of its box;
  `configs/insertion_natural.yaml`). It runs planning alone first: the number of photos with a
  spot decides whether the rules are loosened further before any generation. Molmo's answers are
  saved, so trying other rules re-checks the photos in minutes. Planning result: **349 of 544
  photos (64%) have a natural spot** <!-- run:task2_insertion_natural, run_plan_2026-10-20T145159Z, file:paper/results/task2_insertion_natural_plan.json -->; looser overlap limits
  would add at most 50 (40%), 62 (50%) or 78 (60%) photos. The author then raised both overlap
  limits to 50%: **389 of 544 photos (72%)** have a spot <!-- run:task2_insertion_natural, run_plan_2026-10-20T161419Z, file:paper/results/task2_insertion_natural_plan_50pct.json -->.
  Photos with a verified natural AI edit: [[TBD]].
- **Placement replaced (21 Oct): Qwen3-VL-8B draws the box, the author reviews it.** Many Molmo
  boxes did not make sense: the point was fine but the box was upright for objects that should
  lie flat (39 of 73; a tennis racket standing in mid-air). Qwen3-VL-8B now reads the whole
  scene and draws up to three boxes (same rules), with a second question listing taken places,
  every class of the photo, then any listed class. The author accepted or rejected every box by
  eye on a review page. Pilot (50 photos): **26 with an accepted natural box**; more questions
  for the other 24 proposed boxes for 5 [[TBD: author's review]] and none for the 17 far-away,
  close-up or crowded photos <!-- run:task2_pilot_natural_qwen, run_plan_2026-10-20T191136Z, run_plan_2026-10-20T193654Z, file:data/inlayfake_pilot_natural_qwen/spot_review.json -->.
  Qwen3-VL is from the same family as the week-3 origin judge (Qwen2.5-VL-3B): disclosed, and
  week 3 compares that judge on natural vs random placement.
- **Final placement for the full run (21 Oct):** one merged question to Qwen3-VL-8B per class
  (every labelled object and person as taken, size and proportions, order of preference: free
  space, nearer the camera, cut off by the edge), three checks only (short side >= 15 px and
  0.1% of the image, measured in a generation test; not > 3x thinner than the class; <= 30% of
  any real object or person covered, so edits add rather than replace), one re-ask with the
  failures, then the next class; no review gate (author), every C4 audited by hand afterwards.
  Pilot: **44 of 50** photos with a box <!-- run:task2_pilot_natural_qwen_free, run_plan_2026-10-21T091847Z, file:paper/results/task2_pilot_natural_qwen_merged_plan.json -->. A Qwen self-judge did not agree with the
  author's review (AUROC 0.44 / 0.50) and is not used <!-- file:paper/results/task2_judge_agreement.json -->.
  Paper §4 rewritten (Box 1 prompt, Table 1c). Full run (`configs/insertion_natural_box.yaml`): planning gave **518 of 544** photos a box <!-- run:task2_insertion_natural_box, run_plan_2026-10-21T115944Z, file:paper/results/task2_insertion_natural_box_plan.json -->; verified AI edits [[TBD]].
- **Decision (21 Oct):** photos without an accepted natural box get no AI edit (no forced or
  random fallback), so the AI condition covers a subset of the 544 photos; full-run coverage
  [[TBD]].
- **Compared with random placement on the same photos:** objects rest on a fitting surface more
  often (64% vs 46%) and on walls less often (11% vs 20%), but they are smaller (median 0.5% vs
  0.8% of the image) <!-- file:paper/results/task2_pilot_natural_vs_random.json -->. Proposed fixes:
  the point must be on an allowed surface, a minimum box size with gentler depth scaling, and
  rejecting objects that fill too little of their box.
- **By eye:** about half the 25 look natural (backpack by a tent, car in a car park, boats and
  surfboards on water, traffic lights), about half do not (horse or motorcycle on a wall, giraffe
  head in a window, tiny or smudged objects). Causes: the VLM sometimes points at a wall, and SDXL
  often draws the object much smaller than its box. Decision needed on how to continue.

### 23. Insertion run complete: all 544 photos have C0–C4 (18 Oct)
- The full preview (all 544 photos, C0–C4) is in the repo: `docs/results/week2_ai_insertion/index.html`.
- Both weeks' pages are joined in one site with week tabs, `docs/index.html`, ready to be
  published as the repo's GitHub Pages site (not switched on yet; see the publishing checklist).
  The week folders are `docs/results/week1_real_baseline/` and `docs/results/week2_ai_insertion/`,
  and both pages share one design.
- The bed and train close-ups passed only with boxes in the bottom-left corner (author's
  request), at verifier scores right at the threshold (0.20, 0.25). By eye both are doubtful;
  they are flagged for the human audit <!-- run:task2_insertion, file:paper/results/task2_insertion_final.json -->.
- **Final:** 544 of 544 photos complete; AI edits from the free-space rule 464, the occlusion rule
  38, placed anywhere 42; 1,088 edited images.
- Audit sheet rebuilt from the final ledger, then extended on the author's decision: **every C4
  edit (544)**; C2 shams are not audited (author). 52 rows are flagged `in_random_sample`.
  The CSV and the HTML show the same edits in the same order. To be filled by the author.

### 22. Last 2 AI edits and 4 shams re-run (18 Oct)
- The 4 missing C2 shams added nothing; OWLv2 kept seeing the photo's real train, umbrella, bus
  or book next to the box. Sham check now ignores a detection lying ≥ 25% on a real object of
  that class; 3 new seeds each. The bed and train photos get up to 12 more boxes anywhere.
  **Result:** all 4 shams passed on the first new seed, so 542 of 544 photos are complete. The
  bed and train failed boxes 13–18; on the author's request boxes 19–24 are placed anywhere with
  any overlap allowed; boxes 19–21 failed too. On the author's request, boxes 22–27 are placed
  in the bottom-left corner (running since 14:36; result [[TBD]]).

### 21. Merged run done: 537 of 544; re-run of the 7 dropped photos (18 Oct)
- **Result:** 537 photos have a verified AI edit (from C4 464, from C4-occ 38, placed anywhere 35);
  533 have every condition C0–C4 <!-- run:task2_insertion, file:paper/results/task2_insertion_merged_round_final.json -->.
- **7 dropped**, after 6 boxes × 3 seeds. OWLv2 re-run on their 210 failed attempts: in close-ups
  of one big object (bed, train, toilet) the box lands on the real object, which never counts;
  in cluttered scenes small classes (remote, book, knife) are not recognised.
- **Re-run (author):** 6 more boxes, still anywhere, but not on the real object of the same class
  (≤ 30% overlap). **Result: 5 of 7 passed** (toilet, tv, chair, bed, surfboard); the bed and
  train close-ups stay without an AI edit. **Final: 542 of 544 photos have an AI edit, 538 have
  every condition** <!-- run:task2_insertion, file:paper/results/task2_insertion_final.json -->.

### 20. Decision: the two AI placement rules (C4, C4-occ) merged into one condition C4 (18 Oct)
- **Why:** after seven rounds, 44 of 544 photos still had no AI object under the separate rules
  (C4: touch no real object; C4-occ: cover 20–50% of one). 284 photos had both, 175 only C4, 41
  only C4-occ <!-- file:paper/results/task2_insertion_merge_decision.json -->.
- **New rule (author):** one AI condition (C4): an AI object of a class already in the photo,
  **placed anywhere**. Verified edits are kept: the C4 if the photo has one (its C2 and C3
  already exist), else the C4-occ, else a new box anywhere (up to 6 boxes × 3 seeds; OWLv2 must
  still confirm the object). C2 and C3 always sit at the chosen AI box, so 38 photos that use
  their C4-occ get a new C2 and C3 there.
- **Status:** run restarted with the merged rule (Sun 03:46): AI edit from C4 464, from C4-occ 38,
  42 still to place anywhere <!-- run:task2_insertion, run_verify_2026-10-17T194502Z -->.
- **Cost:** H2 loses its controlled occlusion condition: occlusion effects are now measured on
  the AI edits that happen to cover a real object (observational). 298 verified C4-occ edits leave
  the benchmark because their photos use the C4. The paper (§4) states this.

### 19. Round-7 results; verified edits are always kept (18 Oct)
- The fills' first attempts: 35 of 57 new C4 and 34 of 128 new C4-occ edits passed. Totals now C4
  459, C4-occ 325 verified.
- 13 C4 edits with a class not in the photo had stopped counting when the C4-occ fill later gave
  their photos an in-photo C4-occ. On the author's decision, a verified edit is now always kept;
  they are back (with their C2 and C3), with no regeneration.

### 18. Seven dropped C4 edits brought back (18 Oct)
- 7 C4 edits had been dropped because new boxes were sized like the photo's own (large) objects
  and found no free space. New boxes now also try a typical object size; all 7 get a new box,
  with the C4 rules unchanged. The run was restarted once more (Sun 02:22); nothing was lost.

### 17. More photos get a C4 edit (18 Oct)
- 70 photos had no C4 box. As with C4-occ, a check showed the cause was the box search (one size
  tried), not the rules: the 8 px margin barely matters.
- With several box sizes and unchanged rules, **57 more photos get a C4 box (531 of 544)**, each
  with its C2 sham and C3 paste. No other plan changed; the run was restarted again (Sun 01:28),
  adding an estimated 1.5–2 h.

### 16. More photos get a C4-occ edit (18 Oct)
- 137 of the 544 photos had no C4-occ box. A check showed this was mostly the box search (one size
  tried), not the overlap rule. On the author's decision, covered objects may now be as small as
  32 px (was 64), and several box sizes are tried.
- **Result: 128 more photos get a C4-occ box (535 of 544).** No existing plan changed; the run was
  briefly stopped and restarted (Sun 00:40) and continues with these new C4-occ edits, adding an
  estimated 2–4 h.

### 15. Decision: keep the sham edit C2 (17 Oct)
- 18 Oct: the paper (§4) now also explains why C0, C3 and C4 alone would not be enough (C1 is
  the matched-encoding reference; C2 separates AI objects from AI-repainted pixels) and how C3
  is built (a real COCO object of the same class, cut with its mask and pasted, no AI).
- C2 is the "placebo": SDXL repaints the same box with "empty background", so no object is
  added. It is not counted as an AI object; it shows whether a method detects AI objects or just
  any AI-repainted area, and whether repainting alone disturbs nearby real objects.
- The reasons for every condition (C1–C4), and why C4-occ has its own box, are now written into
  the paper (§4, §6, §7.6).
- Some shams come out as flat grey patches; these will be marked in the human audit.

### 14. Decision: SDXL is the only generator (17 Oct)
- The planned SD2 (second training generator) and FLUX / Kandinsky (held-out generator) are
  dropped for lack of time.
- **What is lost:** the test of whether origin detection works on a generator never seen in
  training (part of H4), and generality across editors: results describe SDXL insertions. The
  paper says so in its limitations.
- **What stays:** robustness to JPEG / resize / blur and the held-out Open Images photos (the rest
  of H4), and every other hypothesis.
- Full run status: round 1 done (1,355 images in 6 h 53 min); on the first attempt 269 of 474 C4
  and 123 of 407 C4-occ edits passed and 472 of 474 shams stayed clean; 491 edits are in round 2.

### 13. Full run set up (17 Oct, run `task2_insertion`)
- `configs/insertion_inphoto.yaml`: the pilot's settings on **the same 544 photos as the week-1
  baseline** (author decision, so weeks 1â€“3 use one photo set), in a new folder
  (`data/inlayfake_v2`) so week-1 data is untouched; SDXL only for now.
- All 544 get a plan (70 of them occlusion-only, 1 placed anywhere); 1,355 images in the first
  round (~7.5 h).
- **Expected GPU time ~28 h** (estimate from the pilot's retry rates at ~20 s per image), so
  about two nights; the run can be stopped and resumed.

### 12. Decision: a box anywhere as the very last resort (17 Oct)
- **Rule:** if no box satisfies the placement rules, an occluding (C4-occ) edit may be placed
  anywhere in the image, flagged `placement_rule: anywhere`. C4 always keeps its no-occlusion
  rule. Earlier plans unchanged; 71 tests pass.
- **Result: 50 of 50 photos** have a verified AI object.
- **Caveat:** the car-interior photo's new object is a tiny truck (21Ã—13 px) that the verifier
  accepts but that is hardly visible; it will probably fail the human audit.

## Results

| Item | Value |
|---|---|
| Pilot (46 photos, SDXL, attempt 1): insertion confirmed by verifier | C4 4/46, C4-occ 1/38 (5/84 = 6%) <!-- run:task2_pilot, file:paper/results/task2_pilot_round1_verify.json --> |
| Pilot: sham edits in which the verifier found the target class | 0/46 (one sham produced a non-target object, a bird) <!-- run:task2_pilot --> |
| Surface pilot (35 photos, SDXL, attempt 1): insertion confirmed | C4 0/35, C4-occ 4/19 (4/54 = 7%) <!-- run:task2_pilot_surface, file:paper/results/task2_pilot_surface_round1_verify.json --> |
| Window-size diagnostic (12 failed edits): confirmed at 2.5Ã— / 1.5Ã— / 1.2Ã— | 1 / 5 / 7 of 12 <!-- file:paper/results/task2_window_scale_diagnostic.txt (scratch diagnostic, not a run) --> |
| Same-class plan (50 photos): planned / with occlusion / classes | 41 / 34 / 31 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_plan.json --> |
| Week-1 plan: photos containing any of the 10 fixed classes | 196 of 544 <!-- run:task1_real_baseline plans + COCO val2017 annotations --> |
| Same-class pilot (window 1.2Ã—, looser check, â‰¤ 3 attempts): verified C4 / C4-occ / clean C2 | 27/41 (66%) / 17/34 (50%) / 41/41 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_final_verify.json --> |
| Same-class pilot, attempt 1, old rule: C4 / C4-occ | 10/41 / 7/34 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_round1_oldrule_verify.json --> |
| Same-class pilot with re-plan (â‰¤ 4 boxes Ã— 3 seeds): verified C4 / C4-occ / clean C2 | 38/41 (93%) / 25/34 (74%) / 39/41 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_replan_verify.json --> |
| Plus class switch and C4-occ-only plans: verified C4 / C4-occ; photos with an AI object | 39/41 / 35/41; 46 of 50 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_classswitch_verify.json --> |
| Plus C4-occ boxes sized from the covered object: C4 / C4-occ; photos with an AI object | 39/41 / 37/42; 48 of 50 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_targetsize_verify.json --> |
| Plus a class not in the photo as last resort: C4 / C4-occ; photos with an AI object | 39/41 / 38/42; 49 of 50 (1 out-of-photo edit) <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_outside_verify.json --> |
| Plus a C4-occ box anywhere as very last resort: C4 / C4-occ; photos with an AI object | 39/41 / 39/43; **50 of 50** (1 doubtful) <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_anywhere_verify.json --> |
| Full run plan (week-1 photo set): photos planned / first-round images / expected GPU time | 544 of 544 / 1,355 / ~28 h (estimate) <!-- run:task2_insertion (plan only), file:paper/results/task2_insertion_plan.json --> |
| Full run: C2 / C3 / C4 / C4-occ images in the final ledger | [[TBD]] |
| Merge decision (18 Oct): photos with C4 and C4-occ / only C4 / only C4-occ / neither (after round 7) | 284 / 175 / 41 / 44 <!-- file:paper/results/task2_insertion_merge_decision.json --> |
| Merged AI condition at restart 5: AI edit from C4 / from C4-occ / still to place anywhere | 464 / 38 / 42 <!-- run:task2_insertion, run_verify_2026-10-17T194502Z, file:paper/results/task2_insertion_merge_decision.json --> |
| Full run, round 1 (attempt 1): passed C2 / C4 / C4-occ | 472/474 / 269/474 / 123/407; 491 to round 2 <!-- run:task2_insertion, file:paper/results/task2_insertion_round1_verify.json (interim) --> |
| Merged run (boxes 1–6): photos with a verified AI edit / all of C0–C4 / dropped | 537 / 533 / 7 <!-- run:task2_insertion, file:paper/results/task2_insertion_merged_round_final.json --> |
| After the re-run of the 7 (boxes 7–12): photos with an AI edit / all of C0–C4 / dropped | 542 / 538 / 2 <!-- run:task2_insertion (11:17), file:paper/results/task2_insertion_merged_round_final.json and log --> |
| **Final (all re-runs): photos with all of C0–C4 / AI edits free-space / occluding / anywhere / edited images** | **544 / 464 / 38 / 42 / 1,088** <!-- run:task2_insertion, file:paper/results/task2_insertion_final.json --> |
| Human audit (all 544 C4 edits; 52 of them a random subset) | [[TBD]] |
| Real objects re-labelled `mixed_modified` | [[TBD]] |
| Human audit (10%): object present / realistic / other change | [[TBD]] |
| GPU time per image on the RTX 4070 laptop | SDXL 21.6 s (pilot, 122 images in 43:57) <!-- run:task2_pilot, file:paper/results/task2_pilot_round1_generate.json -->; full run round 1: 1,355 images in 6 h 53 min (18.3 s each) <!-- run:task2_insertion, insert_progress.log --> |

## Next steps

Done this week: SDXL pilots and the design changes they led to; verification with OWLv2 + SAM;
the full run (all 544 photos have C0–C4); the shared preview and the results website.

1. **Human audit of every C4 edit** (544 rows, `audit_edits.csv`; 0 filled so far), by the author.
2. Remove the C4 edits the audit rejects, finalise the ledger, and rebuild the shared preview.
3. Paper: audit results (§4), Table 1 counts (images and instances per origin and condition,
   `mixed_modified` count) and Figure 1 (example photo with C1–C4).
4. Merge `week-2-ai-insertion` into `main` (a pull request), after the audit.
5. Optional, for week 4: start the Open Images download for the held-out domain.

## To discuss

- **Merged AI condition (18 Oct).** Is losing the controlled occlusion condition acceptable for H2? The plan is to evaluate H2 on real objects the AI object does not cover and to report covered objects separately by visible fraction.
- **Positioning against related work (search of 17 Oct, `docs/literature/related_work_2026-10-17.md`;
  venues from search snippets, to verify).** Closest: COinCO (CVPR 2026: COCO, one object
  *replaced* by SD inpainting, a detector used only as a filter). Gaps no work found covers: origin
  of every object with a real-object false-accusation rate; a real-paste control (C3) and occlusion
  arm (C4-occ); same-class pairs in one photo; frozen detectors on inserted vs real objects; object-level
  abstention. With SDXL only, a held-out generator is no longer one of our edges.
- **Held-out generator: decided 17 Oct, dropped** (SDXL only, Done Â§14). The laptop has 32 GB of
  RAM (`systeminfo`: 32,471 MB), at the FLUX threshold; this no longer matters for this paper.
- **Is one generator enough for CVPR?** Reviewers may ask whether results hold for other
  editors; the paper states this as a limitation.
- **How to fix the insertion rate.** Surface placement was tried and did not help (7%). The
  evidence points to a tighter edit window (1.2â€“1.5Ã— instead of 2.5Ã—; 7/12 at 1.2Ã—). To decide:
  which window size, whether to keep surface placement (it costs 11 of 46 plans and half of C4-occ),
  then a third pilot before the full run.
- Whether OWLv2's misses need a closer look (it rejected some clearly visible objects).
- Same-class insertion: combine with surface placement, and size the box like the photo's own
  instance of that class?
- Whether the sham check should look for any new object, not only the target class.
