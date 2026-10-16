# InlayFake: Week 2 progress (15–21 Oct 2026)

Last updated 17 Oct 2026 · Ryan Wong

Briefing for the project professor. Branch `week-2-ai-insertion`. Generation runs on the RTX 4070
laptop. The SDXL pilot (run `task2_pilot`, `data/inlayfake_pilot`) ran on 16 Oct; full-run
numbers (run `task2_insertion`) are still `[[TBD]]`.

**Goal for the week:** turn the 544 planned real photos into the benchmark by adding AI-inserted
objects under matched controls, and check that each insertion really contains the requested object.

## How an insertion works

(Since 16 Oct the inserted class is one already in the photo; see Done §6.) For each source
photo, week 1 already fixed one plan: a target category (10 classes, balanced), a
box size drawn from real COCO instances of that class, a free-space box and, for 363 photos, a box
covering 20–50% of one real object.

1. Crop a square window of 2.5× the box around it, so the generator sees the scene context.
2. Resize the window to the generator's native size (1024 for SDXL and FLUX, 512 for SD2).
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
| C4 AI insert | AI object in free space | SDXL, SD2 (seen); FLUX.1-Fill (held out, test split only) |
| C5 AI insert + occlusion | AI object covering 20–50% of one real object | SDXL |

## Done

### 1. Verification stage (`verify`): new code
- **Does the object exist?** OWLv2 looks for the target class in the crop window. OWLv2 is an
  independent open-vocabulary detector, deliberately *not* one of the four evaluated detectors, so
  the check does not favour them. An edit passes when OWLv2 finds the class with IoU ≥ 0.3 to the
  planned box and score ≥ 0.2.
- **Sham check.** A C2 edit passes only if OWLv2 does *not* find the class there.
- **Retries.** A failed edit is regenerated with a new seed, up to 3 attempts, then dropped and
  logged. `insert` runs generate → verify in rounds until nothing is left to retry.
- **Exact object outline.** SAM, prompted with the OWLv2 box and clipped to the edit mask, gives the
  inserted object's mask and tight box for the ground truth.
- **What actually changed.** |edited − C1| > 4 grey levels (after a 3×3 opening) is saved per image.
- **Occlusion (C5).** Real objects covered by the insertion get their visible mask, box and visible
  fraction updated; below 10% visible they become ignore regions.
- **Altered real objects.** A real object whose area overlaps the changed region by > 10% is
  re-labelled `mixed_modified` and reported separately.
- **Human audit.** A random 10% of passed edits go on `audit_edits.html` (before/after crops) with
  `audit_edits.csv` to mark y/n: object present, realistic, anything else changed.
- **Final ledger.** `ledger.jsonl` keeps C0/C1 exactly as in week 1 (test-checked, so week-1
  results are unchanged), adds C3 with its pasted mask, and one passed record per C2/C4/C5 job.

### 2. Viewer
The detector viewer now lists C2–C5 per generator, outlines inserted/pasted objects (purple) and
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
  edits and 1 of 38 C5 edits. A visual check agrees: in the failed edits SDXL filled the box with
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
- **Result: no better.** C4 0/35, C5 4/19 (7%). SDXL again filled boxes with background, now
  in plausible places (a bench box on a lawn became lawn). Placement was not the main cause.
- **The cause is the edit window.** The generator sees a square of 2.5× the box, so the object is
  only 40% of its canvas side. On 12 failed edits, a tighter window gave many more insertions:
  window 2.5× → 1/12, 1.5× → 5/12, 1.2× → 7/12, with natural colours (a diagnostic, not a run).
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
- **Changes:** generator window 1.2× the box (was 2.5×); the verifier now also accepts an object
  that lies mostly inside the box even if it is much smaller than the box. Why: 14 of 58
  rejections had a confident detection inside the box (e.g. a frisbee at score 0.74) that failed
  only the overlap rule. A detection on the photo's own same-class object never counts.
- **Result:** a verified AI object in **27 of 41 C4** edits (66%) and **17 of 34 C5** edits (50%),
  up from 6% in the first pilot; all 41 sham edits stay clean. 59 tests pass.
- **What is left:** below the 85% target; several objects float in implausible places (a bus
  in the sky, a car in the trees), and very large classes (train, bed) usually fail.

### 8. Re-plan on failure (16 Oct, run `task2_pilot_inphoto`)
- **Why not loosen the check further:** the 31 dropped edits were checked by eye and contain
  only background, so accepting them would put empty "AI object" labels into the benchmark.
- **Instead:** when all 3 seeds fail on a box, the edit gets a **new box** resting on a ground
  or table surface and sized like the photo's own object of that class, up to 3 new boxes. The
  sham and paste controls move with the C4 box, so they stay matched.
- **Result:** a verified AI object in **38 of 41 C4** edits (93%) and **25 of 34 C5** edits
  (74%); 39 of 41 shams clean. Most of the 19 objects gained on new boxes are clear (dog,
  elephant, train, horse, airplane); 3 are weak and a few still sit in odd places.
- **Remaining:** 3 C4 and 9 C5 edits without an object (4 crowded photos had no room for a new
  box); human audit pending. 62 tests pass.

### 9. Another class as a last resort (17 Oct, run `task2_pilot_inphoto`)
- **Change:** if the new boxes also fail, the edit switches to another class already in the
  photo, smallest first (sham and paste switch with it). Photos with no free space at all may
  still get a C5 edit (an object partly covering a real one). Week-1 and earlier pilot plans are
  unchanged (checked). 65 tests pass.
- **Result:** **46 of the 50 photos** now have at least one verified AI object (was 39);
  C4 39/41, C5 35/41. The new objects include a toilet, motorcycles, a giraffe, a handbag and a
  spoon; 2 are very small.
- **Still without one (4):** a single train, a single zebra, and two crowded street scenes where
  no box fits for any class.

### 10. The last four photos (17 Oct)
- **Fix:** the C5 box was sized like a typical object of the class, far too small to cover a
  train or zebra that fills most of the frame. It is now sized from the object it covers (same
  occlusion rules). The zebra photo gained an AI zebra head; the motorcycle photo a small AI
  motorcycle. 66 tests pass; earlier plans unchanged.
- **Result: 48 of 50 photos** have a verified AI object; C4 39/41, C5 37/42.
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

### 12. Decision: a box anywhere as the very last resort (17 Oct)
- **Rule:** if no box satisfies the placement rules, an occluding (C5) edit may be placed
  anywhere in the image, flagged `placement_rule: anywhere`. C4 always keeps its no-occlusion
  rule. Earlier plans unchanged; 71 tests pass.
- **Result: 50 of 50 photos** have a verified AI object.
- **Caveat:** the car-interior photo's new object is a tiny truck (21×13 px) that the verifier
  accepts but that is hardly visible; it will probably fail the human audit.

## Results

| Item | Value |
|---|---|
| Pilot (46 photos, SDXL, attempt 1): insertion confirmed by verifier | C4 4/46, C5 1/38 (5/84 = 6%) <!-- run:task2_pilot, file:paper/results/task2_pilot_round1_verify.json --> |
| Pilot: sham edits in which the verifier found the target class | 0/46 (one sham produced a non-target object, a bird) <!-- run:task2_pilot --> |
| Surface pilot (35 photos, SDXL, attempt 1): insertion confirmed | C4 0/35, C5 4/19 (4/54 = 7%) <!-- run:task2_pilot_surface, file:paper/results/task2_pilot_surface_round1_verify.json --> |
| Window-size diagnostic (12 failed edits): confirmed at 2.5× / 1.5× / 1.2× | 1 / 5 / 7 of 12 <!-- file:paper/results/task2_window_scale_diagnostic.txt (scratch diagnostic, not a run) --> |
| Same-class plan (50 photos): planned / with occlusion / classes | 41 / 34 / 31 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_plan.json --> |
| Week-1 plan: photos containing any of the 10 fixed classes | 196 of 544 <!-- run:task1_real_baseline plans + COCO val2017 annotations --> |
| Same-class pilot (window 1.2×, looser check, ≤ 3 attempts): verified C4 / C5 / clean C2 | 27/41 (66%) / 17/34 (50%) / 41/41 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_final_verify.json --> |
| Same-class pilot, attempt 1, old rule: C4 / C5 | 10/41 / 7/34 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_round1_oldrule_verify.json --> |
| Same-class pilot with re-plan (≤ 4 boxes × 3 seeds): verified C4 / C5 / clean C2 | 38/41 (93%) / 25/34 (74%) / 39/41 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_replan_verify.json --> |
| Plus class switch and C5-only plans: verified C4 / C5; photos with an AI object | 39/41 / 35/41; 46 of 50 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_classswitch_verify.json --> |
| Plus C5 boxes sized from the covered object: C4 / C5; photos with an AI object | 39/41 / 37/42; 48 of 50 <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_targetsize_verify.json --> |
| Plus a class not in the photo as last resort: C4 / C5; photos with an AI object | 39/41 / 38/42; 49 of 50 (1 out-of-photo edit) <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_outside_verify.json --> |
| Plus a C5 box anywhere as very last resort: C4 / C5; photos with an AI object | 39/41 / 39/43; **50 of 50** (1 doubtful) <!-- run:task2_pilot_inphoto, file:paper/results/task2_pilot_inphoto_anywhere_verify.json --> |
| Full run: C2 / C3 / C4 / C5 images in the final ledger | [[TBD]] |
| Full run: C4 success rate per generator (SDXL, SD2, FLUX) | [[TBD]] |
| Real objects re-labelled `mixed_modified` | [[TBD]] |
| Human audit (10%): object present / realistic / other change | [[TBD]] |
| GPU time per image on the RTX 4070 laptop | SDXL 21.6 s (pilot, 122 images in 43:57) <!-- run:task2_pilot, file:paper/results/task2_pilot_round1_generate.json -->; SD2, FLUX [[TBD]] |

## Next steps

1. Author's visual check of the same-class pilot and its 12-row audit sheet.
2. Re-plan the full 600 sources with the same-class rule (the week-1 plan used fixed classes).
3. Only when the pilot reaches a usable success rate, start the full run.
4. Full run overnight: `python -m inlayfake insert --config configs/insertion.yaml`.
5. Human audit of 10% of edits, then record the success rates here, in the paper and in the log.
6. Start the Open Images download for the week-4 held-out domain.

## To discuss

- **Held-out generator.** FLUX.1-Fill on an 8 GB GPU needs about 32 GB of system RAM for offloading;
  the laptop's RAM decides FLUX or the Kandinsky 2.2 fallback. The laptop has **32 GB** of RAM
  (`systeminfo`: 32,471 MB, 16 Oct; now `hardware.ram_gb` in `configs/default.yaml`). That is at
  the threshold, not above it: FLUX with sequential offload should fit but may page; a short FLUX
  smoke test (or the GGUF-quantised transformer) decides before the full G3 run.
- **How to fix the insertion rate.** Surface placement was tried and did not help (7%). The
  evidence points to a tighter edit window (1.2–1.5× instead of 2.5×; 7/12 at 1.2×). To decide:
  which window size, whether to keep surface placement (it costs 11 of 46 plans and half of C5),
  then a third pilot before the full run.
- Whether OWLv2's misses need a closer look (it rejected some clearly visible objects).
- Same-class insertion: combine with surface placement, and size the box like the photo's own
  instance of that class?
- Whether the sham check should look for any new object, not only the target class.
