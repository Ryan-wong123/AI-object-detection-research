# InlayFake: Week 2 progress (15–21 Oct 2026)

> Terminology: C4-occ = the occlusion placement rule of the AI insertion before 18 Oct 2026 (cover 20–50% of one real object); since then it is part of the single AI condition C4.

18 Oct 2026 · Ryan Wong

Week 2 turns the 544 planned real photos into the benchmark by inserting AI objects under matched
controls (sham edit, real-object paste, insertion with and without occlusion), and verifies every
insertion. The 50-photo SDXL pilot ran on 16 Oct (run `task2_pilot`): the pipeline works end to end,
but SDXL inserted the object in only 6% of edits. A second pilot with boxes on supporting
surfaces (run `task2_pilot_surface`) did no better (7%); a diagnostic points to the edit window
instead (tighter window: 7/12 inserted). The full run waits for that change. Design change:
the AI object is now of a class already in the photo (e.g. a sheep among sheep, not a chair).
With a tighter window (1.2×) and a looser verifier check, the same-class pilot has a verified
AI object in 66% of C4 and 50% of C4-occ edits (was 6%). Giving failed edits a new box on a
surface raised this to **93% of C4 and 74% of C4-occ**; switching to another class in the photo as
a last resort, sizing occluding boxes from the object they cover, and finally allowing a class
not in the photo (flagged) and an occluding box anywhere (flagged) gives **50 of 50 photos** at
least one verified AI object (one of them a doubtful tiny object). The full run on the same
544 photos as the week-1 baseline is set up (all planned; ~28 h of GPU time expected, resumable). Edits without a verified
object are dropped, never labelled.

## Done this week

| Area | What was delivered |
| --- | --- |
| Insertion | Local inpainting: square window of 2.5× the box, generator at native size, composite only inside the mask; pixels outside are identical to C1. Same JPEG settings as C1. |
| Conditions | C2 sham (SDXL), C3 real paste, C4 AI insert (SDXL; SD2 and FLUX dropped 17 Oct). Since 18 Oct C4 and C4-occ are **one AI condition**: a class already in the photo, placed anywhere; C2/C3 at the same box. |
| Verification | OWLv2 (not an evaluated detector) must find the object (IoU ≥ 0.3, score ≥ 0.2); sham edits must contain no object; failed edits are regenerated with a new seed, up to 3 attempts. |
| Annotation | SAM outline of each inserted object; actual changed region; visible fraction of occluded real objects; real objects altered by the editor marked `mixed_modified`. |
| Audit | Random 10% of edits on a before/after page with a y/n sheet. |
| Viewer | Shows C2–C4, inserted objects in purple, altered real objects in amber. |
| Tests | 48 CPU tests pass, including a forced verifier failure that is retried with a new seed. |
| Re-plan on failure | After 3 failed seeds an edit gets a new box on a surface, sized like the photo's own object (sham and paste move with it); up to 3 new boxes. 62 tests pass. |
| Verifier and window | Window 1.2× (was 2.5×); an object mostly inside the box passes even if smaller than it; preview labels each attempt "in benchmark" or "rejected". 59 tests pass. |
| Same-class insert | The inserted class is one already in the photo's audited labels (any class except person); the old 10-class cycle reproduces week 1 exactly. 57 tests pass. |
| Surface placement | New `placement.support`: boxes rest on floor/ground/table from COCO-Stuff labels (off by default); 53 tests pass. |
| Pilot | 46 planned photos, 130 SDXL edits + 46 pastes. Failed edits contain background, not a missed object (checked by eye); no code bug found. Failed boxes are in implausible places (sky, mid-road). Retries stopped after round 1. |

## Results

| Item | Value |
| --- | --- |
| Pilot insertion success (46 photos, SDXL, attempt 1) | C4 4/46, C4-occ 1/38 (6%) <!-- run:task2_pilot --> |
| Sham edits in which the verifier found the target class | 0/46 (one produced a bird) <!-- run:task2_pilot --> |
| SDXL time per image | 21.6 s <!-- run:task2_pilot --> |
| Surface pilot (35 photos, attempt 1) | C4 0/35, C4-occ 4/19 (7%) <!-- run:task2_pilot_surface --> |
| Window 2.5× / 1.5× / 1.2× (12 failed edits, diagnostic) | 1 / 5 / 7 inserted <!-- file:paper/results/task2_window_scale_diagnostic.txt --> |
| Same-class plan (50 photos): planned / with occlusion / classes | 41 / 34 / 31 <!-- run:task2_pilot_inphoto --> |
| Same-class pilot, verified C4 / C4-occ / clean C2 (≤ 3 attempts) | 27/41 / 17/34 / 41/41 <!-- run:task2_pilot_inphoto --> |
| With re-plan on failure (≤ 4 boxes) | 38/41 / 25/34 / 39/41 <!-- run:task2_pilot_inphoto --> |
| Plus class switch: C4 / C4-occ; photos with an AI object | 39/41 / 35/41; 46 of 50 <!-- run:task2_pilot_inphoto --> |
| Plus C4-occ boxes sized from the covered object | 39/41 / 37/42; 48 of 50 <!-- run:task2_pilot_inphoto --> |
| Plus a class not in the photo (last resort, flagged) | 39/41 / 38/42; 49 of 50 <!-- run:task2_pilot_inphoto --> |
| Plus a C4-occ box anywhere (very last resort, flagged) | 39/41 / 39/43; 50 of 50 <!-- run:task2_pilot_inphoto --> |
| Final images per condition (C2 / C3 / C4 / C4-occ) | [[TBD]] |
| Full run round 1: passed C2 / C4 / C4-occ | 472/474 / 269/474 / 123/407; 491 to round 2 <!-- run:task2_insertion (interim) --> |
| Merge (18 Oct): photos with C4+C4-occ / only C4 / only C4-occ / neither | 284 / 175 / 41 / 44 <!-- file:paper/results/task2_insertion_merge_decision.json --> |
| Merged AI condition at restart: from C4 / from C4-occ / to place anywhere | 464 / 38 / 42 <!-- run:task2_insertion, run_verify_2026-10-17T194502Z --> |
| Merged run: photos with an AI edit / all of C0–C4 / dropped | 537 / 533 / 7 <!-- run:task2_insertion, file:paper/results/task2_insertion_merged_round_final.json --> |
| After the re-run of the 7 dropped: AI edit / all of C0–C4 / dropped | 542 / 538 / 2 <!-- run:task2_insertion, file:paper/results/task2_insertion_final.json --> |
| Human audit: present / realistic / other change | [[TBD]] |

## Next week and points to discuss

**Rest of week 2:** change the edit window and run a third pilot, then the overnight `insert`
run, the 10% human audit and the Open Images download.

**To discuss:**

- Related work (17 Oct search, to verify): COinCO (CVPR 2026) is closest (COCO, object
  *replacement*). Our edge: every object's origin + false accusations of real objects, real-paste
  and occlusion controls, same-class pairs, frozen detectors on inserted vs real objects, abstention.
- Edit window 1.2–1.5× instead of 2.5×, and whether to keep surface placement (costs 11 of 46
  plans and half of C4-occ). Then a third pilot.
- OWLv2 rejected some clearly visible objects: does the verifier need a closer look?
- Same-class insertion leaves classes unbalanced and some boxes in the sky: add surface placement
  and size boxes like the photo's own instance?
- Whether the sham check should look for any new object, not only the target class.
- **Decided 18 Oct: C4 and C4-occ merged** into one AI condition (a class in the photo, anywhere) because 44 photos still had no AI object. Verified edits kept (C4 preferred, else C4-occ, else a new box anywhere); C2/C3 at the chosen box. Cost: H2 loses its controlled occlusion condition (covered objects analysed observationally). Acceptable?
- Decided 18 Oct: smaller covered objects (32 px) and more box sizes for C4-occ, so 535 of 544
  photos now have a C4-occ box (was 407); more box sizes for C4 (rules unchanged), so 531 of 544
  have a C4 box (was 474).
- Decided 17 Oct: keep the sham edit C2 as a control (never scored as an AI object; shows
  whether methods detect AI objects or just AI-repainted areas); rationale now in the paper.
- Scope decided 17 Oct: SDXL is the only generator (SD2 and the held-out FLUX dropped for time).
  The held-out-generator test of H4 is lost; post-processing and Open Images remain. Is one
  generator acceptable for the paper?
