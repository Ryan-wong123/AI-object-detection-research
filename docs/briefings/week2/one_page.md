# InlayFake: Week 2 progress (15–21 Oct 2026)

16 Oct 2026 · Ryan Wong

Week 2 turns the 544 planned real photos into the benchmark by inserting AI objects under matched
controls (sham edit, real-object paste, insertion with and without occlusion), and verifies every
insertion. The 50-photo SDXL pilot ran on 16 Oct (run `task2_pilot`): the pipeline works end to end,
but SDXL inserted the object in only 6% of edits. A second pilot with boxes on supporting
surfaces (run `task2_pilot_surface`) did no better (7%); a diagnostic points to the edit window
instead (tighter window: 7/12 inserted). The full run waits for that change.

## Done this week

| Area | What was delivered |
| --- | --- |
| Insertion | Local inpainting: square window of 2.5× the box, generator at native size, composite only inside the mask; pixels outside are identical to C1. Same JPEG settings as C1. |
| Conditions | C2 sham (SDXL), C3 real paste, C4 AI insert (SDXL, SD2; FLUX held out on test only), C5 AI insert + occlusion (SDXL). |
| Verification | OWLv2 (not an evaluated detector) must find the object (IoU ≥ 0.3, score ≥ 0.2); sham edits must contain no object; failed edits are regenerated with a new seed, up to 3 attempts. |
| Annotation | SAM outline of each inserted object; actual changed region; visible fraction of occluded real objects; real objects altered by the editor marked `mixed_modified`. |
| Audit | Random 10% of edits on a before/after page with a y/n sheet. |
| Viewer | Shows C2–C5, inserted objects in purple, altered real objects in amber. |
| Tests | 48 CPU tests pass, including a forced verifier failure that is retried with a new seed. |
| Surface placement | New `placement.support`: boxes rest on floor/ground/table from COCO-Stuff labels (off by default); 53 tests pass. |
| Pilot | 46 planned photos, 130 SDXL edits + 46 pastes. Failed edits contain background, not a missed object (checked by eye); no code bug found. Failed boxes are in implausible places (sky, mid-road). Retries stopped after round 1. |

## Results

| Item | Value |
| --- | --- |
| Pilot insertion success (46 photos, SDXL, attempt 1) | C4 4/46, C5 1/38 (6%) <!-- run:task2_pilot --> |
| Sham edits in which the verifier found the target class | 0/46 (one produced a bird) <!-- run:task2_pilot --> |
| SDXL time per image | 21.6 s <!-- run:task2_pilot --> |
| Surface pilot (35 photos, attempt 1) | C4 0/35, C5 4/19 (7%) <!-- run:task2_pilot_surface --> |
| Window 2.5× / 1.5× / 1.2× (12 failed edits, diagnostic) | 1 / 5 / 7 inserted <!-- file:paper/results/task2_window_scale_diagnostic.txt --> |
| Final images per condition (C2 / C3 / C4 / C5) | [[TBD]] |
| C4 success per generator | [[TBD]] |
| Human audit: present / realistic / other change | [[TBD]] |

## Next week and points to discuss

**Rest of week 2:** change the edit window and run a third pilot, then the overnight `insert`
run, the 10% human audit and the Open Images download.

**To discuss:**

- Edit window 1.2–1.5× instead of 2.5×, and whether to keep surface placement (costs 11 of 46
  plans and half of C5). Then a third pilot.
- OWLv2 rejected some clearly visible objects: does the verifier need a closer look?
- Whether the sham check should look for any new object, not only the target class.
- FLUX.1-Fill needs about 32 GB of system RAM on an 8 GB GPU; otherwise Kandinsky 2.2 is the held-out
  generator. The laptop has 32 GB (at the threshold): a short FLUX smoke test decides.
