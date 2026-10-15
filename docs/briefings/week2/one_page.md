# InlayFake: Week 2 progress (15–21 Oct 2026)

15 Oct 2026 · Ryan Wong

Week 2 turns the 544 planned real photos into the benchmark by inserting AI objects under matched
controls (sham edit, real-object paste, insertion with and without occlusion), and verifies every
insertion. The code is written and tested on CPU; generation on the RTX 4070 laptop has not started,
so all results are `[[TBD]]`.

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

## Results

| Item | Value |
| --- | --- |
| Pilot insertion success (50 photos, SDXL) | [[TBD]] |
| Sham edits that produced an object | [[TBD]] |
| Final images per condition (C2 / C3 / C4 / C5) | [[TBD]] |
| C4 success per generator | [[TBD]] |
| Human audit: present / realistic / other change | [[TBD]] |

## Next week and points to discuss

**Rest of week 2:** pilot with a visual check, then the overnight `insert` run, the 10% human audit
and the Open Images download.

**To discuss:**

- FLUX.1-Fill needs about 32 GB of system RAM on an 8 GB GPU; otherwise Kandinsky 2.2 is the held-out
  generator. [[TBD: laptop RAM]]
- Whether 3 attempts per edit is enough.
