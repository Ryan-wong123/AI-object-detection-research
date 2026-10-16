# InlayFake: Week 2 progress (15–21 Oct 2026)

Last updated 16 Oct 2026 · Ryan Wong

Briefing for the project professor. Branch `week-2-ai-insertion`. Generation runs on the RTX 4070
laptop. The SDXL pilot (run `task2_pilot`, `data/inlayfake_pilot`) ran on 16 Oct; full-run
numbers (run `task2_insertion`) are still `[[TBD]]`.

**Goal for the week:** turn the 544 planned real photos into the benchmark by adding AI-inserted
objects under matched controls, and check that each insertion really contains the requested object.

## How an insertion works

For each source photo, week 1 already fixed one plan: a target category (10 classes, balanced), a
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

## Results

| Item | Value |
|---|---|
| Pilot (46 photos, SDXL, attempt 1): insertion confirmed by verifier | C4 4/46, C5 1/38 (5/84 = 6%) <!-- run:task2_pilot, file:paper/results/task2_pilot_round1_verify.json --> |
| Pilot: sham edits in which the verifier found the target class | 0/46 (one sham produced a non-target object, a bird) <!-- run:task2_pilot --> |
| Full run: C2 / C3 / C4 / C5 images in the final ledger | [[TBD]] |
| Full run: C4 success rate per generator (SDXL, SD2, FLUX) | [[TBD]] |
| Real objects re-labelled `mixed_modified` | [[TBD]] |
| Human audit (10%): object present / realistic / other change | [[TBD]] |
| GPU time per image on the RTX 4070 laptop | SDXL 21.6 s (pilot, 122 images in 43:57) <!-- run:task2_pilot, file:paper/results/task2_pilot_round1_generate.json -->; SD2, FLUX [[TBD]] |

## Next steps

1. Decide how to change the insertion design (see "To discuss"), then re-run the 50-photo pilot.
2. Only when the pilot reaches a usable success rate, start the full run.
3. Full run overnight: `python -m inlayfake insert --config configs/insertion.yaml`.
4. Human audit of 10% of edits, then record the success rates here, in the paper and in the log.
5. Start the Open Images download for the week-4 held-out domain.

## To discuss

- **Held-out generator.** FLUX.1-Fill on an 8 GB GPU needs about 32 GB of system RAM for offloading;
  the laptop's RAM decides FLUX or the Kandinsky 2.2 fallback. The laptop has **32 GB** of RAM
  (`systeminfo`: 32,471 MB, 16 Oct; now `hardware.ram_gb` in `configs/default.yaml`). That is at
  the threshold, not above it: FLUX with sequential offload should fit but may page; a short FLUX
  smoke test (or the GGUF-quantised transformer) decides before the full G3 run.
- **How to fix the 6% insertion rate.** Options: place boxes on supporting surfaces (floor,
  table, ground) instead of any free space; a larger minimum box for small classes (cup, bottle);
  a prompt that mentions the scene; or a generator that is better at adding objects. More
  attempts alone will not reach the 85% target.
- Whether the sham check should look for any new object, not only the target class.
