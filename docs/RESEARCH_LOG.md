# Research log

Dated record of every experiment, decision and finding. Newest entries at the top.
Format: date · run ID (if any) · what · key numbers · interpretation · next step.

> Terminology: C4-occ = the occlusion placement rule of the AI insertion before 18 Oct 2026 (cover 20–50% of one real object); since then it is part of the single AI condition C4.

---

## 2026-10-20: Full natural run, planning only: 349 of 544 photos have a natural spot

- **Run** `task2_insertion_natural`, `run_plan_2026-10-20T145159Z` (v4 rules: up to 30% of a
  person or real object covered, box fill >= 25% at verification);
  `paper/results/task2_insertion_natural_plan.json`.
- **Result:** **349 of 544 photos (64%)** have at least one valid natural spot; 195 have none
  (every photo has an insertable class). 993 valid of 2,363 candidate spots; 75 classes planned
  (most: surfboard 18, toilet 12, horse 10). Main rejections: 85 outside the image, 80 covering a
  whole person, 39 answers without a point, and many points on surfaces the class does not rest on
  (metal, buildings, fences, walls, trees, ...).
- **Why photos have no spot** (kinds of rejection among each photo's candidates): only wrong
  surface 53, real-object overlap 24 (+28 with surface), people 12 (+18 with surface, +11 with
  real), no point in Molmo's answer 21.
- **Upper bound for looser overlap limits** (photos with a candidate rejected only for covering
  a person/real object by at most the limit; other checks may still reject them): 40%: +50
  photos, 50%: +62, 60%: +78. So even 60% would leave about 120 photos without a spot.
- **Process:** speed fixes during the run (Molmo memory, CPU power mode, ~12 s per photo at the
  end); the last restart re-checked all saved photos once from saved answers (124 Molmo
  questions asked, 736 reused). Photos 1–28 are now planned with the depth model on the CPU, as
  all others are.
- **Next:** author decides: keep the v4 rules (349 photos) and start `insert`, or loosen
  overlap limits and re-plan (minutes).

---

## 2026-10-20: Planning first; saved VLM answers make rule changes cheap

- **Author decision:** run the planning stage of the full natural run alone first, look at how
  many of the 544 photos get a spot, then decide whether to loosen the rules before generating.
- **Why a code change:** re-planning with other rules used to ask Molmo every question again
  (~2.5–4.5 h), although Molmo's points do not depend on the rules. Now each answer is saved per
  (photo, class, prompt, model) in `natural_points.jsonl`, and every planned photo stores the
  settings it was planned with (rules, size limits, surface check, prompt file, depth model). A
  photo saved with other settings is re-checked with the current ones from the saved answers
  (depth model + box rules only); Molmo is asked only for a class it never answered. The 19
  photos planned before the stop are re-checked once this way (their answers are read back from
  the saved progress rows). Tests: a re-check gives the same plans as a fresh run with the new
  rules and asks Molmo nothing. No results change.
- **Next:** planning-only run of `configs/insertion_natural.yaml`; report photos with a spot and
  the rejection reasons; author decides on the rules; then `insert`.
- **Slow planning fixed:** the planning-only run (started ~15:00) ran at ~80 s per photo (~11 h
  for 544; before the laptop slept that morning it ran at ~16 s per photo). The planning process
  held 7.8 GB of the GPU's 8 GB plus 1.2 GB in shared system memory, and the GPU was mostly idle
  (6–24% use): Molmo had spilled out of GPU memory. Fix (author's go-ahead): the depth model
  (Depth-Anything-V2-Small) runs on the CPU (`placement.natural.depth.device: cpu`), and cached
  GPU memory is freed after every photo. Stopped at 28 photos and restarted; nothing was lost.
  Depth on CPU vs GPU differs only by rounding, so already saved photos are kept.
- **That was not enough (still 50–120 s per Molmo answer); measured cause:** Molmo's own
  `Embedding.forward` concatenates its 152k-word table with the image-token table on every call,
  a fresh 1.1 GB GPU block per generated token. PyTorch then kept 8.4 GB reserved for 7.5 GB in
  use, more than the GPU has. Fixes: look the two tables up separately (same vectors), and use
  CUDA's own allocator (`backend:cudaMallocAsync`) for planning. On 6 real planning questions:
  reserved 8.4 → 6.9 GB, peak use 7.5 → 6.8 GB, and **all 6 answers identical** to the saved
  ones. Time per answer 50–120 s → 33–45 s.
- **Remaining slowness is the CPU, not the GPU:** during Molmo the GPU is ~70% idle at ~600 MHz
  and the CPU runs at 60–84% of its base clock (Windows "Balanced" power plan, laptop power
  mode). Molmo generates one token at a time, each driven by Python on the CPU. A
  high-performance power mode is the author's setting to change.

---

## 2026-10-20: Full natural run started with looser rules (run `task2_insertion_natural`)

- **Author decisions:** loosen three rules for coverage: an AI object may cover up to 30% of a
  person (was: never) and up to 30% of any other real object (was 15%), and its outline must fill
  at least 25% of its box (was 35%) (commit 75fe93b, `configs/pilot_natural_v4.yaml`). Then: no
  pilot for these rules; the 50-photo pilot v4 was stopped during planning and the full run
  started with them (`configs/insertion_natural.yaml` = v4 settings, test-checked).
- **Expectation:** more photos than v2's 36% (rough estimate ~200; v2 lost most photos to people
  and overlap), somewhat less natural placements; untested, so the result will tell.
  Work dir `data/inlayfake_v3`; the random-placement run in `data/inlayfake_v2` stays.

---

## 2026-10-20: Full natural-placement run prepared (`configs/insertion_natural.yaml`)

- **Author decision:** use the pilot-v2 settings for the full run on the 544 week-1 photos (v3's
  extra spots and classes added one photo of 50 for twice the planning time).
- **Config:** placement, generation and verification identical to `pilot_natural_v2.yaml` (a test
  checks this); differences: the 544 photos, work dir `data/inlayfake_v3`, website folder
  `week2_ai_insertion_natural`, audit sheet with every C4, and the sham rule
  `sham_original_overlap: 0.25` from the first full run.
- **Expected** (from pilot v2: planning 30–55 s per photo; 5.3 SDXL images per planned photo at
  20–45 s): planning ~4.5–8 h, generation ~9–19 h, total ~14–27 h (best guess ~18 h); about 290
  photos with a plan and 190–200 with a verified natural AI edit. GPU contention (a browser) and a
  quiet power mode slowed the pilots 2–4×.

---

## 2026-10-20: Natural pilot v3: more spots and classes do not raise coverage

- **Setup** (author's choice to raise coverage without loosening quality; commit 2814e86,
  `configs/pilot_natural_v3.yaml`): v2's rules, but up to 3 classes per photo (v2: 2), up to 10
  spots per class (v2: 5) at least 7% of the image diagonal apart (v2: 10%), 30 spot versions.
- **Plan** (~70 min, ~100 s per photo; a browser on the GPU slowed part of it): 28 of 50 photos with
  a spot (v2: 27); 82 valid of 332 candidates (v2: 57 of 211). Rejections: 86 covering > 15% of a
  real object, 83 covering a person, 19 on another object, 17 outside the image, 39 on another
  surface (windows, wall tiles, bushes, trees, metal, ...), 6 no point.
- **Generation** (finished 04:43): verified C4 for **19 of 28**, so **19 of 50 photos** (v2: 18);
  the 18 v2 photos are all among them. Failed C4 attempts: 58 not found by OWLv2, 36 filling < 35%
  of the box. Same quality as v2: 68% on ground/table/water, none on walls, objects median 1.9% of
  the image, 1 of 19 under 1%; the edits are mostly identical to v2's (Molmo answers the same
  question the same way).
- **Conclusion:** coverage is limited by photo content (people and other objects where the VLM
  points) and by generation failures, not by the number of candidate spots; v3 doubled planning
  time for one extra photo. `paper/results/task2_pilot_natural_four_way.json`; sheets
  `data/inlayfake_pilot_natural_v3/compare/compare4_1-6.jpg`.

---

## 2026-10-20: Natural placement v2: three fixes, pilot re-run (`task2_pilot_natural_v2`)

- **Fixes** (author approved; commit 06b1b9b; all off by default, on in
  `configs/pilot_natural_v2.yaml`): (1) the VLM point must lie on a surface the class rests on
  (COCO-Stuff surface sets per class group, `supports` in `configs/placement_prompts.yaml`) or on an
  object that carries it (e.g. a cup on a dining table); a point on any other object is rejected;
  (2) boxes ≥ 2% of the image and ≥ 64 px on the short side, depth scaling limited to ×0.7–1.4
  (was 0.5–2); (3) `verify.min_box_fill: 0.35` rejects an AI object whose outline fills < 35% of
  its box (a failed attempt, so it is regenerated). Hand-checked tests for each; 101 pass.
- **Pilot v2** started 00:10 on the same 50 photos, new work dir `data/inlayfake_pilot_natural_v2`
  (v1 kept for comparison), plan / insert / preview in separate processes. Planning runs at
  ~80 s per photo (v1: ~20 s): the GPU is at its memory limit (7.9 of 8.2 GB) with a browser also
  using it, so Molmo spills into system memory (back to ~25-35 s once the browser stopped).
- **Plan v2:** 27 of 50 photos with a spot (v1: 33); of 211 candidates 57 valid. The surface rule
  rejected 42 points: 15 on another real object, 5 on windows, 4 on bushes/trees, 3 on a bridge,
  3 on walls/fences, 2 on metal, 2 on a floor for a table-top class, 8 other. Still the largest
  rejections: 58 covering a person, 55 covering > 15% of a real object.
- **Generation v2** (finished 01:49, run_verify_2026-10-19T174941Z): verified C4 for **18 of 27**
  (9 dropped after all spots failed); C2/C3 with each: **18 of 50 photos**. Failed C4 attempts: 45
  object not found by OWLv2, 38 object filling < 35% of its box (fix 3).
- **Three-way comparison, same photos** (`paper/results/task2_pilot_natural_three_way.json`;
  sheets `data/inlayfake_pilot_natural_v2/compare/compare3_1-5.jpg`):

  | | random | natural v1 | natural v2 |
  |---|---|---|---|
  | photos with an AI edit | 50/50 | 28/50 | 18/50 |
  | object on ground/table/water | 46% | 64% | 72% |
  | object on wall/window/fence | 10 | 3 | 0 |
  | object area / box (median) | 0.38 | 0.28 | 0.55 |
  | object area / image (median) | 0.8% | 0.5% | 1.8% |
  | objects < 1% of the image | 30/50 | 21/28 | 1/18 |

- **By eye:** v2 is the most natural: about 14 of 18 plausible (cow in a field, surfboards on water,
  bicycle on grass, apple on a counter, toothbrush by taps, car in a car park, giraffe on a hill,
  backpack by a tent, ball on a floor, cup on a table, motorcycles on a road), about 4 poor (a sink
  in a towel rail, a bin-like object for a toilet, an odd hair drier, a dark chair in a corner).
  Objects are clearly visible instead of tiny smudges.
- **Trade-off:** coverage falls to 18/50 photos. Main limits: people or other objects in the way
  (rules), OWLv2 misses, and too-small objects now being rejected. On the full 544 this would give
  roughly a third of the photos (estimate from the pilot rate, not measured).
- **Next (author's call):** accept v2 with lower coverage, or loosen where it is cheapest (e.g.
  more candidate spots per photo or a third class, which costs VLM time, not quality).


---

## 2026-10-19: Natural vs random placement on the same 50 pilot photos (comparison)

- **Final natural pilot** (run_verify_2026-10-19T151526Z): verified natural AI edit for **28 of 50**
  photos (33 planned, 5 dropped after all spots failed), each with C2 and C3. Random-placement
  pilot (`task2_pilot_inphoto`): 50 of 50. Numbers: `paper/results/task2_pilot_natural_vs_random.json`.
- **Placement is better:** object resting on ground/floor, table or water: random 23/50 (46%),
  natural 18/28 (64%); on wall/window/fence: random 10/50 (20%), natural 3/28 (11%). Of the 75 valid
  VLM points, 51 lie on a fitting surface (36 ground, 11 water, 4 table), 3 on wall, 4 in sky/trees,
  and 17 on pixels with no COCO-Stuff surface label (often on another object, e.g. the horse placed
  on a scoreboard). So "wall pointing" is mostly points landing on objects or unlabelled areas.
- **Objects are smaller:** object area / box median 0.38 random vs 0.28 natural; object area / image
  0.8% vs 0.5%; under 1% of the image 30/50 vs 21/28. Natural boxes are smaller (median 1.5% vs 2.6%
  of the image, short side 47 vs 72 px) and the depth ratio hit its lower limit 0.5 for 27 of 75
  candidates: VLM spots tend to lie further back than the photo's own object, and single-image
  relative depth exaggerates the shrink. SDXL fills small boxes even less.
- **Side by side** (`data/inlayfake_pilot_natural/compare/compare_1-4.jpg`): natural fixes cases like
  a giraffe floating at the top edge (now in the field), a car above a bus (now on the road), a
  surfboard in the air (now on water); it fails on a horse on a scoreboard, a giraffe head in a
  wall, an oversized bottle over taps, a car drawn tiny in a tall box.
- **Proposed fixes** (not yet applied): (1) the point must lie on a surface allowed for the class
  (COCO-Stuff whitelist per group) and not on another labelled object; (2) a minimum box size (short
  side ≥ 64 px, area ≥ 2% of the image) and gentler depth scaling (limit 0.7–1.4); (3) in
  verification, reject an AI object filling < 35% of its box (counted as a failed attempt).

---

## 2026-10-19: Natural-placement pilot (run `task2_pilot_natural`, 50 pilot photos)

- **Built and tested** (commits 6696760, 63e9fed, 4e3ed3e, eb2c386): `placement.natural` (see
  ARCHITECTURE). 98 CPU tests pass, incl. hand-checked geometry and a fake end-to-end run.
- **Models on the laptop** (RTX 4070, 8 GB): Depth Anything V2 Small works (nearness higher at the
  image bottom, as expected). **InternVL3.5-4B** (planned fallback) loaded in 4-bit (4.2 GB) but
  gave invented, evenly spaced points or refused, at 12–86 s per question: not usable for
  pointing. **Molmo-7B-D** needed adapting to transformers 5.19 (its code targets transformers 4):
  16-bit vision encoder and output layer named for 4-bit loading, single-GPU placement, an
  (empty) tied-weights map, tie_weights accepting new arguments, and a greedy decoding loop with
  Molmo's own tuple cache; TensorFlow (CPU) installed for its preprocessing. Then 6.7 GB in
  4-bit, 5–18 s per pointing question, ~1 s per yes/no question, points in sensible places.
  Molmo often lists many points along one line, so `spread_points` keeps up to 5 spots at least
  10% of the image diagonal apart.
- **Plan, attempt 1:** 27 of 50 photos got a natural spot; of 208 candidates 62 valid; rejected:
  46 "outside the image" (a box centred on a point near the left/right edge could shift only
  15%), 45 covering a person, ~50 covering > 15% of a real object. Fix: a box may slide sideways
  up to half its width (the point still lies on its base). Attempt-1 files kept in the work
  dir's `attempt1/`.
- **Plan, attempt 2** (final pilot plan): **33 of 50 photos** have a natural spot; of 208
  candidates 75 valid; rejected: 62 covering a person, 64 covering > 15% of a real object (the
  author's rules), 4 answers without a point, 2 resting on window/ceiling/sky, 1 outside the
  image. 17 photos: no natural spot. Size from the photo's own same-class object for 39 valid
  candidates, from typical COCO size for 36.
- **Visual check of 12 plans:** natural for cow on grass, sheep on a hillside, surfboard at the
  water, bus at the roadside, apple on a counter, wine glass on a vanity; questionable: a toilet
  box on the wall above the real toilet, a sink on a window sill, a tall thin elephant box, an
  oversized bottle box over taps (depth ratio at its ×2 limit), a train box on the real train's
  front. The realism check is meant to catch such cases; its numbers follow below.
- **GPU memory bug:** planning and generation in one process left Molmo on the GPU (`del` without
  garbage collection), and SDXL slowed from ~20 s to ~11 min per image. Fixed (gc before
  `empty_cache` after every model); generation restarted in a fresh process 21:24.
- **Round 1** (33 photos; ~47 s per SDXL image, GPU clock ~0.7–0.9 GHz although on mains power,
  so probably a quiet MSI performance mode): C2 33/33 clean; OWLv2 accepted 17 of 33 C4.
- **Realism check failed validation.** Molmo rejected all 17 accepted C4. Wording 1 was flawed: the
  glass question was inverted ("in front of any glass" → "no" when no glass is near) and the
  compound support question got "no" every time. Reworded (one plain question each, neutral
  instruction, glass question expects "no") and re-asked on the same 17: still 0/17; "not resting
  on a surface" and "not a realistic size" for all 17, "seen through glass" for 10. By eye the 17
  are mixed: plausible (backpack on grass by a tent, ball on a floor, traffic lights at a
  junction) and poor (tiny cow head in grass, giraffe head in a window, sink on a sill, tiny
  motorcycle on a wall, bed as a dark smudge). So "no" was often right, but Molmo says "no" even
  to clear cases: its yes/no answers do not separate good from bad edits and are not used.
  Answers and a contact sheet: `data/inlayfake_pilot_natural/attempt2_realism/`.
- **Pilot continues with the realism check off** (22:30): OWLv2 only; edits are judged by eye.
- **Placement quality and object size:** many accepted C4 objects are much smaller than their box
  (SDXL fills part of it; OWLv2 accepts a detection ≥ 80% inside the box), which is part of why
  edits look odd.
- **Status at 22:52** (run_verify_2026-10-19T145001Z, after 3 rounds): verified C4 for **25 of the
  33** planned photos (C2 33/33 clean; C3 with each), 5 still retrying on their next natural spot,
  3 dropped (only one valid spot, 3 seeds failed). Over all 50 pilot photos: 25 with a verified
  natural AI edit; 17 had no spot that met the rules. Preview: `data/inlayfake_pilot_natural/
  preview.html`; sheet of the 25: `natural_c4_sheet.jpg`. Final: 28 of 50 (entry above).
- **By eye (25 accepted):** about half plausible (backpack by a tent, ball on a floor, car in a car
  park, boat and surfboards on water, traffic lights at junctions, sheep on a hillside, glass and
  bottle on counters, frisbee in the air); about half poor (horse and motorcycle on walls, giraffe
  head in a window, cat squeezed into a cupboard, sink on a sill, tiny apple and chair, a cow
  head, a bed as a dark smudge). Two causes: Molmo sometimes points at a wall or wrong surface, and
  SDXL often paints the object much smaller than its box. Not yet compared with the random
  placement on the same photos.
- **Next (author's call):** compare with `task2_pilot_inphoto` on the same photos; options are
  tightening the pointing prompt or a check of the surface under the point (COCO-Stuff), and a
  minimum object-to-box size in verification; a realism judge needs a different model or approach.

---

## 2026-10-19: Natural placement planned: VLM pointing prompts (`configs/placement_prompts.yaml`)

- **Why:** some week-2 AI objects are placed unnaturally (e.g. a skateboard in the air, a bird in
  a window pane): ~450 of 544 boxes are random free-space or anywhere spots with no surface or
  depth rule. Planned fix (author): a local VLM (Molmo-7B-D, not the week-3 VLMs) points at
  natural spots, a depth model scales the box from the photo's own same-class object, SDXL
  generates there (C2/C3 at the same box), OWLv2 + VLM yes/no checks, then the human audit.
- **Prompts:** one fixed template, `Point to {where} where another {class} could naturally
  {how}.`, with `where`/`how` from a class table (22 groups) covering all 79 non-person COCO classes
  once (78 in the week-1 photos, plus toaster); worn/held items rest on surfaces (no people are
  generated); bird, airplane, kite and frisbee may be in open sky. Four fixed yes/no check
  questions (supported, not seen through glass, plausible size, one complete object).
- **Status:** prompts only; not implemented or tested yet. The 50-photo pilot will show whether
  they work; changes to them will be logged.
- **Decision (author, 2026-10-19):** a natural-placement box may cover at most **15%** of any
  real labelled object's area (COCO + audit adds), never a person; covered objects keep their
  label with recomputed visibility (ignore below 10% visible, as now). No fitting spot or class:
  the photo gets no AI edit (no random fallback). In the current set 60 of 544 AI edits cover a
  real object (38 occlusion-rule, 22 anywhere). Stored in `configs/placement_prompts.yaml`
  (`box_rules`).

---

## 2026-10-18: Label-audit review on the results website

- **Author request:** show the week-1 label-audit review on GitHub Pages, and keep it updatable
  when a decision changes.
- **What:** `audit --review --share` writes the review page as
  `docs/results/week1_real_baseline/audit.html` (the week's website folder) and uses the photo
  copies the week-1 viewer already put there (`images/<C0 id>.jpg`; any missing one is copied), so
  only the page file is added (0.9 MB). New website tab *Week 1 · Label audit* in `docs/index.html`.
- **Checked** through a local server: 3 tabs, 107 cards, all 70 photos load. Test extended (shared
  page, every image link exists); 91 pass.
- **Updating:** after a decision change, the same command rebuilds the page; commit
  `docs/results/week1_real_baseline` (README steps).

---

## 2026-10-18: Label-audit review page (`audit --review`)

- **Author request:** a clear page to re-check the week-1 label-audit decisions. The old
  `audit.html` (9 Oct) shows only small crops, without the decision or COCO's own labels; the
  review page used on 11 Oct was not kept.
- **What:** `python -m inlayfake audit --config configs/real_baseline.yaml --review` writes
  `data/inlayfake/audit_review.html` from the current decisions file (`common.decisions_path`):
  card k = row k of `audit_decisions.csv`, current decision (colour-coded), a zoom around the
  candidate (orange) with COCO's labels and classes (green), the whole photo, and a Show dropdown
  by decision with counts. Read-only; decisions are still edited in the CSV. Shared site style.
  1 new test; 90 pass. Checked in the browser: 107 cards; filter add 77 / ignore 26 / reject 4.
- Local page (it shows the COCO photos from `data/`), not part of the website.
- **Added (author request):** each card lists the 4 detectors with their best same-class score at
  IoU ≥ 0.5 to the candidate (✓ when ≥ 0.7, i.e. agreeing; "below 0.7"; "no box here"), from the
  saved C0 detections (`data.audit.detector_matches`). Recomputed agreement equals the stored
  `n_detectors` for all 107 candidates (0 mismatches). New candidates also store the agreeing
  detector names. 1 new test.

---

## 2026-10-18: Finding: the 77 audit-added objects have no per-object forensic scores (week 1)

- **What:** `runs/task1_real_baseline/forensics/*.jsonl` has per-object rows only for COCO labels:
  7,764 instance rows = 3,882 objects × 2 conditions, and none for the 77 audit "add" labels
  (checked in `ela.jsonl`). The detector metrics use 3,959 objects (3,882 + 77).
- **Why:** `forensics` ran (9 Oct 02:52 local) before the audit decisions were saved (03:42); it
  scores the labels present at run time and skips photos already done, so the later "add" labels
  were never scored.
- **Effect:** Table 2b's per-object contrast on real objects covers 3,882 objects, not 3,959 (2% left
  out); image-level forensic scores and all detector numbers are unaffected.
- **Fixed (author approved, same day):** the forensic files were backed up
  (`runs/task1_real_baseline/forensics_before_audit_rescore/`), the 70 audited photos' rows removed,
  and `forensics` re-run for those 140 images (57 s, CPU). Now 9,006 rows per method: 154 new rows
  (77 added objects × C0/C1; scored with their box as mask, since audit labels have no COCO
  outline), every other row identical to the backup (0 changed scores).
- **Pending:** `evaluate` (and `report`) on the author's laptop, then Table 2b's per-object contrast
  (n 3,882 → 3,959 objects per condition) in the paper and `paper/results/`.

---

## 2026-10-18: One label-audit file for all weeks (scoring fix before week 3)

- **Found:** scoring stages (`detect`, `forensics`, `evaluate`, `report`, `viewer`) read
  `audit_decisions.csv` from the run's own work dir. Week 2's work dir (`data/inlayfake_v2`) has
  none, so scoring the week-2 images in week 3 would silently have dropped the 77 added labels
  and 26 ignore regions of the week-1 label audit (week-1 and week-3 numbers not comparable).
  Nothing was scored with the wrong labels: week 2 has not been scored yet.
- **Fix (author approved):** `common.decisions_path`: `audit.decisions`, else
  `placement.audit_decisions`, else `<work>/audit_decisions.csv`. `configs/insertion_inphoto.yaml`
  names `data/inlayfake/audit_decisions.csv`; a named file that is missing stops the run. 1 new
  test.
- **Checked:** week 1 and week 2 both resolve to that file and apply 77 adds and 26 ignore regions
  over 544 photos. Week 1 used the same file before, so its results are unchanged.
- **Effect of changing a decision later:** week 1 needs `evaluate` + `report` again (and the
  viewer); week 2 picks it up when scored. Three week-2 AI edits rely on an "add" for their
  "class in the photo" (196442 kite, 315257 bowl, 358525 bed).

---

## 2026-10-18: Results website: week folders renamed, one shared design

- **Author request:** the folder names `docs/viewer/` and `docs/preview/` did not say which week
  or task they hold, and the two pages looked different though they sit side by side in the
  website; the same should hold for later weeks.
- **Folders:** `docs/viewer/task1_real_baseline/` → `docs/results/week1_real_baseline/`,
  `docs/preview/task2_insertion/` → `docs/results/week2_ai_insertion/` (git mv, images unchanged).
  The folder comes from a new `site` block in the config (week, title, folder; `site.dir`
  docs/results); configs without it (pilots, tests) fall back to `docs/results/<run_name>/`.
- **One design:** `src/inlayfake/site.py` holds the shared stylesheet (colour tokens with
  light/dark, sticky header, photo cards, image grid with SVG boxes, legend), the condition names
  (C0 original … C4 AI insert) and the page title ("Week 2 · AI insertion"). The week-1 viewer
  template now takes its base style from it; the week-2 preview was rebuilt on it (same header,
  cards and grid as week 1; planned box dashed yellow, inserted/pasted object purple as in the
  week-1 legend). `docs/index.html` uses the same tokens and the new paths.
- **Checked:** both pages rebuilt (week 1: 544 photos, 66.9 MB; week 2: 2,720 images), shown in
  the tab site through a local server in light and dark mode; week-1 cards render, week-2 dropdown
  works (missing 0, all 544). Tests updated for the new markup and config; all pass.

---

## 2026-10-18: One results website with week tabs (`docs/index.html`)

- **Author request:** join the week-1 viewer and the week-2 preview into one site with tabs, to
  serve as the repo's GitHub Pages site.
- **What:** `docs/index.html` (tab bar: Week 1 real-photo baseline, Week 2 AI insertion; the page
  below is the week's self-contained folder in an iframe; tab kept in the URL hash; "open in a
  new tab" link; light/dark like the week-1 viewer) and `docs/.nojekyll`. Week pages unchanged;
  adding a week = one entry in `WEEKS`. The iframe has a white background because the week-2
  preview sets none (its black text was unreadable on the dark frame).
- **Checked** with a local web server (`.claude/launch.json`, `http.server` on docs/): both tabs
  load, the week-1 tables and photos show, the week-2 Show dropdown works in the frame (all 544 /
  complete 544 / missing 0), 2,720 images.
- **Not done:** switching GitHub Pages on (repo Settings → Pages → branch, `/docs`) is the
  author's step; the site is public, so the publishing checklist applies first.

---

## 2026-10-18: Publishing checklist started (`docs/PUBLISHING_CHECKLIST.md`)

- **What:** a list of things to settle before the repo, previews or paper go public: COCO photo
  licences for the shared previews (`docs/viewer/`, `docs/preview/`), the dataset licence
  statement, double-blind anonymity, 24 commits whose messages carry an assistant co-author
  line (against `CLAUDE.md`; removing them needs a history rewrite, the author's decision), no
  data in git, and the paper's open items. README now says how to update the shared week-2
  preview after the benchmark changes.

---

## 2026-10-18: Week-2 preview shared in the repo (`docs/preview/task2_insertion/`)

- **Author request:** the week-2 preview should be on GitHub like the week-1 viewer.
- **Code:** `preview --share` builds the benchmark view for all photos and writes a self-contained
  copy to `preview.share_dir/<run_name>/` (default `docs/preview/`): index.html plus downscaled
  JPEG copies (`preview.share_max_side` 480, quality 80) of every image it shows, links rewritten
  to `images/`. Boxes are scaled from original pixel coordinates to the fixed 240 px thumbnails,
  so they need no change. 1 new test.
- **Built:** 544 photos, 2,720 images, 95 MB (week-1 viewer: 70 MB); every link resolves inside
  the folder. Like the week-1 viewer it contains downscaled COCO photos, which carry their own
  Flickr licences (the dataset licence statement is still on the paper's To-verify list).

---

## 2026-10-18: Audit sheet: C4 only (544 rows)

- **Author:** audit only the C4 AI insertions; no C2 rows. `verify.audit_only_conditions: [C4]`
  keeps only C4 on the sheet. The random draw is unchanged, so its 52 C4 rows stay a random
  sample of the C4 edits (`in_random_sample: y`). 1 new test. Sheet: 544 rows, bed row 6, train
  row 143; the empty 601-row sheet is kept as `audit_edits_c4_plus_c2.*`.
- **Consequence:** C2 shams are checked by OWLv2 only; the paper says so (§4).
- **Boxes on the audit page (author request):** each card now shows the planned box (dashed
  yellow) on both crops and the verified AI object box (red) on the *after* crop, as overlays in
  % of the crop; the CSV is unchanged. Checked by drawing the overlays onto rows 1 and 6.

---

## 2026-10-18: Human audit extended to every C4 edit (601-row sheet)

- **Author decision:** audit all AI insertions, not only a 10% sample: a sample estimates the
  verifier's error rate but leaves the wrong labels in the other 90% (with 52 sampled C4s, 2 errors
  would mean roughly 5–60 unknown wrong labels among the other 492).
- **Code:** `verify.audit_all_conditions` (full run: `[C4]`) puts those conditions on the sheet in
  full, next to the random 10% sample, which is unchanged and flagged in a new CSV column
  `in_random_sample`. The HTML page is now built from the CSV when one exists, so page and sheet
  never diverge again (the cause of the mismatch found earlier today). 1 new test; 85 pass.
- **Sheet:** 601 rows: 544 C4 (52 of them in the random sample) + 57 C2 (random sample), C4 rows
  first. The bed (10995) is row 6, the train (151857) row 143. The empty 109-row sheet is kept as
  `audit_edits_10pct_only.*`.
- **Next:** author fills the sheet (about 1–1.5 h); rejected C4 edits are then removed and the
  audit results go into §4 and Table 1.

---

## 2026-10-18: Insertion run complete: 544 of 544 photos have C0–C4; audit sheet rebuilt

- **Corner boxes** (restart 9, last check run_verify_2026-10-18T070254Z, 15:02 local): 151857
  (train) passed on corner box 23, seed 2 (OWLv2 0.25); 10995 (bed) on corner box 27, seed 1
  (0.20). Their C2 and C3 passed. Both scores sit at the 0.2 threshold and **by eye both are
  doubtful**: a pale, pillow-like shape by the window; a small grey shape at the left edge. Flagged
  for the human audit; if the author rejects them they are reported as failures.
- **Final benchmark:** all 544 photos have C0, C1, C2, C3 and C4; AI edits from the free-space rule
  464, the occlusion rule 38, placed anywhere 42 (2 of them in the corner); 1,088 edited images
  (`paper/results/task2_insertion_final.json`). The ledger already carries the new
  `merged_source` labels (restart 9 ran the new code).
- **Audit sheet rebuilt** from the final ledger: the old `audit_edits.csv` (86 rows, 17 Oct,
  unfilled, partly no longer in the benchmark) and `.html` (another sample) are kept as
  `audit_edits_before_final.*`. New sheet: 109 edits (10%; 57 C2, 52 C4), CSV rows and numbered HTML
  cards in the same order, no C5 in the page text. The bed edit is in the sample; the train edit
  is not (random sample kept as drawn) and is checked separately.
- **Next:** author fills the audit; then Table 1 counts and week 3.

---

## 2026-10-18: Bed and train photos: boxes in the bottom-left corner (restart 9)

- **Status at the stop (14:36):** plain anywhere boxes 19–21 failed all 9 tries for both 10995
  (bed) and 151857 (train), as did boxes 13–18.
- **Author:** put the AI object in the bottom-left corner of these 2 images.
  `placement.merged.corner_from: 22`, `versions: 27`: boxes 22–27 sit at x = 0 with their bottom
  edge on the image bottom (any overlap; size alternates the photo's own instance and a typical
  COCO size), 3 seeds each, still verified by OWLv2. The box-22 rows already made (plain anywhere)
  and the one unverified image at box 22 were moved out (`replans_before_corner.jsonl`,
  `superseded_box22/`). 1 new test. Scratch dry run: only these 2 jobs, first boxes bed
  [0, 230, 263, 250], train [0, 349, 146, 104]. Restart 9 at 14:36, PID 18448. Result [[TBD]].

---

## 2026-10-18: Paper: why C1 and C2 are needed, and how C3 is made

- **What:** at the author's request, §4 (*Why each condition*) now says how the C3 real paste is
  built (same-class COCO donor from another photo, ≥ 48 px, not touching its border, aspect within
  25%; cut with its COCO mask; scaled to fit the box, bottom-aligned; 3 px feather; no generator;
  same JPEG settings) and adds *Why not only C0, C3 and C4?* (C1: matched-encoding reference, since
  re-encoding alone shifts forensic scores; C2: separates AI-made objects from AI-repainted
  pixels). No new numbers. The author had asked whether C0, C3 and C4 would be enough; C1 and C2
  stay.

---

## 2026-10-18: Bed and train photos: plain anywhere boxes again from box 19 (restart 8)

- **Status at the stop (14:08):** the 4 C2 shams passed on their first new seed (restart 7);
  10995 (bed) and 151857 (train) failed all 16 tries on boxes 13–18 (box 18 seed 1).
- **Author:** for these 2, place the box anywhere, any overlap allowed.
  `placement.merged.avoid_same_class_until: 18`: boxes 19–24 are plain anywhere (as boxes 1–6;
  the verifier still never counts a detection of the real bed or train). Box 18 keeps its box
  for its last 2 seeds. 1 new test. Scratch dry run: only these 2 jobs. Restart 8 at 14:12,
  PID 31452. Result [[TBD]].

---

## 2026-10-18: Last 2 AI edits and 4 C2 shams re-run (restart 7)

- **Author:** add the same class anywhere for 10995 (bed) and 151857 (train); regenerate C2 for
  108440, 203580, 259571, 262938.
- **Why the 4 shams failed** (OWLv2 re-run on their 12 attempts, and checked by eye): no new
  object; every attempt's detection is the photo's **real** train / umbrella / bus / book that
  partly overlaps the box (25–97% of the detection on the real object's box, but IoU < 0.5 so
  the existing guard did not discount it), at the same spot in every seed. More seeds alone
  would fail the same way.
- **Change:** `verify.sham_original_overlap: 0.25` (full run only): a C2 detection lying ≥ 25%
  of its area on a real object of the class is that object, not a new one.
  `placement.merged.c2_attempts: 6`: C2 at a merged box gets 3 more seeds, checked with that
  rule (the 12 failed attempts stay failed). `placement.merged.versions` 12 → 24: the bed and
  train photos get up to 12 more anywhere boxes (not on the real object, as boxes 7–12). C4 and
  C4-occ checks are unchanged. 2 new tests; 82 pass. Scratch dry run: exactly these 6 jobs.
- **Restart 7** at 13:34, PID 16592. Result [[TBD]].

---

## 2026-10-18: Re-run done: 542 of 544 photos have an AI edit (run `task2_insertion` finished)

- **Result** (run_verify_2026-10-18T031732Z, 11:17 local; 18 rounds of 3–7 images): 5 of the 7
  dropped photos passed with boxes 7–12: toilet (box 7), tv (box 8), chair (box 9), bed in 65074
  (box 10), surfboard (box 12); their C2 and C3 passed too. Still without an AI edit: 10995
  (bed) and 151857 (train), both close-ups whose only class is the object filling the frame.
- **Final benchmark:** AI edit for 542 of 544 photos (C4 464, C4-occ 38, anywhere 40); 538 have every
  condition C0–C4; 4 lack only C2 (sham attempts kept "producing" the target, 108440, 203580,
  259571, 262938); 1,080 edited images in the ledger (`paper/results/task2_insertion_final.json`).
- **Next:** human audit of 10% (`audit_edits.csv`), Table 1 counts, then week 3.

---

## 2026-10-18: Merged run done (537 of 544); re-run of the 7 dropped photos (restart 6)

- **Merged run result** (run_verify_2026-10-17T222301Z, 06:23 local; 10 rounds after restart 5):
  AI edit for 537 of 544 photos: from C4 464, from C4-occ 38, newly placed anywhere 35; 7 dropped
  after 6 boxes × 3 seeds. 533 photos have every condition C0–C4; 4 more lack only C2: in all 3 sham
  attempts OWLv2 found the target class at the box (108440 at an adopted C4-occ box, 203580, 259571,
  262938 at anywhere boxes; probably the real object partly inside the box, not yet checked)
  (`paper/results/task2_insertion_merged_round_final.json`). Anywhere boxes passed 35 of the 42
  photos; per round the pass rate fell from 7/42 to 0–4 per round.
- **Why the 7 were dropped** (OWLv2 re-run, read-only, on all 210 failed C4/C4-occ/anywhere attempts
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

## 2026-10-18: Scope change: the two AI placement rules (C4, C4-occ) merged into one condition C4 (restart 5)

- **Why:** after the round-7 check, 44 of 544 photos still had no verified AI object under the
  separate C4 (free space) and C4-occ (covers 20–50% of one object) rules; 284 had both, 175 only C4,
  41 only C4-occ (`paper/results/task2_insertion_merge_decision.json`, read-only recomputation from
  the verify state before the merge).
- **Decision (author):** one AI condition, kept under the code C4: "an AI object of a class
  already in the photo, placed anywhere"; that class rule is the only placement rule. Verified C4
  and C4-occ edits are kept. Photos with both use the C4 (author's choice over a random pick: their
  C2/C3 already exist, no extra GPU time). C2 and C3 sit at the chosen AI box.
- **Code:** `placement.merged` (on in `configs/insertion_inphoto.yaml`). `replan.merged_version`:
  verified C4 → kept; else verified C4-occ adopted as free version 100 (box/class copied,
  `placement_rule: adopted_c5`); else versions 101–106 place a box anywhere (photo's classes,
  free-track class first; size alternates own instance and typical COCO size), 3 seeds each,
  OWLv2 still has to confirm the object. C4-occ is no longer generated; C2/C3 at a merged box wait
  for its verified AI edit. The final ledger has one AI record per photo, relabelled C4, with
  `EditInfo.merged_source` (C4 / C4-occ / anywhere). Preview has a merged view (dropdown: AI source,
  retrying/dropped, C2/C3 missing, with counts). 3 new tests (hand example, end-to-end switch
  from separate to merged, preview); 79 tests pass. Dry run on a scratch copy: 459 C4 kept,
  41 C4-occ adopted, 44 anywhere boxes, 86 jobs.
- **Restart 5:** run stopped in round 8 (75 of 242 images made). Those attempts were verified
  first with the merged config (run_verify_2026-10-17T194502Z, 114 rows): AI edit from C4 464,
  from C4-occ 38, none yet 42 (`merge_decision.json`). Merged `insert` started 03:46, PID 31272,
  round 1: 81 jobs (42 AI anywhere, 38 C2 at adopted C4-occ boxes, 1 C2 retry). Backups:
  `replans_before_merge.jsonl`, `verify_before_merge.jsonl`, `ledger_before_merge.jsonl`,
  `insert_progress_part5.log`, `insert_part5.log`.
- **Consequences (paper §4, PRD):** H2 loses its controlled occlusion condition; H2 is evaluated
  on originals the AI object does not cover, covered objects separately by visible fraction
  (observational). 298 verified C4-occ edits (336 photos have one, 38 adopted) leave the benchmark because their photos use the C4. C2 at an
  adopted C4-occ box repaints part of the covered real object, which verify marks mixed/modified.
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
  false-accusation rate; a real-object paste control (C3) and occlusion arm (C4-occ); same-class
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
  new C4 (fill) passed 35 of 57 on the first try, new C4-occ (fill) 34 of 128; older C4 on a new
  box 13 passed / 48 failed; older C4-occ on a new box 15 / 116. Totals C4 446, C4-occ 325, C2 515 clean;
  281 edits into round 8 (210 C4-occ, 70 C4, 1 C2).
- **Found:** 13 C4 edits that had passed with a class not in the photo stopped counting. The
  rule "outside classes only if the photo has no in-photo AI object" was re-evaluated at every
  step, and the C4-occ fill had given all 13 photos an in-photo C4-occ (3 passed, 10 retrying).
- **Decision (author):** keep them. `replan.track_version` now returns the version of the first
  passing attempt before applying the outside-class restriction, so the rule decides what to
  try, never withdraws a verified edit. Test updated; re-plan and verify tests pass (20).
- **Read-only check:** under the new rule C4 459 (+13), C2 528 (+13), C4-occ unchanged, no edit
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
  (5 photos already have a C4-occ; 2 have C4-occ still retrying).
- **Read-only test:** with a typical COCO size (the size the original planner used), 6 of 7 get
  a new box under unchanged C4 rules; the own-instance size gives none.
- **Fix (author approved):** re-plans now alternate the own-instance size and a typical COCO size
  (6 sizes instead of 5 own-instance sizes); C4/C4-occ rules unchanged. Re-plan and plan tests pass.
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
  `free_category` and used by C2/C3/C4 (and their re-plans), while C4-occ keeps the plan class.
  2 new tests; plan/config/re-plan/verify tests pass (31).
- **Check before applying:** re-planned in a scratch copy and compared with the live plan
  (which holds the C4-occ fill): 487 plans byte-identical, 57 gained a C4 box (5 with another class),
  nothing else changed in them, C4-occ fill reproduced (128). 13 photos stay C4-occ-only.
- **Restart:** stopped at Sun 01:27 (that round at 145/302) and restarted at 01:28 (process
  13064). Plans now: 531 of 544 with a C4 box, 535 with a C4-occ box
  (run_plan_2026-10-17T172754Z.json, paper/results/task2_insertion_plan_fills.json). Kept:
  `plans_before_free_fill.jsonl`, `insert_progress_part2.log`, `insert_part2.log`.
- **Expected cost (estimate):** 57 new C4 jobs, each with its C2 (and C3, no GPU): roughly
  1.5–2 h more.

---

## 2026-10-18: C4-occ fill: 128 more photos get a C4-occ box (run `task2_insertion` restarted)

- **Question (author):** why do 137 of the 544 photos have no C4-occ box? A read-only test re-ran
  the C4-occ box search on them with one rule relaxed at a time: with the same rules but three box
  sizes, 84 fit; box sized from the covered object 103; other objects covered up to 20% (not 5%)
  95; no limit 96; cover 10–50% of the target 102; covered objects down to 32 px (not 64) 114.
  So the main cause was the search (the plan tries one C4-occ box size when the photo also has a C4
  box), not the 5% rule, which I had wrongly named as the cause before checking.
- **Decision (author):** allow covered objects down to 32 px. Implemented as
  `placement.occlusion_fill` (off by default; on in `configs/insertion_inphoto.yaml`): after the
  main search, plans without a C4-occ box get one, trying 6 sizes (typical COCO size, then the photo's
  own instance size) and finally a box sized from the covered object; covered object ≥ 32 px; the
  other C4-occ rules unchanged (20–50% of one object, ≤ 5% of any other). Own seeded generator per
  photo. C4-occ re-plans now use the same 32 px minimum. The main search keeps the original 64 px
  rule so that re-running `plan` reproduces earlier plans. New test; plan/config/re-plan tests
  pass.
- **Check before applying:** re-planned in a scratch copy and compared with the live plan: 416
  plans byte-identical, 128 gained a C4-occ box, 0 other changes, every C2/C3/C4 box and class
  identical; 9 photos still without C4-occ.
- **Restart:** the run (process 14000, round 7 at 65/239) was stopped on Sun 00:39 and restarted
  at 00:40 (process 13792) with the new plan (535 of 544 with a C4-occ box). Kept:
  `plans_before_occlusion_fill.jsonl`, `insert_progress_part1.log`, `insert_part1.log`. All
  finished images and checks are kept; the new C4-occ jobs start at their first seed.
- **Expected cost (estimate):** 128 new C4-occ jobs at C4-occ's retry rates so far, roughly 2–4 h more.

---

## 2026-10-17: Decision: keep C2; condition rationale written into the paper

- **Question (author):** is C2 useful, since its box holds no object and is AI-repainted like
  C4? Could C3/C4/C4-occ alone test real vs AI objects?
- **Answer recorded:** the core real-vs-AI object test needs only objects (C1/C3 real, C4/C4-occ AI),
  but C2 is the placebo-like control: same generator, box and pipeline, no object. It shows
  whether a forensic method detects AI *objects* or any *regenerated region* (H3), and whether
  repainting alone disturbs nearby real objects (H2). It is never scored as an AI object.
- **Decision (author):** keep C2. Remaining cost about 1 h of the run (re-done shams at moved
  C4 boxes).
- **Paper:** §4 now explains why each condition exists, why C2/C3/C4 share one box (and move
  together on re-plan) while C4-occ has its own (no box can avoid all objects and cover 20–50% of one),
  and why C4-occ matters (realistic overlap, hardest case, occlusion control for H2). §6 states that
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
  clean, C4 269/474, C4-occ 123/407 passed on attempt 1; 491 edits went into round 2. Round 1: 1,355
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
  free box, 407 with an occlusion box, 70 C4-occ-only, 2 with a class not in the photo, 1 anywhere.
  First round: 474 C2 + 474 C4 + 407 C4-occ = 1,355 images (~7.5 h); total at the pilot's rates
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
  600 planned; 474 with a free box (C2/C3/C4), 465 with an occlusion box, 126 C4-occ-only, 4 with a
  class not in the photo, 7 placed anywhere. First-attempt jobs: 474 C2 + 474 C4 + 465 C4-occ =
  1,413 SDXL images.
- **Time estimate (from the same-class pilot, not measured on this run):** the pilot generated
  469 images for 125 first-attempt jobs (3.75×; C2 1.6, C4 3.6, C4-occ 6.0 — includes some images
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
  in `pilot_inphoto.yaml`): when no box satisfies the rules after every other fallback, a C4-occ box
  is placed anywhere inside the image with no overlap rule (`placement.place_anywhere`). Class:
  the photo's smallest insertable class, else an outside class. Used by the plan stage (own
  generator; earlier plans byte-identical: week 1 544/544, pilot 1 46/46, same-class 49/49) and
  by C4-occ re-plans whose box cannot be placed. C4 stays rule-abiding (no occlusion), because H2
  relies on it. New ledger field `EditInfo.placement_rule` = "anywhere" so week-3 analyses can
  include or exclude these. `verify` already recomputes the visible part of every covered real
  object. 2 new tests; `pytest -q`: 71 passed.
- **Result** (run_plan_2026-10-16T170747Z.json, run_verify_2026-10-16T171152Z.json,
  paper/results/task2_pilot_inphoto_anywhere_verify.json): all 50 photos planned; 421455 (car
  interior) got an anywhere C4-occ truck that passed on box 2 (OWLv2 0.34). C4 39/41, C4-occ 39/43,
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
  Now an edit may use an outside class only if the photo has no in-photo pass in C4 or C4-occ and the
  other edit is exhausted too; a pass at an outside version stops counting otherwise. Test added;
  `pytest -q`: 69 passed.
- **Result** (run_verify_2026-10-16T165640Z.json,
  paper/results/task2_pilot_inphoto_outside_verify.json): C4 39/41, C4-occ 38/42, C2 39/41; final
  ledger 253 records, audit sample 12. **49 of 50 photos have a verified AI object.** Only one
  edit uses a class not in the photo: 151857, a **clock** (OWLv2 0.64) painted over the train's
  windscreen — clearly there and tonally matched, but implausible; flagged for the audit.
- **Still without one (1):** 421455 (car interior): the "car" box covers 88% of the frame, so no
  box satisfies the C4-occ rule (≤ 5% of any other object); the class is not the problem.

---

## 2026-10-17: C4-occ boxes sized from the covered object: 48 of 50 photos have an AI object

- **Question (author):** how to give the 4 remaining photos a box. Diagnosis per photo (COCO
  boxes): 151857 one train covering 67% of the frame; 562443 one zebra box 61%; 457848 three
  overlapping motorcycles (61%, 52%, 22%) and 5 people, boxes cover 83%; 421455 a car-interior
  shot whose "car" box covers 88% with dog, phone and person inside. None has free space for C4.
- **Bug found:** C4-occ boxes were sized from the class's usual COCO size (often ~2% of an image),
  which can never cover 20% of an object filling most of the frame. Fix:
  `placement.occluding_sized_to_target` sizes the C4-occ box from the object it must cover (30–60%
  of its box area, the class's aspect ratio, ≤ `max_area_frac`), used when the usual size fails,
  in the C4-occ-only plan fallback and in C4-occ re-plans. The C4-occ rules (20–50% of one object, ≤ 5% of
  any other) are unchanged. Test with a frame-filling train; `pytest -q`: 66 passed. Plans
  re-checked: week 1 544/544, pilot 1 46/46, same-class pilot 48/48 identical; 562443 gained a
  plan (run_plan_2026-10-16T162944Z.json).
- **Correction:** 151857 and 457848 already had C4-occ-only plans; their objects failed. The 6
  re-plan rows of 457848 were empty because they were computed with the old sizing; they were
  removed (backup `data/inlayfake_pilot_inphoto/replans_before_targetsize.jsonl`) and
  recomputed.
- **Run** (run_verify_2026-10-16T163304Z.json,
  paper/results/task2_pilot_inphoto_targetsize_verify.json; 3 min): 562443 zebra passed on
  attempt 2 (an AI zebra head covering the grazing zebra's head, convincing); 457848 passed on
  box 2 (a small AI motorcycle, real but small for the scene). Totals: C4 39/41, **C4-occ 37/42**,
  C2 39/41; final ledger 252, audit sample 12. **48 of 50 photos have a verified AI object.**
- **Still without one (2):** 151857, 12 attempts at a train over the real train: SDXL extends
  the real train, which the original-object guard correctly refuses, and there is no other class
  in the photo; 421455, no box satisfies the C4-occ rules. Fixing these would need a rule change
  (another class not in the photo, or a different occlusion rule), so they are left out.

---

## 2026-10-17: Class switch and C4-occ-only plans (run `task2_pilot_inphoto`): 46 of 50 photos have an AI object

- **Author's request:** as many photos as possible must get an AI-inserted object.
- **Class switch** (`edit.max_class_switches`, 0 by default, 3 in `pilot_inphoto.yaml`): after the
  re-planned boxes of the planned class fail, a track moves on to another class already in the
  photo (same rules: audited GT, never person), smallest median instance area in the photo
  first, one class per version. Versions where no box fits are skipped instead of ending the
  job. The class is stored per version (`category` in `replans.jsonl`); C2/C3 follow the C4
  class and box, C4-occ uses its own track's class.
- **C4-occ-only plans:** in `in_photo` mode a photo with no free space for any class may still get
  an occluding box (no C2/C3/C4). That search uses its own seeded generator, so the other plans
  are unchanged: checked byte for byte — week 1 544/544, pilot 1 46/46, same-class pilot 41/41
  identical; 7 photos gained a C4-occ-only plan (48 of 50 planned; run_plan_2026-10-16T155841Z.json).
- **Tests:** 3 new (version skipping, class switch with matched C2/C3, C4-occ-only plan leaves the
  shared generator untouched); `pytest -q`: 65 passed.
- **Run** (run_verify_2026-10-16T162120Z.json,
  paper/results/task2_pilot_inphoto_classswitch_verify.json; 22 min): **C4 39/41**, **C4-occ 35/41**
  (6 of them after a class switch: motorcycle ×2, bus, surfboard, dining table, handbag),
  **C2 39/41** clean, final ledger 248 records, audit sample 11.
- **Per photo:** **46 of 50** sampled photos now have at least one verified AI object (C4 or
  C4-occ); 39 have a C4. Before the change: 39. The 4 without one: a photo with a single train, one
  with a single zebra, and two crowded street scenes (dog, phone, car, truck; 3 motorcycles among
  5 people) where no box fits for any class.
- **Visual check of the 10 new C4-occ objects:** 8 clear (toilet, motorcycles, giraffe, handbag,
  spoon, dining table), 2 weak (a tiny surfboard on the sand, a very small bus).
- **Next:** author audit (11 rows); full-run configuration (re-plan of all 600 sources).

---

## 2026-10-16: Re-plan on failure (run `task2_pilot_inphoto`): 93% of C4 with a verified object

- **Author's request:** the photos must get an AI-inserted object. Loosening the verifier
  further was advised against and not done: all 31 dropped jobs were checked by eye and their last
  attempts contain background (or, under C4-occ boxes, the photo's own object), so passing them would
  put empty `ai_inserted` boxes into the ground truth. Instead, failed jobs get new boxes.
- **Implemented** (`stages/replan.py`, `edit.max_replans`, default 0 = unchanged): when all
  `max_attempts` seeds fail on a box, the job gets box version n+1 (up to `max_replans`). Two
  tracks per source: *free* (C4; C2 sham and C3 paste follow it, so controls stay matched to the
  C4 box) and *occ* (C4-occ). A new box rests on a surface of the class (COCO-Stuff; free space if
  none fits) and is sized like one of the photo's own instances of that class (×0.8–1.2).
  Version 0 keeps the original seeds, so existing images are reused; versions are stored in
  `replans.jsonl`; `EditInfo.plan_version` records the box. `verify` and the final ledger count
  only the current box (one C3 per source). Preview shows "box n" and marks superseded C3.
  Fake verifier gained `fake_fail_until` for tests; 3 new tests; `pytest -q`: 62 passed.
- **Run** (`configs/pilot_inphoto.yaml`, `max_replans: 3`, window 1.2×, looser verifier;
  run_verify_2026-10-16T130208Z.json, paper/results/task2_pilot_inphoto_replan_verify.json;
  boxes in paper/results/task2_pilot_inphoto_replans.jsonl; 65 min wall time):
  31 re-plans created (13 on a surface, 14 free space, 4 with no box that fits).
  **C4 38/41 (93%)** — box 0: 27, box 1: 4, box 2: 7, 3 dropped; **C4-occ 25/34 (74%)** — box 0: 17,
  box 1: 4, box 2: 1, box 3: 3, 9 dropped; **C2 39/41** clean (the 2 missing are sources where no
  new box fit, which also have no C4). Final ledger 223 records, audit sample 10.
- **Visual check of the 19 edits that passed on a new box:** about 16 clearly show the object
  (dog, elephant, train, book on the floor, horse, skis, vase, dining table, traffic light,
  airplane, surfboard on the beach); 3 are weak (a tiny boat, a blurred car, a tiny surfboard);
  a few are still implausible (umbrella above the sea, bed in a hedge).
- **Limits:** when no new box fits (4 crowded photos) the source loses its C2/C3 at the old box
  as well; 85% target met for C4 (93%), not yet for C4-occ (74%); human audit pending.
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
     original object never counts (`verify.original_iou`), so the occluded real object in C4-occ
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
  C4-occ 7/34 (17/75 = 23%; pilot 1 at 2.5×: 6%). SDXL 116 images in 50:07 (25.9 s each). Round 2
  (58 images, 38:31, slowed by a CPU diagnostic running at the same time) was generated and
  then the run was stopped for the rule change; the old-rule results are kept in
  `data/inlayfake_pilot_inphoto/verify_oldrule/`.
- **Final, new rule, up to 3 attempts** (run_verify_2026-10-16T114020Z.json,
  paper/results/task2_pilot_inphoto_final_verify.json): **C4 27/41 (66%)** — 20 on attempt 1,
  6 on 2, 1 on 3, 14 dropped; **C4-occ 17/34 (50%)** — 8 / 9 / 0, 17 dropped; **C2 41/41**; final
  ledger 208 records; audit sample 8. Round 3: 33 + 2 images (19.0 and 22.0 s each).
- **Visual check of all 44 passed C4/C4-occ:** nearly all show a real object of the class (dogs,
  sheep, horse, giraffe, book, broccoli, cups, cat, traffic lights, backpacks, handbag, train).
  Weak: a dark laptop (0.34), a faint umbrella (0.30), a thin surfboard line, a tiny baseball
  glove (0.28). Realism is now the main problem: several float in implausible places (bus in the
  sky, car in the trees, handbag under a hangar roof, tiny giraffe on a hilltop). The dropped
  jobs are mostly such placements and very large classes (train, bed, dining table, surfboard).
- **Interpretation:** the verified-object rate rose from 6% (pilot 1) to 66% C4 / 50% C4-occ,
  through the tighter window, the looser rule and 3 attempts together; the 85% PRD target is not
  met yet. Photos without a verified C4/C4-occ simply have none in the benchmark.
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
  giraffe); the week-1 counts in PAPER §4 (544 planned, 363 with C4-occ) come from the old plan and
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
  28, wall 4, tabletop 3. Cost of the rule: fewer plannable photos, C4-occ halved, cup only 1.
- **Round-1 verify** (run_verify_2026-10-16T085229Z.json; code was uncommitted at run time and is
  committed unchanged with this entry): `C2 passed 35/35` · `C4 passed 0/35` · `C4-occ passed 4/19`
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
  or drop surface placement (it costs 11 of 46 plans and half of C4-occ), and whether to look at the
  OWLv2 misses (threshold 0.2, query "a photo of a {category}").
- **Next:** author decision, then a third pilot before the full run.

---

## 2026-10-16: SDXL pilot (run `task2_pilot`): insertion mostly fails, full run on hold

- **Setup:** RTX 4070 Laptop (8 GB), 32 GB RAM (`systeminfo`: 32,471 MB; now `hardware.ram_gb`),
  torch 2.6.0+cu124, diffusers 0.41.0, transformers 5.19.0. `pytest -q`: 48 passed. Work dir
  `data/inlayfake_pilot`, git 0261677 (dirty only by the `ram_gb` config line).
- **sample / plan:** 50 sources (33 train / 10 val / 7 test); 46 planned, 4 failed, 38 with an
  occlusion box. Jobs: 46 C2 + 46 C4 + 38 C4-occ = 130 SDXL edits; 46 C3 pastes.
- **Speed:** smoke test `generate --limit 3`: images 2–8 took 190 s (≈27 s each; the first
  included the 7 GB download). Round 1 of `insert`: 122 SDXL images in 43:57, **21.6 s per
  image** including model load (run_generate_2026-10-16T074120Z.json). OWLv2 + SAM verify: 130
  images in 42 s.
- **Round-1 verify** (run_verify_2026-10-16T074553Z.json, attempt 1 only):
  `C2: passed 46/46` · `C4: passed 4/46, needs_retry 42` · `C4-occ: passed 1/38, needs_retry 37` ·
  final ledger 189 records · audit sample 5. Insertion success on attempt 1 is therefore
  **5/84 (6%)**, far below the PRD target of ≥ 85%.
- **Retries stopped on purpose:** `insert` was stopped at the start of round 2 (no round-2 image
  was written, so the state is clean and resumable). With 6% per attempt, 3 attempts cannot reach
  the target, and retries would use up the attempt budget before the author decides what to change.
- **Bug hunt, no bug found:** the verifier agrees with a visual check on all 8 smoke-test edits
  (the 2 visible insertions passed, the 6 empty boxes failed), and a 20-tile contact sheet of
  failed C4/C4-occ edits shows plausible background in every box, no missed object. `compose.py`
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

## 2026-10-15: Week 2 started: verification stage, insert loop, viewer for C2–C4

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
  with its pasted mask, one passed record per C2/C4/C4-occ job.
- **Bug caught before the GPU run:** transformers 5.x removed `Owlv2Processor.post_process_object_
  detection` (now `post_process_grounded_object_detection`); the wrapper supports both. The SAM wrapper
  was run with random weights on CPU; real OWLv2/SAM weights cannot be downloaded in the sandbox.
- **Viewer:** lists C2–C4 per generator; inserted objects purple, `mixed_modified` amber; detectors
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
  - plan: 544 planned, **56 failed (9.3%)**, 363 of 544 with a C4-occ occlusion box (66.7%).
    Categories: chair 62, bottle 56, potted plant 49, vase 50, clock 57, bench 51, backpack 65,
    dog 44, cup 59, umbrella 51.
  - generate: 1,088 records = 544 × (C0 + C1), real only; 27 s.
- **Interpretation:** The splits match the 60/20/20 target. The failed plans are photos with no free
  region for a new object (crowded scenes). All later conditions use the same 544 sources, so the
  benchmark under-represents very cluttered scenes; this has been added to the paper's limitations.
  C4-occ will cover only about 2/3 of sources; acceptable, but report C4-occ counts separately.
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
  - Identical JPEG q=95 encoding for C1–C4 to remove format shortcuts.
- **Created:** AGENTS.md, CLAUDE.md, PRD.md, ARCHITECTURE.md, paper/PAPER.md, config and ledger
  schema.
- **Open:** laptop GPU model and VRAM; professor sign-off on the scope.
- **Next:** COCO download + `sample` and `plan` stages; generator wrapper for G1; 50-source pilot.
