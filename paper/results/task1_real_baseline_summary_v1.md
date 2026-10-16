<!-- Source: runs/task1_real_baseline/summary.md, produced on the author's laptop (RTX 4070 Laptop),
     2026-10-08, 20 AP bootstrap resamples, audit decisions all "?" (ignore).
     SUPERSEDED PARTS: (1) AP 95% CIs used the plain percentile bootstrap, which is biased upward for
     COCO AP (intervals sit above the point estimate); fixed to the basic bootstrap in a later commit.
     (2) jpeg_bag scores used an unstabilised ratio that exploded on flat regions; fixed (log ratio
     with +1) in a later commit. All other numbers are unaffected. -->
# Task 1 summary: run `task1_real_baseline` (v1)

## Detector AP on real photos (C0 original, C1 resave)

| detector | cond | AP [95% CI, superseded] | AP50 | AR100 |
|---|---|---|---|---|
| faster_rcnn | C0 | 0.494 [0.479, 0.527] | 0.697 | 0.622 |
| faster_rcnn | C1 | 0.493 [0.481, 0.530] | 0.691 | 0.619 |
| yolo | C0 | 0.526 [0.510, 0.565] | 0.687 | 0.647 |
| yolo | C1 | 0.524 [0.515, 0.561] | 0.686 | 0.646 |
| rtdetr | C0 | 0.508 [0.486, 0.545] | 0.671 | 0.633 |
| rtdetr | C1 | 0.507 [0.488, 0.548] | 0.669 | 0.630 |
| grounding_dino | C0 | 0.511 [0.496, 0.550] | 0.660 | 0.682 |
| grounding_dino | C1 | 0.510 [0.500, 0.547] | 0.660 | 0.683 |

## Effect of resaving (C0 → C1) on per-object detection (score ≥ 0.5, IoU ≥ 0.5)

| detector | objects | recall C0 | recall C1 | lost | gained |
|---|---|---|---|---|---|
| faster_rcnn | 3882 | 0.737 | 0.738 | 0.0059 | 0.0067 |
| yolo | 3882 | 0.514 | 0.513 | 0.0052 | 0.0041 |
| rtdetr | 3882 | 0.597 | 0.595 | 0.0064 | 0.0046 |
| grounding_dino | 3882 | 0.457 | 0.457 | 0.0077 | 0.0075 |

## Forensic image score on real photos (p99 − median of the map)

| method | cond | mean [95% CI] | median | p95 |
|---|---|---|---|---|
| ela | C0 | 5.048 [4.8, 5.311] | 4.526 | 9.741 |
| ela | C1 | 5.235 [4.988, 5.505] | 4.714 | 9.825 |
| srm_residual | C0 | 3.046 [2.947, 3.154] | 2.891 | 5.429 |
| srm_residual | C1 | 2.94 [2.833, 3.047] | 2.733 | 5.399 |
| noise_level | C0 | 10.38 [9.752, 11.01] | 8.992 | 22.12 |
| noise_level | C1 | 10.29 [9.671, 10.96] | 8.895 | 21.54 |
| fft_highfreq | C0 | 0.3654 [0.3551, 0.3754] | 0.3515 | 0.5809 |
| fft_highfreq | C1 | 0.3699 [0.3599, 0.3808] | 0.3588 | 0.599 |
| jpeg_bag (superseded) | C0 | 428.7 [40.61, 1137] | 0.6666 | 2.623 |
| jpeg_bag (superseded) | C1 | 488.3 [65.74, 1158] | 0.6461 | 2.47 |

## Per-object contrast on real objects (inside − context)

| method | cond | n | mean | std | p95 |
|---|---|---|---|---|---|
| ela | C0 | 3882 | 0.4951 | 1.273 | 2.725 |
| ela | C1 | 3882 | 0.5146 | 1.325 | 2.868 |
| srm_residual | C0 | 3882 | 0.298 | 0.827 | 1.792 |
| srm_residual | C1 | 3882 | 0.29 | 0.8109 | 1.75 |
| noise_level | C0 | 3882 | 0.2502 | 1.62 | 2.924 |
| noise_level | C1 | 3882 | 0.2538 | 1.62 | 2.893 |
| fft_highfreq | C0 | 3882 | -0.002014 | 0.04386 | 0.06466 |
| fft_highfreq | C1 | 3882 | -0.001863 | 0.04539 | 0.06737 |
| jpeg_bag (superseded) | C0 | 3882 | -1.49 | 61.83 | 0.0926 |
| jpeg_bag (superseded) | C1 | 3882 | -1.903 | 80.8 | 0.09098 |

## Resave shift of the image score (C1 − C0)

| method | sources | mean diff [95% CI] | relative |
|---|---|---|---|
| ela | 544 | 0.1877 [0.1712, 0.2054] | +3.7% |
| srm_residual | 544 | -0.1063 [-0.1136, -0.09916] | -3.5% |
| noise_level | 544 | -0.08859 [-0.144, -0.04081] | -0.9% |
| fft_highfreq | 544 | 0.004457 [0.002668, 0.006369] | +1.2% |
| jpeg_bag (superseded) | 544 | 59.69 [-74.1, 209.6] | +13.9% |
