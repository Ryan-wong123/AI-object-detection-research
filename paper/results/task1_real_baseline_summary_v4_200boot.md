# Task 1 summary: run `task1_real_baseline`

## Detector AP on real photos (C0 original, C1 resave)

| detector | cond | AP [95% CI] | AP50 | AR100 |
|---|---|---|---|---|
| faster_rcnn | C0 | 0.500 [0.460, 0.515] | 0.702 | 0.628 |
| faster_rcnn | C1 | 0.498 [0.460, 0.510] | 0.695 | 0.625 |
| yolo | C0 | 0.530 [0.484, 0.547] | 0.692 | 0.651 |
| yolo | C1 | 0.528 [0.487, 0.547] | 0.690 | 0.650 |
| rtdetr | C0 | 0.512 [0.479, 0.527] | 0.675 | 0.635 |
| rtdetr | C1 | 0.511 [0.476, 0.532] | 0.673 | 0.632 |
| grounding_dino | C0 | 0.515 [0.481, 0.532] | 0.664 | 0.685 |
| grounding_dino | C1 | 0.515 [0.476, 0.525] | 0.665 | 0.685 |

## Effect of resaving (C0 → C1) on per-object detection

| detector | objects | recall C0 | recall C1 | lost | gained |
|---|---|---|---|---|---|
| faster_rcnn | 3959 | 0.742 | 0.743 | 0.0058 | 0.0066 |
| yolo | 3959 | 0.519 | 0.517 | 0.0053 | 0.0040 |
| rtdetr | 3959 | 0.603 | 0.601 | 0.0066 | 0.0045 |
| grounding_dino | 3959 | 0.461 | 0.461 | 0.0076 | 0.0073 |

## Forensic image score on real photos (p99 − median of the map)

| method | cond | mean [95% CI] | median | p95 |
|---|---|---|---|---|
| ela | C0 | 5.048 [4.795, 5.323] | 4.526 | 9.741 |
| ela | C1 | 5.235 [4.986, 5.498] | 4.714 | 9.825 |
| srm_residual | C0 | 3.046 [2.944, 3.151] | 2.891 | 5.429 |
| srm_residual | C1 | 2.94 [2.837, 3.045] | 2.733 | 5.399 |
| noise_level | C0 | 10.38 [9.773, 11.03] | 8.992 | 22.12 |
| noise_level | C1 | 10.29 [9.683, 10.95] | 8.895 | 21.54 |
| fft_highfreq | C0 | 0.3654 [0.3552, 0.3756] | 0.3515 | 0.5809 |
| fft_highfreq | C1 | 0.3699 [0.359, 0.3816] | 0.3588 | 0.599 |
| jpeg_bag | C0 | 0.5319 [0.5112, 0.5521] | 0.4835 | 0.9927 |
| jpeg_bag | C1 | 0.5071 [0.4868, 0.5291] | 0.4652 | 0.9342 |

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
| jpeg_bag | C0 | 3882 | -0.003126 | 0.05346 | 0.08316 |
| jpeg_bag | C1 | 3882 | -0.002853 | 0.05202 | 0.07962 |

## Resave shift of the image score (C1 − C0)

| method | sources | mean diff [95% CI] | relative |
|---|---|---|---|
| ela | 544 | 0.1877 [0.1704, 0.2057] | +3.7% |
| srm_residual | 544 | -0.1063 [-0.1135, -0.09872] | -3.5% |
| noise_level | 544 | -0.08859 [-0.142, -0.03783] | -0.9% |
| fft_highfreq | 544 | 0.004457 [0.002865, 0.006374] | +1.2% |
| jpeg_bag | 544 | -0.02482 [-0.02749, -0.02232] | -4.7% |
