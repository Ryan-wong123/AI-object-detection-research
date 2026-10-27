# Origin summary (H3, H5): run `task3_detection_natural`

## H3: forensic origin per object (test split; direction and thresholds fit on val)

AI-inserted objects vs real objects (photographed C1-C4, pasted C3). Base rate of
AUPRC = share of AI objects. TPR@5% = AI objects caught at the val 5%-FPR threshold;
false accusation = real objects called AI at the val 80%-TPR threshold. AUROC vs
pasted = AI objects against the real objects pasted at the same boxes (C3). C2 box
= share of sham-edit boxes (repainted background, no object) flagged at that threshold.

| method | mode | dir | AUROC [CI] | AUROC vs pasted [CI] | AUPRC (base) | TPR@5% | TPR@80% thr | false accusation [CI] | of pasted | C2 box flagged [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| ela | image | lower=AI | 0.561 [0.509, 0.604] | 0.518 [0.502, 0.541] | 0.044 (0.034) | 0.144 | 0.744 | 0.660 [0.532, 0.778] | 0.711 | 0.744 [0.656, 0.833] |
| ela | region_mean | lower=AI | 0.477 [0.402, 0.548] | 0.577 [0.508, 0.647] | 0.032 (0.034) | 0.011 | 0.678 | 0.700 [0.609, 0.784] | 0.584 | 0.978 [0.944, 1.000] |
| ela | region_p95 | higher=AI | 0.546 [0.467, 0.625] | 0.447 [0.369, 0.522] | 0.051 (0.034) | 0.178 | 0.822 | 0.787 [0.704, 0.860] | 0.989 | 0.411 [0.311, 0.522] |
| ela | contrast | higher=AI | 0.596 [0.536, 0.656] | 0.486 [0.418, 0.563] | 0.078 (0.034) | 0.211 | 0.922 | 0.902 [0.870, 0.932] | 0.888 | 0.467 [0.367, 0.567] |
| srm_residual | image | lower=AI | 0.475 [0.430, 0.524] | 0.504 [0.493, 0.514] | 0.038 (0.034) | 0.011 | 0.744 | 0.817 [0.728, 0.897] | 0.733 | 0.744 [0.656, 0.833] |
| srm_residual | region_mean | lower=AI | 0.530 [0.465, 0.596] | 0.562 [0.505, 0.631] | 0.036 (0.034) | 0.078 | 0.633 | 0.597 [0.498, 0.697] | 0.596 | 0.944 [0.889, 0.989] |
| srm_residual | region_p95 | lower=AI | 0.572 [0.502, 0.640] | 0.579 [0.516, 0.643] | 0.037 (0.034) | 0.067 | 0.678 | 0.513 [0.417, 0.607] | 0.573 | 0.944 [0.900, 0.989] |
| srm_residual | contrast | higher=AI | 0.603 [0.544, 0.661] | 0.534 [0.470, 0.602] | 0.055 (0.034) | 0.178 | 0.878 | 0.870 [0.842, 0.901] | 0.809 | 0.278 [0.200, 0.367] |
| noise_level | image | lower=AI | 0.542 [0.483, 0.595] | 0.508 [0.496, 0.522] | 0.039 (0.034) | 0.133 | 0.822 | 0.789 [0.668, 0.896] | 0.822 | 0.822 [0.744, 0.900] |
| noise_level | region_mean | lower=AI | 0.558 [0.493, 0.618] | 0.591 [0.535, 0.646] | 0.040 (0.034) | 0.033 | 0.778 | 0.719 [0.652, 0.786] | 0.708 | 0.800 [0.722, 0.878] |
| noise_level | region_p95 | lower=AI | 0.609 [0.547, 0.668] | 0.597 [0.549, 0.646] | 0.047 (0.034) | 0.044 | 0.756 | 0.620 [0.543, 0.694] | 0.663 | 0.744 [0.656, 0.833] |
| noise_level | contrast | lower=AI | 0.499 [0.434, 0.556] | 0.585 [0.511, 0.656] | 0.035 (0.034) | 0.067 | 0.800 | 0.750 [0.702, 0.796] | 0.607 | 0.744 [0.656, 0.844] |
| fft_highfreq | image | higher=AI | 0.495 [0.450, 0.551] | 0.505 [0.499, 0.513] | 0.039 (0.034) | 0.078 | 0.811 | 0.849 [0.749, 0.931] | 0.811 | 0.811 [0.722, 0.878] |
| fft_highfreq | region_mean | higher=AI | 0.531 [0.469, 0.592] | 0.474 [0.417, 0.530] | 0.060 (0.034) | 0.078 | 0.711 | 0.772 [0.717, 0.825] | 0.809 | 0.800 [0.711, 0.878] |
| fft_highfreq | region_p95 | lower=AI | 0.520 [0.457, 0.581] | 0.537 [0.479, 0.591] | 0.037 (0.034) | 0.100 | 0.767 | 0.754 [0.700, 0.803] | 0.719 | 0.733 [0.644, 0.822] |
| fft_highfreq | contrast | higher=AI | 0.581 [0.532, 0.630] | 0.524 [0.454, 0.586] | 0.042 (0.034) | 0.056 | 0.689 | 0.632 [0.590, 0.671] | 0.730 | 0.722 [0.633, 0.811] |
| jpeg_bag | image | lower=AI | 0.517 [0.470, 0.566] | 0.487 [0.468, 0.503] | 0.043 (0.034) | 0.111 | 0.867 | 0.891 [0.814, 0.953] | 0.900 | 0.889 [0.822, 0.944] |
| jpeg_bag | region_mean | higher=AI | 0.565 [0.501, 0.627] | 0.533 [0.468, 0.596] | 0.049 (0.034) | 0.111 | 0.711 | 0.677 [0.618, 0.739] | 0.708 | 0.811 [0.722, 0.889] |
| jpeg_bag | region_p95 | higher=AI | 0.566 [0.504, 0.628] | 0.545 [0.494, 0.598] | 0.060 (0.034) | 0.167 | 0.767 | 0.748 [0.695, 0.801] | 0.742 | 0.878 [0.811, 0.944] |
| jpeg_bag | contrast | higher=AI | 0.528 [0.465, 0.589] | 0.495 [0.425, 0.571] | 0.042 (0.034) | 0.111 | 0.778 | 0.742 [0.708, 0.774] | 0.775 | 0.811 [0.722, 0.889] |
| trufor | image | higher=AI | 0.555 [0.502, 0.605] | 0.494 [0.441, 0.543] | 0.043 (0.034) | 0.089 | 0.678 | 0.631 [0.527, 0.734] | 0.689 | 0.811 [0.722, 0.889] |
| trufor | region_mean | higher=AI | 0.907 [0.870, 0.937] | 0.532 [0.469, 0.594] | 0.420 (0.034) | 0.567 | 0.678 | 0.071 [0.054, 0.094] | 0.708 | 0.444 [0.344, 0.544] |
| trufor | region_p95 | higher=AI | 0.894 [0.860, 0.924] | 0.452 [0.385, 0.514] | 0.276 (0.034) | 0.578 | 0.689 | 0.095 [0.073, 0.123] | 0.753 | 0.489 [0.389, 0.589] |
| trufor | contrast | higher=AI | 0.851 [0.803, 0.894] | 0.308 [0.239, 0.374] | 0.155 (0.034) | 0.422 | 0.733 | 0.132 [0.103, 0.163] | 0.820 | 0.478 [0.367, 0.589] |
| trufor | image_det | higher=AI | 0.636 [0.596, 0.677] | 0.509 [0.459, 0.560] | 0.054 (0.034) | 0.056 | 0.800 | 0.673 [0.568, 0.778] | 0.800 | 0.722 [0.622, 0.811] |
| univfd | image | higher=AI | 0.598 [0.546, 0.645] | 0.480 [0.445, 0.513] | 0.048 (0.034) | 0.111 | 0.800 | 0.641 [0.526, 0.772] | 0.789 | 0.767 [0.678, 0.856] |
| univfd | crop | higher=AI | 0.675 [0.617, 0.731] | 0.334 [0.269, 0.406] | 0.053 (0.034) | 0.044 | 0.811 | 0.521 [0.461, 0.583] | 0.922 | 0.944 [0.900, 0.989] |
| univfd | contrast | higher=AI | 0.603 [0.539, 0.667] | 0.394 [0.335, 0.452] | 0.042 (0.034) | 0.056 | 0.789 | 0.586 [0.512, 0.651] | 0.878 | 0.889 [0.822, 0.956] |

## H5: VLM origin claims per object (Stage 2)

Share of objects the model calls AI / photographed / unknown, describes without an
origin statement (none) or does not mention (no claim box at IoU >= 0.5).

| model | setting | cond | objects | n | AI [CI] | photographed | unknown | none | not mentioned |
|---|---|---|---|---|---|---|---|---|---|
| florence2 | grounded | C1 | photographed | 567 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.660 | 0.340 |
| florence2 | grounded | C3 | photographed | 567 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.653 | 0.347 |
| florence2 | grounded | C3 | real_pasted | 90 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.589 | 0.411 |
| florence2 | grounded | C4 | ai_inserted | 90 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.689 | 0.311 |
| florence2 | grounded | C4 | photographed | 550 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.649 | 0.351 |
| florence2 | plain | C1 | photographed | 567 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.690 | 0.310 |
| florence2 | plain | C3 | photographed | 567 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.705 | 0.295 |
| florence2 | plain | C3 | real_pasted | 90 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.600 | 0.400 |
| florence2 | plain | C4 | ai_inserted | 90 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.667 | 0.333 |
| florence2 | plain | C4 | photographed | 550 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.693 | 0.307 |
| qwen25vl | grounded | C1 | photographed | 567 | 0.000 [0.000, 0.000] | 0.624 | 0.019 | 0.000 | 0.356 |
| qwen25vl | grounded | C3 | photographed | 567 | 0.000 [0.000, 0.000] | 0.601 | 0.032 | 0.000 | 0.367 |
| qwen25vl | grounded | C3 | real_pasted | 90 | 0.011 [0.000, 0.033] | 0.433 | 0.033 | 0.000 | 0.522 |
| qwen25vl | grounded | C4 | ai_inserted | 90 | 0.011 [0.000, 0.033] | 0.556 | 0.044 | 0.000 | 0.389 |
| qwen25vl | grounded | C4 | photographed | 550 | 0.000 [0.000, 0.000] | 0.600 | 0.036 | 0.000 | 0.364 |
| qwen25vl | plain | C1 | photographed | 567 | 0.000 [0.000, 0.000] | 0.134 | 0.152 | 0.000 | 0.714 |
| qwen25vl | plain | C3 | photographed | 567 | 0.000 [0.000, 0.000] | 0.111 | 0.143 | 0.000 | 0.746 |
| qwen25vl | plain | C3 | real_pasted | 90 | 0.000 [0.000, 0.000] | 0.067 | 0.111 | 0.000 | 0.822 |
| qwen25vl | plain | C4 | ai_inserted | 90 | 0.000 [0.000, 0.000] | 0.122 | 0.089 | 0.000 | 0.789 |
| qwen25vl | plain | C4 | photographed | 550 | 0.000 [0.000, 0.000] | 0.125 | 0.129 | 0.000 | 0.745 |

## H5: images with any AI claim

| model | setting | cond | images | any AI claim [CI] |
|---|---|---|---|---|
| florence2 | grounded | C1 | 90 | 0.000 [0.000, 0.000] |
| florence2 | grounded | C3 | 90 | 0.000 [0.000, 0.000] |
| florence2 | grounded | C4 | 90 | 0.000 [0.000, 0.000] |
| florence2 | plain | C1 | 90 | 0.000 [0.000, 0.000] |
| florence2 | plain | C3 | 90 | 0.000 [0.000, 0.000] |
| florence2 | plain | C4 | 90 | 0.000 [0.000, 0.000] |
| qwen25vl | grounded | C1 | 90 | 0.000 [0.000, 0.000] |
| qwen25vl | grounded | C3 | 90 | 0.011 [0.000, 0.033] |
| qwen25vl | grounded | C4 | 90 | 0.011 [0.000, 0.033] |
| qwen25vl | plain | C1 | 90 | 0.000 [0.000, 0.000] |
| qwen25vl | plain | C3 | 90 | 0.000 [0.000, 0.000] |
| qwen25vl | plain | C4 | 90 | 0.000 [0.000, 0.000] |
