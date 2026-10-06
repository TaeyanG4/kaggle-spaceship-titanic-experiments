# H-D-01 — Earth error diagnosis (saved OOF only)

Created: 2026-10-05T22:15:53.288813+09:00. No model trained; segments are target-free raw features.

## Earth vs non-Earth errors

| Model | Earth errors | Earth Δ vs baseline | Non-Earth errors | Non-Earth Δ | Earth log loss | Earth bootstrap Δacc 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 1152 | +0 | 423 | +0 | 0.5044 | - |
| v1_name_fix | 1149 | -3 | 414 | -9 | 0.5043 | [-0.0052, +0.0062] |
| v2 | 1163 | +11 | 412 | -11 | 0.5097 | [-0.0090, +0.0045] |
| v2_logloss | 1193 | +41 | 410 | -13 | 0.5026 | [-0.0166, -0.0009] |
| v2_lightgbm | 1194 | +42 | 407 | -16 | 0.5073 | [-0.0181, -0.0009] |
| v2_xgboost | 1181 | +29 | 403 | -20 | 0.5087 | [-0.0150, +0.0030] |
| v2_blend | 1175 | +23 | 402 | -21 | 0.5033 | [-0.0129, +0.0030] |

## Earth error deltas by segment (rows ≥ 50)

### CryoSleep

| Segment | Rows | Actual rate | Baseline errors | v1_name_fix Δ | v2 Δ | v2_logloss Δ | v2_lightgbm Δ | v2_xgboost Δ | v2_blend Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 3106 | 0.321 | 667 | -7 | +11 | +14 | +20 | +13 | +13 |
| True | 1382 | 0.656 | 463 | +1 | +2 | +23 | +15 | +10 | +5 |
| missing | 114 | 0.412 | 22 | +3 | -2 | +4 | +7 | +6 | +5 |

### AgeBand

| Segment | Rows | Actual rate | Baseline errors | v1_name_fix Δ | v2 Δ | v2_logloss Δ | v2_lightgbm Δ | v2_xgboost Δ | v2_blend Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18-25 | 1540 | 0.390 | 369 | -1 | +10 | +14 | -1 | +8 | +1 |
| 26-40 | 1186 | 0.384 | 273 | -2 | -6 | +5 | +13 | +10 | +7 |
| 41+ | 724 | 0.376 | 194 | -5 | -4 | -4 | +6 | +0 | -3 |
| 0-12 | 571 | 0.594 | 197 | +2 | +6 | +15 | +14 | +5 | +13 |
| 13-17 | 493 | 0.503 | 96 | +0 | +2 | +3 | +0 | -5 | -5 |
| missing | 88 | 0.398 | 23 | +3 | +3 | +8 | +10 | +11 | +10 |

### SpendState

| Segment | Rows | Actual rate | Baseline errors | v1_name_fix Δ | v2 Δ | v2_logloss Δ | v2_lightgbm Δ | v2_xgboost Δ | v2_blend Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| positive | 2871 | 0.298 | 568 | -10 | +6 | +9 | +12 | +8 | +5 |
| zero_all_observed | 1540 | 0.631 | 518 | +3 | +6 | +30 | +27 | +20 | +15 |
| zero_with_missing | 191 | 0.644 | 66 | +4 | -1 | +2 | +3 | +1 | +3 |

### GroupSize

| Segment | Rows | Actual rate | Baseline errors | v1_name_fix Δ | v2 Δ | v2_logloss Δ | v2_lightgbm Δ | v2_xgboost Δ | v2_blend Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3070 | 0.396 | 738 | -8 | -6 | +13 | +13 | +3 | -2 |
| 2 | 580 | 0.448 | 146 | -2 | +6 | +3 | +4 | +6 | +6 |
| 4 | 520 | 0.487 | 154 | +6 | +9 | +20 | +23 | +17 | +16 |
| 3 | 432 | 0.516 | 114 | +1 | +2 | +5 | +2 | +3 | +3 |

### Destination

| Segment | Rows | Actual rate | Baseline errors | v1_name_fix Δ | v2 Δ | v2_logloss Δ | v2_lightgbm Δ | v2_xgboost Δ | v2_blend Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAPPIST-1e | 3101 | 0.389 | 775 | -1 | +6 | +31 | +19 | +9 | +4 |
| PSO J318.5-22 | 712 | 0.499 | 202 | -1 | +0 | +12 | +9 | +10 | +9 |
| 55 Cancri e | 690 | 0.504 | 160 | -2 | +3 | -4 | +11 | +10 | +8 |
| missing | 99 | 0.414 | 15 | +1 | +2 | +2 | +3 | +0 | +2 |

### CabinDeck

| Segment | Rows | Actual rate | Baseline errors | v1_name_fix Δ | v2 Δ | v2_logloss Δ | v2_lightgbm Δ | v2_xgboost Δ | v2_blend Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 2498 | 0.518 | 696 | +2 | -2 | +29 | +22 | +8 | +11 |
| F | 1614 | 0.292 | 331 | -1 | -1 | +8 | -1 | +2 | +1 |
| E | 395 | 0.372 | 98 | -2 | +9 | +2 | +16 | +14 | +8 |
| missing | 95 | 0.400 | 27 | -2 | +5 | +2 | +5 | +5 | +3 |

### CabinSide

| Segment | Rows | Actual rate | Baseline errors | v1_name_fix Δ | v2 Δ | v2_logloss Δ | v2_lightgbm Δ | v2_xgboost Δ | v2_blend Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 2270 | 0.369 | 591 | +3 | +9 | +22 | +29 | +17 | +19 |
| S | 2237 | 0.481 | 534 | -4 | -3 | +17 | +8 | +7 | +1 |
| missing | 95 | 0.400 | 27 | -2 | +5 | +2 | +5 | +5 | +3 |

### CryoSleep_SpendState

| Segment | Rows | Actual rate | Baseline errors | v1_name_fix Δ | v2 Δ | v2_logloss Δ | v2_lightgbm Δ | v2_xgboost Δ | v2_blend Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| False|positive | 2804 | 0.297 | 557 | -9 | +8 | +8 | +10 | +8 | +5 |
| True|zero_all_observed | 1226 | 0.655 | 409 | -1 | +4 | +23 | +13 | +10 | +3 |
| False|zero_all_observed | 272 | 0.537 | 101 | +0 | +1 | +4 | +9 | +4 | +7 |
| True|zero_with_missing | 156 | 0.667 | 54 | +2 | -2 | +0 | +2 | +0 | +2 |
| missing|positive | 67 | 0.328 | 11 | -1 | -2 | +1 | +2 | +0 | +0 |

## Flips vs baseline on Earth rows

| Model | Lost | Gained | Net | Lost near 0.5 (±0.1) | Pred. positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2_logloss | 186 | 145 | +41 | 165 | 0.445 |
| v2_lightgbm | 232 | 190 | +42 | 191 | 0.436 |
| v2_xgboost | 234 | 205 | +29 | 190 | 0.432 |
| v2_blend | 192 | 169 | +23 | 160 | 0.442 |
| v1_name_fix | 101 | 104 | -3 | 96 | 0.458 |
| v2 | 126 | 115 | +11 | 117 | 0.464 |

Earth actual positive rate 0.424; baseline predicted positive rate 0.457.

## Limits

Diagnostic on fitted OOF predictions from one seed-42 SGKF split. Segment deltas of a few rows are within noise; bootstrap CIs are conditional on fitted models. HomePlanet-missing rows are excluded from Earth. Not evidence for segment models or full-OOF threshold tuning.
