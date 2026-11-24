Macro-F1 / recall of the hard 'ball' class, mean ± std over seeds. REAL CWRU data (12 kHz drive end).

| model | temporal | cross_load | cross_severity |
|---|---|---|---|
| HGB + generic spectrum | 0.983 ± 0.000 (ball 1.00) | 0.958 ± 0.000 (ball 1.00) | 0.542 ± 0.000 (ball 0.42) |
| HGB + generic envelope (Hz bins) | 0.983 ± 0.000 (ball 1.00) | 0.625 ± 0.000 (ball 0.97) | 0.649 ± 0.000 (ball 1.00) |
| HGB + physics orders | 0.983 ± 0.000 (ball 1.00) | 0.996 ± 0.000 (ball 1.00) | 0.830 ± 0.000 (ball 0.75) |
| 1D-CNN on raw windows | 1.000 ± 0.000 (ball 1.00) | 0.982 ± 0.018 (ball 1.00) | 0.449 ± 0.023 (ball 1.00) |

- temporal: 513 train / 118 test windows, test classes {'normal': 10, 'outer_race': 36, 'inner_race': 36, 'ball': 36}
- cross_load: 370 train / 200 test windows, test classes {'normal': 20, 'outer_race': 60, 'inner_race': 60, 'ball': 60}
- cross_severity: 357 train / 46 test windows, test classes {'normal': 10, 'outer_race': 12, 'inner_race': 12, 'ball': 12}
